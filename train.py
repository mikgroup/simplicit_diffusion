#!/usr/bin/env python3
"""Train implicit networks on peristalsis data."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusion.prior import DiffusionPrior
from src.dataset import PeristalsisVideoData, PeristalsisVideoTrainDataset, PhaseSequenceDataset
from src.models import MLP, PositionalEncodingConfig, SIREN
from src.utils import make_phase_sweep, sequence_to_gif, tensor_to_chw


@dataclass
class TrainConfig:
    """Hyper-parameter bundle for a training run."""

    data_path: Path
    device: str
    model_type: str
    hidden_dim: int
    num_layers: int
    pe_bands: int
    pe_include_input: bool
    lines_per_frame: int
    num_train_frames: int
    batch_size: int
    epochs: int
    lr: float
    eval_interval: int
    checkpoint_interval: int
    num_fixed_states: int
    num_sweep_steps: int
    log_dir: Path
    exp_name: str
    seed: int
    use_prior: bool
    prior_weight: float
    prior_checkpoint: Optional[Path]
    prior_apply_frequency: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train implicit network on peristalsis data.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_type", choices=["mlp", "siren"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pe_bands", type=int, default=10)
    parser.add_argument("--pe_include_input", action="store_true")
    parser.add_argument("--lines_per_frame", type=int, default=10)
    parser.add_argument("--num_train_frames", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--num_fixed_states", type=int, default=3)
    parser.add_argument("--num_sweep_steps", type=int, default=50)
    parser.add_argument("--log_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_prior", action="store_true")
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--prior_checkpoint", type=Path, default=None)
    parser.add_argument("--prior_apply_frequency", type=int, default=10)
    args = parser.parse_args()

    num_frames = args.num_train_frames if args.num_train_frames > 0 else -1
    return TrainConfig(
        data_path=args.data_path,
        device=args.device,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pe_bands=args.pe_bands,
        pe_include_input=args.pe_include_input,
        lines_per_frame=args.lines_per_frame,
        num_train_frames=num_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_fixed_states=args.num_fixed_states,
        num_sweep_steps=args.num_sweep_steps,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        use_prior=args.use_prior,
        prior_weight=args.prior_weight,
        prior_checkpoint=args.prior_checkpoint,
        prior_apply_frequency=args.prior_apply_frequency,
    )


class TrainingExperiment:
    """Handles training, evaluation, and artifact management."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.exp_dir = self._create_experiment_dir()
        self.writer = SummaryWriter(log_dir=self.exp_dir / "tb")

        self._set_seed()
        self._save_config()

        self.data = PeristalsisVideoData.load(config.data_path, device=self.device)
        self.train_dataset = PeristalsisVideoTrainDataset(
            self.data,
            lines_per_frame=config.lines_per_frame,
            max_frames=config.num_train_frames if config.num_train_frames > 0 else None,
            seed=config.seed,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self._collate_batch,
        )

        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.best_val = float("inf")

        self.prior: Optional[DiffusionPrior] = None
        if config.use_prior:
            if config.prior_checkpoint is None:
                raise ValueError("Set --prior_checkpoint when --use_prior is enabled.")
            self.prior = DiffusionPrior(config.prior_checkpoint, self.device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        for epoch in tqdm(range(1, self.config.epochs + 1), "Epochs", total=self.config.epochs):
            train_loss = self._train_one_epoch()
            self.writer.add_scalar("train/loss", train_loss, epoch)

            if epoch % self.config.eval_interval == 0 or epoch == self.config.epochs:
                val_loss = self._evaluate(epoch)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save_checkpoint(epoch, tag="best")

            if epoch % self.config.checkpoint_interval == 0 and epoch != self.config.epochs:
                self._save_checkpoint(epoch, tag=f"epoch{epoch}")

        self._save_checkpoint(self.config.epochs, tag="final")
        self._export_line_visualization()
        self.writer.close()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _create_experiment_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parts = [timestamp, self.config.model_type, f"lines{self.config.lines_per_frame}"]
        if self.config.exp_name:
            parts.append(self.config.exp_name)
        exp_dir = self.config.log_dir / "simplicit_diffusion" / "runs" / "_".join(parts)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_config(self) -> None:
        config_path = self.exp_dir / "config.json"
        with config_path.open("w") as handle:
            json.dump(asdict(self.config), handle, indent=2, default=str)

    def _build_model(self) -> torch.nn.Module:
        in_dims = 4
        if self.config.model_type == "mlp":
            pe_cfg = PositionalEncodingConfig(
                num_bands=self.config.pe_bands,
                include_input=self.config.pe_include_input,
            )
            return MLP(in_dims, hidden_dim=self.config.hidden_dim, num_layers=self.config.num_layers, pe_config=pe_cfg)
        return SIREN(in_dims, hidden_dim=self.config.hidden_dim, num_layers=self.config.num_layers)

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        self.model.train()
        loss_sum = 0.0
        steps = 0
        for step, batch in enumerate(tqdm(self.train_loader, desc="Train", leave=False)):
            inputs = batch["inputs"].reshape(-1, batch["inputs"].shape[-1])
            targets = batch["targets"].reshape(-1, 1)

            preds = self.model(inputs)
            loss = F.mse_loss(preds, targets)
            if self.prior is not None and step % self.config.prior_apply_frequency == 0:
                full_frames = self._render_full_frames(batch["frame_indices"])
                prior_delta = self.prior(full_frames)
                prior_loss = (prior_delta ** 2).mean()
                loss = loss + self.config.prior_weight * prior_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            steps += 1
        return loss_sum / max(steps, 1)

    def _evaluate(self, global_step: int) -> float:
        eval_dir = self.exp_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        original_ds = PhaseSequenceDataset(self.data, self.data.phases_train, frames=self.data.frames)
        pred_frames, target_frames = self._reconstruct_sequence(original_ds)
        mse_original = F.mse_loss(pred_frames, target_frames).item()
        self.writer.add_scalar("val/mse_original", mse_original, global_step)
        sequence_to_gif(pred_frames, eval_dir / "recon_original.gif", fps=1.0 / self.data.frame_interval)
        sequence_to_gif(target_frames, eval_dir / "target_original.gif", fps=1.0 / self.data.frame_interval)
        if pred_frames.shape[0] > 0:
            self._log_image("val/recon_original/frame0", pred_frames[0], global_step)
        if target_frames.shape[0] > 0:
            self._log_image("val/target_original/frame0", target_frames[0], global_step)

        if self.data.num_future_frames > 0:
            future_ds = PhaseSequenceDataset(self.data, self.data.phases_future, frames=self.data.frames_future)
            future_pred, future_target = self._reconstruct_sequence(future_ds)
            mse_future = F.mse_loss(future_pred, future_target).item()
            self.writer.add_scalar("val/mse_future", mse_future, global_step)
            sequence_to_gif(future_pred, eval_dir / "recon_future.gif", fps=1.0 / self.data.frame_interval)
            if future_pred.shape[0] > 0:
                self._log_image("val/recon_future/frame0", future_pred[0], global_step)
            if future_target.shape[0] > 0:
                self._log_image("val/target_future/frame0", future_target[0], global_step)

        self._evaluate_sweeps(eval_dir)
        return mse_original

    def _reconstruct_sequence(self, dataset: PhaseSequenceDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        predictions: List[torch.Tensor] = []
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Reconstruct", leave=False):
                item = dataset[idx]
                inputs = item["inputs"].reshape(-1, item["inputs"].shape[-1])
                outputs = self.model(inputs)
                predictions.append(outputs.reshape(self.data.height, self.data.width, 1))

        pred_stack = torch.stack(predictions, dim=0)
        targets = dataset.frames[: pred_stack.shape[0]] if dataset.frames is not None else torch.empty(0)
        return pred_stack, targets

    def _render_full_frames(self, frame_indices: torch.Tensor) -> torch.Tensor:
        unique_frames = torch.unique(frame_indices).tolist()
        rendered = []
        for frame_idx in unique_frames:
            phase = self.data.phases_train[int(frame_idx)]
            phase_unit = self.data.phases_to_unit(phase.unsqueeze(0)).squeeze(0)
            phase_grid = phase_unit.view(1, 1, -1).expand(self.data.height, self.data.width, -1)
            inputs = torch.cat([self.data.coords_unit, phase_grid], dim=-1)
            outputs = self.model(inputs.reshape(-1, inputs.shape[-1]))
            frame = outputs.reshape(self.data.height, self.data.width, 1)
            rendered.append(frame.permute(2, 0, 1))
        return torch.stack(rendered, dim=0)
    def _evaluate_sweeps(self, eval_dir: Path) -> None:
        if self.config.num_fixed_states <= 0:
            return

        fixed_values = torch.linspace(0.0, 2.0 * math.pi, self.config.num_fixed_states + 1, device=self.device)[:-1]
        for idx, phi in enumerate(fixed_values):
            sweep_phase2 = make_phase_sweep(
                self.data.phases_train,
                fixed_index=0,
                fixed_value=float(phi),
                num_steps=self.config.num_sweep_steps,
            )
            recon_phase2, _ = self._reconstruct_sequence(PhaseSequenceDataset(self.data, sweep_phase2))
            sequence_to_gif(recon_phase2, eval_dir / f"sweep_phase2_fix_{idx}.gif", fps=10.0)

            sweep_phase1 = make_phase_sweep(
                self.data.phases_train,
                fixed_index=1,
                fixed_value=float(phi),
                num_steps=self.config.num_sweep_steps,
            )
            recon_phase1, _ = self._reconstruct_sequence(PhaseSequenceDataset(self.data, sweep_phase1))
            sequence_to_gif(recon_phase1, eval_dir / f"sweep_phase1_fix_{idx}.gif", fps=10.0)

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        torch.save(ckpt, self.exp_dir / f"checkpoint_{tag}.pt")

    def _export_line_visualization(self) -> None:
        line_mask = self.train_dataset.sampled_line_mask()
        frames = self.data.frames[: self.train_dataset.num_frames]
        highlighted = frames.clone()
        richer_mask = line_mask.to(dtype=torch.bool)
        for frame_idx in range(highlighted.shape[0]):
            rows = torch.nonzero(richer_mask[frame_idx], as_tuple=False).flatten()
            for row in rows:
                highlighted[frame_idx, row] = torch.ones_like(highlighted[frame_idx, row])
        sequence_to_gif(highlighted, self.exp_dir / "train_lines.gif", fps=1.0 / self.data.frame_interval)

    def _log_image(self, tag: str, frame: torch.Tensor, step: int) -> None:
        img = tensor_to_chw(frame)
        if img is None:
            return
        self.writer.add_image(tag, img, step, dataformats="CHW")

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Dict[str, torch.Tensor]:
        inputs = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        frame_indices = torch.tensor([item[2] for item in batch], device=inputs.device, dtype=torch.long)
        row_indices = torch.tensor([item[3] for item in batch], device=inputs.device, dtype=torch.long)
        return {
            "inputs": inputs,
            "targets": targets,
            "frame_indices": frame_indices,
            "row_indices": row_indices,
        }


def main() -> None:
    config = parse_args()
    experiment = TrainingExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train an implicit network on peristalsis horizontal lines with diffusion prior."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _SRC_ROOT):
    print(str(_path))
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

sys.path.insert(0, "/home/shoumik/edm")



import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import dnnlib
except ImportError:
    try:
        from src import dnnlib
    except ImportError:
        dnnlib = None

import pickle

from src.dataset import PeristalsisVideoData, PeristalsisVideoTrainDataset, PhaseSequenceDataset
from src.models import MLP, PositionalEncodingConfig
from src.utils import make_phase_sweep, sequence_to_gif


DEFAULT_DATA_PATH = Path("/home/shoumik/simulation/data/datasets/baseline/peristalsis_data.npz")
DEFAULT_DIFFUSION_SNAPSHOT: Optional[Path] = "/home/shoumik/edm/training-runs/00018-peristalsis-256x256-allrows-fp32/network-final.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train implicit network on peristalsis horizontal-line samples."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/implicit_train"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lines_per_frame", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means use all frames.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=50)
    parser.add_argument("--export_lines_gif", action="store_true", default=False)
    parser.add_argument("--export_sweeps", action="store_true", default=False)
    parser.add_argument("--sweep_steps", type=int, default=50)
    parser.add_argument("--sweep_fixed_states", type=int, default=3)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pe_bands", type=int, default=10)
    parser.add_argument("--pe_scale", type=float, default=1.0)
    parser.add_argument("--pe_no_input", action="store_true", default=False)

    parser.add_argument("--diffusion_snapshot", type=Path, default=DEFAULT_DIFFUSION_SNAPSHOT)
    parser.add_argument("--diffusion_weight", type=float, default=0.1)
    parser.add_argument("--diffusion_num_steps", type=int, default=10)
    parser.add_argument("--diffusion_sigma_min", type=float, default=0.002)
    parser.add_argument("--diffusion_tmax", type=float, default=80.0)
    parser.add_argument("--diffusion_tmin", type=float, default=0.0)
    parser.add_argument("--diffusion_rho", type=float, default=7.0)
    parser.add_argument("--diffusion_mode", type=str, default="sds", choices=["sds", "multistep"],
                        help="Diffusion prior loss mode: 'sds' (single-step SDS) or 'multistep' (Euler trajectory).")
    parser.add_argument("--plot_lines_every", type=int, default=10,
                        help="Log GT vs predicted line plots to TensorBoard every N epochs (0 to disable).")
    parser.add_argument("--log_dir", type=Path, default=None, help="TensorBoard log dir (default: output_dir / 'tb')")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_diffusion_model(snapshot: Path, device: torch.device) -> torch.nn.Module:
    if dnnlib is None:
        raise RuntimeError("dnnlib is required to load EDM snapshots.")
    with dnnlib.util.open_url(str(snapshot), verbose=True) as f:
        diffusion_model = pickle.load(f)["net"].to(device)
    diffusion_model.eval()
    for param in diffusion_model.parameters():
        param.requires_grad_(False)
    return diffusion_model



def _pred_to_row(pred: torch.Tensor) -> torch.Tensor:
    """Reshape implicit-net output (B, W, 1) -> EDM row format (B, 1, 1, W)."""
    return pred.permute(0, 2, 1).unsqueeze(2)


def sds_prior_loss(
    line_pred: torch.Tensor,
    diffusion_model: torch.nn.Module,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_data: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Score Distillation Sampling loss (single EDM forward pass).

    Returns (pseudo_loss, l2_gap) where l2_gap is the interpretable
    L2 distance between the denoised output and the prediction.
    """
    row = _pred_to_row(line_pred)  # (B, 1, 1, W)
    B = row.shape[0]

    sigma = (torch.randn(B, 1, 1, 1, device=row.device) * P_std + P_mean).exp()
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

    noise = torch.randn_like(row)
    x_noisy = row + sigma * noise

    with torch.no_grad():
        denoised = diffusion_model(x_noisy, sigma.squeeze()).to(torch.float32)

    l2_gap = torch.nn.functional.mse_loss(denoised, row.detach()).item()

    grad = weight * (denoised - row.detach())
    loss = (grad * row).mean()
    return loss, l2_gap


def multistep_prior_loss(
    line_pred: torch.Tensor,
    diffusion_model: torch.nn.Module,
    *,
    num_steps: int,
    sigma_min: float,
    tmax: float,
    tmin: float,
    rho: float,
) -> torch.Tensor:
    """Multi-step Euler denoising trajectory loss."""
    if num_steps < 2:
        raise ValueError("diffusion_num_steps must be >= 2.")

    row = _pred_to_row(line_pred)  # (B, 1, 1, W)
    B = row.shape[0]

    noise = torch.randn_like(row)
    tsample = torch.rand(B, device=row.device) * tmax
    if tmin > 0.0:
        tsample = torch.maximum(tsample, torch.full_like(tsample, tmin))

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=row.device)
    t_steps = (
        tsample[:, None] ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - tsample[:, None] ** (1 / rho))
    ) ** rho
    t_steps = diffusion_model.round_sigma(t_steps)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:, :1])], dim=1)

    x_next = (row + noise * tsample[:, None, None, None]).detach()

    for i in range(num_steps):
        t_cur = t_steps[:, i]
        t_next = t_steps[:, i + 1]
        denoised = diffusion_model(x_next, t_cur).to(torch.float32)
        d_cur = (x_next - denoised) / t_cur[:, None, None, None]
        x_next = x_next + (t_next - t_cur)[:, None, None, None] * d_cur

    return torch.nn.functional.l1_loss(row, x_next)


def reconstruct_sequence(
    model: torch.nn.Module,
    dataset: PhaseSequenceDataset,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reconstruct frames and map from [-1, 1] back to [0, 1] for visualization."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            inputs = item["inputs"].reshape(-1, item["inputs"].shape[-1])
            outputs = model(inputs)
            outputs = (outputs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            predictions.append(outputs.reshape(height, width, 1))
    return torch.stack(predictions, dim=0)


def export_line_visualization(
    data: PeristalsisVideoData,
    dataset: PeristalsisVideoTrainDataset,
    output_dir: Path,
) -> None:
    line_mask = dataset.sampled_line_mask()
    frames = data.frames[: dataset.num_frames]
    highlighted = frames.clone()
    richer_mask = line_mask.to(dtype=torch.bool)
    for frame_idx in range(highlighted.shape[0]):
        rows = torch.nonzero(richer_mask[frame_idx], as_tuple=False).flatten()
        for row in rows:
            highlighted[frame_idx, row] = torch.ones_like(highlighted[frame_idx, row])
    sequence_to_gif(highlighted, output_dir / "train_lines.gif", fps=1.0 / data.frame_interval)


def export_phase_sweeps(
    model: torch.nn.Module,
    data: PeristalsisVideoData,
    output_dir: Path,
    *,
    sweep_steps: int,
    fixed_states: int,
) -> None:
    if fixed_states <= 0:
        return
    fixed_values = torch.linspace(0.0, 2.0 * np.pi, fixed_states + 1, device=data.device)[:-1]
    for idx, phi in enumerate(fixed_values):
        sweep_phase2 = make_phase_sweep(
            data.phases_train,
            fixed_index=0,
            fixed_value=float(phi),
            num_steps=sweep_steps,
        )
        recon_phase2 = reconstruct_sequence(model, PhaseSequenceDataset(data, sweep_phase2), data.height, data.width)
        sequence_to_gif(recon_phase2, output_dir / f"sweep_phase2_fix_{idx}.gif", fps=10.0)

        sweep_phase1 = make_phase_sweep(
            data.phases_train,
            fixed_index=1,
            fixed_value=float(phi),
            num_steps=sweep_steps,
        )
        recon_phase1 = reconstruct_sequence(model, PhaseSequenceDataset(data, sweep_phase1), data.height, data.width)
        sequence_to_gif(recon_phase1, output_dir / f"sweep_phase1_fix_{idx}.gif", fps=10.0)


def make_line_comparison_figure(
    gt_rows: np.ndarray,
    pred_rows: np.ndarray,
    titles: list[tuple[int, int]],
) -> plt.Figure:
    """Build a grid of line plots comparing ground-truth vs predicted rows.

    Args:
        gt_rows: (n, W) numpy array, values in [-1, 1].
        pred_rows: (n, W) numpy array, values in [-1, 1].
        titles: List of (frame_idx, row_idx) for each subplot title.

    Returns:
        Matplotlib figure (caller may pass to writer.add_figure).
    """
    n = len(titles)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), squeeze=False)
    W = gt_rows.shape[1]
    positions = np.arange(W)

    for i in range(n):
        ax = axes[i // cols, i % cols]
        ax.plot(positions, gt_rows[i], label="GT", linewidth=1.0)
        ax.plot(positions, pred_rows[i], label="Pred", linewidth=1.0, linestyle="--")
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"f{titles[i][0]} r{titles[i][1]}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle("Ground truth vs predicted rows (unconditioned on row index)", fontsize=14)
    fig.tight_layout()
    return fig


def log_line_comparison_figure(
    writer: SummaryWriter,
    model: torch.nn.Module,
    dataset: PeristalsisVideoTrainDataset,
    device: torch.device,
    epoch: int,
    *,
    num_samples: int = 25,
    seed: Optional[int] = None,
) -> None:
    """Sample random training rows, run the model, and log a GT vs pred line-plot grid to TensorBoard."""
    n = min(num_samples, len(dataset))
    if n == 0:
        return
    rng = np.random.default_rng(seed if seed is not None else epoch)
    indices = rng.choice(len(dataset), size=n, replace=False)

    gt_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    titles_list: list[tuple[int, int]] = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            inputs, targets, frame_idx, row_idx = dataset[int(idx)]
            # targets from dataset are [0, 1]; convert to [-1, 1] to match training
            gt = (targets.cpu().numpy() * 2.0 - 1.0).squeeze()
            inputs_b = inputs.unsqueeze(0).to(device)
            pred = model(inputs_b).squeeze(0).cpu().numpy().squeeze()
            pred = np.clip(pred, -1.0, 1.0)
            gt_list.append(gt)
            pred_list.append(pred)
            titles_list.append((frame_idx, row_idx))

    gt_rows = np.stack(gt_list)
    pred_rows = np.stack(pred_list)
    fig = make_line_comparison_figure(gt_rows, pred_rows, titles_list)
    writer.add_figure("lines/gt_vs_pred", fig, epoch, close=True)


def evaluate_val_mse(
    model: torch.nn.Module,
    data: PeristalsisVideoData,
    device: torch.device,
) -> float:
    """Compute MSE on held-out future frames (generalization metric).

    Uses data.frames_future and data.phases_future.
    Returns mean MSE over all future frames (in [-1, 1] space).
    """
    if data.num_future_frames == 0:
        return float("nan")

    model.eval()
    total_mse = 0.0
    phases_unit = data.phases_to_unit(data.phases_future)  # (T_future, 2)

    with torch.no_grad():
        for t in range(data.num_future_frames):
            phase_feat = phases_unit[t]  # (2,)
            phase_grid = phase_feat.view(1, 1, -1).expand(data.height, data.width, -1)
            inputs = torch.cat([data.coords_unit, phase_grid], dim=-1)  # (H, W, 4)
            inputs_flat = inputs.reshape(-1, 4).to(device)

            pred = model(inputs_flat)  # (H*W, 1)
            pred = pred.reshape(data.height, data.width, 1)

            gt = data.frames_future[t] * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            total_mse += torch.nn.functional.mse_loss(pred, gt).item()

    return total_mse / data.num_future_frames


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. Use --device cpu only if you intend to run on CPU."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir if args.log_dir is not None else args.output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    data = PeristalsisVideoData.load(args.data_path, device=device)
    max_frames = None if args.max_frames <= 0 else args.max_frames
    dataset = PeristalsisVideoTrainDataset(
        data,
        lines_per_frame=args.lines_per_frame,
        max_frames=max_frames,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    pe_config = None
    if args.pe_bands > 0:
        pe_config = PositionalEncodingConfig(
            num_bands=args.pe_bands,
            include_input=not args.pe_no_input,
            scale=args.pe_scale,
        )
    model = MLP(in_dims=4, hidden_dim=args.hidden_dim, num_layers=args.num_layers, pe_config=pe_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    diffusion_model = None
    if args.diffusion_snapshot is not None:
        diffusion_model = load_diffusion_model(args.diffusion_snapshot, device=device)
    elif args.diffusion_weight > 0.0:
        print("Warning: diffusion_weight > 0 but diffusion_snapshot is not set. Skipping diffusion prior.")

    mse_loss = torch.nn.MSELoss()

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch (batch_size={args.batch_size})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_data = 0.0
        epoch_diff = 0.0
        epoch_l2_gap = 0.0
        start_time = time.time()

        for inputs, targets, _, _ in loader:
            targets = targets * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            optimizer.zero_grad()
            preds = model(inputs)
            data_loss = mse_loss(preds, targets)
            loss = data_loss

            diff_loss = torch.tensor(0.0, device=device)
            batch_l2_gap = 0.0
            if diffusion_model is not None and args.diffusion_weight > 0.0:
                if args.diffusion_mode == "sds":
                    diff_loss, batch_l2_gap = sds_prior_loss(
                        preds,
                        diffusion_model,
                        sigma_min=args.diffusion_sigma_min,
                    )
                else:
                    diff_loss = multistep_prior_loss(
                        preds,
                        diffusion_model,
                        num_steps=args.diffusion_num_steps,
                        sigma_min=args.diffusion_sigma_min,
                        tmax=args.diffusion_tmax,
                        tmin=args.diffusion_tmin,
                        rho=args.diffusion_rho,
                    )
                loss = loss + args.diffusion_weight * diff_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_data += data_loss.item()
            epoch_diff += diff_loss.item()
            epoch_l2_gap += batch_l2_gap

        elapsed = time.time() - start_time
        num_batches = max(1, len(loader))
        writer.add_scalar("train/loss", epoch_loss / num_batches, epoch)
        writer.add_scalar("train/data_loss", epoch_data / num_batches, epoch)
        writer.add_scalar("train/diffusion_loss", epoch_diff / num_batches, epoch)
        writer.add_scalar("train/time_per_epoch_s", elapsed, epoch)
        if epoch_l2_gap > 0:
            writer.add_scalar("train/sds_l2_gap", epoch_l2_gap / num_batches, epoch)

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            val_mse = evaluate_val_mse(model, data, device)
            writer.add_scalar("val/mse", val_mse, epoch)
            print(
                f"Epoch {epoch:04d} | "
                f"loss={epoch_loss/num_batches:.6f} "
                f"data={epoch_data/num_batches:.6f} "
                f"diff={epoch_diff/num_batches:.6f} "
                f"val_mse={val_mse:.6f} "
                f"time={elapsed:.2f}s"
            )

        if args.ckpt_every > 0 and (epoch % args.ckpt_every == 0 or epoch == args.epochs):
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )

        if args.plot_lines_every > 0 and (epoch % args.plot_lines_every == 0 or epoch == 1 or epoch == args.epochs):
            log_line_comparison_figure(
                writer, model, dataset, device, epoch,
                num_samples=25, seed=args.seed,
            )

    if args.export_lines_gif:
        export_line_visualization(data, dataset, args.output_dir)
    if args.export_sweeps:
        export_phase_sweeps(
            model,
            data,
            args.output_dir,
            sweep_steps=args.sweep_steps,
            fixed_states=args.sweep_fixed_states,
        )

    writer.close()


if __name__ == "__main__":
    main()

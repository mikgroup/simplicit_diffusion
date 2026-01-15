#!/usr/bin/env python3
"""Score-based diffusion prior wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
except ModuleNotFoundError as exc:
    repo_root = Path(__file__).resolve().parents[1]
    improved_path = repo_root / "external" / "improved-diffusion"
    if improved_path.exists():
        sys.path.insert(0, str(improved_path))
        from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    else:
        raise exc


class DiffusionPrior(nn.Module):
    """Compute a denoising direction from a trained diffusion model."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        defaults = model_and_diffusion_defaults()
        defaults.update(
            {
                "image_size": 256,
                "num_channels": 128,
                "num_res_blocks": 2,
                "learn_sigma": True,
                "diffusion_steps": 1000,
                "noise_schedule": "linear",
            }
        )
        if overrides:
            defaults.update(overrides)

        model, diffusion = create_model_and_diffusion(**defaults)
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        self.diffusion = diffusion
        self.device = device

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 1:
            raise ValueError("Expected frames shape (B, 1, H, W).")

        frames = frames.to(device=self.device, dtype=torch.float32)
        frames_rgb = frames.repeat(1, 3, 1, 1)
        frames_norm = frames_rgb * 2.0 - 1.0

        t = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(frames_norm.shape[0],),
            device=self.device,
        )
        noise = torch.randn_like(frames_norm)
        x_t = self.diffusion.q_sample(frames_norm, t, noise=noise)
        out = self.diffusion.p_mean_variance(self.model, x_t, t)
        pred_xstart = out["pred_xstart"]
        pred_xstart = (pred_xstart + 1.0) * 0.5
        pred_gray = pred_xstart.mean(dim=1, keepdim=True)
        return pred_gray - frames

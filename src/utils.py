#!/usr/bin/env python3
"""Utility helpers for grids, sweeps, and GIFs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch

TAU = 2.0 * math.pi


def create_unit_coord_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Generate a normalized (x, y) grid."""
    xs = torch.linspace(0.0, 1.0, width, device=device)
    ys = torch.linspace(0.0, 1.0, height, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def make_phase_sweep(phases: torch.Tensor, fixed_index: int, fixed_value: float, num_steps: int) -> torch.Tensor:
    """Create a phase sweep with one fixed phase."""
    device = phases.device
    sweep = torch.linspace(0.0, TAU, num_steps, device=device)
    result = torch.zeros((num_steps, 2), device=device)
    result[:, fixed_index] = fixed_value
    result[:, 1 - fixed_index] = sweep
    return result


def sequence_to_gif(frames: torch.Tensor, path: Path, fps: float) -> None:
    """Write frames to a GIF."""
    if frames.numel() == 0:
        return
    frames_np = frames.clamp(0.0, 1.0).cpu().numpy()
    if frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = frames_np[..., 0]
    frames_uint8 = (frames_np * 255).astype(np.uint8)
    imageio.mimsave(path, frames_uint8, fps=fps)


def tensor_to_chw(frame: torch.Tensor) -> Optional[torch.Tensor]:
    """Convert to CHW for TensorBoard."""
    if frame is None or frame.numel() == 0:
        return None
    img = frame.detach().clamp(0.0, 1.0).cpu()
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        return img.unsqueeze(0)
    if img.ndim == 3 and img.shape[0] != 1:
        return img.permute(2, 0, 1)
    return img

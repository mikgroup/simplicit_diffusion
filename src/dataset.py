#!/usr/bin/env python3
"""Dataset utilities for peristalsis reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import TAU, create_unit_coord_grid


def _as_torch(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a NumPy array to a float32 tensor."""
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


@dataclass
class PeristalsisVideoData:
    """Holds tensors from the peristalsis dataset."""

    frames: torch.Tensor  # (T_train, H, W, 1)
    phases_train: torch.Tensor  # (T_train, 2)
    frames_future: torch.Tensor  # (T_future, H, W, 1)
    phases_future: torch.Tensor  # (T_future, 2)
    frame_interval: float
    coords_unit: torch.Tensor  # (H, W, 2), normalized coordinates

    @classmethod
    def load(cls, npz_path: Path | str, *, device: str | torch.device = "cpu") -> "PeristalsisVideoData":
        npz = np.load(npz_path)
        device = torch.device(device)

        frames = _as_torch(npz["frames"], device=device).unsqueeze(-1)
        frames_future = _as_torch(npz["frames_future"], device=device).unsqueeze(-1)
        phases_train = _as_torch(npz["phases_train"], device=device)
        phases_future = _as_torch(npz["phases_future"], device=device)
        frame_interval = float(npz["frame_interval"].item())

        height = frames.shape[1]
        width = frames.shape[2]
        coords_unit = create_unit_coord_grid(height, width, device=device)

        return cls(
            frames=frames,
            phases_train=phases_train,
            frames_future=frames_future,
            phases_future=phases_future,
            frame_interval=frame_interval,
            coords_unit=coords_unit,
        )

    @property
    def device(self) -> torch.device:
        return self.frames.device

    @property
    def height(self) -> int:
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        return self.frames.shape[2]

    @property
    def num_train_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def num_future_frames(self) -> int:
        return self.frames_future.shape[0]

    def phases_to_unit(self, phases: torch.Tensor) -> torch.Tensor:
        """Normalize phases to [0, 1]."""
        return phases / TAU


class PeristalsisVideoTrainDataset(Dataset):
    """Samples horizontal lines from training frames."""

    def __init__(
        self,
        data: PeristalsisVideoData,
        *,
        lines_per_frame: int,
        max_frames: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.lines_per_frame = max(0, min(lines_per_frame, data.height))
        self.num_frames = min(data.num_train_frames, max_frames or data.num_train_frames)
        self.seed = seed

        if self.lines_per_frame == 0 or self.num_frames == 0:
            self.line_indices = torch.empty((self.num_frames, 0), dtype=torch.long, device=data.device)
            self.line_mask = torch.zeros((self.num_frames, data.height), dtype=torch.bool, device=data.device)
            self._index_map = torch.empty((0, 2), dtype=torch.long, device=data.device)
            return

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        line_indices = torch.empty((self.num_frames, self.lines_per_frame), dtype=torch.long)
        for frame_idx in range(self.num_frames):
            if self.lines_per_frame >= data.height:
                indices = torch.arange(data.height, dtype=torch.long)
            else:
                indices = torch.randperm(data.height, generator=generator)[: self.lines_per_frame]
            line_indices[frame_idx] = indices.sort()[0]

        self.line_indices = line_indices.to(device=data.device)
        self.line_mask = torch.zeros((self.num_frames, data.height), dtype=torch.bool, device=data.device)
        frame_idx_grid = torch.arange(self.num_frames, device=data.device).unsqueeze(1).expand_as(self.line_indices)
        self.line_mask[frame_idx_grid, self.line_indices] = True

        frame_indices = torch.arange(self.num_frames, dtype=torch.long, device=data.device).unsqueeze(1)
        frame_indices = frame_indices.expand(-1, self.lines_per_frame)
        self._index_map = torch.stack([frame_indices, self.line_indices], dim=-1).reshape(-1, 2)

        self._phases_unit = data.phases_to_unit(data.phases_train[: self.num_frames])

    def __len__(self) -> int:
        return int(self._index_map.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        frame_idx, row_idx = self._index_map[idx]
        frame_idx_int = int(frame_idx.item())
        row_idx_int = int(row_idx.item())

        coords_xy = self.data.coords_unit[row_idx_int]
        phase_features = self._phases_unit[frame_idx_int].unsqueeze(0).expand_as(coords_xy)
        inputs = torch.cat([coords_xy, phase_features], dim=-1)

        targets = self.data.frames[frame_idx_int, row_idx_int]
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        return inputs, targets, frame_idx_int, row_idx_int

    def sampled_line_mask(self) -> torch.Tensor:
        """Mask of sampled rows per frame."""
        return self.line_mask

    def frame_line_indices(self, frame_idx: int) -> torch.Tensor:
        """Return sampled row indices for a frame."""
        return self.line_indices[frame_idx]


class PhaseSequenceDataset(Dataset):
    """Full-frame coordinates and phases for evaluation."""

    def __init__(
        self,
        data: PeristalsisVideoData,
        phases: torch.Tensor,
        *,
        frames: Optional[torch.Tensor] = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.data = data
        self.phases = phases.to(device=data.device, dtype=torch.float32)
        self.phases_unit = data.phases_to_unit(self.phases)
        self.frames = frames.to(device=data.device, dtype=torch.float32) if frames is not None else None
        self.name = name

        if self.frames is not None and self.frames.ndim == 3:
            self.frames = self.frames.unsqueeze(-1)

    def __len__(self) -> int:
        return self.phases.shape[0]

    def __getitem__(self, idx: int) -> dict:
        phase_feat = self.phases_unit[idx]
        phase_grid = phase_feat.view(1, 1, -1).expand(self.data.height, self.data.width, -1)
        inputs = torch.cat([self.data.coords_unit, phase_grid], dim=-1)

        target = None
        if self.frames is not None and idx < self.frames.shape[0]:
            target = self.frames[idx]

        return {
            "inputs": inputs,
            "phase_raw": self.phases[idx],
            "phase_unit": phase_feat,
            "target": target,
            "index": idx,
            "name": self.name,
        }

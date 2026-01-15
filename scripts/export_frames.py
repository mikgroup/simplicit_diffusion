#!/usr/bin/env python3
"""Export peristalsis frames to PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export peristalsis frames to PNG files.")
    parser.add_argument(
        "--npz_path",
        type=Path,
        default=Path("/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/images/train"))
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of frames to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz_path)
    frames = npz["frames"]
    if args.limit > 0:
        frames = frames[: args.limit]

    for idx, frame in enumerate(frames):
        img = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(args.output_dir / f"frame_{idx:04d}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wrapper for improved-diffusion training."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved-diffusion on peristalsis frames.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/images/train"))
    parser.add_argument(
        "--repo_dir",
        type=Path,
        default=Path("external/improved-diffusion"),
        help="Path to improved-diffusion repo.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--log_dir", type=Path, default=Path("experiments/diffusion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {args.data_dir}")

    script_path = args.repo_dir / "scripts" / "image_train.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing improved-diffusion script: {script_path}")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_LOGDIR"] = str(args.log_dir)

    cmd = [
        "python",
        str(script_path),
        "--data_dir",
        str(args.data_dir),
        "--image_size",
        str(args.image_size),
        "--num_channels",
        str(args.num_channels),
        "--num_res_blocks",
        str(args.num_res_blocks),
        "--learn_sigma",
        "True",
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--noise_schedule",
        args.noise_schedule,
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--microbatch",
        str(args.microbatch),
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()

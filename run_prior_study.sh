#!/usr/bin/env bash
# Prior Benefit Experiments (CUDA)
# Run from repo root: ./run_prior_study.sh
# Compare runs: tensorboard --logdir outputs/prior_study --port 6006

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
COMMON="--max_frames 100 --batch_size 64 --epochs 100 --plot_lines_every 20 --log_every 10 --device cuda"

echo "=== Exp A: Baseline (no prior) ==="
"$PYTHON" src/train_implicit.py $COMMON \
  --diffusion_weight 0 \
  --output_dir outputs/prior_study/expA_no_prior

echo "=== Exp B: Multistep prior, weight 0.01 ==="
"$PYTHON" src/train_implicit.py $COMMON \
  --diffusion_weight 0.01 --diffusion_mode multistep --diffusion_tmax 80 \
  --output_dir outputs/prior_study/expB_multistep_0.01

echo "=== Exp C: Multistep prior, weight 0.03 ==="
"$PYTHON" src/train_implicit.py $COMMON \
  --diffusion_weight 0.03 --diffusion_mode multistep --diffusion_tmax 80 \
  --output_dir outputs/prior_study/expC_multistep_0.03

echo "=== Exp D: Multistep prior, weight 0.05 ==="
"$PYTHON" src/train_implicit.py $COMMON \
  --diffusion_weight 0.05 --diffusion_mode multistep --diffusion_tmax 80 \
  --output_dir outputs/prior_study/expD_multistep_0.05

echo "Done. TensorBoard: tensorboard --logdir outputs/prior_study --port 6006"

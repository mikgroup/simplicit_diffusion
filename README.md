# simplicit_diffusion

Implicit neural representations for peristalsis data with a clean scaffold for diffusion priors.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train with peristalsis data:
   ```bash
   python train.py --data_path "/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz"
   ```

3. Monitor training:
   ```bash
   tensorboard --logdir experiments/simplicit_diffusion/runs
   ```

## Diffusion prior workflow

1. Export peristalsis frames:
   ```bash
   python scripts/export_frames.py \
       --npz_path "/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz" \
       --output_dir data/images/train
   ```

2. Install improved-diffusion (once):
   ```bash
   pip install -e external/improved-diffusion
   ```

3. Train diffusion model:
   ```bash
   python scripts/train_diffusion.py \
       --data_dir data/images/train \
       --image_size 256
   ```

4. Use diffusion prior during training:
   ```bash
   python train.py \
       --data_path "/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz" \
       --use_prior \
       --prior_checkpoint /path/to/model.pt \
       --prior_weight 0.01 \
       --prior_apply_frequency 10
   ```

## Documentation

See the `docs/` directory for architecture, data format, and training details.

# Training

## Default command

```bash
python train.py --data_path "/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz"
```

## Key options

- `--model_type {mlp,siren}`
- `--lines_per_frame N`
- `--epochs N`
- `--hidden_dim N`
- `--lr FLOAT`
- `--batch_size N`

## Outputs

Artifacts are stored under:

```
experiments/simplicit_diffusion/runs/
```

Each run includes TensorBoard logs, checkpoints, and evaluation GIFs.

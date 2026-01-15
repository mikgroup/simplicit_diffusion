# Migration Notes

## Key renames

- `CircleVideoData` -> `PeristalsisVideoData`
- `CircleVideoTrainDataset` -> `PeristalsisVideoTrainDataset`

## Data source

The default data path now points to:

```
/home/shoumik/simulation/data/datasets/realistic/peristalsis_data.npz
```

## Structure changes

- Code moved under `src/`
- Diffusion scaffolding in `diffusion/`
- All documentation under `docs/`

# Data Format

Peristalsis data is sourced from:

```
/home/shoumik/simulation/data/datasets/
```

Each dataset is an NPZ with the following keys:

- `frames`: `(T_train, H, W)` float32
- `phases_train`: `(T_train, 2)` float32, values in `[0, 2π]`
- `frames_future`: `(T_future, H, W)` float32
- `phases_future`: `(T_future, 2)` float32
- `frame_interval`: scalar float

The training pipeline expects these keys and will normalize phases to `[0, 1]` internally.

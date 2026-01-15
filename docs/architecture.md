# Architecture

## Core concept

The model learns an implicit function:

```
f(x, y, phase1, phase2) -> intensity
```

- Inputs are normalized coordinates and phase values.
- Output is a grayscale intensity in `[0, 1]`.
- Phases are treated as state variables rather than time directly.

## Models

- **MLP** with positional encoding (sin/cos bands).
- **SIREN** with sine activations for implicit frequency modeling.

## Training data

Training samples are sparse horizontal line observations per frame, drawn deterministically using a seed for reproducibility.

## Outputs

Training artifacts include:
- Checkpoints (`checkpoint_*.pt`)
- Reconstruction GIFs
- Phase sweep GIFs
- TensorBoard logs

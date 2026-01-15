# Diffusion Prior (Planned)

The first integration target is score-based regularization in coordinate space:

```
loss = reconstruction_loss + lambda * ||score(coords)||^2
```

This will be implemented in `diffusion/prior.py` and wired into `train.py` once the baseline refactor is complete.

# tiny-worlds

Small, physical world-modeling datasets.

## friction-mnist

Two MNIST digits move with per-frame damping and inelastic wall bounces.
Sequences stop when both digits are at rest. The dataset includes positions,
velocities, friction, restitution, stop time, and final positions.

```python
from tiny_worlds.friction_mnist import generate_dataset, visualize_sample

data = generate_dataset(seed=0, num_samples=2_000)
visualize_sample(data, 0)
```

MNIST IDX files are read from `~/data/mnist` by default. Install the optional
visualization/dataset dependencies with:

```bash
uv add --editable "../tiny-worlds[friction-mnist]"
```

# tiny-worlds

Small, physical world-modeling datasets.

## friction-mnist

Two MNIST digits move with per-frame damping and inelastic wall bounces.
Sequences stop when both digits are at rest. The dataset includes positions,
velocities, friction, restitution, bounce count, stop time, and final
positions. Defaults use `T=200`, giving multiple bounces per sequence.

```python
from pathlib import Path
from tiny_worlds.friction_mnist import download_mnist, generate_dataset, visualize_sample

mnist_dir = download_mnist(Path("data") / "mnist")
data = generate_dataset(seed=0, num_samples=2_000, mnist_dir=mnist_dir)
visualize_sample(data, 0)
```

MNIST is not vendored. The package takes an explicit `mnist_dir`, or you can
use `download_mnist` to fetch the IDX files into a directory of your choice.
Install the optional visualization/dataset dependencies with:

```bash
uv add --editable "../tiny-worlds[friction-mnist]"
```

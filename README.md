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

## TwinT

Two T-shaped targets, one pink and one green, one T block, and a circular agent. The
agent pushes the block onto the commanded target; success is 95% coverage. Target
positions and orientations, block pose, and agent pose are sampled per episode and
never overlap.

```bash
uv add --editable "../tiny-worlds[twin-t]"
```

```python
from pathlib import Path

from tiny_worlds.twin_t import PINK, TwinTEnv, load_dataset, pack_trials, unpack

env = TwinTEnv()
obs, info = env.reset(seed=0, command=PINK)
print(obs.pixels.shape, obs.agent_pos, info.coverage, info.target_poses)
obs, reward, terminated, truncated, info = env.step([256.0, 300.0])
frame = env.render()  # 512x512x3 uint8

pack_trials(Path("data"), Path("dist/twint.zarr.zip"))
with load_dataset(Path("dist/twint.zarr.zip")) as dataset:
    print(dataset.episode_count, dataset.frame_count, dataset.split[:])
unpack(Path("dist/twint.zarr.zip"), Path("unpacked"))
```

`T=10` Hz, a 96x96 RGB observation plus the agent position, and a 512x512 workspace.
The policy observation contains only pixels and the agent position; block pose, target
poses, coverage, and success are evaluation-only. The full contract, including the raw
`.npz` trial format and the packed zarr v2 schema, is in [docs/twin-t.md](docs/twin-t.md).

The `twint-data-collector` project is the browser teleop front end that writes the raw
trials and runs `uv run pack`.

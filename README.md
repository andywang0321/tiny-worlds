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

pack_trials(Path("data"), Path("dist/twint.npz"))
with load_dataset(Path("dist/twint.npz")) as dataset:
    print(dataset.episode_count, dataset.frame_count, dataset.split[:])
    episode = dataset.episode(0)
    print(episode.state.shape, episode.action.shape, episode.frames(96).shape)
unpack(Path("dist/twint.npz"), Path("unpacked"))
```

TwinT runs at 10 Hz with a 512x512 workspace. The live environment still renders a 96x96
RGB observation plus the agent position, but the dataset is state-only: each raw trial
stores the agent/block `state`, the per-episode `target` and `target_poses`, and
metadata. Actions are derived as the realized next agent position, and frames are
rendered on demand at any size. The packed distribution is one compressed
`dist/twint.npz` (about 3 MB for 1,000 trials) with a 90/10 train/validation split
stratified by target color. The full contract, including the raw
`.npz` trial format, is in [docs/twin-t.md](docs/twin-t.md).

The v2 format also makes exact task-symmetry augmentation cheap: 16 label-preserving
transforms (8 dihedral elements x 2 target-color swaps) act on `state` and
`target_poses`, frames are re-rendered, and actions stay derived. `augment_trial(trial,
symmetry)` returns a valid `Trial`; `symmetries()` enumerates all 16. These are
training-only relabelings, not new data, and must never be used at evaluation.

```python
from tiny_worlds.twin_t import augment_trial, load_trial, symmetries

trial = load_trial("data/pink/p0001.npz")
augmented = [augment_trial(trial, symmetry) for symmetry in symmetries()]
```

`scripts/augment_trials.py --data data --out data-aug --symmetries all` writes the
augmented trial files. See the
[Symmetry augmentation](docs/twin-t.md#symmetry-augmentation) section for the transform
math and caveats.

The `twint-data-collector` project is the browser teleop front end that writes the raw
trials and runs `uv run pack`.

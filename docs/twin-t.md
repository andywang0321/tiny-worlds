# TwinT

TwinT is PushT with two T-shaped targets, one pink and one green, and one T block. The
agent is a circle; it pushes the block onto the commanded target. Target poses, block
pose, and agent pose are sampled per episode and never overlap.

The package is optional:

```bash
uv add --editable "../tiny-worlds[twin-t]"
```

## Task

| quantity | value |
|---|---|
| workspace | 512 x 512, walls at 5..506 |
| agent | circle, radius 15 |
| block | T, 120 x 120, mass 1, friction 1 |
| targets | two T's, colors fixed, positions random, continuous random orientations |
| control | 10 Hz, 10 physics substeps of 0.01 s |
| action | agent target position in `[0, 512]^2` |
| success | block covers the commanded target at least 95% |
| observation | live env only: 96 x 96 RGB plus the agent's 2D position |

Target poses are sampled with geometric centers in `(90, 90, 422, 422)`, a minimum
center separation of 180, and a continuous orientation in `[-pi, pi]`, the same range as
the block. The block's geometric center starts in `(110, 110, 390, 390)` with a
continuous angle; the agent starts in `(50, 50, 450, 450)`. The stored poses
(`target_poses`, `block_pose`) are T body poses in the PushT convention, so moving from
a sampled center to a pose subtracts the rotated local centroid `(0, 40.7)`. Sampling
rejects any layout where a target leaves the walls, the two targets overlap, the block
or agent overlaps a target, the agent overlaps the block, or the block already covers a
target. The block bounds keep the block inside the walls at any angle; target samples
near a corner whose orientation would leave the walls are rejected, so the accepted
orientation range narrows there. Both T's reach at most 99.4 from their geometric
center.

`PINK = 0`, `GREEN = 1`. The colors are `#FF4FA3` and `#22C55E`. The prompt names the
color; the policy sees only the image and its own position.

## Environment

```python
from tiny_worlds.twin_t import TwinTConfig
from tiny_worlds.twin_t.env import TwinTEnv

env = TwinTEnv(TwinTConfig(terminate_on="task"))
obs, info = env.reset(seed=0, command=0)  # 0 pink, 1 green, None null
obs, reward, terminated, truncated, info = env.step(action)
```

- `obs.pixels` is `(96, 96, 3)` uint8 and `obs.agent_pos` is `(2,)` float32. The
  observation size is a live-environment render choice, not part of the dataset format.
- `info` is a `StepInfo`: block pose, eight block keypoints, target poses, coverage for
  both targets, `command_success`, `task_success`, `achieved_target`, contacts, step.
- `terminate_on="task"` ends on either target (evaluation default).
  `terminate_on="command"` ends only on the commanded target (collector default); a
  block on the wrong target can be pushed off.
- With `command=None`, success and termination use either target.
- `set_command` changes the command mid-episode for the step-50 switch.
- `pause()` zeroes the agent velocity; the collector calls it when a drag is released so
  a paused trial resumes from rest.
- `state()` returns `TwinTState`; `state().as_vector()` is
  `[agent_x, agent_y, block_x, block_y, block_angle]`, the `reset(state=...)` format.
  `reset(state=..., target_poses=...)` reproduces a held-out layout; `target_poses` is
  `(2, 3)` for `[pink, green]` and is sampled when omitted.
- `target_poses` is evaluation-only and never enters the observation.

The agent is speed-limited by `max_agent_speed` (default 250 px/s, 25 px per control
step), applied as a velocity clamp inside the environment so collection and evaluation
have the same dynamics. PushT itself has no explicit cap; this is a TwinT addition for
data quality.

## Raw trial format (v2, state-only)

One successful trial is one uncompressed `.npz` under `data/pink/` or `data/green/`.
The file is a few KB and holds only the simulator state, the per-episode targets, and
metadata. Every array has first dimension `T + 1`, where `T` is the number of executed
control steps; `T + 1` is the number of stored frames including the initial one.

| key | shape | dtype | meaning |
|---|---|---|---|
| `state` | `(T+1, 5)` | float32 | `[agent_x, agent_y, block_x, block_y, block_angle]` |
| `target` | `()` | int8 | commanded color, 0 pink or 1 green |
| `target_poses` | `(2, 3)` | float32 | `[pink, green]` T body poses |
| `meta` | `()` | unicode | JSON: `schema_version`, `seed`, `created_at`, `active_seconds` |

There are no stored images, actions, coverage, success, keypoints, or contact counts.
The collector writes only successful trials; a discarded trial is not written.

**Actions are derived, never stored.** For an episode with rows `t = 0..T`:

```python
action[:-1] = state[1:, :2]
action[-1] = state[-1, :2]
```

The agent is a kinematic body with a speed cap, so `state[t+1, :2]` is exactly the
realized pointer target and is the correct BC target. The final row copies the terminal
agent position so `action` has the same `T + 1` rows as `state`.

Write to a temporary file in the same directory, flush, fsync, then `os.replace` into
place. A crash cannot leave a half-written trial with a final name.

## Packed dataset (v2)

`pack_trials(data_dir, out_path)` writes one compressed `dist/twint.npz`. The whole
dataset is about 3 MB for 1,000 trials, so the plain `.npz` needs no lazy chunking layer
and loads in one eager pass. It is written with `np.savez_compressed` to a temporary
path, fsynced, then `os.replace`d.

```text
state                 (N, 5) float32
episode_ends          (E,) int64      cumulative row ends
episode_target        (E,) int8
episode_target_poses  (E, 2, 3) float32
episode_ids           (E,) S32
episode_created       (E,) float64    unix seconds
episode_seed          (E,) int64
split                 (E,) int8       0 train, 1 validation
meta                  () unicode      JSON root metadata
```

`N == episode_ends[-1]`; each episode's rows are `state[start:end]`. `meta` records
`schema_version`, `fps`, `target_colors`, `target_bounds`, `target_min_separation`,
`success_threshold`, `created_at`, `source`, `split_seed`, `n_episodes`, `n_frames`, and
`episode_counts`. It deliberately does not record an observation size.

**Splits are 90/10 train/validation, stratified by target color.** For each color, the
file shuffles that color's episode indices with `default_rng(split_seed)` and assigns
the first `int(n * 0.9)` to train and the rest to validation. Downstream training keeps
using the validation split for monitoring and checkpoint selection.
`PackSummary.warnings` reports a color whose train or validation split is empty.

## Loading and rendering

```python
from tiny_worlds.twin_t import load_dataset, render_frame, render_frames, unpack

with load_dataset("dist/twint.npz") as dataset:
    print(dataset.episode_count, dataset.frame_count)
    episode = dataset.episode(0)
    print(episode.state.shape, episode.action.shape)
    frames = episode.frames(96)  # (steps + 1, 96, 96, 3) uint8, rendered on demand
    for train_episode in dataset.episodes(0):
        ...

single = render_frame(episode.state[0], episode.target_poses, 96)
batch = render_frames(episode.state, episode.target_poses, 224)
```

`load_dataset(path)` eagerly loads every member of the `.npz` into numpy arrays and
validates the episode ends, shapes, dtypes, id count, and labels. `state`,
`episode_ends`, `episode_target`, `episode_target_poses`, `episode_ids`,
`episode_created`, `episode_seed`, and `split` are the stored members. For downstream
convenience, `dataset.action` derives per-episode actions, while `dataset.target` and
`dataset.target_poses` broadcast the per-episode labels to one value per frame. An
`Episode` exposes `state`, `action`, `target`, `target_poses`, `episode_id`, `split`,
`steps`, and `frames(size)`.

`render_frames(states, target_poses, size)` renders a whole trajectory with one reusable
renderer; `target_poses` may be `(2, 3)` (broadcast over frames) or `(T, 2, 3)`.
`render_frame(state, target_poses, size)` renders a single frame. Rendering 224px from
state reproduces the same scene the 96px live observation was sampled from.

`unpack(packed_path, out_dir)` writes the raw v2 per-trial `.npz` files back out;
`pack_trials` is its inverse.

## Symmetry augmentation

The task is invariant under the dihedral group of the square and under swapping the two
target colors. The world is square, the agent is a disc, the block and targets are rigid
T's, the physics is isotropic, and the targets are visual-only (the block does not collide
with them). Rotating or reflecting the whole state therefore leaves the dynamics
unchanged; swapping the two colors only relabels which target is commanded. These are
exact label-preserving symmetries, not new behaviors.

The 16 transforms are `8 dihedral elements x 2 color swaps`. Each is a
`Symmetry(rotation, mirror, swap_colors)`:

- `rotation` is 0..3 counter-clockwise quarter turns about the workspace centre `W/2`.
- `mirror` is `"x"` (`x -> W - x`, vertical centre line), `"y"` (`y -> W - y`,
  horizontal centre line), or `None`. `"y"` equals `rotation=2` composed with `"x"`, so
  `symmetries()` lists only `None` and `"x"`.
- `swap_colors` reverses `target_poses` and flips `target` (`0 <-> 1`). The block and
  agent are unchanged, so the pushed geometric pose moves to the other color index and
  command success is preserved.

Poses are `[x, y, angle]` and state is
`[agent_x, agent_y, block_x, block_y, block_angle]`. A mirror reflects positions and maps
the T angle to `-theta` (`"x"`) or `pi - theta` (`"y"`); a quarter turn rotates positions
about `W/2` and adds `k * pi / 2` to angles. Angles are wrapped to `[-pi, pi)` and arrays
stay float32. The mirror is applied first, then the rotation, then the color swap.

```python
import numpy as np
from tiny_worlds.twin_t import Symmetry, augment_trial, load_trial, save_trial, symmetries

trial = load_trial("data/pink/p0001.npz")
for symmetry in symmetries():  # 16 transforms, identity first
    augmented = augment_trial(trial, symmetry)
    save_trial("data-aug", augmented, f"p0001_{symmetry.name}")

state, target, poses = Symmetry(rotation=1, mirror="x", swap_colors=True).apply(
    trial.state, trial.target, trial.target_poses
)
```

`augment_trial` returns a valid v2 `Trial`: `meta` is preserved, `state` and
`target_poses` are transformed, `target` is flipped when required, and `action` stays
derived from the transformed state.

`relocate_non_target(state, target, target_poses, rng, ...)` is a separate helper that
resamples only the non-commanded target. It keeps the commanded pose, block, and agent
fixed, samples a geometric centre in `target_bounds` with an angle in
`target_angle_range`, and rejects a candidate that is closer than
`target_min_separation` to the commanded target, leaves the walls, overlaps the other
target, is covered by the block at least 50% at any frame, or overlaps the agent at any
frame. It is deterministic given `rng` and raises `ValueError` after `attempts` failures.

The thin CLI writes one file per `(trial, symmetry)`:

```bash
uv run --all-extras python scripts/augment_trials.py --data data --out data-aug --symmetries all
```

**Training-only.** These transforms relabel existing scenes. They add no new long-tail
behaviors and must not be applied to validation or evaluation; keep evaluation on the
unaugmented split.

Two geometric caveats are worth recording. The frozen transform uses the workspace
centre `W/2 = 256`, while the wall inset is `5..506`, centred at 255.5; this half-unit
offset is far below the task scale. The renderer also rasterizes polygons at integer
pixel coordinates, so a re-rendered symmetric scene can differ from a pixel transform of
the original along object outlines even though the object interiors match exactly. The
tests verify the axis and rotation conventions with a one-pixel rasterization tolerance.

## Legacy conversion

`load_legacy_trial(path)` reads a v1 raw trial (which also contains `obs`, `action`,
`coverage`, `success`, `keypoint`, and `n_contacts`), ignores those removed arrays, and
returns a v2 `Trial` with `schema_version` set to 2 and the original seed, creation time,
and active seconds preserved. It is the only migration helper; `load_trial` rejects v1
files.

## Acceptance tests

- Every reset produces two non-overlapping targets at least 180 apart, a block not on a
  target, and an agent not overlapping the block or a target.
- Placing the block exactly on a target gives coverage 1.0 and success; a block on the
  wrong target is task success but not command success.
- The observation contains pixels and agent position only.
- The speed cap holds over a fast pointer path.
- `render_frames` reproduces the live environment's 96px observations from state alone.
- A raw v2 trial round-trips through `save_trial`/`load_trial`; actions follow the
  derived rule including the terminal copy.
- `pack_trials` -> `load_dataset` -> `unpack` round-trips state, episode ends, labels,
  poses, ids, and splits; 70 + 70 trials split 63/7 per color (train/validation),
  126/14 overall.
- A synthetic v1 trial converts to v2 through `load_legacy_trial`.
- The 16 symmetries preserve pairwise distances and per-target coverage, and the renderer
  confirms the mirror axes and rotation sign; augmented trials validate, round-trip, and
  keep the derived action rule.
- `relocate_non_target` is deterministic, keeps the commanded scene fixed, respects the
  target bounds and separation, and rejects block or agent overlap.

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

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
| targets | two T's, colors fixed, positions random, four axis-aligned orientations |
| control | 10 Hz, 10 physics substeps of 0.01 s |
| action | agent target position in `[0, 512]^2` |
| success | block covers the commanded target at least 95% |
| observation | 96 x 96 RGB plus the agent's 2D position |

Target poses are sampled with geometric centers in `(90, 90, 422, 422)`, a minimum
center separation of 180, and one of `(0, pi/2, pi, 3*pi/2)`. The block's geometric
center starts in `(110, 110, 390, 390)` with a continuous angle; the agent starts in
`(50, 50, 450, 450)`. The stored poses (`target_poses`, `block_pose`) are T body poses
in the PushT convention, so moving from a sampled center to a pose subtracts the
rotated local centroid `(0, 40.7)`. Sampling rejects any layout where the block or
agent overlaps a target, the agent overlaps the block, or the block already covers a
target. The bounds keep every T inside the walls: the axis-aligned target extents are
at most 79.3 from its geometric center, and the continuous-angle block extent is at
most 99.4 from its center.

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

- `obs.pixels` is `(96, 96, 3)` uint8 and `obs.agent_pos` is `(2,)` float32.
- `info` is a `StepInfo`: block pose, eight block keypoints, target poses, coverage for
  both targets, `command_success`, `task_success`, `achieved_target`, contacts, step.
- `terminate_on="task"` ends on either target (evaluation default).
  `terminate_on="command"` ends only on the commanded target (collector default); a
  block on the wrong target can be pushed off.
- With `command=None`, success and termination use either target.
- `set_command` changes the command mid-episode for the step-50 switch.
- `state()` returns `TwinTState`; `state().as_vector()` is
  `[agent_x, agent_y, block_x, block_y, block_angle]`, the `reset(state=...)` format.
  `reset(state=..., target_poses=...)` reproduces a held-out layout; `target_poses` is
  `(2, 3)` for `[pink, green]` and is sampled when omitted.
- `target_poses` is evaluation-only and never enters the observation.

The agent is speed-limited by `max_agent_speed` (default 250 px/s, 25 px per control
step), applied as a velocity clamp inside the environment so collection and evaluation
have the same dynamics. PushT itself has no explicit cap; this is a TwinT addition for
data quality.

## Raw trial format

One successful trial is one uncompressed `.npz` under `data/pink/` or `data/green/`.
Every per-frame array has the same first dimension `T + 1`, where `T` is the number of
executed control actions. The last row is the terminal success frame; its action is a
copy of the last executed action, so the training pair at row `t < T` remains
`(obs[t], action[t])`.

| key | shape | dtype | meaning |
|---|---|---|---|
| `obs` | `(T+1, 96, 96, 3)` | uint8 | observation at each step |
| `state` | `(T+1, 5)` | float32 | `[agent_x, agent_y, block_x, block_y, block_angle]` |
| `action` | `(T+1, 2)` | float32 | pointer target; last row copied |
| `coverage` | `(T+1, 2)` | float32 | coverage of `[pink, green]` |
| `success` | `(T+1,)` | bool | true only on the terminal frame |
| `keypoint` | `(T+1, 9, 2)` | float32 | eight block corners + agent center |
| `n_contacts` | `(T+1, 1)` | int32 | contact points in the step |
| `target` | `()` | int8 | commanded color |
| `target_poses` | `(2, 3)` | float32 | `[pink, green]` poses |
| `meta` | `()` | unicode | JSON string: schema version, seed, created, steps, active seconds |

Write to a temporary file in the same directory, flush, then `os.replace` into place.
A crash cannot leave a half-written trial with a final name. A successful trial is
written immediately; a discarded trial is not written.

## Packed zarr format

`pack` writes one `twint.zarr.zip`. The zip is a zarr v2 store with the zarr keys at its
root, so `zarr.storage.ZipStore` can open it directly. Each chunk uses zstd.

```text
data/img          (N, 96, 96, 3) uint8
data/state        (N, 5) float32
data/action       (N, 2) float32
data/coverage     (N, 2) float32
data/success      (N,) bool
data/keypoint     (N, 9, 2) float32
data/n_contacts   (N, 1) int32
data/target       (N,) int8
data/target_poses (N, 2, 3) float32
meta/episode_ends   (E,) int64
meta/episode_target (E,) int8
meta/episode_success(E,) bool
meta/episode_steps  (E,) int32
meta/episode_ids    (E,) S32
meta/episode_created(E,) float64    unix seconds
meta/episode_seed   (E,) int64
meta/split          (E,) int8      0 train, 1 validation, 2 test
```

`meta/episode_ends` is cumulative over rows, so `N = episode_ends[-1]`.
`meta/episode_steps` counts executed actions, one less than the row count. Splits are
stratified by target color, 80/10/10, with seed 0, and stored so the test split is
fixed. Root attributes record schema version, colors, bounds, threshold, fps, source,
creation time, and split seed.

`load_dataset(path)` opens a `.zarr.zip` lazily and exposes arrays, `episode(i)`,
`episodes(split)`, `episode_count`, and `frame_count`. `unpack(zarr_path, out_dir)`
writes the raw per-trial `.npz` files back out. `pack_trials(data_dir, out_path)` is the
inverse.

## Acceptance tests

- Every reset produces two non-overlapping targets at least 180 apart, a block not on a
  target, and an agent not overlapping the block or a target.
- Placing the block exactly on a target gives coverage 1.0 and success; a block on the
  wrong target is task success but not command success.
- The observation contains pixels and agent position only.
- The speed cap holds over a fast pointer path.
- A raw trial round-trips through `save_trial`/`load_trial`.
- `pack_trials` -> `load_dataset` -> `unpack` round-trips arrays, episode ends, labels,
  and splits.

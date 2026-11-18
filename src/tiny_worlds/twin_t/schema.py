"""Shared TwinT constants and the on-disk data schema.

The raw trial format is one uncompressed ``.npz`` per successful trial. The packed
format is a zarr v2 store inside ``twint.zarr.zip``. Both are defined here so the
environment, the collector, and the loaders cannot disagree about a key or a dtype.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 1

PINK: Final[int] = 0
GREEN: Final[int] = 1
TARGET_NAMES: Final[tuple[str, str]] = ("pink", "green")
TARGET_NAME_TO_INDEX: Final[dict[str, int]] = {"pink": PINK, "green": GREEN}

TARGET_COLORS: Final[tuple[tuple[int, int, int], tuple[int, int, int]]] = (
    (255, 79, 163),
    (34, 197, 94),
)
BLOCK_COLOR: Final[tuple[int, int, int]] = (119, 136, 153)
AGENT_COLOR: Final[tuple[int, int, int]] = (65, 105, 225)
WALL_COLOR: Final[tuple[int, int, int]] = (211, 211, 211)
BACKGROUND_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)

FPS: Final[int] = 10
OBSERVATION_SIZE: Final[int] = 96
SUCCESS_THRESHOLD: Final[float] = 0.95

# Per-frame arrays in one raw trial .npz, all with the same first dimension.
# ``action[-1]`` is a copy of the last executed action: the terminal frame has no
# new action of its own, and the copy keeps every array the same length.
TRIAL_ARRAYS: Final[tuple[str, ...]] = (
    "obs",  # (T+1, 96, 96, 3) uint8
    "state",  # (T+1, 5) float32  [agent_x, agent_y, block_x, block_y, block_angle]
    "action",  # (T+1, 2) float32
    "coverage",  # (T+1, 2) float32 [pink, green]
    "success",  # (T+1,) bool     true only on the terminal frame
    "keypoint",  # (T+1, 9, 2) float32  8 block corners + agent center
    "n_contacts",  # (T+1, 1) int32
)

# Per-episode arrays in one raw trial .npz.
TRIAL_EPISODE_ARRAYS: Final[tuple[str, ...]] = (
    "target",  # () int8
    "target_poses",  # (2, 3) float32
)

# Per-frame arrays packed into the zarr store.
ZARR_DATA_ARRAYS: Final[tuple[str, ...]] = (
    "img",
    "state",
    "action",
    "coverage",
    "success",
    "keypoint",
    "n_contacts",
    "target",
    "target_poses",
)

# Per-episode arrays packed into the zarr store.
ZARR_META_ARRAYS: Final[tuple[str, ...]] = (
    "episode_ends",
    "episode_target",
    "episode_success",
    "episode_steps",
    "episode_ids",
    "episode_created",
    "episode_seed",
    "split",
)

SPLIT_TRAIN: Final[int] = 0
SPLIT_VALIDATION: Final[int] = 1
SPLIT_TEST: Final[int] = 2
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "validation", "test")
SPLIT_FRACTIONS: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)


def target_name(target: int) -> str:
    """Return the color name for a target index."""
    if target not in (PINK, GREEN):
        raise ValueError(f"target must be {PINK} or {GREEN}, got {target!r}")
    return TARGET_NAMES[target]


def target_index(name: str) -> int:
    """Return the target index for a color name."""
    try:
        return TARGET_NAME_TO_INDEX[name]
    except KeyError as error:
        raise ValueError(f"target name must be one of {TARGET_NAMES}, got {name!r}") from error

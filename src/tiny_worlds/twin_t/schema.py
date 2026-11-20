"""Shared TwinT constants and the on-disk data schema.

The raw trial format is one uncompressed ``.npz`` per successful trial holding only
``state``, the per-episode ``target`` and ``target_poses``, and ``meta``. The packed
format is one compressed ``twint.npz``. Both are defined here so the environment, the
collector, and the loaders cannot disagree about a key, a dtype, or a split.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 2

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
SUCCESS_THRESHOLD: Final[float] = 0.95

# Per-frame arrays in one raw trial .npz, all with the same first dimension.
# ``state`` is (T+1, 5) float32: [agent_x, agent_y, block_x, block_y, block_angle].
# Actions are derived from consecutive states and never stored.
TRIAL_ARRAYS: Final[tuple[str, ...]] = ("state",)

# Per-episode arrays in one raw trial .npz.
TRIAL_EPISODE_ARRAYS: Final[tuple[str, ...]] = (
    "target",  # () int8
    "target_poses",  # (2, 3) float32
)

# Members of the packed ``dist/twint.npz`` that vary per frame.
PACKED_DATA_ARRAYS: Final[tuple[str, ...]] = ("state",)

# Members of the packed ``dist/twint.npz`` that vary per episode.
PACKED_EPISODE_ARRAYS: Final[tuple[str, ...]] = (
    "episode_ends",
    "episode_target",
    "episode_target_poses",
    "episode_ids",
    "episode_created",
    "episode_seed",
    "split",
)

# Every member of the packed ``dist/twint.npz``.
PACKED_ARRAYS: Final[tuple[str, ...]] = PACKED_DATA_ARRAYS + PACKED_EPISODE_ARRAYS + ("meta",)

SPLIT_TRAIN: Final[int] = 0
SPLIT_VALIDATION: Final[int] = 1
SPLIT_NAMES: Final[tuple[str, str]] = ("train", "validation")
SPLIT_FRACTIONS: Final[tuple[float, float]] = (0.9, 0.1)


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

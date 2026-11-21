"""Exact task symmetries for TwinT, used as label-preserving augmentation.

The task is invariant under the dihedral group of the square (rotations by a
quarter turn and mirror images) and under swapping the two target colors.  A
symmetry acts on the simulator state and the per-episode target poses; frames
are re-rendered on demand and actions stay derived, so a transformed ``Trial``
is still a valid v2 trial.

These transforms are *not* new data.  They relabel scenes that are already in
the training distribution and must never be applied to evaluation data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point

from .config import DEFAULT_CONFIG, TwinTConfig
from .physics import center_from_pose, coverage, inside_walls, pose_from_center, tee_polygon
from .trials import Trial, validate_trial
from .types import FloatArray

type Mirror = Literal["x", "y"] | None

_QUARTER_TURN: Final[float] = math.pi / 2.0
_TWO_PI: Final[float] = 2.0 * math.pi
# A relocated non-commanded target may be touched by the block at most this much.
MAX_NON_TARGET_COVERAGE: Final[float] = 0.5


@dataclass(frozen=True)
class Symmetry:
    """One exact TwinT task symmetry.

    ``rotation`` is a number of counter-clockwise quarter turns about the
    workspace centre.  ``mirror`` reflects across the vertical centre line
    (``"x"``: ``x -> W - x``) or the horizontal centre line (``"y"``:
    ``y -> W - y``).  ``swap_colors`` exchanges the pink and green targets and
    flips the commanded label.  A symmetry applies the mirror first, then the
    rotation, then the color swap.
    """

    rotation: int = 0
    mirror: Mirror = None
    swap_colors: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.rotation, bool) or not isinstance(self.rotation, int):
            raise TypeError("rotation must be an integer in 0..3")
        if not 0 <= self.rotation <= 3:
            raise ValueError(f"rotation must be in 0..3, got {self.rotation}")
        if self.mirror not in (None, "x", "y"):
            raise ValueError(f"mirror must be None, 'x', or 'y', got {self.mirror!r}")
        if not isinstance(self.swap_colors, bool):
            raise TypeError("swap_colors must be a bool")

    def apply(
        self,
        state: NDArray[np.floating],
        target: int | NDArray[np.integer],
        target_poses: NDArray[np.floating],
    ) -> tuple[FloatArray, NDArray[np.int8], FloatArray]:
        """Transform ``(state, target, target_poses)``; inputs are not mutated."""
        states = _as_state(state)
        poses = _as_poses(target_poses)
        label = _as_target(target)

        rows = states.astype(np.float64).copy()
        transformed = poses.astype(np.float64).copy()

        if self.mirror == "x":
            # Reflect positions across x = W/2; a T is symmetric about its stem,
            # so the pose angle only changes sign.
            rows[..., 0] = _W - rows[..., 0]
            rows[..., 2] = _W - rows[..., 2]
            rows[..., 4] = -rows[..., 4]
            transformed[..., 0] = _W - transformed[..., 0]
            transformed[..., 2] = -transformed[..., 2]
        elif self.mirror == "y":
            # Reflect across y = W/2; the T pose angle becomes pi - theta.
            rows[..., 1] = _W - rows[..., 1]
            rows[..., 3] = _W - rows[..., 3]
            rows[..., 4] = math.pi - rows[..., 4]
            transformed[..., 1] = _W - transformed[..., 1]
            transformed[..., 2] = math.pi - transformed[..., 2]

        if self.rotation:
            # Exact quarter-turn arithmetic (no cos/sin round-off) about W/2.
            rows[..., :2] = _rotate_quarter_turns(rows[..., :2], self.rotation)
            rows[..., 2:4] = _rotate_quarter_turns(rows[..., 2:4], self.rotation)
            rows[..., 4] = rows[..., 4] + self.rotation * _QUARTER_TURN
            transformed[..., :2] = _rotate_quarter_turns(transformed[..., :2], self.rotation)
            transformed[..., 2] = transformed[..., 2] + self.rotation * _QUARTER_TURN

        rows[..., 4] = _wrap_angles(rows[..., 4])
        transformed[..., 2] = _wrap_angles(transformed[..., 2])

        if self.swap_colors:
            transformed = transformed[::-1].copy()
            label = 1 - label

        return (
            rows.astype(np.float32),
            np.array(label, dtype=np.int8),
            transformed.astype(np.float32),
        )

    @property
    def name(self) -> str:
        """A stable, unique name such as ``"r1_mx_swap"`` or ``"identity"``."""
        parts = []
        if self.rotation:
            parts.append(f"r{self.rotation}")
        if self.mirror is not None:
            parts.append(f"m{self.mirror}")
        if self.swap_colors:
            parts.append("swap")
        return "_".join(parts) if parts else "identity"


def symmetries() -> tuple[Symmetry, ...]:
    """The 16 exact symmetries: 8 dihedral elements x color swap, stable order.

    The first eight are the pure dihedral elements (identity, ``r1``, ``r2``,
    ``r3``, ``mx``, ``r1_mx``, ``r2_mx``, ``r3_mx``); the last eight repeat them
    with ``swap_colors=True``.  ``mirror="y"`` is omitted because it equals
    ``r2`` composed with ``mirror="x"``.
    """
    return tuple(
        Symmetry(rotation=rotation, mirror=mirror, swap_colors=swap)
        for swap in (False, True)
        for mirror in (None, "x")
        for rotation in range(4)
    )


def augment_trial(trial: Trial, symmetry: Symmetry) -> Trial:
    """Return a transformed copy of a v2 trial with the same metadata."""
    if not isinstance(trial, Trial):
        raise TypeError(f"trial must be a Trial, got {type(trial).__name__}")
    if not isinstance(symmetry, Symmetry):
        raise TypeError(f"symmetry must be a Symmetry, got {type(symmetry).__name__}")
    state, target, target_poses = symmetry.apply(trial.state, trial.target, trial.target_poses)
    augmented = Trial(state=state, target=target, target_poses=target_poses, meta=trial.meta)
    validate_trial(augmented)
    return augmented


def relocate_non_target(
    state: NDArray[np.floating],
    target: int | NDArray[np.integer],
    target_poses: NDArray[np.floating],
    rng: np.random.Generator,
    config: TwinTConfig = DEFAULT_CONFIG,
    attempts: int = 256,
) -> tuple[FloatArray, NDArray[np.int8], FloatArray]:
    """Move and rotate the non-commanded target to a fresh valid pose.

    The commanded target, the block, and the agent stay fixed.  Candidate
    geometric centres are sampled uniformly inside ``config.target_bounds`` with
    an angle in ``config.target_angle_range``.  A candidate is rejected when it
    is closer than ``config.target_min_separation`` to the commanded target, when
    it leaves the walls, when it overlaps the other target, when the block covers
    it by ``MAX_NON_TARGET_COVERAGE`` or more at any frame, or when it overlaps
    the agent at any frame.  The result is deterministic given ``rng``.

    Raises ``ValueError`` when no candidate is accepted in ``attempts`` tries.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise TypeError("attempts must be an integer")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    states = _as_state(state)
    poses = _as_poses(target_poses)
    label = _as_target(target)
    other = 1 - label
    scale = config.block_scale
    flat = states.reshape(-1, 5)

    commanded_center = center_from_pose(poses[label], scale)
    commanded_polygon = tee_polygon(poses[label], scale)
    block_poses = np.unique(flat[:, 2:5], axis=0)
    agent_positions = np.unique(flat[:, :2], axis=0)

    low_x, low_y, high_x, high_y = config.target_bounds
    low_angle, high_angle = config.target_angle_range
    for _ in range(attempts):
        center = rng.uniform((low_x, low_y), (high_x, high_y))
        angle = rng.uniform(low_angle, high_angle)
        candidate = pose_from_center(center, angle, scale)
        if np.linalg.norm(center - commanded_center) < config.target_min_separation:
            continue
        candidate_polygon = tee_polygon(candidate, scale)
        if not inside_walls(candidate_polygon):
            continue
        if candidate_polygon.intersection(commanded_polygon).area > 0.0:
            continue
        if _block_covers_candidate(block_poses, candidate, scale):
            continue
        if _agent_overlaps_candidate(agent_positions, candidate_polygon, config.agent_radius):
            continue
        relocated = poses.copy()
        relocated[other] = candidate
        return states.copy(), np.array(label, dtype=np.int8), relocated

    raise ValueError(f"could not relocate the non-target pose in {attempts} attempts")


_W = float(DEFAULT_CONFIG.workspace)


def _rotate_quarter_turns(points: NDArray[np.floating], rotation: int) -> FloatArray:
    """Rotate ``(..., 2)`` points counter-clockwise by ``rotation`` quarter turns."""
    x = points[..., 0]
    y = points[..., 1]
    centre = _W / 2.0
    if rotation == 1:
        return np.stack([2.0 * centre - y, x], axis=-1)
    if rotation == 2:
        return np.stack([2.0 * centre - x, 2.0 * centre - y], axis=-1)
    return np.stack([y, 2.0 * centre - x], axis=-1)


def _wrap_angles(angles: NDArray[np.floating]) -> FloatArray:
    """Wrap angles into ``[-pi, pi)``."""
    return (np.asarray(angles, dtype=np.float64) + math.pi) % _TWO_PI - math.pi


def _block_covers_candidate(
    block_poses: NDArray[np.floating],
    candidate: NDArray[np.floating],
    scale: float,
) -> bool:
    targets = candidate[None, :]
    for block_pose in block_poses:
        if float(coverage(block_pose, targets, scale)[0]) >= MAX_NON_TARGET_COVERAGE:
            return True
    return False


def _agent_overlaps_candidate(
    agent_positions: NDArray[np.floating],
    candidate_polygon: object,
    agent_radius: float,
) -> bool:
    for agent_position in agent_positions:
        point = Point(float(agent_position[0]), float(agent_position[1]))
        if point.distance(candidate_polygon) < agent_radius:
            return True
    return False


def _as_state(value: object) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("state is not numeric") from error
    if array.ndim not in (1, 2) or array.shape[-1] != 5:
        raise ValueError(f"state must have shape (5,) or (T, 5), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("state must contain only finite values")
    return array


def _as_poses(value: object) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("target_poses is not numeric") from error
    if array.shape != (2, 3):
        raise ValueError(f"target_poses must have shape (2, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("target_poses must contain only finite values")
    return array


def _as_target(value: object) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"target must be a scalar integer, got {array.dtype} {array.shape}")
    target = int(array)
    if target not in (0, 1):
        raise ValueError(f"target must be 0 or 1, got {target}")
    return target

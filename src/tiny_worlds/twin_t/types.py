"""Shared types for TwinT.

The policy observation carries only pixels and the agent position. The block pose,
target poses, coverage, and success live in :class:`StepInfo` and are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type UInt8Array = NDArray[np.uint8]
type Int32Array = NDArray[np.int32]
type BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Command:
    """A target color index, or the null command."""

    target: int | None

    def __post_init__(self) -> None:
        if self.target not in (None, 0, 1):
            raise ValueError(f"command target must be None, 0, or 1; got {self.target!r}")

    @property
    def is_null(self) -> bool:
        return self.target is None


@dataclass(frozen=True)
class TwinTObservation:
    """What the policy sees."""

    pixels: UInt8Array  # (H, W, 3)
    agent_pos: FloatArray  # (2,)


@dataclass(frozen=True)
class TwinTState:
    """The simulator state used for evaluation and resets, never for the policy."""

    agent_pos: FloatArray  # (2,)
    block_pose: FloatArray  # (3,) [x, y, angle]
    target_poses: FloatArray  # (2, 3) [pink, green]
    command: Command
    step: int

    def as_vector(self) -> FloatArray:
        """Return ``[agent_x, agent_y, block_x, block_y, block_angle]``."""
        return np.concatenate([self.agent_pos, self.block_pose]).astype(np.float32)


@dataclass(frozen=True)
class StepInfo:
    """Environment diagnostics for one control step."""

    block_pose: FloatArray  # (3,)
    block_keypoints: FloatArray  # (8, 2)
    target_poses: FloatArray  # (2, 3)
    coverage: FloatArray  # (2,) [pink, green]
    command_success: bool
    task_success: bool
    achieved_target: int | None
    n_contacts: int
    step: int

"""The narrow environment protocol the collector and evaluation code depend on."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .types import FloatArray, StepInfo, TwinTObservation, TwinTState


class TwinTEnvironment(Protocol):
    """A TwinT environment, real or fake."""

    @property
    def target_poses(self) -> FloatArray:
        """(2, 3) target poses, [pink, green]."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        command: int | None = None,
        state: NDArray[np.floating] | None = None,
    ) -> tuple[TwinTObservation, StepInfo]:
        """Reset the environment and return the first observation and info."""
        ...

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[TwinTObservation, float, bool, bool, StepInfo]:
        """Take one control step."""
        ...

    def set_command(self, command: int | None) -> None:
        """Change the commanded target mid-episode."""
        ...

    def state(self) -> TwinTState:
        """Return the current simulator state."""
        ...

    def render(self, size: int | None = None) -> NDArray[np.uint8]:
        """Return an RGB frame, (size, size, 3) uint8."""
        ...

    def close(self) -> None:
        """Release rendering resources."""
        ...

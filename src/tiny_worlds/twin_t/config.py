"""Configuration for the TwinT environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .schema import AGENT_COLOR, BACKGROUND_COLOR, BLOCK_COLOR, TARGET_COLORS, WALL_COLOR

Color = tuple[int, int, int]
Bounds = tuple[float, float, float, float]
Termination = Literal["task", "command"]


@dataclass(frozen=True)
class TwinTConfig:
    """Geometry, physics, colors, and limits for TwinT.

    ``terminate_on="task"`` ends an episode when either target is covered; this is the
    evaluation default. ``terminate_on="command"`` ends only on the commanded target and
    is what the data collector uses so a human can push the block off a wrong target.
    """

    workspace: int = 512
    control_hz: int = 10
    physics_dt: float = 0.01
    episode_limit: int = 300
    success_threshold: float = 0.95

    agent_radius: float = 15.0
    block_scale: float = 30.0

    target_bounds: Bounds = (90.0, 90.0, 422.0, 422.0)
    target_min_separation: float = 180.0
    target_angle_range: tuple[float, float] = (-math.pi, math.pi)

    # ``block_bounds`` and ``agent_bounds`` are bounds on the geometric center. The
    # block is placed so its geometric center is sampled here, then its body pose is
    # reported in the state. The bounds keep the 120-wide T inside the 5..506 walls for
    # any continuous block angle.
    block_bounds: Bounds = (110.0, 110.0, 390.0, 390.0)
    agent_bounds: Bounds = (50.0, 50.0, 450.0, 450.0)

    max_agent_speed: float = 250.0
    observation_size: int = 96
    render_size: int = 512
    terminate_on: Termination = "task"

    target_colors: tuple[Color, Color] = TARGET_COLORS
    block_color: Color = BLOCK_COLOR
    agent_color: Color = AGENT_COLOR
    wall_color: Color = WALL_COLOR
    background_color: Color = BACKGROUND_COLOR

    def __post_init__(self) -> None:
        if self.workspace <= 0:
            raise ValueError("workspace must be positive")
        if self.control_hz <= 0 or self.physics_dt <= 0:
            raise ValueError("control_hz and physics_dt must be positive")
        if self.substeps != int(1 / (self.physics_dt * self.control_hz)):
            raise ValueError("physics_dt * control_hz must divide 1 evenly")
        if self.episode_limit < 1:
            raise ValueError("episode_limit must be at least 1")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        if self.agent_radius <= 0 or self.block_scale <= 0:
            raise ValueError("agent_radius and block_scale must be positive")
        for name, bounds in (
            ("target_bounds", self.target_bounds),
            ("block_bounds", self.block_bounds),
            ("agent_bounds", self.agent_bounds),
        ):
            _check_bounds(name, bounds, self.workspace)
        low_angle, high_angle = self.target_angle_range
        if not (-math.pi <= low_angle < high_angle <= math.pi):
            raise ValueError("target_angle_range must be an ordered range inside [-pi, pi]")
        if self.target_min_separation <= 0:
            raise ValueError("target_min_separation must be positive")
        if self.max_agent_speed <= 0:
            raise ValueError("max_agent_speed must be positive")
        if self.observation_size <= 0 or self.render_size <= 0:
            raise ValueError("observation_size and render_size must be positive")
        if self.terminate_on not in ("task", "command"):
            raise ValueError("terminate_on must be 'task' or 'command'")
        if len(self.target_colors) != 2:
            raise ValueError("target_colors must have two entries")

    @property
    def substeps(self) -> int:
        """Physics steps per control step."""
        return round(1.0 / (self.physics_dt * self.control_hz))

    @property
    def pixels_per_step(self) -> float:
        """Maximum agent displacement in one control step."""
        return self.max_agent_speed / self.control_hz


def _check_bounds(name: str, bounds: Bounds, workspace: int) -> None:
    low_x, low_y, high_x, high_y = bounds
    if not (0 <= low_x < high_x <= workspace and 0 <= low_y < high_y <= workspace):
        raise ValueError(f"{name} must be an ordered box inside the workspace")


DEFAULT_CONFIG = TwinTConfig()
COLLECTOR_CONFIG = TwinTConfig(episode_limit=600, terminate_on="command")

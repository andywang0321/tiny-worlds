"""Headless pygame rendering for TwinT.

pygame is imported only after forcing the dummy video driver, and no display is
initialised: every frame is drawn to an offscreen surface at the requested size.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from dataclasses import dataclass

import numpy as np
import pygame
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, TwinTConfig
from .physics import WALL_MAX, WALL_MIN, WALL_RADIUS, apply_pose, tee_shape_vertices
from .types import FloatArray, UInt8Array

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Scene:
    """The dynamic parts of one frame."""

    agent_pos: FloatArray
    block_pose: FloatArray
    target_poses: FloatArray


class TwinTRenderer:
    """Draw a scene to an offscreen surface without initialising a display."""

    def __init__(self, config: TwinTConfig) -> None:
        self._config = config
        self._closed = False

    def render(self, size: int, scene: Scene) -> UInt8Array:
        if self._closed:
            raise RuntimeError("renderer is closed")
        if size <= 0:
            raise ValueError("size must be positive")
        surface = pygame.Surface((size, size))
        self._draw(surface, scene)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def close(self) -> None:
        self._closed = True

    def _draw(self, surface: pygame.Surface, scene: Scene) -> None:
        surface.fill(self._config.background_color)
        self._draw_walls(surface)
        for index, pose in enumerate(scene.target_poses):
            self._draw_tee(surface, pose, self._config.target_colors[index])
        self._draw_tee(surface, scene.block_pose, self._config.block_color)
        self._draw_agent(surface, scene.agent_pos)

    def _draw_walls(self, surface: pygame.Surface) -> None:
        scale = self._scale(surface)
        thickness = max(1, round(2 * WALL_RADIUS * scale))
        low, high = WALL_MIN * scale, WALL_MAX * scale
        half = thickness // 2
        color = self._config.wall_color
        pygame.draw.rect(surface, color, (round(low), round(low) - half, round(high - low), thickness))
        pygame.draw.rect(surface, color, (round(low), round(high) - half, round(high - low), thickness))
        pygame.draw.rect(surface, color, (round(low) - half, round(low), thickness, round(high - low)))
        pygame.draw.rect(surface, color, (round(high) - half, round(low), thickness, round(high - low)))

    def _draw_tee(self, surface: pygame.Surface, pose: NDArray[np.floating], color: Color) -> None:
        scale = self._scale(surface)
        for vertices in tee_shape_vertices(self._config.block_scale):
            world = apply_pose(pose, np.array(vertices, dtype=np.float64)) * scale
            points = [(round(x), round(y)) for x, y in world]
            pygame.draw.polygon(surface, color, points)

    def _draw_agent(self, surface: pygame.Surface, agent_pos: NDArray[np.floating]) -> None:
        scale = self._scale(surface)
        center = (round(float(agent_pos[0]) * scale), round(float(agent_pos[1]) * scale))
        radius = max(1, round(self._config.agent_radius * scale))
        pygame.draw.circle(surface, self._config.agent_color, center, radius)
        inner = max(1, radius - max(1, round(4 * scale)))
        pygame.draw.circle(surface, _lighten(self._config.agent_color), center, inner)

    def _scale(self, surface: pygame.Surface) -> float:
        return surface.get_width() / self._config.workspace


def render_frames(states: NDArray[np.floating], target_poses: NDArray[np.floating], size: int) -> UInt8Array:
    """Render a state trajectory at ``size``; ``target_poses`` is ``(2, 3)`` or ``(T, 2, 3)``."""
    rows = _check_states(states)
    poses = _check_target_poses(target_poses, rows.shape[0])
    renderer = TwinTRenderer(DEFAULT_CONFIG)
    try:
        frames = [renderer.render(size, _scene(row, pose)) for row, pose in zip(rows, poses, strict=True)]
    finally:
        renderer.close()
    return np.stack(frames)


def render_frame(state: NDArray[np.floating], target_poses: NDArray[np.floating], size: int) -> UInt8Array:
    """Render one ``[agent_x, agent_y, block_x, block_y, block_angle]`` frame at ``size``."""
    return render_frames(np.asarray(state)[None, :], target_poses, size)[0]


def _check_states(states: NDArray[np.floating]) -> FloatArray:
    rows = np.asarray(states, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 5:
        raise ValueError(f"states must have shape (T, 5) with T >= 1, got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("states must be finite")
    return rows


def _check_target_poses(target_poses: NDArray[np.floating], frames: int) -> FloatArray:
    poses = np.asarray(target_poses, dtype=np.float32)
    if poses.shape == (2, 3):
        return np.broadcast_to(poses, (frames, 2, 3))
    if poses.shape == (frames, 2, 3):
        return poses
    raise ValueError(f"target_poses must have shape (2, 3) or ({frames}, 2, 3), got {poses.shape}")


def _scene(state: NDArray[np.floating], target_poses: NDArray[np.floating]) -> Scene:
    return Scene(agent_pos=state[:2], block_pose=state[2:5], target_poses=target_poses)


def _lighten(color: Color) -> Color:
    red, green, blue = color
    return (min(255, round(red * 1.2)), min(255, round(green * 1.2)), min(255, round(blue * 1.2)))

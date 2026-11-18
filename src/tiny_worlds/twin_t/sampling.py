"""Rejection sampling for valid TwinT layouts.

Sampling works on geometric centers. A sampled center becomes a body pose by
subtracting the rotated local centroid, so the walls are checked against the
actual T and not against its origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point

from .config import TwinTConfig
from .physics import circle_inside_walls, inside_walls, pose_from_center, tee_polygon
from .types import FloatArray

MAX_ATTEMPTS: Final[int] = 2000
AGENT_CLEARANCE: Final[float] = 0.5


class TwinTSamplingError(RuntimeError):
    """No valid layout was found, or a supplied layout is invalid."""


@dataclass(frozen=True)
class Layout:
    """A sampled, mutually non-overlapping initial layout."""

    agent_pos: FloatArray
    block_pose: FloatArray
    target_poses: FloatArray


def parse_state(
    state: NDArray[np.floating], config: TwinTConfig
) -> tuple[FloatArray, FloatArray]:
    """Parse a held-out five-vector; overlaps are allowed, walls are not."""
    vector = _as_float_array(state, (5,), "state")
    agent_pos, block_pose = vector[:2].copy(), vector[2:].copy()
    if not circle_inside_walls(agent_pos, config.agent_radius):
        raise TwinTSamplingError("agent is outside the walls")
    if not inside_walls(tee_polygon(block_pose, config.block_scale)):
        raise TwinTSamplingError("block is outside the walls")
    return agent_pos, block_pose


def parse_target_poses(
    target_poses: NDArray[np.floating], config: TwinTConfig
) -> FloatArray:
    """Parse explicit target poses and require two non-overlapping T's in the walls."""
    poses = _as_float_array(target_poses, (2, 3), "target_poses")
    polygons = [tee_polygon(pose, config.block_scale) for pose in poses]
    if not all(inside_walls(polygon) for polygon in polygons):
        raise TwinTSamplingError("target is outside the walls")
    if polygons[0].intersection(polygons[1]).area > 0.0:
        raise TwinTSamplingError("targets overlap")
    return poses


def sample_layout(
    rng: np.random.Generator,
    config: TwinTConfig,
    target_poses: NDArray[np.floating] | None = None,
) -> Layout:
    """Sample a block and agent (and targets when omitted) that do not overlap."""
    if target_poses is None:
        targets = _sample_targets(rng, config)
    else:
        targets = parse_target_poses(target_poses, config)
    block_pose = _sample_block_pose(rng, config, targets)
    agent_pos = _sample_agent_pos(rng, config, block_pose, targets)
    return Layout(agent_pos=agent_pos, block_pose=block_pose, target_poses=targets)


def sample_targets_avoiding(
    rng: np.random.Generator,
    config: TwinTConfig,
    agent_pos: NDArray[np.floating],
    block_pose: NDArray[np.floating],
) -> FloatArray:
    """Sample target poses that stay clear of a fixed block and agent."""
    for _ in range(MAX_ATTEMPTS):
        targets = _sample_targets_once(rng, config)
        if targets is None:
            continue
        if not _agent_clear(np.asarray(agent_pos), np.asarray(block_pose), targets, config):
            continue
        if _block_overlaps_targets(np.asarray(block_pose), targets, config):
            continue
        return targets
    raise TwinTSamplingError(f"could not sample targets in {MAX_ATTEMPTS} attempts")


def _as_float_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TwinTSamplingError(f"{name} is not numeric") from error
    if array.shape != shape:
        raise TwinTSamplingError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise TwinTSamplingError(f"{name} must be finite")
    return array.astype(np.float32)


def _sample_targets(rng: np.random.Generator, config: TwinTConfig) -> FloatArray:
    for _ in range(MAX_ATTEMPTS):
        targets = _sample_targets_once(rng, config)
        if targets is not None:
            return targets
    raise TwinTSamplingError(f"could not sample targets in {MAX_ATTEMPTS} attempts")


def _sample_targets_once(
    rng: np.random.Generator, config: TwinTConfig
) -> FloatArray | None:
    low_x, low_y, high_x, high_y = config.target_bounds
    centers = rng.uniform((low_x, low_y), (high_x, high_y), size=(2, 2))
    if np.linalg.norm(centers[0] - centers[1]) < config.target_min_separation:
        return None
    low_angle, high_angle = config.target_angle_range
    angles = rng.uniform(low_angle, high_angle, size=2)
    poses = np.stack(
        [pose_from_center(center, float(angle), config.block_scale) for center, angle in zip(centers, angles)]
    )
    return poses if _targets_valid(poses, config) else None


def _targets_valid(poses: FloatArray, config: TwinTConfig) -> bool:
    polygons = [tee_polygon(pose, config.block_scale) for pose in poses]
    if not all(inside_walls(polygon) for polygon in polygons):
        return False
    return polygons[0].intersection(polygons[1]).area <= 0.0


def _sample_block_pose(
    rng: np.random.Generator, config: TwinTConfig, target_poses: FloatArray
) -> FloatArray:
    for _ in range(MAX_ATTEMPTS):
        low_x, low_y, high_x, high_y = config.block_bounds
        center = rng.uniform((low_x, low_y), (high_x, high_y))
        angle = rng.uniform(-np.pi, np.pi)
        pose = pose_from_center(center, angle, config.block_scale)
        if _block_valid(pose, target_poses, config):
            return pose
    raise TwinTSamplingError(f"could not sample a block in {MAX_ATTEMPTS} attempts")


def _block_valid(
    block_pose: FloatArray, target_poses: FloatArray, config: TwinTConfig
) -> bool:
    if not inside_walls(tee_polygon(block_pose, config.block_scale)):
        return False
    return not _block_overlaps_targets(block_pose, target_poses, config)


def _block_overlaps_targets(
    block_pose: NDArray[np.floating], target_poses: NDArray[np.floating], config: TwinTConfig
) -> bool:
    block = tee_polygon(block_pose, config.block_scale)
    for target_pose in target_poses:
        target = tee_polygon(target_pose, config.block_scale)
        if block.intersection(target).area > 0.0:
            return True
    return False


def _sample_agent_pos(
    rng: np.random.Generator,
    config: TwinTConfig,
    block_pose: FloatArray,
    target_poses: FloatArray,
) -> FloatArray:
    for _ in range(MAX_ATTEMPTS):
        low_x, low_y, high_x, high_y = config.agent_bounds
        position = rng.uniform((low_x, low_y), (high_x, high_y))
        if _agent_clear(position, block_pose, target_poses, config):
            return position.astype(np.float32)
    raise TwinTSamplingError(f"could not sample an agent in {MAX_ATTEMPTS} attempts")


def _agent_clear(
    agent_pos: NDArray[np.floating],
    block_pose: NDArray[np.floating],
    target_poses: NDArray[np.floating],
    config: TwinTConfig,
) -> bool:
    if not circle_inside_walls(agent_pos, config.agent_radius):
        return False
    clearance = config.agent_radius + AGENT_CLEARANCE
    obstacles = [tee_polygon(block_pose, config.block_scale)]
    obstacles.extend(tee_polygon(pose, config.block_scale) for pose in target_poses)
    point = Point(float(agent_pos[0]), float(agent_pos[1]))
    return all(point.distance(obstacle) >= clearance for obstacle in obstacles)

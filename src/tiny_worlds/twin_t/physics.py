"""The TwinT T geometry and pymunk physics.

Poses are PushT body poses: ``[x, y, angle]`` where the body origin sits at
``(x, y)`` and local vertices map to ``position + R(angle) * vertex``. The
local geometric centroid is not the origin, so a geometric center becomes a
pose by subtracting the rotated centroid.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pymunk
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .config import TwinTConfig
from .types import FloatArray

Point = tuple[float, float]
Geometry = Polygon | MultiPolygon

K_P: Final[float] = 100.0
K_V: Final[float] = 20.0
WALL_MIN: Final[float] = 5.0
WALL_MAX: Final[float] = 506.0
WALL_RADIUS: Final[float] = 2.0
AGENT_SPAWN: Final[tuple[float, float]] = (256.0, 400.0)
BLOCK_SPAWN: Final[tuple[float, float, float]] = (256.0, 300.0, 0.0)
BLOCK_MASS: Final[float] = 1.0
FRICTION: Final[float] = 1.0

_WALL_BOX = box(WALL_MIN, WALL_MIN, WALL_MAX, WALL_MAX)


def tee_shape_vertices(scale: float) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """Return the two local rectangles of the PushT T at ``scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    length = 4.0 * scale
    bar = ((-length / 2, scale), (length / 2, scale), (length / 2, 0.0), (-length / 2, 0.0))
    stem = ((-scale / 2, scale), (-scale / 2, length), (scale / 2, length), (scale / 2, scale))
    return bar, stem


def local_centroid(scale: float) -> FloatArray:
    """The area-weighted centroid of the T in local coordinates."""
    polygons = [Polygon(vertices) for vertices in tee_shape_vertices(scale)]
    centroid = unary_union(polygons).centroid
    return np.array(centroid.coords[0], dtype=np.float32)


def rotate(points: NDArray[np.floating], angle: float) -> FloatArray:
    """Rotate ``(..., 2)`` points counter-clockwise by ``angle`` radians."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rotation.T


def apply_pose(pose: NDArray[np.floating], points: NDArray[np.floating]) -> FloatArray:
    """Map local ``(..., 2)`` points through a body pose."""
    world = rotate(points, float(pose[2])) + np.asarray(pose[:2], dtype=np.float64)
    return world.astype(np.float32)


def pose_from_center(
    center: NDArray[np.floating], angle: float, scale: float
) -> FloatArray:
    """Convert a geometric center and angle to a T body pose."""
    offset = rotate(local_centroid(scale)[None, :], angle)[0]
    return np.array(
        (float(center[0]) - offset[0], float(center[1]) - offset[1], float(angle)),
        dtype=np.float32,
    )


def center_from_pose(pose: NDArray[np.floating], scale: float) -> FloatArray:
    """Return the geometric center of a T body pose."""
    offset = rotate(local_centroid(scale)[None, :], float(pose[2]))[0]
    return (np.asarray(pose[:2], dtype=np.float64) + offset).astype(np.float32)


def tee_polygons(pose: NDArray[np.floating], scale: float) -> tuple[Polygon, Polygon]:
    """Return the two world-space rectangles of a posed T."""
    polygons = []
    for vertices in tee_shape_vertices(scale):
        world = apply_pose(pose, np.array(vertices, dtype=np.float64))
        polygons.append(Polygon([tuple(vertex) for vertex in world]))
    return polygons[0], polygons[1]


def tee_polygon(pose: NDArray[np.floating], scale: float) -> Geometry:
    """Return the union of a posed T's two rectangles."""
    bar, stem = tee_polygons(pose, scale)
    return unary_union([bar, stem])


def coverage(
    block_pose: NDArray[np.floating],
    target_poses: NDArray[np.floating],
    scale: float,
) -> FloatArray:
    """Fraction of each target covered by the block, clipped to ``[0, 1]``."""
    block = tee_polygon(block_pose, scale)
    values = []
    for pose in target_poses:
        target = tee_polygon(pose, scale)
        values.append(block.intersection(target).area / target.area)
    return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)


def keypoints(pose: NDArray[np.floating], scale: float) -> FloatArray:
    """Return the eight block corners in PushT keypoint order."""
    bar, stem = tee_shape_vertices(scale)
    points = np.concatenate([bar, stem]).astype(np.float64)
    return apply_pose(pose, points)


def inside_walls(geometry: Geometry) -> bool:
    """Whether a polygon lies inside the 5..506 wall box."""
    return bool(_WALL_BOX.covers(geometry))


def circle_inside_walls(center: NDArray[np.floating], radius: float) -> bool:
    """Whether a circle lies inside the 5..506 wall box."""
    x, y = float(center[0]), float(center[1])
    return WALL_MIN <= x - radius and x + radius <= WALL_MAX and WALL_MIN <= y - radius and y + radius <= WALL_MAX


class PhysicsWorld:
    """One pymunk space with a kinematic agent and a dynamic T block."""

    def __init__(self, config: TwinTConfig) -> None:
        self._config = config
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.0
        self._add_walls()
        self.agent = self._add_agent()
        self.block, self.block_shapes = self._add_block()
        self.space.on_collision(0, 0, post_solve=self._count_contacts)
        self.contact_points = 0

    def _add_walls(self) -> None:
        corners = ((WALL_MIN, WALL_MAX), (WALL_MIN, WALL_MIN), (WALL_MAX, WALL_MIN), (WALL_MAX, WALL_MAX))
        edges = list(zip(corners, corners[1:] + corners[:1]))
        walls = [pymunk.Segment(self.space.static_body, first, second, WALL_RADIUS) for first, second in edges]
        for wall in walls:
            wall.friction = FRICTION
        self.space.add(*walls)

    def _add_agent(self) -> pymunk.Body:
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = AGENT_SPAWN
        shape = pymunk.Circle(body, self._config.agent_radius)
        shape.friction = FRICTION
        self.space.add(body, shape)
        return body

    def _add_block(self) -> tuple[pymunk.Body, list[pymunk.Poly]]:
        scale = self._config.block_scale
        bar, stem = tee_shape_vertices(scale)
        inertia = pymunk.moment_for_poly(BLOCK_MASS, vertices=bar) + pymunk.moment_for_poly(
            BLOCK_MASS, vertices=stem
        )
        body = pymunk.Body(BLOCK_MASS, inertia)
        shapes = [pymunk.Poly(body, bar), pymunk.Poly(body, stem)]
        for shape in shapes:
            shape.friction = FRICTION
        body.center_of_gravity = tuple(float(value) for value in local_centroid(scale))
        body.angle = BLOCK_SPAWN[2]
        body.position = BLOCK_SPAWN[:2]
        self.space.add(body, *shapes)
        return body, shapes

    def _count_contacts(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: object) -> None:
        self.contact_points += len(arbiter.contact_point_set.points)

    def set_bodies(self, agent_pos: NDArray[np.floating], block_pose: NDArray[np.floating]) -> None:
        """Place both bodies and settle one physics substep, as PushT does."""
        self.agent.velocity = (0.0, 0.0)
        self.agent.position = (float(agent_pos[0]), float(agent_pos[1]))
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        # The angle setter rotates about the center of gravity and moves the origin.
        self.block.angle = float(block_pose[2])
        self.block.position = (float(block_pose[0]), float(block_pose[1]))
        self.space.step(self._config.physics_dt)
        self.contact_points = 0

    def step_control(
        self, action: NDArray[np.floating], substeps: int, max_agent_speed: float
    ) -> None:
        """Run one PD-controlled control step."""
        target = pymunk.Vec2d(float(action[0]), float(action[1]))
        self.contact_points = 0
        for _ in range(substeps):
            self._substep(target, max_agent_speed)

    def _substep(self, target: pymunk.Vec2d, max_agent_speed: float) -> None:
        acceleration = K_P * (target - self.agent.position) + K_V * (
            pymunk.Vec2d(0.0, 0.0) - self.agent.velocity
        )
        velocity = self.agent.velocity + acceleration * self._config.physics_dt
        speed = velocity.length
        if speed > max_agent_speed:
            velocity = velocity * (max_agent_speed / speed)
        self.agent.velocity = velocity
        self.space.step(self._config.physics_dt)

    def pause(self) -> None:
        """Zero the agent velocity."""
        self.agent.velocity = (0.0, 0.0)

    @property
    def agent_pos(self) -> FloatArray:
        return np.array(self.agent.position, dtype=np.float32)

    @property
    def block_pose(self) -> FloatArray:
        return np.array((self.block.position.x, self.block.position.y, self.block.angle), dtype=np.float32)

    def n_contacts(self, substeps: int) -> int:
        """Contact points in the last control step, as PushT reports them."""
        return math.ceil(self.contact_points / substeps)

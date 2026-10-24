"""Friction Moving MNIST.

Two MNIST digits move with per-frame damping and lose energy at the walls.
Sequences stop when both digits are at rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .mnist import load_mnist

type Float32Array = NDArray[np.float32]
type Int8Array = NDArray[np.int8]
type Int32Array = NDArray[np.int32]
type UInt8Array = NDArray[np.uint8]


class FrictionMNISTConfig(NamedTuple):
    num_frames: int = 200
    canvas_size: int = 64
    digit_size: int = 28
    speed_range: tuple[float, float] = (1.5, 3.0)
    friction_range: tuple[float, float] = (0.96, 0.98)
    restitution_range: tuple[float, float] = (0.7, 0.95)
    stop_speed: float = 0.05
    train_fraction: float = 0.8
    val_fraction: float = 0.1


DEFAULT_CONFIG = FrictionMNISTConfig()


class FrictionMNIST(NamedTuple):
    """A labeled friction-Moving-MNIST pool."""

    frames: UInt8Array  # (N, T, H, W)
    digit_label: Int8Array  # (N, 2)
    digit_index: Int32Array  # (N, 2)
    position: Float32Array  # (N, T, 2, 2), top-left pixels
    velocity: Float32Array  # (N, T, 2, 2), pixels/frame
    friction: Float32Array  # (N, 2), per-frame damping
    restitution: Float32Array  # (N, 2), wall energy retention
    bounce_count: Int32Array  # (N,), wall bounces per sequence
    stop_time: Int32Array  # (N,), first frame with both digits at rest
    final_position: Float32Array  # (N, 2, 2)
    split: Int8Array  # (N,), 0=train, 1=val, 2=test


def _reflect(
    position: Float32Array,
    velocity: Float32Array,
    restitution: Float32Array,
    max_position: float,
) -> tuple[Float32Array, Float32Array]:
    """Reflect positions and damp velocities at the walls."""
    below = position < 0.0
    above = position > max_position
    hit = below | above
    position = np.where(below, -position, position)
    position = np.where(above, 2.0 * max_position - position, position)
    velocity = np.where(hit, -velocity * restitution, velocity)
    return position.astype(np.float32), velocity.astype(np.float32)


def _sample_motion(
    rng: np.random.Generator,
    config: FrictionMNISTConfig,
    num_samples: int,
) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Sample initial positions, velocities, friction, and restitution."""
    max_position = float(config.canvas_size - config.digit_size)
    position = rng.uniform(0.0, max_position, size=(num_samples, 2, 2)).astype(np.float32)
    speed = rng.uniform(*config.speed_range, size=(num_samples, 2, 1)).astype(np.float32)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples, 2, 1)).astype(np.float32)
    velocity = (np.concatenate([np.cos(angle), np.sin(angle)], axis=-1) * speed).astype(np.float32)
    friction = rng.uniform(*config.friction_range, size=(num_samples, 2, 1)).astype(np.float32)
    restitution = rng.uniform(*config.restitution_range, size=(num_samples, 2, 1)).astype(np.float32)
    return position, velocity, friction, restitution


def _simulate(
    rng: np.random.Generator,
    config: FrictionMNISTConfig,
    num_samples: int,
) -> tuple[
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
    Int32Array,
    Int32Array,
    Float32Array,
]:
    """Simulate the full sequences and record bounces and stop times."""
    max_position = float(config.canvas_size - config.digit_size)
    position, velocity, friction, restitution = _sample_motion(rng, config, num_samples)
    positions = np.empty((num_samples, config.num_frames, 2, 2), np.float32)
    velocities = np.empty_like(positions)
    stop_time = np.full(num_samples, -1, dtype=np.int32)

    for frame in range(config.num_frames):
        if frame > 0:
            active = np.linalg.norm(velocity, axis=-1, keepdims=True) > config.stop_speed
            velocity = np.where(active, velocity * friction, 0.0).astype(np.float32)
            position = np.where(active, position + velocity, position)
            position, velocity = _reflect(position, velocity, restitution, max_position)
            slowed = np.linalg.norm(velocity, axis=-1, keepdims=True) < config.stop_speed
            velocity = np.where(slowed, 0.0, velocity).astype(np.float32)

        positions[:, frame] = position
        velocities[:, frame] = velocity
        stopped = np.all(velocity == 0.0, axis=(1, 2))
        stop_time = np.where((stop_time < 0) & stopped, frame, stop_time)

    signs = np.sign(velocities)
    reversals = (signs[:, 1:] != signs[:, :-1]) & (signs[:, 1:] != 0.0) & (signs[:, :-1] != 0.0)
    bounce_count = reversals.sum(axis=(1, 2, 3)).astype(np.int32)
    stop_time = np.where(stop_time < 0, config.num_frames - 1, stop_time)
    final_position = positions[np.arange(num_samples), stop_time]
    return (
        positions,
        velocities,
        friction[..., 0],
        restitution[..., 0],
        bounce_count,
        stop_time,
        final_position,
    )


def _render(
    images: UInt8Array,
    digit_index: Int32Array,
    position: Float32Array,
    config: FrictionMNISTConfig,
) -> UInt8Array:
    """Render each digit at its rounded top-left position."""
    num_sequences, num_frames = position.shape[:2]
    frames = np.zeros(
        (num_sequences, num_frames, config.canvas_size, config.canvas_size),
        dtype=np.uint8,
    )
    patches = images[digit_index]
    xs = np.rint(position[..., 0]).astype(np.int32)
    ys = np.rint(position[..., 1]).astype(np.int32)
    max_xy = config.canvas_size - config.digit_size

    for sequence in range(num_sequences):
        for frame in range(num_frames):
            for digit in range(2):
                x = min(max(int(xs[sequence, frame, digit]), 0), max_xy)
                y = min(max(int(ys[sequence, frame, digit]), 0), max_xy)
                patch = patches[sequence, digit]
                window = frames[
                    sequence,
                    frame,
                    y : y + config.digit_size,
                    x : x + config.digit_size,
                ]
                frames[
                    sequence,
                    frame,
                    y : y + config.digit_size,
                    x : x + config.digit_size,
                ] = np.maximum(window, patch)
    return frames


def _split(
    num_samples: int,
    config: FrictionMNISTConfig,
) -> Int8Array:
    """Assign train/val/test ids in contiguous blocks."""
    num_train = int(num_samples * config.train_fraction)
    num_val = int(num_samples * config.val_fraction)
    split = np.zeros(num_samples, dtype=np.int8)
    split[num_train : num_train + num_val] = 1
    split[num_train + num_val :] = 2
    return split


def generate_dataset(
    seed: int = 0,
    num_samples: int = 2_000,
    *,
    mnist_dir: Path,
    config: FrictionMNISTConfig = DEFAULT_CONFIG,
) -> FrictionMNIST:
    """Generate a friction-Moving-MNIST pool in memory."""
    images, labels = load_mnist(mnist_dir)
    rng = np.random.default_rng(seed)
    digit_index = rng.integers(0, len(images), size=(num_samples, 2), dtype=np.int32)
    positions, velocities, friction, restitution, bounce_count, stop_time, final = _simulate(rng, config, num_samples)
    frames = _render(images, digit_index, positions, config)
    split = _split(num_samples, config)
    order = rng.permutation(num_samples)
    return FrictionMNIST(
        frames=frames[order],
        digit_label=labels[digit_index][order].astype(np.int8),
        digit_index=digit_index[order],
        position=positions[order],
        velocity=velocities[order],
        friction=friction[order],
        restitution=restitution[order],
        bounce_count=bounce_count[order],
        stop_time=stop_time[order],
        final_position=final[order],
        split=split[order],
    )

"""Tests for the friction Moving MNIST world."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from tiny_worlds.friction_mnist import FrictionMNISTConfig, generate_dataset

MNIST_DIR = Path(os.environ.get("TINY_WORLDS_MNIST_DIR", Path.home() / "data" / "mnist"))


def _config() -> FrictionMNISTConfig:
    return FrictionMNISTConfig(
        num_frames=60,
        speed_range=(1.0, 2.0),
        friction_range=(0.85, 0.90),
        restitution_range=(0.7, 0.9),
        stop_speed=0.05,
    )


def test_dataset_shapes_and_physics() -> None:
    data = generate_dataset(0, 5, mnist_dir=MNIST_DIR, config=_config())

    assert data.frames.shape == (5, 60, 64, 64)
    assert data.position.shape == (5, 60, 2, 2)
    assert data.velocity.shape == (5, 60, 2, 2)
    assert data.digit_label.shape == (5, 2)
    assert data.stop_time.shape == (5,)
    assert data.final_position.shape == (5, 2, 2)
    assert data.bounce_count.shape == (5,)

    assert np.all(data.position >= 0.0)
    assert np.all(data.position <= 36.0)
    assert np.all(data.friction >= 0.85)
    assert np.all(data.friction <= 0.90)
    assert np.all(data.restitution >= 0.7)
    assert np.all(data.restitution <= 0.9)
    assert np.all(data.stop_time >= 0)
    assert np.all(data.stop_time < 60)


def test_stop_time_matches_final_position() -> None:
    data = generate_dataset(1, 5, mnist_dir=MNIST_DIR, config=_config())

    for index, stop in enumerate(data.stop_time):
        np.testing.assert_allclose(
            data.final_position[index],
            data.position[index, stop],
        )
        np.testing.assert_allclose(
            data.velocity[index, stop],
            0.0,
        )
        assert np.all(data.frames[index, stop] == data.frames[index, -1])


def test_split_is_disjoint() -> None:
    data = generate_dataset(2, 20, mnist_dir=MNIST_DIR, config=_config())

    assert set(np.unique(data.split)) == {0, 1, 2}
    assert len(data.split) == 20

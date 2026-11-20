"""On-demand rendering from state: environment parity and the public helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tiny_worlds.twin_t.config import DEFAULT_CONFIG
from tiny_worlds.twin_t.env import TwinTEnv
from tiny_worlds.twin_t.render import render_frame, render_frames
from tiny_worlds.twin_t.schema import PINK

SIZE = DEFAULT_CONFIG.observation_size


def test_render_frames_match_the_environment_observation() -> None:
    env = TwinTEnv()
    try:
        obs, _ = env.reset(seed=0, command=PINK)
        poses = env.target_poses
        states = [env.state().as_vector()]
        pixels = [obs.pixels]
        for action in ([256.0, 300.0], [300.0, 320.0], [350.0, 350.0]):
            obs, _, _, _, _ = env.step(np.asarray(action, np.float32))
            states.append(env.state().as_vector())
            pixels.append(obs.pixels)
        rendered = render_frames(np.stack(states), poses, SIZE)
    finally:
        env.close()
    expected = np.stack(pixels)
    assert rendered.shape == expected.shape == (4, SIZE, SIZE, 3)
    assert rendered.dtype == np.uint8
    np.testing.assert_array_equal(rendered, expected)


def test_render_frame_matches_the_first_rendered_frame() -> None:
    env = TwinTEnv()
    try:
        obs, _ = env.reset(seed=2, command=PINK)
        state = env.state().as_vector()
        poses = env.target_poses
        single = render_frame(state, poses, SIZE)
    finally:
        env.close()
    np.testing.assert_array_equal(single, obs.pixels)
    assert single.shape == (SIZE, SIZE, 3)
    assert single.dtype == np.uint8


def test_render_frames_accepts_per_frame_poses() -> None:
    env = TwinTEnv()
    try:
        env.reset(seed=1, command=PINK)
        poses = env.target_poses
        states = np.stack([env.state().as_vector(), env.state().as_vector()])
    finally:
        env.close()
    broadcast = render_frames(states, poses, 32)
    stacked = render_frames(states, np.stack([poses, poses]), 32)
    np.testing.assert_array_equal(broadcast, stacked)


def test_render_frames_rejects_bad_shapes() -> None:
    states = np.zeros((2, 5), np.float32)
    poses = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError, match="states"):
        render_frames(np.zeros((2, 4), np.float32), poses, 32)
    with pytest.raises(ValueError, match="target_poses"):
        render_frames(states, np.zeros((3, 3), np.float32), 32)
    with pytest.raises(ValueError, match="size"):
        render_frames(states, poses, 0)

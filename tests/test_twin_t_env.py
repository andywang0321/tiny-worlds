"""Acceptance tests for the TwinT environment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point

from tiny_worlds.twin_t import DEFAULT_CONFIG, GREEN, PINK, TwinTConfig
from tiny_worlds.twin_t.env import TwinTEnv
from tiny_worlds.twin_t.physics import (
    center_from_pose,
    coverage,
    inside_walls,
    local_centroid,
    pose_from_center,
    tee_polygon,
)
from tiny_worlds.twin_t.sampling import TwinTSamplingError
from tiny_worlds.twin_t.schema import BLOCK_COLOR, TARGET_COLORS

SCALE = DEFAULT_CONFIG.block_scale


def _targets(
    centers: tuple[tuple[float, float], tuple[float, float]] = ((150.0, 150.0), (380.0, 380.0)),
    angles: tuple[float, float] = (0.0, math.pi / 2.0),
) -> np.ndarray:
    return np.stack([pose_from_center(np.array(center), angle, SCALE) for center, angle in zip(centers, angles)])


def _state(block_pose: np.ndarray, agent: tuple[float, float] = (480.0, 480.0)) -> np.ndarray:
    return np.concatenate([np.array(agent, np.float32), np.asarray(block_pose, np.float32)]).astype(np.float32)


def _in_bounds(point: np.ndarray, bounds: tuple[float, float, float, float]) -> bool:
    return bounds[0] - 1e-3 <= point[0] <= bounds[2] + 1e-3 and bounds[1] - 1e-3 <= point[1] <= bounds[3] + 1e-3


def test_reset_shapes_and_info_fields() -> None:
    env = TwinTEnv()
    obs, info = env.reset(seed=0, command=PINK)
    assert obs.pixels.shape == (DEFAULT_CONFIG.observation_size, DEFAULT_CONFIG.observation_size, 3)
    assert obs.pixels.dtype == np.uint8
    assert obs.agent_pos.shape == (2,)
    assert obs.agent_pos.dtype == np.float32
    assert set(obs.__dataclass_fields__) == {"pixels", "agent_pos"}
    assert info.block_pose.shape == (3,)
    assert info.block_keypoints.shape == (8, 2)
    assert info.target_poses.shape == (2, 3)
    assert info.coverage.shape == (2,)
    assert info.target_poses.dtype == np.float32
    assert info.coverage.dtype == np.float32
    assert info.step == 0
    assert info.n_contacts == 0
    assert info.achieved_target is None
    assert env.target_poses.shape == (2, 3)
    assert local_centroid(SCALE)[1] == pytest.approx(40.714, abs=1e-3)
    state = env.state()
    assert state.as_vector().shape == (5,)
    assert state.command.target == PINK
    assert state.step == 0
    np.testing.assert_array_equal(state.target_poses, info.target_poses)


def test_reset_is_deterministic_for_a_fixed_seed() -> None:
    first, second = TwinTEnv(), TwinTEnv()
    obs_a, info_a = first.reset(seed=11, command=GREEN)
    obs_b, info_b = second.reset(seed=11, command=GREEN)
    np.testing.assert_array_equal(obs_a.pixels, obs_b.pixels)
    np.testing.assert_array_equal(obs_a.agent_pos, obs_b.agent_pos)
    np.testing.assert_array_equal(info_a.target_poses, info_b.target_poses)
    np.testing.assert_array_equal(info_a.block_pose, info_b.block_pose)
    obs_c, info_c = first.reset(seed=11, command=GREEN)
    np.testing.assert_array_equal(obs_a.pixels, obs_c.pixels)
    np.testing.assert_array_equal(info_a.target_poses, info_c.target_poses)
    _, info_d = TwinTEnv().reset(seed=12, command=GREEN)
    assert not np.array_equal(info_a.target_poses, info_d.target_poses)


def test_two_hundred_resets_keep_every_shape_disjoint_and_inside() -> None:
    env = TwinTEnv()
    for seed in range(200):
        obs, info = env.reset(seed=seed)
        targets = env.target_poses
        centers = np.stack([center_from_pose(pose, SCALE) for pose in targets])
        assert np.linalg.norm(centers[0] - centers[1]) >= DEFAULT_CONFIG.target_min_separation - 1e-3
        assert _in_bounds(centers[0], DEFAULT_CONFIG.target_bounds)
        assert _in_bounds(centers[1], DEFAULT_CONFIG.target_bounds)
        assert _in_bounds(center_from_pose(info.block_pose, SCALE), DEFAULT_CONFIG.block_bounds)
        assert _in_bounds(obs.agent_pos, DEFAULT_CONFIG.agent_bounds)
        target_polys = [tee_polygon(pose, SCALE) for pose in targets]
        block_poly = tee_polygon(info.block_pose, SCALE)
        assert all(inside_walls(poly) for poly in [block_poly, *target_polys])
        assert target_polys[0].intersection(target_polys[1]).area == 0.0
        assert all(block_poly.intersection(poly).area == 0.0 for poly in target_polys)
        assert np.all(coverage(info.block_pose, targets, SCALE) < 0.05)
        agent = Point(*obs.agent_pos)
        clearance = DEFAULT_CONFIG.agent_radius + 0.5
        assert all(agent.distance(poly) >= clearance for poly in [block_poly, *target_polys])


def test_block_on_target_gives_full_coverage_and_rule_based_termination() -> None:
    targets = _targets()
    state = _state(targets[PINK])
    task = TwinTEnv(TwinTConfig(terminate_on="task"))
    obs, info = task.reset(state=state, target_poses=targets, command=PINK)
    np.testing.assert_allclose(info.coverage, [1.0, 0.0], atol=1e-6)
    assert info.task_success and info.command_success
    assert info.achieved_target == PINK
    _, reward, terminated, truncated, _ = task.step(obs.agent_pos)
    assert terminated and not truncated
    assert reward == pytest.approx(1.0)


def test_wrong_command_is_task_success_but_not_command_success() -> None:
    targets = _targets()
    state = _state(targets[PINK])
    command = TwinTEnv(TwinTConfig(terminate_on="command"))
    obs, info = command.reset(state=state, target_poses=targets, command=GREEN)
    assert info.task_success and not info.command_success
    assert info.achieved_target == PINK
    _, _, terminated, truncated, info = command.step(obs.agent_pos)
    assert not terminated and not truncated
    assert info.task_success and not info.command_success
    evaluation = TwinTEnv(TwinTConfig(terminate_on="task"))
    obs, info = evaluation.reset(state=state, target_poses=targets, command=GREEN)
    assert info.task_success and not info.command_success
    assert evaluation.step(obs.agent_pos)[2]


def test_null_command_terminates_on_task_success() -> None:
    targets = _targets()
    state = _state(targets[PINK])
    env = TwinTEnv(TwinTConfig(terminate_on="command"))
    obs, info = env.reset(state=state, target_poses=targets, command=None)
    assert info.task_success and not info.command_success
    _, reward, terminated, _, _ = env.step(obs.agent_pos)
    assert terminated and reward == pytest.approx(1.0)


def test_set_command_changes_success_and_termination() -> None:
    targets = _targets()
    state = _state(targets[PINK])
    env = TwinTEnv(TwinTConfig(terminate_on="command"))
    obs, _ = env.reset(state=state, target_poses=targets, command=GREEN)
    _, _, terminated, _, info = env.step(obs.agent_pos)
    assert not terminated and not info.command_success
    env.set_command(PINK)
    obs, _, terminated, _, info = env.step(obs.agent_pos)
    assert terminated and info.command_success
    env.set_command(GREEN)
    obs, _, terminated, _, info = env.step(obs.agent_pos)
    assert not terminated and not info.command_success


def test_far_pointer_respects_the_speed_cap() -> None:
    env = TwinTEnv()
    env.reset(seed=3)
    previous = env.state().agent_pos.copy()
    displacements = []
    for _ in range(20):
        env.step(np.array([512.0, 512.0]))
        current = env.state().agent_pos
        displacements.append(float(np.linalg.norm(current - previous)))
        previous = current.copy()
    cap = env.config.pixels_per_step
    assert max(displacements) <= cap + 1.0
    assert max(displacements) >= cap - 1.0


def test_pause_stops_the_agent_when_pointer_is_on_it() -> None:
    env = TwinTEnv()
    env.reset(seed=5)
    env.pause()
    env.step(np.array([512.0, 512.0]))
    env.pause()
    before = env.state().agent_pos.copy()
    env.pause()
    env.step(before)
    after = env.state().agent_pos
    assert np.linalg.norm(after - before) < 1e-3


def test_episode_limit_truncates() -> None:
    env = TwinTEnv(TwinTConfig(episode_limit=3))
    env.reset(seed=1)
    action = env.state().agent_pos.copy()
    results = []
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(action)
        results.append((terminated, truncated, info.step))
    assert results == [(False, False, 1), (False, False, 2), (False, True, 3)]


def test_render_and_observation_are_clean_frames() -> None:
    env = TwinTEnv()
    obs, _ = env.reset(seed=2, command=PINK)
    frame = env.render()
    assert frame.shape == (DEFAULT_CONFIG.render_size, DEFAULT_CONFIG.render_size, 3)
    assert frame.dtype == np.uint8
    assert env.render(128).shape == (128, 128, 3)
    for color in TARGET_COLORS:
        expected = np.array(color, dtype=np.uint8)
        assert np.any(np.all(frame == expected, axis=-1))
        assert np.any(np.all(obs.pixels == expected, axis=-1))
    assert np.any(np.all(obs.pixels == np.array(BLOCK_COLOR, dtype=np.uint8), axis=-1))
    np.testing.assert_array_equal(obs.pixels, env.render(DEFAULT_CONFIG.observation_size))
    assert not np.any(np.all(obs.pixels == np.array([255, 0, 0], dtype=np.uint8), axis=-1))


def test_step_rejects_bad_actions_and_clips_good_ones() -> None:
    env = TwinTEnv()
    env.reset(seed=4)
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0]))
    _, _, _, _, info = env.step(np.array([-100.0, 900.0]))
    assert info.step == 1


def test_explicit_state_allows_overlap_but_validates_input() -> None:
    targets = _targets()
    env = TwinTEnv()
    env.reset(state=_state(targets[GREEN]), target_poses=targets, command=GREEN)
    with pytest.raises(TwinTSamplingError):
        env.reset(state=np.zeros(4), target_poses=targets)
    with pytest.raises(TwinTSamplingError):
        env.reset(state=np.array([np.nan, 480.0, *targets[0]]), target_poses=targets)
    with pytest.raises(TwinTSamplingError):
        env.reset(state=np.array([-50.0, -50.0, *targets[0]]), target_poses=targets)
    with pytest.raises(TwinTSamplingError):
        env.reset(state=_state(targets[PINK]), target_poses=np.stack([targets[PINK], targets[PINK]]))


def test_missing_layout_parts_are_sampled_clear_of_fixed_parts() -> None:
    targets = _targets()
    env = TwinTEnv()
    _, info = env.reset(target_poses=targets)
    np.testing.assert_array_equal(info.target_poses, targets)
    assert np.all(info.coverage < 0.05)
    block = pose_from_center(np.array([256.0, 256.0]), 0.4, SCALE)
    _, info = env.reset(state=_state(block))
    assert np.all(info.coverage < 0.05)
    assert info.target_poses.shape == (2, 3)


def test_close_is_safe_and_idempotent() -> None:
    env = TwinTEnv()
    env.reset(seed=0)
    env.close()
    env.close()

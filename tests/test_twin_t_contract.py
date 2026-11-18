"""The frozen TwinT contract: config, schema, and types."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tiny_worlds.twin_t import (
    COLLECTOR_CONFIG,
    DEFAULT_CONFIG,
    GREEN,
    PINK,
    Command,
    TwinTConfig,
    TwinTObservation,
    TwinTState,
)
from tiny_worlds.twin_t.schema import (
    SCHEMA_VERSION,
    SPLIT_FRACTIONS,
    SPLIT_TRAIN,
    TARGET_COLORS,
    TARGET_NAMES,
    TRIAL_ARRAYS,
    ZARR_DATA_ARRAYS,
    ZARR_META_ARRAYS,
    target_index,
    target_name,
)


def test_default_config_matches_the_preregistration() -> None:
    config = DEFAULT_CONFIG
    assert config.workspace == 512
    assert config.control_hz == 10
    assert config.success_threshold == 0.95
    assert config.target_bounds == (90.0, 90.0, 422.0, 422.0)
    assert config.target_min_separation == 180.0
    assert config.target_angle_range == (-math.pi, math.pi)
    assert config.substeps == 10
    assert config.pixels_per_step == pytest.approx(25.0)
    assert config.terminate_on == "task"


def test_collector_config_terminates_on_the_command() -> None:
    assert COLLECTOR_CONFIG.terminate_on == "command"
    assert COLLECTOR_CONFIG.episode_limit == 600


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        TwinTConfig(success_threshold=1.5)
    with pytest.raises(ValueError):
        TwinTConfig(target_bounds=(422.0, 90.0, 90.0, 422.0))
    with pytest.raises(ValueError):
        TwinTConfig(target_angle_range=(0.0, 0.0))
    with pytest.raises(ValueError):
        TwinTConfig(terminate_on="wrong")  # type: ignore[arg-type]


def test_command_is_pink_green_or_null() -> None:
    assert Command(PINK).target == 0
    assert Command(GREEN).target == 1
    assert Command(None).is_null
    assert not Command(PINK).is_null
    with pytest.raises(ValueError):
        Command(2)


def test_state_vector_is_the_push_t_layout() -> None:
    state = TwinTState(
        agent_pos=np.array([1.0, 2.0], np.float32),
        block_pose=np.array([3.0, 4.0, 5.0], np.float32),
        target_poses=np.zeros((2, 3), np.float32),
        command=Command(PINK),
        step=0,
    )
    np.testing.assert_array_equal(state.as_vector(), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_observation_has_only_pixels_and_agent_position() -> None:
    observation = TwinTObservation(
        pixels=np.zeros((96, 96, 3), np.uint8),
        agent_pos=np.zeros(2, np.float32),
    )
    assert set(observation.__dataclass_fields__) == {"pixels", "agent_pos"}


def test_schema_constants() -> None:
    assert SCHEMA_VERSION == 1
    assert TARGET_NAMES == ("pink", "green")
    assert TARGET_COLORS[PINK] == (255, 79, 163)
    assert TARGET_COLORS[GREEN] == (34, 197, 94)
    assert SPLIT_FRACTIONS == (0.8, 0.1, 0.1)
    assert SPLIT_TRAIN == 0
    assert TRIAL_ARRAYS == ("obs", "state", "action", "coverage", "success", "keypoint", "n_contacts")
    assert "episode_ends" in ZARR_META_ARRAYS
    assert "img" in ZARR_DATA_ARRAYS


def test_target_name_round_trip() -> None:
    for index, name in enumerate(TARGET_NAMES):
        assert target_name(index) == name
        assert target_index(name) == index
    with pytest.raises(ValueError):
        target_name(2)
    with pytest.raises(ValueError):
        target_index("blue")

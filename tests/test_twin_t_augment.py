"""Tests for exact TwinT symmetry augmentation.

Render parity convention (verified empirically against the pygame renderer):

* ``mirror="x"`` (world ``x -> W - x``) is a flip of the image's *width* axis,
  i.e. ``img[:, ::-1]``; ``mirror="y"`` flips the height axis.
* ``rotation=k`` counter-clockwise in world coordinates is
  ``np.rot90(img, -k)`` (equivalently ``np.rot90(img, 4 - k)``) in the rendered
  array, whose row index is world ``y`` and column index is world ``x``.

The renderer rasterizes polygons at integer pixel coordinates and its static
wall inset is centred at 255.5 while the frozen transform uses ``W / 2 = 256``,
so pixel-exact equality at 96 px is impossible: object outlines move by up to
one pixel.  The test therefore checks (a) every per-color mask is within one
pixel of the transformed mask, (b) pixels whose 3x3 neighbourhood is flat in
both images match *exactly* (this is where the angle signs show), and (c) at
96 px at least 98% of raw pixels match after erasing the static walls.  At
512 px the raw wall-erased match rises above 99.5%, so the same test asserts
that tighter bound there.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pytest
from shapely.geometry import Point

from tiny_worlds.twin_t import (
    DEFAULT_CONFIG,
    GREEN,
    PINK,
    SCHEMA_VERSION,
    Symmetry,
    Trial,
    TrialMeta,
    augment_trial,
    load_trial,
    relocate_non_target,
    render_frames,
    save_trial,
    symmetries,
)
from tiny_worlds.twin_t.augment import MAX_NON_TARGET_COVERAGE
from tiny_worlds.twin_t.physics import center_from_pose, coverage, pose_from_center, tee_polygon
from tiny_worlds.twin_t.sampling import sample_layout
from tiny_worlds.twin_t.schema import AGENT_COLOR, BACKGROUND_COLOR, BLOCK_COLOR, TARGET_COLORS, WALL_COLOR
from tiny_worlds.twin_t.trials import validate_trial

CREATED: Final[str] = "2025-01-02T03:04:05Z"
SCALE: Final[float] = DEFAULT_CONFIG.block_scale
SIZE: Final[int] = DEFAULT_CONFIG.observation_size
OBJECT_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    TARGET_COLORS[0],
    TARGET_COLORS[1],
    BLOCK_COLOR,
    AGENT_COLOR,
)
D4: Final[tuple[Symmetry, ...]] = tuple(symmetry for symmetry in symmetries() if not symmetry.swap_colors)


def _layout(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    layout = sample_layout(np.random.default_rng(seed), DEFAULT_CONFIG)
    state = np.concatenate([layout.agent_pos, layout.block_pose]).astype(np.float32)
    return state, layout.target_poses


def _overlapping_layout(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A valid layout whose block partially covers the pink target."""
    layout = sample_layout(np.random.default_rng(seed), DEFAULT_CONFIG)
    target = layout.target_poses[0]
    center = center_from_pose(target, SCALE)
    block_pose = pose_from_center(center + np.array([0.0, -25.0]), float(target[2]) + 0.4, SCALE)
    state = np.concatenate([layout.agent_pos, block_pose]).astype(np.float32)
    return state, layout.target_poses


def _trial(seed: int = 0, steps: int = 3, target: int = PINK) -> Trial:
    base, target_poses = _layout(seed)
    rng = np.random.default_rng(seed + 10_000)
    rows = [base]
    for _ in range(steps):
        row = rows[-1].copy()
        row[:2] += rng.uniform(-4.0, 4.0, size=2)
        row[2:] += rng.uniform(-0.04, 0.04, size=3)
        rows.append(row.astype(np.float32))
    return Trial(
        state=np.stack(rows).astype(np.float32),
        target=np.array(target, dtype=np.int8),
        target_poses=target_poses,
        meta=TrialMeta(SCHEMA_VERSION, seed, CREATED, steps / 10.0),
    )


def test_symmetries_are_the_sixteen_distinct_transforms() -> None:
    values = symmetries()
    assert len(values) == 16
    assert len({symmetry.name for symmetry in values}) == 16
    assert values[0] == Symmetry()
    assert values[0].name == "identity"
    assert [symmetry.name for symmetry in values[:8]] == [
        "identity",
        "r1",
        "r2",
        "r3",
        "mx",
        "r1_mx",
        "r2_mx",
        "r3_mx",
    ]
    assert all(symmetry.swap_colors for symmetry in values[8:])
    expected = {(rotation, mirror, swap) for swap in (False, True) for mirror in (None, "x") for rotation in range(4)}
    assert {(symmetry.rotation, symmetry.mirror, symmetry.swap_colors) for symmetry in values} == expected


def test_symmetry_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError):
        Symmetry(rotation=4)
    with pytest.raises(ValueError):
        Symmetry(rotation=-1)
    with pytest.raises(TypeError):
        Symmetry(rotation=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Symmetry(mirror="z")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Symmetry(swap_colors=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [0, 6, 29])
def test_d4_preserves_geometry_and_coverage(seed: int) -> None:
    state, poses = _overlapping_layout(seed)
    block_before = state[2:5]
    coverage_before = coverage(block_before, poses, SCALE)
    assert 0.0 < float(np.max(coverage_before)) < 1.0
    agent_block = float(np.linalg.norm(state[:2] - state[2:4]))
    target_separation = float(np.linalg.norm(poses[0, :2] - poses[1, :2]))

    for symmetry in D4 + tuple(symmetry for symmetry in symmetries() if symmetry.swap_colors):
        transformed, target, transformed_poses = symmetry.apply(state, np.array(PINK, np.int8), poses)
        assert transformed.dtype == np.float32 and transformed.shape == state.shape
        assert transformed_poses.dtype == np.float32 and transformed_poses.shape == poses.shape
        assert np.all(transformed[4] >= -np.pi) and np.all(transformed[4] < np.pi)
        assert np.all(transformed_poses[:, 2] >= -np.pi) and np.all(transformed_poses[:, 2] < np.pi)

        np.testing.assert_allclose(
            np.linalg.norm(transformed[:2] - transformed[2:4]), agent_block, rtol=1e-5, atol=1e-4
        )
        np.testing.assert_allclose(
            np.linalg.norm(transformed_poses[0, :2] - transformed_poses[1, :2]),
            target_separation,
            rtol=1e-5,
            atol=1e-4,
        )
        after = coverage(transformed[2:5], transformed_poses, SCALE)
        if symmetry.swap_colors:
            np.testing.assert_allclose(after, coverage_before[::-1], atol=1e-5)
            assert int(target) == GREEN
        else:
            np.testing.assert_allclose(after, coverage_before, atol=1e-5)
            assert int(target) == PINK


def test_mirror_y_equals_rotation_two_then_mirror_x() -> None:
    state, poses = _overlapping_layout(1)
    direct = Symmetry(mirror="y").apply(state, np.array(PINK, np.int8), poses)
    composed = Symmetry(rotation=2, mirror="x").apply(state, np.array(PINK, np.int8), poses)
    for actual, expected in zip(direct, composed, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def _erase_walls(image: np.ndarray) -> np.ndarray:
    erased = image.copy()
    wall = np.all(erased == np.array(WALL_COLOR, np.uint8), axis=-1)
    erased[wall] = BACKGROUND_COLOR
    return erased


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    grown = np.zeros_like(mask)
    for row in range(3):
        for column in range(3):
            grown |= padded[row : row + mask.shape[0], column : column + mask.shape[1]]
    return grown


def _flat_interior(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    flat = np.ones(image.shape[:2], dtype=bool)
    for row in range(3):
        for column in range(3):
            neighbourhood = padded[row : row + image.shape[0], column : column + image.shape[1]]
            flat &= np.all(neighbourhood == image, axis=-1)
    return flat


def _expected_image(image: np.ndarray, symmetry: Symmetry) -> np.ndarray:
    expected = image
    if symmetry.mirror == "x":
        expected = np.flip(expected, axis=1)
    elif symmetry.mirror == "y":
        expected = np.flip(expected, axis=0)
    if symmetry.rotation:
        expected = np.rot90(expected, -symmetry.rotation, axes=(0, 1))
    return np.ascontiguousarray(expected)


def _assert_rasterization_parity(actual: np.ndarray, expected: np.ndarray, minimum_equal: float) -> None:
    equal = float(np.mean(actual == expected))
    assert equal >= minimum_equal
    for color in OBJECT_COLORS:
        actual_mask = np.all(actual == np.array(color, np.uint8), axis=-1)
        expected_mask = np.all(expected == np.array(color, np.uint8), axis=-1)
        assert not np.any(actual_mask & ~_dilate(expected_mask))
        assert not np.any(expected_mask & ~_dilate(actual_mask))
    flat = _flat_interior(actual) & _flat_interior(expected)
    assert flat.sum() >= 0.85 * actual.shape[0] * actual.shape[1]
    np.testing.assert_array_equal(actual[flat], expected[flat])


@pytest.mark.parametrize("seed", [0, 6, 29])
def test_render_parity_verifies_the_angle_signs(seed: int) -> None:
    state, poses = _layout(seed)
    original = _erase_walls(render_frames(state[None, :], poses, SIZE)[0])
    for symmetry in D4:
        if symmetry.name == "identity":
            continue
        transformed, _, transformed_poses = symmetry.apply(state, np.array(PINK, np.int8), poses)
        actual = _erase_walls(render_frames(transformed[None, :], transformed_poses, SIZE)[0])
        expected = _expected_image(original, symmetry)
        _assert_rasterization_parity(actual, expected, minimum_equal=0.98)

        # The unverified direction (opposite rot90 or the wrong flip axis) must fail.
        # A half turn is its own inverse, so there is no wrong rot90 sign for r2.
        if symmetry.rotation == 2:
            wrong = None
        elif symmetry.rotation == 0:
            wrong = np.flip(original, axis=0)
        elif symmetry.mirror == "x":
            wrong = np.rot90(np.flip(original, axis=1), symmetry.rotation, axes=(0, 1))
        else:
            wrong = np.rot90(original, symmetry.rotation, axes=(0, 1))
        if wrong is not None:
            assert float(np.mean(actual == wrong)) < 0.95


def test_render_parity_meets_the_wall_erased_fraction_at_512() -> None:
    state, poses = _layout(3)
    original = _erase_walls(render_frames(state[None, :], poses, 512)[0])
    for symmetry in D4:
        if symmetry.name == "identity":
            continue
        transformed, _, transformed_poses = symmetry.apply(state, np.array(PINK, np.int8), poses)
        actual = _erase_walls(render_frames(transformed[None, :], transformed_poses, 512)[0])
        expected = _expected_image(original, symmetry)
        assert float(np.mean(actual == expected)) >= 0.995
        _assert_rasterization_parity(actual, expected, minimum_equal=0.995)


def _swap_target_colors(image: np.ndarray) -> np.ndarray:
    swapped = image.copy()
    pink = np.all(swapped == np.array(TARGET_COLORS[0], np.uint8), axis=-1)
    green = np.all(swapped == np.array(TARGET_COLORS[1], np.uint8), axis=-1)
    swapped[pink] = TARGET_COLORS[1]
    swapped[green] = TARGET_COLORS[0]
    return swapped


def test_swap_colors_only_recolors_the_targets() -> None:
    state, poses = _layout(4)
    for symmetry in D4:
        recolored_symmetry = Symmetry(symmetry.rotation, symmetry.mirror, True)
        plain, _, plain_poses = symmetry.apply(state, np.array(PINK, np.int8), poses)
        swapped, target, swapped_poses = recolored_symmetry.apply(state, np.array(PINK, np.int8), poses)
        assert int(target) == GREEN
        np.testing.assert_array_equal(swapped_poses, plain_poses[::-1])
        plain_image = render_frames(plain[None, :], plain_poses, SIZE)[0]
        swapped_image = render_frames(swapped[None, :], swapped_poses, SIZE)[0]
        np.testing.assert_array_equal(swapped_image, _swap_target_colors(plain_image))


@pytest.mark.parametrize("symmetry", symmetries(), ids=lambda symmetry: symmetry.name)
def test_augment_trial_is_a_valid_v2_trial(symmetry: Symmetry, tmp_path: Path) -> None:
    trial = _trial(seed=3, steps=4, target=PINK)
    augmented = augment_trial(trial, symmetry)
    assert augmented.meta is trial.meta
    validate_trial(augmented)
    assert augmented.steps == trial.steps
    expected_state, expected_target, expected_poses = symmetry.apply(trial.state, trial.target, trial.target_poses)
    np.testing.assert_array_equal(augmented.state, expected_state)
    np.testing.assert_array_equal(augmented.target, expected_target)
    np.testing.assert_array_equal(augmented.target_poses, expected_poses)

    action = augmented.action
    assert action.shape == (trial.steps + 1, 2)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action[:-1], augmented.state[1:, :2])
    np.testing.assert_array_equal(action[-1], augmented.state[-1, :2])

    path = save_trial(tmp_path, augmented, f"aug_{symmetry.name}")
    loaded = load_trial(path)
    np.testing.assert_array_equal(loaded.state, augmented.state)
    np.testing.assert_array_equal(loaded.target, augmented.target)
    np.testing.assert_array_equal(loaded.target_poses, augmented.target_poses)
    assert loaded.meta == augmented.meta


def _assert_candidate_is_valid(
    state: np.ndarray,
    target: int,
    target_poses: np.ndarray,
    config=DEFAULT_CONFIG,
) -> np.ndarray:
    other = 1 - target
    candidate = target_poses[other]
    center = center_from_pose(candidate, config.block_scale)
    low_x, low_y, high_x, high_y = config.target_bounds
    assert low_x <= center[0] <= high_x
    assert low_y <= center[1] <= high_y
    low_angle, high_angle = config.target_angle_range
    assert low_angle <= candidate[2] < high_angle
    commanded_center = center_from_pose(target_poses[target], config.block_scale)
    assert float(np.linalg.norm(center - commanded_center)) >= config.target_min_separation
    candidate_polygon = tee_polygon(candidate, config.block_scale)
    assert candidate_polygon.intersection(tee_polygon(target_poses[target], config.block_scale)).area == 0.0
    for block_pose in state.reshape(-1, 5)[:, 2:5]:
        covered = float(coverage(block_pose, candidate[None, :], config.block_scale)[0])
        assert covered < MAX_NON_TARGET_COVERAGE
    return candidate


@pytest.mark.parametrize("target", [PINK, GREEN])
def test_relocate_non_target_preserves_the_commanded_scene(target: int) -> None:
    trial = _trial(seed=5, steps=3, target=target)
    state, poses = trial.state, trial.target_poses
    relocated_state, relocated_target, relocated_poses = relocate_non_target(
        state, np.array(target, np.int8), poses, np.random.default_rng(123)
    )
    np.testing.assert_array_equal(relocated_state, state)
    assert int(relocated_target) == target
    np.testing.assert_array_equal(relocated_poses[target], poses[target])
    assert not np.array_equal(relocated_poses[1 - target], poses[1 - target])
    _assert_candidate_is_valid(state, target, relocated_poses)
    # The agent must stay clear of the relocated target too.
    candidate_polygon = tee_polygon(relocated_poses[1 - target], SCALE)
    for agent_position in state[:, :2]:
        assert Point(*map(float, agent_position)).distance(candidate_polygon) >= DEFAULT_CONFIG.agent_radius


def test_relocate_non_target_is_deterministic() -> None:
    trial = _trial(seed=7, steps=2)
    first = relocate_non_target(trial.state, trial.target, trial.target_poses, np.random.default_rng(99))
    second = relocate_non_target(trial.state, trial.target, trial.target_poses, np.random.default_rng(99))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_relocate_non_target_raises_when_impossible() -> None:
    trial = _trial(seed=8, steps=1)
    impossible = dataclasses.replace(DEFAULT_CONFIG, target_min_separation=1.0e9)
    with pytest.raises(ValueError, match="relocate"):
        relocate_non_target(
            trial.state,
            trial.target,
            trial.target_poses,
            np.random.default_rng(0),
            config=impossible,
            attempts=8,
        )
    with pytest.raises(ValueError):
        relocate_non_target(trial.state, trial.target, trial.target_poses, np.random.default_rng(0), attempts=0)


def test_apply_is_pure_and_deterministic() -> None:
    state, poses = _overlapping_layout(9)
    state_copy = state.copy()
    poses_copy = poses.copy()
    for symmetry in symmetries():
        first = symmetry.apply(state, np.array(PINK, np.int8), poses)
        second = symmetry.apply(state, np.array(PINK, np.int8), poses)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        repeated_left = symmetry.apply(*first)
        repeated_right = symmetry.apply(*second)
        for left, right in zip(repeated_left, repeated_right, strict=True):
            np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(state, state_copy)
    np.testing.assert_array_equal(poses, poses_copy)


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "augment_trials.py"
    spec = importlib.util.spec_from_file_location("augment_trials", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_augments_a_synthetic_trial_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    for target, episode_id, seed in ((PINK, "p00", 0), (GREEN, "g00", 1)):
        save_trial(data, _trial(seed=seed, steps=2, target=target), episode_id)
    out = tmp_path / "aug"
    cli = _load_cli()
    assert cli.main(["--data", str(data), "--out", str(out), "--symmetries", "identity,swap"]) == 0
    output = capsys.readouterr().out
    assert "read 2 trials" in output
    assert "wrote 4 trials" in output
    assert (out / "pink" / "p00_identity.npz").is_file()
    assert (out / "green" / "p00_swap.npz").is_file()
    assert (out / "green" / "g00_identity.npz").is_file()
    assert (out / "pink" / "g00_swap.npz").is_file()
    loaded = load_trial(out / "green" / "p00_swap.npz")
    expected = augment_trial(load_trial(data / "pink" / "p00.npz"), Symmetry(swap_colors=True))
    np.testing.assert_array_equal(loaded.state, expected.state)
    np.testing.assert_array_equal(loaded.target_poses, expected.target_poses)
    assert int(loaded.target) == GREEN

    all_out = tmp_path / "all"
    assert cli.main(["--data", str(data), "--out", str(all_out), "--symmetries", "all"]) == 0
    assert len(list((all_out / "pink").glob("*.npz"))) + len(list((all_out / "green").glob("*.npz"))) == 32


def test_cli_rejects_unknown_symmetry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    save_trial(data, _trial(seed=0, steps=1), "p00")
    with pytest.raises(SystemExit):
        _load_cli().main(["--data", str(data), "--out", str(tmp_path / "out"), "--symmetries", "nope"])

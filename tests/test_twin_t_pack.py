"""Packed TwinT ``.npz`` format: packer, root metadata, splits, and derived arrays."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tiny_worlds.twin_t.config import DEFAULT_CONFIG
from tiny_worlds.twin_t.dataset import load_dataset
from tiny_worlds.twin_t.pack import PackError, PackSummary, pack_trials
from tiny_worlds.twin_t.schema import (
    FPS,
    GREEN,
    PACKED_ARRAYS,
    PINK,
    SCHEMA_VERSION,
    SPLIT_FRACTIONS,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SUCCESS_THRESHOLD,
    TARGET_COLORS,
)
from tiny_worlds.twin_t.trials import Trial, TrialInvalid, TrialMeta, derive_actions, save_trial

CREATED = "2025-01-02T03:04:05Z"
CREATED_TS = datetime.fromisoformat(CREATED).timestamp()


def make_trial(steps: int, target: int, *, seed: int) -> Trial:
    rng = np.random.default_rng(seed)
    return Trial(
        state=rng.uniform(0, 512, size=(steps + 1, 5)).astype(np.float32),
        target=np.array(target, np.int8),
        target_poses=rng.uniform(0, 512, size=(2, 3)).astype(np.float32),
        meta=TrialMeta(SCHEMA_VERSION, seed, CREATED, steps / FPS),
    )


def write_trial(data_dir: Path, target: int, episode_id: str, steps: int, seed: int) -> Trial:
    trial = make_trial(steps, target, seed=seed)
    save_trial(data_dir, trial, episode_id)
    return trial


def concatenate(trials: list[Trial], name: str) -> np.ndarray:
    return np.concatenate([getattr(trial, name) for trial in trials])


@pytest.fixture()
def packed(tmp_path: Path) -> tuple[Path, Path, list[Trial], PackSummary]:
    data_dir = tmp_path / "data"
    trials = [write_trial(data_dir, PINK, f"p{index:02d}", 2 + index % 4, index) for index in range(10)]
    trials += [write_trial(data_dir, GREEN, f"g{index:02d}", 2 + index % 4, 100 + index) for index in range(10)]
    out_path = tmp_path / "out" / "twint.npz"
    summary = pack_trials(data_dir, out_path)
    return out_path, data_dir, trials, summary


def test_pack_load_round_trip(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    out_path, _, trials, summary = packed
    assert summary.n_episodes == 20
    assert summary.n_frames == sum(trial.steps + 1 for trial in trials)
    assert summary.target_counts == {"pink": 10, "green": 10}
    assert summary.split_counts == {"train": 18, "validation": 2}
    assert summary.warnings == ()
    assert summary.out_path == out_path
    assert summary.sha256 == hashlib.sha256(out_path.read_bytes()).hexdigest()

    with load_dataset(out_path) as dataset:
        assert dataset.episode_count == 20
        assert dataset.frame_count == summary.n_frames
        np.testing.assert_array_equal(dataset.state, concatenate(trials, "state"))
        np.testing.assert_array_equal(dataset.action, _all_actions(trials))
        np.testing.assert_array_equal(dataset.episode_ends, np.cumsum([trial.steps + 1 for trial in trials]))
        np.testing.assert_array_equal(dataset.episode_target, [int(trial.target) for trial in trials])
        np.testing.assert_array_equal(dataset.episode_target_poses, [trial.target_poses for trial in trials])
        expected_target = np.concatenate([np.full(trial.steps + 1, int(trial.target), np.int8) for trial in trials])
        expected_poses = np.concatenate(
            [np.broadcast_to(trial.target_poses, (trial.steps + 1, 2, 3)) for trial in trials]
        )
        np.testing.assert_array_equal(dataset.target, expected_target)
        np.testing.assert_array_equal(dataset.target_poses, expected_poses)
        stored_ids = [value.rstrip(b"\x00") for value in dataset.episode_ids.tolist()]
        expected_ids = [f"p{index:02d}".encode() for index in range(10)]
        expected_ids += [f"g{index:02d}".encode() for index in range(10)]
        assert stored_ids == expected_ids
        np.testing.assert_allclose(dataset.episode_created, CREATED_TS)
        np.testing.assert_array_equal(dataset.episode_seed, [trial.meta.seed for trial in trials])


def _all_actions(trials: list[Trial]) -> np.ndarray:
    return np.concatenate([derive_actions(trial.state) for trial in trials])


def test_packed_file_has_only_the_v2_members(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    out_path = packed[0]
    with np.load(out_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(PACKED_ARRAYS)
        assert archive["state"].shape == (packed[3].n_frames, 5)
        assert archive["state"].dtype == np.float32
        assert archive["episode_ends"].dtype == np.int64
        assert archive["episode_target"].dtype == np.int8
        assert archive["episode_target_poses"].shape == (20, 2, 3)
        assert archive["episode_ids"].dtype == np.dtype("S32")
        assert archive["episode_created"].dtype == np.float64
        assert archive["episode_seed"].dtype == np.int64
        assert archive["split"].dtype == np.int8
        assert archive["meta"].shape == ()
        assert archive["meta"].dtype.kind == "U"
        for removed in ("obs", "action", "coverage", "success", "keypoint", "n_contacts"):
            assert removed not in archive.files


def test_root_metadata(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    out_path, data_dir, _, summary = packed
    with load_dataset(out_path) as dataset:
        attrs = dataset.metadata
    required = {
        "schema_version",
        "fps",
        "target_colors",
        "target_bounds",
        "target_min_separation",
        "success_threshold",
        "created_at",
        "source",
        "split_seed",
        "n_episodes",
        "n_frames",
        "episode_counts",
    }
    assert set(attrs) == required
    assert attrs["schema_version"] == SCHEMA_VERSION
    assert attrs["fps"] == FPS
    assert attrs["target_colors"] == [list(color) for color in TARGET_COLORS]
    assert attrs["target_bounds"] == list(DEFAULT_CONFIG.target_bounds)
    assert attrs["target_min_separation"] == DEFAULT_CONFIG.target_min_separation
    assert attrs["success_threshold"] == SUCCESS_THRESHOLD
    assert "observation_size" not in attrs
    assert attrs["source"] == str(data_dir)
    assert attrs["split_seed"] == 0
    assert attrs["n_episodes"] == summary.n_episodes
    assert attrs["n_frames"] == summary.n_frames
    assert attrs["episode_counts"] == {"pink": 10, "green": 10}
    assert attrs["created_at"].endswith("Z")
    datetime.fromisoformat(attrs["created_at"])
    json.dumps(attrs)


def test_the_packed_file_is_the_only_output(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    out_path = packed[0]
    out_dir = out_path.parent
    assert [entry.name for entry in out_dir.iterdir()] == [out_path.name]
    assert not any(entry.name.endswith(".tmp") for entry in out_dir.rglob("*"))


def test_splits_are_stratified_ninety_ten(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    out_path = packed[0]
    with load_dataset(out_path) as dataset:
        split = np.array(dataset.split)
        targets = np.array(dataset.episode_target)
    assert set(split.tolist()) == {SPLIT_TRAIN, SPLIT_VALIDATION}
    for color in (PINK, GREEN):
        codes, counts = np.unique(split[targets == color], return_counts=True)
        assert dict(zip(codes.tolist(), counts.tolist(), strict=True)) == {SPLIT_TRAIN: 9, SPLIT_VALIDATION: 1}


def test_split_counts_for_seventy_trials_per_colour(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for index in range(70):
        write_trial(data_dir, PINK, f"p{index:03d}", 1, index)
        write_trial(data_dir, GREEN, f"g{index:03d}", 1, 1000 + index)
    summary = pack_trials(data_dir, tmp_path / "twint.npz")
    assert summary.split_counts == {"train": 126, "validation": 14}
    with load_dataset(summary.out_path) as dataset:
        split = np.array(dataset.split)
        targets = np.array(dataset.episode_target)
    for color in (PINK, GREEN):
        counts = np.bincount(split[targets == color], minlength=2)
        assert counts.tolist() == [63, 7]
    assert int(split.size * SPLIT_FRACTIONS[SPLIT_TRAIN]) == 126


def test_splits_are_deterministic_for_a_seed(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    _, data_dir, _, _ = packed
    out_dir = data_dir.parent
    first = pack_trials(data_dir, out_dir / "split_a.npz", split_seed=3)
    second = pack_trials(data_dir, out_dir / "split_b.npz", split_seed=3)
    other = pack_trials(data_dir, out_dir / "split_c.npz", split_seed=4)
    assert first.split_counts == second.split_counts
    np.testing.assert_array_equal(_split_of(first.out_path), _split_of(second.out_path))
    assert not np.array_equal(_split_of(first.out_path), _split_of(other.out_path))


def _split_of(path: Path) -> np.ndarray:
    with load_dataset(path) as dataset:
        return np.array(dataset.split)


def test_overwrite_false_refuses_an_existing_output(
    packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    _, data_dir, _, _ = packed
    out_path = tmp_path / "existing.npz"
    out_path.write_bytes(b"keep")
    with pytest.raises(PackError, match="exists"):
        pack_trials(data_dir, out_path)
    assert out_path.read_bytes() == b"keep"
    pack_trials(data_dir, out_path, overwrite=True)
    assert out_path.stat().st_size > len(b"keep")


def test_tiny_data_warns_about_empty_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_trial(data_dir, PINK, "only", 2, 0)
    summary = pack_trials(data_dir, tmp_path / "tiny.npz")
    assert summary.split_counts == {"train": 0, "validation": 1}
    assert any("pink" in warning and "train" in warning for warning in summary.warnings)


def test_invalid_trial_fails_closed_without_output(
    packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    _, data_dir, _, _ = packed
    bad_path = data_dir / "pink" / "p00.npz"
    with np.load(bad_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    payload["state"][0, 0] = np.nan
    np.savez(bad_path, **payload)
    out_path = tmp_path / "failed" / "twint.npz"
    with pytest.raises(TrialInvalid, match="p00.npz"):
        pack_trials(data_dir, out_path)
    assert not out_path.exists()


def test_episode_id_longer_than_32_bytes_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = save_trial(data_dir, make_trial(2, PINK, seed=0), "short")
    path.rename(path.with_name("x" * 33 + ".npz"))
    out_path = tmp_path / "long.npz"
    with pytest.raises(PackError, match="32"):
        pack_trials(data_dir, out_path)
    assert not out_path.exists()


def test_pack_rejects_an_empty_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(PackError, match="no trials"):
        pack_trials(data_dir, tmp_path / "empty.npz")


def test_out_path_must_be_an_npz(packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    _, data_dir, _, _ = packed
    with pytest.raises(PackError, match="npz"):
        pack_trials(data_dir, tmp_path / "store.zarr")

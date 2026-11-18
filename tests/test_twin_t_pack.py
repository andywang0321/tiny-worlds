"""Packed zarr format: streaming packer, stratification, and zip layout."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from tiny_worlds.twin_t.config import DEFAULT_CONFIG
from tiny_worlds.twin_t.dataset import load_dataset
from tiny_worlds.twin_t.pack import PackError, PackSummary, pack_trials
from tiny_worlds.twin_t.schema import (
    FPS,
    GREEN,
    OBSERVATION_SIZE,
    PINK,
    SCHEMA_VERSION,
    SUCCESS_THRESHOLD,
    TARGET_COLORS,
)
from tiny_worlds.twin_t.trials import Trial, TrialInvalid, TrialMeta, save_trial

CREATED = "2025-01-02T03:04:05Z"
CREATED_TS = datetime.fromisoformat(CREATED).timestamp()


def make_trial(steps: int, target: int, *, seed: int) -> Trial:
    rng = np.random.default_rng(seed)
    n_frames = steps + 1
    action = rng.uniform(0, 512, size=(n_frames, 2)).astype(np.float32)
    action[-1] = action[-2]
    success = np.zeros(n_frames, dtype=np.bool_)
    success[-1] = True
    return Trial(
        obs=rng.integers(0, 256, size=(n_frames, 96, 96, 3), dtype=np.uint8),
        state=rng.normal(size=(n_frames, 5)).astype(np.float32),
        action=action,
        coverage=rng.uniform(0, 1, size=(n_frames, 2)).astype(np.float32),
        success=success,
        keypoint=rng.normal(size=(n_frames, 9, 2)).astype(np.float32),
        n_contacts=rng.integers(0, 4, size=(n_frames, 1), dtype=np.int32),
        target=np.array(target, np.int8),
        target_poses=rng.normal(size=(2, 3)).astype(np.float32),
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
    zip_path = tmp_path / "out" / "twint.zarr.zip"
    summary = pack_trials(data_dir, zip_path)
    return zip_path, data_dir, trials, summary


def split_of(zip_path: Path) -> np.ndarray:
    with load_dataset(zip_path) as dataset:
        return np.array(dataset.split[:])


def test_pack_load_round_trip(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    zip_path, _, trials, summary = packed
    assert summary.n_episodes == 20
    assert summary.n_frames == sum(trial.steps + 1 for trial in trials)
    assert summary.target_counts == {"pink": 10, "green": 10}
    assert summary.split_counts == {"train": 16, "validation": 2, "test": 2}
    assert summary.warnings == ()
    assert summary.out_path == zip_path
    assert summary.sha256 == hashlib.sha256(zip_path.read_bytes()).hexdigest()

    with load_dataset(zip_path) as dataset:
        assert dataset.episode_count == 20
        assert dataset.frame_count == summary.n_frames
        np.testing.assert_array_equal(dataset.img[:], concatenate(trials, "obs"))
        np.testing.assert_array_equal(dataset.state[:], concatenate(trials, "state"))
        np.testing.assert_array_equal(dataset.action[:], concatenate(trials, "action"))
        np.testing.assert_array_equal(dataset.coverage[:], concatenate(trials, "coverage"))
        np.testing.assert_array_equal(dataset.success[:], concatenate(trials, "success"))
        np.testing.assert_array_equal(dataset.keypoint[:], concatenate(trials, "keypoint"))
        np.testing.assert_array_equal(dataset.n_contacts[:], concatenate(trials, "n_contacts"))
        expected_target = np.concatenate(
            [np.full(trial.steps + 1, int(trial.target), np.int8) for trial in trials]
        )
        expected_poses = np.concatenate(
            [np.broadcast_to(trial.target_poses, (trial.steps + 1, 2, 3)) for trial in trials]
        )
        np.testing.assert_array_equal(dataset.target[:], expected_target)
        np.testing.assert_array_equal(dataset.target_poses[:], expected_poses)
        np.testing.assert_array_equal(dataset.episode_ends[:], np.cumsum([trial.steps + 1 for trial in trials]))
        np.testing.assert_array_equal(dataset.episode_target[:], [int(trial.target) for trial in trials])
        assert dataset.episode_success[:].all()
        np.testing.assert_array_equal(dataset.episode_steps[:], [trial.steps for trial in trials])
        stored_ids = [value.rstrip(b"\x00") for value in dataset.episode_ids[:].tolist()]
        expected_ids = [f"p{index:02d}".encode() for index in range(10)]
        expected_ids += [f"g{index:02d}".encode() for index in range(10)]
        assert stored_ids == expected_ids
        np.testing.assert_allclose(dataset.episode_created[:], CREATED_TS)
        np.testing.assert_array_equal(dataset.episode_seed[:], [trial.meta.seed for trial in trials])


def test_zip_is_a_zarr_v2_store_at_the_root(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    zip_path = packed[0]
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert {name.split("/", 1)[0] for name in names} == {".zgroup", ".zattrs", "data", "meta"}
        assert ".zgroup" in names
        assert "data/img/.zarray" in names
        assert "meta/episode_ends/.zarray" in names
        assert json.loads(archive.read(".zgroup"))["zarr_format"] == 2
        image_meta = json.loads(archive.read("data/img/.zarray"))
        assert image_meta["compressor"]["id"] == "zstd"
        assert image_meta["compressor"]["level"] == 5
        assert image_meta["chunks"][0] == 1
        ids_meta = json.loads(archive.read("meta/episode_ids/.zarray"))
        assert ids_meta["dtype"] == "|S32"
        assert ids_meta["compressor"]["id"] == "zstd"


def test_the_zip_is_the_only_output(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    zip_path = packed[0]
    out_dir = zip_path.parent
    assert [entry.name for entry in out_dir.iterdir()] == [zip_path.name]
    assert not any(entry.name.endswith(".zarr") or entry.name.endswith(".tmp") for entry in out_dir.rglob("*"))


def test_root_attributes(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    zip_path, data_dir, _, summary = packed
    store = ZipStore(zip_path, mode="r")
    try:
        attrs = dict(zarr.open_group(store, mode="r").attrs)
    finally:
        store.close()
    required = {
        "schema_version",
        "fps",
        "target_colors",
        "target_bounds",
        "target_min_separation",
        "success_threshold",
        "observation_size",
        "created_at",
        "source",
        "split_seed",
        "n_episodes",
        "n_frames",
        "episode_counts",
    }
    assert required <= set(attrs)
    assert attrs["schema_version"] == SCHEMA_VERSION
    assert attrs["fps"] == FPS
    assert attrs["target_colors"] == [list(color) for color in TARGET_COLORS]
    assert attrs["target_bounds"] == list(DEFAULT_CONFIG.target_bounds)
    assert attrs["target_min_separation"] == DEFAULT_CONFIG.target_min_separation
    assert attrs["success_threshold"] == SUCCESS_THRESHOLD
    assert attrs["observation_size"] == OBSERVATION_SIZE
    assert attrs["source"] == str(data_dir)
    assert attrs["split_seed"] == 0
    assert attrs["n_episodes"] == summary.n_episodes
    assert attrs["n_frames"] == summary.n_frames
    assert attrs["episode_counts"] == {"pink": 10, "green": 10}
    assert attrs["created_at"].endswith("Z")
    datetime.fromisoformat(attrs["created_at"])
    json.dumps(attrs)


def test_splits_partition_the_data_and_are_stratified(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    zip_path = packed[0]
    with load_dataset(zip_path) as dataset:
        split = np.array(dataset.split[:])
        targets = np.array(dataset.episode_target[:])
    groups = [set(np.where(split == code)[0].tolist()) for code in range(3)]
    assert set().union(*groups) == set(range(20))
    assert sum(len(group) for group in groups) == 20
    for color in (PINK, GREEN):
        codes, counts = np.unique(split[targets == color], return_counts=True)
        assert dict(zip(codes.tolist(), counts.tolist(), strict=True)) == {0: 8, 1: 1, 2: 1}


def test_splits_are_deterministic_for_a_seed(packed: tuple[Path, Path, list[Trial], PackSummary]) -> None:
    _, data_dir, _, _ = packed
    out_dir = data_dir.parent
    first = pack_trials(data_dir, out_dir / "split_a.zip", split_seed=3)
    second = pack_trials(data_dir, out_dir / "split_b.zip", split_seed=3)
    other = pack_trials(data_dir, out_dir / "split_c.zip", split_seed=5)
    assert first.split_counts == second.split_counts
    np.testing.assert_array_equal(split_of(first.out_path), split_of(second.out_path))
    assert not np.array_equal(split_of(first.out_path), split_of(other.out_path))


def test_overwrite_false_refuses_an_existing_output(
    packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    _, data_dir, _, _ = packed
    out_path = tmp_path / "existing.zip"
    out_path.write_bytes(b"keep")
    with pytest.raises(PackError, match="exists"):
        pack_trials(data_dir, out_path)
    assert out_path.read_bytes() == b"keep"
    pack_trials(data_dir, out_path, overwrite=True)
    assert out_path.stat().st_size > len(b"keep")


def test_tiny_data_warns_about_empty_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_trial(data_dir, PINK, "only", 2, 0)
    summary = pack_trials(data_dir, tmp_path / "tiny.zip")
    assert summary.split_counts == {"train": 0, "validation": 0, "test": 1}
    assert any("pink" in warning and "train" in warning for warning in summary.warnings)


def test_invalid_trial_fails_closed_without_output(
    packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    _, data_dir, _, _ = packed
    bad_path = data_dir / "pink" / "p00.npz"
    with np.load(bad_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    payload["state"] = payload["state"][:-1]
    np.savez(bad_path, **payload)
    out_path = tmp_path / "failed" / "twint.zarr.zip"
    with pytest.raises(TrialInvalid, match="p00.npz"):
        pack_trials(data_dir, out_path)
    assert not out_path.exists()
    assert not any(entry.name.endswith(".zarr") for entry in tmp_path.rglob("*"))


def test_episode_id_longer_than_32_bytes_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = save_trial(data_dir, make_trial(2, PINK, seed=0), "short")
    path.rename(path.with_name("x" * 33 + ".npz"))
    out_path = tmp_path / "long.zip"
    with pytest.raises(PackError, match="32"):
        pack_trials(data_dir, out_path)
    assert not out_path.exists()


def test_pack_rejects_an_empty_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(PackError, match="no trials"):
        pack_trials(data_dir, tmp_path / "empty.zip")


def test_out_path_must_be_a_zip(packed: tuple[Path, Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    _, data_dir, _, _ = packed
    with pytest.raises(PackError, match="zip"):
        pack_trials(data_dir, tmp_path / "store.zarr")

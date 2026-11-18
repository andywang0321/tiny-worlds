"""Loader: lazy arrays, episodes, splits, plain directories, and unpack."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import zarr

from tiny_worlds.twin_t.dataset import DatasetInvalid, TwinTDataset, UnpackError, load_dataset, unpack
from tiny_worlds.twin_t.pack import PackSummary, pack_trials
from tiny_worlds.twin_t.schema import FPS, GREEN, PINK, SCHEMA_VERSION
from tiny_worlds.twin_t.trials import Trial, TrialMeta, load_trial, save_trial

CREATED = "2025-01-02T03:04:05Z"
CREATED_TS = datetime.fromisoformat(CREATED).timestamp()
ARRAY_NAMES = (
    "img",
    "state",
    "action",
    "coverage",
    "success",
    "keypoint",
    "n_contacts",
    "target",
    "target_poses",
    "episode_ends",
    "episode_target",
    "episode_success",
    "episode_steps",
    "episode_ids",
    "episode_created",
    "episode_seed",
    "split",
)


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


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Trial], PackSummary]:
    base = tmp_path_factory.mktemp("dataset")
    data_dir = base / "data"
    trials = [write_trial(data_dir, PINK, f"p{index:02d}", 2 + index % 4, index) for index in range(10)]
    trials += [write_trial(data_dir, GREEN, f"g{index:02d}", 2 + index % 4, 100 + index) for index in range(10)]
    zip_path = base / "twint.zarr.zip"
    return zip_path, trials, pack_trials(data_dir, zip_path)


def extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return destination


def episode_id(index: int) -> str:
    return f"p{index:02d}" if index < 10 else f"g{index - 10:02d}"


def test_lazy_arrays_and_counts(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, trials, summary = packed
    with load_dataset(zip_path) as dataset:
        assert isinstance(dataset, TwinTDataset)
        assert dataset.episode_count == 20
        assert dataset.frame_count == summary.n_frames
        assert dataset.img.shape == (summary.n_frames, 96, 96, 3)
        assert dataset.img.dtype == np.uint8
        assert dataset.state.dtype == np.float32
        assert dataset.n_contacts.dtype == np.int32
        assert dataset.episode_ids.dtype == np.dtype("S32")
        np.testing.assert_array_equal(dataset.img[:], np.concatenate([trial.obs for trial in trials]))
        np.testing.assert_array_equal(dataset.state[:], np.concatenate([trial.state for trial in trials]))
        np.testing.assert_array_equal(dataset.action[:], np.concatenate([trial.action for trial in trials]))
        np.testing.assert_array_equal(dataset.coverage[:], np.concatenate([trial.coverage for trial in trials]))
        np.testing.assert_array_equal(dataset.keypoint[:], np.concatenate([trial.keypoint for trial in trials]))
        np.testing.assert_array_equal(dataset.n_contacts[:], np.concatenate([trial.n_contacts for trial in trials]))
        np.testing.assert_array_equal(dataset.episode_steps[:], [trial.steps for trial in trials])
        np.testing.assert_array_equal(dataset.episode_seed[:], [trial.meta.seed for trial in trials])
        np.testing.assert_allclose(dataset.episode_created[:], CREATED_TS)


def test_episode_matches_the_raw_trial(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, trials, _ = packed
    with load_dataset(zip_path) as dataset:
        for index in (0, 7, 19):
            trial = trials[index]
            episode = dataset.episode(index)
            np.testing.assert_array_equal(episode.img, trial.obs)
            np.testing.assert_array_equal(episode.state, trial.state)
            np.testing.assert_array_equal(episode.action, trial.action)
            np.testing.assert_array_equal(episode.coverage, trial.coverage)
            np.testing.assert_array_equal(episode.success, trial.success)
            np.testing.assert_array_equal(episode.keypoint, trial.keypoint)
            np.testing.assert_array_equal(episode.n_contacts, trial.n_contacts)
            np.testing.assert_array_equal(episode.target_poses, trial.target_poses)
            assert episode.target == int(trial.target)
            assert episode.episode_id == episode_id(index)
            assert episode.steps == trial.steps
            assert episode.split == int(dataset.split[index])


def test_episodes_can_be_filtered_by_split(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, _, _ = packed
    with load_dataset(zip_path) as dataset:
        splits = np.array(dataset.split[:])
        ids = [value.rstrip(b"\x00").decode() for value in dataset.episode_ids[:]]
        assert [episode.episode_id for episode in dataset.episodes()] == ids
        for code in range(3):
            episodes = list(dataset.episodes(code))
            expected = {ids[index] for index in np.where(splits == code)[0]}
            assert len(episodes) == int((splits == code).sum())
            assert {episode.episode_id for episode in episodes} == expected
            assert all(episode.split == code for episode in episodes)
        with pytest.raises(DatasetInvalid):
            list(dataset.episodes(7))


def test_to_numpy_returns_every_array(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, _, _ = packed
    with load_dataset(zip_path) as dataset:
        arrays = dataset.to_numpy()
        assert set(ARRAY_NAMES) <= set(arrays)
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(arrays[name], getattr(dataset, name)[:])
            assert arrays[name].flags.writeable


def test_episode_index_is_checked(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, _, _ = packed
    with load_dataset(zip_path) as dataset:
        for index in (-1, dataset.episode_count):
            with pytest.raises(DatasetInvalid):
                dataset.episode(index)
        with pytest.raises(DatasetInvalid):
            dataset.episode(1.5)  # type: ignore[arg-type]


def test_close_is_idempotent_and_blocks_access(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    zip_path, _, _ = packed
    dataset = load_dataset(zip_path)
    assert not dataset.closed
    dataset.close()
    dataset.close()
    assert dataset.closed
    with pytest.raises(DatasetInvalid, match="closed"):
        _ = dataset.img
    with load_dataset(zip_path) as managed:
        assert managed.episode_count == 20
    assert managed.closed


def test_a_plain_zarr_directory_is_accepted(
    packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    zip_path, trials, _ = packed
    directory = extract(zip_path, tmp_path / "store.zarr")
    with load_dataset(directory) as dataset:
        assert dataset.episode_count == 20
        np.testing.assert_array_equal(dataset.img[:], np.concatenate([trial.obs for trial in trials]))


def test_unpack_round_trips_the_arrays_and_labels(
    packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    zip_path, _, _ = packed
    out_dir = tmp_path / "raw"
    paths = unpack(zip_path, out_dir)
    assert len(paths) == 20
    with load_dataset(zip_path) as dataset:
        for index, path in enumerate(paths):
            assert path.parent == out_dir / ("pink" if index < 10 else "green")
            trial = load_trial(path)
            episode = dataset.episode(index)
            for name, value in (
                ("obs", episode.img),
                ("state", episode.state),
                ("action", episode.action),
                ("coverage", episode.coverage),
                ("success", episode.success),
                ("keypoint", episode.keypoint),
                ("n_contacts", episode.n_contacts),
                ("target_poses", episode.target_poses),
            ):
                np.testing.assert_array_equal(getattr(trial, name), value, err_msg=f"{path.name}:{name}")
            assert trial.target == episode.target
            assert trial.meta.seed == int(dataset.episode_seed[index])
            assert trial.meta.created_at == CREATED
            assert trial.meta.active_seconds == pytest.approx(episode.steps / FPS)
    repacked = pack_trials(out_dir, tmp_path / "repacked.zip", split_seed=0)
    with load_dataset(repacked.out_path) as again, load_dataset(zip_path) as before:
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(again, name)[:], getattr(before, name)[:], err_msg=name)


def test_unpack_refuses_a_non_empty_directory(
    packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    zip_path, _, _ = packed
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("keep")
    with pytest.raises(UnpackError, match="not empty"):
        unpack(zip_path, out_dir)
    assert (out_dir / "keep.txt").read_text() == "keep"
    unpack(zip_path, out_dir, overwrite=True)
    assert (out_dir / "keep.txt").read_text() == "keep"
    assert (out_dir / "pink").is_dir()


def test_missing_array_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    directory = extract(packed[0], tmp_path / "missing")
    shutil.rmtree(directory / "data" / "action")
    with pytest.raises(DatasetInvalid, match="data/action"):
        load_dataset(directory)


def test_wrong_shape_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    directory = extract(packed[0], tmp_path / "shape")
    path = directory / "data" / "state" / ".zarray"
    metadata = json.loads(path.read_text())
    metadata["shape"][0] -= 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(DatasetInvalid, match="data/state"):
        load_dataset(directory)


def test_wrong_dtype_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    directory = extract(packed[0], tmp_path / "dtype")
    path = directory / "data" / "state" / ".zarray"
    metadata = json.loads(path.read_text())
    metadata["dtype"] = "<i8"
    path.write_text(json.dumps(metadata))
    with pytest.raises(DatasetInvalid, match="data/state"):
        load_dataset(directory)


def test_non_monotonic_episode_ends_fail_closed(
    packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    directory = extract(packed[0], tmp_path / "ends")
    root = zarr.open_group(directory, mode="r+")
    ends = np.array(root["meta/episode_ends"][:])
    ends[1] = ends[0]
    root["meta/episode_ends"][:] = ends
    with pytest.raises(DatasetInvalid, match="episode_ends"):
        load_dataset(directory)


def test_a_bad_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DatasetInvalid, match="nope.zip"):
        load_dataset(tmp_path / "nope.zip")
    plain = tmp_path / "plain.txt"
    plain.write_text("not a dataset")
    with pytest.raises(DatasetInvalid):
        load_dataset(plain)

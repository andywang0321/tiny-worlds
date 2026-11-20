"""Packed ``.npz`` loader: eager arrays, derived conveniences, episodes, and unpack."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tiny_worlds.twin_t.dataset import DatasetInvalid, TwinTDataset, UnpackError, load_dataset, unpack
from tiny_worlds.twin_t.pack import PackSummary, pack_trials
from tiny_worlds.twin_t.render import render_frames
from tiny_worlds.twin_t.schema import FPS, GREEN, PACKED_ARRAYS, PINK, SCHEMA_VERSION
from tiny_worlds.twin_t.trials import Trial, TrialMeta, derive_actions, load_trial, save_trial

CREATED = "2025-01-02T03:04:05Z"
CREATED_TS = datetime.fromisoformat(CREATED).timestamp()
ARRAY_NAMES = PACKED_ARRAYS


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


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Trial], PackSummary]:
    base = tmp_path_factory.mktemp("dataset")
    data_dir = base / "data"
    trials = [write_trial(data_dir, PINK, f"p{index:02d}", 2 + index % 4, index) for index in range(10)]
    trials += [write_trial(data_dir, GREEN, f"g{index:02d}", 2 + index % 4, 100 + index) for index in range(10)]
    out_path = base / "twint.npz"
    return out_path, trials, pack_trials(data_dir, out_path)


def test_eager_arrays_and_counts(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, trials, summary = packed
    with load_dataset(out_path) as dataset:
        assert isinstance(dataset, TwinTDataset)
        assert dataset.episode_count == 20
        assert dataset.frame_count == summary.n_frames
        assert dataset.state.shape == (summary.n_frames, 5)
        assert dataset.state.dtype == np.float32
        np.testing.assert_array_equal(dataset.state, np.concatenate([trial.state for trial in trials]))
        np.testing.assert_array_equal(dataset.action, np.concatenate([derive_actions(trial.state) for trial in trials]))
        np.testing.assert_array_equal(dataset.episode_ends, np.cumsum([trial.steps + 1 for trial in trials]))
        np.testing.assert_array_equal(dataset.episode_seed, [trial.meta.seed for trial in trials])
        assert dataset.episode_ids.dtype == np.dtype("S32")
        np.testing.assert_allclose(dataset.episode_created, CREATED_TS)
        assert not dataset.closed


def test_derived_action_rule_includes_the_terminal_copy(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, trials, _ = packed
    with load_dataset(out_path) as dataset:
        for index, trial in enumerate(trials):
            episode = dataset.episode(index)
            np.testing.assert_array_equal(episode.action[:-1], episode.state[1:, :2])
            np.testing.assert_array_equal(episode.action[-1], episode.state[-1, :2])
            assert episode.steps == trial.steps


def test_target_and_poses_are_broadcast_per_frame(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, trials, _ = packed
    with load_dataset(out_path) as dataset:
        assert dataset.target.shape == (dataset.frame_count,)
        assert dataset.target.dtype == np.int8
        assert dataset.target_poses.shape == (dataset.frame_count, 2, 3)
        start = 0
        for trial in trials:
            end = start + trial.steps + 1
            np.testing.assert_array_equal(dataset.target[start:end], int(trial.target))
            np.testing.assert_array_equal(
                dataset.target_poses[start:end], np.broadcast_to(trial.target_poses, (trial.steps + 1, 2, 3))
            )
            start = end


def test_episode_matches_the_raw_trial(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, trials, _ = packed
    with load_dataset(out_path) as dataset:
        for index in (0, 7, 19):
            trial = trials[index]
            episode = dataset.episode(index)
            np.testing.assert_array_equal(episode.state, trial.state)
            np.testing.assert_array_equal(episode.action, derive_actions(trial.state))
            np.testing.assert_array_equal(episode.target_poses, trial.target_poses)
            assert episode.target == int(trial.target)
            assert episode.episode_id == (f"p{index:02d}" if index < 10 else f"g{index - 10:02d}")
            assert episode.steps == trial.steps
            assert episode.split == int(dataset.split[index])
            frames = episode.frames(32)
            assert frames.shape == (trial.steps + 1, 32, 32, 3)
            assert frames.dtype == np.uint8
            np.testing.assert_array_equal(frames, render_frames(episode.state, episode.target_poses, 32))


def test_episodes_can_be_filtered_by_split(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, _, _ = packed
    with load_dataset(out_path) as dataset:
        splits = np.array(dataset.split)
        ids = [value.rstrip(b"\x00").decode() for value in dataset.episode_ids]
        assert [episode.episode_id for episode in dataset.episodes()] == ids
        for code in (0, 1):
            episodes = list(dataset.episodes(code))
            expected = {ids[index] for index in np.where(splits == code)[0]}
            assert len(episodes) == int((splits == code).sum())
            assert {episode.episode_id for episode in episodes} == expected
            assert all(episode.split == code for episode in episodes)
        with pytest.raises(DatasetInvalid):
            list(dataset.episodes(7))


def test_to_numpy_returns_every_member(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, _, _ = packed
    with load_dataset(out_path) as dataset:
        arrays = dataset.to_numpy()
        assert set(arrays) == set(ARRAY_NAMES)
        for name in ARRAY_NAMES:
            assert arrays[name].flags.writeable
            if name == "meta":
                assert json.loads(str(arrays[name].item())) == dataset.metadata
                continue
            np.testing.assert_array_equal(arrays[name], getattr(dataset, name))


def test_episode_index_is_checked(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, _, _ = packed
    with load_dataset(out_path) as dataset:
        for index in (-1, dataset.episode_count):
            with pytest.raises(DatasetInvalid):
                dataset.episode(index)
        with pytest.raises(DatasetInvalid):
            dataset.episode(1.5)  # type: ignore[arg-type]


def test_close_is_idempotent_and_blocks_access(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, _, _ = packed
    dataset = load_dataset(out_path)
    assert not dataset.closed
    dataset.close()
    dataset.close()
    assert dataset.closed
    with pytest.raises(DatasetInvalid, match="closed"):
        _ = dataset.state
    with load_dataset(out_path) as managed:
        assert managed.episode_count == 20
    assert managed.closed


def test_a_bad_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DatasetInvalid, match="nope.npz"):
        load_dataset(tmp_path / "nope.npz")
    plain = tmp_path / "plain.txt"
    plain.write_text("not a dataset")
    with pytest.raises(DatasetInvalid):
        load_dataset(plain)


def test_a_missing_member_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    del arrays["episode_target"]
    path = tmp_path / "missing.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="episode_target"):
        load_dataset(path)


def test_a_wrong_shape_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    arrays["state"] = arrays["state"][:-1]
    path = tmp_path / "shape.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="state"):
        load_dataset(path)


def test_a_wrong_dtype_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    arrays["state"] = arrays["state"].astype(np.float64)
    path = tmp_path / "dtype.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="state"):
        load_dataset(path)


def test_non_monotonic_episode_ends_fail_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    ends = arrays["episode_ends"].copy()
    ends[1] = ends[0]
    arrays["episode_ends"] = ends
    path = tmp_path / "ends.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="episode_ends"):
        load_dataset(path)


def test_episode_ends_must_match_the_state_rows(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    ends = arrays["episode_ends"].copy()
    ends[-1] += 1
    arrays["episode_ends"] = ends
    path = tmp_path / "row-count.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="state"):
        load_dataset(path)


def test_a_bad_split_code_fails_closed(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    arrays = _members(packed[0])
    split = arrays["split"].copy()
    split[0] = 2
    arrays["split"] = split
    path = tmp_path / "split.npz"
    np.savez(path, **arrays)
    with pytest.raises(DatasetInvalid, match="split"):
        load_dataset(path)


def _members(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def test_unpack_round_trips_the_arrays_and_labels(
    packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path
) -> None:
    out_path, trials, _ = packed
    out_dir = tmp_path / "raw"
    paths = unpack(out_path, out_dir)
    assert len(paths) == 20
    with load_dataset(out_path) as dataset:
        for index, path in enumerate(paths):
            assert path.parent == out_dir / ("pink" if index < 10 else "green")
            trial = load_trial(path)
            episode = dataset.episode(index)
            np.testing.assert_array_equal(trial.state, episode.state)
            np.testing.assert_array_equal(trial.target, np.array(episode.target, np.int8))
            np.testing.assert_array_equal(trial.target_poses, episode.target_poses)
            np.testing.assert_array_equal(trial.action, derive_actions(trials[index].state))
            assert trial.meta.schema_version == SCHEMA_VERSION
            assert trial.meta.seed == int(dataset.episode_seed[index])
            assert trial.meta.created_at == CREATED
            assert trial.meta.active_seconds == pytest.approx(episode.steps / FPS)
    repacked = pack_trials(out_dir, tmp_path / "repacked.npz", split_seed=0)
    with load_dataset(repacked.out_path) as again, load_dataset(out_path) as before:
        for name in ARRAY_NAMES:
            if name == "meta":
                continue
            np.testing.assert_array_equal(getattr(again, name), getattr(before, name), err_msg=name)


def test_unpack_refuses_a_non_empty_directory(packed: tuple[Path, list[Trial], PackSummary], tmp_path: Path) -> None:
    out_path, _, _ = packed
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("keep")
    with pytest.raises(UnpackError, match="not empty"):
        unpack(out_path, out_dir)
    assert (out_dir / "keep.txt").read_text() == "keep"
    unpack(out_path, out_dir, overwrite=True)
    assert (out_dir / "keep.txt").read_text() == "keep"
    assert (out_dir / "pink").is_dir()


def test_metadata_is_a_json_object(packed: tuple[Path, list[Trial], PackSummary]) -> None:
    out_path, _, summary = packed
    with load_dataset(out_path) as dataset:
        attrs = dataset.metadata
    assert isinstance(json.loads(json.dumps(attrs)), dict)
    assert attrs["n_episodes"] == summary.n_episodes

"""Raw v2 trial format: save, load, scan, derived actions, legacy conversion."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from tiny_worlds.twin_t import trials as trials_module
from tiny_worlds.twin_t.schema import GREEN, PINK, SCHEMA_VERSION
from tiny_worlds.twin_t.trials import (
    Trial,
    TrialInvalid,
    TrialMeta,
    derive_actions,
    load_legacy_trial,
    load_trial,
    save_trial,
    scan_trials,
    trial_path,
)

CREATED = "2025-01-02T03:04:05Z"


def make_trial(
    steps: int = 3,
    target: int = PINK,
    *,
    seed: int = 7,
    created_at: str = CREATED,
    active_seconds: float = 0.3,
) -> Trial:
    rng = np.random.default_rng(seed)
    n_frames = steps + 1
    return Trial(
        state=rng.uniform(0, 512, size=(n_frames, 5)).astype(np.float32),
        target=np.array(target, np.int8),
        target_poses=rng.uniform(0, 512, size=(2, 3)).astype(np.float32),
        meta=TrialMeta(SCHEMA_VERSION, seed, created_at, active_seconds),
    )


def replace(trial: Trial, **changes: object) -> Trial:
    return dataclasses.replace(trial, **changes)


def test_save_load_round_trip(tmp_path: Path) -> None:
    trial = make_trial()
    path = save_trial(tmp_path, trial, "pink_0001")
    assert path == tmp_path / "pink" / "pink_0001.npz"
    loaded = load_trial(path)
    np.testing.assert_array_equal(loaded.state, trial.state)
    np.testing.assert_array_equal(loaded.target, trial.target)
    np.testing.assert_array_equal(loaded.target_poses, trial.target_poses)
    assert loaded.meta == trial.meta
    assert loaded.steps == 3


def test_a_raw_trial_has_only_the_v2_keys(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(), "keys")
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"state", "target", "target_poses", "meta"}
        for removed in ("obs", "action", "coverage", "success", "keypoint", "n_contacts"):
            assert removed not in archive.files
        assert archive["state"].shape == (4, 5)
        assert archive["state"].dtype == np.float32
        assert archive["target"].shape == ()
        assert archive["target"].dtype == np.int8
        assert archive["target_poses"].shape == (2, 3)
        assert archive["target_poses"].dtype == np.float32


def test_meta_is_a_json_unicode_scalar(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(seed=11), "meta")
    with np.load(path, allow_pickle=False) as archive:
        meta_array = archive["meta"]
        assert meta_array.shape == ()
        assert meta_array.dtype.kind == "U"
        payload = json.loads(str(meta_array))
    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "seed": 11,
        "created_at": CREATED,
        "active_seconds": pytest.approx(0.3),
    }


def test_derived_actions_are_the_realized_next_agent_position() -> None:
    trial = make_trial(steps=4)
    action = trial.action
    assert action.shape == (5, 2)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action[:-1], trial.state[1:, :2])
    np.testing.assert_array_equal(action[-1], trial.state[-1, :2])
    # The terminal row repeats the final realized position, not the previous state.
    np.testing.assert_array_equal(action[-1], action[-2])
    assert not np.array_equal(action[-1], trial.state[-2, :2])


def test_derived_actions_handle_a_zero_step_trial() -> None:
    trial = make_trial(steps=0)
    assert trial.steps == 0
    action = trial.action
    assert action.shape == (1, 2)
    np.testing.assert_array_equal(action, trial.state[:, :2])


def test_derive_actions_does_not_touch_the_state() -> None:
    state = make_trial().state
    before = state.copy()
    derive_actions(state)
    np.testing.assert_array_equal(state, before)


def test_save_leaves_only_the_final_file(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(), "atomic")
    assert [entry.name for entry in path.parent.iterdir()] == ["atomic.npz"]


def test_failed_save_leaves_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(source: object, destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(trials_module.os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        save_trial(tmp_path, make_trial(), "doomed")
    directory = tmp_path / "pink"
    assert list(directory.iterdir()) == []


def test_trial_path_uses_the_color_directory(tmp_path: Path) -> None:
    assert trial_path(tmp_path, PINK, "p1") == tmp_path / "pink" / "p1.npz"
    assert trial_path(tmp_path, GREEN, "g1") == tmp_path / "green" / "g1.npz"
    for target in (2, -1):
        with pytest.raises(ValueError):
            trial_path(tmp_path, target, "x")


@pytest.mark.parametrize("episode_id", ["", ".", "..", "../escape", "a/b", "a\\b", ".hidden", "with space", "x" * 33])
def test_trial_path_rejects_unsafe_episode_ids(tmp_path: Path, episode_id: str) -> None:
    with pytest.raises(ValueError):
        trial_path(tmp_path, PINK, episode_id)


def test_trial_path_accepts_a_full_32_byte_id(tmp_path: Path) -> None:
    assert trial_path(tmp_path, PINK, "x" * 32).name == "x" * 32 + ".npz"


def _nan_state(trial: Trial) -> Trial:
    state = trial.state.copy()
    state[0, 3] = np.nan
    return replace(trial, state=state)


def _inf_state(trial: Trial) -> Trial:
    state = trial.state.copy()
    state[1, 0] = np.inf
    return replace(trial, state=state)


def _float64_state(trial: Trial) -> Trial:
    return replace(trial, state=trial.state.astype(np.float64))


def _four_column_state(trial: Trial) -> Trial:
    return replace(trial, state=np.zeros((3, 4), np.float32))


def _empty_state(trial: Trial) -> Trial:
    return replace(trial, state=trial.state[:0])


def _bad_target(trial: Trial) -> Trial:
    return replace(trial, target=np.array(2, np.int8))


def _int64_target(trial: Trial) -> Trial:
    return replace(trial, target=np.array(trial.target, np.int64))


def _vector_target(trial: Trial) -> Trial:
    return replace(trial, target=np.array([int(trial.target)], np.int8))


def _bad_target_poses(trial: Trial) -> Trial:
    return replace(trial, target_poses=np.zeros((3, 3), np.float32))


def _float64_target_poses(trial: Trial) -> Trial:
    return replace(trial, target_poses=trial.target_poses.astype(np.float64))


def _nan_target_poses(trial: Trial) -> Trial:
    poses = trial.target_poses.copy()
    poses[1, 2] = np.nan
    return replace(trial, target_poses=poses)


BAD_TRIALS = [
    pytest.param(_nan_state, id="nan-state"),
    pytest.param(_inf_state, id="inf-state"),
    pytest.param(_float64_state, id="float64-state"),
    pytest.param(_four_column_state, id="four-column-state"),
    pytest.param(_empty_state, id="empty-state"),
    pytest.param(_bad_target, id="bad-target"),
    pytest.param(_int64_target, id="int64-target"),
    pytest.param(_vector_target, id="vector-target"),
    pytest.param(_bad_target_poses, id="bad-target-poses"),
    pytest.param(_float64_target_poses, id="float64-target-poses"),
    pytest.param(_nan_target_poses, id="nan-target-poses"),
]


@pytest.mark.parametrize("mutate", BAD_TRIALS)
def test_save_rejects_invalid_trials(tmp_path: Path, mutate: object) -> None:
    trial = mutate(make_trial())  # type: ignore[operator]
    with pytest.raises(TrialInvalid):
        save_trial(tmp_path, trial, "invalid")
    assert list(tmp_path.rglob("*.npz")) == []


@pytest.mark.parametrize(
    "make_meta",
    [
        lambda: TrialMeta(0, 0, CREATED, 1.0),
        lambda: TrialMeta(SCHEMA_VERSION, 1.5, CREATED, 1.0),
        lambda: TrialMeta(SCHEMA_VERSION, 0, "not-a-date", 1.0),
        lambda: TrialMeta(SCHEMA_VERSION, 0, CREATED, -1.0),
        lambda: TrialMeta(SCHEMA_VERSION, 0, CREATED, float("nan")),
    ],
)
def test_meta_rejects_bad_values(make_meta: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_meta()  # type: ignore[operator]


def test_load_reports_the_bad_path(tmp_path: Path) -> None:
    broken = tmp_path / "broken.npz"
    np.savez(broken, state=np.zeros((2, 5), np.float32))
    with pytest.raises(TrialInvalid, match="broken.npz"):
        load_trial(broken)


def test_load_rejects_pickled_arrays(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(), "pickled")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    payload["meta"] = np.array("{}", dtype=object)
    np.savez(path, **payload)
    with pytest.raises(TrialInvalid, match="pickled.npz"):
        load_trial(path)


def test_load_accepts_extra_meta_fields(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(seed=7), "extra")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    meta = json.loads(str(payload["meta"]))
    meta["steps"] = 3
    meta["note"] = "forward compatible"
    payload["meta"] = np.asarray(json.dumps(meta))
    np.savez(path, **payload)
    loaded = load_trial(path)
    assert loaded.meta.seed == 7
    assert loaded.steps == 3


def test_load_rejects_the_legacy_schema_version(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(), "legacy-version")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    meta = json.loads(str(payload["meta"]))
    meta["schema_version"] = 1
    payload["meta"] = np.asarray(json.dumps(meta))
    np.savez(path, **payload)
    with pytest.raises(TrialInvalid, match="schema_version"):
        load_trial(path)


def test_scan_trials_across_both_colors(tmp_path: Path) -> None:
    save_trial(tmp_path, make_trial(2, PINK, seed=1), "a")
    save_trial(tmp_path, make_trial(5, GREEN, seed=2), "b")
    save_trial(tmp_path, make_trial(1, PINK, seed=3), "c")
    records = scan_trials(tmp_path)
    assert [(record.episode_id, record.target) for record in records] == [("a", PINK), ("c", PINK), ("b", GREEN)]
    first = records[0]
    assert first.path == tmp_path / "pink" / "a.npz"
    assert first.steps == 2
    assert first.seed == 1
    assert first.created_at == CREATED
    assert not hasattr(first, "success")


def test_scan_requires_the_directory_color_to_match(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(target=GREEN), "greenish")
    moved = tmp_path / "pink" / "greenish.npz"
    moved.parent.mkdir(parents=True)
    path.rename(moved)
    with pytest.raises(TrialInvalid, match="greenish.npz"):
        scan_trials(tmp_path)


def test_scan_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(TrialInvalid, match="missing"):
        scan_trials(tmp_path / "missing")


def _write_legacy_trial(path: Path, trial: Trial) -> Path:
    frames = trial.state.shape[0]
    rng = np.random.default_rng(123)
    payload = {
        "obs": rng.integers(0, 256, size=(frames, 96, 96, 3), dtype=np.uint8),
        "state": trial.state,
        "action": rng.uniform(0, 512, size=(frames, 2)).astype(np.float32),
        "coverage": rng.uniform(0, 1, size=(frames, 2)).astype(np.float32),
        "success": np.zeros(frames, dtype=np.bool_),
        "keypoint": rng.normal(size=(frames, 9, 2)).astype(np.float32),
        "n_contacts": rng.integers(0, 4, size=(frames, 1), dtype=np.int32),
        "target": trial.target,
        "target_poses": trial.target_poses,
        "meta": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": trial.meta.seed,
                    "created_at": trial.meta.created_at,
                    "active_seconds": trial.meta.active_seconds,
                }
            )
        ),
    }
    payload["success"][-1] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def test_load_legacy_trial_converts_a_v1_trial(tmp_path: Path) -> None:
    original = make_trial(steps=4, seed=9, active_seconds=0.4)
    path = _write_legacy_trial(tmp_path / "legacy" / "old.npz", original)
    converted = load_legacy_trial(path)
    assert converted.meta.schema_version == SCHEMA_VERSION
    assert converted.meta.seed == original.meta.seed
    assert converted.meta.created_at == original.meta.created_at
    assert converted.meta.active_seconds == pytest.approx(original.meta.active_seconds)
    np.testing.assert_array_equal(converted.state, original.state)
    np.testing.assert_array_equal(converted.target, original.target)
    np.testing.assert_array_equal(converted.target_poses, original.target_poses)
    assert converted.steps == 4
    assert not hasattr(converted, "obs")


def test_load_legacy_trial_is_also_loadable_after_saving(tmp_path: Path) -> None:
    original = make_trial(steps=2, seed=5)
    path = _write_legacy_trial(tmp_path / "legacy" / "old.npz", original)
    converted = load_legacy_trial(path)
    saved = save_trial(tmp_path / "converted", converted, "round_trip")
    reloaded = load_trial(saved)
    assert reloaded.meta == converted.meta
    np.testing.assert_array_equal(reloaded.state, original.state)

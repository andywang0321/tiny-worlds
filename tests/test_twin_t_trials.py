"""Raw trial format: save, load, scan, atomicity, and validation."""

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
    action = rng.uniform(0, 512, size=(n_frames, 2)).astype(np.float32)
    if steps > 0:
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
        meta=TrialMeta(SCHEMA_VERSION, seed, created_at, active_seconds),
    )


def replace(trial: Trial, **changes: object) -> Trial:
    return dataclasses.replace(trial, **changes)


def test_save_load_round_trip(tmp_path: Path) -> None:
    trial = make_trial()
    path = save_trial(tmp_path, trial, "pink_0001")
    assert path == tmp_path / "pink" / "pink_0001.npz"
    loaded = load_trial(path)
    for name in ("obs", "state", "action", "coverage", "success", "keypoint", "n_contacts", "target", "target_poses"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(trial, name), err_msg=name)
    assert loaded.meta == trial.meta
    assert loaded.steps == 3
    assert loaded.success[-1]
    np.testing.assert_array_equal(loaded.action[-1], loaded.action[-2])


def test_meta_is_a_json_unicode_scalar(tmp_path: Path) -> None:
    path = save_trial(tmp_path, make_trial(seed=11), "meta")
    with np.load(path, allow_pickle=False) as archive:
        meta_array = archive["meta"]
        assert meta_array.shape == ()
        assert meta_array.dtype.kind == "U"
        payload = json.loads(str(meta_array))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["seed"] == 11
    assert payload["created_at"] == CREATED
    assert payload["active_seconds"] == pytest.approx(0.3)


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


def _bad_coverage(trial: Trial) -> Trial:
    coverage = trial.coverage.copy()
    coverage[0, 0] = 1.5
    return replace(trial, coverage=coverage)


def _negative_coverage(trial: Trial) -> Trial:
    coverage = trial.coverage.copy()
    coverage[1, 1] = -0.1
    return replace(trial, coverage=coverage)


def _nan_state(trial: Trial) -> Trial:
    state = trial.state.copy()
    state[0, 3] = np.nan
    return replace(trial, state=state)


def _inf_keypoint(trial: Trial) -> Trial:
    keypoint = trial.keypoint.copy()
    keypoint[1, 4, 0] = np.inf
    return replace(trial, keypoint=keypoint)


def _short_success(trial: Trial) -> Trial:
    return replace(trial, success=trial.success[:-1])


def _unsuccessful(trial: Trial) -> Trial:
    success = trial.success.copy()
    success[-1] = False
    return replace(trial, success=success)


def _early_success(trial: Trial) -> Trial:
    success = trial.success.copy()
    success[0] = True
    return replace(trial, success=success)


def _float_obs(trial: Trial) -> Trial:
    return replace(trial, obs=trial.obs.astype(np.float32))


def _bad_target(trial: Trial) -> Trial:
    return replace(trial, target=np.array(2, np.int8))


def _bad_target_poses(trial: Trial) -> Trial:
    return replace(trial, target_poses=np.zeros((3, 3), np.float32))


def _bad_contacts(trial: Trial) -> Trial:
    return replace(trial, n_contacts=np.zeros(trial.n_contacts.shape[0], np.int32))


def _bad_obs_size(trial: Trial) -> Trial:
    return replace(trial, obs=np.zeros((trial.obs.shape[0], 95, 95, 3), np.uint8))


BAD_TRIALS = [
    pytest.param(_short_success, id="short-success"),
    pytest.param(_unsuccessful, id="unsuccessful"),
    pytest.param(_early_success, id="early-success"),
    pytest.param(_float_obs, id="float-obs"),
    pytest.param(_bad_coverage, id="coverage-above-one"),
    pytest.param(_negative_coverage, id="coverage-below-zero"),
    pytest.param(_nan_state, id="nan-state"),
    pytest.param(_inf_keypoint, id="inf-keypoint"),
    pytest.param(_bad_target, id="bad-target"),
    pytest.param(_bad_target_poses, id="bad-target-poses"),
    pytest.param(_bad_contacts, id="bad-contacts"),
    pytest.param(_bad_obs_size, id="bad-obs-size"),
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


def test_scan_trials_across_both_colors(tmp_path: Path) -> None:
    save_trial(tmp_path, make_trial(2, PINK, seed=1), "a")
    save_trial(tmp_path, make_trial(5, GREEN, seed=2), "b")
    save_trial(tmp_path, make_trial(1, PINK, seed=3), "c")
    records = scan_trials(tmp_path)
    assert [(record.episode_id, record.target) for record in records] == [("a", PINK), ("c", PINK), ("b", GREEN)]
    first = records[0]
    assert first.path == tmp_path / "pink" / "a.npz"
    assert first.steps == 2
    assert first.success is True
    assert first.seed == 1
    assert first.created_at == CREATED


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

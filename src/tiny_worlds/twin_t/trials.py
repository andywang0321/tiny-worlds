"""The raw TwinT v2 trial format: one uncompressed ``.npz`` per successful trial."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .schema import (
    GREEN,
    PINK,
    SCHEMA_VERSION,
    TARGET_NAMES,
    TRIAL_ARRAYS,
    TRIAL_EPISODE_ARRAYS,
    target_index,
    target_name,
)
from .types import FloatArray

_MAX_EPISODE_ID_BYTES = 32
_SAFE_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUIRED_KEYS = TRIAL_ARRAYS + TRIAL_EPISODE_ARRAYS + ("meta",)


class TrialInvalid(ValueError):
    """A trial or trial file does not match the frozen raw schema."""


@dataclass(frozen=True)
class TrialMeta:
    schema_version: int
    seed: int
    created_at: str
    active_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.created_at, str):
            raise TypeError("created_at must be an ISO 8601 string")
        parse_created_at(self.created_at)
        if isinstance(self.active_seconds, bool) or not isinstance(self.active_seconds, (int, float)):
            raise TypeError("active_seconds must be a number")
        seconds = float(self.active_seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("active_seconds must be finite and non-negative")
        object.__setattr__(self, "active_seconds", seconds)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialMeta:
        try:
            return cls(
                payload["schema_version"],  # type: ignore[arg-type]
                payload["seed"],  # type: ignore[arg-type]
                payload["created_at"],  # type: ignore[arg-type]
                payload["active_seconds"],  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise ValueError(f"trial meta is missing {error}") from error


@dataclass(frozen=True)
class Trial:
    state: FloatArray
    target: NDArray[np.int8]
    target_poses: FloatArray
    meta: TrialMeta

    @property
    def steps(self) -> int:
        return int(self.state.shape[0]) - 1

    @property
    def action(self) -> FloatArray:
        """The realized pointer target per row; actions are never stored."""
        return derive_actions(self.state)


@dataclass(frozen=True)
class TrialRecord:
    path: Path
    episode_id: str
    target: int
    steps: int
    seed: int
    created_at: str


def derive_actions(state: NDArray[np.floating]) -> FloatArray:
    """Derive BC actions from a state trajectory, copying the terminal row.

    ``action[:-1]`` is the next realized agent position and ``action[-1]`` repeats the
    terminal one, so the action array has exactly as many rows as ``state``.
    """
    rows = np.asarray(state)
    actions = np.empty((rows.shape[0], 2), dtype=np.float32)
    actions[:-1] = rows[1:, :2]
    actions[-1] = rows[-1, :2]
    return actions


def parse_created_at(text: str) -> float:
    """Parse an ISO 8601 creation time into unix seconds."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("created_at must be a non-empty ISO 8601 string")
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError as error:
        raise ValueError(f"created_at must be ISO 8601, got {text!r}") from error
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()


def format_created_at(seconds: float) -> str:
    """Format unix seconds as a UTC ISO 8601 string."""
    if not math.isfinite(seconds):
        raise ValueError("created_at seconds must be finite")
    moment = datetime.fromtimestamp(seconds, tz=UTC)
    timespec = "microseconds" if moment.microsecond else "seconds"
    return moment.isoformat(timespec=timespec).replace("+00:00", "Z")


def trial_path(data_dir: str | Path, target: int, episode_id: str) -> Path:
    if isinstance(target, bool) or target not in (PINK, GREEN):
        raise ValueError(f"target must be {PINK} or {GREEN}, got {target!r}")
    _check_episode_id(episode_id)
    return Path(data_dir) / target_name(int(target)) / f"{episode_id}.npz"


def _check_episode_id(episode_id: str) -> None:
    if not isinstance(episode_id, str) or not _SAFE_EPISODE_ID.fullmatch(episode_id):
        raise ValueError(f"episode_id must be a safe file name with no separators, got {episode_id!r}")
    if len(episode_id.encode("utf-8")) > _MAX_EPISODE_ID_BYTES:
        raise ValueError(f"episode_id must be at most {_MAX_EPISODE_ID_BYTES} bytes, got {episode_id!r}")


def validate_trial(trial: Trial) -> None:
    state = np.asarray(trial.state)
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 5:
        raise TrialInvalid(f"state must have shape (T+1, 5) with T+1 >= 1, got {state.shape}")
    if state.dtype != np.dtype(np.float32):
        raise TrialInvalid(f"state dtype must be float32, got {state.dtype}")
    if not np.isfinite(state).all():
        raise TrialInvalid("state must contain only finite values")
    target = np.asarray(trial.target)
    if target.shape != () or target.dtype != np.dtype(np.int8):
        raise TrialInvalid(f"target must be a 0-d int8, got {target.dtype} {target.shape}")
    if int(target) not in (PINK, GREEN):
        raise TrialInvalid(f"target must be {PINK} or {GREEN}, got {int(target)}")
    _check_target_poses(trial.target_poses)


def _check_target_poses(value: object) -> None:
    poses = np.asarray(value)
    if poses.shape != (2, 3) or poses.dtype != np.dtype(np.float32):
        raise TrialInvalid(f"target_poses must be a (2, 3) float32 array, got {poses.dtype} {poses.shape}")
    if not np.isfinite(poses).all():
        raise TrialInvalid("target_poses must contain only finite values")


def save_trial(data_dir: str | Path, trial: Trial, episode_id: str) -> Path:
    validate_trial(trial)
    path = trial_path(data_dir, int(trial.target), episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": np.asarray(trial.state),
        "target": np.asarray(trial.target),
        "target_poses": np.asarray(trial.target_poses),
        "meta": np.asarray(json.dumps(asdict(trial.meta), separators=(",", ":"), allow_nan=False)),
    }
    _write_atomic(path, payload)
    return path


def _write_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_trial(path: str | Path) -> Trial:
    location = Path(path)
    arrays, meta_text = _read_arrays(location)
    meta = _parse_meta(meta_text, location)
    _require_schema_version(meta, location)
    return _validate_loaded(Trial(**arrays, meta=meta), location)


def load_legacy_trial(path: str | Path) -> Trial:
    """Convert a v1 raw trial to the v2 state-only format.

    This is the only migration helper: the old ``obs``/``action``/``coverage``/
    ``success``/``keypoint``/``n_contacts`` members are ignored, and the returned
    trial carries only the v2 arrays with ``schema_version`` set to 2.
    """
    location = Path(path)
    arrays, meta_text = _read_arrays(location)
    meta = _legacy_meta(meta_text, location)
    return _validate_loaded(Trial(**arrays, meta=meta), location)


def _legacy_meta(meta_text: str, path: Path) -> TrialMeta:
    old = _parse_meta(meta_text, path)
    return TrialMeta(SCHEMA_VERSION, old.seed, old.created_at, old.active_seconds)


def _parse_meta(meta_text: str, path: Path) -> TrialMeta:
    try:
        return TrialMeta.from_dict(json.loads(meta_text))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise TrialInvalid(f"trial {path} has invalid meta: {error}") from error


def _require_schema_version(meta: TrialMeta, path: Path) -> None:
    if meta.schema_version != SCHEMA_VERSION:
        raise TrialInvalid(f"trial {path} has schema_version {meta.schema_version}, expected {SCHEMA_VERSION}")


def _validate_loaded(trial: Trial, path: Path) -> Trial:
    try:
        validate_trial(trial)
    except TrialInvalid as error:
        raise TrialInvalid(f"trial {path} is invalid: {error}") from error
    return trial


def _read_arrays(path: Path) -> tuple[dict[str, np.ndarray], str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in _REQUIRED_KEYS if name not in archive.files]
            if missing:
                raise TrialInvalid(f"trial {path} is missing arrays {missing}")
            arrays = {name: np.array(archive[name]) for name in TRIAL_ARRAYS + TRIAL_EPISODE_ARRAYS}
            meta_text = _meta_text(archive["meta"], path)
    except TrialInvalid:
        raise
    except Exception as error:
        raise TrialInvalid(f"cannot read trial {path}: {error}") from error
    return arrays, meta_text


def _meta_text(array: np.ndarray, path: Path) -> str:
    if array.shape != () or array.dtype.kind != "U":
        raise TrialInvalid(f"trial {path} meta must be a 0-d unicode JSON string, got {array.dtype} {array.shape}")
    return str(array.item())


def scan_trials(data_dir: str | Path) -> list[TrialRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise TrialInvalid(f"trial directory not found: {root}")
    if not root.is_dir():
        raise TrialInvalid(f"trial path is not a directory: {root}")
    records: list[TrialRecord] = []
    for color in TARGET_NAMES:
        directory = root / color
        if directory.exists() and not directory.is_dir():
            raise TrialInvalid(f"trial color path is not a directory: {directory}")
        target = target_index(color)
        records.extend(_read_record(path, target) for path in sorted(directory.glob("*.npz")))
    return records


def _read_record(path: Path, expected_target: int) -> TrialRecord:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in ("state", "target", "meta") if name not in archive.files]
            if missing:
                raise TrialInvalid(f"trial {path} is missing arrays {missing}")
            state = np.asarray(archive["state"])
            target = np.asarray(archive["target"])
            meta_text = _meta_text(archive["meta"], path)
    except TrialInvalid:
        raise
    except Exception as error:
        raise TrialInvalid(f"cannot scan trial {path}: {error}") from error
    _check_record_scalars(path, state, target, expected_target)
    meta = _parse_meta(meta_text, path)
    _require_schema_version(meta, path)
    return TrialRecord(
        path=path,
        episode_id=path.stem,
        target=int(target),
        steps=int(state.shape[0]) - 1,
        seed=meta.seed,
        created_at=meta.created_at,
    )


def _check_record_scalars(path: Path, state: np.ndarray, target: np.ndarray, expected_target: int) -> None:
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 5 or state.dtype != np.dtype(np.float32):
        raise TrialInvalid(f"trial {path} state must be a (T+1, 5) float32 array, got {state.dtype} {state.shape}")
    if target.shape != () or target.dtype != np.dtype(np.int8):
        raise TrialInvalid(f"trial {path} target must be a 0-d int8, got {target.dtype} {target.shape}")
    if int(target) != expected_target:
        raise TrialInvalid(f"trial {path} target is {int(target)}, outside its color directory")

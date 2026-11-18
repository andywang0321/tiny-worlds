"""The raw TwinT trial format: one uncompressed ``.npz`` per successful trial."""

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
    OBSERVATION_SIZE,
    PINK,
    TARGET_NAMES,
    TRIAL_ARRAYS,
    TRIAL_EPISODE_ARRAYS,
    target_index,
    target_name,
)
from .types import BoolArray, FloatArray, Int32Array, UInt8Array

_MAX_EPISODE_ID_BYTES = 32
_SAFE_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUIRED_KEYS = TRIAL_ARRAYS + TRIAL_EPISODE_ARRAYS + ("meta",)
_FRAME_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[np.generic]]] = {
    "obs": ((OBSERVATION_SIZE, OBSERVATION_SIZE, 3), np.dtype(np.uint8)),
    "state": ((5,), np.dtype(np.float32)),
    "action": ((2,), np.dtype(np.float32)),
    "coverage": ((2,), np.dtype(np.float32)),
    "success": ((), np.dtype(np.bool_)),
    "keypoint": ((9, 2), np.dtype(np.float32)),
    "n_contacts": ((1,), np.dtype(np.int32)),
}


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
    obs: UInt8Array
    state: FloatArray
    action: FloatArray
    coverage: FloatArray
    success: BoolArray
    keypoint: FloatArray
    n_contacts: Int32Array
    target: NDArray[np.int8]
    target_poses: FloatArray
    meta: TrialMeta

    @property
    def steps(self) -> int:
        return int(self.obs.shape[0]) - 1


@dataclass(frozen=True)
class TrialRecord:
    path: Path
    episode_id: str
    target: int
    steps: int
    success: bool
    seed: int
    created_at: str


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
    obs_shape = np.shape(trial.obs)
    if len(obs_shape) != 4 or obs_shape[0] < 1:
        raise TrialInvalid(f"obs must have shape (T+1, {OBSERVATION_SIZE}, {OBSERVATION_SIZE}, 3), got {obs_shape}")
    rows = int(obs_shape[0])
    for name, (tail, dtype) in _FRAME_SPECS.items():
        array = np.asarray(getattr(trial, name))
        if array.shape != (rows, *tail):
            raise TrialInvalid(f"{name} must have shape {(rows, *tail)}, got {array.shape}")
        if array.dtype != dtype:
            raise TrialInvalid(f"{name} dtype must be {dtype}, got {array.dtype}")
    target = np.asarray(trial.target)
    if target.shape != () or target.dtype != np.dtype(np.int8):
        raise TrialInvalid(f"target must be a 0-d int8, got {target.dtype} {target.shape}")
    if int(target) not in (PINK, GREEN):
        raise TrialInvalid(f"target must be {PINK} or {GREEN}, got {int(target)}")
    poses = np.asarray(trial.target_poses)
    if poses.shape != (2, 3) or poses.dtype != np.dtype(np.float32):
        raise TrialInvalid(f"target_poses must be a (2, 3) float32 array, got {poses.dtype} {poses.shape}")
    _check_values(trial)


def _check_values(trial: Trial) -> None:
    for name in ("state", "action", "coverage", "keypoint", "target_poses"):
        if not np.isfinite(np.asarray(getattr(trial, name))).all():
            raise TrialInvalid(f"{name} must contain only finite values")
    coverage = np.asarray(trial.coverage)
    if np.any((coverage < 0.0) | (coverage > 1.0)):
        raise TrialInvalid("coverage must be in [0, 1]")
    if np.any(np.asarray(trial.n_contacts) < 0):
        raise TrialInvalid("n_contacts must be non-negative")
    success = np.asarray(trial.success)
    if not bool(success[-1]):
        raise TrialInvalid("success[-1] must be true: only successful trials are written")
    if np.any(success[:-1]):
        raise TrialInvalid("success may be true only on the terminal row")


def save_trial(data_dir: str | Path, trial: Trial, episode_id: str) -> Path:
    validate_trial(trial)
    path = trial_path(data_dir, int(trial.target), episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(getattr(trial, name)) for name in TRIAL_ARRAYS}
    payload["target"] = np.asarray(trial.target)
    payload["target_poses"] = np.asarray(trial.target_poses)
    payload["meta"] = np.asarray(json.dumps(asdict(trial.meta), separators=(",", ":"), allow_nan=False))
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
    try:
        meta = TrialMeta.from_dict(json.loads(meta_text))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise TrialInvalid(f"trial {location} has invalid meta: {error}") from error
    trial = Trial(**arrays, meta=meta)
    try:
        validate_trial(trial)
    except TrialInvalid as error:
        raise TrialInvalid(f"trial {location} is invalid: {error}") from error
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
            missing = [name for name in ("target", "success", "meta") if name not in archive.files]
            if missing:
                raise TrialInvalid(f"trial {path} is missing arrays {missing}")
            target = np.asarray(archive["target"])
            success = np.asarray(archive["success"])
            meta_text = _meta_text(archive["meta"], path)
    except TrialInvalid:
        raise
    except Exception as error:
        raise TrialInvalid(f"cannot scan trial {path}: {error}") from error
    _check_record_scalars(path, target, success, expected_target)
    try:
        meta = TrialMeta.from_dict(json.loads(meta_text))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise TrialInvalid(f"trial {path} has invalid meta: {error}") from error
    return TrialRecord(
        path=path,
        episode_id=path.stem,
        target=int(target),
        steps=int(success.size) - 1,
        success=bool(success[-1]),
        seed=meta.seed,
        created_at=meta.created_at,
    )


def _check_record_scalars(path: Path, target: np.ndarray, success: np.ndarray, expected_target: int) -> None:
    if target.shape != () or target.dtype != np.dtype(np.int8):
        raise TrialInvalid(f"trial {path} target must be a 0-d int8, got {target.dtype} {target.shape}")
    if int(target) != expected_target:
        raise TrialInvalid(f"trial {path} target is {int(target)}, outside its color directory")
    if success.ndim != 1 or success.size < 1 or success.dtype != np.dtype(np.bool_):
        raise TrialInvalid(f"trial {path} success must be a non-empty 1-D bool array, got {success.dtype} {success.shape}")

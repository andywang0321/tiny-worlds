"""Pack raw TwinT trials into one compressed ``.npz`` distribution file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG
from .schema import (
    FPS,
    GREEN,
    PINK,
    SCHEMA_VERSION,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SUCCESS_THRESHOLD,
    TARGET_COLORS,
    TARGET_NAMES,
    target_name,
)
from .trials import (
    Trial,
    TrialRecord,
    format_created_at,
    load_trial,
    parse_created_at,
    scan_trials,
)

_MAX_EPISODE_ID_BYTES = 32

# Per-frame members of the packed .npz.
DATA_TAILS: dict[str, tuple[int, ...]] = {"state": (5,)}
DATA_DTYPES: dict[str, np.dtype[np.generic]] = {"state": np.dtype(np.float32)}

# Per-episode members of the packed .npz.
EPISODE_TAILS: dict[str, tuple[int, ...]] = {
    "episode_ends": (),
    "episode_target": (),
    "episode_target_poses": (2, 3),
    "episode_ids": (),
    "episode_created": (),
    "episode_seed": (),
    "split": (),
}
EPISODE_DTYPES: dict[str, np.dtype[np.generic]] = {
    "episode_ends": np.dtype(np.int64),
    "episode_target": np.dtype(np.int8),
    "episode_target_poses": np.dtype(np.float32),
    "episode_ids": np.dtype("S32"),
    "episode_created": np.dtype(np.float64),
    "episode_seed": np.dtype(np.int64),
    "split": np.dtype(np.int8),
}


class PackError(RuntimeError):
    """A raw dataset cannot be packed into a valid distribution file."""


@dataclass(frozen=True)
class PackSummary:
    """What ``pack_trials`` wrote, and any tiny-data split warnings."""

    out_path: Path
    n_episodes: int
    n_frames: int
    target_counts: dict[str, int]
    split_counts: dict[str, int]
    sha256: str
    warnings: tuple[str, ...]


def pack_trials(
    data_dir: str | Path,
    out_path: str | Path,
    *,
    split_seed: int = 0,
    overwrite: bool = False,
) -> PackSummary:
    """Pack every raw trial under ``data_dir`` into the compressed ``.npz`` ``out_path``."""
    output = Path(out_path)
    _check_output(output, overwrite)
    records = scan_trials(data_dir)
    if not records:
        raise PackError(f"no trials found under {Path(data_dir)}")
    _check_episode_ids(records)
    split, warnings = _assign_splits(records, split_seed)
    payload, n_frames = _packed_payload(data_dir, records, split, split_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_packed(output, payload)
    return PackSummary(
        out_path=output,
        n_episodes=len(records),
        n_frames=n_frames,
        target_counts=_target_counts(records),
        split_counts=_split_counts(split),
        sha256=_sha256(output),
        warnings=warnings,
    )


def _check_output(output: Path, overwrite: bool) -> None:
    if output.suffix != ".npz":
        raise PackError(f"pack output must be a .npz file: {output}")
    if output.exists() and output.is_dir():
        raise PackError(f"pack output is a directory: {output}")
    if output.exists() and not overwrite:
        raise PackError(f"pack output already exists: {output}")


def _check_episode_ids(records: list[TrialRecord]) -> None:
    for record in records:
        encoded = record.episode_id.encode("utf-8")
        if not encoded or len(encoded) > _MAX_EPISODE_ID_BYTES:
            raise PackError(f"episode id must be 1..{_MAX_EPISODE_ID_BYTES} bytes: {record.path}")


def _assign_splits(records: list[TrialRecord], split_seed: int) -> tuple[NDArray[np.int8], tuple[str, ...]]:
    """Assign the 90/10 train/validation split, stratified by target color."""
    split = np.full(len(records), SPLIT_TRAIN, dtype=np.int8)
    rng = np.random.default_rng(split_seed)
    warnings: list[str] = []
    for target in (PINK, GREEN):
        indices = np.array([index for index, record in enumerate(records) if record.target == target])
        if indices.size == 0:
            continue
        order = rng.permutation(indices)
        n_train = int(indices.size * SPLIT_FRACTIONS[SPLIT_TRAIN])
        split[order[:n_train]] = SPLIT_TRAIN
        split[order[n_train:]] = SPLIT_VALIDATION
        for code, name in enumerate(SPLIT_NAMES):
            if not np.any(split[indices] == code):
                warnings.append(f"{target_name(target)} split {name!r} is empty")
    return split, tuple(warnings)


def _packed_payload(
    data_dir: str | Path,
    records: list[TrialRecord],
    split: NDArray[np.int8],
    split_seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    trials = _load_trials(records)
    ends = np.cumsum([trial.steps + 1 for trial in trials], dtype=np.int64)
    n_frames = int(ends[-1])
    payload = _array_payload(records, trials, split, ends)
    payload["meta"] = np.asarray(json.dumps(_root_meta(data_dir, records, split_seed, n_frames)))
    return payload, n_frames


def _load_trials(records: list[TrialRecord]) -> list[Trial]:
    trials: list[Trial] = []
    for record in records:
        trial = load_trial(record.path)
        if trial.steps != record.steps:
            raise PackError(f"trial changed while packing: {record.path}")
        trials.append(trial)
    return trials


def _array_payload(
    records: list[TrialRecord],
    trials: list[Trial],
    split: NDArray[np.int8],
    ends: NDArray[np.int64],
) -> dict[str, np.ndarray]:
    return {
        "state": np.concatenate([trial.state for trial in trials]),
        "episode_ends": ends,
        "episode_target": np.array([trial.target for trial in trials], dtype=np.int8),
        "episode_target_poses": np.stack([trial.target_poses for trial in trials]),
        "episode_ids": np.array([record.episode_id.encode("utf-8") for record in records], dtype="S32"),
        "episode_created": np.array([parse_created_at(trial.meta.created_at) for trial in trials], np.float64),
        "episode_seed": np.array([trial.meta.seed for trial in trials], np.int64),
        "split": split.astype(np.int8),
    }


def _root_meta(data_dir: str | Path, records: list[TrialRecord], split_seed: int, n_frames: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "target_colors": [list(color) for color in TARGET_COLORS],
        "target_bounds": list(DEFAULT_CONFIG.target_bounds),
        "target_min_separation": DEFAULT_CONFIG.target_min_separation,
        "success_threshold": SUCCESS_THRESHOLD,
        "created_at": format_created_at(time.time()),
        "source": str(Path(data_dir)),
        "split_seed": int(split_seed),
        "n_episodes": len(records),
        "n_frames": int(n_frames),
        "episode_counts": _target_counts(records),
    }


def _write_packed(output: Path, payload: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _target_counts(records: list[TrialRecord]) -> dict[str, int]:
    counts = {name: 0 for name in TARGET_NAMES}
    for record in records:
        counts[target_name(record.target)] += 1
    return counts


def _split_counts(split: NDArray[np.int8]) -> dict[str, int]:
    counts = np.bincount(split, minlength=len(SPLIT_NAMES))
    return {name: int(counts[code]) for code, name in enumerate(SPLIT_NAMES)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

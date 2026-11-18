"""Pack raw trials into the zarr v2 store inside one ``.zip`` file."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG
from .schema import (
    FPS,
    GREEN,
    OBSERVATION_SIZE,
    PINK,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SUCCESS_THRESHOLD,
    TARGET_COLORS,
    TARGET_NAMES,
    ZARR_DATA_ARRAYS,
    ZARR_META_ARRAYS,
    target_name,
)
from .trials import (
    Trial,
    TrialInvalid,
    TrialRecord,
    format_created_at,
    load_trial,
    parse_created_at,
    scan_trials,
)

_ZSTD_LEVEL = 5
_MAX_EPISODE_ID_BYTES = 32

DATA_TAILS: dict[str, tuple[int, ...]] = {
    "img": (OBSERVATION_SIZE, OBSERVATION_SIZE, 3),
    "state": (5,),
    "action": (2,),
    "coverage": (2,),
    "success": (),
    "keypoint": (9, 2),
    "n_contacts": (1,),
    "target": (),
    "target_poses": (2, 3),
}
DATA_DTYPES: dict[str, np.dtype[np.generic]] = {
    "img": np.dtype(np.uint8),
    "state": np.dtype(np.float32),
    "action": np.dtype(np.float32),
    "coverage": np.dtype(np.float32),
    "success": np.dtype(np.bool_),
    "keypoint": np.dtype(np.float32),
    "n_contacts": np.dtype(np.int32),
    "target": np.dtype(np.int8),
    "target_poses": np.dtype(np.float32),
}
META_DTYPES: dict[str, np.dtype[np.generic]] = {
    "episode_ends": np.dtype(np.int64),
    "episode_target": np.dtype(np.int8),
    "episode_success": np.dtype(np.bool_),
    "episode_steps": np.dtype(np.int32),
    "episode_ids": np.dtype("S32"),
    "episode_created": np.dtype(np.float64),
    "episode_seed": np.dtype(np.int64),
    "split": np.dtype(np.int8),
}
_TRIAL_SOURCES = {name: ("obs" if name == "img" else name) for name in ZARR_DATA_ARRAYS}


class PackError(RuntimeError):
    """A raw dataset cannot be packed into a valid zarr store."""


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
    """Pack every raw trial under ``data_dir`` into the zarr v2 zip ``out_path``."""
    output = Path(out_path)
    _check_output(output, overwrite)
    records = scan_trials(data_dir)
    if not records:
        raise PackError(f"no trials found under {Path(data_dir)}")
    _check_episode_ids(records)
    split, warnings = _assign_splits(records, split_seed)
    n_frames = sum(record.steps + 1 for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    attrs = _root_attrs(data_dir, records, split_seed, n_frames)
    _write_store(output, records, split, attrs)
    return PackSummary(
        out_path=output,
        n_episodes=len(records),
        n_frames=n_frames,
        target_counts=_target_counts(records),
        split_counts=_split_counts(split),
        sha256=_sha256(output),
        warnings=warnings,
    )


def _write_store(
    output: Path, records: list[TrialRecord], split: NDArray[np.int8], attrs: dict[str, object]
) -> None:
    n_frames = sum(record.steps + 1 for record in records)
    temporary_zip = output.with_name(f".{output.name}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", suffix=".zarr", dir=output.parent) as store:
            group = zarr.open_group(store, mode="w", zarr_format=2)
            arrays = _create_arrays(group, records)
            written = _write_all(arrays, records, split)
            if written != n_frames:
                raise PackError(f"packed {written} frames, expected {n_frames} for {output}")
            group.attrs.update(attrs)
            _write_zip(Path(store), temporary_zip)
        os.replace(temporary_zip, output)
    except (PackError, TrialInvalid):
        raise
    except Exception as error:
        raise PackError(f"failed to pack {output}: {error}") from error
    finally:
        temporary_zip.unlink(missing_ok=True)


def _check_output(output: Path, overwrite: bool) -> None:
    if output.suffix != ".zip":
        raise PackError(f"pack output must be a .zip file: {output}")
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
    split = np.full(len(records), SPLIT_TRAIN, dtype=np.int8)
    rng = np.random.default_rng(split_seed)
    warnings: list[str] = []
    for target in (PINK, GREEN):
        indices = np.array([index for index, record in enumerate(records) if record.target == target])
        if indices.size == 0:
            continue
        order = rng.permutation(indices)
        n_train = int(indices.size * 0.8)
        n_validation = int(indices.size * 0.1)
        split[order[:n_train]] = SPLIT_TRAIN
        split[order[n_train : n_train + n_validation]] = SPLIT_VALIDATION
        split[order[n_train + n_validation :]] = SPLIT_TEST
        for code, name in enumerate(SPLIT_NAMES):
            if not np.any(split[indices] == code):
                warnings.append(f"{target_name(target)} split {name!r} is empty")
    return split, tuple(warnings)


def _target_counts(records: list[TrialRecord]) -> dict[str, int]:
    counts = {name: 0 for name in TARGET_NAMES}
    for record in records:
        counts[target_name(record.target)] += 1
    return counts


def _split_counts(split: NDArray[np.int8]) -> dict[str, int]:
    counts = np.bincount(split, minlength=len(SPLIT_NAMES))
    return {name: int(counts[code]) for code, name in enumerate(SPLIT_NAMES)}


def _create_arrays(group: zarr.Group, records: list[TrialRecord]) -> dict[str, zarr.Array]:
    n_frames = sum(record.steps + 1 for record in records)
    arrays: dict[str, zarr.Array] = {}
    for name in ZARR_DATA_ARRAYS:
        tail = DATA_TAILS[name]
        shape = (n_frames, *tail)
        chunks = (1, *tail) if name == "img" else (min(n_frames, 1024), *tail)
        arrays[name] = _create_array(group, f"data/{name}", shape, (chunks, DATA_DTYPES[name]))
    for name in ZARR_META_ARRAYS:
        shape = (len(records),)
        arrays[name] = _create_array(group, f"meta/{name}", shape, (shape, META_DTYPES[name]))
    return arrays


def _create_array(
    group: zarr.Group, key: str, shape: tuple[int, ...], spec: tuple[tuple[int, ...], np.dtype[np.generic]]
) -> zarr.Array:
    chunks, dtype = spec
    return group.create_array(
        key, shape=shape, chunks=chunks, dtype=dtype, compressors=[numcodecs.Zstd(level=_ZSTD_LEVEL)]
    )


def _write_all(
    arrays: dict[str, zarr.Array], records: list[TrialRecord], split: NDArray[np.int8]
) -> int:
    offset = 0
    for index, record in enumerate(records):
        trial = load_trial(record.path)
        if trial.steps + 1 != record.steps + 1:
            raise PackError(f"trial changed while packing: {record.path}")
        end = _write_trial(arrays, trial, offset)
        arrays["episode_ends"][index] = end
        arrays["episode_target"][index] = trial.target
        arrays["episode_success"][index] = trial.success[-1]
        arrays["episode_steps"][index] = trial.steps
        arrays["episode_ids"][index] = record.episode_id.encode("utf-8")
        arrays["episode_created"][index] = parse_created_at(trial.meta.created_at)
        arrays["episode_seed"][index] = trial.meta.seed
        arrays["split"][index] = int(split[index])
        offset = end
    return offset


def _write_trial(arrays: dict[str, zarr.Array], trial: Trial, offset: int) -> int:
    end = offset + trial.steps + 1
    for name, source in _TRIAL_SOURCES.items():
        value = np.asarray(getattr(trial, source))
        arrays[name][offset:end] = np.broadcast_to(value, arrays[name][offset:end].shape)
    return end


def _root_attrs(
    data_dir: str | Path, records: list[TrialRecord], split_seed: int, n_frames: int
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "target_colors": [list(color) for color in TARGET_COLORS],
        "target_bounds": list(DEFAULT_CONFIG.target_bounds),
        "target_min_separation": DEFAULT_CONFIG.target_min_separation,
        "success_threshold": SUCCESS_THRESHOLD,
        "observation_size": OBSERVATION_SIZE,
        "created_at": format_created_at(time.time()),
        "source": str(Path(data_dir)),
        "split_seed": int(split_seed),
        "n_episodes": len(records),
        "n_frames": int(n_frames),
        "episode_counts": _target_counts(records),
    }


def _write_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source).as_posix())
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

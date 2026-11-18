"""Load and unpack the packed TwinT zarr dataset."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import zarr
from numpy.typing import NDArray
from zarr.storage import ZipStore

from .pack import DATA_DTYPES, DATA_TAILS, META_DTYPES
from .schema import FPS, SCHEMA_VERSION, TRIAL_ARRAYS, ZARR_DATA_ARRAYS, ZARR_META_ARRAYS
from .trials import Trial, TrialMeta, format_created_at, save_trial


class DatasetInvalid(ValueError):
    """A packed dataset is missing arrays or does not match the packed schema."""


class UnpackError(RuntimeError):
    """A packed dataset cannot be restored to raw trial files."""


@dataclass(frozen=True)
class Episode:
    """One episode's arrays and labels, materialized from the zarr store."""

    img: NDArray[np.uint8]
    state: NDArray[np.float32]
    action: NDArray[np.float32]
    coverage: NDArray[np.float32]
    success: NDArray[np.bool_]
    keypoint: NDArray[np.float32]
    n_contacts: NDArray[np.int32]
    target: int
    target_poses: NDArray[np.float32]
    episode_id: str
    split: int
    steps: int

    @property
    def obs(self) -> NDArray[np.uint8]:
        """Alias for ``img``, matching the raw trial key."""
        return self.img


_ARRAY_KEYS = {name: f"data/{name}" for name in ZARR_DATA_ARRAYS}
_ARRAY_KEYS |= {name: f"meta/{name}" for name in ZARR_META_ARRAYS}


class TwinTDataset:
    """A lazily read packed TwinT store, either a ``.zip`` or a zarr directory."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._closed = False
        self._store: ZipStore | None = None
        try:
            self._store, self._root = _open_root(self._path)
            self._arrays = _require_arrays(self._root)
            self._ends = _validate(self._arrays)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._store is not None:
            self._store.close()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    closed: bool = property(lambda self: self._closed)
    episode_count: int = property(lambda self: len(self._ends))
    frame_count: int = property(lambda self: int(self._ends[-1]))

    img: zarr.Array = property(lambda self: self._array("data/img"))
    state: zarr.Array = property(lambda self: self._array("data/state"))
    action: zarr.Array = property(lambda self: self._array("data/action"))
    coverage: zarr.Array = property(lambda self: self._array("data/coverage"))
    success: zarr.Array = property(lambda self: self._array("data/success"))
    keypoint: zarr.Array = property(lambda self: self._array("data/keypoint"))
    n_contacts: zarr.Array = property(lambda self: self._array("data/n_contacts"))
    target: zarr.Array = property(lambda self: self._array("data/target"))
    target_poses: zarr.Array = property(lambda self: self._array("data/target_poses"))
    episode_ends: zarr.Array = property(lambda self: self._array("meta/episode_ends"))
    episode_target: zarr.Array = property(lambda self: self._array("meta/episode_target"))
    episode_success: zarr.Array = property(lambda self: self._array("meta/episode_success"))
    episode_steps: zarr.Array = property(lambda self: self._array("meta/episode_steps"))
    episode_ids: zarr.Array = property(lambda self: self._array("meta/episode_ids"))
    episode_created: zarr.Array = property(lambda self: self._array("meta/episode_created"))
    episode_seed: zarr.Array = property(lambda self: self._array("meta/episode_seed"))
    split: zarr.Array = property(lambda self: self._array("meta/split"))

    def episode(self, index: int) -> Episode:
        self._require_open()
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.episode_count:
            raise DatasetInvalid(f"episode index out of range: {index!r}")
        start = 0 if index == 0 else int(self._ends[index - 1])
        end = int(self._ends[index])
        return Episode(
            img=np.asarray(self.img[start:end]),
            state=np.asarray(self.state[start:end]),
            action=np.asarray(self.action[start:end]),
            coverage=np.asarray(self.coverage[start:end]),
            success=np.asarray(self.success[start:end]),
            keypoint=np.asarray(self.keypoint[start:end]),
            n_contacts=np.asarray(self.n_contacts[start:end]),
            target=int(self.episode_target[index]),
            target_poses=np.asarray(self.target_poses[start]),
            episode_id=self._episode_id(index),
            split=int(self.split[index]),
            steps=int(self.episode_steps[index]),
        )

    def episodes(self, split: int | None = None) -> Iterator[Episode]:
        if split is not None and (isinstance(split, bool) or split not in (0, 1, 2)):
            raise DatasetInvalid(f"split must be None, 0, 1, or 2, got {split!r}")
        for index in range(self.episode_count):
            if split is None or int(self.split[index]) == split:
                yield self.episode(index)

    def to_numpy(self) -> dict[str, NDArray[np.generic]]:
        self._require_open()
        return {name: np.asarray(self._array(key)[:]) for name, key in _ARRAY_KEYS.items()}

    def _array(self, key: str) -> zarr.Array:
        self._require_open()
        return self._arrays[key]

    def _require_open(self) -> None:
        if self._closed:
            raise DatasetInvalid(f"dataset is closed: {self._path}")

    def _episode_id(self, index: int) -> str:
        raw = bytes(self.episode_ids[index]).rstrip(b"\x00")
        if b"\x00" in raw:
            raise DatasetInvalid(f"episode {index} id contains a null byte")
        try:
            episode_id = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DatasetInvalid(f"episode {index} has a non-UTF-8 id") from error
        if not episode_id:
            raise DatasetInvalid(f"episode {index} has an empty id")
        return episode_id


def load_dataset(path: str | Path) -> TwinTDataset:
    """Open a packed ``.zarr.zip`` or a zarr directory for lazy reads."""
    return TwinTDataset(path)


def _open_root(path: Path) -> tuple[ZipStore | None, zarr.Group]:
    if path.is_dir():
        try:
            return None, zarr.open_group(path, mode="r")
        except Exception as error:
            raise DatasetInvalid(f"cannot open dataset directory {path}: {error}") from error
    if path.is_file() and path.suffix == ".zip":
        try:
            store = ZipStore(path, mode="r")
        except Exception as error:
            raise DatasetInvalid(f"cannot open dataset {path}: {error}") from error
        try:
            return store, zarr.open_group(store, mode="r")
        except Exception as error:
            store.close()
            raise DatasetInvalid(f"cannot open dataset {path}: {error}") from error
    raise DatasetInvalid(f"dataset path is neither a .zip file nor a zarr directory: {path}")


def _require_arrays(root: zarr.Group) -> dict[str, zarr.Array]:
    arrays: dict[str, zarr.Array] = {}
    for group_name, names in (("data", ZARR_DATA_ARRAYS), ("meta", ZARR_META_ARRAYS)):
        for name in names:
            key = f"{group_name}/{name}"
            try:
                array = root[key]
            except KeyError as error:
                raise DatasetInvalid(f"dataset is missing required array {key!r}") from error
            if not isinstance(array, zarr.Array):
                raise DatasetInvalid(f"dataset entry {key!r} is not an array")
            arrays[key] = array
    return arrays


def _validate(arrays: dict[str, zarr.Array]) -> NDArray[np.int64]:
    ends = _check_ends(arrays["meta/episode_ends"])
    _check_data(arrays, int(ends[-1]))
    _check_meta(arrays, ends)
    return ends


def _check_ends(array: zarr.Array) -> NDArray[np.int64]:
    if array.dtype != np.dtype(np.int64) or array.ndim != 1 or array.shape[0] < 1:
        raise DatasetInvalid(
            f"array 'meta/episode_ends' must be a non-empty int64 vector, got {array.dtype} {array.shape}"
        )
    ends = np.asarray(array[:], dtype=np.int64)
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise DatasetInvalid("array 'meta/episode_ends' must be positive and strictly increasing")
    return ends


def _check_data(arrays: dict[str, zarr.Array], n_frames: int) -> None:
    for name, tail in DATA_TAILS.items():
        array = arrays[f"data/{name}"]
        expected = (n_frames, *tail)
        if array.shape != expected:
            raise DatasetInvalid(f"array 'data/{name}' must have shape {expected}, got {array.shape}")
        if array.dtype != DATA_DTYPES[name]:
            raise DatasetInvalid(f"array 'data/{name}' must have dtype {DATA_DTYPES[name]}, got {array.dtype}")


def _check_meta(arrays: dict[str, zarr.Array], ends: NDArray[np.int64]) -> None:
    episodes = len(ends)
    for name in ZARR_META_ARRAYS:
        array = arrays[f"meta/{name}"]
        if array.shape != (episodes,):
            raise DatasetInvalid(f"array 'meta/{name}' must have shape ({episodes},), got {array.shape}")
        if array.dtype != META_DTYPES[name]:
            raise DatasetInvalid(f"array 'meta/{name}' must have dtype {META_DTYPES[name]}, got {array.dtype}")
    steps = np.asarray(arrays["meta/episode_steps"][:])
    if not np.array_equal(steps, np.diff(ends, prepend=0) - 1):
        raise DatasetInvalid("array 'meta/episode_steps' must equal each episode's row count minus one")
    targets = np.asarray(arrays["meta/episode_target"][:])
    if np.any((targets < 0) | (targets > 1)):
        raise DatasetInvalid("array 'meta/episode_target' must contain only 0 and 1")
    splits = np.asarray(arrays["meta/split"][:])
    if np.any((splits < 0) | (splits > 2)):
        raise DatasetInvalid("array 'meta/split' must contain only 0, 1, and 2")


def unpack(zarr_path: str | Path, out_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write the packed episodes back out as raw trials under ``out_dir/<color>/``."""
    destination = Path(out_dir)
    if destination.exists() and not destination.is_dir():
        raise UnpackError(f"unpack destination is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise UnpackError(f"unpack destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with TwinTDataset(zarr_path) as dataset:
        fps = float(dataset._root.attrs.get("fps", FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise UnpackError(f"dataset fps must be positive, got {fps!r}")
        settings = (int(dataset._root.attrs.get("schema_version", SCHEMA_VERSION)), fps)
        for index in range(dataset.episode_count):
            episode = dataset.episode(index)
            trial = _episode_trial(dataset, episode, index, settings)
            paths.append(save_trial(destination, trial, episode.episode_id))
    return paths


def _episode_trial(dataset: TwinTDataset, episode: Episode, index: int, settings: tuple[int, float]) -> Trial:
    schema_version, fps = settings
    meta = TrialMeta(
        schema_version=schema_version,
        seed=int(dataset.episode_seed[index]),
        created_at=format_created_at(float(dataset.episode_created[index])),
        active_seconds=episode.steps / fps,
    )
    arrays = {name: getattr(episode, "img" if name == "obs" else name) for name in TRIAL_ARRAYS}
    return Trial(**arrays, target=np.array(episode.target, np.int8), target_poses=episode.target_poses, meta=meta)

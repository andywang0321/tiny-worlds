"""Load and unpack the packed TwinT state-only ``.npz`` dataset."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from .pack import DATA_DTYPES, DATA_TAILS, EPISODE_DTYPES, EPISODE_TAILS
from .render import render_frames
from .schema import FPS, PACKED_ARRAYS, SCHEMA_VERSION, SPLIT_NAMES, SPLIT_TRAIN, SPLIT_VALIDATION
from .trials import Trial, TrialMeta, derive_actions, format_created_at, save_trial


class DatasetInvalid(ValueError):
    """A packed dataset is missing arrays or does not match the packed schema."""


class UnpackError(RuntimeError):
    """A packed dataset cannot be restored to raw trial files."""


@dataclass(frozen=True)
class Episode:
    """One episode's state trajectory and labels, materialized from the packed file."""

    state: NDArray[np.float32]
    action: NDArray[np.float32]
    target: int
    target_poses: NDArray[np.float32]
    episode_id: str
    split: int
    steps: int

    def frames(self, size: int) -> NDArray[np.uint8]:
        """Render this episode on demand, ``(steps + 1, size, size, 3)`` uint8."""
        return render_frames(self.state, self.target_poses, size)


class TwinTDataset:
    """An eagerly loaded packed TwinT ``.npz`` file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._closed = False
        members = _read_members(self._path)
        self._ends = _validate(members)
        self._arrays = members
        counts = np.diff(self._ends, prepend=0)
        self._action = _derive_all(members["state"], self._ends)
        self._target = np.repeat(members["episode_target"], counts)
        self._target_poses = np.repeat(members["episode_target_poses"], counts, axis=0)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def episode_count(self) -> int:
        return len(self._ends)

    @property
    def frame_count(self) -> int:
        return int(self._ends[-1])

    @property
    def state(self) -> NDArray[np.float32]:
        return self._array("state")

    @property
    def episode_ends(self) -> NDArray[np.int64]:
        return self._array("episode_ends")

    @property
    def episode_target(self) -> NDArray[np.int8]:
        return self._array("episode_target")

    @property
    def episode_target_poses(self) -> NDArray[np.float32]:
        return self._array("episode_target_poses")

    @property
    def episode_ids(self) -> NDArray[np.bytes_]:
        return self._array("episode_ids")

    @property
    def episode_created(self) -> NDArray[np.float64]:
        return self._array("episode_created")

    @property
    def episode_seed(self) -> NDArray[np.int64]:
        return self._array("episode_seed")

    @property
    def split(self) -> NDArray[np.int8]:
        return self._array("split")

    @property
    def action(self) -> NDArray[np.float32]:
        """Per-frame BC actions derived from consecutive states."""
        self._require_open()
        return self._action

    @property
    def target(self) -> NDArray[np.int8]:
        """Per-episode target broadcast to one value per frame."""
        self._require_open()
        return self._target

    @property
    def target_poses(self) -> NDArray[np.float32]:
        """Per-episode ``(2, 3)`` poses broadcast to one value per frame."""
        self._require_open()
        return self._target_poses

    @property
    def metadata(self) -> dict[str, object]:
        self._require_open()
        payload = json.loads(str(self._arrays["meta"].item()))
        if not isinstance(payload, dict):
            raise DatasetInvalid("packed meta must be a JSON object")
        return payload

    def episode(self, index: int) -> Episode:
        self._require_open()
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.episode_count:
            raise DatasetInvalid(f"episode index out of range: {index!r}")
        start = 0 if index == 0 else int(self._ends[index - 1])
        end = int(self._ends[index])
        return Episode(
            state=self._arrays["state"][start:end],
            action=self._action[start:end],
            target=int(self._arrays["episode_target"][index]),
            target_poses=self._arrays["episode_target_poses"][index],
            episode_id=self._episode_id(index),
            split=int(self._arrays["split"][index]),
            steps=end - start - 1,
        )

    def episodes(self, split: int | None = None) -> Iterator[Episode]:
        if split is not None and (isinstance(split, bool) or split not in (SPLIT_TRAIN, SPLIT_VALIDATION)):
            raise DatasetInvalid(f"split must be None or one of {SPLIT_NAMES}, got {split!r}")
        for index in range(self.episode_count):
            if split is None or int(self._arrays["split"][index]) == split:
                yield self.episode(index)

    def to_numpy(self) -> dict[str, NDArray[np.generic]]:
        self._require_open()
        return {name: np.array(array) for name, array in self._arrays.items()}

    def _array(self, name: str) -> NDArray[np.generic]:
        self._require_open()
        return self._arrays[name]

    def _require_open(self) -> None:
        if self._closed:
            raise DatasetInvalid(f"dataset is closed: {self._path}")

    def _episode_id(self, index: int) -> str:
        raw = bytes(self._arrays["episode_ids"][index]).rstrip(b"\x00")
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
    """Open a packed ``.npz`` file and load every member eagerly."""
    return TwinTDataset(path)


def _read_members(path: Path) -> dict[str, NDArray[np.generic]]:
    if not path.is_file():
        raise DatasetInvalid(f"dataset path is not a file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in PACKED_ARRAYS if name not in archive.files]
            if missing:
                raise DatasetInvalid(f"dataset {path} is missing arrays {missing}")
            return {name: np.array(archive[name]) for name in PACKED_ARRAYS}
    except DatasetInvalid:
        raise
    except Exception as error:
        raise DatasetInvalid(f"cannot open dataset {path}: {error}") from error


def _validate(members: dict[str, NDArray[np.generic]]) -> NDArray[np.int64]:
    ends = _check_ends(members["episode_ends"])
    expected = (int(ends[-1]), *DATA_TAILS["state"])
    state = members["state"]
    if state.shape != expected or state.dtype != DATA_DTYPES["state"]:
        raise DatasetInvalid(
            f"array 'state' must have shape {expected} {DATA_DTYPES['state']}, got {state.dtype} {state.shape}"
        )
    for name, tail in EPISODE_TAILS.items():
        array = members[name]
        shape = (len(ends), *tail)
        if array.shape != shape or array.dtype != EPISODE_DTYPES[name]:
            raise DatasetInvalid(
                f"array {name!r} must have shape {shape} {EPISODE_DTYPES[name]}, got {array.dtype} {array.shape}"
            )
    _check_labels(members)
    _check_meta(members["meta"])
    return ends


def _check_ends(array: NDArray[np.generic]) -> NDArray[np.int64]:
    if array.dtype != np.dtype(np.int64) or array.ndim != 1 or array.shape[0] < 1:
        raise DatasetInvalid(f"array 'episode_ends' must be a non-empty int64 vector, got {array.dtype} {array.shape}")
    ends = np.asarray(array, dtype=np.int64)
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise DatasetInvalid("array 'episode_ends' must be positive and strictly increasing")
    return ends


def _check_labels(members: dict[str, NDArray[np.generic]]) -> None:
    if np.any((members["episode_target"] < 0) | (members["episode_target"] > 1)):
        raise DatasetInvalid("array 'episode_target' must contain only 0 and 1")
    if np.any((members["split"] < 0) | (members["split"] > 1)):
        raise DatasetInvalid("array 'split' must contain only 0 and 1")


def _check_meta(array: NDArray[np.generic]) -> None:
    if array.shape != () or array.dtype.kind != "U":
        raise DatasetInvalid(f"member 'meta' must be a 0-d unicode JSON string, got {array.dtype} {array.shape}")


def _derive_all(state: NDArray[np.float32], ends: NDArray[np.int64]) -> NDArray[np.float32]:
    actions = np.empty((state.shape[0], 2), dtype=np.float32)
    start = 0
    for end in ends:
        stop = int(end)
        actions[start:stop] = derive_actions(state[start:stop])
        start = stop
    return actions


def unpack(packed_path: str | Path, out_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write the packed episodes back out as v2 raw trials under ``out_dir/<color>/``."""
    destination = Path(out_dir)
    _check_destination(destination, overwrite)
    destination.mkdir(parents=True, exist_ok=True)
    with load_dataset(packed_path) as dataset:
        return [_save_episode(dataset, index, destination) for index in range(dataset.episode_count)]


def _check_destination(destination: Path, overwrite: bool) -> None:
    if destination.exists() and not destination.is_dir():
        raise UnpackError(f"unpack destination is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise UnpackError(f"unpack destination is not empty: {destination}")


def _save_episode(dataset: TwinTDataset, index: int, destination: Path) -> Path:
    episode = dataset.episode(index)
    meta = TrialMeta(
        schema_version=SCHEMA_VERSION,
        seed=int(dataset.episode_seed[index]),
        created_at=format_created_at(float(dataset.episode_created[index])),
        active_seconds=episode.steps / FPS,
    )
    trial = Trial(
        state=episode.state,
        target=np.array(episode.target, dtype=np.int8),
        target_poses=episode.target_poses,
        meta=meta,
    )
    return save_trial(destination, trial, episode.episode_id)

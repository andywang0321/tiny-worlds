"""Minimal MNIST IDX loading."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

type UInt8Array = NDArray[np.uint8]
MNIST_DIR = Path.home() / "data" / "mnist"


def _open_binary(path: Path) -> Any:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _read_images(path: Path) -> UInt8Array:
    with _open_binary(path) as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file (magic={magic}).")
        images = np.frombuffer(file.read(), dtype=np.uint8)
    if images.size != count * rows * cols:
        raise ValueError(f"{path} has the wrong number of image bytes.")
    return images.reshape(count, rows, cols)


def _read_labels(path: Path) -> UInt8Array:
    with _open_binary(path) as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file (magic={magic}).")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"{path} has the wrong number of labels.")
    return labels


def _find_pair(directory: Path, prefix: str) -> tuple[Path, Path] | None:
    images = sorted(p for p in directory.glob(f"{prefix}-images*") if p.is_file())
    labels = sorted(p for p in directory.glob(f"{prefix}-labels*") if p.is_file())
    return (images[0], labels[0]) if images and labels else None


def load_mnist(
    mnist_dir: Path = MNIST_DIR,
) -> tuple[UInt8Array, UInt8Array]:
    pairs = [pair for pair in (_find_pair(mnist_dir, "train"), _find_pair(mnist_dir, "t10k")) if pair is not None]
    if not pairs:
        raise FileNotFoundError(f"No MNIST IDX files found in {mnist_dir}.")
    images = np.concatenate([_read_images(image) for image, _ in pairs])
    labels = np.concatenate([_read_labels(label) for _, label in pairs])
    return images, labels

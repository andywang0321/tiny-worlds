"""Augment raw TwinT v2 trials with exact task symmetries.

Usage::

    uv run --all-extras python scripts/augment_trials.py \\
        --data data --out data-aug --symmetries all

Reads the v2 ``.npz`` trials under ``--data`` and writes one transformed trial
per ``(episode, symmetry)`` under ``--out/<color>/``.  The new episode id is
``<episode_id>_<symmetry>``; if that exceeds the 32-byte episode-id limit it is
shortened and made unique with a short hash.  Counts are printed when done.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tiny_worlds.twin_t import (
    Symmetry,
    augment_trial,
    load_trial,
    save_trial,
    scan_trials,
    symmetries,
)
from tiny_worlds.twin_t.schema import TARGET_NAMES, target_name

_MAX_EPISODE_ID_BYTES = 32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="raw v2 trial directory")
    parser.add_argument("--out", type=Path, required=True, help="output trial directory")
    parser.add_argument(
        "--symmetries",
        default="all",
        help="'all' or a comma-separated list of names such as 'identity,swap,r1_mx'",
    )
    args = parser.parse_args(argv)
    try:
        chosen = _select_symmetries(args.symmetries)
    except ValueError as error:
        parser.error(str(error))

    records = scan_trials(args.data)
    counts: Counter[str] = Counter()
    for record in records:
        trial = load_trial(record.path)
        for symmetry in chosen:
            augmented = augment_trial(trial, symmetry)
            save_trial(args.out, augmented, _augmented_id(record.episode_id, symmetry))
            counts[target_name(int(augmented.target))] += 1
            counts[symmetry.name] += 1

    total = sum(counts[symmetry.name] for symmetry in chosen)
    print(f"read {len(records)} trials from {args.data}")
    print(f"wrote {total} trials to {args.out}")
    for name in TARGET_NAMES:
        print(f"  {name}: {counts[name]}")
    for symmetry in chosen:
        print(f"  {symmetry.name}: {counts[symmetry.name]}")
    return 0


def _select_symmetries(value: str) -> tuple[Symmetry, ...]:
    available = {symmetry.name: symmetry for symmetry in symmetries()}
    if value.strip() == "all":
        return tuple(available.values())
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--symmetries must be 'all' or a comma-separated list of names")
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"unknown symmetries: {', '.join(missing)}; choose from all, {', '.join(available)}")
    return tuple(available[name] for name in names)


def _augmented_id(episode_id: str, symmetry: Symmetry) -> str:
    suffix = f"_{symmetry.name}"
    candidate = f"{episode_id}{suffix}"
    if len(candidate.encode("utf-8")) <= _MAX_EPISODE_ID_BYTES:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:6]
    budget = _MAX_EPISODE_ID_BYTES - len(suffix.encode("utf-8")) - len(digest) - 1
    stem = episode_id.encode("utf-8")[:budget].decode("utf-8", "ignore")
    return f"{stem}_{digest}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())

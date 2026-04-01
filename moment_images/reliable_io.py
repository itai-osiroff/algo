"""Atomic persistence and first-valid fallback readers for sweep artifacts."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np


class FirstValidReadError(RuntimeError):
    """Raised when neither primary nor backup artifact could be loaded."""


def backup_path(path: Path) -> Path:
    """Return the deterministic backup path for an artifact."""

    return Path(f"{path}.bak")


def candidate_paths(path: Path) -> Tuple[Path, Path]:
    """Return primary and backup candidate locations in lookup order."""

    return path, backup_path(path)


def _atomic_replace_with_temp(
    path: Path,
    writer: Callable[[Path], None],
    create_backup: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        writer(tmp_path)
        if create_backup and path.exists():
            bak = backup_path(path)
            if bak.exists():
                bak.unlink()
            os.replace(path, bak)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_bytes_atomic(path: Path, data: bytes, create_backup: bool = True) -> None:
    """Atomically write bytes to ``path`` and rotate ``.bak`` when present."""

    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fp:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())

    _atomic_replace_with_temp(path, _writer, create_backup=create_backup)


def write_text_atomic(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    create_backup: bool = True,
) -> None:
    """Atomically persist text."""

    write_bytes_atomic(path, text.encode(encoding), create_backup=create_backup)


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    create_backup: bool = True,
) -> None:
    """Atomically write JSON payload."""

    text = json.dumps(payload, indent=indent, sort_keys=sort_keys)
    if not text.endswith("\n"):
        text += "\n"
    write_text_atomic(path, text, create_backup=create_backup)


def write_npy_atomic(
    path: Path,
    array: Any,
    *,
    allow_pickle: bool = False,
    create_backup: bool = True,
) -> None:
    """Atomically write an ``.npy`` array artifact."""

    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fp:
            np.save(fp, array, allow_pickle=allow_pickle)
            fp.flush()
            os.fsync(fp.fileno())

    _atomic_replace_with_temp(path, _writer, create_backup=create_backup)


def write_npz_atomic(
    path: Path,
    arrays: Dict[str, Any],
    *,
    compressed: bool = True,
    create_backup: bool = True,
) -> None:
    """Atomically write an ``.npz`` bundle."""

    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fp:
            if compressed:
                np.savez_compressed(fp, **arrays)
            else:
                np.savez(fp, **arrays)
            fp.flush()
            os.fsync(fp.fileno())

    _atomic_replace_with_temp(path, _writer, create_backup=create_backup)


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    *,
    create_backup: bool = True,
) -> None:
    """Atomically write CSV rows."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    write_text_atomic(path, buffer.getvalue(), create_backup=create_backup)


def _read_first_valid(
    path: Path,
    loader: Callable[[Path], Any],
    validator: Optional[Callable[[Any], bool]] = None,
) -> Tuple[Any, Path, List[str]]:
    errors: List[str] = []
    for candidate in candidate_paths(path):
        if not candidate.exists():
            errors.append(f"{candidate}: missing")
            continue
        try:
            value = loader(candidate)
            if validator is not None and not validator(value):
                raise ValueError("validation failed")
            return value, candidate, errors
        except Exception as exc:  # noqa: BLE001 - fallback reader aggregates failures.
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    joined = "; ".join(errors) if errors else "no candidates"
    raise FirstValidReadError(
        f"Failed to load primary or backup artifact for {path}: {joined}"
    )


def read_json_first_valid(
    path: Path,
    validator: Optional[Callable[[Any], bool]] = None,
) -> Tuple[Any, Path, List[str]]:
    """Load JSON from primary then backup path."""

    def _loader(candidate: Path) -> Any:
        with candidate.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    return _read_first_valid(path, _loader, validator=validator)


def read_npy_first_valid(
    path: Path,
    *,
    allow_pickle: bool = False,
    validator: Optional[Callable[[np.ndarray], bool]] = None,
) -> Tuple[np.ndarray, Path, List[str]]:
    """Load ``.npy`` from primary then backup path."""

    def _loader(candidate: Path) -> np.ndarray:
        with candidate.open("rb") as fp:
            return np.load(fp, allow_pickle=allow_pickle)

    return _read_first_valid(path, _loader, validator=validator)


def read_npz_first_valid(
    path: Path,
    *,
    allow_pickle: bool = False,
    validator: Optional[Callable[[Dict[str, np.ndarray]], bool]] = None,
) -> Tuple[Dict[str, np.ndarray], Path, List[str]]:
    """Load ``.npz`` from primary then backup path into a dictionary."""

    def _loader(candidate: Path) -> Dict[str, np.ndarray]:
        with np.load(candidate, allow_pickle=allow_pickle) as data:
            return {key: data[key] for key in data.files}

    return _read_first_valid(path, _loader, validator=validator)


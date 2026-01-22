"""Utilities for unpacking im_process workspace archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


DEFAULT_ARCHIVE_NAME = "workspace.npz"
DEFAULT_INDEX_NAME = "workspace_index.json"
DEFAULT_LOADER_NAME = "load_workspace.py"


def resolve_archive_path(input_path: Path) -> Path:
    if input_path.is_dir():
        candidate = input_path / DEFAULT_ARCHIVE_NAME
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Expected {DEFAULT_ARCHIVE_NAME} inside {input_path}, but it was not found."
        )
    if input_path.suffix != ".npz":
        candidate = input_path.with_suffix(".npz")
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Archive {input_path} not found and {candidate} does not exist."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Archive {input_path} does not exist.")
    return input_path


def infer_output_dir(archive_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return archive_path.parent / f"{archive_path.stem}_unpacked"


def summarize_array(value: np.ndarray, output_path: Path) -> Dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "path": output_path.name,
    }


def write_loader(output_dir: Path, index_name: str) -> None:
    loader_path = output_dir / DEFAULT_LOADER_NAME
    loader_code = f""""""Load unpacked im_process workspace arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


INDEX_NAME = "{index_name}"


def load_workspace(base_dir: Path | None = None) -> Dict[str, np.ndarray]:
    """Load all workspace arrays into a dictionary."""
    root = base_dir or Path(__file__).resolve().parent
    index_path = root / INDEX_NAME
    data = json.loads(index_path.read_text())
    arrays: Dict[str, np.ndarray] = {{}}
    for key, meta in data["arrays"].items():
        arrays[key] = np.load(root / meta["path"], allow_pickle=True)
    return arrays


if __name__ == "__main__":
    workspace = load_workspace()
    print(f"Loaded {{len(workspace)}} arrays: {{', '.join(sorted(workspace))}}")
"""
    loader_path.write_text(loader_code)


def unpack_workspace(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(archive_path, allow_pickle=True) as data:
        arrays: Dict[str, Dict[str, Any]] = {}
        for key in sorted(data.files):
            value = data[key]
            output_path = output_dir / f"{key}.npy"
            np.save(output_path, value)
            arrays[key] = summarize_array(value, output_path)

    index = {
        "source_archive": str(archive_path),
        "arrays": arrays,
    }
    (output_dir / DEFAULT_INDEX_NAME).write_text(json.dumps(index, indent=2))
    write_loader(output_dir, DEFAULT_INDEX_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack an im_process workspace .npz archive into individual .npy files "
            "with a loader script for interactive analysis."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a workspace .npz file or a directory containing workspace.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write unpacked arrays (default: <archive>_unpacked).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    archive_path = resolve_archive_path(args.input_path)
    output_dir = infer_output_dir(archive_path, args.output_dir)
    unpack_workspace(archive_path, output_dir)

    print(f"Unpacked workspace to: {output_dir}")
    print(f"Index: {output_dir / DEFAULT_INDEX_NAME}")
    print(f"Loader: {output_dir / DEFAULT_LOADER_NAME}")


if __name__ == "__main__":
    main()

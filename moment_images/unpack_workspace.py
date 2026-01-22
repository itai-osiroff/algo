"""Utilities for unpacking im_process workspace archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


DEFAULT_ARCHIVE_NAME = "workspace.npz"
DEFAULT_INDEX_NAME = "workspace_index.json"
DEFAULT_LOADER_NAME = "load_workspace.py"
DEBUG_STOP_MESSAGE = (
    "Workspace variables loaded. This is the debugger stopping point before saving."
)
WORKSPACE_VARIABLES: Dict[str, np.ndarray] = {}


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


def load_workspace_variables(archive_path: Path) -> Dict[str, np.ndarray]:
    """Load workspace variables one-by-one from a compressed .npz archive."""
    loaded: Dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=True) as data:
        for key in sorted(data.files):
            loaded[key] = data[key]
    return loaded


def save_unpacked_variables(
    variables: Dict[str, np.ndarray],
    output_dir: Path,
    archive_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Dict[str, Any]] = {}
    for key in sorted(variables):
        value = variables[key]
        output_path = output_dir / f"{key}.npy"
        np.save(output_path, value)
        arrays[key] = summarize_array(value, output_path)

    index = {
        "source_archive": str(archive_path),
        "arrays": arrays,
    }
    (output_dir / DEFAULT_INDEX_NAME).write_text(json.dumps(index, indent=2))
    write_loader(output_dir, DEFAULT_INDEX_NAME)


def build_gui() -> None:
    root = tk.Tk()
    root.title("Unpack im_process workspace")
    root.geometry("640x260")

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    save_var = tk.BooleanVar(value=True)

    def choose_input() -> None:
        path = filedialog.askopenfilename(
            title="Select workspace .npz archive",
            filetypes=[("NumPy archives", "*.npz"), ("All files", "*")],
        )
        if path:
            input_var.set(path)

    def choose_output() -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            output_var.set(path)

    def run_unpack() -> None:
        raw_input = input_var.get().strip()
        if not raw_input:
            messagebox.showerror("Missing input", "Please select a workspace .npz file.")
            return
        try:
            archive_path = resolve_archive_path(Path(raw_input))
        except FileNotFoundError as exc:
            messagebox.showerror("Archive not found", str(exc))
            return

        output_dir_value = output_var.get().strip()
        output_dir = infer_output_dir(
            archive_path, Path(output_dir_value) if output_dir_value else None
        )

        global WORKSPACE_VARIABLES
        WORKSPACE_VARIABLES = load_workspace_variables(archive_path)
        print(DEBUG_STOP_MESSAGE)

        if save_var.get():
            save_unpacked_variables(WORKSPACE_VARIABLES, output_dir, archive_path)
            messagebox.showinfo(
                "Unpack complete",
                f"Saved unpacked variables to: {output_dir}",
            )
        else:
            messagebox.showinfo(
                "Load complete",
                "Variables loaded into memory. Saving was skipped.",
            )

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Workspace archive (.npz):").grid(row=0, column=0, sticky="w")
    tk.Entry(frame, textvariable=input_var, width=60).grid(
        row=0, column=1, padx=6, pady=6
    )
    tk.Button(frame, text="Browse", command=choose_input).grid(row=0, column=2)

    tk.Label(frame, text="Output directory (optional):").grid(
        row=1, column=0, sticky="w"
    )
    tk.Entry(frame, textvariable=output_var, width=60).grid(
        row=1, column=1, padx=6, pady=6
    )
    tk.Button(frame, text="Browse", command=choose_output).grid(row=1, column=2)

    tk.Checkbutton(
        frame,
        text="Save unpacked variables",
        variable=save_var,
    ).grid(row=2, column=1, sticky="w", pady=(6, 0))

    tk.Button(frame, text="Load workspace", command=run_unpack).grid(
        row=3, column=1, pady=12, sticky="e"
    )

    root.mainloop()


def main() -> None:
    build_gui()


if __name__ == "__main__":
    main()

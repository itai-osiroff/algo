"""System gain calibration via Teledyne's mean-variance procedure.

This module provides:
- A shared computation core for system gain estimation.
- A CLI entrypoint for scripted/reproducible runs.
- A simple Tkinter GUI alternative with ROI preview.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_ROI_SIZE = 100


@dataclass(frozen=True)
class SystemGainResult:
    """Computed Teledyne gain outputs and run metadata."""

    gain_e_per_adu: float
    gain_adu_per_e: float
    mean_corr_adu: float
    variance_adu2: float
    std_diff_adu: float
    roi: tuple[int, int, int, int]  # (x0, y0, x1, y1) with x1/y1 exclusive
    bias_frame_count: int = 1
    flat1_frame_count: int = 1
    flat2_frame_count: int = 1


@dataclass(frozen=True)
class SystemGainOptions:
    """Input options shared by CLI and GUI routes."""

    bias_image: str | None = None
    bias_dir: str | None = None
    flat1_image: str | None = None
    flat1_dir: str | None = None
    flat2_image: str | None = None
    flat2_dir: str | None = None
    pattern: str | None = None
    roi_size: int = DEFAULT_ROI_SIZE
    roi_center_x: int | None = None
    roi_center_y: int | None = None
    output_json: str | None = None


@dataclass(frozen=True)
class LoadedImageSource:
    """A loaded/averaged source image with provenance metadata."""

    image: np.ndarray
    frame_count: int
    paths: list[Path]
    warnings: list[str]


def _to_gray_float64(image: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.float64)
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return arr[..., 0].astype(np.float64)
    raise ValueError(f"Expected 2D grayscale or 3D image array, got shape {arr.shape}.")


def load_image(path: str | Path) -> np.ndarray:
    """Load one image file as grayscale float64 without normalization."""
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with Image.open(image_path) as image:
        return _to_gray_float64(image)


def _list_image_files(directory: Path, pattern: str | None = None) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    if pattern is not None and pattern.strip():
        paths = [path for path in sorted(directory.glob(pattern)) if path.is_file()]
    else:
        paths = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]

    if not paths:
        pattern_suffix = f" with pattern '{pattern}'" if pattern else ""
        raise ValueError(f"No image files found in {directory}{pattern_suffix}.")
    return paths


def _average_image_files(paths: list[Path], source_name: str) -> np.ndarray:
    running_sum: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None

    for path in paths:
        image = load_image(path)
        if expected_shape is None:
            expected_shape = image.shape
            running_sum = np.zeros(expected_shape, dtype=np.float64)
        if image.shape != expected_shape:
            raise ValueError(
                f"Inconsistent image shape in {source_name} frames: {path} has {image.shape}, "
                f"expected {expected_shape}."
            )
        assert running_sum is not None
        running_sum += image

    assert running_sum is not None
    return running_sum / float(len(paths))


def load_source_data(
    source_name: str,
    image_path: str | None,
    directory_path: str | None,
    pattern: str | None = None,
) -> LoadedImageSource:
    """Load one source from either single image or directory stack."""
    has_image = bool(image_path and image_path.strip())
    has_dir = bool(directory_path and directory_path.strip())
    if has_image == has_dir:
        raise ValueError(
            f"{source_name}: provide exactly one of image path or directory path."
        )

    warnings: list[str] = []
    if has_image:
        resolved = Path(str(image_path))
        image = load_image(resolved)
        return LoadedImageSource(
            image=image,
            frame_count=1,
            paths=[resolved],
            warnings=warnings,
        )

    assert directory_path is not None
    directory = Path(directory_path)
    frame_paths = _list_image_files(directory, pattern=pattern)
    averaged = _average_image_files(frame_paths, source_name=source_name)
    if source_name.lower() == "bias" and len(frame_paths) != 100:
        warnings.append(
            f"Expected ~100 bias frames, found {len(frame_paths)} in {directory}."
        )

    return LoadedImageSource(
        image=averaged,
        frame_count=len(frame_paths),
        paths=frame_paths,
        warnings=warnings,
    )


def resolve_roi_bounds(
    image_shape: tuple[int, int] | list[int],
    roi_size: int = DEFAULT_ROI_SIZE,
    roi_center_x: int | None = None,
    roi_center_y: int | None = None,
) -> tuple[int, int, int, int]:
    """Resolve and validate ROI bounds as (x0, y0, x1, y1)."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must include at least (height, width).")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if roi_size <= 0:
        raise ValueError("ROI size must be positive.")
    if roi_size > height or roi_size > width:
        raise ValueError(
            f"ROI size {roi_size} exceeds image dimensions ({height}, {width})."
        )

    center_x = width // 2 if roi_center_x is None else int(roi_center_x)
    center_y = height // 2 if roi_center_y is None else int(roi_center_y)

    x0 = center_x - roi_size // 2
    y0 = center_y - roi_size // 2
    x1 = x0 + roi_size
    y1 = y0 + roi_size

    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"Resolved ROI {(x0, y0, x1, y1)} is out of bounds for image ({height}, {width})."
        )
    return x0, y0, x1, y1


def roi_rectangle_for_preview(
    image_shape: tuple[int, int] | list[int],
    roi_size: int = DEFAULT_ROI_SIZE,
    roi_center_x: int | None = None,
    roi_center_y: int | None = None,
) -> tuple[int, int, int, int]:
    """Helper used by GUI preview to compute the ROI rectangle."""
    return resolve_roi_bounds(
        image_shape=image_shape,
        roi_size=roi_size,
        roi_center_x=roi_center_x,
        roi_center_y=roi_center_y,
    )


def compute_system_gain(
    bias: np.ndarray,
    flat1: np.ndarray,
    flat2: np.ndarray,
    roi: tuple[int, int, int, int],
) -> SystemGainResult:
    """Compute system gain from bias/flat images using Teledyne's steps."""
    bias_arr = np.asarray(bias, dtype=np.float64)
    flat1_arr = np.asarray(flat1, dtype=np.float64)
    flat2_arr = np.asarray(flat2, dtype=np.float64)

    if bias_arr.ndim != 2 or flat1_arr.ndim != 2 or flat2_arr.ndim != 2:
        raise ValueError("bias, flat1, and flat2 must be 2D grayscale arrays.")
    if bias_arr.shape != flat1_arr.shape or flat1_arr.shape != flat2_arr.shape:
        raise ValueError(
            "bias, flat1, and flat2 must have identical image shapes."
        )

    x0, y0, x1, y1 = roi
    height, width = bias_arr.shape
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
        raise ValueError(
            f"ROI {roi} is invalid for image shape {bias_arr.shape}."
        )

    diff = flat2_arr - flat1_arr  # Preserve negative values exactly as required.
    std_diff = float(np.std(diff[y0:y1, x0:x1], dtype=np.float64))
    variance = (std_diff * std_diff) / 2.0
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"Computed variance must be positive and finite; got {variance}."
        )

    corr = flat1_arr - bias_arr
    mean_corr = float(np.mean(corr[y0:y1, x0:x1], dtype=np.float64))
    gain_e_per_adu = mean_corr / variance
    if not np.isfinite(gain_e_per_adu) or gain_e_per_adu == 0.0:
        raise ValueError(
            f"Computed gain (e-/ADU) must be finite and non-zero; got {gain_e_per_adu}."
        )
    gain_adu_per_e = 1.0 / gain_e_per_adu
    if not np.isfinite(gain_adu_per_e):
        raise ValueError(
            f"Computed reciprocal gain (ADU/e-) must be finite; got {gain_adu_per_e}."
        )

    return SystemGainResult(
        gain_e_per_adu=float(gain_e_per_adu),
        gain_adu_per_e=float(gain_adu_per_e),
        mean_corr_adu=mean_corr,
        variance_adu2=float(variance),
        std_diff_adu=std_diff,
        roi=(x0, y0, x1, y1),
    )


def run_system_gain(options: SystemGainOptions) -> tuple[SystemGainResult, list[str]]:
    """Load inputs, execute gain computation, and attach frame counts."""
    bias = load_source_data(
        source_name="bias",
        image_path=options.bias_image,
        directory_path=options.bias_dir,
        pattern=options.pattern,
    )
    flat1 = load_source_data(
        source_name="flat1",
        image_path=options.flat1_image,
        directory_path=options.flat1_dir,
        pattern=options.pattern,
    )
    flat2 = load_source_data(
        source_name="flat2",
        image_path=options.flat2_image,
        directory_path=options.flat2_dir,
        pattern=options.pattern,
    )

    warnings = [*bias.warnings, *flat1.warnings, *flat2.warnings]

    if bias.image.shape != flat1.image.shape or flat1.image.shape != flat2.image.shape:
        raise ValueError(
            f"Input shapes do not match: bias={bias.image.shape}, "
            f"flat1={flat1.image.shape}, flat2={flat2.image.shape}."
        )

    roi = resolve_roi_bounds(
        image_shape=bias.image.shape,
        roi_size=options.roi_size,
        roi_center_x=options.roi_center_x,
        roi_center_y=options.roi_center_y,
    )
    result = compute_system_gain(bias.image, flat1.image, flat2.image, roi)
    result = replace(
        result,
        bias_frame_count=bias.frame_count,
        flat1_frame_count=flat1.frame_count,
        flat2_frame_count=flat2.frame_count,
    )
    return result, warnings


def format_result(result: SystemGainResult) -> str:
    x0, y0, x1, y1 = result.roi
    lines = [
        "System Gain Calibration Result",
        f"ROI (x0,y0,x1,y1): ({x0}, {y0}, {x1}, {y1})",
        f"Bias frames: {result.bias_frame_count}",
        f"Flat1 frames: {result.flat1_frame_count}",
        f"Flat2 frames: {result.flat2_frame_count}",
        f"Mean corr (ADU): {result.mean_corr_adu:.6f}",
        f"Std diff (ADU): {result.std_diff_adu:.6f}",
        f"Variance (ADU^2): {result.variance_adu2:.6f}",
        f"Gain (e-/ADU): {result.gain_e_per_adu:.6f}",
        f"Gain (ADU/e-): {result.gain_adu_per_e:.6f}",
    ]
    return "\n".join(lines)


def save_result_json(result: SystemGainResult, output_json: str | Path) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(result), fp, indent=2)
    return output_path


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return int(cleaned)


def build_options_from_gui_values(
    source_modes: Mapping[str, str],
    source_paths: Mapping[str, str],
    pattern_text: str,
    roi_size_text: str,
    roi_center_x_text: str,
    roi_center_y_text: str,
    output_json_text: str,
) -> SystemGainOptions:
    """Convert GUI entry fields into the shared options structure."""
    required_keys = {"bias", "flat1", "flat2"}
    if set(source_modes.keys()) != required_keys or set(source_paths.keys()) != required_keys:
        raise ValueError("GUI mappings must contain exactly: bias, flat1, flat2.")

    kwargs: dict[str, str | int | None] = {
        "pattern": pattern_text.strip() or None,
        "roi_size": int(roi_size_text.strip()),
        "roi_center_x": _parse_optional_int(roi_center_x_text),
        "roi_center_y": _parse_optional_int(roi_center_y_text),
        "output_json": output_json_text.strip() or None,
    }

    for key in ("bias", "flat1", "flat2"):
        mode = source_modes[key].strip().lower()
        path = source_paths[key].strip()
        if mode not in {"image", "dir"}:
            raise ValueError(f"Unsupported source mode '{mode}' for {key}.")
        if mode == "image":
            kwargs[f"{key}_image"] = path or None
            kwargs[f"{key}_dir"] = None
        else:
            kwargs[f"{key}_image"] = None
            kwargs[f"{key}_dir"] = path or None

    return SystemGainOptions(**kwargs)


def _options_from_namespace(args: argparse.Namespace) -> SystemGainOptions:
    return SystemGainOptions(
        bias_image=args.bias_image,
        bias_dir=args.bias_dir,
        flat1_image=args.flat1_image,
        flat1_dir=args.flat1_dir,
        flat2_image=args.flat2_image,
        flat2_dir=args.flat2_dir,
        pattern=args.pattern,
        roi_size=args.roi_size,
        roi_center_x=args.roi_center_x,
        roi_center_y=args.roi_center_y,
        output_json=args.output_json,
    )


def _first_image_from_source(
    mode: str,
    path: str,
    pattern: str | None,
) -> Path:
    cleaned_path = path.strip()
    if not cleaned_path:
        raise ValueError("No source path provided for preview.")
    if mode == "image":
        return Path(cleaned_path)
    if mode == "dir":
        paths = _list_image_files(Path(cleaned_path), pattern=pattern)
        return paths[0]
    raise ValueError(f"Unsupported preview source mode: {mode}")


def _launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    root = tk.Tk()
    root.title("System Gain Calibration (Teledyne)")

    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="nsew")
    right = ttk.Frame(root, padding=8)
    right.grid(row=0, column=1, sticky="nsew")
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    source_modes = {
        "bias": tk.StringVar(value="dir"),
        "flat1": tk.StringVar(value="image"),
        "flat2": tk.StringVar(value="image"),
    }
    source_paths = {
        "bias": tk.StringVar(value=""),
        "flat1": tk.StringVar(value=""),
        "flat2": tk.StringVar(value=""),
    }
    pattern_var = tk.StringVar(value="")
    roi_size_var = tk.StringVar(value=str(DEFAULT_ROI_SIZE))
    roi_center_x_var = tk.StringVar(value="")
    roi_center_y_var = tk.StringVar(value="")
    output_json_var = tk.StringVar(value="")
    preview_source_var = tk.StringVar(value="flat1")

    def browse_source(source_key: str) -> None:
        mode = source_modes[source_key].get().strip().lower()
        if mode == "dir":
            selection = filedialog.askdirectory(title=f"Select {source_key} directory")
        else:
            selection = filedialog.askopenfilename(
                title=f"Select {source_key} image",
                filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp")],
            )
        if selection:
            source_paths[source_key].set(selection)

    row = 0
    ttk.Label(left, text="Bias source").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Combobox(
        left,
        textvariable=source_modes["bias"],
        values=("image", "dir"),
        state="readonly",
        width=8,
    ).grid(row=row, column=1, sticky="w", pady=2, padx=4)
    ttk.Entry(left, textvariable=source_paths["bias"], width=40).grid(
        row=row, column=2, sticky="we", pady=2
    )
    ttk.Button(left, text="Browse", command=lambda: browse_source("bias")).grid(
        row=row, column=3, padx=4, pady=2
    )

    row += 1
    ttk.Label(left, text="Flat1 source").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Combobox(
        left,
        textvariable=source_modes["flat1"],
        values=("image", "dir"),
        state="readonly",
        width=8,
    ).grid(row=row, column=1, sticky="w", pady=2, padx=4)
    ttk.Entry(left, textvariable=source_paths["flat1"], width=40).grid(
        row=row, column=2, sticky="we", pady=2
    )
    ttk.Button(left, text="Browse", command=lambda: browse_source("flat1")).grid(
        row=row, column=3, padx=4, pady=2
    )

    row += 1
    ttk.Label(left, text="Flat2 source").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Combobox(
        left,
        textvariable=source_modes["flat2"],
        values=("image", "dir"),
        state="readonly",
        width=8,
    ).grid(row=row, column=1, sticky="w", pady=2, padx=4)
    ttk.Entry(left, textvariable=source_paths["flat2"], width=40).grid(
        row=row, column=2, sticky="we", pady=2
    )
    ttk.Button(left, text="Browse", command=lambda: browse_source("flat2")).grid(
        row=row, column=3, padx=4, pady=2
    )

    row += 1
    ttk.Label(left, text="Directory pattern").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Entry(left, textvariable=pattern_var, width=20).grid(
        row=row, column=1, columnspan=3, sticky="we", pady=2, padx=4
    )

    row += 1
    ttk.Label(left, text="ROI size").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Entry(left, textvariable=roi_size_var, width=12).grid(
        row=row, column=1, sticky="w", pady=2, padx=4
    )
    ttk.Label(left, text="ROI center X").grid(row=row, column=2, sticky="e", pady=2, padx=4)
    ttk.Entry(left, textvariable=roi_center_x_var, width=12).grid(
        row=row, column=3, sticky="w", pady=2
    )

    row += 1
    ttk.Label(left, text="ROI center Y").grid(row=row, column=2, sticky="e", pady=2, padx=4)
    ttk.Entry(left, textvariable=roi_center_y_var, width=12).grid(
        row=row, column=3, sticky="w", pady=2
    )

    row += 1
    ttk.Label(left, text="Output JSON").grid(row=row, column=0, sticky="w", pady=2)
    ttk.Entry(left, textvariable=output_json_var, width=40).grid(
        row=row, column=1, columnspan=2, sticky="we", pady=2, padx=4
    )
    ttk.Button(
        left,
        text="Browse",
        command=lambda: output_json_var.set(
            filedialog.asksaveasfilename(
                title="Save JSON result",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
            )
            or output_json_var.get()
        ),
    ).grid(row=row, column=3, padx=4, pady=2)

    row += 1
    ttk.Label(left, text="Preview source").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        left,
        textvariable=preview_source_var,
        values=("bias", "flat1", "flat2"),
        state="readonly",
        width=8,
    ).grid(row=row, column=1, sticky="w", pady=4, padx=4)

    row += 1
    button_frame = ttk.Frame(left)
    button_frame.grid(row=row, column=0, columnspan=4, sticky="we", pady=6)

    fig = Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ax.set_title("ROI preview")
    canvas.draw()

    results_var = tk.StringVar(value="Run calibration to see results.")
    ttk.Label(left, textvariable=results_var, justify="left").grid(
        row=row + 1, column=0, columnspan=4, sticky="w", pady=8
    )

    def preview_roi() -> None:
        source_key = preview_source_var.get().strip().lower()
        try:
            path = _first_image_from_source(
                mode=source_modes[source_key].get().strip().lower(),
                path=source_paths[source_key].get(),
                pattern=pattern_var.get().strip() or None,
            )
            image = load_image(path)
            roi = roi_rectangle_for_preview(
                image_shape=image.shape,
                roi_size=int(roi_size_var.get().strip()),
                roi_center_x=_parse_optional_int(roi_center_x_var.get()),
                roi_center_y=_parse_optional_int(roi_center_y_var.get()),
            )
        except Exception as exc:  # noqa: BLE001 - report UI errors to user
            messagebox.showerror("Preview failed", str(exc))
            return

        x0, y0, x1, y1 = roi
        ax.clear()
        ax.imshow(image, cmap="gray")
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )
        ax.set_title(f"{source_key} preview: {path.name}")
        canvas.draw()

    def run_from_gui() -> None:
        try:
            options = build_options_from_gui_values(
                source_modes={key: var.get() for key, var in source_modes.items()},
                source_paths={key: var.get() for key, var in source_paths.items()},
                pattern_text=pattern_var.get(),
                roi_size_text=roi_size_var.get(),
                roi_center_x_text=roi_center_x_var.get(),
                roi_center_y_text=roi_center_y_var.get(),
                output_json_text=output_json_var.get(),
            )
            result, warnings = run_system_gain(options)
        except Exception as exc:  # noqa: BLE001 - report UI errors to user
            messagebox.showerror("Calibration failed", str(exc))
            return

        if warnings:
            messagebox.showwarning("Calibration warning", "\n".join(warnings))

        results_var.set(format_result(result))
        if options.output_json:
            output_path = save_result_json(result, options.output_json)
            messagebox.showinfo("Saved", f"Saved result to {output_path}")

    ttk.Button(button_frame, text="Preview ROI", command=preview_roi).pack(
        side="left", padx=4
    )
    ttk.Button(button_frame, text="Run calibration", command=run_from_gui).pack(
        side="left", padx=4
    )

    left.columnconfigure(2, weight=1)
    root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute system gain using Teledyne's mean-variance method."
    )

    bias_group = parser.add_mutually_exclusive_group(required=False)
    bias_group.add_argument("--bias-image", help="Path to a pre-averaged bias image.")
    bias_group.add_argument(
        "--bias-dir",
        help="Directory of dark frames; frames are averaged into the bias image.",
    )

    flat1_group = parser.add_mutually_exclusive_group(required=False)
    flat1_group.add_argument("--flat1-image", help="Path to flat1 image.")
    flat1_group.add_argument(
        "--flat1-dir", help="Directory of frames to average into flat1."
    )

    flat2_group = parser.add_mutually_exclusive_group(required=False)
    flat2_group.add_argument("--flat2-image", help="Path to flat2 image.")
    flat2_group.add_argument(
        "--flat2-dir", help="Directory of frames to average into flat2."
    )

    parser.add_argument(
        "--pattern",
        help="Optional glob pattern for directory inputs, e.g. '*.tif*'.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=DEFAULT_ROI_SIZE,
        help=f"ROI side length in pixels (default: {DEFAULT_ROI_SIZE}).",
    )
    parser.add_argument("--roi-center-x", type=int, help="Optional ROI center X.")
    parser.add_argument("--roi-center-y", type=int, help="Optional ROI center Y.")
    parser.add_argument("--output-json", help="Optional path to write result JSON.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch a simple GUI instead of CLI execution.",
    )
    return parser.parse_args(argv)


def _validate_cli_options(options: SystemGainOptions) -> None:
    for key in ("bias", "flat1", "flat2"):
        image_path = getattr(options, f"{key}_image")
        dir_path = getattr(options, f"{key}_dir")
        if bool(image_path) == bool(dir_path):
            raise ValueError(
                f"Provide exactly one of --{key}-image or --{key}-dir."
            )


def _run_cli(args: argparse.Namespace) -> None:
    options = _options_from_namespace(args)
    _validate_cli_options(options)
    result, warnings = run_system_gain(options)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(format_result(result))
    if options.output_json:
        output_path = save_result_json(result, options.output_json)
        print(f"Saved result JSON: {output_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gui:
        _launch_gui()
        return
    _run_cli(args)


if __name__ == "__main__":
    main()

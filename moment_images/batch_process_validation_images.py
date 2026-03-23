"""
Batch GUI for processing validation images without comparison views.

This script applies the same geometric/intensity/scale pipeline used by
``validate_rectification.py`` across all supported images in one folder and
saves outputs with a ``_processed`` suffix.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import filedialog, messagebox, ttk


if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import im_process
    from validate_rectification import (
        SCALE_MAX,
        SCALE_MIN,
        _apply_camera_defaults,
        _apply_geometry,
        _apply_intensity_window,
        _scale_image,
        _to_gray_float32,
    )
else:
    from . import im_process
    from .validate_rectification import (
        SCALE_MAX,
        SCALE_MIN,
        _apply_camera_defaults,
        _apply_geometry,
        _apply_intensity_window,
        _scale_image,
        _to_gray_float32,
    )


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class BatchProcessOptions:
    """Single settings set applied to all images in one batch run."""

    rotation_deg: int = 0
    flip_h: bool = False
    flip_v: bool = False
    crop_box: tuple[int, int, int, int] | None = None
    apply_intensity_window: bool = False
    intensity_min: float = 0.0
    intensity_max: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    preblur_before_resize: bool = True
    preserve_flux_sum: bool = True

    def normalized(self) -> "BatchProcessOptions":
        rotation_steps = int(round(self.rotation_deg / 90.0))
        normalized_rotation = (rotation_steps % 4) * 90

        normalized_crop = self.crop_box
        if normalized_crop is not None:
            x0, y0, x1, y1 = normalized_crop
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            if x1 <= x0 or y1 <= y0:
                normalized_crop = None
            else:
                normalized_crop = (x0, y0, x1, y1)

        return BatchProcessOptions(
            rotation_deg=normalized_rotation,
            flip_h=bool(self.flip_h),
            flip_v=bool(self.flip_v),
            crop_box=normalized_crop,
            apply_intensity_window=bool(self.apply_intensity_window),
            intensity_min=float(self.intensity_min),
            intensity_max=float(self.intensity_max),
            scale_x=float(self.scale_x),
            scale_y=float(self.scale_y),
            preblur_before_resize=bool(self.preblur_before_resize),
            preserve_flux_sum=bool(self.preserve_flux_sum),
        )


@dataclass(frozen=True)
class BatchFileResult:
    """Result for one image file in the batch."""

    input_path: Path
    output_path: Path
    status: Literal["processed", "skipped", "failed"]
    message: str = ""


@dataclass
class BatchRunSummary:
    """Batch processing summary lists grouped by final status."""

    processed: list[BatchFileResult] = field(default_factory=list)
    skipped: list[BatchFileResult] = field(default_factory=list)
    failed: list[BatchFileResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.processed) + len(self.skipped) + len(self.failed)


def list_supported_images(input_dir: str | Path) -> list[Path]:
    """List supported image files in one folder (non-recursive), sorted by name."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")

    files = [
        child
        for child in input_path.iterdir()
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: (path.name.lower(), path.name))


def build_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Create output path using ``<stem>_processed<suffix>`` naming."""
    source = Path(input_path)
    destination_dir = Path(output_dir)
    return destination_dir / f"{source.stem}_processed{source.suffix}"


def apply_processing_pipeline(image_array: np.ndarray, options: BatchProcessOptions) -> np.ndarray:
    """Apply the same processing order used by GuidedValidationEditor."""
    opts = options.normalized()
    if not (SCALE_MIN <= opts.scale_x <= SCALE_MAX and SCALE_MIN <= opts.scale_y <= SCALE_MAX):
        raise ValueError(f"Scale values must be within [{SCALE_MIN:.3f}, {SCALE_MAX:.3f}].")

    base = _apply_camera_defaults(_to_gray_float32(image_array), camera_type=None)
    geom = _apply_geometry(
        base,
        rotation_deg=opts.rotation_deg,
        flip_h=opts.flip_h,
        flip_v=opts.flip_v,
        crop_box=opts.crop_box,
    )
    intensity = geom
    if opts.apply_intensity_window:
        intensity = _apply_intensity_window(geom, opts.intensity_min, opts.intensity_max)

    processed = _scale_image(
        intensity,
        opts.scale_x,
        opts.scale_y,
        preblur_before_resize=opts.preblur_before_resize,
        preserve_flux_sum=opts.preserve_flux_sum,
    )
    return np.asarray(processed, dtype=np.float32)


def cast_processed_to_dtype(processed_image: np.ndarray, target_dtype: np.dtype | type[np.generic]) -> np.ndarray:
    """Cast processed float image back to input dtype using safe clipping."""
    arr = np.asarray(processed_image)
    dtype = np.dtype(target_dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(np.rint(arr), info.min, info.max)
        return clipped.astype(dtype)

    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype)

    if np.issubdtype(dtype, np.bool_):
        return (arr > 0).astype(dtype)

    raise ValueError(f"Unsupported target dtype for saving: {dtype}")


def save_processed_image(output_path: str | Path, processed_image: np.ndarray, target_dtype: np.dtype) -> None:
    """Save processed image with the same extension and dtype policy."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_array = cast_processed_to_dtype(processed_image, target_dtype)
    Image.fromarray(output_array).save(out_path)


def process_single_image(
    input_path: str | Path,
    output_dir: str | Path,
    options: BatchProcessOptions,
) -> BatchFileResult:
    """Process one image and return status without raising to caller."""
    src = Path(input_path)
    dst = build_output_path(src, output_dir)

    if dst.exists():
        return BatchFileResult(
            input_path=src,
            output_path=dst,
            status="skipped",
            message="output_exists",
        )

    loaded = im_process.load_image(str(src))
    if loaded is None:
        return BatchFileResult(
            input_path=src,
            output_path=dst,
            status="failed",
            message="load_failed",
        )

    try:
        source_array = np.asarray(loaded)
        processed = apply_processing_pipeline(source_array, options)
        save_processed_image(dst, processed, source_array.dtype)
    except Exception as exc:  # noqa: BLE001 - batch should continue on per-file failure
        return BatchFileResult(
            input_path=src,
            output_path=dst,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )

    return BatchFileResult(
        input_path=src,
        output_path=dst,
        status="processed",
    )


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    options: BatchProcessOptions,
) -> BatchRunSummary:
    """Process all supported files in ``input_dir`` and collect summary stats."""
    input_paths = list_supported_images(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = BatchRunSummary()
    for src in input_paths:
        result = process_single_image(src, output_path, options)
        if result.status == "processed":
            summary.processed.append(result)
        elif result.status == "skipped":
            summary.skipped.append(result)
        else:
            summary.failed.append(result)
    return summary


class BatchValidationProcessorGUI:
    """Simple GUI for selecting folders, options, and running batch processing."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Batch Validation Image Processor")

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.rotation_var = tk.StringVar(value="0")
        self.flip_h_var = tk.BooleanVar(value=False)
        self.flip_v_var = tk.BooleanVar(value=False)
        self.crop_enabled_var = tk.BooleanVar(value=False)
        self.crop_x0_var = tk.StringVar(value="0")
        self.crop_y0_var = tk.StringVar(value="0")
        self.crop_x1_var = tk.StringVar(value="0")
        self.crop_y1_var = tk.StringVar(value="0")
        self.intensity_enabled_var = tk.BooleanVar(value=False)
        self.intensity_min_var = tk.StringVar(value="0.0")
        self.intensity_max_var = tk.StringVar(value="255.0")
        self.scale_x_var = tk.StringVar(value="1.0")
        self.scale_y_var = tk.StringVar(value="1.0")
        self.preblur_var = tk.BooleanVar(value=True)
        self.preserve_flux_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(frame, text="Input folder").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.input_dir_var, width=52).grid(row=row, column=1, sticky="we", pady=3, padx=4)
        ttk.Button(frame, text="Browse", command=self._browse_input_dir).grid(row=row, column=2, pady=3)

        row += 1
        ttk.Label(frame, text="Output folder").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.output_dir_var, width=52).grid(row=row, column=1, sticky="we", pady=3, padx=4)
        ttk.Button(frame, text="Browse", command=self._browse_output_dir).grid(row=row, column=2, pady=3)

        row += 1
        geometry_frame = ttk.LabelFrame(frame, text="Geometry")
        geometry_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 4))
        ttk.Label(geometry_frame, text="Rotation").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            geometry_frame,
            textvariable=self.rotation_var,
            values=["0", "90", "180", "270"],
            width=8,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(geometry_frame, text="Flip H", variable=self.flip_h_var).grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(geometry_frame, text="Flip V", variable=self.flip_v_var).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        ttk.Checkbutton(
            geometry_frame,
            text="Enable crop x0,y0,x1,y1",
            variable=self.crop_enabled_var,
        ).grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(geometry_frame, textvariable=self.crop_x0_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Entry(geometry_frame, textvariable=self.crop_y0_var, width=8).grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(geometry_frame, textvariable=self.crop_x1_var, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=4)
        ttk.Entry(geometry_frame, textvariable=self.crop_y1_var, width=8).grid(row=1, column=4, sticky="w", padx=4, pady=4)

        row += 1
        intensity_frame = ttk.LabelFrame(frame, text="Intensity")
        intensity_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        ttk.Checkbutton(
            intensity_frame,
            text="Enable min/max clip",
            variable=self.intensity_enabled_var,
        ).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(intensity_frame, text="Min").grid(row=0, column=1, sticky="e", padx=4, pady=4)
        ttk.Entry(intensity_frame, textvariable=self.intensity_min_var, width=12).grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(intensity_frame, text="Max").grid(row=0, column=3, sticky="e", padx=4, pady=4)
        ttk.Entry(intensity_frame, textvariable=self.intensity_max_var, width=12).grid(row=0, column=4, sticky="w", padx=4, pady=4)

        row += 1
        scale_frame = ttk.LabelFrame(frame, text="Scale/Resize")
        scale_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        ttk.Label(scale_frame, text="Scale X").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(scale_frame, textvariable=self.scale_x_var, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(scale_frame, text="Scale Y").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(scale_frame, textvariable=self.scale_y_var, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(scale_frame, text=f"Range: [{SCALE_MIN:.3f}, {SCALE_MAX:.3f}]").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(scale_frame, text="Pre-blur before resize", variable=self.preblur_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(scale_frame, text="Preserve flux sum", variable=self.preserve_flux_var).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=4, pady=4
        )

        row += 1
        actions_frame = ttk.Frame(frame)
        actions_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        ttk.Button(actions_frame, text="Process", command=self._run_processing).pack(side=tk.RIGHT, padx=4)
        ttk.Button(actions_frame, text="Close", command=self.root.destroy).pack(side=tk.RIGHT, padx=4)

        row += 1
        ttk.Label(frame, textvariable=self.status_var, foreground="navy").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))

        frame.columnconfigure(1, weight=1)

    def _browse_input_dir(self) -> None:
        selection = filedialog.askdirectory(title="Select Input Folder")
        if selection:
            self.input_dir_var.set(selection)

    def _browse_output_dir(self) -> None:
        selection = filedialog.askdirectory(title="Select Output Folder")
        if selection:
            self.output_dir_var.set(selection)

    def _parse_options(self) -> BatchProcessOptions | None:
        try:
            rotation = int(self.rotation_var.get().strip())
            scale_x = float(self.scale_x_var.get().strip())
            scale_y = float(self.scale_y_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Rotation must be integer and scales must be numeric.")
            return None

        if scale_x < SCALE_MIN or scale_x > SCALE_MAX or scale_y < SCALE_MIN or scale_y > SCALE_MAX:
            messagebox.showerror(
                "Invalid Scale",
                f"Scale values must be within [{SCALE_MIN:.3f}, {SCALE_MAX:.3f}].",
            )
            return None

        crop_box: tuple[int, int, int, int] | None = None
        if self.crop_enabled_var.get():
            try:
                x0 = int(self.crop_x0_var.get().strip())
                y0 = int(self.crop_y0_var.get().strip())
                x1 = int(self.crop_x1_var.get().strip())
                y1 = int(self.crop_y1_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid Crop", "Crop coordinates must be integers.")
                return None
            if x1 <= x0 or y1 <= y0:
                messagebox.showerror("Invalid Crop", "Crop must satisfy x1>x0 and y1>y0.")
                return None
            crop_box = (x0, y0, x1, y1)

        intensity_min = 0.0
        intensity_max = 1.0
        if self.intensity_enabled_var.get():
            try:
                intensity_min = float(self.intensity_min_var.get().strip())
                intensity_max = float(self.intensity_max_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid Intensity", "Intensity min/max must be numeric.")
                return None
            if intensity_max <= intensity_min:
                messagebox.showerror("Invalid Intensity", "Intensity max must be greater than min.")
                return None

        return BatchProcessOptions(
            rotation_deg=rotation,
            flip_h=bool(self.flip_h_var.get()),
            flip_v=bool(self.flip_v_var.get()),
            crop_box=crop_box,
            apply_intensity_window=bool(self.intensity_enabled_var.get()),
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            scale_x=scale_x,
            scale_y=scale_y,
            preblur_before_resize=bool(self.preblur_var.get()),
            preserve_flux_sum=bool(self.preserve_flux_var.get()),
        ).normalized()

    def _run_processing(self) -> None:
        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()

        if not input_dir:
            messagebox.showerror("Missing Input", "Please select an input folder.")
            return
        if not output_dir:
            messagebox.showerror("Missing Output", "Please select an output folder.")
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        if not input_path.exists() or not input_path.is_dir():
            messagebox.showerror("Invalid Input", "Input folder does not exist or is not a directory.")
            return

        options = self._parse_options()
        if options is None:
            return

        images = list_supported_images(input_path)
        if not images:
            messagebox.showwarning("No Images", "No supported images found in input folder.")
            return

        self.status_var.set(f"Processing {len(images)} image(s)...")
        self.root.update_idletasks()
        self.root.config(cursor="watch")

        try:
            summary = process_folder(input_path, output_path, options)
        except Exception as exc:  # noqa: BLE001 - surface unexpected batch errors to GUI
            messagebox.showerror("Processing Error", str(exc))
            self.status_var.set("Failed.")
            return
        finally:
            self.root.config(cursor="")

        self.status_var.set(
            f"Done. Processed={len(summary.processed)}, Skipped={len(summary.skipped)}, Failed={len(summary.failed)}"
        )

        details: list[str] = [
            f"Input folder: {input_path}",
            f"Output folder: {output_path}",
            f"Total discovered: {summary.total}",
            f"Processed: {len(summary.processed)}",
            f"Skipped (existing): {len(summary.skipped)}",
            f"Failed: {len(summary.failed)}",
        ]
        if summary.failed:
            details.append("")
            details.append("Failures:")
            for item in summary.failed[:10]:
                details.append(f"- {item.input_path.name}: {item.message}")
            if len(summary.failed) > 10:
                details.append(f"- ... and {len(summary.failed) - 10} more")

        messagebox.showinfo("Batch Processing Complete", "\n".join(details))


def main() -> None:
    root = tk.Tk()
    BatchValidationProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


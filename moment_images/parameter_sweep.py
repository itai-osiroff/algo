"""Parameter sweep runner for ``im_process`` rectification routines.

This module drives ``rectify_views_from_config`` with a configurable grid of
parameters (window size, maximum displacement, Gaussian smoothing, etc.) and
persists the results in an easy-to-compare folder layout. It is intended to run
offline so practitioners can explore trade-offs without stepping through the
interactive ``main`` workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from moment_images.im_process import (
    RectificationRunConfig,
    RectificationResult,
    generate_rectification_figures,
    min_max_normalize,
    rectify_views_from_config,
    save_figures,
    save_workspace_variables,
    extract_views,
    VIEW_DISTANCE_SCALE_FACTOR,
)
from moment_images.sweep_analysis import (
    analyze_summary_data,
    build_metric_record,
    save_analysis,
)


class LiveMetricsPlotter:
    """Render sweep metrics as they are recorded for at-a-glance comparisons."""

    def __init__(self, metric_keys: Sequence[str]):
        self.metric_keys = list(metric_keys)
        self.scopes = [
            ("global_only_metrics", "Global only"),
            ("global_and_local_metrics", "Global + local"),
        ]
        self.run_labels: list[str] = []
        self.data: Dict[str, Dict[str, List[float]]] = {
            scope: {metric: [] for metric in self.metric_keys} for scope, _ in self.scopes
        }

        plt.ion()
        self.fig, axes = plt.subplots(
            len(self.metric_keys), 1, figsize=(10, 3 * len(self.metric_keys)), sharex=True
        )
        self.axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        self.fig.suptitle("Live sweep metrics", fontsize=14)

    def update(self, run_name: str, record: Dict[str, object]) -> None:
        self.run_labels.append(run_name)

        for scope, _ in self.scopes:
            metrics = record.get(scope)
            for metric in self.metric_keys:
                value = float(metrics[metric]) if isinstance(metrics, dict) and metric in metrics else np.nan
                self.data[scope][metric].append(value)

        self._redraw()

    def _redraw(self) -> None:
        x_values = range(len(self.run_labels))

        for axis, metric in zip(self.axes, self.metric_keys):
            axis.clear()
            for scope, label in self.scopes:
                values = self.data[scope][metric]
                if not values or np.all(np.isnan(values)):
                    continue
                axis.plot(x_values, values, marker="o", label=label)
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.legend()

        self.axes[-1].set_xticks(list(x_values))
        self.axes[-1].set_xticklabels(self.run_labels, rotation=45, ha="right")
        self.axes[-1].set_xlabel("Run")

        self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def _build_int_range(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    return list(range(start, stop + 1, step))


def _build_float_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    values: List[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _save_run_artifacts(
    output_dir: Path,
    run_name: str,
    result: RectificationResult,
) -> None:
    config = result.config
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    avg_img_path = run_dir / "avg_rectified.png"
    avg_preview = np.asarray(min_max_normalize(result.avg_image), dtype=np.uint8)
    Image.fromarray(avg_preview).save(avg_img_path)

    np.save(run_dir / "rectified_views.npy", result.rectified_views)

    pixel_displacements_x = None
    pixel_displacements_y = None
    if result.pixel_similarity is not None:
        pixel_displacements_x = getattr(result.pixel_similarity, "disps_x", None)
        pixel_displacements_y = getattr(result.pixel_similarity, "disps_y", None)

    config_dict = asdict(config)
    strategy_code = {"global": 1, "pixel": 2, "combined": 3}.get(config.strategy.lower(), 0)
    workspace_data: Dict[str, object] = {
        "rectified_views": result.rectified_views,
        "average_rectified_image": result.avg_image,
        "global_rectification_matrix": result.global_similarity,
        "global_rectified_views": result.global_rectified_views,
        "average_rectified_global_image": result.global_avg_image,
        "pixel_rectified_views": (
            result.rectified_views if config.strategy.lower() in {"pixel", "combined"} else None
        ),
        "pixel_displacements_x": pixel_displacements_x,
        "pixel_displacements_y": pixel_displacements_y,
        "reference_view_index": np.array(result.ref_view_index),
        "rectification_strategy": np.array(strategy_code),
        "hdr_enabled": np.array(config.hdr_enabled),
        "hdr_exposure_times": (
            np.asarray(config.hdr_exposure_times, dtype=np.float64)
            if config.hdr_exposure_times is not None
            else None
        ),
        "normalize_to_8bit": np.array(config.normalize_to_8bit),
        "run_name": np.array(run_name),
        "config_json": np.array(json.dumps(config_dict, sort_keys=True)),
    }
    save_workspace_variables(str(run_dir / "workspace.npz"), workspace_data)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_dict, fp, indent=2)


def _persist_figures(figures: Dict[str, List[Figure]], base_dir: Path) -> None:
    for section, fig_list in figures.items():
        section_dir = base_dir / section
        save_figures(fig_list, str(section_dir))
        for fig in fig_list:
            plt.close(fig)


def _load_views(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Views file not found: {path}")
    views = np.load(path)
    if views.ndim == 4:
        if views.shape[0] != views.shape[1]:
            raise ValueError("views must have shape (N, N, H, W)")
        return views
    if views.ndim == 5:
        if views.shape[1] != views.shape[2]:
            raise ValueError("views must have shape (E, N, N, H, W)")
        return views
    raise ValueError("views must have shape (N, N, H, W) or (E, N, N, H, W)")


def _generate_views(
    image_path: Path,
    center_x: int,
    center_y: int,
    radius: int,
    grid_size: int,
    view_distance: int | None,
    cache_dir: Path,
    normalize_to_8bit: bool = False,
) -> tuple[np.ndarray, Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image = Image.open(image_path)
    if normalize_to_8bit:
        image_np = np.asarray(min_max_normalize(image))
    else:
        image_np = np.asarray(image).copy()

    distance = view_distance
    if distance is None:
        distance = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * radius))

    base_circle = np.array([center_x, center_y, radius])
    views = extract_views(image_np, base_circle, grid_size, distance)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "generated_views.npy"
    np.save(cache_path, views)
    return views, cache_path


def _generate_exposure_views(
    image_dir: Path,
    center_x: int,
    center_y: int,
    radius: int,
    grid_size: int,
    view_distance: int | None,
    cache_dir: Path,
    normalize_to_8bit: bool = False,
) -> tuple[np.ndarray, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Source image directory is not a folder: {image_dir}")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    distance = view_distance
    if distance is None:
        distance = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * radius))

    base_circle = np.array([center_x, center_y, radius])
    exposure_views: list[np.ndarray] = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if normalize_to_8bit:
            image_np = np.asarray(min_max_normalize(image))
        else:
            image_np = np.asarray(image).copy()
        views = extract_views(image_np, base_circle, grid_size, distance)
        exposure_views.append(views)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "generated_exposure_views.npy"
    exposure_views_arr = np.asarray(exposure_views)
    np.save(cache_path, exposure_views_arr)
    return exposure_views_arr, cache_path


def sweep_parameters(
    views: np.ndarray,
    output_dir: Path,
    window_sizes: Sequence[int],
    max_displacements: Sequence[int],
    gauss_sigmas: Sequence[float],
    disp_gauss_sigmas: Sequence[float],
    global_thresholds: Sequence[float] | None = None,
    pixel_thresholds: Sequence[float] | None = None,
    strategy: str = "pixel",
    save_figures_flag: bool = False,
    zero_fill_roll: bool = True,
    analyze: bool = False,
    live_plot: bool = False,
    hdr_enabled: bool = False,
    hdr_exposure_times: Sequence[float] | None = None,
    combined_use_global_avg_reference: bool = True,
    normalize_to_8bit: bool = False,
) -> None:
    default_config = RectificationRunConfig()
    global_thresholds = list(global_thresholds or [default_config.global_threshold])
    pixel_thresholds = list(pixel_thresholds or [default_config.pixel_threshold])

    if strategy == "global":
        combos = [
            (win, max_disp, gauss, disp_gauss, global_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for global_thresh in global_thresholds
        ]
    elif strategy == "pixel":
        combos = [
            (win, max_disp, gauss, disp_gauss, pixel_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for pixel_thresh in pixel_thresholds
        ]
    else:
        combos = [
            (win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for global_thresh in global_thresholds
            for pixel_thresh in pixel_thresholds
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []
    plotter: LiveMetricsPlotter | None = None

    for combo in tqdm(combos, desc="Sweeping parameters", unit="combo"):
        if strategy == "global":
            win, max_disp, gauss, disp_gauss, global_thresh = combo  # type: ignore[misc]
            pixel_thresh = default_config.pixel_threshold
        elif strategy == "pixel":
            win, max_disp, gauss, disp_gauss, pixel_thresh = combo  # type: ignore[misc]
            global_thresh = default_config.global_threshold
        else:
            win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh = combo  # type: ignore[misc]
        # Keep max_displacement explicit and independent from the patch window size
        # to match RectificationRunConfig semantics across rectification entrypoints.
        config = RectificationRunConfig(
            strategy=strategy,
            similarity_window_size=win,
            refinement_window_size=win,
            max_displacement=max_disp,
            gauss_sigma=gauss,
            gauss_size=int(gauss * 5),
            disp_gauss_sigma=disp_gauss,
            disp_gauss_size=win,
            zero_fill_roll=zero_fill_roll,
            global_threshold=global_thresh,
            pixel_threshold=pixel_thresh,
            hdr_enabled=hdr_enabled,
            hdr_exposure_times=list(hdr_exposure_times) if hdr_exposure_times is not None else None,
            combined_use_global_avg_reference=combined_use_global_avg_reference,
            normalize_to_8bit=normalize_to_8bit,
        )

        result = rectify_views_from_config(views, config)
        run_name_parts = [
            f"win{win}",
            f"disp{max_disp}",
            f"g{gauss}",
            f"dg{disp_gauss}",
        ]
        if strategy in {"global", "combined"}:
            run_name_parts.append(f"gt{global_thresh}")
        if strategy in {"pixel", "combined"}:
            run_name_parts.append(f"pt{pixel_thresh}")
        run_name = "_".join(run_name_parts)
        _save_run_artifacts(output_dir, run_name, result)

        if save_figures_flag:
            figures = generate_rectification_figures(result)  # pass full result so we can save global/pixel maps, originals, overlays, averages
            figures_dir = output_dir / run_name / "figures"
            _persist_figures(figures, figures_dir)

        record = build_metric_record(run_name, config, result)
        summary.append(record)

        if live_plot:
            if plotter is None:
                metrics_dict = record["global_and_local_metrics"]
                metric_keys = list(metrics_dict.keys()) if isinstance(metrics_dict, dict) else []
                plotter = LiveMetricsPlotter(metric_keys)
            plotter.update(run_name, record)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if analyze:
        analysis = analyze_summary_data(summary)
        save_analysis(analysis, output_dir / "analysis.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep im_process rectification parameters")
    parser.add_argument(
        "--views",
        help="Path to a precomputed (N, N, H, W) or (E, N, N, H, W) numpy array of views",
    )
    parser.add_argument("--source-image", help="Path to the source image to extract views from")
    parser.add_argument(
        "--source-image-dir",
        help="Directory of exposure images to extract HDR views from (used with --hdr-assemble)",
    )
    parser.add_argument("--center-x", type=int, help="X coordinate of the central view")
    parser.add_argument("--center-y", type=int, help="Y coordinate of the central view")
    parser.add_argument("--view-radius", type=int, help="Radius of a single view in pixels")
    parser.add_argument("--view-grid-size", type=int, default=3, help="Number of views per dimension to extract")
    parser.add_argument("--view-distance", type=int, help="Override the distance between adjacent view centers")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write sweep artifacts (defaults to sweep_runs/<timestamp>)",
    )
    parser.add_argument("--strategy", choices=["global", "pixel", "combined"], default="pixel")
    parser.add_argument(
        "--window-start",
        type=int,
        default=15,
        help="Pixel-wise patch window size sweep start (pixel / combined refinement)",
    )
    parser.add_argument(
        "--window-stop",
        type=int,
        default=45,
        help="Pixel-wise patch window size sweep stop",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=10,
        help="Pixel-wise patch window size sweep step",
    )
    parser.add_argument(
        "--max-disp-start",
        type=int,
        default=5,
        help=(
            "Explicit pixel-wise maximum displacement sweep start "
            "(independent from window size; maps to RectificationRunConfig.max_displacement)"
        ),
    )
    parser.add_argument(
        "--max-disp-stop",
        type=int,
        default=25,
        help="Explicit pixel-wise maximum displacement sweep stop",
    )
    parser.add_argument(
        "--max-disp-step",
        type=int,
        default=10,
        help="Explicit pixel-wise maximum displacement sweep step",
    )
    parser.add_argument("--gauss-sigma-start", type=float, default=3.0)
    parser.add_argument("--gauss-sigma-stop", type=float, default=7.0)
    parser.add_argument("--gauss-sigma-step", type=float, default=2.0)
    parser.add_argument("--disp-gauss-start", type=float, default=0.0)
    parser.add_argument("--disp-gauss-stop", type=float, default=1.0)
    parser.add_argument("--disp-gauss-step", type=float, default=1.0)
    parser.add_argument("--global-thresh-start", type=float, default=0.15)
    parser.add_argument("--global-thresh-stop", type=float, default=0.15)
    parser.add_argument("--global-thresh-step", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-start", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-stop", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-step", type=float, default=0.05)
    parser.add_argument("--zero-fill-roll",
                         action="store_true",
                         help="Use zero-filled rolling window in pixelwise rectification")
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save rectification figures (non-interactive) for each run",
    )
    parser.add_argument(
        "--hdr-assemble",
        action="store_true",
        help="Assemble HDR views from a stack of exposures shaped (E, N, N, H, W)",
    )
    parser.add_argument(
        "--hdr-exposure-times",
        help="Comma-separated exposure times (seconds) matching the exposure count",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Compute aggregate statistics for sweep metrics after the run",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Display a live graph of tracked metrics while the sweep runs",
    )
    parser.add_argument(
        "--use-global-avg-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the averaged global-rectified image as the pixel-wise reference when"
            " running the combined strategy"
        ),
    )
    parser.add_argument(
        "--normalize-to-8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize source images to 8-bit before view extraction and HDR assembly. "
            "Disabled by default to preserve raw intensities."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch a minimal GUI for configuring the sweep instead of using CLI flags",
    )
    return parser.parse_args(argv)


def _run_from_namespace(args: argparse.Namespace) -> Path:
    if args.save_figures and not args.live_plot:
        plt.switch_backend("Agg")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hdr_assemble and args.views is None and args.source_image_dir is None:
        raise SystemExit(
            "HDR assembly requires --views with shape (E, N, N, H, W) or --source-image-dir."
        )
    if not args.hdr_assemble and args.source_image_dir is not None:
        raise SystemExit("--source-image-dir is only supported with --hdr-assemble.")

    if args.views is None:
        if args.hdr_assemble:
            missing = [
                name
                for name, present in [
                    ("--center-x", args.center_x),
                    ("--center-y", args.center_y),
                    ("--view-radius", args.view_radius),
                ]
                if present is None
            ]
            if missing:
                missing_str = ", ".join(missing)
                raise SystemExit(
                    "When --views is not provided, the following arguments are required: "
                    f"{missing_str}."
                )
            views, cached_views_path = _generate_exposure_views(
                image_dir=Path(args.source_image_dir),
                center_x=args.center_x,
                center_y=args.center_y,
                radius=args.view_radius,
                grid_size=args.view_grid_size,
                view_distance=args.view_distance,
                cache_dir=output_dir,
                normalize_to_8bit=args.normalize_to_8bit,
            )
            print(f"Generated exposure views once and cached to {cached_views_path}")
        else:
            missing = [
                name
                for name, present in [
                    ("--source-image", args.source_image),
                    ("--center-x", args.center_x),
                    ("--center-y", args.center_y),
                    ("--view-radius", args.view_radius),
                ]
                if present is None
            ]
            if missing:
                missing_str = ", ".join(missing)
                raise SystemExit(
                    "When --views is not provided, the following arguments are required: "
                    f"{missing_str}."
                )

            views, cached_views_path = _generate_views(
                image_path=Path(args.source_image),
                center_x=args.center_x,
                center_y=args.center_y,
                radius=args.view_radius,
                grid_size=args.view_grid_size,
                view_distance=args.view_distance,
                cache_dir=output_dir,
                normalize_to_8bit=args.normalize_to_8bit,
            )
            print(f"Generated views once and cached to {cached_views_path}")
    else:
        cached_views_path = Path(args.views)
        views = _load_views(cached_views_path)

    if args.hdr_assemble and views.ndim != 5:
        raise SystemExit("HDR assembly requires views shaped (E, N, N, H, W).")
    if not args.hdr_assemble and views.ndim == 5:
        raise SystemExit("Views include multiple exposures; enable --hdr-assemble to use HDR assembly.")
    exposure_times = _parse_float_list(args.hdr_exposure_times)
    if exposure_times is not None and views.ndim == 5:
        if len(exposure_times) != views.shape[0]:
            raise SystemExit(
                "HDR exposure times must match the number of exposures in --views or --source-image-dir."
            )

    window_sizes = _build_int_range(args.window_start, args.window_stop, args.window_step)
    max_displacements = _build_int_range(args.max_disp_start, args.max_disp_stop, args.max_disp_step)
    gauss_sigmas = _build_float_range(args.gauss_sigma_start, args.gauss_sigma_stop, args.gauss_sigma_step)
    disp_gauss_sigmas = _build_float_range(args.disp_gauss_start, args.disp_gauss_stop, args.disp_gauss_step)
    global_thresholds = _build_float_range(
        args.global_thresh_start, args.global_thresh_stop, args.global_thresh_step
    )
    pixel_thresholds = _build_float_range(
        args.pixel_thresh_start, args.pixel_thresh_stop, args.pixel_thresh_step
    )

    sweep_parameters(
        views=views,
        output_dir=output_dir,
        window_sizes=window_sizes,
        max_displacements=max_displacements,
        gauss_sigmas=gauss_sigmas,
        disp_gauss_sigmas=disp_gauss_sigmas,
        global_thresholds=global_thresholds,
        pixel_thresholds=pixel_thresholds,
        strategy=args.strategy,
        save_figures_flag=args.save_figures,
        zero_fill_roll=args.zero_fill_roll,
        analyze=args.analyze,
        live_plot=args.live_plot,
        hdr_enabled=args.hdr_assemble,
        hdr_exposure_times=exposure_times,
        combined_use_global_avg_reference=args.use_global_avg_reference,
        normalize_to_8bit=args.normalize_to_8bit,
    )
    return output_dir


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return [float(item.strip()) for item in cleaned.split(",") if item.strip()]


def _launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Moment Images Sweep Setup")

    labels = [
        ("Views (.npy)", "views", ""),
        ("Source image", "source_image", ""),
        ("Source image directory (HDR)", "source_image_dir", ""),
        ("Center X", "center_x", ""),
        ("Center Y", "center_y", ""),
        ("View radius", "view_radius", ""),
        ("View grid size", "view_grid_size", "3"),
        ("View distance (optional)", "view_distance", ""),
        ("Output directory", "output_dir", ""),
        ("HDR exposure times (comma-separated)", "hdr_exposure_times", ""),
        ("Window start", "window_start", "15"),
        ("Window stop", "window_stop", "45"),
        ("Window step", "window_step", "10"),
        ("Max disp start (search)", "max_disp_start", "5"),
        ("Max disp stop (search)", "max_disp_stop", "25"),
        ("Max disp step (search)", "max_disp_step", "10"),
        ("Gauss sigma start", "gauss_sigma_start", "3.0"),
        ("Gauss sigma stop", "gauss_sigma_stop", "7.0"),
        ("Gauss sigma step", "gauss_sigma_step", "2.0"),
        ("Disp gauss start", "disp_gauss_start", "0.0"),
        ("Disp gauss stop", "disp_gauss_stop", "1.0"),
        ("Disp gauss step", "disp_gauss_step", "1.0"),
        ("Global thresh start", "global_thresh_start", "0.15"),
        ("Global thresh stop", "global_thresh_stop", "0.15"),
        ("Global thresh step", "global_thresh_step", "0.05"),
        ("Pixel thresh start", "pixel_thresh_start", "0.05"),
        ("Pixel thresh stop", "pixel_thresh_stop", "0.05"),
        ("Pixel thresh step", "pixel_thresh_step", "0.05"),
        ("Zero fill roll", "zero_fill_roll", "True"),
    ]

    entries: dict[str, tk.Entry] = {}

    def choose_file(target: str, filetypes: list[tuple[str, str]] | None = None) -> None:
        selection = filedialog.askopenfilename(filetypes=filetypes)
        if selection:
            entries[target].delete(0, tk.END)
            entries[target].insert(0, selection)

    def choose_directory(target: str) -> None:
        selection = filedialog.askdirectory()
        if selection:
            entries[target].delete(0, tk.END)
            entries[target].insert(0, selection)

    for row, (label, key, default) in enumerate(labels):
        tk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        entry = tk.Entry(root, width=32)
        entry.insert(0, default)
        entry.grid(row=row, column=1, padx=6, pady=2)
        entries[key] = entry
        if key in {"views", "source_image", "source_image_dir"}:
            button_text = "Browse"
            if key == "views":
                filetypes = [("NumPy arrays", "*.npy")]
                tk.Button(root, text=button_text, command=lambda k=key, ft=filetypes: choose_file(k, ft)).grid(
                    row=row, column=2, padx=6, pady=2
                )
            elif key == "source_image":
                filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
                tk.Button(root, text=button_text, command=lambda k=key, ft=filetypes: choose_file(k, ft)).grid(
                    row=row, column=2, padx=6, pady=2
                )
            else:
                tk.Button(
                    root,
                    text=button_text,
                    command=lambda k=key: choose_directory(k),
                ).grid(row=row, column=2, padx=6, pady=2)

    tk.Label(root, text="Strategy").grid(row=len(labels), column=0, sticky="w", padx=6, pady=2)
    strategy_var = tk.StringVar(value="pixel")
    tk.OptionMenu(root, strategy_var, "pixel", "global", "combined").grid(row=len(labels), column=1, padx=6, pady=2, sticky="we")

    save_figures_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Save figures", variable=save_figures_var).grid(
        row=len(labels) + 1, column=1, sticky="w", padx=6, pady=2
    )

    hdr_assemble_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Assemble HDR from exposures", variable=hdr_assemble_var).grid(
        row=len(labels) + 2, column=1, sticky="w", padx=6, pady=2
    )

    analyze_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Analyze metrics after sweep", variable=analyze_var).grid(
        row=len(labels) + 3, column=1, sticky="w", padx=6, pady=2
    )

    live_plot_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Show live metrics plot", variable=live_plot_var).grid(
        row=len(labels) + 4, column=1, sticky="w", padx=6, pady=2
    )

    use_global_avg_reference_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root,
        text="Use global-rectified average as combined reference",
        variable=use_global_avg_reference_var,
    ).grid(row=len(labels) + 5, column=1, sticky="w", padx=6, pady=2)

    normalize_to_8bit_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        root,
        text="Normalize source images to 8-bit",
        variable=normalize_to_8bit_var,
    ).grid(row=len(labels) + 6, column=1, sticky="w", padx=6, pady=2)

    def start_sweep() -> None:
        try:
            args = argparse.Namespace(
                views=entries["views"].get().strip() or None,
                source_image=entries["source_image"].get().strip() or None,
                source_image_dir=entries["source_image_dir"].get().strip() or None,
                center_x=int(entries["center_x"].get()) if entries["center_x"].get().strip() else None,
                center_y=int(entries["center_y"].get()) if entries["center_y"].get().strip() else None,
                view_radius=int(entries["view_radius"].get()) if entries["view_radius"].get().strip() else None,
                view_grid_size=int(entries["view_grid_size"].get()),
                view_distance=_parse_optional_int(entries["view_distance"].get()),
                output_dir=entries["output_dir"].get().strip() or None,
                strategy=strategy_var.get(),
                window_start=int(entries["window_start"].get()),
                window_stop=int(entries["window_stop"].get()),
                window_step=int(entries["window_step"].get()),
                max_disp_start=int(entries["max_disp_start"].get()),
                max_disp_stop=int(entries["max_disp_stop"].get()),
                max_disp_step=int(entries["max_disp_step"].get()),
                gauss_sigma_start=float(entries["gauss_sigma_start"].get()),
                gauss_sigma_stop=float(entries["gauss_sigma_stop"].get()),
                gauss_sigma_step=float(entries["gauss_sigma_step"].get()),
                disp_gauss_start=float(entries["disp_gauss_start"].get()),
                disp_gauss_stop=float(entries["disp_gauss_stop"].get()),
                disp_gauss_step=float(entries["disp_gauss_step"].get()),
                global_thresh_start=float(entries["global_thresh_start"].get()),
                global_thresh_stop=float(entries["global_thresh_stop"].get()),
                global_thresh_step=float(entries["global_thresh_step"].get()),
                pixel_thresh_start=float(entries["pixel_thresh_start"].get()),
                pixel_thresh_stop=float(entries["pixel_thresh_stop"].get()),
                pixel_thresh_step=float(entries["pixel_thresh_step"].get()),
                zero_fill_roll=bool(entries["zero_fill_roll"].get().strip().lower() in {"true", "1", "yes"}),
                save_figures=save_figures_var.get(),
                hdr_assemble=hdr_assemble_var.get(),
                hdr_exposure_times=entries["hdr_exposure_times"].get().strip() or None,
                analyze=analyze_var.get(),
                live_plot=live_plot_var.get(),
                use_global_avg_reference=use_global_avg_reference_var.get(),
                normalize_to_8bit=normalize_to_8bit_var.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        root.withdraw()
        try:
            output_dir = _run_from_namespace(args)
        except Exception as exc:  # noqa: BLE001 - bubble UI errors to user
            root.deiconify()
            messagebox.showerror("Sweep failed", str(exc))
            return

        messagebox.showinfo("Sweep complete", f"Artifacts written to {output_dir}")
        root.destroy()

    tk.Button(root, text="Start sweep", command=start_sweep).grid(
        row=len(labels) + 7, column=1, pady=8
    )

    root.mainloop()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    should_launch_gui = args.gui or (argv is None and len(sys.argv) == 1)
    if should_launch_gui:
        _launch_gui()
        return

    _run_from_namespace(args)


if __name__ == "__main__":
    main()

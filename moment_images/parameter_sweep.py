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
    generate_rectification_figures,
    min_max_normalize,
    rectify_views_from_config,
    save_figures,
    extract_views,
    VIEW_DISTANCE_SCALE_FACTOR,
)


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
    avg_image: np.ndarray,
    rectified_views: np.ndarray,
    config: RectificationRunConfig,
) -> None:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    avg_img_path = run_dir / "avg_rectified.png"
    Image.fromarray(avg_image.astype("uint8")).save(avg_img_path)

    np.save(run_dir / "rectified_views.npy", rectified_views)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2)


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
    if views.ndim != 4 or views.shape[0] != views.shape[1]:
        raise ValueError("views must have shape (N, N, H, W)")
    return views


def _generate_views(
    image_path: Path,
    center_x: int,
    center_y: int,
    radius: int,
    grid_size: int,
    view_distance: int | None,
    cache_dir: Path,
) -> tuple[np.ndarray, Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image = Image.open(image_path)
    normalized = min_max_normalize(image)
    image_np = np.array(normalized)

    distance = view_distance
    if distance is None:
        distance = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * radius))

    base_circle = np.array([center_x, center_y, radius])
    views = extract_views(image_np, base_circle, grid_size, distance)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "generated_views.npy"
    np.save(cache_path, views)
    return views, cache_path


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

    for combo in tqdm(combos, desc="Sweeping parameters", unit="combo"):
        if strategy == "global":
            win, max_disp, gauss, disp_gauss, global_thresh = combo
            pixel_thresh = default_config.pixel_threshold
        elif strategy == "pixel":
            win, max_disp, gauss, disp_gauss, pixel_thresh = combo
            global_thresh = default_config.global_threshold
        else:
            win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh = combo
        config = RectificationRunConfig(
            strategy=strategy,
            similarity_window_size=win,
            refinement_window_size=win,
            max_displacement=max_disp,
            gauss_sigma=gauss,
            gauss_size=int(gauss * 6),
            disp_gauss_sigma=disp_gauss,
            disp_gauss_size=win,
            zero_fill_roll=zero_fill_roll,
            global_threshold=global_thresh,
            pixel_threshold=pixel_thresh,
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
        _save_run_artifacts(output_dir, run_name, result.avg_image, result.rectified_views, config)

        if save_figures_flag:
            figures = generate_rectification_figures(result)  # pass full result so we can save global/pixel maps, originals, overlays, averages
            figures_dir = output_dir / run_name / "figures"
            _persist_figures(figures, figures_dir)

        summary.append(
            {
                "run": run_name,
                "config": asdict(config),
                "avg_intensity": float(np.mean(result.avg_image)),
                "max_intensity": float(np.max(result.avg_image)),
            }
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep im_process rectification parameters")
    parser.add_argument("--views", help="Path to a precomputed (N, N, H, W) numpy array of views")
    parser.add_argument("--source-image", help="Path to the source image to extract views from")
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
    parser.add_argument("--window-start", type=int, default=15)
    parser.add_argument("--window-stop", type=int, default=45)
    parser.add_argument("--window-step", type=int, default=10)
    parser.add_argument("--max-disp-start", type=int, default=5)
    parser.add_argument("--max-disp-stop", type=int, default=25)
    parser.add_argument("--max-disp-step", type=int, default=10)
    parser.add_argument("--gauss-sigma-start", type=float, default=3.0)
    parser.add_argument("--gauss-sigma-stop", type=float, default=7.0)
    parser.add_argument("--gauss-sigma-step", type=float, default=2.0)
    parser.add_argument("--disp-gauss-start", type=float, default=0.0)
    parser.add_argument("--disp-gauss-stop", type=float, default=1.0)
    parser.add_argument("--disp-gauss-step", type=float, default=1.0)
    parser.add_argument("--global-thresh-start", type=float, default=0.25)
    parser.add_argument("--global-thresh-stop", type=float, default=0.25)
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
        "--gui",
        action="store_true",
        help="Launch a minimal GUI for configuring the sweep instead of using CLI flags",
    )
    return parser.parse_args(argv)


def _run_from_namespace(args: argparse.Namespace) -> Path:
    if args.save_figures:
        plt.switch_backend("Agg")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.views is None:
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
        )
        print(f"Generated views once and cached to {cached_views_path}")
    else:
        cached_views_path = Path(args.views)
        views = _load_views(cached_views_path)

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
    )
    return output_dir


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Moment Images Sweep Setup")

    labels = [
        ("Views (.npy)", "views", ""),
        ("Source image", "source_image", ""),
        ("Center X", "center_x", ""),
        ("Center Y", "center_y", ""),
        ("View radius", "view_radius", ""),
        ("View grid size", "view_grid_size", "3"),
        ("View distance (optional)", "view_distance", ""),
        ("Output directory", "output_dir", ""),
        ("Window start", "window_start", "15"),
        ("Window stop", "window_stop", "45"),
        ("Window step", "window_step", "10"),
        ("Max disp start", "max_disp_start", "5"),
        ("Max disp stop", "max_disp_stop", "25"),
        ("Max disp step", "max_disp_step", "10"),
        ("Gauss sigma start", "gauss_sigma_start", "3.0"),
        ("Gauss sigma stop", "gauss_sigma_stop", "7.0"),
        ("Gauss sigma step", "gauss_sigma_step", "2.0"),
        ("Disp gauss start", "disp_gauss_start", "0.0"),
        ("Disp gauss stop", "disp_gauss_stop", "1.0"),
        ("Disp gauss step", "disp_gauss_step", "1.0"),
        ("Global thresh start", "global_thresh_start", "0.25"),
        ("Global thresh stop", "global_thresh_stop", "0.25"),
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

    for row, (label, key, default) in enumerate(labels):
        tk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        entry = tk.Entry(root, width=32)
        entry.insert(0, default)
        entry.grid(row=row, column=1, padx=6, pady=2)
        entries[key] = entry
        if key in {"views", "source_image"}:
            button_text = "Browse"
            filetypes = [("NumPy arrays", "*.npy")] if key == "views" else [("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
            tk.Button(root, text=button_text, command=lambda k=key, ft=filetypes: choose_file(k, ft)).grid(
                row=row, column=2, padx=6, pady=2
            )

    tk.Label(root, text="Strategy").grid(row=len(labels), column=0, sticky="w", padx=6, pady=2)
    strategy_var = tk.StringVar(value="pixel")
    tk.OptionMenu(root, strategy_var, "pixel", "global", "combined").grid(row=len(labels), column=1, padx=6, pady=2, sticky="we")

    save_figures_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Save figures", variable=save_figures_var).grid(
        row=len(labels) + 1, column=1, sticky="w", padx=6, pady=2
    )

    def start_sweep() -> None:
        try:
            args = argparse.Namespace(
                views=entries["views"].get().strip() or None,
                source_image=entries["source_image"].get().strip() or None,
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
        row=len(labels) + 2, column=1, pady=8
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


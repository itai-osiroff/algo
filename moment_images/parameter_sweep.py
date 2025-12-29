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
    rectify_views_from_config,
    save_figures,
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


def sweep_parameters(
    views_path: Path,
    output_dir: Path,
    window_sizes: Sequence[int],
    max_displacements: Sequence[int],
    gauss_sigmas: Sequence[float],
    disp_gauss_sigmas: Sequence[float],
    strategy: str = "pixel",
    save_figures_flag: bool = False,
) -> None:
    views = _load_views(views_path)

    combos = [
        (win, max_disp, gauss, disp_gauss)
        for win in window_sizes
        for max_disp in max_displacements
        for gauss in gauss_sigmas
        for disp_gauss in disp_gauss_sigmas
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []

    for win, max_disp, gauss, disp_gauss in tqdm(combos, desc="Sweeping parameters", unit="combo"):
        config = RectificationRunConfig(
            strategy=strategy,
            similarity_window_size=win,
            refinement_window_size=win,
            max_displacement=max_disp,
            gauss_sigma=gauss,
            gauss_size=int(gauss * 6),
            disp_gauss_sigma=disp_gauss,
            disp_gauss_size=win,
        )

        result = rectify_views_from_config(views, config)
        run_name = f"win{win}_disp{max_disp}_g{gauss}_dg{disp_gauss}"
        _save_run_artifacts(output_dir, run_name, result.avg_image, result.rectified_views, config)

        if save_figures_flag:
            figures = generate_rectification_figures(
                rectified_views=result.rectified_views,
                avg_image=result.avg_image,
                ref_index=result.ref_view_index,
            )
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep im_process rectification parameters")
    parser.add_argument("--views", required=True, help="Path to a (N, N, H, W) numpy array of views")
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
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save rectification figures (non-interactive) for each run",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.save_figures:
        plt.switch_backend("Agg")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    window_sizes = _build_int_range(args.window_start, args.window_stop, args.window_step)
    max_displacements = _build_int_range(args.max_disp_start, args.max_disp_stop, args.max_disp_step)
    gauss_sigmas = _build_float_range(args.gauss_sigma_start, args.gauss_sigma_stop, args.gauss_sigma_step)
    disp_gauss_sigmas = _build_float_range(args.disp_gauss_start, args.disp_gauss_stop, args.disp_gauss_step)

    sweep_parameters(
        views_path=Path(args.views),
        output_dir=output_dir,
        window_sizes=window_sizes,
        max_displacements=max_displacements,
        gauss_sigmas=gauss_sigmas,
        disp_gauss_sigmas=disp_gauss_sigmas,
        strategy=args.strategy,
        save_figures_flag=args.save_figures,
    )


if __name__ == "__main__":
    main()


"""Interactive pairwise image gain estimation via linear regression."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from scipy import stats

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from im_process import save_workspace_variables
else:
    from .im_process import save_workspace_variables

SCATTER_SAMPLE_LIMIT = 100_000
SCATTER_SAMPLE_SEED = 12345


@dataclass(frozen=True)
class LinearGainResult:
    """Summary statistics for one pairwise image regression run."""

    fit_intercept: bool
    regression_mode: str
    gain: float
    intercept: float
    r_value: float
    r_squared: float
    p_value: float
    std_err: float
    pixel_count_total: int
    pixel_count_valid: int
    image_shape: tuple[int, int]
    regression_equation: str


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load one image as a 2D grayscale ``float64`` array."""

    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with Image.open(image_path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {array.shape} for {image_path}."
        )
    return np.asarray(array, dtype=np.float64)


def _pearson_r(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson correlation when defined, otherwise ``nan``."""

    if x_values.size < 2:
        return float("nan")
    x_std = float(np.std(x_values, dtype=np.float64))
    y_std = float(np.std(y_values, dtype=np.float64))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _build_equation(gain: float, intercept: float, fit_intercept: bool) -> str:
    if fit_intercept:
        return f"image2 = {intercept:.6f} + {gain:.6f} * image1"
    return f"image2 = {gain:.6f} * image1"


def compute_linear_gain(
    image1: np.ndarray,
    image2: np.ndarray,
    fit_intercept: bool = True,
) -> tuple[LinearGainResult, dict[str, np.ndarray]]:
    """Flatten paired images, filter invalid values, and fit linear gain."""

    image1_arr = np.asarray(image1, dtype=np.float64)
    image2_arr = np.asarray(image2, dtype=np.float64)
    if image1_arr.ndim != 2 or image2_arr.ndim != 2:
        raise ValueError("image1 and image2 must both be 2D grayscale arrays.")
    if image1_arr.shape != image2_arr.shape:
        raise ValueError(
            f"image1 and image2 must have identical image shapes, got "
            f"{image1_arr.shape} and {image2_arr.shape}."
        )

    image1_flat = image1_arr.reshape(-1)
    image2_flat = image2_arr.reshape(-1)
    valid_mask = np.isfinite(image1_flat) & np.isfinite(image2_flat)
    x_valid = image1_flat[valid_mask]
    y_valid = image2_flat[valid_mask]

    if x_valid.size < 2:
        raise ValueError(
            "At least two finite paired pixels are required to fit the regression."
        )

    if fit_intercept:
        regression = stats.linregress(x_valid, y_valid)
        gain = float(regression.slope)
        intercept = float(regression.intercept)
        r_value = float(regression.rvalue)
        p_value = float(regression.pvalue)
        std_err = float(regression.stderr)
        r_squared = float(r_value * r_value)
        regression_mode = "with_intercept"
    else:
        denominator = float(np.dot(x_valid, x_valid))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "Gain-only regression requires a non-zero finite sum of squared image1 values."
            )
        gain = float(np.dot(x_valid, y_valid) / denominator)
        intercept = 0.0
        r_value = _pearson_r(x_valid, y_valid)
        r_squared = float(r_value * r_value) if np.isfinite(r_value) else float("nan")
        p_value = float("nan")
        std_err = float("nan")
        regression_mode = "gain_only"

    y_pred = intercept + gain * x_valid
    residuals = y_valid - y_pred
    result = LinearGainResult(
        fit_intercept=bool(fit_intercept),
        regression_mode=regression_mode,
        gain=gain,
        intercept=intercept,
        r_value=r_value,
        r_squared=r_squared,
        p_value=p_value,
        std_err=std_err,
        pixel_count_total=int(image1_flat.size),
        pixel_count_valid=int(x_valid.size),
        image_shape=tuple(int(value) for value in image1_arr.shape),
        regression_equation=_build_equation(gain, intercept, fit_intercept=fit_intercept),
    )
    workspace = {
        "image1_flat": image1_flat,
        "image2_flat": image2_flat,
        "valid_mask": valid_mask,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "y_pred": y_pred,
        "residuals": residuals,
    }
    return result, workspace


def select_scatter_sample(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_points: int = SCATTER_SAMPLE_LIMIT,
    random_seed: int = SCATTER_SAMPLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic scatter sample suitable for plotting."""

    x_arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_values and y_values must have identical shapes.")
    if x_arr.size <= max_points:
        return x_arr.copy(), y_arr.copy()

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(x_arr.size, size=max_points, replace=False)
    indices.sort()
    return x_arr[indices], y_arr[indices]


def build_visualization(
    image1: np.ndarray,
    image2: np.ndarray,
    result: LinearGainResult,
    workspace: Mapping[str, np.ndarray],
) -> Figure:
    """Create a 2x2 figure with previews, scatter, and residual histogram."""

    x_valid = np.asarray(workspace["x_valid"], dtype=np.float64)
    y_valid = np.asarray(workspace["y_valid"], dtype=np.float64)
    residuals = np.asarray(workspace["residuals"], dtype=np.float64)
    scatter_sample_x, scatter_sample_y = select_scatter_sample(x_valid, y_valid)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_image1, ax_image2, ax_scatter, ax_residuals = axes.ravel()

    ax_image1.imshow(np.asarray(image1, dtype=np.float64), cmap="gray")
    ax_image1.set_title("Image 1 (X)")
    ax_image1.axis("off")

    ax_image2.imshow(np.asarray(image2, dtype=np.float64), cmap="gray")
    ax_image2.set_title("Image 2 (Y)")
    ax_image2.axis("off")

    ax_scatter.scatter(
        scatter_sample_x,
        scatter_sample_y,
        s=6,
        alpha=0.25,
        edgecolors="none",
        color="#1d3557",
    )
    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    if np.isclose(x_min, x_max):
        delta = 1.0 if x_min == 0.0 else abs(x_min) * 0.05
        line_x = np.array([x_min - delta, x_max + delta], dtype=np.float64)
    else:
        line_x = np.linspace(x_min, x_max, 200, dtype=np.float64)
    line_y = result.intercept + result.gain * line_x
    ax_scatter.plot(line_x, line_y, color="#d62828", linewidth=2.0)
    ax_scatter.set_xlabel("Image 1 pixel value")
    ax_scatter.set_ylabel("Image 2 pixel value")
    ax_scatter.set_title("Flattened pixel regression")
    ax_scatter.grid(True, linestyle=":", alpha=0.4)
    summary_text = "\n".join(
        [
            f"Mode: {result.regression_mode}",
            result.regression_equation,
            f"R = {result.r_value:.6f}",
            f"R^2 = {result.r_squared:.6f}",
            f"Valid pixels = {result.pixel_count_valid}",
        ]
    )
    ax_scatter.text(
        0.02,
        0.98,
        summary_text,
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    bin_count = min(100, max(20, int(np.sqrt(residuals.size))))
    ax_residuals.hist(residuals, bins=bin_count, color="#457b9d", alpha=0.85)
    ax_residuals.axvline(0.0, color="#d62828", linestyle="--", linewidth=1.5)
    ax_residuals.set_title("Residual distribution")
    ax_residuals.set_xlabel("Residual")
    ax_residuals.set_ylabel("Count")
    ax_residuals.grid(True, linestyle=":", alpha=0.4)

    figure.suptitle("Pairwise image linear gain analysis", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def format_result(result: LinearGainResult) -> str:
    """Format regression results for terminal output."""

    lines = [
        "Pairwise Image Linear Gain Result",
        f"Regression mode: {result.regression_mode}",
        f"Equation: {result.regression_equation}",
        f"Image shape: {result.image_shape}",
        f"Total pixels: {result.pixel_count_total}",
        f"Valid paired pixels: {result.pixel_count_valid}",
        f"Gain: {result.gain:.6f}",
        f"Intercept: {result.intercept:.6f}",
        f"R value: {result.r_value:.6f}",
        f"R squared: {result.r_squared:.6f}",
        f"P value: {result.p_value:.6g}",
        f"Std err: {result.std_err:.6g}",
    ]
    return "\n".join(lines)


def _artifact_path(output_base_path: str | Path, suffix: str) -> Path:
    path = Path(output_base_path)
    if path.suffix.lower() in {".npz", ".mat", ".png"}:
        return path.with_suffix(suffix)
    return path.with_name(f"{path.name}{suffix}")


def _build_save_payload(
    image1_path: str | Path,
    image2_path: str | Path,
    image1: np.ndarray,
    image2: np.ndarray,
    result: LinearGainResult,
    workspace: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    scatter_sample_x, scatter_sample_y = select_scatter_sample(
        workspace["x_valid"],
        workspace["y_valid"],
    )
    payload: dict[str, Any] = {
        "image1_path": str(image1_path),
        "image2_path": str(image2_path),
        "image_shape": np.asarray(result.image_shape, dtype=np.int64),
        "fit_intercept": np.asarray(result.fit_intercept, dtype=np.bool_),
        "regression_mode": result.regression_mode,
        "image1": np.asarray(image1, dtype=np.float64),
        "image2": np.asarray(image2, dtype=np.float64),
        "gain": np.asarray(result.gain, dtype=np.float64),
        "intercept": np.asarray(result.intercept, dtype=np.float64),
        "r_value": np.asarray(result.r_value, dtype=np.float64),
        "r_squared": np.asarray(result.r_squared, dtype=np.float64),
        "p_value": np.asarray(result.p_value, dtype=np.float64),
        "std_err": np.asarray(result.std_err, dtype=np.float64),
        "pixel_count_total": np.asarray(result.pixel_count_total, dtype=np.int64),
        "pixel_count_valid": np.asarray(result.pixel_count_valid, dtype=np.int64),
        "regression_equation": result.regression_equation,
        "scatter_sample_x": scatter_sample_x,
        "scatter_sample_y": scatter_sample_y,
    }
    payload.update(workspace)
    return payload


def save_linear_gain_results(
    output_base_path: str | Path,
    figure: Figure,
    *,
    image1_path: str | Path,
    image2_path: str | Path,
    image1: np.ndarray,
    image2: np.ndarray,
    result: LinearGainResult,
    workspace: Mapping[str, np.ndarray],
) -> dict[str, Path]:
    """Persist workspace variables plus a PNG copy of the visualization."""

    npz_path = _artifact_path(output_base_path, ".npz")
    mat_path = _artifact_path(output_base_path, ".mat")
    png_path = _artifact_path(output_base_path, ".png")
    payload = _build_save_payload(
        image1_path=image1_path,
        image2_path=image2_path,
        image1=image1,
        image2=image2,
        result=result,
        workspace=workspace,
    )
    save_workspace_variables(str(npz_path), payload)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization figure to: {png_path}")
    return {"npz": npz_path, "mat": mat_path, "png": png_path}


def _prompt_nonempty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty.")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes", "1", "true"}:
            return True
        if response in {"n", "no", "0", "false"}:
            return False
        print("Please answer with yes or no.")


def main() -> None:
    """Run the interactive pairwise image linear gain workflow."""

    try:
        image1_path = _prompt_nonempty("Enter path for image 1 (independent variable X): ")
        image2_path = _prompt_nonempty("Enter path for image 2 (dependent variable Y): ")
        fit_intercept = _prompt_yes_no(
            "Include an intercept in the regression?",
            default=True,
        )

        image1 = load_grayscale_image(image1_path)
        image2 = load_grayscale_image(image2_path)
        result, workspace = compute_linear_gain(
            image1=image1,
            image2=image2,
            fit_intercept=fit_intercept,
        )
        figure = build_visualization(image1, image2, result, workspace)

        print()
        print(format_result(result))
        print()
        print("Close the figure window to continue to the save prompt.")
        plt.show()

        if _prompt_yes_no("Save results and variables?", default=False):
            output_base_path = _prompt_nonempty(
                "Enter output base path for .npz/.mat/.png files: "
            )
            save_linear_gain_results(
                output_base_path=output_base_path,
                figure=figure,
                image1_path=image1_path,
                image2_path=image2_path,
                image1=image1,
                image2=image2,
                result=result,
                workspace=workspace,
            )
    except Exception as exc:  # noqa: BLE001 - interactive workflow should print user-facing errors.
        print(f"Error: {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()

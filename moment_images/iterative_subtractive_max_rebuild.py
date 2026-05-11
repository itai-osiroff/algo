"""Reference-image rebuilding utilities for combined rectification.

This module is designed to operate on the output of the global rectification
stage before a pixelwise refinement stage consumes an external reference view.
The rebuild stays inexpensive by working with a single globally aligned view
stack, a cheap per-pixel consistency estimate, and a small number of tapered
patch pastes chosen by greedy non-maximum suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


DEFAULT_DISPLACEMENT_METRIC_FAMILY = "intensity"
DEFAULT_DISPLACEMENT_ALPHA = 1.0
BLURRED_GRADIENT_CORR_METRIC_FAMILY = "blurred_gradient_corr"
BLURRED_LAPLACIAN_CORR_METRIC_FAMILY = "blurred_laplacian_corr"
BLURRED_COMPOSITE_CORR_METRIC_FAMILY = "blurred_composite_corr"
LEGACY_DISPLACEMENT_METRIC_FAMILY_ALIASES = {
    "ncc_gradient": BLURRED_GRADIENT_CORR_METRIC_FAMILY,
    "ncc_laplacian": BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
    "composite": BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
}
SUPPORTED_DISPLACEMENT_METRIC_FAMILIES = {
    DEFAULT_DISPLACEMENT_METRIC_FAMILY,
    BLURRED_GRADIENT_CORR_METRIC_FAMILY,
    BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
    BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
}
SUPPORTED_DISPLACEMENT_COMPONENTS = ("gradient", "laplacian")


@dataclass(frozen=True)
class ResolvedReferenceMetricSpec:
    """Canonical metric-family configuration for reference rebuilding."""

    family: str
    alpha: float
    raw_component_weights: Dict[str, float]
    normalized_component_weights: Dict[str, float]
    enabled_auxiliary_components: Tuple[str, ...]
    auxiliary_components_active: bool


@dataclass(frozen=True)
class IterativeSubtractiveMaxRebuildResult:
    """Artifacts produced while rebuilding an external reference image."""

    reference_image: np.ndarray
    sparse_rebuild: np.ndarray
    robust_base_image: np.ndarray
    mean_image: np.ndarray
    std_image: np.ndarray
    feature_map: np.ndarray
    saliency_map: np.ndarray
    peak_coordinates: np.ndarray
    peak_scores: np.ndarray
    donor_view_indices: np.ndarray
    coverage_map: np.ndarray


def normalize_displacement_metric_family(
    metric_family: object | None,
    *,
    default: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY,
) -> str:
    """Return the canonical metric-family name used by the rebuild."""

    family = default if metric_family is None else str(metric_family).strip().lower()
    canonical_family = LEGACY_DISPLACEMENT_METRIC_FAMILY_ALIASES.get(family, family)
    if canonical_family not in SUPPORTED_DISPLACEMENT_METRIC_FAMILIES:
        raise ValueError(
            "metric_family must be one of "
            "'intensity', 'blurred_gradient_corr', 'blurred_laplacian_corr', "
            "or 'blurred_composite_corr'. Legacy aliases "
            "'ncc_gradient', 'ncc_laplacian', and 'composite' are also accepted."
        )
    return canonical_family


def resolve_reference_metric_spec(
    metric_family: object | None,
    *,
    alpha: object | None = DEFAULT_DISPLACEMENT_ALPHA,
    component_weights: Optional[Dict[object, object]] = None,
) -> ResolvedReferenceMetricSpec:
    """Validate and canonicalize the feature family used for saliency."""

    resolved_alpha = DEFAULT_DISPLACEMENT_ALPHA if alpha is None else float(alpha)
    if not np.isfinite(resolved_alpha):
        raise ValueError("alpha must be finite.")
    if resolved_alpha < 0.0 or resolved_alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    family = normalize_displacement_metric_family(metric_family)

    if family == DEFAULT_DISPLACEMENT_METRIC_FAMILY:
        if component_weights is not None:
            raise ValueError("Intensity metric does not accept component_weights.")
        if resolved_alpha != DEFAULT_DISPLACEMENT_ALPHA:
            raise ValueError("Intensity metric requires alpha=1.0.")
        return ResolvedReferenceMetricSpec(
            family=family,
            alpha=DEFAULT_DISPLACEMENT_ALPHA,
            raw_component_weights={},
            normalized_component_weights={},
            enabled_auxiliary_components=(),
            auxiliary_components_active=False,
        )

    if family == BLURRED_GRADIENT_CORR_METRIC_FAMILY:
        if component_weights is not None:
            raise ValueError(
                "blurred_gradient_corr does not accept component_weights."
            )
        weights = {"gradient": 1.0}
        auxiliary_active = resolved_alpha < 1.0
        return ResolvedReferenceMetricSpec(
            family=family,
            alpha=resolved_alpha,
            raw_component_weights=weights,
            normalized_component_weights=weights,
            enabled_auxiliary_components=("gradient",) if auxiliary_active else (),
            auxiliary_components_active=auxiliary_active,
        )

    if family == BLURRED_LAPLACIAN_CORR_METRIC_FAMILY:
        if component_weights is not None:
            raise ValueError(
                "blurred_laplacian_corr does not accept component_weights."
            )
        weights = {"laplacian": 1.0}
        auxiliary_active = resolved_alpha < 1.0
        return ResolvedReferenceMetricSpec(
            family=family,
            alpha=resolved_alpha,
            raw_component_weights=weights,
            normalized_component_weights=weights,
            enabled_auxiliary_components=("laplacian",) if auxiliary_active else (),
            auxiliary_components_active=auxiliary_active,
        )

    parsed_component_weights: Dict[str, float] = {}
    if component_weights is not None:
        for key, value in component_weights.items():
            name = str(key).strip().lower()
            if name not in SUPPORTED_DISPLACEMENT_COMPONENTS:
                raise ValueError(
                    "Composite component weights only support "
                    "'gradient' and 'laplacian'."
                )
            parsed_value = float(value)
            if not np.isfinite(parsed_value):
                raise ValueError("Composite component weights must be finite.")
            if parsed_value < 0.0:
                raise ValueError("Composite component weights must be non-negative.")
            parsed_component_weights[name] = parsed_value

    weight_mass = float(sum(parsed_component_weights.values()))
    if resolved_alpha < 1.0 and weight_mass <= 0.0:
        raise ValueError(
            "blurred_composite_corr requires positive non-intensity component "
            "weights when alpha < 1."
        )

    normalized_component_weights: Dict[str, float] = {}
    if weight_mass > 0.0:
        normalized_component_weights = {
            key: float(value / weight_mass)
            for key, value in parsed_component_weights.items()
            if value > 0.0
        }

    enabled_auxiliary_components = (
        tuple(normalized_component_weights.keys())
        if resolved_alpha < 1.0 and normalized_component_weights
        else ()
    )
    return ResolvedReferenceMetricSpec(
        family=family,
        alpha=resolved_alpha,
        raw_component_weights=parsed_component_weights,
        normalized_component_weights=normalized_component_weights,
        enabled_auxiliary_components=enabled_auxiliary_components,
        auxiliary_components_active=bool(enabled_auxiliary_components),
    )


def iterative_subtractive_max_rebuild(
    global_rectified_views: np.ndarray,
    *,
    global_average: Optional[np.ndarray] = None,
    metric_family: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY,
    alpha: float = DEFAULT_DISPLACEMENT_ALPHA,
    component_weights: Optional[Dict[str, float]] = None,
    robust_base_mode: str = "median",
    trim_fraction: float = 0.1,
    base_reference: Optional[np.ndarray] = None,
    feature_sigma: float = 1.0,
    saliency_smooth_sigma: float = 1.0,
    consistency_eps: float = 0.05,
    patch_radius: int = 5,
    peak_radius: Optional[int] = None,
    peak_threshold_rel: float = 0.1,
    max_peaks: Optional[int] = None,
    donor_mode: str = "best_view",
    patch_taper: str = "hann",
    background_blend: float = 0.2,
) -> IterativeSubtractiveMaxRebuildResult:
    """Rebuild an external reference from globally rectified views.

    Args:
        global_rectified_views: Globally aligned view stack shaped either
            ``(N, N, H, W)`` or ``(V, H, W)``.
        global_average: Optional precomputed average image. When omitted, the
            mean is computed from ``global_rectified_views``.
        metric_family: Feature family used to score salient regions. Supported
            families mirror the pixelwise rectification metric families.
        alpha: Intensity-vs-auxiliary feature blend weight, following the same
            semantics as the pixelwise displacement-score blend.
        component_weights: Optional composite weights for gradient/laplacian.
        robust_base_mode: ``median`` or ``trimmed_mean``.
        trim_fraction: Fraction trimmed from each tail when
            ``robust_base_mode='trimmed_mean'``.
        base_reference: Optional dense background reference to blend against.
            When omitted, a robust base is computed from the globally aligned
            view stack.
        feature_sigma: Optional Gaussian pre-smoothing before feature creation.
        saliency_smooth_sigma: Optional Gaussian smoothing for the saliency map.
        consistency_eps: Denominator stabilizer for the inverse-variance term.
        patch_radius: Radius of pasted patches.
        peak_radius: Radius used for greedy non-maximum suppression. Defaults
            to ``patch_radius``.
        peak_threshold_rel: Relative stop threshold with respect to the best
            saliency peak.
        max_peaks: Optional cap on the number of selected peaks.
        donor_mode: ``robust_base`` or ``best_view``.
        patch_taper: ``hann``, ``gaussian``, or ``flat``.
        background_blend: Weight assigned to the dense base image in the final
            output. The sparse rebuild receives ``1 - background_blend``.

    Returns:
        A result object containing the rebuilt reference and useful diagnostics.
    """

    spec = resolve_reference_metric_spec(
        metric_family,
        alpha=alpha,
        component_weights=component_weights,
    )
    view_stack = _flatten_global_rectified_views(global_rectified_views)
    _, height, width = view_stack.shape

    if patch_radius < 0:
        raise ValueError("patch_radius must be >= 0.")
    if peak_radius is None:
        peak_radius = patch_radius
    if peak_radius < 0:
        raise ValueError("peak_radius must be >= 0.")
    if feature_sigma < 0.0:
        raise ValueError("feature_sigma must be >= 0.")
    if saliency_smooth_sigma < 0.0:
        raise ValueError("saliency_smooth_sigma must be >= 0.")
    if consistency_eps <= 0.0:
        raise ValueError("consistency_eps must be > 0.")
    if not 0.0 <= peak_threshold_rel <= 1.0:
        raise ValueError("peak_threshold_rel must be between 0 and 1.")
    if not 0.0 <= background_blend <= 1.0:
        raise ValueError("background_blend must be between 0 and 1.")
    if donor_mode not in {"robust_base", "best_view"}:
        raise ValueError("donor_mode must be either 'robust_base' or 'best_view'.")

    mean_image = (
        _validate_reference_image(global_average, expected_shape=(height, width))
        if global_average is not None
        else np.mean(view_stack, axis=0, dtype=np.float32)
    )
    std_image = np.std(view_stack, axis=0, dtype=np.float32)

    robust_base_image = (
        _validate_reference_image(base_reference, expected_shape=(height, width))
        if base_reference is not None
        else _compute_robust_base(view_stack, mode=robust_base_mode, trim_fraction=trim_fraction)
    )

    feature_map = _compute_metric_feature_map(
        mean_image,
        spec=spec,
        feature_sigma=feature_sigma,
    )
    std_norm = _normalize_nonnegative_map(std_image)
    saliency_raw = feature_map / (consistency_eps + std_norm)
    if saliency_smooth_sigma > 0.0:
        saliency_raw = gaussian_filter(
            saliency_raw,
            sigma=saliency_smooth_sigma,
            mode="nearest",
        )
    saliency_map = _normalize_nonnegative_map(saliency_raw)

    resolved_max_peaks = _resolve_max_peaks(
        max_peaks,
        height=height,
        width=width,
        suppression_radius=peak_radius,
    )
    peak_coordinates, peak_scores = _select_peaks_greedy_nms(
        saliency_map,
        suppression_radius=peak_radius,
        peak_threshold_rel=peak_threshold_rel,
        max_peaks=resolved_max_peaks,
    )

    patch_window = _make_patch_window(patch_radius, taper=patch_taper)
    if donor_mode == "best_view":
        donor_feature_stack = np.stack(
            [
                _compute_metric_feature_map(view, spec=spec, feature_sigma=feature_sigma)
                for view in view_stack
            ],
            axis=0,
        )
    else:
        donor_feature_stack = None

    accum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    coverage_map = np.zeros((height, width), dtype=np.float32)
    donor_view_indices = np.full((peak_coordinates.shape[0],), -1, dtype=np.int32)

    for peak_idx, ((center_y, center_x), peak_score) in enumerate(
        zip(peak_coordinates, peak_scores)
    ):
        y0, y1, x0, x1 = _bounded_patch_bounds(
            height=height,
            width=width,
            center_y=int(center_y),
            center_x=int(center_x),
            radius=patch_radius,
        )
        window_patch = _crop_patch_window(
            patch_window,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            center_y=int(center_y),
            center_x=int(center_x),
            radius=patch_radius,
        )
        donor_patch, donor_view_index = _select_donor_patch(
            donor_mode=donor_mode,
            view_stack=view_stack,
            donor_feature_stack=donor_feature_stack,
            robust_base_image=robust_base_image,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            window_patch=window_patch,
        )
        donor_view_indices[peak_idx] = donor_view_index
        weighted_window = window_patch * float(peak_score)
        accum[y0:y1, x0:x1] += donor_patch * weighted_window
        weight_sum[y0:y1, x0:x1] += weighted_window
        coverage_map[y0:y1, x0:x1] += window_patch

    sparse_rebuild = robust_base_image.copy()
    valid_weights = weight_sum > 1e-8
    sparse_rebuild[valid_weights] = accum[valid_weights] / weight_sum[valid_weights]

    reference_image = (
        background_blend * robust_base_image
        + (1.0 - background_blend) * sparse_rebuild
    ).astype(np.float32)

    return IterativeSubtractiveMaxRebuildResult(
        reference_image=reference_image,
        sparse_rebuild=sparse_rebuild.astype(np.float32),
        robust_base_image=robust_base_image.astype(np.float32),
        mean_image=mean_image.astype(np.float32),
        std_image=std_image.astype(np.float32),
        feature_map=feature_map.astype(np.float32),
        saliency_map=saliency_map.astype(np.float32),
        peak_coordinates=peak_coordinates.astype(np.int32),
        peak_scores=peak_scores.astype(np.float32),
        donor_view_indices=donor_view_indices,
        coverage_map=coverage_map.astype(np.float32),
    )


def rebuild_reference_image(
    global_rectified_views: np.ndarray,
    **kwargs: object,
) -> IterativeSubtractiveMaxRebuildResult:
    """Alias for the main rebuild entrypoint."""

    return iterative_subtractive_max_rebuild(global_rectified_views, **kwargs)


def _flatten_global_rectified_views(global_rectified_views: np.ndarray) -> np.ndarray:
    """Normalize accepted global-rectification outputs to ``(V, H, W)``."""

    views = np.asarray(global_rectified_views, dtype=np.float32)
    views = np.nan_to_num(views, nan=0.0, posinf=0.0, neginf=0.0)

    if views.ndim == 4:
        rows, cols, height, width = views.shape
        return views.reshape(rows * cols, height, width)
    if views.ndim == 3:
        return views

    raise ValueError(
        "global_rectified_views must have shape (N, N, H, W) or (V, H, W)."
    )


def _validate_reference_image(
    image: np.ndarray | None,
    *,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    """Validate a supplied dense reference image."""

    if image is None:
        raise ValueError("image must not be None.")
    image_arr = np.asarray(image, dtype=np.float32)
    image_arr = np.nan_to_num(image_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if image_arr.shape != expected_shape:
        raise ValueError(
            f"reference image must have shape {expected_shape}, got {image_arr.shape}."
        )
    return image_arr


def _compute_robust_base(
    view_stack: np.ndarray,
    *,
    mode: str,
    trim_fraction: float,
) -> np.ndarray:
    """Compute a dense low-cost robust background image."""

    if mode == "median":
        return np.median(view_stack, axis=0).astype(np.float32)
    if mode == "trimmed_mean":
        return _trimmed_mean(view_stack, trim_fraction=trim_fraction)

    raise ValueError("robust_base_mode must be either 'median' or 'trimmed_mean'.")


def _trimmed_mean(view_stack: np.ndarray, *, trim_fraction: float) -> np.ndarray:
    """Compute a per-pixel trimmed mean over the flattened view axis."""

    if trim_fraction < 0.0 or trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be in the range [0, 0.5).")

    num_views = int(view_stack.shape[0])
    trim_count = int(np.floor(num_views * trim_fraction))
    if trim_count <= 0 or num_views - (2 * trim_count) <= 0:
        return np.mean(view_stack, axis=0, dtype=np.float32)

    sorted_views = np.sort(view_stack, axis=0)
    trimmed = sorted_views[trim_count : num_views - trim_count]
    return np.mean(trimmed, axis=0, dtype=np.float32)


def _compute_metric_feature_map(
    image: np.ndarray,
    *,
    spec: ResolvedReferenceMetricSpec,
    feature_sigma: float,
) -> np.ndarray:
    """Build a normalized feature map that mirrors pixelwise metric semantics."""

    work_image = np.asarray(image, dtype=np.float32)
    work_image = np.nan_to_num(work_image, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_sigma > 0.0:
        work_image = gaussian_filter(work_image, sigma=feature_sigma, mode="nearest")

    components = {
        "intensity": _normalize_nonnegative_map(np.maximum(work_image, 0.0)),
        "gradient": _normalize_nonnegative_map(_gradient_magnitude(work_image)),
        "laplacian": _normalize_nonnegative_map(np.abs(laplace(work_image, mode="nearest"))),
    }

    blended = float(spec.alpha) * components["intensity"]
    if spec.auxiliary_components_active:
        auxiliary = np.zeros_like(blended)
        for name, weight in spec.normalized_component_weights.items():
            auxiliary += float(weight) * components[name]
        blended = blended + (1.0 - float(spec.alpha)) * auxiliary

    return _normalize_nonnegative_map(blended)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute the Euclidean gradient magnitude."""

    grad_y, grad_x = np.gradient(image)
    return np.hypot(grad_y, grad_x).astype(np.float32)


def _normalize_nonnegative_map(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a non-negative map to ``[0, 1]`` using a robust high percentile."""

    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    high = float(np.percentile(arr, 99.0))
    if high <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / high, 0.0, 1.0).astype(np.float32)


def _resolve_max_peaks(
    requested_max_peaks: Optional[int],
    *,
    height: int,
    width: int,
    suppression_radius: int,
) -> int:
    """Bound the number of selected peaks to keep the rebuild inexpensive."""

    if requested_max_peaks is not None:
        if requested_max_peaks < 1:
            raise ValueError("max_peaks must be >= 1 when provided.")
        return int(requested_max_peaks)

    support = max(1, (2 * suppression_radius + 1) ** 2)
    approx_non_overlapping = max(1, (height * width) // support)
    return int(min(128, approx_non_overlapping))


def _select_peaks_greedy_nms(
    saliency_map: np.ndarray,
    *,
    suppression_radius: int,
    peak_threshold_rel: float,
    max_peaks: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily select peaks while suppressing their neighborhoods."""

    work = np.asarray(saliency_map, dtype=np.float32).copy()
    best_peak = float(np.max(work))
    if best_peak <= 0.0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    stop_threshold = float(best_peak * peak_threshold_rel)
    peaks = []
    peak_scores = []
    height, width = work.shape

    while len(peaks) < max_peaks:
        flat_index = int(np.argmax(work))
        score = float(work.flat[flat_index])
        if score <= 0.0 or score < stop_threshold:
            break

        center_y, center_x = np.unravel_index(flat_index, work.shape)
        peaks.append((int(center_y), int(center_x)))
        peak_scores.append(score)

        y0, y1, x0, x1 = _bounded_patch_bounds(
            height=height,
            width=width,
            center_y=int(center_y),
            center_x=int(center_x),
            radius=suppression_radius,
        )
        work[y0:y1, x0:x1] = 0.0

    return np.asarray(peaks, dtype=np.int32), np.asarray(peak_scores, dtype=np.float32)


def _make_patch_window(radius: int, *, taper: str) -> np.ndarray:
    """Create a 2D taper window used for seam-reduced patch blending."""

    size = 2 * radius + 1
    if size <= 1:
        return np.ones((1, 1), dtype=np.float32)

    taper_name = str(taper).strip().lower()
    if taper_name == "hann":
        win_1d = np.hanning(size).astype(np.float32)
    elif taper_name == "gaussian":
        coords = np.arange(size, dtype=np.float32) - radius
        sigma = max(radius / 2.0, 0.5)
        win_1d = np.exp(-0.5 * (coords / sigma) ** 2).astype(np.float32)
    elif taper_name == "flat":
        win_1d = np.ones((size,), dtype=np.float32)
    else:
        raise ValueError("patch_taper must be 'hann', 'gaussian', or 'flat'.")

    window = np.outer(win_1d, win_1d).astype(np.float32)
    max_value = float(np.max(window))
    if max_value <= 0.0:
        return np.ones((size, size), dtype=np.float32)
    return window / max_value


def _bounded_patch_bounds(
    *,
    height: int,
    width: int,
    center_y: int,
    center_x: int,
    radius: int,
) -> Tuple[int, int, int, int]:
    """Return boundary-safe patch bounds around a center point."""

    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    return y0, y1, x0, x1


def _crop_patch_window(
    patch_window: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray:
    """Crop the full taper window so it matches a boundary-clipped patch."""

    full_y0 = center_y - radius
    full_x0 = center_x - radius
    window_y0 = y0 - full_y0
    window_y1 = window_y0 + (y1 - y0)
    window_x0 = x0 - full_x0
    window_x1 = window_x0 + (x1 - x0)
    return patch_window[window_y0:window_y1, window_x0:window_x1]


def _select_donor_patch(
    *,
    donor_mode: str,
    view_stack: np.ndarray,
    donor_feature_stack: Optional[np.ndarray],
    robust_base_image: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    window_patch: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Choose the patch donor for one selected peak."""

    if donor_mode == "robust_base":
        return robust_base_image[y0:y1, x0:x1], -1

    if donor_feature_stack is None:
        raise ValueError("donor_feature_stack must be provided for donor_mode='best_view'.")

    local_scores = np.sum(
        donor_feature_stack[:, y0:y1, x0:x1] * window_patch[None, :, :],
        axis=(1, 2),
    )
    donor_index = int(np.argmax(local_scores))
    return view_stack[donor_index, y0:y1, x0:x1], donor_index


__all__ = [
    "BLURRED_COMPOSITE_CORR_METRIC_FAMILY",
    "BLURRED_GRADIENT_CORR_METRIC_FAMILY",
    "BLURRED_LAPLACIAN_CORR_METRIC_FAMILY",
    "DEFAULT_DISPLACEMENT_ALPHA",
    "DEFAULT_DISPLACEMENT_METRIC_FAMILY",
    "IterativeSubtractiveMaxRebuildResult",
    "ResolvedReferenceMetricSpec",
    "iterative_subtractive_max_rebuild",
    "normalize_displacement_metric_family",
    "rebuild_reference_image",
    "resolve_reference_metric_spec",
]

"""
Interfaces for rectifying view grids extracted from light-field captures.

View stacks are expected as ``(N, N, H, W)`` arrays of floating-point grayscale
intensities. Global rectification strategies operate on translation similarity
matrices shaped ``(N, N, N, N, 2)``, while the pixelwise variant relies on a
``PixelwiseSimilarityResult`` containing dense coordinate and score maps. Both
approaches ultimately align every view to a selected reference and average the
results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

import cv2 as cv
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter


DEFAULT_CORRELATION_PAD_WIDTH = int(125 * 0.45)
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
    "intensity",
    BLURRED_GRADIENT_CORR_METRIC_FAMILY,
    BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
    BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
}
SUPPORTED_DISPLACEMENT_COMPONENTS = ("gradient", "laplacian")


def normalize_displacement_metric_family(
    metric_family: object | None,
    *,
    default: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY,
    allow_unknown: bool = False,
) -> str:
    """Return the canonical family name for local displacement score blending."""

    family = default if metric_family is None else str(metric_family).strip().lower()
    canonical_family = LEGACY_DISPLACEMENT_METRIC_FAMILY_ALIASES.get(family, family)
    if canonical_family not in SUPPORTED_DISPLACEMENT_METRIC_FAMILIES:
        if allow_unknown:
            return canonical_family
        raise ValueError(
            "metric family must be one of "
            "'intensity', 'blurred_gradient_corr', 'blurred_laplacian_corr', "
            "or 'blurred_composite_corr'. Legacy aliases "
            "'ncc_gradient', 'ncc_laplacian', and 'composite' are also accepted."
        )
    return canonical_family


def normalize_optional_displacement_metric_family(
    metric_family: object | None,
    *,
    allow_unknown: bool = False,
) -> Optional[str]:
    """Normalize a possibly-missing family value for artifact loading paths."""

    if metric_family is None:
        return None
    family = str(metric_family).strip()
    if not family:
        return None
    return normalize_displacement_metric_family(
        family,
        allow_unknown=allow_unknown,
    )


@dataclass(frozen=True)
class ResolvedDisplacementMetricSpec:
    """Canonical, validated metric-family configuration for pixelwise matching."""

    family: str
    alpha: float
    raw_component_weights: Dict[str, float]
    normalized_component_weights: Dict[str, float]
    enabled_auxiliary_components: Tuple[str, ...]
    auxiliary_components_active: bool


def resolve_displacement_metric_spec(
    metric_family: object | None,
    *,
    alpha: object | None = DEFAULT_DISPLACEMENT_ALPHA,
    component_weights: Optional[Dict[object, object]] = None,
) -> ResolvedDisplacementMetricSpec:
    """
    Validate and canonicalize pixelwise displacement metric settings.

    ``intensity`` is the strict default and only valid with ``alpha == 1`` and
    no component weights. Gradient/laplacian families are explicit opt-in modes
    that reject user-supplied component weights, while the composite family
    accepts validated gradient/laplacian weights.
    """

    resolved_alpha = DEFAULT_DISPLACEMENT_ALPHA if alpha is None else float(alpha)
    if not np.isfinite(resolved_alpha):
        raise ValueError("alpha must be finite.")
    if resolved_alpha < 0.0 or resolved_alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    family = normalize_displacement_metric_family(metric_family)

    if family == DEFAULT_DISPLACEMENT_METRIC_FAMILY:
        if component_weights is not None:
            raise ValueError("Intensity displacement metric does not accept component_weights.")
        if resolved_alpha != DEFAULT_DISPLACEMENT_ALPHA:
            raise ValueError("Intensity displacement metric requires alpha=1.0.")
        return ResolvedDisplacementMetricSpec(
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
                "blurred_gradient_corr does not accept component_weights; "
                "it is an explicit gradient-only auxiliary metric family."
            )
        weights = {"gradient": 1.0}
        auxiliary_active = resolved_alpha < 1.0
        return ResolvedDisplacementMetricSpec(
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
                "blurred_laplacian_corr does not accept component_weights; "
                "it is an explicit laplacian-only auxiliary metric family."
            )
        weights = {"laplacian": 1.0}
        auxiliary_active = resolved_alpha < 1.0
        return ResolvedDisplacementMetricSpec(
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
                    "Composite displacement component weights only support: "
                    "'gradient', 'laplacian'."
                )
            parsed_value = float(value)
            if not np.isfinite(parsed_value):
                raise ValueError("Composite displacement weights must be finite.")
            if parsed_value < 0.0:
                raise ValueError("Composite displacement weights must be non-negative.")
            parsed_component_weights[name] = parsed_value

    weight_mass = float(sum(parsed_component_weights.values()))
    if resolved_alpha < 1.0 and weight_mass <= 0.0:
        raise ValueError(
            "Blurred composite correlation metric requires positive non-intensity component "
            "weights when displacement_alpha < 1."
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
    return ResolvedDisplacementMetricSpec(
        family=family,
        alpha=resolved_alpha,
        raw_component_weights=parsed_component_weights,
        normalized_component_weights=normalized_component_weights,
        enabled_auxiliary_components=enabled_auxiliary_components,
        auxiliary_components_active=bool(enabled_auxiliary_components),
    )


def _normalize_component_stack(component_stack: np.ndarray) -> np.ndarray:
    """Normalize one ``(H, W, D, D)`` component stack per pixel over displacements."""

    height, width = component_stack.shape[:2]
    flat = component_stack.reshape((height, width, -1))
    min_vals = np.min(flat, axis=2, keepdims=True)
    max_vals = np.max(flat, axis=2, keepdims=True)
    denom = max_vals - min_vals
    safe_denom = np.where(denom <= 1e-12, 1.0, denom)
    normalized = (flat - min_vals) / safe_denom
    tie_mask = denom <= 1e-12
    if np.any(tie_mask):
        normalized = np.where(tie_mask, 0.5, normalized)
    return normalized.reshape(component_stack.shape)


def _summarize_view_score_maps(score_maps: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Return global summary statistics for selected score maps."""

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in score_maps.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def normalize_pixel_shift_power(pixel_shift_power: object) -> float:
    """Validate and normalize the optional powered-moving-view scoring setting."""

    try:
        power = float(pixel_shift_power)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel_shift_power must be a finite positive float.") from exc
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("pixel_shift_power must be a finite positive float.")
    return power


def normalize_pixel_shift_power_apply_to_reference(value: object) -> bool:
    """Normalize the reference-power mode without Python truthiness coercion."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
        if token in mapping:
            return mapping[token]
    raise ValueError(
        "pixel_shift_power_apply_to_reference must be a boolean or one of "
        "true/false, 1/0, yes/no."
    )


def _pixel_value_counts(values: np.ndarray, *, pixel_shift_power: float) -> Dict[str, int]:
    """Describe signed/non-finite inputs and powered-overflow risk for one view."""

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    with np.errstate(over="ignore", invalid="ignore"):
        powered = np.sign(array) * np.power(np.abs(array), pixel_shift_power)
    return {
        "zero": int(np.count_nonzero(finite & (array == 0.0))),
        "negative_finite": int(np.count_nonzero(finite & (array < 0.0))),
        "nan": int(np.count_nonzero(np.isnan(array))),
        "positive_infinity": int(np.count_nonzero(np.isposinf(array))),
        "negative_infinity": int(np.count_nonzero(np.isneginf(array))),
        "post_power_nonfinite": int(np.count_nonzero(~np.isfinite(powered))),
    }


def _signed_power_intensity_values(values: np.ndarray, pixel_shift_power: float) -> np.ndarray:
    """Apply signed power to finite intensity scoring values and keep scores finite."""

    if pixel_shift_power == 1.0:
        return values
    with np.errstate(over="ignore", invalid="ignore"):
        powered = np.sign(values) * np.power(np.abs(values), pixel_shift_power)
    return np.nan_to_num(powered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_pixel_shift_power_inputs(
    views: np.ndarray,
    reference_view: Optional[np.ndarray],
    pixel_shift_power: float,
    *,
    pixel_shift_power_apply_to_reference: bool,
    emit_runtime_warning: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, object]]]:
    """Prepare finite scoring inputs and one structured diagnostic for powered runs."""

    if pixel_shift_power == 1.0:
        return views, reference_view, []

    sanitized_views = np.nan_to_num(
        np.asarray(views, dtype=np.float64, copy=True),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    source_views = np.asarray(views, dtype=np.float64)
    view_counts: Dict[str, Dict[str, int]] = {}
    for i in range(source_views.shape[0]):
        for j in range(source_views.shape[1]):
            view_counts[f"view_{i}_{j}"] = _pixel_value_counts(
                source_views[i, j], pixel_shift_power=pixel_shift_power
            )

    sanitized_reference = reference_view
    if reference_view is not None:
        reference_array = np.asarray(reference_view, dtype=np.float64)
        view_counts["external_reference"] = _pixel_value_counts(
            reference_array,
            pixel_shift_power=(
                pixel_shift_power if pixel_shift_power_apply_to_reference else 1.0
            ),
        )
        sanitized_reference = np.nan_to_num(
            reference_array,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    warning_record: Dict[str, object] = {
        "code": "pixel_shift_power_input_sanitized",
        "pixel_shift_power": pixel_shift_power,
        "views": view_counts,
    }
    needs_warning = any(
        counts["negative_finite"]
        or counts["nan"]
        or counts["positive_infinity"]
        or counts["negative_infinity"]
        for counts in view_counts.values()
    )
    if needs_warning and emit_runtime_warning:
        warnings.warn(
            "pixel_shift_power used signed power after sanitizing negative or non-finite "
            "pixelwise scoring inputs; inspect runtime_warnings for per-view counts.",
            RuntimeWarning,
            stacklevel=3,
        )
    return sanitized_views, sanitized_reference, [warning_record]


def resolve_reference_view(
    views: np.ndarray, ref_view_idx: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Validate ``views`` and resolve the effective reference view indices.

    This helper centralizes shape validation and defaulting to the center view
    so multi-stage rectification pipelines can share reference selection logic
    without re-prompting the user. Existing strategies retain their behavior by
    calling ``select_reference_view`` when desired.
    """

    views_arr, grid_rows, grid_cols, center_i, center_j = RectificationStrategy._default_reference_view(
        views
    )
    if ref_view_idx is None:
        return views_arr, (center_i, center_j)

    ref_i, ref_j = ref_view_idx
    if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
        raise ValueError("ref_view_idx must point to a valid view inside views")

    return views_arr, (ref_i, ref_j)


@dataclass
class PixelwiseSimilarityResult:
    """
    Container for pixelwise patch-matching outputs.

    Attributes mirror ``compute_similarity_matrices``: ``coordinates`` holds the
    best-match destination per pixel, ``scores`` stores the corresponding
    correlation values, and ``disps_y``/``disps_x`` cache the raw displacements
    selected for each view relative to the reference. ``sub_pix_factor`` tracks
    the upsampling factor used to resolve sub-pixel shifts.
    """

    coordinates: np.ndarray
    scores: np.ndarray
    disps_y: np.ndarray
    disps_x: np.ndarray
    sub_pix_factor: int = 2
    reference_is_external: bool = False
    blended_score_maps: Optional[np.ndarray] = None
    component_score_maps: Optional[Dict[str, np.ndarray]] = None
    component_score_maps_raw: Optional[Dict[str, np.ndarray]] = None
    view_score_summary: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    metric_family: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY
    alpha: float = DEFAULT_DISPLACEMENT_ALPHA
    component_weights_raw: Optional[Dict[str, float]] = None
    component_weights_normalized: Optional[Dict[str, float]] = None
    runtime_warnings: List[Dict[str, object]] = field(default_factory=list)
    power_blend_artifacts: Optional[Any] = None

    @property
    def power_blend_diagnostics(self) -> Optional[Dict[str, object]]:
        """Compatibility accessor for callers that consumed blend diagnostics."""

        if self.power_blend_artifacts is None:
            return None
        return self.power_blend_artifacts.diagnostics


class RectificationStrategy(ABC):
    """
    Abstract interface for aligning a grid of extracted views to a reference.

    Implementations accept a square view stack ``(N, N, H, W)`` and return both
    the averaged rectified image and the fully aligned stack. The similarity
    representation (global translations vs. per-pixel flows) is left to
    subclasses.
    """

    @abstractmethod
    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        """Pick a reference view index from a grid of views."""

    @staticmethod
    def _default_reference_view(views: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Validate ``views`` and return its normalized array form with center indices.

        Accept either:
          - (N, N, H, W) : single view grid, or
          - (E, N, N, H, W): multiple exposures of a view grid (will be averaged
            across the exposure axis to produce a single (N, N, H, W) grid).

        This helper centralizes the shape/type checks shared by all rectification
        strategies without altering the existing default behavior (center view
        selection).
        """
        views_arr = np.asarray(views)

        # Accept either a 4D view grid or a 5D stack of exposures (E, N, N, H, W).
        if views_arr.ndim == 5:
            # Collapse exposures by averaging to produce a single (N, N, H, W) grid.
            views_arr = np.mean(views_arr, axis=0)
        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W) or (E, N, N, H, W)")

        grid_rows, grid_cols = views_arr.shape[:2]
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        return views_arr, grid_rows, grid_cols, grid_rows // 2, grid_cols // 2

    @abstractmethod
    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[np.ndarray] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align ``views`` using a strategy-specific similarity description.

        Args:
            views: Grid of extracted grayscale patches with shape
                ``(N, N, H, W)``.
            sim_mat: Pre-computed similarity data for the strategy (shape/type
                defined by the implementation).
            ref_view_idx: Indices of the reference view inside ``views``. When
                omitted, the strategy chooses a default reference.

        Returns:
            A tuple ``(avg_image, rectified_views)`` where ``avg_image`` has
            shape ``(H, W)`` and ``rectified_views`` has shape ``(N, N, H, W)``,
            both aligned to the reference frame defined by ``ref_view_idx``.
        """


class GlobalTranslationRectificationStrategy(RectificationStrategy):
    """
    Apply global translations derived from a similarity matrix to align views.

    This mirrors the original light-field rectification behavior: each view is
    shifted according to its ``(dy, dx)`` offset relative to the chosen
    reference, and all rectified views are averaged to produce the aggregated
    output.
    """

    def __init__(
        self,
        pad_width: int = DEFAULT_CORRELATION_PAD_WIDTH,
        prompt_for_all_views: bool = True,
    ) -> None:
        self.pad_width = int(pad_width)
        self.prompt_for_all_views = prompt_for_all_views

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        _, _, _, center_i, center_j = self._default_reference_view(views)
        return center_i, center_j

    def compute_similarity_matrix(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int],
        thresh: Optional[float] = None,
    ) -> np.ndarray:
        """
        Correlate every view against every other to estimate global translations.

        The reference view indices control which correlations are computed when
        ``prompt_for_all_views`` is ``False``; otherwise all pairwise matches are
        evaluated. The result contains ``(dy, dx)`` offsets in pixels for each
        view pair, padded by ``pad_width`` to avoid edge effects.
        """
        views_arr = np.asarray(views, dtype=np.float64, copy=True)
        if thresh is not None:
            for i in range(views_arr.shape[0]):
                for j in range(views_arr.shape[1]):
                    views_arr[i, j, :, :][views_arr[i, j, :, :] < np.max(views_arr[i, j, :, :]) * thresh] = 0.0

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        num_views = views_arr.shape[0]
        sim_mat = self.pad_width * np.ones((num_views, num_views, num_views, num_views, 2))
        check_all_views = "n"
        if self.prompt_for_all_views:
            check_all_views = input("Check all views for similarity? (y/n): ").lower()

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < num_views and 0 <= ref_j < num_views):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        for i in range(num_views):
            if check_all_views != "y" and i != ref_view_idx[0]:
                continue
            for j in range(num_views):
                if check_all_views != "y" and j != ref_view_idx[1]:
                    continue
                for k in range(num_views):
                    for l in range(num_views):
                        view1 = views_arr[i, j]
                        view2 = views_arr[k, l]

                        padded_view1 = np.pad(view1, self.pad_width)

                        cor = correlate2d(padded_view1, view2, mode="valid")
                        max_cor_loc = np.unravel_index(np.argmax(cor), cor.shape)
                        sim_mat[i, j, k, l, :] = np.subtract(
                            max_cor_loc, (cor.shape[0] // 2, cor.shape[1] // 2)
                        )

        return sim_mat

    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[np.ndarray] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp each view by its global translation and average the aligned stack.

        If ``sim_mat`` is omitted the method computes pairwise translations on
        the fly using ``compute_similarity_matrix``. Warping is done with
        OpenCV's affine transform and the outputs retain the input dimensions.
        """
        views_arr = np.asarray(views, dtype=np.float64)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        ref_i, ref_j = ref_view_idx or self.select_reference_view(views_arr)

        sim_arr = (
            np.asarray(sim_mat)
            if sim_mat is not None
            else self.compute_similarity_matrix(views_arr, (ref_i, ref_j))
        )
        if sim_arr.shape != (grid_rows, grid_cols, grid_rows, grid_cols, 2):
            raise ValueError(
                "sim_mat must have shape (N, N, N, N, 2) matching views grid size"
            )

        if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        rectified_views = np.zeros((grid_rows, grid_cols, height, width), dtype=np.float64)

        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    rectified_views[i, j] = views_arr[i, j]
                    continue

                dy, dx = sim_arr[ref_i, ref_j, i, j]
                warp_mat = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
                rectified_views[i, j] = cv.warpAffine(
                    views_arr[i, j], warp_mat, (width, height), borderMode=cv.BORDER_CONSTANT
                )

        avg_image = np.mean(rectified_views, axis=(0, 1))
        return avg_image, rectified_views


class PixelwisePatchRectificationStrategy(RectificationStrategy):
    """
    Rectify views using per-pixel patch matching against the reference view.

    This strategy correlates a ``k x k`` patch around every pixel in each view
    with rolling windows in the reference view. For every source pixel, the
    correlation peak and its score are stored in a similarity matrix of shape
    ``(H, W, N, N, 2)`` (coordinates) alongside a score matrix of shape
    ``(H, W, N, N)``. A backward warp is then constructed per view to align all
    views to the reference grid.
    """

    def __init__(
        self,
        window_size: int = 41,
        step_size: Optional[int] = None,
        zero_fill_roll: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive odd integer")
        elif window_size % 2 == 0:
            print("Warning: window_size should be odd, incrementing by 1")
            window_size += 1
        self.window_size = int(window_size)
        self.window_radius = self.window_size // 2
        self.step_size = step_size if step_size is not None else self.window_size // 10
        if self.step_size < 1:
            self.step_size = 1
        self.zero_fill_roll = zero_fill_roll

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        _, _, _, center_i, center_j = self._default_reference_view(views)
        return center_i, center_j

    @staticmethod
    def blend_pixelwise_similarity_results(
        anchor: PixelwiseSimilarityResult,
        challenger: PixelwiseSimilarityResult,
        *,
        blend_alpha: float,
        delta_cap: float,
    ) -> Tuple[PixelwiseSimilarityResult, Dict[str, object]]:
        """Blend an anchor field toward a challenger using capped vector residuals."""

        alpha_value = float(blend_alpha)
        cap_value = float(delta_cap)
        if not np.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
            raise ValueError("blend_alpha must be finite and between 0 and 1.")
        if not np.isfinite(cap_value) or cap_value < 0.0:
            raise ValueError("delta_cap must be finite and non-negative.")

        anchor_y = np.asarray(anchor.disps_y, dtype=np.float32)
        anchor_x = np.asarray(anchor.disps_x, dtype=np.float32)
        challenger_y = np.asarray(challenger.disps_y, dtype=np.float32)
        challenger_x = np.asarray(challenger.disps_x, dtype=np.float32)
        expected_coordinate_shape = (*anchor_y.shape, 2)
        if (
            anchor_y.ndim != 4
            or anchor_x.shape != anchor_y.shape
            or challenger_y.shape != anchor_y.shape
            or challenger_x.shape != anchor_y.shape
            or np.asarray(anchor.scores).shape != anchor_y.shape
            or np.asarray(challenger.scores).shape != anchor_y.shape
            or np.asarray(anchor.coordinates).shape != expected_coordinate_shape
            or np.asarray(challenger.coordinates).shape != expected_coordinate_shape
        ):
            raise ValueError("anchor and challenger pixelwise result shapes must match.")
        if int(anchor.sub_pix_factor) != int(challenger.sub_pix_factor):
            raise ValueError("anchor and challenger sub_pix_factor values must match.")
        if bool(anchor.reference_is_external) != bool(challenger.reference_is_external):
            raise ValueError("anchor and challenger reference_is_external values must match.")

        raw_delta_y = challenger_y - anchor_y
        raw_delta_x = challenger_x - anchor_x
        finite_mask = np.isfinite(raw_delta_y) & np.isfinite(raw_delta_x)
        safe_delta_y = np.where(finite_mask, raw_delta_y, 0.0).astype(np.float32)
        safe_delta_x = np.where(finite_mask, raw_delta_x, 0.0).astype(np.float32)
        preclip_magnitude = np.hypot(safe_delta_y, safe_delta_x).astype(np.float64)
        if cap_value == 0.0:
            cap_scale = np.zeros_like(preclip_magnitude)
        else:
            cap_scale = np.minimum(1.0, cap_value / (preclip_magnitude + 1e-12))
        clipped_delta_y = (safe_delta_y * cap_scale).astype(np.float32)
        clipped_delta_x = (safe_delta_x * cap_scale).astype(np.float32)
        final_delta_y = (alpha_value * clipped_delta_y).astype(np.float32)
        final_delta_x = (alpha_value * clipped_delta_x).astype(np.float32)
        final_disps_y = (anchor_y + final_delta_y).astype(np.float32)
        final_disps_x = (anchor_x + final_delta_x).astype(np.float32)

        height, width = anchor_y.shape[:2]
        base_y, base_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        coordinates = np.zeros(expected_coordinate_shape, dtype=np.float32)
        coordinates[..., 0] = base_y[..., None, None] - final_disps_y
        coordinates[..., 1] = base_x[..., None, None] - final_disps_x
        coordinate_oob = (
            (coordinates[..., 0] < 0.0)
            | (coordinates[..., 0] > float(height - 1))
            | (coordinates[..., 1] < 0.0)
            | (coordinates[..., 1] > float(width - 1))
        )
        final_magnitude = np.hypot(final_delta_y, final_delta_x).astype(np.float64)
        gradient_terms: List[np.ndarray] = []
        if height > 1:
            gradient_terms.extend((
                np.gradient(final_delta_y, axis=0),
                np.gradient(final_delta_x, axis=0),
            ))
        if width > 1:
            gradient_terms.extend((
                np.gradient(final_delta_y, axis=1),
                np.gradient(final_delta_x, axis=1),
            ))
        if gradient_terms:
            gradient_magnitude = np.sqrt(sum(np.square(term) for term in gradient_terms))
        else:
            gradient_magnitude = np.zeros_like(final_delta_y, dtype=np.float32)
        diagnostics: Dict[str, object] = {
            "blend_alpha": alpha_value,
            "delta_cap": cap_value,
            "preclip_delta_mean": float(np.mean(preclip_magnitude)),
            "preclip_delta_rms": float(np.sqrt(np.mean(np.square(preclip_magnitude)))),
            "preclip_delta_max": float(np.max(preclip_magnitude)),
            "cap_hit_fraction": float(np.mean(preclip_magnitude > cap_value)),
            "nonfinite_delta_fraction": float(np.mean(~finite_mask)),
            "final_delta_mean": float(np.mean(final_magnitude)),
            "final_delta_rms": float(np.sqrt(np.mean(np.square(final_magnitude)))),
            "final_delta_max": float(np.max(final_magnitude)),
            "delta_gradient_mean": float(np.mean(gradient_magnitude)),
            "delta_gradient_p95": float(np.percentile(gradient_magnitude, 95.0)),
            "delta_gradient_max": float(np.max(gradient_magnitude)),
            "coordinate_oob_fraction": float(np.mean(coordinate_oob)),
            "scores_source": "anchor_inherited",
        }
        from .power_blend import PowerBlendArtifacts, PowerBlendFieldPair

        artifacts = PowerBlendArtifacts(
            fields=PowerBlendFieldPair(anchor=anchor, challenger=challenger),
            diagnostics=diagnostics,
            delta_y=safe_delta_y.copy(),
            delta_x=safe_delta_x.copy(),
            delta_magnitude=preclip_magnitude.astype(np.float32),
        )
        return PixelwiseSimilarityResult(
            coordinates=coordinates,
            scores=np.asarray(anchor.scores).copy(),
            disps_y=final_disps_y,
            disps_x=final_disps_x,
            sub_pix_factor=int(anchor.sub_pix_factor),
            reference_is_external=bool(anchor.reference_is_external),
            blended_score_maps=(
                None if anchor.blended_score_maps is None else np.asarray(anchor.blended_score_maps).copy()
            ),
            component_score_maps=(
                None if anchor.component_score_maps is None
                else {key: np.asarray(value).copy() for key, value in anchor.component_score_maps.items()}
            ),
            component_score_maps_raw=(
                None if anchor.component_score_maps_raw is None
                else {key: np.asarray(value).copy() for key, value in anchor.component_score_maps_raw.items()}
            ),
            view_score_summary=(
                None if anchor.view_score_summary is None else dict(anchor.view_score_summary)
            ),
            metric_family=anchor.metric_family,
            alpha=anchor.alpha,
            component_weights_raw=(
                None if anchor.component_weights_raw is None else dict(anchor.component_weights_raw)
            ),
            component_weights_normalized=(
                None if anchor.component_weights_normalized is None
                else dict(anchor.component_weights_normalized)
            ),
            runtime_warnings=list(anchor.runtime_warnings) + list(challenger.runtime_warnings),
            power_blend_artifacts=artifacts,
        ), diagnostics

    @staticmethod
    def _blend_normalized_components(
        component_stacks_norm: Dict[str, np.ndarray],
        alpha: float,
        normalized_component_weights: Dict[str, float],
    ) -> np.ndarray:
        blended_stack = float(alpha) * component_stacks_norm["intensity"]
        if normalized_component_weights:
            secondary_blend = np.zeros_like(blended_stack)
            for component_name, component_weight in normalized_component_weights.items():
                secondary_blend = secondary_blend + float(component_weight) * component_stacks_norm[component_name]
            blended_stack = blended_stack + (1.0 - float(alpha)) * secondary_blend
        return blended_stack

    def compute_similarity_matrices(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
        disp_gauss_sigma: Optional[float] = None, disp_gauss_size: Optional[int] = None,
        thresh: Optional[float] = None, normalized_conv: Optional[bool] = True,
        sub_pix_factor: int = 2,
        reference_view: Optional[np.ndarray] = None,
        use_sigma_pyramid: bool = False,
        metric_family: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY,
        alpha: float = DEFAULT_DISPLACEMENT_ALPHA,
        component_weights: Optional[Dict[str, float]] = None,
        pixel_shift_power: float = 1.0,
        pixel_shift_power_apply_to_reference: bool = True,
        emit_runtime_warning: bool = True,
    ) -> PixelwiseSimilarityResult:
        """
        Perform per-pixel patch matching between each view and the reference.

        For every non-reference view, the function rolls the image across a
        displacement grid and computes Gaussian-smoothed product scores against
        the (optionally thresholded) reference. The best displacement per pixel
        is recorded along with raw score maps and derived flow fields. These
        families are blurred correlation-like ranking metrics; they do not
        implement true local NCC/ZNCC normalization.

        Args:
            views: Square grid of grayscale patches with shape ``(N, N, H, W)``.
            ref_view_idx: ``(row, col)`` index of the reference view.
            max_disp: Maximum pixel displacement searched in each direction.
            gauss_sigma: Sigma for blurring the per-pixel correlation product.
            gauss_size: Kernel size used for the Gaussian blur.
            disp_gauss_sigma: Optional smoothing applied to the displacement
                fields.
            disp_gauss_size: Kernel size for displacement smoothing.
            thresh: Relative intensity threshold (0–1) below which reference
                pixels are ignored.
            use_sigma_pyramid: When ``True``, iteratively recompute similarities
                with smaller Gaussian sigmas while constraining updates to stay
                within two pixels of the initial displacement estimates.
            sub_pix_factor: Upsampling factor for sub-pixel displacement
                searches. ``1`` keeps full-pixel shifts, ``2`` enables half-pixel
                shifts, and so on.
            alpha: Blend weight for local displacement-score ranking. ``alpha``
                controls the relative importance of the intensity score versus
                the selected auxiliary blurred-correlation component(s).
            pixel_shift_power: Positive signed exponent applied to shifted
                intensity candidates for the intensity-correlation score.
            pixel_shift_power_apply_to_reference: When ``True``, apply the
                same signed exponent to the intensity reference. ``False``
                preserves legacy shifted-candidate-only scoring.
            emit_runtime_warning: Emit a Python warning for signed or
                non-finite powered-run inputs.

        Returns:
            ``PixelwiseSimilarityResult`` containing local displacement scores,
            displacement maps, and target coordinates for each view.
        """
        resolved_pixel_shift_power = normalize_pixel_shift_power(pixel_shift_power)
        resolved_reference_power_mode = normalize_pixel_shift_power_apply_to_reference(
            pixel_shift_power_apply_to_reference
        )
        scoring_views, scoring_reference, runtime_warnings = _sanitize_pixel_shift_power_inputs(
            views,
            reference_view,
            resolved_pixel_shift_power,
            pixel_shift_power_apply_to_reference=resolved_reference_power_mode,
            emit_runtime_warning=emit_runtime_warning,
        )

        if not use_sigma_pyramid:
            result = self._compute_similarity_matrices_single(
                views,
                ref_view_idx,
                max_disp=max_disp,
                gauss_sigma=gauss_sigma,
                gauss_size=gauss_size,
                disp_gauss_sigma=disp_gauss_sigma,
                disp_gauss_size=disp_gauss_size,
                thresh=thresh,
                normalized_conv=normalized_conv,
                sub_pix_factor=sub_pix_factor,
                reference_view=reference_view,
                metric_family=metric_family,
                alpha=alpha,
                component_weights=component_weights,
                pixel_shift_power=resolved_pixel_shift_power,
                pixel_shift_power_apply_to_reference=resolved_reference_power_mode,
                intensity_views=scoring_views,
                intensity_reference_view=scoring_reference,
            )
            result.runtime_warnings = runtime_warnings
            return result

        base_sigma = 7 if gauss_sigma is None else gauss_sigma
        base_size = int(base_sigma * 5) if gauss_size is None else gauss_size
        current_result = self._compute_similarity_matrices_single(
            views,
            ref_view_idx,
            max_disp=max_disp,
            gauss_sigma=base_sigma,
            gauss_size=base_size,
            disp_gauss_sigma=disp_gauss_sigma,
            disp_gauss_size=disp_gauss_size,
            thresh=thresh,
            normalized_conv=normalized_conv,
            sub_pix_factor=sub_pix_factor,
            reference_view=reference_view,
            metric_family=metric_family,
            alpha=alpha,
            component_weights=component_weights,
            pixel_shift_power=resolved_pixel_shift_power,
            pixel_shift_power_apply_to_reference=resolved_reference_power_mode,
            intensity_views=scoring_views,
            intensity_reference_view=scoring_reference,
        )

        current_sigma = int(base_sigma)
        while current_sigma > 1:
            next_sigma = int(current_sigma / 2)
            if next_sigma < 1:
                break

            next_size = int(next_sigma * 5) if gauss_size is None else gauss_size
            current_result = self._compute_similarity_matrices_single(
                views,
                ref_view_idx,
                max_disp=max_disp,
                gauss_sigma=next_sigma,
                gauss_size=next_size,
                disp_gauss_sigma=disp_gauss_sigma,
                disp_gauss_size=disp_gauss_size,
                thresh=thresh,
                normalized_conv=normalized_conv,
                sub_pix_factor=sub_pix_factor,
                reference_view=reference_view,
                prev_result=current_result,
                metric_family=metric_family,
                alpha=alpha,
                component_weights=component_weights,
                pixel_shift_power=resolved_pixel_shift_power,
                pixel_shift_power_apply_to_reference=resolved_reference_power_mode,
                intensity_views=scoring_views,
                intensity_reference_view=scoring_reference,
            )

            current_sigma = next_sigma

        current_result.runtime_warnings = runtime_warnings
        return current_result

    def _compute_similarity_matrices_single(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
        disp_gauss_sigma: Optional[float] = None, disp_gauss_size: Optional[int] = None,
        thresh: Optional[float] = None, normalized_conv: Optional[bool] = True,
        sub_pix_factor: int = 2,
        reference_view: Optional[np.ndarray] = None,
        prev_result: Optional[PixelwiseSimilarityResult] = None,
        metric_family: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY,
        alpha: float = DEFAULT_DISPLACEMENT_ALPHA,
        component_weights: Optional[Dict[str, float]] = None,
        pixel_shift_power: float = 1.0,
        pixel_shift_power_apply_to_reference: bool = True,
        intensity_views: Optional[np.ndarray] = None,
        intensity_reference_view: Optional[np.ndarray] = None,
    ) -> PixelwiseSimilarityResult:
        """Compute pixelwise similarity matrices for a single Gaussian sigma."""
        if sub_pix_factor < 1:
            raise ValueError("sub_pix_factor must be >= 1")
        sub_pix_factor = int(sub_pix_factor)

        views_arr = np.asarray(views, dtype=np.float64, copy=True)
        intensity_views_arr = np.asarray(
            views if intensity_views is None else intensity_views,
            dtype=np.float64,
            copy=True,
        )
        if intensity_views_arr.shape != views_arr.shape:
            raise ValueError("intensity_views must have the same shape as views")
        if thresh is not None:
            for i in range(views_arr.shape[0]):
                for j in range(views_arr.shape[1]):
                    views_arr[i, j, :, :][views_arr[i, j, :, :] < np.max(views_arr[i, j, :, :]) * thresh] = 0.0
                    intensity_views_arr[i, j, :, :][
                        intensity_views_arr[i, j, :, :]
                        < np.max(intensity_views_arr[i, j, :, :]) * thresh
                    ] = 0.0
        
        if max_disp is None:
            max_disp = self.window_radius
        if gauss_sigma is None:
            gauss_sigma = 7
        if gauss_size is None:
            gauss_size = int(gauss_sigma * 5)
        if disp_gauss_sigma is None:
            disp_gauss_sigma = 0.0
        metric_spec = resolve_displacement_metric_spec(
            metric_family,
            alpha=alpha,
            component_weights=component_weights,
        )
        resolved_metric_family = metric_spec.family
        raw_component_weights = metric_spec.raw_component_weights
        normalized_component_weights = metric_spec.normalized_component_weights
        metric_components_enabled = metric_spec.auxiliary_components_active

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")
        original_height, original_width = height, width

        if sub_pix_factor > 1:
            resized_views = np.zeros(
                (grid_rows, grid_cols, height * sub_pix_factor, width * sub_pix_factor),
                dtype=np.float64,
            )
            for i in range(grid_rows):
                for j in range(grid_cols):
                    resized_views[i, j] = cv.resize(
                        views_arr[i, j],
                        (width * sub_pix_factor, height * sub_pix_factor),
                        interpolation=cv.INTER_LINEAR,
                    )
            views_arr = resized_views
            resized_intensity_views = np.zeros(
                (grid_rows, grid_cols, height * sub_pix_factor, width * sub_pix_factor),
                dtype=np.float64,
            )
            for i in range(grid_rows):
                for j in range(grid_cols):
                    resized_intensity_views[i, j] = cv.resize(
                        intensity_views_arr[i, j],
                        (width * sub_pix_factor, height * sub_pix_factor),
                        interpolation=cv.INTER_LINEAR,
                    )
            intensity_views_arr = resized_intensity_views
            height *= sub_pix_factor
            width *= sub_pix_factor
            gauss_sigma *= sub_pix_factor
            gauss_size = int(gauss_sigma * 5)

        effective_window_size = self.window_size * sub_pix_factor
        effective_window_radius = effective_window_size // 2
        effective_step_size = 1 if sub_pix_factor > 1 else self.step_size
        max_disp_scaled = max_disp * sub_pix_factor
        if disp_gauss_size is None:
            disp_gauss_size = effective_window_radius * 2 + 1

        if effective_window_size > height or effective_window_size > width:
            raise ValueError("window_size cannot exceed view dimensions")

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        reference_is_external = reference_view is not None
        if reference_is_external:
            ref_view = np.asarray(reference_view, dtype=np.float32)
            intensity_ref_view = np.asarray(
                reference_view if intensity_reference_view is None else intensity_reference_view,
                dtype=np.float32,
            )
            if ref_view.ndim != 2:
                raise ValueError("reference_view must be a 2D array matching a single view")
            if intensity_ref_view.ndim != 2:
                raise ValueError("intensity_reference_view must be a 2D array matching a single view")
            if ref_view.shape == (original_height, original_width) and sub_pix_factor > 1:
                ref_view = cv.resize(
                    ref_view,
                    (width, height),
                    interpolation=cv.INTER_LINEAR,
                )
            if intensity_ref_view.shape == (original_height, original_width) and sub_pix_factor > 1:
                intensity_ref_view = cv.resize(
                    intensity_ref_view,
                    (width, height),
                    interpolation=cv.INTER_LINEAR,
                )
            if ref_view.shape != (height, width):
                raise ValueError("reference_view must match the view dimensions")
            if intensity_ref_view.shape != (height, width):
                raise ValueError("intensity_reference_view must match the view dimensions")
        else:
            # Setting up and thresholding the reference view according to percentages from maximal value
            ref_view = views_arr[ref_i, ref_j].astype(np.float32, copy=True)
            intensity_ref_view = intensity_views_arr[ref_i, ref_j].astype(np.float32, copy=True)
        if thresh is not None:
            ref_view[ref_view < np.max(ref_view)*thresh] = 0.0
            intensity_ref_view[intensity_ref_view < np.max(intensity_ref_view)*thresh] = 0.0
        if pixel_shift_power_apply_to_reference:
            intensity_ref_view = _signed_power_intensity_values(
                intensity_ref_view,
                pixel_shift_power,
            )

        D = max_disp_scaled * 2 + 1
        disp_arr = np.arange(-max_disp_scaled, max_disp_scaled + 1, dtype=np.float32)

        coordinates = np.zeros((height, width, grid_rows, grid_cols, 2), dtype=np.float32)
        scores = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
        disps_y = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float32)
        disps_x = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float32)
        blended_score_maps = (
            np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
            if metric_components_enabled
            else None
        )
        component_keys: List[str] = ["intensity"]
        if metric_components_enabled:
            component_keys.extend(list(metric_spec.enabled_auxiliary_components))
        component_score_maps = (
            {
                key: np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
                for key in component_keys
            }
            if metric_components_enabled
            else None
        )
        component_score_maps_raw = (
            {
                key: np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
                for key in component_keys
            }
            if metric_components_enabled
            else None
        )
        base_coords = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), -1)
        ref_grad_y = ref_grad_x = ref_laplacian = None
        if "gradient" in metric_spec.enabled_auxiliary_components:
            ref_grad_y, ref_grad_x = np.gradient(ref_view)
        if "laplacian" in metric_spec.enabled_auxiliary_components:
            ref_laplacian = cv.Laplacian(ref_view, cv.CV_32F)
        if normalized_conv:
            intensity_ref_view_blurred = gaussian_filter(
                                                intensity_ref_view,
                                                sigma=gauss_sigma,
                                                truncate=(gauss_size/gauss_sigma),
                                                mode="constant")
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not reference_is_external and i == ref_i and j == ref_j:
                    continue

                p_dy, p_dx = None, None
                limit = None
                if prev_result is not None:
                    p_dy = prev_result.disps_y[:, :, i, j]
                    p_dx = prev_result.disps_x[:, :, i, j]

                    if sub_pix_factor > 1:
                        p_dy = cv.resize(p_dy, (width, height), interpolation=cv.INTER_NEAREST)
                        p_dx = cv.resize(p_dx, (width, height), interpolation=cv.INTER_NEAREST)
                        # Scale displacements to match scaled resolution
                        p_dy = p_dy * sub_pix_factor
                        p_dx = p_dx * sub_pix_factor

                    limit = 2.0 * sub_pix_factor

                component_stacks_raw: Dict[str, np.ndarray] = {
                    key: np.zeros((height, width, D, D), dtype=np.float32)
                    for key in component_keys
                }
                norm_mat_view = None
                if normalized_conv:
                    norm_mat_view = np.zeros((height, width, D, D), dtype=np.float32)

                view = views_arr[i, j].astype(np.float32, copy=False)
                intensity_view = intensity_views_arr[i, j].astype(np.float32, copy=False)
                for s_y_ind in range(0, D, effective_step_size):
                    for s_x_ind in range(0, D, effective_step_size):
                        s_y = s_y_ind - max_disp_scaled
                        s_x = s_x_ind - max_disp_scaled
                        shifted_view = self._roll_with_fill(
                            view, shift_y=s_y, shift_x=s_x, zero_fill=self.zero_fill_roll
                        )
                        intensity_shifted_source = self._roll_with_fill(
                            intensity_view, shift_y=s_y, shift_x=s_x, zero_fill=self.zero_fill_roll
                        )
                        intensity_shifted_view = _signed_power_intensity_values(
                            intensity_shifted_source,
                            pixel_shift_power,
                        )
                        intensity_corr = gaussian_filter(
                            intensity_ref_view * intensity_shifted_view,
                            sigma=gauss_sigma,
                            truncate=(gauss_size/ gauss_sigma),
                            mode="constant",
                        )
                        component_stacks_raw["intensity"][:, :, s_y_ind, s_x_ind] = intensity_corr

                        if "gradient" in metric_spec.enabled_auxiliary_components and ref_grad_x is not None and ref_grad_y is not None:
                            shift_grad_y, shift_grad_x = np.gradient(shifted_view)
                            gradient_corr = gaussian_filter(
                                ref_grad_x * shift_grad_x + ref_grad_y * shift_grad_y,
                                sigma=gauss_sigma,
                                truncate=(gauss_size / gauss_sigma),
                                mode="constant",
                            )
                            component_stacks_raw["gradient"][:, :, s_y_ind, s_x_ind] = gradient_corr

                        if "laplacian" in metric_spec.enabled_auxiliary_components and ref_laplacian is not None:
                            shifted_laplacian = cv.Laplacian(shifted_view, cv.CV_32F)
                            laplacian_corr = gaussian_filter(
                                ref_laplacian * shifted_laplacian,
                                sigma=gauss_sigma,
                                truncate=(gauss_size / gauss_sigma),
                                mode="constant",
                            )
                            component_stacks_raw["laplacian"][:, :, s_y_ind, s_x_ind] = laplacian_corr

                        if p_dy is not None:
                            current_dy = disp_arr[s_y_ind]
                            current_dx = disp_arr[s_x_ind]
                            mask = (np.abs(p_dy - current_dy) <= limit) & (np.abs(p_dx - current_dx) <= limit)
                            for component_name in component_stacks_raw.keys():
                                component_slice = component_stacks_raw[component_name][:, :, s_y_ind, s_x_ind]
                                component_slice[~mask] = 0.0
                                component_stacks_raw[component_name][:, :, s_y_ind, s_x_ind] = component_slice

                        if normalized_conv:
                            norm_shifted_view = intensity_shifted_view
                            norm_mat_view[:, :, s_y_ind, s_x_ind] = gaussian_filter(
                                norm_shifted_view**2,
                                  sigma=gauss_sigma,
                                    truncate=(gauss_size/gauss_sigma),
                                      mode="constant")
                if normalized_conv:
                    norm_mat_view = np.sqrt(
                        norm_mat_view * intensity_ref_view_blurred[:, :, None, None]
                    )
                    norm_mat_view = norm_mat_view + 0.1 * np.max(norm_mat_view)
                    component_stacks_raw["intensity"] = component_stacks_raw["intensity"] / norm_mat_view

                component_stacks_norm: Dict[str, np.ndarray] = {
                    key: _normalize_component_stack(stack)
                    for key, stack in component_stacks_raw.items()
                }
                if metric_components_enabled:
                    blended_stack = self._blend_normalized_components(
                        component_stacks_norm=component_stacks_norm,
                        alpha=metric_spec.alpha,
                        normalized_component_weights=normalized_component_weights,
                    )
                else:
                    blended_stack = component_stacks_norm["intensity"]

                max_val = np.max(blended_stack, axis=(2, 3))
                scores[:, :, i, j] = max_val
                if blended_score_maps is not None:
                    blended_score_maps[:, :, i, j] = max_val

                flat = blended_stack.reshape((height, width, D * D))
                argmax_flat = np.argmax(flat, axis=2)
                idx_y, idx_x = np.unravel_index(argmax_flat, (D, D))
                if component_score_maps is not None and component_score_maps_raw is not None:
                    gather_index = argmax_flat[:, :, None]
                    for component_name in component_keys:
                        flat_norm = component_stacks_norm[component_name].reshape((height, width, D * D))
                        flat_raw = component_stacks_raw[component_name].reshape((height, width, D * D))
                        component_score_maps[component_name][:, :, i, j] = np.take_along_axis(
                            flat_norm, gather_index, axis=2
                        ).squeeze(axis=2)
                        component_score_maps_raw[component_name][:, :, i, j] = np.take_along_axis(
                            flat_raw, gather_index, axis=2
                        ).squeeze(axis=2)

                # Preserve the original "no signal => zero displacement" behavior.
                raw_intensity_max = np.max(np.abs(component_stacks_raw["intensity"]), axis=(2, 3))
                zero_mask = raw_intensity_max <= 1e-12
                if np.any(zero_mask):
                    idx_y = idx_y.copy()
                    idx_x = idx_x.copy()
                    idx_y[zero_mask] = max_disp_scaled
                    idx_x[zero_mask] = max_disp_scaled

                # compute coordinates: base + displacement, apply gaussian filter if sigma is legal
                if disp_gauss_sigma is None or disp_gauss_sigma <= 0.0:
                    disps_y[:, :, i, j] = disp_arr[idx_y]
                    disps_x[:, :, i, j] = disp_arr[idx_x]
                else:
                    disps_y[:, :, i, j] = gaussian_filter(disp_arr[idx_y], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_y]
                    disps_x[:, :, i, j] = gaussian_filter(disp_arr[idx_x], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_x]
                coordinates[:, :, i, j, 0] = base_coords[:, :, 0] - disps_y[:, :, i, j]
                coordinates[:, :, i, j, 1] = base_coords[:, :, 1] - disps_x[:, :, i, j]

        if sub_pix_factor > 1:
            resized_scores = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float64)
            resized_disps_y = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float32)
            resized_disps_x = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float32)
            resized_blended = (
                np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float64)
                if blended_score_maps is not None
                else None
            )
            resized_component_score_maps = (
                {
                    key: np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float64)
                    for key in component_keys
                }
                if component_score_maps is not None
                else None
            )
            resized_component_score_maps_raw = (
                {
                    key: np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float64)
                    for key in component_keys
                }
                if component_score_maps_raw is not None
                else None
            )
            for i in range(grid_rows):
                for j in range(grid_cols):
                    resized_scores[:, :, i, j] = cv.resize(
                        scores[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
                    if resized_blended is not None and blended_score_maps is not None:
                        resized_blended[:, :, i, j] = cv.resize(
                            blended_score_maps[:, :, i, j],
                            (original_width, original_height),
                            interpolation=cv.INTER_LINEAR,
                        )
                    resized_disps_y[:, :, i, j] = cv.resize(
                        disps_y[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
                    resized_disps_x[:, :, i, j] = cv.resize(
                        disps_x[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
                    if resized_component_score_maps is not None and resized_component_score_maps_raw is not None and component_score_maps is not None and component_score_maps_raw is not None:
                        for component_name in component_keys:
                            resized_component_score_maps[component_name][:, :, i, j] = cv.resize(
                                component_score_maps[component_name][:, :, i, j],
                                (original_width, original_height),
                                interpolation=cv.INTER_LINEAR,
                            )
                            resized_component_score_maps_raw[component_name][:, :, i, j] = cv.resize(
                                component_score_maps_raw[component_name][:, :, i, j],
                                (original_width, original_height),
                                interpolation=cv.INTER_LINEAR,
                            )
            scores = resized_scores
            blended_score_maps = resized_blended
            disps_y = resized_disps_y / sub_pix_factor
            disps_x = resized_disps_x / sub_pix_factor
            component_score_maps = resized_component_score_maps
            component_score_maps_raw = resized_component_score_maps_raw
            base_coords = np.stack(
                np.meshgrid(np.arange(original_height), np.arange(original_width), indexing="ij"),
                -1,
            ).astype(np.float32)
            coordinates = np.zeros((original_height, original_width, grid_rows, grid_cols, 2), dtype=np.float32)
            coordinates[:, :, :, :, 0] = base_coords[:, :, 0][..., None, None] - disps_y
            coordinates[:, :, :, :, 1] = base_coords[:, :, 1][..., None, None] - disps_x

        view_score_summary: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
        if blended_score_maps is not None and component_score_maps is not None:
            view_score_summary = {}
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if not reference_is_external and i == ref_i and j == ref_j:
                        continue
                    summary_inputs = {
                        "blended": blended_score_maps[:, :, i, j],
                    }
                    for component_name in component_keys:
                        summary_inputs[component_name] = component_score_maps[component_name][:, :, i, j]
                    view_score_summary[f"view_{i}_{j}"] = _summarize_view_score_maps(summary_inputs)

        return PixelwiseSimilarityResult(
            coordinates=coordinates,
            scores=scores,
            disps_y=disps_y,
            disps_x=disps_x,
            sub_pix_factor=sub_pix_factor,
            reference_is_external=reference_is_external,
            blended_score_maps=blended_score_maps,
            component_score_maps=component_score_maps,
            component_score_maps_raw=component_score_maps_raw,
            view_score_summary=view_score_summary,
            metric_family=resolved_metric_family,
            alpha=metric_spec.alpha,
            component_weights_raw=raw_component_weights or None,
            component_weights_normalized=normalized_component_weights or None,
        )

    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[PixelwiseSimilarityResult] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward-warp each view using per-pixel displacement fields.

        The provided ``PixelwiseSimilarityResult`` supplies absolute coordinates
        for ``cv.remap``; when omitted, similarities are recomputed first. The
        reference view is copied directly into the rectified stack before the
        averaged image is calculated.
        """
        views_arr = np.asarray(views, dtype=np.float64)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        ref_i, ref_j = ref_view_idx or self.select_reference_view(views_arr)

        sim_result = (
            sim_mat
            if sim_mat is not None
            else self.compute_similarity_matrices(views_arr, (ref_i, ref_j))
        )
        if not isinstance(sim_result, PixelwiseSimilarityResult):
            raise TypeError("sim_mat must be a PixelwiseSimilarityResult for pixelwise rectification")

        coordinates = np.asarray(sim_result.coordinates)
        scores = np.asarray(sim_result.scores)
        disps_y = np.asarray(sim_result.disps_y)
        disps_x = np.asarray(sim_result.disps_x)

        if coordinates.shape != (height, width, grid_rows, grid_cols, 2):
            raise ValueError(
                "coordinates must have shape (H, W, N, N, 2) matching views dimensions"
            )
        if scores.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("scores must have shape (H, W, N, N) matching views dimensions")
        if disps_y.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("disps_y must have shape (H, W, N, N) matching views dimensions")
        if disps_x.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("disps_x must have shape (H, W, N, N) matching views dimensions")

        rectified_views = np.zeros((grid_rows, grid_cols, height, width), dtype=np.float64)

        base_mapping = self._identity_mapping(height, width)
        base_y = base_mapping[:, :, 0]
        base_x = base_mapping[:, :, 1]
        reference_is_external = getattr(sim_result, "reference_is_external", False)
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not reference_is_external and i == ref_i and j == ref_j:
                    rectified_views[i, j] = views_arr[i, j]
                    continue

                # map_y = (base_y - disps_y[:, :, i, j]).astype(np.float32)
                # map_x = (base_x - disps_x[:, :, i, j]).astype(np.float32)
                map_y = coordinates[:, :, i, j, 0].astype(np.float32)
                map_x = coordinates[:, :, i, j, 1].astype(np.float32)
                rectified_views[i, j] = cv.remap(
                    views_arr[i, j],
                    map_x,
                    map_y,
                    interpolation=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_CONSTANT,
                )

        avg_image = np.mean(rectified_views, axis=(0, 1))
        return avg_image, rectified_views

    def _build_backward_map(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        base_mapping: np.ndarray,
    ) -> np.ndarray:
        """
        Construct a reverse lookup map from destination coordinates and scores.

        For each candidate coordinate, the mapping keeps only the source pixel
        that achieved the highest score at that destination, falling back to the
        identity map when no valid mapping exists.
        """
        mapping = base_mapping.copy()
        height, width, _ = coordinates.shape
        best_scores = np.full((height, width), -np.inf, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                target_y, target_x = coordinates[y, x]
                if not (0 <= target_y < height and 0 <= target_x < width):
                    continue

                if scores[y, x] > best_scores[target_y, target_x]:
                    best_scores[target_y, target_x] = scores[y, x]
                    mapping[target_y, target_x] = (y, x)

        return mapping

    @staticmethod
    def _identity_mapping(height: int, width: int) -> np.ndarray:
        """Return an ``(H, W, 2)`` array of pixel indices for identity remapping."""
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        return np.stack([yy, xx], axis=-1).astype(np.float32)

    @staticmethod
    def _roll_with_fill(
        array: np.ndarray, shift_y: int, shift_x: int, zero_fill: bool
    ) -> np.ndarray:
        """
        Shift ``array`` by ``(shift_y, shift_x)`` using ``np.roll`` with optional zero fill.

        When ``zero_fill`` is ``True``, regions that would wrap around the image edges
        are explicitly set to zero to avoid reflected artifacts from the roll
        operation.
        """

        shifted = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))
        if not zero_fill:
            return shifted

        rows_to_zero = min(abs(shift_y), shifted.shape[0])
        cols_to_zero = min(abs(shift_x), shifted.shape[1])

        if shift_y > 0:
            shifted[:rows_to_zero, :] = 0
        elif shift_y < 0:
            shifted[-rows_to_zero:, :] = 0

        if shift_x > 0:
            shifted[:, :cols_to_zero] = 0
        elif shift_x < 0:
            shifted[:, -cols_to_zero:] = 0

        return shifted

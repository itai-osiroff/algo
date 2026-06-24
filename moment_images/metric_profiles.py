"""Shared metric registry, profile config, and extended image metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, cast

import cv2 as cv
import numpy as np

from moment_images.im_process import min_max_normalize
from moment_images.reliable_io import read_json_first_valid, write_json_atomic


DEFAULT_GUIDE_SIGMA = 1.0
DEFAULT_SEARCH_WEIGHT_STEP = 0.1
SAVED_ONLY_METRICS = {"saved_score", "saved_mean_dataset_score"}


@dataclass(frozen=True)
class MetricDescriptor:
    """Description of one scalar metric available to ranking profiles."""

    name: str
    label: str
    direction: str
    category: str
    representation: str
    description: str
    read_only: bool = False


@dataclass
class RankingProfileSpec:
    """Named profile composed of weighted metrics."""

    name: str
    label: str
    metric_weights: Dict[str, float]
    description: str = ""
    built_in: bool = False
    allow_negative_weights: bool = False


@dataclass
class SearchDefaults:
    """Configuration for metric-profile search actions."""

    metric_subset: List[str] = field(default_factory=list)
    weight_step: float = DEFAULT_SEARCH_WEIGHT_STEP
    max_metrics: int = 6
    allow_negative_weights: bool = False


@dataclass
class MetricExplorerConfig:
    """Resolved configuration for metrics, profiles, and search defaults."""

    enabled_metrics: List[str]
    profiles: List[RankingProfileSpec]
    default_profile: str
    search_defaults: SearchDefaults
    source_path: Optional[str] = None


def _coerce_float(value: object) -> Optional[float]:
    try:
        numeric = float(cast(float, value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def compute_base_image_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute the original sweep image metrics for one image."""

    as_float = np.asarray(image, dtype=np.float64)
    dx = np.diff(as_float, axis=1)
    dy = np.diff(as_float, axis=0)
    grad_x, grad_y = np.gradient(as_float)
    laplacian = cv.Laplacian(as_float, cv.CV_64F)
    return {
        "average_intensity": float(np.mean(as_float)),
        "maximum_intensity": float(np.max(as_float)),
        "total_variation": float(np.sum(np.abs(dx)) + np.sum(np.abs(dy))),
        "sum_squared_gradient": float(np.sum(np.square(grad_x)) + np.sum(np.square(grad_y))),
        "laplacian_variance": float(np.var(laplacian)),
        "dwt_level2_hv_energy": _haar_level2_hv_energy(as_float),
    }


def _haar_axis_decompose(values: np.ndarray, axis: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    size = values.shape[axis]
    even_size = size - (size % 2)
    if even_size < 2:
        return None, None
    slicer = [slice(None)] * values.ndim
    slicer[axis] = slice(0, even_size)
    trimmed = values[tuple(slicer)]
    even_idx = [slice(None)] * values.ndim
    odd_idx = [slice(None)] * values.ndim
    even_idx[axis] = slice(0, even_size, 2)
    odd_idx[axis] = slice(1, even_size, 2)
    scale = np.sqrt(2.0)
    return (trimmed[tuple(even_idx)] + trimmed[tuple(odd_idx)]) / scale, (
        trimmed[tuple(even_idx)] - trimmed[tuple(odd_idx)]
    ) / scale


def _haar_dwt2_level(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    low_cols, high_cols = _haar_axis_decompose(image, axis=1)
    if low_cols is None or high_cols is None:
        return None, None, None, None
    ll, lh = _haar_axis_decompose(low_cols, axis=0)
    hl, hh = _haar_axis_decompose(high_cols, axis=0)
    return ll, lh, hl, hh


def _haar_level2_hv_energy(image: np.ndarray) -> float:
    ll1, _, _, _ = _haar_dwt2_level(image)
    if ll1 is None:
        return 0.0
    _, lh2, hl2, _ = _haar_dwt2_level(ll1)
    if lh2 is None or hl2 is None:
        return 0.0
    return float(np.sum(np.square(lh2)) + np.sum(np.square(hl2)))


def preview_from_array(array: np.ndarray) -> np.ndarray:
    proxy = min_max_normalize(np.asarray(array, dtype=np.float64))
    return np.asarray(proxy, dtype=np.uint8)


def prepare_avg_image_channel(avg_image: np.ndarray) -> np.ndarray:
    return np.asarray(preview_from_array(avg_image), dtype=np.float64)


def prepare_guide_log_channel(avg_image: np.ndarray, sigma: float = DEFAULT_GUIDE_SIGMA) -> np.ndarray:
    avg_norm = prepare_avg_image_channel(avg_image).astype(np.float32)
    blurred = cv.GaussianBlur(avg_norm, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    lap = cv.Laplacian(blurred, cv.CV_32F)
    return np.abs(np.asarray(lap, dtype=np.float64))


def gradient_magnitude_channel(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    dy = np.zeros_like(arr, dtype=np.float64)
    dx = np.zeros_like(arr, dtype=np.float64)
    dy[1:, :] = arr[1:, :] - arr[:-1, :]
    dx[:, 1:] = arr[:, 1:] - arr[:, :-1]
    return np.sqrt(np.square(dx) + np.square(dy))


def tamura_coefficient(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.abs(arr)
    mean = float(np.mean(arr))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(arr) / mean)


def normalized_dct_channel(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    dct = np.abs(cv.dct(arr))
    dct = np.asarray(dct, dtype=np.float64)
    if dct.size == 0:
        return dct
    dct[0, 0] = 0.0
    total = float(np.sum(dct))
    if total <= 1e-12:
        return np.zeros_like(dct, dtype=np.float64)
    return dct / total


def normalized_shannon_entropy(channel: np.ndarray) -> float:
    arr = np.asarray(channel, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    values = arr.ravel()
    total = float(np.sum(values))
    if total <= 1e-12:
        return 0.0
    probs = values / total
    positive = probs[probs > 0.0]
    if positive.size <= 1:
        return 0.0
    denom = math.log(float(values.size))
    if denom <= 1e-12:
        return 0.0
    return float(-np.sum(positive * np.log(positive))) / denom


def _block_view_8x8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    padded_height = int(math.ceil(max(8, height) / 8.0) * 8)
    padded_width = int(math.ceil(max(8, width) / 8.0) * 8)
    pad_y = padded_height - height
    pad_x = padded_width - width
    if pad_y or pad_x:
        arr = np.pad(arr, ((0, pad_y), (0, pad_x)), mode="edge")
    return arr.reshape(arr.shape[0] // 8, 8, arr.shape[1] // 8, 8).swapaxes(1, 2)


def bayes_dct_entropy(image: np.ndarray, cutoff_sum: int = 6) -> float:
    blocks = _block_view_8x8(image)
    if blocks.size == 0:
        return 0.0
    mask = np.fromfunction(lambda y, x: (y + x) <= int(cutoff_sum), (8, 8), dtype=int)
    values: List[float] = []
    for row_idx in range(blocks.shape[0]):
        for col_idx in range(blocks.shape[1]):
            block = np.asarray(blocks[row_idx, col_idx], dtype=np.float32)
            coeffs = np.abs(cv.dct(block))
            selected = coeffs[mask]
            total = float(np.sum(selected))
            if total <= 1e-12:
                values.append(0.0)
                continue
            values.append(1.0 - float(np.sum(np.square(selected))) / (total * total))
    return float(np.mean(values)) if values else 0.0


def gini_index(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    flat = flat[flat >= 0.0]
    if flat.size <= 1:
        return 0.0
    total = float(np.sum(flat))
    if total <= 1e-12:
        return 0.0
    ordered = np.sort(flat)
    n = ordered.size
    coeff = np.sum((2.0 * np.arange(1, n + 1) - n - 1) * ordered)
    raw = float(coeff / (n * total))
    return max(0.0, min(1.0, raw * n / (n - 1)))


def hoyer_index(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    flat = np.abs(flat)
    if flat.size <= 1:
        return 0.0
    l2 = float(np.linalg.norm(flat, ord=2))
    if l2 <= 1e-12:
        return 0.0
    l1 = float(np.linalg.norm(flat, ord=1))
    scale = math.sqrt(float(flat.size))
    if abs(scale - 1.0) <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (scale - (l1 / l2)) / (scale - 1.0)))


def vollath_f4(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return 0.0
    return float(np.sum(arr[:-1, :] * arr[1:, :]) - np.sum(arr[:-2, :] * arr[2:, :]))


def compute_extended_image_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute old sweep metrics plus explorer avg/guide focus metrics."""

    avg_raw = np.asarray(image, dtype=np.float64)
    avg_image = prepare_avg_image_channel(avg_raw)
    guide_log = prepare_guide_log_channel(avg_raw)
    gradient_mag = gradient_magnitude_channel(avg_image)
    ndct_avg = normalized_dct_channel(avg_image)
    ndct_guide = normalized_dct_channel(guide_log)
    metrics = compute_base_image_metrics(avg_raw)
    metrics.update(
        {
            "ndct_shannon_avg": normalized_shannon_entropy(ndct_avg),
            "ndct_shannon_guide": normalized_shannon_entropy(ndct_guide),
            "ndct_bayes_avg": bayes_dct_entropy(avg_image),
            "ndct_bayes_guide": bayes_dct_entropy(guide_log),
            "gini_avg": gini_index(avg_image),
            "gini_guide": gini_index(guide_log),
            "hoyer_avg": hoyer_index(avg_image),
            "hoyer_guide": hoyer_index(guide_log),
            "vollath_f4_avg": vollath_f4(avg_image),
            "vollath_f4_guide": vollath_f4(guide_log),
            "tamura_gradient_avg": tamura_coefficient(gradient_mag),
            "tamura_guide": tamura_coefficient(guide_log),
        }
    )
    return {name: float(value) for name, value in metrics.items() if math.isfinite(float(value))}


def build_metric_registry(*, include_saved: bool = True) -> Dict[str, MetricDescriptor]:
    """Return the shared metric registry."""

    registry: Dict[str, MetricDescriptor] = {}
    if include_saved:
        registry.update(
            {
                "saved_score": MetricDescriptor(
                    name="saved_score",
                    label="Saved Dataset Score",
                    direction="max",
                    category="saved",
                    representation="saved",
                    description="Dataset-specific score as stored by the original sweep ranking.",
                    read_only=True,
                ),
                "saved_mean_dataset_score": MetricDescriptor(
                    name="saved_mean_dataset_score",
                    label="Saved Mean Dataset Score",
                    direction="max",
                    category="saved",
                    representation="saved",
                    description="Candidate mean-dataset score as stored by the original sweep ranking.",
                    read_only=True,
                ),
            }
        )
    old_metrics = {
        "average_intensity": "Average Intensity",
        "maximum_intensity": "Maximum Intensity",
        "total_variation": "Total Variation",
        "sum_squared_gradient": "Sum Squared Gradient",
        "laplacian_variance": "Laplacian Variance",
        "dwt_level2_hv_energy": "DWT Level-2 HV Energy",
    }
    for name, label in old_metrics.items():
        registry[name] = MetricDescriptor(
            name=name,
            label=label,
            direction="max",
            category="old",
            representation="avg_raw",
            description="Existing sweep metric recomputed from the raw rectified average image.",
        )
    new_metrics = {
        "ndct_shannon_avg": ("nDCT Shannon Entropy (Avg)", "avg_image"),
        "ndct_shannon_guide": ("nDCT Shannon Entropy (Guide)", "guide_log"),
        "ndct_bayes_avg": ("nDCT Bayes Entropy (Avg)", "avg_image"),
        "ndct_bayes_guide": ("nDCT Bayes Entropy (Guide)", "guide_log"),
        "gini_avg": ("Gini Index (Avg)", "avg_image"),
        "gini_guide": ("Gini Index (Guide)", "guide_log"),
        "hoyer_avg": ("Hoyer Index (Avg)", "avg_image"),
        "hoyer_guide": ("Hoyer Index (Guide)", "guide_log"),
        "vollath_f4_avg": ("Vollath F4 (Avg)", "avg_image"),
        "vollath_f4_guide": ("Vollath F4 (Guide)", "guide_log"),
    }
    for name, (label, representation) in new_metrics.items():
        registry[name] = MetricDescriptor(
            name=name,
            label=label,
            direction="max",
            category="new",
            representation=representation,
            description="Raw focus metric recomputed from the workspace artifact.",
        )
    tamura_metrics = {
        "tamura_gradient_avg": ("Tamura Gradient (Avg)", "avg_gradient"),
        "tamura_guide": ("Tamura Coefficient (Guide)", "guide_log"),
    }
    for name, (label, representation) in tamura_metrics.items():
        registry[name] = MetricDescriptor(
            name=name,
            label=label,
            direction="max",
            category="tamura",
            representation=representation,
            description="Tamura coefficient focus metric recomputed from the workspace artifact.",
        )
    return registry


def metric_names_by_categories(
    metric_registry: Mapping[str, MetricDescriptor], categories: Iterable[str]
) -> List[str]:
    category_set = {str(category) for category in categories}
    return [
        name
        for name, descriptor in metric_registry.items()
        if descriptor.category in category_set
    ]


def validate_metric_weights(
    weights: Mapping[str, object],
    metric_registry: Mapping[str, MetricDescriptor],
    *,
    allow_negative_weights: bool = False,
    reject_saved_only: bool = False,
) -> Dict[str, float]:
    valid: Dict[str, float] = {}
    for name, weight in weights.items():
        metric_name = str(name)
        numeric = _coerce_float(weight)
        if metric_name not in metric_registry or numeric is None:
            continue
        if reject_saved_only and metric_name in SAVED_ONLY_METRICS:
            continue
        if numeric == 0.0:
            continue
        if numeric < 0.0 and not allow_negative_weights:
            continue
        valid[metric_name] = float(numeric)
    if sum(abs(weight) for weight in valid.values()) <= 0.0:
        return {}
    return valid


def negative_metric_weight_names(
    weights: Mapping[str, object],
    metric_registry: Mapping[str, MetricDescriptor],
) -> List[str]:
    names: List[str] = []
    for metric_name, metric_weight in weights.items():
        name = str(metric_name)
        numeric = _coerce_float(metric_weight)
        if name in metric_registry and numeric is not None and float(numeric) < 0.0:
            names.append(name)
    return names


def validate_profile_for_sweep(
    profile: RankingProfileSpec,
    metric_registry: Optional[Mapping[str, MetricDescriptor]] = None,
) -> List[str]:
    registry = metric_registry or build_metric_registry()
    issues: List[str] = []
    saved_metrics = sorted(name for name in profile.metric_weights if name in SAVED_ONLY_METRICS)
    if saved_metrics:
        issues.append(
            f"Profile {profile.name!r} uses saved-only metrics that cannot rank new sweeps: "
            f"{', '.join(saved_metrics)}."
        )
    valid = validate_metric_weights(
        profile.metric_weights,
        registry,
        allow_negative_weights=profile.allow_negative_weights,
        reject_saved_only=True,
    )
    if not valid:
        issues.append(f"Profile {profile.name!r} has no sweep-compatible metric weights.")
    return issues


def _default_old_metric_weights(metric_registry: Mapping[str, MetricDescriptor]) -> Dict[str, float]:
    old_metrics = [name for name, desc in metric_registry.items() if desc.category == "old"]
    return {name: 1.0 / max(1, len(old_metrics)) for name in old_metrics}


def _default_new_metric_weights(metric_registry: Mapping[str, MetricDescriptor]) -> Dict[str, float]:
    new_metrics = [name for name, desc in metric_registry.items() if desc.category == "new"]
    return {name: 1.0 / max(1, len(new_metrics)) for name in new_metrics}


def blend_profile_weights(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    left_weight: float = 0.5,
    right_weight: float = 0.5,
) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for name, weight in left.items():
        merged[name] = merged.get(name, 0.0) + left_weight * float(weight)
    for name, weight in right.items():
        merged[name] = merged.get(name, 0.0) + right_weight * float(weight)
    total = float(sum(merged.values()))
    if total <= 0.0:
        return merged
    return {name: value / total for name, value in merged.items() if value > 0.0}


def default_profiles(
    metric_registry: Mapping[str, MetricDescriptor],
    scoring_weights: Optional[Mapping[str, float]] = None,
) -> List[RankingProfileSpec]:
    raw_scoring_weights = scoring_weights or {}
    old_weights = {
        name: float(weight)
        for name, weight in raw_scoring_weights.items()
        if name in metric_registry and metric_registry[name].category == "old"
    }
    if not old_weights:
        old_weights = _default_old_metric_weights(metric_registry)
    old_total = float(sum(old_weights.values()))
    if old_total > 0.0:
        old_weights = {name: value / old_total for name, value in old_weights.items()}
    new_weights = _default_new_metric_weights(metric_registry)
    hybrid = blend_profile_weights(old_weights, new_weights)
    profiles = [
        RankingProfileSpec(
            name="saved_original",
            label="Saved Original",
            metric_weights={"saved_score": 1.0},
            description="Original sweep ordering as stored by the sweep output.",
            built_in=True,
        ),
        RankingProfileSpec(
            name="old_recomputed",
            label="Old Recomputed",
            metric_weights=dict(old_weights),
            description="Original sweep metrics recomputed from workspace.npz.",
            built_in=True,
        ),
        RankingProfileSpec(
            name="new_dct_sparse",
            label="New DCT + Sparsity",
            metric_weights=dict(new_weights),
            description="Equal-weight blend of the new DCT, sparsity, and Vollath metrics.",
            built_in=True,
        ),
        RankingProfileSpec(
            name="tamura_tog",
            label="Tamura ToG",
            metric_weights={"tamura_gradient_avg": 1.0},
            description="Tamura coefficient of the average-image gradient magnitude.",
            built_in=True,
        ),
        RankingProfileSpec(
            name="tamura_focus_pair",
            label="Tamura Focus Pair",
            metric_weights={"tamura_gradient_avg": 0.5, "tamura_guide": 0.5},
            description="Equal-weight Tamura blend of average-image gradient and LoG-guide sparsity.",
            built_in=True,
        ),
        RankingProfileSpec(
            name="hybrid_default",
            label="Hybrid Default",
            metric_weights=dict(hybrid),
            description="Equal-weight blend of the recomputed old metrics and the new metrics.",
            built_in=True,
        ),
    ]
    return profiles


def load_config_payload(path: Optional[Path]) -> Tuple[Optional[Dict[str, object]], Optional[str], List[str]]:
    if path is None:
        return None, None, []
    warnings: List[str] = []
    try:
        payload, used_path, read_warnings = read_json_first_valid(
            path,
            validator=lambda obj: isinstance(obj, dict),
        )
        warnings.extend(read_warnings)
        return cast(Dict[str, object], payload), str(used_path), warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to read metric profile config {path}: {type(exc).__name__}: {exc}")
        return None, None, warnings


def resolve_metric_config(
    review_dir: Optional[Path],
    metric_registry: Mapping[str, MetricDescriptor],
    scoring_weights: Optional[Mapping[str, float]] = None,
    config_path: Optional[Path] = None,
) -> Tuple[MetricExplorerConfig, List[str]]:
    warnings: List[str] = []
    default_profile_list = default_profiles(metric_registry, scoring_weights)
    default_profile = "hybrid_default"
    auto_path = None if config_path is not None or review_dir is None else review_dir / "metrics_config.json"
    payload, source_path, read_warnings = load_config_payload(config_path or auto_path)
    warnings.extend(read_warnings)

    enabled_metrics = list(metric_registry.keys())
    profiles = list(default_profile_list)
    search_defaults = SearchDefaults()
    if payload is not None:
        configured_enabled = payload.get("enabled_metrics")
        if isinstance(configured_enabled, list):
            enabled_metrics = [str(name) for name in configured_enabled if str(name) in metric_registry]
        configured_profiles = payload.get("profiles")
        if isinstance(configured_profiles, list):
            merged = {profile.name: profile for profile in default_profile_list}
            for item in configured_profiles:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                label = str(item.get("label") or name).strip()
                weights_raw = item.get("metric_weights", {})
                if not name or not isinstance(weights_raw, dict):
                    continue
                allow_negative_weights = bool(item.get("allow_negative_weights", False))
                if not allow_negative_weights:
                    dropped_negative = negative_metric_weight_names(weights_raw, metric_registry)
                    if dropped_negative:
                        warnings.append(
                            f"Profile {name!r} has negative weights but allow_negative_weights is false; "
                            f"dropped: {', '.join(sorted(dropped_negative))}."
                        )
                weights = validate_metric_weights(
                    weights_raw,
                    metric_registry,
                    allow_negative_weights=allow_negative_weights,
                )
                if not weights:
                    continue
                merged[name] = RankingProfileSpec(
                    name=name,
                    label=label or name,
                    metric_weights=weights,
                    description=str(item.get("description", "")),
                    built_in=False,
                    allow_negative_weights=allow_negative_weights,
                )
            profiles = list(merged.values())
        default_profile_raw = payload.get("default_profile")
        if isinstance(default_profile_raw, str) and default_profile_raw.strip():
            default_profile = default_profile_raw.strip()
        search_defaults_raw = payload.get("search_defaults")
        if isinstance(search_defaults_raw, dict):
            subset_raw = search_defaults_raw.get("metric_subset", [])
            if isinstance(subset_raw, list):
                search_defaults.metric_subset = [
                    str(name) for name in subset_raw if str(name) in metric_registry
                ]
            step_raw = _coerce_float(search_defaults_raw.get("weight_step"))
            max_metrics_raw = _coerce_float(search_defaults_raw.get("max_metrics"))
            if step_raw is not None and step_raw > 0:
                search_defaults.weight_step = float(step_raw)
            if max_metrics_raw is not None and max_metrics_raw >= 1:
                search_defaults.max_metrics = int(max_metrics_raw)
            search_defaults.allow_negative_weights = bool(
                search_defaults_raw.get("allow_negative_weights", False)
            )

    enabled_set = set(enabled_metrics)
    filtered_profiles: List[RankingProfileSpec] = []
    for profile in profiles:
        enabled_weights = {
            metric_name: float(weight)
            for metric_name, weight in profile.metric_weights.items()
            if metric_name in enabled_set
        }
        enabled_weights = validate_metric_weights(
            enabled_weights,
            metric_registry,
            allow_negative_weights=profile.allow_negative_weights,
        )
        if not enabled_weights:
            continue
        filtered_profiles.append(
            RankingProfileSpec(
                name=profile.name,
                label=profile.label,
                metric_weights=enabled_weights,
                description=profile.description,
                built_in=profile.built_in,
                allow_negative_weights=profile.allow_negative_weights,
            )
        )
    profiles = filtered_profiles
    profile_names = {profile.name for profile in profiles}
    if default_profile not in profile_names and profiles:
        default_profile = "hybrid_default" if "hybrid_default" in profile_names else profiles[0].name
    return (
        MetricExplorerConfig(
            enabled_metrics=enabled_metrics,
            profiles=profiles,
            default_profile=default_profile,
            search_defaults=search_defaults,
            source_path=source_path,
        ),
        warnings,
    )


def resolve_profile_weights_for_sweep(
    profile_name: str,
    *,
    config_path: Optional[Path] = None,
    scoring_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], RankingProfileSpec, List[str]]:
    registry = build_metric_registry()
    config, warnings = resolve_metric_config(
        None,
        registry,
        scoring_weights=scoring_weights,
        config_path=config_path,
    )
    profile = next((item for item in config.profiles if item.name == profile_name), None)
    if profile is None:
        raise ValueError(f"Unknown scoring profile {profile_name!r}.")
    issues = validate_profile_for_sweep(profile, registry)
    if issues:
        raise ValueError(" ".join(issues))
    weights = validate_metric_weights(
        profile.metric_weights,
        registry,
        allow_negative_weights=profile.allow_negative_weights,
        reject_saved_only=True,
    )
    if not weights:
        raise ValueError(f"Scoring profile {profile_name!r} has no sweep-compatible metric weights.")
    return weights, profile, warnings


def profile_to_config_payload(profile: RankingProfileSpec) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "name": profile.name,
        "label": profile.label,
        "metric_weights": dict(profile.metric_weights),
        "description": profile.description,
    }
    if profile.allow_negative_weights:
        payload["allow_negative_weights"] = True
    return payload


def save_metric_config(
    path: Path,
    config: MetricExplorerConfig,
    *,
    existing_payload: Optional[Mapping[str, object]] = None,
) -> Path:
    merged_payload = dict(existing_payload) if existing_payload is not None else {}
    merged_payload["enabled_metrics"] = list(config.enabled_metrics)
    merged_payload["profiles"] = [
        profile_to_config_payload(profile)
        for profile in config.profiles
        if not profile.built_in
    ]
    merged_payload["default_profile"] = config.default_profile
    merged_payload["search_defaults"] = {
        "metric_subset": list(config.search_defaults.metric_subset),
        "weight_step": float(config.search_defaults.weight_step),
        "max_metrics": int(config.search_defaults.max_metrics),
        "allow_negative_weights": bool(config.search_defaults.allow_negative_weights),
    }
    write_json_atomic(path, merged_payload, indent=2, sort_keys=False, create_backup=True)
    config.source_path = str(path)
    return path


_preview_from_array = preview_from_array
_prepare_avg_image_channel = prepare_avg_image_channel
_prepare_guide_log_channel = prepare_guide_log_channel
_gradient_magnitude_channel = gradient_magnitude_channel
_tamura_coefficient = tamura_coefficient
_normalized_dct_channel = normalized_dct_channel
_normalized_shannon_entropy = normalized_shannon_entropy
_bayes_dct_entropy = bayes_dct_entropy
_gini_index = gini_index
_hoyer_index = hoyer_index
_vollath_f4 = vollath_f4
_validated_metric_weights = validate_metric_weights
_negative_metric_weight_names = negative_metric_weight_names
_metric_names_by_categories = metric_names_by_categories
_blend_profile_weights = blend_profile_weights
_default_profiles = default_profiles
_load_config_payload = load_config_payload
_resolve_metric_config = resolve_metric_config
_profile_to_config_payload = profile_to_config_payload

"""Interactive explorer for re-ranking sweep outputs from raw workspace artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

import cv2 as cv
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from moment_images.im_process import min_max_normalize
from moment_images.reliable_io import (
    read_json_first_valid,
    read_npz_first_valid,
    write_csv_atomic,
    write_json_atomic,
)
from moment_images.sweep_analysis import compute_image_metrics
from moment_images.sweep_visualization import (
    CandidateVisualizationRow,
    SweepVisualizationModel,
    load_sweep_visualization_model,
)


REGISTRY_VERSION = "2026-05-24-v1"
DEFAULT_GUIDE_SIGMA = 1.0
DEFAULT_SEARCH_WEIGHT_STEP = 0.1
DEFAULT_TOP_K = 5
ANNOTATION_LABEL_SCORES = {"bad": 0.0, "good": 1.0, "top": 2.0}


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


@dataclass
class SearchDefaults:
    """Configuration for search actions in the explorer."""

    metric_subset: List[str] = field(default_factory=list)
    weight_step: float = DEFAULT_SEARCH_WEIGHT_STEP
    max_metrics: int = 6


@dataclass
class MetricExplorerConfig:
    """Resolved configuration for metrics, profiles, and search defaults."""

    enabled_metrics: List[str]
    profiles: List[RankingProfileSpec]
    default_profile: str
    search_defaults: SearchDefaults
    source_path: Optional[str] = None


@dataclass
class QualityAnnotation:
    """One manual quality judgment for a stage/candidate/dataset tuple."""

    stage: str
    candidate_id: str
    dataset_id: str
    label: str
    note: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def score(self) -> float:
        return ANNOTATION_LABEL_SCORES.get(self.label, float("nan"))


@dataclass
class RawMetricRecord:
    """Raw per-dataset metrics recomputed from ``workspace.npz``."""

    row_id: str
    stage: str
    candidate_id: str
    dataset_id: str
    strategy: str
    family: Optional[str]
    alpha: Optional[float]
    formulation_id: Optional[str]
    preview_path: Optional[str]
    workspace_path: Optional[str]
    saved_score: Optional[float]
    saved_mean_dataset_score: Optional[float]
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProfileCandidateScore:
    """Ranked score for one candidate under a selected profile."""

    profile_name: str
    stage: str
    candidate_id: str
    strategy: str
    family: Optional[str]
    alpha: Optional[float]
    rank: int
    score: float
    mean_dataset_score: float
    std_dataset_score: float
    missing_datasets: List[str]
    dataset_scores: Dict[str, float]
    contributions: Dict[str, float]


@dataclass
class ProfileEvaluationSummary:
    """Aggregate profile quality against manual annotations."""

    stage: str
    profile_name: str
    pairwise_agreement: Optional[float]
    top1_hit_rate: Optional[float]
    mean_label_top_k: Optional[float]
    spearman_correlation: Optional[float]
    num_labeled_items: int
    num_pairwise_comparisons: int
    num_datasets_considered: int


@dataclass
class DisagreementRecord:
    """One case where a lower-quality label outranks a higher-quality label."""

    stage: str
    profile_name: str
    dataset_id: str
    higher_candidate_id: str
    higher_label: str
    higher_rank: int
    lower_candidate_id: str
    lower_label: str
    lower_rank: int
    rank_gap: int
    label_gap: float


@dataclass
class SearchResult:
    """Outcome of one profile search weight combination."""

    stage: str
    profile_name: str
    metric_weights: Dict[str, float]
    pairwise_agreement: Optional[float]
    top1_hit_rate: Optional[float]
    mean_label_top_k: Optional[float]
    spearman_correlation: Optional[float]


ComparisonRecordKey = Tuple[str, str, str]


@dataclass
class ComparisonMetricRow:
    """One metric row shown in the A/B comparison table."""

    metric_name: str
    metric_label: str
    category: str
    direction: str
    value_a: float
    value_b: float
    delta: float
    winner: str


@dataclass
class ComparisonDisplayBundle:
    """Display-ready image channels for one raw record."""

    preview: Optional[np.ndarray]
    avg_image: Optional[np.ndarray]
    guide_log: Optional[np.ndarray]
    warnings: List[str] = field(default_factory=list)

    def get(self, view_name: str) -> Optional[np.ndarray]:
        if view_name == "preview":
            return self.preview
        if view_name == "avg_image":
            return self.avg_image
        if view_name == "guide_log":
            return self.guide_log
        return None


@dataclass
class SweepMetricExplorerModel:
    """Canonical model used by the raw metric explorer."""

    base_model: SweepVisualizationModel
    review_dir: Path
    cache_path: Path
    annotations_path: Path
    config: MetricExplorerConfig
    metric_registry: Dict[str, MetricDescriptor]
    raw_records: List[RawMetricRecord]
    profile_results: Dict[Tuple[str, str], List[ProfileCandidateScore]]
    evaluations: List[ProfileEvaluationSummary]
    disagreements: List[DisagreementRecord]
    annotations: Dict[str, QualityAnnotation]
    search_results: List[SearchResult]
    warnings: List[str]
    scoring_config: Dict[str, float]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _open_with_system(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        target = str(path)
        system_name = platform.system().lower()
        if system_name.startswith("win"):
            import os

            os.startfile(target)  # type: ignore[attr-defined]
            return True, None
        if system_name == "darwin":
            subprocess.run(["open", target], check=False)
            return True, None
        subprocess.run(["xdg-open", target], check=False)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _coerce_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _annotation_key(stage: str, candidate_id: str, dataset_id: str) -> str:
    return f"{stage}:{candidate_id}:{dataset_id}"


def _real_datasets(model: SweepVisualizationModel) -> List[str]:
    return list(model.datasets or ["aggregate"])


def _metric_row_key(record: RawMetricRecord) -> Tuple[str, str, str]:
    return record.stage, record.candidate_id, record.dataset_id


def _stable_unique_strings(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _candidate_lookup(rows: Sequence[CandidateVisualizationRow]) -> Dict[Tuple[str, str], CandidateVisualizationRow]:
    return {(row.stage, row.candidate_id): row for row in rows}


def _record_lookup(records: Sequence[RawMetricRecord]) -> Dict[Tuple[str, str, str], RawMetricRecord]:
    return {_metric_row_key(record): record for record in records}


def _record_key_to_text(key: Optional[ComparisonRecordKey]) -> str:
    if key is None:
        return "not set"
    stage_name, candidate_id, dataset_id = key
    return f"{candidate_id} / {dataset_id} / {stage_name}"


def _stringify_param(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:g}"
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _candidate_score_for_dataset(
    row: CandidateVisualizationRow,
    dataset_id: str,
) -> Optional[float]:
    if dataset_id == "aggregate":
        return row.score if row.score is not None else row.mean_dataset_score
    if dataset_id in row.dataset_scores:
        return row.dataset_scores.get(dataset_id)
    return row.score


def _dataset_runtime_for_row(row: CandidateVisualizationRow, dataset_id: str) -> Optional[float]:
    if dataset_id == "aggregate":
        return row.mean_runtime_seconds
    return row.dataset_runtimes.get(dataset_id, row.mean_runtime_seconds)


def _normalize_score(value: float, min_value: float, max_value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return 0.0
    if abs(max_value - min_value) <= 1e-12:
        return 0.5
    return float((value - min_value) / (max_value - min_value))


def _weighted_average(values: Mapping[str, float], weights: Mapping[str, float]) -> Optional[float]:
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in weights.items():
        value = values.get(name)
        if value is None or not math.isfinite(float(value)):
            continue
        weighted_sum += float(weight) * float(value)
        weight_sum += float(weight)
    if weight_sum <= 0.0:
        return None
    return weighted_sum / weight_sum


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.zeros(len(values), dtype=np.float64)
    idx = 0
    while idx < len(order):
        start = idx
        value = float(values[order[idx]])
        while idx + 1 < len(order) and float(values[order[idx + 1]]) == value:
            idx += 1
        end = idx
        mean_rank = (start + end + 2) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = mean_rank
        idx += 1
    return ranks


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    x_std = float(np.std(x_arr))
    y_std = float(np.std(y_arr))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson_correlation(_average_ranks(xs), _average_ranks(ys))


def _preview_from_array(array: np.ndarray) -> np.ndarray:
    proxy = min_max_normalize(np.asarray(array, dtype=np.float64))
    preview = np.asarray(proxy, dtype=np.uint8)
    return preview


def _prepare_avg_image_channel(avg_image: np.ndarray) -> np.ndarray:
    return np.asarray(_preview_from_array(avg_image), dtype=np.float64)


def _prepare_guide_log_channel(avg_image: np.ndarray, sigma: float = DEFAULT_GUIDE_SIGMA) -> np.ndarray:
    avg_norm = _prepare_avg_image_channel(avg_image).astype(np.float32)
    blurred = cv.GaussianBlur(avg_norm, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    lap = cv.Laplacian(blurred, cv.CV_32F)
    return np.abs(np.asarray(lap, dtype=np.float64))


def _normalized_dct_channel(image: np.ndarray) -> np.ndarray:
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


def _normalized_shannon_entropy(channel: np.ndarray) -> float:
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
    entropy = float(-np.sum(positive * np.log(positive)))
    denom = math.log(float(values.size))
    if denom <= 1e-12:
        return 0.0
    return entropy / denom


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


def _bayes_dct_entropy(image: np.ndarray, cutoff_sum: int = 6) -> float:
    blocks = _block_view_8x8(image)
    if blocks.size == 0:
        return 0.0
    mask = np.fromfunction(
        lambda y, x: (y + x) <= int(cutoff_sum),
        (8, 8),
        dtype=int,
    )
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


def _gini_index(values: np.ndarray) -> float:
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


def _hoyer_index(values: np.ndarray) -> float:
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


def _vollath_f4(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return 0.0
    return float(np.sum(arr[:-1, :] * arr[1:, :]) - np.sum(arr[:-2, :] * arr[2:, :]))


def build_metric_registry() -> Dict[str, MetricDescriptor]:
    """Return the built-in metric registry for the explorer."""

    registry = {
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
    return registry


def _default_old_metric_weights(metric_registry: Mapping[str, MetricDescriptor]) -> Dict[str, float]:
    old_metrics = [name for name, desc in metric_registry.items() if desc.category == "old"]
    return {name: 1.0 / max(1, len(old_metrics)) for name in old_metrics}


def _default_new_metric_weights(metric_registry: Mapping[str, MetricDescriptor]) -> Dict[str, float]:
    new_metrics = [name for name, desc in metric_registry.items() if desc.category == "new"]
    return {name: 1.0 / max(1, len(new_metrics)) for name in new_metrics}


def _blend_profile_weights(
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


def _default_profiles(
    metric_registry: Mapping[str, MetricDescriptor],
    scoring_weights: Mapping[str, float],
) -> List[RankingProfileSpec]:
    old_weights = {
        name: float(weight)
        for name, weight in scoring_weights.items()
        if name in metric_registry and metric_registry[name].category == "old"
    }
    if not old_weights:
        old_weights = _default_old_metric_weights(metric_registry)
    old_total = float(sum(old_weights.values()))
    if old_total > 0.0:
        old_weights = {name: value / old_total for name, value in old_weights.items()}

    new_weights = _default_new_metric_weights(metric_registry)
    hybrid = _blend_profile_weights(old_weights, new_weights)
    return [
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
            name="hybrid_default",
            label="Hybrid Default",
            metric_weights=dict(hybrid),
            description="Equal-weight blend of the recomputed old metrics and the new metrics.",
            built_in=True,
        ),
    ]


def _load_config_payload(path: Optional[Path]) -> Tuple[Optional[Dict[str, object]], Optional[str], List[str]]:
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
        warnings.append(f"Failed to read metric explorer config {path}: {type(exc).__name__}: {exc}")
        return None, None, warnings


def _resolve_metric_config(
    review_dir: Path,
    metric_registry: Mapping[str, MetricDescriptor],
    scoring_weights: Mapping[str, float],
    config_path: Optional[Path],
) -> Tuple[MetricExplorerConfig, List[str]]:
    warnings: List[str] = []
    default_profiles = _default_profiles(metric_registry, scoring_weights)
    default_profile = "hybrid_default"
    auto_path = None if config_path is not None else review_dir / "metrics_config.json"
    payload, source_path, read_warnings = _load_config_payload(config_path or auto_path)
    warnings.extend(read_warnings)

    enabled_metrics = list(metric_registry.keys())
    profiles = list(default_profiles)
    search_defaults = SearchDefaults()
    if payload is not None:
        configured_enabled = payload.get("enabled_metrics")
        if isinstance(configured_enabled, list):
            enabled_metrics = [
                str(name)
                for name in configured_enabled
                if str(name) in metric_registry
            ]
        configured_profiles = payload.get("profiles")
        if isinstance(configured_profiles, list):
            merged = {profile.name: profile for profile in default_profiles}
            for item in configured_profiles:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                label = str(item.get("label") or name).strip()
                weights_raw = item.get("metric_weights", {})
                if not name or not isinstance(weights_raw, dict):
                    continue
                weights = {
                    str(metric_name): float(metric_weight)
                    for metric_name, metric_weight in weights_raw.items()
                    if str(metric_name) in metric_registry and _coerce_float(metric_weight) is not None
                }
                if not weights:
                    continue
                merged[name] = RankingProfileSpec(
                    name=name,
                    label=label or name,
                    metric_weights=weights,
                    description=str(item.get("description", "")),
                    built_in=False,
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
                    str(name)
                    for name in subset_raw
                    if str(name) in metric_registry
                ]
            step_raw = _coerce_float(search_defaults_raw.get("weight_step"))
            max_metrics_raw = _coerce_float(search_defaults_raw.get("max_metrics"))
            if step_raw is not None and step_raw > 0:
                search_defaults.weight_step = float(step_raw)
            if max_metrics_raw is not None and max_metrics_raw >= 1:
                search_defaults.max_metrics = int(max_metrics_raw)

    enabled_set = set(enabled_metrics)
    profiles = [
        RankingProfileSpec(
            name=profile.name,
            label=profile.label,
            metric_weights={
                metric_name: float(weight)
                for metric_name, weight in profile.metric_weights.items()
                if metric_name in enabled_set
            },
            description=profile.description,
            built_in=profile.built_in,
        )
        for profile in profiles
        if any(metric_name in enabled_set for metric_name in profile.metric_weights)
    ]
    profile_names = {profile.name for profile in profiles}
    if default_profile not in profile_names:
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


def _scoring_config(model: SweepVisualizationModel) -> Dict[str, float]:
    scoring = cast(Dict[str, object], model.experiment_metadata.get("scoring", {}))
    spec = cast(Dict[str, object], model.experiment_metadata.get("spec", {}))
    spec_scoring = cast(Dict[str, object], spec.get("scoring", {}))
    payload = spec_scoring or scoring
    return {
        "runtime_penalty_weight": float(payload.get("runtime_penalty_weight", 0.0)),
        "robustness_lambda": float(payload.get("robustness_lambda", 0.5)),
        "missing_dataset_penalty": float(payload.get("missing_dataset_penalty", 1.0)),
    }


def _extract_identity_from_workspace(
    workspace_path: Path,
    *,
    experiment_root: Path,
    metric_phase_dir: Optional[Path],
) -> Optional[Tuple[str, str, str]]:
    parts = workspace_path.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if len(parts) >= idx + 5:
            return parts[idx + 1], parts[idx + 2], parts[idx + 3]
    if workspace_path.parent.name and workspace_path.parent.parent is not None:
        candidate_id = workspace_path.parent.parent.name
        return "legacy", candidate_id, "aggregate"
    return None


def _candidate_dataset_ids(row: CandidateVisualizationRow, model: SweepVisualizationModel) -> List[str]:
    ids = set(row.dataset_scores.keys()) | set(row.dataset_metrics.keys()) | set(row.preview_paths.keys())
    if not ids:
        ids.update(_real_datasets(model))
    return sorted(str(dataset_id) for dataset_id in ids)


def _load_annotations(path: Path) -> Dict[str, QualityAnnotation]:
    if not path.exists():
        return {}
    try:
        payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    annotations: Dict[str, QualityAnnotation] = {}
    for row in cast(List[object], payload.get("annotations", [])):
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        dataset_id = str(row.get("dataset_id", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not stage or not candidate_id or not dataset_id or label not in ANNOTATION_LABEL_SCORES:
            continue
        annotation = QualityAnnotation(
            stage=stage,
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            label=label,
            note=str(row.get("note", "")),
            updated_at=str(row.get("updated_at", _utc_now_iso())),
        )
        annotations[_annotation_key(stage, candidate_id, dataset_id)] = annotation
    return annotations


def save_annotations(model: SweepMetricExplorerModel) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "annotations": [asdict(annotation) for annotation in model.annotations.values()],
    }
    write_json_atomic(model.annotations_path, payload, indent=2, sort_keys=False, create_backup=True)


def _load_metric_cache(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries", {})
    return cast(Dict[str, Dict[str, object]], entries if isinstance(entries, dict) else {})


def _save_metric_cache(path: Path, entries: Mapping[str, Dict[str, object]]) -> None:
    payload = {
        "schema_version": 1,
        "registry_version": REGISTRY_VERSION,
        "generated_at": _utc_now_iso(),
        "entries": dict(entries),
    }
    write_json_atomic(path, payload, indent=2, sort_keys=False, create_backup=True)


def _cache_key_for_workspace(workspace_path: Optional[Path]) -> str:
    if workspace_path is None or not workspace_path.exists():
        return "missing"
    stat = workspace_path.stat()
    return "|".join(
        [
            REGISTRY_VERSION,
            str(workspace_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ]
    )


def _load_workspace_avg_image(workspace_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], List[str]]:
    warnings: List[str] = []
    try:
        payload, _, read_warnings = read_npz_first_valid(
            workspace_path,
            allow_pickle=True,
            validator=lambda obj: isinstance(obj, dict),
        )
        warnings.extend(read_warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to load workspace {workspace_path}: {type(exc).__name__}: {exc}")
        return None, {}, warnings
    avg_image = cast(Optional[np.ndarray], payload.get("average_rectified_image"))
    if not isinstance(avg_image, np.ndarray):
        rectified = cast(Optional[np.ndarray], payload.get("rectified_views"))
        if isinstance(rectified, np.ndarray) and rectified.ndim >= 4:
            avg_image = np.mean(np.asarray(rectified, dtype=np.float64), axis=(0, 1))
    if not isinstance(avg_image, np.ndarray):
        warnings.append("Workspace missing average_rectified_image and usable rectified_views.")
        return None, payload, warnings
    return np.asarray(avg_image, dtype=np.float64), payload, warnings


def compute_metrics_from_workspace(
    workspace_path: Path,
) -> Tuple[Dict[str, float], List[str]]:
    """Compute all raw metrics for a workspace artifact."""

    avg_image_raw, payload, warnings = _load_workspace_avg_image(workspace_path)
    if avg_image_raw is None:
        return {}, warnings

    avg_image = _prepare_avg_image_channel(avg_image_raw)
    guide_log = _prepare_guide_log_channel(avg_image_raw)
    ndct_avg = _normalized_dct_channel(avg_image)
    ndct_guide = _normalized_dct_channel(guide_log)

    metrics = compute_image_metrics(np.asarray(avg_image_raw, dtype=np.float64))
    metrics.update(
        {
            "ndct_shannon_avg": _normalized_shannon_entropy(ndct_avg),
            "ndct_shannon_guide": _normalized_shannon_entropy(ndct_guide),
            "ndct_bayes_avg": _bayes_dct_entropy(avg_image),
            "ndct_bayes_guide": _bayes_dct_entropy(guide_log),
            "gini_avg": _gini_index(avg_image),
            "gini_guide": _gini_index(guide_log),
            "hoyer_avg": _hoyer_index(avg_image),
            "hoyer_guide": _hoyer_index(guide_log),
            "vollath_f4_avg": _vollath_f4(avg_image),
            "vollath_f4_guide": _vollath_f4(guide_log),
        }
    )
    global_avg = cast(Optional[np.ndarray], payload.get("average_rectified_global_image"))
    if isinstance(global_avg, np.ndarray):
        global_norm = _prepare_avg_image_channel(np.asarray(global_avg, dtype=np.float64))
        metrics["saved_global_preview_mean"] = float(np.mean(global_norm))
    return {name: float(value) for name, value in metrics.items() if math.isfinite(float(value))}, warnings


def compute_image_channels(
    workspace_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Load the comparison-ready average and guide images for a workspace."""

    avg_image_raw, _, warnings = _load_workspace_avg_image(workspace_path)
    if avg_image_raw is None:
        return None, None, warnings
    avg_image = _prepare_avg_image_channel(avg_image_raw)
    guide_log = _prepare_guide_log_channel(avg_image_raw)
    return avg_image, guide_log, warnings


def load_comparison_display_bundle(record: RawMetricRecord) -> ComparisonDisplayBundle:
    """Load preview and raw-derived display channels for one comparison record."""

    preview: Optional[np.ndarray] = None
    warnings = list(record.warnings)
    if record.preview_path:
        preview_path = Path(record.preview_path)
        if preview_path.exists():
            try:
                preview = np.asarray(Image.open(preview_path).convert("L"), dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"Failed to load preview {preview_path.name}: {type(exc).__name__}: {exc}"
                )
    avg_image = guide_log = None
    if record.workspace_path:
        avg_image, guide_log, channel_warnings = compute_image_channels(Path(record.workspace_path))
        warnings.extend(channel_warnings)
    if preview is None:
        preview = avg_image
    return ComparisonDisplayBundle(
        preview=preview,
        avg_image=avg_image,
        guide_log=guide_log,
        warnings=_stable_unique_strings(warnings),
    )


def _scan_raw_records(
    model: SweepVisualizationModel,
    review_dir: Path,
    metric_registry: Mapping[str, MetricDescriptor],
) -> Tuple[List[RawMetricRecord], List[str], Path]:
    cache_path = review_dir / "cache" / "metric_cache.json"
    cache_entries = _load_metric_cache(cache_path)
    warnings: List[str] = []
    raw_records: List[RawMetricRecord] = []
    candidate_lookup = _candidate_lookup(model.candidates)

    for row in model.candidates:
        dataset_ids = _candidate_dataset_ids(row, model)
        for dataset_id in dataset_ids:
            base_dir = (
                model.metric_phase_dir
                if row.stage == "metric_phase" and model.metric_phase_dir is not None
                else model.experiment_root
            )
            workspace_path = base_dir / "runs" / row.stage / row.candidate_id / dataset_id / "workspace.npz"
            preview_path = row.preview_paths.get(dataset_id) or row.preview_paths.get("aggregate")
            if preview_path is None:
                preview_candidate = base_dir / "runs" / row.stage / row.candidate_id / dataset_id / "avg_rectified.png"
                preview_path = str(preview_candidate) if preview_candidate.exists() else None
            cache_key = _cache_key_for_workspace(workspace_path if workspace_path.exists() else None)
            entry = cache_entries.get(_annotation_key(row.stage, row.candidate_id, dataset_id), {})
            entry_key = str(entry.get("cache_key", ""))
            metrics: Dict[str, float] = {}
            metric_warnings: List[str] = []
            if entry_key == cache_key:
                cached_metrics = entry.get("metrics", {})
                if isinstance(cached_metrics, dict):
                    metrics = {
                        str(name): float(value)
                        for name, value in cached_metrics.items()
                        if str(name) in metric_registry and _coerce_float(value) is not None
                    }
                    metric_warnings = [str(item) for item in cast(List[object], entry.get("warnings", []))]
            elif workspace_path.exists():
                metrics, metric_warnings = compute_metrics_from_workspace(workspace_path)
                cache_entries[_annotation_key(row.stage, row.candidate_id, dataset_id)] = {
                    "cache_key": cache_key,
                    "metrics": metrics,
                    "warnings": metric_warnings,
                }
            else:
                metric_warnings = [f"Workspace not found for {row.stage}/{row.candidate_id}/{dataset_id}."]
                cache_entries[_annotation_key(row.stage, row.candidate_id, dataset_id)] = {
                    "cache_key": cache_key,
                    "metrics": {},
                    "warnings": metric_warnings,
                }

            metrics["saved_score"] = _candidate_score_for_dataset(row, dataset_id) or float("nan")
            metrics["saved_mean_dataset_score"] = row.mean_dataset_score or float("nan")
            parsed_config = dict(row.config)
            strategy = str(parsed_config.get("strategy", row.strategy))
            family = row.family
            alpha = row.alpha
            formulation_id = row.formulation_id
            raw_records.append(
                RawMetricRecord(
                    row_id=_annotation_key(row.stage, row.candidate_id, dataset_id),
                    stage=row.stage,
                    candidate_id=row.candidate_id,
                    dataset_id=dataset_id,
                    strategy=strategy,
                    family=family,
                    alpha=alpha,
                    formulation_id=formulation_id,
                    preview_path=preview_path,
                    workspace_path=str(workspace_path) if workspace_path.exists() else None,
                    saved_score=_candidate_score_for_dataset(row, dataset_id),
                    saved_mean_dataset_score=row.mean_dataset_score,
                    metrics={
                        name: float(value)
                        for name, value in metrics.items()
                        if name in metric_registry and math.isfinite(float(value))
                    },
                    warnings=metric_warnings,
                )
            )
            warnings.extend(metric_warnings)
    _save_metric_cache(cache_path, cache_entries)
    return raw_records, warnings, cache_path


def _saved_original_profile_result(
    row: CandidateVisualizationRow,
) -> ProfileCandidateScore:
    datasets = sorted(row.dataset_scores.keys())
    mean_score = row.mean_dataset_score if row.mean_dataset_score is not None else float("nan")
    std_score = row.std_dataset_score if row.std_dataset_score is not None else 0.0
    score = row.score if row.score is not None else mean_score
    return ProfileCandidateScore(
        profile_name="saved_original",
        stage=row.stage,
        candidate_id=row.candidate_id,
        strategy=row.strategy,
        family=row.family,
        alpha=row.alpha,
        rank=row.rank,
        score=float(score) if score is not None else float("-inf"),
        mean_dataset_score=float(mean_score) if mean_score is not None else float("nan"),
        std_dataset_score=float(std_score),
        missing_datasets=list(row.missing_datasets),
        dataset_scores=dict(row.dataset_scores),
        contributions={"saved_score": float(score) if score is not None else float("nan")},
    )


def _score_profile_for_stage(
    stage_name: str,
    profile: RankingProfileSpec,
    model: SweepVisualizationModel,
    raw_records: Sequence[RawMetricRecord],
    metric_registry: Mapping[str, MetricDescriptor],
    scoring_config: Mapping[str, float],
) -> List[ProfileCandidateScore]:
    stage_rows = [row for row in model.candidates if row.stage == stage_name]
    if not stage_rows:
        return []
    if profile.name == "saved_original":
        return [_saved_original_profile_result(row) for row in stage_rows]

    record_lookup = _record_lookup(raw_records)
    datasets = _real_datasets(model)
    metric_weights = {
        name: float(weight)
        for name, weight in profile.metric_weights.items()
        if name in metric_registry and float(weight) > 0.0
    }
    if not metric_weights:
        return []

    dataset_metric_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {
        dataset_id: {} for dataset_id in datasets
    }
    dataset_runtime_bounds: Dict[str, Tuple[float, float]] = {}
    for dataset_id in datasets:
        runtime_values: List[float] = []
        for metric_name in metric_weights:
            metric_values: List[float] = []
            for row in stage_rows:
                record = record_lookup.get((stage_name, row.candidate_id, dataset_id))
                if record is None:
                    continue
                numeric = record.metrics.get(metric_name)
                if numeric is not None and math.isfinite(float(numeric)):
                    metric_values.append(float(numeric))
            if metric_values:
                dataset_metric_bounds[dataset_id][metric_name] = (
                    float(np.min(metric_values)),
                    float(np.max(metric_values)),
                )
            else:
                dataset_metric_bounds[dataset_id][metric_name] = (0.0, 0.0)
        for row in stage_rows:
            runtime = _dataset_runtime_for_row(row, dataset_id)
            if runtime is not None and math.isfinite(float(runtime)):
                runtime_values.append(float(runtime))
        if runtime_values:
            dataset_runtime_bounds[dataset_id] = (
                float(np.min(runtime_values)),
                float(np.max(runtime_values)),
            )
        else:
            dataset_runtime_bounds[dataset_id] = (0.0, 0.0)

    scored_rows: List[ProfileCandidateScore] = []
    candidate_lookup = _candidate_lookup(stage_rows)
    for row in stage_rows:
        dataset_scores: Dict[str, float] = {}
        contributions_accumulator: Dict[str, List[float]] = {}
        missing: List[str] = []
        for dataset_id in datasets:
            record = record_lookup.get((stage_name, row.candidate_id, dataset_id))
            if record is None:
                missing.append(dataset_id)
                continue
            normalized_values: Dict[str, float] = {}
            for metric_name, weight in metric_weights.items():
                raw_value = record.metrics.get(metric_name)
                if raw_value is None or not math.isfinite(float(raw_value)):
                    normalized_values = {}
                    break
                min_value, max_value = dataset_metric_bounds[dataset_id][metric_name]
                normalized = _normalize_score(float(raw_value), min_value, max_value)
                if metric_registry[metric_name].direction == "min":
                    normalized = 1.0 - normalized
                normalized_values[metric_name] = normalized
            if not normalized_values:
                missing.append(dataset_id)
                continue
            dataset_score = _weighted_average(normalized_values, metric_weights)
            if dataset_score is None:
                missing.append(dataset_id)
                continue
            runtime_penalty = float(scoring_config.get("runtime_penalty_weight", 0.0))
            if runtime_penalty > 0.0:
                runtime = _dataset_runtime_for_row(row, dataset_id)
                if runtime is not None and math.isfinite(float(runtime)):
                    rt_min, rt_max = dataset_runtime_bounds[dataset_id]
                    dataset_score -= runtime_penalty * _normalize_score(float(runtime), rt_min, rt_max)
            dataset_scores[dataset_id] = float(dataset_score)
            for metric_name, normalized in normalized_values.items():
                contributions_accumulator.setdefault(metric_name, []).append(
                    float(metric_weights[metric_name]) * normalized
                )

        if dataset_scores:
            values = list(dataset_scores.values())
            mean_score = float(np.mean(values))
            std_score = float(np.std(values))
        else:
            mean_score = float("-inf")
            std_score = 0.0
        score = mean_score
        score -= float(scoring_config.get("robustness_lambda", 0.5)) * std_score
        score -= float(scoring_config.get("missing_dataset_penalty", 1.0)) * len(missing)
        contributions = {
            metric_name: float(np.mean(values))
            for metric_name, values in contributions_accumulator.items()
            if values
        }
        scored_rows.append(
            ProfileCandidateScore(
                profile_name=profile.name,
                stage=stage_name,
                candidate_id=row.candidate_id,
                strategy=row.strategy,
                family=row.family,
                alpha=row.alpha,
                rank=0,
                score=float(score),
                mean_dataset_score=float(mean_score),
                std_dataset_score=float(std_score),
                missing_datasets=missing,
                dataset_scores=dataset_scores,
                contributions=contributions,
            )
        )

    scored_rows.sort(key=lambda item: (item.score, item.mean_dataset_score), reverse=True)
    for idx, item in enumerate(scored_rows, start=1):
        item.rank = idx
    return scored_rows


def _evaluate_profile_for_stage(
    stage_name: str,
    profile_name: str,
    results: Sequence[ProfileCandidateScore],
    annotations: Mapping[str, QualityAnnotation],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> ProfileEvaluationSummary:
    dataset_to_rows: Dict[str, List[Tuple[ProfileCandidateScore, QualityAnnotation]]] = {}
    for result in results:
        for dataset_id in result.dataset_scores.keys():
            annotation = annotations.get(_annotation_key(stage_name, result.candidate_id, dataset_id))
            if annotation is None:
                continue
            dataset_to_rows.setdefault(dataset_id, []).append((result, annotation))

    pairwise_values: List[float] = []
    top1_hits: List[float] = []
    mean_topk_scores: List[float] = []
    rank_values: List[float] = []
    label_scores: List[float] = []
    pair_count = 0
    labeled_count = 0
    for dataset_id, rows in dataset_to_rows.items():
        if not rows:
            continue
        labeled_count += len(rows)
        rows_sorted = sorted(rows, key=lambda item: item[0].rank)
        best_label = max(annotation.score for _, annotation in rows_sorted)
        top1_hits.append(1.0 if rows_sorted[0][1].score >= best_label else 0.0)
        top_slice = rows_sorted[:top_k]
        mean_topk_scores.append(float(np.mean([annotation.score for _, annotation in top_slice])))
        rank_values.extend([float(result.rank) for result, _ in rows_sorted])
        label_scores.extend([annotation.score for _, annotation in rows_sorted])
        for left_idx in range(len(rows_sorted)):
            left_result, left_annotation = rows_sorted[left_idx]
            for right_idx in range(left_idx + 1, len(rows_sorted)):
                right_result, right_annotation = rows_sorted[right_idx]
                if left_annotation.score == right_annotation.score:
                    continue
                pair_count += 1
                higher = left_annotation if left_annotation.score > right_annotation.score else right_annotation
                higher_rank = left_result.rank if left_annotation.score > right_annotation.score else right_result.rank
                lower_rank = right_result.rank if left_annotation.score > right_annotation.score else left_result.rank
                if higher_rank < lower_rank:
                    pairwise_values.append(1.0)
                elif higher_rank == lower_rank:
                    pairwise_values.append(0.5)
                else:
                    pairwise_values.append(0.0)

    return ProfileEvaluationSummary(
        stage=stage_name,
        profile_name=profile_name,
        pairwise_agreement=float(np.mean(pairwise_values)) if pairwise_values else None,
        top1_hit_rate=float(np.mean(top1_hits)) if top1_hits else None,
        mean_label_top_k=float(np.mean(mean_topk_scores)) if mean_topk_scores else None,
        spearman_correlation=_spearman_correlation(label_scores, [-rank for rank in rank_values]),
        num_labeled_items=labeled_count,
        num_pairwise_comparisons=pair_count,
        num_datasets_considered=len(dataset_to_rows),
    )


def _build_disagreements(
    stage_name: str,
    profile_name: str,
    results: Sequence[ProfileCandidateScore],
    annotations: Mapping[str, QualityAnnotation],
) -> List[DisagreementRecord]:
    rows_by_dataset: Dict[str, List[Tuple[ProfileCandidateScore, QualityAnnotation]]] = {}
    for result in results:
        for dataset_id in result.dataset_scores.keys():
            annotation = annotations.get(_annotation_key(stage_name, result.candidate_id, dataset_id))
            if annotation is None:
                continue
            rows_by_dataset.setdefault(dataset_id, []).append((result, annotation))

    disagreements: List[DisagreementRecord] = []
    for dataset_id, rows in rows_by_dataset.items():
        rows = sorted(rows, key=lambda item: item[0].rank)
        for left_idx, (left_result, left_annotation) in enumerate(rows):
            for right_idx in range(left_idx + 1, len(rows)):
                right_result, right_annotation = rows[right_idx]
                if left_annotation.score >= right_annotation.score:
                    continue
                disagreements.append(
                    DisagreementRecord(
                        stage=stage_name,
                        profile_name=profile_name,
                        dataset_id=dataset_id,
                        higher_candidate_id=right_result.candidate_id,
                        higher_label=right_annotation.label,
                        higher_rank=right_result.rank,
                        lower_candidate_id=left_result.candidate_id,
                        lower_label=left_annotation.label,
                        lower_rank=left_result.rank,
                        rank_gap=left_result.rank - right_result.rank,
                        label_gap=right_annotation.score - left_annotation.score,
                    )
                )
    disagreements.sort(key=lambda item: (item.label_gap, item.rank_gap), reverse=True)
    return disagreements


def _generate_weight_combinations(num_metrics: int, step: float) -> Iterable[Tuple[float, ...]]:
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("weight_step must evenly divide 1.0")

    def _recurse(slots: int, remaining: int) -> Iterable[Tuple[int, ...]]:
        if slots == 1:
            yield (remaining,)
            return
        for count in range(remaining + 1):
            for tail in _recurse(slots - 1, remaining - count):
                yield (count, *tail)

    for counts in _recurse(num_metrics, units):
        if sum(count > 0 for count in counts) == 0:
            continue
        yield tuple(count * step for count in counts)


def search_profile_combinations(
    explorer_model: SweepMetricExplorerModel,
    *,
    stage_name: str,
    metric_names: Sequence[str],
    weight_step: float = DEFAULT_SEARCH_WEIGHT_STEP,
) -> List[SearchResult]:
    """Search a nonnegative simplex grid for a better profile."""

    candidate_metrics = [name for name in metric_names if name in explorer_model.metric_registry]
    if not candidate_metrics or len(candidate_metrics) > 6:
        return []
    results: List[SearchResult] = []
    for weights in _generate_weight_combinations(len(candidate_metrics), weight_step):
        metric_weights = {
            name: float(weight)
            for name, weight in zip(candidate_metrics, weights)
            if weight > 0.0
        }
        profile = RankingProfileSpec(
            name=f"search_{len(results) + 1:04d}",
            label="Search",
            metric_weights=metric_weights,
        )
        scored = _score_profile_for_stage(
            stage_name,
            profile,
            explorer_model.base_model,
            explorer_model.raw_records,
            explorer_model.metric_registry,
            explorer_model.scoring_config,
        )
        summary = _evaluate_profile_for_stage(
            stage_name,
            profile.name,
            scored,
            explorer_model.annotations,
        )
        results.append(
            SearchResult(
                stage=stage_name,
                profile_name=profile.name,
                metric_weights=metric_weights,
                pairwise_agreement=summary.pairwise_agreement,
                top1_hit_rate=summary.top1_hit_rate,
                mean_label_top_k=summary.mean_label_top_k,
                spearman_correlation=summary.spearman_correlation,
            )
        )
    results.sort(
        key=lambda item: (
            -1.0 if item.pairwise_agreement is None else -float(item.pairwise_agreement),
            -1.0 if item.top1_hit_rate is None else -float(item.top1_hit_rate),
            -1.0 if item.mean_label_top_k is None else -float(item.mean_label_top_k),
            -1.0 if item.spearman_correlation is None else -float(item.spearman_correlation),
        )
    )
    return results


def load_sweep_metric_explorer_model(
    path: Path,
    *,
    metrics_config_path: Optional[Path] = None,
) -> SweepMetricExplorerModel:
    """Load sweep artifacts and recompute focus metrics from raw workspace bundles."""

    base_model = load_sweep_visualization_model(path)
    review_dir = base_model.experiment_root / "metric_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = review_dir / "annotations.json"
    annotations = _load_annotations(annotations_path)
    metric_registry = build_metric_registry()
    config, config_warnings = _resolve_metric_config(
        review_dir,
        metric_registry,
        base_model.scoring_weights,
        metrics_config_path,
    )
    enabled_registry = {
        name: metric_registry[name]
        for name in config.enabled_metrics
        if name in metric_registry
    }
    raw_records, record_warnings, cache_path = _scan_raw_records(
        base_model,
        review_dir,
        enabled_registry,
    )
    scoring_config = _scoring_config(base_model)

    profile_results: Dict[Tuple[str, str], List[ProfileCandidateScore]] = {}
    evaluations: List[ProfileEvaluationSummary] = []
    disagreements: List[DisagreementRecord] = []
    for stage_name in base_model.stage_order:
        for profile in config.profiles:
            scored = _score_profile_for_stage(
                stage_name,
                profile,
                base_model,
                raw_records,
                enabled_registry,
                scoring_config,
            )
            if not scored:
                continue
            profile_results[(stage_name, profile.name)] = scored
            summary = _evaluate_profile_for_stage(
                stage_name,
                profile.name,
                scored,
                annotations,
            )
            evaluations.append(summary)
            disagreements.extend(
                _build_disagreements(stage_name, profile.name, scored, annotations)
            )

    search_results: List[SearchResult] = []
    if (
        config.search_defaults.metric_subset
        and len(config.search_defaults.metric_subset) <= config.search_defaults.max_metrics
        and annotations
    ):
        for stage_name in base_model.stage_order:
            search_results.extend(
                search_profile_combinations(
                    SweepMetricExplorerModel(
                        base_model=base_model,
                        review_dir=review_dir,
                        cache_path=cache_path,
                        annotations_path=annotations_path,
                        config=config,
                        metric_registry=enabled_registry,
                        raw_records=raw_records,
                        profile_results=profile_results,
                        evaluations=evaluations,
                        disagreements=disagreements,
                        annotations=annotations,
                        search_results=[],
                        warnings=[],
                        scoring_config=scoring_config,
                    ),
                    stage_name=stage_name,
                    metric_names=config.search_defaults.metric_subset,
                    weight_step=config.search_defaults.weight_step,
                )
            )

    return SweepMetricExplorerModel(
        base_model=base_model,
        review_dir=review_dir,
        cache_path=cache_path,
        annotations_path=annotations_path,
        config=config,
        metric_registry=enabled_registry,
        raw_records=raw_records,
        profile_results=profile_results,
        evaluations=evaluations,
        disagreements=disagreements,
        annotations=annotations,
        search_results=search_results,
        warnings=config_warnings + record_warnings,
        scoring_config=scoring_config,
    )


def _raw_metric_fieldnames(model: SweepMetricExplorerModel) -> List[str]:
    metric_names = sorted(model.metric_registry.keys())
    return [
        "stage",
        "candidate_id",
        "dataset_id",
        "strategy",
        "family",
        "alpha",
        "formulation_id",
        "saved_score",
        "saved_mean_dataset_score",
        "preview_path",
        "workspace_path",
        "warnings_json",
        *metric_names,
    ]


def _profile_result_rows(model: SweepMetricExplorerModel) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (stage_name, profile_name), results in sorted(model.profile_results.items()):
        for result in results:
            rows.append(
                {
                    "stage": stage_name,
                    "profile_name": profile_name,
                    "candidate_id": result.candidate_id,
                    "rank": result.rank,
                    "score": result.score,
                    "mean_dataset_score": result.mean_dataset_score,
                    "std_dataset_score": result.std_dataset_score,
                    "missing_datasets": "|".join(result.missing_datasets),
                    "dataset_scores_json": _compact_json(result.dataset_scores),
                    "contributions_json": _compact_json(result.contributions),
                    "strategy": result.strategy,
                    "family": result.family,
                    "alpha": result.alpha,
                }
            )
    return rows


def _evaluation_rows(model: SweepMetricExplorerModel) -> List[Dict[str, object]]:
    return [
        {
            "stage": item.stage,
            "profile_name": item.profile_name,
            "pairwise_agreement": item.pairwise_agreement,
            "top1_hit_rate": item.top1_hit_rate,
            "mean_label_top_k": item.mean_label_top_k,
            "spearman_correlation": item.spearman_correlation,
            "num_labeled_items": item.num_labeled_items,
            "num_pairwise_comparisons": item.num_pairwise_comparisons,
            "num_datasets_considered": item.num_datasets_considered,
        }
        for item in model.evaluations
    ]


def _disagreement_rows(model: SweepMetricExplorerModel) -> List[Dict[str, object]]:
    return [
        {
            "stage": item.stage,
            "profile_name": item.profile_name,
            "dataset_id": item.dataset_id,
            "higher_candidate_id": item.higher_candidate_id,
            "higher_label": item.higher_label,
            "higher_rank": item.higher_rank,
            "lower_candidate_id": item.lower_candidate_id,
            "lower_label": item.lower_label,
            "lower_rank": item.lower_rank,
            "rank_gap": item.rank_gap,
            "label_gap": item.label_gap,
        }
        for item in model.disagreements
    ]


def _search_rows(model: SweepMetricExplorerModel) -> List[Dict[str, object]]:
    return [
        {
            "stage": item.stage,
            "profile_name": item.profile_name,
            "pairwise_agreement": item.pairwise_agreement,
            "top1_hit_rate": item.top1_hit_rate,
            "mean_label_top_k": item.mean_label_top_k,
            "spearman_correlation": item.spearman_correlation,
            "metric_weights_json": _compact_json(item.metric_weights),
        }
        for item in model.search_results
    ]


def export_metric_report(model: SweepMetricExplorerModel, output_dir: Path) -> Path:
    """Export the current model to CSV/JSON/HTML report artifacts."""

    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    candidate_metrics_path = export_dir / "candidate_metrics.csv"
    profile_results_path = export_dir / "profile_results.csv"
    search_results_path = export_dir / "search_results.csv"
    disagreements_path = export_dir / "disagreements.csv"
    report_json_path = export_dir / "report.json"
    html_path = export_dir / "index.html"

    write_csv_atomic(
        candidate_metrics_path,
        _raw_metric_fieldnames(model),
        (
            {
                **{
                    "stage": record.stage,
                    "candidate_id": record.candidate_id,
                    "dataset_id": record.dataset_id,
                    "strategy": record.strategy,
                    "family": record.family,
                    "alpha": record.alpha,
                    "formulation_id": record.formulation_id,
                    "saved_score": record.saved_score,
                    "saved_mean_dataset_score": record.saved_mean_dataset_score,
                    "preview_path": record.preview_path,
                    "workspace_path": record.workspace_path,
                    "warnings_json": _compact_json(record.warnings),
                },
                **{name: record.metrics.get(name) for name in sorted(model.metric_registry.keys())},
            }
            for record in model.raw_records
        ),
        create_backup=True,
    )
    write_csv_atomic(
        profile_results_path,
        [
            "stage",
            "profile_name",
            "candidate_id",
            "rank",
            "score",
            "mean_dataset_score",
            "std_dataset_score",
            "missing_datasets",
            "dataset_scores_json",
            "contributions_json",
            "strategy",
            "family",
            "alpha",
        ],
        _profile_result_rows(model),
        create_backup=True,
    )
    write_csv_atomic(
        search_results_path,
        [
            "stage",
            "profile_name",
            "pairwise_agreement",
            "top1_hit_rate",
            "mean_label_top_k",
            "spearman_correlation",
            "metric_weights_json",
        ],
        _search_rows(model),
        create_backup=True,
    )
    write_csv_atomic(
        disagreements_path,
        [
            "stage",
            "profile_name",
            "dataset_id",
            "higher_candidate_id",
            "higher_label",
            "higher_rank",
            "lower_candidate_id",
            "lower_label",
            "lower_rank",
            "rank_gap",
            "label_gap",
        ],
        _disagreement_rows(model),
        create_backup=True,
    )

    report_payload = {
        "generated_at": _utc_now_iso(),
        "registry_version": REGISTRY_VERSION,
        "review_dir": str(model.review_dir),
        "config": {
            "enabled_metrics": model.config.enabled_metrics,
            "default_profile": model.config.default_profile,
            "profiles": [
                {
                    "name": profile.name,
                    "label": profile.label,
                    "metric_weights": profile.metric_weights,
                    "description": profile.description,
                    "built_in": profile.built_in,
                }
                for profile in model.config.profiles
            ],
            "search_defaults": asdict(model.config.search_defaults),
            "source_path": model.config.source_path,
        },
        "scoring_config": model.scoring_config,
        "warnings": model.warnings,
        "evaluations": _evaluation_rows(model),
        "files": {
            "candidate_metrics_csv": candidate_metrics_path.name,
            "profile_results_csv": profile_results_path.name,
            "search_results_csv": search_results_path.name,
            "disagreements_csv": disagreements_path.name,
            "index_html": html_path.name,
        },
    }
    write_json_atomic(report_json_path, report_payload, indent=2, sort_keys=False, create_backup=True)

    top_rows = []
    default_profile = model.config.default_profile
    for stage_name in model.base_model.stage_order:
        results = model.profile_results.get((stage_name, default_profile), [])
        top_rows.extend(results[: min(5, len(results))])
    top_table = "".join(
        f"<tr><td>{html.escape(row.stage)}</td><td>{html.escape(row.candidate_id)}</td>"
        f"<td>{row.rank}</td><td>{row.score:.4f}</td><td>{html.escape(row.profile_name)}</td></tr>"
        for row in top_rows
    ) or "<tr><td colspan='5'>No results</td></tr>"
    evaluation_rows = "".join(
        f"<tr><td>{html.escape(item.stage)}</td><td>{html.escape(item.profile_name)}</td>"
        f"<td>{'n/a' if item.pairwise_agreement is None else f'{item.pairwise_agreement:.3f}'}</td>"
        f"<td>{'n/a' if item.top1_hit_rate is None else f'{item.top1_hit_rate:.3f}'}</td>"
        f"<td>{'n/a' if item.mean_label_top_k is None else f'{item.mean_label_top_k:.3f}'}</td>"
        f"<td>{'n/a' if item.spearman_correlation is None else f'{item.spearman_correlation:.3f}'}</td></tr>"
        for item in model.evaluations
    ) or "<tr><td colspan='6'>No evaluation rows</td></tr>"
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Metric Explorer Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Raw Metric Explorer Report</h1>
  <p>Generated at {html.escape(report_payload["generated_at"])}</p>
  <h2>Summary</h2>
  <pre>{html.escape(json.dumps(report_payload["config"], indent=2))}</pre>
  <h2>Top Candidates ({html.escape(default_profile)})</h2>
  <table>
    <thead><tr><th>Stage</th><th>Candidate</th><th>Rank</th><th>Score</th><th>Profile</th></tr></thead>
    <tbody>{top_table}</tbody>
  </table>
  <h2>Evaluations</h2>
  <table>
    <thead><tr><th>Stage</th><th>Profile</th><th>Pairwise</th><th>Top-1</th><th>Mean Label Top-K</th><th>Spearman</th></tr></thead>
    <tbody>{evaluation_rows}</tbody>
  </table>
  <h2>Warnings</h2>
  <pre>{html.escape(chr(10).join(model.warnings) if model.warnings else 'None')}</pre>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return export_dir


def _figure_message(text: str) -> Figure:
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    return fig


def create_channel_comparison_figure(
    model: SweepMetricExplorerModel,
    records: Sequence[RawMetricRecord],
    *,
    channel_name: str,
    figure_scale: float = 1.0,
) -> Figure:
    """Render preview + selected channel for one or more records."""

    if not records:
        return _figure_message("No selected records.")
    safe_scale = max(0.75, float(figure_scale))
    per_column_width = 5.8 * safe_scale
    per_row_height = 4.2 * safe_scale
    fig = Figure(
        figsize=(
            max(9.5, per_column_width * max(1, len(records))),
            max(7.0, per_row_height * 2.0),
        )
    )
    axes = fig.subplots(2, len(records), squeeze=False)
    for col_idx, record in enumerate(records):
        preview_ax = axes[0, col_idx]
        channel_ax = axes[1, col_idx]
        preview = None
        if record.preview_path and Path(record.preview_path).exists():
            preview = np.asarray(Image.open(record.preview_path).convert("L"), dtype=np.float64)
        avg_image = guide_image = None
        if record.workspace_path:
            avg_image, guide_image, _ = compute_image_channels(Path(record.workspace_path))
        selected = avg_image if channel_name == "avg_image" else guide_image
        for ax, title, array, cmap in (
            (
                preview_ax,
                f"{record.candidate_id} / {record.dataset_id}\nPreview",
                preview,
                "gray",
            ),
            (
                channel_ax,
                "Avg Image" if channel_name == "avg_image" else "Guide LoG",
                selected,
                "gray" if channel_name == "avg_image" else "viridis",
            ),
        ):
            if isinstance(array, np.ndarray) and array.ndim == 2 and array.size:
                image = ax.imshow(array, cmap=cmap, aspect="equal")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                ax.axis("off")
            ax.set_title(title)
    fig.tight_layout()
    return fig


def _format_record_metric_lines(model: SweepMetricExplorerModel, record: RawMetricRecord) -> str:
    label = model.annotations.get(_annotation_key(record.stage, record.candidate_id, record.dataset_id))
    lines = [
        f"stage: {record.stage}",
        f"candidate: {record.candidate_id}",
        f"dataset: {record.dataset_id}",
        f"strategy: {record.strategy}",
        f"family: {record.family or 'n/a'}",
        f"alpha: {'n/a' if record.alpha is None else f'{record.alpha:g}'}",
        f"label: {label.label if label is not None else 'unlabeled'}",
        "",
    ]
    for metric_name in sorted(model.metric_registry.keys()):
        value = record.metrics.get(metric_name)
        metric_label = model.metric_registry[metric_name].label
        lines.append(
            f"{metric_label}: {'n/a' if value is None else f'{value:.6f}'}"
        )
    if record.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"- {item}" for item in record.warnings)
    return "\n".join(lines)


def _format_pairwise_metric_delta_lines(
    model: SweepMetricExplorerModel,
    record_a: RawMetricRecord,
    record_b: RawMetricRecord,
) -> List[str]:
    lines = [
        "pairwise metric deltas (A - B):",
        f"A: {_record_key_to_text((record_a.stage, record_a.candidate_id, record_a.dataset_id))}",
        f"B: {_record_key_to_text((record_b.stage, record_b.candidate_id, record_b.dataset_id))}",
        "",
    ]
    for row in build_comparison_metric_rows(model, record_a, record_b):
        lines.append(
            f"{row.metric_label}: "
            f"A={row.value_a:.6f} | B={row.value_b:.6f} | delta={row.delta:+.6f} | winner={row.winner}"
        )
    return lines


def build_comparison_metric_rows(
    model: SweepMetricExplorerModel,
    record_a: RawMetricRecord,
    record_b: RawMetricRecord,
) -> List[ComparisonMetricRow]:
    """Build structured A/B metric rows for the comparison table."""

    rows: List[ComparisonMetricRow] = []
    for metric_name, descriptor in model.metric_registry.items():
        value_a = record_a.metrics.get(metric_name)
        value_b = record_b.metrics.get(metric_name)
        if value_a is None or value_b is None:
            continue
        delta = float(value_a) - float(value_b)
        if abs(delta) <= 1e-12:
            winner = "="
        elif descriptor.direction == "min":
            winner = "A" if delta < 0 else "B"
        else:
            winner = "A" if delta > 0 else "B"
        rows.append(
            ComparisonMetricRow(
                metric_name=metric_name,
                metric_label=descriptor.label,
                category=descriptor.category,
                direction=descriptor.direction,
                value_a=float(value_a),
                value_b=float(value_b),
                delta=delta,
                winner=winner,
            )
        )
    return rows


def _resolve_comparison_records(
    record_lookup: Mapping[ComparisonRecordKey, RawMetricRecord],
    selected_rows: Sequence[ProfileCandidateScore],
    *,
    dataset_id: str,
    comparison_mode: str,
    slot_a: Optional[ComparisonRecordKey],
    slot_b: Optional[ComparisonRecordKey],
) -> List[RawMetricRecord]:
    if comparison_mode == "A vs B":
        ordered_keys = [slot_a, slot_b]
        seen: set[ComparisonRecordKey] = set()
        records: List[RawMetricRecord] = []
        for key in ordered_keys:
            if key is None or key in seen or key not in record_lookup:
                continue
            seen.add(key)
            records.append(record_lookup[key])
        return records

    records = [
        record_lookup[(row.stage, row.candidate_id, dataset_id)]
        for row in selected_rows
        if (row.stage, row.candidate_id, dataset_id) in record_lookup
    ]
    return records


def _format_evaluation_text(model: SweepMetricExplorerModel, stage_name: str) -> str:
    rows = [row for row in model.evaluations if row.stage == stage_name]
    if not rows:
        return "No evaluation data for this stage."
    lines = [f"stage: {stage_name}", ""]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    row.profile_name,
                    f"pairwise={'n/a' if row.pairwise_agreement is None else f'{row.pairwise_agreement:.3f}'}",
                    f"top1={'n/a' if row.top1_hit_rate is None else f'{row.top1_hit_rate:.3f}'}",
                    f"top{DEFAULT_TOP_K}={'n/a' if row.mean_label_top_k is None else f'{row.mean_label_top_k:.3f}'}",
                    f"spearman={'n/a' if row.spearman_correlation is None else f'{row.spearman_correlation:.3f}'}",
                    f"labels={row.num_labeled_items}",
                ]
            )
        )
    return "\n".join(lines)


def _format_search_results_text(results: Sequence[SearchResult]) -> str:
    if not results:
        return "No search results."
    lines = []
    for idx, result in enumerate(results[:20], start=1):
        lines.append(
            f"{idx:02d}. pairwise={'n/a' if result.pairwise_agreement is None else f'{result.pairwise_agreement:.3f}'} | "
            f"top1={'n/a' if result.top1_hit_rate is None else f'{result.top1_hit_rate:.3f}'} | "
            f"spearman={'n/a' if result.spearman_correlation is None else f'{result.spearman_correlation:.3f}'} | "
            f"weights={_compact_json(result.metric_weights)}"
        )
    return "\n".join(lines)


class _ComparisonImageInspector:
    """Interactive single-run image inspector used inside the comparison workspace."""

    VIEW_OPTIONS = ("preview", "avg_image", "guide_log")
    VIEW_LABELS = {
        "preview": "Preview",
        "avg_image": "Avg Image",
        "guide_log": "Guide LoG",
    }
    CONTRAST_OPTIONS = ("auto", "p2-p98", "p5-p95")
    CONTRAST_LABELS = {
        "auto": "Auto",
        "p2-p98": "P2-P98",
        "p5-p95": "P5-P95",
    }

    def __init__(
        self,
        app: "SweepMetricExplorerApp",
        parent: Any,
        *,
        slot_name: str,
        on_view_changed: Optional[Any] = None,
        on_limits_changed: Optional[Any] = None,
        on_open_large: Optional[Any] = None,
        allow_open_large: bool = True,
    ):
        self.app = app
        self.tk = app.tk
        self.ttk = app.ttk
        self.slot_name = slot_name
        self.on_view_changed = on_view_changed
        self.on_limits_changed = on_limits_changed
        self.on_open_large = on_open_large
        self.allow_open_large = allow_open_large

        self.frame = self.ttk.Frame(parent)
        self.title_var = self.tk.StringVar(value=f"Run {slot_name}")
        self.meta_var = self.tk.StringVar(value=f"Set slot {slot_name} from the leaderboard.")
        self.warning_var = self.tk.StringVar(value="")
        self.view_var = self.tk.StringVar(value="preview")
        self.contrast_var = self.tk.StringVar(value="auto")

        self._record: Optional[RawMetricRecord] = None
        self._bundle: Optional[ComparisonDisplayBundle] = None
        self._record_key: Optional[ComparisonRecordKey] = None
        self._suspend_limit_events = False
        self._last_rendered_shape: Optional[Tuple[int, int]] = None

        header = self.ttk.Frame(self.frame)
        header.pack(fill="x", padx=6, pady=(6, 0))
        self.ttk.Label(header, textvariable=self.title_var).pack(anchor="w")
        self.ttk.Label(header, textvariable=self.meta_var, justify="left", wraplength=700).pack(
            anchor="w", pady=(2, 0)
        )
        self.ttk.Label(header, textvariable=self.warning_var, justify="left", wraplength=700).pack(
            anchor="w", pady=(2, 0)
        )

        controls = self.ttk.Frame(self.frame)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="View").pack(side="left")
        view_combo = self.ttk.Combobox(
            controls,
            textvariable=self.view_var,
            state="readonly",
            width=12,
            values=list(self.VIEW_OPTIONS),
        )
        view_combo.pack(side="left", padx=4)
        view_combo.bind("<<ComboboxSelected>>", self._handle_view_change)

        self.ttk.Label(controls, text="Contrast").pack(side="left", padx=(8, 0))
        contrast_combo = self.ttk.Combobox(
            controls,
            textvariable=self.contrast_var,
            state="readonly",
            width=10,
            values=list(self.CONTRAST_OPTIONS),
        )
        contrast_combo.pack(side="left", padx=4)
        contrast_combo.bind("<<ComboboxSelected>>", lambda _evt: self.render(reset_view=False))

        self.ttk.Button(controls, text="Fit", command=self.fit_view).pack(side="left", padx=(10, 3))
        self.ttk.Button(controls, text="1:1", command=self.pixel_view).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Reset", command=self.reset_view_state).pack(side="left", padx=3)
        if self.allow_open_large and self.on_open_large is not None:
            self.ttk.Button(controls, text="Open Large", command=self._open_large).pack(side="left", padx=3)

        self.toolbar_frame = self.ttk.Frame(self.frame)
        self.toolbar_frame.pack(fill="x", padx=6, pady=(0, 2))
        self.canvas_frame = self.ttk.Frame(self.frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.figure = Figure(figsize=(6.4, 6.2))
        self.axis = self.figure.add_subplot(111)
        self.canvas = self.app.FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        try:
            self.toolbar = self.app.NavigationToolbar2Tk(
                self.canvas,
                self.toolbar_frame,
                pack_toolbar=False,
            )
        except TypeError:
            self.toolbar = self.app.NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        if hasattr(self.toolbar, "pack"):
            self.toolbar.pack(fill="x")
        self.axis.callbacks.connect("xlim_changed", lambda _axes: self._notify_limits_changed())
        self.axis.callbacks.connect("ylim_changed", lambda _axes: self._notify_limits_changed())
        self._render_placeholder(f"Run {slot_name} is not set.")

    @property
    def record(self) -> Optional[RawMetricRecord]:
        return self._record

    @property
    def record_key(self) -> Optional[ComparisonRecordKey]:
        return self._record_key

    def current_view(self) -> str:
        return self.view_var.get()

    def current_contrast(self) -> str:
        return self.contrast_var.get()

    def current_image_shape(self) -> Optional[Tuple[int, int]]:
        image = self.current_image()
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            return None
        return int(image.shape[0]), int(image.shape[1])

    def current_image(self) -> Optional[np.ndarray]:
        if self._bundle is None:
            return None
        image = self._bundle.get(self.view_var.get())
        if isinstance(image, np.ndarray) and image.ndim == 2 and image.size:
            return np.asarray(image, dtype=np.float64)
        return None

    def has_image(self) -> bool:
        return self.current_image() is not None

    def set_view(self, view_name: str, *, emit_event: bool = False, reset_view: bool = False) -> None:
        if view_name not in self.VIEW_OPTIONS:
            return
        changed = self.view_var.get() != view_name
        self.view_var.set(view_name)
        self.render(reset_view=reset_view or changed)
        if emit_event and changed and self.on_view_changed is not None:
            self.on_view_changed(self, view_name)

    def set_contrast(self, contrast_mode: str, *, redraw: bool = True) -> None:
        if contrast_mode not in self.CONTRAST_OPTIONS:
            return
        self.contrast_var.set(contrast_mode)
        if redraw:
            self.render(reset_view=False)

    def set_record(
        self,
        record: Optional[RawMetricRecord],
        bundle: Optional[ComparisonDisplayBundle],
        *,
        annotation_label: str = "",
    ) -> None:
        previous_key = self._record_key
        self._record = record
        self._bundle = bundle
        self._record_key = _metric_row_key(record) if record is not None else None
        if record is None:
            self.title_var.set(f"Run {self.slot_name}")
            self.meta_var.set(f"Set slot {self.slot_name} from the leaderboard.")
            self.warning_var.set("")
            self._render_placeholder(f"Run {self.slot_name} is not set.")
            return

        self.title_var.set(
            f"Run {self.slot_name}: {record.candidate_id} | {record.dataset_id}"
        )
        meta_parts = [
            f"stage={record.stage}",
            f"strategy={record.strategy}",
        ]
        if record.family:
            meta_parts.append(f"family={record.family}")
        if record.alpha is not None:
            meta_parts.append(f"alpha={record.alpha:g}")
        meta_parts.append(f"label={annotation_label or 'unlabeled'}")
        self.meta_var.set(" | ".join(meta_parts))
        warning_text = ""
        if bundle is not None and bundle.warnings:
            warning_text = f"Warnings: {' | '.join(bundle.warnings[:3])}"
        self.warning_var.set(warning_text)
        self.render(reset_view=previous_key != self._record_key)

    def fit_view(self) -> None:
        shape = self.current_image_shape()
        if shape is None:
            return
        height, width = shape
        self.apply_axes_limits((-0.5, width - 0.5), (height - 0.5, -0.5))

    def pixel_view(self) -> None:
        shape = self.current_image_shape()
        if shape is None:
            return
        self.canvas.draw()
        bbox = self.axis.get_window_extent()
        visible_width = max(1.0, min(float(shape[1]), float(bbox.width)))
        visible_height = max(1.0, min(float(shape[0]), float(bbox.height)))
        xlim, ylim = self.get_axes_limits()
        if xlim is None or ylim is None:
            self.fit_view()
            xlim, ylim = self.get_axes_limits()
            if xlim is None or ylim is None:
                return
        center_x = 0.5 * (xlim[0] + xlim[1])
        center_y = 0.5 * (ylim[0] + ylim[1])
        full_x = self._bounded_limits(center_x, visible_width, -0.5, shape[1] - 0.5)
        full_y = self._bounded_limits(center_y, visible_height, -0.5, shape[0] - 0.5)
        if ylim[1] < ylim[0]:
            full_y = (full_y[1], full_y[0])
        self.apply_axes_limits(full_x, full_y)

    def reset_view_state(self) -> None:
        self.contrast_var.set("auto")
        self.fit_view()
        self.render(reset_view=False)

    def get_axes_limits(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        if not self.axis.images:
            return None, None
        xlim = cast(Tuple[float, float], tuple(float(value) for value in self.axis.get_xlim()))
        ylim = cast(Tuple[float, float], tuple(float(value) for value in self.axis.get_ylim()))
        return xlim, ylim

    def apply_axes_limits(
        self,
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
    ) -> None:
        self._suspend_limit_events = True
        try:
            self.axis.set_xlim(*x_limits)
            self.axis.set_ylim(*y_limits)
        finally:
            self._suspend_limit_events = False
        self.canvas.draw_idle()

    def clone_view_state_from(self, other: "_ComparisonImageInspector") -> None:
        self.set_view(other.current_view(), emit_event=False, reset_view=True)
        self.set_contrast(other.current_contrast(), redraw=True)
        xlim, ylim = other.get_axes_limits()
        if xlim is not None and ylim is not None and self.has_image():
            self.apply_axes_limits(xlim, ylim)

    def render(self, *, reset_view: bool) -> None:
        image = self.current_image()
        previous_shape = self._last_rendered_shape
        previous_limits = self.get_axes_limits()
        if image is None:
            placeholder = self.VIEW_LABELS.get(self.view_var.get(), self.view_var.get())
            self._render_placeholder(f"{placeholder} unavailable.")
            return

        self.axis.clear()
        vmin, vmax = self._contrast_limits(image)
        cmap = "viridis" if self.view_var.get() == "guide_log" else "gray"
        self.axis.imshow(
            image,
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_title(self.VIEW_LABELS.get(self.view_var.get(), self.view_var.get()))
        if reset_view or previous_limits[0] is None or previous_shape != image.shape:
            self.fit_view()
        else:
            self.apply_axes_limits(cast(Tuple[float, float], previous_limits[0]), cast(Tuple[float, float], previous_limits[1]))
        self._last_rendered_shape = (int(image.shape[0]), int(image.shape[1]))
        self.canvas.draw_idle()

    def _render_placeholder(self, text: str) -> None:
        self.axis.clear()
        self.axis.text(0.5, 0.5, text, ha="center", va="center")
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_title(self.VIEW_LABELS.get(self.view_var.get(), self.view_var.get()))
        self._last_rendered_shape = None
        self.canvas.draw_idle()

    def _handle_view_change(self, _event: Optional[object] = None) -> None:
        self.render(reset_view=False)
        if self.on_view_changed is not None:
            self.on_view_changed(self, self.view_var.get())

    def _open_large(self) -> None:
        if self.on_open_large is not None:
            self.on_open_large(self.slot_name)

    def _notify_limits_changed(self) -> None:
        if self._suspend_limit_events or self.on_limits_changed is None or not self.axis.images:
            return
        self.on_limits_changed(self)

    def _contrast_limits(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        mode = self.contrast_var.get()
        if mode == "p2-p98":
            low, high = np.percentile(image, [2.0, 98.0])
        elif mode == "p5-p95":
            low, high = np.percentile(image, [5.0, 95.0])
        else:
            return None, None
        low_value = float(low)
        high_value = float(high)
        if not math.isfinite(low_value) or not math.isfinite(high_value) or high_value <= low_value:
            return None, None
        return low_value, high_value

    @staticmethod
    def _bounded_limits(center: float, span: float, lower: float, upper: float) -> Tuple[float, float]:
        total_width = upper - lower
        if span >= total_width:
            return lower, upper
        half = span / 2.0
        start = center - half
        end = center + half
        if start < lower:
            end += lower - start
            start = lower
        if end > upper:
            start -= end - upper
            end = upper
        return start, end


class SweepMetricExplorerApp:
    """Tkinter GUI for loading, re-scoring, annotating, and exporting raw metrics."""

    def __init__(self, root: Any, initial_path: Optional[Path] = None, metrics_config_path: Optional[Path] = None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = __import__("tkinter.filedialog", fromlist=["filedialog"])
        self.messagebox = __import__("tkinter.messagebox", fromlist=["messagebox"])
        backend_tkagg = __import__(
            "matplotlib.backends.backend_tkagg",
            fromlist=["FigureCanvasTkAgg", "NavigationToolbar2Tk"],
        )
        self.FigureCanvasTkAgg = backend_tkagg.FigureCanvasTkAgg
        self.NavigationToolbar2Tk = backend_tkagg.NavigationToolbar2Tk

        self.root.title("Sweep Raw Metric Explorer")
        self.root.geometry("1440x980")
        self.model: Optional[SweepMetricExplorerModel] = None
        self.metrics_config_path = metrics_config_path
        self._comparison_image_cache: Dict[ComparisonRecordKey, ComparisonDisplayBundle] = {}
        self._comparison_popouts: Dict[str, Dict[str, Any]] = {}
        self._comparison_sync_in_progress = False
        self._comparison_view_sync_in_progress = False

        self.path_var = tk.StringVar(value=str(initial_path) if initial_path else "")
        self.status_var = tk.StringVar(value="Idle")
        self.stage_var = tk.StringVar(value="")
        self.profile_var = tk.StringVar(value="")
        self.dataset_var = tk.StringVar(value="")
        self.comparison_sync_var = tk.BooleanVar(value=True)
        self.comparison_lock_views_var = tk.BooleanVar(value=True)
        self.comparison_layout_var = tk.StringVar(value="Horizontal")
        self.comparison_summary_var = tk.StringVar(value="Set A and B from the leaderboard to begin.")
        self.comparison_slot_a_var = tk.StringVar(value="A: not set")
        self.comparison_slot_b_var = tk.StringVar(value="B: not set")
        self.search_stage_var = tk.StringVar(value="")
        self.search_profile_name_var = tk.StringVar(value="custom_profile")
        self.search_step_var = tk.StringVar(value=f"{DEFAULT_SEARCH_WEIGHT_STEP:g}")
        self.search_max_metrics_var = tk.StringVar(value="6")
        self.weight_json_var = tk.StringVar(value="{}")
        self._comparison_slot_a_key: Optional[ComparisonRecordKey] = None
        self._comparison_slot_b_key: Optional[ComparisonRecordKey] = None

        self._leaderboard_rows: List[ProfileCandidateScore] = []
        self._leaderboard_row_lookup: Dict[str, ProfileCandidateScore] = {}
        self._search_results: List[SearchResult] = []

        self._build_ui()
        if initial_path is not None:
            self.load_from_path(initial_path)

    def _build_ui(self) -> None:
        top = self.ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        self.ttk.Label(top, text="Sweep folder").pack(side="left")
        entry = self.ttk.Entry(top, textvariable=self.path_var, width=90)
        entry.pack(side="left", padx=6, fill="x", expand=True)
        self.ttk.Button(top, text="Browse...", command=self._on_browse).pack(side="left", padx=3)
        self.ttk.Button(top, text="Load", command=self._on_load).pack(side="left", padx=3)
        self.ttk.Button(top, text="Reload", command=self._on_reload).pack(side="left", padx=3)
        self.ttk.Button(top, text="Export...", command=self._on_export_dialog).pack(side="left", padx=3)
        self.ttk.Button(top, text="Open Folder", command=self._on_open_folder).pack(side="left", padx=3)
        self.ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

        self.notebook = self.ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

        self.leaderboard_tab = self.ttk.Frame(self.notebook)
        self.comparison_tab = self.ttk.Frame(self.notebook)
        self.evaluation_tab = self.ttk.Frame(self.notebook)
        self.search_tab = self.ttk.Frame(self.notebook)

        self.notebook.add(self.leaderboard_tab, text="Leaderboard")
        self.notebook.add(self.comparison_tab, text="Comparison")
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        self.notebook.add(self.search_tab, text="Profiles / Search")

        self._build_leaderboard_tab()
        self._build_comparison_tab()
        self._build_evaluation_tab()
        self._build_search_tab()

    def _build_leaderboard_tab(self) -> None:
        controls = self.ttk.Frame(self.leaderboard_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="Stage").pack(side="left")
        self.stage_combo = self.ttk.Combobox(controls, textvariable=self.stage_var, state="readonly", width=16)
        self.stage_combo.pack(side="left", padx=4)
        self.stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_leaderboard())
        self.ttk.Label(controls, text="Profile").pack(side="left", padx=(8, 0))
        self.profile_combo = self.ttk.Combobox(controls, textvariable=self.profile_var, state="readonly", width=22)
        self.profile_combo.pack(side="left", padx=4)
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_leaderboard())
        self.ttk.Label(controls, text="Dataset").pack(side="left", padx=(8, 0))
        self.dataset_combo = self.ttk.Combobox(controls, textvariable=self.dataset_var, state="readonly", width=12)
        self.dataset_combo.pack(side="left", padx=4)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_leaderboard())
        self.ttk.Button(controls, text="Label Top", command=lambda: self._set_labels_for_selection("top")).pack(side="left", padx=(12, 3))
        self.ttk.Button(controls, text="Label Good", command=lambda: self._set_labels_for_selection("good")).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Label Bad", command=lambda: self._set_labels_for_selection("bad")).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Clear Label", command=self._clear_labels_for_selection).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Compare A", command=lambda: self._set_comparison_slot("A")).pack(side="left", padx=(12, 3))
        self.ttk.Button(controls, text="Compare B", command=lambda: self._set_comparison_slot("B")).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Compare 2", command=self._assign_comparison_slots_from_selection).pack(side="left", padx=3)

        split = self.ttk.Panedwindow(self.leaderboard_tab, orient="horizontal")
        split.pack(fill="both", expand=True, padx=6, pady=6)
        left = self.ttk.Frame(split)
        right = self.ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=2)

        columns = ("rank", "candidate", "score", "delta", "saved_rank", "strategy", "label")
        self.leaderboard_tree = self.ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        headers = {
            "rank": "Rank",
            "candidate": "Candidate",
            "score": "Score",
            "delta": "Rank Δ",
            "saved_rank": "Saved Rank",
            "strategy": "Strategy",
            "label": "Label",
        }
        widths = {"rank": 60, "candidate": 140, "score": 100, "delta": 70, "saved_rank": 90, "strategy": 100, "label": 90}
        for column in columns:
            self.leaderboard_tree.heading(column, text=headers[column])
            self.leaderboard_tree.column(column, width=widths[column], anchor="center")
        tree_scroll = self.ttk.Scrollbar(left, orient="vertical", command=self.leaderboard_tree.yview)
        self.leaderboard_tree.configure(yscrollcommand=tree_scroll.set)
        self.leaderboard_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.leaderboard_tree.bind("<<TreeviewSelect>>", lambda _evt: self._on_leaderboard_selection_changed())

        self.leaderboard_detail = self.tk.Text(right, wrap="word")
        self.leaderboard_detail.pack(fill="both", expand=True)

    def _build_comparison_tab(self) -> None:
        controls = self.ttk.Frame(self.comparison_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Button(controls, text="Set A", command=lambda: self._set_comparison_slot("A")).pack(
            side="left", padx=(0, 3)
        )
        self.ttk.Button(controls, text="Set B", command=lambda: self._set_comparison_slot("B")).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Use 2 Selected", command=self._assign_comparison_slots_from_selection).pack(
            side="left", padx=3
        )
        self.ttk.Button(controls, text="Swap", command=self._swap_comparison_slots).pack(side="left", padx=3)
        self.ttk.Button(controls, text="Clear Slots", command=self._clear_comparison_slots).pack(side="left", padx=3)
        self.ttk.Label(controls, text="Layout").pack(side="left", padx=(12, 0))
        self.comparison_layout_combo = self.ttk.Combobox(
            controls,
            textvariable=self.comparison_layout_var,
            state="readonly",
            width=12,
            values=["Horizontal", "Vertical"],
        )
        self.comparison_layout_combo.pack(side="left", padx=4)
        self.comparison_layout_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_comparison_layout())
        self.ttk.Checkbutton(
            controls,
            text="Sync Zoom/Pan",
            variable=self.comparison_sync_var,
        ).pack(side="left", padx=(12, 4))
        self.ttk.Checkbutton(
            controls,
            text="Lock Views",
            variable=self.comparison_lock_views_var,
        ).pack(side="left", padx=4)

        slot_frame = self.ttk.Frame(self.comparison_tab)
        slot_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.ttk.Label(slot_frame, textvariable=self.comparison_slot_a_var).pack(side="left", padx=(0, 16))
        self.ttk.Label(slot_frame, textvariable=self.comparison_slot_b_var).pack(side="left")
        self.ttk.Label(
            slot_frame,
            text="Use the Matplotlib toolbars under each image for zoom, pan, back, and home.",
        ).pack(side="right")

        comparison_split = self.ttk.Panedwindow(self.comparison_tab, orient="vertical")
        comparison_split.pack(fill="both", expand=True, padx=6, pady=6)
        inspectors_frame = self.ttk.Frame(comparison_split)
        details_frame = self.ttk.Frame(comparison_split)
        comparison_split.add(inspectors_frame, weight=5)
        comparison_split.add(details_frame, weight=2)

        self.comparison_inspector_grid = self.ttk.Frame(inspectors_frame)
        self.comparison_inspector_grid.pack(fill="both", expand=True)
        self.comparison_inspector_a = _ComparisonImageInspector(
            self,
            self.comparison_inspector_grid,
            slot_name="A",
            on_view_changed=self._on_inspector_view_changed,
            on_limits_changed=self._on_inspector_limits_changed,
            on_open_large=self._open_single_inspector_window,
        )
        self.comparison_inspector_b = _ComparisonImageInspector(
            self,
            self.comparison_inspector_grid,
            slot_name="B",
            on_view_changed=self._on_inspector_view_changed,
            on_limits_changed=self._on_inspector_limits_changed,
            on_open_large=self._open_single_inspector_window,
        )
        self._refresh_comparison_layout()

        self.ttk.Label(details_frame, textvariable=self.comparison_summary_var, justify="left").pack(
            anchor="w", padx=4, pady=(4, 6)
        )
        metric_columns = ("metric", "a_value", "b_value", "delta", "winner", "category")
        self.comparison_metric_tree = self.ttk.Treeview(
            details_frame,
            columns=metric_columns,
            show="headings",
            height=12,
        )
        for column, label, width in (
            ("metric", "Metric", 240),
            ("a_value", "A", 110),
            ("b_value", "B", 110),
            ("delta", "A-B", 110),
            ("winner", "Preferred", 90),
            ("category", "Group", 90),
        ):
            self.comparison_metric_tree.heading(column, text=label)
            anchor = "w" if column == "metric" else "center"
            self.comparison_metric_tree.column(column, width=width, anchor=anchor)
        metric_scroll = self.ttk.Scrollbar(
            details_frame,
            orient="vertical",
            command=self.comparison_metric_tree.yview,
        )
        self.comparison_metric_tree.configure(yscrollcommand=metric_scroll.set)
        self.comparison_metric_tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=(0, 4))
        metric_scroll.pack(side="right", fill="y", padx=(0, 4), pady=(0, 4))

    def _build_evaluation_tab(self) -> None:
        controls = self.ttk.Frame(self.evaluation_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="Stage").pack(side="left")
        self.evaluation_stage_combo = self.ttk.Combobox(
            controls,
            textvariable=self.search_stage_var,
            state="readonly",
            width=16,
        )
        self.evaluation_stage_combo.pack(side="left", padx=4)
        self.evaluation_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_evaluation())

        self.evaluation_text = self.tk.Text(self.evaluation_tab, height=12, wrap="word")
        self.evaluation_text.pack(fill="x", padx=6, pady=6)
        self.disagreement_tree = self.ttk.Treeview(
            self.evaluation_tab,
            columns=("profile", "dataset", "higher", "lower", "gap"),
            show="headings",
        )
        for column, label, width in (
            ("profile", "Profile", 160),
            ("dataset", "Dataset", 90),
            ("higher", "Preferred", 180),
            ("lower", "Outranked By", 180),
            ("gap", "Rank Gap", 90),
        ):
            self.disagreement_tree.heading(column, text=label)
            self.disagreement_tree.column(column, width=width, anchor="center")
        self.disagreement_tree.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_search_tab(self) -> None:
        controls = self.ttk.Frame(self.search_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="Stage").pack(side="left")
        self.search_stage_combo = self.ttk.Combobox(
            controls,
            textvariable=self.search_stage_var,
            state="readonly",
            width=16,
        )
        self.search_stage_combo.pack(side="left", padx=4)
        self.ttk.Label(controls, text="Weight Step").pack(side="left", padx=(8, 0))
        self.ttk.Entry(controls, textvariable=self.search_step_var, width=6).pack(side="left", padx=4)
        self.ttk.Label(controls, text="Max Metrics").pack(side="left", padx=(8, 0))
        self.ttk.Entry(controls, textvariable=self.search_max_metrics_var, width=6).pack(side="left", padx=4)
        self.ttk.Button(controls, text="Search", command=self._run_search).pack(side="left", padx=(10, 4))
        self.ttk.Button(controls, text="Build Equal Profile", command=self._build_equal_weight_profile).pack(side="left", padx=4)
        self.ttk.Button(controls, text="Build JSON Profile", command=self._build_json_profile).pack(side="left", padx=4)

        main = self.ttk.Panedwindow(self.search_tab, orient="horizontal")
        main.pack(fill="both", expand=True, padx=6, pady=6)
        left = self.ttk.Frame(main)
        right = self.ttk.Frame(main)
        main.add(left, weight=2)
        main.add(right, weight=3)

        self.metric_listbox = self.tk.Listbox(left, selectmode="multiple", exportselection=False)
        self.metric_listbox.pack(fill="both", expand=True)

        editor = self.ttk.Frame(right)
        editor.pack(fill="x", padx=4, pady=4)
        self.ttk.Label(editor, text="Profile Name").pack(side="left")
        self.ttk.Entry(editor, textvariable=self.search_profile_name_var, width=28).pack(side="left", padx=4)
        self.weight_json_text = self.tk.Text(right, height=10, wrap="word")
        self.weight_json_text.pack(fill="x", padx=4, pady=4)
        self.search_results_text = self.tk.Text(right, wrap="word")
        self.search_results_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _set_text(self, widget: Any, text: str) -> None:
        widget.delete("1.0", self.tk.END)
        widget.insert(self.tk.END, text)

    def _on_browse(self) -> None:
        selection = self.filedialog.askdirectory()
        if selection:
            self.path_var.set(selection)

    def _on_load(self) -> None:
        raw_path = self.path_var.get().strip()
        if not raw_path:
            self.messagebox.showwarning("Metric Explorer", "Choose a sweep directory first.")
            return
        self.load_from_path(Path(raw_path))

    def _on_reload(self) -> None:
        raw_path = self.path_var.get().strip()
        if raw_path:
            self.load_from_path(Path(raw_path))

    def _on_open_folder(self) -> None:
        raw_path = self.path_var.get().strip()
        if not raw_path:
            return
        ok, err = _open_with_system(Path(raw_path))
        if not ok:
            self.messagebox.showerror("Open Folder", err or "Failed to open folder.")

    def _on_export_dialog(self) -> None:
        if self.model is None:
            self.messagebox.showwarning("Export", "Load a sweep first.")
            return
        selection = self.filedialog.askdirectory()
        if selection:
            export_metric_report(self.model, Path(selection))
            self.status_var.set(f"Exported to {selection}")

    def load_from_path(self, path: Path) -> None:
        try:
            model = load_sweep_metric_explorer_model(path, metrics_config_path=self.metrics_config_path)
        except Exception as exc:  # noqa: BLE001
            self.status_var.set("Load failed")
            self.messagebox.showerror("Metric Explorer", str(exc))
            return
        self.model = model
        self.path_var.set(str(path))
        self.status_var.set(f"Loaded {len(model.raw_records)} raw metric rows")
        self._populate_from_model(model)

    def _populate_from_model(self, model: SweepMetricExplorerModel) -> None:
        self._comparison_image_cache.clear()
        self._comparison_slot_a_key = None
        self._comparison_slot_b_key = None
        self._close_comparison_popouts()
        self._update_comparison_slot_labels()
        stage_values = list(model.base_model.stage_order)
        self.stage_combo["values"] = stage_values
        self.evaluation_stage_combo["values"] = stage_values
        self.search_stage_combo["values"] = stage_values
        if stage_values:
            default_stage = model.base_model.primary_stage_name
            self.stage_var.set(default_stage)
            self.search_stage_var.set(default_stage)
        profile_values = [profile.name for profile in model.config.profiles]
        self.profile_combo["values"] = profile_values
        if profile_values:
            self.profile_var.set(model.config.default_profile)
        self.dataset_combo["values"] = _real_datasets(model.base_model)
        self.dataset_var.set((_real_datasets(model.base_model) or ["aggregate"])[0])
        self.metric_listbox.delete(0, self.tk.END)
        for metric_name in sorted(model.metric_registry.keys()):
            self.metric_listbox.insert(self.tk.END, metric_name)
        for metric_name in model.config.search_defaults.metric_subset:
            try:
                idx = sorted(model.metric_registry.keys()).index(metric_name)
            except ValueError:
                continue
            self.metric_listbox.selection_set(idx)
        self._set_text(
            self.weight_json_text,
            json.dumps(
                model.config.profiles[0].metric_weights if model.config.profiles else {},
                indent=2,
                sort_keys=True,
            ),
        )
        self.refresh_leaderboard()
        self.refresh_comparison()
        self.refresh_evaluation()
        self._set_text(self.search_results_text, _format_search_results_text(model.search_results))

    def _update_comparison_slot_labels(self) -> None:
        self.comparison_slot_a_var.set(f"A: {_record_key_to_text(self._comparison_slot_a_key)}")
        self.comparison_slot_b_var.set(f"B: {_record_key_to_text(self._comparison_slot_b_key)}")

    def _comparison_slot_key_ref(self, slot_name: str) -> Optional[ComparisonRecordKey]:
        return self._comparison_slot_a_key if slot_name == "A" else self._comparison_slot_b_key

    def _set_comparison_slot_key(self, slot_name: str, key: Optional[ComparisonRecordKey]) -> None:
        if slot_name == "A":
            self._comparison_slot_a_key = key
        else:
            self._comparison_slot_b_key = key

    def _set_comparison_slot(self, slot_name: str) -> None:
        if self.model is None:
            return
        selected = self._selected_leaderboard_rows()
        if not selected:
            self.messagebox.showwarning("Comparison", "Select a run in the leaderboard first.")
            return
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        row = selected[0]
        key = (row.stage, row.candidate_id, dataset_id)
        self._set_comparison_slot_key(slot_name, key)
        self._update_comparison_slot_labels()
        self.refresh_comparison()

    def _assign_comparison_slots_from_selection(self) -> None:
        if self.model is None:
            return
        selected = self._selected_leaderboard_rows()
        if not selected:
            self.messagebox.showwarning("Comparison", "Select one or two runs in the leaderboard first.")
            return
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        ordered = selected[:2]
        if ordered:
            self._comparison_slot_a_key = (ordered[0].stage, ordered[0].candidate_id, dataset_id)
        if len(ordered) > 1:
            self._comparison_slot_b_key = (ordered[1].stage, ordered[1].candidate_id, dataset_id)
        self._update_comparison_slot_labels()
        self.refresh_comparison()

    def _swap_comparison_slots(self) -> None:
        self._comparison_slot_a_key, self._comparison_slot_b_key = (
            self._comparison_slot_b_key,
            self._comparison_slot_a_key,
        )
        self._update_comparison_slot_labels()
        self.refresh_comparison()

    def _clear_comparison_slots(self) -> None:
        self._comparison_slot_a_key = None
        self._comparison_slot_b_key = None
        self._update_comparison_slot_labels()
        self.refresh_comparison()

    def _refresh_comparison_layout(self) -> None:
        if not hasattr(self, "comparison_inspector_a"):
            return
        for child in self.comparison_inspector_grid.winfo_children():
            child.grid_forget()
        orientation = self.comparison_layout_var.get()
        if orientation == "Vertical":
            self.comparison_inspector_grid.rowconfigure(0, weight=1)
            self.comparison_inspector_grid.rowconfigure(1, weight=1)
            self.comparison_inspector_grid.columnconfigure(0, weight=1)
            self.comparison_inspector_grid.columnconfigure(1, weight=0)
            self.comparison_inspector_a.frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            self.comparison_inspector_b.frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        else:
            self.comparison_inspector_grid.rowconfigure(0, weight=1)
            self.comparison_inspector_grid.rowconfigure(1, weight=0)
            self.comparison_inspector_grid.columnconfigure(0, weight=1)
            self.comparison_inspector_grid.columnconfigure(1, weight=1)
            self.comparison_inspector_a.frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            self.comparison_inspector_b.frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

    def _record_for_key(self, key: Optional[ComparisonRecordKey]) -> Optional[RawMetricRecord]:
        if self.model is None or key is None:
            return None
        return _record_lookup(self.model.raw_records).get(key)

    def _display_bundle_for_record(self, record: Optional[RawMetricRecord]) -> Optional[ComparisonDisplayBundle]:
        if record is None:
            return None
        key = _metric_row_key(record)
        cached = self._comparison_image_cache.get(key)
        if cached is not None:
            return cached
        bundle = load_comparison_display_bundle(record)
        self._comparison_image_cache[key] = bundle
        return bundle

    def _current_profile_results(self) -> List[ProfileCandidateScore]:
        if self.model is None:
            return []
        return list(
            self.model.profile_results.get(
                (self.stage_var.get(), self.profile_var.get()),
                [],
            )
        )

    def refresh_leaderboard(self) -> None:
        if self.model is None:
            return
        self._leaderboard_rows = self._current_profile_results()
        candidate_rows = {
            row.candidate_id: row
            for row in self.model.base_model.candidates
            if row.stage == self.stage_var.get()
        }
        saved_rows = {
            row.candidate_id: row.rank
            for row in self.model.base_model.candidates
            if row.stage == self.stage_var.get()
        }
        self._leaderboard_row_lookup = {row.candidate_id: row for row in self._leaderboard_rows}
        self.leaderboard_tree.delete(*self.leaderboard_tree.get_children())
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        for row in self._leaderboard_rows:
            saved_rank = saved_rows.get(row.candidate_id)
            label = self.model.annotations.get(_annotation_key(row.stage, row.candidate_id, dataset_id))
            dataset_score = row.dataset_scores.get(dataset_id, row.score)
            delta = ""
            if saved_rank is not None:
                delta = f"{saved_rank - row.rank:+d}"
            self.leaderboard_tree.insert(
                "",
                "end",
                iid=row.candidate_id,
                values=(
                    row.rank,
                    row.candidate_id,
                    f"{dataset_score:.4f}",
                    delta,
                    saved_rank if saved_rank is not None else "n/a",
                    row.strategy,
                    label.label if label is not None else "",
                ),
            )
        if self._leaderboard_rows:
            self.leaderboard_tree.selection_set(self._leaderboard_rows[0].candidate_id)
        self._on_leaderboard_selection_changed()

    def _selected_leaderboard_rows(self) -> List[ProfileCandidateScore]:
        return [
            self._leaderboard_row_lookup[item_id]
            for item_id in self.leaderboard_tree.selection()
            if item_id in self._leaderboard_row_lookup
        ]

    def _on_leaderboard_selection_changed(self) -> None:
        if self.model is None:
            return
        selected = self._selected_leaderboard_rows()
        if not selected:
            self._set_text(self.leaderboard_detail, "No selection.")
            return
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        record_lookup = _record_lookup(self.model.raw_records)
        lines = []
        for row in selected:
            record = record_lookup.get((row.stage, row.candidate_id, dataset_id))
            if record is None:
                lines.append(f"{row.candidate_id}: no raw record for dataset {dataset_id}")
                continue
            lines.append(_format_record_metric_lines(self.model, record))
            lines.append("")
        self._set_text(self.leaderboard_detail, "\n".join(lines).strip())
        self.refresh_comparison()

    def _comparison_annotation_label(self, record: Optional[RawMetricRecord]) -> str:
        if self.model is None or record is None:
            return ""
        annotation = self.model.annotations.get(
            _annotation_key(record.stage, record.candidate_id, record.dataset_id)
        )
        return annotation.label if annotation is not None else ""

    def _current_comparison_records(self) -> List[RawMetricRecord]:
        records = [
            self._record_for_key(self._comparison_slot_a_key),
            self._record_for_key(self._comparison_slot_b_key),
        ]
        return [record for record in records if record is not None]

    def _refresh_comparison_metric_table(
        self,
        record_a: Optional[RawMetricRecord],
        record_b: Optional[RawMetricRecord],
    ) -> None:
        self.comparison_metric_tree.delete(*self.comparison_metric_tree.get_children())
        if self.model is None or record_a is None or record_b is None:
            return
        for row in build_comparison_metric_rows(self.model, record_a, record_b):
            self.comparison_metric_tree.insert(
                "",
                "end",
                values=(
                    row.metric_label,
                    f"{row.value_a:.6f}",
                    f"{row.value_b:.6f}",
                    f"{row.delta:+.6f}",
                    row.winner,
                    row.category,
                ),
            )

    def _comparison_summary_text(
        self,
        record_a: Optional[RawMetricRecord],
        record_b: Optional[RawMetricRecord],
    ) -> str:
        if record_a is None and record_b is None:
            return "Set A and B from the leaderboard to begin."
        if record_a is None or record_b is None:
            missing = "A" if record_a is None else "B"
            ready = "B" if record_a is None else "A"
            ready_record = record_b if record_a is None else record_a
            assert ready_record is not None
            return (
                f"Run {ready} is ready ({ready_record.candidate_id} / {ready_record.dataset_id}). "
                f"Set slot {missing} to start direct A/B comparison."
            )
        dataset_status = (
            "same dataset"
            if record_a.dataset_id == record_b.dataset_id
            else f"different datasets ({record_a.dataset_id} vs {record_b.dataset_id})"
        )
        return (
            f"Comparing {record_a.candidate_id} against {record_b.candidate_id} "
            f"with {dataset_status}. Use each toolbar for local inspection; "
            f"{'zoom/pan is synced' if self.comparison_sync_var.get() else 'zoom/pan is independent'}."
        )

    def refresh_comparison(self) -> None:
        if self.model is None:
            return
        record_a = self._record_for_key(self._comparison_slot_a_key)
        record_b = self._record_for_key(self._comparison_slot_b_key)
        self.comparison_inspector_a.set_record(
            record_a,
            self._display_bundle_for_record(record_a),
            annotation_label=self._comparison_annotation_label(record_a),
        )
        self.comparison_inspector_b.set_record(
            record_b,
            self._display_bundle_for_record(record_b),
            annotation_label=self._comparison_annotation_label(record_b),
        )
        self.comparison_summary_var.set(self._comparison_summary_text(record_a, record_b))
        self._refresh_comparison_metric_table(record_a, record_b)
        self._refresh_open_comparison_popouts()

    def _on_inspector_view_changed(self, source: _ComparisonImageInspector, view_name: str) -> None:
        if not self.comparison_lock_views_var.get() or self._comparison_view_sync_in_progress:
            return
        other = self.comparison_inspector_b if source is self.comparison_inspector_a else self.comparison_inspector_a
        self._comparison_view_sync_in_progress = True
        try:
            other.set_view(view_name, emit_event=False, reset_view=False)
        finally:
            self._comparison_view_sync_in_progress = False
        self._refresh_open_comparison_popouts()

    def _on_inspector_limits_changed(self, source: _ComparisonImageInspector) -> None:
        if not self.comparison_sync_var.get() or self._comparison_sync_in_progress:
            return
        other = self.comparison_inspector_b if source is self.comparison_inspector_a else self.comparison_inspector_a
        if not source.has_image() or not other.has_image():
            return
        xlim, ylim = source.get_axes_limits()
        if xlim is None or ylim is None:
            return
        self._comparison_sync_in_progress = True
        try:
            other.apply_axes_limits(xlim, ylim)
        finally:
            self._comparison_sync_in_progress = False
        self._refresh_open_comparison_popouts()

    def _open_single_inspector_window(self, slot_name: str) -> None:
        source = self.comparison_inspector_a if slot_name == "A" else self.comparison_inspector_b
        if source.record is None:
            self.messagebox.showwarning("Comparison Viewer", f"Set slot {slot_name} first.")
            return
        existing = self._comparison_popouts.get(slot_name)
        if existing is not None:
            window = existing["window"]
            if window.winfo_exists():
                self._sync_popout_from_main(slot_name)
                window.deiconify()
                window.lift()
                window.focus_force()
                return
        viewer = self.tk.Toplevel(self.root)
        viewer.title(f"Run {slot_name} Viewer")
        viewer.geometry("1200x980")
        inspector = _ComparisonImageInspector(
            self,
            viewer,
            slot_name=slot_name,
            on_view_changed=None,
            on_limits_changed=None,
            on_open_large=None,
            allow_open_large=False,
        )
        inspector.frame.pack(fill="both", expand=True)
        self._comparison_popouts[slot_name] = {"window": viewer, "inspector": inspector}

        def _close_popout() -> None:
            self._comparison_popouts.pop(slot_name, None)
            viewer.destroy()

        viewer.protocol("WM_DELETE_WINDOW", _close_popout)
        self._sync_popout_from_main(slot_name)

    def _sync_popout_from_main(self, slot_name: str) -> None:
        popout = self._comparison_popouts.get(slot_name)
        if popout is None:
            return
        target = cast(_ComparisonImageInspector, popout["inspector"])
        source = self.comparison_inspector_a if slot_name == "A" else self.comparison_inspector_b
        target.set_record(
            source.record,
            self._display_bundle_for_record(source.record),
            annotation_label=self._comparison_annotation_label(source.record),
        )
        if source.record is not None:
            target.clone_view_state_from(source)

    def _refresh_open_comparison_popouts(self) -> None:
        for slot_name, payload in list(self._comparison_popouts.items()):
            window = payload.get("window")
            if window is None or not window.winfo_exists():
                self._comparison_popouts.pop(slot_name, None)
                continue
            self._sync_popout_from_main(slot_name)

    def _close_comparison_popouts(self) -> None:
        for slot_name, payload in list(self._comparison_popouts.items()):
            window = payload.get("window")
            if window is not None and window.winfo_exists():
                window.destroy()
            self._comparison_popouts.pop(slot_name, None)

    def refresh_evaluation(self) -> None:
        if self.model is None:
            return
        stage_name = self.search_stage_var.get() or self.stage_var.get()
        self._set_text(self.evaluation_text, _format_evaluation_text(self.model, stage_name))
        self.disagreement_tree.delete(*self.disagreement_tree.get_children())
        for idx, row in enumerate(
            [item for item in self.model.disagreements if item.stage == stage_name][:100],
            start=1,
        ):
            self.disagreement_tree.insert(
                "",
                "end",
                iid=f"disagreement_{idx}",
                values=(
                    row.profile_name,
                    row.dataset_id,
                    f"{row.higher_candidate_id} ({row.higher_label})",
                    f"{row.lower_candidate_id} ({row.lower_label})",
                    row.rank_gap,
                ),
            )

    def _set_labels_for_selection(self, label: str) -> None:
        if self.model is None:
            return
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        for row in self._selected_leaderboard_rows():
            key = _annotation_key(row.stage, row.candidate_id, dataset_id)
            self.model.annotations[key] = QualityAnnotation(
                stage=row.stage,
                candidate_id=row.candidate_id,
                dataset_id=dataset_id,
                label=label,
            )
        save_annotations(self.model)
        self.model = load_sweep_metric_explorer_model(
            Path(self.path_var.get().strip()),
            metrics_config_path=self.metrics_config_path,
        )
        self._populate_from_model(self.model)

    def _clear_labels_for_selection(self) -> None:
        if self.model is None:
            return
        dataset_id = self.dataset_var.get() or (_real_datasets(self.model.base_model)[0])
        for row in self._selected_leaderboard_rows():
            self.model.annotations.pop(_annotation_key(row.stage, row.candidate_id, dataset_id), None)
        save_annotations(self.model)
        self.model = load_sweep_metric_explorer_model(
            Path(self.path_var.get().strip()),
            metrics_config_path=self.metrics_config_path,
        )
        self._populate_from_model(self.model)

    def _selected_metric_names(self) -> List[str]:
        if self.model is None:
            return []
        names = sorted(self.model.metric_registry.keys())
        return [names[idx] for idx in self.metric_listbox.curselection()]

    def _build_equal_weight_profile(self) -> None:
        if self.model is None:
            return
        selected = self._selected_metric_names()
        if not selected:
            self.messagebox.showwarning("Profiles", "Select at least one metric.")
            return
        weight = 1.0 / len(selected)
        weights = {name: weight for name in selected}
        self._set_text(self.weight_json_text, json.dumps(weights, indent=2, sort_keys=True))
        self._build_json_profile()

    def _build_json_profile(self) -> None:
        if self.model is None:
            return
        try:
            weights = json.loads(self.weight_json_text.get("1.0", self.tk.END))
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Profiles", f"Invalid weight JSON: {exc}")
            return
        if not isinstance(weights, dict):
            self.messagebox.showerror("Profiles", "Weight JSON must be an object mapping metric names to weights.")
            return
        metric_weights = {
            str(name): float(value)
            for name, value in weights.items()
            if str(name) in self.model.metric_registry and _coerce_float(value) is not None and float(value) > 0.0
        }
        if not metric_weights:
            self.messagebox.showwarning("Profiles", "No valid positive weights found.")
            return
        profile_name = self.search_profile_name_var.get().strip() or "custom_profile"
        profile = RankingProfileSpec(
            name=profile_name,
            label=profile_name,
            metric_weights=metric_weights,
            description="Temporary custom profile built in the GUI.",
            built_in=False,
        )
        self.model.config.profiles.append(profile)
        profile_values = [item.name for item in self.model.config.profiles]
        self.profile_combo["values"] = profile_values
        self.profile_var.set(profile_name)
        for stage_name in self.model.base_model.stage_order:
            self.model.profile_results[(stage_name, profile.name)] = _score_profile_for_stage(
                stage_name,
                profile,
                self.model.base_model,
                self.model.raw_records,
                self.model.metric_registry,
                self.model.scoring_config,
            )
            summary = _evaluate_profile_for_stage(
                stage_name,
                profile.name,
                self.model.profile_results[(stage_name, profile.name)],
                self.model.annotations,
            )
            self.model.evaluations.append(summary)
            self.model.disagreements.extend(
                _build_disagreements(
                    stage_name,
                    profile.name,
                    self.model.profile_results[(stage_name, profile.name)],
                    self.model.annotations,
                )
            )
        self.refresh_leaderboard()
        self.refresh_evaluation()

    def _run_search(self) -> None:
        if self.model is None:
            return
        selected = self._selected_metric_names()
        if not selected:
            self.messagebox.showwarning("Search", "Select one or more metrics first.")
            return
        if len(selected) > int(self.search_max_metrics_var.get().strip() or "6"):
            self.messagebox.showwarning("Search", "Selected metric count exceeds the configured max.")
            return
        try:
            step = float(self.search_step_var.get().strip())
        except ValueError:
            self.messagebox.showerror("Search", "Weight step must be numeric.")
            return
        results = search_profile_combinations(
            self.model,
            stage_name=self.search_stage_var.get() or self.stage_var.get(),
            metric_names=selected,
            weight_step=step,
        )
        self._search_results = results
        self.model.search_results = results
        self._set_text(self.search_results_text, _format_search_results_text(results))


def launch_sweep_metric_explorer(
    initial_path: Optional[Path] = None,
    *,
    metrics_config_path: Optional[Path] = None,
) -> None:
    import tkinter as tk

    root = tk.Tk()
    SweepMetricExplorerApp(root, initial_path=initial_path, metrics_config_path=metrics_config_path)
    root.mainloop()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore raw sweep metrics from workspace.npz artifacts.")
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        help="Optional sweep output directory to load on startup.",
    )
    parser.add_argument(
        "--export-dir",
        help="Optional output directory for headless export without launching the GUI.",
    )
    parser.add_argument(
        "--metrics-config",
        help="Optional JSON config for enabled metrics, profiles, and search defaults.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics_config_path = Path(args.metrics_config) if args.metrics_config else None
    if args.export_dir:
        if not args.sweep_dir:
            raise ValueError("--export-dir requires a sweep_dir argument.")
        model = load_sweep_metric_explorer_model(
            Path(args.sweep_dir),
            metrics_config_path=metrics_config_path,
        )
        export_metric_report(model, Path(args.export_dir))
        return
    initial_path = Path(args.sweep_dir) if args.sweep_dir else None
    launch_sweep_metric_explorer(initial_path=initial_path, metrics_config_path=metrics_config_path)


if __name__ == "__main__":
    main()

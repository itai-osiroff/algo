"""Manual bead-center oracle metrics for validation-guided sweep shortcuts."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from moment_images.reliable_io import read_json_first_valid, read_npz_first_valid, write_json_atomic
from moment_images.trio_oracle_metrics import (
    TRIO_LF_SOURCE_KIND,
    TrioOracleImport,
    TrioOracleRecord,
    load_trio_oracle_import,
    trio_record_key,
    write_trio_oracle_import,
)


HUMAN_BEAD_ORACLE_MODE = "human_bead"
HUMAN_BEAD_ORACLE_PROFILE_NAME = "human_bead_oracle"
HUMAN_BEAD_ORACLE_SCORE = "human_bead_oracle_score_v1"
HUMAN_BEAD_EVALUATED_BEAD_COUNT = "human_bead_evaluated_bead_count"
HUMAN_BEAD_FWHM_RADIUS_MEAN = "human_bead_fwhm_radius_px_mean"
HUMAN_BEAD_SUPPORT_RADIUS_MEAN = "human_bead_support_radius_px_mean"
HUMAN_BEAD_HALO_TO_CORE_RATIO_MEAN = "human_bead_halo_to_core_ratio_mean"
HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN = "human_bead_core_energy_fraction_mean"
HUMAN_BEAD_ELONGATION_MEAN = "human_bead_elongation_mean"
HUMAN_BEAD_CENTER_SHIFT_P95 = "human_bead_center_shift_px_p95"
HUMAN_BEAD_RESIDUAL_GRADIENT_P95 = "human_bead_residual_gradient_p95"
HUMAN_BEAD_RESIDUAL_GRADIENT_MAX = "human_bead_residual_gradient_max"

HUMAN_BEAD_REQUIRED_METRICS = (
    HUMAN_BEAD_FWHM_RADIUS_MEAN,
    HUMAN_BEAD_SUPPORT_RADIUS_MEAN,
    HUMAN_BEAD_HALO_TO_CORE_RATIO_MEAN,
    HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN,
    HUMAN_BEAD_ELONGATION_MEAN,
)
HUMAN_BEAD_NUMERIC_METRICS = (
    HUMAN_BEAD_ORACLE_SCORE,
    HUMAN_BEAD_EVALUATED_BEAD_COUNT,
    HUMAN_BEAD_FWHM_RADIUS_MEAN,
    HUMAN_BEAD_SUPPORT_RADIUS_MEAN,
    HUMAN_BEAD_HALO_TO_CORE_RATIO_MEAN,
    HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN,
    HUMAN_BEAD_ELONGATION_MEAN,
    HUMAN_BEAD_CENTER_SHIFT_P95,
    HUMAN_BEAD_RESIDUAL_GRADIENT_P95,
    HUMAN_BEAD_RESIDUAL_GRADIENT_MAX,
)


@dataclass(frozen=True)
class HumanBead:
    bead_id: str
    x: float
    y: float
    radius_px: float


@dataclass(frozen=True)
class HumanBeadDatasetAnnotations:
    dataset_id: str
    image_path: Optional[str]
    image_shape: Tuple[int, int]
    default_radius_px: float
    beads: List[HumanBead]


@dataclass(frozen=True)
class HumanBeadOracleResult:
    metrics: Dict[str, float]
    diagnostics: Dict[str, object]
    rankable: bool
    warnings: List[str]


def _coerce_float(value: object, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _coerce_shape(value: object, *, label: str) -> Tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be [height, width].")
    height = int(_coerce_float(value[0], label=f"{label}[0]"))
    width = int(_coerce_float(value[1], label=f"{label}[1]"))
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} dimensions must be positive.")
    return height, width


def _parse_bead(raw: Mapping[str, object], *, dataset_id: str, index: int, default_radius: float) -> HumanBead:
    bead_id = str(raw.get("id") or raw.get("bead_id") or f"b{index:03d}").strip()
    if not bead_id:
        bead_id = f"b{index:03d}"
    radius = _coerce_float(raw.get("radius_px", default_radius), label=f"{dataset_id} bead {bead_id} radius")
    if radius <= 0.0:
        raise ValueError(f"{dataset_id} bead {bead_id} radius must be positive.")
    return HumanBead(
        bead_id=bead_id,
        x=_coerce_float(raw.get("x"), label=f"{dataset_id} bead {bead_id} x"),
        y=_coerce_float(raw.get("y"), label=f"{dataset_id} bead {bead_id} y"),
        radius_px=radius,
    )


def _parse_dataset_annotations(raw: Mapping[str, object], *, index: int) -> HumanBeadDatasetAnnotations:
    dataset_id = str(raw.get("dataset_id") or raw.get("id") or "").strip()
    if not dataset_id:
        raise ValueError(f"annotation dataset {index} must include dataset_id.")
    image_shape = _coerce_shape(raw.get("image_shape"), label=f"{dataset_id}.image_shape")
    default_radius = _coerce_float(raw.get("default_radius_px", 4.0), label=f"{dataset_id}.default_radius_px")
    if default_radius <= 0.0:
        raise ValueError(f"{dataset_id}.default_radius_px must be positive.")
    raw_beads = raw.get("beads")
    if not isinstance(raw_beads, list):
        raise ValueError(f"{dataset_id}.beads must be a list.")
    beads: List[HumanBead] = []
    for bead_index, item in enumerate(raw_beads, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{dataset_id}.beads[{bead_index}] must be an object.")
        beads.append(
            _parse_bead(
                item,
                dataset_id=dataset_id,
                index=bead_index,
                default_radius=default_radius,
            )
        )
    return HumanBeadDatasetAnnotations(
        dataset_id=dataset_id,
        image_path=(None if raw.get("image_path") in {None, ""} else str(raw.get("image_path"))),
        image_shape=image_shape,
        default_radius_px=default_radius,
        beads=beads,
    )


def load_human_bead_annotations(path: Path) -> Dict[str, HumanBeadDatasetAnnotations]:
    payload, _used_path, _warnings = read_json_first_valid(
        Path(path),
        validator=lambda obj: isinstance(obj, dict),
    )
    raw_datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    if not isinstance(raw_datasets, list):
        raise ValueError("human bead annotations must contain a 'datasets' list.")
    annotations: Dict[str, HumanBeadDatasetAnnotations] = {}
    for index, item in enumerate(raw_datasets, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"annotation dataset {index} must be an object.")
        parsed = _parse_dataset_annotations(item, index=index)
        if parsed.dataset_id in annotations:
            raise ValueError(f"duplicate human bead annotation dataset: {parsed.dataset_id}")
        annotations[parsed.dataset_id] = parsed
    return annotations


def _bead_to_payload(bead: HumanBead) -> Dict[str, object]:
    return {
        "id": bead.bead_id,
        "x": float(bead.x),
        "y": float(bead.y),
        "radius_px": float(bead.radius_px),
    }


def write_human_bead_annotations(
    path: Path,
    annotations: Mapping[str, HumanBeadDatasetAnnotations],
) -> None:
    datasets = []
    for dataset_id in sorted(annotations):
        item = annotations[dataset_id]
        datasets.append(
            {
                "dataset_id": item.dataset_id,
                "image_path": item.image_path,
                "image_shape": list(item.image_shape),
                "default_radius_px": float(item.default_radius_px),
                "beads": [_bead_to_payload(bead) for bead in item.beads],
            }
        )
    write_json_atomic(
        Path(path),
        {"version": 1, "datasets": datasets},
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def scaled_beads_for_image(
    annotation: HumanBeadDatasetAnnotations,
    image_shape: Tuple[int, int],
    *,
    tolerance: float = 1e-6,
) -> List[HumanBead]:
    ann_h, ann_w = annotation.image_shape
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    if img_h <= 0 or img_w <= 0:
        raise ValueError("image_shape dimensions must be positive.")
    scale_y = float(img_h) / float(ann_h)
    scale_x = float(img_w) / float(ann_w)
    if abs(scale_x - scale_y) > tolerance * max(1.0, abs(scale_x), abs(scale_y)):
        raise ValueError(
            f"non-uniform shape change for {annotation.dataset_id}: "
            f"annotation={annotation.image_shape}, image={(img_h, img_w)}"
        )
    scale = 0.5 * (scale_x + scale_y)
    return [
        HumanBead(
            bead_id=bead.bead_id,
            x=float(bead.x) * scale_x,
            y=float(bead.y) * scale_y,
            radius_px=float(bead.radius_px) * scale,
        )
        for bead in annotation.beads
    ]


def _robust_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray(np.clip((arr - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def _patch_with_grid(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    arr = np.asarray(image, dtype=np.float32)
    cx, cy = center
    pad = max(4, int(math.ceil(float(radius))))
    x0 = max(0, int(math.floor(cx)) - pad)
    x1 = min(arr.shape[1], int(math.floor(cx)) + pad + 1)
    y0 = max(0, int(math.floor(cy)) - pad)
    y1 = min(arr.shape[0], int(math.floor(cy)) + pad + 1)
    patch = arr[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape, dtype=np.float32)
    x_grid = xx + float(x0)
    y_grid = yy + float(y0)
    return patch, x_grid, y_grid, x0, y0


def _refine_center_to_local_peak(
    image: np.ndarray,
    bead: HumanBead,
) -> Tuple[float, float, float]:
    search_radius = max(2.0, 0.75 * float(bead.radius_px))
    patch, x_grid, y_grid, _x0, _y0 = _patch_with_grid(image, (bead.x, bead.y), search_radius)
    if patch.size == 0:
        return float(bead.x), float(bead.y), 0.0
    rr = np.sqrt(np.square(x_grid - float(bead.x)) + np.square(y_grid - float(bead.y)))
    values = np.where(rr <= search_radius, patch, -np.inf)
    if not np.any(np.isfinite(values)):
        return float(bead.x), float(bead.y), 0.0
    peak_index = np.unravel_index(int(np.nanargmax(values)), values.shape)
    refined_x = float(x_grid[peak_index])
    refined_y = float(y_grid[peak_index])
    shift = math.hypot(refined_x - float(bead.x), refined_y - float(bead.y))
    return refined_x, refined_y, float(shift)


def _equivalent_radius(mask_count: int) -> Optional[float]:
    if mask_count <= 0:
        return None
    return float(math.sqrt(float(mask_count) / math.pi))


def _weighted_elongation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Optional[float]:
    total = float(np.sum(weights))
    if total <= 1e-10:
        return None
    mean_x = float(np.sum(x * weights) / total)
    mean_y = float(np.sum(y * weights) / total)
    dx = x - mean_x
    dy = y - mean_y
    cov_xx = float(np.sum(np.square(dx) * weights) / total)
    cov_yy = float(np.sum(np.square(dy) * weights) / total)
    cov_xy = float(np.sum(dx * dy * weights) / total)
    trace = cov_xx + cov_yy
    disc = max(0.0, (cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy)
    lambda_max = 0.5 * (trace + math.sqrt(disc))
    lambda_min = 0.5 * (trace - math.sqrt(disc))
    if lambda_max <= 1e-10 or lambda_min <= 1e-10:
        return 1.0
    return float(math.sqrt(lambda_max / lambda_min))


def _compute_one_bead(image: np.ndarray, bead: HumanBead) -> Optional[Dict[str, float]]:
    refined_x, refined_y, shift = _refine_center_to_local_peak(image, bead)
    measurement_radius = max(8.0, 3.0 * float(bead.radius_px))
    patch, x_grid, y_grid, _x0, _y0 = _patch_with_grid(image, (refined_x, refined_y), measurement_radius)
    if patch.size == 0:
        return None
    rr = np.sqrt(np.square(x_grid - refined_x) + np.square(y_grid - refined_y))
    core_mask = rr <= float(bead.radius_px)
    annulus_mask = (rr > 1.5 * float(bead.radius_px)) & (rr <= 3.0 * float(bead.radius_px))
    support_mask = rr <= 3.0 * float(bead.radius_px)
    if not np.any(core_mask) or not np.any(support_mask):
        return None

    background_values = patch[annulus_mask] if np.any(annulus_mask) else patch[~core_mask]
    background = float(np.percentile(background_values, 50.0)) if background_values.size else float(np.min(patch))
    adjusted = np.clip(patch - background, 0.0, None)
    peak_values = adjusted[rr <= max(2.0, float(bead.radius_px))]
    peak = float(np.max(peak_values)) if peak_values.size else float(np.max(adjusted))
    if peak <= 1e-8:
        return None

    fwhm_mask = (adjusted >= 0.5 * peak) & support_mask
    support_signal_mask = (adjusted >= 0.10 * peak) & support_mask
    fwhm_radius = _equivalent_radius(int(np.count_nonzero(fwhm_mask)))
    support_radius = _equivalent_radius(int(np.count_nonzero(support_signal_mask)))
    if fwhm_radius is None or support_radius is None:
        return None

    core_values = adjusted[core_mask]
    annulus_values = adjusted[annulus_mask]
    core_mean = float(np.mean(core_values)) if core_values.size else 0.0
    halo_mean = float(np.mean(annulus_values)) if annulus_values.size else 0.0
    support_energy = float(np.sum(adjusted[support_mask]))
    core_energy = float(np.sum(core_values))
    positive_support = support_mask & (adjusted > 0.0)
    elongation = _weighted_elongation(
        x_grid[positive_support],
        y_grid[positive_support],
        adjusted[positive_support],
    )
    if elongation is None:
        return None

    return {
        HUMAN_BEAD_FWHM_RADIUS_MEAN: float(fwhm_radius),
        HUMAN_BEAD_SUPPORT_RADIUS_MEAN: float(support_radius),
        HUMAN_BEAD_HALO_TO_CORE_RATIO_MEAN: float(halo_mean / max(core_mean, 1e-8)),
        HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN: float(core_energy / max(support_energy, 1e-8)),
        HUMAN_BEAD_ELONGATION_MEAN: float(elongation),
        HUMAN_BEAD_CENTER_SHIFT_P95: float(shift),
    }


def _finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _finite_p95(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(finite, 95.0)) if finite else None


def _score_lower_is_better(value: float, scale: float) -> float:
    return float(math.exp(-max(0.0, float(value)) / max(float(scale), 1e-8)))


def _score_ratio_excess(value: float, ideal: float, scale: float) -> float:
    return float(math.exp(-max(0.0, float(value) - float(ideal)) / max(float(scale), 1e-8)))


def _optional_residual_gradient_metrics(workspace_payload: Optional[Mapping[str, object]]) -> Dict[str, float]:
    if not workspace_payload or "pixel_blended_scores" not in workspace_payload:
        return {}
    scores = np.asarray(workspace_payload["pixel_blended_scores"], dtype=np.float32)
    if scores.ndim < 2 or scores.size == 0:
        return {}
    spatial = scores
    while spatial.ndim > 2:
        spatial = np.mean(spatial, axis=-1)
    if spatial.shape[0] < 2 or spatial.shape[1] < 2:
        return {}
    gy, gx = np.gradient(np.asarray(spatial, dtype=np.float64))
    magnitude = np.sqrt(np.square(gx) + np.square(gy))
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return {}
    return {
        HUMAN_BEAD_RESIDUAL_GRADIENT_P95: float(np.percentile(finite, 95.0)),
        HUMAN_BEAD_RESIDUAL_GRADIENT_MAX: float(np.max(finite)),
    }


def _oracle_score(metrics: Mapping[str, float], annotation: HumanBeadDatasetAnnotations) -> float:
    radius = max(1e-8, float(np.mean([bead.radius_px for bead in annotation.beads])))
    components: Dict[str, Tuple[float, float]] = {
        "fwhm": (
            0.25,
            _score_ratio_excess(float(metrics[HUMAN_BEAD_FWHM_RADIUS_MEAN]) / radius, ideal=0.55, scale=0.75),
        ),
        "support": (
            0.20,
            _score_ratio_excess(float(metrics[HUMAN_BEAD_SUPPORT_RADIUS_MEAN]) / radius, ideal=1.20, scale=1.00),
        ),
        "halo": (
            0.20,
            _score_lower_is_better(float(metrics[HUMAN_BEAD_HALO_TO_CORE_RATIO_MEAN]), scale=0.60),
        ),
        "core_energy": (
            0.15,
            max(0.0, min(1.0, float(metrics[HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN]))),
        ),
        "elongation": (
            0.15,
            _score_ratio_excess(float(metrics[HUMAN_BEAD_ELONGATION_MEAN]), ideal=1.05, scale=1.0),
        ),
    }
    if HUMAN_BEAD_RESIDUAL_GRADIENT_P95 in metrics:
        components["residual_gradient"] = (
            0.05,
            _score_lower_is_better(float(metrics[HUMAN_BEAD_RESIDUAL_GRADIENT_P95]), scale=0.25),
        )
    total_weight = sum(weight for weight, _score in components.values())
    return float(sum(weight * score for weight, score in components.values()) / max(total_weight, 1e-8))


def compute_human_bead_oracle_metrics(
    image: np.ndarray,
    annotation: HumanBeadDatasetAnnotations,
    *,
    workspace_payload: Optional[Mapping[str, object]] = None,
) -> HumanBeadOracleResult:
    arr = _robust_unit(np.asarray(image, dtype=np.float32))
    warnings: List[str] = []
    if arr.ndim != 2 or arr.size == 0:
        return HumanBeadOracleResult(
            metrics={},
            diagnostics={"validation_rankable": False, "oracle_mode": HUMAN_BEAD_ORACLE_MODE},
            rankable=False,
            warnings=["empty_or_invalid_candidate_image"],
        )

    try:
        beads = scaled_beads_for_image(annotation, tuple(arr.shape))
    except ValueError as exc:
        return HumanBeadOracleResult(
            metrics={},
            diagnostics={
                "validation_rankable": False,
                "oracle_mode": HUMAN_BEAD_ORACLE_MODE,
                "ranking_exclusion_reason": "annotation_shape_mismatch",
            },
            rankable=False,
            warnings=[str(exc)],
        )

    per_bead = [_compute_one_bead(arr, bead) for bead in beads]
    evaluated = [item for item in per_bead if item is not None]
    required_count = min(3, len(beads))
    metrics: Dict[str, float] = {
        HUMAN_BEAD_EVALUATED_BEAD_COUNT: float(len(evaluated)),
    }
    for metric_name in HUMAN_BEAD_REQUIRED_METRICS:
        value = _finite_mean(item.get(metric_name) for item in evaluated)
        if value is not None:
            metrics[metric_name] = value
    center_shift = _finite_p95(item.get(HUMAN_BEAD_CENTER_SHIFT_P95) for item in evaluated)
    if center_shift is not None:
        metrics[HUMAN_BEAD_CENTER_SHIFT_P95] = center_shift
    metrics.update(_optional_residual_gradient_metrics(workspace_payload))

    missing_required = [name for name in HUMAN_BEAD_REQUIRED_METRICS if name not in metrics]
    rankable = len(evaluated) >= required_count and not missing_required
    exclusion_reason = ""
    if not rankable:
        exclusion_reason = "insufficient_bead_evidence" if len(evaluated) < required_count else "missing_required_metrics"
        warnings.append(exclusion_reason)
    else:
        metrics[HUMAN_BEAD_ORACLE_SCORE] = _oracle_score(metrics, annotation)

    diagnostics: Dict[str, object] = {
        "validation_rankable": bool(rankable),
        "oracle_mode": HUMAN_BEAD_ORACLE_MODE,
        "human_bead_annotation_count": len(beads),
    }
    if exclusion_reason:
        diagnostics["ranking_exclusion_reason"] = exclusion_reason
    return HumanBeadOracleResult(
        metrics=metrics,
        diagnostics=diagnostics,
        rankable=rankable,
        warnings=warnings,
    )


def load_candidate_workspace(workspace_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    payload, _used_path, warnings = read_npz_first_valid(Path(workspace_path), allow_pickle=True)
    avg = payload.get("average_rectified_image")
    if not isinstance(avg, np.ndarray):
        rectified = payload.get("rectified_views")
        if isinstance(rectified, np.ndarray):
            avg = np.mean(np.asarray(rectified, dtype=np.float32), axis=(0, 1))
    if not isinstance(avg, np.ndarray):
        raise ValueError(f"workspace missing average_rectified_image: {workspace_path}")
    return np.asarray(avg, dtype=np.float32), payload, warnings


def _workspace_path_from_stage_result(record: Mapping[str, object]) -> Path:
    artifact_paths = record.get("artifact_paths")
    if isinstance(artifact_paths, Mapping) and artifact_paths.get("workspace"):
        return Path(str(artifact_paths["workspace"]))
    run_dir = record.get("run_dir")
    if run_dir:
        return Path(str(run_dir)) / "workspace.npz"
    raise ValueError(f"stage result has no workspace path for candidate {record.get('candidate_id', '')}")


def build_human_bead_oracle_import(
    *,
    stage_name: str,
    stage_results: Sequence[Mapping[str, object]],
    annotations: Mapping[str, HumanBeadDatasetAnnotations],
    source_paths: Sequence[Path],
) -> TrioOracleImport:
    pending_records: List[TrioOracleRecord] = []
    for stage_result in stage_results:
        if stage_result.get("status") != "ok":
            continue
        dataset_id = str(stage_result.get("dataset_id", "")).strip()
        candidate_id = str(stage_result.get("candidate_id", "")).strip()
        if not dataset_id or not candidate_id or dataset_id not in annotations:
            continue
        warnings: List[str] = []
        try:
            image, payload, read_warnings = load_candidate_workspace(_workspace_path_from_stage_result(stage_result))
            warnings.extend(read_warnings)
            result = compute_human_bead_oracle_metrics(
                image,
                annotations[dataset_id],
                workspace_payload=payload,
            )
            metrics = dict(result.metrics)
            diagnostics = dict(result.diagnostics)
            rankable = result.rankable
            exclusion_reason = str(diagnostics.get("ranking_exclusion_reason", "") or "") or None
            warnings.extend(result.warnings)
        except Exception as exc:  # noqa: BLE001 - mark this candidate unrankable.
            metrics = {HUMAN_BEAD_EVALUATED_BEAD_COUNT: 0.0}
            diagnostics = {
                "validation_rankable": False,
                "oracle_mode": HUMAN_BEAD_ORACLE_MODE,
                "ranking_exclusion_reason": "human_bead_scoring_failed",
            }
            rankable = False
            exclusion_reason = "human_bead_scoring_failed"
            warnings.append(f"human_bead_scoring_failed:{type(exc).__name__}:{exc}")
        pending_records.append(
            TrioOracleRecord(
                source_kind=TRIO_LF_SOURCE_KIND,
                stage=stage_name,
                candidate_id=candidate_id,
                dataset_id=dataset_id,
                generation_id=(
                    None
                    if stage_result.get("generation_id") in {None, ""}
                    else str(stage_result.get("generation_id"))
                ),
                row_id=f"{stage_name}:{candidate_id}",
                validation_rank=None,
                metrics=metrics,
                diagnostics=diagnostics,
                validation_rankable=rankable,
                ranking_exclusion_reason=exclusion_reason,
                source_path=str(source_paths[0]) if source_paths else None,
                warnings=_stable_unique_strings(warnings),
            )
        )

    ranked_by_dataset: Dict[str, List[TrioOracleRecord]] = {}
    for record in pending_records:
        if record.validation_rankable and HUMAN_BEAD_ORACLE_SCORE in record.metrics:
            ranked_by_dataset.setdefault(record.dataset_id, []).append(record)
    rank_lookup: Dict[Tuple[str, str, Optional[str]], int] = {}
    for dataset_id, records in ranked_by_dataset.items():
        ordered = sorted(
            records,
            key=lambda item: float(item.metrics[HUMAN_BEAD_ORACLE_SCORE]),
            reverse=True,
        )
        for rank, record in enumerate(ordered, start=1):
            rank_lookup[(dataset_id, record.candidate_id, record.generation_id)] = rank

    final_records = [
        TrioOracleRecord(
            source_kind=record.source_kind,
            stage=record.stage,
            candidate_id=record.candidate_id,
            dataset_id=record.dataset_id,
            generation_id=record.generation_id,
            row_id=record.row_id,
            validation_rank=rank_lookup.get((record.dataset_id, record.candidate_id, record.generation_id)),
            metrics=record.metrics,
            diagnostics=record.diagnostics,
            validation_rankable=record.validation_rankable,
            ranking_exclusion_reason=record.ranking_exclusion_reason,
            report_path=record.report_path,
            overlay_path=record.overlay_path,
            diff_path=record.diff_path,
            source_path=record.source_path,
            warnings=record.warnings,
        )
        for record in pending_records
    ]
    return TrioOracleImport(
        records=final_records,
        baselines={},
        source_paths=_stable_unique_strings(str(path) for path in source_paths),
        warnings=[],
    )


def write_human_bead_oracle_import(review_dir: Path, import_data: TrioOracleImport) -> TrioOracleImport:
    existing, read_warnings = load_trio_oracle_import(Path(review_dir))
    records = {} if existing is None else {trio_record_key(record): record for record in existing.records}
    for record in import_data.records:
        records[trio_record_key(record)] = record
    baselines = {} if existing is None else dict(existing.baselines)
    baselines.update(import_data.baselines)
    merged = TrioOracleImport(
        records=list(records.values()),
        baselines=baselines,
        source_paths=_stable_unique_strings(
            [
                *(existing.source_paths if existing is not None else []),
                *import_data.source_paths,
            ]
        ),
        warnings=_stable_unique_strings(
            [
                *(existing.warnings if existing is not None else []),
                *read_warnings,
                *import_data.warnings,
            ]
        ),
    )
    write_trio_oracle_import(review_dir, merged)
    return merged


def _stable_unique_strings(values: Iterable[object]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def save_human_bead_preview_sheet(
    path: Path,
    image: np.ndarray,
    annotation: HumanBeadDatasetAnnotations,
) -> Optional[Path]:
    arr = (_robust_unit(image) * 255.0).astype(np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return None
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        beads = scaled_beads_for_image(annotation, tuple(arr.shape))
    except ValueError:
        beads = annotation.beads
    for bead in beads:
        x = float(bead.x)
        y = float(bead.y)
        r = float(bead.radius_px)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 64, 64), width=2)
        draw.ellipse((x - 3.0 * r, y - 3.0 * r, x + 3.0 * r, y + 3.0 * r), outline=(64, 192, 255), width=1)
        draw.line((x - 4, y, x + 4, y), fill=(255, 255, 0), width=1)
        draw.line((x, y - 4, x, y + 4), fill=(255, 255, 0), width=1)
        draw.text((x + r + 2, y - r), bead.bead_id, fill=(255, 255, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return Path(path)


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


class _AnnotationSession:
    def __init__(
        self,
        dataset_images: Mapping[str, Path],
        output_path: Path,
        existing: Optional[Mapping[str, HumanBeadDatasetAnnotations]] = None,
        default_radius_px: float = 4.0,
    ) -> None:
        self.dataset_ids = list(dataset_images)
        self.dataset_images = dict(dataset_images)
        self.output_path = Path(output_path)
        self.images = {dataset_id: _load_image(path) for dataset_id, path in dataset_images.items()}
        self.annotations: Dict[str, HumanBeadDatasetAnnotations] = {}
        for dataset_id in self.dataset_ids:
            image = self.images[dataset_id]
            existing_item = None if existing is None else existing.get(dataset_id)
            self.annotations[dataset_id] = existing_item or HumanBeadDatasetAnnotations(
                dataset_id=dataset_id,
                image_path=str(self.dataset_images[dataset_id]),
                image_shape=tuple(int(dim) for dim in image.shape[:2]),
                default_radius_px=float(default_radius_px),
                beads=[],
            )
        self.index = 0
        self.selected_index: Optional[int] = None
        self.dirty = False

    @property
    def current_dataset_id(self) -> str:
        return self.dataset_ids[self.index]

    @property
    def current_annotation(self) -> HumanBeadDatasetAnnotations:
        return self.annotations[self.current_dataset_id]

    def set_current_annotation(self, annotation: HumanBeadDatasetAnnotations) -> None:
        self.annotations[self.current_dataset_id] = annotation
        self.dirty = True

    def save(self) -> None:
        write_human_bead_annotations(self.output_path, self.annotations)
        self.dirty = False

    def run(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        def redraw() -> None:
            ax.clear()
            dataset_id = self.current_dataset_id
            ax.imshow(_robust_unit(self.images[dataset_id]), cmap="gray")
            title_suffix = " *unsaved" if self.dirty else ""
            ax.set_title(
                f"{dataset_id} ({self.index + 1}/{len(self.dataset_ids)})"
                "  left-click add/move, +/- radius, delete remove, n/p switch, s save"
                f"{title_suffix}"
            )
            ax.set_axis_off()
            for bead_idx, bead in enumerate(self.current_annotation.beads):
                color = "yellow" if bead_idx == self.selected_index else "red"
                core = plt.Circle((bead.x, bead.y), bead.radius_px, fill=False, color=color, linewidth=1.8)
                support = plt.Circle((bead.x, bead.y), 3.0 * bead.radius_px, fill=False, color="cyan", linewidth=0.8)
                ax.add_patch(core)
                ax.add_patch(support)
                ax.plot([bead.x - 3, bead.x + 3], [bead.y, bead.y], color=color, linewidth=1)
                ax.plot([bead.x, bead.x], [bead.y - 3, bead.y + 3], color=color, linewidth=1)
                ax.text(bead.x + bead.radius_px + 2, bead.y, bead.bead_id, color=color, fontsize=8)
            fig.canvas.draw_idle()

        def nearest_bead(x: float, y: float) -> Optional[int]:
            distances = [
                math.hypot(float(bead.x) - x, float(bead.y) - y)
                for bead in self.current_annotation.beads
            ]
            if not distances:
                return None
            idx = int(np.argmin(distances))
            return idx if distances[idx] <= max(6.0, 1.5 * self.current_annotation.beads[idx].radius_px) else None

        def on_click(event: Any) -> None:
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            ann = self.current_annotation
            selected = nearest_bead(float(event.xdata), float(event.ydata))
            beads = list(ann.beads)
            if selected is None:
                bead_id = f"b{len(beads) + 1:03d}"
                beads.append(HumanBead(bead_id=bead_id, x=float(event.xdata), y=float(event.ydata), radius_px=ann.default_radius_px))
                self.selected_index = len(beads) - 1
            else:
                selected_bead = beads[selected]
                beads[selected] = HumanBead(
                    bead_id=selected_bead.bead_id,
                    x=float(event.xdata),
                    y=float(event.ydata),
                    radius_px=selected_bead.radius_px,
                )
                self.selected_index = selected
            self.set_current_annotation(
                HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads)
            )
            redraw()

        def adjust_radius(delta: float) -> None:
            if self.selected_index is None:
                return
            ann = self.current_annotation
            if self.selected_index >= len(ann.beads):
                return
            beads = list(ann.beads)
            bead = beads[self.selected_index]
            beads[self.selected_index] = HumanBead(
                bead_id=bead.bead_id,
                x=bead.x,
                y=bead.y,
                radius_px=max(0.5, bead.radius_px + delta),
            )
            self.set_current_annotation(
                HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads)
            )
            redraw()

        def on_scroll(event: Any) -> None:
            adjust_radius(0.5 if getattr(event, "button", "") == "up" else -0.5)

        def on_key(event: Any) -> None:
            key = str(getattr(event, "key", "") or "").lower()
            if key == "s":
                self.save()
                redraw()
            elif key in {"n", "right"}:
                self.index = (self.index + 1) % len(self.dataset_ids)
                self.selected_index = None
                redraw()
            elif key in {"p", "left"}:
                self.index = (self.index - 1) % len(self.dataset_ids)
                self.selected_index = None
                redraw()
            elif key in {"+", "="}:
                adjust_radius(0.5)
            elif key in {"-", "_"}:
                adjust_radius(-0.5)
            elif key in {"delete", "backspace"} and self.selected_index is not None:
                ann = self.current_annotation
                beads = [bead for idx, bead in enumerate(ann.beads) if idx != self.selected_index]
                self.selected_index = None
                self.set_current_annotation(
                    HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads)
                )
                redraw()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
        if self.dirty:
            self.save()


def _parse_dataset_image(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-image must use dataset_id=path")
        dataset_id, path_text = value.split("=", 1)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError("--dataset-image dataset_id cannot be empty")
        result[dataset_id] = Path(path_text)
    return result


def _cmd_annotate(args: argparse.Namespace) -> None:
    dataset_images = _parse_dataset_image(args.dataset_image or [])
    if not dataset_images:
        raise ValueError("annotate requires at least one --dataset-image dataset_id=path")
    existing = load_human_bead_annotations(args.output) if Path(args.output).exists() else None
    session = _AnnotationSession(
        dataset_images,
        Path(args.output),
        existing=existing,
        default_radius_px=float(args.default_radius_px),
    )
    session.run()


def _cmd_preview(args: argparse.Namespace) -> None:
    annotations = load_human_bead_annotations(Path(args.annotations))
    output_dir = Path(args.output_dir)
    written = 0
    for dataset_id, annotation in annotations.items():
        if not annotation.image_path:
            continue
        image_path = Path(annotation.image_path)
        if not image_path.is_absolute():
            image_path = Path(args.annotations).parent / image_path
        if not image_path.exists():
            continue
        output = save_human_bead_preview_sheet(output_dir / f"{dataset_id}_human_beads.png", _load_image(image_path), annotation)
        if output is not None:
            written += 1
    print(f"Wrote {written} human bead preview image(s) to {output_dir}")


def _cmd_score_workspace(args: argparse.Namespace) -> None:
    annotations = load_human_bead_annotations(Path(args.annotations))
    if args.dataset_id not in annotations:
        raise ValueError(f"No human bead annotations for dataset {args.dataset_id!r}")
    image, payload, _warnings = load_candidate_workspace(Path(args.workspace))
    result = compute_human_bead_oracle_metrics(
        image,
        annotations[args.dataset_id],
        workspace_payload=payload,
    )
    print(
        write_json_atomic(
            Path(args.output),
            {
                "dataset_id": args.dataset_id,
                "rankable": result.rankable,
                "metrics": result.metrics,
                "diagnostics": result.diagnostics,
                "warnings": result.warnings,
            },
            indent=2,
            sort_keys=True,
            create_backup=False,
        )
        or f"Wrote {args.output}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate and score manual bead-center oracle metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="Interactively click bead centers and radii.")
    annotate.add_argument("--dataset-image", action="append", default=[], help="dataset_id=path/to/image")
    annotate.add_argument("--output", required=True, type=Path)
    annotate.add_argument("--default-radius-px", type=float, default=4.0)
    annotate.set_defaults(func=_cmd_annotate)

    preview = subparsers.add_parser("preview", help="Render bead annotation preview sheets.")
    preview.add_argument("--annotations", required=True, type=Path)
    preview.add_argument("--output-dir", required=True, type=Path)
    preview.set_defaults(func=_cmd_preview)

    score = subparsers.add_parser("score-workspace", help="Score one workspace.npz against one dataset annotation.")
    score.add_argument("--annotations", required=True, type=Path)
    score.add_argument("--dataset-id", required=True)
    score.add_argument("--workspace", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.set_defaults(func=_cmd_score_workspace)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

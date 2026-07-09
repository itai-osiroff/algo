"""Headless trio benchmark for LF sweep outputs against external references.

The tool ranks completed light-field sweep candidates against a processed
validation-camera image, while reporting a processed ThorCam image as the
baseline comparator. Validation and ThorCam images are never fed back into the
LF rectification algorithm; they are only used for post-run selection/reporting.
"""

from __future__ import annotations

import argparse
import heapq
import html
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import cv2 as cv
import numpy as np
from PIL import Image
from scipy import optimize

from moment_images.reliable_io import write_csv_atomic, write_json_atomic, write_text_atomic
from moment_images.unpack_workspace import load_workspace_variables
from moment_images.validate_rectification import load_rectified_reference
from moment_images.sweep_visualization import (
    GENERATION_SCOPE_ACTIVE,
    GENERATION_SCOPE_ALL,
    candidate_rows_for_generation_scope,
    load_sweep_visualization_model,
    run_records_for_generation_scope,
)


PROFILE_NAME = "validation_hybrid_v1"
DEFAULT_SCALE_SEARCH = "0.90:1.10:11"
DEFAULT_GEOMETRIC_SCALE_SEARCH = "0.50:2.00:31"
DEFAULT_REGISTRATION_ROTATION_SEARCH = "-45:45:19"
DEFAULT_TOP_K = 12
DEFAULT_RESIDUAL_TRANSLATION_MAX_PX = 5.0
REGISTRATION_PROFILE_VALIDATE_WORKSPACE = "validate_workspace_transform_v1"
REGISTRATION_PROFILE_MANUAL_SIMILARITY = "manual_similarity_v1"
REGISTRATION_PROFILE_BEAD_GEOMETRIC = "bead_geometric_v2"
REGISTRATION_PROFILE_BEAD_SIMILARITY = "bead_similarity_v1"
REGISTRATION_PROFILE_CENTERED_ECC = "centered_ecc_legacy"
DEFAULT_REGISTRATION_PROFILE = REGISTRATION_PROFILE_BEAD_GEOMETRIC
DEFAULT_REGISTRATION_CALIBRATION_GROUP = "default"
DEFAULT_REGISTRATION_CALIBRATION_MODE = "group"
DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX = 15.0
DEFAULT_REGISTRATION_MIN_MATCHED_BEADS = 3
DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE = 0.10
DEFAULT_REGISTRATION_MAX_DIM = 768
DEFAULT_REGISTRATION_COARSE_MAX_DIM = 384
DEFAULT_REGISTRATION_REFINE_MAX_DIM = 768
DEFAULT_REGISTRATION_TOP_HYPOTHESES = 24
DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N = 12
DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES = 256
DEFAULT_REGISTRATION_DENSE_HYPOTHESES = 64
DEFAULT_REGISTRATION_REFINE_MAX_EVALS = 120
BEAD_LOW_CONFIDENCE_WARNING = "low-confidence bead detection; bead and sharpness terms omitted"


@dataclass(frozen=True)
class BenchmarkConfig:
    sweep_dir: Path
    dataset_id: str
    scene_id: Optional[str] = None
    validation_image: Optional[Path] = None
    thorcam_image: Optional[Path] = None
    output_dir: Optional[Path] = None
    validation_workspace: Optional[Path] = None
    stage: Optional[str] = None
    generation_scope: str = GENERATION_SCOPE_ALL
    anchor_row_id: str = "auto"
    scale_values: Optional[Sequence[float]] = None
    rotation_values: Optional[Sequence[float]] = None
    top_k: int = DEFAULT_TOP_K
    residual_translation_max_px: float = DEFAULT_RESIDUAL_TRANSLATION_MAX_PX
    registration_profile: str = DEFAULT_REGISTRATION_PROFILE
    use_validation_workspace_transforms: bool = False
    registration_annotations: Optional[Path] = None
    registration_calibration_group: str = DEFAULT_REGISTRATION_CALIBRATION_GROUP
    registration_calibration_mode: str = DEFAULT_REGISTRATION_CALIBRATION_MODE
    registration_suspect_residual_px: float = DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX
    registration_min_matched_beads: int = DEFAULT_REGISTRATION_MIN_MATCHED_BEADS
    registration_min_structure_score: float = DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE
    registration_top_hypotheses: int = DEFAULT_REGISTRATION_TOP_HYPOTHESES
    registration_diagnostics_top_n: int = DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N
    registration_coarse_max_dim: int = DEFAULT_REGISTRATION_COARSE_MAX_DIM
    registration_refine_max_dim: int = DEFAULT_REGISTRATION_REFINE_MAX_DIM
    registration_prefilter_hypotheses: int = DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES
    registration_dense_hypotheses: int = DEFAULT_REGISTRATION_DENSE_HYPOTHESES
    registration_refine_max_evals: int = DEFAULT_REGISTRATION_REFINE_MAX_EVALS
    registration_progress: bool = False
    enable_nonrigid_diagnostic: bool = False


@dataclass
class SimilarityRegistrationResult:
    status: str
    aligned_image: np.ndarray
    forward_matrix: np.ndarray
    scale: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    ecc_score: Optional[float]
    ecc_warp_matrix: np.ndarray
    residual_translation_x: float = 0.0
    residual_translation_y: float = 0.0
    residual_translation_norm: float = 0.0
    residual_applied: bool = False
    registration_profile: str = DEFAULT_REGISTRATION_PROFILE
    registration_confidence: Optional[float] = None
    registration_suspect: bool = False
    registration_suspect_reasons: List[str] = field(default_factory=list)
    matched_bead_count: int = 0
    median_bead_error_px: Optional[float] = None
    common_fov_fraction: Optional[float] = None
    anchor_selection_reason: Optional[str] = None
    registration_constraint_level: str = "image_only"
    registration_underconstrained: bool = False
    registration_initializer: str = ""
    registration_calibration_group: Optional[str] = None
    registration_transform_source: Optional[str] = None
    registration_point_count: int = 0
    registration_fit_rms_px: Optional[float] = None
    validation_rankable: bool = True
    ranking_exclusion_reason: Optional[str] = None
    registration_point_pairs: List[Dict[str, List[float]]] = field(default_factory=list)
    registration_refined_score: Optional[float] = None
    registration_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    registration_runtime_seconds: Optional[float] = None
    hypotheses_generated: int = 0
    hypotheses_prefiltered: int = 0
    hypotheses_dense_scored: int = 0
    hypotheses_refined: int = 0
    optimizer_evaluations: int = 0
    registration_timing_seconds: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationMetricResult:
    metrics: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    bead_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class CandidateImageRecord:
    row: Any
    record: Optional[Any]
    image: np.ndarray
    workspace_path: Optional[str]
    preview_path: Optional[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResultRow:
    source_kind: str
    row_id: str
    candidate_id: str
    stage: str
    dataset_id: str
    generation_id: Optional[str]
    strategy: str
    saved_rank: Optional[int]
    validation_rank: Optional[int]
    saved_score: Optional[float]
    metrics: Dict[str, Optional[float]]
    registration: SimilarityRegistrationResult
    workspace_path: Optional[str] = None
    preview_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    best_lf_row_id: Optional[str]
    rows: List[BenchmarkResultRow]
    summary: Dict[str, Any]


@dataclass(frozen=True)
class ReferenceImages:
    validation: np.ndarray
    thorcam: np.ndarray
    validation_source: str
    thorcam_source: str
    warnings: List[str] = field(default_factory=list)
    workspace_transform_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationMetricReference:
    validation_unit: np.ndarray
    structure_channel: np.ndarray
    bead_centers: List[Tuple[float, float]]


@dataclass(frozen=True)
class RegistrationBeadCandidate:
    x: float
    y: float
    radius: float
    response: float
    confidence: float


@dataclass(frozen=True)
class GeometricReferencePrepared:
    reference_unit: np.ndarray
    reference_structure: np.ndarray
    reference_bead_channel: np.ndarray
    reference_beads: List[RegistrationBeadCandidate]
    reference_scale: float


@dataclass(frozen=True)
class GeometricReferenceCache:
    coarse: GeometricReferencePrepared
    refine: GeometricReferencePrepared


@dataclass(frozen=True)
class GeometricMovingPrepared:
    moving_unit: np.ndarray
    moving_structure: np.ndarray
    moving_bead_channel: np.ndarray
    moving_beads: List[RegistrationBeadCandidate]
    moving_scale: float


@dataclass(frozen=True)
class GeometricRegistrationPrepared:
    reference_unit: np.ndarray
    moving_unit: np.ndarray
    reference_structure: np.ndarray
    moving_structure: np.ndarray
    reference_bead_channel: np.ndarray
    moving_bead_channel: np.ndarray
    reference_beads: List[RegistrationBeadCandidate]
    moving_beads: List[RegistrationBeadCandidate]
    reference_scale: float
    moving_scale: float


@dataclass(frozen=True)
class RegistrationHypothesisSeed:
    matrix: np.ndarray
    initializer: str
    constraint_level: str
    score: float
    matched_bead_count: int
    median_bead_error_px: Optional[float]
    common_fov_fraction: float


@dataclass(frozen=True)
class RegistrationHypothesis:
    matrix: np.ndarray
    initializer: str
    constraint_level: str
    score: float
    matched_bead_count: int
    median_bead_error_px: Optional[float]
    common_fov_fraction: float
    refined_score: Optional[float] = None
    optimizer_evaluations: int = 0


@dataclass(frozen=True)
class ManualSimilarityFit:
    matrix: np.ndarray
    scale: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    rms_error_px: float
    point_count: int


@dataclass(frozen=True)
class ManualAnnotationRecord:
    scene_id: str
    dataset_id: str
    source_kind: str
    calibration_group: str
    reference_shape: Tuple[int, int]
    moving_shape: Tuple[int, int]
    reference_points: np.ndarray
    moving_points: np.ndarray
    anchor_row_id: Optional[str] = None


@dataclass(frozen=True)
class ManualResolvedTransform:
    matrix: np.ndarray
    calibration_group: str
    transform_source: str
    initializer: str
    point_count: int
    fit_rms_px: Optional[float]
    underconstrained: bool
    source_annotation_scene_id: Optional[str]
    point_pairs: List[Dict[str, List[float]]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_scale_search(spec: str) -> List[float]:
    """Parse ``start:stop:count`` into an inclusive scale grid."""

    values = _parse_linear_search(spec, label="scale search")
    if any(value <= 0.0 for value in values):
        raise ValueError("scale search bounds must be positive")
    return values


def parse_rotation_search(spec: str) -> List[float]:
    """Parse ``start:stop:count`` into an inclusive rotation grid in degrees."""

    return _parse_linear_search(spec, label="rotation search")


def _parse_linear_search(spec: str, *, label: str) -> List[float]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"{label} must use start:stop:count syntax")
    start = float(parts[0])
    stop = float(parts[1])
    count = int(parts[2])
    if count <= 0:
        raise ValueError(f"{label} count must be positive")
    if not (math.isfinite(start) and math.isfinite(stop)):
        raise ValueError(f"{label} bounds must be finite")
    if count == 1:
        return [start]
    return [float(value) for value in np.linspace(start, stop, count)]


def fit_manual_similarity_transform(
    moving_points: Sequence[Sequence[float]],
    reference_points: Sequence[Sequence[float]],
) -> ManualSimilarityFit:
    """Fit a shear-free moving-to-reference similarity transform from point pairs."""

    moving = _as_point_array(moving_points, label="moving_points")
    reference = _as_point_array(reference_points, label="reference_points")
    if moving.shape != reference.shape:
        raise ValueError("moving_points and reference_points must have the same shape")
    if moving.shape[0] < 2:
        raise ValueError("at least two point pairs are required for a full similarity fit")
    moving_mean = np.mean(moving, axis=0)
    reference_mean = np.mean(reference, axis=0)
    moving_centered = moving - moving_mean
    reference_centered = reference - reference_mean
    variance = float(np.sum(moving_centered * moving_centered))
    if variance <= 1e-12:
        raise ValueError("manual similarity fit requires non-degenerate moving points")
    covariance = moving_centered.T @ reference_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("manual similarity fit produced an invalid scale")
    linear = scale * rotation
    translation = reference_mean - (linear @ moving_mean)
    matrix = np.array(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
        ],
        dtype=np.float64,
    )
    transformed = _transform_points(matrix, moving)
    rms_error = float(np.sqrt(np.mean(np.sum((transformed - reference) ** 2, axis=1))))
    fit_scale, rotation_deg, tx, ty = _decompose_similarity(matrix)
    return ManualSimilarityFit(
        matrix=matrix,
        scale=fit_scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        rms_error_px=rms_error,
        point_count=int(moving.shape[0]),
    )


def _as_point_array(points: Sequence[Sequence[float]], *, label: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{label} must be an Nx2 array of x,y points")
    if arr.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return arr


def _load_manual_annotation_records(path: Path) -> List[ManualAnnotationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_annotations = payload.get("annotations") if isinstance(payload, Mapping) else None
    if not isinstance(raw_annotations, list):
        raise ValueError("manual registration annotations must contain an 'annotations' list")
    records: List[ManualAnnotationRecord] = []
    for index, raw in enumerate(raw_annotations, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"manual annotation {index} must be an object")
        points_payload = raw.get("points")
        if not isinstance(points_payload, list) or not points_payload:
            raise ValueError(f"manual annotation {index} must contain a non-empty points list")
        reference_points: List[Sequence[float]] = []
        moving_points: List[Sequence[float]] = []
        for point_index, point in enumerate(points_payload, start=1):
            if not isinstance(point, Mapping):
                raise ValueError(f"manual annotation {index} point {point_index} must be an object")
            reference_points.append(cast(Sequence[float], point.get("reference")))
            moving_points.append(cast(Sequence[float], point.get("moving")))
        records.append(
            ManualAnnotationRecord(
                scene_id=str(raw.get("scene_id", "")),
                dataset_id=str(raw.get("dataset_id", "")),
                source_kind=str(raw.get("source_kind", "")).strip().lower(),
                calibration_group=str(raw.get("calibration_group", DEFAULT_REGISTRATION_CALIBRATION_GROUP)),
                reference_shape=_parse_annotation_shape(raw.get("reference_shape"), label=f"annotation {index} reference_shape"),
                moving_shape=_parse_annotation_shape(raw.get("moving_shape"), label=f"annotation {index} moving_shape"),
                reference_points=_as_point_array(reference_points, label=f"annotation {index} reference points"),
                moving_points=_as_point_array(moving_points, label=f"annotation {index} moving points"),
                anchor_row_id=_optional_str(raw.get("anchor_row_id")),
            )
        )
    return records


def _parse_annotation_shape(value: Any, *, label: str) -> Tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{label} must be [height, width]")
    height = int(value[0])
    width = int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} must be positive")
    return height, width


def estimate_similarity_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    scale_values: Sequence[float],
    rotation_values: Optional[Sequence[float]] = None,
    registration_top_hypotheses: int = DEFAULT_REGISTRATION_TOP_HYPOTHESES,
    registration_diagnostics_top_n: int = DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N,
    registration_coarse_max_dim: int = DEFAULT_REGISTRATION_COARSE_MAX_DIM,
    registration_refine_max_dim: int = DEFAULT_REGISTRATION_REFINE_MAX_DIM,
    registration_prefilter_hypotheses: int = DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES,
    registration_dense_hypotheses: int = DEFAULT_REGISTRATION_DENSE_HYPOTHESES,
    registration_refine_max_evals: int = DEFAULT_REGISTRATION_REFINE_MAX_EVALS,
    registration_profile: str = DEFAULT_REGISTRATION_PROFILE,
    registration_suspect_residual_px: float = DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX,
    registration_min_matched_beads: int = DEFAULT_REGISTRATION_MIN_MATCHED_BEADS,
    registration_min_structure_score: float = DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE,
    geometric_reference_cache: Optional[GeometricReferenceCache] = None,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    """Register ``moving`` onto ``reference`` using a shear-free similarity transform."""

    ref = _as_float_2d(reference)
    mov = _as_float_2d(moving)
    if ref.size == 0 or mov.size == 0:
        raise ValueError("reference and moving images must be non-empty")

    profile = _normalize_registration_profile(registration_profile)
    if profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        raise ValueError("manual_similarity_v1 requires registration_annotations via run_benchmark")
    if profile == REGISTRATION_PROFILE_VALIDATE_WORKSPACE:
        raise ValueError("validate_workspace_transform_v1 requires validation_workspace via run_benchmark")
    if profile == REGISTRATION_PROFILE_CENTERED_ECC:
        return _estimate_centered_ecc_registration(
            ref,
            mov,
            scale_values=scale_values,
            registration_profile=profile,
            registration_suspect_residual_px=registration_suspect_residual_px,
            registration_min_matched_beads=registration_min_matched_beads,
            registration_min_structure_score=registration_min_structure_score,
            quality_reference_cache=quality_reference_cache,
        )
    if profile == REGISTRATION_PROFILE_BEAD_GEOMETRIC:
        return _estimate_bead_geometric_registration(
            ref,
            mov,
            scale_values=scale_values,
            rotation_values=parse_rotation_search(DEFAULT_REGISTRATION_ROTATION_SEARCH)
            if rotation_values is None
            else rotation_values,
            registration_profile=profile,
            registration_suspect_residual_px=registration_suspect_residual_px,
            registration_min_matched_beads=registration_min_matched_beads,
            registration_min_structure_score=registration_min_structure_score,
            registration_top_hypotheses=registration_top_hypotheses,
            registration_diagnostics_top_n=registration_diagnostics_top_n,
            registration_coarse_max_dim=registration_coarse_max_dim,
            registration_refine_max_dim=registration_refine_max_dim,
            registration_prefilter_hypotheses=registration_prefilter_hypotheses,
            registration_dense_hypotheses=registration_dense_hypotheses,
            registration_refine_max_evals=registration_refine_max_evals,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=quality_reference_cache,
        )
    return _estimate_bead_similarity_registration(
        ref,
        mov,
        scale_values=scale_values,
        registration_profile=profile,
        registration_suspect_residual_px=registration_suspect_residual_px,
        registration_min_matched_beads=registration_min_matched_beads,
        registration_min_structure_score=registration_min_structure_score,
        quality_reference_cache=quality_reference_cache,
    )


def _estimate_centered_ecc_registration(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    scale_values: Sequence[float],
    registration_profile: str,
    registration_suspect_residual_px: float,
    registration_min_matched_beads: int,
    registration_min_structure_score: float,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    """Register ``moving`` onto ``reference`` using centered scale sweep + Euclidean ECC."""

    ref_norm = _normalize_for_registration(ref)
    best: Optional[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = None
    warnings: List[str] = []
    for scale in scale_values:
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            warnings.append(f"invalid scale skipped: {scale}")
            continue
        pre_matrix = _pre_scale_matrix(mov.shape, ref.shape, float(scale))
        canvas = cv.warpAffine(
            mov,
            pre_matrix,
            (ref.shape[1], ref.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )
        canvas_norm = _normalize_for_registration(canvas)
        if float(np.std(canvas_norm)) <= 1e-8 or float(np.std(ref_norm)) <= 1e-8:
            score = -float(np.mean(np.abs(ref_norm - canvas_norm)))
            warp = np.eye(2, 3, dtype=np.float32)
            aligned = canvas
        else:
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 250, 1e-6)
            try:
                cc, warp = cv.findTransformECC(
                    ref_norm,
                    canvas_norm,
                    warp,
                    cv.MOTION_EUCLIDEAN,
                    criteria,
                    None,
                    5,
                )
                score = float(cc)
                aligned = cv.warpAffine(
                    canvas,
                    warp,
                    (ref.shape[1], ref.shape[0]),
                    flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0.0,
                )
            except cv.error as exc:
                warnings.append(f"ECC failed for scale {scale:.6g}: {exc.__class__.__name__}")
                continue
        if best is None or score > best[0]:
            best = (score, float(scale), pre_matrix, warp.copy(), aligned)

    if best is None:
        pre_matrix = _pre_scale_matrix(mov.shape, ref.shape, 1.0)
        aligned = cv.warpAffine(
            mov,
            pre_matrix,
            (ref.shape[1], ref.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )
        forward = pre_matrix.astype(np.float64)
        scale, rotation_deg, tx, ty = _decompose_similarity(forward)
        result = SimilarityRegistrationResult(
            status="failed",
            aligned_image=aligned.astype(np.float32),
            forward_matrix=forward,
            scale=scale,
            rotation_deg=rotation_deg,
            translation_x=tx,
            translation_y=ty,
            ecc_score=None,
            ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
            registration_profile=registration_profile,
            warnings=warnings or ["similarity registration failed"],
        )
        return _with_registration_quality(
            result,
            ref,
            mov.shape,
            scale_values=scale_values,
            residual_suspect_px=registration_suspect_residual_px,
            min_matched_beads=registration_min_matched_beads,
            min_structure_score=registration_min_structure_score,
            reference_cache=quality_reference_cache,
        )

    score, best_scale, pre_matrix, warp, aligned = best
    forward = _combine_forward_matrix(pre_matrix, warp)
    scale, rotation_deg, tx, ty = _decompose_similarity(forward)
    result = SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=best_scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=score,
        ecc_warp_matrix=warp.astype(np.float32),
        registration_profile=registration_profile,
        warnings=warnings,
    )
    return _with_registration_quality(
        result,
        ref,
        mov.shape,
        scale_values=scale_values,
        residual_suspect_px=registration_suspect_residual_px,
        min_matched_beads=registration_min_matched_beads,
        min_structure_score=registration_min_structure_score,
        reference_cache=quality_reference_cache,
    )


def _estimate_bead_geometric_registration(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    scale_values: Sequence[float],
    rotation_values: Sequence[float],
    registration_profile: str,
    registration_suspect_residual_px: float,
    registration_min_matched_beads: int,
    registration_min_structure_score: float,
    registration_top_hypotheses: int,
    registration_diagnostics_top_n: int,
    registration_coarse_max_dim: int,
    registration_refine_max_dim: int,
    registration_prefilter_hypotheses: int,
    registration_dense_hypotheses: int,
    registration_refine_max_evals: int,
    geometric_reference_cache: Optional[GeometricReferenceCache],
    quality_reference_cache: Optional[ValidationMetricReference],
) -> SimilarityRegistrationResult:
    started = time.perf_counter()
    timings: Dict[str, float] = {}

    phase = time.perf_counter()
    reference_cache = geometric_reference_cache or GeometricReferenceCache(
        coarse=_prepare_geometric_reference(ref, max_dim=registration_coarse_max_dim),
        refine=_prepare_geometric_reference(ref, max_dim=registration_refine_max_dim),
    )
    coarse_moving = _prepare_geometric_moving(mov, max_dim=registration_coarse_max_dim)
    refine_moving = _prepare_geometric_moving(mov, max_dim=registration_refine_max_dim)
    coarse_prepared = _combine_geometric_prepared(reference_cache.coarse, coarse_moving)
    refine_prepared = _combine_geometric_prepared(reference_cache.refine, refine_moving)
    timings["preparation"] = time.perf_counter() - phase

    coarse_scale_values = [
        float(value) * coarse_prepared.reference_scale / coarse_prepared.moving_scale for value in scale_values
    ]
    phase = time.perf_counter()
    seeds, hypotheses_generated = _generate_geometric_hypothesis_seeds(
        coarse_prepared,
        scale_values=coarse_scale_values,
        rotation_values=rotation_values,
        prefilter_hypotheses=max(1, int(registration_prefilter_hypotheses)),
    )
    timings["seed_generation"] = time.perf_counter() - phase

    if not seeds:
        fallback = _estimate_bead_similarity_registration(
            ref,
            mov,
            scale_values=scale_values,
            registration_profile=REGISTRATION_PROFILE_BEAD_SIMILARITY,
            registration_suspect_residual_px=registration_suspect_residual_px,
            registration_min_matched_beads=registration_min_matched_beads,
            registration_min_structure_score=registration_min_structure_score,
            quality_reference_cache=quality_reference_cache,
        )
        fallback.registration_profile = registration_profile
        fallback.registration_constraint_level = "image_only"
        fallback.registration_underconstrained = True
        fallback.registration_initializer = "dense_fallback"
        return _attach_registration_runtime(
            fallback,
            started=started,
            timings=timings,
            hypotheses_generated=hypotheses_generated,
            hypotheses_prefiltered=0,
            hypotheses_dense_scored=0,
            hypotheses_refined=0,
            optimizer_evaluations=0,
        )

    phase = time.perf_counter()
    dense_limit = max(1, min(int(registration_dense_hypotheses), len(seeds)))
    dense_hypotheses = [
        _score_geometric_seed_dense(seed, coarse_prepared)
        for seed in seeds[:dense_limit]
    ]
    dense_hypotheses.sort(key=lambda item: item.score, reverse=True)
    timings["dense_scoring"] = time.perf_counter() - phase

    phase = time.perf_counter()
    refine_limit = max(1, min(int(registration_top_hypotheses), len(dense_hypotheses)))
    refine_inputs = [
        _hypothesis_to_prepared_resolution(hypothesis, coarse_prepared, refine_prepared)
        for hypothesis in dense_hypotheses[:refine_limit]
    ]
    refine_scale_values = [
        float(value) * refine_prepared.reference_scale / refine_prepared.moving_scale for value in scale_values
    ]
    refined = [
        _refine_geometric_hypothesis(
            hypothesis,
            refine_prepared,
            scale_bounds=(min(refine_scale_values), max(refine_scale_values)),
            rotation_bounds=(min(rotation_values), max(rotation_values)),
            max_evals=max(1, int(registration_refine_max_evals)),
        )
        for hypothesis in refine_inputs
    ]
    optimizer_evaluations = sum(item.optimizer_evaluations for item in refined)
    timings["refinement"] = time.perf_counter() - phase

    refined.sort(key=lambda item: item.refined_score if item.refined_score is not None else item.score, reverse=True)
    best = refined[0]

    phase = time.perf_counter()
    forward = _scale_affine_between_resolutions(
        best.matrix,
        ref_scale=refine_prepared.reference_scale,
        mov_scale=refine_prepared.moving_scale,
    )
    aligned = cv.warpAffine(
        mov,
        forward.astype(np.float32),
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    timings["final_warp"] = time.perf_counter() - phase

    scale, rotation_deg, tx, ty = _decompose_similarity(forward)
    result = SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=best.score,
        ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
        registration_profile=registration_profile,
        registration_constraint_level=best.constraint_level,
        registration_underconstrained=best.constraint_level != "full",
        registration_initializer=best.initializer,
        registration_refined_score=best.refined_score,
        registration_hypotheses=[
            _hypothesis_summary(item, refine_prepared, index)
            for index, item in enumerate(refined[: max(0, registration_diagnostics_top_n)], start=1)
        ],
    )
    phase = time.perf_counter()
    result = _with_registration_quality(
        result,
        ref,
        mov.shape,
        scale_values=scale_values,
        residual_suspect_px=registration_suspect_residual_px,
        min_matched_beads=registration_min_matched_beads,
        min_structure_score=registration_min_structure_score,
        reference_cache=quality_reference_cache,
    )
    timings["quality"] = time.perf_counter() - phase
    return _attach_registration_runtime(
        result,
        started=started,
        timings=timings,
        hypotheses_generated=hypotheses_generated,
        hypotheses_prefiltered=len(seeds),
        hypotheses_dense_scored=len(dense_hypotheses),
        hypotheses_refined=len(refined),
        optimizer_evaluations=optimizer_evaluations,
    )


def _estimate_manual_similarity_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    config: BenchmarkConfig,
    source_kind: str,
    annotations: Sequence[ManualAnnotationRecord],
    anchor_record: Optional[CandidateImageRecord] = None,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    started = time.perf_counter()
    transform = _resolve_manual_transform(
        annotations,
        config=config,
        source_kind=source_kind,
        reference_shape=reference.shape[:2],
        moving_shape=moving.shape[:2],
        anchor_record=anchor_record,
    )
    aligned = cv.warpAffine(
        _as_float_2d(moving),
        np.asarray(transform.matrix, dtype=np.float32),
        (reference.shape[1], reference.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    scale, rotation_deg, tx, ty = _decompose_similarity(transform.matrix)
    result = SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=np.asarray(transform.matrix, dtype=np.float64),
        scale=scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=None,
        ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
        registration_profile=REGISTRATION_PROFILE_MANUAL_SIMILARITY,
        registration_constraint_level="single_point" if transform.underconstrained else "full",
        registration_underconstrained=transform.underconstrained,
        registration_initializer=transform.initializer,
        registration_calibration_group=transform.calibration_group,
        registration_transform_source=transform.transform_source,
        registration_point_count=transform.point_count,
        registration_fit_rms_px=transform.fit_rms_px,
        registration_point_pairs=transform.point_pairs,
        registration_hypotheses=[
            {
                "rank": 1,
                "initializer": transform.initializer,
                "constraint_level": "single_point" if transform.underconstrained else "full",
                "score": None,
                "refined_score": None,
                "matched_bead_count": "",
                "median_bead_error_px": transform.fit_rms_px,
                "common_fov_fraction": "",
                "optimizer_evaluations": 0,
                "scale": scale,
                "rotation_deg": rotation_deg,
                "translation_x": tx,
                "translation_y": ty,
            }
        ],
        warnings=transform.warnings,
    )
    result = _with_registration_quality(
        result,
        reference,
        moving.shape[:2],
        scale_values=cast(Sequence[float], config.scale_values),
        residual_suspect_px=config.registration_suspect_residual_px,
        min_matched_beads=config.registration_min_matched_beads,
        min_structure_score=config.registration_min_structure_score,
        reference_cache=quality_reference_cache,
    )
    result.validation_rankable, result.ranking_exclusion_reason = _registration_rankability(result)
    return _attach_registration_runtime(
        result,
        started=started,
        timings={"manual_resolution": max(0.0, time.perf_counter() - started)},
        hypotheses_generated=0,
        hypotheses_prefiltered=0,
        hypotheses_dense_scored=0,
        hypotheses_refined=0,
        optimizer_evaluations=0,
    )


def _workspace_grid_identity_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    config: BenchmarkConfig,
    initializer: str,
    transform_source: str,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    ref = _as_float_2d(reference)
    mov = _as_float_2d(moving)
    matrix = np.eye(2, 3, dtype=np.float64)
    aligned = cv.warpAffine(
        mov,
        matrix.astype(np.float32),
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    result = SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=matrix,
        scale=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        ecc_score=None,
        ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
        registration_profile=REGISTRATION_PROFILE_VALIDATE_WORKSPACE,
        registration_constraint_level="workspace_transform",
        registration_underconstrained=False,
        registration_initializer=initializer,
        registration_transform_source=transform_source,
        validation_rankable=True,
    )
    result = _with_registration_quality(
        result,
        ref,
        mov.shape[:2],
        scale_values=[1.0],
        residual_suspect_px=config.registration_suspect_residual_px,
        min_matched_beads=config.registration_min_matched_beads,
        min_structure_score=config.registration_min_structure_score,
        reference_cache=quality_reference_cache,
    )
    result.validation_rankable, result.ranking_exclusion_reason = _registration_rankability(result)
    return result


def _workspace_transform_source_for(references: ReferenceImages, source_kind: str) -> str:
    sources = references.workspace_transform_info.get("sources", {})
    source_info: Mapping[str, Any] = {}
    if isinstance(sources, Mapping):
        raw = sources.get(source_kind, {})
        if isinstance(raw, Mapping):
            source_info = raw
    return str(source_info.get("transform_source") or f"{source_kind}_alignment_final_warp_matrix")


def _resolve_manual_transform(
    annotations: Sequence[ManualAnnotationRecord],
    *,
    config: BenchmarkConfig,
    source_kind: str,
    reference_shape: Tuple[int, int],
    moving_shape: Tuple[int, int],
    anchor_record: Optional[CandidateImageRecord],
) -> ManualResolvedTransform:
    source = str(source_kind).strip().lower()
    if source not in {"lf", "thorcam"}:
        raise ValueError(f"unsupported manual registration source kind: {source_kind!r}")
    group = str(config.registration_calibration_group or DEFAULT_REGISTRATION_CALIBRATION_GROUP)
    mode = _normalize_registration_calibration_mode(config.registration_calibration_mode)
    scene_id = str(config.scene_id or config.dataset_id)
    dataset_id = str(config.dataset_id)
    anchor_ids = _anchor_identity_set(anchor_record)
    group_records = [
        record
        for record in annotations
        if record.source_kind == source and record.calibration_group == group and _manual_annotation_anchor_matches(record, source, anchor_ids)
    ]
    if not group_records:
        raise ValueError(f"manual_similarity_v1 has no annotations for source={source!r}, group={group!r}")
    mismatched = [
        record
        for record in group_records
        if record.reference_shape != tuple(reference_shape) or record.moving_shape != tuple(moving_shape)
    ]
    if mismatched:
        examples = ", ".join(
            f"{record.scene_id or record.dataset_id}:ref={record.reference_shape}:mov={record.moving_shape}"
            for record in mismatched[:3]
        )
        raise ValueError(
            "manual_similarity_v1 incompatible annotation shapes for "
            f"source={source!r}, group={group!r}; expected ref={tuple(reference_shape)}, "
            f"moving={tuple(moving_shape)}; got {examples}"
        )
    full_records = [record for record in group_records if len(record.moving_points) >= 2]
    if not full_records:
        raise ValueError(
            f"manual_similarity_v1 needs at least one full annotation with two point pairs for "
            f"source={source!r}, group={group!r}"
        )
    scene_records = [
        record
        for record in group_records
        if _manual_annotation_matches_scene(record, scene_id=scene_id, dataset_id=dataset_id)
    ]
    if mode == "scene":
        scene_full = [record for record in scene_records if len(record.moving_points) >= 2]
        if scene_full:
            fit = _fit_manual_record(scene_full[-1])
            return _manual_transform_from_fit(
                fit,
                calibration_group=group,
                transform_source="scene_annotation",
                initializer="manual_points",
                source_annotation_scene_id=scene_full[-1].scene_id,
                point_pairs=_point_pairs_payload(scene_full[-1]),
            )

    group_transform = _manual_group_transform(full_records, calibration_group=group)
    scene_single = [record for record in scene_records if len(record.moving_points) == 1]
    if scene_single:
        record = scene_single[-1]
        matrix = _manual_single_point_translation_override(group_transform.matrix, record)
        return ManualResolvedTransform(
            matrix=matrix,
            calibration_group=group,
            transform_source="group_calibration_with_scene_translation",
            initializer="group_plus_single_point_translation",
            point_count=1,
            fit_rms_px=0.0,
            underconstrained=True,
            source_annotation_scene_id=record.scene_id,
            point_pairs=_point_pairs_payload(record),
            warnings=["manual registration is underconstrained by one point; group scale/rotation reused"],
        )
    return group_transform


def _manual_group_transform(
    records: Sequence[ManualAnnotationRecord],
    *,
    calibration_group: str,
) -> ManualResolvedTransform:
    fits = [_fit_manual_record(record) for record in records]
    scales = [fit.scale for fit in fits]
    rotations = [fit.rotation_deg for fit in fits]
    tx_values = [fit.translation_x for fit in fits]
    ty_values = [fit.translation_y for fit in fits]
    scale = float(math.exp(float(np.median(np.log(np.maximum(scales, 1e-12))))))
    rotation = float(np.median(rotations))
    tx = float(np.median(tx_values))
    ty = float(np.median(ty_values))
    matrix = _similarity_matrix_from_values(scale, rotation, tx, ty)
    return ManualResolvedTransform(
        matrix=matrix,
        calibration_group=calibration_group,
        transform_source="group_calibration",
        initializer="group_calibration_median",
        point_count=int(sum(fit.point_count for fit in fits)),
        fit_rms_px=float(np.median([fit.rms_error_px for fit in fits])),
        underconstrained=False,
        source_annotation_scene_id=None if len(records) != 1 else records[0].scene_id,
        point_pairs=[point for record in records for point in _point_pairs_payload(record)],
    )


def _manual_transform_from_fit(
    fit: ManualSimilarityFit,
    *,
    calibration_group: str,
    transform_source: str,
    initializer: str,
    source_annotation_scene_id: Optional[str],
    point_pairs: List[Dict[str, List[float]]],
) -> ManualResolvedTransform:
    return ManualResolvedTransform(
        matrix=fit.matrix,
        calibration_group=calibration_group,
        transform_source=transform_source,
        initializer=initializer,
        point_count=fit.point_count,
        fit_rms_px=fit.rms_error_px,
        underconstrained=False,
        source_annotation_scene_id=source_annotation_scene_id,
        point_pairs=point_pairs,
    )


def _fit_manual_record(record: ManualAnnotationRecord) -> ManualSimilarityFit:
    return fit_manual_similarity_transform(record.moving_points, record.reference_points)


def _manual_single_point_translation_override(
    group_matrix: np.ndarray,
    record: ManualAnnotationRecord,
) -> np.ndarray:
    moving = np.asarray(record.moving_points[0], dtype=np.float64)
    reference = np.asarray(record.reference_points[0], dtype=np.float64)
    matrix = np.asarray(group_matrix, dtype=np.float64).copy()
    linear = matrix[:, :2]
    matrix[:, 2] = reference - (linear @ moving)
    return matrix


def _similarity_matrix_from_values(scale: float, rotation_deg: float, tx: float, ty: float) -> np.ndarray:
    theta = math.radians(float(rotation_deg))
    cos_t = math.cos(theta) * float(scale)
    sin_t = math.sin(theta) * float(scale)
    return np.array([[cos_t, -sin_t, float(tx)], [sin_t, cos_t, float(ty)]], dtype=np.float64)


def _point_pairs_payload(record: ManualAnnotationRecord) -> List[Dict[str, List[float]]]:
    return [
        {
            "reference": [float(reference[0]), float(reference[1])],
            "moving": [float(moving[0]), float(moving[1])],
        }
        for reference, moving in zip(record.reference_points, record.moving_points)
    ]


def _anchor_identity_set(anchor_record: Optional[CandidateImageRecord]) -> set[str]:
    if anchor_record is None:
        return set()
    return {
        text
        for text in [
            str(getattr(anchor_record.row, "row_id", "")),
            str(getattr(anchor_record.row, "candidate_id", "")),
        ]
        if text
    }


def _manual_annotation_anchor_matches(record: ManualAnnotationRecord, source_kind: str, anchor_ids: set[str]) -> bool:
    if source_kind != "lf" or not record.anchor_row_id or not anchor_ids:
        return True
    return record.anchor_row_id in anchor_ids


def _manual_annotation_matches_scene(
    record: ManualAnnotationRecord,
    *,
    scene_id: str,
    dataset_id: str,
) -> bool:
    return record.scene_id == scene_id or record.dataset_id == dataset_id


def _normalize_registration_calibration_mode(value: str) -> str:
    text = str(value or DEFAULT_REGISTRATION_CALIBRATION_MODE).strip().lower()
    if text in {"group", "scene"}:
        return text
    raise ValueError("registration_calibration_mode must be 'group' or 'scene'")


def _registration_rankability(result: SimilarityRegistrationResult) -> Tuple[bool, Optional[str]]:
    if result.status != "aligned":
        return False, f"registration_status={result.status}"
    if result.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY and not result.registration_transform_source:
        return False, "manual_transform_missing"
    return True, None


def _prepare_geometric_reference(ref: np.ndarray, *, max_dim: int) -> GeometricReferencePrepared:
    ref_small, ref_scale = _downsample_for_registration(ref, max_dim=max(1, int(max_dim)))
    ref_unit = _robust_unit(ref_small)
    ref_bead = _bead_registration_channel(ref_unit)
    return GeometricReferencePrepared(
        reference_unit=ref_unit,
        reference_structure=_structure_channel(ref_unit),
        reference_bead_channel=ref_bead,
        reference_beads=_detect_registration_beads(ref_bead),
        reference_scale=ref_scale,
    )


def _attach_registration_runtime(
    result: SimilarityRegistrationResult,
    *,
    started: float,
    timings: Mapping[str, float],
    hypotheses_generated: int,
    hypotheses_prefiltered: int,
    hypotheses_dense_scored: int,
    hypotheses_refined: int,
    optimizer_evaluations: int,
) -> SimilarityRegistrationResult:
    total = max(0.0, time.perf_counter() - float(started))
    timing_payload = {str(key): float(value) for key, value in timings.items()}
    timing_payload["total"] = total
    result.registration_runtime_seconds = total
    result.hypotheses_generated = int(hypotheses_generated)
    result.hypotheses_prefiltered = int(hypotheses_prefiltered)
    result.hypotheses_dense_scored = int(hypotheses_dense_scored)
    result.hypotheses_refined = int(hypotheses_refined)
    result.optimizer_evaluations = int(optimizer_evaluations)
    result.registration_timing_seconds = timing_payload
    return result


def _prepare_geometric_moving(mov: np.ndarray, *, max_dim: int) -> GeometricMovingPrepared:
    mov_small, mov_scale = _downsample_for_registration(mov, max_dim=max(1, int(max_dim)))
    mov_unit = _robust_unit(mov_small)
    mov_bead = _bead_registration_channel(mov_unit)
    return GeometricMovingPrepared(
        moving_unit=mov_unit,
        moving_structure=_structure_channel(mov_unit),
        moving_bead_channel=mov_bead,
        moving_beads=_detect_registration_beads(mov_bead),
        moving_scale=mov_scale,
    )


def _combine_geometric_prepared(
    reference: GeometricReferencePrepared,
    moving: GeometricMovingPrepared,
) -> GeometricRegistrationPrepared:
    return GeometricRegistrationPrepared(
        reference_unit=reference.reference_unit,
        moving_unit=moving.moving_unit,
        reference_structure=reference.reference_structure,
        moving_structure=moving.moving_structure,
        reference_bead_channel=reference.reference_bead_channel,
        moving_bead_channel=moving.moving_bead_channel,
        reference_beads=reference.reference_beads,
        moving_beads=moving.moving_beads,
        reference_scale=reference.reference_scale,
        moving_scale=moving.moving_scale,
    )


def _prepare_geometric_registration(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    max_dim: int = DEFAULT_REGISTRATION_REFINE_MAX_DIM,
) -> GeometricRegistrationPrepared:
    return _combine_geometric_prepared(
        _prepare_geometric_reference(ref, max_dim=max_dim),
        _prepare_geometric_moving(mov, max_dim=max_dim),
    )


def _prepare_geometric_reference_cache(ref: np.ndarray, config: BenchmarkConfig) -> GeometricReferenceCache:
    return GeometricReferenceCache(
        coarse=_prepare_geometric_reference(ref, max_dim=config.registration_coarse_max_dim),
        refine=_prepare_geometric_reference(ref, max_dim=config.registration_refine_max_dim),
    )


def _bead_registration_channel(image: np.ndarray) -> np.ndarray:
    unit = _robust_unit(image)
    background = cv.GaussianBlur(unit, (0, 0), 10.0)
    highpass = np.maximum(unit - background, 0.0)
    dog_small = cv.GaussianBlur(unit, (0, 0), 1.0)
    dog_large = cv.GaussianBlur(unit, (0, 0), 4.0)
    dog = np.maximum(dog_small - dog_large, 0.0)
    log = np.abs(cv.Laplacian(cv.GaussianBlur(unit, (0, 0), 1.2), cv.CV_32F))
    return _robust_unit(0.50 * dog + 0.35 * highpass + 0.15 * log)


def _detect_registration_beads(channel: np.ndarray, *, max_candidates: int = 80) -> List[RegistrationBeadCandidate]:
    arr = _robust_unit(channel)
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return []
    smooth = cv.GaussianBlur(arr, (0, 0), 0.8)
    threshold = max(0.10, float(np.percentile(smooth, 98.7)))
    dilated = cv.dilate(smooth, np.ones((5, 5), dtype=np.uint8))
    peak_mask = (smooth >= threshold) & (smooth >= dilated - 1e-6)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        np.asarray(peak_mask, dtype=np.uint8),
        connectivity=8,
    )
    candidates: List[RegistrationBeadCandidate] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area < 1:
            continue
        cx, cy = centroids[label]
        x = int(round(float(cx)))
        y = int(round(float(cy)))
        radius = 7
        x0 = max(0, x - radius)
        x1 = min(arr.shape[1], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(arr.shape[0], y + radius + 1)
        patch = smooth[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        weights = np.maximum(patch - threshold * 0.5, 0.0)
        total = float(np.sum(weights))
        if total > 1e-9:
            yy, xx = np.indices(weights.shape)
            cx = x0 + float(np.sum(xx * weights) / total)
            cy = y0 + float(np.sum(yy * weights) / total)
        response = float(np.max(patch))
        if response <= 0.0:
            continue
        support = float(np.count_nonzero(patch >= threshold * 0.65))
        candidate_radius = max(1.0, math.sqrt(max(1.0, support) / math.pi))
        confidence = response * min(1.0, support / 12.0)
        candidates.append(
            RegistrationBeadCandidate(
                x=float(cx),
                y=float(cy),
                radius=float(candidate_radius),
                response=response,
                confidence=float(confidence),
            )
        )
    candidates.sort(key=lambda item: (item.confidence, item.response), reverse=True)
    return candidates[:max_candidates]


def _generate_geometric_hypothesis_seeds(
    prepared: GeometricRegistrationPrepared,
    *,
    scale_values: Sequence[float],
    rotation_values: Sequence[float],
    prefilter_hypotheses: int,
) -> Tuple[List[RegistrationHypothesisSeed], int]:
    ref_beads = prepared.reference_beads
    mov_beads = prepared.moving_beads
    hypotheses: List[RegistrationHypothesisSeed] = []
    generated = 0
    if len(ref_beads) >= 2 and len(mov_beads) >= 2:
        pair_hypotheses, pair_generated = _generate_pair_hypothesis_seeds(
            prepared,
            scale_bounds=(min(scale_values), max(scale_values)),
            rotation_bounds=(min(rotation_values), max(rotation_values)),
            prefilter_hypotheses=prefilter_hypotheses,
        )
        hypotheses.extend(pair_hypotheses)
        generated += pair_generated
    if not hypotheses and len(ref_beads) >= 1 and len(mov_beads) >= 1:
        single_hypotheses, single_generated = _generate_single_bead_hypothesis_seeds(
            prepared,
            scale_values=scale_values,
            rotation_values=rotation_values,
            prefilter_hypotheses=prefilter_hypotheses,
        )
        hypotheses.extend(single_hypotheses)
        generated += single_generated
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return hypotheses[: max(1, prefilter_hypotheses)], generated


def _generate_pair_hypothesis_seeds(
    prepared: GeometricRegistrationPrepared,
    *,
    scale_bounds: Tuple[float, float],
    rotation_bounds: Tuple[float, float],
    prefilter_hypotheses: int,
) -> Tuple[List[RegistrationHypothesisSeed], int]:
    ref_beads = prepared.reference_beads[:28]
    mov_beads = prepared.moving_beads[:28]
    heap: List[Tuple[float, int, RegistrationHypothesisSeed]] = []
    generated = 0
    counter = 0
    min_scale, max_scale = scale_bounds
    for ref_i in range(len(ref_beads)):
        for ref_j in range(ref_i + 1, len(ref_beads)):
            ref_a = ref_beads[ref_i]
            ref_b = ref_beads[ref_j]
            ref_vec = np.array([ref_b.x - ref_a.x, ref_b.y - ref_a.y], dtype=np.float64)
            ref_len = float(np.linalg.norm(ref_vec))
            if ref_len < 8.0:
                continue
            ref_angle = math.atan2(ref_vec[1], ref_vec[0])
            for mov_i in range(len(mov_beads)):
                for mov_j in range(mov_i + 1, len(mov_beads)):
                    for swap in (False, True):
                        mov_a = mov_beads[mov_j] if swap else mov_beads[mov_i]
                        mov_b = mov_beads[mov_i] if swap else mov_beads[mov_j]
                        mov_vec = np.array([mov_b.x - mov_a.x, mov_b.y - mov_a.y], dtype=np.float64)
                        mov_len = float(np.linalg.norm(mov_vec))
                        if mov_len < 8.0:
                            continue
                        scale = ref_len / mov_len
                        if scale < min_scale or scale > max_scale:
                            continue
                        angle = _normalize_angle_deg(math.degrees(ref_angle - math.atan2(mov_vec[1], mov_vec[0])))
                        if angle < rotation_bounds[0] or angle > rotation_bounds[1]:
                            continue
                        matrix = _similarity_from_anchor_pair(mov_a, ref_a, scale, angle)
                        scored = _score_geometric_seed(
                            matrix,
                            prepared,
                            initializer="bead_pair",
                            constraint_level="full",
                        )
                        generated += 1
                        counter = _push_seed_heap(heap, scored, max(1, prefilter_hypotheses), counter)
    return _seed_heap_to_sorted_list(heap), generated


def _generate_single_bead_hypothesis_seeds(
    prepared: GeometricRegistrationPrepared,
    *,
    scale_values: Sequence[float],
    rotation_values: Sequence[float],
    prefilter_hypotheses: int,
) -> Tuple[List[RegistrationHypothesisSeed], int]:
    heap: List[Tuple[float, int, RegistrationHypothesisSeed]] = []
    generated = 0
    counter = 0
    ref_beads = prepared.reference_beads[: min(3, len(prepared.reference_beads))]
    mov_beads = prepared.moving_beads[: min(12, len(prepared.moving_beads))]
    for ref_bead in ref_beads:
        for mov_bead in mov_beads:
            for scale in scale_values:
                for rotation in rotation_values:
                    matrix = _similarity_from_anchor_pair(mov_bead, ref_bead, float(scale), float(rotation))
                    scored = _score_geometric_seed(
                        matrix,
                        prepared,
                        initializer="single_bead_grid",
                        constraint_level="single_bead",
                    )
                    generated += 1
                    counter = _push_seed_heap(heap, scored, max(1, prefilter_hypotheses), counter)
    return _seed_heap_to_sorted_list(heap), generated


def _push_seed_heap(
    heap: List[Tuple[float, int, RegistrationHypothesisSeed]],
    seed: RegistrationHypothesisSeed,
    limit: int,
    counter: int,
) -> int:
    entry = (float(seed.score), int(counter), seed)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def _seed_heap_to_sorted_list(heap: List[Tuple[float, int, RegistrationHypothesisSeed]]) -> List[RegistrationHypothesisSeed]:
    return [item[2] for item in sorted(heap, key=lambda entry: entry[0], reverse=True)]


def _similarity_from_anchor_pair(
    moving_bead: RegistrationBeadCandidate,
    reference_bead: RegistrationBeadCandidate,
    scale: float,
    rotation_deg: float,
) -> np.ndarray:
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta) * float(scale)
    sin_t = math.sin(theta) * float(scale)
    tx = reference_bead.x - (cos_t * moving_bead.x - sin_t * moving_bead.y)
    ty = reference_bead.y - (sin_t * moving_bead.x + cos_t * moving_bead.y)
    return np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=np.float64)


def _normalize_angle_deg(angle: float) -> float:
    value = float(angle)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _score_geometric_seed(
    matrix: np.ndarray,
    prepared: GeometricRegistrationPrepared,
    *,
    initializer: str,
    constraint_level: str,
) -> RegistrationHypothesisSeed:
    matched_count, median_error = _match_hypothesis_beads(matrix, prepared, radius=8.0)
    common_fov = _common_fov_fraction_analytic(
        prepared.moving_unit.shape,
        prepared.reference_unit.shape,
        matrix,
    )
    bead_count = max(1, len(prepared.reference_beads))
    inlier_score = matched_count / bead_count
    error_score = 0.0 if median_error is None else max(0.0, 1.0 - median_error / 8.0)
    if constraint_level == "single_bead":
        score = 0.55 * inlier_score + 0.20 * error_score + 0.25 * common_fov
    else:
        score = 0.65 * inlier_score + 0.25 * error_score + 0.10 * common_fov
    return RegistrationHypothesisSeed(
        matrix=np.asarray(matrix, dtype=np.float64),
        initializer=initializer,
        constraint_level=constraint_level,
        score=float(score),
        matched_bead_count=int(matched_count),
        median_bead_error_px=median_error,
        common_fov_fraction=float(common_fov),
    )


def _score_geometric_seed_dense(
    seed: RegistrationHypothesisSeed,
    prepared: GeometricRegistrationPrepared,
) -> RegistrationHypothesis:
    bead_count = max(1, len(prepared.reference_beads))
    inlier_score = seed.matched_bead_count / bead_count
    error_score = 0.0 if seed.median_bead_error_px is None else max(0.0, 1.0 - seed.median_bead_error_px / 8.0)
    dense_score = _dense_hypothesis_score(seed.matrix, prepared)
    if seed.constraint_level == "single_bead":
        score = 0.30 * inlier_score + 0.20 * error_score + 0.35 * dense_score + 0.15 * seed.common_fov_fraction
    else:
        score = 0.45 * inlier_score + 0.25 * error_score + 0.20 * dense_score + 0.10 * seed.common_fov_fraction
    return RegistrationHypothesis(
        matrix=np.asarray(seed.matrix, dtype=np.float64),
        initializer=seed.initializer,
        constraint_level=seed.constraint_level,
        score=float(score),
        matched_bead_count=int(seed.matched_bead_count),
        median_bead_error_px=seed.median_bead_error_px,
        common_fov_fraction=float(seed.common_fov_fraction),
    )


def _hypothesis_to_prepared_resolution(
    hypothesis: RegistrationHypothesis,
    source: GeometricRegistrationPrepared,
    target: GeometricRegistrationPrepared,
) -> RegistrationHypothesis:
    full_matrix = _scale_affine_between_resolutions(
        hypothesis.matrix,
        ref_scale=source.reference_scale,
        mov_scale=source.moving_scale,
    )
    target_matrix = _scale_affine_to_resolution(
        full_matrix,
        ref_scale=target.reference_scale,
        mov_scale=target.moving_scale,
    )
    return RegistrationHypothesis(
        matrix=target_matrix,
        initializer=hypothesis.initializer,
        constraint_level=hypothesis.constraint_level,
        score=hypothesis.score,
        matched_bead_count=hypothesis.matched_bead_count,
        median_bead_error_px=hypothesis.median_bead_error_px,
        common_fov_fraction=hypothesis.common_fov_fraction,
        refined_score=hypothesis.refined_score,
        optimizer_evaluations=hypothesis.optimizer_evaluations,
    )


def _score_geometric_hypothesis(
    matrix: np.ndarray,
    prepared: GeometricRegistrationPrepared,
    *,
    initializer: str,
    constraint_level: str,
) -> RegistrationHypothesis:
    matched_count, median_error = _match_hypothesis_beads(matrix, prepared, radius=8.0)
    common_fov = _common_fov_fraction(
        prepared.moving_unit.shape,
        prepared.reference_unit.shape,
        matrix,
    )
    bead_count = max(1, len(prepared.reference_beads))
    inlier_score = matched_count / bead_count
    error_score = 0.0 if median_error is None else max(0.0, 1.0 - median_error / 8.0)
    dense_score = _dense_hypothesis_score(matrix, prepared)
    if constraint_level == "single_bead":
        score = 0.30 * inlier_score + 0.20 * error_score + 0.35 * dense_score + 0.15 * common_fov
    else:
        score = 0.45 * inlier_score + 0.25 * error_score + 0.20 * dense_score + 0.10 * common_fov
    return RegistrationHypothesis(
        matrix=np.asarray(matrix, dtype=np.float64),
        initializer=initializer,
        constraint_level=constraint_level,
        score=float(score),
        matched_bead_count=int(matched_count),
        median_bead_error_px=median_error,
        common_fov_fraction=float(common_fov),
    )


def _match_hypothesis_beads(
    matrix: np.ndarray,
    prepared: GeometricRegistrationPrepared,
    *,
    radius: float,
) -> Tuple[int, Optional[float]]:
    if not prepared.reference_beads or not prepared.moving_beads:
        return 0, None
    moving_points = np.array([[bead.x, bead.y] for bead in prepared.moving_beads], dtype=np.float64)
    transformed = _transform_points(matrix, moving_points)
    reference_centers = [(bead.x, bead.y) for bead in prepared.reference_beads]
    moving_centers = [(float(x), float(y)) for x, y in transformed]
    return _match_bead_centers(reference_centers, moving_centers, radius=radius)


def _dense_hypothesis_score(matrix: np.ndarray, prepared: GeometricRegistrationPrepared) -> float:
    warped = cv.warpAffine(
        prepared.moving_structure,
        np.asarray(matrix, dtype=np.float32),
        (prepared.reference_structure.shape[1], prepared.reference_structure.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return max(0.0, _correlation(prepared.reference_structure, warped))


def _refine_geometric_hypothesis(
    hypothesis: RegistrationHypothesis,
    prepared: GeometricRegistrationPrepared,
    *,
    scale_bounds: Tuple[float, float],
    rotation_bounds: Tuple[float, float],
    max_evals: int,
) -> RegistrationHypothesis:
    initial = _matrix_to_similarity_params(hypothesis.matrix)
    evaluations = 0

    def objective(params: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return _geometric_objective(
            params,
            prepared,
            constraint_level=hypothesis.constraint_level,
            scale_bounds=scale_bounds,
            rotation_bounds=rotation_bounds,
        )

    try:
        refined = optimize.minimize(
            objective,
            initial,
            method="Powell",
            options={
                "maxiter": 45,
                "maxfev": max(1, int(max_evals)),
                "xtol": 1e-3,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        params = np.asarray(refined.x if refined.success else initial, dtype=np.float64)
    except Exception:
        params = initial
    matrix = _similarity_params_to_matrix(params)
    scored = _score_geometric_hypothesis(
        matrix,
        prepared,
        initializer=hypothesis.initializer,
        constraint_level=hypothesis.constraint_level,
    )
    refined_score = 1.0 - float(objective(_matrix_to_similarity_params(scored.matrix)))
    return RegistrationHypothesis(
        matrix=scored.matrix,
        initializer=scored.initializer,
        constraint_level=scored.constraint_level,
        score=scored.score,
        matched_bead_count=scored.matched_bead_count,
        median_bead_error_px=scored.median_bead_error_px,
        common_fov_fraction=scored.common_fov_fraction,
        refined_score=refined_score,
        optimizer_evaluations=int(evaluations),
    )


def _geometric_objective(
    params: np.ndarray,
    prepared: GeometricRegistrationPrepared,
    *,
    constraint_level: str,
    scale_bounds: Tuple[float, float],
    rotation_bounds: Tuple[float, float],
) -> float:
    matrix = _similarity_params_to_matrix(params)
    scale = float(math.exp(float(params[0])))
    rotation = float(params[1])
    penalty = 0.0
    if scale < scale_bounds[0] or scale > scale_bounds[1]:
        penalty += min(5.0, abs(scale - min(max(scale, scale_bounds[0]), scale_bounds[1])) * 5.0)
    if rotation < rotation_bounds[0] or rotation > rotation_bounds[1]:
        penalty += min(5.0, abs(rotation - min(max(rotation, rotation_bounds[0]), rotation_bounds[1])) / 20.0)
    matched_count, median_error = _match_hypothesis_beads(matrix, prepared, radius=10.0)
    bead_count = max(1, len(prepared.reference_beads))
    inlier_loss = 1.0 - min(1.0, matched_count / bead_count)
    error_loss = 1.0 if median_error is None else min(1.0, median_error / 10.0)
    dense_loss = 1.0 - _dense_hypothesis_score(matrix, prepared)
    fov_loss = 1.0 - _common_fov_fraction(prepared.moving_unit.shape, prepared.reference_unit.shape, matrix)
    if constraint_level == "single_bead":
        return 0.25 * inlier_loss + 0.15 * error_loss + 0.45 * dense_loss + 0.15 * fov_loss + penalty
    return 0.45 * inlier_loss + 0.25 * error_loss + 0.20 * dense_loss + 0.10 * fov_loss + penalty


def _matrix_to_similarity_params(matrix: np.ndarray) -> np.ndarray:
    scale, rotation, tx, ty = _decompose_similarity(matrix)
    return np.array([math.log(max(scale, 1e-6)), rotation, tx, ty], dtype=np.float64)


def _similarity_params_to_matrix(params: np.ndarray) -> np.ndarray:
    scale = math.exp(float(params[0]))
    theta = math.radians(float(params[1]))
    cos_t = math.cos(theta) * scale
    sin_t = math.sin(theta) * scale
    return np.array([[cos_t, -sin_t, float(params[2])], [sin_t, cos_t, float(params[3])]], dtype=np.float64)


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([pts, ones])
    return np.asarray(homogeneous @ np.asarray(matrix, dtype=np.float64).T, dtype=np.float64)


def _hypothesis_summary(
    hypothesis: RegistrationHypothesis,
    prepared: GeometricRegistrationPrepared,
    rank: int,
) -> Dict[str, Any]:
    scale, rotation, tx, ty = _decompose_similarity(hypothesis.matrix)
    return {
        "rank": rank,
        "initializer": hypothesis.initializer,
        "constraint_level": hypothesis.constraint_level,
        "score": hypothesis.score,
        "refined_score": hypothesis.refined_score,
        "matched_bead_count": hypothesis.matched_bead_count,
        "median_bead_error_px": hypothesis.median_bead_error_px,
        "common_fov_fraction": hypothesis.common_fov_fraction,
        "optimizer_evaluations": hypothesis.optimizer_evaluations,
        "scale": scale * prepared.moving_scale / prepared.reference_scale,
        "rotation_deg": rotation,
        "translation_x": tx / prepared.reference_scale,
        "translation_y": ty / prepared.reference_scale,
    }


def _estimate_bead_similarity_registration(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    scale_values: Sequence[float],
    registration_profile: str,
    registration_suspect_residual_px: float,
    registration_min_matched_beads: int,
    registration_min_structure_score: float,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    ref_channel_full = _registration_channel(ref)
    mov_channel_full = _registration_channel(mov)
    ref_channel, mov_channel, ref_scale, mov_scale = _downsample_registration_pair(
        ref_channel_full,
        mov_channel_full,
        max_dim=DEFAULT_REGISTRATION_MAX_DIM,
    )
    warnings: List[str] = []
    best: Optional[Tuple[float, float, np.ndarray, np.ndarray]] = None
    rotations = _coarse_registration_rotations()
    for scale in scale_values:
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            warnings.append(f"invalid scale skipped: {scale}")
            continue
        small_scale = float(scale) * ref_scale / mov_scale
        for rotation_deg in rotations:
            pre_small = _pre_similarity_matrix(mov_channel.shape, ref_channel.shape, small_scale, rotation_deg)
            canvas = cv.warpAffine(
                mov_channel,
                pre_small,
                (ref_channel.shape[1], ref_channel.shape[0]),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0.0,
            )
            dx, dy = _phase_translation_on_channels(canvas, ref_channel)
            for sx, sy in ((dx, dy), (-dx, -dy), (0.0, 0.0)):
                residual = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], dtype=np.float64)
                forward_small = _matrix_multiply_2x3(residual, pre_small)
                shifted = cv.warpAffine(
                    mov_channel,
                    forward_small.astype(np.float32),
                    (ref_channel.shape[1], ref_channel.shape[0]),
                    flags=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                score = _registration_channel_score(ref_channel, shifted)
                if best is None or score > best[0]:
                    best = (score, float(scale), forward_small, shifted)

    if best is None:
        forward = _pre_similarity_matrix(mov.shape, ref.shape, 1.0, 0.0).astype(np.float64)
        aligned = cv.warpAffine(
            mov,
            forward.astype(np.float32),
            (ref.shape[1], ref.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )
        scale, rotation_deg, tx, ty = _decompose_similarity(forward)
        result = SimilarityRegistrationResult(
            status="failed",
            aligned_image=aligned.astype(np.float32),
            forward_matrix=forward,
            scale=scale,
            rotation_deg=rotation_deg,
            translation_x=tx,
            translation_y=ty,
            ecc_score=None,
            ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
            registration_profile=registration_profile,
            warnings=warnings or ["bead similarity registration failed"],
        )
        return _with_registration_quality(
            result,
            ref,
            mov.shape,
            scale_values=scale_values,
            residual_suspect_px=registration_suspect_residual_px,
            min_matched_beads=registration_min_matched_beads,
            min_structure_score=registration_min_structure_score,
            reference_cache=quality_reference_cache,
        )

    score, _selected_scale, forward_small, _shifted = best
    forward = _scale_affine_between_resolutions(forward_small, ref_scale=ref_scale, mov_scale=mov_scale)
    aligned = cv.warpAffine(
        mov,
        forward.astype(np.float32),
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    scale, rotation_deg, tx, ty = _decompose_similarity(forward)
    result = SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=score,
        ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
        registration_profile=registration_profile,
        warnings=warnings,
    )
    return _with_registration_quality(
        result,
        ref,
        mov.shape,
        scale_values=scale_values,
        residual_suspect_px=registration_suspect_residual_px,
        min_matched_beads=registration_min_matched_beads,
        min_structure_score=registration_min_structure_score,
        reference_cache=quality_reference_cache,
    )


def apply_anchor_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    anchor: SimilarityRegistrationResult,
    *,
    residual_translation_max_px: float,
    scale_values: Optional[Sequence[float]] = None,
    registration_suspect_residual_px: float = DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX,
    registration_min_matched_beads: int = DEFAULT_REGISTRATION_MIN_MATCHED_BEADS,
    registration_min_structure_score: float = DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE,
    reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    """Apply an anchor similarity transform with capped residual translation."""

    ref = _as_float_2d(reference)
    mov = _as_float_2d(moving)
    forward = np.asarray(anchor.forward_matrix, dtype=np.float64).copy()
    aligned = cv.warpAffine(
        mov,
        forward.astype(np.float32),
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warnings = list(anchor.warnings)

    bead_residual = _bead_residual_translation(aligned, ref)
    dx, dy = bead_residual if bead_residual is not None else _phase_translation(aligned, ref)
    norm = float(math.hypot(dx, dy))
    residual_applied = False
    if norm <= max(0.0, float(residual_translation_max_px)) and norm > 1e-6:
        aligned, dx, dy = _apply_best_translation(aligned, ref, dx, dy)
        residual_applied = True
        residual_forward = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        forward = _matrix_multiply_2x3(residual_forward, forward)
    elif norm > max(0.0, float(residual_translation_max_px)):
        warnings.append(
            f"Residual translation {norm:.3f}px exceeds cap "
            f"{float(residual_translation_max_px):.3f}px; not applied"
        )
        dx = 0.0
        dy = 0.0

    scale, rotation_deg, tx, ty = _decompose_similarity(forward)
    result = SimilarityRegistrationResult(
        status=anchor.status,
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=anchor.ecc_score,
        ecc_warp_matrix=anchor.ecc_warp_matrix.copy(),
        residual_translation_x=float(dx),
        residual_translation_y=float(dy),
        residual_translation_norm=norm,
        residual_applied=residual_applied,
        registration_profile=anchor.registration_profile,
        anchor_selection_reason=anchor.anchor_selection_reason,
        registration_constraint_level=anchor.registration_constraint_level,
        registration_underconstrained=anchor.registration_underconstrained,
        registration_initializer=anchor.registration_initializer,
        registration_calibration_group=anchor.registration_calibration_group,
        registration_transform_source=anchor.registration_transform_source,
        registration_point_count=anchor.registration_point_count,
        registration_fit_rms_px=anchor.registration_fit_rms_px,
        validation_rankable=anchor.validation_rankable,
        ranking_exclusion_reason=anchor.ranking_exclusion_reason,
        registration_point_pairs=list(anchor.registration_point_pairs),
        registration_refined_score=anchor.registration_refined_score,
        registration_hypotheses=anchor.registration_hypotheses,
        registration_runtime_seconds=anchor.registration_runtime_seconds,
        hypotheses_generated=anchor.hypotheses_generated,
        hypotheses_prefiltered=anchor.hypotheses_prefiltered,
        hypotheses_dense_scored=anchor.hypotheses_dense_scored,
        hypotheses_refined=anchor.hypotheses_refined,
        optimizer_evaluations=anchor.optimizer_evaluations,
        registration_timing_seconds=dict(anchor.registration_timing_seconds),
        warnings=_unique_strings(warnings),
    )
    result = _with_registration_quality(
        result,
        ref,
        mov.shape,
        scale_values=[anchor.scale] if scale_values is None else scale_values,
        residual_suspect_px=min(float(registration_suspect_residual_px), max(0.0, float(residual_translation_max_px))),
        min_matched_beads=registration_min_matched_beads,
        min_structure_score=registration_min_structure_score,
        reference_cache=reference_cache,
    )
    result.validation_rankable, result.ranking_exclusion_reason = _registration_rankability(result)
    return result


def compute_validation_metrics(
    validation_image: np.ndarray,
    candidate_image: np.ndarray,
    *,
    reference_cache: Optional[ValidationMetricReference] = None,
) -> ValidationMetricResult:
    """Compute validation agreement metrics for one registered candidate."""

    validation = _robust_unit(_as_float_2d(validation_image))
    candidate = _robust_unit(_as_float_2d(candidate_image))
    if validation.shape != candidate.shape:
        min_h = min(validation.shape[0], candidate.shape[0])
        min_w = min(validation.shape[1], candidate.shape[1])
        validation = validation[:min_h, :min_w]
        candidate = candidate[:min_h, :min_w]
    prepared = (
        reference_cache
        if reference_cache is not None and reference_cache.validation_unit.shape == validation.shape
        else _prepare_validation_metric_reference_from_unit(validation)
    )

    adjusted = _fit_gain_offset(candidate, prepared.validation_unit)
    signal = _signal_score(prepared.validation_unit, adjusted)
    structure = max(0.0, _correlation(prepared.structure_channel, _structure_channel(adjusted)))
    signal_structure = _weighted_score(
        {"signal": signal, "structure": structure},
        {"signal": 0.30, "structure": 0.20},
    )

    centers = prepared.bead_centers
    warnings: List[str] = []
    bead: Optional[float]
    sharpness: Optional[float]
    if len(centers) < 3:
        bead = None
        sharpness = None
        warnings.append(BEAD_LOW_CONFIDENCE_WARNING)
        weights_used = {"validation_signal": 0.60, "validation_structure": 0.40}
        hybrid = signal_structure
    else:
        candidate_centers = _detect_bead_centers(adjusted)
        bead = _bead_score(centers, candidate_centers)
        sharpness = _sharpness_score(prepared.validation_unit, adjusted, centers)
        weights_used = {
            "validation_signal": 0.30,
            "validation_structure": 0.20,
            "validation_bead": 0.25,
            "validation_sharpness": 0.25,
        }
        hybrid = _weighted_score(
            {
                "signal": signal,
                "structure": structure,
                "bead": bead,
                "sharpness": sharpness,
            },
            {"signal": 0.30, "structure": 0.20, "bead": 0.25, "sharpness": 0.25},
        )

    return ValidationMetricResult(
        metrics={
            PROFILE_NAME: _clip01(hybrid),
            "validation_signal": _clip01(signal),
            "validation_structure": _clip01(structure),
            "validation_bead": None if bead is None else _clip01(bead),
            "validation_sharpness": None if sharpness is None else _clip01(sharpness),
            "validation_signal_structure_score": _clip01(signal_structure),
        },
        weights_used=weights_used,
        bead_count=len(centers),
        warnings=warnings,
    )


def _prepare_validation_metric_reference(validation_image: np.ndarray) -> ValidationMetricReference:
    return _prepare_validation_metric_reference_from_unit(_robust_unit(_as_float_2d(validation_image)))


def _prepare_validation_metric_reference_from_unit(validation_unit: np.ndarray) -> ValidationMetricReference:
    unit = np.asarray(validation_unit, dtype=np.float32)
    return ValidationMetricReference(
        validation_unit=unit,
        structure_channel=_structure_channel(unit),
        bead_centers=_detect_bead_centers(unit),
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the trio benchmark and write CSV/JSON/HTML artifacts."""

    cfg = _normalize_config(config)
    references = _load_reference_images(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation = references.validation
    thorcam = references.thorcam
    metric_reference_cache = _prepare_validation_metric_reference(validation)
    geometric_reference_cache = (
        _prepare_geometric_reference_cache(validation, cfg)
        if cfg.registration_profile == REGISTRATION_PROFILE_BEAD_GEOMETRIC
        else None
    )
    manual_annotations: List[ManualAnnotationRecord] = []
    if cfg.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        if cfg.registration_annotations is None:
            raise ValueError("manual_similarity_v1 requires registration_annotations")
        manual_annotations = _load_manual_annotation_records(cfg.registration_annotations)
    model = load_sweep_visualization_model(cfg.sweep_dir)
    stage = cfg.stage or str(model.primary_stage_name)
    candidate_images, skipped_candidate_warnings = _collect_candidate_images(model, cfg, stage)
    if not candidate_images:
        raise ValueError(
            f"No LF candidates with loadable images found for stage={stage!r}, "
            f"dataset_id={cfg.dataset_id!r}."
        )

    if cfg.use_validation_workspace_transforms:
        anchor_record = _select_anchor(candidate_images, cfg.anchor_row_id)
        anchor_registration = _workspace_grid_identity_registration(
            validation,
            anchor_record.image,
            config=cfg,
            initializer="validation_workspace_identity_lf",
            transform_source="identity_lf_grid",
            quality_reference_cache=metric_reference_cache,
        )
        anchor_registration.anchor_selection_reason = (
            "explicit_anchor_row_id" if cfg.anchor_row_id != "auto" else "saved_rank_auto_workspace_transform"
        )
    elif cfg.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        anchor_record = _select_anchor(candidate_images, cfg.anchor_row_id)
        anchor_registration = _estimate_manual_similarity_registration(
            validation,
            anchor_record.image,
            config=cfg,
            source_kind="lf",
            annotations=manual_annotations,
            anchor_record=anchor_record,
            quality_reference_cache=metric_reference_cache,
        )
        anchor_registration.anchor_selection_reason = (
            "explicit_anchor_row_id" if cfg.anchor_row_id != "auto" else "saved_rank_auto_manual_profile"
        )
    else:
        anchor_record, anchor_registration = _select_registration_anchor(
            validation,
            candidate_images,
            cfg,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=metric_reference_cache,
        )

    lf_rows: List[BenchmarkResultRow] = []
    for candidate in candidate_images:
        registration = apply_anchor_registration(
            validation,
            candidate.image,
            anchor_registration,
            residual_translation_max_px=cfg.residual_translation_max_px,
            scale_values=cast(Sequence[float], cfg.scale_values),
            registration_suspect_residual_px=cfg.registration_suspect_residual_px,
            registration_min_matched_beads=cfg.registration_min_matched_beads,
            registration_min_structure_score=cfg.registration_min_structure_score,
            reference_cache=metric_reference_cache,
        )
        metrics = compute_validation_metrics(validation, registration.aligned_image, reference_cache=metric_reference_cache)
        row = candidate.row
        record = candidate.record
        lf_rows.append(
            BenchmarkResultRow(
                source_kind="lf_candidate",
                row_id=str(getattr(row, "row_id", getattr(record, "row_id", ""))),
                candidate_id=str(getattr(row, "candidate_id", getattr(record, "candidate_id", ""))),
                stage=str(getattr(row, "stage", getattr(record, "stage", stage))),
                dataset_id=cfg.dataset_id,
                generation_id=_optional_str(getattr(row, "generation_id", getattr(record, "generation_id", None))),
                strategy=str(getattr(row, "strategy", "")),
                saved_rank=_optional_int(getattr(row, "rank", None)),
                validation_rank=None,
                saved_score=_optional_float(getattr(row, "score", None)),
                metrics=metrics.metrics,
                registration=registration,
                workspace_path=candidate.workspace_path,
                preview_path=candidate.preview_path,
                params=_jsonable_mapping(getattr(row, "params", {})),
                config=_jsonable_mapping(getattr(row, "config", {})),
                warnings=_unique_strings(candidate.warnings + registration.warnings + metrics.warnings),
            )
        )

    lf_rows.sort(
        key=lambda item: (
            bool(item.registration.validation_rankable),
            _score_sort_value(item.metrics.get(PROFILE_NAME)),
        ),
        reverse=True,
    )
    rank = 1
    for row in lf_rows:
        if row.registration.validation_rankable:
            row.validation_rank = rank
            rank += 1

    if cfg.use_validation_workspace_transforms:
        thor_source = _workspace_transform_source_for(references, "thorcam")
        thor_registration = _workspace_grid_identity_registration(
            validation,
            thorcam,
            config=cfg,
            initializer="validation_workspace_transform",
            transform_source=thor_source,
            quality_reference_cache=metric_reference_cache,
        )
    elif cfg.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        thor_registration = _estimate_manual_similarity_registration(
            validation,
            thorcam,
            config=cfg,
            source_kind="thorcam",
            annotations=manual_annotations,
            quality_reference_cache=metric_reference_cache,
        )
    else:
        thor_registration = estimate_similarity_registration(
            validation,
            thorcam,
            scale_values=cast(Sequence[float], cfg.scale_values),
            rotation_values=cast(Sequence[float], cfg.rotation_values),
            registration_top_hypotheses=cfg.registration_top_hypotheses,
            registration_diagnostics_top_n=cfg.registration_diagnostics_top_n,
            registration_coarse_max_dim=cfg.registration_coarse_max_dim,
            registration_refine_max_dim=cfg.registration_refine_max_dim,
            registration_prefilter_hypotheses=cfg.registration_prefilter_hypotheses,
            registration_dense_hypotheses=cfg.registration_dense_hypotheses,
            registration_refine_max_evals=cfg.registration_refine_max_evals,
            registration_profile=cfg.registration_profile,
            registration_suspect_residual_px=cfg.registration_suspect_residual_px,
            registration_min_matched_beads=cfg.registration_min_matched_beads,
            registration_min_structure_score=cfg.registration_min_structure_score,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=metric_reference_cache,
        )
    thor_metrics = compute_validation_metrics(validation, thor_registration.aligned_image, reference_cache=metric_reference_cache)
    thor_row = BenchmarkResultRow(
        source_kind="thorcam_baseline",
        row_id="thorcam_baseline",
        candidate_id="thorcam",
        stage="baseline",
        dataset_id=cfg.dataset_id,
        generation_id=None,
        strategy="thorcam",
        saved_rank=None,
        validation_rank=None,
        saved_score=None,
        metrics=thor_metrics.metrics,
        registration=thor_registration,
        workspace_path=None,
        preview_path=references.thorcam_source,
        warnings=_unique_strings(thor_registration.warnings + thor_metrics.warnings),
    )

    rows = lf_rows + [thor_row]
    _write_bead_detection_figure(figures_dir / "validation_beads.png", validation)
    if cfg.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        _write_manual_points_figure(figures_dir / "manual_points_lf.png", validation, anchor_record.image, anchor_registration)
        _write_manual_points_figure(figures_dir / "manual_points_thorcam.png", validation, thorcam, thor_registration)
    if cfg.use_validation_workspace_transforms:
        _write_workspace_transform_diagnostic_figure(
            figures_dir / "validation_workspace_validation_overlay.png",
            validation,
            validation,
        )
        _write_workspace_transform_diagnostic_figure(
            figures_dir / "validation_workspace_thorcam_overlay.png",
            validation,
            thorcam,
        )
    top_rows = lf_rows[: max(0, cfg.top_k)]
    for row in top_rows:
        _write_comparison_figures(figures_dir, validation, row)
        _write_bead_detection_figure(figures_dir / f"{_figure_prefix(row)}_beads.png", row.registration.aligned_image)
        if cfg.enable_nonrigid_diagnostic:
            _write_affine_diagnostic_figure(figures_dir, validation, row)
    _write_comparison_figures(figures_dir, validation, thor_row)
    _write_bead_detection_figure(figures_dir / "baseline_thorcam_beads.png", thor_row.registration.aligned_image)
    if cfg.enable_nonrigid_diagnostic:
        _write_affine_diagnostic_figure(figures_dir, validation, thor_row)

    best = next((row for row in lf_rows if row.registration.validation_rankable), None)
    summary = _build_summary(
        cfg,
        stage,
        anchor_record,
        anchor_registration,
        best,
        thor_row,
        rows,
        model_warnings=_unique_strings(getattr(model, "warnings", [])),
        skipped_candidate_warnings=skipped_candidate_warnings,
        reference_sources=references,
    )
    _write_results_csv(cfg.output_dir / "trio_benchmark_results.csv", rows)
    _write_registration_hypotheses_csv(cfg.output_dir / "registration_hypotheses.csv", rows)
    if cfg.registration_profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        _write_manual_calibration_artifact(
            cfg.output_dir / "trio_registration_calibration.json",
            cfg,
            anchor_registration,
            thor_registration,
        )
    if cfg.use_validation_workspace_transforms:
        _write_validation_workspace_transform_artifact(
            cfg.output_dir / "trio_validation_workspace_transforms.json",
            cfg,
            references,
            anchor_registration,
            thor_registration,
        )
    write_json_atomic(
        cfg.output_dir / "trio_benchmark_summary.json",
        summary,
        sort_keys=True,
        create_backup=False,
    )
    write_text_atomic(
        cfg.output_dir / "trio_benchmark_report.html",
        _render_report_html(summary, rows, figures_dir),
        create_backup=False,
    )
    return BenchmarkResult(best_lf_row_id=None if best is None else best.row_id, rows=rows, summary=summary)


def _normalize_config(config: BenchmarkConfig) -> BenchmarkConfig:
    generation_scope = _normalize_generation_scope_arg(config.generation_scope)
    use_validation_workspace_transforms = bool(
        config.use_validation_workspace_transforms
        or str(config.registration_profile).strip() == REGISTRATION_PROFILE_VALIDATE_WORKSPACE
    )
    registration_profile = (
        REGISTRATION_PROFILE_VALIDATE_WORKSPACE
        if use_validation_workspace_transforms
        else _normalize_registration_profile(config.registration_profile)
    )
    scale_values = (
        list(config.scale_values)
        if config.scale_values is not None
        else _default_scale_values_for_registration_profile(registration_profile)
    )
    rotation_values = (
        list(config.rotation_values)
        if config.rotation_values is not None
        else parse_rotation_search(DEFAULT_REGISTRATION_ROTATION_SEARCH)
    )
    if config.output_dir is None:
        raise ValueError("output_dir is required")
    return BenchmarkConfig(
        sweep_dir=Path(config.sweep_dir),
        dataset_id=str(config.dataset_id),
        scene_id=None if config.scene_id is None else str(config.scene_id),
        validation_image=None if config.validation_image is None else Path(config.validation_image),
        thorcam_image=None if config.thorcam_image is None else Path(config.thorcam_image),
        output_dir=Path(config.output_dir),
        validation_workspace=None if config.validation_workspace is None else Path(config.validation_workspace),
        stage=config.stage,
        generation_scope=generation_scope,
        anchor_row_id=config.anchor_row_id,
        scale_values=scale_values,
        rotation_values=rotation_values,
        top_k=int(config.top_k),
        residual_translation_max_px=float(config.residual_translation_max_px),
        registration_profile=registration_profile,
        use_validation_workspace_transforms=use_validation_workspace_transforms,
        registration_annotations=None if config.registration_annotations is None else Path(config.registration_annotations),
        registration_calibration_group=str(
            config.registration_calibration_group or DEFAULT_REGISTRATION_CALIBRATION_GROUP
        ),
        registration_calibration_mode=_normalize_registration_calibration_mode(config.registration_calibration_mode),
        registration_suspect_residual_px=float(config.registration_suspect_residual_px),
        registration_min_matched_beads=int(config.registration_min_matched_beads),
        registration_min_structure_score=float(config.registration_min_structure_score),
        registration_top_hypotheses=max(1, int(config.registration_top_hypotheses)),
        registration_diagnostics_top_n=max(0, int(config.registration_diagnostics_top_n)),
        registration_coarse_max_dim=max(1, int(config.registration_coarse_max_dim)),
        registration_refine_max_dim=max(1, int(config.registration_refine_max_dim)),
        registration_prefilter_hypotheses=max(1, int(config.registration_prefilter_hypotheses)),
        registration_dense_hypotheses=max(1, int(config.registration_dense_hypotheses)),
        registration_refine_max_evals=max(1, int(config.registration_refine_max_evals)),
        registration_progress=bool(config.registration_progress),
        enable_nonrigid_diagnostic=bool(config.enable_nonrigid_diagnostic),
    )


def _load_reference_images(config: BenchmarkConfig) -> ReferenceImages:
    workspace: Optional[Mapping[str, np.ndarray]] = None
    workspace_path = config.validation_workspace
    if workspace_path is not None:
        workspace = load_workspace_variables(Path(workspace_path))
    if config.use_validation_workspace_transforms:
        if workspace is None or workspace_path is None:
            raise ValueError("use_validation_workspace_transforms requires --validation-workspace")
        return _load_reference_images_from_workspace_transforms(workspace, workspace_path, config)

    validation, validation_source = _load_reference_image(
        image_path=config.validation_image,
        workspace=workspace,
        workspace_path=workspace_path,
        workspace_key="validation_processed",
        label="validation",
    )
    thorcam, thorcam_source = _load_reference_image(
        image_path=config.thorcam_image,
        workspace=workspace,
        workspace_path=workspace_path,
        workspace_key="thorcam_processed",
        label="thorcam",
    )
    return ReferenceImages(
        validation=validation,
        thorcam=thorcam,
        validation_source=validation_source,
        thorcam_source=thorcam_source,
        warnings=_validation_workspace_alignment_warnings(workspace, workspace_path),
    )


def _load_reference_images_from_workspace_transforms(
    workspace: Mapping[str, np.ndarray],
    workspace_path: Path,
    config: BenchmarkConfig,
) -> ReferenceImages:
    if "rectified_image" not in workspace:
        raise ValueError(f"{workspace_path} does not contain 'rectified_image'")
    rectified = _as_float_2d(np.asarray(workspace["rectified_image"]))
    warnings = _validation_workspace_alignment_warnings(workspace, workspace_path)
    if config.registration_annotations is not None:
        warnings.append(
            "use_validation_workspace_transforms enabled; registration_annotations ignored"
        )
    validation, validation_info = _warp_workspace_processed_to_rectified_grid(
        workspace,
        workspace_path,
        prefix="validation",
        rectified_shape=rectified.shape,
    )
    thorcam, thorcam_info = _warp_workspace_processed_to_rectified_grid(
        workspace,
        workspace_path,
        prefix="thorcam",
        rectified_shape=rectified.shape,
    )
    warnings.extend(validation_info["warnings"])
    warnings.extend(thorcam_info["warnings"])
    transform_info = {
        "workspace_path": str(workspace_path),
        "rectified_grid_shape": [int(rectified.shape[0]), int(rectified.shape[1])],
        "sources": {
            "validation": validation_info,
            "thorcam": thorcam_info,
        },
    }
    return ReferenceImages(
        validation=validation,
        thorcam=thorcam,
        validation_source=f"{workspace_path}#validation_processed->{validation_info['transform_source']}",
        thorcam_source=f"{workspace_path}#thorcam_processed->{thorcam_info['transform_source']}",
        warnings=_unique_strings(warnings),
        workspace_transform_info=transform_info,
    )


def _warp_workspace_processed_to_rectified_grid(
    workspace: Mapping[str, np.ndarray],
    workspace_path: Path,
    *,
    prefix: str,
    rectified_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_key = f"{prefix}_processed"
    matrix_key = f"{prefix}_alignment_final_warp_matrix"
    if image_key not in workspace:
        raise ValueError(f"{workspace_path} does not contain {image_key!r}")
    source = _as_float_2d(np.asarray(workspace[image_key]))
    matrix = _workspace_affine_matrix(workspace.get(matrix_key))
    warnings: List[str] = []
    transform_source = matrix_key
    if matrix is None:
        if tuple(source.shape[:2]) != tuple(rectified_shape):
            raise ValueError(
                f"{workspace_path} has no usable {matrix_key!r} and {image_key!r} shape "
                f"{tuple(source.shape[:2])} differs from rectified grid {tuple(rectified_shape)}"
            )
        matrix = np.eye(2, 3, dtype=np.float32)
        transform_source = "identity_workspace_same_shape"
        warnings.append(f"{prefix}:{transform_source}:{matrix_key}")
    warped = cv.warpAffine(
        source,
        np.asarray(matrix, dtype=np.float32),
        (int(rectified_shape[1]), int(rectified_shape[0])),
        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    info: Dict[str, Any] = {
        "image_key": image_key,
        "matrix_key": matrix_key,
        "transform_source": transform_source,
        "source_shape": [int(source.shape[0]), int(source.shape[1])],
        "output_shape": [int(rectified_shape[0]), int(rectified_shape[1])],
        "matrix": np.asarray(matrix, dtype=float).tolist(),
        "warnings": warnings,
    }
    return warped.astype(np.float32), info


def _workspace_affine_matrix(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _validation_workspace_alignment_warnings(
    workspace: Optional[Mapping[str, np.ndarray]],
    workspace_path: Optional[Path],
) -> List[str]:
    if workspace is None or workspace_path is None:
        return []
    warnings: List[str] = []
    for prefix in ("validation", "thorcam"):
        auto_align = _workspace_bool(workspace.get(f"{prefix}_auto_align"))
        status = _workspace_text(workspace.get(f"{prefix}_alignment_status"))
        if auto_align is True and _is_suspect_validation_alignment_status(status):
            warnings.append(
                "validation workspace alignment suspect: "
                f"{prefix}_alignment_status={status}; source={workspace_path}"
            )
    return warnings


def _is_suspect_validation_alignment_status(status: Optional[str]) -> bool:
    if not status:
        return False
    return (
        status == "low_confidence_match"
        or status == "no_containment_match"
        or status.startswith("ecc_failed:")
    )


def _workspace_text(value: Optional[np.ndarray]) -> Optional[str]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _workspace_bool(value: Optional[np.ndarray]) -> Optional[bool]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    item = arr.reshape(-1)[0]
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, (np.integer, int)):
        return bool(item)
    if isinstance(item, bytes):
        text = item.decode("utf-8", errors="replace").strip().lower()
    else:
        text = str(item).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _load_reference_image(
    *,
    image_path: Optional[Path],
    workspace: Optional[Mapping[str, np.ndarray]],
    workspace_path: Optional[Path],
    workspace_key: str,
    label: str,
) -> Tuple[np.ndarray, str]:
    if image_path is not None:
        path = Path(image_path)
        return _load_grayscale_image(path), str(path)
    if workspace is None or workspace_path is None:
        raise ValueError(
            f"{label} reference is required: provide --{label}-image "
            "or --validation-workspace"
        )
    if workspace_key not in workspace:
        raise ValueError(f"{workspace_path} does not contain {workspace_key!r}")
    image = _as_float_2d(np.asarray(workspace[workspace_key]))
    return image, f"{workspace_path}#{workspace_key}"


def _normalize_generation_scope_arg(scope: str) -> str:
    text = str(scope).strip()
    if text.lower() == GENERATION_SCOPE_ALL.lower():
        return GENERATION_SCOPE_ALL
    if text.lower() == GENERATION_SCOPE_ACTIVE.lower():
        return GENERATION_SCOPE_ACTIVE
    return text


def _collect_candidate_images(
    model: Any,
    config: BenchmarkConfig,
    stage: str,
) -> Tuple[List[CandidateImageRecord], List[str]]:
    rows = [
        row
        for row in candidate_rows_for_generation_scope(model, config.generation_scope)
        if str(getattr(row, "stage", "")) == stage
    ]
    records = [
        record
        for record in run_records_for_generation_scope(model, config.generation_scope)
        if str(getattr(record, "stage", "")) == stage
        and str(getattr(record, "dataset_id", "")) == config.dataset_id
    ]
    record_by_key = {
        (
            str(getattr(record, "stage", "")),
            str(getattr(record, "candidate_id", "")),
            _optional_str(getattr(record, "generation_id", None)),
        ): record
        for record in records
    }
    result: List[CandidateImageRecord] = []
    skipped_warnings: List[str] = []
    for row in rows:
        key = (
            str(getattr(row, "stage", "")),
            str(getattr(row, "candidate_id", "")),
            _optional_str(getattr(row, "generation_id", None)),
        )
        record = record_by_key.get(key)
        try:
            image, workspace_path, preview_path, warnings = _load_candidate_image(
                row,
                record,
                dataset_id=config.dataset_id,
            )
        except Exception as exc:
            skipped_warnings.append(
                "skipped candidate "
                f"{str(getattr(row, 'stage', stage))}:{str(getattr(row, 'candidate_id', 'unknown'))} "
                f"dataset={config.dataset_id}: {type(exc).__name__}: {exc}"
            )
            continue
        result.append(
            CandidateImageRecord(
                row=row,
                record=record,
                image=image,
                workspace_path=workspace_path,
                preview_path=preview_path,
                warnings=warnings,
            )
        )
    return result, skipped_warnings


def _load_candidate_image(
    row: Any,
    record: Optional[Any],
    *,
    dataset_id: str,
) -> Tuple[np.ndarray, Optional[str], Optional[str], List[str]]:
    workspace_path = _optional_str(getattr(record, "workspace_path", None)) if record is not None else None
    preview_path = _optional_str(getattr(record, "preview_path", None)) if record is not None else None
    if not preview_path:
        preview_paths = getattr(row, "preview_paths", {})
        if isinstance(preview_paths, Mapping):
            preview_path = _optional_str(preview_paths.get(dataset_id) or preview_paths.get("aggregate"))
    if not workspace_path and preview_path:
        adjacent_workspace = Path(preview_path).with_name("workspace.npz")
        if adjacent_workspace.exists():
            workspace_path = str(adjacent_workspace)
    warnings: List[str] = []
    if workspace_path:
        try:
            return (
                _as_float_2d(load_rectified_reference(workspace_path)),
                workspace_path,
                preview_path,
                warnings,
            )
        except Exception as exc:
            warnings.append(f"workspace_load_failed:{type(exc).__name__}")
    if preview_path:
        warnings.append("preview_fallback")
        return _load_grayscale_image(Path(preview_path)), workspace_path, preview_path, warnings
    raise FileNotFoundError("candidate has neither readable workspace_path nor preview_path")


def _select_anchor(records: Sequence[CandidateImageRecord], anchor_row_id: str) -> CandidateImageRecord:
    if anchor_row_id != "auto":
        for record in records:
            row_id = str(getattr(record.row, "row_id", ""))
            candidate_id = str(getattr(record.row, "candidate_id", ""))
            if anchor_row_id in {row_id, candidate_id}:
                return record
        raise ValueError(f"anchor row id not found among loadable candidates: {anchor_row_id}")
    return min(records, key=lambda item: (_optional_int(getattr(item.row, "rank", None)) or 10**9, str(getattr(item.row, "row_id", ""))))


def _emit_registration_progress(config: BenchmarkConfig, message: str) -> None:
    if config.registration_progress:
        print(message, file=sys.stderr, flush=True)


def _select_registration_anchor(
    validation: np.ndarray,
    records: Sequence[CandidateImageRecord],
    config: BenchmarkConfig,
    *,
    geometric_reference_cache: Optional[GeometricReferenceCache] = None,
    quality_reference_cache: Optional[ValidationMetricReference] = None,
) -> Tuple[CandidateImageRecord, SimilarityRegistrationResult]:
    if config.anchor_row_id != "auto":
        anchor_record = _select_anchor(records, config.anchor_row_id)
        registration = estimate_similarity_registration(
            validation,
            anchor_record.image,
            scale_values=cast(Sequence[float], config.scale_values),
            rotation_values=cast(Sequence[float], config.rotation_values),
            registration_top_hypotheses=config.registration_top_hypotheses,
            registration_diagnostics_top_n=config.registration_diagnostics_top_n,
            registration_coarse_max_dim=config.registration_coarse_max_dim,
            registration_refine_max_dim=config.registration_refine_max_dim,
            registration_prefilter_hypotheses=config.registration_prefilter_hypotheses,
            registration_dense_hypotheses=config.registration_dense_hypotheses,
            registration_refine_max_evals=config.registration_refine_max_evals,
            registration_profile=config.registration_profile,
            registration_suspect_residual_px=config.registration_suspect_residual_px,
            registration_min_matched_beads=config.registration_min_matched_beads,
            registration_min_structure_score=config.registration_min_structure_score,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=quality_reference_cache,
        )
        registration.anchor_selection_reason = "explicit_anchor_row_id"
        return anchor_record, registration

    if config.registration_profile == REGISTRATION_PROFILE_CENTERED_ECC:
        anchor_record = _select_anchor(records, config.anchor_row_id)
        registration = estimate_similarity_registration(
            validation,
            anchor_record.image,
            scale_values=cast(Sequence[float], config.scale_values),
            rotation_values=cast(Sequence[float], config.rotation_values),
            registration_top_hypotheses=config.registration_top_hypotheses,
            registration_diagnostics_top_n=config.registration_diagnostics_top_n,
            registration_coarse_max_dim=config.registration_coarse_max_dim,
            registration_refine_max_dim=config.registration_refine_max_dim,
            registration_prefilter_hypotheses=config.registration_prefilter_hypotheses,
            registration_dense_hypotheses=config.registration_dense_hypotheses,
            registration_refine_max_evals=config.registration_refine_max_evals,
            registration_profile=config.registration_profile,
            registration_suspect_residual_px=config.registration_suspect_residual_px,
            registration_min_matched_beads=config.registration_min_matched_beads,
            registration_min_structure_score=config.registration_min_structure_score,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=quality_reference_cache,
        )
        registration.anchor_selection_reason = "saved_rank_auto_legacy_profile"
        return anchor_record, registration

    evaluated: List[Tuple[float, CandidateImageRecord, SimilarityRegistrationResult]] = []
    for index, record in enumerate(records, start=1):
        _emit_registration_progress(
            config,
            f"Registering anchor candidate {index}/{len(records)}: {getattr(record.row, 'row_id', '')}",
        )
        registration = estimate_similarity_registration(
            validation,
            record.image,
            scale_values=cast(Sequence[float], config.scale_values),
            rotation_values=cast(Sequence[float], config.rotation_values),
            registration_top_hypotheses=config.registration_top_hypotheses,
            registration_diagnostics_top_n=config.registration_diagnostics_top_n,
            registration_coarse_max_dim=config.registration_coarse_max_dim,
            registration_refine_max_dim=config.registration_refine_max_dim,
            registration_prefilter_hypotheses=config.registration_prefilter_hypotheses,
            registration_dense_hypotheses=config.registration_dense_hypotheses,
            registration_refine_max_evals=config.registration_refine_max_evals,
            registration_profile=config.registration_profile,
            registration_suspect_residual_px=config.registration_suspect_residual_px,
            registration_min_matched_beads=config.registration_min_matched_beads,
            registration_min_structure_score=config.registration_min_structure_score,
            geometric_reference_cache=geometric_reference_cache,
            quality_reference_cache=quality_reference_cache,
        )
        confidence = registration.registration_confidence
        score = -float("inf") if confidence is None else float(confidence)
        evaluated.append((score, record, registration))
    if not evaluated:
        raise ValueError("anchor selection received no loadable candidates")
    evaluated.sort(
        key=lambda item: (
            item[0],
            _score_sort_value(item[2].ecc_score),
            -float(_optional_int(getattr(item[1].row, "rank", None)) or 10**9),
            str(getattr(item[1].row, "row_id", "")),
        ),
        reverse=True,
    )
    _score, anchor_record, registration = evaluated[0]
    registration.anchor_selection_reason = "best_registration_confidence"
    return anchor_record, registration


def _write_results_csv(path: Path, rows: Sequence[BenchmarkResultRow]) -> None:
    fieldnames = [
        "source_kind",
        "validation_rank",
        "row_id",
        "candidate_id",
        "stage",
        "dataset_id",
        "generation_id",
        "strategy",
        "saved_rank",
        "saved_score",
        PROFILE_NAME,
        "validation_signal",
        "validation_structure",
        "validation_bead",
        "validation_sharpness",
        "validation_signal_structure_score",
        "registration_status",
        "registration_profile",
        "registration_confidence",
        "registration_suspect",
        "registration_suspect_reasons",
        "matched_bead_count",
        "median_bead_error_px",
        "common_fov_fraction",
        "anchor_selection_reason",
        "registration_constraint_level",
        "registration_underconstrained",
        "registration_initializer",
        "registration_calibration_group",
        "registration_transform_source",
        "registration_point_count",
        "registration_fit_rms_px",
        "validation_rankable",
        "ranking_exclusion_reason",
        "registration_refined_score",
        "registration_runtime_seconds",
        "hypotheses_generated",
        "hypotheses_prefiltered",
        "hypotheses_dense_scored",
        "hypotheses_refined",
        "optimizer_evaluations",
        "registration_scale",
        "registration_rotation_deg",
        "registration_translation_x",
        "registration_translation_y",
        "residual_translation_x",
        "residual_translation_y",
        "residual_translation_norm",
        "residual_applied",
        "params_json",
        "config_json",
        "workspace_path",
        "preview_path",
        "warnings",
    ]
    write_csv_atomic(path, fieldnames, [_row_to_csv(row) for row in rows], create_backup=False)


def _write_registration_hypotheses_csv(path: Path, rows: Sequence[BenchmarkResultRow]) -> None:
    fieldnames = [
        "source_kind",
        "row_id",
        "candidate_id",
        "hypothesis_rank",
        "initializer",
        "constraint_level",
        "score",
        "refined_score",
        "matched_bead_count",
        "median_bead_error_px",
        "common_fov_fraction",
        "optimizer_evaluations",
        "registration_runtime_seconds",
        "hypotheses_generated",
        "hypotheses_prefiltered",
        "hypotheses_dense_scored",
        "hypotheses_refined",
        "scale",
        "rotation_deg",
        "translation_x",
        "translation_y",
    ]
    payloads: List[Dict[str, Any]] = []
    for row in rows:
        for item in row.registration.registration_hypotheses:
            payloads.append(
                {
                    "source_kind": row.source_kind,
                    "row_id": row.row_id,
                    "candidate_id": row.candidate_id,
                    "hypothesis_rank": item.get("rank", ""),
                    "initializer": item.get("initializer", ""),
                    "constraint_level": item.get("constraint_level", ""),
                    "score": _csv_float(_optional_float(item.get("score"))),
                    "refined_score": _csv_float(_optional_float(item.get("refined_score"))),
                    "matched_bead_count": item.get("matched_bead_count", ""),
                    "median_bead_error_px": _csv_float(_optional_float(item.get("median_bead_error_px"))),
                    "common_fov_fraction": _csv_float(_optional_float(item.get("common_fov_fraction"))),
                    "optimizer_evaluations": item.get("optimizer_evaluations", ""),
                    "registration_runtime_seconds": _csv_float(row.registration.registration_runtime_seconds),
                    "hypotheses_generated": row.registration.hypotheses_generated,
                    "hypotheses_prefiltered": row.registration.hypotheses_prefiltered,
                    "hypotheses_dense_scored": row.registration.hypotheses_dense_scored,
                    "hypotheses_refined": row.registration.hypotheses_refined,
                    "scale": _csv_float(_optional_float(item.get("scale"))),
                    "rotation_deg": _csv_float(_optional_float(item.get("rotation_deg"))),
                    "translation_x": _csv_float(_optional_float(item.get("translation_x"))),
                    "translation_y": _csv_float(_optional_float(item.get("translation_y"))),
                }
            )
    write_csv_atomic(path, fieldnames, payloads, create_backup=False)


def _row_to_csv(row: BenchmarkResultRow) -> Dict[str, Any]:
    reg = row.registration
    payload: Dict[str, Any] = {
        "source_kind": row.source_kind,
        "validation_rank": "" if row.validation_rank is None else row.validation_rank,
        "row_id": row.row_id,
        "candidate_id": row.candidate_id,
        "stage": row.stage,
        "dataset_id": row.dataset_id,
        "generation_id": row.generation_id or "",
        "strategy": row.strategy,
        "saved_rank": "" if row.saved_rank is None else row.saved_rank,
        "saved_score": _csv_float(row.saved_score),
        "registration_status": reg.status,
        "registration_profile": reg.registration_profile,
        "registration_confidence": _csv_float(reg.registration_confidence),
        "registration_suspect": str(bool(reg.registration_suspect)).lower(),
        "registration_suspect_reasons": "|".join(reg.registration_suspect_reasons),
        "matched_bead_count": reg.matched_bead_count,
        "median_bead_error_px": _csv_float(reg.median_bead_error_px),
        "common_fov_fraction": _csv_float(reg.common_fov_fraction),
        "anchor_selection_reason": reg.anchor_selection_reason or "",
        "registration_constraint_level": reg.registration_constraint_level,
        "registration_underconstrained": str(bool(reg.registration_underconstrained)).lower(),
        "registration_initializer": reg.registration_initializer,
        "registration_calibration_group": reg.registration_calibration_group or "",
        "registration_transform_source": reg.registration_transform_source or "",
        "registration_point_count": reg.registration_point_count,
        "registration_fit_rms_px": _csv_float(reg.registration_fit_rms_px),
        "validation_rankable": str(bool(reg.validation_rankable)).lower(),
        "ranking_exclusion_reason": reg.ranking_exclusion_reason or "",
        "registration_refined_score": _csv_float(reg.registration_refined_score),
        "registration_runtime_seconds": _csv_float(reg.registration_runtime_seconds),
        "hypotheses_generated": reg.hypotheses_generated,
        "hypotheses_prefiltered": reg.hypotheses_prefiltered,
        "hypotheses_dense_scored": reg.hypotheses_dense_scored,
        "hypotheses_refined": reg.hypotheses_refined,
        "optimizer_evaluations": reg.optimizer_evaluations,
        "registration_scale": _csv_float(reg.scale),
        "registration_rotation_deg": _csv_float(reg.rotation_deg),
        "registration_translation_x": _csv_float(reg.translation_x),
        "registration_translation_y": _csv_float(reg.translation_y),
        "residual_translation_x": _csv_float(reg.residual_translation_x),
        "residual_translation_y": _csv_float(reg.residual_translation_y),
        "residual_translation_norm": _csv_float(reg.residual_translation_norm),
        "residual_applied": str(bool(reg.residual_applied)).lower(),
        "params_json": json.dumps(row.params, sort_keys=True, separators=(",", ":")),
        "config_json": json.dumps(row.config, sort_keys=True, separators=(",", ":")),
        "workspace_path": row.workspace_path or "",
        "preview_path": row.preview_path or "",
        "warnings": "; ".join(row.warnings),
    }
    for metric_name in [
        PROFILE_NAME,
        "validation_signal",
        "validation_structure",
        "validation_bead",
        "validation_sharpness",
        "validation_signal_structure_score",
    ]:
        payload[metric_name] = _csv_float(row.metrics.get(metric_name))
    return payload


def _build_summary(
    config: BenchmarkConfig,
    stage: str,
    anchor_record: CandidateImageRecord,
    anchor_registration: SimilarityRegistrationResult,
    best: Optional[BenchmarkResultRow],
    thor_row: BenchmarkResultRow,
    rows: Sequence[BenchmarkResultRow],
    *,
    model_warnings: Sequence[str],
    skipped_candidate_warnings: Sequence[str],
    reference_sources: ReferenceImages,
) -> Dict[str, Any]:
    deltas: Dict[str, Optional[float]] = {}
    if best is not None:
        for name, value in best.metrics.items():
            thor_value = thor_row.metrics.get(name)
            if isinstance(value, (int, float)) and isinstance(thor_value, (int, float)):
                deltas[name] = float(value) - float(thor_value)
    warnings = _unique_strings(
        list(model_warnings)
        + list(reference_sources.warnings)
        + list(skipped_candidate_warnings)
        + [warning for row in rows for warning in row.warnings]
        + (["nonrigid diagnostic requested; primary score remains similarity-only"] if config.enable_nonrigid_diagnostic else [])
    )
    return {
        "config": {
            "profile_name": PROFILE_NAME,
            "sweep_dir": str(config.sweep_dir),
            "dataset_id": config.dataset_id,
            "scene_id": config.scene_id,
            "validation_image": reference_sources.validation_source,
            "thorcam_image": reference_sources.thorcam_source,
            "validation_workspace": None if config.validation_workspace is None else str(config.validation_workspace),
            "output_dir": str(config.output_dir),
            "stage": stage,
            "generation_scope": config.generation_scope,
            "anchor_row_id": config.anchor_row_id,
            "scale_values": list(config.scale_values or []),
            "rotation_values": list(config.rotation_values or []),
            "top_k": config.top_k,
            "residual_translation_max_px": config.residual_translation_max_px,
            "registration_profile": config.registration_profile,
            "use_validation_workspace_transforms": config.use_validation_workspace_transforms,
            "registration_annotations": None if config.registration_annotations is None else str(config.registration_annotations),
            "registration_calibration_group": config.registration_calibration_group,
            "registration_calibration_mode": config.registration_calibration_mode,
            "registration_suspect_residual_px": config.registration_suspect_residual_px,
            "registration_min_matched_beads": config.registration_min_matched_beads,
            "registration_min_structure_score": config.registration_min_structure_score,
            "registration_top_hypotheses": config.registration_top_hypotheses,
            "registration_diagnostics_top_n": config.registration_diagnostics_top_n,
            "registration_coarse_max_dim": config.registration_coarse_max_dim,
            "registration_refine_max_dim": config.registration_refine_max_dim,
            "registration_prefilter_hypotheses": config.registration_prefilter_hypotheses,
            "registration_dense_hypotheses": config.registration_dense_hypotheses,
            "registration_refine_max_evals": config.registration_refine_max_evals,
            "registration_progress": config.registration_progress,
            "enable_nonrigid_diagnostic": config.enable_nonrigid_diagnostic,
        },
        "best_candidate": None if best is None else _row_to_summary(best),
        "thorcam_baseline": _row_to_summary(thor_row),
        "best_lf_minus_thorcam": deltas,
        "anchor": {
            "row_id": str(getattr(anchor_record.row, "row_id", "")),
            "candidate_id": str(getattr(anchor_record.row, "candidate_id", "")),
            "registration": _registration_summary(anchor_registration),
        },
        "num_lf_candidates": sum(1 for row in rows if row.source_kind == "lf_candidate"),
        "warnings": warnings,
    }


def _row_to_summary(row: BenchmarkResultRow) -> Dict[str, Any]:
    return {
        "source_kind": row.source_kind,
        "row_id": row.row_id,
        "candidate_id": row.candidate_id,
        "stage": row.stage,
        "dataset_id": row.dataset_id,
        "generation_id": row.generation_id,
        "strategy": row.strategy,
        "saved_rank": row.saved_rank,
        "validation_rank": row.validation_rank,
        "saved_score": row.saved_score,
        "metrics": row.metrics,
        "registration": _registration_summary(row.registration),
        "workspace_path": row.workspace_path,
        "preview_path": row.preview_path,
        "warnings": row.warnings,
        "params": row.params,
        "config": row.config,
    }


def _registration_summary(reg: SimilarityRegistrationResult) -> Dict[str, Any]:
    return {
        "status": reg.status,
        "registration_profile": reg.registration_profile,
        "registration_confidence": reg.registration_confidence,
        "registration_suspect": reg.registration_suspect,
        "registration_suspect_reasons": reg.registration_suspect_reasons,
        "matched_bead_count": reg.matched_bead_count,
        "median_bead_error_px": reg.median_bead_error_px,
        "common_fov_fraction": reg.common_fov_fraction,
        "anchor_selection_reason": reg.anchor_selection_reason,
        "registration_constraint_level": reg.registration_constraint_level,
        "registration_underconstrained": reg.registration_underconstrained,
        "registration_initializer": reg.registration_initializer,
        "registration_calibration_group": reg.registration_calibration_group,
        "registration_transform_source": reg.registration_transform_source,
        "registration_point_count": reg.registration_point_count,
        "registration_fit_rms_px": reg.registration_fit_rms_px,
        "validation_rankable": reg.validation_rankable,
        "ranking_exclusion_reason": reg.ranking_exclusion_reason,
        "registration_point_pairs": reg.registration_point_pairs,
        "registration_refined_score": reg.registration_refined_score,
        "registration_runtime_seconds": reg.registration_runtime_seconds,
        "hypotheses_generated": reg.hypotheses_generated,
        "hypotheses_prefiltered": reg.hypotheses_prefiltered,
        "hypotheses_dense_scored": reg.hypotheses_dense_scored,
        "hypotheses_refined": reg.hypotheses_refined,
        "optimizer_evaluations": reg.optimizer_evaluations,
        "registration_timing_seconds": reg.registration_timing_seconds,
        "registration_hypotheses": reg.registration_hypotheses,
        "scale": reg.scale,
        "rotation_deg": reg.rotation_deg,
        "translation_x": reg.translation_x,
        "translation_y": reg.translation_y,
        "ecc_score": reg.ecc_score,
        "residual_translation_x": reg.residual_translation_x,
        "residual_translation_y": reg.residual_translation_y,
        "residual_translation_norm": reg.residual_translation_norm,
        "residual_applied": reg.residual_applied,
        "forward_matrix": np.asarray(reg.forward_matrix, dtype=float).tolist(),
        "warnings": reg.warnings,
    }


def _render_report_html(summary: Mapping[str, Any], rows: Sequence[BenchmarkResultRow], figures_dir: Path) -> str:
    table_rows = []
    for row in rows:
        score = row.metrics.get(PROFILE_NAME)
        overlay = _overlay_filename(row)
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row.source_kind)}</td>"
            f"<td>{'' if row.validation_rank is None else row.validation_rank}</td>"
            f"<td>{html.escape(row.row_id)}</td>"
            f"<td>{html.escape(row.candidate_id)}</td>"
            f"<td>{_html_float(score)}</td>"
            f"<td>{_html_float(row.metrics.get('validation_signal'))}</td>"
            f"<td>{_html_float(row.metrics.get('validation_structure'))}</td>"
            f"<td>{_html_float(row.metrics.get('validation_bead'))}</td>"
            f"<td>{_html_float(row.metrics.get('validation_sharpness'))}</td>"
            f"<td>{_html_float(row.registration.registration_confidence)}</td>"
            f"<td>{html.escape(str(row.registration.registration_suspect).lower())}</td>"
            f"<td>{html.escape(row.registration.registration_constraint_level)}</td>"
            f"<td>{html.escape(row.registration.registration_initializer)}</td>"
            f"<td>{html.escape(row.registration.registration_transform_source or '')}</td>"
            f"<td>{html.escape(str(row.registration.validation_rankable).lower())}</td>"
            f"<td>{_html_float(row.registration.residual_translation_norm)}</td>"
            f"<td><a href=\"figures/{html.escape(overlay)}\">overlay</a></td>"
            f"<td>{html.escape('; '.join(row.warnings))}</td>"
            "</tr>"
        )
    best = cast(Optional[Mapping[str, Any]], summary.get("best_candidate"))
    best_text = "None" if best is None else str(best.get("row_id", ""))
    return "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset=\"utf-8\"><title>Trio Benchmark Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ddd;padding:6px;font-size:13px;}th{background:#f2f2f2;text-align:left;}"
            "code{background:#f7f7f7;padding:2px 4px;}</style></head><body>",
            "<h1>Trio Benchmark Report</h1>",
            f"<p><strong>Profile:</strong> <code>{PROFILE_NAME}</code></p>",
            f"<p><strong>Best LF row:</strong> {html.escape(best_text)}</p>",
            "<h2>Ranking</h2>",
            "<table><thead><tr><th>Source</th><th>Rank</th><th>Row ID</th><th>Candidate</th>"
            "<th>Hybrid</th><th>Signal</th><th>Structure</th><th>Bead</th><th>Sharpness</th>"
            "<th>Reg Conf</th><th>Reg Suspect</th><th>Constraint</th><th>Initializer</th>"
            "<th>Transform Source</th><th>Rankable</th><th>Residual Norm</th><th>Figure</th><th>Warnings</th></tr></thead><tbody>",
            *table_rows,
            "</tbody></table>",
            "<h2>Summary JSON</h2>",
            f"<pre>{html.escape(str(summary))}</pre>",
            "</body></html>",
        ]
    )


def _write_comparison_figures(figures_dir: Path, validation: np.ndarray, row: BenchmarkResultRow) -> None:
    prefix = _figure_prefix(row)
    val = _robust_unit(validation)
    cand = _robust_unit(row.registration.aligned_image)
    if val.shape != cand.shape:
        min_h = min(val.shape[0], cand.shape[0])
        min_w = min(val.shape[1], cand.shape[1])
        val = val[:min_h, :min_w]
        cand = cand[:min_h, :min_w]
    overlay = np.zeros((val.shape[0], val.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = val
    overlay[..., 1] = cand
    diff = np.abs(val - cand)
    _save_unit_image(figures_dir / f"{prefix}_overlay.png", overlay)
    _save_unit_image(figures_dir / f"{prefix}_diff.png", diff)


def _write_affine_diagnostic_figure(figures_dir: Path, validation: np.ndarray, row: BenchmarkResultRow) -> None:
    affine_aligned = _affine_diagnostic_align(validation, row.registration.aligned_image)
    val = _robust_unit(validation)
    cand = _robust_unit(affine_aligned)
    if val.shape != cand.shape:
        min_h = min(val.shape[0], cand.shape[0])
        min_w = min(val.shape[1], cand.shape[1])
        val = val[:min_h, :min_w]
        cand = cand[:min_h, :min_w]
    overlay = np.zeros((val.shape[0], val.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = val
    overlay[..., 1] = cand
    _save_unit_image(figures_dir / f"{_figure_prefix(row)}_affine_diagnostic_overlay.png", overlay)


def _affine_diagnostic_align(validation: np.ndarray, similarity_aligned: np.ndarray) -> np.ndarray:
    ref = _as_float_2d(validation)
    mov = _as_float_2d(similarity_aligned)
    if ref.shape != mov.shape or ref.size == 0:
        return mov
    ref_norm = _normalize_for_registration(ref)
    mov_norm = _normalize_for_registration(mov)
    if float(np.std(ref_norm)) <= 1e-8 or float(np.std(mov_norm)) <= 1e-8:
        return mov
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 150, 1e-6)
    try:
        _cc, warp = cv.findTransformECC(ref_norm, mov_norm, warp, cv.MOTION_AFFINE, criteria, None, 5)
        return cv.warpAffine(
            mov,
            warp,
            (ref.shape[1], ref.shape[0]),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)
    except cv.error:
        return mov


def _write_bead_detection_figure(path: Path, image: np.ndarray) -> None:
    unit = _robust_unit(image)
    channel = _bead_registration_channel(unit)
    beads = _detect_registration_beads(channel)
    rgb = np.dstack([unit, unit, unit]).astype(np.float32)
    out = np.asarray(np.clip(rgb * 255.0, 0.0, 255.0), dtype=np.uint8)
    for bead in beads[:80]:
        center = (int(round(bead.x)), int(round(bead.y)))
        radius = max(3, int(round(bead.radius + 2.0)))
        cv.circle(out, center, radius, (255, 0, 0), 1)
        cv.circle(out, center, 1, (0, 255, 255), -1)
    Image.fromarray(out).save(path)


def _write_manual_points_figure(
    path: Path,
    validation: np.ndarray,
    moving: np.ndarray,
    registration: SimilarityRegistrationResult,
) -> None:
    val = _robust_unit(validation)
    aligned = _robust_unit(
        cv.warpAffine(
            _as_float_2d(moving),
            np.asarray(registration.forward_matrix, dtype=np.float32),
            (val.shape[1], val.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )
    )
    overlay = np.zeros((val.shape[0], val.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = val
    overlay[..., 1] = aligned
    out = np.asarray(np.clip(overlay * 255.0, 0.0, 255.0), dtype=np.uint8)
    for pair in registration.registration_point_pairs[:200]:
        reference = pair.get("reference", [])
        moving_point = pair.get("moving", [])
        if len(reference) >= 2:
            cv.circle(out, (int(round(reference[0])), int(round(reference[1]))), 4, (255, 0, 0), 1)
        if len(moving_point) >= 2:
            transformed = _transform_points(registration.forward_matrix, np.asarray([moving_point], dtype=np.float64))[0]
            cv.drawMarker(
                out,
                (int(round(transformed[0])), int(round(transformed[1]))),
                (0, 255, 255),
                markerType=cv.MARKER_CROSS,
                markerSize=9,
                thickness=1,
            )
    Image.fromarray(out).save(path)


def _write_manual_calibration_artifact(
    path: Path,
    config: BenchmarkConfig,
    lf_registration: SimilarityRegistrationResult,
    thor_registration: SimilarityRegistrationResult,
) -> None:
    payload = {
        "version": 1,
        "registration_profile": REGISTRATION_PROFILE_MANUAL_SIMILARITY,
        "annotations_path": None if config.registration_annotations is None else str(config.registration_annotations),
        "calibration_group": config.registration_calibration_group,
        "calibration_mode": config.registration_calibration_mode,
        "scene_id": config.scene_id,
        "dataset_id": config.dataset_id,
        "sources": {
            "lf": _manual_registration_artifact_summary(lf_registration),
            "thorcam": _manual_registration_artifact_summary(thor_registration),
        },
    }
    write_json_atomic(path, payload, sort_keys=True, create_backup=False)


def _write_validation_workspace_transform_artifact(
    path: Path,
    config: BenchmarkConfig,
    references: ReferenceImages,
    lf_registration: SimilarityRegistrationResult,
    thor_registration: SimilarityRegistrationResult,
) -> None:
    payload = {
        "version": 1,
        "registration_profile": REGISTRATION_PROFILE_VALIDATE_WORKSPACE,
        "validation_workspace": None if config.validation_workspace is None else str(config.validation_workspace),
        "scene_id": config.scene_id,
        "dataset_id": config.dataset_id,
        "workspace_transform_info": references.workspace_transform_info,
        "sources": {
            "lf": _manual_registration_artifact_summary(lf_registration),
            "thorcam": _manual_registration_artifact_summary(thor_registration),
        },
        "warnings": references.warnings,
    }
    write_json_atomic(path, payload, sort_keys=True, create_backup=False)


def _write_workspace_transform_diagnostic_figure(path: Path, reference: np.ndarray, warped: np.ndarray) -> None:
    ref = _robust_unit(reference)
    mov = _robust_unit(warped)
    if ref.shape != mov.shape:
        min_h = min(ref.shape[0], mov.shape[0])
        min_w = min(ref.shape[1], mov.shape[1])
        ref = ref[:min_h, :min_w]
        mov = mov[:min_h, :min_w]
    overlay = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = ref
    overlay[..., 1] = mov
    _save_unit_image(path, overlay)


def _manual_registration_artifact_summary(registration: SimilarityRegistrationResult) -> Dict[str, Any]:
    return {
        "status": registration.status,
        "transform_source": registration.registration_transform_source,
        "initializer": registration.registration_initializer,
        "point_count": registration.registration_point_count,
        "fit_rms_px": registration.registration_fit_rms_px,
        "underconstrained": registration.registration_underconstrained,
        "rankable": registration.validation_rankable,
        "ranking_exclusion_reason": registration.ranking_exclusion_reason,
        "scale": registration.scale,
        "rotation_deg": registration.rotation_deg,
        "translation_x": registration.translation_x,
        "translation_y": registration.translation_y,
        "forward_matrix": np.asarray(registration.forward_matrix, dtype=float).tolist(),
        "point_pairs": registration.registration_point_pairs,
        "warnings": registration.warnings,
    }


def _figure_prefix(row: BenchmarkResultRow) -> str:
    if row.source_kind == "thorcam_baseline":
        return "baseline_thorcam"
    rank = row.validation_rank or 0
    return f"rank_{rank:03d}_{_safe_slug(row.stage)}_{_safe_slug(row.candidate_id)}"


def _overlay_filename(row: BenchmarkResultRow) -> str:
    return f"{_figure_prefix(row)}_overlay.png"


def _save_unit_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        out = np.asarray(np.round(arr * 255.0), dtype=np.uint8)
    else:
        out = np.asarray(np.round(arr * 255.0), dtype=np.uint8)
    Image.fromarray(out).save(path)


def _load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image)
    return _as_float_2d(arr)


def _as_float_2d(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.float32)
            arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _normalize_for_registration(image: np.ndarray) -> np.ndarray:
    return np.asarray(_robust_unit(image), dtype=np.float32)


def _normalize_registration_profile(profile: str) -> str:
    text = str(profile or DEFAULT_REGISTRATION_PROFILE).strip()
    if text in {
        REGISTRATION_PROFILE_VALIDATE_WORKSPACE,
        REGISTRATION_PROFILE_MANUAL_SIMILARITY,
        REGISTRATION_PROFILE_BEAD_GEOMETRIC,
        REGISTRATION_PROFILE_BEAD_SIMILARITY,
        REGISTRATION_PROFILE_CENTERED_ECC,
    }:
        return text
    raise ValueError(
        "unsupported registration profile "
        f"{profile!r}; expected {REGISTRATION_PROFILE_BEAD_GEOMETRIC!r}, "
        f"{REGISTRATION_PROFILE_BEAD_SIMILARITY!r}, {REGISTRATION_PROFILE_CENTERED_ECC!r}, "
        f"{REGISTRATION_PROFILE_MANUAL_SIMILARITY!r}, or {REGISTRATION_PROFILE_VALIDATE_WORKSPACE!r}"
    )


def _default_scale_values_for_registration_profile(profile: str) -> List[float]:
    if profile == REGISTRATION_PROFILE_BEAD_GEOMETRIC:
        return parse_scale_search(DEFAULT_GEOMETRIC_SCALE_SEARCH)
    if profile == REGISTRATION_PROFILE_MANUAL_SIMILARITY:
        return [1.0]
    if profile == REGISTRATION_PROFILE_VALIDATE_WORKSPACE:
        return [1.0]
    return parse_scale_search(DEFAULT_SCALE_SEARCH)


def _registration_channel(image: np.ndarray) -> np.ndarray:
    unit = _normalize_for_registration(image)
    if unit.size == 0:
        return unit
    small = cv.GaussianBlur(unit, (0, 0), 1.0)
    broad = cv.GaussianBlur(unit, (0, 0), 6.0)
    bead = np.maximum(small - broad, 0.0)
    structure = _structure_channel(unit)
    return _robust_unit(0.65 * bead + 0.35 * structure)


def _downsample_registration_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    max_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    ref, ref_scale = _downsample_for_registration(reference, max_dim=max_dim)
    mov, mov_scale = _downsample_for_registration(moving, max_dim=max_dim)
    return ref, mov, ref_scale, mov_scale


def _downsample_for_registration(image: np.ndarray, *, max_dim: int) -> Tuple[np.ndarray, float]:
    arr = np.asarray(image, dtype=np.float32)
    longest = max(arr.shape[:2]) if arr.ndim >= 2 else 0
    if longest <= 0 or longest <= max_dim:
        return arr, 1.0
    scale = float(max_dim) / float(longest)
    resized = cv.resize(arr, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return np.asarray(resized, dtype=np.float32), scale


def _coarse_registration_rotations() -> List[float]:
    return [float(value) for value in np.linspace(-30.0, 30.0, 13)]


def _phase_translation_on_channels(source: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if src.shape != tgt.shape or src.size == 0:
        return 0.0, 0.0
    if float(np.std(src)) <= 1e-8 or float(np.std(tgt)) <= 1e-8:
        return 0.0, 0.0
    try:
        shift, _response = cv.phaseCorrelate(src, tgt)
        return float(shift[0]), float(shift[1])
    except cv.error:
        return 0.0, 0.0


def _registration_channel_score(reference_channel: np.ndarray, moving_channel: np.ndarray) -> float:
    corr = _correlation(reference_channel, moving_channel)
    mae = float(np.mean(np.abs(reference_channel - moving_channel)))
    return 0.80 * max(0.0, corr) + 0.20 * (1.0 - min(1.0, mae))


def _scale_affine_between_resolutions(matrix_small: np.ndarray, *, ref_scale: float, mov_scale: float) -> np.ndarray:
    small = np.vstack([np.asarray(matrix_small, dtype=np.float64), [0.0, 0.0, 1.0]])
    source_scale = np.array([[mov_scale, 0.0, 0.0], [0.0, mov_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    reference_unscale = np.array(
        [[1.0 / ref_scale, 0.0, 0.0], [0.0, 1.0 / ref_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.asarray((reference_unscale @ small @ source_scale)[:2, :], dtype=np.float64)


def _scale_affine_to_resolution(matrix_full: np.ndarray, *, ref_scale: float, mov_scale: float) -> np.ndarray:
    full = np.vstack([np.asarray(matrix_full, dtype=np.float64), [0.0, 0.0, 1.0]])
    moving_unscale = np.array(
        [[1.0 / mov_scale, 0.0, 0.0], [0.0, 1.0 / mov_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    reference_scale = np.array([[ref_scale, 0.0, 0.0], [0.0, ref_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray((reference_scale @ full @ moving_unscale)[:2, :], dtype=np.float64)


def _robust_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray(np.clip((arr - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def _pre_scale_matrix(source_shape: Tuple[int, int], reference_shape: Tuple[int, int], scale: float) -> np.ndarray:
    src_h, src_w = source_shape[:2]
    ref_h, ref_w = reference_shape[:2]
    offset_x = (ref_w - (src_w * scale)) / 2.0
    offset_y = (ref_h - (src_h * scale)) / 2.0
    return np.array([[scale, 0.0, offset_x], [0.0, scale, offset_y]], dtype=np.float32)


def _pre_similarity_matrix(
    source_shape: Tuple[int, int],
    reference_shape: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> np.ndarray:
    src_h, src_w = source_shape[:2]
    ref_h, ref_w = reference_shape[:2]
    source_center = (src_w / 2.0, src_h / 2.0)
    matrix = cv.getRotationMatrix2D(source_center, float(rotation_deg), float(scale)).astype(np.float64)
    matrix[0, 2] += (ref_w - src_w) / 2.0
    matrix[1, 2] += (ref_h - src_h) / 2.0
    return np.asarray(matrix, dtype=np.float32)


def _combine_forward_matrix(pre_matrix: np.ndarray, ecc_warp: np.ndarray) -> np.ndarray:
    pre3 = np.vstack([np.asarray(pre_matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
    warp3 = np.vstack([np.asarray(ecc_warp, dtype=np.float64), [0.0, 0.0, 1.0]])
    try:
        combined = np.linalg.inv(warp3) @ pre3
    except np.linalg.LinAlgError:
        combined = pre3
    return np.asarray(combined[:2, :], dtype=np.float64)


def _matrix_multiply_2x3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left3 = np.vstack([np.asarray(left, dtype=np.float64), [0.0, 0.0, 1.0]])
    right3 = np.vstack([np.asarray(right, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.asarray((left3 @ right3)[:2, :], dtype=np.float64)


def _decompose_similarity(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    mat = np.asarray(matrix, dtype=np.float64)
    a = float(mat[0, 0])
    b = float(mat[1, 0])
    scale = float(math.hypot(a, b))
    rotation = float(math.degrees(math.atan2(b, a))) if scale > 0.0 else 0.0
    return scale, rotation, float(mat[0, 2]), float(mat[1, 2])


def _phase_translation(source: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    src = _normalize_for_registration(source)
    tgt = _normalize_for_registration(target)
    if src.shape != tgt.shape or src.size == 0:
        return 0.0, 0.0
    if float(np.std(src)) <= 1e-8 or float(np.std(tgt)) <= 1e-8:
        return 0.0, 0.0
    try:
        shift, _response = cv.phaseCorrelate(src, tgt)
        return float(shift[0]), float(shift[1])
    except cv.error:
        return 0.0, 0.0


def _bead_residual_translation(source: np.ndarray, target: np.ndarray) -> Optional[Tuple[float, float]]:
    source_centers = _detect_bead_centers(source)
    target_centers = _detect_bead_centers(target)
    if not source_centers or not target_centers:
        return None
    deltas: List[Tuple[float, float]] = []
    used: set[int] = set()
    for sx, sy in source_centers:
        best_idx = -1
        best_dist = float("inf")
        for idx, (tx, ty) in enumerate(target_centers):
            if idx in used:
                continue
            dist = float(math.hypot(tx - sx, ty - sy))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0 and best_dist <= 25.0:
            used.add(best_idx)
            tx, ty = target_centers[best_idx]
            deltas.append((tx - sx, ty - sy))
    if not deltas:
        return None
    dx = float(np.median([item[0] for item in deltas]))
    dy = float(np.median([item[1] for item in deltas]))
    return dx, dy


def _apply_best_translation(source: np.ndarray, target: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, float, float]:
    candidates = []
    for sx, sy in [(dx, dy), (-dx, -dy)]:
        matrix = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], dtype=np.float32)
        shifted = cv.warpAffine(
            source,
            matrix,
            (target.shape[1], target.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0.0,
        )
        score = float(np.mean(np.abs(_robust_unit(shifted) - _robust_unit(target))))
        candidates.append((score, shifted, float(sx), float(sy)))
    _score, shifted, sx, sy = min(candidates, key=lambda item: item[0])
    return shifted, sx, sy


def _fit_gain_offset(candidate: np.ndarray, validation: np.ndarray) -> np.ndarray:
    x = np.asarray(candidate, dtype=np.float64).reshape(-1)
    y = np.asarray(validation, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return np.asarray(candidate, dtype=np.float32)
    x_valid = x[mask]
    y_valid = y[mask]
    design = np.column_stack([x_valid, np.ones_like(x_valid)])
    try:
        gain, offset = np.linalg.lstsq(design, y_valid, rcond=None)[0]
    except np.linalg.LinAlgError:
        gain, offset = 1.0, 0.0
    adjusted = np.asarray(candidate, dtype=np.float64) * float(gain) + float(offset)
    return np.asarray(np.clip(adjusted, 0.0, 1.0), dtype=np.float32)


def _signal_score(validation: np.ndarray, candidate: np.ndarray) -> float:
    mae = float(np.mean(np.abs(validation - candidate)))
    return 1.0 - min(1.0, mae)


def _structure_score(validation: np.ndarray, candidate: np.ndarray) -> float:
    val_repr = _structure_channel(validation)
    cand_repr = _structure_channel(candidate)
    return max(0.0, _correlation(val_repr, cand_repr))


def _structure_channel(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    grad_x = cv.Sobel(arr, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(arr, cv.CV_32F, 0, 1, ksize=3)
    grad = cv.magnitude(grad_x, grad_y)
    log = np.abs(cv.Laplacian(cv.GaussianBlur(arr, (0, 0), 1.0), cv.CV_32F))
    return _robust_unit(0.5 * grad + 0.5 * log)


def _detect_bead_centers(image: np.ndarray) -> List[Tuple[float, float]]:
    arr = _robust_unit(image)
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return []
    blurred = cv.GaussianBlur(arr, (0, 0), 1.0)
    threshold = max(0.20, float(np.percentile(blurred, 99.0)))
    mask = np.asarray(blurred >= threshold, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    centers: List[Tuple[float, float]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area < 2:
            continue
        x0 = int(stats[label, cv.CC_STAT_LEFT])
        y0 = int(stats[label, cv.CC_STAT_TOP])
        w = int(stats[label, cv.CC_STAT_WIDTH])
        h = int(stats[label, cv.CC_STAT_HEIGHT])
        patch = blurred[y0 : y0 + h, x0 : x0 + w]
        patch_mask = labels[y0 : y0 + h, x0 : x0 + w] == label
        weights = np.where(patch_mask, patch, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-12:
            cx, cy = centroids[label]
        else:
            yy, xx = np.indices(weights.shape)
            cx = x0 + float(np.sum(xx * weights) / total)
            cy = y0 + float(np.sum(yy * weights) / total)
        centers.append((float(cx), float(cy)))
    return centers


def _bead_score(validation_centers: Sequence[Tuple[float, float]], candidate_centers: Sequence[Tuple[float, float]]) -> float:
    if not validation_centers:
        return 0.0
    if not candidate_centers:
        return 0.0
    radius = 5.0
    used: set[int] = set()
    distances: List[float] = []
    misses = 0
    for vx, vy in validation_centers:
        best_idx = -1
        best_dist = float("inf")
        for idx, (cx, cy) in enumerate(candidate_centers):
            if idx in used:
                continue
            dist = float(math.hypot(cx - vx, cy - vy))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0 and best_dist <= radius:
            used.add(best_idx)
            distances.append(best_dist)
        else:
            misses += 1
    false_positives = max(0, len(candidate_centers) - len(used))
    mean_dist = float(np.mean(distances)) if distances else radius
    miss_rate = misses / max(1, len(validation_centers))
    false_rate = false_positives / max(1, len(candidate_centers))
    distance_penalty = min(1.0, mean_dist / radius)
    return 1.0 - min(1.0, (distance_penalty + miss_rate + false_rate) / 3.0)


def _sharpness_score(validation: np.ndarray, candidate: np.ndarray, centers: Sequence[Tuple[float, float]]) -> float:
    scores: List[float] = []
    for center in centers:
        val_width = _local_width(validation, center)
        cand_width = _local_width(candidate, center)
        if val_width is None or cand_width is None:
            continue
        excess = max(0.0, cand_width - val_width)
        scores.append(float(math.exp(-excess / 3.0)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _local_width(image: np.ndarray, center: Tuple[float, float], radius: int = 8) -> Optional[float]:
    arr = np.asarray(image, dtype=np.float32)
    cx, cy = center
    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(arr.shape[1], int(math.floor(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(arr.shape[0], int(math.floor(cy)) + radius + 1)
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = patch - float(np.min(patch))
    total = float(np.sum(patch))
    if total <= 1e-8:
        return None
    yy, xx = np.indices(patch.shape)
    xx = xx + x0 - cx
    yy = yy + y0 - cy
    r2 = np.square(xx) + np.square(yy)
    return float(math.sqrt(float(np.sum(r2 * patch) / total)))


def _with_registration_quality(
    result: SimilarityRegistrationResult,
    reference: np.ndarray,
    moving_shape: Tuple[int, int],
    *,
    scale_values: Sequence[float],
    residual_suspect_px: float,
    min_matched_beads: int,
    min_structure_score: float,
    reference_cache: Optional[ValidationMetricReference] = None,
) -> SimilarityRegistrationResult:
    ref = _robust_unit(reference)
    aligned = _robust_unit(result.aligned_image)
    if ref.shape != aligned.shape:
        min_h = min(ref.shape[0], aligned.shape[0])
        min_w = min(ref.shape[1], aligned.shape[1])
        ref = ref[:min_h, :min_w]
        aligned = aligned[:min_h, :min_w]
    prepared = (
        reference_cache
        if reference_cache is not None and reference_cache.validation_unit.shape == ref.shape
        else _prepare_validation_metric_reference_from_unit(ref)
    )

    structure = max(0.0, _correlation(prepared.structure_channel, _structure_channel(aligned)))
    reference_centers = prepared.bead_centers
    aligned_centers = _detect_bead_centers(aligned)
    matched_count, median_error = _match_bead_centers(reference_centers, aligned_centers, radius=8.0)
    common_fov = _common_fov_fraction(moving_shape, ref.shape, result.forward_matrix)
    bead_coverage = matched_count / max(1, len(reference_centers))
    bead_error_score = 0.0 if median_error is None else max(0.0, 1.0 - float(median_error) / 8.0)
    bead_quality = bead_coverage * bead_error_score
    confidence = _clip01(0.45 * bead_quality + 0.40 * structure + 0.15 * common_fov)

    reasons: List[str] = []
    if result.residual_translation_norm > max(0.0, float(residual_suspect_px)):
        reasons.append(
            f"residual_translation_norm>{float(residual_suspect_px):.3f}"
        )
    required_matches = min(max(0, int(min_matched_beads)), len(reference_centers))
    if len(reference_centers) == 0 and max(0, int(min_matched_beads)) > 0:
        reasons.append("validation_bead_count=0")
    elif matched_count < required_matches:
        reasons.append(f"matched_bead_count<{required_matches}")
    if structure < float(min_structure_score):
        reasons.append(f"structure_score<{float(min_structure_score):.3f}")
    if _scale_at_search_boundary(result.scale, scale_values):
        reasons.append("scale_at_search_boundary")

    result.registration_confidence = confidence
    result.registration_suspect = bool(reasons)
    result.registration_suspect_reasons = reasons
    result.matched_bead_count = int(matched_count)
    result.median_bead_error_px = None if median_error is None else float(median_error)
    result.common_fov_fraction = float(common_fov)
    if reasons:
        result.warnings = _unique_strings(result.warnings + [f"registration_suspect:{'|'.join(reasons)}"])
    else:
        result.warnings = _unique_strings(result.warnings)
    return result


def _match_bead_centers(
    reference_centers: Sequence[Tuple[float, float]],
    moving_centers: Sequence[Tuple[float, float]],
    *,
    radius: float,
) -> Tuple[int, Optional[float]]:
    if not reference_centers or not moving_centers:
        return 0, None
    used: set[int] = set()
    distances: List[float] = []
    for vx, vy in reference_centers:
        best_idx = -1
        best_dist = float("inf")
        for idx, (cx, cy) in enumerate(moving_centers):
            if idx in used:
                continue
            dist = float(math.hypot(cx - vx, cy - vy))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx >= 0 and best_dist <= radius:
            used.add(best_idx)
            distances.append(best_dist)
    if not distances:
        return 0, None
    return len(distances), float(np.median(distances))


def _common_fov_fraction(
    moving_shape: Tuple[int, int],
    reference_shape: Tuple[int, int],
    forward_matrix: np.ndarray,
) -> float:
    mov_h, mov_w = moving_shape[:2]
    ref_h, ref_w = reference_shape[:2]
    if mov_h <= 0 or mov_w <= 0 or ref_h <= 0 or ref_w <= 0:
        return 0.0
    mask = np.ones((mov_h, mov_w), dtype=np.float32)
    warped = cv.warpAffine(
        mask,
        np.asarray(forward_matrix, dtype=np.float32),
        (ref_w, ref_h),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return float(np.count_nonzero(warped > 0.5)) / float(warped.size)


def _common_fov_fraction_analytic(
    moving_shape: Tuple[int, int],
    reference_shape: Tuple[int, int],
    forward_matrix: np.ndarray,
) -> float:
    mov_h, mov_w = moving_shape[:2]
    ref_h, ref_w = reference_shape[:2]
    if mov_h <= 0 or mov_w <= 0 or ref_h <= 0 or ref_w <= 0:
        return 0.0
    corners = np.array(
        [
            [0.0, 0.0],
            [float(mov_w), 0.0],
            [float(mov_w), float(mov_h)],
            [0.0, float(mov_h)],
        ],
        dtype=np.float64,
    )
    transformed = _transform_points(forward_matrix, corners)
    min_x = float(np.min(transformed[:, 0]))
    max_x = float(np.max(transformed[:, 0]))
    min_y = float(np.min(transformed[:, 1]))
    max_y = float(np.max(transformed[:, 1]))
    width = max(0.0, min(max_x, float(ref_w)) - max(min_x, 0.0))
    height = max(0.0, min(max_y, float(ref_h)) - max(min_y, 0.0))
    return _clip01((width * height) / max(1.0, float(ref_w * ref_h)))


def _scale_at_search_boundary(scale: float, scale_values: Sequence[float]) -> bool:
    values = [float(value) for value in scale_values if math.isfinite(float(value)) and float(value) > 0.0]
    unique = sorted(set(round(value, 12) for value in values))
    if len(unique) <= 1:
        return False
    lower = float(unique[0])
    upper = float(unique[-1])
    tolerance = max(1e-6, 0.01 * max(abs(lower), abs(upper)))
    return abs(float(scale) - lower) <= tolerance or abs(float(scale) - upper) <= tolerance


def _weighted_score(values: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, weight in weights.items():
        value = values.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _score_sort_value(value: Optional[float]) -> float:
    if value is None or not math.isfinite(float(value)):
        return -float("inf")
    return float(value)


def _csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.9g}"


def _html_float(value: Optional[float]) -> str:
    text = _csv_float(value)
    return "" if not text else html.escape(text)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "item"


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _jsonable_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[str(key)] = item
        else:
            result[str(key)] = str(item)
    return result


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank LF sweep candidates against validation/ThorCam image trio.")
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--validation-image", default=None, type=Path)
    parser.add_argument("--thorcam-image", default=None, type=Path)
    parser.add_argument("--validation-workspace", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--generation-scope", default=GENERATION_SCOPE_ALL)
    parser.add_argument("--anchor-row-id", default="auto")
    parser.add_argument("--scale-search", default=None)
    parser.add_argument("--registration-rotation-search", default=DEFAULT_REGISTRATION_ROTATION_SEARCH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--residual-translation-max-px", type=float, default=DEFAULT_RESIDUAL_TRANSLATION_MAX_PX)
    parser.add_argument(
        "--registration-profile",
        default=DEFAULT_REGISTRATION_PROFILE,
        choices=[
            REGISTRATION_PROFILE_VALIDATE_WORKSPACE,
            REGISTRATION_PROFILE_MANUAL_SIMILARITY,
            REGISTRATION_PROFILE_BEAD_GEOMETRIC,
            REGISTRATION_PROFILE_BEAD_SIMILARITY,
            REGISTRATION_PROFILE_CENTERED_ECC,
        ],
    )
    parser.add_argument("--use-validation-workspace-transforms", action="store_true")
    parser.add_argument("--registration-annotations", default=None, type=Path)
    parser.add_argument("--registration-calibration-group", default=DEFAULT_REGISTRATION_CALIBRATION_GROUP)
    parser.add_argument("--registration-calibration-mode", default=DEFAULT_REGISTRATION_CALIBRATION_MODE, choices=["group", "scene"])
    parser.add_argument("--registration-suspect-residual-px", type=float, default=DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX)
    parser.add_argument("--registration-min-matched-beads", type=int, default=DEFAULT_REGISTRATION_MIN_MATCHED_BEADS)
    parser.add_argument("--registration-min-structure-score", type=float, default=DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE)
    parser.add_argument("--registration-top-hypotheses", type=int, default=DEFAULT_REGISTRATION_TOP_HYPOTHESES)
    parser.add_argument("--registration-diagnostics-top-n", type=int, default=DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N)
    parser.add_argument("--registration-coarse-max-dim", type=int, default=DEFAULT_REGISTRATION_COARSE_MAX_DIM)
    parser.add_argument("--registration-refine-max-dim", type=int, default=DEFAULT_REGISTRATION_REFINE_MAX_DIM)
    parser.add_argument("--registration-prefilter-hypotheses", type=int, default=DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES)
    parser.add_argument("--registration-dense-hypotheses", type=int, default=DEFAULT_REGISTRATION_DENSE_HYPOTHESES)
    parser.add_argument("--registration-refine-max-evals", type=int, default=DEFAULT_REGISTRATION_REFINE_MAX_EVALS)
    parser.add_argument("--registration-progress", action="store_true")
    parser.add_argument("--enable-nonrigid-diagnostic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=args.sweep_dir,
            dataset_id=args.dataset_id,
            scene_id=args.dataset_id,
            validation_image=args.validation_image,
            thorcam_image=args.thorcam_image,
            output_dir=args.output_dir,
            validation_workspace=args.validation_workspace,
            stage=args.stage,
            generation_scope=args.generation_scope,
            anchor_row_id=args.anchor_row_id,
            scale_values=None if args.scale_search is None else parse_scale_search(args.scale_search),
            rotation_values=parse_rotation_search(args.registration_rotation_search),
            top_k=args.top_k,
            residual_translation_max_px=args.residual_translation_max_px,
            registration_profile=args.registration_profile,
            use_validation_workspace_transforms=args.use_validation_workspace_transforms,
            registration_annotations=args.registration_annotations,
            registration_calibration_group=args.registration_calibration_group,
            registration_calibration_mode=args.registration_calibration_mode,
            registration_suspect_residual_px=args.registration_suspect_residual_px,
            registration_min_matched_beads=args.registration_min_matched_beads,
            registration_min_structure_score=args.registration_min_structure_score,
            registration_top_hypotheses=args.registration_top_hypotheses,
            registration_diagnostics_top_n=args.registration_diagnostics_top_n,
            registration_coarse_max_dim=args.registration_coarse_max_dim,
            registration_refine_max_dim=args.registration_refine_max_dim,
            registration_prefilter_hypotheses=args.registration_prefilter_hypotheses,
            registration_dense_hypotheses=args.registration_dense_hypotheses,
            registration_refine_max_evals=args.registration_refine_max_evals,
            registration_progress=args.registration_progress,
            enable_nonrigid_diagnostic=args.enable_nonrigid_diagnostic,
        )
    )
    print(f"Wrote trio benchmark to: {args.output_dir}")
    if result.best_lf_row_id is not None:
        print(f"Best LF row: {result.best_lf_row_id}")


if __name__ == "__main__":
    main()

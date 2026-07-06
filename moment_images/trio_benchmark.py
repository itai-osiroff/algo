"""Headless trio benchmark for LF sweep outputs against external references.

The tool ranks completed light-field sweep candidates against a processed
validation-camera image, while reporting a processed ThorCam image as the
baseline comparator. Validation and ThorCam images are never fed back into the
LF rectification algorithm; they are only used for post-run selection/reporting.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import cv2 as cv
import numpy as np
from PIL import Image

from moment_images.reliable_io import write_csv_atomic, write_json_atomic, write_text_atomic
from moment_images.validate_rectification import load_rectified_reference
from moment_images.sweep_visualization import (
    candidate_rows_for_generation_scope,
    load_sweep_visualization_model,
    run_records_for_generation_scope,
)


PROFILE_NAME = "validation_hybrid_v1"
DEFAULT_SCALE_SEARCH = "0.90:1.10:11"
DEFAULT_TOP_K = 12
DEFAULT_RESIDUAL_TRANSLATION_MAX_PX = 5.0
BEAD_LOW_CONFIDENCE_WARNING = "low-confidence bead detection; bead and sharpness terms omitted"


@dataclass(frozen=True)
class BenchmarkConfig:
    sweep_dir: Path
    dataset_id: str
    validation_image: Path
    thorcam_image: Path
    output_dir: Path
    stage: Optional[str] = None
    generation_scope: str = "all"
    anchor_row_id: str = "auto"
    scale_values: Optional[Sequence[float]] = None
    top_k: int = DEFAULT_TOP_K
    residual_translation_max_px: float = DEFAULT_RESIDUAL_TRANSLATION_MAX_PX
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
    record: Any
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


def parse_scale_search(spec: str) -> List[float]:
    """Parse ``start:stop:count`` into an inclusive scale grid."""

    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise ValueError("scale search must use start:stop:count syntax, e.g. 0.90:1.10:11")
    start = float(parts[0])
    stop = float(parts[1])
    count = int(parts[2])
    if count <= 0:
        raise ValueError("scale search count must be positive")
    if not (math.isfinite(start) and math.isfinite(stop)):
        raise ValueError("scale search bounds must be finite")
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("scale search bounds must be positive")
    if count == 1:
        return [start]
    return [float(value) for value in np.linspace(start, stop, count)]


def estimate_similarity_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    scale_values: Sequence[float],
) -> SimilarityRegistrationResult:
    """Register ``moving`` onto ``reference`` using scale sweep + Euclidean ECC."""

    ref = _as_float_2d(reference)
    mov = _as_float_2d(moving)
    if ref.size == 0 or mov.size == 0:
        raise ValueError("reference and moving images must be non-empty")

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
        return SimilarityRegistrationResult(
            status="failed",
            aligned_image=aligned.astype(np.float32),
            forward_matrix=forward,
            scale=scale,
            rotation_deg=rotation_deg,
            translation_x=tx,
            translation_y=ty,
            ecc_score=None,
            ecc_warp_matrix=np.eye(2, 3, dtype=np.float32),
            warnings=warnings or ["similarity registration failed"],
        )

    score, best_scale, pre_matrix, warp, aligned = best
    forward = _combine_forward_matrix(pre_matrix, warp)
    scale, rotation_deg, tx, ty = _decompose_similarity(forward)
    return SimilarityRegistrationResult(
        status="aligned",
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=best_scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=score,
        ecc_warp_matrix=warp.astype(np.float32),
        warnings=warnings,
    )


def apply_anchor_registration(
    reference: np.ndarray,
    moving: np.ndarray,
    anchor: SimilarityRegistrationResult,
    *,
    residual_translation_max_px: float,
) -> SimilarityRegistrationResult:
    """Apply an anchor similarity transform with capped residual translation."""

    ref = _as_float_2d(reference)
    mov = _as_float_2d(moving)
    pre_matrix = _pre_scale_matrix(mov.shape, ref.shape, anchor.scale)
    canvas = cv.warpAffine(
        mov,
        pre_matrix,
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    aligned = cv.warpAffine(
        canvas,
        anchor.ecc_warp_matrix,
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )
    forward = _combine_forward_matrix(pre_matrix, anchor.ecc_warp_matrix)
    warnings = list(anchor.warnings)

    dx, dy = _phase_translation(aligned, ref)
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
    return SimilarityRegistrationResult(
        status=anchor.status,
        aligned_image=aligned.astype(np.float32),
        forward_matrix=forward,
        scale=anchor.scale,
        rotation_deg=rotation_deg,
        translation_x=tx,
        translation_y=ty,
        ecc_score=anchor.ecc_score,
        ecc_warp_matrix=anchor.ecc_warp_matrix.copy(),
        residual_translation_x=float(dx),
        residual_translation_y=float(dy),
        residual_translation_norm=norm,
        residual_applied=residual_applied,
        warnings=_unique_strings(warnings),
    )


def compute_validation_metrics(
    validation_image: np.ndarray,
    candidate_image: np.ndarray,
) -> ValidationMetricResult:
    """Compute validation agreement metrics for one registered candidate."""

    validation = _robust_unit(_as_float_2d(validation_image))
    candidate = _robust_unit(_as_float_2d(candidate_image))
    if validation.shape != candidate.shape:
        min_h = min(validation.shape[0], candidate.shape[0])
        min_w = min(validation.shape[1], candidate.shape[1])
        validation = validation[:min_h, :min_w]
        candidate = candidate[:min_h, :min_w]

    adjusted = _fit_gain_offset(candidate, validation)
    signal = _signal_score(validation, adjusted)
    structure = _structure_score(validation, adjusted)
    signal_structure = _weighted_score(
        {"signal": signal, "structure": structure},
        {"signal": 0.30, "structure": 0.20},
    )

    centers = _detect_bead_centers(validation)
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
        sharpness = _sharpness_score(validation, adjusted, centers)
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


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the trio benchmark and write CSV/JSON/HTML artifacts."""

    cfg = _normalize_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation = _load_grayscale_image(cfg.validation_image)
    thorcam = _load_grayscale_image(cfg.thorcam_image)
    model = load_sweep_visualization_model(cfg.sweep_dir)
    stage = cfg.stage or str(model.primary_stage_name)
    candidate_images = _collect_candidate_images(model, cfg, stage)
    if not candidate_images:
        raise ValueError(
            f"No LF candidates with loadable images found for stage={stage!r}, "
            f"dataset_id={cfg.dataset_id!r}."
        )

    anchor_record = _select_anchor(candidate_images, cfg.anchor_row_id)
    anchor_registration = estimate_similarity_registration(
        validation,
        anchor_record.image,
        scale_values=cast(Sequence[float], cfg.scale_values),
    )

    lf_rows: List[BenchmarkResultRow] = []
    for candidate in candidate_images:
        registration = apply_anchor_registration(
            validation,
            candidate.image,
            anchor_registration,
            residual_translation_max_px=cfg.residual_translation_max_px,
        )
        metrics = compute_validation_metrics(validation, registration.aligned_image)
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

    lf_rows.sort(key=lambda item: _score_sort_value(item.metrics.get(PROFILE_NAME)), reverse=True)
    for idx, row in enumerate(lf_rows, start=1):
        row.validation_rank = idx

    thor_registration = estimate_similarity_registration(
        validation,
        thorcam,
        scale_values=cast(Sequence[float], cfg.scale_values),
    )
    thor_metrics = compute_validation_metrics(validation, thor_registration.aligned_image)
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
        preview_path=str(cfg.thorcam_image),
        warnings=_unique_strings(thor_registration.warnings + thor_metrics.warnings),
    )

    rows = lf_rows + [thor_row]
    top_rows = lf_rows[: max(0, cfg.top_k)]
    for row in top_rows:
        _write_comparison_figures(figures_dir, validation, row)
        if cfg.enable_nonrigid_diagnostic:
            _write_affine_diagnostic_figure(figures_dir, validation, row)
    _write_comparison_figures(figures_dir, validation, thor_row)
    if cfg.enable_nonrigid_diagnostic:
        _write_affine_diagnostic_figure(figures_dir, validation, thor_row)

    best = lf_rows[0] if lf_rows else None
    summary = _build_summary(
        cfg,
        stage,
        anchor_record,
        anchor_registration,
        best,
        thor_row,
        rows,
        model_warnings=_unique_strings(getattr(model, "warnings", [])),
    )
    _write_results_csv(cfg.output_dir / "trio_benchmark_results.csv", rows)
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
    scale_values = list(config.scale_values) if config.scale_values is not None else parse_scale_search(DEFAULT_SCALE_SEARCH)
    return BenchmarkConfig(
        sweep_dir=Path(config.sweep_dir),
        dataset_id=str(config.dataset_id),
        validation_image=Path(config.validation_image),
        thorcam_image=Path(config.thorcam_image),
        output_dir=Path(config.output_dir),
        stage=config.stage,
        generation_scope=config.generation_scope,
        anchor_row_id=config.anchor_row_id,
        scale_values=scale_values,
        top_k=int(config.top_k),
        residual_translation_max_px=float(config.residual_translation_max_px),
        enable_nonrigid_diagnostic=bool(config.enable_nonrigid_diagnostic),
    )


def _collect_candidate_images(model: Any, config: BenchmarkConfig, stage: str) -> List[CandidateImageRecord]:
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
    for row in rows:
        key = (
            str(getattr(row, "stage", "")),
            str(getattr(row, "candidate_id", "")),
            _optional_str(getattr(row, "generation_id", None)),
        )
        record = record_by_key.get(key)
        if record is None:
            continue
        try:
            image, workspace_path, preview_path, warnings = _load_candidate_image(record)
        except Exception:
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
    return result


def _load_candidate_image(record: Any) -> Tuple[np.ndarray, Optional[str], Optional[str], List[str]]:
    workspace_path = _optional_str(getattr(record, "workspace_path", None))
    preview_path = _optional_str(getattr(record, "preview_path", None))
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
        "registration_scale",
        "registration_rotation_deg",
        "registration_translation_x",
        "registration_translation_y",
        "residual_translation_x",
        "residual_translation_y",
        "residual_applied",
        "workspace_path",
        "preview_path",
        "warnings",
    ]
    write_csv_atomic(path, fieldnames, [_row_to_csv(row) for row in rows], create_backup=False)


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
        "registration_scale": _csv_float(reg.scale),
        "registration_rotation_deg": _csv_float(reg.rotation_deg),
        "registration_translation_x": _csv_float(reg.translation_x),
        "registration_translation_y": _csv_float(reg.translation_y),
        "residual_translation_x": _csv_float(reg.residual_translation_x),
        "residual_translation_y": _csv_float(reg.residual_translation_y),
        "residual_applied": str(bool(reg.residual_applied)).lower(),
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
) -> Dict[str, Any]:
    deltas: Dict[str, Optional[float]] = {}
    if best is not None:
        for name, value in best.metrics.items():
            thor_value = thor_row.metrics.get(name)
            if isinstance(value, (int, float)) and isinstance(thor_value, (int, float)):
                deltas[name] = float(value) - float(thor_value)
    warnings = _unique_strings(
        list(model_warnings)
        + [warning for row in rows for warning in row.warnings]
        + (["nonrigid diagnostic requested; primary score remains similarity-only"] if config.enable_nonrigid_diagnostic else [])
    )
    return {
        "config": {
            "profile_name": PROFILE_NAME,
            "sweep_dir": str(config.sweep_dir),
            "dataset_id": config.dataset_id,
            "validation_image": str(config.validation_image),
            "thorcam_image": str(config.thorcam_image),
            "output_dir": str(config.output_dir),
            "stage": stage,
            "generation_scope": config.generation_scope,
            "anchor_row_id": config.anchor_row_id,
            "scale_values": list(config.scale_values or []),
            "top_k": config.top_k,
            "residual_translation_max_px": config.residual_translation_max_px,
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
            "<th>Figure</th><th>Warnings</th></tr></thead><tbody>",
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
    parser.add_argument("--validation-image", required=True, type=Path)
    parser.add_argument("--thorcam-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--generation-scope", default="all")
    parser.add_argument("--anchor-row-id", default="auto")
    parser.add_argument("--scale-search", default=DEFAULT_SCALE_SEARCH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--residual-translation-max-px", type=float, default=DEFAULT_RESIDUAL_TRANSLATION_MAX_PX)
    parser.add_argument("--enable-nonrigid-diagnostic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=args.sweep_dir,
            dataset_id=args.dataset_id,
            validation_image=args.validation_image,
            thorcam_image=args.thorcam_image,
            output_dir=args.output_dir,
            stage=args.stage,
            generation_scope=args.generation_scope,
            anchor_row_id=args.anchor_row_id,
            scale_values=parse_scale_search(args.scale_search),
            top_k=args.top_k,
            residual_translation_max_px=args.residual_translation_max_px,
            enable_nonrigid_diagnostic=args.enable_nonrigid_diagnostic,
        )
    )
    print(f"Wrote trio benchmark to: {args.output_dir}")
    if result.best_lf_row_id is not None:
        print(f"Best LF row: {result.best_lf_row_id}")


if __name__ == "__main__":
    main()

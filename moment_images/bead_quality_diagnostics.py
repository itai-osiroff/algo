"""Bead compactness diagnostics for validation-guided ranking review."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw


BEAD_QUALITY_CSV = "bead_quality_metrics.csv"
DISAGREEMENT_BUCKETS = {
    "proxy_sharper_and_compacter",
    "proxy_sharper_but_noisier",
    "oracle_better_signal_but_soft",
    "roi_or_coverage_mismatch",
    "insufficient_bead_evidence",
}


@dataclass(frozen=True)
class BeadQualityMetrics:
    """Compactness diagnostics measured at validation-derived bead centers."""

    validation_bead_count: int
    evaluated_bead_count: int
    candidate_width_mean_px: Optional[float]
    validation_width_mean_px: Optional[float]
    width_excess_mean_px: Optional[float]
    compactness_score: Optional[float]
    support_area_ratio_mean: Optional[float]
    support_compactness_score: Optional[float]
    peak_contrast_mean: Optional[float]
    ringing_penalty_mean: Optional[float]
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BeadQualityRecord:
    """One trio/sweep identity row plus bead-quality metrics."""

    source_kind: str
    stage: str
    dataset_id: str
    candidate_id: str
    row_id: str
    generation_id: Optional[str]
    validation_rank: Optional[int]
    validation_rankable: Optional[bool]
    metrics: BeadQualityMetrics
    bead_crop_path: Optional[str] = None


BEAD_QUALITY_FIELDNAMES = [
    "source_kind",
    "stage",
    "dataset_id",
    "candidate_id",
    "row_id",
    "generation_id",
    "validation_rank",
    "validation_rankable",
    "validation_bead_count",
    "evaluated_bead_count",
    "candidate_width_mean_px",
    "validation_width_mean_px",
    "width_excess_mean_px",
    "compactness_score",
    "support_area_ratio_mean",
    "support_compactness_score",
    "peak_contrast_mean",
    "ringing_penalty_mean",
    "bead_crop_path",
    "warnings",
]


def _finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_int(value: object) -> Optional[int]:
    numeric = _finite_float(value)
    return None if numeric is None else int(numeric)


def _optional_bool(value: object) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _robust_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray(np.clip((arr - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def detect_validation_bead_centers(image: np.ndarray) -> List[Tuple[float, float]]:
    """Detect bead centers from the validation image only."""

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


def _patch_and_radius_grid(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    cx, cy = center
    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(arr.shape[1], int(math.floor(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(arr.shape[0], int(math.floor(cy)) + radius + 1)
    patch = arr[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    rr = np.sqrt(np.square(xx + x0 - cx) + np.square(yy + y0 - cy))
    return patch, rr


def _local_width(image: np.ndarray, center: Tuple[float, float], radius: int = 8) -> Optional[float]:
    patch, rr = _patch_and_radius_grid(image, center, radius)
    if patch.size == 0:
        return None
    patch = patch - float(np.min(patch))
    total = float(np.sum(patch))
    if total <= 1e-8:
        return None
    return float(math.sqrt(float(np.sum(np.square(rr) * patch) / total)))


def _support_area(image: np.ndarray, center: Tuple[float, float], radius: int = 8) -> Optional[float]:
    patch, _ = _patch_and_radius_grid(image, center, radius)
    if patch.size == 0:
        return None
    patch = patch - float(np.min(patch))
    peak = float(np.max(patch))
    if peak <= 1e-8:
        return None
    return float(np.count_nonzero(patch >= 0.30 * peak))


def _peak_contrast(image: np.ndarray, center: Tuple[float, float], radius: int = 10) -> Optional[float]:
    patch, rr = _patch_and_radius_grid(image, center, radius)
    if patch.size == 0:
        return None
    core = patch[rr <= 3.0]
    background = patch[(rr >= 6.0) & (rr <= float(radius))]
    if core.size == 0 or background.size == 0:
        return None
    return float(max(0.0, float(np.percentile(core, 95.0)) - float(np.percentile(background, 50.0))))


def _ringing_penalty(image: np.ndarray, center: Tuple[float, float], radius: int = 12) -> Optional[float]:
    patch, rr = _patch_and_radius_grid(image, center, radius)
    if patch.size == 0:
        return None
    core = patch[rr <= 3.0]
    ring = patch[(rr >= 5.0) & (rr <= 10.0)]
    if core.size == 0 or ring.size == 0:
        return None
    peak = float(np.percentile(core, 95.0))
    if peak <= 1e-8:
        return None
    ring_signal = float(np.percentile(ring, 95.0))
    return float(np.clip(ring_signal / peak, 0.0, 1.0))


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(finite))


def _score_from_excess(excess: Optional[float], scale: float) -> Optional[float]:
    if excess is None:
        return None
    return float(math.exp(-max(0.0, float(excess)) / scale))


def compute_bead_quality_metrics(
    validation_image: np.ndarray,
    candidate_image: np.ndarray,
    *,
    centers: Optional[Sequence[Tuple[float, float]]] = None,
) -> BeadQualityMetrics:
    """Measure candidate bead compactness at validation-derived centers."""

    validation = _robust_unit(validation_image)
    candidate = _robust_unit(candidate_image)
    selected_centers = list(centers) if centers is not None else detect_validation_bead_centers(validation)
    warnings: List[str] = []
    if not selected_centers:
        return BeadQualityMetrics(
            validation_bead_count=0,
            evaluated_bead_count=0,
            candidate_width_mean_px=None,
            validation_width_mean_px=None,
            width_excess_mean_px=None,
            compactness_score=None,
            support_area_ratio_mean=None,
            support_compactness_score=None,
            peak_contrast_mean=None,
            ringing_penalty_mean=None,
            warnings=["insufficient_bead_evidence"],
        )

    validation_widths: List[Optional[float]] = []
    candidate_widths: List[Optional[float]] = []
    support_ratios: List[Optional[float]] = []
    peak_contrasts: List[Optional[float]] = []
    ringing_penalties: List[Optional[float]] = []
    evaluated = 0
    for center in selected_centers:
        val_width = _local_width(validation, center)
        cand_width = _local_width(candidate, center)
        val_area = _support_area(validation, center)
        cand_area = _support_area(candidate, center)
        if val_width is None or cand_width is None:
            continue
        evaluated += 1
        validation_widths.append(val_width)
        candidate_widths.append(cand_width)
        if val_area is not None and cand_area is not None and val_area > 0.0:
            support_ratios.append(float(cand_area) / float(val_area))
        peak_contrasts.append(_peak_contrast(candidate, center))
        ringing_penalties.append(_ringing_penalty(candidate, center))

    if evaluated == 0:
        warnings.append("insufficient_bead_evidence")
    validation_width_mean = _mean_optional(validation_widths)
    candidate_width_mean = _mean_optional(candidate_widths)
    width_excess = (
        None
        if validation_width_mean is None or candidate_width_mean is None
        else float(candidate_width_mean - validation_width_mean)
    )
    support_ratio = _mean_optional(support_ratios)
    support_excess = None if support_ratio is None else support_ratio - 1.0
    return BeadQualityMetrics(
        validation_bead_count=len(selected_centers),
        evaluated_bead_count=evaluated,
        candidate_width_mean_px=candidate_width_mean,
        validation_width_mean_px=validation_width_mean,
        width_excess_mean_px=width_excess,
        compactness_score=_score_from_excess(width_excess, scale=3.0),
        support_area_ratio_mean=support_ratio,
        support_compactness_score=_score_from_excess(support_excess, scale=1.5),
        peak_contrast_mean=_mean_optional(peak_contrasts),
        ringing_penalty_mean=_mean_optional(ringing_penalties),
        warnings=warnings,
    )


def classify_bead_disagreement(
    *,
    oracle_metrics: BeadQualityMetrics,
    proxy_metrics: BeadQualityMetrics,
    oracle_valid_fov_fraction: Optional[float] = None,
    proxy_valid_fov_fraction: Optional[float] = None,
    oracle_candidate_roi_coverage: Optional[float] = None,
    proxy_candidate_roi_coverage: Optional[float] = None,
) -> str:
    """Classify why a proxy-ranked candidate beat an oracle-ranked candidate."""

    if (
        oracle_metrics.evaluated_bead_count <= 0
        or proxy_metrics.evaluated_bead_count <= 0
        or oracle_metrics.validation_bead_count <= 0
        or proxy_metrics.validation_bead_count <= 0
    ):
        return "insufficient_bead_evidence"

    coverage_values = [
        oracle_valid_fov_fraction,
        proxy_valid_fov_fraction,
        oracle_candidate_roi_coverage,
        proxy_candidate_roi_coverage,
    ]
    if any(value is not None and value < 0.50 for value in coverage_values):
        return "roi_or_coverage_mismatch"

    proxy_noisier = (
        proxy_metrics.ringing_penalty_mean is not None
        and (
            proxy_metrics.ringing_penalty_mean >= 0.06
            or (
                oracle_metrics.ringing_penalty_mean is not None
                and proxy_metrics.ringing_penalty_mean > oracle_metrics.ringing_penalty_mean + 0.08
            )
        )
    )
    proxy_width_better = (
        proxy_metrics.width_excess_mean_px is not None
        and oracle_metrics.width_excess_mean_px is not None
        and proxy_metrics.width_excess_mean_px < oracle_metrics.width_excess_mean_px - 0.25
    )
    proxy_support_better = (
        proxy_metrics.support_area_ratio_mean is not None
        and oracle_metrics.support_area_ratio_mean is not None
        and proxy_metrics.support_area_ratio_mean < oracle_metrics.support_area_ratio_mean - 0.10
    )
    if proxy_noisier:
        return "proxy_sharper_but_noisier"
    if proxy_width_better and proxy_support_better:
        return "proxy_sharper_and_compacter"
    return "oracle_better_signal_but_soft"


def bead_quality_record_to_row(record: BeadQualityRecord) -> Dict[str, object]:
    row: Dict[str, object] = {
        "source_kind": record.source_kind,
        "stage": record.stage,
        "dataset_id": record.dataset_id,
        "candidate_id": record.candidate_id,
        "row_id": record.row_id,
        "generation_id": record.generation_id or "",
        "validation_rank": "" if record.validation_rank is None else record.validation_rank,
        "validation_rankable": "" if record.validation_rankable is None else str(bool(record.validation_rankable)).lower(),
        "bead_crop_path": record.bead_crop_path or "",
        "warnings": "; ".join(record.metrics.warnings),
    }
    row.update(
        {
            name: value
            for name, value in asdict(record.metrics).items()
            if name != "warnings"
        }
    )
    return row


def bead_quality_record_from_row(row: Mapping[str, object]) -> BeadQualityRecord:
    metrics = BeadQualityMetrics(
        validation_bead_count=int(_optional_int(row.get("validation_bead_count")) or 0),
        evaluated_bead_count=int(_optional_int(row.get("evaluated_bead_count")) or 0),
        candidate_width_mean_px=_finite_float(row.get("candidate_width_mean_px")),
        validation_width_mean_px=_finite_float(row.get("validation_width_mean_px")),
        width_excess_mean_px=_finite_float(row.get("width_excess_mean_px")),
        compactness_score=_finite_float(row.get("compactness_score")),
        support_area_ratio_mean=_finite_float(row.get("support_area_ratio_mean")),
        support_compactness_score=_finite_float(row.get("support_compactness_score")),
        peak_contrast_mean=_finite_float(row.get("peak_contrast_mean")),
        ringing_penalty_mean=_finite_float(row.get("ringing_penalty_mean")),
        warnings=[item.strip() for item in str(row.get("warnings", "")).split(";") if item.strip()],
    )
    return BeadQualityRecord(
        source_kind=str(row.get("source_kind", "")).strip(),
        stage=str(row.get("stage", "")).strip(),
        dataset_id=str(row.get("dataset_id", "")).strip(),
        candidate_id=str(row.get("candidate_id", "")).strip(),
        row_id=str(row.get("row_id", "")).strip(),
        generation_id=str(row.get("generation_id", "")).strip() or None,
        validation_rank=_optional_int(row.get("validation_rank")),
        validation_rankable=_optional_bool(row.get("validation_rankable")),
        metrics=metrics,
        bead_crop_path=str(row.get("bead_crop_path", "")).strip() or None,
    )


def write_bead_quality_csv(path: Path, records: Sequence[BeadQualityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEAD_QUALITY_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(bead_quality_record_to_row(record))


def read_bead_quality_csv(path: Path) -> List[BeadQualityRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [bead_quality_record_from_row(row) for row in csv.DictReader(handle)]


def find_bead_quality_csv(trio_results_path: Path) -> Optional[Path]:
    path = Path(trio_results_path)
    candidates = [path / BEAD_QUALITY_CSV] if path.is_dir() else [path.with_name(BEAD_QUALITY_CSV)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _unit_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.round(_robust_unit(image) * 255.0), dtype=np.uint8)


def _crop_uint8(image: np.ndarray, center: Tuple[float, float], radius: int) -> Image.Image:
    arr = _unit_to_uint8(image)
    cx, cy = center
    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(arr.shape[1], int(math.floor(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(arr.shape[0], int(math.floor(cy)) + radius + 1)
    crop = arr[y0:y1, x0:x1]
    canvas = Image.new("L", (2 * radius + 1, 2 * radius + 1), color=0)
    patch = Image.fromarray(crop)
    canvas.paste(patch, (0, 0))
    return canvas


def save_bead_crop_sheet(
    path: Path,
    validation_image: np.ndarray,
    candidate_image: np.ndarray,
    centers: Sequence[Tuple[float, float]],
    *,
    radius: int = 10,
    max_beads: int = 12,
) -> Optional[Path]:
    """Save a compact validation/candidate bead crop sheet for review."""

    selected = list(centers)[:max(0, int(max_beads))]
    if not selected:
        return None
    tile = 2 * radius + 1
    gap = 4
    label_height = 12
    width = len(selected) * (2 * tile + gap) + max(0, len(selected) - 1) * gap
    height = tile + label_height
    sheet = Image.new("RGB", (max(width, 1), max(height, 1)), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, center in enumerate(selected):
        x = idx * (2 * tile + 2 * gap)
        val_crop = _crop_uint8(validation_image, center, radius).convert("RGB")
        cand_crop = _crop_uint8(candidate_image, center, radius).convert("RGB")
        sheet.paste(val_crop, (x, 0))
        sheet.paste(cand_crop, (x + tile + gap, 0))
        draw.text((x, tile), f"{idx + 1}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path

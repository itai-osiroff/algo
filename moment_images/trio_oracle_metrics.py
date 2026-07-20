"""Import and cache external trio benchmark metrics for metric explorer use."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

from moment_images.reliable_io import read_json_first_valid, write_json_atomic


DEFAULT_TRIO_SEED_TOP_K = 8

TRIO_ORACLE_CACHE_FILENAME = "trio_oracle_metrics.json"
TRIO_ORACLE_PROFILE_NAME = "trio_oracle_thorcam_fov"
TRIO_PRIMARY_SCORE = "validation_hybrid_thorcam_fov_v1"
TRIO_ROW_LOCAL_SCORE = "validation_hybrid_v1"
TRIO_FULL_GRID_SCORE = "validation_hybrid_full_grid_v1"
TRIO_VALID_FOV_FRACTION = "validation_valid_fov_fraction"
TRIO_ROI_FRACTION = "validation_metric_roi_fraction"
TRIO_ROI_PIXEL_COUNT = "validation_metric_roi_pixel_count"
TRIO_BEAD_COUNT = "validation_metric_bead_count"
TRIO_CANDIDATE_ROI_COVERAGE = "validation_metric_candidate_roi_coverage"
TRIO_DELTA_VS_THORCAM = "validation_hybrid_thorcam_fov_delta_vs_thorcam"
TRIO_LF_SOURCE_KIND = "lf_candidate"
TRIO_THORCAM_SOURCE_KIND = "thorcam_baseline"

TRIO_METADATA_EXPORT_FIELDS = [
    "trio_validation_rank",
    "trio_validation_rankable",
    "trio_ranking_exclusion_reason",
    "trio_report_path",
    "trio_overlay_path",
    "trio_diff_path",
    "trio_source_path",
]


@dataclass
class TrioOracleRecord:
    """One imported trio benchmark row joined to an LF sweep candidate."""

    source_kind: str
    stage: str
    candidate_id: str
    dataset_id: str
    generation_id: Optional[str]
    row_id: str
    validation_rank: Optional[int]
    metrics: Dict[str, float]
    diagnostics: Dict[str, object]
    validation_rankable: Optional[bool] = None
    ranking_exclusion_reason: Optional[str] = None
    report_path: Optional[str] = None
    overlay_path: Optional[str] = None
    diff_path: Optional[str] = None
    source_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class TrioOracleImport:
    """Normalized trio benchmark import payload."""

    records: List[TrioOracleRecord]
    baselines: Dict[str, Dict[str, float]]
    source_paths: List[str]
    warnings: List[str] = field(default_factory=list)
    imported_at: str = field(default_factory=lambda: _utc_now_iso())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def trio_cache_path(review_dir: Path) -> Path:
    return review_dir / TRIO_ORACLE_CACHE_FILENAME


def trio_numeric_metric_names() -> List[str]:
    return [
        TRIO_PRIMARY_SCORE,
        TRIO_ROW_LOCAL_SCORE,
        TRIO_FULL_GRID_SCORE,
        TRIO_VALID_FOV_FRACTION,
        TRIO_ROI_FRACTION,
        TRIO_ROI_PIXEL_COUNT,
        TRIO_BEAD_COUNT,
        TRIO_CANDIDATE_ROI_COVERAGE,
        "validation_signal",
        "validation_structure",
        "validation_bead",
        "validation_sharpness",
        "validation_signal_structure_score",
        "registration_confidence",
        "matched_bead_count",
        "median_bead_error_px",
        "common_fov_fraction",
        "registration_fit_rms_px",
        "registration_refined_score",
        "registration_runtime_seconds",
    ]


def trio_diagnostic_field_names() -> List[str]:
    return [
        "validation_rankable",
        "ranking_exclusion_reason",
        "registration_status",
        "registration_profile",
        "registration_suspect",
        "registration_suspect_reasons",
        "anchor_selection_reason",
        "registration_constraint_level",
        "registration_underconstrained",
        "registration_initializer",
        "registration_calibration_group",
        "registration_transform_source",
        "registration_point_count",
        "hypotheses_generated",
        "hypotheses_prefiltered",
        "hypotheses_dense_scored",
        "hypotheses_refined",
        "optimizer_evaluations",
    ]


def trio_bool_text(value: object) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def trio_optional_int(value: object) -> Optional[int]:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def trio_record_key(record: TrioOracleRecord) -> Tuple[str, str, str, Optional[str]]:
    return record.stage, record.candidate_id, record.dataset_id, record.generation_id


def split_trio_row_id(row_id: str) -> Tuple[str, str]:
    text = str(row_id or "").strip()
    if ":" in text:
        stage, candidate = text.split(":", 1)
        return stage, candidate
    return "", text


def find_trio_record_for_identity(
    trio_lookup: Mapping[Tuple[str, str, str, Optional[str]], TrioOracleRecord],
    *,
    stage: str,
    candidate_id: str,
    dataset_id: str,
    generation_id: Optional[str],
) -> Optional[TrioOracleRecord]:
    exact = trio_lookup.get((stage, candidate_id, dataset_id, generation_id))
    if exact is not None:
        return exact
    return trio_lookup.get((stage, candidate_id, dataset_id, None))


def trio_rankable(record: TrioOracleRecord) -> bool:
    if record.validation_rankable is not None:
        return bool(record.validation_rankable)
    rankable = record.diagnostics.get("validation_rankable")
    if isinstance(rankable, bool):
        return rankable
    if trio_bool_text(rankable) is False:
        return False
    return all(not warning.startswith("trio_rankable=false") for warning in record.warnings)


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "item"


def _trio_report_path(csv_path: Path) -> Optional[str]:
    report = csv_path.parent / "trio_benchmark_report.html"
    return str(report) if report.exists() else None


def _trio_figure_paths(csv_path: Path, row: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    figures_dir = csv_path.parent / "figures"
    if not figures_dir.exists():
        return None, None
    source_kind = str(row.get("source_kind", "")).strip()
    if source_kind == TRIO_THORCAM_SOURCE_KIND:
        prefix = "baseline_thorcam"
    else:
        rank = trio_optional_int(row.get("validation_rank")) or 0
        prefix = (
            f"rank_{rank:03d}_"
            f"{_safe_slug(row.get('stage', ''))}_"
            f"{_safe_slug(row.get('candidate_id', ''))}"
        )
    overlay = figures_dir / f"{prefix}_overlay.png"
    diff = figures_dir / f"{prefix}_diff.png"
    return (str(overlay) if overlay.exists() else None, str(diff) if diff.exists() else None)


def _discover_trio_result_files(paths: Sequence[Path]) -> Tuple[List[Path], List[Path], List[str]]:
    detailed: List[Path] = []
    aggregate: List[Path] = []
    warnings: List[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            found = sorted(path.rglob("trio_benchmark_results.csv"))
            detailed.extend(found)
            batch_path = path / "trio_batch_results.csv"
            if not found and batch_path.exists():
                aggregate.append(batch_path)
            if not found and not batch_path.exists():
                warnings.append(f"No trio result CSVs found in {path}.")
            continue
        if not path.exists():
            warnings.append(f"Trio result path not found: {path}")
            continue
        if path.name == "trio_benchmark_results.csv":
            detailed.append(path)
            continue
        if path.name == "trio_batch_results.csv":
            found = sorted(path.parent.rglob("trio_benchmark_results.csv"))
            if found:
                detailed.extend(found)
            else:
                aggregate.append(path)
            continue
        warnings.append(f"Unsupported trio result file ignored: {path}")
    unique_detailed = list(dict.fromkeys(path.resolve() for path in detailed))
    unique_aggregate = list(dict.fromkeys(path.resolve() for path in aggregate))
    return unique_detailed, unique_aggregate, _stable_unique_strings(warnings)


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rankability_from_row(row: Mapping[str, str], default: Optional[bool]) -> Tuple[Optional[bool], Optional[str], List[str]]:
    rankable = trio_bool_text(row.get("validation_rankable"))
    if rankable is None:
        rankable = trio_bool_text(row.get("best_validation_rankable"))
    if rankable is None:
        rankable = default
    exclusion_reason = _optional_str(row.get("ranking_exclusion_reason")) or _optional_str(
        row.get("best_ranking_exclusion_reason")
    )
    warnings: List[str] = []
    if rankable is False:
        warnings.append(
            f"trio_rankable=false"
            + (f":{exclusion_reason}" if exclusion_reason else "")
        )
    return rankable, exclusion_reason, warnings


def _diagnostics_from_row(row: Mapping[str, str]) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {
        name: row.get(name, "")
        for name in trio_diagnostic_field_names()
        if str(row.get(name, "")).strip()
    }
    rankable = trio_bool_text(row.get("validation_rankable"))
    if rankable is None:
        rankable = trio_bool_text(row.get("best_validation_rankable"))
    if rankable is not None:
        diagnostics["validation_rankable"] = rankable
    return diagnostics


def _parse_trio_benchmark_csv(path: Path) -> Tuple[List[TrioOracleRecord], Dict[str, Dict[str, float]], List[str]]:
    records: List[TrioOracleRecord] = []
    baselines: Dict[str, Dict[str, float]] = {}
    warnings: List[str] = []
    rows = _read_csv_dicts(path)
    for row in rows:
        source_kind = str(row.get("source_kind", "")).strip()
        dataset_id = str(row.get("dataset_id", "")).strip()
        row_id = str(row.get("row_id", "")).strip()
        stage = str(row.get("stage", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not stage or not candidate_id:
            inferred_stage, inferred_candidate = split_trio_row_id(row_id)
            stage = stage or inferred_stage
            candidate_id = candidate_id or inferred_candidate
        generation_id = _optional_str(row.get("generation_id"))
        metrics = {
            name: float(value)
            for name in trio_numeric_metric_names()
            if (value := _coerce_float(row.get(name))) is not None
        }
        rankable, exclusion_reason, rankability_warnings = _rankability_from_row(row, default=None)
        diagnostics = _diagnostics_from_row(row)
        if exclusion_reason:
            diagnostics["ranking_exclusion_reason"] = exclusion_reason
        row_warnings = [
            item.strip()
            for item in str(row.get("warnings", "")).split(";")
            if item.strip()
        ]
        row_warnings.extend(rankability_warnings)
        if source_kind == TRIO_THORCAM_SOURCE_KIND and dataset_id:
            baselines[dataset_id] = dict(metrics)
            continue
        if source_kind != TRIO_LF_SOURCE_KIND:
            continue
        if not stage or not candidate_id or not dataset_id:
            warnings.append(f"Skipped trio row with incomplete identity in {path.name}: {row_id or candidate_id}")
            continue
        overlay_path, diff_path = _trio_figure_paths(path, row)
        records.append(
            TrioOracleRecord(
                source_kind=source_kind,
                stage=stage,
                candidate_id=candidate_id,
                dataset_id=dataset_id,
                generation_id=generation_id,
                row_id=row_id or f"{stage}:{candidate_id}",
                validation_rank=trio_optional_int(row.get("validation_rank")),
                metrics=metrics,
                diagnostics=diagnostics,
                validation_rankable=rankable,
                ranking_exclusion_reason=exclusion_reason,
                report_path=_trio_report_path(path),
                overlay_path=overlay_path,
                diff_path=diff_path,
                source_path=str(path),
                warnings=_stable_unique_strings(row_warnings),
            )
        )
    return records, baselines, _stable_unique_strings(warnings)


def _parse_trio_batch_csv(path: Path) -> Tuple[List[TrioOracleRecord], Dict[str, Dict[str, float]], List[str]]:
    records: List[TrioOracleRecord] = []
    baselines: Dict[str, Dict[str, float]] = {}
    warnings: List[str] = []
    for row in _read_csv_dicts(path):
        if str(row.get("status", "")).strip().lower() not in {"ok", "success", "completed"}:
            warnings.append(
                f"Skipped failed trio batch scene {row.get('scene_id', '')}: {row.get('error', '')}"
            )
            continue
        dataset_id = str(row.get("dataset_id", "")).strip()
        row_id = str(row.get("best_lf_row_id", "")).strip()
        stage_from_row_id, candidate_from_row_id = split_trio_row_id(row_id)
        stage = stage_from_row_id or str(row.get("stage", "")).strip() or str(row.get("best_lf_stage", "")).strip()
        candidate_id = (
            str(row.get("best_lf_candidate_id", "")).strip()
            or candidate_from_row_id
            or row_id
        )
        if not stage or not candidate_id or not dataset_id:
            warnings.append(f"Skipped batch scene without joinable best row: {row.get('scene_id', '')}")
            continue
        metrics = {
            TRIO_PRIMARY_SCORE: row.get("best_validation_hybrid_thorcam_fov_v1"),
            TRIO_ROW_LOCAL_SCORE: row.get("best_validation_hybrid_v1"),
            TRIO_FULL_GRID_SCORE: row.get("best_validation_hybrid_full_grid_v1"),
            TRIO_VALID_FOV_FRACTION: row.get("best_validation_valid_fov_fraction"),
            TRIO_ROI_FRACTION: row.get("best_validation_metric_roi_fraction"),
            TRIO_ROI_PIXEL_COUNT: row.get("best_validation_metric_roi_pixel_count"),
            TRIO_BEAD_COUNT: row.get("best_validation_metric_bead_count"),
            TRIO_CANDIDATE_ROI_COVERAGE: row.get("best_validation_metric_candidate_roi_coverage"),
            "registration_confidence": row.get("registration_confidence"),
            "registration_fit_rms_px": row.get("registration_fit_rms_px"),
            "registration_runtime_seconds": row.get("registration_runtime_seconds"),
        }
        numeric_metrics = {
            name: float(value)
            for name, raw_value in metrics.items()
            if (value := _coerce_float(raw_value)) is not None
        }
        thorcam_metrics = {
            TRIO_PRIMARY_SCORE: row.get("thorcam_validation_hybrid_thorcam_fov_v1"),
            TRIO_ROW_LOCAL_SCORE: row.get("thorcam_validation_hybrid_v1"),
            TRIO_FULL_GRID_SCORE: row.get("thorcam_validation_hybrid_full_grid_v1"),
            TRIO_VALID_FOV_FRACTION: row.get("thorcam_validation_valid_fov_fraction"),
            TRIO_ROI_FRACTION: row.get("thorcam_validation_metric_roi_fraction"),
            TRIO_ROI_PIXEL_COUNT: row.get("thorcam_validation_metric_roi_pixel_count"),
            TRIO_BEAD_COUNT: row.get("thorcam_validation_metric_bead_count"),
            TRIO_CANDIDATE_ROI_COVERAGE: row.get("thorcam_validation_metric_candidate_roi_coverage"),
        }
        baselines[dataset_id] = {
            name: float(value)
            for name, raw_value in thorcam_metrics.items()
            if (value := _coerce_float(raw_value)) is not None
        }
        rankable, exclusion_reason, rankability_warnings = _rankability_from_row(row, default=True)
        diagnostics = _diagnostics_from_row(row)
        if rankable is not None:
            diagnostics["validation_rankable"] = rankable
        if exclusion_reason:
            diagnostics["ranking_exclusion_reason"] = exclusion_reason
        row_warnings = [
            item.strip()
            for item in str(row.get("warnings", "")).split(";")
            if item.strip()
        ]
        row_warnings.extend(rankability_warnings)
        records.append(
            TrioOracleRecord(
                source_kind=TRIO_LF_SOURCE_KIND,
                stage=stage,
                candidate_id=candidate_id,
                dataset_id=dataset_id,
                generation_id=_optional_str(row.get("generation_id")),
                row_id=row_id or f"{stage}:{candidate_id}",
                validation_rank=trio_optional_int(row.get("best_validation_rank")) or 1,
                metrics=numeric_metrics,
                diagnostics=diagnostics,
                validation_rankable=rankable,
                ranking_exclusion_reason=exclusion_reason,
                report_path=_optional_str(row.get("report_path")),
                source_path=str(path),
                warnings=_stable_unique_strings(row_warnings),
            )
        )
    return records, baselines, _stable_unique_strings(warnings)


def _trio_import_to_payload(import_data: TrioOracleImport) -> Dict[str, object]:
    return {
        "schema_version": 2,
        "imported_at": import_data.imported_at,
        "source_paths": list(import_data.source_paths),
        "warnings": list(import_data.warnings),
        "baselines": import_data.baselines,
        "records": [
            {
                "source_kind": record.source_kind,
                "stage": record.stage,
                "candidate_id": record.candidate_id,
                "dataset_id": record.dataset_id,
                "generation_id": record.generation_id,
                "row_id": record.row_id,
                "validation_rank": record.validation_rank,
                "metrics": dict(record.metrics),
                "diagnostics": dict(record.diagnostics),
                "validation_rankable": record.validation_rankable,
                "ranking_exclusion_reason": record.ranking_exclusion_reason,
                "report_path": record.report_path,
                "overlay_path": record.overlay_path,
                "diff_path": record.diff_path,
                "source_path": record.source_path,
                "warnings": list(record.warnings),
            }
            for record in import_data.records
        ],
    }


def _rankability_from_cached_item(item: Mapping[str, object], diagnostics: Mapping[str, object], warnings: List[str]) -> Optional[bool]:
    if "validation_rankable" in item:
        item_rankable = item.get("validation_rankable")
        if isinstance(item_rankable, bool):
            return item_rankable
        parsed = trio_bool_text(item_rankable)
        if parsed is not None:
            return parsed
    if "validation_rankable" in diagnostics:
        diag_rankable = diagnostics.get("validation_rankable")
        if isinstance(diag_rankable, bool):
            return diag_rankable
        parsed = trio_bool_text(diag_rankable)
        if parsed is not None:
            return parsed
    if any(str(warning).startswith("trio_rankable=false") for warning in warnings):
        return False
    return None


def _trio_import_from_payload(payload: Mapping[str, object]) -> TrioOracleImport:
    records: List[TrioOracleRecord] = []
    for item in cast(List[object], payload.get("records", [])):
        if not isinstance(item, Mapping):
            continue
        metrics_raw = item.get("metrics", {})
        diagnostics_raw = item.get("diagnostics", {})
        diagnostics = (
            dict(cast(Mapping[str, object], diagnostics_raw))
            if isinstance(diagnostics_raw, Mapping)
            else {}
        )
        warnings = (
            [str(value) for value in cast(List[object], item.get("warnings", []))]
            if isinstance(item.get("warnings"), list)
            else []
        )
        records.append(
            TrioOracleRecord(
                source_kind=str(item.get("source_kind", TRIO_LF_SOURCE_KIND)),
                stage=str(item.get("stage", "")),
                candidate_id=str(item.get("candidate_id", "")),
                dataset_id=str(item.get("dataset_id", "")),
                generation_id=_optional_str(item.get("generation_id")),
                row_id=str(item.get("row_id", "")),
                validation_rank=trio_optional_int(item.get("validation_rank")),
                metrics={
                    str(name): float(value)
                    for name, value in cast(Mapping[str, object], metrics_raw).items()
                    if _coerce_float(value) is not None
                }
                if isinstance(metrics_raw, Mapping)
                else {},
                diagnostics=diagnostics,
                validation_rankable=_rankability_from_cached_item(item, diagnostics, warnings),
                ranking_exclusion_reason=_optional_str(item.get("ranking_exclusion_reason"))
                or _optional_str(diagnostics.get("ranking_exclusion_reason")),
                report_path=_optional_str(item.get("report_path")),
                overlay_path=_optional_str(item.get("overlay_path")),
                diff_path=_optional_str(item.get("diff_path")),
                source_path=_optional_str(item.get("source_path")),
                warnings=warnings,
            )
        )
    baselines_raw = payload.get("baselines", {})
    baselines: Dict[str, Dict[str, float]] = {}
    if isinstance(baselines_raw, Mapping):
        for dataset_id, metrics in baselines_raw.items():
            if not isinstance(metrics, Mapping):
                continue
            baselines[str(dataset_id)] = {
                str(name): float(value)
                for name, value in metrics.items()
                if _coerce_float(value) is not None
            }
    return TrioOracleImport(
        records=records,
        baselines=baselines,
        source_paths=[
            str(value)
            for value in cast(List[object], payload.get("source_paths", []))
        ]
        if isinstance(payload.get("source_paths"), list)
        else [],
        warnings=[
            str(value)
            for value in cast(List[object], payload.get("warnings", []))
        ]
        if isinstance(payload.get("warnings"), list)
        else [],
        imported_at=str(payload.get("imported_at") or _utc_now_iso()),
    )


def load_trio_oracle_import(review_dir: Path) -> Tuple[Optional[TrioOracleImport], List[str]]:
    path = trio_cache_path(review_dir)
    if not path.exists():
        return None, []
    try:
        payload, _used_path, read_warnings = read_json_first_valid(
            path,
            validator=lambda obj: isinstance(obj, dict),
        )
        return _trio_import_from_payload(cast(Mapping[str, object], payload)), read_warnings
    except Exception as exc:  # noqa: BLE001
        return None, [f"Failed to load trio oracle cache {path}: {type(exc).__name__}: {exc}"]


def write_trio_oracle_import(review_dir: Path, import_data: TrioOracleImport) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        trio_cache_path(review_dir),
        _trio_import_to_payload(import_data),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def import_trio_oracle_results(review_dir: Path, paths: Sequence[Path]) -> TrioOracleImport:
    detailed_paths, aggregate_paths, warnings = _discover_trio_result_files(paths)
    records: List[TrioOracleRecord] = []
    baselines: Dict[str, Dict[str, float]] = {}
    source_paths: List[str] = []
    for path in detailed_paths:
        source_paths.append(str(path))
        parsed_records, parsed_baselines, parsed_warnings = _parse_trio_benchmark_csv(path)
        records.extend(parsed_records)
        baselines.update(parsed_baselines)
        warnings.extend(parsed_warnings)
    for path in aggregate_paths:
        source_paths.append(str(path))
        parsed_records, parsed_baselines, parsed_warnings = _parse_trio_batch_csv(path)
        records.extend(parsed_records)
        baselines.update(parsed_baselines)
        warnings.extend(parsed_warnings)

    deduped: Dict[Tuple[str, str, str, Optional[str]], TrioOracleRecord] = {}
    for record in records:
        deduped[trio_record_key(record)] = record
    import_data = TrioOracleImport(
        records=list(deduped.values()),
        baselines=baselines,
        source_paths=_stable_unique_strings(source_paths),
        warnings=_stable_unique_strings(warnings),
    )
    write_trio_oracle_import(review_dir, import_data)
    return import_data

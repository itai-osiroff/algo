"""Interactive visualization and export helpers for parameter sweep outputs."""

from __future__ import annotations

import argparse
import html
import json
import math
import platform
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from moment_images.im_process import save_workspace_variables
from moment_images.rectification_strategy import (
    normalize_optional_displacement_metric_family,
)
from moment_images.reliable_io import (
    read_json_first_valid,
    read_npz_first_valid,
    write_csv_atomic,
    write_json_atomic,
)


DEFAULT_PRIMARY_METRIC = "laplacian_variance"
DEFAULT_FALLBACK_METRICS = (
    "sum_squared_gradient",
    "dwt_level2_hv_energy",
    "total_variation",
    "maximum_intensity",
    "average_intensity",
)
STAGE_ORDER = ("legacy", "stage1", "stage2", "metric_phase")
SUCCESS_STATUSES = {"ok", "skipped_valid", "recomputed"}
DEFAULT_REPORT_TOP_K = 6
GENERATION_SCOPE_ALL = "All"
GENERATION_SCOPE_ACTIVE = "Active"
SENSITIVITY_MIN_GROUP_COUNT = 1
SENSITIVITY_OLS_MIN_EXTRA_ROWS = 2
EXTENSION_MIN_ABSOLUTE_EFFECT = 0.02
EXTENSION_MIN_RELATIVE_EFFECT = 0.15
EXTENSION_MAX_SUGGESTIONS = 8
RECTIFICATION_VIEW_CHANNELS = (
    "preview",
    "displacement_magnitude",
    "blended_score",
    "component_score",
    "global_final_delta",
)
RECTIFICATION_VIEW_LABELS = {
    "preview": "Preview",
    "displacement_magnitude": "Displacement Magnitude",
    "blended_score": "Blended Score",
    "component_score": "Component Score",
    "global_final_delta": "Global - Final",
}
RECTIFICATION_VIEW_CMAPS = {
    "preview": "gray",
    "displacement_magnitude": "viridis",
    "blended_score": "viridis",
    "component_score": "viridis",
    "global_final_delta": "coolwarm",
}


@dataclass
class CandidateVisualizationRow:
    """Normalized candidate row used by the GUI, plots, and export bundle."""

    row_id: str
    candidate_id: str
    stage: str
    display_name: str
    rank: int
    strategy: str
    score: Optional[float]
    mean_dataset_score: Optional[float]
    std_dataset_score: Optional[float]
    ranking_metric_name: Optional[str]
    ranking_metric_value: Optional[float]
    dataset_scores: Dict[str, float]
    dataset_runtimes: Dict[str, float]
    dataset_metrics: Dict[str, Dict[str, float]]
    metric_means: Dict[str, float]
    mean_runtime_seconds: Optional[float]
    missing_datasets: List[str]
    params: Dict[str, object]
    config: Dict[str, object]
    family: Optional[str]
    alpha: Optional[float]
    formulation_id: Optional[str]
    displacement_field_metrics: Dict[str, float] = field(default_factory=dict)
    inter_view_consistency_metrics: Dict[str, float] = field(default_factory=dict)
    preview_paths: Dict[str, str] = field(default_factory=dict)
    generation_id: Optional[str] = None


@dataclass
class StageVisualizationSummary:
    """Stage-level execution and ranking summary."""

    name: str
    num_runs_planned: int
    num_runs_recorded: int
    num_successful: int
    num_failed: int
    num_pending: int
    num_running: int
    skipped_valid_count: int
    recomputed_count: int
    rankings_count: int
    execution: Dict[str, object]
    failed_runs: List[Dict[str, object]]
    source_path: Optional[str]


@dataclass
class RunMetricRecord:
    """Per-run metric record used for dataset-specific analysis and diagnostics."""

    row_id: str
    run_name: str
    stage: str
    candidate_id: str
    dataset_id: str
    strategy: str
    family: Optional[str]
    alpha: Optional[float]
    formulation_id: Optional[str]
    params: Dict[str, object]
    config: Dict[str, object]
    score: Optional[float]
    mean_dataset_score: Optional[float]
    mean_runtime_seconds: Optional[float]
    metric_means: Dict[str, float]
    displacement_field_metrics: Dict[str, float]
    inter_view_consistency_metrics: Dict[str, float]
    displacement_score_diagnostics: Dict[str, object]
    preview_path: Optional[str]
    workspace_path: Optional[str]
    metric_record_path: Optional[str]
    ranking_metric_name: Optional[str]
    ranking_metric_value: Optional[float]
    generation_id: Optional[str] = None


RectificationResultKey = Tuple[str, str, str, Optional[str]]


@dataclass(frozen=True)
class RectificationViewSelection:
    """A stage/candidate/dataset selection for interactive rectification viewing."""

    stage: str
    candidate_id: str
    dataset_id: str
    generation_id: Optional[str] = None
    component_name: Optional[str] = None

    @property
    def key(self) -> RectificationResultKey:
        return self.stage, self.candidate_id, self.dataset_id, self.generation_id


@dataclass
class RectificationDisplayBundle:
    """Display-ready rectification result channels for one candidate/dataset."""

    selection: RectificationViewSelection
    preview_path: Optional[str]
    workspace_path: Optional[str]
    preview: Optional[np.ndarray]
    displacement_magnitude: Optional[np.ndarray]
    blended_score: Optional[np.ndarray]
    component_score: Optional[np.ndarray]
    global_final_delta: Optional[np.ndarray]
    warnings: List[str] = field(default_factory=list)

    def get(self, channel_name: str) -> Optional[np.ndarray]:
        if channel_name == "preview":
            return self.preview
        if channel_name == "displacement_magnitude":
            return self.displacement_magnitude
        if channel_name == "blended_score":
            return self.blended_score
        if channel_name == "component_score":
            return self.component_score
        if channel_name == "global_final_delta":
            return self.global_final_delta
        return None


@dataclass
class SweepVisualizationModel:
    """Canonical representation of legacy and modern sweep outputs."""

    source_dir: Path
    mode: str
    path_scope: str
    experiment_root: Path
    metric_phase_dir: Optional[Path]
    primary_stage_name: str
    primary_metric: str
    primary_metric_source: str
    datasets: List[str]
    strategies: List[str]
    metric_names: List[str]
    displacement_metric_names: List[str]
    consistency_metric_names: List[str]
    parameter_names: List[str]
    stage_order: List[str]
    stages: Dict[str, StageVisualizationSummary]
    stage_candidate_ids: Dict[str, List[str]]
    candidates: List[CandidateVisualizationRow]
    run_metric_records: List[RunMetricRecord]
    scoring_weights: Dict[str, float]
    warnings: List[str]
    provenance: Dict[str, object]
    experiment_metadata: Dict[str, object]
    generation_summary: Dict[str, object]
    previous_generation_candidates: List[CandidateVisualizationRow]
    all_generation_candidates: List[CandidateVisualizationRow]
    all_generation_run_metric_records: List[RunMetricRecord]
    default_generation_scope: str
    alpha_family_summary: List[Dict[str, object]]
    alpha_ranked_runs: List[Dict[str, object]]
    diagnostic_availability: Dict[str, object]
    candidate_ids: List[str]
    score_matrix: np.ndarray
    runtime_matrix: np.ndarray
    metric_tensor: np.ndarray
    parameter_numeric_matrix: np.ndarray
    parameter_text_matrix: np.ndarray


@dataclass
class ParameterValueEffect:
    """Grouped target values for one parameter value."""

    value: str
    numeric_value: Optional[float]
    count: int
    mean_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    candidate_ids: List[str]


@dataclass
class ParameterSensitivityRow:
    """Impact summary for one parameter under one stage/dataset/value scope."""

    rank: int
    parameter_name: str
    stage: str
    dataset_scope: str
    value_option: str
    impact_score: float
    best_value: Optional[str]
    worst_value: Optional[str]
    best_mean_value: Optional[float]
    worst_mean_value: Optional[float]
    num_values: int
    num_rows: int
    min_group_count: int
    max_group_count: int
    coverage_warning: Optional[str]
    ols_delta_r2: Optional[float]
    ols_status: str
    previous_impact_score: Optional[float] = None
    previous_best_value: Optional[str] = None
    generation_delta: Optional[float] = None
    value_effects: List[ParameterValueEffect] = field(default_factory=list)


@dataclass
class ExtensionSuggestion:
    """Conservative recommendation for the next sweep extension."""

    rank: int
    rule_id: str
    priority: int
    stage: str
    parameter_name: Optional[str]
    dataset_scope: str
    title: str
    recommendation: str
    evidence: str
    confidence: str
    related_values: List[str] = field(default_factory=list)
    generation_id: Optional[str] = None
    previous_generation_id: Optional[str] = None


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


def _candidate_path_exists(path: Path) -> bool:
    return path.exists() or Path(f"{path}.bak").exists()


def _read_json_optional(
    path: Path,
    *,
    validator: Optional[Any] = None,
    warnings: List[str],
) -> Tuple[Optional[Any], Optional[Path]]:
    if not _candidate_path_exists(path):
        return None, None
    try:
        payload, used_path, read_warnings = read_json_first_valid(path, validator=validator)
        warnings.extend(read_warnings)
        return payload, used_path
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to read {path.name}: {type(exc).__name__}: {exc}")
        return None, None


def _coerce_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _coerce_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sorted_unique(values: Iterable[str]) -> List[str]:
    return sorted({value for value in values if value})


def _mapping_to_float_dict(payload: object) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, float] = {}
    for key, value in payload.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            result[str(key)] = numeric
    return result


def _nested_metrics_to_float_dict(payload: object) -> Dict[str, Dict[str, float]]:
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for dataset_id, metrics in payload.items():
        metric_map = _mapping_to_float_dict(metrics)
        if metric_map:
            result[str(dataset_id)] = metric_map
    return result


def _mean_metric_maps(dataset_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for metric_map in dataset_metrics.values():
        for metric_name, metric_value in metric_map.items():
            values.setdefault(metric_name, []).append(float(metric_value))
    return {
        metric_name: float(np.mean(metric_values))
        for metric_name, metric_values in values.items()
        if metric_values
    }


def _mean_value(values: Dict[str, float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(list(values.values())))


def _mean_named_metric_maps(metric_maps: Sequence[Dict[str, float]]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for metric_map in metric_maps:
        for metric_name, metric_value in metric_map.items():
            values.setdefault(metric_name, []).append(float(metric_value))
    return {
        metric_name: float(np.mean(metric_values))
        for metric_name, metric_values in values.items()
        if metric_values
    }


def _resolve_visualization_scope(path: Path) -> Tuple[Path, Path, Optional[Path], str]:
    selected_path = Path(path)
    selected_name = selected_path.name.lower()
    metric_markers = ("metric_phase_summary.json", "alpha_report.json", "summary.json", "leaderboard.json")
    root_markers = ("experiment_manifest.json", "run_state.json", "stage1_summary.json", "stage2_summary.json", "leaderboard.json")

    if selected_name == "metric_experiments":
        experiment_root = selected_path.parent
        return selected_path, experiment_root, selected_path, "metric_phase_dir"

    metric_here = any(_candidate_path_exists(selected_path / name) for name in metric_markers)
    root_here = any(_candidate_path_exists(selected_path / name) for name in root_markers)
    if metric_here and not root_here:
        parent = selected_path.parent
        if parent != selected_path and any(_candidate_path_exists(parent / name) for name in root_markers):
            return selected_path, parent, selected_path, "metric_phase_dir"

    metric_phase_dir = selected_path / "metric_experiments"
    if any(_candidate_path_exists(selected_path / name) for name in root_markers):
        return selected_path, selected_path, (metric_phase_dir if metric_phase_dir.exists() else None), "experiment_root"

    return selected_path, selected_path, None, "legacy"


def _resolve_primary_metric(
    experiment_manifest: Optional[Dict[str, object]],
    run_state: Optional[Dict[str, object]],
    scoring_weights: Dict[str, float],
) -> Tuple[str, str]:
    if isinstance(experiment_manifest, dict):
        reporting = cast(Dict[str, object], experiment_manifest.get("reporting", {}))
        metric = str(reporting.get("primary_metric", "")).strip()
        if metric:
            return metric, "experiment_manifest.reporting.primary_metric"

    if isinstance(run_state, dict):
        fingerprint_payload = cast(Dict[str, object], run_state.get("fingerprint_payload", {}))
        spec_payload = cast(Dict[str, object], fingerprint_payload.get("spec", {}))
        reporting = cast(Dict[str, object], spec_payload.get("reporting", {}))
        metric = str(reporting.get("primary_metric", "")).strip()
        if metric:
            return metric, "run_state.fingerprint_payload.spec.reporting.primary_metric"

    if scoring_weights:
        return next(iter(scoring_weights.keys())), "scoring.metric_weights"

    return DEFAULT_PRIMARY_METRIC, "default"


def _resolve_configured_stage_mode(
    experiment_manifest: Optional[Dict[str, object]],
    run_state: Optional[Dict[str, object]],
) -> str:
    valid_modes = {"two_stage", "single_stage"}
    if isinstance(experiment_manifest, dict):
        manifest_stages = cast(Dict[str, object], experiment_manifest.get("stages", {}))
        mode = str(manifest_stages.get("mode", "")).strip().lower()
        if mode in valid_modes:
            return mode
    if isinstance(run_state, dict):
        fingerprint_payload = cast(Dict[str, object], run_state.get("fingerprint_payload", {}))
        spec_payload = cast(Dict[str, object], fingerprint_payload.get("spec", {}))
        spec_stages = cast(Dict[str, object], spec_payload.get("stages", {}))
        mode = str(spec_stages.get("mode", "")).strip().lower()
        if mode in valid_modes:
            return mode
    return "two_stage"


def _normalize_alpha_summary_rows(
    payload: object,
    primary_metric: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not isinstance(payload, dict):
        return [], []

    summary_rows: List[Dict[str, object]] = []
    raw_summary = payload.get("alpha_family_summary")
    if isinstance(raw_summary, list):
        metric_key = f"{primary_metric}_mean"
        for item in raw_summary:
            if not isinstance(item, dict):
                continue
            summary_rows.append(
                {
                    "family": normalize_optional_displacement_metric_family(
                        item.get("family"),
                        allow_unknown=True,
                    ),
                    "alpha": _coerce_float(item.get("alpha")),
                    "count": int(item.get("count", 0) or 0),
                    "mean_score": _coerce_float(item.get("mean_score")),
                    "mean_dataset_score": _coerce_float(item.get("mean_dataset_score")),
                    "primary_metric_mean": _coerce_float(
                        item.get("primary_metric_mean", item.get(metric_key))
                    ),
                    "mean_runtime_seconds": _coerce_float(item.get("mean_runtime_seconds")),
                    "mean_displacement_magnitude": _coerce_float(
                        item.get(
                            "mean_displacement_magnitude",
                            item.get("mean_displacement_magnitude_mean"),
                        )
                    ),
                    "mean_view_std": _coerce_float(
                        item.get("mean_view_std", item.get("mean_view_std_mean"))
                    ),
                }
            )
        summary_rows = [
            item
            for item in summary_rows
            if item.get("family") is not None and item.get("alpha") is not None
        ]
        summary_rows.sort(key=lambda item: (str(item["family"]), float(cast(float, item["alpha"]))))

    ranked_runs: List[Dict[str, object]] = []
    raw_ranked = payload.get("ranked_runs")
    if isinstance(raw_ranked, list):
        ranked_runs = []
        for item in raw_ranked:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            family = normalize_optional_displacement_metric_family(
                row.get("family"),
                allow_unknown=True,
            )
            if family is not None:
                row["family"] = family
            ranked_runs.append(row)

    return summary_rows, ranked_runs


def _generation_context_from_sources(
    run_state: Optional[Dict[str, object]],
    experiment_manifest: Optional[Dict[str, object]],
) -> Dict[str, object]:
    generations_raw: List[Dict[str, object]] = []
    if isinstance(run_state, dict):
        for item in cast(List[object], run_state.get("spec_generations", [])):
            if isinstance(item, dict):
                generations_raw.append(dict(item))

    active_generation_id = None
    if isinstance(run_state, dict):
        active_generation_id = _coerce_optional_str(run_state.get("active_generation_id"))
    if active_generation_id is None and isinstance(experiment_manifest, dict):
        active_generation_id = _coerce_optional_str(experiment_manifest.get("generation_id"))
    if active_generation_id is None and generations_raw:
        active_generation_id = _coerce_optional_str(generations_raw[-1].get("generation_id"))

    ordered_generation_ids = [
        str(item.get("generation_id"))
        for item in generations_raw
        if _coerce_optional_str(item.get("generation_id")) is not None
    ]
    previous_generation_id = None
    if active_generation_id and active_generation_id in ordered_generation_ids:
        active_idx = ordered_generation_ids.index(active_generation_id)
        if active_idx > 0:
            previous_generation_id = ordered_generation_ids[active_idx - 1]
    elif len(ordered_generation_ids) >= 2:
        previous_generation_id = ordered_generation_ids[-2]

    status_counts: Counter[str] = Counter()
    generation_run_counts: Counter[str] = Counter()
    stage_status_counts: Dict[str, Counter[str]] = {}
    if isinstance(run_state, dict):
        runs = cast(Dict[str, object], run_state.get("runs", {}))
        for entry in runs.values():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "unknown") or "unknown")
            stage = str(entry.get("stage", "unknown") or "unknown")
            generation_id = _coerce_optional_str(entry.get("generation_id")) or "unknown"
            status_counts[status] += 1
            generation_run_counts[generation_id] += 1
            stage_status_counts.setdefault(stage, Counter())[status] += 1

    merge_history = []
    if isinstance(run_state, dict):
        merge_history = [
            dict(item)
            for item in cast(List[object], run_state.get("merge_history", []))
            if isinstance(item, dict)
        ]

    return {
        "active_generation_id": active_generation_id,
        "previous_generation_id": previous_generation_id,
        "spec_generations": generations_raw,
        "ordered_generation_ids": ordered_generation_ids,
        "run_status_counts": dict(status_counts),
        "run_generation_counts": dict(generation_run_counts),
        "stage_status_counts": {
            stage: dict(counts) for stage, counts in stage_status_counts.items()
        },
        "merge_history": merge_history,
    }


def _load_generation_product_candidates(
    *,
    experiment_root: Path,
    generation_id: Optional[str],
    primary_metric: str,
    warnings: List[str],
) -> Tuple[List[CandidateVisualizationRow], List[str]]:
    if not generation_id:
        return [], []
    products_dir = experiment_root / "generations" / generation_id / "products"
    if not products_dir.exists():
        return [], []

    rows: List[CandidateVisualizationRow] = []
    used_files: List[str] = []
    locations = [
        ("stage1", products_dir / "stage1_summary.json", products_dir),
        ("stage2", products_dir / "stage2_summary.json", products_dir),
        (
            "metric_phase",
            products_dir / "metric_experiments" / "metric_phase_summary.json",
            products_dir / "metric_experiments",
        ),
    ]
    for stage_name, summary_path, base_dir in locations:
        payload, used_path = _read_json_optional(
            summary_path,
            validator=lambda obj: isinstance(obj, dict),
            warnings=warnings,
        )
        if used_path is not None:
            used_files.append(str(used_path))
        if not isinstance(payload, dict):
            continue
        rankings = [
            cast(Dict[str, object], item)
            for item in cast(List[object], payload.get("rankings", []))
            if isinstance(item, dict)
        ]
        for rank, ranking in enumerate(rankings, start=1):
            rows.append(
                _candidate_row_from_ranking(
                    stage_name=stage_name,
                    ranking=ranking,
                    base_dir=base_dir,
                    rank_fallback=rank,
                    primary_metric=primary_metric,
                )
            )
    return rows, used_files


def _with_generation_candidate_identity(
    row: CandidateVisualizationRow,
    generation_id: Optional[str],
) -> CandidateVisualizationRow:
    generation_text = generation_id or row.generation_id or "unknown"
    base_row_id = row.row_id
    if base_row_id.startswith(f"{generation_text}:"):
        qualified_row_id = base_row_id
    else:
        qualified_row_id = f"{generation_text}:{base_row_id}"
    return replace(
        row,
        row_id=qualified_row_id,
        display_name=f"{generation_text}:{row.stage}:{row.candidate_id}",
        generation_id=generation_text,
    )


def _with_generation_record_identity(
    record: RunMetricRecord,
    generation_id: Optional[str],
) -> RunMetricRecord:
    generation_text = generation_id or record.generation_id or "unknown"
    base_row_id = record.row_id
    if base_row_id.startswith(f"{generation_text}:"):
        qualified_row_id = base_row_id
    else:
        qualified_row_id = f"{generation_text}:{base_row_id}"
    return replace(record, row_id=qualified_row_id, generation_id=generation_text)


def _candidate_lookup_by_generation_rows(
    rows: Sequence[CandidateVisualizationRow],
) -> Dict[Tuple[str, str], CandidateVisualizationRow]:
    return {(row.stage, row.candidate_id): row for row in rows}


def _load_generation_product_run_records(
    *,
    experiment_root: Path,
    generation_id: Optional[str],
    primary_metric: str,
    candidate_rows: Sequence[CandidateVisualizationRow],
    warnings: List[str],
) -> Tuple[List[RunMetricRecord], List[str]]:
    if not generation_id:
        return [], []
    products_dir = experiment_root / "generations" / generation_id / "products"
    if not products_dir.exists():
        return [], []

    used_files: List[str] = []
    records_by_id: Dict[str, RunMetricRecord] = {}
    candidate_lookup = _candidate_lookup_by_generation_rows(candidate_rows)
    locations: List[Tuple[Optional[str], Path, Path]] = [
        (None, products_dir / "summary.json", products_dir),
        (
            "metric_phase",
            products_dir / "metric_experiments" / "summary.json",
            products_dir / "metric_experiments",
        ),
    ]
    for stage_hint, summary_path, base_dir in locations:
        payload, used_path = _read_json_optional(
            summary_path,
            validator=lambda obj: isinstance(obj, list),
            warnings=warnings,
        )
        if used_path is not None:
            used_files.append(str(used_path))
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            record = _build_run_metric_record(
                metric_record=cast(Dict[str, object], item),
                primary_metric=primary_metric,
                stage_hint=stage_hint,
                base_dir=base_dir,
                run_lookup={},
                candidate_lookup=candidate_lookup,
            )
            if record is not None:
                _register_run_metric_record(
                    records_by_id,
                    _with_generation_record_identity(record, generation_id),
                )
    return list(records_by_id.values()), used_files


def _build_all_generation_collections(
    *,
    experiment_root: Path,
    generation_summary: Dict[str, object],
    active_candidates: Sequence[CandidateVisualizationRow],
    active_run_metric_records: Sequence[RunMetricRecord],
    primary_metric: str,
    warnings: List[str],
) -> Tuple[List[CandidateVisualizationRow], List[RunMetricRecord], List[str]]:
    active_generation_id = _coerce_optional_str(generation_summary.get("active_generation_id"))
    ordered_generation_ids = [
        generation_id
        for generation_id in cast(List[object], generation_summary.get("ordered_generation_ids", []))
        if _coerce_optional_str(generation_id) is not None
        for generation_id in [str(generation_id)]
    ]
    if active_generation_id and active_generation_id not in ordered_generation_ids:
        ordered_generation_ids.append(active_generation_id)

    used_files: List[str] = []
    candidates_by_generation: Dict[str, List[CandidateVisualizationRow]] = {}
    records_by_generation: Dict[str, List[RunMetricRecord]] = {}

    for generation_id in ordered_generation_ids:
        if generation_id == active_generation_id:
            candidates_by_generation[generation_id] = [
                _with_generation_candidate_identity(row, generation_id)
                for row in active_candidates
            ]
            records_by_generation[generation_id] = [
                _with_generation_record_identity(record, generation_id)
                for record in active_run_metric_records
            ]
            continue
        generation_candidates, candidate_files = _load_generation_product_candidates(
            experiment_root=experiment_root,
            generation_id=generation_id,
            primary_metric=primary_metric,
            warnings=warnings,
        )
        used_files.extend(candidate_files)
        qualified_candidates = [
            _with_generation_candidate_identity(row, generation_id)
            for row in generation_candidates
        ]
        candidates_by_generation[generation_id] = qualified_candidates
        generation_records, record_files = _load_generation_product_run_records(
            experiment_root=experiment_root,
            generation_id=generation_id,
            primary_metric=primary_metric,
            candidate_rows=qualified_candidates,
            warnings=warnings,
        )
        used_files.extend(record_files)
        records_by_generation[generation_id] = generation_records

    if not ordered_generation_ids:
        return list(active_candidates), list(active_run_metric_records), used_files

    all_candidates: List[CandidateVisualizationRow] = []
    all_records: List[RunMetricRecord] = []
    for generation_id in ordered_generation_ids:
        all_candidates.extend(candidates_by_generation.get(generation_id, []))
        all_records.extend(records_by_generation.get(generation_id, []))
    return all_candidates, all_records, used_files


def _peek_workspace_keys(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    for candidate in (path, Path(f"{path}.bak")):
        if not candidate.exists():
            continue
        try:
            with np.load(candidate, allow_pickle=True) as payload:
                return list(payload.files)
        except Exception:
            continue
    return []


def _load_workspace_diagnostics(
    path: Optional[Path],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    warnings: List[str] = []
    if path is None:
        return {}, ["workspace.npz unavailable"]
    try:
        payload, _, read_warnings = read_npz_first_valid(
            path,
            allow_pickle=True,
            validator=lambda obj: isinstance(obj, dict),
        )
        warnings.extend(read_warnings)
        return payload, warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to read workspace.npz: {type(exc).__name__}: {exc}")
        return {}, warnings


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


def _numeric_param(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = _coerce_float(value)
    return float("nan") if numeric is None else float(numeric)


def _params_from_config(config: Dict[str, object]) -> Dict[str, object]:
    params = {
        str(key): value
        for key, value in config.items()
        if key != "hdr_exposure_times" and value is not None
    }
    family = normalize_optional_displacement_metric_family(
        params.get("displacement_metric_family"),
        allow_unknown=True,
    )
    if family is not None:
        params["displacement_metric_family"] = family
    return params


def _canonicalize_metric_family_fields(mapping: Dict[str, object]) -> Dict[str, object]:
    normalized = {str(key): value for key, value in mapping.items()}
    family = normalize_optional_displacement_metric_family(
        normalized.get("displacement_metric_family"),
        allow_unknown=True,
    )
    if family is not None:
        normalized["displacement_metric_family"] = family
    return normalized


def _resolve_ranking_metric(
    metrics: Dict[str, float],
    primary_metric: str,
    fallback_metrics: Sequence[str] = DEFAULT_FALLBACK_METRICS,
) -> Tuple[Optional[str], Optional[float]]:
    for metric_name in (primary_metric, *fallback_metrics):
        if metric_name not in metrics:
            continue
        numeric = _coerce_float(metrics[metric_name])
        if numeric is not None:
            return metric_name, numeric
    return None, None


def _extract_metric_axes(
    config: Dict[str, object],
    params: Dict[str, object],
    metric_formulation: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    formulation = metric_formulation or {}
    family = formulation.get("family")
    if family is None:
        family = config.get("displacement_metric_family", params.get("displacement_metric_family"))
    alpha = formulation.get("alpha")
    if alpha is None:
        alpha = config.get("displacement_alpha", params.get("displacement_alpha"))
    formulation_id = formulation.get("id")
    if formulation_id is None:
        formulation_id = config.get("metric_formulation_id", params.get("metric_formulation_id"))

    family_text = normalize_optional_displacement_metric_family(
        family,
        allow_unknown=True,
    )
    alpha_value = _coerce_float(alpha)
    formulation_text = (
        str(formulation_id).strip()
        if formulation_id is not None and str(formulation_id).strip()
        else None
    )
    return family_text, alpha_value, formulation_text


def _discover_preview_paths(
    base_dir: Path,
    stage_name: str,
    candidate_id: str,
    dataset_ids: Sequence[str],
) -> Dict[str, str]:
    preview_paths: Dict[str, str] = {}
    for dataset_id in dataset_ids:
        preview_path = base_dir / "runs" / stage_name / candidate_id / dataset_id / "avg_rectified.png"
        if preview_path.exists():
            preview_paths[dataset_id] = str(preview_path)
    return preview_paths


def _discover_run_artifact_paths(
    base_dir: Path,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    run_dir = base_dir / "runs" / stage_name / candidate_id / dataset_id
    preview_path = run_dir / "avg_rectified.png"
    workspace_path = run_dir / "workspace.npz"
    metric_record_path = run_dir / "metric_record.json"
    return (
        str(preview_path) if preview_path.exists() else None,
        str(workspace_path) if _candidate_path_exists(workspace_path) else None,
        str(metric_record_path) if _candidate_path_exists(metric_record_path) else None,
    )


def _infer_run_identity(
    run_name: str,
    *,
    stage_hint: Optional[str],
    run_lookup: Dict[str, Dict[str, object]],
    path_hint: Optional[Path] = None,
) -> Tuple[str, str, str]:
    run_state_entry = run_lookup.get(run_name)
    if isinstance(run_state_entry, dict):
        stage_name = str(run_state_entry.get("stage", stage_hint or "summary"))
        candidate_id = str(run_state_entry.get("candidate_id", ""))
        dataset_id = str(run_state_entry.get("dataset_id", ""))
        if stage_name and candidate_id and dataset_id:
            return stage_name, candidate_id, dataset_id

    stage_name = stage_hint or "summary"
    remainder = run_name
    stage_match = re.match(r"^(legacy|stage1|stage2|metric_phase)_(.+)$", run_name)
    if stage_match is not None:
        stage_name = stage_match.group(1)
        remainder = stage_match.group(2)
    elif stage_name == "metric_phase" and remainder.startswith("metric_"):
        remainder = remainder[len("metric_") :]

    if "_" in remainder:
        candidate_id, dataset_id = remainder.rsplit("_", 1)
        if candidate_id and dataset_id:
            return stage_name, candidate_id, dataset_id

    if path_hint is not None:
        parts = path_hint.parts
        if len(parts) >= 3:
            candidate_id = parts[-2]
            dataset_id = parts[-1]
            if len(parts) >= 5 and parts[-4] == "runs":
                stage_name = parts[-3]
            return stage_name, candidate_id, dataset_id

    return stage_name, remainder or run_name, "aggregate"


def _aggregate_metric_maps_from_run_records(
    run_records: Sequence[RunMetricRecord],
    extractor: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for record in run_records:
        metric_map = cast(Dict[str, float], getattr(record, extractor))
        if not metric_map:
            continue
        grouped.setdefault((record.stage, record.candidate_id), []).append(metric_map)
    return {
        key: _mean_named_metric_maps(metric_maps)
        for key, metric_maps in grouped.items()
        if metric_maps
    }


def _build_diagnostic_availability(run_records: Sequence[RunMetricRecord]) -> Dict[str, object]:
    components = set()
    has_score_maps = False
    has_displacement_maps = False
    has_component_stats = False
    checked_workspaces: Dict[Optional[str], List[str]] = {}

    for record in run_records:
        diagnostics = record.displacement_score_diagnostics
        component_stats = diagnostics.get("component_selected_stats")
        if isinstance(component_stats, dict) and component_stats:
            has_component_stats = True
            components.update(str(name) for name in component_stats.keys())

        workspace_path = record.workspace_path
        if workspace_path not in checked_workspaces:
            checked_workspaces[workspace_path] = _peek_workspace_keys(
                None if workspace_path is None else Path(workspace_path)
            )
        workspace_keys = checked_workspaces[workspace_path]
        if "pixel_blended_scores" in workspace_keys:
            has_score_maps = True
        if "pixel_displacements_x" in workspace_keys and "pixel_displacements_y" in workspace_keys:
            has_displacement_maps = True
        for key in workspace_keys:
            if key.startswith("pixel_component_score_"):
                components.add(key.replace("pixel_component_score_", "", 1))

    return {
        "score_maps": has_score_maps,
        "displacement_maps": has_displacement_maps,
        "component_stats": has_component_stats,
        "components": sorted(components),
    }


def _stage_progress_from_sources(
    *,
    stage_name: str,
    summary_payload: Optional[Dict[str, object]],
    run_state: Optional[Dict[str, object]],
    source_path: Optional[Path],
) -> StageVisualizationSummary:
    stage_state = {}
    run_entries: List[Dict[str, object]] = []
    if isinstance(run_state, dict):
        stages_map = cast(Dict[str, object], run_state.get("stages", {}))
        stage_state = cast(Dict[str, object], stages_map.get(stage_name, {}))
        runs_map = cast(Dict[str, object], run_state.get("runs", {}))
        run_entries = [
            cast(Dict[str, object], value)
            for value in runs_map.values()
            if isinstance(value, dict) and str(value.get("stage", "")) == stage_name
        ]

    status_counts = Counter(str(entry.get("status", "pending")) for entry in run_entries)
    recorded_runs = int(
        cast(Dict[str, object], summary_payload).get("num_runs", 0)
        if isinstance(summary_payload, dict)
        else len(run_entries)
    )
    planned_runs = int(stage_state.get("planned_runs", 0) or 0)
    if planned_runs <= 0:
        planned_runs = recorded_runs if recorded_runs > 0 else len(run_entries)

    successful_runs = (
        int(cast(Dict[str, object], summary_payload).get("num_successful", 0))
        if isinstance(summary_payload, dict)
        else sum(status_counts.get(status, 0) for status in SUCCESS_STATUSES)
    )
    failed_runs_count = (
        int(cast(Dict[str, object], summary_payload).get("num_failed", 0))
        if isinstance(summary_payload, dict)
        else status_counts.get("failed", 0)
    )
    running_runs = status_counts.get("running", 0)
    pending_runs = status_counts.get("pending", 0)
    if planned_runs > 0 and pending_runs == 0:
        pending_runs = max(0, planned_runs - successful_runs - failed_runs_count - running_runs)

    failed_runs = []
    execution = {}
    rankings_count = 0
    if isinstance(summary_payload, dict):
        failed_runs = [
            cast(Dict[str, object], item)
            for item in cast(List[object], summary_payload.get("failed_runs", []))
            if isinstance(item, dict)
        ]
        execution = cast(Dict[str, object], summary_payload.get("execution", {}))
        rankings_count = len(
            [
                item
                for item in cast(List[object], summary_payload.get("rankings", []))
                if isinstance(item, dict)
            ]
        )

    skipped_valid_count = int(
        stage_state.get("skipped_valid_count", status_counts.get("skipped_valid", 0)) or 0
    )
    recomputed_count = int(
        stage_state.get("recomputed_count", status_counts.get("recomputed", 0)) or 0
    )
    return StageVisualizationSummary(
        name=stage_name,
        num_runs_planned=int(planned_runs),
        num_runs_recorded=int(recorded_runs),
        num_successful=int(successful_runs),
        num_failed=int(failed_runs_count),
        num_pending=int(pending_runs),
        num_running=int(running_runs),
        skipped_valid_count=int(skipped_valid_count),
        recomputed_count=int(recomputed_count),
        rankings_count=int(rankings_count),
        execution=dict(execution),
        failed_runs=failed_runs,
        source_path=str(source_path) if source_path is not None else None,
    )


def _candidate_row_from_ranking(
    *,
    stage_name: str,
    ranking: Dict[str, object],
    base_dir: Path,
    rank_fallback: int,
    primary_metric: str,
) -> CandidateVisualizationRow:
    candidate_id = str(ranking.get("candidate_id", f"{stage_name}_{rank_fallback:04d}"))
    config = _canonicalize_metric_family_fields(
        cast(Dict[str, object], ranking.get("config", {}))
    )
    params = _canonicalize_metric_family_fields(
        cast(Dict[str, object], ranking.get("params", {}))
    )
    dataset_scores = _mapping_to_float_dict(ranking.get("dataset_scores", {}))
    dataset_runtimes = _mapping_to_float_dict(ranking.get("runtime_seconds", {}))
    dataset_metrics = _nested_metrics_to_float_dict(ranking.get("dataset_metrics", {}))
    metric_means = _mean_metric_maps(dataset_metrics)
    family, alpha, formulation_id = _extract_metric_axes(config, params)
    preview_paths = _discover_preview_paths(
        base_dir=base_dir,
        stage_name=stage_name,
        candidate_id=candidate_id,
        dataset_ids=sorted(set(dataset_scores) | set(dataset_metrics) | set(dataset_runtimes)),
    )
    ranking_metric_name, ranking_metric_value = _resolve_ranking_metric(metric_means, primary_metric)
    return CandidateVisualizationRow(
        row_id=f"{stage_name}:{candidate_id}",
        candidate_id=candidate_id,
        stage=stage_name,
        display_name=f"{stage_name}:{candidate_id}",
        rank=int(ranking.get("rank", rank_fallback)),
        strategy=str(ranking.get("strategy", config.get("strategy", "unknown"))),
        score=_coerce_float(ranking.get("score")),
        mean_dataset_score=_coerce_float(ranking.get("mean_dataset_score")),
        std_dataset_score=_coerce_float(ranking.get("std_dataset_score")),
        ranking_metric_name=ranking_metric_name,
        ranking_metric_value=ranking_metric_value,
        dataset_scores=dataset_scores,
        dataset_runtimes=dataset_runtimes,
        dataset_metrics=dataset_metrics,
        metric_means=metric_means,
        mean_runtime_seconds=_mean_value(dataset_runtimes),
        missing_datasets=[str(item) for item in cast(List[object], ranking.get("missing_datasets", []))],
        params=dict(params),
        config=dict(config),
        family=family,
        alpha=alpha,
        formulation_id=formulation_id,
        preview_paths=preview_paths,
        generation_id=_coerce_optional_str(ranking.get("generation_id")),
    )


def _candidate_row_from_legacy_record(
    *,
    output_dir: Path,
    record: Dict[str, object],
    rank: int,
    primary_metric: str,
) -> CandidateVisualizationRow:
    run_name = str(record.get("run", f"legacy_{rank:04d}"))
    config = _canonicalize_metric_family_fields(
        cast(Dict[str, object], record.get("config", {}))
    )
    params = _params_from_config(config)
    metrics = _mapping_to_float_dict(record.get("global_and_local_metrics", {}))
    ranking_metric_name, ranking_metric_value = _resolve_ranking_metric(metrics, primary_metric)
    family, alpha, formulation_id = _extract_metric_axes(
        config,
        params,
        cast(Optional[Dict[str, object]], record.get("metric_formulation")),
    )
    preview_path = output_dir / run_name / "avg_rectified.png"
    preview_paths = {"aggregate": str(preview_path)} if preview_path.exists() else {}
    dataset_scores = {"aggregate": float(ranking_metric_value)} if ranking_metric_value is not None else {}
    return CandidateVisualizationRow(
        row_id=f"legacy:{run_name}",
        candidate_id=run_name,
        stage="legacy",
        display_name=run_name,
        rank=rank,
        strategy=str(config.get("strategy", "legacy")),
        score=ranking_metric_value,
        mean_dataset_score=ranking_metric_value,
        std_dataset_score=0.0 if ranking_metric_value is not None else None,
        ranking_metric_name=ranking_metric_name,
        ranking_metric_value=ranking_metric_value,
        dataset_scores=dataset_scores,
        dataset_runtimes={},
        dataset_metrics={"aggregate": metrics} if metrics else {},
        metric_means=dict(metrics),
        mean_runtime_seconds=None,
        missing_datasets=[],
        params=params,
        config=dict(config),
        family=family,
        alpha=alpha,
        formulation_id=formulation_id,
        preview_paths=preview_paths,
        generation_id=_coerce_optional_str(record.get("generation_id")),
    )


def _load_legacy_records(output_dir: Path, warnings: List[str]) -> List[Dict[str, object]]:
    payload, _ = _read_json_optional(
        output_dir / "summary.json",
        validator=lambda obj: isinstance(obj, list),
        warnings=warnings,
    )
    if isinstance(payload, list):
        return [cast(Dict[str, object], item) for item in payload if isinstance(item, dict)]

    records: List[Dict[str, object]] = []
    for record_path in sorted(output_dir.glob("*/metric_record.json")):
        record_payload, _ = _read_json_optional(
            record_path,
            validator=lambda obj: isinstance(obj, dict),
            warnings=warnings,
        )
        if isinstance(record_payload, dict):
            records.append(record_payload)
    return records


def _order_metric_names(metric_names: Iterable[str], scoring_weights: Dict[str, float]) -> List[str]:
    unique_names = list(dict.fromkeys(metric_names))
    ordered = [name for name in scoring_weights.keys() if name in unique_names]
    ordered.extend(sorted(name for name in unique_names if name not in ordered))
    return ordered


def _build_alpha_family_summary(
    rows: Sequence[CandidateVisualizationRow | RunMetricRecord],
    primary_metric: str,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[CandidateVisualizationRow | RunMetricRecord]] = {}
    for row in rows:
        if row.family is None or row.alpha is None:
            continue
        grouped.setdefault((row.family, float(row.alpha)), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (family, alpha), family_rows in grouped.items():
        score_values = [row.score for row in family_rows if row.score is not None]
        mean_dataset_values = [
            row.mean_dataset_score for row in family_rows if row.mean_dataset_score is not None
        ]
        runtime_values = [
            row.mean_runtime_seconds for row in family_rows if row.mean_runtime_seconds is not None
        ]
        primary_metric_values = [
            row.metric_means[primary_metric]
            for row in family_rows
            if primary_metric in row.metric_means
        ]
        displacement_values = [
            row.displacement_field_metrics.get("mean_magnitude")
            for row in family_rows
            if row.displacement_field_metrics.get("mean_magnitude") is not None
        ]
        consistency_values = [
            row.inter_view_consistency_metrics.get("mean_view_std")
            for row in family_rows
            if row.inter_view_consistency_metrics.get("mean_view_std") is not None
        ]
        summary_rows.append(
            {
                "family": family,
                "alpha": float(alpha),
                "count": len(family_rows),
                "mean_score": float(np.mean(score_values)) if score_values else None,
                "mean_dataset_score": (
                    float(np.mean(mean_dataset_values)) if mean_dataset_values else None
                ),
                "primary_metric_mean": (
                    float(np.mean(primary_metric_values)) if primary_metric_values else None
                ),
                "mean_runtime_seconds": (
                    float(np.mean(runtime_values)) if runtime_values else None
                ),
                "mean_displacement_magnitude": (
                    float(np.mean(cast(List[float], displacement_values))) if displacement_values else None
                ),
                "mean_view_std": (
                    float(np.mean(cast(List[float], consistency_values))) if consistency_values else None
                ),
            }
        )
    summary_rows.sort(key=lambda item: (str(item["family"]), float(item["alpha"])))
    return summary_rows


def _build_alpha_ranked_runs_from_run_records(
    run_records: Sequence[RunMetricRecord],
) -> List[Dict[str, object]]:
    ranked_rows: List[Dict[str, object]] = []
    for rank, record in enumerate(
        sorted(
            (
                item
                for item in run_records
                if item.ranking_metric_value is not None
                and item.family is not None
                and item.alpha is not None
            ),
            key=lambda item: cast(float, item.ranking_metric_value),
            reverse=True,
        ),
        start=1,
    ):
        ranked_rows.append(
            {
                "rank": rank,
                "run": record.run_name,
                "family": record.family,
                "alpha": record.alpha,
                "formulation_id": record.formulation_id,
                "ranking_metric_name": record.ranking_metric_name,
                "ranking_metric_value": record.ranking_metric_value,
                "score": record.score,
                "mean_dataset_score": record.mean_dataset_score,
                "mean_runtime_seconds": record.mean_runtime_seconds,
                "metric_record_path": record.metric_record_path,
            }
        )
    return ranked_rows


def _collect_metric_records_from_summary(
    summary_payload: Optional[List[Dict[str, object]]],
    primary_metric: str,
) -> List[CandidateVisualizationRow]:
    if not summary_payload:
        return []
    rows: List[CandidateVisualizationRow] = []
    for rank, record in enumerate(summary_payload, start=1):
        metrics = _mapping_to_float_dict(record.get("global_and_local_metrics", {}))
        if not metrics:
            continue
        ranking_metric_name, ranking_metric_value = _resolve_ranking_metric(metrics, primary_metric)
        config = cast(Dict[str, object], record.get("config", {}))
        params = _params_from_config(config)
        family, alpha, formulation_id = _extract_metric_axes(
            config,
            params,
            cast(Optional[Dict[str, object]], record.get("metric_formulation")),
        )
        rows.append(
            CandidateVisualizationRow(
                row_id=f"summary:{record.get('run', rank)}",
                candidate_id=str(record.get("run", f"summary_{rank:04d}")),
                stage="summary",
                display_name=str(record.get("run", f"summary_{rank:04d}")),
                rank=rank,
                strategy=str(config.get("strategy", "summary")),
                score=ranking_metric_value,
                mean_dataset_score=ranking_metric_value,
                std_dataset_score=None,
                ranking_metric_name=ranking_metric_name,
                ranking_metric_value=ranking_metric_value,
                dataset_scores={},
                dataset_runtimes={},
                dataset_metrics={},
                metric_means=metrics,
                mean_runtime_seconds=None,
                missing_datasets=[],
                params=params,
                config=config,
                family=family,
                alpha=alpha,
                formulation_id=formulation_id,
            )
        )
    return rows


def _candidate_lookup(model_rows: Sequence[CandidateVisualizationRow]) -> Dict[Tuple[str, str], CandidateVisualizationRow]:
    return {(row.stage, row.candidate_id): row for row in model_rows}


def _build_run_metric_record(
    *,
    metric_record: Dict[str, object],
    primary_metric: str,
    stage_hint: Optional[str],
    base_dir: Path,
    run_lookup: Dict[str, Dict[str, object]],
    candidate_lookup: Dict[Tuple[str, str], CandidateVisualizationRow],
    metric_record_path: Optional[Path] = None,
) -> Optional[RunMetricRecord]:
    run_name = str(metric_record.get("run", "")).strip()
    if not run_name:
        return None

    path_hint = metric_record_path.parent if metric_record_path is not None else None
    stage_name, candidate_id, dataset_id = _infer_run_identity(
        run_name,
        stage_hint=stage_hint,
        run_lookup=run_lookup,
        path_hint=path_hint,
    )
    config = _canonicalize_metric_family_fields(
        cast(Dict[str, object], metric_record.get("config", {}))
    )
    params = _params_from_config(config)
    image_metrics = _mapping_to_float_dict(metric_record.get("global_and_local_metrics", {}))
    global_only_metrics = _mapping_to_float_dict(metric_record.get("global_only_metrics", {}))
    family, alpha, formulation_id = _extract_metric_axes(
        config,
        params,
        cast(Optional[Dict[str, object]], metric_record.get("metric_formulation")),
    )
    ranking_metric_name, ranking_metric_value = _resolve_ranking_metric(image_metrics, primary_metric)
    preview_path, workspace_path, discovered_metric_record_path = _discover_run_artifact_paths(
        base_dir=base_dir,
        stage_name=stage_name,
        candidate_id=candidate_id,
        dataset_id=dataset_id,
    )
    candidate_row = candidate_lookup.get((stage_name, candidate_id))
    score = None
    mean_dataset_score = None
    mean_runtime_seconds = None
    if candidate_row is not None:
        score = candidate_row.dataset_scores.get(dataset_id, candidate_row.score)
        mean_dataset_score = candidate_row.mean_dataset_score
        mean_runtime_seconds = candidate_row.dataset_runtimes.get(
            dataset_id,
            candidate_row.mean_runtime_seconds,
        )
    generation_id = (
        _coerce_optional_str(metric_record.get("generation_id"))
        or _coerce_optional_str(run_lookup.get(run_name, {}).get("generation_id"))
        or (candidate_row.generation_id if candidate_row is not None else None)
    )

    return RunMetricRecord(
        row_id=f"{stage_name}:{candidate_id}:{dataset_id}",
        run_name=run_name,
        stage=stage_name,
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        strategy=str(config.get("strategy", candidate_row.strategy if candidate_row is not None else "unknown")),
        family=family,
        alpha=alpha,
        formulation_id=formulation_id,
        params=params,
        config=dict(config),
        score=score,
        mean_dataset_score=mean_dataset_score,
        mean_runtime_seconds=mean_runtime_seconds,
        metric_means=image_metrics or global_only_metrics,
        displacement_field_metrics=_mapping_to_float_dict(metric_record.get("displacement_field_metrics", {})),
        inter_view_consistency_metrics=_mapping_to_float_dict(
            metric_record.get("inter_view_consistency_metrics", {})
        ),
        displacement_score_diagnostics=cast(
            Dict[str, object],
            metric_record.get("displacement_score_diagnostics", {}) or {},
        ),
        preview_path=preview_path,
        workspace_path=workspace_path,
        metric_record_path=(
            str(metric_record_path)
            if metric_record_path is not None
            else discovered_metric_record_path
        ),
        ranking_metric_name=ranking_metric_name,
        ranking_metric_value=ranking_metric_value,
        generation_id=generation_id,
    )


def _merge_run_metric_record(
    existing: RunMetricRecord,
    incoming: RunMetricRecord,
) -> RunMetricRecord:
    return RunMetricRecord(
        row_id=existing.row_id,
        run_name=existing.run_name or incoming.run_name,
        stage=existing.stage or incoming.stage,
        candidate_id=existing.candidate_id or incoming.candidate_id,
        dataset_id=existing.dataset_id or incoming.dataset_id,
        strategy=existing.strategy or incoming.strategy,
        family=existing.family or incoming.family,
        alpha=existing.alpha if existing.alpha is not None else incoming.alpha,
        formulation_id=existing.formulation_id or incoming.formulation_id,
        params=existing.params or incoming.params,
        config=existing.config or incoming.config,
        score=existing.score if existing.score is not None else incoming.score,
        mean_dataset_score=(
            existing.mean_dataset_score
            if existing.mean_dataset_score is not None
            else incoming.mean_dataset_score
        ),
        mean_runtime_seconds=(
            existing.mean_runtime_seconds
            if existing.mean_runtime_seconds is not None
            else incoming.mean_runtime_seconds
        ),
        metric_means=existing.metric_means or incoming.metric_means,
        displacement_field_metrics=(
            existing.displacement_field_metrics or incoming.displacement_field_metrics
        ),
        inter_view_consistency_metrics=(
            existing.inter_view_consistency_metrics or incoming.inter_view_consistency_metrics
        ),
        displacement_score_diagnostics=(
            existing.displacement_score_diagnostics or incoming.displacement_score_diagnostics
        ),
        preview_path=existing.preview_path or incoming.preview_path,
        workspace_path=existing.workspace_path or incoming.workspace_path,
        metric_record_path=existing.metric_record_path or incoming.metric_record_path,
        ranking_metric_name=existing.ranking_metric_name or incoming.ranking_metric_name,
        ranking_metric_value=(
            existing.ranking_metric_value
            if existing.ranking_metric_value is not None
            else incoming.ranking_metric_value
        ),
        generation_id=existing.generation_id or incoming.generation_id,
    )


def _register_run_metric_record(
    storage: Dict[str, RunMetricRecord],
    record: Optional[RunMetricRecord],
) -> None:
    if record is None:
        return
    existing = storage.get(record.row_id)
    storage[record.row_id] = record if existing is None else _merge_run_metric_record(existing, record)


def _scan_metric_record_paths(base_dir: Path) -> List[Path]:
    return sorted(base_dir.glob("runs/*/*/*/metric_record.json"))


def _load_sweep_visualization_model_impl(path: Path) -> SweepVisualizationModel:
    selected_path = Path(path)
    source_dir, experiment_root, metric_phase_dir, path_scope = _resolve_visualization_scope(selected_path)
    warnings: List[str] = []
    used_files: List[str] = []

    experiment_manifest, manifest_path = _read_json_optional(
        experiment_root / "experiment_manifest.json",
        validator=lambda obj: isinstance(obj, dict),
        warnings=warnings,
    )
    if manifest_path is not None:
        used_files.append(str(manifest_path))
    run_state, run_state_path = _read_json_optional(
        experiment_root / "run_state.json",
        validator=lambda obj: isinstance(obj, dict),
        warnings=warnings,
    )
    if run_state_path is not None:
        used_files.append(str(run_state_path))

    modern_artifacts_present = path_scope in {"experiment_root", "metric_phase_dir"} or any(
        _candidate_path_exists(experiment_root / filename)
        for filename in ("stage1_summary.json", "stage2_summary.json", "leaderboard.json", "run_state.json")
    )

    scoring_weights: Dict[str, float] = {}
    experiment_metadata: Dict[str, object] = {
        "output_dir": str(source_dir.resolve()),
        "experiment_root": str(experiment_root.resolve()),
        "path_scope": path_scope,
    }
    datasets: List[str] = []
    strategies: List[str] = []

    if isinstance(experiment_manifest, dict):
        experiment_metadata.update(
            {
                "generated_at": experiment_manifest.get("generated_at"),
                "strategies": experiment_manifest.get("strategies"),
                "stages": experiment_manifest.get("stages"),
                "scoring": experiment_manifest.get("scoring"),
                "reporting": experiment_manifest.get("reporting"),
            }
        )
        strategies = [
            str(item)
            for item in cast(List[object], experiment_manifest.get("strategies", []))
        ]
        scoring = cast(Dict[str, object], experiment_manifest.get("scoring", {}))
        scoring_weights = _mapping_to_float_dict(scoring.get("metric_weights", {}))
        datasets = [
            str(item.get("dataset_id"))
            for item in cast(List[Dict[str, object]], experiment_manifest.get("datasets", []))
            if isinstance(item, dict) and item.get("dataset_id") is not None
        ]

    if isinstance(run_state, dict):
        experiment_metadata.update(
            {
                "created_at": run_state.get("created_at"),
                "updated_at": run_state.get("updated_at"),
                "experiment_fingerprint": run_state.get("experiment_fingerprint"),
            }
        )
        fingerprint_payload = cast(Dict[str, object], run_state.get("fingerprint_payload", {}))
        spec_payload = cast(Dict[str, object], fingerprint_payload.get("spec", {}))
        if not scoring_weights:
            scoring = cast(Dict[str, object], spec_payload.get("scoring", {}))
            scoring_weights = _mapping_to_float_dict(scoring.get("metric_weights", {}))
        if not strategies:
            strategies = [
                str(item) for item in cast(List[object], spec_payload.get("strategies", []))
            ]
        if not datasets:
            datasets = [
                str(item.get("dataset_id"))
                for item in cast(List[Dict[str, object]], fingerprint_payload.get("datasets", []))
                if isinstance(item, dict) and item.get("dataset_id") is not None
            ]
        if spec_payload:
            experiment_metadata["spec"] = spec_payload

    primary_metric, primary_metric_source = _resolve_primary_metric(
        cast(Optional[Dict[str, object]], experiment_manifest),
        cast(Optional[Dict[str, object]], run_state),
        scoring_weights,
    )
    generation_summary = _generation_context_from_sources(
        cast(Optional[Dict[str, object]], run_state),
        cast(Optional[Dict[str, object]], experiment_manifest),
    )
    configured_stage_mode = _resolve_configured_stage_mode(
        cast(Optional[Dict[str, object]], experiment_manifest),
        cast(Optional[Dict[str, object]], run_state),
    )
    stage_summaries: Dict[str, StageVisualizationSummary] = {}
    stage_candidate_ids: Dict[str, List[str]] = {}
    candidate_rows: List[CandidateVisualizationRow] = []
    alpha_rows_from_records: List[CandidateVisualizationRow] = []
    alpha_family_summary: List[Dict[str, object]] = []
    alpha_ranked_runs: List[Dict[str, object]] = []
    alpha_summary_source = "none"
    metric_alpha_report_payload: Optional[Dict[str, object]] = None
    root_alpha_report_payload: Optional[Dict[str, object]] = None
    run_metric_records_by_id: Dict[str, RunMetricRecord] = {}

    if modern_artifacts_present:
        summary_locations: Dict[str, Path] = {
            "stage1": experiment_root / "stage1_summary.json",
            "stage2": experiment_root / "stage2_summary.json",
        }
        if metric_phase_dir is not None:
            summary_locations["metric_phase"] = metric_phase_dir / "metric_phase_summary.json"

        summary_payloads: Dict[str, Optional[Dict[str, object]]] = {}
        summary_paths: Dict[str, Optional[Path]] = {}
        for stage_name, summary_path in summary_locations.items():
            payload, used_path = _read_json_optional(
                summary_path,
                validator=lambda obj: isinstance(obj, dict),
                warnings=warnings,
            )
            summary_payloads[stage_name] = cast(Optional[Dict[str, object]], payload)
            summary_paths[stage_name] = used_path
            if used_path is not None:
                used_files.append(str(used_path))

        root_summary_payload, root_summary_path = _read_json_optional(
            experiment_root / "summary.json",
            validator=lambda obj: isinstance(obj, list),
            warnings=warnings,
        )
        if root_summary_path is not None:
            used_files.append(str(root_summary_path))
        alpha_rows_from_records.extend(
            _collect_metric_records_from_summary(
                cast(Optional[List[Dict[str, object]]], root_summary_payload),
                primary_metric=primary_metric,
            )
        )

        metric_summary_payload: Optional[List[Dict[str, object]]] = None
        metric_alpha_report_path: Optional[Path] = None
        if metric_phase_dir is not None:
            metric_summary_raw, metric_summary_path = _read_json_optional(
                metric_phase_dir / "summary.json",
                validator=lambda obj: isinstance(obj, list),
                warnings=warnings,
            )
            if metric_summary_path is not None:
                used_files.append(str(metric_summary_path))
            metric_summary_payload = cast(Optional[List[Dict[str, object]]], metric_summary_raw)
            alpha_rows_from_records.extend(
                _collect_metric_records_from_summary(
                    metric_summary_payload,
                    primary_metric=primary_metric,
                )
            )
            metric_alpha_report_payload, metric_alpha_report_path = _read_json_optional(
                metric_phase_dir / "alpha_report.json",
                validator=lambda obj: isinstance(obj, dict),
                warnings=warnings,
            )
            if metric_alpha_report_path is not None:
                used_files.append(str(metric_alpha_report_path))

        root_alpha_report_payload, root_alpha_report_path = _read_json_optional(
            experiment_root / "alpha_report.json",
            validator=lambda obj: isinstance(obj, dict),
            warnings=warnings,
        )
        if root_alpha_report_path is not None:
            used_files.append(str(root_alpha_report_path))

        for stage_name in ("stage1", "stage2", "metric_phase"):
            summary_payload = summary_payloads.get(stage_name)
            if summary_payload is None and not (
                isinstance(run_state, dict)
                and stage_name in cast(Dict[str, object], run_state.get("stages", {}))
            ):
                continue

            base_dir = (
                metric_phase_dir
                if stage_name == "metric_phase" and metric_phase_dir is not None
                else experiment_root
            )
            stage_summary = _stage_progress_from_sources(
                stage_name=stage_name,
                summary_payload=summary_payload,
                run_state=cast(Optional[Dict[str, object]], run_state),
                source_path=summary_paths.get(stage_name),
            )
            stage_summaries[stage_name] = stage_summary

            rankings: List[Dict[str, object]] = []
            if isinstance(summary_payload, dict):
                rankings = [
                    cast(Dict[str, object], item)
                    for item in cast(List[object], summary_payload.get("rankings", []))
                    if isinstance(item, dict)
                ]
            stage_rows = [
                _candidate_row_from_ranking(
                    stage_name=stage_name,
                    ranking=ranking,
                    base_dir=base_dir,
                    rank_fallback=rank,
                    primary_metric=primary_metric,
                )
                for rank, ranking in enumerate(rankings, start=1)
            ]
            stage_rows.sort(key=lambda row: (row.rank, row.candidate_id))
            stage_candidate_ids[stage_name] = [row.row_id for row in stage_rows]
            candidate_rows.extend(stage_rows)

        if not stage_summaries:
            raise ValueError(
                f"No readable stage summaries found under {source_dir}; cannot build visualization model."
            )

        candidate_lookup = _candidate_lookup(candidate_rows)
        run_lookup = {
            str(key): cast(Dict[str, object], value)
            for key, value in cast(
                Dict[str, object],
                cast(Dict[str, object], run_state or {}).get("runs", {}),
            ).items()
            if isinstance(value, dict)
        }
        for record in cast(Optional[List[Dict[str, object]]], root_summary_payload) or []:
            _register_run_metric_record(
                run_metric_records_by_id,
                _build_run_metric_record(
                    metric_record=record,
                    primary_metric=primary_metric,
                    stage_hint=None,
                    base_dir=experiment_root,
                    run_lookup=run_lookup,
                    candidate_lookup=candidate_lookup,
                ),
            )
        for record in metric_summary_payload or []:
            _register_run_metric_record(
                run_metric_records_by_id,
                _build_run_metric_record(
                    metric_record=record,
                    primary_metric=primary_metric,
                    stage_hint="metric_phase",
                    base_dir=metric_phase_dir if metric_phase_dir is not None else experiment_root,
                    run_lookup=run_lookup,
                    candidate_lookup=candidate_lookup,
                ),
            )

        metric_record_paths = _scan_metric_record_paths(experiment_root)
        if metric_phase_dir is not None:
            metric_record_paths.extend(_scan_metric_record_paths(metric_phase_dir))
        for metric_record_path in sorted(set(metric_record_paths)):
            payload, used_path = _read_json_optional(
                metric_record_path,
                validator=lambda obj: isinstance(obj, dict),
                warnings=warnings,
            )
            if used_path is not None:
                used_files.append(str(used_path))
            if not isinstance(payload, dict):
                continue
            in_metric_phase_dir = metric_phase_dir is not None and "metric_experiments" in metric_record_path.parts
            _register_run_metric_record(
                run_metric_records_by_id,
                _build_run_metric_record(
                    metric_record=payload,
                    primary_metric=primary_metric,
                    stage_hint="metric_phase" if in_metric_phase_dir else None,
                    base_dir=metric_phase_dir if in_metric_phase_dir and metric_phase_dir is not None else experiment_root,
                    run_lookup=run_lookup,
                    candidate_lookup=candidate_lookup,
                    metric_record_path=used_path or metric_record_path,
                ),
            )

        stage2_present = bool(summary_payloads.get("stage2"))
        if path_scope == "metric_phase_dir":
            mode = "metric_phase_dir"
        elif configured_stage_mode == "single_stage" and "stage1" in stage_summaries and "stage2" not in stage_summaries:
            mode = "spec_single_stage"
        else:
            mode = "spec_full" if stage2_present else "spec_partial"
    else:
        legacy_records = _load_legacy_records(source_dir, warnings)
        if not legacy_records:
            raise ValueError(f"No readable sweep artifacts found under {source_dir}.")
        legacy_rows = [
            _candidate_row_from_legacy_record(
                output_dir=source_dir,
                record=record,
                rank=rank,
                primary_metric=primary_metric,
            )
            for rank, record in enumerate(legacy_records, start=1)
        ]
        legacy_rows.sort(
            key=lambda row: (
                -(row.ranking_metric_value if row.ranking_metric_value is not None else float("-inf")),
                row.candidate_id,
            )
        )
        for rank, row in enumerate(legacy_rows, start=1):
            row.rank = rank
        candidate_rows.extend(legacy_rows)
        stage_candidate_ids["legacy"] = [row.row_id for row in legacy_rows]
        stage_summaries["legacy"] = StageVisualizationSummary(
            name="legacy",
            num_runs_planned=len(legacy_rows),
            num_runs_recorded=len(legacy_rows),
            num_successful=len(legacy_rows),
            num_failed=0,
            num_pending=0,
            num_running=0,
            skipped_valid_count=0,
            recomputed_count=0,
            rankings_count=len(legacy_rows),
            execution={},
            failed_runs=[],
            source_path=str(source_dir / "summary.json"),
        )
        alpha_rows_from_records.extend(legacy_rows)
        candidate_lookup = _candidate_lookup(candidate_rows)
        for record in legacy_records:
            _register_run_metric_record(
                run_metric_records_by_id,
                _build_run_metric_record(
                    metric_record=record,
                    primary_metric=primary_metric,
                    stage_hint="legacy",
                    base_dir=source_dir,
                    run_lookup={},
                    candidate_lookup=candidate_lookup,
                ),
            )
        mode = "legacy"

    run_metric_records = sorted(
        run_metric_records_by_id.values(),
        key=lambda item: (
            STAGE_ORDER.index(item.stage) if item.stage in STAGE_ORDER else len(STAGE_ORDER),
            item.candidate_id,
            item.dataset_id,
        ),
    )
    metric_phase_alpha_records = [
        record
        for record in run_metric_records
        if record.stage == "metric_phase" and record.family is not None and record.alpha is not None
    ]
    if isinstance(metric_alpha_report_payload, dict):
        alpha_family_summary, alpha_ranked_runs = _normalize_alpha_summary_rows(
            metric_alpha_report_payload,
            primary_metric=primary_metric,
        )
        alpha_summary_source = "metric_phase.alpha_report"
    elif metric_phase_alpha_records:
        alpha_family_summary = _build_alpha_family_summary(metric_phase_alpha_records, primary_metric)
        alpha_ranked_runs = _build_alpha_ranked_runs_from_run_records(metric_phase_alpha_records)
        alpha_summary_source = "metric_phase.run_metric_records"
        if metric_phase_dir is not None:
            warnings.append(
                "Metric-phase alpha report unavailable; synthesized alpha summary from metric-phase records."
            )
    displacement_field_agg = _aggregate_metric_maps_from_run_records(
        run_metric_records,
        "displacement_field_metrics",
    )
    consistency_agg = _aggregate_metric_maps_from_run_records(
        run_metric_records,
        "inter_view_consistency_metrics",
    )
    for row in candidate_rows:
        row.displacement_field_metrics = displacement_field_agg.get((row.stage, row.candidate_id), {})
        row.inter_view_consistency_metrics = consistency_agg.get((row.stage, row.candidate_id), {})

    if not datasets:
        datasets = _sorted_unique(
            dataset_id
            for row in candidate_rows
            for dataset_id in list(row.dataset_scores.keys()) + list(row.dataset_metrics.keys())
        )
    if not datasets:
        datasets = _sorted_unique(record.dataset_id for record in run_metric_records)
    if not datasets and candidate_rows:
        datasets = ["aggregate"]
    if not strategies:
        strategies = _sorted_unique(
            list(row.strategy for row in candidate_rows)
            + [record.strategy for record in run_metric_records]
        )

    parameter_names = _sorted_unique(
        list(param_name for row in candidate_rows for param_name in row.params.keys())
        + [param_name for record in run_metric_records for param_name in record.params.keys()]
    )
    metric_names = _order_metric_names(
        list(metric_name for row in candidate_rows for metric_name in row.metric_means.keys())
        + [metric_name for record in run_metric_records for metric_name in record.metric_means.keys()],
        scoring_weights=scoring_weights,
    )
    displacement_metric_names = _sorted_unique(
        list(metric_name for row in candidate_rows for metric_name in row.displacement_field_metrics.keys())
        + [
            metric_name
            for record in run_metric_records
            for metric_name in record.displacement_field_metrics.keys()
        ]
    )
    consistency_metric_names = _sorted_unique(
        list(metric_name for row in candidate_rows for metric_name in row.inter_view_consistency_metrics.keys())
        + [
            metric_name
            for record in run_metric_records
            for metric_name in record.inter_view_consistency_metrics.keys()
        ]
    )
    stage_order = [stage_name for stage_name in STAGE_ORDER if stage_name in stage_summaries]
    if not stage_order:
        stage_order = [candidate_rows[0].stage] if candidate_rows else []
    primary_stage_name = "metric_phase" if "metric_phase" in stage_summaries else stage_order[-1]

    ordered_candidate_rows: List[CandidateVisualizationRow] = []
    for stage_name in stage_order:
        rows = [row for row in candidate_rows if row.stage == stage_name]
        rows.sort(key=lambda row: (row.rank, row.candidate_id))
        ordered_candidate_rows.extend(rows)

    previous_generation_candidates, previous_generation_files = _load_generation_product_candidates(
        experiment_root=experiment_root,
        generation_id=_coerce_optional_str(generation_summary.get("previous_generation_id")),
        primary_metric=primary_metric,
        warnings=warnings,
    )
    used_files.extend(previous_generation_files)
    generation_summary["previous_generation_candidate_count"] = len(previous_generation_candidates)
    all_generation_candidates, all_generation_run_metric_records, all_generation_files = _build_all_generation_collections(
        experiment_root=experiment_root,
        generation_summary=generation_summary,
        active_candidates=ordered_candidate_rows,
        active_run_metric_records=run_metric_records,
        primary_metric=primary_metric,
        warnings=warnings,
    )
    used_files.extend(all_generation_files)
    generation_summary["all_generation_candidate_count"] = len(all_generation_candidates)
    generation_summary["all_generation_run_record_count"] = len(all_generation_run_metric_records)

    candidate_ids = [row.row_id for row in ordered_candidate_rows]
    dataset_index = {dataset_id: idx for idx, dataset_id in enumerate(datasets)}
    metric_index = {metric_name: idx for idx, metric_name in enumerate(metric_names)}
    parameter_index = {param_name: idx for idx, param_name in enumerate(parameter_names)}
    score_matrix = np.full((len(ordered_candidate_rows), len(datasets)), np.nan, dtype=np.float64)
    runtime_matrix = np.full_like(score_matrix, np.nan)
    metric_tensor = np.full(
        (len(ordered_candidate_rows), len(datasets), len(metric_names)),
        np.nan,
        dtype=np.float64,
    )
    parameter_numeric_matrix = np.full(
        (len(ordered_candidate_rows), len(parameter_names)),
        np.nan,
        dtype=np.float64,
    )
    parameter_text_matrix = np.empty((len(ordered_candidate_rows), len(parameter_names)), dtype=object)
    parameter_text_matrix[:] = ""

    for row_idx, row in enumerate(ordered_candidate_rows):
        for dataset_id, score in row.dataset_scores.items():
            if dataset_id in dataset_index:
                score_matrix[row_idx, dataset_index[dataset_id]] = float(score)
        for dataset_id, runtime in row.dataset_runtimes.items():
            if dataset_id in dataset_index:
                runtime_matrix[row_idx, dataset_index[dataset_id]] = float(runtime)
        for dataset_id, metrics in row.dataset_metrics.items():
            if dataset_id not in dataset_index:
                continue
            for metric_name, metric_value in metrics.items():
                if metric_name in metric_index:
                    metric_tensor[row_idx, dataset_index[dataset_id], metric_index[metric_name]] = float(metric_value)
        for param_name, param_value in row.params.items():
            if param_name not in parameter_index:
                continue
            param_idx = parameter_index[param_name]
            parameter_numeric_matrix[row_idx, param_idx] = _numeric_param(param_value)
            parameter_text_matrix[row_idx, param_idx] = _stringify_param(param_value)

    if not alpha_family_summary:
        if metric_phase_dir is not None:
            alpha_source_rows = [
                row
                for row in ordered_candidate_rows
                if row.stage == "metric_phase" and row.family is not None and row.alpha is not None
            ]
            if alpha_source_rows:
                alpha_summary_source = "metric_phase.candidate_rows"
                warnings.append(
                    "Metric-phase alpha report unavailable; synthesized alpha summary from metric-phase candidate rows."
                )
                alpha_family_summary = _build_alpha_family_summary(alpha_source_rows, primary_metric)
            elif isinstance(root_alpha_report_payload, dict):
                alpha_family_summary, alpha_ranked_runs = _normalize_alpha_summary_rows(
                    root_alpha_report_payload,
                    primary_metric=primary_metric,
                )
                alpha_summary_source = "experiment_root.alpha_report"
        elif isinstance(root_alpha_report_payload, dict):
            alpha_family_summary, alpha_ranked_runs = _normalize_alpha_summary_rows(
                root_alpha_report_payload,
                primary_metric=primary_metric,
            )
            alpha_summary_source = "experiment_root.alpha_report"
        elif mode == "legacy":
            alpha_source_rows = [
                row for row in ordered_candidate_rows if row.family is not None and row.alpha is not None
            ]
            if not alpha_source_rows:
                alpha_source_rows = [
                    row for row in alpha_rows_from_records if row.family is not None and row.alpha is not None
                ]
                if alpha_source_rows:
                    alpha_summary_source = "summary_records"
            elif alpha_source_rows:
                alpha_summary_source = "candidate_rows"
            alpha_family_summary = _build_alpha_family_summary(alpha_source_rows, primary_metric)

    diagnostic_availability = _build_diagnostic_availability(run_metric_records)
    provenance = {
        "source_dir": str(source_dir.resolve()),
        "experiment_root": str(experiment_root.resolve()),
        "metric_phase_dir": None if metric_phase_dir is None else str(metric_phase_dir.resolve()),
        "path_scope": path_scope,
        "used_files": sorted(dict.fromkeys(used_files)),
        "mode": mode,
        "configured_stage_mode": configured_stage_mode,
        "primary_metric_source": primary_metric_source,
        "alpha_summary_source": alpha_summary_source,
        "active_generation_id": generation_summary.get("active_generation_id"),
        "previous_generation_id": generation_summary.get("previous_generation_id"),
    }
    experiment_metadata.update(
        {
            "mode": mode,
            "configured_stage_mode": configured_stage_mode,
            "dataset_count": len(datasets),
            "candidate_count": len(ordered_candidate_rows),
            "run_record_count": len(run_metric_records),
            "primary_stage_name": primary_stage_name,
            "primary_metric": primary_metric,
            "primary_metric_source": primary_metric_source,
            "active_generation_id": generation_summary.get("active_generation_id"),
            "previous_generation_id": generation_summary.get("previous_generation_id"),
        }
    )

    return SweepVisualizationModel(
        source_dir=source_dir,
        mode=mode,
        path_scope=path_scope,
        experiment_root=experiment_root,
        metric_phase_dir=metric_phase_dir,
        primary_stage_name=primary_stage_name,
        primary_metric=primary_metric,
        primary_metric_source=primary_metric_source,
        datasets=datasets,
        strategies=strategies,
        metric_names=metric_names,
        displacement_metric_names=displacement_metric_names,
        consistency_metric_names=consistency_metric_names,
        parameter_names=parameter_names,
        stage_order=stage_order,
        stages=stage_summaries,
        stage_candidate_ids=stage_candidate_ids,
        candidates=ordered_candidate_rows,
        run_metric_records=run_metric_records,
        scoring_weights=scoring_weights,
        warnings=warnings,
        provenance=provenance,
        experiment_metadata=experiment_metadata,
        generation_summary=generation_summary,
        previous_generation_candidates=previous_generation_candidates,
        all_generation_candidates=all_generation_candidates,
        all_generation_run_metric_records=all_generation_run_metric_records,
        default_generation_scope=GENERATION_SCOPE_ALL,
        alpha_family_summary=alpha_family_summary,
        alpha_ranked_runs=alpha_ranked_runs,
        diagnostic_availability=diagnostic_availability,
        candidate_ids=candidate_ids,
        score_matrix=score_matrix,
        runtime_matrix=runtime_matrix,
        metric_tensor=metric_tensor,
        parameter_numeric_matrix=parameter_numeric_matrix,
        parameter_text_matrix=parameter_text_matrix,
    )


def load_sweep_visualization_model(path: Path) -> SweepVisualizationModel:
    """Load a legacy or modern sweep folder into a canonical visualization model."""

    return _load_sweep_visualization_model_impl(path)

    output_dir = Path(path)
    if not output_dir.exists():
        raise FileNotFoundError(f"Sweep output directory not found: {output_dir}")

    warnings: List[str] = []
    used_files: List[str] = []

    experiment_manifest, manifest_path = _read_json_optional(
        output_dir / "experiment_manifest.json",
        validator=lambda obj: isinstance(obj, dict),
        warnings=warnings,
    )
    if manifest_path is not None:
        used_files.append(str(manifest_path))
    run_state, run_state_path = _read_json_optional(
        output_dir / "run_state.json",
        validator=lambda obj: isinstance(obj, dict),
        warnings=warnings,
    )
    if run_state_path is not None:
        used_files.append(str(run_state_path))

    modern_artifacts_present = any(
        _candidate_path_exists(output_dir / filename)
        for filename in ("stage1_summary.json", "stage2_summary.json", "leaderboard.json", "run_state.json")
    )

    scoring_weights: Dict[str, float] = {}
    experiment_metadata: Dict[str, object] = {"output_dir": str(output_dir.resolve())}
    datasets: List[str] = []
    strategies: List[str] = []

    if isinstance(experiment_manifest, dict):
        experiment_metadata.update(
            {
                "generated_at": experiment_manifest.get("generated_at"),
                "strategies": experiment_manifest.get("strategies"),
                "stages": experiment_manifest.get("stages"),
                "scoring": experiment_manifest.get("scoring"),
                "reporting": experiment_manifest.get("reporting"),
            }
        )
        strategies = [str(item) for item in cast(List[object], experiment_manifest.get("strategies", []))]
        scoring = cast(Dict[str, object], experiment_manifest.get("scoring", {}))
        scoring_weights = _mapping_to_float_dict(scoring.get("metric_weights", {}))
        datasets = [
            str(item.get("dataset_id"))
            for item in cast(List[Dict[str, object]], experiment_manifest.get("datasets", []))
            if isinstance(item, dict) and item.get("dataset_id") is not None
        ]

    if isinstance(run_state, dict):
        experiment_metadata.update(
            {
                "created_at": run_state.get("created_at"),
                "updated_at": run_state.get("updated_at"),
                "experiment_fingerprint": run_state.get("experiment_fingerprint"),
            }
        )
        fingerprint_payload = cast(Dict[str, object], run_state.get("fingerprint_payload", {}))
        spec_payload = cast(Dict[str, object], fingerprint_payload.get("spec", {}))
        if not scoring_weights:
            scoring = cast(Dict[str, object], spec_payload.get("scoring", {}))
            scoring_weights = _mapping_to_float_dict(scoring.get("metric_weights", {}))
        if not strategies:
            strategies = [str(item) for item in cast(List[object], spec_payload.get("strategies", []))]
        if not datasets:
            datasets = [
                str(item.get("dataset_id"))
                for item in cast(List[Dict[str, object]], fingerprint_payload.get("datasets", []))
                if isinstance(item, dict) and item.get("dataset_id") is not None
            ]
        if spec_payload:
            experiment_metadata["spec"] = spec_payload

    primary_metric = next(iter(scoring_weights.keys()), DEFAULT_PRIMARY_METRIC)
    stage_summaries: Dict[str, StageVisualizationSummary] = {}
    stage_candidate_ids: Dict[str, List[str]] = {}
    candidate_rows: List[CandidateVisualizationRow] = []
    alpha_rows_from_records: List[CandidateVisualizationRow] = []

    if modern_artifacts_present:
        metric_output_dir = output_dir / "metric_experiments"
        summary_locations = {
            "stage1": output_dir / "stage1_summary.json",
            "stage2": output_dir / "stage2_summary.json",
            "metric_phase": metric_output_dir / "metric_phase_summary.json",
        }
        summary_payloads: Dict[str, Optional[Dict[str, object]]] = {}
        summary_paths: Dict[str, Optional[Path]] = {}
        for stage_name, summary_path in summary_locations.items():
            payload, used_path = _read_json_optional(
                summary_path,
                validator=lambda obj: isinstance(obj, dict),
                warnings=warnings,
            )
            summary_payloads[stage_name] = cast(Optional[Dict[str, object]], payload)
            summary_paths[stage_name] = used_path
            if used_path is not None:
                used_files.append(str(used_path))

        raw_summary_payload, raw_summary_path = _read_json_optional(
            output_dir / "summary.json",
            validator=lambda obj: isinstance(obj, list),
            warnings=warnings,
        )
        if raw_summary_path is not None:
            used_files.append(str(raw_summary_path))
        alpha_rows_from_records.extend(
            _collect_metric_records_from_summary(
                cast(Optional[List[Dict[str, object]]], raw_summary_payload),
                primary_metric=primary_metric,
            )
        )

        metric_summary_payload, metric_summary_path = _read_json_optional(
            metric_output_dir / "summary.json",
            validator=lambda obj: isinstance(obj, list),
            warnings=warnings,
        )
        if metric_summary_path is not None:
            used_files.append(str(metric_summary_path))
        alpha_rows_from_records.extend(
            _collect_metric_records_from_summary(
                cast(Optional[List[Dict[str, object]]], metric_summary_payload),
                primary_metric=primary_metric,
            )
        )

        for stage_name in ("stage1", "stage2", "metric_phase"):
            base_dir = metric_output_dir if stage_name == "metric_phase" else output_dir
            summary_payload = summary_payloads.get(stage_name)
            if summary_payload is None and not (
                isinstance(run_state, dict)
                and stage_name in cast(Dict[str, object], run_state.get("stages", {}))
            ):
                continue

            stage_summary = _stage_progress_from_sources(
                stage_name=stage_name,
                summary_payload=summary_payload,
                run_state=cast(Optional[Dict[str, object]], run_state),
                source_path=summary_paths.get(stage_name),
            )
            stage_summaries[stage_name] = stage_summary

            rankings = []
            if isinstance(summary_payload, dict):
                rankings = [
                    cast(Dict[str, object], item)
                    for item in cast(List[object], summary_payload.get("rankings", []))
                    if isinstance(item, dict)
                ]

            stage_rows = [
                _candidate_row_from_ranking(
                    stage_name=stage_name,
                    ranking=ranking,
                    base_dir=base_dir,
                    rank_fallback=rank,
                    primary_metric=primary_metric,
                )
                for rank, ranking in enumerate(rankings, start=1)
            ]
            stage_rows.sort(key=lambda row: (row.rank, row.candidate_id))
            stage_candidate_ids[stage_name] = [row.row_id for row in stage_rows]
            candidate_rows.extend(stage_rows)

        if "stage1" not in stage_summaries and "stage2" not in stage_summaries:
            raise ValueError(
                f"No readable stage summaries found under {output_dir}; cannot build visualization model."
            )
        stage2_present = bool(summary_payloads.get("stage2"))
        mode = "spec_full" if stage2_present else "spec_partial"
    else:
        legacy_records = _load_legacy_records(output_dir, warnings)
        if not legacy_records:
            raise ValueError(f"No readable sweep artifacts found under {output_dir}.")

        legacy_rows = [
            _candidate_row_from_legacy_record(
                output_dir=output_dir,
                record=record,
                rank=rank,
                primary_metric=primary_metric,
            )
            for rank, record in enumerate(legacy_records, start=1)
        ]
        legacy_rows.sort(
            key=lambda row: (
                -(row.ranking_metric_value if row.ranking_metric_value is not None else float("-inf")),
                row.candidate_id,
            )
        )
        for rank, row in enumerate(legacy_rows, start=1):
            row.rank = rank
        candidate_rows.extend(legacy_rows)
        stage_candidate_ids["legacy"] = [row.row_id for row in legacy_rows]
        stage_summaries["legacy"] = StageVisualizationSummary(
            name="legacy",
            num_runs_planned=len(legacy_rows),
            num_runs_recorded=len(legacy_rows),
            num_successful=len(legacy_rows),
            num_failed=0,
            num_pending=0,
            num_running=0,
            skipped_valid_count=0,
            recomputed_count=0,
            rankings_count=len(legacy_rows),
            execution={},
            failed_runs=[],
            source_path=str(output_dir / "summary.json"),
        )
        alpha_rows_from_records.extend(legacy_rows)
        mode = "legacy"

    if not datasets:
        datasets = _sorted_unique(
            dataset_id
            for row in candidate_rows
            for dataset_id in list(row.dataset_scores.keys()) + list(row.dataset_metrics.keys())
        )
    if not datasets and candidate_rows:
        datasets = ["aggregate"]
    if not strategies:
        strategies = _sorted_unique(row.strategy for row in candidate_rows)

    parameter_names = _sorted_unique(
        param_name for row in candidate_rows for param_name in row.params.keys()
    )
    metric_names = _order_metric_names(
        (metric_name for row in candidate_rows for metric_name in row.metric_means.keys()),
        scoring_weights=scoring_weights,
    )
    stage_order = [stage_name for stage_name in STAGE_ORDER if stage_name in stage_summaries]
    if not stage_order:
        stage_order = [candidate_rows[0].stage] if candidate_rows else []
    primary_stage_name = "metric_phase" if "metric_phase" in stage_summaries else stage_order[-1]

    ordered_candidate_rows: List[CandidateVisualizationRow] = []
    for stage_name in stage_order:
        rows = [row for row in candidate_rows if row.stage == stage_name]
        rows.sort(key=lambda row: (row.rank, row.candidate_id))
        ordered_candidate_rows.extend(rows)

    candidate_ids = [row.row_id for row in ordered_candidate_rows]
    dataset_index = {dataset_id: idx for idx, dataset_id in enumerate(datasets)}
    metric_index = {metric_name: idx for idx, metric_name in enumerate(metric_names)}
    parameter_index = {param_name: idx for idx, param_name in enumerate(parameter_names)}
    score_matrix = np.full((len(ordered_candidate_rows), len(datasets)), np.nan, dtype=np.float64)
    runtime_matrix = np.full_like(score_matrix, np.nan)
    metric_tensor = np.full(
        (len(ordered_candidate_rows), len(datasets), len(metric_names)),
        np.nan,
        dtype=np.float64,
    )
    parameter_numeric_matrix = np.full(
        (len(ordered_candidate_rows), len(parameter_names)),
        np.nan,
        dtype=np.float64,
    )
    parameter_text_matrix = np.empty(
        (len(ordered_candidate_rows), len(parameter_names)),
        dtype=object,
    )
    parameter_text_matrix[:] = ""

    for row_idx, row in enumerate(ordered_candidate_rows):
        for dataset_id, score in row.dataset_scores.items():
            if dataset_id in dataset_index:
                score_matrix[row_idx, dataset_index[dataset_id]] = float(score)
        for dataset_id, runtime in row.dataset_runtimes.items():
            if dataset_id in dataset_index:
                runtime_matrix[row_idx, dataset_index[dataset_id]] = float(runtime)
        for dataset_id, metrics in row.dataset_metrics.items():
            if dataset_id not in dataset_index:
                continue
            for metric_name, metric_value in metrics.items():
                if metric_name in metric_index:
                    metric_tensor[row_idx, dataset_index[dataset_id], metric_index[metric_name]] = float(metric_value)
        for param_name, param_value in row.params.items():
            if param_name not in parameter_index:
                continue
            param_idx = parameter_index[param_name]
            parameter_numeric_matrix[row_idx, param_idx] = _numeric_param(param_value)
            parameter_text_matrix[row_idx, param_idx] = _stringify_param(param_value)

    alpha_source_rows = [row for row in ordered_candidate_rows if row.family is not None and row.alpha is not None]
    if not alpha_source_rows:
        alpha_source_rows = [
            row for row in alpha_rows_from_records if row.family is not None and row.alpha is not None
        ]
    alpha_family_summary = _build_alpha_family_summary(alpha_source_rows, primary_metric)

    provenance = {
        "source_dir": str(output_dir.resolve()),
        "used_files": sorted(dict.fromkeys(used_files)),
        "mode": mode,
    }
    experiment_metadata.update(
        {
            "mode": mode,
            "dataset_count": len(datasets),
            "candidate_count": len(ordered_candidate_rows),
            "primary_stage_name": primary_stage_name,
            "primary_metric": primary_metric,
        }
    )

    return SweepVisualizationModel(
        source_dir=output_dir,
        mode=mode,
        primary_stage_name=primary_stage_name,
        primary_metric=primary_metric,
        datasets=datasets,
        strategies=strategies,
        metric_names=metric_names,
        parameter_names=parameter_names,
        stage_order=stage_order,
        stages=stage_summaries,
        stage_candidate_ids=stage_candidate_ids,
        candidates=ordered_candidate_rows,
        scoring_weights=scoring_weights,
        warnings=warnings,
        provenance=provenance,
        experiment_metadata=experiment_metadata,
        alpha_family_summary=alpha_family_summary,
        candidate_ids=candidate_ids,
        score_matrix=score_matrix,
        runtime_matrix=runtime_matrix,
        metric_tensor=metric_tensor,
        parameter_numeric_matrix=parameter_numeric_matrix,
        parameter_text_matrix=parameter_text_matrix,
    )


def generation_scope_options(model: SweepVisualizationModel) -> List[str]:
    generation_ids = [
        str(generation_id)
        for generation_id in cast(List[object], model.generation_summary.get("ordered_generation_ids", []))
        if _coerce_optional_str(generation_id) is not None
    ]
    active_generation_id = _coerce_optional_str(model.generation_summary.get("active_generation_id"))
    if active_generation_id and active_generation_id not in generation_ids:
        generation_ids.append(active_generation_id)
    values = [GENERATION_SCOPE_ALL, GENERATION_SCOPE_ACTIVE]
    values.extend(generation_id for generation_id in generation_ids if generation_id not in values)
    return values


def _normalize_generation_scope(model: SweepVisualizationModel, scope: Optional[str]) -> str:
    if not scope:
        return model.default_generation_scope
    if scope in {GENERATION_SCOPE_ALL, GENERATION_SCOPE_ACTIVE}:
        return scope
    return str(scope)


def _rows_matching_generation_scope(
    rows: Sequence[Any],
    *,
    active_rows: Sequence[Any],
    scope: str,
    active_generation_id: Optional[str],
) -> List[Any]:
    if scope == GENERATION_SCOPE_ALL:
        return list(rows)
    if scope == GENERATION_SCOPE_ACTIVE:
        return list(active_rows)
    return [row for row in rows if getattr(row, "generation_id", None) == scope]


def _sort_key_for_scope_rank(row: CandidateVisualizationRow) -> Tuple[float, int, str]:
    score = row.score
    if score is None or not math.isfinite(float(score)):
        score_value = -float("inf")
    else:
        score_value = float(score)
    return (-score_value, row.rank, row.row_id)


def _recompute_scoped_candidate_ranks(
    rows: Sequence[CandidateVisualizationRow],
) -> List[CandidateVisualizationRow]:
    grouped: Dict[str, List[CandidateVisualizationRow]] = {}
    for row in rows:
        grouped.setdefault(row.stage, []).append(row)
    result: List[CandidateVisualizationRow] = []
    for stage_name in [stage for stage in STAGE_ORDER if stage in grouped] + sorted(
        stage for stage in grouped if stage not in STAGE_ORDER
    ):
        stage_rows = sorted(grouped[stage_name], key=_sort_key_for_scope_rank)
        for rank, row in enumerate(stage_rows, start=1):
            result.append(replace(row, rank=rank))
    return result


def candidate_rows_for_generation_scope(
    model: SweepVisualizationModel,
    scope: Optional[str],
) -> List[CandidateVisualizationRow]:
    normalized_scope = _normalize_generation_scope(model, scope)
    active_generation_id = _coerce_optional_str(model.generation_summary.get("active_generation_id"))
    rows = _rows_matching_generation_scope(
        model.all_generation_candidates,
        active_rows=model.candidates,
        scope=normalized_scope,
        active_generation_id=active_generation_id,
    )
    return _recompute_scoped_candidate_ranks(cast(List[CandidateVisualizationRow], rows))


def run_records_for_generation_scope(
    model: SweepVisualizationModel,
    scope: Optional[str],
) -> List[RunMetricRecord]:
    normalized_scope = _normalize_generation_scope(model, scope)
    active_generation_id = _coerce_optional_str(model.generation_summary.get("active_generation_id"))
    return cast(
        List[RunMetricRecord],
        _rows_matching_generation_scope(
            model.all_generation_run_metric_records,
            active_rows=model.run_metric_records,
            scope=normalized_scope,
            active_generation_id=active_generation_id,
        ),
    )


def _rows_for_stage(model: SweepVisualizationModel, stage_name: Optional[str]) -> List[CandidateVisualizationRow]:
    target_stage = stage_name or model.primary_stage_name
    return [
        row
        for row in candidate_rows_for_generation_scope(model, model.default_generation_scope)
        if row.stage == target_stage
    ]


@dataclass
class EffectRow:
    source_kind: str
    stage: str
    candidate_id: str
    dataset_id: Optional[str]
    strategy: str
    family: Optional[str]
    alpha: Optional[float]
    params: Dict[str, object]
    score: Optional[float]
    mean_dataset_score: Optional[float]
    mean_runtime_seconds: Optional[float]
    metric_means: Dict[str, float]
    displacement_field_metrics: Dict[str, float]
    inter_view_consistency_metrics: Dict[str, float]


def _alpha_filter_value(alpha: Optional[float]) -> str:
    return "All" if alpha is None else f"{alpha:g}"


def _value_options(model: SweepVisualizationModel) -> List[str]:
    options = ["score", "mean_dataset_score", "mean_runtime_seconds"]
    options.extend(f"metric:{metric_name}" for metric_name in model.metric_names)
    options.extend(f"displacement:{metric_name}" for metric_name in model.displacement_metric_names)
    options.extend(f"consistency:{metric_name}" for metric_name in model.consistency_metric_names)
    return options


def _row_value_for_option(row: CandidateVisualizationRow, option: str) -> Optional[float]:
    if option == "score":
        return row.score
    if option == "mean_dataset_score":
        return row.mean_dataset_score
    if option == "mean_runtime_seconds":
        return row.mean_runtime_seconds
    if option.startswith("metric:"):
        return row.metric_means.get(option.split(":", 1)[1])
    if option.startswith("displacement:"):
        return row.displacement_field_metrics.get(option.split(":", 1)[1])
    if option.startswith("consistency:"):
        return row.inter_view_consistency_metrics.get(option.split(":", 1)[1])
    return None


def _effect_row_value_for_option(row: EffectRow, option: str) -> Optional[float]:
    if option == "score":
        return row.score
    if option == "mean_dataset_score":
        return row.mean_dataset_score
    if option == "mean_runtime_seconds":
        return row.mean_runtime_seconds
    if option.startswith("metric:"):
        return row.metric_means.get(option.split(":", 1)[1])
    if option.startswith("displacement:"):
        return row.displacement_field_metrics.get(option.split(":", 1)[1])
    if option.startswith("consistency:"):
        return row.inter_view_consistency_metrics.get(option.split(":", 1)[1])
    return None


def _candidate_rows_to_effect_rows(rows: Sequence[CandidateVisualizationRow]) -> List[EffectRow]:
    return [
        EffectRow(
            source_kind="candidate",
            stage=row.stage,
            candidate_id=row.candidate_id,
            dataset_id=None,
            strategy=row.strategy,
            family=row.family,
            alpha=row.alpha,
            params=row.params,
            score=row.score,
            mean_dataset_score=row.mean_dataset_score,
            mean_runtime_seconds=row.mean_runtime_seconds,
            metric_means=row.metric_means,
            displacement_field_metrics=row.displacement_field_metrics,
            inter_view_consistency_metrics=row.inter_view_consistency_metrics,
        )
        for row in rows
    ]


def _run_records_to_effect_rows(records: Sequence[RunMetricRecord]) -> List[EffectRow]:
    return [
        EffectRow(
            source_kind="run",
            stage=record.stage,
            candidate_id=record.candidate_id,
            dataset_id=record.dataset_id,
            strategy=record.strategy,
            family=record.family,
            alpha=record.alpha,
            params=record.params,
            score=record.score,
            mean_dataset_score=record.mean_dataset_score,
            mean_runtime_seconds=record.mean_runtime_seconds,
            metric_means=record.metric_means,
            displacement_field_metrics=record.displacement_field_metrics,
            inter_view_consistency_metrics=record.inter_view_consistency_metrics,
        )
        for record in records
    ]


def _filter_effect_rows(
    rows: Sequence[EffectRow],
    *,
    stage_name: Optional[str],
    strategy: str,
    family: str,
    alpha: str,
    dataset_scope: str,
    parameter_filters: Dict[str, str],
    ignored_params: Sequence[str] = (),
) -> List[EffectRow]:
    filtered: List[EffectRow] = []
    ignored = set(ignored_params)
    for row in rows:
        if stage_name and stage_name != "All" and row.stage != stage_name:
            continue
        if strategy != "All" and row.strategy != strategy:
            continue
        if family != "All" and row.family != family:
            continue
        if alpha != "All" and _alpha_filter_value(row.alpha) != alpha:
            continue
        if dataset_scope != "aggregate":
            if row.dataset_id != dataset_scope:
                continue
        elif row.dataset_id is not None and row.source_kind == "run":
            continue
        matches = True
        for parameter_name, filter_value in parameter_filters.items():
            if parameter_name in ignored or filter_value == "All":
                continue
            if _stringify_param(row.params.get(parameter_name)) != filter_value:
                matches = False
                break
        if matches:
            filtered.append(row)
    return filtered


def _stage_parameter_rankings(
    model: SweepVisualizationModel,
    stage_name: Optional[str] = None,
) -> List[str]:
    sensitivity_rows = compute_parameter_sensitivity(
        model,
        stage_name=stage_name or model.primary_stage_name,
        dataset_scope="aggregate",
        value_option="score",
        filters={},
    )
    sensitivity_order = [
        row.parameter_name
        for row in sensitivity_rows
        if row.num_values > 1 and row.impact_score > 0
    ]
    if sensitivity_order:
        return sensitivity_order
    rows = _rows_for_stage(model, stage_name or model.primary_stage_name)
    counts: List[Tuple[int, str]] = []
    for parameter_name in model.parameter_names:
        distinct_values = {
            _stringify_param(row.params.get(parameter_name))
            for row in rows
            if parameter_name in row.params
        }
        if len(distinct_values) > 1:
            counts.append((len(distinct_values), parameter_name))
    counts.sort(key=lambda item: (-item[0], item[1]))
    ordered = [parameter_name for _, parameter_name in counts]
    if ordered:
        return ordered
    return list(model.parameter_names)


def _plot_progress_bars(ax: Any, model: SweepVisualizationModel) -> None:
    stage_names = model.stage_order
    planned = [model.stages[stage_name].num_runs_planned for stage_name in stage_names]
    successful = [model.stages[stage_name].num_successful for stage_name in stage_names]
    failed = [model.stages[stage_name].num_failed for stage_name in stage_names]
    pending = [model.stages[stage_name].num_pending for stage_name in stage_names]
    y_positions = np.arange(len(stage_names))

    ax.barh(y_positions, successful, color="#4f8a8b", label="successful")
    ax.barh(y_positions, failed, left=successful, color="#d95d39", label="failed")
    ax.barh(
        y_positions,
        pending,
        left=np.array(successful) + np.array(failed),
        color="#d9c26c",
        label="pending",
    )
    for idx, planned_runs in enumerate(planned):
        ax.text(planned_runs + 0.3, idx, f"planned {planned_runs}", va="center", fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_names)
    ax.set_xlabel("Runs")
    ax.set_title("Stage Progress")
    ax.legend(loc="best")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)


def create_overview_figure(model: SweepVisualizationModel) -> Figure:
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    _plot_progress_bars(ax, model)
    fig.tight_layout()
    return fig


def create_effects_figure(
    model: SweepVisualizationModel,
    *,
    mode: str = "1D Trend",
    stage_name: Optional[str] = None,
    dataset_scope: str = "aggregate",
    x_parameter_name: Optional[str] = None,
    y_parameter_name: Optional[str] = None,
    value_option: Optional[str] = None,
    strategy_filter: str = "All",
    family_filter: str = "All",
    alpha_filter: str = "All",
    parameter_filters: Optional[Dict[str, str]] = None,
) -> Figure:
    preferred_params = _stage_parameter_rankings(model, stage_name or model.primary_stage_name)
    x_parameter = x_parameter_name or (preferred_params[0] if preferred_params else None)
    y_parameter = y_parameter_name or (
        preferred_params[1] if len(preferred_params) > 1 else (preferred_params[0] if preferred_params else None)
    )
    option = value_option or ("score" if "score" in _value_options(model) else _value_options(model)[0])
    filters = dict(parameter_filters or {})

    base_rows = (
        _candidate_rows_to_effect_rows(candidate_rows_for_generation_scope(model, model.default_generation_scope))
        if dataset_scope == "aggregate"
        else _run_records_to_effect_rows(run_records_for_generation_scope(model, model.default_generation_scope))
    )
    filtered_rows = _filter_effect_rows(
        base_rows,
        stage_name=stage_name or model.primary_stage_name,
        strategy=strategy_filter,
        family=family_filter,
        alpha=alpha_filter,
        dataset_scope=dataset_scope,
        parameter_filters=filters,
        ignored_params=[param for param in (x_parameter, y_parameter) if param],
    )

    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if not filtered_rows:
        ax.text(0.5, 0.5, "No rows match the current effects filters.", ha="center", va="center")
        ax.axis("off")
        return fig
    if x_parameter is None:
        ax.text(0.5, 0.5, "No parameter metadata available.", ha="center", va="center")
        ax.axis("off")
        return fig

    if mode == "2D Heatmap":
        if y_parameter is None:
            ax.text(0.5, 0.5, "Choose two parameters for the heatmap view.", ha="center", va="center")
            ax.axis("off")
            return fig
        cell_values: Dict[Tuple[str, str], List[float]] = {}
        x_numeric: Dict[str, float] = {}
        y_numeric: Dict[str, float] = {}
        for row in filtered_rows:
            value = _effect_row_value_for_option(row, option)
            if value is None or not math.isfinite(value):
                continue
            x_label = _stringify_param(row.params.get(x_parameter))
            y_label = _stringify_param(row.params.get(y_parameter))
            cell_values.setdefault((x_label, y_label), []).append(float(value))
            x_numeric[x_label] = _numeric_param(row.params.get(x_parameter))
            y_numeric[y_label] = _numeric_param(row.params.get(y_parameter))
        if not cell_values:
            ax.text(0.5, 0.5, f"No finite values for '{option}'.", ha="center", va="center")
            ax.axis("off")
            return fig
        x_labels = sorted(cell_values.keys(), key=lambda item: (item[0], item[1]))
        unique_x = sorted(
            {item[0] for item in x_labels},
            key=lambda label: (not math.isfinite(x_numeric.get(label, float("nan"))), x_numeric.get(label, float("nan")), label),
        )
        unique_y = sorted(
            {item[1] for item in x_labels},
            key=lambda label: (not math.isfinite(y_numeric.get(label, float("nan"))), y_numeric.get(label, float("nan")), label),
        )
        matrix = np.full((len(unique_y), len(unique_x)), np.nan, dtype=np.float64)
        counts = np.zeros_like(matrix, dtype=np.int32)
        x_lookup = {label: idx for idx, label in enumerate(unique_x)}
        y_lookup = {label: idx for idx, label in enumerate(unique_y)}
        for (x_label, y_label), values in cell_values.items():
            matrix[y_lookup[y_label], x_lookup[x_label]] = float(np.mean(values))
            counts[y_lookup[y_label], x_lookup[x_label]] = len(values)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(unique_x)))
        ax.set_xticklabels(unique_x, rotation=25, ha="right")
        ax.set_yticks(range(len(unique_y)))
        ax.set_yticklabels(unique_y)
        ax.set_xlabel(x_parameter)
        ax.set_ylabel(y_parameter)
        ax.set_title(f"{dataset_scope}: {option} heatmap")
        for y_idx in range(len(unique_y)):
            for x_idx in range(len(unique_x)):
                if counts[y_idx, x_idx] > 0:
                    ax.text(x_idx, y_idx, str(int(counts[y_idx, x_idx])), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    grouped_values: Dict[str, List[float]] = {}
    grouped_numeric: Dict[str, float] = {}
    for row in filtered_rows:
        value = _effect_row_value_for_option(row, option)
        if value is None or not math.isfinite(value):
            continue
        label = _stringify_param(row.params.get(x_parameter))
        grouped_values.setdefault(label, []).append(float(value))
        grouped_numeric[label] = _numeric_param(row.params.get(x_parameter))
    if not grouped_values:
        ax.text(0.5, 0.5, f"No finite values for '{option}'.", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = sorted(
        grouped_values.keys(),
        key=lambda label: (
            not math.isfinite(grouped_numeric.get(label, float("nan"))),
            grouped_numeric.get(label, float("nan")),
            label,
        ),
    )
    means = np.asarray([float(np.mean(grouped_values[label])) for label in labels], dtype=np.float64)
    mins = np.asarray([float(np.min(grouped_values[label])) for label in labels], dtype=np.float64)
    maxs = np.asarray([float(np.max(grouped_values[label])) for label in labels], dtype=np.float64)
    counts = np.asarray([len(grouped_values[label]) for label in labels], dtype=np.int32)
    x_numeric = np.asarray([grouped_numeric[label] for label in labels], dtype=np.float64)

    if np.all(np.isfinite(x_numeric)):
        ax.plot(x_numeric, means, marker="o", color="#2a6f97")
        ax.fill_between(x_numeric, mins, maxs, color="#2a6f97", alpha=0.18)
        ax.set_xticks(x_numeric)
    else:
        x_positions = np.arange(len(labels))
        ax.plot(x_positions, means, marker="o", color="#2a6f97")
        ax.fill_between(x_positions, mins, maxs, color="#2a6f97", alpha=0.18)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    for idx, count in enumerate(counts):
        x_value = x_numeric[idx] if np.all(np.isfinite(x_numeric)) else idx
        ax.annotate(f"n={int(count)}", (x_value, means[idx]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel(x_parameter)
    ax.set_ylabel(option.replace("metric:", "").replace("displacement:", "").replace("consistency:", ""))
    ax.set_title(f"{dataset_scope}: {option} vs {x_parameter}")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def create_coverage_figure(
    model: SweepVisualizationModel,
    *,
    stage_name: Optional[str] = None,
    parameter_name: Optional[str] = None,
    value_option: Optional[str] = None,
) -> Figure:
    return create_effects_figure(
        model,
        mode="1D Trend",
        stage_name=stage_name,
        dataset_scope="aggregate",
        x_parameter_name=parameter_name,
        value_option=value_option,
        strategy_filter="All",
        family_filter="All",
        alpha_filter="All",
        parameter_filters={},
    )


def _progression_summary_lines(model: SweepVisualizationModel) -> List[str]:
    lines: List[str] = []
    if "stage1" not in model.stages:
        lines.append("Progression view is only available for staged experiment outputs.")
        return lines
    if "stage2" not in model.stages:
        if model.mode == "spec_single_stage":
            lines.append("Single-stage baseline run; no stage-2 progression comparison.")
        else:
            lines.append("Progression view is only available for staged experiment outputs.")
        return lines

    stage1_rows = _rows_for_stage(model, "stage1")
    stage2_rows = _rows_for_stage(model, "stage2")
    if not stage1_rows:
        lines.append("No ranked stage 1 candidates were loaded.")
        return lines

    lines.append(
        f"Stage 1 ranked {len(stage1_rows)} candidates; stage 2 evaluated {len(stage2_rows)} refinement candidates."
    )
    if stage2_rows:
        best_stage1 = stage1_rows[0]
        best_stage2 = stage2_rows[0]
        if best_stage1.score is not None and best_stage2.score is not None:
            delta = best_stage2.score - best_stage1.score
            lines.append(
                f"Best-score delta from stage 1 to stage 2: {delta:+.4f} "
                f"({best_stage1.candidate_id} -> {best_stage2.candidate_id})."
            )
    else:
        stage2_summary = model.stages["stage2"]
        if stage2_summary.num_runs_planned > 0:
            lines.append(
                f"Stage 2 is planned but incomplete: {stage2_summary.num_pending} pending / "
                f"{stage2_summary.num_running} running of {stage2_summary.num_runs_planned} planned runs."
            )

    shortlist_size = min(5, len(stage1_rows))
    shortlist = stage1_rows[:shortlist_size]
    for parameter_name in model.parameter_names[:12]:
        stage1_values = sorted(
            {
                _stringify_param(row.params.get(parameter_name))
                for row in shortlist
                if parameter_name in row.params
            }
        )
        stage2_values = sorted(
            {
                _stringify_param(row.params.get(parameter_name))
                for row in stage2_rows
                if parameter_name in row.params
            }
        )
        if not stage1_values and not stage2_values:
            continue
        if stage1_values == stage2_values:
            continue
        lines.append(
            f"{parameter_name}: stage1 shortlist {stage1_values or ['n/a']} -> "
            f"stage2 {stage2_values or ['n/a']}"
        )
    return lines


def create_progression_figure(model: SweepVisualizationModel) -> Figure:
    fig = Figure(figsize=(11, 7))
    top_ax = fig.add_subplot(211)
    bottom_ax = fig.add_subplot(212)

    if "stage1" not in model.stages and "stage2" not in model.stages:
        top_ax.text(0.5, 0.5, "Legacy sweeps have no staged progression.", ha="center", va="center")
        top_ax.axis("off")
        _plot_progress_bars(bottom_ax, model)
        fig.tight_layout()
        return fig
    if "stage2" not in model.stages:
        message = (
            "Single-stage baseline run; no stage-2 progression comparison."
            if model.mode == "spec_single_stage"
            else "Progression view is only available for staged experiment outputs."
        )
        top_ax.text(0.5, 0.5, message, ha="center", va="center")
        top_ax.axis("off")
        _plot_progress_bars(bottom_ax, model)
        fig.tight_layout()
        return fig

    stage1_rows = _rows_for_stage(model, "stage1")[:5]
    stage2_rows = _rows_for_stage(model, "stage2")[:5]
    labels = [row.candidate_id for row in stage1_rows]
    labels2 = [row.candidate_id for row in stage2_rows]
    if stage1_rows:
        top_ax.bar(
            np.arange(len(stage1_rows)) - 0.2,
            [row.score if row.score is not None else np.nan for row in stage1_rows],
            width=0.4,
            label="stage1",
            color="#5c80bc",
        )
    if stage2_rows:
        top_ax.bar(
            np.arange(len(stage2_rows)) + 0.2,
            [row.score if row.score is not None else np.nan for row in stage2_rows],
            width=0.4,
            label="stage2",
            color="#9bc53d",
        )
    combined_labels = labels if len(labels) >= len(labels2) else labels2
    top_ax.set_xticks(range(len(combined_labels)))
    top_ax.set_xticklabels(combined_labels, rotation=20, ha="right")
    top_ax.set_ylabel("score")
    top_ax.set_title("Top Candidates Across Stage 1 and Stage 2")
    top_ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    top_ax.legend(loc="best")

    lines = _progression_summary_lines(model)
    bottom_ax.axis("off")
    bottom_ax.text(0.01, 0.99, "\n".join(lines[:14]), va="top", ha="left", family="monospace")
    fig.tight_layout()
    return fig


def _weighted_metric_contributions(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
) -> Tuple[List[str], np.ndarray]:
    if not model.scoring_weights:
        return [], np.zeros((0, 0), dtype=np.float64)

    metric_names = [name for name in model.scoring_weights.keys() if name in model.metric_names]
    if not metric_names:
        return [], np.zeros((0, 0), dtype=np.float64)

    metric_matrix = np.full((len(rows), len(metric_names)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for metric_idx, metric_name in enumerate(metric_names):
            metric_matrix[row_idx, metric_idx] = row.metric_means.get(metric_name, np.nan)

    contributions = np.zeros_like(metric_matrix)
    for metric_idx, metric_name in enumerate(metric_names):
        column = metric_matrix[:, metric_idx]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        min_value = float(np.min(column[finite_mask]))
        max_value = float(np.max(column[finite_mask]))
        if math.isclose(min_value, max_value):
            normalized = np.where(finite_mask, 0.5, np.nan)
        else:
            normalized = (column - min_value) / (max_value - min_value)
        contributions[:, metric_idx] = normalized * float(model.scoring_weights[metric_name])
    return metric_names, contributions


def create_significance_figure(model: SweepVisualizationModel, *, top_n: int = 6) -> Figure:
    rows = _rows_for_stage(model, model.primary_stage_name)[:max(1, top_n)]
    fig = Figure(figsize=(12, 8))
    score_ax = fig.add_subplot(221)
    heatmap_ax = fig.add_subplot(222)
    scatter_ax = fig.add_subplot(223)
    contrib_ax = fig.add_subplot(224)

    if not rows:
        score_ax.text(0.5, 0.5, "No ranked rows available.", ha="center", va="center")
        score_ax.axis("off")
        heatmap_ax.axis("off")
        scatter_ax.axis("off")
        contrib_ax.axis("off")
        fig.tight_layout()
        return fig

    labels = [row.candidate_id for row in rows]
    scores = [row.score if row.score is not None else np.nan for row in rows]
    score_ax.bar(range(len(rows)), scores, color="#3d5a80")
    score_ax.set_xticks(range(len(rows)))
    score_ax.set_xticklabels(labels, rotation=20, ha="right")
    score_ax.set_ylabel("score")
    score_ax.set_title(f"Top {len(rows)} {model.primary_stage_name} candidates")
    score_ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    heatmap_rows = []
    heatmap_labels = []
    for row in rows:
        values = [row.dataset_scores.get(dataset_id, np.nan) for dataset_id in model.datasets]
        if not any(np.isfinite(values)):
            values = [
                row.dataset_metrics.get(dataset_id, {}).get(model.primary_metric, np.nan)
                for dataset_id in model.datasets
            ]
        heatmap_rows.append(values)
        heatmap_labels.append(row.candidate_id)
    heatmap = np.asarray(heatmap_rows, dtype=np.float64)
    if heatmap.size and np.any(np.isfinite(heatmap)):
        image = heatmap_ax.imshow(heatmap, aspect="auto", cmap="viridis")
        heatmap_ax.set_yticks(range(len(heatmap_labels)))
        heatmap_ax.set_yticklabels(heatmap_labels)
        heatmap_ax.set_xticks(range(len(model.datasets)))
        heatmap_ax.set_xticklabels(model.datasets, rotation=25, ha="right")
        heatmap_ax.set_title("Per-dataset quality/score")
        fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)
    else:
        heatmap_ax.text(0.5, 0.5, "No per-dataset score data available.", ha="center", va="center")
        heatmap_ax.axis("off")

    runtime_rows = _rows_for_stage(model, model.primary_stage_name)
    xs = [row.mean_runtime_seconds for row in runtime_rows if row.mean_runtime_seconds is not None and row.score is not None]
    ys = [row.score for row in runtime_rows if row.mean_runtime_seconds is not None and row.score is not None]
    labels_runtime = [row.candidate_id for row in runtime_rows if row.mean_runtime_seconds is not None and row.score is not None]
    if xs and ys:
        scatter_ax.scatter(xs, ys, c="#ee6c4d", alpha=0.8)
        for x_value, y_value, label in zip(xs[:8], ys[:8], labels_runtime[:8]):
            scatter_ax.annotate(label, (x_value, y_value), fontsize=8, alpha=0.8)
        scatter_ax.set_xlabel("mean runtime (s)")
        scatter_ax.set_ylabel("score")
        scatter_ax.set_title("Runtime vs quality tradeoff")
        scatter_ax.grid(True, linestyle=":", alpha=0.4)
    else:
        scatter_ax.text(0.5, 0.5, "Runtime metadata unavailable.", ha="center", va="center")
        scatter_ax.axis("off")

    metric_names, contributions = _weighted_metric_contributions(model, rows)
    if metric_names and contributions.size:
        bottom = np.zeros(len(rows), dtype=np.float64)
        for metric_idx, metric_name in enumerate(metric_names):
            values = contributions[:, metric_idx]
            contrib_ax.bar(labels, values, bottom=bottom, label=metric_name)
            bottom += np.nan_to_num(values, nan=0.0)
        contrib_ax.set_title("Weighted metric contributions")
        contrib_ax.set_ylabel("weighted normalized score")
        contrib_ax.tick_params(axis="x", rotation=20)
        contrib_ax.legend(fontsize=8)
    else:
        contrib_ax.text(0.5, 0.5, "Scoring weights unavailable; showing rank gaps only.", ha="center", va="center")
        contrib_ax.axis("off")

    fig.tight_layout()
    return fig


def create_metric_formulation_figure(model: SweepVisualizationModel) -> Optional[Figure]:
    if not model.alpha_family_summary:
        return None

    fig = Figure(figsize=(12, 8))
    quality_ax = fig.add_subplot(221)
    runtime_ax = fig.add_subplot(222)
    displacement_ax = fig.add_subplot(223)
    consistency_ax = fig.add_subplot(224)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in model.alpha_family_summary:
        grouped.setdefault(str(row["family"]), []).append(dict(row))

    for family, rows in sorted(grouped.items(), key=lambda item: item[0]):
        rows.sort(key=lambda row: float(row["alpha"]))
        xs_quality = [
            float(row["alpha"])
            for row in rows
            if row.get("mean_score") is not None or row.get("primary_metric_mean") is not None
        ]
        ys_quality = [
            float(row["mean_score"])
            if row.get("mean_score") is not None
            else float(row["primary_metric_mean"])
            for row in rows
            if row.get("mean_score") is not None or row.get("primary_metric_mean") is not None
        ]
        if xs_quality:
            quality_ax.plot(xs_quality, ys_quality, marker="o", label=family)

        xs_runtime = [
            float(row["alpha"])
            for row in rows
            if row.get("mean_runtime_seconds") is not None
        ]
        ys_runtime = [
            float(row["mean_runtime_seconds"])
            for row in rows
            if row.get("mean_runtime_seconds") is not None
        ]
        if xs_runtime:
            runtime_ax.plot(xs_runtime, ys_runtime, marker="o", label=family)

        xs_displacement = [
            float(row["alpha"])
            for row in rows
            if row.get("mean_displacement_magnitude") is not None
        ]
        ys_displacement = [
            float(row["mean_displacement_magnitude"])
            for row in rows
            if row.get("mean_displacement_magnitude") is not None
        ]
        if xs_displacement:
            displacement_ax.plot(xs_displacement, ys_displacement, marker="o", label=family)

        xs_consistency = [
            float(row["alpha"])
            for row in rows
            if row.get("mean_view_std") is not None
        ]
        ys_consistency = [
            float(row["mean_view_std"])
            for row in rows
            if row.get("mean_view_std") is not None
        ]
        if xs_consistency:
            consistency_ax.plot(xs_consistency, ys_consistency, marker="o", label=family)

    quality_ax.set_xlabel("alpha (local displacement score blend weight)")
    quality_ax.set_ylabel("mean score / primary metric")
    quality_ax.set_title("Quality vs metric formulation alpha")
    quality_ax.grid(True, linestyle=":", alpha=0.4)
    if quality_ax.lines:
        quality_ax.legend()

    runtime_ax.set_xlabel("alpha (local displacement score blend weight)")
    runtime_ax.set_ylabel("mean runtime (s)")
    runtime_ax.set_title("Runtime vs metric formulation alpha")
    runtime_ax.grid(True, linestyle=":", alpha=0.4)
    if runtime_ax.lines:
        runtime_ax.legend()

    displacement_ax.set_xlabel("alpha (local displacement score blend weight)")
    displacement_ax.set_ylabel("mean displacement magnitude")
    displacement_ax.set_title("Displacement vs metric formulation alpha")
    displacement_ax.grid(True, linestyle=":", alpha=0.4)
    if displacement_ax.lines:
        displacement_ax.legend()

    consistency_ax.set_xlabel("alpha (local displacement score blend weight)")
    consistency_ax.set_ylabel("mean view std")
    consistency_ax.set_title("Consistency vs metric formulation alpha")
    consistency_ax.grid(True, linestyle=":", alpha=0.4)
    if consistency_ax.lines:
        consistency_ax.legend()

    fig.tight_layout()
    return fig


def create_metric_diagnostics_figure(
    record: Optional[RunMetricRecord],
    *,
    component_name: str = "gradient",
) -> Optional[Figure]:
    if record is None:
        return None

    payload, read_warnings = _load_workspace_diagnostics(
        None if record.workspace_path is None else Path(record.workspace_path)
    )
    blended = cast(Optional[np.ndarray], payload.get("pixel_blended_scores"))
    component = cast(Optional[np.ndarray], payload.get(f"pixel_component_score_{component_name}"))
    preview = _load_preview_array(record.preview_path)
    disp_x = cast(Optional[np.ndarray], payload.get("pixel_displacements_x"))
    disp_y = cast(Optional[np.ndarray], payload.get("pixel_displacements_y"))
    magnitude = None
    if isinstance(disp_x, np.ndarray) and isinstance(disp_y, np.ndarray):
        magnitude = np.sqrt(np.square(np.asarray(disp_x, dtype=np.float64)) + np.square(np.asarray(disp_y, dtype=np.float64)))

    fig = Figure(figsize=(12, 8))
    blended_ax = fig.add_subplot(221)
    component_ax = fig.add_subplot(222)
    disp_ax = fig.add_subplot(223)
    text_ax = fig.add_subplot(224)

    def _draw_map(ax: Any, array: Optional[np.ndarray], title: str) -> None:
        if not isinstance(array, np.ndarray):
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.axis("off")
            return
        arr = np.asarray(array, dtype=np.float64)
        while arr.ndim > 2:
            arr = np.nanmean(arr, axis=-1)
        image = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    _draw_map(blended_ax, blended if blended is not None else preview, "Blended score map / preview")
    _draw_map(component_ax, component, f"{component_name} score map")
    _draw_map(disp_ax, magnitude, "Displacement magnitude")

    text_ax.axis("off")
    diagnostics = record.displacement_score_diagnostics
    lines = [
        f"run: {record.run_name}",
        f"stage: {record.stage}",
        f"candidate: {record.candidate_id}",
        f"dataset: {record.dataset_id}",
        f"family: {record.family or 'n/a'}",
        f"alpha: {'n/a' if record.alpha is None else f'{record.alpha:g}'}",
        "",
    ]
    if read_warnings:
        lines.extend(read_warnings[:2])
        lines.append("")
    if record.metric_means:
        lines.append("image metrics:")
        for key in sorted(record.metric_means.keys())[:4]:
            lines.append(f"  {key}: {record.metric_means[key]:.4f}")
    if record.displacement_field_metrics:
        lines.append("")
        lines.append("displacement field:")
        for key in sorted(record.displacement_field_metrics.keys())[:4]:
            lines.append(f"  {key}: {record.displacement_field_metrics[key]:.4f}")
    component_stats = diagnostics.get("component_selected_stats")
    if isinstance(component_stats, dict) and component_name in component_stats:
        stats = cast(Dict[str, float], component_stats[component_name])
        lines.append("")
        lines.append(f"{component_name} selected stats:")
        for key in ("mean", "std", "min", "max"):
            if key in stats:
                lines.append(f"  {key}: {float(stats[key]):.4f}")
    text_ax.text(0.01, 0.99, "\n".join(lines[:20]), va="top", ha="left", family="monospace")
    fig.tight_layout()
    return fig


def _candidate_row_to_export_dict(row: CandidateVisualizationRow) -> Dict[str, object]:
    return {
        "row_id": row.row_id,
        "candidate_id": row.candidate_id,
        "stage": row.stage,
        "display_name": row.display_name,
        "rank": row.rank,
        "strategy": row.strategy,
        "score": row.score,
        "mean_dataset_score": row.mean_dataset_score,
        "std_dataset_score": row.std_dataset_score,
        "ranking_metric_name": row.ranking_metric_name,
        "ranking_metric_value": row.ranking_metric_value,
        "dataset_scores": row.dataset_scores,
        "dataset_runtimes": row.dataset_runtimes,
        "dataset_metrics": row.dataset_metrics,
        "metric_means": row.metric_means,
        "mean_runtime_seconds": row.mean_runtime_seconds,
        "missing_datasets": row.missing_datasets,
        "params": row.params,
        "config": row.config,
        "family": row.family,
        "alpha": row.alpha,
        "formulation_id": row.formulation_id,
        "generation_id": row.generation_id,
        "displacement_field_metrics": row.displacement_field_metrics,
        "inter_view_consistency_metrics": row.inter_view_consistency_metrics,
        "preview_paths": row.preview_paths,
    }


def _run_metric_record_to_export_dict(record: RunMetricRecord) -> Dict[str, object]:
    return {
        "row_id": record.row_id,
        "run_name": record.run_name,
        "stage": record.stage,
        "candidate_id": record.candidate_id,
        "dataset_id": record.dataset_id,
        "strategy": record.strategy,
        "family": record.family,
        "alpha": record.alpha,
        "formulation_id": record.formulation_id,
        "generation_id": record.generation_id,
        "params": record.params,
        "config": record.config,
        "score": record.score,
        "mean_dataset_score": record.mean_dataset_score,
        "mean_runtime_seconds": record.mean_runtime_seconds,
        "metric_means": record.metric_means,
        "displacement_field_metrics": record.displacement_field_metrics,
        "inter_view_consistency_metrics": record.inter_view_consistency_metrics,
        "displacement_score_diagnostics": record.displacement_score_diagnostics,
        "preview_path": record.preview_path,
        "workspace_path": record.workspace_path,
        "metric_record_path": record.metric_record_path,
        "ranking_metric_name": record.ranking_metric_name,
        "ranking_metric_value": record.ranking_metric_value,
    }


def _default_export_effect_parameters(model: SweepVisualizationModel) -> Tuple[Optional[str], Optional[str]]:
    ranked = _stage_parameter_rankings(model, model.primary_stage_name)
    if not ranked:
        return None, None
    if len(ranked) == 1:
        return ranked[0], ranked[0]
    return ranked[0], ranked[1]


def _default_metric_diagnostics_record(model: SweepVisualizationModel) -> Optional[RunMetricRecord]:
    scoped_records = run_records_for_generation_scope(model, model.default_generation_scope)
    scoped_candidates = candidate_rows_for_generation_scope(model, model.default_generation_scope)
    preferred_stage = "metric_phase" if any(record.stage == "metric_phase" for record in scoped_records) else model.primary_stage_name
    candidate_order = {
        (row.stage, row.candidate_id, row.generation_id): idx
        for idx, row in enumerate(scoped_candidates)
    }
    eligible = [
        record
        for record in scoped_records
        if record.stage == preferred_stage and (record.workspace_path or record.displacement_score_diagnostics)
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda record: (
            candidate_order.get((record.stage, record.candidate_id, record.generation_id), len(candidate_order)),
            record.dataset_id,
        )
    )
    return eligible[0]


def _default_component_name(model: SweepVisualizationModel) -> str:
    components = cast(List[str], model.diagnostic_availability.get("components", []))
    return components[0] if components else "gradient"


def _report_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "item"


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _real_datasets(model: SweepVisualizationModel) -> List[str]:
    return list(model.datasets or ["aggregate"])


def _default_comparison_dataset(model: SweepVisualizationModel) -> str:
    datasets = _real_datasets(model)
    return datasets[0] if datasets else "aggregate"


def _candidate_score_for_dataset(
    row: CandidateVisualizationRow,
    dataset_id: str,
    primary_metric: str,
) -> Optional[float]:
    if dataset_id == "aggregate":
        return row.score if row.score is not None else row.mean_dataset_score
    if dataset_id in row.dataset_scores:
        return row.dataset_scores.get(dataset_id)
    metric_value = row.dataset_metrics.get(dataset_id, {}).get(primary_metric)
    return metric_value if metric_value is not None else row.score


def _candidate_runtime_for_dataset(row: CandidateVisualizationRow, dataset_id: str) -> Optional[float]:
    if dataset_id == "aggregate":
        return row.mean_runtime_seconds
    return row.dataset_runtimes.get(dataset_id, row.mean_runtime_seconds)


def _candidate_metric_value_for_dataset(
    row: CandidateVisualizationRow,
    dataset_id: str,
    primary_metric: str,
) -> Tuple[str, Optional[float]]:
    if dataset_id == "aggregate":
        metric_name = row.ranking_metric_name or primary_metric
        metric_value = row.ranking_metric_value
        if metric_value is None:
            metric_value = row.metric_means.get(metric_name, row.metric_means.get(primary_metric))
        return metric_name, metric_value

    metric_name = primary_metric
    metric_value = row.dataset_metrics.get(dataset_id, {}).get(primary_metric)
    if metric_value is None:
        metric_name = row.ranking_metric_name or primary_metric
        metric_value = row.dataset_scores.get(dataset_id)
    return metric_name, metric_value


def _higher_is_better_value_option(value_option: str) -> bool:
    return value_option != "mean_runtime_seconds"


def _candidate_sensitivity_value(
    model: SweepVisualizationModel,
    row: CandidateVisualizationRow,
    *,
    dataset_scope: str,
    value_option: str,
) -> Optional[float]:
    if value_option == "score":
        return _candidate_score_for_dataset(row, dataset_scope, model.primary_metric)
    if value_option == "mean_dataset_score":
        return row.mean_dataset_score
    if value_option == "mean_runtime_seconds":
        return _candidate_runtime_for_dataset(row, dataset_scope)
    if value_option.startswith("metric:"):
        metric_name = value_option.split(":", 1)[1]
        if dataset_scope == "aggregate":
            return row.metric_means.get(metric_name)
        return row.dataset_metrics.get(dataset_scope, {}).get(metric_name)
    if value_option.startswith("displacement:"):
        return row.displacement_field_metrics.get(value_option.split(":", 1)[1])
    if value_option.startswith("consistency:"):
        return row.inter_view_consistency_metrics.get(value_option.split(":", 1)[1])
    return None


def _parse_sensitivity_filters(
    filters: Optional[Dict[str, object]],
) -> Tuple[str, str, str, Dict[str, str]]:
    payload = dict(filters or {})
    strategy_filter = str(payload.pop("strategy", "All") or "All")
    family_filter = str(payload.pop("family", "All") or "All")
    alpha_filter = str(payload.pop("alpha", "All") or "All")
    parameter_filters: Dict[str, str] = {}
    nested = payload.pop("parameters", None)
    if isinstance(nested, dict):
        parameter_filters.update(
            {
                str(key): str(value)
                for key, value in nested.items()
                if str(value or "All") != "All"
            }
        )
    for key, value in payload.items():
        rendered = str(value or "All")
        if rendered != "All":
            parameter_filters[str(key)] = rendered
    return strategy_filter, family_filter, alpha_filter, parameter_filters


def _candidate_rows_for_sensitivity(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
    *,
    stage_name: Optional[str],
    dataset_scope: str,
    value_option: str,
    filters: Optional[Dict[str, object]],
) -> List[Tuple[CandidateVisualizationRow, float]]:
    strategy_filter, family_filter, alpha_filter, parameter_filters = _parse_sensitivity_filters(filters)
    target_stage = stage_name or model.primary_stage_name
    filtered: List[Tuple[CandidateVisualizationRow, float]] = []
    for row in rows:
        if target_stage != "All" and row.stage != target_stage:
            continue
        if strategy_filter != "All" and row.strategy != strategy_filter:
            continue
        if family_filter != "All" and row.family != family_filter:
            continue
        if alpha_filter != "All" and _alpha_filter_value(row.alpha) != alpha_filter:
            continue
        filter_match = True
        for parameter_name, expected_value in parameter_filters.items():
            if _stringify_param(row.params.get(parameter_name)) != expected_value:
                filter_match = False
                break
        if not filter_match:
            continue
        value = _candidate_sensitivity_value(
            model,
            row,
            dataset_scope=dataset_scope,
            value_option=value_option,
        )
        if value is None or not math.isfinite(value):
            continue
        filtered.append((row, float(value)))
    return filtered


def _ols_r2(design: np.ndarray, target: np.ndarray) -> Optional[float]:
    if design.size == 0 or target.size == 0:
        return None
    centered = target - float(np.mean(target))
    total = float(np.dot(centered, centered))
    if math.isclose(total, 0.0):
        return None
    try:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residual = target - design @ coefficients
    residual_sum = float(np.dot(residual, residual))
    return max(0.0, min(1.0, 1.0 - residual_sum / total))


def _ols_delta_r2_by_parameter(
    rows_with_values: Sequence[Tuple[CandidateVisualizationRow, float]],
    parameter_names: Sequence[str],
) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    result = {parameter_name: None for parameter_name in parameter_names}
    statuses = {parameter_name: "insufficient" for parameter_name in parameter_names}
    if len(rows_with_values) < 3:
        return result, statuses

    y = np.asarray([value for _, value in rows_with_values], dtype=np.float64)
    if not np.any(np.isfinite(y)) or math.isclose(float(np.nanmin(y)), float(np.nanmax(y))):
        return result, {parameter_name: "constant_target" for parameter_name in parameter_names}

    feature_groups: Dict[str, np.ndarray] = {}
    for parameter_name in parameter_names:
        labels = [_stringify_param(row.params.get(parameter_name)) for row, _ in rows_with_values]
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            statuses[parameter_name] = "constant_parameter"
            continue
        numeric_values = np.asarray(
            [_numeric_param(row.params.get(parameter_name)) for row, _ in rows_with_values],
            dtype=np.float64,
        )
        if np.all(np.isfinite(numeric_values)) and len(set(numeric_values.tolist())) > 2:
            std = float(np.std(numeric_values))
            if math.isclose(std, 0.0):
                statuses[parameter_name] = "constant_parameter"
                continue
            feature_groups[parameter_name] = ((numeric_values - float(np.mean(numeric_values))) / std).reshape(-1, 1)
            continue
        columns = []
        for label in unique_labels[1:]:
            columns.append(np.asarray([1.0 if item == label else 0.0 for item in labels], dtype=np.float64))
        if columns:
            feature_groups[parameter_name] = np.column_stack(columns)

    active_groups = {name: matrix for name, matrix in feature_groups.items() if matrix.size > 0}
    if not active_groups:
        return result, statuses
    matrices = list(active_groups.values())
    full_feature_matrix = np.column_stack(matrices)
    full_design = np.column_stack([np.ones(len(rows_with_values), dtype=np.float64), full_feature_matrix])
    if len(rows_with_values) < full_design.shape[1] + SENSITIVITY_OLS_MIN_EXTRA_ROWS:
        return result, statuses

    full_r2 = _ols_r2(full_design, y)
    if full_r2 is None:
        return result, statuses

    for parameter_name, matrix in active_groups.items():
        reduced_matrices = [
            other_matrix
            for other_name, other_matrix in active_groups.items()
            if other_name != parameter_name
        ]
        if reduced_matrices:
            reduced_design = np.column_stack(
                [np.ones(len(rows_with_values), dtype=np.float64), *reduced_matrices]
            )
        else:
            reduced_design = np.ones((len(rows_with_values), 1), dtype=np.float64)
        reduced_r2 = _ols_r2(reduced_design, y)
        if reduced_r2 is None:
            reduced_r2 = 0.0
        result[parameter_name] = max(0.0, float(full_r2 - reduced_r2))
        statuses[parameter_name] = "ok"
    return result, statuses


def _compute_parameter_sensitivity_for_rows(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
    *,
    stage_name: Optional[str],
    dataset_scope: str,
    value_option: str,
    filters: Optional[Dict[str, object]],
    previous_rows: Sequence[CandidateVisualizationRow] = (),
    attach_previous: bool = True,
) -> List[ParameterSensitivityRow]:
    target_stage = stage_name or model.primary_stage_name
    rows_with_values = _candidate_rows_for_sensitivity(
        model,
        rows,
        stage_name=target_stage,
        dataset_scope=dataset_scope,
        value_option=value_option,
        filters=filters,
    )
    parameter_names = [
        parameter_name
        for parameter_name in model.parameter_names
        if any(parameter_name in row.params for row, _ in rows_with_values)
    ]
    ols_values, ols_statuses = _ols_delta_r2_by_parameter(rows_with_values, parameter_names)

    previous_lookup: Dict[str, ParameterSensitivityRow] = {}
    if attach_previous and previous_rows:
        previous_rows_result = _compute_parameter_sensitivity_for_rows(
            model,
            previous_rows,
            stage_name=target_stage,
            dataset_scope=dataset_scope,
            value_option=value_option,
            filters=filters,
            previous_rows=(),
            attach_previous=False,
        )
        previous_lookup = {row.parameter_name: row for row in previous_rows_result}

    higher_is_better = _higher_is_better_value_option(value_option)
    sensitivity_rows: List[ParameterSensitivityRow] = []
    for parameter_name in parameter_names:
        grouped: Dict[str, List[Tuple[CandidateVisualizationRow, float]]] = {}
        numeric_by_label: Dict[str, float] = {}
        for row, value in rows_with_values:
            if parameter_name not in row.params:
                continue
            label = _stringify_param(row.params.get(parameter_name))
            grouped.setdefault(label, []).append((row, value))
            numeric_by_label[label] = _numeric_param(row.params.get(parameter_name))
        if not grouped:
            continue

        effects: List[ParameterValueEffect] = []
        for label, group_rows in grouped.items():
            values = [value for _, value in group_rows if math.isfinite(value)]
            if not values:
                continue
            effects.append(
                ParameterValueEffect(
                    value=label,
                    numeric_value=(
                        float(numeric_by_label[label])
                        if math.isfinite(numeric_by_label.get(label, float("nan")))
                        else None
                    ),
                    count=len(values),
                    mean_value=float(np.mean(values)),
                    min_value=float(np.min(values)),
                    max_value=float(np.max(values)),
                    candidate_ids=sorted({row.candidate_id for row, _ in group_rows}),
                )
            )
        if not effects:
            continue

        means = [effect.mean_value for effect in effects if effect.mean_value is not None]
        num_values = len(effects)
        min_count = min(effect.count for effect in effects)
        max_count = max(effect.count for effect in effects)
        coverage_warning = None
        if num_values < 2:
            coverage_warning = "only one finite parameter value"
            impact_score = 0.0
        elif min_count <= SENSITIVITY_MIN_GROUP_COUNT:
            coverage_warning = f"sparse value coverage; minimum group count is {min_count}"
            impact_score = float(max(means) - min(means))
        else:
            impact_score = float(max(means) - min(means))

        ranked_effects = sorted(
            effects,
            key=lambda effect: (
                effect.mean_value if effect.mean_value is not None else float("-inf")
            ),
            reverse=higher_is_better,
        )
        best_effect = ranked_effects[0] if ranked_effects else None
        worst_effect = ranked_effects[-1] if ranked_effects else None
        previous = previous_lookup.get(parameter_name)
        sensitivity_rows.append(
            ParameterSensitivityRow(
                rank=0,
                parameter_name=parameter_name,
                stage=target_stage,
                dataset_scope=dataset_scope,
                value_option=value_option,
                impact_score=impact_score,
                best_value=None if best_effect is None else best_effect.value,
                worst_value=None if worst_effect is None else worst_effect.value,
                best_mean_value=None if best_effect is None else best_effect.mean_value,
                worst_mean_value=None if worst_effect is None else worst_effect.mean_value,
                num_values=num_values,
                num_rows=sum(effect.count for effect in effects),
                min_group_count=min_count,
                max_group_count=max_count,
                coverage_warning=coverage_warning,
                ols_delta_r2=ols_values.get(parameter_name),
                ols_status=ols_statuses.get(parameter_name, "insufficient"),
                previous_impact_score=None if previous is None else previous.impact_score,
                previous_best_value=None if previous is None else previous.best_value,
                generation_delta=(
                    None
                    if previous is None
                    else float(impact_score - previous.impact_score)
                ),
                value_effects=sorted(
                    effects,
                    key=lambda effect: (
                        effect.numeric_value is None,
                        effect.numeric_value if effect.numeric_value is not None else 0.0,
                        effect.value,
                    ),
                ),
            )
        )

    sensitivity_rows.sort(
        key=lambda row: (
            -row.impact_score,
            -(row.ols_delta_r2 if row.ols_delta_r2 is not None else -1.0),
            row.parameter_name,
        )
    )
    for rank, row in enumerate(sensitivity_rows, start=1):
        row.rank = rank
    return sensitivity_rows


def compute_parameter_sensitivity(
    model: SweepVisualizationModel,
    stage_name: Optional[str] = None,
    dataset_scope: str = "aggregate",
    value_option: str = "score",
    filters: Optional[Dict[str, object]] = None,
) -> List[ParameterSensitivityRow]:
    """Rank parameters by grouped value impact for a stage and dataset scope."""

    return _compute_parameter_sensitivity_for_rows(
        model,
        candidate_rows_for_generation_scope(model, model.default_generation_scope),
        stage_name=stage_name or model.primary_stage_name,
        dataset_scope=dataset_scope,
        value_option=value_option,
        filters=filters,
        previous_rows=model.previous_generation_candidates,
        attach_previous=True,
    )


def _stage_score_span(model: SweepVisualizationModel, stage_name: str) -> float:
    values = [
        float(row.score)
        for row in _rows_for_stage(model, stage_name)
        if row.score is not None and math.isfinite(row.score)
    ]
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _suggestion_effect_threshold(model: SweepVisualizationModel, stage_name: str) -> float:
    return max(
        EXTENSION_MIN_ABSOLUTE_EFFECT,
        EXTENSION_MIN_RELATIVE_EFFECT * _stage_score_span(model, stage_name),
    )


def _numeric_effects(row: ParameterSensitivityRow) -> List[ParameterValueEffect]:
    return [
        effect
        for effect in row.value_effects
        if effect.numeric_value is not None and math.isfinite(effect.numeric_value)
    ]


def generate_next_extension_suggestions(
    model: SweepVisualizationModel,
    sensitivity_rows: Sequence[ParameterSensitivityRow],
) -> List[ExtensionSuggestion]:
    """Generate conservative next-sweep recommendations from sensitivity results."""

    active_generation_id = _coerce_optional_str(model.generation_summary.get("active_generation_id"))
    previous_generation_id = _coerce_optional_str(model.generation_summary.get("previous_generation_id"))
    suggestions: List[ExtensionSuggestion] = []
    primary_rows = [
        row
        for row in sensitivity_rows
        if row.value_option == "score"
        and row.dataset_scope == "aggregate"
        and row.num_values >= 2
    ]
    if not primary_rows:
        return [
            ExtensionSuggestion(
                rank=1,
                rule_id="no_high_confidence",
                priority=999,
                stage=model.primary_stage_name,
                parameter_name=None,
                dataset_scope="aggregate",
                title="No high-confidence extension suggestion",
                recommendation="Collect more successful ranked candidates before extending the search space.",
                evidence="No aggregate score sensitivity rows with at least two parameter values were available.",
                confidence="low",
                generation_id=active_generation_id,
                previous_generation_id=previous_generation_id,
            )
        ]

    by_parameter = {row.parameter_name: row for row in primary_rows}
    for row in primary_rows:
        threshold = _suggestion_effect_threshold(model, row.stage)
        if row.impact_score < threshold:
            continue
        numeric_effects = _numeric_effects(row)
        if numeric_effects and row.best_value is not None:
            numeric_values = [cast(float, effect.numeric_value) for effect in numeric_effects]
            best_effect = next((effect for effect in numeric_effects if effect.value == row.best_value), None)
            if best_effect is not None and best_effect.numeric_value is not None:
                min_numeric = min(numeric_values)
                max_numeric = max(numeric_values)
                if math.isclose(best_effect.numeric_value, max_numeric):
                    suggestions.append(
                        ExtensionSuggestion(
                            rank=0,
                            rule_id="boundary_winner",
                            priority=10,
                            stage=row.stage,
                            parameter_name=row.parameter_name,
                            dataset_scope="aggregate",
                            title=f"Extend {row.parameter_name} upward",
                            recommendation=(
                                f"Add values above {row.best_value} for {row.parameter_name}; "
                                "the best observed value is at the upper boundary."
                            ),
                            evidence=(
                                f"Aggregate score changes by {_format_float(row.impact_score)} across "
                                f"{row.num_values} values; best={row.best_value}, worst={row.worst_value}."
                            ),
                            confidence="high",
                            related_values=[effect.value for effect in numeric_effects],
                            generation_id=active_generation_id,
                            previous_generation_id=previous_generation_id,
                        )
                    )
                elif math.isclose(best_effect.numeric_value, min_numeric):
                    suggestions.append(
                        ExtensionSuggestion(
                            rank=0,
                            rule_id="boundary_winner",
                            priority=10,
                            stage=row.stage,
                            parameter_name=row.parameter_name,
                            dataset_scope="aggregate",
                            title=f"Extend {row.parameter_name} downward",
                            recommendation=(
                                f"Add values below {row.best_value} for {row.parameter_name}; "
                                "the best observed value is at the lower boundary."
                            ),
                            evidence=(
                                f"Aggregate score changes by {_format_float(row.impact_score)} across "
                                f"{row.num_values} values; best={row.best_value}, worst={row.worst_value}."
                            ),
                            confidence="high",
                            related_values=[effect.value for effect in numeric_effects],
                            generation_id=active_generation_id,
                            previous_generation_id=previous_generation_id,
                        )
                    )
                elif row.num_values >= 3:
                    suggestions.append(
                        ExtensionSuggestion(
                            rank=0,
                            rule_id="densify_interior",
                            priority=30,
                            stage=row.stage,
                            parameter_name=row.parameter_name,
                            dataset_scope="aggregate",
                            title=f"Densify around {row.parameter_name}={row.best_value}",
                            recommendation=(
                                f"Add intermediate values around {row.best_value} for {row.parameter_name}; "
                                "the best observed value is interior and the parameter is impactful."
                            ),
                            evidence=(
                                f"Aggregate score changes by {_format_float(row.impact_score)} across "
                                f"{row.num_values} values."
                            ),
                            confidence="medium",
                            related_values=[effect.value for effect in numeric_effects],
                            generation_id=active_generation_id,
                            previous_generation_id=previous_generation_id,
                        )
                    )
        if row.min_group_count <= SENSITIVITY_MIN_GROUP_COUNT:
            suggestions.append(
                ExtensionSuggestion(
                    rank=0,
                    rule_id="sparse_high_impact",
                    priority=40,
                    stage=row.stage,
                    parameter_name=row.parameter_name,
                    dataset_scope="aggregate",
                    title=f"Confirm sparse {row.parameter_name} effect",
                    recommendation=(
                        f"Repeat or add nearby values for {row.parameter_name}; at least one value group "
                        "has very low coverage but the apparent effect is large."
                    ),
                    evidence=(
                        f"Minimum group count is {row.min_group_count}; aggregate score impact is "
                        f"{_format_float(row.impact_score)}."
                    ),
                    confidence="medium",
                    related_values=[effect.value for effect in row.value_effects],
                    generation_id=active_generation_id,
                    previous_generation_id=previous_generation_id,
                )
            )
        if (
            row.previous_impact_score is not None
            and row.previous_best_value is not None
            and row.best_value is not None
            and row.best_value != row.previous_best_value
        ):
            suggestions.append(
                ExtensionSuggestion(
                    rank=0,
                    rule_id="extension_delta",
                    priority=25,
                    stage=row.stage,
                    parameter_name=row.parameter_name,
                    dataset_scope="aggregate",
                    title=f"Recheck new {row.parameter_name} winner",
                    recommendation=(
                        f"Explore around {row.best_value}; the active generation changed the best "
                        f"{row.parameter_name} value from {row.previous_best_value}."
                    ),
                    evidence=(
                        f"Impact changed by {_format_float(row.generation_delta)} versus the previous generation."
                    ),
                    confidence="medium",
                    related_values=[row.previous_best_value, row.best_value],
                    generation_id=active_generation_id,
                    previous_generation_id=previous_generation_id,
                )
            )

    suggestion_stage = primary_rows[0].stage if primary_rows else model.primary_stage_name
    dataset_rows_by_parameter: Dict[str, Dict[str, ParameterSensitivityRow]] = {}
    for dataset_id in model.datasets:
        dataset_rows = compute_parameter_sensitivity(
            model,
            stage_name=suggestion_stage,
            dataset_scope=dataset_id,
            value_option="score",
            filters={},
        )
        for row in dataset_rows:
            if row.parameter_name in by_parameter:
                dataset_rows_by_parameter.setdefault(row.parameter_name, {})[dataset_id] = row
    for parameter_name, dataset_rows in dataset_rows_by_parameter.items():
        aggregate = by_parameter[parameter_name]
        if aggregate.impact_score < _suggestion_effect_threshold(model, aggregate.stage):
            continue
        best_by_dataset = {
            dataset_id: row.best_value
            for dataset_id, row in dataset_rows.items()
            if row.best_value is not None and row.num_values >= 2
        }
        distinct_best = sorted({value for value in best_by_dataset.values() if value is not None})
        if len(distinct_best) < 2:
            continue
        evidence = ", ".join(
            f"{dataset_id}->{value}"
            for dataset_id, value in sorted(best_by_dataset.items(), key=lambda item: item[0])
        )
        suggestions.append(
            ExtensionSuggestion(
                rank=0,
                rule_id="dataset_disagreement",
                priority=20,
                stage=aggregate.stage,
                parameter_name=parameter_name,
                dataset_scope="per_dataset",
                title=f"Resolve dataset disagreement for {parameter_name}",
                recommendation=(
                    f"Add values around the competing {parameter_name} optima or add representative datasets; "
                    "different datasets prefer different values."
                ),
                evidence=evidence,
                confidence="medium",
                related_values=distinct_best,
                generation_id=active_generation_id,
                previous_generation_id=previous_generation_id,
            )
        )

    if not suggestions:
        suggestions.append(
            ExtensionSuggestion(
                rank=1,
                rule_id="no_high_confidence",
                priority=999,
                stage=model.primary_stage_name,
                parameter_name=None,
                dataset_scope="aggregate",
                title="No high-confidence extension suggestion",
                recommendation=(
                    "Keep the current search space or collect more candidates before extending; "
                    "no parameter crossed the conservative effect thresholds."
                ),
                evidence="All aggregate score effects were below the configured threshold or lacked coverage.",
                confidence="low",
                generation_id=active_generation_id,
                previous_generation_id=previous_generation_id,
            )
        )

    suggestions.sort(key=lambda item: (item.priority, item.parameter_name or "", item.rule_id))
    unique: List[ExtensionSuggestion] = []
    seen = set()
    for suggestion in suggestions:
        key = (suggestion.rule_id, suggestion.parameter_name, suggestion.recommendation)
        if key in seen:
            continue
        seen.add(key)
        unique.append(suggestion)
        if len(unique) >= EXTENSION_MAX_SUGGESTIONS:
            break
    for rank, suggestion in enumerate(unique, start=1):
        suggestion.rank = rank
    return unique


def _parameter_value_effect_to_export_dict(effect: ParameterValueEffect) -> Dict[str, object]:
    return {
        "value": effect.value,
        "numeric_value": effect.numeric_value,
        "count": effect.count,
        "mean_value": effect.mean_value,
        "min_value": effect.min_value,
        "max_value": effect.max_value,
        "candidate_ids": effect.candidate_ids,
    }


def _parameter_sensitivity_to_export_dict(row: ParameterSensitivityRow) -> Dict[str, object]:
    return {
        "rank": row.rank,
        "parameter_name": row.parameter_name,
        "stage": row.stage,
        "dataset_scope": row.dataset_scope,
        "value_option": row.value_option,
        "impact_score": row.impact_score,
        "best_value": row.best_value,
        "worst_value": row.worst_value,
        "best_mean_value": row.best_mean_value,
        "worst_mean_value": row.worst_mean_value,
        "num_values": row.num_values,
        "num_rows": row.num_rows,
        "min_group_count": row.min_group_count,
        "max_group_count": row.max_group_count,
        "coverage_warning": row.coverage_warning,
        "ols_delta_r2": row.ols_delta_r2,
        "ols_status": row.ols_status,
        "previous_impact_score": row.previous_impact_score,
        "previous_best_value": row.previous_best_value,
        "generation_delta": row.generation_delta,
        "value_effects": [
            _parameter_value_effect_to_export_dict(effect)
            for effect in row.value_effects
        ],
    }


def _extension_suggestion_to_export_dict(suggestion: ExtensionSuggestion) -> Dict[str, object]:
    return {
        "rank": suggestion.rank,
        "rule_id": suggestion.rule_id,
        "priority": suggestion.priority,
        "stage": suggestion.stage,
        "parameter_name": suggestion.parameter_name,
        "dataset_scope": suggestion.dataset_scope,
        "title": suggestion.title,
        "recommendation": suggestion.recommendation,
        "evidence": suggestion.evidence,
        "confidence": suggestion.confidence,
        "related_values": suggestion.related_values,
        "generation_id": suggestion.generation_id,
        "previous_generation_id": suggestion.previous_generation_id,
    }


def create_parameter_sensitivity_figure(
    model: SweepVisualizationModel,
    sensitivity_rows: Optional[Sequence[ParameterSensitivityRow]] = None,
    *,
    stage_name: Optional[str] = None,
    dataset_scope: str = "aggregate",
    value_option: str = "score",
    filters: Optional[Dict[str, object]] = None,
    top_n: int = 8,
) -> Figure:
    rows = list(
        sensitivity_rows
        if sensitivity_rows is not None
        else compute_parameter_sensitivity(
            model,
            stage_name=stage_name or model.primary_stage_name,
            dataset_scope=dataset_scope,
            value_option=value_option,
            filters=filters,
        )
    )
    rows = rows[: max(1, top_n)]
    fig = Figure(figsize=(11, max(4, 0.45 * max(1, len(rows)) + 2.5)))
    ax = fig.add_subplot(111)
    if not rows:
        ax.text(0.5, 0.5, "No parameter sensitivity data available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    labels = [row.parameter_name for row in rows]
    y_positions = np.arange(len(rows))
    impacts = np.asarray([row.impact_score for row in rows], dtype=np.float64)
    ax.barh(y_positions, impacts, color="#2a6f97")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("grouped mean delta")
    ax.set_title(f"{rows[0].stage}: {rows[0].dataset_scope} {rows[0].value_option} sensitivity")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    for idx, row in enumerate(rows):
        detail = f"{row.best_value} vs {row.worst_value}"
        if row.ols_delta_r2 is not None:
            detail += f" | dR2={row.ols_delta_r2:.3f}"
        ax.text(row.impact_score, idx, f"  {detail}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def _find_run_metric_record(
    model: SweepVisualizationModel,
    *,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
    generation_id: Optional[str] = None,
) -> Optional[RunMetricRecord]:
    return next(
        (
            record
            for record in run_records_for_generation_scope(model, model.default_generation_scope)
            if record.stage == stage_name
            and record.candidate_id == candidate_id
            and record.dataset_id == dataset_id
            and (generation_id is None or record.generation_id == generation_id)
        ),
        None,
    )


def _unpack_rectification_key(
    key: RectificationResultKey | Tuple[str, str, str],
) -> Tuple[str, str, str, Optional[str]]:
    if len(key) == 4:
        stage_name, candidate_id, dataset_id, generation_id = key
        return stage_name, candidate_id, dataset_id, generation_id
    stage_name, candidate_id, dataset_id = cast(Tuple[str, str, str], key)
    return stage_name, candidate_id, dataset_id, None


def _load_preview_array(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    preview_path = Path(path)
    if not preview_path.exists():
        return None
    try:
        return np.asarray(Image.open(preview_path).convert("L"), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def _load_preview_array_with_warnings(path: Optional[str]) -> Tuple[Optional[np.ndarray], List[str]]:
    if not path:
        return None, ["Preview image unavailable."]
    preview_path = Path(path)
    if not preview_path.exists():
        return None, [f"Preview image not found: {preview_path}"]
    try:
        return np.asarray(Image.open(preview_path).convert("L"), dtype=np.float64), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"Failed to read preview image: {type(exc).__name__}: {exc}"]


def _collapse_image_like(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if not isinstance(array, np.ndarray):
        return None
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return None
    while arr.ndim > 2:
        arr = np.nanmean(arr, axis=-1)
    return arr


def _finite_2d_channel(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    arr = _collapse_image_like(array)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
        return None
    return np.asarray(arr, dtype=np.float64)


def _rectification_candidate_row(
    model: SweepVisualizationModel,
    selection: RectificationViewSelection,
) -> Optional[CandidateVisualizationRow]:
    return next(
        (
            row
            for row in candidate_rows_for_generation_scope(model, model.default_generation_scope)
            if row.stage == selection.stage and row.candidate_id == selection.candidate_id
            and (selection.generation_id is None or row.generation_id == selection.generation_id)
        ),
        None,
    )


def _rectification_record(
    model: SweepVisualizationModel,
    selection: RectificationViewSelection,
) -> Optional[RunMetricRecord]:
    return _find_run_metric_record(
        model,
        stage_name=selection.stage,
        candidate_id=selection.candidate_id,
        dataset_id=selection.dataset_id,
        generation_id=selection.generation_id,
    )


def _rectification_preview_path(
    row: Optional[CandidateVisualizationRow],
    record: Optional[RunMetricRecord],
    dataset_id: str,
) -> Optional[str]:
    if row is not None:
        preview_path = row.preview_paths.get(dataset_id) or row.preview_paths.get("aggregate")
        if preview_path:
            return preview_path
    return None if record is None else record.preview_path


def _root_artifact_generation_matches(
    metric_record_path: Path,
    generation_id: Optional[str],
) -> bool:
    if generation_id is None:
        return True
    if not _candidate_path_exists(metric_record_path):
        return False
    try:
        payload, _used_path, _warnings = read_json_first_valid(
            metric_record_path,
            validator=lambda obj: isinstance(obj, dict),
        )
    except Exception:  # noqa: BLE001
        return False
    return _coerce_optional_str(cast(Dict[str, object], payload).get("generation_id")) == generation_id


def _root_run_artifact_paths_for_selection(
    model: SweepVisualizationModel,
    *,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
    generation_id: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    base_dir = (
        model.metric_phase_dir
        if stage_name == "metric_phase" and model.metric_phase_dir is not None
        else model.experiment_root
    )
    run_dir = base_dir / "runs" / stage_name / candidate_id / dataset_id
    metric_record_path = run_dir / "metric_record.json"
    if not _root_artifact_generation_matches(metric_record_path, generation_id):
        return None, None, None
    preview_path = run_dir / "avg_rectified.png"
    workspace_path = run_dir / "workspace.npz"
    return (
        str(preview_path) if preview_path.exists() else None,
        str(workspace_path) if _candidate_path_exists(workspace_path) else None,
        str(metric_record_path) if _candidate_path_exists(metric_record_path) else None,
    )


def _rectification_artifact_paths(
    model: SweepVisualizationModel,
    selection: RectificationViewSelection,
    row: Optional[CandidateVisualizationRow],
    record: Optional[RunMetricRecord],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    preview_path = _rectification_preview_path(row, record, selection.dataset_id)
    workspace_path = None if record is None else record.workspace_path
    metric_record_path = None if record is None else record.metric_record_path
    if preview_path is not None and workspace_path is not None:
        return preview_path, workspace_path, metric_record_path

    root_preview, root_workspace, root_metric = _root_run_artifact_paths_for_selection(
        model,
        stage_name=selection.stage,
        candidate_id=selection.candidate_id,
        dataset_id=selection.dataset_id,
        generation_id=selection.generation_id,
    )
    return (
        preview_path or root_preview,
        workspace_path or root_workspace,
        metric_record_path or root_metric,
    )


def load_rectification_display_bundle(
    model: SweepVisualizationModel,
    selection: RectificationViewSelection,
) -> RectificationDisplayBundle:
    """Load display-ready image channels for one rectification result."""

    row = _rectification_candidate_row(model, selection)
    record = _rectification_record(model, selection)
    preview_path, workspace_path, _metric_record_path = _rectification_artifact_paths(
        model,
        selection,
        row,
        record,
    )
    preview, warnings = _load_preview_array_with_warnings(preview_path)
    payload: Dict[str, np.ndarray] = {}
    if workspace_path:
        payload, workspace_warnings = _load_workspace_diagnostics(Path(workspace_path))
        warnings.extend(workspace_warnings)
    else:
        warnings.append("workspace.npz unavailable")

    disp_x = _finite_2d_channel(cast(Optional[np.ndarray], payload.get("pixel_displacements_x")))
    disp_y = _finite_2d_channel(cast(Optional[np.ndarray], payload.get("pixel_displacements_y")))
    displacement_magnitude = None
    if disp_x is not None and disp_y is not None:
        if disp_x.shape == disp_y.shape:
            displacement_magnitude = np.sqrt(np.square(disp_x) + np.square(disp_y))
        else:
            warnings.append(
                "Displacement map shapes do not match: "
                f"x={disp_x.shape}, y={disp_y.shape}"
            )

    component_name = selection.component_name or "gradient"
    blended_score = _finite_2d_channel(cast(Optional[np.ndarray], payload.get("pixel_blended_scores")))
    component_score = _finite_2d_channel(
        cast(Optional[np.ndarray], payload.get(f"pixel_component_score_{component_name}"))
    )
    global_avg = _finite_2d_channel(
        cast(Optional[np.ndarray], payload.get("average_rectified_global_image"))
    )
    final_avg = _finite_2d_channel(cast(Optional[np.ndarray], payload.get("average_rectified_image")))
    global_final_delta = None
    if global_avg is not None and final_avg is not None:
        if global_avg.shape == final_avg.shape:
            global_final_delta = global_avg - final_avg
        else:
            warnings.append(
                "Global/final average shapes do not match: "
                f"global={global_avg.shape}, final={final_avg.shape}"
            )

    return RectificationDisplayBundle(
        selection=selection,
        preview_path=preview_path,
        workspace_path=workspace_path,
        preview=preview,
        displacement_magnitude=displacement_magnitude,
        blended_score=blended_score,
        component_score=component_score,
        global_final_delta=global_final_delta,
        warnings=_sorted_unique(warnings),
    )


def _comparison_artifacts_for_candidate(
    model: SweepVisualizationModel,
    row: CandidateVisualizationRow,
    *,
    dataset_id: str,
    component_name: str,
) -> Dict[str, object]:
    record = _find_run_metric_record(
        model,
        stage_name=row.stage,
        candidate_id=row.candidate_id,
        dataset_id=dataset_id,
        generation_id=row.generation_id,
    )
    preview_path, workspace_path, _metric_record_path = _rectification_artifact_paths(
        model,
        RectificationViewSelection(
            stage=row.stage,
            candidate_id=row.candidate_id,
            dataset_id=dataset_id,
            generation_id=row.generation_id,
            component_name=component_name,
        ),
        row,
        record,
    )
    preview = _load_preview_array(preview_path)

    payload: Dict[str, np.ndarray] = {}
    read_warnings: List[str] = []
    if workspace_path is not None:
        payload, read_warnings = _load_workspace_diagnostics(Path(workspace_path))

    blended = _collapse_image_like(cast(Optional[np.ndarray], payload.get("pixel_blended_scores")))
    component = _collapse_image_like(
        cast(Optional[np.ndarray], payload.get(f"pixel_component_score_{component_name}"))
    )
    disp_x = cast(Optional[np.ndarray], payload.get("pixel_displacements_x"))
    disp_y = cast(Optional[np.ndarray], payload.get("pixel_displacements_y"))
    magnitude = None
    if isinstance(disp_x, np.ndarray) and isinstance(disp_y, np.ndarray):
        magnitude = _collapse_image_like(
            np.sqrt(
                np.square(np.asarray(disp_x, dtype=np.float64))
                + np.square(np.asarray(disp_y, dtype=np.float64))
            )
        )
    global_avg = _collapse_image_like(
        cast(Optional[np.ndarray], payload.get("average_rectified_global_image"))
    )
    final_avg = _collapse_image_like(
        cast(Optional[np.ndarray], payload.get("average_rectified_image"))
    )
    delta = None
    if (
        isinstance(global_avg, np.ndarray)
        and isinstance(final_avg, np.ndarray)
        and global_avg.shape == final_avg.shape
    ):
        delta = np.asarray(global_avg, dtype=np.float64) - np.asarray(final_avg, dtype=np.float64)

    return {
        "record": record,
        "preview_path": preview_path,
        "preview": preview,
        "blended": blended,
        "component": component,
        "magnitude": magnitude,
        "delta": delta,
        "warnings": read_warnings,
    }


def _draw_image_panel(
    fig: Figure,
    ax: Any,
    array: Optional[np.ndarray],
    *,
    title: str,
    cmap: str = "gray",
    unavailable_message: str = "Unavailable",
) -> None:
    if not isinstance(array, np.ndarray):
        ax.text(0.5, 0.5, unavailable_message, ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        return
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        ax.text(0.5, 0.5, unavailable_message, ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        return
    image = ax.imshow(arr, aspect="equal", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def _top_candidates_for_stage(
    model: SweepVisualizationModel,
    stage_name: Optional[str] = None,
    *,
    limit: int = DEFAULT_REPORT_TOP_K,
) -> List[CandidateVisualizationRow]:
    rows = _rows_for_stage(model, stage_name or model.primary_stage_name)
    return rows[: max(1, min(limit, len(rows)))] if rows else []


def _comparison_rows_need_stage_labels(rows: Sequence[CandidateVisualizationRow]) -> bool:
    if len({row.stage for row in rows}) > 1:
        return True
    candidate_counts = Counter(row.candidate_id for row in rows)
    return any(count > 1 for count in candidate_counts.values())


def _comparison_row_label(
    row: CandidateVisualizationRow,
    rows: Sequence[CandidateVisualizationRow],
) -> str:
    if _comparison_rows_need_stage_labels(rows):
        return f"{row.stage}:{row.candidate_id}"
    return row.candidate_id


def _comparison_stage_header(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
) -> str:
    stages = list(dict.fromkeys(row.stage for row in rows))
    if not stages:
        return f"stage: {model.primary_stage_name}"
    if len(stages) == 1:
        return f"stage: {stages[0]}"
    return f"stages: {', '.join(stages)}"


def _varying_parameter_lines(rows: Sequence[CandidateVisualizationRow]) -> List[str]:
    varying_names = [
        name
        for name in sorted({param for row in rows for param in row.params.keys()})
        if len({_stringify_param(row.params.get(name)) for row in rows}) > 1
    ]
    if not varying_names:
        return ["varying parameters: none"]

    lines = ["varying parameters:"]
    for name in varying_names[:12]:
        assignments = ", ".join(
            f"{row.candidate_id}={_stringify_param(row.params.get(name))}" for row in rows
        )
        lines.append(f"  {name}: {assignments}")
    return lines


def _format_comparison_text(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
    *,
    dataset_id: str,
) -> str:
    if not rows:
        return "No candidates available for comparison."

    lines = [
        _comparison_stage_header(model, rows),
        f"dataset: {dataset_id}",
        f"candidates compared: {len(rows)}",
        "",
    ]
    for row in rows:
        dataset_score = _candidate_score_for_dataset(row, dataset_id, model.primary_metric)
        runtime = _candidate_runtime_for_dataset(row, dataset_id)
        metric_name, metric_value = _candidate_metric_value_for_dataset(
            row,
            dataset_id,
            model.primary_metric,
        )
        lines.append(
            " | ".join(
                [
                    f"rank {row.rank}",
                    _comparison_row_label(row, rows),
                    f"stage={row.stage}",
                    f"strategy={row.strategy}",
                    f"dataset_score={_format_float(dataset_score)}",
                    f"mean_score={_format_float(row.mean_dataset_score)}",
                    f"runtime={_format_float(runtime, precision=3)}s",
                    f"{metric_name}={_format_float(metric_value)}",
                    f"mean_disp={_format_float(row.displacement_field_metrics.get('mean_magnitude'))}",
                    f"mean_view_std={_format_float(row.inter_view_consistency_metrics.get('mean_view_std'))}",
                ]
            )
        )
    lines.extend(["", *_varying_parameter_lines(rows)])
    return "\n".join(lines)


def create_preview_comparison_figure(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
    *,
    dataset_id: str,
) -> Figure:
    fig = Figure(figsize=(max(12, 3.4 * max(1, len(rows))), 4.5))
    if not rows:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No candidates available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    axes = fig.subplots(1, len(rows), squeeze=False)
    for col_idx, row in enumerate(rows):
        artifact = _comparison_artifacts_for_candidate(
            model,
            row,
            dataset_id=dataset_id,
            component_name=_default_component_name(model),
        )
        preview = cast(Optional[np.ndarray], artifact["preview"])
        ax = axes[0, col_idx]
        _draw_image_panel(
            fig,
            ax,
            preview,
            title=(
                f"{_comparison_row_label(row, rows)}\n"
                f"rank {row.rank} | score "
                f"{_format_float(_candidate_score_for_dataset(row, dataset_id, model.primary_metric))}"
            ),
            cmap="gray",
            unavailable_message="Preview unavailable",
        )
    fig.suptitle(f"Preview comparison for {dataset_id}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def create_comparison_figure(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
    *,
    dataset_id: str,
    component_name: str,
) -> Figure:
    fig = Figure(figsize=(max(13, 3.4 * max(1, len(rows))), 13))
    if not rows:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No candidates available for comparison.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    panel_titles = [
        "Averaged rectified preview",
        "Displacement magnitude",
        "Blended score map",
        f"{component_name} score map",
        "Global minus final average",
    ]
    axes = fig.subplots(len(panel_titles), len(rows), squeeze=False)
    for col_idx, row in enumerate(rows):
        artifact = _comparison_artifacts_for_candidate(
            model,
            row,
            dataset_id=dataset_id,
            component_name=component_name,
        )
        arrays = [
            cast(Optional[np.ndarray], artifact["preview"]),
            cast(Optional[np.ndarray], artifact["magnitude"]),
            cast(Optional[np.ndarray], artifact["blended"]),
            cast(Optional[np.ndarray], artifact["component"]),
            cast(Optional[np.ndarray], artifact["delta"]),
        ]
        cmaps = ["gray", "viridis", "viridis", "viridis", "coolwarm"]
        messages = [
            "Preview unavailable",
            "Displacement map unavailable",
            "Blended score map unavailable",
            f"{component_name} map unavailable",
            "Global/final delta unavailable",
        ]
        for row_idx, (title, array, cmap, message) in enumerate(
            zip(panel_titles, arrays, cmaps, messages)
        ):
            axis = axes[row_idx, col_idx]
            rendered_title = title
            if row_idx == 0:
                rendered_title = f"{_comparison_row_label(row, rows)} (rank {row.rank})\n{title}"
            _draw_image_panel(
                fig,
                axis,
                array,
                title=rendered_title,
                cmap=cmap,
                unavailable_message=message,
            )
    fig.suptitle(
        f"Candidate comparison for {dataset_id} ({component_name})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def create_candidate_robustness_figure(
    model: SweepVisualizationModel,
    rows: Sequence[CandidateVisualizationRow],
) -> Figure:
    fig = Figure(
        figsize=(
            max(7, 0.75 * max(1, len(model.datasets)) + 4),
            max(4, 0.6 * max(1, len(rows)) + 3),
        )
    )
    ax = fig.add_subplot(111)
    if not rows:
        ax.text(0.5, 0.5, "No candidates available for robustness view.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    datasets = _real_datasets(model)
    matrix = np.full((len(rows), len(datasets)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for dataset_idx, dataset_id in enumerate(datasets):
            score_value = _candidate_score_for_dataset(row, dataset_id, model.primary_metric)
            matrix[row_idx, dataset_idx] = score_value if score_value is not None else np.nan
    if not np.any(np.isfinite(matrix)):
        ax.text(0.5, 0.5, "No per-dataset score data available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_comparison_row_label(row, rows) for row in rows])
    ax.set_title("Selected candidates across datasets")
    ax.set_xlabel("dataset")
    ax.set_ylabel("candidate")
    for row_idx in range(matrix.shape[0]):
        for dataset_idx in range(matrix.shape[1]):
            if math.isfinite(matrix[row_idx, dataset_idx]):
                ax.text(
                    dataset_idx,
                    row_idx,
                    f"{matrix[row_idx, dataset_idx]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _candidate_csv_rows(model: SweepVisualizationModel) -> List[Dict[str, object]]:
    generation_scope = model.default_generation_scope
    return [
        {
            "generation_scope": generation_scope,
            "row_id": row.row_id,
            "stage": row.stage,
            "candidate_id": row.candidate_id,
            "rank": row.rank,
            "strategy": row.strategy,
            "score": row.score,
            "mean_dataset_score": row.mean_dataset_score,
            "std_dataset_score": row.std_dataset_score,
            "mean_runtime_seconds": row.mean_runtime_seconds,
            "ranking_metric_name": row.ranking_metric_name,
            "ranking_metric_value": row.ranking_metric_value,
            "family": row.family,
            "alpha": row.alpha,
            "formulation_id": row.formulation_id,
            "generation_id": row.generation_id,
            "missing_datasets": "|".join(row.missing_datasets),
            "dataset_scores_json": _compact_json(row.dataset_scores),
            "dataset_runtimes_json": _compact_json(row.dataset_runtimes),
            "dataset_metrics_json": _compact_json(row.dataset_metrics),
            "params_json": _compact_json(row.params),
            "config_json": _compact_json(row.config),
        }
        for row in candidate_rows_for_generation_scope(model, generation_scope)
    ]


def _run_metric_csv_rows(model: SweepVisualizationModel) -> List[Dict[str, object]]:
    generation_scope = model.default_generation_scope
    return [
        {
            "generation_scope": generation_scope,
            "row_id": record.row_id,
            "stage": record.stage,
            "candidate_id": record.candidate_id,
            "dataset_id": record.dataset_id,
            "run_name": record.run_name,
            "strategy": record.strategy,
            "family": record.family,
            "alpha": record.alpha,
            "formulation_id": record.formulation_id,
            "generation_id": record.generation_id,
            "score": record.score,
            "mean_dataset_score": record.mean_dataset_score,
            "mean_runtime_seconds": record.mean_runtime_seconds,
            "ranking_metric_name": record.ranking_metric_name,
            "ranking_metric_value": record.ranking_metric_value,
            "metric_means_json": _compact_json(record.metric_means),
            "displacement_field_metrics_json": _compact_json(record.displacement_field_metrics),
            "inter_view_consistency_metrics_json": _compact_json(
                record.inter_view_consistency_metrics
            ),
            "params_json": _compact_json(record.params),
            "config_json": _compact_json(record.config),
            "preview_path": record.preview_path,
            "workspace_path": record.workspace_path,
            "metric_record_path": record.metric_record_path,
        }
        for record in run_records_for_generation_scope(model, generation_scope)
    ]


def _parameter_sensitivity_csv_rows(
    rows: Sequence[ParameterSensitivityRow],
) -> List[Dict[str, object]]:
    return [
        {
            "rank": row.rank,
            "stage": row.stage,
            "dataset_scope": row.dataset_scope,
            "value_option": row.value_option,
            "parameter_name": row.parameter_name,
            "impact_score": row.impact_score,
            "best_value": row.best_value,
            "best_mean_value": row.best_mean_value,
            "worst_value": row.worst_value,
            "worst_mean_value": row.worst_mean_value,
            "num_values": row.num_values,
            "num_rows": row.num_rows,
            "min_group_count": row.min_group_count,
            "max_group_count": row.max_group_count,
            "coverage_warning": row.coverage_warning,
            "ols_delta_r2": row.ols_delta_r2,
            "ols_status": row.ols_status,
            "previous_impact_score": row.previous_impact_score,
            "previous_best_value": row.previous_best_value,
            "generation_delta": row.generation_delta,
            "value_effects_json": _compact_json(
                [_parameter_value_effect_to_export_dict(effect) for effect in row.value_effects]
            ),
        }
        for row in rows
    ]


def _write_candidate_leaderboard_csv(model: SweepVisualizationModel, path: Path) -> None:
    write_csv_atomic(
        path,
        [
            "generation_scope",
            "row_id",
            "stage",
            "candidate_id",
            "rank",
            "strategy",
            "score",
            "mean_dataset_score",
            "std_dataset_score",
            "mean_runtime_seconds",
            "ranking_metric_name",
            "ranking_metric_value",
            "family",
            "alpha",
            "formulation_id",
            "generation_id",
            "missing_datasets",
            "dataset_scores_json",
            "dataset_runtimes_json",
            "dataset_metrics_json",
            "params_json",
            "config_json",
        ],
        _candidate_csv_rows(model),
        create_backup=True,
    )


def _write_run_metric_records_csv(model: SweepVisualizationModel, path: Path) -> None:
    write_csv_atomic(
        path,
        [
            "generation_scope",
            "row_id",
            "stage",
            "candidate_id",
            "dataset_id",
            "run_name",
            "strategy",
            "family",
            "alpha",
            "formulation_id",
            "generation_id",
            "score",
            "mean_dataset_score",
            "mean_runtime_seconds",
            "ranking_metric_name",
            "ranking_metric_value",
            "metric_means_json",
            "displacement_field_metrics_json",
            "inter_view_consistency_metrics_json",
            "params_json",
            "config_json",
            "preview_path",
            "workspace_path",
            "metric_record_path",
        ],
        _run_metric_csv_rows(model),
        create_backup=True,
    )


def _write_parameter_sensitivity_csv(
    rows: Sequence[ParameterSensitivityRow],
    path: Path,
) -> None:
    write_csv_atomic(
        path,
        [
            "rank",
            "stage",
            "dataset_scope",
            "value_option",
            "parameter_name",
            "impact_score",
            "best_value",
            "best_mean_value",
            "worst_value",
            "worst_mean_value",
            "num_values",
            "num_rows",
            "min_group_count",
            "max_group_count",
            "coverage_warning",
            "ols_delta_r2",
            "ols_status",
            "previous_impact_score",
            "previous_best_value",
            "generation_delta",
            "value_effects_json",
        ],
        _parameter_sensitivity_csv_rows(rows),
        create_backup=True,
    )


def _stage_status_lines(model: SweepVisualizationModel) -> List[str]:
    return [
        (
            f"{stage_name}: planned={stage.num_runs_planned}, recorded={stage.num_runs_recorded}, "
            f"successful={stage.num_successful}, failed={stage.num_failed}, "
            f"pending={stage.num_pending}, running={stage.num_running}"
        )
        for stage_name, stage in ((name, model.stages[name]) for name in model.stage_order)
    ]


def _diagnostic_availability_lines(model: SweepVisualizationModel) -> List[str]:
    components = cast(List[str], model.diagnostic_availability.get("components", []))
    return [
        f"score_maps={bool(model.diagnostic_availability.get('score_maps'))}",
        f"displacement_maps={bool(model.diagnostic_availability.get('displacement_maps'))}",
        f"component_stats={bool(model.diagnostic_availability.get('component_stats'))}",
        f"components={', '.join(components) if components else 'n/a'}",
    ]


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    return (
        "<table><thead><tr>"
        + head_html
        + "</tr></thead><tbody>"
        + "".join(row_html)
        + "</tbody></table>"
    )


def _build_report_html(
    model: SweepVisualizationModel,
    *,
    summary_images: Sequence[str],
    top_rows: Sequence[CandidateVisualizationRow],
    report_stage: str,
    report_top_k: int,
    default_component: str,
    table_files: Sequence[str],
    data_files: Sequence[str],
    report_assets: Sequence[str],
    report_context_path: str,
    dataset_assets: Dict[str, Dict[str, object]],
    robustness_asset: Optional[str],
    parameter_sensitivity_rows: Sequence[ParameterSensitivityRow],
    extension_suggestions: Sequence[ExtensionSuggestion],
) -> str:
    top_table = _html_table(
        ["rank", "candidate", "strategy", "score", "mean score", "runtime (s)", "family", "alpha"],
        [
            [
                str(row.rank),
                row.candidate_id,
                row.strategy,
                _format_float(row.score),
                _format_float(row.mean_dataset_score),
                _format_float(row.mean_runtime_seconds, precision=3),
                row.family or "n/a",
                "n/a" if row.alpha is None else f"{row.alpha:g}",
            ]
            for row in top_rows
        ],
    )
    sensitivity_table = _html_table(
        ["rank", "parameter", "impact", "best", "worst", "ols dR2", "warning"],
        [
            [
                str(row.rank),
                row.parameter_name,
                _format_float(row.impact_score),
                f"{row.best_value or 'n/a'} ({_format_float(row.best_mean_value)})",
                f"{row.worst_value or 'n/a'} ({_format_float(row.worst_mean_value)})",
                _format_float(row.ols_delta_r2),
                row.coverage_warning or "",
            ]
            for row in parameter_sensitivity_rows[:8]
        ],
    )
    suggestions_table = _html_table(
        ["rank", "suggestion", "parameter", "confidence", "evidence"],
        [
            [
                str(suggestion.rank),
                suggestion.recommendation,
                suggestion.parameter_name or "n/a",
                suggestion.confidence,
                suggestion.evidence,
            ]
            for suggestion in extension_suggestions[:8]
        ],
    )
    summary_image_html = "".join(
        (
            f'<figure><img src="{html.escape(path)}" alt="{html.escape(path)}">'
            f"<figcaption>{html.escape(path)}</figcaption></figure>"
        )
        for path in summary_images
    )
    dataset_sections: List[str] = []
    for dataset_id, assets in dataset_assets.items():
        preview_asset = cast(Optional[str], assets.get("preview_sheet"))
        comparison_asset = cast(Optional[str], assets.get("comparison_sheet"))
        diagnostic_assets = cast(Dict[str, str], assets.get("diagnostic_sheets", {}))
        section_parts = [f"<section><h2>Dataset: {html.escape(dataset_id)}</h2>"]
        if preview_asset:
            section_parts.append(
                f'<figure><img src="{html.escape(preview_asset)}" alt="{html.escape(preview_asset)}">'
                f"<figcaption>Preview comparison</figcaption></figure>"
            )
        if comparison_asset:
            section_parts.append(
                f'<figure><img src="{html.escape(comparison_asset)}" alt="{html.escape(comparison_asset)}">'
                f"<figcaption>Main comparison using component {html.escape(default_component)}</figcaption></figure>"
            )
        for component_name, asset_path in diagnostic_assets.items():
            section_parts.append(
                f'<figure><img src="{html.escape(asset_path)}" alt="{html.escape(asset_path)}">'
                f"<figcaption>Diagnostic comparison: {html.escape(component_name)}</figcaption></figure>"
            )
        section_parts.append("</section>")
        dataset_sections.append("".join(section_parts))

    warning_items = "".join(
        f"<li>{html.escape(warning)}</li>" for warning in model.warnings[:20]
    ) or "<li>none</li>"
    links = "".join(
        f'<li><a href="{html.escape(path)}">{html.escape(path)}</a></li>'
        for path in [
            *table_files,
            *data_files,
            report_context_path,
            "visualization_bundle.npz",
            "visualization_bundle.mat",
            *report_assets,
        ]
    )
    robustness_html = (
        f'<figure><img src="{html.escape(robustness_asset)}" alt="{html.escape(robustness_asset)}">'
        "<figcaption>Selected-candidate robustness heatmap</figcaption></figure>"
        if robustness_asset
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sweep Visualization Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    h1, h2 {{ margin-bottom: 0.35em; }}
    section {{ margin-top: 28px; }}
    pre {{ background: #f4f6f8; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
  </style>
</head>
<body>
  <h1>Sweep Visualization Report</h1>
  <p>Stage <strong>{html.escape(report_stage)}</strong>, top {report_top_k} candidates, default diagnostic component <strong>{html.escape(default_component)}</strong>.</p>

  <section>
    <h2>Summary</h2>
    <pre>{html.escape(chr(10).join([
        f"mode={model.mode}",
        f"primary_metric={model.primary_metric}",
        f"primary_metric_source={model.primary_metric_source}",
        f"warning_count={len(model.warnings)}",
        f"datasets={', '.join(_real_datasets(model))}",
    ]))}</pre>
    <pre>{html.escape(chr(10).join(_stage_status_lines(model)))}</pre>
    <pre>{html.escape(chr(10).join(_diagnostic_availability_lines(model)))}</pre>
  </section>

  <section>
    <h2>Top Candidates</h2>
    {top_table}
  </section>

  <section>
    <h2>Parameter Sensitivity</h2>
    {sensitivity_table}
  </section>

  <section>
    <h2>Next Extension Suggestions</h2>
    {suggestions_table}
  </section>

  <section>
    <h2>Summary Figures</h2>
    <div class="grid">{summary_image_html}{robustness_html}</div>
  </section>

  {''.join(dataset_sections)}

  <section>
    <h2>Warnings</h2>
    <ul>{warning_items}</ul>
  </section>

  <section>
    <h2>Files</h2>
    <ul>{links}</ul>
  </section>
</body>
</html>
"""


def export_sweep_visualization_bundle(model: SweepVisualizationModel, output_dir: Path) -> Path:
    """Export default plots plus an aggregated `.npz` / `.mat` reproducibility bundle."""

    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = export_dir / "assets"
    tables_dir = export_dir / "tables"
    data_dir = export_dir / "data"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    generation_scope = model.default_generation_scope
    scoped_candidates = candidate_rows_for_generation_scope(model, generation_scope)
    scoped_run_records = run_records_for_generation_scope(model, generation_scope)
    effects_x, effects_y = _default_export_effect_parameters(model)
    report_stage = model.primary_stage_name
    default_component = _default_component_name(model)
    top_rows = _top_candidates_for_stage(model, report_stage, limit=DEFAULT_REPORT_TOP_K)
    parameter_sensitivity_rows = compute_parameter_sensitivity(
        model,
        stage_name=report_stage,
        dataset_scope="aggregate",
        value_option="score",
        filters={},
    )
    extension_suggestions = generate_next_extension_suggestions(
        model,
        parameter_sensitivity_rows,
    )
    plots: Dict[str, Figure] = {
        "overview.png": create_overview_figure(model),
        "coverage.png": create_coverage_figure(model),
        "progression.png": create_progression_figure(model),
        "significance.png": create_significance_figure(model),
        "sensitivity.png": create_parameter_sensitivity_figure(
            model,
            parameter_sensitivity_rows,
            stage_name=report_stage,
            dataset_scope="aggregate",
            value_option="score",
        ),
    }
    if effects_x is not None:
        plots["effects_trend.png"] = create_effects_figure(
            model,
            mode="1D Trend",
            stage_name=model.primary_stage_name,
            dataset_scope="aggregate",
            x_parameter_name=effects_x,
            value_option="score",
            strategy_filter="All",
            family_filter="All",
            alpha_filter="All",
            parameter_filters={},
        )
    if effects_x is not None and effects_y is not None:
        plots["effects_heatmap.png"] = create_effects_figure(
            model,
            mode="2D Heatmap",
            stage_name=model.primary_stage_name,
            dataset_scope="aggregate",
            x_parameter_name=effects_x,
            y_parameter_name=effects_y,
            value_option="score",
            strategy_filter="All",
            family_filter="All",
            alpha_filter="All",
            parameter_filters={},
        )
    metric_fig = create_metric_formulation_figure(model)
    if metric_fig is not None:
        plots["metric_formulations.png"] = metric_fig
    diagnostics_record = _default_metric_diagnostics_record(model)
    diagnostics_fig = create_metric_diagnostics_figure(
        diagnostics_record,
        component_name=(
            cast(List[str], model.diagnostic_availability.get("components", []))[0]
            if cast(List[str], model.diagnostic_availability.get("components", []))
            else "gradient"
        ),
    )
    if diagnostics_fig is not None:
        plots["metric_diagnostics.png"] = diagnostics_fig

    plot_files: List[str] = []
    for filename, figure in plots.items():
        figure.savefig(export_dir / filename, dpi=180)
        plot_files.append(filename)
        figure.clf()

    report_assets: List[str] = []
    topk_overview_rel = "assets/topk_overview.png"
    create_significance_figure(model, top_n=max(1, len(top_rows))).savefig(
        export_dir / topk_overview_rel,
        dpi=180,
    )
    report_assets.append(topk_overview_rel)

    dataset_assets: Dict[str, Dict[str, object]] = {}
    diagnostic_components = cast(List[str], model.diagnostic_availability.get("components", []))
    diagnostic_components = diagnostic_components or [default_component]
    for dataset_id in _real_datasets(model):
        dataset_slug = _report_slug(dataset_id)
        preview_rel = f"assets/preview_{dataset_slug}.png"
        create_preview_comparison_figure(model, top_rows, dataset_id=dataset_id).savefig(
            export_dir / preview_rel,
            dpi=180,
        )
        report_assets.append(preview_rel)

        comparison_rel = f"assets/comparison_{dataset_slug}.png"
        create_comparison_figure(
            model,
            top_rows,
            dataset_id=dataset_id,
            component_name=default_component,
        ).savefig(export_dir / comparison_rel, dpi=180)
        report_assets.append(comparison_rel)

        per_component_assets: Dict[str, str] = {}
        for component_name in diagnostic_components:
            component_slug = _report_slug(component_name)
            diagnostic_rel = f"assets/diagnostic_{dataset_slug}_{component_slug}.png"
            create_comparison_figure(
                model,
                top_rows,
                dataset_id=dataset_id,
                component_name=component_name,
            ).savefig(export_dir / diagnostic_rel, dpi=180)
            report_assets.append(diagnostic_rel)
            per_component_assets[component_name] = diagnostic_rel

        dataset_assets[dataset_id] = {
            "preview_sheet": preview_rel,
            "comparison_sheet": comparison_rel,
            "diagnostic_sheets": per_component_assets,
        }

    robustness_rel: Optional[str] = None
    if top_rows:
        robustness_rel = "assets/robustness_heatmap.png"
        create_candidate_robustness_figure(model, top_rows).savefig(
            export_dir / robustness_rel,
            dpi=180,
        )
        report_assets.append(robustness_rel)

    table_files = [
        "tables/candidate_leaderboard.csv",
        "tables/run_metric_records.csv",
        "tables/parameter_sensitivity.csv",
    ]
    _write_candidate_leaderboard_csv(model, export_dir / table_files[0])
    _write_run_metric_records_csv(model, export_dir / table_files[1])
    _write_parameter_sensitivity_csv(parameter_sensitivity_rows, export_dir / table_files[2])

    parameter_sensitivity_rel = "data/parameter_sensitivity.json"
    next_extension_suggestions_rel = "data/next_extension_suggestions.json"
    write_json_atomic(
        export_dir / parameter_sensitivity_rel,
        [_parameter_sensitivity_to_export_dict(row) for row in parameter_sensitivity_rows],
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    write_json_atomic(
        export_dir / next_extension_suggestions_rel,
        [_extension_suggestion_to_export_dict(suggestion) for suggestion in extension_suggestions],
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    report_context = {
        "generated_at": _utc_now_iso(),
        "generation_scope": generation_scope,
        "report_stage": report_stage,
        "report_top_k": DEFAULT_REPORT_TOP_K,
        "default_component": default_component,
        "summary_images": [*plot_files, topk_overview_rel],
        "report_assets": report_assets,
        "table_files": table_files,
        "data_files": [parameter_sensitivity_rel, next_extension_suggestions_rel],
        "top_candidates": [_candidate_row_to_export_dict(row) for row in top_rows],
        "parameter_sensitivity": [
            _parameter_sensitivity_to_export_dict(row) for row in parameter_sensitivity_rows
        ],
        "next_extension_suggestions": [
            _extension_suggestion_to_export_dict(suggestion)
            for suggestion in extension_suggestions
        ],
        "dataset_assets": dataset_assets,
        "generation_summary": model.generation_summary,
        "warnings": model.warnings,
    }
    report_context_rel = "data/report_context.json"
    write_json_atomic(
        export_dir / report_context_rel,
        report_context,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    index_html_rel = "index.html"
    (export_dir / index_html_rel).write_text(
        _build_report_html(
            model,
            summary_images=[*plot_files, topk_overview_rel],
            top_rows=top_rows,
            report_stage=report_stage,
            report_top_k=DEFAULT_REPORT_TOP_K,
            default_component=default_component,
            table_files=table_files,
            data_files=[parameter_sensitivity_rel, next_extension_suggestions_rel],
            report_assets=report_assets,
            report_context_path=report_context_rel,
            dataset_assets=dataset_assets,
            robustness_asset=robustness_rel,
            parameter_sensitivity_rows=parameter_sensitivity_rows,
            extension_suggestions=extension_suggestions,
        ),
        encoding="utf-8",
    )

    stage_names = model.stage_order
    dataset_index = {dataset_id: idx for idx, dataset_id in enumerate(model.datasets)}
    metric_index = {metric_name: idx for idx, metric_name in enumerate(model.metric_names)}
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    scoped_score_matrix = np.full((len(scoped_candidates), len(model.datasets)), np.nan, dtype=np.float64)
    scoped_runtime_matrix = np.full((len(scoped_candidates), len(model.datasets)), np.nan, dtype=np.float64)
    scoped_metric_tensor = np.full(
        (len(scoped_candidates), len(model.datasets), len(model.metric_names)),
        np.nan,
        dtype=np.float64,
    )
    scoped_parameter_numeric_matrix = np.full(
        (len(scoped_candidates), len(model.parameter_names)),
        np.nan,
        dtype=np.float64,
    )
    scoped_parameter_text_matrix = np.empty(
        (len(scoped_candidates), len(model.parameter_names)),
        dtype=object,
    )
    scoped_parameter_text_matrix[:] = ""
    for row_idx, row in enumerate(scoped_candidates):
        for dataset_id, value in row.dataset_scores.items():
            if dataset_id in dataset_index and value is not None:
                scoped_score_matrix[row_idx, dataset_index[dataset_id]] = float(value)
        for dataset_id, value in row.dataset_runtimes.items():
            if dataset_id in dataset_index and value is not None:
                scoped_runtime_matrix[row_idx, dataset_index[dataset_id]] = float(value)
        for dataset_id, metrics in row.dataset_metrics.items():
            if dataset_id not in dataset_index:
                continue
            for metric_name, value in metrics.items():
                if metric_name in metric_index and value is not None:
                    scoped_metric_tensor[row_idx, dataset_index[dataset_id], metric_index[metric_name]] = float(value)
        for parameter_name, param_value in row.params.items():
            if parameter_name not in parameter_index:
                continue
            param_idx = parameter_index[parameter_name]
            scoped_parameter_numeric_matrix[row_idx, param_idx] = _numeric_param(param_value)
            scoped_parameter_text_matrix[row_idx, param_idx] = _stringify_param(param_value)
    bundle_payload: Dict[str, object] = {
        "generation_scope": np.asarray(generation_scope, dtype=object),
        "candidate_ids": np.asarray([row.row_id for row in scoped_candidates], dtype=object),
        "candidate_labels": np.asarray([row.display_name for row in scoped_candidates], dtype=object),
        "candidate_stages": np.asarray([row.stage for row in scoped_candidates], dtype=object),
        "candidate_strategies": np.asarray([row.strategy for row in scoped_candidates], dtype=object),
        "candidate_ranks": np.asarray([row.rank for row in scoped_candidates], dtype=np.int32),
        "candidate_generation_ids": np.asarray(
            [row.generation_id or "" for row in scoped_candidates],
            dtype=object,
        ),
        "candidate_scores": np.asarray(
            [row.score if row.score is not None else np.nan for row in scoped_candidates],
            dtype=np.float64,
        ),
        "candidate_mean_dataset_scores": np.asarray(
            [
                row.mean_dataset_score if row.mean_dataset_score is not None else np.nan
                for row in scoped_candidates
            ],
            dtype=np.float64,
        ),
        "candidate_std_dataset_scores": np.asarray(
            [
                row.std_dataset_score if row.std_dataset_score is not None else np.nan
                for row in scoped_candidates
            ],
            dtype=np.float64,
        ),
        "dataset_ids": np.asarray(model.datasets, dtype=object),
        "metric_names": np.asarray(model.metric_names, dtype=object),
        "parameter_names": np.asarray(model.parameter_names, dtype=object),
        "score_matrix": scoped_score_matrix,
        "runtime_matrix": scoped_runtime_matrix,
        "metric_tensor": scoped_metric_tensor,
        "parameter_numeric_matrix": scoped_parameter_numeric_matrix,
        "parameter_text_matrix": np.asarray(scoped_parameter_text_matrix, dtype=object),
        "stage_names": np.asarray(stage_names, dtype=object),
        "stage_planned_runs": np.asarray(
            [model.stages[name].num_runs_planned for name in stage_names],
            dtype=np.int32,
        ),
        "stage_recorded_runs": np.asarray(
            [model.stages[name].num_runs_recorded for name in stage_names],
            dtype=np.int32,
        ),
        "stage_successful_runs": np.asarray(
            [model.stages[name].num_successful for name in stage_names],
            dtype=np.int32,
        ),
        "stage_failed_runs": np.asarray(
            [model.stages[name].num_failed for name in stage_names],
            dtype=np.int32,
        ),
        "stage_pending_runs": np.asarray(
            [model.stages[name].num_pending for name in stage_names],
            dtype=np.int32,
        ),
        "stage_running_runs": np.asarray(
            [model.stages[name].num_running for name in stage_names],
            dtype=np.int32,
        ),
        "alpha_family_summary_json": np.asarray(
            [_compact_json(item) for item in model.alpha_family_summary],
            dtype=object,
        ),
        "alpha_ranked_runs_json": np.asarray(
            [_compact_json(item) for item in model.alpha_ranked_runs],
            dtype=object,
        ),
        "candidate_table_json": np.asarray(
            [_compact_json(_candidate_row_to_export_dict(row)) for row in scoped_candidates],
            dtype=object,
        ),
        "run_metric_records_json": np.asarray(
            [_compact_json(_run_metric_record_to_export_dict(record)) for record in scoped_run_records],
            dtype=object,
        ),
        "parameter_sensitivity_json": np.asarray(
            [_compact_json(_parameter_sensitivity_to_export_dict(row)) for row in parameter_sensitivity_rows],
            dtype=object,
        ),
        "next_extension_suggestions_json": np.asarray(
            [
                _compact_json(_extension_suggestion_to_export_dict(suggestion))
                for suggestion in extension_suggestions
            ],
            dtype=object,
        ),
        "warnings": np.asarray(model.warnings, dtype=object),
        "provenance_json": np.asarray(_compact_json(model.provenance), dtype=object),
        "experiment_metadata_json": np.asarray(
            _compact_json(model.experiment_metadata),
            dtype=object,
        ),
        "generation_summary_json": np.asarray(
            _compact_json(model.generation_summary),
            dtype=object,
        ),
        "diagnostic_availability_json": np.asarray(
            _compact_json(model.diagnostic_availability),
            dtype=object,
        ),
        "primary_metric_source": np.asarray(model.primary_metric_source, dtype=object),
    }
    save_workspace_variables(str(export_dir / "visualization_bundle.npz"), bundle_payload)

    manifest = {
        "generated_at": _utc_now_iso(),
        "source_dir": str(model.source_dir.resolve()),
        "mode": model.mode,
        "primary_stage_name": model.primary_stage_name,
        "primary_metric": model.primary_metric,
        "primary_metric_source": model.primary_metric_source,
        "plot_files": plot_files,
        "bundle_npz": "visualization_bundle.npz",
        "bundle_mat": "visualization_bundle.mat",
        "warning_count": len(model.warnings),
        "index_html": index_html_rel,
        "report_top_k": DEFAULT_REPORT_TOP_K,
        "default_component": default_component,
        "table_files": table_files,
        "data_files": [parameter_sensitivity_rel, next_extension_suggestions_rel, report_context_rel],
        "report_assets": report_assets,
        "active_generation_id": model.generation_summary.get("active_generation_id"),
        "previous_generation_id": model.generation_summary.get("previous_generation_id"),
    }
    write_json_atomic(
        export_dir / "visualization_manifest.json",
        manifest,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    return export_dir


def _format_overview_text(model: SweepVisualizationModel) -> str:
    lines = [
        f"mode: {model.mode}",
        f"source: {model.source_dir}",
        f"scope: {model.path_scope}",
        f"experiment root: {model.experiment_root}",
        f"datasets ({len(model.datasets)}): {', '.join(model.datasets) if model.datasets else 'n/a'}",
        f"strategies: {', '.join(model.strategies) if model.strategies else 'n/a'}",
        f"primary stage: {model.primary_stage_name}",
        f"primary metric: {model.primary_metric} ({model.primary_metric_source})",
        f"run metric records: {len(model.run_metric_records)}",
        f"active generation: {model.generation_summary.get('active_generation_id') or 'n/a'}",
        f"previous generation: {model.generation_summary.get('previous_generation_id') or 'n/a'}",
        "",
        "stages:",
    ]
    for stage_name in model.stage_order:
        stage = model.stages[stage_name]
        lines.append(
            f"  {stage_name}: planned={stage.num_runs_planned} successful={stage.num_successful} "
            f"failed={stage.num_failed} pending={stage.num_pending} running={stage.num_running}"
        )
    lines.extend(
        [
            "",
            "diagnostics:",
            f"  score maps: {bool(model.diagnostic_availability.get('score_maps'))}",
            f"  displacement maps: {bool(model.diagnostic_availability.get('displacement_maps'))}",
            f"  component stats: {bool(model.diagnostic_availability.get('component_stats'))}",
            "  components: "
            + ", ".join(cast(List[str], model.diagnostic_availability.get("components", [])) or ["n/a"]),
        ]
    )
    if model.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"  - {warning}" for warning in model.warnings[:12])
    return "\n".join(lines)


def _format_effects_text(
    model: SweepVisualizationModel,
    *,
    stage_name: str,
    dataset_scope: str,
    mode: str,
    x_parameter_name: Optional[str],
    y_parameter_name: Optional[str],
    value_option: str,
    strategy_filter: str,
    family_filter: str,
    alpha_filter: str,
    parameter_filters: Dict[str, str],
) -> str:
    base_rows = (
        _candidate_rows_to_effect_rows(candidate_rows_for_generation_scope(model, model.default_generation_scope))
        if dataset_scope == "aggregate"
        else _run_records_to_effect_rows(run_records_for_generation_scope(model, model.default_generation_scope))
    )
    filtered = _filter_effect_rows(
        base_rows,
        stage_name=stage_name,
        strategy=strategy_filter,
        family=family_filter,
        alpha=alpha_filter,
        dataset_scope=dataset_scope,
        parameter_filters=parameter_filters,
        ignored_params=[param for param in (x_parameter_name, y_parameter_name) if param],
    )
    lines = [
        f"stage: {stage_name}",
        f"dataset scope: {dataset_scope}",
        f"mode: {mode}",
        f"value: {value_option}",
        f"rows: {len(filtered)}",
        f"x parameter: {x_parameter_name or 'n/a'}",
    ]
    if mode == "2D Heatmap":
        lines.append(f"y parameter: {y_parameter_name or 'n/a'}")
    lines.extend(
        [
            f"strategy filter: {strategy_filter}",
            f"family filter: {family_filter}",
            f"alpha filter: {alpha_filter}",
        ]
    )
    active_filters = [
        f"{name}={value}"
        for name, value in sorted(parameter_filters.items(), key=lambda item: item[0])
        if value != "All"
    ]
    lines.append("parameter filters: " + (", ".join(active_filters) if active_filters else "none"))
    lines.append("")
    lines.append("parameter coverage:")
    for parameter_name in model.parameter_names:
        values = Counter(
            _stringify_param(row.params.get(parameter_name))
            for row in filtered
            if parameter_name in row.params
        )
        if not values:
            continue
        rendered = ", ".join(
            f"{label} ({count})"
            for label, count in sorted(values.items(), key=lambda item: item[0])
        )
        lines.append(f"  {parameter_name}: {rendered}")
    return "\n".join(lines)


def _format_significance_text(model: SweepVisualizationModel) -> str:
    rows = _rows_for_stage(model, model.primary_stage_name)
    if not rows:
        return "No ranked rows available."
    lines: List[str] = []
    best = rows[0]
    lines.append(f"best candidate: {best.candidate_id}")
    if best.score is not None:
        lines.append(f"best score: {best.score:.6f}")
    if len(rows) > 1 and best.score is not None and rows[1].score is not None:
        gap = best.score - cast(float, rows[1].score)
        lines.append(f"gap to runner-up: {gap:+.6f}")
    if best.mean_runtime_seconds is not None:
        lines.append(f"mean runtime: {best.mean_runtime_seconds:.3f}s")
    if best.metric_means:
        lines.append("metric means:")
        for metric_name in model.metric_names[:8]:
            if metric_name in best.metric_means:
                lines.append(f"  {metric_name}: {best.metric_means[metric_name]:.4f}")
    return "\n".join(lines)


def _format_metric_summary_text(model: SweepVisualizationModel) -> str:
    lines = [
        "family, alpha, count, mean_score, primary_metric_mean, mean_runtime_seconds, mean_displacement_magnitude, mean_view_std"
    ]
    lines.extend(
        ", ".join(
            [
                str(item.get("family")),
                "n/a" if item.get("alpha") is None else f"{float(item.get('alpha', 0.0)):.3f}",
                str(item.get("count")),
                str(item.get("mean_score")),
                str(item.get("primary_metric_mean")),
                str(item.get("mean_runtime_seconds")),
                str(item.get("mean_displacement_magnitude")),
                str(item.get("mean_view_std")),
            ]
        )
        for item in model.alpha_family_summary
    )
    if model.alpha_ranked_runs:
        lines.extend(["", "ranked runs:"])
        for row in model.alpha_ranked_runs[:8]:
            lines.append(
                ", ".join(
                    [
                        str(row.get("rank", "")),
                        str(row.get("family", "")),
                        str(row.get("alpha", "")),
                        str(row.get("ranking_metric_name", "")),
                        str(row.get("ranking_metric_value", "")),
                    ]
                )
            )
    return "\n".join(lines)


def _format_metric_diagnostics_text(record: Optional[RunMetricRecord]) -> str:
    if record is None:
        return "No diagnostic run record available."
    lines = [
        f"run: {record.run_name}",
        f"stage: {record.stage}",
        f"candidate: {record.candidate_id}",
        f"dataset: {record.dataset_id}",
        f"strategy: {record.strategy}",
        f"family: {record.family or 'n/a'}",
        f"alpha: {'n/a' if record.alpha is None else f'{record.alpha:g}'}",
        f"preview: {record.preview_path or 'n/a'}",
        f"workspace: {record.workspace_path or 'n/a'}",
        f"metric_record: {record.metric_record_path or 'n/a'}",
    ]
    if record.metric_means:
        lines.extend(["", "image metrics:"])
        for key in sorted(record.metric_means.keys())[:6]:
            lines.append(f"  {key}: {record.metric_means[key]:.4f}")
    if record.displacement_field_metrics:
        lines.extend(["", "displacement field metrics:"])
        for key in sorted(record.displacement_field_metrics.keys())[:6]:
            lines.append(f"  {key}: {record.displacement_field_metrics[key]:.4f}")
    if record.inter_view_consistency_metrics:
        lines.extend(["", "inter-view consistency:"])
        for key in sorted(record.inter_view_consistency_metrics.keys())[:6]:
            lines.append(f"  {key}: {record.inter_view_consistency_metrics[key]:.4f}")
    diagnostics = record.displacement_score_diagnostics
    if diagnostics:
        lines.extend(["", "diagnostic keys:"])
        lines.extend(f"  {key}" for key in sorted(diagnostics.keys()))
    return "\n".join(lines)


def _format_parameter_sensitivity_text(rows: Sequence[ParameterSensitivityRow]) -> str:
    if not rows:
        return "No parameter sensitivity rows available for the selected filters."
    first = rows[0]
    lines = [
        f"stage: {first.stage}",
        f"dataset scope: {first.dataset_scope}",
        f"value: {first.value_option}",
        f"parameters ranked: {len(rows)}",
        "",
    ]
    for row in rows[:8]:
        lines.append(
            f"#{row.rank} {row.parameter_name}: impact={_format_float(row.impact_score)}, "
            f"best={row.best_value or 'n/a'} ({_format_float(row.best_mean_value)}), "
            f"worst={row.worst_value or 'n/a'} ({_format_float(row.worst_mean_value)}), "
            f"ols={_format_float(row.ols_delta_r2)} [{row.ols_status}]"
        )
        if row.coverage_warning:
            lines.append(f"  warning: {row.coverage_warning}")
        if row.previous_impact_score is not None:
            lines.append(
                f"  previous generation: impact={_format_float(row.previous_impact_score)}, "
                f"best={row.previous_best_value or 'n/a'}, delta={_format_float(row.generation_delta)}"
            )
    return "\n".join(lines)


def _format_extension_suggestions_text(suggestions: Sequence[ExtensionSuggestion]) -> str:
    if not suggestions:
        return "No extension suggestions available."
    lines: List[str] = []
    for suggestion in suggestions:
        lines.extend(
            [
                f"#{suggestion.rank} {suggestion.title}",
                f"parameter: {suggestion.parameter_name or 'n/a'}",
                f"confidence: {suggestion.confidence}",
                f"recommendation: {suggestion.recommendation}",
                f"evidence: {suggestion.evidence}",
                "",
            ]
        )
    return "\n".join(lines).strip()


class _RectificationImageInspector:
    """Interactive zoomable image panel for one rectification result."""

    def __init__(
        self,
        app: "SweepVisualizationApp",
        parent: Any,
        *,
        slot_name: str,
        on_channel_changed: Optional[Callable[["_RectificationImageInspector", str], None]] = None,
        on_limits_changed: Optional[Callable[["_RectificationImageInspector"], None]] = None,
    ) -> None:
        self.app = app
        self.slot_name = slot_name
        self.on_channel_changed = on_channel_changed
        self.on_limits_changed = on_limits_changed
        self.bundle: Optional[RectificationDisplayBundle] = None
        self._last_rendered_shape: Optional[Tuple[int, int]] = None
        self._suspend_limit_events = False

        self.frame = self.app.ttk.LabelFrame(parent, text=f"Result {slot_name}")
        controls = self.app.ttk.Frame(self.frame)
        controls.pack(fill="x", padx=6, pady=4)
        self.app.ttk.Label(controls, text="Channel").pack(side="left")
        self.channel_var = self.app.tk.StringVar(value=RECTIFICATION_VIEW_LABELS["preview"])
        self.channel_combo = self.app.ttk.Combobox(
            controls,
            textvariable=self.channel_var,
            state="readonly",
            values=[RECTIFICATION_VIEW_LABELS[name] for name in RECTIFICATION_VIEW_CHANNELS],
            width=24,
        )
        self.channel_combo.pack(side="left", padx=4)
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _evt: self._handle_channel_changed())
        self.app.ttk.Button(controls, text="Fit", command=self.fit_view).pack(side="left", padx=4)
        self.status_var = self.app.tk.StringVar(value="not set")
        self.app.ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=8)

        self.toolbar_frame = self.app.ttk.Frame(self.frame)
        self.toolbar_frame.pack(fill="x", padx=6, pady=(0, 2))
        self.canvas_frame = self.app.ttk.Frame(self.frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.figure = Figure(figsize=(6.2, 5.8))
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
        self._render_placeholder(f"Result {slot_name} is not set.")

    def current_channel(self) -> str:
        label = self.channel_var.get()
        for channel_name, channel_label in RECTIFICATION_VIEW_LABELS.items():
            if label == channel_label or label == channel_name:
                return channel_name
        return "preview"

    def set_channel(self, channel_name: str, *, emit_event: bool = False, reset_view: bool = False) -> None:
        if channel_name not in RECTIFICATION_VIEW_CHANNELS:
            channel_name = "preview"
        self.channel_var.set(RECTIFICATION_VIEW_LABELS[channel_name])
        self.render(reset_view=reset_view)
        if emit_event and self.on_channel_changed is not None:
            self.on_channel_changed(self, channel_name)

    def current_image(self) -> Optional[np.ndarray]:
        if self.bundle is None:
            return None
        image = self.bundle.get(self.current_channel())
        if isinstance(image, np.ndarray) and image.ndim == 2 and image.size:
            return np.asarray(image, dtype=np.float64)
        return None

    def has_image(self) -> bool:
        return self.current_image() is not None

    def set_bundle(
        self,
        bundle: Optional[RectificationDisplayBundle],
        *,
        reset_view: bool = True,
    ) -> None:
        previous_key = None if self.bundle is None else self.bundle.selection.key
        self.bundle = bundle
        if bundle is None:
            self.status_var.set("not set")
            self._render_placeholder(f"Result {self.slot_name} is not set.")
            return
        self.status_var.set(
            f"{bundle.selection.candidate_id} / {bundle.selection.dataset_id} / {bundle.selection.stage}"
        )
        self.render(reset_view=reset_view or bundle.selection.key != previous_key)

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

    def clone_view_state_from(self, other: "_RectificationImageInspector") -> None:
        self.set_channel(other.current_channel(), emit_event=False, reset_view=False)
        xlim, ylim = other.get_axes_limits()
        if xlim is not None and ylim is not None and self.has_image():
            self.apply_axes_limits(xlim, ylim)

    def fit_view(self) -> None:
        image = self.current_image()
        if image is None:
            return
        height, width = image.shape
        self._suspend_limit_events = True
        try:
            self.axis.set_xlim(-0.5, width - 0.5)
            self.axis.set_ylim(height - 0.5, -0.5)
        finally:
            self._suspend_limit_events = False
        self.canvas.draw_idle()
        self._notify_limits_changed()

    def render(self, *, reset_view: bool) -> None:
        image = self.current_image()
        previous_shape = self._last_rendered_shape
        previous_limits = self.get_axes_limits()
        channel_name = self.current_channel()
        title = RECTIFICATION_VIEW_LABELS.get(channel_name, channel_name)
        if image is None:
            self._render_placeholder(f"{title} unavailable.")
            return
        self.axis.clear()
        self.axis.imshow(
            image,
            cmap=RECTIFICATION_VIEW_CMAPS.get(channel_name, "gray"),
            aspect="equal",
            interpolation="nearest",
        )
        self.axis.set_title(title)
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self._last_rendered_shape = (int(image.shape[0]), int(image.shape[1]))
        if reset_view or previous_limits[0] is None or previous_shape != image.shape:
            self.fit_view()
        else:
            self.apply_axes_limits(
                cast(Tuple[float, float], previous_limits[0]),
                cast(Tuple[float, float], previous_limits[1]),
            )
        self.canvas.draw_idle()

    def _render_placeholder(self, text: str) -> None:
        self.axis.clear()
        self.axis.text(0.5, 0.5, text, ha="center", va="center")
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_title(RECTIFICATION_VIEW_LABELS.get(self.current_channel(), self.current_channel()))
        self._last_rendered_shape = None
        self.canvas.draw_idle()

    def _handle_channel_changed(self) -> None:
        channel_name = self.current_channel()
        self.render(reset_view=False)
        if self.on_channel_changed is not None:
            self.on_channel_changed(self, channel_name)

    def _notify_limits_changed(self) -> None:
        if self._suspend_limit_events or self.on_limits_changed is None:
            return
        self.on_limits_changed(self)


class SweepVisualizationApp:
    """Tkinter GUI for loading, browsing, previewing, and exporting sweep results."""

    def __init__(self, root: Any, initial_path: Optional[Path] = None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = __import__("tkinter.filedialog", fromlist=["filedialog"])
        self.messagebox = __import__("tkinter.messagebox", fromlist=["messagebox"])
        self.FigureCanvasTkAgg = __import__(
            "matplotlib.backends.backend_tkagg",
            fromlist=["FigureCanvasTkAgg"],
        ).FigureCanvasTkAgg

        self.root.title("Sweep Results Visualizer")
        self.root.geometry("1280x960")
        self.model: Optional[SweepVisualizationModel] = None
        self._canvases: Dict[str, Any] = {}
        self._preview_image_ref: Any = None
        self._metric_tab_visible = False
        self._effect_filter_vars: Dict[str, Any] = {}
        self._effect_filter_combos: Dict[str, Any] = {}

        self.path_var = tk.StringVar(value=str(initial_path) if initial_path else "")
        self.status_var = tk.StringVar(value="Idle")
        self.coverage_stage_var = tk.StringVar(value="")
        self.coverage_dataset_var = tk.StringVar(value="aggregate")
        self.coverage_mode_var = tk.StringVar(value="1D Trend")
        self.coverage_param_var = tk.StringVar(value="")
        self.coverage_y_param_var = tk.StringVar(value="")
        self.coverage_value_var = tk.StringVar(value="score")
        self.coverage_strategy_var = tk.StringVar(value="All")
        self.coverage_family_var = tk.StringVar(value="All")
        self.coverage_alpha_var = tk.StringVar(value="All")
        self.preview_stage_var = tk.StringVar(value="")
        self.preview_candidate_var = tk.StringVar(value="")
        self.preview_dataset_var = tk.StringVar(value="")
        self.diagnostic_stage_var = tk.StringVar(value="")
        self.diagnostic_candidate_var = tk.StringVar(value="")
        self.diagnostic_dataset_var = tk.StringVar(value="")
        self.diagnostic_component_var = tk.StringVar(value="gradient")

        self._build_ui()
        if initial_path is not None:
            self.load_from_path(Path(initial_path))

    def _build_ui(self) -> None:
        top = self.ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        self.ttk.Label(top, text="Sweep folder").pack(side="left")
        path_entry = self.ttk.Entry(top, textvariable=self.path_var, width=90)
        path_entry.pack(side="left", padx=6, fill="x", expand=True)
        self.ttk.Button(top, text="Browse...", command=self._on_browse).pack(side="left", padx=3)
        self.ttk.Button(top, text="Load", command=self._on_load).pack(side="left", padx=3)
        self.ttk.Button(top, text="Reload", command=self._on_reload).pack(side="left", padx=3)
        self.ttk.Button(top, text="Export...", command=self._on_export_dialog).pack(side="left", padx=3)
        self.ttk.Button(top, text="Open Folder", command=self._on_open_folder).pack(side="left", padx=3)
        self.ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

        self.notebook = self.ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

        self.overview_tab = self.ttk.Frame(self.notebook)
        self.coverage_tab = self.ttk.Frame(self.notebook)
        self.progression_tab = self.ttk.Frame(self.notebook)
        self.significance_tab = self.ttk.Frame(self.notebook)
        self.metric_tab = self.ttk.Frame(self.notebook)
        self.diagnostics_tab = self.ttk.Frame(self.notebook)

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.coverage_tab, text="Effects")
        self.notebook.add(self.progression_tab, text="Progression")
        self.notebook.add(self.significance_tab, text="Significance")
        self.notebook.add(self.diagnostics_tab, text="Metric Diagnostics")

        self._build_overview_tab()
        self._build_coverage_tab()
        self._build_progression_tab()
        self._build_significance_tab()
        self._build_metric_tab()
        self._build_diagnostics_tab()

    def _build_overview_tab(self) -> None:
        self.overview_text = self.tk.Text(self.overview_tab, height=14, wrap="word")
        self.overview_text.pack(fill="x", padx=6, pady=6)

        preview_frame = self.ttk.Frame(self.overview_tab)
        preview_frame.pack(fill="x", padx=6, pady=4)
        self.ttk.Label(preview_frame, text="Preview stage").pack(side="left")
        self.preview_stage_combo = self.ttk.Combobox(
            preview_frame,
            textvariable=self.preview_stage_var,
            state="readonly",
            width=14,
        )
        self.preview_stage_combo.pack(side="left", padx=4)
        self.preview_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_preview_candidates())
        self.ttk.Label(preview_frame, text="Candidate").pack(side="left", padx=(10, 0))
        self.preview_candidate_combo = self.ttk.Combobox(
            preview_frame,
            textvariable=self.preview_candidate_var,
            state="readonly",
            width=22,
        )
        self.preview_candidate_combo.pack(side="left", padx=4)
        self.preview_candidate_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_preview_datasets())
        self.ttk.Label(preview_frame, text="Dataset").pack(side="left", padx=(10, 0))
        self.preview_dataset_combo = self.ttk.Combobox(
            preview_frame,
            textvariable=self.preview_dataset_var,
            state="readonly",
            width=22,
        )
        self.preview_dataset_combo.pack(side="left", padx=4)
        self.ttk.Button(preview_frame, text="Open Preview", command=self._open_preview).pack(side="left", padx=8)

        self.overview_plot_frame = self.ttk.Frame(self.overview_tab)
        self.overview_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_coverage_tab(self) -> None:
        controls = self.ttk.Frame(self.coverage_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="Stage").pack(side="left")
        self.coverage_stage_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_stage_var,
            state="readonly",
            width=14,
        )
        self.coverage_stage_combo.pack(side="left", padx=4)
        self.coverage_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(controls, text="Dataset").pack(side="left", padx=(8, 0))
        self.coverage_dataset_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_dataset_var,
            state="readonly",
            width=12,
        )
        self.coverage_dataset_combo.pack(side="left", padx=4)
        self.coverage_dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(controls, text="Mode").pack(side="left", padx=(8, 0))
        self.coverage_mode_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_mode_var,
            state="readonly",
            width=12,
            values=["1D Trend", "2D Heatmap"],
        )
        self.coverage_mode_combo.pack(side="left", padx=4)
        self.coverage_mode_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(controls, text="X").pack(side="left", padx=(8, 0))
        self.coverage_param_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_param_var,
            state="readonly",
            width=18,
        )
        self.coverage_param_combo.pack(side="left", padx=4)
        self.coverage_param_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(controls, text="Y").pack(side="left", padx=(8, 0))
        self.coverage_y_param_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_y_param_var,
            state="readonly",
            width=18,
        )
        self.coverage_y_param_combo.pack(side="left", padx=4)
        self.coverage_y_param_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(controls, text="Value").pack(side="left", padx=(8, 0))
        self.coverage_value_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_value_var,
            state="readonly",
            width=22,
        )
        self.coverage_value_combo.pack(side="left", padx=4)
        self.coverage_value_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())

        filters = self.ttk.Frame(self.coverage_tab)
        filters.pack(fill="x", padx=6, pady=4)
        self.ttk.Label(filters, text="Strategy").pack(side="left")
        self.coverage_strategy_combo = self.ttk.Combobox(
            filters,
            textvariable=self.coverage_strategy_var,
            state="readonly",
            width=12,
        )
        self.coverage_strategy_combo.pack(side="left", padx=4)
        self.coverage_strategy_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(filters, text="Family").pack(side="left", padx=(8, 0))
        self.coverage_family_combo = self.ttk.Combobox(
            filters,
            textvariable=self.coverage_family_var,
            state="readonly",
            width=16,
        )
        self.coverage_family_combo.pack(side="left", padx=4)
        self.coverage_family_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
        self.ttk.Label(filters, text="Alpha").pack(side="left", padx=(8, 0))
        self.coverage_alpha_combo = self.ttk.Combobox(
            filters,
            textvariable=self.coverage_alpha_var,
            state="readonly",
            width=10,
        )
        self.coverage_alpha_combo.pack(side="left", padx=4)
        self.coverage_alpha_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())

        self.coverage_filter_frame = self.ttk.Frame(self.coverage_tab)
        self.coverage_filter_frame.pack(fill="x", padx=6, pady=4)

        self.coverage_text = self.tk.Text(self.coverage_tab, height=12, wrap="word")
        self.coverage_text.pack(fill="x", padx=6, pady=4)
        self.coverage_plot_frame = self.ttk.Frame(self.coverage_tab)
        self.coverage_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_progression_tab(self) -> None:
        self.progression_text = self.tk.Text(self.progression_tab, height=12, wrap="word")
        self.progression_text.pack(fill="x", padx=6, pady=6)
        self.progression_plot_frame = self.ttk.Frame(self.progression_tab)
        self.progression_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_significance_tab(self) -> None:
        self.significance_text = self.tk.Text(self.significance_tab, height=12, wrap="word")
        self.significance_text.pack(fill="x", padx=6, pady=6)
        self.significance_plot_frame = self.ttk.Frame(self.significance_tab)
        self.significance_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_metric_tab(self) -> None:
        self.metric_text = self.tk.Text(self.metric_tab, height=12, wrap="word")
        self.metric_text.pack(fill="x", padx=6, pady=6)
        self.metric_plot_frame = self.ttk.Frame(self.metric_tab)
        self.metric_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_diagnostics_tab(self) -> None:
        controls = self.ttk.Frame(self.diagnostics_tab)
        controls.pack(fill="x", padx=6, pady=6)
        self.ttk.Label(controls, text="Stage").pack(side="left")
        self.diagnostic_stage_combo = self.ttk.Combobox(
            controls,
            textvariable=self.diagnostic_stage_var,
            state="readonly",
            width=14,
        )
        self.diagnostic_stage_combo.pack(side="left", padx=4)
        self.diagnostic_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_diagnostic_candidates())
        self.ttk.Label(controls, text="Candidate").pack(side="left", padx=(8, 0))
        self.diagnostic_candidate_combo = self.ttk.Combobox(
            controls,
            textvariable=self.diagnostic_candidate_var,
            state="readonly",
            width=20,
        )
        self.diagnostic_candidate_combo.pack(side="left", padx=4)
        self.diagnostic_candidate_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_diagnostic_datasets())
        self.ttk.Label(controls, text="Dataset").pack(side="left", padx=(8, 0))
        self.diagnostic_dataset_combo = self.ttk.Combobox(
            controls,
            textvariable=self.diagnostic_dataset_var,
            state="readonly",
            width=16,
        )
        self.diagnostic_dataset_combo.pack(side="left", padx=4)
        self.diagnostic_dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_metric_diagnostics())
        self.ttk.Label(controls, text="Component").pack(side="left", padx=(8, 0))
        self.diagnostic_component_combo = self.ttk.Combobox(
            controls,
            textvariable=self.diagnostic_component_var,
            state="readonly",
            width=14,
        )
        self.diagnostic_component_combo.pack(side="left", padx=4)
        self.diagnostic_component_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_metric_diagnostics())

        self.diagnostics_text = self.tk.Text(self.diagnostics_tab, height=12, wrap="word")
        self.diagnostics_text.pack(fill="x", padx=6, pady=6)
        self.diagnostics_plot_frame = self.ttk.Frame(self.diagnostics_tab)
        self.diagnostics_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _set_text_widget(self, widget: Any, text: str) -> None:
        widget.delete("1.0", self.tk.END)
        widget.insert(self.tk.END, text)

    def _draw_figure(self, key: str, parent: Any, figure: Optional[Figure]) -> None:
        existing = self._canvases.pop(key, None)
        if existing is not None:
            existing.get_tk_widget().destroy()
        if figure is None:
            return
        canvas = self.FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvases[key] = canvas

    def _on_browse(self) -> None:
        selection = self.filedialog.askdirectory()
        if selection:
            self.path_var.set(selection)

    def _on_load(self) -> None:
        raw_path = self.path_var.get().strip()
        if not raw_path:
            self.messagebox.showwarning("Load Sweep", "Choose a sweep output directory first.")
            return
        self.load_from_path(Path(raw_path))

    def _on_reload(self) -> None:
        if self.path_var.get().strip():
            self.load_from_path(Path(self.path_var.get().strip()))

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
            self.export_to(Path(selection))

    def load_from_path(self, path: Path) -> None:
        try:
            model = load_sweep_visualization_model(path)
        except Exception as exc:  # noqa: BLE001
            self.status_var.set("Load failed")
            self.messagebox.showerror("Load Sweep", str(exc))
            return
        self.path_var.set(str(path))
        self.model = model
        self.status_var.set(f"Loaded {model.mode}")
        self._populate_from_model(model)

    def _rebuild_effect_filter_controls(self, model: SweepVisualizationModel) -> None:
        for widget in self.coverage_filter_frame.winfo_children():
            widget.destroy()
        self._effect_filter_vars = {}
        self._effect_filter_combos = {}
        for idx, parameter_name in enumerate(model.parameter_names):
            row = idx // 4
            col = (idx % 4) * 2
            values = sorted(
                {
                    _stringify_param(row_item.params.get(parameter_name))
                    for row_item in candidate_rows_for_generation_scope(model, model.default_generation_scope)
                    if parameter_name in row_item.params
                }
                | {
                    _stringify_param(record.params.get(parameter_name))
                    for record in run_records_for_generation_scope(model, model.default_generation_scope)
                    if parameter_name in record.params
                }
            )
            var = self.tk.StringVar(value="All")
            label = self.ttk.Label(self.coverage_filter_frame, text=parameter_name)
            label.grid(row=row, column=col, sticky="w", padx=3, pady=2)
            combo = self.ttk.Combobox(
                self.coverage_filter_frame,
                textvariable=var,
                state="readonly",
                width=16,
                values=["All", *values],
            )
            combo.grid(row=row, column=col + 1, sticky="we", padx=3, pady=2)
            combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_effects())
            self._effect_filter_vars[parameter_name] = var
            self._effect_filter_combos[parameter_name] = combo

    def _populate_from_model(self, model: SweepVisualizationModel) -> None:
        self._set_text_widget(self.overview_text, _format_overview_text(model))
        self._set_text_widget(self.progression_text, "\n".join(_progression_summary_lines(model)))
        self._set_text_widget(self.significance_text, _format_significance_text(model))
        self._set_text_widget(self.metric_text, _format_metric_summary_text(model))

        stage_values = ["All", *model.stage_order]
        self.coverage_stage_combo["values"] = stage_values
        self.coverage_stage_var.set(model.primary_stage_name)
        self.coverage_dataset_combo["values"] = ["aggregate", *model.datasets]
        self.coverage_dataset_var.set("aggregate")
        preferred_params = _stage_parameter_rankings(model, model.primary_stage_name)
        parameter_values = preferred_params if preferred_params else model.parameter_names
        self.coverage_param_combo["values"] = parameter_values
        self.coverage_y_param_combo["values"] = parameter_values
        self.coverage_param_var.set(parameter_values[0] if parameter_values else "")
        self.coverage_y_param_var.set(parameter_values[1] if len(parameter_values) > 1 else (parameter_values[0] if parameter_values else ""))
        value_options = _value_options(model)
        self.coverage_value_combo["values"] = value_options
        self.coverage_value_var.set(
            "score" if "score" in value_options else (value_options[0] if value_options else "")
        )
        self.coverage_strategy_combo["values"] = ["All", *model.strategies]
        self.coverage_strategy_var.set("All")
        families = _sorted_unique(
            [row.family for row in model.candidates if row.family]
            + [record.family for record in model.run_metric_records if record.family]
        )
        self.coverage_family_combo["values"] = ["All", *families]
        self.coverage_family_var.set("All")
        alpha_values = _sorted_unique(
            _alpha_filter_value(row.alpha)
            for row in model.candidates
            if row.alpha is not None
        )
        alpha_values = _sorted_unique(
            alpha_values
            + [
                _alpha_filter_value(record.alpha)
                for record in model.run_metric_records
                if record.alpha is not None
            ]
        )
        self.coverage_alpha_combo["values"] = ["All", *alpha_values]
        self.coverage_alpha_var.set("All")
        self.coverage_mode_var.set("1D Trend")
        self._rebuild_effect_filter_controls(model)

        preview_stage_values = [
            stage_name
            for stage_name in model.stage_order
            if any(row.preview_paths for row in _rows_for_stage(model, stage_name))
        ]
        self.preview_stage_combo["values"] = preview_stage_values
        self.preview_stage_var.set(preview_stage_values[0] if preview_stage_values else "")
        self._refresh_preview_candidates()

        diagnostic_stage_values = _sorted_unique(record.stage for record in model.run_metric_records)
        self.diagnostic_stage_combo["values"] = diagnostic_stage_values
        self.diagnostic_stage_var.set("metric_phase" if "metric_phase" in diagnostic_stage_values else (diagnostic_stage_values[0] if diagnostic_stage_values else ""))
        diagnostic_components = cast(List[str], model.diagnostic_availability.get("components", []))
        self.diagnostic_component_combo["values"] = diagnostic_components or ["gradient", "laplacian", "intensity"]
        self.diagnostic_component_var.set((diagnostic_components[0] if diagnostic_components else "gradient"))
        self._refresh_diagnostic_candidates()

        self._update_metric_tab_visibility(bool(model.alpha_family_summary))
        self.refresh_effects()
        self._draw_figure("overview", self.overview_plot_frame, create_overview_figure(model))
        self._draw_figure("progression", self.progression_plot_frame, create_progression_figure(model))
        self._draw_figure("significance", self.significance_plot_frame, create_significance_figure(model))
        self._draw_figure("metric", self.metric_plot_frame, create_metric_formulation_figure(model))
        self.refresh_metric_diagnostics()

    def _update_metric_tab_visibility(self, visible: bool) -> None:
        if visible and not self._metric_tab_visible:
            self.notebook.add(self.metric_tab, text="Metric Formulations")
            self._metric_tab_visible = True
        elif not visible and self._metric_tab_visible:
            self.notebook.forget(self.metric_tab)
            self._metric_tab_visible = False

    def refresh_effects(self) -> None:
        if self.model is None:
            return
        stage_name = self.coverage_stage_var.get() or self.model.primary_stage_name
        parameter_name = self.coverage_param_var.get() or (self.model.parameter_names[0] if self.model.parameter_names else None)
        y_parameter_name = self.coverage_y_param_var.get() or parameter_name
        value_option = self.coverage_value_var.get() or "score"
        dataset_scope = self.coverage_dataset_var.get() or "aggregate"
        mode = self.coverage_mode_var.get() or "1D Trend"
        strategy_filter = self.coverage_strategy_var.get() or "All"
        family_filter = self.coverage_family_var.get() or "All"
        alpha_filter = self.coverage_alpha_var.get() or "All"
        parameter_filters = {
            parameter_name: var.get() or "All"
            for parameter_name, var in self._effect_filter_vars.items()
        }
        self._set_text_widget(
            self.coverage_text,
            _format_effects_text(
                self.model,
                stage_name=stage_name,
                dataset_scope=dataset_scope,
                mode=mode,
                x_parameter_name=parameter_name,
                y_parameter_name=y_parameter_name,
                value_option=value_option,
                strategy_filter=strategy_filter,
                family_filter=family_filter,
                alpha_filter=alpha_filter,
                parameter_filters=parameter_filters,
            ),
        )
        self._draw_figure(
            "coverage",
            self.coverage_plot_frame,
            create_effects_figure(
                self.model,
                mode=mode,
                stage_name=stage_name,
                dataset_scope=dataset_scope,
                x_parameter_name=parameter_name,
                y_parameter_name=y_parameter_name,
                value_option=value_option,
                strategy_filter=strategy_filter,
                family_filter=family_filter,
                alpha_filter=alpha_filter,
                parameter_filters=parameter_filters,
            ),
        )

    def refresh_coverage(self) -> None:
        self.refresh_effects()

    def _selected_diagnostic_record(self) -> Optional[RunMetricRecord]:
        if self.model is None:
            return None
        stage_name = self.diagnostic_stage_var.get()
        candidate_id = self.diagnostic_candidate_var.get()
        dataset_id = self.diagnostic_dataset_var.get()
        return next(
            (
                record
                for record in self.model.run_metric_records
                if record.stage == stage_name
                and record.candidate_id == candidate_id
                and record.dataset_id == dataset_id
            ),
            None,
        )

    def _refresh_diagnostic_candidates(self) -> None:
        if self.model is None:
            self.diagnostic_candidate_combo["values"] = []
            self.diagnostic_candidate_var.set("")
            return
        stage_name = self.diagnostic_stage_var.get()
        candidate_values = _sorted_unique(
            record.candidate_id for record in self.model.run_metric_records if record.stage == stage_name
        )
        self.diagnostic_candidate_combo["values"] = candidate_values
        self.diagnostic_candidate_var.set(candidate_values[0] if candidate_values else "")
        self._refresh_diagnostic_datasets()

    def _refresh_diagnostic_datasets(self) -> None:
        if self.model is None:
            self.diagnostic_dataset_combo["values"] = []
            self.diagnostic_dataset_var.set("")
            return
        stage_name = self.diagnostic_stage_var.get()
        candidate_id = self.diagnostic_candidate_var.get()
        dataset_values = _sorted_unique(
            record.dataset_id
            for record in self.model.run_metric_records
            if record.stage == stage_name and record.candidate_id == candidate_id
        )
        self.diagnostic_dataset_combo["values"] = dataset_values
        self.diagnostic_dataset_var.set(dataset_values[0] if dataset_values else "")
        self.refresh_metric_diagnostics()

    def refresh_metric_diagnostics(self) -> None:
        record = self._selected_diagnostic_record()
        self._set_text_widget(self.diagnostics_text, _format_metric_diagnostics_text(record))
        self._draw_figure(
            "diagnostics",
            self.diagnostics_plot_frame,
            create_metric_diagnostics_figure(
                record,
                component_name=self.diagnostic_component_var.get() or "gradient",
            ),
        )

    def _refresh_preview_candidates(self) -> None:
        if self.model is None:
            self.preview_candidate_combo["values"] = []
            self.preview_candidate_var.set("")
            return
        stage_name = self.preview_stage_var.get()
        rows = [row for row in _rows_for_stage(self.model, stage_name) if row.preview_paths]
        candidate_values = [row.candidate_id for row in rows]
        self.preview_candidate_combo["values"] = candidate_values
        self.preview_candidate_var.set(candidate_values[0] if candidate_values else "")
        self._refresh_preview_datasets()

    def _refresh_preview_datasets(self) -> None:
        if self.model is None:
            self.preview_dataset_combo["values"] = []
            self.preview_dataset_var.set("")
            return
        stage_name = self.preview_stage_var.get()
        candidate_id = self.preview_candidate_var.get()
        row = next(
            (
                item
                for item in _rows_for_stage(self.model, stage_name)
                if item.candidate_id == candidate_id
            ),
            None,
        )
        dataset_values = sorted(row.preview_paths.keys()) if row is not None else []
        self.preview_dataset_combo["values"] = dataset_values
        self.preview_dataset_var.set(dataset_values[0] if dataset_values else "")

    def _open_preview(self) -> None:
        if self.model is None:
            return
        stage_name = self.preview_stage_var.get()
        candidate_id = self.preview_candidate_var.get()
        dataset_id = self.preview_dataset_var.get()
        row = next(
            (
                item
                for item in _rows_for_stage(self.model, stage_name)
                if item.candidate_id == candidate_id
            ),
            None,
        )
        if row is None or dataset_id not in row.preview_paths:
            self.messagebox.showwarning("Preview", "No preview image found for the selected row.")
            return
        preview_path = Path(row.preview_paths[dataset_id])
        if not preview_path.exists():
            self.messagebox.showwarning("Preview", f"Preview image not found: {preview_path}")
            return

        preview_window = self.tk.Toplevel(self.root)
        preview_window.title(f"{candidate_id} / {dataset_id}")
        image = Image.open(preview_path)
        if max(image.size) > 960:
            image.thumbnail((960, 960))
        photo = __import__("PIL.ImageTk", fromlist=["PhotoImage"]).PhotoImage(image)
        self._preview_image_ref = photo
        self.ttk.Label(preview_window, image=photo).pack(fill="both", expand=True, padx=8, pady=8)

    def export_to(self, output_dir: Path) -> Optional[Path]:
        if self.model is None:
            return None
        try:
            exported = export_sweep_visualization_bundle(self.model, output_dir)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Export", str(exc))
            return None
        self.status_var.set(f"Exported to {exported}")
        return exported


class SweepVisualizationApp(SweepVisualizationApp):
    """Enhanced GUI variant installed below via method rebinding."""

    pass


def _sweep_visualization_app_init(
    self: SweepVisualizationApp,
    root: Any,
    initial_path: Optional[Path] = None,
) -> None:
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

    self.root.title("Sweep Results Visualizer")
    self.root.geometry("1380x980")
    self.model = None
    self._canvases = {}
    self._preview_image_ref = None
    self._rectification_bundle_cache: Dict[Tuple[str, str, str, str, str], RectificationDisplayBundle] = {}
    self._comparison_slot_a_key: Optional[RectificationResultKey] = None
    self._comparison_slot_b_key: Optional[RectificationResultKey] = None
    self._comparison_slots_cleared = False
    self._comparison_sync_in_progress = False
    self._comparison_channel_sync_in_progress = False
    self._metric_tab_visible = False
    self._effect_filter_vars = {}
    self._effect_filter_combos = {}
    self._leaderboard_row_lookup: Dict[str, CandidateVisualizationRow] = {}
    self._leaderboard_sort_column = "rank"
    self._leaderboard_sort_reverse = False

    self.path_var = tk.StringVar(value=str(initial_path) if initial_path else "")
    self.status_var = tk.StringVar(value="Idle")
    self.coverage_stage_var = tk.StringVar(value="")
    self.coverage_dataset_var = tk.StringVar(value="aggregate")
    self.coverage_mode_var = tk.StringVar(value="1D Trend")
    self.coverage_param_var = tk.StringVar(value="")
    self.coverage_y_param_var = tk.StringVar(value="")
    self.coverage_value_var = tk.StringVar(value="score")
    self.coverage_strategy_var = tk.StringVar(value="All")
    self.coverage_family_var = tk.StringVar(value="All")
    self.coverage_alpha_var = tk.StringVar(value="All")
    self.sensitivity_stage_var = tk.StringVar(value="")
    self.sensitivity_dataset_var = tk.StringVar(value="aggregate")
    self.sensitivity_value_var = tk.StringVar(value="score")
    self.sensitivity_strategy_var = tk.StringVar(value="All")
    self.sensitivity_family_var = tk.StringVar(value="All")
    self.sensitivity_alpha_var = tk.StringVar(value="All")
    self.preview_stage_var = tk.StringVar(value="")
    self.preview_candidate_var = tk.StringVar(value="")
    self.preview_dataset_var = tk.StringVar(value="")
    self.leaderboard_stage_var = tk.StringVar(value="")
    self.leaderboard_dataset_var = tk.StringVar(value="aggregate")
    self.leaderboard_strategy_var = tk.StringVar(value="All")
    self.leaderboard_family_var = tk.StringVar(value="All")
    self.leaderboard_alpha_var = tk.StringVar(value="All")
    self.generation_scope_var = tk.StringVar(value=GENERATION_SCOPE_ALL)
    self.comparison_stage_a_var = tk.StringVar(value="")
    self.comparison_stage_b_var = tk.StringVar(value="")
    self.comparison_stage_var = self.comparison_stage_a_var
    self.comparison_dataset_var = tk.StringVar(value="")
    self.comparison_component_var = tk.StringVar(value="gradient")
    self.comparison_sync_var = tk.BooleanVar(value=True)
    self.comparison_lock_channel_var = tk.BooleanVar(value=True)
    self.comparison_slot_a_var = tk.StringVar(value="A: not set")
    self.comparison_slot_b_var = tk.StringVar(value="B: not set")
    self.diagnostic_stage_var = tk.StringVar(value="")
    self.diagnostic_candidate_var = tk.StringVar(value="")
    self.diagnostic_dataset_var = tk.StringVar(value="")
    self.diagnostic_component_var = tk.StringVar(value="gradient")
    self._sensitivity_row_lookup: Dict[str, ParameterSensitivityRow] = {}
    self._extension_suggestion_lookup: Dict[str, ExtensionSuggestion] = {}

    self._build_ui()
    if initial_path is not None:
        self.load_from_path(Path(initial_path))


def _sweep_visualization_build_ui(self: SweepVisualizationApp) -> None:
    top = self.ttk.Frame(self.root)
    top.pack(fill="x", padx=8, pady=6)

    self.ttk.Label(top, text="Sweep folder").pack(side="left")
    path_entry = self.ttk.Entry(top, textvariable=self.path_var, width=90)
    path_entry.pack(side="left", padx=6, fill="x", expand=True)
    self.ttk.Button(top, text="Browse...", command=self._on_browse).pack(side="left", padx=3)
    self.ttk.Button(top, text="Load", command=self._on_load).pack(side="left", padx=3)
    self.ttk.Button(top, text="Reload", command=self._on_reload).pack(side="left", padx=3)
    self.ttk.Button(top, text="Export...", command=self._on_export_dialog).pack(side="left", padx=3)
    self.ttk.Button(top, text="Open Folder", command=self._on_open_folder).pack(side="left", padx=3)
    self.ttk.Label(top, text="Generation").pack(side="left", padx=(10, 3))
    self.generation_scope_combo = self.ttk.Combobox(
        top,
        textvariable=self.generation_scope_var,
        state="readonly",
        width=16,
        values=[GENERATION_SCOPE_ALL],
    )
    self.generation_scope_combo.pack(side="left", padx=3)
    self.generation_scope_combo.bind("<<ComboboxSelected>>", lambda _evt: self._on_generation_scope_changed())
    self.ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

    self.notebook = self.ttk.Notebook(self.root)
    self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

    self.overview_tab = self.ttk.Frame(self.notebook)
    self.leaderboard_tab = self.ttk.Frame(self.notebook)
    self.comparison_tab = self.ttk.Frame(self.notebook)
    self.coverage_tab = self.ttk.Frame(self.notebook)
    self.sensitivity_tab = self.ttk.Frame(self.notebook)
    self.extensions_tab = self.ttk.Frame(self.notebook)
    self.progression_tab = self.ttk.Frame(self.notebook)
    self.significance_tab = self.ttk.Frame(self.notebook)
    self.metric_tab = self.ttk.Frame(self.notebook)
    self.diagnostics_tab = self.ttk.Frame(self.notebook)

    self.notebook.add(self.overview_tab, text="Overview")
    self.notebook.add(self.leaderboard_tab, text="Leaderboard")
    self.notebook.add(self.comparison_tab, text="Comparison")
    self.notebook.add(self.coverage_tab, text="Effects")
    self.notebook.add(self.sensitivity_tab, text="Sensitivity")
    self.notebook.add(self.extensions_tab, text="Next Extensions")
    self.notebook.add(self.progression_tab, text="Progression")
    self.notebook.add(self.significance_tab, text="Significance")
    self.notebook.add(self.diagnostics_tab, text="Metric Diagnostics")

    self._build_overview_tab()
    self._build_leaderboard_tab()
    self._build_comparison_tab()
    self._build_coverage_tab()
    self._build_sensitivity_tab()
    self._build_extensions_tab()
    self._build_progression_tab()
    self._build_significance_tab()
    self._build_metric_tab()
    self._build_diagnostics_tab()


def _sweep_visualization_build_leaderboard_tab(self: SweepVisualizationApp) -> None:
    controls = self.ttk.Frame(self.leaderboard_tab)
    controls.pack(fill="x", padx=6, pady=6)
    for label, variable, width in (
        ("Stage", self.leaderboard_stage_var, 14),
        ("Dataset", self.leaderboard_dataset_var, 12),
        ("Strategy", self.leaderboard_strategy_var, 14),
        ("Family", self.leaderboard_family_var, 16),
        ("Alpha", self.leaderboard_alpha_var, 10),
    ):
        self.ttk.Label(controls, text=label).pack(side="left", padx=(0 if label == "Stage" else 8, 0))
        combo = self.ttk.Combobox(
            controls,
            textvariable=variable,
            state="readonly",
            width=width,
        )
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_leaderboard())
        if label == "Stage":
            self.leaderboard_stage_combo = combo
        elif label == "Dataset":
            self.leaderboard_dataset_combo = combo
        elif label == "Strategy":
            self.leaderboard_strategy_combo = combo
        elif label == "Family":
            self.leaderboard_family_combo = combo
        else:
            self.leaderboard_alpha_combo = combo

    tree_frame = self.ttk.Frame(self.leaderboard_tab)
    tree_frame.pack(fill="both", expand=True, padx=6, pady=4)
    columns = (
        "rank",
        "candidate_id",
        "strategy",
        "score",
        "runtime",
        "ranking_metric",
        "family",
        "alpha",
        "missing_datasets",
    )
    self.leaderboard_tree = self.ttk.Treeview(
        tree_frame,
        columns=columns,
        show="headings",
        selectmode="extended",
        height=14,
    )
    widths = {
        "rank": 70,
        "candidate_id": 190,
        "strategy": 110,
        "score": 110,
        "runtime": 110,
        "ranking_metric": 150,
        "family": 130,
        "alpha": 90,
        "missing_datasets": 150,
    }
    labels = {
        "rank": "Rank",
        "candidate_id": "Candidate",
        "strategy": "Strategy",
        "score": "Score",
        "runtime": "Runtime (s)",
        "ranking_metric": "Ranking Metric",
        "family": "Family",
        "alpha": "Alpha",
        "missing_datasets": "Missing Datasets",
    }
    for column in columns:
        self.leaderboard_tree.heading(
            column,
            text=labels[column],
            command=lambda col=column: self._on_leaderboard_sort(col),
        )
        self.leaderboard_tree.column(column, width=widths[column], anchor="w")
    tree_scroll = self.ttk.Scrollbar(tree_frame, orient="vertical", command=self.leaderboard_tree.yview)
    self.leaderboard_tree.configure(yscrollcommand=tree_scroll.set)
    self.leaderboard_tree.pack(side="left", fill="both", expand=True)
    tree_scroll.pack(side="right", fill="y")
    self.leaderboard_tree.bind("<<TreeviewSelect>>", lambda _evt: self._on_leaderboard_selection_changed())

    self.leaderboard_detail_text = self.tk.Text(self.leaderboard_tab, height=14, wrap="word")
    self.leaderboard_detail_text.pack(fill="x", padx=6, pady=6)


def _sweep_visualization_build_sensitivity_tab(self: SweepVisualizationApp) -> None:
    controls = self.ttk.Frame(self.sensitivity_tab)
    controls.pack(fill="x", padx=6, pady=6)
    for label, variable, width in (
        ("Stage", self.sensitivity_stage_var, 14),
        ("Dataset", self.sensitivity_dataset_var, 12),
        ("Value", self.sensitivity_value_var, 24),
        ("Strategy", self.sensitivity_strategy_var, 14),
        ("Family", self.sensitivity_family_var, 16),
        ("Alpha", self.sensitivity_alpha_var, 10),
    ):
        self.ttk.Label(controls, text=label).pack(side="left", padx=(0 if label == "Stage" else 8, 0))
        combo = self.ttk.Combobox(
            controls,
            textvariable=variable,
            state="readonly",
            width=width,
        )
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_sensitivity())
        if label == "Stage":
            self.sensitivity_stage_combo = combo
        elif label == "Dataset":
            self.sensitivity_dataset_combo = combo
        elif label == "Value":
            self.sensitivity_value_combo = combo
        elif label == "Strategy":
            self.sensitivity_strategy_combo = combo
        elif label == "Family":
            self.sensitivity_family_combo = combo
        else:
            self.sensitivity_alpha_combo = combo

    tree_frame = self.ttk.Frame(self.sensitivity_tab)
    tree_frame.pack(fill="both", expand=True, padx=6, pady=4)
    columns = (
        "rank",
        "parameter",
        "impact",
        "best",
        "worst",
        "groups",
        "rows",
        "min_n",
        "ols",
    )
    self.sensitivity_tree = self.ttk.Treeview(
        tree_frame,
        columns=columns,
        show="headings",
        selectmode="browse",
        height=10,
    )
    labels = {
        "rank": "Rank",
        "parameter": "Parameter",
        "impact": "Impact",
        "best": "Best Value",
        "worst": "Worst Value",
        "groups": "Values",
        "rows": "Rows",
        "min_n": "Min n",
        "ols": "OLS dR2",
    }
    widths = {
        "rank": 65,
        "parameter": 210,
        "impact": 100,
        "best": 180,
        "worst": 180,
        "groups": 80,
        "rows": 80,
        "min_n": 80,
        "ols": 100,
    }
    for column in columns:
        self.sensitivity_tree.heading(column, text=labels[column])
        self.sensitivity_tree.column(column, width=widths[column], anchor="w")
    self.sensitivity_tree.pack(side="left", fill="both", expand=True)
    tree_scroll = self.ttk.Scrollbar(tree_frame, orient="vertical", command=self.sensitivity_tree.yview)
    self.sensitivity_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side="right", fill="y")

    self.sensitivity_text = self.tk.Text(self.sensitivity_tab, height=10, wrap="word")
    self.sensitivity_text.pack(fill="x", padx=6, pady=4)
    self.sensitivity_plot_frame = self.ttk.Frame(self.sensitivity_tab)
    self.sensitivity_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)


def _sweep_visualization_build_extensions_tab(self: SweepVisualizationApp) -> None:
    tree_frame = self.ttk.Frame(self.extensions_tab)
    tree_frame.pack(fill="both", expand=True, padx=6, pady=6)
    columns = ("rank", "rule", "parameter", "confidence", "recommendation")
    self.extensions_tree = self.ttk.Treeview(
        tree_frame,
        columns=columns,
        show="headings",
        selectmode="browse",
        height=10,
    )
    labels = {
        "rank": "Rank",
        "rule": "Rule",
        "parameter": "Parameter",
        "confidence": "Confidence",
        "recommendation": "Recommendation",
    }
    widths = {
        "rank": 70,
        "rule": 170,
        "parameter": 180,
        "confidence": 110,
        "recommendation": 620,
    }
    for column in columns:
        self.extensions_tree.heading(column, text=labels[column])
        self.extensions_tree.column(column, width=widths[column], anchor="w")
    self.extensions_tree.pack(side="left", fill="both", expand=True)
    tree_scroll = self.ttk.Scrollbar(tree_frame, orient="vertical", command=self.extensions_tree.yview)
    self.extensions_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side="right", fill="y")

    self.extensions_text = self.tk.Text(self.extensions_tab, height=14, wrap="word")
    self.extensions_text.pack(fill="x", padx=6, pady=6)


def _sweep_visualization_build_comparison_tab(self: SweepVisualizationApp) -> None:
    controls = self.ttk.Frame(self.comparison_tab)
    controls.pack(fill="x", padx=6, pady=6)
    self.ttk.Button(controls, text="Set A", command=lambda: self._set_comparison_slot("A")).pack(side="left", padx=(0, 3))
    self.ttk.Button(controls, text="Set B", command=lambda: self._set_comparison_slot("B")).pack(side="left", padx=3)
    self.ttk.Button(controls, text="Use 2 Selected", command=self._assign_comparison_slots_from_selection).pack(side="left", padx=3)
    self.ttk.Button(controls, text="Swap", command=self._swap_comparison_slots).pack(side="left", padx=3)
    self.ttk.Button(controls, text="Clear", command=self._clear_comparison_slots).pack(side="left", padx=3)
    self.ttk.Label(controls, text="A Stage").pack(side="left")
    self.comparison_stage_a_combo = self.ttk.Combobox(
        controls,
        textvariable=self.comparison_stage_a_var,
        state="readonly",
        width=14,
    )
    self.comparison_stage_a_combo.pack(side="left", padx=4)
    self.comparison_stage_a_combo.bind("<<ComboboxSelected>>", lambda _evt: self._on_comparison_stage_changed("A"))
    self.comparison_stage_combo = self.comparison_stage_a_combo
    self.ttk.Label(controls, text="B Stage").pack(side="left", padx=(8, 0))
    self.comparison_stage_b_combo = self.ttk.Combobox(
        controls,
        textvariable=self.comparison_stage_b_var,
        state="readonly",
        width=14,
    )
    self.comparison_stage_b_combo.pack(side="left", padx=4)
    self.comparison_stage_b_combo.bind("<<ComboboxSelected>>", lambda _evt: self._on_comparison_stage_changed("B"))
    self.ttk.Label(controls, text="Dataset").pack(side="left", padx=(8, 0))
    self.comparison_dataset_combo = self.ttk.Combobox(
        controls,
        textvariable=self.comparison_dataset_var,
        state="readonly",
        width=14,
    )
    self.comparison_dataset_combo.pack(side="left", padx=4)
    self.comparison_dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_comparison())
    self.ttk.Label(controls, text="Component").pack(side="left", padx=(8, 0))
    self.comparison_component_combo = self.ttk.Combobox(
        controls,
        textvariable=self.comparison_component_var,
        state="readonly",
        width=16,
    )
    self.comparison_component_combo.pack(side="left", padx=4)
    self.comparison_component_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_comparison())
    self.ttk.Checkbutton(
        controls,
        text="Sync Zoom/Pan",
        variable=self.comparison_sync_var,
    ).pack(side="left", padx=(12, 4))
    self.ttk.Checkbutton(
        controls,
        text="Lock Channel",
        variable=self.comparison_lock_channel_var,
    ).pack(side="left", padx=4)

    slot_frame = self.ttk.Frame(self.comparison_tab)
    slot_frame.pack(fill="x", padx=6, pady=(0, 4))
    self.ttk.Label(slot_frame, textvariable=self.comparison_slot_a_var).pack(side="left", padx=(0, 16))
    self.ttk.Label(slot_frame, textvariable=self.comparison_slot_b_var).pack(side="left")

    comparison_split = self.ttk.Panedwindow(self.comparison_tab, orient="vertical")
    comparison_split.pack(fill="both", expand=True, padx=6, pady=4)
    inspector_frame = self.ttk.Frame(comparison_split)
    details_frame = self.ttk.Frame(comparison_split)
    comparison_split.add(inspector_frame, weight=5)
    comparison_split.add(details_frame, weight=2)

    inspector_frame.columnconfigure(0, weight=1)
    inspector_frame.columnconfigure(1, weight=1)
    inspector_frame.rowconfigure(0, weight=1)
    self.comparison_inspector_a = _RectificationImageInspector(
        self,
        inspector_frame,
        slot_name="A",
        on_channel_changed=self._on_comparison_channel_changed,
        on_limits_changed=self._on_comparison_limits_changed,
    )
    self.comparison_inspector_b = _RectificationImageInspector(
        self,
        inspector_frame,
        slot_name="B",
        on_channel_changed=self._on_comparison_channel_changed,
        on_limits_changed=self._on_comparison_limits_changed,
    )
    self.comparison_inspector_a.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    self.comparison_inspector_b.frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    self.comparison_text = self.tk.Text(details_frame, height=8, wrap="word")
    self.comparison_text.pack(fill="x", padx=0, pady=(0, 4))
    self.comparison_robustness_plot_frame = self.ttk.Frame(self.comparison_tab)
    self.comparison_robustness_plot_frame.pack(fill="both", expand=True, padx=6, pady=4)


def _sweep_visualization_build_diagnostics_tab(self: SweepVisualizationApp) -> None:
    controls = self.ttk.Frame(self.diagnostics_tab)
    controls.pack(fill="x", padx=6, pady=6)
    self.ttk.Label(controls, text="Stage").pack(side="left")
    self.diagnostic_stage_combo = self.ttk.Combobox(
        controls,
        textvariable=self.diagnostic_stage_var,
        state="readonly",
        width=14,
    )
    self.diagnostic_stage_combo.pack(side="left", padx=4)
    self.diagnostic_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_diagnostic_candidates())
    self.ttk.Label(controls, text="Candidate").pack(side="left", padx=(8, 0))
    self.diagnostic_candidate_combo = self.ttk.Combobox(
        controls,
        textvariable=self.diagnostic_candidate_var,
        state="readonly",
        width=20,
    )
    self.diagnostic_candidate_combo.pack(side="left", padx=4)
    self.diagnostic_candidate_combo.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_diagnostic_datasets())
    self.ttk.Label(controls, text="Dataset").pack(side="left", padx=(8, 0))
    self.diagnostic_dataset_combo = self.ttk.Combobox(
        controls,
        textvariable=self.diagnostic_dataset_var,
        state="readonly",
        width=16,
    )
    self.diagnostic_dataset_combo.pack(side="left", padx=4)
    self.diagnostic_dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_metric_diagnostics())
    self.ttk.Label(controls, text="Component").pack(side="left", padx=(8, 0))
    self.diagnostic_component_combo = self.ttk.Combobox(
        controls,
        textvariable=self.diagnostic_component_var,
        state="readonly",
        width=14,
    )
    self.diagnostic_component_combo.pack(side="left", padx=4)
    self.diagnostic_component_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_metric_diagnostics())
    self.ttk.Button(controls, text="Open Preview", command=self._open_selected_diagnostic_preview).pack(side="left", padx=(10, 4))
    self.ttk.Button(controls, text="Open Run Folder", command=self._open_selected_diagnostic_folder).pack(side="left", padx=4)
    self.ttk.Button(controls, text="Open metric_record", command=self._open_selected_metric_record).pack(side="left", padx=4)

    self.diagnostics_text = self.tk.Text(self.diagnostics_tab, height=12, wrap="word")
    self.diagnostics_text.pack(fill="x", padx=6, pady=6)
    self.diagnostics_plot_frame = self.ttk.Frame(self.diagnostics_tab)
    self.diagnostics_plot_frame.pack(fill="both", expand=True, padx=6, pady=6)


def _sweep_visualization_leaderboard_candidate_rows(
    self: SweepVisualizationApp,
) -> List[CandidateVisualizationRow]:
    if self.model is None:
        return []
    stage_name = self.leaderboard_stage_var.get() or self.model.primary_stage_name
    strategy_filter = self.leaderboard_strategy_var.get() or "All"
    family_filter = self.leaderboard_family_var.get() or "All"
    alpha_filter = self.leaderboard_alpha_var.get() or "All"
    rows: List[CandidateVisualizationRow] = []
    for row in candidate_rows_for_generation_scope(self.model, self.model.default_generation_scope):
        if stage_name != "All" and row.stage != stage_name:
            continue
        if strategy_filter != "All" and row.strategy != strategy_filter:
            continue
        if family_filter != "All" and row.family != family_filter:
            continue
        if alpha_filter != "All" and _alpha_filter_value(row.alpha) != alpha_filter:
            continue
        rows.append(row)
    return rows


def _sweep_visualization_leaderboard_candidate_label(
    self: SweepVisualizationApp,
    row: CandidateVisualizationRow,
) -> str:
    stage_name = self.leaderboard_stage_var.get() or (self.model.primary_stage_name if self.model else "")
    if self.model is not None and self.model.default_generation_scope == GENERATION_SCOPE_ALL:
        prefix = row.generation_id or "unknown"
        return f"{prefix}:{row.stage}:{row.candidate_id}" if stage_name == "All" else f"{prefix}:{row.candidate_id}"
    return f"{row.stage}:{row.candidate_id}" if stage_name == "All" else row.candidate_id


def _sweep_visualization_sorted_leaderboard_rows(
    self: SweepVisualizationApp,
    rows: Sequence[CandidateVisualizationRow],
) -> List[CandidateVisualizationRow]:
    if self.model is None:
        return list(rows)
    dataset_scope = self.leaderboard_dataset_var.get() or "aggregate"
    column = self._leaderboard_sort_column
    if column in {"rank", "score", "runtime", "ranking_metric", "alpha", "missing_datasets"}:

        def numeric_value(row: CandidateVisualizationRow) -> Optional[float]:
            if column == "rank":
                return float(row.rank)
            if column == "score":
                return _candidate_score_for_dataset(row, dataset_scope, self.model.primary_metric)
            if column == "runtime":
                return _candidate_runtime_for_dataset(row, dataset_scope)
            if column == "ranking_metric":
                return _candidate_metric_value_for_dataset(
                    row,
                    dataset_scope,
                    self.model.primary_metric,
                )[1]
            if column == "alpha":
                return row.alpha
            return float(len(row.missing_datasets))

        finite_rows = []
        missing_rows = []
        for row in rows:
            value = numeric_value(row)
            if value is None or not math.isfinite(value):
                missing_rows.append(row)
            else:
                finite_rows.append((float(value), row))
        finite_rows.sort(
            key=lambda item: (item[0], item[1].rank, item[1].candidate_id),
            reverse=self._leaderboard_sort_reverse,
        )
        return [row for _, row in finite_rows] + missing_rows

    return sorted(
        rows,
        key=lambda row: (self._leaderboard_candidate_label(row), row.rank),
        reverse=self._leaderboard_sort_reverse,
    )


def _sweep_visualization_leaderboard_row_values(
    self: SweepVisualizationApp,
    row: CandidateVisualizationRow,
) -> Tuple[str, ...]:
    assert self.model is not None
    dataset_scope = self.leaderboard_dataset_var.get() or "aggregate"
    metric_name, metric_value = _candidate_metric_value_for_dataset(
        row,
        dataset_scope,
        self.model.primary_metric,
    )
    return (
        str(row.rank),
        self._leaderboard_candidate_label(row),
        row.strategy,
        _format_float(_candidate_score_for_dataset(row, dataset_scope, self.model.primary_metric)),
        _format_float(_candidate_runtime_for_dataset(row, dataset_scope), precision=3),
        f"{metric_name}: {_format_float(metric_value)}",
        row.family or "n/a",
        "n/a" if row.alpha is None else f"{row.alpha:g}",
        ", ".join(row.missing_datasets) if row.missing_datasets else "none",
    )


def _sweep_visualization_selected_leaderboard_rows(
    self: SweepVisualizationApp,
) -> List[CandidateVisualizationRow]:
    return [
        self._leaderboard_row_lookup[item_id]
        for item_id in self.leaderboard_tree.selection()
        if item_id in self._leaderboard_row_lookup
    ]


def _sweep_visualization_format_leaderboard_detail_text(
    self: SweepVisualizationApp,
    rows: Sequence[CandidateVisualizationRow],
) -> str:
    if not rows or self.model is None:
        return "No leaderboard selection."
    dataset_scope = self.leaderboard_dataset_var.get() or "aggregate"
    if len(rows) == 1:
        row = rows[0]
        metric_name, metric_value = _candidate_metric_value_for_dataset(
            row,
            dataset_scope,
            self.model.primary_metric,
        )
        lines = [
            f"candidate: {row.candidate_id}",
            f"stage: {row.stage}",
            f"generation: {row.generation_id or 'n/a'}",
            f"strategy: {row.strategy}",
            f"rank: {row.rank}",
            f"dataset scope: {dataset_scope}",
            f"score: {_format_float(_candidate_score_for_dataset(row, dataset_scope, self.model.primary_metric))}",
            f"runtime: {_format_float(_candidate_runtime_for_dataset(row, dataset_scope), precision=3)}s",
            f"{metric_name}: {_format_float(metric_value)}",
            f"family: {row.family or 'n/a'}",
            f"alpha: {'n/a' if row.alpha is None else f'{row.alpha:g}'}",
            f"missing datasets: {', '.join(row.missing_datasets) if row.missing_datasets else 'none'}",
            "",
            "params:",
            json.dumps(row.params, indent=2, sort_keys=True),
            "",
            "config:",
            json.dumps(row.config, indent=2, sort_keys=True),
        ]
        return "\n".join(lines)

    lines = [
        f"selected candidates: {', '.join(row.candidate_id for row in rows)}",
        f"dataset scope: {dataset_scope}",
        "",
    ]
    lines.extend(_varying_parameter_lines(rows))
    lines.append("")
    for row in rows:
        lines.append(
            f"{row.candidate_id}: score={_format_float(_candidate_score_for_dataset(row, dataset_scope, self.model.primary_metric))}, "
            f"runtime={_format_float(_candidate_runtime_for_dataset(row, dataset_scope), precision=3)}s"
        )
    return "\n".join(lines)


def _sweep_visualization_on_leaderboard_sort(
    self: SweepVisualizationApp,
    column: str,
) -> None:
    if self._leaderboard_sort_column == column:
        self._leaderboard_sort_reverse = not self._leaderboard_sort_reverse
    else:
        self._leaderboard_sort_column = column
        self._leaderboard_sort_reverse = column in {"score", "ranking_metric"}
    self.refresh_leaderboard()


def _sweep_visualization_seed_diagnostics_from_row(
    self: SweepVisualizationApp,
    row: CandidateVisualizationRow,
    *,
    dataset_hint: Optional[str] = None,
) -> None:
    if self.model is None:
        return
    scoped_records = run_records_for_generation_scope(self.model, self.model.default_generation_scope)
    candidate_values = _sorted_unique(
        _diagnostic_candidate_key(record)
        for record in scoped_records
        if record.stage == row.stage
    )
    selected_candidate_key = f"{row.generation_id}:{row.stage}:{row.candidate_id}" if row.generation_id else f"{row.stage}:{row.candidate_id}"
    dataset_values = _sorted_unique(
        record.dataset_id
        for record in scoped_records
        if record.stage == row.stage
        and record.candidate_id == row.candidate_id
        and (row.generation_id is None or record.generation_id == row.generation_id)
    )
    self.diagnostic_stage_combo["values"] = _sorted_unique(record.stage for record in scoped_records)
    self.diagnostic_stage_var.set(row.stage)
    self.diagnostic_candidate_combo["values"] = candidate_values
    self.diagnostic_candidate_var.set(
        selected_candidate_key if selected_candidate_key in candidate_values else (candidate_values[0] if candidate_values else "")
    )
    self.diagnostic_dataset_combo["values"] = dataset_values
    preferred_dataset = dataset_hint if dataset_hint in dataset_values else None
    self.diagnostic_dataset_var.set(preferred_dataset or (dataset_values[0] if dataset_values else ""))


def _sweep_visualization_on_leaderboard_selection_changed(
    self: SweepVisualizationApp,
) -> None:
    selected_rows = self._selected_leaderboard_rows()
    self._set_text_widget(
        self.leaderboard_detail_text,
        self._format_leaderboard_detail_text(selected_rows),
    )
    if selected_rows and self.model is not None:
        if self._comparison_slot_a_key is None and self._comparison_slot_b_key is None and not self._comparison_slots_cleared:
            self._set_comparison_stage_for_slot("A", selected_rows[0].stage)
            self._comparison_slot_a_key = self._selection_for_row(selected_rows[0]).key
            if len(selected_rows) > 1:
                self._set_comparison_stage_for_slot("B", selected_rows[1].stage)
                self._comparison_slot_b_key = self._selection_for_row(selected_rows[1]).key
        self._seed_diagnostics_from_row(
            selected_rows[0],
            dataset_hint=self.comparison_dataset_var.get() or _default_comparison_dataset(self.model),
        )
    self.refresh_comparison()


def _sweep_visualization_refresh_leaderboard(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    selected_ids = list(self.leaderboard_tree.selection())
    rows = self._sorted_leaderboard_rows(self._leaderboard_candidate_rows())
    self._leaderboard_row_lookup = {row.row_id: row for row in rows}
    self.leaderboard_tree.delete(*self.leaderboard_tree.get_children())
    for row in rows:
        self.leaderboard_tree.insert("", "end", iid=row.row_id, values=self._leaderboard_row_values(row))
    reselection = [item_id for item_id in selected_ids if item_id in self._leaderboard_row_lookup]
    if reselection:
        self.leaderboard_tree.selection_set(reselection)
    elif rows:
        self.leaderboard_tree.selection_set(rows[0].row_id)
    self._on_leaderboard_selection_changed()


def _sweep_visualization_sensitivity_filters(self: SweepVisualizationApp) -> Dict[str, object]:
    return {
        "strategy": self.sensitivity_strategy_var.get() or "All",
        "family": self.sensitivity_family_var.get() or "All",
        "alpha": self.sensitivity_alpha_var.get() or "All",
    }


def _sweep_visualization_refresh_sensitivity(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    stage_name = self.sensitivity_stage_var.get() or self.model.primary_stage_name
    dataset_scope = self.sensitivity_dataset_var.get() or "aggregate"
    value_option = self.sensitivity_value_var.get() or "score"
    rows = compute_parameter_sensitivity(
        self.model,
        stage_name=stage_name,
        dataset_scope=dataset_scope,
        value_option=value_option,
        filters=self._sensitivity_filters(),
    )
    self._sensitivity_row_lookup = {
        f"sensitivity:{row.rank}:{row.parameter_name}": row for row in rows
    }
    self.sensitivity_tree.delete(*self.sensitivity_tree.get_children())
    for item_id, row in self._sensitivity_row_lookup.items():
        self.sensitivity_tree.insert(
            "",
            "end",
            iid=item_id,
            values=(
                str(row.rank),
                row.parameter_name,
                _format_float(row.impact_score),
                f"{row.best_value or 'n/a'} ({_format_float(row.best_mean_value)})",
                f"{row.worst_value or 'n/a'} ({_format_float(row.worst_mean_value)})",
                str(row.num_values),
                str(row.num_rows),
                str(row.min_group_count),
                _format_float(row.ols_delta_r2),
            ),
        )
    if rows:
        self.sensitivity_tree.selection_set(next(iter(self._sensitivity_row_lookup.keys())))
    self._set_text_widget(self.sensitivity_text, _format_parameter_sensitivity_text(rows))
    self._draw_figure(
        "sensitivity",
        self.sensitivity_plot_frame,
        create_parameter_sensitivity_figure(
            self.model,
            rows,
            stage_name=stage_name,
            dataset_scope=dataset_scope,
            value_option=value_option,
        ),
    )
    self.refresh_extension_suggestions()


def _sweep_visualization_refresh_extension_suggestions(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    stage_name = self.model.primary_stage_name
    rows = compute_parameter_sensitivity(
        self.model,
        stage_name=stage_name,
        dataset_scope="aggregate",
        value_option="score",
        filters={},
    )
    suggestions = generate_next_extension_suggestions(self.model, rows)
    self._extension_suggestion_lookup = {
        f"suggestion:{suggestion.rank}:{suggestion.rule_id}:{suggestion.parameter_name or 'none'}": suggestion
        for suggestion in suggestions
    }
    self.extensions_tree.delete(*self.extensions_tree.get_children())
    for item_id, suggestion in self._extension_suggestion_lookup.items():
        self.extensions_tree.insert(
            "",
            "end",
            iid=item_id,
            values=(
                str(suggestion.rank),
                suggestion.rule_id,
                suggestion.parameter_name or "n/a",
                suggestion.confidence,
                suggestion.recommendation,
            ),
        )
    if suggestions:
        self.extensions_tree.selection_set(next(iter(self._extension_suggestion_lookup.keys())))
    self._set_text_widget(
        self.extensions_text,
        _format_extension_suggestions_text(suggestions),
    )


def _sweep_visualization_comparison_dataset(self: SweepVisualizationApp) -> str:
    if self.model is None:
        return self.comparison_dataset_var.get() or "aggregate"
    return self.comparison_dataset_var.get() or _default_comparison_dataset(self.model)


def _comparison_key_identity(key: RectificationResultKey) -> Tuple[str, str, Optional[str]]:
    stage_name, candidate_id, _dataset_id, generation_id = _unpack_rectification_key(key)
    return stage_name, candidate_id, generation_id


def _sweep_visualization_comparison_stage_for_slot(
    self: SweepVisualizationApp,
    slot_name: str,
) -> str:
    if self.model is None:
        return ""
    var = self.comparison_stage_a_var if slot_name == "A" else self.comparison_stage_b_var
    stage_name = var.get() or self.model.primary_stage_name
    if stage_name not in self.model.stage_order:
        stage_name = self.model.primary_stage_name
        var.set(stage_name)
    return stage_name


def _sweep_visualization_set_comparison_stage_for_slot(
    self: SweepVisualizationApp,
    slot_name: str,
    stage_name: str,
) -> None:
    var = self.comparison_stage_a_var if slot_name == "A" else self.comparison_stage_b_var
    var.set(stage_name)


def _sweep_visualization_default_comparison_key_for_stage(
    self: SweepVisualizationApp,
    stage_name: str,
    *,
    exclude_key: Optional[RectificationResultKey] = None,
) -> Optional[RectificationResultKey]:
    if self.model is None:
        return None
    rows = _rows_for_stage(self.model, stage_name)
    if not rows:
        return None
    fallback = rows[0]
    exclude_identity = _comparison_key_identity(exclude_key) if exclude_key is not None else None
    for row in rows:
        row_identity = (row.stage, row.candidate_id, row.generation_id)
        if exclude_identity is not None and row_identity == exclude_identity:
            continue
        return self._selection_for_row(row).key
    return self._selection_for_row(fallback).key


def _sweep_visualization_comparison_slot_needs_seed(
    self: SweepVisualizationApp,
    key: Optional[RectificationResultKey],
    stage_name: str,
) -> bool:
    if key is None:
        return True
    key_stage, _candidate_id, _dataset_id, _generation_id = _unpack_rectification_key(key)
    if key_stage != stage_name:
        return True
    return self._row_for_comparison_key(key) is None


def _sweep_visualization_sync_comparison_slots_to_stage_controls(
    self: SweepVisualizationApp,
    *,
    force_slot: Optional[str] = None,
) -> None:
    if self.model is None or self._comparison_slots_cleared:
        return
    stage_a = self._comparison_stage_for_slot("A")
    stage_b = self._comparison_stage_for_slot("B")

    key_a = self._comparison_slot_a_key
    key_b = self._comparison_slot_b_key
    if force_slot in {None, "A"} and (
        force_slot == "A" or self._comparison_slot_needs_seed(key_a, stage_a)
    ):
        exclude = key_b if stage_a == stage_b else None
        key_a = self._default_comparison_key_for_stage(stage_a, exclude_key=exclude)
        self._comparison_slot_a_key = key_a

    if force_slot in {None, "B"} and (
        force_slot == "B" or self._comparison_slot_needs_seed(key_b, stage_b)
    ):
        exclude = key_a if stage_a == stage_b else None
        key_b = self._default_comparison_key_for_stage(stage_b, exclude_key=exclude)
        self._comparison_slot_b_key = key_b

    if (
        stage_a == stage_b
        and key_a is not None
        and key_b is not None
        and _comparison_key_identity(key_a) == _comparison_key_identity(key_b)
    ):
        replacement = self._default_comparison_key_for_stage(stage_b, exclude_key=key_a)
        if replacement is not None:
            self._comparison_slot_b_key = replacement


def _sweep_visualization_on_comparison_stage_changed(
    self: SweepVisualizationApp,
    slot_name: str,
) -> None:
    self._comparison_slots_cleared = False
    self._sync_comparison_slots_to_stage_controls(force_slot=slot_name)
    self.refresh_comparison()


def _sweep_visualization_selection_for_row(
    self: SweepVisualizationApp,
    row: CandidateVisualizationRow,
) -> RectificationViewSelection:
    return RectificationViewSelection(
        stage=row.stage,
        candidate_id=row.candidate_id,
        dataset_id=self._comparison_dataset(),
        generation_id=row.generation_id,
        component_name=self.comparison_component_var.get() or _default_component_name(self.model) if self.model else None,
    )


def _sweep_visualization_row_for_key(
    self: SweepVisualizationApp,
    key: Optional[RectificationResultKey],
) -> Optional[CandidateVisualizationRow]:
    if self.model is None or key is None:
        return None
    stage_name, candidate_id, _dataset_id, generation_id = _unpack_rectification_key(key)
    return next(
        (
            row
            for row in candidate_rows_for_generation_scope(self.model, self.model.default_generation_scope)
            if row.stage == stage_name and row.candidate_id == candidate_id
            and (generation_id is None or row.generation_id == generation_id)
        ),
        None,
    )


def _sweep_visualization_key_label(
    self: SweepVisualizationApp,
    key: Optional[RectificationResultKey],
) -> str:
    if key is None:
        return "not set"
    stage_name, candidate_id, dataset_id, generation_id = _unpack_rectification_key(key)
    generation_text = f" / {generation_id}" if generation_id else ""
    return f"{candidate_id} / {dataset_id} / {stage_name}{generation_text}"


def _sweep_visualization_bundle_for_key(
    self: SweepVisualizationApp,
    key: Optional[RectificationResultKey],
) -> Optional[RectificationDisplayBundle]:
    if self.model is None or key is None:
        return None
    stage_name, candidate_id, dataset_id, generation_id = _unpack_rectification_key(key)
    component_name = self.comparison_component_var.get() or _default_component_name(self.model)
    cache_key = (stage_name, candidate_id, dataset_id, generation_id or "", component_name)
    cached = self._rectification_bundle_cache.get(cache_key)
    if cached is not None:
        return cached
    bundle = load_rectification_display_bundle(
        self.model,
        RectificationViewSelection(
            stage=stage_name,
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            generation_id=generation_id,
            component_name=component_name,
        ),
    )
    self._rectification_bundle_cache[cache_key] = bundle
    return bundle


def _sweep_visualization_refresh_comparison_slot_labels(self: SweepVisualizationApp) -> None:
    self.comparison_slot_a_var.set(f"A: {self._comparison_key_label(self._comparison_slot_a_key)}")
    self.comparison_slot_b_var.set(f"B: {self._comparison_key_label(self._comparison_slot_b_key)}")


def _sweep_visualization_set_comparison_slot(
    self: SweepVisualizationApp,
    slot_name: str,
) -> None:
    selected = self._selected_leaderboard_rows()
    if not selected:
        self.messagebox.showwarning("Comparison", "Select a leaderboard row first.")
        return
    row = selected[0] if slot_name == "A" or len(selected) == 1 else selected[1]
    key = self._selection_for_row(row).key
    self._set_comparison_stage_for_slot(slot_name, row.stage)
    if slot_name == "A":
        self._comparison_slot_a_key = key
    else:
        self._comparison_slot_b_key = key
    self._comparison_slots_cleared = False
    self.refresh_comparison()


def _sweep_visualization_assign_comparison_slots_from_selection(self: SweepVisualizationApp) -> None:
    selected = self._selected_leaderboard_rows()
    if len(selected) < 2:
        self.messagebox.showwarning("Comparison", "Select two leaderboard rows to compare.")
        return
    self._set_comparison_stage_for_slot("A", selected[0].stage)
    self._set_comparison_stage_for_slot("B", selected[1].stage)
    self._comparison_slot_a_key = self._selection_for_row(selected[0]).key
    self._comparison_slot_b_key = self._selection_for_row(selected[1]).key
    self._comparison_slots_cleared = False
    self.refresh_comparison()


def _sweep_visualization_swap_comparison_slots(self: SweepVisualizationApp) -> None:
    self._comparison_slot_a_key, self._comparison_slot_b_key = (
        self._comparison_slot_b_key,
        self._comparison_slot_a_key,
    )
    stage_a = self.comparison_stage_a_var.get()
    self.comparison_stage_a_var.set(self.comparison_stage_b_var.get())
    self.comparison_stage_b_var.set(stage_a)
    self.refresh_comparison()


def _sweep_visualization_clear_comparison_slots(self: SweepVisualizationApp) -> None:
    self._comparison_slot_a_key = None
    self._comparison_slot_b_key = None
    self._comparison_slots_cleared = True
    self.refresh_comparison()


def _sweep_visualization_update_comparison_dataset_keys(self: SweepVisualizationApp) -> None:
    dataset_id = self._comparison_dataset()
    if self._comparison_slot_a_key is not None:
        stage_name, candidate_id, _old_dataset, generation_id = _unpack_rectification_key(self._comparison_slot_a_key)
        self._comparison_slot_a_key = (stage_name, candidate_id, dataset_id, generation_id)
    if self._comparison_slot_b_key is not None:
        stage_name, candidate_id, _old_dataset, generation_id = _unpack_rectification_key(self._comparison_slot_b_key)
        self._comparison_slot_b_key = (stage_name, candidate_id, dataset_id, generation_id)


def _sweep_visualization_ensure_default_comparison_slots(self: SweepVisualizationApp) -> None:
    if self.model is None or self._comparison_slots_cleared:
        return
    self._sync_comparison_slots_to_stage_controls()


def _sweep_visualization_on_comparison_channel_changed(
    self: SweepVisualizationApp,
    source: _RectificationImageInspector,
    channel_name: str,
) -> None:
    if not self.comparison_lock_channel_var.get() or self._comparison_channel_sync_in_progress:
        return
    other = self.comparison_inspector_b if source is self.comparison_inspector_a else self.comparison_inspector_a
    self._comparison_channel_sync_in_progress = True
    try:
        other.set_channel(channel_name, emit_event=False, reset_view=False)
    finally:
        self._comparison_channel_sync_in_progress = False


def _sweep_visualization_on_comparison_limits_changed(
    self: SweepVisualizationApp,
    source: _RectificationImageInspector,
) -> None:
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


def _sweep_visualization_comparison_rows(
    self: SweepVisualizationApp,
) -> List[CandidateVisualizationRow]:
    if self.model is None:
        return []
    keyed_rows = [
        row
        for row in (
            self._row_for_comparison_key(self._comparison_slot_a_key),
            self._row_for_comparison_key(self._comparison_slot_b_key),
        )
        if row is not None
    ]
    if keyed_rows:
        return keyed_rows
    selected = self._selected_leaderboard_rows()
    unique_rows: List[CandidateVisualizationRow] = []
    seen = set()
    for row in selected:
        if row.row_id in seen:
            continue
        seen.add(row.row_id)
        unique_rows.append(row)
    if unique_rows:
        return unique_rows[:2]
    stage_a = self._comparison_stage_for_slot("A")
    stage_b = self._comparison_stage_for_slot("B")
    key_a = self._default_comparison_key_for_stage(stage_a)
    key_b = self._default_comparison_key_for_stage(
        stage_b,
        exclude_key=key_a if stage_a == stage_b else None,
    )
    return [
        row
        for row in (
            self._row_for_comparison_key(key_a),
            self._row_for_comparison_key(key_b),
        )
        if row is not None
    ]


def _sweep_visualization_refresh_comparison(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    self._update_comparison_dataset_keys()
    self._ensure_default_comparison_slots()
    dataset_id = self._comparison_dataset()
    rows = self._comparison_rows()
    bundle_a = self._bundle_for_comparison_key(self._comparison_slot_a_key)
    bundle_b = self._bundle_for_comparison_key(self._comparison_slot_b_key)
    self.comparison_inspector_a.set_bundle(bundle_a, reset_view=False)
    self.comparison_inspector_b.set_bundle(bundle_b, reset_view=False)
    if self.comparison_lock_channel_var.get() and bundle_a is not None and bundle_b is not None:
        self.comparison_inspector_b.set_channel(
            self.comparison_inspector_a.current_channel(),
            emit_event=False,
            reset_view=False,
        )
    self._refresh_comparison_slot_labels()
    self._set_text_widget(
        self.comparison_text,
        _format_comparison_text(self.model, rows, dataset_id=dataset_id),
    )
    self._draw_figure(
        "comparison_robustness",
        self.comparison_robustness_plot_frame,
        create_candidate_robustness_figure(self.model, rows),
    )
    if rows:
        self._seed_diagnostics_from_row(rows[0], dataset_hint=dataset_id)
        self.refresh_metric_diagnostics()


def _sweep_visualization_refresh_metric_diagnostics(self: SweepVisualizationApp) -> None:
    record = self._selected_diagnostic_record()
    self._set_text_widget(self.diagnostics_text, _format_metric_diagnostics_text(record))
    self._draw_figure(
        "diagnostics",
        self.diagnostics_plot_frame,
        create_metric_diagnostics_figure(
            record,
            component_name=self.diagnostic_component_var.get() or "gradient",
        ),
    )


def _sweep_visualization_open_rectification_viewer(
    self: SweepVisualizationApp,
    bundle: RectificationDisplayBundle,
    *,
    title: str,
) -> None:
    viewer = self.tk.Toplevel(self.root)
    viewer.title(title)
    viewer.geometry("980x860")
    inspector = _RectificationImageInspector(self, viewer, slot_name="Viewer")
    inspector.frame.pack(fill="both", expand=True)
    inspector.set_bundle(bundle, reset_view=True)
    viewer._rectification_inspector = inspector
    if bundle.warnings:
        warning_text = self.tk.Text(viewer, height=min(5, max(2, len(bundle.warnings))), wrap="word")
        warning_text.insert(self.tk.END, "\n".join(bundle.warnings[:8]))
        warning_text.configure(state="disabled")
        warning_text.pack(fill="x", padx=6, pady=(0, 6))
    self._preview_image_ref = inspector


def _sweep_visualization_open_preview(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    stage_name = self.preview_stage_var.get()
    candidate_key = self.preview_candidate_var.get()
    dataset_id = self.preview_dataset_var.get()
    if not stage_name or not candidate_key or not dataset_id:
        self.messagebox.showwarning("Preview", "Select a stage, candidate, and dataset first.")
        return
    row = next(
        (
            item
            for item in _rows_for_stage(self.model, stage_name)
            if _preview_candidate_key(item) == candidate_key or item.candidate_id == candidate_key
        ),
        None,
    )
    if row is None:
        self.messagebox.showwarning("Preview", "No preview image found for the selected row.")
        return
    selection = RectificationViewSelection(
        stage=row.stage,
        candidate_id=row.candidate_id,
        dataset_id=dataset_id,
        generation_id=row.generation_id,
        component_name=self.comparison_component_var.get() or _default_component_name(self.model),
    )
    bundle = load_rectification_display_bundle(self.model, selection)
    if bundle.preview is None and not any(bundle.get(channel) is not None for channel in RECTIFICATION_VIEW_CHANNELS):
        self.messagebox.showwarning("Preview", "No displayable image found for the selected result.")
        return
    self._open_rectification_viewer(
        bundle,
        title=f"{self._leaderboard_candidate_label(row)} / {dataset_id}",
    )


def _sweep_visualization_open_selected_diagnostic_preview(self: SweepVisualizationApp) -> None:
    record = self._selected_diagnostic_record()
    if record is None or self.model is None:
        self.messagebox.showwarning("Preview", "No diagnostic run is selected.")
        return
    bundle = load_rectification_display_bundle(
        self.model,
        RectificationViewSelection(
            stage=record.stage,
            candidate_id=record.candidate_id,
            dataset_id=record.dataset_id,
            generation_id=record.generation_id,
            component_name=self.diagnostic_component_var.get() or _default_component_name(self.model),
        ),
    )
    if bundle.preview is None and not any(bundle.get(channel) is not None for channel in RECTIFICATION_VIEW_CHANNELS):
        self.messagebox.showwarning("Preview", "No displayable image found for the selected run.")
        return
    self._open_rectification_viewer(
        bundle,
        title=f"{record.candidate_id} / {record.dataset_id}",
    )


def _sweep_visualization_open_selected_diagnostic_folder(self: SweepVisualizationApp) -> None:
    record = self._selected_diagnostic_record()
    candidate_path = None
    if record is not None:
        for path_text in (record.metric_record_path, record.workspace_path, record.preview_path):
            if path_text:
                candidate_path = Path(path_text).parent
                break
    if candidate_path is None:
        self.messagebox.showwarning("Run Folder", "No run folder could be resolved.")
        return
    ok, err = _open_with_system(candidate_path)
    if not ok:
        self.messagebox.showerror("Run Folder", err or "Failed to open run folder.")


def _sweep_visualization_open_selected_metric_record(self: SweepVisualizationApp) -> None:
    record = self._selected_diagnostic_record()
    metric_record_path = None if record is None else record.metric_record_path
    if not metric_record_path:
        self.messagebox.showwarning("metric_record", "No metric_record.json available for the selected run.")
        return
    ok, err = _open_with_system(Path(metric_record_path))
    if not ok:
        self.messagebox.showerror("metric_record", err or "Failed to open metric_record.json.")


def _preview_candidate_key(row: CandidateVisualizationRow) -> str:
    return row.row_id


def _diagnostic_candidate_key(record: RunMetricRecord) -> str:
    generation_prefix = f"{record.generation_id}:" if record.generation_id else ""
    return f"{generation_prefix}{record.stage}:{record.candidate_id}"


def _diagnostic_record_matches_candidate_key(record: RunMetricRecord, key: str) -> bool:
    return key in {
        record.candidate_id,
        f"{record.stage}:{record.candidate_id}",
        _diagnostic_candidate_key(record),
    }


def _sweep_visualization_on_generation_scope_changed(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    requested = self.generation_scope_var.get() or GENERATION_SCOPE_ALL
    self.model.default_generation_scope = _normalize_generation_scope(self.model, requested)
    self.generation_scope_var.set(self.model.default_generation_scope)
    self._rectification_bundle_cache.clear()
    self._comparison_slot_a_key = None
    self._comparison_slot_b_key = None
    self._comparison_slots_cleared = False
    self._populate_from_model(self.model)


def _sweep_visualization_refresh_preview_candidates(self: SweepVisualizationApp) -> None:
    if self.model is None:
        self.preview_candidate_combo["values"] = []
        self.preview_candidate_var.set("")
        return
    stage_name = self.preview_stage_var.get()
    rows = [row for row in _rows_for_stage(self.model, stage_name) if row.preview_paths]
    candidate_values = [_preview_candidate_key(row) for row in rows]
    self.preview_candidate_combo["values"] = candidate_values
    self.preview_candidate_var.set(candidate_values[0] if candidate_values else "")
    self._refresh_preview_datasets()


def _sweep_visualization_refresh_preview_datasets(self: SweepVisualizationApp) -> None:
    if self.model is None:
        self.preview_dataset_combo["values"] = []
        self.preview_dataset_var.set("")
        return
    stage_name = self.preview_stage_var.get()
    candidate_key = self.preview_candidate_var.get()
    row = next(
        (
            item
            for item in _rows_for_stage(self.model, stage_name)
            if _preview_candidate_key(item) == candidate_key or item.candidate_id == candidate_key
        ),
        None,
    )
    dataset_values = sorted(row.preview_paths.keys()) if row is not None else []
    self.preview_dataset_combo["values"] = dataset_values
    self.preview_dataset_var.set(dataset_values[0] if dataset_values else "")


def _sweep_visualization_selected_diagnostic_record(self: SweepVisualizationApp) -> Optional[RunMetricRecord]:
    if self.model is None:
        return None
    stage_name = self.diagnostic_stage_var.get()
    candidate_key = self.diagnostic_candidate_var.get()
    dataset_id = self.diagnostic_dataset_var.get()
    return next(
        (
            record
            for record in run_records_for_generation_scope(self.model, self.model.default_generation_scope)
            if record.stage == stage_name
            and _diagnostic_record_matches_candidate_key(record, candidate_key)
            and record.dataset_id == dataset_id
        ),
        None,
    )


def _sweep_visualization_refresh_diagnostic_candidates(self: SweepVisualizationApp) -> None:
    if self.model is None:
        self.diagnostic_candidate_combo["values"] = []
        self.diagnostic_candidate_var.set("")
        return
    stage_name = self.diagnostic_stage_var.get()
    candidate_values = _sorted_unique(
        _diagnostic_candidate_key(record)
        for record in run_records_for_generation_scope(self.model, self.model.default_generation_scope)
        if record.stage == stage_name
    )
    self.diagnostic_candidate_combo["values"] = candidate_values
    self.diagnostic_candidate_var.set(candidate_values[0] if candidate_values else "")
    self._refresh_diagnostic_datasets()


def _sweep_visualization_refresh_diagnostic_datasets(self: SweepVisualizationApp) -> None:
    if self.model is None:
        self.diagnostic_dataset_combo["values"] = []
        self.diagnostic_dataset_var.set("")
        return
    stage_name = self.diagnostic_stage_var.get()
    candidate_key = self.diagnostic_candidate_var.get()
    dataset_values = _sorted_unique(
        record.dataset_id
        for record in run_records_for_generation_scope(self.model, self.model.default_generation_scope)
        if record.stage == stage_name and _diagnostic_record_matches_candidate_key(record, candidate_key)
    )
    self.diagnostic_dataset_combo["values"] = dataset_values
    self.diagnostic_dataset_var.set(dataset_values[0] if dataset_values else "")
    self.refresh_metric_diagnostics()


def _sweep_visualization_populate_from_model(
    self: SweepVisualizationApp,
    model: SweepVisualizationModel,
) -> None:
    self._rectification_bundle_cache.clear()
    self._comparison_slot_a_key = None
    self._comparison_slot_b_key = None
    self._comparison_slots_cleared = False
    self._set_text_widget(self.overview_text, _format_overview_text(model))
    self._set_text_widget(self.progression_text, "\n".join(_progression_summary_lines(model)))
    self._set_text_widget(self.significance_text, _format_significance_text(model))
    self._set_text_widget(self.metric_text, _format_metric_summary_text(model))
    generation_values = generation_scope_options(model)
    self.generation_scope_combo["values"] = generation_values
    if model.default_generation_scope not in generation_values:
        model.default_generation_scope = generation_values[0] if generation_values else GENERATION_SCOPE_ALL
    self.generation_scope_var.set(model.default_generation_scope)
    scoped_candidates = candidate_rows_for_generation_scope(model, model.default_generation_scope)
    scoped_records = run_records_for_generation_scope(model, model.default_generation_scope)

    stage_values = ["All", *model.stage_order]
    self.coverage_stage_combo["values"] = stage_values
    self.coverage_stage_var.set(model.primary_stage_name)
    self.coverage_dataset_combo["values"] = ["aggregate", *model.datasets]
    self.coverage_dataset_var.set("aggregate")
    preferred_params = _stage_parameter_rankings(model, model.primary_stage_name)
    parameter_values = preferred_params if preferred_params else model.parameter_names
    self.coverage_param_combo["values"] = parameter_values
    self.coverage_y_param_combo["values"] = parameter_values
    self.coverage_param_var.set(parameter_values[0] if parameter_values else "")
    self.coverage_y_param_var.set(
        parameter_values[1] if len(parameter_values) > 1 else (parameter_values[0] if parameter_values else "")
    )
    value_options = _value_options(model)
    self.coverage_value_combo["values"] = value_options
    self.coverage_value_var.set(
        "score" if "score" in value_options else (value_options[0] if value_options else "")
    )
    self.coverage_strategy_combo["values"] = ["All", *model.strategies]
    self.coverage_strategy_var.set("All")
    families = _sorted_unique(
        [row.family for row in scoped_candidates if row.family]
        + [record.family for record in scoped_records if record.family]
    )
    alpha_values = _sorted_unique(
        [_alpha_filter_value(row.alpha) for row in scoped_candidates if row.alpha is not None]
        + [
            _alpha_filter_value(record.alpha)
            for record in scoped_records
            if record.alpha is not None
        ]
    )
    self.coverage_family_combo["values"] = ["All", *families]
    self.coverage_family_var.set("All")
    self.coverage_alpha_combo["values"] = ["All", *alpha_values]
    self.coverage_alpha_var.set("All")
    self.coverage_mode_var.set("1D Trend")
    self._rebuild_effect_filter_controls(model)

    self.sensitivity_stage_combo["values"] = stage_values
    self.sensitivity_stage_var.set(model.primary_stage_name)
    self.sensitivity_dataset_combo["values"] = ["aggregate", *model.datasets]
    self.sensitivity_dataset_var.set("aggregate")
    self.sensitivity_value_combo["values"] = value_options
    self.sensitivity_value_var.set(
        "score" if "score" in value_options else (value_options[0] if value_options else "")
    )
    self.sensitivity_strategy_combo["values"] = ["All", *model.strategies]
    self.sensitivity_strategy_var.set("All")
    self.sensitivity_family_combo["values"] = ["All", *families]
    self.sensitivity_family_var.set("All")
    self.sensitivity_alpha_combo["values"] = ["All", *alpha_values]
    self.sensitivity_alpha_var.set("All")

    preview_stage_values = [
        stage_name
        for stage_name in model.stage_order
        if any(row.preview_paths for row in _rows_for_stage(model, stage_name))
    ]
    self.preview_stage_combo["values"] = preview_stage_values
    self.preview_stage_var.set(preview_stage_values[0] if preview_stage_values else "")
    self._refresh_preview_candidates()

    self.leaderboard_stage_combo["values"] = stage_values
    self.leaderboard_stage_var.set(model.primary_stage_name)
    self.leaderboard_dataset_combo["values"] = ["aggregate", *model.datasets]
    self.leaderboard_dataset_var.set("aggregate")
    self.leaderboard_strategy_combo["values"] = ["All", *model.strategies]
    self.leaderboard_strategy_var.set("All")
    self.leaderboard_family_combo["values"] = ["All", *families]
    self.leaderboard_family_var.set("All")
    self.leaderboard_alpha_combo["values"] = ["All", *alpha_values]
    self.leaderboard_alpha_var.set("All")

    dataset_values = _real_datasets(model)
    self.comparison_stage_combo["values"] = model.stage_order
    self.comparison_stage_a_combo["values"] = model.stage_order
    self.comparison_stage_b_combo["values"] = model.stage_order
    self.comparison_stage_a_var.set(model.primary_stage_name)
    self.comparison_stage_b_var.set(model.primary_stage_name)
    self.comparison_dataset_combo["values"] = dataset_values
    self.comparison_dataset_var.set(_default_comparison_dataset(model))
    diagnostic_components = cast(List[str], model.diagnostic_availability.get("components", []))
    diagnostic_components = diagnostic_components or ["gradient"]
    self.comparison_component_combo["values"] = diagnostic_components
    self.comparison_component_var.set(_default_component_name(model))

    diagnostic_stage_values = _sorted_unique(record.stage for record in scoped_records)
    self.diagnostic_stage_combo["values"] = diagnostic_stage_values
    self.diagnostic_stage_var.set(
        model.primary_stage_name if model.primary_stage_name in diagnostic_stage_values else (diagnostic_stage_values[0] if diagnostic_stage_values else "")
    )
    self.diagnostic_component_combo["values"] = diagnostic_components
    self.diagnostic_component_var.set(_default_component_name(model))
    self._refresh_diagnostic_candidates()

    self._update_metric_tab_visibility(bool(model.alpha_family_summary))
    self.refresh_effects()
    self.refresh_sensitivity()
    self.refresh_leaderboard()
    self._draw_figure("overview", self.overview_plot_frame, create_overview_figure(model))
    self._draw_figure("progression", self.progression_plot_frame, create_progression_figure(model))
    self._draw_figure("significance", self.significance_plot_frame, create_significance_figure(model))
    self._draw_figure("metric", self.metric_plot_frame, create_metric_formulation_figure(model))
    self.refresh_comparison()
    self.refresh_metric_diagnostics()


SweepVisualizationApp.__init__ = _sweep_visualization_app_init
SweepVisualizationApp._build_ui = _sweep_visualization_build_ui
SweepVisualizationApp._on_generation_scope_changed = _sweep_visualization_on_generation_scope_changed
SweepVisualizationApp._build_leaderboard_tab = _sweep_visualization_build_leaderboard_tab
SweepVisualizationApp._build_sensitivity_tab = _sweep_visualization_build_sensitivity_tab
SweepVisualizationApp._build_extensions_tab = _sweep_visualization_build_extensions_tab
SweepVisualizationApp._build_comparison_tab = _sweep_visualization_build_comparison_tab
SweepVisualizationApp._build_diagnostics_tab = _sweep_visualization_build_diagnostics_tab
SweepVisualizationApp._leaderboard_candidate_rows = _sweep_visualization_leaderboard_candidate_rows
SweepVisualizationApp._leaderboard_candidate_label = _sweep_visualization_leaderboard_candidate_label
SweepVisualizationApp._sorted_leaderboard_rows = _sweep_visualization_sorted_leaderboard_rows
SweepVisualizationApp._leaderboard_row_values = _sweep_visualization_leaderboard_row_values
SweepVisualizationApp._selected_leaderboard_rows = _sweep_visualization_selected_leaderboard_rows
SweepVisualizationApp._format_leaderboard_detail_text = _sweep_visualization_format_leaderboard_detail_text
SweepVisualizationApp._on_leaderboard_sort = _sweep_visualization_on_leaderboard_sort
SweepVisualizationApp._seed_diagnostics_from_row = _sweep_visualization_seed_diagnostics_from_row
SweepVisualizationApp._on_leaderboard_selection_changed = _sweep_visualization_on_leaderboard_selection_changed
SweepVisualizationApp.refresh_leaderboard = _sweep_visualization_refresh_leaderboard
SweepVisualizationApp._sensitivity_filters = _sweep_visualization_sensitivity_filters
SweepVisualizationApp.refresh_sensitivity = _sweep_visualization_refresh_sensitivity
SweepVisualizationApp.refresh_extension_suggestions = _sweep_visualization_refresh_extension_suggestions
SweepVisualizationApp._comparison_dataset = _sweep_visualization_comparison_dataset
SweepVisualizationApp._comparison_stage_for_slot = _sweep_visualization_comparison_stage_for_slot
SweepVisualizationApp._set_comparison_stage_for_slot = _sweep_visualization_set_comparison_stage_for_slot
SweepVisualizationApp._default_comparison_key_for_stage = _sweep_visualization_default_comparison_key_for_stage
SweepVisualizationApp._comparison_slot_needs_seed = _sweep_visualization_comparison_slot_needs_seed
SweepVisualizationApp._sync_comparison_slots_to_stage_controls = _sweep_visualization_sync_comparison_slots_to_stage_controls
SweepVisualizationApp._on_comparison_stage_changed = _sweep_visualization_on_comparison_stage_changed
SweepVisualizationApp._selection_for_row = _sweep_visualization_selection_for_row
SweepVisualizationApp._row_for_comparison_key = _sweep_visualization_row_for_key
SweepVisualizationApp._comparison_key_label = _sweep_visualization_key_label
SweepVisualizationApp._bundle_for_comparison_key = _sweep_visualization_bundle_for_key
SweepVisualizationApp._refresh_comparison_slot_labels = _sweep_visualization_refresh_comparison_slot_labels
SweepVisualizationApp._set_comparison_slot = _sweep_visualization_set_comparison_slot
SweepVisualizationApp._assign_comparison_slots_from_selection = _sweep_visualization_assign_comparison_slots_from_selection
SweepVisualizationApp._swap_comparison_slots = _sweep_visualization_swap_comparison_slots
SweepVisualizationApp._clear_comparison_slots = _sweep_visualization_clear_comparison_slots
SweepVisualizationApp._update_comparison_dataset_keys = _sweep_visualization_update_comparison_dataset_keys
SweepVisualizationApp._ensure_default_comparison_slots = _sweep_visualization_ensure_default_comparison_slots
SweepVisualizationApp._on_comparison_channel_changed = _sweep_visualization_on_comparison_channel_changed
SweepVisualizationApp._on_comparison_limits_changed = _sweep_visualization_on_comparison_limits_changed
SweepVisualizationApp._comparison_rows = _sweep_visualization_comparison_rows
SweepVisualizationApp.refresh_comparison = _sweep_visualization_refresh_comparison
SweepVisualizationApp.refresh_metric_diagnostics = _sweep_visualization_refresh_metric_diagnostics
SweepVisualizationApp._open_rectification_viewer = _sweep_visualization_open_rectification_viewer
SweepVisualizationApp._refresh_preview_candidates = _sweep_visualization_refresh_preview_candidates
SweepVisualizationApp._refresh_preview_datasets = _sweep_visualization_refresh_preview_datasets
SweepVisualizationApp._open_preview = _sweep_visualization_open_preview
SweepVisualizationApp._selected_diagnostic_record = _sweep_visualization_selected_diagnostic_record
SweepVisualizationApp._refresh_diagnostic_candidates = _sweep_visualization_refresh_diagnostic_candidates
SweepVisualizationApp._refresh_diagnostic_datasets = _sweep_visualization_refresh_diagnostic_datasets
SweepVisualizationApp._open_selected_diagnostic_preview = _sweep_visualization_open_selected_diagnostic_preview
SweepVisualizationApp._open_selected_diagnostic_folder = _sweep_visualization_open_selected_diagnostic_folder
SweepVisualizationApp._open_selected_metric_record = _sweep_visualization_open_selected_metric_record
SweepVisualizationApp._populate_from_model = _sweep_visualization_populate_from_model


def launch_sweep_visualization(initial_path: Optional[Path] = None) -> None:
    import tkinter as tk

    root = tk.Tk()
    SweepVisualizationApp(root, initial_path=initial_path)
    root.mainloop()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize parameter sweep outputs.")
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        help="Optional sweep output directory to load on startup.",
    )
    parser.add_argument(
        "--export-dir",
        help="Optional output directory for headless export without launching the GUI.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.export_dir:
        if not args.sweep_dir:
            raise ValueError("--export-dir requires a sweep_dir argument.")
        model = load_sweep_visualization_model(Path(args.sweep_dir))
        export_sweep_visualization_bundle(model, Path(args.export_dir))
        return
    initial_path = Path(args.sweep_dir) if args.sweep_dir else None
    launch_sweep_visualization(initial_path=initial_path)


if __name__ == "__main__":
    main()

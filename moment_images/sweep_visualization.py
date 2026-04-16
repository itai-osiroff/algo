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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from moment_images.im_process import save_workspace_variables
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
    alpha_family_summary: List[Dict[str, object]]
    alpha_ranked_runs: List[Dict[str, object]]
    diagnostic_availability: Dict[str, object]
    candidate_ids: List[str]
    score_matrix: np.ndarray
    runtime_matrix: np.ndarray
    metric_tensor: np.ndarray
    parameter_numeric_matrix: np.ndarray
    parameter_text_matrix: np.ndarray


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
                    "family": item.get("family"),
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
        ranked_runs = [dict(item) for item in raw_ranked if isinstance(item, dict)]

    return summary_rows, ranked_runs


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
    return {
        str(key): value
        for key, value in config.items()
        if key != "hdr_exposure_times" and value is not None
    }


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

    family_text = str(family).strip() if family is not None and str(family).strip() else None
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
    config = cast(Dict[str, object], ranking.get("config", {}))
    params = cast(Dict[str, object], ranking.get("params", {}))
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
        params={str(key): value for key, value in params.items()},
        config={str(key): value for key, value in config.items()},
        family=family,
        alpha=alpha,
        formulation_id=formulation_id,
        preview_paths=preview_paths,
    )


def _candidate_row_from_legacy_record(
    *,
    output_dir: Path,
    record: Dict[str, object],
    rank: int,
    primary_metric: str,
) -> CandidateVisualizationRow:
    run_name = str(record.get("run", f"legacy_{rank:04d}"))
    config = cast(Dict[str, object], record.get("config", {}))
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
        config={str(key): value for key, value in config.items()},
        family=family,
        alpha=alpha,
        formulation_id=formulation_id,
        preview_paths=preview_paths,
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
    config = cast(Dict[str, object], metric_record.get("config", {}))
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
        config={str(key): value for key, value in config.items()},
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
        mode = "metric_phase_dir" if path_scope == "metric_phase_dir" else ("spec_full" if stage2_present else "spec_partial")
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
        "primary_metric_source": primary_metric_source,
        "alpha_summary_source": alpha_summary_source,
    }
    experiment_metadata.update(
        {
            "mode": mode,
            "dataset_count": len(datasets),
            "candidate_count": len(ordered_candidate_rows),
            "run_record_count": len(run_metric_records),
            "primary_stage_name": primary_stage_name,
            "primary_metric": primary_metric,
            "primary_metric_source": primary_metric_source,
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


def _rows_for_stage(model: SweepVisualizationModel, stage_name: Optional[str]) -> List[CandidateVisualizationRow]:
    target_stage = stage_name or model.primary_stage_name
    return [row for row in model.candidates if row.stage == target_stage]


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
        _candidate_rows_to_effect_rows(model.candidates)
        if dataset_scope == "aggregate"
        else _run_records_to_effect_rows(model.run_metric_records)
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
    if "stage1" not in model.stages or "stage2" not in model.stages:
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

    quality_ax.set_xlabel("alpha")
    quality_ax.set_ylabel("mean score / primary metric")
    quality_ax.set_title("Quality vs metric formulation alpha")
    quality_ax.grid(True, linestyle=":", alpha=0.4)
    if quality_ax.lines:
        quality_ax.legend()

    runtime_ax.set_xlabel("alpha")
    runtime_ax.set_ylabel("mean runtime (s)")
    runtime_ax.set_title("Runtime vs metric formulation alpha")
    runtime_ax.grid(True, linestyle=":", alpha=0.4)
    if runtime_ax.lines:
        runtime_ax.legend()

    displacement_ax.set_xlabel("alpha")
    displacement_ax.set_ylabel("mean displacement magnitude")
    displacement_ax.set_title("Displacement vs metric formulation alpha")
    displacement_ax.grid(True, linestyle=":", alpha=0.4)
    if displacement_ax.lines:
        displacement_ax.legend()

    consistency_ax.set_xlabel("alpha")
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
    preferred_stage = "metric_phase" if any(record.stage == "metric_phase" for record in model.run_metric_records) else model.primary_stage_name
    candidate_order = {
        (row.stage, row.candidate_id): idx
        for idx, row in enumerate(model.candidates)
    }
    eligible = [
        record
        for record in model.run_metric_records
        if record.stage == preferred_stage and (record.workspace_path or record.displacement_score_diagnostics)
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda record: (
            candidate_order.get((record.stage, record.candidate_id), len(candidate_order)),
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


def _find_run_metric_record(
    model: SweepVisualizationModel,
    *,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
) -> Optional[RunMetricRecord]:
    return next(
        (
            record
            for record in model.run_metric_records
            if record.stage == stage_name
            and record.candidate_id == candidate_id
            and record.dataset_id == dataset_id
        ),
        None,
    )


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


def _collapse_image_like(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if not isinstance(array, np.ndarray):
        return None
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return None
    while arr.ndim > 2:
        arr = np.nanmean(arr, axis=-1)
    return arr


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
    )
    preview_path = row.preview_paths.get(dataset_id) or row.preview_paths.get("aggregate")
    if preview_path is None and record is not None:
        preview_path = record.preview_path
    preview = _load_preview_array(preview_path)

    payload: Dict[str, np.ndarray] = {}
    read_warnings: List[str] = []
    if record is not None and record.workspace_path is not None:
        payload, read_warnings = _load_workspace_diagnostics(Path(record.workspace_path))

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
    image = ax.imshow(arr, aspect="auto", cmap=cmap)
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
        f"stage: {rows[0].stage if rows else model.primary_stage_name}",
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
                    row.candidate_id,
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
                f"{row.candidate_id}\n"
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
                rendered_title = f"{row.candidate_id} (rank {row.rank})\n{title}"
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
    ax.set_yticklabels([row.candidate_id for row in rows])
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
    return [
        {
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
            "missing_datasets": "|".join(row.missing_datasets),
            "dataset_scores_json": _compact_json(row.dataset_scores),
            "dataset_runtimes_json": _compact_json(row.dataset_runtimes),
            "dataset_metrics_json": _compact_json(row.dataset_metrics),
            "params_json": _compact_json(row.params),
            "config_json": _compact_json(row.config),
        }
        for row in model.candidates
    ]


def _run_metric_csv_rows(model: SweepVisualizationModel) -> List[Dict[str, object]]:
    return [
        {
            "stage": record.stage,
            "candidate_id": record.candidate_id,
            "dataset_id": record.dataset_id,
            "run_name": record.run_name,
            "strategy": record.strategy,
            "family": record.family,
            "alpha": record.alpha,
            "formulation_id": record.formulation_id,
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
        for record in model.run_metric_records
    ]


def _write_candidate_leaderboard_csv(model: SweepVisualizationModel, path: Path) -> None:
    write_csv_atomic(
        path,
        [
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
            "stage",
            "candidate_id",
            "dataset_id",
            "run_name",
            "strategy",
            "family",
            "alpha",
            "formulation_id",
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
    report_assets: Sequence[str],
    report_context_path: str,
    dataset_assets: Dict[str, Dict[str, object]],
    robustness_asset: Optional[str],
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

    effects_x, effects_y = _default_export_effect_parameters(model)
    report_stage = model.primary_stage_name
    default_component = _default_component_name(model)
    top_rows = _top_candidates_for_stage(model, report_stage, limit=DEFAULT_REPORT_TOP_K)
    plots: Dict[str, Figure] = {
        "overview.png": create_overview_figure(model),
        "coverage.png": create_coverage_figure(model),
        "progression.png": create_progression_figure(model),
        "significance.png": create_significance_figure(model),
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
    ]
    _write_candidate_leaderboard_csv(model, export_dir / table_files[0])
    _write_run_metric_records_csv(model, export_dir / table_files[1])

    report_context = {
        "generated_at": _utc_now_iso(),
        "report_stage": report_stage,
        "report_top_k": DEFAULT_REPORT_TOP_K,
        "default_component": default_component,
        "summary_images": [*plot_files, topk_overview_rel],
        "report_assets": report_assets,
        "table_files": table_files,
        "top_candidates": [_candidate_row_to_export_dict(row) for row in top_rows],
        "dataset_assets": dataset_assets,
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
            report_assets=report_assets,
            report_context_path=report_context_rel,
            dataset_assets=dataset_assets,
            robustness_asset=robustness_rel,
        ),
        encoding="utf-8",
    )

    stage_names = model.stage_order
    bundle_payload: Dict[str, object] = {
        "candidate_ids": np.asarray(model.candidate_ids, dtype=object),
        "candidate_labels": np.asarray([row.display_name for row in model.candidates], dtype=object),
        "candidate_stages": np.asarray([row.stage for row in model.candidates], dtype=object),
        "candidate_strategies": np.asarray([row.strategy for row in model.candidates], dtype=object),
        "candidate_ranks": np.asarray([row.rank for row in model.candidates], dtype=np.int32),
        "candidate_scores": np.asarray(
            [row.score if row.score is not None else np.nan for row in model.candidates],
            dtype=np.float64,
        ),
        "candidate_mean_dataset_scores": np.asarray(
            [
                row.mean_dataset_score if row.mean_dataset_score is not None else np.nan
                for row in model.candidates
            ],
            dtype=np.float64,
        ),
        "candidate_std_dataset_scores": np.asarray(
            [
                row.std_dataset_score if row.std_dataset_score is not None else np.nan
                for row in model.candidates
            ],
            dtype=np.float64,
        ),
        "dataset_ids": np.asarray(model.datasets, dtype=object),
        "metric_names": np.asarray(model.metric_names, dtype=object),
        "parameter_names": np.asarray(model.parameter_names, dtype=object),
        "score_matrix": model.score_matrix,
        "runtime_matrix": model.runtime_matrix,
        "metric_tensor": model.metric_tensor,
        "parameter_numeric_matrix": model.parameter_numeric_matrix,
        "parameter_text_matrix": np.asarray(model.parameter_text_matrix, dtype=object),
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
            [_compact_json(_candidate_row_to_export_dict(row)) for row in model.candidates],
            dtype=object,
        ),
        "run_metric_records_json": np.asarray(
            [_compact_json(_run_metric_record_to_export_dict(record)) for record in model.run_metric_records],
            dtype=object,
        ),
        "warnings": np.asarray(model.warnings, dtype=object),
        "provenance_json": np.asarray(_compact_json(model.provenance), dtype=object),
        "experiment_metadata_json": np.asarray(
            _compact_json(model.experiment_metadata),
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
        "report_assets": report_assets,
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
        _candidate_rows_to_effect_rows(model.candidates)
        if dataset_scope == "aggregate"
        else _run_records_to_effect_rows(model.run_metric_records)
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
                    for row_item in model.candidates
                    if parameter_name in row_item.params
                }
                | {
                    _stringify_param(record.params.get(parameter_name))
                    for record in model.run_metric_records
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
    self.FigureCanvasTkAgg = __import__(
        "matplotlib.backends.backend_tkagg",
        fromlist=["FigureCanvasTkAgg"],
    ).FigureCanvasTkAgg

    self.root.title("Sweep Results Visualizer")
    self.root.geometry("1380x980")
    self.model = None
    self._canvases = {}
    self._preview_image_ref = None
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
    self.preview_stage_var = tk.StringVar(value="")
    self.preview_candidate_var = tk.StringVar(value="")
    self.preview_dataset_var = tk.StringVar(value="")
    self.leaderboard_stage_var = tk.StringVar(value="")
    self.leaderboard_dataset_var = tk.StringVar(value="aggregate")
    self.leaderboard_strategy_var = tk.StringVar(value="All")
    self.leaderboard_family_var = tk.StringVar(value="All")
    self.leaderboard_alpha_var = tk.StringVar(value="All")
    self.comparison_stage_var = tk.StringVar(value="")
    self.comparison_dataset_var = tk.StringVar(value="")
    self.comparison_component_var = tk.StringVar(value="gradient")
    self.diagnostic_stage_var = tk.StringVar(value="")
    self.diagnostic_candidate_var = tk.StringVar(value="")
    self.diagnostic_dataset_var = tk.StringVar(value="")
    self.diagnostic_component_var = tk.StringVar(value="gradient")

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
    self.ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

    self.notebook = self.ttk.Notebook(self.root)
    self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

    self.overview_tab = self.ttk.Frame(self.notebook)
    self.leaderboard_tab = self.ttk.Frame(self.notebook)
    self.comparison_tab = self.ttk.Frame(self.notebook)
    self.coverage_tab = self.ttk.Frame(self.notebook)
    self.progression_tab = self.ttk.Frame(self.notebook)
    self.significance_tab = self.ttk.Frame(self.notebook)
    self.metric_tab = self.ttk.Frame(self.notebook)
    self.diagnostics_tab = self.ttk.Frame(self.notebook)

    self.notebook.add(self.overview_tab, text="Overview")
    self.notebook.add(self.leaderboard_tab, text="Leaderboard")
    self.notebook.add(self.comparison_tab, text="Comparison")
    self.notebook.add(self.coverage_tab, text="Effects")
    self.notebook.add(self.progression_tab, text="Progression")
    self.notebook.add(self.significance_tab, text="Significance")
    self.notebook.add(self.diagnostics_tab, text="Metric Diagnostics")

    self._build_overview_tab()
    self._build_leaderboard_tab()
    self._build_comparison_tab()
    self._build_coverage_tab()
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


def _sweep_visualization_build_comparison_tab(self: SweepVisualizationApp) -> None:
    controls = self.ttk.Frame(self.comparison_tab)
    controls.pack(fill="x", padx=6, pady=6)
    self.ttk.Label(controls, text="Stage").pack(side="left")
    self.comparison_stage_combo = self.ttk.Combobox(
        controls,
        textvariable=self.comparison_stage_var,
        state="readonly",
        width=14,
    )
    self.comparison_stage_combo.pack(side="left", padx=4)
    self.comparison_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_comparison())
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

    self.comparison_text = self.tk.Text(self.comparison_tab, height=12, wrap="word")
    self.comparison_text.pack(fill="x", padx=6, pady=6)
    self.comparison_plot_frame = self.ttk.Frame(self.comparison_tab)
    self.comparison_plot_frame.pack(fill="both", expand=True, padx=6, pady=4)
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
    for row in self.model.candidates:
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
    candidate_values = _sorted_unique(
        record.candidate_id for record in self.model.run_metric_records if record.stage == row.stage
    )
    dataset_values = _sorted_unique(
        record.dataset_id
        for record in self.model.run_metric_records
        if record.stage == row.stage and record.candidate_id == row.candidate_id
    )
    self.diagnostic_stage_combo["values"] = _sorted_unique(record.stage for record in self.model.run_metric_records)
    self.diagnostic_stage_var.set(row.stage)
    self.diagnostic_candidate_combo["values"] = candidate_values
    self.diagnostic_candidate_var.set(
        row.candidate_id if row.candidate_id in candidate_values else (candidate_values[0] if candidate_values else "")
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
        self.comparison_stage_var.set(selected_rows[0].stage)
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


def _sweep_visualization_comparison_rows(
    self: SweepVisualizationApp,
) -> List[CandidateVisualizationRow]:
    if self.model is None:
        return []
    stage_name = self.comparison_stage_var.get() or self.model.primary_stage_name
    selected = [row for row in self._selected_leaderboard_rows() if row.stage == stage_name]
    unique_rows: List[CandidateVisualizationRow] = []
    seen = set()
    for row in selected:
        if row.row_id in seen:
            continue
        seen.add(row.row_id)
        unique_rows.append(row)
    if 2 <= len(unique_rows) <= DEFAULT_REPORT_TOP_K:
        return unique_rows[:DEFAULT_REPORT_TOP_K]
    return _top_candidates_for_stage(self.model, stage_name, limit=DEFAULT_REPORT_TOP_K)


def _sweep_visualization_refresh_comparison(self: SweepVisualizationApp) -> None:
    if self.model is None:
        return
    dataset_id = self.comparison_dataset_var.get() or _default_comparison_dataset(self.model)
    rows = self._comparison_rows()
    self._set_text_widget(
        self.comparison_text,
        _format_comparison_text(self.model, rows, dataset_id=dataset_id),
    )
    self._draw_figure(
        "comparison_main",
        self.comparison_plot_frame,
        create_comparison_figure(
            self.model,
            rows,
            dataset_id=dataset_id,
            component_name=self.comparison_component_var.get() or _default_component_name(self.model),
        ),
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


def _sweep_visualization_open_selected_diagnostic_preview(self: SweepVisualizationApp) -> None:
    record = self._selected_diagnostic_record()
    preview_path = None if record is None else record.preview_path
    if not preview_path:
        self.messagebox.showwarning("Preview", "No preview image available for the selected run.")
        return
    ok, err = _open_with_system(Path(preview_path))
    if not ok:
        self.messagebox.showerror("Preview", err or "Failed to open preview.")


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


def _sweep_visualization_populate_from_model(
    self: SweepVisualizationApp,
    model: SweepVisualizationModel,
) -> None:
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
        [row.family for row in model.candidates if row.family]
        + [record.family for record in model.run_metric_records if record.family]
    )
    alpha_values = _sorted_unique(
        [_alpha_filter_value(row.alpha) for row in model.candidates if row.alpha is not None]
        + [
            _alpha_filter_value(record.alpha)
            for record in model.run_metric_records
            if record.alpha is not None
        ]
    )
    self.coverage_family_combo["values"] = ["All", *families]
    self.coverage_family_var.set("All")
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
    self.comparison_stage_var.set(model.primary_stage_name)
    self.comparison_dataset_combo["values"] = dataset_values
    self.comparison_dataset_var.set(_default_comparison_dataset(model))
    diagnostic_components = cast(List[str], model.diagnostic_availability.get("components", []))
    diagnostic_components = diagnostic_components or ["gradient"]
    self.comparison_component_combo["values"] = diagnostic_components
    self.comparison_component_var.set(_default_component_name(model))

    diagnostic_stage_values = _sorted_unique(record.stage for record in model.run_metric_records)
    self.diagnostic_stage_combo["values"] = diagnostic_stage_values
    self.diagnostic_stage_var.set(
        model.primary_stage_name if model.primary_stage_name in diagnostic_stage_values else (diagnostic_stage_values[0] if diagnostic_stage_values else "")
    )
    self.diagnostic_component_combo["values"] = diagnostic_components
    self.diagnostic_component_var.set(_default_component_name(model))
    self._refresh_diagnostic_candidates()

    self._update_metric_tab_visibility(bool(model.alpha_family_summary))
    self.refresh_effects()
    self.refresh_leaderboard()
    self._draw_figure("overview", self.overview_plot_frame, create_overview_figure(model))
    self._draw_figure("progression", self.progression_plot_frame, create_progression_figure(model))
    self._draw_figure("significance", self.significance_plot_frame, create_significance_figure(model))
    self._draw_figure("metric", self.metric_plot_frame, create_metric_formulation_figure(model))
    self.refresh_comparison()
    self.refresh_metric_diagnostics()


SweepVisualizationApp.__init__ = _sweep_visualization_app_init
SweepVisualizationApp._build_ui = _sweep_visualization_build_ui
SweepVisualizationApp._build_leaderboard_tab = _sweep_visualization_build_leaderboard_tab
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
SweepVisualizationApp._comparison_rows = _sweep_visualization_comparison_rows
SweepVisualizationApp.refresh_comparison = _sweep_visualization_refresh_comparison
SweepVisualizationApp.refresh_metric_diagnostics = _sweep_visualization_refresh_metric_diagnostics
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

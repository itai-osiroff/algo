"""Interactive visualization and export helpers for parameter sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
import platform
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
from moment_images.reliable_io import read_json_first_valid, write_json_atomic


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
class SweepVisualizationModel:
    """Canonical representation of legacy and modern sweep outputs."""

    source_dir: Path
    mode: str
    primary_stage_name: str
    primary_metric: str
    datasets: List[str]
    strategies: List[str]
    metric_names: List[str]
    parameter_names: List[str]
    stage_order: List[str]
    stages: Dict[str, StageVisualizationSummary]
    stage_candidate_ids: Dict[str, List[str]]
    candidates: List[CandidateVisualizationRow]
    scoring_weights: Dict[str, float]
    warnings: List[str]
    provenance: Dict[str, object]
    experiment_metadata: Dict[str, object]
    alpha_family_summary: List[Dict[str, object]]
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
    rows: Sequence[CandidateVisualizationRow],
    primary_metric: str,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[CandidateVisualizationRow]] = {}
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
            }
        )
    summary_rows.sort(key=lambda item: (str(item["family"]), float(item["alpha"])))
    return summary_rows


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


def load_sweep_visualization_model(path: Path) -> SweepVisualizationModel:
    """Load a legacy or modern sweep folder into a canonical visualization model."""

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


def _value_options(model: SweepVisualizationModel) -> List[str]:
    options = ["score", "mean_dataset_score", "mean_runtime_seconds"]
    options.extend(f"metric:{metric_name}" for metric_name in model.metric_names)
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
    return None


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


def create_coverage_figure(
    model: SweepVisualizationModel,
    *,
    stage_name: Optional[str] = None,
    parameter_name: Optional[str] = None,
    value_option: Optional[str] = None,
) -> Figure:
    rows = _rows_for_stage(model, stage_name)
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if not rows:
        ax.text(0.5, 0.5, "No candidate rows available for coverage plotting.", ha="center", va="center")
        ax.axis("off")
        return fig

    parameter = parameter_name or (model.parameter_names[0] if model.parameter_names else None)
    option = value_option or ("score" if any(row.score is not None for row in rows) else _value_options(model)[-1])
    if parameter is None:
        ax.text(0.5, 0.5, "No parameter metadata available.", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_values = np.asarray(
        [_numeric_param(row.params.get(parameter)) for row in rows],
        dtype=np.float64,
    )
    text_values = [_stringify_param(row.params.get(parameter)) for row in rows]
    y_values = np.asarray(
        [
            _row_value_for_option(row, option)
            if _row_value_for_option(row, option) is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    valid_mask = np.isfinite(y_values)
    if not np.any(valid_mask):
        ax.text(0.5, 0.5, f"No finite values for '{option}'.", ha="center", va="center")
        ax.axis("off")
        return fig

    if np.all(np.isfinite(numeric_values)):
        ax.scatter(numeric_values[valid_mask], y_values[valid_mask], c="#2a6f97", alpha=0.8)
        ax.set_xlabel(parameter)
        ax.set_xticks(sorted(set(numeric_values[valid_mask])))
    else:
        categories = sorted(dict.fromkeys(text_values))
        x_lookup = {label: idx for idx, label in enumerate(categories)}
        x_values = np.asarray([x_lookup[label] for label in text_values], dtype=np.float64)
        ax.scatter(x_values[valid_mask], y_values[valid_mask], c="#2a6f97", alpha=0.8)
        ax.set_xlabel(parameter)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=30, ha="right")

    ax.set_ylabel(option.replace("metric:", ""))
    ax.set_title(f"{stage_name or model.primary_stage_name}: {parameter} vs {option}")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


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

    fig = Figure(figsize=(10, 5))
    quality_ax = fig.add_subplot(121)
    runtime_ax = fig.add_subplot(122)
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
    }


def export_sweep_visualization_bundle(model: SweepVisualizationModel, output_dir: Path) -> Path:
    """Export default plots plus an aggregated `.npz` / `.mat` reproducibility bundle."""

    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    plots: Dict[str, Figure] = {
        "overview.png": create_overview_figure(model),
        "coverage.png": create_coverage_figure(model),
        "progression.png": create_progression_figure(model),
        "significance.png": create_significance_figure(model),
    }
    metric_fig = create_metric_formulation_figure(model)
    if metric_fig is not None:
        plots["metric_formulations.png"] = metric_fig

    plot_files: List[str] = []
    for filename, figure in plots.items():
        figure.savefig(export_dir / filename, dpi=180)
        plot_files.append(filename)
        figure.clf()

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
        "candidate_table_json": np.asarray(
            [_compact_json(_candidate_row_to_export_dict(row)) for row in model.candidates],
            dtype=object,
        ),
        "warnings": np.asarray(model.warnings, dtype=object),
        "provenance_json": np.asarray(_compact_json(model.provenance), dtype=object),
        "experiment_metadata_json": np.asarray(
            _compact_json(model.experiment_metadata),
            dtype=object,
        ),
    }
    save_workspace_variables(str(export_dir / "visualization_bundle.npz"), bundle_payload)

    manifest = {
        "generated_at": _utc_now_iso(),
        "source_dir": str(model.source_dir.resolve()),
        "mode": model.mode,
        "primary_stage_name": model.primary_stage_name,
        "primary_metric": model.primary_metric,
        "plot_files": plot_files,
        "bundle_npz": "visualization_bundle.npz",
        "bundle_mat": "visualization_bundle.mat",
        "warning_count": len(model.warnings),
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
        f"datasets ({len(model.datasets)}): {', '.join(model.datasets) if model.datasets else 'n/a'}",
        f"strategies: {', '.join(model.strategies) if model.strategies else 'n/a'}",
        f"primary stage: {model.primary_stage_name}",
        f"primary metric: {model.primary_metric}",
        "",
        "stages:",
    ]
    for stage_name in model.stage_order:
        stage = model.stages[stage_name]
        lines.append(
            f"  {stage_name}: planned={stage.num_runs_planned} successful={stage.num_successful} "
            f"failed={stage.num_failed} pending={stage.num_pending} running={stage.num_running}"
        )
    if model.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"  - {warning}" for warning in model.warnings[:12])
    return "\n".join(lines)


def _format_coverage_text(model: SweepVisualizationModel, stage_name: str) -> str:
    rows = _rows_for_stage(model, stage_name)
    if not rows:
        return "No candidate rows available."
    lines = [f"stage: {stage_name}", ""]
    for parameter_name in model.parameter_names:
        counter = Counter(
            _stringify_param(row.params.get(parameter_name))
            for row in rows
            if parameter_name in row.params
        )
        if not counter:
            continue
        rendered = ", ".join(
            f"{value} ({count})" for value, count in sorted(counter.items(), key=lambda item: item[0])
        )
        lines.append(f"{parameter_name}: {rendered}")
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

        self.path_var = tk.StringVar(value=str(initial_path) if initial_path else "")
        self.status_var = tk.StringVar(value="Idle")
        self.coverage_stage_var = tk.StringVar(value="")
        self.coverage_param_var = tk.StringVar(value="")
        self.coverage_value_var = tk.StringVar(value="score")
        self.preview_stage_var = tk.StringVar(value="")
        self.preview_candidate_var = tk.StringVar(value="")
        self.preview_dataset_var = tk.StringVar(value="")

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

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.coverage_tab, text="Coverage")
        self.notebook.add(self.progression_tab, text="Progression")
        self.notebook.add(self.significance_tab, text="Significance")

        self._build_overview_tab()
        self._build_coverage_tab()
        self._build_progression_tab()
        self._build_significance_tab()
        self._build_metric_tab()

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
        self.coverage_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_coverage())
        self.ttk.Label(controls, text="Parameter").pack(side="left", padx=(8, 0))
        self.coverage_param_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_param_var,
            state="readonly",
            width=18,
        )
        self.coverage_param_combo.pack(side="left", padx=4)
        self.coverage_param_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_coverage())
        self.ttk.Label(controls, text="Value").pack(side="left", padx=(8, 0))
        self.coverage_value_combo = self.ttk.Combobox(
            controls,
            textvariable=self.coverage_value_var,
            state="readonly",
            width=22,
        )
        self.coverage_value_combo.pack(side="left", padx=4)
        self.coverage_value_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_coverage())

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

    def _populate_from_model(self, model: SweepVisualizationModel) -> None:
        self._set_text_widget(self.overview_text, _format_overview_text(model))
        self._set_text_widget(self.progression_text, "\n".join(_progression_summary_lines(model)))
        self._set_text_widget(self.significance_text, _format_significance_text(model))

        metric_lines = ["family, alpha, count, mean_score, primary_metric_mean, mean_runtime_seconds"]
        metric_lines.extend(
            ", ".join(
                [
                    str(item.get("family")),
                    f"{float(item.get('alpha', 0.0)):.3f}",
                    str(item.get("count")),
                    str(item.get("mean_score")),
                    str(item.get("primary_metric_mean")),
                    str(item.get("mean_runtime_seconds")),
                ]
            )
            for item in model.alpha_family_summary
        )
        self._set_text_widget(self.metric_text, "\n".join(metric_lines))

        stage_values = model.stage_order
        self.coverage_stage_combo["values"] = stage_values
        self.coverage_stage_var.set(model.primary_stage_name)
        self.coverage_param_combo["values"] = model.parameter_names
        self.coverage_param_var.set(model.parameter_names[0] if model.parameter_names else "")
        value_options = _value_options(model)
        self.coverage_value_combo["values"] = value_options
        self.coverage_value_var.set(
            "score" if "score" in value_options else (value_options[0] if value_options else "")
        )

        preview_stage_values = [
            stage_name
            for stage_name in model.stage_order
            if any(row.preview_paths for row in _rows_for_stage(model, stage_name))
        ]
        self.preview_stage_combo["values"] = preview_stage_values
        self.preview_stage_var.set(preview_stage_values[0] if preview_stage_values else "")
        self._refresh_preview_candidates()

        self._update_metric_tab_visibility(bool(model.alpha_family_summary))
        self.refresh_coverage()
        self._draw_figure("overview", self.overview_plot_frame, create_overview_figure(model))
        self._draw_figure("progression", self.progression_plot_frame, create_progression_figure(model))
        self._draw_figure("significance", self.significance_plot_frame, create_significance_figure(model))
        self._draw_figure("metric", self.metric_plot_frame, create_metric_formulation_figure(model))

    def _update_metric_tab_visibility(self, visible: bool) -> None:
        if visible and not self._metric_tab_visible:
            self.notebook.add(self.metric_tab, text="Metric Formulations")
            self._metric_tab_visible = True
        elif not visible and self._metric_tab_visible:
            self.notebook.forget(self.metric_tab)
            self._metric_tab_visible = False

    def refresh_coverage(self) -> None:
        if self.model is None:
            return
        stage_name = self.coverage_stage_var.get() or self.model.primary_stage_name
        parameter_name = self.coverage_param_var.get() or (
            self.model.parameter_names[0] if self.model.parameter_names else None
        )
        value_option = self.coverage_value_var.get() or "score"
        self._set_text_widget(self.coverage_text, _format_coverage_text(self.model, stage_name))
        self._draw_figure(
            "coverage",
            self.coverage_plot_frame,
            create_coverage_figure(
                self.model,
                stage_name=stage_name,
                parameter_name=parameter_name,
                value_option=value_option,
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    initial_path = Path(args.sweep_dir) if args.sweep_dir else None
    launch_sweep_visualization(initial_path=initial_path)


if __name__ == "__main__":
    main()

"""Generate/rebuild analysis and summary artifacts from sweep outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

from moment_images.parameter_sweep import (
    SweepExperimentSpec,
    VALIDATION_ORACLE_DIR_NAME,
    VALIDATION_ORACLE_STAGE_NAME,
    _artifact_paths_payload,
    _best_configs_payload,
    _metric_records_from_stage_results,
    _snapshot_generation_products,
    _write_leaderboard_csv,
    _write_root_ranked_outputs,
    _write_stage_outputs,
    _write_validation_phase_artifacts,
    score_and_rank_runs,
)
from moment_images.reliable_io import read_json_first_valid, write_json_atomic, write_text_atomic
from moment_images.sweep_analysis import (
    _load_summary_records_with_fallback,
    analyze_summary_data,
    generate_alpha_report,
    save_analysis,
)


SUCCESSFUL_REBUILD_STATUSES = {"ok", "skipped_valid", "recomputed"}


class RebuildDataError(RuntimeError):
    """Raised when rebuild inputs cannot be reconciled safely."""

    def __init__(self, message: str, diagnostics: Optional[Dict[str, object]] = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass
class RunStateScope:
    run_state: Optional[Dict[str, object]]
    experiment_root: Path
    state_path: Optional[Path] = None
    state_source: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class StageDiscovery:
    stage_name: str
    output_dir: Path
    source: str
    results: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_entries: List[Dict[str, object]] = field(default_factory=list)
    filesystem_extra_runs: List[Dict[str, object]] = field(default_factory=list)

    def to_report(self) -> Dict[str, object]:
        return {
            "stage": self.stage_name,
            "output_dir": str(self.output_dir),
            "source": self.source,
            "num_results": len(self.results),
            "warnings": list(self.warnings),
            "skipped_entries": list(self.skipped_entries),
            "filesystem_extra_runs": list(self.filesystem_extra_runs),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analysis or rebuild summary artifacts from sweep outputs."
    )
    parser.add_argument("output_dir", help="Sweep output directory")
    parser.add_argument("--spec", help="Optional spec JSON. Required for rebuilding stage/final summaries.")
    parser.add_argument(
        "--rebuild-summaries",
        action="store_true",
        help="Rebuild stage1/stage2/final summary+leaderboard artifacts from run folders.",
    )
    parser.add_argument(
        "--repair-run-state",
        nargs="?",
        const="preview",
        choices=("preview", "apply"),
        help=(
            "Preview or apply run_state.json repairs for raw run folders missing from state. "
            "Requires --rebuild-summaries and --spec. Omit the value to preview."
        ),
    )
    parser.add_argument(
        "--primary-metric",
        default="laplacian_variance",
        help="Primary ranking metric for alpha report (default: laplacian_variance)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/corrupt artifacts instead of tolerating partial outputs.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> Dict[str, object]:
    payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    return cast(Dict[str, object], payload)


def _read_json_dict(
    path: Path,
    *,
    label: str,
    warnings: Optional[List[str]] = None,
) -> Tuple[Dict[str, object], Path]:
    try:
        payload, source_path, read_warnings = read_json_first_valid(
            path,
            validator=lambda obj: isinstance(obj, dict),
        )
    except Exception as exc:  # noqa: BLE001 - wrap with rebuild-specific context.
        raise RebuildDataError(
            f"Failed to read {label}: {type(exc).__name__}: {exc}",
            diagnostics={
                "artifact": label,
                "path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    if warnings is not None:
        warnings.extend(read_warnings)
    return cast(Dict[str, object], payload), source_path


def _load_config_for_run(
    *,
    run_dir: Path,
    metric_record: Dict[str, object],
    warnings: List[str],
) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    try:
        config, _ = _read_json_dict(config_path, label=f"config artifact {config_path}", warnings=warnings)
        return config
    except RebuildDataError as exc:
        embedded_config = metric_record.get("config")
        if isinstance(embedded_config, dict):
            warnings.append(
                f"{config_path}: using metric_record.config because config.json could not be read "
                f"({type(exc.__cause__).__name__ if exc.__cause__ is not None else type(exc).__name__})."
            )
            return dict(cast(Dict[str, object], embedded_config))
        warnings.append(f"{config_path}: config.json unavailable and metric_record.config missing; using empty config.")
        return {}


def _coerce_nonempty_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _candidate_signature_from_config_payload(config_payload: Mapping[str, object]) -> str:
    return _hash_payload(dict(config_payload))


def _run_signature(stage_name: str, candidate_signature: str, dataset_signature: str) -> str:
    return _hash_payload(
        {
            "stage": stage_name,
            "candidate_signature": candidate_signature,
            "dataset_signature": dataset_signature,
        }
    )


def _run_state_path_for_scope(output_dir: Path) -> Tuple[Optional[Path], Path]:
    direct = output_dir / "run_state.json"
    if direct.exists() or Path(f"{direct}.bak").exists():
        return direct, output_dir
    if output_dir.name == "metric_experiments":
        parent = output_dir.parent
        parent_state = parent / "run_state.json"
        if parent_state.exists() or Path(f"{parent_state}.bak").exists():
            return parent_state, parent
    return None, output_dir


def _load_run_state_for_scope(output_dir: Path) -> RunStateScope:
    state_path, experiment_root = _run_state_path_for_scope(output_dir)
    if state_path is None:
        return RunStateScope(run_state=None, experiment_root=experiment_root)
    try:
        payload, source_path, warnings = read_json_first_valid(
            state_path,
            validator=lambda obj: isinstance(obj, dict),
        )
    except Exception as exc:  # noqa: BLE001 - fail-safe rebuilds must not guess without state.
        raise RebuildDataError(
            f"Failed to read run_state.json for {output_dir}: {type(exc).__name__}: {exc}",
            diagnostics={
                "artifact": "run_state.json",
                "output_dir": str(output_dir),
                "state_path": str(state_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    return RunStateScope(
        run_state=cast(Dict[str, object], payload),
        experiment_root=experiment_root,
        state_path=state_path,
        state_source=source_path,
        warnings=warnings,
    )


def _active_generation_id(run_state: Optional[Dict[str, object]]) -> Optional[str]:
    if not isinstance(run_state, dict):
        return None
    generation_id = _coerce_nonempty_str(run_state.get("active_generation_id"))
    if generation_id is not None:
        return generation_id
    generations = [
        item
        for item in cast(List[object], run_state.get("spec_generations", []))
        if isinstance(item, dict) and _coerce_nonempty_str(item.get("generation_id")) is not None
    ]
    if generations:
        return _coerce_nonempty_str(cast(Dict[str, object], generations[-1]).get("generation_id"))
    return None


def _path_relative_to(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None


def _relocated_run_dir_candidates(
    *,
    run_dir: Path,
    output_dir: Path,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
    run_state: Optional[Dict[str, object]],
) -> List[Path]:
    candidates: List[Path] = []
    if isinstance(run_state, dict):
        compatibility = run_state.get("path_compatibility")
        if isinstance(compatibility, dict):
            old_output_dir = _coerce_nonempty_str(compatibility.get("output_dir"))
            if old_output_dir is not None:
                suffix = _path_relative_to(run_dir, Path(old_output_dir))
                if suffix is not None:
                    candidates.append(output_dir / suffix)

    parts = list(run_dir.parts)
    if "runs" in parts:
        runs_index = len(parts) - 1 - parts[::-1].index("runs")
        candidates.append(output_dir / Path(*parts[runs_index:]))

    candidates.append(output_dir / "runs" / stage_name / candidate_id / dataset_id)
    return candidates


def _resolve_entry_run_dir(
    entry: Dict[str, object],
    output_dir: Path,
    stage_name: str,
    run_state: Optional[Dict[str, object]],
) -> Path:
    candidate_id = str(entry.get("candidate_id", ""))
    dataset_id = str(entry.get("dataset_id", ""))
    run_dir_text = _coerce_nonempty_str(entry.get("run_dir"))
    if run_dir_text:
        run_dir = Path(run_dir_text)
        if run_dir.exists():
            return run_dir
        scoped = output_dir / run_dir
        if scoped.exists():
            return scoped
        for relocated in _relocated_run_dir_candidates(
            run_dir=run_dir,
            output_dir=output_dir,
            stage_name=stage_name,
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            run_state=run_state,
        ):
            if relocated.exists():
                return relocated

    return output_dir / "runs" / stage_name / candidate_id / dataset_id


def _run_state_entry_brief(entry: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_uid": entry.get("run_uid"),
        "stage": entry.get("stage"),
        "candidate_id": entry.get("candidate_id"),
        "dataset_id": entry.get("dataset_id"),
        "status": entry.get("status"),
        "active": entry.get("active", True),
        "run_dir": entry.get("run_dir"),
        "generation_id": entry.get("generation_id"),
    }


def _skip_run_state_entry(discovery: StageDiscovery, entry: Dict[str, object], reason: str) -> None:
    skipped = _run_state_entry_brief(entry)
    skipped["reason"] = reason
    discovery.skipped_entries.append(skipped)


def _normalize_metric_record_identity_from_entry(
    *,
    metric_record: Dict[str, object],
    entry: Dict[str, object],
    stage_name: str,
    run_uid: str,
    candidate_id: str,
    dataset_id: str,
    strategy: str,
    config: Dict[str, object],
) -> Dict[str, object]:
    normalized = dict(metric_record)
    normalized.update(
        {
            "run": run_uid,
            "run_uid": run_uid,
            "stage": stage_name,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "config": dict(config),
        }
    )
    for key in ("candidate_signature", "dataset_signature", "run_signature", "artifact_paths"):
        if key in entry:
            normalized[key] = entry[key]
    return normalized


def _run_result_from_entry(
    *,
    output_dir: Path,
    stage_name: str,
    entry: Dict[str, object],
    active_generation_id: Optional[str],
    run_state: Optional[Dict[str, object]],
    discovery: StageDiscovery,
) -> Optional[Dict[str, object]]:
    if str(entry.get("stage", "")) != stage_name:
        return None
    if entry.get("active", True) is False:
        _skip_run_state_entry(discovery, entry, "inactive")
        return None
    status = str(entry.get("status", "") or "")
    if status not in SUCCESSFUL_REBUILD_STATUSES:
        _skip_run_state_entry(discovery, entry, f"status:{status or '<empty>'}")
        return None

    candidate_id = _coerce_nonempty_str(entry.get("candidate_id"))
    dataset_id = _coerce_nonempty_str(entry.get("dataset_id"))
    if candidate_id is None or dataset_id is None:
        raise RebuildDataError(
            f"Active successful {stage_name} run_state entry is missing candidate_id or dataset_id.",
            diagnostics={"stage": stage_name, "entry": _run_state_entry_brief(entry)},
        )

    run_dir = _resolve_entry_run_dir(entry, output_dir, stage_name, run_state)
    metric_record_path = run_dir / "metric_record.json"
    run_uid = _coerce_nonempty_str(entry.get("run_uid")) or f"{stage_name}_{candidate_id}_{dataset_id}"
    try:
        metric_record, _ = _read_json_dict(
            metric_record_path,
            label=f"metric_record for {run_uid}",
            warnings=discovery.warnings,
        )
    except RebuildDataError as exc:
        diagnostics = dict(exc.diagnostics)
        diagnostics.update(
            {
                "stage": stage_name,
                "run_uid": run_uid,
                "entry": _run_state_entry_brief(entry),
                "run_dir": str(run_dir),
            }
        )
        raise RebuildDataError(
            f"Active successful {stage_name} run '{run_uid}' cannot be rebuilt from metric_record.json.",
            diagnostics=diagnostics,
        ) from exc
    config = _load_config_for_run(run_dir=run_dir, metric_record=metric_record, warnings=discovery.warnings)
    generation_id = (
        _coerce_nonempty_str(entry.get("generation_id"))
        or _coerce_nonempty_str(metric_record.get("generation_id"))
        or active_generation_id
    )
    if generation_id is not None:
        metric_record["generation_id"] = generation_id

    for key in ("candidate_signature", "dataset_signature", "run_signature", "artifact_paths"):
        if key in entry and key not in metric_record:
            metric_record[key] = entry[key]
    strategy = (
        _coerce_nonempty_str(entry.get("strategy"))
        or _coerce_nonempty_str(config.get("subpixel_mode"))
        or _coerce_nonempty_str(config.get("strategy"))
        or "unknown"
    )
    metric_record = _normalize_metric_record_identity_from_entry(
        metric_record=metric_record,
        entry=entry,
        stage_name=stage_name,
        run_uid=run_uid,
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        strategy=strategy,
        config=config,
    )

    return {
        "run": run_uid,
        "status": "ok",
        "stage": stage_name,
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "strategy": strategy,
        "run_uid": run_uid,
        "run_dir": str(run_dir),
        "config": config,
        "params": dict(config),
        "runtime_seconds": float(metric_record.get("runtime_seconds", entry.get("runtime_seconds", 0.0)) or 0.0),
        "metric_record": metric_record,
        "candidate_signature": entry.get("candidate_signature") or metric_record.get("candidate_signature"),
        "dataset_signature": entry.get("dataset_signature") or metric_record.get("dataset_signature"),
        "run_signature": entry.get("run_signature") or metric_record.get("run_signature"),
        "generation_id": generation_id,
        "artifact_paths": entry.get("artifact_paths") or metric_record.get("artifact_paths") or _artifact_paths_payload(run_dir),
    }


def _infer_run_ids_from_path(
    metric_record_path: Path,
    config: Dict[str, object],
    known_dataset_ids: Sequence[str],
) -> Tuple[str, str]:
    candidate_part = metric_record_path.parents[1].name
    dataset_part = metric_record_path.parents[0].name
    config_dataset = _coerce_nonempty_str(config.get("dataset_id"))
    if config_dataset is not None:
        dataset_id = config_dataset
        if dataset_part == dataset_id:
            candidate_id = candidate_part
        elif candidate_part == dataset_id:
            candidate_id = dataset_part
        else:
            candidate_id = candidate_part
        return candidate_id, dataset_id

    if candidate_part in known_dataset_ids and dataset_part not in known_dataset_ids:
        return dataset_part, candidate_part
    return candidate_part, dataset_part


def _filesystem_run_refs(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    for metric_record_path in sorted(output_dir.glob(f"runs/{stage_name}/*/*/metric_record.json")):
        candidate_id, dataset_id = _infer_run_ids_from_path(metric_record_path, {}, known_dataset_ids)
        refs.append(
            {
                "stage": stage_name,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "run_dir": str(metric_record_path.parent),
                "metric_record": str(metric_record_path),
            }
        )
    return refs


def _filesystem_run_dir_refs(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    stage_root = output_dir / "runs" / stage_name
    if not stage_root.exists():
        return refs
    for run_dir in sorted(path for path in stage_root.glob("*/*") if path.is_dir()):
        metric_record_path = run_dir / "metric_record.json"
        candidate_id, dataset_id = _infer_run_ids_from_path(metric_record_path, {}, known_dataset_ids)
        refs.append(
            {
                "stage": stage_name,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "run_dir": str(run_dir),
                "metric_record": str(metric_record_path),
            }
        )
    return refs


def _artifact_validation_for_recovered_run(run_dir: Path) -> Dict[str, object]:
    required_paths = {
        "config": run_dir / "config.json",
        "metric_record": run_dir / "metric_record.json",
        "rectified_views": run_dir / "rectified_views.npy",
        "workspace": run_dir / "workspace.npz",
        "avg_rectified": run_dir / "avg_rectified.png",
    }
    missing = [
        f"{key}:{path.name}"
        for key, path in required_paths.items()
        if not path.exists() and not Path(f"{path}.bak").exists()
    ]
    return {
        "run_dir": str(run_dir),
        "valid": not missing,
        "missing": missing,
        "corrupt": [],
    }


def _repair_rejection(ref: Dict[str, object], reason: str, **details: object) -> Dict[str, object]:
    rejected = {
        "stage": ref.get("stage"),
        "candidate_id": ref.get("candidate_id"),
        "dataset_id": ref.get("dataset_id"),
        "run_dir": ref.get("run_dir"),
        "metric_record": ref.get("metric_record"),
        "reason": reason,
    }
    rejected.update(details)
    return rejected


def _repair_entry_summary(entry: Dict[str, object], *, generation_inferred: bool) -> Dict[str, object]:
    return {
        "run_uid": entry.get("run_uid"),
        "stage": entry.get("stage"),
        "candidate_id": entry.get("candidate_id"),
        "dataset_id": entry.get("dataset_id"),
        "generation_id": entry.get("generation_id"),
        "generation_inferred": generation_inferred,
        "run_dir": entry.get("run_dir"),
        "candidate_signature": entry.get("candidate_signature"),
        "dataset_signature": entry.get("dataset_signature"),
        "run_signature": entry.get("run_signature"),
        "artifact_paths": entry.get("artifact_paths"),
        "entry": entry,
    }


def _build_repair_entry_from_ref(
    ref: Dict[str, object],
    *,
    active_generation_id: Optional[str],
    known_dataset_ids: Sequence[str],
    warnings: List[str],
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    stage_name = str(ref["stage"])
    run_dir = Path(str(ref["run_dir"]))
    metric_record_path = run_dir / "metric_record.json"
    if not metric_record_path.exists() and not Path(f"{metric_record_path}.bak").exists():
        return None, _repair_rejection(ref, "metric_record_missing")

    try:
        metric_record, _ = _read_json_dict(
            metric_record_path,
            label=f"metric_record artifact {metric_record_path}",
            warnings=warnings,
        )
    except RebuildDataError as exc:
        return None, _repair_rejection(ref, "metric_record_unreadable", error=str(exc))

    config = _load_config_for_run(run_dir=run_dir, metric_record=metric_record, warnings=warnings)
    if not config:
        return None, _repair_rejection(ref, "config_unusable")

    candidate_id, dataset_id = _infer_run_ids_from_path(metric_record_path, config, known_dataset_ids)
    ref = dict(ref)
    ref["candidate_id"] = candidate_id
    ref["dataset_id"] = dataset_id
    if dataset_id not in set(known_dataset_ids):
        return None, _repair_rejection(ref, "unknown_dataset")

    metric_generation_id = _coerce_nonempty_str(metric_record.get("generation_id"))
    if (
        active_generation_id is not None
        and metric_generation_id is not None
        and metric_generation_id != active_generation_id
    ):
        return None, _repair_rejection(
            ref,
            "generation_mismatch",
            generation_id=metric_generation_id,
            active_generation_id=active_generation_id,
        )

    generation_id = metric_generation_id or active_generation_id
    if generation_id is None:
        return None, _repair_rejection(ref, "generation_missing")

    run_uid = f"{stage_name}_{candidate_id}_{dataset_id}"
    candidate_signature = _coerce_nonempty_str(metric_record.get("candidate_signature"))
    if candidate_signature is None:
        candidate_signature = _candidate_signature_from_config_payload(config)
    dataset_signature = _coerce_nonempty_str(metric_record.get("dataset_signature"))
    if dataset_signature is None:
        dataset_signature = _hash_payload({"dataset_id": dataset_id})
    run_signature = _coerce_nonempty_str(metric_record.get("run_signature"))
    if run_signature is None:
        run_signature = _run_signature(stage_name, candidate_signature, dataset_signature)
    artifact_paths = metric_record.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        artifact_paths = _artifact_paths_payload(run_dir)

    entry: Dict[str, object] = {
        "run_uid": run_uid,
        "stage": stage_name,
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "status": "ok",
        "active": True,
        "generation_id": generation_id,
        "run_dir": str(run_dir),
        "candidate_signature": candidate_signature,
        "dataset_signature": dataset_signature,
        "run_signature": run_signature,
        "artifact_paths": dict(cast(Dict[str, object], artifact_paths)),
        "attempts": [],
        "artifact_validation": _artifact_validation_for_recovered_run(run_dir),
    }
    return _repair_entry_summary(entry, generation_inferred=metric_generation_id is None), None


def _repair_scope_specs(output_dir: Path) -> List[Tuple[Path, str]]:
    if _is_metric_phase_dir(output_dir):
        return [(output_dir, "metric_phase")]
    specs = [(output_dir, "stage1"), (output_dir, "stage2")]
    metric_output_dir = output_dir / "metric_experiments"
    if _has_report_inputs(metric_output_dir):
        specs.append((metric_output_dir, "metric_phase"))
    return specs


def _collect_run_state_repairs(
    output_dir: Path,
    dataset_ids: Sequence[str],
) -> Tuple[RunStateScope, Dict[str, object]]:
    run_state_scope = _load_run_state_for_scope(output_dir)
    run_state = run_state_scope.run_state
    active_generation = _active_generation_id(run_state)
    warnings = list(run_state_scope.warnings)
    adoptable_entries: List[Dict[str, object]] = []
    rejected_entries: List[Dict[str, object]] = []
    scope_reports: List[Dict[str, object]] = []

    if not isinstance(run_state, dict):
        warnings.append("No run_state.json was found; repair is not needed because rebuild will use filesystem discovery.")
    else:
        for stage_output_dir, stage_name in _repair_scope_specs(output_dir):
            represented_keys = _run_state_keys_for_stage(run_state, stage_name)
            refs = _filesystem_run_dir_refs(stage_output_dir, stage_name, dataset_ids)
            scope_reports.append(
                {
                    "stage": stage_name,
                    "output_dir": str(stage_output_dir),
                    "num_run_dirs": len(refs),
                }
            )
            for ref in refs:
                path_key = (str(ref["candidate_id"]), str(ref["dataset_id"]))
                if path_key in represented_keys:
                    continue
                entry_summary, rejection = _build_repair_entry_from_ref(
                    ref,
                    active_generation_id=active_generation,
                    known_dataset_ids=dataset_ids,
                    warnings=warnings,
                )
                if rejection is not None:
                    rejected_entries.append(rejection)
                    continue
                if entry_summary is None:
                    continue
                key = (str(entry_summary["candidate_id"]), str(entry_summary["dataset_id"]))
                if key in represented_keys:
                    continue
                adoptable_entries.append(entry_summary)

    report = {
        "status": "preview",
        "mode": "preview",
        "generated_at": _utc_now_iso(),
        "output_dir": str(output_dir),
        "state_path": None if run_state_scope.state_path is None else str(run_state_scope.state_path),
        "state_source": None if run_state_scope.state_source is None else str(run_state_scope.state_source),
        "active_generation_id": active_generation,
        "counts": {
            "adoptable": len(adoptable_entries),
            "rejected": len(rejected_entries),
            "applied": 0,
            "scopes": len(scope_reports),
        },
        "adoptable_entries": adoptable_entries,
        "rejected_entries": rejected_entries,
        "warnings": warnings,
        "scopes": scope_reports,
    }
    return run_state_scope, report


def _format_run_state_repair_report_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Run State Repair Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Mode: {report.get('mode')}",
        f"- Generated at: {report.get('generated_at')}",
        f"- Output directory: {report.get('output_dir')}",
        f"- State path: {report.get('state_path')}",
        f"- Active generation: {report.get('active_generation_id')}",
        "",
        "## Counts",
        "",
    ]
    counts = report.get("counts", {})
    if isinstance(counts, dict):
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")

    adoptable = cast(List[object], report.get("adoptable_entries", []))
    if adoptable:
        lines.extend(["", "## Adoptable Entries", ""])
        for item in adoptable:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('run_uid')} ({item.get('stage')}, {item.get('candidate_id')}, "
                    f"{item.get('dataset_id')}, generation={item.get('generation_id')})"
                )

    rejected = cast(List[object], report.get("rejected_entries", []))
    if rejected:
        lines.extend(["", "## Rejected Entries", ""])
        for item in rejected:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('stage')} {item.get('candidate_id')}/{item.get('dataset_id')}: "
                    f"{item.get('reason')} ({item.get('run_dir')})"
                )

    warnings = cast(List[object], report.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def _write_run_state_repair_report(output_dir: Path, report: Dict[str, object]) -> Tuple[Path, Path]:
    json_path = output_dir / "run_state_repair_report.json"
    markdown_path = output_dir / "run_state_repair_report.md"
    write_json_atomic(json_path, report, indent=2, sort_keys=False, create_backup=True)
    write_text_atomic(markdown_path, _format_run_state_repair_report_markdown(report), create_backup=True)
    return json_path, markdown_path


def _repair_run_state_for_rebuild(output_dir: Path, spec: SweepExperimentSpec, mode: str) -> Dict[str, object]:
    dataset_ids = [dataset.dataset_id for dataset in spec.datasets]
    run_state_scope, report = _collect_run_state_repairs(output_dir, dataset_ids)
    report["mode"] = mode
    report["status"] = "preview" if mode == "preview" else "applied"
    if mode == "apply":
        run_state = run_state_scope.run_state
        if not isinstance(run_state, dict) or run_state_scope.state_path is None:
            raise RebuildDataError(
                "Cannot apply run_state repair because no usable run_state.json was found.",
                diagnostics=report,
            )
        runs = run_state.get("runs", {})
        if not isinstance(runs, dict):
            raise RebuildDataError(
                "Cannot apply run_state repair because run_state.json has no usable 'runs' object.",
                diagnostics=report,
            )
        applied = 0
        for item in cast(List[Dict[str, object]], report["adoptable_entries"]):
            entry = item.get("entry")
            if not isinstance(entry, dict):
                continue
            run_uid = _coerce_nonempty_str(entry.get("run_uid"))
            if run_uid is None:
                continue
            cast(Dict[str, object], runs)[run_uid] = entry
            applied += 1
        cast(Dict[str, object], report["counts"])["applied"] = applied
        write_json_atomic(run_state_scope.state_path, run_state, indent=2, sort_keys=False, create_backup=True)
    _write_run_state_repair_report(output_dir, report)
    return report


def _run_state_keys_for_stage(run_state: Dict[str, object], stage_name: str) -> set[Tuple[str, str]]:
    runs = run_state.get("runs", {})
    if not isinstance(runs, dict):
        raise RebuildDataError(
            "run_state.json has no usable 'runs' object.",
            diagnostics={"stage": stage_name, "runs_type": type(runs).__name__},
        )
    keys: set[Tuple[str, str]] = set()
    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("stage", "")) != stage_name:
            continue
        candidate_id = _coerce_nonempty_str(entry.get("candidate_id"))
        dataset_id = _coerce_nonempty_str(entry.get("dataset_id"))
        if candidate_id is not None and dataset_id is not None:
            keys.add((candidate_id, dataset_id))
    return keys


def _discover_stage_results_from_run_state(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
    run_state_scope: RunStateScope,
) -> StageDiscovery:
    run_state = run_state_scope.run_state
    discovery = StageDiscovery(
        stage_name=stage_name,
        output_dir=output_dir,
        source="run_state",
        warnings=list(run_state_scope.warnings),
    )
    if not isinstance(run_state, dict):
        return discovery

    represented_keys = _run_state_keys_for_stage(run_state, stage_name)
    filesystem_refs = _filesystem_run_refs(output_dir, stage_name, known_dataset_ids)
    extras = [
        ref
        for ref in filesystem_refs
        if (str(ref["candidate_id"]), str(ref["dataset_id"])) not in represented_keys
    ]
    if extras:
        discovery.filesystem_extra_runs = extras
        raise RebuildDataError(
            f"{stage_name} has raw run folder(s) not represented in run_state.json.",
            diagnostics=discovery.to_report(),
        )

    active_generation = _active_generation_id(run_state)
    runs = cast(Dict[str, object], run_state.get("runs", {}))
    if not isinstance(runs, dict):
        raise RebuildDataError(
            "run_state.json has no usable 'runs' object.",
            diagnostics=discovery.to_report(),
        )

    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        result = _run_result_from_entry(
            output_dir=output_dir,
            stage_name=stage_name,
            entry=cast(Dict[str, object], entry),
            active_generation_id=active_generation,
            run_state=run_state,
            discovery=discovery,
        )
        if result is not None:
            discovery.results.append(result)
    discovery.results.sort(key=lambda item: str(item.get("run_uid", item.get("run", ""))))
    return discovery


def _discover_stage_results_from_filesystem(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
) -> StageDiscovery:
    discovery = StageDiscovery(stage_name=stage_name, output_dir=output_dir, source="filesystem")
    for metric_record_path in sorted(output_dir.glob(f"runs/{stage_name}/*/*/metric_record.json")):
        run_dir = metric_record_path.parent
        metric_record, _ = _read_json_dict(
            metric_record_path,
            label=f"metric_record artifact {metric_record_path}",
            warnings=discovery.warnings,
        )
        config = _load_config_for_run(run_dir=run_dir, metric_record=metric_record, warnings=discovery.warnings)
        record_run_name = str(metric_record.get("run", ""))

        candidate_id, dataset_id = _infer_run_ids_from_path(metric_record_path, config, known_dataset_ids)

        generation_id = _coerce_nonempty_str(metric_record.get("generation_id"))
        strategy = str(config.get("subpixel_mode", "unknown"))
        discovery.results.append(
            {
                "run": record_run_name or f"{dataset_id}_{candidate_id}",
                "status": "ok",
                "stage": stage_name,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "strategy": strategy,
                "run_uid": f"{stage_name}_{candidate_id}_{dataset_id}",
                "run_dir": str(run_dir),
                "config": config,
                "params": dict(config),
                "runtime_seconds": float(metric_record.get("runtime_seconds", 0.0)),
                "metric_record": metric_record,
                "generation_id": generation_id,
                "artifact_paths": metric_record.get("artifact_paths") or _artifact_paths_payload(run_dir),
            }
        )
    return discovery


def _discover_stage_results(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
    run_state_scope: RunStateScope,
) -> StageDiscovery:
    if run_state_scope.run_state is not None:
        return _discover_stage_results_from_run_state(output_dir, stage_name, known_dataset_ids, run_state_scope)
    return _discover_stage_results_from_filesystem(output_dir, stage_name, known_dataset_ids)


def _stage_generation_id(
    run_state: Optional[Dict[str, object]],
    results: Sequence[Dict[str, object]],
) -> Optional[str]:
    active_generation = _active_generation_id(run_state)
    if active_generation is not None:
        return active_generation
    for result in results:
        generation_id = _coerce_nonempty_str(result.get("generation_id"))
        if generation_id is not None:
            return generation_id
    return None


def _validate_rebuild_dataset_ids(
    stage_name: str,
    results: Sequence[Dict[str, object]],
    dataset_ids: Sequence[str],
) -> None:
    known = set(dataset_ids)
    unknown_runs = [
        {
            "run": result.get("run"),
            "run_uid": result.get("run_uid"),
            "candidate_id": result.get("candidate_id"),
            "dataset_id": result.get("dataset_id"),
            "run_dir": result.get("run_dir"),
        }
        for result in results
        if str(result.get("dataset_id", "")) not in known
    ]
    if unknown_runs:
        unknown_ids = sorted({str(item.get("dataset_id")) for item in unknown_runs})
        raise RebuildDataError(
            f"Unknown dataset id(s) discovered for {stage_name}: {', '.join(unknown_ids)}.",
            diagnostics={
                "stage": stage_name,
                "known_dataset_ids": list(dataset_ids),
                "unknown_dataset_ids": unknown_ids,
                "runs": unknown_runs,
            },
        )


def _write_ranked_outputs(
    output_dir: Path,
    rankings: Sequence[Dict[str, object]],
    metric_records: Sequence[Dict[str, object]],
) -> None:
    write_json_atomic(
        output_dir / "summary.json",
        list(metric_records),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    write_json_atomic(output_dir / "leaderboard.json", list(rankings), indent=2, sort_keys=False, create_backup=True)
    _write_leaderboard_csv(rankings, output_dir / "leaderboard.csv")
    write_json_atomic(
        output_dir / "best_configs.json",
        _best_configs_payload(rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def _append_discovery_context(
    context: Optional[Dict[str, object]],
    discoveries: Sequence[StageDiscovery],
    active_generation: Optional[str],
) -> None:
    if context is None:
        return
    context["active_generation_id"] = active_generation
    scopes = cast(List[object], context.setdefault("scopes", []))
    warnings = cast(List[object], context.setdefault("warnings", []))
    for discovery in discoveries:
        scopes.append(discovery.to_report())
        warnings.extend(discovery.warnings)


def _mark_artifact_writes_started(context: Optional[Dict[str, object]]) -> None:
    if context is not None:
        context["artifact_writes_started"] = True


def _stage_execution_payload(discovery: StageDiscovery) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "rebuilt": True,
        "source": discovery.source,
    }
    if discovery.warnings:
        payload["warnings"] = list(discovery.warnings)
    if discovery.skipped_entries:
        payload["skipped_entries"] = list(discovery.skipped_entries)
    return payload


def _validation_skip_enabled(spec: SweepExperimentSpec) -> bool:
    validation_spec = spec.validation_oracle_sweep
    return bool(
        validation_spec is not None
        and validation_spec.enabled
        and validation_spec.skip_image_metric_stage
    )


def _load_validation_rankings_for_rebuild(output_dir: Path) -> List[Dict[str, object]]:
    candidates = (
        output_dir / VALIDATION_ORACLE_DIR_NAME / "validation_phase_rankings.json",
        output_dir / "validation_leaderboard.json",
        output_dir / f"{VALIDATION_ORACLE_STAGE_NAME}_summary.json",
    )
    diagnostics: Dict[str, object] = {"checked_paths": [str(path) for path in candidates]}
    for path in candidates:
        if not path.exists() and not Path(f"{path}.bak").exists():
            continue
        try:
            payload, _, _ = read_json_first_valid(path)
        except Exception as exc:  # noqa: BLE001
            diagnostics.setdefault("read_errors", []).append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        rankings_raw: object
        if isinstance(payload, dict):
            rankings_raw = payload.get("rankings", [])
        else:
            rankings_raw = payload
        if not isinstance(rankings_raw, list):
            continue
        rankings = [
            cast(Dict[str, object], item)
            for item in rankings_raw
            if isinstance(item, dict)
        ]
        if rankings:
            return rankings
    raise RebuildDataError(
        "validation rankings are required to rebuild validation-guided skip-mode summaries",
        diagnostics,
    )


def _rebuild_summaries(
    output_dir: Path,
    spec: SweepExperimentSpec,
    context: Optional[Dict[str, object]] = None,
) -> Tuple[int, int, Optional[str]]:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    run_state_scope = _load_run_state_for_scope(output_dir)
    run_state = run_state_scope.run_state
    active_generation = _active_generation_id(run_state)
    stage1_discovery = _discover_stage_results(output_dir, "stage1", dataset_ids, run_state_scope)
    if _validation_skip_enabled(spec):
        validation_discovery = _discover_stage_results(
            output_dir,
            VALIDATION_ORACLE_STAGE_NAME,
            dataset_ids,
            run_state_scope,
        )
        _append_discovery_context(
            context,
            [stage1_discovery, validation_discovery],
            active_generation,
        )

        stage1_results = stage1_discovery.results
        validation_results = validation_discovery.results
        _validate_rebuild_dataset_ids("stage1", stage1_results, dataset_ids)
        _validate_rebuild_dataset_ids(VALIDATION_ORACLE_STAGE_NAME, validation_results, dataset_ids)
        validation_rankings = _load_validation_rankings_for_rebuild(output_dir)
        validation_metric_records = _metric_records_from_stage_results(validation_results)

        _mark_artifact_writes_started(context)
        stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
        _write_stage_outputs(
            "stage1",
            stage1_results,
            stage1_rankings,
            output_dir,
            execution=_stage_execution_payload(stage1_discovery),
            generation_id=_stage_generation_id(run_state, stage1_results),
        )
        _write_validation_phase_artifacts(
            output_dir=output_dir,
            stage_results=validation_results,
            rankings=validation_rankings,
            execution=_stage_execution_payload(validation_discovery),
            generation_id=_stage_generation_id(run_state, validation_results),
        )
        _write_root_ranked_outputs(
            output_dir=output_dir,
            rankings=validation_rankings,
            metric_records=validation_metric_records,
            analyze=spec.analyze,
        )
        return len(stage1_results), len(validation_results), active_generation

    stage2_discovery = _discover_stage_results(output_dir, "stage2", dataset_ids, run_state_scope)
    _append_discovery_context(context, [stage1_discovery, stage2_discovery], active_generation)

    stage1_results = stage1_discovery.results
    stage2_results = stage2_discovery.results
    _validate_rebuild_dataset_ids("stage1", stage1_results, dataset_ids)
    _validate_rebuild_dataset_ids("stage2", stage2_results, dataset_ids)

    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    stage2_rankings: List[Dict[str, object]] = []
    if stage2_results:
        stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)

    _mark_artifact_writes_started(context)
    _write_stage_outputs(
        "stage1",
        stage1_results,
        stage1_rankings,
        output_dir,
        execution=_stage_execution_payload(stage1_discovery),
        generation_id=_stage_generation_id(run_state, stage1_results),
    )

    final_rankings = stage1_rankings
    if stage2_results:
        _write_stage_outputs(
            "stage2",
            stage2_results,
            stage2_rankings,
            output_dir,
            execution=_stage_execution_payload(stage2_discovery),
            generation_id=_stage_generation_id(run_state, stage2_results),
        )
        final_rankings = stage2_rankings

    all_metric_records = [
        cast(Dict[str, object], row["metric_record"]) for row in (stage1_results + stage2_results)
    ]
    _write_ranked_outputs(output_dir, final_rankings, all_metric_records)
    return len(stage1_results), len(stage2_results), active_generation


def _rebuild_metric_phase_summaries(
    output_dir: Path,
    spec: SweepExperimentSpec,
    context: Optional[Dict[str, object]] = None,
) -> Tuple[int, Optional[str], Path]:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    run_state_scope = _load_run_state_for_scope(output_dir)
    run_state = run_state_scope.run_state
    active_generation = _active_generation_id(run_state)
    experiment_root = run_state_scope.experiment_root
    metric_phase_discovery = _discover_stage_results(output_dir, "metric_phase", dataset_ids, run_state_scope)
    _append_discovery_context(context, [metric_phase_discovery], active_generation)

    metric_phase_results = metric_phase_discovery.results
    _validate_rebuild_dataset_ids("metric_phase", metric_phase_results, dataset_ids)
    metric_phase_rankings = score_and_rank_runs(
        metric_phase_results,
        spec.scoring,
        dataset_ids=dataset_ids,
    )
    _mark_artifact_writes_started(context)
    _write_stage_outputs(
        "metric_phase",
        metric_phase_results,
        metric_phase_rankings,
        output_dir,
        execution=_stage_execution_payload(metric_phase_discovery),
        generation_id=_stage_generation_id(run_state, metric_phase_results),
    )
    metric_records = [cast(Dict[str, object], row["metric_record"]) for row in metric_phase_results]
    _write_ranked_outputs(output_dir, metric_phase_rankings, metric_records)
    return len(metric_phase_results), active_generation, experiment_root


def _copy_product_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _snapshot_root_alpha_products(experiment_root: Path, generation_id: str) -> None:
    products_dir = experiment_root / "generations" / generation_id / "products"
    for filename in (
        "alpha_report.json",
        "alpha_report.csv",
        "alpha_report_warnings.json",
        "alpha_report_primary_vs_alpha.png",
        "alpha_report_displacement_vs_alpha.png",
    ):
        _copy_product_if_exists(experiment_root / filename, products_dir / filename)


def _snapshot_active_generation_products(experiment_root: Path, generation_id: Optional[str]) -> None:
    if generation_id is None:
        return
    _snapshot_generation_products(experiment_root, generation_id)
    _snapshot_root_alpha_products(experiment_root, generation_id)


def _format_rebuild_failure_report_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Rebuild Failure Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Generated at: {report.get('generated_at')}",
        f"- Error: {report.get('error_type')}: {report.get('message')}",
        f"- Phase: {report.get('phase')}",
        f"- Output directory: {report.get('output_dir')}",
        f"- Spec path: {report.get('spec_path')}",
        f"- Active generation: {report.get('active_generation_id')}",
        f"- Artifact writes started: {report.get('artifact_writes_started')}",
        "",
        "## Diagnostics",
        "",
    ]
    diagnostics = report.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for key, value in diagnostics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append(f"- {diagnostics}")

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in cast(List[object], warnings):
            lines.append(f"- {warning}")

    scopes = report.get("scopes", [])
    if scopes:
        lines.extend(["", "## Scopes", ""])
        for scope in cast(List[object], scopes):
            lines.append(f"- {scope}")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- Inspect run_state.json and its .bak file.",
            "- Check the listed run folders and metric_record.json artifacts.",
            "- Re-run summary rebuild after resolving the reported mismatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_rebuild_failure_report(
    output_dir: Path,
    exc: BaseException,
    context: Dict[str, object],
) -> Tuple[Path, Path]:
    diagnostics: Dict[str, object] = {}
    if isinstance(exc, RebuildDataError):
        diagnostics.update(exc.diagnostics)
    report = {
        "status": "failed",
        "generated_at": _utc_now_iso(),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "command": list(sys.argv),
        "output_dir": str(output_dir),
        "spec_path": context.get("spec_path"),
        "primary_metric": context.get("primary_metric"),
        "strict": context.get("strict"),
        "phase": context.get("phase"),
        "active_generation_id": context.get("active_generation_id") or diagnostics.get("active_generation_id"),
        "artifact_writes_started": bool(context.get("artifact_writes_started", False)),
        "warnings": list(cast(List[object], context.get("warnings", []))),
        "scopes": list(cast(List[object], context.get("scopes", []))),
        "diagnostics": diagnostics,
    }
    json_path = output_dir / "rebuild_failure_report.json"
    markdown_path = output_dir / "rebuild_failure_report.md"
    write_json_atomic(json_path, report, indent=2, sort_keys=False, create_backup=True)
    write_text_atomic(markdown_path, _format_rebuild_failure_report_markdown(report), create_backup=True)
    return json_path, markdown_path


def _has_report_inputs(output_dir: Path) -> bool:
    summary_files = (
        "summary.json",
        "stage1_summary.json",
        "stage2_summary.json",
        "metric_phase_summary.json",
        "validation_phase_summary.json",
    )
    if any((output_dir / name).exists() for name in summary_files):
        return True
    return any(output_dir.glob("runs/*/*/*/metric_record.json"))


def _is_metric_phase_dir(output_dir: Path) -> bool:
    if output_dir.name == "metric_experiments":
        return True
    if (output_dir / "metric_phase_summary.json").exists():
        return True
    return (output_dir / "runs" / "metric_phase").exists()


def _iter_report_scopes(output_dir: Path) -> List[Tuple[str, Path]]:
    if _is_metric_phase_dir(output_dir):
        return [("metric_phase", output_dir)]

    scopes: List[Tuple[str, Path]] = []
    if _has_report_inputs(output_dir):
        scopes.append(("baseline", output_dir))

    metric_output_dir = output_dir / "metric_experiments"
    if _has_report_inputs(metric_output_dir):
        scopes.append(("metric_phase", metric_output_dir))
    return scopes


def _generate_reports_for_scope(
    output_dir: Path,
    *,
    primary_metric: str,
    tolerate_partial_outputs: bool,
) -> Dict[str, object]:
    records, warnings = _load_summary_records_with_fallback(
        output_dir=output_dir,
        tolerate_partial_outputs=tolerate_partial_outputs,
    )
    analysis = analyze_summary_data(records)
    save_analysis(analysis, output_dir / "analysis.json")
    alpha_report = generate_alpha_report(
        output_dir=output_dir,
        primary_metric=primary_metric,
        tolerate_partial_outputs=tolerate_partial_outputs,
    )
    return {
        "records": records,
        "warnings": warnings,
        "analysis": analysis,
        "alpha_report": alpha_report,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tolerate_partial_outputs = not args.strict
    rebuild_context: Optional[Dict[str, object]] = None
    snapshot_requests: List[Tuple[Path, Optional[str]]] = []

    if args.repair_run_state and not args.rebuild_summaries:
        raise ValueError("--repair-run-state requires --rebuild-summaries.")

    if args.rebuild_summaries:
        rebuild_context = {
            "output_dir": str(output_dir),
            "spec_path": args.spec,
            "primary_metric": args.primary_metric,
            "strict": bool(args.strict),
            "phase": "initializing",
            "artifact_writes_started": False,
            "warnings": [],
            "scopes": [],
        }

    try:
        if args.rebuild_summaries:
            if not args.spec:
                raise ValueError("--rebuild-summaries requires --spec so rankings can be recomputed.")
            cast(Dict[str, object], rebuild_context)["phase"] = "load_spec"
            spec = SweepExperimentSpec.from_dict(_load_json(Path(args.spec)))
            if args.repair_run_state:
                cast(Dict[str, object], rebuild_context)["phase"] = "repair_run_state"
                repair_report = _repair_run_state_for_rebuild(output_dir, spec, str(args.repair_run_state))
                counts = cast(Dict[str, object], repair_report["counts"])
                print(
                    "Run-state repair "
                    f"{repair_report['status']}: adoptable={counts['adoptable']}, "
                    f"rejected={counts['rejected']}, applied={counts['applied']}"
                )
                if args.repair_run_state == "preview":
                    return
            if _is_metric_phase_dir(output_dir):
                cast(Dict[str, object], rebuild_context)["phase"] = "rebuild_metric_phase"
                metric_count, active_generation, experiment_root = _rebuild_metric_phase_summaries(
                    output_dir,
                    spec,
                    rebuild_context,
                )
                snapshot_requests.append((experiment_root, active_generation))
                print(f"Rebuilt stage summaries from run folders: metric_phase={metric_count}")
            else:
                cast(Dict[str, object], rebuild_context)["phase"] = "rebuild_baseline"
                s1_count, s2_count, active_generation = _rebuild_summaries(output_dir, spec, rebuild_context)
                final_stage_label = (
                    VALIDATION_ORACLE_STAGE_NAME if _validation_skip_enabled(spec) else "stage2"
                )
                print(
                    "Rebuilt stage summaries from run folders: "
                    f"stage1={s1_count}, {final_stage_label}={s2_count}"
                )
                metric_output_dir = output_dir / "metric_experiments"
                if _has_report_inputs(metric_output_dir):
                    cast(Dict[str, object], rebuild_context)["phase"] = "rebuild_metric_phase"
                    metric_count, metric_active_generation, _experiment_root = _rebuild_metric_phase_summaries(
                        metric_output_dir,
                        spec,
                        rebuild_context,
                    )
                    active_generation = active_generation or metric_active_generation
                    print(f"Rebuilt metric phase summaries from run folders: metric_phase={metric_count}")
                snapshot_requests.append((output_dir, active_generation))

        report_scopes = _iter_report_scopes(output_dir)
        if not report_scopes:
            raise FileNotFoundError(f"No summary or per-run metric artifacts found under {output_dir}")

        for scope_name, scope_dir in report_scopes:
            if rebuild_context is not None:
                rebuild_context["phase"] = f"generate_reports:{scope_name}"
            report_data = _generate_reports_for_scope(
                scope_dir,
                primary_metric=args.primary_metric,
                tolerate_partial_outputs=tolerate_partial_outputs,
            )
            records = cast(List[Dict[str, object]], report_data["records"])
            warnings = cast(List[str], report_data["warnings"])
            analysis = cast(Dict[str, object], report_data["analysis"])
            alpha_report = cast(Dict[str, object], report_data["alpha_report"])

            print(f"[{scope_name}] Loaded {len(records)} metric records from {scope_dir}")
            if warnings:
                print(f"[{scope_name}] Encountered {len(warnings)} artifact warning(s) during record discovery.")
            print(f"[{scope_name}] analysis scopes: {list(analysis.keys())}")
            print(f"[{scope_name}] alpha report keys: {list(alpha_report.keys())}")

        for experiment_root, generation_id in snapshot_requests:
            if rebuild_context is not None:
                rebuild_context["phase"] = "snapshot_generation_products"
            _snapshot_active_generation_products(experiment_root, generation_id)
    except Exception as exc:
        if rebuild_context is not None:
            try:
                json_path, markdown_path = _write_rebuild_failure_report(output_dir, exc, rebuild_context)
                print(
                    "Summary rebuild failed. "
                    f"Wrote failure reports: {json_path}, {markdown_path}"
                )
            except Exception as report_exc:  # noqa: BLE001 - preserve original failure.
                print(f"Summary rebuild failed, and failure report writing also failed: {report_exc}")
        raise


if __name__ == "__main__":
    main()

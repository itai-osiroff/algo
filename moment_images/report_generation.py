"""Generate/rebuild analysis and summary artifacts from sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

from moment_images.parameter_sweep import (
    SweepExperimentSpec,
    _artifact_paths_payload,
    _best_configs_payload,
    _snapshot_generation_products,
    _write_leaderboard_csv,
    _write_stage_outputs,
    score_and_rank_runs,
)
from moment_images.reliable_io import write_json_atomic
from moment_images.sweep_analysis import (
    _load_summary_records_with_fallback,
    analyze_summary_data,
    generate_alpha_report,
    save_analysis,
)


SUCCESSFUL_REBUILD_STATUSES = {"ok", "skipped_valid", "recomputed"}


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


def _load_json(path: Path) -> Dict[str, object]:
    return cast(Dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _coerce_nonempty_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _load_run_state_for_scope(output_dir: Path) -> Tuple[Optional[Dict[str, object]], Path]:
    state_path, experiment_root = _run_state_path_for_scope(output_dir)
    if state_path is None:
        return None, experiment_root
    try:
        return _load_json(state_path), experiment_root
    except Exception:  # noqa: BLE001 - legacy scan remains available for damaged state.
        return None, experiment_root


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


def _resolve_entry_run_dir(entry: Dict[str, object], output_dir: Path, stage_name: str) -> Path:
    run_dir_text = _coerce_nonempty_str(entry.get("run_dir"))
    if run_dir_text:
        run_dir = Path(run_dir_text)
        if run_dir.is_absolute() or run_dir.exists():
            return run_dir
        scoped = output_dir / run_dir
        if scoped.exists():
            return scoped

    candidate_id = str(entry.get("candidate_id", ""))
    dataset_id = str(entry.get("dataset_id", ""))
    return output_dir / "runs" / stage_name / candidate_id / dataset_id


def _run_result_from_entry(
    *,
    output_dir: Path,
    stage_name: str,
    entry: Dict[str, object],
    active_generation_id: Optional[str],
) -> Optional[Dict[str, object]]:
    if str(entry.get("stage", "")) != stage_name:
        return None
    if entry.get("active", True) is False:
        return None
    if str(entry.get("status", "") or "") not in SUCCESSFUL_REBUILD_STATUSES:
        return None

    candidate_id = _coerce_nonempty_str(entry.get("candidate_id"))
    dataset_id = _coerce_nonempty_str(entry.get("dataset_id"))
    if candidate_id is None or dataset_id is None:
        return None

    run_dir = _resolve_entry_run_dir(entry, output_dir, stage_name)
    metric_record_path = run_dir / "metric_record.json"
    config_path = run_dir / "config.json"
    if not metric_record_path.exists() or not config_path.exists():
        return None

    metric_record = _load_json(metric_record_path)
    config = _load_json(config_path)
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

    run_uid = _coerce_nonempty_str(entry.get("run_uid")) or f"{stage_name}_{candidate_id}_{dataset_id}"
    return {
        "run": _coerce_nonempty_str(metric_record.get("run")) or run_uid,
        "status": "ok",
        "stage": stage_name,
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "strategy": _coerce_nonempty_str(entry.get("strategy"))
        or _coerce_nonempty_str(config.get("subpixel_mode"))
        or "unknown",
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


def _discover_stage_results_from_run_state(
    output_dir: Path,
    stage_name: str,
    run_state: Optional[Dict[str, object]],
) -> Optional[List[Dict[str, object]]]:
    if not isinstance(run_state, dict):
        return None
    active_generation = _active_generation_id(run_state)
    runs = cast(Dict[str, object], run_state.get("runs", {}))
    if not isinstance(runs, dict):
        return []

    results: List[Dict[str, object]] = []
    for entry in runs.values():
        if not isinstance(entry, dict):
            continue
        result = _run_result_from_entry(
            output_dir=output_dir,
            stage_name=stage_name,
            entry=cast(Dict[str, object], entry),
            active_generation_id=active_generation,
        )
        if result is not None:
            results.append(result)
    results.sort(key=lambda item: str(item.get("run_uid", item.get("run", ""))))
    return results


def _discover_stage_results_from_filesystem(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for metric_record_path in sorted(output_dir.glob(f"runs/{stage_name}/*/*/metric_record.json")):
        run_dir = metric_record_path.parent
        config_path = run_dir / "config.json"
        metric_record = _load_json(metric_record_path)
        config = _load_json(config_path) if config_path.exists() else {}
        path_part_a = metric_record_path.parents[1].name
        path_part_b = metric_record_path.parents[0].name
        record_run_name = str(metric_record.get("run", ""))

        if path_part_a in known_dataset_ids and path_part_b not in known_dataset_ids:
            dataset_id, candidate_id = path_part_a, path_part_b
        elif path_part_b in known_dataset_ids and path_part_a not in known_dataset_ids:
            dataset_id, candidate_id = path_part_b, path_part_a
        elif str(config.get("dataset_id", "")) in known_dataset_ids:
            dataset_id = str(config["dataset_id"])
            candidate_id = path_part_b if path_part_b != dataset_id else path_part_a
        else:
            dataset_id = path_part_a
            candidate_id = path_part_b

        generation_id = _coerce_nonempty_str(metric_record.get("generation_id"))
        strategy = str(config.get("subpixel_mode", "unknown"))
        results.append(
            {
                "run": record_run_name or f"{dataset_id}_{candidate_id}",
                "status": "ok",
                "stage": stage_name,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "strategy": strategy,
                "config": config,
                "params": dict(config),
                "runtime_seconds": float(metric_record.get("runtime_seconds", 0.0)),
                "metric_record": metric_record,
                "generation_id": generation_id,
            }
        )
    return results


def _discover_stage_results(
    output_dir: Path,
    stage_name: str,
    known_dataset_ids: Sequence[str],
    run_state: Optional[Dict[str, object]],
) -> List[Dict[str, object]]:
    state_results = _discover_stage_results_from_run_state(output_dir, stage_name, run_state)
    if state_results is not None:
        return state_results
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


def _rebuild_summaries(output_dir: Path, spec: SweepExperimentSpec) -> Tuple[int, int, Optional[str]]:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    run_state, _experiment_root = _load_run_state_for_scope(output_dir)
    stage1_results = _discover_stage_results(output_dir, "stage1", dataset_ids, run_state)
    stage2_results = _discover_stage_results(output_dir, "stage2", dataset_ids, run_state)

    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs(
        "stage1",
        stage1_results,
        stage1_rankings,
        output_dir,
        execution={"rebuilt": True},
        generation_id=_stage_generation_id(run_state, stage1_results),
    )

    final_rankings = stage1_rankings
    if stage2_results:
        stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)
        _write_stage_outputs(
            "stage2",
            stage2_results,
            stage2_rankings,
            output_dir,
            execution={"rebuilt": True},
            generation_id=_stage_generation_id(run_state, stage2_results),
        )
        final_rankings = stage2_rankings

    all_metric_records = [
        cast(Dict[str, object], row["metric_record"]) for row in (stage1_results + stage2_results)
    ]
    _write_ranked_outputs(output_dir, final_rankings, all_metric_records)
    return len(stage1_results), len(stage2_results), _active_generation_id(run_state)


def _rebuild_metric_phase_summaries(output_dir: Path, spec: SweepExperimentSpec) -> Tuple[int, Optional[str], Path]:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    run_state, experiment_root = _load_run_state_for_scope(output_dir)
    metric_phase_results = _discover_stage_results(output_dir, "metric_phase", dataset_ids, run_state)
    metric_phase_rankings = score_and_rank_runs(
        metric_phase_results,
        spec.scoring,
        dataset_ids=dataset_ids,
    )
    _write_stage_outputs(
        "metric_phase",
        metric_phase_results,
        metric_phase_rankings,
        output_dir,
        execution={"rebuilt": True},
        generation_id=_stage_generation_id(run_state, metric_phase_results),
    )
    metric_records = [cast(Dict[str, object], row["metric_record"]) for row in metric_phase_results]
    _write_ranked_outputs(output_dir, metric_phase_rankings, metric_records)
    return len(metric_phase_results), _active_generation_id(run_state), experiment_root


def _snapshot_active_generation_products(experiment_root: Path, generation_id: Optional[str]) -> None:
    if generation_id is None:
        return
    _snapshot_generation_products(experiment_root, generation_id)


def _has_report_inputs(output_dir: Path) -> bool:
    summary_files = (
        "summary.json",
        "stage1_summary.json",
        "stage2_summary.json",
        "metric_phase_summary.json",
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

    if args.rebuild_summaries:
        if not args.spec:
            raise ValueError("--rebuild-summaries requires --spec so rankings can be recomputed.")
        spec = SweepExperimentSpec.from_dict(_load_json(Path(args.spec)))
        if _is_metric_phase_dir(output_dir):
            metric_count, active_generation, experiment_root = _rebuild_metric_phase_summaries(output_dir, spec)
            _snapshot_active_generation_products(experiment_root, active_generation)
            print(f"Rebuilt stage summaries from run folders: metric_phase={metric_count}")
        else:
            s1_count, s2_count, active_generation = _rebuild_summaries(output_dir, spec)
            print(f"Rebuilt stage summaries from run folders: stage1={s1_count}, stage2={s2_count}")
            metric_output_dir = output_dir / "metric_experiments"
            if _has_report_inputs(metric_output_dir):
                metric_count, metric_active_generation, experiment_root = _rebuild_metric_phase_summaries(
                    metric_output_dir,
                    spec,
                )
                active_generation = active_generation or metric_active_generation
                print(f"Rebuilt metric phase summaries from run folders: metric_phase={metric_count}")
            _snapshot_active_generation_products(output_dir, active_generation)

    report_scopes = _iter_report_scopes(output_dir)
    if not report_scopes:
        raise FileNotFoundError(f"No summary or per-run metric artifacts found under {output_dir}")

    for scope_name, scope_dir in report_scopes:
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


if __name__ == "__main__":
    main()

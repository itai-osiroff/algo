"""Generate/rebuild analysis and summary artifacts from sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

from moment_images.parameter_sweep import (
    SweepExperimentSpec,
    _best_configs_payload,
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


def _discover_stage_results(
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

        strategy = str(config.get("subpixel_mode", "unknown"))
        results.append(
            {
                "run": record_run_name or f"{dataset_id}_{candidate_id}",
                "status": "ok",
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "strategy": strategy,
                "config": config,
                "params": dict(config),
                "runtime_seconds": float(metric_record.get("runtime_seconds", 0.0)),
                "metric_record": metric_record,
            }
        )
    return results


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


def _rebuild_summaries(output_dir: Path, spec: SweepExperimentSpec) -> Tuple[int, int]:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    stage1_results = _discover_stage_results(output_dir, "stage1", dataset_ids)
    stage2_results = _discover_stage_results(output_dir, "stage2", dataset_ids)

    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs("stage1", stage1_results, stage1_rankings, output_dir, execution={"rebuilt": True})

    final_rankings = stage1_rankings
    if stage2_results:
        stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)
        _write_stage_outputs("stage2", stage2_results, stage2_rankings, output_dir, execution={"rebuilt": True})
        final_rankings = stage2_rankings

    all_metric_records = [
        cast(Dict[str, object], row["metric_record"]) for row in (stage1_results + stage2_results)
    ]
    _write_ranked_outputs(output_dir, final_rankings, all_metric_records)
    return len(stage1_results), len(stage2_results)


def _rebuild_metric_phase_summaries(output_dir: Path, spec: SweepExperimentSpec) -> int:
    dataset_ids = [item.dataset_id for item in spec.datasets]
    metric_phase_results = _discover_stage_results(output_dir, "metric_phase", dataset_ids)
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
    )
    metric_records = [cast(Dict[str, object], row["metric_record"]) for row in metric_phase_results]
    _write_ranked_outputs(output_dir, metric_phase_rankings, metric_records)
    return len(metric_phase_results)


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
            metric_count = _rebuild_metric_phase_summaries(output_dir, spec)
            print(f"Rebuilt stage summaries from run folders: metric_phase={metric_count}")
        else:
            s1_count, s2_count = _rebuild_summaries(output_dir, spec)
            print(f"Rebuilt stage summaries from run folders: stage1={s1_count}, stage2={s2_count}")
            metric_output_dir = output_dir / "metric_experiments"
            if _has_report_inputs(metric_output_dir):
                metric_count = _rebuild_metric_phase_summaries(metric_output_dir, spec)
                print(f"Rebuilt metric phase summaries from run folders: metric_phase={metric_count}")

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

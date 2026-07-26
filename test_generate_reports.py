import os
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import generate_reports
from moment_images.parameter_sweep import (
    SweepScoringSpec,
    _rank_stage_with_validation_oracle,
)
from moment_images.trio_oracle_metrics import (
    TRIO_LF_SOURCE_KIND,
    TRIO_PRIMARY_SCORE,
    TrioOracleRecord,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run(
    base_dir: Path,
    *,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
    run_name: str,
    metric_value: float,
    formulation_id: str,
    alpha: float,
    generation_id: str | None = None,
) -> None:
    run_dir = base_dir / "runs" / stage_name / candidate_id / dataset_id
    config = {
        "dataset_id": dataset_id,
        "subpixel_mode": "pixel",
        "displacement_metric_family": "ncc_gradient",
        "displacement_alpha": alpha,
        "metric_formulation_id": formulation_id,
    }
    metric_record = {
        "run": run_name,
        "config": dict(config),
        "metric_formulation": {
            "id": formulation_id,
            "family": "ncc_gradient",
            "alpha": alpha,
        },
        "global_and_local_metrics": {
            "laplacian_variance": metric_value,
        },
        "displacement_field_metrics": {"mean_magnitude": 0.5},
        "inter_view_consistency_metrics": {"mean_view_std": 0.2},
        "runtime_seconds": 0.1,
    }
    if generation_id is not None:
        metric_record["generation_id"] = generation_id
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metric_record.json", metric_record)


def _run_dir(base_dir: Path, stage_name: str, candidate_id: str, dataset_id: str) -> Path:
    return base_dir / "runs" / stage_name / candidate_id / dataset_id


def _write_run_state(
    output_dir: Path,
    *,
    active_generation_id: str,
    runs: dict[str, dict[str, object]],
) -> None:
    _write_json(
        output_dir / "run_state.json",
        {
            "schema_version": 2,
            "active_generation_id": active_generation_id,
            "spec_generations": [
                {"generation_id": "gen_old", "action": "run"},
                {"generation_id": active_generation_id, "action": "extend"},
            ],
            "runs": runs,
        },
    )


def _run_state_entry(
    *,
    stage_name: str,
    candidate_id: str,
    dataset_id: str,
    run_dir: Path,
    generation_id: str | None,
    active: bool = True,
    status: str = "ok",
) -> dict[str, object]:
    run_uid = f"{stage_name}_{candidate_id}_{dataset_id}"
    return {
        "run_uid": run_uid,
        "stage": stage_name,
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "status": status,
        "active": active,
        "generation_id": generation_id,
        "run_dir": str(run_dir),
        "candidate_signature": f"candidate-{candidate_id}",
        "dataset_signature": f"dataset-{dataset_id}",
        "run_signature": f"signature-{run_uid}",
        "artifact_paths": {"metric_record": str(run_dir / "metric_record.json")},
    }


def _write_spec(
    spec_path: Path,
    *,
    validation_enabled: bool = False,
    validation_skip: bool = False,
) -> None:
    payload: dict[str, object] = {
        "datasets": [{"id": "d1", "views_path": "unused.npy"}],
        "strategies": ["pixel"],
        "stages": {
            "mode": "single_stage",
            "total_budget_per_dataset": 1,
            "stage1_fraction": 1.0,
            "top_k": 1,
        },
        "scoring": {
            "metric_weights": {"laplacian_variance": 1.0},
            "robustness_lambda": 0.0,
            "missing_dataset_penalty": 0.0,
        },
        "analyze": True,
    }
    if validation_enabled or validation_skip:
        if validation_enabled and not validation_skip:
            cast_stages = payload["stages"]
            assert isinstance(cast_stages, dict)
            cast_stages["mode"] = "two_stage"
        payload["validation_oracle_sweep"] = {
            "enabled": True,
            "top_k": 1,
            "budget_per_dataset": 1,
            "skip_image_metric_stage": bool(validation_skip),
            "scenes": [
                {
                    "dataset_id": "d1",
                    "validation_workspace": "unused_validation_workspace.npz",
                }
            ],
        }
    _write_json(
        spec_path,
        payload,
    )


def _set_rebuild_argv(monkeypatch, output_dir: Path, spec_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )


def _set_live_validation_report_argv(monkeypatch, output_dir: Path, spec_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
            "--live-validation-report",
        ],
    )


def _validation_stage_result(candidate_id: str, dataset_id: str, score: float = 1.0) -> dict[str, object]:
    run_uid = f"validation_phase_{candidate_id}_{dataset_id}"
    return {
        "run": run_uid,
        "status": "ok",
        "stage": "validation_phase",
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "strategy": "pixel",
        "run_uid": run_uid,
        "run_dir": f"C:/source/runs/validation_phase/{candidate_id}/{dataset_id}",
        "config": {"strategy": "pixel", "metric_formulation_id": "baseline"},
        "params": {"strategy": "pixel", "metric_formulation_id": "baseline"},
        "runtime_seconds": 0.1,
        "metric_record": {
            "run": run_uid,
            "config": {"strategy": "pixel", "metric_formulation_id": "baseline"},
            "global_and_local_metrics": {"laplacian_variance": score},
        },
        "generation_id": "gen_new",
        "run_signature": f"signature-{run_uid}",
        "artifact_paths": {
            "run_dir": f"C:/source/runs/validation_phase/{candidate_id}/{dataset_id}",
            "metric_record": f"C:/source/runs/validation_phase/{candidate_id}/{dataset_id}/metric_record.json",
        },
    }


def _oracle_record(
    candidate_id: str,
    dataset_id: str,
    score: float,
    *,
    rankable: bool = True,
) -> TrioOracleRecord:
    return TrioOracleRecord(
        source_kind=TRIO_LF_SOURCE_KIND,
        stage="validation_phase",
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        generation_id="gen_new",
        row_id=f"validation_phase:{candidate_id}:{dataset_id}",
        validation_rank=1 if rankable else None,
        validation_rankable=rankable,
        ranking_exclusion_reason=None if rankable else "underconstrained",
        metrics={TRIO_PRIMARY_SCORE: score},
        diagnostics={},
        source_path="oracle.csv",
    )


def _set_repair_argv(monkeypatch, output_dir: Path, spec_path: Path, mode: str | None = None) -> None:
    argv = [
        "generate_reports.py",
        str(output_dir),
        "--spec",
        str(spec_path),
        "--rebuild-summaries",
        "--repair-run-state",
    ]
    if mode is not None:
        argv.append(mode)
    monkeypatch.setattr(sys, "argv", argv)


def _read_failure_report(output_dir: Path) -> dict[str, object]:
    report_path = output_dir / "rebuild_failure_report.json"
    assert report_path.exists()
    markdown_path = output_dir / "rebuild_failure_report.md"
    assert markdown_path.exists()
    return json.loads(report_path.read_text(encoding="utf-8"))


def _read_repair_report(output_dir: Path) -> dict[str, object]:
    report_path = output_dir / "run_state_repair_report.json"
    assert report_path.exists()
    markdown_path = output_dir / "run_state_repair_report.md"
    assert markdown_path.exists()
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_main_rebuilds_nested_metric_phase_reports(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    metric_dir = output_dir / "metric_experiments"
    spec_path = tmp_path / "spec.json"

    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c0000",
        dataset_id="d1",
        run_name="stage1_c0000_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_run(
        metric_dir,
        stage_name="metric_phase",
        candidate_id="m1c0000",
        dataset_id="d1",
        run_name="metric_phase_m1c0000_d1",
        metric_value=9.0,
        formulation_id="exp_grad",
        alpha=0.25,
    )

    _write_json(
        spec_path,
        {
            "datasets": [{"id": "d1", "views_path": "unused.npy"}],
            "strategies": ["pixel"],
            "stages": {
                "mode": "single_stage",
                "total_budget_per_dataset": 1,
                "stage1_fraction": 1.0,
                "top_k": 1,
            },
            "scoring": {
                "metric_weights": {"laplacian_variance": 1.0},
                "robustness_lambda": 0.0,
                "missing_dataset_penalty": 0.0,
            },
            "analyze": True,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )

    generate_reports.main()

    assert (output_dir / "stage1_summary.json").exists()
    assert (metric_dir / "metric_phase_summary.json").exists()
    assert (metric_dir / "analysis.json").exists()
    assert (metric_dir / "alpha_report.json").exists()

    alpha_report = json.loads((metric_dir / "alpha_report.json").read_text(encoding="utf-8"))
    assert alpha_report["num_runs_considered"] == 1
    assert alpha_report["ranked_runs"][0]["run"] == "metric_phase_m1c0000_d1"
    assert alpha_report["ranked_runs"][0]["formulation_id"] == "exp_grad"


def test_rebuild_summaries_preserves_active_generation_id(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    active_run_dir = _run_dir(output_dir, "stage1", "c_active", "d1")
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=active_run_dir,
                generation_id="gen_new",
            )
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )

    generate_reports.main()

    stage_summary = json.loads((output_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    summary_records = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert stage_summary["generation_id"] == "gen_new"
    assert {row["generation_id"] for row in stage_summary["rankings"]} == {"gen_new"}
    assert {record["generation_id"] for record in summary_records} == {"gen_new"}


def test_rebuild_summaries_promotes_validation_phase_for_skip_mode(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, validation_skip=True)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="validation_phase",
        candidate_id="c_validation",
        dataset_id="d1",
        run_name="validation_phase_c_validation_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    validation_ranking = {
        "candidate_id": "c_validation",
        "strategy": "pixel",
        "config": {"strategy": "pixel", "dataset_id": "d1"},
        "params": {"similarity_window_size": 9},
        "score": 0.98,
        "mean_dataset_score": 0.98,
        "std_dataset_score": 0.0,
        "missing_datasets": [],
        "dataset_scores": {"d1": 0.98},
        "runtime_seconds": {"d1": 0.1},
        "dataset_metrics": {"d1": {"laplacian_variance": 3.0}},
        "rank": 1,
        "stage": "validation_phase",
        "score_metric": "validation_hybrid_v1",
    }
    _write_json(output_dir / "validation_oracle" / "validation_phase_rankings.json", [validation_ranking])
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
            "validation_phase_c_validation_d1": _run_state_entry(
                stage_name="validation_phase",
                candidate_id="c_validation",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "validation_phase", "c_validation", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    best_configs = json.loads((output_dir / "best_configs.json").read_text(encoding="utf-8"))
    validation_summary = json.loads((output_dir / "validation_phase_summary.json").read_text(encoding="utf-8"))
    root_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert leaderboard == [validation_ranking]
    assert best_configs["overall_best"]["candidate_id"] == "c_validation"
    assert validation_summary["rankings"] == [validation_ranking]
    assert [record["run"] for record in root_summary] == ["validation_phase_c_validation_d1"]
    assert (output_dir / "validation_leaderboard.csv").exists()


def test_rebuild_summaries_uses_validation_oracle_manifest_rankings_for_skip_mode(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, validation_skip=True)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="validation_phase",
        candidate_id="c_validation",
        dataset_id="d1",
        run_name="validation_phase_c_validation_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    validation_ranking = {
        "candidate_id": "c_validation",
        "strategy": "pixel",
        "config": {"strategy": "pixel", "dataset_id": "d1"},
        "params": {"similarity_window_size": 9},
        "score": 0.98,
        "mean_dataset_score": 0.98,
        "std_dataset_score": 0.0,
        "missing_datasets": [],
        "dataset_scores": {"d1": 0.98},
        "runtime_seconds": {"d1": 0.1},
        "dataset_metrics": {"d1": {"laplacian_variance": 3.0}},
        "rank": 1,
        "stage": "validation_phase",
        "score_metric": "human_bead_oracle_score_v1",
    }
    _write_json(
        output_dir / "validation_oracle" / "validation_oracle_manifest.json",
        {
            "status": "completed",
            "oracle_mode": "human_bead",
            "validation_phase_rankings": [validation_ranking],
        },
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
            "validation_phase_c_validation_d1": _run_state_entry(
                stage_name="validation_phase",
                candidate_id="c_validation",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "validation_phase", "c_validation", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    validation_summary = json.loads((output_dir / "validation_phase_summary.json").read_text(encoding="utf-8"))

    assert leaderboard == [validation_ranking]
    assert validation_summary["rankings"] == [validation_ranking]


def test_rebuild_summaries_refreshes_validation_phase_without_skip_mode(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, validation_enabled=True)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_stage2",
        dataset_id="d1",
        run_name="stage2_c_stage2_d1",
        metric_value=7.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="validation_phase",
        candidate_id="c_validation",
        dataset_id="d1",
        run_name="validation_phase_c_validation_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    validation_ranking = {
        "candidate_id": "c_validation",
        "strategy": "pixel",
        "config": {"strategy": "pixel", "dataset_id": "d1"},
        "params": {"similarity_window_size": 9},
        "score": 0.98,
        "mean_dataset_score": 0.98,
        "std_dataset_score": 0.0,
        "missing_datasets": [],
        "dataset_scores": {"d1": 0.98},
        "runtime_seconds": {"d1": 0.1},
        "dataset_metrics": {"d1": {"laplacian_variance": 3.0}},
        "rank": 1,
        "stage": "validation_phase",
        "score_metric": "validation_hybrid_v1",
    }
    _write_json(output_dir / "validation_oracle" / "validation_phase_rankings.json", [validation_ranking])
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
            "stage2_c_stage2_d1": _run_state_entry(
                stage_name="stage2",
                candidate_id="c_stage2",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage2", "c_stage2", "d1"),
                generation_id="gen_new",
            ),
            "validation_phase_c_validation_d1": _run_state_entry(
                stage_name="validation_phase",
                candidate_id="c_validation",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "validation_phase", "c_validation", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    validation_summary = json.loads((output_dir / "validation_phase_summary.json").read_text(encoding="utf-8"))
    validation_records = json.loads((output_dir / "validation_summary.json").read_text(encoding="utf-8"))
    root_leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    root_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert validation_summary["rankings"] == [validation_ranking]
    assert [record["run"] for record in validation_records] == ["validation_phase_c_validation_d1"]
    assert root_leaderboard[0]["candidate_id"] == "c_stage2"
    assert [record["run"] for record in root_summary] == ["stage1_c_stage1_d1", "stage2_c_stage2_d1"]
    assert (output_dir / "validation_leaderboard.csv").exists()


def test_rebuild_summaries_allows_stage1_only_partial_validation_skip_mode(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, validation_skip=True)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    stage_summary = json.loads((output_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    root_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert stage_summary["num_runs"] == 1
    assert leaderboard[0]["candidate_id"] == "c_stage1"
    assert [record["run"] for record in root_summary] == ["stage1_c_stage1_d1"]
    assert not (output_dir / "validation_phase_summary.json").exists()
    assert not (output_dir / "rebuild_failure_report.json").exists()


def test_rebuild_summaries_uses_run_state_metric_weights_when_profile_config_is_unavailable(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_payload["scoring"] = {
        "profile_name": "missing_custom_profile",
        "profile_config_path": str(tmp_path / "missing_metrics_config.json"),
        "runtime_penalty_weight": 0.0,
        "robustness_lambda": 0.0,
        "missing_dataset_penalty": 0.0,
        "metric_scope": "global_and_local_metrics",
    }
    _write_json(spec_path, spec_payload)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    run_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    run_state["fingerprint_payload"] = {
        "spec": {
            "scoring": {
                "profile_name": "missing_custom_profile",
                "profile_config_path": str(tmp_path / "old_metrics_config.json"),
                "metric_weights": {"laplacian_variance": 1.0},
            }
        }
    }
    _write_json(output_dir / "run_state.json", run_state)
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert leaderboard[0]["candidate_id"] == "c_stage1"
    assert not (output_dir / "rebuild_failure_report.json").exists()


def test_successful_rebuild_clears_stale_failure_reports(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    for name in (
        "rebuild_failure_report.json",
        "rebuild_failure_report.json.bak",
        "rebuild_failure_report.md",
        "rebuild_failure_report.md.bak",
    ):
        (output_dir / name).write_text("stale failure", encoding="utf-8")
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    for name in (
        "rebuild_failure_report.json",
        "rebuild_failure_report.json.bak",
        "rebuild_failure_report.md",
        "rebuild_failure_report.md.bak",
    ):
        assert not (output_dir / name).exists()


def test_live_validation_report_marks_partial_human_bead_snapshot_without_mutating_source_root(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    annotation_path = tmp_path / "human_beads.json"
    _write_spec(spec_path, validation_skip=True)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    validation_spec = spec_payload["validation_oracle_sweep"]
    assert isinstance(validation_spec, dict)
    datasets = spec_payload["datasets"]
    assert isinstance(datasets, list)
    datasets.append({"id": "d2", "views_path": "unused_d2.npy"})
    validation_spec["oracle_mode"] = "human_bead"
    validation_spec["scenes"] = [
        {
            "dataset_id": "d1",
            "human_bead_annotations": str(annotation_path),
        },
        {
            "dataset_id": "d2",
            "human_bead_annotations": str(annotation_path),
        },
    ]
    _write_json(spec_path, spec_payload)
    _write_json(
        annotation_path,
        {
            "datasets": [
                {
                    "dataset_id": "d1",
                    "image_shape": [96, 96],
                    "default_radius_px": 4.0,
                    "beads": [
                        {"id": "b1", "x": 24.0, "y": 24.0, "radius_px": 4.0},
                        {"id": "b2", "x": 48.0, "y": 48.0, "radius_px": 4.0},
                        {"id": "b3", "x": 72.0, "y": 72.0, "radius_px": 4.0},
                    ],
                },
                {
                    "dataset_id": "d2",
                    "image_shape": [96, 96],
                    "default_radius_px": 4.0,
                    "beads": [
                        {"id": "b1", "x": 24.0, "y": 24.0, "radius_px": 4.0},
                        {"id": "b2", "x": 48.0, "y": 48.0, "radius_px": 4.0},
                        {"id": "b3", "x": 72.0, "y": 72.0, "radius_px": 4.0},
                    ],
                },
            ]
        },
    )
    _write_run(
        output_dir,
        stage_name="validation_phase",
        candidate_id="c_validation",
        dataset_id="d1",
        run_name="validation_phase_c_validation_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    image = np.zeros((96, 96), dtype=np.float32)
    for x, y in ((24, 24), (48, 48), (72, 72)):
        image[y - 2 : y + 3, x - 2 : x + 3] = 1.0
    np.savez_compressed(
        _run_dir(output_dir, "validation_phase", "c_validation", "d1") / "workspace.npz",
        average_rectified_image=image,
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "validation_phase_c_validation_d1": _run_state_entry(
                stage_name="validation_phase",
                candidate_id="c_validation",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "validation_phase", "c_validation", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _write_json(output_dir / "leaderboard.json", [{"candidate_id": "final_before_live_report"}])
    original_run_state = (output_dir / "run_state.json").read_text(encoding="utf-8")
    original_leaderboard = (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    _set_live_validation_report_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    snapshots = list((output_dir / "reports" / "live_validation").iterdir())
    assert len(snapshots) == 1
    snapshot_dir = snapshots[0]
    manifest = json.loads((snapshot_dir / "live_validation_report.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((snapshot_dir / "validation_leaderboard.json").read_text(encoding="utf-8"))
    validation_summary = json.loads((snapshot_dir / "validation_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "provisional"
    assert manifest["source_sweep_dir"] == str(output_dir.resolve())
    assert manifest["observed_dataset_ids"] == ["d1"]
    assert manifest["ranked_dataset_ids"] == ["d1"]
    assert manifest["unrankable_observed_dataset_ids"] == []
    assert manifest["missing_dataset_ids"] == ["d2"]
    assert leaderboard[0]["candidate_id"] == "c_validation"
    assert leaderboard[0]["score_metric"] == "human_bead_oracle_score_v1"
    assert leaderboard[0]["report_status"] == "provisional"
    assert leaderboard[0]["ranked_dataset_ids"] == ["d1"]
    assert leaderboard[0]["missing_report_dataset_ids"] == ["d2"]
    assert leaderboard[0]["unrankable_report_dataset_ids"] == []
    assert validation_summary[0]["run_dir"] == str(
        _run_dir(output_dir, "validation_phase", "c_validation", "d1")
    )
    assert validation_summary[0]["artifact_paths"]["metric_record"] == str(
        _run_dir(output_dir, "validation_phase", "c_validation", "d1") / "metric_record.json"
    )
    assert (snapshot_dir / "metric_review" / "trio_oracle_metrics.json").exists()
    assert (snapshot_dir / "validation_phase_summary.json").exists()
    assert (snapshot_dir / "leaderboard.json").exists()
    assert (snapshot_dir / "summary.json").exists()
    assert (snapshot_dir / "analysis.json").exists()
    assert (snapshot_dir / "alpha_report.json").exists()
    assert (output_dir / "run_state.json").read_text(encoding="utf-8") == original_run_state
    assert (output_dir / "leaderboard.json").read_text(encoding="utf-8") == original_leaderboard
    assert not (output_dir / "validation_oracle").exists()
    assert not (output_dir / "metric_review").exists()


def test_validation_oracle_ranker_preserves_strict_dataset_rankability_by_default():
    stage_results = [
        _validation_stage_result("c_good", "d1", 1.0),
        _validation_stage_result("c_good", "d2", 1.0),
    ]
    validation_import = SimpleNamespace(
        records=[
            _oracle_record("c_good", "d1", 0.8),
            _oracle_record("c_good", "d2", 0.9, rankable=False),
        ]
    )

    with pytest.raises(ValueError, match="d2"):
        _rank_stage_with_validation_oracle(
            stage_name="validation_phase",
            stage_results=stage_results,
            validation_import=validation_import,
            dataset_ids=["d1", "d2"],
            scoring=SweepScoringSpec(robustness_lambda=0.0, missing_dataset_penalty=10.0),
            score_metric=TRIO_PRIMARY_SCORE,
        )


def test_validation_oracle_ranker_allows_live_rankable_subset_without_missing_penalty():
    stage_results = [
        _validation_stage_result("c_good", "d1", 1.0),
        _validation_stage_result("c_good", "d2", 1.0),
        _validation_stage_result("c_bad", "d1", 1.0),
        _validation_stage_result("c_bad", "d2", 1.0),
    ]
    validation_import = SimpleNamespace(
        records=[
            _oracle_record("c_good", "d1", 0.8),
            _oracle_record("c_good", "d2", 0.99, rankable=False),
            _oracle_record("c_bad", "d1", 0.6),
            _oracle_record("c_bad", "d2", 0.99, rankable=False),
        ]
    )

    rankings = _rank_stage_with_validation_oracle(
        stage_name="validation_phase",
        stage_results=stage_results,
        validation_import=validation_import,
        dataset_ids=["d1", "d2"],
        scoring=SweepScoringSpec(robustness_lambda=0.0, missing_dataset_penalty=10.0),
        score_metric=TRIO_PRIMARY_SCORE,
        allow_partial_dataset_rankability=True,
    )

    assert [row["candidate_id"] for row in rankings] == ["c_good", "c_bad"]
    assert rankings[0]["score"] == pytest.approx(0.8)
    assert rankings[0]["missing_datasets"] == []
    assert rankings[0]["ranked_dataset_ids"] == ["d1"]
    assert rankings[0]["unrankable_observed_dataset_ids"] == ["d2"]


def test_validation_oracle_ranker_live_fails_when_no_completed_run_is_rankable():
    stage_results = [
        _validation_stage_result("c_good", "d1", 1.0),
        _validation_stage_result("c_good", "d2", 1.0),
    ]
    validation_import = SimpleNamespace(
        records=[
            _oracle_record("c_good", "d1", 0.8, rankable=False),
            _oracle_record("c_good", "d2", 0.9, rankable=False),
        ]
    )

    with pytest.raises(ValueError, match="no rankable validation rankings"):
        _rank_stage_with_validation_oracle(
            stage_name="validation_phase",
            stage_results=stage_results,
            validation_import=validation_import,
            dataset_ids=["d1", "d2"],
            scoring=SweepScoringSpec(robustness_lambda=0.0, missing_dataset_penalty=10.0),
            score_metric=TRIO_PRIMARY_SCORE,
            allow_partial_dataset_rankability=True,
        )


def test_rebuild_summaries_fails_closed_when_skip_mode_lacks_validation_rankings(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, validation_skip=True)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_stage1",
        dataset_id="d1",
        run_name="stage1_c_stage1_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="validation_phase",
        candidate_id="c_validation",
        dataset_id="d1",
        run_name="validation_phase_c_validation_d1",
        metric_value=3.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_stage1_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_stage1",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_stage1", "d1"),
                generation_id="gen_new",
            ),
            "validation_phase_c_validation_d1": _run_state_entry(
                stage_name="validation_phase",
                candidate_id="c_validation",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "validation_phase", "c_validation", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    with pytest.raises(Exception, match="validation rankings"):
        generate_reports.main()

    report = _read_failure_report(output_dir)
    assert "validation rankings" in report["message"]


def test_rebuild_summaries_uses_active_run_state_not_stale_run_folders(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_old",
        dataset_id="d1",
        run_name="stage1_c_old_d1",
        metric_value=99.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_old",
    )
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_old_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_old",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_old", "d1"),
                generation_id="gen_old",
                active=False,
            ),
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            ),
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )

    generate_reports.main()

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    summary_records = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in leaderboard] == ["c_active"]
    assert [record["run"] for record in summary_records] == ["stage1_c_active_d1"]


def test_rebuild_summaries_snapshots_metric_phase_generation_products(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    metric_dir = output_dir / "metric_experiments"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        metric_dir,
        stage_name="metric_phase",
        candidate_id="m1c0000",
        dataset_id="d1",
        run_name="metric_phase_m1c0000_d1",
        metric_value=9.0,
        formulation_id="exp_grad",
        alpha=0.25,
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            ),
            "metric_phase_m1c0000_d1": _run_state_entry(
                stage_name="metric_phase",
                candidate_id="m1c0000",
                dataset_id="d1",
                run_dir=_run_dir(metric_dir, "metric_phase", "m1c0000", "d1"),
                generation_id="gen_new",
            ),
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(output_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )

    generate_reports.main()

    metric_summary = json.loads((metric_dir / "metric_phase_summary.json").read_text(encoding="utf-8"))
    product_summary_path = (
        output_dir
        / "generations"
        / "gen_new"
        / "products"
        / "metric_experiments"
        / "metric_phase_summary.json"
    )
    product_summary = json.loads(product_summary_path.read_text(encoding="utf-8"))
    assert metric_summary["generation_id"] == "gen_new"
    assert {row["generation_id"] for row in metric_summary["rankings"]} == {"gen_new"}
    assert product_summary["generation_id"] == "gen_new"


def test_rebuild_summaries_resolves_relocated_run_state_paths(tmp_path, monkeypatch):
    original_dir = tmp_path / "original" / "experiment"
    relocated_dir = tmp_path / "relocated" / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        relocated_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    stale_run_dir = _run_dir(original_dir, "stage1", "c_active", "d1")
    _write_run_state(
        relocated_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=stale_run_dir,
                generation_id="gen_new",
            )
        },
    )
    state = json.loads((relocated_dir / "run_state.json").read_text(encoding="utf-8"))
    state["path_compatibility"] = {"output_dir": str(original_dir)}
    _write_json(relocated_dir / "run_state.json", state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reports.py",
            str(relocated_dir),
            "--spec",
            str(spec_path),
            "--rebuild-summaries",
        ],
    )

    generate_reports.main()

    leaderboard = json.loads((relocated_dir / "leaderboard.json").read_text(encoding="utf-8"))
    stage_summary = json.loads((relocated_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in leaderboard] == ["c_active"]
    assert stage_summary["generation_id"] == "gen_new"


def test_rebuild_summaries_uses_run_state_backup_when_primary_is_corrupt(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_old",
        dataset_id="d1",
        run_name="stage1_c_old_d1",
        metric_value=99.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_old",
    )
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    state_payload = {
        "schema_version": 2,
        "active_generation_id": "gen_new",
        "spec_generations": [{"generation_id": "gen_new", "action": "extend"}],
        "runs": {
            "stage1_c_old_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_old",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_old", "d1"),
                generation_id="gen_old",
                active=False,
            ),
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            ),
        },
    }
    _write_json(output_dir / "run_state.json.bak", state_payload)
    (output_dir / "run_state.json").write_text("{broken", encoding="utf-8")
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in leaderboard] == ["c_active"]


def test_rebuild_summaries_aborts_with_failure_report_when_run_state_unreadable(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_json(output_dir / "summary.json", [{"run": "previous"}])
    previous_summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    (output_dir / "run_state.json").write_text("{broken", encoding="utf-8")
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    with pytest.raises(Exception, match="run_state"):
        generate_reports.main()

    report = _read_failure_report(output_dir)
    report_text = json.dumps(report)
    assert report["status"] == "failed"
    assert report["error_type"] == "RebuildDataError"
    assert "run_state.json" in report_text
    assert (output_dir / "summary.json").read_text(encoding="utf-8") == previous_summary


def test_rebuild_summaries_aborts_when_state_omits_existing_raw_run(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_extra",
        dataset_id="d1",
        run_name="stage1_c_extra_d1",
        metric_value=7.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_json(output_dir / "stage1_summary.json", {"stage": "stage1", "num_runs": 1})
    previous_stage_summary = (output_dir / "stage1_summary.json").read_text(encoding="utf-8")
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    with pytest.raises(Exception, match="not represented in run_state"):
        generate_reports.main()

    report = _read_failure_report(output_dir)
    report_text = json.dumps(report)
    assert "c_extra" in report_text
    assert "filesystem_extra_runs" in report_text
    assert (output_dir / "stage1_summary.json").read_text(encoding="utf-8") == previous_stage_summary


def test_repair_run_state_preview_reports_adoptable_stage2_run_without_writing_state(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_missing",
        dataset_id="d1",
        run_name="stage2_c_missing_d1",
        metric_value=9.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    original_state = (output_dir / "run_state.json").read_text(encoding="utf-8")
    _set_repair_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    report = _read_repair_report(output_dir)
    assert report["mode"] == "preview"
    assert report["status"] == "preview"
    assert report["counts"]["adoptable"] == 1
    assert report["counts"]["rejected"] == 0
    adopted = report["adoptable_entries"][0]
    assert adopted["run_uid"] == "stage2_c_missing_d1"
    assert adopted["generation_id"] == "gen_new"
    assert adopted["generation_inferred"] is True
    assert (output_dir / "run_state.json").read_text(encoding="utf-8") == original_state
    assert not (output_dir / "stage2_summary.json").exists()


def test_repair_run_state_apply_adopts_stage2_run_and_rebuilds(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage2_c_active_d1",
        metric_value=7.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_missing",
        dataset_id="d1",
        run_name="stage2_c_missing_d1",
        metric_value=9.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            ),
            "stage2_c_active_d1": _run_state_entry(
                stage_name="stage2",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage2", "c_active", "d1"),
                generation_id="gen_new",
            ),
        },
    )
    _set_repair_argv(monkeypatch, output_dir, spec_path, "apply")

    generate_reports.main()

    report = _read_repair_report(output_dir)
    assert report["mode"] == "apply"
    assert report["status"] == "applied"
    state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert "stage1_c_active_d1" in state["runs"]
    assert "stage2_c_active_d1" in state["runs"]
    adopted_entry = state["runs"]["stage2_c_missing_d1"]
    assert adopted_entry["status"] == "ok"
    assert adopted_entry["active"] is True
    assert adopted_entry["generation_id"] == "gen_new"
    assert adopted_entry["artifact_paths"]["metric_record"].endswith("metric_record.json")
    assert (output_dir / "run_state.json.bak").exists()
    stage2_summary = json.loads((output_dir / "stage2_summary.json").read_text(encoding="utf-8"))
    assert {row["run"] for row in stage2_summary["runs"]} == {
        "stage2_c_active_d1",
        "stage2_c_missing_d1",
    }


def test_repair_run_state_preview_rejects_unknown_dataset(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_bad",
        dataset_id="d_missing",
        run_name="stage2_c_bad_d_missing",
        metric_value=9.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _set_repair_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    report = _read_repair_report(output_dir)
    assert report["counts"]["adoptable"] == 0
    assert report["counts"]["rejected"] == 1
    assert report["rejected_entries"][0]["reason"] == "unknown_dataset"
    state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert "stage2_c_bad_d_missing" not in state["runs"]


def test_repair_run_state_preview_rejects_generation_mismatch(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_old",
        dataset_id="d1",
        run_name="stage2_c_old_d1",
        metric_value=9.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_old",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _set_repair_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    report = _read_repair_report(output_dir)
    assert report["counts"]["adoptable"] == 0
    assert report["rejected_entries"][0]["reason"] == "generation_mismatch"
    assert report["rejected_entries"][0]["generation_id"] == "gen_old"


def test_repair_run_state_preview_rejects_corrupt_or_missing_metric_record(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run(
        output_dir,
        stage_name="stage2",
        candidate_id="c_corrupt",
        dataset_id="d1",
        run_name="stage2_c_corrupt_d1",
        metric_value=9.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    (_run_dir(output_dir, "stage2", "c_corrupt", "d1") / "metric_record.json").write_text(
        "{broken",
        encoding="utf-8",
    )
    missing_dir = _run_dir(output_dir, "stage2", "c_missing", "d1")
    missing_dir.mkdir(parents=True)
    _write_json(
        missing_dir / "config.json",
        {
            "dataset_id": "d1",
            "subpixel_mode": "pixel",
            "displacement_metric_family": "ncc_gradient",
            "displacement_alpha": 1.0,
            "metric_formulation_id": "baseline",
        },
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _set_repair_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    report = _read_repair_report(output_dir)
    assert report["counts"]["adoptable"] == 0
    assert {entry["reason"] for entry in report["rejected_entries"]} == {
        "metric_record_unreadable",
        "metric_record_missing",
    }


def test_rebuild_summaries_falls_back_to_metric_record_config(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    run_dir = _run_dir(output_dir, "stage1", "c_active", "d1")
    (run_dir / "config.json").unlink()
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=run_dir,
                generation_id="gen_new",
            )
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    stage_summary = json.loads((output_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert stage_summary["runs"][0]["config"]["dataset_id"] == "d1"
    assert any("config.json" in warning for warning in stage_summary["execution"]["warnings"])


def test_rebuild_summaries_aborts_with_report_for_unknown_dataset(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d_missing",
        run_name="stage1_c_active_d_missing",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    with pytest.raises(Exception, match="Unknown dataset"):
        generate_reports.main()

    report = _read_failure_report(output_dir)
    report_text = json.dumps(report)
    assert "d_missing" in report_text
    assert "stage1_c_active_d_missing" in report_text


def test_rebuild_summaries_snapshots_reports_after_generation(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment"
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path)
    _write_run(
        output_dir,
        stage_name="stage1",
        candidate_id="c_active",
        dataset_id="d1",
        run_name="stage1_c_active_d1",
        metric_value=5.0,
        formulation_id="baseline",
        alpha=1.0,
        generation_id="gen_new",
    )
    _write_run_state(
        output_dir,
        active_generation_id="gen_new",
        runs={
            "stage1_c_active_d1": _run_state_entry(
                stage_name="stage1",
                candidate_id="c_active",
                dataset_id="d1",
                run_dir=_run_dir(output_dir, "stage1", "c_active", "d1"),
                generation_id="gen_new",
            )
        },
    )
    _set_rebuild_argv(monkeypatch, output_dir, spec_path)

    generate_reports.main()

    products_dir = output_dir / "generations" / "gen_new" / "products"
    assert (products_dir / "analysis.json").exists()
    assert (products_dir / "alpha_report.json").exists()

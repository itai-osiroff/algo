import os
import json
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import generate_reports


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

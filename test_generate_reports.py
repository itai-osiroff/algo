import os
import json
import sys
from pathlib import Path

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
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metric_record.json", metric_record)


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

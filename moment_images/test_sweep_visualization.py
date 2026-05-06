import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from moment_images.sweep_visualization import (
    SweepVisualizationApp,
    create_candidate_robustness_figure,
    create_comparison_figure,
    create_effects_figure,
    create_progression_figure,
    export_sweep_visualization_bundle,
    load_sweep_visualization_model,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), 127, dtype=np.uint8)).save(path)


def _metrics(base: float) -> dict:
    return {
        "average_intensity": base,
        "maximum_intensity": base + 1.0,
        "total_variation": base + 2.0,
        "sum_squared_gradient": base + 3.0,
        "laplacian_variance": base + 4.0,
        "dwt_level2_hv_energy": base + 5.0,
    }


def _displacement_metrics(base: float) -> dict:
    return {
        "mean_magnitude": base,
        "std_magnitude": base + 0.5,
        "p95_magnitude": base + 1.0,
        "max_magnitude": base + 1.5,
    }


def _consistency_metrics(base: float) -> dict:
    return {
        "mean_view_std": base,
        "p95_view_std": base + 0.2,
        "max_view_std": base + 0.4,
    }


def _diagnostics_payload() -> dict:
    return {
        "metric_family": "ncc_gradient",
        "alpha": 0.2,
        "component_selected_stats": {
            "gradient": {"mean": 0.8, "std": 0.1, "min": 0.5, "max": 0.9},
            "laplacian": {"mean": 0.6, "std": 0.2, "min": 0.2, "max": 0.8},
        },
    }


def _write_workspace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pixel_blended_scores=np.arange(16, dtype=np.float64).reshape(4, 4),
        pixel_component_score_gradient=np.full((4, 4), 0.75, dtype=np.float64),
        pixel_component_score_laplacian=np.full((4, 4), 0.55, dtype=np.float64),
        pixel_displacements_x=np.full((4, 4), 0.4, dtype=np.float64),
        pixel_displacements_y=np.full((4, 4), 0.3, dtype=np.float64),
    )


def _write_alpha_report(path: Path) -> None:
    _write_json(
        path,
        {
            "primary_metric": "laplacian_variance",
            "ranked_runs": [
                {
                    "rank": 1,
                    "run": "metric_m1c0000_d1",
                    "family": "ncc_gradient",
                    "alpha": 0.2,
                    "ranking_metric_name": "laplacian_variance",
                    "ranking_metric_value": 42.0,
                }
            ],
            "alpha_family_summary": [
                {
                    "family": "ncc_gradient",
                    "alpha": 0.2,
                    "count": 2,
                    "laplacian_variance_mean": 42.0,
                    "mean_runtime_seconds": 9.5,
                    "mean_displacement_magnitude_mean": 1.3,
                    "mean_view_std_mean": 0.25,
                },
                {
                    "family": "ncc_laplacian",
                    "alpha": 0.5,
                    "count": 1,
                    "laplacian_variance_mean": 31.0,
                    "mean_runtime_seconds": 11.5,
                    "mean_displacement_magnitude_mean": 1.7,
                    "mean_view_std_mean": 0.35,
                },
            ],
        },
    )


def _write_metric_record(
    path: Path,
    *,
    run: str,
    family: str,
    alpha: float,
    formulation_id: str,
    metric_base: float,
) -> None:
    _write_json(
        path,
        {
            "run": run,
            "config": {
                "strategy": "combined",
                "displacement_metric_family": family,
                "displacement_alpha": alpha,
                "metric_formulation_id": formulation_id,
                "max_displacement": 6,
                "gauss_sigma": 3.0,
            },
            "metric_formulation": {
                "family": family,
                "alpha": alpha,
                "id": formulation_id,
            },
            "global_and_local_metrics": _metrics(metric_base),
            "displacement_field_metrics": _displacement_metrics(metric_base / 10.0),
            "inter_view_consistency_metrics": _consistency_metrics(metric_base / 100.0),
            "displacement_score_diagnostics": _diagnostics_payload(),
        },
    )


def _ranking(
    candidate_id: str,
    *,
    score: float,
    rank: int,
    datasets: list[str],
    stage: str,
    family: str = "intensity",
    alpha: float = 1.0,
    strategy: str = "combined",
    params: dict | None = None,
) -> dict:
    formulation_id = "intensity_default" if family == "intensity" else f"{family}_{alpha:g}"
    param_payload = {
        "max_displacement": 5 + rank,
        "gauss_sigma": 3.0,
        "pixel_threshold": 0.05,
        "displacement_metric_family": family,
        "displacement_alpha": alpha,
        "metric_formulation_id": formulation_id,
    }
    if params:
        param_payload.update(params)
    config_payload = {
        "strategy": strategy,
        "similarity_window_size": 21,
        "refinement_window_size": 21,
        "max_displacement": param_payload["max_displacement"],
        "gauss_sigma": param_payload["gauss_sigma"],
        "pixel_threshold": param_payload["pixel_threshold"],
        "displacement_metric_family": family,
        "displacement_alpha": alpha,
        "metric_formulation_id": formulation_id,
    }
    dataset_scores = {dataset_id: score - 0.02 * idx for idx, dataset_id in enumerate(datasets)}
    dataset_metrics = {
        dataset_id: _metrics(10.0 * rank + idx)
        for idx, dataset_id in enumerate(datasets)
    }
    runtimes = {dataset_id: 12.0 + rank + idx for idx, dataset_id in enumerate(datasets)}
    return {
        "candidate_id": candidate_id,
        "strategy": strategy,
        "config": config_payload,
        "params": param_payload,
        "score": score,
        "mean_dataset_score": float(np.mean(list(dataset_scores.values()))),
        "std_dataset_score": float(np.std(list(dataset_scores.values()))),
        "missing_datasets": [],
        "dataset_scores": dataset_scores,
        "runtime_seconds": runtimes,
        "dataset_metrics": dataset_metrics,
        "rank": rank,
    }


def _stage_summary(stage: str, rankings: list[dict], failed_runs: list[dict] | None = None) -> dict:
    failures = failed_runs or []
    return {
        "stage": stage,
        "num_runs": len(rankings) + len(failures),
        "num_successful": len(rankings),
        "num_failed": len(failures),
        "rankings": rankings,
        "failed_runs": failures,
        "runs": [],
        "execution": {"selected_workers": 1, "requested_workers": 1},
    }


def _run_state_payload(
    output_dir: Path,
    datasets: list[str],
    *,
    include_stage2_runs: bool = True,
    include_metric_phase: bool = False,
    stage_mode: str = "two_stage",
) -> dict:
    runs = {
        "stage1_s1c0000_d1": {
            "run_uid": "stage1_s1c0000_d1",
            "stage": "stage1",
            "candidate_id": "s1c0000",
            "dataset_id": datasets[0],
            "status": "ok",
        }
    }
    if stage_mode == "two_stage" and include_stage2_runs:
        runs["stage2_s2c0000_d1"] = {
            "run_uid": "stage2_s2c0000_d1",
            "stage": "stage2",
            "candidate_id": "s2c0000",
            "dataset_id": datasets[0],
            "status": "pending",
        }
        runs["stage2_s2c0000_d2"] = {
            "run_uid": "stage2_s2c0000_d2",
            "stage": "stage2",
            "candidate_id": "s2c0000",
            "dataset_id": datasets[-1],
            "status": "running",
        }
    if include_metric_phase:
        runs["metric_phase_m1c0000_d1"] = {
            "run_uid": "metric_phase_m1c0000_d1",
            "stage": "metric_phase",
            "candidate_id": "m1c0000",
            "dataset_id": datasets[0],
            "status": "ok",
        }

    stages = {
        "stage1": {"planned_runs": 4, "completed_runs": 4},
    }
    if stage_mode == "two_stage":
        stages["stage2"] = {"planned_runs": 4, "completed_runs": 0}
    if include_metric_phase:
        stages["metric_phase"] = {"planned_runs": 2, "completed_runs": 2}

    return {
        "schema_version": 1,
        "created_at": "2026-04-01T00:00:00+00:00",
        "updated_at": "2026-04-01T01:00:00+00:00",
        "experiment_fingerprint": "abc123",
        "path_compatibility": {"output_dir": str(output_dir)},
        "fingerprint_payload": {
            "spec": {
                "strategies": ["combined"],
                "stages": {"top_k": 1, "mode": stage_mode},
                "scoring": {"metric_weights": {"laplacian_variance": 1.0}},
            },
            "datasets": [{"dataset_id": dataset_id} for dataset_id in datasets],
        },
        "stages": stages,
        "runs": runs,
    }


def _write_manifest(path: Path, datasets: list[str], *, stage_mode: str = "two_stage") -> None:
    _write_json(
        path,
        {
            "generated_at": "2026-04-01T01:00:00+00:00",
            "strategies": ["combined"],
            "datasets": [{"dataset_id": dataset_id} for dataset_id in datasets],
            "stages": {"top_k": 1, "mode": stage_mode},
            "scoring": {"metric_weights": {"laplacian_variance": 1.0}},
            "reporting": {"primary_metric": "laplacian_variance"},
        },
    )


def _create_modern_fixture(
    base_dir: Path,
    *,
    include_stage2_summary: bool = True,
    include_metric_phase: bool = False,
    include_alpha_report: bool = False,
    include_workspace: bool = False,
    stage_mode: str = "two_stage",
) -> Path:
    output_dir = base_dir / "modern"
    datasets = ["d1", "d2"]
    _write_manifest(output_dir / "experiment_manifest.json", datasets, stage_mode=stage_mode)
    _write_json(
        output_dir / "run_state.json",
        _run_state_payload(
            output_dir,
            datasets,
            include_stage2_runs=(stage_mode == "two_stage" and not include_stage2_summary),
            include_metric_phase=include_metric_phase,
            stage_mode=stage_mode,
        ),
    )
    stage1_rankings = [
        _ranking("s1c0000", score=0.88, rank=1, datasets=datasets, stage="stage1"),
        _ranking("s1c0001", score=0.63, rank=2, datasets=datasets, stage="stage1", params={"max_displacement": 9}),
    ]
    _write_json(output_dir / "stage1_summary.json", _stage_summary("stage1", stage1_rankings))
    for dataset_id in datasets:
        _write_preview(output_dir / "runs" / "stage1" / "s1c0000" / dataset_id / "avg_rectified.png")
    if stage_mode == "two_stage" and include_stage2_summary:
        stage2_rankings = [
            _ranking("s2c0000", score=0.91, rank=1, datasets=datasets, stage="stage2", params={"max_displacement": 7}),
        ]
        _write_json(output_dir / "stage2_summary.json", _stage_summary("stage2", stage2_rankings))
        for dataset_id in datasets:
            _write_preview(output_dir / "runs" / "stage2" / "s2c0000" / dataset_id / "avg_rectified.png")
    summary_rows = [
        {
            "run": "stage1_s1c0000_d1",
            "config": {
                "strategy": "combined",
                "displacement_metric_family": "intensity",
                "displacement_alpha": 1.0,
                "metric_formulation_id": "intensity_default",
                "max_displacement": 6,
                "gauss_sigma": 3.0,
            },
            "metric_formulation": {"family": "intensity", "alpha": 1.0, "id": "intensity_default"},
            "global_and_local_metrics": _metrics(10.0),
            "displacement_field_metrics": _displacement_metrics(1.0),
            "inter_view_consistency_metrics": _consistency_metrics(0.2),
            "displacement_score_diagnostics": _diagnostics_payload(),
        }
    ]
    _write_json(output_dir / "summary.json", summary_rows)
    if include_workspace:
        _write_workspace(output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "workspace.npz")

    if include_metric_phase:
        metric_dir = output_dir / "metric_experiments"
        metric_rankings = [
            _ranking("m1c0000", score=0.95, rank=1, datasets=datasets, stage="metric_phase", family="ncc_gradient", alpha=0.2),
            _ranking("m1c0001", score=0.82, rank=2, datasets=datasets, stage="metric_phase", family="ncc_gradient", alpha=0.8),
            _ranking("m1c0002", score=0.76, rank=3, datasets=datasets, stage="metric_phase", family="ncc_laplacian", alpha=0.5),
        ]
        _write_json(metric_dir / "metric_phase_summary.json", _stage_summary("metric_phase", metric_rankings))
        _write_json(
            metric_dir / "summary.json",
            [
                {
                    "run": "metric_m1c0000_d1",
                    "config": {
                        "strategy": "combined",
                        "displacement_metric_family": "ncc_gradient",
                        "displacement_alpha": 0.2,
                        "max_displacement": 6,
                        "gauss_sigma": 3.0,
                    },
                    "metric_formulation": {"family": "ncc_gradient", "alpha": 0.2, "id": "grad_0p2"},
                    "global_and_local_metrics": _metrics(20.0),
                    "displacement_field_metrics": _displacement_metrics(1.4),
                    "inter_view_consistency_metrics": _consistency_metrics(0.3),
                    "displacement_score_diagnostics": _diagnostics_payload(),
                }
            ],
        )
        for dataset_id in datasets:
            _write_preview(metric_dir / "runs" / "metric_phase" / "m1c0000" / dataset_id / "avg_rectified.png")
        if include_workspace:
            _write_workspace(metric_dir / "runs" / "metric_phase" / "m1c0000" / "d1" / "workspace.npz")
        if include_alpha_report:
            _write_alpha_report(metric_dir / "alpha_report.json")
    return output_dir


def _create_legacy_fixture(base_dir: Path, *, include_metric_axes: bool = False) -> Path:
    output_dir = base_dir / "legacy"
    config = {"strategy": "pixel", "similarity_window_size": 15, "max_displacement": 5}
    if include_metric_axes:
        config.update(
            {
                "displacement_metric_family": "ncc_gradient",
                "displacement_alpha": 0.5,
                "metric_formulation_id": "baseline",
            }
        )
    summary = [
        {
            "run": "win15_disp5",
            "config": config,
            "metric_formulation": (
                {"family": "ncc_gradient", "alpha": 0.5, "id": "baseline"} if include_metric_axes else {}
            ),
            "global_and_local_metrics": _metrics(5.0),
        },
        {
            "run": "win25_disp9",
            "config": {**config, "similarity_window_size": 25, "max_displacement": 9},
            "metric_formulation": (
                {"family": "ncc_gradient", "alpha": 0.7, "id": "variant"} if include_metric_axes else {}
            ),
            "global_and_local_metrics": _metrics(9.0),
        },
    ]
    _write_json(output_dir / "summary.json", summary)
    for run_name in ("win15_disp5", "win25_disp9"):
        _write_preview(output_dir / run_name / "avg_rectified.png")
    return output_dir


def test_load_sweep_visualization_model_supports_legacy_outputs(tmp_path):
    output_dir = _create_legacy_fixture(tmp_path)

    model = load_sweep_visualization_model(output_dir)

    assert model.mode == "legacy"
    assert model.primary_stage_name == "legacy"
    assert model.datasets == ["aggregate"]
    assert model.stage_order == ["legacy"]
    assert model.score_matrix.shape == (2, 1)
    assert model.candidates[0].candidate_id == "win25_disp9"


def test_load_sweep_visualization_model_supports_full_modern_outputs(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True)

    model = load_sweep_visualization_model(output_dir)

    assert model.mode == "spec_full"
    assert model.stage_order == ["stage1", "stage2"]
    assert model.primary_stage_name == "stage2"
    assert model.datasets == ["d1", "d2"]
    assert model.stages["stage2"].num_successful == 1
    assert model.score_matrix.shape[1] == 2
    assert model.candidates[0].preview_paths
    assert not model.alpha_family_summary


def test_load_sweep_visualization_model_supports_partial_modern_outputs(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=False)

    model = load_sweep_visualization_model(output_dir)

    assert model.mode == "spec_partial"
    assert "stage2" in model.stages
    assert model.stages["stage2"].num_runs_planned == 4
    assert model.stages["stage2"].num_pending == 1
    assert model.stages["stage2"].num_running == 1
    assert not model.alpha_family_summary


def test_load_sweep_visualization_model_supports_single_stage_modern_outputs(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )

    model = load_sweep_visualization_model(output_dir)
    progression = create_progression_figure(model)
    export_dir = tmp_path / "single_stage_export"
    export_sweep_visualization_bundle(model, export_dir)

    assert model.mode == "spec_single_stage"
    assert model.stage_order == ["stage1"]
    assert model.primary_stage_name == "stage1"
    assert "Single-stage baseline run" in progression.axes[0].texts[0].get_text()
    assert (export_dir / "progression.png").exists()


def test_load_sweep_visualization_model_supports_metric_phase_outputs(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_alpha_report=True,
        include_workspace=True,
    )

    model = load_sweep_visualization_model(output_dir)

    assert model.primary_stage_name == "metric_phase"
    assert "metric_phase" in model.stage_order
    assert model.alpha_family_summary
    assert model.alpha_ranked_runs
    families = {row["family"] for row in model.alpha_family_summary}
    assert {"blurred_gradient_corr", "blurred_laplacian_corr"}.issubset(families)
    assert model.diagnostic_availability["score_maps"] is True
    assert "gradient" in model.diagnostic_availability["components"]


def test_load_sweep_visualization_model_supports_direct_metric_phase_directory(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_alpha_report=True,
    )

    model = load_sweep_visualization_model(output_dir / "metric_experiments")

    assert model.mode == "metric_phase_dir"
    assert model.path_scope == "metric_phase_dir"
    assert model.primary_stage_name == "metric_phase"
    assert model.stage_order == ["stage1", "stage2", "metric_phase"]


def test_load_sweep_visualization_model_prefers_reporting_primary_metric(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True)
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scoring"]["metric_weights"] = {"sum_squared_gradient": 1.0, "laplacian_variance": 0.5}
    manifest["reporting"]["primary_metric"] = "laplacian_variance"
    _write_json(manifest_path, manifest)

    model = load_sweep_visualization_model(output_dir)

    assert model.primary_metric == "laplacian_variance"
    assert model.primary_metric_source == "experiment_manifest.reporting.primary_metric"


def test_load_sweep_visualization_model_warns_but_loads_when_optional_artifact_is_corrupt(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True, include_metric_phase=True)
    (output_dir / "metric_experiments" / "summary.json").write_text("{broken", encoding="utf-8")

    model = load_sweep_visualization_model(output_dir)

    assert model.mode == "spec_full"
    assert model.warnings
    assert any("summary.json" in warning for warning in model.warnings)


def test_load_sweep_visualization_model_synthesizes_metric_phase_alpha_summary_from_raw_records(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True, include_metric_phase=True)
    metric_dir = output_dir / "metric_experiments"
    (metric_dir / "metric_phase_summary.json").unlink()
    (metric_dir / "summary.json").unlink()
    _write_alpha_report(output_dir / "alpha_report.json")
    _write_metric_record(
        metric_dir / "runs" / "metric_phase" / "m1c0000" / "d1" / "metric_record.json",
        run="metric_phase_m1c0000_d1",
        family="ncc_gradient",
        alpha=0.2,
        formulation_id="grad_0p2",
        metric_base=20.0,
    )
    _write_metric_record(
        metric_dir / "runs" / "metric_phase" / "m1c0001" / "d1" / "metric_record.json",
        run="metric_phase_m1c0001_d1",
        family="ncc_gradient",
        alpha=0.8,
        formulation_id="grad_0p8",
        metric_base=24.0,
    )

    model = load_sweep_visualization_model(output_dir)

    assert model.provenance["alpha_summary_source"] == "metric_phase.run_metric_records"
    assert {float(row["alpha"]) for row in model.alpha_family_summary} == {0.2, 0.8}
    assert all(float(row["alpha"]) != 0.5 for row in model.alpha_family_summary)
    assert any(
        "Metric-phase alpha report unavailable; synthesized alpha summary" in warning
        for warning in model.warnings
    )


def test_load_sweep_visualization_model_falls_back_to_root_alpha_report_when_metric_phase_has_no_alpha_data(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True, include_metric_phase=True)
    metric_dir = output_dir / "metric_experiments"
    (metric_dir / "metric_phase_summary.json").unlink()
    (metric_dir / "summary.json").unlink()
    _write_alpha_report(output_dir / "alpha_report.json")

    model = load_sweep_visualization_model(output_dir)

    assert model.provenance["alpha_summary_source"] == "experiment_root.alpha_report"
    assert any(float(row["alpha"]) == 0.5 for row in model.alpha_family_summary)


def test_load_sweep_visualization_model_scopes_metric_phase_candidate_alpha_fallback(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True, include_metric_phase=True)
    metric_dir = output_dir / "metric_experiments"
    _write_json(
        metric_dir / "metric_phase_summary.json",
        _stage_summary(
            "metric_phase",
            [
                _ranking("m1c0000", score=0.95, rank=1, datasets=["d1", "d2"], stage="metric_phase", family="ncc_gradient", alpha=0.2),
                _ranking("m1c0001", score=0.82, rank=2, datasets=["d1", "d2"], stage="metric_phase", family="ncc_gradient", alpha=0.8),
            ],
        ),
    )
    (metric_dir / "summary.json").unlink()
    _write_alpha_report(output_dir / "alpha_report.json")

    model = load_sweep_visualization_model(output_dir)

    assert model.provenance["alpha_summary_source"] == "metric_phase.candidate_rows"
    assert {float(row["alpha"]) for row in model.alpha_family_summary} == {0.2, 0.8}
    assert all(float(row["alpha"]) != 0.5 for row in model.alpha_family_summary)
    assert any(
        "Metric-phase alpha report unavailable; synthesized alpha summary from metric-phase candidate rows."
        in warning
        for warning in model.warnings
    )


def test_export_sweep_visualization_bundle_writes_bundle_and_plots(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_alpha_report=True,
        include_workspace=True,
    )
    model = load_sweep_visualization_model(output_dir)
    export_dir = tmp_path / "export"

    exported = export_sweep_visualization_bundle(model, export_dir)

    assert exported == export_dir
    assert (export_dir / "visualization_bundle.npz").exists()
    assert (export_dir / "visualization_bundle.mat").exists()
    assert (export_dir / "visualization_manifest.json").exists()
    assert (export_dir / "overview.png").exists()
    assert (export_dir / "coverage.png").exists()
    assert (export_dir / "effects_trend.png").exists()
    assert (export_dir / "effects_heatmap.png").exists()
    assert (export_dir / "progression.png").exists()
    assert (export_dir / "significance.png").exists()
    assert (export_dir / "metric_diagnostics.png").exists()
    assert (export_dir / "index.html").exists()
    assert (export_dir / "tables" / "candidate_leaderboard.csv").exists()
    assert (export_dir / "tables" / "run_metric_records.csv").exists()
    assert (export_dir / "data" / "report_context.json").exists()
    assert (export_dir / "assets" / "topk_overview.png").exists()
    assert (export_dir / "assets" / "preview_d1.png").exists()
    assert (export_dir / "assets" / "comparison_d1.png").exists()
    assert (export_dir / "assets" / "diagnostic_d1_gradient.png").exists()
    assert (export_dir / "assets" / "robustness_heatmap.png").exists()
    with np.load(export_dir / "visualization_bundle.npz", allow_pickle=True) as bundle:
        assert "score_matrix" in bundle.files
        assert "metric_tensor" in bundle.files
        assert "run_metric_records_json" in bundle.files
        assert "diagnostic_availability_json" in bundle.files
        assert "primary_metric_source" in bundle.files
        assert bundle["score_matrix"].shape[1] == 2
        assert bundle["metric_tensor"].ndim == 3
    manifest = json.loads((export_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_html"] == "index.html"
    assert manifest["report_top_k"] == 6
    assert "tables/candidate_leaderboard.csv" in manifest["table_files"]
    assert "assets/topk_overview.png" in manifest["report_assets"]


def test_create_effects_figure_supports_aggregate_and_dataset_specific_grouping(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=True,
    )
    stage1_summary_path = output_dir / "stage1_summary.json"
    stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    rankings = stage1_summary["rankings"]
    rankings[1]["params"]["max_displacement"] = rankings[0]["params"]["max_displacement"]
    rankings[1]["config"]["max_displacement"] = rankings[0]["config"]["max_displacement"]
    _write_json(stage1_summary_path, stage1_summary)

    root_summary_path = output_dir / "summary.json"
    root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
    duplicate = dict(root_summary[0])
    duplicate["run"] = "stage1_s1c0001_d1"
    duplicate["config"] = dict(duplicate["config"])
    duplicate["config"]["gauss_sigma"] = root_summary[0]["config"]["gauss_sigma"]
    duplicate["config"]["max_displacement"] = root_summary[0]["config"].get("max_displacement", 5)
    root_summary.append(duplicate)
    _write_json(root_summary_path, root_summary)

    model = load_sweep_visualization_model(output_dir)
    trend = create_effects_figure(
        model,
        mode="1D Trend",
        stage_name="stage1",
        dataset_scope="aggregate",
        x_parameter_name="max_displacement",
        value_option="score",
        parameter_filters={},
    )
    heatmap = create_effects_figure(
        model,
        mode="2D Heatmap",
        stage_name="stage1",
        dataset_scope="d1",
        x_parameter_name="max_displacement",
        y_parameter_name="gauss_sigma",
        value_option="metric:laplacian_variance",
        parameter_filters={},
    )

    assert trend.axes[0].get_xlabel() == "max_displacement"
    assert any(text.get_text() == "n=2" for text in trend.axes[0].texts)
    assert "heatmap" in heatmap.axes[0].get_title().lower()
    assert any(text.get_text() == "2" for text in heatmap.axes[0].texts)


def test_create_comparison_and_robustness_figures_render(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=True,
    )
    model = load_sweep_visualization_model(output_dir)
    rows = [row for row in model.candidates if row.stage == model.primary_stage_name][:2]

    comparison = create_comparison_figure(
        model,
        rows,
        dataset_id="d1",
        component_name="gradient",
    )
    robustness = create_candidate_robustness_figure(model, rows)

    assert len(comparison.axes) >= 5
    assert "comparison" in comparison._suptitle.get_text().lower()
    assert robustness.axes[0].get_xlabel() == "dataset"
    assert robustness.axes[0].get_ylabel() == "candidate"


def test_export_bundle_without_workspace_still_writes_report(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=False,
    )
    model = load_sweep_visualization_model(output_dir)
    export_dir = tmp_path / "no_workspace_export"

    export_sweep_visualization_bundle(model, export_dir)

    assert (export_dir / "index.html").exists()
    html_text = (export_dir / "index.html").read_text(encoding="utf-8")
    assert "Sweep Visualization Report" in html_text
    assert "Dataset: d1" in html_text


def test_export_report_handles_partial_and_legacy_models(tmp_path):
    partial_dir = _create_modern_fixture(tmp_path, include_stage2_summary=False)
    legacy_dir = _create_legacy_fixture(tmp_path)
    partial_model = load_sweep_visualization_model(partial_dir)
    legacy_model = load_sweep_visualization_model(legacy_dir)

    partial_export = tmp_path / "partial_export"
    legacy_export = tmp_path / "legacy_export"
    export_sweep_visualization_bundle(partial_model, partial_export)
    export_sweep_visualization_bundle(legacy_model, legacy_export)

    partial_html = (partial_export / "index.html").read_text(encoding="utf-8")
    legacy_html = (legacy_export / "index.html").read_text(encoding="utf-8")
    assert "stage2: planned=4" in partial_html
    assert "Dataset: aggregate" in legacy_html


def _can_create_tk() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def _make_root_or_skip():
    if not _can_create_tk():
        pytest.skip("Tkinter UI unavailable in this environment")
    import tkinter as tk
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception:
        pytest.skip("Tkinter root unavailable in this environment")


def test_gui_loads_model_and_toggles_metric_tab(tmp_path):
    metric_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_alpha_report=True,
        include_workspace=True,
    )
    legacy_dir = _create_legacy_fixture(tmp_path, include_metric_axes=False)

    root = _make_root_or_skip()
    app = SweepVisualizationApp(root)
    app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))

    app.load_from_path(metric_dir)
    root.update()
    tab_names = [app.notebook.tab(tab_id, "text") for tab_id in app.notebook.tabs()]
    assert "Leaderboard" in tab_names
    assert "Comparison" in tab_names
    assert "Effects" in tab_names
    assert "Metric Diagnostics" in tab_names
    assert "Metric Formulations" in tab_names

    app.load_from_path(legacy_dir)
    root.update()
    tab_names = [app.notebook.tab(tab_id, "text") for tab_id in app.notebook.tabs()]
    assert "Metric Formulations" not in tab_names
    root.destroy()


def test_gui_leaderboard_selection_drives_comparison_and_diagnostics(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=True,
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
    root.update()

    children = app.leaderboard_tree.get_children()
    assert children
    app.leaderboard_tree.selection_set(children[0])
    app._on_leaderboard_selection_changed()
    root.update()

    detail_text = app.leaderboard_detail_text.get("1.0", "end").strip()
    comparison_text = app.comparison_text.get("1.0", "end").strip()
    diagnostics_text = app.diagnostics_text.get("1.0", "end").strip()

    assert "candidate:" in detail_text
    assert "dataset:" in comparison_text
    assert "run:" in diagnostics_text
    root.destroy()


def test_gui_export_writes_bundle(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True, include_metric_phase=True)
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
    export_dir = tmp_path / "gui_export"

    exported = app.export_to(export_dir)

    deadline = time.time() + 1.0
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)
    assert exported == export_dir
    assert (export_dir / "visualization_bundle.npz").exists()
    assert (export_dir / "visualization_manifest.json").exists()
    root.destroy()


def test_main_supports_headless_export_dir(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=True,
    )
    export_dir = tmp_path / "headless_export"

    main([str(output_dir), "--export-dir", str(export_dir)])

    assert (export_dir / "index.html").exists()
    assert (export_dir / "visualization_bundle.npz").exists()
    assert (export_dir / "tables" / "candidate_leaderboard.csv").exists()

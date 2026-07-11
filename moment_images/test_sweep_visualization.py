import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from moment_images import report_generation
from moment_images.sweep_visualization import (
    RectificationViewSelection,
    SweepVisualizationApp,
    _comparison_artifacts_for_candidate,
    _effect_row_identity_dict,
    _effect_search_rows,
    _format_comparison_text,
    _format_effect_search_detail,
    candidate_rows_for_generation_scope,
    create_candidate_robustness_figure,
    create_comparison_figure,
    create_effects_figure,
    create_parameter_sensitivity_figure,
    create_progression_figure,
    export_sweep_visualization_bundle,
    compute_parameter_sensitivity,
    generate_next_extension_suggestions,
    generation_scope_options,
    load_rectification_display_bundle,
    load_sweep_visualization_model,
    run_records_for_generation_scope,
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
        average_rectified_global_image=np.full((4, 4), 3.0, dtype=np.float64),
        average_rectified_image=np.full((4, 4), 1.25, dtype=np.float64),
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
    include_validation_phase: bool = False,
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
    if include_validation_phase:
        validation_rankings = [
            _ranking(
                "v1c0000",
                score=0.97,
                rank=1,
                datasets=datasets,
                stage="validation_phase",
                params={"max_displacement": 4},
            )
        ]
        _write_json(
            output_dir / "validation_phase_summary.json",
            _stage_summary("validation_phase", validation_rankings),
        )
        _write_json(
            output_dir / "validation_summary.json",
            [
                {
                    "run": "validation_phase_v1c0000_d1",
                    "config": {
                        "strategy": "combined",
                        "max_displacement": 4,
                        "gauss_sigma": 3.0,
                    },
                    "global_and_local_metrics": _metrics(18.0),
                    "displacement_field_metrics": _displacement_metrics(1.1),
                    "inter_view_consistency_metrics": _consistency_metrics(0.25),
                    "displacement_score_diagnostics": _diagnostics_payload(),
                }
            ],
        )
        for dataset_id in datasets:
            _write_preview(
                output_dir
                / "runs"
                / "validation_phase"
                / "v1c0000"
                / dataset_id
                / "avg_rectified.png"
            )
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


def test_rectification_display_bundle_loads_preview_only_result(tmp_path):
    output_dir = _create_legacy_fixture(tmp_path)
    model = load_sweep_visualization_model(output_dir)
    row = model.candidates[0]

    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage=row.stage,
            candidate_id=row.candidate_id,
            dataset_id="aggregate",
            component_name="gradient",
        ),
    )

    assert bundle.preview is not None
    assert bundle.preview.shape == (16, 16)
    assert bundle.displacement_magnitude is None
    assert bundle.blended_score is None
    assert any("workspace.npz unavailable" in warning for warning in bundle.warnings)


def test_rectification_display_bundle_loads_workspace_channels(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_workspace=True,
    )
    model = load_sweep_visualization_model(output_dir)

    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage="stage1",
            candidate_id="s1c0000",
            dataset_id="d1",
            component_name="gradient",
        ),
    )

    assert bundle.preview is not None
    assert bundle.blended_score is not None
    assert bundle.component_score is not None
    assert bundle.displacement_magnitude is not None
    np.testing.assert_allclose(bundle.displacement_magnitude, np.full((4, 4), 0.5))
    assert bundle.global_final_delta is not None
    np.testing.assert_allclose(bundle.global_final_delta, np.full((4, 4), 1.75))


def test_rectification_display_bundle_warns_on_bad_inputs(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_workspace=True,
    )
    preview_path = output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "avg_rectified.png"
    preview_path.write_bytes(b"not an image")
    workspace_path = output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "workspace.npz"
    np.savez_compressed(
        workspace_path,
        pixel_displacements_x=np.zeros((2, 3), dtype=np.float64),
        pixel_displacements_y=np.zeros((3, 2), dtype=np.float64),
        average_rectified_global_image=np.zeros((2, 2), dtype=np.float64),
        average_rectified_image=np.zeros((3, 3), dtype=np.float64),
    )
    model = load_sweep_visualization_model(output_dir)

    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage="stage1",
            candidate_id="s1c0000",
            dataset_id="d1",
            component_name="gradient",
        ),
    )

    assert bundle.preview is None
    assert bundle.displacement_magnitude is None
    assert bundle.global_final_delta is None
    assert any("Failed to read preview image" in warning for warning in bundle.warnings)
    assert any("Displacement map shapes do not match" in warning for warning in bundle.warnings)
    assert any("Global/final average shapes do not match" in warning for warning in bundle.warnings)


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


def test_load_sweep_visualization_model_supports_validation_phase_as_non_primary(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_validation_phase=True,
    )

    model = load_sweep_visualization_model(output_dir)

    assert model.primary_stage_name == "stage2"
    assert model.stage_order == ["stage1", "stage2", "validation_phase"]
    validation_rows = [row for row in model.candidates if row.stage == "validation_phase"]
    assert [row.candidate_id for row in validation_rows] == ["v1c0000"]
    assert any(record.stage == "validation_phase" for record in model.run_metric_records)


def test_load_sweep_visualization_model_normalizes_stale_internal_metric_run_names(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
    )
    metric_dir = output_dir / "metric_experiments"
    stale_run_name = "metric_stage1_m1c0000_d1"
    summary_path = metric_dir / "summary.json"
    summary_records = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_records[0]["run"] = stale_run_name
    _write_json(summary_path, summary_records)
    _write_metric_record(
        metric_dir / "runs" / "metric_phase" / "m1c0000" / "d1" / "metric_record.json",
        run=stale_run_name,
        family="ncc_gradient",
        alpha=0.2,
        formulation_id="grad_0p2",
        metric_base=20.0,
    )

    model = load_sweep_visualization_model(output_dir)

    metric_records = [record for record in model.run_metric_records if record.stage == "metric_phase"]
    assert any(
        record.candidate_id == "m1c0000" and record.dataset_id == "d1"
        for record in metric_records
    )
    assert not any(record.stage == "metric_stage1" for record in model.run_metric_records)
    assert not any(record.candidate_id == "stage1_m1c0000" for record in model.run_metric_records)


def test_report_rebuild_normalizes_stale_internal_metric_record_identity(tmp_path):
    metric_dir = tmp_path / "metric_experiments"
    run_dir = metric_dir / "runs" / "metric_phase" / "m1c0000" / "d1"
    _write_metric_record(
        run_dir / "metric_record.json",
        run="metric_stage1_m1c0000_d1",
        family="intensity",
        alpha=1.0,
        formulation_id="intensity_default",
        metric_base=20.0,
    )
    _write_json(
        run_dir / "config.json",
        {
            "strategy": "combined",
            "displacement_metric_family": "intensity",
            "displacement_alpha": 1.0,
            "metric_formulation_id": "intensity_default",
        },
    )
    entry = {
        "run_uid": "metric_phase_m1c0000_d1",
        "stage": "metric_phase",
        "candidate_id": "m1c0000",
        "dataset_id": "d1",
        "status": "skipped_valid",
        "run_dir": str(run_dir),
        "generation_id": "gen1",
    }
    discovery = report_generation.StageDiscovery(
        stage_name="metric_phase",
        output_dir=metric_dir,
        source="run_state",
    )

    result = report_generation._run_result_from_entry(
        output_dir=metric_dir,
        stage_name="metric_phase",
        entry=entry,
        active_generation_id="gen1",
        run_state={"runs": {"metric_phase_m1c0000_d1": entry}},
        discovery=discovery,
    )

    assert result is not None
    assert result["run"] == "metric_phase_m1c0000_d1"
    assert result["metric_record"]["run"] == "metric_phase_m1c0000_d1"
    assert result["metric_record"]["stage"] == "metric_phase"
    assert result["metric_record"]["candidate_id"] == "m1c0000"
    assert result["metric_record"]["dataset_id"] == "d1"


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
    assert (export_dir / "sensitivity.png").exists()
    assert (export_dir / "metric_diagnostics.png").exists()
    assert (export_dir / "index.html").exists()
    assert (export_dir / "tables" / "candidate_leaderboard.csv").exists()
    assert (export_dir / "tables" / "run_metric_records.csv").exists()
    assert (export_dir / "tables" / "parameter_sensitivity.csv").exists()
    assert (export_dir / "data" / "report_context.json").exists()
    assert (export_dir / "data" / "parameter_sensitivity.json").exists()
    assert (export_dir / "data" / "next_extension_suggestions.json").exists()
    assert (export_dir / "assets" / "topk_overview.png").exists()
    assert (export_dir / "assets" / "preview_d1.png").exists()
    assert (export_dir / "assets" / "comparison_d1.png").exists()
    assert (export_dir / "assets" / "diagnostic_d1_gradient.png").exists()
    assert (export_dir / "assets" / "robustness_heatmap.png").exists()
    with np.load(export_dir / "visualization_bundle.npz", allow_pickle=True) as bundle:
        assert "score_matrix" in bundle.files
        assert "metric_tensor" in bundle.files
        assert "run_metric_records_json" in bundle.files
        assert "parameter_sensitivity_json" in bundle.files
        assert "next_extension_suggestions_json" in bundle.files
        assert "generation_summary_json" in bundle.files
        assert "diagnostic_availability_json" in bundle.files
        assert "primary_metric_source" in bundle.files
        assert bundle["score_matrix"].shape[1] == 2
        assert bundle["metric_tensor"].ndim == 3
    manifest = json.loads((export_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_html"] == "index.html"
    assert manifest["report_top_k"] == 6
    assert "tables/candidate_leaderboard.csv" in manifest["table_files"]
    assert "tables/parameter_sensitivity.csv" in manifest["table_files"]
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


def test_effect_search_rows_find_exact_aggregate_params_with_identity(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)

    active_ranking = _ranking(
        "s1c0000",
        score=0.91,
        rank=1,
        datasets=["d1", "d2"],
        stage="stage1",
        params={"max_displacement": 6},
    )
    active_ranking["generation_id"] = "gen_new"
    old_ranking = _ranking(
        "s1c0000",
        score=0.51,
        rank=1,
        datasets=["d1", "d2"],
        stage="stage1",
        params={"max_displacement": 11},
    )
    old_ranking["generation_id"] = "gen_old"
    _write_json(output_dir / "stage1_summary.json", _stage_summary("stage1", [active_ranking]))
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )

    model = load_sweep_visualization_model(output_dir)
    model.default_generation_scope = "All"

    all_rows = _effect_search_rows(model, "Aggregate candidates", {})
    exact_rows = _effect_search_rows(model, "Aggregate candidates", {"max_displacement": "11"})

    assert len(all_rows) == 2
    assert [row.generation_id for row in exact_rows] == ["gen_old"]
    assert exact_rows[0].row_id == "gen_old:stage1:s1c0000"
    assert exact_rows[0].candidate_id == "s1c0000"

    identity = _effect_row_identity_dict(exact_rows[0])
    assert identity["row_id"] == "gen_old:stage1:s1c0000"
    assert identity["generation_id"] == "gen_old"
    assert identity["candidate_id"] == "s1c0000"


def test_effect_search_rows_find_dataset_runs_with_artifacts(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
        include_workspace=True,
    )
    _write_metric_record(
        output_dir / "metric_experiments" / "runs" / "metric_phase" / "m1c0000" / "d1" / "metric_record.json",
        run="metric_m1c0000_d1",
        family="ncc_gradient",
        alpha=0.2,
        formulation_id="grad_0p2",
        metric_base=20.0,
    )

    model = load_sweep_visualization_model(output_dir)

    rows = _effect_search_rows(
        model,
        "Dataset runs",
        {"displacement_metric_family": "blurred_gradient_corr", "displacement_alpha": "0.2"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_kind == "run"
    assert row.dataset_id == "d1"
    assert row.run_name == "metric_m1c0000_d1"
    assert row.preview_path and row.preview_path.endswith("avg_rectified.png")
    assert row.workspace_path and row.workspace_path.endswith("workspace.npz")
    assert row.metric_record_path and row.metric_record_path.endswith("metric_record.json")

    detail = _format_effect_search_detail(row)
    assert "run: metric_m1c0000_d1" in detail
    assert "metric_record:" in detail


def test_compute_parameter_sensitivity_ranks_aggregate_and_dataset_effects(tmp_path):
    output_dir = _create_modern_fixture(tmp_path, include_stage2_summary=True)
    datasets = ["d1", "d2"]
    max_values = [4, 4, 8, 8, 12, 12]
    sigma_values = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    aggregate_scores = [0.10, 0.15, 0.45, 0.50, 0.90, 0.95]
    d1_scores = [0.95, 0.90, 0.50, 0.45, 0.10, 0.05]
    d2_scores = [0.05, 0.10, 0.50, 0.55, 0.90, 0.95]
    rankings = []
    for idx, (max_value, sigma_value, score) in enumerate(
        zip(max_values, sigma_values, aggregate_scores)
    ):
        row = _ranking(
            f"s1c{idx:04d}",
            score=score,
            rank=idx + 1,
            datasets=datasets,
            stage="stage1",
            params={"max_displacement": max_value, "gauss_sigma": sigma_value},
        )
        row["dataset_scores"] = {"d1": d1_scores[idx], "d2": d2_scores[idx]}
        row["mean_dataset_score"] = float(np.mean(list(row["dataset_scores"].values())))
        row["std_dataset_score"] = float(np.std(list(row["dataset_scores"].values())))
        rankings.append(row)
    _write_json(output_dir / "stage1_summary.json", _stage_summary("stage1", rankings))

    model = load_sweep_visualization_model(output_dir)
    aggregate_rows = compute_parameter_sensitivity(
        model,
        stage_name="stage1",
        dataset_scope="aggregate",
        value_option="score",
        filters={},
    )
    d1_rows = compute_parameter_sensitivity(
        model,
        stage_name="stage1",
        dataset_scope="d1",
        value_option="score",
        filters={},
    )
    d2_rows = compute_parameter_sensitivity(
        model,
        stage_name="stage1",
        dataset_scope="d2",
        value_option="score",
        filters={},
    )
    figure = create_parameter_sensitivity_figure(model, aggregate_rows)

    assert aggregate_rows[0].parameter_name == "max_displacement"
    assert aggregate_rows[0].impact_score > 0.70
    assert aggregate_rows[0].ols_status == "ok"
    assert aggregate_rows[0].ols_delta_r2 is not None
    assert next(row for row in d1_rows if row.parameter_name == "max_displacement").best_value == "4"
    assert next(row for row in d2_rows if row.parameter_name == "max_displacement").best_value == "12"
    constant_row = next(row for row in aggregate_rows if row.parameter_name == "pixel_threshold")
    assert constant_row.coverage_warning == "only one finite parameter value"
    assert figure.axes[0].get_xlabel() == "grouped mean delta"


def test_extension_suggestions_use_previous_generation_and_boundary_evidence(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run", "created_at": "2026-04-01T00:00:00+00:00"},
        {"generation_id": "gen_new", "action": "extend", "created_at": "2026-04-02T00:00:00+00:00"},
    ]
    _write_json(output_dir / "run_state.json", run_state)

    datasets = ["d1", "d2"]
    current_rankings = []
    for idx, (max_value, score) in enumerate([(4, 0.20), (8, 0.50), (12, 0.92)]):
        row = _ranking(
            f"s1c{idx:04d}",
            score=score,
            rank=idx + 1,
            datasets=datasets,
            stage="stage1",
            params={"max_displacement": max_value},
        )
        row["generation_id"] = "gen_new"
        current_rankings.append(row)
    _write_json(output_dir / "stage1_summary.json", _stage_summary("stage1", current_rankings))

    previous_rankings = []
    for idx, (max_value, score) in enumerate([(4, 0.25), (8, 0.70)]):
        row = _ranking(
            f"s1c_old{idx:04d}",
            score=score,
            rank=idx + 1,
            datasets=datasets,
            stage="stage1",
            params={"max_displacement": max_value},
        )
        row["generation_id"] = "gen_old"
        previous_rankings.append(row)
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", previous_rankings),
    )

    model = load_sweep_visualization_model(output_dir)
    rows = compute_parameter_sensitivity(
        model,
        stage_name="stage1",
        dataset_scope="aggregate",
        value_option="score",
        filters={},
    )
    suggestions = generate_next_extension_suggestions(model, rows)

    max_row = next(row for row in rows if row.parameter_name == "max_displacement")
    assert model.generation_summary["previous_generation_id"] == "gen_old"
    assert model.generation_summary["previous_generation_candidate_count"] == 2
    assert max_row.previous_best_value == "8"
    assert max_row.best_value == "12"
    assert any(suggestion.rule_id == "boundary_winner" for suggestion in suggestions)
    assert any(suggestion.rule_id == "extension_delta" for suggestion in suggestions)


def test_generation_scope_helpers_load_all_generations_with_qualified_rows(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run", "created_at": "2026-04-01T00:00:00+00:00"},
        {"generation_id": "gen_mid", "action": "extend", "created_at": "2026-04-02T00:00:00+00:00"},
        {"generation_id": "gen_new", "action": "extend", "created_at": "2026-04-03T00:00:00+00:00"},
    ]
    _write_json(output_dir / "run_state.json", run_state)

    active_rankings = []
    for idx, score in enumerate([0.91, 0.82]):
        row = _ranking(
            f"s1c{idx:04d}",
            score=score,
            rank=idx + 1,
            datasets=["d1", "d2"],
            stage="stage1",
        )
        row["generation_id"] = "gen_new"
        active_rankings.append(row)
    _write_json(output_dir / "stage1_summary.json", _stage_summary("stage1", active_rankings))

    for generation_id, score in [("gen_old", 0.51), ("gen_mid", 0.73)]:
        row = _ranking(
            "s1c0000",
            score=score,
            rank=1,
            datasets=["d1", "d2"],
            stage="stage1",
        )
        row["generation_id"] = generation_id
        _write_json(
            output_dir / "generations" / generation_id / "products" / "stage1_summary.json",
            _stage_summary("stage1", [row]),
        )

    model = load_sweep_visualization_model(output_dir)

    assert generation_scope_options(model) == ["All", "Active", "gen_old", "gen_mid", "gen_new"]
    active_rows = candidate_rows_for_generation_scope(model, "Active")
    all_rows = candidate_rows_for_generation_scope(model, "All")
    assert [row.generation_id for row in active_rows] == ["gen_new", "gen_new"]
    assert sorted(row.generation_id for row in all_rows) == ["gen_mid", "gen_new", "gen_new", "gen_old"]
    assert [row.rank for row in all_rows] == [1, 2, 3, 4]
    assert [row.score for row in all_rows] == [0.91, 0.82, 0.73, 0.51]
    assert len({row.row_id for row in all_rows}) == 4
    assert len([row for row in all_rows if row.candidate_id == "s1c0000"]) == 3
    assert candidate_rows_for_generation_scope(model, None) == all_rows


def test_generation_scope_helpers_include_historical_run_records(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)

    active_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    active_summary[0]["generation_id"] = "gen_new"
    _write_json(output_dir / "summary.json", active_summary)

    old_record = dict(active_summary[0])
    old_record["run"] = "stage1_s1c0000_d1"
    old_record["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "summary.json",
        [old_record],
    )
    old_ranking = _ranking(
        "s1c0000",
        score=0.44,
        rank=1,
        datasets=["d1", "d2"],
        stage="stage1",
    )
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )

    model = load_sweep_visualization_model(output_dir)

    all_records = run_records_for_generation_scope(model, "All")
    assert [record.generation_id for record in all_records] == ["gen_old", "gen_new"]
    assert len({record.row_id for record in all_records}) == 2
    assert run_records_for_generation_scope(model, "Active")[0].generation_id == "gen_new"


def test_historical_generation_bundle_loads_archived_product_preview(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)
    old_ranking = _ranking("s1c_old0000", score=0.44, rank=1, datasets=["d1"], stage="stage1")
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )
    _write_preview(
        output_dir
        / "generations"
        / "gen_old"
        / "products"
        / "runs"
        / "stage1"
        / "s1c_old0000"
        / "d1"
        / "avg_rectified.png"
    )

    model = load_sweep_visualization_model(output_dir)
    model.default_generation_scope = "All"
    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage="stage1",
            candidate_id="s1c_old0000",
            dataset_id="d1",
            generation_id="gen_old",
        ),
    )

    assert bundle.preview is not None
    assert bundle.preview_path is not None
    assert "generations" in bundle.preview_path


def test_historical_generation_bundle_falls_back_to_matching_root_run_artifacts(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
        include_workspace=True,
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)
    old_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    old_summary[0]["generation_id"] = "gen_old"
    _write_json(output_dir / "generations" / "gen_old" / "products" / "summary.json", old_summary)
    old_ranking = _ranking("s1c0000", score=0.44, rank=1, datasets=["d1"], stage="stage1")
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )
    root_metric_record = output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "metric_record.json"
    _write_metric_record(
        root_metric_record,
        run="stage1_s1c0000_d1",
        family="intensity",
        alpha=1.0,
        formulation_id="intensity_default",
        metric_base=10.0,
    )
    metric_payload = json.loads(root_metric_record.read_text(encoding="utf-8"))
    metric_payload["generation_id"] = "gen_old"
    _write_json(root_metric_record, metric_payload)

    model = load_sweep_visualization_model(output_dir)
    model.default_generation_scope = "All"
    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage="stage1",
            candidate_id="s1c0000",
            dataset_id="d1",
            generation_id="gen_old",
            component_name="gradient",
        ),
    )

    assert bundle.preview is not None
    assert bundle.blended_score is not None
    assert bundle.preview_path == str(output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "avg_rectified.png")
    assert bundle.workspace_path == str(output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "workspace.npz")


def test_historical_generation_root_fallback_rejects_mismatched_generation_id(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)
    old_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    old_summary[0]["generation_id"] = "gen_old"
    _write_json(output_dir / "generations" / "gen_old" / "products" / "summary.json", old_summary)
    old_ranking = _ranking("s1c0000", score=0.44, rank=1, datasets=["d1"], stage="stage1")
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )
    root_metric_record = output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "metric_record.json"
    _write_metric_record(
        root_metric_record,
        run="stage1_s1c0000_d1",
        family="intensity",
        alpha=1.0,
        formulation_id="intensity_default",
        metric_base=10.0,
    )
    metric_payload = json.loads(root_metric_record.read_text(encoding="utf-8"))
    metric_payload["generation_id"] = "gen_new"
    _write_json(root_metric_record, metric_payload)

    model = load_sweep_visualization_model(output_dir)
    model.default_generation_scope = "All"
    bundle = load_rectification_display_bundle(
        model,
        RectificationViewSelection(
            stage="stage1",
            candidate_id="s1c0000",
            dataset_id="d1",
            generation_id="gen_old",
        ),
    )

    assert bundle.preview is None
    assert bundle.preview_path is None
    assert any("Preview image unavailable" in warning for warning in bundle.warnings)


def test_comparison_artifacts_use_generation_aware_run_record(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
        include_workspace=True,
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)
    active_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    active_summary[0]["generation_id"] = "gen_new"
    _write_json(output_dir / "summary.json", active_summary)
    old_summary = [dict(active_summary[0], generation_id="gen_old")]
    _write_json(output_dir / "generations" / "gen_old" / "products" / "summary.json", old_summary)
    old_ranking = _ranking("s1c0000", score=0.44, rank=1, datasets=["d1"], stage="stage1")
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )
    root_metric_record = output_dir / "runs" / "stage1" / "s1c0000" / "d1" / "metric_record.json"
    _write_metric_record(
        root_metric_record,
        run="stage1_s1c0000_d1",
        family="intensity",
        alpha=1.0,
        formulation_id="intensity_default",
        metric_base=10.0,
    )
    metric_payload = json.loads(root_metric_record.read_text(encoding="utf-8"))
    metric_payload["generation_id"] = "gen_old"
    _write_json(root_metric_record, metric_payload)

    model = load_sweep_visualization_model(output_dir)
    model.default_generation_scope = "All"
    row = next(
        row
        for row in candidate_rows_for_generation_scope(model, "All")
        if row.candidate_id == "s1c0000" and row.generation_id == "gen_old"
    )
    artifact = _comparison_artifacts_for_candidate(
        model,
        row,
        dataset_id="d1",
        component_name="gradient",
    )

    assert artifact["preview"] is not None
    assert artifact["component"] is not None
    assert artifact["record"].generation_id == "gen_old"


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


def test_format_comparison_text_reports_mixed_stages(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_metric_phase=True,
    )
    model = load_sweep_visualization_model(output_dir)
    stage2_row = next(row for row in model.candidates if row.stage == "stage2")
    metric_row = next(row for row in model.candidates if row.stage == "metric_phase")

    text = _format_comparison_text(model, [stage2_row, metric_row], dataset_id="d1")

    assert "stages: stage2, metric_phase" in text
    assert "stage=stage2" in text
    assert "stage=metric_phase" in text


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
    assert "Parameter Locator" in tab_names
    assert tab_names.index("Parameter Locator") == tab_names.index("Effects") + 1
    assert "Sensitivity" in tab_names
    assert "Next Extensions" in tab_names
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


def test_gui_comparison_slots_channel_lock_and_sync(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_workspace=True,
    )
    for dataset_id in ("d1", "d2"):
        _write_preview(output_dir / "runs" / "stage1" / "s1c0001" / dataset_id / "avg_rectified.png")
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
    root.update()

    app.leaderboard_stage_var.set("stage1")
    app.refresh_leaderboard()
    children = app.leaderboard_tree.get_children()
    assert len(children) >= 2
    app.leaderboard_tree.selection_set(children[:2])
    app._assign_comparison_slots_from_selection()
    root.update()

    assert app._comparison_slot_a_key is not None
    assert app._comparison_slot_b_key is not None
    assert app.comparison_inspector_a.has_image()
    assert app.comparison_inspector_b.has_image()

    app.comparison_inspector_b.set_channel("component_score", emit_event=False, reset_view=False)
    app.comparison_inspector_a.set_channel("preview", emit_event=True, reset_view=False)
    assert app.comparison_inspector_b.current_channel() == "preview"

    app.comparison_inspector_a.apply_axes_limits((0.0, 5.0), (5.0, 0.0))
    app._on_comparison_limits_changed(app.comparison_inspector_a)
    assert app.comparison_inspector_b.get_axes_limits() == ((0.0, 5.0), (5.0, 0.0))

    app.comparison_sync_var.set(False)
    app.comparison_inspector_a.apply_axes_limits((1.0, 4.0), (4.0, 1.0))
    app._on_comparison_limits_changed(app.comparison_inspector_a)
    assert app.comparison_inspector_b.get_axes_limits() == ((0.0, 5.0), (5.0, 0.0))
    root.destroy()


def test_gui_comparison_stage_pair_controls_seed_cross_stage_slots(tmp_path):
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

    app.comparison_stage_a_var.set("stage2")
    app._on_comparison_stage_changed("A")
    app.comparison_stage_b_var.set("metric_phase")
    app._on_comparison_stage_changed("B")
    root.update()

    assert app._comparison_slot_a_key is not None
    assert app._comparison_slot_b_key is not None
    assert app._comparison_slot_a_key[0] == "stage2"
    assert app._comparison_slot_b_key[0] == "metric_phase"
    assert app.comparison_inspector_a.has_image()
    assert app.comparison_inspector_b.has_image()
    comparison_text = app.comparison_text.get("1.0", "end").strip()
    assert "stages: stage2, metric_phase" in comparison_text
    root.destroy()


def test_gui_use_two_selected_preserves_cross_stage_comparison(tmp_path):
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

    app.leaderboard_stage_var.set("All")
    app.refresh_leaderboard()
    selected_ids = [
        row_id
        for row_id, row in app._leaderboard_row_lookup.items()
        if row.stage in {"stage2", "metric_phase"}
    ][:2]
    assert len(selected_ids) == 2
    app.leaderboard_tree.selection_set(selected_ids)
    app._assign_comparison_slots_from_selection()
    root.update()

    assert app._comparison_slot_a_key is not None
    assert app._comparison_slot_b_key is not None
    assert {app._comparison_slot_a_key[0], app._comparison_slot_b_key[0]} == {"stage2", "metric_phase"}
    app.refresh_comparison()
    assert app._comparison_slot_a_key is not None
    assert app._comparison_slot_b_key is not None
    assert {app._comparison_slot_a_key[0], app._comparison_slot_b_key[0]} == {"stage2", "metric_phase"}
    root.destroy()


def test_gui_preview_actions_use_zoomable_viewer(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_workspace=True,
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    app.messagebox.showwarning = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
    captures = []
    app._open_rectification_viewer = lambda bundle, *, title: captures.append((bundle, title))
    root.update()

    app.preview_stage_var.set("stage1")
    app._refresh_preview_candidates()
    app.preview_candidate_var.set("s1c0000")
    app._refresh_preview_datasets()
    app.preview_dataset_var.set("d1")
    app._open_preview()

    app.diagnostic_stage_var.set("stage1")
    app._refresh_diagnostic_candidates()
    app.diagnostic_candidate_var.set("s1c0000")
    app._refresh_diagnostic_datasets()
    app.diagnostic_dataset_var.set("d1")
    app._open_selected_diagnostic_preview()

    assert len(captures) == 2
    assert captures[0][0].selection.candidate_id == "s1c0000"
    assert captures[1][0].selection.dataset_id == "d1"
    root.destroy()


def test_gui_preview_tab_opens_generation_qualified_candidate(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    run_state = _run_state_payload(output_dir, ["d1", "d2"], stage_mode="single_stage")
    run_state["active_generation_id"] = "gen_new"
    run_state["spec_generations"] = [
        {"generation_id": "gen_old", "action": "run"},
        {"generation_id": "gen_new", "action": "extend"},
    ]
    _write_json(output_dir / "run_state.json", run_state)
    old_ranking = _ranking("s1c_old0000", score=0.44, rank=1, datasets=["d1"], stage="stage1")
    old_ranking["generation_id"] = "gen_old"
    _write_json(
        output_dir / "generations" / "gen_old" / "products" / "stage1_summary.json",
        _stage_summary("stage1", [old_ranking]),
    )
    _write_preview(
        output_dir
        / "generations"
        / "gen_old"
        / "products"
        / "runs"
        / "stage1"
        / "s1c_old0000"
        / "d1"
        / "avg_rectified.png"
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    app.messagebox.showwarning = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
    captures = []
    app._open_rectification_viewer = lambda bundle, *, title: captures.append((bundle, title))
    root.update()

    app.generation_scope_var.set("All")
    app._on_generation_scope_changed()
    app.preview_stage_var.set("stage1")
    app._refresh_preview_candidates()
    old_key = "gen_old:stage1:s1c_old0000"
    assert old_key in app.preview_candidate_combo["values"]
    app.preview_candidate_var.set(old_key)
    app._refresh_preview_datasets()
    app.preview_dataset_var.set("d1")
    app._open_preview()

    assert len(captures) == 1
    assert captures[0][0].selection.generation_id == "gen_old"
    assert captures[0][0].selection.candidate_id == "s1c_old0000"
    root.destroy()


def test_gui_effect_search_filters_and_shows_identity_detail(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    root.update()

    app.effect_search_source_var.set("Aggregate candidates")
    app.effect_search_filter_vars["max_displacement"].set("6")
    app._refresh_effect_search_results()
    root.update()

    children = app.effect_search_tree.get_children()
    assert len(children) == 1
    app.effect_search_tree.selection_set(children[0])
    app._on_effect_search_selection_changed()
    detail_text = app.effect_search_detail_text.get("1.0", "end").strip()

    assert "row_id: stage1:s1c0000" in detail_text
    assert "generation:" in detail_text
    assert "stage: stage1" in detail_text
    assert "candidate: s1c0000" in detail_text
    assert "params:" in detail_text
    root.destroy()


def test_gui_parameter_locator_lives_outside_effects_tab(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    root.update()

    assert app.effect_search_frame.master is app.locator_tab
    assert app.effect_search_frame not in app.coverage_tab.winfo_children()
    assert app.effect_search_tree.master.master is app.effect_search_frame
    root.destroy()


def test_gui_effect_filters_can_seed_parameter_locator(tmp_path):
    output_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        stage_mode="single_stage",
    )
    root = _make_root_or_skip()
    app = SweepVisualizationApp(root, initial_path=output_dir)
    root.update()

    app.effect_search_source_var.set("Aggregate candidates")
    app.effect_search_filter_vars["max_displacement"].set("9")
    app.effect_search_filter_vars["gauss_sigma"].set("999")
    app._refresh_effect_search_results()
    assert len(app.effect_search_tree.get_children()) == 0

    app._effect_filter_vars["max_displacement"].set("6")
    app._send_effects_filters_to_locator()
    root.update()

    assert app.effect_search_filter_vars["max_displacement"].get() == "6"
    assert app.effect_search_filter_vars["gauss_sigma"].get() == "All"
    assert app.notebook.select() == str(app.locator_tab)
    children = app.effect_search_tree.get_children()
    assert len(children) == 1
    assert "stage1:s1c0000" in children[0]
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

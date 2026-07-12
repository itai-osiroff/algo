import json
import argparse
import csv
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import moment_images.parameter_sweep as parameter_sweep
from moment_images.im_process import RectificationResult, RectificationRunConfig, rectify_views_from_config
from moment_images.parameter_sweep import (
    ExperimentalMetricSweepSpec,
    MetricFormulationSpec,
    PreparedDataset,
    SweepDatasetSpec,
    SweepCandidate,
    SweepExperimentSpec,
    SweepScoringSpec,
    SweepSearchSpace,
    SweepStageRefinementSpec,
    SweepStageSpec,
    generate_coarse_candidates,
    generate_refined_candidates,
    run_sweep_experiment,
    score_and_rank_runs,
)


def _pixel_space(**overrides: object) -> SweepSearchSpace:
    base = dict(
        strategy="pixel",
        similarity_window_sizes=[1, 3],
        max_displacements=[0, 1],
        gauss_sigmas=[1.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.05],
        sub_pix_factors=[1, 2],
        pixel_sigma_pyramid=[False, True],
        zero_fill_roll=[True],
    )
    base.update(overrides)
    return SweepSearchSpace(**base)


def _metric_record(laplacian_variance: float) -> dict:
    return {
        "global_only_metrics": None,
        "global_and_local_metrics": {
            "average_intensity": 0.0,
            "maximum_intensity": 0.0,
            "total_variation": 0.0,
            "sum_squared_gradient": 0.0,
            "laplacian_variance": float(laplacian_variance),
            "dwt_level2_hv_energy": 0.0,
        },
    }


def _build_eval_result(task: dict, status: str = "ok") -> dict:
    payload = {
        "status": status,
        "stage": task["stage"],
        "candidate_id": task["candidate_id"],
        "dataset_id": task["dataset_id"],
        "strategy": task["strategy"],
        "run_uid": task["run_uid"],
        "runtime_seconds": 0.01,
        "config": task["config"],
        "params": task["params"],
        "task_index": task["task_index"],
    }
    if status == "ok":
        payload["metric_record"] = _metric_record(1.0)
    else:
        payload["error"] = "worker crashed"
    return payload


def _install_fast_artifact_runner(monkeypatch):
    calls: list[str] = []

    def fake_run(task: dict) -> tuple[dict, bool]:
        calls.append(str(task["run_uid"]))
        config = RectificationRunConfig(**task["config"])
        views_shape = tuple(int(dim) for dim in task["views_shape"])
        rectified_shape = views_shape if len(views_shape) == 4 else views_shape[1:]
        rectified = np.zeros(rectified_shape, dtype=np.float32)
        avg = np.zeros(rectified_shape[-2:], dtype=np.float32)
        result = RectificationResult(
            avg_image=avg,
            rectified_views=rectified,
            similarity_data=None,
            config=config,
            ref_view_index=(0, 0),
        )
        run_dir = Path(task["run_dir"])
        artifact_paths = parameter_sweep._artifact_paths_payload(run_dir)
        metric_record = parameter_sweep.build_metric_record(str(task["run_uid"]), config, result)
        metric_record.update(
            {
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            }
        )
        parameter_sweep._save_run_artifacts(
            Path(task["run_parent_dir"]),
            str(task["run_name"]),
            result,
            metric_record=metric_record,
        )
        return (
            {
                "status": "ok",
                "stage": task["stage"],
                "candidate_id": task["candidate_id"],
                "dataset_id": task["dataset_id"],
                "strategy": task["strategy"],
                "run_uid": task["run_uid"],
                "run_dir": str(run_dir),
                "runtime_seconds": 0.01,
                "config": task["config"],
                "params": task["params"],
                "task_index": task["task_index"],
                "metric_record": metric_record,
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            },
            False,
        )

    monkeypatch.setattr(parameter_sweep, "_run_task_isolated", fake_run)
    return calls


def _install_window_score_artifact_runner(monkeypatch):
    calls: list[dict] = []

    def fake_run(task: dict) -> tuple[dict, bool]:
        config = RectificationRunConfig(**task["config"])
        calls.append(
            {
                "run_uid": str(task["run_uid"]),
                "stage": str(task["stage"]),
                "window": int(config.similarity_window_size),
            }
        )
        views_shape = tuple(int(dim) for dim in task["views_shape"])
        rectified_shape = views_shape if len(views_shape) == 4 else views_shape[1:]
        rectified = np.zeros(rectified_shape, dtype=np.float32)
        avg = np.zeros(rectified_shape[-2:], dtype=np.float32)
        result = RectificationResult(
            avg_image=avg,
            rectified_views=rectified,
            similarity_data=None,
            config=config,
            ref_view_index=(0, 0),
        )
        run_dir = Path(task["run_dir"])
        artifact_paths = parameter_sweep._artifact_paths_payload(run_dir)
        metric_record = parameter_sweep.build_metric_record(str(task["run_uid"]), config, result)
        score = float(config.similarity_window_size)
        metric_record["global_and_local_metrics"]["laplacian_variance"] = score
        if isinstance(metric_record.get("global_only_metrics"), dict):
            metric_record["global_only_metrics"]["laplacian_variance"] = score
        metric_record.update(
            {
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            }
        )
        parameter_sweep._save_run_artifacts(
            Path(task["run_parent_dir"]),
            str(task["run_name"]),
            result,
            metric_record=metric_record,
        )
        return (
            {
                "status": "ok",
                "stage": task["stage"],
                "candidate_id": task["candidate_id"],
                "dataset_id": task["dataset_id"],
                "strategy": task["strategy"],
                "run_uid": task["run_uid"],
                "run_dir": str(run_dir),
                "runtime_seconds": 0.01,
                "config": task["config"],
                "params": task["params"],
                "task_index": task["task_index"],
                "metric_record": metric_record,
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            },
            False,
        )

    monkeypatch.setattr(parameter_sweep, "_run_task_isolated", fake_run)
    return calls


def _structured_avg_image(window_size: int) -> np.ndarray:
    if int(window_size) == 1:
        return np.arange(64, dtype=np.float32).reshape(8, 8)
    if int(window_size) == 3:
        image = np.zeros((8, 8), dtype=np.float32)
        image[3:5, 3:5] = 10.0
        return image
    return np.indices((8, 8)).sum(axis=0).astype(np.float32) % 2


def _install_structured_artifact_runner(monkeypatch):
    calls: list[dict] = []

    def fake_run(task: dict) -> tuple[dict, bool]:
        config = RectificationRunConfig(**task["config"])
        window_size = int(config.similarity_window_size)
        calls.append(
            {
                "run_uid": str(task["run_uid"]),
                "stage": str(task["stage"]),
                "window": window_size,
            }
        )
        avg = _structured_avg_image(window_size)
        rectified = np.broadcast_to(avg, (2, 2, *avg.shape)).astype(np.float32)
        result = RectificationResult(
            avg_image=avg,
            rectified_views=rectified,
            similarity_data=None,
            config=config,
            ref_view_index=(0, 0),
        )
        run_dir = Path(task["run_dir"])
        artifact_paths = parameter_sweep._artifact_paths_payload(run_dir)
        metric_record = parameter_sweep.build_metric_record(str(task["run_uid"]), config, result)
        metric_record.update(
            {
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            }
        )
        parameter_sweep._save_run_artifacts(
            Path(task["run_parent_dir"]),
            str(task["run_name"]),
            result,
            metric_record=metric_record,
        )
        return (
            {
                "status": "ok",
                "stage": task["stage"],
                "candidate_id": task["candidate_id"],
                "dataset_id": task["dataset_id"],
                "strategy": task["strategy"],
                "run_uid": task["run_uid"],
                "run_dir": str(run_dir),
                "runtime_seconds": 0.01,
                "config": task["config"],
                "params": task["params"],
                "task_index": task["task_index"],
                "metric_record": metric_record,
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": task.get("generation_id"),
                "artifact_paths": artifact_paths,
            },
            False,
        )

    monkeypatch.setattr(parameter_sweep, "_run_task_isolated", fake_run)
    return calls


def _remove_metric_from_run_records(output_dir: Path, metric_name: str) -> None:
    for path in output_dir.glob("runs/*/*/*/metric_record.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("global_and_local_metrics")
        if isinstance(metrics, dict):
            metrics.pop(metric_name, None)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_generate_coarse_candidates_global_ignores_pixel_only_dimensions():
    search_space = SweepSearchSpace(
        strategy="global",
        correlation_pad_widths=[10, 20],
        global_thresholds=[0.1, 0.2],
        max_displacements=[0, 1, 2],
        sub_pix_factors=[1, 2],
        pixel_sigma_pyramid=[False, True],
    )

    candidates = generate_coarse_candidates(search_space)

    assert len(candidates) == 4
    assert {candidate.config.correlation_pad_width for candidate in candidates} == {10, 20}
    assert {candidate.config.global_threshold for candidate in candidates} == {0.1, 0.2}
    assert all(candidate.config.max_displacement is None for candidate in candidates)
    assert all(candidate.config.strategy == "global" for candidate in candidates)


def test_generate_coarse_candidates_pixel_includes_subpixel_and_sigma_pyramid():
    search_space = _pixel_space(
        similarity_window_sizes=[3],
        max_displacements=[1],
        sub_pix_factors=[1, 2],
        pixel_sigma_pyramid=[False, True],
    )

    candidates = generate_coarse_candidates(search_space)

    assert len(candidates) == 4
    assert {candidate.config.sub_pix_factor for candidate in candidates} == {1, 2}
    assert {candidate.config.pixel_sigma_pyramid for candidate in candidates} == {False, True}


def test_even_window_is_normalized_and_metadata_is_stable():
    search_space = _pixel_space(
        similarity_window_sizes=[10],
        max_displacements=[0],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )

    candidates = generate_coarse_candidates(search_space)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.config.similarity_window_size == 11
    assert candidate.params["similarity_window_size"] == 11
    assert candidate.config.zero_fill_roll is True


def test_generate_refined_candidates_respects_budget_and_deduplicates():
    search_space = _pixel_space(
        similarity_window_sizes=[1, 3, 5],
        max_displacements=[0, 1],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
    )
    coarse = generate_coarse_candidates(search_space)
    for idx, candidate in enumerate(coarse):
        candidate.candidate_id = f"c{idx:03d}"
    lookup = {candidate.candidate_id: candidate for candidate in coarse}
    ranked = [{"candidate_id": coarse[0].candidate_id, "score": 1.0}]

    refined = generate_refined_candidates(
        ranked_candidates=ranked,
        candidate_lookup=lookup,
        search_spaces={"pixel": search_space},
        top_k=1,
        budget=3,
    )

    assert len(refined) <= 3
    signatures = {json.dumps(candidate.params, sort_keys=True) for candidate in refined}
    assert len(signatures) == len(refined)


def test_generate_refined_candidates_auto_fine_interpolates_and_extrapolates():
    search_space = _pixel_space(
        similarity_window_sizes=[15, 25],
        max_displacements=[5, 15],
        gauss_sigmas=[1.0, 3.0],
        disp_gauss_sigmas=[0.0, 2.0],
        pixel_thresholds=[0.1, 0.3],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )
    coarse = generate_coarse_candidates(search_space)
    for idx, candidate in enumerate(coarse):
        candidate.candidate_id = f"c{idx:03d}"
    seed = next(
        candidate
        for candidate in coarse
        if candidate.params["similarity_window_size"] == 15
        and candidate.params["max_displacement"] == 5
        and candidate.params["gauss_sigma"] == 1.0
        and candidate.params["pixel_threshold"] == 0.1
    )
    lookup = {candidate.candidate_id: candidate for candidate in coarse}

    refined = generate_refined_candidates(
        ranked_candidates=[{"candidate_id": seed.candidate_id, "score": 1.0}],
        candidate_lookup=lookup,
        search_spaces={"pixel": search_space},
        top_k=1,
        budget=None,
    )

    windows = {candidate.params["similarity_window_size"] for candidate in refined}
    max_displacements = {candidate.params["max_displacement"] for candidate in refined}
    gauss_sigmas = {candidate.params["gauss_sigma"] for candidate in refined}
    thresholds = {candidate.params["pixel_threshold"] for candidate in refined}

    assert {11, 15, 21}.issubset(windows)
    assert {0, 5, 10}.issubset(max_displacements)
    assert {0.0, 1.0, 2.0}.issubset(gauss_sigmas)
    assert {0.0, 0.1, 0.2}.issubset(thresholds)
    assert any(candidate.params["similarity_window_size"] not in {15, 25} for candidate in refined)


def test_generate_refined_candidates_inside_only_skips_edge_extrapolation():
    search_space = _pixel_space(
        similarity_window_sizes=[15, 25],
        max_displacements=[5, 15],
        gauss_sigmas=[1.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.1],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )
    coarse = generate_coarse_candidates(search_space)
    for idx, candidate in enumerate(coarse):
        candidate.candidate_id = f"c{idx:03d}"
    seed = next(candidate for candidate in coarse if candidate.params["similarity_window_size"] == 15)
    lookup = {candidate.candidate_id: candidate for candidate in coarse}

    refined = generate_refined_candidates(
        ranked_candidates=[{"candidate_id": seed.candidate_id, "score": 1.0}],
        candidate_lookup=lookup,
        search_spaces={"pixel": search_space},
        top_k=1,
        budget=None,
        refinement=SweepStageRefinementSpec(extrapolation="inside_only"),
    )

    windows = {candidate.params["similarity_window_size"] for candidate in refined}
    assert 21 in windows
    assert 11 not in windows


def test_generate_refined_candidates_neighbor_method_preserves_configured_values():
    search_space = _pixel_space(
        similarity_window_sizes=[15, 25],
        max_displacements=[5, 15],
        gauss_sigmas=[1.0, 3.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.1],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )
    coarse = generate_coarse_candidates(search_space)
    for idx, candidate in enumerate(coarse):
        candidate.candidate_id = f"c{idx:03d}"
    seed = next(candidate for candidate in coarse if candidate.params["similarity_window_size"] == 15)
    lookup = {candidate.candidate_id: candidate for candidate in coarse}

    refined = generate_refined_candidates(
        ranked_candidates=[{"candidate_id": seed.candidate_id, "score": 1.0}],
        candidate_lookup=lookup,
        search_spaces={"pixel": search_space},
        top_k=1,
        budget=None,
        refinement=SweepStageRefinementSpec(method="neighbor"),
    )

    windows = {candidate.params["similarity_window_size"] for candidate in refined}
    gauss_sigmas = {candidate.params["gauss_sigma"] for candidate in refined}
    assert windows <= {15, 25}
    assert gauss_sigmas <= {1.0, 3.0}


def test_score_and_rank_runs_applies_missing_dataset_penalty():
    run_results = [
        {
            "status": "ok",
            "candidate_id": "A",
            "dataset_id": "d1",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(10.0),
        },
        {
            "status": "ok",
            "candidate_id": "A",
            "dataset_id": "d2",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(10.0),
        },
        {
            "status": "ok",
            "candidate_id": "B",
            "dataset_id": "d1",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(8.0),
        },
        {
            "status": "ok",
            "candidate_id": "B",
            "dataset_id": "d2",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(0.0),
        },
        {
            "status": "ok",
            "candidate_id": "C",
            "dataset_id": "d1",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(9.0),
        },
    ]
    scoring = SweepScoringSpec(
        metric_weights={"laplacian_variance": 1.0},
        runtime_penalty_weight=0.0,
        robustness_lambda=0.5,
        missing_dataset_penalty=1.0,
    )

    rankings = score_and_rank_runs(run_results, scoring, dataset_ids=["d1", "d2"])

    assert rankings[0]["candidate_id"] == "A"
    by_id = {row["candidate_id"]: row for row in rankings}
    assert by_id["C"]["score"] < by_id["B"]["score"]


def test_score_and_rank_runs_accepts_new_explorer_metric():
    def record(candidate_id: str, metric_value: float) -> dict:
        payload = {
            "status": "ok",
            "candidate_id": candidate_id,
            "dataset_id": "d1",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(0.0),
        }
        payload["metric_record"]["global_and_local_metrics"]["vollath_f4_guide"] = metric_value
        return payload

    rankings = score_and_rank_runs(
        [record("low", 1.0), record("high", 10.0)],
        SweepScoringSpec(metric_weights={"vollath_f4_guide": 1.0}),
        dataset_ids=["d1"],
    )

    assert rankings[0]["candidate_id"] == "high"


def test_score_and_rank_runs_accepts_tamura_profile_metric():
    def record(candidate_id: str, metric_value: float) -> dict:
        payload = {
            "status": "ok",
            "candidate_id": candidate_id,
            "dataset_id": "d1",
            "strategy": "pixel",
            "config": {},
            "params": {},
            "runtime_seconds": 1.0,
            "metric_record": _metric_record(0.0),
        }
        payload["metric_record"]["global_and_local_metrics"]["tamura_gradient_avg"] = metric_value
        return payload

    rankings = score_and_rank_runs(
        [record("low", 0.25), record("high", 2.5)],
        SweepScoringSpec(metric_weights={"tamura_gradient_avg": 1.0}),
        dataset_ids=["d1"],
    )

    assert rankings[0]["candidate_id"] == "high"


def test_experiment_spec_resolves_scoring_profile(tmp_path):
    config_path = tmp_path / "metrics_config.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "guide_focus",
                        "label": "Guide Focus",
                        "metric_weights": {"vollath_f4_guide": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    spec = SweepExperimentSpec.from_dict(
        {
            "datasets": [{"id": "d1", "views": str(tmp_path / "views.npy")}],
            "scoring": {
                "profile_name": "guide_focus",
                "profile_config_path": str(config_path),
            },
        }
    )

    assert spec.scoring.profile_name == "guide_focus"
    assert spec.scoring.metric_weights == {"vollath_f4_guide": 1.0}


def test_experiment_spec_rejects_profile_and_explicit_weights(tmp_path):
    with pytest.raises(ValueError, match="profile_name and metric_weights"):
        SweepExperimentSpec.from_dict(
            {
                "datasets": [{"id": "d1", "views": str(tmp_path / "views.npy")}],
                "scoring": {
                    "profile_name": "hybrid_default",
                    "metric_weights": {"laplacian_variance": 1.0},
                },
            }
        )


def test_select_stage_worker_count_respects_memory_cap():
    tasks = [
        {"estimated_peak_bytes": 200},
        {"estimated_peak_bytes": 200},
        {"estimated_peak_bytes": 200},
        {"estimated_peak_bytes": 200},
        {"estimated_peak_bytes": 200},
    ]

    selection = parameter_sweep._select_stage_worker_count(
        tasks=tasks,
        requested_workers=8,
        available_memory_bytes=1000,
    )

    assert selection["memory_cap_workers"] == 3
    assert selection["selected_workers"] == 3
    assert selection["requested_workers"] == 8
    assert selection["max_task_peak_bytes"] == 200


def test_evaluate_stage_degrades_worker_count_and_continues(monkeypatch, tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    datasets = [
        PreparedDataset(
            dataset_id="d1",
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=False,
            views_shape=tuple(int(dim) for dim in views.shape),
            views_dtype=str(views.dtype),
            views_itemsize=int(views.dtype.itemsize),
        ),
        PreparedDataset(
            dataset_id="d2",
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=False,
            views_shape=tuple(int(dim) for dim in views.shape),
            views_dtype=str(views.dtype),
            views_itemsize=int(views.dtype.itemsize),
        ),
    ]
    candidates = [
        SweepCandidate(
            strategy="pixel",
            params={"similarity_window_size": 1},
            config=RectificationRunConfig(strategy="pixel", similarity_window_size=1, max_displacement=0),
            candidate_id="c0",
        ),
        SweepCandidate(
            strategy="pixel",
            params={"similarity_window_size": 3},
            config=RectificationRunConfig(strategy="pixel", similarity_window_size=3, max_displacement=0),
            candidate_id="c1",
        ),
    ]

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 10**12)

    call_state = {"count": 0}

    def fake_execute_pool_batch(tasks, workers, on_result=None):
        call_state["count"] += 1
        assert workers in {2, 4}
        if call_state["count"] == 1:
            completed = [_build_eval_result(tasks[0], status="ok")]
            if on_result is not None:
                on_result(tasks[0], completed[0])
            pending = list(tasks[1:])
            return completed, pending, BrokenProcessPool("pool crashed")
        completed = [_build_eval_result(task, status="ok") for task in tasks]
        if on_result is not None:
            for task, result in zip(tasks, completed):
                on_result(task, result)
        return completed, [], None

    monkeypatch.setattr(parameter_sweep, "_execute_pool_batch", fake_execute_pool_batch)
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: (_build_eval_result(task, status="ok"), False),
    )

    evaluation = parameter_sweep._evaluate_stage(
        stage_name="stage1",
        candidates=candidates,
        datasets=datasets,
        output_dir=tmp_path,
        save_figures_flag=False,
        requested_workers=4,
    )

    assert len(evaluation.stage_results) == 4
    assert evaluation.execution["pool_crash_count"] == 1
    assert evaluation.execution["degrade_events"][0]["workers_before"] == 4
    assert evaluation.execution["degrade_events"][0]["workers_after"] == 2
    assert call_state["count"] == 2


def test_evaluate_stage_persists_pooled_results_before_batch_retry(monkeypatch, tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    dataset = PreparedDataset(
        dataset_id="d1",
        views_path=views_path,
        hdr_enabled=False,
        hdr_exposure_times=None,
        normalize_to_8bit=False,
        views_shape=tuple(int(dim) for dim in views.shape),
        views_dtype=str(views.dtype),
        views_itemsize=int(views.dtype.itemsize),
    )
    candidates = [
        SweepCandidate(
            strategy="pixel",
            params={"similarity_window_size": 1},
            config=RectificationRunConfig(strategy="pixel", similarity_window_size=1, max_displacement=0),
            candidate_id=f"c{idx}",
        )
        for idx in range(2)
    ]
    run_state = {"runs": {}, "stages": {}}
    run_state_path = tmp_path / "run_state.json"
    persisted: dict[str, object] = {}
    call_state = {"count": 0}

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 10**12)

    def fake_execute_pool_batch(tasks, workers, on_result=None):
        call_state["count"] += 1
        if call_state["count"] == 1:
            completed = [_build_eval_result(tasks[0], status="ok")]
            if on_result is not None:
                on_result(tasks[0], completed[0])
            state = json.loads(run_state_path.read_text(encoding="utf-8"))
            entry = state["runs"][tasks[0]["run_uid"]]
            persisted["status"] = entry["status"]
            persisted["attempts"] = len(entry["attempts"])
            persisted["completed_runs"] = state["stages"]["stage1"]["completed_runs"]
            return completed, list(tasks[1:]), BrokenProcessPool("pool crashed")
        completed = [_build_eval_result(task, status="ok") for task in tasks]
        if on_result is not None:
            for task, result in zip(tasks, completed):
                on_result(task, result)
        return completed, [], None

    monkeypatch.setattr(parameter_sweep, "_execute_pool_batch", fake_execute_pool_batch)
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: (_build_eval_result(task, status="ok"), False),
    )

    evaluation = parameter_sweep._evaluate_stage(
        stage_name="stage1",
        candidates=candidates,
        datasets=[dataset],
        output_dir=tmp_path,
        save_figures_flag=False,
        requested_workers=2,
        run_state=run_state,
        run_state_path=run_state_path,
    )

    assert len(evaluation.stage_results) == 2
    assert persisted == {"status": "ok", "attempts": 1, "completed_runs": 1}


def test_evaluate_stage_surfaces_pooled_persistence_failure_without_pool_retry(monkeypatch, tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    datasets = [
        PreparedDataset(
            dataset_id="d1",
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=False,
            views_shape=tuple(int(dim) for dim in views.shape),
            views_dtype=str(views.dtype),
            views_itemsize=int(views.dtype.itemsize),
        ),
        PreparedDataset(
            dataset_id="d2",
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=False,
            views_shape=tuple(int(dim) for dim in views.shape),
            views_dtype=str(views.dtype),
            views_itemsize=int(views.dtype.itemsize),
        ),
    ]
    candidates = [
        SweepCandidate(
            strategy="pixel",
            params={"similarity_window_size": 1},
            config=RectificationRunConfig(strategy="pixel", similarity_window_size=1, max_displacement=0),
            candidate_id="c0",
        )
    ]
    run_state = {"runs": {}, "stages": {}}
    run_state_path = tmp_path / "run_state.json"
    executor_state = {"instances": 0, "submit_calls": 0}
    original_persist_run_state = parameter_sweep._persist_run_state

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 10**12)

    class FakeFuture:
        def __init__(self, task):
            self.task = task

        def result(self):
            return _build_eval_result(self.task, status="ok")

    class FakeExecutor:
        def __init__(self, max_workers, initializer=None):
            executor_state["instances"] += 1
            self.max_workers = max_workers
            self.initializer = initializer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, task):
            executor_state["submit_calls"] += 1
            return FakeFuture(task)

    def fail_on_result_persist(state_path, state):
        stage_state = state.get("stages", {}).get("stage1", {})
        if int(stage_state.get("completed_runs", 0)) > 0:
            raise RuntimeError("persist exploded")
        original_persist_run_state(state_path, state)

    monkeypatch.setattr(parameter_sweep, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parameter_sweep, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(parameter_sweep, "_persist_run_state", fail_on_result_persist)
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("callback persistence failures must not trigger isolated retry"),
    )

    with pytest.raises(RuntimeError, match="persist exploded"):
        parameter_sweep._evaluate_stage(
            stage_name="stage1",
            candidates=candidates,
            datasets=datasets,
            output_dir=tmp_path,
            save_figures_flag=False,
            requested_workers=2,
            run_state=run_state,
            run_state_path=run_state_path,
        )

    assert executor_state["instances"] == 1
    assert executor_state["submit_calls"] == 2


def test_evaluate_stage_records_failed_isolated_worker(monkeypatch, tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    dataset = PreparedDataset(
        dataset_id="d1",
        views_path=views_path,
        hdr_enabled=False,
        hdr_exposure_times=None,
        normalize_to_8bit=False,
        views_shape=tuple(int(dim) for dim in views.shape),
        views_dtype=str(views.dtype),
        views_itemsize=int(views.dtype.itemsize),
    )
    candidate = SweepCandidate(
        strategy="pixel",
        params={"similarity_window_size": 1},
        config=RectificationRunConfig(strategy="pixel", similarity_window_size=1, max_displacement=0),
        candidate_id="c0",
    )

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 10**12)
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: (_build_eval_result(task, status="failed"), True),
    )

    evaluation = parameter_sweep._evaluate_stage(
        stage_name="stage1",
        candidates=[candidate],
        datasets=[dataset],
        output_dir=tmp_path,
        save_figures_flag=False,
        requested_workers=1,
    )

    assert len(evaluation.stage_results) == 1
    assert evaluation.stage_results[0]["status"] == "failed"
    assert evaluation.execution["isolated_crash_count"] == 1
    assert evaluation.execution["used_isolated_mode"] is True


def test_run_sweep_experiment_writes_stage_and_leaderboard_outputs(tmp_path):
    views_a = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_b = (np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4) + 1.0) * 1.1
    views_a_path = tmp_path / "views_a.npy"
    views_b_path = tmp_path / "views_b.npy"
    np.save(views_a_path, views_a)
    np.save(views_b_path, views_b)

    pixel_space = _pixel_space(
        similarity_window_sizes=[1],
        max_displacements=[0],
        gauss_sigmas=[1.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.05],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )
    spec = SweepExperimentSpec(
        datasets=[
            SweepDatasetSpec(dataset_id="d1", views_path=str(views_a_path)),
            SweepDatasetSpec(dataset_id="d2", views_path=str(views_b_path)),
        ],
        search_spaces={"pixel": pixel_space},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "exp"),
        save_figures=False,
        analyze=False,
    )

    output_dir = run_sweep_experiment(spec)

    assert (output_dir / "stage1_summary.json").exists()
    assert (output_dir / "stage2_summary.json").exists()
    assert (output_dir / "leaderboard.json").exists()
    assert (output_dir / "leaderboard.csv").exists()
    assert (output_dir / "best_configs.json").exists()
    assert not (output_dir / "metric_experiments").exists()
    stage1_summary = json.loads((output_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert "execution" in stage1_summary
    assert "selected_workers" in stage1_summary["execution"]

    stage1_candidate_id = str(stage1_summary["rankings"][0]["candidate_id"])
    assert stage1_candidate_id.startswith("s1c")
    stage1_run = output_dir / "runs" / "stage1" / stage1_candidate_id / "d1"
    assert (stage1_run / "workspace.npz").exists()
    assert (stage1_run / "rectified_views.npy").exists()
    assert (stage1_run / "avg_rectified.png").exists()
    stage1_config = json.loads((stage1_run / "config.json").read_text(encoding="utf-8"))
    assert stage1_config["displacement_metric_family"] == "intensity"
    assert stage1_config["displacement_alpha"] == 1.0
    assert stage1_config["metric_formulation_id"] == "intensity_default"

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    best = leaderboard[0]
    best_candidate_id = str(best["candidate_id"])
    best_stage = "stage2" if (output_dir / "runs" / "stage2" / best_candidate_id).exists() else "stage1"
    best_run_dir = output_dir / "runs" / best_stage / best_candidate_id / "d1"

    config_dict = json.loads((best_run_dir / "config.json").read_text(encoding="utf-8"))
    config = RectificationRunConfig(**config_dict)
    direct = rectify_views_from_config(np.load(views_a_path), config)
    stored = np.load(best_run_dir / "rectified_views.npy")
    np.testing.assert_allclose(stored, direct.rectified_views)


def test_run_sweep_experiment_stage2_uses_auto_fine_values(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    pixel_space = _pixel_space(
        similarity_window_sizes=[15, 25],
        max_displacements=[0],
        gauss_sigmas=[1.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.05],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )
    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": pixel_space},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=4, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "auto_fine_exp"),
        save_figures=False,
        analyze=False,
    )

    output_dir = run_sweep_experiment(spec)
    stage2_summary = json.loads((output_dir / "stage2_summary.json").read_text(encoding="utf-8"))
    stage2_windows = {
        int(record["config"]["similarity_window_size"])
        for record in stage2_summary["runs"]
        if record["status"] == "ok"
    }

    assert any(window not in {15, 25} for window in stage2_windows)


def test_run_sweep_backfills_missing_scoring_metrics_for_reused_artifacts(monkeypatch, tmp_path):
    calls = _install_structured_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1, 3], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(mode="single_stage", total_budget_per_dataset=2, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "backfill_reuse"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(base_spec)
    _remove_metric_from_run_records(out_dir, "gini_avg")

    calls.clear()
    updated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces=base_spec.search_spaces,
        strategies=["pixel"],
        stages=base_spec.stages,
        scoring=SweepScoringSpec(metric_weights={"gini_avg": 1.0}),
        output_dir=str(out_dir),
        save_figures=False,
        analyze=False,
    )
    run_sweep_experiment(updated_spec, output_dir=out_dir, extend_existing=True)

    assert calls == []
    summary = json.loads((out_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert summary["execution"]["metric_backfilled_count"] == 2
    assert summary["execution"]["metric_backfill_failed_count"] == 0
    for path in sorted(out_dir.glob("runs/stage1/*/d1/metric_record.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "gini_avg" in payload["global_and_local_metrics"]
        assert payload["metric_backfilled"] is True
        assert "gini_avg" in payload["metric_backfilled_names"]


def test_run_sweep_backfills_missing_tamura_metric_for_reused_artifacts(monkeypatch, tmp_path):
    calls = _install_structured_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1, 3], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(mode="single_stage", total_budget_per_dataset=2, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "tamura_backfill_reuse"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(base_spec)
    _remove_metric_from_run_records(out_dir, "tamura_gradient_avg")

    calls.clear()
    updated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces=base_spec.search_spaces,
        strategies=["pixel"],
        stages=base_spec.stages,
        scoring=SweepScoringSpec(metric_weights={"tamura_gradient_avg": 1.0}),
        output_dir=str(out_dir),
        save_figures=False,
        analyze=False,
    )
    run_sweep_experiment(updated_spec, output_dir=out_dir, extend_existing=True)

    assert calls == []
    summary = json.loads((out_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert summary["execution"]["metric_backfilled_count"] == 2
    assert summary["execution"]["metric_backfill_failed_count"] == 0
    for path in sorted(out_dir.glob("runs/stage1/*/d1/metric_record.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "tamura_gradient_avg" in payload["global_and_local_metrics"]
        assert payload["metric_backfilled"] is True
        assert "tamura_gradient_avg" in payload["metric_backfilled_names"]


def test_run_sweep_dry_run_reports_metric_backfill_without_writing(monkeypatch, tmp_path):
    _install_structured_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(mode="single_stage", total_budget_per_dataset=1, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "backfill_dry_run"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(base_spec)
    _remove_metric_from_run_records(out_dir, "gini_avg")
    metric_record_path = next(out_dir.glob("runs/stage1/*/d1/metric_record.json"))
    before = metric_record_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    dry_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces=base_spec.search_spaces,
        strategies=["pixel"],
        stages=base_spec.stages,
        scoring=SweepScoringSpec(metric_weights={"gini_avg": 1.0}),
        output_dir=str(out_dir),
        save_figures=False,
        analyze=False,
    )
    run_sweep_experiment(dry_spec, output_dir=out_dir, extend_existing=True, dry_run=True)

    dry_run_path = next((out_dir / "generations").glob("*/dry_run.json"))
    dry_run_payload = json.loads(dry_run_path.read_text(encoding="utf-8"))
    assert dry_run_payload["stages"]["stage1"]["metric_backfilled_count"] == 1
    assert metric_record_path.read_text(encoding="utf-8") == before


def test_run_sweep_backfilled_metric_guides_stage2_seed(monkeypatch, tmp_path):
    calls = _install_structured_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    single_stage_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1, 3], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(mode="single_stage", total_budget_per_dataset=2, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "backfill_stage2_seed"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(single_stage_spec)
    _remove_metric_from_run_records(out_dir, "gini_avg")

    calls.clear()
    two_stage_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces=single_stage_spec.search_spaces,
        strategies=["pixel"],
        stages=SweepStageSpec(mode="two_stage", total_budget_per_dataset=4, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"gini_avg": 1.0}),
        output_dir=str(out_dir),
        save_figures=False,
        analyze=False,
    )
    run_sweep_experiment(two_stage_spec, output_dir=out_dir, extend_existing=True)

    assert not [call for call in calls if call["stage"] == "stage1"]
    stage1_summary = json.loads((out_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert stage1_summary["rankings"][0]["config"]["similarity_window_size"] == 3
    stage2_summary = json.loads((out_dir / "stage2_summary.json").read_text(encoding="utf-8"))
    stage2_windows = {
        int(record["config"]["similarity_window_size"])
        for record in stage2_summary["runs"]
        if record["status"] == "ok"
    }
    assert stage2_windows == {3, 5}


def test_run_sweep_recomputes_when_missing_metric_workspace_is_corrupt(monkeypatch, tmp_path):
    calls = _install_structured_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(mode="single_stage", total_budget_per_dataset=1, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "backfill_corrupt_workspace"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)
    _remove_metric_from_run_records(out_dir, "gini_avg")
    workspace_path = next(out_dir.glob("runs/stage1/*/d1/workspace.npz"))
    workspace_path.write_text("corrupted", encoding="utf-8")

    calls.clear()
    updated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces=spec.search_spaces,
        strategies=["pixel"],
        stages=spec.stages,
        scoring=SweepScoringSpec(metric_weights={"gini_avg": 1.0}),
        output_dir=str(out_dir),
        save_figures=False,
        analyze=False,
    )
    run_sweep_experiment(updated_spec, output_dir=out_dir, extend_existing=True)

    assert calls
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert "recomputed" in {entry["status"] for entry in state["runs"].values()}


def test_parallel_and_serial_runs_have_identical_leaderboard_order(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    pixel_space = _pixel_space(
        similarity_window_sizes=[1, 3],
        max_displacements=[0],
        gauss_sigmas=[1.0],
        disp_gauss_sigmas=[0.0],
        pixel_thresholds=[0.05],
        sub_pix_factors=[1],
        pixel_sigma_pyramid=[False],
        zero_fill_roll=[True],
    )

    spec_serial = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": pixel_space},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=4, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "serial"),
        save_figures=False,
        analyze=False,
    )
    spec_parallel = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": pixel_space},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=4, stage1_fraction=0.5, top_k=1, max_workers=2),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "parallel"),
        save_figures=False,
        analyze=False,
    )

    out_serial = run_sweep_experiment(spec_serial)
    out_parallel = run_sweep_experiment(spec_parallel)

    leaderboard_serial = json.loads((out_serial / "leaderboard.json").read_text(encoding="utf-8"))
    leaderboard_parallel = json.loads((out_parallel / "leaderboard.json").read_text(encoding="utf-8"))

    serial_configs = [row["config"] for row in leaderboard_serial]
    parallel_configs = [row["config"] for row in leaderboard_parallel]
    assert serial_configs == parallel_configs


def test_metric_formulations_expand_candidates_with_alpha_dimension():
    base_candidates = generate_coarse_candidates(
        _pixel_space(
            similarity_window_sizes=[1],
            max_displacements=[0],
            gauss_sigmas=[1.0],
            disp_gauss_sigmas=[0.0],
            pixel_thresholds=[0.05],
            sub_pix_factors=[1],
            pixel_sigma_pyramid=[False],
            zero_fill_roll=[True],
        )
    )
    formulations = [
        MetricFormulationSpec(formulation_id="f_grad", family="ncc_gradient", alphas=[0.0, 0.5, 1.0]),
        MetricFormulationSpec(formulation_id="f_lap", family="ncc_laplacian", alphas=[0.5]),
    ]
    expanded = parameter_sweep._expand_candidates_with_metric_formulations(base_candidates, formulations)

    assert len(expanded) == 4
    families = {candidate.config.displacement_metric_family for candidate in expanded}
    alphas = {candidate.config.displacement_alpha for candidate in expanded}
    assert families == {"blurred_gradient_corr", "blurred_laplacian_corr"}
    assert alphas == {0.0, 0.5, 1.0}


def test_metric_formulation_accepts_intensity_baseline_only():
    formulation = MetricFormulationSpec(
        formulation_id="intensity_default",
        family="intensity",
        alphas=[1.0],
    )

    assert formulation.family == "intensity"
    assert formulation.alphas == [1.0]


def test_metric_formulation_rejects_intensity_alpha_or_weights():
    with pytest.raises(ValueError, match="alpha=1.0"):
        MetricFormulationSpec(
            formulation_id="bad_intensity_alpha",
            family="intensity",
            alphas=[0.5],
        )

    with pytest.raises(ValueError, match="component_weights"):
        MetricFormulationSpec(
            formulation_id="bad_intensity_weights",
            family="intensity",
            alphas=[1.0],
            component_weights={"gradient": 1.0},
        )


def test_combined_search_space_propagates_reference_mode_into_candidates():
    candidates = generate_coarse_candidates(
        SweepSearchSpace(
            strategy="combined",
            refinement_window_sizes=[1],
            max_displacements=[0],
            gauss_sigmas=[1.0],
            disp_gauss_sigmas=[0.0],
            global_thresholds=[0.1],
            pixel_thresholds=[0.05],
            correlation_pad_widths=[3],
            sub_pix_factors=[1],
            pixel_sigma_pyramid=[False],
            zero_fill_roll=[True],
            combined_reference_mode="iterative_subtractive_max_rebuild",
        )
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.config.combined_reference_mode == "iterative_subtractive_max_rebuild"
    assert candidate.config.combined_use_global_avg_reference is False
    assert candidate.params["combined_reference_mode"] == "iterative_subtractive_max_rebuild"


def test_spec_from_dict_maps_legacy_metric_formulations_to_experimental_lane():
    payload = {
        "datasets": [{"id": "d1", "views": "x.npy"}],
        "strategies": ["pixel"],
        "search_spaces": {
            "pixel": {
                "similarity_window_sizes": [1],
                "max_displacements": [0],
                "gauss_sigmas": [1.0],
                "disp_gauss_sigmas": [0.0],
                "pixel_thresholds": [0.05],
                "sub_pix_factors": [1],
                "pixel_sigma_pyramid": [False],
                "zero_fill_roll": [True],
            }
        },
        "metric_formulations": [
            {"id": "legacy_grad", "family": "ncc_gradient", "alphas": [0.5]}
        ],
    }

    spec = SweepExperimentSpec.from_dict(payload)

    assert spec.experimental_metric_sweep is not None
    assert spec.experimental_metric_sweep.enabled is True
    assert spec.experimental_metric_sweep.candidate_source == "baseline_shortlist"
    assert len(spec.experimental_metric_sweep.formulations) == 1
    assert spec.experimental_metric_sweep.formulations[0].formulation_id == "legacy_grad"
    assert spec.experimental_metric_sweep.formulations[0].family == "blurred_gradient_corr"


def test_spec_from_dict_supports_independent_metric_search_stages(tmp_path):
    views_path = tmp_path / "views.npy"
    np.save(views_path, np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4))
    payload = {
        "datasets": [{"id": "d1", "views": str(views_path)}],
        "strategies": ["pixel"],
        "search_spaces": {
            "pixel": {
                "similarity_window_sizes": [1, 3],
                "max_displacements": [0],
                "gauss_sigmas": [1.0],
                "disp_gauss_sigmas": [0.0],
                "pixel_thresholds": [0.05],
                "sub_pix_factors": [1],
                "pixel_sigma_pyramid": [False],
                "zero_fill_roll": [True],
            }
        },
        "experimental_metric_sweep": {
            "enabled": True,
            "candidate_source": "independent_search",
            "shortlist_top_k": 1,
            "search_stages": {
                "mode": "single_stage",
                "total_budget_per_dataset": 2,
                "top_k": 1,
                "max_workers": 1,
            },
            "formulations": [
                {"id": "intensity_default", "family": "intensity", "alphas": [1.0]},
            ],
        },
    }

    spec = SweepExperimentSpec.from_dict(payload)

    assert spec.experimental_metric_sweep is not None
    assert spec.experimental_metric_sweep.candidate_source == "independent_search"
    assert spec.experimental_metric_sweep.search_stages is not None
    assert spec.experimental_metric_sweep.search_stages.mode == "single_stage"
    assert spec.experimental_metric_sweep.search_stages.total_budget_per_dataset == 2


def test_spec_from_dict_supports_single_stage_mode_and_defaults_to_two_stage():
    payload = {
        "datasets": [{"id": "d1", "views": "x.npy"}],
        "strategies": ["pixel"],
        "search_spaces": {
            "pixel": {
                "similarity_window_sizes": [1],
                "max_displacements": [0],
                "gauss_sigmas": [1.0],
                "disp_gauss_sigmas": [0.0],
                "pixel_thresholds": [0.05],
                "sub_pix_factors": [1],
                "pixel_sigma_pyramid": [False],
                "zero_fill_roll": [True],
            }
        },
        "stages": {
            "mode": "single_stage",
            "total_budget_per_dataset": 3,
        },
    }

    single_stage_spec = SweepExperimentSpec.from_dict(payload)
    default_stage_spec = SweepExperimentSpec.from_dict(
        {
            "datasets": payload["datasets"],
            "strategies": payload["strategies"],
            "search_spaces": payload["search_spaces"],
        }
    )

    assert single_stage_spec.stages.mode == "single_stage"
    assert default_stage_spec.stages.mode == "two_stage"


def test_experimental_metric_sweep_rejects_global_only_strategies(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    with pytest.raises(ValueError, match="pixel/combined"):
        SweepExperimentSpec(
            datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
            search_spaces={
                "global": SweepSearchSpace(
                    strategy="global",
                    correlation_pad_widths=[54],
                    global_thresholds=[0.15],
                )
            },
            strategies=["global"],
            experimental_metric_sweep=ExperimentalMetricSweepSpec(
                enabled=True,
                shortlist_top_k=1,
                formulations=[
                    MetricFormulationSpec(
                        formulation_id="exp_grad",
                        family="ncc_gradient",
                        alphas=[0.5],
                    )
                ],
            ),
        )


def test_run_sweep_experiment_writes_run_state_without_root_alpha_report(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "state_alpha"),
        save_figures=False,
        analyze=True,
    )

    out_dir = run_sweep_experiment(spec)
    assert (out_dir / "run_state.json").exists()
    assert not (out_dir / "alpha_report.json").exists()
    assert not (out_dir / "alpha_report.csv").exists()
    assert (out_dir / "experiment_manifest.json").exists()
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline_displacement_mode"]["family"] == "intensity"
    assert manifest["baseline_displacement_mode"]["alpha"] == 1.0


def test_run_sweep_experiment_single_stage_writes_stage1_only_outputs(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 3],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=2,
            stage1_fraction=0.5,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "single_stage"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)

    assert (out_dir / "stage1_summary.json").exists()
    assert not (out_dir / "stage2_summary.json").exists()
    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "best_configs.json").exists()
    assert not (out_dir / "runs" / "stage2").exists()
    stage1_summary = json.loads((out_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    stage1_candidate_id = str(stage1_summary["rankings"][0]["candidate_id"])
    assert stage1_candidate_id.startswith("s1c")
    assert (out_dir / "runs" / "stage1" / stage1_candidate_id / "d1" / "workspace.npz").exists()

    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    run_state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["mode"] == "single_stage"
    assert run_state["fingerprint_payload"]["spec"]["stages"]["mode"] == "single_stage"
    assert "stage2" not in run_state["stages"]


def test_run_sweep_experiment_single_stage_still_supports_metric_phase(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            stage1_fraction=0.5,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.25, 0.75],
                )
            ],
        ),
        output_dir=str(tmp_path / "single_stage_metric_phase"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)
    metric_dir = out_dir / "metric_experiments"

    assert (out_dir / "stage1_summary.json").exists()
    assert not (out_dir / "stage2_summary.json").exists()
    assert (metric_dir / "metric_phase_summary.json").exists()
    config_paths = sorted(metric_dir.glob("runs/metric_phase/*/*/config.json"))
    assert config_paths
    formulation_ids = {
        json.loads(path.read_text(encoding="utf-8"))["metric_formulation_id"]
        for path in config_paths
    }
    assert formulation_ids == {"exp_grad"}
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline_displacement_mode"]["metric_formulation_id"] == "intensity_default"

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == 2
    assert state["active_generation_id"]
    assert state["spec_generations"]
    assert state["runs"]


def test_run_sweep_resume_skips_recompute_for_valid_artifacts(monkeypatch, tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "resume_ok"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)

    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: (_build_eval_result(task, status="failed"), True),
    )
    run_sweep_experiment(spec, output_dir=out_dir)

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    statuses = {entry["status"] for entry in state["runs"].values()}
    assert "skipped_valid" in statuses


def test_run_sweep_recomputes_when_metric_record_is_corrupted(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "resume_recompute"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)

    metric_record_paths = sorted(out_dir.glob("runs/*/*/*/metric_record.json"))
    assert metric_record_paths
    metric_record_paths[0].write_text("{corrupted", encoding="utf-8")

    run_sweep_experiment(spec, output_dir=out_dir)
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    statuses = {entry["status"] for entry in state["runs"].values()}
    assert "recomputed" in statuses


def test_run_sweep_blocks_mismatched_fingerprint(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "fingerprint"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(base_spec)

    changed_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[3], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "fingerprint"),
        save_figures=False,
        analyze=False,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        run_sweep_experiment(changed_spec, output_dir=out_dir)


def test_resume_fingerprint_accepts_legacy_metric_shortlist_defaults():
    old_payload = {
        "schema_version": 2,
        "output_dir": "C:/old/sweep",
        "spec": {
            "output_dir": "C:/old/sweep",
            "datasets": [],
            "stages": {"mode": "two_stage", "top_k": 1},
            "experimental_metric_sweep": {
                "enabled": True,
                "shortlist_top_k": 1,
                "formulations": [
                    {
                        "id": "legacy_grad",
                        "family": "blurred_gradient_corr",
                        "alphas": [0.5],
                    }
                ],
            },
        },
        "datasets": [],
    }
    new_payload = json.loads(json.dumps(old_payload))
    new_payload["output_dir"] = "C:/new/sweep"
    new_payload["spec"]["output_dir"] = "C:/new/sweep"
    new_payload["spec"]["experimental_metric_sweep"]["candidate_source"] = "baseline_shortlist"
    new_payload["spec"]["experimental_metric_sweep"]["search_stages"] = None

    assert parameter_sweep._fingerprint_payloads_resume_compatible(
        old_payload,
        new_payload,
    )

    independent_payload = json.loads(json.dumps(new_payload))
    independent_payload["spec"]["experimental_metric_sweep"][
        "candidate_source"
    ] = "independent_search"
    independent_payload["spec"]["experimental_metric_sweep"]["search_stages"] = {
        "mode": "single_stage",
        "total_budget_per_dataset": 1,
        "top_k": 1,
    }

    assert not parameter_sweep._fingerprint_payloads_resume_compatible(
        old_payload,
        independent_payload,
    )


def test_run_sweep_resume_adopts_relocated_run_state(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    old_views_dir = tmp_path / "old_input"
    old_views_dir.mkdir()
    old_views_path = old_views_dir / "views.npy"
    np.save(old_views_path, views)

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(old_views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "original_run"),
        save_figures=False,
        analyze=False,
    )
    original_out_dir = run_sweep_experiment(base_spec)

    relocated_output_dir = tmp_path / "relocated_run"
    shutil.copytree(original_out_dir, relocated_output_dir)

    new_views_dir = tmp_path / "new_input"
    new_views_dir.mkdir()
    new_views_path = new_views_dir / "views.npy"
    np.save(new_views_path, views)

    relocated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(new_views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(relocated_output_dir),
        save_figures=False,
        analyze=False,
    )

    relocation_rules = parameter_sweep._parse_path_relocation_rules(
        [f"{tmp_path / 'original_run'}={relocated_output_dir}"]
    )
    out_dir = run_sweep_experiment(
        relocated_spec,
        output_dir=relocated_output_dir,
        adopt_relocated=True,
        relocation_rules=relocation_rules,
    )

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["path_compatibility"]["output_dir"] == str(relocated_output_dir.resolve())
    assert state["relocation_history"]
    assert (
        state["experiment_fingerprint"]
        == parameter_sweep._compute_experiment_fingerprint(state["fingerprint_payload"])
    )
    statuses = {entry["status"] for entry in state["runs"].values()}
    assert "skipped_valid" in statuses


def test_run_sweep_dry_run_adopts_relocated_profile_config_path(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old_views_path = old_root / "views.npy"
    new_views_path = new_root / "views.npy"
    np.save(old_views_path, views)
    np.save(new_views_path, views)
    old_config_path = old_root / "metrics_config.json"
    new_config_path = new_root / "metrics_config.json"
    old_config_path.write_text("{}", encoding="utf-8")
    new_config_path.write_text("{}", encoding="utf-8")

    base_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(old_views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(
            metric_weights={"laplacian_variance": 1.0},
            profile_config_path=str(old_config_path),
        ),
        output_dir=str(old_root / "run"),
        save_figures=False,
        analyze=False,
    )
    original_out_dir = run_sweep_experiment(base_spec)
    relocated_output_dir = new_root / "run"
    shutil.copytree(original_out_dir, relocated_output_dir)

    relocated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(new_views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(
            metric_weights={"laplacian_variance": 1.0},
            profile_config_path=str(new_config_path),
        ),
        output_dir=str(relocated_output_dir),
        save_figures=False,
        analyze=False,
    )

    relocation_rules = parameter_sweep._parse_path_relocation_rules(
        [f"{old_root}={new_root}"]
    )
    out_dir = run_sweep_experiment(
        relocated_spec,
        output_dir=relocated_output_dir,
        adopt_relocated=True,
        relocation_rules=relocation_rules,
        dry_run=True,
    )

    assert out_dir == relocated_output_dir


def test_parse_experiment_spec_relocates_output_dataset_and_profile_config(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    new_root.mkdir()
    views_path = new_root / "views.npy"
    np.save(views_path, np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4))
    config_path = new_root / "metrics_config.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "guide_focus",
                        "metric_weights": {"vollath_f4_guide": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "output_dir": str(old_root / "sweep"),
                "datasets": [{"id": "d1", "views": str(old_root / "views.npy")}],
                "scoring": {
                    "profile_name": "guide_focus",
                    "profile_config_path": str(old_root / "metrics_config.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    rules = parameter_sweep._parse_path_relocation_rules([f"{old_root}={new_root}"])
    spec, report = parameter_sweep._parse_experiment_spec(spec_path, relocation_rules=rules)

    assert spec.output_dir == str((new_root / "sweep").resolve())
    assert spec.datasets[0].views_path == str(views_path.resolve())
    assert spec.scoring.profile_config_path == str(config_path.resolve())
    assert spec.scoring.metric_weights == {"vollath_f4_guide": 1.0}
    assert {item["field"] for item in report["applied"]} >= {
        "output_dir",
        "datasets[d1].views",
        "scoring.profile_config_path",
    }


def test_run_sweep_rejects_relocated_state_without_adopt_flag(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    old_views_path = tmp_path / "old_views.npy"
    new_views_path = tmp_path / "new_views.npy"
    np.save(old_views_path, views)
    np.save(new_views_path, views)
    original_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(old_views_path))],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "original"),
        save_figures=False,
        analyze=False,
    )
    original_dir = run_sweep_experiment(original_spec)
    relocated_dir = tmp_path / "relocated"
    shutil.copytree(original_dir, relocated_dir)
    relocated_spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(new_views_path))],
        search_spaces=original_spec.search_spaces,
        strategies=original_spec.strategies,
        stages=original_spec.stages,
        scoring=original_spec.scoring,
        output_dir=str(relocated_dir),
        save_figures=False,
        analyze=False,
    )

    with pytest.raises(ValueError, match="adopt-relocated"):
        run_sweep_experiment(relocated_spec, output_dir=relocated_dir)


def test_run_sweep_relocated_source_image_uses_matching_copied_cache(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    source_path = tmp_path / "old_source.png"
    source_path.write_bytes(b"placeholder")

    def fake_generate_views(*, cache_dir: Path, **_kwargs):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "generated_views.npy"
        np.save(cache_path, views)
        return views, cache_path

    monkeypatch.setattr(parameter_sweep, "_generate_views", fake_generate_views)
    base_spec = SweepExperimentSpec(
        datasets=[
            SweepDatasetSpec(
                dataset_id="d1",
                source_image=str(source_path),
                center_x=2,
                center_y=2,
                view_radius=1,
            )
        ],
        search_spaces={"pixel": _pixel_space(similarity_window_sizes=[1], max_displacements=[0], sub_pix_factors=[1], pixel_sigma_pyramid=[False])},
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / "source_original"),
        save_figures=False,
        analyze=False,
    )
    original_dir = run_sweep_experiment(base_spec)
    relocated_dir = tmp_path / "source_relocated"
    shutil.copytree(original_dir, relocated_dir)
    source_path.unlink()

    relocated_spec = SweepExperimentSpec(
        datasets=base_spec.datasets,
        search_spaces=base_spec.search_spaces,
        strategies=base_spec.strategies,
        stages=base_spec.stages,
        scoring=base_spec.scoring,
        output_dir=str(relocated_dir),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(
        relocated_spec,
        output_dir=relocated_dir,
        adopt_relocated=True,
    )

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert "skipped_valid" in {entry["status"] for entry in state["runs"].values()}


def test_run_sweep_adopt_relocated_rejects_incompatible_dataset(tmp_path):
    old_views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    new_views = np.arange(100, dtype=np.float32).reshape(2, 2, 5, 5)
    old_views_path = tmp_path / "old_views.npy"
    new_views_path = tmp_path / "new_views.npy"
    np.save(old_views_path, old_views)
    np.save(new_views_path, new_views)
    base_spec = _storage_spec(tmp_path, old_views_path, "incompatible_reloc_old", [1])
    original_dir = run_sweep_experiment(base_spec)
    relocated_dir = tmp_path / "incompatible_reloc_new"
    shutil.copytree(original_dir, relocated_dir)
    relocated_spec = _storage_spec(tmp_path, new_views_path, "incompatible_reloc_new", [1])

    with pytest.raises(ValueError, match="fingerprint"):
        run_sweep_experiment(
            relocated_spec,
            output_dir=relocated_dir,
            adopt_relocated=True,
        )


def test_run_from_namespace_relocation_dry_run_writes_preview_without_run_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    new_root.mkdir()
    views_path = new_root / "views.npy"
    np.save(views_path, np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4))
    spec_path = tmp_path / "reloc_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "output_dir": str(old_root / "sweep"),
                "strategies": ["pixel"],
                "datasets": [{"id": "d1", "views": str(old_root / "views.npy")}],
                "search_spaces": {
                    "pixel": {
                        "similarity_window_sizes": [1],
                        "max_displacements": [0],
                        "sub_pix_factors": [1],
                        "pixel_sigma_pyramid": [False],
                    }
                },
                "stages": {"total_budget_per_dataset": 2, "stage1_fraction": 0.5, "top_k": 1, "max_workers": 1},
                "scoring": {"metric_weights": {"laplacian_variance": 1.0}},
                "analyze": False,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        spec=str(spec_path),
        output_dir=None,
        workers=None,
        metrics_config=None,
        scoring_profile=None,
        extend_existing=False,
        merge_from=None,
        relocate_path=[f"{old_root}={new_root}"],
        adopt_relocated=False,
        dry_run=True,
    )

    out_dir = parameter_sweep._run_from_namespace(args)

    assert out_dir == (new_root / "sweep").resolve()
    dry_run_paths = list((out_dir / "generations").glob("*/dry_run.json"))
    assert len(dry_run_paths) == 1
    dry_run_payload = json.loads(dry_run_paths[0].read_text(encoding="utf-8"))
    assert dry_run_payload["relocation"]["applied"]
    assert not (out_dir / "run_state.json").exists()


def _storage_spec(tmp_path, views_path, output_name: str, windows: list[int]) -> SweepExperimentSpec:
    return SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=windows,
                max_displacements=[0],
                gauss_sigmas=[1.0],
                disp_gauss_sigmas=[0.0],
                pixel_thresholds=[0.05],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
                zero_fill_roll=[True],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            total_budget_per_dataset=max(2, len(windows) * 2),
            stage1_fraction=0.5,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        output_dir=str(tmp_path / output_name),
        save_figures=False,
        analyze=False,
    )


def test_run_sweep_extend_existing_creates_generation_products_and_reuses_runs(monkeypatch, tmp_path):
    calls = _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    base_spec = _storage_spec(tmp_path, views_path, "extendable", [1])
    out_dir = run_sweep_experiment(base_spec)
    base_state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    base_generation = base_state["active_generation_id"]
    base_stage1 = json.loads((out_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    base_candidate_id = str(base_stage1["rankings"][0]["candidate_id"])
    base_run_dir = out_dir / "runs" / "stage1" / base_candidate_id / "d1"
    assert base_run_dir.exists()

    calls.clear()
    extended_spec = _storage_spec(tmp_path, views_path, "extendable", [1, 3])
    run_sweep_experiment(extended_spec, output_dir=out_dir, extend_existing=True)

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == 2
    assert state["active_generation_id"] != base_generation
    assert len(state["spec_generations"]) >= 2
    assert base_run_dir.exists()
    assert "skipped_valid" in {entry["status"] for entry in state["runs"].values()}
    assert calls

    active_generation = state["active_generation_id"]
    products_dir = out_dir / "generations" / active_generation / "products"
    assert (products_dir / "stage1_summary.json").exists()
    assert (products_dir / "stage2_summary.json").exists()
    assert (products_dir / "leaderboard.json").exists()
    assert (products_dir / "experiment_manifest.json").exists()
    copied_run_dirs = sorted(products_dir.glob("runs/*/*/*"))
    assert copied_run_dirs
    assert any((run_dir / "avg_rectified.png").exists() for run_dir in copied_run_dirs)
    assert any((run_dir / "workspace.npz").exists() for run_dir in copied_run_dirs)
    assert any((run_dir / "metric_record.json").exists() for run_dir in copied_run_dirs)


def test_run_sweep_merge_from_copies_missing_and_skips_duplicates(monkeypatch, tmp_path):
    calls = _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    source_spec = _storage_spec(tmp_path, views_path, "merge_source", [1, 3])
    source_dir = run_sweep_experiment(source_spec)
    canonical_base = _storage_spec(tmp_path, views_path, "merge_canonical", [1])
    canonical_dir = run_sweep_experiment(canonical_base)

    calls.clear()
    canonical_extended = _storage_spec(tmp_path, views_path, "merge_canonical", [1, 3])
    run_sweep_experiment(
        canonical_extended,
        output_dir=canonical_dir,
        merge_from=[source_dir],
    )

    state = json.loads((canonical_dir / "run_state.json").read_text(encoding="utf-8"))
    merge_log = json.loads(
        (canonical_dir / "generations" / state["active_generation_id"] / "merge_log.json").read_text(
            encoding="utf-8"
        )
    )
    assert merge_log["copied"]
    assert merge_log["skipped_duplicates"]
    assert not calls
    for item in merge_log["copied"]:
        assert Path(item["destination_run_dir"]).exists()


def test_run_sweep_merge_rejects_incompatible_dataset(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    source_views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    current_views = np.arange(100, dtype=np.float32).reshape(2, 2, 5, 5)
    source_views_path = tmp_path / "source_views.npy"
    current_views_path = tmp_path / "current_views.npy"
    np.save(source_views_path, source_views)
    np.save(current_views_path, current_views)

    source_dir = run_sweep_experiment(
        _storage_spec(tmp_path, source_views_path, "incompatible_source", [1])
    )
    current_spec = _storage_spec(tmp_path, current_views_path, "incompatible_current", [1])
    current_dir = run_sweep_experiment(current_spec, merge_from=[source_dir])

    state = json.loads((current_dir / "run_state.json").read_text(encoding="utf-8"))
    merge_log = json.loads(
        (current_dir / "generations" / state["active_generation_id"] / "merge_log.json").read_text(
            encoding="utf-8"
        )
    )
    assert not merge_log["copied"]
    assert merge_log["rejected"]
    assert (current_dir / "leaderboard.json").exists()


def test_run_sweep_dry_run_writes_only_generation_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _storage_spec(tmp_path, views_path, "dry_run_root", [1])
    out_dir = run_sweep_experiment(spec, dry_run=True)

    dry_run_paths = list((out_dir / "generations").glob("*/dry_run.json"))
    assert len(dry_run_paths) == 1
    dry_run_payload = json.loads(dry_run_paths[0].read_text(encoding="utf-8"))
    assert dry_run_payload["stages"]["stage1"]["dry_run"] is True
    assert dry_run_payload["stages"]["stage1"]["pending_run_count"] == 1
    assert not (out_dir / "run_state.json").exists()
    assert not (out_dir / "stage1_summary.json").exists()
    assert not (out_dir / "runs").exists()


def test_run_sweep_experiment_with_metric_phase_outputs(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.25, 0.75],
                ),
                MetricFormulationSpec(
                    formulation_id="exp_lap",
                    family="ncc_laplacian",
                    alphas=[0.5],
                ),
            ],
        ),
        output_dir=str(tmp_path / "metric_phase"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)
    metric_dir = out_dir / "metric_experiments"
    assert metric_dir.exists()
    assert not (out_dir / "alpha_report.json").exists()
    assert (metric_dir / "metric_phase_summary.json").exists()
    assert (metric_dir / "leaderboard.json").exists()
    assert (metric_dir / "summary.json").exists()
    assert (metric_dir / "alpha_report.json").exists()

    config_paths = sorted(metric_dir.glob("runs/metric_phase/*/*/config.json"))
    assert config_paths
    config_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in config_paths
    ]
    formulation_ids = {payload["metric_formulation_id"] for payload in config_payloads}
    assert {"exp_grad", "exp_lap"}.issubset(formulation_ids)
    assert {
        payload["displacement_metric_family"] for payload in config_payloads
    } == {"blurred_gradient_corr", "blurred_laplacian_corr"}


def _write_fake_trio_csv(
    output_dir: Path,
    *,
    dataset_id: str,
    stage: str,
    generation_id: str | None,
    rankings: list[dict],
    rankable: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "source_kind": "thorcam_baseline",
            "dataset_id": dataset_id,
            "row_id": "thorcam",
            "stage": "thorcam",
            "candidate_id": "thorcam",
            "generation_id": generation_id or "",
            "validation_rank": "",
            "validation_rankable": "true",
            "validation_hybrid_thorcam_fov_v1": 0.35,
        }
    ]
    sorted_rankings = sorted(
        rankings,
        key=lambda item: float(item["oracle_score"]),
        reverse=True,
    )
    for rank, item in enumerate(sorted_rankings, start=1):
        rows.append(
            {
                "source_kind": "lf_candidate",
                "dataset_id": dataset_id,
                "row_id": f"{stage}:{item['candidate_id']}",
                "stage": stage,
                "candidate_id": item["candidate_id"],
                "generation_id": generation_id or "",
                "validation_rank": rank if rankable else "",
                "validation_rankable": str(bool(rankable)).lower(),
                "ranking_exclusion_reason": "" if rankable else "registration_underconstrained",
                "validation_hybrid_thorcam_fov_v1": item["oracle_score"],
            }
        )
    with (output_dir / "trio_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_kind",
                "dataset_id",
                "row_id",
                "stage",
                "candidate_id",
                "generation_id",
                "validation_rank",
                "validation_rankable",
                "ranking_exclusion_reason",
                "validation_hybrid_thorcam_fov_v1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _install_fake_trio_runner(monkeypatch, *, rankable: bool = True):
    calls: list[dict[str, object]] = []

    def fake_run_benchmark(config):
        stage = str(config.stage)
        dataset_id = str(config.dataset_id)
        summary = json.loads(
            (Path(config.sweep_dir) / f"{stage}_summary.json").read_text(encoding="utf-8")
        )
        scored_rankings: list[dict] = []
        for ranking in summary["rankings"]:
            window = int(ranking["params"]["similarity_window_size"])
            if stage == "stage1":
                score = {1: 0.95, 5: 0.45, 9: 0.10}.get(window, 0.05)
            else:
                score = 1.0 - abs(window - 1) / 10.0
            scored = dict(ranking)
            scored["oracle_score"] = score
            scored_rankings.append(scored)
        _write_fake_trio_csv(
            Path(config.output_dir),
            dataset_id=dataset_id,
            stage=stage,
            generation_id=str(summary.get("generation_id") or ""),
            rankings=scored_rankings,
            rankable=rankable,
        )
        calls.append(
            {
                "stage": stage,
                "dataset_id": dataset_id,
                "output_dir": str(config.output_dir),
                "rankable": rankable,
                "generation_scope": str(config.generation_scope),
                "validation_workspace": (
                    None
                    if getattr(config, "validation_workspace", None) is None
                    else str(config.validation_workspace)
                ),
                "validation_image": (
                    None
                    if getattr(config, "validation_image", None) is None
                    else str(config.validation_image)
                ),
                "thorcam_image": (
                    None
                    if getattr(config, "thorcam_image", None) is None
                    else str(config.thorcam_image)
                ),
                "use_validation_workspace_transforms": bool(
                    getattr(config, "use_validation_workspace_transforms", False)
                ),
            }
        )
        return SimpleNamespace(best_lf_row_id=None, rows=[], summary={})

    monkeypatch.setattr(parameter_sweep, "run_benchmark", fake_run_benchmark)
    return calls


def _validation_oracle_spec(
    tmp_path,
    views_path,
    *,
    output_name: str = "validation_oracle",
    rank_top_k: int = 1,
) -> SweepExperimentSpec:
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    validation_path.write_bytes(b"validation")
    thorcam_path.write_bytes(b"thorcam")
    return SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 5, 9],
                max_displacements=[0],
                gauss_sigmas=[1.0],
                disp_gauss_sigmas=[0.0],
                pixel_thresholds=[0.05],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
                zero_fill_roll=[True],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            total_budget_per_dataset=5,
            stage1_fraction=0.6,
            top_k=1,
            max_workers=1,
            refinement=SweepStageRefinementSpec(method="neighbor"),
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        validation_oracle_sweep=parameter_sweep.ValidationOracleSweepSpec(
            enabled=True,
            top_k=rank_top_k,
            budget_per_dataset=2,
            scenes=[
                {
                    "dataset_id": "d1",
                    "scene_id": "scene_d1",
                    "validation_image": str(validation_path),
                    "thorcam_image": str(thorcam_path),
                }
            ],
        ),
        output_dir=str(tmp_path / output_name),
        save_figures=False,
        analyze=False,
    )


def test_validation_oracle_sweep_records_missing_scene_failure(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    monkeypatch.setattr(
        parameter_sweep,
        "run_benchmark",
        lambda config: pytest.fail("invalid oracle scenes must fail before benchmarking"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(tmp_path, views_path)
    spec.validation_oracle_sweep.scenes = []

    out_dir = run_sweep_experiment(spec)

    assert (out_dir / "stage2_summary.json").exists()
    assert (out_dir / "leaderboard.json").exists()
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_oracle_sweep"]["status"] == "failed"
    assert manifest["validation_oracle_sweep"]["failed_stage"] == "scene_validation"
    assert "d1" in manifest["validation_oracle_sweep"]["error"]


def test_validation_oracle_sweep_accepts_workspace_only_scene(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    trio_calls = _install_fake_trio_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)
    validation_workspace = tmp_path / "validation_workspace.npz"
    validation_workspace.write_bytes(b"workspace")

    spec = _validation_oracle_spec(
        tmp_path,
        views_path,
        output_name="validation_workspace_only",
    )
    spec.validation_oracle_sweep.scenes = [
        {
            "dataset_id": "d1",
            "scene_id": "scene_d1",
            "validation_workspace": str(validation_workspace),
        }
    ]

    out_dir = run_sweep_experiment(spec)

    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_oracle_sweep"]["status"] == "completed"
    assert [call["validation_workspace"] for call in trio_calls] == [
        str(validation_workspace),
        str(validation_workspace),
    ]
    assert {call["validation_image"] for call in trio_calls} == {None}
    assert {call["thorcam_image"] for call in trio_calls} == {None}
    assert {call["use_validation_workspace_transforms"] for call in trio_calls} == {True}


def test_validation_oracle_sweep_reports_missing_workspace_file(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    monkeypatch.setattr(
        parameter_sweep,
        "run_benchmark",
        lambda config: pytest.fail("invalid workspace scene must fail before benchmarking"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)
    missing_workspace = tmp_path / "missing_validation_workspace.npz"

    spec = _validation_oracle_spec(
        tmp_path,
        views_path,
        output_name="validation_missing_workspace",
    )
    spec.validation_oracle_sweep.scenes = [
        {
            "dataset_id": "d1",
            "validation_workspace": str(missing_workspace),
            "use_validation_workspace_transforms": True,
        }
    ]

    out_dir = run_sweep_experiment(spec)

    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    oracle_manifest = manifest["validation_oracle_sweep"]
    assert oracle_manifest["status"] == "failed"
    assert oracle_manifest["failed_stage"] == "scene_validation"
    assert "d1" in oracle_manifest["error"]
    assert str(missing_workspace) in oracle_manifest["error"]


def test_validation_oracle_sweep_dry_run_reports_plan_without_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    monkeypatch.setattr(
        parameter_sweep,
        "run_benchmark",
        lambda config: pytest.fail("dry run must not run trio benchmarks"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(tmp_path, views_path, output_name="validation_dry_run")
    out_dir = run_sweep_experiment(spec, dry_run=True)

    dry_run_path = next((out_dir / "generations").glob("*/dry_run.json"))
    dry_run_payload = json.loads(dry_run_path.read_text(encoding="utf-8"))
    validation_plan = dry_run_payload["validation_oracle_sweep"]
    assert validation_plan["enabled"] is True
    assert validation_plan["status"] == "planned_after_stage1_oracle_ranking"
    assert validation_plan["num_scenes"] == 1
    assert validation_plan["num_stage1_benchmarks"] == 1
    assert validation_plan["estimated_validation_phase_runs"] == 2
    assert not (out_dir / "validation_oracle").exists()
    assert not (out_dir / "validation_phase_summary.json").exists()


def test_validation_oracle_sweep_uses_oracle_rankings_for_validation_phase(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    trio_calls = _install_fake_trio_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(tmp_path, views_path)
    out_dir = run_sweep_experiment(spec)

    stage2_summary = json.loads((out_dir / "stage2_summary.json").read_text(encoding="utf-8"))
    validation_summary = json.loads(
        (out_dir / "validation_phase_summary.json").read_text(encoding="utf-8")
    )
    stage2_windows = {
        int(row["params"]["similarity_window_size"]) for row in stage2_summary["runs"]
    }
    validation_windows = {
        int(row["params"]["similarity_window_size"]) for row in validation_summary["runs"]
    }

    assert stage2_windows == {5, 9}
    assert validation_windows == {1, 5}
    assert validation_summary["stage"] == "validation_phase"
    assert validation_summary["rankings"][0]["score"] > validation_summary["rankings"][1]["score"]
    assert (out_dir / "validation_leaderboard.json").exists()
    assert (out_dir / "validation_leaderboard.csv").exists()
    assert (out_dir / "validation_best_configs.json").exists()
    assert (out_dir / "metric_review" / "trio_oracle_metrics.json").exists()
    cache = json.loads(
        (out_dir / "metric_review" / "trio_oracle_metrics.json").read_text(encoding="utf-8")
    )
    assert {record["stage"] for record in cache["records"]} == {"stage1", "validation_phase"}
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_oracle_sweep"]["status"] == "completed"
    assert manifest["validation_oracle_sweep"]["num_ranked"] == 2
    assert [call["stage"] for call in trio_calls] == ["stage1", "validation_phase"]
    assert {call["generation_scope"] for call in trio_calls} == {"Active"}


def test_validation_oracle_sweep_honors_explicit_generation_scope(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    trio_calls = _install_fake_trio_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(
        tmp_path,
        views_path,
        output_name="validation_generation_scope",
    )
    spec.validation_oracle_sweep.defaults["generation_scope"] = "gen_manual"

    run_sweep_experiment(spec)

    assert [call["generation_scope"] for call in trio_calls] == ["gen_manual", "gen_manual"]


def test_validation_oracle_sweep_fails_when_trio_rows_are_unrankable(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    _install_fake_trio_runner(monkeypatch, rankable=False)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(tmp_path, views_path, output_name="validation_unrankable")

    out_dir = run_sweep_experiment(spec)

    assert (out_dir / "stage2_summary.json").exists()
    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "summary.json").exists()
    assert not (out_dir / "validation_phase_summary.json").exists()
    assert not Path(f"{out_dir / 'validation_phase_summary.json'}.bak").exists()
    assert not (out_dir / "validation_leaderboard.json").exists()
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    oracle_manifest = manifest["validation_oracle_sweep"]
    assert oracle_manifest["status"] == "failed"
    assert oracle_manifest["failed_stage"] == "stage1_ranking"
    assert "rankable" in oracle_manifest["error"]
    side_manifest = json.loads(
        (out_dir / "validation_oracle" / "validation_oracle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert side_manifest["status"] == "failed"


def test_validation_oracle_failure_after_placeholder_cleans_validation_summary(monkeypatch, tmp_path):
    _install_window_score_artifact_runner(monkeypatch)
    calls: list[str] = []

    def fake_run_benchmark(config):
        stage = str(config.stage)
        calls.append(stage)
        if stage == "validation_phase":
            raise RuntimeError("validation benchmark exploded")
        summary = json.loads(
            (Path(config.sweep_dir) / f"{stage}_summary.json").read_text(encoding="utf-8")
        )
        scored_rankings = []
        for ranking in summary["rankings"]:
            scored = dict(ranking)
            scored["oracle_score"] = 1.0
            scored_rankings.append(scored)
        _write_fake_trio_csv(
            Path(config.output_dir),
            dataset_id=str(config.dataset_id),
            stage=stage,
            generation_id=str(summary.get("generation_id") or ""),
            rankings=scored_rankings,
            rankable=True,
        )
        return SimpleNamespace(best_lf_row_id=None, rows=[], summary={})

    monkeypatch.setattr(parameter_sweep, "run_benchmark", fake_run_benchmark)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(
        tmp_path,
        views_path,
        output_name="validation_placeholder_failure",
    )
    out_dir = run_sweep_experiment(spec)

    assert calls == ["stage1", "validation_phase"]
    assert (out_dir / "stage2_summary.json").exists()
    assert (out_dir / "leaderboard.json").exists()
    assert not (out_dir / "validation_phase_summary.json").exists()
    assert not Path(f"{out_dir / 'validation_phase_summary.json'}.bak").exists()
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_oracle_sweep"]["status"] == "failed"
    assert manifest["validation_oracle_sweep"]["failed_stage"] == "validation_phase_benchmark"


def test_validation_oracle_dry_run_omits_strict_and_legacy_strict_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    monkeypatch.setattr(
        parameter_sweep,
        "run_benchmark",
        lambda config: pytest.fail("dry run must not run trio benchmarks"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = _validation_oracle_spec(tmp_path, views_path, output_name="validation_legacy_strict")
    spec.validation_oracle_sweep.defaults["strict"] = False
    out_dir = run_sweep_experiment(spec, dry_run=True)

    dry_run_path = next((out_dir / "generations").glob("*/dry_run.json"))
    dry_run_payload = json.loads(dry_run_path.read_text(encoding="utf-8"))
    assert "strict" not in dry_run_payload["validation_oracle_sweep"]

    payload = {
        "datasets": [{"id": "d1", "views": str(views_path)}],
        "search_spaces": {
            "pixel": {
                "similarity_window_sizes": [1],
                "max_displacements": [0],
                "gauss_sigmas": [1.0],
                "disp_gauss_sigmas": [0.0],
                "pixel_thresholds": [0.05],
                "sub_pix_factors": [1],
                "pixel_sigma_pyramid": [False],
                "zero_fill_roll": [True],
            }
        },
        "strategies": ["pixel"],
        "stages": {"mode": "two_stage"},
        "validation_oracle_sweep": {"enabled": False, "strict": False},
    }
    parsed = SweepExperimentSpec.from_dict(payload)
    assert not hasattr(parsed.validation_oracle_sweep, "strict")


def test_independent_metric_search_uses_per_formulation_search_space(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 3],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="independent_search",
            search_stages=SweepStageSpec(
                mode="single_stage",
                total_budget_per_dataset=2,
                top_k=1,
                max_workers=1,
            ),
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                ),
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.5],
                ),
            ],
        ),
        output_dir=str(tmp_path / "independent_metric_search"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)
    config_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((out_dir / "metric_experiments").glob("runs/metric_phase/*/*/config.json"))
    ]

    windows_by_family: dict[str, set[int]] = {}
    for payload in config_payloads:
        windows_by_family.setdefault(payload["displacement_metric_family"], set()).add(
            int(payload["similarity_window_size"])
        )

    assert windows_by_family == {
        "intensity": {1, 3},
        "blurred_gradient_corr": {1, 3},
    }
    metric_summary = json.loads(
        (out_dir / "metric_experiments" / "metric_phase_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert metric_summary["stage"] == "metric_phase"
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metric_experiments"]["candidate_source"] == "independent_search"
    assert manifest["metric_experiments"]["num_candidates"] == 4
    assert {
        item["family"] for item in manifest["metric_experiments"]["per_formulation"]
    } == {"intensity", "blurred_gradient_corr"}


def test_independent_metric_search_reuses_identical_baseline_artifacts(monkeypatch, tmp_path):
    calls = _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="independent_search",
            search_stages=SweepStageSpec(
                mode="single_stage",
                total_budget_per_dataset=1,
                top_k=1,
                max_workers=1,
            ),
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                )
            ],
        ),
        output_dir=str(tmp_path / "cross_stage_reuse"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)

    assert [call for call in calls if call.startswith("metric_phase_")] == []
    metric_summary = json.loads(
        (out_dir / "metric_experiments" / "metric_phase_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert metric_summary["execution"]["skipped_valid_count"] == 1


def test_independent_metric_search_rewrites_reused_internal_stage_identity(monkeypatch, tmp_path):
    calls = _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="independent_search",
            search_stages=SweepStageSpec(
                total_budget_per_dataset=2,
                stage1_fraction=0.5,
                top_k=1,
                max_workers=1,
            ),
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                )
            ],
        ),
        output_dir=str(tmp_path / "internal_stage_reuse"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)
    metric_dir = out_dir / "metric_experiments"

    assert [call for call in calls if call.startswith("metric_phase_")] == []
    metric_summary = json.loads(
        (metric_dir / "metric_phase_summary.json").read_text(encoding="utf-8")
    )
    summary_records = json.loads((metric_dir / "summary.json").read_text(encoding="utf-8"))
    raw_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in metric_dir.glob("runs/metric_phase/*/*/metric_record.json")
    ]

    summary_run_names = [
        row["metric_record"]["run"]
        for row in metric_summary["runs"]
        if row.get("status") == "ok"
    ]
    public_run_names = (
        summary_run_names
        + [record["run"] for record in summary_records]
        + [record["run"] for record in raw_records]
    )
    assert public_run_names
    assert all(run_name.startswith("metric_phase_") for run_name in public_run_names)
    assert not any(run_name.startswith("metric_stage1_") for run_name in public_run_names)


def test_dry_run_reports_independent_metric_search_plan(monkeypatch, tmp_path):
    _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 3],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="independent_search",
            search_stages=SweepStageSpec(
                mode="single_stage",
                total_budget_per_dataset=2,
                top_k=1,
                max_workers=1,
            ),
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                ),
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.5],
                ),
            ],
        ),
        output_dir=str(tmp_path / "dry_metric_plan"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec, dry_run=True)
    dry_run_path = next((out_dir / "generations").glob("*/dry_run.json"))
    dry_run_payload = json.loads(dry_run_path.read_text(encoding="utf-8"))

    assert dry_run_payload["metric_experiments"]["candidate_source"] == "independent_search"
    assert dry_run_payload["metric_experiments"]["num_candidates"] == 4
    assert dry_run_payload["metric_experiments"]["num_runs"] == 4
    assert not (out_dir / "metric_experiments" / "runs").exists()


def test_dry_run_estimates_baseline_shortlist_metric_plan_without_rankings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parameter_sweep,
        "_run_task_isolated",
        lambda task: pytest.fail("dry run must not execute tasks"),
    )
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 3],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=2,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="baseline_shortlist",
            shortlist_top_k=2,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                ),
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.5, 1.0],
                ),
            ],
        ),
        output_dir=str(tmp_path / "dry_metric_estimate"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec, dry_run=True)
    dry_run_path = next((out_dir / "generations").glob("*/dry_run.json"))
    dry_run_payload = json.loads(dry_run_path.read_text(encoding="utf-8"))
    metric_plan = dry_run_payload["metric_experiments"]

    assert metric_plan["candidate_source"] == "baseline_shortlist"
    assert metric_plan["status"] == "estimated_requires_baseline_rankings"
    assert metric_plan["exact"] is False
    assert metric_plan["num_planned_runs"] == 6
    assert metric_plan["pending_run_count"] == 6
    assert metric_plan["estimate_reason"] == "baseline_rankings_unavailable"
    assert dry_run_payload["totals"]["pending_run_count"] == (
        dry_run_payload["stages"]["stage1"]["pending_run_count"] + 6
    )
    assert not (out_dir / "metric_experiments" / "runs").exists()


def test_preview_reports_exact_baseline_shortlist_metric_reuse(monkeypatch, tmp_path):
    calls = _install_fast_artifact_runner(monkeypatch)
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=1,
            top_k=1,
            max_workers=1,
        ),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="baseline_shortlist",
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                )
            ],
        ),
        output_dir=str(tmp_path / "exact_metric_preview"),
        save_figures=False,
        analyze=False,
    )

    out_dir = run_sweep_experiment(spec)
    assert calls
    calls.clear()

    preview = parameter_sweep.build_sweep_run_preview(spec, output_dir=out_dir)
    metric_plan = preview["metric_experiments"]

    assert metric_plan["candidate_source"] == "baseline_shortlist"
    assert metric_plan["status"] == "planned"
    assert metric_plan["exact"] is True
    assert metric_plan["num_planned_runs"] == 1
    assert metric_plan["pending_run_count"] == 0
    assert metric_plan["skipped_valid_count"] == 1
    assert preview["totals"]["pending_run_count"] == 0
    assert calls == []


def test_count_only_preview_does_not_require_generated_source_image_views(tmp_path):
    spec = SweepExperimentSpec(
        datasets=[
            SweepDatasetSpec(
                dataset_id="d1",
                source_image=str(tmp_path / "source.png"),
                center_x=10,
                center_y=10,
                view_radius=4,
            ),
            SweepDatasetSpec(
                dataset_id="d2",
                source_image=str(tmp_path / "other.png"),
                center_x=12,
                center_y=12,
                view_radius=4,
            ),
        ],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1, 3],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(
            mode="single_stage",
            total_budget_per_dataset=2,
            top_k=1,
            max_workers=1,
        ),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            candidate_source="independent_search",
            search_stages=SweepStageSpec(
                mode="single_stage",
                total_budget_per_dataset=1,
                top_k=1,
                max_workers=1,
            ),
            formulations=[
                MetricFormulationSpec(
                    formulation_id="intensity_default",
                    family="intensity",
                    alphas=[1.0],
                )
            ],
        ),
        output_dir=str(tmp_path / "source_count_only"),
        save_figures=False,
        analyze=False,
    )

    preview = parameter_sweep.build_sweep_run_preview(
        spec,
        reuse_existing_artifacts=False,
    )

    assert preview["stages"]["stage1"]["planned_run_count"] == 4
    assert preview["stages"]["stage1"]["pending_run_count"] == 4
    assert preview["stages"]["stage1"]["exact"] is False
    assert preview["metric_experiments"]["candidate_source"] == "independent_search"
    assert preview["metric_experiments"]["num_planned_runs"] == 2
    assert preview["metric_experiments"]["pending_run_count"] == 2
    assert preview["metric_experiments"]["exact"] is False
    assert preview["totals"]["planned_run_count"] == 6
    assert preview["totals"]["pending_run_count"] == 6
    assert not (tmp_path / "source_count_only" / "datasets").exists()


def test_run_sweep_resume_recomputes_corrupt_metric_phase_artifact(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.5],
                )
            ],
        ),
        output_dir=str(tmp_path / "metric_resume"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)

    metric_records = sorted(
        out_dir.glob("metric_experiments/runs/metric_phase/*/*/metric_record.json")
    )
    assert metric_records
    metric_records[0].write_text("{corrupted", encoding="utf-8")

    run_sweep_experiment(spec, output_dir=out_dir)
    run_state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    statuses = {entry["status"] for entry in run_state["runs"].values()}
    assert "recomputed" in statuses


def test_run_sweep_resume_normalizes_stale_running_metric_phase_entries(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.5],
                )
            ],
        ),
        output_dir=str(tmp_path / "metric_stale_running"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    for entry in state["runs"].values():
        if entry["stage"] == "metric_phase":
            entry["status"] = "running"
    (out_dir / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    run_sweep_experiment(spec, output_dir=out_dir)

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    metric_statuses = {
        entry["status"] for entry in state["runs"].values() if entry["stage"] == "metric_phase"
    }
    assert "running" not in metric_statuses
    assert "skipped_valid" in metric_statuses


def test_run_sweep_resume_recovers_partial_metric_phase_alpha_outputs(tmp_path):
    views = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    views_path = tmp_path / "views.npy"
    np.save(views_path, views)

    spec = SweepExperimentSpec(
        datasets=[SweepDatasetSpec(dataset_id="d1", views_path=str(views_path))],
        search_spaces={
            "pixel": _pixel_space(
                similarity_window_sizes=[1],
                max_displacements=[0],
                sub_pix_factors=[1],
                pixel_sigma_pyramid=[False],
            )
        },
        strategies=["pixel"],
        stages=SweepStageSpec(total_budget_per_dataset=2, stage1_fraction=0.5, top_k=1, max_workers=1),
        scoring=SweepScoringSpec(metric_weights={"laplacian_variance": 1.0}),
        experimental_metric_sweep=ExperimentalMetricSweepSpec(
            enabled=True,
            shortlist_top_k=1,
            formulations=[
                MetricFormulationSpec(
                    formulation_id="exp_grad",
                    family="ncc_gradient",
                    alphas=[0.0, 0.5, 1.0],
                )
            ],
        ),
        output_dir=str(tmp_path / "metric_partial_resume"),
        save_figures=False,
        analyze=False,
    )
    out_dir = run_sweep_experiment(spec)
    metric_dir = out_dir / "metric_experiments"

    for config_path in sorted(metric_dir.glob("runs/metric_phase/*/*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if float(config["displacement_alpha"]) != 0.0:
            shutil.rmtree(config_path.parents[1])

    for artifact_name in ("metric_phase_summary.json", "leaderboard.json", "summary.json", "alpha_report.json"):
        artifact_path = metric_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    for entry in state["runs"].values():
        if entry["stage"] == "metric_phase":
            entry["status"] = "running"
    (out_dir / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    run_sweep_experiment(spec, output_dir=out_dir)

    assert (metric_dir / "metric_phase_summary.json").exists()
    assert (metric_dir / "summary.json").exists()
    assert (metric_dir / "alpha_report.json").exists()

    alpha_values = {
        float(json.loads(path.read_text(encoding="utf-8"))["displacement_alpha"])
        for path in metric_dir.glob("runs/metric_phase/*/*/config.json")
    }
    assert alpha_values == {0.0, 0.5, 1.0}

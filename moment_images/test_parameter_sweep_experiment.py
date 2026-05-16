import json
from concurrent.futures.process import BrokenProcessPool
import shutil

import numpy as np
import pytest

import moment_images.parameter_sweep as parameter_sweep
from moment_images.im_process import RectificationRunConfig, rectify_views_from_config
from moment_images.parameter_sweep import (
    ExperimentalMetricSweepSpec,
    MetricFormulationSpec,
    PreparedDataset,
    SweepDatasetSpec,
    SweepCandidate,
    SweepExperimentSpec,
    SweepScoringSpec,
    SweepSearchSpace,
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

    stage1_run = output_dir / "runs" / "stage1" / "s1c0000" / "d1"
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
    assert len(spec.experimental_metric_sweep.formulations) == 1
    assert spec.experimental_metric_sweep.formulations[0].formulation_id == "legacy_grad"
    assert spec.experimental_metric_sweep.formulations[0].family == "blurred_gradient_corr"


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
    assert (out_dir / "runs" / "stage1" / "s1c0000" / "d1" / "workspace.npz").exists()

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
    assert state["schema_version"] == 1
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

    out_dir = run_sweep_experiment(relocated_spec, output_dir=relocated_output_dir)

    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["path_compatibility"]["output_dir"] == str(relocated_output_dir.resolve())
    assert (
        state["experiment_fingerprint"]
        == parameter_sweep._compute_experiment_fingerprint(state["fingerprint_payload"])
    )
    statuses = {entry["status"] for entry in state["runs"].values()}
    assert "skipped_valid" in statuses


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

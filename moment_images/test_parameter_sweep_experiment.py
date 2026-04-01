import json
from concurrent.futures.process import BrokenProcessPool

import numpy as np

import moment_images.parameter_sweep as parameter_sweep
from moment_images.im_process import RectificationRunConfig, rectify_views_from_config
from moment_images.parameter_sweep import (
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

    def fake_execute_pool_batch(tasks, workers):
        call_state["count"] += 1
        assert workers in {2, 4}
        if call_state["count"] == 1:
            completed = [_build_eval_result(tasks[0], status="ok")]
            pending = list(tasks[1:])
            return completed, pending, BrokenProcessPool("pool crashed")
        completed = [_build_eval_result(task, status="ok") for task in tasks]
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
    stage1_summary = json.loads((output_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    assert "execution" in stage1_summary
    assert "selected_workers" in stage1_summary["execution"]

    stage1_run = output_dir / "runs" / "stage1" / "s1c0000" / "d1"
    assert (stage1_run / "workspace.npz").exists()
    assert (stage1_run / "rectified_views.npy").exists()
    assert (stage1_run / "avg_rectified.png").exists()

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

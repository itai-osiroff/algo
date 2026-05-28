import csv
import json
from pathlib import Path

import cv2 as cv
import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")

from moment_images import sweep_metric_explorer as explorer
from moment_images.sweep_metric_explorer import (
    HYDRATE_MODE_LAZY,
    _ensure_profile_results,
    _bayes_dct_entropy,
    _gini_index,
    _hoyer_index,
    _normalized_dct_channel,
    _normalized_shannon_entropy,
    _prepare_guide_log_channel,
    _resolve_comparison_records,
    _vollath_f4,
    build_comparison_metric_rows,
    create_channel_comparison_figure,
    export_metric_report,
    load_comparison_display_bundle,
    load_sweep_metric_explorer_model,
    main,
    search_profile_combinations,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_preview(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _write_workspace(path: Path, avg_image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rectified_views = np.stack(
        [
            np.stack([avg_image, avg_image], axis=0),
            np.stack([avg_image, avg_image], axis=0),
        ],
        axis=0,
    ).astype(np.float64)
    np.savez_compressed(
        path,
        average_rectified_image=np.asarray(avg_image, dtype=np.float64),
        average_rectified_global_image=np.asarray(cv.GaussianBlur(avg_image.astype(np.float32), (0, 0), 1.6), dtype=np.float64),
        rectified_views=rectified_views,
        config_json=np.array(json.dumps({"strategy": "combined"}, sort_keys=True)),
    )


def _checkerboard(size: int = 32, scale: float = 255.0) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return (((xx + yy) % 2) * scale).astype(np.float64)


def _square_feature(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float64)
    image[10:22, 14:18] = 255.0
    return image


def _stripe_feature(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float64)
    image[:, ::4] = 255.0
    return image


def _make_modern_fixture(base_dir: Path) -> Path:
    output_dir = base_dir / "metric_explorer_modern"
    datasets = ["d1", "d2"]
    _write_json(
        output_dir / "experiment_manifest.json",
        {
            "generated_at": "2026-05-24T12:00:00+00:00",
            "strategies": ["combined"],
            "datasets": [{"dataset_id": dataset_id} for dataset_id in datasets],
            "stages": {"mode": "single_stage", "top_k": 2},
            "scoring": {
                "metric_weights": {"laplacian_variance": 1.0},
                "robustness_lambda": 0.0,
                "missing_dataset_penalty": 0.0,
            },
            "reporting": {"primary_metric": "laplacian_variance"},
        },
    )
    _write_json(
        output_dir / "run_state.json",
        {
            "schema_version": 1,
            "created_at": "2026-05-24T12:00:00+00:00",
            "updated_at": "2026-05-24T12:10:00+00:00",
            "experiment_fingerprint": "fixture",
            "fingerprint_payload": {
                "spec": {
                    "strategies": ["combined"],
                    "stages": {"mode": "single_stage", "top_k": 2},
                    "scoring": {
                        "metric_weights": {"laplacian_variance": 1.0},
                        "robustness_lambda": 0.0,
                        "missing_dataset_penalty": 0.0,
                    },
                },
                "datasets": [{"dataset_id": dataset_id} for dataset_id in datasets],
            },
            "stages": {"stage1": {"planned_runs": 4, "completed_runs": 4}},
            "runs": {},
        },
    )

    sharp = _square_feature()
    blurred = cv.GaussianBlur(sharp.astype(np.float32), (0, 0), 2.0)
    very_blurred = cv.GaussianBlur(sharp.astype(np.float32), (0, 0), 4.0)
    extreme_blur = cv.GaussianBlur(sharp.astype(np.float32), (0, 0), 6.0)
    images = {
        ("c_good", "d1"): sharp,
        ("c_good", "d2"): blurred,
        ("c_bad", "d1"): very_blurred,
        ("c_bad", "d2"): extreme_blur,
    }
    rankings = [
        {
            "candidate_id": "c_bad",
            "strategy": "combined",
            "config": {
                "strategy": "combined",
                "displacement_metric_family": "intensity",
                "displacement_alpha": 1.0,
                "metric_formulation_id": "intensity_default",
                "max_displacement": 8,
                "gauss_sigma": 5.0,
            },
            "params": {
                "max_displacement": 8,
                "gauss_sigma": 5.0,
                "pixel_threshold": 0.05,
            },
            "score": 0.92,
            "mean_dataset_score": 0.92,
            "std_dataset_score": 0.01,
            "missing_datasets": [],
            "dataset_scores": {"d1": 0.91, "d2": 0.93},
            "runtime_seconds": {"d1": 11.0, "d2": 11.5},
            "dataset_metrics": {
                "d1": {"laplacian_variance": 90.0},
                "d2": {"laplacian_variance": 91.0},
            },
            "rank": 1,
        },
        {
            "candidate_id": "c_good",
            "strategy": "combined",
            "config": {
                "strategy": "combined",
                "displacement_metric_family": "intensity",
                "displacement_alpha": 1.0,
                "metric_formulation_id": "intensity_default",
                "max_displacement": 5,
                "gauss_sigma": 3.0,
            },
            "params": {
                "max_displacement": 5,
                "gauss_sigma": 3.0,
                "pixel_threshold": 0.05,
            },
            "score": 0.51,
            "mean_dataset_score": 0.51,
            "std_dataset_score": 0.02,
            "missing_datasets": [],
            "dataset_scores": {"d1": 0.52, "d2": 0.50},
            "runtime_seconds": {"d1": 8.0, "d2": 8.5},
            "dataset_metrics": {
                "d1": {"laplacian_variance": 52.0},
                "d2": {"laplacian_variance": 50.0},
            },
            "rank": 2,
        },
    ]
    _write_json(
        output_dir / "stage1_summary.json",
        {
            "stage": "stage1",
            "num_runs": 2,
            "num_successful": 2,
            "num_failed": 0,
            "rankings": rankings,
            "failed_runs": [],
            "runs": [],
            "execution": {"selected_workers": 1, "requested_workers": 1},
        },
    )
    _write_json(output_dir / "summary.json", [])

    for (candidate_id, dataset_id), image in images.items():
        run_dir = output_dir / "runs" / "stage1" / candidate_id / dataset_id
        _write_preview(run_dir / "avg_rectified.png", image)
        _write_workspace(run_dir / "workspace.npz", image)

    return output_dir


def _write_annotations(output_dir: Path) -> None:
    _write_json(
        output_dir / "metric_review" / "annotations.json",
        {
            "schema_version": 1,
            "annotations": [
                {"stage": "stage1", "candidate_id": "c_good", "dataset_id": "d1", "label": "top"},
                {"stage": "stage1", "candidate_id": "c_good", "dataset_id": "d2", "label": "good"},
                {"stage": "stage1", "candidate_id": "c_bad", "dataset_id": "d1", "label": "bad"},
                {"stage": "stage1", "candidate_id": "c_bad", "dataset_id": "d2", "label": "bad"},
            ],
        },
    )


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


def test_new_metric_primitives_behave_as_expected():
    sparse = np.zeros((16, 16), dtype=np.float64)
    sparse[8, 8] = 255.0
    uniform = np.full((16, 16), 64.0, dtype=np.float64)
    sharp = _square_feature()
    blurred = cv.GaussianBlur(sharp.astype(np.float32), (0, 0), 2.5)
    stripes = _stripe_feature()
    stripe_blur = cv.GaussianBlur(stripes.astype(np.float32), (0, 0), 2.0)

    assert _gini_index(sparse) > _gini_index(uniform)
    assert _hoyer_index(sparse) > _hoyer_index(uniform)
    assert _vollath_f4(sharp) > _vollath_f4(blurred)

    sharp_ndct = _normalized_dct_channel(sharp)
    blur_ndct = _normalized_dct_channel(blurred)
    assert _normalized_shannon_entropy(sharp_ndct) > _normalized_shannon_entropy(blur_ndct)
    assert _bayes_dct_entropy(stripes) > _bayes_dct_entropy(stripe_blur)

    sharp_guide = _prepare_guide_log_channel(sharp)
    blur_guide = _prepare_guide_log_channel(blurred)
    assert _gini_index(sharp_guide) > _gini_index(blur_guide)
    assert _hoyer_index(sharp_guide) > _hoyer_index(blur_guide)
    assert _vollath_f4(sharp_guide) > _vollath_f4(blur_guide)


def test_load_sweep_metric_explorer_model_recomputes_metrics_and_profiles(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)

    model = load_sweep_metric_explorer_model(output_dir)

    assert model.base_model.primary_stage_name == "stage1"
    assert len(model.raw_records) == 4
    assert {"ndct_shannon_avg", "gini_avg", "hoyer_avg", "vollath_f4_avg"}.issubset(
        model.metric_registry.keys()
    )
    saved = model.profile_results[("stage1", "saved_original")]
    new_only = model.profile_results[("stage1", "new_dct_sparse")]
    assert saved[0].candidate_id == "c_bad"
    assert new_only[0].candidate_id == "c_good"
    good_d1 = next(
        record for record in model.raw_records if record.candidate_id == "c_good" and record.dataset_id == "d1"
    )
    bad_d2 = next(
        record for record in model.raw_records if record.candidate_id == "c_bad" and record.dataset_id == "d2"
    )
    assert good_d1.metrics["vollath_f4_avg"] > bad_d2.metrics["vollath_f4_avg"]
    assert good_d1.metrics["hoyer_avg"] > bad_d2.metrics["hoyer_avg"]


def test_lazy_load_defers_workspace_metrics_until_profile_is_requested(tmp_path, monkeypatch):
    output_dir = _make_modern_fixture(tmp_path)
    calls: list[Path] = []
    original = explorer.compute_metrics_from_workspace

    def _counting_loader(workspace_path: Path):
        calls.append(Path(workspace_path))
        return original(workspace_path)

    monkeypatch.setattr(explorer, "compute_metrics_from_workspace", _counting_loader)

    model = load_sweep_metric_explorer_model(output_dir, hydrate_mode=HYDRATE_MODE_LAZY)

    assert calls == []
    assert ("stage1", "saved_original") in model.profile_results
    assert ("stage1", "old_recomputed") in model.profile_results
    assert ("stage1", "new_dct_sparse") not in model.profile_results

    _ensure_profile_results(model, stage_name="stage1", profile_name="new_dct_sparse", include_analysis=True)

    assert len(calls) == 4
    assert ("stage1", "new_dct_sparse") in model.profile_results


def test_lazy_reload_reuses_metric_cache_without_reopening_workspaces(tmp_path, monkeypatch):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir, hydrate_mode=HYDRATE_MODE_LAZY)
    _ensure_profile_results(model, stage_name="stage1", profile_name="new_dct_sparse", include_analysis=True)

    calls: list[Path] = []
    original = explorer.compute_metrics_from_workspace

    def _counting_loader(workspace_path: Path):
        calls.append(Path(workspace_path))
        return original(workspace_path)

    monkeypatch.setattr(explorer, "compute_metrics_from_workspace", _counting_loader)
    reloaded = load_sweep_metric_explorer_model(output_dir, hydrate_mode=HYDRATE_MODE_LAZY)

    assert calls == []
    assert ("stage1", "new_dct_sparse") in reloaded.profile_results


def test_annotations_enable_evaluation_and_search(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    _write_annotations(output_dir)

    model = load_sweep_metric_explorer_model(output_dir)

    saved_eval = next(
        item for item in model.evaluations if item.stage == "stage1" and item.profile_name == "saved_original"
    )
    new_eval = next(
        item for item in model.evaluations if item.stage == "stage1" and item.profile_name == "new_dct_sparse"
    )
    assert saved_eval.pairwise_agreement is not None
    assert new_eval.pairwise_agreement is not None
    assert new_eval.pairwise_agreement > saved_eval.pairwise_agreement
    assert any(item.profile_name == "saved_original" for item in model.disagreements)

    search_results = search_profile_combinations(
        model,
        stage_name="stage1",
        metric_names=["ndct_shannon_avg", "gini_avg", "hoyer_avg"],
        weight_step=0.5,
    )
    assert search_results
    assert search_results[0].pairwise_agreement is not None
    assert [explorer._search_result_sort_key(item) for item in search_results] == sorted(
        explorer._search_result_sort_key(item) for item in search_results
    )


def test_replace_search_results_preserves_quality_order_and_stage_isolation(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    stage2_results = [
        explorer.SearchResult(
            stage="stage2",
            profile_name="search_0003",
            metric_weights={"other": 1.0},
            pairwise_agreement=0.8,
            top1_hit_rate=0.8,
            mean_label_top_k=0.8,
            spearman_correlation=0.8,
        ),
        explorer.SearchResult(
            stage="stage2",
            profile_name="search_0004",
            metric_weights={"other": 0.5},
            pairwise_agreement=0.7,
            top1_hit_rate=0.7,
            mean_label_top_k=0.7,
            spearman_correlation=0.7,
        ),
    ]
    model.search_results = list(stage2_results)

    worse_name_better_quality = explorer.SearchResult(
        stage="stage1",
        profile_name="search_0002",
        metric_weights={"better": 1.0},
        pairwise_agreement=1.0,
        top1_hit_rate=1.0,
        mean_label_top_k=1.0,
        spearman_correlation=1.0,
    )
    better_name_worse_quality = explorer.SearchResult(
        stage="stage1",
        profile_name="search_0001",
        metric_weights={"worse": 1.0},
        pairwise_agreement=0.5,
        top1_hit_rate=1.0,
        mean_label_top_k=1.0,
        spearman_correlation=1.0,
    )

    explorer._replace_search_results_for_stage(
        model,
        "stage1",
        [better_name_worse_quality, worse_name_better_quality],
    )

    stage1_results = [item for item in model.search_results if item.stage == "stage1"]
    assert [item.profile_name for item in stage1_results] == ["search_0002", "search_0001"]
    assert [item for item in model.search_results if item.stage == "stage2"] == stage2_results

    replacement = explorer.SearchResult(
        stage="stage1",
        profile_name="search_0005",
        metric_weights={"replacement": 1.0},
        pairwise_agreement=0.9,
        top1_hit_rate=0.9,
        mean_label_top_k=0.9,
        spearman_correlation=0.9,
    )
    explorer._replace_search_results_for_stage(model, "stage1", [replacement])
    assert [item.profile_name for item in model.search_results if item.stage == "stage1"] == ["search_0005"]
    assert [item for item in model.search_results if item.stage == "stage2"] == stage2_results


def test_search_results_ui_helpers_preserve_stored_order(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    model.search_results = [
        explorer.SearchResult(
            stage="stage1",
            profile_name="search_0002",
            metric_weights={"better": 1.0},
            pairwise_agreement=1.0,
            top1_hit_rate=1.0,
            mean_label_top_k=1.0,
            spearman_correlation=1.0,
        ),
        explorer.SearchResult(
            stage="stage1",
            profile_name="search_0001",
            metric_weights={"worse": 1.0},
            pairwise_agreement=0.5,
            top1_hit_rate=1.0,
            mean_label_top_k=1.0,
            spearman_correlation=1.0,
        ),
    ]
    app = explorer.SweepMetricExplorerApp.__new__(explorer.SweepMetricExplorerApp)
    app.model = model

    results = app._search_results_for_stage("stage1")
    rendered = explorer._format_search_results_text(results)

    assert [item.profile_name for item in results] == ["search_0002", "search_0001"]
    assert '"better":1.0' in rendered.splitlines()[0]


def test_export_search_results_csv_preserves_quality_order(tmp_path, monkeypatch):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    model.search_results = [
        explorer.SearchResult(
            stage="stage1",
            profile_name="search_0002",
            metric_weights={"better": 1.0},
            pairwise_agreement=1.0,
            top1_hit_rate=1.0,
            mean_label_top_k=1.0,
            spearman_correlation=1.0,
        ),
        explorer.SearchResult(
            stage="stage1",
            profile_name="search_0001",
            metric_weights={"worse": 1.0},
            pairwise_agreement=0.5,
            top1_hit_rate=1.0,
            mean_label_top_k=1.0,
            spearman_correlation=1.0,
        ),
    ]
    monkeypatch.setattr(explorer, "_hydrate_model_for_export", lambda model: None)

    export_dir = tmp_path / "search_order_export"
    export_metric_report(model, export_dir)

    with (export_dir / "search_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["profile_name"] for row in rows if row["stage"] == "stage1"] == [
        "search_0002",
        "search_0001",
    ]


def test_export_metric_report_hydrates_lazy_model(tmp_path, monkeypatch):
    output_dir = _make_modern_fixture(tmp_path)
    _write_annotations(output_dir)
    model = load_sweep_metric_explorer_model(output_dir, hydrate_mode=HYDRATE_MODE_LAZY)
    calls: list[Path] = []
    original = explorer.compute_metrics_from_workspace

    def _counting_loader(workspace_path: Path):
        calls.append(Path(workspace_path))
        return original(workspace_path)

    monkeypatch.setattr(explorer, "compute_metrics_from_workspace", _counting_loader)

    export_dir = tmp_path / "lazy_export"
    export_metric_report(model, export_dir)

    assert calls
    assert ("stage1", "new_dct_sparse") in model.profile_results
    assert (export_dir / "candidate_metrics.csv").exists()
    assert (export_dir / "profile_results.csv").exists()


def test_export_metric_report_and_main(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    _write_annotations(output_dir)
    model = load_sweep_metric_explorer_model(output_dir)

    export_dir = tmp_path / "export"
    exported = export_metric_report(model, export_dir)

    assert exported == export_dir
    assert (export_dir / "candidate_metrics.csv").exists()
    assert (export_dir / "profile_results.csv").exists()
    assert (export_dir / "search_results.csv").exists()
    assert (export_dir / "disagreements.csv").exists()
    assert (export_dir / "report.json").exists()
    assert (export_dir / "index.html").exists()

    cli_export = tmp_path / "cli_export"
    main([str(output_dir), "--export-dir", str(cli_export)])
    assert (cli_export / "candidate_metrics.csv").exists()


def test_channel_comparison_figure_preserves_image_aspect_ratio(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    records = [
        record
        for record in model.raw_records
        if record.stage == "stage1" and record.dataset_id == "d1"
    ]

    figure = create_channel_comparison_figure(model, records[:1], channel_name="avg_image")

    image_axes = [ax for ax in figure.axes if ax.images]
    assert image_axes
    assert all(ax.get_aspect() == 1.0 for ax in image_axes)


def test_channel_comparison_figure_grows_with_scale(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    records = [
        record
        for record in model.raw_records
        if record.stage == "stage1" and record.dataset_id == "d1"
    ][:2]

    small = create_channel_comparison_figure(model, records, channel_name="avg_image", figure_scale=1.0)
    large = create_channel_comparison_figure(model, records, channel_name="avg_image", figure_scale=2.0)

    assert large.get_figwidth() > small.get_figwidth()
    assert large.get_figheight() > small.get_figheight()


def test_resolve_comparison_records_supports_selected_and_slot_modes(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    record_lookup = {
        (record.stage, record.candidate_id, record.dataset_id): record
        for record in model.raw_records
    }
    selected_rows = model.profile_results[("stage1", "new_dct_sparse")]

    selected_mode = _resolve_comparison_records(
        record_lookup,
        selected_rows,
        dataset_id="d1",
        comparison_mode="Selected",
        slot_a=None,
        slot_b=None,
    )
    slot_mode = _resolve_comparison_records(
        record_lookup,
        selected_rows,
        dataset_id="d1",
        comparison_mode="A vs B",
        slot_a=("stage1", "c_good", "d1"),
        slot_b=("stage1", "c_bad", "d2"),
    )

    assert [record.candidate_id for record in selected_mode] == ["c_good", "c_bad"]
    assert [(record.candidate_id, record.dataset_id) for record in slot_mode] == [
        ("c_good", "d1"),
        ("c_bad", "d2"),
    ]


def test_load_comparison_display_bundle_returns_preview_avg_and_guide(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    record = next(
        item for item in model.raw_records if item.stage == "stage1" and item.candidate_id == "c_good" and item.dataset_id == "d1"
    )

    bundle = load_comparison_display_bundle(record)

    assert bundle.preview is not None
    assert bundle.avg_image is not None
    assert bundle.guide_log is not None
    assert bundle.get("preview").shape == bundle.get("avg_image").shape
    assert bundle.get("guide_log").shape == bundle.get("avg_image").shape


def test_build_comparison_metric_rows_identifies_preferred_side(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    model = load_sweep_metric_explorer_model(output_dir)
    good = next(
        item for item in model.raw_records if item.stage == "stage1" and item.candidate_id == "c_good" and item.dataset_id == "d1"
    )
    bad = next(
        item for item in model.raw_records if item.stage == "stage1" and item.candidate_id == "c_bad" and item.dataset_id == "d1"
    )

    rows = build_comparison_metric_rows(model, good, bad)

    vollath = next(row for row in rows if row.metric_name == "vollath_f4_avg")
    shannon = next(row for row in rows if row.metric_name == "ndct_shannon_avg")
    assert vollath.winner == "A"
    assert vollath.delta > 0
    assert shannon.winner == "A"


def test_gui_labeling_preserves_filters_and_selection(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    root = _make_root_or_skip()
    try:
        app = explorer.SweepMetricExplorerApp(root)
        app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
        app.messagebox.showwarning = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))

        app.load_from_path(output_dir)
        root.update()

        app.profile_var.set("old_recomputed")
        app.dataset_var.set("d2")
        app.refresh_leaderboard()
        root.update()

        app.leaderboard_tree.selection_set("c_good")
        app._on_leaderboard_selection_changed()
        root.update()

        app._set_labels_for_selection("top")
        root.update()

        assert app.stage_var.get() == "stage1"
        assert app.profile_var.get() == "old_recomputed"
        assert app.dataset_var.get() == "d2"
        assert app.leaderboard_tree.selection() == ("c_good",)
        assert app.model is not None
        assert app.model.annotations[explorer._annotation_key("stage1", "c_good", "d2")].label == "top"
    finally:
        root.destroy()


def test_gui_custom_profile_persists_to_next_load(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    config_path = output_dir / "metric_review" / "metrics_config.json"
    weights = {"average_intensity": 0.25, "laplacian_variance": 0.75}

    root = _make_root_or_skip()
    try:
        app = explorer.SweepMetricExplorerApp(root)
        app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
        app.messagebox.showwarning = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))

        app.load_from_path(output_dir)
        app.search_profile_name_var.set("saved_custom")
        app._set_text(app.weight_json_text, json.dumps(weights, indent=2, sort_keys=True))
        app._build_json_profile()
        root.update()
    finally:
        root.destroy()

    assert config_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved_rows = [row for row in payload["profiles"] if row["name"] == "saved_custom"]
    assert len(saved_rows) == 1
    assert saved_rows[0]["metric_weights"] == weights

    reloaded = load_sweep_metric_explorer_model(output_dir)
    matches = [profile for profile in reloaded.config.profiles if profile.name == "saved_custom"]
    assert len(matches) == 1
    assert matches[0].metric_weights == weights


def test_gui_custom_profile_overwrites_same_name(tmp_path):
    output_dir = _make_modern_fixture(tmp_path)
    config_path = output_dir / "metric_review" / "metrics_config.json"
    first_weights = {"laplacian_variance": 1.0}
    second_weights = {"average_intensity": 1.0}

    root = _make_root_or_skip()
    try:
        app = explorer.SweepMetricExplorerApp(root)
        app.messagebox.showerror = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))
        app.messagebox.showwarning = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args))

        app.load_from_path(output_dir)
        app.search_profile_name_var.set("saved_custom")
        app._set_text(app.weight_json_text, json.dumps(first_weights, indent=2, sort_keys=True))
        app._build_json_profile()
        app._set_text(app.weight_json_text, json.dumps(second_weights, indent=2, sort_keys=True))
        app._build_json_profile()
        root.update()
    finally:
        root.destroy()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved_rows = [row for row in payload["profiles"] if row["name"] == "saved_custom"]
    assert len(saved_rows) == 1
    assert saved_rows[0]["metric_weights"] == second_weights

    reloaded = load_sweep_metric_explorer_model(output_dir)
    matches = [profile for profile in reloaded.config.profiles if profile.name == "saved_custom"]
    assert len(matches) == 1
    assert matches[0].metric_weights == second_weights

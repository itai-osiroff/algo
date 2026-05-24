import json
from pathlib import Path

import cv2 as cv
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from moment_images.sweep_metric_explorer import (
    _bayes_dct_entropy,
    _gini_index,
    _hoyer_index,
    _normalized_dct_channel,
    _normalized_shannon_entropy,
    _prepare_guide_log_channel,
    _vollath_f4,
    create_channel_comparison_figure,
    export_metric_report,
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

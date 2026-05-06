import json

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from moment_images.im_process import (
    RectificationRunConfig,
    RectificationResult,
    assemble_HDR,
    extract_views,
    generate_rectification_figures,
    min_max_normalize,
    rectify_views_from_config,
)
from moment_images.parameter_sweep import (
    LiveMetricsPlotter,
    _generate_views,
    _save_run_artifacts,
    sweep_parameters,
)


def test_parameter_sweep_matches_direct_rectification(tmp_path):
    views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)

    window_sizes = [1]
    max_displacements = [0]
    gauss_sigmas = [1.0]
    disp_gauss_sigmas = [0.0]
    pixel_thresholds = [0.05]

    output_dir = tmp_path / "runs"

    sweep_parameters(
        views=views,
        output_dir=output_dir,
        window_sizes=window_sizes,
        max_displacements=max_displacements,
        gauss_sigmas=gauss_sigmas,
        disp_gauss_sigmas=disp_gauss_sigmas,
        pixel_thresholds=pixel_thresholds,
        strategy="pixel",
        save_figures_flag=False,
    )

    run_dir = output_dir / "win1_disp0_g1.0_dg0.0_pt0.05"
    sweep_rectified = np.load(run_dir / "rectified_views.npy")
    with np.load(run_dir / "workspace.npz") as workspace:
        sweep_rectified_workspace = workspace["rectified_views"]
        sweep_avg_workspace = workspace["average_rectified_image"]
        assert "pixel_displacements_x" in workspace.files
        assert "pixel_displacements_y" in workspace.files
        assert "reference_view_index" in workspace.files
        assert "rectification_strategy" in workspace.files
        assert "run_name" in workspace.files
        assert "config_json" in workspace.files
        assert "pixel_blended_scores" not in workspace.files
        assert "coordinates" not in workspace.files
        assert "scores" not in workspace.files
        assert "pixel_component_score_gradient" not in workspace.files
        assert workspace["reference_view_index"].shape == (2,)
        assert workspace["pixel_displacements_x"].ndim == 4
        assert workspace["pixel_displacements_y"].ndim == 4
        assert int(np.asarray(workspace["rectification_strategy"]).item()) == 2
        assert (
            str(np.asarray(workspace["run_name"]).item())
            == "win1_disp0_g1.0_dg0.0_pt0.05"
        )
        parsed_config = json.loads(str(np.asarray(workspace["config_json"]).item()))
        assert parsed_config["strategy"] == "pixel"
        assert parsed_config["displacement_metric_family"] == "intensity"
        assert parsed_config["displacement_alpha"] == 1.0
        assert parsed_config["metric_formulation_id"] == "intensity_default"

    direct_config = RectificationRunConfig(
        strategy="pixel",
        similarity_window_size=1,
        refinement_window_size=1,
        max_displacement=0,
        gauss_sigma=1.0,
        gauss_size=6,
        disp_gauss_sigma=0.0,
        disp_gauss_size=1,
        pixel_threshold=0.05,
    )

    direct_result = rectify_views_from_config(views, direct_config)

    np.testing.assert_allclose(sweep_rectified, direct_result.rectified_views)
    np.testing.assert_allclose(sweep_rectified_workspace, direct_result.rectified_views)
    np.testing.assert_allclose(sweep_avg_workspace, direct_result.avg_image)
    np.testing.assert_allclose(sweep_rectified.mean(axis=(0, 1)), direct_result.avg_image)
    assert direct_result.pixel_similarity is not None
    assert direct_result.pixel_similarity.blended_score_maps is None
    assert direct_result.pixel_similarity.component_score_maps is None
    assert (run_dir / "workspace.mat").exists()
    assert not (run_dir / "rectified_views.npz").exists()


def test_non_intensity_pixel_run_saves_metric_workspace_artifacts(tmp_path):
    views = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[3.0, 4.0], [5.0, 6.0]],
            ],
        ],
        dtype=np.float32,
    )
    config = RectificationRunConfig(
        strategy="pixel",
        similarity_window_size=1,
        refinement_window_size=1,
        max_displacement=0,
        gauss_sigma=1.0,
        gauss_size=5,
        disp_gauss_sigma=0.0,
        disp_gauss_size=1,
        pixel_threshold=0.05,
        sub_pix_factor=1,
        displacement_metric_family="ncc_gradient",
        displacement_alpha=0.5,
    )

    result = rectify_views_from_config(views, config)
    _save_run_artifacts(tmp_path, "gradient_metric_run", result)

    run_dir = tmp_path / "gradient_metric_run"
    with np.load(run_dir / "workspace.npz") as workspace:
        assert "pixel_blended_scores" in workspace.files
        assert "pixel_component_score_gradient" in workspace.files
        assert "pixel_component_score_laplacian" not in workspace.files
        parsed_config = json.loads(str(np.asarray(workspace["config_json"]).item()))
        assert parsed_config["displacement_metric_family"] == "blurred_gradient_corr"
        assert parsed_config["displacement_alpha"] == 0.5


def test_live_metrics_plotter_tracks_metrics():
    plt.switch_backend("Agg")
    plotter = LiveMetricsPlotter(["average_intensity", "maximum_intensity"])
    record = {
        "run": "run1",
        "config": {},
        "global_only_metrics": {
            "average_intensity": 1.0,
            "maximum_intensity": 2.0,
        },
        "global_and_local_metrics": {
            "average_intensity": 0.5,
            "maximum_intensity": 1.5,
        },
    }

    plotter.update("run1", record)

    assert plotter.run_labels == ["run1"]
    assert plotter.data["global_only_metrics"]["average_intensity"] == [1.0]
    assert plotter.data["global_and_local_metrics"]["maximum_intensity"] == [1.5]
    plt.close(plotter.fig)


def test_assemble_hdr_respects_normalization_toggle():
    img1 = Image.fromarray(np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16))
    img2 = Image.fromarray(np.array([[1200, 2200], [3200, 4200]], dtype=np.uint16))
    exposure_times = np.array([1.0, 0.5], dtype=np.float32)

    hdr_raw = assemble_HDR(
        [img1, img2], exposure_times=exposure_times, normalize_to_8bit=False
    )
    hdr_norm = assemble_HDR(
        [img1, img2], exposure_times=exposure_times, normalize_to_8bit=True
    )

    hdr_raw_arr = np.asarray(hdr_raw)
    hdr_norm_arr = np.asarray(hdr_norm)

    assert float(np.max(hdr_raw_arr)) > 255.0
    assert hdr_norm_arr.dtype == np.uint8
    assert float(np.max(hdr_norm_arr)) <= 255.0


def test_rectify_views_from_config_hdr_honors_normalization_toggle():
    base = (np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) + 1.0) * 300.0
    views = np.stack([base, base * 1.1], axis=0)

    common_kwargs = dict(
        strategy="pixel",
        similarity_window_size=1,
        refinement_window_size=1,
        max_displacement=0,
        gauss_sigma=1.0,
        gauss_size=6,
        disp_gauss_sigma=0.0,
        disp_gauss_size=1,
        pixel_threshold=0.05,
        sub_pix_factor=1,
        hdr_enabled=True,
        hdr_exposure_times=[1.0, 0.5],
    )

    raw_result = rectify_views_from_config(
        views, RectificationRunConfig(**common_kwargs, normalize_to_8bit=False)
    )
    normalized_result = rectify_views_from_config(
        views, RectificationRunConfig(**common_kwargs, normalize_to_8bit=True)
    )

    assert float(np.max(raw_result.rectified_views)) > 255.0
    assert float(np.max(normalized_result.rectified_views)) <= 255.0 + 1e-6


def test_generate_views_respects_normalization_toggle(tmp_path):
    image_arr = (np.arange(400, dtype=np.uint16).reshape(20, 20) + 1) * 50
    image_path = tmp_path / "source.tiff"
    Image.fromarray(image_arr).save(image_path)

    raw_views, _ = _generate_views(
        image_path=image_path,
        center_x=10,
        center_y=10,
        radius=2,
        grid_size=1,
        view_distance=4,
        cache_dir=tmp_path / "raw_cache",
        normalize_to_8bit=False,
    )
    normalized_views, _ = _generate_views(
        image_path=image_path,
        center_x=10,
        center_y=10,
        radius=2,
        grid_size=1,
        view_distance=4,
        cache_dir=tmp_path / "norm_cache",
        normalize_to_8bit=True,
    )

    expected_norm = extract_views(
        np.asarray(min_max_normalize(Image.fromarray(image_arr))),
        np.array([10, 10, 2]),
        num_views=1,
        view_dist=4,
    )

    assert float(np.max(raw_views)) > 255.0
    assert float(np.max(normalized_views)) <= 255.0
    np.testing.assert_array_equal(normalized_views, expected_norm)


def test_generate_rectification_figures_supports_high_dynamic_range_inputs():
    config = RectificationRunConfig(strategy="combined", normalize_to_8bit=False)
    result = RectificationResult(
        avg_image=np.full((4, 4), 2200.0, dtype=np.float32),
        rectified_views=np.full((1, 1, 4, 4), 2100.0, dtype=np.float32),
        similarity_data=None,
        config=config,
        ref_view_index=(0, 0),
        input_views=np.full((1, 1, 4, 4), 2000.0, dtype=np.float32),
        global_similarity=None,
        global_rectified_views=np.full((1, 1, 4, 4), 2050.0, dtype=np.float32),
        global_avg_image=np.full((4, 4), 2150.0, dtype=np.float32),
        pixel_similarity=None,
    )

    figures = generate_rectification_figures(result)

    assert len(figures["averages"]) >= 3
    for fig_group in figures.values():
        for fig in fig_group:
            plt.close(fig)

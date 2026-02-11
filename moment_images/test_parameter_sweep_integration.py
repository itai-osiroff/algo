import numpy as np
from matplotlib import pyplot as plt

from moment_images.im_process import RectificationRunConfig, rectify_views_from_config
from moment_images.parameter_sweep import LiveMetricsPlotter, sweep_parameters


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
    sweep_rectified_npz = np.load(run_dir / "rectified_views.npz")["rectified_views"]

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
    np.testing.assert_allclose(sweep_rectified_npz, direct_result.rectified_views)
    np.testing.assert_allclose(sweep_rectified.mean(axis=(0, 1)), direct_result.avg_image)


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

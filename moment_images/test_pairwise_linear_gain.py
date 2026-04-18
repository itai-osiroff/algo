import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from PIL import Image

from moment_images.pairwise_linear_gain import (
    SCATTER_SAMPLE_LIMIT,
    build_visualization,
    compute_linear_gain,
    load_grayscale_image,
    save_linear_gain_results,
    select_scatter_sample,
)


def _save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def test_compute_linear_gain_recovers_known_slope_and_intercept():
    image1 = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    image2 = 7.0 + 2.5 * image1

    result, workspace = compute_linear_gain(image1, image2, fit_intercept=True)

    assert result.fit_intercept is True
    assert result.regression_mode == "with_intercept"
    assert result.gain == pytest.approx(2.5)
    assert result.intercept == pytest.approx(7.0)
    assert result.r_value == pytest.approx(1.0)
    assert result.r_squared == pytest.approx(1.0)
    assert result.pixel_count_total == 12
    assert result.pixel_count_valid == 12
    assert workspace["x_valid"].shape == (12,)
    assert workspace["residuals"].shape == (12,)


def test_compute_linear_gain_recovers_known_slope_in_gain_only_mode():
    image1 = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    image2 = 3.25 * image1

    result, workspace = compute_linear_gain(image1, image2, fit_intercept=False)

    assert result.fit_intercept is False
    assert result.regression_mode == "gain_only"
    assert result.gain == pytest.approx(3.25)
    assert result.intercept == pytest.approx(0.0)
    assert result.r_value == pytest.approx(1.0)
    assert result.r_squared == pytest.approx(1.0)
    assert np.isnan(result.p_value)
    assert np.isnan(result.std_err)
    np.testing.assert_allclose(workspace["y_pred"], image2.reshape(-1))


def test_compute_linear_gain_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical image shapes"):
        compute_linear_gain(np.zeros((2, 2)), np.zeros((2, 3)))


def test_load_grayscale_image_rejects_rgb_input(tmp_path):
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    image_path = tmp_path / "rgb.png"
    _save_image(image_path, rgb)

    with pytest.raises(ValueError, match="2D grayscale image"):
        load_grayscale_image(image_path)


def test_compute_linear_gain_filters_invalid_pairs_and_reports_counts():
    image1 = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float64)
    image2 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    result, workspace = compute_linear_gain(image1, image2, fit_intercept=True)

    assert result.pixel_count_total == 4
    assert result.pixel_count_valid == 2
    np.testing.assert_array_equal(
        workspace["valid_mask"],
        np.array([True, False, False, True]),
    )
    np.testing.assert_allclose(workspace["x_valid"], np.array([1.0, 4.0]))
    np.testing.assert_allclose(workspace["y_valid"], np.array([10.0, 40.0]))


def test_save_linear_gain_results_creates_npz_mat_and_png(tmp_path):
    image1 = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    image2 = 5.0 + 1.5 * image1
    result, workspace = compute_linear_gain(image1, image2, fit_intercept=True)
    figure = build_visualization(image1, image2, result, workspace)

    paths = save_linear_gain_results(
        output_base_path=tmp_path / "gain_results",
        figure=figure,
        image1_path="input_a.tiff",
        image2_path="input_b.tiff",
        image1=image1,
        image2=image2,
        result=result,
        workspace=workspace,
    )

    assert paths["npz"].exists()
    assert paths["mat"].exists()
    assert paths["png"].exists()

    expected_keys = {
        "image1_path",
        "image2_path",
        "image_shape",
        "fit_intercept",
        "regression_mode",
        "image1",
        "image2",
        "image1_flat",
        "image2_flat",
        "valid_mask",
        "x_valid",
        "y_valid",
        "gain",
        "intercept",
        "r_value",
        "r_squared",
        "p_value",
        "std_err",
        "y_pred",
        "residuals",
        "scatter_sample_x",
        "scatter_sample_y",
        "pixel_count_total",
        "pixel_count_valid",
        "regression_equation",
    }
    with np.load(paths["npz"], allow_pickle=True) as bundle:
        assert expected_keys.issubset(set(bundle.files))
        assert bool(bundle["fit_intercept"]) is True
        assert bundle["regression_mode"].item() == "with_intercept"

    mat_bundle = scipy.io.loadmat(paths["mat"])
    assert "gain" in mat_bundle
    assert "regression_mode" in mat_bundle

    matplotlib.pyplot.close(figure)


def test_visualization_builder_returns_figure_and_scatter_sampling_is_capped():
    x_values = np.arange(SCATTER_SAMPLE_LIMIT + 37, dtype=np.float64)
    y_values = 2.0 * x_values

    sample_x_1, sample_y_1 = select_scatter_sample(x_values, y_values)
    sample_x_2, sample_y_2 = select_scatter_sample(x_values, y_values)

    assert sample_x_1.shape == (SCATTER_SAMPLE_LIMIT,)
    assert sample_y_1.shape == (SCATTER_SAMPLE_LIMIT,)
    np.testing.assert_array_equal(sample_x_1, sample_x_2)
    np.testing.assert_array_equal(sample_y_1, sample_y_2)

    image1 = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    image2 = 2.0 * image1
    result, workspace = compute_linear_gain(image1, image2, fit_intercept=False)
    figure = build_visualization(image1, image2, result, workspace)

    assert len(figure.axes) == 4
    matplotlib.pyplot.close(figure)

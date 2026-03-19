import numpy as np
import pytest
from PIL import Image

from moment_images.main_thor_gain import (
    SystemGainOptions,
    build_options_from_gui_values,
    compute_system_gain,
    load_source_data,
    roi_rectangle_for_preview,
    run_system_gain,
    resolve_roi_bounds,
)


def _save_tiff(path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def test_compute_system_gain_matches_known_synthetic_gain():
    rng = np.random.default_rng(123)
    gain_true_e_per_adu = 5.0
    mean_corr_adu = 200.0
    variance_single_adu2 = mean_corr_adu / gain_true_e_per_adu
    std_single_adu = np.sqrt(variance_single_adu2)

    bias = np.full((200, 200), 100.0, dtype=np.float64)
    flat1 = bias + rng.normal(mean_corr_adu, std_single_adu, size=(200, 200))
    flat2 = bias + rng.normal(mean_corr_adu, std_single_adu, size=(200, 200))

    roi = resolve_roi_bounds(bias.shape, roi_size=100)
    result = compute_system_gain(bias=bias, flat1=flat1, flat2=flat2, roi=roi)

    assert result.gain_e_per_adu == pytest.approx(gain_true_e_per_adu, rel=0.12)
    assert result.gain_adu_per_e == pytest.approx(1.0 / gain_true_e_per_adu, rel=0.12)


def test_load_source_data_averages_directory_stack(tmp_path):
    arr1 = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    arr2 = np.array([[14, 24], [34, 44]], dtype=np.uint16)
    arr3 = np.array([[18, 28], [38, 48]], dtype=np.uint16)

    _save_tiff(tmp_path / "a.tiff", arr1)
    _save_tiff(tmp_path / "b.tiff", arr2)
    _save_tiff(tmp_path / "c.tiff", arr3)

    loaded = load_source_data(
        source_name="flat1",
        image_path=None,
        directory_path=str(tmp_path),
        pattern="*.tiff",
    )

    expected = (arr1.astype(np.float64) + arr2.astype(np.float64) + arr3.astype(np.float64)) / 3.0
    np.testing.assert_allclose(loaded.image, expected)
    assert loaded.frame_count == 3


def test_roi_default_center_and_override():
    default_roi = resolve_roi_bounds((220, 320), roi_size=100)
    assert default_roi == (110, 60, 210, 160)

    override_roi = resolve_roi_bounds((220, 320), roi_size=100, roi_center_x=120, roi_center_y=130)
    assert override_roi == (70, 80, 170, 180)


def test_negative_preserving_diff_matches_manual_std():
    bias = np.zeros((2, 2), dtype=np.float64)
    flat1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    flat2 = np.array([[20.0, 10.0], [40.0, 30.0]], dtype=np.float64)
    diff = flat2 - flat1
    assert np.any(diff < 0)

    result = compute_system_gain(bias=bias, flat1=flat1, flat2=flat2, roi=(0, 0, 2, 2))
    assert result.std_diff_adu == pytest.approx(float(np.std(diff)))


def test_validation_failures_for_shapes_roi_and_variance():
    with pytest.raises(ValueError, match="identical image shapes"):
        compute_system_gain(
            bias=np.zeros((4, 4)),
            flat1=np.zeros((4, 5)),
            flat2=np.zeros((4, 4)),
            roi=(0, 0, 4, 4),
        )

    with pytest.raises(ValueError, match="out of bounds"):
        resolve_roi_bounds((10, 10), roi_size=8, roi_center_x=1, roi_center_y=1)

    with pytest.raises(ValueError, match="variance must be positive"):
        compute_system_gain(
            bias=np.zeros((10, 10)),
            flat1=np.ones((10, 10)),
            flat2=np.ones((10, 10)),
            roi=(0, 0, 10, 10),
        )


def test_build_options_from_gui_values_parses_modes_and_fields():
    options = build_options_from_gui_values(
        source_modes={"bias": "dir", "flat1": "image", "flat2": "dir"},
        source_paths={"bias": "C:/data/bias", "flat1": "C:/data/f1.tiff", "flat2": "C:/data/f2"},
        pattern_text="*.tiff",
        roi_size_text="120",
        roi_center_x_text="",
        roi_center_y_text="64",
        output_json_text="results/out.json",
    )

    assert options.bias_dir == "C:/data/bias"
    assert options.bias_image is None
    assert options.flat1_image == "C:/data/f1.tiff"
    assert options.flat1_dir is None
    assert options.flat2_dir == "C:/data/f2"
    assert options.flat2_image is None
    assert options.pattern == "*.tiff"
    assert options.roi_size == 120
    assert options.roi_center_x is None
    assert options.roi_center_y == 64
    assert options.output_json == "results/out.json"


def test_roi_rectangle_for_preview_matches_resolved_roi():
    rect = roi_rectangle_for_preview((300, 500), roi_size=100, roi_center_x=260, roi_center_y=140)
    assert rect == (210, 90, 310, 190)


def test_run_system_gain_reports_bias_frame_count_warning(tmp_path):
    bias_dir = tmp_path / "bias"
    flat1_dir = tmp_path / "flat1"
    flat2_dir = tmp_path / "flat2"
    bias_dir.mkdir()
    flat1_dir.mkdir()
    flat2_dir.mkdir()

    for idx in range(3):
        _save_tiff(bias_dir / f"bias_{idx:02d}.tiff", np.full((120, 120), 100 + idx, dtype=np.uint16))

    rng = np.random.default_rng(5)
    for idx in range(4):
        flat1_noise = rng.normal(loc=0.0, scale=3.0, size=(120, 120))
        flat2_noise = rng.normal(loc=0.0, scale=3.0, size=(120, 120))
        flat1_image = np.clip(300 + flat1_noise + idx, 0, 65535).astype(np.uint16)
        flat2_image = np.clip(300 + flat2_noise + idx, 0, 65535).astype(np.uint16)
        _save_tiff(flat1_dir / f"flat1_{idx:02d}.tiff", flat1_image)
        _save_tiff(flat2_dir / f"flat2_{idx:02d}.tiff", flat2_image)

    options = SystemGainOptions(
        bias_dir=str(bias_dir),
        flat1_dir=str(flat1_dir),
        flat2_dir=str(flat2_dir),
        pattern="*.tiff",
        roi_size=100,
    )
    result, warnings = run_system_gain(options)
    assert result.bias_frame_count == 3
    assert result.flat1_frame_count == 4
    assert result.flat2_frame_count == 4
    assert warnings
    assert "Expected ~100 bias frames, found 3" in warnings[0]

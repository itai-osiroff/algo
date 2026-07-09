import matplotlib

matplotlib.use("Agg")

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytest

import moment_images.validate_rectification as validate_rectification
from moment_images.validate_rectification import (
    AlignmentPointSet,
    AlignmentResult,
    VALIDATION_MODE_FIT,
    VALIDATION_MODE_PRESERVE,
    ValidationImageResult,
    ValidationOptions,
    _apply_intensity_window,
    _estimate_similarity_reference_to_input,
    _save_workspace_data,
    _scale_image,
    auto_align_images,
    compare_images,
    load_rectified_reference,
    prepare_images_for_comparison,
)


def _synthetic_reference() -> np.ndarray:
    ref = np.zeros((120, 140), dtype=np.float32)
    cv.circle(ref, (35, 30), 10, 180, -1)
    cv.rectangle(ref, (80, 70), (120, 110), 220, -1)
    cv.line(ref, (0, 119), (139, 0), 90, 2)
    return cv.GaussianBlur(ref, (5, 5), 0)


def test_load_rectified_reference_from_npy_stack(tmp_path):
    rectified_views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    path = tmp_path / "rectified_views.npy"
    np.save(path, rectified_views)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, rectified_views.mean(axis=(0, 1)))


def test_load_rectified_reference_from_npz_rectified_views(tmp_path):
    rectified_views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    path = tmp_path / "rectified_views.npz"
    np.savez_compressed(path, rectified_views=rectified_views)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, rectified_views.mean(axis=(0, 1)))


def test_load_rectified_reference_prefers_average_image(tmp_path):
    average = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "workspace.npz"
    np.savez_compressed(path, average_rectified_image=average)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, average)


def test_manual_scaling_preserved_in_preserve_mode():
    ref = _synthetic_reference()
    target = cv.resize(ref, (40, 30), interpolation=cv.INTER_LINEAR)
    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=False)

    comp = prepare_images_for_comparison(ref, target, options=options)

    assert comp.as_is_input.shape == target.shape
    assert comp.registered_input.shape == target.shape
    assert comp.registered_input.shape != ref.shape


def test_fit_reference_mode_resizes_deterministically():
    ref = _synthetic_reference()
    target = cv.resize(ref, (40, 30), interpolation=cv.INTER_LINEAR)
    options = ValidationOptions(mode=VALIDATION_MODE_FIT, auto_align=False)

    comp = prepare_images_for_comparison(ref, target, options=options)

    assert comp.as_is_input.shape == target.shape
    assert comp.registered_input.shape == target.shape
    assert comp.registered_reference.shape == target.shape


def test_fit_reference_alignment_preserves_processed_input_geometry():
    ref = _synthetic_reference()
    target = cv.resize(ref, (72, 48), interpolation=cv.INTER_LINEAR)
    options = ValidationOptions(mode=VALIDATION_MODE_FIT, auto_align=True)

    aligned = auto_align_images(ref, target, options=options)

    assert aligned.status == "aligned_fit_reference"
    assert aligned.aligned_image.shape == target.shape
    assert aligned.roi_reference == (0, 0, target.shape[1], target.shape[0])
    assert aligned.roi_input == (0, 0, target.shape[1], target.shape[0])


def test_joint_normalization_diff_exposes_offset():
    ref = np.tile(np.linspace(0, 100, 120, dtype=np.float32), (80, 1))
    target = ref + 20.0
    options = ValidationOptions(
        mode=VALIDATION_MODE_PRESERVE,
        auto_align=False,
        joint_norm_percentiles=(0.0, 100.0),
    )

    comp = prepare_images_for_comparison(ref, target, options=options)

    assert comp.match_found
    assert comp.diff_native_image is not None
    assert comp.diff_normalized_image is not None
    np.testing.assert_allclose(comp.diff_native_image, np.full_like(comp.diff_native_image, 20.0), rtol=0.0, atol=1e-5)
    assert float(np.mean(comp.diff_normalized_image)) > 0.1
    assert float(np.nanmin(comp.diff_normalized_image)) >= 0.0
    assert float(np.nanmax(comp.diff_normalized_image)) <= 1.0


def test_low_confidence_match_suppresses_diff_and_overlay():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(80, 80)).astype(np.float32)
    target = rng.normal(size=(30, 30)).astype(np.float32)
    options = ValidationOptions(
        mode=VALIDATION_MODE_PRESERVE,
        auto_align=False,
        template_score_threshold=0.95,
    )

    comp = prepare_images_for_comparison(ref, target, options=options)

    assert not comp.match_found
    assert comp.diff_native_image is None
    assert comp.diff_normalized_image is None
    assert comp.overlay_native_image is None
    assert comp.overlay_normalized_image is None
    assert comp.status in {"low_confidence_match", "no_containment_match"}


def test_anisotropic_scale_translation_alignment_finds_expected_roi():
    ref = _synthetic_reference()
    patch = ref[24:104, 28:118]  # h=80, w=90
    distorted = cv.resize(patch, (72, 64), interpolation=cv.INTER_LINEAR)
    corrected = cv.resize(distorted, (90, 80), interpolation=cv.INTER_LINEAR)

    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=True)
    aligned = auto_align_images(ref, corrected, options=options)

    assert aligned.match_found
    assert aligned.roi_reference is not None
    x, y, w, h = aligned.roi_reference
    assert w == corrected.shape[1]
    assert h == corrected.shape[0]
    assert abs(x - 28) <= 3
    assert abs(y - 24) <= 3


def test_estimate_similarity_from_single_point_returns_translation():
    points = AlignmentPointSet(
        reference_points=((10.0, 15.0),),
        input_points=((13.5, 9.0),),
    )

    matrix, rmse = _estimate_similarity_reference_to_input(points)

    np.testing.assert_allclose(
        matrix,
        np.array([[1.0, 0.0, 3.5], [0.0, 1.0, -6.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_estimate_similarity_from_three_points_recovers_rotation_scale_translation():
    angle = np.deg2rad(30.0)
    scale = 1.25
    tx = 7.0
    ty = -4.5
    expected = np.array(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), tx],
            [scale * np.sin(angle), scale * np.cos(angle), ty],
        ],
        dtype=np.float32,
    )
    reference_points = np.array(
        [
            [10.0, 15.0],
            [40.0, 12.0],
            [20.0, 55.0],
        ],
        dtype=np.float32,
    )
    input_points_h = np.column_stack([reference_points, np.ones(reference_points.shape[0], dtype=np.float32)])
    input_points = input_points_h @ expected.T
    points = AlignmentPointSet(
        reference_points=tuple(map(tuple, reference_points.tolist())),
        input_points=tuple(map(tuple, input_points.tolist())),
    )

    matrix, rmse = _estimate_similarity_reference_to_input(points)

    np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-5)
    assert rmse == pytest.approx(0.0, abs=1e-5)


def test_estimate_similarity_rejects_duplicate_points():
    points = AlignmentPointSet(
        reference_points=((5.0, 5.0), (5.0, 5.0)),
        input_points=((10.0, 10.0), (11.0, 10.0)),
    )

    with pytest.raises(ValueError, match="zero variance"):
        _estimate_similarity_reference_to_input(points)


def test_point_guided_alignment_warps_input_to_reference_grid_and_masks_border():
    ref = _synthetic_reference()
    point_matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    target = cv.warpAffine(
        ref,
        point_matrix,
        (ref.shape[1], ref.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    points = AlignmentPointSet(
        reference_points=((35.0, 30.0),),
        input_points=((40.0, 27.0),),
    )
    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=False)

    alignment = auto_align_images(ref, target, options=options, point_pairs=points)
    comp = prepare_images_for_comparison(ref, target, options=options, alignment_result=alignment)

    assert alignment.match_found
    assert alignment.alignment_model == "point_similarity"
    assert alignment.aligned_image.shape == ref.shape
    assert alignment.valid_mask is not None
    assert comp.match_found
    assert comp.registered_valid_mask is not None
    assert not bool(comp.registered_valid_mask[10, -1])
    assert comp.diff_native_image is not None
    assert np.isnan(comp.diff_native_image[10, -1])
    valid_diff = comp.diff_native_image[comp.registered_valid_mask]
    assert float(np.nanmean(valid_diff)) < 1e-4


def test_point_guided_alignment_keeps_point_result_when_residual_ecc_fails(monkeypatch):
    ref = _synthetic_reference()
    target = np.roll(ref, shift=2, axis=1)
    points = AlignmentPointSet(
        reference_points=((35.0, 30.0),),
        input_points=((37.0, 30.0),),
    )
    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=True)

    def fail_ecc(_ref_img, _target_img):
        raise cv.error("forced failure")

    monkeypatch.setattr(validate_rectification, "_ecc_align_same_size", fail_ecc)

    alignment = auto_align_images(
        ref,
        target,
        options=options,
        point_pairs=points,
        run_ecc_after_points=True,
    )

    assert alignment.status.startswith("point_aligned_residual_ecc_failed")
    assert alignment.alignment_model == "point_similarity"
    assert alignment.residual_warp_matrix is None
    np.testing.assert_allclose(alignment.point_transform_matrix, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]))


def test_differing_shapes_still_produce_figures_without_crash():
    rng = np.random.default_rng(7)
    ref = rng.uniform(0, 255, size=(64, 64)).astype(np.float32)
    target = rng.uniform(0, 255, size=(90, 40)).astype(np.float32)

    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=False)
    figures = compare_images(ref, target, "Synthetic", options=options)

    try:
        assert len(figures) == 6
    finally:
        for fig in figures:
            plt.close(fig)


def test_alignment_failure_returns_status_without_silent_change():
    ref = np.zeros((64, 64), dtype=np.float32)
    target = np.full((64, 64), np.nan, dtype=np.float32)

    options = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=True)
    aligned = auto_align_images(ref, target, options=options)

    assert aligned.status.startswith("ecc_failed")
    assert np.isnan(aligned.aligned_image).all()


def _expected_scale_image(
    src: np.ndarray,
    scale_x: float,
    scale_y: float,
    *,
    preblur_before_resize: bool,
    preserve_flux_sum: bool = False,
) -> np.ndarray:
    def _expected_gaussian_downsample_kernel_1d(downsample_factor: float) -> np.ndarray:
        safe_downsample = max(float(downsample_factor), 1e-12)
        if safe_downsample >= 1.0:
            return np.array([1.0], dtype=np.float32)

        sigma = (1.0 / safe_downsample) / 2.0
        radius = max(1, int(np.ceil(3.0 * sigma)))
        coords = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_sum = float(np.sum(kernel))
        if not np.isfinite(kernel_sum) or kernel_sum <= 0.0:
            return np.array([1.0 / safe_downsample], dtype=np.float32)

        kernel = kernel / kernel_sum
        kernel = kernel * (1.0 / safe_downsample)
        return np.asarray(kernel, dtype=np.float32)

    src_arr = np.asarray(src, dtype=np.float32)
    src_h, src_w = src_arr.shape[:2]
    scaled_w = max(1, int(round(src_w * float(scale_x))))
    scaled_h = max(1, int(round(src_h * float(scale_y))))

    resize_source = src_arr
    if preblur_before_resize:
        safe_scale_x = max(float(scale_x), 1e-12)
        safe_scale_y = max(float(scale_y), 1e-12)
        downsample_x = safe_scale_x if safe_scale_x < 1.0 else 1.0
        downsample_y = safe_scale_y if safe_scale_y < 1.0 else 1.0
        if downsample_x < 1.0 or downsample_y < 1.0:
            kernel_x = _expected_gaussian_downsample_kernel_1d(downsample_x)
            kernel_y = _expected_gaussian_downsample_kernel_1d(downsample_y)
            resize_source = cv.sepFilter2D(
                src_arr,
                ddepth=-1,
                kernelX=kernel_x,
                kernelY=kernel_y,
                borderType=cv.BORDER_REFLECT_101,
            )

    expected = np.asarray(cv.resize(resize_source, (scaled_w, scaled_h), interpolation=cv.INTER_LINEAR), dtype=np.float32)

    did_downsample = scaled_w < src_w or scaled_h < src_h
    if preserve_flux_sum and did_downsample:
        source_finite = np.isfinite(src_arr)
        expected_finite = np.isfinite(expected)
        if np.any(source_finite) and np.any(expected_finite):
            source_sum = float(np.sum(src_arr[source_finite], dtype=np.float64))
            expected_sum = float(np.sum(expected[expected_finite], dtype=np.float64))
            if np.isfinite(source_sum) and np.isfinite(expected_sum) and abs(expected_sum) > 1e-12:
                expected = expected * np.float32(source_sum / expected_sum)
    return expected


def test_scale_image_without_preblur_matches_cv_resize():
    src = np.linspace(-7.5, 23.75, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=False)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=False)

    assert actual.shape == (4, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_preblur_downscale_isotropic_matches_gaussian_then_resize():
    src = np.linspace(-11.0, 31.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=True)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=True)

    assert actual.shape == (4, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_preblur_downscale_anisotropic_matches_gaussian_then_resize():
    src = np.linspace(4.0, 104.0, 80, dtype=np.float32).reshape(8, 10)
    actual = _scale_image(src, 0.5, 0.25, preblur_before_resize=True)
    expected = _expected_scale_image(src, 0.5, 0.25, preblur_before_resize=True)

    assert actual.shape == (2, 5)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_preblur_upscale_matches_cv_resize():
    src = np.linspace(3.0, 36.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 2.0, 1.5, preblur_before_resize=True)
    expected = _expected_scale_image(src, 2.0, 1.5, preblur_before_resize=True)

    assert actual.shape == (12, 12)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_preblur_downsample_isotropic_without_flux_preserves_sum():
    src = np.full((20, 20), 4.0, dtype=np.float32)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=True, preserve_flux_sum=False)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=True, preserve_flux_sum=False)

    assert actual.shape == (10, 10)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(actual, dtype=np.float64),
        np.sum(src, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_scale_image_preblur_downsample_anisotropic_without_flux_preserves_sum():
    src = np.full((20, 16), 2.0, dtype=np.float32)
    actual = _scale_image(src, 0.5, 0.25, preblur_before_resize=True, preserve_flux_sum=False)
    expected = _expected_scale_image(src, 0.5, 0.25, preblur_before_resize=True, preserve_flux_sum=False)

    assert actual.shape == (5, 8)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(actual, dtype=np.float64),
        np.sum(src, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_scale_image_identity_matches_cv_resize_with_nan():
    src = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, np.nan, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    actual = _scale_image(src, 1.0, 1.0, preblur_before_resize=False)
    expected = _expected_scale_image(src, 1.0, 1.0, preblur_before_resize=False)

    assert actual.shape == src.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=True)


def test_scale_image_flux_preservation_downsample_conserves_sum():
    src = np.linspace(2.0, 98.0, 80, dtype=np.float32).reshape(8, 10)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=False, preserve_flux_sum=True)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=False, preserve_flux_sum=True)

    assert actual.shape == (4, 5)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(actual, dtype=np.float64),
        np.sum(src, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_scale_image_flux_preservation_anisotropic_preblur_conserves_sum():
    src = np.linspace(-5.0, 195.0, 240, dtype=np.float32).reshape(12, 20)
    actual = _scale_image(src, 0.5, 0.25, preblur_before_resize=True, preserve_flux_sum=True)
    expected = _expected_scale_image(src, 0.5, 0.25, preblur_before_resize=True, preserve_flux_sum=True)

    assert actual.shape == (3, 10)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(actual, dtype=np.float64),
        np.sum(src, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_scale_image_flux_preservation_upscale_matches_no_flux_behavior():
    src = np.linspace(3.0, 36.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 2.0, 1.5, preblur_before_resize=True, preserve_flux_sum=True)
    expected = _expected_scale_image(src, 2.0, 1.5, preblur_before_resize=True, preserve_flux_sum=False)

    assert actual.shape == (12, 12)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_flux_preservation_zero_sum_downsample_stays_finite():
    src = np.zeros((8, 6), dtype=np.float32)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=False, preserve_flux_sum=True)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=False, preserve_flux_sum=True)

    assert actual.shape == (4, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_apply_intensity_window_clips_without_rescaling():
    src = np.array([[-10.0, 0.0, 2.5, 10.0, 50.0]], dtype=np.float32)
    actual = _apply_intensity_window(src, 0.0, 10.0)
    expected = np.array([[0.0, 0.0, 2.5, 10.0, 10.0]], dtype=np.float32)

    np.testing.assert_allclose(actual, expected)
    assert float(np.max(actual)) == 10.0


def test_save_workspace_data_persists_resize_processing_flags(tmp_path):
    rectified = np.zeros((8, 8), dtype=np.float32)
    val_path = tmp_path / "val.png"
    thor_path = tmp_path / "thor.png"
    save_path = tmp_path / "validation_workspace.npz"

    val_u8 = np.full((8, 8), 80, dtype=np.uint8)
    thor_u8 = np.full((8, 8), 140, dtype=np.uint8)
    assert cv.imwrite(str(val_path), val_u8)
    assert cv.imwrite(str(thor_path), thor_u8)

    opts = ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=False)
    val_result = ValidationImageResult(
        processed_image=rectified.copy(),
        options=opts,
        alignment_result=None,
        scale_x=1.0,
        scale_y=1.0,
        preblur_before_resize=True,
        preserve_flux_sum=True,
        intensity_min=0.0,
        intensity_max=255.0,
        crop_box=None,
        rotation_deg=0,
        flip_h=False,
        flip_v=False,
    )
    thor_result = ValidationImageResult(
        processed_image=rectified.copy(),
        options=opts,
        alignment_result=None,
        scale_x=1.0,
        scale_y=1.0,
        preblur_before_resize=False,
        preserve_flux_sum=False,
        intensity_min=0.0,
        intensity_max=255.0,
        crop_box=None,
        rotation_deg=0,
        flip_h=False,
        flip_v=False,
    )

    _save_workspace_data(
        save_path=str(save_path),
        rectified_img=rectified,
        val_path=str(val_path),
        thor_path=str(thor_path),
        val_result=val_result,
        thor_result=thor_result,
    )

    with np.load(save_path) as workspace:
        assert "validation_preblur_before_resize" in workspace
        assert "validation_preserve_flux_sum" in workspace
        assert "thorcam_preblur_before_resize" in workspace
        assert "thorcam_preserve_flux_sum" in workspace
        assert bool(workspace["validation_preblur_before_resize"]) is True
        assert bool(workspace["validation_preserve_flux_sum"]) is True
        assert bool(workspace["thorcam_preblur_before_resize"]) is False
        assert bool(workspace["thorcam_preserve_flux_sum"]) is False
        assert workspace["validation_alignment_point_reference_points"].shape == (0, 2)
        assert workspace["validation_alignment_point_input_points"].shape == (0, 2)
        assert np.isnan(workspace["validation_alignment_point_transform_matrix"]).all()
        assert np.isnan(workspace["validation_alignment_residual_warp_matrix"]).all()
        assert np.isnan(workspace["validation_alignment_final_warp_matrix"]).all()


def test_save_workspace_data_persists_point_alignment_metadata(tmp_path):
    rectified = np.zeros((8, 8), dtype=np.float32)
    val_path = tmp_path / "val.png"
    thor_path = tmp_path / "thor.png"
    save_path = tmp_path / "validation_workspace.npz"

    assert cv.imwrite(str(val_path), np.full((8, 8), 80, dtype=np.uint8))
    assert cv.imwrite(str(thor_path), np.full((8, 8), 140, dtype=np.uint8))

    points = AlignmentPointSet(
        reference_points=((1.0, 2.0), (5.0, 2.0)),
        input_points=((2.0, 3.0), (6.0, 3.0)),
    )
    point_matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    residual_matrix = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, -0.5]], dtype=np.float32)
    final_matrix = np.array([[1.0, 0.0, 1.25], [0.0, 1.0, 0.5]], dtype=np.float32)
    alignment = AlignmentResult(
        aligned_image=rectified.copy(),
        status="point_aligned_with_residual_ecc",
        mode=VALIDATION_MODE_PRESERVE,
        match_found=True,
        warp_matrix=final_matrix,
        roi_reference=(0, 0, 8, 8),
        roi_input=(0, 0, 8, 8),
        alignment_model="point_similarity+translation_ecc",
        point_pairs=points,
        point_transform_matrix=point_matrix,
        residual_warp_matrix=residual_matrix,
        point_rmse=0.125,
    )
    val_result = ValidationImageResult(
        processed_image=rectified.copy(),
        options=ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=True),
        alignment_result=alignment,
        scale_x=1.0,
        scale_y=1.0,
        preblur_before_resize=True,
        preserve_flux_sum=True,
        intensity_min=0.0,
        intensity_max=255.0,
        crop_box=None,
        rotation_deg=0,
        flip_h=False,
        flip_v=False,
        alignment_points=points,
        point_alignment_enabled=True,
        point_alignment_residual_ecc=True,
    )
    thor_result = ValidationImageResult(
        processed_image=rectified.copy(),
        options=ValidationOptions(mode=VALIDATION_MODE_PRESERVE, auto_align=False),
        alignment_result=None,
        scale_x=1.0,
        scale_y=1.0,
        preblur_before_resize=True,
        preserve_flux_sum=True,
        intensity_min=0.0,
        intensity_max=255.0,
        crop_box=None,
        rotation_deg=0,
        flip_h=False,
        flip_v=False,
    )

    _save_workspace_data(
        save_path=str(save_path),
        rectified_img=rectified,
        val_path=str(val_path),
        thor_path=str(thor_path),
        val_result=val_result,
        thor_result=thor_result,
    )

    with np.load(save_path) as workspace:
        np.testing.assert_allclose(workspace["validation_alignment_point_reference_points"], np.array(points.reference_points))
        np.testing.assert_allclose(workspace["validation_alignment_point_input_points"], np.array(points.input_points))
        np.testing.assert_allclose(workspace["validation_alignment_point_transform_matrix"], point_matrix)
        np.testing.assert_allclose(workspace["validation_alignment_residual_warp_matrix"], residual_matrix)
        np.testing.assert_allclose(workspace["validation_alignment_final_warp_matrix"], final_matrix)
        assert workspace["validation_alignment_model"].item() == "point_similarity+translation_ecc"
        assert float(workspace["validation_alignment_point_rmse"]) == pytest.approx(0.125)
        assert bool(workspace["validation_point_alignment_enabled"]) is True
        assert bool(workspace["validation_point_alignment_residual_ecc"]) is True

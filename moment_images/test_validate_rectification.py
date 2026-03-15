import matplotlib

matplotlib.use("Agg")

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from moment_images.validate_rectification import (
    VALIDATION_MODE_FIT,
    VALIDATION_MODE_PRESERVE,
    ValidationImageResult,
    ValidationOptions,
    _apply_intensity_window,
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

    assert comp.registered_input.shape == ref.shape
    assert comp.registered_reference.shape == ref.shape


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
    src_arr = np.asarray(src, dtype=np.float32)
    src_h, src_w = src_arr.shape[:2]
    scaled_w = max(1, int(round(src_w * float(scale_x))))
    scaled_h = max(1, int(round(src_h * float(scale_y))))

    resize_source = src_arr
    if preblur_before_resize:
        safe_scale_x = max(float(scale_x), 1e-12)
        safe_scale_y = max(float(scale_y), 1e-12)
        kernel_w = max(1, int(np.ceil(1.0 / safe_scale_x)))
        kernel_h = max(1, int(np.ceil(1.0 / safe_scale_y)))
        kernel_w = min(kernel_w, src_w)
        kernel_h = min(kernel_h, src_h)
        if kernel_w > 1 or kernel_h > 1:
            resize_source = cv.blur(src_arr, (kernel_w, kernel_h))

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


def test_scale_image_preblur_downscale_isotropic_matches_blur_then_resize():
    src = np.linspace(-11.0, 31.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=True)
    expected = _expected_scale_image(src, 0.5, 0.5, preblur_before_resize=True)

    assert actual.shape == (4, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_scale_image_preblur_downscale_anisotropic_matches_blur_then_resize():
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

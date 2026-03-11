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
    assert comp.diff_image is not None
    assert float(np.mean(comp.diff_image)) > 0.1


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
    assert comp.diff_image is None
    assert comp.overlay_image is None
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
        assert len(figures) == 4
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


def _assert_finite_range_preserved(src: np.ndarray, out: np.ndarray) -> None:
    src_vals = np.asarray(src, dtype=np.float32)
    out_vals = np.asarray(out, dtype=np.float32)
    src_finite = src_vals[np.isfinite(src_vals)]
    out_finite = out_vals[np.isfinite(out_vals)]

    assert src_finite.size > 0
    assert out_finite.size > 0
    np.testing.assert_allclose(float(np.min(out_finite)), float(np.min(src_finite)), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(float(np.max(out_finite)), float(np.max(src_finite)), rtol=0.0, atol=1e-5)


def test_scale_image_without_preblur_preserves_finite_range():
    src = np.linspace(-7.5, 23.75, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=False)

    assert actual.shape == (4, 3)
    _assert_finite_range_preserved(src, actual)


def test_scale_image_preblur_downscale_isotropic_preserves_finite_range():
    src = np.linspace(-11.0, 31.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 0.5, 0.5, preblur_before_resize=True)

    assert actual.shape == (4, 3)
    _assert_finite_range_preserved(src, actual)


def test_scale_image_preblur_downscale_anisotropic_preserves_finite_range():
    src = np.linspace(4.0, 104.0, 80, dtype=np.float32).reshape(8, 10)
    actual = _scale_image(src, 0.5, 0.25, preblur_before_resize=True)

    assert actual.shape == (2, 5)
    _assert_finite_range_preserved(src, actual)


def test_scale_image_preblur_upscale_preserves_finite_range():
    src = np.linspace(3.0, 36.0, 48, dtype=np.float32).reshape(8, 6)
    actual = _scale_image(src, 2.0, 1.5, preblur_before_resize=True)

    assert actual.shape == (12, 12)
    _assert_finite_range_preserved(src, actual)


def test_scale_image_preserves_nan_and_finite_range():
    src = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, np.nan, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    actual = _scale_image(src, 1.0, 1.0, preblur_before_resize=False)

    assert actual.shape == src.shape
    assert np.isnan(actual[1, 1])
    _assert_finite_range_preserved(src, actual)


def test_apply_intensity_window_clips_without_rescaling():
    src = np.array([[-10.0, 0.0, 2.5, 10.0, 50.0]], dtype=np.float32)
    actual = _apply_intensity_window(src, 0.0, 10.0)
    expected = np.array([[0.0, 0.0, 2.5, 10.0, 10.0]], dtype=np.float32)

    np.testing.assert_allclose(actual, expected)
    assert float(np.max(actual)) == 10.0


def test_save_workspace_data_persists_preblur_flags(tmp_path):
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
        assert "thorcam_preblur_before_resize" in workspace
        assert bool(workspace["validation_preblur_before_resize"]) is True
        assert bool(workspace["thorcam_preblur_before_resize"]) is False

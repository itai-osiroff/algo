import matplotlib

matplotlib.use("Agg")

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from moment_images.validate_rectification import (
    VALIDATION_MODE_FIT,
    VALIDATION_MODE_PRESERVE,
    ValidationOptions,
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

import cv2 as cv
import numpy as np

from moment_images.bead_quality_diagnostics import (
    classify_bead_disagreement,
    compute_bead_quality_metrics,
)


def _bead_scene(shape: tuple[int, int] = (96, 96)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    for x, y, radius, value in [
        (22, 24, 3, 1.0),
        (68, 28, 4, 0.9),
        (45, 64, 3, 0.8),
        (75, 72, 3, 0.7),
    ]:
        cv.circle(image, (x, y), radius, value, -1)
    return cv.GaussianBlur(image, (0, 0), 0.8)


def _ringing_candidate(reference: np.ndarray) -> np.ndarray:
    ring = cv.GaussianBlur(reference, (0, 0), 2.4) - cv.GaussianBlur(reference, (0, 0), 4.8)
    return np.asarray(np.clip(reference + 1.8 * np.abs(ring), 0.0, 1.5), dtype=np.float32)


def test_bead_quality_metrics_prefer_compact_candidate_over_blurred_candidate():
    validation = _bead_scene()
    compact = validation.copy()
    blurred = cv.GaussianBlur(validation, (0, 0), 4.0)

    compact_metrics = compute_bead_quality_metrics(validation, compact)
    blurred_metrics = compute_bead_quality_metrics(validation, blurred)

    assert compact_metrics.validation_bead_count >= 4
    assert compact_metrics.evaluated_bead_count == compact_metrics.validation_bead_count
    assert compact_metrics.candidate_width_mean_px < blurred_metrics.candidate_width_mean_px
    assert compact_metrics.width_excess_mean_px < blurred_metrics.width_excess_mean_px
    assert compact_metrics.compactness_score > blurred_metrics.compactness_score
    assert compact_metrics.support_compactness_score > blurred_metrics.support_compactness_score


def test_bead_disagreement_classification_separates_compactness_from_ringing():
    validation = _bead_scene()
    compact = compute_bead_quality_metrics(validation, validation)
    blurred = compute_bead_quality_metrics(validation, cv.GaussianBlur(validation, (0, 0), 4.0))
    ringing = compute_bead_quality_metrics(validation, _ringing_candidate(validation))

    assert classify_bead_disagreement(oracle_metrics=blurred, proxy_metrics=compact) == "proxy_sharper_and_compacter"
    assert classify_bead_disagreement(oracle_metrics=blurred, proxy_metrics=ringing) == "proxy_sharper_but_noisier"
    assert (
        classify_bead_disagreement(
            oracle_metrics=compute_bead_quality_metrics(np.zeros_like(validation), validation),
            proxy_metrics=compact,
        )
        == "insufficient_bead_evidence"
    )

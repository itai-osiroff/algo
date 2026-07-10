import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest
from PIL import Image

from moment_images.test_sweep_visualization import _create_modern_fixture
import moment_images.trio_benchmark as trio_benchmark
from moment_images.trio_benchmark import (
    PROFILE_NAME,
    BenchmarkConfig,
    apply_anchor_registration,
    compute_validation_metrics,
    estimate_similarity_registration,
    parse_scale_search,
    run_benchmark,
)


def _synthetic_beads(shape: tuple[int, int] = (160, 160)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    for x, y, radius, value in [
        (35, 38, 4, 1.0),
        (92, 34, 3, 0.9),
        (121, 88, 5, 0.85),
        (54, 118, 4, 0.75),
        (104, 125, 3, 0.65),
    ]:
        cv.circle(image, (x, y), radius, value, -1)
    return cv.GaussianBlur(image, (0, 0), 0.8)


def _single_synthetic_bead(shape: tuple[int, int] = (120, 120)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    cv.circle(image, (72, 51), 4, 1.0, -1)
    return cv.GaussianBlur(image, (0, 0), 0.9)


def _warp_reference_to_moving(
    reference: np.ndarray,
    moving_shape: tuple[int, int],
    moving_to_reference: np.ndarray,
) -> np.ndarray:
    reference_to_moving = cv.invertAffineTransform(np.asarray(moving_to_reference, dtype=np.float32))
    return cv.warpAffine(
        reference,
        reference_to_moving,
        moving_shape[::-1],
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.float32)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if high <= low:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
        scaled = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)
    Image.fromarray(scaled).save(path)


def _manual_similarity_matrix(scale: float, rotation_deg: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta) * scale
    sin_t = np.sin(theta) * scale
    return np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=np.float64)


def _write_manual_annotations(path: Path, annotations: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "annotations": annotations}), encoding="utf-8")


def _write_validation_transform_workspace(
    path: Path,
    *,
    rectified: np.ndarray,
    validation_processed: np.ndarray,
    thorcam_processed: np.ndarray,
    validation_matrix: np.ndarray,
    thorcam_matrix: np.ndarray,
    validation_alignment_status: str | None = "point_aligned",
    thorcam_alignment_status: str | None = "aligned_preserve_same_shape",
    validation_alignment_model: str | None = "point_similarity",
    thorcam_alignment_model: str | None = "translation_ecc",
    validation_alignment_output_shape: tuple[int, int] | None = None,
    thorcam_alignment_output_shape: tuple[int, int] | None = None,
    include_alignment_output_shape: bool = True,
) -> None:
    payload = {
        "rectified_image": np.asarray(rectified, dtype=np.float32),
        "validation_processed": np.asarray(validation_processed, dtype=np.float32),
        "thorcam_processed": np.asarray(thorcam_processed, dtype=np.float32),
        "validation_alignment_final_warp_matrix": np.asarray(validation_matrix, dtype=np.float32),
        "thorcam_alignment_final_warp_matrix": np.asarray(thorcam_matrix, dtype=np.float32),
    }
    if validation_alignment_status is not None:
        payload["validation_alignment_status"] = np.array(validation_alignment_status)
    if thorcam_alignment_status is not None:
        payload["thorcam_alignment_status"] = np.array(thorcam_alignment_status)
    if validation_alignment_model is not None:
        payload["validation_alignment_model"] = np.array(validation_alignment_model)
    if thorcam_alignment_model is not None:
        payload["thorcam_alignment_model"] = np.array(thorcam_alignment_model)
    if include_alignment_output_shape:
        validation_shape = validation_alignment_output_shape or tuple(rectified.shape[:2])
        thorcam_shape = thorcam_alignment_output_shape or tuple(rectified.shape[:2])
        payload["validation_alignment_output_shape"] = np.array(validation_shape, dtype=np.int32)
        payload["thorcam_alignment_output_shape"] = np.array(thorcam_shape, dtype=np.int32)
    np.savez_compressed(path, **payload)


def _source_from_grid_image(reference: np.ndarray, source_shape: tuple[int, int], matrix: np.ndarray) -> np.ndarray:
    source = np.zeros(source_shape, dtype=np.float32)
    reference_to_source = np.asarray(matrix, dtype=np.float32)
    corners = np.array([[0.0, 0.0], [float(reference.shape[1] - 1), float(reference.shape[0] - 1)]], dtype=np.float32)
    mapped = trio_benchmark._transform_points(reference_to_source, corners)
    x0 = int(round(mapped[0, 0]))
    y0 = int(round(mapped[0, 1]))
    x1 = x0 + reference.shape[1]
    y1 = y0 + reference.shape[0]
    source[y0:y1, x0:x1] = reference
    return source


def _manual_annotation(
    *,
    scene_id: str,
    dataset_id: str,
    source_kind: str,
    calibration_group: str,
    reference_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    matrix: np.ndarray,
    anchor_row_id: str | None = None,
    point_count: int = 3,
) -> dict:
    moving_points = np.array([[18.0, 17.0], [72.0, 24.0], [38.0, 82.0]], dtype=np.float64)[:point_count]
    reference_points = trio_benchmark._transform_points(matrix, moving_points)
    payload = {
        "scene_id": scene_id,
        "dataset_id": dataset_id,
        "source_kind": source_kind,
        "calibration_group": calibration_group,
        "reference_shape": list(reference_shape),
        "moving_shape": list(moving_shape),
        "points": [
            {"moving": [float(mx), float(my)], "reference": [float(rx), float(ry)]}
            for (mx, my), (rx, ry) in zip(moving_points, reference_points)
        ],
    }
    if anchor_row_id is not None:
        payload["anchor_row_id"] = anchor_row_id
    return payload


def test_parse_scale_search_returns_inclusive_grid():
    values = parse_scale_search("0.90:1.10:5")

    np.testing.assert_allclose(values, [0.90, 0.95, 1.00, 1.05, 1.10])


def test_similarity_registration_recovers_no_shear_transform():
    reference = _synthetic_beads()
    center = (reference.shape[1] / 2.0, reference.shape[0] / 2.0)
    ref_to_moving = cv.getRotationMatrix2D(center, -4.0, 0.94)
    ref_to_moving[:, 2] += np.array([7.0, -5.0], dtype=np.float32)
    moving = cv.warpAffine(reference, ref_to_moving, reference.shape[::-1], flags=cv.INTER_LINEAR)

    result = estimate_similarity_registration(
        reference,
        moving,
        scale_values=parse_scale_search("0.95:1.15:9"),
    )

    assert result.status == "aligned"
    assert abs(result.scale - (1.0 / 0.94)) <= 0.08
    assert abs(abs(result.rotation_deg) - 4.0) <= 2.0
    assert result.aligned_image.shape == reference.shape
    assert float(np.mean(np.abs(result.aligned_image - reference))) < 0.08
    linear = result.forward_matrix[:, :2]
    assert abs(float(np.dot(linear[:, 0], linear[:, 1]))) < 1e-3
    np.testing.assert_allclose(
        np.linalg.norm(linear[:, 0]),
        np.linalg.norm(linear[:, 1]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_manual_similarity_fit_recovers_no_shear_transform():
    expected = _manual_similarity_matrix(1.23, 14.0, 8.5, -6.0)
    moving_points = np.array([[8.0, 9.0], [44.0, 13.0], [14.0, 51.0], [62.0, 58.0]], dtype=np.float64)
    reference_points = trio_benchmark._transform_points(expected, moving_points)

    fit = trio_benchmark.fit_manual_similarity_transform(moving_points, reference_points)

    np.testing.assert_allclose(fit.matrix, expected, atol=1e-8)
    assert fit.point_count == 4
    assert fit.rms_error_px < 1e-8
    linear = fit.matrix[:, :2]
    assert abs(float(np.dot(linear[:, 0], linear[:, 1]))) < 1e-10
    np.testing.assert_allclose(
        np.linalg.norm(linear[:, 0]),
        np.linalg.norm(linear[:, 1]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_geometric_registration_recovers_multibead_different_fov_transform():
    reference = _synthetic_beads((190, 190))
    moving_shape = (150, 150)
    center = (moving_shape[1] / 2.0, moving_shape[0] / 2.0)
    moving_to_reference = cv.getRotationMatrix2D(center, 11.0, 1.28).astype(np.float32)
    moving_to_reference[:, 2] += np.array([24.0, -9.0], dtype=np.float32)
    moving = _warp_reference_to_moving(reference, moving_shape, moving_to_reference)
    moving = np.asarray(0.35 + 1.8 * moving, dtype=np.float32)

    result = estimate_similarity_registration(
        reference,
        moving,
        scale_values=parse_scale_search("0.50:2.00:31"),
        registration_profile="bead_geometric_v2",
    )

    assert result.status == "aligned"
    assert result.registration_profile == "bead_geometric_v2"
    assert result.registration_constraint_level == "full"
    assert result.registration_underconstrained is False
    assert result.registration_initializer == "bead_pair"
    assert result.matched_bead_count >= 3
    assert abs(result.scale - 1.28) <= 0.12
    assert abs(abs(result.rotation_deg) - 11.0) <= 3.0
    assert float(np.mean(np.abs(_robust_unit_for_test(result.aligned_image) - _robust_unit_for_test(reference)))) < 0.25
    linear = result.forward_matrix[:, :2]
    assert abs(float(np.dot(linear[:, 0], linear[:, 1]))) < 1e-3
    np.testing.assert_allclose(
        np.linalg.norm(linear[:, 0]),
        np.linalg.norm(linear[:, 1]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_geometric_registration_reports_single_bead_underconstrained_without_match_count_failure():
    reference = _single_synthetic_bead()
    moving_shape = (96, 96)
    center = (moving_shape[1] / 2.0, moving_shape[0] / 2.0)
    moving_to_reference = cv.getRotationMatrix2D(center, -17.0, 1.45).astype(np.float32)
    moving_to_reference[:, 2] += np.array([18.0, 7.0], dtype=np.float32)
    moving = _warp_reference_to_moving(reference, moving_shape, moving_to_reference)

    result = estimate_similarity_registration(
        reference,
        moving,
        scale_values=parse_scale_search("0.50:2.00:31"),
        registration_profile="bead_geometric_v2",
    )

    assert result.status == "aligned"
    assert result.registration_constraint_level == "single_bead"
    assert result.registration_underconstrained is True
    assert result.registration_initializer == "single_bead_grid"
    assert result.matched_bead_count >= 1
    assert not any("matched_bead_count<3" in reason for reason in result.registration_suspect_reasons)


def test_geometric_registration_respects_hypothesis_runtime_limits():
    reference = _synthetic_beads((190, 190))
    moving_shape = (150, 150)
    center = (moving_shape[1] / 2.0, moving_shape[0] / 2.0)
    moving_to_reference = cv.getRotationMatrix2D(center, 11.0, 1.28).astype(np.float32)
    moving_to_reference[:, 2] += np.array([24.0, -9.0], dtype=np.float32)
    moving = _warp_reference_to_moving(reference, moving_shape, moving_to_reference)

    result = estimate_similarity_registration(
        reference,
        moving,
        scale_values=parse_scale_search("0.50:2.00:31"),
        registration_profile="bead_geometric_v2",
        registration_prefilter_hypotheses=7,
        registration_dense_hypotheses=3,
        registration_top_hypotheses=2,
        registration_refine_max_evals=30,
    )

    assert result.status == "aligned"
    assert result.hypotheses_generated >= result.hypotheses_prefiltered
    assert result.hypotheses_prefiltered <= 7
    assert result.hypotheses_dense_scored <= 3
    assert result.hypotheses_refined <= 2
    assert result.optimizer_evaluations > 0
    assert result.registration_runtime_seconds is not None


def _robust_unit_for_test(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.5))
    if high <= low:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def test_residual_translation_above_cap_is_not_applied():
    reference = _synthetic_beads()
    anchor = estimate_similarity_registration(reference, reference, scale_values=[1.0])
    shifted = cv.warpAffine(
        reference,
        np.array([[1.0, 0.0, 11.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        reference.shape[::-1],
        flags=cv.INTER_LINEAR,
    )

    result = apply_anchor_registration(
        reference,
        shifted,
        anchor,
        residual_translation_max_px=3.0,
    )

    assert result.residual_applied is False
    assert result.residual_translation_norm > 3.0
    assert result.registration_suspect is True
    assert any("residual_translation_norm" in reason for reason in result.registration_suspect_reasons)
    assert any("Residual translation" in warning for warning in result.warnings)


def test_validation_metrics_penalize_blur_and_center_errors():
    validation = _synthetic_beads()
    sharp = validation.copy()
    blurred = cv.GaussianBlur(validation, (0, 0), 4.0)
    shifted = cv.warpAffine(
        validation,
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        validation.shape[::-1],
        flags=cv.INTER_LINEAR,
    )

    sharp_metrics = compute_validation_metrics(validation, sharp)
    blurred_metrics = compute_validation_metrics(validation, blurred)
    shifted_metrics = compute_validation_metrics(validation, shifted)

    assert sharp_metrics.metrics[PROFILE_NAME] > blurred_metrics.metrics[PROFILE_NAME]
    assert sharp_metrics.metrics["validation_sharpness"] > blurred_metrics.metrics["validation_sharpness"]
    assert sharp_metrics.metrics["validation_bead"] > shifted_metrics.metrics["validation_bead"]


def test_validation_metrics_accept_prepared_reference_cache():
    validation = _synthetic_beads()
    blurred = cv.GaussianBlur(validation, (0, 0), 3.0)
    reference_cache = trio_benchmark._prepare_validation_metric_reference(validation)

    direct = compute_validation_metrics(validation, blurred)
    cached = compute_validation_metrics(validation, blurred, reference_cache=reference_cache)

    for name, value in direct.metrics.items():
        cached_value = cached.metrics[name]
        if value is None:
            assert cached_value is None
        else:
            assert cached_value is not None
            np.testing.assert_allclose(cached_value, value, rtol=1e-6, atol=1e-6)
    assert cached.weights_used == direct.weights_used
    assert cached.bead_count == direct.bead_count


def test_low_confidence_bead_detection_renormalizes_to_image_metrics():
    validation = np.zeros((80, 80), dtype=np.float32)
    candidate = np.zeros_like(validation)

    metrics = compute_validation_metrics(validation, candidate)

    assert metrics.metrics["validation_bead"] is None
    assert metrics.metrics["validation_sharpness"] is None
    assert metrics.metrics[PROFILE_NAME] == metrics.metrics["validation_signal_structure_score"]
    assert any("low-confidence bead detection" in warning for warning in metrics.warnings)


def test_run_benchmark_uses_validate_workspace_transform_grid(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    validation_matrix = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    thorcam_matrix = np.array([[1.0, 0.0, 11.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    validation_processed = _source_from_grid_image(rectified, (84, 88), validation_matrix)
    thorcam_processed = _source_from_grid_image(rectified, (82, 91), thorcam_matrix)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=validation_processed,
        thorcam_processed=thorcam_processed,
        validation_matrix=validation_matrix,
        thorcam_matrix=thorcam_matrix,
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=rectified)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            use_validation_workspace_transforms=True,
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    best_registration = result.summary["best_candidate"]["registration"]
    thor_registration = result.summary["thorcam_baseline"]["registration"]
    assert result.best_lf_row_id == "stage2:candidate_a"
    assert result.summary["config"]["use_validation_workspace_transforms"] is True
    assert result.summary["config"]["registration_profile"] == "validate_workspace_transform_v1"
    assert best_registration["registration_profile"] == "validate_workspace_transform_v1"
    assert best_registration["registration_initializer"] == "validation_workspace_identity_lf"
    assert best_registration["registration_transform_source"] == "identity_lf_grid"
    assert best_registration["hypotheses_generated"] == 0
    assert best_registration["optimizer_evaluations"] == 0
    np.testing.assert_allclose(best_registration["forward_matrix"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert thor_registration["registration_initializer"] == "validation_workspace_transform"
    assert thor_registration["registration_transform_source"] == "thorcam_alignment_final_warp_matrix"
    assert (output_dir / "trio_validation_workspace_transforms.json").exists()
    assert (output_dir / "figures" / "validation_workspace_validation_overlay.png").exists()
    assert (output_dir / "figures" / "validation_workspace_thorcam_overlay.png").exists()


def test_validate_workspace_transform_missing_validation_matrix_fails_even_when_same_shape(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    nan_matrix = np.full((2, 3), np.nan, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=rectified,
        thorcam_processed=rectified,
        validation_matrix=nan_matrix,
        thorcam_matrix=nan_matrix,
    )

    model = SimpleNamespace(primary_stage_name="stage2")
    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: [])
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: [])

    with pytest.raises(ValueError, match="missing_matrix"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_workspace=validation_workspace,
                output_dir=output_dir,
                use_validation_workspace_transforms=True,
                residual_translation_max_px=0.0,
                top_k=1,
            )
        )


def test_validate_workspace_transform_invalid_thorcam_identity_shape_mismatch_is_unrankable(
    tmp_path, monkeypatch
):
    rectified = _synthetic_beads((64, 64))
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=rectified,
        thorcam_processed=rectified[:32, :21],
        validation_matrix=identity,
        thorcam_matrix=identity,
        thorcam_alignment_status="not_run",
        thorcam_alignment_model="none",
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=rectified)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            use_validation_workspace_transforms=True,
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    thorcam = result.summary["thorcam_baseline"]
    thor_registration = thorcam["registration"]
    assert result.best_lf_row_id == "stage2:candidate_a"
    assert thor_registration["status"] == "failed"
    assert thor_registration["validation_rankable"] is False
    assert thor_registration["ranking_exclusion_reason"] == "not_run"
    assert thorcam["metrics"][PROFILE_NAME] is None
    assert PROFILE_NAME not in result.summary["best_lf_minus_thorcam"]

    transform_payload = json.loads(
        (output_dir / "trio_validation_workspace_transforms.json").read_text(encoding="utf-8")
    )
    thorcam_info = transform_payload["workspace_transform_info"]["sources"]["thorcam"]
    assert thorcam_info["valid_for_scoring"] is False
    assert thorcam_info["invalid_reason"] == "not_run"
    assert thorcam_info["source_shape_matches_rectified"] is False
    assert thorcam_info["matrix_is_identity"] is True
    assert thorcam_info["replay_shape_matches_rectified"] is True

    with (output_dir / "trio_benchmark_results.csv").open(newline="", encoding="utf-8") as fp:
        csv_rows = list(csv.DictReader(fp))
    thorcam_row = next(row for row in csv_rows if row["source_kind"] == "thorcam_baseline")
    assert thorcam_row[PROFILE_NAME] == ""
    assert thorcam_row["validation_rankable"] == "false"
    assert thorcam_row["ranking_exclusion_reason"] == "not_run"


def test_validate_workspace_transform_accepts_successful_identity_with_explicit_rectified_shape(
    tmp_path, monkeypatch
):
    rectified = _synthetic_beads((64, 64))
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=rectified,
        thorcam_processed=rectified[:32, :21],
        validation_matrix=identity,
        thorcam_matrix=identity,
        thorcam_alignment_status="aligned_target_in_reference",
        thorcam_alignment_model="translation_ecc",
        thorcam_alignment_output_shape=rectified.shape,
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=rectified)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            use_validation_workspace_transforms=True,
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    thorcam = result.summary["thorcam_baseline"]
    thor_registration = thorcam["registration"]
    assert thor_registration["status"] == "aligned"
    assert thor_registration["validation_rankable"] is True
    assert thorcam["metrics"][PROFILE_NAME] is not None
    assert PROFILE_NAME in result.summary["best_lf_minus_thorcam"]

    transform_payload = json.loads(
        (output_dir / "trio_validation_workspace_transforms.json").read_text(encoding="utf-8")
    )
    thorcam_info = transform_payload["workspace_transform_info"]["sources"]["thorcam"]
    assert thorcam_info["valid_for_replay"] is True
    assert thorcam_info["valid_for_scoring"] is True
    assert thorcam_info["invalid_reason"] is None
    assert thorcam_info["alignment_output_shape"] == [64, 64]
    assert thorcam_info["replay_output_shape"] == [64, 64]
    assert thorcam_info["replay_shape_matches_rectified"] is True


def test_validate_workspace_transform_validation_identity_shape_mismatch_fails_closed(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=rectified[:32, :21],
        thorcam_processed=rectified,
        validation_matrix=identity,
        thorcam_matrix=identity,
        validation_alignment_status="not_run",
        validation_alignment_model="none",
    )

    model = SimpleNamespace(primary_stage_name="stage2")
    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: [])
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: [])

    with pytest.raises(ValueError, match="not_run"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_workspace=validation_workspace,
                output_dir=output_dir,
                use_validation_workspace_transforms=True,
                top_k=1,
            )
        )


def test_validate_workspace_transform_validation_output_shape_mismatch_fails_closed(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    validation_matrix = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    validation_processed = _source_from_grid_image(rectified, (78, 80), validation_matrix)
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=validation_processed,
        thorcam_processed=rectified,
        validation_matrix=validation_matrix,
        thorcam_matrix=identity,
        validation_alignment_output_shape=(63, 64),
    )

    model = SimpleNamespace(primary_stage_name="stage2")
    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: [])
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: [])

    with pytest.raises(ValueError, match="output_shape_mismatch"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_workspace=validation_workspace,
                output_dir=output_dir,
                use_validation_workspace_transforms=True,
                top_k=1,
            )
        )


def test_validate_workspace_transform_validation_missing_status_shape_mismatch_fails_closed(
    tmp_path, monkeypatch
):
    rectified = _synthetic_beads((64, 64))
    validation_matrix = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    validation_processed = _source_from_grid_image(rectified, (78, 80), validation_matrix)
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=validation_processed,
        thorcam_processed=rectified,
        validation_matrix=validation_matrix,
        thorcam_matrix=identity,
        validation_alignment_status=None,
        validation_alignment_model=None,
        validation_alignment_output_shape=rectified.shape,
    )

    model = SimpleNamespace(primary_stage_name="stage2")
    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: [])
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: [])

    with pytest.raises(ValueError, match="suspect_alignment_status"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_workspace=validation_workspace,
                output_dir=output_dir,
                use_validation_workspace_transforms=True,
                top_k=1,
            )
        )


def test_validate_workspace_transform_missing_output_shape_is_unrankable_for_thorcam(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=rectified,
        thorcam_processed=rectified[:32, :21],
        validation_matrix=identity,
        thorcam_matrix=identity,
        include_alignment_output_shape=False,
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=rectified)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            use_validation_workspace_transforms=True,
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    thorcam = result.summary["thorcam_baseline"]
    assert thorcam["registration"]["status"] == "failed"
    assert thorcam["registration"]["ranking_exclusion_reason"] == "missing_output_shape"
    transform_payload = json.loads(
        (output_dir / "trio_validation_workspace_transforms.json").read_text(encoding="utf-8")
    )
    thorcam_info = transform_payload["workspace_transform_info"]["sources"]["thorcam"]
    assert thorcam_info["alignment_output_shape"] is None
    assert thorcam_info["valid_for_replay"] is False
    assert thorcam_info["valid_for_scoring"] is False


def test_validate_workspace_transform_missing_matrix_different_shape_fails_closed(tmp_path, monkeypatch):
    rectified = _synthetic_beads((64, 64))
    validation_processed = np.zeros((70, 72), dtype=np.float32)
    thorcam_processed = rectified.copy()
    nan_matrix = np.full((2, 3), np.nan, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=rectified,
        validation_processed=validation_processed,
        thorcam_processed=thorcam_processed,
        validation_matrix=nan_matrix,
        thorcam_matrix=np.eye(2, 3, dtype=np.float32),
    )

    model = SimpleNamespace(primary_stage_name="stage2")
    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: [])
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: [])

    with pytest.raises(ValueError, match="validation_alignment_final_warp_matrix"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_workspace=validation_workspace,
                output_dir=output_dir,
                use_validation_workspace_transforms=True,
                top_k=1,
            )
        )


def test_workspace_transform_sparse_lf_match_is_underconstrained_not_match_count_suspect(
    tmp_path, monkeypatch
):
    validation = _synthetic_beads((96, 96))
    sparse_candidate = np.zeros_like(validation)
    sparse_candidate[30:46, 27:43] = validation[30:46, 27:43]
    identity = np.eye(2, 3, dtype=np.float32)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    _write_validation_transform_workspace(
        validation_workspace,
        rectified=validation,
        validation_processed=validation,
        thorcam_processed=validation,
        validation_matrix=identity,
        thorcam_matrix=identity,
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=sparse_candidate)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_sparse",
            stage="stage2",
            candidate_id="candidate_sparse",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_sparse",
            stage="stage2",
            candidate_id="candidate_sparse",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            use_validation_workspace_transforms=True,
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    registration = result.summary["best_candidate"]["registration"]
    assert registration["matched_bead_count"] == 1
    assert registration["registration_constraint_level"] == "single_bead"
    assert registration["registration_underconstrained"] is True
    assert registration["validation_rankable"] is True
    assert not any("matched_bead_count" in reason for reason in registration["registration_suspect_reasons"])


def test_run_benchmark_uses_manual_group_calibration_for_unannotated_scene(tmp_path, monkeypatch):
    validation = _synthetic_beads((128, 128))
    moving_shape = (96, 96)
    manual_matrix = _manual_similarity_matrix(1.18, 8.0, 11.0, -4.0)
    moving = _warp_reference_to_moving(validation, moving_shape, manual_matrix)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    annotations_path = tmp_path / "manual_annotations.json"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, moving)
    _write_manual_annotations(
        annotations_path,
        [
            _manual_annotation(
                scene_id="annotated_scene",
                dataset_id="annotated_scene",
                source_kind="lf",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=manual_matrix,
                anchor_row_id="stage2:candidate_a",
            ),
            _manual_annotation(
                scene_id="annotated_scene",
                dataset_id="annotated_scene",
                source_kind="thorcam",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=manual_matrix,
            ),
        ],
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=moving)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_b",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_b",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir,
            registration_profile="manual_similarity_v1",
            registration_annotations=annotations_path,
            registration_calibration_group="rig_a",
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    best = result.summary["best_candidate"]
    assert result.best_lf_row_id == "stage2:candidate_a"
    assert best["registration"]["registration_profile"] == "manual_similarity_v1"
    assert best["registration"]["registration_initializer"] == "group_calibration_median"
    assert best["registration"]["registration_transform_source"] == "group_calibration"
    assert best["registration"]["registration_calibration_group"] == "rig_a"
    assert best["registration"]["registration_point_count"] == 3
    assert best["registration"]["validation_rankable"] is True
    assert abs(best["registration"]["scale"] - 1.18) < 1e-6
    assert abs(best["registration"]["rotation_deg"] - 8.0) < 1e-6
    assert (output_dir / "trio_registration_calibration.json").exists()
    assert (output_dir / "figures" / "manual_points_lf.png").exists()
    assert (output_dir / "figures" / "manual_points_thorcam.png").exists()


def test_manual_group_calibration_allows_single_point_scene_translation_override(tmp_path, monkeypatch):
    validation = _synthetic_beads((128, 128))
    moving_shape = (96, 96)
    group_matrix = _manual_similarity_matrix(1.18, 8.0, 11.0, -4.0)
    scene_matrix = _manual_similarity_matrix(1.18, 8.0, 15.0, -7.0)
    moving = _warp_reference_to_moving(validation, moving_shape, scene_matrix)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    annotations_path = tmp_path / "manual_annotations.json"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, moving)
    _write_manual_annotations(
        annotations_path,
        [
            _manual_annotation(
                scene_id="annotated_scene",
                dataset_id="annotated_scene",
                source_kind="lf",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=group_matrix,
                anchor_row_id="stage2:candidate_a",
            ),
            _manual_annotation(
                scene_id="scene_b",
                dataset_id="scene_b",
                source_kind="lf",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=scene_matrix,
                anchor_row_id="stage2:candidate_a",
                point_count=1,
            ),
            _manual_annotation(
                scene_id="annotated_scene",
                dataset_id="annotated_scene",
                source_kind="thorcam",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=group_matrix,
            ),
            _manual_annotation(
                scene_id="scene_b",
                dataset_id="scene_b",
                source_kind="thorcam",
                calibration_group="rig_a",
                reference_shape=validation.shape,
                moving_shape=moving_shape,
                matrix=scene_matrix,
                point_count=1,
            ),
        ],
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=moving)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_b",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_b",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir,
            registration_profile="manual_similarity_v1",
            registration_annotations=annotations_path,
            registration_calibration_group="rig_a",
            residual_translation_max_px=0.0,
            top_k=1,
        )
    )

    registration = result.summary["best_candidate"]["registration"]
    assert registration["registration_initializer"] == "group_plus_single_point_translation"
    assert registration["registration_underconstrained"] is True
    assert registration["registration_point_count"] == 1
    assert abs(registration["translation_x"] - 15.0) < 1e-6
    assert abs(registration["translation_y"] + 7.0) < 1e-6


def test_manual_registration_missing_calibration_fails_closed(tmp_path, monkeypatch):
    validation = _synthetic_beads((64, 64))
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, validation)

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=validation)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    with pytest.raises(ValueError, match="manual_similarity_v1 requires registration_annotations"):
        run_benchmark(
            BenchmarkConfig(
                sweep_dir=sweep_dir,
                dataset_id="scene_a",
                validation_image=validation_path,
                thorcam_image=thorcam_path,
                output_dir=output_dir,
                registration_profile="manual_similarity_v1",
                registration_calibration_group="rig_a",
                top_k=1,
            )
        )


def test_run_benchmark_writes_ranked_outputs_and_preview_fallback(tmp_path, monkeypatch):
    validation = _synthetic_beads((96, 96))
    sharp = validation.copy()
    blurred = cv.GaussianBlur(validation, (0, 0), 3.0)
    thorcam = cv.GaussianBlur(validation, (0, 0), 4.0)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, thorcam)

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=sharp)

    run2 = sweep_dir / "run2"
    run2.mkdir()
    preview2 = run2 / "avg_rectified.png"
    _write_image(preview2, blurred)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={"alpha": 1.0},
            config={"strategy": "combined"},
            score=10.0,
        ),
        SimpleNamespace(
            row_id="stage2:candidate_b",
            stage="stage2",
            candidate_id="candidate_b",
            generation_id="gen1",
            rank=2,
            strategy="combined",
            params={"alpha": 0.5},
            config={"strategy": "combined"},
            score=8.0,
        ),
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        ),
        SimpleNamespace(
            row_id="stage2:candidate_b",
            stage="stage2",
            candidate_id="candidate_b",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=None,
            preview_path=str(preview2),
            metric_record_path=None,
        ),
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir,
            scale_values=[1.0],
            residual_translation_max_px=2.0,
            top_k=2,
            enable_nonrigid_diagnostic=True,
        )
    )

    assert result.best_lf_row_id == "stage2:candidate_a"
    assert (output_dir / "trio_benchmark_results.csv").exists()
    assert (output_dir / "trio_benchmark_summary.json").exists()
    assert (output_dir / "trio_benchmark_report.html").exists()
    assert (output_dir / "figures" / "rank_001_stage2_candidate_a_overlay.png").exists()
    assert (output_dir / "figures" / "rank_001_stage2_candidate_a_affine_diagnostic_overlay.png").exists()

    with (output_dir / "trio_benchmark_results.csv").open(newline="", encoding="utf-8") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert "params_json" in csv_rows[0]
    assert "config_json" in csv_rows[0]
    assert "registration_profile" in csv_rows[0]
    assert "registration_confidence" in csv_rows[0]
    assert "registration_suspect" in csv_rows[0]
    assert "registration_suspect_reasons" in csv_rows[0]
    assert "matched_bead_count" in csv_rows[0]
    assert "median_bead_error_px" in csv_rows[0]
    assert "common_fov_fraction" in csv_rows[0]
    assert "anchor_selection_reason" in csv_rows[0]
    assert "residual_translation_norm" in csv_rows[0]
    assert "registration_constraint_level" in csv_rows[0]
    assert "registration_underconstrained" in csv_rows[0]
    assert "registration_initializer" in csv_rows[0]
    assert "registration_refined_score" in csv_rows[0]
    assert "registration_runtime_seconds" in csv_rows[0]
    assert "hypotheses_generated" in csv_rows[0]
    assert "hypotheses_prefiltered" in csv_rows[0]
    assert "hypotheses_dense_scored" in csv_rows[0]
    assert "hypotheses_refined" in csv_rows[0]
    assert "optimizer_evaluations" in csv_rows[0]
    assert [row["source_kind"] for row in csv_rows].count("lf_candidate") == 2
    assert [row["source_kind"] for row in csv_rows].count("thorcam_baseline") == 1
    first_lf_row = next(row for row in csv_rows if row["candidate_id"] == "candidate_a")
    assert json.loads(first_lf_row["params_json"]) == {"alpha": 1.0}
    assert json.loads(first_lf_row["config_json"]) == {"strategy": "combined"}
    fallback_row = next(row for row in csv_rows if row["candidate_id"] == "candidate_b")
    assert "preview_fallback" in fallback_row["warnings"]

    summary = json.loads((output_dir / "trio_benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["best_candidate"]["row_id"] == "stage2:candidate_a"
    assert summary["config"]["profile_name"] == PROFILE_NAME
    assert summary["config"]["registration_profile"] == "bead_geometric_v2"
    assert "registration_confidence" in summary["best_candidate"]["registration"]
    assert "matched_bead_count" in summary["best_candidate"]["registration"]
    assert "registration_constraint_level" in summary["best_candidate"]["registration"]
    assert "registration_initializer" in summary["best_candidate"]["registration"]
    assert "registration_runtime_seconds" in summary["best_candidate"]["registration"]
    assert "hypotheses_dense_scored" in summary["best_candidate"]["registration"]
    assert summary["config"]["registration_coarse_max_dim"] == 384
    assert summary["config"]["registration_refine_max_dim"] == 768
    assert summary["config"]["registration_prefilter_hypotheses"] == 256
    assert summary["config"]["registration_dense_hypotheses"] == 64
    assert summary["config"]["registration_refine_max_evals"] == 120
    assert summary["config"]["registration_progress"] is False
    assert "best_lf_minus_thorcam" in summary
    assert result.rows[0].source_kind == "lf_candidate"


def test_run_benchmark_auto_anchor_selects_registration_confident_candidate_by_default(tmp_path, monkeypatch):
    validation = _synthetic_beads((96, 96))
    mismatched = np.flipud(validation)
    thorcam = cv.GaussianBlur(validation, (0, 0), 3.0)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, thorcam)

    run_bad = sweep_dir / "bad"
    run_bad.mkdir()
    workspace_bad = run_bad / "workspace.npz"
    np.savez_compressed(workspace_bad, average_rectified_image=mismatched)

    run_good = sweep_dir / "good"
    run_good.mkdir()
    workspace_good = run_good / "workspace.npz"
    np.savez_compressed(workspace_good, average_rectified_image=validation)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_bad",
            stage="stage2",
            candidate_id="candidate_bad",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        ),
        SimpleNamespace(
            row_id="stage2:candidate_good",
            stage="stage2",
            candidate_id="candidate_good",
            generation_id="gen1",
            rank=2,
            strategy="combined",
            params={},
            config={},
            score=8.0,
        ),
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_bad",
            stage="stage2",
            candidate_id="candidate_bad",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace_bad),
            preview_path=None,
            metric_record_path=None,
        ),
        SimpleNamespace(
            row_id="stage2:candidate_good",
            stage="stage2",
            candidate_id="candidate_good",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace_good),
            preview_path=None,
            metric_record_path=None,
        ),
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir,
            scale_values=[1.0],
            top_k=1,
        )
    )

    assert result.summary["anchor"]["candidate_id"] == "candidate_good"
    assert result.summary["anchor"]["registration"]["registration_profile"] == "bead_geometric_v2"
    assert result.summary["anchor"]["registration"]["anchor_selection_reason"] == "best_registration_confidence"


def test_run_benchmark_uses_profile_specific_default_scale_search(tmp_path, monkeypatch):
    validation = _synthetic_beads((96, 96))
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_path = tmp_path / "validation.tif"
    thorcam_path = tmp_path / "thorcam.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, validation)

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=validation)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    default_result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir / "default",
            scale_values=None,
        )
    )
    legacy_result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=output_dir / "legacy",
            scale_values=None,
            registration_profile="centered_ecc_legacy",
        )
    )

    assert default_result.summary["config"]["registration_profile"] == "bead_geometric_v2"
    assert default_result.summary["config"]["scale_values"][0] == 0.5
    assert default_result.summary["config"]["scale_values"][-1] == 2.0
    assert len(default_result.summary["config"]["scale_values"]) == 31
    assert legacy_result.summary["config"]["registration_profile"] == "centered_ecc_legacy"
    assert legacy_result.summary["config"]["scale_values"][0] == 0.9
    assert legacy_result.summary["config"]["scale_values"][-1] == 1.1
    assert len(legacy_result.summary["config"]["scale_values"]) == 11


def test_run_benchmark_reads_references_from_validation_workspace(tmp_path, monkeypatch):
    validation = _synthetic_beads((96, 96))
    thorcam = cv.GaussianBlur(validation, (0, 0), 3.0)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    np.savez_compressed(
        validation_workspace,
        validation_processed=validation.astype(np.float32),
        thorcam_processed=thorcam.astype(np.float32),
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=validation)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_image=None,
            thorcam_image=None,
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            scale_values=[1.0],
            top_k=1,
        )
    )

    assert result.best_lf_row_id == "stage2:candidate_a"
    summary = json.loads((output_dir / "trio_benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["validation_image"] == f"{validation_workspace}#validation_processed"
    assert summary["config"]["thorcam_image"] == f"{validation_workspace}#thorcam_processed"


def test_run_benchmark_warns_on_failed_validation_workspace_alignment(tmp_path, monkeypatch):
    validation = _synthetic_beads((96, 96))

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    validation_workspace = tmp_path / "validation_workspace.npz"
    np.savez_compressed(
        validation_workspace,
        validation_processed=validation.astype(np.float32),
        thorcam_processed=validation.astype(np.float32),
        validation_auto_align=np.array(True),
        validation_alignment_status=np.array("low_confidence_match"),
        thorcam_auto_align=np.array(True),
        thorcam_alignment_status=np.array("ecc_failed:error"),
    )

    run1 = sweep_dir / "run1"
    run1.mkdir()
    workspace1 = run1 / "workspace.npz"
    np.savez_compressed(workspace1, average_rectified_image=validation)

    model = SimpleNamespace(primary_stage_name="stage2")
    rows = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            generation_id="gen1",
            rank=1,
            strategy="combined",
            params={},
            config={},
            score=10.0,
        )
    ]
    records = [
        SimpleNamespace(
            row_id="stage2:candidate_a",
            stage="stage2",
            candidate_id="candidate_a",
            dataset_id="scene_a",
            generation_id="gen1",
            workspace_path=str(workspace1),
            preview_path=None,
            metric_record_path=None,
        )
    ]

    monkeypatch.setattr("moment_images.trio_benchmark.load_sweep_visualization_model", lambda path: model)
    monkeypatch.setattr("moment_images.trio_benchmark.candidate_rows_for_generation_scope", lambda model, scope: rows)
    monkeypatch.setattr("moment_images.trio_benchmark.run_records_for_generation_scope", lambda model, scope: records)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="scene_a",
            validation_workspace=validation_workspace,
            output_dir=output_dir,
            scale_values=[1.0],
            top_k=1,
        )
    )

    warnings = result.summary["warnings"]
    assert any("validation workspace alignment suspect" in warning for warning in warnings)
    assert any("validation_alignment_status=low_confidence_match" in warning for warning in warnings)
    assert any("thorcam_alignment_status=ecc_failed:error" in warning for warning in warnings)


def test_run_benchmark_default_generation_scope_matches_sweep_visualization(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=True,
        stage_mode="single_stage",
    )
    validation = _synthetic_beads((16, 16))
    validation_path = tmp_path / "validation_scope.tif"
    thorcam_path = tmp_path / "thorcam_scope.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, validation)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="d1",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=tmp_path / "scope_out",
            stage="stage1",
            scale_values=[1.0],
            top_k=1,
        )
    )

    assert result.best_lf_row_id == "stage1:s1c0000"


def test_run_benchmark_uses_candidate_preview_when_run_record_missing(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=True,
        include_workspace=False,
        stage_mode="two_stage",
    )
    validation = np.full((16, 16), 127, dtype=np.float32)
    validation[4:12, 4:12] = 255
    validation_path = tmp_path / "validation_preview.tif"
    thorcam_path = tmp_path / "thorcam_preview.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, validation)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="d1",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=tmp_path / "preview_out",
            stage="stage2",
            scale_values=[1.0],
            top_k=1,
        )
    )

    assert result.best_lf_row_id == "stage2:s2c0000"
    assert "preview_fallback" in result.rows[0].warnings


def test_run_benchmark_reports_skipped_candidate_artifacts(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=False,
        stage_mode="single_stage",
    )
    validation = np.full((16, 16), 127, dtype=np.float32)
    validation[4:12, 4:12] = 255
    validation_path = tmp_path / "validation_skipped.tif"
    thorcam_path = tmp_path / "thorcam_skipped.tif"
    _write_image(validation_path, validation)
    _write_image(thorcam_path, validation)

    result = run_benchmark(
        BenchmarkConfig(
            sweep_dir=sweep_dir,
            dataset_id="d1",
            validation_image=validation_path,
            thorcam_image=thorcam_path,
            output_dir=tmp_path / "skipped_out",
            stage="stage1",
            scale_values=[1.0],
            top_k=1,
        )
    )

    assert result.best_lf_row_id == "stage1:s1c0000"
    assert any("skipped candidate stage1:s1c0001" in warning for warning in result.summary["warnings"])

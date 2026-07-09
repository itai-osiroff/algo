import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
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

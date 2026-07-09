import json
import csv

import numpy as np

from moment_images.test_sweep_visualization import _create_modern_fixture
from moment_images.trio_benchmark_batch import main, run_batch_from_manifest


def test_run_batch_from_manifest_writes_scene_outputs_and_aggregate_summary(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=True,
        stage_mode="single_stage",
    )
    validation_workspace = tmp_path / "validation_workspace.npz"
    reference = np.full((4, 4), 1.25, dtype=np.float32)
    np.savez_compressed(
        validation_workspace,
        validation_processed=reference,
        thorcam_processed=reference,
    )
    manifest_path = tmp_path / "trio_manifest.json"
    output_dir = tmp_path / "batch_output"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "stage": "stage1",
                    "scale_search": "1.0:1.0:1",
                    "top_k": 1,
                    "registration_coarse_max_dim": 96,
                    "registration_refine_max_dim": 128,
                    "registration_prefilter_hypotheses": 5,
                    "registration_dense_hypotheses": 3,
                    "registration_refine_max_evals": 20,
                    "registration_progress": True,
                },
                "scenes": [
                    {
                        "scene_id": "scene_one",
                        "sweep_dir": str(sweep_dir),
                        "dataset_id": "d1",
                        "validation_workspace": str(validation_workspace),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_batch_from_manifest(manifest_path, output_dir)

    scene_dir = output_dir / "scene_one"
    assert summary["num_scenes"] == 1
    assert summary["num_successful"] == 1
    assert summary["scenes"][0]["scene_id"] == "scene_one"
    assert summary["scenes"][0]["status"] == "ok"
    assert summary["scenes"][0]["best_lf_row_id"] == "stage1:s1c0000"
    assert "registration_confidence" in summary["scenes"][0]
    assert "registration_constraint_level" in summary["scenes"][0]
    assert "registration_underconstrained" in summary["scenes"][0]
    assert "registration_initializer" in summary["scenes"][0]
    assert "registration_runtime_seconds" in summary["scenes"][0]
    assert "hypotheses_dense_scored" in summary["scenes"][0]
    assert (scene_dir / "trio_benchmark_results.csv").exists()
    assert (scene_dir / "trio_benchmark_summary.json").exists()
    assert (output_dir / "trio_batch_summary.json").exists()
    assert (output_dir / "trio_batch_results.csv").exists()
    assert (output_dir / "trio_batch_report.html").exists()
    with (output_dir / "trio_batch_results.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert "registration_constraint_level" in rows[0]
    assert "registration_underconstrained" in rows[0]
    assert "registration_initializer" in rows[0]
    assert "registration_runtime_seconds" in rows[0]
    assert "hypotheses_dense_scored" in rows[0]
    scene_summary = json.loads((scene_dir / "trio_benchmark_summary.json").read_text(encoding="utf-8"))
    assert scene_summary["config"]["registration_coarse_max_dim"] == 96
    assert scene_summary["config"]["registration_refine_max_dim"] == 128
    assert scene_summary["config"]["registration_prefilter_hypotheses"] == 5
    assert scene_summary["config"]["registration_dense_hypotheses"] == 3
    assert scene_summary["config"]["registration_refine_max_evals"] == 20
    assert scene_summary["config"]["registration_progress"] is True


def test_batch_manifest_forwards_manual_calibration_fields(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=True,
        stage_mode="single_stage",
    )
    validation_workspace = tmp_path / "validation_workspace.npz"
    reference = np.full((4, 4), 1.25, dtype=np.float32)
    np.savez_compressed(
        validation_workspace,
        validation_processed=reference,
        thorcam_processed=reference,
    )
    annotations_path = tmp_path / "manual_annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": [
                    {
                        "scene_id": "annotated_scene",
                        "dataset_id": "d1",
                        "source_kind": "lf",
                        "calibration_group": "rig_a",
                        "anchor_row_id": "stage1:s1c0000",
                        "reference_shape": [4, 4],
                        "moving_shape": [4, 4],
                        "points": [
                            {"moving": [0.0, 0.0], "reference": [0.0, 0.0]},
                            {"moving": [3.0, 0.0], "reference": [3.0, 0.0]},
                            {"moving": [0.0, 3.0], "reference": [0.0, 3.0]},
                        ],
                    },
                    {
                        "scene_id": "annotated_scene",
                        "dataset_id": "d1",
                        "source_kind": "thorcam",
                        "calibration_group": "rig_a",
                        "reference_shape": [4, 4],
                        "moving_shape": [4, 4],
                        "points": [
                            {"moving": [0.0, 0.0], "reference": [0.0, 0.0]},
                            {"moving": [3.0, 0.0], "reference": [3.0, 0.0]},
                            {"moving": [0.0, 3.0], "reference": [0.0, 3.0]},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "trio_manifest.json"
    output_dir = tmp_path / "batch_output"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "stage": "stage1",
                    "top_k": 1,
                    "registration_profile": "manual_similarity_v1",
                    "registration_annotations": str(annotations_path),
                    "registration_calibration_group": "rig_a",
                    "registration_calibration_mode": "group",
                },
                "scenes": [
                    {
                        "scene_id": "scene_one",
                        "sweep_dir": str(sweep_dir),
                        "dataset_id": "d1",
                        "validation_workspace": str(validation_workspace),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_batch_from_manifest(manifest_path, output_dir)

    assert summary["num_successful"] == 1
    scene_summary = json.loads(
        (output_dir / "scene_one" / "trio_benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert scene_summary["config"]["registration_profile"] == "manual_similarity_v1"
    assert scene_summary["config"]["registration_annotations"] == str(annotations_path)
    assert scene_summary["config"]["registration_calibration_group"] == "rig_a"
    assert scene_summary["config"]["registration_calibration_mode"] == "group"
    assert scene_summary["best_candidate"]["registration"]["registration_initializer"] == "group_calibration_median"
    assert "registration_transform_source" in summary["scenes"][0]


def test_batch_manifest_forwards_validation_workspace_transform_flag(tmp_path):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=True,
        stage_mode="single_stage",
    )
    validation_workspace = tmp_path / "validation_workspace.npz"
    reference = np.full((4, 4), 1.25, dtype=np.float32)
    identity = np.eye(2, 3, dtype=np.float32)
    np.savez_compressed(
        validation_workspace,
        rectified_image=reference,
        validation_processed=reference,
        thorcam_processed=reference,
        validation_alignment_final_warp_matrix=identity,
        thorcam_alignment_final_warp_matrix=identity,
    )
    manifest_path = tmp_path / "trio_manifest.json"
    output_dir = tmp_path / "batch_output"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "stage": "stage1",
                    "top_k": 1,
                    "use_validation_workspace_transforms": True,
                },
                "scenes": [
                    {
                        "scene_id": "scene_one",
                        "sweep_dir": str(sweep_dir),
                        "dataset_id": "d1",
                        "validation_workspace": str(validation_workspace),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_batch_from_manifest(manifest_path, output_dir)

    assert summary["num_successful"] == 1
    assert summary["scenes"][0]["registration_profile"] == "validate_workspace_transform_v1"
    assert summary["scenes"][0]["registration_transform_source"] == "identity_lf_grid"
    scene_summary = json.loads(
        (output_dir / "scene_one" / "trio_benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert scene_summary["config"]["use_validation_workspace_transforms"] is True
    assert scene_summary["config"]["registration_profile"] == "validate_workspace_transform_v1"


def test_batch_main_emits_scene_progress_to_stderr(tmp_path, capsys):
    sweep_dir = _create_modern_fixture(
        tmp_path,
        include_stage2_summary=False,
        include_workspace=True,
        stage_mode="single_stage",
    )
    validation_workspace = tmp_path / "validation_workspace.npz"
    reference = np.full((4, 4), 1.25, dtype=np.float32)
    np.savez_compressed(
        validation_workspace,
        validation_processed=reference,
        thorcam_processed=reference,
    )
    manifest_path = tmp_path / "trio_manifest.json"
    output_dir = tmp_path / "batch_output"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "stage": "stage1",
                    "scale_search": "1.0:1.0:1",
                    "top_k": 1,
                },
                "scenes": [
                    {
                        "scene_id": "scene_one",
                        "sweep_dir": str(sweep_dir),
                        "dataset_id": "d1",
                        "validation_workspace": str(validation_workspace),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    main(["--manifest", str(manifest_path), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert "Starting scene 1/1: scene_one" in captured.err
    assert "Finished scene 1/1: scene_one" in captured.err

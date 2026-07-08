import json
import csv

import numpy as np

from moment_images.test_sweep_visualization import _create_modern_fixture
from moment_images.trio_benchmark_batch import run_batch_from_manifest


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

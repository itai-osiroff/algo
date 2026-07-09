import json
from types import SimpleNamespace

import numpy as np

from moment_images import trio_registration_annotator as annotator


def test_build_annotation_record_preserves_clicked_points_and_shapes():
    record = annotator.build_annotation_record(
        scene={"scene_id": "scene_a", "dataset_id": "d1", "registration_calibration_group": "rig_a"},
        source_kind="lf",
        anchor=SimpleNamespace(row_id="stage2:candidate_a", candidate_id="candidate_a"),
        validation=np.zeros((12, 16), dtype=np.float32),
        moving=np.zeros((8, 10), dtype=np.float32),
        point_pairs=[
            ((1.5, 2.5), (4.5, 5.5)),
            ((3.0, 4.0), (6.0, 7.0)),
        ],
    )

    assert record["scene_id"] == "scene_a"
    assert record["dataset_id"] == "d1"
    assert record["source_kind"] == "lf"
    assert record["calibration_group"] == "rig_a"
    assert record["anchor_row_id"] == "stage2:candidate_a"
    assert record["reference_shape"] == [12, 16]
    assert record["moving_shape"] == [8, 10]
    assert record["points"] == [
        {"reference": [1.5, 2.5], "moving": [4.5, 5.5]},
        {"reference": [3.0, 4.0], "moving": [6.0, 7.0]},
    ]


def test_write_annotation_payload_round_trips_json(tmp_path):
    output_path = tmp_path / "annotations.json"

    annotator.write_annotation_payload(
        output_path,
        [
            {
                "scene_id": "scene_a",
                "dataset_id": "d1",
                "source_kind": "thorcam",
                "calibration_group": "rig_a",
                "reference_shape": [12, 16],
                "moving_shape": [8, 10],
                "points": [{"reference": [1.0, 2.0], "moving": [3.0, 4.0]}],
            }
        ],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["annotations"][0]["source_kind"] == "thorcam"

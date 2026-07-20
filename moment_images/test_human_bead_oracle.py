import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moment_images.human_bead_oracle import (
    HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN,
    HUMAN_BEAD_ELONGATION_MEAN,
    HUMAN_BEAD_ORACLE_SCORE,
    HUMAN_BEAD_RESIDUAL_GRADIENT_MAX,
    HUMAN_BEAD_RESIDUAL_GRADIENT_P95,
    HumanBead,
    HumanBeadDatasetAnnotations,
    compute_human_bead_oracle_metrics,
    build_arg_parser,
    load_human_bead_annotations,
    main,
    scaled_beads_for_image,
    write_human_bead_annotations,
)


def _gaussian_bead(
    shape: tuple[int, int] = (64, 64),
    *,
    center: tuple[float, float] = (32.0, 32.0),
    sigma_x: float = 2.0,
    sigma_y: float | None = None,
) -> np.ndarray:
    if sigma_y is None:
        sigma_y = sigma_x
    yy, xx = np.indices(shape, dtype=np.float32)
    cx, cy = center
    values = np.exp(
        -0.5
        * (
            np.square((xx - float(cx)) / float(sigma_x))
            + np.square((yy - float(cy)) / float(sigma_y))
        )
    )
    return values.astype(np.float32)


def test_annotation_json_round_trips_and_rejects_duplicate_datasets(tmp_path: Path):
    annotations = {
        "d1": HumanBeadDatasetAnnotations(
            dataset_id="d1",
            image_path="preview.png",
            image_shape=(64, 64),
            default_radius_px=4.0,
            beads=[HumanBead(bead_id="b001", x=32.0, y=31.0, radius_px=4.0)],
        )
    }
    path = tmp_path / "beads.json"

    write_human_bead_annotations(path, annotations)
    loaded = load_human_bead_annotations(path)

    assert loaded["d1"].image_shape == (64, 64)
    assert loaded["d1"].beads[0].bead_id == "b001"
    assert loaded["d1"].beads[0].radius_px == pytest.approx(4.0)

    duplicate_payload = {
        "version": 1,
        "datasets": [
            {
                "dataset_id": "d1",
                "image_shape": [64, 64],
                "default_radius_px": 4,
                "beads": [{"id": "b001", "x": 32, "y": 31, "radius_px": 4}],
            },
            {
                "dataset_id": "d1",
                "image_shape": [64, 64],
                "default_radius_px": 4,
                "beads": [{"id": "b002", "x": 12, "y": 13, "radius_px": 4}],
            },
        ],
    }
    path.write_text(json.dumps(duplicate_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_human_bead_annotations(path)


def test_annotation_loading_rejects_invalid_radius(tmp_path: Path):
    path = tmp_path / "invalid_radius.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "dataset_id": "d1",
                        "image_shape": [64, 64],
                        "default_radius_px": 4,
                        "beads": [{"id": "b001", "x": 32, "y": 31, "radius_px": 0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="radius"):
        load_human_bead_annotations(path)


def test_scaled_beads_accepts_uniform_scaling_and_rejects_non_uniform_shape():
    annotation = HumanBeadDatasetAnnotations(
        dataset_id="d1",
        image_path=None,
        image_shape=(64, 64),
        default_radius_px=4.0,
        beads=[HumanBead(bead_id="b001", x=16.0, y=20.0, radius_px=4.0)],
    )

    scaled = scaled_beads_for_image(annotation, (128, 128))

    assert scaled[0].x == pytest.approx(32.0)
    assert scaled[0].y == pytest.approx(40.0)
    assert scaled[0].radius_px == pytest.approx(8.0)

    with pytest.raises(ValueError, match="non-uniform"):
        scaled_beads_for_image(annotation, (128, 96))


def test_human_bead_metrics_prefer_compact_isotropic_candidate():
    annotation = HumanBeadDatasetAnnotations(
        dataset_id="d1",
        image_path=None,
        image_shape=(64, 64),
        default_radius_px=4.0,
        beads=[
            HumanBead(bead_id="b001", x=20.0, y=20.0, radius_px=4.0),
            HumanBead(bead_id="b002", x=44.0, y=20.0, radius_px=4.0),
            HumanBead(bead_id="b003", x=32.0, y=44.0, radius_px=4.0),
        ],
    )
    compact = (
        _gaussian_bead(center=(20, 20), sigma_x=1.8)
        + _gaussian_bead(center=(44, 20), sigma_x=1.8)
        + _gaussian_bead(center=(32, 44), sigma_x=1.8)
    )
    blurred = (
        _gaussian_bead(center=(20, 20), sigma_x=4.5)
        + _gaussian_bead(center=(44, 20), sigma_x=4.5)
        + _gaussian_bead(center=(32, 44), sigma_x=4.5)
    )
    elongated = (
        _gaussian_bead(center=(20, 20), sigma_x=5.5, sigma_y=1.4)
        + _gaussian_bead(center=(44, 20), sigma_x=5.5, sigma_y=1.4)
        + _gaussian_bead(center=(32, 44), sigma_x=5.5, sigma_y=1.4)
    )

    compact_result = compute_human_bead_oracle_metrics(compact, annotation)
    blurred_result = compute_human_bead_oracle_metrics(blurred, annotation)
    elongated_result = compute_human_bead_oracle_metrics(elongated, annotation)

    assert compact_result.rankable
    assert compact_result.metrics[HUMAN_BEAD_ORACLE_SCORE] > blurred_result.metrics[HUMAN_BEAD_ORACLE_SCORE]
    assert compact_result.metrics[HUMAN_BEAD_ORACLE_SCORE] > elongated_result.metrics[HUMAN_BEAD_ORACLE_SCORE]
    assert compact_result.metrics[HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN] > blurred_result.metrics[HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN]
    assert elongated_result.metrics[HUMAN_BEAD_ELONGATION_MEAN] > compact_result.metrics[HUMAN_BEAD_ELONGATION_MEAN]


def test_human_bead_metrics_include_optional_residual_gradient_diagnostics():
    annotation = HumanBeadDatasetAnnotations(
        dataset_id="d1",
        image_path=None,
        image_shape=(32, 32),
        default_radius_px=4.0,
        beads=[HumanBead(bead_id="b001", x=16.0, y=16.0, radius_px=4.0)],
    )
    image = _gaussian_bead(shape=(32, 32), center=(16, 16), sigma_x=1.8)
    score_maps = np.zeros((32, 32, 2, 2), dtype=np.float32)
    score_maps[16:, :, :, :] = 1.0

    without_scores = compute_human_bead_oracle_metrics(image, annotation)
    with_scores = compute_human_bead_oracle_metrics(
        image,
        annotation,
        workspace_payload={"pixel_blended_scores": score_maps},
    )

    assert HUMAN_BEAD_RESIDUAL_GRADIENT_P95 not in without_scores.metrics
    assert with_scores.metrics[HUMAN_BEAD_RESIDUAL_GRADIENT_P95] >= 0.0
    assert with_scores.metrics[HUMAN_BEAD_RESIDUAL_GRADIENT_MAX] > 0.0


def test_cli_annotate_help_smoke(capsys):
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["annotate", "--help"])

    assert exc_info.value.code == 0
    assert "dataset-image" in capsys.readouterr().out


def test_cli_preview_and_score_workspace_smoke(tmp_path: Path):
    image = _gaussian_bead(shape=(32, 32), center=(16, 16), sigma_x=1.8)
    preview_image = (image / np.max(image) * 255.0).astype(np.uint8)
    Image.fromarray(preview_image).save(tmp_path / "preview.png")
    annotations = {
        "d1": HumanBeadDatasetAnnotations(
            dataset_id="d1",
            image_path="preview.png",
            image_shape=(32, 32),
            default_radius_px=4.0,
            beads=[HumanBead(bead_id="b001", x=16.0, y=16.0, radius_px=4.0)],
        )
    }
    annotations_path = tmp_path / "beads.json"
    write_human_bead_annotations(annotations_path, annotations)
    preview_dir = tmp_path / "previews"

    main(
        [
            "preview",
            "--annotations",
            str(annotations_path),
            "--output-dir",
            str(preview_dir),
        ]
    )

    assert (preview_dir / "d1_human_beads.png").exists()

    workspace_path = tmp_path / "workspace.npz"
    np.savez(workspace_path, average_rectified_image=image.astype(np.float32))
    output_path = tmp_path / "score.json"

    main(
        [
            "score-workspace",
            "--annotations",
            str(annotations_path),
            "--dataset-id",
            "d1",
            "--workspace",
            str(workspace_path),
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rankable"] is True
    assert HUMAN_BEAD_ORACLE_SCORE in payload["metrics"]

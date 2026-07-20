import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moment_images.human_bead_oracle import (
    HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN,
    HUMAN_BEAD_ELONGATION_MEAN,
    HUMAN_BEAD_ORACLE_SCORE,
    HUMAN_BEAD_PER_BEAD_METRICS,
    HUMAN_BEAD_RESIDUAL_GRADIENT_MAX,
    HUMAN_BEAD_RESIDUAL_GRADIENT_P95,
    HumanBead,
    HumanBeadDatasetAnnotations,
    _AnnotationSession,
    build_human_bead_oracle_import,
    build_arg_parser,
    compute_human_bead_oracle_metrics,
    load_human_bead_annotations,
    main,
    scaled_beads_for_image,
    write_human_bead_annotations,
)
from moment_images.human_bead_workbench import (
    HumanBeadWorkbenchState,
    collect_dataset_images_from_folder,
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


def test_annotation_session_preserves_existing_datasets_not_in_current_run(tmp_path: Path):
    first_image = np.full((32, 32), 128, dtype=np.uint8)
    Image.fromarray(first_image).save(tmp_path / "d1.png")
    annotations_path = tmp_path / "beads.json"
    write_human_bead_annotations(
        annotations_path,
        {
            "d1": HumanBeadDatasetAnnotations(
                dataset_id="d1",
                image_path="d1.png",
                image_shape=(32, 32),
                default_radius_px=4.0,
                beads=[HumanBead(bead_id="b001", x=12.0, y=12.0, radius_px=4.0)],
            ),
            "d2": HumanBeadDatasetAnnotations(
                dataset_id="d2",
                image_path="d2.png",
                image_shape=(32, 32),
                default_radius_px=5.0,
                beads=[HumanBead(bead_id="b002", x=20.0, y=20.0, radius_px=5.0)],
            ),
        },
    )
    existing = load_human_bead_annotations(annotations_path)

    session = _AnnotationSession(
        {"d1": tmp_path / "d1.png"},
        annotations_path,
        existing=existing,
    )
    session.save()
    loaded = load_human_bead_annotations(annotations_path)

    assert sorted(loaded) == ["d1", "d2"]
    assert loaded["d2"].beads[0].bead_id == "b002"
    assert loaded["d2"].beads[0].x == pytest.approx(20.0)


def test_workbench_state_edits_beads_and_undo_redo(tmp_path: Path):
    state = HumanBeadWorkbenchState(
        {"d1": tmp_path / "d1.png"},
        tmp_path / "beads.json",
        image_shapes={"d1": (32, 32)},
        default_radius_px=4.0,
    )

    bead_index = state.add_bead(10.0, 12.0)
    state.move_selected_bead(11.5, 13.5)
    state.adjust_selected_radius(-10.0)

    bead = state.current_annotation.beads[bead_index]
    assert bead.bead_id == "b001"
    assert bead.x == pytest.approx(11.5)
    assert bead.y == pytest.approx(13.5)
    assert bead.radius_px == pytest.approx(0.5)

    state.delete_selected_bead()
    assert state.current_annotation.beads == []

    assert state.undo() is True
    assert len(state.current_annotation.beads) == 1
    assert state.current_annotation.beads[0].x == pytest.approx(11.5)

    assert state.redo() is True
    assert state.current_annotation.beads == []


def test_workbench_state_undo_after_add_clears_stale_selection(tmp_path: Path):
    state = HumanBeadWorkbenchState(
        {"d1": tmp_path / "d1.png"},
        tmp_path / "beads.json",
        image_shapes={"d1": (32, 32)},
    )

    state.add_bead(10.0, 12.0)

    assert state.undo() is True
    assert state.current_annotation.beads == []
    assert state.selected_index is None


def test_workbench_state_save_preserves_unopened_existing_datasets(tmp_path: Path):
    annotations_path = tmp_path / "beads.json"
    write_human_bead_annotations(
        annotations_path,
        {
            "d1": HumanBeadDatasetAnnotations(
                dataset_id="d1",
                image_path="d1.png",
                image_shape=(32, 32),
                default_radius_px=4.0,
                beads=[],
            ),
            "d2": HumanBeadDatasetAnnotations(
                dataset_id="d2",
                image_path="d2.png",
                image_shape=(64, 64),
                default_radius_px=5.0,
                beads=[HumanBead(bead_id="b001", x=20.0, y=21.0, radius_px=5.0)],
            ),
        },
    )
    existing = load_human_bead_annotations(annotations_path)
    state = HumanBeadWorkbenchState(
        {"d1": tmp_path / "d1.png"},
        annotations_path,
        existing=existing,
        image_shapes={"d1": (32, 32)},
        default_radius_px=4.0,
    )

    state.add_bead(8.0, 9.0)
    state.save()
    loaded = load_human_bead_annotations(annotations_path)

    assert sorted(loaded) == ["d1", "d2"]
    assert loaded["d1"].beads[0].x == pytest.approx(8.0)
    assert loaded["d2"].beads[0].x == pytest.approx(20.0)


def test_collect_dataset_images_from_folder_maps_supported_file_stems(tmp_path: Path):
    for name in ("scene_b.tif", "scene_a.PNG", "ignore.txt", "scene_c.jpeg"):
        (tmp_path / name).write_bytes(b"x")

    dataset_images = collect_dataset_images_from_folder(tmp_path)

    assert list(dataset_images) == ["scene_a", "scene_b", "scene_c"]
    assert dataset_images["scene_a"] == tmp_path / "scene_a.PNG"


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
    per_bead = compact_result.diagnostics[HUMAN_BEAD_PER_BEAD_METRICS]
    assert [item["bead_id"] for item in per_bead] == ["b001", "b002", "b003"]
    assert all(item["rankable"] is True for item in per_bead)
    assert HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN in per_bead[0]["metrics"]


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


def test_cli_annotate_gui_help_smoke(capsys):
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["annotate-gui", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "dataset-image" in help_text
    assert "dataset-folder" in help_text
    assert "default-radius-px" in help_text


def test_cli_annotate_gui_dispatches_to_workbench(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "manual.png"
    image_path.write_bytes(b"x")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "scene_a.png").write_bytes(b"x")
    output_path = tmp_path / "beads.json"
    calls = []

    def fake_launch(*, dataset_images, output_path, default_radius_px, parent=None):
        calls.append(
            {
                "dataset_images": dict(dataset_images),
                "output_path": output_path,
                "default_radius_px": default_radius_px,
                "parent": parent,
            }
        )
        return output_path

    monkeypatch.setattr(
        "moment_images.human_bead_oracle.launch_human_bead_workbench",
        fake_launch,
    )

    main(
        [
            "annotate-gui",
            "--dataset-image",
            f"manual={image_path}",
            "--dataset-folder",
            str(folder),
            "--output",
            str(output_path),
            "--default-radius-px",
            "6",
        ]
    )

    assert calls == [
        {
            "dataset_images": {"manual": image_path, "scene_a": folder / "scene_a.png"},
            "output_path": output_path,
            "default_radius_px": 6.0,
            "parent": None,
        }
    ]


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
    assert payload["diagnostics"][HUMAN_BEAD_PER_BEAD_METRICS][0]["bead_id"] == "b001"
    assert HUMAN_BEAD_CORE_ENERGY_FRACTION_MEAN in payload["diagnostics"][HUMAN_BEAD_PER_BEAD_METRICS][0]["metrics"]


def test_human_bead_oracle_import_uses_dataset_specific_source_paths(tmp_path: Path):
    for dataset_id, center in {"d1": (16.0, 16.0), "d2": (20.0, 20.0)}.items():
        run_dir = tmp_path / "runs" / "stage1" / "c_good" / dataset_id
        run_dir.mkdir(parents=True)
        np.savez(
            run_dir / "workspace.npz",
            average_rectified_image=_gaussian_bead(shape=(32, 32), center=center, sigma_x=1.8),
        )
    annotations = {
        "d1": HumanBeadDatasetAnnotations(
            dataset_id="d1",
            image_path=None,
            image_shape=(32, 32),
            default_radius_px=4.0,
            beads=[HumanBead(bead_id="b001", x=16.0, y=16.0, radius_px=4.0)],
        ),
        "d2": HumanBeadDatasetAnnotations(
            dataset_id="d2",
            image_path=None,
            image_shape=(32, 32),
            default_radius_px=4.0,
            beads=[HumanBead(bead_id="b001", x=20.0, y=20.0, radius_px=4.0)],
        ),
    }
    d1_annotations = tmp_path / "d1_beads.json"
    d2_annotations = tmp_path / "d2_beads.json"
    stage_results = [
        {
            "status": "ok",
            "stage": "stage1",
            "candidate_id": "c_good",
            "dataset_id": dataset_id,
            "run_dir": str(tmp_path / "runs" / "stage1" / "c_good" / dataset_id),
        }
        for dataset_id in ("d1", "d2")
    ]

    import_data = build_human_bead_oracle_import(
        stage_name="stage1",
        stage_results=stage_results,
        annotations=annotations,
        annotation_source_paths_by_dataset={
            "d1": d1_annotations,
            "d2": d2_annotations,
        },
    )

    source_by_dataset = {record.dataset_id: record.source_path for record in import_data.records}
    assert source_by_dataset == {
        "d1": str(d1_annotations),
        "d2": str(d2_annotations),
    }
    assert import_data.source_paths == [str(d1_annotations), str(d2_annotations)]

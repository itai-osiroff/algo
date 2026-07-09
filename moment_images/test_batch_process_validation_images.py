from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from moment_images.batch_process_validation_images import (
    BatchAlignmentPreset,
    BatchProcessOptions,
    apply_processing_pipeline,
    build_output_path,
    list_supported_images,
    load_workspace_batch_preset,
    process_folder,
)
from moment_images.validate_rectification import (
    _apply_camera_defaults,
    _apply_geometry,
    _apply_intensity_window,
    _scale_image,
    _to_gray_float32,
    _warp_to_grid,
)


def _save_u8_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _write_validation_workspace(
    path: Path,
    *,
    validation_matrix: np.ndarray | None = None,
    thorcam_matrix: np.ndarray | None = None,
) -> None:
    rectified = np.zeros((6, 7), dtype=np.float32)
    validation_processed = np.zeros((8, 9), dtype=np.float32)
    thorcam_processed = np.zeros((5, 4), dtype=np.float32)
    nan_matrix = np.full((2, 3), np.nan, dtype=np.float32)
    np.savez_compressed(
        path,
        rectified_image=rectified,
        validation_processed=validation_processed,
        validation_mode=np.array("preserve_scale"),
        validation_scale_factors=np.array([0.75, 1.25], dtype=np.float32),
        validation_preblur_before_resize=np.array(False),
        validation_preserve_flux_sum=np.array(True),
        validation_intensity_window=np.array([12.0, 40.0], dtype=np.float32),
        validation_crop_box=np.array([1, 2, 5, 9], dtype=np.int32),
        validation_rotation_deg=np.array(90, dtype=np.int32),
        validation_flip_h=np.array(True),
        validation_flip_v=np.array(False),
        validation_alignment_status=np.array("point_aligned"),
        validation_alignment_model=np.array("point_similarity"),
        validation_alignment_final_warp_matrix=nan_matrix if validation_matrix is None else validation_matrix,
        thorcam_processed=thorcam_processed,
        thorcam_mode=np.array("fit_reference"),
        thorcam_scale_factors=np.array([1.5, 0.5], dtype=np.float32),
        thorcam_preblur_before_resize=np.array(True),
        thorcam_preserve_flux_sum=np.array(False),
        thorcam_intensity_window=np.array([3.0, 55.0], dtype=np.float32),
        thorcam_crop_box=np.array([-1, -1, -1, -1], dtype=np.int32),
        thorcam_rotation_deg=np.array(180, dtype=np.int32),
        thorcam_flip_h=np.array(False),
        thorcam_flip_v=np.array(True),
        thorcam_alignment_status=np.array("not_run"),
        thorcam_alignment_model=np.array("none"),
        thorcam_alignment_final_warp_matrix=nan_matrix if thorcam_matrix is None else thorcam_matrix,
    )


def test_apply_processing_pipeline_matches_validation_chain():
    src = np.arange(60, dtype=np.float32).reshape(10, 6)
    options = BatchProcessOptions(
        rotation_deg=90,
        flip_h=True,
        flip_v=False,
        crop_box=(1, 2, 5, 9),
        apply_intensity_window=True,
        intensity_min=12.0,
        intensity_max=40.0,
        scale_x=0.75,
        scale_y=1.5,
        preblur_before_resize=True,
        preserve_flux_sum=True,
    )

    actual = apply_processing_pipeline(src, options)

    base = _apply_camera_defaults(_to_gray_float32(src), camera_type=None)
    geom = _apply_geometry(
        base,
        rotation_deg=options.rotation_deg,
        flip_h=options.flip_h,
        flip_v=options.flip_v,
        crop_box=options.crop_box,
    )
    intensity = _apply_intensity_window(geom, options.intensity_min, options.intensity_max)
    expected = _scale_image(
        intensity,
        options.scale_x,
        options.scale_y,
        preblur_before_resize=options.preblur_before_resize,
        preserve_flux_sum=options.preserve_flux_sum,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_load_workspace_batch_preset_reads_validation_settings(tmp_path: Path):
    workspace_path = tmp_path / "validation_workspace.npz"
    final_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    _write_validation_workspace(workspace_path, validation_matrix=final_matrix)

    preset = load_workspace_batch_preset(workspace_path, "validation")

    assert preset.source_prefix == "validation"
    assert preset.options.rotation_deg == 90
    assert preset.options.flip_h is True
    assert preset.options.flip_v is False
    assert preset.options.crop_box == (1, 2, 5, 9)
    assert preset.options.apply_intensity_window is True
    assert preset.options.intensity_min == 12.0
    assert preset.options.intensity_max == 40.0
    assert preset.options.scale_x == np.float32(0.75)
    assert preset.options.scale_y == np.float32(1.25)
    assert preset.options.preblur_before_resize is False
    assert preset.options.preserve_flux_sum is True
    assert preset.output_shape == (6, 7)
    assert preset.alignment_status == "point_aligned"
    assert preset.alignment_model == "point_similarity"
    np.testing.assert_allclose(preset.final_alignment_matrix, final_matrix)


def test_load_workspace_batch_preset_reads_thorcam_settings(tmp_path: Path):
    workspace_path = tmp_path / "validation_workspace.npz"
    final_matrix = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    _write_validation_workspace(workspace_path, thorcam_matrix=final_matrix)

    preset = load_workspace_batch_preset(workspace_path, "thorcam")

    assert preset.source_prefix == "thorcam"
    assert preset.options.rotation_deg == 180
    assert preset.options.flip_h is False
    assert preset.options.flip_v is True
    assert preset.options.crop_box is None
    assert preset.options.intensity_min == 3.0
    assert preset.options.intensity_max == 55.0
    assert preset.options.scale_x == np.float32(1.5)
    assert preset.options.scale_y == np.float32(0.5)
    assert preset.options.preblur_before_resize is True
    assert preset.options.preserve_flux_sum is False
    assert preset.output_shape == (5, 4)
    np.testing.assert_allclose(preset.final_alignment_matrix, final_matrix)


def test_load_workspace_batch_preset_ignores_missing_final_alignment_matrix(tmp_path: Path):
    workspace_path = tmp_path / "validation_workspace.npz"
    _write_validation_workspace(workspace_path)

    preset = load_workspace_batch_preset(workspace_path, "validation")

    assert preset.final_alignment_matrix is None
    assert preset.output_shape is None


def test_apply_processing_pipeline_applies_fixed_alignment_after_preprocessing():
    src = np.zeros((6, 8), dtype=np.float32)
    src[2:4, 3:5] = 100.0
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    preset = BatchAlignmentPreset(
        source_prefix="validation",
        options=BatchProcessOptions(),
        final_alignment_matrix=matrix,
        output_shape=src.shape,
        alignment_status="point_aligned",
        alignment_model="point_similarity",
    )

    actual = apply_processing_pipeline(src, preset.options, alignment_preset=preset)
    expected, _ = _warp_to_grid(src, matrix, src.shape)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=True)


def test_apply_processing_pipeline_without_matrix_keeps_preprocessing_only():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    options = BatchProcessOptions(scale_x=1.0, scale_y=1.0)
    preset = BatchAlignmentPreset(
        source_prefix="validation",
        options=options,
        final_alignment_matrix=None,
        output_shape=None,
        alignment_status="not_run",
        alignment_model="none",
    )

    actual = apply_processing_pipeline(src, options, alignment_preset=preset)
    expected = apply_processing_pipeline(src, options)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_list_supported_images_is_top_level_non_recursive_and_sorted(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "nested").mkdir()

    _save_u8_image(input_dir / "b.JPG", np.full((4, 4), 10, dtype=np.uint8))
    _save_u8_image(input_dir / "a.png", np.full((4, 4), 20, dtype=np.uint8))
    _save_u8_image(input_dir / "nested" / "inner.png", np.full((4, 4), 30, dtype=np.uint8))
    (input_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    discovered = list_supported_images(input_dir)
    names = [path.name for path in discovered]
    assert names == ["a.png", "b.JPG"]


def test_build_output_path_adds_processed_suffix(tmp_path: Path):
    source = tmp_path / "frame_001.tif"
    output_dir = tmp_path / "out"
    output = build_output_path(source, output_dir)
    assert output == output_dir / "frame_001_processed.tif"


def test_build_output_path_converts_non_tiff_inputs_to_tiff(tmp_path: Path):
    source = tmp_path / "frame_001.png"
    output_dir = tmp_path / "out"
    output = build_output_path(source, output_dir)
    assert output == output_dir / "frame_001_processed.tiff"


def test_process_folder_skips_existing_outputs(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    _save_u8_image(input_dir / "sample.png", np.full((8, 8), 125, dtype=np.uint8))
    _save_u8_image(output_dir / "sample_processed.tiff", np.full((8, 8), 5, dtype=np.uint8))

    summary = process_folder(input_dir, output_dir, BatchProcessOptions())

    assert len(summary.processed) == 0
    assert len(summary.skipped) == 1
    assert len(summary.failed) == 0
    assert summary.skipped[0].input_path.name == "sample.png"


def test_process_folder_continues_after_failure(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    _save_u8_image(input_dir / "good.png", np.full((9, 11), 80, dtype=np.uint8))
    (input_dir / "bad.png").write_text("this is not a valid image file", encoding="utf-8")

    summary = process_folder(input_dir, output_dir, BatchProcessOptions())

    assert len(summary.processed) == 1
    assert len(summary.failed) == 1
    assert len(summary.skipped) == 0
    assert (output_dir / "good_processed.tiff").exists()
    assert summary.failed[0].input_path.name == "bad.png"


def test_process_folder_saves_float32_output_even_for_uint16_input(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    uint16_img = np.array([[0, 1000], [2000, 65535]], dtype=np.uint16)
    Image.fromarray(uint16_img).save(input_dir / "raw.tiff")

    summary = process_folder(input_dir, output_dir, BatchProcessOptions())
    assert len(summary.processed) == 1
    assert len(summary.failed) == 0

    output = np.asarray(Image.open(output_dir / "raw_processed.tiff"))
    assert output.dtype == np.float32


def test_process_folder_applies_alignment_preset(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    src = np.zeros((6, 8), dtype=np.float32)
    src[2:4, 3:5] = 100.0
    Image.fromarray(src).save(input_dir / "raw.tiff")
    preset = BatchAlignmentPreset(
        source_prefix="validation",
        options=BatchProcessOptions(),
        final_alignment_matrix=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32),
        output_shape=src.shape,
        alignment_status="point_aligned",
        alignment_model="point_similarity",
    )

    summary = process_folder(input_dir, output_dir, preset.options, alignment_preset=preset)

    assert len(summary.processed) == 1
    output = np.asarray(Image.open(output_dir / "raw_processed.tiff"))
    expected, _ = _warp_to_grid(src, preset.final_alignment_matrix, src.shape)
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-6, equal_nan=True)

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from moment_images.batch_process_validation_images import (
    BatchProcessOptions,
    apply_processing_pipeline,
    build_output_path,
    list_supported_images,
    process_folder,
)
from moment_images.validate_rectification import (
    _apply_camera_defaults,
    _apply_geometry,
    _apply_intensity_window,
    _scale_image,
    _to_gray_float32,
)


def _save_u8_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


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

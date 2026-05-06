import sys
import types

import cv2
import numpy as np

pg_class_stub = types.ModuleType("PG_class")
pg_class_stub.PG_controller = object
sys.modules.setdefault("PG_class", pg_class_stub)

from acquire_images_PG import (
    KEY_EXPOSURE_UP,
    MAX_PREVIEW_ZOOM,
    PreviewState,
    ZOOM_STEP_IN,
    _apply_zoom_step,
    _on_mouse,
    _reset_zoom,
    adjust_exposure,
    adjust_gain,
    build_preview_display,
    capture_snapshot,
    compute_zoom_crop,
    handle_preview_key,
)


class FakePG:
    def __init__(self):
        self.exposure = 100.0
        self.gain = 2.5
        self.set_exposure_calls = 0
        self.set_gain_calls = 0

    def get_exp(self):
        return self.exposure

    def set_exposure(self, new_exp):
        self.set_exposure_calls += 1
        self.exposure = min(120.0, max(80.0, float(new_exp)))
        return self.exposure

    def get_gain(self):
        return self.gain

    def set_gain(self, new_gain):
        self.set_gain_calls += 1
        self.gain = min(3.0, max(0.0, float(new_gain)))
        return self.gain


def test_compute_zoom_crop_clamps_to_image_edges():
    assert compute_zoom_crop((100, 200), zoom_factor=4.0, center_x=5, center_y=4) == (
        0,
        25,
        0,
        50,
    )
    assert compute_zoom_crop((100, 200), zoom_factor=4.0, center_x=198, center_y=98) == (
        75,
        100,
        150,
        200,
    )


def test_on_mouse_maps_display_coords_back_to_raw_when_zoomed():
    raw_frame = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)
    state = PreviewState(zoom_factor=2.0, zoom_center_x=150, zoom_center_y=70)

    build_preview_display(
        raw_frame=raw_frame,
        state=state,
        global_row0=0,
        global_col0=0,
        mode_label="stream",
        max_window_width=200,
        max_window_height=200,
    )

    _on_mouse(
        cv2.EVENT_MOUSEMOVE,
        10,
        state.display_header_height + 12,
        0,
        state,
    )

    assert state.mouse_valid is True
    assert state.mouse_x == 105
    assert state.mouse_y == 51


def test_zoom_step_clamps_and_reset_clears_center():
    state = PreviewState(
        zoom_factor=MAX_PREVIEW_ZOOM,
        zoom_center_x=10,
        zoom_center_y=11,
    )

    zoom_factor = _apply_zoom_step(state, ZOOM_STEP_IN)
    assert zoom_factor == MAX_PREVIEW_ZOOM

    _reset_zoom(state)
    assert state.zoom_factor == 1.0
    assert state.zoom_center_x is None
    assert state.zoom_center_y is None


def test_adjust_exposure_and_gain_use_controller_clamps():
    pg = FakePG()
    state = PreviewState(current_exposure=100.0, current_gain=2.5)

    exposure = adjust_exposure(pg=pg, state=state, increase=True)
    assert exposure == 120.0
    assert state.current_exposure == 120.0

    state.current_gain = 2.5
    gain = adjust_gain(pg=pg, state=state, increase=True)
    assert gain == 3.0
    assert state.current_gain == 3.0


def test_capture_snapshot_saves_raw_frame_while_zoomed(tmp_path):
    raw_frame = np.arange(12, dtype=np.uint16).reshape(3, 4)
    state = PreviewState(format_index=1, zoom_factor=4.0, zoom_center_x=2, zoom_center_y=1)

    build_preview_display(
        raw_frame=raw_frame,
        state=state,
        global_row0=0,
        global_col0=0,
        mode_label="stream",
        max_window_width=200,
        max_window_height=200,
    )

    saved_path = capture_snapshot(
        raw_frame=raw_frame,
        state=state,
        output_dir=tmp_path,
        mode_label="stream",
    )

    loaded = np.load(saved_path)
    np.testing.assert_array_equal(loaded, raw_frame)


def test_handle_preview_key_rejects_live_exposure_in_single_mode(tmp_path):
    pg = FakePG()
    state = PreviewState(current_exposure=100.0)
    raw_frame = np.zeros((4, 4), dtype=np.uint16)

    should_continue = handle_preview_key(
        key=KEY_EXPOSURE_UP,
        state=state,
        pg=pg,
        mode="single",
        raw_frame=raw_frame,
        output_dir=tmp_path,
    )

    assert should_continue is True
    assert "stream mode" in state.status_text
    assert pg.set_exposure_calls == 0

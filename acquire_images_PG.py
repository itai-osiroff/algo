import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PG_class import PG_controller


SAVE_FORMAT_CYCLE = ("tiff", "npy", "png")
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_FORMAT = ord("f")
KEY_ZOOM_IN = ord("i")
KEY_ZOOM_OUT = ord("o")
KEY_ZOOM_RESET = ord("0")
KEY_EXPOSURE_DOWN = ord("[")
KEY_EXPOSURE_UP = ord("]")
KEY_GAIN_DOWN = ord(",")
KEY_GAIN_UP = ord(".")
DEFAULT_MAX_WINDOW_WIDTH = 1600
DEFAULT_MAX_WINDOW_HEIGHT = 900
WINDOW_FRAME_MARGIN_WIDTH = 80
WINDOW_FRAME_MARGIN_HEIGHT = 140
PREVIEW_LINE_HEIGHT = 24
PREVIEW_HEADER_PADDING = 8
PREVIEW_TEXT_ORIGIN_X = 10
PREVIEW_TEXT_LINE1_Y = 22
PREVIEW_TEXT_LINE2_Y = 46
PREVIEW_TEXT_LINE3_Y = 70
PREVIEW_TEXT_FONT_SCALE = 0.55
PREVIEW_TEXT_THICKNESS = 1
MIN_PREVIEW_ZOOM = 1.0
MAX_PREVIEW_ZOOM = 8.0
ZOOM_STEP_IN = 1.25
ZOOM_STEP_OUT = 0.8
EXPOSURE_STEP_UP = 1.25
EXPOSURE_STEP_DOWN = 0.8
GAIN_STEP = 1.0
STATUS_TTL_FRAMES = 90


@dataclass
class PreviewState:
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_valid: bool = False
    format_index: int = 0
    frame_width: int = 0
    frame_height: int = 0
    display_scale: float = 1.0
    display_header_height: int = 0
    display_frame_width: int = 0
    display_frame_height: int = 0
    view_row0: int = 0
    view_col0: int = 0
    view_width: int = 0
    view_height: int = 0
    zoom_factor: float = 1.0
    zoom_center_x: Optional[int] = None
    zoom_center_y: Optional[int] = None
    current_exposure: Optional[float] = None
    current_gain: Optional[float] = None
    status_text: str = ""
    status_frames_remaining: int = 0


def centered_roi(sensor_height, sensor_width, crop_height, crop_width, row_offset=0, col_offset=0):
    """Return ROI as [r1, r2, c1, c2] (inclusive bounds) around frame center."""
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("crop_height and crop_width must be positive.")
    if crop_height > sensor_height or crop_width > sensor_width:
        raise ValueError("Crop size cannot be larger than sensor size.")

    center_row = sensor_height // 2 + int(row_offset)
    center_col = sensor_width // 2 + int(col_offset)

    half_h = crop_height // 2
    half_w = crop_width // 2

    r1 = center_row - half_h
    c1 = center_col - half_w

    r1 = max(0, min(r1, sensor_height - crop_height))
    c1 = max(0, min(c1, sensor_width - crop_width))

    r2 = r1 + crop_height - 1
    c2 = c1 + crop_width - 1
    return [int(r1), int(r2), int(c1), int(c2)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live/raw preview for Point Grey camera with pixel readout and snapshots."
    )
    parser.add_argument(
        "--mode",
        choices=("stream", "single"),
        default="stream",
        help="Preview mode: continuous stream or one captured frame.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional snapshot output directory. If omitted, prompts at startup.",
    )
    parser.add_argument(
        "--pixel-format",
        default="Mono8",
        help="Camera pixel format (for example Mono8 or Mono16).",
    )
    parser.add_argument(
        "--adc-bit-depth",
        default="Bit12",
        help="Camera ADC bit depth (for example Bit12).",
    )
    return parser.parse_args()


def _validate_or_create_output_dir(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, f"Cannot create directory '{path}': {exc}"

    if not path.is_dir():
        return False, f"Path '{path}' is not a directory."

    probe_path = path / ".pg_write_probe.tmp"
    try:
        with open(probe_path, "wb") as probe:
            probe.write(b"ok")
        probe_path.unlink()
    except Exception as exc:
        return False, f"Directory '{path}' is not writable: {exc}"

    return True, ""


def resolve_output_dir(cli_output_dir):
    if cli_output_dir:
        candidate = Path(cli_output_dir).expanduser()
        ok, reason = _validate_or_create_output_dir(candidate)
        if ok:
            return candidate
        print(f"Invalid --output-dir: {reason}")

    while True:
        text = input("Enter snapshot output directory: ").strip()
        if not text:
            print("Output directory cannot be empty.")
            continue
        candidate = Path(text).expanduser()
        ok, reason = _validate_or_create_output_dir(candidate)
        if ok:
            return candidate
        print(f"Invalid output directory: {reason}")


def _on_mouse(event, x, y, _flags, state):
    if event < 0:
        return

    if (
        state.display_scale <= 0
        or state.frame_width <= 0
        or state.frame_height <= 0
        or state.view_width <= 0
        or state.view_height <= 0
    ):
        state.mouse_valid = False
        return

    local_y = int(y) - state.display_header_height
    local_x = int(x)

    if (
        local_x < 0
        or local_y < 0
        or local_x >= state.display_frame_width
        or local_y >= state.display_frame_height
    ):
        state.mouse_valid = False
        return

    raw_x = state.view_col0 + min(int(local_x / state.display_scale), state.view_width - 1)
    raw_y = state.view_row0 + min(int(local_y / state.display_scale), state.view_height - 1)
    state.mouse_x = min(raw_x, state.frame_width - 1)
    state.mouse_y = min(raw_y, state.frame_height - 1)
    state.mouse_valid = True

    if event == cv2.EVENT_LBUTTONDOWN:
        state.zoom_center_x = state.mouse_x
        state.zoom_center_y = state.mouse_y
        _set_status(
            state,
            f"Zoom center locked at ({state.mouse_x},{state.mouse_y})",
        )


def _snapshot_format(state):
    return SAVE_FORMAT_CYCLE[state.format_index]


def _cycle_snapshot_format(state):
    state.format_index = (state.format_index + 1) % len(SAVE_FORMAT_CYCLE)
    return _snapshot_format(state)


def _set_status(state, message, ttl=STATUS_TTL_FRAMES):
    state.status_text = str(message)
    state.status_frames_remaining = max(0, int(ttl))


def _tick_status(state):
    if state.status_frames_remaining <= 0:
        state.status_text = ""
        return

    state.status_frames_remaining -= 1
    if state.status_frames_remaining <= 0:
        state.status_text = ""


def _clamp_zoom_factor(zoom_factor):
    return float(np.clip(float(zoom_factor), MIN_PREVIEW_ZOOM, MAX_PREVIEW_ZOOM))


def _apply_zoom_step(state, multiplier):
    state.zoom_factor = _clamp_zoom_factor(state.zoom_factor * float(multiplier))
    return state.zoom_factor


def _reset_zoom(state):
    state.zoom_factor = MIN_PREVIEW_ZOOM
    state.zoom_center_x = None
    state.zoom_center_y = None


def compute_zoom_crop(frame_shape, zoom_factor, center_x=None, center_y=None):
    frame_h = int(frame_shape[0])
    frame_w = int(frame_shape[1])
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("frame_shape must describe a non-empty image.")

    zoom_factor = _clamp_zoom_factor(zoom_factor)
    view_h = min(frame_h, max(1, int(round(frame_h / zoom_factor))))
    view_w = min(frame_w, max(1, int(round(frame_w / zoom_factor))))

    if center_x is None:
        center_x = frame_w // 2
    if center_y is None:
        center_y = frame_h // 2

    center_x = int(np.clip(int(center_x), 0, frame_w - 1))
    center_y = int(np.clip(int(center_y), 0, frame_h - 1))

    col0 = center_x - view_w // 2
    row0 = center_y - view_h // 2
    col0 = int(np.clip(col0, 0, frame_w - view_w))
    row0 = int(np.clip(row0, 0, frame_h - view_h))

    return row0, row0 + view_h, col0, col0 + view_w


def _ensure_single_image(frame):
    frame = np.asarray(frame)
    frame = np.squeeze(frame)
    if frame.ndim != 2:
        raise ValueError(
            f"Expected a single 2D frame for preview; got shape={frame.shape}."
        )
    return frame


def _format_pixel_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _format_status_value(value, precision=1):
    if value is None:
        return "-"

    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def _preview_info_lines(raw_frame, state, global_row0, global_col0, mode_label):
    h, w = raw_frame.shape[:2]
    info_line1 = (
        f"mode={mode_label} save={_snapshot_format(state)} "
        "keys:q s f click i/o 0 [/] ,/."
    )
    info_line2 = "local(x,y)=(-,-) global(x,y)=(-,-) value=-"

    if state.mouse_valid and (0 <= state.mouse_x < w) and (0 <= state.mouse_y < h):
        local_x = state.mouse_x
        local_y = state.mouse_y
        global_x = global_col0 + local_x
        global_y = global_row0 + local_y
        value = raw_frame[local_y, local_x]
        info_line2 = (
            f"local(x,y)=({local_x},{local_y}) "
            f"global(x,y)=({global_x},{global_y}) "
            f"value={_format_pixel_value(value)}"
        )

    status_text = state.status_text if state.status_text else "-"
    info_line3 = (
        f"zoom={state.zoom_factor:.2f}x "
        f"exp={_format_status_value(state.current_exposure, precision=0)}us "
        f"gain={_format_status_value(state.current_gain, precision=1)}dB "
        f"status={status_text}"
    )

    return info_line1, info_line2, info_line3


def _frame_to_display_u8(raw_frame):
    if raw_frame.dtype == np.uint8:
        return raw_frame

    if raw_frame.dtype == np.bool_:
        return raw_frame.astype(np.uint8) * 255

    if np.issubdtype(raw_frame.dtype, np.unsignedinteger):
        shift = max(raw_frame.dtype.itemsize * 8 - 8, 0)
        if shift:
            return np.right_shift(raw_frame, shift).astype(np.uint8)
        return raw_frame.astype(np.uint8)

    if np.issubdtype(raw_frame.dtype, np.integer):
        clipped = np.clip(raw_frame, 0, None).astype(np.float32)
    else:
        clipped = np.nan_to_num(
            raw_frame.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clipped = np.clip(clipped, 0.0, None)

    max_value = float(np.max(clipped)) if clipped.size else 0.0
    if max_value <= 0:
        return np.zeros(raw_frame.shape, dtype=np.uint8)

    scale = 255.0 if max_value <= 1.0 else 255.0 / max_value
    return cv2.convertScaleAbs(clipped, alpha=scale)


def _preview_window_limits():
    try:
        import ctypes

        user32 = ctypes.windll.user32
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))
        if screen_width > 0 and screen_height > 0:
            return (
                max(320, screen_width - WINDOW_FRAME_MARGIN_WIDTH),
                max(240, screen_height - WINDOW_FRAME_MARGIN_HEIGHT),
            )
    except Exception:
        pass

    return DEFAULT_MAX_WINDOW_WIDTH, DEFAULT_MAX_WINDOW_HEIGHT


def _scaled_frame_shape(frame_width, frame_height, max_width, max_height, allow_upscale=False):
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    scale = min(max_width / frame_width, max_height / frame_height)
    if not allow_upscale:
        scale = min(1.0, scale)
    display_width = max(1, int(round(frame_width * scale)))
    display_height = max(1, int(round(frame_height * scale)))
    return scale, display_width, display_height


def build_preview_display(raw_frame, state, global_row0, global_col0, mode_label, max_window_width, max_window_height):
    raw_h, raw_w = raw_frame.shape[:2]
    row0, row1, col0, col1 = compute_zoom_crop(
        frame_shape=raw_frame.shape[:2],
        zoom_factor=state.zoom_factor,
        center_x=state.zoom_center_x,
        center_y=state.zoom_center_y,
    )
    view_frame = raw_frame[row0:row1, col0:col1]

    info_line1, info_line2, info_line3 = _preview_info_lines(
        raw_frame=raw_frame,
        state=state,
        global_row0=global_row0,
        global_col0=global_col0,
        mode_label=mode_label,
    )

    header_height = PREVIEW_LINE_HEIGHT * 3 + PREVIEW_HEADER_PADDING
    available_height = max(1, max_window_height - header_height)
    scale, display_w, display_h = _scaled_frame_shape(
        frame_width=view_frame.shape[1],
        frame_height=view_frame.shape[0],
        max_width=max_window_width,
        max_height=available_height,
        allow_upscale=state.zoom_factor > MIN_PREVIEW_ZOOM,
    )

    frame_display = _frame_to_display_u8(view_frame)
    if (display_w, display_h) != view_frame.shape[1::-1]:
        interpolation = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
        frame_display = cv2.resize(
            frame_display,
            (display_w, display_h),
            interpolation=interpolation,
        )

    display = np.zeros((header_height + display_h, display_w), dtype=np.uint8)
    display[header_height:, :] = frame_display

    state.frame_width = raw_w
    state.frame_height = raw_h
    state.display_scale = scale
    state.display_header_height = header_height
    state.display_frame_width = display_w
    state.display_frame_height = display_h
    state.view_row0 = row0
    state.view_col0 = col0
    state.view_width = view_frame.shape[1]
    state.view_height = view_frame.shape[0]

    cv2.putText(
        display,
        info_line1,
        (PREVIEW_TEXT_ORIGIN_X, PREVIEW_TEXT_LINE1_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        PREVIEW_TEXT_FONT_SCALE,
        255,
        PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        info_line2,
        (PREVIEW_TEXT_ORIGIN_X, PREVIEW_TEXT_LINE2_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        PREVIEW_TEXT_FONT_SCALE,
        255,
        PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        info_line3,
        (PREVIEW_TEXT_ORIGIN_X, PREVIEW_TEXT_LINE3_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        PREVIEW_TEXT_FONT_SCALE,
        255,
        PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return display


def _create_preview_window(window_name, state, display_shape):
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO

    display_h, display_w = display_shape[:2]
    cv2.namedWindow(window_name, flags)
    cv2.resizeWindow(window_name, int(display_w), int(display_h))
    cv2.setMouseCallback(window_name, _on_mouse, state)


def _window_is_closed(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def save_raw_snapshot(raw_frame, output_dir, mode_label, file_format):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"pg_{mode_label}_{timestamp}"
    extension = f".{file_format}"
    out_path = output_dir / f"{stem}{extension}"

    if file_format == "npy":
        np.save(out_path, raw_frame)
        return out_path

    ok = cv2.imwrite(str(out_path), raw_frame)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for '{out_path}'.")
    return out_path


def capture_snapshot(raw_frame, state, output_dir, mode_label):
    return save_raw_snapshot(
        raw_frame=raw_frame,
        output_dir=output_dir,
        mode_label=mode_label,
        file_format=_snapshot_format(state),
    )


def adjust_exposure(pg, state, increase):
    if state.current_exposure is None:
        state.current_exposure = float(pg.get_exp())

    multiplier = EXPOSURE_STEP_UP if increase else EXPOSURE_STEP_DOWN
    target_exposure = float(state.current_exposure) * multiplier
    applied_exposure = float(pg.set_exposure(target_exposure))
    state.current_exposure = applied_exposure
    return applied_exposure


def adjust_gain(pg, state, increase):
    if state.current_gain is None:
        state.current_gain = float(pg.get_gain())

    gain_step = GAIN_STEP if increase else -GAIN_STEP
    target_gain = float(state.current_gain) + gain_step
    applied_gain = float(pg.set_gain(target_gain))
    state.current_gain = applied_gain
    return applied_gain


def handle_preview_key(key, state, pg, mode, raw_frame, output_dir):
    if key == KEY_QUIT:
        return False

    if key == KEY_FORMAT:
        new_format = _cycle_snapshot_format(state)
        print(f"Snapshot format set to: {new_format}")
        _set_status(state, f"Snapshot format set to {new_format}")
        return True

    if key == KEY_SAVE:
        try:
            saved_path = capture_snapshot(
                raw_frame=raw_frame,
                state=state,
                output_dir=output_dir,
                mode_label=mode,
            )
            print(f"Saved snapshot: {saved_path}")
            _set_status(state, f"Saved snapshot: {saved_path.name}")
        except Exception as exc:
            print(f"Snapshot save failed: {exc}")
            _set_status(state, f"Snapshot save failed: {exc}")
        return True

    if key == KEY_ZOOM_IN:
        zoom_factor = _apply_zoom_step(state, ZOOM_STEP_IN)
        _set_status(state, f"Zoom set to {zoom_factor:.2f}x")
        return True

    if key == KEY_ZOOM_OUT:
        zoom_factor = _apply_zoom_step(state, ZOOM_STEP_OUT)
        _set_status(state, f"Zoom set to {zoom_factor:.2f}x")
        return True

    if key == KEY_ZOOM_RESET:
        _reset_zoom(state)
        _set_status(state, "Zoom reset")
        return True

    if key in (KEY_EXPOSURE_DOWN, KEY_EXPOSURE_UP):
        if mode != "stream":
            _set_status(state, "Exposure control is only available in stream mode")
            return True

        try:
            exposure = adjust_exposure(pg=pg, state=state, increase=(key == KEY_EXPOSURE_UP))
            _set_status(state, f"Exposure set to {_format_status_value(exposure, precision=0)}us")
        except Exception as exc:
            _set_status(state, f"Exposure update failed: {exc}")
        return True

    if key in (KEY_GAIN_DOWN, KEY_GAIN_UP):
        if mode != "stream":
            _set_status(state, "Gain control is only available in stream mode")
            return True

        try:
            gain = adjust_gain(pg=pg, state=state, increase=(key == KEY_GAIN_UP))
            _set_status(state, f"Gain set to {_format_status_value(gain, precision=1)}dB")
        except Exception as exc:
            _set_status(state, f"Gain update failed: {exc}")
        return True

    return True


def run_preview(pg, mode, output_dir, window_name="PG Preview"):
    state = PreviewState()
    roi = [int(v) for v in np.asarray(pg.big_roi, dtype=int).tolist()]
    global_row0 = roi[0]
    global_col0 = roi[2]
    max_window_width, max_window_height = _preview_window_limits()
    current_window_shape = None
    try:
        state.current_exposure = float(pg.get_exp())
    except Exception:
        state.current_exposure = None
    try:
        state.current_gain = float(pg.get_gain())
    except Exception:
        state.current_gain = None

    static_frame = None
    if mode == "single":
        static_frame = _ensure_single_image(pg.get_frame(split=False, full_screen=False))

    try:
        while True:
            if mode == "stream":
                raw_frame = _ensure_single_image(pg.get_frame(split=False, full_screen=False))
            else:
                raw_frame = static_frame

            preview = build_preview_display(
                raw_frame=raw_frame,
                state=state,
                global_row0=global_row0,
                global_col0=global_col0,
                mode_label=mode,
                max_window_width=max_window_width,
                max_window_height=max_window_height,
            )

            preview_shape = preview.shape[:2]
            if preview_shape != current_window_shape:
                _create_preview_window(window_name, state, preview.shape)
                current_window_shape = preview_shape

            cv2.imshow(window_name, preview)
            _tick_status(state)

            key = cv2.waitKey(1 if mode == "stream" else 30) & 0xFF
            if _window_is_closed(window_name):
                break
            if key != 0xFF:
                should_continue = handle_preview_key(
                    key=key,
                    state=state,
                    pg=pg,
                    mode=mode,
                    raw_frame=raw_frame,
                    output_dir=output_dir,
                )
                if not should_continue:
                    break
    finally:
        try:
            pg.stop()
        finally:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    # crop_height = 3104
    # crop_width = 3252
    crop_height = 4600
    crop_width = 5320
    row_offset = 0
    col_offset = 0
    sensor_width = 5320
    sensor_height = 4600
    roi = centered_roi(
        sensor_height=sensor_height,
        sensor_width=sensor_width,
        crop_height=crop_height,
        crop_width=crop_width,
        row_offset=row_offset,
        col_offset=col_offset,
    )

    output_dir = resolve_output_dir(args.output_dir)
    pg = PG_controller({
        "PG_num_mean": 1,
        "PG_exp": 10e3,
        "PG_gain": 15,
        "PG_sensor_width": sensor_width,
        "PG_sensor_height": sensor_height,
        "PG_single_roi": True,
        "PG_roi": roi,
        "PG_pixel_format": args.pixel_format,
        "PG_adc_bit_depth": args.adc_bit_depth,
    })
    run_preview(
        pg=pg,
        mode=args.mode,
        output_dir=output_dir,
        window_name=f"PG {args.mode} ROI {roi}",
    )


if __name__ == "__main__":
    main()

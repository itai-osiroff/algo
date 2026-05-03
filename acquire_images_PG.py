import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from PG_class import PG_controller


SAVE_FORMAT_CYCLE = ("tiff", "npy", "png")
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_FORMAT = ord("f")
DEFAULT_MAX_WINDOW_WIDTH = 1600
DEFAULT_MAX_WINDOW_HEIGHT = 900
WINDOW_FRAME_MARGIN_WIDTH = 80
WINDOW_FRAME_MARGIN_HEIGHT = 140
PREVIEW_LINE_HEIGHT = 24
PREVIEW_HEADER_PADDING = 8
PREVIEW_TEXT_ORIGIN_X = 10
PREVIEW_TEXT_LINE1_Y = 22
PREVIEW_TEXT_LINE2_Y = 46
PREVIEW_TEXT_FONT_SCALE = 0.55
PREVIEW_TEXT_THICKNESS = 1


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

    if state.display_scale <= 0 or state.frame_width <= 0 or state.frame_height <= 0:
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

    state.mouse_x = min(int(local_x / state.display_scale), state.frame_width - 1)
    state.mouse_y = min(int(local_y / state.display_scale), state.frame_height - 1)
    state.mouse_valid = True


def _snapshot_format(state):
    return SAVE_FORMAT_CYCLE[state.format_index]


def _cycle_snapshot_format(state):
    state.format_index = (state.format_index + 1) % len(SAVE_FORMAT_CYCLE)
    return _snapshot_format(state)


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


def _preview_info_lines(raw_frame, state, global_row0, global_col0, mode_label):
    h, w = raw_frame.shape[:2]
    info_line1 = (
        f"mode={mode_label} save={_snapshot_format(state)} "
        f"keys: q=quit s=snap f=format"
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

    return info_line1, info_line2


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


def _scaled_frame_shape(frame_width, frame_height, max_width, max_height):
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    scale = min(1.0, max_width / frame_width, max_height / frame_height)
    display_width = max(1, int(round(frame_width * scale)))
    display_height = max(1, int(round(frame_height * scale)))
    return scale, display_width, display_height


def build_preview_display(raw_frame, state, global_row0, global_col0, mode_label, max_window_width, max_window_height):
    raw_h, raw_w = raw_frame.shape[:2]
    info_line1, info_line2 = _preview_info_lines(
        raw_frame=raw_frame,
        state=state,
        global_row0=global_row0,
        global_col0=global_col0,
        mode_label=mode_label,
    )

    header_height = PREVIEW_LINE_HEIGHT * 2 + PREVIEW_HEADER_PADDING
    available_height = max(1, max_window_height - header_height)
    scale, display_w, display_h = _scaled_frame_shape(
        frame_width=raw_w,
        frame_height=raw_h,
        max_width=max_window_width,
        max_height=available_height,
    )

    frame_display = _frame_to_display_u8(raw_frame)
    if (display_w, display_h) != (raw_w, raw_h):
        frame_display = cv2.resize(
            frame_display,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )

    display = np.zeros((header_height + display_h, display_w), dtype=np.uint8)
    display[header_height:, :] = frame_display

    state.frame_width = raw_w
    state.frame_height = raw_h
    state.display_scale = scale
    state.display_header_height = header_height
    state.display_frame_width = display_w
    state.display_frame_height = display_h

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


def run_preview(pg, mode, output_dir, window_name="PG Preview"):
    state = PreviewState()
    roi = [int(v) for v in np.asarray(pg.big_roi, dtype=int).tolist()]
    global_row0 = roi[0]
    global_col0 = roi[2]
    max_window_width, max_window_height = _preview_window_limits()
    current_window_shape = None

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

            key = cv2.waitKey(1 if mode == "stream" else 30) & 0xFF
            if _window_is_closed(window_name):
                break
            if key == KEY_QUIT:
                break
            if key == KEY_FORMAT:
                new_format = _cycle_snapshot_format(state)
                print(f"Snapshot format set to: {new_format}")
                continue
            if key == KEY_SAVE:
                save_format = _snapshot_format(state)
                try:
                    saved_path = save_raw_snapshot(
                        raw_frame=raw_frame,
                        output_dir=output_dir,
                        mode_label=mode,
                        file_format=save_format,
                    )
                    print(f"Saved snapshot: {saved_path}")
                except Exception as exc:
                    print(f"Snapshot save failed: {exc}")
    finally:
        try:
            pg.stop()
        finally:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    # crop_height = 3104
    # crop_width = 3252
    crop_height = 500
    crop_width = 700
    row_offset = -100
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
        "PG_exp": 3e3,
        "PG_gain": 0,
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

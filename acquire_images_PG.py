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


@dataclass
class PreviewState:
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_valid: bool = False
    format_index: int = 0


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
    if event >= 0:
        state.mouse_x = int(x)
        state.mouse_y = int(y)
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


def _text_color_for_frame(frame):
    if np.issubdtype(frame.dtype, np.integer):
        return int(np.iinfo(frame.dtype).max)
    return 255


def _format_pixel_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def annotate_preview_frame(raw_frame, state, global_row0, global_col0, mode_label):
    annotated = raw_frame.copy()
    h, w = annotated.shape[:2]

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

    line_height = 24
    overlay_height = line_height * 2 + 8
    cv2.rectangle(annotated, (0, 0), (w - 1, overlay_height), 0, thickness=-1)

    color = _text_color_for_frame(annotated)
    cv2.putText(
        annotated,
        info_line1,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        info_line2,
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


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

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse, state)

    static_frame = None
    if mode == "single":
        static_frame = _ensure_single_image(pg.get_frame(split=False, full_screen=False))

    try:
        while True:
            if mode == "stream":
                raw_frame = _ensure_single_image(pg.get_frame(split=False, full_screen=False))
            else:
                raw_frame = static_frame

            preview = annotate_preview_frame(
                raw_frame=raw_frame,
                state=state,
                global_row0=global_row0,
                global_col0=global_col0,
                mode_label=mode,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1 if mode == "stream" else 30) & 0xFF
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

    crop_height = 3104
    crop_width = 3252
    row_offset = -150
    col_offset = 50
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
        "PG_gain": 20,
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

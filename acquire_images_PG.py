from PG_class import PG_controller
import cv2


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

    # Keep ROI inside frame.
    r1 = max(0, min(r1, sensor_height - crop_height))
    c1 = max(0, min(c1, sensor_width - crop_width))

    r2 = r1 + crop_height - 1
    c2 = c1 + crop_width - 1
    return [int(r1), int(r2), int(c1), int(c2)]


def stream_video(pg, window_name="PG Stream"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = pg.get_frame(split=False, full_screen=False)
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        try:
            pg.stop()
        finally:
            cv2.destroyAllWindows()


def main():
    # Plug in your crop values here.
    crop_height = 3104
    crop_width = 3252
    row_offset = -150  # positive -> down, negative -> up
    col_offset = 50  # positive -> right, negative -> left
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

    pg = PG_controller({
        "PG_num_mean": 1,
        "PG_exp": 10e3,
        "PG_gain": 20,
        "PG_sensor_width": sensor_width,
        "PG_sensor_height": sensor_height,
        "PG_single_roi": True,
        "PG_roi": roi,
    })
    stream_video(pg, window_name=f"PG Stream ROI {roi}")

if __name__ == "__main__":
    main()

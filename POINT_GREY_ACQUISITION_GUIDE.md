# Point Grey Acquisition Guide

This guide covers the Point Grey / FLIR (Spinnaker) acquisition path implemented by [acquire_images_PG.py](acquire_images_PG.py) and [PG_class.py](PG_class.py).

After this first mention, the rest of the guide uses "Point Grey".

For the main rectification, sweep, validation, and gain-calibration workflows,
see [README.md](README.md) and [moment_images/README.md](moment_images/README.md).
The Point Grey path is optional unless you need live camera preview or raw
snapshot capture.

## Overview

### `acquire_images_PG.py`

The preview script provides:

- live or single-frame preview from the configured Point Grey camera
- raw pixel readout under the mouse cursor
- software zoom with a click-to-lock zoom center
- snapshot capture in `tiff`, `npy`, or `png`
- stream-mode exposure and gain adjustment from the keyboard

### `PG_class.py`

The controller wrapper provides:

- camera initialization through `simple_pyspin.Camera`
- software-triggered frame capture
- ROI handling, including single-ROI mode used by the preview script
- exposure and gain getters/setters with camera-node limit validation and clamping
- frame collection helpers such as `get_frame()` and `get_frames()`
- additional acquisition helpers such as `get_unsaturated_photo()` and `get_HDR_photo()` for code paths outside the preview script

## Prerequisites

You need all of the following before the preview script is expected to work:

- a supported Point Grey / FLIR camera reachable through a working Spinnaker environment
- Python bindings for `pyspin` and `simple_pyspin`
- the packages needed by the script itself, especially `numpy` and `opencv-python`
- a desktop session where OpenCV can open preview windows

Notes:

- The preview path is environment-dependent. Matching Python packages alone are not enough if the local Spinnaker runtime or camera drivers are missing.
- The preview is GUI-based. Headless environments are not a supported operator path.

## CLI Interface

Run the script from the repository root:

```bash
python acquire_images_PG.py --mode stream --output-dir path/to/snaps
```

Supported arguments:

| Argument | Meaning | Default |
| --- | --- | --- |
| `--mode` | Preview mode: `stream` for continuous capture, `single` for one captured frame reused in the loop | `stream` |
| `--output-dir` | Snapshot directory. If omitted, the script prompts at startup and verifies the path is writable | prompt |
| `--pixel-format` | Requested camera pixel format passed into `PG_controller` | `Mono8` |
| `--adc-bit-depth` | Requested camera ADC bit depth passed into `PG_controller` | `Bit12` |

The script validates the output directory before preview begins.

## Current Runtime Defaults

`main()` currently hard-codes the preview ROI and camera startup settings:

- sensor size: `5320 x 4600`
- crop size: `2000 x 2000`
- ROI helper: `centered_roi(sensor_height, sensor_width, crop_height, crop_width, row_offset, col_offset)`
- offsets: `row_offset=250`, `col_offset=0`
- preview script mode: single-ROI capture (`PG_single_roi=True`)
- initial exposure: `10_000 us`
- initial gain: `15 dB`
- averaging count: `PG_num_mean=1`

Those values are code-defined today. They are not exposed as CLI options.

## Preview Behavior

### `stream` vs `single`

- `stream` captures a fresh frame on each loop iteration and supports live exposure/gain updates.
- `single` captures one frame before the loop starts and keeps redisplaying that frame. Exposure and gain controls are intentionally disabled in this mode.

### ROI and display behavior

- The script previews the current `pg.big_roi` region, not the full sensor frame.
- The ROI is generated in `main()` with `centered_roi(...)` and then passed into `PG_controller` via `PG_roi`.
- The preview keeps track of both local ROI coordinates and global sensor coordinates.

### Pixel readout

- The on-screen status line reports local `(x, y)`, global `(x, y)`, and the raw pixel value under the cursor.
- The pixel readout comes from the raw captured frame, not from the 8-bit display buffer.

### Dynamic display scaling

- The visible ROI is resized to fit the available preview window.
- The script uses OS-reported screen dimensions when available and otherwise falls back to `1600 x 900`.
- Upscaling is only enabled when the user is zoomed in above `1.0x`.

### Zoom behavior

- Zoom is implemented as a crop on the raw frame followed by display resizing.
- Clicking the preview locks the zoom center to the current pixel coordinate.
- Resetting zoom returns to the full configured ROI view.

## Controls

| Input | Action | Notes |
| --- | --- | --- |
| `q` | Quit preview | Stops the loop and closes the window |
| `s` | Save snapshot | Saves the current raw frame to the selected output directory |
| `f` | Cycle snapshot format | Rotates through `tiff -> npy -> png -> ...` |
| left click | Lock zoom center | Uses the pixel under the cursor as the zoom center |
| `i` | Zoom in | Multiplies zoom by `1.25` up to the configured max |
| `o` | Zoom out | Multiplies zoom by `0.8` down to the configured min |
| `0` | Reset zoom | Returns to `1.0x` and clears the stored zoom center |
| `[` | Decrease exposure | `stream` mode only |
| `]` | Increase exposure | `stream` mode only |
| `,` | Decrease gain | `stream` mode only |
| `.` | Increase gain | `stream` mode only |

If exposure or gain keys are pressed in `single` mode, the script keeps running and surfaces a status message instead of changing the camera state.

## Snapshot Behavior

Snapshots can be saved in these formats:

- `tiff`
- `npy`
- `png`

File naming follows this pattern:

```text
pg_<mode>_<timestamp>.<ext>
```

Examples:

- `pg_stream_20260506_153012_123456.tiff`
- `pg_single_20260506_153045_654321.npy`

Important behavior:

- Snapshots save the raw frame, not the zoomed display crop.
- `npy` stores the NumPy array directly and is the most exact round-trip format.
- Image formats are written through OpenCV using the raw frame array.

## Controller Behavior Worth Preserving

The current Point Grey flow relies on these controller behaviors:

- exposure and gain setters clamp requested values to the camera node's reported min/max limits
- unsupported `PG_adc_bit_depth` or `PG_pixel_format` values fail fast
- the preview script depends on single-ROI behavior and expects `get_frame(split=False, full_screen=False)` to return one 2D image
- camera acquisition is software-triggered, not free-running

For contributors, those behaviors are part of the public contract even though they are implemented in plain Python rather than behind a formal interface.

## Current Limitations

- ROI values are still defined in code inside `main()` and are not user-configurable from the CLI.
- The preview script is single-ROI oriented even though `PG_class.py` has broader ROI support.
- Camera setup is environment-dependent and may require local vendor/runtime work outside the repository.
- `single` mode is intentionally static: it is useful for inspection, not for live camera tuning.
- `PG_load_ROI()` still carries hard-coded fallback ROI values in `PG_class.py`, but the preview script overrides them with its own explicit `PG_roi`.

## Troubleshooting

### Missing `pyspin` or `simple_pyspin`

Typical symptoms:

- import failure before preview starts
- camera initialization fails immediately

Check:

- the active Python environment
- that the bindings match the installed Spinnaker stack
- that the machine has the vendor runtime expected by the camera

### Unsupported pixel format or ADC bit depth

Typical symptom:

- `PG_controller` raises an error during startup when applying `PG_pixel_format` or `PG_adc_bit_depth`

Check:

- the exact strings passed to `--pixel-format` and `--adc-bit-depth`
- whether the connected camera exposes those node values in the current mode

### Unwritable output directory

Typical symptom:

- startup prompt rejects the path
- snapshot save reports a write error

Check:

- the directory exists or can be created
- the current user can write to it
- the target is a directory and not a file path

### Exposure or gain keys do nothing in `single` mode

This is expected. The preview loop shows a status message and keeps running, but live exposure/gain control is reserved for `stream` mode.

### Preview window does not appear

Check:

- that OpenCV can create GUI windows in the current session
- that the environment is not headless
- that the camera is reachable and returning frames

## Testing

Use the hardware-free preview/control test file to validate the script-side contract without a camera:

```bash
pytest test_acquire_images_PG.py
```

That test file stubs `PG_class` and checks preview math, zoom behavior, snapshot handling, and stream-only exposure control semantics.

# Validate Rectification GUI Guide

This guide explains how to use `moment_images/validate_rectification.py` end-to-end.

## What This Tool Does

`validate_rectification.py` lets you compare a rectified reference image against two external camera images:

- Validation camera image
- Thorcam image

For each camera image, you can:

- Apply geometry edits (rotate, flip, crop)
- Adjust intensity window
- Apply anisotropic scaling (`X`, `Y`)
- Choose comparison mode (`preserve_scale` or `fit_reference`)
- Run translation-only ECC alignment
- Inspect four preview tabs before committing

At the end, the tool generates comparison figures and can save a `validation_workspace.npz` with full metadata for reproducibility.

## Prerequisites

- Python environment with project dependencies installed.
- A rectification output file:
  - `.npz` workspace/sweep output containing one of:
    - `average_rectified_image`
    - `rectified_image`
    - `rectified_views`
  - or `.npy` rectified views stack
- Two camera images (Validation + Thorcam), typically grayscale-compatible image formats (`png`, `jpg`, `tif`, etc.).

## Launch

From the project root:

```bash
python -m moment_images.validate_rectification
```

You will be prompted to select files in this order:

1. Rectification output (`.npz` or `.npy`)
2. Validation camera image
3. Thorcam image

Then two editor windows run sequentially:

1. Validation editor
2. Thorcam editor

## Window Layout

The editor is divided into:

1. Control rows at the top
2. Status bar under controls
3. Notebook tabs for visual previews

The top controls are split across multiple rows so `Done`/`Cancel` stay visible on full-screen and narrower displays.

## Control Reference

### Geometry

- `Rot L`: rotate +90 degrees
- `Rot R`: rotate -90 degrees
- `Flip H`: horizontal mirror
- `Flip V`: vertical mirror
- `Crop ON/OFF`: enables rectangle selection on the **As-Is** tab

Crop details:

- Turn crop on, then drag a rectangle in the Input view of the **As-Is** tab.
- Crop coordinates are mapped back to the pre-crop geometry image.
- Any geometry/intensity/scale change marks alignment as stale.

### Scale

- `X`: horizontal scale factor
- `Y`: vertical scale factor
- `Pre-blur before resize`: apply a normalized box blur before scaling (default: on)
- Valid range is shown in the GUI: `[0.001, 8.000]`

Scale behavior:

- Numeric validation happens as you type.
- Invalid values are rejected on commit and reverted to last valid values.
- When pre-blur is enabled, a uniform energy-conserving kernel is applied before resize:
  - `kx = ceil(1 / X)`
  - `ky = ceil(1 / Y)`
  - (with minimum kernel size `1` per axis)
- After resize, finite output values are linearly remapped to the finite min/max
  of the pre-resize image, so X/Y scaling preserves the input dynamic range.
- Scale changes invalidate previous alignment results.

### Alignment

- `Mode`: choose one:
  - `preserve_scale`
  - `fit_reference`
- `Auto Align`:
  - On: preview uses stored alignment result only when it is up-to-date
  - Off: no ECC alignment is applied
- `Run Alignment`: executes translation-ECC with current edits

Mode behavior:

- `preserve_scale`: keeps edited image dimensions; matching/ROI logic handles containment.
- `fit_reference`: resizes input to reference size before alignment/comparison.

### Intensity

- `Min` slider: lower clip bound
- `Max` slider: upper clip bound

Windowing behavior:

- Input values are clipped to `[Min, Max]` in native image units (no fixed-range remap).
- If `Min >= Max`, the GUI nudges `Max` upward to maintain a valid window.

### Outputs

- Dropdown: choose reset scope:
  - `geometry`
  - `intensity`
  - `scale`
  - `alignment`
- `Reset Section`: resets only selected scope
- `Reset All`: resets everything (including mode and auto-align defaults)

### Final Actions

- `Done`: commits this image and closes the editor
- `Cancel`: aborts this image editor

## Preview Tabs

### As-Is

- Left: reference image (joint-normalized with current edited image before X/Y display scaling)
- Right: edited input view used for crop interaction
- Shows crop rectangle overlay when crop is set

### Registered ROI

- Shows reference/input after registration rules are applied
- Includes mode/score context in title when available

### Difference

- `|reference - input|` on joint-normalized registered pair
- If not available, tab shows status text explaining why

### Overlay

- RGB composite:
  - Red channel: reference
  - Green channel: input
- If not available, tab shows status text explaining why

## Status Bar Meanings

The status bar summarizes current state, including:

- active mode
- input size after edits
- whether crop is set
- alignment state (`stale`, `disabled`, or alignment status code)
- template/ECC scores when available
- comparison status

If you changed geometry/intensity/scale and `Auto Align` is on, you will usually see:

- `Alignment=stale (click Run Alignment)`

Run `Run Alignment` to refresh alignment with current edits.

## Recommended Editing Workflow (Per Image)

1. Set geometry first (`Rot/Flip/Crop`).
2. Set intensity (`Min/Max`) to roughly match contrast/dynamic range.
3. Set scale factors (`X`, `Y`) if needed.
4. Choose `Mode`:
   - start with `preserve_scale` if you care about manual scale factors
   - use `fit_reference` if you want direct same-size comparison
5. Click `Run Alignment`.
6. Review `Registered ROI`, `Difference`, and `Overlay`.
7. Click `Done`.

Repeat for the second camera image.

## After Both Editors

The script generates four Matplotlib figures per camera image:

1. As-is pair
2. Registered pair
3. Difference map
4. Overlay

You can then choose to save:

- Figure images (via project save helper)
- `validation_workspace.npz`

The workspace includes:

- Processed images for both cameras
- Mode/auto-align settings
- Scale factors, intensity window, crop, rotation, flips
- Alignment status/mode/scores/ROIs
- Backward-compatible keys for existing workflows

## Troubleshooting

### `Done`/`Cancel` or other buttons are hard to find

- Controls are now split into multiple rows; `Done` and `Cancel` are in the bottom control row.
- Maximize/full-screen the window and ensure OS display scaling is not excessively large.

### `Alignment=stale`

- You edited geometry/intensity/scale after last alignment.
- Click `Run Alignment` again.

### Difference/Overlay says unavailable

Common causes:

- No containment match between reference and input
- Low template confidence
- No valid same-size registered ROI

Try:

- better crop around shared content
- adjusted scale factors
- switching mode (`preserve_scale` vs `fit_reference`)
- rerunning alignment

### Alignment status starts with `ecc_failed:...`

- ECC failed on current inputs (often low texture, invalid data, or unstable intensity mapping).
- Try improving intensity window, crop to informative content, and rerun alignment.

## Notes on Camera Defaults

No camera-specific defaults are applied automatically before editing.

Validation and Thorcam now enter the editor through the same preprocessing path
(same load/grayscale/intensity/scale/alignment pipeline). Use Geometry controls
(rotate/flip/crop) to set orientation manually when needed.

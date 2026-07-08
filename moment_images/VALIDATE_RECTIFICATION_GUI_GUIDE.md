# Validate Rectification GUI Guide

This guide explains how to use `moment_images/validate_rectification.py` end-to-end.

For the broader package map, see [README.md](README.md). For sweep outputs that
feed validation, see [PARAMETER_SWEEP_GUIDE.md](PARAMETER_SWEEP_GUIDE.md).

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
- Inspect six preview tabs before committing

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
- `Pre-blur before resize`: apply a Gaussian anti-aliasing blur before scaling (default: on)
- Valid range is shown in the GUI: `[0.001, 8.000]`

Scale behavior:

- Numeric validation happens as you type.
- Invalid values are rejected on commit and reverted to last valid values.
- When pre-blur is enabled and an axis is downsampled (`scale < 1`), a Gaussian is applied before resize:
  - `sigma_x = (1 / X) / 2` for downsampled X (otherwise no X blur)
  - `sigma_y = (1 / Y) / 2` for downsampled Y (otherwise no Y blur)
  - The separable kernel is gain-normalized so the effective 2D kernel sum is
    `(1 / dx) * (1 / dy)`, where `dx = min(X, 1)` and `dy = min(Y, 1)`.
- After resize, values stay in native units from the blur/resize pipeline
  (no post-resize min/max remap).
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
- `fit_reference`: keeps the edited input geometry and resizes the reference to input size for alignment/comparison outputs.

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

- Left: reference image with its own display bounds
- Right: edited input view used for crop interaction
- Input uses independent display bounds (estimated from the pre-scale crop-preview source)
- Shows crop rectangle overlay when crop is set

### Registered ROI

- Shows reference/input after registration rules are applied
- Both panels use the same shared native-unit bounds for direct visual comparison
- Includes mode/score context in title when available

### Diff (Native)

- `|reference - input|` computed on the registered pair in native intensity units
- Colorbar starts at 0 and scales to observed max absolute difference
- If not available, tab shows status text explaining why

### Diff (Norm)

- `|reference - input|` on joint-normalized registered pair
- Normalized display range is fixed to `[0, 1]`
- If not available, tab shows status text explaining why

### Overlay (Native)

- RGB composite built from registered images in native units with one shared bound pair:
  - Red channel: reference
  - Green channel: input
- Both channel colorbars use the same native-unit range
- If not available, tab shows status text explaining why

### Overlay (Norm)

- RGB composite:
  - Red channel: reference
  - Green channel: input
- Joint-normalized channel colorbars are fixed to `[0, 1]`
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
   - use `fit_reference` if you want direct same-size comparison in the edited input geometry
5. Click `Run Alignment`.
6. Review `Registered ROI`, `Diff (Native)`, `Diff (Norm)`, `Overlay (Native)`, and `Overlay (Norm)`.
7. Click `Done`.

Repeat for the second camera image.

## After Both Editors

The script generates six Matplotlib figures per camera image:

1. As-is pair
2. Registered pair
3. Difference map (native units)
4. Difference map (joint-normalized)
5. Overlay (native units)
6. Overlay (joint-normalized)

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

## Batch Validation Image Preprocessing

`moment_images/batch_process_validation_images.py` provides a GUI-only helper
for applying shared preprocessing settings to a folder of validation images:

```bash
python -m moment_images.batch_process_validation_images
```

The batch helper supports rotation, flips, optional crop coordinates, optional
min/max intensity clipping, X/Y scaling, pre-blur before resizing, and optional
flux-sum preservation.

Current limitation: this helper does not expose a normal argparse command line
or `--help` reference. Run it from a desktop session and configure it through
the GUI.

## Headless Trio Benchmark

`moment_images/trio_benchmark.py` ranks completed LF sweep candidates against a
processed Validation Camera image and scores a ThorCam image as a baseline
comparator. The sweep algorithm stays blind to those reference images; this
tool only post-processes completed sweep artifacts.

```bash
python -m moment_images.trio_benchmark --sweep-dir path/to/sweep_output --dataset-id scene_id --validation-image path/to/validation_processed.tif --thorcam-image path/to/thorcam_processed.tif --output-dir path/to/trio_report
```

The same command can read the processed references directly from a saved
`validate_rectification.py` workspace:

```bash
python -m moment_images.trio_benchmark --sweep-dir path/to/sweep_output --dataset-id scene_id --validation-workspace path/to/validation_workspace.npz --output-dir path/to/trio_report
```

The benchmark writes `trio_benchmark_results.csv`,
`trio_benchmark_summary.json`, `trio_benchmark_report.html`, and a `figures/`
folder. Results use the stable `validation_hybrid_v1` metric profile and
preserve sweep row identity (`row_id`, `stage`, `candidate_id`, `dataset_id`,
and `generation_id`) so later utilities can join the records back to sweep
artifacts.

The default registration profile is `bead_similarity_v1`. It keeps the primary
registration shear-free, auto-selects the LF anchor by bead/structure
registration confidence, and writes `registration_confidence`,
`registration_suspect`, `registration_suspect_reasons`, matched bead counts,
median bead error, common-FOV fraction, and residual translation norm into the
CSV/JSON/HTML outputs. Use `--registration-profile centered_ecc_legacy` when
you need to compare against the older centered ECC initializer.

For repeated evaluation across multiple scenes, use a manifest and the batch
runner:

```bash
python -m moment_images.trio_benchmark_batch --manifest path/to/trio_manifest.json --output-dir path/to/trio_batch_report
```

`moment_images/trio_benchmark_manifest_example.json` shows the expected schema.
Each scene writes its normal trio report under a scene-specific subfolder, while
the batch output folder also receives `trio_batch_results.csv`,
`trio_batch_summary.json`, and `trio_batch_report.html`.

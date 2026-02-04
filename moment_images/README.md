# Moment Images Utilities

This package focuses on rectifying light-field view grids extracted from grayscale images. It includes:

- **Interactive workflows** in `im_process.py` for selecting viewpoints, rectifying a grid, and inspecting figures.
- **Automated rectification** via the `rectify_views_from_config` helper so external tools can run the same pipeline.
- **Parameter sweeping** with `parameter_sweep.py` to explore trade-offs across window sizes, displacement bounds, and Gaussian filters.
- **Sweep analysis** utilities in `sweep_analysis.py` for summarizing metrics across many rectification runs.

> All file paths and commands in this README are relative to the `moment_images/` package directory.

---

## Repository map

- `im_process.py` — Interactive rectification workflow and reusable rectification helpers.
- `rectification_strategy.py` — Core algorithms for global translation and pixelwise patch rectification.
- `parameter_sweep.py` — CLI/GUI runner that drives rectification over a parameter grid.
- `sweep_analysis.py` — Metric computation and aggregation for sweep outputs.
- `unpack_workspace.py` — GUI helper for unpacking `.npz` workspace archives saved by `im_process.py`.

---

## Quick start

### Interactive rectification

Run the interactive workflow to select viewpoints manually or via circle detection, then step through rectification and figures:

```bash
python -m moment_images.im_process
```

### Parameter sweep (CLI)

Use a precomputed view grid (or an HDR stack) to run a sweep non-interactively:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy
```

### Parameter sweep (GUI)

Launch the GUI to configure the sweep without CLI flags:

```bash
python -m moment_images.parameter_sweep --gui
```

---

## Interactive pipeline (`im_process.py`)

The interactive workflow is organized around a set of core stages. Understanding these stages is helpful when you later use `rectify_views_from_config` or the sweep runner.

### 1) Load and normalize images

- **Image selection** happens in `get_image_paths()`. Users can pick a directory, mark it as HDR, and optionally select a single image by number or name.
- **Normalization** uses `min_max_normalize()` to scale each grayscale image to 0–255 for consistent downstream processing.
- If the directory is HDR-capable, you can either assemble HDR immediately with `assemble_HDR()` or delay HDR assembly until after global rectification.

### 2) Viewpoint selection

`im_process.py` supports two ways to locate the grid of viewpoints:

1. **Manual selection**
   - You visualize the normalized image and enter the center coordinates (`x`, `y`) plus radius for the grid origin.
   - The selected circle is previewed with `display_circles()` before continuing.

2. **Circle detection**
   - `select_roi()` picks a region of interest by center, brightest pixel, explicit coordinates, or full-image mode.
   - `detect_and_adjust_circles()` runs Hough circle detection in the ROI and re-maps the results back into full-image coordinates.
   - You can review detected circles and approve which image(s) to process.

### 3) View grid extraction

Once the base circle is chosen, `extract_views()` slices a square grid of patches:

- The grid size is user-selected (e.g., 3×3).
- Each patch is centered at a fixed distance from its neighbors (`view_dist`).
- Any patch that would fall outside the image bounds is skipped and left as zeros.

This produces a `(N, N, H, W)` stack of grayscale view patches that feeds directly into the rectification strategies.

### 4) Rectification strategies

The interactive flow ultimately routes to one of the strategies defined in `rectification_strategy.py`.

#### Global translation strategy

`GlobalTranslationRectificationStrategy`:

- Builds a **global similarity matrix** by correlating each view against the reference view (`compute_similarity_matrix`).
- Uses the argmax of the cross-correlation to extract integer `(dy, dx)` translations for each view.
- Applies the translations using `cv.warpAffine`, then averages all rectified views into a single output image.

This path is efficient and robust when a single translation per view is sufficient.

#### Pixelwise patch strategy

`PixelwisePatchRectificationStrategy`:

- For each non-reference view, it rolls the image over a displacement grid and computes Gaussian-filtered correlation scores against the reference.
- It stores **per-pixel** displacement fields (`disps_y`, `disps_x`) and correlation scores in a `PixelwiseSimilarityResult`.
- The final alignment uses `cv.remap` with the full per-pixel coordinate map, allowing local corrections beyond a single translation.

Key tunable behaviors in the pixelwise strategy:

- **Window size**: controls the neighborhood used for correlation.
- **Max displacement**: bounds the search region.
- **Gaussian smoothing**: applied to the correlation product to stabilize the correlation peak.
- **Displacement smoothing**: optional Gaussian filtering on the displacement fields to reduce noise.
- **Sub-pixel factor**: increases resolution by resizing views and searching at finer increments.

#### Combined global + pixel refinement

The **combined** strategy runs global translation first, then refines with pixelwise patch matching on the globally rectified stack. When enabled, the global average can be used as the pixelwise reference (`combined_use_global_avg_reference`).

#### Sigma pyramid refinement (pixelwise)

Pixelwise rectification can enable `use_sigma_pyramid` (or `pixel_sigma_pyramid=True` in `RectificationRunConfig`). This:

- Starts with a larger Gaussian sigma.
- Iteratively halves sigma, recomputing displacement maps at each scale.
- Constrains per-pixel updates to stay within 2 pixels of the previous displacement estimate to avoid false matches.

### 5) Output figures and workspace snapshots

`im_process.py` can generate and save a rich set of figures (original views, similarity maps, rectified views, overlays, averages). Use `save_figures()` to persist these with sanitized titles as filenames. The workflow can also save a compressed `workspace.npz` of intermediate arrays so you can revisit or analyze later.

To unpack saved workspace archives:

```bash
python -m moment_images.unpack_workspace
```

---

## Automated rectification (`rectify_views_from_config`)

For scripted or swept runs, `rectify_views_from_config()` provides a single entry point that mirrors the interactive flow:

1. **Input validation**
   - Accepts `(N, N, H, W)` view grids.
   - Accepts `(E, N, N, H, W)` stacks for HDR when `hdr_enabled=True`.

2. **Reference view selection**
   - `resolve_reference_view()` defaults to the center view unless a specific index is passed.

3. **Strategy execution**
   - Runs global, pixelwise, or combined rectification based on `RectificationRunConfig`.

4. **HDR assembly (optional)**
   - When `hdr_enabled=True`, rectified views are combined across exposures by `assemble_HDR()`.

5. **Return object**
   - The result is a `RectificationResult` containing the averaged image, rectified view stack, similarity data, and optional global/pixelwise diagnostics for downstream visualization.

---

## Parameter sweeping (`parameter_sweep.py`)

The sweep runner lets you explore parameter combinations offline and save results for comparison.

### Inputs and view generation

The sweep expects either:

- `--views` pointing to a precomputed NumPy array shaped `(N, N, H, W)` or `(E, N, N, H, W)`.
- Or a **source image** + center coordinates + radius so it can generate the view grid once and cache it as `generated_views.npy`.
- Or a **source image directory** (with `--hdr-assemble`) to build an exposure stack and cache it as `generated_exposure_views.npy`.

### Sweep dimensions

The grid is built from:

- Window sizes (`--window-start/stop/step`)
- Max displacements (`--max-disp-start/stop/step`)
- Gaussian smoothing (`--gauss-sigma-*`)
- Displacement smoothing (`--disp-gauss-*`)
- Thresholds:
  - **Global thresholds** (`--global-thresh-*`) apply to global correlation masking.
  - **Pixel thresholds** (`--pixel-thresh-*`) apply to pixelwise reference thresholding.

### Live plotting and figures

- `--live-plot` shows a real-time matplotlib view of tracked metrics after each run.
- `--save-figures` saves the same figures as the interactive workflow under each run folder.

### Outputs

Each run creates a folder with:

- `avg_rectified.png` — averaged rectified output.
- `rectified_views.npy` — full rectified stack.
- `config.json` — the configuration used for this run.

The sweep writes a top-level `summary.json` with metrics for each run. When `--analyze` is enabled it also writes:

- `analysis.json` — min/max/mean summaries by metric.
- `analysis.csv` — tabular view of the same summaries.

---

## Sweep analysis (`sweep_analysis.py`)

`build_metric_record()` and `compute_image_metrics()` measure quality for each run using both **global-only** and **global+local** averages (when applicable).

Metrics include:

- **Average intensity**: mean pixel value.
- **Maximum intensity**: max pixel value.
- **Total variation**: sum of absolute differences in X and Y.
- **Sum of squared gradients**: sum of gradient magnitudes squared.
- **Laplacian variance**: variance of the Laplacian response.
- **Laplacian energy**: sum of squared Laplacian response.

`analyze_summary_data()` aggregates these across all runs and records minima, maxima, and means, which are then saved by `save_analysis()`.

---

## Tips and troubleshooting

- **Image bounds**: `extract_views()` will zero-fill patches that fall outside the image. If you see blank views, verify the center point and radius.
- **Pixelwise performance**: pixelwise rectification is computationally heavy because it evaluates many displacements per pixel. Start with smaller grids and tighter `max_disp` settings.
- **Sub-pixel searches**: increasing `sub_pix_factor` improves alignment but increases compute and memory usage.
- **HDR workflows**: ensure `hdr_exposure_times` matches the number of exposure images when HDR assembly is enabled.

---

## References

For deeper implementation details, review the docstrings in `im_process.py` and `rectification_strategy.py`, which mirror the flow described above.

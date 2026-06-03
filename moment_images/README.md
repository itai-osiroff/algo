# Moment Images Utilities

This package focuses on rectifying light-field view grids extracted from grayscale images. It includes:

- **Interactive workflows** in `im_process.py` for selecting viewpoints, rectifying a grid, and inspecting figures.
- **Automated rectification** via the `rectify_views_from_config` helper so external tools can run the same pipeline.
- **Reliability-first parameter sweeping** with `parameter_sweep.py` to compare displacement-selection metric families and alpha weighting under one execution protocol.
- **Sweep analysis/reporting** utilities in `sweep_analysis.py` for metric summaries and alpha-centric comparison reports across many runs.

> All file paths and commands in this README are relative to the `moment_images/` package directory.
>
> For the separate Point Grey acquisition workflow at the repository root, see `../POINT_GREY_ACQUISITION_GUIDE.md`.

---

## Repository map

- `im_process.py` - Interactive rectification workflow and reusable rectification helpers.
- `rectification_strategy.py` - Core algorithms for global translation and pixelwise patch rectification.
- `parameter_sweep.py` - CLI/GUI runner for two-stage intensity-first sweeps, optional metric-formulation experiments, resumable execution, and ranking.
- `sweep_analysis.py` - Metric computation, aggregate analysis, and alpha/family report generation.
- `unpack_workspace.py` - GUI helper for unpacking `.npz` workspace archives saved by `im_process.py`.
- `validate_rectification.py` - Tool to validate rectification results against new images from other cameras.

---

## Quick start

### Interactive rectification

Run the interactive workflow to select viewpoints manually or via circle detection, then step through rectification and figures:

```bash
python -m moment_images.im_process
```

### Parameter sweep (CLI)

Use a JSON experiment spec (recommended) for two-stage coarse-to-fine, multi-dataset exploration:

```bash
python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json
```

Legacy single-dataset CLI flags are still supported:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy
```

### Parameter sweep (GUI)

Launch the spec-based GUI to configure and run the new two-stage sweep without manually editing JSON:

```bash
python -m moment_images.parameter_sweep --gui
```

GUI highlights:

- Add/remove multiple datasets (`views`, `source_image`, or `source_image_dir`).
- Edit basic search ranges plus advanced scoring/search toggles.
- Save/load spec JSON and run in background with live logs.

### System gain calibration (CLI + GUI)

Compute camera system gain using the Teledyne mean-variance workflow:

- bias = dark reference (single pre-averaged image or averaged directory stack)
- flat1 + flat2 = even-illumination captures (single images or averaged stacks)
- gain = `mean(flat1 - bias) / variance((flat2 - flat1) ROI)` where variance is derived from the ROI standard deviation as `std^2 / 2`

CLI example with single images:

```bash
python -m moment_images.main_thor_gain \
  --bias-image path/to/bias.tiff \
  --flat1-image path/to/flat1.tiff \
  --flat2-image path/to/flat2.tiff \
  --roi-size 100 \
  --output-json gain_result.json
```

CLI example with directory stacks:

```bash
python -m moment_images.main_thor_gain \
  --bias-dir path/to/bias_frames \
  --flat1-dir path/to/flat1_frames \
  --flat2-dir path/to/flat2_frames \
  --pattern "*.tif*" \
  --roi-size 100
```

Launch the GUI alternative:

```bash
python -m moment_images.main_thor_gain --gui
```

---

## Interactive pipeline (`im_process.py`)

The interactive workflow is organized around a set of core stages. Understanding these stages is helpful when you later use `rectify_views_from_config` or the sweep runner.

### 1) Load and normalize images

- **Image selection** happens in `get_image_paths()`. Users can pick a directory, mark it as HDR, and optionally select a single image by number or name.
- **Normalization** uses `min_max_normalize()` to scale each grayscale image to 0-255 for consistent downstream processing.
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

- The grid size is user-selected (e.g., 3x3).
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

### Automation entrypoints

`RectificationRunConfig` is the public configuration surface for scripted runs and sweep-generated `config.json` files. The local displacement-score fields most likely to matter when automating runs are:

- `displacement_metric_family`
- `displacement_alpha`
- `displacement_component_weights`
- `metric_formulation_id`

Saved configs use canonical metric family names such as `blurred_gradient_corr`, `blurred_laplacian_corr`, and `blurred_composite_corr` even when a legacy alias was accepted on input.

---

## Parameter sweeping (`parameter_sweep.py`)

The sweep runner supports two modes:

- **Spec mode (recommended):** two-stage coarse-to-fine, multi-dataset, strategy-aware exploration.
- **Legacy mode:** single-pass cartesian sweep via individual CLI flags.

Run spec mode with:

```bash
python -m moment_images.parameter_sweep --spec path/to/spec.json
```

A complete template is included in `sweep_experiment_spec_example.json`.

### Inputs and view generation

Each dataset in the spec can use either:

- `views` pointing to a precomputed NumPy array shaped `(N, N, H, W)` or `(E, N, N, H, W)`.
- Or a **source image** + center coordinates + radius so the runner can generate and cache `generated_views.npy`.
- Or a **source image directory** to build an exposure stack and cache `generated_exposure_views.npy`.

### Sweep dimensions (spec mode)

Search spaces are strategy-aware:

- `pixel`: window size, max displacement, Gaussian sigmas, pixel threshold, sub-pixel factor, sigma pyramid flag, zero-fill roll.
- `combined`: global pad/threshold plus pixel refinement controls, including global-average reference toggle.
- `global`: optional baseline mode using global-only knobs.

Baseline execution mode is configured under `stages.mode`:

- `two_stage` (default): coarse stage-1 ranking followed by stage-2 auto-fine refinement.
- `single_stage`: one-pass baseline evaluation over the configured candidate grid.

By default, stage 2 is planned from the top stage-1 candidates using `stages.refinement.method="auto_fine"`. Numeric parameters receive interpolated midpoint values between neighboring coarse settings, plus bounded half-step extrapolation when a top candidate is at a coarse-grid edge. Integer parameters are normalized to valid integer values, window sizes remain positive odd values, and thresholds are clamped to `[0, 1]`. Boolean/categorical parameters stay on configured values. To reproduce the older behavior that only reused configured neighboring values, set `stages.refinement.method="neighbor"`.

Baseline stage-1/stage-2 candidates run in **pure intensity mode** by default (`family=intensity`, `alpha=1.0`). Optional metric experiments are configured under `experimental_metric_sweep`. This second phase runs **after** baseline ranking and only expands metric formulations for shortlisted pixel/combined candidates.

Each formulation defines:

- `id`: stable formulation identifier for reporting and traceability.
- `family`: `blurred_gradient_corr`, `blurred_laplacian_corr`, or `blurred_composite_corr`.
- `alphas`: one or more alpha values to evaluate.
- `component_weights` (`blurred_composite_corr` only): non-intensity component weights (validated and normalized internally).

Legacy aliases `ncc_gradient`, `ncc_laplacian`, and `composite` are still accepted on input, but saved configs and downstream reports use the canonical names above.

Baseline candidate generation is the cross-product of:

- strategy search-space parameters

Optional metric-experiment candidate generation is the cross-product of:

- shortlisted baseline candidates
- formulation family/weights
- alpha values

This keeps the experiment narrative explicit: **global pre-alignment -> intensity-only baseline ranking -> optional metric-formulation experiments on shortlisted candidates -> reconstruction -> diagnostics/quality metrics -> side-by-side alpha/family comparison**.

### Shared metric profiles

Profiles saved by the raw metric explorer can also rank future parameter sweeps. A spec may either provide explicit `scoring.metric_weights` or reference a shared profile:

```json
"scoring": {
  "profile_name": "guide_focus_mix",
  "profile_config_path": "metric_review/metrics_config.json",
  "metric_scope": "global_and_local_metrics",
  "runtime_penalty_weight": 0.0,
  "robustness_lambda": 0.5,
  "missing_dataset_penalty": 1.0
}
```

The CLI can override these with `--metrics-config path/to/metrics_config.json --scoring-profile guide_focus_mix`. `saved_score` and `saved_mean_dataset_score` remain explorer-only baselines and are rejected for future sweep ranking.

### Alpha semantics (local displacement score blending)

The non-intensity displacement families are **blurred-correlation** score formulations. They use Gaussian-smoothed product scores and do **not** implement true local NCC/ZNCC normalization. Legacy aliases `ncc_gradient`, `ncc_laplacian`, and `composite` are still accepted on input, but new configs and outputs use the canonical names below.

For pair families (`blurred_gradient_corr`, `blurred_laplacian_corr`):

- `alpha` is the local displacement score blend weight:
  `score = alpha * I + (1 - alpha) * S`

For `blurred_composite_corr`:

- `W_intensity = alpha`
- non-intensity terms are normalized to sum to 1
- each non-intensity term gets `W_k = (1 - alpha) * normalized_k`

`alpha=1` is intensity-only local displacement ranking. `alpha=0` is non-intensity-only local displacement ranking.

### Live plotting and figures

- `--live-plot` shows a real-time matplotlib view of tracked metrics after each run.
- `--save-figures` saves the same figures as the interactive workflow under each run folder.

### Outputs (spec mode)

Top-level outputs include:

- `stage1_summary.json` - coarse-stage run details, failures, and rankings.
- `stage2_summary.json` - refinement-stage run details, failures, and rankings (`two_stage` only).
- `leaderboard.json` and `leaderboard.csv` - final ranked configurations.
- `best_configs.json` - overall best and best-per-strategy configurations.
- `summary.json` - flattened metric records for successful runs.
- `analysis.json` and `analysis.csv` - aggregate metric summaries (when enabled).
- `run_state.json` - durable resumable execution state (attempt history, statuses, artifact validation).
- `experiment_manifest.json` - deterministic manifest of datasets, baseline displacement mode, optional metric experiments, scoring, reporting, and run-state location.
- `generations/<generation_id>/products/` - a historical snapshot of the summaries, leaderboards, analysis, and manifest generated by that run, extension, or merge.
- `generations/<generation_id>/merge_log.json` - source/copy/duplicate/rejection provenance when `--merge-from` is used.
- `generations/<generation_id>/dry_run.json` - planned reuse/copy/reject/pending task report when `--dry-run` is used.

When `experimental_metric_sweep.enabled=true`, additional artifacts are written under `metric_experiments/`:

- `metric_phase_summary.json`
- `leaderboard.json` and `leaderboard.csv`
- `summary.json`
- `alpha_report.json`, `alpha_report.csv`, and `alpha_report_warnings.json`

This optional metric phase can run after either baseline mode: the default `two_stage` flow or `single_stage`.

Each run creates a folder with:

- `avg_rectified.png` - averaged rectified output.
- `rectified_views.npy` - full rectified stack (legacy-compatible output).
- `workspace.npz` - workspace-style compressed archive containing `rectified_views`, `average_rectified_image`, and stage diagnostics.
- `workspace.mat` - MATLAB companion export of the same workspace variables.
- `config.json` - the configuration used for this run.
- `metric_record.json` - per-run metric payload used for resume skipping and partial-report reconstruction.

### Resume and compatibility checks

- Every spec run writes `run_state.json` atomically.
- Restarting in the same output directory resumes when the stored experiment fingerprint matches the current experiment.
- If a run directory was copied or relocated, resume can adopt it when the semantic fingerprint payload is still compatible after normalizing machine-specific paths and volatile cache metadata.
- True spec or dataset mismatches are still hard failures (no partial overwrite of incompatible results).
- Completed runs are skipped only when required artifacts still validate; missing/corrupt artifacts trigger recompute for that run only.
- Failure history is append-only: retries add attempts without deleting prior error records.
- Candidate directories in new spec-mode runs use deterministic hash IDs (`s1c...`, `s2c...`, `m1c...`) derived from the normalized rectification config, so equivalent runs can be reused across extensions and merges.
- To extend an existing sweep after expanding a parameter range, run the expanded spec with `--output-dir` pointing at the canonical sweep root and add `--extend-existing`. The runner creates a new generation, reuses valid matching artifacts, evaluates only missing runs, replans stage 2 from the expanded stage-1 ranking, and refreshes the root-level products.
- To merge compatible completed sweeps, run the canonical spec with `--merge-from path/to/source_a path/to/source_b`. Valid duplicate run signatures are skipped, missing compatible run artifacts are copied into deterministic destination paths, and incompatible/conflicting sources are recorded in the generation merge log.
- Add `--dry-run` with spec mode to write `generations/<generation_id>/dry_run.json` without executing rectification or refreshing root-level products.

### Schema reference (additive fields)

- `RectificationRunConfig` (`config.json` per run):
  - `displacement_metric_family`
  - `displacement_alpha`
  - `displacement_component_weights`
  - `metric_formulation_id`
  - saved `config.json` files use canonical metric family names even when the input spec used a legacy alias
- Stage summaries (`stage1_summary.json`, `stage2_summary.json`):
  - baseline candidates use intensity-first displacement mode (`intensity`, `alpha=1.0`)
  - per-run statuses include `pending`, `running`, `ok`, `failed`, `skipped_valid`, `recomputed`
  - artifact-validation details and failure metadata are retained for resume
- `experiment_manifest.json`:
  - `generation_id`
  - `generation_products_dir`
  - `baseline_displacement_mode`
  - `experimental_metric_sweep`
  - `single_stage` baseline runs can still feed the optional `metric_experiments/` phase
- Metric experiment summary (`metric_experiments/metric_phase_summary.json`):
  - shortlisted baseline candidates expanded across configured formulation/alpha settings
- `run_state.json`:
  - `schema_version` (`2` for generation-aware outputs)
  - `active_generation_id`
  - `spec_generations`
  - `merge_history`
  - experiment fingerprint payload
  - resolved dataset identities (path, shape, size, mtime)
  - per-run status, attempts, error history, artifact validation state, active flag, generation id, and stable signatures
- `metric_record.json` / `summary.json` rows:
  - `candidate_signature`, `dataset_signature`, `run_signature`, `generation_id`, and explicit artifact paths
  - `metric_formulation` metadata
  - canonicalized metric family names in config/formulation metadata
  - displacement-field metrics
  - inter-view consistency metrics
  - displacement score diagnostics (blended and component-level)
- `alpha_report.json`:
  - saved under `metric_experiments/alpha_report.*` when the optional metric phase is enabled
  - ranking by configured primary metric with fallback chain
  - alpha-vs-quality/displacement/consistency aggregates by family
  - warning list for skipped/missing/corrupt artifacts (also saved as `alpha_report_warnings.json`)

### Sweep visualization and report export

Use the interactive visualizer to inspect completed or partial sweep folders:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output
```

The GUI now emphasizes two tasks:

- `Leaderboard` for stage/dataset filtered ranking, sorting, and parameter inspection.
- `Comparison` for candidate-vs-candidate preview and diagnostic map comparison on one dataset.

The existing `Metric Diagnostics` tab remains available for single-run drill-down and direct opening of preview/run artifacts.

To export a self-contained static report without launching Tk:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output --export-dir path/to/report_folder
```

The export keeps the existing bundle outputs (`visualization_bundle.npz`, `.mat`, root PNG plots) and additionally writes:

- `index.html` - static overview page with summary figures, top candidates, warnings, and per-dataset sections.
- `tables/candidate_leaderboard.csv`
- `tables/run_metric_records.csv`
- `data/report_context.json`
- `assets/` - top-K overview, preview sheets, diagnostic comparison sheets, and robustness heatmap.

### Raw metric explorer

Use the raw metric explorer when you want to re-rank existing sweep runs from `workspace.npz` rather than the originally saved sweep metrics:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output
```

For headless export:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --export-dir path/to/report_folder
```

This helper:

- reuses the sweep folder/stage/candidate layout from `sweep_visualization.py`
- recomputes metrics from `average_rectified_image` in each `workspace.npz`
- keeps the existing saved metrics and scores available as comparison baselines
- supports manual `top` / `good` / `bad` annotations stored under `metric_review/annotations.json`
- compares built-in and custom ranking profiles, including mixed old+new metric combinations
- saves custom profiles in the same config format accepted by `parameter_sweep.py`
- provides a persistent A/B comparison workspace with per-image Matplotlib toolbars, syncable zoom/pan, fit/1:1 controls, and pop-out viewers

Built-in new metrics include:

- normalized-DCT Shannon entropy on the rectified average and on an LoG guide map
- Bayes/DCTS-style block-DCT entropy on the rectified average and on an LoG guide map
- Gini index on the rectified average and on an LoG guide map
- Hoyer sparsity index on the rectified average and on an LoG guide map
- Vollath F4 on the rectified average and on an LoG guide map

Explorer-owned outputs are written under `metric_review/` and include:

- `candidate_metrics.csv`
- `profile_results.csv`
- `search_results.csv`
- `disagreements.csv`
- `report.json`
- optional `index.html` export summaries

---

## Sweep analysis (`sweep_analysis.py`)

`build_metric_record()` and `compute_image_metrics()` measure quality for each run using both **global-only** and **global+local** averages (when applicable). `build_metric_record()` also captures formulation metadata and displacement diagnostics used for alpha-centered comparisons.

Metrics include:

- **Average intensity**: mean pixel value.
- **Maximum intensity**: max pixel value.
- **Total variation**: sum of absolute differences in X and Y.
- **Sum of squared gradients**: sum of gradient magnitudes squared.
- **Laplacian variance**: variance of the Laplacian response.
- **Level-2 Haar DWT HV energy**: sum of squared coefficients from level-2 horizontal and vertical detail subbands.

`analyze_summary_data()` aggregates these across all runs and records minima, maxima, and means, which are then saved by `save_analysis()`.

`generate_alpha_report()` provides robust reporting for completed or partial experiments:

- loads final summaries when present, then falls back to stage summaries and per-run metrics
- uses first-valid loading (`primary`, then `.bak`) for supported artifacts
- records warnings for missing/corrupt artifacts and continues with valid subsets
- ranks by configurable primary metric (`laplacian_variance` default) with deterministic fallback metrics
- writes:
  - `alpha_report.json`
  - `alpha_report.csv`
  - `alpha_report_warnings.json`
  - alpha-trend plots for quality, displacement statistics, and consistency

---

## Validation tool (`validate_rectification.py`)

This tool allows you to validate your rectification results by comparing the rectified workspace output against new images from other cameras (e.g., Validation Camera, Thorcam).
The editor's manual `X/Y` scaling controls support an optional pre-blur anti-aliasing mode (Gaussian prefilter before bilinear resize, enabled by default) with sigma tied to per-axis downsampling.
In `fit_reference` mode, comparison overlays/diffs are generated in the processed input geometry (reference is resized to the edited input).

### Full GUI Guide

For complete step-by-step usage and control behavior, see:

- `moment_images/VALIDATE_RECTIFICATION_GUI_GUIDE.md`

The guide covers:

- Launch flow and file selection
- Detailed control reference (Geometry, Intensity, Scale, Alignment, Outputs)
- Preview tabs and status bar interpretation
- Recommended workflow per image
- Save outputs and troubleshooting

### Usage

1. Run the script:
   ```bash
   python -m moment_images.validate_rectification
   ```
2. Select either an `im_process` workspace (`.npz`) or a sweep output (`workspace.npz` or `rectified_views.npy`).
3. Select the Validation Camera image and Thorcam image when prompted.
4. Edit each image in the GUI and click `Done`.
5. Inspect the generated comparison figures.
6. Save the results and workspace if desired.

---

## Tips and troubleshooting

- **Image bounds**: `extract_views()` will zero-fill patches that fall outside the image. If you see blank views, verify the center point and radius.
- **Pixelwise performance**: pixelwise rectification is computationally heavy because it evaluates many displacements per pixel. Start with smaller grids and tighter `max_disp` settings.
- **Sub-pixel searches**: increasing `sub_pix_factor` improves alignment but increases compute and memory usage.
- **HDR workflows**: ensure `hdr_exposure_times` matches the number of exposure images when HDR assembly is enabled.

---

## References

For deeper implementation details, review the docstrings in `im_process.py` and `rectification_strategy.py`, which mirror the flow described above.



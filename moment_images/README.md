# Moment Images Package

`moment_images` contains the active light-field rectification, sweep,
validation, reporting, and gain-calibration tools.

Use this README as the package map. Detailed lab workflows live in focused
guides:

- [Rectification Guide](RECTIFICATION_GUIDE.md)
- [Parameter Sweep Guide](PARAMETER_SWEEP_GUIDE.md)
- [Sweep Reporting Guide](SWEEP_REPORTING_GUIDE.md)
- [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md)
- [Sweep Metric Explorer Guide](SWEEP_METRIC_EXPLORER_GUIDE.md)
- [Gain Calibration Guide](GAIN_CALIBRATION_GUIDE.md)

For Point Grey / FLIR camera preview and acquisition, see
[../POINT_GREY_ACQUISITION_GUIDE.md](../POINT_GREY_ACQUISITION_GUIDE.md).

## Package Workflows

| Workflow | Tool | Guide |
| --- | --- | --- |
| Interactive light-field rectification | `im_process.py` | [Rectification Guide](RECTIFICATION_GUIDE.md) |
| Scripted rectification | `rectify_views_from_config()` in `im_process.py` | [Rectification Guide](RECTIFICATION_GUIDE.md) |
| Parameter sweeps | `parameter_sweep.py` | [Parameter Sweep Guide](PARAMETER_SWEEP_GUIDE.md) |
| Sweep visualization and static export | `sweep_visualization.py` | [Sweep Reporting Guide](SWEEP_REPORTING_GUIDE.md) |
| Raw metric exploration and profile search | `sweep_metric_explorer.py` | [Sweep Metric Explorer Guide](SWEEP_METRIC_EXPLORER_GUIDE.md) |
| Summary/alpha report regeneration | `../generate_reports.py` | [Sweep Reporting Guide](SWEEP_REPORTING_GUIDE.md) |
| Rectification validation against external cameras | `validate_rectification.py` | [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md) |
| Batch validation image preprocessing | `batch_process_validation_images.py` | [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md) |
| Teledyne mean-variance system gain | `main_thor_gain.py` | [Gain Calibration Guide](GAIN_CALIBRATION_GUIDE.md) |
| Pairwise image linear gain regression | `pairwise_linear_gain.py` | [Gain Calibration Guide](GAIN_CALIBRATION_GUIDE.md) |
| Workspace unpacking | `unpack_workspace.py` | [Rectification Guide](RECTIFICATION_GUIDE.md) |

## Quick Commands

Run interactive rectification:

```bash
python -m moment_images.im_process
```

Run a spec-based parameter sweep:

```bash
python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json
```

Launch the sweep configuration GUI:

```bash
python -m moment_images.parameter_sweep --gui
```

Inspect a completed sweep:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output
```

Export a static sweep report:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output --export-dir path/to/report_folder
```

Re-rank a completed sweep from raw workspaces:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output
```

Run validation against Validation Camera and Thorcam images:

```bash
python -m moment_images.validate_rectification
```

Compute Teledyne mean-variance gain:

```bash
python -m moment_images.main_thor_gain --gui
```

## Main Concepts

- A **view grid** is a NumPy array shaped `(N, N, H, W)` for one exposure, or
  `(E, N, N, H, W)` for an exposure stack.
- `im_process.py` can build view grids interactively from source images and
  save workspace snapshots for later analysis.
- `rectify_views_from_config()` is the scripted entrypoint that runs the same
  rectification strategies without user prompts.
- `parameter_sweep.py` evaluates many rectification configurations and writes
  durable run folders, summary files, leaderboards, and optional metric
  experiment reports.
- `sweep_visualization.py` and `sweep_metric_explorer.py` are post-processing
  tools. They inspect completed sweep outputs; they do not acquire images.

## Rectification Strategies

The package supports three strategy families:

- `global`: estimate one integer translation per view against a reference view.
- `pixel`: estimate local displacement fields by patch matching.
- `combined`: run global pre-alignment, then pixelwise local refinement.

Pixelwise and combined runs can use smoothing, sub-pixel upscaling, sigma
pyramid refinement, and alternate local displacement score families. New sweep
configs should use canonical metric family names:

- `intensity`
- `blurred_gradient_corr`
- `blurred_laplacian_corr`
- `blurred_composite_corr`

Legacy aliases are still accepted where the code supports them, but saved
configs and reports use canonical names.

## Output Files At A Glance

Common rectification and sweep artifacts include:

- `avg_rectified.png`: preview image of the averaged rectified result.
- `rectified_views.npy`: rectified view stack.
- `workspace.npz`: reproducible workspace containing core arrays and
  diagnostics.
- `workspace.mat`: MATLAB companion export when workspace saving is used.
- `config.json`: run configuration.
- `metric_record.json`: per-run metric payload.
- `leaderboard.json` / `leaderboard.csv`: ranked sweep configurations.
- `run_state.json`: resumable sweep execution state.
- `experiment_manifest.json`: resolved sweep manifest and generation metadata.

See [Parameter Sweep Guide](PARAMETER_SWEEP_GUIDE.md) and
[Sweep Reporting Guide](SWEEP_REPORTING_GUIDE.md) for full output references.

## GUI And Headless Notes

Several tools are GUI-first and require a desktop session:

- `im_process.py`
- `parameter_sweep.py --gui`
- `sweep_visualization.py` without `--export-dir`
- `sweep_metric_explorer.py` without `--export-dir`
- `validate_rectification.py`
- `batch_process_validation_images.py`
- `main_thor_gain.py --gui`
- `unpack_workspace.py`

`batch_process_validation_images.py` is GUI-only today. Running it with `--help`
does not provide a normal argparse command reference.


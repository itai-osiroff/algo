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
- [Human Bead Oracle Guide](HUMAN_BEAD_ORACLE_GUIDE.md)
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
| Human bead oracle annotation and scoring | `human_bead_oracle.py` | [Human Bead Oracle Guide](HUMAN_BEAD_ORACLE_GUIDE.md) |
| Batch validation image preprocessing | `batch_process_validation_images.py` | [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md) |
| Trio benchmark against validation/ThorCam | `trio_benchmark.py` | [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md) |
| Batch trio benchmark evaluation | `trio_benchmark_batch.py` | [Validation GUI Guide](VALIDATE_RECTIFICATION_GUI_GUIDE.md) |
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

Import trio benchmark validation-oracle scores into the metric explorer:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --trio-results path/to/trio_benchmark_results.csv --export-dir path/to/metric_report
```

The trio oracle profile appears after importing trio results or loading the
saved `metric_review/trio_oracle_metrics.json` cache. Rows marked unrankable by
the trio benchmark remain inspectable but are excluded from oracle ranking and
validation-refinement seed export.

Create human bead-center annotations with the batch workbench:

```bash
python -m moment_images.human_bead_oracle annotate-gui --dataset-image scene_id=path/to/preview.png --output path/to/human_beads.json
```

Preview and score human bead annotations:

```bash
python -m moment_images.human_bead_oracle preview --annotations path/to/human_beads.json --output-dir path/to/previews
python -m moment_images.human_bead_oracle score-workspace --annotations path/to/human_beads.json --dataset-id scene_id --workspace path/to/workspace.npz --output path/to/score.json
```

When `parameter_sweep.py --gui` uses
`validation_oracle_sweep.oracle_mode = "human_bead"`, the Bead annotations field
is the convenient path for one shared annotation JSON. Per-dataset annotation
files are supported through advanced Scenes JSON overrides with
`human_bead_annotations`.

Run validation against Validation Camera and Thorcam images:

```bash
python -m moment_images.validate_rectification
```

Rank completed LF sweep candidates against processed Validation Camera and
ThorCam images:

```bash
python -m moment_images.trio_benchmark --sweep-dir path/to/sweep_output --dataset-id scene_id --validation-image path/to/validation_processed.tif --thorcam-image path/to/thorcam_processed.tif --output-dir path/to/trio_report
```

Use a saved `validate_rectification.py` workspace instead of exported reference
images:

```bash
python -m moment_images.trio_benchmark --sweep-dir path/to/sweep_output --dataset-id scene_id --validation-workspace path/to/validation_workspace.npz --output-dir path/to/trio_report
```

By default, trio registration uses `bead_geometric_v2`: a similarity-only
geometric bead initializer with local refinement, wider scale search, and
explicit underconstrained labels for one-bead scenes. Use
`--registration-profile bead_similarity_v1` or
`--registration-profile centered_ecc_legacy` to compare against older
initializers.

Run a manifest of trio scenes:

```bash
python -m moment_images.trio_benchmark_batch --manifest path/to/trio_manifest.json --output-dir path/to/trio_batch_report
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
- `human_bead_oracle.py annotate-gui`
- `human_bead_oracle.py annotate`
- `validate_rectification.py`
- `batch_process_validation_images.py`
- `main_thor_gain.py --gui`
- `unpack_workspace.py`

`batch_process_validation_images.py` is GUI-only today. Running it with `--help`
does not provide a normal argparse command reference.


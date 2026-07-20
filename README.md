# Light-Field Rectification and Lab Utilities

This repository contains Python tools used around the light-field rectification
workflow:

- extract and rectify light-field view grids
- run parameter sweeps across rectification settings
- inspect and export sweep reports
- validate rectified outputs against external camera images
- compute camera gain calibration values
- preview and capture raw frames from a Point Grey / FLIR camera

The `moment_images/` package is the main workflow. Point Grey acquisition is a
separate, lab-specific path at the repository root and requires a working
Spinnaker camera environment.

## Workflow Map

| Task | Start here | Main command |
| --- | --- | --- |
| Interactive rectification | [Rectification Guide](moment_images/RECTIFICATION_GUIDE.md) | `python -m moment_images.im_process` |
| Automated parameter sweep | [Parameter Sweep Guide](moment_images/PARAMETER_SWEEP_GUIDE.md) | `python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json` |
| Sweep GUI/report export | [Sweep Reporting Guide](moment_images/SWEEP_REPORTING_GUIDE.md) | `python -m moment_images.sweep_visualization path/to/sweep_output` |
| Raw metric re-ranking | [Sweep Metric Explorer Guide](moment_images/SWEEP_METRIC_EXPLORER_GUIDE.md) | `python -m moment_images.sweep_metric_explorer path/to/sweep_output` |
| Validation against other cameras | [Validation GUI Guide](moment_images/VALIDATE_RECTIFICATION_GUI_GUIDE.md) | `python -m moment_images.validate_rectification` |
| Human bead oracle annotations | [Human Bead Oracle Guide](moment_images/HUMAN_BEAD_ORACLE_GUIDE.md) | `python -m moment_images.human_bead_oracle annotate --dataset-image dataset_id=path/to/image --output path/to/human_beads.json` |
| Gain calibration | [Gain Calibration Guide](moment_images/GAIN_CALIBRATION_GUIDE.md) | `python -m moment_images.main_thor_gain --gui` |
| Point Grey / FLIR acquisition | [Point Grey Acquisition Guide](POINT_GREY_ACQUISITION_GUIDE.md) | `python acquire_images_PG.py --mode stream --output-dir path/to/pg_snaps` |

For a package-level command map, see [moment_images/README.md](moment_images/README.md).

## Environment Setup

Create and activate a Python environment, then install the tracked dependencies:

```bash
python -m pip install -r requirements.txt
```

Core workflows use:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `opencv-python`
- `tqdm`
- `h5py`
- `Django`
- `matlabengine` on Python versions below 3.14 when MATLAB export/integration is needed

Camera acquisition additionally requires:

- `pyspin`
- `simple_pyspin`
- a compatible FLIR/Spinnaker runtime and camera driver installation
- a desktop session where OpenCV can open preview windows

If you only run `moment_images` rectification, sweep, validation, or gain tools,
the Point Grey camera stack is optional.

## Quick Start

Run the interactive rectification workflow:

```bash
python -m moment_images.im_process
```

Run the included spec-based sweep template:

```bash
python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json
```

Open a completed sweep in the visualizer:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output
```

Export a static sweep report without opening the GUI:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output --export-dir path/to/report_folder
```

Run validation against external camera images:

```bash
python -m moment_images.validate_rectification
```

Run Point Grey live preview and raw snapshot capture:

```bash
python acquire_images_PG.py --mode stream --output-dir path/to/pg_snaps
```

## Command Reference

| Command | Purpose | Notes |
| --- | --- | --- |
| `python -m moment_images.im_process` | Interactive light-field view selection and rectification | Prompts for images, view grid settings, strategy choices, and save options |
| `python -m moment_images.parameter_sweep --spec path/to/spec.json` | Recommended automated sweep workflow | Supports staged sweeps, resume, extend, merge, dry-run, scoring profiles, and optional metric experiments |
| `python -m moment_images.parameter_sweep --gui` | GUI for building and running sweep specs | Useful when editing JSON by hand is inconvenient |
| `python -m moment_images.sweep_visualization path/to/sweep_output` | Inspect sweep leaderboards and candidate comparisons | Also supports `--export-dir` for static reports |
| `python -m moment_images.sweep_metric_explorer path/to/sweep_output` | Re-rank completed sweeps from raw `workspace.npz` artifacts | Supports annotations, custom metric profiles, `--metrics-config`, and trio oracle import via `--trio-results` |
| `python -m moment_images.human_bead_oracle annotate` | Interactively click bead centers/radii and write human-bead annotation JSON | Requires `--dataset-image dataset_id=path` and `--output`; opens a matplotlib window |
| `python -m moment_images.human_bead_oracle preview` | Render preview sheets for a human-bead annotation JSON | Requires `--annotations` and `--output-dir` |
| `python -m moment_images.human_bead_oracle score-workspace` | Score one `workspace.npz` against one dataset's human-bead annotations | Requires `--annotations`, `--dataset-id`, `--workspace`, and `--output` |
| `python generate_reports.py path/to/sweep_output` | Rebuild analysis/alpha report artifacts from sweep outputs | Use `--rebuild-summaries --spec path/to/spec.json` when summary products need regeneration |
| `python -m moment_images.validate_rectification` | Compare rectified output against Validation Camera and Thorcam images | GUI-only workflow |
| `python -m moment_images.batch_process_validation_images` | Batch preprocess validation images with shared geometry/intensity/scale settings | GUI-only; it does not expose a normal argparse `--help` interface today |
| `python -m moment_images.main_thor_gain --gui` | Compute system gain by Teledyne mean-variance method | Also supports CLI image/directory inputs |
| `python -m moment_images.pairwise_linear_gain` | Interactive two-image linear gain regression | Prompts for image paths and save location |
| `python -m moment_images.unpack_workspace` | Unpack saved `.npz` workspace arrays | GUI helper |
| `python acquire_images_PG.py --mode stream --output-dir path/to/snaps` | Point Grey / FLIR preview and raw snapshot capture | Requires camera SDK/runtime |

## Testing

Run the package tests:

```bash
pytest moment_images
```

Run repository-level tests:

```bash
pytest test_generate_reports.py test_acquire_images_PG.py
```

Useful documentation-adjacent checks:

```bash
pytest test_generate_reports.py
pytest test_acquire_images_PG.py
pytest moment_images/test_sweep_visualization.py moment_images/test_sweep_metric_explorer.py
```

## Troubleshooting

- Import errors for `pyspin` or `simple_pyspin`: install the camera bindings and
  verify the local FLIR/Spinnaker runtime. These packages are only needed for
  Point Grey acquisition.
- GUI window does not open: run from a desktop session with display access.
  The validation, sweep GUI, batch preprocessing, and camera preview workflows
  are not intended as headless operator paths.
- Sweep resume fails with a spec mismatch: use the same spec and dataset
  identities, or use `--extend-existing`/`--merge-from` when intentionally
  expanding compatible completed work.
- Sweep report lacks raw metrics: confirm each run folder contains
  `workspace.npz`; the raw metric explorer depends on that file.
- Validation alignment is stale: click `Run Alignment` again after geometry,
  intensity, or scale edits.


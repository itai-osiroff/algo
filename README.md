# Light-Field Rectification and Acquisition Utilities

This repository currently has two active workflows:

- `moment_images/` for light-field view extraction, rectification, parameter sweeping, analysis, and validation
- `acquire_images_PG.py` plus `PG_class.py` for Point Grey / FLIR camera preview and raw snapshot capture

`moment_images` is the main package workflow. Camera acquisition is optional, lab-specific, and depends on a working Point Grey / FLIR (Spinnaker) environment.

## Documentation Map

- [moment_images/README.md](moment_images/README.md): main package guide for rectification, sweeping, outputs, and analysis
- [moment_images/VALIDATE_RECTIFICATION_GUI_GUIDE.md](moment_images/VALIDATE_RECTIFICATION_GUI_GUIDE.md): end-to-end guide for the validation GUI
- [POINT_GREY_ACQUISITION_GUIDE.md](POINT_GREY_ACQUISITION_GUIDE.md): Point Grey acquisition workflow, controls, and troubleshooting

## Environment

### Core Python dependencies

The `moment_images` workflow is built around the packages already tracked in `requirements.txt`, especially:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `opencv-python`
- `tqdm`
- `h5py`
- `Django`
- `matlabengine` on Python versions below 3.14 when that integration is needed

### Camera acquisition-specific dependencies

The Point Grey workflow additionally depends on:

- `pyspin`
- `simple_pyspin`
- a Spinnaker-compatible camera runtime / SDK installation that matches the connected hardware and Python environment
- an OpenCV-capable desktop session for live preview windows

If you only use `moment_images`, you can treat the camera stack as optional and consult [POINT_GREY_ACQUISITION_GUIDE.md](POINT_GREY_ACQUISITION_GUIDE.md) only when you need hardware capture.

## Quick Start

Run the interactive rectification workflow:

```bash
python -m moment_images.im_process
```

Run the spec-based parameter sweep:

```bash
python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json
```

Run the Point Grey preview and save raw snapshots into an explicit output folder:

```bash
python acquire_images_PG.py --mode stream --output-dir path/to/pg_snaps
```

## Testing

Run the package test suite:

```bash
pytest moment_images
```

Run the hardware-free preview/control tests for the Point Grey script:

```bash
pytest test_acquire_images_PG.py
```

## Notes

- `moment_images/README.md` is the source of truth for sweep configuration fields, run artifacts, and rectification behavior.
- The Point Grey preview script is intentionally separate from `moment_images` because it is tied to one hardware path and a GUI-based operator workflow.

# Moment Images Utilities

This package contains the interactive `im_process.py` light-field rectification workflow and a companion parameter-sweeping CLI to run the same pipeline non-interactively.

## Running the interactive pipeline

Run the script directly when you want to manually step through viewpoint selection and visualise outputs:

```bash
python -m moment_images.im_process
```

The script will prompt you for image selection and viewpoint choices. Figures are displayed at the end, and you can optionally save them as high-resolution JPG files.

> Tip: When running from the repository root, Python automatically discovers the package thanks to the script-local import helper, so no extra `PYTHONPATH` configuration is required.

## Sweeping rectification parameters

Use `parameter_sweep.py` to iterate over window sizes, displacement limits, and Gaussian smoothing values while saving results for each combination:

```bash
python -m moment_images.parameter_sweep \
  --views path/to/views.npy \
  --output-dir sweep_runs/experiment1 \
  --strategy pixel \
  --window-start 21 --window-stop 41 --window-step 10 \
  --max-disp-start 5 --max-disp-stop 25 --max-disp-step 10 \
  --gauss-sigma-start 3.0 --gauss-sigma-stop 7.0 --gauss-sigma-step 2.0 \
  --disp-gauss-start 0.0 --disp-gauss-stop 1.0 --disp-gauss-step 1.0
```

### Required inputs
- **`--views`** must point to a NumPy array shaped `(N, N, H, W)` containing the grayscale viewpoint patches you want to rectify.
- **`--strategy`** accepts `global`, `pixel`, or `combined` to match the strategies available in the interactive flow.

### Outputs
- Each parameter combination writes a folder under `--output-dir` with the averaged rectified image (`avg_rectified.png`), rectified stack (`rectified_views.npy`), and the exact configuration (`config.json`).
- A top-level `summary.json` collects per-run metrics (e.g., average and maximum intensities) for quick comparison.
- Add `--save-figures` to capture the same figures produced in the interactive workflow. Figures are written under each run folder in a logical hierarchy: `figures/rectified_views`, `figures/overlays`, and `figures/averages`.

The CLI uses sensible defaults so you can start a sweep with minimal arguments:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy
```

Progress is reported with `tqdm`, and the output directory defaults to `sweep_runs/<timestamp>` when not specified. Figure saving runs headlessly and does not display the plots while producing files matching the interactive sizes and formats.

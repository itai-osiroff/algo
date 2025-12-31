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

Use `parameter_sweep.py` to iterate over window sizes, displacement limits, and Gaussian smoothing values while saving results for each combination. You can launch a minimal GUI for configuring the sweep (no CLI flags required) with:

```bash
python -m moment_images.parameter_sweep --gui
```

The GUI collects the same inputs as the CLI with friendly defaults. If you prefer the command line, the sweep still builds the view grid for you from a source image before iterating so every combination rectifies the same cached stack:

```bash
python -m moment_images.parameter_sweep \
  --source-image path/to/source.png \
  --center-x 640 --center-y 360 \
  --view-radius 125 \
  --view-grid-size 3 \
  --output-dir sweep_runs/experiment1 \
  --strategy pixel \
  --window-start 21 --window-stop 41 --window-step 10 \
  --max-disp-start 5 --max-disp-stop 25 --max-disp-step 10 \
  --gauss-sigma-start 3.0 --gauss-sigma-stop 7.0 --gauss-sigma-step 2.0 \
  --disp-gauss-start 0.0 --disp-gauss-stop 1.0 --disp-gauss-step 1.0 \
  --pixel-thresh-start 0.05 --pixel-thresh-stop 0.15 --pixel-thresh-step 0.05
```

### Required inputs
- Provide either **`--views`** (a NumPy array shaped `(N, N, H, W)`) or the combination of **`--source-image`**, **`--center-x`**, **`--center-y`**, and **`--view-radius`** so the sweep can extract views once before iterating.
- Use **`--view-grid-size`** to control how many views per dimension are extracted. Add **`--view-distance`** to override the computed distance between adjacent view centers.
- **`--strategy`** accepts `global`, `pixel`, or `combined` to match the strategies available in the interactive flow.
- Threshold sweeps: **`--global-thresh-*`** control the correlation mask used in the global strategy, while **`--pixel-thresh-*`** affect pixel-wise patch similarity calculations.

### Outputs
- The sweep caches the generated view grid to `generated_views.npy` inside `--output-dir` before running any rectification iterations.
- Each parameter combination writes a folder under `--output-dir` with the averaged rectified image (`avg_rectified.png`), rectified stack (`rectified_views.npy`), and the exact configuration (`config.json`).
- A top-level `summary.json` collects per-run metrics for both the globally averaged image and the globally+locally refined average (intensity, gradients, and Laplacian energy/variance). Add `--analyze` to also write aggregated extrema and means: the detailed structure remains in `analysis.json`, and a compact `analysis.csv` provides a quick, filterable table with the same min/max/mean data per metric and alignment scope.
- Add `--save-figures` to capture the same figures produced in the interactive workflow. Figures are written under each run folder in a logical hierarchy: `figures/rectified_views`, `figures/overlays`, and `figures/averages`.

The CLI uses sensible defaults so you can start a sweep with minimal arguments:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy
```

Progress is reported with `tqdm`, and the output directory defaults to `sweep_runs/<timestamp>` when not specified. Figure saving runs headlessly and does not display the plots while producing files matching the interactive sizes and formats.

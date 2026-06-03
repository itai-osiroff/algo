# Gain Calibration Guide

This guide covers the two gain-related workflows in `moment_images`.

## Teledyne Mean-Variance System Gain

Use `main_thor_gain.py` to compute camera system gain from a bias image and two
flat-field captures.

The method uses:

```text
gain = mean(flat1 - bias) / variance((flat2 - flat1) ROI)
```

The implementation derives variance from the ROI standard deviation as
`std^2 / 2`.

### GUI

Launch:

```bash
python -m moment_images.main_thor_gain --gui
```

The GUI lets you choose bias, flat1, and flat2 inputs, ROI settings, and an
optional output JSON path.

### CLI With Single Images

```bash
python -m moment_images.main_thor_gain \
  --bias-image path/to/bias.tiff \
  --flat1-image path/to/flat1.tiff \
  --flat2-image path/to/flat2.tiff \
  --roi-size 100 \
  --output-json gain_result.json
```

### CLI With Directory Stacks

```bash
python -m moment_images.main_thor_gain \
  --bias-dir path/to/bias_frames \
  --flat1-dir path/to/flat1_frames \
  --flat2-dir path/to/flat2_frames \
  --pattern "*.tif*" \
  --roi-size 100
```

Directory inputs are averaged into bias/flat images before the gain calculation.

### Options

| Option | Meaning |
| --- | --- |
| `--bias-image` / `--bias-dir` | Bias input as one image or a directory stack |
| `--flat1-image` / `--flat1-dir` | First flat input as one image or a directory stack |
| `--flat2-image` / `--flat2-dir` | Second flat input as one image or a directory stack |
| `--pattern` | Glob pattern for directory inputs, for example `"*.tif*"` |
| `--roi-size` | Square ROI side length in pixels; default is `100` |
| `--roi-center-x`, `--roi-center-y` | Optional ROI center |
| `--output-json` | Optional JSON result path |
| `--gui` | Launch GUI instead of CLI execution |

## Pairwise Linear Gain Regression

Use `pairwise_linear_gain.py` when you need an interactive linear regression
between two images:

```bash
python -m moment_images.pairwise_linear_gain
```

The script prompts for:

1. image 1 path, used as the independent variable `X`
2. image 2 path, used as the dependent variable `Y`
3. whether to include an intercept in the regression
4. whether to save outputs after reviewing the figure

Saved outputs share the requested base path:

- `.npz`: workspace variables and regression results
- `.mat`: MATLAB companion export
- `.png`: visualization figure

## Choosing A Workflow

- Use `main_thor_gain.py` for the standard mean-variance system gain workflow.
- Use `pairwise_linear_gain.py` for exploratory two-image linear regression.

## Troubleshooting

- Input shapes differ: use images with matching dimensions for pairwise
  regression.
- Directory stacks load no files: check `--pattern` and file extensions.
- ROI is invalid: choose an ROI center and size that fit within the image.
- GUI does not open: run from a desktop session with display access.
- Output JSON is missing: pass `--output-json` or choose a save path in the GUI.


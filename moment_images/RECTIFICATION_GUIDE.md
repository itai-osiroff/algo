# Rectification Guide

This guide covers interactive and scripted light-field rectification with
`moment_images.im_process`.

## When To Use This

Use rectification when you have a light-field image or a saved view grid and
want to align the sub-views into an averaged rectified output.

Typical inputs are:

- a grayscale-compatible source image selected interactively
- a directory of source images, optionally treated as HDR exposures
- a precomputed view grid shaped `(N, N, H, W)`
- an HDR exposure view grid shaped `(E, N, N, H, W)`

## Interactive Workflow

Launch from the repository root:

```bash
python -m moment_images.im_process
```

The interactive workflow walks through:

1. Select source image(s).
2. Choose viewpoint selection mode: manual or circle detection.
3. Extract an `N x N` grid of view patches.
4. Choose rectification strategy and parameters.
5. Review figures and save outputs/workspace when needed.

The workflow can save figures, rectified arrays, and a compressed workspace for
later analysis.

## Viewpoint Selection

Manual selection is useful when the view grid is clear and the operator knows
the center/radius.

Circle detection is useful when the source image has detectable circular view
features. It uses an ROI plus Hough circle detection, then maps detections back
to full-image coordinates for review.

If extracted views contain blank patches, check:

- center coordinates
- radius
- view grid size
- view distance
- whether any patch falls outside the source image bounds

## Rectification Strategies

### Global

`global` estimates one translation per view against the reference view, warps
each view, and averages the result. It is the fastest option and works best
when each view only needs one overall shift.

### Pixel

`pixel` estimates per-pixel displacement fields by patch matching. It is slower
and more memory-intensive, but can correct local deformation that a single
global translation cannot handle.

Important parameters:

- `similarity_window_size`: local patch size.
- `max_displacement`: displacement search bound.
- `gauss_sigma`: smoothing applied while scoring candidate displacements.
- `disp_gauss_sigma`: optional smoothing on displacement fields.
- `sub_pix_factor`: upscaling factor for finer displacement search.
- `pixel_sigma_pyramid`: multi-sigma refinement path.
- `zero_fill_roll`: avoid wraparound values when rolling candidate views.

### Combined

`combined` runs global translation first, then pixelwise refinement. The pixel
reference can be the global average, the center view, or the iterative
subtractive-max rebuild mode exposed by the sweep CLI.

## Scripted Rectification

Use `rectify_views_from_config()` when another script or sweep needs
deterministic rectification without prompts.

Minimal example:

```python
import numpy as np

from moment_images.im_process import RectificationRunConfig, rectify_views_from_config

views = np.load("path/to/views.npy")
config = RectificationRunConfig(
    strategy="combined",
    similarity_window_size=25,
    max_displacement=15,
    gauss_sigma=5.0,
    disp_gauss_sigma=1.0,
    pixel_threshold=0.05,
)

result = rectify_views_from_config(views, config)
np.save("rectified_views.npy", result.rectified_views)
```

`RectificationRunConfig` accepts the same main strategy controls used by sweeps.
For local displacement score families, prefer canonical names:

- `intensity`
- `blurred_gradient_corr`
- `blurred_laplacian_corr`
- `blurred_composite_corr`

## Workspaces And Outputs

Common saved artifacts:

- `avg_rectified.png`: averaged rectified image preview.
- `rectified_views.npy`: rectified view stack.
- `workspace.npz`: compressed workspace with arrays and diagnostics.
- `workspace.mat`: MATLAB companion export for compatible variables.
- figure image files: original views, similarity maps, rectified views,
  overlays, and averages when figure saving is enabled.

Use the workspace unpacker when you need to inspect saved arrays:

```bash
python -m moment_images.unpack_workspace
```

`unpack_workspace.py` is a GUI helper.

## Troubleshooting

- Blank extracted views: verify center/radius/grid settings and image bounds.
- Pixelwise runs are slow: reduce grid size, patch window, `max_displacement`,
  or `sub_pix_factor`.
- Memory pressure: avoid large sub-pixel factors and broad displacement grids on
  full-size views.
- Unexpected metric family names in saved configs: the code canonicalizes legacy
  aliases on output.
- GUI figures do not appear: run from a desktop session with display access.


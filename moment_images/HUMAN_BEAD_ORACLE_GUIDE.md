# Human Bead Oracle Guide

This guide covers `moment_images.human_bead_oracle`, the manual bead-center
oracle used to score sweep candidates without running trio validation
benchmarking.

For sweep setup, see [PARAMETER_SWEEP_GUIDE.md](PARAMETER_SWEEP_GUIDE.md).
For metric explorer exports, see
[SWEEP_METRIC_EXPLORER_GUIDE.md](SWEEP_METRIC_EXPLORER_GUIDE.md).

## What This Tool Does

The human bead oracle scores rectified LF outputs against manually clicked bead
centers. It is useful when you have bead locations in the LF output frame and
want validation-guided ranking without Validation Camera / ThorCam references.

It provides:

- an interactive annotation command
- preview image rendering for annotation review
- one-workspace scoring for debugging
- integration with `validation_oracle_sweep.oracle_mode = "human_bead"`
- imported oracle metrics in the metric explorer

## Annotation JSON

The annotation file contains one `datasets` entry per sweep dataset. Coordinates
are in image pixel coordinates for the image used during annotation.

```json
{
  "version": 1,
  "datasets": [
    {
      "dataset_id": "scene_a",
      "image_path": "scene_a_preview.png",
      "image_shape": [512, 512],
      "default_radius_px": 4.0,
      "beads": [
        {"id": "b001", "x": 124.5, "y": 220.0, "radius_px": 4.0},
        {"id": "b002", "x": 260.0, "y": 190.5, "radius_px": 4.0}
      ]
    }
  ]
}
```

Fields:

- `dataset_id`: must match the sweep dataset id.
- `image_path`: optional preview image path used by `preview`; relative paths
  are resolved from the annotation JSON folder.
- `image_shape`: `[height, width]` for the annotated image.
- `default_radius_px`: radius assigned to newly clicked beads.
- `beads`: bead centers and radii. `id` may also be written as `bead_id`.

If a candidate image is uniformly scaled relative to `image_shape`, bead
coordinates and radii are scaled automatically. Non-uniform shape changes are
rejected because they would distort bead geometry.

## Create Annotations

Run from the project root:

```bash
python -m moment_images.human_bead_oracle annotate-gui \
  --dataset-image scene_a=path/to/scene_a_preview.png \
  --dataset-image scene_b=path/to/scene_b_preview.png \
  --dataset-folder path/to/more_preview_images \
  --output path/to/human_beads.json \
  --default-radius-px 4
```

The command opens the Tkinter annotation workbench. `--dataset-folder` imports
`.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` files and maps filename stems to
dataset ids. You can also launch the workbench with only `--output` and add
images or folders from the window.

Controls:

- left click empty space: add a bead
- left click near a bead: select and move that bead
- toolbar pan/zoom: inspect dense images before clicking
- bead table / inspector: select, edit id/x/y/radius, and apply changes
- `+` or keypad plus: increase selected bead radius
- `-` or keypad minus: decrease selected bead radius
- `delete` or `backspace`: remove selected bead
- `n` or right arrow: next dataset
- `p` or left arrow: previous dataset
- `Ctrl+S`: save
- `Ctrl+Z` / `Ctrl+Y`: undo / redo

The workbench autosaves when switching datasets and asks whether to save on
close. Re-running `annotate-gui` against an existing JSON preserves datasets
that are not opened in the session.

The older lightweight Matplotlib annotator remains available:

```bash
python -m moment_images.human_bead_oracle annotate \
  --dataset-image scene_a=path/to/scene_a_preview.png \
  --output path/to/human_beads.json \
  --default-radius-px 4
```

## Preview Annotations

Render overlay previews for every dataset whose annotation entry has an
available `image_path`:

```bash
python -m moment_images.human_bead_oracle preview \
  --annotations path/to/human_beads.json \
  --output-dir path/to/human_bead_previews
```

Outputs are named:

```text
<dataset_id>_human_beads.png
```

Use these previews before running a long sweep. They catch wrong dataset ids,
bad image paths, shifted coordinates, and radius mistakes.

## Score One Workspace

Use `score-workspace` to inspect one candidate before wiring the oracle into a
sweep:

```bash
python -m moment_images.human_bead_oracle score-workspace \
  --annotations path/to/human_beads.json \
  --dataset-id scene_a \
  --workspace path/to/workspace.npz \
  --output path/to/human_bead_score.json
```

The output JSON includes:

- `rankable`: whether enough bead evidence was found
- `metrics`: aggregate scalar metrics used for ranking
- `diagnostics`: rankability metadata plus `human_bead_per_bead_metrics`
- `warnings`: scoring or loading warnings

Important aggregate metrics:

- `human_bead_oracle_score_v1`: ranking score, higher is better
- `human_bead_evaluated_bead_count`: number of beads scored
- `human_bead_fwhm_radius_px_mean`: smaller compact core is better
- `human_bead_support_radius_px_mean`: smaller support halo is better
- `human_bead_halo_to_core_ratio_mean`: lower halo is better
- `human_bead_core_energy_fraction_mean`: higher core concentration is better
- `human_bead_elongation_mean`: lower elongation is better
- `human_bead_center_shift_px_p95`: lower local peak shift is better

`human_bead_per_bead_metrics` is structured JSON, not a set of ranking columns.
Each bead entry contains `bead_id`, `rankable`, `x`, `y`, `radius_px`, and that
bead's metric values.

## Use In Parameter Sweep

Set `validation_oracle_sweep.oracle_mode` to `"human_bead"` and provide
`human_bead_annotations` in defaults or per scene.

Shared annotation JSON for all datasets:

```json
{
  "validation_oracle_sweep": {
    "enabled": true,
    "oracle_mode": "human_bead",
    "top_k": 8,
    "budget_per_dataset": 4,
    "skip_image_metric_stage": true,
    "defaults": {
      "human_bead_annotations": "path/to/human_beads.json"
    },
    "scenes": [
      {"dataset_id": "scene_a"},
      {"dataset_id": "scene_b"}
    ]
  }
}
```

Per-dataset annotation files:

```json
{
  "validation_oracle_sweep": {
    "enabled": true,
    "oracle_mode": "human_bead",
    "top_k": 8,
    "scenes": [
      {
        "dataset_id": "scene_a",
        "human_bead_annotations": "path/to/scene_a_beads.json"
      },
      {
        "dataset_id": "scene_b",
        "human_bead_annotations": "path/to/scene_b_beads.json"
      }
    ]
  }
}
```

In human-bead mode the sweep does not run trio benchmarking. It evaluates LF
candidate `workspace.npz` artifacts directly, ranks stage 1 by
`human_bead_oracle_score_v1`, then ranks validation-phase candidates with the
same score.

## GUI Setup

In `python -m moment_images.parameter_sweep --gui`:

1. Enable `Validation Oracle Sweep`.
2. Set `Mode` to `human_bead`.
3. Choose the workflow:
   - normal validation: run image-metric stage 2, then validate
   - validation-guided refinement: skip image-metric stage 2
4. Use `Bead annotations` and `Browse` for one shared annotation JSON.
5. Use `Annotate` beside `Bead annotations` to open the workbench from the
   sweep GUI. If the path is empty, choose where to save `human_beads.json`.
6. For per-dataset annotation files, enter scene overrides in `Scenes JSON`
   using `human_bead_annotations`.

The dataset row validation workspace / validation image / ThorCam fields are
not required in human-bead mode.

## Outputs And Metric Explorer

Human-bead oracle runs write:

```text
validation_oracle/
  stage1/human_bead_oracle_import.json
  validation_phase/human_bead_oracle_import.json
metric_review/
  trio_oracle_metrics.json
```

The `metric_review/trio_oracle_metrics.json` filename is reused for compatibility
with the existing metric explorer cache. Human-bead records inside that cache
carry `oracle_mode: "human_bead"` diagnostics and
`human_bead_oracle_score_v1` metrics.

When the metric explorer loads a human-bead cache:

- the `human_bead_oracle` profile appears
- human-bead scalar metrics appear in `candidate_metrics.csv`
- `oracle_diagnostics_json` contains per-bead detail
- `validation_refinement_seeds.json` uses `human_bead_oracle_score_v1` for
  human-bead-only caches

## Troubleshooting

### The sweep says annotations are missing

Check that every sweep dataset has either:

- a matching `dataset_id` entry in the shared annotation JSON, or
- a scene override with `human_bead_annotations`.

Dataset ids must match exactly.

### A candidate is unrankable

Common causes:

- too few annotated beads are visible in the candidate
- annotated image shape changed non-uniformly
- bead radius is too small or too large for the rendered candidate
- the workspace lacks a usable `average_rectified_image` or `rectified_views`

Inspect `warnings`, `ranking_exclusion_reason`, and
`human_bead_per_bead_metrics` in the score output or exported diagnostics.

### Preview writes zero images

The annotation entries may not have `image_path`, or the paths may not exist.
Relative `image_path` values are resolved relative to the annotation JSON file.

### The GUI still asks for validation images

Confirm the validation oracle `Mode` is `human_bead`. Trio validation mode still
requires a validation workspace or a validation/ThorCam image pair.

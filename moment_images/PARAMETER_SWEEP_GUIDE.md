# Parameter Sweep Guide

This guide covers `moment_images.parameter_sweep`, the automated runner for
evaluating rectification settings across one or more datasets.

## Recommended Command

Use spec mode for new work:

```bash
python -m moment_images.parameter_sweep --spec moment_images/sweep_experiment_spec_example.json
```

Use the GUI when building or editing a spec interactively:

```bash
python -m moment_images.parameter_sweep --gui
```

Legacy single-dataset CLI flags remain available:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy --strategy pixel
```

For pixel and combined legacy sweeps, shifted-view power is configured with a
positive range:

- `--pixel-shift-power-start` (default `1.0`)
- `--pixel-shift-power-stop` (default `1.0`)
- `--pixel-shift-power-step` (default `1.0`)

For example, this evaluates signed powers `0.5`, `1.0`, `1.5`, and `2.0`:

```bash
python -m moment_images.parameter_sweep --views path/to/views.npy --strategy pixel --pixel-shift-power-start 0.5 --pixel-shift-power-stop 2.0 --pixel-shift-power-step 0.5
```

## Spec Structure

A sweep spec usually contains:

- `output_dir`: sweep root directory.
- `strategies`: any of `global`, `pixel`, `combined`.
- `datasets`: one or more view-grid or source-image datasets.
- `search_spaces`: strategy-specific parameter candidates.
- `stages`: `two_stage` or `single_stage` execution settings.
- `scoring`: metric weights or a shared profile.
- `experimental_metric_sweep`: optional local displacement metric experiments.
- `reporting`: alpha report and primary metric settings.
- `save_figures`: whether to save full figure sets per run.
- `analyze`: whether to write aggregate analysis artifacts.

The tracked template is:

```text
moment_images/sweep_experiment_spec_example.json
```

## Dataset Inputs

Each dataset needs a stable `id` and one input source:

- `views`: path to a saved view grid shaped `(N, N, H, W)` or
  `(E, N, N, H, W)`.
- `source_image`: image path plus `center_x`, `center_y`, `view_radius`, and
  optional `view_grid_size`/`view_distance`.
- `source_image_dir`: exposure image directory for HDR view generation.

Set `normalize_to_8bit` explicitly. The current default is `false`, preserving
raw intensities during source extraction and HDR handling.

## Search Spaces

Common pixel/combined search fields:

- `similarity_window_sizes` or `refinement_window_sizes`
- `max_displacements`
- `gauss_sigmas`
- `disp_gauss_sigmas`
- `pixel_thresholds`
- `sub_pix_factors`
- `pixel_sigma_pyramid`
- `pixel_shift_powers` (positive signed powers, default `[1.0]`)
- `zero_fill_roll`

In the spec-based sweep GUI, enter comma-separated values in the **Pixel shift
powers** and **Combined pixel shift powers** controls. The two strategy lists
are independent; every value must be finite and greater than zero.

`pixel_shift_powers` applies only to pixelwise intensity matching (including the
pixelwise stage of `combined`). At values other than `1.0`, the shifted moving
view is evaluated as `sign(x) * abs(x) ** power`; non-finite inputs are treated
as zero only in the scoring copy and recorded in each run's diagnostics. This
does not alter global matching, raw/HDR view stacks, or rectified output. HDR
diagnostics describe the first exposure used to estimate the pixel displacement
field, not subsequent exposures that only receive that field.

Combined/global-specific fields:

- `correlation_pad_widths`
- `global_thresholds`
- `combined_use_global_avg_reference`
- `combined_reference_mode`

`combined_reference_mode` may select:

- `global_average`
- `center_view`
- `iterative_subtractive_max_rebuild`

## Stages

`stages.mode` controls sweep execution:

- `two_stage`: coarse stage 1 followed by refinement of top candidates.
- `single_stage`: one pass over the configured candidate grid.

For `two_stage`, the default refinement method is `auto_fine`, which inserts
midpoint and bounded edge candidates around high-ranking coarse settings.
`neighbor` is available for the older behavior that reuses configured neighbor
values only.

## Scoring Profiles

`scoring.metric_weights` can define metric weights directly. A spec can also
reference metric explorer profiles:

```json
{
  "scoring": {
    "profile_name": "guide_focus_mix",
    "profile_config_path": "metric_review/metrics_config.json",
    "metric_scope": "global_and_local_metrics",
    "runtime_penalty_weight": 0.0,
    "robustness_lambda": 0.5,
    "missing_dataset_penalty": 1.0
  }
}
```

CLI overrides are available:

```bash
python -m moment_images.parameter_sweep --spec sweep_spec.json --metrics-config metric_review/metrics_config.json --scoring-profile guide_focus_mix
```

Profiles that depend on `saved_score` or `saved_mean_dataset_score` are
comparison-only and cannot rank new runs.

## Optional Metric Experiments

`experimental_metric_sweep` runs metric-family comparisons under
`metric_experiments/`. It supports two candidate sources:

- `candidate_source: "independent_search"` runs a separate metric search for
  each formulation/alpha lane. `search_stages.total_budget_per_dataset` is the
  per-lane budget, so adding formulations or alpha values increases the total
  run count. This is the GUI default and is the right mode when a metric family
  might prefer different rectification parameters than the vanilla intensity
  baseline.
- `candidate_source: "baseline_shortlist"` preserves the legacy behavior. It
  takes the final baseline leaderboard, keeps `shortlist_top_k` pixel/combined
  candidates, strips metric axes, and reruns those candidates across the
  requested formulations. Existing specs without `candidate_source` use this
  mode for compatibility.

Supported formulation families:

- `intensity` (`alphas` must be `[1.0]`; this is the vanilla comparator)
- `blurred_gradient_corr`
- `blurred_laplacian_corr`
- `blurred_composite_corr`

For pair families, `alpha` blends intensity with the selected auxiliary score:

```text
score = alpha * intensity + (1 - alpha) * auxiliary
```

For `blurred_composite_corr`, `alpha` is the intensity weight and
`component_weights` distribute the remaining weight across non-intensity
components.

## Validation Oracle Sweep

`validation_oracle_sweep` adds a validation-guided phase after the normal
two-stage sweep. The recommended reference input is the
`validation_workspace.npz` saved by `validate_rectification.py`; the GUI exposes
this as the dataset row `Workspace` field. Workspace rows are emitted with
`use_validation_workspace_transforms: true`, so trio benchmarking replays the
saved validation/ThorCam alignment on the rectified grid.

Set `skip_image_metric_stage: true` to skip the normal image-metric `stage2`.
That mode runs `stage1`, ranks it with the validation oracle, runs
`validation_phase` from that oracle ranking, and promotes the validation-ranked
results to the root `leaderboard.json`, `best_configs.json`, and
`summary.json` outputs. In the GUI this is exposed as
`Validation-guided refinement: skip image-metric stage2`.

```json
{
  "validation_oracle_sweep": {
    "enabled": true,
    "top_k": 8,
    "budget_per_dataset": 4,
    "skip_image_metric_stage": true,
    "defaults": {
      "registration_profile": "bead_geometric_v2"
    },
    "scenes": [
      {
        "dataset_id": "scene_a",
        "validation_workspace": "path/to/validation_workspace.npz",
        "use_validation_workspace_transforms": true
      }
    ]
  }
}
```

Legacy exported references still work by providing both `validation_image` and
`thorcam_image` for a scene. Use the advanced scene JSON for non-reference
fields such as `scene_id`, `anchor_row_id`, and registration tuning.

To rank validation phases from manually clicked bead centers instead of trio
benchmark references, set `oracle_mode: "human_bead"` and provide
`human_bead_annotations`. See
[HUMAN_BEAD_ORACLE_GUIDE.md](HUMAN_BEAD_ORACLE_GUIDE.md) for the annotation
workflow, JSON schema, GUI setup, and output interpretation.

## Resume, Extend, Merge, Dry Run

Spec-mode sweeps write durable state to `run_state.json`.

Common operations:

```bash
python -m moment_images.parameter_sweep --spec sweep_spec.json --output-dir path/to/sweep
python -m moment_images.parameter_sweep --spec expanded_spec.json --output-dir path/to/sweep --extend-existing
python -m moment_images.parameter_sweep --spec canonical_spec.json --output-dir path/to/sweep --merge-from path/to/source_a path/to/source_b
python -m moment_images.parameter_sweep --spec moved_spec.json --relocate-path old/root=new/root --adopt-relocated --dry-run
python -m moment_images.parameter_sweep --spec sweep_spec.json --output-dir path/to/sweep --dry-run
```

- Resume reuses valid completed artifacts when the experiment fingerprint is
  compatible.
- `--extend-existing` is for compatible spec expansion.
- `--merge-from` copies valid compatible completed runs from other sweep roots.
- `--relocate-path OLD=NEW` rewrites spec paths before loading datasets and
  scoring profiles. Repeat it for multiple moved roots.
- `--adopt-relocated` updates copied `run_state.json` path metadata only when
  the normalized experiment fingerprint is compatible. Start with `--dry-run`
  to preview reused and pending runs before refreshing root-level products.
- `--dry-run` writes a generation dry-run report without executing
  rectification or refreshing root-level products.

When a sweep root and its inputs were moved together, use explicit root
mappings instead of editing `run_state.json` by hand:

```bash
python -m moment_images.parameter_sweep \
  --spec sweep_spec.json \
  --relocate-path C:/old/sweep_root=D:/new/sweep_root \
  --relocate-path C:/old/datasets=D:/new/datasets \
  --adopt-relocated \
  --dry-run
```

If the dry-run report under `generations/<generation_id>/dry_run.json` looks
right, rerun the same command without `--dry-run`. Datasets generated from
`source_image` or `source_image_dir` can resume from copied cached views under
`datasets/<dataset_id>/` when those views match the saved dataset identity.

## Output Layout

Modern sweep roots commonly contain:

- `stage1_summary.json`
- `stage2_summary.json` for `two_stage`
- `leaderboard.json`
- `leaderboard.csv`
- `best_configs.json`
- `summary.json`
- `analysis.json` / `analysis.csv` when analysis is enabled
- `run_state.json`
- `experiment_manifest.json`
- `validation_oracle/`, `validation_leaderboard.json`, and
  `validation_phase_summary.json` when validation oracle sweep completes
- no `stage2_summary.json` when validation oracle runs with
  `skip_image_metric_stage: true`
- stale `stage2` products from a reused output directory are moved under
  `generations/<generation_id>/inactive_artifacts/stage2/`
- `runs/<stage>/<candidate>/<dataset>/...`
- `generations/<generation_id>/products/...`

Each run folder commonly contains:

- `avg_rectified.png`
- `rectified_views.npy`
- `workspace.npz`
- `workspace.mat`
- `config.json`
- `metric_record.json`

When metric experiments are enabled, additional outputs are written under
`metric_experiments/`, including:

- `metric_phase_summary.json`
- `leaderboard.json`
- `leaderboard.csv`
- `summary.json`
- `alpha_report.json`
- `alpha_report.csv`
- `alpha_report_warnings.json`

## Troubleshooting

- Resume rejects a sweep: verify the spec, dataset paths, shapes, file sizes,
  and modification times. Use `--extend-existing` only for compatible expansion.
- Missing or corrupt run artifacts: the runner recomputes affected runs when
  required artifacts fail validation.
- Too many candidates: narrow search spaces or reduce `total_budget_per_dataset`.
- Pixel/combined runs are slow: reduce `max_displacements`, window sizes,
  `sub_pix_factors`, or worker count.
- GUI does not open: use a desktop session or run spec mode from the CLI.


# Bead-Prior Foldover Handoff

Context for continuing the bead-prior reconstruction debugging work in `moment_images`.

## Files And Code Paths

- Sweep orchestrator: `moment_images/parameter_sweep.py`
- Rectification orchestration and saved workspace fields: `moment_images/im_process.py`
- Bead-prior implementation: `moment_images/bead_prior_reconstruction.py`
- Sweep metric record construction: `moment_images/sweep_analysis.py`
- Visualization/diagnostic availability: `moment_images/sweep_visualization.py`

Important current code behavior:

- `parameter_sweep.py` saves bead-prior artifacts into each run's `workspace.npz` and `workspace.mat`.
- `pixel_rectified_views` in sweep workspaces is currently an alias of final `result.rectified_views` for `pixel` and `combined` strategies, not the pre-bead-prior pixel output.
- For bead-prior runs, compare `pre_bead_prior_rectified_views` against `rectified_views` to inspect the bead-prior effect.
- `bead_prior_residual_y/x` are one shared image-sized residual warp added to every view's existing pixelwise backward coordinate map:

```python
map_y = base_coordinates[:, :, i, j, 0] + residual_y
map_x = base_coordinates[:, :, i, j, 1] + residual_x
```

## Workspace Artifacts

Bead-prior-specific workspace artifacts:

- `pre_bead_prior_average_rectified_image`: average image immediately before bead-prior residual remapping.
- `pre_bead_prior_rectified_views`: rectified view stack before bead-prior residual remapping.
- `bead_prior_residual_y`: final accepted bead-prior residual Y field.
- `bead_prior_residual_x`: final accepted bead-prior residual X field.
- `bead_prior_region_labels`: integer connected-component labels for bead-support regions.
- `bead_prior_diagnostics_json`: JSON diagnostics from bead-prior detection, residual construction, and safety checks.

`bead_prior_region_labels` interpretation:

- `0` means background.
- Positive labels are connected bead-support regions from the detector.
- Region labels are not anchor IDs, confidence values, displacement magnitudes, or accepted-residual masks.
- Residual support can differ from `labels > 0`; inspect `hypot(bead_prior_residual_y, bead_prior_residual_x)` for the actual accepted residual field.

## Current Diagnostic Evidence

Earlier representative run:

```json
{
  "affected_pixel_fraction": 0.001775,
  "anchor_count": 41,
  "beadness_max": 1.0,
  "enabled": true,
  "flux_retry_count": 0,
  "foldover_retry_count": 1,
  "max_foldover_fraction": 0.24933010819657883,
  "max_residual": 0.6086527705192566,
  "mean_flux_ratio": 0.9993370134340036,
  "mean_foldover_fraction": 0.11705248821191332,
  "mean_residual": 0.25685206055641174,
  "multi_region_count": 9,
  "region_count": 20,
  "rejected_region_count": 19,
  "single_region_count": 6,
  "suspected_multi_region_count": 5
}
```

After sweeping lower values for existing exposed bead-prior parameters:

```json
{
  "affected_pixel_fraction": 0.0025375,
  "anchor_count": 32,
  "beadness_max": 1.0,
  "enabled": true,
  "flux_retry_count": 0,
  "foldover_retry_count": 1,
  "max_foldover_fraction": 0.20028191456762887,
  "max_residual": 0.04985220730304718,
  "mean_flux_ratio": 0.9999565440247754,
  "mean_foldover_fraction": 0.07302032800726276,
  "mean_residual": 0.018491340801119804,
  "multi_region_count": 4,
  "region_count": 21,
  "rejected_region_count": 17,
  "single_region_count": 10,
  "suspected_multi_region_count": 7
}
```

Key observation:

- Even with max residual around `0.05 px`, `17 / 21` regions are rejected and final affected pixels remain around `0.25%`.
- A residual this small is unlikely to create `20%` foldover by itself in normal coordinate fields.

## Current Diagnosis

The likely root cause is that the bead-prior foldover gate is measuring absolute foldover in:

```text
base_coordinates + residual
```

rather than foldover newly introduced by the bead-prior residual.

Since rejected regions are scaled to `0.0` and the final diagnostics still report high foldover, the pixelwise `base_coordinates` likely already contain foldover inside bead-labeled regions. The bead-prior stage is probably being rejected for inherited base-coordinate foldover, not for damage it introduced.

This means lower bead-prior strength/ROI parameters cannot solve the issue. The current safety gate is likely too restrictive in the wrong dimension.

## Current Exposed Sweep Knobs

Already exposed in `parameter_sweep.py`:

- `bead_prior_enabled`
- `bead_prior_roi_radii`
- `bead_prior_max_residuals`
- `bead_prior_single_strengths`
- `bead_prior_multi_strengths`
- `bead_prior_suspected_multi_strengths`
- `bead_prior_min_prominences`

Not currently exposed through sweep:

- `bead_prior_min_distance`
- `bead_prior_background_sigma`
- `bead_prior_saddle_drop`
- `bead_prior_max_flux_change_fraction`
- foldover threshold
- foldover retry scale
- per-region accepted/rejected foldover diagnostics

## Recommended Practical Next Steps

Do not start by simply relaxing the foldover threshold. First separate inherited foldover from bead-prior-induced foldover.

1. Add foldover diagnostics that compare base and candidate maps:

```text
base_foldover_fraction
candidate_foldover_fraction
added_foldover_fraction
worsened_foldover_fraction
```

Prefer per-region diagnostics as well:

```text
base_foldover_fraction_by_region
candidate_foldover_fraction_by_region
added_foldover_fraction_by_region
rejected_region_ids
accepted_region_ids
```

2. Change the foldover rejection rule from absolute final foldover to added/worsened foldover.

Current logic:

```text
Reject if final coordinate map has >2% foldover.
```

Recommended logic:

```text
Reject if bead-prior residual creates new foldover or meaningfully worsens existing foldover.
```

Keep absolute final foldover as a warning metric, not the primary rejection rule.

3. Promote the relevant diagnostics into sweep metrics in `sweep_analysis.py`, at least:

```text
bead_prior_rejected_region_count
bead_prior_foldover_retry_count
bead_prior_mean_foldover_fraction
bead_prior_max_foldover_fraction
bead_prior_base_mean_foldover_fraction
bead_prior_base_max_foldover_fraction
bead_prior_added_mean_foldover_fraction
bead_prior_added_max_foldover_fraction
```

4. Only after the above is visible in sweep results, consider exposing threshold knobs through `parameter_sweep.py`, such as:

```text
bead_prior_max_added_foldover_fraction
bead_prior_max_foldover_worsening_fraction
```

5. Add tests around the key distinction:

- If base coordinates already have foldover and bead residual is zero/tiny, do not reject solely due to inherited foldover.
- If bead residual introduces new foldover, reject or downscale.
- Diagnostics report both base and candidate foldover.

## Suggested Implementation Areas

Likely patch locations:

- `moment_images/bead_prior_reconstruction.py`
  - Add base-vs-candidate foldover helpers.
  - Add per-region diagnostics.
  - Change rejection logic to use added/worsened foldover.

- `moment_images/im_process.py`
  - Add any new `RectificationRunConfig` / `BeadPriorSettings` fields if thresholds become tunable.

- `moment_images/parameter_sweep.py`
  - Add new search-space fields only after diagnostics confirm which thresholds matter.

- `moment_images/sweep_analysis.py`
  - Promote new diagnostics to sweep metrics.

- Tests:
  - `moment_images/test_bead_prior_reconstruction.py`
  - `moment_images/test_parameter_sweep_experiment.py`
  - possibly `moment_images/test_sweep_metric_explorer.py` if new metrics are registered there.

## Current Recommendation

Escalate from parameter tuning to implementation logic revision, but do not discard bead-prior reconstruction entirely.

The detector and region labels appear useful. The problem appears concentrated in the acceptance/veto logic: the foldover safety gate likely attributes pre-existing pixelwise coordinate foldover to the bead-prior residual.

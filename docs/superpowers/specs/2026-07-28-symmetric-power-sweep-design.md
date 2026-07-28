# Symmetric power-field sweep

## Objective

Run one unattended two-stage combined-registration sweep across four datasets. Select one robust, shared configuration for symmetric intensity powering and capped power-field blending. A small later validation phase is out of this sweep budget.

## Fixed registration settings

- Combined strategy with global-average reference.
- `correlation_pad_width=54`, `global_threshold=0.1`, `refinement_window_size=19`.
- `max_displacement=8`, `disp_gauss_sigma=0`, `pixel_threshold=0.05`, `sub_pix_factor=1`, `pixel_sigma_pyramid=false`, and `zero_fill_roll=true`.
- Reference and shifted moving image use the same signed intensity power: `pixel_shift_power_apply_to_reference=true`.

## Search design

The per-dataset budget is 132, for at most 528 registration runs over four datasets.

Stage 1 exhaustively evaluates 32 settings per dataset:

- Gaussian sigma: 10 and 20.
- Direct symmetric power: 1, 1.5, 2, and 2.5.
- Blended field candidates: raised power 1.5, 2, and 2.5; blend alpha 0.25 and 0.5; vector residual cap 0.5 and 1 pixel.

For each sigma this is four direct-power candidates plus twelve blend candidates, giving `2 * (4 + 3 * 2 * 2) = 32` candidates.

Stage 2 receives the remaining 99 candidates per dataset. It uses `auto_fine` and `half_step_bounded` refinement around the global robust top four stage-1 candidates. This deliberately selects shared settings across all datasets; it does not attempt per-dataset optimization.

## Deliberate exclusions

- No `p=3`: prior results showed increasing displacement-boundary saturation with no consistent inter-view consistency gain.
- No asymmetric power control: the current JSON schema cross-products `pixel_shift_power_apply_to_reference`, which would also create unwanted asymmetric blend candidates. The autonomous one-file sweep therefore tests only the new symmetric formulation.
- No sweep of thresholds, displacement limit, windows, pyramid mode, or displacement smoothing.

## Evaluation

Use robust aggregation over all four datasets. Image sharpness metrics are useful but must not solely determine the conclusion; after the run, inspect inter-view consistency, displacement-boundary saturation, blend residual/cap diagnostics, and the small independent validation phase.

## Dataset placeholders

The executable specification contains four placeholder entries (`dataset_1` through `dataset_4`) with `views` paths to replace before execution.

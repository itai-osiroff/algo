# Sweep Metric Explorer Guide

This guide explains how to use `moment_images/sweep_metric_explorer.py` to re-rank an existing sweep from the raw `workspace.npz` artifacts.

## What This Tool Does

`sweep_metric_explorer.py` is a post-processing helper for completed sweep outputs.

It does **not** run a new sweep. Instead, it:

- loads an existing sweep folder
- reads each run's `workspace.npz`
- recomputes metrics from the raw rectified output
- compares those new metrics against the original saved sweep ranking
- lets you label outputs as `top`, `good`, or `bad`
- evaluates whether a metric or metric combination ranks the visually better outputs higher

This is the right tool when you want to answer questions like:

- "Does Shannon entropy of normalized DCT actually help?"
- "Do Gini / Hoyer / Vollath F4 rank the images I prefer?"
- "Is there a better combination of new metrics and old sweep metrics?"

## Prerequisites

- A Python environment with the project dependencies installed.
- An existing sweep output folder created by `parameter_sweep.py`.
- Sweep runs that include `workspace.npz` files under the normal sweep layout, for example:

```text
path/to/sweep_output/
  stage1_summary.json
  run_state.json
  experiment_manifest.json
  runs/
    stage1/
      candidate_id/
        dataset_id/
          avg_rectified.png
          workspace.npz
```

The explorer works best on the modern spec-based sweep outputs. It reuses the same folder/stage/candidate structure as `sweep_visualization.py`.

## Launch

From the project root:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output
```

Optional config:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --metrics-config path/to/metrics_config.json
```

Headless export:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --export-dir path/to/report_folder
```

## What Gets Computed

The explorer recomputes metrics from `average_rectified_image` inside each `workspace.npz`.

It builds these internal representations:

- `avg_image`: normalized rectified average image
- `guide_log`: absolute LoG-style guide image derived from the rectified average
- normalized DCT coefficient maps for entropy-based metrics

### Old metrics kept for comparison

The explorer recomputes the existing sweep image metrics:

- `average_intensity`
- `maximum_intensity`
- `total_variation`
- `sum_squared_gradient`
- `laplacian_variance`
- `dwt_level2_hv_energy`

It also keeps the original saved sweep scores available as read-only baselines:

- `saved_score`
- `saved_mean_dataset_score`

### New metrics available

The current built-in new metrics are:

- `ndct_shannon_avg`
- `ndct_shannon_guide`
- `ndct_bayes_avg`
- `ndct_bayes_guide`
- `gini_avg`
- `gini_guide`
- `hoyer_avg`
- `hoyer_guide`
- `vollath_f4_avg`
- `vollath_f4_guide`

In practice, this lets you compare:

- direct-image vs guide-map versions
- old metrics vs new metrics
- mixed profiles that combine both

## Window Layout

The GUI has four tabs:

1. `Leaderboard`
2. `Comparison`
3. `Evaluation`
4. `Profiles / Search`

At the top of the window you also have:

- `Browse...`
- `Load`
- `Reload`
- `Export...`
- `Open Folder`

## Recommended Workflow

### 1. Load a sweep

Open the sweep output directory with:

- `Browse...` then `Load`
- or start the tool with the sweep path directly on the command line

After loading, the status bar shows how many raw metric rows were found.

### 2. Pick the stage and profile

In the `Leaderboard` tab choose:

- `Stage`
- `Profile`
- `Dataset`

The built-in profiles are:

- `saved_original`: the original sweep ranking as saved
- `old_recomputed`: old sweep metrics recomputed from `workspace.npz`
- `new_dct_sparse`: equal-weight blend of the new DCT / sparsity / Vollath metrics
- `hybrid_default`: equal-weight blend of old recomputed + new metrics

Use these first before building anything custom.

### 3. Inspect how the ranking changes

The leaderboard shows:

- current rank
- candidate id
- profile score
- rank delta relative to the saved sweep rank
- saved rank
- strategy
- current manual label for the selected dataset

Useful interpretation:

- a positive `Rank Δ` means the candidate moved upward relative to the saved ranking
- a negative `Rank Δ` means the candidate moved downward

### 4. Compare actual images

The `Comparison` tab is now a persistent `A / B` workspace rather than a single static preview figure.

The smoothest flow is:

1. Select one run in `Leaderboard` and click `Compare A`
2. Select a second run and click `Compare B`
3. Or select two rows and click `Compare 2`
4. Switch to `Comparison`

The comparison view shows two independent inspector panels:

- `Run A`
- `Run B`

Each panel has:

- a dedicated Matplotlib toolbar
- a `View` selector:
  - `preview`
  - `avg_image`
  - `guide_log`
- a `Contrast` selector:
  - `auto`
  - `p2-p98`
  - `p5-p95`
- `Fit`
- `1:1`
- `Reset`
- `Open Large`

This is the intended inspection workflow:

- use the Matplotlib toolbar for real zoom / pan / home / back behavior
- use `Fit` to jump back to the full image
- use `1:1` when you want a truer pixel-scale view of a region
- use `Open Large` to pop one side into a dedicated full-size viewer
- use `Contrast` when focus differences are subtle

The comparison-level controls are:

- `Set A`
- `Set B`
- `Use 2 Selected`
- `Swap`
- `Clear Slots`
- `Layout`
  - `Horizontal`
  - `Vertical`
- `Sync Zoom/Pan`
- `Lock Views`

Use `Sync Zoom/Pan` when you want the same region to stay aligned between both runs.

Use `Lock Views` when switching one panel from `preview` to `avg_image` or `guide_log` should automatically switch the other panel too.

The lower comparison table now shows metric-by-metric A/B deltas and which side each metric prefers.

Use this tab to sanity-check whether a metric is rewarding genuinely sharper images or just boosting contrast/noise artifacts.

### Comparing two runs directly

If you want a stable side-by-side comparison between two runs:

1. Select the first run in `Leaderboard`
2. Click `Compare A` or `Set A`
3. Select the second run
4. Click `Compare B` or `Set B`
5. Open `Comparison`

Now the comparison tab shows exactly those two runs together, even if the current leaderboard selection changes later.

When both runs are set, the metric table also includes deltas in `A - B` form plus a `Preferred` column, so you can quickly see which run each metric favors.

### 5. Add manual labels

Back in `Leaderboard`, select one or more rows and use:

- `Label Top`
- `Label Good`
- `Label Bad`
- `Clear Label`

Labels are **dataset-specific**, so the selected `Dataset` matters.

This is the core human-in-the-loop step: once you label a few examples, the tool can evaluate whether a profile agrees with your judgments.

### 6. Review evaluation results

Open the `Evaluation` tab.

It reports, for the selected stage:

- pairwise agreement
- top-1 hit rate
- mean label quality in top-`5`
- Spearman correlation
- number of labeled items

It also shows disagreement rows where a lower-quality label outranked a higher-quality label.

This tab is usually the fastest way to answer:

- "Is the new metric helping?"
- "Which profile agrees best with my visual judgment?"

### 7. Build custom combinations

Open `Profiles / Search`.

Select metrics from the metric list on the left, then either:

- click `Build Equal Profile`
- edit the JSON weights manually and click `Build JSON Profile`
- run `Search`

#### Build Equal Profile

This assigns equal weights to the selected metrics and creates a temporary custom profile in the current GUI session.

#### Build JSON Profile

Enter a JSON object such as:

```json
{
  "vollath_f4_guide": 0.4,
  "hoyer_guide": 0.2,
  "gini_avg": 0.2,
  "laplacian_variance": 0.2
}
```

Then provide a profile name and click `Build JSON Profile`.

This creates a temporary in-memory profile and immediately makes it available in the `Profile` dropdown.

Important current behavior:

- custom profiles built in the GUI are **not automatically saved** to disk
- if you want a profile to be repeatable across sessions, use a metrics config file

#### Search

Search explores nonnegative weight combinations over the selected metrics.

Controls:

- `Stage`
- `Weight Step`
- `Max Metrics`

Current constraints:

- the search expects a small metric subset
- more than 6 selected metrics is intentionally blocked
- the search only becomes useful once you have annotations

The search results panel lists the best combinations found, ordered by the evaluation criteria.

## Annotations and Cache

The explorer creates a helper-owned folder under the sweep root:

```text
metric_review/
```

Important files there:

- `annotations.json`
- `cache/metric_cache.json`

### `annotations.json`

This stores your manual labels:

- `top`
- `good`
- `bad`

It is safe to keep this file with the sweep output if you want reproducible comparisons.

### `cache/metric_cache.json`

This stores recomputed metric values keyed by:

- workspace path
- file size
- file modification time
- internal registry version

That means:

- the explorer loads faster the next time
- changing a `workspace.npz` causes recomputation automatically

## Exported Outputs

Using `Export...` in the GUI or `--export-dir` on the CLI writes a report folder containing:

- `candidate_metrics.csv`
- `profile_results.csv`
- `search_results.csv`
- `disagreements.csv`
- `report.json`
- `index.html`

What each file is for:

- `candidate_metrics.csv`: one row per candidate/dataset raw metric record
- `profile_results.csv`: final ranked results for each stage/profile
- `search_results.csv`: profile-search results from the current model
- `disagreements.csv`: cases where the ranking disagreed with your labels
- `report.json`: machine-readable summary of config, warnings, and evaluations
- `index.html`: lightweight static summary

## Metrics Config File

You can optionally provide a config file with:

- enabled metrics
- extra named profiles
- default profile
- search defaults

If `--metrics-config` is not provided, the tool will also look for:

```text
metric_review/metrics_config.json
```

### Example config

```json
{
  "enabled_metrics": [
    "saved_score",
    "laplacian_variance",
    "ndct_shannon_avg",
    "ndct_bayes_avg",
    "gini_guide",
    "hoyer_guide",
    "vollath_f4_guide"
  ],
  "default_profile": "guide_focus_mix",
  "profiles": [
    {
      "name": "guide_focus_mix",
      "label": "Guide Focus Mix",
      "description": "Guide-heavy mix of new and old metrics.",
      "metric_weights": {
        "vollath_f4_guide": 0.35,
        "hoyer_guide": 0.20,
        "gini_guide": 0.20,
        "ndct_shannon_avg": 0.10,
        "ndct_bayes_avg": 0.10,
        "laplacian_variance": 0.05
      }
    }
  ],
  "search_defaults": {
    "metric_subset": [
      "vollath_f4_guide",
      "hoyer_guide",
      "gini_guide",
      "laplacian_variance"
    ],
    "weight_step": 0.1,
    "max_metrics": 6
  }
}
```

Notes:

- any profile defined here is merged with the built-in profiles
- metrics not listed in `enabled_metrics` are hidden from ranking/search
- invalid metric names are ignored

## Interpreting the Results

A few practical rules help avoid false confidence:

- Start with `saved_original`, `old_recomputed`, `new_dct_sparse`, and `hybrid_default`.
- Do not trust a metric after looking at only one dataset.
- Use annotations across multiple datasets before relying on search results.
- Pay attention to the `Comparison` tab when a metric moves a candidate sharply upward.
- Use the disagreement table as the main debugging view when a profile looks promising numerically but wrong visually.

## Limitations and Current Behavior

Current implementation details worth knowing:

- The explorer is a post-analysis tool only; it does not modify or rerun the sweep.
- It expects `workspace.npz` to be present for raw metric recomputation.
- If a workspace is missing, the run can still appear via saved metadata, but raw metric coverage will be incomplete.
- Custom profiles built interactively in the GUI are session-only unless you save them into a metrics config file yourself.
- Search results are based on the currently loaded annotations and metric subset.
- The guide-map channel is an implementation choice for this explorer, not a guaranteed universally best representation.

## Troubleshooting

### The tool loads but metrics are missing

Likely causes:

- missing `workspace.npz`
- corrupt `workspace.npz`
- runs that only have previews / summaries but not raw workspaces

Check:

- `metric_review/cache/metric_cache.json`
- `report.json`
- warning text in the GUI

### The ranking does not change between profiles

Possible reasons:

- selected metrics are highly correlated on this sweep
- the active profile still heavily favors the old metrics
- the sweep does not have enough variation to separate candidates

Try:

- using the `new_dct_sparse` profile directly
- reducing the profile to 2-4 metrics
- labeling a few more examples and checking `Evaluation`

### Search gives unhelpful results

Possible reasons:

- too few annotations
- too many selected metrics
- weight step too coarse for the behavior you want

Try:

- label at least a few `top` and `bad` items per dataset
- search over a smaller metric subset
- compare the search winner against `hybrid_default` instead of trusting it blindly

## Minimal End-to-End Recipe

If you just want a practical starting point:

1. Run:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output
```

2. Compare `saved_original`, `new_dct_sparse`, and `hybrid_default`.
3. Select one dataset at a time.
4. Label a handful of obvious examples as `top` and `bad`.
5. Check `Evaluation`.
6. If `hybrid_default` still looks weak, go to `Profiles / Search` and search over:
   - `vollath_f4_guide`
   - `hoyer_guide`
   - `gini_guide`
   - `laplacian_variance`
7. Export the results to keep the comparison reproducible.

This is usually enough to tell whether the new metrics are genuinely improving the ranking or only changing it.

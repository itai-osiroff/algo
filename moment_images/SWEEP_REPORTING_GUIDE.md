# Sweep Reporting Guide

This guide covers tools that inspect, export, or rebuild reports from completed
parameter sweep outputs.

## Tools Covered

| Tool | Purpose |
| --- | --- |
| `sweep_visualization.py` | GUI and headless static report export for sweep leaderboards and candidate comparisons |
| `sweep_metric_explorer.py` | Raw `workspace.npz` metric recomputation, manual labels, and custom profile search |
| `generate_reports.py` | Rebuild summary, analysis, leaderboard, and alpha-report artifacts from sweep folders |

For the full raw metric explorer GUI workflow, see
[SWEEP_METRIC_EXPLORER_GUIDE.md](SWEEP_METRIC_EXPLORER_GUIDE.md).

## Sweep Visualizer

Open a sweep output directory:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output
```

The GUI focuses on:

- leaderboard filtering by stage and dataset
- candidate configuration inspection
- candidate-vs-candidate preview and diagnostic comparison
- metric diagnostics for individual runs

Export a static report without launching the GUI:

```bash
python -m moment_images.sweep_visualization path/to/sweep_output --export-dir path/to/report_folder
```

Exported outputs include:

- `index.html`
- `visualization_bundle.npz`
- `visualization_bundle.mat`
- root PNG summary plots
- `tables/candidate_leaderboard.csv`
- `tables/run_metric_records.csv`
- `data/report_context.json`
- `assets/` preview and diagnostic images

The visualizer supports modern generation-aware sweep roots and legacy summary
layouts where possible.

## Raw Metric Explorer

Open a completed sweep:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output
```

Use a metrics config:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --metrics-config path/to/metrics_config.json
```

Export a headless report:

```bash
python -m moment_images.sweep_metric_explorer path/to/sweep_output --export-dir path/to/report_folder
```

The explorer recomputes metrics from each run's `workspace.npz`, supports
manual `top` / `good` / `bad` labels, compares built-in and custom ranking
profiles, and can save reusable metric profiles under `metric_review/`.

Explorer-owned files include:

- `metric_review/annotations.json`
- `metric_review/cache/metric_cache.json`
- `metric_review/metrics_config.json`

Headless export writes:

- `candidate_metrics.csv`
- `profile_results.csv`
- `search_results.csv`
- `disagreements.csv`
- `report.json`
- `index.html`

## Report Regeneration

Use `generate_reports.py` when sweep products need to be rebuilt from existing
run folders:

```bash
python generate_reports.py path/to/sweep_output
```

Rebuild stage/final summaries and leaderboards from run folders:

```bash
python generate_reports.py path/to/sweep_output --spec path/to/spec.json --rebuild-summaries
```

Choose the alpha-report primary metric:

```bash
python generate_reports.py path/to/sweep_output --primary-metric laplacian_variance
```

Fail instead of tolerating missing/corrupt artifacts:

```bash
python generate_reports.py path/to/sweep_output --strict
```

`--spec` is required when rebuilding stage/final summaries because the tool
needs the experiment definition to reconstruct rankings consistently.

## Alpha Reports

Alpha reports summarize optional metric experiment results by formulation
family and alpha value. They are written when metric experiments and reporting
are enabled, or when regenerated from compatible outputs.

Common alpha report files:

- `alpha_report.json`
- `alpha_report.csv`
- `alpha_report_warnings.json`

Warnings are expected when reports are built from partial outputs. Review the
warnings file before treating rankings as complete.

## Partial And Corrupt Outputs

The reporting tools are designed to extract useful results from partial sweeps:

- missing summaries can sometimes be reconstructed from per-run
  `metric_record.json`
- missing previews may still allow table exports
- missing `workspace.npz` prevents raw metric recomputation for that run
- corrupt JSON/NPZ artifacts are recorded as warnings unless strict mode is used

For final lab reports, prefer complete sweep roots with valid `workspace.npz`,
`metric_record.json`, and preview artifacts for every selected run.

## Troubleshooting

- Visualizer shows no candidates: check that the selected directory is the sweep
  root, not a single run folder.
- Raw metric explorer has missing metrics: confirm `workspace.npz` exists under
  each run folder.
- Export report lacks images: confirm `avg_rectified.png` or equivalent preview
  artifacts exist.
- Summary rebuild fails: pass the matching `--spec`, and avoid rebuilding from
  incompatible copied run folders.
- GUI does not open: use the `--export-dir` headless path or run from a desktop
  session.


"""Utilities for recording and analyzing sweep metrics without cluttering the CLI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import cv2 as cv
import numpy as np

from moment_images.im_process import RectificationRunConfig, RectificationResult
from moment_images.rectification_strategy import (
    DEFAULT_DISPLACEMENT_METRIC_FAMILY,
    normalize_optional_displacement_metric_family,
)
from moment_images.reliable_io import (
    read_json_first_valid,
    write_csv_atomic,
    write_json_atomic,
)
from moment_images.metric_profiles import compute_extended_image_metrics

ImageMetrics = Dict[str, float]


def _haar_axis_decompose(values: np.ndarray, axis: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return low/high Haar components along an axis, trimming odd lengths."""

    size = values.shape[axis]
    even_size = size - (size % 2)
    if even_size < 2:
        return None, None

    slicer = [slice(None)] * values.ndim
    slicer[axis] = slice(0, even_size)
    trimmed = values[tuple(slicer)]

    even_idx = [slice(None)] * values.ndim
    odd_idx = [slice(None)] * values.ndim
    even_idx[axis] = slice(0, even_size, 2)
    odd_idx[axis] = slice(1, even_size, 2)

    scale = np.sqrt(2.0)
    low = (trimmed[tuple(even_idx)] + trimmed[tuple(odd_idx)]) / scale
    high = (trimmed[tuple(even_idx)] - trimmed[tuple(odd_idx)]) / scale
    return low, high


def _haar_dwt2_level(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute one 2D Haar decomposition level (LL, LH, HL, HH)."""

    low_cols, high_cols = _haar_axis_decompose(image, axis=1)
    if low_cols is None or high_cols is None:
        return None, None, None, None

    ll, lh = _haar_axis_decompose(low_cols, axis=0)
    hl, hh = _haar_axis_decompose(high_cols, axis=0)
    return ll, lh, hl, hh


def _haar_level2_hv_energy(image: np.ndarray) -> float:
    """Energy of level-2 Haar horizontal/vertical detail subbands."""

    ll1, _, _, _ = _haar_dwt2_level(image)
    if ll1 is None:
        return 0.0

    _, lh2, hl2, _ = _haar_dwt2_level(ll1)
    if lh2 is None or hl2 is None:
        return 0.0

    return float(np.sum(np.square(lh2)) + np.sum(np.square(hl2)))


def compute_image_metrics(image: np.ndarray) -> ImageMetrics:
    """Compute intensity and edge-focused statistics for a single image."""

    as_float = image.astype(np.float64)
    dx = np.diff(as_float, axis=1)
    dy = np.diff(as_float, axis=0)

    grad_x, grad_y = np.gradient(as_float)
    laplacian = cv.Laplacian(as_float, cv.CV_64F)

    return {
        "average_intensity": float(np.mean(as_float)),
        "maximum_intensity": float(np.max(as_float)),
        "total_variation": float(np.sum(np.abs(dx)) + np.sum(np.abs(dy))),
        "sum_squared_gradient": float(np.sum(np.square(grad_x)) + np.sum(np.square(grad_y))),
        "laplacian_variance": float(np.var(laplacian)),
        "dwt_level2_hv_energy": _haar_level2_hv_energy(as_float),
    }


def _score_map_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compute_displacement_field_metrics(result: RectificationResult) -> Optional[Dict[str, float]]:
    if result.pixel_similarity is None:
        return None

    disps_y = getattr(result.pixel_similarity, "disps_y", None)
    disps_x = getattr(result.pixel_similarity, "disps_x", None)
    if disps_y is None or disps_x is None:
        return None
    disp_y = np.asarray(disps_y, dtype=np.float64)
    disp_x = np.asarray(disps_x, dtype=np.float64)
    magnitudes = np.sqrt(np.square(disp_y) + np.square(disp_x))
    return {
        "mean_magnitude": float(np.mean(magnitudes)),
        "std_magnitude": float(np.std(magnitudes)),
        "p95_magnitude": float(np.percentile(magnitudes, 95.0)),
        "max_magnitude": float(np.max(magnitudes)),
        "outlier_rate_gt2px": float(np.mean(magnitudes > 2.0)),
    }


def _compute_inter_view_consistency_metrics(result: RectificationResult) -> Dict[str, float]:
    stack = np.asarray(result.rectified_views, dtype=np.float64).reshape((-1, *result.rectified_views.shape[2:]))
    per_pixel_std = np.std(stack, axis=0)
    return {
        "mean_view_std": float(np.mean(per_pixel_std)),
        "p95_view_std": float(np.percentile(per_pixel_std, 95.0)),
        "max_view_std": float(np.max(per_pixel_std)),
    }


def _extract_displacement_score_diagnostics(result: RectificationResult) -> Optional[Dict[str, object]]:
    similarity = result.pixel_similarity
    if similarity is None:
        return None

    diagnostics: Dict[str, object] = {
        "metric_family": (
            normalize_optional_displacement_metric_family(
                getattr(similarity, "metric_family", DEFAULT_DISPLACEMENT_METRIC_FAMILY),
                allow_unknown=True,
            )
            or DEFAULT_DISPLACEMENT_METRIC_FAMILY
        ),
        "alpha": float(getattr(similarity, "alpha", 1.0)),
        "component_weights_raw": getattr(similarity, "component_weights_raw", None),
        "component_weights_normalized": getattr(similarity, "component_weights_normalized", None),
    }
    blended = getattr(similarity, "blended_score_maps", None)
    if blended is not None:
        diagnostics["blended_score_stats"] = _score_map_summary(np.asarray(blended))
    component_maps = getattr(similarity, "component_score_maps", None)
    if isinstance(component_maps, dict):
        diagnostics["component_selected_stats"] = {
            key: _score_map_summary(np.asarray(value))
            for key, value in component_maps.items()
        }
    view_summary = getattr(similarity, "view_score_summary", None)
    if isinstance(view_summary, dict):
        diagnostics["view_score_summary"] = view_summary
    return diagnostics


def build_metric_record(
    run_name: str, config: RectificationRunConfig, result: RectificationResult
) -> Dict[str, object]:
    """Create a JSON-serializable metrics record for a sweep iteration."""

    record: Dict[str, object] = {
        "run": run_name,
        "config": asdict(config),
        "metric_formulation": {
            "id": config.metric_formulation_id,
            "family": config.displacement_metric_family,
            "alpha": float(config.displacement_alpha),
            "component_weights": (
                dict(config.displacement_component_weights)
                if config.displacement_component_weights is not None
                else None
            ),
        },
        "global_only_metrics": None,
        "global_and_local_metrics": None,
        "displacement_field_metrics": _compute_displacement_field_metrics(result),
        "inter_view_consistency_metrics": _compute_inter_view_consistency_metrics(result),
        "displacement_score_diagnostics": _extract_displacement_score_diagnostics(result),
        "runtime_warnings": list(result.runtime_warnings),
    }

    if result.global_avg_image is not None:
        record["global_only_metrics"] = compute_extended_image_metrics(result.global_avg_image)

    record["global_and_local_metrics"] = compute_extended_image_metrics(result.avg_image)
    return record


def _summarize_metric(values: List[Tuple[str, float]]) -> Dict[str, object]:
    run_values = [val for _, val in values]
    min_run, min_val = min(values, key=lambda item: item[1])
    max_run, max_val = max(values, key=lambda item: item[1])

    return {
        "min": {"run": min_run, "value": float(min_val)},
        "max": {"run": max_run, "value": float(max_val)},
        "mean": float(np.mean(run_values)),
    }


def analyze_summary_data(summary: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate metrics across sweep results for quick comparisons."""

    analysis: Dict[str, object] = {}
    scopes = {
        "global_only_metrics": "global_only",
        "global_and_local_metrics": "global_and_local",
    }

    for key, label in scopes.items():
        available = [(entry["run"], entry[key]) for entry in summary if entry.get(key)]
        if not available:
            continue

        metrics_summary: Dict[str, object] = {}
        metric_keys = list(available[0][1].keys())
        for metric_key in metric_keys:
            metric_values = [
                (run, metrics[metric_key]) for run, metrics in available if metrics is not None
            ]
            metrics_summary[metric_key] = _summarize_metric(metric_values)

        analysis[label] = metrics_summary

    return analysis


def save_analysis(analysis: Dict[str, object], path: Path) -> None:
    """Persist analysis output to disk if any metrics were collected."""

    if not analysis:
        return

    write_json_atomic(path, analysis, indent=2, sort_keys=False, create_backup=True)

    csv_rows: List[Dict[str, object]] = []
    for scope, metric_summaries in analysis.items():
        for metric, summary in cast(Dict[str, Dict[str, object]], metric_summaries).items():
            csv_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "min_run": cast(Dict[str, object], summary["min"])["run"],
                    "min_value": cast(Dict[str, object], summary["min"])["value"],
                    "max_run": cast(Dict[str, object], summary["max"])["run"],
                    "max_value": cast(Dict[str, object], summary["max"])["value"],
                    "mean": summary["mean"],
                }
            )
    write_csv_atomic(
        path.with_suffix(".csv"),
        ["scope", "metric", "min_run", "min_value", "max_run", "max_value", "mean"],
        csv_rows,
        create_backup=True,
    )


def _load_summary_records_with_fallback(
    output_dir: Path,
    tolerate_partial_outputs: bool,
) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    records: List[Dict[str, object]] = []

    summary_path = output_dir / "summary.json"
    try:
        payload, _, read_warnings = read_json_first_valid(
            summary_path,
            validator=lambda obj: isinstance(obj, list),
        )
        warnings.extend(read_warnings)
        records = [cast(Dict[str, object], item) for item in cast(List[object], payload) if isinstance(item, dict)]
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to read summary.json: {type(exc).__name__}: {exc}")
        if not tolerate_partial_outputs:
            raise

    if records:
        return records, warnings

    for stage_name in ("stage1", "stage2"):
        stage_path = output_dir / f"{stage_name}_summary.json"
        try:
            payload, _, read_warnings = read_json_first_valid(
                stage_path,
                validator=lambda obj: isinstance(obj, dict),
            )
            warnings.extend(read_warnings)
            runs = cast(List[object], cast(Dict[str, object], payload).get("runs", []))
            for run_payload in runs:
                if not isinstance(run_payload, dict):
                    continue
                if run_payload.get("status") != "ok":
                    continue
                metric_record = run_payload.get("metric_record")
                if isinstance(metric_record, dict):
                    records.append(cast(Dict[str, object], metric_record))
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"Failed to read {stage_name}_summary.json: {type(exc).__name__}: {exc}"
            )
            if not tolerate_partial_outputs:
                raise

    if records:
        return records, warnings

    metric_record_paths = sorted(output_dir.glob("runs/*/*/*/metric_record.json"))
    for record_path in metric_record_paths:
        try:
            payload, _, read_warnings = read_json_first_valid(
                record_path,
                validator=lambda obj: isinstance(obj, dict),
            )
            warnings.extend(read_warnings)
            records.append(cast(Dict[str, object], payload))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to read {record_path}: {type(exc).__name__}: {exc}")
            if not tolerate_partial_outputs:
                raise

    return records, warnings


def _resolve_ranking_metric(
    metrics: Dict[str, object],
    primary_metric: str,
    fallback_metrics: Sequence[str],
) -> Tuple[Optional[str], Optional[float]]:
    candidate_order = [primary_metric, *fallback_metrics]
    for metric_name in candidate_order:
        value = metrics.get(metric_name)
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            return metric_name, numeric
    return None, None


def _aggregate_alpha_rows(
    run_rows: Sequence[Dict[str, object]],
    primary_metric: str,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for row in run_rows:
        grouped.setdefault((str(row["family"]), float(row["alpha"])), []).append(dict(row))

    aggregated_rows: List[Dict[str, object]] = []
    for (family, alpha), rows in grouped.items():
        primary_values = [
            float(cast(Dict[str, object], row["global_and_local_metrics"]).get(primary_metric))
            for row in rows
            if cast(Dict[str, object], row["global_and_local_metrics"]).get(primary_metric) is not None
        ]
        disp_values = [
            float(cast(Dict[str, object], row.get("displacement_field_metrics", {})).get("mean_magnitude"))
            for row in rows
            if isinstance(row.get("displacement_field_metrics"), dict)
            and cast(Dict[str, object], row["displacement_field_metrics"]).get("mean_magnitude") is not None
        ]
        consistency_values = [
            float(cast(Dict[str, object], row.get("inter_view_consistency_metrics", {})).get("mean_view_std"))
            for row in rows
            if isinstance(row.get("inter_view_consistency_metrics"), dict)
            and cast(Dict[str, object], row["inter_view_consistency_metrics"]).get("mean_view_std") is not None
        ]
        aggregated_rows.append(
            {
                "family": family,
                "alpha": alpha,
                "count": len(rows),
                f"{primary_metric}_mean": float(np.mean(primary_values)) if primary_values else None,
                "mean_displacement_magnitude_mean": float(np.mean(disp_values)) if disp_values else None,
                "mean_view_std_mean": float(np.mean(consistency_values)) if consistency_values else None,
            }
        )
    aggregated_rows.sort(key=lambda item: (str(item["family"]), float(item["alpha"])))
    return aggregated_rows


def _save_alpha_report_plots(
    output_dir: Path,
    alpha_rows: Sequence[Dict[str, object]],
    primary_metric: str,
    warnings: List[str],
) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Plot generation unavailable: {type(exc).__name__}: {exc}")
        return

    per_family: Dict[str, List[Dict[str, object]]] = {}
    for row in alpha_rows:
        per_family.setdefault(str(row["family"]), []).append(dict(row))

    if not per_family:
        return

    metric_key = f"{primary_metric}_mean"
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    has_lines = False
    for family, rows in sorted(per_family.items(), key=lambda item: item[0]):
        xs = [float(item["alpha"]) for item in rows if item.get(metric_key) is not None]
        ys = [float(item[metric_key]) for item in rows if item.get(metric_key) is not None]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=family)
        has_lines = True
    ax.set_xlabel("alpha (local displacement score blend weight)")
    ax.set_ylabel(metric_key)
    ax.set_title("Primary Metric vs Alpha by Metric Family")
    ax.grid(True, linestyle=":", alpha=0.5)
    if has_lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_report_primary_vs_alpha.png", dpi=180)

    fig2 = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)
    has_lines = False
    for family, rows in sorted(per_family.items(), key=lambda item: item[0]):
        xs = [float(item["alpha"]) for item in rows if item.get("mean_displacement_magnitude_mean") is not None]
        ys = [
            float(item["mean_displacement_magnitude_mean"])
            for item in rows
            if item.get("mean_displacement_magnitude_mean") is not None
        ]
        if not xs:
            continue
        ax2.plot(xs, ys, marker="o", label=family)
        has_lines = True
    ax2.set_xlabel("alpha (local displacement score blend weight)")
    ax2.set_ylabel("mean displacement magnitude")
    ax2.set_title("Displacement Magnitude vs Alpha by Metric Family")
    ax2.grid(True, linestyle=":", alpha=0.5)
    if has_lines:
        ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "alpha_report_displacement_vs_alpha.png", dpi=180)


def generate_alpha_report(
    output_dir: Path,
    *,
    primary_metric: str = "laplacian_variance",
    tolerate_partial_outputs: bool = True,
) -> Dict[str, object]:
    """Build alpha/family-centric report artifacts for final or partial sweeps."""

    output_dir = Path(output_dir)
    records, warnings = _load_summary_records_with_fallback(
        output_dir=output_dir,
        tolerate_partial_outputs=tolerate_partial_outputs,
    )

    fallback_metrics = [
        "sum_squared_gradient",
        "dwt_level2_hv_energy",
        "total_variation",
        "maximum_intensity",
        "average_intensity",
    ]
    run_rows: List[Dict[str, object]] = []
    for record in records:
        metrics = cast(Dict[str, object], record.get("global_and_local_metrics", {}))
        if not metrics:
            warnings.append(f"Run '{record.get('run', '<unknown>')}' missing global_and_local_metrics.")
            continue

        metric_name, ranking_value = _resolve_ranking_metric(metrics, primary_metric, fallback_metrics)
        if metric_name is None or ranking_value is None:
            warnings.append(f"Run '{record.get('run', '<unknown>')}' has no usable ranking metric.")
            continue

        formulation = cast(Dict[str, object], record.get("metric_formulation", {}))
        config = cast(Dict[str, object], record.get("config", {}))
        family = (
            normalize_optional_displacement_metric_family(
                formulation.get("family"),
                allow_unknown=True,
            )
            or normalize_optional_displacement_metric_family(
                config.get("displacement_metric_family"),
                allow_unknown=True,
            )
            or DEFAULT_DISPLACEMENT_METRIC_FAMILY
        )
        alpha = float(
            formulation.get("alpha")
            if formulation.get("alpha") is not None
            else config.get("displacement_alpha", 1.0)
        )
        formulation_id = str(
            formulation.get("id")
            or config.get("metric_formulation_id")
            or "intensity_default"
        )
        run_rows.append(
            {
                "run": str(record.get("run", "")),
                "formulation_id": formulation_id,
                "family": family,
                "alpha": alpha,
                "ranking_metric_name": metric_name,
                "ranking_metric_value": float(ranking_value),
                "global_and_local_metrics": metrics,
                "displacement_field_metrics": record.get("displacement_field_metrics"),
                "inter_view_consistency_metrics": record.get("inter_view_consistency_metrics"),
                "displacement_score_diagnostics": record.get("displacement_score_diagnostics"),
            }
        )

    run_rows.sort(key=lambda item: float(item["ranking_metric_value"]), reverse=True)
    for rank, row in enumerate(run_rows, start=1):
        row["rank"] = rank

    alpha_rows = _aggregate_alpha_rows(run_rows, primary_metric=primary_metric)

    report_payload: Dict[str, object] = {
        "generated_at": _utc_now_iso(),
        "primary_metric": primary_metric,
        "fallback_metrics": fallback_metrics,
        "num_runs_considered": len(run_rows),
        "num_records_loaded": len(records),
        "warnings": warnings,
        "ranked_runs": run_rows,
        "alpha_family_summary": alpha_rows,
    }
    write_json_atomic(
        output_dir / "alpha_report.json",
        report_payload,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    csv_rows: List[Dict[str, object]] = []
    for row in run_rows:
        metrics = cast(Dict[str, object], row["global_and_local_metrics"])
        disp_metrics = cast(Optional[Dict[str, object]], row.get("displacement_field_metrics"))
        consistency = cast(Optional[Dict[str, object]], row.get("inter_view_consistency_metrics"))
        csv_rows.append(
            {
                "rank": row.get("rank"),
                "run": row.get("run"),
                "formulation_id": row.get("formulation_id"),
                "family": row.get("family"),
                "alpha": row.get("alpha"),
                "ranking_metric_name": row.get("ranking_metric_name"),
                "ranking_metric_value": row.get("ranking_metric_value"),
                primary_metric: metrics.get(primary_metric),
                "sum_squared_gradient": metrics.get("sum_squared_gradient"),
                "dwt_level2_hv_energy": metrics.get("dwt_level2_hv_energy"),
                "mean_displacement_magnitude": None if disp_metrics is None else disp_metrics.get("mean_magnitude"),
                "mean_view_std": None if consistency is None else consistency.get("mean_view_std"),
            }
        )
    write_csv_atomic(
        output_dir / "alpha_report.csv",
        [
            "rank",
            "run",
            "formulation_id",
            "family",
            "alpha",
            "ranking_metric_name",
            "ranking_metric_value",
            primary_metric,
            "sum_squared_gradient",
            "dwt_level2_hv_energy",
            "mean_displacement_magnitude",
            "mean_view_std",
        ],
        csv_rows,
        create_backup=True,
    )
    write_json_atomic(
        output_dir / "alpha_report_warnings.json",
        {"warnings": warnings},
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    _save_alpha_report_plots(
        output_dir=output_dir,
        alpha_rows=alpha_rows,
        primary_metric=primary_metric,
        warnings=warnings,
    )
    return report_payload


def _utc_now_iso() -> str:
    return cast(str, np.datetime64("now").astype(str)) + "Z"

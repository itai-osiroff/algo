"""Utilities for recording and analyzing sweep metrics without cluttering the CLI."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import numpy as np

from moment_images.im_process import RectificationRunConfig, RectificationResult

ImageMetrics = Dict[str, float]


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
        "laplacian_energy": float(np.sum(np.square(laplacian))),
    }


def build_metric_record(
    run_name: str, config: RectificationRunConfig, result: RectificationResult
) -> Dict[str, object]:
    """Create a JSON-serializable metrics record for a sweep iteration."""

    record: Dict[str, object] = {
        "run": run_name,
        "config": asdict(config),
        "global_only_metrics": None,
        "global_and_local_metrics": None,
    }

    if result.global_avg_image is not None:
        record["global_only_metrics"] = compute_image_metrics(result.global_avg_image)

    record["global_and_local_metrics"] = compute_image_metrics(result.avg_image)

    return record


def _summarize_metric(values: List[tuple[str, float]]) -> Dict[str, object]:
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
    """Persist analysis output to disk if any metrics were collected.

    The JSON file is retained for lossless, structured interchange while a CSV
    companion is produced for a quick, human-readable overview.
    """

    if not analysis:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        json.dump(analysis, fp, indent=2)

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scope",
                "metric",
                "min_run",
                "min_value",
                "max_run",
                "max_value",
                "mean",
            ],
        )
        writer.writeheader()
        for scope, metric_summaries in analysis.items():
            for metric, summary in metric_summaries.items():
                writer.writerow(
                    {
                        "scope": scope,
                        "metric": metric,
                        "min_run": summary["min"]["run"],
                        "min_value": summary["min"]["value"],
                        "max_run": summary["max"]["run"],
                        "max_value": summary["max"]["value"],
                        "mean": summary["mean"],
                    }
                )

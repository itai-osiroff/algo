import csv

import numpy as np
import pytest

from moment_images.im_process import RectificationRunConfig, RectificationResult
from moment_images.sweep_analysis import (
    analyze_summary_data,
    build_metric_record,
    compute_image_metrics,
    save_analysis,
)


def _make_metrics(base: float) -> dict:
    return {
        "average_intensity": base,
        "maximum_intensity": base + 1,
        "total_variation": base + 2,
        "sum_squared_gradient": base + 3,
        "laplacian_variance": base + 4,
        "dwt_level2_hv_energy": base + 5,
    }


def test_build_metric_record_handles_global_and_local_metrics():
    config = RectificationRunConfig(strategy="combined")
    result = RectificationResult(
        avg_image=np.ones((2, 2), dtype=float),
        rectified_views=np.ones((1, 1, 2, 2), dtype=float),
        similarity_data=None,
        config=config,
        ref_view_index=(0, 0),
        input_views=None,
        global_similarity=None,
        global_rectified_views=None,
        global_avg_image=np.full((2, 2), 2.0, dtype=float),
        pixel_similarity=None,
    )

    record = build_metric_record("run1", config, result)

    assert pytest.approx(record["global_only_metrics"]["average_intensity"]) == 2.0
    assert pytest.approx(record["global_and_local_metrics"]["average_intensity"]) == 1.0


def test_analyze_summary_data_reports_extrema_and_means():
    summary = [
        {
            "run": "run1",
            "global_only_metrics": _make_metrics(1.0),
            "global_and_local_metrics": _make_metrics(2.0),
        },
        {
            "run": "run2",
            "global_only_metrics": _make_metrics(3.0),
            "global_and_local_metrics": _make_metrics(4.0),
        },
    ]

    analysis = analyze_summary_data(summary)

    assert analysis["global_only"]["average_intensity"]["min"]["run"] == "run1"
    assert analysis["global_only"]["average_intensity"]["max"]["run"] == "run2"
    assert pytest.approx(analysis["global_and_local"]["maximum_intensity"]["mean"]) == 4.0
    assert analysis["global_and_local"]["dwt_level2_hv_energy"]["max"]["run"] == "run2"
    assert pytest.approx(analysis["global_and_local"]["dwt_level2_hv_energy"]["mean"]) == 8.0


def test_save_analysis_emits_json_and_csv(tmp_path):
    summary = [
        {
            "run": "run1",
            "global_only_metrics": _make_metrics(1.0),
            "global_and_local_metrics": _make_metrics(2.0),
        },
        {
            "run": "run2",
            "global_only_metrics": _make_metrics(3.0),
            "global_and_local_metrics": _make_metrics(4.0),
        },
    ]

    analysis = analyze_summary_data(summary)
    save_path = tmp_path / "analysis.json"

    save_analysis(analysis, save_path)

    csv_path = save_path.with_suffix(".csv")
    assert save_path.exists()
    assert csv_path.exists()

    csv_content = csv_path.read_text(encoding="utf-8")
    assert "scope,metric,min_run,min_value,max_run,max_value,mean" in csv_content
    assert "global_and_local,average_intensity,run1,2.0,run2,4.0,3.0" in csv_content


def test_compute_image_metrics_reports_level2_dwt_metric():
    image = np.arange(64, dtype=np.float64).reshape(8, 8)

    metrics = compute_image_metrics(image)

    assert "dwt_level2_hv_energy" in metrics
    assert metrics["dwt_level2_hv_energy"] > 0.0

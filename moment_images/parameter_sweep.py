"""Parameter sweep runner for ``im_process`` rectification routines.

This module drives ``rectify_views_from_config`` with a configurable grid of
parameters (window size, maximum displacement, Gaussian smoothing, etc.) and
persists the results in an easy-to-compare folder layout. It is intended to run
offline so practitioners can explore trade-offs without stepping through the
interactive ``main`` workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from moment_images.im_process import (
    RectificationRunConfig,
    RectificationResult,
    generate_rectification_figures,
    min_max_normalize,
    rectify_views_from_config,
    save_figures,
    save_workspace_variables,
    extract_views,
    VIEW_DISTANCE_SCALE_FACTOR,
)
from moment_images.reliable_io import (
    FirstValidReadError,
    read_json_first_valid,
    read_npy_first_valid,
    read_npz_first_valid,
    write_csv_atomic,
    write_json_atomic,
    write_npy_atomic,
)
from moment_images.sweep_analysis import (
    analyze_summary_data,
    build_metric_record,
    generate_alpha_report,
    save_analysis,
)


MEMORY_SAFETY_FRACTION = 0.70
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class LiveMetricsPlotter:
    """Render sweep metrics as they are recorded for at-a-glance comparisons."""

    def __init__(self, metric_keys: Sequence[str]):
        self.metric_keys = list(metric_keys)
        self.scopes = [
            ("global_only_metrics", "Global only"),
            ("global_and_local_metrics", "Global + local"),
        ]
        self.run_labels: list[str] = []
        self.data: Dict[str, Dict[str, List[float]]] = {
            scope: {metric: [] for metric in self.metric_keys} for scope, _ in self.scopes
        }

        plt.ion()
        self.fig, axes = plt.subplots(
            len(self.metric_keys), 1, figsize=(10, 3 * len(self.metric_keys)), sharex=True
        )
        self.axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        self.fig.suptitle("Live sweep metrics", fontsize=14)

    def update(self, run_name: str, record: Dict[str, object]) -> None:
        self.run_labels.append(run_name)

        for scope, _ in self.scopes:
            metrics = record.get(scope)
            for metric in self.metric_keys:
                value = float(metrics[metric]) if isinstance(metrics, dict) and metric in metrics else np.nan
                self.data[scope][metric].append(value)

        self._redraw()

    def _redraw(self) -> None:
        x_values = range(len(self.run_labels))

        for axis, metric in zip(self.axes, self.metric_keys):
            axis.clear()
            for scope, label in self.scopes:
                values = self.data[scope][metric]
                if not values or np.all(np.isnan(values)):
                    continue
                axis.plot(x_values, values, marker="o", label=label)
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.legend()

        self.axes[-1].set_xticks(list(x_values))
        self.axes[-1].set_xticklabels(self.run_labels, rotation=45, ha="right")
        self.axes[-1].set_xlabel("Run")

        self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def _build_int_range(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    return list(range(start, stop + 1, step))


def _build_float_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    values: List[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "laplacian_variance": 0.25,
    "dwt_level2_hv_energy": 0.25,
    "sum_squared_gradient": 0.25,
    "total_variation": 0.25,
}


@dataclass
class SweepDatasetSpec:
    """Specification for one dataset participating in a sweep experiment."""

    dataset_id: str
    views_path: Optional[str] = None
    source_image: Optional[str] = None
    source_image_dir: Optional[str] = None
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    view_radius: Optional[int] = None
    view_grid_size: int = 3
    view_distance: Optional[int] = None
    hdr_enabled: Optional[bool] = None
    hdr_exposure_times: Optional[List[float]] = None
    normalize_to_8bit: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SweepDatasetSpec":
        dataset_id = str(payload.get("id") or payload.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError("Each dataset entry must provide a non-empty 'id'.")

        exposure_times = payload.get("hdr_exposure_times")
        parsed_exposure_times: Optional[List[float]] = None
        if exposure_times is not None:
            parsed_exposure_times = [float(val) for val in exposure_times]  # type: ignore[arg-type]

        hdr_enabled_raw = payload.get("hdr_enabled")
        hdr_enabled = None if hdr_enabled_raw is None else bool(hdr_enabled_raw)

        return cls(
            dataset_id=dataset_id,
            views_path=cast(Optional[str], payload.get("views") or payload.get("views_path")),
            source_image=cast(Optional[str], payload.get("source_image")),
            source_image_dir=cast(Optional[str], payload.get("source_image_dir")),
            center_x=cast(Optional[int], payload.get("center_x")),
            center_y=cast(Optional[int], payload.get("center_y")),
            view_radius=cast(Optional[int], payload.get("view_radius")),
            view_grid_size=int(payload.get("view_grid_size", 3)),
            view_distance=cast(Optional[int], payload.get("view_distance")),
            hdr_enabled=hdr_enabled,
            hdr_exposure_times=parsed_exposure_times,
            normalize_to_8bit=bool(payload.get("normalize_to_8bit", False)),
        )


@dataclass
class SweepSearchSpace:
    """Strategy-specific parameter search space."""

    strategy: str
    similarity_window_sizes: List[int] = field(default_factory=lambda: [15, 25, 35, 45])
    refinement_window_sizes: List[int] = field(default_factory=lambda: [15, 25, 35, 45])
    max_displacements: List[int] = field(default_factory=lambda: [5, 15, 25])
    gauss_sigmas: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
    disp_gauss_sigmas: List[float] = field(default_factory=lambda: [0.0, 1.0])
    global_thresholds: List[float] = field(default_factory=lambda: [0.15])
    pixel_thresholds: List[float] = field(default_factory=lambda: [0.05])
    sub_pix_factors: List[int] = field(default_factory=lambda: [1, 2])
    pixel_sigma_pyramid: List[bool] = field(default_factory=lambda: [False, True])
    zero_fill_roll: List[bool] = field(default_factory=lambda: [True])
    correlation_pad_widths: List[int] = field(default_factory=lambda: [RectificationRunConfig().correlation_pad_width])
    combined_use_global_avg_reference: List[bool] = field(default_factory=lambda: [True])
    gauss_sizes: Optional[List[int]] = None
    disp_gauss_sizes: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, strategy: str, payload: Optional[Dict[str, object]]) -> "SweepSearchSpace":
        raw = payload or {}
        return cls(
            strategy=strategy,
            similarity_window_sizes=[int(v) for v in raw.get("similarity_window_sizes", raw.get("window_sizes", [15, 25, 35, 45]))],  # type: ignore[arg-type]
            refinement_window_sizes=[int(v) for v in raw.get("refinement_window_sizes", raw.get("window_sizes", [15, 25, 35, 45]))],  # type: ignore[arg-type]
            max_displacements=[int(v) for v in raw.get("max_displacements", [5, 15, 25])],  # type: ignore[arg-type]
            gauss_sigmas=[float(v) for v in raw.get("gauss_sigmas", [3.0, 5.0, 7.0])],  # type: ignore[arg-type]
            disp_gauss_sigmas=[float(v) for v in raw.get("disp_gauss_sigmas", [0.0, 1.0])],  # type: ignore[arg-type]
            global_thresholds=[float(v) for v in raw.get("global_thresholds", [0.15])],  # type: ignore[arg-type]
            pixel_thresholds=[float(v) for v in raw.get("pixel_thresholds", [0.05])],  # type: ignore[arg-type]
            sub_pix_factors=[int(v) for v in raw.get("sub_pix_factors", [1, 2])],  # type: ignore[arg-type]
            pixel_sigma_pyramid=[bool(v) for v in raw.get("pixel_sigma_pyramid", [False, True])],  # type: ignore[arg-type]
            zero_fill_roll=[bool(v) for v in raw.get("zero_fill_roll", [True])],  # type: ignore[arg-type]
            correlation_pad_widths=[int(v) for v in raw.get("correlation_pad_widths", [RectificationRunConfig().correlation_pad_width])],  # type: ignore[arg-type]
            combined_use_global_avg_reference=[bool(v) for v in raw.get("combined_use_global_avg_reference", [True])],  # type: ignore[arg-type]
            gauss_sizes=None
            if raw.get("gauss_sizes") is None
            else [int(v) for v in raw.get("gauss_sizes", [])],  # type: ignore[arg-type]
            disp_gauss_sizes=None
            if raw.get("disp_gauss_sizes") is None
            else [int(v) for v in raw.get("disp_gauss_sizes", [])],  # type: ignore[arg-type]
        )


@dataclass
class SweepStageSpec:
    """Controls two-stage coarse/fine search behavior."""

    total_budget_per_dataset: int = 320
    stage1_fraction: float = 0.6
    top_k: int = 8
    max_workers: Optional[int] = None


@dataclass
class SweepScoringSpec:
    """Controls composite metric scoring and robustness aggregation."""

    metric_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    runtime_penalty_weight: float = 0.0
    robustness_lambda: float = 0.5
    missing_dataset_penalty: float = 1.0
    metric_scope: str = "global_and_local_metrics"


@dataclass
class MetricFormulationSpec:
    """Displacement metric family definition used as a sweep dimension."""

    formulation_id: str
    family: str = "ncc_gradient"
    alphas: List[float] = field(default_factory=lambda: [0.5])
    component_weights: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object], default_idx: int) -> "MetricFormulationSpec":
        raw_id = str(payload.get("id") or payload.get("formulation_id") or "").strip()
        formulation_id = raw_id or f"formulation_{default_idx:02d}"

        family = str(payload.get("family", "ncc_gradient")).strip().lower()
        if family not in {"ncc_gradient", "ncc_laplacian", "composite"}:
            raise ValueError(
                "metric_formulations[].family must be one of "
                "'ncc_gradient', 'ncc_laplacian', or 'composite'."
            )

        alphas_raw = cast(Optional[List[object]], payload.get("alphas"))
        if alphas_raw is None:
            alpha_single = payload.get("alpha")
            if alpha_single is None:
                alphas = [0.5]
            else:
                alphas = [float(alpha_single)]
        else:
            if not alphas_raw:
                raise ValueError("metric_formulations[].alphas cannot be empty.")
            alphas = [float(item) for item in alphas_raw]
        for alpha in alphas:
            if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
                raise ValueError("metric_formulations alpha values must be finite and in [0, 1].")

        raw_weights_payload = cast(Optional[Dict[str, object]], payload.get("component_weights"))
        parsed_weights: Optional[Dict[str, float]] = None
        if raw_weights_payload is not None:
            parsed_weights = {}
            for key, raw_value in raw_weights_payload.items():
                numeric = float(raw_value)
                if not np.isfinite(numeric):
                    raise ValueError("component_weights values must be finite.")
                if numeric < 0.0:
                    raise ValueError("component_weights values must be non-negative.")
                parsed_weights[str(key)] = numeric

        return cls(
            formulation_id=formulation_id,
            family=family,
            alphas=alphas,
            component_weights=parsed_weights,
        )


@dataclass
class SweepReportingSpec:
    """Controls alpha/family report generation after the experiment."""

    primary_metric: str = "laplacian_variance"
    tolerate_partial_outputs: bool = True
    generate_alpha_report: bool = True


def _clone_metric_formulation(spec: MetricFormulationSpec) -> MetricFormulationSpec:
    return MetricFormulationSpec(
        formulation_id=str(spec.formulation_id),
        family=str(spec.family),
        alphas=[float(alpha) for alpha in spec.alphas],
        component_weights=(
            None if spec.component_weights is None else dict(spec.component_weights)
        ),
    )


def _parse_metric_formulations_payload(
    formulations_raw: Optional[List[Dict[str, object]]],
    *,
    field_name: str,
) -> List[MetricFormulationSpec]:
    if formulations_raw is None:
        return []
    if not isinstance(formulations_raw, list):
        raise ValueError(f"{field_name} must be a list.")
    for idx, item in enumerate(formulations_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be an object.")
    formulations = [
        MetricFormulationSpec.from_dict(cast(Dict[str, object], item), idx + 1)
        for idx, item in enumerate(formulations_raw)
    ]
    formulation_ids = [item.formulation_id for item in formulations]
    if len(formulation_ids) != len(set(formulation_ids)):
        raise ValueError(f"{field_name} IDs must be unique.")
    return formulations


@dataclass
class ExperimentalMetricSweepSpec:
    """Optional second-phase sweep over displacement metric formulations."""

    enabled: bool = False
    shortlist_top_k: int = 8
    formulations: List[MetricFormulationSpec] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls, payload: Dict[str, object], *, default_shortlist_top_k: int
    ) -> "ExperimentalMetricSweepSpec":
        enabled = bool(payload.get("enabled", False))
        shortlist_raw = payload.get("shortlist_top_k")
        shortlist_top_k = int(
            default_shortlist_top_k if shortlist_raw is None else shortlist_raw
        )
        if shortlist_top_k < 1:
            raise ValueError(
                "experimental_metric_sweep.shortlist_top_k must be >= 1."
            )
        formulations = _parse_metric_formulations_payload(
            cast(Optional[List[Dict[str, object]]], payload.get("formulations")),
            field_name="experimental_metric_sweep.formulations",
        )
        if enabled and not formulations:
            raise ValueError(
                "experimental_metric_sweep.formulations cannot be empty when enabled."
            )
        return cls(
            enabled=enabled,
            shortlist_top_k=shortlist_top_k,
            formulations=formulations,
        )


@dataclass
class SweepExperimentSpec:
    """Top-level experiment definition loaded from ``--spec`` JSON."""

    datasets: List[SweepDatasetSpec]
    search_spaces: Dict[str, SweepSearchSpace]
    strategies: List[str] = field(default_factory=lambda: ["pixel", "combined"])
    stages: SweepStageSpec = field(default_factory=SweepStageSpec)
    scoring: SweepScoringSpec = field(default_factory=SweepScoringSpec)
    experimental_metric_sweep: Optional[ExperimentalMetricSweepSpec] = None
    metric_formulations: Optional[List[MetricFormulationSpec]] = None
    reporting: SweepReportingSpec = field(default_factory=SweepReportingSpec)
    output_dir: Optional[str] = None
    save_figures: bool = False
    analyze: bool = True
    keep_all_run_artifacts: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        stage_top_k = max(1, int(self.stages.top_k))

        experimental = self.experimental_metric_sweep
        if experimental is None:
            experimental = ExperimentalMetricSweepSpec(
                enabled=False,
                shortlist_top_k=stage_top_k,
                formulations=[],
            )

        legacy_formulations = (
            [_clone_metric_formulation(item) for item in self.metric_formulations]
            if self.metric_formulations is not None
            else []
        )
        if legacy_formulations and not experimental.formulations:
            experimental = ExperimentalMetricSweepSpec(
                enabled=True,
                shortlist_top_k=experimental.shortlist_top_k,
                formulations=legacy_formulations,
            )

        if int(experimental.shortlist_top_k) < 1:
            raise ValueError(
                "experimental_metric_sweep.shortlist_top_k must be >= 1."
            )
        if experimental.enabled and not experimental.formulations:
            raise ValueError(
                "experimental_metric_sweep requires at least one formulation when enabled."
            )
        if experimental.enabled and not any(
            strategy in {"pixel", "combined"} for strategy in self.strategies
        ):
            raise ValueError(
                "experimental_metric_sweep requires at least one pixel/combined strategy."
            )

        cloned_formulations = [
            _clone_metric_formulation(item) for item in experimental.formulations
        ]
        formulation_ids = [item.formulation_id for item in cloned_formulations]
        if len(formulation_ids) != len(set(formulation_ids)):
            raise ValueError(
                "experimental_metric_sweep.formulations IDs must be unique."
            )

        self.experimental_metric_sweep = ExperimentalMetricSweepSpec(
            enabled=bool(experimental.enabled),
            shortlist_top_k=int(experimental.shortlist_top_k),
            formulations=cloned_formulations,
        )
        self.metric_formulations = [
            _clone_metric_formulation(item)
            for item in self.experimental_metric_sweep.formulations
        ]

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SweepExperimentSpec":
        datasets_raw = cast(List[Dict[str, object]], payload.get("datasets", []))
        if not datasets_raw:
            raise ValueError("spec must provide at least one dataset entry under 'datasets'.")
        datasets = [SweepDatasetSpec.from_dict(item) for item in datasets_raw]

        strategies = [str(item).lower() for item in cast(List[object], payload.get("strategies", ["pixel", "combined"]))]
        if not strategies:
            strategies = ["pixel", "combined"]
        valid_strategies = {"pixel", "combined", "global"}
        invalid = [strategy for strategy in strategies if strategy not in valid_strategies]
        if invalid:
            raise ValueError(f"Unsupported strategies in spec: {invalid}")

        search_spaces_raw = cast(Dict[str, Dict[str, object]], payload.get("search_spaces", {}))
        search_spaces: Dict[str, SweepSearchSpace] = {}
        for strategy in strategies:
            search_spaces[strategy] = SweepSearchSpace.from_dict(
                strategy=strategy,
                payload=search_spaces_raw.get(strategy),
            )

        stages_raw = cast(Dict[str, object], payload.get("stages", {}))
        stages = SweepStageSpec(
            total_budget_per_dataset=int(stages_raw.get("total_budget_per_dataset", 320)),
            stage1_fraction=float(stages_raw.get("stage1_fraction", 0.6)),
            top_k=int(stages_raw.get("top_k", 8)),
            max_workers=cast(Optional[int], stages_raw.get("max_workers")),
        )

        scoring_raw = cast(Dict[str, object], payload.get("scoring", {}))
        metric_weights = cast(Optional[Dict[str, float]], scoring_raw.get("metric_weights"))
        scoring = SweepScoringSpec(
            metric_weights=dict(metric_weights) if metric_weights else dict(DEFAULT_SCORE_WEIGHTS),
            runtime_penalty_weight=float(scoring_raw.get("runtime_penalty_weight", 0.0)),
            robustness_lambda=float(scoring_raw.get("robustness_lambda", 0.5)),
            missing_dataset_penalty=float(scoring_raw.get("missing_dataset_penalty", 1.0)),
            metric_scope=str(scoring_raw.get("metric_scope", "global_and_local_metrics")),
        )

        has_legacy_formulations = "metric_formulations" in payload
        legacy_formulations = (
            _parse_metric_formulations_payload(
                cast(Optional[List[Dict[str, object]]], payload.get("metric_formulations")),
                field_name="metric_formulations",
            )
            if has_legacy_formulations
            else None
        )

        experimental_raw = cast(
            Optional[Dict[str, object]], payload.get("experimental_metric_sweep")
        )
        if experimental_raw is not None:
            if not isinstance(experimental_raw, dict):
                raise ValueError("experimental_metric_sweep must be an object.")
            experimental_metric_sweep = ExperimentalMetricSweepSpec.from_dict(
                experimental_raw,
                default_shortlist_top_k=stages.top_k,
            )
        elif legacy_formulations is not None:
            experimental_metric_sweep = ExperimentalMetricSweepSpec(
                enabled=bool(legacy_formulations),
                shortlist_top_k=stages.top_k,
                formulations=legacy_formulations,
            )
        else:
            experimental_metric_sweep = ExperimentalMetricSweepSpec(
                enabled=False,
                shortlist_top_k=stages.top_k,
                formulations=[],
            )

        reporting_raw = cast(Dict[str, object], payload.get("reporting", {}))
        reporting = SweepReportingSpec(
            primary_metric=str(reporting_raw.get("primary_metric", "laplacian_variance")),
            tolerate_partial_outputs=bool(reporting_raw.get("tolerate_partial_outputs", True)),
            generate_alpha_report=bool(reporting_raw.get("generate_alpha_report", True)),
        )

        return cls(
            datasets=datasets,
            search_spaces=search_spaces,
            strategies=strategies,
            stages=stages,
            scoring=scoring,
            experimental_metric_sweep=experimental_metric_sweep,
            metric_formulations=legacy_formulations,
            reporting=reporting,
            output_dir=cast(Optional[str], payload.get("output_dir")),
            save_figures=bool(payload.get("save_figures", False)),
            analyze=bool(payload.get("analyze", True)),
            keep_all_run_artifacts=bool(payload.get("keep_all_run_artifacts", True)),
            random_seed=int(payload.get("random_seed", 0)),
        )


@dataclass
class PreparedDataset:
    """Runtime-ready dataset after views have been validated/generated once."""

    dataset_id: str
    views_path: Path
    hdr_enabled: bool
    hdr_exposure_times: Optional[List[float]]
    normalize_to_8bit: bool
    views_shape: Tuple[int, ...]
    views_dtype: str
    views_itemsize: int
    resolved_views_path: Optional[str] = None
    views_file_size_bytes: Optional[int] = None
    views_file_mtime_ns: Optional[int] = None


@dataclass
class StageEvaluationResult:
    """Result payload for one sweep stage including execution diagnostics."""

    stage_results: List[Dict[str, object]]
    execution: Dict[str, object]


@dataclass
class SweepCandidate:
    """One strategy-specific rectification candidate."""

    strategy: str
    params: Dict[str, object]
    config: RectificationRunConfig
    candidate_id: str = ""


def _normalize_window_size(window: int) -> int:
    if window < 1:
        raise ValueError("window size must be positive")
    if window % 2 == 0:
        return window + 1
    return window


def _derive_gauss_size(gauss_sigma: float) -> int:
    size = max(1, int(round(gauss_sigma * 5)))
    return size


def _strategy_param_keys(strategy: str) -> List[str]:
    if strategy == "pixel":
        return [
            "similarity_window_size",
            "max_displacement",
            "gauss_sigma",
            "disp_gauss_sigma",
            "pixel_threshold",
            "sub_pix_factor",
            "pixel_sigma_pyramid",
            "zero_fill_roll",
        ]
    if strategy == "combined":
        return [
            "correlation_pad_width",
            "global_threshold",
            "refinement_window_size",
            "max_displacement",
            "gauss_sigma",
            "disp_gauss_sigma",
            "pixel_threshold",
            "sub_pix_factor",
            "pixel_sigma_pyramid",
            "combined_use_global_avg_reference",
            "zero_fill_roll",
        ]
    if strategy == "global":
        return [
            "correlation_pad_width",
            "global_threshold",
        ]
    raise ValueError(f"Unsupported strategy: {strategy}")


def _candidate_signature(candidate: SweepCandidate) -> str:
    return json.dumps(asdict(candidate.config), sort_keys=True)


def _limit_candidates(candidates: Sequence[SweepCandidate], budget: Optional[int]) -> List[SweepCandidate]:
    if budget is None:
        return list(candidates)
    if budget <= 0:
        return []
    if len(candidates) <= budget:
        return list(candidates)
    raw = np.linspace(0, len(candidates) - 1, budget)
    indices = sorted({int(round(val)) for val in raw})
    if not indices:
        return []
    while len(indices) < budget:
        for idx in range(len(candidates)):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= budget:
                break
    indices = sorted(indices[:budget])
    return [candidates[idx] for idx in indices]


def _assign_candidate_ids(candidates: Sequence[SweepCandidate], prefix: str) -> List[SweepCandidate]:
    assigned: List[SweepCandidate] = []
    for idx, candidate in enumerate(candidates):
        clone = SweepCandidate(
            strategy=candidate.strategy,
            params=dict(candidate.params),
            config=candidate.config,
            candidate_id=f"{prefix}{idx:04d}",
        )
        assigned.append(clone)
    return assigned


def _metric_formulation_from_params(
    params: Dict[str, object],
) -> Tuple[str, float, Optional[Dict[str, float]], str]:
    family = str(params.get("displacement_metric_family", "ncc_gradient")).strip().lower()
    alpha = float(params.get("displacement_alpha", 0.5))
    raw_weights_obj = params.get("displacement_component_weights")
    parsed_weights: Optional[Dict[str, float]] = None
    if raw_weights_obj is not None:
        parsed_weights = {
            str(key): float(value)
            for key, value in cast(Dict[str, object], raw_weights_obj).items()
        }
    formulation_id = str(params.get("metric_formulation_id", "default_ncc_gradient_a0p5")).strip()
    if not formulation_id:
        formulation_id = "default_ncc_gradient_a0p5"
    return family, alpha, parsed_weights, formulation_id


def _expand_candidates_with_metric_formulations(
    candidates: Sequence[SweepCandidate],
    metric_formulations: Sequence[MetricFormulationSpec],
) -> List[SweepCandidate]:
    expanded: List[SweepCandidate] = []
    for candidate in candidates:
        for formulation in metric_formulations:
            for alpha in formulation.alphas:
                params = dict(candidate.params)
                params["displacement_metric_family"] = formulation.family
                params["displacement_alpha"] = float(alpha)
                params["metric_formulation_id"] = formulation.formulation_id
                params["displacement_component_weights"] = (
                    None
                    if formulation.component_weights is None
                    else dict(formulation.component_weights)
                )
                expanded.append(_build_candidate_from_params(candidate.strategy, params))
    expanded.sort(key=lambda item: json.dumps(item.params, sort_keys=True))
    return expanded


def _dedupe_candidates(candidates: Sequence[SweepCandidate]) -> List[SweepCandidate]:
    deduped: List[SweepCandidate] = []
    seen = set()
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def _baseline_metric_formulations() -> List[MetricFormulationSpec]:
    return [
        MetricFormulationSpec(
            formulation_id="baseline_ncc_gradient_a0p5",
            family="ncc_gradient",
            alphas=[0.5],
            component_weights=None,
        )
    ]


def _build_candidate_from_params(strategy: str, params: Dict[str, object]) -> SweepCandidate:
    defaults = RectificationRunConfig()
    (
        displacement_metric_family,
        displacement_alpha,
        displacement_component_weights,
        metric_formulation_id,
    ) = _metric_formulation_from_params(params)
    if strategy == "pixel":
        window = _normalize_window_size(int(params["similarity_window_size"]))
        gauss_sigma = float(params["gauss_sigma"])
        disp_gauss_sigma = float(params["disp_gauss_sigma"])
        gauss_size = int(params.get("gauss_size", _derive_gauss_size(gauss_sigma)))
        disp_gauss_size = int(params.get("disp_gauss_size", window))
        config = RectificationRunConfig(
            strategy="pixel",
            similarity_window_size=window,
            refinement_window_size=window,
            max_displacement=int(params["max_displacement"]),
            gauss_sigma=gauss_sigma,
            gauss_size=gauss_size,
            disp_gauss_sigma=disp_gauss_sigma,
            disp_gauss_size=disp_gauss_size,
            global_threshold=defaults.global_threshold,
            pixel_threshold=float(params["pixel_threshold"]),
            pixel_sigma_pyramid=bool(params["pixel_sigma_pyramid"]),
            zero_fill_roll=bool(params["zero_fill_roll"]),
            sub_pix_factor=int(params["sub_pix_factor"]),
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        normalized_params = dict(params)
        normalized_params["similarity_window_size"] = window
        normalized_params["gauss_size"] = gauss_size
        normalized_params["disp_gauss_size"] = disp_gauss_size
        return SweepCandidate(strategy="pixel", params=normalized_params, config=config)

    if strategy == "combined":
        window = _normalize_window_size(int(params["refinement_window_size"]))
        gauss_sigma = float(params["gauss_sigma"])
        disp_gauss_sigma = float(params["disp_gauss_sigma"])
        gauss_size = int(params.get("gauss_size", _derive_gauss_size(gauss_sigma)))
        disp_gauss_size = int(params.get("disp_gauss_size", window))
        config = RectificationRunConfig(
            strategy="combined",
            similarity_window_size=defaults.similarity_window_size,
            refinement_window_size=window,
            correlation_pad_width=int(params["correlation_pad_width"]),
            max_displacement=int(params["max_displacement"]),
            gauss_sigma=gauss_sigma,
            gauss_size=gauss_size,
            disp_gauss_sigma=disp_gauss_sigma,
            disp_gauss_size=disp_gauss_size,
            global_threshold=float(params["global_threshold"]),
            pixel_threshold=float(params["pixel_threshold"]),
            pixel_sigma_pyramid=bool(params["pixel_sigma_pyramid"]),
            zero_fill_roll=bool(params["zero_fill_roll"]),
            sub_pix_factor=int(params["sub_pix_factor"]),
            combined_use_global_avg_reference=bool(params["combined_use_global_avg_reference"]),
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        normalized_params = dict(params)
        normalized_params["refinement_window_size"] = window
        normalized_params["gauss_size"] = gauss_size
        normalized_params["disp_gauss_size"] = disp_gauss_size
        return SweepCandidate(strategy="combined", params=normalized_params, config=config)

    if strategy == "global":
        config = RectificationRunConfig(
            strategy="global",
            correlation_pad_width=int(params["correlation_pad_width"]),
            global_threshold=float(params["global_threshold"]),
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        return SweepCandidate(strategy="global", params=dict(params), config=config)

    raise ValueError(f"Unsupported strategy: {strategy}")


def _coarse_param_grid(search_space: SweepSearchSpace) -> List[Dict[str, object]]:
    strategy = search_space.strategy.lower()
    if strategy == "pixel":
        values = product(
            search_space.similarity_window_sizes,
            search_space.max_displacements,
            search_space.gauss_sigmas,
            search_space.disp_gauss_sigmas,
            search_space.pixel_thresholds,
            search_space.sub_pix_factors,
            search_space.pixel_sigma_pyramid,
            search_space.zero_fill_roll,
        )
        return [
            {
                "similarity_window_size": int(win),
                "max_displacement": int(max_disp),
                "gauss_sigma": float(gauss_sigma),
                "disp_gauss_sigma": float(disp_gauss_sigma),
                "pixel_threshold": float(pixel_threshold),
                "sub_pix_factor": int(sub_pix_factor),
                "pixel_sigma_pyramid": bool(pixel_sigma_pyramid),
                "zero_fill_roll": bool(zero_fill_roll),
            }
            for (
                win,
                max_disp,
                gauss_sigma,
                disp_gauss_sigma,
                pixel_threshold,
                sub_pix_factor,
                pixel_sigma_pyramid,
                zero_fill_roll,
            ) in values
        ]

    if strategy == "combined":
        values = product(
            search_space.correlation_pad_widths,
            search_space.global_thresholds,
            search_space.refinement_window_sizes,
            search_space.max_displacements,
            search_space.gauss_sigmas,
            search_space.disp_gauss_sigmas,
            search_space.pixel_thresholds,
            search_space.sub_pix_factors,
            search_space.pixel_sigma_pyramid,
            search_space.combined_use_global_avg_reference,
            search_space.zero_fill_roll,
        )
        return [
            {
                "correlation_pad_width": int(correlation_pad_width),
                "global_threshold": float(global_threshold),
                "refinement_window_size": int(refinement_window_size),
                "max_displacement": int(max_disp),
                "gauss_sigma": float(gauss_sigma),
                "disp_gauss_sigma": float(disp_gauss_sigma),
                "pixel_threshold": float(pixel_threshold),
                "sub_pix_factor": int(sub_pix_factor),
                "pixel_sigma_pyramid": bool(pixel_sigma_pyramid),
                "combined_use_global_avg_reference": bool(combined_use_global_avg_reference),
                "zero_fill_roll": bool(zero_fill_roll),
            }
            for (
                correlation_pad_width,
                global_threshold,
                refinement_window_size,
                max_disp,
                gauss_sigma,
                disp_gauss_sigma,
                pixel_threshold,
                sub_pix_factor,
                pixel_sigma_pyramid,
                combined_use_global_avg_reference,
                zero_fill_roll,
            ) in values
        ]

    if strategy == "global":
        values = product(
            search_space.correlation_pad_widths,
            search_space.global_thresholds,
        )
        return [
            {
                "correlation_pad_width": int(correlation_pad_width),
                "global_threshold": float(global_threshold),
            }
            for correlation_pad_width, global_threshold in values
        ]

    raise ValueError(f"Unsupported strategy: {strategy}")


def generate_coarse_candidates(
    search_space: SweepSearchSpace, budget: Optional[int] = None
) -> List[SweepCandidate]:
    """Generate deterministic coarse candidates for one strategy."""

    params = _coarse_param_grid(search_space)
    params.sort(key=lambda item: json.dumps(item, sort_keys=True))
    candidates = [_build_candidate_from_params(search_space.strategy.lower(), item) for item in params]
    deduped: List[SweepCandidate] = []
    seen = set()
    for candidate in candidates:
        sig = _candidate_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    return _limit_candidates(deduped, budget)


def _neighbor_values(values: Sequence[object], current: object) -> List[object]:
    if not values:
        return [current]
    normalized = list(values)
    if current not in normalized:
        return [current]
    idx = normalized.index(current)
    indices = sorted({max(0, idx - 1), idx, min(len(normalized) - 1, idx + 1)})
    return [normalized[i] for i in indices]


def generate_refined_candidates(
    ranked_candidates: Sequence[Dict[str, object]],
    candidate_lookup: Dict[str, SweepCandidate],
    search_spaces: Dict[str, SweepSearchSpace],
    top_k: int = 8,
    budget: Optional[int] = None,
) -> List[SweepCandidate]:
    """Generate neighborhood refinements around Top-K coarse candidates."""

    top_items = [item for item in ranked_candidates if item.get("candidate_id") in candidate_lookup][:top_k]
    generated: List[SweepCandidate] = []
    seen = set()

    for item in top_items:
        candidate_id = cast(str, item.get("candidate_id"))
        coarse = candidate_lookup[candidate_id]
        strategy = coarse.strategy
        search_space = search_spaces[strategy]
        params = dict(coarse.params)

        base_candidate = _build_candidate_from_params(strategy, params)
        base_sig = _candidate_signature(base_candidate)
        if base_sig not in seen:
            seen.add(base_sig)
            generated.append(base_candidate)

        for key in _strategy_param_keys(strategy):
            if key not in params:
                continue
            values_attr = f"{key}s"
            strategy_values: Sequence[object]
            if key == "similarity_window_size":
                strategy_values = list(search_space.similarity_window_sizes)
            elif key == "refinement_window_size":
                strategy_values = list(search_space.refinement_window_sizes)
            elif key == "max_displacement":
                strategy_values = list(search_space.max_displacements)
            elif key == "gauss_sigma":
                strategy_values = list(search_space.gauss_sigmas)
            elif key == "disp_gauss_sigma":
                strategy_values = list(search_space.disp_gauss_sigmas)
            elif key == "pixel_threshold":
                strategy_values = list(search_space.pixel_thresholds)
            elif key == "sub_pix_factor":
                strategy_values = list(search_space.sub_pix_factors)
            elif key == "pixel_sigma_pyramid":
                strategy_values = list(search_space.pixel_sigma_pyramid)
            elif key == "zero_fill_roll":
                strategy_values = list(search_space.zero_fill_roll)
            elif key == "correlation_pad_width":
                strategy_values = list(search_space.correlation_pad_widths)
            elif key == "global_threshold":
                strategy_values = list(search_space.global_thresholds)
            elif key == "combined_use_global_avg_reference":
                strategy_values = list(search_space.combined_use_global_avg_reference)
            else:
                strategy_values = cast(Sequence[object], getattr(search_space, values_attr, []))

            for neighbor in _neighbor_values(strategy_values, params[key]):
                if neighbor == params[key]:
                    continue
                variant = dict(params)
                variant[key] = neighbor
                candidate = _build_candidate_from_params(strategy, variant)
                sig = _candidate_signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                generated.append(candidate)

    generated.sort(key=lambda item: json.dumps(item.params, sort_keys=True))
    return _limit_candidates(generated, budget)


def _compose_run_config(
    candidate: SweepCandidate,
    dataset: PreparedDataset,
) -> RectificationRunConfig:
    config_dict = asdict(candidate.config)
    config_dict["hdr_enabled"] = dataset.hdr_enabled
    config_dict["hdr_exposure_times"] = dataset.hdr_exposure_times
    config_dict["normalize_to_8bit"] = dataset.normalize_to_8bit
    return RectificationRunConfig(**config_dict)


def _stage_budgets(stage_spec: SweepStageSpec) -> Tuple[int, int]:
    total = max(1, int(stage_spec.total_budget_per_dataset))
    stage1 = int(round(total * float(stage_spec.stage1_fraction)))
    stage1 = max(1, min(stage1, total))
    stage2 = max(1, total - stage1)
    return stage1, stage2


def _parse_experiment_spec(path: Path) -> SweepExperimentSpec:
    if not path.exists() and not Path(f"{path}.bak").exists():
        raise FileNotFoundError(f"Sweep spec file not found: {path}")
    payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a JSON object.")
    return SweepExperimentSpec.from_dict(payload)


def _save_run_artifacts(
    output_dir: Path,
    run_name: str,
    result: RectificationResult,
    metric_record: Optional[Dict[str, object]] = None,
) -> None:
    config = result.config
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    avg_img_path = run_dir / "avg_rectified.png"
    avg_preview = np.asarray(min_max_normalize(result.avg_image), dtype=np.uint8)
    Image.fromarray(avg_preview).save(avg_img_path)

    write_npy_atomic(
        run_dir / "rectified_views.npy",
        result.rectified_views,
        allow_pickle=False,
        create_backup=True,
    )

    pixel_displacements_x = None
    pixel_displacements_y = None
    pixel_blended_scores = None
    pixel_component_scores: Dict[str, np.ndarray] = {}
    if result.pixel_similarity is not None:
        pixel_displacements_x = getattr(result.pixel_similarity, "disps_x", None)
        pixel_displacements_y = getattr(result.pixel_similarity, "disps_y", None)
        pixel_blended_scores = getattr(result.pixel_similarity, "blended_score_maps", None)
        component_scores = getattr(result.pixel_similarity, "component_score_maps", None)
        if isinstance(component_scores, dict):
            for key, value in component_scores.items():
                pixel_component_scores[str(key)] = np.asarray(value)

    config_dict = asdict(config)
    strategy_code = {"global": 1, "pixel": 2, "combined": 3}.get(config.strategy.lower(), 0)
    workspace_data: Dict[str, object] = {
        "rectified_views": result.rectified_views,
        "average_rectified_image": result.avg_image,
        "global_rectification_matrix": result.global_similarity,
        "global_rectified_views": result.global_rectified_views,
        "average_rectified_global_image": result.global_avg_image,
        "pixel_rectified_views": (
            result.rectified_views if config.strategy.lower() in {"pixel", "combined"} else None
        ),
        "pixel_displacements_x": pixel_displacements_x,
        "pixel_displacements_y": pixel_displacements_y,
        "pixel_blended_scores": pixel_blended_scores,
        "reference_view_index": np.array(result.ref_view_index),
        "rectification_strategy": np.array(strategy_code),
        "hdr_enabled": np.array(config.hdr_enabled),
        "hdr_exposure_times": (
            np.asarray(config.hdr_exposure_times, dtype=np.float64)
            if config.hdr_exposure_times is not None
            else None
        ),
        "normalize_to_8bit": np.array(config.normalize_to_8bit),
        "run_name": np.array(run_name),
        "config_json": np.array(json.dumps(config_dict, sort_keys=True)),
        "displacement_metric_family": np.array(config.displacement_metric_family),
        "displacement_alpha": np.array(config.displacement_alpha, dtype=np.float64),
        "metric_formulation_id": np.array(config.metric_formulation_id),
    }
    for component_name, component_values in pixel_component_scores.items():
        workspace_data[f"pixel_component_score_{component_name}"] = component_values
    save_workspace_variables(str(run_dir / "workspace.npz"), workspace_data)

    write_json_atomic(run_dir / "config.json", config_dict, indent=2, sort_keys=True, create_backup=True)
    if metric_record is not None:
        write_json_atomic(
            run_dir / "metric_record.json",
            metric_record,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )


def _persist_figures(figures: Dict[str, List[Figure]], base_dir: Path) -> None:
    for section, fig_list in figures.items():
        section_dir = base_dir / section
        save_figures(fig_list, str(section_dir))
        for fig in fig_list:
            plt.close(fig)


def _load_views(path: Path) -> np.ndarray:
    try:
        views, _, _ = read_npy_first_valid(path, allow_pickle=False)
    except FirstValidReadError as exc:
        raise FileNotFoundError(f"Views file not found or unreadable: {path}") from exc
    if views.ndim == 4:
        if views.shape[0] != views.shape[1]:
            raise ValueError("views must have shape (N, N, H, W)")
        return views
    if views.ndim == 5:
        if views.shape[1] != views.shape[2]:
            raise ValueError("views must have shape (E, N, N, H, W)")
        return views
    raise ValueError("views must have shape (N, N, H, W) or (E, N, N, H, W)")


def _generate_views(
    image_path: Path,
    center_x: int,
    center_y: int,
    radius: int,
    grid_size: int,
    view_distance: int | None,
    cache_dir: Path,
    normalize_to_8bit: bool = False,
) -> tuple[np.ndarray, Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image = Image.open(image_path)
    if normalize_to_8bit:
        image_np = np.asarray(min_max_normalize(image))
    else:
        image_np = np.asarray(image).copy()

    distance = view_distance
    if distance is None:
        distance = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * radius))

    base_circle = np.array([center_x, center_y, radius])
    views = extract_views(image_np, base_circle, grid_size, distance)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "generated_views.npy"
    write_npy_atomic(cache_path, views, allow_pickle=False, create_backup=True)
    return views, cache_path


def _generate_exposure_views(
    image_dir: Path,
    center_x: int,
    center_y: int,
    radius: int,
    grid_size: int,
    view_distance: int | None,
    cache_dir: Path,
    normalize_to_8bit: bool = False,
) -> tuple[np.ndarray, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Source image directory is not a folder: {image_dir}")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    distance = view_distance
    if distance is None:
        distance = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * radius))

    base_circle = np.array([center_x, center_y, radius])
    exposure_views: list[np.ndarray] = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if normalize_to_8bit:
            image_np = np.asarray(min_max_normalize(image))
        else:
            image_np = np.asarray(image).copy()
        views = extract_views(image_np, base_circle, grid_size, distance)
        exposure_views.append(views)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "generated_exposure_views.npy"
    exposure_views_arr = np.asarray(exposure_views)
    write_npy_atomic(cache_path, exposure_views_arr, allow_pickle=False, create_backup=True)
    return exposure_views_arr, cache_path


def _prepare_dataset(dataset: SweepDatasetSpec, output_dir: Path) -> PreparedDataset:
    dataset_dir = output_dir / "datasets" / dataset.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_sources = [
        dataset.views_path is not None,
        dataset.source_image is not None,
        dataset.source_image_dir is not None,
    ]
    if sum(data_sources) != 1:
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' must provide exactly one source: "
            "'views_path', 'source_image', or 'source_image_dir'."
        )

    if dataset.views_path is not None:
        views_path = Path(dataset.views_path)
        views_arr = _load_views(views_path)
        inferred_hdr = views_arr.ndim == 5
        hdr_enabled = inferred_hdr if dataset.hdr_enabled is None else dataset.hdr_enabled
        if hdr_enabled and views_arr.ndim != 5:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' has non-HDR views but hdr_enabled=True."
            )
        if not hdr_enabled and views_arr.ndim == 5:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' has exposure stack views; enable HDR for this dataset."
            )
        if dataset.hdr_exposure_times is not None and hdr_enabled and len(dataset.hdr_exposure_times) != views_arr.shape[0]:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' exposure time count does not match exposure axis."
            )
        stat = views_path.stat()
        return PreparedDataset(
            dataset_id=dataset.dataset_id,
            views_path=views_path,
            hdr_enabled=bool(hdr_enabled),
            hdr_exposure_times=dataset.hdr_exposure_times,
            normalize_to_8bit=dataset.normalize_to_8bit,
            views_shape=tuple(int(dim) for dim in views_arr.shape),
            views_dtype=str(views_arr.dtype),
            views_itemsize=int(views_arr.dtype.itemsize),
            resolved_views_path=str(views_path.resolve()),
            views_file_size_bytes=int(stat.st_size),
            views_file_mtime_ns=int(stat.st_mtime_ns),
        )

    if dataset.source_image is not None:
        missing = [
            name
            for name, present in [
                ("center_x", dataset.center_x),
                ("center_y", dataset.center_y),
                ("view_radius", dataset.view_radius),
            ]
            if present is None
        ]
        if missing:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' source_image requires fields: {', '.join(missing)}"
            )
        views_arr, views_path = _generate_views(
            image_path=Path(dataset.source_image),
            center_x=int(dataset.center_x),
            center_y=int(dataset.center_y),
            radius=int(dataset.view_radius),
            grid_size=int(dataset.view_grid_size),
            view_distance=dataset.view_distance,
            cache_dir=dataset_dir,
            normalize_to_8bit=dataset.normalize_to_8bit,
        )
        if dataset.hdr_enabled:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' uses a single source image and cannot enable HDR."
            )
        stat = views_path.stat()
        return PreparedDataset(
            dataset_id=dataset.dataset_id,
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=dataset.normalize_to_8bit,
            views_shape=tuple(int(dim) for dim in views_arr.shape),
            views_dtype=str(views_arr.dtype),
            views_itemsize=int(views_arr.dtype.itemsize),
            resolved_views_path=str(views_path.resolve()),
            views_file_size_bytes=int(stat.st_size),
            views_file_mtime_ns=int(stat.st_mtime_ns),
        )

    # source_image_dir branch
    missing = [
        name
        for name, present in [
            ("center_x", dataset.center_x),
            ("center_y", dataset.center_y),
            ("view_radius", dataset.view_radius),
        ]
        if present is None
    ]
    if missing:
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' source_image_dir requires fields: {', '.join(missing)}"
        )
    views_arr, views_path = _generate_exposure_views(
        image_dir=Path(cast(str, dataset.source_image_dir)),
        center_x=int(dataset.center_x),
        center_y=int(dataset.center_y),
        radius=int(dataset.view_radius),
        grid_size=int(dataset.view_grid_size),
        view_distance=dataset.view_distance,
        cache_dir=dataset_dir,
        normalize_to_8bit=dataset.normalize_to_8bit,
    )
    hdr_enabled = True if dataset.hdr_enabled is None else dataset.hdr_enabled
    if not hdr_enabled:
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' uses source_image_dir but HDR is disabled."
        )
    if dataset.hdr_exposure_times is not None and len(dataset.hdr_exposure_times) != views_arr.shape[0]:
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' exposure time count does not match exposure stack."
        )
    stat = views_path.stat()
    return PreparedDataset(
        dataset_id=dataset.dataset_id,
        views_path=views_path,
        hdr_enabled=True,
        hdr_exposure_times=dataset.hdr_exposure_times,
        normalize_to_8bit=dataset.normalize_to_8bit,
        views_shape=tuple(int(dim) for dim in views_arr.shape),
        views_dtype=str(views_arr.dtype),
        views_itemsize=int(views_arr.dtype.itemsize),
        resolved_views_path=str(views_path.resolve()),
        views_file_size_bytes=int(stat.st_size),
        views_file_mtime_ns=int(stat.st_mtime_ns),
    )


def _prepare_datasets(spec: SweepExperimentSpec, output_dir: Path) -> List[PreparedDataset]:
    seen: set[str] = set()
    prepared: List[PreparedDataset] = []
    for dataset in spec.datasets:
        if dataset.dataset_id in seen:
            raise ValueError(f"Duplicate dataset id '{dataset.dataset_id}' in experiment spec.")
        seen.add(dataset.dataset_id)
        prepared.append(_prepare_dataset(dataset, output_dir=output_dir))
    prepared.sort(key=lambda item: item.dataset_id)
    return prepared


RUN_STATE_FILENAME = "run_state.json"
RUN_STATE_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dataset_identity_payload(dataset: PreparedDataset) -> Dict[str, object]:
    return {
        "dataset_id": dataset.dataset_id,
        "resolved_views_path": dataset.resolved_views_path or str(dataset.views_path.resolve()),
        "views_shape": list(dataset.views_shape),
        "views_dtype": dataset.views_dtype,
        "views_itemsize": int(dataset.views_itemsize),
        "views_file_size_bytes": dataset.views_file_size_bytes,
        "views_file_mtime_ns": dataset.views_file_mtime_ns,
    }


def _build_experiment_fingerprint_payload(
    spec: SweepExperimentSpec,
    prepared_datasets: Sequence[PreparedDataset],
    output_dir: Path,
) -> Dict[str, object]:
    spec_snapshot = asdict(spec)
    spec_snapshot["output_dir"] = str(output_dir.resolve())
    datasets_payload = sorted(
        [_dataset_identity_payload(dataset) for dataset in prepared_datasets],
        key=lambda item: str(item["dataset_id"]),
    )
    return {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "output_dir": str(output_dir.resolve()),
        "spec": spec_snapshot,
        "datasets": datasets_payload,
    }


def _compute_experiment_fingerprint(payload: Dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _persist_run_state(state_path: Path, state: Dict[str, object]) -> None:
    state["updated_at"] = _utc_now_iso()
    write_json_atomic(state_path, state, indent=2, sort_keys=False, create_backup=True)


def _load_or_initialize_run_state(
    output_dir: Path,
    fingerprint_payload: Dict[str, object],
) -> Tuple[Dict[str, object], Path]:
    state_path = output_dir / RUN_STATE_FILENAME
    fingerprint = _compute_experiment_fingerprint(fingerprint_payload)
    if state_path.exists() or Path(f"{state_path}.bak").exists():
        loaded, _, _ = read_json_first_valid(
            state_path,
            validator=lambda obj: isinstance(obj, dict),
        )
        state = cast(Dict[str, object], loaded)
        existing_fingerprint = str(state.get("experiment_fingerprint", ""))
        if existing_fingerprint != fingerprint:
            raise ValueError(
                "Output directory contains run_state.json with a different experiment "
                "fingerprint. Refusing to resume with mismatched spec/dataset/path."
            )
        existing_output_dir = str(cast(Dict[str, object], state.get("path_compatibility", {})).get("output_dir", ""))
        current_output_dir = str(output_dir.resolve())
        if existing_output_dir and existing_output_dir != current_output_dir:
            raise ValueError(
                "run_state.json output path mismatch detected. "
                f"state={existing_output_dir}, current={current_output_dir}"
            )
        return state, state_path

    state: Dict[str, object] = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "experiment_fingerprint": fingerprint,
        "path_compatibility": {
            "output_dir": str(output_dir.resolve()),
        },
        "fingerprint_payload": fingerprint_payload,
        "stages": {},
        "runs": {},
    }
    _persist_run_state(state_path, state)
    return state, state_path


def _validate_run_artifacts(run_dir: Path) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {
        "run_dir": str(run_dir),
        "valid": False,
        "missing": [],
        "corrupt": [],
    }
    required_paths = {
        "config": run_dir / "config.json",
        "metric_record": run_dir / "metric_record.json",
        "rectified_views": run_dir / "rectified_views.npy",
        "workspace": run_dir / "workspace.npz",
        "avg_rectified": run_dir / "avg_rectified.png",
    }
    missing = cast(List[str], diagnostics["missing"])
    corrupt = cast(List[str], diagnostics["corrupt"])
    for key, path in required_paths.items():
        if not path.exists() and not Path(f"{path}.bak").exists():
            missing.append(f"{key}:{path.name}")

    if missing:
        diagnostics["valid"] = False
        return diagnostics

    try:
        read_json_first_valid(required_paths["config"], validator=lambda obj: isinstance(obj, dict))
    except Exception as exc:  # noqa: BLE001
        corrupt.append(f"config:{type(exc).__name__}")
    try:
        read_json_first_valid(
            required_paths["metric_record"],
            validator=lambda obj: isinstance(obj, dict) and "global_and_local_metrics" in cast(Dict[str, object], obj),
        )
    except Exception as exc:  # noqa: BLE001
        corrupt.append(f"metric_record:{type(exc).__name__}")
    try:
        read_npy_first_valid(required_paths["rectified_views"], allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        corrupt.append(f"rectified_views:{type(exc).__name__}")
    try:
        read_npz_first_valid(
            required_paths["workspace"],
            allow_pickle=True,
            validator=lambda obj: "rectified_views" in obj and "average_rectified_image" in obj,
        )
    except Exception as exc:  # noqa: BLE001
        corrupt.append(f"workspace:{type(exc).__name__}")
    if not required_paths["avg_rectified"].exists() and not Path(f"{required_paths['avg_rectified']}.bak").exists():
        corrupt.append("avg_rectified:missing")

    diagnostics["valid"] = not corrupt and not missing
    return diagnostics


def _load_cached_stage_result(task: Dict[str, object], run_dir: Path) -> Dict[str, object]:
    config_payload, _, _ = read_json_first_valid(
        run_dir / "config.json",
        validator=lambda obj: isinstance(obj, dict),
    )
    metric_record, _, _ = read_json_first_valid(
        run_dir / "metric_record.json",
        validator=lambda obj: isinstance(obj, dict),
    )
    return {
        "status": "ok",
        "stage": str(task["stage"]),
        "candidate_id": str(task["candidate_id"]),
        "dataset_id": str(task["dataset_id"]),
        "strategy": str(task["strategy"]),
        "run_uid": str(task["run_uid"]),
        "run_dir": str(run_dir),
        "runtime_seconds": float(0.0),
        "config": config_payload,
        "params": task["params"],
        "metric_record": metric_record,
        "reused_artifacts": True,
    }


def _set_parallel_thread_limits() -> None:
    for env_var in THREAD_LIMIT_ENV_VARS:
        os.environ[env_var] = "1"


def _pool_worker_initializer() -> None:
    _set_parallel_thread_limits()
    try:
        import cv2 as cv  # Local import keeps startup cost inside workers.

        cv.setNumThreads(1)
    except Exception:  # noqa: BLE001 - thread cap is best-effort.
        return


def _get_macos_available_memory_bytes() -> Optional[int]:
    try:
        vm_stat_output = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001 - OS-level querying can fail on restricted environments.
        return None

    page_size = 4096
    free_pages = 0
    inactive_pages = 0
    speculative_pages = 0
    for line in vm_stat_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Mach Virtual Memory Statistics:"):
            if "page size of " in stripped and " bytes" in stripped:
                try:
                    token = stripped.split("page size of ", 1)[1].split(" bytes", 1)[0]
                    page_size = int(token.replace(".", "").strip())
                except Exception:  # noqa: BLE001
                    page_size = 4096
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        value_token = raw_value.strip().replace(".", "")
        try:
            page_count = int(value_token)
        except Exception:  # noqa: BLE001
            continue
        lowered = key.strip().lower()
        if lowered == "pages free":
            free_pages = page_count
        elif lowered == "pages inactive":
            inactive_pages = page_count
        elif lowered == "pages speculative":
            speculative_pages = page_count

    total_pages = free_pages + inactive_pages + speculative_pages
    if total_pages <= 0:
        return None
    return int(total_pages * page_size)


def _get_available_memory_bytes() -> Optional[int]:
    system_name = platform.system().lower()
    if system_name == "windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32 = ctypes.windll.kernel32
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
                return None
            return int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001
            return None

    try:
        page_count = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if page_count > 0 and page_size > 0:
            return int(page_count * page_size)
    except Exception:  # noqa: BLE001
        pass

    if system_name == "darwin":
        return _get_macos_available_memory_bytes()
    return None


def _task_view_geometry(task: Dict[str, object]) -> Tuple[int, int, int, int, int]:
    shape = tuple(int(dim) for dim in cast(Sequence[int], task["views_shape"]))
    if len(shape) == 4:
        grid_n = int(shape[0])
        height = int(shape[2])
        width = int(shape[3])
        exposures = 1
        return exposures, grid_n, height, width, int(np.prod(shape, dtype=np.int64))
    if len(shape) == 5:
        exposures = int(shape[0])
        grid_n = int(shape[1])
        height = int(shape[3])
        width = int(shape[4])
        return exposures, grid_n, height, width, int(np.prod(shape, dtype=np.int64))
    raise ValueError(f"Unsupported view shape in task: {shape}")


def _estimate_global_peak_bytes(task: Dict[str, object]) -> int:
    exposures, grid_n, height, width, total_elements = _task_view_geometry(task)
    grid_views = grid_n * grid_n
    float64_bytes = 8
    input_itemsize = max(1, int(task.get("views_itemsize", 8)))

    loaded_input_bytes = total_elements * input_itemsize
    base_views_bytes = grid_views * height * width * float64_bytes
    similarity_bytes = (grid_n**4) * 2 * float64_bytes
    rectified_bytes = grid_views * height * width * float64_bytes
    hdr_rectified_stack_bytes = exposures * grid_views * height * width * float64_bytes if exposures > 1 else 0
    avg_image_bytes = height * width * float64_bytes
    return int(
        loaded_input_bytes
        + base_views_bytes
        + similarity_bytes
        + rectified_bytes
        + hdr_rectified_stack_bytes
        + avg_image_bytes
    )


def _estimate_pixel_peak_bytes(task: Dict[str, object]) -> int:
    config = cast(Dict[str, object], task["config"])
    exposures, grid_n, height, width, total_elements = _task_view_geometry(task)
    grid_views = grid_n * grid_n
    float64_bytes = 8
    float32_bytes = 4
    input_itemsize = max(1, int(task.get("views_itemsize", 8)))

    sub_pix_factor = max(1, int(config.get("sub_pix_factor", 1)))
    window_size = int(config.get("similarity_window_size", config.get("refinement_window_size", 41)))
    max_disp_raw = config.get("max_displacement")
    if max_disp_raw is None:
        max_disp = max(1, window_size // 2)
    else:
        max_disp = max(0, int(max_disp_raw))
    max_disp_scaled = max_disp * sub_pix_factor
    displacement_volume = max(1, 2 * max_disp_scaled + 1)

    scaled_height = height * sub_pix_factor
    scaled_width = width * sub_pix_factor

    loaded_input_bytes = total_elements * input_itemsize
    base_views_bytes = grid_views * height * width * float64_bytes
    scaled_views_bytes = grid_views * scaled_height * scaled_width * float64_bytes
    corr_buffers_bytes = (
        scaled_height
        * scaled_width
        * displacement_volume
        * displacement_volume
        * float32_bytes
        * 2
    )
    per_pixel_similarity_bytes = (2 * float32_bytes) + float64_bytes + (2 * float32_bytes)
    similarity_scaled_bytes = scaled_height * scaled_width * grid_views * per_pixel_similarity_bytes
    similarity_resized_bytes = (
        height * width * grid_views * per_pixel_similarity_bytes if sub_pix_factor > 1 else 0
    )
    rectified_bytes = grid_views * height * width * float64_bytes
    hdr_rectified_stack_bytes = exposures * grid_views * height * width * float64_bytes if exposures > 1 else 0
    avg_image_bytes = height * width * float64_bytes

    return int(
        loaded_input_bytes
        + base_views_bytes
        + scaled_views_bytes
        + corr_buffers_bytes
        + similarity_scaled_bytes
        + similarity_resized_bytes
        + rectified_bytes
        + hdr_rectified_stack_bytes
        + avg_image_bytes
    )


def _estimate_task_peak_bytes(task: Dict[str, object]) -> int:
    strategy = str(task["strategy"]).lower()
    if strategy == "global":
        return _estimate_global_peak_bytes(task)
    if strategy == "pixel":
        return _estimate_pixel_peak_bytes(task)
    if strategy == "combined":
        return _estimate_global_peak_bytes(task) + _estimate_pixel_peak_bytes(task)
    return _estimate_global_peak_bytes(task)


def _select_stage_worker_count(
    tasks: Sequence[Dict[str, object]],
    requested_workers: int,
    available_memory_bytes: Optional[int],
) -> Dict[str, object]:
    if not tasks:
        return {
            "requested_workers": int(max(1, requested_workers)),
            "available_memory_bytes": available_memory_bytes,
            "max_task_peak_bytes": 0,
            "memory_cap_workers": None,
            "selected_workers": 1,
            "safety_fraction": MEMORY_SAFETY_FRACTION,
        }

    max_task_peak_bytes = max(int(task["estimated_peak_bytes"]) for task in tasks)
    memory_cap_workers: Optional[int] = None
    if available_memory_bytes is not None and max_task_peak_bytes > 0:
        capped = int(math.floor((float(available_memory_bytes) * MEMORY_SAFETY_FRACTION) / float(max_task_peak_bytes)))
        memory_cap_workers = max(1, capped)

    selected_workers = max(1, min(int(requested_workers), len(tasks)))
    if memory_cap_workers is not None:
        selected_workers = max(1, min(selected_workers, memory_cap_workers))

    return {
        "requested_workers": int(max(1, requested_workers)),
        "available_memory_bytes": available_memory_bytes,
        "max_task_peak_bytes": int(max_task_peak_bytes),
        "memory_cap_workers": memory_cap_workers,
        "selected_workers": int(selected_workers),
        "safety_fraction": MEMORY_SAFETY_FRACTION,
    }


def _build_worker_crash_result(
    task: Dict[str, object],
    elapsed_seconds: float,
    exc: BaseException,
) -> Dict[str, object]:
    return {
        "status": "failed",
        "stage": str(task["stage"]),
        "candidate_id": str(task["candidate_id"]),
        "dataset_id": str(task["dataset_id"]),
        "strategy": str(task["strategy"]),
        "run_uid": str(task["run_uid"]),
        "runtime_seconds": float(elapsed_seconds),
        "config": task["config"],
        "params": task["params"],
        "error": f"Worker process terminated abruptly: {exc}",
        "traceback": traceback.format_exc(),
        "failure_kind": "worker_crash",
    }


def _run_task_isolated(task: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            initializer=_pool_worker_initializer,
            max_tasks_per_child=1,
        ) as executor:
            future = executor.submit(_evaluate_candidate_task, task)
            result = future.result()
        result["task_index"] = task["task_index"]
        return result, False
    except Exception as exc:  # noqa: BLE001 - protects sweep continuity.
        elapsed = time.perf_counter() - started
        failed = _build_worker_crash_result(task, elapsed, exc)
        failed["task_index"] = task["task_index"]
        return failed, True


def _execute_pool_batch(
    tasks: Sequence[Dict[str, object]],
    workers: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Optional[BaseException]]:
    completed_results: List[Dict[str, object]] = []
    completed_indices: set[int] = set()
    pending_tasks = list(tasks)
    executor_exc: Optional[BaseException] = None

    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_pool_worker_initializer) as executor:
            future_map = {
                executor.submit(_evaluate_candidate_task, task): task for task in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                    result["task_index"] = task["task_index"]
                    completed_results.append(result)
                    completed_indices.add(int(task["task_index"]))
                except Exception as exc:  # noqa: BLE001 - recover by degrading worker count.
                    executor_exc = exc
                    break

            if executor_exc is None:
                pending_tasks = []
            else:
                pending_tasks = [
                    task
                    for task in tasks
                    if int(task["task_index"]) not in completed_indices
                ]
    except Exception as exc:  # noqa: BLE001 - covers pool init/spawn failures.
        executor_exc = exc
        pending_tasks = [
            task
            for task in tasks
            if int(task["task_index"]) not in completed_indices
        ]

    return completed_results, pending_tasks, executor_exc


def _evaluate_candidate_task(task: Dict[str, object]) -> Dict[str, object]:
    run_uid = str(task["run_uid"])
    stage = str(task["stage"])
    candidate_id = str(task["candidate_id"])
    dataset_id = str(task["dataset_id"])
    strategy = str(task["strategy"])
    started = time.perf_counter()
    try:
        views = _load_views(Path(cast(str, task["views_path"])))
        config = RectificationRunConfig(**cast(Dict[str, object], task["config"]))
        result = rectify_views_from_config(views, config)
        output_dir = Path(cast(str, task["run_parent_dir"]))
        run_name = str(task["run_name"])
        metric_record = build_metric_record(run_uid, config, result)
        _save_run_artifacts(output_dir, run_name, result, metric_record=metric_record)
        if bool(task.get("save_figures", False)):
            figures = generate_rectification_figures(result)
            _persist_figures(figures, output_dir / run_name / "figures")
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "stage": stage,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "run_uid": run_uid,
            "run_dir": str(output_dir / run_name),
            "runtime_seconds": float(elapsed),
            "config": asdict(config),
            "params": task["params"],
            "metric_record": metric_record,
        }
    except Exception as exc:  # noqa: BLE001 - preserve failed run details
        elapsed = time.perf_counter() - started
        return {
            "status": "failed",
            "stage": stage,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "run_uid": run_uid,
            "runtime_seconds": float(elapsed),
            "config": task["config"],
            "params": task["params"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _default_max_workers(explicit: Optional[int]) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _evaluate_stage(
    stage_name: str,
    candidates: Sequence[SweepCandidate],
    datasets: Sequence[PreparedDataset],
    output_dir: Path,
    save_figures_flag: bool,
    requested_workers: int,
    run_state: Optional[Dict[str, object]] = None,
    run_state_path: Optional[Path] = None,
) -> StageEvaluationResult:
    tasks: List[Dict[str, object]] = []
    for candidate in candidates:
        candidate_root = output_dir / "runs" / stage_name / candidate.candidate_id
        for dataset in datasets:
            config = _compose_run_config(candidate, dataset)
            run_name = dataset.dataset_id
            run_uid = f"{stage_name}_{candidate.candidate_id}_{dataset.dataset_id}"
            tasks.append(
                {
                    "stage": stage_name,
                    "candidate_id": candidate.candidate_id,
                    "dataset_id": dataset.dataset_id,
                    "strategy": candidate.strategy,
                    "run_uid": run_uid,
                    "run_name": run_name,
                    "run_parent_dir": str(candidate_root),
                    "views_path": str(dataset.views_path),
                    "config": asdict(config),
                    "params": dict(candidate.params),
                    "save_figures": save_figures_flag,
                    "views_shape": list(dataset.views_shape),
                    "views_dtype": dataset.views_dtype,
                    "views_itemsize": int(dataset.views_itemsize),
                }
            )
    tasks.sort(key=lambda item: (str(item["candidate_id"]), str(item["dataset_id"])))
    for idx, task in enumerate(tasks):
        task["task_index"] = idx
        task["estimated_peak_bytes"] = int(_estimate_task_peak_bytes(task))
        task["run_dir"] = str(Path(cast(str, task["run_parent_dir"])) / str(task["run_name"]))

    if not tasks:
        return StageEvaluationResult(
            stage_results=[],
            execution={
                "requested_workers": int(max(1, requested_workers)),
                "selected_workers": 1,
                "available_memory_bytes": _get_available_memory_bytes(),
                "max_task_peak_bytes": 0,
                "memory_cap_workers": None,
                "safety_fraction": MEMORY_SAFETY_FRACTION,
                "degrade_events": [],
                "pool_crash_count": 0,
                "isolated_crash_count": 0,
                "skipped_valid_count": 0,
                "recomputed_count": 0,
            },
        )

    stage_results: List[Dict[str, object]] = []
    tasks_to_execute: List[Dict[str, object]] = []
    skipped_valid_count = 0
    recomputed_count = 0
    state_runs = (
        cast(Dict[str, Dict[str, object]], run_state.setdefault("runs", {}))
        if run_state is not None
        else {}
    )
    if run_state is not None:
        stages_state = cast(Dict[str, Dict[str, object]], run_state.setdefault("stages", {}))
        stage_slot = cast(Dict[str, object], stages_state.setdefault(stage_name, {}))
        stage_slot["planned_runs"] = len(tasks)
        stage_slot["last_started_at"] = _utc_now_iso()
        if run_state_path is not None:
            _persist_run_state(run_state_path, run_state)

    for task in tasks:
        run_uid = str(task["run_uid"])
        run_dir = Path(cast(str, task["run_dir"]))
        entry = state_runs.get(run_uid)
        if entry is None and run_state is not None:
            entry = {
                "run_uid": run_uid,
                "stage": str(task["stage"]),
                "candidate_id": str(task["candidate_id"]),
                "dataset_id": str(task["dataset_id"]),
                "status": "pending",
                "run_dir": str(run_dir),
                "attempts": [],
                "artifact_validation": {},
            }
            state_runs[run_uid] = entry
        previous_status = str(entry.get("status", "pending")) if entry is not None else "pending"
        artifact_validation = _validate_run_artifacts(run_dir)
        if entry is not None:
            entry["artifact_validation"] = artifact_validation

        if bool(artifact_validation.get("valid", False)):
            try:
                cached_result = _load_cached_stage_result(task, run_dir)
            except Exception:
                task["_recompute_after_invalid"] = previous_status in {"ok", "recomputed", "skipped_valid"}
                tasks_to_execute.append(task)
                continue
            cached_result["task_index"] = task["task_index"]
            stage_results.append(cached_result)
            skipped_valid_count += 1
            if entry is not None:
                attempts = cast(List[Dict[str, object]], entry.setdefault("attempts", []))
                attempts.append(
                    {
                        "timestamp": _utc_now_iso(),
                        "status": "skipped_valid",
                        "reason": "artifacts_valid",
                    }
                )
                entry["status"] = "skipped_valid"
            continue

        task["_recompute_after_invalid"] = previous_status in {"ok", "recomputed", "skipped_valid"}
        tasks_to_execute.append(task)

    if run_state is not None and run_state_path is not None:
        _persist_run_state(run_state_path, run_state)

    progress_desc = f"Evaluating {stage_name}"
    _set_parallel_thread_limits()
    available_memory_bytes = _get_available_memory_bytes()
    worker_plan = _select_stage_worker_count(tasks_to_execute, requested_workers, available_memory_bytes)
    selected_workers = int(worker_plan["selected_workers"])

    degrade_events: List[Dict[str, object]] = []
    pool_crash_count = 0
    isolated_crash_count = 0
    completed_task_indices: set[int] = set()

    remaining_tasks = list(tasks_to_execute)
    current_workers = selected_workers
    with tqdm(total=len(tasks), desc=progress_desc, unit="run") as progress_bar:
        if stage_results:
            completed_task_indices.update(int(item["task_index"]) for item in stage_results)
            progress_bar.update(len(stage_results))
        while remaining_tasks:
            if run_state is not None:
                for task in remaining_tasks:
                    run_uid = str(task["run_uid"])
                    entry = state_runs.get(run_uid)
                    if entry is None:
                        continue
                    entry["status"] = "running"
                    entry["last_started_at"] = _utc_now_iso()
                if run_state_path is not None:
                    _persist_run_state(run_state_path, run_state)

            if current_workers <= 1:
                for task in remaining_tasks:
                    result, crashed = _run_task_isolated(task)
                    if crashed:
                        isolated_crash_count += 1
                    stage_results.append(result)
                    completed_task_indices.add(int(task["task_index"]))
                    progress_bar.update(1)
                    if run_state is not None:
                        entry = state_runs.get(str(task["run_uid"]))
                        if entry is not None:
                            attempts = cast(List[Dict[str, object]], entry.setdefault("attempts", []))
                            attempts.append(
                                {
                                    "timestamp": _utc_now_iso(),
                                    "status": result.get("status"),
                                    "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
                                    "error": result.get("error"),
                                }
                            )
                            if result.get("status") == "ok":
                                if bool(task.get("_recompute_after_invalid", False)):
                                    entry["status"] = "recomputed"
                                    recomputed_count += 1
                                else:
                                    entry["status"] = "ok"
                            else:
                                entry["status"] = "failed"
                            entry["artifact_validation"] = _validate_run_artifacts(Path(cast(str, task["run_dir"])))
                            if run_state_path is not None:
                                _persist_run_state(run_state_path, run_state)
                break

            completed_batch, pending_batch, exec_error = _execute_pool_batch(
                remaining_tasks,
                workers=current_workers,
            )
            stage_results.extend(completed_batch)
            completed_task_indices.update(int(item["task_index"]) for item in completed_batch)
            if completed_batch:
                progress_bar.update(len(completed_batch))
                if run_state is not None:
                    by_uid = {str(task["run_uid"]): task for task in remaining_tasks}
                    for result in completed_batch:
                        run_uid = str(result["run_uid"])
                        task = by_uid.get(run_uid)
                        entry = state_runs.get(run_uid)
                        if task is None or entry is None:
                            continue
                        attempts = cast(List[Dict[str, object]], entry.setdefault("attempts", []))
                        attempts.append(
                            {
                                "timestamp": _utc_now_iso(),
                                "status": result.get("status"),
                                "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
                                "error": result.get("error"),
                            }
                        )
                        if result.get("status") == "ok":
                            if bool(task.get("_recompute_after_invalid", False)):
                                entry["status"] = "recomputed"
                                recomputed_count += 1
                            else:
                                entry["status"] = "ok"
                        else:
                            entry["status"] = "failed"
                        entry["artifact_validation"] = _validate_run_artifacts(Path(cast(str, task["run_dir"])))
                    if run_state_path is not None:
                        _persist_run_state(run_state_path, run_state)

            if exec_error is None:
                break

            pool_crash_count += 1
            next_workers = max(1, current_workers // 2)
            degrade_events.append(
                {
                    "event": "pool_failure",
                    "error_type": type(exec_error).__name__,
                    "error": str(exec_error),
                    "broken_process_pool": isinstance(exec_error, BrokenProcessPool),
                    "workers_before": int(current_workers),
                    "workers_after": int(next_workers),
                    "remaining_tasks": len(pending_batch),
                }
            )
            remaining_tasks = pending_batch
            current_workers = next_workers

    stage_results.sort(key=lambda item: int(item.get("task_index", 0)))
    execution = dict(worker_plan)
    execution["degrade_events"] = degrade_events
    execution["pool_crash_count"] = int(pool_crash_count)
    execution["isolated_crash_count"] = int(isolated_crash_count)
    execution["completed_runs"] = len(stage_results)
    execution["skipped_valid_count"] = int(skipped_valid_count)
    execution["recomputed_count"] = int(recomputed_count)
    execution["used_isolated_mode"] = bool(isolated_crash_count > 0 or current_workers == 1)
    if run_state is not None:
        stages_state = cast(Dict[str, Dict[str, object]], run_state.setdefault("stages", {}))
        stage_slot = cast(Dict[str, object], stages_state.setdefault(stage_name, {}))
        stage_slot["completed_runs"] = len(stage_results)
        stage_slot["skipped_valid_count"] = int(skipped_valid_count)
        stage_slot["recomputed_count"] = int(recomputed_count)
        stage_slot["last_completed_at"] = _utc_now_iso()
        if run_state_path is not None:
            _persist_run_state(run_state_path, run_state)
    return StageEvaluationResult(stage_results=stage_results, execution=execution)


def _normalize_score(value: float, min_value: float, max_value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return 0.0
    if abs(max_value - min_value) <= 1e-12:
        return 0.5
    return float((value - min_value) / (max_value - min_value))


def score_and_rank_runs(
    run_results: Sequence[Dict[str, object]],
    scoring: SweepScoringSpec,
    dataset_ids: Sequence[str],
) -> List[Dict[str, object]]:
    """Score and rank candidates from one stage using normalized composite metrics."""

    successful = [record for record in run_results if record.get("status") == "ok"]
    if not successful:
        return []

    metric_weights = dict(scoring.metric_weights)
    if not metric_weights:
        raise ValueError("At least one metric weight must be configured.")
    weight_sum = float(sum(metric_weights.values()))
    if weight_sum <= 0:
        raise ValueError("Metric weights must sum to a positive value.")

    dataset_metric_values: Dict[str, Dict[str, List[float]]] = {
        dataset_id: {metric: [] for metric in metric_weights.keys()} for dataset_id in dataset_ids
    }
    dataset_runtime_values: Dict[str, List[float]] = {dataset_id: [] for dataset_id in dataset_ids}
    for record in successful:
        dataset_id = str(record["dataset_id"])
        metrics = cast(Dict[str, float], cast(Dict[str, object], record["metric_record"]).get(scoring.metric_scope))
        if not isinstance(metrics, dict):
            metrics = cast(
                Dict[str, float],
                cast(Dict[str, object], record["metric_record"]).get("global_and_local_metrics", {}),
            )
        for metric in metric_weights.keys():
            if metric in metrics:
                dataset_metric_values[dataset_id][metric].append(float(metrics[metric]))
        dataset_runtime_values[dataset_id].append(float(record.get("runtime_seconds", 0.0)))

    dataset_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for dataset_id, metric_map in dataset_metric_values.items():
        dataset_bounds[dataset_id] = {}
        for metric, values in metric_map.items():
            if not values:
                dataset_bounds[dataset_id][metric] = (0.0, 0.0)
            else:
                dataset_bounds[dataset_id][metric] = (float(np.min(values)), float(np.max(values)))
    runtime_bounds: Dict[str, Tuple[float, float]] = {}
    for dataset_id, runtimes in dataset_runtime_values.items():
        if not runtimes:
            runtime_bounds[dataset_id] = (0.0, 0.0)
        else:
            runtime_bounds[dataset_id] = (float(np.min(runtimes)), float(np.max(runtimes)))

    grouped: Dict[str, Dict[str, object]] = {}
    for record in successful:
        candidate_id = str(record["candidate_id"])
        dataset_id = str(record["dataset_id"])
        metrics = cast(Dict[str, float], cast(Dict[str, object], record["metric_record"]).get(scoring.metric_scope))
        if not isinstance(metrics, dict):
            metrics = cast(
                Dict[str, float],
                cast(Dict[str, object], record["metric_record"]).get("global_and_local_metrics", {}),
            )

        weighted_sum = 0.0
        for metric, weight in metric_weights.items():
            metric_value = float(metrics.get(metric, float("nan")))
            min_val, max_val = dataset_bounds[dataset_id][metric]
            normalized = _normalize_score(metric_value, min_val, max_val)
            weighted_sum += float(weight) * normalized
        dataset_score = weighted_sum / weight_sum

        runtime = float(record.get("runtime_seconds", 0.0))
        if scoring.runtime_penalty_weight > 0:
            rt_min, rt_max = runtime_bounds[dataset_id]
            runtime_norm = _normalize_score(runtime, rt_min, rt_max)
            dataset_score -= float(scoring.runtime_penalty_weight) * runtime_norm

        slot = grouped.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "strategy": record["strategy"],
                "config": record["config"],
                "params": record["params"],
                "dataset_scores": {},
                "dataset_metrics": {},
                "runtime_seconds": {},
            },
        )
        cast(Dict[str, float], slot["dataset_scores"])[dataset_id] = float(dataset_score)
        cast(Dict[str, Dict[str, float]], slot["dataset_metrics"])[dataset_id] = metrics
        cast(Dict[str, float], slot["runtime_seconds"])[dataset_id] = runtime

    rankings: List[Dict[str, object]] = []
    expected_datasets = list(dataset_ids)
    for slot in grouped.values():
        dataset_scores_map = cast(Dict[str, float], slot["dataset_scores"])
        present_scores = [dataset_scores_map[dataset_id] for dataset_id in expected_datasets if dataset_id in dataset_scores_map]
        missing = [dataset_id for dataset_id in expected_datasets if dataset_id not in dataset_scores_map]
        if present_scores:
            mean_score = float(np.mean(present_scores))
            std_score = float(np.std(present_scores))
        else:
            mean_score = -float("inf")
            std_score = 0.0
        robust = mean_score - float(scoring.robustness_lambda) * std_score
        robust -= float(scoring.missing_dataset_penalty) * len(missing)
        rankings.append(
            {
                "candidate_id": slot["candidate_id"],
                "strategy": slot["strategy"],
                "config": slot["config"],
                "params": slot["params"],
                "score": float(robust),
                "mean_dataset_score": float(mean_score),
                "std_dataset_score": float(std_score),
                "missing_datasets": missing,
                "dataset_scores": slot["dataset_scores"],
                "runtime_seconds": slot["runtime_seconds"],
                "dataset_metrics": slot["dataset_metrics"],
            }
        )

    rankings.sort(key=lambda item: float(item["score"]), reverse=True)
    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx
    return rankings


def _write_leaderboard_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "strategy",
        "score",
        "mean_dataset_score",
        "std_dataset_score",
        "missing_datasets",
    ]
    csv_rows = [
        {
            "rank": row.get("rank"),
            "candidate_id": row.get("candidate_id"),
            "strategy": row.get("strategy"),
            "score": row.get("score"),
            "mean_dataset_score": row.get("mean_dataset_score"),
            "std_dataset_score": row.get("std_dataset_score"),
            "missing_datasets": ",".join(cast(List[str], row.get("missing_datasets", []))),
        }
        for row in rows
    ]
    write_csv_atomic(path, fieldnames, csv_rows, create_backup=True)


def _write_stage_outputs(
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
    output_dir: Path,
    execution: Optional[Dict[str, object]] = None,
) -> None:
    successful = [record for record in stage_results if record.get("status") == "ok"]
    failed = [record for record in stage_results if record.get("status") != "ok"]
    payload = {
        "stage": stage_name,
        "num_runs": len(stage_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "rankings": list(rankings),
        "failed_runs": failed,
        "runs": list(stage_results),
        "execution": dict(execution or {}),
    }
    summary_path = output_dir / f"{stage_name}_summary.json"
    write_json_atomic(summary_path, payload, indent=2, sort_keys=False, create_backup=True)


def _best_configs_payload(rankings: Sequence[Dict[str, object]]) -> Dict[str, object]:
    best_by_strategy: Dict[str, Dict[str, object]] = {}
    for row in rankings:
        strategy = str(row["strategy"])
        if strategy not in best_by_strategy:
            best_by_strategy[strategy] = dict(row)
    return {
        "overall_best": dict(rankings[0]) if rankings else None,
        "best_by_strategy": best_by_strategy,
    }


def _candidate_without_metric_axes(candidate: SweepCandidate) -> SweepCandidate:
    metric_keys = {
        "displacement_metric_family",
        "displacement_alpha",
        "metric_formulation_id",
        "displacement_component_weights",
    }
    params = {
        key: value for key, value in dict(candidate.params).items() if key not in metric_keys
    }
    return _build_candidate_from_params(candidate.strategy, params)


def _collect_metric_experiment_candidates(
    final_rankings: Sequence[Dict[str, object]],
    baseline_candidate_lookup: Dict[str, SweepCandidate],
    experimental_spec: ExperimentalMetricSweepSpec,
) -> List[SweepCandidate]:
    shortlisted_candidates: List[SweepCandidate] = []
    seen_shortlist = set()
    for row in final_rankings:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id in seen_shortlist:
            continue
        candidate = baseline_candidate_lookup.get(candidate_id)
        if candidate is None:
            continue
        if candidate.strategy not in {"pixel", "combined"}:
            continue
        shortlisted_candidates.append(candidate)
        seen_shortlist.add(candidate_id)
        if len(shortlisted_candidates) >= int(experimental_spec.shortlist_top_k):
            break

    if not shortlisted_candidates:
        return []

    base_candidates = _dedupe_candidates(
        [_candidate_without_metric_axes(candidate) for candidate in shortlisted_candidates]
    )
    metric_candidates = _expand_candidates_with_metric_formulations(
        base_candidates,
        experimental_spec.formulations,
    )
    return _dedupe_candidates(metric_candidates)


def _write_phase_summary_artifacts(
    *,
    output_dir: Path,
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
    execution: Optional[Dict[str, object]],
    reporting: SweepReportingSpec,
    analyze: bool,
) -> None:
    _write_stage_outputs(
        stage_name,
        stage_results,
        rankings,
        output_dir,
        execution=execution,
    )
    write_json_atomic(
        output_dir / "leaderboard.json",
        list(rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    _write_leaderboard_csv(rankings, output_dir / "leaderboard.csv")
    write_json_atomic(
        output_dir / "best_configs.json",
        _best_configs_payload(rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    all_metric_records: List[Dict[str, object]] = []
    for stage_result in stage_results:
        if stage_result.get("status") == "ok":
            all_metric_records.append(cast(Dict[str, object], stage_result["metric_record"]))
    write_json_atomic(
        output_dir / "summary.json",
        all_metric_records,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    if analyze and all_metric_records:
        analysis = analyze_summary_data(all_metric_records)
        save_analysis(analysis, output_dir / "analysis.json")
    if reporting.generate_alpha_report:
        generate_alpha_report(
            output_dir=output_dir,
            primary_metric=reporting.primary_metric,
            tolerate_partial_outputs=reporting.tolerate_partial_outputs,
        )


def run_sweep_experiment(spec: SweepExperimentSpec, output_dir: Optional[Path] = None) -> Path:
    """Run a full two-stage, multi-dataset sweep experiment."""

    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else (
            Path(spec.output_dir)
            if spec.output_dir
            else Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    prepared_datasets = _prepare_datasets(spec, resolved_output_dir)
    dataset_ids = [dataset.dataset_id for dataset in prepared_datasets]
    fingerprint_payload = _build_experiment_fingerprint_payload(
        spec=spec,
        prepared_datasets=prepared_datasets,
        output_dir=resolved_output_dir,
    )
    run_state, run_state_path = _load_or_initialize_run_state(
        output_dir=resolved_output_dir,
        fingerprint_payload=fingerprint_payload,
    )

    stage1_budget, stage2_budget = _stage_budgets(spec.stages)
    baseline_metric_formulations = _baseline_metric_formulations()

    coarse_candidates_base: List[SweepCandidate] = []
    for strategy in spec.strategies:
        search_space = spec.search_spaces[strategy]
        coarse_candidates_base.extend(generate_coarse_candidates(search_space, budget=None))
    coarse_candidates = _expand_candidates_with_metric_formulations(
        coarse_candidates_base,
        baseline_metric_formulations,
    )
    coarse_candidates = _dedupe_candidates(coarse_candidates)
    coarse_candidates = _limit_candidates(coarse_candidates, stage1_budget)
    coarse_candidates = _assign_candidate_ids(coarse_candidates, "s1c")
    coarse_lookup = {candidate.candidate_id: candidate for candidate in coarse_candidates}

    requested_workers = _default_max_workers(spec.stages.max_workers)

    stage1_eval = _evaluate_stage(
        stage_name="stage1",
        candidates=coarse_candidates,
        datasets=prepared_datasets,
        output_dir=resolved_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=requested_workers,
        run_state=run_state,
        run_state_path=run_state_path,
    )
    stage1_results = stage1_eval.stage_results
    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs(
        "stage1",
        stage1_results,
        stage1_rankings,
        resolved_output_dir,
        execution=stage1_eval.execution,
    )

    refined_candidates = generate_refined_candidates(
        ranked_candidates=stage1_rankings,
        candidate_lookup=coarse_lookup,
        search_spaces=spec.search_spaces,
        top_k=spec.stages.top_k,
        budget=stage2_budget,
    )
    if not refined_candidates and stage1_rankings:
        # Fallback: keep exploring with the best stage-1 candidate if refinement is empty.
        best_id = str(stage1_rankings[0]["candidate_id"])
        refined_candidates = [coarse_lookup[best_id]]
    refined_candidates = _dedupe_candidates(refined_candidates)
    refined_candidates = _assign_candidate_ids(refined_candidates, "s2c")
    refined_lookup = {candidate.candidate_id: candidate for candidate in refined_candidates}

    stage2_eval = _evaluate_stage(
        stage_name="stage2",
        candidates=refined_candidates,
        datasets=prepared_datasets,
        output_dir=resolved_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=requested_workers,
        run_state=run_state,
        run_state_path=run_state_path,
    )
    stage2_results = stage2_eval.stage_results
    stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs(
        "stage2",
        stage2_results,
        stage2_rankings,
        resolved_output_dir,
        execution=stage2_eval.execution,
    )

    final_rankings = stage2_rankings if stage2_rankings else stage1_rankings

    write_json_atomic(
        resolved_output_dir / "leaderboard.json",
        final_rankings,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    _write_leaderboard_csv(final_rankings, resolved_output_dir / "leaderboard.csv")

    write_json_atomic(
        resolved_output_dir / "best_configs.json",
        _best_configs_payload(final_rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    all_metric_records: List[Dict[str, object]] = []
    for stage_result in (stage1_results + stage2_results):
        if stage_result.get("status") == "ok":
            all_metric_records.append(cast(Dict[str, object], stage_result["metric_record"]))
    write_json_atomic(
        resolved_output_dir / "summary.json",
        all_metric_records,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    if spec.analyze and all_metric_records:
        analysis = analyze_summary_data(all_metric_records)
        save_analysis(analysis, resolved_output_dir / "analysis.json")
    if spec.reporting.generate_alpha_report:
        generate_alpha_report(
            output_dir=resolved_output_dir,
            primary_metric=spec.reporting.primary_metric,
            tolerate_partial_outputs=spec.reporting.tolerate_partial_outputs,
        )

    metric_phase_manifest: Dict[str, object] = {
        "enabled": bool(spec.experimental_metric_sweep and spec.experimental_metric_sweep.enabled),
        "output_dir": None,
        "status": "disabled",
    }
    experimental_metric_spec = (
        spec.experimental_metric_sweep
        if spec.experimental_metric_sweep is not None
        else ExperimentalMetricSweepSpec(
            enabled=False,
            shortlist_top_k=spec.stages.top_k,
            formulations=[],
        )
    )
    if experimental_metric_spec.enabled:
        metric_output_dir = resolved_output_dir / "metric_experiments"
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        baseline_candidate_lookup = dict(coarse_lookup)
        baseline_candidate_lookup.update(refined_lookup)
        metric_candidates = _collect_metric_experiment_candidates(
            final_rankings=final_rankings,
            baseline_candidate_lookup=baseline_candidate_lookup,
            experimental_spec=experimental_metric_spec,
        )
        metric_candidates = _assign_candidate_ids(metric_candidates, "m1c")

        metric_eval = _evaluate_stage(
            stage_name="metric_phase",
            candidates=metric_candidates,
            datasets=prepared_datasets,
            output_dir=metric_output_dir,
            save_figures_flag=spec.save_figures,
            requested_workers=requested_workers,
            run_state=run_state,
            run_state_path=run_state_path,
        )
        metric_results = metric_eval.stage_results
        metric_rankings = score_and_rank_runs(
            metric_results, spec.scoring, dataset_ids=dataset_ids
        )
        _write_phase_summary_artifacts(
            output_dir=metric_output_dir,
            stage_name="metric_phase",
            stage_results=metric_results,
            rankings=metric_rankings,
            execution=metric_eval.execution,
            reporting=spec.reporting,
            analyze=spec.analyze,
        )
        metric_phase_manifest = {
            "enabled": True,
            "output_dir": str(metric_output_dir.resolve()),
            "status": "completed",
            "shortlist_top_k": int(experimental_metric_spec.shortlist_top_k),
            "num_formulations": len(experimental_metric_spec.formulations),
            "num_candidates": len(metric_candidates),
            "num_runs": len(metric_results),
            "num_ranked": len(metric_rankings),
        }

    write_json_atomic(
        resolved_output_dir / "experiment_manifest.json",
        {
            "generated_at": _utc_now_iso(),
            "output_dir": str(resolved_output_dir.resolve()),
            "strategies": list(spec.strategies),
            "baseline_metric_formulations": [
                asdict(item) for item in baseline_metric_formulations
            ],
            "experimental_metric_sweep": (
                None
                if spec.experimental_metric_sweep is None
                else asdict(spec.experimental_metric_sweep)
            ),
            "metric_experiments": metric_phase_manifest,
            "datasets": [_dataset_identity_payload(dataset) for dataset in prepared_datasets],
            "stages": asdict(spec.stages),
            "scoring": asdict(spec.scoring),
            "reporting": asdict(spec.reporting),
            "run_state_path": str(run_state_path),
        },
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    _persist_run_state(run_state_path, run_state)
    return resolved_output_dir


def sweep_parameters(
    views: np.ndarray,
    output_dir: Path,
    window_sizes: Sequence[int],
    max_displacements: Sequence[int],
    gauss_sigmas: Sequence[float],
    disp_gauss_sigmas: Sequence[float],
    global_thresholds: Sequence[float] | None = None,
    pixel_thresholds: Sequence[float] | None = None,
    strategy: str = "pixel",
    save_figures_flag: bool = False,
    zero_fill_roll: bool = True,
    analyze: bool = False,
    live_plot: bool = False,
    hdr_enabled: bool = False,
    hdr_exposure_times: Sequence[float] | None = None,
    combined_use_global_avg_reference: bool = True,
    normalize_to_8bit: bool = False,
) -> None:
    default_config = RectificationRunConfig()
    global_thresholds = list(global_thresholds or [default_config.global_threshold])
    pixel_thresholds = list(pixel_thresholds or [default_config.pixel_threshold])

    if strategy == "global":
        combos = [
            (win, max_disp, gauss, disp_gauss, global_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for global_thresh in global_thresholds
        ]
    elif strategy == "pixel":
        combos = [
            (win, max_disp, gauss, disp_gauss, pixel_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for pixel_thresh in pixel_thresholds
        ]
    else:
        combos = [
            (win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for global_thresh in global_thresholds
            for pixel_thresh in pixel_thresholds
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []
    plotter: LiveMetricsPlotter | None = None

    for combo in tqdm(combos, desc="Sweeping parameters", unit="combo"):
        if strategy == "global":
            win, max_disp, gauss, disp_gauss, global_thresh = combo  # type: ignore[misc]
            pixel_thresh = default_config.pixel_threshold
        elif strategy == "pixel":
            win, max_disp, gauss, disp_gauss, pixel_thresh = combo  # type: ignore[misc]
            global_thresh = default_config.global_threshold
        else:
            win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh = combo  # type: ignore[misc]
        # Keep max_displacement explicit and independent from the patch window size
        # to match RectificationRunConfig semantics across rectification entrypoints.
        config = RectificationRunConfig(
            strategy=strategy,
            similarity_window_size=win,
            refinement_window_size=win,
            max_displacement=max_disp,
            gauss_sigma=gauss,
            gauss_size=int(gauss * 5),
            disp_gauss_sigma=disp_gauss,
            disp_gauss_size=win,
            zero_fill_roll=zero_fill_roll,
            global_threshold=global_thresh,
            pixel_threshold=pixel_thresh,
            hdr_enabled=hdr_enabled,
            hdr_exposure_times=list(hdr_exposure_times) if hdr_exposure_times is not None else None,
            combined_use_global_avg_reference=combined_use_global_avg_reference,
            normalize_to_8bit=normalize_to_8bit,
        )

        result = rectify_views_from_config(views, config)
        run_name_parts = [
            f"win{win}",
            f"disp{max_disp}",
            f"g{gauss}",
            f"dg{disp_gauss}",
        ]
        if strategy in {"global", "combined"}:
            run_name_parts.append(f"gt{global_thresh}")
        if strategy in {"pixel", "combined"}:
            run_name_parts.append(f"pt{pixel_thresh}")
        run_name = "_".join(run_name_parts)
        record = build_metric_record(run_name, config, result)
        _save_run_artifacts(output_dir, run_name, result, metric_record=record)

        if save_figures_flag:
            figures = generate_rectification_figures(result)  # pass full result so we can save global/pixel maps, originals, overlays, averages
            figures_dir = output_dir / run_name / "figures"
            _persist_figures(figures, figures_dir)

        summary.append(record)

        if live_plot:
            if plotter is None:
                metrics_dict = record["global_and_local_metrics"]
                metric_keys = list(metrics_dict.keys()) if isinstance(metrics_dict, dict) else []
                plotter = LiveMetricsPlotter(metric_keys)
            plotter.update(run_name, record)

    write_json_atomic(
        output_dir / "summary.json",
        summary,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    if analyze:
        analysis = analyze_summary_data(summary)
        save_analysis(analysis, output_dir / "analysis.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep im_process rectification parameters")
    parser.add_argument(
        "--spec",
        help=(
            "Path to a JSON experiment spec for the two-stage multi-dataset sweep "
            "(recommended path for new workflows)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override parallel worker count in --spec mode.",
    )
    parser.add_argument(
        "--views",
        help="Path to a precomputed (N, N, H, W) or (E, N, N, H, W) numpy array of views",
    )
    parser.add_argument("--source-image", help="Path to the source image to extract views from")
    parser.add_argument(
        "--source-image-dir",
        help="Directory of exposure images to extract HDR views from (used with --hdr-assemble)",
    )
    parser.add_argument("--center-x", type=int, help="X coordinate of the central view")
    parser.add_argument("--center-y", type=int, help="Y coordinate of the central view")
    parser.add_argument("--view-radius", type=int, help="Radius of a single view in pixels")
    parser.add_argument("--view-grid-size", type=int, default=3, help="Number of views per dimension to extract")
    parser.add_argument("--view-distance", type=int, help="Override the distance between adjacent view centers")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write sweep artifacts (defaults to sweep_runs/<timestamp>)",
    )
    parser.add_argument("--strategy", choices=["global", "pixel", "combined"], default="pixel")
    parser.add_argument(
        "--window-start",
        type=int,
        default=15,
        help="Pixel-wise patch window size sweep start (pixel / combined refinement)",
    )
    parser.add_argument(
        "--window-stop",
        type=int,
        default=45,
        help="Pixel-wise patch window size sweep stop",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=10,
        help="Pixel-wise patch window size sweep step",
    )
    parser.add_argument(
        "--max-disp-start",
        type=int,
        default=5,
        help=(
            "Explicit pixel-wise maximum displacement sweep start "
            "(independent from window size; maps to RectificationRunConfig.max_displacement)"
        ),
    )
    parser.add_argument(
        "--max-disp-stop",
        type=int,
        default=25,
        help="Explicit pixel-wise maximum displacement sweep stop",
    )
    parser.add_argument(
        "--max-disp-step",
        type=int,
        default=10,
        help="Explicit pixel-wise maximum displacement sweep step",
    )
    parser.add_argument("--gauss-sigma-start", type=float, default=3.0)
    parser.add_argument("--gauss-sigma-stop", type=float, default=7.0)
    parser.add_argument("--gauss-sigma-step", type=float, default=2.0)
    parser.add_argument("--disp-gauss-start", type=float, default=0.0)
    parser.add_argument("--disp-gauss-stop", type=float, default=1.0)
    parser.add_argument("--disp-gauss-step", type=float, default=1.0)
    parser.add_argument("--global-thresh-start", type=float, default=0.15)
    parser.add_argument("--global-thresh-stop", type=float, default=0.15)
    parser.add_argument("--global-thresh-step", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-start", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-stop", type=float, default=0.05)
    parser.add_argument("--pixel-thresh-step", type=float, default=0.05)
    parser.add_argument("--zero-fill-roll",
                         action="store_true",
                         help="Use zero-filled rolling window in pixelwise rectification")
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save rectification figures (non-interactive) for each run",
    )
    parser.add_argument(
        "--hdr-assemble",
        action="store_true",
        help="Assemble HDR views from a stack of exposures shaped (E, N, N, H, W)",
    )
    parser.add_argument(
        "--hdr-exposure-times",
        help="Comma-separated exposure times (seconds) matching the exposure count",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Compute aggregate statistics for sweep metrics after the run",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Display a live graph of tracked metrics while the sweep runs",
    )
    parser.add_argument(
        "--use-global-avg-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the averaged global-rectified image as the pixel-wise reference when"
            " running the combined strategy"
        ),
    )
    parser.add_argument(
        "--normalize-to-8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize source images to 8-bit before view extraction and HDR assembly. "
            "Disabled by default to preserve raw intensities."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch a minimal GUI for configuring the sweep instead of using CLI flags",
    )
    return parser.parse_args(argv)


def _run_from_namespace(args: argparse.Namespace) -> Path:
    spec_path = getattr(args, "spec", None)
    if spec_path:
        spec = _parse_experiment_spec(Path(spec_path))
        if args.output_dir is not None:
            spec.output_dir = args.output_dir
        workers_override = getattr(args, "workers", None)
        if workers_override is not None:
            spec.stages.max_workers = int(workers_override)
        return run_sweep_experiment(spec)

    if args.save_figures and not args.live_plot:
        plt.switch_backend("Agg")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hdr_assemble and args.views is None and args.source_image_dir is None:
        raise SystemExit(
            "HDR assembly requires --views with shape (E, N, N, H, W) or --source-image-dir."
        )
    if not args.hdr_assemble and args.source_image_dir is not None:
        raise SystemExit("--source-image-dir is only supported with --hdr-assemble.")

    if args.views is None:
        if args.hdr_assemble:
            missing = [
                name
                for name, present in [
                    ("--center-x", args.center_x),
                    ("--center-y", args.center_y),
                    ("--view-radius", args.view_radius),
                ]
                if present is None
            ]
            if missing:
                missing_str = ", ".join(missing)
                raise SystemExit(
                    "When --views is not provided, the following arguments are required: "
                    f"{missing_str}."
                )
            views, cached_views_path = _generate_exposure_views(
                image_dir=Path(args.source_image_dir),
                center_x=args.center_x,
                center_y=args.center_y,
                radius=args.view_radius,
                grid_size=args.view_grid_size,
                view_distance=args.view_distance,
                cache_dir=output_dir,
                normalize_to_8bit=args.normalize_to_8bit,
            )
            print(f"Generated exposure views once and cached to {cached_views_path}")
        else:
            missing = [
                name
                for name, present in [
                    ("--source-image", args.source_image),
                    ("--center-x", args.center_x),
                    ("--center-y", args.center_y),
                    ("--view-radius", args.view_radius),
                ]
                if present is None
            ]
            if missing:
                missing_str = ", ".join(missing)
                raise SystemExit(
                    "When --views is not provided, the following arguments are required: "
                    f"{missing_str}."
                )

            views, cached_views_path = _generate_views(
                image_path=Path(args.source_image),
                center_x=args.center_x,
                center_y=args.center_y,
                radius=args.view_radius,
                grid_size=args.view_grid_size,
                view_distance=args.view_distance,
                cache_dir=output_dir,
                normalize_to_8bit=args.normalize_to_8bit,
            )
            print(f"Generated views once and cached to {cached_views_path}")
    else:
        cached_views_path = Path(args.views)
        views = _load_views(cached_views_path)

    if args.hdr_assemble and views.ndim != 5:
        raise SystemExit("HDR assembly requires views shaped (E, N, N, H, W).")
    if not args.hdr_assemble and views.ndim == 5:
        raise SystemExit("Views include multiple exposures; enable --hdr-assemble to use HDR assembly.")
    exposure_times = _parse_float_list(args.hdr_exposure_times)
    if exposure_times is not None and views.ndim == 5:
        if len(exposure_times) != views.shape[0]:
            raise SystemExit(
                "HDR exposure times must match the number of exposures in --views or --source-image-dir."
            )

    window_sizes = _build_int_range(args.window_start, args.window_stop, args.window_step)
    max_displacements = _build_int_range(args.max_disp_start, args.max_disp_stop, args.max_disp_step)
    gauss_sigmas = _build_float_range(args.gauss_sigma_start, args.gauss_sigma_stop, args.gauss_sigma_step)
    disp_gauss_sigmas = _build_float_range(args.disp_gauss_start, args.disp_gauss_stop, args.disp_gauss_step)
    global_thresholds = _build_float_range(
        args.global_thresh_start, args.global_thresh_stop, args.global_thresh_step
    )
    pixel_thresholds = _build_float_range(
        args.pixel_thresh_start, args.pixel_thresh_stop, args.pixel_thresh_step
    )

    sweep_parameters(
        views=views,
        output_dir=output_dir,
        window_sizes=window_sizes,
        max_displacements=max_displacements,
        gauss_sigmas=gauss_sigmas,
        disp_gauss_sigmas=disp_gauss_sigmas,
        global_thresholds=global_thresholds,
        pixel_thresholds=pixel_thresholds,
        strategy=args.strategy,
        save_figures_flag=args.save_figures,
        zero_fill_roll=args.zero_fill_roll,
        analyze=args.analyze,
        live_plot=args.live_plot,
        hdr_enabled=args.hdr_assemble,
        hdr_exposure_times=exposure_times,
        combined_use_global_avg_reference=args.use_global_avg_reference,
        normalize_to_8bit=args.normalize_to_8bit,
    )
    return output_dir


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return [float(item.strip()) for item in cleaned.split(",") if item.strip()]


@dataclass
class SpecGuiDatasetState:
    dataset_id: str = ""
    source_type: str = "views"
    source_path: str = ""
    center_x: str = ""
    center_y: str = ""
    view_radius: str = ""
    view_grid_size: str = "3"
    view_distance: str = ""
    hdr_enabled: bool = False
    hdr_exposure_times: str = ""
    normalize_to_8bit: bool = False


@dataclass
class SpecGuiMetricExperimentRowState:
    formulation_id: str = "exp_grad_alpha"
    family: str = "ncc_gradient"
    alphas: str = "0.0,0.5,1.0"
    gradient_weight: str = "0.5"
    laplacian_weight: str = "0.5"


@dataclass
class SpecGuiState:
    output_dir: str = ""
    strategy_pixel: bool = True
    strategy_combined: bool = True
    strategy_global: bool = False
    total_budget_per_dataset: str = "320"
    stage1_fraction: str = "0.6"
    top_k: str = "8"
    max_workers: str = ""
    analyze: bool = True
    save_figures: bool = False
    pixel_similarity_window_sizes: str = "15,25,35"
    pixel_max_displacements: str = "5,15"
    pixel_gauss_sigmas: str = "3.0,5.0"
    pixel_disp_gauss_sigmas: str = "0.0,1.0"
    pixel_thresholds: str = "0.05"
    combined_refinement_window_sizes: str = "15,25"
    combined_max_displacements: str = "5,15"
    combined_gauss_sigmas: str = "3.0,5.0"
    combined_disp_gauss_sigmas: str = "0.0,1.0"
    combined_pixel_thresholds: str = "0.05"
    combined_global_thresholds: str = "0.1,0.15"
    combined_correlation_pad_widths: str = str(RectificationRunConfig().correlation_pad_width)
    pixel_sub_pix_factors: str = "1,2"
    pixel_sigma_pyramid: str = "false,true"
    pixel_zero_fill_roll: str = "true"
    combined_sub_pix_factors: str = "1,2"
    combined_sigma_pyramid: str = "false,true"
    combined_zero_fill_roll: str = "true"
    combined_use_global_avg_reference: str = "true"
    global_thresholds: str = "0.15"
    global_correlation_pad_widths: str = str(RectificationRunConfig().correlation_pad_width)
    weight_laplacian_variance: str = "0.25"
    weight_dwt_level2_hv_energy: str = "0.25"
    weight_sum_squared_gradient: str = "0.25"
    weight_total_variation: str = "0.25"
    runtime_penalty_weight: str = "0.0"
    robustness_lambda: str = "0.5"
    missing_dataset_penalty: str = "1.0"
    metric_experiments_enabled: bool = False
    metric_experiments_shortlist_top_k: str = "8"
    metric_experiment_rows: List[SpecGuiMetricExperimentRowState] = field(
        default_factory=lambda: [SpecGuiMetricExperimentRowState()]
    )
    datasets: List[SpecGuiDatasetState] = field(
        default_factory=lambda: [SpecGuiDatasetState(dataset_id="dataset_a")]
    )
    json_preview: str = ""


def _parse_int_list_csv(value: str, field_name: str) -> List[int]:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    try:
        result = [int(item.strip()) for item in cleaned.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain comma-separated integers.") from exc
    if not result:
        raise ValueError(f"{field_name} cannot be empty.")
    return result


def _parse_float_list_csv(value: str, field_name: str) -> List[float]:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    try:
        result = [float(item.strip()) for item in cleaned.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain comma-separated numbers.") from exc
    if not result:
        raise ValueError(f"{field_name} cannot be empty.")
    return result


def _parse_bool_list_csv(value: str, field_name: str) -> List[bool]:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    result: List[bool] = []
    for item in cleaned.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(
                f"{field_name} must contain comma-separated booleans (true/false)."
            )
        result.append(mapping[token])
    if not result:
        raise ValueError(f"{field_name} cannot be empty.")
    return result


def _list_to_csv(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def _state_metric_row_to_spec(
    row: SpecGuiMetricExperimentRowState, row_index: int
) -> Dict[str, object]:
    formulation_id = row.formulation_id.strip()
    if not formulation_id:
        raise ValueError(
            f"metric experiment row {row_index} formulation id is required."
        )
    family = row.family.strip().lower()
    if family not in {"ncc_gradient", "ncc_laplacian", "composite"}:
        raise ValueError(
            f"metric experiment row {row_index} family must be one of "
            "'ncc_gradient', 'ncc_laplacian', 'composite'."
        )
    alphas = _parse_float_list_csv(
        row.alphas, f"metric experiment row {row_index} alphas"
    )
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0 or not np.isfinite(alpha):
            raise ValueError(
                f"metric experiment row {row_index} alphas must be finite and within [0, 1]."
            )

    formulation_payload: Dict[str, object] = {
        "id": formulation_id,
        "family": family,
        "alphas": alphas,
    }
    if family == "composite":
        gradient_weight = float(row.gradient_weight.strip())
        laplacian_weight = float(row.laplacian_weight.strip())
        if not np.isfinite(gradient_weight) or not np.isfinite(laplacian_weight):
            raise ValueError(
                f"metric experiment row {row_index} composite weights must be finite."
            )
        if gradient_weight < 0.0 or laplacian_weight < 0.0:
            raise ValueError(
                f"metric experiment row {row_index} composite weights must be non-negative."
            )
        if any(float(alpha) < 1.0 for alpha in alphas) and (gradient_weight + laplacian_weight <= 0.0):
            raise ValueError(
                f"metric experiment row {row_index} composite weights must sum to a positive value when alpha < 1."
            )
        formulation_payload["component_weights"] = {
            "gradient": gradient_weight,
            "laplacian": laplacian_weight,
        }
    return formulation_payload


def _metric_formulation_payload_to_state_row(
    payload: Dict[str, object], default_idx: int
) -> SpecGuiMetricExperimentRowState:
    family = str(payload.get("family", "ncc_gradient")).strip().lower()
    component_weights = cast(Optional[Dict[str, object]], payload.get("component_weights"))
    gradient_weight = "0.5"
    laplacian_weight = "0.5"
    if component_weights is not None:
        gradient_weight = str(component_weights.get("gradient", 0.0))
        laplacian_weight = str(component_weights.get("laplacian", 0.0))
    return SpecGuiMetricExperimentRowState(
        formulation_id=str(payload.get("id") or payload.get("formulation_id") or f"formulation_{default_idx:02d}"),
        family=family,
        alphas=_list_to_csv(cast(List[object], payload.get("alphas", []))) or "0.5",
        gradient_weight=gradient_weight,
        laplacian_weight=laplacian_weight,
    )


def state_to_spec_dict(state: SpecGuiState) -> Dict[str, object]:
    """Convert GUI state into canonical JSON payload for SweepExperimentSpec."""
    strategies: List[str] = []
    if state.strategy_pixel:
        strategies.append("pixel")
    if state.strategy_combined:
        strategies.append("combined")
    if state.strategy_global:
        strategies.append("global")
    if not strategies:
        strategies = ["pixel", "combined"]

    datasets: List[Dict[str, object]] = []
    for row in state.datasets:
        entry: Dict[str, object] = {"id": row.dataset_id.strip()}
        path = row.source_path.strip()
        if row.source_type == "views":
            entry["views"] = path
        elif row.source_type == "source_image":
            entry["source_image"] = path
        elif row.source_type == "source_image_dir":
            entry["source_image_dir"] = path
        else:
            entry["views"] = path

        if row.center_x.strip():
            entry["center_x"] = int(row.center_x.strip())
        if row.center_y.strip():
            entry["center_y"] = int(row.center_y.strip())
        if row.view_radius.strip():
            entry["view_radius"] = int(row.view_radius.strip())
        if row.view_grid_size.strip():
            entry["view_grid_size"] = int(row.view_grid_size.strip())
        if row.view_distance.strip():
            entry["view_distance"] = int(row.view_distance.strip())
        entry["hdr_enabled"] = bool(row.hdr_enabled)
        if row.hdr_exposure_times.strip():
            entry["hdr_exposure_times"] = _parse_float_list_csv(
                row.hdr_exposure_times, f"dataset '{row.dataset_id}' exposure times"
            )
        entry["normalize_to_8bit"] = bool(row.normalize_to_8bit)
        datasets.append(entry)

    search_spaces: Dict[str, Dict[str, object]] = {}
    if "pixel" in strategies:
        search_spaces["pixel"] = {
            "similarity_window_sizes": _parse_int_list_csv(
                state.pixel_similarity_window_sizes, "pixel window sizes"
            ),
            "max_displacements": _parse_int_list_csv(
                state.pixel_max_displacements, "pixel max displacements"
            ),
            "gauss_sigmas": _parse_float_list_csv(
                state.pixel_gauss_sigmas, "pixel gauss sigmas"
            ),
            "disp_gauss_sigmas": _parse_float_list_csv(
                state.pixel_disp_gauss_sigmas, "pixel displacement gauss sigmas"
            ),
            "pixel_thresholds": _parse_float_list_csv(
                state.pixel_thresholds, "pixel thresholds"
            ),
            "sub_pix_factors": _parse_int_list_csv(
                state.pixel_sub_pix_factors, "pixel sub-pixel factors"
            ),
            "pixel_sigma_pyramid": _parse_bool_list_csv(
                state.pixel_sigma_pyramid, "pixel sigma pyramid"
            ),
            "zero_fill_roll": _parse_bool_list_csv(
                state.pixel_zero_fill_roll, "pixel zero fill roll"
            ),
        }
    if "combined" in strategies:
        search_spaces["combined"] = {
            "refinement_window_sizes": _parse_int_list_csv(
                state.combined_refinement_window_sizes, "combined refinement window sizes"
            ),
            "max_displacements": _parse_int_list_csv(
                state.combined_max_displacements, "combined max displacements"
            ),
            "gauss_sigmas": _parse_float_list_csv(
                state.combined_gauss_sigmas, "combined gauss sigmas"
            ),
            "disp_gauss_sigmas": _parse_float_list_csv(
                state.combined_disp_gauss_sigmas, "combined displacement gauss sigmas"
            ),
            "pixel_thresholds": _parse_float_list_csv(
                state.combined_pixel_thresholds, "combined pixel thresholds"
            ),
            "global_thresholds": _parse_float_list_csv(
                state.combined_global_thresholds, "combined global thresholds"
            ),
            "correlation_pad_widths": _parse_int_list_csv(
                state.combined_correlation_pad_widths, "combined correlation pad widths"
            ),
            "sub_pix_factors": _parse_int_list_csv(
                state.combined_sub_pix_factors, "combined sub-pixel factors"
            ),
            "pixel_sigma_pyramid": _parse_bool_list_csv(
                state.combined_sigma_pyramid, "combined sigma pyramid"
            ),
            "zero_fill_roll": _parse_bool_list_csv(
                state.combined_zero_fill_roll, "combined zero fill roll"
            ),
            "combined_use_global_avg_reference": _parse_bool_list_csv(
                state.combined_use_global_avg_reference,
                "combined use global average reference",
            ),
        }
    if "global" in strategies:
        search_spaces["global"] = {
            "global_thresholds": _parse_float_list_csv(
                state.global_thresholds, "global thresholds"
            ),
            "correlation_pad_widths": _parse_int_list_csv(
                state.global_correlation_pad_widths, "global correlation pad widths"
            ),
        }

    payload: Dict[str, object] = {
        "output_dir": state.output_dir.strip() or None,
        "strategies": strategies,
        "datasets": datasets,
        "search_spaces": search_spaces,
        "stages": {
            "total_budget_per_dataset": int(state.total_budget_per_dataset.strip()),
            "stage1_fraction": float(state.stage1_fraction.strip()),
            "top_k": int(state.top_k.strip()),
            "max_workers": (
                int(state.max_workers.strip()) if state.max_workers.strip() else None
            ),
        },
        "scoring": {
            "metric_weights": {
                "laplacian_variance": float(state.weight_laplacian_variance.strip()),
                "dwt_level2_hv_energy": float(
                    state.weight_dwt_level2_hv_energy.strip()
                ),
                "sum_squared_gradient": float(
                    state.weight_sum_squared_gradient.strip()
                ),
                "total_variation": float(state.weight_total_variation.strip()),
            },
            "runtime_penalty_weight": float(state.runtime_penalty_weight.strip()),
            "robustness_lambda": float(state.robustness_lambda.strip()),
            "missing_dataset_penalty": float(state.missing_dataset_penalty.strip()),
        },
        "save_figures": bool(state.save_figures),
        "analyze": bool(state.analyze),
    }
    if state.metric_experiments_enabled:
        if not state.metric_experiment_rows:
            raise ValueError(
                "At least one metric experiment formulation row is required when enabled."
            )
        payload["experimental_metric_sweep"] = {
            "enabled": True,
            "shortlist_top_k": int(state.metric_experiments_shortlist_top_k.strip()),
            "formulations": [
                _state_metric_row_to_spec(row, idx + 1)
                for idx, row in enumerate(state.metric_experiment_rows)
            ],
        }
    return payload


def spec_dict_to_state(payload: Dict[str, object]) -> SpecGuiState:
    """Convert JSON payload into GUI state using tolerant defaults."""
    state = SpecGuiState()
    state.output_dir = str(payload.get("output_dir") or "")

    strategies = [str(item).lower() for item in cast(List[object], payload.get("strategies", []))]
    state.strategy_pixel = "pixel" in strategies or not strategies
    state.strategy_combined = "combined" in strategies or not strategies
    state.strategy_global = "global" in strategies

    stages = cast(Dict[str, object], payload.get("stages", {}))
    state.total_budget_per_dataset = str(
        stages.get("total_budget_per_dataset", state.total_budget_per_dataset)
    )
    state.stage1_fraction = str(stages.get("stage1_fraction", state.stage1_fraction))
    state.top_k = str(stages.get("top_k", state.top_k))
    state.max_workers = (
        "" if stages.get("max_workers") is None else str(stages.get("max_workers"))
    )
    state.save_figures = bool(payload.get("save_figures", state.save_figures))
    state.analyze = bool(payload.get("analyze", state.analyze))

    search_spaces = cast(Dict[str, Dict[str, object]], payload.get("search_spaces", {}))
    pixel_space = cast(Dict[str, object], search_spaces.get("pixel", {}))
    if pixel_space:
        state.pixel_similarity_window_sizes = _list_to_csv(
            cast(List[object], pixel_space.get("similarity_window_sizes", []))
        ) or state.pixel_similarity_window_sizes
        state.pixel_max_displacements = _list_to_csv(
            cast(List[object], pixel_space.get("max_displacements", []))
        ) or state.pixel_max_displacements
        state.pixel_gauss_sigmas = _list_to_csv(
            cast(List[object], pixel_space.get("gauss_sigmas", []))
        ) or state.pixel_gauss_sigmas
        state.pixel_disp_gauss_sigmas = _list_to_csv(
            cast(List[object], pixel_space.get("disp_gauss_sigmas", []))
        ) or state.pixel_disp_gauss_sigmas
        state.pixel_thresholds = _list_to_csv(
            cast(List[object], pixel_space.get("pixel_thresholds", []))
        ) or state.pixel_thresholds
        state.pixel_sub_pix_factors = _list_to_csv(
            cast(List[object], pixel_space.get("sub_pix_factors", []))
        ) or state.pixel_sub_pix_factors
        state.pixel_sigma_pyramid = _list_to_csv(
            cast(List[object], pixel_space.get("pixel_sigma_pyramid", []))
        ) or state.pixel_sigma_pyramid
        state.pixel_zero_fill_roll = _list_to_csv(
            cast(List[object], pixel_space.get("zero_fill_roll", []))
        ) or state.pixel_zero_fill_roll

    combined_space = cast(Dict[str, object], search_spaces.get("combined", {}))
    if combined_space:
        state.combined_refinement_window_sizes = _list_to_csv(
            cast(List[object], combined_space.get("refinement_window_sizes", []))
        ) or state.combined_refinement_window_sizes
        state.combined_max_displacements = _list_to_csv(
            cast(List[object], combined_space.get("max_displacements", []))
        ) or state.combined_max_displacements
        state.combined_gauss_sigmas = _list_to_csv(
            cast(List[object], combined_space.get("gauss_sigmas", []))
        ) or state.combined_gauss_sigmas
        state.combined_disp_gauss_sigmas = _list_to_csv(
            cast(List[object], combined_space.get("disp_gauss_sigmas", []))
        ) or state.combined_disp_gauss_sigmas
        state.combined_pixel_thresholds = _list_to_csv(
            cast(List[object], combined_space.get("pixel_thresholds", []))
        ) or state.combined_pixel_thresholds
        state.combined_global_thresholds = _list_to_csv(
            cast(List[object], combined_space.get("global_thresholds", []))
        ) or state.combined_global_thresholds
        state.combined_correlation_pad_widths = _list_to_csv(
            cast(List[object], combined_space.get("correlation_pad_widths", []))
        ) or state.combined_correlation_pad_widths
        state.combined_sub_pix_factors = _list_to_csv(
            cast(List[object], combined_space.get("sub_pix_factors", []))
        ) or state.combined_sub_pix_factors
        state.combined_sigma_pyramid = _list_to_csv(
            cast(List[object], combined_space.get("pixel_sigma_pyramid", []))
        ) or state.combined_sigma_pyramid
        state.combined_zero_fill_roll = _list_to_csv(
            cast(List[object], combined_space.get("zero_fill_roll", []))
        ) or state.combined_zero_fill_roll
        state.combined_use_global_avg_reference = _list_to_csv(
            cast(List[object], combined_space.get("combined_use_global_avg_reference", []))
        ) or state.combined_use_global_avg_reference

    global_space = cast(Dict[str, object], search_spaces.get("global", {}))
    if global_space:
        state.global_thresholds = _list_to_csv(
            cast(List[object], global_space.get("global_thresholds", []))
        ) or state.global_thresholds
        state.global_correlation_pad_widths = _list_to_csv(
            cast(List[object], global_space.get("correlation_pad_widths", []))
        ) or state.global_correlation_pad_widths

    scoring = cast(Dict[str, object], payload.get("scoring", {}))
    metric_weights = cast(Dict[str, object], scoring.get("metric_weights", {}))
    if metric_weights:
        state.weight_laplacian_variance = str(
            metric_weights.get("laplacian_variance", state.weight_laplacian_variance)
        )
        state.weight_dwt_level2_hv_energy = str(
            metric_weights.get("dwt_level2_hv_energy", state.weight_dwt_level2_hv_energy)
        )
        state.weight_sum_squared_gradient = str(
            metric_weights.get("sum_squared_gradient", state.weight_sum_squared_gradient)
        )
        state.weight_total_variation = str(
            metric_weights.get("total_variation", state.weight_total_variation)
        )
    state.runtime_penalty_weight = str(
        scoring.get("runtime_penalty_weight", state.runtime_penalty_weight)
    )
    state.robustness_lambda = str(scoring.get("robustness_lambda", state.robustness_lambda))
    state.missing_dataset_penalty = str(
        scoring.get("missing_dataset_penalty", state.missing_dataset_penalty)
    )

    experimental_payload = cast(
        Optional[Dict[str, object]], payload.get("experimental_metric_sweep")
    )
    formulations_payload: List[Dict[str, object]] = []
    if experimental_payload is not None:
        state.metric_experiments_enabled = bool(experimental_payload.get("enabled", False))
        shortlist_top_k = experimental_payload.get("shortlist_top_k")
        if shortlist_top_k is None:
            shortlist_top_k = stages.get("top_k", state.metric_experiments_shortlist_top_k)
        state.metric_experiments_shortlist_top_k = str(shortlist_top_k)
        raw_formulations = cast(
            List[object], experimental_payload.get("formulations", [])
        )
        formulations_payload = [
            cast(Dict[str, object], item) for item in raw_formulations if isinstance(item, dict)
        ]
    elif "metric_formulations" in payload:
        raw_formulations = cast(List[object], payload.get("metric_formulations", []))
        formulations_payload = [
            cast(Dict[str, object], item) for item in raw_formulations if isinstance(item, dict)
        ]
        state.metric_experiments_enabled = bool(formulations_payload)
        state.metric_experiments_shortlist_top_k = str(stages.get("top_k", state.top_k))
    else:
        state.metric_experiments_enabled = False
        state.metric_experiments_shortlist_top_k = str(stages.get("top_k", state.top_k))

    if formulations_payload:
        state.metric_experiment_rows = [
            _metric_formulation_payload_to_state_row(item, idx + 1)
            for idx, item in enumerate(formulations_payload)
        ]
    else:
        state.metric_experiment_rows = [SpecGuiMetricExperimentRowState()]

    datasets_payload = cast(List[Dict[str, object]], payload.get("datasets", []))
    datasets: List[SpecGuiDatasetState] = []
    for entry in datasets_payload:
        source_type = "views"
        source_path = ""
        if entry.get("source_image") is not None:
            source_type = "source_image"
            source_path = str(entry.get("source_image") or "")
        elif entry.get("source_image_dir") is not None:
            source_type = "source_image_dir"
            source_path = str(entry.get("source_image_dir") or "")
        else:
            source_path = str(entry.get("views") or entry.get("views_path") or "")
        row = SpecGuiDatasetState(
            dataset_id=str(entry.get("id") or entry.get("dataset_id") or ""),
            source_type=source_type,
            source_path=source_path,
            center_x="" if entry.get("center_x") is None else str(entry.get("center_x")),
            center_y="" if entry.get("center_y") is None else str(entry.get("center_y")),
            view_radius=""
            if entry.get("view_radius") is None
            else str(entry.get("view_radius")),
            view_grid_size=str(entry.get("view_grid_size", 3)),
            view_distance=""
            if entry.get("view_distance") is None
            else str(entry.get("view_distance")),
            hdr_enabled=bool(entry.get("hdr_enabled", False)),
            hdr_exposure_times=_list_to_csv(
                cast(List[object], entry.get("hdr_exposure_times", []))
            ),
            normalize_to_8bit=bool(entry.get("normalize_to_8bit", False)),
        )
        datasets.append(row)
    if datasets:
        state.datasets = datasets
    state.json_preview = json.dumps(payload, indent=2)
    return state


def validate_state(state: SpecGuiState) -> List[str]:
    """Validate GUI state before save/run and return list of issues."""
    issues: List[str] = []
    strategies = [
        strategy
        for strategy, enabled in [
            ("pixel", state.strategy_pixel),
            ("combined", state.strategy_combined),
            ("global", state.strategy_global),
        ]
        if enabled
    ]
    if not strategies:
        issues.append("Select at least one strategy.")

    try:
        if int(state.total_budget_per_dataset.strip()) < 1:
            issues.append("Total budget per dataset must be >= 1.")
    except Exception:
        issues.append("Total budget per dataset must be an integer.")

    try:
        fraction = float(state.stage1_fraction.strip())
        if not (0.0 < fraction < 1.0):
            issues.append("Stage1 fraction must be between 0 and 1.")
    except Exception:
        issues.append("Stage1 fraction must be numeric.")

    try:
        if int(state.top_k.strip()) < 1:
            issues.append("Top-k must be >= 1.")
    except Exception:
        issues.append("Top-k must be an integer.")

    if state.max_workers.strip():
        try:
            if int(state.max_workers.strip()) < 1:
                issues.append("Max workers must be >= 1 when provided.")
        except Exception:
            issues.append("Max workers must be an integer when provided.")

    if not state.datasets:
        issues.append("At least one dataset row is required.")
    ids = [row.dataset_id.strip() for row in state.datasets if row.dataset_id.strip()]
    if len(ids) != len(set(ids)):
        issues.append("Dataset IDs must be unique.")

    for idx, row in enumerate(state.datasets, start=1):
        prefix = f"Dataset row {idx}"
        if not row.dataset_id.strip():
            issues.append(f"{prefix}: dataset id is required.")
        if row.source_type not in {"views", "source_image", "source_image_dir"}:
            issues.append(f"{prefix}: invalid source type '{row.source_type}'.")
        if not row.source_path.strip():
            issues.append(f"{prefix}: source path is required.")
        if row.source_type in {"source_image", "source_image_dir"}:
            for field_name, value in [
                ("center_x", row.center_x),
                ("center_y", row.center_y),
                ("view_radius", row.view_radius),
            ]:
                if not value.strip():
                    issues.append(
                        f"{prefix}: {field_name} is required for {row.source_type}."
                    )
                else:
                    try:
                        int(value.strip())
                    except Exception:
                        issues.append(f"{prefix}: {field_name} must be an integer.")
        if row.view_grid_size.strip():
            try:
                if int(row.view_grid_size.strip()) < 1:
                    issues.append(f"{prefix}: view_grid_size must be >= 1.")
            except Exception:
                issues.append(f"{prefix}: view_grid_size must be an integer.")
        if row.view_distance.strip():
            try:
                int(row.view_distance.strip())
            except Exception:
                issues.append(f"{prefix}: view_distance must be an integer.")
        if row.hdr_exposure_times.strip():
            try:
                _parse_float_list_csv(
                    row.hdr_exposure_times, f"{prefix} hdr_exposure_times"
                )
            except ValueError as exc:
                issues.append(str(exc))

    parser_checks: List[Tuple[str, str, str]] = []
    if state.strategy_pixel:
        parser_checks.extend(
            [
                ("pixel window sizes", state.pixel_similarity_window_sizes, "int"),
                ("pixel max displacements", state.pixel_max_displacements, "int"),
                ("pixel gauss sigmas", state.pixel_gauss_sigmas, "float"),
                ("pixel disp gauss sigmas", state.pixel_disp_gauss_sigmas, "float"),
                ("pixel thresholds", state.pixel_thresholds, "float"),
                ("pixel sub-pixel factors", state.pixel_sub_pix_factors, "int"),
                ("pixel sigma pyramid", state.pixel_sigma_pyramid, "bool"),
                ("pixel zero fill roll", state.pixel_zero_fill_roll, "bool"),
            ]
        )
    if state.strategy_combined:
        parser_checks.extend(
            [
                ("combined refinement windows", state.combined_refinement_window_sizes, "int"),
                ("combined max displacements", state.combined_max_displacements, "int"),
                ("combined gauss sigmas", state.combined_gauss_sigmas, "float"),
                ("combined disp gauss sigmas", state.combined_disp_gauss_sigmas, "float"),
                ("combined pixel thresholds", state.combined_pixel_thresholds, "float"),
                ("combined global thresholds", state.combined_global_thresholds, "float"),
                ("combined correlation pad widths", state.combined_correlation_pad_widths, "int"),
                ("combined sub-pixel factors", state.combined_sub_pix_factors, "int"),
                ("combined sigma pyramid", state.combined_sigma_pyramid, "bool"),
                ("combined zero fill roll", state.combined_zero_fill_roll, "bool"),
                ("combined global avg reference", state.combined_use_global_avg_reference, "bool"),
            ]
        )
    if state.strategy_global:
        parser_checks.extend(
            [
                ("global thresholds", state.global_thresholds, "float"),
                ("global correlation pad widths", state.global_correlation_pad_widths, "int"),
            ]
        )
    for field_name, value, field_type in parser_checks:
        try:
            if field_type == "int":
                _parse_int_list_csv(value, field_name)
            elif field_type == "float":
                _parse_float_list_csv(value, field_name)
            else:
                _parse_bool_list_csv(value, field_name)
        except ValueError as exc:
            issues.append(str(exc))

    try:
        weights = [
            float(state.weight_laplacian_variance.strip()),
            float(state.weight_dwt_level2_hv_energy.strip()),
            float(state.weight_sum_squared_gradient.strip()),
            float(state.weight_total_variation.strip()),
        ]
        if sum(weights) <= 0:
            issues.append("Metric weights must sum to a positive value.")
    except Exception:
        issues.append("Metric weights must be numeric.")

    for label, raw in [
        ("runtime penalty weight", state.runtime_penalty_weight),
        ("robustness lambda", state.robustness_lambda),
        ("missing dataset penalty", state.missing_dataset_penalty),
    ]:
        try:
            float(raw.strip())
        except Exception:
            issues.append(f"{label} must be numeric.")

    if state.metric_experiments_enabled:
        if not state.strategy_pixel and not state.strategy_combined:
            issues.append(
                "Experimental metrics require at least one pixel/combined strategy."
            )
        try:
            if int(state.metric_experiments_shortlist_top_k.strip()) < 1:
                issues.append(
                    "Metric experiment shortlist top-k must be >= 1."
                )
        except Exception:
            issues.append("Metric experiment shortlist top-k must be an integer.")

        if not state.metric_experiment_rows:
            issues.append(
                "At least one metric experiment formulation row is required when enabled."
            )
        formulation_ids: List[str] = []
        for idx, row in enumerate(state.metric_experiment_rows, start=1):
            formulation_id = row.formulation_id.strip()
            if not formulation_id:
                issues.append(f"Metric experiment row {idx}: id is required.")
            else:
                formulation_ids.append(formulation_id)

            family = row.family.strip().lower()
            if family not in {"ncc_gradient", "ncc_laplacian", "composite"}:
                issues.append(
                    f"Metric experiment row {idx}: invalid family '{row.family}'."
                )
                continue
            try:
                alphas = _parse_float_list_csv(
                    row.alphas, f"Metric experiment row {idx} alphas"
                )
                if any(
                    (not np.isfinite(alpha)) or alpha < 0.0 or alpha > 1.0
                    for alpha in alphas
                ):
                    issues.append(
                        f"Metric experiment row {idx}: alphas must be finite and within [0, 1]."
                    )
            except ValueError as exc:
                issues.append(str(exc))
                continue

            if family == "composite":
                try:
                    gradient_weight = float(row.gradient_weight.strip())
                    laplacian_weight = float(row.laplacian_weight.strip())
                except Exception:
                    issues.append(
                        f"Metric experiment row {idx}: composite weights must be numeric."
                    )
                    continue
                if not np.isfinite(gradient_weight) or not np.isfinite(laplacian_weight):
                    issues.append(
                        f"Metric experiment row {idx}: composite weights must be finite."
                    )
                if gradient_weight < 0.0 or laplacian_weight < 0.0:
                    issues.append(
                        f"Metric experiment row {idx}: composite weights must be non-negative."
                    )
                if any(alpha < 1.0 for alpha in alphas) and (gradient_weight + laplacian_weight <= 0.0):
                    issues.append(
                        f"Metric experiment row {idx}: composite weights must sum to a positive value when alpha < 1."
                    )

        if len(formulation_ids) != len(set(formulation_ids)):
            issues.append("Metric experiment formulation IDs must be unique.")

    return issues


def _save_spec_payload(path: Path, payload: Dict[str, object]) -> None:
    write_json_atomic(path, payload, indent=2, sort_keys=False, create_backup=True)


def _load_spec_payload(path: Path) -> Dict[str, object]:
    payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    if not isinstance(payload, dict):
        raise ValueError("Spec file must contain a JSON object.")
    return cast(Dict[str, object], payload)


def _open_with_system(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        target = str(path)
        if platform.system().lower().startswith("win"):
            os.startfile(target)  # type: ignore[attr-defined]
            return True, None
        if platform.system().lower() == "darwin":
            subprocess.run(["open", target], check=False)
            return True, None
        subprocess.run(["xdg-open", target], check=False)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


class SpecGuiApp:
    def __init__(self, root: Any):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = __import__("tkinter.filedialog", fromlist=["filedialog"])
        self.messagebox = __import__("tkinter.messagebox", fromlist=["messagebox"])
        self.state = SpecGuiState()
        self.dataset_rows: List[Dict[str, Any]] = []
        self.metric_experiment_rows: List[Dict[str, Any]] = []
        self.editable_widgets: List[Any] = []
        self.log_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
        self.run_thread: Optional[threading.Thread] = None
        self._is_running = False
        self.status_var = tk.StringVar(value="Idle")
        self.show_advanced_var = tk.BooleanVar(value=False)

        self.root.title("Moment Images Sweep (Spec Mode)")
        self.root.geometry("1280x900")
        self._build_ui()
        self._set_state(self.state)
        self._poll_logs()

    def _register_editable(self, widget: Any) -> None:
        self.editable_widgets.append(widget)

    def _set_editable_enabled(self, enabled: bool) -> None:
        desired = "normal" if enabled else "disabled"
        for widget in self.editable_widgets:
            try:
                widget.configure(state=desired)
            except Exception:
                pass
        if enabled and hasattr(self, "metric_experiments_enabled_var"):
            self._toggle_metric_experiment_controls()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(self.tk.END, message + "\n")
        self.log_text.see(self.tk.END)
        self.log_text.configure(state="disabled")

    def _build_labeled_entry(
        self, parent: Any, row: int, label_text: str, default_value: str, width: int = 22
    ) -> Any:
        self.ttk.Label(parent, text=label_text).grid(
            row=row, column=0, sticky="w", padx=4, pady=2
        )
        entry = self.ttk.Entry(parent, width=width)
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        self._register_editable(entry)
        return entry

    def _build_ui(self) -> None:
        top_actions = self.ttk.Frame(self.root)
        top_actions.pack(fill="x", padx=8, pady=6)

        self.load_spec_btn = self.ttk.Button(
            top_actions, text="Load Spec...", command=self._on_load_spec
        )
        self.load_spec_btn.pack(side="left", padx=3)
        self._register_editable(self.load_spec_btn)
        self.save_spec_btn = self.ttk.Button(
            top_actions, text="Save Spec...", command=self._on_save_spec
        )
        self.save_spec_btn.pack(side="left", padx=3)
        self._register_editable(self.save_spec_btn)
        self.save_template_btn = self.ttk.Button(
            top_actions,
            text="Save Spec As Template...",
            command=lambda: self._on_save_spec(template=True),
        )
        self.save_template_btn.pack(side="left", padx=3)
        self._register_editable(self.save_template_btn)
        self.refresh_json_btn = self.ttk.Button(
            top_actions, text="Refresh JSON Preview", command=self._on_refresh_json_preview
        )
        self.refresh_json_btn.pack(side="left", padx=3)
        self._register_editable(self.refresh_json_btn)
        self.apply_json_btn = self.ttk.Button(
            top_actions, text="Apply JSON Preview", command=self._on_apply_json_preview
        )
        self.apply_json_btn.pack(side="left", padx=3)
        self._register_editable(self.apply_json_btn)

        self.run_btn = self.ttk.Button(top_actions, text="Run Sweep", command=self._on_run)
        self.run_btn.pack(side="right", padx=3)
        self._register_editable(self.run_btn)
        self.status_label = self.ttk.Label(top_actions, textvariable=self.status_var)
        self.status_label.pack(side="right", padx=12)

        main_container = self.ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=8, pady=4)

        scroll_canvas = self.tk.Canvas(main_container)
        scrollbar = self.ttk.Scrollbar(
            main_container, orient="vertical", command=scroll_canvas.yview
        )
        self.content_frame = self.ttk.Frame(scroll_canvas)
        self.content_frame.bind(
            "<Configure>",
            lambda _evt: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        basic_frame = self.ttk.LabelFrame(self.content_frame, text="Basic Settings")
        basic_frame.pack(fill="x", pady=6)
        self.output_dir_entry = self._build_labeled_entry(
            basic_frame, 0, "Output Directory", ""
        )
        browse_output_btn = self.ttk.Button(
            basic_frame, text="Browse...", command=self._on_browse_output_dir
        )
        browse_output_btn.grid(row=0, column=2, sticky="w", padx=4, pady=2)
        self._register_editable(browse_output_btn)

        strat_frame = self.ttk.Frame(basic_frame)
        strat_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self.ttk.Label(strat_frame, text="Strategies:").pack(side="left")
        self.strategy_pixel_var = self.tk.BooleanVar(value=True)
        self.strategy_combined_var = self.tk.BooleanVar(value=True)
        self.strategy_global_var = self.tk.BooleanVar(value=False)
        for text, var in [
            ("Pixel", self.strategy_pixel_var),
            ("Combined", self.strategy_combined_var),
            ("Global", self.strategy_global_var),
        ]:
            chk = self.ttk.Checkbutton(strat_frame, text=text, variable=var)
            chk.pack(side="left", padx=5)
            self._register_editable(chk)

        stage_frame = self.ttk.Frame(basic_frame)
        stage_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self.ttk.Label(stage_frame, text="Total budget").grid(row=0, column=0, sticky="w")
        self.total_budget_entry = self.ttk.Entry(stage_frame, width=8)
        self.total_budget_entry.grid(row=0, column=1, padx=4)
        self._register_editable(self.total_budget_entry)
        self.ttk.Label(stage_frame, text="Stage1 fraction").grid(row=0, column=2, sticky="w")
        self.stage1_fraction_entry = self.ttk.Entry(stage_frame, width=6)
        self.stage1_fraction_entry.grid(row=0, column=3, padx=4)
        self._register_editable(self.stage1_fraction_entry)
        self.ttk.Label(stage_frame, text="Top-K").grid(row=0, column=4, sticky="w")
        self.top_k_entry = self.ttk.Entry(stage_frame, width=6)
        self.top_k_entry.grid(row=0, column=5, padx=4)
        self._register_editable(self.top_k_entry)
        self.ttk.Label(stage_frame, text="Max workers").grid(row=0, column=6, sticky="w")
        self.max_workers_entry = self.ttk.Entry(stage_frame, width=8)
        self.max_workers_entry.grid(row=0, column=7, padx=4)
        self._register_editable(self.max_workers_entry)

        toggles_frame = self.ttk.Frame(basic_frame)
        toggles_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self.analyze_var = self.tk.BooleanVar(value=True)
        self.save_figures_var = self.tk.BooleanVar(value=False)
        analyze_chk = self.ttk.Checkbutton(toggles_frame, text="Analyze", variable=self.analyze_var)
        analyze_chk.pack(side="left", padx=5)
        self._register_editable(analyze_chk)
        save_fig_chk = self.ttk.Checkbutton(
            toggles_frame, text="Save Figures", variable=self.save_figures_var
        )
        save_fig_chk.pack(side="left", padx=5)
        self._register_editable(save_fig_chk)

        search_frame = self.ttk.LabelFrame(self.content_frame, text="Search Space (Core Lists)")
        search_frame.pack(fill="x", pady=6)
        left = self.ttk.Frame(search_frame)
        left.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        right = self.ttk.Frame(search_frame)
        right.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        self.pixel_window_entry = self._build_labeled_entry(left, 0, "Pixel window sizes", "15,25,35")
        self.pixel_disp_entry = self._build_labeled_entry(left, 1, "Pixel max displacements", "5,15")
        self.pixel_gauss_entry = self._build_labeled_entry(left, 2, "Pixel gauss sigmas", "3.0,5.0")
        self.pixel_disp_gauss_entry = self._build_labeled_entry(left, 3, "Pixel disp gauss sigmas", "0.0,1.0")
        self.pixel_thresh_entry = self._build_labeled_entry(left, 4, "Pixel thresholds", "0.05")

        self.comb_window_entry = self._build_labeled_entry(right, 0, "Combined refinement windows", "15,25")
        self.comb_disp_entry = self._build_labeled_entry(right, 1, "Combined max displacements", "5,15")
        self.comb_gauss_entry = self._build_labeled_entry(right, 2, "Combined gauss sigmas", "3.0,5.0")
        self.comb_disp_gauss_entry = self._build_labeled_entry(right, 3, "Combined disp gauss sigmas", "0.0,1.0")
        self.comb_thresh_entry = self._build_labeled_entry(right, 4, "Combined pixel thresholds", "0.05")
        self.comb_global_thresh_entry = self._build_labeled_entry(right, 5, "Combined global thresholds", "0.1,0.15")
        self.comb_corr_pad_entry = self._build_labeled_entry(
            right, 6, "Combined correlation pad widths", str(RectificationRunConfig().correlation_pad_width)
        )

        dataset_frame = self.ttk.LabelFrame(self.content_frame, text="Datasets")
        dataset_frame.pack(fill="x", pady=6)
        dataset_btns = self.ttk.Frame(dataset_frame)
        dataset_btns.pack(fill="x", padx=4, pady=4)
        add_dataset_btn = self.ttk.Button(dataset_btns, text="Add Dataset", command=self._add_dataset_row)
        add_dataset_btn.pack(side="left", padx=2)
        self._register_editable(add_dataset_btn)
        self.dataset_rows_frame = self.ttk.Frame(dataset_frame)
        self.dataset_rows_frame.pack(fill="x", padx=4, pady=4)

        advanced_container = self.ttk.LabelFrame(self.content_frame, text="Advanced Settings")
        advanced_container.pack(fill="x", pady=6)
        toggle_btn = self.ttk.Checkbutton(
            advanced_container,
            text="Show advanced controls",
            variable=self.show_advanced_var,
            command=self._toggle_advanced,
        )
        toggle_btn.pack(anchor="w", padx=4, pady=2)
        self._register_editable(toggle_btn)

        self.advanced_frame = self.ttk.Frame(advanced_container)
        adv_left = self.ttk.Frame(self.advanced_frame)
        adv_left.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        adv_right = self.ttk.Frame(self.advanced_frame)
        adv_right.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        self.pixel_subpix_entry = self._build_labeled_entry(adv_left, 0, "Pixel sub-pixel factors", "1,2")
        self.pixel_pyramid_entry = self._build_labeled_entry(adv_left, 1, "Pixel sigma pyramid", "false,true")
        self.pixel_zfill_entry = self._build_labeled_entry(adv_left, 2, "Pixel zero fill roll", "true")
        self.comb_subpix_entry = self._build_labeled_entry(adv_left, 3, "Combined sub-pixel factors", "1,2")
        self.comb_pyramid_entry = self._build_labeled_entry(adv_left, 4, "Combined sigma pyramid", "false,true")
        self.comb_zfill_entry = self._build_labeled_entry(adv_left, 5, "Combined zero fill roll", "true")
        self.comb_use_global_ref_entry = self._build_labeled_entry(adv_left, 6, "Combined use global avg ref", "true")
        self.global_thresholds_entry = self._build_labeled_entry(adv_left, 7, "Global thresholds", "0.15")
        self.global_corr_pad_entry = self._build_labeled_entry(
            adv_left, 8, "Global correlation pad widths", str(RectificationRunConfig().correlation_pad_width)
        )
        self.weight_lap_entry = self._build_labeled_entry(adv_right, 0, "Weight laplacian_variance", "0.25")
        self.weight_dwt_entry = self._build_labeled_entry(adv_right, 1, "Weight dwt_level2_hv_energy", "0.25")
        self.weight_grad_entry = self._build_labeled_entry(adv_right, 2, "Weight sum_squared_gradient", "0.25")
        self.weight_tv_entry = self._build_labeled_entry(adv_right, 3, "Weight total_variation", "0.25")
        self.runtime_penalty_entry = self._build_labeled_entry(adv_right, 4, "Runtime penalty weight", "0.0")
        self.robustness_entry = self._build_labeled_entry(adv_right, 5, "Robustness lambda", "0.5")
        self.missing_penalty_entry = self._build_labeled_entry(adv_right, 6, "Missing dataset penalty", "1.0")

        metric_exp_container = self.ttk.LabelFrame(
            self.advanced_frame, text="Experimental Metrics (Optional)"
        )
        metric_exp_container.pack(fill="x", padx=4, pady=6)
        metric_exp_controls = self.ttk.Frame(metric_exp_container)
        metric_exp_controls.pack(fill="x", padx=4, pady=2)
        self.metric_experiments_enabled_var = self.tk.BooleanVar(value=False)
        metric_enabled_chk = self.ttk.Checkbutton(
            metric_exp_controls,
            text="Enable experimental metric sweep",
            variable=self.metric_experiments_enabled_var,
            command=self._toggle_metric_experiment_controls,
        )
        metric_enabled_chk.pack(side="left", padx=4)
        self._register_editable(metric_enabled_chk)
        self.ttk.Label(metric_exp_controls, text="Shortlist top-k").pack(
            side="left", padx=(12, 2)
        )
        self.metric_shortlist_entry = self.ttk.Entry(metric_exp_controls, width=8)
        self.metric_shortlist_entry.insert(0, "8")
        self.metric_shortlist_entry.pack(side="left", padx=2)
        self._register_editable(self.metric_shortlist_entry)
        self.metric_add_row_btn = self.ttk.Button(
            metric_exp_controls,
            text="Add Formulation",
            command=self._add_metric_experiment_row,
        )
        self.metric_add_row_btn.pack(side="left", padx=8)
        self._register_editable(self.metric_add_row_btn)

        self.metric_exp_headers = self.ttk.Frame(metric_exp_container)
        self.metric_exp_headers.pack(fill="x", padx=4, pady=(2, 0))
        self.ttk.Label(self.metric_exp_headers, text="id", width=18).pack(side="left")
        self.ttk.Label(self.metric_exp_headers, text="family", width=14).pack(side="left")
        self.ttk.Label(self.metric_exp_headers, text="alphas", width=20).pack(side="left")
        self.ttk.Label(self.metric_exp_headers, text="grad w", width=10).pack(side="left")
        self.ttk.Label(self.metric_exp_headers, text="lap w", width=10).pack(side="left")

        self.metric_experiment_rows_frame = self.ttk.Frame(metric_exp_container)
        self.metric_experiment_rows_frame.pack(fill="x", padx=4, pady=2)
        self._toggle_metric_experiment_controls()

        json_frame = self.ttk.LabelFrame(self.content_frame, text="JSON Preview / Editor (Optional)")
        json_frame.pack(fill="both", expand=True, pady=6)
        self.json_text = self.tk.Text(json_frame, height=12, wrap="none")
        self.json_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._register_editable(self.json_text)

        log_frame = self.ttk.LabelFrame(self.content_frame, text="Run Log")
        log_frame.pack(fill="both", expand=True, pady=6)
        self.log_text = self.tk.Text(log_frame, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _toggle_advanced(self) -> None:
        if self.show_advanced_var.get():
            self.advanced_frame.pack(fill="x", padx=4, pady=4)
        else:
            self.advanced_frame.pack_forget()

    def _toggle_metric_experiment_controls(self) -> None:
        enabled = bool(self.metric_experiments_enabled_var.get())
        if enabled:
            self.metric_exp_headers.pack(fill="x", padx=4, pady=(2, 0))
            self.metric_experiment_rows_frame.pack(fill="x", padx=4, pady=2)
            self.metric_shortlist_entry.configure(state="normal")
            self.metric_add_row_btn.configure(state="normal")
        else:
            self.metric_exp_headers.pack_forget()
            self.metric_experiment_rows_frame.pack_forget()
            self.metric_shortlist_entry.configure(state="disabled")
            self.metric_add_row_btn.configure(state="disabled")

    def _clear_dataset_rows(self) -> None:
        for row in self.dataset_rows:
            row["frame"].destroy()
        self.dataset_rows = []

    def _add_dataset_row(self, initial: Optional[SpecGuiDatasetState] = None) -> None:
        initial_state = initial or SpecGuiDatasetState(dataset_id=f"dataset_{len(self.dataset_rows)+1}")
        row_frame = self.ttk.Frame(self.dataset_rows_frame)
        row_frame.pack(fill="x", pady=2)
        id_var = self.tk.StringVar(value=initial_state.dataset_id)
        source_type_var = self.tk.StringVar(value=initial_state.source_type)
        source_path_var = self.tk.StringVar(value=initial_state.source_path)
        center_x_var = self.tk.StringVar(value=initial_state.center_x)
        center_y_var = self.tk.StringVar(value=initial_state.center_y)
        radius_var = self.tk.StringVar(value=initial_state.view_radius)
        grid_var = self.tk.StringVar(value=initial_state.view_grid_size)
        dist_var = self.tk.StringVar(value=initial_state.view_distance)
        hdr_enabled_var = self.tk.BooleanVar(value=initial_state.hdr_enabled)
        exposure_var = self.tk.StringVar(value=initial_state.hdr_exposure_times)
        normalize_var = self.tk.BooleanVar(value=initial_state.normalize_to_8bit)

        self.ttk.Label(row_frame, text="ID").pack(side="left")
        id_entry = self.ttk.Entry(row_frame, textvariable=id_var, width=10)
        id_entry.pack(side="left", padx=2)
        self._register_editable(id_entry)
        self.ttk.Label(row_frame, text="Source").pack(side="left")
        source_menu = self.ttk.OptionMenu(
            row_frame, source_type_var, source_type_var.get(), "views", "source_image", "source_image_dir"
        )
        source_menu.pack(side="left", padx=2)
        self._register_editable(source_menu)
        path_entry = self.ttk.Entry(row_frame, textvariable=source_path_var, width=28)
        path_entry.pack(side="left", padx=2)
        self._register_editable(path_entry)
        browse_btn = self.ttk.Button(
            row_frame,
            text="Browse",
            command=lambda st=source_type_var, sp=source_path_var: self._browse_dataset_path(st, sp),
        )
        browse_btn.pack(side="left", padx=2)
        self._register_editable(browse_btn)

        for label, var, width in [("cx", center_x_var, 4), ("cy", center_y_var, 4), ("r", radius_var, 4), ("grid", grid_var, 4), ("dist", dist_var, 4)]:
            self.ttk.Label(row_frame, text=label).pack(side="left")
            entry = self.ttk.Entry(row_frame, textvariable=var, width=width)
            entry.pack(side="left", padx=2)
            self._register_editable(entry)

        hdr_chk = self.ttk.Checkbutton(row_frame, text="HDR", variable=hdr_enabled_var)
        hdr_chk.pack(side="left", padx=2)
        self._register_editable(hdr_chk)
        self.ttk.Label(row_frame, text="exp").pack(side="left")
        exp_entry = self.ttk.Entry(row_frame, textvariable=exposure_var, width=10)
        exp_entry.pack(side="left", padx=2)
        self._register_editable(exp_entry)
        norm_chk = self.ttk.Checkbutton(row_frame, text="8-bit", variable=normalize_var)
        norm_chk.pack(side="left", padx=2)
        self._register_editable(norm_chk)
        remove_btn = self.ttk.Button(
            row_frame, text="Remove", command=lambda frame=row_frame: self._remove_dataset_row(frame)
        )
        remove_btn.pack(side="left", padx=2)
        self._register_editable(remove_btn)

        self.dataset_rows.append(
            {
                "frame": row_frame,
                "id_var": id_var,
                "source_type_var": source_type_var,
                "source_path_var": source_path_var,
                "center_x_var": center_x_var,
                "center_y_var": center_y_var,
                "radius_var": radius_var,
                "grid_var": grid_var,
                "dist_var": dist_var,
                "hdr_enabled_var": hdr_enabled_var,
                "exposure_var": exposure_var,
                "normalize_var": normalize_var,
            }
        )

    def _remove_dataset_row(self, frame: Any) -> None:
        if len(self.dataset_rows) <= 1:
            self.messagebox.showwarning("Dataset", "At least one dataset is required.")
            return
        for idx, row in enumerate(self.dataset_rows):
            if row["frame"] == frame:
                row["frame"].destroy()
                del self.dataset_rows[idx]
                break

    def _clear_metric_experiment_rows(self) -> None:
        for row in self.metric_experiment_rows:
            row["frame"].destroy()
        self.metric_experiment_rows = []

    def _add_metric_experiment_row(
        self, initial: Optional[SpecGuiMetricExperimentRowState] = None
    ) -> None:
        initial_state = initial or SpecGuiMetricExperimentRowState(
            formulation_id=f"exp_formulation_{len(self.metric_experiment_rows) + 1}"
        )
        row_frame = self.ttk.Frame(self.metric_experiment_rows_frame)
        row_frame.pack(fill="x", pady=1)

        id_var = self.tk.StringVar(value=initial_state.formulation_id)
        family_var = self.tk.StringVar(value=initial_state.family)
        alphas_var = self.tk.StringVar(value=initial_state.alphas)
        gradient_weight_var = self.tk.StringVar(value=initial_state.gradient_weight)
        laplacian_weight_var = self.tk.StringVar(value=initial_state.laplacian_weight)

        id_entry = self.ttk.Entry(row_frame, textvariable=id_var, width=22)
        id_entry.pack(side="left", padx=2)
        self._register_editable(id_entry)
        family_menu = self.ttk.OptionMenu(
            row_frame,
            family_var,
            family_var.get(),
            "ncc_gradient",
            "ncc_laplacian",
            "composite",
        )
        family_menu.pack(side="left", padx=2)
        self._register_editable(family_menu)
        alphas_entry = self.ttk.Entry(row_frame, textvariable=alphas_var, width=24)
        alphas_entry.pack(side="left", padx=2)
        self._register_editable(alphas_entry)
        gradient_entry = self.ttk.Entry(row_frame, textvariable=gradient_weight_var, width=10)
        gradient_entry.pack(side="left", padx=2)
        self._register_editable(gradient_entry)
        laplacian_entry = self.ttk.Entry(row_frame, textvariable=laplacian_weight_var, width=10)
        laplacian_entry.pack(side="left", padx=2)
        self._register_editable(laplacian_entry)
        remove_btn = self.ttk.Button(
            row_frame,
            text="Remove",
            command=lambda frame=row_frame: self._remove_metric_experiment_row(frame),
        )
        remove_btn.pack(side="left", padx=4)
        self._register_editable(remove_btn)

        self.metric_experiment_rows.append(
            {
                "frame": row_frame,
                "id_var": id_var,
                "family_var": family_var,
                "alphas_var": alphas_var,
                "gradient_weight_var": gradient_weight_var,
                "laplacian_weight_var": laplacian_weight_var,
            }
        )

    def _remove_metric_experiment_row(self, frame: Any) -> None:
        if len(self.metric_experiment_rows) <= 1:
            self.messagebox.showwarning(
                "Metric Experiments", "At least one formulation row is required."
            )
            return
        for idx, row in enumerate(self.metric_experiment_rows):
            if row["frame"] == frame:
                row["frame"].destroy()
                del self.metric_experiment_rows[idx]
                break

    def _browse_dataset_path(self, source_type_var: Any, path_var: Any) -> None:
        source_type = source_type_var.get()
        if source_type == "views":
            selected = self.filedialog.askopenfilename(filetypes=[("NumPy arrays", "*.npy")])
        elif source_type == "source_image":
            selected = self.filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
            )
        else:
            selected = self.filedialog.askdirectory()
        if selected:
            path_var.set(selected)

    def _on_browse_output_dir(self) -> None:
        selected = self.filedialog.askdirectory()
        if selected:
            self.output_dir_entry.delete(0, self.tk.END)
            self.output_dir_entry.insert(0, selected)

    def _get_state(self) -> SpecGuiState:
        datasets: List[SpecGuiDatasetState] = []
        for row in self.dataset_rows:
            datasets.append(
                SpecGuiDatasetState(
                    dataset_id=row["id_var"].get().strip(),
                    source_type=row["source_type_var"].get().strip(),
                    source_path=row["source_path_var"].get().strip(),
                    center_x=row["center_x_var"].get().strip(),
                    center_y=row["center_y_var"].get().strip(),
                    view_radius=row["radius_var"].get().strip(),
                    view_grid_size=row["grid_var"].get().strip() or "3",
                    view_distance=row["dist_var"].get().strip(),
                    hdr_enabled=bool(row["hdr_enabled_var"].get()),
                    hdr_exposure_times=row["exposure_var"].get().strip(),
                    normalize_to_8bit=bool(row["normalize_var"].get()),
                )
            )

        metric_rows: List[SpecGuiMetricExperimentRowState] = []
        for row in self.metric_experiment_rows:
            metric_rows.append(
                SpecGuiMetricExperimentRowState(
                    formulation_id=row["id_var"].get().strip(),
                    family=row["family_var"].get().strip(),
                    alphas=row["alphas_var"].get().strip(),
                    gradient_weight=row["gradient_weight_var"].get().strip(),
                    laplacian_weight=row["laplacian_weight_var"].get().strip(),
                )
            )

        return SpecGuiState(
            output_dir=self.output_dir_entry.get().strip(),
            strategy_pixel=bool(self.strategy_pixel_var.get()),
            strategy_combined=bool(self.strategy_combined_var.get()),
            strategy_global=bool(self.strategy_global_var.get()),
            total_budget_per_dataset=self.total_budget_entry.get().strip(),
            stage1_fraction=self.stage1_fraction_entry.get().strip(),
            top_k=self.top_k_entry.get().strip(),
            max_workers=self.max_workers_entry.get().strip(),
            analyze=bool(self.analyze_var.get()),
            save_figures=bool(self.save_figures_var.get()),
            pixel_similarity_window_sizes=self.pixel_window_entry.get().strip(),
            pixel_max_displacements=self.pixel_disp_entry.get().strip(),
            pixel_gauss_sigmas=self.pixel_gauss_entry.get().strip(),
            pixel_disp_gauss_sigmas=self.pixel_disp_gauss_entry.get().strip(),
            pixel_thresholds=self.pixel_thresh_entry.get().strip(),
            combined_refinement_window_sizes=self.comb_window_entry.get().strip(),
            combined_max_displacements=self.comb_disp_entry.get().strip(),
            combined_gauss_sigmas=self.comb_gauss_entry.get().strip(),
            combined_disp_gauss_sigmas=self.comb_disp_gauss_entry.get().strip(),
            combined_pixel_thresholds=self.comb_thresh_entry.get().strip(),
            combined_global_thresholds=self.comb_global_thresh_entry.get().strip(),
            combined_correlation_pad_widths=self.comb_corr_pad_entry.get().strip(),
            pixel_sub_pix_factors=self.pixel_subpix_entry.get().strip(),
            pixel_sigma_pyramid=self.pixel_pyramid_entry.get().strip(),
            pixel_zero_fill_roll=self.pixel_zfill_entry.get().strip(),
            combined_sub_pix_factors=self.comb_subpix_entry.get().strip(),
            combined_sigma_pyramid=self.comb_pyramid_entry.get().strip(),
            combined_zero_fill_roll=self.comb_zfill_entry.get().strip(),
            combined_use_global_avg_reference=self.comb_use_global_ref_entry.get().strip(),
            global_thresholds=self.global_thresholds_entry.get().strip(),
            global_correlation_pad_widths=self.global_corr_pad_entry.get().strip(),
            weight_laplacian_variance=self.weight_lap_entry.get().strip(),
            weight_dwt_level2_hv_energy=self.weight_dwt_entry.get().strip(),
            weight_sum_squared_gradient=self.weight_grad_entry.get().strip(),
            weight_total_variation=self.weight_tv_entry.get().strip(),
            runtime_penalty_weight=self.runtime_penalty_entry.get().strip(),
            robustness_lambda=self.robustness_entry.get().strip(),
            missing_dataset_penalty=self.missing_penalty_entry.get().strip(),
            metric_experiments_enabled=bool(self.metric_experiments_enabled_var.get()),
            metric_experiments_shortlist_top_k=self.metric_shortlist_entry.get().strip(),
            metric_experiment_rows=metric_rows,
            datasets=datasets,
            json_preview=self.json_text.get("1.0", self.tk.END).strip(),
        )

    def _set_state(self, state: SpecGuiState) -> None:
        self.output_dir_entry.delete(0, self.tk.END)
        self.output_dir_entry.insert(0, state.output_dir)
        self.strategy_pixel_var.set(state.strategy_pixel)
        self.strategy_combined_var.set(state.strategy_combined)
        self.strategy_global_var.set(state.strategy_global)
        self.total_budget_entry.delete(0, self.tk.END)
        self.total_budget_entry.insert(0, state.total_budget_per_dataset)
        self.stage1_fraction_entry.delete(0, self.tk.END)
        self.stage1_fraction_entry.insert(0, state.stage1_fraction)
        self.top_k_entry.delete(0, self.tk.END)
        self.top_k_entry.insert(0, state.top_k)
        self.max_workers_entry.delete(0, self.tk.END)
        self.max_workers_entry.insert(0, state.max_workers)
        self.analyze_var.set(state.analyze)
        self.save_figures_var.set(state.save_figures)
        self.metric_experiments_enabled_var.set(state.metric_experiments_enabled)
        self.metric_shortlist_entry.delete(0, self.tk.END)
        self.metric_shortlist_entry.insert(0, state.metric_experiments_shortlist_top_k)

        field_mapping = [
            (self.pixel_window_entry, state.pixel_similarity_window_sizes),
            (self.pixel_disp_entry, state.pixel_max_displacements),
            (self.pixel_gauss_entry, state.pixel_gauss_sigmas),
            (self.pixel_disp_gauss_entry, state.pixel_disp_gauss_sigmas),
            (self.pixel_thresh_entry, state.pixel_thresholds),
            (self.comb_window_entry, state.combined_refinement_window_sizes),
            (self.comb_disp_entry, state.combined_max_displacements),
            (self.comb_gauss_entry, state.combined_gauss_sigmas),
            (self.comb_disp_gauss_entry, state.combined_disp_gauss_sigmas),
            (self.comb_thresh_entry, state.combined_pixel_thresholds),
            (self.comb_global_thresh_entry, state.combined_global_thresholds),
            (self.comb_corr_pad_entry, state.combined_correlation_pad_widths),
            (self.pixel_subpix_entry, state.pixel_sub_pix_factors),
            (self.pixel_pyramid_entry, state.pixel_sigma_pyramid),
            (self.pixel_zfill_entry, state.pixel_zero_fill_roll),
            (self.comb_subpix_entry, state.combined_sub_pix_factors),
            (self.comb_pyramid_entry, state.combined_sigma_pyramid),
            (self.comb_zfill_entry, state.combined_zero_fill_roll),
            (self.comb_use_global_ref_entry, state.combined_use_global_avg_reference),
            (self.global_thresholds_entry, state.global_thresholds),
            (self.global_corr_pad_entry, state.global_correlation_pad_widths),
            (self.weight_lap_entry, state.weight_laplacian_variance),
            (self.weight_dwt_entry, state.weight_dwt_level2_hv_energy),
            (self.weight_grad_entry, state.weight_sum_squared_gradient),
            (self.weight_tv_entry, state.weight_total_variation),
            (self.runtime_penalty_entry, state.runtime_penalty_weight),
            (self.robustness_entry, state.robustness_lambda),
            (self.missing_penalty_entry, state.missing_dataset_penalty),
        ]
        for widget, value in field_mapping:
            widget.delete(0, self.tk.END)
            widget.insert(0, value)

        self._clear_dataset_rows()
        for dataset in state.datasets:
            self._add_dataset_row(dataset)
        self._clear_metric_experiment_rows()
        for metric_row in state.metric_experiment_rows:
            self._add_metric_experiment_row(metric_row)
        if not self.metric_experiment_rows:
            self._add_metric_experiment_row()
        self._toggle_metric_experiment_controls()

        self.json_text.delete("1.0", self.tk.END)
        if state.json_preview:
            self.json_text.insert(self.tk.END, state.json_preview)

    def _on_refresh_json_preview(self) -> None:
        try:
            payload = state_to_spec_dict(self._get_state())
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("JSON Preview", str(exc))
            return
        self.json_text.delete("1.0", self.tk.END)
        self.json_text.insert(self.tk.END, json.dumps(payload, indent=2))

    def _on_apply_json_preview(self) -> None:
        raw = self.json_text.get("1.0", self.tk.END).strip()
        if not raw:
            self.messagebox.showwarning("JSON Preview", "JSON preview is empty.")
            return
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("JSON must be an object.")
            state = spec_dict_to_state(cast(Dict[str, object], payload))
            self._set_state(state)
            issues = validate_state(state)
            if issues:
                self.messagebox.showwarning(
                    "Spec Loaded with Warnings",
                    "\n".join(issues[:10]),
                )
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("JSON Preview", str(exc))

    def _on_load_spec(self) -> None:
        selected = self.filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")]
        )
        if not selected:
            return
        try:
            payload = _load_spec_payload(Path(selected))
            state = spec_dict_to_state(payload)
            self._set_state(state)
            issues = validate_state(state)
            if issues:
                self.messagebox.showwarning(
                    "Spec Loaded with Warnings",
                    "\n".join(issues[:10]),
                )
            self._append_log(f"Loaded spec: {selected}")
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Load Spec", str(exc))

    def _on_save_spec(self, template: bool = False) -> None:
        state = self._get_state()
        issues = validate_state(state)
        if issues:
            self.messagebox.showerror("Save Spec", "\n".join(issues[:10]))
            return
        try:
            payload = state_to_spec_dict(state)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Save Spec", str(exc))
            return
        default_name = "sweep_spec_template.json" if template else "sweep_spec.json"
        selected = self.filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not selected:
            return
        _save_spec_payload(Path(selected), payload)
        self._append_log(f"Saved spec: {selected}")

    def _run_worker(self, payload: Dict[str, object]) -> None:
        self.log_queue.put({"type": "log", "message": "Preparing experiment spec..."})
        try:
            spec = SweepExperimentSpec.from_dict(payload)
            self.log_queue.put({"type": "log", "message": f"Running sweep with {len(spec.datasets)} dataset(s)..."} )
            output_dir = run_sweep_experiment(spec)
            self.log_queue.put(
                {"type": "done", "output_dir": str(output_dir), "message": "Sweep completed successfully."}
            )
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def _on_run(self) -> None:
        if self._is_running:
            return
        state = self._get_state()
        issues = validate_state(state)
        if issues:
            self.messagebox.showerror("Run Sweep", "\n".join(issues[:12]))
            return
        try:
            payload = state_to_spec_dict(state)
            SweepExperimentSpec.from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Run Sweep", str(exc))
            return
        self.status_var.set("Running...")
        self._is_running = True
        self._set_editable_enabled(False)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", self.tk.END)
        self.log_text.configure(state="disabled")
        self._append_log("Starting background sweep run...")
        self.run_thread = threading.Thread(target=self._run_worker, args=(payload,), daemon=True)
        self.run_thread.start()

    def _on_run_done(self, output_dir: str, message: str) -> None:
        self.status_var.set("Completed")
        self._set_editable_enabled(True)
        self._is_running = False
        self._append_log(message)
        self._show_completion_dialog(Path(output_dir))

    def _on_run_error(self, message: str, trace_text: str) -> None:
        self.status_var.set("Failed")
        self._set_editable_enabled(True)
        self._is_running = False
        self._append_log("Sweep failed.")
        self._append_log(message)
        self.messagebox.showerror("Sweep Failed", f"{message}\n\n{trace_text}")

    def _poll_logs(self) -> None:
        while True:
            try:
                event = self.log_queue.get_nowait()
            except queue.Empty:
                break
            event_type = str(event.get("type", "log"))
            if event_type == "log":
                self._append_log(str(event.get("message", "")))
            elif event_type == "done":
                self._on_run_done(
                    str(event.get("output_dir", "")),
                    str(event.get("message", "Sweep completed.")),
                )
            elif event_type == "error":
                self._on_run_error(
                    str(event.get("message", "Unknown error")),
                    str(event.get("traceback", "")),
                )
        self.root.after(150, self._poll_logs)

    def _show_completion_dialog(self, output_dir: Path) -> None:
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Sweep Complete")
        dialog.geometry("540x180")
        self.ttk.Label(
            dialog, text=f"Artifacts written to:\n{output_dir}", wraplength=520
        ).pack(padx=10, pady=10, anchor="w")
        button_frame = self.ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)

        def open_output() -> None:
            ok, err = _open_with_system(output_dir)
            if not ok:
                self.messagebox.showerror("Open Output Folder", err or "Failed to open folder.")

        def open_leaderboard() -> None:
            leaderboard = output_dir / "leaderboard.json"
            if not leaderboard.exists():
                self.messagebox.showwarning("Open Leaderboard", f"File not found: {leaderboard}")
                return
            ok, err = _open_with_system(leaderboard)
            if not ok:
                self.messagebox.showerror("Open Leaderboard", err or "Failed to open file.")

        self.ttk.Button(button_frame, text="Open Output Folder", command=open_output).pack(side="left", padx=4)
        self.ttk.Button(button_frame, text="Open leaderboard.json", command=open_leaderboard).pack(side="left", padx=4)
        self.ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side="right", padx=4)


def launch_spec_gui() -> None:
    import tkinter as tk

    root = tk.Tk()
    SpecGuiApp(root)
    root.mainloop()


def _legacy_launch_gui_unused() -> None:
    launch_spec_gui()
    return

    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Moment Images Sweep Setup")

    labels = [
        ("Views (.npy)", "views", ""),
        ("Source image", "source_image", ""),
        ("Source image directory (HDR)", "source_image_dir", ""),
        ("Center X", "center_x", ""),
        ("Center Y", "center_y", ""),
        ("View radius", "view_radius", ""),
        ("View grid size", "view_grid_size", "3"),
        ("View distance (optional)", "view_distance", ""),
        ("Output directory", "output_dir", ""),
        ("HDR exposure times (comma-separated)", "hdr_exposure_times", ""),
        ("Window start", "window_start", "15"),
        ("Window stop", "window_stop", "45"),
        ("Window step", "window_step", "10"),
        ("Max disp start (search)", "max_disp_start", "5"),
        ("Max disp stop (search)", "max_disp_stop", "25"),
        ("Max disp step (search)", "max_disp_step", "10"),
        ("Gauss sigma start", "gauss_sigma_start", "3.0"),
        ("Gauss sigma stop", "gauss_sigma_stop", "7.0"),
        ("Gauss sigma step", "gauss_sigma_step", "2.0"),
        ("Disp gauss start", "disp_gauss_start", "0.0"),
        ("Disp gauss stop", "disp_gauss_stop", "1.0"),
        ("Disp gauss step", "disp_gauss_step", "1.0"),
        ("Global thresh start", "global_thresh_start", "0.15"),
        ("Global thresh stop", "global_thresh_stop", "0.15"),
        ("Global thresh step", "global_thresh_step", "0.05"),
        ("Pixel thresh start", "pixel_thresh_start", "0.05"),
        ("Pixel thresh stop", "pixel_thresh_stop", "0.05"),
        ("Pixel thresh step", "pixel_thresh_step", "0.05"),
        ("Zero fill roll", "zero_fill_roll", "True"),
    ]

    entries: dict[str, tk.Entry] = {}

    def choose_file(target: str, filetypes: list[tuple[str, str]] | None = None) -> None:
        selection = filedialog.askopenfilename(filetypes=filetypes)
        if selection:
            entries[target].delete(0, tk.END)
            entries[target].insert(0, selection)

    def choose_directory(target: str) -> None:
        selection = filedialog.askdirectory()
        if selection:
            entries[target].delete(0, tk.END)
            entries[target].insert(0, selection)

    for row, (label, key, default) in enumerate(labels):
        tk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        entry = tk.Entry(root, width=32)
        entry.insert(0, default)
        entry.grid(row=row, column=1, padx=6, pady=2)
        entries[key] = entry
        if key in {"views", "source_image", "source_image_dir"}:
            button_text = "Browse"
            if key == "views":
                filetypes = [("NumPy arrays", "*.npy")]
                tk.Button(root, text=button_text, command=lambda k=key, ft=filetypes: choose_file(k, ft)).grid(
                    row=row, column=2, padx=6, pady=2
                )
            elif key == "source_image":
                filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
                tk.Button(root, text=button_text, command=lambda k=key, ft=filetypes: choose_file(k, ft)).grid(
                    row=row, column=2, padx=6, pady=2
                )
            else:
                tk.Button(
                    root,
                    text=button_text,
                    command=lambda k=key: choose_directory(k),
                ).grid(row=row, column=2, padx=6, pady=2)

    tk.Label(root, text="Strategy").grid(row=len(labels), column=0, sticky="w", padx=6, pady=2)
    strategy_var = tk.StringVar(value="pixel")
    tk.OptionMenu(root, strategy_var, "pixel", "global", "combined").grid(row=len(labels), column=1, padx=6, pady=2, sticky="we")

    save_figures_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Save figures", variable=save_figures_var).grid(
        row=len(labels) + 1, column=1, sticky="w", padx=6, pady=2
    )

    hdr_assemble_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Assemble HDR from exposures", variable=hdr_assemble_var).grid(
        row=len(labels) + 2, column=1, sticky="w", padx=6, pady=2
    )

    analyze_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Analyze metrics after sweep", variable=analyze_var).grid(
        row=len(labels) + 3, column=1, sticky="w", padx=6, pady=2
    )

    live_plot_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Show live metrics plot", variable=live_plot_var).grid(
        row=len(labels) + 4, column=1, sticky="w", padx=6, pady=2
    )

    use_global_avg_reference_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root,
        text="Use global-rectified average as combined reference",
        variable=use_global_avg_reference_var,
    ).grid(row=len(labels) + 5, column=1, sticky="w", padx=6, pady=2)

    normalize_to_8bit_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        root,
        text="Normalize source images to 8-bit",
        variable=normalize_to_8bit_var,
    ).grid(row=len(labels) + 6, column=1, sticky="w", padx=6, pady=2)

    def start_sweep() -> None:
        try:
            args = argparse.Namespace(
                views=entries["views"].get().strip() or None,
                source_image=entries["source_image"].get().strip() or None,
                source_image_dir=entries["source_image_dir"].get().strip() or None,
                center_x=int(entries["center_x"].get()) if entries["center_x"].get().strip() else None,
                center_y=int(entries["center_y"].get()) if entries["center_y"].get().strip() else None,
                view_radius=int(entries["view_radius"].get()) if entries["view_radius"].get().strip() else None,
                view_grid_size=int(entries["view_grid_size"].get()),
                view_distance=_parse_optional_int(entries["view_distance"].get()),
                output_dir=entries["output_dir"].get().strip() or None,
                strategy=strategy_var.get(),
                window_start=int(entries["window_start"].get()),
                window_stop=int(entries["window_stop"].get()),
                window_step=int(entries["window_step"].get()),
                max_disp_start=int(entries["max_disp_start"].get()),
                max_disp_stop=int(entries["max_disp_stop"].get()),
                max_disp_step=int(entries["max_disp_step"].get()),
                gauss_sigma_start=float(entries["gauss_sigma_start"].get()),
                gauss_sigma_stop=float(entries["gauss_sigma_stop"].get()),
                gauss_sigma_step=float(entries["gauss_sigma_step"].get()),
                disp_gauss_start=float(entries["disp_gauss_start"].get()),
                disp_gauss_stop=float(entries["disp_gauss_stop"].get()),
                disp_gauss_step=float(entries["disp_gauss_step"].get()),
                global_thresh_start=float(entries["global_thresh_start"].get()),
                global_thresh_stop=float(entries["global_thresh_stop"].get()),
                global_thresh_step=float(entries["global_thresh_step"].get()),
                pixel_thresh_start=float(entries["pixel_thresh_start"].get()),
                pixel_thresh_stop=float(entries["pixel_thresh_stop"].get()),
                pixel_thresh_step=float(entries["pixel_thresh_step"].get()),
                zero_fill_roll=bool(entries["zero_fill_roll"].get().strip().lower() in {"true", "1", "yes"}),
                save_figures=save_figures_var.get(),
                hdr_assemble=hdr_assemble_var.get(),
                hdr_exposure_times=entries["hdr_exposure_times"].get().strip() or None,
                analyze=analyze_var.get(),
                live_plot=live_plot_var.get(),
                use_global_avg_reference=use_global_avg_reference_var.get(),
                normalize_to_8bit=normalize_to_8bit_var.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        root.withdraw()
        try:
            output_dir = _run_from_namespace(args)
        except Exception as exc:  # noqa: BLE001 - bubble UI errors to user
            root.deiconify()
            messagebox.showerror("Sweep failed", str(exc))
            return

        messagebox.showinfo("Sweep complete", f"Artifacts written to {output_dir}")
        root.destroy()

    tk.Button(root, text="Start sweep", command=start_sweep).grid(
        row=len(labels) + 7, column=1, pady=8
    )

    root.mainloop()


def _launch_gui() -> None:
    """Launch the spec-based sweep GUI."""
    launch_spec_gui()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    should_launch_gui = args.gui or (argv is None and len(sys.argv) == 1)
    if should_launch_gui:
        _launch_gui()
        return

    _run_from_namespace(args)


if __name__ == "__main__":
    main()

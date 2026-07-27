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
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from moment_images import sweep_scheduler
from moment_images.memory_enforcement import create_memory_enforcer, worker_registration_environment
from moment_images.power_blend import PowerBlendConfig, PowerBlendSearchSpace
from moment_images.im_process import (
    COMBINED_REFERENCE_MODE_GLOBAL_AVERAGE,
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
from moment_images.rectification_strategy import (
    BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
    BLURRED_GRADIENT_CORR_METRIC_FAMILY,
    BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
    DEFAULT_DISPLACEMENT_ALPHA,
    DEFAULT_DISPLACEMENT_METRIC_FAMILY,
    normalize_displacement_metric_family,
    normalize_pixel_shift_power,
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
from moment_images.sweep_scheduler import (
    MEMORY_SAFETY_FRACTION,
    MEMORY_ENFORCEMENT_AUTO,
    MEMORY_ENFORCEMENT_MODES,
    SCHEDULER_MODE_ADAPTIVE_MEMORY,
    SCHEDULER_MODE_FIXED_BATCH,
    SCHEDULER_MODES,
    SweepSchedulingSpec,
    build_memory_limit_result,
)
from moment_images.metric_profiles import (
    build_metric_registry,
    compute_extended_image_metrics,
    default_profiles,
    resolve_metric_config,
    resolve_profile_weights_for_sweep,
)
from moment_images.human_bead_oracle import (
    HUMAN_BEAD_ORACLE_MODE,
    HUMAN_BEAD_ORACLE_SCORE,
    build_human_bead_oracle_import,
    load_human_bead_annotations,
    write_human_bead_oracle_import,
)
from moment_images.sweep_analysis import (
    analyze_summary_data,
    build_metric_record,
    generate_alpha_report,
    save_analysis,
)
from moment_images.trio_benchmark import BenchmarkConfig, run_benchmark
from moment_images.trio_oracle_metrics import (
    TRIO_LF_SOURCE_KIND,
    TRIO_PRIMARY_SCORE,
    find_trio_record_for_identity,
    import_trio_oracle_results,
    trio_rankable,
)


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST = "baseline_shortlist"
METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH = "independent_search"
METRIC_CANDIDATE_SOURCES = {
    METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST,
    METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH,
}
METRIC_SEARCH_STAGE1_NAME = "metric_stage1"
METRIC_PHASE_STAGE_NAME = "metric_phase"
VALIDATION_ORACLE_WORKFLOW_NORMAL = "normal_validation"
VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE = "validation_guided_refinement"
VALIDATION_ORACLE_WORKFLOW_MODES = {
    VALIDATION_ORACLE_WORKFLOW_NORMAL,
    VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE,
}
VALIDATION_ORACLE_MODE_TRIO = "trio_validation"
VALIDATION_ORACLE_MODE_HUMAN_BEAD = HUMAN_BEAD_ORACLE_MODE
VALIDATION_ORACLE_MODES = {
    VALIDATION_ORACLE_MODE_TRIO,
    VALIDATION_ORACLE_MODE_HUMAN_BEAD,
}
VALIDATION_ORACLE_STAGE_NAME = "validation_phase"
VALIDATION_ORACLE_DIR_NAME = "validation_oracle"
VALIDATION_ORACLE_PATH_KEYS = (
    "validation_image",
    "thorcam_image",
    "validation_workspace",
    "registration_annotations",
    "human_bead_annotations",
)
VALIDATION_ORACLE_REFERENCE_KEYS = (
    "validation_workspace",
    "validation_image",
    "thorcam_image",
)
VALIDATION_ORACLE_GUI_OWNED_SCENE_KEYS = VALIDATION_ORACLE_REFERENCE_KEYS + (
    "use_validation_workspace_transforms",
)
VALIDATION_ORACLE_GUI_OWNED_DEFAULT_KEYS = (
    "human_bead_annotations",
)
VALIDATION_ORACLE_GENERATED_ROOT_ARTIFACTS = (
    f"{VALIDATION_ORACLE_STAGE_NAME}_summary.json",
    "validation_leaderboard.json",
    "validation_leaderboard.csv",
    "validation_best_configs.json",
    "validation_summary.json",
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
    pixel_shift_powers: List[float] = field(default_factory=lambda: [1.0])
    pixel_power_blend_enabled: List[bool] = field(default_factory=lambda: [False])
    pixel_power_blend_raised_powers: List[float] = field(default_factory=list)
    pixel_power_blend_alphas: List[float] = field(default_factory=lambda: [0.0])
    pixel_power_blend_delta_caps: List[float] = field(default_factory=lambda: [0.0])
    correlation_pad_widths: List[int] = field(default_factory=lambda: [RectificationRunConfig().correlation_pad_width])
    combined_use_global_avg_reference: List[bool] = field(default_factory=lambda: [True])
    combined_reference_mode: Optional[str] = None
    gauss_sizes: Optional[List[int]] = None
    disp_gauss_sizes: Optional[List[int]] = None

    def power_blend_search_space(self) -> PowerBlendSearchSpace:
        return PowerBlendSearchSpace.from_lists(
            self.pixel_power_blend_enabled,
            self.pixel_power_blend_raised_powers,
            self.pixel_power_blend_alphas,
            self.pixel_power_blend_delta_caps,
        )

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
            pixel_shift_powers=[float(v) for v in raw.get("pixel_shift_powers", [1.0])],  # type: ignore[arg-type]
            pixel_power_blend_enabled=[bool(v) for v in raw.get("pixel_power_blend_enabled", [False])],  # type: ignore[arg-type]
            pixel_power_blend_raised_powers=[float(v) for v in raw.get("pixel_power_blend_raised_powers", [])],  # type: ignore[arg-type]
            pixel_power_blend_alphas=[float(v) for v in raw.get("pixel_power_blend_alphas", [0.0])],  # type: ignore[arg-type]
            pixel_power_blend_delta_caps=[float(v) for v in raw.get("pixel_power_blend_delta_caps", [0.0])],  # type: ignore[arg-type]
            correlation_pad_widths=[int(v) for v in raw.get("correlation_pad_widths", [RectificationRunConfig().correlation_pad_width])],  # type: ignore[arg-type]
            combined_use_global_avg_reference=[bool(v) for v in raw.get("combined_use_global_avg_reference", [True])],  # type: ignore[arg-type]
            combined_reference_mode=cast(
                Optional[str],
                raw.get("combined_reference_mode"),
            ),
            gauss_sizes=None
            if raw.get("gauss_sizes") is None
            else [int(v) for v in raw.get("gauss_sizes", [])],  # type: ignore[arg-type]
            disp_gauss_sizes=None
            if raw.get("disp_gauss_sizes") is None
            else [int(v) for v in raw.get("disp_gauss_sizes", [])],  # type: ignore[arg-type]
        )


@dataclass
class SweepStageRefinementSpec:
    """Controls how two-stage sweeps generate stage-2 candidates."""

    method: Literal["auto_fine", "neighbor"] = "auto_fine"
    extrapolation: Literal["half_step_bounded", "inside_only", "full_step_outside"] = "half_step_bounded"

    def __post_init__(self) -> None:
        normalized_method = str(self.method).strip().lower()
        if normalized_method not in {"auto_fine", "neighbor"}:
            raise ValueError("stages.refinement.method must be either 'auto_fine' or 'neighbor'.")
        self.method = cast(Literal["auto_fine", "neighbor"], normalized_method)

        normalized_extrapolation = str(self.extrapolation).strip().lower()
        if normalized_extrapolation not in {
            "half_step_bounded",
            "inside_only",
            "full_step_outside",
        }:
            raise ValueError(
                "stages.refinement.extrapolation must be one of "
                "'half_step_bounded', 'inside_only', or 'full_step_outside'."
            )
        self.extrapolation = cast(
            Literal["half_step_bounded", "inside_only", "full_step_outside"],
            normalized_extrapolation,
        )

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "SweepStageRefinementSpec":
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("stages.refinement must be an object when provided.")
        raw = payload or {}
        return cls(
            method=cast(Literal["auto_fine", "neighbor"], str(raw.get("method", "auto_fine"))),
            extrapolation=cast(
                Literal["half_step_bounded", "inside_only", "full_step_outside"],
                str(raw.get("extrapolation", "half_step_bounded")),
            ),
        )


@dataclass
class SweepStageSpec:
    """Controls two-stage coarse/fine search behavior."""

    mode: Literal["two_stage", "single_stage"] = "two_stage"
    total_budget_per_dataset: int = 320
    stage1_fraction: float = 0.6
    top_k: int = 8
    max_workers: Optional[int] = None
    refinement: SweepStageRefinementSpec = field(default_factory=SweepStageRefinementSpec)
    scheduling: SweepSchedulingSpec = field(default_factory=SweepSchedulingSpec)

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode).strip().lower()
        if normalized_mode not in {"two_stage", "single_stage"}:
            raise ValueError("stages.mode must be either 'two_stage' or 'single_stage'.")
        self.mode = cast(Literal["two_stage", "single_stage"], normalized_mode)
        if not isinstance(self.refinement, SweepStageRefinementSpec):
            self.refinement = SweepStageRefinementSpec.from_dict(
                cast(Optional[Dict[str, object]], self.refinement)
            )
        if not isinstance(self.scheduling, SweepSchedulingSpec):
            self.scheduling = SweepSchedulingSpec.from_dict(
                cast(Optional[Dict[str, object]], self.scheduling)
            )


def _stage_spec_from_dict(payload: Optional[Dict[str, object]]) -> SweepStageSpec:
    raw = payload or {}
    return SweepStageSpec(
        mode=cast(
            Literal["two_stage", "single_stage"],
            str(raw.get("mode", "two_stage")),
        ),
        total_budget_per_dataset=int(raw.get("total_budget_per_dataset", 320)),
        stage1_fraction=float(raw.get("stage1_fraction", 0.6)),
        top_k=int(raw.get("top_k", 8)),
        max_workers=(
            None
            if raw.get("max_workers") is None
            else int(cast(object, raw.get("max_workers")))
        ),
        refinement=SweepStageRefinementSpec.from_dict(
            cast(Optional[Dict[str, object]], raw.get("refinement"))
        ),
        scheduling=SweepSchedulingSpec.from_dict(
            cast(Optional[Dict[str, object]], raw.get("scheduling"))
        ),
    )


@dataclass
class SweepScoringSpec:
    """Controls composite metric scoring and robustness aggregation."""

    metric_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    runtime_penalty_weight: float = 0.0
    robustness_lambda: float = 0.5
    missing_dataset_penalty: float = 1.0
    metric_scope: str = "global_and_local_metrics"
    profile_name: Optional[str] = None
    profile_config_path: Optional[str] = None


@dataclass
class MetricFormulationSpec:
    """Local displacement-score family definition used as a sweep dimension."""

    formulation_id: str
    family: str = BLURRED_GRADIENT_CORR_METRIC_FAMILY
    alphas: List[float] = field(default_factory=lambda: [0.5])
    component_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        formulation_id = str(self.formulation_id).strip()
        if not formulation_id:
            raise ValueError("metric_formulations[].id must be non-empty.")
        self.formulation_id = formulation_id

        self.family = normalize_displacement_metric_family(self.family)

        if not self.alphas:
            raise ValueError("metric_formulations[].alphas cannot be empty.")
        normalized_alphas = [float(item) for item in self.alphas]
        for alpha in normalized_alphas:
            if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
                raise ValueError("metric_formulations alpha values must be finite and in [0, 1].")
        self.alphas = normalized_alphas

        if self.family == DEFAULT_DISPLACEMENT_METRIC_FAMILY:
            if self.component_weights is not None:
                raise ValueError("Intensity metric formulations do not accept component_weights.")
            if self.alphas != [DEFAULT_DISPLACEMENT_ALPHA]:
                raise ValueError("Intensity metric formulations require alpha=1.0.")
            return

        if self.component_weights is None:
            return

        normalized_weights: Dict[str, float] = {}
        for key, raw_value in self.component_weights.items():
            numeric = float(raw_value)
            if not np.isfinite(numeric):
                raise ValueError("component_weights values must be finite.")
            if numeric < 0.0:
                raise ValueError("component_weights values must be non-negative.")
            normalized_weights[str(key)] = numeric
        if (
            self.family == BLURRED_COMPOSITE_CORR_METRIC_FAMILY
            and any(alpha < 1.0 for alpha in self.alphas)
            and sum(normalized_weights.values()) <= 0.0
        ):
            raise ValueError(
                "component_weights must sum to a positive value when "
                "blurred_composite_corr alpha < 1."
            )
        self.component_weights = normalized_weights

    @classmethod
    def from_dict(cls, payload: Dict[str, object], default_idx: int) -> "MetricFormulationSpec":
        raw_id = str(payload.get("id") or payload.get("formulation_id") or "").strip()
        formulation_id = raw_id or f"formulation_{default_idx:02d}"

        family = normalize_displacement_metric_family(
            payload.get("family", BLURRED_GRADIENT_CORR_METRIC_FAMILY)
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


def _clone_stage_spec(spec: Optional[SweepStageSpec]) -> Optional[SweepStageSpec]:
    if spec is None:
        return None
    return SweepStageSpec(
        mode=spec.mode,
        total_budget_per_dataset=int(spec.total_budget_per_dataset),
        stage1_fraction=float(spec.stage1_fraction),
        top_k=int(spec.top_k),
        max_workers=spec.max_workers,
        refinement=SweepStageRefinementSpec(
            method=spec.refinement.method,
            extrapolation=spec.refinement.extrapolation,
        ),
        scheduling=SweepSchedulingSpec(
            mode=spec.scheduling.mode,
            memory_safety_fraction=spec.scheduling.memory_safety_fraction,
            memory_telemetry=spec.scheduling.memory_telemetry,
            worker_recycle_tasks=spec.scheduling.worker_recycle_tasks,
            memory_enforcement=spec.scheduling.memory_enforcement,
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
    candidate_source: Literal["baseline_shortlist", "independent_search"] = (
        METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST
    )
    search_stages: Optional[SweepStageSpec] = None

    def __post_init__(self) -> None:
        self.shortlist_top_k = int(self.shortlist_top_k)
        if self.shortlist_top_k < 1:
            raise ValueError("experimental_metric_sweep.shortlist_top_k must be >= 1.")

        source = str(self.candidate_source).strip().lower()
        if source not in METRIC_CANDIDATE_SOURCES:
            raise ValueError(
                "experimental_metric_sweep.candidate_source must be either "
                "'baseline_shortlist' or 'independent_search'."
            )
        self.candidate_source = cast(
            Literal["baseline_shortlist", "independent_search"], source
        )

        if self.search_stages is not None and not isinstance(
            self.search_stages, SweepStageSpec
        ):
            self.search_stages = _stage_spec_from_dict(
                cast(Dict[str, object], self.search_stages)
            )
        if self.candidate_source == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH:
            if self.search_stages is None:
                self.search_stages = SweepStageSpec(
                    total_budget_per_dataset=self.shortlist_top_k,
                    top_k=self.shortlist_top_k,
                )
        else:
            self.search_stages = None

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
        candidate_source = str(
            payload.get(
                "candidate_source", METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST
            )
        )
        search_stages_raw = payload.get("search_stages")
        if search_stages_raw is not None and not isinstance(search_stages_raw, dict):
            raise ValueError("experimental_metric_sweep.search_stages must be an object.")
        search_stages = (
            _stage_spec_from_dict(cast(Dict[str, object], search_stages_raw))
            if search_stages_raw is not None
            else None
        )
        return cls(
            enabled=enabled,
            shortlist_top_k=shortlist_top_k,
            formulations=formulations,
            candidate_source=cast(
                Literal["baseline_shortlist", "independent_search"], candidate_source
            ),
            search_stages=search_stages,
        )


@dataclass
class ValidationOracleSweepSpec:
    """Optional validation-guided sweep that scores LF outputs against trio references."""

    enabled: bool = False
    oracle_mode: str = VALIDATION_ORACLE_MODE_TRIO
    top_k: int = 8
    budget_per_dataset: Optional[int] = None
    skip_image_metric_stage: bool = False
    scenes: List[Dict[str, object]] = field(default_factory=list)
    defaults: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.oracle_mode = str(self.oracle_mode or VALIDATION_ORACLE_MODE_TRIO).strip()
        if self.oracle_mode not in VALIDATION_ORACLE_MODES:
            raise ValueError(
                "validation_oracle_sweep.oracle_mode must be one of: "
                + ", ".join(sorted(VALIDATION_ORACLE_MODES))
            )
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise ValueError("validation_oracle_sweep.top_k must be >= 1.")
        if self.budget_per_dataset is not None:
            self.budget_per_dataset = int(self.budget_per_dataset)
            if self.budget_per_dataset < 1:
                raise ValueError(
                    "validation_oracle_sweep.budget_per_dataset must be >= 1 when provided."
                )
        if not isinstance(self.defaults, dict):
            raise ValueError("validation_oracle_sweep.defaults must be an object.")
        self.defaults = dict(self.defaults)
        if not isinstance(self.scenes, list):
            raise ValueError("validation_oracle_sweep.scenes must be a list.")
        normalized_scenes: List[Dict[str, object]] = []
        for idx, scene in enumerate(self.scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"validation_oracle_sweep.scenes[{idx}] must be an object.")
            normalized_scenes.append(dict(scene))
        self.scenes = normalized_scenes

    @classmethod
    def from_dict(
        cls,
        payload: Optional[Dict[str, object]],
        *,
        default_top_k: int,
    ) -> "ValidationOracleSweepSpec":
        if payload is None:
            return cls(enabled=False, top_k=default_top_k)
        if not isinstance(payload, dict):
            raise ValueError("validation_oracle_sweep must be an object.")
        return cls(
            enabled=bool(payload.get("enabled", False)),
            oracle_mode=str(payload.get("oracle_mode", VALIDATION_ORACLE_MODE_TRIO)),
            top_k=int(payload.get("top_k", default_top_k)),
            budget_per_dataset=(
                None
                if payload.get("budget_per_dataset") is None
                else int(cast(object, payload.get("budget_per_dataset")))
            ),
            skip_image_metric_stage=bool(payload.get("skip_image_metric_stage", False)),
            scenes=cast(List[Dict[str, object]], payload.get("scenes", [])),
            defaults=cast(Dict[str, object], payload.get("defaults", {})),
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
    validation_oracle_sweep: Optional[ValidationOracleSweepSpec] = None
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
                candidate_source=experimental.candidate_source,
                search_stages=_clone_stage_spec(experimental.search_stages),
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
            candidate_source=experimental.candidate_source,
            search_stages=_clone_stage_spec(experimental.search_stages),
        )
        self.metric_formulations = [
            _clone_metric_formulation(item)
            for item in self.experimental_metric_sweep.formulations
        ]
        validation = self.validation_oracle_sweep
        if validation is None:
            validation = ValidationOracleSweepSpec(enabled=False, top_k=stage_top_k)
        elif not isinstance(validation, ValidationOracleSweepSpec):
            validation = ValidationOracleSweepSpec.from_dict(
                cast(Dict[str, object], validation),
                default_top_k=stage_top_k,
            )
        if (
            validation.enabled
            and self.stages.mode != "two_stage"
            and not validation.skip_image_metric_stage
        ):
            raise ValueError(
                "validation_oracle_sweep requires stages.mode='two_stage' unless "
                "validation_oracle_sweep.skip_image_metric_stage is enabled."
            )
        self.validation_oracle_sweep = ValidationOracleSweepSpec(
            enabled=bool(validation.enabled),
            oracle_mode=validation.oracle_mode,
            top_k=int(validation.top_k),
            budget_per_dataset=validation.budget_per_dataset,
            skip_image_metric_stage=bool(validation.skip_image_metric_stage),
            scenes=[dict(scene) for scene in validation.scenes],
            defaults=dict(validation.defaults),
        )

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
            mode=cast(
                Literal["two_stage", "single_stage"],
                str(stages_raw.get("mode", "two_stage")),
            ),
            total_budget_per_dataset=int(stages_raw.get("total_budget_per_dataset", 320)),
            stage1_fraction=float(stages_raw.get("stage1_fraction", 0.6)),
            top_k=int(stages_raw.get("top_k", 8)),
            max_workers=cast(Optional[int], stages_raw.get("max_workers")),
            refinement=SweepStageRefinementSpec.from_dict(
                cast(Optional[Dict[str, object]], stages_raw.get("refinement"))
            ),
            scheduling=SweepSchedulingSpec.from_dict(
                cast(Optional[Dict[str, object]], stages_raw.get("scheduling"))
            ),
        )

        scoring_raw = cast(Dict[str, object], payload.get("scoring", {}))
        profile_name_raw = scoring_raw.get("profile_name")
        profile_name = (
            str(profile_name_raw).strip()
            if isinstance(profile_name_raw, str) and profile_name_raw.strip()
            else None
        )
        profile_config_path_raw = scoring_raw.get("profile_config_path")
        profile_config_path = (
            str(profile_config_path_raw).strip()
            if isinstance(profile_config_path_raw, str) and profile_config_path_raw.strip()
            else None
        )
        if profile_name and "metric_weights" in scoring_raw:
            raise ValueError("scoring cannot define both profile_name and metric_weights.")
        metric_weights = cast(Optional[Dict[str, float]], scoring_raw.get("metric_weights"))
        resolved_metric_weights = dict(metric_weights) if metric_weights else dict(DEFAULT_SCORE_WEIGHTS)
        if profile_name:
            resolved_metric_weights, _, _ = resolve_profile_weights_for_sweep(
                profile_name,
                config_path=Path(profile_config_path) if profile_config_path else None,
                scoring_weights=DEFAULT_SCORE_WEIGHTS,
            )
        scoring = SweepScoringSpec(
            metric_weights=resolved_metric_weights,
            runtime_penalty_weight=float(scoring_raw.get("runtime_penalty_weight", 0.0)),
            robustness_lambda=float(scoring_raw.get("robustness_lambda", 0.5)),
            missing_dataset_penalty=float(scoring_raw.get("missing_dataset_penalty", 1.0)),
            metric_scope=str(scoring_raw.get("metric_scope", "global_and_local_metrics")),
            profile_name=profile_name,
            profile_config_path=profile_config_path,
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

        validation_oracle_sweep = ValidationOracleSweepSpec.from_dict(
            cast(Optional[Dict[str, object]], payload.get("validation_oracle_sweep")),
            default_top_k=stages.top_k,
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
            validation_oracle_sweep=validation_oracle_sweep,
            metric_formulations=legacy_formulations,
            reporting=reporting,
            output_dir=cast(Optional[str], payload.get("output_dir")),
            save_figures=bool(payload.get("save_figures", False)),
            analyze=bool(payload.get("analyze", True)),
            keep_all_run_artifacts=bool(payload.get("keep_all_run_artifacts", True)),
            random_seed=int(payload.get("random_seed", 0)),
        )


@dataclass(frozen=True)
class PathRelocationRule:
    """Map paths from an old root to a new root when resuming copied sweeps."""

    old_root: str
    new_root: str


def _path_norm_for_compare(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _path_from_user_text(value: str) -> Path:
    return Path(os.path.expanduser(str(value).strip())).resolve(strict=False)


def _parse_path_relocation_rules(raw_rules: Optional[Sequence[str]]) -> List[PathRelocationRule]:
    parsed: List[PathRelocationRule] = []
    seen_old: set[str] = set()
    for raw in raw_rules or []:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(
                f"Invalid relocation mapping '{raw}'. Expected OLD=NEW."
            )
        old_raw, new_raw = text.split("=", 1)
        old_root = old_raw.strip()
        new_root = new_raw.strip()
        if not old_root or not new_root:
            raise ValueError(
                f"Invalid relocation mapping '{raw}'. OLD and NEW must be non-empty."
            )
        old_path = _path_from_user_text(old_root)
        new_path = _path_from_user_text(new_root)
        old_key = _path_norm_for_compare(old_path)
        if old_key in seen_old:
            raise ValueError(f"Duplicate relocation old root: {old_root}")
        seen_old.add(old_key)
        parsed.append(
            PathRelocationRule(old_root=str(old_path), new_root=str(new_path))
        )
    parsed.sort(key=lambda rule: len(_path_norm_for_compare(Path(rule.old_root))), reverse=True)
    return parsed


def _relocation_rules_to_payload(
    rules: Sequence[PathRelocationRule],
) -> List[Dict[str, str]]:
    return [{"old": rule.old_root, "new": rule.new_root} for rule in rules]


def _parse_relocation_rules_payload(payload: object) -> List[PathRelocationRule]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("relocation.path_mappings must be a list.")
    raw_rules: List[str] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"relocation.path_mappings[{idx}] must be an object.")
        old_root = str(item.get("old") or item.get("old_root") or "").strip()
        new_root = str(item.get("new") or item.get("new_root") or "").strip()
        if not old_root or not new_root:
            raise ValueError(
                f"relocation.path_mappings[{idx}] must define old and new."
            )
        raw_rules.append(f"{old_root}={new_root}")
    return _parse_path_relocation_rules(raw_rules)


def _relocate_path_string(
    value: object,
    rules: Sequence[PathRelocationRule],
) -> Tuple[object, Optional[Dict[str, str]]]:
    if not isinstance(value, str) or not value.strip() or not rules:
        return value, None
    try:
        candidate = _path_from_user_text(value)
    except Exception:
        return value, None
    candidate_norm = _path_norm_for_compare(candidate)
    for rule in rules:
        old_root = _path_from_user_text(rule.old_root)
        new_root = _path_from_user_text(rule.new_root)
        old_norm = _path_norm_for_compare(old_root)
        if candidate_norm != old_norm and not candidate_norm.startswith(old_norm + os.sep):
            continue
        try:
            suffix = candidate.relative_to(old_root)
        except ValueError:
            suffix_text = candidate_norm[len(old_norm) :].lstrip("\\/")
            suffix = Path(suffix_text)
        relocated = new_root / suffix
        return str(relocated), {"from": str(value), "to": str(relocated)}
    return value, None


def _relocation_options_from_spec_payload(
    payload: Mapping[str, object],
) -> Tuple[List[PathRelocationRule], bool]:
    relocation = payload.get("relocation")
    if not isinstance(relocation, dict):
        return [], False
    rules = _parse_relocation_rules_payload(relocation.get("path_mappings"))
    adopt = bool(relocation.get("adopt_relocated", False))
    return rules, adopt


def _apply_relocation_to_spec_payload(
    payload: Mapping[str, object],
    rules: Sequence[PathRelocationRule],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    relocated = json.loads(json.dumps(payload))
    applied: List[Dict[str, str]] = []

    def apply_field(container: Dict[str, object], key: str, label: str) -> None:
        new_value, change = _relocate_path_string(container.get(key), rules)
        if change is not None:
            container[key] = new_value
            applied.append({"field": label, **change})

    apply_field(relocated, "output_dir", "output_dir")

    scoring = relocated.get("scoring")
    if isinstance(scoring, dict):
        apply_field(cast(Dict[str, object], scoring), "profile_config_path", "scoring.profile_config_path")

    datasets = relocated.get("datasets")
    if isinstance(datasets, list):
        for idx, entry in enumerate(datasets):
            if not isinstance(entry, dict):
                continue
            dataset_id = str(entry.get("id") or entry.get("dataset_id") or idx)
            dataset_entry = cast(Dict[str, object], entry)
            for key in ("views", "views_path", "source_image", "source_image_dir"):
                apply_field(dataset_entry, key, f"datasets[{dataset_id}].{key}")

    validation_oracle = relocated.get("validation_oracle_sweep")
    if isinstance(validation_oracle, dict):
        oracle_entry = cast(Dict[str, object], validation_oracle)
        defaults = oracle_entry.get("defaults")
        if isinstance(defaults, dict):
            defaults_entry = cast(Dict[str, object], defaults)
            for key in VALIDATION_ORACLE_PATH_KEYS:
                apply_field(defaults_entry, key, f"validation_oracle_sweep.defaults.{key}")
        scenes = oracle_entry.get("scenes")
        if isinstance(scenes, list):
            for idx, scene in enumerate(scenes):
                if not isinstance(scene, dict):
                    continue
                scene_id = str(
                    scene.get("scene_id")
                    or scene.get("dataset_id")
                    or idx
                )
                scene_entry = cast(Dict[str, object], scene)
                for key in VALIDATION_ORACLE_PATH_KEYS:
                    apply_field(scene_entry, key, f"validation_oracle_sweep.scenes[{scene_id}].{key}")

    report: Dict[str, object] = {
        "rules": _relocation_rules_to_payload(rules),
        "applied": applied,
    }
    return relocated, report


def _resolve_relative_path_fields(container: Dict[str, object], keys: Sequence[str], spec_path: Path) -> None:
    for key in keys:
        value = container.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if not path.is_absolute():
            container[key] = str((spec_path.parent / path).resolve())


def _resolve_spec_relative_paths_payload(
    payload: Dict[str, object],
    spec_path: Path,
) -> Dict[str, object]:
    resolved = json.loads(json.dumps(payload))
    scoring = resolved.get("scoring")
    if isinstance(scoring, dict):
        profile_config_path_raw = scoring.get("profile_config_path")
        if isinstance(profile_config_path_raw, str) and profile_config_path_raw.strip():
            profile_config_path = Path(profile_config_path_raw)
            if not profile_config_path.is_absolute():
                scoring["profile_config_path"] = str((spec_path.parent / profile_config_path).resolve())
    validation_oracle = resolved.get("validation_oracle_sweep")
    if isinstance(validation_oracle, dict):
        oracle_entry = cast(Dict[str, object], validation_oracle)
        defaults = oracle_entry.get("defaults")
        if isinstance(defaults, dict):
            _resolve_relative_path_fields(
                cast(Dict[str, object], defaults),
                VALIDATION_ORACLE_PATH_KEYS,
                spec_path,
            )
        scenes = oracle_entry.get("scenes")
        if isinstance(scenes, list):
            for scene in scenes:
                if isinstance(scene, dict):
                    _resolve_relative_path_fields(
                        cast(Dict[str, object], scene),
                        VALIDATION_ORACLE_PATH_KEYS,
                        spec_path,
                    )
    return resolved


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
    views_content_sha256: Optional[str] = None


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
            "pixel_shift_power",
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
            "pixel_shift_power",
        ]
    if strategy == "global":
        return [
            "correlation_pad_width",
            "global_threshold",
        ]
    raise ValueError(f"Unsupported strategy: {strategy}")


def _candidate_config_signature_payload(config_payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        str(key): value
        for key, value in config_payload.items()
        if str(key) not in DATASET_CONFIG_KEYS
    }


def _candidate_signature_from_config_payload(config_payload: Mapping[str, object]) -> str:
    return _hash_payload(_candidate_config_signature_payload(config_payload))


def _candidate_signature(candidate: SweepCandidate) -> str:
    return _candidate_signature_from_config_payload(asdict(candidate.config))


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
    used_ids: Dict[str, str] = {}
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        hash_len = CANDIDATE_ID_HASH_LEN
        candidate_id = f"{prefix}{signature[:hash_len]}"
        while candidate_id in used_ids and used_ids[candidate_id] != signature:
            hash_len += 4
            candidate_id = f"{prefix}{signature[:hash_len]}"
        clone = SweepCandidate(
            strategy=candidate.strategy,
            params=dict(candidate.params),
            config=candidate.config,
            candidate_id=candidate_id,
        )
        used_ids[candidate_id] = signature
        assigned.append(clone)
    return assigned


def _metric_formulation_from_params(
    params: Dict[str, object],
) -> Tuple[str, float, Optional[Dict[str, float]], str]:
    family = normalize_displacement_metric_family(
        params.get("displacement_metric_family", "intensity")
    )
    alpha = float(params.get("displacement_alpha", 1.0))
    raw_weights_obj = params.get("displacement_component_weights")
    parsed_weights: Optional[Dict[str, float]] = None
    if raw_weights_obj is not None:
        parsed_weights = {
            str(key): float(value)
            for key, value in cast(Dict[str, object], raw_weights_obj).items()
        }
    formulation_id = str(params.get("metric_formulation_id", "intensity_default")).strip()
    if not formulation_id:
        formulation_id = "intensity_default"
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
            pixel_shift_power=float(params.get("pixel_shift_power", defaults.pixel_shift_power)),
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        normalized_params = dict(params)
        normalized_params["similarity_window_size"] = window
        normalized_params["gauss_size"] = gauss_size
        normalized_params["disp_gauss_size"] = disp_gauss_size
        normalized_params["pixel_shift_power"] = config.pixel_shift_power
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
            pixel_power_blend_enabled=bool(params.get("pixel_power_blend_enabled", False)),
            pixel_power_blend_raised_power=cast(Optional[float], params.get("pixel_power_blend_raised_power")),
            pixel_power_blend_alpha=float(params.get("pixel_power_blend_alpha", 0.0)),
            pixel_power_blend_delta_cap=float(params.get("pixel_power_blend_delta_cap", 0.0)),
            zero_fill_roll=bool(params["zero_fill_roll"]),
            sub_pix_factor=int(params["sub_pix_factor"]),
            pixel_shift_power=float(params.get("pixel_shift_power", defaults.pixel_shift_power)),
            combined_use_global_avg_reference=bool(params["combined_use_global_avg_reference"]),
            combined_reference_mode=cast(Optional[str], params.get("combined_reference_mode")),
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        normalized_params = dict(params)
        normalized_params["refinement_window_size"] = window
        normalized_params["gauss_size"] = gauss_size
        normalized_params["disp_gauss_size"] = disp_gauss_size
        normalized_params["pixel_shift_power"] = config.pixel_shift_power
        normalized_params["pixel_power_blend_enabled"] = config.pixel_power_blend_enabled
        normalized_params["pixel_power_blend_raised_power"] = config.pixel_power_blend_raised_power
        normalized_params["pixel_power_blend_alpha"] = config.pixel_power_blend_alpha
        normalized_params["pixel_power_blend_delta_cap"] = config.pixel_power_blend_delta_cap
        return SweepCandidate(strategy="combined", params=normalized_params, config=config)

    if strategy == "global":
        config = RectificationRunConfig(
            strategy="global",
            correlation_pad_width=int(params["correlation_pad_width"]),
            global_threshold=float(params["global_threshold"]),
            pixel_shift_power=defaults.pixel_shift_power,
            displacement_metric_family=displacement_metric_family,
            displacement_alpha=displacement_alpha,
            displacement_component_weights=displacement_component_weights,
            metric_formulation_id=metric_formulation_id,
        )
        normalized_params = dict(params)
        normalized_params["pixel_shift_power"] = config.pixel_shift_power
        return SweepCandidate(strategy="global", params=normalized_params, config=config)

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
            search_space.pixel_shift_powers,
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
                "pixel_shift_power": float(pixel_shift_power),
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
                pixel_shift_power,
            ) in values
        ]

    if strategy == "combined":
        power_blend_options = search_space.power_blend_search_space().coarse_options()
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
            search_space.pixel_shift_powers,
            power_blend_options,
        )
        candidates: List[Dict[str, object]] = []
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
            pixel_shift_power,
            power_blend,
        ) in values:
            if power_blend is not None and (bool(pixel_sigma_pyramid) or float(pixel_shift_power) != 1.0):
                continue
            power_blend_params = (
                power_blend.flat_params()
                if power_blend is not None
                else {
                    "pixel_power_blend_enabled": False,
                    "pixel_power_blend_raised_power": None,
                    "pixel_power_blend_alpha": 0.0,
                    "pixel_power_blend_delta_cap": 0.0,
                }
            )
            candidates.append({
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
                "combined_reference_mode": search_space.combined_reference_mode,
                "zero_fill_roll": bool(zero_fill_roll),
                "pixel_shift_power": float(pixel_shift_power),
                **power_blend_params,
            }
            )
        return candidates

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


_INT_REFINEMENT_KEYS = {
    "similarity_window_size",
    "refinement_window_size",
    "max_displacement",
    "sub_pix_factor",
    "correlation_pad_width",
}
_FLOAT_REFINEMENT_KEYS = {
    "gauss_sigma",
    "disp_gauss_sigma",
    "global_threshold",
    "pixel_threshold",
    "pixel_shift_power",
    "pixel_power_blend_raised_power",
    "pixel_power_blend_alpha",
    "pixel_power_blend_delta_cap",
}
_NUMERIC_REFINEMENT_KEYS = _INT_REFINEMENT_KEYS | _FLOAT_REFINEMENT_KEYS


def _search_space_values_for_key(search_space: SweepSearchSpace, key: str) -> Sequence[object]:
    if key == "similarity_window_size":
        return list(search_space.similarity_window_sizes)
    if key == "refinement_window_size":
        return list(search_space.refinement_window_sizes)
    if key == "max_displacement":
        return list(search_space.max_displacements)
    if key == "gauss_sigma":
        return list(search_space.gauss_sigmas)
    if key == "disp_gauss_sigma":
        return list(search_space.disp_gauss_sigmas)
    if key == "pixel_threshold":
        return list(search_space.pixel_thresholds)
    if key == "sub_pix_factor":
        return list(search_space.sub_pix_factors)
    if key == "pixel_sigma_pyramid":
        return list(search_space.pixel_sigma_pyramid)
    if key == "zero_fill_roll":
        return list(search_space.zero_fill_roll)
    if key == "pixel_shift_power":
        return list(search_space.pixel_shift_powers)
    if key == "pixel_power_blend_enabled":
        return list(search_space.pixel_power_blend_enabled)
    if key == "correlation_pad_width":
        return list(search_space.correlation_pad_widths)
    if key == "global_threshold":
        return list(search_space.global_thresholds)
    if key == "combined_use_global_avg_reference":
        return list(search_space.combined_use_global_avg_reference)
    return cast(Sequence[object], getattr(search_space, f"{key}s", []))


def _clean_refinement_float(value: float) -> float:
    return float(round(float(value), 12))


def _normalize_refinement_axis_value(key: str, value: object) -> object:
    if key in {"pixel_sigma_pyramid", "zero_fill_roll", "combined_use_global_avg_reference", "pixel_power_blend_enabled"}:
        return bool(value)

    if key in _INT_REFINEMENT_KEYS:
        numeric = int(round(float(value)))
        if key in {"similarity_window_size", "refinement_window_size"}:
            return _normalize_window_size(max(1, numeric))
        if key == "max_displacement":
            return max(0, numeric)
        if key == "sub_pix_factor":
            return max(1, numeric)
        if key == "correlation_pad_width":
            return max(0, numeric)
        return numeric

    if key in _FLOAT_REFINEMENT_KEYS:
        numeric = _clean_refinement_float(float(value))
        if key in {"global_threshold", "pixel_threshold"}:
            return _clean_refinement_float(min(1.0, max(0.0, numeric)))
        if key == "pixel_shift_power":
            return _clean_refinement_float(max(np.finfo(float).eps, numeric))
        return _clean_refinement_float(max(0.0, numeric))

    return value


def _dedupe_axis_values(values: Iterable[object]) -> List[object]:
    deduped: List[object] = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _numeric_axis_values(
    values: Sequence[object],
    current: object,
    key: str,
) -> List[object]:
    normalized = [
        _normalize_refinement_axis_value(key, value)
        for value in values
    ]
    normalized.append(_normalize_refinement_axis_value(key, current))
    deduped = _dedupe_axis_values(normalized)
    return sorted(deduped, key=lambda item: float(cast(float | int, item)))


def _axis_step(values: Sequence[object], current: object, key: str) -> float:
    if key not in _NUMERIC_REFINEMENT_KEYS:
        return 1.0
    numeric_values = [float(cast(float | int, value)) for value in _numeric_axis_values(values, current, key)]
    diffs = [
        abs(right - left)
        for left, right in zip(numeric_values[:-1], numeric_values[1:])
        if abs(right - left) > 0.0
    ]
    return min(diffs) if diffs else 1.0


def _auto_fine_axis_values(
    values: Sequence[object],
    current: object,
    key: str,
    refinement: SweepStageRefinementSpec,
) -> List[object]:
    current_value = _normalize_refinement_axis_value(key, current)
    if key not in _NUMERIC_REFINEMENT_KEYS:
        normalized_values = _dedupe_axis_values(
            _normalize_refinement_axis_value(key, value)
            for value in values
        )
        if current_value not in normalized_values:
            return [current_value]
        return _neighbor_values(normalized_values, current_value)

    numeric_values = _numeric_axis_values(values, current_value, key)
    if len(numeric_values) < 2:
        return [current_value]

    idx = numeric_values.index(current_value)
    raw_generated = [float(cast(float | int, current_value))]
    if idx > 0:
        previous_value = float(cast(float | int, numeric_values[idx - 1]))
        raw_generated.append((previous_value + float(cast(float | int, current_value))) / 2.0)
    if idx < len(numeric_values) - 1:
        next_value = float(cast(float | int, numeric_values[idx + 1]))
        raw_generated.append((next_value + float(cast(float | int, current_value))) / 2.0)

    if refinement.extrapolation != "inside_only":
        factor = 0.5 if refinement.extrapolation == "half_step_bounded" else 1.0
        if idx == 0 and idx < len(numeric_values) - 1:
            next_value = float(cast(float | int, numeric_values[idx + 1]))
            step = next_value - float(cast(float | int, current_value))
            if step > 0.0:
                raw_generated.append(float(cast(float | int, current_value)) - step * factor)
        if idx == len(numeric_values) - 1 and idx > 0:
            previous_value = float(cast(float | int, numeric_values[idx - 1]))
            step = float(cast(float | int, current_value)) - previous_value
            if step > 0.0:
                raw_generated.append(float(cast(float | int, current_value)) + step * factor)

    normalized_generated = [
        _normalize_refinement_axis_value(key, value)
        for value in raw_generated
    ]
    return sorted(
        _dedupe_axis_values(normalized_generated),
        key=lambda item: float(cast(float | int, item)),
    )


def _prepare_refinement_variant_params(
    base_params: Dict[str, object],
    variant_values: Dict[str, object],
) -> Dict[str, object]:
    variant = dict(base_params)
    variant.update(variant_values)
    variant.pop("gauss_size", None)
    variant.pop("disp_gauss_size", None)
    return variant


def _power_blend_flat_params(option: Optional[PowerBlendConfig]) -> Dict[str, object]:
    if option is None:
        return {
            "pixel_power_blend_enabled": False,
            "pixel_power_blend_raised_power": None,
            "pixel_power_blend_alpha": 0.0,
            "pixel_power_blend_delta_cap": 0.0,
        }
    return option.flat_params()


def _refinement_keys_for_candidate(candidate: SweepCandidate) -> List[str]:
    keys = _strategy_param_keys(candidate.strategy)
    if candidate.strategy == "combined" and candidate.config.power_blend_config is not None:
        return [
            key for key in keys
            if key not in {"pixel_sigma_pyramid", "pixel_shift_power"}
        ]
    return keys


def _refinement_sort_key(
    candidate: SweepCandidate,
    seed_params: Dict[str, object],
    keys: Sequence[str],
    axis_steps: Mapping[str, float],
) -> Tuple[int, float, str]:
    changed = 0
    distance = 0.0
    for key in keys:
        value = candidate.params.get(key)
        seed_value = seed_params.get(key)
        if value != seed_value:
            changed += 1
        if key in _NUMERIC_REFINEMENT_KEYS and value is not None and seed_value is not None:
            step = max(float(axis_steps.get(key, 1.0)), 1e-12)
            distance += abs(float(cast(float | int, value)) - float(cast(float | int, seed_value))) / step
        elif value != seed_value:
            distance += 1.0
    return (changed, _clean_refinement_float(distance), json.dumps(candidate.params, sort_keys=True))


def _local_auto_fine_candidates_for_seed(
    coarse: SweepCandidate,
    search_space: SweepSearchSpace,
    refinement: SweepStageRefinementSpec,
) -> List[SweepCandidate]:
    strategy = coarse.strategy
    seed_params = dict(coarse.params)
    keys = [key for key in _refinement_keys_for_candidate(coarse) if key in seed_params]
    power_blend_options: List[Optional[PowerBlendConfig]] = [None]
    if strategy == "combined":
        power_blend_options = search_space.power_blend_search_space().auto_options(
            coarse.config.power_blend_config,
            refinement.extrapolation,
        )
    axis_items: List[Tuple[str, List[object]]] = []
    axis_steps: Dict[str, float] = {}

    for key in keys:
        configured_values = _search_space_values_for_key(search_space, key)
        axis_items.append(
            (
                key,
                _auto_fine_axis_values(
                    configured_values,
                    seed_params[key],
                    key,
                    refinement,
                ),
            )
        )
        axis_steps[key] = _axis_step(configured_values, seed_params[key], key)

    if not axis_items:
        return [_build_candidate_from_params(strategy, seed_params)]

    candidates: List[SweepCandidate] = []
    seen = set()
    axis_keys = [key for key, _ in axis_items]
    axis_values = [values for _, values in axis_items]
    for values_combo in product(*axis_values):
        variant_values = dict(zip(axis_keys, values_combo))
        for power_blend in power_blend_options:
            if strategy == "combined":
                variant_values.update(_power_blend_flat_params(power_blend))
            variant_params = _prepare_refinement_variant_params(seed_params, variant_values)
            candidate = _build_candidate_from_params(strategy, variant_params)
            sig = _candidate_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: _refinement_sort_key(candidate, seed_params, keys, axis_steps)
    )
    return candidates


def _round_robin_candidate_groups(
    groups: Sequence[Sequence[SweepCandidate]],
    budget: Optional[int],
) -> List[SweepCandidate]:
    if budget is not None and budget <= 0:
        return []
    positions = [0 for _ in groups]
    selected: List[SweepCandidate] = []
    seen = set()

    while True:
        progressed = False
        for group_idx, group in enumerate(groups):
            while positions[group_idx] < len(group):
                candidate = group[positions[group_idx]]
                positions[group_idx] += 1
                sig = _candidate_signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                selected.append(candidate)
                progressed = True
                break
            if budget is not None and len(selected) >= budget:
                return selected
        if not progressed:
            break
    return selected


def _generate_neighbor_refined_candidates(
    ranked_candidates: Sequence[Dict[str, object]],
    candidate_lookup: Dict[str, SweepCandidate],
    search_spaces: Dict[str, SweepSearchSpace],
    top_k: int = 8,
    budget: Optional[int] = None,
) -> List[SweepCandidate]:
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

        for key in _refinement_keys_for_candidate(coarse):
            if key not in params:
                continue
            strategy_values = _search_space_values_for_key(search_space, key)

            for neighbor in _neighbor_values(strategy_values, params[key]):
                if neighbor == params[key]:
                    continue
                variant = _prepare_refinement_variant_params(params, {key: neighbor})
                candidate = _build_candidate_from_params(strategy, variant)
                sig = _candidate_signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                generated.append(candidate)

        if strategy == "combined":
            for power_blend in search_space.power_blend_search_space().neighbor_options(
                coarse.config.power_blend_config
            ):
                if power_blend == coarse.config.power_blend_config:
                    continue
                variant = _prepare_refinement_variant_params(
                    params, _power_blend_flat_params(power_blend)
                )
                candidate = _build_candidate_from_params(strategy, variant)
                sig = _candidate_signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                generated.append(candidate)

    generated.sort(key=lambda item: json.dumps(item.params, sort_keys=True))
    return _limit_candidates(generated, budget)


def generate_refined_candidates(
    ranked_candidates: Sequence[Dict[str, object]],
    candidate_lookup: Dict[str, SweepCandidate],
    search_spaces: Dict[str, SweepSearchSpace],
    top_k: int = 8,
    budget: Optional[int] = None,
    refinement: Optional[SweepStageRefinementSpec] = None,
) -> List[SweepCandidate]:
    """Generate stage-2 refinements around Top-K coarse candidates."""

    refinement_spec = refinement or SweepStageRefinementSpec()
    if refinement_spec.method == "neighbor":
        return _generate_neighbor_refined_candidates(
            ranked_candidates=ranked_candidates,
            candidate_lookup=candidate_lookup,
            search_spaces=search_spaces,
            top_k=top_k,
            budget=budget,
        )

    top_items = [item for item in ranked_candidates if item.get("candidate_id") in candidate_lookup][:top_k]
    groups: List[List[SweepCandidate]] = []
    for item in top_items:
        candidate_id = cast(str, item.get("candidate_id"))
        coarse = candidate_lookup[candidate_id]
        search_space = search_spaces[coarse.strategy]
        groups.append(
            _local_auto_fine_candidates_for_seed(
                coarse,
                search_space,
                refinement_spec,
            )
        )

    return _round_robin_candidate_groups(groups, budget)


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


def _parse_experiment_spec(
    path: Path,
    relocation_rules: Sequence[PathRelocationRule] = (),
) -> Tuple[SweepExperimentSpec, Dict[str, object]]:
    if not path.exists() and not Path(f"{path}.bak").exists():
        raise FileNotFoundError(f"Sweep spec file not found: {path}")
    payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a JSON object.")
    resolved_payload = _resolve_spec_relative_paths_payload(cast(Dict[str, object], payload), path)
    relocated_payload, relocation_report = _apply_relocation_to_spec_payload(
        resolved_payload,
        relocation_rules,
    )
    return SweepExperimentSpec.from_dict(relocated_payload), relocation_report


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
    power_blend_workspace: Dict[str, object] = {}
    if result.pixel_similarity is not None:
        pixel_displacements_x = getattr(result.pixel_similarity, "disps_x", None)
        pixel_displacements_y = getattr(result.pixel_similarity, "disps_y", None)
        pixel_blended_scores = getattr(result.pixel_similarity, "blended_score_maps", None)
        component_scores = getattr(result.pixel_similarity, "component_score_maps", None)
        if isinstance(component_scores, dict):
            for key, value in component_scores.items():
                pixel_component_scores[str(key)] = np.asarray(value)
        power_blend_artifacts = getattr(result.pixel_similarity, "power_blend_artifacts", None)
        if power_blend_artifacts is not None:
            power_blend_workspace.update(power_blend_artifacts.workspace_values())

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
        "runtime_warnings_json": np.array(json.dumps(result.runtime_warnings, sort_keys=True)),
        "displacement_metric_family": np.array(config.displacement_metric_family),
        "displacement_alpha": np.array(config.displacement_alpha, dtype=np.float64),
        "metric_formulation_id": np.array(config.metric_formulation_id),
    }
    for component_name, component_values in pixel_component_scores.items():
        workspace_data[f"pixel_component_score_{component_name}"] = component_values
    workspace_data.update(power_blend_workspace)
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


def _read_dataset_identity_payloads(output_dir: Path) -> Dict[str, Dict[str, object]]:
    for metadata_path in (
        output_dir / RUN_STATE_FILENAME,
        Path(f"{output_dir / RUN_STATE_FILENAME}.bak"),
        output_dir / "experiment_manifest.json",
        Path(f"{output_dir / 'experiment_manifest.json'}.bak"),
    ):
        if not metadata_path.exists():
            continue
        try:
            payload, _, _ = read_json_first_valid(
                metadata_path,
                validator=lambda obj: isinstance(obj, dict),
            )
        except Exception:
            continue
        root = cast(Dict[str, object], payload)
        if metadata_path.name.startswith(RUN_STATE_FILENAME):
            fingerprint_payload = root.get("fingerprint_payload")
            datasets_payload = (
                fingerprint_payload.get("datasets")
                if isinstance(fingerprint_payload, dict)
                else None
            )
        else:
            datasets_payload = root.get("datasets")
        if not isinstance(datasets_payload, list):
            continue
        identities: Dict[str, Dict[str, object]] = {}
        for entry in datasets_payload:
            if not isinstance(entry, dict):
                continue
            dataset_id = str(entry.get("dataset_id", "")).strip()
            if dataset_id:
                identities[dataset_id] = cast(Dict[str, object], entry)
        if identities:
            return identities
    return {}


def _cached_prepared_dataset(
    *,
    dataset: SweepDatasetSpec,
    cache_path: Path,
    expected_identity: Optional[Mapping[str, object]],
    hdr_enabled: bool,
) -> Optional[PreparedDataset]:
    if expected_identity is None or not cache_path.exists():
        return None
    views_arr = _load_views(cache_path)
    if hdr_enabled and views_arr.ndim != 5:
        return None
    if not hdr_enabled and views_arr.ndim == 5:
        return None
    if (
        dataset.hdr_exposure_times is not None
        and hdr_enabled
        and len(dataset.hdr_exposure_times) != views_arr.shape[0]
    ):
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' exposure time count does not match cached exposure axis."
        )
    stat = cache_path.stat()
    prepared = PreparedDataset(
        dataset_id=dataset.dataset_id,
        views_path=cache_path,
        hdr_enabled=bool(hdr_enabled),
        hdr_exposure_times=dataset.hdr_exposure_times,
        normalize_to_8bit=dataset.normalize_to_8bit,
        views_shape=tuple(int(dim) for dim in views_arr.shape),
        views_dtype=str(views_arr.dtype),
        views_itemsize=int(views_arr.dtype.itemsize),
        resolved_views_path=str(cache_path.resolve()),
        views_file_size_bytes=int(stat.st_size),
        views_file_mtime_ns=int(stat.st_mtime_ns),
        views_content_sha256=_file_content_sha256(cache_path),
    )
    if _dataset_signature(prepared) != _dataset_signature_from_identity_payload(expected_identity):
        return None
    return prepared


def _prepare_dataset(
    dataset: SweepDatasetSpec,
    output_dir: Path,
    dataset_identities: Optional[Mapping[str, Mapping[str, object]]] = None,
    *,
    allow_dataset_generation: bool = True,
) -> PreparedDataset:
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
            views_content_sha256=_file_content_sha256(views_path),
        )

    dataset_dir = output_dir / "datasets" / dataset.dataset_id
    expected_identity = (
        dataset_identities.get(dataset.dataset_id)
        if dataset_identities is not None
        else None
    )
    if not allow_dataset_generation:
        if dataset.source_image is not None:
            cached = _cached_prepared_dataset(
                dataset=dataset,
                cache_path=dataset_dir / "generated_views.npy",
                expected_identity=expected_identity,
                hdr_enabled=False,
            )
            if cached is not None:
                return cached
        if dataset.source_image_dir is not None:
            cached = _cached_prepared_dataset(
                dataset=dataset,
                cache_path=dataset_dir / "generated_exposure_views.npy",
                expected_identity=expected_identity,
                hdr_enabled=True if dataset.hdr_enabled is None else bool(dataset.hdr_enabled),
            )
            if cached is not None:
                return cached
        raise ValueError(
            f"Dataset '{dataset.dataset_id}' requires view generation; "
            "planned-run preview cannot generate dataset caches."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if dataset.source_image is not None:
        source_image_path = Path(dataset.source_image)
        if not source_image_path.exists():
            cached = _cached_prepared_dataset(
                dataset=dataset,
                cache_path=dataset_dir / "generated_views.npy",
                expected_identity=expected_identity,
                hdr_enabled=False,
            )
            if cached is not None:
                return cached
            raise FileNotFoundError(
                f"Dataset '{dataset.dataset_id}' source_image not found after relocation: "
                f"{source_image_path}"
            )
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
            image_path=source_image_path,
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
            views_content_sha256=_file_content_sha256(views_path),
        )

    # source_image_dir branch
    source_image_dir_path = Path(cast(str, dataset.source_image_dir))
    if not source_image_dir_path.exists():
        cached = _cached_prepared_dataset(
            dataset=dataset,
            cache_path=dataset_dir / "generated_exposure_views.npy",
            expected_identity=expected_identity,
            hdr_enabled=True if dataset.hdr_enabled is None else bool(dataset.hdr_enabled),
        )
        if cached is not None:
            return cached
        raise FileNotFoundError(
            f"Dataset '{dataset.dataset_id}' source_image_dir not found after relocation: "
            f"{source_image_dir_path}"
        )
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
        image_dir=source_image_dir_path,
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
        views_content_sha256=_file_content_sha256(views_path),
    )


def _prepare_datasets(
    spec: SweepExperimentSpec,
    output_dir: Path,
    dataset_identities: Optional[Mapping[str, Mapping[str, object]]] = None,
    *,
    allow_dataset_generation: bool = True,
) -> List[PreparedDataset]:
    seen: set[str] = set()
    prepared: List[PreparedDataset] = []
    for dataset in spec.datasets:
        if dataset.dataset_id in seen:
            raise ValueError(f"Duplicate dataset id '{dataset.dataset_id}' in experiment spec.")
        seen.add(dataset.dataset_id)
        prepared.append(
            _prepare_dataset(
                dataset,
                output_dir=output_dir,
                dataset_identities=dataset_identities,
                allow_dataset_generation=allow_dataset_generation,
            )
        )
    prepared.sort(key=lambda item: item.dataset_id)
    return prepared


RUN_STATE_FILENAME = "run_state.json"
RUN_STATE_SCHEMA_VERSION = 3
CANDIDATE_ID_HASH_LEN = 12
DATASET_CONFIG_KEYS = {"hdr_enabled", "hdr_exposure_times", "normalize_to_8bit"}


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: object, length: Optional[int] = None) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def _file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dataset_identity_payload(dataset: PreparedDataset) -> Dict[str, object]:
    return {
        "dataset_id": dataset.dataset_id,
        "hdr_enabled": bool(dataset.hdr_enabled),
        "hdr_exposure_times": dataset.hdr_exposure_times,
        "normalize_to_8bit": bool(dataset.normalize_to_8bit),
        "resolved_views_path": dataset.resolved_views_path or str(dataset.views_path.resolve()),
        "views_shape": list(dataset.views_shape),
        "views_dtype": dataset.views_dtype,
        "views_itemsize": int(dataset.views_itemsize),
        "views_file_size_bytes": dataset.views_file_size_bytes,
        "views_file_mtime_ns": dataset.views_file_mtime_ns,
        "views_content_sha256": dataset.views_content_sha256,
    }


def _dataset_signature_payload(identity_payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        "dataset_id": str(identity_payload.get("dataset_id", "")),
        "hdr_enabled": bool(identity_payload.get("hdr_enabled", False)),
        "hdr_exposure_times": identity_payload.get("hdr_exposure_times"),
        "normalize_to_8bit": bool(identity_payload.get("normalize_to_8bit", False)),
        "views_shape": list(cast(Sequence[object], identity_payload.get("views_shape", []))),
        "views_dtype": str(identity_payload.get("views_dtype", "")),
        "views_itemsize": int(identity_payload.get("views_itemsize", 0) or 0),
        "views_file_size_bytes": identity_payload.get("views_file_size_bytes"),
        "views_content_sha256": identity_payload.get("views_content_sha256"),
    }


def _dataset_signature_from_identity_payload(identity_payload: Mapping[str, object]) -> str:
    return _hash_payload(_dataset_signature_payload(identity_payload))


def _dataset_signature(dataset: PreparedDataset) -> str:
    return _dataset_signature_from_identity_payload(_dataset_identity_payload(dataset))


def _run_signature(stage_name: str, candidate_signature: str, dataset_signature: str) -> str:
    return _hash_payload(
        {
            "stage": stage_name,
            "candidate_signature": candidate_signature,
            "dataset_signature": dataset_signature,
        }
    )


def _artifact_paths_payload(run_dir: Path) -> Dict[str, str]:
    return {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.json"),
        "metric_record": str(run_dir / "metric_record.json"),
        "rectified_views": str(run_dir / "rectified_views.npy"),
        "workspace": str(run_dir / "workspace.npz"),
        "avg_rectified": str(run_dir / "avg_rectified.png"),
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


def _normalize_fingerprint_payload_for_resume_compat(
    payload: Dict[str, object],
) -> Dict[str, object]:
    """
    Remove machine-specific path details and volatile cache mtimes.

    This keeps the precise fingerprint for normal operation, while allowing a
    copied run directory to be adopted on another machine if the semantic spec
    and prepared dataset identities are otherwise unchanged.
    """

    normalized = dict(payload)
    normalized["schema_version"] = "<schema_version>"
    if normalized.get("output_dir") is not None:
        normalized["output_dir"] = "<output_dir>"

    spec_payload = normalized.get("spec")
    if isinstance(spec_payload, dict):
        spec_snapshot = dict(spec_payload)
        if spec_snapshot.get("output_dir") is not None:
            spec_snapshot["output_dir"] = "<output_dir>"
        # Scheduling is an execution policy, not part of the experiment's
        # semantic identity.  This lets an interrupted sweep resume in a
        # different scheduler mode without creating a new generation.
        stages_snapshot = spec_snapshot.get("stages")
        if isinstance(stages_snapshot, dict):
            normalized_stages = dict(stages_snapshot)
            normalized_stages.pop("scheduling", None)
            spec_snapshot["stages"] = normalized_stages
        experimental_snapshot = spec_snapshot.get("experimental_metric_sweep")
        if isinstance(experimental_snapshot, dict):
            normalized_experimental = dict(experimental_snapshot)
            search_stages_snapshot = normalized_experimental.get("search_stages")
            if isinstance(search_stages_snapshot, dict):
                normalized_search_stages = dict(search_stages_snapshot)
                normalized_search_stages.pop("scheduling", None)
                normalized_experimental["search_stages"] = normalized_search_stages
            spec_snapshot["experimental_metric_sweep"] = normalized_experimental
        scoring_snapshot = spec_snapshot.get("scoring")
        if isinstance(scoring_snapshot, dict):
            normalized_scoring = dict(scoring_snapshot)
            if normalized_scoring.get("profile_config_path") is not None:
                normalized_scoring["profile_config_path"] = "<profile_config_path>"
            spec_snapshot["scoring"] = normalized_scoring
        datasets_snapshot = spec_snapshot.get("datasets")
        if isinstance(datasets_snapshot, list):
            normalized_datasets: List[object] = []
            for entry in datasets_snapshot:
                if not isinstance(entry, dict):
                    normalized_datasets.append(entry)
                    continue
                dataset_entry = dict(entry)
                for key in ("views_path", "source_image", "source_image_dir"):
                    if dataset_entry.get(key) is not None:
                        dataset_entry[key] = f"<{key}>"
                normalized_datasets.append(dataset_entry)
            spec_snapshot["datasets"] = normalized_datasets
        spec_snapshot = _normalize_metric_sweep_fingerprint_defaults(spec_snapshot)
        normalized["spec"] = spec_snapshot

    datasets_payload = normalized.get("datasets")
    if isinstance(datasets_payload, list):
        normalized_dataset_payload: List[object] = []
        for entry in datasets_payload:
            if not isinstance(entry, dict):
                normalized_dataset_payload.append(entry)
                continue
            dataset_entry = dict(entry)
            if dataset_entry.get("resolved_views_path") is not None:
                dataset_entry["resolved_views_path"] = "<resolved_views_path>"
            dataset_entry.pop("views_file_mtime_ns", None)
            normalized_dataset_payload.append(dataset_entry)
        normalized["datasets"] = normalized_dataset_payload

    return normalized


def _normalize_validation_oracle_fingerprint_defaults(
    spec_snapshot: Dict[str, object],
) -> Dict[str, object]:
    normalized_spec = dict(spec_snapshot)
    oracle_raw = normalized_spec.get("validation_oracle_sweep")
    if not isinstance(oracle_raw, dict):
        return normalized_spec

    oracle = dict(oracle_raw)
    oracle.pop("strict", None)
    oracle.setdefault("oracle_mode", VALIDATION_ORACLE_MODE_TRIO)
    oracle.setdefault("skip_image_metric_stage", False)
    defaults_raw = oracle.get("defaults")
    if isinstance(defaults_raw, dict):
        defaults = dict(defaults_raw)
        defaults.pop("strict", None)
        oracle["defaults"] = defaults
    scenes_raw = oracle.get("scenes")
    if isinstance(scenes_raw, list):
        scenes: List[object] = []
        for scene_raw in scenes_raw:
            if isinstance(scene_raw, dict):
                scene = dict(scene_raw)
                scene.pop("strict", None)
                scenes.append(scene)
            else:
                scenes.append(scene_raw)
        oracle["scenes"] = scenes
    normalized_spec["validation_oracle_sweep"] = oracle
    return normalized_spec


def _normalize_metric_sweep_fingerprint_defaults(
    spec_snapshot: Dict[str, object],
) -> Dict[str, object]:
    normalized_spec = dict(spec_snapshot)
    experimental_raw = normalized_spec.get("experimental_metric_sweep")
    if not isinstance(experimental_raw, dict):
        return _normalize_validation_oracle_fingerprint_defaults(normalized_spec)

    experimental = dict(experimental_raw)
    source = str(
        experimental.get("candidate_source", METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST)
    ).strip().lower()
    if source in METRIC_CANDIDATE_SOURCES:
        experimental["candidate_source"] = source
        if source == METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST:
            experimental["search_stages"] = None
    elif "candidate_source" not in experimental:
        experimental["candidate_source"] = METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST
        experimental["search_stages"] = None
    normalized_spec["experimental_metric_sweep"] = experimental
    return _normalize_validation_oracle_fingerprint_defaults(normalized_spec)


def _fingerprint_payloads_resume_compatible(
    existing_payload_raw: object,
    current_payload: Dict[str, object],
) -> bool:
    if not isinstance(existing_payload_raw, dict):
        return False
    existing_payload = cast(Dict[str, object], existing_payload_raw)
    return _normalize_fingerprint_payload_for_resume_compat(
        existing_payload
    ) == _normalize_fingerprint_payload_for_resume_compat(current_payload)


def _dataset_signatures_from_fingerprint_payload(
    fingerprint_payload_raw: object,
) -> Dict[str, str]:
    if not isinstance(fingerprint_payload_raw, dict):
        return {}
    datasets_payload = fingerprint_payload_raw.get("datasets")
    if not isinstance(datasets_payload, list):
        return {}
    signatures: Dict[str, str] = {}
    for entry in datasets_payload:
        if not isinstance(entry, dict):
            continue
        dataset_id = str(entry.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        signatures[dataset_id] = _dataset_signature_from_identity_payload(
            cast(Dict[str, object], entry)
        )
    return signatures


def _fingerprint_payload_datasets_compatible(
    existing_payload_raw: object,
    current_payload: Dict[str, object],
) -> bool:
    existing = _dataset_signatures_from_fingerprint_payload(existing_payload_raw)
    current = _dataset_signatures_from_fingerprint_payload(current_payload)
    return bool(existing) and existing == current


def _new_generation_id(action: str, fingerprint: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"gen_{timestamp}_{action}_{fingerprint[:8]}"


def _generation_dir(output_dir: Path, generation_id: str) -> Path:
    return output_dir / "generations" / generation_id


def _ensure_unique_generation_id(output_dir: Path, generation_id: str) -> str:
    base_id = generation_id
    counter = 1
    while _generation_dir(output_dir, generation_id).exists():
        generation_id = f"{base_id}_{counter:02d}"
        counter += 1
    return generation_id


def _ensure_run_state_v2(
    state: Dict[str, object],
    output_dir: Path,
    fingerprint_payload: Dict[str, object],
) -> Dict[str, object]:
    if int(state.get("schema_version", 1) or 1) >= RUN_STATE_SCHEMA_VERSION:
        state.setdefault("spec_generations", [])
        state.setdefault("merge_history", [])
        state.setdefault("active_generation_id", None)
        return state

    migrated_at = _utc_now_iso()
    legacy_generation_id = f"gen_legacy_{migrated_at.replace(':', '').replace('-', '')}"
    state["schema_version"] = RUN_STATE_SCHEMA_VERSION
    state.setdefault("spec_generations", [])
    cast(List[Dict[str, object]], state["spec_generations"]).append(
        {
            "generation_id": legacy_generation_id,
            "action": "migrated_v1",
            "created_at": migrated_at,
            "experiment_fingerprint": state.get("experiment_fingerprint"),
            "fingerprint_payload": state.get("fingerprint_payload", fingerprint_payload),
        }
    )
    state["active_generation_id"] = legacy_generation_id
    state.setdefault("merge_history", [])
    runs = cast(Dict[str, Dict[str, object]], state.setdefault("runs", {}))
    dataset_signatures = _dataset_signatures_from_fingerprint_payload(
        state.get("fingerprint_payload", fingerprint_payload)
    )
    for run_uid, entry in runs.items():
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage", "") or str(run_uid).split("_", 1)[0])
        candidate_id = str(entry.get("candidate_id", ""))
        dataset_id = str(entry.get("dataset_id", ""))
        run_dir = Path(str(entry.get("run_dir", "")))
        config_payload: Optional[Dict[str, object]] = None
        if run_dir:
            try:
                loaded, _, _ = read_json_first_valid(
                    run_dir / "config.json",
                    validator=lambda obj: isinstance(obj, dict),
                )
                config_payload = cast(Dict[str, object], loaded)
            except Exception:  # noqa: BLE001 - migration keeps partial state usable.
                config_payload = None
        if config_payload is not None:
            entry["candidate_signature"] = _candidate_signature_from_config_payload(config_payload)
        else:
            entry.setdefault("candidate_signature", _hash_payload({"candidate_id": candidate_id}))
        entry["dataset_signature"] = dataset_signatures.get(
            dataset_id,
            _hash_payload({"dataset_id": dataset_id}),
        )
        entry["run_signature"] = _run_signature(
            stage_name,
            str(entry["candidate_signature"]),
            str(entry["dataset_signature"]),
        )
        entry["generation_id"] = legacy_generation_id
        entry.setdefault("active", True)
        if run_dir:
            entry["artifact_paths"] = _artifact_paths_payload(run_dir)

    path_compatibility = cast(Dict[str, object], state.setdefault("path_compatibility", {}))
    path_compatibility.setdefault("output_dir", str(output_dir.resolve()))
    return state


def _register_spec_generation(
    *,
    state: Dict[str, object],
    generation_id: str,
    action: str,
    fingerprint: str,
    fingerprint_payload: Dict[str, object],
    merge_sources: Sequence[Path],
) -> None:
    generations = cast(List[Dict[str, object]], state.setdefault("spec_generations", []))
    if not any(str(item.get("generation_id")) == generation_id for item in generations):
        generations.append(
            {
                "generation_id": generation_id,
                "action": action,
                "created_at": _utc_now_iso(),
                "experiment_fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "merge_sources": [str(path.resolve()) for path in merge_sources],
            }
        )
    state["active_generation_id"] = generation_id


def _record_relocation_history(
    *,
    state: Dict[str, object],
    old_output_dir: str,
    new_output_dir: str,
    relocation_rules: Sequence[PathRelocationRule],
    relocation_report: Optional[Mapping[str, object]],
) -> None:
    if not old_output_dir or old_output_dir == new_output_dir:
        return
    history = cast(List[Dict[str, object]], state.setdefault("relocation_history", []))
    history.append(
        {
            "timestamp": _utc_now_iso(),
            "old_output_dir": old_output_dir,
            "new_output_dir": new_output_dir,
            "path_mappings": _relocation_rules_to_payload(relocation_rules),
            "applied": list(cast(Sequence[object], (relocation_report or {}).get("applied", []))),
        }
    )


def _select_active_generation(
    *,
    output_dir: Path,
    state: Dict[str, object],
    fingerprint: str,
    action: str,
) -> str:
    if action == "resume":
        active = state.get("active_generation_id")
        if isinstance(active, str) and active:
            return active
    return _ensure_unique_generation_id(
        output_dir,
        _new_generation_id(action, fingerprint),
    )


def _persist_run_state(state_path: Path, state: Dict[str, object]) -> None:
    state["updated_at"] = _utc_now_iso()
    write_json_atomic(state_path, state, indent=2, sort_keys=False, create_backup=True)


def _adaptive_learning_state_for_host(
    run_state: Optional[Dict[str, object]],
) -> sweep_scheduler.AdaptiveLearningState:
    """Load learned profiles only when the host memory signature still matches."""

    if run_state is None:
        return sweep_scheduler.AdaptiveLearningState()
    signature = sweep_scheduler.adaptive_host_memory_signature()
    stored = run_state.get("adaptive_scheduler_learning")
    if not isinstance(stored, Mapping) or stored.get("host_memory_signature") != signature:
        run_state["adaptive_scheduler_learning"] = {
            "host_memory_signature": signature,
            **sweep_scheduler.AdaptiveLearningState().to_dict(),
        }
        return sweep_scheduler.AdaptiveLearningState()
    return sweep_scheduler.AdaptiveLearningState.from_dict(stored)


def _persist_adaptive_learning_state(
    run_state: Optional[Dict[str, object]],
    learning: sweep_scheduler.AdaptiveLearningState,
) -> None:
    if run_state is None:
        return
    run_state["adaptive_scheduler_learning"] = {
        "host_memory_signature": sweep_scheduler.adaptive_host_memory_signature(),
        **learning.to_dict(),
    }


def _load_or_initialize_run_state(
    output_dir: Path,
    fingerprint_payload: Dict[str, object],
    *,
    allow_extend: bool = False,
    allow_merge: bool = False,
    adopt_relocated: bool = False,
    relocation_rules: Sequence[PathRelocationRule] = (),
    relocation_report: Optional[Mapping[str, object]] = None,
    merge_sources: Sequence[Path] = (),
    dry_run: bool = False,
) -> Tuple[Dict[str, object], Path, str]:
    state_path = output_dir / RUN_STATE_FILENAME
    fingerprint = _compute_experiment_fingerprint(fingerprint_payload)
    action = "resume"
    if allow_extend and allow_merge:
        action = "extend_merge"
    elif allow_extend:
        action = "extend"
    elif allow_merge:
        action = "merge"

    if state_path.exists() or Path(f"{state_path}.bak").exists():
        loaded, _, _ = read_json_first_valid(
            state_path,
            validator=lambda obj: isinstance(obj, dict),
        )
        state = cast(Dict[str, object], loaded)
        state = _ensure_run_state_v2(state, output_dir, fingerprint_payload)
        path_compatibility = cast(
            Dict[str, object],
            state.setdefault("path_compatibility", {}),
        )
        existing_output_dir = str(path_compatibility.get("output_dir", ""))
        current_output_dir = str(output_dir.resolve())
        existing_fingerprint = str(state.get("experiment_fingerprint", ""))
        if existing_fingerprint != fingerprint:
            if _fingerprint_payloads_resume_compatible(
                state.get("fingerprint_payload"),
                fingerprint_payload,
            ):
                if (
                    existing_output_dir
                    and existing_output_dir != current_output_dir
                    and not adopt_relocated
                ):
                    raise ValueError(
                        "run_state.json output path mismatch detected for a relocated sweep. "
                        f"state={existing_output_dir}, current={current_output_dir}. "
                        "Use --adopt-relocated with --relocate-path OLD=NEW after verifying "
                        "the copied sweep root and dataset paths."
                    )
                state["experiment_fingerprint"] = fingerprint
                state["fingerprint_payload"] = fingerprint_payload
                _record_relocation_history(
                    state=state,
                    old_output_dir=existing_output_dir,
                    new_output_dir=current_output_dir,
                    relocation_rules=relocation_rules,
                    relocation_report=relocation_report,
                )
                path_compatibility["output_dir"] = str(output_dir.resolve())
                if not dry_run:
                    _persist_run_state(state_path, state)
            elif (allow_extend or allow_merge) and _fingerprint_payload_datasets_compatible(
                state.get("fingerprint_payload"),
                fingerprint_payload,
            ):
                state["experiment_fingerprint"] = fingerprint
                state["fingerprint_payload"] = fingerprint_payload
                path_compatibility["output_dir"] = str(output_dir.resolve())
            else:
                raise ValueError(
                    "Output directory contains run_state.json with a different experiment "
                    "fingerprint. Refusing to resume with mismatched spec/dataset/path. "
                    "Use --extend-existing or --merge-from only when the datasets are compatible."
                )
        existing_output_dir = str(path_compatibility.get("output_dir", ""))
        if existing_output_dir and existing_output_dir != current_output_dir:
            if not adopt_relocated:
                raise ValueError(
                    "run_state.json output path mismatch detected. "
                    f"state={existing_output_dir}, current={current_output_dir}. "
                    "Use --adopt-relocated with --relocate-path OLD=NEW after verifying "
                    "the copied sweep root and dataset paths."
                )
            _record_relocation_history(
                state=state,
                old_output_dir=existing_output_dir,
                new_output_dir=current_output_dir,
                relocation_rules=relocation_rules,
                relocation_report=relocation_report,
            )
            path_compatibility["output_dir"] = current_output_dir
        generation_id = _select_active_generation(
            output_dir=output_dir,
            state=state,
            fingerprint=fingerprint,
            action=action,
        )
        if action != "resume" or not state.get("active_generation_id"):
            _register_spec_generation(
                state=state,
                generation_id=generation_id,
                action=action,
                fingerprint=fingerprint,
                fingerprint_payload=fingerprint_payload,
                merge_sources=merge_sources,
            )
        if not dry_run:
            _persist_run_state(state_path, state)
        return state, state_path, generation_id

    generation_id = _ensure_unique_generation_id(
        output_dir,
        _new_generation_id("run" if not allow_merge else action, fingerprint),
    )
    state: Dict[str, object] = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "experiment_fingerprint": fingerprint,
        "path_compatibility": {
            "output_dir": str(output_dir.resolve()),
        },
        "fingerprint_payload": fingerprint_payload,
        "active_generation_id": generation_id,
        "spec_generations": [
            {
                "generation_id": generation_id,
                "action": "run" if action == "resume" else action,
                "created_at": _utc_now_iso(),
                "experiment_fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "merge_sources": [str(path.resolve()) for path in merge_sources],
            }
        ],
        "merge_history": [],
        "stages": {},
        "runs": {},
    }
    if not dry_run:
        _persist_run_state(state_path, state)
    return state, state_path, generation_id


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


def _metric_formulation_payload_for_task(task: Dict[str, object]) -> Dict[str, object]:
    family, alpha, component_weights, formulation_id = _metric_formulation_from_params(
        dict(cast(Dict[str, object], task["params"]))
    )
    return {
        "id": formulation_id,
        "family": family,
        "alpha": float(alpha),
        "component_weights": (
            None if component_weights is None else dict(component_weights)
        ),
    }


def _normalize_cached_metric_record_identity(
    task: Dict[str, object],
    metric_record: Dict[str, object],
    artifact_paths: Dict[str, str],
) -> Dict[str, object]:
    normalized = dict(metric_record)
    normalized.update(
        {
            "run": str(task["run_uid"]),
            "run_uid": str(task["run_uid"]),
            "stage": str(task["stage"]),
            "candidate_id": str(task["candidate_id"]),
            "dataset_id": str(task["dataset_id"]),
            "strategy": str(task["strategy"]),
            "config": dict(cast(Dict[str, object], task["config"])),
            "metric_formulation": _metric_formulation_payload_for_task(task),
            "candidate_signature": task.get("candidate_signature"),
            "dataset_signature": task.get("dataset_signature"),
            "run_signature": task.get("run_signature"),
            "generation_id": task.get("generation_id"),
            "artifact_paths": artifact_paths,
        }
    )
    return normalized


def _load_cached_stage_result(
    task: Dict[str, object],
    run_dir: Path,
    *,
    persist_identity: bool = True,
) -> Dict[str, object]:
    config_payload, _, _ = read_json_first_valid(
        run_dir / "config.json",
        validator=lambda obj: isinstance(obj, dict),
    )
    metric_record_payload, _, _ = read_json_first_valid(
        run_dir / "metric_record.json",
        validator=lambda obj: isinstance(obj, dict),
    )
    artifact_paths = _artifact_paths_payload(run_dir)
    config_payload = dict(cast(Dict[str, object], config_payload))
    task_config = dict(cast(Dict[str, object], task["config"]))
    metric_record = _normalize_cached_metric_record_identity(
        task,
        dict(cast(Dict[str, object], metric_record_payload)),
        artifact_paths,
    )
    if persist_identity:
        if config_payload != task_config:
            write_json_atomic(
                run_dir / "config.json",
                task_config,
                indent=2,
                sort_keys=False,
                create_backup=True,
            )
        if metric_record != metric_record_payload:
            write_json_atomic(
                run_dir / "metric_record.json",
                metric_record,
                indent=2,
                sort_keys=False,
                create_backup=True,
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
        "config": task_config,
        "params": task["params"],
        "metric_record": metric_record,
        "candidate_signature": task.get("candidate_signature"),
        "dataset_signature": task.get("dataset_signature"),
        "run_signature": task.get("run_signature"),
        "generation_id": task.get("generation_id"),
        "artifact_paths": artifact_paths,
        "reused_artifacts": True,
    }


def _scoring_metric_names(scoring: Optional[SweepScoringSpec]) -> List[str]:
    if scoring is None:
        return []
    names: List[str] = []
    for metric_name, weight in scoring.metric_weights.items():
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric) or numeric == 0.0:
            continue
        names.append(str(metric_name))
    return names


def _metric_value_is_available(metrics: Mapping[str, object], metric_name: str) -> bool:
    try:
        value = float(cast(float, metrics.get(metric_name)))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def _effective_metric_scope(metric_record: Mapping[str, object], scoring: SweepScoringSpec) -> str:
    metrics = metric_record.get(scoring.metric_scope)
    if isinstance(metrics, dict):
        return scoring.metric_scope
    return "global_and_local_metrics"


def _workspace_metric_image_for_scope(workspace_path: Path, metric_scope: str) -> np.ndarray:
    payload, _, _ = read_npz_first_valid(
        workspace_path,
        allow_pickle=True,
        validator=lambda obj: "rectified_views" in obj and "average_rectified_image" in obj,
    )
    image_key = (
        "average_rectified_global_image"
        if metric_scope == "global_only_metrics"
        else "average_rectified_image"
    )
    image = cast(Optional[np.ndarray], payload.get(image_key))
    if not isinstance(image, np.ndarray):
        raise ValueError(f"workspace.npz missing {image_key}.")
    return np.asarray(image, dtype=np.float64)


def _backfill_cached_stage_result_metrics(
    cached_result: Dict[str, object],
    run_dir: Path,
    scoring: Optional[SweepScoringSpec],
    *,
    dry_run: bool,
) -> Tuple[bool, List[str]]:
    """Populate missing scoring metrics in a reused run's metric record."""

    if scoring is None:
        return False, []
    required_metrics = _scoring_metric_names(scoring)
    if not required_metrics:
        return False, []

    metric_record = cast(Dict[str, object], cached_result["metric_record"])
    metric_scope = _effective_metric_scope(metric_record, scoring)
    scope_metrics_raw = metric_record.get(metric_scope)
    scope_metrics: Dict[str, object] = (
        dict(cast(Dict[str, object], scope_metrics_raw))
        if isinstance(scope_metrics_raw, dict)
        else {}
    )
    missing_metrics = [
        metric_name
        for metric_name in required_metrics
        if not _metric_value_is_available(scope_metrics, metric_name)
    ]
    if not missing_metrics:
        return False, []

    computed_metrics = compute_extended_image_metrics(
        _workspace_metric_image_for_scope(run_dir / "workspace.npz", metric_scope)
    )
    backfilled_names: List[str] = []
    for metric_name in missing_metrics:
        value = computed_metrics.get(metric_name)
        if value is None or not math.isfinite(float(value)):
            continue
        scope_metrics[metric_name] = float(value)
        backfilled_names.append(metric_name)

    if not backfilled_names:
        return False, []

    metric_record[metric_scope] = scope_metrics
    metric_record["metric_backfilled"] = True
    metric_record["metric_backfilled_at"] = _utc_now_iso()
    metric_record["metric_backfilled_names"] = sorted(
        set(
            [str(name) for name in metric_record.get("metric_backfilled_names", [])]
            if isinstance(metric_record.get("metric_backfilled_names"), list)
            else []
        )
        | set(backfilled_names)
    )
    cached_result["metric_record"] = metric_record
    cached_result["metric_backfilled"] = True
    cached_result["metric_backfilled_names"] = list(backfilled_names)

    if not dry_run:
        write_json_atomic(
            run_dir / "metric_record.json",
            metric_record,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )
    return True, backfilled_names


def _set_parallel_thread_limits() -> None:
    for env_var in THREAD_LIMIT_ENV_VARS:
        os.environ[env_var] = "1"


def _pool_worker_initializer() -> None:
    _set_parallel_thread_limits()
    cgroup_procs = os.environ.get("MOMENT_IMAGES_WORKER_CGROUP_PROCS")
    if cgroup_procs:
        try:
            Path(cgroup_procs).write_text(f"{os.getpid()}\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - required containment must not be bypassed.
            raise RuntimeError(
                f"Worker could not join delegated memory cgroup: {cgroup_procs}"
            ) from exc
    try:
        import cv2 as cv  # Local import keeps startup cost inside workers.

        cv.setNumThreads(1)
    except Exception:  # noqa: BLE001 - thread cap is best-effort.
        return


def _get_macos_available_memory_bytes() -> Optional[int]:
    return sweep_scheduler.get_macos_available_memory_bytes()


def _get_psutil_available_memory_bytes() -> Optional[int]:
    return sweep_scheduler.get_psutil_available_memory_bytes()


def _get_current_process_rss_bytes() -> Optional[int]:
    return sweep_scheduler.get_current_process_rss_bytes()


def _get_available_memory_bytes_stdlib() -> Optional[int]:
    return sweep_scheduler.get_available_memory_bytes_stdlib()


def _get_available_memory_bytes() -> Optional[int]:
    return sweep_scheduler.get_available_memory_bytes()


def _require_adaptive_memory_telemetry(scheduling: SweepSchedulingSpec) -> None:
    if (
        scheduling.mode == SCHEDULER_MODE_ADAPTIVE_MEMORY
        and sweep_scheduler.psutil is None
    ):
        raise RuntimeError(
            "adaptive_memory scheduling requires psutil for live available-memory and RSS telemetry. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )


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
    scheduling: Optional[SweepSchedulingSpec] = None,
) -> Dict[str, object]:
    return sweep_scheduler.plan_workers(
        tasks,
        requested_workers,
        scheduling=scheduling or SweepSchedulingSpec(),
        available_bytes=available_memory_bytes,
    ).to_dict()


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
        "candidate_signature": task.get("candidate_signature"),
        "dataset_signature": task.get("dataset_signature"),
        "run_signature": task.get("run_signature"),
        "generation_id": task.get("generation_id"),
        "run_dir": task.get("run_dir"),
        "artifact_paths": task.get("artifact_paths"),
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


def _process_pool_kwargs(workers: int, worker_recycle_tasks: Optional[int] = None) -> Dict[str, object]:
    return sweep_scheduler.process_pool_kwargs(
        workers,
        worker_recycle_tasks,
        initializer=_pool_worker_initializer,
        executor_factory=ProcessPoolExecutor,
    )


def _execute_pool_batch(
    tasks: Sequence[Dict[str, object]],
    workers: int,
    worker_recycle_tasks: Optional[int] = None,
    on_result: Optional[Callable[[Dict[str, object], Dict[str, object]], None]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Optional[BaseException]]:
    if callable(worker_recycle_tasks) and on_result is None:
        on_result = cast(Callable[[Dict[str, object], Dict[str, object]], None], worker_recycle_tasks)
        worker_recycle_tasks = None

    completed_results: List[Dict[str, object]] = []
    completed_indices: set[int] = set()
    pending_tasks = list(tasks)
    executor_exc: Optional[BaseException] = None
    callback_exc: Optional[BaseException] = None

    try:
        with ProcessPoolExecutor(**_process_pool_kwargs(workers, worker_recycle_tasks)) as executor:
            future_map = {
                executor.submit(_evaluate_candidate_task, task): task for task in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - recover by degrading worker count.
                    executor_exc = exc
                    break
                result["task_index"] = task["task_index"]
                completed_results.append(result)
                completed_indices.add(int(task["task_index"]))
                if on_result is not None:
                    try:
                        on_result(task, result)
                    except Exception as exc:  # noqa: BLE001 - surface callback/persistence failures unchanged.
                        callback_exc = exc
                        raise

            if executor_exc is None:
                pending_tasks = []
            else:
                pending_tasks = [
                    task
                    for task in tasks
                    if int(task["task_index"]) not in completed_indices
                ]
    except Exception as exc:  # noqa: BLE001 - covers pool init/spawn failures.
        if callback_exc is not None and exc is callback_exc:
            raise
        executor_exc = exc
        pending_tasks = [
            task
            for task in tasks
            if int(task["task_index"]) not in completed_indices
        ]

    return completed_results, pending_tasks, executor_exc


def _execute_adaptive_pool_batch(
    tasks: Sequence[Dict[str, object]],
    workers: int,
    memory_safety_fraction: float,
    worker_recycle_tasks: Optional[int],
    memory_telemetry: bool,
    learning: Optional[sweep_scheduler.AdaptiveLearningState] = None,
    memory_limit_bytes: Optional[int] = None,
    memory_enforcer: Optional[object] = None,
    on_result: Optional[Callable[[Dict[str, object], Dict[str, object]], None]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Optional[BaseException], Dict[str, object]]:
    registration_env = (
        worker_registration_environment(cast(Any, memory_enforcer))
        if memory_enforcer is not None
        else {}
    )
    monitor_stop = threading.Event()
    monitor_samples: Dict[str, Optional[int]] = {
        "max_aggregate_worker_rss_bytes_observed": None,
        "min_available_memory_bytes_observed": None,
    }
    worker_pids: set[int] = set()
    worker_started_callback: Optional[Callable[[int], None]] = None
    if memory_enforcer is not None and not registration_env:
        def worker_started_callback(pid: int) -> None:
            cast(Any, memory_enforcer).add_worker(pid)
            worker_pids.add(pid)

    def monitor_memory() -> None:
        while not monitor_stop.wait(0.25):
            aggregate = (
                cast(Any, memory_enforcer).current_bytes()
                if memory_enforcer is not None
                else None
            )
            if aggregate is None and worker_pids and sweep_scheduler.psutil is not None:
                try:
                    aggregate = sum(
                        int(sweep_scheduler.psutil.Process(pid).memory_info().rss)
                        for pid in worker_pids
                    )
                except Exception:
                    aggregate = None
            if aggregate is not None:
                previous = monitor_samples["max_aggregate_worker_rss_bytes_observed"]
                monitor_samples["max_aggregate_worker_rss_bytes_observed"] = (
                    int(aggregate)
                    if previous is None
                    else max(int(previous), int(aggregate))
                )
            available = _get_available_memory_bytes()
            if available is not None:
                previous = monitor_samples["min_available_memory_bytes_observed"]
                monitor_samples["min_available_memory_bytes_observed"] = (
                    int(available)
                    if previous is None
                    else min(int(previous), int(available))
                )

    monitor = threading.Thread(target=monitor_memory, name="sweep-parent-memory-monitor", daemon=True)
    monitor.start()
    previous_env = {key: os.environ.get(key) for key in registration_env}
    os.environ.update(registration_env)
    try:
        outcome = sweep_scheduler.execute_pool_attempt(
            list(tasks),
            workers=workers,
            scheduling=SweepSchedulingSpec(
                mode=SCHEDULER_MODE_ADAPTIVE_MEMORY,
                memory_safety_fraction=memory_safety_fraction,
                memory_telemetry=memory_telemetry,
                worker_recycle_tasks=worker_recycle_tasks,
            ),
            worker_fn=_evaluate_candidate_task,
            initializer=_pool_worker_initializer,
            on_result=on_result,
            memory_probe=_get_available_memory_bytes,
            executor_factory=ProcessPoolExecutor,
            wait_fn=wait,
            learning=learning,
            memory_limit_bytes=memory_limit_bytes,
            worker_started_callback=worker_started_callback,
        )
        if memory_enforcer is not None:
            memory_events = cast(Any, memory_enforcer).memory_events()
            outcome.telemetry["memory_events"] = dict(memory_events)
            outcome.telemetry["memory_oom_detected"] = bool(
                int(memory_events.get("oom", 0) or 0)
                or int(memory_events.get("oom_kill", 0) or 0)
            )
    finally:
        monitor_stop.set()
        monitor.join(timeout=0.5)
        if "outcome" in locals():
            outcome.telemetry.update(monitor_samples)
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        if memory_enforcer is not None:
            cast(Any, memory_enforcer).close()
    return (
        outcome.completed_results,
        outcome.pending_tasks,
        outcome.error,
        outcome.telemetry,
    )


def _start_worker_memory_sampler(enabled: bool) -> Tuple[Optional[int], Optional[threading.Event], Optional[threading.Thread], Dict[str, Optional[int]]]:
    """Sample worker RSS while a task runs without changing its compute path."""

    rss_start = _get_current_process_rss_bytes() if enabled else None
    samples: Dict[str, Optional[int]] = {"rss_peak_bytes": rss_start}
    if not enabled or rss_start is None:
        return rss_start, None, None, samples

    stopped = threading.Event()

    def sample() -> None:
        while not stopped.wait(0.05):
            rss = _get_current_process_rss_bytes()
            if rss is not None:
                current = samples["rss_peak_bytes"]
                samples["rss_peak_bytes"] = rss if current is None else max(current, rss)

    sampler = threading.Thread(target=sample, name="sweep-rss-sampler", daemon=True)
    sampler.start()
    return rss_start, stopped, sampler, samples


def _finish_worker_memory_sampler(
    *,
    rss_start: Optional[int],
    stopped: Optional[threading.Event],
    sampler: Optional[threading.Thread],
    samples: Mapping[str, Optional[int]],
) -> Dict[str, Optional[int]]:
    if stopped is not None:
        stopped.set()
    if sampler is not None:
        sampler.join(timeout=0.2)
    rss_end = _get_current_process_rss_bytes()
    peak = samples.get("rss_peak_bytes")
    if rss_end is not None:
        peak = rss_end if peak is None else max(peak, rss_end)
    return {
        "rss_bytes_start": rss_start,
        "rss_bytes_peak": peak,
        "rss_bytes_end": rss_end,
        "rss_delta_bytes": (
            None if rss_start is None or rss_end is None else int(rss_end - rss_start)
        ),
        "rss_peak_growth_bytes": (
            None if rss_start is None or peak is None else max(0, int(peak - rss_start))
        ),
        "sample_interval_seconds": 0.05,
    }


def _evaluate_candidate_task(task: Dict[str, object]) -> Dict[str, object]:
    run_uid = str(task["run_uid"])
    stage = str(task["stage"])
    candidate_id = str(task["candidate_id"])
    dataset_id = str(task["dataset_id"])
    strategy = str(task["strategy"])
    collect_memory_telemetry = bool(task.get("memory_telemetry", True))
    rss_start, sampler_stop, sampler, rss_samples = _start_worker_memory_sampler(
        collect_memory_telemetry
    )
    started = time.perf_counter()
    try:
        views = _load_views(Path(cast(str, task["views_path"])))
        config = RectificationRunConfig(**cast(Dict[str, object], task["config"]))
        result = rectify_views_from_config(
            views,
            config,
            power_blend_cache_dir=Path(cast(str, task["power_blend_cache_dir"])),
            power_blend_cache_key=str(task["power_blend_cache_key"]),
        )
        output_dir = Path(cast(str, task["run_parent_dir"]))
        run_name = str(task["run_name"])
        metric_record = build_metric_record(run_uid, config, result)
        run_dir = output_dir / run_name
        artifact_paths = _artifact_paths_payload(run_dir)
        identity_payload = {
            "candidate_signature": task.get("candidate_signature"),
            "dataset_signature": task.get("dataset_signature"),
            "run_signature": task.get("run_signature"),
            "generation_id": task.get("generation_id"),
            "artifact_paths": artifact_paths,
        }
        metric_record.update(identity_payload)
        _save_run_artifacts(output_dir, run_name, result, metric_record=metric_record)
        if bool(task.get("save_figures", False)):
            figures = generate_rectification_figures(result)
            _persist_figures(figures, output_dir / run_name / "figures")
        elapsed = time.perf_counter() - started
        payload = {
            "status": "ok",
            "stage": stage,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "run_uid": run_uid,
            "run_dir": str(run_dir),
            "runtime_seconds": float(elapsed),
            "config": asdict(config),
            "params": task["params"],
            "metric_record": metric_record,
            "candidate_signature": task.get("candidate_signature"),
            "dataset_signature": task.get("dataset_signature"),
            "run_signature": task.get("run_signature"),
            "generation_id": task.get("generation_id"),
            "artifact_paths": artifact_paths,
            "worker_pid": os.getpid(),
        }
        if collect_memory_telemetry:
            payload["memory_telemetry"] = _finish_worker_memory_sampler(
                rss_start=rss_start,
                stopped=sampler_stop,
                sampler=sampler,
                samples=rss_samples,
            )
        return payload
    except Exception as exc:  # noqa: BLE001 - preserve failed run details
        elapsed = time.perf_counter() - started
        payload = {
            "status": "failed",
            "stage": stage,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "run_uid": run_uid,
            "runtime_seconds": float(elapsed),
            "config": task["config"],
            "params": task["params"],
            "candidate_signature": task.get("candidate_signature"),
            "dataset_signature": task.get("dataset_signature"),
            "run_signature": task.get("run_signature"),
            "generation_id": task.get("generation_id"),
            "run_dir": task.get("run_dir"),
            "artifact_paths": task.get("artifact_paths"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
        }
        if collect_memory_telemetry:
            payload["memory_telemetry"] = _finish_worker_memory_sampler(
                rss_start=rss_start,
                stopped=sampler_stop,
                sampler=sampler,
                samples=rss_samples,
            )
        return payload


def _default_max_workers(explicit: Optional[int]) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _update_stage_progress_state(
    *,
    run_state: Optional[Dict[str, object]],
    stage_name: str,
    planned_runs: int,
    completed_runs: int,
    skipped_valid_count: int,
    recomputed_count: int,
    mark_completed: bool = False,
) -> None:
    if run_state is None:
        return
    stages_state = cast(Dict[str, Dict[str, object]], run_state.setdefault("stages", {}))
    stage_slot = cast(Dict[str, object], stages_state.setdefault(stage_name, {}))
    stage_slot["planned_runs"] = int(planned_runs)
    stage_slot["completed_runs"] = int(completed_runs)
    stage_slot["skipped_valid_count"] = int(skipped_valid_count)
    stage_slot["recomputed_count"] = int(recomputed_count)
    if mark_completed:
        stage_slot["last_completed_at"] = _utc_now_iso()


def _record_stage_task_result(
    *,
    task: Dict[str, object],
    result: Dict[str, object],
    state_runs: Dict[str, Dict[str, object]],
    run_state: Optional[Dict[str, object]],
    run_state_path: Optional[Path],
    stage_name: str,
    planned_runs: int,
    completed_runs: int,
    skipped_valid_count: int,
    recomputed_count: int,
) -> int:
    if run_state is None:
        return recomputed_count

    entry = state_runs.get(str(task["run_uid"]))
    if entry is not None:
        artifact_validation = _validate_run_artifacts(Path(cast(str, task["run_dir"])))
        if result.get("status") == "ok" and not bool(artifact_validation.get("valid", False)):
            result["status"] = "failed"
            result["failure_kind"] = "artifact_invalid"
            result["error"] = "Worker reported success but required artifacts failed validation."
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
        entry["artifact_validation"] = artifact_validation

    _update_stage_progress_state(
        run_state=run_state,
        stage_name=stage_name,
        planned_runs=planned_runs,
        completed_runs=completed_runs,
        skipped_valid_count=skipped_valid_count,
        recomputed_count=recomputed_count,
        mark_completed=True,
    )
    if run_state_path is not None:
        _persist_run_state(run_state_path, run_state)
    return recomputed_count


def _read_source_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists() and not Path(f"{path}.bak").exists():
        return None
    try:
        payload, _, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))
    except Exception:  # noqa: BLE001 - invalid source metadata is handled as incompatible.
        return None
    return cast(Dict[str, object], payload)


def _source_experiment_root(source_root: Path) -> Path:
    source_root = Path(source_root)
    return source_root.parent if source_root.name == "metric_experiments" else source_root


def _source_base_for_stage(source_root: Path, stage_name: str) -> Path:
    root = _source_experiment_root(source_root)
    if stage_name in {METRIC_PHASE_STAGE_NAME, METRIC_SEARCH_STAGE1_NAME}:
        return root / "metric_experiments"
    return root


def _source_dataset_signatures(source_root: Path) -> Dict[str, str]:
    root = _source_experiment_root(source_root)
    state_payload = _read_source_json(root / RUN_STATE_FILENAME)
    if state_payload is not None:
        signatures = _dataset_signatures_from_fingerprint_payload(
            state_payload.get("fingerprint_payload")
        )
        if signatures:
            return signatures
    manifest_payload = _read_source_json(root / "experiment_manifest.json")
    if manifest_payload is not None:
        datasets = manifest_payload.get("datasets")
        if isinstance(datasets, list):
            return {
                str(entry.get("dataset_id")): _dataset_signature_from_identity_payload(
                    cast(Dict[str, object], entry)
                )
                for entry in datasets
                if isinstance(entry, dict) and entry.get("dataset_id") is not None
            }
    return {}


def _copy_run_artifacts_dir(source_run_dir: Path, destination_run_dir: Path) -> None:
    destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_run_dir)


def _merge_artifacts_for_tasks(
    *,
    tasks: Sequence[Dict[str, object]],
    stage_name: str,
    output_dir: Path,
    merge_sources: Sequence[Path],
    merge_log: Optional[Dict[str, object]],
    dry_run: bool,
) -> None:
    if not merge_sources:
        return

    task_by_identity = {
        (
            str(task.get("candidate_signature", "")),
            str(task.get("dataset_signature", "")),
        ): task
        for task in tasks
    }
    copied = cast(List[Dict[str, object]], (merge_log or {}).setdefault("copied", []))
    would_copy = cast(List[Dict[str, object]], (merge_log or {}).setdefault("would_copy", []))
    skipped = cast(List[Dict[str, object]], (merge_log or {}).setdefault("skipped_duplicates", []))
    rejected = cast(List[Dict[str, object]], (merge_log or {}).setdefault("rejected", []))
    conflicts = cast(List[Dict[str, object]], (merge_log or {}).setdefault("conflicts", []))
    source_summaries = cast(List[Dict[str, object]], (merge_log or {}).setdefault("sources", []))

    for raw_source in merge_sources:
        source_root = Path(raw_source)
        source_base = _source_base_for_stage(source_root, stage_name)
        source_summary = {
            "source_root": str(source_root.resolve()),
            "stage": stage_name,
            "source_base": str(source_base.resolve()),
        }
        source_summaries.append(source_summary)
        source_dataset_signatures = _source_dataset_signatures(source_root)
        if not source_dataset_signatures:
            rejected.append(
                {
                    "source_root": str(source_root),
                    "stage": stage_name,
                    "reason": "missing_dataset_identity_metadata",
                }
            )
            continue

        for source_run_dir in sorted((source_base / "runs" / stage_name).glob("*/*")):
            if not source_run_dir.is_dir():
                continue
            source_validation = _validate_run_artifacts(source_run_dir)
            if not bool(source_validation.get("valid", False)):
                rejected.append(
                    {
                        "source_run_dir": str(source_run_dir),
                        "stage": stage_name,
                        "reason": "invalid_source_artifacts",
                        "artifact_validation": source_validation,
                    }
                )
                continue
            dataset_id = source_run_dir.name
            dataset_signature = source_dataset_signatures.get(dataset_id)
            if not dataset_signature:
                rejected.append(
                    {
                        "source_run_dir": str(source_run_dir),
                        "stage": stage_name,
                        "dataset_id": dataset_id,
                        "reason": "unknown_source_dataset",
                    }
                )
                continue
            config_payload = _read_source_json(source_run_dir / "config.json")
            if config_payload is None:
                rejected.append(
                    {
                        "source_run_dir": str(source_run_dir),
                        "stage": stage_name,
                        "dataset_id": dataset_id,
                        "reason": "unreadable_source_config",
                    }
                )
                continue
            candidate_signature = _candidate_signature_from_config_payload(config_payload)
            task = task_by_identity.get((candidate_signature, dataset_signature))
            if task is None:
                rejected.append(
                    {
                        "source_run_dir": str(source_run_dir),
                        "stage": stage_name,
                        "dataset_id": dataset_id,
                        "candidate_signature": candidate_signature,
                        "dataset_signature": dataset_signature,
                        "reason": "not_in_active_plan_or_dataset_mismatch",
                    }
                )
                continue

            destination_run_dir = Path(cast(str, task["run_dir"]))
            destination_validation = _validate_run_artifacts(destination_run_dir)
            log_payload = {
                "source_run_dir": str(source_run_dir),
                "destination_run_dir": str(destination_run_dir),
                "stage": stage_name,
                "candidate_id": task.get("candidate_id"),
                "dataset_id": task.get("dataset_id"),
                "candidate_signature": candidate_signature,
                "dataset_signature": dataset_signature,
                "run_signature": task.get("run_signature"),
            }
            if bool(destination_validation.get("valid", False)):
                skipped.append({**log_payload, "reason": "destination_already_valid"})
                continue
            if destination_run_dir.exists():
                conflicts.append(
                    {
                        **log_payload,
                        "reason": "destination_exists_but_is_not_valid",
                        "artifact_validation": destination_validation,
                    }
                )
                continue
            if dry_run:
                would_copy.append(log_payload)
                continue
            _copy_run_artifacts_dir(source_run_dir, destination_run_dir)
            copied.append(log_payload)


def _adopt_existing_signature_artifacts(
    *,
    tasks: Sequence[Dict[str, object]],
    state_runs: Dict[str, Dict[str, object]],
) -> None:
    """Copy valid artifacts from older run IDs into the current deterministic path."""

    entries_by_signature: Dict[str, Dict[str, object]] = {}
    for entry in state_runs.values():
        if not isinstance(entry, dict):
            continue
        run_signature = str(entry.get("run_signature", ""))
        if run_signature and run_signature not in entries_by_signature:
            entries_by_signature[run_signature] = entry

    for task in tasks:
        run_uid = str(task["run_uid"])
        if run_uid in state_runs:
            continue
        run_signature = str(task.get("run_signature", ""))
        source_entry = entries_by_signature.get(run_signature)
        if source_entry is None:
            continue
        source_run_dir_raw = source_entry.get("run_dir")
        if not source_run_dir_raw:
            continue
        source_run_dir = Path(str(source_run_dir_raw))
        destination_run_dir = Path(cast(str, task["run_dir"]))
        if source_run_dir == destination_run_dir:
            continue
        if not bool(_validate_run_artifacts(source_run_dir).get("valid", False)):
            continue
        if destination_run_dir.exists():
            continue
        _copy_run_artifacts_dir(source_run_dir, destination_run_dir)


def _adopt_existing_candidate_dataset_artifacts(
    *,
    tasks: Sequence[Dict[str, object]],
    state_runs: Dict[str, Dict[str, object]],
) -> None:
    """Copy valid artifacts with the same config and dataset across stages."""

    entries_by_identity: Dict[Tuple[str, str], Dict[str, object]] = {}
    for entry in state_runs.values():
        if not isinstance(entry, dict):
            continue
        candidate_signature = str(entry.get("candidate_signature", ""))
        dataset_signature = str(entry.get("dataset_signature", ""))
        if not candidate_signature or not dataset_signature:
            continue
        key = (candidate_signature, dataset_signature)
        if key not in entries_by_identity:
            entries_by_identity[key] = cast(Dict[str, object], entry)

    for task in tasks:
        destination_run_dir = Path(cast(str, task["run_dir"]))
        if destination_run_dir.exists():
            continue
        key = (
            str(task.get("candidate_signature", "")),
            str(task.get("dataset_signature", "")),
        )
        source_entry = entries_by_identity.get(key)
        if source_entry is None:
            continue
        source_run_dir_raw = source_entry.get("run_dir")
        if not source_run_dir_raw:
            continue
        source_run_dir = Path(str(source_run_dir_raw))
        if source_run_dir == destination_run_dir:
            continue
        if not bool(_validate_run_artifacts(source_run_dir).get("valid", False)):
            continue
        _copy_run_artifacts_dir(source_run_dir, destination_run_dir)


def _evaluate_stage(
    stage_name: str,
    candidates: Sequence[SweepCandidate],
    datasets: Sequence[PreparedDataset],
    output_dir: Path,
    save_figures_flag: bool,
    requested_workers: int,
    scoring: Optional[SweepScoringSpec] = None,
    run_state: Optional[Dict[str, object]] = None,
    run_state_path: Optional[Path] = None,
    generation_id: Optional[str] = None,
    merge_sources: Sequence[Path] = (),
    merge_log: Optional[Dict[str, object]] = None,
    dry_run: bool = False,
    scheduling: Optional[SweepSchedulingSpec] = None,
) -> StageEvaluationResult:
    scheduling = scheduling or SweepSchedulingSpec()
    tasks: List[Dict[str, object]] = []
    for candidate in candidates:
        candidate_root = output_dir / "runs" / stage_name / candidate.candidate_id
        candidate_signature = _candidate_signature(candidate)
        for dataset in datasets:
            config = _compose_run_config(candidate, dataset)
            run_name = dataset.dataset_id
            run_uid = f"{stage_name}_{candidate.candidate_id}_{dataset.dataset_id}"
            dataset_signature = _dataset_signature(dataset)
            run_signature = _run_signature(stage_name, candidate_signature, dataset_signature)
            run_dir = candidate_root / run_name
            cache_payload = asdict(config)
            cache_payload.pop("pixel_power_blend_alpha", None)
            cache_payload.pop("pixel_power_blend_delta_cap", None)
            tasks.append(
                {
                    "stage": stage_name,
                    "candidate_id": candidate.candidate_id,
                    "dataset_id": dataset.dataset_id,
                    "strategy": candidate.strategy,
                    "run_uid": run_uid,
                    "run_name": run_name,
                    "run_parent_dir": str(candidate_root),
                    "run_dir": str(run_dir),
                    "views_path": str(dataset.views_path),
                    "config": asdict(config),
                    "params": dict(candidate.params),
                    "save_figures": save_figures_flag,
                    "views_shape": list(dataset.views_shape),
                    "views_dtype": dataset.views_dtype,
                    "views_itemsize": int(dataset.views_itemsize),
                    "memory_telemetry": bool(scheduling.memory_telemetry),
                    "candidate_signature": candidate_signature,
                    "dataset_signature": dataset_signature,
                    "run_signature": run_signature,
                    "generation_id": generation_id,
                    "artifact_paths": _artifact_paths_payload(run_dir),
                    "power_blend_cache_dir": str(output_dir / ".pixel_power_blend_cache"),
                    "power_blend_cache_key": _hash_payload(
                        {
                            "version": 2,
                            "dataset_signature": dataset_signature,
                            "field_config": cache_payload,
                        }
                    ),
                }
            )
    tasks.sort(key=lambda item: (str(item["candidate_id"]), str(item["dataset_id"])))
    for idx, task in enumerate(tasks):
        task["task_index"] = idx
        task["estimated_peak_bytes"] = int(_estimate_task_peak_bytes(task))

    if not tasks:
        return StageEvaluationResult(
            stage_results=[],
            execution={
                "requested_workers": int(max(1, requested_workers)),
                "selected_workers": 1,
                "available_memory_bytes": _get_available_memory_bytes(),
                "max_task_peak_bytes": 0,
                "memory_cap_workers": None,
                "safety_fraction": float(scheduling.memory_safety_fraction),
                "memory_safety_fraction": float(scheduling.memory_safety_fraction),
                "scheduler_mode": scheduling.mode,
                "memory_budget_bytes": None,
                "min_memory_budget_bytes_observed": None,
                "max_memory_budget_bytes_observed": None,
                "max_concurrent_tasks_observed": 0,
                "max_reserved_peak_bytes_observed": 0,
                "min_available_memory_bytes_observed": None,
                "memory_telemetry_enabled": bool(scheduling.memory_telemetry),
                "worker_recycle_tasks": scheduling.worker_recycle_tasks,
                "degrade_events": [],
                "pool_crash_count": 0,
                "isolated_crash_count": 0,
                "skipped_valid_count": 0,
                "recomputed_count": 0,
                "metric_backfilled_count": 0,
                "metric_backfill_failed_count": 0,
                "dry_run": bool(dry_run),
            },
        )

    stage_results: List[Dict[str, object]] = []
    tasks_to_execute: List[Dict[str, object]] = []
    skipped_valid_count = 0
    recomputed_count = 0
    metric_backfilled_count = 0
    metric_backfill_failed_count = 0
    state_runs = (
        cast(Dict[str, Dict[str, object]], run_state.setdefault("runs", {}))
        if run_state is not None
        else {}
    )
    if state_runs and not dry_run:
        _adopt_existing_signature_artifacts(tasks=tasks, state_runs=state_runs)
        _adopt_existing_candidate_dataset_artifacts(tasks=tasks, state_runs=state_runs)
    _merge_artifacts_for_tasks(
        tasks=tasks,
        stage_name=stage_name,
        output_dir=output_dir,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=dry_run,
    )
    if run_state is not None:
        stages_state = cast(Dict[str, Dict[str, object]], run_state.setdefault("stages", {}))
        stage_slot = cast(Dict[str, object], stages_state.setdefault(stage_name, {}))
        stage_slot["planned_runs"] = len(tasks)
        stage_slot["last_started_at"] = _utc_now_iso()
        stage_slot["generation_id"] = generation_id
        if run_state_path is not None and not dry_run:
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
                "candidate_signature": task.get("candidate_signature"),
                "dataset_signature": task.get("dataset_signature"),
                "run_signature": task.get("run_signature"),
                "generation_id": generation_id,
                "active": True,
                "status": "pending",
                "run_dir": str(run_dir),
                "artifact_paths": task.get("artifact_paths"),
                "attempts": [],
                "artifact_validation": {},
            }
            state_runs[run_uid] = entry
        elif entry is not None:
            entry["candidate_signature"] = task.get("candidate_signature")
            entry["dataset_signature"] = task.get("dataset_signature")
            entry["run_signature"] = task.get("run_signature")
            entry["generation_id"] = generation_id
            entry["active"] = True
            entry["run_dir"] = str(run_dir)
            entry["artifact_paths"] = task.get("artifact_paths")
        previous_status = str(entry.get("status", "pending")) if entry is not None else "pending"
        if entry is not None and previous_status == "running":
            entry["status"] = "pending"
            previous_status = "pending"
        artifact_validation = _validate_run_artifacts(run_dir)
        if entry is not None:
            entry["artifact_validation"] = artifact_validation

        if bool(artifact_validation.get("valid", False)):
            try:
                cached_result = _load_cached_stage_result(
                    task,
                    run_dir,
                    persist_identity=not dry_run,
                )
            except Exception:
                task["_recompute_after_invalid"] = previous_status in {"ok", "recomputed", "skipped_valid"}
                tasks_to_execute.append(task)
                continue
            try:
                backfilled, backfilled_names = _backfill_cached_stage_result_metrics(
                    cached_result,
                    run_dir,
                    scoring,
                    dry_run=dry_run,
                )
            except Exception:
                metric_backfill_failed_count += 1
                task["_recompute_after_invalid"] = previous_status in {"ok", "recomputed", "skipped_valid"}
                tasks_to_execute.append(task)
                continue
            cached_result["task_index"] = task["task_index"]
            stage_results.append(cached_result)
            skipped_valid_count += 1
            if backfilled:
                metric_backfilled_count += 1
            if entry is not None:
                attempts = cast(List[Dict[str, object]], entry.setdefault("attempts", []))
                attempt = {
                    "timestamp": _utc_now_iso(),
                    "status": "skipped_valid",
                    "reason": "artifacts_valid",
                }
                if backfilled:
                    attempt["metric_backfilled_names"] = list(backfilled_names)
                attempts.append(attempt)
                entry["status"] = "skipped_valid"
            continue

        task["_recompute_after_invalid"] = previous_status in {"ok", "recomputed", "skipped_valid"}
        tasks_to_execute.append(task)

    _update_stage_progress_state(
        run_state=run_state,
        stage_name=stage_name,
        planned_runs=len(tasks),
        completed_runs=len(stage_results),
        skipped_valid_count=skipped_valid_count,
        recomputed_count=recomputed_count,
    )
    if run_state is not None and run_state_path is not None:
        if dry_run:
            pass
        else:
            _persist_run_state(run_state_path, run_state)

    if dry_run:
        pending_count = len(tasks_to_execute)
        dry_run_available_memory = _get_available_memory_bytes()
        dry_run_worker_plan = _select_stage_worker_count(
            tasks_to_execute or tasks,
            requested_workers,
            dry_run_available_memory,
            scheduling=scheduling,
        )
        execution = {
            **dry_run_worker_plan,
            "requested_workers": int(max(1, requested_workers)),
            "selected_workers": 0,
            "available_memory_bytes": dry_run_available_memory,
            "max_task_peak_bytes": max(int(task["estimated_peak_bytes"]) for task in tasks),
            "memory_telemetry_enabled": bool(scheduling.memory_telemetry),
            "worker_recycle_tasks": scheduling.worker_recycle_tasks,
            "min_memory_budget_bytes_observed": dry_run_worker_plan.get("memory_budget_bytes"),
            "max_memory_budget_bytes_observed": dry_run_worker_plan.get("memory_budget_bytes"),
            "max_concurrent_tasks_observed": 0,
            "max_reserved_peak_bytes_observed": 0,
            "min_available_memory_bytes_observed": None,
            "degrade_events": [],
            "pool_crash_count": 0,
            "isolated_crash_count": 0,
            "completed_runs": len(stage_results),
            "skipped_valid_count": int(skipped_valid_count),
            "recomputed_count": 0,
            "metric_backfilled_count": int(metric_backfilled_count),
            "metric_backfill_failed_count": int(metric_backfill_failed_count),
            "pending_run_count": int(pending_count),
            "dry_run": True,
            "planned_run_uids": [str(task["run_uid"]) for task in tasks],
            "pending_run_uids": [str(task["run_uid"]) for task in tasks_to_execute],
        }
        return StageEvaluationResult(stage_results=stage_results, execution=execution)

    if tasks_to_execute:
        _require_adaptive_memory_telemetry(scheduling)

    if run_state is not None and run_state_path is not None:
        _persist_run_state(run_state_path, run_state)

    progress_desc = f"Evaluating {stage_name}"
    _set_parallel_thread_limits()
    available_memory_bytes = _get_available_memory_bytes()
    worker_plan = _select_stage_worker_count(
        tasks_to_execute,
        requested_workers,
        available_memory_bytes,
        scheduling=scheduling,
    )
    selected_workers = int(worker_plan["selected_workers"])
    adaptive_learning = _adaptive_learning_state_for_host(run_state)
    memory_limit_bytes = cast(Optional[int], worker_plan.get("memory_budget_bytes"))
    memory_enforcement_status: Dict[str, object] = {
        "memory_enforcement": "advisory",
        "memory_enforcement_backend": "none",
        "memory_enforcement_reason": "fixed_batch does not use adaptive memory containment",
        "memory_limit_bytes": None,
    }
    if scheduling.mode == SCHEDULER_MODE_ADAPTIVE_MEMORY:
        if memory_limit_bytes is None:
            if scheduling.memory_enforcement == "required":
                raise RuntimeError(
                    "Required memory enforcement needs a known available-memory snapshot. "
                    "Install psutil and verify host memory telemetry."
                )
            memory_enforcement_status = {
                "memory_enforcement": "advisory",
                "memory_enforcement_backend": "none",
                "memory_enforcement_reason": "available memory is unknown",
                "memory_limit_bytes": None,
            }

    degrade_events: List[Dict[str, object]] = []
    pool_crash_count = 0
    isolated_crash_count = 0
    completed_task_indices: set[int] = set()
    scheduler_telemetry: Dict[str, object] = {
        "scheduler_mode": scheduling.mode,
        "memory_budget_bytes": worker_plan.get("memory_budget_bytes"),
        "min_memory_budget_bytes_observed": worker_plan.get("memory_budget_bytes"),
        "max_memory_budget_bytes_observed": worker_plan.get("memory_budget_bytes"),
        "max_concurrent_tasks_observed": 0,
        "max_reserved_peak_bytes_observed": 0,
        "min_available_memory_bytes_observed": None,
        "max_worker_rss_bytes_observed": None,
        "max_aggregate_worker_rss_bytes_observed": None,
        "memory_telemetry_enabled": bool(scheduling.memory_telemetry),
        "memory_pressure_wait_count": 0,
        "memory_pressure_wait_seconds": 0.0,
        "memory_events": {},
        "worker_recycle_tasks": scheduling.worker_recycle_tasks,
        **memory_enforcement_status,
        "learned_profile_count": 0,
        "cold_profile_count": len(tasks_to_execute),
    }

    def _merge_scheduler_telemetry(batch_telemetry: Mapping[str, object]) -> None:
        if (
            scheduler_telemetry.get("memory_budget_bytes") is None
            and batch_telemetry.get("memory_budget_bytes") is not None
        ):
            scheduler_telemetry["memory_budget_bytes"] = batch_telemetry.get("memory_budget_bytes")
        min_budget = batch_telemetry.get("min_memory_budget_bytes_observed")
        if min_budget is not None:
            current_budget = scheduler_telemetry["min_memory_budget_bytes_observed"]
            scheduler_telemetry["min_memory_budget_bytes_observed"] = (
                int(min_budget)
                if current_budget is None
                else min(int(current_budget), int(min_budget))
            )
        max_budget = batch_telemetry.get("max_memory_budget_bytes_observed")
        if max_budget is not None:
            current_budget = scheduler_telemetry["max_memory_budget_bytes_observed"]
            scheduler_telemetry["max_memory_budget_bytes_observed"] = (
                int(max_budget)
                if current_budget is None
                else max(int(current_budget), int(max_budget))
            )
        scheduler_telemetry["max_concurrent_tasks_observed"] = max(
            int(scheduler_telemetry["max_concurrent_tasks_observed"]),
            int(batch_telemetry.get("max_concurrent_tasks_observed", 0) or 0),
        )
        scheduler_telemetry["max_reserved_peak_bytes_observed"] = max(
            int(scheduler_telemetry["max_reserved_peak_bytes_observed"]),
            int(batch_telemetry.get("max_reserved_peak_bytes_observed", 0) or 0),
        )
        observed = batch_telemetry.get("min_available_memory_bytes_observed")
        if observed is not None:
            current = scheduler_telemetry["min_available_memory_bytes_observed"]
            scheduler_telemetry["min_available_memory_bytes_observed"] = (
                int(observed)
                if current is None
                else min(int(current), int(observed))
            )
        aggregate_rss = batch_telemetry.get("max_aggregate_worker_rss_bytes_observed")
        if aggregate_rss is not None:
            current = scheduler_telemetry["max_aggregate_worker_rss_bytes_observed"]
            scheduler_telemetry["max_aggregate_worker_rss_bytes_observed"] = (
                int(aggregate_rss)
                if current is None
                else max(int(current), int(aggregate_rss))
            )
        scheduler_telemetry["memory_pressure_wait_count"] = (
            int(scheduler_telemetry["memory_pressure_wait_count"])
            + int(batch_telemetry.get("memory_pressure_wait_count", 0) or 0)
        )
        scheduler_telemetry["memory_pressure_wait_seconds"] = (
            float(scheduler_telemetry["memory_pressure_wait_seconds"])
            + float(batch_telemetry.get("memory_pressure_wait_seconds", 0.0) or 0.0)
        )
        batch_memory_events = batch_telemetry.get("memory_events")
        if isinstance(batch_memory_events, Mapping):
            memory_events = cast(Dict[str, int], scheduler_telemetry["memory_events"])
            for event, count in batch_memory_events.items():
                try:
                    memory_events[str(event)] = int(memory_events.get(str(event), 0)) + int(count)
                except (TypeError, ValueError):
                    continue

    remaining_tasks = list(tasks_to_execute)
    current_workers = selected_workers
    execution_started = time.perf_counter()
    valid_new_completion_times: List[float] = []

    def _record_learning_if_valid(task: Dict[str, object], result: Dict[str, object]) -> None:
        if result.get("status") != "ok":
            return
        if not bool(_validate_run_artifacts(Path(cast(str, task["run_dir"]))).get("valid", False)):
            return
        memory = result.get("memory_telemetry")
        if isinstance(memory, Mapping):
            worker_peak = memory.get("rss_bytes_peak")
            if worker_peak is not None:
                current_peak = scheduler_telemetry["max_worker_rss_bytes_observed"]
                scheduler_telemetry["max_worker_rss_bytes_observed"] = (
                    int(worker_peak)
                    if current_peak is None
                    else max(int(current_peak), int(worker_peak))
                )
            peak_growth = memory.get("rss_peak_growth_bytes")
            if peak_growth is not None:
                adaptive_learning.record_success(
                    task,
                    peak_growth_bytes=int(peak_growth),
                    runtime_seconds=float(result.get("runtime_seconds", 0.0)),
                )
                _persist_adaptive_learning_state(run_state, adaptive_learning)
                if run_state is not None and run_state_path is not None:
                    _persist_run_state(run_state_path, run_state)
        valid_new_completion_times.append(time.perf_counter())
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

            if current_workers <= 1 and scheduling.mode == SCHEDULER_MODE_FIXED_BATCH:
                _merge_scheduler_telemetry(
                    {
                        "max_concurrent_tasks_observed": 1 if remaining_tasks else 0,
                        "max_reserved_peak_bytes_observed": max(
                            (int(task.get("estimated_peak_bytes", 0)) for task in remaining_tasks),
                            default=0,
                        ),
                    }
                )
                for task in remaining_tasks:
                    result, crashed = _run_task_isolated(task)
                    if crashed:
                        isolated_crash_count += 1
                    stage_results.append(result)
                    completed_task_indices.add(int(task["task_index"]))
                    progress_bar.update(1)
                    recomputed_count = _record_stage_task_result(
                        task=task,
                        result=result,
                        state_runs=state_runs,
                        run_state=run_state,
                        run_state_path=run_state_path,
                        stage_name=stage_name,
                        planned_runs=len(tasks),
                        completed_runs=len(stage_results),
                        skipped_valid_count=skipped_valid_count,
                        recomputed_count=recomputed_count,
                    )
                    _record_learning_if_valid(task, result)
                break

            def _commit_pooled_result(task: Dict[str, object], result: Dict[str, object]) -> None:
                nonlocal recomputed_count
                task_index = int(result["task_index"])
                if task_index in completed_task_indices:
                    return
                stage_results.append(result)
                completed_task_indices.add(task_index)
                progress_bar.update(1)
                recomputed_count = _record_stage_task_result(
                    task=task,
                    result=result,
                    state_runs=state_runs,
                    run_state=run_state,
                    run_state_path=run_state_path,
                    stage_name=stage_name,
                    planned_runs=len(tasks),
                    completed_runs=len(stage_results),
                    skipped_valid_count=skipped_valid_count,
                    recomputed_count=recomputed_count,
                )
                _record_learning_if_valid(task, result)

            isolated_adaptive_attempt = (
                scheduling.mode == SCHEDULER_MODE_ADAPTIVE_MEMORY and current_workers <= 1
            )
            attempt_tasks = remaining_tasks[:1] if isolated_adaptive_attempt else remaining_tasks
            batch_telemetry: Mapping[str, object] = {}
            if scheduling.mode == SCHEDULER_MODE_ADAPTIVE_MEMORY:
                attempt_enforcer = (
                    create_memory_enforcer(
                        scheduling.memory_enforcement,
                        int(memory_limit_bytes),
                    )
                    if memory_limit_bytes is not None
                    else None
                )
                if attempt_enforcer is not None:
                    if (
                        attempt_enforcer.status.backend == "windows-job-object"
                        and scheduling.worker_recycle_tasks is not None
                    ):
                        attempt_enforcer.close()
                        if scheduling.memory_enforcement == "required":
                            raise RuntimeError(
                                "required Windows Job Object enforcement cannot be combined with "
                                "worker_recycle_tasks because replacement worker attachment is not "
                                "guaranteed before dispatch."
                            )
                        attempt_enforcer = None
                        memory_enforcement_status.update(
                            {
                                "memory_enforcement": "advisory",
                                "memory_enforcement_backend": "windows-job-object",
                                "memory_enforcement_reason": (
                                    "worker recycling disables guaranteed Job Object attachment; "
                                    "using advisory admission"
                                ),
                                "memory_limit_bytes": None,
                            }
                        )
                    elif attempt_enforcer is not None:
                        memory_enforcement_status.update(
                            {
                                "memory_enforcement": attempt_enforcer.status.mode,
                                "memory_enforcement_backend": attempt_enforcer.status.backend,
                                "memory_enforcement_reason": attempt_enforcer.status.reason,
                                "memory_limit_bytes": attempt_enforcer.status.limit_bytes,
                            }
                        )
                    scheduler_telemetry.update(memory_enforcement_status)
                completed_batch, pending_batch, exec_error, batch_telemetry = _execute_adaptive_pool_batch(
                    attempt_tasks,
                    workers=current_workers,
                    memory_safety_fraction=scheduling.memory_safety_fraction,
                    worker_recycle_tasks=(
                        None if isolated_adaptive_attempt else scheduling.worker_recycle_tasks
                    ),
                    memory_telemetry=scheduling.memory_telemetry,
                    learning=adaptive_learning,
                    memory_limit_bytes=memory_limit_bytes,
                    memory_enforcer=attempt_enforcer,
                    on_result=_commit_pooled_result,
                )
                _merge_scheduler_telemetry(batch_telemetry)
            else:
                fixed_reserved = sum(
                    int(task.get("estimated_peak_bytes", 0))
                    for task in remaining_tasks[:current_workers]
                )
                _merge_scheduler_telemetry(
                    {
                        "max_concurrent_tasks_observed": min(current_workers, len(remaining_tasks)),
                        "max_reserved_peak_bytes_observed": fixed_reserved,
                    }
                )
                completed_batch, pending_batch, exec_error = _execute_pool_batch(
                    remaining_tasks,
                    workers=current_workers,
                    worker_recycle_tasks=scheduling.worker_recycle_tasks,
                    on_result=_commit_pooled_result,
                )
            if completed_batch:
                by_uid = {str(task["run_uid"]): task for task in remaining_tasks}
                for result in completed_batch:
                    task = by_uid.get(str(result["run_uid"]))
                    if task is None:
                        continue
                    _commit_pooled_result(task, result)

            if exec_error is None:
                if isolated_adaptive_attempt and len(remaining_tasks) > 1:
                    remaining_tasks = [
                        task
                        for task in remaining_tasks
                        if int(task["task_index"]) not in completed_task_indices
                    ]
                    continue
                break

            if isolated_adaptive_attempt:
                task = attempt_tasks[0]
                if bool(batch_telemetry.get("memory_oom_detected", False)):
                    result = build_memory_limit_result(
                        task,
                        "Worker memory containment reported an OOM memory-limit event.",
                    )
                else:
                    isolated_crash_count += 1
                    result = _build_worker_crash_result(task, 0.0, exec_error)
                result["task_index"] = task["task_index"]
                _commit_pooled_result(task, result)
                remaining_tasks = [
                    pending_task
                    for pending_task in remaining_tasks
                    if int(pending_task["task_index"]) != int(task["task_index"])
                ]
                continue

            pool_crash_count += 1
            next_workers = max(1, current_workers // 2)
            degrade_events.append(
                {
                    "event": (
                        "memory_enforcement_oom"
                        if bool(batch_telemetry.get("memory_oom_detected", False))
                        else "pool_failure"
                    ),
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
    execution.update(scheduler_telemetry)
    execution["degrade_events"] = degrade_events
    execution["pool_crash_count"] = int(pool_crash_count)
    execution["isolated_crash_count"] = int(isolated_crash_count)
    execution["completed_runs"] = len(stage_results)
    execution["skipped_valid_count"] = int(skipped_valid_count)
    execution["recomputed_count"] = int(recomputed_count)
    execution["metric_backfilled_count"] = int(metric_backfilled_count)
    execution["metric_backfill_failed_count"] = int(metric_backfill_failed_count)
    execution["used_isolated_mode"] = bool(isolated_crash_count > 0 or current_workers == 1)
    elapsed_minutes = max((time.perf_counter() - execution_started) / 60.0, 1e-9)
    execution["completed_valid_new_runs"] = len(valid_new_completion_times)
    execution["completed_valid_new_runs_per_minute"] = len(valid_new_completion_times) / elapsed_minutes
    rolling_cutoff = time.perf_counter() - 60.0
    execution["rolling_completed_valid_new_runs_per_minute"] = float(
        sum(completed_at >= rolling_cutoff for completed_at in valid_new_completion_times)
    )
    execution["learned_profile_count"] = sum(
        1 for model in adaptive_learning.models.values() if model.sample_count > 0
    )
    execution["cold_profile_count"] = sum(
        1 for task in tasks_to_execute if adaptive_learning.model_for(task) is None
    )
    if run_state is not None:
        active_run_uids = {str(task["run_uid"]) for task in tasks}
        for run_uid, entry in state_runs.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("stage")) == stage_name:
                entry["active"] = run_uid in active_run_uids
    _update_stage_progress_state(
        run_state=run_state,
        stage_name=stage_name,
        planned_runs=len(tasks),
        completed_runs=len(stage_results),
        skipped_valid_count=skipped_valid_count,
        recomputed_count=recomputed_count,
        mark_completed=True,
    )
    if run_state is not None and run_state_path is not None:
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
    weight_sum = float(sum(abs(weight) for weight in metric_weights.values()))
    if weight_sum <= 0:
        raise ValueError("Metric weights must have a positive absolute sum.")

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
                "candidate_signature": record.get("candidate_signature"),
                "generation_id": record.get("generation_id"),
                "dataset_scores": {},
                "dataset_metrics": {},
                "runtime_seconds": {},
                "run_signatures": {},
                "run_dirs": {},
                "artifact_paths": {},
            },
        )
        cast(Dict[str, float], slot["dataset_scores"])[dataset_id] = float(dataset_score)
        cast(Dict[str, Dict[str, float]], slot["dataset_metrics"])[dataset_id] = metrics
        cast(Dict[str, float], slot["runtime_seconds"])[dataset_id] = runtime
        cast(Dict[str, object], slot["run_signatures"])[dataset_id] = record.get("run_signature")
        cast(Dict[str, object], slot["run_dirs"])[dataset_id] = record.get("run_dir")
        cast(Dict[str, object], slot["artifact_paths"])[dataset_id] = record.get("artifact_paths")

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
                "candidate_signature": slot.get("candidate_signature"),
                "generation_id": slot.get("generation_id"),
                "score": float(robust),
                "mean_dataset_score": float(mean_score),
                "std_dataset_score": float(std_score),
                "missing_datasets": missing,
                "dataset_scores": slot["dataset_scores"],
                "runtime_seconds": slot["runtime_seconds"],
                "dataset_metrics": slot["dataset_metrics"],
                "run_signatures": slot["run_signatures"],
                "run_dirs": slot["run_dirs"],
                "artifact_paths": slot["artifact_paths"],
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
    generation_id: Optional[str] = None,
    create_backup: bool = True,
) -> None:
    successful = [record for record in stage_results if record.get("status") == "ok"]
    failed = [record for record in stage_results if record.get("status") != "ok"]
    payload = {
        "stage": stage_name,
        "generation_id": generation_id,
        "num_runs": len(stage_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "rankings": list(rankings),
        "failed_runs": failed,
        "runs": list(stage_results),
        "execution": dict(execution or {}),
    }
    summary_path = output_dir / f"{stage_name}_summary.json"
    write_json_atomic(
        summary_path,
        payload,
        indent=2,
        sort_keys=False,
        create_backup=create_backup,
    )


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


def _metric_formulation_variants(
    formulations: Sequence[MetricFormulationSpec],
) -> List[MetricFormulationSpec]:
    variants: List[MetricFormulationSpec] = []
    for formulation in formulations:
        for alpha in formulation.alphas:
            variants.append(
                MetricFormulationSpec(
                    formulation_id=formulation.formulation_id,
                    family=formulation.family,
                    alphas=[float(alpha)],
                    component_weights=(
                        None
                        if formulation.component_weights is None
                        else dict(formulation.component_weights)
                    ),
                )
            )
    return variants


def _metric_variant_payload_from_values(
    formulation_id: str,
    family: str,
    alpha: float,
    component_weights: Optional[Dict[str, float]],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": str(formulation_id),
        "family": normalize_displacement_metric_family(family),
        "alpha": float(alpha),
    }
    if component_weights is not None:
        payload["component_weights"] = dict(component_weights)
    return payload


def _metric_variant_key_from_values(
    formulation_id: str,
    family: str,
    alpha: float,
    component_weights: Optional[Dict[str, float]],
) -> str:
    return json.dumps(
        _metric_variant_payload_from_values(
            formulation_id,
            family,
            alpha,
            component_weights,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def _metric_variant_key_from_formulation(formulation: MetricFormulationSpec) -> str:
    return _metric_variant_key_from_values(
        formulation.formulation_id,
        formulation.family,
        float(formulation.alphas[0]),
        formulation.component_weights,
    )


def _metric_variant_key_from_params(params: Mapping[str, object]) -> str:
    family, alpha, component_weights, formulation_id = _metric_formulation_from_params(
        dict(params)
    )
    return _metric_variant_key_from_values(
        formulation_id,
        family,
        alpha,
        component_weights,
    )


def _metric_variant_payload_from_key(key: str) -> Dict[str, object]:
    payload = json.loads(key)
    return cast(Dict[str, object], payload)


def _metric_search_stages(experimental_spec: ExperimentalMetricSweepSpec) -> SweepStageSpec:
    return _clone_stage_spec(experimental_spec.search_stages) or SweepStageSpec(
        total_budget_per_dataset=experimental_spec.shortlist_top_k,
        top_k=experimental_spec.shortlist_top_k,
    )


def _effective_metric_search_workers(requested_workers: int, search_stages: SweepStageSpec) -> int:
    parent_cap = max(1, int(requested_workers))
    if search_stages.max_workers is None:
        return parent_cap
    return max(1, min(parent_cap, int(search_stages.max_workers)))


def _independent_metric_stage_budgets(stage_spec: SweepStageSpec) -> Tuple[int, int]:
    total = max(1, int(stage_spec.total_budget_per_dataset))
    if stage_spec.mode == "single_stage":
        return total, 0
    return _stage_budgets(stage_spec)


def _eligible_metric_search_strategies(spec: SweepExperimentSpec) -> List[str]:
    return [strategy for strategy in spec.strategies if strategy in {"pixel", "combined"}]


def _base_metric_search_candidates(
    spec: SweepExperimentSpec,
    *,
    budget: int,
) -> List[SweepCandidate]:
    candidates: List[SweepCandidate] = []
    for strategy in _eligible_metric_search_strategies(spec):
        candidates.extend(generate_coarse_candidates(spec.search_spaces[strategy], budget=None))
    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=lambda item: json.dumps(item.params, sort_keys=True))
    return _limit_candidates(candidates, budget)


def _collect_independent_metric_stage_candidates(
    spec: SweepExperimentSpec,
    experimental_spec: ExperimentalMetricSweepSpec,
    *,
    budget_per_variant: int,
) -> List[SweepCandidate]:
    candidates: List[SweepCandidate] = []
    for variant in _metric_formulation_variants(experimental_spec.formulations):
        base_candidates = _base_metric_search_candidates(
            spec,
            budget=budget_per_variant,
        )
        candidates.extend(
            _expand_candidates_with_metric_formulations(base_candidates, [variant])
        )
    return _dedupe_candidates(candidates)


def _collect_independent_metric_refined_candidates(
    *,
    metric_stage1_rankings: Sequence[Dict[str, object]],
    metric_stage1_lookup: Dict[str, SweepCandidate],
    search_spaces: Dict[str, SweepSearchSpace],
    stage_spec: SweepStageSpec,
    budget_per_variant: int,
) -> List[SweepCandidate]:
    grouped_rankings: Dict[str, List[Dict[str, object]]] = {}
    for row in metric_stage1_rankings:
        params = cast(Dict[str, object], row.get("params", {}))
        grouped_rankings.setdefault(_metric_variant_key_from_params(params), []).append(
            row
        )

    refined_candidates: List[SweepCandidate] = []
    for rows in grouped_rankings.values():
        refined = generate_refined_candidates(
            ranked_candidates=rows,
            candidate_lookup=metric_stage1_lookup,
            search_spaces=search_spaces,
            top_k=stage_spec.top_k,
            budget=budget_per_variant,
            refinement=stage_spec.refinement,
        )
        if not refined and rows:
            best_id = str(rows[0].get("candidate_id", ""))
            best = metric_stage1_lookup.get(best_id)
            if best is not None:
                refined = [best]
        refined_candidates.extend(refined)

    return _dedupe_candidates(refined_candidates)


def _metric_candidate_count_payload(
    candidates: Sequence[SweepCandidate],
    *,
    dataset_count: int,
) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        key = _metric_variant_key_from_params(candidate.params)
        counts[key] = counts.get(key, 0) + 1

    rows: List[Dict[str, object]] = []
    for key in sorted(counts):
        payload = _metric_variant_payload_from_key(key)
        payload["num_candidates"] = int(counts[key])
        payload["num_runs"] = int(counts[key] * dataset_count)
        rows.append(payload)
    return rows


def _metric_experiment_plan_payload(
    *,
    experimental_spec: ExperimentalMetricSweepSpec,
    candidates: Sequence[SweepCandidate],
    dataset_count: int,
    status: str,
    output_dir: Optional[Path] = None,
    stage1_candidates: Sequence[SweepCandidate] = (),
    stage1_execution: Optional[Dict[str, object]] = None,
    final_execution: Optional[Dict[str, object]] = None,
    num_ranked: int = 0,
    num_results: Optional[int] = None,
) -> Dict[str, object]:
    search_stages = _metric_search_stages(experimental_spec)
    planned_final_runs = len(candidates) * dataset_count
    result_count = planned_final_runs if num_results is None else int(num_results)
    payload: Dict[str, object] = {
        "enabled": True,
        "candidate_source": experimental_spec.candidate_source,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
        "status": status,
        "shortlist_top_k": int(experimental_spec.shortlist_top_k),
        "search_stages": asdict(search_stages)
        if experimental_spec.candidate_source == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        else None,
        "num_formulations": len(experimental_spec.formulations),
        "num_variants": len(_metric_formulation_variants(experimental_spec.formulations)),
        "num_candidates": len(candidates),
        "num_planned_runs": int(planned_final_runs),
        "num_runs": int(result_count),
        "num_ranked": int(num_ranked),
        "per_formulation": _metric_candidate_count_payload(
            candidates,
            dataset_count=dataset_count,
        ),
    }
    if stage1_candidates:
        payload["internal_stage1"] = {
            "stage": METRIC_SEARCH_STAGE1_NAME,
            "num_candidates": len(stage1_candidates),
            "num_planned_runs": len(stage1_candidates) * dataset_count,
            "per_formulation": _metric_candidate_count_payload(
                stage1_candidates,
                dataset_count=dataset_count,
            ),
            "execution": dict(stage1_execution or {}),
        }
    if final_execution is not None:
        payload["execution"] = dict(final_execution)
    return payload


def _metric_experiment_dry_run_payload(
    *,
    spec: SweepExperimentSpec,
    prepared_datasets: Sequence[PreparedDataset],
) -> Dict[str, object]:
    experimental_spec = (
        spec.experimental_metric_sweep
        if spec.experimental_metric_sweep is not None
        else ExperimentalMetricSweepSpec(
            enabled=False,
            shortlist_top_k=spec.stages.top_k,
            formulations=[],
        )
    )
    if not experimental_spec.enabled:
        return {
            "enabled": False,
            "candidate_source": experimental_spec.candidate_source,
            "output_dir": None,
            "status": "disabled",
        }
    if experimental_spec.candidate_source == METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST:
        return {
            "enabled": True,
            "candidate_source": experimental_spec.candidate_source,
            "output_dir": None,
            "status": "requires_completed_baseline_rankings",
            "shortlist_top_k": int(experimental_spec.shortlist_top_k),
            "num_formulations": len(experimental_spec.formulations),
            "num_variants": len(_metric_formulation_variants(experimental_spec.formulations)),
        }

    search_stages = _metric_search_stages(experimental_spec)
    stage1_budget, stage2_budget = _independent_metric_stage_budgets(search_stages)
    stage1_candidates = _collect_independent_metric_stage_candidates(
        spec,
        experimental_spec,
        budget_per_variant=stage1_budget,
    )
    if search_stages.mode == "single_stage":
        final_candidates = stage1_candidates
    else:
        # Candidate identities for refinement depend on metric-stage-1 rankings.
        final_candidates = []
        for variant in _metric_formulation_variants(experimental_spec.formulations):
            base_candidates = _base_metric_search_candidates(
                spec,
                budget=stage2_budget,
            )
            final_candidates.extend(
                _expand_candidates_with_metric_formulations(base_candidates, [variant])
            )
        final_candidates = _dedupe_candidates(final_candidates)

    return _metric_experiment_plan_payload(
        experimental_spec=experimental_spec,
        candidates=final_candidates,
        dataset_count=len(prepared_datasets),
        status="planned",
        stage1_candidates=stage1_candidates
        if search_stages.mode != "single_stage"
        else (),
    )


def _stage_preview_payload(execution: Mapping[str, object]) -> Dict[str, object]:
    planned_uids = execution.get("planned_run_uids")
    planned_count = (
        len(cast(Sequence[object], planned_uids))
        if isinstance(planned_uids, list)
        else int(execution.get("completed_runs", 0) or 0)
        + int(execution.get("pending_run_count", 0) or 0)
    )
    return {
        **dict(execution),
        "planned_run_count": int(planned_count),
        "pending_run_count": int(execution.get("pending_run_count", 0) or 0),
        "skipped_valid_count": int(execution.get("skipped_valid_count", 0) or 0),
        "exact": True,
    }


def _metric_pending_count(payload: Mapping[str, object]) -> int:
    pending = int(payload.get("pending_run_count", payload.get("num_runs", 0)) or 0)
    internal = payload.get("internal_stage1")
    if isinstance(internal, dict):
        execution = internal.get("execution")
        if isinstance(execution, dict):
            pending += int(execution.get("pending_run_count", 0) or 0)
    return pending


def _metric_skipped_valid_count(payload: Mapping[str, object]) -> int:
    skipped = int(payload.get("skipped_valid_count", 0) or 0)
    internal = payload.get("internal_stage1")
    if isinstance(internal, dict):
        execution = internal.get("execution")
        if isinstance(execution, dict):
            skipped += int(execution.get("skipped_valid_count", 0) or 0)
    return skipped


def _metric_planned_count(payload: Mapping[str, object]) -> int:
    planned = int(payload.get("num_planned_runs", 0) or 0)
    internal = payload.get("internal_stage1")
    if isinstance(internal, dict):
        planned += int(internal.get("num_planned_runs", 0) or 0)
    return planned


def _disabled_metric_preview_payload(
    experimental_spec: ExperimentalMetricSweepSpec,
) -> Dict[str, object]:
    return {
        "enabled": False,
        "candidate_source": experimental_spec.candidate_source,
        "output_dir": None,
        "status": "disabled",
        "num_planned_runs": 0,
        "pending_run_count": 0,
        "skipped_valid_count": 0,
        "exact": True,
    }


def _estimate_baseline_shortlist_metric_preview(
    *,
    experimental_spec: ExperimentalMetricSweepSpec,
    baseline_candidate_lookup: Mapping[str, SweepCandidate],
    dataset_count: int,
) -> Dict[str, object]:
    eligible_candidates = [
        candidate
        for candidate in baseline_candidate_lookup.values()
        if candidate.strategy in {"pixel", "combined"}
    ]
    eligible_bases = _dedupe_candidates(
        [_candidate_without_metric_axes(candidate) for candidate in eligible_candidates]
    )
    shortlist_count = min(int(experimental_spec.shortlist_top_k), len(eligible_bases))
    estimated_candidates = _expand_candidates_with_metric_formulations(
        eligible_bases[:shortlist_count],
        experimental_spec.formulations,
    )
    estimated_candidates = _dedupe_candidates(estimated_candidates)
    planned_runs = len(estimated_candidates) * int(dataset_count)
    payload = _metric_experiment_plan_payload(
        experimental_spec=experimental_spec,
        candidates=estimated_candidates,
        dataset_count=dataset_count,
        status="estimated_requires_baseline_rankings",
    )
    payload.update(
        {
            "pending_run_count": int(planned_runs),
            "skipped_valid_count": 0,
            "exact": False,
            "estimate_reason": "baseline_rankings_unavailable",
        }
    )
    return payload


def _preview_independent_metric_search(
    *,
    spec: SweepExperimentSpec,
    experimental_spec: ExperimentalMetricSweepSpec,
    prepared_datasets: Sequence[PreparedDataset],
    metric_output_dir: Path,
    requested_workers: int,
    run_state: Dict[str, object],
    run_state_path: Optional[Path],
    generation_id: Optional[str],
    merge_sources: Sequence[Path],
    merge_log: Dict[str, object],
) -> Dict[str, object]:
    search_stages = _metric_search_stages(experimental_spec)
    metric_requested_workers = _effective_metric_search_workers(requested_workers, search_stages)
    stage1_budget, stage2_budget = _independent_metric_stage_budgets(search_stages)
    metric_stage1_candidates: List[SweepCandidate] = []
    metric_stage1_execution: Optional[Dict[str, object]] = None

    if search_stages.mode == "single_stage":
        metric_candidates = _assign_candidate_ids(
            _collect_independent_metric_stage_candidates(
                spec,
                experimental_spec,
                budget_per_variant=stage1_budget,
            ),
            "m1c",
        )
    else:
        metric_stage1_candidates = _assign_candidate_ids(
            _collect_independent_metric_stage_candidates(
                spec,
                experimental_spec,
                budget_per_variant=stage1_budget,
            ),
            "m1s1c",
        )
        metric_stage1_eval = _evaluate_stage(
            stage_name=METRIC_SEARCH_STAGE1_NAME,
            candidates=metric_stage1_candidates,
            datasets=prepared_datasets,
            output_dir=metric_output_dir,
            save_figures_flag=spec.save_figures,
            requested_workers=metric_requested_workers,
            scoring=spec.scoring,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
            dry_run=True,
            scheduling=search_stages.scheduling,
        )
        metric_stage1_execution = metric_stage1_eval.execution
        metric_candidates = _assign_candidate_ids(
            _dedupe_candidates(
                _expand_candidates_with_metric_formulations(
                    _base_metric_search_candidates(spec, budget=stage2_budget),
                    _metric_formulation_variants(experimental_spec.formulations),
                )
            ),
            "m1c",
        )

    metric_eval = _evaluate_stage(
        stage_name=METRIC_PHASE_STAGE_NAME,
        candidates=metric_candidates,
        datasets=prepared_datasets,
        output_dir=metric_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=metric_requested_workers,
        scoring=spec.scoring,
        run_state=run_state,
        run_state_path=run_state_path,
        generation_id=generation_id,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=True,
        scheduling=search_stages.scheduling,
    )
    payload = _metric_experiment_plan_payload(
        experimental_spec=experimental_spec,
        candidates=metric_candidates,
        dataset_count=len(prepared_datasets),
        status="planned",
        output_dir=metric_output_dir,
        stage1_candidates=metric_stage1_candidates,
        stage1_execution=metric_stage1_execution,
        final_execution=metric_eval.execution,
    )
    payload.update(
        {
            "pending_run_count": int(metric_eval.execution.get("pending_run_count", 0) or 0),
            "skipped_valid_count": int(metric_eval.execution.get("skipped_valid_count", 0) or 0),
            "exact": True,
        }
    )
    return payload


def _preview_baseline_shortlist_metric_phase(
    *,
    spec: SweepExperimentSpec,
    experimental_spec: ExperimentalMetricSweepSpec,
    prepared_datasets: Sequence[PreparedDataset],
    final_rankings: Sequence[Dict[str, object]],
    baseline_candidate_lookup: Mapping[str, SweepCandidate],
    metric_output_dir: Path,
    requested_workers: int,
    run_state: Dict[str, object],
    run_state_path: Optional[Path],
    generation_id: Optional[str],
    merge_sources: Sequence[Path],
    merge_log: Dict[str, object],
) -> Dict[str, object]:
    if not final_rankings:
        return _estimate_baseline_shortlist_metric_preview(
            experimental_spec=experimental_spec,
            baseline_candidate_lookup=baseline_candidate_lookup,
            dataset_count=len(prepared_datasets),
        )

    metric_candidates = _collect_metric_experiment_candidates(
        final_rankings=final_rankings,
        baseline_candidate_lookup=dict(baseline_candidate_lookup),
        experimental_spec=experimental_spec,
    )
    metric_candidates = _assign_candidate_ids(metric_candidates, "m1c")
    metric_eval = _evaluate_stage(
        stage_name=METRIC_PHASE_STAGE_NAME,
        candidates=metric_candidates,
        datasets=prepared_datasets,
        output_dir=metric_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=requested_workers,
        scoring=spec.scoring,
        run_state=run_state,
        run_state_path=run_state_path,
        generation_id=generation_id,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=True,
        scheduling=spec.stages.scheduling,
    )
    payload = _metric_experiment_plan_payload(
        experimental_spec=experimental_spec,
        candidates=metric_candidates,
        dataset_count=len(prepared_datasets),
        status="planned",
        output_dir=metric_output_dir,
        final_execution=metric_eval.execution,
    )
    payload.update(
        {
            "pending_run_count": int(metric_eval.execution.get("pending_run_count", 0) or 0),
            "skipped_valid_count": int(metric_eval.execution.get("skipped_valid_count", 0) or 0),
            "exact": True,
        }
    )
    return payload


def _count_only_stage_payload(planned_runs: int, *, status: str = "count_only") -> Dict[str, object]:
    return {
        "planned_run_count": int(planned_runs),
        "pending_run_count": int(planned_runs),
        "skipped_valid_count": 0,
        "exact": False,
        "status": status,
        "estimate_reason": "dataset_metadata_not_required",
    }


def _count_only_independent_metric_payload(
    *,
    spec: SweepExperimentSpec,
    experimental_spec: ExperimentalMetricSweepSpec,
    dataset_count: int,
) -> Dict[str, object]:
    search_stages = _metric_search_stages(experimental_spec)
    stage1_budget, stage2_budget = _independent_metric_stage_budgets(search_stages)
    stage1_candidates: List[SweepCandidate] = []
    if search_stages.mode == "single_stage":
        metric_candidates = _collect_independent_metric_stage_candidates(
            spec,
            experimental_spec,
            budget_per_variant=stage1_budget,
        )
    else:
        stage1_candidates = _collect_independent_metric_stage_candidates(
            spec,
            experimental_spec,
            budget_per_variant=stage1_budget,
        )
        metric_candidates = []
        for variant in _metric_formulation_variants(experimental_spec.formulations):
            base_candidates = _base_metric_search_candidates(spec, budget=stage2_budget)
            metric_candidates.extend(
                _expand_candidates_with_metric_formulations(base_candidates, [variant])
            )
        metric_candidates = _dedupe_candidates(metric_candidates)

    payload = _metric_experiment_plan_payload(
        experimental_spec=experimental_spec,
        candidates=metric_candidates,
        dataset_count=dataset_count,
        status="count_only",
        stage1_candidates=stage1_candidates,
    )
    payload.update(
        {
            "pending_run_count": int(payload.get("num_planned_runs", 0) or 0),
            "skipped_valid_count": 0,
            "exact": False,
            "estimate_reason": "dataset_metadata_not_required",
        }
    )
    internal = payload.get("internal_stage1")
    if isinstance(internal, dict):
        internal["execution"] = _count_only_stage_payload(
            int(internal.get("num_planned_runs", 0) or 0)
        )
    return payload


def _build_count_only_sweep_run_preview(
    *,
    spec: SweepExperimentSpec,
    resolved_output_dir: Path,
    extend_existing: bool,
    merge_sources: Sequence[Path],
    relocation_rules: Sequence[PathRelocationRule],
    relocation_report: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    dataset_count = len(spec.datasets)
    stage_mode = spec.stages.mode
    stage1_budget, stage2_budget = _stage_budgets(spec.stages)
    validation_spec = _validation_oracle_spec(spec)
    skip_image_metric_stage = _skip_image_metric_stage_for_validation(validation_spec)
    baseline_budget = (
        stage1_budget
        if stage_mode == "two_stage"
        else max(1, int(spec.stages.total_budget_per_dataset))
    )
    coarse_candidates_base: List[SweepCandidate] = []
    for strategy in spec.strategies:
        search_space = spec.search_spaces[strategy]
        coarse_candidates_base.extend(generate_coarse_candidates(search_space, budget=None))
    coarse_candidates = _assign_candidate_ids(
        _limit_candidates(_dedupe_candidates(coarse_candidates_base), baseline_budget),
        "s1c",
    )
    stages_payload: Dict[str, object] = {
        "stage1": _count_only_stage_payload(len(coarse_candidates) * dataset_count)
    }
    refined_lookup: Dict[str, SweepCandidate] = {}
    if stage_mode == "two_stage" and not skip_image_metric_stage:
        stages_payload["stage2"] = _count_only_stage_payload(
            max(1, int(stage2_budget)) * dataset_count,
            status="estimated_requires_stage1_rankings",
        )

    experimental_spec = (
        spec.experimental_metric_sweep
        if spec.experimental_metric_sweep is not None
        else ExperimentalMetricSweepSpec(
            enabled=False,
            shortlist_top_k=spec.stages.top_k,
            formulations=[],
        )
    )
    if not experimental_spec.enabled:
        metric_payload = _disabled_metric_preview_payload(experimental_spec)
    elif experimental_spec.candidate_source == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH:
        metric_payload = _count_only_independent_metric_payload(
            spec=spec,
            experimental_spec=experimental_spec,
            dataset_count=dataset_count,
        )
    else:
        baseline_candidate_lookup = {candidate.candidate_id: candidate for candidate in coarse_candidates}
        baseline_candidate_lookup.update(refined_lookup)
        metric_payload = _estimate_baseline_shortlist_metric_preview(
            experimental_spec=experimental_spec,
            baseline_candidate_lookup=baseline_candidate_lookup,
            dataset_count=dataset_count,
        )
    validation_phase_budget = _validation_oracle_phase_budget(validation_spec, stage2_budget)
    validation_payload = _validation_oracle_dry_run_payload(
        validation_spec=validation_spec,
        dataset_ids=[dataset.dataset_id for dataset in spec.datasets],
        validation_phase_budget=validation_phase_budget,
    )

    stage_planned = sum(
        int(cast(Mapping[str, object], payload).get("planned_run_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    stage_pending = sum(
        int(cast(Mapping[str, object], payload).get("pending_run_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    stage_skipped = sum(
        int(cast(Mapping[str, object], payload).get("skipped_valid_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    return {
        "generation_id": "preview_count_only",
        "generated_at": _utc_now_iso(),
        "canonical_output_dir": str(resolved_output_dir.resolve()),
        "extend_existing": bool(extend_existing),
        "merge_sources": [str(path.resolve()) for path in merge_sources],
        "relocation": relocation_report
        or {
            "rules": _relocation_rules_to_payload(relocation_rules),
            "applied": [],
        },
        "stages": stages_payload,
        "metric_experiments": metric_payload,
        "validation_oracle_sweep": validation_payload,
        "totals": {
            "planned_run_count": int(
                stage_planned
                + _metric_planned_count(metric_payload)
                + _validation_oracle_planned_count(validation_payload)
            ),
            "pending_run_count": int(
                stage_pending
                + _metric_pending_count(metric_payload)
                + _validation_oracle_planned_count(validation_payload)
            ),
            "skipped_valid_count": int(stage_skipped + _metric_skipped_valid_count(metric_payload)),
            "exact": False,
        },
        "merge_log": {
            "generation_id": "preview_count_only",
            "created_at": _utc_now_iso(),
            "canonical_output_dir": str(resolved_output_dir.resolve()),
            "merge_sources": [str(path.resolve()) for path in merge_sources],
            "sources": [],
            "copied": [],
            "would_copy": [],
            "skipped_duplicates": [],
            "rejected": [],
            "conflicts": [],
        },
    }


def build_sweep_run_preview(
    spec: SweepExperimentSpec,
    output_dir: Optional[Path] = None,
    *,
    extend_existing: bool = False,
    merge_from: Optional[Sequence[Path]] = None,
    adopt_relocated: bool = False,
    relocation_rules: Sequence[PathRelocationRule] = (),
    relocation_report: Optional[Mapping[str, object]] = None,
    allow_dataset_generation: bool = False,
    reuse_existing_artifacts: bool = True,
) -> Dict[str, object]:
    """Build a non-rectifying run-count preview for baseline and metric stages."""

    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else (
            Path(spec.output_dir)
            if spec.output_dir
            else Path("sweep_runs") / "preview"
        )
    )
    merge_sources = [Path(path) for path in (merge_from or [])]
    if not reuse_existing_artifacts:
        return _build_count_only_sweep_run_preview(
            spec=spec,
            resolved_output_dir=resolved_output_dir,
            extend_existing=extend_existing,
            merge_sources=merge_sources,
            relocation_rules=relocation_rules,
            relocation_report=relocation_report,
        )
    dataset_identities = _read_dataset_identity_payloads(resolved_output_dir)
    prepared_datasets = _prepare_datasets(
        spec,
        resolved_output_dir,
        dataset_identities=dataset_identities,
        allow_dataset_generation=allow_dataset_generation,
    )
    dataset_ids = [dataset.dataset_id for dataset in prepared_datasets]
    validation_spec = _validation_oracle_spec(spec)
    validation_phase_budget = _validation_oracle_phase_budget(
        validation_spec,
        _stage_budgets(spec.stages)[1],
    )
    _validate_validation_oracle_scene_lookup(
        validation_spec,
        dataset_ids,
    )
    fingerprint_payload = _build_experiment_fingerprint_payload(
        spec=spec,
        prepared_datasets=prepared_datasets,
        output_dir=resolved_output_dir,
    )
    run_state, run_state_path, generation_id = _load_or_initialize_run_state(
        output_dir=resolved_output_dir,
        fingerprint_payload=fingerprint_payload,
        allow_extend=extend_existing,
        allow_merge=bool(merge_sources),
        adopt_relocated=bool(adopt_relocated),
        relocation_rules=relocation_rules,
        relocation_report=relocation_report,
        merge_sources=merge_sources,
        dry_run=True,
    )
    merge_log: Dict[str, object] = {
        "generation_id": generation_id,
        "created_at": _utc_now_iso(),
        "canonical_output_dir": str(resolved_output_dir.resolve()),
        "merge_sources": [str(path.resolve()) for path in merge_sources],
        "sources": [],
        "copied": [],
        "would_copy": [],
        "skipped_duplicates": [],
        "rejected": [],
        "conflicts": [],
    }

    stage_mode = spec.stages.mode
    stage1_budget, stage2_budget = _stage_budgets(spec.stages)
    skip_image_metric_stage = _skip_image_metric_stage_for_validation(validation_spec)
    baseline_budget = (
        stage1_budget
        if stage_mode == "two_stage"
        else max(1, int(spec.stages.total_budget_per_dataset))
    )
    coarse_candidates_base: List[SweepCandidate] = []
    for strategy in spec.strategies:
        search_space = spec.search_spaces[strategy]
        coarse_candidates_base.extend(generate_coarse_candidates(search_space, budget=None))
    coarse_candidates = _assign_candidate_ids(
        _limit_candidates(_dedupe_candidates(coarse_candidates_base), baseline_budget),
        "s1c",
    )
    coarse_lookup = {candidate.candidate_id: candidate for candidate in coarse_candidates}
    requested_workers = _default_max_workers(spec.stages.max_workers)

    stage1_eval = _evaluate_stage(
        stage_name="stage1",
        candidates=coarse_candidates,
        datasets=prepared_datasets,
        output_dir=resolved_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=requested_workers,
        scoring=spec.scoring,
        run_state=run_state,
        run_state_path=run_state_path,
        generation_id=generation_id,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=True,
        scheduling=spec.stages.scheduling,
    )
    stage1_rankings = score_and_rank_runs(
        stage1_eval.stage_results,
        spec.scoring,
        dataset_ids=dataset_ids,
    )
    stages_payload: Dict[str, object] = {
        "stage1": _stage_preview_payload(stage1_eval.execution)
    }

    refined_lookup: Dict[str, SweepCandidate] = {}
    stage2_rankings: List[Dict[str, object]] = []
    if stage_mode == "two_stage" and not skip_image_metric_stage:
        refined_candidates = generate_refined_candidates(
            ranked_candidates=stage1_rankings,
            candidate_lookup=coarse_lookup,
            search_spaces=spec.search_spaces,
            top_k=spec.stages.top_k,
            budget=stage2_budget,
            refinement=spec.stages.refinement,
        )
        if not refined_candidates and stage1_rankings:
            best_id = str(stage1_rankings[0]["candidate_id"])
            refined_candidates = [coarse_lookup[best_id]]
        refined_candidates = _assign_candidate_ids(_dedupe_candidates(refined_candidates), "s2c")
        refined_lookup = {candidate.candidate_id: candidate for candidate in refined_candidates}
        stage2_eval = _evaluate_stage(
            stage_name="stage2",
            candidates=refined_candidates,
            datasets=prepared_datasets,
            output_dir=resolved_output_dir,
            save_figures_flag=spec.save_figures,
            requested_workers=requested_workers,
            scoring=spec.scoring,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
            dry_run=True,
            scheduling=spec.stages.scheduling,
        )
        stage2_rankings = score_and_rank_runs(
            stage2_eval.stage_results,
            spec.scoring,
            dataset_ids=dataset_ids,
        )
        stages_payload["stage2"] = _stage_preview_payload(stage2_eval.execution)

    final_rankings = stage2_rankings if stage2_rankings else stage1_rankings
    experimental_spec = (
        spec.experimental_metric_sweep
        if spec.experimental_metric_sweep is not None
        else ExperimentalMetricSweepSpec(
            enabled=False,
            shortlist_top_k=spec.stages.top_k,
            formulations=[],
        )
    )
    if not experimental_spec.enabled:
        metric_payload = _disabled_metric_preview_payload(experimental_spec)
    else:
        metric_output_dir = resolved_output_dir / "metric_experiments"
        if experimental_spec.candidate_source == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH:
            metric_payload = _preview_independent_metric_search(
                spec=spec,
                experimental_spec=experimental_spec,
                prepared_datasets=prepared_datasets,
                metric_output_dir=metric_output_dir,
                requested_workers=requested_workers,
                run_state=run_state,
                run_state_path=run_state_path,
                generation_id=generation_id,
                merge_sources=merge_sources,
                merge_log=merge_log,
            )
        else:
            baseline_candidate_lookup = dict(coarse_lookup)
            baseline_candidate_lookup.update(refined_lookup)
            metric_payload = _preview_baseline_shortlist_metric_phase(
                spec=spec,
                experimental_spec=experimental_spec,
                prepared_datasets=prepared_datasets,
                final_rankings=final_rankings,
                baseline_candidate_lookup=baseline_candidate_lookup,
                metric_output_dir=metric_output_dir,
                requested_workers=requested_workers,
                run_state=run_state,
                run_state_path=run_state_path,
                generation_id=generation_id,
                merge_sources=merge_sources,
                merge_log=merge_log,
            )

    validation_payload = _validation_oracle_dry_run_payload(
        validation_spec=validation_spec,
        dataset_ids=dataset_ids,
        validation_phase_budget=validation_phase_budget,
    )
    stage_planned = sum(
        int(cast(Mapping[str, object], payload).get("planned_run_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    stage_pending = sum(
        int(cast(Mapping[str, object], payload).get("pending_run_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    stage_skipped = sum(
        int(cast(Mapping[str, object], payload).get("skipped_valid_count", 0) or 0)
        for payload in stages_payload.values()
        if isinstance(payload, dict)
    )
    totals = {
        "planned_run_count": int(
            stage_planned
            + _metric_planned_count(metric_payload)
            + _validation_oracle_planned_count(validation_payload)
        ),
        "pending_run_count": int(
            stage_pending
            + _metric_pending_count(metric_payload)
            + _validation_oracle_planned_count(validation_payload)
        ),
        "skipped_valid_count": int(stage_skipped + _metric_skipped_valid_count(metric_payload)),
        "exact": bool(
            all(
                bool(cast(Mapping[str, object], payload).get("exact", False))
                for payload in stages_payload.values()
                if isinstance(payload, dict)
            )
            and bool(metric_payload.get("exact", False))
            and bool(validation_payload.get("exact", False))
        ),
    }
    return {
        "generation_id": generation_id,
        "generated_at": _utc_now_iso(),
        "canonical_output_dir": str(resolved_output_dir.resolve()),
        "extend_existing": bool(extend_existing),
        "merge_sources": [str(path.resolve()) for path in merge_sources],
        "relocation": relocation_report
        or {
            "rules": _relocation_rules_to_payload(relocation_rules),
            "applied": [],
        },
        "stages": stages_payload,
        "metric_experiments": metric_payload,
        "validation_oracle_sweep": validation_payload,
        "totals": totals,
        "merge_log": merge_log,
    }


def _write_phase_summary_artifacts(
    *,
    output_dir: Path,
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
    execution: Optional[Dict[str, object]],
    reporting: SweepReportingSpec,
    analyze: bool,
    generation_id: Optional[str] = None,
) -> None:
    _write_stage_outputs(
        stage_name,
        stage_results,
        rankings,
        output_dir,
        execution=execution,
        generation_id=generation_id,
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
    if reporting.generate_alpha_report and stage_name == "metric_phase":
        generate_alpha_report(
            output_dir=output_dir,
            primary_metric=reporting.primary_metric,
            tolerate_partial_outputs=reporting.tolerate_partial_outputs,
        )


def _validation_oracle_spec(spec: SweepExperimentSpec) -> ValidationOracleSweepSpec:
    return spec.validation_oracle_sweep or ValidationOracleSweepSpec(
        enabled=False,
        top_k=spec.stages.top_k,
    )


def _skip_image_metric_stage_for_validation(validation_spec: ValidationOracleSweepSpec) -> bool:
    return bool(validation_spec.enabled and validation_spec.skip_image_metric_stage)


def _validation_oracle_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validation_oracle_scene_has_workspace(scene: Mapping[str, object]) -> bool:
    return bool(_validation_oracle_text(scene.get("validation_workspace")))


def _validation_oracle_scene_has_image_pair(scene: Mapping[str, object]) -> bool:
    return bool(_validation_oracle_text(scene.get("validation_image"))) and bool(
        _validation_oracle_text(scene.get("thorcam_image"))
    )


def _validation_oracle_scene_human_bead_annotations(scene: Mapping[str, object]) -> str:
    return _validation_oracle_text(scene.get("human_bead_annotations"))


def _validation_oracle_score_metric(validation_spec: ValidationOracleSweepSpec) -> str:
    return (
        HUMAN_BEAD_ORACLE_SCORE
        if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD
        else TRIO_PRIMARY_SCORE
    )


def _strip_validation_oracle_gui_owned_scene_fields(
    scene_payload: Mapping[str, object],
) -> Dict[str, object]:
    scene_entry = dict(scene_payload)
    scene_entry.pop("strict", None)
    for key in VALIDATION_ORACLE_GUI_OWNED_SCENE_KEYS:
        scene_entry.pop(key, None)
    return scene_entry


def _validate_validation_oracle_scene_lookup(
    validation_spec: ValidationOracleSweepSpec,
    dataset_ids: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    if not validation_spec.enabled:
        return {}

    scenes_by_dataset: Dict[str, Dict[str, object]] = {}
    duplicates: List[str] = []
    for scene in validation_spec.scenes:
        dataset_id = str(scene.get("dataset_id") or scene.get("id") or "").strip()
        if not dataset_id:
            raise ValueError("validation_oracle_sweep.scenes entries must include dataset_id.")
        if dataset_id in scenes_by_dataset:
            duplicates.append(dataset_id)
        scenes_by_dataset[dataset_id] = dict(scene)
    if duplicates:
        raise ValueError(
            "validation_oracle_sweep has duplicate scene entries for dataset(s): "
            + ", ".join(sorted(set(duplicates)))
        )

    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in scenes_by_dataset]
    if missing:
        raise ValueError(
            "validation_oracle_sweep requires one scene for every dataset; missing: "
            + ", ".join(missing)
        )

    for dataset_id in dataset_ids:
        merged = dict(validation_spec.defaults)
        merged.update(scenes_by_dataset[dataset_id])
        missing_paths = []
        for key in VALIDATION_ORACLE_PATH_KEYS:
            value = merged.get(key)
            if value is None or not str(value).strip():
                continue
            if not Path(str(value)).exists():
                missing_paths.append(f"{key}={value}")
        if missing_paths:
            raise ValueError(
                "validation_oracle_sweep scene for dataset "
                f"{dataset_id!r} references missing file(s): "
                + ", ".join(missing_paths)
            )
        if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD:
            annotations_path = _validation_oracle_scene_human_bead_annotations(merged)
            if not annotations_path:
                raise ValueError(
                    "validation_oracle_sweep human_bead scene for dataset "
                    f"{dataset_id!r} must provide human_bead_annotations."
                )
            annotations = load_human_bead_annotations(Path(annotations_path))
            if dataset_id not in annotations:
                raise ValueError(
                    "validation_oracle_sweep human_bead annotations "
                    f"{annotations_path!r} do not contain dataset {dataset_id!r}."
                )
            scenes_by_dataset[dataset_id] = merged
            continue
        if not (
            _validation_oracle_scene_has_workspace(merged)
            or _validation_oracle_scene_has_image_pair(merged)
        ):
            raise ValueError(
                "validation_oracle_sweep scene for dataset "
                f"{dataset_id!r} must provide validation_workspace or both "
                "validation_image and thorcam_image."
            )
        scenes_by_dataset[dataset_id] = merged
    return scenes_by_dataset


def _validation_oracle_phase_budget(
    validation_spec: ValidationOracleSweepSpec,
    fallback_budget: int,
) -> int:
    if validation_spec.budget_per_dataset is not None:
        return max(1, int(validation_spec.budget_per_dataset))
    return max(1, int(fallback_budget))


def _validation_oracle_dry_run_payload(
    *,
    validation_spec: ValidationOracleSweepSpec,
    dataset_ids: Sequence[str],
    validation_phase_budget: int,
) -> Dict[str, object]:
    if not validation_spec.enabled:
        return {
            "enabled": False,
            "output_dir": None,
            "status": "disabled",
        }
    _validate_validation_oracle_scene_lookup(validation_spec, dataset_ids)
    return {
        "enabled": True,
        "status": "planned_after_stage1_oracle_ranking",
        "stage": VALIDATION_ORACLE_STAGE_NAME,
        "oracle_mode": validation_spec.oracle_mode,
        "score_metric": _validation_oracle_score_metric(validation_spec),
        "top_k": int(validation_spec.top_k),
        "budget_per_dataset": int(validation_phase_budget),
        "skip_image_metric_stage": bool(validation_spec.skip_image_metric_stage),
        "num_scenes": len(dataset_ids),
        "num_stage1_benchmarks": (
            0
            if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD
            else len(dataset_ids)
        ),
        "estimated_validation_phase_runs": int(validation_phase_budget * len(dataset_ids)),
        "estimated_validation_phase_benchmarks": (
            0
            if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD
            else len(dataset_ids)
        ),
        "exact": False,
    }


def _validation_oracle_planned_count(payload: Mapping[str, object]) -> int:
    if not bool(payload.get("enabled", False)):
        return 0
    return int(payload.get("estimated_validation_phase_runs", 0) or 0)


def _benchmark_config_from_validation_scene(
    *,
    validation_spec: ValidationOracleSweepSpec,
    scene: Mapping[str, object],
    sweep_dir: Path,
    dataset_id: str,
    stage_name: str,
    output_dir: Path,
) -> BenchmarkConfig:
    benchmark_payload: Dict[str, object] = dict(validation_spec.defaults)
    benchmark_payload.update(dict(scene))
    benchmark_payload.pop("strict", None)
    benchmark_payload.pop("id", None)
    benchmark_payload.setdefault("generation_scope", "Active")
    if (
        _validation_oracle_scene_has_workspace(benchmark_payload)
        and "use_validation_workspace_transforms" not in benchmark_payload
    ):
        benchmark_payload["use_validation_workspace_transforms"] = True
    field_names = {item.name for item in fields(BenchmarkConfig)}
    unsupported = sorted(set(benchmark_payload) - field_names)
    if unsupported:
        raise ValueError(
            "Unsupported validation_oracle_sweep scene/default field(s): "
            + ", ".join(unsupported)
        )
    benchmark_payload.update(
        {
            "sweep_dir": sweep_dir,
            "dataset_id": dataset_id,
            "stage": stage_name,
            "output_dir": output_dir,
            "top_k": int(validation_spec.top_k),
        }
    )
    for key in VALIDATION_ORACLE_PATH_KEYS + ("sweep_dir", "output_dir"):
        value = benchmark_payload.get(key)
        if value is None or isinstance(value, Path):
            continue
        if isinstance(value, str) and not value.strip():
            benchmark_payload[key] = None
        else:
            benchmark_payload[key] = Path(str(value))
    return BenchmarkConfig(**benchmark_payload)


def _run_validation_oracle_benchmarks(
    *,
    validation_spec: ValidationOracleSweepSpec,
    scene_lookup: Mapping[str, Mapping[str, object]],
    sweep_dir: Path,
    dataset_ids: Sequence[str],
    stage_name: str,
    oracle_output_dir: Optional[Path] = None,
    metric_review_dir: Optional[Path] = None,
) -> Tuple[object, Dict[str, object]]:
    oracle_root = oracle_output_dir or (sweep_dir / VALIDATION_ORACLE_DIR_NAME)
    stage_root = oracle_root / stage_name
    csv_paths: List[Path] = []
    scene_results: List[Dict[str, object]] = []
    for dataset_id in dataset_ids:
        benchmark_dir = stage_root / dataset_id
        config = _benchmark_config_from_validation_scene(
            validation_spec=validation_spec,
            scene=scene_lookup[dataset_id],
            sweep_dir=sweep_dir,
            dataset_id=dataset_id,
            stage_name=stage_name,
            output_dir=benchmark_dir,
        )
        result = run_benchmark(config)
        csv_path = benchmark_dir / "trio_benchmark_results.csv"
        if not csv_path.exists():
            raise ValueError(
                "validation_oracle_sweep benchmark did not write expected trio CSV: "
                f"{csv_path}"
            )
        csv_paths.append(csv_path)
        scene_results.append(
            {
                "dataset_id": dataset_id,
                "stage": stage_name,
                "scene_id": config.scene_id,
                "output_dir": str(benchmark_dir.resolve()),
                "result_csv": str(csv_path.resolve()),
                "best_lf_row_id": getattr(result, "best_lf_row_id", None),
            }
        )

    import_data = import_trio_oracle_results(metric_review_dir or (sweep_dir / "metric_review"), [oracle_root])
    manifest = {
        "stage": stage_name,
        "score_metric": TRIO_PRIMARY_SCORE,
        "csv_paths": [str(path.resolve()) for path in csv_paths],
        "scenes": scene_results,
        "num_records": len(import_data.records),
        "warnings": list(import_data.warnings),
    }
    write_json_atomic(
        oracle_root / f"{stage_name}_trio_import.json",
        manifest,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    return import_data, manifest


def _human_bead_annotations_for_scenes(
    scene_lookup: Mapping[str, Mapping[str, object]],
    dataset_ids: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, Path]]:
    annotations_by_dataset: Dict[str, object] = {}
    loaded_by_path: Dict[Path, Dict[str, object]] = {}
    source_paths_by_dataset: Dict[str, Path] = {}
    for dataset_id in dataset_ids:
        annotation_path_text = _validation_oracle_scene_human_bead_annotations(
            scene_lookup[dataset_id]
        )
        if not annotation_path_text:
            raise ValueError(
                "validation_oracle_sweep human_bead scene for dataset "
                f"{dataset_id!r} must provide human_bead_annotations."
            )
        annotation_path = Path(annotation_path_text)
        source_paths_by_dataset[dataset_id] = annotation_path.resolve()
        if annotation_path not in loaded_by_path:
            loaded_by_path[annotation_path] = cast(
                Dict[str, object],
                load_human_bead_annotations(annotation_path),
            )
        annotations = loaded_by_path[annotation_path]
        if dataset_id not in annotations:
            raise ValueError(
                "validation_oracle_sweep human_bead annotations "
                f"{annotation_path!s} do not contain dataset {dataset_id!r}."
            )
        annotations_by_dataset[dataset_id] = annotations[dataset_id]
    return annotations_by_dataset, source_paths_by_dataset


def _run_human_bead_oracle_stage(
    *,
    scene_lookup: Mapping[str, Mapping[str, object]],
    sweep_dir: Path,
    dataset_ids: Sequence[str],
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    oracle_output_dir: Optional[Path] = None,
    metric_review_dir: Optional[Path] = None,
) -> Tuple[object, Dict[str, object]]:
    oracle_root = oracle_output_dir or (sweep_dir / VALIDATION_ORACLE_DIR_NAME)
    stage_root = oracle_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    annotations_by_dataset, source_paths_by_dataset = _human_bead_annotations_for_scenes(
        scene_lookup,
        dataset_ids,
    )
    stage_import = build_human_bead_oracle_import(
        stage_name=stage_name,
        stage_results=stage_results,
        annotations=annotations_by_dataset,
        annotation_source_paths_by_dataset=source_paths_by_dataset,
    )
    import_data = write_human_bead_oracle_import(
        metric_review_dir or (sweep_dir / "metric_review"),
        stage_import,
    )
    source_paths = list(dict.fromkeys(source_paths_by_dataset.values()))
    manifest = {
        "stage": stage_name,
        "oracle_mode": VALIDATION_ORACLE_MODE_HUMAN_BEAD,
        "score_metric": HUMAN_BEAD_ORACLE_SCORE,
        "source_paths": [str(path.resolve()) for path in source_paths],
        "num_records": len(stage_import.records),
        "num_rankable": sum(1 for record in stage_import.records if record.validation_rankable),
        "warnings": list(stage_import.warnings),
    }
    write_json_atomic(
        stage_root / "human_bead_oracle_import.json",
        manifest,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    return import_data, manifest


def _rank_stage_with_validation_oracle(
    *,
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    validation_import: object,
    dataset_ids: Sequence[str],
    scoring: SweepScoringSpec,
    score_metric: str = TRIO_PRIMARY_SCORE,
    allow_partial_dataset_rankability: bool = False,
) -> List[Dict[str, object]]:
    records = cast(Any, validation_import).records
    trio_lookup = {
        (
            str(record.stage),
            str(record.candidate_id),
            str(record.dataset_id),
            record.generation_id,
        ): record
        for record in records
        if str(record.source_kind) == TRIO_LF_SOURCE_KIND
    }
    rankable_count_by_dataset = {dataset_id: 0 for dataset_id in dataset_ids}
    grouped: Dict[str, Dict[str, object]] = {}

    for record in stage_results:
        if record.get("status") != "ok":
            continue
        candidate_id = str(record.get("candidate_id", ""))
        dataset_id = str(record.get("dataset_id", ""))
        if not candidate_id or dataset_id not in rankable_count_by_dataset:
            continue
        trio_record = find_trio_record_for_identity(
            trio_lookup,
            stage=stage_name,
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            generation_id=cast(Optional[str], record.get("generation_id")),
        )
        if trio_record is None or not trio_rankable(trio_record):
            continue
        score = trio_record.metrics.get(score_metric)
        if score is None or not math.isfinite(float(score)):
            continue
        rankable_count_by_dataset[dataset_id] += 1
        slot = grouped.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "strategy": record["strategy"],
                "config": record["config"],
                "params": record["params"],
                "candidate_signature": record.get("candidate_signature"),
                "generation_id": record.get("generation_id"),
                "dataset_scores": {},
                "dataset_metrics": {},
                "dataset_trio_diagnostics": {},
                "runtime_seconds": {},
                "run_signatures": {},
                "run_dirs": {},
                "artifact_paths": {},
            },
        )
        cast(Dict[str, float], slot["dataset_scores"])[dataset_id] = float(score)
        cast(Dict[str, Dict[str, float]], slot["dataset_metrics"])[dataset_id] = dict(
            trio_record.metrics
        )
        cast(Dict[str, Dict[str, object]], slot["dataset_trio_diagnostics"])[dataset_id] = {
            "validation_rank": trio_record.validation_rank,
            "validation_rankable": trio_record.validation_rankable,
            "ranking_exclusion_reason": trio_record.ranking_exclusion_reason,
            "row_id": trio_record.row_id,
            "source_path": trio_record.source_path,
            **dict(trio_record.diagnostics),
        }
        cast(Dict[str, float], slot["runtime_seconds"])[dataset_id] = float(
            record.get("runtime_seconds", 0.0)
        )
        cast(Dict[str, object], slot["run_signatures"])[dataset_id] = record.get(
            "run_signature"
        )
        cast(Dict[str, object], slot["run_dirs"])[dataset_id] = record.get("run_dir")
        cast(Dict[str, object], slot["artifact_paths"])[dataset_id] = record.get(
            "artifact_paths"
        )

    missing_rankable = [
        dataset_id for dataset_id, count in rankable_count_by_dataset.items() if count <= 0
    ]
    if missing_rankable and not allow_partial_dataset_rankability:
        raise ValueError(
            f"validation_oracle_sweep produced no rankable {score_metric} LF rows "
            "for dataset(s): "
            + ", ".join(missing_rankable)
        )
    ranked_dataset_ids = [
        dataset_id for dataset_id in dataset_ids if rankable_count_by_dataset.get(dataset_id, 0) > 0
    ]
    if not ranked_dataset_ids:
        raise ValueError("validation_oracle_sweep produced no rankable validation rankings.")

    rankings: List[Dict[str, object]] = []
    for slot in grouped.values():
        dataset_scores_map = cast(Dict[str, float], slot["dataset_scores"])
        present_scores = [
            dataset_scores_map[dataset_id]
            for dataset_id in ranked_dataset_ids
            if dataset_id in dataset_scores_map
        ]
        missing = [
            dataset_id for dataset_id in ranked_dataset_ids if dataset_id not in dataset_scores_map
        ]
        mean_score = float(np.mean(present_scores)) if present_scores else -float("inf")
        std_score = float(np.std(present_scores)) if present_scores else 0.0
        robust = mean_score - float(scoring.robustness_lambda) * std_score
        robust -= float(scoring.missing_dataset_penalty) * len(missing)
        rankings.append(
            {
                "candidate_id": slot["candidate_id"],
                "strategy": slot["strategy"],
                "config": slot["config"],
                "params": slot["params"],
                "candidate_signature": slot.get("candidate_signature"),
                "generation_id": slot.get("generation_id"),
                "score": float(robust),
                "mean_dataset_score": float(mean_score),
                "std_dataset_score": float(std_score),
                "missing_datasets": missing,
                "dataset_scores": slot["dataset_scores"],
                "dataset_metrics": slot["dataset_metrics"],
                "dataset_trio_diagnostics": slot["dataset_trio_diagnostics"],
                "runtime_seconds": slot["runtime_seconds"],
                "run_signatures": slot["run_signatures"],
                "run_dirs": slot["run_dirs"],
                "artifact_paths": slot["artifact_paths"],
                "score_metric": score_metric,
                "ranked_dataset_ids": list(ranked_dataset_ids),
                "unrankable_observed_dataset_ids": list(missing_rankable),
            }
        )
    if not rankings:
        raise ValueError("validation_oracle_sweep produced no rankable validation rankings.")
    rankings.sort(key=lambda item: float(item["score"]), reverse=True)
    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx
    return rankings


def _write_validation_phase_artifacts(
    *,
    output_dir: Path,
    stage_results: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
    execution: Optional[Dict[str, object]],
    generation_id: Optional[str],
) -> None:
    _write_stage_outputs(
        VALIDATION_ORACLE_STAGE_NAME,
        stage_results,
        rankings,
        output_dir,
        execution=execution,
        generation_id=generation_id,
    )
    write_json_atomic(
        output_dir / "validation_leaderboard.json",
        list(rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    _write_leaderboard_csv(rankings, output_dir / "validation_leaderboard.csv")
    write_json_atomic(
        output_dir / "validation_best_configs.json",
        _best_configs_payload(rankings),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )
    metric_records = [
        cast(Dict[str, object], stage_result["metric_record"])
        for stage_result in stage_results
        if stage_result.get("status") == "ok" and "metric_record" in stage_result
    ]
    write_json_atomic(
        output_dir / "validation_summary.json",
        metric_records,
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def _metric_records_from_stage_results(
    stage_results: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    return [
        cast(Dict[str, object], stage_result["metric_record"])
        for stage_result in stage_results
        if stage_result.get("status") == "ok" and "metric_record" in stage_result
    ]


def _write_root_ranked_outputs(
    *,
    output_dir: Path,
    rankings: Sequence[Dict[str, object]],
    metric_records: Sequence[Dict[str, object]],
    analyze: bool,
) -> None:
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

    write_json_atomic(
        output_dir / "summary.json",
        list(metric_records),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    if analyze and metric_records:
        analysis = analyze_summary_data(metric_records)
        save_analysis(analysis, output_dir / "analysis.json")


def _remove_path_and_backup(path: Path) -> None:
    for candidate in (path, Path(f"{path}.bak")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue


def _clear_validation_oracle_root_artifacts(output_dir: Path) -> None:
    for filename in VALIDATION_ORACLE_GENERATED_ROOT_ARTIFACTS:
        _remove_path_and_backup(output_dir / filename)


def _validation_oracle_manifest_base(
    *,
    validation_oracle_dir: Path,
    validation_spec: ValidationOracleSweepSpec,
    validation_phase_budget: int,
) -> Dict[str, object]:
    return {
        "enabled": True,
        "output_dir": str(validation_oracle_dir.resolve()),
        "stage": VALIDATION_ORACLE_STAGE_NAME,
        "oracle_mode": validation_spec.oracle_mode,
        "score_metric": _validation_oracle_score_metric(validation_spec),
        "top_k": int(validation_spec.top_k),
        "budget_per_dataset": int(validation_phase_budget),
        "skip_image_metric_stage": bool(validation_spec.skip_image_metric_stage),
    }


def _write_validation_oracle_manifest(
    validation_oracle_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    write_json_atomic(
        validation_oracle_dir / "validation_oracle_manifest.json",
        dict(manifest),
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def _validation_oracle_failed_manifest(
    *,
    validation_oracle_dir: Path,
    validation_spec: ValidationOracleSweepSpec,
    validation_phase_budget: int,
    failed_stage: str,
    error: BaseException,
) -> Dict[str, object]:
    manifest = _validation_oracle_manifest_base(
        validation_oracle_dir=validation_oracle_dir,
        validation_spec=validation_spec,
        validation_phase_budget=validation_phase_budget,
    )
    manifest.update(
        {
            "status": "failed",
            "failed_stage": failed_stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    return manifest


def _run_validation_oracle_workflow(
    *,
    spec: SweepExperimentSpec,
    validation_spec: ValidationOracleSweepSpec,
    validation_phase_budget: int,
    stage1_results: Sequence[Dict[str, object]],
    coarse_lookup: Mapping[str, SweepCandidate],
    prepared_datasets: Sequence[PreparedDataset],
    dataset_ids: Sequence[str],
    output_dir: Path,
    requested_workers: int,
    run_state: Dict[str, object],
    run_state_path: Path,
    generation_id: Optional[str],
    merge_sources: Sequence[Path],
    merge_log: Dict[str, object],
) -> Dict[str, object]:
    validation_oracle_dir = output_dir / VALIDATION_ORACLE_DIR_NAME
    validation_oracle_dir.mkdir(parents=True, exist_ok=True)
    _clear_validation_oracle_root_artifacts(output_dir)

    failed_stage = "scene_validation"
    try:
        validation_scene_lookup = _validate_validation_oracle_scene_lookup(
            validation_spec,
            dataset_ids,
        )

        failed_stage = "stage1_benchmark"
        if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD:
            stage1_trio_import, stage1_trio_manifest = _run_human_bead_oracle_stage(
                scene_lookup=validation_scene_lookup,
                sweep_dir=output_dir,
                dataset_ids=dataset_ids,
                stage_name="stage1",
                stage_results=stage1_results,
            )
        else:
            stage1_trio_import, stage1_trio_manifest = _run_validation_oracle_benchmarks(
                validation_spec=validation_spec,
                scene_lookup=validation_scene_lookup,
                sweep_dir=output_dir,
                dataset_ids=dataset_ids,
                stage_name="stage1",
            )

        failed_stage = "stage1_ranking"
        oracle_stage1_rankings = _rank_stage_with_validation_oracle(
            stage_name="stage1",
            stage_results=stage1_results,
            validation_import=stage1_trio_import,
            dataset_ids=dataset_ids,
            scoring=spec.scoring,
            score_metric=_validation_oracle_score_metric(validation_spec),
        )
        write_json_atomic(
            validation_oracle_dir / "stage1_rankings.json",
            oracle_stage1_rankings,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )

        failed_stage = "validation_phase_candidates"
        validation_candidates = generate_refined_candidates(
            ranked_candidates=oracle_stage1_rankings,
            candidate_lookup=dict(coarse_lookup),
            search_spaces=spec.search_spaces,
            top_k=validation_spec.top_k,
            budget=validation_phase_budget,
            refinement=spec.stages.refinement,
        )
        if not validation_candidates and oracle_stage1_rankings:
            best_id = str(oracle_stage1_rankings[0]["candidate_id"])
            validation_candidates = [coarse_lookup[best_id]]
        validation_candidates = _assign_candidate_ids(
            _dedupe_candidates(validation_candidates),
            "v1c",
        )

        failed_stage = "validation_phase_eval"
        validation_eval = _evaluate_stage(
            stage_name=VALIDATION_ORACLE_STAGE_NAME,
            candidates=validation_candidates,
            datasets=prepared_datasets,
            output_dir=output_dir,
            save_figures_flag=spec.save_figures,
            requested_workers=requested_workers,
            scoring=spec.scoring,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
            dry_run=False,
            scheduling=spec.stages.scheduling,
        )
        validation_phase_results = validation_eval.stage_results

        failed_stage = "validation_phase_image_ranking"
        validation_image_rankings = score_and_rank_runs(
            validation_phase_results,
            spec.scoring,
            dataset_ids=dataset_ids,
        )
        _write_stage_outputs(
            VALIDATION_ORACLE_STAGE_NAME,
            validation_phase_results,
            validation_image_rankings,
            output_dir,
            execution=validation_eval.execution,
            generation_id=generation_id,
            create_backup=False,
        )

        failed_stage = "validation_phase_benchmark"
        if validation_spec.oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD:
            validation_trio_import, validation_trio_manifest = _run_human_bead_oracle_stage(
                scene_lookup=validation_scene_lookup,
                sweep_dir=output_dir,
                dataset_ids=dataset_ids,
                stage_name=VALIDATION_ORACLE_STAGE_NAME,
                stage_results=validation_phase_results,
            )
        else:
            validation_trio_import, validation_trio_manifest = _run_validation_oracle_benchmarks(
                validation_spec=validation_spec,
                scene_lookup=validation_scene_lookup,
                sweep_dir=output_dir,
                dataset_ids=dataset_ids,
                stage_name=VALIDATION_ORACLE_STAGE_NAME,
            )

        failed_stage = "validation_phase_ranking"
        validation_phase_rankings = _rank_stage_with_validation_oracle(
            stage_name=VALIDATION_ORACLE_STAGE_NAME,
            stage_results=validation_phase_results,
            validation_import=validation_trio_import,
            dataset_ids=dataset_ids,
            scoring=spec.scoring,
            score_metric=_validation_oracle_score_metric(validation_spec),
        )
        _write_validation_phase_artifacts(
            output_dir=output_dir,
            stage_results=validation_phase_results,
            rankings=validation_phase_rankings,
            execution=validation_eval.execution,
            generation_id=generation_id,
        )
        write_json_atomic(
            validation_oracle_dir / "validation_phase_rankings.json",
            validation_phase_rankings,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )
        validation_oracle_manifest = _validation_oracle_manifest_base(
            validation_oracle_dir=validation_oracle_dir,
            validation_spec=validation_spec,
            validation_phase_budget=validation_phase_budget,
        )
        validation_oracle_manifest.update(
            {
                "status": "completed",
                "num_candidates": len(validation_candidates),
                "num_runs": len(validation_phase_results),
                "num_ranked": len(validation_phase_rankings),
                "stage1": stage1_trio_manifest,
                "validation_phase": validation_trio_manifest,
                "validation_phase_results": list(validation_phase_results),
                "validation_phase_rankings": list(validation_phase_rankings),
                "validation_phase_execution": validation_eval.execution,
            }
        )
    except Exception as exc:  # noqa: BLE001 - validation oracle is advisory.
        _clear_validation_oracle_root_artifacts(output_dir)
        validation_oracle_manifest = _validation_oracle_failed_manifest(
            validation_oracle_dir=validation_oracle_dir,
            validation_spec=validation_spec,
            validation_phase_budget=validation_phase_budget,
            failed_stage=failed_stage,
            error=exc,
        )

    _write_validation_oracle_manifest(validation_oracle_dir, validation_oracle_manifest)
    return validation_oracle_manifest


def _run_independent_metric_search_phase(
    *,
    spec: SweepExperimentSpec,
    experimental_metric_spec: ExperimentalMetricSweepSpec,
    prepared_datasets: Sequence[PreparedDataset],
    dataset_ids: Sequence[str],
    metric_output_dir: Path,
    requested_workers: int,
    run_state: Dict[str, object],
    run_state_path: Path,
    generation_id: Optional[str],
    merge_sources: Sequence[Path],
    merge_log: Dict[str, object],
) -> Dict[str, object]:
    search_stages = _metric_search_stages(experimental_metric_spec)
    metric_requested_workers = _effective_metric_search_workers(requested_workers, search_stages)
    stage1_budget, stage2_budget = _independent_metric_stage_budgets(search_stages)
    metric_stage1_candidates: List[SweepCandidate] = []
    metric_stage1_execution: Optional[Dict[str, object]] = None

    if search_stages.mode == "single_stage":
        metric_candidates = _assign_candidate_ids(
            _collect_independent_metric_stage_candidates(
                spec,
                experimental_metric_spec,
                budget_per_variant=stage1_budget,
            ),
            "m1c",
        )
    else:
        metric_stage1_candidates = _assign_candidate_ids(
            _collect_independent_metric_stage_candidates(
                spec,
                experimental_metric_spec,
                budget_per_variant=stage1_budget,
            ),
            "m1s1c",
        )
        metric_stage1_lookup = {
            candidate.candidate_id: candidate for candidate in metric_stage1_candidates
        }
        metric_stage1_eval = _evaluate_stage(
            stage_name=METRIC_SEARCH_STAGE1_NAME,
            candidates=metric_stage1_candidates,
            datasets=prepared_datasets,
            output_dir=metric_output_dir,
            save_figures_flag=spec.save_figures,
            requested_workers=metric_requested_workers,
            scoring=spec.scoring,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
            dry_run=False,
            scheduling=search_stages.scheduling,
        )
        metric_stage1_execution = metric_stage1_eval.execution
        metric_stage1_rankings = score_and_rank_runs(
            metric_stage1_eval.stage_results,
            spec.scoring,
            dataset_ids=dataset_ids,
        )
        _write_stage_outputs(
            METRIC_SEARCH_STAGE1_NAME,
            metric_stage1_eval.stage_results,
            metric_stage1_rankings,
            metric_output_dir,
            execution=metric_stage1_eval.execution,
            generation_id=generation_id,
        )
        metric_candidates = _assign_candidate_ids(
            _collect_independent_metric_refined_candidates(
                metric_stage1_rankings=metric_stage1_rankings,
                metric_stage1_lookup=metric_stage1_lookup,
                search_spaces=spec.search_spaces,
                stage_spec=search_stages,
                budget_per_variant=stage2_budget,
            ),
            "m1c",
        )

    metric_eval = _evaluate_stage(
        stage_name=METRIC_PHASE_STAGE_NAME,
        candidates=metric_candidates,
        datasets=prepared_datasets,
        output_dir=metric_output_dir,
        save_figures_flag=spec.save_figures,
        requested_workers=metric_requested_workers,
        scoring=spec.scoring,
        run_state=run_state,
        run_state_path=run_state_path,
        generation_id=generation_id,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=False,
        scheduling=search_stages.scheduling,
    )
    metric_results = metric_eval.stage_results
    metric_rankings = score_and_rank_runs(
        metric_results,
        spec.scoring,
        dataset_ids=dataset_ids,
    )
    _write_phase_summary_artifacts(
        output_dir=metric_output_dir,
        stage_name=METRIC_PHASE_STAGE_NAME,
        stage_results=metric_results,
        rankings=metric_rankings,
        execution=metric_eval.execution,
        reporting=spec.reporting,
        analyze=spec.analyze,
        generation_id=generation_id,
    )
    return _metric_experiment_plan_payload(
        experimental_spec=experimental_metric_spec,
        candidates=metric_candidates,
        dataset_count=len(prepared_datasets),
        status="completed",
        output_dir=metric_output_dir,
        stage1_candidates=metric_stage1_candidates,
        stage1_execution=metric_stage1_execution,
        final_execution=metric_eval.execution,
        num_ranked=len(metric_rankings),
        num_results=len(metric_results),
    )


def _copy_product_if_exists(source: Path, destination: Path) -> None:
    if not source.exists() and not Path(f"{source}.bak").exists():
        return
    source_to_copy = source if source.exists() else Path(f"{source}.bak")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_to_copy, destination)


def _snapshot_run_display_artifacts(source_runs_dir: Path, destination_runs_dir: Path) -> None:
    if not source_runs_dir.exists():
        return
    for source_run_dir in sorted(source_runs_dir.glob("*/*/*")):
        if not source_run_dir.is_dir():
            continue
        relative_run_dir = source_run_dir.relative_to(source_runs_dir)
        destination_run_dir = destination_runs_dir / relative_run_dir
        for filename in ("avg_rectified.png", "workspace.npz", "metric_record.json"):
            _copy_product_if_exists(source_run_dir / filename, destination_run_dir / filename)


def _snapshot_generation_products(output_dir: Path, generation_id: str) -> Path:
    products_dir = _generation_dir(output_dir, generation_id) / "products"
    for filename in (
        "stage1_summary.json",
        "stage2_summary.json",
        "validation_phase_summary.json",
        "validation_leaderboard.json",
        "validation_leaderboard.csv",
        "validation_best_configs.json",
        "validation_summary.json",
        "leaderboard.json",
        "leaderboard.csv",
        "best_configs.json",
        "summary.json",
        "analysis.json",
        "analysis.csv",
        "experiment_manifest.json",
    ):
        _copy_product_if_exists(output_dir / filename, products_dir / filename)
    _snapshot_run_display_artifacts(output_dir / "runs", products_dir / "runs")

    metric_dir = output_dir / "metric_experiments"
    if metric_dir.exists():
        metric_products_dir = products_dir / "metric_experiments"
        for filename in (
            "metric_phase_summary.json",
            "leaderboard.json",
            "leaderboard.csv",
            "best_configs.json",
            "summary.json",
            "analysis.json",
            "analysis.csv",
            "alpha_report.json",
            "alpha_report.csv",
            "alpha_report_warnings.json",
            "alpha_report_primary_vs_alpha.png",
            "alpha_report_displacement_vs_alpha.png",
        ):
            _copy_product_if_exists(metric_dir / filename, metric_products_dir / filename)
        _snapshot_run_display_artifacts(metric_dir / "runs", metric_products_dir / "runs")
    validation_dir = output_dir / VALIDATION_ORACLE_DIR_NAME
    if validation_dir.exists():
        validation_products_dir = products_dir / VALIDATION_ORACLE_DIR_NAME
        for source_path in sorted(validation_dir.rglob("*")):
            if source_path.is_file():
                destination = validation_products_dir / source_path.relative_to(validation_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
    return products_dir


def _archive_validation_skip_stage2_artifacts(output_dir: Path, generation_id: str) -> None:
    archive_root = _generation_dir(output_dir, generation_id) / "inactive_artifacts" / "stage2"
    moved: List[Dict[str, str]] = []

    def move_if_exists(source: Path, relative_destination: Path) -> None:
        if not source.exists():
            return
        destination = archive_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(
                f"{destination.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            )
        shutil.move(str(source), str(destination))
        moved.append(
            {
                "source": str(source),
                "destination": str(destination),
            }
        )

    move_if_exists(output_dir / "stage2_summary.json", Path("stage2_summary.json"))
    move_if_exists(output_dir / "stage2_summary.json.bak", Path("stage2_summary.json.bak"))
    move_if_exists(output_dir / "runs" / "stage2", Path("runs") / "stage2")

    if not moved:
        return
    write_json_atomic(
        archive_root / "manifest.json",
        {
            "generated_at": _utc_now_iso(),
            "reason": "validation_oracle_skip_image_metric_stage",
            "generation_id": generation_id,
            "moved": moved,
        },
        indent=2,
        sort_keys=False,
        create_backup=True,
    )


def run_sweep_experiment(
    spec: SweepExperimentSpec,
    output_dir: Optional[Path] = None,
    *,
    extend_existing: bool = False,
    merge_from: Optional[Sequence[Path]] = None,
    adopt_relocated: bool = False,
    relocation_rules: Sequence[PathRelocationRule] = (),
    relocation_report: Optional[Mapping[str, object]] = None,
    dry_run: bool = False,
) -> Path:
    """Run a staged or single-stage, multi-dataset sweep experiment."""

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

    dataset_identities = _read_dataset_identity_payloads(resolved_output_dir)
    prepared_datasets = _prepare_datasets(
        spec,
        resolved_output_dir,
        dataset_identities=dataset_identities,
    )
    dataset_ids = [dataset.dataset_id for dataset in prepared_datasets]
    validation_spec = _validation_oracle_spec(spec)
    validation_phase_budget = _validation_oracle_phase_budget(
        validation_spec,
        _stage_budgets(spec.stages)[1],
    )
    fingerprint_payload = _build_experiment_fingerprint_payload(
        spec=spec,
        prepared_datasets=prepared_datasets,
        output_dir=resolved_output_dir,
    )
    merge_sources = [Path(path) for path in (merge_from or [])]
    run_state, run_state_path, generation_id = _load_or_initialize_run_state(
        output_dir=resolved_output_dir,
        fingerprint_payload=fingerprint_payload,
        allow_extend=extend_existing,
        allow_merge=bool(merge_sources),
        adopt_relocated=bool(adopt_relocated),
        relocation_rules=relocation_rules,
        relocation_report=relocation_report,
        merge_sources=merge_sources,
        dry_run=dry_run,
    )
    generation_root = _generation_dir(resolved_output_dir, generation_id)
    generation_root.mkdir(parents=True, exist_ok=True)
    merge_log: Dict[str, object] = {
        "generation_id": generation_id,
        "created_at": _utc_now_iso(),
        "canonical_output_dir": str(resolved_output_dir.resolve()),
        "merge_sources": [str(path.resolve()) for path in merge_sources],
        "sources": [],
        "copied": [],
        "would_copy": [],
        "skipped_duplicates": [],
        "rejected": [],
        "conflicts": [],
    }

    stage_mode = spec.stages.mode
    stage1_budget, stage2_budget = _stage_budgets(spec.stages)
    skip_image_metric_stage = _skip_image_metric_stage_for_validation(validation_spec)
    if skip_image_metric_stage and not dry_run:
        _archive_validation_skip_stage2_artifacts(resolved_output_dir, generation_id)
    baseline_budget = (
        stage1_budget
        if stage_mode == "two_stage"
        else max(1, int(spec.stages.total_budget_per_dataset))
    )

    coarse_candidates_base: List[SweepCandidate] = []
    for strategy in spec.strategies:
        search_space = spec.search_spaces[strategy]
        coarse_candidates_base.extend(generate_coarse_candidates(search_space, budget=None))
    coarse_candidates = list(coarse_candidates_base)
    coarse_candidates = _dedupe_candidates(coarse_candidates)
    coarse_candidates = _limit_candidates(coarse_candidates, baseline_budget)
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
        scoring=spec.scoring,
        run_state=run_state,
        run_state_path=run_state_path,
        generation_id=generation_id,
        merge_sources=merge_sources,
        merge_log=merge_log,
        dry_run=dry_run,
        scheduling=spec.stages.scheduling,
    )
    stage1_results = stage1_eval.stage_results
    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    if not dry_run:
        _write_stage_outputs(
            "stage1",
            stage1_results,
            stage1_rankings,
            resolved_output_dir,
            execution=stage1_eval.execution,
            generation_id=generation_id,
        )

    validation_oracle_manifest: Dict[str, object] = {
        "enabled": bool(validation_spec.enabled),
        "output_dir": None,
        "status": "disabled",
    }

    refined_lookup: Dict[str, SweepCandidate] = {}
    stage2_results: List[Dict[str, object]] = []
    stage2_rankings: List[Dict[str, object]] = []
    if stage_mode == "two_stage" and not skip_image_metric_stage:
        refined_candidates = generate_refined_candidates(
            ranked_candidates=stage1_rankings,
            candidate_lookup=coarse_lookup,
            search_spaces=spec.search_spaces,
            top_k=spec.stages.top_k,
            budget=stage2_budget,
            refinement=spec.stages.refinement,
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
            scoring=spec.scoring,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
            dry_run=dry_run,
            scheduling=spec.stages.scheduling,
        )
        stage2_results = stage2_eval.stage_results
        stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)
        if not dry_run:
            _write_stage_outputs(
                "stage2",
                stage2_results,
                stage2_rankings,
                resolved_output_dir,
                execution=stage2_eval.execution,
                generation_id=generation_id,
            )

    if dry_run:
        experimental_metric_spec = (
            spec.experimental_metric_sweep
            if spec.experimental_metric_sweep is not None
            else ExperimentalMetricSweepSpec(
                enabled=False,
                shortlist_top_k=spec.stages.top_k,
                formulations=[],
            )
        )
        if not experimental_metric_spec.enabled:
            metric_dry_run_payload = _disabled_metric_preview_payload(experimental_metric_spec)
        else:
            metric_output_dir = resolved_output_dir / "metric_experiments"
            if (
                experimental_metric_spec.candidate_source
                == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
            ):
                metric_dry_run_payload = _preview_independent_metric_search(
                    spec=spec,
                    experimental_spec=experimental_metric_spec,
                    prepared_datasets=prepared_datasets,
                    metric_output_dir=metric_output_dir,
                    requested_workers=requested_workers,
                    run_state=run_state,
                    run_state_path=run_state_path,
                    generation_id=generation_id,
                    merge_sources=merge_sources,
                    merge_log=merge_log,
                )
            else:
                baseline_candidate_lookup = dict(coarse_lookup)
                baseline_candidate_lookup.update(refined_lookup)
                metric_dry_run_payload = _preview_baseline_shortlist_metric_phase(
                    spec=spec,
                    experimental_spec=experimental_metric_spec,
                    prepared_datasets=prepared_datasets,
                    final_rankings=stage2_rankings if stage2_rankings else stage1_rankings,
                    baseline_candidate_lookup=baseline_candidate_lookup,
                    metric_output_dir=metric_output_dir,
                    requested_workers=requested_workers,
                    run_state=run_state,
                    run_state_path=run_state_path,
                    generation_id=generation_id,
                    merge_sources=merge_sources,
                    merge_log=merge_log,
                )
        validation_dry_run_payload = _validation_oracle_dry_run_payload(
            validation_spec=validation_spec,
            dataset_ids=dataset_ids,
            validation_phase_budget=validation_phase_budget,
        )
        stages_payload: Dict[str, object] = {
            "stage1": _stage_preview_payload(stage1_eval.execution),
            **(
                {"stage2": _stage_preview_payload(stage2_eval.execution)}
                if stage_mode == "two_stage" and "stage2_eval" in locals()
                else {}
            ),
        }
        stage_planned = sum(
            int(cast(Mapping[str, object], payload).get("planned_run_count", 0) or 0)
            for payload in stages_payload.values()
            if isinstance(payload, dict)
        )
        stage_pending = sum(
            int(cast(Mapping[str, object], payload).get("pending_run_count", 0) or 0)
            for payload in stages_payload.values()
            if isinstance(payload, dict)
        )
        stage_skipped = sum(
            int(cast(Mapping[str, object], payload).get("skipped_valid_count", 0) or 0)
            for payload in stages_payload.values()
            if isinstance(payload, dict)
        )
        dry_run_payload = {
            "generation_id": generation_id,
            "generated_at": _utc_now_iso(),
            "canonical_output_dir": str(resolved_output_dir.resolve()),
            "extend_existing": bool(extend_existing),
            "merge_sources": [str(path.resolve()) for path in merge_sources],
            "relocation": relocation_report
            or {
                "rules": _relocation_rules_to_payload(relocation_rules),
                "applied": [],
            },
            "stages": stages_payload,
            "metric_experiments": metric_dry_run_payload,
            "validation_oracle_sweep": validation_dry_run_payload,
            "totals": {
                "planned_run_count": int(
                    stage_planned
                    + _metric_planned_count(metric_dry_run_payload)
                    + _validation_oracle_planned_count(validation_dry_run_payload)
                ),
                "pending_run_count": int(
                    stage_pending
                    + _metric_pending_count(metric_dry_run_payload)
                    + _validation_oracle_planned_count(validation_dry_run_payload)
                ),
                "skipped_valid_count": int(stage_skipped + _metric_skipped_valid_count(metric_dry_run_payload)),
                "exact": bool(
                    all(
                        bool(cast(Mapping[str, object], payload).get("exact", False))
                        for payload in stages_payload.values()
                        if isinstance(payload, dict)
                    )
                    and bool(metric_dry_run_payload.get("exact", False))
                    and bool(validation_dry_run_payload.get("exact", False))
                ),
            },
            "merge_log": merge_log,
        }
        write_json_atomic(
            generation_root / "dry_run.json",
            dry_run_payload,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )
        print(f"Dry run written to {generation_root / 'dry_run.json'}")
        return resolved_output_dir

    final_rankings = stage2_rankings if stage2_rankings else stage1_rankings
    all_metric_records: List[Dict[str, object]] = _metric_records_from_stage_results(
        stage1_results + stage2_results
    )
    _write_root_ranked_outputs(
        output_dir=resolved_output_dir,
        rankings=final_rankings,
        metric_records=all_metric_records,
        analyze=spec.analyze,
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
        if (
            experimental_metric_spec.candidate_source
            == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        ):
            metric_phase_manifest = _run_independent_metric_search_phase(
                spec=spec,
                experimental_metric_spec=experimental_metric_spec,
                prepared_datasets=prepared_datasets,
                dataset_ids=dataset_ids,
                metric_output_dir=metric_output_dir,
                requested_workers=requested_workers,
                run_state=run_state,
                run_state_path=run_state_path,
                generation_id=generation_id,
                merge_sources=merge_sources,
                merge_log=merge_log,
            )
        else:
            baseline_candidate_lookup = dict(coarse_lookup)
            baseline_candidate_lookup.update(refined_lookup)
            metric_candidates = _collect_metric_experiment_candidates(
                final_rankings=final_rankings,
                baseline_candidate_lookup=baseline_candidate_lookup,
                experimental_spec=experimental_metric_spec,
            )
            metric_candidates = _assign_candidate_ids(metric_candidates, "m1c")

            metric_eval = _evaluate_stage(
                stage_name=METRIC_PHASE_STAGE_NAME,
                candidates=metric_candidates,
                datasets=prepared_datasets,
                output_dir=metric_output_dir,
                save_figures_flag=spec.save_figures,
                requested_workers=requested_workers,
                scoring=spec.scoring,
                run_state=run_state,
                run_state_path=run_state_path,
                generation_id=generation_id,
                merge_sources=merge_sources,
                merge_log=merge_log,
                dry_run=False,
                scheduling=spec.stages.scheduling,
            )
            metric_results = metric_eval.stage_results
            metric_rankings = score_and_rank_runs(
                metric_results, spec.scoring, dataset_ids=dataset_ids
            )
            _write_phase_summary_artifacts(
                output_dir=metric_output_dir,
                stage_name=METRIC_PHASE_STAGE_NAME,
                stage_results=metric_results,
                rankings=metric_rankings,
                execution=metric_eval.execution,
                reporting=spec.reporting,
                analyze=spec.analyze,
                generation_id=generation_id,
            )
            metric_phase_manifest = _metric_experiment_plan_payload(
                experimental_spec=experimental_metric_spec,
                candidates=metric_candidates,
                dataset_count=len(prepared_datasets),
                status="completed",
                output_dir=metric_output_dir,
                final_execution=metric_eval.execution,
                num_ranked=len(metric_rankings),
                num_results=len(metric_results),
            )

    if validation_spec.enabled:
        validation_oracle_manifest = _run_validation_oracle_workflow(
            spec=spec,
            validation_spec=validation_spec,
            validation_phase_budget=validation_phase_budget,
            stage1_results=stage1_results,
            coarse_lookup=coarse_lookup,
            prepared_datasets=prepared_datasets,
            dataset_ids=dataset_ids,
            output_dir=resolved_output_dir,
            requested_workers=requested_workers,
            run_state=run_state,
            run_state_path=run_state_path,
            generation_id=generation_id,
            merge_sources=merge_sources,
            merge_log=merge_log,
        )
        if (
            skip_image_metric_stage
            and validation_oracle_manifest.get("status") == "completed"
        ):
            validation_rankings = [
                cast(Dict[str, object], item)
                for item in cast(
                    Sequence[object],
                    validation_oracle_manifest.get("validation_phase_rankings", []),
                )
                if isinstance(item, dict)
            ]
            validation_results = [
                cast(Dict[str, object], item)
                for item in cast(
                    Sequence[object],
                    validation_oracle_manifest.get("validation_phase_results", []),
                )
                if isinstance(item, dict)
            ]
            validation_metric_records = _metric_records_from_stage_results(validation_results)
            _write_root_ranked_outputs(
                output_dir=resolved_output_dir,
                rankings=validation_rankings,
                metric_records=validation_metric_records,
                analyze=spec.analyze,
            )

    write_json_atomic(
        resolved_output_dir / "experiment_manifest.json",
        {
            "generated_at": _utc_now_iso(),
            "output_dir": str(resolved_output_dir.resolve()),
            "generation_id": generation_id,
            "strategies": list(spec.strategies),
            "baseline_displacement_mode": {
                "family": RectificationRunConfig().displacement_metric_family,
                "alpha": float(RectificationRunConfig().displacement_alpha),
                "metric_formulation_id": RectificationRunConfig().metric_formulation_id,
            },
            "experimental_metric_sweep": (
                None
                if spec.experimental_metric_sweep is None
                else asdict(spec.experimental_metric_sweep)
            ),
            "metric_experiments": metric_phase_manifest,
            "validation_oracle_sweep": validation_oracle_manifest,
            "datasets": [_dataset_identity_payload(dataset) for dataset in prepared_datasets],
            "stages": asdict(spec.stages),
            "scoring": asdict(spec.scoring),
            "reporting": asdict(spec.reporting),
            "run_state_path": str(run_state_path),
            "generation_products_dir": str((_generation_dir(resolved_output_dir, generation_id) / "products").resolve()),
        },
        indent=2,
        sort_keys=False,
        create_backup=True,
    )

    if merge_sources:
        write_json_atomic(
            generation_root / "merge_log.json",
            merge_log,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )
        write_json_atomic(
            generation_root / "products" / "merge_log.json",
            merge_log,
            indent=2,
            sort_keys=False,
            create_backup=True,
        )
        merge_history = cast(List[Dict[str, object]], run_state.setdefault("merge_history", []))
        merge_history.append(
            {
                "generation_id": generation_id,
                "timestamp": _utc_now_iso(),
                "merge_sources": [str(path.resolve()) for path in merge_sources],
                "copied": len(cast(List[object], merge_log.get("copied", []))),
                "skipped_duplicates": len(cast(List[object], merge_log.get("skipped_duplicates", []))),
                "rejected": len(cast(List[object], merge_log.get("rejected", []))),
                "conflicts": len(cast(List[object], merge_log.get("conflicts", []))),
            }
        )

    _snapshot_generation_products(resolved_output_dir, generation_id)
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
    combined_reference_mode: str | None = None,
    normalize_to_8bit: bool = False,
    pixel_shift_powers: Sequence[float] | None = None,
) -> None:
    default_config = RectificationRunConfig()
    global_thresholds = list(global_thresholds or [default_config.global_threshold])
    pixel_thresholds = list(pixel_thresholds or [default_config.pixel_threshold])
    pixel_shift_powers = list(pixel_shift_powers or [default_config.pixel_shift_power])

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
            (win, max_disp, gauss, disp_gauss, pixel_thresh, pixel_shift_power)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for pixel_thresh in pixel_thresholds
            for pixel_shift_power in pixel_shift_powers
        ]
    else:
        combos = [
            (win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh, pixel_shift_power)
            for win in window_sizes
            for max_disp in max_displacements
            for gauss in gauss_sigmas
            for disp_gauss in disp_gauss_sigmas
            for global_thresh in global_thresholds
            for pixel_thresh in pixel_thresholds
            for pixel_shift_power in pixel_shift_powers
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []
    plotter: LiveMetricsPlotter | None = None

    for combo in tqdm(combos, desc="Sweeping parameters", unit="combo"):
        if strategy == "global":
            win, max_disp, gauss, disp_gauss, global_thresh = combo  # type: ignore[misc]
            pixel_thresh = default_config.pixel_threshold
            pixel_shift_power = default_config.pixel_shift_power
        elif strategy == "pixel":
            win, max_disp, gauss, disp_gauss, pixel_thresh, pixel_shift_power = combo  # type: ignore[misc]
            global_thresh = default_config.global_threshold
        else:
            win, max_disp, gauss, disp_gauss, global_thresh, pixel_thresh, pixel_shift_power = combo  # type: ignore[misc]
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
            combined_reference_mode=combined_reference_mode,
            normalize_to_8bit=normalize_to_8bit,
            pixel_shift_power=float(pixel_shift_power),
        )

        result = rectify_views_from_config(views, config)
        run_name_parts = [
            f"win{win}",
            f"disp{max_disp}",
            f"g{gauss}",
            f"dg{disp_gauss}",
        ]
        if (
            strategy == "combined"
            and config.combined_reference_mode != COMBINED_REFERENCE_MODE_GLOBAL_AVERAGE
        ):
            run_name_parts.append(f"ref_{config.combined_reference_mode}")
        if strategy in {"global", "combined"}:
            run_name_parts.append(f"gt{global_thresh}")
        if strategy in {"pixel", "combined"}:
            run_name_parts.append(f"pt{pixel_thresh}")
            run_name_parts.append(f"psp{config.pixel_shift_power}")
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
            "Path to a JSON experiment spec for the staged or single-stage "
            "multi-dataset sweep (recommended path for new workflows)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override parallel worker count in --spec mode.",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=sorted(SCHEDULER_MODES),
        default=None,
        help="Override stages.scheduling.mode in --spec mode.",
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=None,
        help="Override stages.scheduling.memory_safety_fraction in --spec mode.",
    )
    parser.add_argument(
        "--worker-recycle-tasks",
        type=int,
        default=None,
        help="Recycle process-pool workers after this many tasks in --spec mode.",
    )
    parser.add_argument(
        "--no-memory-telemetry",
        action="store_false",
        dest="memory_telemetry",
        default=None,
        help="Disable psutil memory telemetry in --spec mode.",
    )
    parser.add_argument(
        "--memory-enforcement",
        choices=sorted(MEMORY_ENFORCEMENT_MODES),
        default=None,
        help="Override stages.scheduling.memory_enforcement in --spec mode.",
    )
    parser.add_argument(
        "--metrics-config",
        help="Metric explorer config file containing shared scoring profiles.",
    )
    parser.add_argument(
        "--scoring-profile",
        help="Shared metric profile name to use for sweep ranking.",
    )
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help=(
            "Allow a changed spec to extend an existing --output-dir when dataset "
            "identities are compatible. Creates a new generation and reuses valid runs."
        ),
    )
    parser.add_argument(
        "--merge-from",
        nargs="+",
        default=None,
        help=(
            "One or more completed sweep roots to merge into --output-dir before "
            "running missing compatible spec-mode tasks."
        ),
    )
    parser.add_argument(
        "--relocate-path",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help=(
            "Relocate spec paths from OLD root to NEW root before running. "
            "May be provided multiple times; spec mode only."
        ),
    )
    parser.add_argument(
        "--adopt-relocated",
        action="store_true",
        help=(
            "Allow a copied sweep root to update run_state.json path metadata "
            "when the normalized experiment fingerprint is compatible."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "In spec mode, write generations/<id>/dry_run.json with planned reused, "
            "merged, rejected, and pending runs without executing rectification."
        ),
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
    parser.add_argument(
        "--pixel-shift-power-start",
        type=float,
        default=1.0,
        help="Shifted moving-view power sweep start for pixel/combined intensity scoring",
    )
    parser.add_argument("--pixel-shift-power-stop", type=float, default=1.0)
    parser.add_argument("--pixel-shift-power-step", type=float, default=1.0)
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
        "--combined-reference-mode",
        choices=[
            "global_average",
            "center_view",
            "iterative_subtractive_max_rebuild",
        ],
        default=None,
        help=(
            "Override the combined-stage pixel reference source. When omitted, "
            "--use-global-avg-reference preserves the legacy global-average vs "
            "center-view behavior."
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
        relocation_rules = _parse_path_relocation_rules(
            cast(Optional[Sequence[str]], getattr(args, "relocate_path", None))
        )
        spec, relocation_report = _parse_experiment_spec(
            Path(spec_path),
            relocation_rules=relocation_rules,
        )
        if args.output_dir is not None:
            spec.output_dir = args.output_dir
        workers_override = getattr(args, "workers", None)
        if workers_override is not None:
            spec.stages.max_workers = int(workers_override)
        scheduler_mode_override = getattr(args, "scheduler_mode", None)
        memory_fraction_override = getattr(args, "memory_safety_fraction", None)
        worker_recycle_override = getattr(args, "worker_recycle_tasks", None)
        memory_telemetry_override = getattr(args, "memory_telemetry", None)
        memory_enforcement_override = getattr(args, "memory_enforcement", None)
        if any(
            item is not None
            for item in (
                scheduler_mode_override,
                memory_fraction_override,
                worker_recycle_override,
                memory_telemetry_override,
                memory_enforcement_override,
            )
        ):
            overridden_scheduling = SweepSchedulingSpec(
                mode=cast(
                    Literal["adaptive_memory", "fixed_batch"],
                    scheduler_mode_override or spec.stages.scheduling.mode,
                ),
                memory_safety_fraction=(
                    float(memory_fraction_override)
                    if memory_fraction_override is not None
                    else spec.stages.scheduling.memory_safety_fraction
                ),
                memory_telemetry=(
                    bool(memory_telemetry_override)
                    if memory_telemetry_override is not None
                    else spec.stages.scheduling.memory_telemetry
                ),
                worker_recycle_tasks=(
                    int(worker_recycle_override)
                    if worker_recycle_override is not None
                    else spec.stages.scheduling.worker_recycle_tasks
                ),
                memory_enforcement=cast(
                    Literal["auto", "required", "advisory"],
                    memory_enforcement_override
                    or spec.stages.scheduling.memory_enforcement,
                ),
            )
            spec.stages.scheduling = overridden_scheduling
            if (
                spec.experimental_metric_sweep is not None
                and spec.experimental_metric_sweep.search_stages is not None
            ):
                spec.experimental_metric_sweep.search_stages.scheduling = SweepSchedulingSpec(
                    mode=overridden_scheduling.mode,
                    memory_safety_fraction=overridden_scheduling.memory_safety_fraction,
                    memory_telemetry=overridden_scheduling.memory_telemetry,
                    worker_recycle_tasks=overridden_scheduling.worker_recycle_tasks,
                    memory_enforcement=overridden_scheduling.memory_enforcement,
                )
        metrics_config_override = getattr(args, "metrics_config", None)
        scoring_profile_override = getattr(args, "scoring_profile", None)
        if metrics_config_override is not None:
            spec.scoring.profile_config_path = str(Path(metrics_config_override).resolve())
        if scoring_profile_override is not None:
            spec.scoring.profile_name = str(scoring_profile_override)
        if scoring_profile_override is not None or (
            metrics_config_override is not None and spec.scoring.profile_name
        ):
            if not spec.scoring.profile_name:
                raise ValueError("--metrics-config requires --scoring-profile unless the spec already sets scoring.profile_name.")
            weights, _, _ = resolve_profile_weights_for_sweep(
                spec.scoring.profile_name,
                config_path=Path(spec.scoring.profile_config_path) if spec.scoring.profile_config_path else None,
                scoring_weights=DEFAULT_SCORE_WEIGHTS,
            )
            spec.scoring.metric_weights = weights
        return run_sweep_experiment(
            spec,
            extend_existing=bool(getattr(args, "extend_existing", False)),
            merge_from=[
                Path(item) for item in (getattr(args, "merge_from", None) or [])
            ],
            adopt_relocated=bool(getattr(args, "adopt_relocated", False)),
            relocation_rules=relocation_rules,
            relocation_report=relocation_report,
            dry_run=bool(getattr(args, "dry_run", False)),
        )

    if (
        bool(getattr(args, "extend_existing", False))
        or bool(getattr(args, "dry_run", False))
        or bool(getattr(args, "merge_from", None))
        or bool(getattr(args, "relocate_path", None))
        or bool(getattr(args, "adopt_relocated", False))
        or getattr(args, "scheduler_mode", None) is not None
        or getattr(args, "memory_safety_fraction", None) is not None
        or getattr(args, "worker_recycle_tasks", None) is not None
        or getattr(args, "memory_telemetry", None) is not None
        or getattr(args, "memory_enforcement", None) is not None
    ):
        raise SystemExit(
            "--extend-existing, --merge-from, --relocate-path, --adopt-relocated, "
            "--dry-run, and scheduler overrides are supported only with --spec."
        )

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
    pixel_shift_powers = _build_float_range(
        float(getattr(args, "pixel_shift_power_start", 1.0)),
        float(getattr(args, "pixel_shift_power_stop", 1.0)),
        float(getattr(args, "pixel_shift_power_step", 1.0)),
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
        combined_reference_mode=args.combined_reference_mode,
        normalize_to_8bit=args.normalize_to_8bit,
        pixel_shift_powers=pixel_shift_powers,
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
    validation_workspace_path: str = ""
    validation_image_path: str = ""
    thorcam_image_path: str = ""
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
    formulation_id: str = "intensity_default"
    family: str = DEFAULT_DISPLACEMENT_METRIC_FAMILY
    alphas: str = "1.0"
    gradient_weight: str = "0.5"
    laplacian_weight: str = "0.5"


@dataclass
class SpecGuiRelocationRuleState:
    old_root: str = ""
    new_root: str = ""


@dataclass
class SpecGuiState:
    output_dir: str = ""
    run_mode: str = "normal"
    dry_run: bool = False
    adopt_relocated: bool = False
    relocation_rules: List[SpecGuiRelocationRuleState] = field(default_factory=list)
    strategy_pixel: bool = True
    strategy_combined: bool = True
    strategy_global: bool = False
    stage_mode: str = "two_stage"
    total_budget_per_dataset: str = "320"
    stage1_fraction: str = "0.6"
    top_k: str = "8"
    max_workers: str = ""
    scheduler_mode: str = SCHEDULER_MODE_ADAPTIVE_MEMORY
    memory_safety_fraction: str = str(MEMORY_SAFETY_FRACTION)
    memory_telemetry: bool = True
    worker_recycle_tasks: str = ""
    memory_enforcement: str = MEMORY_ENFORCEMENT_AUTO
    refinement_method: str = "auto_fine"
    refinement_extrapolation: str = "half_step_bounded"
    analyze: bool = True
    save_figures: bool = False
    pixel_similarity_window_sizes: str = "15,25,35"
    pixel_max_displacements: str = "5,15"
    pixel_gauss_sigmas: str = "3.0,5.0"
    pixel_disp_gauss_sigmas: str = "0.0,1.0"
    pixel_thresholds: str = "0.05"
    pixel_shift_powers: str = "1.0"
    combined_refinement_window_sizes: str = "15,25"
    combined_max_displacements: str = "5,15"
    combined_gauss_sigmas: str = "3.0,5.0"
    combined_disp_gauss_sigmas: str = "0.0,1.0"
    combined_pixel_thresholds: str = "0.05"
    combined_pixel_shift_powers: str = "1.0"
    combined_power_blend_enabled: str = "false"
    combined_power_blend_raised_powers: str = "2.0"
    combined_power_blend_alphas: str = "0.5"
    combined_power_blend_delta_caps: str = "1.0"
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
    metric_weights: Dict[str, str] = field(
        default_factory=lambda: {
            "laplacian_variance": "0.25",
            "dwt_level2_hv_energy": "0.25",
            "sum_squared_gradient": "0.25",
            "total_variation": "0.25",
        }
    )
    metric_profile_config_path: str = ""
    scoring_profile_name: str = ""
    metric_scope: str = "global_and_local_metrics"
    runtime_penalty_weight: str = "0.0"
    robustness_lambda: str = "0.5"
    missing_dataset_penalty: str = "1.0"
    metric_experiments_enabled: bool = False
    metric_experiments_candidate_source: str = METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
    metric_experiments_shortlist_top_k: str = "8"
    metric_experiment_rows: List[SpecGuiMetricExperimentRowState] = field(
        default_factory=lambda: [SpecGuiMetricExperimentRowState()]
    )
    validation_oracle_enabled: bool = False
    validation_oracle_mode: str = VALIDATION_ORACLE_MODE_TRIO
    validation_oracle_top_k: str = "8"
    validation_oracle_budget_per_dataset: str = ""
    validation_oracle_skip_image_metric_stage: bool = False
    validation_oracle_workflow_mode: str = VALIDATION_ORACLE_WORKFLOW_NORMAL
    validation_oracle_human_bead_annotations: str = ""
    validation_oracle_defaults_json: str = "{}"
    validation_oracle_scenes_json: str = "[]"
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


def _parse_pixel_shift_power_list_csv(value: str, field_name: str) -> List[float]:
    """Parse powers with the same finite-positive contract as rectification."""

    values = _parse_float_list_csv(value, field_name)
    try:
        return [normalize_pixel_shift_power(item) for item in values]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain finite positive numbers.") from exc


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


def _legacy_state_metric_weights(state: SpecGuiState) -> Dict[str, str]:
    return {
        "laplacian_variance": state.weight_laplacian_variance,
        "dwt_level2_hv_energy": state.weight_dwt_level2_hv_energy,
        "sum_squared_gradient": state.weight_sum_squared_gradient,
        "total_variation": state.weight_total_variation,
    }


def _state_metric_weights(state: SpecGuiState) -> Dict[str, str]:
    if not state.metric_weights:
        return _legacy_state_metric_weights(state)
    default_weights = {
        "laplacian_variance": "0.25",
        "dwt_level2_hv_energy": "0.25",
        "sum_squared_gradient": "0.25",
        "total_variation": "0.25",
    }
    if dict(state.metric_weights) == default_weights:
        return _legacy_state_metric_weights(state)
    return dict(state.metric_weights)


def _parse_state_metric_weights(state: SpecGuiState) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    registry = build_metric_registry(include_saved=False)
    for name, raw in _state_metric_weights(state).items():
        metric_name = str(name).strip()
        if not metric_name:
            continue
        if metric_name not in registry:
            raise ValueError(f"Unknown metric weight name: {metric_name}")
        try:
            value = float(str(raw).strip())
        except Exception as exc:
            raise ValueError(f"Metric weight {metric_name} must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Metric weight {metric_name} must be finite.")
        if value != 0.0:
            weights[metric_name] = value
    if sum(abs(weight) for weight in weights.values()) <= 0.0:
        raise ValueError("Metric weights must have a positive absolute sum.")
    return weights


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
    family = normalize_displacement_metric_family(row.family)
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
    if family == DEFAULT_DISPLACEMENT_METRIC_FAMILY:
        if alphas != [DEFAULT_DISPLACEMENT_ALPHA]:
            raise ValueError(
                f"metric experiment row {row_index} intensity family requires alpha=1.0."
            )
        return formulation_payload
    if family == BLURRED_COMPOSITE_CORR_METRIC_FAMILY:
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
    family = normalize_displacement_metric_family(
        payload.get("family", DEFAULT_DISPLACEMENT_METRIC_FAMILY),
        allow_unknown=True,
    )
    component_weights = cast(Optional[Dict[str, object]], payload.get("component_weights"))
    gradient_weight = "0.5"
    laplacian_weight = "0.5"
    if component_weights is not None:
        gradient_weight = str(component_weights.get("gradient", 0.0))
        laplacian_weight = str(component_weights.get("laplacian", 0.0))
    return SpecGuiMetricExperimentRowState(
        formulation_id=str(payload.get("id") or payload.get("formulation_id") or f"formulation_{default_idx:02d}"),
        family=family,
        alphas=_list_to_csv(cast(List[object], payload.get("alphas", []))) or "1.0",
        gradient_weight=gradient_weight,
        laplacian_weight=laplacian_weight,
    )


def _state_relocation_rules(state: SpecGuiState) -> List[PathRelocationRule]:
    raw_rules: List[str] = []
    for idx, rule in enumerate(state.relocation_rules, start=1):
        old_root = rule.old_root.strip()
        new_root = rule.new_root.strip()
        if not old_root and not new_root:
            continue
        if not old_root or not new_root:
            raise ValueError(
                f"Relocation row {idx} must define both old root and new root."
            )
        raw_rules.append(f"{old_root}={new_root}")
    return _parse_path_relocation_rules(raw_rules)


def _validation_oracle_workflow_mode_for_skip(skip_image_metric_stage: bool) -> str:
    return (
        VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE
        if skip_image_metric_stage
        else VALIDATION_ORACLE_WORKFLOW_NORMAL
    )


def _validation_oracle_skip_image_stage_from_state(state: SpecGuiState) -> bool:
    workflow_mode = str(state.validation_oracle_workflow_mode or "").strip()
    if workflow_mode == VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE:
        return True
    if workflow_mode == VALIDATION_ORACLE_WORKFLOW_NORMAL:
        return bool(state.validation_oracle_skip_image_metric_stage)
    return bool(state.validation_oracle_skip_image_metric_stage)


def _validation_oracle_mode_from_state(state: SpecGuiState) -> str:
    return str(state.validation_oracle_mode or VALIDATION_ORACLE_MODE_TRIO).strip()


HUMAN_BEAD_WORKBENCH_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _launch_human_bead_workbench_for_sweep(
    *,
    annotation_path_text: str,
    output_dir_text: str,
    dataset_images: Mapping[str, Path],
    choose_output_path: Optional[Callable[[Path], Optional[Path]]] = None,
    launcher: Optional[Callable[..., Path]] = None,
    parent: Any = None,
    default_radius_px: float = 4.0,
) -> str:
    cleaned_path = str(annotation_path_text or "").strip()
    if cleaned_path:
        output_path = Path(cleaned_path)
    else:
        output_dir = Path(str(output_dir_text or "").strip() or ".")
        default_path = output_dir / "human_beads.json"
        selected_path = (
            choose_output_path(default_path)
            if choose_output_path is not None
            else default_path
        )
        if selected_path is None:
            return ""
        output_path = Path(selected_path)

    if launcher is None:
        from moment_images.human_bead_oracle import launch_human_bead_workbench as launcher

    saved_path = launcher(
        dataset_images={dataset_id: Path(path) for dataset_id, path in dataset_images.items()},
        output_path=output_path,
        default_radius_px=float(default_radius_px),
        parent=parent,
    )
    return str(saved_path or output_path)


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
            "pixel_shift_powers": _parse_pixel_shift_power_list_csv(
                state.pixel_shift_powers, "pixel shift powers"
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
            "pixel_shift_powers": _parse_pixel_shift_power_list_csv(
                state.combined_pixel_shift_powers, "combined pixel shift powers"
            ),
            "pixel_power_blend_enabled": _parse_bool_list_csv(
                state.combined_power_blend_enabled, "combined power blend enabled"
            ),
            "pixel_power_blend_raised_powers": _parse_float_list_csv(
                state.combined_power_blend_raised_powers, "combined raised powers"
            ),
            "pixel_power_blend_alphas": _parse_float_list_csv(
                state.combined_power_blend_alphas, "combined power blend alphas"
            ),
            "pixel_power_blend_delta_caps": _parse_float_list_csv(
                state.combined_power_blend_delta_caps, "combined power blend delta caps"
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

    stage_mode = state.stage_mode.strip().lower() or "two_stage"
    if stage_mode not in {"two_stage", "single_stage"}:
        raise ValueError("Execution mode must be either 'two_stage' or 'single_stage'.")
    refinement_method = state.refinement_method.strip().lower() or "auto_fine"
    if refinement_method not in {"auto_fine", "neighbor"}:
        raise ValueError("Refinement method must be either 'auto_fine' or 'neighbor'.")
    refinement_extrapolation = state.refinement_extrapolation.strip().lower() or "half_step_bounded"
    if refinement_extrapolation not in {
        "half_step_bounded",
        "inside_only",
        "full_step_outside",
    }:
        raise ValueError(
            "Refinement extrapolation must be one of 'half_step_bounded', "
            "'inside_only', or 'full_step_outside'."
        )
    if stage_mode == "two_stage":
        stage1_fraction_value = float(state.stage1_fraction.strip())
    else:
        try:
            stage1_fraction_value = float(state.stage1_fraction.strip())
        except Exception:
            stage1_fraction_value = SweepStageSpec().stage1_fraction
    scheduler_mode = state.scheduler_mode.strip().lower() or SCHEDULER_MODE_ADAPTIVE_MEMORY
    if scheduler_mode not in SCHEDULER_MODES:
        raise ValueError("Scheduler mode must be either 'adaptive_memory' or 'fixed_batch'.")
    memory_safety_fraction = float(state.memory_safety_fraction.strip())
    memory_enforcement = state.memory_enforcement.strip().lower() or MEMORY_ENFORCEMENT_AUTO
    worker_recycle_tasks = (
        int(state.worker_recycle_tasks.strip())
        if state.worker_recycle_tasks.strip()
        else None
    )
    scheduling_payload: Dict[str, object] = {
        "mode": scheduler_mode,
        "memory_safety_fraction": memory_safety_fraction,
        "memory_telemetry": bool(state.memory_telemetry),
        "worker_recycle_tasks": worker_recycle_tasks,
        "memory_enforcement": memory_enforcement,
    }
    SweepSchedulingSpec.from_dict(scheduling_payload)

    scoring_payload: Dict[str, object] = {
        "runtime_penalty_weight": float(state.runtime_penalty_weight.strip()),
        "robustness_lambda": float(state.robustness_lambda.strip()),
        "missing_dataset_penalty": float(state.missing_dataset_penalty.strip()),
        "metric_scope": state.metric_scope.strip() or "global_and_local_metrics",
    }
    if state.scoring_profile_name.strip():
        scoring_payload["profile_name"] = state.scoring_profile_name.strip()
        if state.metric_profile_config_path.strip():
            scoring_payload["profile_config_path"] = state.metric_profile_config_path.strip()
    else:
        scoring_payload["metric_weights"] = _parse_state_metric_weights(state)

    payload: Dict[str, object] = {
        "output_dir": state.output_dir.strip() or None,
        "strategies": strategies,
        "datasets": datasets,
        "search_spaces": search_spaces,
        "stages": {
            "mode": stage_mode,
            "total_budget_per_dataset": int(state.total_budget_per_dataset.strip()),
            "stage1_fraction": stage1_fraction_value,
            "top_k": int(state.top_k.strip()),
            "max_workers": (
                int(state.max_workers.strip()) if state.max_workers.strip() else None
            ),
            "scheduling": dict(scheduling_payload),
            "refinement": {
                "method": refinement_method,
                "extrapolation": refinement_extrapolation,
            },
        },
        "scoring": scoring_payload,
        "save_figures": bool(state.save_figures),
        "analyze": bool(state.analyze),
    }
    if state.metric_experiments_enabled:
        if not state.metric_experiment_rows:
            raise ValueError(
                "At least one metric experiment formulation row is required when enabled."
            )
        metric_candidate_source = (
            state.metric_experiments_candidate_source.strip().lower()
            or METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        )
        if metric_candidate_source not in METRIC_CANDIDATE_SOURCES:
            raise ValueError(
                "Metric experiment source must be baseline_shortlist or independent_search."
            )
        metric_budget = int(state.metric_experiments_shortlist_top_k.strip())
        experimental_payload: Dict[str, object] = {
            "enabled": True,
            "candidate_source": metric_candidate_source,
            "shortlist_top_k": metric_budget,
            "formulations": [
                _state_metric_row_to_spec(row, idx + 1)
                for idx, row in enumerate(state.metric_experiment_rows)
            ],
        }
        if metric_candidate_source == METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH:
            experimental_payload["search_stages"] = {
                "mode": "two_stage",
                "total_budget_per_dataset": metric_budget,
                "stage1_fraction": stage1_fraction_value,
                "top_k": int(state.top_k.strip()),
                "max_workers": (
                    int(state.max_workers.strip()) if state.max_workers.strip() else None
                ),
                "scheduling": dict(scheduling_payload),
                "refinement": {
                    "method": refinement_method,
                    "extrapolation": refinement_extrapolation,
                },
            }
        payload["experimental_metric_sweep"] = experimental_payload
    if state.validation_oracle_enabled:
        oracle_mode = _validation_oracle_mode_from_state(state)
        if oracle_mode not in VALIDATION_ORACLE_MODES:
            raise ValueError(
                "Validation oracle mode must be one of: "
                + ", ".join(sorted(VALIDATION_ORACLE_MODES))
            )
        defaults_text = state.validation_oracle_defaults_json.strip() or "{}"
        scenes_text = state.validation_oracle_scenes_json.strip() or "[]"
        defaults_payload = json.loads(defaults_text)
        scenes_payload = json.loads(scenes_text)
        if not isinstance(defaults_payload, dict):
            raise ValueError("Validation oracle defaults JSON must be an object.")
        if not isinstance(scenes_payload, list) or not all(
            isinstance(item, dict) for item in scenes_payload
        ):
            raise ValueError(
                "Validation oracle scene overrides JSON must be a list of objects."
            )
        defaults_payload = dict(defaults_payload)
        defaults_payload.pop("strict", None)
        if oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD:
            annotation_path = state.validation_oracle_human_bead_annotations.strip()
            if annotation_path:
                defaults_payload["human_bead_annotations"] = annotation_path
        scene_overrides_by_dataset: Dict[str, Dict[str, object]] = {}
        for idx, scene_payload_raw in enumerate(scenes_payload, start=1):
            scene_payload = _strip_validation_oracle_gui_owned_scene_fields(
                scene_payload_raw
            )
            dataset_id = str(scene_payload.get("dataset_id") or "").strip()
            if not dataset_id:
                raise ValueError(
                    "Validation oracle scene override "
                    f"{idx} must include dataset_id."
                )
            if dataset_id in scene_overrides_by_dataset:
                raise ValueError(
                    "Validation oracle scene overrides must have unique dataset_id values."
                )
            scene_overrides_by_dataset[dataset_id] = scene_payload

        scene_entries: List[Dict[str, object]] = []
        dataset_ids = {row.dataset_id.strip() for row in state.datasets}
        unknown_overrides = sorted(set(scene_overrides_by_dataset) - dataset_ids)
        if unknown_overrides:
            raise ValueError(
                "Validation oracle scene override dataset_id not found in datasets: "
                + ", ".join(unknown_overrides)
            )
        for row in state.datasets:
            dataset_id = row.dataset_id.strip()
            validation_workspace = row.validation_workspace_path.strip()
            validation_image = row.validation_image_path.strip()
            thorcam_image = row.thorcam_image_path.strip()
            has_image_pair = bool(validation_image and thorcam_image)
            has_partial_image_pair = bool(validation_image or thorcam_image) and not has_image_pair
            scene_entry = dict(scene_overrides_by_dataset.get(dataset_id, {}))
            scene_entry["dataset_id"] = dataset_id
            if oracle_mode != VALIDATION_ORACLE_MODE_HUMAN_BEAD:
                if not validation_workspace and not has_image_pair:
                    raise ValueError(
                        f"Dataset '{dataset_id}' validation workspace or "
                        "validation/ThorCam image pair is required "
                        "when validation oracle is enabled."
                    )
                if has_partial_image_pair and not validation_workspace:
                    raise ValueError(
                        f"Dataset '{dataset_id}' must provide both validation and "
                        "ThorCam image paths when no validation workspace is set."
                    )
                if validation_workspace:
                    scene_entry["validation_workspace"] = validation_workspace
                    scene_entry["use_validation_workspace_transforms"] = True
                else:
                    scene_entry["validation_image"] = validation_image
                    scene_entry["thorcam_image"] = thorcam_image
            scene_entries.append(scene_entry)
        oracle_payload: Dict[str, object] = {
            "enabled": True,
            "oracle_mode": oracle_mode,
            "top_k": int(state.validation_oracle_top_k.strip()),
            "skip_image_metric_stage": _validation_oracle_skip_image_stage_from_state(state),
            "defaults": defaults_payload,
            "scenes": scene_entries,
        }
        if state.validation_oracle_budget_per_dataset.strip():
            oracle_payload["budget_per_dataset"] = int(
                state.validation_oracle_budget_per_dataset.strip()
            )
        payload["validation_oracle_sweep"] = oracle_payload
    relocation_rules = _state_relocation_rules(state)
    if relocation_rules or state.adopt_relocated:
        payload["relocation"] = {
            "path_mappings": _relocation_rules_to_payload(relocation_rules),
            "adopt_relocated": bool(state.adopt_relocated),
        }
    return payload


def spec_dict_to_state(payload: Dict[str, object]) -> SpecGuiState:
    """Convert JSON payload into GUI state using tolerant defaults."""
    state = SpecGuiState()
    state.output_dir = str(payload.get("output_dir") or "")
    relocation = payload.get("relocation")
    if isinstance(relocation, dict):
        state.adopt_relocated = bool(relocation.get("adopt_relocated", False))
        state.relocation_rules = [
            SpecGuiRelocationRuleState(
                old_root=rule.old_root,
                new_root=rule.new_root,
            )
            for rule in _parse_relocation_rules_payload(relocation.get("path_mappings"))
        ]

    strategies = [str(item).lower() for item in cast(List[object], payload.get("strategies", []))]
    state.strategy_pixel = "pixel" in strategies or not strategies
    state.strategy_combined = "combined" in strategies or not strategies
    state.strategy_global = "global" in strategies

    stages = cast(Dict[str, object], payload.get("stages", {}))
    state.stage_mode = str(stages.get("mode", state.stage_mode) or state.stage_mode)
    state.total_budget_per_dataset = str(
        stages.get("total_budget_per_dataset", state.total_budget_per_dataset)
    )
    state.stage1_fraction = str(stages.get("stage1_fraction", state.stage1_fraction))
    state.top_k = str(stages.get("top_k", state.top_k))
    state.max_workers = (
        "" if stages.get("max_workers") is None else str(stages.get("max_workers"))
    )
    scheduling = SweepSchedulingSpec.from_dict(
        cast(Optional[Dict[str, object]], stages.get("scheduling"))
    )
    state.scheduler_mode = scheduling.mode
    state.memory_safety_fraction = str(scheduling.memory_safety_fraction)
    state.memory_telemetry = bool(scheduling.memory_telemetry)
    state.worker_recycle_tasks = (
        ""
        if scheduling.worker_recycle_tasks is None
        else str(scheduling.worker_recycle_tasks)
    )
    state.memory_enforcement = scheduling.memory_enforcement
    refinement = cast(Dict[str, object], stages.get("refinement", {}))
    state.refinement_method = str(refinement.get("method", state.refinement_method))
    state.refinement_extrapolation = str(
        refinement.get("extrapolation", state.refinement_extrapolation)
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
        state.pixel_shift_powers = _list_to_csv(
            cast(List[object], pixel_space.get("pixel_shift_powers", []))
        ) or state.pixel_shift_powers
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
        state.combined_pixel_shift_powers = _list_to_csv(
            cast(List[object], combined_space.get("pixel_shift_powers", []))
        ) or state.combined_pixel_shift_powers
        state.combined_power_blend_enabled = _list_to_csv(
            cast(List[object], combined_space.get("pixel_power_blend_enabled", []))
        ) or state.combined_power_blend_enabled
        state.combined_power_blend_raised_powers = _list_to_csv(
            cast(List[object], combined_space.get("pixel_power_blend_raised_powers", []))
        ) or state.combined_power_blend_raised_powers
        state.combined_power_blend_alphas = _list_to_csv(
            cast(List[object], combined_space.get("pixel_power_blend_alphas", []))
        ) or state.combined_power_blend_alphas
        state.combined_power_blend_delta_caps = _list_to_csv(
            cast(List[object], combined_space.get("pixel_power_blend_delta_caps", []))
        ) or state.combined_power_blend_delta_caps
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
    state.scoring_profile_name = str(scoring.get("profile_name") or "")
    state.metric_profile_config_path = str(scoring.get("profile_config_path") or "")
    state.metric_scope = str(scoring.get("metric_scope", state.metric_scope))
    metric_weights = cast(Dict[str, object], scoring.get("metric_weights", {}))
    if metric_weights:
        state.metric_weights = {
            str(name): str(value) for name, value in metric_weights.items()
        }
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
        state.metric_experiments_candidate_source = str(
            experimental_payload.get(
                "candidate_source", METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST
            )
        )
        shortlist_top_k = experimental_payload.get("shortlist_top_k")
        if shortlist_top_k is None:
            search_stages = experimental_payload.get("search_stages")
            if isinstance(search_stages, dict):
                shortlist_top_k = search_stages.get(
                    "total_budget_per_dataset",
                    state.metric_experiments_shortlist_top_k,
                )
            else:
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
        state.metric_experiments_candidate_source = METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST
        state.metric_experiments_shortlist_top_k = str(stages.get("top_k", state.top_k))
    else:
        state.metric_experiments_enabled = False
        state.metric_experiments_candidate_source = METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        state.metric_experiments_shortlist_top_k = str(stages.get("top_k", state.top_k))

    if formulations_payload:
        state.metric_experiment_rows = [
            _metric_formulation_payload_to_state_row(item, idx + 1)
            for idx, item in enumerate(formulations_payload)
        ]
    else:
        state.metric_experiment_rows = [SpecGuiMetricExperimentRowState()]

    validation_row_paths_by_dataset: Dict[str, Dict[str, str]] = {}
    validation_payload = cast(
        Optional[Dict[str, object]], payload.get("validation_oracle_sweep")
    )
    if validation_payload is not None:
        state.validation_oracle_enabled = bool(validation_payload.get("enabled", False))
        state.validation_oracle_mode = str(
            validation_payload.get("oracle_mode", VALIDATION_ORACLE_MODE_TRIO)
        )
        if state.validation_oracle_mode not in VALIDATION_ORACLE_MODES:
            state.validation_oracle_mode = VALIDATION_ORACLE_MODE_TRIO
        state.validation_oracle_top_k = str(
            validation_payload.get("top_k", state.validation_oracle_top_k)
        )
        state.validation_oracle_budget_per_dataset = (
            ""
            if validation_payload.get("budget_per_dataset") is None
            else str(validation_payload.get("budget_per_dataset"))
        )
        state.validation_oracle_skip_image_metric_stage = bool(
            validation_payload.get("skip_image_metric_stage", False)
        )
        state.validation_oracle_workflow_mode = _validation_oracle_workflow_mode_for_skip(
            state.validation_oracle_skip_image_metric_stage
        )
        defaults_payload = validation_payload.get("defaults", {})
        scenes_payload = validation_payload.get("scenes", [])
        defaults_payload_for_gui = (
            dict(defaults_payload) if isinstance(defaults_payload, dict) else {}
        )
        state.validation_oracle_human_bead_annotations = _validation_oracle_text(
            defaults_payload_for_gui.get("human_bead_annotations")
        )
        for key in VALIDATION_ORACLE_GUI_OWNED_DEFAULT_KEYS:
            defaults_payload_for_gui.pop(key, None)
        state.validation_oracle_defaults_json = json.dumps(
            defaults_payload_for_gui,
            indent=2,
        )
        advanced_scenes: List[Dict[str, object]] = []
        if isinstance(scenes_payload, list):
            for scene_payload_raw in scenes_payload:
                if not isinstance(scene_payload_raw, dict):
                    continue
                scene_payload = dict(scene_payload_raw)
                dataset_id = str(scene_payload.get("dataset_id") or "").strip()
                if dataset_id:
                    validation_workspace = _validation_oracle_text(
                        scene_payload.get("validation_workspace")
                    )
                    validation_row_paths_by_dataset[dataset_id] = {
                        "validation_workspace_path": validation_workspace,
                        "validation_image_path": ""
                        if validation_workspace
                        else str(scene_payload.get("validation_image") or ""),
                        "thorcam_image_path": ""
                        if validation_workspace
                        else str(scene_payload.get("thorcam_image") or ""),
                    }
                scene_payload = _strip_validation_oracle_gui_owned_scene_fields(
                    scene_payload
                )
                if scene_payload:
                    advanced_scenes.append(scene_payload)
        state.validation_oracle_scenes_json = json.dumps(
            advanced_scenes,
            indent=2,
        )
    else:
        state.validation_oracle_enabled = False
        state.validation_oracle_mode = VALIDATION_ORACLE_MODE_TRIO
        state.validation_oracle_human_bead_annotations = ""
        state.validation_oracle_skip_image_metric_stage = False
        state.validation_oracle_workflow_mode = VALIDATION_ORACLE_WORKFLOW_NORMAL

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
            validation_workspace_path=validation_row_paths_by_dataset.get(
                str(entry.get("id") or entry.get("dataset_id") or ""),
                {},
            ).get("validation_workspace_path", ""),
            validation_image_path=validation_row_paths_by_dataset.get(
                str(entry.get("id") or entry.get("dataset_id") or ""),
                {},
            ).get("validation_image_path", ""),
            thorcam_image_path=validation_row_paths_by_dataset.get(
                str(entry.get("id") or entry.get("dataset_id") or ""),
                {},
            ).get("thorcam_image_path", ""),
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
    run_mode = state.run_mode.strip().lower() or "normal"
    if run_mode not in {"normal", "extend"}:
        issues.append("Run mode must be normal or extend.")
    if run_mode == "extend" and not state.output_dir.strip():
        issues.append("Extend existing sweep requires an output directory.")
    try:
        _state_relocation_rules(state)
    except ValueError as exc:
        issues.append(str(exc))
    if state.adopt_relocated and not state.output_dir.strip():
        issues.append("Adopt relocated sweep requires an output directory.")

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

    stage_mode = state.stage_mode.strip().lower() or "two_stage"
    if stage_mode not in {"two_stage", "single_stage"}:
        issues.append("Execution mode must be either 'two_stage' or 'single_stage'.")

    refinement_method = state.refinement_method.strip().lower() or "auto_fine"
    if refinement_method not in {"auto_fine", "neighbor"}:
        issues.append("Refinement method must be either 'auto_fine' or 'neighbor'.")

    refinement_extrapolation = state.refinement_extrapolation.strip().lower() or "half_step_bounded"
    if refinement_extrapolation not in {
        "half_step_bounded",
        "inside_only",
        "full_step_outside",
    }:
        issues.append(
            "Refinement extrapolation must be one of 'half_step_bounded', "
            "'inside_only', or 'full_step_outside'."
        )

    try:
        if int(state.total_budget_per_dataset.strip()) < 1:
            issues.append("Total budget per dataset must be >= 1.")
    except Exception:
        issues.append("Total budget per dataset must be an integer.")

    if stage_mode == "two_stage":
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

    scheduler_mode = state.scheduler_mode.strip().lower() or SCHEDULER_MODE_ADAPTIVE_MEMORY
    if scheduler_mode not in SCHEDULER_MODES:
        issues.append("Scheduler mode must be either 'adaptive_memory' or 'fixed_batch'.")
    memory_enforcement = state.memory_enforcement.strip().lower() or MEMORY_ENFORCEMENT_AUTO
    if memory_enforcement not in MEMORY_ENFORCEMENT_MODES:
        issues.append("Memory enforcement must be 'auto', 'required', or 'advisory'.")
    try:
        memory_fraction = float(state.memory_safety_fraction.strip())
        if not np.isfinite(memory_fraction) or memory_fraction <= 0.0 or memory_fraction > 1.0:
            issues.append("Memory safety fraction must be in (0, 1].")
    except Exception:
        issues.append("Memory safety fraction must be numeric.")
    if state.worker_recycle_tasks.strip():
        try:
            if int(state.worker_recycle_tasks.strip()) < 1:
                issues.append("Worker recycle tasks must be >= 1 when provided.")
        except Exception:
            issues.append("Worker recycle tasks must be an integer when provided.")

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
                ("pixel shift powers", state.pixel_shift_powers, "pixel_shift_power"),
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
                (
                    "combined pixel shift powers",
                    state.combined_pixel_shift_powers,
                    "pixel_shift_power",
                ),
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
            elif field_type == "pixel_shift_power":
                _parse_pixel_shift_power_list_csv(value, field_name)
            else:
                _parse_bool_list_csv(value, field_name)
        except ValueError as exc:
            issues.append(str(exc))

    if state.scoring_profile_name.strip():
        try:
            resolve_profile_weights_for_sweep(
                state.scoring_profile_name.strip(),
                config_path=Path(state.metric_profile_config_path.strip())
                if state.metric_profile_config_path.strip()
                else None,
                scoring_weights=DEFAULT_SCORE_WEIGHTS,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Scoring profile is invalid: {exc}")
    else:
        try:
            _parse_state_metric_weights(state)
        except ValueError as exc:
            issues.append(str(exc))

    for label, raw in [
        ("runtime penalty weight", state.runtime_penalty_weight),
        ("robustness lambda", state.robustness_lambda),
        ("missing dataset penalty", state.missing_dataset_penalty),
    ]:
        try:
            float(raw.strip())
        except Exception:
            issues.append(f"{label} must be numeric.")
    if state.metric_scope.strip() not in {"global_and_local_metrics", "global_only_metrics"}:
        issues.append("Metric scope must be global_and_local_metrics or global_only_metrics.")

    if state.metric_experiments_enabled:
        if not state.strategy_pixel and not state.strategy_combined:
            issues.append(
                "Experimental metrics require at least one pixel/combined strategy."
            )
        metric_candidate_source = (
            state.metric_experiments_candidate_source.strip().lower()
            or METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        )
        if metric_candidate_source not in METRIC_CANDIDATE_SOURCES:
            issues.append(
                "Metric experiment source must be baseline_shortlist or independent_search."
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

            try:
                family = normalize_displacement_metric_family(row.family)
            except ValueError:
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

            if family == DEFAULT_DISPLACEMENT_METRIC_FAMILY:
                if alphas != [DEFAULT_DISPLACEMENT_ALPHA]:
                    issues.append(
                        f"Metric experiment row {idx}: intensity family requires alpha=1.0."
                    )
                continue

            if family == BLURRED_COMPOSITE_CORR_METRIC_FAMILY:
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

    if state.validation_oracle_enabled:
        oracle_mode = _validation_oracle_mode_from_state(state)
        if oracle_mode not in VALIDATION_ORACLE_MODES:
            issues.append(
                "Validation oracle mode must be one of: "
                + ", ".join(sorted(VALIDATION_ORACLE_MODES))
            )
        is_human_bead_mode = oracle_mode == VALIDATION_ORACLE_MODE_HUMAN_BEAD
        defaults_payload_for_validation: Dict[str, object] = {}
        scenes_payload_for_validation: List[Dict[str, object]] = []
        if (
            stage_mode != "two_stage"
            and not _validation_oracle_skip_image_stage_from_state(state)
        ):
            issues.append(
                "Validation oracle sweep requires two_stage execution mode unless "
                "skip image-metric stage2 is enabled."
            )
        try:
            defaults_payload_raw = json.loads(
                state.validation_oracle_defaults_json.strip() or "{}"
            )
            if isinstance(defaults_payload_raw, dict):
                defaults_payload_for_validation = dict(defaults_payload_raw)
            else:
                issues.append("Validation oracle defaults JSON must be an object.")
        except Exception as exc:
            issues.append(f"Validation oracle defaults JSON is invalid: {exc}")
        try:
            scenes_payload_raw = json.loads(
                state.validation_oracle_scenes_json.strip() or "[]"
            )
            if isinstance(scenes_payload_raw, list) and all(
                isinstance(item, dict) for item in scenes_payload_raw
            ):
                scenes_payload_for_validation = [
                    dict(item) for item in scenes_payload_raw
                ]
            else:
                issues.append(
                    "Validation oracle scene overrides JSON must be a list of objects."
                )
        except Exception as exc:
            issues.append(f"Validation oracle scene overrides JSON is invalid: {exc}")

        if is_human_bead_mode:
            has_default_annotations = bool(
                state.validation_oracle_human_bead_annotations.strip()
                or _validation_oracle_text(
                    defaults_payload_for_validation.get("human_bead_annotations")
                )
            )
            dataset_ids = {row.dataset_id.strip() for row in state.datasets}
            annotated_scene_ids = {
                str(scene.get("dataset_id") or "").strip()
                for scene in scenes_payload_for_validation
                if _validation_oracle_text(scene.get("human_bead_annotations"))
            }
            if not has_default_annotations and not dataset_ids.issubset(
                annotated_scene_ids
            ):
                issues.append(
                    "Validation oracle human bead annotations path is required "
                    "in the annotations field, defaults JSON, or every scene override."
                )
        else:
            for idx, row in enumerate(state.datasets, start=1):
                prefix = f"Dataset row {idx}"
                validation_workspace = row.validation_workspace_path.strip()
                validation_image = row.validation_image_path.strip()
                thorcam_image = row.thorcam_image_path.strip()
                has_image_pair = bool(validation_image and thorcam_image)
                has_partial_image_pair = (
                    bool(validation_image or thorcam_image) and not has_image_pair
                )
                if not validation_workspace and not has_image_pair:
                    issues.append(
                        f"{prefix}: workspace or validation/ThorCam image pair "
                        "is required when validation oracle is enabled."
                    )
                if has_partial_image_pair and not validation_workspace:
                    issues.append(
                        f"{prefix}: both validation and ThorCam image paths are "
                        "required when no validation workspace is set."
                    )
        try:
            if int(state.validation_oracle_top_k.strip()) < 1:
                issues.append("Validation oracle top-k must be >= 1.")
        except Exception:
            issues.append("Validation oracle top-k must be an integer.")
        if state.validation_oracle_budget_per_dataset.strip():
            try:
                if int(state.validation_oracle_budget_per_dataset.strip()) < 1:
                    issues.append("Validation oracle budget per dataset must be >= 1.")
            except Exception:
                issues.append("Validation oracle budget per dataset must be an integer.")

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
        self.relocation_rows: List[Dict[str, Any]] = []
        self.editable_widgets: List[Any] = []
        self.log_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
        self.run_thread: Optional[threading.Thread] = None
        self._is_running = False
        self.preview_thread: Optional[threading.Thread] = None
        self._is_previewing = False
        self.status_var = tk.StringVar(value="Idle")
        self.planned_runs_var = tk.StringVar(value="Planned runs: not updated")
        self.planned_runs_detail_var = tk.StringVar(value="")
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.run_mode_var = tk.StringVar(value="normal")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.adopt_relocated_var = tk.BooleanVar(value=False)

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
        if enabled and hasattr(self, "stage_mode_var"):
            self._toggle_stage_mode_controls()
        if enabled and hasattr(self, "metric_experiments_enabled_var"):
            self._toggle_metric_experiment_controls()

    def _update_run_button_text(self) -> None:
        if not hasattr(self, "run_btn"):
            return
        if bool(self.dry_run_var.get()):
            label = "Preview Run"
        elif self.run_mode_var.get() == "extend":
            label = "Extend Sweep"
        else:
            label = "Run Sweep"
        self.run_btn.configure(text=label)

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

        run_mode_frame = self.ttk.Frame(basic_frame)
        run_mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self.ttk.Label(run_mode_frame, text="Run Mode").pack(side="left", padx=(0, 8))
        normal_radio = self.ttk.Radiobutton(
            run_mode_frame,
            text="New / Resume exact spec",
            variable=self.run_mode_var,
            value="normal",
            command=self._update_run_button_text,
        )
        normal_radio.pack(side="left", padx=4)
        self._register_editable(normal_radio)
        extend_radio = self.ttk.Radiobutton(
            run_mode_frame,
            text="Extend existing sweep",
            variable=self.run_mode_var,
            value="extend",
            command=self._update_run_button_text,
        )
        extend_radio.pack(side="left", padx=4)
        self._register_editable(extend_radio)
        dry_run_check = self.ttk.Checkbutton(
            run_mode_frame,
            text="Dry run only",
            variable=self.dry_run_var,
            command=self._update_run_button_text,
        )
        dry_run_check.pack(side="left", padx=12)
        self._register_editable(dry_run_check)

        plan_frame = self.ttk.Frame(basic_frame)
        plan_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        self.update_planned_runs_btn = self.ttk.Button(
            plan_frame,
            text="Update planned runs",
            command=self._on_update_planned_runs,
        )
        self.update_planned_runs_btn.pack(side="left", padx=(0, 8))
        self._register_editable(self.update_planned_runs_btn)
        plan_text_frame = self.ttk.Frame(plan_frame)
        plan_text_frame.pack(side="left", fill="x", expand=True)
        self.ttk.Label(
            plan_text_frame,
            textvariable=self.planned_runs_var,
        ).pack(anchor="w")
        self.ttk.Label(
            plan_text_frame,
            textvariable=self.planned_runs_detail_var,
        ).pack(anchor="w")

        relocation_frame = self.ttk.LabelFrame(basic_frame, text="Relocation")
        relocation_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        relocation_btns = self.ttk.Frame(relocation_frame)
        relocation_btns.pack(fill="x", padx=4, pady=3)
        add_reloc_btn = self.ttk.Button(
            relocation_btns,
            text="Add Mapping",
            command=self._add_relocation_row,
        )
        add_reloc_btn.pack(side="left", padx=2)
        self._register_editable(add_reloc_btn)
        preview_reloc_btn = self.ttk.Button(
            relocation_btns,
            text="Preview",
            command=self._on_preview_relocation,
        )
        preview_reloc_btn.pack(side="left", padx=2)
        self._register_editable(preview_reloc_btn)
        apply_reloc_btn = self.ttk.Button(
            relocation_btns,
            text="Apply",
            command=self._on_apply_relocation,
        )
        apply_reloc_btn.pack(side="left", padx=2)
        self._register_editable(apply_reloc_btn)
        adopt_check = self.ttk.Checkbutton(
            relocation_btns,
            text="Adopt relocated sweep",
            variable=self.adopt_relocated_var,
        )
        adopt_check.pack(side="left", padx=12)
        self._register_editable(adopt_check)
        self.relocation_rows_frame = self.ttk.Frame(relocation_frame)
        self.relocation_rows_frame.pack(fill="x", padx=4, pady=3)

        strat_frame = self.ttk.Frame(basic_frame)
        strat_frame.grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=2)
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
        stage_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self.ttk.Label(stage_frame, text="Execution mode").grid(row=0, column=0, sticky="w")
        self.stage_mode_var = self.tk.StringVar(value="two_stage")
        self.stage_mode_combo = self.ttk.Combobox(
            stage_frame,
            textvariable=self.stage_mode_var,
            values=["two_stage", "single_stage"],
            width=14,
            state="readonly",
        )
        self.stage_mode_combo.grid(row=0, column=1, padx=4)
        self.stage_mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda _evt: self._toggle_stage_mode_controls(),
        )
        self._register_editable(self.stage_mode_combo)
        self.ttk.Label(stage_frame, text="Total budget").grid(row=0, column=2, sticky="w")
        self.total_budget_entry = self.ttk.Entry(stage_frame, width=8)
        self.total_budget_entry.grid(row=0, column=3, padx=4)
        self._register_editable(self.total_budget_entry)
        self.stage1_fraction_label = self.ttk.Label(stage_frame, text="Stage1 fraction")
        self.stage1_fraction_label.grid(row=0, column=4, sticky="w")
        self.stage1_fraction_entry = self.ttk.Entry(stage_frame, width=6)
        self.stage1_fraction_entry.grid(row=0, column=5, padx=4)
        self._register_editable(self.stage1_fraction_entry)
        self.ttk.Label(stage_frame, text="Top-K").grid(row=0, column=6, sticky="w")
        self.top_k_entry = self.ttk.Entry(stage_frame, width=6)
        self.top_k_entry.grid(row=0, column=7, padx=4)
        self._register_editable(self.top_k_entry)
        self.ttk.Label(stage_frame, text="Max workers").grid(row=0, column=8, sticky="w")
        self.max_workers_entry = self.ttk.Entry(stage_frame, width=8)
        self.max_workers_entry.grid(row=0, column=9, padx=4)
        self._register_editable(self.max_workers_entry)
        self.ttk.Label(stage_frame, text="Refinement").grid(row=1, column=0, sticky="w")
        self.refinement_method_var = self.tk.StringVar(value="auto_fine")
        self.refinement_method_combo = self.ttk.Combobox(
            stage_frame,
            textvariable=self.refinement_method_var,
            values=["auto_fine", "neighbor"],
            width=14,
            state="readonly",
        )
        self.refinement_method_combo.grid(row=1, column=1, padx=4, pady=2)
        self._register_editable(self.refinement_method_combo)
        self.ttk.Label(stage_frame, text="Edge policy").grid(row=1, column=2, sticky="w")
        self.refinement_extrapolation_var = self.tk.StringVar(value="half_step_bounded")
        self.refinement_extrapolation_combo = self.ttk.Combobox(
            stage_frame,
            textvariable=self.refinement_extrapolation_var,
            values=["half_step_bounded", "inside_only", "full_step_outside"],
            width=18,
            state="readonly",
        )
        self.refinement_extrapolation_combo.grid(row=1, column=3, padx=4, pady=2)
        self._register_editable(self.refinement_extrapolation_combo)
        self.ttk.Label(stage_frame, text="Scheduler").grid(row=2, column=0, sticky="w")
        self.scheduler_mode_var = self.tk.StringVar(value=SCHEDULER_MODE_ADAPTIVE_MEMORY)
        self.scheduler_mode_combo = self.ttk.Combobox(
            stage_frame,
            textvariable=self.scheduler_mode_var,
            values=[SCHEDULER_MODE_ADAPTIVE_MEMORY, SCHEDULER_MODE_FIXED_BATCH],
            width=16,
            state="readonly",
        )
        self.scheduler_mode_combo.grid(row=2, column=1, padx=4, pady=2)
        self._register_editable(self.scheduler_mode_combo)
        self.ttk.Label(stage_frame, text="RAM budget fraction (0-1)").grid(row=2, column=2, sticky="w")
        self.memory_safety_entry = self.ttk.Entry(stage_frame, width=6)
        self.memory_safety_entry.grid(row=2, column=3, padx=4, pady=2)
        self._register_editable(self.memory_safety_entry)
        self.memory_telemetry_var = self.tk.BooleanVar(value=True)
        memory_telemetry_chk = self.ttk.Checkbutton(
            stage_frame,
            text="Record memory telemetry",
            variable=self.memory_telemetry_var,
        )
        memory_telemetry_chk.grid(row=2, column=4, columnspan=2, sticky="w", padx=4, pady=2)
        self._register_editable(memory_telemetry_chk)
        self.ttk.Label(stage_frame, text="Recycle after tasks").grid(row=2, column=6, sticky="w")
        self.worker_recycle_entry = self.ttk.Entry(stage_frame, width=8)
        self.worker_recycle_entry.grid(row=2, column=7, padx=4, pady=2)
        self._register_editable(self.worker_recycle_entry)
        self.ttk.Label(stage_frame, text="Memory enforcement").grid(row=3, column=0, sticky="w")
        self.memory_enforcement_var = self.tk.StringVar(value=MEMORY_ENFORCEMENT_AUTO)
        self.memory_enforcement_combo = self.ttk.Combobox(
            stage_frame,
            textvariable=self.memory_enforcement_var,
            values=sorted(MEMORY_ENFORCEMENT_MODES),
            width=16,
            state="readonly",
        )
        self.memory_enforcement_combo.grid(row=3, column=1, padx=4, pady=2)
        self._register_editable(self.memory_enforcement_combo)

        toggles_frame = self.ttk.Frame(basic_frame)
        toggles_frame.grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=2)
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
        self.pixel_shift_power_entry = self._build_labeled_entry(left, 5, "Pixel shift powers", "1.0")

        self.comb_window_entry = self._build_labeled_entry(right, 0, "Combined refinement windows", "15,25")
        self.comb_disp_entry = self._build_labeled_entry(right, 1, "Combined max displacements", "5,15")
        self.comb_gauss_entry = self._build_labeled_entry(right, 2, "Combined gauss sigmas", "3.0,5.0")
        self.comb_disp_gauss_entry = self._build_labeled_entry(right, 3, "Combined disp gauss sigmas", "0.0,1.0")
        self.comb_thresh_entry = self._build_labeled_entry(right, 4, "Combined pixel thresholds", "0.05")
        self.comb_shift_power_entry = self._build_labeled_entry(right, 5, "Combined pixel shift powers", "1.0")
        self.comb_power_blend_enabled_entry = self._build_labeled_entry(right, 6, "Power blend enabled", "false")
        self.comb_power_blend_raised_entry = self._build_labeled_entry(right, 7, "Power blend raised powers", "2.0")
        self.comb_power_blend_alpha_entry = self._build_labeled_entry(right, 8, "Power blend alphas", "0.5")
        self.comb_power_blend_cap_entry = self._build_labeled_entry(right, 9, "Power blend delta caps", "1.0")
        self.comb_global_thresh_entry = self._build_labeled_entry(right, 10, "Combined global thresholds", "0.1,0.15")
        self.comb_corr_pad_entry = self._build_labeled_entry(
            right, 11, "Combined correlation pad widths", str(RectificationRunConfig().correlation_pad_width)
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
        self.metric_scope_entry = self._build_labeled_entry(adv_right, 4, "Metric scope", "global_and_local_metrics")
        self.runtime_penalty_entry = self._build_labeled_entry(adv_right, 5, "Runtime penalty weight", "0.0")
        self.robustness_entry = self._build_labeled_entry(adv_right, 6, "Robustness lambda", "0.5")
        self.missing_penalty_entry = self._build_labeled_entry(adv_right, 7, "Missing dataset penalty", "1.0")
        self.metric_profile_config_entry = self._build_labeled_entry(adv_right, 8, "Metric profile config", "")
        profile_btn = self.ttk.Button(adv_right, text="Browse Profiles", command=self._browse_metric_profile_config)
        profile_btn.grid(row=9, column=1, sticky="w", padx=4, pady=2)
        self._register_editable(profile_btn)
        self.ttk.Label(adv_right, text="Scoring profile").grid(row=10, column=0, sticky="w", padx=2, pady=2)
        self.scoring_profile_var = self.tk.StringVar(value="")
        self.scoring_profile_combo = self.ttk.Combobox(
            adv_right,
            textvariable=self.scoring_profile_var,
            values=[""],
            width=28,
        )
        self.scoring_profile_combo.grid(row=10, column=1, sticky="ew", padx=4, pady=2)
        self.scoring_profile_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_profile_weights())
        self._register_editable(self.scoring_profile_combo)
        self.ttk.Label(adv_right, text="Resolved metric weights").grid(row=11, column=0, sticky="nw", padx=2, pady=2)
        self.metric_weight_tree = self.ttk.Treeview(
            adv_right,
            columns=("metric", "weight"),
            show="headings",
            height=7,
        )
        self.metric_weight_tree.heading("metric", text="metric")
        self.metric_weight_tree.heading("weight", text="weight")
        self.metric_weight_tree.column("metric", width=185, stretch=True)
        self.metric_weight_tree.column("weight", width=80, stretch=False)
        self.metric_weight_tree.grid(row=11, column=1, sticky="ew", padx=4, pady=2)
        self.metric_weight_tree.bind("<Double-1>", self._edit_metric_weight_cell)
        self._register_editable(self.metric_weight_tree)
        metric_btns = self.ttk.Frame(adv_right)
        metric_btns.grid(row=12, column=1, sticky="w", padx=4, pady=2)
        add_metric_btn = self.ttk.Button(metric_btns, text="Add Weight", command=self._add_metric_weight_row)
        add_metric_btn.pack(side="left", padx=2)
        remove_metric_btn = self.ttk.Button(metric_btns, text="Remove Weight", command=self._remove_selected_metric_weight)
        remove_metric_btn.pack(side="left", padx=2)
        self._register_editable(add_metric_btn)
        self._register_editable(remove_metric_btn)

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
        self.ttk.Label(metric_exp_controls, text="Source").pack(
            side="left", padx=(12, 2)
        )
        self.metric_candidate_source_var = self.tk.StringVar(
            value=METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        )
        self.metric_candidate_source_combo = self.ttk.Combobox(
            metric_exp_controls,
            textvariable=self.metric_candidate_source_var,
            values=[
                METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH,
                METRIC_CANDIDATE_SOURCE_BASELINE_SHORTLIST,
            ],
            width=18,
            state="readonly",
        )
        self.metric_candidate_source_combo.pack(side="left", padx=2)
        self._register_editable(self.metric_candidate_source_combo)
        self.ttk.Label(metric_exp_controls, text="Budget/top-k").pack(
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

        validation_container = self.ttk.LabelFrame(
            self.advanced_frame, text="Validation Oracle Sweep (Optional)"
        )
        validation_container.pack(fill="x", padx=4, pady=6)
        validation_controls = self.ttk.Frame(validation_container)
        validation_controls.pack(fill="x", padx=4, pady=2)
        self.validation_oracle_enabled_var = self.tk.BooleanVar(value=False)
        validation_enabled_chk = self.ttk.Checkbutton(
            validation_controls,
            text="Enable validation-guided sweep",
            variable=self.validation_oracle_enabled_var,
        )
        validation_enabled_chk.pack(side="left", padx=4)
        self._register_editable(validation_enabled_chk)
        self.ttk.Label(validation_controls, text="Mode").pack(side="left", padx=(12, 2))
        self.validation_oracle_mode_var = self.tk.StringVar(value=VALIDATION_ORACLE_MODE_TRIO)
        validation_mode_menu = self.ttk.OptionMenu(
            validation_controls,
            self.validation_oracle_mode_var,
            VALIDATION_ORACLE_MODE_TRIO,
            VALIDATION_ORACLE_MODE_TRIO,
            VALIDATION_ORACLE_MODE_HUMAN_BEAD,
        )
        validation_mode_menu.pack(side="left", padx=2)
        self._register_editable(validation_mode_menu)
        self.ttk.Label(validation_controls, text="Top-k").pack(side="left", padx=(12, 2))
        self.validation_oracle_top_k_entry = self.ttk.Entry(validation_controls, width=8)
        self.validation_oracle_top_k_entry.insert(0, "8")
        self.validation_oracle_top_k_entry.pack(side="left", padx=2)
        self._register_editable(self.validation_oracle_top_k_entry)
        self.ttk.Label(validation_controls, text="Budget").pack(side="left", padx=(12, 2))
        self.validation_oracle_budget_entry = self.ttk.Entry(validation_controls, width=8)
        self.validation_oracle_budget_entry.pack(side="left", padx=2)
        self._register_editable(self.validation_oracle_budget_entry)

        validation_workflow_frame = self.ttk.Frame(validation_container)
        validation_workflow_frame.pack(fill="x", padx=8, pady=(2, 4))
        self.validation_oracle_workflow_var = self.tk.StringVar(
            value=VALIDATION_ORACLE_WORKFLOW_NORMAL
        )
        normal_workflow_radio = self.ttk.Radiobutton(
            validation_workflow_frame,
            text="Normal validation: run image-metric stage2, then validate",
            variable=self.validation_oracle_workflow_var,
            value=VALIDATION_ORACLE_WORKFLOW_NORMAL,
        )
        normal_workflow_radio.pack(anchor="w", pady=1)
        self._register_editable(normal_workflow_radio)
        skip_workflow_radio = self.ttk.Radiobutton(
            validation_workflow_frame,
            text="Validation-guided refinement: skip image-metric stage2",
            variable=self.validation_oracle_workflow_var,
            value=VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE,
        )
        skip_workflow_radio.pack(anchor="w", pady=1)
        self._register_editable(skip_workflow_radio)

        bead_annotations_frame = self.ttk.Frame(validation_container)
        bead_annotations_frame.pack(fill="x", padx=8, pady=(0, 4))
        self.ttk.Label(bead_annotations_frame, text="Bead annotations").pack(side="left")
        self.validation_oracle_human_bead_annotations_var = self.tk.StringVar(value="")
        bead_annotations_entry = self.ttk.Entry(
            bead_annotations_frame,
            textvariable=self.validation_oracle_human_bead_annotations_var,
            width=72,
        )
        bead_annotations_entry.pack(side="left", fill="x", expand=True, padx=4)
        self._register_editable(bead_annotations_entry)
        bead_annotations_browse_btn = self.ttk.Button(
            bead_annotations_frame,
            text="Browse",
            command=lambda: self._browse_json_path(
                self.validation_oracle_human_bead_annotations_var
            ),
        )
        bead_annotations_browse_btn.pack(side="left", padx=2)
        self._register_editable(bead_annotations_browse_btn)
        bead_annotations_annotate_btn = self.ttk.Button(
            bead_annotations_frame,
            text="Annotate",
            command=self._open_human_bead_workbench,
        )
        bead_annotations_annotate_btn.pack(side="left", padx=2)
        self._register_editable(bead_annotations_annotate_btn)

        self.ttk.Label(validation_container, text="Defaults JSON").pack(anchor="w", padx=8)
        self.validation_oracle_defaults_text = self.tk.Text(
            validation_container,
            height=4,
            wrap="none",
        )
        self.validation_oracle_defaults_text.insert(self.tk.END, "{}")
        self.validation_oracle_defaults_text.pack(fill="x", padx=8, pady=(0, 4))
        self._register_editable(self.validation_oracle_defaults_text)
        self.ttk.Label(validation_container, text="Scenes JSON").pack(anchor="w", padx=8)
        self.validation_oracle_scenes_text = self.tk.Text(
            validation_container,
            height=5,
            wrap="none",
        )
        self.validation_oracle_scenes_text.insert(self.tk.END, "[]")
        self.validation_oracle_scenes_text.pack(fill="x", padx=8, pady=(0, 6))
        self._register_editable(self.validation_oracle_scenes_text)

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
            self.metric_candidate_source_combo.configure(state="readonly")
            self.metric_shortlist_entry.configure(state="normal")
            self.metric_add_row_btn.configure(state="normal")
        else:
            self.metric_exp_headers.pack_forget()
            self.metric_experiment_rows_frame.pack_forget()
            self.metric_candidate_source_combo.configure(state="disabled")
            self.metric_shortlist_entry.configure(state="disabled")
            self.metric_add_row_btn.configure(state="disabled")

    def _toggle_stage_mode_controls(self) -> None:
        stage_mode = self.stage_mode_var.get().strip().lower() or "two_stage"
        if stage_mode == "single_stage":
            self.stage1_fraction_entry.configure(state="disabled")
            self.stage1_fraction_label.configure(text="Stage1 fraction (unused)")
            self.refinement_method_combo.configure(state="disabled")
            self.refinement_extrapolation_combo.configure(state="disabled")
        else:
            self.stage1_fraction_entry.configure(state="normal")
            self.stage1_fraction_label.configure(text="Stage1 fraction")
            self.refinement_method_combo.configure(state="readonly")
            self.refinement_extrapolation_combo.configure(state="readonly")

    def _metric_weights_from_table(self) -> Dict[str, str]:
        weights: Dict[str, str] = {}
        for item_id in self.metric_weight_tree.get_children():
            values = self.metric_weight_tree.item(item_id, "values")
            if len(values) < 2:
                continue
            metric = str(values[0]).strip()
            weight = str(values[1]).strip()
            if metric:
                weights[metric] = weight
        return weights

    def _set_metric_weight_rows(self, weights: Mapping[str, object]) -> None:
        self.metric_weight_tree.delete(*self.metric_weight_tree.get_children())
        for metric, weight in weights.items():
            self.metric_weight_tree.insert("", "end", values=(str(metric), str(weight)))

    def _refresh_profile_choices(self, selected: str = "") -> None:
        registry = build_metric_registry()
        profiles = default_profiles(registry, DEFAULT_SCORE_WEIGHTS)
        config_path = self.metric_profile_config_entry.get().strip()
        if config_path:
            try:
                config, _ = resolve_metric_config(
                    None,
                    registry,
                    scoring_weights=DEFAULT_SCORE_WEIGHTS,
                    config_path=Path(config_path),
                )
                profiles = config.profiles
            except Exception:
                profiles = default_profiles(registry, DEFAULT_SCORE_WEIGHTS)
        names = [""] + [profile.name for profile in profiles if profile.name != "saved_original"]
        self.scoring_profile_combo.configure(values=names)
        if selected in names:
            self.scoring_profile_var.set(selected)

    def _load_selected_profile_weights(self) -> None:
        profile_name = self.scoring_profile_var.get().strip()
        if not profile_name:
            return
        try:
            weights, _, _ = resolve_profile_weights_for_sweep(
                profile_name,
                config_path=Path(self.metric_profile_config_entry.get().strip())
                if self.metric_profile_config_entry.get().strip()
                else None,
                scoring_weights=DEFAULT_SCORE_WEIGHTS,
            )
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showwarning("Metric Profile", str(exc))
            return
        self._set_metric_weight_rows(weights)

    def _browse_metric_profile_config(self) -> None:
        selected = self.filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")]
        )
        if not selected:
            return
        self.metric_profile_config_entry.delete(0, self.tk.END)
        self.metric_profile_config_entry.insert(0, selected)
        self._refresh_profile_choices(selected=self.scoring_profile_var.get().strip())
        self._load_selected_profile_weights()

    def _add_metric_weight_row(self) -> None:
        registry = build_metric_registry(include_saved=False)
        existing = set(self._metric_weights_from_table())
        metric_name = next((name for name in registry if name not in existing), "laplacian_variance")
        self.metric_weight_tree.insert("", "end", values=(metric_name, "0.0"))

    def _remove_selected_metric_weight(self) -> None:
        for item_id in self.metric_weight_tree.selection():
            self.metric_weight_tree.delete(item_id)

    def _edit_metric_weight_cell(self, event: object) -> None:
        row_id = self.metric_weight_tree.identify_row(getattr(event, "y", 0))
        column_id = self.metric_weight_tree.identify_column(getattr(event, "x", 0))
        if not row_id or column_id not in {"#1", "#2"}:
            return
        values = list(self.metric_weight_tree.item(row_id, "values"))
        if len(values) < 2:
            return
        current = values[0 if column_id == "#1" else 1]
        prompt = "Metric name" if column_id == "#1" else "Metric weight"
        simpledialog = __import__("tkinter.simpledialog", fromlist=["simpledialog"])
        value = simpledialog.askstring("Metric Weight", prompt, initialvalue=str(current), parent=self.root)
        if value is None:
            return
        values[0 if column_id == "#1" else 1] = value.strip()
        self.metric_weight_tree.item(row_id, values=values)

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
        validation_workspace_var = self.tk.StringVar(value=initial_state.validation_workspace_path)
        validation_image_var = self.tk.StringVar(value=initial_state.validation_image_path)
        thorcam_image_var = self.tk.StringVar(value=initial_state.thorcam_image_path)
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

        self.ttk.Label(row_frame, text="Workspace").pack(side="left")
        workspace_entry = self.ttk.Entry(row_frame, textvariable=validation_workspace_var, width=24)
        workspace_entry.pack(side="left", padx=2)
        self._register_editable(workspace_entry)
        workspace_browse_btn = self.ttk.Button(
            row_frame,
            text="Browse",
            command=lambda sp=validation_workspace_var: self._browse_workspace_path(sp),
        )
        workspace_browse_btn.pack(side="left", padx=2)
        self._register_editable(workspace_browse_btn)

        for label, var in [
            ("Validation", validation_image_var),
            ("ThorCam", thorcam_image_var),
        ]:
            self.ttk.Label(row_frame, text=label).pack(side="left")
            image_entry = self.ttk.Entry(row_frame, textvariable=var, width=24)
            image_entry.pack(side="left", padx=2)
            self._register_editable(image_entry)
            image_browse_btn = self.ttk.Button(
                row_frame,
                text="Browse",
                command=lambda sp=var: self._browse_image_path(sp),
            )
            image_browse_btn.pack(side="left", padx=2)
            self._register_editable(image_browse_btn)

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
                "validation_workspace_var": validation_workspace_var,
                "validation_image_var": validation_image_var,
                "thorcam_image_var": thorcam_image_var,
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
            DEFAULT_DISPLACEMENT_METRIC_FAMILY,
            BLURRED_GRADIENT_CORR_METRIC_FAMILY,
            BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
            BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
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

    def _clear_relocation_rows(self) -> None:
        for row in self.relocation_rows:
            row["frame"].destroy()
        self.relocation_rows = []

    def _add_relocation_row(
        self,
        initial: Optional[SpecGuiRelocationRuleState] = None,
    ) -> None:
        initial_state = initial or SpecGuiRelocationRuleState()
        row_frame = self.ttk.Frame(self.relocation_rows_frame)
        row_frame.pack(fill="x", pady=2)
        old_var = self.tk.StringVar(value=initial_state.old_root)
        new_var = self.tk.StringVar(value=initial_state.new_root)

        self.ttk.Label(row_frame, text="Old root").pack(side="left", padx=2)
        old_entry = self.ttk.Entry(row_frame, textvariable=old_var, width=38)
        old_entry.pack(side="left", padx=2)
        self._register_editable(old_entry)
        self.ttk.Label(row_frame, text="New root").pack(side="left", padx=2)
        new_entry = self.ttk.Entry(row_frame, textvariable=new_var, width=38)
        new_entry.pack(side="left", padx=2)
        self._register_editable(new_entry)
        remove_btn = self.ttk.Button(
            row_frame,
            text="Remove",
            command=lambda frame=row_frame: self._remove_relocation_row(frame),
        )
        remove_btn.pack(side="left", padx=4)
        self._register_editable(remove_btn)

        self.relocation_rows.append(
            {
                "frame": row_frame,
                "old_var": old_var,
                "new_var": new_var,
            }
        )

    def _remove_relocation_row(self, frame: Any) -> None:
        for idx, row in enumerate(self.relocation_rows):
            if row["frame"] == frame:
                row["frame"].destroy()
                del self.relocation_rows[idx]
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

    def _browse_image_path(self, path_var: Any) -> None:
        selected = self.filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if selected:
            path_var.set(selected)

    def _browse_workspace_path(self, path_var: Any) -> None:
        selected = self.filedialog.askopenfilename(
            filetypes=[("Validation Workspace", "*.npz"), ("NPZ files", "*.npz")]
        )
        if selected:
            path_var.set(selected)

    def _browse_json_path(self, path_var: Any) -> None:
        selected = self.filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if selected:
            path_var.set(selected)

    def _human_bead_dataset_images_from_rows(self) -> Dict[str, Path]:
        dataset_images: Dict[str, Path] = {}
        for row in self.dataset_rows:
            dataset_id = str(row["id_var"].get()).strip()
            source_type = str(row["source_type_var"].get()).strip()
            source_path = str(row["source_path_var"].get()).strip()
            if not dataset_id or source_type != "source_image" or not source_path:
                continue
            path = Path(source_path)
            if path.suffix.lower() in HUMAN_BEAD_WORKBENCH_IMAGE_EXTENSIONS and path.exists():
                dataset_images[dataset_id] = path
        return dataset_images

    def _open_human_bead_workbench(self) -> None:
        def choose_output_path(default_path: Path) -> Optional[Path]:
            selected = self.filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=str(default_path.parent),
                initialfile=default_path.name,
            )
            return None if not selected else Path(selected)

        try:
            saved_path = _launch_human_bead_workbench_for_sweep(
                annotation_path_text=self.validation_oracle_human_bead_annotations_var.get(),
                output_dir_text=self.output_dir_entry.get(),
                dataset_images=self._human_bead_dataset_images_from_rows(),
                choose_output_path=choose_output_path,
                parent=self.root,
            )
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Human Bead Annotations", str(exc))
            return
        if saved_path:
            self.validation_oracle_human_bead_annotations_var.set(saved_path)

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
                    validation_workspace_path=row["validation_workspace_var"].get().strip(),
                    validation_image_path=row["validation_image_var"].get().strip(),
                    thorcam_image_path=row["thorcam_image_var"].get().strip(),
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

        relocation_rules: List[SpecGuiRelocationRuleState] = []
        for row in self.relocation_rows:
            relocation_rules.append(
                SpecGuiRelocationRuleState(
                    old_root=row["old_var"].get().strip(),
                    new_root=row["new_var"].get().strip(),
                )
            )

        return SpecGuiState(
            output_dir=self.output_dir_entry.get().strip(),
            run_mode=self.run_mode_var.get().strip(),
            dry_run=bool(self.dry_run_var.get()),
            adopt_relocated=bool(self.adopt_relocated_var.get()),
            relocation_rules=relocation_rules,
            strategy_pixel=bool(self.strategy_pixel_var.get()),
            strategy_combined=bool(self.strategy_combined_var.get()),
            strategy_global=bool(self.strategy_global_var.get()),
            stage_mode=self.stage_mode_var.get().strip(),
            total_budget_per_dataset=self.total_budget_entry.get().strip(),
            stage1_fraction=self.stage1_fraction_entry.get().strip(),
            top_k=self.top_k_entry.get().strip(),
            max_workers=self.max_workers_entry.get().strip(),
            scheduler_mode=self.scheduler_mode_var.get().strip(),
            memory_safety_fraction=self.memory_safety_entry.get().strip(),
            memory_telemetry=bool(self.memory_telemetry_var.get()),
            worker_recycle_tasks=self.worker_recycle_entry.get().strip(),
            memory_enforcement=self.memory_enforcement_var.get().strip(),
            refinement_method=self.refinement_method_var.get().strip(),
            refinement_extrapolation=self.refinement_extrapolation_var.get().strip(),
            analyze=bool(self.analyze_var.get()),
            save_figures=bool(self.save_figures_var.get()),
            pixel_similarity_window_sizes=self.pixel_window_entry.get().strip(),
            pixel_max_displacements=self.pixel_disp_entry.get().strip(),
            pixel_gauss_sigmas=self.pixel_gauss_entry.get().strip(),
            pixel_disp_gauss_sigmas=self.pixel_disp_gauss_entry.get().strip(),
            pixel_thresholds=self.pixel_thresh_entry.get().strip(),
            pixel_shift_powers=self.pixel_shift_power_entry.get().strip(),
            combined_refinement_window_sizes=self.comb_window_entry.get().strip(),
            combined_max_displacements=self.comb_disp_entry.get().strip(),
            combined_gauss_sigmas=self.comb_gauss_entry.get().strip(),
            combined_disp_gauss_sigmas=self.comb_disp_gauss_entry.get().strip(),
            combined_pixel_thresholds=self.comb_thresh_entry.get().strip(),
            combined_pixel_shift_powers=self.comb_shift_power_entry.get().strip(),
            combined_power_blend_enabled=self.comb_power_blend_enabled_entry.get().strip(),
            combined_power_blend_raised_powers=self.comb_power_blend_raised_entry.get().strip(),
            combined_power_blend_alphas=self.comb_power_blend_alpha_entry.get().strip(),
            combined_power_blend_delta_caps=self.comb_power_blend_cap_entry.get().strip(),
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
            metric_weights=self._metric_weights_from_table(),
            metric_profile_config_path=self.metric_profile_config_entry.get().strip(),
            scoring_profile_name=self.scoring_profile_var.get().strip(),
            metric_scope=self.metric_scope_entry.get().strip(),
            runtime_penalty_weight=self.runtime_penalty_entry.get().strip(),
            robustness_lambda=self.robustness_entry.get().strip(),
            missing_dataset_penalty=self.missing_penalty_entry.get().strip(),
            metric_experiments_enabled=bool(self.metric_experiments_enabled_var.get()),
            metric_experiments_candidate_source=self.metric_candidate_source_var.get().strip(),
            metric_experiments_shortlist_top_k=self.metric_shortlist_entry.get().strip(),
            metric_experiment_rows=metric_rows,
            validation_oracle_enabled=bool(self.validation_oracle_enabled_var.get()),
            validation_oracle_mode=self.validation_oracle_mode_var.get().strip(),
            validation_oracle_top_k=self.validation_oracle_top_k_entry.get().strip(),
            validation_oracle_budget_per_dataset=self.validation_oracle_budget_entry.get().strip(),
            validation_oracle_skip_image_metric_stage=(
                self.validation_oracle_workflow_var.get()
                == VALIDATION_ORACLE_WORKFLOW_SKIP_IMAGE_STAGE
            ),
            validation_oracle_workflow_mode=(
                self.validation_oracle_workflow_var.get()
                if self.validation_oracle_workflow_var.get() in VALIDATION_ORACLE_WORKFLOW_MODES
                else VALIDATION_ORACLE_WORKFLOW_NORMAL
            ),
            validation_oracle_human_bead_annotations=(
                self.validation_oracle_human_bead_annotations_var.get().strip()
            ),
            validation_oracle_defaults_json=self.validation_oracle_defaults_text.get(
                "1.0",
                self.tk.END,
            ).strip(),
            validation_oracle_scenes_json=self.validation_oracle_scenes_text.get(
                "1.0",
                self.tk.END,
            ).strip(),
            datasets=datasets,
            json_preview=self.json_text.get("1.0", self.tk.END).strip(),
        )

    def _set_state(self, state: SpecGuiState) -> None:
        self.output_dir_entry.delete(0, self.tk.END)
        self.output_dir_entry.insert(0, state.output_dir)
        self.run_mode_var.set(state.run_mode if state.run_mode in {"normal", "extend"} else "normal")
        self.dry_run_var.set(bool(state.dry_run))
        self.adopt_relocated_var.set(bool(state.adopt_relocated))
        self.strategy_pixel_var.set(state.strategy_pixel)
        self.strategy_combined_var.set(state.strategy_combined)
        self.strategy_global_var.set(state.strategy_global)
        self.stage_mode_var.set(state.stage_mode)
        self.total_budget_entry.delete(0, self.tk.END)
        self.total_budget_entry.insert(0, state.total_budget_per_dataset)
        self.stage1_fraction_entry.delete(0, self.tk.END)
        self.stage1_fraction_entry.insert(0, state.stage1_fraction)
        self.top_k_entry.delete(0, self.tk.END)
        self.top_k_entry.insert(0, state.top_k)
        self.max_workers_entry.delete(0, self.tk.END)
        self.max_workers_entry.insert(0, state.max_workers)
        self.scheduler_mode_var.set(state.scheduler_mode)
        self.memory_safety_entry.delete(0, self.tk.END)
        self.memory_safety_entry.insert(0, state.memory_safety_fraction)
        self.memory_telemetry_var.set(state.memory_telemetry)
        self.worker_recycle_entry.delete(0, self.tk.END)
        self.worker_recycle_entry.insert(0, state.worker_recycle_tasks)
        self.memory_enforcement_var.set(state.memory_enforcement)
        self.refinement_method_var.set(state.refinement_method)
        self.refinement_extrapolation_var.set(state.refinement_extrapolation)
        self.analyze_var.set(state.analyze)
        self.save_figures_var.set(state.save_figures)
        self.metric_experiments_enabled_var.set(state.metric_experiments_enabled)
        self.metric_candidate_source_var.set(
            state.metric_experiments_candidate_source
            if state.metric_experiments_candidate_source in METRIC_CANDIDATE_SOURCES
            else METRIC_CANDIDATE_SOURCE_INDEPENDENT_SEARCH
        )
        self.metric_shortlist_entry.delete(0, self.tk.END)
        self.metric_shortlist_entry.insert(0, state.metric_experiments_shortlist_top_k)
        self.validation_oracle_enabled_var.set(state.validation_oracle_enabled)
        validation_oracle_mode = state.validation_oracle_mode
        if validation_oracle_mode not in VALIDATION_ORACLE_MODES:
            validation_oracle_mode = VALIDATION_ORACLE_MODE_TRIO
        self.validation_oracle_mode_var.set(validation_oracle_mode)
        self.validation_oracle_top_k_entry.delete(0, self.tk.END)
        self.validation_oracle_top_k_entry.insert(0, state.validation_oracle_top_k)
        self.validation_oracle_budget_entry.delete(0, self.tk.END)
        self.validation_oracle_budget_entry.insert(0, state.validation_oracle_budget_per_dataset)
        workflow_mode = state.validation_oracle_workflow_mode
        if workflow_mode not in VALIDATION_ORACLE_WORKFLOW_MODES:
            workflow_mode = _validation_oracle_workflow_mode_for_skip(
                state.validation_oracle_skip_image_metric_stage
            )
        self.validation_oracle_workflow_var.set(workflow_mode)
        self.validation_oracle_human_bead_annotations_var.set(
            state.validation_oracle_human_bead_annotations
        )
        self.validation_oracle_defaults_text.delete("1.0", self.tk.END)
        self.validation_oracle_defaults_text.insert(
            self.tk.END,
            state.validation_oracle_defaults_json or "{}",
        )
        self.validation_oracle_scenes_text.delete("1.0", self.tk.END)
        self.validation_oracle_scenes_text.insert(
            self.tk.END,
            state.validation_oracle_scenes_json or "[]",
        )

        field_mapping = [
            (self.pixel_window_entry, state.pixel_similarity_window_sizes),
            (self.pixel_disp_entry, state.pixel_max_displacements),
            (self.pixel_gauss_entry, state.pixel_gauss_sigmas),
            (self.pixel_disp_gauss_entry, state.pixel_disp_gauss_sigmas),
            (self.pixel_thresh_entry, state.pixel_thresholds),
            (self.pixel_shift_power_entry, state.pixel_shift_powers),
            (self.comb_window_entry, state.combined_refinement_window_sizes),
            (self.comb_disp_entry, state.combined_max_displacements),
            (self.comb_gauss_entry, state.combined_gauss_sigmas),
            (self.comb_disp_gauss_entry, state.combined_disp_gauss_sigmas),
            (self.comb_thresh_entry, state.combined_pixel_thresholds),
            (self.comb_shift_power_entry, state.combined_pixel_shift_powers),
            (self.comb_power_blend_enabled_entry, state.combined_power_blend_enabled),
            (self.comb_power_blend_raised_entry, state.combined_power_blend_raised_powers),
            (self.comb_power_blend_alpha_entry, state.combined_power_blend_alphas),
            (self.comb_power_blend_cap_entry, state.combined_power_blend_delta_caps),
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
            (self.metric_scope_entry, state.metric_scope),
            (self.metric_profile_config_entry, state.metric_profile_config_path),
            (self.runtime_penalty_entry, state.runtime_penalty_weight),
            (self.robustness_entry, state.robustness_lambda),
            (self.missing_penalty_entry, state.missing_dataset_penalty),
        ]
        for widget, value in field_mapping:
            widget.delete(0, self.tk.END)
            widget.insert(0, value)
        self._set_metric_weight_rows(_state_metric_weights(state))
        self._refresh_profile_choices(selected=state.scoring_profile_name)
        self.scoring_profile_var.set(state.scoring_profile_name)
        if state.scoring_profile_name:
            self._load_selected_profile_weights()

        self._clear_dataset_rows()
        for dataset in state.datasets:
            self._add_dataset_row(dataset)
        self._clear_relocation_rows()
        for rule in state.relocation_rules:
            self._add_relocation_row(rule)
        self._clear_metric_experiment_rows()
        for metric_row in state.metric_experiment_rows:
            self._add_metric_experiment_row(metric_row)
        if not self.metric_experiment_rows:
            self._add_metric_experiment_row()
        self._toggle_stage_mode_controls()
        self._toggle_metric_experiment_controls()

        self.json_text.delete("1.0", self.tk.END)
        if state.json_preview:
            self.json_text.insert(self.tk.END, state.json_preview)
        self._update_run_button_text()

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

    def _relocated_payload_from_current_state(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        state = self._get_state()
        payload = state_to_spec_dict(state)
        rules, _ = _relocation_options_from_spec_payload(payload)
        return _apply_relocation_to_spec_payload(payload, rules)

    def _on_preview_relocation(self) -> None:
        try:
            relocated_payload, report = self._relocated_payload_from_current_state()
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Relocation Preview", str(exc))
            return
        applied = cast(List[object], report.get("applied", []))
        self.json_text.delete("1.0", self.tk.END)
        self.json_text.insert(self.tk.END, json.dumps(relocated_payload, indent=2))
        self.messagebox.showinfo(
            "Relocation Preview",
            f"Relocation preview applied {len(applied)} path change(s).",
        )

    def _on_apply_relocation(self) -> None:
        try:
            relocated_payload, report = self._relocated_payload_from_current_state()
            state = spec_dict_to_state(relocated_payload)
            self._set_state(state)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Apply Relocation", str(exc))
            return
        applied = cast(List[object], report.get("applied", []))
        self._append_log(f"Applied relocation to {len(applied)} path(s).")

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

    def _format_planned_runs_summary(self, preview: Mapping[str, object]) -> Tuple[str, str]:
        totals = cast(Mapping[str, object], preview.get("totals", {}))
        stages = cast(Mapping[str, object], preview.get("stages", {}))
        metric = cast(Mapping[str, object], preview.get("metric_experiments", {}))

        total_planned = int(totals.get("planned_run_count", 0) or 0)
        total_pending = int(totals.get("pending_run_count", 0) or 0)
        total_reused = int(totals.get("skipped_valid_count", 0) or 0)
        exact_label = "exact" if bool(totals.get("exact", False)) else "includes estimates"
        summary = (
            f"Planned: {total_planned} | Pending: {total_pending} | "
            f"Reusable: {total_reused} ({exact_label})"
        )

        stage_parts: List[str] = []
        for stage_name in ("stage1", "stage2"):
            stage = stages.get(stage_name)
            if not isinstance(stage, dict):
                continue
            stage_parts.append(
                f"{stage_name}: {int(stage.get('pending_run_count', 0) or 0)} pending"
            )
        metric_detail = "Metric: disabled"
        if bool(metric.get("enabled", False)):
            metric_pending = _metric_pending_count(metric)
            metric_reused = _metric_skipped_valid_count(metric)
            metric_prefix = "exact" if bool(metric.get("exact", False)) else "estimated"
            metric_detail = (
                f"Metric: {metric_prefix} {metric_pending} pending, "
                f"{metric_reused} reusable"
            )
        detail = " | ".join(stage_parts + [metric_detail])
        return summary, detail

    def _preview_worker(self, payload: Dict[str, object], run_mode: str) -> None:
        try:
            relocation_rules, adopt_relocated = _relocation_options_from_spec_payload(payload)
            relocated_payload, relocation_report = _apply_relocation_to_spec_payload(
                payload,
                relocation_rules,
            )
            spec = SweepExperimentSpec.from_dict(relocated_payload)
            kwargs: Dict[str, object] = {
                "extend_existing": run_mode == "extend",
                "relocation_rules": relocation_rules,
                "relocation_report": relocation_report,
                "allow_dataset_generation": False,
                "reuse_existing_artifacts": False,
            }
            if adopt_relocated:
                kwargs["adopt_relocated"] = True
            preview = build_sweep_run_preview(spec, **kwargs)
            self.log_queue.put({"type": "preview_done", "preview": preview})
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(
                {
                    "type": "preview_error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def _on_update_planned_runs(self) -> None:
        if self._is_previewing:
            return
        state = self._get_state()
        issues = validate_state(state)
        if issues:
            self.planned_runs_var.set("Plan unavailable")
            self.planned_runs_detail_var.set(issues[0])
            self.messagebox.showerror("Update Planned Runs", "\n".join(issues[:12]))
            return
        try:
            payload = state_to_spec_dict(state)
            relocation_rules, _ = _relocation_options_from_spec_payload(payload)
            relocated_payload, _ = _apply_relocation_to_spec_payload(payload, relocation_rules)
            SweepExperimentSpec.from_dict(relocated_payload)
        except Exception as exc:  # noqa: BLE001
            self.planned_runs_var.set("Plan unavailable")
            self.planned_runs_detail_var.set(str(exc))
            self.messagebox.showerror("Update Planned Runs", str(exc))
            return

        self._is_previewing = True
        self.update_planned_runs_btn.configure(state="disabled")
        self.planned_runs_var.set("Updating planned runs...")
        self.planned_runs_detail_var.set("")
        self.preview_thread = threading.Thread(
            target=self._preview_worker,
            args=(payload, state.run_mode.strip().lower() or "normal"),
            daemon=True,
        )
        self.preview_thread.start()

    def _run_worker(self, payload: Dict[str, object], run_mode: str, dry_run: bool) -> None:
        self.log_queue.put({"type": "log", "message": "Preparing experiment spec..."})
        try:
            relocation_rules, adopt_relocated = _relocation_options_from_spec_payload(payload)
            relocated_payload, relocation_report = _apply_relocation_to_spec_payload(
                payload,
                relocation_rules,
            )
            spec = SweepExperimentSpec.from_dict(relocated_payload)
            extend_existing = run_mode == "extend"
            mode_label = "extend existing sweep" if extend_existing else "new/resume exact spec"
            self.log_queue.put(
                {
                    "type": "log",
                    "message": (
                        f"Running sweep with {len(spec.datasets)} dataset(s); "
                        f"mode={mode_label}; dry_run={bool(dry_run)}."
                    ),
                }
            )
            kwargs: Dict[str, object] = {
                "extend_existing": extend_existing,
                "dry_run": bool(dry_run),
            }
            if relocation_rules:
                kwargs["relocation_rules"] = relocation_rules
                kwargs["relocation_report"] = relocation_report
            if adopt_relocated:
                kwargs["adopt_relocated"] = True
            output_dir = run_sweep_experiment(
                spec,
                **kwargs,
            )
            if dry_run:
                message = "Dry run completed. No rectification tasks were executed."
            else:
                message = "Sweep completed successfully."
            self.log_queue.put(
                {
                    "type": "done",
                    "output_dir": str(output_dir),
                    "message": message,
                    "dry_run": bool(dry_run),
                }
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
            relocation_rules, _ = _relocation_options_from_spec_payload(payload)
            relocated_payload, _ = _apply_relocation_to_spec_payload(payload, relocation_rules)
            SweepExperimentSpec.from_dict(relocated_payload)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Run Sweep", str(exc))
            return
        run_mode = state.run_mode.strip().lower() or "normal"
        dry_run = bool(state.dry_run)
        output_dir = Path(state.output_dir.strip()) if state.output_dir.strip() else None
        if run_mode == "extend" and output_dir is not None and not (output_dir / RUN_STATE_FILENAME).exists():
            self.messagebox.showwarning(
                "Extend Existing Sweep",
                (
                    "The selected output directory does not contain run_state.json. "
                    "Extension is intended for an existing sweep root."
                ),
            )
        if run_mode == "extend" and not dry_run:
            confirmed = self.messagebox.askyesno(
                "Confirm Extension",
                (
                    "This will extend the existing sweep, refresh root-level reports, "
                    "and save historical products under generations/<generation_id>/products/. "
                    "Continue?"
                ),
            )
            if not confirmed:
                return
        self.status_var.set("Running...")
        self._is_running = True
        self._set_editable_enabled(False)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", self.tk.END)
        self.log_text.configure(state="disabled")
        self._append_log("Starting background sweep run...")
        self.run_thread = threading.Thread(
            target=self._run_worker,
            args=(payload, run_mode, dry_run),
            daemon=True,
        )
        self.run_thread.start()

    def _on_run_done(self, output_dir: str, message: str, dry_run: bool = False) -> None:
        self.status_var.set("Completed")
        self._set_editable_enabled(True)
        self._is_running = False
        self._append_log(message)
        if dry_run:
            dry_run_path = self._latest_dry_run_path(Path(output_dir))
            if dry_run_path is not None:
                self._append_log(f"Dry-run report: {dry_run_path}")
        self._show_completion_dialog(Path(output_dir), dry_run=dry_run)

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
                    bool(event.get("dry_run", False)),
                )
            elif event_type == "error":
                self._on_run_error(
                    str(event.get("message", "Unknown error")),
                    str(event.get("traceback", "")),
                )
            elif event_type == "preview_done":
                self._is_previewing = False
                if not self._is_running:
                    self.update_planned_runs_btn.configure(state="normal")
                preview = cast(Mapping[str, object], event.get("preview", {}))
                summary, detail = self._format_planned_runs_summary(preview)
                self.planned_runs_var.set(summary)
                self.planned_runs_detail_var.set(detail)
                self._append_log(f"Planned run preview updated. {summary}")
            elif event_type == "preview_error":
                self._is_previewing = False
                if not self._is_running:
                    self.update_planned_runs_btn.configure(state="normal")
                message = str(event.get("message", "Unknown error"))
                self.planned_runs_var.set("Plan unavailable")
                self.planned_runs_detail_var.set(message)
                self._append_log("Planned run preview failed.")
                self._append_log(message)
        self.root.after(150, self._poll_logs)

    def _latest_dry_run_path(self, output_dir: Path) -> Optional[Path]:
        candidates = sorted(
            (output_dir / "generations").glob("*/dry_run.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _show_completion_dialog(self, output_dir: Path, dry_run: bool = False) -> None:
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Dry Run Complete" if dry_run else "Sweep Complete")
        dialog.geometry("620x200")
        dry_run_path = self._latest_dry_run_path(output_dir) if dry_run else None
        message = (
            f"Dry-run report written under:\n{output_dir}"
            if dry_run
            else f"Artifacts written to:\n{output_dir}"
        )
        self.ttk.Label(
            dialog, text=message, wraplength=600
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

        def open_dry_run() -> None:
            if dry_run_path is None or not dry_run_path.exists():
                self.messagebox.showwarning("Open Dry Run", "No dry_run.json report was found.")
                return
            ok, err = _open_with_system(dry_run_path)
            if not ok:
                self.messagebox.showerror("Open Dry Run", err or "Failed to open file.")

        self.ttk.Button(button_frame, text="Open Output Folder", command=open_output).pack(side="left", padx=4)
        if dry_run:
            self.ttk.Button(button_frame, text="Open dry_run.json", command=open_dry_run).pack(side="left", padx=4)
        else:
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

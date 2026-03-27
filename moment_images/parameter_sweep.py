"""Parameter sweep runner for ``im_process`` rectification routines.

This module drives ``rectify_views_from_config`` with a configurable grid of
parameters (window size, maximum displacement, Gaussian smoothing, etc.) and
persists the results in an easy-to-compare folder layout. It is intended to run
offline so practitioners can explore trade-offs without stepping through the
interactive ``main`` workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
from moment_images.sweep_analysis import (
    analyze_summary_data,
    build_metric_record,
    save_analysis,
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
class SweepExperimentSpec:
    """Top-level experiment definition loaded from ``--spec`` JSON."""

    datasets: List[SweepDatasetSpec]
    search_spaces: Dict[str, SweepSearchSpace]
    strategies: List[str] = field(default_factory=lambda: ["pixel", "combined"])
    stages: SweepStageSpec = field(default_factory=SweepStageSpec)
    scoring: SweepScoringSpec = field(default_factory=SweepScoringSpec)
    output_dir: Optional[str] = None
    save_figures: bool = False
    analyze: bool = True
    keep_all_run_artifacts: bool = True
    random_seed: int = 0

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

        return cls(
            datasets=datasets,
            search_spaces=search_spaces,
            strategies=strategies,
            stages=stages,
            scoring=scoring,
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


def _build_candidate_from_params(strategy: str, params: Dict[str, object]) -> SweepCandidate:
    defaults = RectificationRunConfig()
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
    if not path.exists():
        raise FileNotFoundError(f"Sweep spec file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a JSON object.")
    return SweepExperimentSpec.from_dict(payload)


def _save_run_artifacts(
    output_dir: Path,
    run_name: str,
    result: RectificationResult,
) -> None:
    config = result.config
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    avg_img_path = run_dir / "avg_rectified.png"
    avg_preview = np.asarray(min_max_normalize(result.avg_image), dtype=np.uint8)
    Image.fromarray(avg_preview).save(avg_img_path)

    np.save(run_dir / "rectified_views.npy", result.rectified_views)

    pixel_displacements_x = None
    pixel_displacements_y = None
    if result.pixel_similarity is not None:
        pixel_displacements_x = getattr(result.pixel_similarity, "disps_x", None)
        pixel_displacements_y = getattr(result.pixel_similarity, "disps_y", None)

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
    }
    save_workspace_variables(str(run_dir / "workspace.npz"), workspace_data)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_dict, fp, indent=2)


def _persist_figures(figures: Dict[str, List[Figure]], base_dir: Path) -> None:
    for section, fig_list in figures.items():
        section_dir = base_dir / section
        save_figures(fig_list, str(section_dir))
        for fig in fig_list:
            plt.close(fig)


def _load_views(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Views file not found: {path}")
    views = np.load(path)
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
    np.save(cache_path, views)
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
    np.save(cache_path, exposure_views_arr)
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
        return PreparedDataset(
            dataset_id=dataset.dataset_id,
            views_path=views_path,
            hdr_enabled=bool(hdr_enabled),
            hdr_exposure_times=dataset.hdr_exposure_times,
            normalize_to_8bit=dataset.normalize_to_8bit,
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
        return PreparedDataset(
            dataset_id=dataset.dataset_id,
            views_path=views_path,
            hdr_enabled=False,
            hdr_exposure_times=None,
            normalize_to_8bit=dataset.normalize_to_8bit,
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
    return PreparedDataset(
        dataset_id=dataset.dataset_id,
        views_path=views_path,
        hdr_enabled=True,
        hdr_exposure_times=dataset.hdr_exposure_times,
        normalize_to_8bit=dataset.normalize_to_8bit,
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
        _save_run_artifacts(output_dir, run_name, result)
        if bool(task.get("save_figures", False)):
            figures = generate_rectification_figures(result)
            _persist_figures(figures, output_dir / run_name / "figures")
        metric_record = build_metric_record(run_uid, config, result)
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
    max_workers: int,
) -> List[Dict[str, object]]:
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
                }
            )
    tasks.sort(key=lambda item: (str(item["candidate_id"]), str(item["dataset_id"])))
    for idx, task in enumerate(tasks):
        task["task_index"] = idx

    if not tasks:
        return []

    stage_results: List[Dict[str, object]] = []
    progress_desc = f"Evaluating {stage_name}"
    if max_workers <= 1:
        for task in tqdm(tasks, desc=progress_desc, unit="run"):
            result = _evaluate_candidate_task(task)
            result["task_index"] = task["task_index"]
            stage_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_evaluate_candidate_task, task): task for task in tasks
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc=progress_desc, unit="run"):
                task = future_map[future]
                result = future.result()
                result["task_index"] = task["task_index"]
                stage_results.append(result)

    stage_results.sort(key=lambda item: int(item.get("task_index", 0)))
    return stage_results


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "rank",
                "candidate_id",
                "strategy",
                "score",
                "mean_dataset_score",
                "std_dataset_score",
                "missing_datasets",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row.get("rank"),
                    "candidate_id": row.get("candidate_id"),
                    "strategy": row.get("strategy"),
                    "score": row.get("score"),
                    "mean_dataset_score": row.get("mean_dataset_score"),
                    "std_dataset_score": row.get("std_dataset_score"),
                    "missing_datasets": ",".join(cast(List[str], row.get("missing_datasets", []))),
                }
            )


def _write_stage_outputs(
    stage_name: str,
    stage_results: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
    output_dir: Path,
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
    }
    summary_path = output_dir / f"{stage_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


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

    stage1_budget, stage2_budget = _stage_budgets(spec.stages)

    coarse_candidates: List[SweepCandidate] = []
    for strategy in spec.strategies:
        search_space = spec.search_spaces[strategy]
        coarse_candidates.extend(generate_coarse_candidates(search_space, budget=None))
    coarse_candidates = _limit_candidates(coarse_candidates, stage1_budget)
    coarse_candidates = _assign_candidate_ids(coarse_candidates, "s1c")
    coarse_lookup = {candidate.candidate_id: candidate for candidate in coarse_candidates}

    max_workers = _default_max_workers(spec.stages.max_workers)

    stage1_results = _evaluate_stage(
        stage_name="stage1",
        candidates=coarse_candidates,
        datasets=prepared_datasets,
        output_dir=resolved_output_dir,
        save_figures_flag=spec.save_figures,
        max_workers=max_workers,
    )
    stage1_rankings = score_and_rank_runs(stage1_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs("stage1", stage1_results, stage1_rankings, resolved_output_dir)

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
    refined_candidates = _assign_candidate_ids(refined_candidates, "s2c")

    stage2_results = _evaluate_stage(
        stage_name="stage2",
        candidates=refined_candidates,
        datasets=prepared_datasets,
        output_dir=resolved_output_dir,
        save_figures_flag=spec.save_figures,
        max_workers=max_workers,
    )
    stage2_rankings = score_and_rank_runs(stage2_results, spec.scoring, dataset_ids=dataset_ids)
    _write_stage_outputs("stage2", stage2_results, stage2_rankings, resolved_output_dir)

    final_rankings = stage2_rankings if stage2_rankings else stage1_rankings
    with (resolved_output_dir / "leaderboard.json").open("w", encoding="utf-8") as fp:
        json.dump(final_rankings, fp, indent=2)
    _write_leaderboard_csv(final_rankings, resolved_output_dir / "leaderboard.csv")

    with (resolved_output_dir / "best_configs.json").open("w", encoding="utf-8") as fp:
        json.dump(_best_configs_payload(final_rankings), fp, indent=2)

    all_metric_records: List[Dict[str, object]] = []
    for stage_result in (stage1_results + stage2_results):
        if stage_result.get("status") == "ok":
            all_metric_records.append(cast(Dict[str, object], stage_result["metric_record"]))
    with (resolved_output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(all_metric_records, fp, indent=2)

    if spec.analyze and all_metric_records:
        analysis = analyze_summary_data(all_metric_records)
        save_analysis(analysis, resolved_output_dir / "analysis.json")

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
        _save_run_artifacts(output_dir, run_name, result)

        if save_figures_flag:
            figures = generate_rectification_figures(result)  # pass full result so we can save global/pixel maps, originals, overlays, averages
            figures_dir = output_dir / run_name / "figures"
            _persist_figures(figures, figures_dir)

        record = build_metric_record(run_name, config, result)
        summary.append(record)

        if live_plot:
            if plotter is None:
                metrics_dict = record["global_and_local_metrics"]
                metric_keys = list(metrics_dict.keys()) if isinstance(metrics_dict, dict) else []
                plotter = LiveMetricsPlotter(metric_keys)
            plotter.update(run_name, record)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

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


def _launch_gui() -> None:
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    should_launch_gui = args.gui or (argv is None and len(sys.argv) == 1)
    if should_launch_gui:
        _launch_gui()
        return

    _run_from_namespace(args)


if __name__ == "__main__":
    main()

"""Typed power-blend settings, artifacts, and reusable field cache."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from .reliable_io import write_npz_atomic


@dataclass(frozen=True)
class PowerBlendConfig:
    """The coupled settings for one enabled power-field blend."""

    raised_power: float
    alpha: float
    delta_cap: float

    def __post_init__(self) -> None:
        raised_power = float(self.raised_power)
        alpha = float(self.alpha)
        delta_cap = float(self.delta_cap)
        if not np.isfinite(raised_power) or raised_power <= 1.0:
            raise ValueError("pixel_power_blend_raised_power must be finite and greater than 1.0.")
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("pixel_power_blend_alpha must be finite and between 0 and 1.")
        if not np.isfinite(delta_cap) or delta_cap < 0.0:
            raise ValueError("pixel_power_blend_delta_cap must be finite and non-negative.")
        object.__setattr__(self, "raised_power", raised_power)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "delta_cap", delta_cap)

    def flat_params(self) -> Dict[str, object]:
        return {
            "pixel_power_blend_enabled": True,
            "pixel_power_blend_raised_power": self.raised_power,
            "pixel_power_blend_alpha": self.alpha,
            "pixel_power_blend_delta_cap": self.delta_cap,
        }


@dataclass(frozen=True)
class PowerBlendSearchSpace:
    """Flat-search-space adapter that yields only valid coupled options."""

    enabled: Tuple[bool, ...]
    raised_powers: Tuple[float, ...]
    alphas: Tuple[float, ...]
    delta_caps: Tuple[float, ...]

    @classmethod
    def from_lists(
        cls,
        enabled: Iterable[bool],
        raised_powers: Iterable[float],
        alphas: Iterable[float],
        delta_caps: Iterable[float],
    ) -> "PowerBlendSearchSpace":
        return cls(
            enabled=tuple(bool(value) for value in enabled),
            raised_powers=tuple(float(value) for value in raised_powers),
            alphas=tuple(float(value) for value in alphas),
            delta_caps=tuple(float(value) for value in delta_caps),
        )

    def coarse_options(self) -> List[Optional[PowerBlendConfig]]:
        options: List[Optional[PowerBlendConfig]] = []
        for enabled in self.enabled:
            if not enabled:
                options.append(None)
                continue
            if not self.raised_powers:
                raise ValueError("Combined power blend requires pixel_power_blend_raised_powers when enabled.")
            options.extend(
                PowerBlendConfig(raised_power, alpha, delta_cap)
                for raised_power, alpha, delta_cap in product(
                    self.raised_powers, self.alphas, self.delta_caps
                )
            )
        return _dedupe(options)

    def neighbor_options(self, current: Optional[PowerBlendConfig]) -> List[Optional[PowerBlendConfig]]:
        if current is None:
            return [None]
        options = [current]
        for field, values in (
            ("raised_power", self.raised_powers),
            ("alpha", self.alphas),
            ("delta_cap", self.delta_caps),
        ):
            for value in _neighbor_values(values, getattr(current, field)):
                kwargs = {
                    "raised_power": current.raised_power,
                    "alpha": current.alpha,
                    "delta_cap": current.delta_cap,
                }
                kwargs[field] = value
                options.append(PowerBlendConfig(**kwargs))
        return _dedupe(options)

    def auto_options(
        self,
        current: Optional[PowerBlendConfig],
        extrapolation: str,
    ) -> List[Optional[PowerBlendConfig]]:
        if current is None:
            return [None]
        raised_powers = _auto_numeric_values(
            self.raised_powers,
            current.raised_power,
            extrapolation,
            lambda value: max(float(np.nextafter(1.0, np.inf)), float(value)),
        )
        alphas = _auto_numeric_values(
            self.alphas, current.alpha, extrapolation, lambda value: min(1.0, max(0.0, float(value)))
        )
        delta_caps = _auto_numeric_values(
            self.delta_caps, current.delta_cap, extrapolation, lambda value: max(0.0, float(value))
        )
        return _dedupe(
            PowerBlendConfig(raised_power, alpha, delta_cap)
            for raised_power, alpha, delta_cap in product(raised_powers, alphas, delta_caps)
        )


def _dedupe(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _neighbor_values(values: Iterable[float], current: float) -> List[float]:
    normalized = sorted(set(float(value) for value in values) | {float(current)})
    index = normalized.index(float(current))
    return [normalized[idx] for idx in sorted({max(0, index - 1), index, min(len(normalized) - 1, index + 1)})]


def _auto_numeric_values(
    values: Iterable[float], current: float, extrapolation: str, normalize: Callable[[float], float]
) -> List[float]:
    normalized = sorted(set(normalize(value) for value in values) | {normalize(current)})
    current_value = normalize(current)
    if len(normalized) < 2:
        return [current_value]
    index = normalized.index(current_value)
    generated = [current_value]
    if index > 0:
        generated.append((normalized[index - 1] + current_value) / 2.0)
    if index < len(normalized) - 1:
        generated.append((normalized[index + 1] + current_value) / 2.0)
    if extrapolation != "inside_only":
        factor = 0.5 if extrapolation == "half_step_bounded" else 1.0
        if index == 0:
            generated.append(current_value - (normalized[1] - current_value) * factor)
        if index == len(normalized) - 1:
            generated.append(current_value + (current_value - normalized[-2]) * factor)
    return sorted(set(normalize(value) for value in generated))


@dataclass
class PowerBlendFieldPair:
    """The two reusable matching fields, including their run diagnostics."""

    anchor: Any
    challenger: Any

    CACHE_VERSION = 2

    def cache_arrays(self) -> Dict[str, object]:
        return {
            "cache_version": np.array(self.CACHE_VERSION),
            "anchor_coordinates": self.anchor.coordinates,
            "anchor_scores": self.anchor.scores,
            "anchor_disps_y": self.anchor.disps_y,
            "anchor_disps_x": self.anchor.disps_x,
            "anchor_runtime_warnings_json": np.array(json.dumps(self.anchor.runtime_warnings, sort_keys=True)),
            "challenger_coordinates": self.challenger.coordinates,
            "challenger_scores": self.challenger.scores,
            "challenger_disps_y": self.challenger.disps_y,
            "challenger_disps_x": self.challenger.disps_x,
            "challenger_runtime_warnings_json": np.array(json.dumps(self.challenger.runtime_warnings, sort_keys=True)),
            "sub_pix_factor": np.array(self.anchor.sub_pix_factor),
            "reference_is_external": np.array(self.anchor.reference_is_external),
            "metric_family": np.array(self.anchor.metric_family),
            "alpha": np.array(self.anchor.alpha),
        }

    @classmethod
    def load(cls, path: Path) -> Optional["PowerBlendFieldPair"]:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as arrays:
                required = {
                    "cache_version", "anchor_coordinates", "anchor_scores", "anchor_disps_y", "anchor_disps_x",
                    "anchor_runtime_warnings_json", "challenger_coordinates", "challenger_scores",
                    "challenger_disps_y", "challenger_disps_x", "challenger_runtime_warnings_json",
                    "sub_pix_factor", "reference_is_external", "metric_family", "alpha",
                }
                if not required.issubset(arrays.files) or int(np.asarray(arrays["cache_version"]).item()) != cls.CACHE_VERSION:
                    return None
                from .rectification_strategy import PixelwiseSimilarityResult

                common = {
                    "sub_pix_factor": int(np.asarray(arrays["sub_pix_factor"]).item()),
                    "reference_is_external": bool(np.asarray(arrays["reference_is_external"]).item()),
                    "metric_family": str(np.asarray(arrays["metric_family"]).item()),
                    "alpha": float(np.asarray(arrays["alpha"]).item()),
                }
                anchor = PixelwiseSimilarityResult(
                    coordinates=np.asarray(arrays["anchor_coordinates"], dtype=np.float32),
                    scores=np.asarray(arrays["anchor_scores"], dtype=np.float64),
                    disps_y=np.asarray(arrays["anchor_disps_y"], dtype=np.float32),
                    disps_x=np.asarray(arrays["anchor_disps_x"], dtype=np.float32),
                    runtime_warnings=json.loads(str(np.asarray(arrays["anchor_runtime_warnings_json"]).item())),
                    **common,
                )
                challenger = PixelwiseSimilarityResult(
                    coordinates=np.asarray(arrays["challenger_coordinates"], dtype=np.float32),
                    scores=np.asarray(arrays["challenger_scores"], dtype=np.float64),
                    disps_y=np.asarray(arrays["challenger_disps_y"], dtype=np.float32),
                    disps_x=np.asarray(arrays["challenger_disps_x"], dtype=np.float32),
                    runtime_warnings=json.loads(str(np.asarray(arrays["challenger_runtime_warnings_json"]).item())),
                    **common,
                )
                return cls(anchor=anchor, challenger=challenger)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None


@dataclass
class PowerBlendArtifacts:
    fields: PowerBlendFieldPair
    diagnostics: Dict[str, object]
    delta_y: np.ndarray
    delta_x: np.ndarray
    delta_magnitude: np.ndarray

    def workspace_values(self) -> Dict[str, object]:
        return {
            "pixel_power_blend_diagnostics_json": np.array(json.dumps(self.diagnostics, sort_keys=True)),
            "pixel_power_blend_anchor_displacements_x": self.fields.anchor.disps_x,
            "pixel_power_blend_anchor_displacements_y": self.fields.anchor.disps_y,
            "pixel_power_blend_challenger_displacements_x": self.fields.challenger.disps_x,
            "pixel_power_blend_challenger_displacements_y": self.fields.challenger.disps_y,
            "pixel_power_blend_delta_x": self.delta_x,
            "pixel_power_blend_delta_y": self.delta_y,
            "pixel_power_blend_delta_magnitude": self.delta_magnitude,
        }


class PowerBlendFieldCache:
    """A process-safe, single-flight cache for reusable matching fields."""

    def __init__(self, cache_dir: Optional[Path], *, timeout_seconds: float = 900.0, poll_seconds: float = 0.1) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def get_or_compute(
        self,
        cache_key: Optional[str],
        compute: Callable[[], PowerBlendFieldPair],
    ) -> Tuple[PowerBlendFieldPair, bool]:
        path = self._cache_path(cache_key)
        if path is None:
            return compute(), False
        cached = PowerBlendFieldPair.load(path)
        if cached is not None:
            return cached, True
        lock_dir = Path(f"{path}.lock")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                lock_dir.mkdir(parents=True)
                (lock_dir / "owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
                break
            except FileExistsError:
                cached = PowerBlendFieldPair.load(path)
                if cached is not None:
                    return cached, True
                if time.monotonic() >= deadline:
                    stale_lock = Path(f"{lock_dir}.stale-{uuid4().hex}")
                    try:
                        os.replace(lock_dir, stale_lock)
                    except FileNotFoundError:
                        continue
                    shutil.rmtree(stale_lock, ignore_errors=True)
                    deadline = time.monotonic() + self.timeout_seconds
                    continue
                time.sleep(self.poll_seconds)
        try:
            cached = PowerBlendFieldPair.load(path)
            if cached is not None:
                return cached, True
            fields = compute()
            write_npz_atomic(path, fields.cache_arrays(), create_backup=False)
            return fields, False
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)

    def _cache_path(self, cache_key: Optional[str]) -> Optional[Path]:
        if self.cache_dir is None or not cache_key:
            return None
        safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in cache_key)
        return self.cache_dir / f"{safe_key}.npz"

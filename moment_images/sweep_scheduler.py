"""Adaptive process scheduling helpers for parameter sweeps."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple, cast

try:
    import psutil
except Exception:  # noqa: BLE001 - memory telemetry falls back to stdlib probes.
    psutil = None  # type: ignore[assignment]


MEMORY_SAFETY_FRACTION = 0.70
ADAPTIVE_COLD_START_MULTIPLIER = 1.50
ADAPTIVE_PEAK_MARGIN = 1.25
ADAPTIVE_SCHEDULER_MODEL_VERSION = 1
MEMORY_PRESSURE_POLL_SECONDS = 0.25
SCHEDULER_MODE_ADAPTIVE_MEMORY = "adaptive_memory"
SCHEDULER_MODE_FIXED_BATCH = "fixed_batch"
SCHEDULER_MODES = {
    SCHEDULER_MODE_ADAPTIVE_MEMORY,
    SCHEDULER_MODE_FIXED_BATCH,
}
MEMORY_ENFORCEMENT_AUTO = "auto"
MEMORY_ENFORCEMENT_REQUIRED = "required"
MEMORY_ENFORCEMENT_ADVISORY = "advisory"
MEMORY_ENFORCEMENT_MODES = {
    MEMORY_ENFORCEMENT_AUTO,
    MEMORY_ENFORCEMENT_REQUIRED,
    MEMORY_ENFORCEMENT_ADVISORY,
}


@dataclass
class SweepSchedulingSpec:
    """Controls process admission for sweep task execution."""

    mode: Literal["adaptive_memory", "fixed_batch"] = SCHEDULER_MODE_ADAPTIVE_MEMORY
    memory_safety_fraction: float = MEMORY_SAFETY_FRACTION
    memory_telemetry: bool = True
    worker_recycle_tasks: Optional[int] = None
    memory_enforcement: Literal["auto", "required", "advisory"] = MEMORY_ENFORCEMENT_AUTO

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ValueError(
                "stages.scheduling.mode must be either 'adaptive_memory' or 'fixed_batch'."
            )
        normalized_mode = self.mode.strip().lower()
        if normalized_mode not in SCHEDULER_MODES:
            raise ValueError(
                "stages.scheduling.mode must be either 'adaptive_memory' or 'fixed_batch'."
            )
        self.mode = cast(Literal["adaptive_memory", "fixed_batch"], normalized_mode)

        if isinstance(self.memory_safety_fraction, bool) or not isinstance(
            self.memory_safety_fraction,
            (int, float),
        ):
            raise ValueError("stages.scheduling.memory_safety_fraction must be a number in (0, 1].")
        fraction = float(self.memory_safety_fraction)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError("stages.scheduling.memory_safety_fraction must be in (0, 1].")
        self.memory_safety_fraction = fraction

        if not isinstance(self.memory_telemetry, bool):
            raise ValueError("stages.scheduling.memory_telemetry must be a boolean.")

        if not isinstance(self.memory_enforcement, str):
            raise ValueError(
                "stages.scheduling.memory_enforcement must be 'auto', 'required', or 'advisory'."
            )
        normalized_enforcement = self.memory_enforcement.strip().lower()
        if normalized_enforcement not in MEMORY_ENFORCEMENT_MODES:
            raise ValueError(
                "stages.scheduling.memory_enforcement must be 'auto', 'required', or 'advisory'."
            )
        self.memory_enforcement = cast(
            Literal["auto", "required", "advisory"],
            normalized_enforcement,
        )

        if self.worker_recycle_tasks is not None:
            if isinstance(self.worker_recycle_tasks, bool) or not isinstance(
                self.worker_recycle_tasks,
                int,
            ):
                raise ValueError(
                    "stages.scheduling.worker_recycle_tasks must be an integer >= 1 when provided."
                )
            if self.worker_recycle_tasks < 1:
                raise ValueError(
                    "stages.scheduling.worker_recycle_tasks must be >= 1 when provided."
                )

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, object]]) -> "SweepSchedulingSpec":
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("stages.scheduling must be an object when provided.")
        raw = payload or {}
        return cls(
            mode=cast(
                Literal["adaptive_memory", "fixed_batch"],
                raw.get("mode", SCHEDULER_MODE_ADAPTIVE_MEMORY),
            ),
            memory_safety_fraction=cast(
                float,
                raw.get("memory_safety_fraction", MEMORY_SAFETY_FRACTION),
            ),
            memory_telemetry=cast(bool, raw.get("memory_telemetry", True)),
            worker_recycle_tasks=cast(Optional[int], raw.get("worker_recycle_tasks")),
            memory_enforcement=cast(
                Literal["auto", "required", "advisory"],
                raw.get("memory_enforcement", MEMORY_ENFORCEMENT_AUTO),
            ),
        )


@dataclass(frozen=True)
class WorkerPlan:
    requested_workers: int
    available_memory_bytes: Optional[int]
    max_task_peak_bytes: int
    memory_cap_workers: Optional[int]
    selected_workers: int
    safety_fraction: float
    memory_safety_fraction: float
    scheduler_mode: str
    memory_budget_bytes: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PoolBatchOutcome:
    completed_results: list[dict]
    pending_tasks: list[dict]
    error: Optional[BaseException]
    telemetry: Dict[str, object]


@dataclass
class LearnedTaskProfile:
    """Observed resource data for one memory-equivalent task profile."""

    sample_count: int = 0
    max_peak_growth_bytes: int = 0
    runtime_ewma_seconds: Optional[float] = None

    def record_success(self, peak_growth_bytes: int, runtime_seconds: float) -> None:
        self.sample_count += 1
        self.max_peak_growth_bytes = max(
            int(self.max_peak_growth_bytes),
            max(0, int(peak_growth_bytes)),
        )
        runtime = max(0.0, float(runtime_seconds))
        if self.runtime_ewma_seconds is None:
            self.runtime_ewma_seconds = runtime
        else:
            self.runtime_ewma_seconds = (
                0.25 * runtime + 0.75 * float(self.runtime_ewma_seconds)
            )


def memory_profile_key(task: Mapping[str, object]) -> str:
    """Return a stable key containing only settings that materially affect memory."""

    config = task.get("config")
    config_map = config if isinstance(config, Mapping) else {}
    profile = {
        "strategy": str(task.get("strategy", config_map.get("strategy", ""))).lower(),
        "views_shape": list(task.get("views_shape", [])),
        "views_dtype": str(task.get("views_dtype", "")),
        "save_figures": bool(task.get("save_figures", False)),
        "hdr_enabled": bool(config_map.get("hdr_enabled", False)),
        "sub_pix_factor": int(config_map.get("sub_pix_factor", 1) or 1),
        "max_displacement": config_map.get("max_displacement"),
        "similarity_window_size": config_map.get("similarity_window_size"),
        "refinement_window_size": config_map.get("refinement_window_size"),
        "displacement_metric_family": config_map.get("displacement_metric_family"),
        "displacement_component_weights": config_map.get("displacement_component_weights"),
        "combined_reference_mode": config_map.get("combined_reference_mode"),
    }
    return json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class AdaptiveLearningState:
    """In-memory task profile models used to predict admission cost and runtime."""

    models: Dict[str, LearnedTaskProfile] = field(default_factory=dict)

    def model_for(self, task: Mapping[str, object]) -> Optional[LearnedTaskProfile]:
        return self.models.get(memory_profile_key(task))

    def reservation_bytes(self, task: Mapping[str, object]) -> int:
        model = self.model_for(task)
        if model is not None and model.sample_count > 0:
            return max(1, int(math.ceil(model.max_peak_growth_bytes * ADAPTIVE_PEAK_MARGIN)))
        estimated = max(1, int(task.get("estimated_peak_bytes", 1) or 1))
        return int(math.ceil(estimated * ADAPTIVE_COLD_START_MULTIPLIER))

    def runtime_seconds(self, task: Mapping[str, object]) -> float:
        model = self.model_for(task)
        if model is not None and model.runtime_ewma_seconds is not None:
            return max(0.001, float(model.runtime_ewma_seconds))
        return max(
            0.001,
            float(
                task.get("estimated_runtime_seconds", task.get("runtime_seconds", 60.0))
                or 60.0
            ),
        )

    def record_success(
        self,
        task: Mapping[str, object],
        *,
        peak_growth_bytes: int,
        runtime_seconds: float,
    ) -> None:
        key = memory_profile_key(task)
        model = self.models.setdefault(key, LearnedTaskProfile())
        model.record_success(peak_growth_bytes, runtime_seconds)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": ADAPTIVE_SCHEDULER_MODEL_VERSION,
            "profiles": {
                key: {
                    "sample_count": int(model.sample_count),
                    "max_peak_growth_bytes": int(model.max_peak_growth_bytes),
                    "runtime_ewma_seconds": model.runtime_ewma_seconds,
                }
                for key, model in self.models.items()
                if model.sample_count > 0
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "AdaptiveLearningState":
        if not isinstance(payload, Mapping):
            return cls()
        if payload.get("model_version") != ADAPTIVE_SCHEDULER_MODEL_VERSION:
            return cls()
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, Mapping):
            return cls()
        models: Dict[str, LearnedTaskProfile] = {}
        for key, raw_model in raw_profiles.items():
            if not isinstance(key, str) or not isinstance(raw_model, Mapping):
                continue
            try:
                sample_count = int(raw_model.get("sample_count", 0))
                growth = max(0, int(raw_model.get("max_peak_growth_bytes", 0)))
                runtime_raw = raw_model.get("runtime_ewma_seconds")
                runtime = None if runtime_raw is None else max(0.0, float(runtime_raw))
            except (TypeError, ValueError):
                continue
            if sample_count > 0:
                models[key] = LearnedTaskProfile(sample_count, growth, runtime)
        return cls(models=models)


def adaptive_host_memory_signature() -> Dict[str, object]:
    """Return a portable identity for deciding whether learned RSS is reusable."""

    total_memory: Optional[int] = None
    if psutil is not None:
        try:
            total_memory = int(psutil.virtual_memory().total)
        except Exception:
            total_memory = None
    if total_memory is None:
        try:
            total_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            total_memory = None
    return {
        "model_version": ADAPTIVE_SCHEDULER_MODEL_VERSION,
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
        "total_memory_bytes": total_memory,
    }


@dataclass(frozen=True)
class AdaptiveTaskDecision:
    action: Literal["submit", "wait", "reject"]
    task: Optional[dict]
    reservation_bytes: int
    reason: str


def build_memory_limit_result(task: Mapping[str, object], message: str) -> Dict[str, object]:
    """Build the canonical result for work excluded by memory containment."""

    return {
        "status": "failed",
        "stage": task.get("stage"),
        "candidate_id": task.get("candidate_id"),
        "dataset_id": task.get("dataset_id"),
        "strategy": task.get("strategy"),
        "run_uid": task.get("run_uid"),
        "runtime_seconds": 0.0,
        "config": task.get("config"),
        "params": task.get("params"),
        "candidate_signature": task.get("candidate_signature"),
        "dataset_signature": task.get("dataset_signature"),
        "run_signature": task.get("run_signature"),
        "generation_id": task.get("generation_id"),
        "task_index": task.get("task_index"),
        "run_dir": task.get("run_dir"),
        "artifact_paths": task.get("artifact_paths"),
        "failure_kind": "memory_limit_exceeded",
        "error": message,
    }


def choose_adaptive_task(
    pending_tasks: Sequence[dict],
    *,
    learning: AdaptiveLearningState,
    now_seconds: float,
    available_bytes: int,
    safety_fraction: float,
    active_unspent_growth_bytes: int,
    active_rss_bytes: int,
    memory_limit_bytes: Optional[int],
) -> AdaptiveTaskDecision:
    """Choose a fitting task, prioritizing starved work before short work."""

    live_growth_budget = max(
        0,
        int(
            math.floor(
                (int(available_bytes) + int(active_unspent_growth_bytes))
                * safety_fraction
            )
        ),
    )
    fitting: list[tuple[dict, int]] = []
    rejected: Optional[tuple[dict, int]] = None
    for task in pending_tasks:
        reservation = learning.reservation_bytes(task)
        exceeds_limit = (
            memory_limit_bytes is not None
            and int(active_rss_bytes) + int(active_unspent_growth_bytes) + reservation
            > int(memory_limit_bytes)
        )
        if exceeds_limit:
            if rejected is None:
                rejected = (task, reservation)
            continue
        if int(active_unspent_growth_bytes) + reservation <= live_growth_budget:
            fitting.append((task, reservation))

    if fitting:
        starved = [
            item
            for item in fitting
            if now_seconds - float(item[0].get("queued_at_seconds", now_seconds))
            >= 2.0 * learning.runtime_seconds(item[0])
        ]
        if starved:
            task, reservation = min(
                starved,
                key=lambda item: (
                    float(item[0].get("queued_at_seconds", now_seconds)),
                    int(item[0].get("task_index", 0)),
                ),
            )
            return AdaptiveTaskDecision("submit", task, reservation, "starvation_priority")
        task, reservation = min(
            fitting,
            key=lambda item: (
                learning.runtime_seconds(item[0]),
                int(item[0].get("task_index", 0)),
            ),
        )
        return AdaptiveTaskDecision("submit", task, reservation, "throughput_priority")

    if rejected is not None:
        task, reservation = rejected
        return AdaptiveTaskDecision("reject", task, reservation, "memory_limit_exceeded")
    if not active_unspent_growth_bytes and pending_tasks:
        return AdaptiveTaskDecision("wait", None, 0, "waiting_for_memory")
    return AdaptiveTaskDecision("wait", None, 0, "waiting_for_memory")


def get_psutil_available_memory_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def get_current_process_rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def get_macos_available_memory_bytes() -> Optional[int]:
    try:
        vm_stat_output = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
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
                except Exception:
                    page_size = 4096
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        value_token = raw_value.strip().replace(".", "")
        try:
            page_count = int(value_token)
        except Exception:
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


def get_available_memory_bytes_stdlib() -> Optional[int]:
    if sys.platform.startswith("linux"):
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        except Exception:
            return None
        values: Dict[str, int] = {}
        for line in meminfo.splitlines():
            key, _, raw_value = line.partition(":")
            if not raw_value:
                continue
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                values[key] = int(parts[0]) * 1024
            except Exception:
                continue
        if "MemAvailable" in values:
            return values["MemAvailable"]
        free = values.get("MemFree")
        buffers = values.get("Buffers", 0)
        cached = values.get("Cached", 0)
        if free is not None:
            return int(free + buffers + cached)
        return None

    if sys.platform == "darwin":
        macos_available = get_macos_available_memory_bytes()
        if macos_available is not None:
            return macos_available
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            free_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return int(page_size * free_pages)
        except Exception:
            return None

    if platform.system().lower() == "windows":
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
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None

    return None


def get_available_memory_bytes() -> Optional[int]:
    psutil_available = get_psutil_available_memory_bytes()
    if psutil_available is not None:
        return psutil_available
    return get_available_memory_bytes_stdlib()


def plan_workers(
    tasks: Sequence[Mapping[str, object]],
    requested_workers: int,
    scheduling: Optional[SweepSchedulingSpec] = None,
    available_bytes: Optional[int] = None,
) -> WorkerPlan:
    scheduling = scheduling or SweepSchedulingSpec()
    requested = int(max(1, requested_workers))
    task_count = len(tasks)
    max_task_peak_bytes = (
        max(int(task.get("estimated_peak_bytes", 0) or 0) for task in tasks)
        if tasks
        else 0
    )
    memory_budget_bytes = (
        None
        if available_bytes is None
        else int(math.floor(float(available_bytes) * scheduling.memory_safety_fraction))
    )
    memory_cap_workers: Optional[int] = None
    if memory_budget_bytes is not None and max_task_peak_bytes > 0:
        capped = int(math.floor(float(memory_budget_bytes) / float(max_task_peak_bytes)))
        memory_cap_workers = max(1, capped)

    if not tasks:
        selected_workers = 1
    else:
        selected_workers = max(1, min(requested, task_count))
        if scheduling.mode == SCHEDULER_MODE_FIXED_BATCH and memory_cap_workers is not None:
            selected_workers = max(1, min(selected_workers, memory_cap_workers))

    return WorkerPlan(
        requested_workers=requested,
        available_memory_bytes=available_bytes,
        max_task_peak_bytes=int(max_task_peak_bytes),
        memory_cap_workers=memory_cap_workers,
        selected_workers=int(selected_workers),
        safety_fraction=float(scheduling.memory_safety_fraction),
        memory_safety_fraction=float(scheduling.memory_safety_fraction),
        scheduler_mode=scheduling.mode,
        memory_budget_bytes=memory_budget_bytes,
    )


def _executor_supports_max_tasks_per_child(executor_factory: object) -> bool:
    try:
        import inspect

        parameters = inspect.signature(executor_factory).parameters
    except Exception:
        return True
    return "max_tasks_per_child" in parameters or any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def process_pool_kwargs(
    workers: int,
    worker_recycle_tasks: Optional[int] = None,
    *,
    initializer: Optional[Callable[[], None]] = None,
    executor_factory: object = ProcessPoolExecutor,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"max_workers": int(workers)}
    if initializer is not None:
        kwargs["initializer"] = initializer
    if worker_recycle_tasks is not None:
        if not _executor_supports_max_tasks_per_child(executor_factory):
            raise RuntimeError(
                "stages.scheduling.worker_recycle_tasks requires a Python runtime whose "
                "ProcessPoolExecutor supports max_tasks_per_child."
            )
        kwargs["max_tasks_per_child"] = int(worker_recycle_tasks)
    return kwargs


def _pool_startup_probe() -> None:
    """Give ProcessPoolExecutor a harmless unit of work to start every worker."""

    time.sleep(0.05)


def execute_pool_attempt(
    tasks: Sequence[dict],
    workers: int,
    scheduling: SweepSchedulingSpec,
    worker_fn: Callable[[dict], dict],
    initializer: Optional[Callable[[], None]],
    on_result: Optional[Callable[[dict, dict], None]] = None,
    *,
    memory_probe: Callable[[], Optional[int]] = get_available_memory_bytes,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    wait_fn: Callable[..., Tuple[Sequence[object], Sequence[object]]] = wait,
    sleep_fn: Callable[[float], None] = time.sleep,
    learning: Optional[AdaptiveLearningState] = None,
    memory_limit_bytes: Optional[int] = None,
    worker_started_callback: Optional[Callable[[int], None]] = None,
) -> PoolBatchOutcome:
    if scheduling.mode != SCHEDULER_MODE_ADAPTIVE_MEMORY:
        raise ValueError("execute_pool_attempt only supports adaptive_memory scheduling.")

    completed_results: list[dict] = []
    completed_indices: set[int] = set()
    pending_tasks = list(tasks)
    executor_exc: Optional[BaseException] = None
    callback_exc: Optional[BaseException] = None
    active_tasks: Dict[object, dict] = {}
    active_reserved_bytes: Dict[object, int] = {}
    learning = learning or AdaptiveLearningState()
    queued_at_seconds = time.monotonic()
    for task in pending_tasks:
        task.setdefault("queued_at_seconds", queued_at_seconds)
    epoch_budget_bytes: Optional[int] = None
    telemetry: Dict[str, object] = {
        "scheduler_mode": SCHEDULER_MODE_ADAPTIVE_MEMORY,
        "memory_budget_bytes": None,
        "min_memory_budget_bytes_observed": None,
        "max_memory_budget_bytes_observed": None,
        "max_concurrent_tasks_observed": 0,
        "max_reserved_peak_bytes_observed": 0,
        "min_available_memory_bytes_observed": None,
        "memory_telemetry_enabled": bool(scheduling.memory_telemetry),
        "worker_recycle_tasks": scheduling.worker_recycle_tasks,
        "learned_profile_count": 0,
        "cold_profile_count": 0,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_pressure_wait_count": 0,
        "memory_pressure_wait_seconds": 0.0,
    }

    def task_estimate(task: Mapping[str, object]) -> int:
        return learning.reservation_bytes(task)

    def reserved_total() -> int:
        return int(sum(active_reserved_bytes.values()))

    def record_budget(budget: Optional[int]) -> None:
        if budget is None:
            return
        if telemetry["memory_budget_bytes"] is None:
            telemetry["memory_budget_bytes"] = int(budget)
        current_min = telemetry["min_memory_budget_bytes_observed"]
        current_max = telemetry["max_memory_budget_bytes_observed"]
        telemetry["min_memory_budget_bytes_observed"] = (
            int(budget) if current_min is None else min(int(current_min), int(budget))
        )
        telemetry["max_memory_budget_bytes_observed"] = (
            int(budget) if current_max is None else max(int(current_max), int(budget))
        )

    def observe_available_memory() -> Optional[int]:
        available = memory_probe()
        if available is not None and bool(scheduling.memory_telemetry):
            current_min = telemetry["min_available_memory_bytes_observed"]
            telemetry["min_available_memory_bytes_observed"] = (
                int(available)
                if current_min is None
                else min(int(current_min), int(available))
            )
        return available

    def budget_from_available(available: Optional[int], reserved: int = 0) -> Optional[int]:
        if available is None:
            return None
        budget = int(
            math.floor(
                float(int(available) + int(reserved))
                * float(scheduling.memory_safety_fraction)
            )
        )
        record_budget(budget)
        return budget

    def effective_budget() -> Optional[int]:
        nonlocal epoch_budget_bytes
        if not active_tasks:
            epoch_budget_bytes = budget_from_available(observe_available_memory())
            return epoch_budget_bytes
        if epoch_budget_bytes is None:
            observe_available_memory()
            return None
        pressure_budget = budget_from_available(observe_available_memory(), reserved_total())
        if pressure_budget is None:
            return epoch_budget_bytes
        return min(epoch_budget_bytes, pressure_budget)

    def update_concurrency_telemetry() -> None:
        telemetry["max_concurrent_tasks_observed"] = max(
            int(telemetry["max_concurrent_tasks_observed"]),
            len(active_tasks),
        )
        telemetry["max_reserved_peak_bytes_observed"] = max(
            int(telemetry["max_reserved_peak_bytes_observed"]),
            reserved_total(),
        )

    def submit_pending(executor: object, pending_index: int) -> None:
        task = pending_tasks.pop(pending_index)
        future = executor.submit(worker_fn, task)  # type: ignore[attr-defined]
        active_tasks[future] = task
        active_reserved_bytes[future] = task_estimate(task)
        update_concurrency_telemetry()

    def admit_ready_tasks(executor: object) -> bool:
        waiting_for_memory = False
        while pending_tasks and len(active_tasks) < int(workers):
            budget = effective_budget()
            current_reserved = reserved_total()
            selected_index: Optional[int] = None
            choice: Optional[AdaptiveTaskDecision] = None
            if budget is None:
                if active_tasks:
                    break
                selected_index = 0
            else:
                available = observe_available_memory()
                if available is None:
                    if active_tasks:
                        break
                    selected_index = 0
                else:
                    choice = choose_adaptive_task(
                        pending_tasks,
                        learning=learning,
                        now_seconds=time.monotonic(),
                        available_bytes=int(available),
                        safety_fraction=scheduling.memory_safety_fraction,
                        active_unspent_growth_bytes=current_reserved,
                        active_rss_bytes=0,
                        memory_limit_bytes=memory_limit_bytes,
                    )
                    if choice.action == "reject" and choice.task is not None:
                        task = choice.task
                        pending_tasks.remove(task)
                        result = build_memory_limit_result(
                            task,
                            "Predicted task growth exceeds the configured aggregate memory limit "
                            f"({choice.reservation_bytes} bytes reservation; "
                            f"{memory_limit_bytes} bytes limit).",
                        )
                        completed_results.append(result)
                        completed_indices.add(int(task["task_index"]))
                        if on_result is not None:
                            on_result(task, result)
                        continue
                    if (
                        choice.action == "submit"
                        and choice.task is not None
                        and current_reserved + choice.reservation_bytes <= budget
                    ):
                        selected_index = pending_tasks.index(choice.task)
            if selected_index is None:
                waiting_for_memory = choice is not None and choice.action == "wait"
                break
            submit_pending(executor, selected_index)
        return waiting_for_memory

    def unfinished_tasks() -> list[dict]:
        return [
            task
            for task in tasks
            if int(task["task_index"]) not in completed_indices
        ]

    def cancel_active_futures() -> None:
        for future in list(active_tasks):
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass

    def record_memory_pressure_wait() -> None:
        sleep_fn(MEMORY_PRESSURE_POLL_SECONDS)
        telemetry["memory_pressure_wait_count"] = (
            int(telemetry["memory_pressure_wait_count"]) + 1
        )
        telemetry["memory_pressure_wait_seconds"] = (
            float(telemetry["memory_pressure_wait_seconds"])
            + MEMORY_PRESSURE_POLL_SECONDS
        )

    try:
        kwargs = process_pool_kwargs(
            workers,
            scheduling.worker_recycle_tasks,
            initializer=initializer,
            executor_factory=executor_factory,
        )
        with executor_factory(**kwargs) as executor:
            if worker_started_callback is not None:
                probes = [executor.submit(_pool_startup_probe) for _ in range(int(workers))]  # type: ignore[attr-defined]
                for probe in probes:
                    probe.result()  # type: ignore[attr-defined]
                processes = getattr(executor, "_processes", {})
                process_values = (
                    processes.values() if isinstance(processes, Mapping) else ()
                )
                pids = sorted(
                    {
                        int(pid)
                        for process in process_values
                        if (pid := getattr(process, "pid", None)) is not None and int(pid) > 0
                    }
                )
                if not pids:
                    raise RuntimeError("ProcessPoolExecutor did not expose started worker PIDs")
                for pid in pids:
                    worker_started_callback(pid)
            while active_tasks or pending_tasks:
                if not active_tasks and pending_tasks:
                    waiting_for_memory = admit_ready_tasks(executor)
                    if not active_tasks:
                        if waiting_for_memory:
                            record_memory_pressure_wait()
                        continue
                done_futures, _ = wait_fn(list(active_tasks.keys()), return_when=FIRST_COMPLETED)
                for future in list(done_futures):
                    task = active_tasks.pop(future)
                    active_reserved_bytes.pop(future, None)
                    try:
                        result = future.result()  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001 - caller degrades worker count.
                        if executor_exc is None:
                            executor_exc = exc
                        continue
                    result["task_index"] = task["task_index"]
                    completed_results.append(result)
                    completed_indices.add(int(task["task_index"]))
                    if on_result is not None:
                        try:
                            on_result(task, result)
                        except Exception as exc:  # noqa: BLE001 - persistence errors are fatal.
                            callback_exc = exc
                            raise
                if executor_exc is not None:
                    cancel_active_futures()
                    break
                waiting_for_memory = admit_ready_tasks(executor)
                if not active_tasks and pending_tasks and waiting_for_memory:
                    record_memory_pressure_wait()

            pending_tasks = [] if executor_exc is None else unfinished_tasks()
    except Exception as exc:  # noqa: BLE001 - covers pool init/spawn failures.
        if callback_exc is not None and exc is callback_exc:
            raise
        executor_exc = exc
        pending_tasks = unfinished_tasks()

    return PoolBatchOutcome(
        completed_results=completed_results,
        pending_tasks=pending_tasks,
        error=executor_exc,
        telemetry={
            **telemetry,
            "learned_profile_count": sum(
                1 for model in learning.models.values() if model.sample_count > 0
            ),
            "cold_profile_count": sum(
                1 for task in tasks if learning.model_for(task) is None
            ),
        },
    )

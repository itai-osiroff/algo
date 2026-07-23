"""Adaptive process scheduling helpers for parameter sweeps."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple, cast

try:
    import psutil
except Exception:  # noqa: BLE001 - memory telemetry falls back to stdlib probes.
    psutil = None  # type: ignore[assignment]


MEMORY_SAFETY_FRACTION = 0.70
SCHEDULER_MODE_ADAPTIVE_MEMORY = "adaptive_memory"
SCHEDULER_MODE_FIXED_BATCH = "fixed_batch"
SCHEDULER_MODES = {
    SCHEDULER_MODE_ADAPTIVE_MEMORY,
    SCHEDULER_MODE_FIXED_BATCH,
}


@dataclass
class SweepSchedulingSpec:
    """Controls process admission for sweep task execution."""

    mode: Literal["adaptive_memory", "fixed_batch"] = SCHEDULER_MODE_ADAPTIVE_MEMORY
    memory_safety_fraction: float = MEMORY_SAFETY_FRACTION
    memory_telemetry: bool = True
    worker_recycle_tasks: Optional[int] = None

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
    }

    def task_estimate(task: Mapping[str, object]) -> int:
        return max(1, int(task.get("estimated_peak_bytes", 1) or 1))

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

    def admit_ready_tasks(executor: object) -> None:
        while pending_tasks and len(active_tasks) < int(workers):
            budget = effective_budget()
            current_reserved = reserved_total()
            selected_index: Optional[int] = None
            if budget is None:
                if active_tasks:
                    break
                selected_index = 0
            else:
                for idx, task in enumerate(pending_tasks):
                    if current_reserved + task_estimate(task) <= budget:
                        selected_index = idx
                        break
                if selected_index is None and not active_tasks:
                    selected_index = 0
            if selected_index is None:
                break
            submit_pending(executor, selected_index)

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

    try:
        kwargs = process_pool_kwargs(
            workers,
            scheduling.worker_recycle_tasks,
            initializer=initializer,
            executor_factory=executor_factory,
        )
        with executor_factory(**kwargs) as executor:
            admit_ready_tasks(executor)
            while active_tasks or pending_tasks:
                if not active_tasks and pending_tasks:
                    admit_ready_tasks(executor)
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
                admit_ready_tasks(executor)

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
        telemetry=telemetry,
    )

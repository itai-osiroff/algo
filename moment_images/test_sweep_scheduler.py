from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from typing import Callable

import pytest

import moment_images.parameter_sweep as parameter_sweep
from moment_images.parameter_sweep import SweepSchedulingSpec
from moment_images.sweep_scheduler import (
    AdaptiveLearningState,
    choose_adaptive_task,
    memory_profile_key,
)


def _scheduler_task(run_uid: str, estimated_peak_bytes: int, task_index: int) -> dict:
    return {
        "stage": "stage1",
        "candidate_id": run_uid,
        "dataset_id": "d1",
        "strategy": "pixel",
        "run_uid": run_uid,
        "runtime_seconds": 0.01,
        "config": {"strategy": "pixel"},
        "params": {},
        "task_index": task_index,
        "estimated_peak_bytes": int(estimated_peak_bytes),
    }


def _build_eval_result(task: dict, status: str = "ok") -> dict:
    payload = {
        "status": status,
        "stage": task["stage"],
        "candidate_id": task["candidate_id"],
        "dataset_id": task["dataset_id"],
        "strategy": task["strategy"],
        "run_uid": task["run_uid"],
        "runtime_seconds": 0.01,
        "config": task["config"],
        "params": task["params"],
        "task_index": task["task_index"],
    }
    if status == "ok":
        payload["metric_record"] = {
            "global_only_metrics": None,
            "global_and_local_metrics": {"laplacian_variance": 1.0},
        }
    else:
        payload["error"] = "worker crashed"
    return payload


class _FakeFuture:
    def __init__(self, task: dict, result_factory: Callable[[dict], dict]):
        self.task = task
        self._result_factory = result_factory

    def result(self) -> dict:
        return self._result_factory(self.task)


class _FakeExecutor:
    def __init__(
        self,
        max_workers: int,
        initializer=None,
        max_tasks_per_child=None,
        *,
        result_factory: Callable[[dict], dict],
        submitted: list[str],
    ) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.max_tasks_per_child = max_tasks_per_child
        self._result_factory = result_factory
        self._submitted = submitted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, func, task: dict) -> _FakeFuture:
        self._submitted.append(str(task["run_uid"]))
        return _FakeFuture(task, self._result_factory)


def _install_fake_executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_factory: Callable[[dict], dict] | None = None,
) -> list[str]:
    submitted: list[str] = []

    def factory(max_workers, initializer=None, max_tasks_per_child=None):
        return _FakeExecutor(
            max_workers,
            initializer=initializer,
            max_tasks_per_child=max_tasks_per_child,
            result_factory=result_factory or _build_eval_result,
            submitted=submitted,
        )

    monkeypatch.setattr(parameter_sweep, "ProcessPoolExecutor", factory)
    return submitted


def test_adaptive_scheduler_does_not_inflate_budget_while_active(monkeypatch):
    tasks = [
        _scheduler_task("large", 400, 0),
        _scheduler_task("medium", 300, 1),
        _scheduler_task("small", 100, 2),
    ]
    wait_snapshots: list[list[str]] = []
    completion_order = ["small", "large", "medium"]

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 1000)
    _install_fake_executor(monkeypatch)

    def fake_wait(futures, return_when):
        active = list(futures)
        wait_snapshots.append([str(future.task["run_uid"]) for future in active])
        for run_uid in completion_order:
            for future in active:
                if future.task["run_uid"] == run_uid:
                    return [future], [item for item in active if item is not future]
        raise AssertionError("No matching fake future to complete")

    monkeypatch.setattr(parameter_sweep, "wait", fake_wait)

    completed, pending, error, telemetry = parameter_sweep._execute_adaptive_pool_batch(
        tasks,
        workers=3,
        memory_safety_fraction=0.7,
        worker_recycle_tasks=None,
        memory_telemetry=True,
        on_result=None,
    )

    assert error is None
    assert pending == []
    assert [result["run_uid"] for result in completed] == ["large", "small", "medium"]
    assert wait_snapshots == [["large"], ["medium", "small"], ["medium"]]
    assert telemetry["max_concurrent_tasks_observed"] == 2
    assert telemetry["max_reserved_peak_bytes_observed"] == 600


def test_adaptive_scheduler_serializes_when_memory_probe_unknown(monkeypatch):
    tasks = [
        _scheduler_task("a", 100, 0),
        _scheduler_task("b", 100, 1),
        _scheduler_task("c", 100, 2),
    ]
    wait_snapshots: list[list[str]] = []

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: None)
    _install_fake_executor(monkeypatch)

    def fake_wait(futures, return_when):
        active = list(futures)
        wait_snapshots.append([str(future.task["run_uid"]) for future in active])
        return [active[0]], active[1:]

    monkeypatch.setattr(parameter_sweep, "wait", fake_wait)

    completed, pending, error, telemetry = parameter_sweep._execute_adaptive_pool_batch(
        tasks,
        workers=3,
        memory_safety_fraction=0.7,
        worker_recycle_tasks=None,
        memory_telemetry=True,
        on_result=None,
    )

    assert error is None
    assert pending == []
    assert [result["run_uid"] for result in completed] == ["a", "b", "c"]
    assert wait_snapshots == [["a"], ["b"], ["c"]]
    assert telemetry["memory_budget_bytes"] is None
    assert telemetry["max_concurrent_tasks_observed"] == 1


def test_adaptive_scheduler_commits_other_done_successes_before_pool_retry(monkeypatch):
    tasks = [
        _scheduler_task("fail", 100, 0),
        _scheduler_task("ok", 100, 1),
        _scheduler_task("later", 100, 2),
    ]
    committed: list[str] = []

    monkeypatch.setattr(parameter_sweep, "_get_available_memory_bytes", lambda: 1000)

    def result_factory(task: dict) -> dict:
        if task["run_uid"] == "fail":
            raise BrokenProcessPool("pool crashed")
        return _build_eval_result(task)

    _install_fake_executor(monkeypatch, result_factory=result_factory)

    def fake_wait(futures, return_when):
        active = list(futures)
        by_uid = {future.task["run_uid"]: future for future in active}
        return [by_uid["fail"], by_uid["ok"]], [
            future for future in active if future.task["run_uid"] not in {"fail", "ok"}
        ]

    monkeypatch.setattr(parameter_sweep, "wait", fake_wait)

    completed, pending, error, telemetry = parameter_sweep._execute_adaptive_pool_batch(
        tasks,
        workers=3,
        memory_safety_fraction=0.7,
        worker_recycle_tasks=None,
        memory_telemetry=True,
        on_result=lambda task, _result: committed.append(str(task["run_uid"])),
    )

    assert isinstance(error, BrokenProcessPool)
    assert [result["run_uid"] for result in completed] == ["ok"]
    assert committed == ["ok"]
    assert [task["run_uid"] for task in pending] == ["fail", "later"]
    assert telemetry["max_concurrent_tasks_observed"] == 3


def test_adaptive_worker_plan_keeps_process_cap_when_memory_unknown():
    tasks = [{"estimated_peak_bytes": 100}, {"estimated_peak_bytes": 100}]

    adaptive = parameter_sweep._select_stage_worker_count(
        tasks,
        requested_workers=8,
        available_memory_bytes=None,
        scheduling=SweepSchedulingSpec(mode="adaptive_memory"),
    )
    fixed = parameter_sweep._select_stage_worker_count(
        tasks,
        requested_workers=8,
        available_memory_bytes=None,
        scheduling=SweepSchedulingSpec(mode="fixed_batch"),
    )

    assert adaptive["selected_workers"] == 2
    assert adaptive["scheduler_mode"] == "adaptive_memory"
    assert fixed["selected_workers"] == 2
    assert fixed["scheduler_mode"] == "fixed_batch"


def test_learning_state_replaces_cold_estimate_with_observed_peak_growth():
    task = _scheduler_task("pixel", 1_000, 0)
    task["views_shape"] = [2, 2, 16, 16]
    task["views_dtype"] = "float32"
    task["save_figures"] = False
    learning = AdaptiveLearningState()

    assert learning.reservation_bytes(task) == 1_500

    learning.record_success(task, peak_growth_bytes=800, runtime_seconds=4.0)

    assert learning.reservation_bytes(task) == 1_000
    model = learning.models[memory_profile_key(task)]
    assert model.sample_count == 1
    assert model.runtime_ewma_seconds == 4.0


def test_choose_adaptive_task_promotes_starved_task_over_shorter_work():
    long_waiting = _scheduler_task("long", 100, 0)
    short_waiting = _scheduler_task("short", 100, 1)
    for task in (long_waiting, short_waiting):
        task["views_shape"] = [2, 2, 16, 16]
        task["views_dtype"] = "float32"
        task["save_figures"] = False
    long_waiting["queued_at_seconds"] = 0.0
    short_waiting["queued_at_seconds"] = 10.0
    learning = AdaptiveLearningState()
    learning.record_success(long_waiting, peak_growth_bytes=50, runtime_seconds=5.0)
    learning.record_success(short_waiting, peak_growth_bytes=50, runtime_seconds=1.0)

    choice = choose_adaptive_task(
        [long_waiting, short_waiting],
        learning=learning,
        now_seconds=11.0,
        available_bytes=1_000,
        safety_fraction=0.7,
        active_unspent_growth_bytes=0,
        active_rss_bytes=0,
        memory_limit_bytes=1_000,
    )

    assert choice.task is long_waiting
    assert choice.reason == "starvation_priority"


def test_choose_adaptive_task_rejects_predicted_oversize_task():
    task = _scheduler_task("oversize", 1_000, 0)
    task["views_shape"] = [2, 2, 16, 16]
    task["views_dtype"] = "float32"
    task["save_figures"] = False

    choice = choose_adaptive_task(
        [task],
        learning=AdaptiveLearningState(),
        now_seconds=0.0,
        available_bytes=500,
        safety_fraction=0.7,
        active_unspent_growth_bytes=0,
        active_rss_bytes=0,
        memory_limit_bytes=500,
    )

    assert choice.task is None
    assert choice.rejected_task is task
    assert choice.reason == "memory_limit_exceeded"

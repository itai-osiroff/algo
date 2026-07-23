from __future__ import annotations

from pathlib import Path

import pytest


class _TempCgroupFilesystem:
    """A temp-directory cgroup model that supplies child control files."""

    def __init__(self, *, allow_writes: bool = True) -> None:
        self.allow_writes = allow_writes

    def path(self, value: str) -> Path:
        return Path(value)

    def is_writable(self, path: Path) -> bool:
        return self.allow_writes and path.exists()

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, value: str) -> None:
        path.write_text(value, encoding="utf-8")

    def mkdir(self, path: Path) -> None:
        path.mkdir()
        for name, value in {
            "memory.high": "max\n",
            "memory.max": "max\n",
            "memory.current": "0\n",
            "memory.events.local": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
            "cgroup.procs": "",
        }.items():
            (path / name).write_text(value, encoding="utf-8")

    def remove_child(self, path: Path) -> bool:
        for item in path.iterdir():
            item.unlink()
        path.rmdir()
        return True


class _FakeWindowsJobApi:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.created_with: list[int] = []
        self.assigned: list[tuple[object, int]] = []
        self.closed: list[object] = []

    def create_job(self, limit_bytes: int) -> object:
        if self.error is not None:
            raise self.error
        self.created_with.append(limit_bytes)
        return "job-handle"

    def assign_process(self, job: object, pid: int) -> None:
        self.assigned.append((job, pid))

    def close_job(self, job: object) -> None:
        self.closed.append(job)


def _cgroup_root(tmp_path: Path, *, controllers: str = "cpu memory io\n") -> Path:
    root = tmp_path / "delegated"
    root.mkdir()
    (root / "cgroup.controllers").write_text(controllers, encoding="utf-8")
    (root / "cgroup.subtree_control").write_text("cpu\n", encoding="utf-8")
    return root


def test_linux_enforcer_sets_limits_tracks_usage_and_cleans_empty_child(tmp_path):
    from moment_images.memory_enforcement import create_memory_enforcer

    root = _cgroup_root(tmp_path)
    enforcer = create_memory_enforcer(
        "required",
        1_000,
        platform_name="linux",
        environ={"MOMENT_IMAGES_CGROUP_ROOT": str(root)},
        linux_fs=_TempCgroupFilesystem(),
    )

    child = next(root.glob("moment-images-*"))
    assert enforcer.status.mode == "enforced"
    assert enforcer.status.backend == "linux-cgroup-v2"
    assert enforcer.status.limit_bytes == 1_000
    assert (root / "cgroup.subtree_control").read_text(encoding="utf-8") == "+memory\n"
    assert (child / "memory.high").read_text(encoding="utf-8") == "900\n"
    assert (child / "memory.max").read_text(encoding="utf-8") == "1000\n"

    (child / "memory.current").write_text("321\n", encoding="utf-8")
    enforcer.add_worker(1234)
    assert (child / "cgroup.procs").read_text(encoding="utf-8") == "1234\n"
    assert enforcer.current_bytes() == 321

    (child / "cgroup.procs").write_text("", encoding="utf-8")
    enforcer.close()
    assert not child.exists()


def test_linux_enforcer_reads_local_events_and_falls_back_to_global_events(tmp_path):
    from moment_images.memory_enforcement import create_memory_enforcer

    root = _cgroup_root(tmp_path)
    enforcer = create_memory_enforcer(
        "auto",
        1_000,
        platform_name="linux",
        environ={"MOMENT_IMAGES_CGROUP_ROOT": str(root)},
        linux_fs=_TempCgroupFilesystem(),
    )
    child = next(root.glob("moment-images-*"))
    (child / "memory.events.local").write_text("high 2\noom 1\noom_kill 1\n", encoding="utf-8")

    assert enforcer.memory_events() == {"high": 2, "oom": 1, "oom_kill": 1}

    (child / "memory.events.local").unlink()
    (child / "memory.events").write_text("high 3\noom 0\n", encoding="utf-8")
    assert enforcer.memory_events() == {"high": 3, "oom": 0}


def test_linux_auto_falls_back_to_advisory_when_no_delegated_root(tmp_path):
    from moment_images.memory_enforcement import create_memory_enforcer

    enforcer = create_memory_enforcer(
        "auto",
        1_000,
        platform_name="linux",
        environ={},
        linux_fs=_TempCgroupFilesystem(),
    )

    assert enforcer.status.mode == "advisory"
    assert enforcer.status.backend == "linux-cgroup-v2"
    assert "MOMENT_IMAGES_CGROUP_ROOT" in enforcer.status.reason
    assert enforcer.current_bytes() is None
    assert enforcer.memory_events() == {}


def test_linux_required_fails_when_memory_controller_is_not_delegated(tmp_path):
    from moment_images.memory_enforcement import create_memory_enforcer

    root = _cgroup_root(tmp_path, controllers="cpu io\n")
    with pytest.raises(RuntimeError, match="memory controller"):
        create_memory_enforcer(
            "required",
            1_000,
            platform_name="linux",
            environ={"MOMENT_IMAGES_CGROUP_ROOT": str(root)},
            linux_fs=_TempCgroupFilesystem(),
        )


def test_advisory_policy_never_attempts_platform_enforcement(tmp_path):
    from moment_images.memory_enforcement import create_memory_enforcer

    root = _cgroup_root(tmp_path)
    enforcer = create_memory_enforcer(
        "advisory",
        1_000,
        platform_name="linux",
        environ={"MOMENT_IMAGES_CGROUP_ROOT": str(root)},
        linux_fs=_TempCgroupFilesystem(),
    )

    assert enforcer.status.mode == "advisory"
    assert list(root.glob("moment-images-*")) == []


def test_windows_uses_injected_job_api_and_closes_the_job():
    from moment_images.memory_enforcement import create_memory_enforcer

    api = _FakeWindowsJobApi()
    enforcer = create_memory_enforcer(
        "auto",
        2_000,
        platform_name="win32",
        windows_api=api,
    )

    assert enforcer.status.mode == "enforced"
    assert enforcer.status.backend == "windows-job-object"
    assert api.created_with == [2_000]
    enforcer.add_worker(777)
    enforcer.close()
    assert api.assigned == [("job-handle", 777)]
    assert api.closed == ["job-handle"]


def test_windows_auto_falls_back_and_required_raises_when_job_creation_fails():
    from moment_images.memory_enforcement import create_memory_enforcer

    api = _FakeWindowsJobApi(error=OSError("job creation unavailable"))
    advisory = create_memory_enforcer(
        "auto",
        2_000,
        platform_name="win32",
        windows_api=api,
    )
    assert advisory.status.mode == "advisory"
    assert "job creation unavailable" in advisory.status.reason

    with pytest.raises(RuntimeError, match="job creation unavailable"):
        create_memory_enforcer(
            "required",
            2_000,
            platform_name="win32",
            windows_api=_FakeWindowsJobApi(error=OSError("job creation unavailable")),
        )

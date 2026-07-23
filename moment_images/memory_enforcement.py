"""Cross-platform worker-pool memory containment adapters.

Linux uses a caller-delegated cgroup v2 root. Windows uses a Job Object when
the standard-library ctypes adapter can create one. The module intentionally
does not discover or alter a host-wide cgroup hierarchy.
"""

from __future__ import annotations

import ctypes
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol, runtime_checkable


MemoryEnforcementPolicy = Literal["auto", "required", "advisory"]


@dataclass(frozen=True)
class MemoryEnforcementStatus:
    mode: Literal["enforced", "advisory"]
    backend: str
    reason: str
    limit_bytes: int | None


@runtime_checkable
class MemoryEnforcer(Protocol):
    @property
    def status(self) -> MemoryEnforcementStatus: ...

    def add_worker(self, pid: int) -> None: ...

    def current_bytes(self) -> int | None: ...

    def memory_events(self) -> dict[str, int]: ...

    def close(self) -> None: ...


@runtime_checkable
class LinuxCgroupFilesystem(Protocol):
    def path(self, value: str) -> Path: ...

    def is_writable(self, path: Path) -> bool: ...

    def read_text(self, path: Path) -> str: ...

    def write_text(self, path: Path, value: str) -> None: ...

    def mkdir(self, path: Path) -> None: ...

    def remove_child(self, path: Path) -> bool: ...


@runtime_checkable
class WindowsJobApi(Protocol):
    def create_job(self, limit_bytes: int) -> object: ...

    def assign_process(self, job: object, pid: int) -> None: ...

    def close_job(self, job: object) -> None: ...


class _EnforcementUnavailable(RuntimeError):
    pass


class _PathLinuxCgroupFilesystem:
    def path(self, value: str) -> Path:
        return Path(value)

    def is_writable(self, path: Path) -> bool:
        return os.access(path, os.W_OK)

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, value: str) -> None:
        path.write_text(value, encoding="utf-8")

    def mkdir(self, path: Path) -> None:
        path.mkdir()

    def remove_child(self, path: Path) -> bool:
        try:
            path.rmdir()
        except OSError:
            return False
        return True


class _AdvisoryMemoryEnforcer:
    def __init__(self, backend: str, reason: str) -> None:
        self._status = MemoryEnforcementStatus(
            mode="advisory",
            backend=backend,
            reason=reason,
            limit_bytes=None,
        )

    @property
    def status(self) -> MemoryEnforcementStatus:
        return self._status

    def add_worker(self, pid: int) -> None:
        del pid

    def current_bytes(self) -> int | None:
        return None

    def memory_events(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        return None


class _LinuxCgroupMemoryEnforcer:
    def __init__(self, root: Path, filesystem: LinuxCgroupFilesystem, limit_bytes: int) -> None:
        self._filesystem = filesystem
        self._root = root
        self._child = root / f"moment-images-{os.getpid()}-{uuid.uuid4().hex}"
        self._closed = False

        self._enable_memory_controller()
        try:
            self._filesystem.mkdir(self._child)
            self._filesystem.write_text(self._child / "memory.high", f"{limit_bytes * 9 // 10}\n")
            self._filesystem.write_text(self._child / "memory.max", f"{limit_bytes}\n")
        except Exception:
            self._filesystem.remove_child(self._child)
            raise

        self._status = MemoryEnforcementStatus(
            mode="enforced",
            backend="linux-cgroup-v2",
            reason=f"worker cgroup created at {self._child}",
            limit_bytes=limit_bytes,
        )

    @property
    def status(self) -> MemoryEnforcementStatus:
        return self._status

    def _enable_memory_controller(self) -> None:
        subtree_control = self._root / "cgroup.subtree_control"
        enabled = set(self._filesystem.read_text(subtree_control).split())
        if "memory" not in enabled:
            self._filesystem.write_text(subtree_control, "+memory\n")

    def add_worker(self, pid: int) -> None:
        if self._closed:
            raise RuntimeError("Memory enforcer is already closed")
        if pid <= 0:
            raise ValueError("Worker pid must be a positive integer")
        self._filesystem.write_text(self._child / "cgroup.procs", f"{pid}\n")

    def current_bytes(self) -> int | None:
        return _read_int_file(self._filesystem, self._child / "memory.current")

    def memory_events(self) -> dict[str, int]:
        local_events = self._child / "memory.events.local"
        try:
            text = self._filesystem.read_text(local_events)
        except OSError:
            try:
                text = self._filesystem.read_text(self._child / "memory.events")
            except OSError:
                return {}
        return _parse_memory_events(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            active_pids = self._filesystem.read_text(self._child / "cgroup.procs").split()
        except OSError:
            return
        if not active_pids:
            self._filesystem.remove_child(self._child)


class _WindowsJobMemoryEnforcer:
    def __init__(self, api: WindowsJobApi, limit_bytes: int) -> None:
        self._api = api
        self._job = api.create_job(limit_bytes)
        self._closed = False
        self._status = MemoryEnforcementStatus(
            mode="enforced",
            backend="windows-job-object",
            reason="aggregate Job Object memory limit is active",
            limit_bytes=limit_bytes,
        )

    @property
    def status(self) -> MemoryEnforcementStatus:
        return self._status

    def add_worker(self, pid: int) -> None:
        if self._closed:
            raise RuntimeError("Memory enforcer is already closed")
        if pid <= 0:
            raise ValueError("Worker pid must be a positive integer")
        self._api.assign_process(self._job, pid)

    def current_bytes(self) -> int | None:
        return None

    def memory_events(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._api.close_job(self._job)


def worker_registration_environment(enforcer: MemoryEnforcer) -> dict[str, str]:
    """Return inherited worker setup needed to join a Linux child cgroup.

    The worker writes its own PID during the process-pool initializer, before
    it receives an evaluation task. Other backends intentionally return no
    environment because their attachment must remain parent-controlled.
    """

    if isinstance(enforcer, _LinuxCgroupMemoryEnforcer):
        return {"MOMENT_IMAGES_WORKER_CGROUP_PROCS": str(enforcer._child / "cgroup.procs")}
    return {}


class _CtypesWindowsJobApi:
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    def __init__(self) -> None:
        if os.name != "nt":
            raise _EnforcementUnavailable("Windows Job Objects are unavailable on this host")

        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._extended_limit_information = _ExtendedLimitInformation
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def create_job(self, limit_bytes: int) -> object:
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _windows_error("CreateJobObjectW")

        info = self._extended_limit_information()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = limit_bytes
        if not self._kernel32.SetInformationJobObject(
            handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.close_job(handle)
            raise _windows_error("SetInformationJobObject")
        return handle

    def assign_process(self, job: object, pid: int) -> None:
        process = self._kernel32.OpenProcess(
            self._PROCESS_SET_QUOTA | self._PROCESS_TERMINATE,
            False,
            pid,
        )
        if not process:
            raise _windows_error(f"OpenProcess({pid})")
        try:
            if not self._kernel32.AssignProcessToJobObject(job, process):
                raise _windows_error(f"AssignProcessToJobObject({pid})")
        finally:
            self.close_job(process)

    def close_job(self, job: object) -> None:
        if job:
            self._kernel32.CloseHandle(job)


def create_memory_enforcer(
    policy: MemoryEnforcementPolicy,
    limit_bytes: int,
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    linux_fs: LinuxCgroupFilesystem | None = None,
    windows_api: WindowsJobApi | None = None,
) -> MemoryEnforcer:
    """Create a worker-pool memory enforcer for the requested platform policy."""
    if policy not in {"auto", "required", "advisory"}:
        raise ValueError(f"Unknown memory enforcement policy: {policy!r}")
    if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes <= 0:
        raise ValueError("limit_bytes must be a positive integer")

    platform = (platform_name or sys.platform).lower()
    if policy == "advisory":
        return _AdvisoryMemoryEnforcer("none", "memory enforcement policy is advisory")

    if platform.startswith("linux"):
        try:
            return _create_linux_enforcer(environ or os.environ, linux_fs or _PathLinuxCgroupFilesystem(), limit_bytes)
        except _EnforcementUnavailable as error:
            return _unavailable_enforcer(policy, "linux-cgroup-v2", str(error))

    if platform.startswith("win"):
        try:
            api = windows_api or _CtypesWindowsJobApi()
            return _WindowsJobMemoryEnforcer(api, limit_bytes)
        except (OSError, _EnforcementUnavailable) as error:
            return _unavailable_enforcer(policy, "windows-job-object", str(error))

    return _unavailable_enforcer(
        policy,
        "none",
        f"memory enforcement is unsupported on platform {platform!r}",
    )


def _create_linux_enforcer(
    environ: Mapping[str, str],
    filesystem: LinuxCgroupFilesystem,
    limit_bytes: int,
) -> MemoryEnforcer:
    configured_root = environ.get("MOMENT_IMAGES_CGROUP_ROOT")
    if not configured_root:
        raise _EnforcementUnavailable("MOMENT_IMAGES_CGROUP_ROOT must name a delegated cgroup v2 root")

    root = filesystem.path(configured_root)
    controllers = root / "cgroup.controllers"
    subtree_control = root / "cgroup.subtree_control"
    if not root.is_dir() or not controllers.is_file() or not subtree_control.is_file():
        raise _EnforcementUnavailable("MOMENT_IMAGES_CGROUP_ROOT is not a cgroup v2 root")
    if "memory" not in set(filesystem.read_text(controllers).split()):
        raise _EnforcementUnavailable("delegated cgroup v2 root does not expose the memory controller")
    if not filesystem.is_writable(root) or not filesystem.is_writable(subtree_control):
        raise _EnforcementUnavailable("delegated cgroup v2 root or cgroup.subtree_control is not writable")

    try:
        return _LinuxCgroupMemoryEnforcer(root, filesystem, limit_bytes)
    except OSError as error:
        raise _EnforcementUnavailable(f"could not configure delegated cgroup v2 root: {error}") from error


def _unavailable_enforcer(policy: MemoryEnforcementPolicy, backend: str, reason: str) -> MemoryEnforcer:
    if policy == "required":
        raise RuntimeError(f"Required memory enforcement unavailable: {reason}")
    return _AdvisoryMemoryEnforcer(backend, reason)


def _read_int_file(filesystem: LinuxCgroupFilesystem, path: Path) -> int | None:
    try:
        value = filesystem.read_text(path).strip()
        return int(value)
    except (OSError, ValueError):
        return None


def _parse_memory_events(text: str) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(" ")
        if not separator:
            continue
        try:
            events[key] = int(value.strip())
        except ValueError:
            continue
    return events


def _windows_error(operation: str) -> OSError:
    error = ctypes.get_last_error()
    return OSError(error, f"{operation} failed with Windows error {error}")

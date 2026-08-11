"""Bounded process-ancestry discovery for conversation owner binding."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_MAX_WALK = 64
_HOST_NAMES = frozenset({"claude", "codex", "pi"})
_CLAUDE_HOST_NAMES = frozenset({"claude"})
_NODE_NAMES = frozenset({"node", "nodejs"})
_NODE_HOST_PATHS = {
    "@anthropic-ai/claude-code": "claude",
    "@openai/codex": "codex",
    "@earendil-works/pi-coding-agent": "pi",
}
ERROR_BAD_LENGTH = 24


@dataclass(frozen=True)
class ProcessInfo:
    """One sanitized process-ancestry row."""

    pid: int
    ppid: int
    name: str
    argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class HostResolution:
    """A bounded ancestry chain and its nearest recognized host, if any."""

    chain: tuple[ProcessInfo, ...]
    host: ProcessInfo | None


def _normalized_name(name: str) -> str:
    basename = str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return basename[:-4] if basename.endswith(".exe") else basename


def host_kind(process: str | ProcessInfo) -> str | None:
    """Return the recognized host kind from its image name and launch argv."""
    if isinstance(process, str):
        name = process
        argv: tuple[str, ...] = ()
    else:
        name = process.name
        argv = process.argv
    direct = _normalized_name(name)
    if direct in _HOST_NAMES:
        return direct
    if argv:
        argv0 = _normalized_name(argv[0])
        if argv0 in _HOST_NAMES:
            return argv0
        if direct in _NODE_NAMES or argv0 in _NODE_NAMES:
            normalized_args = [arg.replace("\\", "/").casefold() for arg in argv[1:]]
            for package_path, kind in _NODE_HOST_PATHS.items():
                if any(package_path in arg for arg in normalized_args):
                    return kind
    return None


def is_host(process: str | ProcessInfo) -> bool:
    """Return whether ``process`` is one of the supported agent hosts."""
    return host_kind(process) is not None


def is_claude_host(process: str | ProcessInfo) -> bool:
    """Return whether ``process`` is a supported Claude host."""
    return host_kind(process) in _CLAUDE_HOST_NAMES


def _walk(
    start_pid: int, reader: Callable[[int], ProcessInfo | None]
) -> HostResolution:
    chain: list[ProcessInfo] = []
    visited: set[int] = set()
    pid = start_pid
    for _ in range(_MAX_WALK):
        if pid <= 0 or pid in visited:
            break
        visited.add(pid)
        entry = reader(pid)
        if entry is None:
            break
        chain.append(entry)
        if is_host(entry):
            return HostResolution(tuple(chain), entry)
        pid = entry.ppid
    return HostResolution(tuple(chain), None)


def resolve_from_snapshot(
    start_pid: int, snapshot: Mapping[int, ProcessInfo]
) -> HostResolution:
    """Resolve from an injected process snapshot (deterministic test seam)."""
    return _walk(start_pid, snapshot.get)


def _read_linux_process(
    pid: int, *, proc_root: Path = Path("/proc")
) -> ProcessInfo | None:
    """Read argv/name and parent without parsing the ambiguous ``stat`` file."""
    process_dir = proc_root / str(pid)
    try:
        status = (process_dir / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        raw_cmdline = (process_dir / "cmdline").read_bytes()
    except OSError:
        raw_cmdline = b""
    argv = tuple(
        part.decode("utf-8", errors="replace")
        for part in raw_cmdline.split(b"\0")
        if part
    )
    if argv:
        name = argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    else:
        try:
            name = (
                (process_dir / "comm")
                .read_text(encoding="utf-8", errors="replace")
                .rstrip("\r\n")
            )
        except OSError:
            return None
    ppid: int | None = None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                ppid = int(line.partition(":")[2].strip())
            except ValueError:
                return None
            break
    if not name or ppid is None:
        return None
    return ProcessInfo(pid, ppid, name, argv)


def _create_toolhelp_snapshot(
    kernel32: object, *, get_last_error: Callable[[], int] = ctypes.get_last_error
) -> int:
    """Create a process snapshot, retrying one transient bad-length failure."""
    invalid = ctypes.c_void_p(-1).value
    for attempt in range(2):
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # type: ignore[attr-defined]
        if snapshot != invalid:
            return int(snapshot)
        error = get_last_error()
        if error != ERROR_BAD_LENGTH or attempt == 1:
            raise OSError(error, "CreateToolhelp32Snapshot failed")
    raise AssertionError("unreachable")


def _split_windows_command_line(command_line: str) -> tuple[str, ...]:
    """Split a Windows command line with the same rules as ``CreateProcess``."""
    shell32 = getattr(ctypes, "WinDLL")("shell32", use_last_error=True)  # noqa: B009
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    shell32.CommandLineToArgvW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    values = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not values:
        return ()
    try:
        return tuple(values[index] for index in range(count.value))
    finally:
        kernel32.LocalFree(values)


def _windows_command_lines() -> dict[int, tuple[str, ...]]:
    """Return process argv from one CIM query; inaccessible rows are omitted."""
    powershell = shutil.which("powershell.exe")
    if not powershell or not Path(powershell).is_file():
        powershell = str(
            Path(os.environ.get("SYSTEMROOT", "C:/Windows"))
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )
    try:
        completed = subprocess.run(  # noqa: S603
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    rows = value if isinstance(value, list) else [value]
    result: dict[int, tuple[str, ...]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("CommandLine"), str):
            continue
        try:
            pid = int(row["ProcessId"])
            result[pid] = _split_windows_command_line(row["CommandLine"])
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return result


def _windows_snapshot() -> dict[int, ProcessInfo]:
    """Capture all Windows processes with one Toolhelp snapshot."""

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ProcessID", ctypes.c_uint32),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_uint32),
            ("cntThreads", ctypes.c_uint32),
            ("th32ParentProcessID", ctypes.c_uint32),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_uint32),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    snapshot = _create_toolhelp_snapshot(kernel32)
    command_lines = _windows_command_lines()
    result: dict[int, ProcessInfo] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            result[pid] = ProcessInfo(
                pid=pid,
                ppid=int(entry.th32ParentProcessID),
                name=str(entry.szExeFile),
                argv=command_lines.get(pid, ()),
            )
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return result


def resolve_nearest_host(start_pid: int | None = None) -> HostResolution:
    """Resolve the nearest host over the full Claude/Codex/Pi host set."""
    pid = os.getpid() if start_pid is None else start_pid
    if os.name == "nt":
        return resolve_from_snapshot(pid, _windows_snapshot())
    return _walk(pid, _read_linux_process)

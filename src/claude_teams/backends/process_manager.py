"""Windows-native process lifecycle management for agent backends."""

import contextlib
import ctypes
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar

from claude_teams import eventlog
from claude_teams.backends.contracts import SpawnRequest, SpawnResult
from claude_teams.teams import validate_safe_name


@dataclass
class ProcessInfo:
    """Runtime information for a spawned agent process."""

    pid: int
    name: str
    agent_id: str
    team_name: str
    backend: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: IO[str] | None
    started_at: float
    exit_logged: bool = False


class WindowsJobObject:
    """Best-effort Windows Job Object wrapper for child cleanup."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: ClassVar[int] = 0x00002000
    _JobObjectExtendedLimitInformation: ClassVar[int] = 9

    def __init__(self) -> None:
        """Create a kill-on-close job object on Windows."""
        self._handle: int | None = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        size = ctypes.sizeof(info)
        ok = kernel32.SetInformationJobObject(
            handle,
            self._JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            size,
        )
        if not ok:
            kernel32.CloseHandle(handle)
            return
        self._handle = handle

    def assign(self, process: subprocess.Popen[str]) -> None:
        """Assign a process to the job object when available."""
        if self._handle is None or os.name != "nt":
            return
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject(self._handle, process_handle)

    def close(self) -> None:
        """Close the underlying job handle."""
        if self._handle is None or os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle(self._handle)
        self._handle = None


class WindowsProcessManager:
    """Manage spawned agent CLIs through ``subprocess.Popen``."""

    def __init__(self) -> None:
        """Initialize the process registry and shared job object."""
        self._processes: dict[str, ProcessInfo] = {}
        self._job = WindowsJobObject()

    def spawn_process(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
    ) -> SpawnResult:
        """Start an agent process and return its PID handle."""
        log_path = self.log_path(request.team_name, request.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n[{started_at}] starting {cmd[0]}\n")
        log_handle.flush()

        merged_env = os.environ.copy()
        merged_env.update(env)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        interactive_console = self._should_use_interactive_console(backend_type)
        popen_log_handle: IO[str] | None = log_handle
        if interactive_console:
            cmd = self._with_debug_file(cmd, log_path)
            log_handle.write(
                "[interactive console] stdout/stderr are attached to the agent window\n"
            )
            log_handle.flush()
            log_handle.close()
            popen_log_handle = None
            creationflags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

        try:
            if interactive_console:
                process = subprocess.Popen(  # noqa: S603 - backend argv is built by adapters.
                    cmd,
                    cwd=request.cwd,
                    env=merged_env,
                    stdin=None,
                    stdout=None,
                    stderr=None,
                    text=True,
                    creationflags=creationflags,
                )
            else:
                process = subprocess.Popen(  # noqa: S603 - backend argv is built by adapters.
                    cmd,
                    cwd=request.cwd,
                    env=merged_env,
                    stdin=subprocess.PIPE,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                )
        except BaseException:
            log_handle.close()
            raise

        self._job.assign(process)
        handle = str(process.pid)
        self._processes[handle] = ProcessInfo(
            pid=process.pid,
            name=request.name,
            agent_id=request.agent_id,
            team_name=request.team_name,
            backend=backend_type,
            process=process,
            log_path=log_path,
            log_handle=popen_log_handle,
            started_at=time.time(),
        )
        eventlog.log_event(
            request.team_name,
            "teammate_spawned",
            name=request.name,
            backend=backend_type,
            pid=process.pid,
            log_path=str(log_path),
        )
        if not interactive_console:
            self._open_windows_terminal_tail(request.team_name, request.name, log_path)
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    def health_check(self, handle: str) -> tuple[bool, str]:
        """Return process liveness for a PID handle."""
        info = self._processes.get(handle)
        if info is not None:
            exit_code = info.process.poll()
            if exit_code is None:
                return True, "process running"
            if not info.exit_logged:
                info.exit_logged = True
                eventlog.log_event(
                    info.team_name,
                    "teammate_exited",
                    name=info.name,
                    backend=info.backend,
                    pid=info.pid,
                    exit_code=exit_code,
                )
                self._close_log(info)
            return False, f"process exited ({exit_code})"
        if self._pid_alive(handle):
            return True, "process exists by pid"
        return False, "process not found"

    def kill_process(self, handle: str, timeout_s: float = 10.0) -> None:
        """Terminate a process by PID handle, escalating to kill if needed."""
        info = self._processes.get(handle)
        if info is None:
            self._kill_pid(handle)
            return

        if info.process.poll() is None:
            self._request_shutdown(info.process)
            try:
                info.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                info.process.kill()
                info.process.wait(timeout=timeout_s)
        eventlog.log_event(
            info.team_name,
            "teammate_killed",
            name=info.name,
            backend=info.backend,
            pid=info.pid,
        )
        self._close_log(info)
        self._processes.pop(handle, None)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Try to stop a process without force-killing it."""
        info = self._processes.get(handle)
        if info is None:
            return not self._pid_alive(handle)
        if info.process.poll() is not None:
            return True
        self._request_shutdown(info.process)
        try:
            info.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
        self._close_log(info)
        return True

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Read captured stdout/stderr from the process log."""
        info = self._processes.get(handle)
        if info is None:
            return ""
        return read_log_tail(info.log_path, lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Write text to a running process stdin when a pipe exists."""
        info = self._processes.get(handle)
        if info is None or info.process.stdin is None:
            return
        suffix = "\n" if enter else ""
        info.process.stdin.write(text + suffix)
        info.process.stdin.flush()

    def log_path(self, team_name: str, agent_name: str) -> Path:
        """Return the log file path for a team member."""
        safe_team = validate_safe_name(team_name, "team name")
        safe_agent = validate_safe_name(agent_name, "agent name")
        override = os.environ.get("WIN_AGENT_TEAMS_LOG_DIR")
        if override:
            return Path(override).expanduser() / safe_team / f"{safe_agent}.log"
        return (
            Path.home() / ".claude" / "teams" / safe_team / "logs" / f"{safe_agent}.log"
        )

    def _request_shutdown(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            try:
                process.send_signal(ctrl_break)
            except OSError:
                pass
            else:
                return
        process.terminate()

    def _kill_pid(self, handle: str) -> None:
        try:
            pid = int(handle)
        except ValueError:
            return
        if os.name == "nt":
            taskkill = (
                shutil.which("taskkill.exe") or "C:\\Windows\\System32\\taskkill.exe"
            )
            subprocess.run(  # noqa: S603 - PID is parsed as int before invocation.
                [taskkill, "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)

    def _pid_alive(self, handle: str) -> bool:
        try:
            pid = int(handle)
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def _close_log(self, info: ProcessInfo) -> None:
        if info.log_handle is not None and not info.log_handle.closed:
            info.log_handle.flush()
            info.log_handle.close()

    def _should_use_interactive_console(self, backend_type: str) -> bool:
        if backend_type != "claude-code":
            return False
        if os.environ.get("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        return os.name == "nt"

    def _with_debug_file(self, cmd: list[str], log_path: Path) -> list[str]:
        if "--debug-file" in cmd:
            return cmd
        updated = list(cmd)
        insert_at = updated.index("--") if "--" in updated else len(updated)
        updated[insert_at:insert_at] = ["--debug-file", str(log_path)]
        return updated

    def _open_windows_terminal_tail(
        self, team_name: str, agent_name: str, log_path: Path
    ) -> None:
        if os.environ.get("USE_WINDOWS_TERMINAL", "").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        wt = shutil.which("wt.exe")
        if wt is None:
            return
        title = f"{agent_name}@{team_name}"
        command = [
            wt,
            "-w",
            "0",
            "nt",
            "--title",
            title,
            "--",
            "powershell",
            "-NoExit",
            "-Command",
            f"Get-Content -LiteralPath '{log_path}' -Wait -Tail 80",
        ]
        subprocess.Popen(  # noqa: S603 - opens log tail in Windows Terminal only.
            command,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )


def read_log_tail(path: Path, lines: int | None = None) -> str:
    """Read a full log or its last ``lines`` lines."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if lines is None:
        return text
    if lines <= 0:
        return ""
    return "\n".join(text.splitlines()[-lines:])


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


process_manager = WindowsProcessManager()

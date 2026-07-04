"""Platform process lifecycle management for agent backends."""

import contextlib
import ctypes
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar

from claude_teams.backends.contracts import SpawnRequest, SpawnResult

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_NAME_LEN = 64
_TMUX_SPAWN_FIELD_COUNT = 3
_PROC_STAT_SPLIT_FIELD_COUNT = 2
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_KILL_ON_EXIT_ENV = "WIN_AGENT_TEAMS_KILL_ON_EXIT"
_NO_BREAKAWAY_ENV = "WIN_AGENT_TEAMS_NO_BREAKAWAY"
_LINUX_LAUNCHER_ENV = "WIN_AGENT_TEAMS_LINUX_LAUNCHER"
_LINUX_TERMINAL_ENV = "WIN_AGENT_TEAMS_LINUX_TERMINAL"
_TERMINAL_LAUNCHER_VALUE = "terminal"
_TMUX_LAUNCHER_VALUE = "tmux"
_LINUX_TERMINAL_PID_GRACE_SECONDS = 5.0
_LINUX_DESKTOP_ENV_KEYS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)


def _validate_safe_name(name: str, label: str = "name") -> str:
    """Validate a filesystem-safe team or agent identifier."""
    if not _VALID_NAME_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")  # noqa: TRY003
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"{label} too long: {name!r}")  # noqa: TRY003
    return name


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Return a boolean feature flag from common environment values."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _build_posix_shell_command(cwd: str, cmd: list[str], env: dict[str, str]) -> str:
    """Build a shell command that exports env vars before execing the agent."""
    export_parts = [f"export {key}={shlex.quote(value)};" for key, value in env.items()]
    export_prefix = f"{' '.join(export_parts)} " if export_parts else ""
    return f"cd {shlex.quote(cwd)} && {export_prefix}exec {shlex.join(cmd)}"


def _read_windows_creation_token(pid: int) -> str | None:
    """Return a Windows process's creation FILETIME as an opaque string token.

    Uses ``GetProcessTimes`` (creation time is immutable for a process's
    lifetime), so a reused PID yields a different token. Returns ``None`` when
    the process is gone or the times are unreadable (e.g. access denied) — the
    caller must fail closed on ``None``.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [
        ctypes.POINTER(ctypes.c_uint64)
    ] * 4
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    process_handle = kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not process_handle:
        return None
    try:
        creation = ctypes.c_uint64()
        exit_t = ctypes.c_uint64()
        kernel_t = ctypes.c_uint64()
        user_t = ctypes.c_uint64()
        ok = kernel32.GetProcessTimes(
            process_handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        )
        if not ok or creation.value == 0:
            return None
        return str(creation.value)
    finally:
        kernel32.CloseHandle(process_handle)


def _read_linux_creation_token(pid: int) -> str | None:
    """Return a Linux process's ``starttime`` (field 22) as an opaque token.

    ``starttime`` (clock ticks since boot) is fixed per process, so a reused
    PID yields a different token. The stat line is split with ``rsplit(") ", 1)``
    first because ``comm`` (field 2) can itself contain spaces and parentheses;
    only after the final ``") "`` are the space-delimited fields safe to index.
    Field 22 is index 19 of the post-``comm`` remainder. Returns ``None`` on any
    read/parse failure.
    """
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        stat = stat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    parts = stat.rsplit(") ", 1)
    if len(parts) != _PROC_STAT_SPLIT_FIELD_COUNT:
        return None
    fields = parts[1].split()
    # Field 3 (state) is fields[0]; field 22 (starttime) is fields[19].
    if len(fields) <= 19:  # noqa: PLR2004 - field index from proc(5).
        return None
    starttime = fields[19]
    return starttime if starttime.isdigit() else None


def creation_token(handle: str) -> str | None:
    """Return an opaque, PID-reuse-distinguishing creation token, or ``None``.

    ``None`` means the PID is not live or its creation metadata is unreadable;
    callers gating destructive operations must treat ``None`` as "not owned".
    """
    try:
        pid = int(handle)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os.name == "nt":
        return _read_windows_creation_token(pid)
    return _read_linux_creation_token(pid)


class _PidOwnershipMixin:
    """Shared PID-reuse-safe ownership/liveness for all process managers.

    Provides the fail-closed ``owns_process`` gate (used before any destructive
    PID operation) and the token-aware liveness tail shared by the managers'
    ``health_check`` implementations. Subclasses supply ``self._processes`` and
    ``self._pid_alive``.
    """

    _processes: dict

    def _pid_alive(self, handle: str) -> bool:  # pragma: no cover - subclass provides
        raise NotImplementedError

    def creation_token(self, handle: str) -> str | None:
        """Return the live creation token for ``handle`` (module-level compute)."""
        return creation_token(handle)

    def resolve_agent_pid(
        self,
        handle: str,
        team_name: str,  # noqa: ARG002 - part of the override interface
        agent_name: str,  # noqa: ARG002 - part of the override interface
    ) -> str:
        """Return the authoritative agent PID for ``handle`` (default: ``handle``).

        Overridden by launcher-style managers (Linux terminal) where ``handle``
        is a launcher PID and the real agent runs under a different PID
        recorded in a sidecar.
        """
        return handle

    def _tracked_alive(self, info: object) -> bool:  # pragma: no cover - subclass
        """Whether the in-memory tracked child/pane for ``info`` is really alive.

        Manager-specific and PID-reuse-safe: it must prove OUR original
        process/pane is still running, never merely that some process owns the
        numeric PID (which a reused PID would satisfy).
        """
        raise NotImplementedError

    def _has_live_registry_entry(self, handle: str) -> bool:
        """Whether this manager still owns a live in-memory child for ``handle``.

        Proves the tracked child/pane is alive (``_tracked_alive``) rather than
        trusting bare PID existence, so a stale in-memory entry whose PID was
        reused by a foreign process is NOT treated as owned.
        """
        info = self._processes.get(handle)
        return info is not None and self._tracked_alive(info)

    def owns_process(self, handle: str, expected_token: str | None) -> bool:
        """Return whether ``handle`` is provably still our process (fail-closed).

        ``True`` only when EITHER this manager still has current in-memory
        ownership of a live child for ``handle``, OR the live PID's creation
        token equals ``expected_token``. A tokenless expectation, an unreadable
        live token (dead / access denied), or a mismatch all return ``False`` —
        so a reused or foreign PID is never gracefully-shut-down or killed.
        """
        if self._has_live_registry_entry(handle):
            return True
        if not expected_token:
            return False
        live = creation_token(handle)
        return live is not None and live == expected_token

    def _pid_health_with_token(
        self, handle: str, expected_token: str | None
    ) -> tuple[bool, str]:
        """Token-aware liveness for the no-in-memory-registry case.

        With ``expected_token`` set, a live PID whose token differs (reuse) or
        is unreadable is reported dead. Without a token, falls back to bare PID
        liveness (backward compatible for records predating tokens — display
        only; destructive ops still gate on ``owns_process``).
        """
        if expected_token:
            live = creation_token(handle)
            if live is None:
                return False, "process not found or token unreadable"
            if live != expected_token:
                return False, "pid reused (token mismatch)"
            return True, "process exists by pid (token match)"
        if self._pid_alive(handle):
            return True, "process exists by pid"
        return False, "process not found"


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


@dataclass
class TmuxProcessInfo:
    """Runtime information for a spawned tmux pane/window."""

    pid: int
    name: str
    agent_id: str
    team_name: str
    backend: str
    target_id: str
    pane_id: str
    log_path: Path
    started_at: float


@dataclass
class LinuxTerminalProcessInfo:
    """Runtime information for a spawned Linux terminal window."""

    pid: int
    name: str
    agent_id: str
    team_name: str
    backend: str
    terminal_process: subprocess.Popen[str]
    log_path: Path
    agent_pid_path: Path
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


class WindowsProcessManager(_PidOwnershipMixin):
    """Manage spawned agent CLIs through ``subprocess.Popen``."""

    def __init__(self) -> None:
        """Initialize the process registry and shared job object."""
        self._processes: dict[str, ProcessInfo] = {}
        self._kill_on_exit = _env_flag(_KILL_ON_EXIT_ENV)
        self._job = WindowsJobObject()

    def spawn_process(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
        *,
        is_interactive: bool = False,
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
        if not _env_flag(_NO_BREAKAWAY_ENV):
            creationflags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        interactive_console = self._should_use_interactive_console(
            backend_type, is_interactive=is_interactive
        )
        popen_log_handle: IO[str] | None = log_handle
        if interactive_console:
            if backend_type == "claude-code":
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
                process = self._popen(
                    cmd,
                    creationflags,
                    cwd=request.cwd,
                    env=merged_env,
                    stdin=None,
                    stdout=None,
                    stderr=None,
                    text=True,
                )
            else:
                stdin = (
                    subprocess.DEVNULL
                    if backend_type == "claude-code" and is_interactive
                    else subprocess.PIPE
                )
                process = self._popen(
                    cmd,
                    creationflags,
                    cwd=request.cwd,
                    env=merged_env,
                    stdin=stdin,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except BaseException:
            log_handle.close()
            raise

        if self._kill_on_exit:
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
        if not interactive_console:
            self._open_windows_terminal_tail(request.team_name, request.name, log_path)
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    def _popen(
        self, cmd: list[str], creationflags: int, **kwargs: object
    ) -> subprocess.Popen[str]:
        """Spawn a process, retrying once without breakaway if it's denied.

        Some ambient Job Objects forbid ``CREATE_BREAKAWAY_FROM_JOB``, which
        makes ``CreateProcess`` fail with ``OSError(winerror=5)``
        (ERROR_ACCESS_DENIED). When that happens and the breakaway bit was
        set, retry once with it cleared so the agent still spawns (falling
        back to living inside the server's job, matching prior behavior).
        Any other failure — or a failure when breakaway wasn't requested —
        propagates unchanged.
        """
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        used_breakaway = bool(breakaway) and bool(creationflags & breakaway)
        try:
            return subprocess.Popen(  # noqa: S603 - backend argv is built by adapters.
                cmd, creationflags=creationflags, **kwargs
            )
        except OSError as err:
            denied = getattr(err, "winerror", None) == _ERROR_ACCESS_DENIED
            if not used_breakaway or not denied:
                raise
            log_handle = kwargs.get("stdout")
            warning = (
                "[warning] CREATE_BREAKAWAY_FROM_JOB denied by ambient job; "
                "retrying without breakaway\n"
            )
            if log_handle is not None and hasattr(log_handle, "write"):
                with contextlib.suppress(ValueError, OSError):
                    log_handle.write(warning)
                    log_handle.flush()
            fallback_flags = creationflags & ~breakaway
            return subprocess.Popen(  # noqa: S603 - backend argv is built by adapters.
                cmd, creationflags=fallback_flags, **kwargs
            )

    def health_check(
        self, handle: str, expected_token: str | None = None
    ) -> tuple[bool, str]:
        """Return process liveness for a PID handle.

        ``expected_token`` (a prior :func:`creation_token`) makes liveness
        PID-reuse-safe for recovered records after a server/host restart: a
        live-but-reused PID is reported dead. Ignored while this manager still
        owns the child in-memory (same-process spawn).
        """
        info = self._processes.get(handle)
        if info is not None:
            exit_code = info.process.poll()
            if exit_code is None:
                return True, "process running"
            if not info.exit_logged:
                info.exit_logged = True
                self._close_log(info)
            return False, f"process exited ({exit_code})"
        return self._pid_health_with_token(handle, expected_token)

    def _tracked_alive(self, info: object) -> bool:
        """Our tracked Popen child is alive only while ``poll()`` is ``None``."""
        return info.process.poll() is None  # type: ignore[attr-defined]

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
                self._kill_pid(str(info.pid))
                info.process.wait(timeout=timeout_s)
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
        safe_team = _validate_safe_name(team_name, "team name")
        safe_agent = _validate_safe_name(agent_name, "agent name")
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
        except ValueError:
            return False
        if os.name == "nt":
            return self._windows_pid_alive(pid)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _windows_pid_alive(self, pid: int) -> bool:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        process_handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not process_handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(process_handle)

    def _close_log(self, info: ProcessInfo) -> None:
        if info.log_handle is not None and not info.log_handle.closed:
            info.log_handle.flush()
            info.log_handle.close()

    def _should_use_interactive_console(
        self, backend_type: str, *, is_interactive: bool = False
    ) -> bool:
        _ = backend_type
        if not is_interactive:
            return False
        if os.name != "nt":
            return False
        flag = os.environ.get("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE")
        if flag is not None:
            return _env_flag("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE")
        return True

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


class TmuxProcessManager(_PidOwnershipMixin):
    """Manage spawned agent CLIs through tmux panes or windows."""

    def __init__(self) -> None:
        """Initialize the tmux target registry."""
        self._processes: dict[str, TmuxProcessInfo] = {}

    def spawn_process(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
        *,
        is_interactive: bool = False,
    ) -> SpawnResult:
        """Start an agent process in tmux and return its pane PID handle."""
        _ = is_interactive
        self._require_tmux()
        if backend_type == "claude-code":
            cmd = self._with_debug_file(
                cmd, self.log_path(request.team_name, request.name)
            )

        log_path = self.log_path(request.team_name, request.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        command = self._build_shell_command(request.cwd, cmd, env)
        tmux_args, target_kind = self._build_tmux_spawn_args(request, command)

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{started_at}] starting {cmd[0]} in tmux\n")
            log_handle.flush()

        merged_env = os.environ.copy()
        merged_env.update(env)
        result = subprocess.run(  # noqa: S603 - tmux argv is built internally.
            tmux_args,
            check=True,
            capture_output=True,
            text=True,
            env=merged_env,
        )
        window_id, pane_id, pid = self._parse_tmux_spawn_output(result.stdout)
        target_id = window_id if target_kind == "window" else pane_id
        handle = str(pid)
        self._processes[handle] = TmuxProcessInfo(
            pid=pid,
            name=request.name,
            agent_id=request.agent_id,
            team_name=request.team_name,
            backend=backend_type,
            target_id=target_id,
            pane_id=pane_id,
            log_path=log_path,
            started_at=time.time(),
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"[tmux] target={target_id} pane={pane_id} pid={pid}\n")
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    def health_check(
        self, handle: str, expected_token: str | None = None
    ) -> tuple[bool, str]:
        """Return process liveness for a PID handle (token-aware after restart)."""
        info = self._processes.get(handle)
        if info is not None:
            alive, detail = self._pane_alive(info.pane_id)
            if alive:
                return True, detail
            if self._pid_alive(handle):
                return True, "process exists by pid"
            return False, detail
        return self._pid_health_with_token(handle, expected_token)

    def _tracked_alive(self, info: object) -> bool:
        """Ownership is proven by pane liveness, never a (reusable) bare PID."""
        alive, _ = self._pane_alive(info.pane_id)  # type: ignore[attr-defined]
        return alive

    def kill_process(self, handle: str, timeout_s: float = 10.0) -> None:
        """Kill a tmux pane/window or fall back to killing a PID."""
        info = self._processes.pop(handle, None)
        if info is not None:
            self._kill_tmux_target(info.target_id)
            if not self._wait_pid_exit(info.pid, timeout_s):
                self._kill_pid(str(info.pid))
            return
        self._kill_pid(handle)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Ask a tmux pane to stop with Ctrl-C before force-kill is needed."""
        info = self._processes.get(handle)
        if info is None:
            return not self._pid_alive(handle)
        if not self._pane_alive(info.pane_id)[0]:
            return True
        tmux = self._tmux_binary()
        subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, "send-keys", "-t", info.pane_id, "C-c"],
            check=False,
            capture_output=True,
            text=True,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pane_alive(info.pane_id)[0]:
                return True
            time.sleep(0.1)
        return False

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture output from the tmux pane."""
        info = self._processes.get(handle)
        if info is None:
            return ""
        args = [self._tmux_binary(), "capture-pane", "-p", "-t", info.pane_id, "-J"]
        if lines is None:
            args.extend(["-S", "-"])
        elif lines <= 0:
            return ""
        else:
            args.extend(["-S", f"-{lines}"])
        result = subprocess.run(  # noqa: S603 - tmux argv is built internally.
            args,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        return result.stdout

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send literal text to a tmux pane."""
        info = self._processes.get(handle)
        if info is None:
            return
        tmux = self._tmux_binary()
        subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, "send-keys", "-l", "-t", info.pane_id, "--", text],
            check=False,
            capture_output=True,
            text=True,
        )
        if enter:
            subprocess.run(  # noqa: S603 - tmux argv is built internally.
                [tmux, "send-keys", "-t", info.pane_id, "Enter"],
                check=False,
                capture_output=True,
                text=True,
            )

    def log_path(self, team_name: str, agent_name: str) -> Path:
        """Return the log file path for a team member."""
        safe_team = _validate_safe_name(team_name, "team name")
        safe_agent = _validate_safe_name(agent_name, "agent name")
        override = os.environ.get("WIN_AGENT_TEAMS_LOG_DIR")
        if override:
            return Path(override).expanduser() / safe_team / f"{safe_agent}.log"
        return (
            Path.home() / ".claude" / "teams" / safe_team / "logs" / f"{safe_agent}.log"
        )

    def _build_shell_command(
        self, cwd: str, cmd: list[str], env: dict[str, str]
    ) -> str:
        return _build_posix_shell_command(cwd, cmd, env)

    def _build_tmux_spawn_args(
        self, request: SpawnRequest, command: str
    ) -> tuple[list[str], str]:
        title = f"{request.name}@{request.team_name}"
        fmt = "#{window_id}\t#{pane_id}\t#{pane_pid}"
        explicit_target = os.environ.get("WIN_AGENT_TEAMS_TMUX_TARGET", "").strip()
        if explicit_target:
            if _env_flag("USE_TMUX_WINDOWS"):
                return (
                    [
                        "tmux",
                        "new-window",
                        "-dP",
                        "-t",
                        explicit_target,
                        "-F",
                        fmt,
                        "-n",
                        title,
                        command,
                    ],
                    "window",
                )
            return (
                [
                    "tmux",
                    "split-window",
                    "-dP",
                    "-t",
                    explicit_target,
                    "-F",
                    fmt,
                    command,
                ],
                "pane",
            )

        if self._inside_tmux():
            if _env_flag("USE_TMUX_WINDOWS"):
                return (
                    [
                        "tmux",
                        "new-window",
                        "-dP",
                        "-F",
                        fmt,
                        "-n",
                        title,
                        command,
                    ],
                    "window",
                )
            return (
                ["tmux", "split-window", "-dP", "-F", fmt, command],
                "pane",
            )

        session_name = self._session_name(request.team_name)
        if self._tmux_session_exists(session_name):
            return (
                [
                    "tmux",
                    "new-window",
                    "-dP",
                    "-t",
                    session_name,
                    "-F",
                    fmt,
                    "-n",
                    title,
                    command,
                ],
                "window",
            )
        return (
            [
                "tmux",
                "new-session",
                "-dP",
                "-s",
                session_name,
                "-n",
                title,
                "-F",
                fmt,
                command,
            ],
            "window",
        )

    def _parse_tmux_spawn_output(self, output: str) -> tuple[str, str, int]:
        fields = output.strip().split("\t")
        if len(fields) != _TMUX_SPAWN_FIELD_COUNT:
            msg = f"Unexpected tmux spawn output: {output!r}"
            raise RuntimeError(msg)
        window_id, pane_id, pid_text = fields
        try:
            pid = int(pid_text)
        except ValueError as err:
            msg = f"Unexpected tmux pane PID: {pid_text!r}"
            raise RuntimeError(msg) from err
        return window_id, pane_id, pid

    def _pane_alive(self, pane_id: str) -> tuple[bool, str]:
        tmux = self._tmux_binary()
        result = subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, "display-message", "-p", "-t", pane_id, "#{pane_dead}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "tmux target not found"
        if result.stdout.strip() == "1":
            return False, "tmux pane dead"
        return True, "tmux pane running"

    def _kill_tmux_target(self, target_id: str) -> None:
        command = "kill-window" if target_id.startswith("@") else "kill-pane"
        tmux = self._tmux_binary()
        subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, command, "-t", target_id],
            check=False,
            capture_output=True,
            text=True,
        )

    def _kill_pid(self, handle: str) -> None:
        try:
            pid = int(handle)
        except ValueError:
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)

    def _pid_alive(self, handle: str) -> bool:
        try:
            pid = int(handle)
        except ValueError:
            return False
        if self._pid_is_zombie(pid):
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _wait_pid_exit(self, pid: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pid_alive(str(pid)):
                return True
            time.sleep(0.05)
        return not self._pid_alive(str(pid))

    def _pid_is_zombie(self, pid: int) -> bool:
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            stat = stat_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        parts = stat.rsplit(") ", 1)
        return len(parts) == _PROC_STAT_SPLIT_FIELD_COUNT and parts[1].startswith("Z ")

    def _inside_tmux(self) -> bool:
        return bool(os.environ.get("TMUX"))

    def _tmux_session_exists(self, session_name: str) -> bool:
        tmux = self._tmux_binary()
        result = subprocess.run(  # noqa: S603 - session name is validated.
            [tmux, "has-session", "-t", session_name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _session_name(self, team_name: str) -> str:
        safe_team = _validate_safe_name(team_name, "team name")
        return f"win-agent-teams-{safe_team[:40]}"

    def _with_debug_file(self, cmd: list[str], log_path: Path) -> list[str]:
        if "--debug-file" in cmd:
            return cmd
        updated = list(cmd)
        insert_at = updated.index("--") if "--" in updated else len(updated)
        updated[insert_at:insert_at] = ["--debug-file", str(log_path)]
        return updated

    def _tmux_binary(self) -> str:
        return shutil.which("tmux") or "tmux"

    def _require_tmux(self) -> None:
        if shutil.which("tmux") is not None:
            return
        msg = "Could not find 'tmux' on PATH. Install tmux to spawn agents on Linux."
        raise FileNotFoundError(msg)


class LinuxTerminalProcessManager(_PidOwnershipMixin):
    """Manage spawned agent CLIs through Linux terminal emulator windows."""

    def __init__(self) -> None:
        """Initialize the terminal process registry."""
        self._processes: dict[str, LinuxTerminalProcessInfo] = {}

    def spawn_process(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
        *,
        is_interactive: bool = False,
    ) -> SpawnResult:
        """Start an agent process in a new terminal emulator window."""
        _ = is_interactive
        log_path = self.log_path(request.team_name, request.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if backend_type == "claude-code":
            cmd = self._with_debug_file(cmd, log_path)

        terminal = self._discover_terminal()
        title = f"{request.name}@{request.team_name}"
        shell_command = self._build_shell_command(request.cwd, cmd, env)
        agent_pid_path = self._agent_pid_path(log_path)
        with contextlib.suppress(OSError):
            agent_pid_path.unlink()
        terminal_shell_command = self._with_agent_pid_file(
            shell_command,
            agent_pid_path,
            log_path,
        )
        terminal_cmd = self._terminal_command(terminal, title, terminal_shell_command)
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{started_at}] starting {cmd[0]} in {terminal}\n")
            log_handle.write(f"[agent pid file] {agent_pid_path}\n")
            log_handle.write(f"[terminal command] {shlex.join(terminal_cmd)}\n")
            log_handle.flush()

        merged_env = self._desktop_env(os.environ.copy())
        merged_env.update(env)
        with log_path.open("a", encoding="utf-8") as terminal_log:
            process = subprocess.Popen(  # noqa: S603 - terminal argv is built internally.
                terminal_cmd,
                cwd=request.cwd,
                env=merged_env,
                stdin=None,
                stdout=terminal_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        handle = str(process.pid)
        self._processes[handle] = LinuxTerminalProcessInfo(
            pid=process.pid,
            name=request.name,
            agent_id=request.agent_id,
            team_name=request.team_name,
            backend=backend_type,
            terminal_process=process,
            log_path=log_path,
            agent_pid_path=agent_pid_path,
            started_at=time.time(),
        )
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    def health_check(
        self, handle: str, expected_token: str | None = None
    ) -> tuple[bool, str]:
        """Return terminal process liveness for a PID handle (token-aware).

        While this manager still owns the launcher in-memory it prefers the
        real agent PID (from the sidecar). After a restart there is no
        in-memory info; the persisted ``expected_token`` (the launcher PID's
        token — see the plan's documented residual) gates liveness so a reused
        launcher PID is reported dead rather than falsely alive.
        """
        info = self._processes.get(handle)
        if info is not None:
            agent_status = self._agent_pid_health(info)
            if agent_status is not None:
                return agent_status
            return self._terminal_launcher_health(info)
        return self._pid_health_with_token(handle, expected_token)

    def _tracked_alive(self, info: object) -> bool:
        """Prefer the real agent PID (sidecar); else the launcher child liveness."""
        status = self._agent_pid_health(info)  # type: ignore[arg-type]
        if status is not None:
            return status[0]
        return info.terminal_process.poll() is None  # type: ignore[attr-defined]

    def resolve_agent_pid(self, handle: str, team_name: str, agent_name: str) -> str:
        """Return the real agent PID from the sidecar, else the launcher handle.

        The sidecar path is deterministic from the log path, so the agent PID
        is recoverable after a restart (no in-memory info) too — which lets the
        server report the true agent liveness rather than the exited launcher's.
        """
        info = self._processes.get(handle)
        if info is not None:
            pid = self._read_pid_file(info.agent_pid_path)
            return str(pid) if pid is not None else handle
        try:
            path = self._agent_pid_path(self.log_path(team_name, agent_name))
        except ValueError:
            return handle
        pid = self._read_pid_file(path)
        return str(pid) if pid is not None else handle

    def kill_process(self, handle: str, timeout_s: float = 10.0) -> None:
        """Terminate a terminal process by PID handle."""
        info = self._processes.pop(handle, None)
        if info is None:
            self._kill_pid(handle)
            return
        agent_pid = self._read_pid_file(info.agent_pid_path)
        if agent_pid is not None and self._pid_alive(str(agent_pid)):
            with contextlib.suppress(OSError):
                os.kill(agent_pid, signal.SIGTERM)
            if not self._wait_pid_exit(agent_pid, timeout_s):
                self._kill_pid(str(agent_pid))
        process = info.terminal_process
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_pid(str(info.pid))
                process.wait(timeout=timeout_s)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Try to close a terminal window without force-killing it."""
        info = self._processes.get(handle)
        if info is None:
            return not self._pid_alive(handle)
        agent_pid = self._read_pid_file(info.agent_pid_path)
        if agent_pid is not None:
            if self._pid_alive(str(agent_pid)):
                with contextlib.suppress(OSError):
                    os.kill(agent_pid, signal.SIGTERM)
                if not self._wait_pid_exit(agent_pid, timeout_s):
                    return False
            return self._terminate_terminal_process(info, timeout_s)
        if info.terminal_process.poll() is not None:
            return True
        return self._terminate_terminal_process(info, timeout_s)

    def _terminate_terminal_process(
        self, info: LinuxTerminalProcessInfo, timeout_s: float
    ) -> bool:
        with contextlib.suppress(OSError):
            info.terminal_process.terminate()
        try:
            info.terminal_process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
        return True

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Read the terminal launch log."""
        info = self._processes.get(handle)
        if info is None:
            return ""
        return read_log_tail(info.log_path, lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Cannot send stdin to a separately managed terminal window."""
        _ = handle, text, enter

    def log_path(self, team_name: str, agent_name: str) -> Path:
        """Return the log file path for a team member."""
        safe_team = _validate_safe_name(team_name, "team name")
        safe_agent = _validate_safe_name(agent_name, "agent name")
        override = os.environ.get("WIN_AGENT_TEAMS_LOG_DIR")
        if override:
            return Path(override).expanduser() / safe_team / f"{safe_agent}.log"
        return (
            Path.home() / ".claude" / "teams" / safe_team / "logs" / f"{safe_agent}.log"
        )

    def _build_shell_command(
        self, cwd: str, cmd: list[str], env: dict[str, str]
    ) -> str:
        return _build_posix_shell_command(cwd, cmd, env)

    def _with_agent_pid_file(
        self, shell_command: str, agent_pid_path: Path, log_path: Path
    ) -> str:
        pid_file = shlex.quote(str(agent_pid_path))
        launch_log = shlex.quote(str(log_path))
        return (
            f"printf '%s\\n' \"$$\" > {pid_file}; "
            f"printf '[agent pid] %s\\n' \"$$\" >> {launch_log}; "
            f"{shell_command}"
        )

    def _agent_pid_health(
        self, info: LinuxTerminalProcessInfo
    ) -> tuple[bool, str] | None:
        agent_pid = self._read_pid_file(info.agent_pid_path)
        if agent_pid is None:
            return None
        if self._pid_alive(str(agent_pid)):
            return True, "agent process running"
        if not info.exit_logged:
            info.exit_logged = True
            with info.log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"[agent exited] pid={agent_pid}\n")
        return False, "agent process exited"

    def _terminal_launcher_health(
        self, info: LinuxTerminalProcessInfo
    ) -> tuple[bool, str]:
        exit_code = info.terminal_process.poll()
        if exit_code is None:
            return True, "terminal process running"
        if (
            exit_code == 0
            and time.time() - info.started_at < _LINUX_TERMINAL_PID_GRACE_SECONDS
        ):
            return True, "terminal launcher exited; waiting for agent pid"
        if not info.exit_logged:
            info.exit_logged = True
            with info.log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"[terminal exited] code={exit_code}\n")
        return False, f"terminal process exited ({exit_code})"

    def _discover_terminal(self) -> str:
        override = os.environ.get(_LINUX_TERMINAL_ENV, "").strip()
        if override:
            resolved = shutil.which(override)
            if resolved:
                return resolved
            if Path(override).exists():
                return override
            msg = f"Configured terminal not found: {override!r}"
            raise FileNotFoundError(msg)

        for candidate in (
            "qterminal",
            "gnome-terminal",
            "x-terminal-emulator",
            "xfce4-terminal",
            "konsole",
            "mate-terminal",
            "lxterminal",
            "xterm",
        ):
            if candidate == "qterminal" and self._process_name_running(candidate):
                continue
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        msg = (
            "Could not find a supported terminal emulator on PATH. "
            f"Set {_LINUX_TERMINAL_ENV} to a terminal command."
        )
        raise FileNotFoundError(msg)

    def _terminal_command(
        self, terminal: str, title: str, shell_command: str
    ) -> list[str]:
        name = Path(terminal).name
        if name == "qterminal":
            return [terminal, "-e", "bash", "-lc", shell_command]
        if name in {"gnome-terminal", "kgx"}:
            # --wait keeps the client process alive for the agent's lifetime;
            # without it gnome-terminal forks to a server and exits instantly,
            # making health checks report a false exit.
            return [
                terminal,
                "--wait",
                "--title",
                title,
                "--",
                "bash",
                "-lc",
                shell_command,
            ]
        if name == "xfce4-terminal":
            return [
                terminal,
                "--title",
                title,
                "--command",
                f"bash -lc {shlex.quote(shell_command)}",
            ]
        if name == "konsole":
            return [
                terminal,
                "--new-tab",
                "-p",
                f"tabtitle={title}",
                "-e",
                "bash",
                "-lc",
                shell_command,
            ]
        if name in {"mate-terminal", "lxterminal"}:
            return [
                terminal,
                "--title",
                title,
                "-e",
                f"bash -lc {shlex.quote(shell_command)}",
            ]
        return [terminal, "-T", title, "-e", "bash", "-lc", shell_command]

    def _desktop_env(self, env: dict[str, str]) -> dict[str, str]:
        if all(env.get(key) for key in _LINUX_DESKTOP_ENV_KEYS):
            return env
        parent_env = self._read_process_env(os.getppid())
        for key in _LINUX_DESKTOP_ENV_KEYS:
            if not env.get(key) and parent_env.get(key):
                env[key] = parent_env[key]
        return env

    def _agent_pid_path(self, log_path: Path) -> Path:
        return log_path.with_suffix(".pid")

    def _read_pid_file(self, path: Path) -> int | None:
        try:
            pid_text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(pid_text)
        except ValueError:
            return None

    def _process_name_running(self, name: str) -> bool:
        pgrep = shutil.which("pgrep")
        if pgrep is None:
            return False
        result = subprocess.run(  # noqa: S603 - process name is controlled internally.
            [pgrep, "-x", name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _read_process_env(self, pid: int) -> dict[str, str]:
        environ = Path("/proc") / str(pid) / "environ"
        try:
            raw = environ.read_bytes()
        except OSError:
            return {}
        result: dict[str, str] = {}
        for item in raw.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            result[key.decode(errors="replace")] = value.decode(errors="replace")
        return result

    def _with_debug_file(self, cmd: list[str], log_path: Path) -> list[str]:
        if "--debug-file" in cmd:
            return cmd
        updated = list(cmd)
        insert_at = updated.index("--") if "--" in updated else len(updated)
        updated[insert_at:insert_at] = ["--debug-file", str(log_path)]
        return updated

    def _pid_alive(self, handle: str) -> bool:
        try:
            pid = int(handle)
        except ValueError:
            return False
        if self._pid_is_zombie(pid):
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _wait_pid_exit(self, pid: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pid_alive(str(pid)):
                return True
            time.sleep(0.05)
        return not self._pid_alive(str(pid))

    def _pid_is_zombie(self, pid: int) -> bool:
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            stat = stat_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        parts = stat.rsplit(") ", 1)
        return len(parts) == _PROC_STAT_SPLIT_FIELD_COUNT and parts[1].startswith("Z ")

    def _kill_pid(self, handle: str) -> None:
        try:
            pid = int(handle)
        except ValueError:
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


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


if os.name == "nt":
    process_manager = WindowsProcessManager()
elif os.environ.get(_LINUX_LAUNCHER_ENV, "").strip().lower() == _TMUX_LAUNCHER_VALUE:
    process_manager = TmuxProcessManager()
else:
    process_manager = LinuxTerminalProcessManager()

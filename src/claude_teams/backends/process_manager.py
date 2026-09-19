"""Platform process lifecycle management for agent backends."""

import contextlib
import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Any, ClassVar, cast

from claude_teams.agent_output import CORRELATION_FIELD, correlation_marker_token
from claude_teams.backends.contracts import SpawnRequest, SpawnResult
from claude_teams.filelock import file_lock

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_NAME_LEN = 64
_TMUX_SPAWN_FIELD_COUNT = 3
#: tmux window and pane ids, as emitted by ``#{window_id}``/``#{pane_id}``.
#: tmux formats these with ``@%u``/``%%%u``, so the language is ASCII digits
#: only. ``\d`` would also accept Unicode decimal digits such as U+0661, which
#: tmux can never emit and can never address either.
_TMUX_WINDOW_ID = re.compile(r"@[0-9]+")
_TMUX_PANE_ID = re.compile(r"%[0-9]+")
#: Likewise ``int()`` accepts "+3", "٣" and surrounding whitespace; a pane PID
#: is a plain positive decimal or it is not a pane PID.
_TMUX_PANE_PID = re.compile(r"[1-9][0-9]*")
_PROC_STAT_SPLIT_FIELD_COUNT = 2
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_KILL_ON_EXIT_ENV = "WIN_AGENT_TEAMS_KILL_ON_EXIT"
_NO_BREAKAWAY_ENV = "WIN_AGENT_TEAMS_NO_BREAKAWAY"
_NO_WT_TABS_ENV = "WIN_AGENT_TEAMS_NO_WT_TABS"
# Opt back in to the legacy direct-launch for codex tabs (codex.exe as the tab's
# console-root process, no powershell wrapper). Default off: codex now uses the
# same exit-0 wrapper as claude, which auto-closes the tab on kill/error and
# keeps the prompt off wt's command line. Kept as a fallback in case a future
# codex build regresses the (now-verified) wrapper compatibility.
_CODEX_DIRECT_LAUNCH_ENV = "WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH"
_WT_TAB_PID_TIMEOUT_SECONDS = 12.0
_WT_TAB_PID_POLL_SECONDS = 0.1
# How long a freshly-launched WT tab agent must stay alive to be considered a
# healthy start. A degraded WT window opens the tab and runs the wrapper (so its
# PID is recorded) but hands it no usable console, so codex's TUI aborts within
# ~1s ("stdin is not a terminal") and the wrapper shell exits. A PID that dies
# inside this window is treated as an immediate abort and retried in a fresh
# console. Env-tunable; 0 disables the check.
_WT_TAB_SETTLE_SECONDS = 2.0
_WT_TAB_SETTLE_ENV = "WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS"
_LINUX_LAUNCHER_ENV = "WIN_AGENT_TEAMS_LINUX_LAUNCHER"
_LINUX_TERMINAL_ENV = "WIN_AGENT_TEAMS_LINUX_TERMINAL"
_TERMINAL_LAUNCHER_VALUE = "terminal"
_TMUX_LAUNCHER_VALUE = "tmux"
_HERDR_LAUNCHER_VALUE = "herdr"
_HERDR_SESSION_ENV = "WIN_AGENT_TEAMS_HERDR_SESSION"
#: Herdr session names are a CLI argument and a path component, so the
#: language is deliberately narrow: no whitespace, no separators, no leading
#: dash that argv would read as a flag.
_HERDR_SESSION_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
#: Pin every agent of this process to one workspace label, or opt back out to
#: the pre-per-repo behaviour ("the active workspace") with the sentinel.
_HERDR_WORKSPACE_ENV = "WIN_AGENT_TEAMS_HERDR_WORKSPACE"
_HERDR_ACTIVE_WORKSPACE_SENTINEL = "-"
#: Labels are free-form display text, not a path component or a selector, so
#: the session language would be far too narrow here. Only the things that
#: would break a terminal, an argv parser or a filesystem are excluded.
_HERDR_LABEL_MAX_LEN = 128
_HERDR_DEFAULT_LABEL = "agents"
_HERDR_CALL_TIMEOUT_SECONDS = 15.0
#: clap exits 2 for a usage error: our argv is wrong, not Herdr's state.
_HERDR_USAGE_EXIT_CODE = 2
_HERDR_START_TIMEOUT_SECONDS = 20.0
_HERDR_START_POLL_SECONDS = 0.2
_HERDR_TERMINATE_TIMEOUT_SECONDS = 5.0
#: Herdr error codes that settle "that object no longer exists".
_HERDR_NOT_FOUND_CODES = frozenset({"not_found", "pane_not_found", "tab_not_found"})
#: A never-attached server has no workspace to put a tab in yet.
_HERDR_NO_WORKSPACE_CODES = frozenset({"workspace_not_found", "no_active_workspace"})
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


def _powershell_quote(value: str) -> str:
    """Quote a value as a PowerShell single-quoted literal (``'`` -> ``''``)."""
    return "'" + str(value).replace("'", "''") + "'"


def _force_kill_pid(handle: str) -> None:
    """Force-kill a PID, best-effort, on both Windows and POSIX.

    Shared by every process manager's ``_kill_pid``. It exists because the
    three managers each carried their own copy and only one of them grew the
    Windows branch: the other two called ``signal.SIGKILL`` unguarded, which
    does not exist on Windows and would raise ``AttributeError`` rather than
    failing quietly. Keeping one implementation is what stops that drifting
    apart again.

    A non-integer handle is ignored on both platforms. On POSIX an ``OSError``
    from ``os.kill`` (gone, or not ours to signal) is suppressed too. The
    Windows branch does not suppress: ``subprocess.run`` is already
    ``check=False``, so a failed ``taskkill`` is reported through its exit code
    rather than raised, and an ``OSError`` there means ``taskkill.exe`` itself
    could not be executed — a broken environment worth surfacing, not a dead
    PID worth ignoring. Callers treat killing as best-effort and prove
    ownership separately.
    """
    try:
        pid = int(handle)
    except ValueError:
        return
    if os.name == "nt":
        taskkill = shutil.which("taskkill.exe") or "C:\\Windows\\System32\\taskkill.exe"
        subprocess.run(  # noqa: S603 - PID is parsed as int before invocation.
            [taskkill, "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


def _read_windows_creation_token(pid: int) -> str | None:
    """Return a Windows process's creation FILETIME as an opaque string token.

    Uses ``GetProcessTimes`` (creation time is immutable for a process's
    lifetime), so a reused PID yields a different token. Returns ``None`` when
    the process is gone or the times are unreadable (e.g. access denied) — the
    caller must fail closed on ``None``.
    """
    # ``WinDLL`` is a Windows-only ctypes member (this function only runs on
    # Windows); ``getattr`` keeps the type checker platform-agnostic.
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
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


def _codex_creation_epoch_ms(value: object) -> int:
    """Return a monotonic sort key from a WMI CreationDate JSON value.

    ``ConvertTo-Json`` renders the ``CreationDate`` DateTime either as
    ``/Date(<ms>)/`` or as an ISO-8601 string depending on the PowerShell
    version. Both are monotonically increasing once reduced to their digits, and
    a single query uses one consistent format, so extracting the digits yields a
    valid newest-first ordering key. Unparseable values sort oldest.
    """
    if not isinstance(value, str):
        return 0
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else 0


#: Provably this manager's process: in-memory ownership, or a matching token.
OWNERSHIP_OURS = "ours"
#: Provably NOT the process we mean: token mismatch, no token, or a dead PID.
#: The only answer that may authorize reclaiming a lease or a record claim.
OWNERSHIP_NOT_OURS = "not_ours"
#: Alive, but ownership is unprovable (creation metadata unreadable). Never a
#: licence to reclaim: "I could not check" is not "the holder is gone".
OWNERSHIP_INDETERMINATE = "indeterminate"


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

    def provides_tty(self, backend_type: str, *, is_interactive: bool = False) -> bool:
        """Return whether a spawned agent will run attached to a real TTY.

        Backends whose CLI has both an interactive TUI and a head-less
        entrypoint (Codex: ``codex`` vs ``codex exec``) call this BEFORE
        building the command, so they can pick the TUI when the agent gets a
        real console and fall back to head-less mode when it does not.

        The tmux and Linux-terminal managers always run agents inside a real
        terminal, so the default is True; the Windows manager overrides this
        to mirror its interactive-console decision.
        """
        _ = backend_type, is_interactive
        return True

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

    def ownership_probe(self, handle: str, expected_token: str | None) -> str:
        """Classify ``handle`` as :data:`OWNERSHIP_OURS`/``NOT_OURS``/``INDETERMINATE``.

        ``owns_process`` collapses the last two into ``False``, which is right
        for a **destructive** gate — never kill what you cannot prove is yours —
        but wrong for a **reclaim** gate. Reclaiming a lease or a delivery-record
        claim means "the holder is gone, so I may resume in its place", and an
        unreadable creation token is not proof of that: it is most often a
        transient access error against a process that is very much alive. Acting
        on it authorizes a concurrent resume of one conversation.

        So the three answers are kept apart:

        - ``OURS`` — in-memory ownership of a live child, or a matching live
          creation token.
        - ``NOT_OURS`` — a token mismatch (PID reuse), a tokenless expectation,
          or a PID that is not alive at all. Provably not the holder we mean.
        - ``INDETERMINATE`` — the PID is alive but its creation token could not
          be read, so we cannot tell ownership from reuse. Callers gating a
          reclaim must treat this as "still held".
        """
        if self._has_live_registry_entry(handle):
            return OWNERSHIP_OURS
        if not expected_token:
            return OWNERSHIP_NOT_OURS
        # On Windows a terminated process's immutable creation FILETIME can
        # remain readable briefly while another process still holds a handle.
        # Token equality alone therefore does not prove liveness. Check the
        # exit state first so an owner-bound watcher exits as soon as its
        # coordinator does, even during that retained-handle window.
        if not self._pid_alive(handle):
            return OWNERSHIP_NOT_OURS
        live = creation_token(handle)
        if live is not None:
            return OWNERSHIP_OURS if live == expected_token else OWNERSHIP_NOT_OURS
        # The token is unreadable. A PID that is not alive is settled: gone.
        # A PID that IS alive with an unreadable token is genuinely unknown.
        if self._pid_alive(handle):
            return OWNERSHIP_INDETERMINATE
        return OWNERSHIP_NOT_OURS

    def owns_process(self, handle: str, expected_token: str | None) -> bool:
        """Return whether ``handle`` is provably still our process (fail-closed).

        ``True`` only when EITHER this manager still has current in-memory
        ownership of a live child for ``handle``, OR the live PID's creation
        token equals ``expected_token``. A tokenless expectation, an unreadable
        live token (dead / access denied), or a mismatch all return ``False`` —
        so a reused or foreign PID is never gracefully-shut-down or killed.

        Deliberately still a bool: this gates *destruction*, where "unproven"
        and "not ours" must behave identically. Reclaim gates want
        :meth:`ownership_probe`, which keeps the two apart.
        """
        return self.ownership_probe(handle, expected_token) == OWNERSHIP_OURS

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
class WindowsTerminalTabInfo:
    """Runtime information for an agent launched in a Windows Terminal tab.

    ``pid`` is the in-tab PowerShell launcher's PID (the real process whose
    lifetime wraps the agent's), NOT the transient ``wt.exe`` PID — see
    :meth:`WindowsProcessManager._spawn_in_terminal_tab`. ``token`` is that
    PID's creation token captured at spawn so liveness is PID-reuse-safe.
    """

    pid: int
    token: str | None
    name: str
    agent_id: str
    team_name: str
    backend: str
    launcher: subprocess.Popen[str]
    log_path: Path
    sidecar_path: Path
    # ``None`` for codex tabs, which launch ``codex.exe`` directly (no
    # powershell wrapper) and recover the PID via the correlation token.
    wrapper_path: Path | None
    started_at: float
    exit_logged: bool = False


class WindowsTerminalTabSpawnError(RuntimeError):
    """Raised when a Windows Terminal tab agent never reports its PID."""

    def __init__(self, title: str) -> None:
        """Record which tab failed to confirm its launcher PID."""
        super().__init__(
            f"Windows Terminal tab {title!r} did not report an agent PID "
            f"within {_WT_TAB_PID_TIMEOUT_SECONDS:.0f}s"
        )


class WindowsTerminalTabImmediateExitError(WindowsTerminalTabSpawnError):
    """Raised when a WT tab agent starts but exits within the settle window.

    The tab opened and the wrapper recorded its PID, but the agent process died
    almost immediately — the signature of a degraded WT window that hands the
    tab no usable console, so codex's TUI aborts with "stdin is not a terminal".
    Subclasses :class:`WindowsTerminalTabSpawnError` so a single ``except``
    covers both tab failure modes; the caller falls back to a new console
    window (which always allocates a real TTY).
    """

    def __init__(self, title: str) -> None:
        """Record which tab agent exited immediately after launch."""
        RuntimeError.__init__(
            self,
            f"Windows Terminal tab {title!r} agent exited within "
            f"{_WT_TAB_SETTLE_SECONDS:.1f}s of launch (no usable console)",
        )


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
        # ``WinDLL`` is a Windows-only ctypes member (this function only runs on
        # Windows); ``getattr`` keeps the type checker platform-agnostic.
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
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
        # ``WinDLL`` is a Windows-only ctypes member (this function only runs on
        # Windows); ``getattr`` keeps the type checker platform-agnostic.
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
        kernel32.AssignProcessToJobObject(self._handle, process_handle)

    def close(self) -> None:
        """Close the underlying job handle."""
        if self._handle is None or os.name != "nt":
            return
        # ``WinDLL`` is a Windows-only ctypes member (this function only runs on
        # Windows); ``getattr`` keeps the type checker platform-agnostic.
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
        kernel32.CloseHandle(self._handle)
        self._handle = None


class WindowsProcessManager(_PidOwnershipMixin):
    """Manage spawned agent CLIs through ``subprocess.Popen``."""

    def __init__(self) -> None:
        """Initialize the process registry and shared job object."""
        self._processes: dict[str, ProcessInfo] = {}
        self._tabs: dict[str, WindowsTerminalTabInfo] = {}
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
        if interactive_console and not _env_flag(_NO_WT_TABS_ENV):
            wt = shutil.which("wt.exe")
            if wt is not None:
                try:
                    return self._spawn_in_terminal_tab(
                        request,
                        cmd,
                        env,
                        backend_type,
                        log_path,
                        log_handle,
                        wt=wt,
                        creationflags=creationflags,
                    )
                except WindowsTerminalTabImmediateExitError as err:
                    # The tab opened and ran the wrapper but the agent exited
                    # within the settle window — a degraded WT window that gives
                    # the tab no usable console, so codex's TUI aborts at once.
                    # The process is confirmed dead, so a retry cannot
                    # double-run it: fall back to a dedicated new console
                    # window, which always allocates a real TTY. Reopen the log
                    # the tab helper closed so the console path below can keep
                    # writing to it. (The ambiguous "PID never reported" case
                    # still propagates: the tab may be alive but slow, and a
                    # blind retry there could run the agent twice.)
                    with contextlib.suppress(ValueError, OSError):
                        log_handle.close()
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write(
                        f"[wt tab spawn failed] {err}\n"
                        "[fallback] launching agent in a new console window\n"
                    )
                    log_handle.flush()
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
        self, cmd: list[str], creationflags: int, **kwargs: Any
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
        tab = self._tabs.get(handle)
        if tab is not None:
            return self._tab_health(handle, tab, expected_token)
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
        return cast(ProcessInfo, info).process.poll() is None

    def kill_process(self, handle: str, timeout_s: float = 10.0) -> None:
        """Terminate a process by PID handle, escalating to kill if needed."""
        tab = self._tabs.get(handle)
        if tab is not None:
            self._terminate_tab(handle, tab, timeout_s)
            return
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
        tab = self._tabs.get(handle)
        if tab is not None:
            return self._graceful_shutdown_tab(handle, tab, timeout_s)
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
        tab = self._tabs.get(handle)
        if tab is not None:
            return read_log_tail(tab.log_path, lines)
        info = self._processes.get(handle)
        if info is None:
            return ""
        return read_log_tail(info.log_path, lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Write text to a running process stdin when a pipe exists."""
        if handle in self._tabs:
            return
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
        _force_kill_pid(handle)

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
        # ``WinDLL`` is a Windows-only ctypes member (this function only runs on
        # Windows); ``getattr`` keeps the type checker platform-agnostic.
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
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
            # ``get_last_error`` is a Windows-only ctypes member; ``getattr``
            # keeps the type checker platform-agnostic.
            return getattr(ctypes, "get_last_error")() == _ERROR_ACCESS_DENIED  # noqa: B009
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

    def provides_tty(self, backend_type: str, *, is_interactive: bool = False) -> bool:
        """Return whether a spawned agent will get a real Windows console.

        Both interactive-console paths give the agent a real console TTY: a
        Windows Terminal tab (wrapper ``.ps1`` inside the tab) and the
        ``CREATE_NEW_CONSOLE`` fallback. Only the non-interactive path (stdin
        pipe + log-file stdout) is TTY-less.
        """
        return self._should_use_interactive_console(
            backend_type, is_interactive=is_interactive
        )

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
            stdin=subprocess.DEVNULL,
        )

    def _spawn_in_terminal_tab(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
        log_path: Path,
        log_handle: IO[str],
        *,
        wt: str,
        creationflags: int,
    ) -> SpawnResult:
        """Launch an interactive agent as a tab in a per-team Windows Terminal.

        Agents in the same team share one window (``-w wt-team-<team>``); each
        gets its own ``nt`` tab titled ``<agent>@<team>``. Because ``wt.exe`` is
        a launcher that hands off to a running Windows Terminal and exits at
        once, its PID is useless for lifecycle. Instead a generated wrapper
        ``.ps1`` records the in-tab PowerShell PID to a sidecar file (the same
        pattern the Linux terminal manager uses) and that PID becomes the
        handle, so ``health_check``/``kill``/token liveness all work.
        """
        sidecar_path = log_path.with_name(f"{log_path.stem}.pid")
        with contextlib.suppress(OSError):
            sidecar_path.unlink()
        title = f"{request.name}@{request.team_name}"
        window_id = self._tab_window_id(request.team_name)
        # Shared across both backends. ``--suppressApplicationTitle`` keeps the
        # tab labelled with the agent name (the agent CLI otherwise rewrites the
        # console title at runtime, reverting the tab to a generic name).
        wt_head = [
            wt,
            "-w",
            window_id,
            "nt",
            "--title",
            title,
            "--suppressApplicationTitle",
        ]

        # Legacy direct-launch for codex (codex.exe as the tab's console-root
        # process) is now opt-in only. By default codex takes the same wrapper
        # path as claude: a runtime spike confirmed codex's TUI *and* state
        # hooks work fine under the powershell wrapper (the old "codex must be
        # console root" constraint was really the since-fixed hook-quoting bug),
        # and the wrapper's ``exit 0`` auto-closes the tab on kill/error instead
        # of leaving a lingering ``[process exited]`` tab. Baking the argv into
        # the .ps1 also keeps the prompt off wt's command line entirely.
        codex_direct = backend_type == "codex" and _env_flag(_CODEX_DIRECT_LAUNCH_ENV)
        wrapper_path: Path | None = None
        if codex_direct:
            # ``codex.exe`` launched directly; recover the agent PID afterwards
            # by scanning for the ``codex.exe`` whose argv carries the unique
            # per-agent correlation token. ``-d`` sets the tab's start dir (codex
            # itself also gets ``-C <cwd>`` in build_command).
            #
            # The codex argv sits directly on the wt command line, so wt's own
            # parser sees it. wt treats ``;`` as a sub-command delimiter and
            # splits on it *even inside a quoted token*, which truncates a prompt
            # at its first ``;`` and spawns a junk tab per trailing fragment.
            # Escaping ``;`` -> ``\;`` passes a literal ``;`` to codex intact.
            safe = self._escape_wt_passthrough(cmd)
            wt_cmd = [*wt_head, "-d", request.cwd, "--", *safe]
        else:
            if backend_type == "claude-code":
                cmd = self._with_debug_file(cmd, log_path)
            wrapper_path = log_path.with_name(f"{log_path.stem}.launch.ps1")
            self._write_tab_wrapper(wrapper_path, request.cwd, cmd, env, sidecar_path)
            wt_cmd = [
                *wt_head,
                "--",
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper_path),
            ]
        log_handle.write(
            f"[windows terminal tab] window={window_id!r} title={title!r}\n"
            f"[wt command] {subprocess.list2cmdline(wt_cmd)}\n"
        )
        log_handle.flush()
        log_handle.close()

        # NOTE: a tab attached to an EXISTING Windows Terminal window does not
        # inherit this Popen env. Codex identity is delivered via argv, so the
        # only env at risk is the bundled-tools PATH prepend + CODEX_MANAGED_BY_NPM
        # (build_env). If a second-tab codex fails to resolve bundled tools in
        # smoke, relocate the PATH prepend to a codex ``-c shell_environment_policy``
        # override.
        merged_env = os.environ.copy()
        merged_env.update(env)
        launcher = self._popen(
            wt_cmd,
            creationflags,
            cwd=request.cwd,
            env=merged_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if codex_direct:
            # PID discovery scans argv for the same server-issued marker the
            # codex backend embedded in the prompt. Without an id there is no
            # marker to find, so the discovery cannot succeed — fail loudly
            # rather than scan for a derived token that is not in the argv.
            correlation_id = (request.extra or {}).get(CORRELATION_FIELD)
            pid = (
                self._await_codex_tab_pid(correlation_marker_token(correlation_id))
                if correlation_id
                else None
            )
            if pid is None:
                raise WindowsTerminalTabSpawnError(title)
            # Persist the discovered PID to the same sidecar so restart recovery
            # (_read_pid_file) is unchanged whether or not a wrapper was used.
            with contextlib.suppress(OSError):
                sidecar_path.write_text(str(pid), encoding="utf-8")
        else:
            pid = self._await_tab_pid(sidecar_path)
            if pid is None:
                raise WindowsTerminalTabSpawnError(title)
        handle = str(pid)
        # Capture the creation token BEFORE the settle poll so liveness is
        # PID-reuse-safe across the window: if the tab's PID is recycled by a
        # foreign process mid-settle, the token diverges and the start is
        # correctly treated as an abort rather than a false survival.
        token = creation_token(handle)
        if not self._tab_survived_settle(handle, token):
            self._reap_failed_tab(handle, sidecar_path, wrapper_path)
            raise WindowsTerminalTabImmediateExitError(title)
        self._tabs[handle] = WindowsTerminalTabInfo(
            pid=pid,
            token=token,
            name=request.name,
            agent_id=request.agent_id,
            team_name=request.team_name,
            backend=backend_type,
            launcher=launcher,
            log_path=log_path,
            sidecar_path=sidecar_path,
            wrapper_path=wrapper_path,
            started_at=time.time(),
        )
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    @staticmethod
    def _escape_wt_passthrough(cmd: list[str]) -> list[str]:
        r"""Escape wt.exe's ``;`` command-delimiter in a passthrough argv.

        ``wt … -- <argv>`` puts ``<argv>`` on wt's own command line, where ``;``
        starts a new sub-command (a new tab) -- and wt splits on it even inside
        the double-quoted token ``subprocess`` produces. For a codex agent whose
        prompt contains ``;`` this truncates the prompt at the first ``;`` and
        opens a junk tab for each trailing fragment. wt strips a leading
        backslash, so ``\;`` reaches the child as a literal ``;`` without
        splitting. Applied only to the direct codex launch; the claude launch
        bakes its argv into a ``.ps1`` wrapper and never exposes it to wt.
        """
        return [token.replace(";", r"\;") for token in cmd]

    def _write_tab_wrapper(
        self,
        wrapper_path: Path,
        cwd: str,
        cmd: list[str],
        env: dict[str, str],
        sidecar_path: Path,
    ) -> None:
        """Write the per-spawn PowerShell wrapper that runs the agent in a tab.

        The wrapper sets the per-agent env overrides explicitly (a tab attached
        to an existing Windows Terminal does NOT inherit our env), records its
        own ``$PID`` to the sidecar, then runs the agent in the foreground so
        the PowerShell process's lifetime tracks the agent's. The agent argv is
        baked in as PowerShell single-quoted literals, so the free-form prompt
        argument needs no fragile cross-shell quoting.
        """
        lines = ["$ErrorActionPreference = 'Stop'"]
        lines.extend(
            f"$env:{key} = {_powershell_quote(value)}" for key, value in env.items()
        )
        lines.append(f"Set-Location -LiteralPath {_powershell_quote(cwd)}")
        lines.append(
            f'"$PID" | Out-File -FilePath {_powershell_quote(str(sidecar_path))} '
            f"-Encoding ascii"
        )
        agent_call = "& " + " ".join(_powershell_quote(part) for part in cmd)
        lines.append(agent_call)
        # Always exit 0 so Windows Terminal's default graceful ``closeOnExit``
        # closes the tab when the agent finishes OR is killed. On kill we
        # terminate the agent subtree (not this shell), so control returns here
        # and the shell exits cleanly -- matching the old console's auto-close.
        lines.append("exit 0")
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        # Written as BYTES, deliberately. ``write_text`` opens in text mode, and
        # Python's newline translation would then rewrite every LF to CRLF --
        # including the ones INSIDE the single-quoted prompt literal, so the
        # agent would receive a prompt the caller never sent. The explicit
        # ``\r\n`` join above already gives PowerShell the line endings it wants.
        # BOM required: Windows PowerShell 5.1 decodes a BOM-less file as ANSI
        # (Windows-1252), corrupting non-ASCII bytes (e.g. the correlation-marker
        # em-dash). utf-8-sig writes the BOM so 5.1 reads it as UTF-8.
        wrapper_path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8-sig"))

    def _tab_window_id(self, team_name: str) -> str:
        """Return the Windows Terminal ``-w`` window name for a team.

        Prefixed so an all-numeric team name is not misread by ``wt`` as a
        numeric window id (e.g. ``-w 0`` = most-recent window).
        """
        return f"wt-team-{_validate_safe_name(team_name, 'team name')}"

    def _await_tab_pid(self, sidecar_path: Path) -> int | None:
        """Poll the sidecar for the in-tab PowerShell PID until it appears."""
        deadline = time.monotonic() + _WT_TAB_PID_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            pid = self._read_pid_file(sidecar_path)
            if pid is not None:
                return pid
            time.sleep(_WT_TAB_PID_POLL_SECONDS)
        return self._read_pid_file(sidecar_path)

    def _read_pid_file(self, path: Path) -> int | None:
        try:
            pid_text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(pid_text)
        except ValueError:
            return None

    def _tab_survived_settle(
        self, handle: str, expected_token: str | None = None
    ) -> bool:
        """Return whether a freshly-launched tab agent stays alive past settle.

        A degraded WT window opens the tab and runs the wrapper (so its PID is
        recorded) but hands it no usable console, so codex's TUI aborts within
        ~1s and the wrapper shell exits. Poll briefly: a PID still live at the
        deadline means a healthy start; an early death means an immediate abort
        the caller should retry in a fresh console. ``expected_token`` (the
        PID's creation token captured before the poll) makes this PID-reuse-safe
        — a recycled PID whose token diverges is reported dead. The settle time
        is env-tunable (``WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS``); ``0``
        disables the check and always reports the start as healthy.
        """
        settle = self._tab_settle_seconds()
        if settle <= 0:
            return True
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            alive, _ = self._pid_health_with_token(handle, expected_token)
            if not alive:
                return False
            time.sleep(_WT_TAB_PID_POLL_SECONDS)
        alive, _ = self._pid_health_with_token(handle, expected_token)
        return alive

    @staticmethod
    def _tab_settle_seconds() -> float:
        """Return the tab settle window in seconds (env-overridable, >= 0)."""
        raw = os.environ.get(_WT_TAB_SETTLE_ENV)
        if raw is None:
            return _WT_TAB_SETTLE_SECONDS
        try:
            return max(0.0, float(raw))
        except ValueError:
            return _WT_TAB_SETTLE_SECONDS

    def _reap_failed_tab(
        self, handle: str, sidecar_path: Path, wrapper_path: Path | None
    ) -> None:
        """Kill any lingering tab subtree and remove its wrapper/sidecar files.

        Called when a tab agent aborts immediately so the failed attempt leaves
        nothing behind before the caller retries in a new console window. The
        tab was never registered in ``self._tabs``, so only the OS process and
        the on-disk sidecar/wrapper need cleaning up.
        """
        with contextlib.suppress(OSError, ValueError):
            if self._pid_alive(handle):
                self._kill_pid(handle)
        for path in (sidecar_path, wrapper_path):
            if path is None:
                continue
            with contextlib.suppress(OSError):
                path.unlink()

    def _await_codex_tab_pid(self, token: str) -> int | None:
        """Discover the codex.exe PID whose argv carries ``token``.

        The codex backend appends the per-agent correlation token to the prompt
        argv, so it appears in the child ``codex.exe``'s CommandLine. Poll until
        it shows up (the tab child spawns slightly after ``wt`` returns), then
        return its PID (newest by CreationDate if several match).
        """
        deadline = time.monotonic() + _WT_TAB_PID_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            pid = self._find_codex_pid_by_token(token)
            if pid is not None:
                return pid
            time.sleep(_WT_TAB_PID_POLL_SECONDS)
        return self._find_codex_pid_by_token(token)

    def _find_codex_pid_by_token(self, token: str) -> int | None:
        """Return the codex.exe PID whose CommandLine contains ``token``.

        Matching is done in Python (not a WQL ``LIKE``) to avoid escaping the
        token and to allow newest-first selection on duplicate matches.
        """
        powershell = shutil.which("powershell") or "powershell"
        result = subprocess.run(  # noqa: S603 - fixed argv, token filtered in Python.
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='codex.exe'\" "
                "| Select-Object ProcessId,CommandLine,CreationDate "
                "| ConvertTo-Json -Compress",
            ],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return None
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return None
        records = parsed if isinstance(parsed, list) else [parsed]
        matches: list[tuple[int, int]] = []  # (creation_epoch_ms, pid)
        for record in records:
            if not isinstance(record, dict):
                continue
            command_line = record.get("CommandLine") or ""
            if token not in command_line:
                continue
            try:
                pid = int(record.get("ProcessId"))
            except (TypeError, ValueError):
                continue
            matches.append((_codex_creation_epoch_ms(record.get("CreationDate")), pid))
        if not matches:
            return None
        # Newest CreationDate wins so a stale same-token process can't shadow the
        # freshly spawned tab.
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def _tab_health(
        self, handle: str, tab: WindowsTerminalTabInfo, expected_token: str | None
    ) -> tuple[bool, str]:
        """PID-reuse-safe liveness for a tab agent via its launcher token."""
        alive, detail = self._pid_health_with_token(handle, expected_token or tab.token)
        if not alive and not tab.exit_logged:
            tab.exit_logged = True
            with (
                contextlib.suppress(OSError),
                tab.log_path.open("a", encoding="utf-8") as fh,
            ):
                fh.write(f"[tab agent exited] {detail}\n")
        return alive, detail

    def _graceful_shutdown_tab(
        self, handle: str, tab: WindowsTerminalTabInfo, timeout_s: float
    ) -> bool:
        """Stop a tab agent by terminating its subtree; reap files if it stops."""
        return self._terminate_tab(handle, tab, timeout_s)

    def _terminate_tab(
        self, handle: str, tab: WindowsTerminalTabInfo, timeout_s: float
    ) -> bool:
        """Kill the agent subtree so the wrapper shell exits 0 and the tab closes.

        The handle is the wrapper PowerShell's PID; its children are the agent
        (and any grandchildren). Killing the children -- not the shell -- lets
        the shell's ``& agent`` call return and reach ``exit 0``, so Windows
        Terminal's graceful ``closeOnExit`` removes the tab. Only if the shell
        refuses to exit is it force-killed (leaving a lingering tab, but a dead
        agent).
        """
        if tab.wrapper_path is None:
            # Codex tab: the handle IS the codex PID (no wrapper shell in
            # between), so kill its subtree directly. WT is left showing a
            # "[process exited]" tab because codex exits non-zero on kill -- the
            # wrapper's ``exit 0`` auto-close trick is unavailable here.
            stopped = not self._pid_alive(handle)
            if not stopped:
                self._kill_pid(handle)  # taskkill /PID <pid> /T /F
                stopped = self._win_wait_pid_exit(handle, timeout_s) or (
                    not self._pid_alive(handle)
                )
            self._tabs.pop(handle, None)
            self._cleanup_tab_files(tab)
            return stopped
        stopped = not self._pid_alive(handle)
        if not stopped:
            for child in self._child_pids(handle):
                self._kill_pid(str(child))
            stopped = self._win_wait_pid_exit(handle, timeout_s)
            if not stopped:
                self._kill_pid(handle)
                stopped = not self._pid_alive(handle)
        self._tabs.pop(handle, None)
        self._cleanup_tab_files(tab)
        return stopped

    def _cleanup_tab_files(self, tab: WindowsTerminalTabInfo) -> None:
        for path in (tab.sidecar_path, tab.wrapper_path):
            if path is None:
                continue
            with contextlib.suppress(OSError):
                path.unlink()

    def _child_pids(self, handle: str) -> list[int]:
        """Return the direct child PIDs of ``handle`` (the wrapper shell)."""
        try:
            pid = int(handle)
        except ValueError:
            return []
        powershell = shutil.which("powershell") or "powershell"
        result = subprocess.run(  # noqa: S603 - PID is parsed as int before use.
            [
                powershell,
                "-NoProfile",
                "-Command",
                f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' "
                f"| Select-Object -ExpandProperty ProcessId",
            ],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        pids: list[int] = []
        for token in result.stdout.split():
            try:
                pids.append(int(token))
            except ValueError:
                continue
        return pids

    def _win_wait_pid_exit(self, handle: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pid_alive(handle):
                return True
            time.sleep(0.05)
        return not self._pid_alive(handle)


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
            errors="replace",
            env=merged_env,
            stdin=subprocess.DEVNULL,
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
        alive, _ = self._pane_alive(cast(TmuxProcessInfo, info).pane_id)
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
            errors="replace",
            stdin=subprocess.DEVNULL,
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
            errors="replace",
            stdin=subprocess.DEVNULL,
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
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        if enter:
            subprocess.run(  # noqa: S603 - tmux argv is built internally.
                [tmux, "send-keys", "-t", info.pane_id, "Enter"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                stdin=subprocess.DEVNULL,
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
        """Parse ``@window<TAB>%pane<TAB>pid``, refusing anything that is not that.

        All three fields are a machine protocol and all three are later used to
        ADDRESS the pane. Counting the fields is not enough: a malformed window
        or pane id registers a spawn that reports success and can then never be
        health-checked, signalled or killed. Validate each one.
        """
        fields = output.strip().split("\t")
        if len(fields) != _TMUX_SPAWN_FIELD_COUNT:
            msg = f"Unexpected tmux spawn output: {output!r}"
            raise RuntimeError(msg)
        window_id, pane_id, pid_text = fields
        if not _TMUX_WINDOW_ID.fullmatch(window_id):
            msg = f"Unexpected tmux window id: {window_id!r}"
            raise RuntimeError(msg)
        if not _TMUX_PANE_ID.fullmatch(pane_id):
            msg = f"Unexpected tmux pane id: {pane_id!r}"
            raise RuntimeError(msg)
        if not _TMUX_PANE_PID.fullmatch(pid_text):
            msg = f"Unexpected tmux pane PID: {pid_text!r}"
            raise RuntimeError(msg)
        return window_id, pane_id, int(pid_text)

    def _pane_alive(self, pane_id: str) -> tuple[bool, str]:
        tmux = self._tmux_binary()
        result = subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, "display-message", "-p", "-t", pane_id, "#{pane_dead}"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "tmux target not found"
        # ``#{pane_dead}`` is a machine protocol: tmux answers exactly "0" or
        # "1". Anything else is an unreadable answer, and _tracked_alive uses
        # this result as PROOF that a pane is ours -- so an unreadable answer
        # has to fail closed. Reading "not 1" as alive would let a replacement
        # character, or any unexpected text, claim ownership of a pane we
        # cannot actually address.
        status = result.stdout.strip()
        if status == "1":
            return False, "tmux pane dead"
        if status != "0":
            return False, f"unreadable tmux pane status: {status!r}"
        return True, "tmux pane running"

    def _kill_tmux_target(self, target_id: str) -> None:
        command = "kill-window" if target_id.startswith("@") else "kill-pane"
        tmux = self._tmux_binary()
        subprocess.run(  # noqa: S603 - tmux argv is built internally.
            [tmux, command, "-t", target_id],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )

    def _kill_pid(self, handle: str) -> None:
        _force_kill_pid(handle)

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
            errors="replace",
            stdin=subprocess.DEVNULL,
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
        """Ownership is proven by OUR terminal launcher child, never a bare PID.

        Deliberately does NOT trust the sidecar agent PID's bare liveness: that
        PID can be reused by a foreign process after the agent exits, and this
        result gates destructive ops via ``owns_process``. If the launcher child
        exited we fall through to the token comparison (fail closed). The
        sidecar PID's liveness is still used for non-destructive display/cleanup
        in ``server_simple._agent_alive`` (the accepted residual).
        """
        return cast(LinuxTerminalProcessInfo, info).terminal_process.poll() is None

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
            "foot",
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
            errors="replace",
            stdin=subprocess.DEVNULL,
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
        _force_kill_pid(handle)


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


class HerdrCommandError(RuntimeError):
    """A Herdr CLI call failed, or answered with something we cannot trust.

    Carries the machine-readable ``code`` when Herdr supplied one so callers
    can tell "that object is gone" (settled) from "the control plane is
    unavailable" (indeterminate) without re-parsing text.
    """

    def __init__(self, argv: list[str], code: str, detail: str) -> None:
        """Build the message from the command, Herdr's error code and detail."""
        self.code = code
        self.detail = detail
        super().__init__(f"herdr {' '.join(argv[1:])!r} failed [{code}]: {detail}")


def _require_pane(created: dict[str, Any], tab_id: str) -> dict[str, Any]:
    """Return the create response's root pane, or fail with a named error."""
    pane = created.get("root_pane")
    if not isinstance(pane, dict):
        msg = "herdr tab create returned no pane"
        raise HerdrCommandError(["herdr", "tab", "create"], "malformed", msg)
    if not pane.get("pane_id") or not tab_id:
        msg = "herdr tab create returned no pane_id/tab_id"
        raise HerdrCommandError(["herdr", "tab", "create"], "malformed", msg)
    return pane


def _pid_is_live(pid: int) -> bool:
    """Whether ``pid`` is a live, non-zombie process.

    The zombie check is not optional: a reaped-but-unwaited process still
    answers ``kill(pid, 0)`` AND still has a readable ``/proc`` creation
    token, so without it a dead agent reads as owned-and-alive -- and a
    graceful shutdown would wait out its entire timeout on a corpse.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return not _pid_is_zombie(pid)
    except OSError:
        return False
    return not _pid_is_zombie(pid)


def _pid_is_zombie(pid: int) -> bool:
    """Whether ``pid`` is a reaped-but-unwaited zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    closing = stat.rfind(")")
    if closing == -1:
        return False
    fields = stat[closing + 1 :].split()
    return bool(fields) and fields[0] == "Z"


def _require_creation_token(handle: str) -> str:
    """Return a non-empty creation token for ``handle``, or refuse to continue.

    A null token cannot prove ownership later: ``_probe`` compares stored and
    live tokens, and ``None == None`` would forge a match for a PID we know
    nothing about.
    """
    token = creation_token(handle)
    if not token:
        msg = (
            f"could not read a creation token for pane PID {handle}; refusing "
            "to register an agent whose ownership cannot be proven"
        )
        raise HerdrSpawnError(msg)
    return token


class HerdrServerUnavailableError(RuntimeError):
    """No Herdr server could be reached or started for the selected session."""


class HerdrOwnershipUnprovenError(RuntimeError):
    """A stop was requested for a live PID whose ownership cannot be proven."""


class HerdrSpawnError(RuntimeError):
    """A Herdr tab was created but the agent could not be safely registered."""


class _HerdrProbe(Enum):
    """What we can actually prove about a tracked Herdr agent right now.

    A bool cannot carry this: a control-plane hiccup and a dead process must
    lead to opposite answers for ``health_check`` (stay alive) and for
    anything that kills (refuse to act).
    """

    OWNED = "owned"
    #: The pane is gone but the process is not: Herdr gives a *moved* pane a
    #: new workspace-qualified id, so the agent outlives our stored pane id.
    PANE_GONE = "pane_gone"
    PID_GONE = "pid_gone"
    IDENTITY_MISMATCH = "identity_mismatch"
    INDETERMINATE = "indeterminate"


class _WorkspaceLookup(Enum):
    """What a ``workspace list`` actually told us about a label.

    "No such workspace" and "could not read the listing" demand opposite
    actions -- create one, versus leave the routing to Herdr -- so they must
    never collapse into a single falsy answer.
    """

    #: A workspace carries this label; its id is usable.
    MATCH = "match"
    #: The listing parsed and authoritatively holds no such label.
    EMPTY = "empty"
    #: The listing failed or could not be trusted. Not evidence of absence.
    UNKNOWN = "unknown"


@dataclass
class HerdrProcessInfo:
    """Runtime information for an agent spawned into a Herdr tab."""

    pid: int
    #: Never ``None``: a null token would make the later equality check
    #: ``None == None`` and forge ownership of a reused PID.
    creation_token: str
    session_name: str | None
    socket_endpoint: str | None
    name: str
    agent_id: str
    team_name: str
    backend: str
    tab_id: str
    pane_id: str
    workspace_id: str
    log_path: Path
    started_at: float


class HerdrProcessManager(_PidOwnershipMixin):
    """Manage spawned agent CLIs as tabs in a Herdr workspace."""

    def __init__(self) -> None:
        """Capture the configured session name. Pure: no probe, no subprocess.

        The manager is constructed at import time, so resolving the socket
        endpoint here would make importing this module start talking to (or
        starting) a Herdr server. The endpoint is resolved on first spawn.
        """
        self._processes: dict[str, HerdrProcessInfo] = {}
        self._session = self._configured_session()
        self._workspace_override = self._configured_workspace()
        self.socket_endpoint: str | None = None
        #: A server WE started, retained so it can be reaped.
        self._server_child: subprocess.Popen[bytes] | None = None

    @staticmethod
    def _configured_session() -> str | None:
        """Return the pinned session name, or ``None`` when unpinned."""
        raw = os.environ.get(_HERDR_SESSION_ENV, "").strip()
        if not raw:
            return None
        if not _HERDR_SESSION_RE.match(raw):
            msg = (
                f"Invalid Herdr session name: {raw!r}. Expected "
                f"{_HERDR_SESSION_RE.pattern}."
            )
            raise ValueError(msg)
        return raw

    @staticmethod
    def _configured_workspace() -> str | None:
        """Return the pinned workspace label, the sentinel, or ``None``.

        Unlike a session name, a label is display text passed as one argv
        token, so ``repo.name`` or ``my repo`` are perfectly ordinary and must
        be accepted. Only what would corrupt a terminal, be read as a flag, or
        overflow a filesystem is refused -- and only for an EXPLICIT override:
        a label derived from a folder is sanitised instead (a repo may not be
        renamed just to be spawnable).
        """
        raw = os.environ.get(_HERDR_WORKSPACE_ENV, "")
        if not raw.strip():
            return None
        label = raw.strip()
        if label == _HERDR_ACTIVE_WORKSPACE_SENTINEL:
            return label
        if (
            label.startswith("-")
            or len(label) > _HERDR_LABEL_MAX_LEN
            or not label.isprintable()
        ):
            msg = (
                f"Invalid Herdr workspace label: {raw!r}. Expected 1-"
                f"{_HERDR_LABEL_MAX_LEN} printable characters, no leading '-' "
                f"(use {_HERDR_ACTIVE_WORKSPACE_SENTINEL!r} for the active "
                "workspace)."
            )
            raise ValueError(msg)
        return label

    def _herdr_argv(self, *args: str) -> list[str]:
        """Build a Herdr command, routed to our session.

        ``--session`` is a TOP-LEVEL prefix, not a per-subcommand flag, and
        every call must go through here: pane and tab ids only mean anything
        within one session, so an unrouted ``tab close`` could shut a tab
        belonging to the user.
        """
        base = [self._binary()]
        if self._session:
            base += ["--session", self._session]
        return [*base, *args]

    @staticmethod
    def _binary() -> str:
        """Return the herdr binary name (resolved by PATH at call time)."""
        return "herdr"

    def _run_herdr(
        self,
        *args: str,
        expect: str,
        timeout: float = _HERDR_CALL_TIMEOUT_SECONDS,
        allow_empty_success: bool = False,
    ) -> dict[str, Any]:
        """Run one finite Herdr control command and return its validated result.

        Every failure mode gets a distinguishable ``code`` because callers act
        on the difference: ``not_found`` is settled ("that object is gone"),
        while ``timeout`` and friends are indeterminate and must never be
        allowed to read as a dead agent.
        """
        argv, completed = self._invoke(args, timeout)
        envelope = self._parse_envelope(
            argv, completed, allow_empty=allow_empty_success
        )
        return self._validated_result(
            argv, completed, envelope, expect, allow_empty_success=allow_empty_success
        )

    def _invoke(
        self, args: tuple[str, ...], timeout: float, errors: str = "strict"
    ) -> tuple[list[str], subprocess.CompletedProcess[str]]:
        """Run one herdr command, mapping process-level failures to our error.

        ``errors`` is the decode policy: strict for JSON (a machine protocol
        we parse), replace for terminal display text.
        """
        argv = self._herdr_argv(*args)
        try:
            completed = subprocess.run(  # noqa: S603 - argv is built internally.
                argv,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors=errors,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise HerdrCommandError(argv, "timeout", str(exc)) from exc
        except UnicodeDecodeError as exc:
            raise HerdrCommandError(argv, "malformed", str(exc)) from exc
        except OSError as exc:
            raise HerdrCommandError(argv, "unavailable", str(exc)) from exc
        return argv, completed

    def _run_herdr_raw(
        self, *args: str, timeout: float = _HERDR_CALL_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        """Run a ``--json`` query that answers with a bare object.

        ``status server``/``status client``/``session list`` return plain
        objects rather than the ``{"id", "result": {"type": ...}}`` envelope
        the control commands use.
        """
        argv, completed = self._invoke(args, timeout)
        envelope = self._parse_envelope(argv, completed)
        error = envelope.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "error")
            raise HerdrCommandError(argv, code, str(error.get("message") or ""))
        if completed.returncode != 0:
            raise HerdrCommandError(argv, "error", completed.stderr.strip())
        return envelope

    def _run_herdr_text(
        self, *args: str, timeout: float = _HERDR_CALL_TIMEOUT_SECONDS
    ) -> str:
        """Run a command that prints plain text rather than JSON.

        ``pane read`` emits the terminal snapshot itself on stdout -- there is
        no envelope to validate, so this seam returns the text as-is.

        Decoding is deliberately TOLERANT here, unlike the JSON path: this is
        display text from an arbitrary program's terminal, so a stray byte
        should show as U+FFFD rather than fail the whole read. The strict
        policy exists to protect a protocol we parse and act on; nothing acts
        on this.
        """
        argv, completed = self._invoke(args, timeout, errors="replace")
        if completed.returncode != 0:
            raise HerdrCommandError(argv, "error", completed.stderr.strip())
        return completed.stdout or ""

    @staticmethod
    def _validated_result(
        argv: list[str],
        completed: subprocess.CompletedProcess[str],
        envelope: dict[str, Any],
        expect: str,
        *,
        allow_empty_success: bool = False,
    ) -> dict[str, Any]:
        """Validate an enveloped control response and return its result object."""
        error = envelope.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "error")
            raise HerdrCommandError(argv, code, str(error.get("message") or ""))
        if completed.returncode == _HERDR_USAGE_EXIT_CODE:
            raise HerdrCommandError(argv, "cli_usage", completed.stderr.strip())
        if completed.returncode != 0:
            raise HerdrCommandError(argv, "error", completed.stderr.strip())

        result = envelope.get("result")
        if result is None and not envelope and completed.returncode == 0:
            if not allow_empty_success:
                # Only commands observed to answer silently may be assumed to
                # have succeeded. Generalising it would let a response-bearing
                # command's empty answer fabricate the semantics we asked for.
                raise HerdrCommandError(
                    argv, "malformed", "empty response from a command that answers"
                )
            return {"type": expect}
        if not isinstance(result, dict):
            raise HerdrCommandError(argv, "malformed", "response has no result object")
        if result.get("type") != expect:
            raise HerdrCommandError(
                argv,
                "malformed",
                f"expected result type {expect!r}, got {result.get('type')!r}",
            )
        return result

    def _ensure_server(self) -> str:
        """Return the socket for our session, starting a server only if truly absent.

        Reuses whatever already answers -- including the session the user is
        sitting in -- and revalidates on every spawn, because a server that
        died since the last spawn must be noticed rather than cached forever.
        """
        self._reap_server_child()
        endpoint = self._server_socket()
        if endpoint is not None:
            self._note_endpoint(endpoint)
            return endpoint

        # POSIX flock blocks; only the readiness poll inside is bounded.
        with file_lock(self._start_lock_path()):
            endpoint = self._server_socket()  # another process may have won
            if endpoint is None:
                endpoint = self._start_server()
        self._note_endpoint(endpoint)
        return endpoint

    def _note_endpoint(self, endpoint: str) -> None:
        """Record the live endpoint, so a handoff becomes visible to _probe."""
        self.socket_endpoint = endpoint

    def _reap_server_child(self) -> None:
        """Reap a server we started that has since exited, so it leaves no zombie."""
        child = self._server_child
        if child is not None and child.poll() is not None:
            self._server_child = None

    def _server_socket(self) -> str | None:
        """Return the running server's socket, or ``None`` only if *confirmed* absent.

        ``status --json`` is NOT enveloped like the control commands: it
        answers with a bare object, so it is read through ``_run_herdr_raw``.

        The distinction that matters here is "confirmed absent" versus "could
        not tell". A timeout against a busy live server, a malformed answer or
        a missing binary are not evidence that no server exists, and treating
        them as such would start a second daemon beside a healthy one.
        """
        status = self._run_herdr_raw("status", "server", "--json")
        running = status.get("running")
        if running is False:
            return None
        if running is not True:
            msg = f"status server returned no usable 'running' flag: {status!r}"
            raise HerdrCommandError(
                self._herdr_argv("status", "server"), "malformed", msg
            )
        socket = status.get("socket")
        if isinstance(socket, str) and socket:
            return socket
        return self._session_socket_from_list()

    def _session_socket_from_list(self) -> str:
        """Ask ``session list`` for our session's socket path.

        Never guessed: the real layout for a named session is
        ``<config>/sessions/<name>/herdr.sock``, which a hand-built path got
        wrong, and it moves with ``HERDR_CONFIG_PATH``.
        """
        listing = self._run_herdr_raw("session", "list", "--json")
        wanted = self._session or "default"
        for row in listing.get("sessions") or []:
            if isinstance(row, dict) and row.get("name") == wanted:
                socket = row.get("socket_path")
                if isinstance(socket, str) and socket:
                    return socket
        msg = f"session {wanted!r} has no socket_path in session list"
        raise HerdrCommandError(self._herdr_argv("session", "list"), "malformed", msg)

    def _start_server(self) -> str:
        """Start a headless Herdr server and wait, bounded, for it to answer."""
        try:
            child = self._popen_herdr_server()
        except OSError as exc:
            msg = f"could not launch a herdr server: {exc}"
            raise HerdrServerUnavailableError(msg) from exc
        self._server_child = child
        deadline = time.monotonic() + _HERDR_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if child.poll() is not None:
                self._server_child = None
                msg = (
                    f"herdr server exited immediately (rc={child.returncode}) for "
                    f"session {self._session or 'default'}"
                )
                raise HerdrServerUnavailableError(msg)
            try:
                endpoint = self._server_socket()
            except HerdrCommandError:
                endpoint = None  # still coming up
            if endpoint is not None:
                return endpoint
            time.sleep(_HERDR_START_POLL_SECONDS)
        # Never became ready: stop the child WE started rather than leaking a
        # half-started daemon for the next process to trip over.
        self._terminate_server_child()
        msg = (
            "herdr server did not become ready within "
            f"{_HERDR_START_TIMEOUT_SECONDS:.0f}s for session "
            f"{self._session or 'default'}; start it with "
            f"{' '.join(self._herdr_argv('server'))}"
        )
        raise HerdrServerUnavailableError(msg)

    def _terminate_server_child(self) -> None:
        """Stop only the server this manager started, best-effort and bounded."""
        child = self._server_child
        self._server_child = None
        if child is None or child.poll() is not None:
            return
        with contextlib.suppress(OSError):
            child.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            child.wait(timeout=_HERDR_TERMINATE_TIMEOUT_SECONDS)
            return
        with contextlib.suppress(OSError):
            child.kill()
        # Reap it, or the force-killed child lingers as a zombie.
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            child.wait(timeout=_HERDR_TERMINATE_TIMEOUT_SECONDS)

    def _popen_herdr_server(self) -> subprocess.Popen[bytes]:
        """Launch the long-running daemon.

        Its own seam, separate from ``_run_herdr``: ``subprocess.run`` would
        block for the server's whole lifetime and yields no JSON to validate.
        """
        return subprocess.Popen(  # noqa: S603 - argv is built internally.
            self._herdr_argv("server"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _start_lock_path(self) -> Path:
        """Return the lock path shared by everyone targeting this session.

        Keyed by the Herdr session, under Herdr's own config directory, not by
        our session directory: two unrelated team sessions aiming at the same
        server must contend for the SAME lock, and the path has to exist before
        the server does. Honours ``HERDR_CONFIG_PATH`` so a relocated config
        does not silently split the lock in two.
        """
        return self._config_root() / (
            f"win-agent-teams-{self._session or 'default'}.start.lock"
        )

    def _config_root(self) -> Path:
        """Return Herdr's config directory, honouring ``HERDR_CONFIG_PATH``.

        HERDR_CONFIG_PATH "overrides config file path" (herdr --help): it names
        the config FILE, so the directory is its parent. Treating it as a
        directory would make file_lock's mkdir fail inside an existing file --
        or, when it does not exist yet, create a directory exactly where Herdr
        later wants to write its config.
        """
        config = os.environ.get("HERDR_CONFIG_PATH")
        return (
            Path(config).expanduser().parent
            if config
            else Path.home() / ".config" / "herdr"
        )

    def _workspace_lock_path(self, label: str) -> Path:
        """Return the lock shared by everyone creating this repo's workspace.

        A label is free-form text -- it can hold ``/``, ``..``, spaces or 128
        characters of Unicode -- so it is hashed rather than interpolated: a
        raw label would otherwise escape the directory or overflow the name
        limit. The session is encoded in the name the same way
        ``_start_lock_path`` does it, so two Herdr sessions working on the same
        repo do not serialise against each other.
        """
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
        session = self._session or "default"
        return self._config_root() / f"win-agent-teams-{session}.ws-{digest}.lock"

    def _workspace_label(self, cwd: str) -> str | None:
        """Return the workspace label for an agent working in ``cwd``.

        ``None`` means "do not route at all": the sentinel restores the
        pre-per-repo behaviour of using whatever workspace is active.
        """
        override = self._workspace_override
        if override == _HERDR_ACTIVE_WORKSPACE_SENTINEL:
            return None
        if override:
            return override
        return self._sanitise_label(self._repo_name(cwd))

    def _repo_name(self, cwd: str) -> str:
        """Name the repository ``cwd`` belongs to, else the folder itself.

        The git common dir is the MAIN checkout's, so every worktree of one
        repo answers with the same name and therefore shares one workspace --
        which is what the user asked for. Everything about this is best-effort:
        a missing, broken or slow git must degrade to the folder name, never
        fail a spawn.
        """
        common_dir = self._git_common_dir(cwd)
        if common_dir:
            path = Path(common_dir)
            # A normal repo's common dir is "<checkout>/.git"; a bare one is
            # the repository itself, conventionally "<name>.git".
            # removesuffix, not .stem: a bare repo named "portal.data" keeps
            # its name, while "portal.git" loses only the git suffix.
            name = (
                path.parent.name
                if path.name == ".git"
                else path.name.removesuffix(".git")
            )
            if name:
                return name
        return Path(cwd).name

    @staticmethod
    def _git_common_dir(cwd: str) -> str | None:
        """Ask git which repository ``cwd`` belongs to, or give up quietly."""
        # Resolved through PATH at call time, exactly like the herdr binary:
        # pinning an absolute path would break every non-standard install.
        argv = [
            "git",
            "-C",
            cwd,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - argv is built internally.
                argv,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                # Strict, unlike the terminal-display reads: a path we cannot
                # decode is a failed probe, not a label full of U+FFFD.
                errors="strict",
                timeout=_HERDR_CALL_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    @staticmethod
    def _sanitise_label(raw: str) -> str:
        """Coerce a derived name into a label Herdr and a lock file can hold."""
        cleaned = "".join(ch for ch in raw if ch.isprintable()).strip()
        cleaned = cleaned.lstrip("-").strip()
        return cleaned[:_HERDR_LABEL_MAX_LEN] or _HERDR_DEFAULT_LABEL

    def _resolve_workspace(self, label: str) -> tuple[_WorkspaceLookup, str | None]:
        """Find the workspace carrying ``label``, or say why we cannot.

        Herdr allows duplicate labels, so the choice must be deterministic --
        otherwise repeated spawns of one repo would scatter across the
        duplicates. Anything unreadable degrades to ``UNKNOWN``: only a
        listing we fully understood may be read as absence, because absence is
        what makes us create a workspace.
        """
        try:
            listing = self._run_herdr("workspace", "list", expect="workspace_list")
        except HerdrCommandError:
            return _WorkspaceLookup.UNKNOWN, None
        workspaces = listing.get("workspaces")
        if not isinstance(workspaces, list):
            return _WorkspaceLookup.UNKNOWN, None

        candidates: list[tuple[int, str]] = []
        for entry in workspaces:
            if not isinstance(entry, dict):
                return _WorkspaceLookup.UNKNOWN, None
            if entry.get("label") != label:
                continue
            workspace_id = entry.get("workspace_id")
            if not isinstance(workspace_id, str) or not workspace_id:
                # The one candidate we cannot read is exactly the one that must
                # not be mistaken for absence: creating would duplicate it.
                return _WorkspaceLookup.UNKNOWN, None
            number = entry.get("number")
            key = (
                number
                if isinstance(number, int) and not isinstance(number, bool)
                else sys.maxsize
            )
            candidates.append((key, workspace_id))
        if not candidates:
            return _WorkspaceLookup.EMPTY, None
        return _WorkspaceLookup.MATCH, min(candidates)[1]

    def spawn_process(
        self,
        request: SpawnRequest,
        cmd: list[str],
        env: dict[str, str],
        backend_type: str,
        *,
        is_interactive: bool = False,
    ) -> SpawnResult:
        """Start an agent in a fresh Herdr tab and return its pane PID handle."""
        _ = is_interactive
        if backend_type == "claude-code":
            cmd = self._with_debug_file(
                cmd, self.log_path(request.team_name, request.name)
            )

        log_path = self.log_path(request.team_name, request.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint = self._ensure_server()
        command = _build_posix_shell_command(request.cwd, cmd, env)

        label = self._workspace_label(request.cwd)
        created = self._create_tab(request, env, label)
        # The rollback scope opens HERE, the moment a tab may exist -- BEFORE
        # the response is validated. A malformed create response that still
        # carried a real tab id would otherwise raise straight past cleanup and
        # leave an orphan tab, and the live findings showed response shapes are
        # exactly what must not be assumed.
        tab_id = self._tab_id_of(created)
        try:
            pane = _require_pane(created, tab_id)
            pane_id = str(pane["pane_id"])
            workspace_id = str(pane.get("workspace_id") or "")

            self._run_herdr(
                "pane", "run", pane_id, command, expect="ok", allow_empty_success=True
            )
            pid = self._pane_shell_pid(pane_id)
            handle = str(pid)
            token = _require_creation_token(handle)
            # Provenance is written BEFORE the record is committed, so a failing
            # log write cannot leave a live agent the caller believes never
            # started.
            self._write_provenance(
                log_path, tab_id, pane_id, handle, workspace_id, label
            )
        except BaseException as exc:
            self._rollback_tab(tab_id, exc)
            raise

        self._processes[handle] = HerdrProcessInfo(
            pid=pid,
            creation_token=token,
            session_name=self._session,
            socket_endpoint=endpoint,
            name=request.name,
            agent_id=request.agent_id,
            team_name=request.team_name,
            backend=backend_type,
            tab_id=tab_id,
            pane_id=pane_id,
            workspace_id=workspace_id,
            log_path=log_path,
            started_at=time.time(),
        )
        return SpawnResult(process_handle=handle, backend_type=backend_type)

    def _probe(self, info: HerdrProcessInfo) -> _HerdrProbe:
        """Classify what we can prove about ``info`` right now.

        Local process identity is settled FIRST, because it is immutable and
        cheap: a PID plus its creation token says whether our original process
        still exists regardless of what Herdr can currently say. Only then does
        the pane have anything to add. Asking Herdr first lets one unreadable
        token, or one timed-out CLI call, decide liveness -- in opposite
        directions, both wrong.
        """
        local = self._local_identity(info)
        if local is not None:
            return local
        # Our process is alive and provably still ours. Anything the pane says
        # from here can only downgrade manageability, never liveness.
        try:
            pid = self._pane_shell_pid(info.pane_id)
        except HerdrCommandError as exc:
            if exc.code in _HERDR_NOT_FOUND_CODES:
                return _HerdrProbe.PANE_GONE
            return _HerdrProbe.INDETERMINATE
        if pid != info.pid:
            # The pane now hosts someone else: our agent is alive but is no
            # longer reachable through this object.
            return _HerdrProbe.PANE_GONE
        # The pane, the PID and the token all check out. If the endpoint moved
        # underneath us (a supported ``herdr --handoff`` keeps panes alive),
        # that is the proof needed to rebind -- and it is the same immutable
        # session selector, because every call went through _herdr_argv.
        if info.socket_endpoint != self.socket_endpoint:
            self._log_rebind(info, self.socket_endpoint)
            info.socket_endpoint = self.socket_endpoint
        return _HerdrProbe.OWNED

    @staticmethod
    def _local_identity(info: HerdrProcessInfo) -> _HerdrProbe | None:
        """Settle PID/token identity, or ``None`` when the process is still ours.

        ``None`` means "still ours, ask Herdr next"; anything else is already
        decided without Herdr's help.
        """
        stored = info.creation_token
        if not stored:
            # Spawn refuses to register a null token, so this is a corrupt
            # record and must never read as ownership.
            return _HerdrProbe.IDENTITY_MISMATCH
        live = creation_token(str(info.pid))
        if live is None:
            # Unreadable: either the PID is gone or we merely could not read
            # it. creation_token cannot distinguish those, so ask separately.
            if not _pid_is_live(info.pid):
                return _HerdrProbe.PID_GONE
            return _HerdrProbe.INDETERMINATE
        if live != stored:
            # Readable and different: the number was recycled, so our original
            # process is definitively gone.
            return _HerdrProbe.IDENTITY_MISMATCH
        return None

    @staticmethod
    def _log_rebind(info: HerdrProcessInfo, new_endpoint: str | None) -> None:
        """Record an endpoint rebind: it is a real lifecycle event."""
        with (
            contextlib.suppress(OSError),
            info.log_path.open("a", encoding="utf-8") as handle,
        ):
            handle.write(
                f"[herdr] rebound {info.name} from {info.socket_endpoint} to "
                f"{new_endpoint} (pane={info.pane_id} pid={info.pid} "
                "token matched)\n"
            )

    def _tracked_alive(self, info: object) -> bool:
        """Ownership proof for the mixin's in-memory short-circuit.

        ``ownership_probe`` returns OURS without re-reading the token the
        moment this is true, so only a fully proven ``OWNED`` may pass.
        """
        return self._probe(cast(HerdrProcessInfo, info)) is _HerdrProbe.OWNED

    def health_check(
        self, handle: str, expected_token: str | None = None
    ) -> tuple[bool, str]:
        """Report liveness, without turning control-plane trouble into death."""
        info = self._processes.get(handle)
        if info is None:
            return self._pid_health_with_token(handle, expected_token)
        state = self._probe(info)
        if state is _HerdrProbe.OWNED:
            return True, f"herdr pane {info.pane_id} alive"
        if state in (_HerdrProbe.PANE_GONE, _HerdrProbe.INDETERMINATE):
            # The process and its token still match; only Herdr's view of it
            # is unavailable (a moved pane, a timed-out CLI).
            return True, f"degraded: {state.value}; pid {info.pid} still ours"
        return False, f"herdr agent {state.value}"

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Type into the agent's pane -- only when the pane is provably ours."""
        info = self._owned_info(handle, "send")
        if info is None:
            return
        self._run_herdr("pane", "send-text", info.pane_id, text, expect="ok")
        if enter:
            self._run_herdr("pane", "send-keys", info.pane_id, "enter", expect="ok")

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Read recent pane output, or ``""`` when the pane is not provably ours."""
        if lines is not None and lines <= 0:
            return ""
        info = self._owned_info(handle, "capture")
        if info is None:
            return ""
        for source in ("recent-unwrapped", "visible"):
            args = ["pane", "read", info.pane_id, "--source", source]
            if lines is not None:
                args += ["--lines", str(lines)]
            text = self._run_herdr_text(*args)
            if text:
                return text
        return ""

    def kill_process(self, handle: str, timeout_s: float = 10.0) -> None:
        """Stop an agent, scoping each action to what is actually proven.

        Closing the tab is a Herdr *object* operation, so it needs proven
        object identity; signalling the PID needs only a matching creation
        token. Those are different proofs, and conflating them is how a
        recycled PID gets killed or a moved-pane agent gets orphaned.
        """
        info = self._processes.get(handle)
        if info is None:
            self._kill_pid(handle)
            return
        state = self._probe(info)
        if state is _HerdrProbe.OWNED:
            with contextlib.suppress(HerdrCommandError):
                self._run_herdr("tab", "close", info.tab_id, expect="ok")
            if self._wait_pid_exit(info.pid, timeout_s):
                self._processes.pop(handle, None)
                return
            self._force_kill_settled(handle, info)
            return
        if state in (_HerdrProbe.PID_GONE, _HerdrProbe.IDENTITY_MISMATCH):
            # The original process is provably gone; there is nothing to signal.
            self._processes.pop(handle, None)
            return
        # PANE_GONE / INDETERMINATE: no tab operation, because the id we hold
        # may no longer be ours -- but the process may well still be running.
        self._force_kill_settled(handle, info)

    def _force_kill_settled(self, handle: str, info: HerdrProcessInfo) -> None:
        """Force-kill while the PID is still ours, or refuse to claim success.

        Only two outcomes may drop the record: killed, or provably not ours
        any more. An unreadable token on a live PID is neither, and silently
        forgetting the agent there is precisely how a live worker ends up with
        nothing left to manage it.
        """
        live = creation_token(handle)
        if live is not None and live == info.creation_token:
            self._kill_pid(handle)
            self._processes.pop(handle, None)
            return
        if live is not None or not _pid_is_live(info.pid):
            # Readable and different, or simply gone: not our process.
            self._processes.pop(handle, None)
            return
        msg = (
            f"cannot prove ownership of PID {handle} for agent {info.name!r}: "
            "its creation token is unreadable while the process is alive. "
            "Refusing to signal it, and keeping the record so the agent is "
            "not silently abandoned."
        )
        raise HerdrOwnershipUnprovenError(msg)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Ask the agent to stop; report only what the PID actually proves."""
        info = self._processes.get(handle)
        if info is None:
            return not self._pid_alive(handle)
        if self._original_process_gone(info):
            return True
        if self._probe(info) is _HerdrProbe.OWNED:
            with contextlib.suppress(HerdrCommandError):
                self._run_herdr(
                    "pane", "send-keys", info.pane_id, "ctrl+c", expect="ok"
                )
        else:
            live = creation_token(handle)
            if live is not None and live == info.creation_token:
                self._interrupt_pid(handle)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._original_process_gone(info):
                return True
            time.sleep(0.1)
        return False

    @staticmethod
    def _original_process_gone(info: HerdrProcessInfo) -> bool:
        """Whether the process we spawned has definitively exited.

        Decided from process identity, never from Herdr's view of the pane: a
        pane that now hosts someone else says nothing about whether OUR
        process exited. An unreadable token on a live PID is indeterminate,
        never success -- callers read ``True`` as "no force kill needed" and
        resume the agent, which would let old and resumed workers overlap.
        """
        if not _pid_is_live(info.pid):
            return True
        live = creation_token(str(info.pid))
        return live is not None and live != info.creation_token

    def _interrupt_pid(self, handle: str) -> None:
        """Send SIGINT to a PID we have just re-proven is ours."""
        with contextlib.suppress(ProcessLookupError, ValueError, OSError):
            os.kill(int(handle), signal.SIGINT)

    def _pid_alive(self, handle: str) -> bool:
        """Whether the PID exists and is not a zombie.

        One definition, shared with ``_pid_is_live``: two liveness helpers
        that disagree about zombies is how a dead agent reads as alive.
        """
        try:
            pid = int(handle)
        except (TypeError, ValueError):
            return False
        return _pid_is_live(pid)

    def _wait_pid_exit(self, pid: int, timeout_s: float) -> bool:
        """Wait, bounded, for ``pid`` to leave the process table."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pid_alive(str(pid)):
                return True
            time.sleep(0.1)
        return not self._pid_alive(str(pid))

    def _kill_pid(self, handle: str) -> None:
        """Force-kill a PID, best-effort."""
        _force_kill_pid(handle)

    def _owned_info(self, handle: str, operation: str) -> HerdrProcessInfo | None:
        """Return the tracked record only when its pane is provably ours.

        A refusal is recorded rather than silent: "nothing happened" with no
        explanation is indistinguishable from a bug, and these refusals are
        exactly the interesting ones.
        """
        info = self._processes.get(handle)
        if info is None:
            return None
        state = self._probe(info)
        if state is _HerdrProbe.OWNED:
            return info
        with (
            contextlib.suppress(OSError),
            info.log_path.open("a", encoding="utf-8") as log_handle,
        ):
            log_handle.write(
                f"[herdr] refused {operation} on pane {info.pane_id}: {state.value}\n"
            )
        return None

    def _create_tab(
        self, request: SpawnRequest, env: dict[str, str], label: str | None
    ) -> dict[str, Any]:
        """Create the agent's tab in its repository's workspace.

        Routing by repository is the point: without ``--workspace`` Herdr puts
        every agent in whatever workspace happens to be active, so agents from
        three repos pile into one. An authoritative absence therefore CREATES
        the repo's workspace rather than falling through to an unqualified
        create -- the fall-through would silently land in someone else's.
        """
        outcome, workspace_id = (
            (_WorkspaceLookup.UNKNOWN, None)
            if label is None
            else self._resolve_workspace(label)
        )
        if outcome is _WorkspaceLookup.EMPTY:
            return self._create_in_new_workspace(request, env, label)
        try:
            return self._run_herdr(
                *self._tab_create_args(request, env, workspace_id),
                expect="tab_created",
            )
        except HerdrCommandError as exc:
            if exc.code not in _HERDR_NO_WORKSPACE_CODES:
                raise
        # Either the workspace we matched was closed underneath us, or (on the
        # unrouted path) this is a freshly started headless server with no
        # workspace at all. Both are Herdr's own word for "there is nothing to
        # put a tab in", which is the one thing that licenses creating one.
        return self._create_in_new_workspace(request, env, label)

    def _create_in_new_workspace(
        self, request: SpawnRequest, env: dict[str, str], label: str | None
    ) -> dict[str, Any]:
        """Create the repo's workspace, once, even under parallel spawns.

        ``workspace create`` takes the same options as ``tab create`` and
        yields the same ``root_pane``/``tab``, so the agent's tab IS the new
        workspace's root tab.
        """
        if label is None:
            # Unrouted: nothing to serialise on and nothing to re-check.
            return self._workspace_create(request, env, label=None)
        with file_lock(self._workspace_lock_path(label)):
            # Someone may have created it while we waited; and a listing we
            # cannot read is never grounds for creating a second one.
            outcome, workspace_id = self._resolve_workspace(label)
            if outcome is not _WorkspaceLookup.EMPTY:
                try:
                    return self._run_herdr(
                        *self._tab_create_args(request, env, workspace_id),
                        expect="tab_created",
                    )
                except HerdrCommandError as exc:
                    if exc.code not in _HERDR_NO_WORKSPACE_CODES:
                        raise
            return self._workspace_create(request, env, label)

    def _workspace_create(
        self, request: SpawnRequest, env: dict[str, str], label: str | None
    ) -> dict[str, Any]:
        """Create a workspace and adopt its root tab as the agent's tab."""
        args = self._tab_create_args(request, env, None)
        if label is not None:
            # The WORKSPACE is named after the repo; the tab keeps the agent.
            args[args.index("--label") + 1] = label
        created = self._run_herdr(*["workspace", *args[1:]], expect="workspace_created")
        # The workspace's own tab is labelled "1"; restore the agent label.
        tab = created.get("tab")
        if isinstance(tab, dict) and tab.get("tab_id"):
            with contextlib.suppress(HerdrCommandError):
                self._run_herdr(
                    "tab",
                    "rename",
                    str(tab["tab_id"]),
                    f"{request.name}@{request.team_name}",
                    expect="ok",
                )
        return created

    def _tab_create_args(
        self, request: SpawnRequest, env: dict[str, str], workspace_id: str | None
    ) -> list[str]:
        """Build ``tab create`` arguments.

        Env keys arrive already validated by ``process_base._spawn_with_command``
        against ``_SAFE_ENV_KEY``; values are passed as single argv tokens, so
        spaces, quotes and newlines need no escaping here.
        """
        args = ["tab", "create"]
        if workspace_id:
            args += ["--workspace", workspace_id]
        args += [
            "--cwd",
            request.cwd,
            "--label",
            f"{request.name}@{request.team_name}",
            "--no-focus",
        ]
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        return args

    def _pane_shell_pid(self, pane_id: str) -> int:
        """Return the pane's shell PID, which ``exec`` makes the agent itself."""
        result = self._run_herdr(
            "pane", "process-info", "--pane", pane_id, expect="pane_process_info"
        )
        info = result.get("process_info")
        if not isinstance(info, dict):
            msg = "process-info returned no process_info object"
            raise HerdrCommandError(["herdr", "pane", "process-info"], "malformed", msg)
        raw = info.get("shell_pid")
        if not isinstance(raw, int) or raw <= 0:
            msg = f"process-info returned a non-usable shell_pid: {raw!r}"
            raise HerdrCommandError(["herdr", "pane", "process-info"], "malformed", msg)
        return raw

    @staticmethod
    def _tab_id_of(created: dict[str, Any]) -> str:
        """Extract a usable tab id from a create response, however partial."""
        for key in ("tab", "root_pane"):
            section = created.get(key)
            if isinstance(section, dict):
                tab_id = section.get("tab_id")
                if isinstance(tab_id, str) and tab_id:
                    return tab_id
        return ""

    def _write_provenance(
        self,
        log_path: Path,
        tab_id: str,
        pane_id: str,
        handle: str,
        workspace_id: str = "",
        label: str | None = None,
    ) -> None:
        """Record where this agent lives, for anyone debugging it later.

        The workspace id is the one Herdr actually returned; the label is only
        what we ASKED for, which on the unrouted path is deliberately not the
        active workspace's own label -- so it is named as a request, not as a
        fact about that workspace.
        """
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[herdr] session={self._session or 'default'} tab={tab_id} "
                f"pane={pane_id} pid={handle} workspace={workspace_id or '?'} "
                f"requested_label={label or _HERDR_ACTIVE_WORKSPACE_SENTINEL}\n"
            )

    def _rollback_tab(self, tab_id: str, original: BaseException) -> None:
        """Close a half-spawned tab, attaching (never masking) a cleanup failure."""
        if not tab_id:
            return
        try:
            self._run_herdr("tab", "close", tab_id, expect="ok")
        except HerdrCommandError as cleanup_error:
            original.add_note(
                f"cleanup: could not close herdr tab {tab_id}: {cleanup_error}"
            )

    def _close_tab_quietly(self, tab_id: str) -> None:
        """Close a tab during cleanup without masking the original failure."""
        with contextlib.suppress(HerdrCommandError):
            self._run_herdr("tab", "close", tab_id, expect="ok")

    def _with_debug_file(self, cmd: list[str], log_path: Path) -> list[str]:
        """Mirror the tmux manager's claude-code debug-file wiring."""
        return [*cmd, "--debug-file", str(log_path)]

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

    @staticmethod
    def _parse_envelope(
        argv: list[str],
        completed: subprocess.CompletedProcess[str],
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        """Decode the JSON envelope: stdout on success, stderr on error.

        BOTH streams are parsed, and an error envelope wins wherever it
        appears. Returning the first parseable object would let a stdout
        success hide an error that Herdr reported on stderr.
        """
        parsed: list[dict[str, Any]] = []
        for stream in (completed.stdout, completed.stderr):
            text = (stream or "").strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed.append(value)
        for value in parsed:
            if isinstance(value.get("error"), dict):
                return value
        if parsed:
            return parsed[0]
        if completed.returncode == _HERDR_USAGE_EXIT_CODE:
            return {}
        if (
            allow_empty
            and completed.returncode == 0
            and not (completed.stdout or completed.stderr)
        ):
            # ``pane run`` is observed to succeed silently. Silence counts as
            # success only for such a command AND only when the exit code
            # agrees; a non-zero exit with no payload still fails below.
            return {}
        raise HerdrCommandError(
            argv, "malformed", (completed.stdout or completed.stderr or "").strip()
        )


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


def _select_linux_manager(launcher: str) -> type:
    """Map the configured launcher value to its manager class.

    Pure and total, so selection can be tested without reloading the
    module-level singleton. Note what is NOT here: ``HERDR_ENV``. Running
    inside a Herdr pane is caller context, not consent to make Herdr this
    server's launcher, and auto-detecting it would silently change behavior
    for existing deployments.
    """
    value = launcher.strip().lower()
    if value == _HERDR_LAUNCHER_VALUE:
        return HerdrProcessManager
    if value == _TMUX_LAUNCHER_VALUE:
        return TmuxProcessManager
    return LinuxTerminalProcessManager


if os.name == "nt":
    process_manager = WindowsProcessManager()
else:
    process_manager = _select_linux_manager(os.environ.get(_LINUX_LAUNCHER_ENV, ""))()

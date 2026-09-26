"""Bounded, cancellable Windows named-pipe writes for the Claude channel (F5).

A blocking ``open()``/``write()`` on a pipe has no deadline: a host that stops
reading wedges the writer forever, and ``Thread.join(timeout)`` only abandons
it. Every write here is overlapped I/O with a real operation deadline: on
expiry the write is cancelled with ``CancelIoEx`` and its completion awaited
for a bounded grace. A write whose completion is not observed in time is
parked with its storage (it may still complete late, so the outcome stays
uncertain); its path accepts no new write until it drains, and the process
holds at most ``MAX_PARKED`` such writes.

The server end is verified before any byte is written:
``GetNamedPipeServerProcessId`` must equal the expected host PID, which is a
stronger owner check than the POSIX socket-filename rule.

Windows-only entry points are resolved through ``getattr(ctypes, ...)`` as in
``procinfo`` and ``process_manager``, so the module imports (and type-checks)
on every platform.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
ERROR_FILE_NOT_FOUND = 2
ERROR_PATH_NOT_FOUND = 3
ERROR_PIPE_BUSY = 231
ERROR_SEM_TIMEOUT = 121
ERROR_IO_PENDING = 997
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
PIPE_PREFIX = "\\\\.\\pipe\\"
# Bounded wait for a requested cancellation to complete before parking.
CANCEL_GRACE_MS = 1000


@dataclass(frozen=True)
class PipeResult:
    """One bounded pipe write attempt's outcome; never carries payload text."""

    ok: bool
    reason: str = ""
    write_started: bool = False


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    ]


_KERNEL32: Any = None
_INVALID = ctypes.c_void_p(-1).value


def _k() -> Any:
    global _KERNEL32  # noqa: PLW0603 - one lazily bound DLL per process.
    if _KERNEL32 is not None:
        return _KERNEL32
    k = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    handle, dword, lpdword = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
    k.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        dword,
        dword,
        ctypes.c_void_p,
        dword,
        dword,
        handle,
    ]
    k.CreateFileW.restype = handle
    k.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, dword]
    k.WaitNamedPipeW.restype = ctypes.c_int
    k.GetNamedPipeServerProcessId.argtypes = [handle, lpdword]
    k.GetNamedPipeServerProcessId.restype = ctypes.c_int
    k.CreateEventW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    k.CreateEventW.restype = handle
    k.WriteFile.argtypes = [handle, ctypes.c_void_p, dword, lpdword, ctypes.c_void_p]
    k.WriteFile.restype = ctypes.c_int
    k.WaitForSingleObject.argtypes = [handle, dword]
    k.WaitForSingleObject.restype = dword
    k.CancelIoEx.argtypes = [handle, ctypes.c_void_p]
    k.CancelIoEx.restype = ctypes.c_int
    k.GetOverlappedResult.argtypes = [handle, ctypes.c_void_p, lpdword, ctypes.c_int]
    k.GetOverlappedResult.restype = ctypes.c_int
    k.CloseHandle.argtypes = [handle]
    k.CloseHandle.restype = ctypes.c_int
    _KERNEL32 = k
    return k


def _last_error() -> int:
    return int(getattr(ctypes, "get_last_error")())  # noqa: B009


def _remaining_ms(end: float) -> int:
    return max(0, int((end - time.monotonic()) * 1000))


def is_pipe_path(path: str) -> bool:
    """Return whether ``path`` names a local named pipe."""
    return path.casefold().startswith(PIPE_PREFIX.casefold())


def pipe_exists(path: str) -> bool:
    """Probe for a pipe without connecting to it (busy still means present)."""
    if not is_pipe_path(path):
        return False
    k = _k()
    if k.WaitNamedPipeW(path, 1):
        return True
    return _last_error() not in (ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND)


def _open(path: str, end: float) -> tuple[Any, str]:
    k = _k()
    while True:
        handle = k.CreateFileW(
            path, GENERIC_WRITE, 0, None, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, None
        )
        if handle not in (None, _INVALID):
            return handle, ""
        error = _last_error()
        if error in (ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND):
            return None, "socket_missing"
        if error != ERROR_PIPE_BUSY:
            return None, "open_failed"
        remaining = _remaining_ms(end)
        if remaining <= 0:
            return None, "pipe_busy"
        if not k.WaitNamedPipeW(path, remaining) and _last_error() in (
            ERROR_SEM_TIMEOUT,
        ):
            return None, "pipe_busy"


def _server_pid(handle: Any) -> int | None:
    pid = ctypes.c_uint32()
    if not _k().GetNamedPipeServerProcessId(handle, ctypes.byref(pid)):
        return None
    return int(pid.value)


@dataclass
class _Pending:
    """Kernel-referenced storage for one write; freed only after completion."""

    path: str
    handle: Any
    event: Any
    buffer: Any
    overlapped: _Overlapped
    written: ctypes.c_uint32


# Writes whose completion was not observed within the bounded drain. Their
# storage and handles must outlive the kernel's use of them, so they are
# parked instead of freed, and reaped once their event signals. While one is
# parked for a path, that path accepts no new write (a late completion is
# still possible), and the process holds at most ``MAX_PARKED`` of them.
_PARKED: list[_Pending] = []
_PARKED_LOCK = threading.Lock()
MAX_PARKED = 8


def _outcome(pending: _Pending, expected: int) -> str:
    """Classify a completed write; a completion that raced cancel still counts."""
    k = _k()
    if not k.GetOverlappedResult(
        pending.handle,
        ctypes.byref(pending.overlapped),
        ctypes.byref(pending.written),
        0,
    ):
        return "write_failed" if pending.written.value == 0 else "short_write"
    return "" if pending.written.value == expected else "short_write"


def _free(pending: _Pending) -> None:
    k = _k()
    k.CloseHandle(pending.event)
    k.CloseHandle(pending.handle)


def _park(pending: _Pending) -> None:
    with _PARKED_LOCK:
        _PARKED.append(pending)


def reap_parked() -> int:
    """Free parked writes whose completion has signalled; return how many remain.

    Safe to call periodically (the notifier does, independent of new posts).
    """
    k = _k()
    with _PARKED_LOCK:
        still: list[_Pending] = []
        for pending in _PARKED:
            try:
                done = k.WaitForSingleObject(pending.event, 0) == WAIT_OBJECT_0
            except Exception:  # Keep it parked: storage safety beats cleanup.
                done = False
            if done:
                _free(pending)
            else:
                still.append(pending)
        _PARKED[:] = still
        return len(still)


def _admission(path: str) -> str:
    """Refuse a write while this path has an undrained one, or at the cap."""
    with _PARKED_LOCK:
        if any(pending.path == path for pending in _PARKED):
            return "channel_busy"
        if len(_PARKED) >= MAX_PARKED:
            return "parked_cap"
    return ""


def _write(path: str, handle: Any, payload: bytes, end: float) -> tuple[str, bool]:
    """Return ``(reason, owns_handle)``; ``owns_handle`` False means parked.

    After ``WriteFile`` is issued, every exit — including an exception from a
    wait, cancel or completion query — either observes completion or parks the
    storage; it never lets the OVERLAPPED/buffer be freed under pending I/O.
    """
    k = _k()
    event = k.CreateEventW(None, 1, 0, None)
    if not event:
        return "event_failed", True
    pending = _Pending(
        path,
        handle,
        event,
        ctypes.create_string_buffer(payload, len(payload)),
        _Overlapped(),
        ctypes.c_uint32(),
    )
    pending.overlapped.hEvent = event
    try:
        started = k.WriteFile(
            handle, pending.buffer, len(payload), None, ctypes.byref(pending.overlapped)
        )
        if not started and _last_error() != ERROR_IO_PENDING:
            k.CloseHandle(event)
            # Refused synchronously: no byte was accepted.
            return "write_refused", True
    except Exception:
        # Unknown whether the write was issued: keep everything alive.
        _park(pending)
        return "cancel_pending", False
    try:
        waited = k.WaitForSingleObject(event, _remaining_ms(end))
        if waited != WAIT_OBJECT_0:
            # Deadline or wait failure: request cancellation, then allow a
            # short, bounded drain. Cancellation can lose to a completed write,
            # so the outcome is read from the completion, not assumed.
            k.CancelIoEx(handle, ctypes.byref(pending.overlapped))
            if k.WaitForSingleObject(event, CANCEL_GRACE_MS) != WAIT_OBJECT_0:
                _park(pending)
                return "cancel_pending", False
        reason = _outcome(pending, len(payload))
        if reason and waited != WAIT_OBJECT_0:
            reason = "timeout" if waited == WAIT_TIMEOUT else "wait_failed"
    except Exception:
        # Parking below is what keeps the storage safe; cancel is best-effort.
        with contextlib.suppress(Exception):
            k.CancelIoEx(handle, ctypes.byref(pending.overlapped))
        _park(pending)
        return "cancel_pending", False
    k.CloseHandle(event)
    return reason, True


def post(  # noqa: PLR0911 - one return per refusal reason.
    path: str, payload: bytes, deadline: float, *, expected_pid: int | None
) -> PipeResult:
    """Write ``payload`` then close, within ``deadline`` (+ a bounded drain).

    With ``expected_pid``, nothing is written unless the server process of the
    very handle used for the write is exactly that PID (fail closed on a query
    error). ``write_started`` is True once ``WriteFile`` was issued: from then
    on a failure is *uncertain* — the host may have accepted the bytes, even
    after this call returns if the write was parked — and must never be
    treated as proof of non-delivery. Errors are codes only, never message
    text, because the payload carries a credential.
    """
    if not is_pipe_path(path):
        return PipeResult(False, "socket_missing")
    if not math.isfinite(deadline) or deadline <= 0:
        return PipeResult(False, "invalid_deadline")
    end = time.monotonic() + deadline
    owns = True
    handle = None
    try:
        reap_parked()
        refused = _admission(path)
        if refused:
            return PipeResult(False, refused)
        handle, reason = _open(path, end)
        if handle is None:
            return PipeResult(False, reason)
        if expected_pid is not None and _server_pid(handle) != expected_pid:
            return PipeResult(False, "socket_not_owned")
        reason, owns = _write(path, handle, payload, end)
    except Exception as err:  # Never let transport errors escape.
        return PipeResult(False, type(err).__name__, write_started=not owns)
    finally:
        if owns and handle is not None:
            _k().CloseHandle(handle)
    return PipeResult(
        not reason,
        reason,
        write_started=reason not in ("event_failed", "write_refused"),
    )


def open_for_test(path: str) -> Any:
    """Open a client end without writing (test seam for busy/unblock cases)."""
    handle, _ = _open(path, time.monotonic() + 1.0)
    return handle


def close_for_test(handle: Any) -> None:
    """Close a handle returned by :func:`open_for_test`."""
    if handle is not None:
        _k().CloseHandle(handle)

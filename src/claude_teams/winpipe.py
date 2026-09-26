"""Bounded, cancellable Windows named-pipe writes for the Claude channel (F5).

A blocking ``open()``/``write()`` on a pipe has no deadline: a host that stops
reading wedges the writer forever, and ``Thread.join(timeout)`` only abandons
it. Every write here is overlapped I/O with a real operation deadline: on
expiry the write is cancelled with ``CancelIoEx`` and its completion awaited
before the buffer is released, so nothing is left running and no late write
can land after the call returns.

The server end is verified before any byte is written:
``GetNamedPipeServerProcessId`` must equal the expected host PID, which is a
stronger owner check than the POSIX socket-filename rule.

Windows-only entry points are resolved through ``getattr(ctypes, ...)`` as in
``procinfo`` and ``process_manager``, so the module imports (and type-checks)
on every platform.
"""

from __future__ import annotations

import ctypes
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


@dataclass(frozen=True)
class PipeResult:
    """One bounded pipe write attempt's outcome; never carries payload text."""

    ok: bool
    reason: str = ""


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


def _write(handle: Any, payload: bytes, end: float) -> str:
    k = _k()
    event = k.CreateEventW(None, 1, 0, None)
    if not event:
        return "event_failed"
    buffer = ctypes.create_string_buffer(payload, len(payload))
    overlapped = _Overlapped()
    overlapped.hEvent = event
    written = ctypes.c_uint32()
    try:
        started = k.WriteFile(
            handle, buffer, len(payload), None, ctypes.byref(overlapped)
        )
        if not started and _last_error() != ERROR_IO_PENDING:
            return "write_failed"
        waited = k.WaitForSingleObject(event, _remaining_ms(end))
        if waited == WAIT_OBJECT_0:
            if not k.GetOverlappedResult(
                handle, ctypes.byref(overlapped), ctypes.byref(written), 0
            ):
                return "write_failed"
            return "" if written.value == len(payload) else "short_write"
        # Deadline (or a wait failure): cancel, then wait for the cancellation
        # to complete so the kernel is done with ``buffer`` before it is freed.
        k.CancelIoEx(handle, ctypes.byref(overlapped))
        k.GetOverlappedResult(
            handle, ctypes.byref(overlapped), ctypes.byref(written), 1
        )
        return "timeout" if waited == WAIT_TIMEOUT else "wait_failed"
    finally:
        k.CloseHandle(event)


def post(
    path: str, payload: bytes, deadline: float, *, expected_pid: int | None
) -> PipeResult:
    """Write ``payload`` then close, within ``deadline`` seconds in total.

    With ``expected_pid``, nothing is written unless the pipe's server process
    is exactly that PID. Errors are reported by code only, never by message
    text, because the payload carries a credential.
    """
    if not is_pipe_path(path):
        return PipeResult(False, "socket_missing")
    end = time.monotonic() + deadline
    try:
        handle, reason = _open(path, end)
        if handle is None:
            return PipeResult(False, reason)
        try:
            if expected_pid is not None and _server_pid(handle) != expected_pid:
                return PipeResult(False, "socket_not_owned")
            reason = _write(handle, payload, end)
        finally:
            _k().CloseHandle(handle)
    except Exception as err:  # Never let transport errors escape.
        return PipeResult(False, type(err).__name__)
    return PipeResult(not reason, reason)


def open_for_test(path: str) -> Any:
    """Open a client end without writing (test seam for busy/unblock cases)."""
    handle, _ = _open(path, time.monotonic() + 1.0)
    return handle


def close_for_test(handle: Any) -> None:
    """Close a handle returned by :func:`open_for_test`."""
    if handle is not None:
        _k().CloseHandle(handle)

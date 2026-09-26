"""Windows named-pipe client: bounded, cancellable, owner-verified writes (F5)."""

import ctypes
import os
import sys
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows pipes")

PIPE_ACCESS_INBOUND = 0x00000001
PIPE_TYPE_BYTE = 0x00000000
ERROR_BROKEN_PIPE = 109
ERROR_PIPE_CONNECTED = 535
ERROR_NO_DATA = 232


def _kernel32():
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    kernel32.CreateNamedPipeW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateNamedPipeW.restype = ctypes.c_void_p
    kernel32.ConnectNamedPipe.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.ConnectNamedPipe.restype = ctypes.c_int
    kernel32.ReadFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = ctypes.c_int
    kernel32.DisconnectNamedPipe.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel32


class PipeServer:
    """One-instance inbound byte pipe served from a thread of this process."""

    def __init__(self, *, read: bool = True, in_buffer: int = 4096) -> None:
        self.name = rf"\\.\pipe\wat-test-{uuid.uuid4().hex}"
        self.k = _kernel32()
        self.handle = self.k.CreateNamedPipeW(
            self.name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE, 1, 0, in_buffer, 0, None
        )
        assert self.handle not in (None, ctypes.c_void_p(-1).value)
        self.read = read
        self.data = b""
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        ok = self.k.ConnectNamedPipe(self.handle, None)
        # A client that connected (or even wrote and closed) before this call
        # yields PIPE_CONNECTED / NO_DATA; its bytes are still buffered.
        if not ok and ctypes.get_last_error() not in (
            ERROR_PIPE_CONNECTED,
            ERROR_NO_DATA,
        ):
            return
        if not self.read:
            self.release.wait(10)
            return
        buffer = ctypes.create_string_buffer(4096)
        count = ctypes.c_uint32()
        while self.k.ReadFile(self.handle, buffer, 4096, ctypes.byref(count), None):
            self.data += buffer.raw[: count.value]

    def close(self) -> None:
        from claude_teams import winpipe

        self.release.set()
        if self.thread.is_alive():
            # Unblock a ConnectNamedPipe that never saw a client.
            handle = winpipe.open_for_test(self.name)
            if handle is not None:
                winpipe.close_for_test(handle)
        self.thread.join(5)
        self.k.DisconnectNamedPipe(self.handle)
        self.k.CloseHandle(self.handle)


@pytest.fixture
def server():
    servers: list[PipeServer] = []

    def make(**kwargs):
        created = PipeServer(**kwargs)
        servers.append(created)
        return created

    yield make
    for created in servers:
        created.close()


def test_post_writes_payload_and_closes(server):
    from claude_teams import winpipe

    pipe = server()
    result = winpipe.post(pipe.name, b"line-1\nline-2\n", 5.0, expected_pid=os.getpid())
    assert result.ok, result.reason
    pipe.thread.join(5)
    assert pipe.data == b"line-1\nline-2\n"


def test_missing_pipe_is_socket_missing():
    from claude_teams import winpipe

    result = winpipe.post(
        rf"\\.\pipe\wat-test-missing-{uuid.uuid4().hex}", b"x\n", 1.0, expected_pid=None
    )
    assert not result.ok
    assert result.reason == "socket_missing"


def test_server_pid_mismatch_writes_nothing(server):
    from claude_teams import winpipe

    pipe = server()
    result = winpipe.post(pipe.name, b"secret\n", 2.0, expected_pid=os.getpid() + 1)
    assert not result.ok
    assert result.reason == "socket_not_owned"
    pipe.thread.join(5)
    assert pipe.data == b""


def test_stalled_reader_is_cancelled_within_deadline(server):
    from claude_teams import winpipe

    pipe = server(read=False, in_buffer=64)
    started = time.monotonic()
    result = winpipe.post(pipe.name, b"x" * (4 * 1024 * 1024), 0.5, expected_pid=None)
    elapsed = time.monotonic() - started
    assert not result.ok
    assert result.reason == "timeout"
    assert elapsed < 3.0
    assert threading.active_count() < 50  # no poster thread left behind


def test_busy_pipe_waits_only_until_deadline(server):
    from claude_teams import winpipe

    pipe = server(read=False)
    first = winpipe.open_for_test(pipe.name)
    try:
        started = time.monotonic()
        result = winpipe.post(pipe.name, b"x\n", 0.5, expected_pid=None)
        assert not result.ok
        assert result.reason in {"timeout", "pipe_busy"}
        assert time.monotonic() - started < 3.0
    finally:
        winpipe.close_for_test(first)


def test_pipe_exists(server):
    from claude_teams import winpipe

    pipe = server()
    assert winpipe.pipe_exists(pipe.name)
    assert not winpipe.pipe_exists(rf"\\.\pipe\wat-test-missing-{uuid.uuid4().hex}")

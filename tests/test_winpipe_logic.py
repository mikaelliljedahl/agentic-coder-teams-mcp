"""Overlapped-write outcome rules with a scripted kernel32 (runs on every OS)."""

# ruff: noqa: N802 - fake methods mirror the Win32 export names.

import ctypes

import pytest

from claude_teams import winpipe

PIPE = "\\\\.\\pipe\\LOCAL\\cc-msg-test"


class FakeKernel:
    """Scripted kernel32: each wait returns the next scripted value."""

    def __init__(
        self,
        *,
        write_ok=0,
        write_error=winpipe.ERROR_IO_PENDING,
        waits=(winpipe.WAIT_OBJECT_0,),
        completed=None,
        server_pid=77,
    ):
        self.write_ok = write_ok
        self.write_error = write_error
        self.waits = list(waits)
        self.completed = completed
        self.server_pid = server_pid
        self.error = 0
        self.closed: list[int] = []
        self.calls: list[str] = []
        self.wait_raises = False

    def CreateFileW(self, *args):
        self.calls.append("CreateFileW")
        return 10

    def WaitNamedPipeW(self, *args):
        return 1

    def GetNamedPipeServerProcessId(self, handle, pid_ref):
        if self.server_pid is None:
            return 0
        pid_ref._obj.value = self.server_pid
        return 1

    def CreateEventW(self, *args):
        return 20

    def WriteFile(self, handle, buffer, size, written, overlapped):
        self.calls.append("WriteFile")
        self.size = size
        self.error = self.write_error
        return self.write_ok

    def WaitForSingleObject(self, handle, ms):
        self.calls.append(f"Wait({ms})")
        if self.wait_raises:
            raise OSError
        return self.waits.pop(0) if self.waits else winpipe.WAIT_OBJECT_0

    def CancelIoEx(self, handle, overlapped):
        self.calls.append("CancelIoEx")
        return 0

    def GetOverlappedResult(self, handle, overlapped, written_ref, wait):
        count = self.size if self.completed is None else self.completed
        written_ref._obj.value = count
        return 1 if count else 0

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


@pytest.fixture
def kernel(monkeypatch):
    def install(**kwargs):
        fake = FakeKernel(**kwargs)
        monkeypatch.setattr(winpipe, "_KERNEL32", fake)
        monkeypatch.setattr(winpipe, "_last_error", lambda: fake.error)
        monkeypatch.setattr(winpipe, "_PARKED", [])
        return fake

    return install


def test_immediate_completion_is_success(kernel):
    fake = kernel(write_ok=1, write_error=0)
    result = winpipe.post(PIPE, b"abc", 1.0, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (True, "", True)
    assert sorted(fake.closed) == [10, 20]


def test_cancel_that_loses_to_completion_counts_as_delivered(kernel):
    fake = kernel(waits=[winpipe.WAIT_TIMEOUT, winpipe.WAIT_OBJECT_0])
    result = winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    assert result.ok
    assert "CancelIoEx" in fake.calls


def test_cancelled_write_is_an_uncertain_timeout(kernel):
    kernel(waits=[winpipe.WAIT_TIMEOUT, winpipe.WAIT_OBJECT_0], completed=0)
    result = winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (False, "timeout", True)


def test_undrained_cancel_is_parked_not_freed(kernel):
    fake = kernel(waits=[winpipe.WAIT_TIMEOUT, winpipe.WAIT_TIMEOUT])
    result = winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (
        False,
        "cancel_pending",
        True,
    )
    assert fake.closed == []  # handle, event and buffer stay alive
    assert len(winpipe._PARKED) == 1
    fake.waits = [winpipe.WAIT_OBJECT_0]  # the parked write finally completed
    winpipe.reap_parked()
    assert winpipe._PARKED == []
    assert sorted(fake.closed) == [10, 20]


def test_synchronous_refusal_means_nothing_was_written(kernel):
    kernel(write_ok=0, write_error=232)
    result = winpipe.post(PIPE, b"abc", 1.0, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (
        False,
        "write_refused",
        False,
    )


@pytest.mark.parametrize("server_pid", [None, 78])
def test_owner_query_failure_or_mismatch_writes_nothing(kernel, server_pid):
    fake = kernel(server_pid=server_pid)
    result = winpipe.post(PIPE, b"abc", 1.0, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (
        False,
        "socket_not_owned",
        False,
    )
    assert "WriteFile" not in fake.calls
    assert fake.closed == [10]


def test_short_write_is_reported(kernel):
    kernel(completed=2)
    result = winpipe.post(PIPE, b"abc", 1.0, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (
        False,
        "short_write",
        True,
    )


def test_handle_width_is_pointer_sized():
    assert ctypes.sizeof(winpipe._Overlapped) in (20, 32)


def test_parked_path_refuses_a_second_write(kernel):
    fake = kernel(waits=[winpipe.WAIT_TIMEOUT, winpipe.WAIT_TIMEOUT])
    first = winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    assert first.reason == "cancel_pending"
    fake.waits = [winpipe.WAIT_TIMEOUT]  # the parked write is still pending
    second = winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    assert (second.ok, second.reason, second.write_started) == (
        False,
        "channel_busy",
        False,
    )
    assert fake.calls.count("WriteFile") == 1
    assert len(winpipe._PARKED) == 1


def test_global_parking_cap(kernel, monkeypatch):
    monkeypatch.setattr(winpipe, "MAX_PARKED", 2)
    fake = kernel()
    fake.waits = [winpipe.WAIT_TIMEOUT] * 20
    for index in range(2):
        winpipe.post(f"{PIPE}-{index}", b"abc", 0.1, expected_pid=77)
    assert len(winpipe._PARKED) == 2
    result = winpipe.post(f"{PIPE}-other", b"abc", 0.1, expected_pid=77)
    assert (result.ok, result.reason) == (False, "parked_cap")
    assert fake.calls.count("WriteFile") == 2


def test_exception_after_write_issued_parks_storage(kernel):
    fake = kernel()
    fake.wait_raises = True
    result = winpipe.post(PIPE, b"abc", 1.0, expected_pid=77)
    assert not result.ok
    assert result.write_started
    assert fake.closed == []
    assert len(winpipe._PARKED) == 1
    assert "CancelIoEx" in fake.calls


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_deadline_writes_nothing(kernel, deadline):
    fake = kernel()
    result = winpipe.post(PIPE, b"abc", deadline, expected_pid=77)
    assert (result.ok, result.reason, result.write_started) == (
        False,
        "invalid_deadline",
        False,
    )
    assert fake.calls == []


def test_reaper_runs_without_a_new_post(kernel):
    fake = kernel(waits=[winpipe.WAIT_TIMEOUT, winpipe.WAIT_TIMEOUT])
    winpipe.post(PIPE, b"abc", 0.1, expected_pid=77)
    fake.waits = [winpipe.WAIT_OBJECT_0]
    assert winpipe.reap_parked() == 0
    assert sorted(fake.closed) == [10, 20]

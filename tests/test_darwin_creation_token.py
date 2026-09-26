"""Portable and native coverage for Darwin process creation tokens."""

import ctypes
import datetime
import errno
import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from claude_teams.backends import process_manager as pm


def _prefix(sec: int = 123, usec: int = 42) -> bytes:
    return struct.pack("<qi", sec, usec) + b"\xa5\x5a\xff\xff"


def test_parser_ignores_nonzero_padding() -> None:
    assert pm._parse_darwin_kinfo_start_time(_prefix(123, 42)) == (123, 42)


@pytest.mark.parametrize(
    "raw",
    [b"", b"x" * 11, _prefix(0), _prefix(-1), _prefix(1, -1), _prefix(1, 1_000_000)],
)
def test_parser_rejects_invalid_prefix(raw: bytes) -> None:
    assert pm._parse_darwin_kinfo_start_time(raw) is None


class FakeSysctl:
    """Record the ABI calls and supply a controlled second reply."""

    def __init__(
        self,
        reply: bytes = _prefix(),
        *,
        first_error=False,
        second_error=False,
        second_errno: int | None = None,
        capacity: int | None = None,
        returned: int | None = None,
    ) -> None:
        self.reply = reply
        self.first_error = first_error
        self.second_error = second_error
        self.second_errno = second_errno
        self.capacity = len(reply) if capacity is None else capacity
        self.returned = len(reply) if returned is None else returned
        self.calls: list[tuple[list[int], int, bool, object, int]] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, mib, count, oldp, oldlenp, newp, newlen) -> int:
        self.calls.append(
            ([mib[i] for i in range(count)], count, bool(oldp), newp, newlen)
        )
        size = ctypes.cast(oldlenp, ctypes.POINTER(ctypes.c_size_t))
        if not oldp:
            size[0] = self.capacity
            return -1 if self.first_error else 0
        if self.second_error:
            if self.second_errno is not None:
                ctypes.set_errno(self.second_errno)
            return -1
        ctypes.memmove(oldp, self.reply, min(len(self.reply), self.capacity))
        size[0] = self.returned
        return 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"first_error": True}, None),
        ({"capacity": 0}, None),
        ({"second_error": True}, None),
        ({"second_error": True, "second_errno": errno.ENOMEM}, None),
        ({"returned": 0}, None),
        ({"returned": 11}, None),
        ({"returned": 17}, None),
        ({"reply": _prefix(0)}, None),
        ({"reply": _prefix(123, 42)}, "123.000042"),
    ],
)
def test_ctypes_reader_contract(
    monkeypatch, kwargs: dict, expected: str | None
) -> None:
    fake = FakeSysctl(**kwargs)
    monkeypatch.setattr(pm.ctypes, "CDLL", lambda _: SimpleNamespace(sysctl=fake))

    assert pm._read_darwin_creation_token(777) == expected
    assert fake.calls[0] == ([1, 14, 1, 777], 4, False, None, 0)
    if not fake.first_error and fake.capacity:
        assert fake.calls[1] == ([1, 14, 1, 777], 4, True, None, 0)
    assert fake.argtypes == [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    assert fake.restype is ctypes.c_int


def test_ctypes_reader_load_failure(monkeypatch) -> None:
    def fail(_):
        raise OSError

    monkeypatch.setattr(pm.ctypes, "CDLL", fail)
    assert pm._read_darwin_creation_token(777) is None


@pytest.mark.parametrize(
    ("windows", "darwin", "reader"),
    [(True, True, "windows"), (False, True, "darwin"), (False, False, "linux")],
)
def test_dispatch_precedence(
    monkeypatch, windows: bool, darwin: bool, reader: str
) -> None:
    monkeypatch.setattr(pm, "_creation_token_is_windows", lambda: windows)
    monkeypatch.setattr(pm, "_creation_token_is_darwin", lambda: darwin)
    for name in ("windows", "darwin", "linux"):
        monkeypatch.setattr(pm, f"_read_{name}_creation_token", lambda pid, n=name: n)
    assert pm.creation_token("777") == reader


@pytest.mark.skipif(sys.platform != "darwin", reason="native Darwin only")
def test_native_live_children_and_independent_start_time() -> None:
    children = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        for _ in range(2)
    ]
    try:
        tokens = [pm.creation_token(str(child.pid)) for child in children]
        assert all(tokens)
        assert tokens[0] is not None
        assert tokens[0] != tokens[1]
        assert pm.creation_token(str(children[0].pid)) == tokens[0]
        started = subprocess.check_output(  # noqa: S603 - fixed executable/arguments
            ["/bin/ps", "-o", "lstart=", "-p", str(children[0].pid)],
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        ).strip()
        independent = (
            datetime.datetime.strptime(started, "%a %b %d %H:%M:%S %Y")
            .replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
            .timestamp()
        )
        assert abs(float(tokens[0].split(".")[0]) - independent) <= 1
    finally:
        for child in children:
            child.terminate()
            child.wait()


@pytest.mark.skipif(sys.platform != "darwin", reason="native Darwin only")
def test_native_pid_one_and_missing_pid() -> None:
    assert pm.creation_token("1") is not None
    assert pm.creation_token("99999999") is None

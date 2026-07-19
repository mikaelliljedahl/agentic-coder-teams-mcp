"""The one cross-process advisory file lock this project has.

Extracted from ``server_simple`` so the delivery status store can take the
*same* lock model as the agent registry rather than a second, subtly different
one. The distinction matters: several per-agent MCP servers share one session
directory, so a ``threading.Lock`` (which is all the inbox cursor path uses,
deliberately, because its owner is the single writer) would serialize threads
inside one server and let two servers interleave.

Windows and POSIX differ in kind, not just spelling. ``msvcrt.locking`` has no
blocking-with-timeout mode that is safe to interrupt, so the Windows path polls
a non-blocking acquire until a deadline and then raises; ``fcntl.flock`` blocks
in the kernel. Both are advisory: they coordinate this project's writers with
each other, nothing more.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

#: How long the Windows path polls before giving up. POSIX blocks instead.
LOCK_TIMEOUT_SECONDS = 30.0
LOCK_RETRY_SECONDS = 0.05
#: Byte range locked. One byte is enough for an advisory whole-file lock, and
#: locking a range beyond EOF is what ``msvcrt`` requires anyway.
LOCK_SIZE = 1


class FileLockTimeoutError(TimeoutError):
    """Raised when a lock could not be acquired within the timeout."""


def lock_handle(handle: Any, *, timeout_s: float = LOCK_TIMEOUT_SECONDS) -> None:
    """Take an exclusive advisory lock on an open binary file handle."""
    if os.name == "nt":
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, LOCK_SIZE)
            except OSError as err:
                if time.monotonic() >= deadline:
                    raise FileLockTimeoutError from err
                time.sleep(LOCK_RETRY_SECONDS)
            else:
                return
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def unlock_handle(handle: Any) -> None:
    """Release a lock taken by :func:`lock_handle`."""
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, LOCK_SIZE)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path, *, timeout_s: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold an exclusive lock on the sidecar lock file at ``path``.

    The lock file is never the data file: it is opened ``a+b`` and never
    truncated, so a writer that replaces the data file atomically cannot pull
    the lock out from under a concurrent holder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        lock_handle(handle, timeout_s=timeout_s)
        try:
            yield
        finally:
            unlock_handle(handle)


__all__ = [
    "LOCK_RETRY_SECONDS",
    "LOCK_SIZE",
    "LOCK_TIMEOUT_SECONDS",
    "FileLockTimeoutError",
    "file_lock",
    "lock_handle",
    "unlock_handle",
]

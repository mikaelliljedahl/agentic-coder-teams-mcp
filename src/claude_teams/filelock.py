"""Cross-platform file-based locking helpers for local team state."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Context manager providing an exclusive file-based lock.

    Windows uses ``msvcrt.locking`` and POSIX uses ``fcntl.flock``. The fork is
    Windows-first, but keeping the fallback avoids import-time failures in
    tooling that inspects the package from non-Windows hosts.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+", encoding="utf-8") as lock_file:
        _lock(lock_file.fileno())
        try:
            yield
        finally:
            _unlock(lock_file.fileno())


def _lock(fd: int) -> None:
    if os.name == "nt":
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                time.sleep(0.05)
            else:
                return
    else:
        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)

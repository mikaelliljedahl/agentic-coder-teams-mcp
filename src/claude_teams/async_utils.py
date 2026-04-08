"""Async helpers for running blocking local operations safely."""

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_blocking(fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a blocking function in a worker thread.

    Args:
        fn (Callable[P, T]): Blocking callable to execute.
        *args (P.args): Positional arguments forwarded to ``fn``.
        **kwargs (P.kwargs): Keyword arguments forwarded to ``fn``.

    Returns:
        T: Result returned by ``fn``.

    """
    return await asyncio.to_thread(fn, *args, **kwargs)

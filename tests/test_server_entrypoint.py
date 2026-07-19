"""Tests for the ``claude_teams.server`` entry-point shim.

``server.py`` merely re-exports ``main``/``mcp`` from ``server_simple`` and
runs ``main()`` under ``__main__``. These tests cover the module body without
starting the blocking MCP server.
"""

import runpy
import warnings

from claude_teams import server, server_simple


def _run_as_main(module: str) -> None:
    """Execute ``module`` under ``__main__`` via runpy.

    The module is already imported, so runpy emits a benign ``RuntimeWarning``
    about it being in ``sys.modules``; that is expected for this re-execution
    pattern (deleting the cached module would risk a second, divergent copy),
    so it is suppressed rather than surfaced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module(module, run_name="__main__")


def test_server_reexports_are_identical():
    # The shim must forward the *same* objects, not shadow copies.
    assert server.main is server_simple.main
    assert server.mcp is server_simple.mcp
    assert set(server.__all__) == {"main", "mcp"}


def test_server_main_block_invokes_main(monkeypatch):
    calls = []
    # runpy re-imports the module fresh and binds ``main`` from server_simple,
    # so patch the source symbol rather than the already-imported ``server.main``.
    monkeypatch.setattr(server_simple, "main", lambda: calls.append(True))

    _run_as_main("claude_teams.server")

    assert calls == [True]


def test_server_simple_main_runs_mcp(monkeypatch):
    calls = []
    # Patch at the class so the (singleton) mcp instance's blocking run is a
    # no-op; main() must forward to it exactly once.
    monkeypatch.setattr(
        type(server_simple.mcp), "run", lambda self, *a, **k: calls.append(True)
    )

    server_simple.main()

    assert calls == [True]


def test_server_simple_main_block(monkeypatch):
    calls = []
    # The fresh runpy execution builds a new FastMCP of the same class, so the
    # class-level patch also intercepts its ``__main__`` -> main() -> mcp.run().
    monkeypatch.setattr(
        type(server_simple.mcp), "run", lambda self, *a, **k: calls.append(True)
    )

    _run_as_main("claude_teams.server_simple")

    assert calls == [True]

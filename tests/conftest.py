import os

import pytest

# Ambient variables a developer shell can inherit (Claude Desktop's MCP config
# exports WIN_AGENT_TEAMS_NATIVE_WAKE/NATIVE_DOWNSTREAM; spawned agents carry
# identity and session-dir variables). CI sets none of them, so strip them to
# make local runs match CI.
_AMBIENT_AGENT_ENV_EXACT = frozenset(
    {
        "AGENT_NAME",
        "AGENT_SESSION_ID",
        "AGENT_PARENT_NAME",
        "CODEX_HOME",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_MESSAGING_TOKEN",
    }
)
_AMBIENT_AGENT_ENV_PREFIXES = ("WIN_AGENT_TEAMS_", "CLAUDE_TEAMS_")


def is_ambient_agent_env(key: str) -> bool:
    """Return whether ``key`` is agent/win-agent-teams state a test must not inherit."""
    return key in _AMBIENT_AGENT_ENV_EXACT or key.startswith(
        _AMBIENT_AGENT_ENV_PREFIXES
    )


# Scrub at conftest load, before any ``claude_teams`` import: several modules
# read these at import time (``server_simple`` identity and EXTERNAL_ONLY tool
# registration, ``process_manager`` launcher selection), and module/session
# scoped fixtures run before the function-scoped autouse fixture below.
for _key in [k for k in os.environ if is_ambient_agent_env(k)]:
    del os.environ[_key]

pytest_plugins = ["tests.test_backends._base_support"]


@pytest.fixture(autouse=True)
def _clear_inherited_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the suite from an inherited spawned-agent environment.

    When this suite runs inside a managed agent (e.g. a review agent spawned
    by win-agent-teams), ``AGENT_SESSION_ID``/``AGENT_NAME``/
    ``AGENT_PARENT_NAME`` are already set in the process environment and
    ``server_simple._AGENT_SESSION_ID`` is captured from them at import time.
    Tests that only reset ``server_simple._session_id`` would then still
    recover the real session id via ``_recover_session_id``, corrupting
    lead-mode session creation/recovery tests. Clear both the env vars and
    the captured module globals before every test. The load-time scrub above
    handles import-time reads; this per-test pass catches anything a test or
    fixture wrote straight into ``os.environ`` without monkeypatch.
    """
    for key in [k for k in os.environ if is_ambient_agent_env(k)]:
        monkeypatch.delenv(key)
    from claude_teams import server_simple

    monkeypatch.setattr(server_simple, "_AGENT_NAME", "")
    monkeypatch.setattr(server_simple, "_AGENT_SESSION_ID", "")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "")
    monkeypatch.setattr(server_simple, "IDENTITY", server_simple.ROOT_LEAD_NAME)
    monkeypatch.setattr(server_simple, "_IDENTITY_UNRESOLVED", False)


@pytest.fixture
def posix_launcher_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the host as POSIX for nested Linux-launcher env propagation.

    ``nested_linux_launcher_env`` is deliberately empty on Windows, so tests of
    the Linux propagation path must pin the host predicate instead of relying
    on the CI runner's OS. Patching ``os.name`` itself would also flip
    unrelated call-time checks (e.g. ``filelock``) and break on Windows.
    """
    from claude_teams.backends import process_manager

    monkeypatch.setattr(process_manager, "_launcher_host_is_windows", lambda: False)

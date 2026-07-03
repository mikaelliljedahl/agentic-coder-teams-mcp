import pytest

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
    the captured module global before every test.
    """
    monkeypatch.delenv("AGENT_SESSION_ID", raising=False)
    monkeypatch.delenv("AGENT_NAME", raising=False)
    monkeypatch.delenv("AGENT_PARENT_NAME", raising=False)
    from claude_teams import server_simple

    monkeypatch.setattr(server_simple, "_AGENT_SESSION_ID", "")

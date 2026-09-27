"""Guard: the suite starts every test from a clean win-agent-teams env.

Claude Desktop's MCP config exports ``WIN_AGENT_TEAMS_NATIVE_WAKE`` and
``WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM``, and spawned agents carry identity
variables. Those leak into a developer's shell; ``conftest.py`` strips them
at load time (before any ``claude_teams`` import) and again per test, so
local runs match CI.
"""

import os

import pytest

from tests.conftest import is_ambient_agent_env

_AMBIENT_EXACT = (
    "AGENT_NAME",
    "AGENT_SESSION_ID",
    "AGENT_PARENT_NAME",
    "CODEX_HOME",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
)
_AMBIENT_PREFIXES = ("WIN_AGENT_TEAMS_", "CLAUDE_TEAMS_")


def test_ambient_agent_env_is_cleared() -> None:
    """No ambient agent/native-wake variable is visible inside a test."""
    leaked = sorted(
        key
        for key in os.environ
        if key in _AMBIENT_EXACT or key.startswith(_AMBIENT_PREFIXES)
    )
    assert leaked == []


@pytest.mark.parametrize(
    "key",
    [
        "WIN_AGENT_TEAMS_NATIVE_WAKE",
        "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM",
        "WIN_AGENT_TEAMS_SESSION_DIR",
        "WIN_AGENT_TEAMS_EXTERNAL_ONLY",
        "WIN_AGENT_TEAMS_LINUX_LAUNCHER",
        "CLAUDE_TEAMS_PERMISSION_MODE",
        *_AMBIENT_EXACT,
    ],
)
def test_predicate_matches_known_leaky_variables(key: str) -> None:
    """The conftest scrub covers every variable known to flip test outcomes."""
    assert is_ambient_agent_env(key)


@pytest.mark.parametrize(
    "key", ["HOME", "PATH", "TMUX", "DISPLAY", "XDG_RUNTIME_DIR", "COMSPEC"]
)
def test_predicate_leaves_host_state_alone(key: str) -> None:
    """Generic host/desktop variables are deliberately not scrubbed."""
    assert not is_ambient_agent_env(key)

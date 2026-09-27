"""Spawn and resume must scrub inherited Claude host channels when opted in."""

import pytest

from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_base import BaseBackend, process_manager


@pytest.mark.parametrize("backend", ["claude-code", "codex", "pi"])
@pytest.mark.parametrize("operation", ["spawn", "resume"])
def test_spawn_resume_scrub(backend, operation, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    for key in (
        "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM",
        "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE",
        "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX",
    ):
        monkeypatch.delenv(key, raising=False)
    from claude_teams.backends.claude_code import ClaudeCodeBackend
    from claude_teams.backends.codex import CodexBackend
    from claude_teams.backends.pi import PiBackend

    instance = {
        "claude-code": ClaudeCodeBackend,
        "codex": CodexBackend,
        "pi": PiBackend,
    }[backend]()
    monkeypatch.setattr(instance, "build_env", lambda _: {"EXAMPLE": "value"})
    monkeypatch.setattr(instance, "build_command", lambda _: ["fake"])
    monkeypatch.setattr(instance, "build_resume_command", lambda *a: ["fake"])
    observed = []
    monkeypatch.setattr(
        process_manager,
        "spawn_process",
        lambda request, argv, env_vars, *a, **k: observed.append(env_vars),
    )
    request = SpawnRequest(
        agent_id="id",
        name="agent",
        team_name="team",
        prompt="task",
        model="",
        agent_type="",
        color="",
        cwd=".",
        lead_session_id="team-lead",
    )
    if operation == "spawn":
        BaseBackend.spawn(instance, request)
    else:
        BaseBackend.resume(instance, request, "thread")
    assert observed == [
        {
            "EXAMPLE": "value",
            # The effective master flag propagates (plan §2.7).
            "WIN_AGENT_TEAMS_NATIVE_WAKE": "1",
            "CLAUDE_CODE_MESSAGING_SOCKET": "",
            "CLAUDE_CODE_MESSAGING_TOKEN": "",
        }
    ]

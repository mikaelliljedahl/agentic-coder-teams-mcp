"""Part A: launch metadata and factual first-marker diagnosis."""

import asyncio
from pathlib import Path

import pytest

from claude_teams import server_simple
from claude_teams.agent_output import BINDING_LEGACY, BindingResult
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.pi import PiBackend


def _request(tmp_path: Path, extra: dict[str, str]) -> SpawnRequest:
    return SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="task",
        model="model",
        agent_type="worker",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="team-lead",
        extra=extra,
    )


def test_state_hook_args_match_actual_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude = ClaudeCodeBackend()
    codex = CodexBackend()
    pi = PiBackend()
    assert claude.state_hook_args(
        _request(tmp_path, {"hooks_settings_path": "hooks.json"})
    ) == ["--settings", "hooks.json"]
    assert claude.state_hook_args(_request(tmp_path, {})) == []
    monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")
    assert (
        claude.state_hook_args(
            _request(tmp_path, {"hooks_settings_path": "hooks.json"})
        )
        == []
    )
    assert (
        codex.state_hook_args(
            _request(tmp_path, {"hook_overrides": '["-c", "hooks={}"]'})
        )
        == []
    )
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS")
    assert codex.state_hook_args(
        _request(tmp_path, {"hook_overrides": '["-c", "hooks={}"]'})
    ) == ["-c", "hooks={}", "--dangerously-bypass-hook-trust"]
    monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "0")
    assert (
        codex.state_hook_args(
            _request(tmp_path, {"hook_overrides": '["-c", "hooks={}"]'})
        )
        == []
    )
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX")
    assert codex.state_hook_args(_request(tmp_path, {})) == []
    assert pi.state_hook_args(
        _request(
            tmp_path,
            {
                "pi_state_extension_path": "state.ts",
                "pi_wake_extension_path": "wake.ts",
            },
        )
    ) == ["-e", "state.ts"]
    assert (
        pi.state_hook_args(_request(tmp_path, {"pi_wake_extension_path": "wake.ts"}))
        == []
    )


@pytest.mark.parametrize(
    ("marker", "now", "alive", "wired", "expected"),
    [
        (None, 1045.0, True, True, True),
        (None, 1044.999, True, True, False),
        ({"ts": 1000.0}, 1060.0, True, True, False),
        ({"ts": 999.999}, 1060.0, True, True, True),
        ({"state": "waiting"}, 1060.0, True, True, True),
        (None, 1060.0, False, True, False),
        (None, 1060.0, True, False, False),
    ],
)
def test_diagnosis_predicate(
    marker, now, alive, wired, expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_FIRST_MARKER_SECONDS", raising=False)
    agent = {
        "backend": "codex",
        "cwd": "work",
        "launch_started_at": 1000.0,
        "launch_interactive": True,
        "hooks_wired": wired,
    }
    result = server_simple._startup_diagnosis(agent, marker, alive, now)
    assert result["no_marker_since_launch"] is expected
    assert (result["startup_hint"] is not None) is expected


def test_legacy_external_and_clock_step(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        server_simple._startup_diagnosis({"backend": "codex"}, None, True, 1060.0)[
            "no_marker_since_launch"
        ]
        is None
    )
    assert (
        server_simple._startup_diagnosis(
            {"backend": "external", "launch_started_at": 1000.0, "hooks_wired": True},
            None,
            True,
            1060.0,
        )["no_marker_since_launch"]
        is None
    )
    agent = {
        "backend": "codex",
        "cwd": "work",
        "launch_started_at": 1000.0,
        "launch_interactive": True,
        "hooks_wired": True,
    }
    # A backward clock step can make a genuine child marker look pre-launch.
    assert (
        server_simple._startup_diagnosis(agent, {"ts": 999.0}, True, 1060.0)[
            "no_marker_since_launch"
        ]
        is True
    )
    agent["hooks_wired"] = False
    monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "1")
    assert (
        server_simple._startup_diagnosis(agent, None, True, 1060.0)[
            "no_marker_since_launch"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("backend", "phrase"),
    [
        ("codex", "folder-trust"),
        ("claude-code", "workspace-trust"),
        ("pi", "project-trust"),
    ],
)
def test_interactive_hints(backend: str, phrase: str) -> None:
    agent = {
        "backend": backend,
        "cwd": "work",
        "launch_started_at": 1000.0,
        "launch_interactive": True,
        "hooks_wired": True,
    }
    hint = server_simple._startup_diagnosis(agent, None, True, 1060.0)["startup_hint"]
    assert isinstance(hint, str)
    assert phrase in hint
    assert "Likely causes" in hint


def test_headless_hint() -> None:
    agent = {
        "backend": "codex",
        "launch_started_at": 1000.0,
        "launch_interactive": False,
        "hooks_wired": True,
    }
    hint = server_simple._startup_diagnosis(agent, None, True, 1060.0)["startup_hint"]
    assert isinstance(hint, str)
    assert "CLI startup problem or hook failure" in hint


def test_all_five_surfaces_preserve_stale_waiting_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "startup-session"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
    monkeypatch.setattr(server_simple, "_session_id", session_id)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})
    (tmp_path / session_id).mkdir()
    agent = {
        "name": "worker",
        "pid": 4242,
        "backend": "codex",
        "session_id": session_id,
        "status": "running",
        "spawned_at": 900.0,
        "cwd": "work",
        "launch_started_at": 1000.0,
        "launch_interactive": True,
        "hooks_wired": True,
    }
    server_simple._save_agents(session_id, [agent])
    server_simple._state_marker_file(session_id, "worker").write_text(
        '{"state":"waiting","event":"Stop","ts":999.0}', encoding="utf-8"
    )
    monkeypatch.setattr(
        server_simple.process_manager, "health_check", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1060.0)
    calls = []

    def binding(*args, **kwargs):
        calls.append(1)
        return BindingResult(BINDING_LEGACY, None)

    monkeypatch.setattr(server_simple, "_resolve_agent_binding", binding)
    views = [
        asyncio.run(server_simple.check_agent("worker")),
        asyncio.run(server_simple.check_agent("worker", full=True)),
        asyncio.run(server_simple.agent_status())[0],
        asyncio.run(server_simple.list_agents())[0],
        asyncio.run(server_simple.list_agents(full=True))[0],
    ]
    assert len(calls) == 3  # existing check/full-list binding only
    assert all(v["no_marker_since_launch"] is True for v in views)
    assert len({v["startup_hint"] for v in views}) == 1
    assert all(v["state"] == "waiting" for v in views[:4])
    assert views[0]["stalled"] is False
    assert views[2]["stalled"] is False
    assert views[0]["heartbeat_age_s"] == 61.0
    assert views[2]["heartbeat_age_s"] == 61.0


def test_tool_docs_define_predicate() -> None:
    for tool in (
        server_simple.check_agent,
        server_simple.list_agents,
        server_simple.agent_status,
    ):
        doc = tool.__doc__ or ""
        assert "no_marker_since_launch" in doc
        assert "folder-trust" in doc
        assert "heuristic" in doc


def test_list_agents_doc_only_names_returned_state_field() -> None:
    doc = server_simple.list_agents.__doc__ or ""
    assert "changes ``state`` or ``stalled``" not in doc

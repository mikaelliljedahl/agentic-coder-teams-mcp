"""R1/R2/R3 — lead session recovery after restart.

Covers the cwd+identity fallback (single-lead auto-adopt), the multi-lead
guard (>=2 bound sessions disables auto-adopt), the recovery nudge on
dict-returning tools, and the resume_session / session_info tools.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple as ss


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    base = tmp_path / "sessions"
    base.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    monkeypatch.setattr(ss, "_TEAMS_BASE", tmp_path / "teams")
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.setattr(ss, "_pending_recovery", {})
    monkeypatch.setattr(ss, "_inbox_locks", {})
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    monkeypatch.chdir(work)
    return SimpleNamespace(base=base, work=work)


def _agent(name: str = "worker") -> dict:
    return {"name": name, "pid": 4242, "backend": "codex", "status": "running"}


def _write_session(base: Path, sid: str, agents: list[dict]) -> Path:
    d = base / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents.json").write_text(json.dumps(agents), encoding="utf-8")
    return d


def _write_binding(base: Path, fname: str, sid: str, *, parent: str = "old") -> Path:
    bdir = base / "bindings"
    bdir.mkdir(parents=True, exist_ok=True)
    cwd = str(Path.cwd().resolve())
    meta = {
        "session_id": sid,
        "identity": ss.IDENTITY,
        "cwd": cwd,
        "binding_key": f"identity={ss.IDENTITY}\nparent={parent}\ncwd={cwd}",
        "lead_token": f"tok-{sid}",
        "updated_at": "2026-07-01T00:00:00+00:00",
    }
    path = bdir / f"{fname}.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


class TestExactAndEnvPaths:
    def test_exact_binding_key_still_recovers(
        self, workspace: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid = "11111111-1111-1111-1111-111111111111"
        _write_session(workspace.base, sid, [_agent()])
        monkeypatch.setattr(ss, "_session_id", sid)
        ss._persist_session_binding(sid)  # writes the exact-key binding
        monkeypatch.setattr(ss, "_session_id", "")

        assert ss._recover_session_id() == sid

    def test_env_session_id_wins(
        self, workspace: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_session(workspace.base, "cand", [_agent()])
        _write_binding(workspace.base, "b", "cand")
        monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "env-session")

        assert ss._recover_session_id() == "env-session"


class TestCwdFallback:
    def test_adopts_newest_after_parent_change(
        self, workspace: SimpleNamespace
    ) -> None:
        sid = "22222222-2222-2222-2222-222222222222"
        _write_session(workspace.base, sid, [_agent()])
        _write_binding(workspace.base, "stale", sid, parent="old-ppid")

        assert ss._active_session_id() == sid
        assert ss._pending_recovery["adopted_session"]["session_id"] == sid
        assert ss._pending_recovery["adopted_session"]["agent_count"] == 1

    def test_ignores_binding_with_only_terminal_agents(
        self, workspace: SimpleNamespace
    ) -> None:
        sid = "33333333-3333-3333-3333-333333333333"
        _write_session(workspace.base, sid, [])  # no resumable agents
        _write_binding(workspace.base, "b", sid)

        assert ss._recover_session_id() == ""

    def test_ignores_session_with_only_killed_agents(
        self, workspace: SimpleNamespace
    ) -> None:
        # B3: legacy pre-R5 kill left status="killed" records; a session with
        # ONLY killed agents must not be adopted as recoverable.
        sid = "3a3a3a3a-3a3a-3a3a-3a3a-3a3a3a3a3a3a"
        _write_session(
            workspace.base, sid, [{"name": "w", "pid": 1, "status": "killed"}]
        )
        _write_binding(workspace.base, "b", sid)

        assert ss._recover_session_id() == ""

    def test_ignores_binding_beyond_retention(self, workspace: SimpleNamespace) -> None:
        sid = "44444444-4444-4444-4444-444444444444"
        _write_session(workspace.base, sid, [_agent()])
        path = _write_binding(workspace.base, "old", sid)
        old = time.time() - 40 * 86400
        os.utime(path, (old, old))

        assert ss._recover_session_id() == ""

    def test_no_autoadopt_env_disables_adoption(
        self, workspace: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid = "55555555-5555-5555-5555-555555555555"
        _write_session(workspace.base, sid, [_agent()])
        _write_binding(workspace.base, "b", sid)
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_AUTOADOPT", "1")

        assert ss._recover_session_id() == ""
        assert "recoverable_sessions" in ss._pending_recovery


class TestSingleVsMultiLead:
    def test_single_binding_history_auto_adopts(
        self, workspace: SimpleNamespace
    ) -> None:
        sid = "66666666-6666-6666-6666-666666666666"
        _write_session(workspace.base, sid, [_agent()])
        _write_binding(workspace.base, "b", sid)

        assert ss._recover_session_id() == sid

    def test_two_candidates_do_not_auto_adopt(self, workspace: SimpleNamespace) -> None:
        sid1 = "77777777-7777-7777-7777-777777777777"
        sid2 = "88888888-8888-8888-8888-888888888888"
        _write_session(workspace.base, sid1, [_agent("a")])
        _write_session(workspace.base, sid2, [_agent("b")])
        _write_binding(workspace.base, "b1", sid1)
        _write_binding(workspace.base, "b2", sid2)

        assert ss._recover_session_id() == ""
        recoverable = ss._pending_recovery["recoverable_sessions"]
        assert {r["session_id"] for r in recoverable} == {sid1, sid2}

    def test_same_cwd_single_candidate_from_other_lead_not_auto_adopted(
        self, workspace: SimpleNamespace
    ) -> None:
        # Two bound sessions (two leads), but only one has resumable agents.
        with_agents = "99999999-9999-9999-9999-999999999999"
        empty = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        _write_session(workspace.base, with_agents, [_agent()])
        _write_session(workspace.base, empty, [])
        _write_binding(workspace.base, "b1", with_agents)
        _write_binding(workspace.base, "b2", empty)

        # Multi-lead history (2 distinct bound sessions) disables auto-adopt
        # even though there is only one candidate.
        assert ss._recover_session_id() == ""


class TestRecoveryNudgeAndTools:
    def test_check_agent_surfaces_recoverable_sessions_and_hint(
        self, workspace: SimpleNamespace
    ) -> None:
        sid1 = "10000000-0000-0000-0000-000000000001"
        sid2 = "10000000-0000-0000-0000-000000000002"
        _write_session(workspace.base, sid1, [_agent("a")])
        _write_session(workspace.base, sid2, [_agent("b")])
        _write_binding(workspace.base, "b1", sid1)
        _write_binding(workspace.base, "b2", sid2)

        result = asyncio.run(ss.check_agent("whoever"))

        assert "recoverable_sessions" in result
        assert "recovery_hint" in result
        assert "resume_session" in result["recovery_hint"]

    def test_resume_session_pins_and_rewrites_binding(
        self, workspace: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid1 = "20000000-0000-0000-0000-000000000001"
        sid2 = "20000000-0000-0000-0000-000000000002"
        _write_session(workspace.base, sid1, [_agent("a")])
        _write_session(workspace.base, sid2, [_agent("b")])
        _write_binding(workspace.base, "b1", sid1)
        _write_binding(workspace.base, "b2", sid2)

        result = asyncio.run(ss.resume_session(sid2))

        assert result["success"] is True
        assert result["session_id"] == sid2
        assert result["agent_count"] == 1
        assert ss._session_id == sid2
        # The exact-key fast path now hits for sid2.
        monkeypatch.setattr(ss, "_session_id", "")
        assert ss._recover_session_id() == sid2

    def test_resume_session_unknown_uuid_fails(
        self, workspace: SimpleNamespace
    ) -> None:
        result = asyncio.run(ss.resume_session("40000000-0000-0000-0000-000000000009"))
        assert result["success"] is False
        assert result["reason"] == "session_not_found"

    def test_resume_session_rejects_non_uuid_and_traversal(
        self, workspace: SimpleNamespace
    ) -> None:
        for bad in ("does-not-exist", "../../etc", "..", "a/b"):
            result = asyncio.run(ss.resume_session(bad))
            assert result["success"] is False
            assert result["reason"] == "invalid_session_id"

    def test_session_info_lists_candidates(self, workspace: SimpleNamespace) -> None:
        sid1 = "30000000-0000-0000-0000-000000000001"
        sid2 = "30000000-0000-0000-0000-000000000002"
        _write_session(workspace.base, sid1, [_agent("a")])
        _write_session(workspace.base, sid2, [_agent("b")])
        _write_binding(workspace.base, "b1", sid1)
        _write_binding(workspace.base, "b2", sid2)

        info = asyncio.run(ss.session_info())

        assert info["identity"] == ss.IDENTITY
        got = {r["session_id"] for r in info["recoverable_sessions"]}
        assert got == {sid1, sid2}


class TestToolDescriptionsMentionRecovery:
    @pytest.mark.asyncio
    async def test_list_agents_and_agent_status_point_at_session_info(self) -> None:
        for tool_name in ("list_agents", "agent_status"):
            tool = await ss.mcp.get_tool(tool_name)
            assert "session_info" in (tool.description or "")

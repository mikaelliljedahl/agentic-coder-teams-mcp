"""Tests for pi worker identity: literal per-agent MCP config + fail-loud guards.

Covers the two-part fix for the pi worker session-hijack bug:

1. Each spawned pi worker gets a per-agent MCP config file with *literal*
   ``AGENT_*`` values, passed via ``--mcp-config`` (mirrors the Claude path).
2. When ``AGENT_NAME`` is empty but a spawned-subagent signal
   (``WIN_AGENT_TEAMS_SESSION_DIR``) is present, identity is *unresolved* rather
   than silently masquerading as ``team-lead``; the identity-bearing tools then
   refuse loudly instead of hijacking the lead's inbox/session.
"""

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple as ss
from claude_teams.backends import pi as pi_module
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.pi import PiBackend


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Relocate import-time-bound bases + identity globals off real user state."""
    base = tmp_path / "sessions"
    base.mkdir()
    teams = tmp_path / "teams"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    monkeypatch.setattr(ss, "_TEAMS_BASE", teams)
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    monkeypatch.setattr(ss, "_AGENT_PARENT_NAME", "")
    monkeypatch.setattr(ss, "IDENTITY", ss.ROOT_LEAD_NAME)
    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", False)
    monkeypatch.setattr(ss, "_pending_recovery", {})
    monkeypatch.setattr(ss, "_inbox_locks", {})
    monkeypatch.chdir(work)
    return SimpleNamespace(base=base, teams=teams, work=work)


def _make_session(base: Path, sid: str, agents_json: str = "[]") -> Path:
    d = base / sid
    (d / "mcp").mkdir(parents=True)
    (d / "agents.json").write_text(agents_json, encoding="utf-8")
    return d


def _pi_request(tmp_path: Path, *, with_config: bool = True) -> SpawnRequest:
    extra: dict = {"session_dir": str(tmp_path / "sess")}
    if with_config:
        extra["pi_mcp_config_path"] = str(
            tmp_path / "sess" / "mcp" / "worker.pi.mcp.json"
        )
    return SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="team-lead",
        reasoning_effort="",
        extra=extra,
    )


@pytest.fixture
def _direct_launch(monkeypatch):
    monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["node", "cli.js"])
    monkeypatch.setattr(pi_module.process_manager, "provides_tty", lambda *a, **k: True)


# ---------------------------------------------------------------------------
# Per-agent literal pi MCP config
# ---------------------------------------------------------------------------


def test_pi_config_has_literal_identity(isolated):
    sid = str(uuid.uuid4())
    (isolated.base / sid / "mcp").mkdir(parents=True)
    path = ss._write_pi_mcp_config(sid, "worker-1", "team-lead")
    text = path.read_text(encoding="utf-8")
    entry = json.loads(text)["mcpServers"]["win-agent-teams"]
    assert entry["env"]["AGENT_NAME"] == "worker-1"
    assert entry["env"]["AGENT_SESSION_ID"] == sid
    assert entry["env"]["AGENT_PARENT_NAME"] == "team-lead"
    assert entry["env"]["CLAUDE_TEAMS_PERMISSION_MODE"] == "bypass"
    # directTools lives at the server-ENTRY level (per review F4), not doc-top.
    assert entry["directTools"] is True
    assert "${" not in text
    assert path.name == "worker-1.pi.mcp.json"


def test_pi_config_distinct_from_claude_config(isolated):
    sid = str(uuid.uuid4())
    (isolated.base / sid / "mcp").mkdir(parents=True)
    claude = ss._write_mcp_config(sid, "worker-1", "team-lead")
    pi = ss._write_pi_mcp_config(sid, "worker-1", "team-lead")
    assert claude.name == "worker-1.mcp.json"
    assert pi.name == "worker-1.pi.mcp.json"
    assert claude != pi
    # The Claude file is untouched: no pi-specific keys leak into it.
    cdata = json.loads(claude.read_text(encoding="utf-8"))
    centry = cdata["mcpServers"]["win-agent-teams"]
    assert "directTools" not in centry
    assert "CLAUDE_TEAMS_PERMISSION_MODE" not in centry["env"]


# ---------------------------------------------------------------------------
# --mcp-config wiring in the pi backend
# ---------------------------------------------------------------------------


def test_build_command_includes_mcp_config(tmp_path, _direct_launch):
    req = _pi_request(tmp_path)
    cmd = PiBackend().build_command(req)
    cfg = (req.extra or {})["pi_mcp_config_path"]
    assert "--mcp-config" in cmd
    idx = cmd.index("--mcp-config")
    assert cmd[idx + 1] == cfg  # discrete argv token, not a joined string


def test_build_resume_command_includes_mcp_config(tmp_path, _direct_launch):
    req = _pi_request(tmp_path)
    cmd = PiBackend().build_resume_command(req, "sid-abc")
    cfg = (req.extra or {})["pi_mcp_config_path"]
    assert "--mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == cfg


def test_build_command_omits_mcp_config_when_absent(tmp_path, _direct_launch):
    req = _pi_request(tmp_path, with_config=False)
    cmd = PiBackend().build_command(req)
    assert "--mcp-config" not in cmd


# ---------------------------------------------------------------------------
# _hook_extra writes the pi config even with state hooks disabled
# ---------------------------------------------------------------------------


def test_hook_extra_writes_pi_config_with_state_hooks_off(isolated, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")
    monkeypatch.setattr(ss, "_ensure_pi_mcp_config", lambda: None)
    sid = str(uuid.uuid4())
    (isolated.base / sid / "mcp").mkdir(parents=True)
    extra = ss._hook_extra(sid, "worker-1", "pi")
    assert "pi_mcp_config_path" in extra
    assert Path(extra["pi_mcp_config_path"]).exists()
    # kill switch still suppresses the -e extensions
    assert "pi_state_extension_path" not in extra
    assert "pi_wake_extension_path" not in extra


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def test_resolve_identity_spawned_subagent_is_unresolved():
    name, unresolved = ss._resolve_identity(
        {"AGENT_NAME": "", "WIN_AGENT_TEAMS_SESSION_DIR": "/x/y"}
    )
    assert unresolved is True
    assert name != ss.ROOT_LEAD_NAME


def test_resolve_identity_root_lead_is_team_lead():
    name, unresolved = ss._resolve_identity({})
    assert unresolved is False
    assert name == ss.ROOT_LEAD_NAME


def test_resolve_identity_named_worker():
    name, unresolved = ss._resolve_identity(
        {"AGENT_NAME": "worker-1", "WIN_AGENT_TEAMS_SESSION_DIR": "/x/y"}
    )
    assert unresolved is False
    assert name == "worker-1"


# ---------------------------------------------------------------------------
# Tools refuse under unresolved identity
# ---------------------------------------------------------------------------


def _unresolved(monkeypatch):
    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", True)
    monkeypatch.setattr(ss, "IDENTITY", ss._UNRESOLVED_IDENTITY)


def test_send_message_refuses_when_identity_unresolved(isolated, monkeypatch):
    _unresolved(monkeypatch)
    monkeypatch.setattr(ss, "_active_session_id", lambda **k: "s1")
    result = asyncio.run(ss.send_message("hi"))
    assert result["success"] is False
    assert result["reason"] == "identity_unresolved"


def test_read_messages_refuses_and_does_not_touch_lead_inbox(isolated, monkeypatch):
    sid = str(uuid.uuid4())
    _make_session(isolated.base, sid)
    # Pre-seed the lead inbox to prove it is neither read nor cursored.
    lead_inbox = ss._inbox_file(sid, ss.ROOT_LEAD_NAME)
    lead_inbox.write_text(
        json.dumps({"from": "worker", "text": "secret", "ts": "t"}) + "\n",
        encoding="utf-8",
    )
    _unresolved(monkeypatch)
    monkeypatch.setattr(ss, "_active_session_id", lambda **k: sid)
    result = asyncio.run(ss.read_messages())
    assert result["success"] is False
    assert result["reason"] == "identity_unresolved"
    # The lead's unread cursor must NOT have advanced (no cursor file written).
    assert not ss._inbox_cursor_file(sid, ss.ROOT_LEAD_NAME).exists()


def test_resume_session_refuses_when_identity_unresolved(isolated, monkeypatch):
    sid = str(uuid.uuid4())
    _make_session(isolated.base, sid)
    _unresolved(monkeypatch)
    result = asyncio.run(ss.resume_session(sid))
    assert result["success"] is False
    assert result["reason"] == "identity_unresolved"


def test_spawn_agent_refuses_when_identity_unresolved(isolated, monkeypatch):
    _unresolved(monkeypatch)
    sessions_before = {p.name for p in isolated.base.iterdir()}
    result = asyncio.run(ss.spawn_agent("do stuff", name="worker"))
    assert result["success"] is False
    assert result["reason"] == "identity_unresolved"
    # No orphan session/binding created on disk, and no unhandled ValueError.
    assert {p.name for p in isolated.base.iterdir()} == sessions_before


def test_recover_session_id_no_silent_autoadopt_for_unresolved_child(
    isolated, monkeypatch
):
    _unresolved(monkeypatch)
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    # A single candidate + single-lead history would normally be auto-adopted.
    monkeypatch.setattr(
        ss,
        "_candidate_sessions",
        lambda: [{"session_id": "abc", "agent_count": 1, "last_activity": None}],
    )
    monkeypatch.setattr(ss, "_distinct_binding_sessions", lambda: {"abc"})
    result = ss._recover_session_id()
    assert result == ""
    assert "adopted_session" not in ss._pending_recovery
    assert "recovery_hint" not in ss._pending_recovery


# ---------------------------------------------------------------------------
# Root-lead regression: identity resolves to team-lead, tools work, nudge kept
# ---------------------------------------------------------------------------


def test_root_lead_send_message_works(isolated, monkeypatch):
    """A root lead's identity resolves far enough to classify a recipient.

    This test arrived on ``main`` asserting ``success is True`` for a recipient
    that does not exist in an empty registry, because ``send_message`` used to
    re-route any unknown name to the lead. R5 removed that: a typo silently
    delivered upstream is read by the wrong agent or by nobody, so an unknown
    recipient is now refused (``_classify_recipient``).

    The assertion is ported rather than deleted, because the property it guards
    is about **identity**, not deliverability — the section header says so, and
    the empty registry is incidental to ``_make_session``'s default. Reaching
    ``recipient_not_addressable`` proves the guarded property precisely: the
    call resolved ``IDENTITY`` to ``team-lead``, found the session, loaded the
    registry, and got all the way to recipient classification. An identity
    regression fails earlier and differently, which this still catches.
    """
    sid = str(uuid.uuid4())
    _make_session(isolated.base, sid)
    monkeypatch.setattr(ss, "IDENTITY", ss.ROOT_LEAD_NAME)
    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", False)
    monkeypatch.setattr(ss, "_active_session_id", lambda **k: sid)
    result = asyncio.run(ss.send_message("hi", to="worker"))
    assert result["reason"] == "recipient_not_addressable"
    assert result["recipient_class"] == "unknown"
    # The identity-failure modes this regression exists to catch:
    assert result["reason"] not in {"session_not_found", "identity_unresolved"}


def test_recovery_hint_retained_for_root_lead(isolated, monkeypatch):
    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", False)
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    cands = [
        {"session_id": "a", "agent_count": 1, "last_activity": None},
        {"session_id": "b", "agent_count": 1, "last_activity": None},
    ]
    monkeypatch.setattr(ss, "_candidate_sessions", lambda: cands)
    monkeypatch.setattr(ss, "_distinct_binding_sessions", lambda: {"a", "b"})
    result = ss._recover_session_id()
    assert result == ""
    assert "recovery_hint" in ss._pending_recovery
    assert ss._pending_recovery["recoverable_sessions"] == cands

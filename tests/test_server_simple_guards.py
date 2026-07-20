"""Guard / error-branch coverage for ``claude_teams.server_simple``.

These target the defensive branches the behaviour-focused suites do not reach:
pure-helper guards, corrupt-file tolerances, binding/recovery filtering, and the
"no active session" / "not found" / fake-registry paths of the MCP tools. Every
test asserts an observable return, selection, or state change (per the plan's
review dispositions); fault injection monkeypatches ``Path.open``/``read_text``/
``stat``/``rglob``/``iterdir`` rather than filesystem permissions.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple as ss


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Relocate every import-time-bound base + module global off real state.

    ``_SESSION_BASE``/``_TEAMS_BASE`` and the identity globals ``IDENTITY``/
    ``_AGENT_PARENT_NAME``/``_AGENT_SESSION_ID`` are bound at import, so patching
    ``Path.home`` is insufficient — patch the module attributes directly. Also
    reset the mutable lead-session globals and clear env overrides so a stray
    developer environment (e.g. running the suite inside a spawned agent with
    ``AGENT_NAME``/``AGENT_PARENT_NAME`` set) cannot leak into a test. Autouse so
    no guard test can accidentally touch real user state; tests that need the
    paths still request ``isolated`` for its return value.
    """
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
    monkeypatch.setattr(ss, "_pending_recovery", {})
    monkeypatch.setattr(ss, "_inbox_locks", {})
    for var in (
        "WIN_AGENT_TEAMS_IDLE_SECONDS",
        "WIN_AGENT_TEAMS_RETENTION_DAYS",
        "WIN_AGENT_TEAMS_LOG_DIR",
        "WIN_AGENT_TEAMS_NO_AUTOADOPT",
        "WIN_AGENT_TEAMS_PARENT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(work)
    return SimpleNamespace(base=base, teams=teams, work=work)


def _make_session(base: Path, sid: str, agents_json: str) -> Path:
    sdir = base / sid
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "agents.json").write_text(agents_json, encoding="utf-8")
    return sdir


def _write_binding(base: Path, digest: str, meta: dict, mtime: float | None = None):
    bindings = base / ss._BINDINGS_DIR_NAME
    bindings.mkdir(parents=True, exist_ok=True)
    path = bindings / f"{digest}.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_idle_seconds_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_IDLE_SECONDS", "not-a-number")
    assert ss._idle_seconds() == ss._FOLLOW_UP_IDLE_SECONDS


def test_read_state_marker_corrupt_returns_none(isolated):
    sid = "s1"
    sdir = isolated.base / sid
    sdir.mkdir()
    (sdir / "state-worker.json").write_text("{not valid", encoding="utf-8")
    assert ss._read_state_marker(sid, "worker") is None


def test_read_state_marker_non_dict_returns_none(isolated):
    sid = "s1"
    sdir = isolated.base / sid
    sdir.mkdir()
    (sdir / "state-worker.json").write_text("[1, 2]", encoding="utf-8")
    assert ss._read_state_marker(sid, "worker") is None


def test_read_json_object_corrupt_returns_empty(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("{broken", encoding="utf-8")
    assert ss._read_json_object(path) == {}


def test_read_json_object_non_dict_returns_empty(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert ss._read_json_object(path) == {}


def test_unique_agent_name_multiple_collisions():
    agents = [{"name": "foo"}, {"name": "foo-2"}]
    assert ss._unique_agent_name("foo", agents) == "foo-3"


def test_is_session_dir_rejects_non_uuid(isolated):
    d = isolated.base / "not-a-uuid"
    d.mkdir()
    (d / "agents.json").write_text("[]", encoding="utf-8")
    assert ss._is_session_dir(d) is False


def test_session_has_live_agent_corrupt_registry(isolated):
    _make_session(isolated.base, "s1", "{corrupt")
    assert ss._session_has_live_agent("s1") is False


def test_session_has_live_agent_non_list_registry(isolated):
    _make_session(isolated.base, "s1", '{"not": "a list"}')
    assert ss._session_has_live_agent("s1") is False


def test_remove_team_logs_skips_when_log_dir_override(isolated, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(isolated.teams / "custom"))
    logdir = isolated.teams / "s1"
    logdir.mkdir(parents=True)
    ss._remove_team_logs("s1")
    # Override active -> the default team-log dir is intentionally left intact.
    assert logdir.exists()


def test_safe_float_bad_values():
    assert ss._safe_float("abc") == 0.0
    assert ss._safe_float(object()) == 0.0
    assert ss._safe_float(None) == 0.0
    assert ss._safe_float("3.5") == 3.5


def test_read_agent_output_unknown_backend_returns_none():
    agent = {
        "backend": "mystery",
        "spawned_at": 1.0,
        "cwd": "/some/where",
        "name": "a",
        "session_id": "s",
    }
    assert ss._read_agent_output(agent) is None


def test_last_non_empty_line_all_blank():
    assert ss._last_non_empty_line("  \n\n \t ") == ""
    assert ss._last_non_empty_line(None) == ""
    assert ss._last_non_empty_line("a\n  b  \n") == "b"


def test_hook_extra_unknown_backend_returns_empty():
    assert ss._hook_extra("sid", "agent", "mystery-backend") == {}


def test_pi_wake_extension_dir_override_existing(isolated, monkeypatch, tmp_path):
    ext = tmp_path / "custom-wake"
    ext.mkdir()
    monkeypatch.setenv("WIN_AGENT_TEAMS_PI_WAKE_EXTENSION", str(ext))
    assert ss._pi_wake_extension_dir() == ext


def test_pi_wake_extension_dir_override_missing_returns_none(
    isolated, monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "WIN_AGENT_TEAMS_PI_WAKE_EXTENSION", str(tmp_path / "does-not-exist")
    )
    assert ss._pi_wake_extension_dir() is None


def test_hook_extra_pi_emits_both_extension_keys(isolated, monkeypatch, tmp_path):
    # Do not touch real ~/.pi state.
    monkeypatch.setattr(ss, "_ensure_pi_mcp_config", lambda: None)
    state_dir = tmp_path / "wat-state"
    wake_dir = tmp_path / "wat-wake"
    monkeypatch.setattr(ss, "_pi_state_extension_dir", lambda: state_dir)
    monkeypatch.setattr(ss, "_pi_wake_extension_dir", lambda: wake_dir)

    extra = ss._hook_extra("sid", "agent", "pi")

    assert extra == {
        "pi_state_extension_path": str(state_dir),
        "pi_wake_extension_path": str(wake_dir),
    }


def test_hook_extra_pi_state_hooks_off_disables_both(isolated, monkeypatch):
    # WIN_AGENT_TEAMS_STATE_HOOKS=0 is a single kill switch for BOTH pi
    # extensions (state reporting AND inbox-wake).
    monkeypatch.setattr(ss, "_ensure_pi_mcp_config", lambda: None)
    monkeypatch.setattr(ss, "_pi_state_extension_dir", lambda: Path("/x/state"))
    monkeypatch.setattr(ss, "_pi_wake_extension_dir", lambda: Path("/x/wake"))
    monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")

    assert ss._hook_extra("sid", "agent", "pi") == {}


def test_hook_extra_pi_missing_wake_dir_omits_key(isolated, monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "_ensure_pi_mcp_config", lambda: None)
    monkeypatch.setattr(ss, "_pi_state_extension_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(ss, "_pi_wake_extension_dir", lambda: None)

    extra = ss._hook_extra("sid", "agent", "pi")

    assert "pi_wake_extension_path" not in extra
    assert extra["pi_state_extension_path"] == str(tmp_path / "state")


def test_marker_timestamp_non_numeric():
    assert ss._marker_timestamp({"ts": "x"}) is None
    assert ss._marker_timestamp({"ts": True}) is None  # bool is rejected
    assert ss._marker_timestamp(None) is None
    assert ss._marker_timestamp({"ts": 12.5}) == 12.5


def test_recovery_note_clears_adopted_session(isolated, monkeypatch):
    monkeypatch.setattr(
        ss, "_pending_recovery", {"adopted_session": {"session_id": "s1"}}
    )
    note = ss._recovery_note()
    assert note == {"adopted_session": {"session_id": "s1"}}
    # One-shot: cleared after first surface.
    assert ss._pending_recovery == {}


# --------------------------------------------------------------------------
# _dir_newest_mtime error branches
# --------------------------------------------------------------------------


def test_dir_newest_mtime_rglob_oserror_returns_stat(isolated, monkeypatch):
    d = isolated.base / "s1"
    d.mkdir()
    base_mtime = d.stat().st_mtime

    def boom(self, *args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "rglob", boom)
    # Falls back to the directory's own mtime when descent fails.
    assert ss._dir_newest_mtime(d) == base_mtime


def test_dir_newest_mtime_skips_unstattable_entry(isolated, monkeypatch):
    d = isolated.base / "s1"
    d.mkdir()
    child = d / "child.txt"
    child.write_text("x", encoding="utf-8")

    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "child.txt":
            raise OSError
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    # The unreadable child is skipped; the dir's own mtime survives.
    assert ss._dir_newest_mtime(d) == real_stat(d).st_mtime


# --------------------------------------------------------------------------
# cleanup_old_sessions / _maybe_cleanup_old_sessions
# --------------------------------------------------------------------------


def test_cleanup_old_sessions_missing_base(isolated, monkeypatch):
    monkeypatch.setattr(ss, "_SESSION_BASE", isolated.base / "does-not-exist")
    assert ss.cleanup_old_sessions() == []


def test_cleanup_old_sessions_iterdir_oserror(isolated, monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "iterdir", boom)
    assert ss.cleanup_old_sessions() == []


def test_maybe_cleanup_tolerates_corrupt_stamp(isolated):
    stamp = isolated.base / ss._CLEANUP_STAMP_NAME
    stamp.write_text("not-a-float", encoding="utf-8")
    # Corrupt stamp is treated as "never cleaned" (last=0.0); cleanup proceeds
    # and rewrites the stamp with a real epoch float.
    ss._maybe_cleanup_old_sessions()
    assert float(stamp.read_text(encoding="utf-8").strip()) > 0.0


# --------------------------------------------------------------------------
# _message_recipient
# --------------------------------------------------------------------------


def test_message_recipient_known_agent_used_verbatim(isolated):
    _make_session(isolated.base, "s1", json.dumps([{"name": "worker", "pid": 1}]))
    recipient, warning = ss._message_recipient("worker", "s1")
    assert recipient == "worker"
    assert warning is None


def test_message_recipient_unknown_routes_to_lead_with_warning(isolated):
    _make_session(isolated.base, "s1", "[]")
    recipient, warning = ss._message_recipient("ghost", "s1")
    assert recipient == ss.ROOT_LEAD_NAME
    assert warning is not None
    assert "ghost" in warning


# --------------------------------------------------------------------------
# Binding helpers: _prune_superseded_bindings / _iter_binding_metas
# --------------------------------------------------------------------------


def test_prune_superseded_bindings_no_dir_is_noop(isolated):
    # bindings/ does not exist yet -> early return, no error.
    assert ss._prune_superseded_bindings("s1") is None


def test_prune_superseded_bindings_skips_unreadable(isolated, monkeypatch):
    path = _write_binding(isolated.base, "deadbeef", {"session_id": "s1"})
    real_read = Path.read_text

    def flaky_read(self, *args, **kwargs):
        if self.name == "deadbeef.json":
            raise OSError
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    ss._prune_superseded_bindings("s1")
    # Unreadable binding is skipped rather than unlinked.
    assert path.exists()


def test_iter_binding_metas_skips_non_json(isolated):
    bindings = isolated.base / ss._BINDINGS_DIR_NAME
    bindings.mkdir(parents=True)
    (bindings / "note.txt").write_text("ignore me", encoding="utf-8")
    (bindings / "good.json").write_text(json.dumps({"session_id": "s1"}), "utf-8")
    metas = list(ss._iter_binding_metas())
    names = {p.name for p, _m, _t in metas}
    assert names == {"good.json"}


def test_iter_binding_metas_skips_unreadable(isolated, monkeypatch):
    _write_binding(isolated.base, "aaaa", {"session_id": "s1"})
    real_read = Path.read_text

    def flaky_read(self, *args, **kwargs):
        if self.name == "aaaa.json":
            raise OSError
        return real_read(self, *args, **kwargs)

    # is_file()/suffix pass; the read inside the try raises OSError -> skipped.
    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert list(ss._iter_binding_metas()) == []


# --------------------------------------------------------------------------
# _candidate_sessions / _distinct_binding_sessions filtering
# --------------------------------------------------------------------------


def _binding_meta(sid: str, *, identity=None, cwd=None) -> dict:
    return {
        "session_id": sid,
        "identity": ss.IDENTITY if identity is None else identity,
        "cwd": str(Path.cwd().resolve()) if cwd is None else cwd,
        "updated_at": "2025-01-01T00:00:00+00:00",
    }


def test_candidate_sessions_filters_bad_bindings(isolated):
    # Good candidate: matching identity/cwd + a live registry.
    _make_session(
        isolated.base, "good", json.dumps([{"name": "w", "status": "running"}])
    )
    _write_binding(isolated.base, "b_good", _binding_meta("good"))
    # Foreign identity/cwd -> skipped (line 481).
    _write_binding(
        isolated.base, "b_foreign", _binding_meta("foreign", identity="someone-else")
    )
    # Missing session_id -> skipped (line 486).
    _write_binding(isolated.base, "b_nosid", _binding_meta(""))
    # Points at a corrupt registry -> skipped (lines 489-490).
    _make_session(isolated.base, "corrupt", "{broken")
    _write_binding(isolated.base, "b_corrupt", _binding_meta("corrupt"))
    # Registry is a dict, not a list -> skipped (line 492).
    _make_session(isolated.base, "notlist", '{"x": 1}')
    _write_binding(isolated.base, "b_notlist", _binding_meta("notlist"))

    result = ss._candidate_sessions()
    assert [c["session_id"] for c in result] == ["good"]


def test_distinct_binding_sessions_filters_foreign_and_stale(isolated):
    _make_session(isolated.base, "fresh", json.dumps([{"name": "w"}]))
    _write_binding(isolated.base, "b_fresh", _binding_meta("fresh"))
    # Foreign identity -> excluded (line 537).
    _write_binding(isolated.base, "b_foreign", _binding_meta("other", identity="nope"))
    # Stale mtime (well beyond retention) -> excluded (line 539).
    old = time.time() - ss._RETENTION_DAYS_DEFAULT * 86400.0 - 10_000.0
    _write_binding(isolated.base, "b_stale", _binding_meta("stale"), mtime=old)

    assert ss._distinct_binding_sessions() == {"fresh"}


# --------------------------------------------------------------------------
# MCP tool guards (per-tool expectation matrix)
# --------------------------------------------------------------------------


def _no_session(monkeypatch):
    monkeypatch.setattr(ss, "_active_session_id", lambda **kwargs: "")


def test_send_message_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    result = asyncio.run(ss.send_message("hi"))
    assert result["success"] is False
    assert result["reason"] == "session_not_found"


def test_read_messages_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    result = asyncio.run(ss.read_messages())
    assert result["messages"] == []
    assert result["unread_count"] == 0
    assert result["seq"] is None


def test_kill_agent_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    result = asyncio.run(ss.kill_agent("worker"))
    assert result == {"success": False, "name": "worker", "reason": "session_not_found"}


def test_kill_agent_unknown_agent(isolated, monkeypatch):
    _make_session(isolated.base, "s1", "[]")
    monkeypatch.setattr(ss, "_active_session_id", lambda **kwargs: "s1")
    result = asyncio.run(ss.kill_agent("ghost"))
    assert result == {"success": False, "name": "ghost"}


def test_resume_session_blank_id(isolated):
    result = asyncio.run(ss.resume_session("   "))
    assert result == {"success": False, "reason": "session_id_required"}


def test_resume_session_tolerates_corrupt_registry(isolated):
    sid = str(uuid.uuid4())
    _make_session(isolated.base, sid, "{corrupt registry")
    result = asyncio.run(ss.resume_session(sid))
    assert result["success"] is True
    assert result["session_id"] == sid
    assert result["agent_count"] == 0


def test_session_info_tolerates_corrupt_registry(isolated, monkeypatch):
    sid = str(uuid.uuid4())
    _make_session(isolated.base, sid, "{corrupt")
    monkeypatch.setattr(ss, "_active_session_id", lambda **kwargs: sid)
    result = asyncio.run(ss.session_info())
    assert result["session_id"] == sid
    assert result["agent_count"] == 0


def test_list_agents_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    assert asyncio.run(ss.list_agents()) == []


def test_agent_status_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    assert asyncio.run(ss.agent_status()) == []


def test_list_backends_enumerates_registry(isolated, monkeypatch):
    class FakeBackend:
        binary_name = "foo-bin"

        def default_model(self):
            return "m1"

        def supported_models(self):
            return ["m1", "m2"]

    fake = SimpleNamespace(
        list_available=lambda: ["foo"],
        get=lambda name: FakeBackend(),
    )
    monkeypatch.setattr(ss, "registry", fake)
    result = asyncio.run(ss.list_backends())
    assert result == [
        {
            "name": "foo",
            "binary": "foo-bin",
            "default_model": "m1",
            "supported_models": ["m1", "m2"],
        }
    ]


def test_follow_up_agent_no_session(isolated, monkeypatch):
    _no_session(monkeypatch)
    result = asyncio.run(ss.follow_up_agent("worker", "next"))
    assert result["success"] is False
    assert result["reason"] == "session_not_found"


def test_follow_up_agent_unsupported_backend(isolated, monkeypatch):
    _make_session(
        isolated.base,
        "s1",
        json.dumps([{"name": "worker", "pid": 1, "backend": "totally-bogus"}]),
    )
    monkeypatch.setattr(ss, "_active_session_id", lambda **kwargs: "s1")
    result = asyncio.run(ss.follow_up_agent("worker", "next"))
    assert result["success"] is False
    assert result["reason"] == "backend_not_supported"

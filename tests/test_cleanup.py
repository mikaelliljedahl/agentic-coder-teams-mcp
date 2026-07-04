"""R6 — 30-day auto cleanup of stale session dirs, team logs, and bindings.

Cleanup only ever touches real (UUID-named, registry-bearing) session dirs;
never the bindings dir, the cleanup stamp, the active session, or a session
with a live agent. Best-effort and throttled to at most once per day.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple as ss

_UUID_A = "11111111-1111-1111-1111-11111111aaaa"
_UUID_B = "22222222-2222-2222-2222-22222222bbbb"


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    teams = tmp_path / "teams"
    monkeypatch.setattr(ss, "_SESSION_BASE", sessions)
    monkeypatch.setattr(ss, "_TEAMS_BASE", teams)
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.delenv("WIN_AGENT_TEAMS_LOG_DIR", raising=False)
    monkeypatch.delenv("WIN_AGENT_TEAMS_RETENTION_DAYS", raising=False)
    return SimpleNamespace(sessions=sessions, teams=teams)


def _make_session(sessions: Path, sid: str, agents: list[dict] | None = None) -> Path:
    d = sessions / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents.json").write_text(json.dumps(agents or []), encoding="utf-8")
    return d


def _age(path: Path, days: float) -> None:
    old = time.time() - days * 86400
    for entry in [path, *path.rglob("*")]:
        os.utime(entry, (old, old))


class TestCleanupRemoval:
    def test_removes_session_dir_older_than_cutoff(self, base: SimpleNamespace) -> None:
        d = _make_session(base.sessions, _UUID_A)
        _age(d, 40)

        removed = ss.cleanup_old_sessions(max_age_days=30)

        assert _UUID_A in removed
        assert not d.exists()

    def test_keeps_recent_session_dir(self, base: SimpleNamespace) -> None:
        d = _make_session(base.sessions, _UUID_A)  # fresh mtime

        removed = ss.cleanup_old_sessions(max_age_days=30)

        assert removed == []
        assert d.exists()

    def test_never_removes_active_session(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _make_session(base.sessions, _UUID_A)
        _age(d, 99)
        monkeypatch.setattr(ss, "_session_id", _UUID_A)

        removed = ss.cleanup_old_sessions(max_age_days=30)

        assert removed == []
        assert d.exists()

    def test_never_removes_session_with_live_agent(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _make_session(base.sessions, _UUID_A, [{"name": "w", "pid": 1}])
        _age(d, 99)
        monkeypatch.setattr(ss, "_agent_alive", lambda agent: True)

        removed = ss.cleanup_old_sessions(max_age_days=30)

        assert removed == []
        assert d.exists()

    def test_removes_matching_team_log_dir(self, base: SimpleNamespace) -> None:
        d = _make_session(base.sessions, _UUID_A)
        _age(d, 40)
        team = base.teams / _UUID_A
        team.mkdir(parents=True)
        (team / "worker.log").write_text("log", encoding="utf-8")

        ss.cleanup_old_sessions(max_age_days=30)

        assert not team.exists()


class TestCleanupSafety:
    def test_never_deletes_bindings_dir_or_last_cleanup_stamp(
        self, base: SimpleNamespace
    ) -> None:
        bindings = base.sessions / "bindings"
        bindings.mkdir()
        orphan = bindings / "orphan.json"
        orphan.write_text(json.dumps({"session_id": "gone-session"}), encoding="utf-8")
        stamp = base.sessions / ".last-cleanup"
        stamp.write_text("0", encoding="utf-8")
        _age(bindings, 99)
        os.utime(stamp, (time.time() - 99 * 86400,) * 2)

        ss.cleanup_old_sessions(max_age_days=30)

        assert bindings.is_dir()  # dir itself survives
        assert stamp.exists()
        assert not orphan.exists()  # orphan binding file pruned

    def test_prunes_orphan_binding_files(self, base: SimpleNamespace) -> None:
        bindings = base.sessions / "bindings"
        bindings.mkdir()
        live_sid = _UUID_B
        _make_session(base.sessions, live_sid, [{"name": "w"}])
        (bindings / "live.json").write_text(
            json.dumps({"session_id": live_sid}), encoding="utf-8"
        )
        (bindings / "dead.json").write_text(
            json.dumps({"session_id": "no-such-session"}), encoding="utf-8"
        )

        ss.cleanup_old_sessions(max_age_days=30)

        assert (bindings / "live.json").exists()
        assert not (bindings / "dead.json").exists()

    def test_cleanup_swallows_errors(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_session(base.sessions, _UUID_A)

        def boom(path: Path) -> float:
            raise OSError("boom")

        monkeypatch.setattr(ss, "_dir_newest_mtime", boom)
        # Must not raise even though per-entry inspection fails.
        ss._maybe_cleanup_old_sessions()


class TestRetentionAndThrottle:
    def test_retention_days_env_override(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _make_session(base.sessions, _UUID_A)
        _age(d, 5)
        monkeypatch.setenv("WIN_AGENT_TEAMS_RETENTION_DAYS", "3")

        removed = ss.cleanup_old_sessions()  # uses env retention (3 days)

        assert _UUID_A in removed

    def test_invalid_retention_falls_back_to_default(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_RETENTION_DAYS", "-1")
        assert ss._retention_days() == 30.0
        monkeypatch.setenv("WIN_AGENT_TEAMS_RETENTION_DAYS", "garbage")
        assert ss._retention_days() == 30.0

    def test_throttled_by_stamp_file(
        self, base: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            ss, "cleanup_old_sessions", lambda *a, **k: calls.append(1) or []
        )

        ss._maybe_cleanup_old_sessions()
        ss._maybe_cleanup_old_sessions()  # within a day → throttled

        assert len(calls) == 1

"""Tests for the external-member inbox-wake Stop hook (``member_wake``).

Covers the plan's full test matrix (docs/features/external-member-wake/plan.md
section 5, cases 1-20): the M0..M5 (+M2b) decision path, the
``install_member_wake`` tool, hooks wiring, lead/member coexistence, the
credential-absence invariant, and the adversarial cells.
"""

import asyncio
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import hooks, member_wake, procinfo
from claude_teams import server_simple as ss

MEMBER = "ext"


def _payload(**kw: object) -> dict:
    """Build a minimal Stop-hook stdin payload, overridable per test."""
    base: dict = {
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        "background_tasks": [],
    }
    base.update(kw)
    return base


def _member_record(
    name: str = MEMBER,
    status: str = "running",
    backend: str = "external",
    source: str = "join_ticket",
) -> dict:
    return {
        "name": name,
        "backend": backend,
        "spawned_by_source": source,
        "status": status,
    }


def _write_inbox(session_dir: Path, reader: str, senders: list[str]) -> None:
    lines = [json.dumps({"from": s, "text": f"hi from {s}"}) for s in senders]
    (session_dir / f"inbox-{reader}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_cursors(session_dir: Path, reader: str, cursors: dict[str, int]) -> None:
    (session_dir / f"inbox-{reader}.pos.json").write_text(
        json.dumps(cursors), encoding="utf-8"
    )


def _guard_path(session_dir: Path, member: str) -> Path:
    return session_dir / f"wake-progress-member-{member}.json"


def _write_member_guard(
    session_dir: Path,
    member: str,
    senders: dict[str, dict[str, int]],
    noprogress: int,
) -> None:
    _guard_path(session_dir, member).write_text(
        json.dumps(
            {
                "schema": "lead-wake-progress/1",
                "reader": f"member-{member}",
                "senders": senders,
                "noprogress_blocks": noprogress,
                "ts": 0.0,
            }
        ),
        encoding="utf-8",
    )


def _age_joined_dir(session_dir: Path, seconds: float) -> None:
    """Backdate every liveness-relevant file in the joined dir."""
    old = time.time() - seconds
    for pattern in ("state-*.json", "inbox-*.jsonl", "agents.json"):
        for path in session_dir.glob(pattern):
            os.utime(path, (old, old))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear identity + kill-switch env so a dev shell cannot leak into a test."""
    for var in (
        "AGENT_NAME",
        "WIN_AGENT_TEAMS_LEAD_WAKE",
        "WIN_AGENT_TEAMS_MEMBER_WAKE",
        "WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS",
        "WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def joined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A joined lead session dir with a live external membership for MEMBER.

    A fresh ``state-team-lead.json`` marker keeps the M2b liveness TTL armed
    by default; membership records are faked via ``server_simple._load_agents``
    (the same seam the lead-wake tests use).
    """
    d = tmp_path / "joined-sess"
    d.mkdir()
    (d / "state-team-lead.json").write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": time.time()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ss, "_load_agents", lambda _sid: [_member_record()])
    return d


def _evaluate(joined: Path, payload: dict | None = None, member: str = MEMBER):
    return member_wake.evaluate_member(
        payload if payload is not None else _payload(),
        member=member,
        joined_session_dir=str(joined),
    )


class TestKillSwitch:
    def test_member_kill_switch_off_allows(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 7 / 14: MEMBER_WAKE=0 short-circuits before any scan.
        monkeypatch.setenv("WIN_AGENT_TEAMS_MEMBER_WAKE", "0")

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M0"

    def test_member_unset_falls_back_to_lead_kill_switch(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 14: MEMBER unset + LEAD=0 -> both wakes disabled.
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE", "0")

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M0"

    def test_member_explicit_on_overrides_lead_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Case 14: MEMBER=1 + LEAD=0 -> member-wake still evaluates (a missing
        # joined dir must land on M1, proving M0 was passed).
        monkeypatch.setenv("WIN_AGENT_TEAMS_MEMBER_WAKE", "1")
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE", "0")

        result = _evaluate(tmp_path / "missing")

        assert result.action == "allow"
        assert result.code == "M1"


class TestFailOpenGates:
    def test_missing_joined_dir_allows(self, tmp_path: Path) -> None:
        # Case 6: joined dir missing / not a directory -> fail-open allow.
        result = _evaluate(tmp_path / "gone")

        assert result.action == "allow"
        assert result.code == "M1"

    def test_no_membership_record_allows(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 5: not a member here -> self-disable allow.
        monkeypatch.setattr(ss, "_load_agents", lambda _sid: [])

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M2"

    @pytest.mark.parametrize("status", ["left", "killed"])
    def test_left_or_terminal_membership_allows(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path, status: str
    ) -> None:
        # Cases 4 + 18: only status "running" is live; left/killed fail open.
        monkeypatch.setattr(
            ss, "_load_agents", lambda _sid: [_member_record(status=status)]
        )
        _write_inbox(joined, MEMBER, ["team-lead"])  # unread would block if live

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M2"

    def test_wrong_backend_or_source_is_not_a_membership(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # A same-named spawned agent record is NOT an external membership.
        monkeypatch.setattr(
            ss,
            "_load_agents",
            lambda _sid: [
                _member_record(backend="claude-code", source="spawn"),
            ],
        )

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M2"

    def test_unreadable_registry_allows(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        def _boom(_sid: str) -> list[dict]:
            raise OSError("locked")

        monkeypatch.setattr(ss, "_load_agents", _boom)

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M2"


class TestAbandonedTeamTtl:
    def test_stale_joined_session_allows(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 13: record still "running" but no lead-side activity within the
        # TTL -> fail-open allow instead of blocking forever.
        monkeypatch.setenv("WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS", "60")
        _write_inbox(joined, MEMBER, ["team-lead"])
        _age_joined_dir(joined, 3600)

        result = _evaluate(joined)

        assert result.action == "allow"
        assert result.code == "M2b"

    def test_fresh_activity_keeps_gate_armed(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 13: fresh activity within the TTL -> still blocks on unread.
        monkeypatch.setenv("WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS", "3600")
        _write_inbox(joined, MEMBER, ["team-lead"])

        result = _evaluate(joined)

        assert result.action == "block"
        assert result.code == "M3"


class TestDecisionCore:
    def test_unread_blocks_with_external_read_reason(self, joined: Path) -> None:
        # Case 1: unread in the joined inbox-<member> -> block; the reason names
        # external_read, never the ambient read_messages tool.
        _write_inbox(joined, MEMBER, ["team-lead"])

        result = _evaluate(joined)

        assert result.action == "block"
        assert result.code == "M3"
        assert result.reason is not None
        assert "team-lead" in result.reason
        assert "external_read" in result.reason
        assert "read_messages" not in result.reason

    def test_armed_watcher_for_joined_dir_allows(self, joined: Path) -> None:
        # Case 2: no unread + a running watcher for the JOINED dir -> allow.
        payload = _payload(
            background_tasks=[
                {
                    "status": "running",
                    "command": ss._watch_command_bash(joined, reader=MEMBER),
                }
            ]
        )

        result = _evaluate(joined, payload)

        assert result.action == "allow"
        assert result.code == "M4"

    def test_not_armed_blocks_with_reader_scoped_watch_command(
        self, joined: Path
    ) -> None:
        # Case 3: no unread + not armed -> block with the arm instruction that
        # renders the joined-dir watch command including --reader <member>.
        result = _evaluate(joined)

        assert result.action == "block"
        assert result.code == "M5"
        assert result.reason is not None
        assert "claude_teams.cli" in result.reason
        assert "watch" in result.reason
        assert str(joined) in result.reason
        assert "--reader" in result.reason
        assert MEMBER in result.reason
        assert "background" in result.reason.lower()
        # Hook-emitted commands are deliberately unbound: the hook's parent is a
        # transient wrapper, so a baked-in owner PID would die instantly (exit 4).
        assert "--owner-pid" not in result.reason
        assert "--owner-token" not in result.reason
        # The leave_team escape hatch (review-1 Major 1).
        assert "leave_team" in result.reason

    def test_watcher_for_different_session_does_not_count(
        self, tmp_path: Path, joined: Path
    ) -> None:
        # Case 19: armed near-miss -> a watcher for ANOTHER dir still blocks.
        other = tmp_path / "other-sess"
        other.mkdir()
        payload = _payload(
            background_tasks=[
                {
                    "status": "running",
                    "command": ss._watch_command_bash(other, reader=MEMBER),
                }
            ]
        )

        result = _evaluate(joined, payload)

        assert result.action == "block"
        assert result.code == "M5"


class TestProgressGuard:
    def test_guard_fail_open_after_shared_cap(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Cases 8 + 15: the guard cap deliberately shares
        # WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS; at the cap it fails open.
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "2")
        _write_member_guard(joined, MEMBER, {}, 1)  # next no-progress block -> 2

        result = _evaluate(joined, _payload(stop_hook_active=True))

        assert result.action == "allow"

    def test_guard_resets_after_cursor_advance(self, joined: Path) -> None:
        # Case 8: cursor advance resets the counter; the block proceeds.
        _write_inbox(joined, MEMBER, ["team-lead", "team-lead"])  # total 2
        _write_cursors(joined, MEMBER, {"team-lead": 1})  # unread 1
        _write_member_guard(joined, MEMBER, {"team-lead": {"total": 1, "cursor": 0}}, 2)

        result = _evaluate(joined, _payload(stop_hook_active=True))

        assert result.action == "block"
        assert result.code == "M3"
        guard = json.loads(_guard_path(joined, MEMBER).read_text(encoding="utf-8"))
        assert guard["noprogress_blocks"] == 0
        assert set(guard) == {
            "schema",
            "reader",
            "senders",
            "noprogress_blocks",
            "ts",
        }
        assert guard["schema"] == "lead-wake-progress/1"
        assert guard["reader"] == f"member-{MEMBER}"
        assert "owner_generation" not in guard
        assert not (joined / f"wake-progress-{MEMBER}.json").exists()

    def test_guard_not_consulted_without_stop_hook_active(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # A first Stop (stop_hook_active false) never fail-opens.
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "2")
        _write_member_guard(joined, MEMBER, {}, 5)

        result = _evaluate(joined, _payload(stop_hook_active=False))

        assert result.action == "block"
        assert result.code == "M5"

    def test_guard_file_is_member_prefixed_even_for_team_lead_name(
        self, monkeypatch: pytest.MonkeyPatch, joined: Path
    ) -> None:
        # Case 17: a member literally named "team-lead" writes
        # wake-progress-member-team-lead.json, never the lead's guard file.
        monkeypatch.setattr(
            ss, "_load_agents", lambda _sid: [_member_record(name="team-lead")]
        )
        _write_inbox(joined, "team-lead", ["alice"])

        result = _evaluate(joined, member="team-lead")

        assert result.action == "block"
        assert (joined / "wake-progress-member-team-lead.json").exists()
        assert not (joined / "wake-progress-team-lead.json").exists()


class TestMainEntrypoint:
    def test_main_block_logs_member_wake_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        joined: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Case 16: stderr line is prefixed win-agent-teams/member-wake.
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload())))

        member_wake.main(["--joined-session-dir", str(joined), "--member", MEMBER])

        out = capsys.readouterr()
        decision = json.loads(out.out)
        assert decision["decision"] == "block"
        assert "win-agent-teams/member-wake" in out.err
        assert "lead-wake" not in out.err

    def test_main_allow_prints_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload())))

        member_wake.main(
            ["--joined-session-dir", str(tmp_path / "gone"), "--member", MEMBER]
        )

        out = capsys.readouterr()
        assert out.out == ""
        assert "win-agent-teams/member-wake" in out.err

    def test_main_garbage_stdin_never_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Case 20: corrupt stdin + missing dir -> no raise, no block.
        monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))

        member_wake.main(
            ["--joined-session-dir", str(tmp_path / "gone"), "--member", MEMBER]
        )

        assert capsys.readouterr().out == ""

    def test_main_missing_member_arg_never_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        joined: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Case 20: a mis-baked hook without --member fails open (allow).
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload())))

        member_wake.main(["--joined-session-dir", str(joined)])

        assert capsys.readouterr().out == ""


@pytest.fixture
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Relocate session base + home + cwd off real state (mirror lead-wake suite)."""
    base = tmp_path / "sessions"
    base.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    monkeypatch.setattr(ss, "IDENTITY", ss.ROOT_LEAD_NAME)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    host = procinfo.ProcessInfo(101, 1, "claude.exe")
    monkeypatch.setattr(
        ss.procinfo,
        "resolve_nearest_host",
        lambda: procinfo.HostResolution(chain=(host,), host=host),
    )
    monkeypatch.setattr(
        ss.process_manager_module, "creation_token", lambda _pid: "lead-token"
    )
    return SimpleNamespace(base=base, home=home, work=work)


def _make_joined_session(base: Path) -> str:
    sid = str(uuid.uuid4())
    d = base / sid
    d.mkdir()
    (d / "agents.json").write_text(json.dumps([_member_record()]), encoding="utf-8")
    return sid


def _stop_commands(config: dict) -> list[str]:
    return [g["hooks"][0]["command"] for g in config["hooks"]["Stop"]]


def _user_settings(ns: SimpleNamespace) -> dict:
    return json.loads(
        (ns.home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )


class TestInstallMemberWakeTool:
    def test_install_defaults_to_user_scope(self, _isolated: SimpleNamespace) -> None:
        sid = _make_joined_session(_isolated.base)

        result = asyncio.run(ss.install_member_wake(sid, MEMBER))

        settings = _isolated.home / ".claude" / "settings.json"
        assert Path(result["path"]) == settings
        assert result["action"] == "installed"
        assert result["member"] == MEMBER
        assert result["scope"] == "user"
        assert Path(result["joined_session_dir"]) == _isolated.base / sid
        cmds = _stop_commands(_user_settings(_isolated))
        assert any("claude_teams.member_wake" in c for c in cmds)
        assert any("--joined-session-dir" in c and "--member" in c for c in cmds)

    def test_install_is_idempotent(self, _isolated: SimpleNamespace) -> None:
        # Case 11: re-run replaces in place, no duplicate group.
        sid = _make_joined_session(_isolated.base)

        asyncio.run(ss.install_member_wake(sid, MEMBER))
        asyncio.run(ss.install_member_wake(sid, MEMBER))

        cmds = _stop_commands(_user_settings(_isolated))
        assert len([c for c in cmds if "claude_teams.member_wake" in c]) == 1

    def test_remove_drops_only_the_member_group(
        self, _isolated: SimpleNamespace
    ) -> None:
        sid = _make_joined_session(_isolated.base)
        asyncio.run(ss.install_member_wake(sid, MEMBER))

        result = asyncio.run(ss.install_member_wake(sid, MEMBER, remove=True))

        assert result["action"] == "removed"
        config = _user_settings(_isolated)
        stop = config.get("hooks", {}).get("Stop", [])
        assert not any(
            "claude_teams.member_wake" in g["hooks"][0]["command"] for g in stop
        )

    def test_project_scope_writes_cwd_settings(
        self, _isolated: SimpleNamespace
    ) -> None:
        sid = _make_joined_session(_isolated.base)

        result = asyncio.run(ss.install_member_wake(sid, MEMBER, scope="project"))

        assert Path(result["path"]) == _isolated.work / ".claude" / "settings.json"

    def test_invalid_session_id_errors(self, _isolated: SimpleNamespace) -> None:
        # Case 12: bad joined_session_id -> error, nothing written.
        result = asyncio.run(ss.install_member_wake("not-a-uuid", MEMBER))

        assert result.get("success") is False
        assert result["reason"] == "invalid_session_id"
        assert not (_isolated.home / ".claude" / "settings.json").exists()

    def test_unknown_session_errors(self, _isolated: SimpleNamespace) -> None:
        result = asyncio.run(ss.install_member_wake(str(uuid.uuid4()), MEMBER))

        assert result.get("success") is False
        assert result["reason"] == "session_not_found"

    def test_empty_member_name_errors(self, _isolated: SimpleNamespace) -> None:
        sid = _make_joined_session(_isolated.base)

        result = asyncio.run(ss.install_member_wake(sid, "   "))

        assert "error" in result

    def test_bad_scope_errors(self, _isolated: SimpleNamespace) -> None:
        sid = _make_joined_session(_isolated.base)

        result = asyncio.run(ss.install_member_wake(sid, MEMBER, scope="global"))

        assert "error" in result


class TestCoexistence:
    def test_member_and_lead_groups_coexist(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Case 10: install member then lead -> both Stop groups present.
        sid = _make_joined_session(_isolated.base)
        monkeypatch.setattr(ss, "_session_id", sid)

        asyncio.run(ss.install_member_wake(sid, MEMBER))
        asyncio.run(ss.install_lead_wake(scope="user"))

        cmds = _stop_commands(_user_settings(_isolated))
        assert any("claude_teams.member_wake" in c for c in cmds)
        assert any("claude_teams.lead_wake" in c for c in cmds)

    def test_removing_member_preserves_lead(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid = _make_joined_session(_isolated.base)
        monkeypatch.setattr(ss, "_session_id", sid)
        asyncio.run(ss.install_lead_wake(scope="user"))
        asyncio.run(ss.install_member_wake(sid, MEMBER))

        asyncio.run(ss.install_member_wake(sid, MEMBER, remove=True))

        cmds = _stop_commands(_user_settings(_isolated))
        assert not any("claude_teams.member_wake" in c for c in cmds)
        assert any("claude_teams.lead_wake" in c for c in cmds)

    def test_removing_lead_preserves_member(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid = _make_joined_session(_isolated.base)
        monkeypatch.setattr(ss, "_session_id", sid)
        asyncio.run(ss.install_member_wake(sid, MEMBER))
        asyncio.run(ss.install_lead_wake(scope="user"))

        asyncio.run(ss.install_lead_wake(remove=True, scope="user"))

        cmds = _stop_commands(_user_settings(_isolated))
        assert any("claude_teams.member_wake" in c for c in cmds)
        assert not any("claude_teams.lead_wake" in c for c in cmds)


class TestCredentialAbsence:
    def test_settings_and_argv_carry_no_token(self, _isolated: SimpleNamespace) -> None:
        # Case 9: the baked hook carries only the member name + joined dir.
        sid = _make_joined_session(_isolated.base)

        asyncio.run(ss.install_member_wake(sid, MEMBER))

        text = (_isolated.home / ".claude" / "settings.json").read_text(
            encoding="utf-8"
        )
        assert "member_token" not in text
        assert "wam1:" not in text
        argv = hooks._member_wake_command(_isolated.base / sid, MEMBER)
        assert "--joined-session-dir" in argv
        assert "--member" in argv
        assert MEMBER in argv
        assert not any("token" in token.lower() for token in argv)


class TestHooksWiring:
    def test_member_wake_command_shape(self, tmp_path: Path) -> None:
        argv = hooks._member_wake_command(tmp_path, MEMBER)

        assert argv[1:3] == ["-m", "claude_teams.member_wake"]
        assert argv[argv.index("--joined-session-dir") + 1] == tmp_path.as_posix()
        assert argv[argv.index("--member") + 1] == MEMBER

    def test_member_wake_hook_matcher_shape(self, tmp_path: Path) -> None:
        matcher = hooks._member_wake_hook_matcher(tmp_path, MEMBER)

        entry = matcher["hooks"][0]
        assert entry["type"] == "command"
        assert "claude_teams.member_wake" in entry["command"]
        assert entry["timeout"] == hooks._WAKE_HOOK_TIMEOUT_SECONDS


class TestWatchCommandReader:
    def test_watch_command_bash_forwards_reader(self, tmp_path: Path) -> None:
        cmd = ss._watch_command_bash(tmp_path, reader=MEMBER)

        assert "--reader" in cmd
        assert MEMBER in cmd

    def test_watch_command_bash_default_unchanged(self, tmp_path: Path) -> None:
        cmd = ss._watch_command_bash(tmp_path)

        assert "--reader" not in cmd

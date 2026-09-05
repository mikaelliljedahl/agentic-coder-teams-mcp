"""Tests for the Claude Code lead inbox-wake decision hook (``lead_wake``)."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from claude_teams import lead_wake, procinfo


def _payload(**kw: object) -> dict:
    """Build a minimal Stop-hook stdin payload, overridable per test."""
    base: dict = {
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        "background_tasks": [],
    }
    base.update(kw)
    return base


def _write_inbox(session_dir: Path, reader: str, senders: list[str]) -> None:
    """Append one message per ``senders`` entry to ``inbox-<reader>.jsonl``."""
    lines = [json.dumps({"from": s, "text": f"hi from {s}"}) for s in senders]
    (session_dir / f"inbox-{reader}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_cursors(session_dir: Path, reader: str, cursors: dict[str, int]) -> None:
    (session_dir / f"inbox-{reader}.pos.json").write_text(
        json.dumps(cursors), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear identity + kill-switch env so a dev shell cannot leak into a test."""
    for var in (
        "AGENT_NAME",
        "WIN_AGENT_TEAMS_LEAD_WAKE",
        "WIN_AGENT_TEAMS_LEAD_WAKE_OWNER",
        "WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS",
    ):
        monkeypatch.delenv(var, raising=False)

    # Legacy decision tests intentionally exercise private per-agent wiring.
    # Tests that pass owner_mode explicitly still cover the shared-hook matrix.
    original_evaluate = lead_wake.evaluate

    def _evaluate_private(*args: Any, **kwargs: Any) -> lead_wake.WakeDecision:
        if "owner_mode" not in kwargs:
            kwargs["owner_mode"] = "private"
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(lead_wake, "evaluate", _evaluate_private)


def _host(pid: int, name: str = "claude.exe") -> procinfo.HostResolution:
    entry = procinfo.ProcessInfo(pid=pid, ppid=1, name=name)
    return procinfo.HostResolution(chain=(entry,), host=entry)


class TestOwnerGate:
    def test_foreign_owner_short_circuits_before_all_session_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from claude_teams import server_simple

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise AssertionError

        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(22)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "b"
        )
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", _boom)
        monkeypatch.setattr(server_simple, "_active_session_id", _boom)
        monkeypatch.setattr(lead_wake, "_scan_senders", _boom)
        monkeypatch.setattr(lead_wake, "_write_guard", _boom)

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="a",
        )

        assert (result.action, result.code, result.log["why"]) == (
            "allow",
            "D0b",
            "not-owner",
        )

    @pytest.mark.parametrize("name", ["codex.exe", "node.exe"])
    def test_non_claude_nearest_host_is_owner_unknown_before_session_work(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        chain = (
            procinfo.ProcessInfo(1, 2, "python.exe"),
            procinfo.ProcessInfo(2, 3, name),
            procinfo.ProcessInfo(3, 0, "claude.exe"),
        )
        monkeypatch.setattr(
            lead_wake.procinfo,
            "resolve_nearest_host",
            lambda: procinfo.HostResolution(chain=chain, host=chain[1]),
        )
        monkeypatch.setattr(
            lead_wake.process_manager,
            "creation_token",
            lambda _pid: "token",
        )
        monkeypatch.setattr(
            lead_wake,
            "_resolve_session_dir",
            lambda _arg: (_ for _ in ()).throw(AssertionError("must not resolve")),
        )

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=2,
            owner_host_token="token",
        )

        assert (result.code, result.log["why"]) == ("D0b", "owner-unknown")

    def test_pid_match_token_mismatch_is_not_owner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(11)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "new"
        )

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="old",
        )

        assert result.code == "D0b"
        assert result.log["why"] == "not-owner"

    def test_matching_owner_reaches_session_resolution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(11)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "tok"
        )
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(lead_wake, "_live_subagent_names", lambda *_args: [])

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="tok",
        )

        assert result.code == "D2"

    def test_matching_owner_preserves_d3_and_d4_decisions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from claude_teams import server_simple

        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(11)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "tok"
        )
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda *_args: ["worker"]
        )
        _write_inbox(tmp_path, "team-lead", ["worker"])

        unread = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="tok",
        )
        assert (unread.code, unread.action) == ("D3", "block")

        (tmp_path / "inbox-team-lead.jsonl").unlink()
        armed = lead_wake.evaluate(
            _payload(
                background_tasks=[
                    {
                        "status": "running",
                        "command": server_simple._watch_command_bash(tmp_path),
                    }
                ]
            ),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="tok",
        )
        assert (armed.code, armed.action) == ("D4", "allow")

    def test_foreign_owner_main_writes_no_stdout_or_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(22)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "b"
        )
        monkeypatch.setattr(
            lead_wake,
            "_resolve_session_dir",
            lambda _arg: (_ for _ in ()).throw(AssertionError),
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload())))

        lead_wake.main(
            [
                "--owner-mode",
                "bound",
                "--owner-host-pid",
                "11",
                "--owner-host-token",
                "a",
            ]
        )

        assert capsys.readouterr().out == ""
        assert not (tmp_path / "wake-progress-team-lead.json").exists()

    @pytest.mark.parametrize(
        ("mode", "pid", "token"),
        [
            (None, None, None),
            ("bound", None, None),
            ("bound", 1, None),
            ("bound", None, "token"),
            ("private", 1, "token"),
            ("other", None, None),
        ],
    )
    def test_unknown_or_inconsistent_owner_allows_without_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str | None,
        pid: object,
        token: object,
    ) -> None:
        monkeypatch.setattr(
            lead_wake,
            "_resolve_session_dir",
            lambda _arg: (_ for _ in ()).throw(AssertionError("must not resolve")),
        )

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode=mode,
            owner_host_pid=pid,
            owner_host_token=token,
        )

        assert result.code == "D0b"
        assert result.log["why"] == "owner-unknown"

    @pytest.mark.parametrize(
        ("pid", "token"), [("x", "t"), (-1, "t"), (True, "t"), (1, "")]
    )
    def test_malformed_bound_values_allow(self, pid: object, token: object) -> None:
        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=pid,
            owner_host_token=token,
        )

        assert result.code == "D0b"
        assert result.log["why"] == "owner-unknown"

    @pytest.mark.parametrize("failure", [OSError("walk"), None])
    def test_walk_failure_or_disappearance_allows_unknown(
        self, monkeypatch: pytest.MonkeyPatch, failure: BaseException | None
    ) -> None:
        if failure is not None:

            def _raise() -> procinfo.HostResolution:
                raise failure

            monkeypatch.setattr(lead_wake.procinfo, "resolve_nearest_host", _raise)
        else:
            monkeypatch.setattr(
                lead_wake.procinfo,
                "resolve_nearest_host",
                lambda: procinfo.HostResolution(chain=(), host=None),
            )

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=1,
            owner_host_token="token",
        )

        assert result.code == "D0b"
        assert result.log["why"] == "owner-unknown"

    def test_owner_kill_switch_skips_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_OWNER", "0")
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(lead_wake, "_live_subagent_names", lambda *_args: [])

        result = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=-1,
            owner_host_token="",
        )

        assert result.code == "D2"

    def test_restart_sequence_old_owner_silent_new_owner_blocks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda *_args: ["worker"]
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "a"
        )
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(11)
        )

        owner_a = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="a",
        )

        assert owner_a.code == "D5"
        assert owner_a.action == "block"

        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(22)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "a2"
        )
        monkeypatch.setattr(
            lead_wake,
            "_resolve_session_dir",
            lambda _arg: (_ for _ in ()).throw(
                AssertionError("old binding must not touch disk")
            ),
        )
        restarted_with_old_binding = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=11,
            owner_host_token="a",
        )

        assert restarted_with_old_binding.code == "D0b"
        assert restarted_with_old_binding.log["why"] == "not-owner"

        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        reinstalled_for_a2 = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=22,
            owner_host_token="a2",
        )
        assert reinstalled_for_a2.code == "D5"
        assert reinstalled_for_a2.action == "block"

        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(11)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "a"
        )
        old_a_with_new_binding = lead_wake.evaluate(
            _payload(),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=22,
            owner_host_token="a2",
        )
        assert old_a_with_new_binding.code == "D0b"
        assert old_a_with_new_binding.log["why"] == "not-owner"


class TestKillSwitchAndFailOpen:
    def test_kill_switch_allows_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # D0: kill switch off short-circuits before any discovery runs.
        def _boom(_arg: object) -> Path:
            raise AssertionError  # discovery must not run when kill switch off

        monkeypatch.setattr(lead_wake, "_resolve_session_dir", _boom)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE", "0")

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D0"

    def test_wake_allows_when_no_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # D1: session not resolved -> fail-open allow.
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: None)

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D1"

    def test_wake_allows_when_no_live_subagents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # D2: session resolved but no live subagents -> allow.
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(lead_wake, "_live_subagent_names", lambda _sd, _id: [])

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"


class TestLiveSubagentScoping:
    """Fynd 1: the live-subagent check must scope to the caller's own children.

    Drives the real ``_live_subagent_names`` through ``evaluate`` by faking the
    session's agents.json via ``server_simple._load_agents``. A leaf worker (its
    own single record) must hit D2 (allow) and never arm a watcher for its own
    inbox; only an agent that actually leads live children proceeds past D2.
    """

    def _resolve_with_agents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        agents: list[dict],
    ) -> None:
        from claude_teams import server_simple

        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(server_simple, "_load_agents", lambda _sid: agents)
        monkeypatch.setattr(server_simple, "_agent_alive", lambda _agent: True)

    def test_leaf_agent_only_self_record_allows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The exact observed bug: worker1 is the ONLY record (parent unset) and
        # must not treat itself as a live subagent -> D2 allow, no self-arm.
        monkeypatch.setenv("AGENT_NAME", "worker1")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [{"name": "worker1", "status": "running", "parent": None}],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"
        assert result.log["live_subagents"] == 0

    def test_sibling_is_not_counted_as_child(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # worker1 and worker2 share parent "team-lead"; a sibling is not a
        # child, so worker1 still allows at D2.
        monkeypatch.setenv("AGENT_NAME", "worker1")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [
                {"name": "worker1", "status": "running", "parent": "team-lead"},
                {"name": "worker2", "status": "running", "parent": "team-lead"},
            ],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"
        assert result.log["live_subagents"] == 0

    def test_real_live_child_proceeds_past_d2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # "mid" leads a live child "leaf" (parent == mid) -> NOT D2; with no
        # unread and no armed watcher it lands on the D5 arm instruction.
        monkeypatch.setenv("AGENT_NAME", "mid")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [
                {"name": "mid", "status": "running", "parent": "team-lead"},
                {"name": "leaf", "status": "running", "parent": "mid"},
            ],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.code != "D2"
        assert result.action == "block"
        assert result.code == "D5"
        assert result.log["live_subagents"] == 1

    def test_terminal_child_is_not_live(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # "mid"'s only child is killed -> no live children -> D2 allow.
        monkeypatch.setenv("AGENT_NAME", "mid")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [
                {"name": "mid", "status": "running", "parent": "team-lead"},
                {"name": "leaf", "status": "killed", "parent": "mid"},
            ],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"

    def test_running_child_with_dead_process_is_not_live(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from claude_teams import server_simple

        monkeypatch.setenv("AGENT_NAME", "mid")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [{"name": "leaf", "status": "running", "parent": "mid", "pid": 1}],
        )
        monkeypatch.setattr(server_simple, "_agent_alive", lambda _agent: False)

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"

    def test_liveness_probe_error_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from claude_teams import server_simple

        monkeypatch.setenv("AGENT_NAME", "mid")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [{"name": "leaf", "status": "running", "parent": "mid", "pid": 1}],
        )

        def _raise(_agent: dict) -> bool:
            raise OSError

        monkeypatch.setattr(server_simple, "_agent_alive", _raise)

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"

    def test_left_external_member_is_not_live(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A member that called leave_team (status "left") has permanently
        # departed and its inbox is drained, so the lead has nothing to wait
        # for -> D2 allow. "left" is not a _TERMINAL_STATUSES value, so this
        # guards the wake family against nagging forever after a leave_team.
        monkeypatch.setenv("AGENT_NAME", "team-lead")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [
                {
                    "name": "qa",
                    "status": "left",
                    "parent": "team-lead",
                    "backend": "external",
                },
            ],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"

    def test_legacy_records_without_parent_still_exclude_self(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Pre-fix session: no record carries a parent field. The fallback still
        # excludes self, so a legacy leaf (only its own record) allows at D2
        # rather than regressing into the self-count bug.
        monkeypatch.setenv("AGENT_NAME", "worker1")
        self._resolve_with_agents(
            monkeypatch,
            tmp_path,
            [{"name": "worker1", "status": "running"}],
        )

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D2"


def _watch_bg_task(session_dir: Path, *, status: str = "running") -> dict:
    """A background_tasks entry mirroring a real ``watch`` invocation."""
    from claude_teams import server_simple

    return {
        "id": "bgx",
        "type": "shell",
        "status": status,
        "description": "Start background inbox watcher",
        "command": server_simple._watch_command_bash(session_dir),
    }


class TestDecisionCore:
    @pytest.fixture(autouse=True)
    def _resolved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda _sd, _id: ["worker"]
        )

    def test_wake_blocks_read_messages_when_unread_present(
        self, tmp_path: Path
    ) -> None:
        # D3: live subagents + unread from alice -> block naming alice.
        _write_inbox(tmp_path, "team-lead", ["alice"])

        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "block"
        assert result.code == "D3"
        assert result.reason is not None  # a block always carries a reason
        assert "alice" in result.reason
        assert "read_messages" in result.reason

    def test_wake_allows_when_armed_bg_task_matches(self, tmp_path: Path) -> None:
        # D4: no unread + a running watcher for THIS session -> allow.
        payload = _payload(background_tasks=[_watch_bg_task(tmp_path)])

        result = lead_wake.evaluate(payload, reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D4"

    def test_wake_allows_when_armed_bg_task_is_unbound(self, tmp_path: Path) -> None:
        # D4: the hook-suggested (owner-unbound) command, once running, is
        # recognised as armed — the re-arm loop closes on what D5 emits.
        from claude_teams import server_simple

        payload = _payload(
            background_tasks=[
                {
                    "status": "running",
                    "command": server_simple._watch_command_bash(
                        tmp_path, bind_owner=False
                    ),
                }
            ]
        )

        result = lead_wake.evaluate(payload, reader_arg="team-lead")

        assert result.action == "allow"
        assert result.code == "D4"

    def test_wake_blocks_arm_instruction_when_not_armed_no_unread(
        self, tmp_path: Path
    ) -> None:
        # D5: no unread + no watcher -> block with the operational arm text.
        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "block"
        assert result.code == "D5"
        assert result.reason is not None  # a block always carries a reason
        # The rendered real watch command is embedded (persistent-false-negative
        # guard: the model must be able to run exactly what the match recognises).
        assert "claude_teams.cli" in result.reason
        assert "watch" in result.reason
        assert str(tmp_path) in result.reason
        assert "background" in result.reason.lower()
        # Hook-emitted commands are deliberately unbound: the hook's parent is a
        # transient wrapper, so a baked-in owner PID would die instantly (exit 4).
        assert "--owner-pid" not in result.reason
        assert "--owner-token" not in result.reason
        # Operational, not an imperative token demand (spike probe a).
        assert "must now output" not in result.reason.lower()


def _write_guard(
    session_dir: Path,
    reader: str,
    senders: dict[str, dict[str, int]],
    noprogress: int,
) -> None:
    (session_dir / f"wake-progress-{reader}.json").write_text(
        json.dumps(
            {
                "schema": "lead-wake-progress/1",
                "reader": reader,
                "senders": senders,
                "noprogress_blocks": noprogress,
                "ts": 0.0,
            }
        ),
        encoding="utf-8",
    )


def _read_guard(session_dir: Path, reader: str) -> dict:
    return json.loads(
        (session_dir / f"wake-progress-{reader}.json").read_text(encoding="utf-8")
    )


class TestProgressGuard:
    @pytest.fixture(autouse=True)
    def _resolved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda _sd, _id: ["worker"]
        )

    def test_wake_progress_guard_fail_open_after_cap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # D6: stop_hook_active true, no cursor advance, count reaches cap -> allow.
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "3")
        _write_guard(tmp_path, "team-lead", {}, 2)  # next no-progress block -> 3

        result = lead_wake.evaluate(
            _payload(stop_hook_active=True), reader_arg="team-lead"
        )

        assert result.action == "allow"
        assert result.code == "D6"

    def test_wake_progress_guard_resets_after_productive_wake(
        self, tmp_path: Path
    ) -> None:
        # Prior snapshot: alice cursor=0. Now alice cursor=1 (read one) but a NEW
        # message arrived, so unread STAYS 1 while the CURSOR advanced 0->1. A
        # cursor-keyed guard resets and the block PROCEEDS (productive wake must
        # not be shortened, F3); an unread-keyed guard would wrongly see
        # no-progress. This is the regression the plan's F1 fix targets.
        _write_inbox(tmp_path, "team-lead", ["alice", "alice"])  # total 2
        _write_cursors(tmp_path, "team-lead", {"alice": 1})  # unread 1
        _write_guard(tmp_path, "team-lead", {"alice": {"total": 1, "cursor": 0}}, 2)

        result = lead_wake.evaluate(
            _payload(stop_hook_active=True), reader_arg="team-lead"
        )

        assert result.action == "block"  # NOT fail-open
        assert result.code == "D3"
        guard = _read_guard(tmp_path, "team-lead")
        assert guard["noprogress_blocks"] == 0
        assert guard["senders"]["alice"]["cursor"] == 1

    def test_wake_guard_does_not_fail_open_without_stop_hook_active(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # stop_hook_active gates consulting the guard: a first Stop (false) never
        # fail-opens even with a high stored count; it re-blocks (D5).
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "3")
        _write_guard(tmp_path, "team-lead", {}, 5)

        result = lead_wake.evaluate(
            _payload(stop_hook_active=False), reader_arg="team-lead"
        )

        assert result.action == "block"
        assert result.code == "D5"

    def test_bound_owner_change_resets_old_guard_generation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "3")
        _write_guard(tmp_path, "team-lead", {}, 2)  # old schema has no owner
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(7)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "new"
        )

        result = lead_wake.evaluate(
            _payload(stop_hook_active=True),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=7,
            owner_host_token="new",
        )

        assert result.code == "D5"
        guard = _read_guard(tmp_path, "team-lead")
        assert guard["noprogress_blocks"] == 1
        assert guard["owner_generation"] == "7:new"

    def test_bound_owner_change_resets_different_owner_generation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS", "3")
        prior = {
            "schema": "lead-wake-progress/1",
            "reader": "team-lead",
            "senders": {},
            "noprogress_blocks": 2,
            "ts": 0.0,
            "owner_generation": "6:old",
        }
        (tmp_path / "wake-progress-team-lead.json").write_text(
            json.dumps(prior), encoding="utf-8"
        )
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(7)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "new"
        )

        result = lead_wake.evaluate(
            _payload(stop_hook_active=True),
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=7,
            owner_host_token="new",
        )

        assert result.code == "D5"
        guard = _read_guard(tmp_path, "team-lead")
        assert guard["noprogress_blocks"] == 1
        assert guard["owner_generation"] == "7:new"


class TestIdentityAndArming:
    def test_wake_nested_lead_uses_agent_name_inbox(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # AGENT_NAME=mid must scan inbox-mid.jsonl, NOT inbox-team-lead.jsonl
        # (regression vs the Pi team-lead-only bug; FR6/AC3).
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda _sd, _id: ["worker"]
        )
        monkeypatch.setenv("AGENT_NAME", "mid")
        _write_inbox(tmp_path, "mid", ["bob"])
        _write_inbox(tmp_path, "team-lead", ["zoe"])

        # reader_arg is the baked default; AGENT_NAME is authoritative.
        result = lead_wake.evaluate(_payload(), reader_arg="team-lead")

        assert result.action == "block"
        assert result.code == "D3"
        assert result.reason is not None  # a block always carries a reason
        assert "bob" in result.reason
        assert "zoe" not in result.reason

    def test_arming_match_is_separator_insensitive_and_session_scoped(
        self, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "sess-abc"
        session_dir.mkdir()
        # Backslash-separated command referencing THIS session (by id) matches.
        win_cmd = (
            "python -m claude_teams.cli watch "
            + f"C:\\Users\\x\\.claude\\agent-sessions\\{session_dir.name}"
        )
        assert lead_wake._command_matches_session(win_cmd, session_dir) is True
        # The unbound rendering the Stop hook now emits still matches.
        from claude_teams import server_simple

        assert (
            lead_wake._command_matches_session(
                server_simple._watch_command_bash(session_dir, bind_owner=False),
                session_dir,
            )
            is True
        )
        # A watch command for a DIFFERENT session does NOT match.
        other = "python -m claude_teams.cli watch /home/x/sessions/other-999"
        assert lead_wake._command_matches_session(other, session_dir) is False
        # A non-watch command never matches even for this session.
        assert (
            lead_wake._command_matches_session(f"echo {session_dir}", session_dir)
            is False
        )

        # _is_armed honours running status + the match.
        assert (
            lead_wake._is_armed(
                _payload(background_tasks=[{"status": "running", "command": win_cmd}]),
                session_dir,
            )
            is True
        )
        # A non-running (e.g. completed) task is not armed.
        assert (
            lead_wake._is_armed(
                _payload(
                    background_tasks=[{"status": "completed", "command": win_cmd}]
                ),
                session_dir,
            )
            is False
        )
        # A watcher for a different session leaves this session un-armed.
        assert (
            lead_wake._is_armed(
                _payload(background_tasks=[{"status": "running", "command": other}]),
                session_dir,
            )
            is False
        )


class TestMainEntrypoint:
    def test_stdout_flush_failure_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flush_error = OSError("broken pipe")

        class FlushFails(io.StringIO):
            def flush(self) -> None:
                raise flush_error

        stream = FlushFails()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(
            lead_wake,
            "evaluate",
            lambda *_args, **_kwargs: lead_wake.WakeDecision(
                "block", "D5", reason="block"
            ),
        )

        lead_wake.main([])

        assert '"decision": "block"' in stream.getvalue()

    @pytest.mark.parametrize("argv", [["--unknown"], ["--owner-mode"]])
    def test_malformed_argv_subprocess_exits_zero_without_block_json(
        self, argv: list[str]
    ) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "claude_teams.lead_wake", *argv],
            input="{}",
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

        assert completed.returncode == 0
        assert '"decision": "block"' not in completed.stdout

    def test_evaluation_failure_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            lead_wake,
            "evaluate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        lead_wake.main([])

        assert '"decision": "block"' not in capsys.readouterr().out

    @pytest.mark.parametrize(
        "program",
        [
            (
                "import claude_teams.lead_wake as m;"
                "m.evaluate=lambda *a,**k:(_ for _ in ()).throw(RuntimeError());"
                "m.main([])"
            ),
            (
                "import claude_teams.lead_wake as m;"
                "m._log_line=lambda *a:(_ for _ in ()).throw(RuntimeError());"
                "m.main([])"
            ),
            "import claude_teams.lead_wake as m;m.sys.stderr=None;m.main([])",
            (
                "import claude_teams.lead_wake as m;m.sys.stdout=None;"
                "m.evaluate=lambda *a,**k:m.WakeDecision('block','D5',reason='x');"
                "m.main([])"
            ),
        ],
    )
    def test_entrypoint_internal_failures_exit_zero_in_subprocess(
        self, program: str
    ) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            input="{}",
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

        assert completed.returncode == 0
        assert '"decision": "block"' not in completed.stdout

    def test_broken_stderr_and_stdout_are_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream_failure = OSError("broken stream")

        class Broken:
            def write(self, _value: str) -> int:
                raise stream_failure

        monkeypatch.setattr(
            lead_wake,
            "evaluate",
            lambda *_args, **_kwargs: lead_wake.WakeDecision(
                "block", "D5", reason="block"
            ),
        )
        monkeypatch.setattr(sys, "stderr", Broken())
        monkeypatch.setattr(sys, "stdout", Broken())

        lead_wake.main([])

    def test_main_block_prints_decision_and_logs_to_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # D5 block via the real stdin/stdout entrypoint (mirrors test_hooks
        # io.StringIO faking). stdout carries the decision; stderr the log line.
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda _sd, _id: ["worker"]
        )
        payload = json.dumps(_payload())
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        lead_wake.main(
            [
                "--session-dir",
                str(tmp_path),
                "--reader",
                "team-lead",
                "--owner-mode",
                "private",
            ]
        )

        out = capsys.readouterr()
        decision = json.loads(out.out)
        assert decision["decision"] == "block"
        assert "claude_teams.cli" in decision["reason"]
        # Structured, body-free log on stderr; no message text leaks.
        assert "win-agent-teams/lead-wake" in out.err
        assert "hi from" not in out.err

    def test_main_allow_prints_nothing_to_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # D2 allow: no live subagents -> no stdout decision, still a stderr log.
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: tmp_path)
        monkeypatch.setattr(lead_wake, "_live_subagent_names", lambda _sd, _id: [])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload())))

        lead_wake.main(
            [
                "--session-dir",
                str(tmp_path),
                "--reader",
                "team-lead",
                "--owner-mode",
                "private",
            ]
        )

        out = capsys.readouterr()
        assert out.out == ""
        assert "win-agent-teams/lead-wake" in out.err

    def test_main_corrupt_stdin_fails_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Corrupt stdin must not raise and must not block (fail-open).
        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: None)
        monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))

        lead_wake.main(["--reader", "team-lead", "--owner-mode", "private"])

        assert capsys.readouterr().out == ""

"""Tests for the `win-agent-teams watch` CLI helper (item 5)."""

import json
import threading
import time
from pathlib import Path

import typer.main
from typer.testing import CliRunner

from claude_teams import cli
from claude_teams.backends.process_manager import (
    OWNERSHIP_NOT_OURS,
    OWNERSHIP_OURS,
)
from claude_teams.cli import app

runner = CliRunner()


def test_watch_default_timeout_is_1800_seconds() -> None:
    """Metadata-only: never invoke `watch` without an explicit timeout here."""
    command = typer.main.get_command(app)
    watch_cmd = command.commands["watch"]
    timeout_opt = next(
        p for p in watch_cmd.params if "--timeout" in getattr(p, "opts", [])
    )

    assert timeout_opt.default == 1800.0


def test_watch_exits_2_on_timeout_with_no_change(tmp_path: Path) -> None:
    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 2


def test_watch_exits_4_when_bound_owner_is_gone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli.process_manager,
        "ownership_probe",
        lambda handle, token: OWNERSHIP_NOT_OURS,
    )

    result = runner.invoke(
        app,
        [
            "watch",
            str(tmp_path),
            "--owner-pid",
            "123",
            "--owner-token",
            "born-1",
        ],
    )

    assert result.exit_code == 4


def test_watch_keeps_running_while_bound_owner_matches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        cli.process_manager,
        "ownership_probe",
        lambda handle, token: OWNERSHIP_OURS,
    )

    result = runner.invoke(
        app,
        [
            "watch",
            str(tmp_path),
            "--timeout",
            "0",
            "--owner-pid",
            "123",
            "--owner-token",
            "born-1",
        ],
    )

    assert result.exit_code == 2


def test_watch_rejects_partial_owner_binding(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["watch", str(tmp_path), "--owner-pid", "123", "--timeout", "0"],
    )

    assert result.exit_code == 1
    assert "must be supplied together" in result.stderr


def test_watch_exits_0_and_prints_path_when_file_created(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"

    def _create_after_delay() -> None:
        time.sleep(0.3)
        target.write_text('{"state": "waiting", "event": "Stop", "ts": 1.0}')

    thread = threading.Thread(target=_create_after_delay)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "5"])
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_exits_0_and_prints_path_when_file_mtime_changes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"
    target.write_text('{"state": "running", "event": "Start", "ts": 1.0}')

    def _touch_after_delay() -> None:
        time.sleep(0.3)
        # Ensure the mtime actually advances on filesystems with coarse
        # resolution by bumping it explicitly rather than relying on wall time.
        new_mtime = target.stat().st_mtime + 5
        target.write_text('{"state": "waiting", "event": "Stop", "ts": 2.0}')
        import os

        os.utime(target, (new_mtime, new_mtime))

    thread = threading.Thread(target=_touch_after_delay)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "5"])
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_detects_same_mtime_rewrite(tmp_path: Path, monkeypatch) -> None:
    import os

    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"
    target.write_text('{"state": "running", "event": "Start", "ts": 1.0}')
    original_mtime = target.stat().st_mtime

    def _rewrite_same_mtime_after_delay() -> None:
        time.sleep(0.3)
        # Rewrite with different (larger) content but pin the mtime back to
        # its original value, simulating an atomic-replace/rewrite that
        # preserves the second-resolution mtime.
        target.write_text(
            '{"state": "waiting", "event": "Stop", "ts": 2.0, "extra": "padding"}'
        )
        os.utime(target, (original_mtime, original_mtime))

    thread = threading.Thread(target=_rewrite_same_mtime_after_delay)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "5"])
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_respects_custom_pattern(tmp_path: Path) -> None:
    target = tmp_path / "output-report.md"

    def _create_after_delay() -> None:
        time.sleep(0.3)
        target.write_text("done")

    thread = threading.Thread(target=_create_after_delay)
    thread.start()
    try:
        result = runner.invoke(
            app,
            [
                "watch",
                str(tmp_path),
                "--timeout",
                "5",
                "--pattern",
                "output-*.md",
            ],
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "output-report.md" in result.stdout


def test_watch_default_pattern_ignores_non_matching_files(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("noise")

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 2


def test_watch_ignores_running_transitions_until_waiting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"

    def _transition() -> None:
        time.sleep(0.06)
        target.write_text('{"state":"running","event":"PreToolUse"}')
        time.sleep(0.08)
        target.write_text('{"state":"running","event":"PostToolUse"}')
        time.sleep(0.08)
        target.write_text('{"state":"waiting","event":"Stop"}')

    thread = threading.Thread(target=_transition)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "waiting"
    assert wake["agent"] == "worker"
    assert wake["path"].endswith("state-worker.json")


def test_watch_running_transition_alone_times_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    target = tmp_path / "state-worker.json"

    def _write_running() -> None:
        time.sleep(0.05)
        target.write_text('{"state":"running","event":"PreToolUse"}')

    thread = threading.Thread(target=_write_running)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "0.3"])
    finally:
        thread.join()

    assert result.exit_code == 2
    assert result.stdout == ""


def test_watch_preexisting_waiting_marker_is_not_a_new_edge(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    (tmp_path / "state-worker.json").write_text('{"state":"waiting","event":"Stop"}')

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "0.15"])

    assert result.exit_code == 2
    assert result.stdout == ""


def test_watch_wakes_for_preexisting_unread_message(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "done"}) + "\n")

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    wake = json.loads(lines[0])
    assert wake == {
        "reason": "message",
        "from": ["worker"],
        "path": str(inbox),
    }


def test_watch_does_not_wake_for_consumed_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    inbox = tmp_path / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "done"}) + "\n")
    (tmp_path / "inbox-team-lead.pos.json").write_text('{"worker":1}')

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "0.15"])

    assert result.exit_code == 2
    assert result.stdout == ""


def test_watch_uses_nested_agent_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_NAME", "parent-agent")
    wrong_inbox = tmp_path / "inbox-team-lead.jsonl"
    wrong_inbox.write_text(json.dumps({"from": "wrong", "text": "ignore"}) + "\n")
    expected_inbox = tmp_path / "inbox-parent-agent.jsonl"
    expected_inbox.write_text(
        json.dumps({"from": "child", "text": "wake parent"}) + "\n"
    )

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"
    assert wake["from"] == ["child"]
    assert wake["path"] == str(expected_inbox)


def test_watch_detects_completed_append_after_partial_line(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    inbox = tmp_path / "inbox-team-lead.jsonl"

    def _append_in_two_stages() -> None:
        time.sleep(0.06)
        inbox.write_text('{"from":"worker"')
        time.sleep(0.08)
        with inbox.open("a", encoding="utf-8") as handle:
            handle.write(',"text":"done"}\n')

    thread = threading.Thread(target=_append_in_two_stages)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"
    assert wake["from"] == ["worker"]


def test_watch_message_reason_wins_over_waiting(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.1)
    marker = tmp_path / "state-worker.json"
    inbox = tmp_path / "inbox-team-lead.jsonl"

    def _make_both_ready() -> None:
        time.sleep(0.03)
        marker.write_text('{"state":"waiting","event":"Stop"}')
        inbox.write_text(json.dumps({"from": "worker", "text": "done"}) + "\n")

    thread = threading.Thread(target=_make_both_ready)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"


def test_watch_custom_state_pattern_remains_semantic(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    marker = tmp_path / "state-worker.json"

    def _write_running() -> None:
        time.sleep(0.05)
        marker.write_text('{"state":"running","event":"PreToolUse"}')

    thread = threading.Thread(target=_write_running)
    thread.start()
    try:
        result = runner.invoke(
            app,
            [
                "watch",
                str(tmp_path),
                "--timeout",
                "0.25",
                "--pattern",
                "state-worker.json",
            ],
        )
    finally:
        thread.join()

    assert result.exit_code == 2


def test_watch_message_wakes_during_running_marker_churn(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.05)
    marker = tmp_path / "state-worker.json"
    inbox = tmp_path / "inbox-team-lead.jsonl"

    def _write_running_and_message() -> None:
        time.sleep(0.03)
        marker.write_text('{"state":"running","event":"PreToolUse"}')
        inbox.write_text(json.dumps({"from": "worker", "text": "progress"}) + "\n")

    thread = threading.Thread(target=_write_running_and_message)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"
    assert wake["from"] == ["worker"]


def test_watch_custom_pattern_keeps_inbox_enabled_by_default(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "wake"}) + "\n")

    result = runner.invoke(
        app,
        ["watch", str(tmp_path), "--timeout", "1", "--pattern", "report.md"],
    )

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"


def test_watch_treats_corrupt_cursor_as_unread(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "wake"}) + "\n")
    (tmp_path / "inbox-team-lead.pos.json").write_text("{broken")

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "message"
    assert wake["from"] == ["worker"]


def test_watch_ignores_subagent_stop_waiting(tmp_path: Path, monkeypatch) -> None:
    """A SubagentStop waiting marker is an agent's own Task subagent finishing;
    the agent is still working, so it must not wake the coordinator."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"

    def _write_subagent_stop() -> None:
        time.sleep(0.05)
        target.write_text('{"state":"waiting","event":"SubagentStop"}')

    thread = threading.Thread(target=_write_subagent_stop)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "0.4"])
    finally:
        thread.join()

    assert result.exit_code == 2
    assert result.stdout == ""


def test_watch_wakes_on_stop_after_subagent_stop(tmp_path: Path, monkeypatch) -> None:
    """After an ignored SubagentStop, the agent's real end-of-turn Stop still
    wakes the coordinator."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    target = tmp_path / "state-worker.json"

    def _sequence() -> None:
        time.sleep(0.05)
        target.write_text('{"state":"waiting","event":"SubagentStop"}')
        time.sleep(0.08)
        target.write_text('{"state":"waiting","event":"Stop"}')

    thread = threading.Thread(target=_sequence)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "waiting"
    assert wake["agent"] == "worker"


def test_watch_settle_suppresses_transient_waiting(tmp_path: Path, monkeypatch) -> None:
    """A waiting marker that flips back to running within the settle window is
    churn (agent parked briefly, then resumed) and must not wake."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.3)
    target = tmp_path / "state-worker.json"

    def _flap() -> None:
        time.sleep(0.05)
        target.write_text('{"state":"waiting","event":"Stop"}')
        time.sleep(0.1)
        target.write_text('{"state":"running","event":"PreToolUse"}')

    thread = threading.Thread(target=_flap)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "0.7"])
    finally:
        thread.join()

    assert result.exit_code == 2
    assert result.stdout == ""


def test_watch_settle_wakes_persistent_waiting(tmp_path: Path, monkeypatch) -> None:
    """A waiting marker that stays waiting past the settle window wakes the
    coordinator (genuine end-of-task)."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.2)
    target = tmp_path / "state-worker.json"

    def _write_waiting() -> None:
        time.sleep(0.05)
        target.write_text('{"state":"waiting","event":"Stop"}')

    thread = threading.Thread(target=_write_waiting)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "2"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "waiting"
    assert wake["agent"] == "worker"


def test_watch_settles_overlapping_waits_independently(
    tmp_path: Path, monkeypatch
) -> None:
    """A persistent waiting marker must still wake even when a later, transient
    waiting marker arrives and then resumes. A single-candidate tracker would
    overwrite (and lose) the persistent one."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.2)
    marker_a = tmp_path / "state-worker-a.json"
    marker_b = tmp_path / "state-worker-b.json"

    def _sequence() -> None:
        time.sleep(0.05)
        marker_a.write_text('{"state":"waiting","event":"Stop"}')  # persistent
        time.sleep(0.05)
        marker_b.write_text('{"state":"waiting","event":"Stop"}')  # transient
        time.sleep(0.06)
        marker_b.write_text('{"state":"running","event":"PreToolUse"}')

    thread = threading.Thread(target=_sequence)
    thread.start()
    try:
        result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1.5"])
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "waiting"
    assert wake["agent"] == "worker-a"


def test_watch_output_not_starved_by_settling_wait(tmp_path: Path, monkeypatch) -> None:
    """When an output lands in the same poll a pending wait matures, the output
    must win — otherwise `before = after` consumes the output edge and the next
    invocation baselines it, so it is never reported (message > output > waiting)."""
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.2)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.1)
    marker = tmp_path / "state-worker.json"
    output = tmp_path / "report.md"

    def _sequence() -> None:
        time.sleep(0.05)
        marker.write_text('{"state":"waiting","event":"Stop"}')  # matures ~0.3
        time.sleep(0.2)  # ~0.25: output lands after maturation, same poll as wake
        output.write_text("done")

    thread = threading.Thread(target=_sequence)
    thread.start()
    try:
        result = runner.invoke(
            app, ["watch", str(tmp_path), "--timeout", "2", "--pattern", "*"]
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake["reason"] == "output"
    assert wake["path"] == str(output)


def test_waiting_agent_tolerates_unhashable_event(tmp_path: Path) -> None:
    """A marker whose JSON ``event`` is a non-string (valid JSON, unhashable)
    must not crash the membership check; it is treated as actionable."""
    marker = tmp_path / "state-worker.json"
    marker.write_text('{"state":"waiting","event":[]}')

    assert cli._waiting_agent(marker) == "worker"


def test_settle_seconds_from_env_rejects_bad_values(monkeypatch) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS", raising=False)
    assert cli._settle_seconds_from_env() == 15.0

    for bad in ("abc", "nan", "-1", "inf"):
        monkeypatch.setenv("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS", bad)
        assert cli._settle_seconds_from_env() == 15.0

    monkeypatch.setenv("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS", "0")
    assert cli._settle_seconds_from_env() == 0.0
    monkeypatch.setenv("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS", "3.5")
    assert cli._settle_seconds_from_env() == 3.5


def test_watch_no_inbox_preserves_artifact_only_behavior(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    inbox = tmp_path / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "ignore"}) + "\n")
    output = tmp_path / "report.md"

    def _write_output() -> None:
        time.sleep(0.07)
        output.write_text("done")

    thread = threading.Thread(target=_write_output)
    thread.start()
    try:
        result = runner.invoke(
            app,
            [
                "watch",
                str(tmp_path),
                "--timeout",
                "2",
                "--pattern",
                "report.md",
                "--no-inbox",
            ],
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake == {"reason": "output", "path": str(output)}


# ---------------------------------------------------------------------------
# T3b — `watch --reader` overrides the env-based reader.
# ---------------------------------------------------------------------------


def test_watch_reader_flag_overrides_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_NAME", "other")
    # The env identity's inbox has traffic that must be IGNORED when --reader wins.
    other_inbox = tmp_path / "inbox-other.jsonl"
    other_inbox.write_text(json.dumps({"from": "noise", "text": "ignore"}) + "\n")
    lead_inbox = tmp_path / "inbox-team-lead.jsonl"
    lead_inbox.write_text(json.dumps({"from": "worker", "text": "wake lead"}) + "\n")

    result = runner.invoke(
        app, ["watch", str(tmp_path), "--timeout", "1", "--reader", "team-lead"]
    )

    assert result.exit_code == 0
    wake = json.loads(result.stdout)
    assert wake == {
        "reason": "message",
        "from": ["worker"],
        "path": str(lead_inbox),
    }


# ---------------------------------------------------------------------------
# T2b — `inbox-status` non-consuming generation probe.
# ---------------------------------------------------------------------------


def _session_base(tmp_path: Path, monkeypatch) -> Path:
    from claude_teams import server_simple as ss

    base = tmp_path / "sessions"
    base.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    return base


def test_inbox_status_reports_schema_per_sender(tmp_path: Path, monkeypatch) -> None:
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "abc"
    sdir.mkdir()
    inbox = sdir / "inbox-team-lead.jsonl"
    inbox.write_text(
        json.dumps({"from": "worker", "text": "a"})
        + "\n"
        + json.dumps({"from": "worker", "text": "b"})
        + "\n"
    )
    (sdir / "inbox-team-lead.pos.json").write_text('{"worker":1}')

    result = runner.invoke(app, ["inbox-status", str(sdir)])

    assert result.exit_code == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "schema": "inbox-status/1",
        "reader": "team-lead",
        "senders": {"worker": {"total": 2, "cursor": 1, "unread": 1}},
    }


def test_inbox_status_empty_inbox_emits_empty_senders(
    tmp_path: Path, monkeypatch
) -> None:
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "def"
    sdir.mkdir()

    result = runner.invoke(app, ["inbox-status", str(sdir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "schema": "inbox-status/1",
        "reader": "team-lead",
        "senders": {},
    }


def test_inbox_status_is_non_consuming(tmp_path: Path, monkeypatch) -> None:
    from claude_teams.messaging import unread_sender_counts

    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "ghi"
    sdir.mkdir()
    inbox = sdir / "inbox-team-lead.jsonl"
    inbox.write_text(json.dumps({"from": "worker", "text": "a"}) + "\n")
    cursor = sdir / "inbox-team-lead.pos.json"

    result = runner.invoke(app, ["inbox-status", str(sdir)])
    assert result.exit_code == 0

    # No cursor was written, and the unread view is unchanged afterwards.
    assert not cursor.exists()
    assert unread_sender_counts(inbox, cursor) == {"worker": 1}


def test_inbox_status_respects_reader_flag(tmp_path: Path, monkeypatch) -> None:
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "jkl"
    sdir.mkdir()
    (sdir / "inbox-team-lead.jsonl").write_text(
        json.dumps({"from": "noise", "text": "ignore"}) + "\n"
    )
    (sdir / "inbox-reviewer.jsonl").write_text(
        json.dumps({"from": "worker", "text": "x"}) + "\n"
    )

    result = runner.invoke(app, ["inbox-status", str(sdir), "--reader", "reviewer"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reader"] == "reviewer"
    assert payload["senders"] == {"worker": {"total": 1, "cursor": 0, "unread": 1}}


def test_inbox_status_uses_nested_agent_identity(tmp_path: Path, monkeypatch) -> None:
    """Without --reader, inbox-status resolves the reader from AGENT_NAME.

    This pins the cross-language contract the Pi wake extension relies on: it
    calls ``inbox-status`` WITHOUT ``--reader`` for a nested lead, so the CLI
    must apply the same ambient default as ``watch`` (AGENT_NAME, else
    team-lead) rather than hardcoding team-lead.
    """
    monkeypatch.setenv("AGENT_NAME", "worker-1")
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "nested"
    sdir.mkdir()
    # team-lead's inbox has traffic that MUST NOT be reported for a nested lead.
    (sdir / "inbox-team-lead.jsonl").write_text(
        json.dumps({"from": "noise", "text": "ignore"}) + "\n"
    )
    (sdir / "inbox-worker-1.jsonl").write_text(
        json.dumps({"from": "boss", "text": "x"}) + "\n"
    )

    result = runner.invoke(app, ["inbox-status", str(sdir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reader"] == "worker-1"
    assert payload["senders"] == {"boss": {"total": 1, "cursor": 0, "unread": 1}}


def test_inbox_status_defaults_to_team_lead_without_agent_name(
    tmp_path: Path, monkeypatch
) -> None:
    """With no AGENT_NAME and no --reader, the reader falls back to team-lead."""
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "root"
    sdir.mkdir()
    (sdir / "inbox-team-lead.jsonl").write_text(
        json.dumps({"from": "worker", "text": "x"}) + "\n"
    )

    result = runner.invoke(app, ["inbox-status", str(sdir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reader"] == "team-lead"
    assert payload["senders"] == {"worker": {"total": 1, "cursor": 0, "unread": 1}}


def test_inbox_status_outside_base_exits_4(tmp_path: Path, monkeypatch) -> None:
    _session_base(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    result = runner.invoke(app, ["inbox-status", str(outside)])

    assert result.exit_code == 4
    assert result.stdout == ""
    assert result.stderr != ""


def test_inbox_status_nonexistent_dir_exits_4(tmp_path: Path, monkeypatch) -> None:
    base = _session_base(tmp_path, monkeypatch)

    result = runner.invoke(app, ["inbox-status", str(base / "missing")])

    assert result.exit_code == 4
    assert result.stdout == ""
    assert result.stderr != ""


# ---------------------------------------------------------------------------
# Unsafe --reader validation exits 1 for BOTH inbox-status and watch.
# ---------------------------------------------------------------------------


def test_inbox_status_unsafe_reader_exits_1(tmp_path: Path, monkeypatch) -> None:
    base = _session_base(tmp_path, monkeypatch)
    sdir = base / "unsafe"
    sdir.mkdir()

    result = runner.invoke(app, ["inbox-status", str(sdir), "--reader", "../evil"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr != ""


def test_watch_unsafe_reader_exits_1(tmp_path: Path) -> None:
    # A reader carrying a path separator must be rejected before any watching,
    # so it can never interpolate into inbox-<reader>.jsonl / .pos.json paths.
    result = runner.invoke(
        app, ["watch", str(tmp_path), "--timeout", "1", "--reader", "../evil"]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr != ""


def test_inbox_status_probe_matches_following_consuming_read_messages(
    tmp_path: Path, monkeypatch
) -> None:
    """The non-consuming probe reports exactly what a real read_messages drains.

    Stronger T2b: after the probe, drive the actual consuming
    ``server_simple.read_messages`` and assert it returns the same messages the
    probe reported unread, and that only THEN is the generation consumed.
    """
    import asyncio

    from claude_teams import server_simple as ss

    base = _session_base(tmp_path, monkeypatch)
    session_id = "read-through"
    sdir = base / session_id
    sdir.mkdir()
    monkeypatch.setattr(ss, "_session_id", session_id)
    monkeypatch.setattr(ss, "IDENTITY", "team-lead")
    monkeypatch.setattr(ss, "_inbox_locks", {})

    inbox = sdir / "inbox-team-lead.jsonl"
    inbox.write_text(
        json.dumps({"from": "worker", "text": "a"})
        + "\n"
        + json.dumps({"from": "worker", "text": "b"})
        + "\n"
    )

    probe = runner.invoke(app, ["inbox-status", str(sdir)])
    assert probe.exit_code == 0
    payload = json.loads(probe.stdout)
    assert payload["senders"] == {"worker": {"total": 2, "cursor": 0, "unread": 2}}

    # A following REAL consuming read returns exactly the probed messages.
    result = asyncio.run(ss.read_messages())
    assert [m["text"] for m in result["messages"]] == ["a", "b"]

    # The probe itself did not consume: only the real read advanced the cursor.
    after = runner.invoke(app, ["inbox-status", str(sdir)])
    assert json.loads(after.stdout)["senders"]["worker"]["unread"] == 0

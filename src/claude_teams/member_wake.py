"""Deterministic inbox-wake decision hook for external team members.

Invoked as a ``Stop`` hook (``python -m claude_teams.member_wake
--joined-session-dir <dir> --member <name>``) in a manually-started session
that has joined a lead's team via ``join_team``. On every member turn end it
verifies — deterministically, not by trusting the model to remember — that the
member's inbox in the **joined** lead's session dir is drained and that an
inbox watcher is armed for that dir, and either allows the stop or blocks with
an operational instruction to call ``external_read(member_token=...)`` or to
arm the reader-scoped watcher.

This mirrors :mod:`claude_teams.lead_wake` (whose scan/armed/guard machinery
it reuses **by import**) with a member-shaped decision path:

- **M0** member kill switch (``WIN_AGENT_TEAMS_MEMBER_WAKE``, falling back to
  the lead kill switch when unset) — off → allow.
- **M1** the joined session dir comes from the baked argument ONLY (never from
  process-identity discovery, which would find the member session's own dir);
  missing/not a dir → allow.
- **M2** membership gate: the joined ``agents.json`` must carry a live
  (``status == "running"``) external join record for the member; anything
  else (absent, ``left``, terminal, unreadable) fails open → allow.
- **M2b** abandoned-team TTL: no lead-side activity (state markers, inboxes,
  ``agents.json``) within ``WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS``
  (default 6h) fails open → allow, so a dead team never blocks forever.
- **M3** unread in the joined ``inbox-<member>`` → block (external_read).
- **M4** a running watcher for the joined dir → allow.
- **M5** not armed → block with the reader-scoped watch command and the
  ``leave_team`` escape hatch.

Never-unstoppable contract identical to ``lead_wake``: exit 0 always, a block
prints only ``{"decision":"block","reason":...}``, and the no-progress guard
(shared cap env ``WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS`` — deliberate)
fails open. The guard file is ``wake-progress-member-<member>.json`` so it can
never collide with the lead's ``wake-progress-<reader>.json``. No credential
is ever baked or read: detecting unread is a token-free scan, and the hook
only *instructs* the model to call ``external_read`` with the token it already
holds in its transcript.
"""

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

from claude_teams import lead_wake, server_simple
from claude_teams.agent_output import SPAWNED_BY_SOURCE_FIELD, SPAWNED_BY_SOURCE_JOIN
from claude_teams.lead_wake import (
    WakeDecision,
    _apply_guard,
    _is_armed,
    _read_payload,
    _scan_senders,
)

_MEMBER_KILL_SWITCH_ENV = "WIN_AGENT_TEAMS_MEMBER_WAKE"
_TTL_ENV = "WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS"
_DEFAULT_TTL_SECONDS = 21600.0  # 6 hours
# Files whose mtimes count as lead-side activity for the M2b liveness TTL.
# ``agents.json`` is included beyond the plan's state/inbox pair because a
# just-joined quiet team may have neither a state marker nor an inbox yet,
# while ``agents.json`` always exists and is rewritten on membership changes.
_ACTIVITY_GLOBS = ("state-*.json", "inbox-*.jsonl", "agents.json")


def _member_kill_switch_on() -> bool:
    """Return whether member-wake is enabled (M0 truth table).

    ``WIN_AGENT_TEAMS_MEMBER_WAKE`` governs member-wake: ON unless explicitly
    ``0``. When it is unset (or blank), fall back to the lead kill switch so a
    single ``WIN_AGENT_TEAMS_LEAD_WAKE=0`` disables both — but an explicit
    ``...MEMBER_WAKE=1`` re-enables member-wake independently. This is a new
    helper because ``lead_wake._kill_switch_on`` reads only the lead var.
    """
    raw = os.environ.get(_MEMBER_KILL_SWITCH_ENV)
    if raw is None or not raw.strip():
        return lead_wake._kill_switch_on()
    return raw.strip() != "0"


def _guard_identity(member: str) -> str:
    """Guard-file identity: yields ``wake-progress-member-<member>.json``.

    The ``member-`` prefix guarantees the file never collides with the lead's
    ``wake-progress-<reader>.json``, even for a member named ``team-lead``.
    """
    return f"member-{member}"


def _membership_live(joined_session_dir: Path, member: str) -> bool:
    """Return whether the joined ``agents.json`` carries a LIVE membership (M2).

    Live requires the matching record (``name == member``,
    ``backend == "external"``, ``spawned_by_source == "join_ticket"``) to have
    ``status == "running"``. Everything else — no record, ``left``, terminal
    statuses, unreadable registry — fails open to ``False`` (allow).
    """
    try:
        records = server_simple._load_agents(joined_session_dir.name)
    except Exception:
        return False
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("name") != member:
            continue
        if rec.get("backend") != "external":
            continue
        if rec.get(SPAWNED_BY_SOURCE_FIELD) != SPAWNED_BY_SOURCE_JOIN:
            continue
        status = rec.get("status")
        if status in server_simple._TERMINAL_STATUSES:
            return False
        return status == "running"
    return False


def _ttl_seconds() -> float:
    return server_simple._strict_positive_seconds(_TTL_ENV, _DEFAULT_TTL_SECONDS)


def _joined_session_recently_active(joined_session_dir: Path, now: float) -> bool:
    """Return whether the joined session shows activity within the TTL (M2b).

    Scans the newest mtime across the joined dir's state markers, inboxes,
    and ``agents.json``. No files at all, or nothing fresh within the TTL,
    means the team looks abandoned → fail open (allow).
    """
    newest: float | None = None
    for pattern in _ACTIVITY_GLOBS:
        with suppress(OSError):
            for path in joined_session_dir.glob(pattern):
                with suppress(OSError):
                    mtime = path.stat().st_mtime
                    newest = mtime if newest is None else max(newest, mtime)
    if newest is None:
        return False
    return (now - newest) <= _ttl_seconds()


def _member_read_reason(unread_senders: list[str]) -> str:
    """Operational M3 reason: unread present, call external_read and drain."""
    senders = ", ".join(unread_senders)
    return (
        f"Unread messages are waiting in your member inbox from: {senders}. "
        "Call external_read(member_token=...) to process them, and keep "
        "calling it while has_more is true before ending your turn."
    )


def _member_arm_reason(joined_session_dir: Path, member: str) -> str:
    """Operational M5 reason: arm the reader-scoped watcher, or leave the team."""
    cmd = server_simple._watch_command_bash(
        joined_session_dir, reader=member, bind_owner=False
    )
    return (
        "An inbox watcher is not currently running for the team session you "
        "joined, so lead messages will not wake you while you are idle. Start "
        "the watcher now as a background task using the Bash tool with "
        f"run_in_background set to true: {cmd}  Once it is running in the "
        "background, you may end your turn. If you are finished as a member "
        "of this team, call leave_team(member_token=...) instead to stop "
        "these reminders."
    )


def evaluate_member(  # noqa: PLR0911 - one return per decision-table row.
    payload: dict,
    *,
    member: str,
    joined_session_dir: str | Path | None,
) -> WakeDecision:
    """Evaluate one member Stop payload against the decision table (M0..M5)."""
    log: dict = {"ran": True, "member": member}

    if not _member_kill_switch_on():
        log.update(decision="allow", why="kill-switch-off")
        return WakeDecision("allow", "M0", log=log)

    member = (member or "").strip()
    joined = Path(joined_session_dir) if joined_session_dir else None
    if not member or joined is None or not joined.is_dir():
        log.update(decision="allow", why="no-joined-session")
        return WakeDecision("allow", "M1", log=log)
    log["joined_session_dir"] = str(joined)

    if not _membership_live(joined, member):
        log.update(decision="allow", why="membership-not-live")
        return WakeDecision("allow", "M2", log=log)

    if not _joined_session_recently_active(joined, time.time()):
        log.update(decision="allow", why="joined-session-stale")
        return WakeDecision("allow", "M2b", log=log)

    identity = _guard_identity(member)
    senders = _scan_senders(joined, member)
    unread_senders = sorted(s for s, v in senders.items() if v["unread"] > 0)
    log["unread_senders"] = unread_senders
    armed = _is_armed(payload, joined)
    log["armed"] = armed
    log["stop_hook_active"] = bool(payload.get("stop_hook_active"))

    if unread_senders:
        block = WakeDecision(
            "block", "M3", reason=_member_read_reason(unread_senders), log=log
        )
        log.update(decision="block", why="unread")
        return _apply_guard(
            session_dir=joined,
            identity=identity,
            senders=senders,
            payload=payload,
            block=block,
            log=log,
        )
    if armed:
        log.update(decision="allow", why="armed")
        return WakeDecision("allow", "M4", log=log)
    block = WakeDecision(
        "block", "M5", reason=_member_arm_reason(joined, member), log=log
    )
    log.update(decision="block", why="not-armed")
    return _apply_guard(
        session_dir=joined,
        identity=identity,
        senders=senders,
        payload=payload,
        block=block,
        log=log,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m claude_teams.member_wake")
    parser.add_argument("--joined-session-dir", default=None)
    parser.add_argument("--member", default="")
    return parser.parse_args(argv)


def _log_line(decision: WakeDecision) -> str:
    """Render the single structured, body-free observability line."""
    return "win-agent-teams/member-wake " + json.dumps(
        {"code": decision.code, **decision.log}, separators=(",", ":")
    )


def main(argv: list[str] | None = None) -> None:
    """Read the Stop payload from stdin, evaluate, and print the decision.

    Identical never-unstoppable contract to ``lead_wake.main``: a ``block``
    prints ``{"decision":"block","reason":...}`` to stdout (exit 0, which
    continues the turn and feeds ``reason`` back to the model); an ``allow``
    prints nothing. One structured log line always goes to stderr. The hook
    never emits ``{"continue":false}`` and never exits non-zero, so it can
    never make the member session unstoppable.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    payload = _read_payload()
    decision = evaluate_member(
        payload,
        member=args.member,
        joined_session_dir=args.joined_session_dir,
    )
    sys.stderr.write(_log_line(decision) + "\n")
    if decision.action == "block":
        sys.stdout.write(
            json.dumps({"decision": "block", "reason": decision.reason}) + "\n"
        )


if __name__ == "__main__":
    main()

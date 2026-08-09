"""Deterministic inbox-wake decision hook for Claude Code lead agents.

Invoked as a ``Stop`` hook (``python -m claude_teams.lead_wake --session-dir
<dir> --reader <name>``). On every lead turn end it verifies — deterministically,
not by trusting the model to remember — that an inbox watcher is armed for the
session, and either allows the stop (a tracked watcher carries the wait) or
blocks with an operational instruction to arm the watcher or to call
``read_messages``.

Design B, ``background_tasks`` variant: arming is verified from the
``background_tasks`` array the harness delivers in the ``Stop`` stdin payload, so
no pid/marker file is needed. The only disk artifact this module writes is the
progress-guard file ``wake-progress-<reader>.json`` under the session dir, which
bounds a no-progress block loop well under the harness's hard 8-block cap. All
waiting is zero-token: the hook returns immediately in every branch; the long
wait is owned by the tracked watcher process.

This module lives outside ``claude_teams.hooks`` (which is import-light and
imported by both ``cli`` and ``server_simple``) because it needs session/inbox
discovery from ``server_simple``/``messaging``; keeping it separate avoids an
import cycle. The settings-wiring helpers stay in ``hooks``.
"""

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from claude_teams import messaging, server_simple

_KILL_SWITCH_ENV = "WIN_AGENT_TEAMS_LEAD_WAKE"
_MAX_NOPROGRESS_ENV = "WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS"
_DEFAULT_MAX_NOPROGRESS = 3
_ROOT_LEAD_NAME = "team-lead"
_PROGRESS_SCHEMA = "lead-wake-progress/1"


@dataclass
class WakeDecision:
    """Outcome of one Stop-hook evaluation.

    ``action`` is ``"allow"`` (plain exit 0, no ``decision`` printed) or
    ``"block"`` (prints ``{"decision":"block","reason":...}``). ``code`` is the
    decision-table row (``D0``..``D6``). ``log`` carries the structured,
    body-free observability fields.
    """

    action: str
    code: str
    reason: str | None = None
    log: dict = field(default_factory=dict)


def _kill_switch_on() -> bool:
    """Return ``True`` unless the kill switch is explicitly set to ``0``."""
    return os.environ.get(_KILL_SWITCH_ENV, "1").strip() != "0"


def _max_noprogress() -> int:
    """Return the no-progress block cap before D6 fail-open (default 3)."""
    raw = os.environ.get(_MAX_NOPROGRESS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_NOPROGRESS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_NOPROGRESS
    return value if value >= 1 else _DEFAULT_MAX_NOPROGRESS


def _resolve_identity(reader_arg: str) -> str:
    """Resolve the inbox reader identity.

    ``AGENT_NAME`` (set for every spawned nested lead) is authoritative so a
    nested lead reads its OWN inbox; the baked ``--reader`` value is the default
    for the top-level install where ``AGENT_NAME`` is empty. Falls back to
    ``team-lead``. Mirrors ``server_simple.IDENTITY`` semantics.
    """
    env_name = os.environ.get("AGENT_NAME", "").strip()
    if env_name:
        return env_name
    return reader_arg.strip() if reader_arg and reader_arg.strip() else _ROOT_LEAD_NAME


def _resolve_session_dir(session_dir_arg: str | None) -> Path | None:
    """Resolve the active session dir via ``server_simple`` discovery.

    Reuses (does not reimplement) the same discovery the ``session-dir`` CLI
    calls: ``_active_session_id(create=False)`` honours ``AGENT_SESSION_ID``,
    then the workspace binding, then the cwd+identity fallback. Never uses the
    Stop stdin ``session_id`` (that is Claude Code's transcript id). Falls back
    to the baked ``--session-dir`` argument only if discovery yields nothing but
    the baked dir still exists. Fail-open: any error yields ``None`` (D1).
    """
    with suppress(Exception):
        session_id = server_simple._active_session_id(create=False)
        if session_id:
            return server_simple._session_dir(session_id)
    if session_dir_arg:
        candidate = Path(session_dir_arg)
        if candidate.is_dir():
            return candidate
    return None


def _live_subagent_names(session_dir: Path, identity: str) -> list[str]:
    """Return names of this agent's live (non-terminal) CHILD subagents.

    Scoped to the caller's own children so a leaf worker never counts itself
    (or a sibling) as something it must wait for: a record counts only when its
    ``parent`` field equals ``identity`` AND its ``name`` differs from
    ``identity`` (self is always excluded). Empty means the agent leads no live
    children, so it has nothing to wait for (D2 fast allow) — exactly what a
    leaf worker must hit.

    Backward tolerance: agent records written before the ``parent`` field
    existed carry no parent at all. When NOT ONE record in the registry has a
    ``parent`` key, we cannot scope by parentage, so we fall back to "every
    non-terminal agent except self". That fallback still excludes self, so a
    legacy leaf session (its own single record) can never regress into the
    self-count bug, while a legacy lead still sees its (unscoped) live agents.
    In a mixed session (some records have ``parent``, some don't) the scoped
    predicate applies and parentless records simply do not match. Fail-open:
    any error yields ``[]``.
    """
    try:
        session_id = session_dir.name
        records = [
            agent
            for agent in server_simple._load_agents(session_id)
            if isinstance(agent, dict)
        ]
    except Exception:
        return []

    any_parent = any("parent" in rec for rec in records)
    live: list[str] = []
    for rec in records:
        name = str(rec.get("name") or "")
        if name == identity:
            continue  # never count self
        status = rec.get("status")
        if status in server_simple._TERMINAL_STATUSES or status == "left":
            continue  # killed (terminal) and left (departed member) are not live
        if any_parent:
            if rec.get("parent") == identity:
                live.append(name)
        else:
            live.append(name)
    return live


def _scan_senders(session_dir: Path, identity: str) -> dict[str, dict[str, int]]:
    """Return per-sender ``{total, cursor, unread}`` from one read-only scan.

    Mirrors the ``inbox-status`` CLI computation exactly (cli.py): it composes
    ``read_inbox_by_sender`` + ``load_inbox_cursors`` and never writes a cursor.
    This single scan feeds both the unread discriminator and the progress
    guard's cursor snapshot.
    """
    inbox_path = session_dir / f"inbox-{identity}.jsonl"
    cursor_path = session_dir / f"inbox-{identity}.pos.json"
    by_sender = messaging.read_inbox_by_sender(inbox_path)
    cursors = messaging.load_inbox_cursors(cursor_path)
    senders: dict[str, dict[str, int]] = {}
    for sender, messages in by_sender.items():
        total = len(messages)
        cursor = cursors.get(sender, 0)
        unread = total - min(cursor, total)
        senders[sender] = {"total": total, "cursor": cursor, "unread": unread}
    return senders


def _command_matches_session(command: str, session_dir: Path) -> bool:
    r"""Return whether ``command`` is a ``watch`` invocation for ``session_dir``.

    Separator-insensitive (``\`` normalised to ``/``) and session-scoped:
    matches on the ``claude_teams.cli`` + ``watch`` token set AND a reference to
    this session dir (full path or its basename = the session id), so a watcher
    for a DIFFERENT session does not count as armed.
    """
    norm = command.replace("\\", "/")
    if "claude_teams.cli" not in norm or "watch" not in norm:
        return False
    dir_norm = str(session_dir).replace("\\", "/")
    if dir_norm and dir_norm in norm:
        return True
    basename = Path(str(session_dir)).name
    return bool(basename) and basename in norm


def _is_armed(payload: dict, session_dir: Path) -> bool:
    """Return whether a running tracked watcher for this session is present."""
    tasks = payload.get("background_tasks")
    if not isinstance(tasks, list):
        return False
    for entry in tasks:
        if not isinstance(entry, dict) or entry.get("status") != "running":
            continue
        command = entry.get("command")
        if isinstance(command, str) and _command_matches_session(command, session_dir):
            return True
    return False


def _read_reason(unread_senders: list[str]) -> str:
    """Operational D3 reason: unread present, call read_messages and drain."""
    senders = ", ".join(unread_senders)
    return (
        f"Unread messages are waiting in your inbox from: {senders}. Call "
        "read_messages to process them, and keep calling it while has_more is "
        "true before ending your turn."
    )


def _arm_reason(session_dir: Path) -> str:
    """Operational D5 reason: no watcher armed, start it as a background task."""
    cmd = server_simple._watch_command_bash(session_dir, bind_owner=False)
    return (
        "An inbox watcher is not currently running for this session, so worker "
        "replies will not wake you while you are idle. Start the watcher now as "
        "a background task using the Bash tool with run_in_background set to "
        f"true: {cmd}  Once it is running in the background, you may end your "
        "turn."
    )


def _guard_file(session_dir: Path, identity: str) -> Path:
    return session_dir / f"wake-progress-{identity}.json"


def _read_guard(session_dir: Path, identity: str) -> dict | None:
    """Return the prior progress snapshot, or ``None`` if absent/corrupt."""
    path = _guard_file(session_dir, identity)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_guard(
    session_dir: Path,
    identity: str,
    senders: dict[str, dict[str, int]],
    noprogress: int,
) -> None:
    """Atomically persist the progress snapshot (mirror hooks._write_marker_atomic)."""
    snapshot = {
        sender: {"total": v["total"], "cursor": v["cursor"]}
        for sender, v in senders.items()
    }
    marker = {
        "schema": _PROGRESS_SCHEMA,
        "reader": identity,
        "senders": snapshot,
        "noprogress_blocks": noprogress,
        "ts": time.time(),
    }
    path = _guard_file(session_dir, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(marker), encoding="utf-8")
    tmp.replace(path)


def _cursor_advanced(prior: dict | None, senders: dict[str, dict[str, int]]) -> bool:
    """Return whether any sender's cursor advanced vs the prior snapshot.

    Progress is keyed on **cursor**, never on an unread delta: a same-window
    drain+arrival can leave unread flat while the cursor advances, so an
    unread-keyed guard would wrongly fail-open (plan F1).
    """
    if not prior:
        return False
    prior_senders = prior.get("senders")
    if not isinstance(prior_senders, dict):
        return False
    for sender, v in senders.items():
        prior_cursor = 0
        prior_entry = prior_senders.get(sender)
        if isinstance(prior_entry, dict) and isinstance(prior_entry.get("cursor"), int):
            prior_cursor = prior_entry["cursor"]
        if v["cursor"] > prior_cursor:
            return True
    return False


def _apply_guard(
    *,
    session_dir: Path,
    identity: str,
    senders: dict[str, dict[str, int]],
    payload: dict,
    block: WakeDecision,
    log: dict,
) -> WakeDecision:
    """Consult the progress guard on a would-be block (D3/D5).

    A productive wake (cursor advanced) resets the counter and the block
    proceeds unshortened. Otherwise, only when ``stop_hook_active`` is true is
    the no-progress counter incremented; at the cap it fails open (D6, toward
    stoppable). ``stop_hook_active`` gates consulting the guard — never a
    standalone skip.
    """
    prior = _read_guard(session_dir, identity)
    stop_active = bool(payload.get("stop_hook_active"))
    cap = _max_noprogress()

    if _cursor_advanced(prior, senders):
        _write_guard(session_dir, identity, senders, 0)
        log["noprogress_blocks"] = 0
        return block

    prior_count = int(prior.get("noprogress_blocks", 0)) if prior else 0
    if stop_active:
        noprogress = prior_count + 1
        if noprogress >= cap:
            _write_guard(session_dir, identity, senders, 0)
            log["noprogress_blocks"] = noprogress
            log.update(decision="allow", why="guard-fail-open")
            return WakeDecision("allow", "D6", log=log)
    else:
        noprogress = prior_count
    _write_guard(session_dir, identity, senders, noprogress)
    log["noprogress_blocks"] = noprogress
    return block


def evaluate(
    payload: dict,
    *,
    reader_arg: str,
    session_dir_arg: str | None = None,
) -> WakeDecision:
    """Evaluate one Stop payload against the decision table (D0..D6)."""
    log: dict = {"ran": True}

    if not _kill_switch_on():
        log.update(decision="allow", why="kill-switch-off")
        return WakeDecision("allow", "D0", log=log)

    identity = _resolve_identity(reader_arg)
    log["identity"] = identity

    session_dir = _resolve_session_dir(session_dir_arg)
    log["session_resolved"] = session_dir is not None
    if session_dir is None:
        log.update(decision="allow", why="no-session")
        return WakeDecision("allow", "D1", log=log)

    live = _live_subagent_names(session_dir, identity)
    log["live_subagents"] = len(live)
    if not live:
        log.update(decision="allow", why="no-live-subagents")
        return WakeDecision("allow", "D2", log=log)

    senders = _scan_senders(session_dir, identity)
    unread_senders = sorted(s for s, v in senders.items() if v["unread"] > 0)
    log["unread_senders"] = unread_senders
    armed = _is_armed(payload, session_dir)
    log["armed"] = armed
    log["stop_hook_active"] = bool(payload.get("stop_hook_active"))

    if unread_senders:
        block = WakeDecision(
            "block", "D3", reason=_read_reason(unread_senders), log=log
        )
        log.update(decision="block", why="unread")
        return _apply_guard(
            session_dir=session_dir,
            identity=identity,
            senders=senders,
            payload=payload,
            block=block,
            log=log,
        )
    if armed:
        log.update(decision="allow", why="armed")
        return WakeDecision("allow", "D4", log=log)
    block = WakeDecision("block", "D5", reason=_arm_reason(session_dir), log=log)
    log.update(decision="block", why="not-armed")
    return _apply_guard(
        session_dir=session_dir,
        identity=identity,
        senders=senders,
        payload=payload,
        block=block,
        log=log,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m claude_teams.lead_wake")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--reader", default=_ROOT_LEAD_NAME)
    return parser.parse_args(argv)


def _read_payload() -> dict:
    """Read and parse the Stop-hook JSON payload from stdin (tolerant)."""
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _log_line(decision: WakeDecision) -> str:
    """Render the single structured, body-free observability line (FR25)."""
    return "win-agent-teams/lead-wake " + json.dumps(
        {"code": decision.code, **decision.log}, separators=(",", ":")
    )


def main(argv: list[str] | None = None) -> None:
    """Read the Stop payload from stdin, evaluate, and print the decision.

    A ``block`` prints ``{"decision":"block","reason":...}`` to stdout (exit 0,
    which continues the turn and feeds ``reason`` back to the model); an
    ``allow`` prints nothing. One structured log line always goes to stderr. The
    hook never emits ``{"continue":false}`` and never exits non-zero, so it can
    never make the lead unstoppable.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    payload = _read_payload()
    decision = evaluate(
        payload,
        reader_arg=args.reader,
        session_dir_arg=args.session_dir,
    )
    sys.stderr.write(_log_line(decision) + "\n")
    if decision.action == "block":
        sys.stdout.write(
            json.dumps({"decision": "block", "reason": decision.reason}) + "\n"
        )


if __name__ == "__main__":
    main()

"""Minimal CLI for win-agent-teams."""

import fnmatch
import json
import math
import os
import signal
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from claude_teams import leases, server_simple
from claude_teams.agent_output import (
    SPAWNED_BY_FIELD,
    SPAWNED_BY_SOURCE_FIELD,
    SPAWNED_BY_SOURCE_OPERATOR,
)
from claude_teams.backends.registry import registry
from claude_teams.hooks import _SAFE_AGENT_RE
from claude_teams.messaging import (
    load_inbox_cursors,
    read_inbox_by_sender,
    unread_sender_counts,
)
from claude_teams.server_simple import mcp

_WATCH_POLL_SECONDS = 0.5
_WATCH_DEFAULT_PATTERN = "state-*.json"

# Waiting markers written by these hook events are NOT coordinator-actionable:
# ``SubagentStop`` fires when one of an agent's OWN built-in Task subagents
# finishes, while the agent itself is still mid-task and will resume on its
# next tool call. Waking a coordinator for it is a false positive.
_NON_ACTIONABLE_WAITING_EVENTS: frozenset[str] = frozenset({"SubagentStop"})

_WATCH_SETTLE_DEFAULT_SECONDS = 15.0


def _settle_seconds_from_env() -> float:
    """Return the settle window from the environment, falling back to the default.

    A malformed override (non-numeric, ``NaN``, infinite, or negative) falls
    back to :data:`_WATCH_SETTLE_DEFAULT_SECONDS` rather than breaking the CLI or
    — in the ``NaN`` case — silently making ``now - since >= settle`` never true,
    which would suppress every genuine wake. ``0`` is a valid override (settle
    disabled).
    """
    raw = os.environ.get("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS")
    if raw is None:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    if not math.isfinite(value) or value < 0:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    return value


# Seconds an actionable ``waiting`` marker must persist before it wakes the
# coordinator. An agent that merely parks between operations (e.g. a
# backgrounded bash or a brief yield) flips waiting->running again within this
# window; requiring the marker to settle suppresses that churn. Env-overridable.
_WATCH_SETTLE_SECONDS = _settle_seconds_from_env()

app = typer.Typer(
    name="win-agent-teams",
    help="Spawn and communicate with Claude Code and Codex agents.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve() -> None:
    """Start the MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


@app.command()
def backends(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON."),
) -> None:
    """List available backends."""
    rows = []
    for name, backend in registry:
        rows.append(
            {
                "name": name,
                "binary": backend.binary_name,
                "default_model": backend.default_model(),
                "supported_models": backend.supported_models(),
            }
        )
    if output_json:
        console.print_json(json.dumps(rows))
        return
    if not rows:
        console.print("[yellow]No backends available.[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title="Available Backends")
    table.add_column("Name", style="bold cyan")
    table.add_column("Binary")
    table.add_column("Default Model", style="green")
    table.add_column("Supported Models")
    for row in rows:
        table.add_row(
            row["name"],
            row["binary"],
            row["default_model"],
            ", ".join(row["supported_models"]),
        )
    console.print(table)


lease_app = typer.Typer(
    name="lease",
    help="Operator escape hatch for a stuck per-agent delivery lease (A4b).",
    no_args_is_help=True,
)
app.add_typer(lease_app, name="lease")


def _authorize(session_id: str, token: str) -> None:
    """Abort unless ``token`` is the session's recovery token.

    This path can terminate a live child and fence out a running delivery, so
    it is deliberately CLI-only and gated. Nothing reachable over MCP can call
    it: ordinary ``kill_agent`` never bypasses a live lease.
    """
    if not token or token != server_simple._ensure_lead_token(session_id):
        console.print("[red]Invalid or missing session recovery token.[/red]")
        raise typer.Exit(code=2)


def _lease_or_exit(session_id: str, agent: str):
    try:
        lease = leases.active_lease(server_simple._leases_file(session_id), agent)
    except leases.LeaseStoreError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=5) from exc
    if lease is None:
        console.print(f"[yellow]No lease held for {agent!r}.[/yellow]")
        raise typer.Exit(code=1)
    return lease


def _describe(lease, *, holder_live: bool, child_alive: bool | None) -> dict:
    return {
        "agent": lease.agent,
        "operation_id": lease.operation_id,
        "generation": lease.generation,
        "backend_session_id": lease.backend_session_id,
        # The attempt nonce: what an operator needs to check by hand whether
        # the prompt actually reached the target before forcing anything.
        "attempt_nonce": lease.nonce,
        "holder_pid": lease.holder_pid,
        "holder_live": holder_live,
        "resumed_child_alive": child_alive,
        "deadline": lease.deadline,
    }


@lease_app.command("inspect")
def lease_inspect(
    session_id: str = typer.Argument(..., help="Session id holding the agent."),
    agent: str = typer.Argument(..., help="Agent name."),
    token: str = typer.Option(..., "--token", help="Session recovery token."),
) -> None:
    """Show the attempt nonce, holder liveness, and resumed-child liveness."""
    _authorize(session_id, token)
    lease = _lease_or_exit(session_id, agent)
    holder_live = server_simple._lease_holder_live(
        lease.holder_pid, lease.holder_create_token
    )
    console.print_json(
        json.dumps(_describe(lease, holder_live=holder_live, child_alive=None))
    )


@lease_app.command("clear")
def lease_clear(
    session_id: str = typer.Argument(..., help="Session id holding the agent."),
    agent: str = typer.Argument(..., help="Agent name."),
    token: str = typer.Option(..., "--token", help="Session recovery token."),
) -> None:
    """Clear a lease whose holder is dead or token-mismatched.

    Refuses a provably live holder: clearing under one would let a second
    caller resume into a delivery that is still in progress. Use ``force`` for
    a live-but-hung holder — it fences the holder out first.
    """
    _authorize(session_id, token)
    lease = _lease_or_exit(session_id, agent)
    if server_simple._lease_holder_live(lease.holder_pid, lease.holder_create_token):
        console.print(
            "[red]Holder is provably live; refusing to clear. "
            "Use `lease force` if it is hung.[/red]"
        )
        raise typer.Exit(code=3)
    with server_simple._agents_file_lock(session_id):
        # Under the registry lock, which is what ``reserve_lease`` holds when it
        # grants. Clearing outside it can drop a lease that was granted in the
        # meantime, letting a third caller resume the same conversation.
        _, persisted = leases.force_clear_lease(
            server_simple._leases_file(session_id),
            agent,
            expect_operation_id=lease.operation_id,
        )
    if not persisted:
        console.print(
            "[red]The lease was not cleared: either it moved to another "
            "operation, or the store could not be written.[/red]"
        )
        raise typer.Exit(code=4)
    console.print(f"[green]Cleared stale lease for {agent!r}.[/green]")


@lease_app.command("force")
def lease_force(
    session_id: str = typer.Argument(..., help="Session id holding the agent."),
    agent: str = typer.Argument(..., help="Agent name."),
    token: str = typer.Option(..., "--token", help="Session recovery token."),
) -> None:
    """Fence out a live-but-overdue holder, then release the lease.

    Order is load-bearing. The **fencing generation is bumped first**, so the
    original holder can no longer win its finalize CAS — otherwise it could
    wake up after we terminated its child and write a record describing a
    delivery that no longer exists. Only then is the resumed child terminated
    (and only when ownership is provable), and only then is the lease cleared.

    "Overdue" is enforced, not merely described: a holder that is provably live
    and still inside its deadline is doing exactly what the lease is for, and
    forcing it kills a delivery in progress. Wait for the deadline, or use
    ``lease clear`` once the holder is gone.
    """
    _authorize(session_id, token)
    lease = _lease_or_exit(session_id, agent)
    lease_path = server_simple._leases_file(session_id)

    holder_live = server_simple._lease_holder_live(
        lease.holder_pid, lease.holder_create_token
    )
    if holder_live and time.time() < lease.deadline:
        console.print(
            "[red]Holder is live and not yet overdue "
            f"(deadline {lease.deadline}); refusing to force. Forcing here "
            "would terminate a delivery that is still within its lease.[/red]"
        )
        raise typer.Exit(code=3)

    fenced_generation = None
    terminated = False
    cleared = None
    persisted = False
    # All three steps run inside ONE registry transaction, and the ordering
    # inside it is the protocol.
    #
    # Splitting them was the defect: the operation id was revalidated only at
    # the final compare-and-swap, so if the inspected operation finalized and a
    # queued caller was granted the target in between — which ``reserve_lease``
    # may legitimately do the moment the lock is free — this command fenced and
    # killed THAT caller's live delivery and reported the mismatch afterwards,
    # when nothing could be undone. Revalidating early is not enough on its own
    # either: the handoff can happen between the check and the kill.
    #
    # ``reserve_lease`` grants only while holding this same lock, so holding it
    # across the whole sequence is what makes the validation meaningful. The
    # cost is that registry readers block for the duration; ``kill_process`` is
    # a terminate, not the graceful-shutdown path, so that is bounded and short.
    with server_simple._agents_transaction(session_id) as agents:
        current = leases.active_lease(lease_path, agent)
        if current is None or current.operation_id != lease.operation_id:
            console.print(
                "[red]The lease moved to another operation "
                f"({None if current is None else current.operation_id}) after "
                "it was inspected. Nothing was fenced, killed or cleared. "
                "Re-inspect and retry.[/red]"
            )
            raise typer.Exit(code=4)

        # Step 1 — fence, so the holder can no longer win its finalize CAS and
        # write a record describing a delivery that is about to not exist.
        record = next((a for a in agents if a.get("name") == agent), None)
        child_pid = None
        if record is not None:
            fenced_generation = server_simple._bump_generation(record)
            child_pid = record.get("pid")
            server_simple._save_agents_transaction(session_id, agents)

        # Step 2 — terminate the resumed child, only when ownership is provable.
        if record is not None and child_pid is not None:
            create_token = server_simple._agent_create_token(record)
            if server_simple.process_manager.owns_process(str(child_pid), create_token):
                server_simple.process_manager.kill_process(str(child_pid))
                terminated = True

        # Step 3 — release the lease. The CAS is kept even though nothing can
        # have moved under the held lock: it is cheap, and it keeps the clear
        # honest if this sequence is ever split again.
        cleared, persisted = leases.force_clear_lease(
            lease_path, agent, expect_operation_id=lease.operation_id
        )
    console.print_json(
        json.dumps(
            {
                "forced": persisted,
                "agent": agent,
                "fenced_generation": fenced_generation,
                "attempt_nonce": lease.nonce,
                "child_terminated": terminated,
                "lease_cleared": persisted,
                "held_by_operation": None if cleared is None else cleared.operation_id,
            }
        )
    )
    if not persisted:
        raise typer.Exit(code=4)


@app.command("adopt")
def adopt(
    session_id: str = typer.Argument(..., help="Session id holding the agent."),
    agent: str = typer.Argument(..., help="Agent name to adopt."),
    parent: str = typer.Argument(..., help="Agent name to record as the spawner."),
    token: str = typer.Option(..., "--token", help="Session recovery token."),
    expect_generation: int = typer.Option(
        ...,
        "--expect-generation",
        help="Record generation you observed; the write is refused if it moved.",
    ),
) -> None:
    """Record who spawned an agent whose ``spawned_by`` field is missing.

    Deliberately CLI-only and operator-driven. An MCP-callable equivalent would
    reintroduce exactly the hole the direction guard closes: "callable only by
    a caller claiming parentage" is tautological, because the operation itself
    writes the caller as the spawner and an agent's identity is self-asserted.
    A confused worker could adopt its own coordinator and then legitimately
    follow it up, turning a one-call mistake into a two-call one.

    Gated on the session recovery token *and* the generation you read, so an
    adoption cannot be replayed against a record that changed underneath it.
    The parentage is stored as operator-asserted rather than spawn-derived,
    because it was asserted by a human and not observed at spawn.

    Adoption is **recovery for a missing spawner, not re-parenting.** A record
    that already names a spawner is refused: with the token and the generation
    this would otherwise be able to move any record to any parent, which is
    broader than the contract and would let an operator hand one agent
    follow-up rights over another's child. Kill and respawn instead.

    A worker with filesystem access can still edit ``agents.json`` directly.
    That is an accepted non-goal: the guard prevents accidents, not bypasses.
    """
    _authorize(session_id, token)
    with server_simple._agents_transaction(session_id) as agents:
        record = next((a for a in agents if a.get("name") == agent), None)
        if record is None:
            console.print(f"[red]No agent named {agent!r} in {session_id!r}.[/red]")
            raise typer.Exit(code=1)
        current = record.get(SPAWNED_BY_FIELD)
        if isinstance(current, str) and current:
            console.print(
                f"[red]{agent!r} already records {current!r} as its spawner. "
                "adopt only fills in a MISSING spawner; it does not re-parent. "
                "Kill and respawn the agent if the parentage is wrong.[/red]"
            )
            raise typer.Exit(code=4)
        actual = server_simple._record_generation(record)
        if actual != expect_generation:
            console.print(
                f"[red]Record generation is {actual}, not the expected "
                f"{expect_generation}; refusing to adopt a record that moved."
                "[/red]"
            )
            raise typer.Exit(code=3)
        record[SPAWNED_BY_FIELD] = parent
        record[SPAWNED_BY_SOURCE_FIELD] = SPAWNED_BY_SOURCE_OPERATOR
        generation = server_simple._bump_generation(record)
        server_simple._save_agents_transaction(session_id, agents)
    console.print_json(
        json.dumps(
            {
                "adopted": True,
                "agent": agent,
                "spawned_by": parent,
                "spawned_by_source": SPAWNED_BY_SOURCE_OPERATOR,
                "generation": generation,
            }
        )
    )


def _snapshot_mtimes(session_dir: Path, pattern: str) -> dict[str, tuple[int, int]]:
    """Return ``{path: (mtime_ns, size)}`` for files matching ``pattern``.

    Using ``(mtime_ns, size)`` instead of a bare ``st_mtime`` float catches
    same-second (or same-tick) atomic-replace/rewrites that preserve the
    exposed mtime but change file size, not just mtime increases.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    if not session_dir.is_dir():
        return snapshot
    for entry in session_dir.iterdir():
        if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
            try:
                stat = entry.stat()
                snapshot[str(entry)] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
    return snapshot


def _changed_paths(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]
) -> list[str]:
    """Return paths that are new in ``after`` or whose ``(mtime_ns, size)`` differs."""
    changed = []
    for path, identity in after.items():
        prior = before.get(path)
        if prior is None or identity != prior:
            changed.append(path)
    return changed


def _path_identity(path: Path) -> tuple[int, int] | None:
    """Return one file's ``(mtime_ns, size)`` identity, or ``None`` if absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _waiting_agent(path: Path) -> str | None:
    """Return the agent name for a coordinator-actionable waiting marker.

    Returns ``None`` when ``path`` is not such a marker. A marker is actionable
    only when its state is ``waiting`` AND the hook event
    that produced it is not in :data:`_NON_ACTIONABLE_WAITING_EVENTS`. This
    filters out ``SubagentStop`` churn (an agent's own Task subagent finishing)
    the same way the caller already ignores ``running`` transitions. Markers with
    no recorded ``event`` are treated as actionable for backward compatibility.
    """
    if not (path.name.startswith("state-") and path.suffix == ".json"):
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or marker.get("state") != "waiting":
        return None
    event = marker.get("event")
    # ``event`` may be any JSON value; guard the membership test so a non-string
    # (unhashable list/dict) does not raise and regress marker tolerance. A
    # missing or non-string event is treated as actionable.
    if isinstance(event, str) and event in _NON_ACTIONABLE_WAITING_EVENTS:
        return None
    return path.stem.removeprefix("state-")


def _emit_wake(payload: dict) -> None:
    """Print one unwrapped JSONL wake record for a harness to inspect."""
    typer.echo(json.dumps(payload, separators=(",", ":")))


def _require_safe_reader(reader: str) -> str:
    """Return ``reader`` when it is safe to interpolate into inbox filenames.

    Reuses the repo's shared safe-agent-name invariant
    (``claude_teams.hooks._SAFE_AGENT_RE``) so a reader value can never carry
    path separators or control characters into ``inbox-<reader>.jsonl`` /
    ``inbox-<reader>.pos.json`` paths.
    """
    if not _SAFE_AGENT_RE.match(reader):
        msg = f"unsafe reader name: {reader!r}"
        raise ValueError(msg)
    return reader


@app.command()
def watch(
    session_dir: str = typer.Argument(..., help="Directory to watch."),
    timeout: float = typer.Option(
        60.0, "--timeout", "-t", help="Seconds to wait before giving up."
    ),
    pattern: str = typer.Option(
        _WATCH_DEFAULT_PATTERN,
        "--pattern",
        "-p",
        help="Glob pattern (relative to session_dir) to watch, e.g. state-*.json.",
    ),
    watch_inbox: bool = typer.Option(
        True,
        "--inbox/--no-inbox",
        help="Wake for unread messages to this orchestrator (enabled by default).",
    ),
    reader: str | None = typer.Option(
        None,
        "--reader",
        help=(
            "Inbox reader identity to watch. Overrides AGENT_NAME/team-lead; "
            "omit for the current env-based behavior."
        ),
    ),
) -> None:
    """Block until an agent is waiting, an inbox is unread, or output changes.

    State-marker changes are semantic: ``running`` lifecycle transitions and
    ``SubagentStop`` (a worker's own Task subagent finishing) are ignored, and a
    marker whose state is ``waiting`` exits 0 only after it *persists* as waiting
    for a short settle window (``WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS``, default
    15s) — a marker that resumes ``running`` inside the window is suppressed as
    a brief park. Other files selected by PATTERN wake on any creation/change.
    Unless ``--no-inbox`` is passed, unread messages for ``AGENT_NAME`` (or the
    root ``team-lead`` identity, or an explicit ``--reader NAME``) also exit 0
    without consuming them. When several
    signals are ready in one poll the priority is message > output > waiting.

    Success prints one JSON object with ``reason`` equal to ``message``,
    ``waiting``, or ``output``. Timeout prints nothing and exits 2; re-check
    status after exit 2 because a waiting transition may precede the initial
    marker snapshot, or a genuine waiting edge may still be inside its settle
    window at the deadline.
    """
    directory = Path(session_dir)
    deadline = time.monotonic() + timeout
    before = _snapshot_mtimes(directory, pattern)

    if reader is not None:
        try:
            _require_safe_reader(reader)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        reader_name = reader
    else:
        reader_name = os.environ.get("AGENT_NAME", "").strip() or "team-lead"
    inbox_path = directory / f"inbox-{reader_name}.jsonl"
    cursor_path = directory / f"inbox-{reader_name}.pos.json"
    inbox_before = _path_identity(inbox_path)

    if watch_inbox:
        unread = unread_sender_counts(inbox_path, cursor_path)
        if unread:
            _emit_wake(
                {
                    "reason": "message",
                    "from": list(unread),
                    "path": str(inbox_path),
                }
            )
            raise typer.Exit(code=0)

    # A waiting marker only wakes the coordinator once it has stayed waiting for
    # _WATCH_SETTLE_SECONDS. Every actionable candidate is tracked by its marker
    # path — a coordinator watches ALL agents' markers, so overlapping waits must
    # each settle independently; a single slot would drop one when another arrives
    # or resumes. Value is the first-seen monotonic time.
    pending_waits: dict[str, float] = {}

    while True:
        now = time.monotonic()
        after = _snapshot_mtimes(directory, pattern)
        changed = _changed_paths(before, after)
        inbox_after = _path_identity(inbox_path)

        # Explicit communication wins when message and state/output edges land
        # in the same polling interval.
        if watch_inbox and inbox_after != inbox_before:
            unread = unread_sender_counts(inbox_path, cursor_path)
            if unread:
                _emit_wake(
                    {
                        "reason": "message",
                        "from": list(unread),
                        "path": str(inbox_path),
                    }
                )
                raise typer.Exit(code=0)
        inbox_before = inbox_after

        waiting: list[tuple[str, str]] = []
        outputs: list[str] = []
        for changed_path in changed:
            path = Path(changed_path)
            if path.name.startswith("state-") and path.suffix == ".json":
                agent = _waiting_agent(path)
                if agent is not None:
                    waiting.append((agent, changed_path))
            else:
                outputs.append(changed_path)

        # Advance after every edge, including ignored running/corrupt markers,
        # so one non-ready write cannot be rediscovered forever.
        before = after

        # Register every actionable waiting edge seen this tick, keeping the
        # earliest first-seen time while a marker stays a candidate.
        for _agent, wpath in waiting:
            pending_waits.setdefault(wpath, now)

        # Emit outputs BEFORE any settled wait. A wait settles over several
        # polls, so an output edge can land in the same poll a wait matures;
        # since `before = after` has already consumed that output edge, waking
        # on the wait first would drop the output for good. Priority is
        # message > output > waiting (message is handled at the top of the loop).
        if outputs:
            _emit_wake({"reason": "output", "path": outputs[0]})
            raise typer.Exit(code=0)

        # Settle each candidate against its CURRENT state: drop any that flipped
        # back to running (or to a non-actionable SubagentStop) within the
        # window, and wake on the first that has stayed waiting long enough.
        # Iterate in insertion order so the earliest-seen settled wait wins.
        for wpath in list(pending_waits):
            current = _waiting_agent(Path(wpath))
            if current is None:
                del pending_waits[wpath]
            elif now - pending_waits[wpath] >= _WATCH_SETTLE_SECONDS:
                _emit_wake({"reason": "waiting", "agent": current, "path": wpath})
                raise typer.Exit(code=0)
        if time.monotonic() >= deadline:
            raise typer.Exit(code=2)
        time.sleep(_WATCH_POLL_SECONDS)


@app.command(name="session-dir")
def session_dir() -> None:
    """Print the current workspace session as ``id<TAB>dir<TAB>identity``.

    Discovery-only: resolves the active/recoverable session WITHOUT creating a
    session directory. On success exits 0 with exactly one tab-separated line on
    stdout and nothing on stderr. When no session exists it exits 3 with empty
    stdout. An internal error exits 1 with a message on stderr only.
    """
    try:
        session_id = server_simple._active_session_id(create=False)
        line = (
            f"{session_id}\t{server_simple._session_dir(session_id)}\t{server_simple.IDENTITY}"
            if session_id
            else None
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if line is None:
        raise typer.Exit(code=3)
    typer.echo(line)


@app.command(name="inbox-status")
def inbox_status(
    session_dir: str = typer.Argument(..., help="Session directory to probe."),
    reader: str | None = typer.Option(
        None,
        "--reader",
        help=(
            "Inbox reader identity to probe. Overrides AGENT_NAME/team-lead; "
            "omit for the current env-based behavior (matching `watch`)."
        ),
    ),
) -> None:
    """Emit a non-consuming inbox generation snapshot as one JSON object.

    Prints ``{"schema":"inbox-status/1","reader":...,"senders":{<from>:
    {"total":N,"cursor":M,"unread":K}}}`` where ``unread = total -
    min(cursor, total)``; an empty inbox yields ``"senders":{}``. This never
    writes a cursor. A bad/nonexistent/outside-base ``session_dir`` exits 4 with
    a message on stderr and no stdout; an internal error exits 1 with stderr
    only.
    """
    # Resolve the reader identically to `watch`: an explicit --reader wins,
    # otherwise the ambient AGENT_NAME (else the root team-lead). Never hardcode
    # team-lead — a nested lead must probe its OWN inbox.
    if reader is not None:
        try:
            _require_safe_reader(reader)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        reader_name = reader
    else:
        reader_name = os.environ.get("AGENT_NAME", "").strip() or "team-lead"

    directory = Path(session_dir)
    base = server_simple._SESSION_BASE.resolve()
    if not directory.is_dir() or directory.resolve().parent != base:
        typer.echo(f"session_dir not under session base: {session_dir!r}", err=True)
        raise typer.Exit(code=4)

    try:
        inbox_path = directory / f"inbox-{reader_name}.jsonl"
        cursor_path = directory / f"inbox-{reader_name}.pos.json"
        by_sender = read_inbox_by_sender(inbox_path)
        cursors = load_inbox_cursors(cursor_path)
        senders: dict[str, dict[str, int]] = {}
        for sender, messages in by_sender.items():
            total = len(messages)
            cursor = cursors.get(sender, 0)
            unread = total - min(cursor, total)
            senders[sender] = {"total": total, "cursor": cursor, "unread": unread}
        payload = {
            "schema": "inbox-status/1",
            "reader": reader_name,
            "senders": senders,
        }
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    app()

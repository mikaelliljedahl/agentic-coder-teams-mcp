"""Drive the Herdr launcher through the real MCP server path, in-process.

Spawning through :func:`server_simple.spawn_agent` rather than the process
manager directly is the whole point: hooks, the ``state-<agent>.json`` marker
and the inbox are the *server's* work, so only this path shows whether the
on-disk contract survives a new launcher.

Usage (needs a real ``herdr`` binary; creates and uses a disposable session)::

    ./.venv/bin/python scripts/herdr_nested_check.py

Prints ``PASS``/``FAIL`` per check and exits non-zero if any check failed.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SESSION = "nested"
CHILD = "herdr-grandchild"
PANE_PAINT_SECONDS = 15
REPORT_TIMEOUT_SECONDS = 240
POLL_SECONDS = 5
MIN_PANE_CHARS = 200

os.environ["WIN_AGENT_TEAMS_LINUX_LAUNCHER"] = "herdr"
os.environ["WIN_AGENT_TEAMS_HERDR_SESSION"] = SESSION

import claude_teams.server_simple as server  # noqa: E402
from claude_teams.backends import process_manager as pm  # noqa: E402

_results: list[tuple[str, bool]] = []


def say(message: str) -> None:
    """Write a line of progress to stdout."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record and report one check."""
    _results.append((name, ok))
    say(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def _marker_is_fresh(marker: Path, started_at: float) -> bool:
    """Whether the marker exists, parses, and was written after ``started_at``."""
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("event")) and float(payload.get("ts", 0)) >= started_at


def _result(raw: str) -> dict:
    """Return a herdr response's result object, or ``{}`` for an error envelope.

    Closing the last tab can take its workspace with it, after which
    ``tab list`` answers with an error rather than an empty list -- so the
    final cleanup check must not assume success.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    result = parsed.get("result") if isinstance(parsed, dict) else None
    return result if isinstance(result, dict) else {}


def _tab_labels(workspace_id: str) -> list[str] | None:
    """Return every tab label in a workspace, or ``None`` if Herdr did not answer.

    ``None`` matters: mapping an error to "no tabs" would make the cleanup
    check pass merely because Herdr became unavailable.
    """
    raw = herdr("tab", "list", "--workspace", workspace_id)
    result = _result(raw)
    if not result:
        return None
    return [str(tab.get("label") or "") for tab in result.get("tabs") or []]


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` still exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def herdr(*args: str) -> str:
    """Run a herdr command against the disposable session."""
    done = subprocess.run(  # noqa: S603 - argv is built here, not user input.
        ["herdr", "--session", SESSION, *args],  # noqa: S607 - PATH lookup is intended.
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return done.stdout.strip() or done.stderr.strip()


async def run_checks() -> int:
    """Spawn a child agent through Herdr and verify what reached disk."""
    check(
        "herdr launcher selected",
        type(pm.process_manager).__name__ == "HerdrProcessManager",
        type(pm.process_manager).__name__,
    )

    # Anchor every artifact check to this moment: a marker or inbox line left
    # by an earlier run must not be able to satisfy this one.
    started_at = time.time()
    spawned = await server.spawn_agent(
        prompt=(
            "You are a test agent. Do exactly one thing: use the win-agent-teams "
            'MCP tool send_message with to="lead" and the single line: '
            "GRANDCHILD ALIVE. Then stop. Do not read or edit any file."
        ),
        backend="codex",
        name=CHILD,
        model="low",
        cwd=str(Path.cwd()),
    )
    say("spawn result: " + json.dumps(spawned, default=str)[:400])
    check("spawn returned a pid", bool(spawned.get("pid")), str(spawned.get("pid")))

    session_dir = Path(spawned["session_dir"])
    marker = session_dir / f"state-{CHILD}.json"

    workspaces = _result(herdr("workspace", "list")).get("workspaces") or []
    workspace_id = workspaces[0]["workspace_id"] if workspaces else ""
    tabs = json.loads(herdr("tab", "list", "--workspace", workspace_id))["result"][
        "tabs"
    ]
    check(
        "agent has its own herdr tab",
        any(CHILD in (label or "") for label in tabs),
        str(tabs),
    )

    # Read the pane only after the CLI has had a chance to paint; reading
    # immediately captures the command line we just typed, not a running TUI.
    await asyncio.sleep(PANE_PAINT_SECONDS)
    pane_id = pm.process_manager._processes[str(spawned["pid"])].pane_id
    pane_text = herdr("pane", "read", pane_id, "--source", "visible", "--lines", "40")
    say("--- pane ---\n" + pane_text[-1200:] + "\n--- end pane ---")
    # Deliberately weak on its own: the command line also contains "codex".
    # The state marker below is the real proof that the agent ran, because
    # spawn_agent writes no marker -- only a hook inside the agent does.
    check(
        "pane shows agent content",
        len(pane_text) > MIN_PANE_CHARS,
        f"{len(pane_text)}c",
    )

    # Watch OUR OWN identity's inbox, never "team-lead": this script may be
    # running inside a spawned agent, and a child sends to its PARENT's inbox.
    # Assuming team-lead is exactly the mistake CLAUDE.md warns about, and it
    # made the first run report a false negative.
    identity = os.environ.get("AGENT_NAME") or "team-lead"
    inbox = session_dir / f"inbox-{identity}.jsonl"
    say(f"watching inbox: {inbox}")

    deadline = time.monotonic() + REPORT_TIMEOUT_SECONDS
    got_marker = got_message = False
    while time.monotonic() < deadline:
        got_marker = got_marker or _marker_is_fresh(marker, started_at)
        if inbox.exists() and "GRANDCHILD ALIVE" in inbox.read_text(encoding="utf-8"):
            got_message = True
            break
        await asyncio.sleep(POLL_SECONDS)

    check("state marker written", got_marker, str(marker) if got_marker else "missing")
    if got_marker:
        say("marker: " + marker.read_text(encoding="utf-8").strip())
    check("grandchild reported upstream", got_message)

    say("kill: " + json.dumps(await server.kill_agent(name=CHILD), default=str)[:200])
    await asyncio.sleep(3)
    labels_after = _tab_labels(workspace_id)
    check(
        "kill closed the tab",
        labels_after is not None
        and not any(CHILD in (label or "") for label in labels_after),
        str(labels_after),
    )
    check("kill removed the state marker", not marker.exists())
    check("killed process is gone", not _pid_alive(int(spawned["pid"])))

    failed = [name for name, ok in _results if not ok]
    say("\nSUMMARY: " + ("ALL PASS" if not failed else f"FAILED: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_checks()))

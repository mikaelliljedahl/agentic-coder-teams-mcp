"""Linux smoke for ``spawn_agent(trust_cwd=True)`` (PR #73), after merging main.

Linux counterpart of ``trust_smoke.py``. Each step starts a FRESH
win-agent-teams MCP server over stdio with that step's environment, in its own
lead workspace. Interactive Codex agents launch through the configured Linux
launcher (herdr here). Steps:

- L0  control: no ``trust_cwd`` in an explicitly untrusted cwd -> the folder
      trust prompt blocks, so no marker and ``no_marker_since_launch=true``.
- L1  ``trust_cwd=True``, native flags on -> marker, isolated CODEX_HOME
      printed, follow-up ``delivered`` via ``codex_queue``, config unchanged.
- L1R the same with the native flags unset -> follow-up takes the resume path.
- L2  ``backend=claude-code`` -> ``trust_cwd_unsupported_backend``.
- L4  cwd containing U+2019 -> ``trust_cwd_unsafe_path``, no agent.

Usage: ``.venv/bin/python trust_smoke_linux.py [L0 L1 L1R L2 L4]``.
"""

from __future__ import annotations

import asyncio
import filecmp
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[4]
PY = REPO / ".venv" / "bin" / "python"
TEMP = Path(os.environ.get("SMOKE_TMP", tempfile.gettempdir()))
REAL_CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
FLAGS = (
    "WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH",
    "WIN_AGENT_TEAMS_NO_WT_TABS",
    "WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE",
)
IDENTITY_VARS = (
    "AGENT_NAME",
    "AGENT_SESSION_ID",
    "AGENT_PARENT_NAME",
    "WIN_AGENT_TEAMS_SESSION_DIR",
)
POLL_SECONDS = 5
RESULTS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class Setup:
    """Disposable paths shared by every step."""

    home: Path
    config: Path
    before: Path
    cwd: Path
    workspace: Path
    real_before: Path


def log(msg: str) -> None:
    """Write a timestamped progress line to stdout."""
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


def setup() -> Setup:
    """Create the isolated CODEX_HOME, the hostile cwd and the byte copies."""
    home = TEMP / f"wat-trust-{uuid.uuid4().hex}"
    home.mkdir()
    shutil.copy2(REAL_CODEX_HOME / "auth.json", home)
    shutil.copy2(REAL_CODEX_HOME / "config.toml", home)
    config = home / "config.toml"
    cwd = TEMP / "trust smoke space ;%!&^"
    cwd.mkdir(exist_ok=True)
    entries = f"[projects.'{cwd}']\ntrust_level = 'untrusted'\n"
    with config.open("a", encoding="utf-8") as fh:
        fh.write("\n" + entries)
    before = home / "config.before"
    shutil.copy2(config, before)
    real_before = TEMP / f"wat-real-config-{uuid.uuid4().hex}.toml"
    shutil.copy2(REAL_CODEX_HOME / "config.toml", real_before)
    workspace = TEMP / f"wat-trust-lead-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return Setup(home, config, before, cwd, workspace, real_before)


def same_bytes(before: Path, after: Path) -> bool:
    """Return whether the two files are byte-identical."""
    return filecmp.cmp(before, after, shallow=False)


def server_env(
    home: Path,
    flags: dict[str, str],
    path_prefix: str | None = None,
    *,
    resume: bool = False,
) -> dict[str, str]:
    """Build the MCP server environment for one step."""
    env = dict(os.environ)
    for key in (*FLAGS, *IDENTITY_VARS):
        env.pop(key, None)
    env["CODEX_HOME"] = str(home)
    env["WIN_AGENT_TEAMS_NATIVE_WAKE"] = "1"
    env["WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM"] = "1"
    if resume or os.environ.get("SMOKE_RESUME") == "1":
        env.pop("WIN_AGENT_TEAMS_NATIVE_WAKE", None)
        env.pop("WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM", None)
    env.update(flags)
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env["PATH"]
    return env


def payload(result: Any) -> dict[str, Any]:
    """Decode a tool result into a dict, keeping raw text on failure."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    text = "".join(getattr(part, "text", "") for part in result.content)
    try:
        return json.loads(text)
    except ValueError:
        return {"_raw": text, "_isError": result.isError}


async def call(session: ClientSession, tool: str, **args: Any) -> dict[str, Any]:
    """Call one MCP tool with a generous read timeout."""
    result = await session.call_tool(
        tool, args, read_timeout_seconds=timedelta(seconds=600)
    )
    return payload(result)


async def with_server(
    env: dict[str, str],
    workspace: Path,
    body: Callable[[ClientSession], Awaitable[None]],
) -> None:
    """Run ``body`` against a fresh stdio MCP server started with ``env``."""
    params = StdioServerParameters(
        command=str(PY),
        args=["-m", "claude_teams.server_simple"],
        env=env,
        cwd=str(workspace),
    )
    with (workspace / "server-stderr.log").open("a", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            await body(session)


def last_text(status: dict[str, Any]) -> str:
    """Return the agent's last message, falling back to its last line."""
    return str(status.get("last_message") or status.get("last_line") or "")


async def wait_for(
    session: ClientSession,
    name: str,
    token: str,
    limit_s: float = 300,
) -> dict[str, Any]:
    """Poll ``check_agent`` until the agent is waiting after saying ``token``."""
    deadline = time.time() + limit_s
    status: dict[str, Any] = {}
    while time.time() < deadline:
        status = await call(session, "check_agent", name=name, full=True)
        if token in last_text(status) and status.get("state") == "waiting":
            return status
        await asyncio.sleep(POLL_SECONDS)
    return status


def pick(status: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return the subset of ``status`` worth recording."""
    return {key: status.get(key) for key in keys}


async def live_step(
    step: str, flags: dict[str, str], env: Setup, *, resume: bool = False
) -> None:
    """Spawn with trust_cwd, confirm the marker and CODEX_HOME, then follow up."""
    name = step.lower()
    record: dict[str, Any] = {"step": step, "flags": flags, "pass": False}

    async def body(session: ClientSession) -> None:
        spawn = await call(
            session,
            "spawn_agent",
            prompt=(
                "Run a shell command that prints the value of the CODEX_HOME "
                "environment variable. Then reply with exactly one line: "
                "CODEX_HOME=<value> FIRST"
            ),
            name=name,
            backend="codex",
            model="cheapest",
            cwd=str(env.cwd),
            trust_cwd=True,
        )
        record["spawn"] = spawn
        log(f"{step} spawn -> {json.dumps(spawn)[:400]}")
        if spawn.get("success") is False or spawn.get("_isError"):
            return
        first = await wait_for(session, name, "FIRST")
        record["first"] = pick(
            first,
            "state",
            "binding",
            "backend_session_id",
            "no_marker_since_launch",
            "last_message",
        )
        log(f"{step} first -> {json.dumps(record['first'])}")
        home_ok = str(env.home).lower() in last_text(first).lower()
        spawn_bytes_ok = same_bytes(env.before, env.config)
        follow = await call(
            session,
            "follow_up_agent",
            name=name,
            prompt="Reply with exactly: SECOND",
            idempotency_key=f"{name}-{uuid.uuid4().hex[:6]}",
        )
        record["follow_up"] = pick(
            follow, "status", "method", "replaced_existing", "pid", "reason"
        )
        log(f"{step} follow_up -> {json.dumps(record['follow_up'])}")
        second = await wait_for(session, name, "SECOND", limit_s=240)
        record["second"] = pick(second, "state", "backend_session_id", "last_message")
        log(f"{step} second -> {json.dumps(record['second'])}")
        follow_bytes_ok = same_bytes(env.before, env.config)
        record["kill"] = await call(session, "kill_agent", name=name)
        record["pass"] = (
            "FIRST" in last_text(first)
            and home_ok
            and spawn_bytes_ok
            and follow.get("status") == "delivered"
            and "SECOND" in last_text(second)
            and follow_bytes_ok
        )

    await with_server(server_env(env.home, flags, resume=resume), env.workspace, body)
    RESULTS[step] = record
    log(f"{step} PASS={record['pass']}")


async def control_step(env: Setup, wait_s: float = 75) -> None:
    """Without trust_cwd the folder-trust prompt must block the agent."""
    name = "l0"
    record: dict[str, Any] = {"step": "L0", "pass": False}

    async def body(session: ClientSession) -> None:
        spawn = await call(
            session,
            "spawn_agent",
            prompt="Reply with exactly: FIRST",
            name=name,
            backend="codex",
            model="cheapest",
            cwd=str(env.cwd),
        )
        record["spawn"] = pick(spawn, "success", "pid", "reason")
        await asyncio.sleep(wait_s)
        status = await call(session, "check_agent", name=name, full=True)
        record["status"] = pick(
            status, "state", "no_marker_since_launch", "startup_hint", "last_message"
        )
        log(f"L0 status -> {json.dumps(record['status'])}")
        record["kill"] = await call(session, "kill_agent", name=name)
        record["pass"] = (
            status.get("no_marker_since_launch") is True
            and "FIRST" not in last_text(status)
            and same_bytes(env.before, env.config)
        )

    await with_server(server_env(env.home, {}), env.workspace, body)
    RESULTS["L0"] = record
    log(f"L0 PASS={record['pass']}")


async def refusal_step(
    step: str,
    flags: dict[str, str],
    expected: str,
    env: Setup,
    path_prefix: str | None = None,
    *,
    backend: str = "codex",
    cwd: Path | None = None,
) -> None:
    """Require a trust_cwd spawn to be refused with ``expected`` and no agent."""
    name = step.lower()
    record: dict[str, Any] = {"step": step, "flags": flags, "expected": expected}

    async def body(session: ClientSession) -> None:
        spawn = await call(
            session,
            "spawn_agent",
            prompt="Reply OK",
            name=name,
            backend=backend,
            model="cheapest" if backend == "codex" else "",
            cwd=str(cwd or env.cwd),
            trust_cwd=True,
        )
        agents = await call(session, "list_agents")
        record["spawn"] = spawn
        record["list_agents"] = agents
        record["pass"] = (
            spawn.get("success") is False
            and spawn.get("reason") == expected
            and f'"{name}"' not in json.dumps(agents)
        )

    await with_server(server_env(env.home, flags, path_prefix), env.workspace, body)
    RESULTS[step] = record
    log(f"{step} -> {json.dumps(record['spawn'])} PASS={record['pass']}")


async def main() -> int:
    """Run the selected steps (default: all) and print a pass map."""
    only = set(sys.argv[1:])
    env = setup()
    log(f"isolated CODEX_HOME={env.home} cwd={env.cwd} lead={env.workspace}")
    unsafe = TEMP / "trust smoke \u2019quote"
    unsafe.mkdir(exist_ok=True)
    steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("L0", lambda: control_step(env)),
        ("L1", lambda: live_step("L1", {}, env)),
        ("L1R", lambda: live_step("L1R", {}, env, resume=True)),
        (
            "L2",
            lambda: refusal_step(
                "L2", {}, "trust_cwd_unsupported_backend", env, backend="claude-code"
            ),
        ),
        (
            "L4",
            lambda: refusal_step("L4", {}, "trust_cwd_unsafe_path", env, cwd=unsafe),
        ),
    ]
    try:
        for step, run in steps:
            if only and step not in only:
                continue
            log(f"===== {step} =====")
            try:
                await run()
            except Exception as exc:  # keep going and report
                RESULTS[step] = {"step": step, "pass": False, "exception": repr(exc)}
                log(f"{step} EXCEPTION {exc!r}")
        real_ok = same_bytes(env.real_before, REAL_CODEX_HOME / "config.toml")
        RESULTS["real_config_unchanged"] = {"pass": real_ok}
    finally:
        out = env.workspace / "results.json"
        out.write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
        log(f"results -> {out}")
        shutil.rmtree(unsafe, ignore_errors=True)
        shutil.rmtree(env.home, ignore_errors=True)
        env.real_before.unlink(missing_ok=True)
    log(json.dumps({step: result.get("pass") for step, result in RESULTS.items()}))
    return 0 if all(result.get("pass") for result in RESULTS.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

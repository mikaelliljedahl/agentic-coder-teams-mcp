"""Relay helpers for one-shot backend output delivery."""

import asyncio
import time
from pathlib import Path

from claude_teams import messaging
from claude_teams.async_utils import run_blocking
from claude_teams.backends import registry
from claude_teams.teams import validate_safe_name
from claude_teams.server_runtime import (
    _ONE_SHOT_RESULT_MAX_CHARS,
    _ONE_SHOT_TIMEOUT_S,
    _strip_ansi,
    logger,
)


def log_relay_task_exception(task: asyncio.Task) -> None:
    """Log unhandled exceptions from one-shot relay background tasks.

    Args:
        task (asyncio.Task): Background relay task to inspect.

    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("One-shot relay task failed: %s", exc, exc_info=exc)


def build_agent_auth_notice(team_name: str, capability: str) -> str:
    """Build agent attach instructions for the bootstrap inbox message.

    Args:
        team_name (str): Team name the capability belongs to.
        capability (str): Agent capability token.

    Returns:
        str: User-facing attach instructions appended to the prompt.

    """
    return (
        "\n\n[claude-teams auth]\n"
        "Before using coordination tools from a separate MCP session, attach with:\n"
        f'team_attach(team_name="{team_name}", capability="{capability}")\n'
        "After attachment, your session will be authorized for agent-scoped actions.\n"
    )


def create_one_shot_result_path(team_name: str, agent_name: str) -> Path:
    """Create the output capture path for one-shot backends.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.

    Returns:
        Path: Output file path under the team runs directory.

    """
    safe_team_name = validate_safe_name(team_name, "team name")
    safe_agent_name = validate_safe_name(agent_name, "agent name")
    runs_dir = messaging.TEAMS_DIR / safe_team_name / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    return runs_dir / f"{safe_agent_name}-{timestamp}.last-message.txt"


async def relay_one_shot_result(
    team_name: str,
    agent_name: str,
    backend_type: str,
    process_handle: str,
    result_file: Path | None,
    color: str,
) -> None:
    """Wait for a one-shot backend to finish and relay its output.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        backend_type (str): Backend identifier.
        process_handle (str): Backend process handle.
        result_file (Path | None): Optional output file for file-based capture.
        color (str): Agent display color for inbox routing.

    """
    deadline = time.monotonic() + _ONE_SHOT_TIMEOUT_S
    backend_obj = None
    text = ""

    try:
        backend_obj = registry.get(backend_type)
    except KeyError:
        logger.warning(
            "One-shot backend not available for result relay: %s", backend_type
        )
        if result_file is None:
            await messaging.send_plain_message(
                team_name,
                agent_name,
                "team-lead",
                (
                    f"{agent_name} ({backend_type}) finished, but the backend could not "
                    "be resolved for output capture."
                ),
                summary="teammate_result",
                color=color,
            )
            return

    while time.monotonic() < deadline:
        if result_file is not None:
            try:
                if await run_blocking(result_file.exists):
                    text = (await run_blocking(result_file.read_text)).strip()
            except Exception:
                logger.exception("Failed reading one-shot result file: %s", result_file)
            if text:
                break

        if backend_obj is not None:
            status = await run_blocking(backend_obj.health_check, process_handle)
            if not status.alive:
                break

        sleep_seconds = min(0.5, max(0.0, deadline - time.monotonic()))
        if sleep_seconds == 0.0:
            break
        await asyncio.sleep(sleep_seconds)

    if not text and result_file is not None:
        try:
            if await run_blocking(result_file.exists):
                text = (await run_blocking(result_file.read_text)).strip()
        except Exception:
            logger.exception("Failed reading one-shot result file: %s", result_file)

    if not text and backend_obj is not None:
        try:
            captured = await run_blocking(backend_obj.capture, process_handle)
            text = _strip_ansi(captured).strip()
        except Exception:
            logger.debug(
                "Failed to capture pane output for %s: %s",
                agent_name,
                process_handle,
                exc_info=True,
            )

    if not text and time.monotonic() >= deadline:
        await messaging.send_plain_message(
            team_name,
            agent_name,
            "team-lead",
            f"{agent_name} timed out before producing output.",
            summary="teammate_timeout",
            color=color,
        )
        return

    if not text:
        text = f"{agent_name} ({backend_type}) finished, but no output was captured."

    if len(text) > _ONE_SHOT_RESULT_MAX_CHARS:
        text = text[:_ONE_SHOT_RESULT_MAX_CHARS] + "\n\n[truncated]"

    await messaging.send_plain_message(
        team_name,
        agent_name,
        "team-lead",
        text,
        summary="teammate_result",
        color=color,
    )

    if result_file is not None:
        try:
            await run_blocking(result_file.unlink, missing_ok=True)
        except OSError:
            pass

    if backend_obj is not None:
        try:
            await run_blocking(backend_obj.kill, process_handle)
        except Exception:
            pass

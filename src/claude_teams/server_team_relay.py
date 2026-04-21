"""Relay helpers for one-shot backend output delivery."""

import asyncio
import contextlib
import subprocess
import time
from pathlib import Path

from claude_teams import messaging
from claude_teams.async_utils import run_blocking
from claude_teams.backends import registry
from claude_teams.backends.base import Backend
from claude_teams.errors import BackendNotRegisteredError
from claude_teams.server_runtime import (
    _ONE_SHOT_RESULT_MAX_CHARS,
    _ONE_SHOT_TIMEOUT_S,
    _strip_ansi,
    logger,
)
from claude_teams.teams import validate_safe_name


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


def log_retain_pane_failure(exc: BaseException) -> None:
    """Log non-fatal failures from ``Backend.retain_pane_after_exit``.

    ``retain_pane_after_exit`` runs after a one-shot backend spawn
    returns; when it fails the spawn itself has already succeeded, so
    the pane-retention error is recorded at ``warning`` level (not
    ``error``) — it is operational breadcrumb, not a reason to fail
    the user-facing call. Injected via ``SpawnDependencies`` so the
    orchestration core never imports this logger directly.

    Args:
        exc: The caught exception raised by ``retain_pane_after_exit``.

    """
    logger.warning("retain_pane_after_exit failed: %s", exc, exc_info=exc)


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


async def _read_result_file(result_file: Path) -> str:
    """Read the one-shot output file if present, returning its trimmed contents.

    Args:
        result_file (Path): File path written by the backend.

    Returns:
        str: Trimmed file contents, or "" if the file is absent or unreadable.

    """
    try:
        if await run_blocking(result_file.exists):
            return (await run_blocking(result_file.read_text)).strip()
    except Exception:
        logger.exception("Failed reading one-shot result file: %s", result_file)
    return ""


async def _capture_pane_output(
    backend_obj: Backend,
    process_handle: str,
    agent_name: str,
) -> str:
    """Capture pane output from the backend as an ANSI-stripped string.

    Args:
        backend_obj (Backend): Backend to capture from.
        process_handle (str): Backend process handle.
        agent_name (str): Agent name for debug logging.

    Returns:
        str: Captured pane text, or "" on failure.

    """
    try:
        captured = await run_blocking(backend_obj.capture, process_handle)
        return _strip_ansi(captured).strip()
    except Exception:
        logger.debug(
            "Failed to capture pane output for %s: %s",
            agent_name,
            process_handle,
            exc_info=True,
        )
    return ""


async def _poll_for_one_shot_output(
    backend_obj: Backend | None,
    result_file: Path | None,
    process_handle: str,
    deadline: float,
) -> str:
    """Poll the result file and backend health until output or deadline.

    The loop exits early in three cases: the file yielded text, the backend
    reported itself dead, or the remaining budget dropped to zero. When
    ``backend_obj`` is ``None`` (registry lookup failed earlier), the health
    check is skipped — the loop relies entirely on the file path.

    Args:
        backend_obj (Backend | None): Backend for health checks, or ``None``.
        result_file (Path | None): Optional output file.
        process_handle (str): Backend process handle for health checks.
        deadline (float): Monotonic deadline for polling.

    Returns:
        str: Collected output text, or "" if the loop ends without output.

    """
    text = ""
    while time.monotonic() < deadline:
        if result_file is not None:
            text = await _read_result_file(result_file)
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
    return text


async def relay_one_shot_result(
    team_name: str,
    agent_name: str,
    backend_type: str,
    process_handle: str,
    result_file: Path | None,
    color: str,
) -> None:
    """Wait for a one-shot backend to finish and relay its output.

    Orchestrates three phases via extracted helpers: registry resolution,
    polling, and fallback capture. Final steps (deadline check, send, and
    cleanup) remain inline because each is a terminal decision tied to the
    team-lead mailbox lifecycle.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        backend_type (str): Backend identifier.
        process_handle (str): Backend process handle.
        result_file (Path | None): Optional output file for file-based capture.
        color (str): Agent display color for inbox routing.

    """
    deadline = time.monotonic() + _ONE_SHOT_TIMEOUT_S
    backend_obj: Backend | None = None

    try:
        backend_obj = registry.get(backend_type)
    except BackendNotRegisteredError:
        logger.warning(
            "One-shot backend not available for result relay: %s", backend_type
        )
        if result_file is None:
            await messaging.send_plain_message(
                team_name,
                agent_name,
                "team-lead",
                (
                    f"{agent_name} ({backend_type}) finished, but the backend "
                    "could not be resolved for output capture."
                ),
                summary="teammate_result",
                color=color,
            )
            return

    text = await _poll_for_one_shot_output(
        backend_obj, result_file, process_handle, deadline
    )

    if not text and result_file is not None:
        text = await _read_result_file(result_file)

    if not text and backend_obj is not None:
        text = await _capture_pane_output(backend_obj, process_handle, agent_name)

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
        with contextlib.suppress(OSError):
            await run_blocking(result_file.unlink, missing_ok=True)

    if backend_obj is not None:
        try:
            await run_blocking(backend_obj.kill, process_handle)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "Failed to kill backend process %r after relay: %s",
                process_handle,
                exc,
            )

"""Read fallback output from Codex and Claude Code session logs."""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_MTIME_SLACK_SECONDS = 2.0
_REVERSE_READ_CHUNK_SIZE = 64 * 1024
_CODEX_CORRELATION_PREFIX = "wat-corr:"
_CORRELATION_SCAN_MAX_LINES = 500


def codex_correlation_token(agent_id: str) -> str:
    """Return the stable per-agent marker embedded in the Codex prompt.

    Two Codex agents spawned in the same ``cwd`` at nearly the same time are
    otherwise indistinguishable before Codex's own session id is known
    (matching falls back to cwd + start-time + mtime). The codex backend
    appends this token to the initial prompt so the agent's rollout file can
    be bound deterministically to the right logical agent.
    """
    return f"{_CODEX_CORRELATION_PREFIX}{agent_id}"


@dataclass(frozen=True)
class AgentOutput:
    """Latest assistant output found in an agent rollout file."""

    last_activity_at: float
    last_message: str | None
    rollout_path: str
    backend_session_id: str | None = None
    busy_hint: bool = False


def read_codex_output(
    spawned_at: float,
    cwd: str,
    max_bytes: int = 10_240,
    *,
    backend_session_id: str | None = None,
    correlation_token: str | None = None,
) -> AgentOutput | None:
    """Read the latest Codex assistant output for a spawned agent."""
    if spawned_at <= 0 or not cwd:
        return None

    normalized_cwd = _normalize_path(cwd)
    if not normalized_cwd:
        return None

    candidates = _matching_codex_rollouts(
        spawned_at, normalized_cwd, backend_session_id, correlation_token
    )
    if not candidates:
        return None

    mtime, path, backend_session_id = max(candidates, key=lambda item: item[0])
    message = _last_codex_message(path)
    if message is None and backend_session_id is None:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_utf8(message, max_bytes) if message else None,
        rollout_path=str(path),
        backend_session_id=backend_session_id,
    )


def read_claude_output(
    spawned_at: float,
    cwd: str,
    max_bytes: int = 10_240,
    *,
    backend_session_id: str | None = None,
) -> AgentOutput | None:
    """Read the latest Claude Code assistant output for a spawned agent."""
    if spawned_at <= 0 or not cwd:
        return None

    resolved_cwd = _resolve_path_text(cwd)
    if not resolved_cwd:
        return None

    encoded_cwd = _encode_claude_cwd(resolved_cwd)
    project_dir = Path.home() / ".claude" / "projects" / encoded_cwd
    candidates = []
    for mtime, path in _matching_jsonl_files(project_dir, spawned_at):
        session_id = _claude_session_id(path)
        if backend_session_id:
            if session_id == backend_session_id:
                candidates.append((mtime, path))
            continue
        if _started_after(_claude_started_at(path), spawned_at):
            candidates.append((mtime, path))
    if not candidates:
        return None

    mtime, path = max(candidates, key=lambda item: item[0])
    backend_session_id = _claude_session_id(path)
    message = _last_claude_message(path)
    if message is None and backend_session_id is None:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_utf8(message, max_bytes) if message else None,
        rollout_path=str(path),
        backend_session_id=backend_session_id,
    )


def _matching_codex_rollouts(
    spawned_at: float,
    normalized_cwd: str,
    backend_session_id: str | None,
    correlation_token: str | None = None,
) -> list[tuple[float, Path, str | None]]:
    candidates: list[tuple[float, Path, str | None]] = []
    for directory in _codex_candidate_dirs(
        spawned_at, include_all=bool(backend_session_id)
    ):
        for mtime, path in _matching_jsonl_files(
            directory, spawned_at, pattern="rollout-*.jsonl"
        ):
            meta = _first_json_object(path)
            if not isinstance(meta, dict) or meta.get("type") != "session_meta":
                continue
            payload = meta.get("payload")
            if not isinstance(payload, dict):
                continue
            session_id = payload.get("id")
            if backend_session_id:
                if session_id != backend_session_id:
                    continue
            elif not _started_after(
                _parse_timestamp(payload.get("timestamp")), spawned_at
            ):
                continue
            meta_cwd = payload.get("cwd")
            if (
                isinstance(meta_cwd, str)
                and _normalize_path(meta_cwd) == normalized_cwd
            ):
                candidates.append(
                    (mtime, path, session_id if isinstance(session_id, str) else None)
                )
    # Before Codex's own session id is known, cwd + start-time matching cannot
    # tell two concurrently-spawned agents apart. If a correlation token was
    # injected into the prompt, prefer the rollout that actually contains it so
    # the binding is deterministic. Fall back to the unfiltered set when no
    # rollout carries the token yet (e.g. Codex has not flushed the prompt, or
    # the agent was spawned before this marker existed).
    if backend_session_id is None and correlation_token:
        token_matched = [
            item
            for item in candidates
            if _rollout_contains_token(item[1], correlation_token)
        ]
        if token_matched:
            return token_matched
    return candidates


def _rollout_contains_token(
    path: Path, token: str, max_lines: int = _CORRELATION_SCAN_MAX_LINES
) -> bool:
    """Return whether ``token`` appears in the first ``max_lines`` of a rollout.

    The token is embedded in the initial user prompt, which Codex records near
    the start of the rollout. A bounded forward scan keeps this cheap and
    avoids reading large rollout files in full.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle):
                if index >= max_lines:
                    return False
                if token in raw:
                    return True
    except OSError:
        return False
    return False


def _matching_jsonl_files(
    directory: Path, spawned_at: float, *, pattern: str = "*.jsonl"
) -> list[tuple[float, Path]]:
    if not directory.exists():
        return []

    cutoff = spawned_at - _MTIME_SLACK_SECONDS
    matches: list[tuple[float, Path]] = []
    with contextlib.suppress(OSError):
        for path in directory.glob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                matches.append((mtime, path))
    return matches


def _codex_candidate_dirs(
    spawned_at: float, *, include_all: bool = False
) -> list[Path]:
    try:
        utc_time = datetime.fromtimestamp(spawned_at, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return []

    roots = (utc_time, utc_time.astimezone())
    days = {
        (dt + timedelta(days=offset)).date() for dt in roots for offset in (-1, 0, 1)
    }
    base = Path.home() / ".codex" / "sessions"
    directories = [
        base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        for day in sorted(days)
    ]
    if include_all:
        with contextlib.suppress(OSError):
            directories.extend(path for path in base.glob("*/*/*") if path.is_dir())
    return list(dict.fromkeys(directories))


def _first_json_object(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    continue
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _last_codex_message(path: Path) -> str | None:
    for line in _iter_lines_reverse(path):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("type") != "response_item":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict) or payload.get("role") != "assistant":
            continue
        text = _content_text(payload.get("content"), "output_text")
        if text is not None:
            return text
    return None


def _last_claude_message(path: Path) -> str | None:
    for line in _iter_lines_reverse(path):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("type") != "assistant":
            continue
        message = item.get("message")
        if not isinstance(message, dict):
            continue
        text = _content_text(message.get("content"), "text", allow_string=True)
        if text is not None:
            return text
    return None


def _claude_session_id(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                session_id = item.get("sessionId")
                if isinstance(session_id, str):
                    return session_id
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _claude_started_at(path: Path) -> float | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                started_at = _parse_timestamp(item.get("timestamp"))
                if started_at is not None:
                    return started_at
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _started_after(started_at: float | None, spawned_at: float) -> bool:
    if started_at is None:
        return True
    return started_at >= spawned_at - _MTIME_SLACK_SECONDS


def _parse_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _content_text(
    content: object, text_type: str, *, allow_string: bool = False
) -> str | None:
    if allow_string and isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == text_type
        and isinstance(item.get("text"), str)
    ]
    if not parts:
        return None
    return "".join(parts)


def _iter_lines_reverse(path: Path) -> Iterator[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""
            while position > 0:
                read_size = min(_REVERSE_READ_CHUNK_SIZE, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
                lines = buffer.split(b"\n")
                buffer = lines[0]
                for raw in reversed(lines[1:]):
                    line = raw.rstrip(b"\r")
                    if line.strip():
                        yield line.decode("utf-8", errors="replace")
            if buffer.strip():
                yield buffer.rstrip(b"\r").decode("utf-8", errors="replace")
    except OSError:
        return


def _truncate_utf8(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _normalize_path(value: str) -> str:
    resolved = _resolve_path_text(value)
    if not resolved:
        return ""
    return os.path.normcase(os.path.normpath(resolved))


def _resolve_path_text(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return str(Path(value).expanduser())


def _encode_claude_cwd(cwd: str) -> str:
    return re.sub(r"[\\/:]", "-", cwd)

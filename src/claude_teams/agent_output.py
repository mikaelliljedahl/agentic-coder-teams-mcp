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


@dataclass(frozen=True)
class AgentOutput:
    """Latest assistant output found in an agent rollout file."""

    last_activity_at: float
    last_message: str
    rollout_path: str


def read_codex_output(
    spawned_at: float, cwd: str, max_bytes: int = 10_240
) -> AgentOutput | None:
    """Read the latest Codex assistant output for a spawned agent."""
    if spawned_at <= 0 or not cwd:
        return None

    normalized_cwd = _normalize_path(cwd)
    if not normalized_cwd:
        return None

    candidates = _matching_codex_rollouts(spawned_at, normalized_cwd)
    if not candidates:
        return None

    mtime, path = max(candidates, key=lambda item: item[0])
    message = _last_codex_message(path)
    if message is None:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_utf8(message, max_bytes),
        rollout_path=str(path),
    )


def read_claude_output(
    spawned_at: float, cwd: str, max_bytes: int = 10_240
) -> AgentOutput | None:
    """Read the latest Claude Code assistant output for a spawned agent."""
    if spawned_at <= 0 or not cwd:
        return None

    resolved_cwd = _resolve_path_text(cwd)
    if not resolved_cwd:
        return None

    encoded_cwd = _encode_claude_cwd(resolved_cwd)
    project_dir = Path.home() / ".claude" / "projects" / encoded_cwd
    candidates = _matching_jsonl_files(project_dir, spawned_at)
    if not candidates:
        return None

    mtime, path = max(candidates, key=lambda item: item[0])
    message = _last_claude_message(path)
    if message is None:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_utf8(message, max_bytes),
        rollout_path=str(path),
    )


def _matching_codex_rollouts(
    spawned_at: float, normalized_cwd: str
) -> list[tuple[float, Path]]:
    candidates: list[tuple[float, Path]] = []
    for directory in _codex_candidate_dirs(spawned_at):
        for mtime, path in _matching_jsonl_files(
            directory, spawned_at, pattern="rollout-*.jsonl"
        ):
            meta = _first_json_object(path)
            if not isinstance(meta, dict) or meta.get("type") != "session_meta":
                continue
            payload = meta.get("payload")
            if not isinstance(payload, dict):
                continue
            meta_cwd = payload.get("cwd")
            if (
                isinstance(meta_cwd, str)
                and _normalize_path(meta_cwd) == normalized_cwd
            ):
                candidates.append((mtime, path))
    return candidates


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


def _codex_candidate_dirs(spawned_at: float) -> list[Path]:
    try:
        utc_time = datetime.fromtimestamp(spawned_at, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return []

    roots = (utc_time, utc_time.astimezone())
    days = {
        (dt + timedelta(days=offset)).date() for dt in roots for offset in (-1, 0, 1)
    }
    base = Path.home() / ".codex" / "sessions"
    return [
        base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        for day in sorted(days)
    ]


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

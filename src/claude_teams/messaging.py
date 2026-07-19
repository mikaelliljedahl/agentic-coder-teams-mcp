"""Pure helpers for reading the disk-backed JSONL inbox protocol."""

import json
import os
import uuid
from pathlib import Path


def load_inbox_cursors(path: Path) -> dict[str, int]:
    """Load valid non-negative per-sender cursor counts from ``path``."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in value.items()
        if isinstance(key, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }


def save_inbox_cursors(path: Path, cursors: dict[str, int]) -> None:
    """Atomically persist per-sender cursor counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cursors), encoding="utf-8")
    tmp.replace(path)


def read_inbox_by_sender(path: Path) -> dict[str, list[tuple[int, dict]]]:
    """Group valid inbox messages by sender while retaining global positions."""
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    by_sender: dict[str, list[tuple[int, dict]]] = {}
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        sender = message.get("from")
        if not isinstance(sender, str) or not sender:
            continue
        by_sender.setdefault(sender, []).append((index, message))
    return by_sender


def unread_sender_counts(inbox_path: Path, cursor_path: Path) -> dict[str, int]:
    """Return positive unread counts per sender without advancing cursors."""
    by_sender = read_inbox_by_sender(inbox_path)
    cursors = load_inbox_cursors(cursor_path)
    result: dict[str, int] = {}
    for sender, messages in by_sender.items():
        total = len(messages)
        consumed = min(cursors.get(sender, 0), total)
        unread = total - consumed
        if unread:
            result[sender] = unread
    return result

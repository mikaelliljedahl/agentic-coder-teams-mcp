"""JSONL event logging for Windows agent team activity."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_teams.filelock import file_lock
from claude_teams.teams import validate_safe_name


def _now_iso() -> str:
    """Return a UTC ISO timestamp for event records."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def event_log_path(team_name: str, base_dir: Path | None = None) -> Path:
    """Return the JSONL event log path for a team."""
    safe_team_name = validate_safe_name(team_name, "team name")
    teams_dir = (base_dir / "teams") if base_dir else Path.home() / ".claude" / "teams"
    return teams_dir / safe_team_name / "events.jsonl"


def log_event(team_name: str, event: str, **payload: Any) -> None:
    """Append an event record to ``events.jsonl``.

    Event logging is best-effort and should never be used as the commit point
    for team state. Callers intentionally do not catch failures here; the
    function keeps its own failure surface small by creating directories and
    using the same file lock helper as inbox/task state.
    """
    path = event_log_path(team_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now_iso(), "event": event, **payload}
    lock_path = path.parent / ".events.lock"
    with file_lock(lock_path), path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

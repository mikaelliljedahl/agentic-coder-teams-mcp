"""Read fallback output from Codex and Claude Code session logs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

_MTIME_SLACK_SECONDS = 2.0
_REVERSE_READ_CHUNK_SIZE = 64 * 1024
_CODEX_CORRELATION_PREFIX = "wat-corr:"
_CORRELATION_SCAN_MAX_LINES = 500
_LAST_MESSAGE_BUDGET = 1000  # max chars returned for last_message (marker included)


_CORRELATION_MARKER_TEMPLATE = (
    "[win-agent-teams correlation id: {token} — internal marker, ignore this line]"
)

#: Correlation field present and usable — the record can be bound.
CORRELATION_VALID = "valid"
#: Correlation field absent — the record predates correlation. Compatibility
#: case only; the id must never be re-derived to fill the gap.
CORRELATION_LEGACY = "legacy"
#: Correlation field present but empty, blank, or of the wrong type — corrupt.
CORRELATION_UNVERIFIED = "unverified"

CORRELATION_FIELD = "correlation_id"

#: R2/C1 — the agent-record field naming who spawned this agent. Written at
#: spawn from the spawning server's ``IDENTITY`` and preserved verbatim across
#: every resume. Its absence is meaningful: a record written before C1 shipped
#: cannot be backfilled and must be refused rather than silently allowed.
SPAWNED_BY_FIELD = "spawned_by"
#: How the parentage in :data:`SPAWNED_BY_FIELD` was established.
SPAWNED_BY_SOURCE_FIELD = "spawned_by_source"
#: Derived from the spawning call itself — the normal case.
SPAWNED_BY_SOURCE_SPAWN = "spawn"
#: Asserted by an operator through the CLI recovery path, not observed at
#: spawn. Recorded distinctly so a later reader can tell the two apart.
SPAWNED_BY_SOURCE_OPERATOR = "operator_asserted"


def new_correlation_id() -> str:
    """Return a fresh per-spawn correlation id.

    Generated once per spawn rather than derived from agent name + session id:
    a killed agent's name can be reused once its record is removed, so a
    derived token could identify two different conversations.
    """
    return uuid.uuid4().hex


def correlation_marker_token(correlation_id: str) -> str:
    """Return the scannable token form of a correlation id."""
    return f"{_CODEX_CORRELATION_PREFIX}{correlation_id}"


def correlation_marker(correlation_id: str) -> str:
    """Return the human-readable marker line embedded in an agent's prompt."""
    return _CORRELATION_MARKER_TEMPLATE.format(
        token=correlation_marker_token(correlation_id)
    )


def correlated_prompt(prompt: str, correlation_id: str, *, single_line: bool) -> str:
    """Append the correlation marker to ``prompt``.

    ``single_line`` selects the argv form, joined by a space so the result
    introduces no CLI-sensitive character. The multi-line form is used for
    content that reaches the agent through a file (the Claude prompt sidecar)
    or through a backend that quotes the prompt verbatim (Codex).
    """
    separator = " " if single_line else "\n\n"
    return f"{prompt}{separator}{correlation_marker(correlation_id)}"


def classify_correlation(record: Mapping[str, object]) -> tuple[str, str | None]:
    """Classify an agent record's correlation field.

    Returns ``(status, correlation_id)``. The absent and malformed cases are
    deliberately distinct: **absent** means the record predates correlation
    (``legacy``), **present but unusable** means it is corrupt
    (``unverified``). Only the first is a compatibility case, and neither is
    ever resolved by silently re-deriving an id.
    """
    if CORRELATION_FIELD not in record:
        return CORRELATION_LEGACY, None
    value = record[CORRELATION_FIELD]
    if not isinstance(value, str) or not value.strip():
        return CORRELATION_UNVERIFIED, None
    return CORRELATION_VALID, value


#: Bumped whenever the binding grammar changes in a way that makes previously
#: cached bindings untrustworthy. A cache entry stamped with an older version
#: is discarded rather than reused.
BINDING_GRAMMAR_VERSION = 1

#: The stored/scanned transcript was positively identified as this agent's.
BINDING_BOUND = "bound"
#: Claude sidecar spawn whose read receipt has not landed yet — retriable.
BINDING_PENDING = "pending"
#: No trustworthy binding exists and none is expected to appear — terminal.
BINDING_UNVERIFIED = "unverified"
#: More than one transcript carries the token; guessing is not allowed.
BINDING_AMBIGUOUS = "ambiguous"
#: Record predates correlation. Read-only use is fine; follow-up refuses (R8).
BINDING_LEGACY = "legacy"
#: A candidate could not be scanned (I/O error) — retriable.
BINDING_INDETERMINATE = "indeterminate"

#: Outcomes a caller may usefully retry. Retrying a terminal outcome spins
#: forever; giving up on a retriable one fails a spawn that was about to bind.
RETRIABLE_BINDING_OUTCOMES = frozenset({BINDING_PENDING, BINDING_INDETERMINATE})
TERMINAL_BINDING_OUTCOMES = frozenset(
    {BINDING_UNVERIFIED, BINDING_AMBIGUOUS, BINDING_LEGACY}
)

#: How long after a Claude sidecar spawn zero matches still counts as
#: ``pending`` rather than ``unverified``.
DEFAULT_SIDECAR_PENDING_WINDOW_S = 120.0

#: Record field recording which transport carried the initial prompt.
PROMPT_TRANSPORT_FIELD = "prompt_transport"
PROMPT_TRANSPORT_SIDECAR = "sidecar"

_HEADER_HASH_BYTES = 4096


@dataclass(frozen=True)
class AgentOutput:
    """Latest assistant output found in an agent rollout file."""

    last_activity_at: float
    last_message: str | None
    rollout_path: str
    backend_session_id: str | None = None


@dataclass(frozen=True)
class BindingResult:
    """Outcome of resolving an agent record to a concrete transcript.

    ``outcome`` is one of the ``BINDING_*`` constants. ``output`` is populated
    only for ``bound`` and ``legacy``: the other four outcomes deliberately
    carry no transcript-derived data, because there is no transcript we are
    entitled to attribute to this agent.
    """

    outcome: str
    output: AgentOutput | None = None

    @property
    def bound(self) -> bool:
        """Whether the record was positively bound to a transcript."""
        return self.outcome == BINDING_BOUND

    @property
    def retriable(self) -> bool:
        """Whether a caller should retry rather than treat this as final."""
        return self.outcome in RETRIABLE_BINDING_OUTCOMES


def read_codex_output(
    spawned_at: float,
    cwd: str,
    max_bytes: int = _LAST_MESSAGE_BUDGET,
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
        last_message=_truncate_tail(message, max_bytes) if message else None,
        rollout_path=str(path),
        backend_session_id=backend_session_id,
    )


def read_claude_output(
    spawned_at: float,
    cwd: str,
    max_bytes: int = _LAST_MESSAGE_BUDGET,
    *,
    backend_session_id: str | None = None,
    correlation_token: str | None = None,
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
    # Before Claude's own session id is known, cwd + start-time matching cannot
    # tell a spawned agent's transcript apart from any other Claude session in
    # the same project dir. The server embeds a per-spawn correlation marker in
    # the initial prompt, so prefer the transcript that actually carries it.
    # Falls back to the unfiltered set when no transcript shows the marker yet.
    if backend_session_id is None and correlation_token:
        token_matched = [
            item
            for item in candidates
            if _file_contains_token(item[1], correlation_token)
        ]
        if token_matched:
            candidates = token_matched
    if not candidates:
        return None

    mtime, path = max(candidates, key=lambda item: item[0])
    backend_session_id = _claude_session_id(path)
    message = _last_claude_message(path)
    if message is None and backend_session_id is None:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_tail(message, max_bytes) if message else None,
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
            if _file_contains_token(item[1], correlation_token)
        ]
        if token_matched:
            return token_matched
    return candidates


def _scan_token(
    path: Path, token: str, max_lines: int = _CORRELATION_SCAN_MAX_LINES
) -> bool | None:
    """Scan the head of a log for ``token``.

    Returns ``True`` (present), ``False`` (definitely absent) or ``None``
    (the scan could not complete). The third case is the point of this
    function: an unreadable candidate is *not* evidence of absence, and
    collapsing it into ``False`` is what lets a failed read masquerade as a
    confident "this is not the agent's transcript".

    The token is embedded in the initial user prompt, which both Codex and
    Claude Code record near the start of their session logs. A bounded forward
    scan keeps this cheap and avoids reading large files in full.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle):
                if index >= max_lines:
                    return False
                if token in raw:
                    return True
    except OSError:
        return None
    return False


def _file_contains_token(
    path: Path, token: str, max_lines: int = _CORRELATION_SCAN_MAX_LINES
) -> bool:
    """Boolean view of :func:`_scan_token` for the pre-ladder legacy readers."""
    return _scan_token(path, token, max_lines) is True


def _matching_jsonl_files(
    directory: Path,
    spawned_at: float,
    *,
    pattern: str = "*.jsonl",
    strict: bool = False,
) -> list[tuple[float, Path]]:
    """Return ``(mtime, path)`` for transcripts inside the spawn's mtime window.

    ``strict`` propagates an enumeration ``OSError`` instead of swallowing it.
    The validation ladder needs that distinction: a directory listing that
    failed tells us nothing about whether a matching transcript exists, and
    reporting it as an empty candidate set turns "we could not look" into a
    terminal "there is nothing there".
    """
    if not directory.exists():
        return []

    cutoff = spawned_at - _MTIME_SLACK_SECONDS
    matches: list[tuple[float, Path]] = []
    try:
        for path in directory.glob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                matches.append((mtime, path))
    except OSError:
        if strict:
            raise
    return matches


def _codex_candidate_dirs(
    spawned_at: float, *, include_all: bool = False, strict: bool = False
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
        try:
            directories.extend(path for path in base.glob("*/*/*") if path.is_dir())
        except OSError:
            if strict:
                raise
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
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        mapping = cast("dict[str, object]", item)
        if mapping.get("type") != text_type:
            continue
        text = mapping.get("text")
        if isinstance(text, str):
            parts.append(text)
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


def _truncate_tail(text: str, budget: int = _LAST_MESSAGE_BUDGET) -> str:
    """Return at most ``budget`` characters, keeping the tail of ``text``.

    Truncation is character/code-point based (``str`` slicing) and never splits
    a UTF-8 byte sequence; it may split a grapheme cluster, which is acceptable.
    When the text is truncated, an ASCII English marker is prepended and counted
    against the budget, so ``len(result) <= budget`` holds for every positive
    budget. When the budget is too small to fit a marker, a raw tail of
    ``budget`` characters is returned instead.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = f"[truncated: showing last {{n}} of {len(text)} chars]\n"
    rendered = marker.format(n=budget)  # upper bound on marker length
    if len(rendered) >= budget:
        # Budget too small for a marker -> return a raw tail.
        return text[-budget:]
    tail_budget = budget - len(rendered)
    tail = text[-tail_budget:]
    return marker.format(n=len(tail)) + tail


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


# ---------------------------------------------------------------------------
# A2 — explicit validation ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CacheEntry:
    """A previously validated binding, plus everything needed to revalidate it.

    ``(mtime_ns, size)`` is only a change detector, so identity is stored as
    the OS file id (device + inode, where the platform provides one) *plus* a
    hash of a fixed-length header prefix. The header length is frozen at the
    size seen when the entry was written, so a pure append leaves the hash
    unchanged while a rewrite or truncation does not.
    """

    path: str
    device: int
    inode: int
    size: int
    header_size: int
    header_hash: str
    session_id: str
    grammar_version: int


_BINDING_CACHE: dict[tuple[str, str, str, str], _CacheEntry] = {}


def clear_binding_cache() -> None:
    """Drop every cached binding. Used by tests and by explicit invalidation."""
    _BINDING_CACHE.clear()


def binding_cache_size() -> int:
    """Return how many validated bindings are currently cached."""
    return len(_BINDING_CACHE)


def _header_digest(path: Path, header_size: int) -> str | None:
    """Return a hash of the first ``header_size`` bytes, or ``None`` on error."""
    if header_size <= 0:
        return ""
    try:
        with path.open("rb") as handle:
            data = handle.read(header_size)
    except OSError:
        return None
    if len(data) < header_size:
        # The file shrank below the header we hashed: a truncation or rewrite.
        return None
    return hashlib.sha256(data).hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int] | None:
    """Return ``(device, inode, size)`` for ``path``, or ``None`` on error."""
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_dev, info.st_ino, info.st_size


def _cache_entry_for(path: Path, session_id: str) -> _CacheEntry | None:
    identity = _stat_identity(path)
    if identity is None:
        return None
    device, inode, size = identity
    header_size = min(_HEADER_HASH_BYTES, size)
    digest = _header_digest(path, header_size)
    if digest is None:
        return None
    return _CacheEntry(
        path=str(path),
        device=device,
        inode=inode,
        size=size,
        header_size=header_size,
        header_hash=digest,
        session_id=session_id,
        grammar_version=BINDING_GRAMMAR_VERSION,
    )


def _cache_entry_still_valid(entry: _CacheEntry, parsed_session_id: str | None) -> bool:
    """Revalidate a cache entry against the file it was written from."""
    if entry.grammar_version != BINDING_GRAMMAR_VERSION:
        return False
    path = Path(entry.path)
    identity = _stat_identity(path)
    if identity is None:  # path disappeared
        return False
    device, inode, size = identity
    if (device, inode) != (entry.device, entry.inode):
        return False
    if size < entry.size:  # truncation
        return False
    if _header_digest(path, entry.header_size) != entry.header_hash:
        return False
    return parsed_session_id == entry.session_id


class _TranscriptBinder:
    """Per-backend transcript enumeration for the validation ladder."""

    def __init__(self, spawned_at: float, cwd: str) -> None:
        self.spawned_at = spawned_at
        self.cwd = cwd

    @property
    def cache_scope(self) -> str:
        raise NotImplementedError

    def resolve_by_session_id(self, session_id: str) -> Path | None:
        """Tier 1: open the stored session's transcript directly, no cutoff."""
        raise NotImplementedError

    def candidates(self, *, all_history: bool) -> list[Path]:
        """Tier 2: candidate transcripts, optionally ignoring the mtime window."""
        raise NotImplementedError

    def session_id(self, path: Path) -> str | None:
        raise NotImplementedError

    def last_message(self, path: Path) -> str | None:
        raise NotImplementedError

    def legacy_read(
        self, backend_session_id: str | None, max_bytes: int
    ) -> AgentOutput | None:
        raise NotImplementedError

    def scan(
        self, token: str, *, all_history: bool, extra: Path | None = None
    ) -> tuple[list[Path], bool]:
        """Return ``(matches, incomplete)`` for the token across candidates.

        ``extra`` is the tier-1 transcript (the stored session id resolved to a
        path). It joins the candidate set rather than short-circuiting it: a
        tier-1 hit alone cannot answer the count gate, because a *second*
        transcript may also carry the token, and that is ``ambiguous`` rather
        than a licence to keep the stored binding.

        ``incomplete`` short-circuits the count gate: a match count computed
        from a partially failed scan is not a count we are entitled to use.
        """
        try:
            paths = list(self.candidates(all_history=all_history))
        except OSError:
            # Enumeration failed, so there is no candidate set to count. Gate 2
            # short-circuits on ``incomplete`` precisely so that a count is
            # never computed from a scan that did not finish.
            return [], True
        if extra is not None:
            paths.append(extra)
        matches: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            hit = _scan_token(path, token)
            if hit is None:
                return [], True
            if hit:
                matches.append(path)
        return matches, False


class _ClaudeBinder(_TranscriptBinder):
    def __init__(self, spawned_at: float, cwd: str) -> None:
        super().__init__(spawned_at, cwd)
        resolved = _resolve_path_text(cwd)
        self.project_dir = (
            Path.home() / ".claude" / "projects" / _encode_claude_cwd(resolved)
            if resolved
            else None
        )
        self._scope = _normalize_path(cwd)

    @property
    def cache_scope(self) -> str:
        return self._scope

    def _all_transcripts(self) -> list[Path]:
        """Every transcript in the project dir. Raises ``OSError`` if unlistable.

        A missing project dir is genuine absence; a failed listing is not, and
        the callers turn the latter into ``indeterminate``.
        """
        if self.project_dir is None or not self.project_dir.exists():
            return []
        return sorted(self.project_dir.glob("*.jsonl"))

    def resolve_by_session_id(self, session_id: str) -> Path | None:
        if self.project_dir is None:
            return None
        # Claude names a transcript after its session id, so try that first:
        # one open instead of a directory walk. Fall back to parsing, because
        # the name is a convention rather than a contract.
        direct = self.project_dir / f"{session_id}.jsonl"
        if direct.exists() and _claude_session_id(direct) == session_id:
            return direct
        for path in self._all_transcripts():
            if _claude_session_id(path) == session_id:
                return path
        return None

    def candidates(self, *, all_history: bool) -> list[Path]:
        if self.project_dir is None:
            return []
        if all_history:
            return self._all_transcripts()
        return [
            path
            for _mtime, path in _matching_jsonl_files(
                self.project_dir, self.spawned_at, strict=True
            )
        ]

    def session_id(self, path: Path) -> str | None:
        return _claude_session_id(path)

    def last_message(self, path: Path) -> str | None:
        return _last_claude_message(path)

    def legacy_read(
        self, backend_session_id: str | None, max_bytes: int
    ) -> AgentOutput | None:
        return read_claude_output(
            self.spawned_at,
            self.cwd,
            max_bytes,
            backend_session_id=backend_session_id,
        )


class _CodexBinder(_TranscriptBinder):
    def __init__(self, spawned_at: float, cwd: str) -> None:
        super().__init__(spawned_at, cwd)
        self._scope = _normalize_path(cwd)

    @property
    def cache_scope(self) -> str:
        return self._scope

    def _payload(self, path: Path) -> dict[str, Any] | None:
        meta = _first_json_object(path)
        if not isinstance(meta, dict) or meta.get("type") != "session_meta":
            return None
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            return None
        meta_cwd = payload.get("cwd")
        if not isinstance(meta_cwd, str) or _normalize_path(meta_cwd) != self._scope:
            return None
        return cast("dict[str, Any]", payload)

    def _rollouts(self, *, all_history: bool) -> list[Path]:
        """Candidate rollouts. Raises ``OSError`` if a listing fails.

        See ``_ClaudeBinder._all_transcripts``: a failed enumeration must not
        be laundered into an empty candidate set.
        """
        paths: list[Path] = []
        for directory in _codex_candidate_dirs(
            self.spawned_at, include_all=all_history, strict=True
        ):
            if all_history:
                if not directory.exists():
                    continue
                paths.extend(sorted(directory.glob("rollout-*.jsonl")))
                continue
            paths.extend(
                path
                for _mtime, path in _matching_jsonl_files(
                    directory, self.spawned_at, pattern="rollout-*.jsonl", strict=True
                )
            )
        return list(dict.fromkeys(paths))

    def resolve_by_session_id(self, session_id: str) -> Path | None:
        for path in self._rollouts(all_history=True):
            payload = self._payload(path)
            if payload is not None and payload.get("id") == session_id:
                return path
        return None

    def candidates(self, *, all_history: bool) -> list[Path]:
        return [
            path
            for path in self._rollouts(all_history=all_history)
            if self._payload(path) is not None
        ]

    def session_id(self, path: Path) -> str | None:
        payload = self._payload(path)
        if payload is None:
            return None
        value = payload.get("id")
        return value if isinstance(value, str) and value else None

    def last_message(self, path: Path) -> str | None:
        return _last_codex_message(path)

    def legacy_read(
        self, backend_session_id: str | None, max_bytes: int
    ) -> AgentOutput | None:
        return read_codex_output(
            self.spawned_at,
            self.cwd,
            max_bytes,
            backend_session_id=backend_session_id,
        )


def _make_binder(backend: str, spawned_at: float, cwd: str) -> _TranscriptBinder | None:
    if backend == "claude-code":
        return _ClaudeBinder(spawned_at, cwd)
    if backend == "codex":
        return _CodexBinder(spawned_at, cwd)
    return None


def _build_output(
    binder: _TranscriptBinder, path: Path, session_id: str | None, max_bytes: int
) -> AgentOutput | None:
    identity = _stat_identity(path)
    if identity is None:
        return None
    message = binder.last_message(path)
    if message is None and session_id is None:
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return AgentOutput(
        last_activity_at=mtime,
        last_message=_truncate_tail(message, max_bytes) if message else None,
        rollout_path=str(path),
        backend_session_id=session_id,
    )


def _sidecar_pending(
    record: Mapping[str, object],
    child_alive: Callable[[], bool],
    now: float,
    spawned_at: float,
    window_s: float,
) -> bool:
    """Gate 0: decide whether zero matches means "not yet" rather than "never".

    For a Claude sidecar spawn, argv carries only a "read this file"
    instruction, so the correlation token cannot appear in the transcript
    until the agent has read the file and its tool result has been recorded.
    Until then zero matches says nothing. The window closes when the child
    dies or the deadline passes, after which zero matches falls through to the
    count gate and means ``unverified`` like any other.
    """
    if str(record.get(PROMPT_TRANSPORT_FIELD) or "") != PROMPT_TRANSPORT_SIDECAR:
        return False
    if window_s <= 0 or now - spawned_at >= window_s:
        return False
    return bool(child_alive())


def resolve_agent_binding(  # noqa: PLR0911 - one return per named gate outcome.
    record: Mapping[str, object],
    *,
    child_alive: Callable[[], bool],
    now: float | None = None,
    sidecar_pending_window_s: float = DEFAULT_SIDECAR_PENDING_WINDOW_S,
    max_bytes: int = _LAST_MESSAGE_BUDGET,
    bounded_only: bool = False,
) -> BindingResult:
    """Bind an agent record to its transcript through explicit, ordered gates.

    Gates, in evaluation order:

    1. **Metadata** — correlation field absent -> ``legacy`` (read-only use
       still works); present but unusable -> ``unverified``.
    2. **Scan** — enumerate candidates and scan for the token. Any candidate
       that cannot be read makes the whole scan ``indeterminate``; no match
       count is computed from a partial scan.
    3. **Count** — zero matches -> ``unverified`` (or ``pending``, see below);
       two or more -> ``ambiguous``. There is deliberately no max-mtime
       fallback: the token, not recency, decides identity.
    4. **Session id** — exactly one match but no parseable ``sessionId`` ->
       ``unverified``, because there is no id to re-pin to.

    Gate 0 (the sidecar-pending gate) is written in the plan as the first
    gate, but it is by construction a *branch of the count gate*: it only
    changes the meaning of **zero matches**, and it cannot precede the
    metadata gate because a legacy record has no token to scan for in the
    first place. It is therefore evaluated where zero matches is decided.

    Candidate enumeration is two-tier. Tier 1 resolves the stored session id
    to its transcript directly and ignores the mtime cutoff, so a long-running
    session older than the window is still revalidated with a single file
    open. Tier 1 does **not** short-circuit tier 2: the stored transcript joins
    the candidate set as an extra, cutoff-free candidate, because a tier-1 hit
    alone cannot answer the count gate — a second transcript may also carry the
    token, and that is ``ambiguous`` rather than a licence to keep the stored
    binding. Tier 2 tries the mtime window first and falls back to all history.
    Successful bindings are cached;
    ``pending``/``unverified``/``ambiguous``/``indeterminate`` never are.

    ``bounded_only`` drops the all-history fallback, so the cost of a call is
    capped by the mtime window. It exists for A6's "stay cheap" consumers
    (``agent_status``'s no-marker fallback), which are required to answer from
    a bounded amount of work and to say ``unknown`` rather than pay for a full
    history walk.
    """
    backend = str(record.get("backend") or "")
    spawned_at = _record_float(record.get("spawned_at"))
    cwd = str(record.get("cwd") or "")
    stored_session_id = _record_str(record.get("backend_session_id"))
    # ``None`` when the record lacks the lookup metadata a read needs, or names
    # a backend we have no reader for. That is a separate question from the
    # metadata gate below, which is why it does not short-circuit it.
    binder = _make_binder(backend, spawned_at, cwd) if spawned_at > 0 and cwd else None

    # Gate 1 — metadata. Evaluated before the data guards: a record predating
    # correlation is ``legacy`` whether or not it also predates the lookup
    # fields, and calling it ``unverified`` would misreport a compatibility
    # case as corruption.
    status, correlation_id = classify_correlation(record)
    if status == CORRELATION_LEGACY:
        output = binder.legacy_read(stored_session_id, max_bytes) if binder else None
        return BindingResult(BINDING_LEGACY, output)
    if status != CORRELATION_VALID or not correlation_id:
        return BindingResult(BINDING_UNVERIFIED)
    if binder is None:
        return BindingResult(BINDING_UNVERIFIED)

    token = correlation_marker_token(correlation_id)
    key = (backend, binder.cache_scope, correlation_id, stored_session_id or "")

    cached = _BINDING_CACHE.get(key)
    if cached is not None:
        path = Path(cached.path)
        parsed = binder.session_id(path)
        if _cache_entry_still_valid(cached, parsed):
            output = _build_output(binder, path, cached.session_id, max_bytes)
            if output is not None:
                return BindingResult(BINDING_BOUND, output)
        del _BINDING_CACHE[key]

    # Gate 2, tier 1 — resolve the stored binding to a concrete path, with no
    # mtime cutoff, so a long-running session older than the window is still
    # revalidated by a single open rather than excluded.
    try:
        stored_path = (
            binder.resolve_by_session_id(stored_session_id)
            if stored_session_id
            else None
        )
    except OSError:
        # Tier 1 enumerates too (the name-as-session-id convention is not a
        # contract, so it falls back to a directory walk). A failed walk is
        # "we could not look", not "the stored transcript is gone".
        return BindingResult(BINDING_INDETERMINATE)

    # Gate 2, tier 2 — correction scan: window first, then all history.
    matches, incomplete = binder.scan(token, all_history=False, extra=stored_path)
    if incomplete:
        return BindingResult(BINDING_INDETERMINATE)
    if not matches and not bounded_only:
        matches, incomplete = binder.scan(token, all_history=True, extra=stored_path)
        if incomplete:
            return BindingResult(BINDING_INDETERMINATE)

    # Gate 3 — count (with gate 0 branching the zero case).
    if not matches:
        current = time.time() if now is None else now
        if _sidecar_pending(
            record, child_alive, current, spawned_at, sidecar_pending_window_s
        ):
            return BindingResult(BINDING_PENDING)
        return BindingResult(BINDING_UNVERIFIED)
    if len(matches) > 1:
        return BindingResult(BINDING_AMBIGUOUS)

    return _bind_path(binder, key, matches[0], max_bytes)


def _bind_path(
    binder: _TranscriptBinder,
    key: tuple[str, str, str, str],
    path: Path,
    max_bytes: int,
) -> BindingResult:
    """Gate 4 — session-id gate, then cache and return the bound result."""
    session_id = binder.session_id(path)
    if not session_id:
        return BindingResult(BINDING_UNVERIFIED)
    output = _build_output(binder, path, session_id, max_bytes)
    if output is None:
        return BindingResult(BINDING_UNVERIFIED)
    entry = _cache_entry_for(path, session_id)
    if entry is not None:
        _BINDING_CACHE[key] = entry
    return BindingResult(BINDING_BOUND, output)


def _record_float(value: object) -> float:
    try:
        return float(cast(Any, value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _record_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None

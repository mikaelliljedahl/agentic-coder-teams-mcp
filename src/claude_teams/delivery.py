"""A3/A4 — child-liveness early failure and nonce delivery confirmation (R6).

Liveness, transcript growth and state-marker transitions are all *wrong
oracles* for "did the follow-up prompt reach this agent?". Markers are keyed on
agent name and hooks write only ``state``/``event``/``ts``, so a surviving old
process and a freshly resumed one write byte-identical markers. Transcript
growth says only that *something* is writing.

So every delivery attempt embeds a cryptographically random nonce in the final
prompt, and delivery is confirmed by finding that exact nonce in a **named
receipt record** of the transcript whose ``backend_session_id`` is being
resumed. Confirmation requires **both** child survival (A3) and the receipt
record (A4); neither alone is evidence.

Four rules in this module are load-bearing, each guarding a specific failure:

1. **Semantic, not substring.** The nonce is extracted from a *parsed* receipt
   record, so a nonce echoed in a CLI diagnostic or in serialized argv cannot
   confirm a delivery that never happened.
2. **Record boundary, not raw EOF.** :meth:`ReceiptScanner.snapshot` records the
   offset of the last *complete* JSONL record. Starting at raw EOF could hand
   the parser an unparsable fragment, and the readers skip malformed lines
   *permanently* — so the record would never be reconsidered once its remaining
   bytes arrived. Partial bytes are retained between polls rather than skipped.
3. **Identity/size regression is rotation, not absence.** Continuity across a
   rotation is established by backend session id **plus** file identity. The
   correlation token corroborates when present but is never a precondition: a
   successor may legitimately not replay the spawn marker, and requiring it
   would fail a delivery that genuinely landed. More than one candidate
   successor is ``ambiguous``, never a guess.
4. **Bound expiry is not terminal.** With a live child an exhausted scan is
   ``unconfirmed`` (R6's "live uncertainty"); only a dead child with no receipt
   is ``failed(not_delivered)``. Keeping the two apart is what lets a retry
   reconcile a prior attempt instead of delivering the prompt twice.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

#: Fixed delimiter preceding the high-entropy delivery id. Matching is on the
#: parsed payload and requires the FULL id: the prefix alone, a truncated id,
#: or an id embedded in a longer hex run must never confirm.
DELIVERY_MARKER_PREFIX = "wat-deliver:"

_NONCE_BYTES = 16
_NONCE_HEX_LEN = _NONCE_BYTES * 2

#: The lookarounds are the grammar's teeth: without them ``wat-deliver:<id>0``
#: would confirm ``<id>``, letting one attempt's marker satisfy another's.
_MARKER_RE = re.compile(
    rf"(?<![0-9a-f]){re.escape(DELIVERY_MARKER_PREFIX)}"
    rf"([0-9a-f]{{{_NONCE_HEX_LEN}}})(?![0-9a-f])"
)

_MARKER_TEMPLATE = (
    "[win-agent-teams delivery id: {token} — internal marker, ignore this line]"
)

#: The nonce was found in a named receipt record of the bound transcript.
SCAN_FOUND = "found"
#: Nothing yet. Not evidence of absence — the child may still be writing.
SCAN_PENDING = "pending"
#: More than one candidate successor transcript; guessing is not allowed.
SCAN_AMBIGUOUS = "ambiguous"

#: Recipient-confirmed (R6). The only status that may be reported as success.
DELIVERY_DELIVERED = "delivered"
#: Bound expired with a live child — R6 "live uncertainty". NOT terminal.
DELIVERY_UNCONFIRMED = "unconfirmed"
#: Definite non-delivery: the child is dead and no receipt exists. Terminal.
DELIVERY_FAILED = "failed"
#: The transcript rotated into more than one candidate successor.
DELIVERY_AMBIGUOUS = "ambiguous"

#: How long a child must survive before its death stops counting as the A3
#: "resume never attached" signal and starts counting as a mid-delivery death.
DEFAULT_SETTLE_SECONDS = 2.0
#: How long confirmation scans before returning a non-terminal ``unconfirmed``.
DEFAULT_CONFIRM_BOUND_SECONDS = 20.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.25

_READ_CHUNK = 256 * 1024


def new_delivery_nonce() -> str:
    """Return a fresh cryptographically random per-attempt delivery nonce.

    ``secrets`` rather than ``uuid4``/``random``: the nonce is the sole
    evidence distinguishing this attempt's prompt from any other text in the
    conversation, so it must not be predictable or repeatable.
    """
    return secrets.token_hex(_NONCE_BYTES)


def delivery_marker_token(nonce: str) -> str:
    """Return the scannable token form of a delivery nonce."""
    return f"{DELIVERY_MARKER_PREFIX}{nonce}"


def delivery_marker(nonce: str) -> str:
    """Return the human-readable marker line embedded in a delivery prompt."""
    return _MARKER_TEMPLATE.format(token=delivery_marker_token(nonce))


def delivered_prompt(prompt: str, nonce: str, *, single_line: bool) -> str:
    """Append the delivery marker to ``prompt``.

    ``single_line`` selects the argv form (joined by a space so it introduces
    no CLI-sensitive character), mirroring ``agent_output.correlated_prompt``.
    """
    separator = " " if single_line else "\n\n"
    return f"{prompt}{separator}{delivery_marker(nonce)}"


# ---------------------------------------------------------------------------
# Named receipt records
# ---------------------------------------------------------------------------


def _texts_from_content(content: object) -> list[str]:
    """Flatten a message ``content`` field to the user-visible text parts."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for item in cast("list[Any]", content):
        if isinstance(item, str):
            texts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        mapping = cast("dict[str, object]", item)
        kind = mapping.get("type")
        if kind in {"text", "input_text"}:
            value = mapping.get("text")
            if isinstance(value, str):
                texts.append(value)
        elif kind == "tool_result":
            # A sidecar delivery's receipt: the tool result carrying the prompt
            # file's contents into context. argv held only the read instruction.
            texts.extend(_texts_from_content(mapping.get("content")))
    return texts


def _claude_receipt_texts(record: dict[str, object]) -> list[str]:
    if record.get("type") != "user":
        return []
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    mapping = cast("dict[str, object]", message)
    if mapping.get("role") not in {None, "user"}:
        return []
    return _texts_from_content(mapping.get("content"))


def _codex_receipt_texts(record: dict[str, object]) -> list[str]:
    if record.get("type") != "response_item":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    mapping = cast("dict[str, object]", payload)
    if mapping.get("role") != "user":
        return []
    return _texts_from_content(mapping.get("content"))


def receipt_nonces(record: object, backend: str) -> set[str]:
    """Return the delivery nonces carried by ``record``'s receipt payload.

    Only the backend's **named receipt record class** is inspected:

    - **claude-code** — the ``type: "user"`` record, whether the nonce arrives
      as literal user text (argv transport) or inside the ``tool_result`` for
      the prompt-file read (sidecar transport).
    - **codex** — the rollout record for user input, the same record class the
      correlation-token scanner reads, tightened here from a raw substring
      search to a parsed field.

    Assistant output, tool invocations, and CLI diagnostics are deliberately
    not receipt records: a nonce appearing only there proves the text was
    echoed or logged, not that it entered the agent's context as a prompt.
    """
    if not isinstance(record, dict):
        return set()
    mapping = cast("dict[str, object]", record)
    if backend == "claude-code":
        texts = _claude_receipt_texts(mapping)
    elif backend == "codex":
        texts = _codex_receipt_texts(mapping)
    else:
        return set()
    found: set[str] = set()
    for text in texts:
        found.update(_MARKER_RE.findall(text))
    return found


def transcript_session_id(path: Path, backend: str) -> str | None:
    """Return the backend session id recorded in ``path``, or ``None``."""
    for record in _iter_head_records(path):
        if backend == "claude-code":
            value = record.get("sessionId")
            if isinstance(value, str) and value:
                return value
        elif backend == "codex":
            if record.get("type") != "session_meta":
                continue
            payload = record.get("payload")
            if isinstance(payload, dict):
                value = cast("dict[str, object]", payload).get("id")
                if isinstance(value, str) and value:
                    return value
    return None


def _iter_head_records(path: Path, limit: int = 50) -> Iterable[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle):
                if index >= limit:
                    return
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield cast("dict[str, object]", item)
    except OSError:
        return


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryOutcome:
    """Result of one bounded confirmation attempt."""

    status: str
    reason: str = ""

    @property
    def delivered(self) -> bool:
        """Whether the prompt was positively confirmed at the recipient."""
        return self.status == DELIVERY_DELIVERED

    @property
    def terminal(self) -> bool:
        """Whether this outcome may be reported as settled (R6)."""
        return self.status in {DELIVERY_DELIVERED, DELIVERY_FAILED}


class ReceiptScanner:
    """Incremental, rotation-aware scanner for one attempt's receipt record.

    Usage is ``snapshot()`` **before** the resume, then ``poll(nonce)``
    repeatedly. Snapshotting first is what makes the scan an observation of
    *this* attempt: records that already existed cannot confirm it.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        backend: str,
        backend_session_id: str,
        successors: Callable[[], list[Path]] | None = None,
    ) -> None:
        """Build a scanner for one attempt against ``path``.

        ``successors`` supplies rotation candidates lazily. There is
        deliberately no correlation-token parameter: the token corroborates a
        binding elsewhere, but inside rotation it could only ever act as a
        selector between candidates, and selecting is guessing.
        """
        self.path = path
        self.backend = backend
        self.backend_session_id = backend_session_id
        self._successors = successors
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._buffer = b""

    # -- snapshotting -------------------------------------------------------

    def snapshot(self) -> None:
        """Record the offset of the last COMPLETE record, plus file identity.

        Any trailing partial bytes stay *after* the offset, so they are re-read
        on the next poll and matched once the record is finished — the
        skip-malformed-permanently trap the plan calls out.
        """
        self._buffer = b""
        self._offset = 0
        self._identity = None
        if self.path is None:
            return
        self._identity = _stat_identity(self.path)
        self._offset = _last_complete_offset(self.path)

    def rewind(self) -> None:
        """Anchor at the START of the file, to scan history rather than growth.

        Used when reconciling a *previous* attempt: the record being looked for
        was written before this call began, so anchoring at the current tail —
        which is what :meth:`snapshot` does — would deliberately skip it.
        """
        self._buffer = b""
        self._offset = 0
        self._identity = _stat_identity(self.path) if self.path else None

    # -- polling ------------------------------------------------------------

    def poll(self, nonce: str) -> str:
        """Scan any bytes written since the last poll for ``nonce``."""
        if self.path is None:
            return SCAN_PENDING
        identity = _stat_identity(self.path)
        if identity is None or identity != self._identity or self._shrank():
            rotated = self._follow_rotation()
            if rotated == SCAN_AMBIGUOUS:
                return SCAN_AMBIGUOUS
            if rotated == SCAN_PENDING:
                return SCAN_PENDING
        return self._scan_new_bytes(nonce)

    def _shrank(self) -> bool:
        try:
            return self.path is not None and self.path.stat().st_size < self._offset
        except OSError:
            return True

    def _follow_rotation(self) -> str:
        """Re-anchor on a successor transcript, or report why we cannot.

        Continuity is backend session id **plus** file identity. The
        correlation token only *corroborates* when several candidates survive
        that test; it is never required, because a successor may legitimately
        not replay the spawn marker.
        """
        if self._successors is None:
            return SCAN_PENDING
        try:
            candidates = [
                path
                for path in self._successors()
                if path.exists()
                and transcript_session_id(path, self.backend) == self.backend_session_id
            ]
        except OSError:
            return SCAN_PENDING
        if not candidates:
            return SCAN_PENDING
        if len(candidates) > 1:
            # Unconditionally ambiguous. The correlation token is NOT consulted
            # here: it is written at spawn, and a successor may legitimately
            # not replay it, so its presence in one candidate is no evidence
            # that the other is not the live conversation. Reducing the set
            # with it would attribute a delivery on a guess — the false-receipt
            # failure this whole module exists to eliminate.
            return SCAN_AMBIGUOUS
        self.path = candidates[0]
        self._identity = _stat_identity(self.path)
        self._offset = 0
        self._buffer = b""
        return SCAN_FOUND

    def _scan_new_bytes(self, nonce: str) -> str:
        path = self.path
        if path is None:
            return SCAN_PENDING
        # Byte-oriented throughout, so the offset is directly comparable with
        # ``st_size`` in the regression check. Decoding happens per record.
        try:
            with path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read(_READ_CHUNK)
        except OSError:
            return SCAN_PENDING
        if not chunk:
            return SCAN_PENDING
        self._offset += len(chunk)
        self._buffer += chunk
        head, separator, tail = self._buffer.rpartition(b"\n")
        if not separator:
            # No complete record yet: retain every byte for the next poll
            # rather than advancing past a fragment we could never re-read.
            return SCAN_PENDING
        self._buffer = tail
        for line in head.split(b"\n"):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if nonce in receipt_nonces(record, self.backend):
                return SCAN_FOUND
        return SCAN_PENDING


def _stat_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _last_complete_offset(path: Path) -> int:
    """Return the byte offset just past the file's last newline.

    This is the boundary of the last **complete** JSONL record. Anything after
    it is a partial write that must be re-read, not skipped.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return 0
    index = data.rfind(b"\n")
    return 0 if index < 0 else index + 1


def confirm_delivery(
    scanner: ReceiptScanner,
    nonce: str,
    *,
    child_alive: Callable[[], bool],
    bound_s: float,
    poll_interval_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    settle_s: float = DEFAULT_SETTLE_SECONDS,
) -> DeliveryOutcome:
    """Poll for ``nonce`` until it lands, the child dies, or the bound expires.

    The clock and the sleep are injected rather than taken from :mod:`time` so
    the timing behaviour is testable without wall-clock races.

    Three distinct non-success outcomes, and collapsing any two of them would
    misdescribe reality:

    - **child dead inside the settle window** → ``resume_not_confirmed``. This
      is A3: ``claude --resume <bad-id>`` exits within a second, so the resume
      never attached at all. The agent record must be left untouched.
    - **child dead after the settle window, no receipt** → ``not_delivered``.
      Definite non-delivery, terminal (R6).
    - **bound expired, child still alive** → ``unconfirmed``. A transcript
      write buffered past the bound can still arrive, so this is neither
      delivered nor terminally failed; a retry must reconcile before resending.

    The scan runs *before* the liveness check on every iteration, so a receipt
    already on disk wins over a child that has since exited.
    """
    start = clock()
    while True:
        result = scanner.poll(nonce)
        if result == SCAN_FOUND:
            return DeliveryOutcome(DELIVERY_DELIVERED)
        if result == SCAN_AMBIGUOUS:
            return DeliveryOutcome(DELIVERY_AMBIGUOUS, "rotation_ambiguous")
        if not child_alive():
            elapsed = clock() - start
            reason = "resume_not_confirmed" if elapsed < settle_s else "not_delivered"
            return DeliveryOutcome(DELIVERY_FAILED, reason)
        if clock() - start >= bound_s:
            return DeliveryOutcome(DELIVERY_UNCONFIRMED, "scan_expired")
        sleep(poll_interval_s)


def prompt_file_name(agent_name: str, nonce: str) -> str:
    """Return the per-attempt prompt sidecar filename (A5).

    The nonce is in the filename so two concurrent calls to the same agent can
    never collide on one path, and so a file can be attributed back to the
    attempt that wrote it during cleanup.
    """
    return f"{agent_name}.{nonce}.prompt.txt"


def prompt_file_glob(agent_name: str) -> str:
    """Return the glob matching every per-attempt prompt file for an agent."""
    return f"{agent_name}.*.prompt.txt"


def stale_prompt_files(
    directory: Path, agent_name: str, *, older_than: float, now: float
) -> list[Path]:
    """Return this agent's prompt files older than ``older_than`` seconds.

    Age-based only, and deliberately never "delete this agent's other files
    because a new call started": that would race a concurrent attempt whose
    CLI has not read its file yet.
    """
    stale: list[Path] = []
    with contextlib.suppress(OSError):
        for path in directory.glob(prompt_file_glob(agent_name)):
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime >= older_than:
                    stale.append(path)
    return stale


def remove_prompt_file(path: Path | None) -> None:
    """Best-effort removal of one attempt's prompt sidecar."""
    if path is None:
        return
    with contextlib.suppress(OSError):
        Path(path).unlink(missing_ok=True)


__all__ = [
    "DELIVERY_AMBIGUOUS",
    "DELIVERY_DELIVERED",
    "DELIVERY_FAILED",
    "DELIVERY_MARKER_PREFIX",
    "DELIVERY_UNCONFIRMED",
    "SCAN_AMBIGUOUS",
    "SCAN_FOUND",
    "SCAN_PENDING",
    "DeliveryOutcome",
    "ReceiptScanner",
    "confirm_delivery",
    "delivered_prompt",
    "delivery_marker",
    "delivery_marker_token",
    "new_delivery_nonce",
    "prompt_file_glob",
    "prompt_file_name",
    "receipt_nonces",
    "remove_prompt_file",
    "stale_prompt_files",
    "transcript_session_id",
]

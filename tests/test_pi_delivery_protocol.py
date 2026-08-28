"""Pi under the message-delivery protocol: correlation, binding, and receipts.

Pi arrived on main (279a171) declaring ``supports_resume() -> True`` but with
no correlation marker and no named receipt record. Under this branch that made
a Pi agent classify as ``unverified`` and refuse follow-up — and the refusal was
an accident of ``_make_binder`` returning ``None``, not a considered answer.

Pi can in fact support the full protocol, and these tests pin why:

- its prompt reaches the agent verbatim (``PiBackend._prompt_args`` passes
  ``request.prompt`` as a single positional arg on the ``node`` launch path),
  so it can carry the correlation marker the way Codex does;
- its transcript records name their role (``type: "message"`` with
  ``message.role``), so ``user`` is an identifiable receipt record class;
- its storage is deterministic and per-agent
  (``--session-dir <session>/pi-sessions/<agent>``), so the candidate set is
  scoped by construction rather than by an mtime window.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import agent_output as ao
from claude_teams import server_simple
from claude_teams.agent_output import (
    _CODEX_CORRELATION_PREFIX,
    correlation_marker,
    correlation_marker_token,
)
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.pi import PiBackend
from claude_teams.backends.registry import canonical_backend_name
from claude_teams.delivery import delivery_marker_token, receipt_nonces


class _FakeBackend:
    def __init__(self) -> None:
        self.last_request: SpawnRequest | None = None

    def default_model(self) -> str:
        return "medium"

    def resolve_model(self, model: str) -> str:
        return model

    def resolve_launch(
        self, model: str, reasoning_effort: str | None
    ) -> tuple[str, str | None]:
        return (model if model.strip() else self.default_model()), reasoning_effort

    def spawn(self, request: SpawnRequest) -> SimpleNamespace:
        self.last_request = request
        return SimpleNamespace(process_handle="789")


class _FakeRegistry:
    def __init__(self, backend: object, name: str) -> None:
        self._backend = backend
        self._name = name

    def resolve_name(self, name: str) -> str:
        return canonical_backend_name(name)

    def default_backend(self) -> str:
        return self._name

    def get(self, backend: str) -> object:
        assert backend == self._name
        return self._backend


def _request(tmp_path: Path, **extra: str) -> SpawnRequest:
    return SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="",
        agent_type="worker",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="lead",
        permission_mode="bypass",
        extra={"session_dir": str(tmp_path / "sess"), **extra},
    )


# ---------------------------------------------------------------------------
# Correlation marker — Pi's prompt is verbatim, so it carries the marker
# ---------------------------------------------------------------------------


def test_pi_prompt_carries_exactly_one_marker(tmp_path: Path) -> None:
    request = _request(tmp_path, correlation_id="corr-pi")

    prompt = PiBackend()._correlated_prompt(request)

    assert prompt.count(_CODEX_CORRELATION_PREFIX) == 1
    assert correlation_marker_token("corr-pi") in prompt
    assert prompt.startswith("do stuff")


def test_pi_prompt_without_persisted_id_carries_no_marker(tmp_path: Path) -> None:
    """A record predating correlation stays legacy: no id is invented for it."""
    assert PiBackend()._correlated_prompt(_request(tmp_path)) == "do stuff"


@pytest.mark.parametrize("builder", ["build_command", "build_resume_command"])
def test_pi_spawn_and_resume_argv_both_carry_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    """Resume is correlated too — the same R8/A4 reason as Claude and Codex.

    A resume whose transcript cannot be identified is exactly the false
    ``delivered`` receipt R6 exists to prevent.
    """
    monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["node", "cli.js"])
    monkeypatch.setattr(PiBackend, "_headless", lambda self: True)
    request = _request(tmp_path, correlation_id="corr-pi")

    backend = PiBackend()
    cmd = (
        backend.build_command(request)
        if builder == "build_command"
        else backend.build_resume_command(request, "pi-session-id")
    )

    assert cmd[-1].count(_CODEX_CORRELATION_PREFIX) == 1
    assert correlation_marker_token("corr-pi") in cmd[-1]


# ---------------------------------------------------------------------------
# Named receipt record — pi's `type: "message"` + `role: "user"`
# ---------------------------------------------------------------------------


def _pi_message(role: str, text: str) -> dict:
    return {
        "type": "message",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def test_pi_user_record_is_a_receipt() -> None:
    # A full 32-hex id: the marker grammar deliberately refuses a short or
    # truncated one, so a realistic nonce is what the receipt must carry.
    nonce = "0123456789abcdef0123456789abcdef"
    record = _pi_message("user", f"follow up {delivery_marker_token(nonce)}")

    assert receipt_nonces(record, "pi") == {nonce}


def test_pi_assistant_record_is_not_a_receipt() -> None:
    """An echo in assistant output proves the text was written, not received."""
    # A full 32-hex id: the marker grammar deliberately refuses a short or
    # truncated one, so a realistic nonce is what the receipt must carry.
    nonce = "0123456789abcdef0123456789abcdef"
    record = _pi_message("assistant", f"I see {delivery_marker_token(nonce)}")

    assert receipt_nonces(record, "pi") == set()


# ---------------------------------------------------------------------------
# Binding ladder — Pi gets a binder scoped to its own session dir
# ---------------------------------------------------------------------------


def _write_pi_session(directory: Path, session_id: str, *rows: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.jsonl"
    header = {"type": "session", "version": 3, "id": session_id, "cwd": "C:/x"}
    path.write_text(
        "\n".join(json.dumps(r) for r in (header, *rows)) + "\n", encoding="utf-8"
    )
    return path


def _pi_record(pi_dir: Path, correlation_id: str | None = "corr-pi") -> dict:
    record: dict = {
        "backend": "pi",
        "name": "worker",
        "session_id": "sess",
        "spawned_at": 1_000.0,
        "cwd": "C:/x",
        ao.PI_SESSION_DIR_FIELD: str(pi_dir),
    }
    if correlation_id is not None:
        record[ao.CORRELATION_FIELD] = correlation_id
    return record


def _bind(record: dict):
    return ao.resolve_agent_binding(record, child_alive=lambda: False)


def test_pi_binds_to_its_own_marked_transcript(tmp_path: Path) -> None:
    pi_dir = tmp_path / "pi-sessions" / "worker"
    marker = correlation_marker("corr-pi")
    _write_pi_session(
        pi_dir,
        "worker",
        _pi_message("user", f"do stuff\n\n{marker}"),
        _pi_message("assistant", "on it"),
    )

    result = _bind(_pi_record(pi_dir))

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "worker"
    assert result.output.last_message == "on it"


def test_pi_zero_marker_matches_is_unverified_not_newest_mtime(
    tmp_path: Path,
) -> None:
    """No marker, no binding — the same rule the ladder applies everywhere."""
    pi_dir = tmp_path / "pi-sessions" / "worker"
    _write_pi_session(pi_dir, "worker", _pi_message("assistant", "unmarked"))

    assert _bind(_pi_record(pi_dir)).outcome == ao.BINDING_UNVERIFIED


def test_pi_two_marker_matches_is_ambiguous(tmp_path: Path) -> None:
    """Name reuse leaves two marked transcripts in one dir — never guess.

    Pi scopes storage by agent *name* (``--session-id <agent>``), so a reused
    name writes into the same directory as the agent it replaced. The marker
    cannot prevent that collision, but it makes it detectable: two transcripts
    carrying the same token is ``ambiguous``, not a licence to pick one.
    """
    pi_dir = tmp_path / "pi-sessions" / "worker"
    marker = correlation_marker("corr-pi")
    _write_pi_session(pi_dir, "worker", _pi_message("user", marker))
    _write_pi_session(pi_dir, "worker-old", _pi_message("user", marker))

    assert _bind(_pi_record(pi_dir)).outcome == ao.BINDING_AMBIGUOUS


def test_pi_record_without_correlation_id_is_legacy(tmp_path: Path) -> None:
    pi_dir = tmp_path / "pi-sessions" / "worker"
    _write_pi_session(pi_dir, "worker", _pi_message("assistant", "hi"))

    result = _bind(_pi_record(pi_dir, correlation_id=None))

    assert result.outcome == ao.BINDING_LEGACY
    assert result.output is not None
    assert result.output.last_message == "hi"


# ---------------------------------------------------------------------------
# Server wiring — the record field is actually written, and only for pi
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_name", "expect_field"), [("pi", True), ("claude-code", False)]
)
async def test_spawn_records_the_pi_session_dir_only_for_pi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    expect_field: bool,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, backend_name))

    result = await server_simple.spawn_agent(
        "do stuff", name="worker", backend=backend_name, cwd=str(tmp_path)
    )
    session_id = result["session_id"]
    record = next(
        a for a in server_simple._load_agents(session_id) if a["name"] == "worker"
    )

    assert (ao.PI_SESSION_DIR_FIELD in record) is expect_field
    if expect_field:
        assert record[ao.PI_SESSION_DIR_FIELD] == str(
            server_simple._pi_session_dir(session_id, "worker")
        )
        # The server leaves a pi prompt unmarked; the backend appends the one
        # marker, exactly as it does for codex.
        request = backend.last_request
        assert request is not None
        assert request.prompt == "do stuff"
        assert (request.extra or {})[ao.CORRELATION_FIELD] == record["correlation_id"]
        assert (
            PiBackend()._correlated_prompt(request).count(_CODEX_CORRELATION_PREFIX)
            == 1
        )


def test_pi_record_without_session_dir_is_unverified(tmp_path: Path) -> None:
    """A pi record predating the stored dir cannot be bound by guesswork.

    The dir is persisted at spawn precisely because it cannot be re-derived
    later without re-implementing the backend's layout in the reader.
    """
    record = _pi_record(tmp_path)
    del record[ao.PI_SESSION_DIR_FIELD]

    assert _bind(record).outcome == ao.BINDING_UNVERIFIED

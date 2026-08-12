"""Tests for A1/A1b: per-spawn correlation id and server-owned prompt transport."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import agent_output as ao
from claude_teams import server_simple
from claude_teams.agent_output import (
    _CODEX_CORRELATION_PREFIX,
    classify_correlation,
    correlation_marker,
    correlation_marker_token,
    read_claude_output,
)
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import SpawnRequest


class _FakeBackend:
    def __init__(self) -> None:
        self.last_request: SpawnRequest | None = None

    def default_model(self) -> str:
        return "sonnet"

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
    def __init__(self, backend: object, name: str = "claude-code") -> None:
        self._backend = backend
        self._name = name

    def default_backend(self) -> str:
        return self._name

    def get(self, backend: str) -> object:
        assert backend == self._name
        return self._backend


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    return backend


def _read_text(path: str) -> str:
    """Read a file by path (sync helper so async tests avoid ASYNC240)."""
    return Path(path).read_text(encoding="utf-8")


def _record(session_id: str, name: str = "worker") -> dict:
    agents = server_simple._load_agents(session_id)
    return next(a for a in agents if a["name"] == name)


# --------------------------------------------------------------------------
# A1 — transport selection and marker form
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_prompt_uses_argv_with_single_line_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _session(tmp_path, monkeypatch)

    result = await server_simple.spawn_agent(
        "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    assert (request.extra or {}).get("prompt_file_path") is None
    correlation_id = _record(result["session_id"])["correlation_id"]
    assert correlation_id
    marker = correlation_marker(correlation_id)
    assert request.prompt == f"plain prompt {marker}"
    assert "\n" not in request.prompt
    assert "\r" not in request.prompt


@pytest.mark.asyncio
async def test_transport_decision_sees_only_the_user_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, monkeypatch)
    seen: list[str] = []
    real = server_simple._needs_prompt_file

    def _spy(prompt: str) -> bool:
        seen.append(prompt)
        return real(prompt)

    monkeypatch.setattr(server_simple, "_needs_prompt_file", _spy)

    await server_simple.spawn_agent(
        "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    assert seen == ["plain prompt"]


@pytest.mark.asyncio
async def test_sensitive_prompt_uses_sidecar_with_newline_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _session(tmp_path, monkeypatch)
    prompt = "first 'line'\nsecond \"line\""

    result = await server_simple.spawn_agent(
        prompt, name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    prompt_path = (request.extra or {})["prompt_file_path"]
    correlation_id = _record(result["session_id"])["correlation_id"]
    marker = correlation_marker(correlation_id)

    # The file the agent actually reads carries the marker, newline-delimited.
    assert _read_text(prompt_path) == f"{prompt}\n\n{marker}"
    # ...and argv carries only the read instruction pointing at that same file.
    assert ClaudeCodeBackend()._prompt_arg(request) == (
        "Read your complete task prompt from UTF-8 file path "
        f"{prompt_path} then follow the file contents exactly."
    )


# --------------------------------------------------------------------------
# A1 — the marker is visible in the transcript, for both transports
# --------------------------------------------------------------------------


def _claude_project_dir(home: Path, cwd: Path) -> Path:
    encoded = re.sub(r"[^a-zA-Z0-9]", "-", str(cwd.resolve()))
    return home / ".claude" / "projects" / encoded


def _write_jsonl(path: Path, rows: list[dict], mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _transcript(session_id: str, user_content: object, spawned_at: float) -> list[dict]:
    return [
        {
            "type": "user",
            "timestamp": _ts(spawned_at + 1),
            "sessionId": session_id,
            "message": {"role": "user", "content": user_content},
        },
        {
            "type": "assistant",
            "timestamp": _ts(spawned_at + 2),
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "on it"}],
            },
        },
    ]


@pytest.mark.parametrize("transport", ["argv", "sidecar"])
def test_marker_is_visible_in_claude_transcript_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    cwd = tmp_path / "work"
    cwd.mkdir()
    spawned_at = 1_762_969_000.0
    correlation_id = "abc123"
    token = correlation_marker_token(correlation_id)
    final = f"do stuff\n\n{correlation_marker(correlation_id)}"

    if transport == "argv":
        # argv: the single-line prompt is recorded verbatim as the user turn.
        final = f"do stuff {correlation_marker(correlation_id)}"
        user_content: object = final
    else:
        # sidecar: the agent reads the file, so the marker lands in the
        # tool result that carries the file contents into context.
        user_content = [{"type": "tool_result", "content": final}]

    project = _claude_project_dir(tmp_path / "home", cwd)
    _write_jsonl(
        project / "mine.jsonl",
        _transcript("mine", user_content, spawned_at),
        spawned_at + 5,
    )
    # Decoy with a NEWER mtime: max-mtime alone would pick the wrong file.
    _write_jsonl(
        project / "other.jsonl",
        _transcript("other", "unrelated conversation", spawned_at),
        spawned_at + 50,
    )

    output = read_claude_output(spawned_at, str(cwd), correlation_token=token)

    assert output is not None
    assert output.backend_session_id == "mine"
    assert Path(output.rollout_path).name == "mine.jsonl"


# --------------------------------------------------------------------------
# A1b — Codex consumes the persisted id and carries exactly one marker
# --------------------------------------------------------------------------


def test_codex_prompt_carries_exactly_one_marker(tmp_path: Path) -> None:
    request = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="gpt",
        agent_type="worker",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="lead",
        permission_mode="bypass",
        extra={"correlation_id": "deadbeef"},
    )

    prompt = CodexBackend()._correlated_prompt(request)

    assert prompt.count(_CODEX_CORRELATION_PREFIX) == 1
    assert correlation_marker_token("deadbeef") in prompt


def test_codex_prompt_without_persisted_id_carries_no_marker(tmp_path: Path) -> None:
    request = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="gpt",
        agent_type="worker",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="lead",
        permission_mode="bypass",
    )

    assert CodexBackend()._correlated_prompt(request) == "do stuff"


@pytest.mark.asyncio
async def test_codex_spawn_prompt_is_not_marked_by_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, "codex"))

    result = await server_simple.spawn_agent(
        "do stuff", name="worker", backend="codex", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    # The server leaves the codex prompt alone; codex appends the one marker.
    assert request.prompt == "do stuff"
    correlation_id = _record(result["session_id"])["correlation_id"]
    assert (request.extra or {})["correlation_id"] == correlation_id
    assert (
        CodexBackend()._correlated_prompt(request).count(_CODEX_CORRELATION_PREFIX) == 1
    )


def test_pi_multiline_prompt_gets_a_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pi's ``@file`` fallback needs the server to actually write the file.

    ``PiBackend._prompt_args`` falls back to ``@<prompt_file>`` whenever argv
    is unsafe (the ``pi.cmd`` shim truncates at the first newline, and the
    headless path has a command-line ceiling) -- but that branch was dead,
    because the server wrote a sidecar for ``claude-code`` only. The prompt
    itself stays un-marked: pi's backend appends the one correlation marker.
    """
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    final_prompt, extra = server_simple._materialize_prompt(
        "sess",
        "worker",
        "pi",
        "first line\nsecond line",
        "corr-pi",
        file_token="nonce-1",
    )

    path = extra["prompt_file_path"]
    assert Path(path).read_text(encoding="utf-8") == "first line\nsecond line"
    assert final_prompt == "first line\nsecond line"
    assert _CODEX_CORRELATION_PREFIX not in final_prompt


def test_pi_short_single_line_prompt_stays_in_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    _, extra = server_simple._materialize_prompt(
        "sess",
        "worker",
        "pi",
        "do stuff",
        "corr-pi",
        file_token="nonce-1",
    )

    assert extra.get("prompt_file_path") is None


# --------------------------------------------------------------------------
# A1c — the Claude path is as unambiguous as the Codex path
#
# Ported from main's `test_claude_build_command_embeds_correlation_token` /
# `..._embeds_token_with_prompt_file`, which asserted the *backend* injected the
# marker into argv. That injection is removed: the server owns materialization
# for both transports, so the backend injecting too would double-mark. The
# property those tests protected — a spawned Claude agent's prompt carries a
# marker, exactly once, on both transports — is asserted here instead, at the
# layer that now owns it.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "transport"),
    [("plain prompt", "argv"), ("first 'line'\nsecond \"line\"", "sidecar")],
)
async def test_claude_spawn_prompt_carries_exactly_one_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str, transport: str
) -> None:
    backend = _session(tmp_path, monkeypatch)

    result = await server_simple.spawn_agent(
        prompt, name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    correlation_id = _record(result["session_id"])["correlation_id"]
    assert correlation_id
    prompt_file = (request.extra or {}).get("prompt_file_path")
    assert (prompt_file is not None) == (transport == "sidecar")

    # Whichever transport the agent actually reads must carry the marker once.
    delivered = _read_text(prompt_file) if prompt_file else request.prompt
    assert delivered.count(_CODEX_CORRELATION_PREFIX) == 1
    assert correlation_marker_token(correlation_id) in delivered

    # And argv, as built by the real backend, must not add a second one.
    assert ClaudeCodeBackend().build_command(request)[-1].count(
        _CODEX_CORRELATION_PREFIX
    ) == (0 if prompt_file else 1)


def test_claude_resume_prompt_is_correlated(tmp_path: Path) -> None:
    """A resume is correlated too — inverted from main's assertion.

    Main asserted the marker was **absent** from the resume command, on the
    reasoning that resume already knows the backend session id. This branch
    correlates resume deliberately (R8/A4): the stored session id is exactly
    what may be wrong, and a resume whose transcript cannot be identified is
    the false-``delivered`` receipt that R6 exists to prevent. Dropping the
    marker on resume would also downgrade the agent to ``legacy`` at read time,
    which per R8 means it could never be followed up again.
    """
    final_prompt, extra = server_simple._materialize_prompt(
        "sess",
        "worker",
        "claude-code",
        "follow up",
        "corr-resume",
        file_token="nonce-1",
        delivery_nonce="nonce-1",
    )

    assert extra.get("prompt_file_path") is None
    assert correlation_marker_token("corr-resume") in final_prompt
    assert final_prompt.count(_CODEX_CORRELATION_PREFIX) == 1


def test_claude_legacy_resume_mints_no_correlation_id(tmp_path: Path) -> None:
    """A record that predates correlation stays legacy across resume.

    The counterpart to the inversion above: correlating resume must not be
    implemented by inventing an id for a conversation that never carried one,
    which would bind against a marker no transcript can contain.
    """
    final_prompt, _ = server_simple._materialize_prompt(
        "sess",
        "worker",
        "claude-code",
        "follow up",
        None,
        file_token="nonce-1",
        delivery_nonce="nonce-1",
    )

    assert _CODEX_CORRELATION_PREFIX not in final_prompt


# --------------------------------------------------------------------------
# A1b — persistence, restart, and classification
# --------------------------------------------------------------------------


def _write_marked_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict,
    correlation_id: str,
    *,
    session_id: str,
) -> None:
    """Write a Claude transcript carrying ``correlation_id``'s marker."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    project_dir = _claude_project_dir(tmp_path, Path(record["cwd"]))
    project_dir.mkdir(parents=True, exist_ok=True)
    stamp = (
        datetime.fromtimestamp(float(record["spawned_at"]) + 5, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    rows = [
        {
            "type": "user",
            "timestamp": stamp,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": f"task {correlation_marker_token(correlation_id)}",
            },
        },
        {
            "type": "assistant",
            "timestamp": stamp,
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        },
    ]
    path = project_dir / f"{session_id}.jsonl"
    body = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(body, encoding="utf-8")
    mtime = float(record["spawned_at"]) + 10
    os.utime(path, (mtime, mtime))


@pytest.mark.asyncio
async def test_correlation_id_survives_spawn_restart_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, monkeypatch)
    result = await server_simple.spawn_agent(
        "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )
    correlation_id = _record(result["session_id"])["correlation_id"]

    # Simulate a restart: re-read the record from disk and go through the
    # read path, which must recover the binding from the persisted id alone.
    # A2 moved the scan into the ladder, so the observable is the binding
    # itself rather than an argument handed to the reader.
    record = _record(result["session_id"])
    _write_marked_transcript(
        tmp_path, monkeypatch, record, correlation_id, session_id="restarted-session"
    )

    binding = server_simple._resolve_agent_binding(record)

    assert binding.outcome == ao.BINDING_BOUND
    assert binding.output is not None
    assert binding.output.backend_session_id == "restarted-session"


def test_absent_correlation_field_is_legacy_and_never_rederived() -> None:
    record = {"name": "worker", "session_id": "sess"}

    status, correlation_id = classify_correlation(record)

    assert status == "legacy"
    assert correlation_id is None


@pytest.mark.parametrize("value", ["", "   ", 42, None, ["x"], {}])
def test_malformed_correlation_field_is_unverified(value: object) -> None:
    record = {"name": "worker", "session_id": "sess", "correlation_id": value}

    status, correlation_id = classify_correlation(record)

    assert status == "unverified"
    assert correlation_id is None


@pytest.mark.parametrize("value", ["", "   ", 42, None, ["x"], {}, True])
def test_persisted_malformed_id_after_restart_is_unverified_not_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    """The full persist -> reload-from-disk -> read path, not just the classifier.

    Absent and malformed must never be conflated: **absent** means the record
    predates correlation (a compatibility case), **malformed** means it is
    corrupt. Only the first is a compatibility case, and neither is ever
    resolved by re-deriving an id. Proving this at the classifier alone says
    nothing about what survives a JSON round trip — ``True`` in particular is
    an ``int`` subclass, and an unconstrained reader could accept it.
    """
    _session(tmp_path, monkeypatch)
    result = await_spawn(tmp_path)

    # Corrupt the persisted id on disk, then reload as a restarted server would.
    agents = server_simple._load_agents(result)
    agents[0]["correlation_id"] = value
    server_simple._save_agents(result, agents)
    reloaded = _record(result)

    assert reloaded["correlation_id"] == value or reloaded["correlation_id"] is None
    assert classify_correlation(reloaded) == ("unverified", None)

    binding = server_simple._resolve_agent_binding(reloaded)

    assert binding.outcome == ao.BINDING_UNVERIFIED, (
        "a corrupt persisted id must be unverified, never legacy: legacy means "
        "'predates correlation' and is the only compatibility case"
    )
    assert binding.output is None, "an unverified read may not carry transcript data"


def await_spawn(tmp_path: Path) -> str:
    """Spawn one agent synchronously and return its session id."""
    import asyncio

    result = asyncio.run(
        server_simple.spawn_agent(
            "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
        )
    )
    return str(result["session_id"])


def test_valid_correlation_field_is_valid() -> None:
    status, correlation_id = classify_correlation({"correlation_id": "abc123"})

    assert status == "valid"
    assert correlation_id == "abc123"


def test_legacy_record_reads_without_a_correlation_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        ao,
        "read_codex_output",
        lambda *args, **kwargs: seen.update(kwargs) or None,
    )

    binding = server_simple._resolve_agent_binding(
        {
            "name": "worker",
            "session_id": "sess",
            "backend": "codex",
            "spawned_at": 100.0,
            "cwd": str(tmp_path),
        }
    )

    # No id was ever issued for this record, so none is invented: the ladder
    # stops at the metadata gate and falls back to the un-correlated reader.
    assert binding.outcome == ao.BINDING_LEGACY
    assert "correlation_token" not in seen


@pytest.mark.asyncio
async def test_reused_agent_name_after_kill_gets_a_distinct_correlation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, monkeypatch)
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda *a, **k: False
    )

    first = await server_simple.spawn_agent(
        "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )
    first_id = _record(first["session_id"])["correlation_id"]
    await server_simple.kill_agent("worker")
    second = await server_simple.spawn_agent(
        "plain prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )
    second_id = _record(second["session_id"])["correlation_id"]

    assert first_id != second_id
    assert correlation_marker_token(first_id) != correlation_marker_token(second_id)


def test_new_correlation_ids_are_unique() -> None:
    assert len({ao.new_correlation_id() for _ in range(100)}) == 100

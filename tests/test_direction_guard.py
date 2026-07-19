"""R2 — session resume is downstream-only (C1 persistence, C2 enforcement).

Scope note, deliberately stated here because it bounds every assertion below:

**Spoofing is out of scope.** ``server_simple.IDENTITY`` is read from an env
var at import time by the *caller's own* process, and the MCP server enforcing
this rule IS that process. A worker can therefore assert any identity it likes,
and a worker with filesystem access can edit ``agents.json`` directly. Nothing
in this file tests, or claims, resistance to a deliberate bypass. What it does
test is that the *accidental* upstream resume — a worker calling
``follow_up_agent`` on its own coordinator, which is kill-and-respawn and so
restarts that coordinator's process — is refused, and refused before anything
observable has changed.

The registry is flat: every agent in a session shares one ``agents.json`` and
there are no sub-sessions. Parentage is therefore a field on the record, not a
tree structure, which is why a sibling is a distinct case from a child.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import Result
from typer.testing import CliRunner

from claude_teams import cli, server_simple
from claude_teams.agent_output import (
    BINDING_BOUND,
    SPAWNED_BY_FIELD,
    SPAWNED_BY_SOURCE_FIELD,
    SPAWNED_BY_SOURCE_OPERATOR,
    SPAWNED_BY_SOURCE_SPAWN,
    AgentOutput,
    BindingResult,
)
from claude_teams.delivery import DELIVERY_MARKER_PREFIX
from tests.test_follow_up_delivery import (
    _append,
    _Clock,
    _FakeRegistry,
    _FakeResumeBackend,
)

SESSION = "guard-session"

#: name -> recorded spawner. ``None`` means the field is absent entirely, i.e.
#: a record written before C1 shipped.
PARENTAGE: dict[str, str | None] = {
    "lead-a": "team-lead",
    "worker-b": "lead-a",
    "worker-c": "lead-a",
    "orphan": None,
}


def _backend_session(name: str) -> str:
    return f"bs-{name}"


def _user_record(backend_session_id: str, text: str) -> dict:
    return {
        "type": "user",
        "sessionId": backend_session_id,
        "message": {"role": "user", "content": text},
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A flat session holding a lead, two siblings under it, and an orphan."""
    session_dir = tmp_path / "sessions" / SESSION
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", SESSION)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})

    work = tmp_path / "work"
    work.mkdir()

    transcripts: dict[str, Path] = {}
    records: list[dict] = []
    for name, spawner in PARENTAGE.items():
        transcript = tmp_path / f"{name}.jsonl"
        _append(transcript, _user_record(_backend_session(name), "the original task"))
        transcripts[name] = transcript
        record: dict[str, object] = {
            "name": name,
            "pid": 123,
            "backend": "claude-code",
            "session_id": SESSION,
            "status": "running",
            "spawned_at": 100.0,
            "cwd": str(work),
            "backend_session_id": _backend_session(name),
            "model": "model",
            "permission_mode": "bypass",
            "reasoning_effort": None,
            "correlation_id": f"corr-{name}",
        }
        if spawner is not None:
            record[SPAWNED_BY_FIELD] = spawner
            record[SPAWNED_BY_SOURCE_FIELD] = SPAWNED_BY_SOURCE_SPAWN
        records.append(record)

    server_simple._save_agents(SESSION, records)
    server_simple._persist_session_binding(SESSION)

    def _binding(agent: dict) -> BindingResult:
        name = str(agent.get("name"))
        return BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=900.0,
                last_message="done",
                rollout_path=str(transcripts[name]),
                backend_session_id=_backend_session(name),
            ),
        )

    monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding)
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)
    clock = _Clock()
    monkeypatch.setattr(server_simple, "_delivery_clock", clock)
    monkeypatch.setattr(server_simple, "_delivery_sleep", clock.sleep)
    monkeypatch.setattr(server_simple, "_DELIVERY_CALL_BUDGET_SECONDS", 10.0)
    monkeypatch.setattr(server_simple, "_DELIVERY_POLL_SECONDS", 1.0)
    return SimpleNamespace(
        tmp_path=tmp_path,
        session_dir=session_dir,
        transcripts=transcripts,
        clock=clock,
    )


def _as(monkeypatch: pytest.MonkeyPatch, identity: str) -> None:
    """Run the next call as ``identity``, i.e. as that agent's own MCP server."""
    monkeypatch.setattr(server_simple, "IDENTITY", identity)


def _install(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))


def _child_alive(monkeypatch: pytest.MonkeyPatch, alive: bool) -> None:
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (alive and handle == "789", "x"),
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by relative path.

    A whole-tree snapshot rather than targeted assertions: it catches the PID
    and generation in ``agents.json``, a regenerated MCP config, a prompt
    sidecar, and an acquired lease in one comparison, and it also catches a
    side effect nobody thought to name.
    """
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _record(name: str) -> dict:
    return next(a for a in server_simple._load_agents(SESSION) if a["name"] == name)


# ==========================================================================
# C2 — the upstream refusal, and that it changes nothing
# ==========================================================================


@pytest.mark.asyncio
async def test_worker_following_up_its_lead_is_refused(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "worker-b")

    result = await server_simple.follow_up_agent("lead-a", "please stop", "k14")

    assert result["success"] is False
    assert result["reason"] == "not_spawner"
    assert backend.resume_calls == []


@pytest.mark.asyncio
async def test_upstream_refusal_names_the_rule_and_points_at_send_message(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "worker-b")

    result = await server_simple.follow_up_agent("lead-a", "please stop", "k15")

    detail = result["detail"]
    assert "send_message" in detail
    assert "downstream" in detail
    assert "lead-a" in detail
    assert result["retriable"] is False


@pytest.mark.asyncio
async def test_upstream_refusal_changes_nothing_on_disk(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PID, no MCP config, no prompt sidecar — and after Phase A, no lease
    acquired and no generation bump either.
    """
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "worker-b")
    before = _snapshot(env.session_dir)
    before_generation = server_simple._record_generation(_record("lead-a"))

    await server_simple.follow_up_agent("lead-a", "please stop", "k16")

    assert _snapshot(env.session_dir) == before
    assert server_simple._record_generation(_record("lead-a")) == before_generation
    assert not server_simple._leases_file(SESSION).exists()


@pytest.mark.asyncio
async def test_sibling_follow_up_is_refused(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Siblings share a spawner but neither spawned the other."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "worker-b")

    result = await server_simple.follow_up_agent("worker-c", "do my work", "k17")

    assert result["success"] is False
    assert result["reason"] == "not_spawner"
    assert backend.resume_calls == []


@pytest.mark.asyncio
async def test_record_without_a_spawner_refuses_rather_than_allowing(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-C1 record cannot be backfilled, and a silent allow would disable
    the guard exactly during the upgrade window it matters in.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "team-lead")

    result = await server_simple.follow_up_agent("orphan", "carry on", "k18")

    assert result["success"] is False
    assert result["reason"] == "parent_unknown"
    assert backend.resume_calls == []
    assert "adopt" in result["detail"]


# ==========================================================================
# C2 — the downstream direction still works
# ==========================================================================


@pytest.mark.asyncio
async def test_nested_lead_can_follow_up_the_agent_it_spawned(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lead-a`` is itself a spawned agent; it may still resume its own child."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcripts["worker-b"],
            _user_record(
                _backend_session("worker-b"),
                f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}",
            ),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "lead-a")

    result = await server_simple.follow_up_agent("worker-b", "next prompt", "k19")

    assert result["success"] is True
    assert result["status"] == "delivered"
    assert _record("worker-b")["pid"] == 789


@pytest.mark.asyncio
async def test_resume_preserves_the_spawner_on_the_record(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_receipt(nonce: str) -> None:
        _append(
            env.transcripts["worker-b"],
            _user_record(
                _backend_session("worker-b"),
                f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}",
            ),
        )

    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "lead-a")

    await server_simple.follow_up_agent("worker-b", "next prompt", "k20")

    after = _record("worker-b")
    assert after[SPAWNED_BY_FIELD] == "lead-a"
    assert after[SPAWNED_BY_SOURCE_FIELD] == SPAWNED_BY_SOURCE_SPAWN


# ==========================================================================
# C1 — the field is written at spawn
# ==========================================================================


@pytest.mark.asyncio
async def test_spawn_records_the_spawner_from_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", SESSION)
    (tmp_path / "sessions" / SESSION / "mcp").mkdir(parents=True)
    server_simple._save_agents(SESSION, [])
    monkeypatch.setattr(server_simple, "IDENTITY", "lead-a")

    class _SpawnBackend:
        def resolve_launch(self, model, effort):
            return (model or "model", effort)

        def spawn(self, request):
            return SimpleNamespace(process_handle="4242")

    _install(monkeypatch, _SpawnBackend())
    monkeypatch.setattr(
        server_simple.process_manager, "creation_token", lambda handle: "tok"
    )

    await server_simple.spawn_agent(
        "do a thing", name="new-worker", backend="claude-code"
    )

    record = _record("new-worker")
    assert record[SPAWNED_BY_FIELD] == "lead-a"
    assert record[SPAWNED_BY_SOURCE_FIELD] == SPAWNED_BY_SOURCE_SPAWN


# ==========================================================================
# Recovery is an operator action, not an agent-callable tool
# ==========================================================================


@pytest.mark.asyncio
async def test_adopt_agent_is_not_registered_as_an_mcp_tool(env) -> None:
    """An agent-callable adopt would reintroduce the hole C2 closes.

    "Callable only by a caller claiming parentage" is tautological: the
    operation itself writes the caller as the spawner, and the caller's
    identity is self-asserted. A confused worker could adopt its own lead and
    then pass the direction check.
    """
    names = {tool.name for tool in await server_simple.mcp._list_tools()}

    assert "follow_up_agent" in names, "the listing is real"
    assert "adopt_agent" not in names
    assert not any("adopt" in name for name in names)


def _lead_token() -> str:
    return server_simple._ensure_lead_token(SESSION)


def _adopt(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["adopt", *args])


def test_cli_adopt_writes_the_spawner_as_operator_asserted(env) -> None:
    generation = server_simple._record_generation(_record("orphan"))

    result = _adopt(
        SESSION,
        "orphan",
        "team-lead",
        "--token",
        _lead_token(),
        "--expect-generation",
        str(generation),
    )

    assert result.exit_code == 0, result.output
    record = _record("orphan")
    assert record[SPAWNED_BY_FIELD] == "team-lead"
    assert record[SPAWNED_BY_SOURCE_FIELD] == SPAWNED_BY_SOURCE_OPERATOR
    assert json.loads(result.stdout)["adopted"] is True


def test_cli_adopt_refuses_a_stale_generation(env) -> None:
    generation = server_simple._record_generation(_record("orphan"))

    result = _adopt(
        SESSION,
        "orphan",
        "team-lead",
        "--token",
        _lead_token(),
        "--expect-generation",
        str(generation + 1),
    )

    assert result.exit_code != 0
    # Asserted on the message, not merely on a non-zero exit: an unknown
    # subcommand also exits non-zero, which would let this pass vacuously.
    assert "generation" in result.output
    assert SPAWNED_BY_FIELD not in _record("orphan")


def test_cli_adopt_refuses_a_bad_token(env) -> None:
    result = _adopt(
        SESSION,
        "orphan",
        "team-lead",
        "--token",
        "not-the-token",
        "--expect-generation",
        "0",
    )

    assert result.exit_code == 2
    assert "recovery token" in result.output
    assert SPAWNED_BY_FIELD not in _record("orphan")


@pytest.mark.asyncio
async def test_cli_adopted_record_then_passes_the_direction_guard(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adoption is the only supported recovery path for a pre-C1 record."""
    _adopt(
        SESSION,
        "orphan",
        "team-lead",
        "--token",
        _lead_token(),
        "--expect-generation",
        str(server_simple._record_generation(_record("orphan"))),
    )

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcripts["orphan"],
            _user_record(
                _backend_session("orphan"),
                f"carry on {DELIVERY_MARKER_PREFIX}{nonce}",
            ),
        )

    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)
    _as(monkeypatch, "team-lead")

    result = await server_simple.follow_up_agent("orphan", "carry on", "k21")

    assert result["success"] is True
    assert _record("orphan")[SPAWNED_BY_SOURCE_FIELD] == SPAWNED_BY_SOURCE_OPERATOR

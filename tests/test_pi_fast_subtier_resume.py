"""Regression: pi fast subtiers survive spawn persistence and follow-up.

A tier name must never be persisted or re-resolved. ``spawn_agent`` resolves
``medium-fast``/``high-fast`` to a concrete ``(slug, thinking)`` pair before
building the request, records those concrete values in ``agents.json``, and
``_build_resume_request`` deliberately reuses them verbatim rather than
resolving the tier a second time. These tests walk the real server path —
spawn request, persisted record, resume request, resume argv — so a later
ladder remap cannot silently move a running agent to a different model.
"""

from pathlib import Path

import pytest

from claude_teams import server_simple as ss
from claude_teams.backends.base import SpawnRequest, SpawnResult
from claude_teams.backends.pi import PiBackend
from claude_teams.backends.registry import canonical_backend_name

PAIRS = [
    ("medium-fast", "gpt-6-sol", "low"),
    ("high-fast", "gpt-6-sol", "medium"),
]


class _CapturingPi(PiBackend):
    """The real pi backend with only the process-launch boundary stubbed out."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: SpawnRequest | None = None

    def _available_model_ids(self) -> list[str]:
        # Empty discovery: skip catalog validation, exercise the tier table.
        return []

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        self.last_request = request
        return SpawnResult(process_handle="4321", backend_type="pi")


class _PiRegistry:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def resolve_name(self, name: str) -> str:
        return canonical_backend_name(name)

    def default_backend(self) -> str:
        return "pi"

    def get(self, backend: str) -> object:
        assert backend == "pi"
        return self._backend


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CapturingPi:
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
    monkeypatch.setattr(ss, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(ss, "_session_id", "")
    # Keep the real-home ~/.pi write out of the test.
    monkeypatch.setattr(ss, "_ensure_pi_mcp_config", lambda: None)
    monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["node", "cli.js"])
    backend = _CapturingPi()
    monkeypatch.setattr(ss, "registry", _PiRegistry(backend))
    return backend


@pytest.mark.parametrize(("tier", "slug", "thinking"), PAIRS)
@pytest.mark.asyncio
async def test_fast_tier_spawn_persists_concrete_pair(
    server: _CapturingPi, tmp_path: Path, tier: str, slug: str, thinking: str
) -> None:
    """The tier name resolves before persistence and is never itself stored."""
    result = await ss.spawn_agent(
        "prompt", name="worker", backend="pi", model=tier, cwd=str(tmp_path)
    )

    assert server.last_request is not None
    assert (server.last_request.model, server.last_request.reasoning_effort) == (
        slug,
        thinking,
    )

    records = ss._load_agents(result["session_id"])
    record = next(r for r in records if r["name"] == "worker")
    assert (record["model"], record["reasoning_effort"]) == (slug, thinking)
    assert record["model"] != tier


@pytest.mark.parametrize(("tier", "slug", "thinking"), PAIRS)
@pytest.mark.asyncio
async def test_resume_reuses_the_persisted_pair_verbatim(
    server: _CapturingPi, tmp_path: Path, tier: str, slug: str, thinking: str
) -> None:
    """Follow-up carries the stored slug/effort through without re-resolving."""
    result = await ss.spawn_agent(
        "prompt", name="worker", backend="pi", model=tier, cwd=str(tmp_path)
    )
    session_id = result["session_id"]
    record = next(r for r in ss._load_agents(session_id) if r["name"] == "worker")

    model, _permission, effort, _cid, request, _extra = ss._build_resume_request(
        session_id,
        record,
        "worker",
        str(tmp_path),
        server,
        "pi",
        "follow-up prompt",
        "nonce-abc",
    )

    assert (model, effort) == (slug, thinking)
    assert (request.model, request.reasoning_effort) == (slug, thinking)

    cmd = PiBackend().build_resume_command(request, "backend-sid")
    assert cmd[cmd.index("--model") + 1] == f"openai-codex/{slug}"
    assert cmd[cmd.index("--thinking") + 1] == thinking

"""§2 restart safety — the fail-closed invariant end to end.

After a server/host restart the process manager has no in-memory ownership, so
a recovered agent record must never let follow_up_agent shut down or kill a
PID it cannot prove is still ours (tokenless or token-mismatched). Instead the
agent is resumed via its backend_session_id. Resume must also persist a fresh
create token.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple as ss
from claude_teams.backends.contracts import SpawnRequest


class _FakeResumeBackend:
    def __init__(self) -> None:
        self.resume_calls: list[tuple[SpawnRequest, str]] = []

    def supports_resume(self) -> bool:
        return True

    def default_model(self) -> str:
        return "model"

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        self.resume_calls.append((request, backend_session_id))
        return SimpleNamespace(process_handle="789")


class _FakeRegistry:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def get(self, backend: str) -> object:
        return self.backend


@pytest.fixture
def follow_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    backend = _FakeResumeBackend()
    session_dir = tmp_path / "sessions" / "session-id"
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(ss, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(ss, "_session_id", "session-id")
    monkeypatch.setattr(ss, "registry", _FakeRegistry(backend))
    monkeypatch.setattr(ss, "_inbox_locks", {})
    return SimpleNamespace(backend=backend, tmp_path=tmp_path)


def _write_agent(tmp_path: Path, **overrides: object) -> None:
    record = {
        "name": "worker",
        "pid": 123,
        "backend": "codex",
        "session_id": "session-id",
        "status": "running",
        "spawned_at": 100.0,
        "cwd": str(tmp_path / "work"),
        "backend_session_id": "backend-session-id",
        "model": "model",
        "permission_mode": "bypass",
        "reasoning_effort": None,
    }
    record.update(overrides)
    ss._save_agents("session-id", [record])


def _no_shutdown(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"graceful": [], "kill": []}
    monkeypatch.setattr(
        ss.process_manager,
        "graceful_shutdown",
        lambda h, timeout_s=10.0: calls["graceful"].append(h) or True,
    )
    monkeypatch.setattr(
        ss.process_manager, "kill_process", lambda h, *a, **k: calls["kill"].append(h)
    )
    return calls


def _idle_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Looks alive+idle so the (pre-gate) code would otherwise shut it down.
    monkeypatch.setattr(ss.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(
        ss,
        "read_codex_output",
        lambda spawned_at, cwd, **kwargs: SimpleNamespace(
            last_activity_at=900.0,
            last_message="done",
            backend_session_id="backend-session-id",
            busy_hint=False,
        ),
    )


def test_tokenless_recovered_record_never_graceful_shutdowns_pid(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No create_token (record predates tokens) but the PID looks alive+idle.
    _write_agent(follow_up.tmp_path)  # no create_token
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (True, "alive"),
    )
    _idle_output(monkeypatch)
    calls = _no_shutdown(monkeypatch)

    result = asyncio.run(ss.follow_up_agent("worker", "continue"))

    assert result["success"] is True
    assert calls["graceful"] == []  # never touched the unproven PID
    assert calls["kill"] == []
    assert follow_up.backend.resume_calls[0][1] == "backend-session-id"


def test_reused_pid_does_not_get_graceful_shutdown(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stored token mismatches the live PID's token (PID reuse after reboot).
    _write_agent(follow_up.tmp_path, create_token="stale-token")  # noqa: S106
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(ss.process_manager, "owns_process", lambda h, token: False)
    _idle_output(monkeypatch)
    calls = _no_shutdown(monkeypatch)

    result = asyncio.run(ss.follow_up_agent("worker", "continue"))

    assert result["success"] is True
    assert calls["graceful"] == []
    assert calls["kill"] == []
    assert follow_up.backend.resume_calls[0][1] == "backend-session-id"


def test_agent_alive_prefers_resolved_agent_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # B2: for launcher-style backends _agent_alive must check the resolved
    # agent PID, not the (possibly exited) launcher PID.
    agent = {"pid": 100, "session_id": "s", "name": "w", "create_token": None}
    monkeypatch.setattr(ss.process_manager, "resolve_agent_pid", lambda h, t, a: "200")
    seen: dict[str, str] = {}

    def fake_health(handle: str, expected_token: str | None = None) -> tuple[bool, str]:
        seen["handle"] = handle
        return (True, "alive")

    monkeypatch.setattr(ss.process_manager, "health_check", fake_health)

    assert ss._agent_alive(agent) is True
    assert seen["handle"] == "200"  # the real agent PID, not launcher 100


def test_follow_up_resume_persists_new_create_token(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_agent(follow_up.tmp_path, create_token="stale-token")  # noqa: S106
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (False, "dead"),
    )
    _idle_output(monkeypatch)
    monkeypatch.setattr(
        ss.process_manager, "creation_token", lambda h: f"fresh-token-{h}"
    )

    result = asyncio.run(ss.follow_up_agent("worker", "continue"))

    assert result["success"] is True
    agents = ss._load_agents("session-id")
    assert agents[0]["create_token"] == "fresh-token-789"  # noqa: S105

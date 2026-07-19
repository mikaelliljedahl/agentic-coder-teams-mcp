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
from claude_teams.agent_output import BINDING_BOUND, AgentOutput, BindingResult
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.delivery import DeliveryOutcome


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
    # These tests are about the fail-closed PID gate and token persistence, not
    # about A4 confirmation, so the confirmation outcome is pinned exactly as
    # the binding is. Confirmation itself is proven against real transcripts in
    # tests/test_delivery_confirmation.py and tests/test_follow_up_delivery.py.
    monkeypatch.setattr(
        ss, "confirm_delivery", lambda *a, **k: DeliveryOutcome("delivered", "")
    )
    return SimpleNamespace(backend=backend, tmp_path=tmp_path)


def _write_agent(tmp_path: Path, **overrides: object) -> None:
    record: dict[str, object] = {
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
        # R2: follow-up is downstream-only, and the default test IDENTITY is
        # the root lead. The direction guard itself is covered in
        # tests/test_direction_guard.py.
        "spawned_by": "team-lead",
        "spawned_by_source": "spawn",
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
        "_resolve_agent_binding",
        lambda agent: BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=900.0,
                last_message="done",
                rollout_path="t.jsonl",
                backend_session_id="backend-session-id",
            ),
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

    result = asyncio.run(ss.follow_up_agent("worker", "continue", "k39"))

    assert result["success"] is True
    assert calls["graceful"] == []  # never touched the unproven PID
    assert calls["kill"] == []
    assert follow_up.backend.resume_calls[0][1] == "backend-session-id"


def test_reused_pid_does_not_get_graceful_shutdown(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stored token mismatches the live PID's token (PID reuse after reboot).
    _write_agent(follow_up.tmp_path, create_token="stale-token")
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(ss.process_manager, "owns_process", lambda h, token: False)
    _idle_output(monkeypatch)
    calls = _no_shutdown(monkeypatch)

    result = asyncio.run(ss.follow_up_agent("worker", "continue", "k40"))

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


def _busy_by_timer_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Alive with RECENT activity (10s ago < _FOLLOW_UP_IDLE_SECONDS): the
    # inactivity timer alone would classify this agent as busy.
    monkeypatch.setattr(ss.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(
        ss,
        "_resolve_agent_binding",
        lambda agent: BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=990.0,
                last_message="done",
                rollout_path="t.jsonl",
                backend_session_id="backend-session-id",
            ),
        ),
    )


def _write_state_marker(state: str) -> None:
    path = ss._state_marker_file("session-id", "worker")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'{{"state": "{state}", "event": "Stop", "ts": 990.0}}', encoding="utf-8"
    )


def test_waiting_marker_allows_immediate_follow_up(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A "waiting" state marker is authoritative: even though activity is recent
    # (the inactivity timer would say busy), the agent is parked at a wait hook
    # and must be resumable immediately.
    _write_agent(follow_up.tmp_path, create_token="tok")
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(ss.process_manager, "owns_process", lambda h, token: True)
    _busy_by_timer_output(monkeypatch)
    _no_shutdown(monkeypatch)
    _write_state_marker("waiting")

    result = asyncio.run(ss.follow_up_agent("worker", "continue", "k41"))

    assert result["success"] is True
    assert follow_up.backend.resume_calls[0][1] == "backend-session-id"


def test_recent_activity_without_waiting_marker_waits_rather_than_resuming(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No "waiting" marker: the inactivity timer still guards a genuinely busy
    # agent (recent activity, running marker) from being torn down. Since B2
    # that guard is a bounded wait rather than a refusal — but the invariant
    # under test is unchanged: the busy agent's process is NOT replaced.
    monkeypatch.setattr(ss, "_DELIVERY_CALL_BUDGET_SECONDS", 0.0)
    _write_agent(follow_up.tmp_path, create_token="tok")
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (True, "alive"),
    )
    _busy_by_timer_output(monkeypatch)
    _write_state_marker("running")

    result = asyncio.run(ss.follow_up_agent("worker", "continue", "k42"))

    assert result["success"] is False
    assert result["status"] == "queued"
    assert result["phase"] == "pending"
    assert follow_up.backend.resume_calls == []


def test_follow_up_resume_persists_new_create_token(
    follow_up: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_agent(follow_up.tmp_path, create_token="stale-token")
    monkeypatch.setattr(
        ss.process_manager,
        "health_check",
        lambda h, expected_token=None: (False, "dead"),
    )
    _idle_output(monkeypatch)
    monkeypatch.setattr(
        ss.process_manager, "creation_token", lambda h: f"fresh-token-{h}"
    )

    result = asyncio.run(ss.follow_up_agent("worker", "continue", "k43"))

    assert result["success"] is True
    agents = ss._load_agents("session-id")
    assert agents[0]["create_token"] == "fresh-token-789"

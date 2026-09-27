"""Part B trust preflight and launch-contract tests."""

import os

import pytest

from claude_teams import server_simple
from claude_teams.backends import process_manager as process_manager_module


@pytest.mark.parametrize(
    ("backend", "interactive", "binary", "cwd", "direct", "reason"),
    [
        (
            "pi",
            True,
            "/usr/bin/codex",
            "/workspace/ok",
            False,
            "trust_cwd_unsupported_backend",
        ),
        ("codex", False, "codex.cmd", "/workspace/bad'", True, "trust_cwd_headless"),
        (
            "codex",
            True,
            "codex.cmd",
            "/workspace/bad'",
            True,
            "trust_cwd_unsafe_transport",
        ),
        (
            "codex",
            True,
            "/usr/bin/codex",
            "/workspace/ok",
            True,
            "trust_cwd_unsafe_transport",
        ),
        (
            "codex",
            True,
            "/usr/bin/codex",
            "/workspace/bad'",
            False,
            "trust_cwd_unsafe_path",
        ),
    ],
)
def test_preflight_precedence(
    monkeypatch, backend, interactive, binary, cwd, direct, reason
):
    monkeypatch.setattr(
        process_manager_module, "codex_direct_launch_enabled", lambda: direct
    )
    assert (
        server_simple._trust_cwd_preflight(backend, cwd, interactive, binary) == reason
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_name", "interactive", "cwd", "reason"),
    [
        ("pi", True, "/workspace/ok", "trust_cwd_unsupported_backend"),
        ("codex", False, "/workspace/ok", "trust_cwd_headless"),
        ("codex", True, "/workspace/bad'", "trust_cwd_unsafe_path"),
        ("codex", True, "/workspace/bad\u2019", "trust_cwd_unsafe_path"),
        ("codex", True, "/workspace/ok\n", "trust_cwd_unsafe_path"),
    ],
)
async def test_trust_refusal_creates_no_session(
    tmp_path, monkeypatch, backend_name, interactive, cwd, reason
):
    from tests.test_correlation_transport import _FakeBackend, _FakeRegistry

    backend = _FakeBackend()
    monkeypatch.setattr(backend, "is_interactive", True, raising=False)
    monkeypatch.setattr(
        backend, "discover_binary", lambda: "/usr/bin/codex", raising=False
    )
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, backend_name))
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: interactive
    )
    result = await server_simple.spawn_agent(
        "hello", backend=backend_name, cwd=cwd, trust_cwd=True
    )
    assert result["reason"] == reason
    assert not (tmp_path / "sessions").exists()
    assert backend.last_request is None


@pytest.mark.asyncio
@pytest.mark.parametrize("trust_cwd", [False, True])
async def test_spawn_records_trust_and_pins_approved_launch(
    tmp_path, monkeypatch, trust_cwd
):
    from tests.test_correlation_transport import _FakeBackend, _FakeRegistry

    backend = _FakeBackend()
    monkeypatch.setattr(backend, "is_interactive", True, raising=False)
    monkeypatch.setattr(
        backend, "discover_binary", lambda: "/usr/bin/codex", raising=False
    )
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, "codex"))
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: True
    )

    result = await server_simple.spawn_agent(
        "hello", backend="codex", cwd=str(tmp_path), trust_cwd=trust_cwd
    )
    record = server_simple._load_agents(result["session_id"])[0]
    request = backend.last_request
    assert request is not None
    assert record["trust_cwd"] is trust_cwd
    assert record["launch_interactive"] is True
    extra = request.extra or {}
    if trust_cwd:
        assert extra["codex_trust_cwd"] == "1"
        assert extra["codex_trust_binary"] == "/usr/bin/codex"
        assert extra["codex_trust_interactive"] == "1"
    else:
        assert not any(key.startswith("codex_trust_") for key in extra)


def test_direct_launch_flag_does_not_refuse_on_posix(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        process_manager_module,
        "os",
        SimpleNamespace(name="posix", environ=os.environ),
    )
    monkeypatch.setenv("WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH", "1")
    assert process_manager_module.codex_direct_launch_enabled() is False


def test_spawn_doc_describes_security():
    doc = server_simple.spawn_agent.__doc__ or ""
    for phrase in ("repository-controlled", "managed", "interactive launches only"):
        assert phrase in doc

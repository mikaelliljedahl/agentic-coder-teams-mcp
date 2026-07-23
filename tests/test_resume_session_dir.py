"""Regression: the delivery resume request must carry ``session_dir``.

The delivery protocol extracted resume-request construction into
``server_simple._build_resume_request``. That extraction dropped the
``"session_dir"`` key that the spawn path sets, so a resumed pi agent fell back
to ``request.cwd`` for its ``--session-dir`` — a different, non-existent base
than spawn used — and ``--continue`` found no session, settling delivery as
``resume_not_confirmed``. These tests pin the key at the exact production
construction site and prove the downstream effect on the pi backend.
"""

from types import SimpleNamespace

import pytest

from claude_teams import server_simple
from claude_teams.backends.pi import PiBackend

SESSION = "session-xyz"
AGENT = "worker-pi"


class _FakeBackend:
    """Minimal stand-in; ``_build_resume_request`` only needs ``default_model``."""

    def default_model(self) -> str:
        return "gpt-5.6-sol"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp session base with a cwd deliberately distinct from it."""
    base = tmp_path / "sessions"
    session_dir = base / SESSION
    (session_dir / "mcp").mkdir(parents=True)
    (session_dir / "prompts").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", base)
    cwd = tmp_path / "repo"
    cwd.mkdir()
    return SimpleNamespace(base=base, session_dir=session_dir, cwd=str(cwd))


def _build_resume(agent_cwd: str, backend_name: str = "codex"):
    """Return the ``SpawnRequest`` the server builds for a resume."""
    result = server_simple._build_resume_request(
        SESSION,
        {"model": "gpt-5.6-sol"},
        AGENT,
        agent_cwd,  # intentionally NOT the session base
        _FakeBackend(),
        backend_name,
        "follow-up prompt",
        "nonce-abc",
    )
    return result[4]  # (model, permission_mode, effort, correlation_id, request, extra)


def test_resume_request_carries_authoritative_session_dir(env):
    """The resume ``extra`` must set ``session_dir`` to the session dir, not cwd.

    Backend-agnostic: the key is added before the per-backend hook spread, so
    ``codex`` is used here to keep the construction free of pi's real-home
    ``~/.pi`` write.
    """
    request = _build_resume(env.cwd, "codex")

    assert request.extra["session_dir"] == str(server_simple._session_dir(SESSION))
    # It must be the session base, never the (different) working directory.
    assert request.extra["session_dir"] != env.cwd


def test_resumed_pi_session_dir_is_the_session_base_not_cwd(env, monkeypatch):
    """Fed the server's resume request, pi targets the session base, not cwd."""
    # Faithful pi build, but keep it hermetic: skip the real-home ~/.pi write.
    monkeypatch.setattr(server_simple, "_ensure_pi_mcp_config", lambda: None)
    monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["node", "cli.js"])
    request = _build_resume(env.cwd, "pi")

    cmd = PiBackend().build_resume_command(request, "backend-sid")

    assert cmd.count("--session-dir") == 1
    got = cmd[cmd.index("--session-dir") + 1]
    expected = str(server_simple._session_dir(SESSION) / "pi-sessions" / AGENT)
    assert got == expected
    assert env.cwd not in got

    # The dropped key also stranded the pi state-marker env; it is back too.
    built_env = PiBackend().build_env(request)
    assert built_env["WIN_AGENT_TEAMS_SESSION_DIR"] == str(
        server_simple._session_dir(SESSION)
    )

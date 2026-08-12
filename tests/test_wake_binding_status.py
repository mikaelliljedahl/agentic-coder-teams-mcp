"""Tests for the ``wake_binding`` nudge returned by ``spawn_agent``.

The lead-wake hook binds to the conversation that installed it, and the
binding dies with that conversation's process. Nothing prompts a re-install,
because the symptom of a dead binding is *silence*. ``spawn_agent`` is the
moment the lead acquires something to be woken for, so it reports the binding
state there.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import hooks, procinfo, server_simple


def _claude_host(pid: int = 4242) -> procinfo.HostResolution:
    entry = procinfo.ProcessInfo(pid=pid, ppid=1, name="claude.exe")
    return procinfo.HostResolution(chain=(entry,), host=entry)


def _codex_host(pid: int = 99) -> procinfo.HostResolution:
    entry = procinfo.ProcessInfo(pid=pid, ppid=1, name="codex.exe")
    return procinfo.HostResolution(chain=(entry,), host=entry)


@pytest.fixture
def scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Point both settings scopes at empty temp dirs."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    return SimpleNamespace(project=project, home=home)


def _install_group(
    path: Path, *, session_dir: Path, owner_pid: int | None, token: str | None
) -> None:
    """Write a settings file carrying one lead-wake Stop group."""
    if owner_pid is None:
        matcher = hooks._wake_hook_matcher(
            session_dir, "team-lead", owner_mode="private"
        )
    else:
        matcher = hooks._wake_hook_matcher(
            session_dir,
            "team-lead",
            owner_mode="bound",
            owner_host_pid=owner_pid,
            owner_host_token=token,
        )
    config = server_simple._install_wake_hook({}, matcher, remove=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    server_simple._write_json_object_atomic(path, config)


class TestWakeBindingStatus:
    def test_absent_when_no_settings_file_exists(
        self, scopes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "absent"
        assert "install_lead_wake" in status["hint"]

    def test_bound_when_installed_group_matches_this_host(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.project / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=4242,
            token="tok",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "bound"
        assert "hint" not in status

    def test_stale_when_installed_group_names_another_process(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The restart case: same folder, different (new) host process."""
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.project / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=1111,
            token="other",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "stale"
        assert "install_lead_wake" in status["hint"]

    def test_stale_when_pid_matches_but_token_differs(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recycled PID must not read as a live binding."""
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.project / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=4242,
            token="stale-token",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "stale"

    def test_legacy_group_without_owner_binding_is_reported(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.project / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=None,
            token=None,
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "legacy"
        assert "install_lead_wake" in status["hint"]

    def test_user_scope_is_checked_too(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.home / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=4242,
            token="tok",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "bound"

    def test_a_bound_scope_wins_over_a_stale_one(
        self, scopes: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Groups from several scopes all run; one live binding is enough."""
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        _install_group(
            scopes.home / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=1111,
            token="other",
        )
        _install_group(
            scopes.project / ".claude" / "settings.json",
            session_dir=tmp_path,
            owner_pid=4242,
            token="tok",
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "bound"

    def test_non_claude_host_is_not_applicable(
        self, scopes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lead-wake hook is Claude-only; a Codex lead is not nagged."""
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _codex_host)

        status = server_simple._wake_binding_status()

        assert status["state"] == "not_applicable"
        assert "hint" not in status

    def test_walk_failure_is_unknown_and_never_raises(
        self, scopes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom() -> procinfo.HostResolution:
            raise OSError

        monkeypatch.setattr(procinfo, "resolve_nearest_host", _boom)

        status = server_simple._wake_binding_status()

        assert status["state"] == "unknown"

    def test_unreadable_token_is_unknown(
        self, scopes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: None,
        )

        status = server_simple._wake_binding_status()

        assert status["state"] == "unknown"

    def test_corrupt_settings_file_is_tolerated(
        self, scopes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(procinfo, "resolve_nearest_host", _claude_host)
        monkeypatch.setattr(
            server_simple.process_manager_module,
            "creation_token",
            lambda _pid: "tok",
        )
        settings = scopes.project / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text("{not json", encoding="utf-8")

        status = server_simple._wake_binding_status()

        assert status["state"] == "absent"


class TestSpawnAgentReportsBinding:
    @pytest.mark.asyncio
    async def test_spawn_result_carries_the_binding_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_spawn_agent_watch_contract import _FakeBackend, _FakeRegistry

        monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
        monkeypatch.setattr(server_simple, "_session_id", "")
        monkeypatch.setattr(server_simple, "registry", _FakeRegistry(_FakeBackend()))
        monkeypatch.setattr(
            server_simple,
            "_wake_binding_status",
            lambda: {"state": "stale", "hint": "call install_lead_wake"},
        )

        result = await server_simple.spawn_agent(
            "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
        )

        assert result["wake_binding"] == {
            "state": "stale",
            "hint": "call install_lead_wake",
        }

    @pytest.mark.asyncio
    async def test_binding_probe_failure_never_breaks_spawn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_spawn_agent_watch_contract import _FakeBackend, _FakeRegistry

        monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
        monkeypatch.setattr(server_simple, "_session_id", "")
        monkeypatch.setattr(server_simple, "registry", _FakeRegistry(_FakeBackend()))

        def _boom() -> dict:
            raise RuntimeError

        monkeypatch.setattr(server_simple, "_wake_binding_status", _boom)

        result = await server_simple.spawn_agent(
            "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
        )

        assert result["name"] == "worker"
        assert result["wake_binding"] == {"state": "unknown"}


def test_spawn_agent_docstring_documents_the_binding_field() -> None:
    description = server_simple.spawn_agent.__doc__ or ""

    assert "wake_binding" in description
    assert "install_lead_wake" in description

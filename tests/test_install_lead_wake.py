"""Tests for the ``install_lead_wake`` MCP tool and its pure upsert helper."""

import ast
import asyncio
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import lead_wake, procinfo
from claude_teams import server_simple as ss


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Relocate session base + home + cwd off real state (mirror guards suite)."""
    base = tmp_path / "sessions"
    base.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    monkeypatch.setattr(ss, "IDENTITY", ss.ROOT_LEAD_NAME)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    sid = str(uuid.uuid4())
    session = base / sid
    session.mkdir()
    (session / "agents.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(ss, "_session_id", sid)
    host = procinfo.ProcessInfo(101, 1, "claude.exe")
    monkeypatch.setattr(
        ss.procinfo,
        "resolve_nearest_host",
        lambda: procinfo.HostResolution(chain=(host,), host=host),
    )
    monkeypatch.setattr(
        ss.process_manager_module, "creation_token", lambda _pid: "token-a"
    )
    return SimpleNamespace(
        base=base, home=home, work=work, sid=sid, session=session, host=host
    )


def _stop_commands(config: dict) -> list[str]:
    return [g["hooks"][0]["command"] for g in config["hooks"]["Stop"]]


def _host(pid: int, name: str = "claude.exe") -> procinfo.HostResolution:
    entry = procinfo.ProcessInfo(pid, 1, name)
    return procinfo.HostResolution(chain=(entry,), host=entry)


class TestUpsertHelper:
    def test_install_is_idempotent_and_preserves_unrelated_hooks(self) -> None:
        cmd = "py -m claude_teams.lead_wake --reader team-lead"
        matcher = {"hooks": [{"type": "command", "command": cmd}]}
        config = {
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "my own emit hook"}]}
                ],
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "unrelated"}]}
                ],
            }
        }

        once = ss._install_wake_hook(config, matcher, remove=False)
        twice = ss._install_wake_hook(once, matcher, remove=False)

        wake = [c for c in _stop_commands(twice) if "claude_teams.lead_wake" in c]
        assert len(wake) == 1  # never duplicated
        assert "my own emit hook" in _stop_commands(twice)  # unrelated Stop kept
        assert "PreToolUse" in twice["hooks"]  # unrelated event kept

    def test_remove_drops_only_the_wake_group(self) -> None:
        cmd = "py -m claude_teams.lead_wake --reader team-lead"
        matcher = {"hooks": [{"type": "command", "command": cmd}]}
        config = {
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "my own emit hook"}]}
                ]
            }
        }
        installed = ss._install_wake_hook(config, matcher, remove=False)

        removed = ss._install_wake_hook(installed, matcher, remove=True)

        cmds = _stop_commands(removed)
        assert not any("claude_teams.lead_wake" in c for c in cmds)
        assert "my own emit hook" in cmds


class TestInstallLeadWakeTool:
    def test_install_writes_project_settings_with_wake_hook(
        self, _isolated: SimpleNamespace
    ) -> None:
        result = asyncio.run(ss.install_lead_wake())

        settings = _isolated.work / ".claude" / "settings.json"
        assert settings.exists()
        assert result["success"] is True
        assert Path(result["path"]) == settings
        config = json.loads(settings.read_text(encoding="utf-8"))
        cmds = _stop_commands(config)
        assert any("claude_teams.lead_wake" in c for c in cmds)
        assert any("--reader" in c and "team-lead" in c for c in cmds)
        wake = next(c for c in cmds if "claude_teams.lead_wake" in c)
        assert '"--owner-mode" "bound"' in wake
        assert '"--owner-host-pid" "101"' in wake
        assert '"--owner-host-token" "token-a"' in wake
        assert shlex.split(wake, posix=True) == ss.hooks._wake_command(
            _isolated.session,
            "team-lead",
            owner_mode="bound",
            owner_host_pid=101,
            owner_host_token="token-a",
        )
        assert result["binding"] == {
            "scope": "conversation",
            "survives_restart": False,
        }
        assert "re-run install_lead_wake" in result["note"]

    @pytest.mark.skipif(os.name != "nt", reason="cmd.exe parsing is Windows-only")
    def test_shell_command_round_trips_argv_through_windows_cmd(self) -> None:
        expected = ["C:/session dir", "token with spaces"]
        argv = [
            sys.executable,
            "-c",
            "import sys;print(repr(sys.argv[1:]))",
            *expected,
        ]
        command = ss.hooks._shell_quote_command(argv)

        completed = subprocess.run(  # noqa: S602
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert ast.literal_eval(completed.stdout.strip()) == expected

    def test_install_is_idempotent_through_tool(
        self, _isolated: SimpleNamespace
    ) -> None:
        asyncio.run(ss.install_lead_wake())
        asyncio.run(ss.install_lead_wake())

        config = json.loads(
            (_isolated.work / ".claude" / "settings.json").read_text(encoding="utf-8")
        )
        wake = [c for c in _stop_commands(config) if "claude_teams.lead_wake" in c]
        assert len(wake) == 1

    def test_remove_through_tool(self, _isolated: SimpleNamespace) -> None:
        asyncio.run(ss.install_lead_wake())
        result = asyncio.run(ss.install_lead_wake(remove=True))

        assert result["action"] == "removed"
        config = json.loads(
            (_isolated.work / ".claude" / "settings.json").read_text(encoding="utf-8")
        )
        stop = config.get("hooks", {}).get("Stop", [])
        assert not any(
            "claude_teams.lead_wake" in g["hooks"][0]["command"] for g in stop
        )

    def test_scope_user_writes_home_settings(self, _isolated: SimpleNamespace) -> None:
        result = asyncio.run(ss.install_lead_wake(scope="user"))

        user_settings = _isolated.home / ".claude" / "settings.json"
        assert Path(result["path"]) == user_settings
        assert user_settings.exists()

    @pytest.mark.parametrize(
        ("reason", "setup"),
        [
            ("host_not_found", "missing"),
            ("host_token_unavailable", "token"),
            ("host_walk_failed", "walk"),
            ("no_active_session", "session"),
        ],
    )
    def test_refusal_preserves_existing_settings_bytes(
        self,
        _isolated: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
        reason: str,
        setup: str,
    ) -> None:
        settings = _isolated.work / ".claude" / "settings.json"
        settings.parent.mkdir()
        original = b'{"custom":  true, "hooks": {}}\r\n'
        settings.write_bytes(original)
        if setup == "missing":
            monkeypatch.setattr(
                ss.procinfo,
                "resolve_nearest_host",
                lambda: procinfo.HostResolution(chain=(_isolated.host,), host=None),
            )
        elif setup == "token":
            monkeypatch.setattr(
                ss.process_manager_module, "creation_token", lambda _pid: None
            )
        elif setup == "walk":
            monkeypatch.setattr(
                ss.procinfo,
                "resolve_nearest_host",
                lambda: (_ for _ in ()).throw(OSError("denied")),
            )
        else:
            monkeypatch.setattr(ss, "_session_id", "")
            monkeypatch.setattr(ss, "_active_session_id", lambda **_kw: "")

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is False
        assert result["reason"] == reason
        assert settings.read_bytes() == original

    def test_refusal_does_not_create_absent_settings_directory(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ss.procinfo,
            "resolve_nearest_host",
            lambda: procinfo.HostResolution(chain=(), host=None),
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "host_not_found"
        assert not (_isolated.work / ".claude").exists()

    def test_walk_failure_does_not_create_absent_settings_directory(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ss.procinfo,
            "resolve_nearest_host",
            lambda: (_ for _ in ()).throw(OSError("walk failed")),
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "host_walk_failed"
        assert not (_isolated.work / ".claude").exists()

    @pytest.mark.parametrize("failure", [PermissionError("denied"), None])
    def test_token_read_failure_refuses_before_touching_settings(
        self,
        _isolated: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
        failure: BaseException | None,
    ) -> None:
        if failure is None:
            monkeypatch.setattr(
                ss.process_manager_module, "creation_token", lambda _pid: None
            )
        else:
            monkeypatch.setattr(
                ss.process_manager_module,
                "creation_token",
                lambda _pid: (_ for _ in ()).throw(failure),
            )
        monkeypatch.setattr(
            ss,
            "_lead_wake_settings_path",
            lambda _scope: (_ for _ in ()).throw(
                AssertionError("settings path must not be resolved")
            ),
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "host_token_unavailable"
        assert not (_isolated.work / ".claude").exists()

    def test_nearest_codex_host_refuses_instead_of_using_outer_claude(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        codex = procinfo.ProcessInfo(202, 303, "codex.exe")
        outer_claude = procinfo.ProcessInfo(303, 1, "claude.exe")
        monkeypatch.setattr(
            ss.procinfo,
            "resolve_nearest_host",
            lambda: procinfo.HostResolution(chain=(codex, outer_claude), host=codex),
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "host_not_found"
        assert [row["name"] for row in result["chain"]] == [
            "codex.exe",
            "claude.exe",
        ]
        assert not (_isolated.work / ".claude").exists()

    def test_session_dir_must_exist_before_settings_path_is_resolved(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ss, "_session_id", str(uuid.uuid4()))
        monkeypatch.setattr(
            ss,
            "_lead_wake_settings_path",
            lambda _scope: (_ for _ in ()).throw(
                AssertionError("settings path must not be resolved")
            ),
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "no_active_session"
        assert not (_isolated.work / ".claude").exists()

    def test_remove_resolves_neither_owner_nor_session(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asyncio.run(ss.install_lead_wake())
        monkeypatch.setattr(
            ss.procinfo,
            "resolve_nearest_host",
            lambda: (_ for _ in ()).throw(AssertionError("owner resolution")),
        )
        monkeypatch.setattr(
            ss,
            "_active_session_id",
            lambda **_kw: (_ for _ in ()).throw(AssertionError("session resolution")),
        )

        result = asyncio.run(ss.install_lead_wake(remove=True))

        assert result["action"] == "removed"

    def test_reinstall_hands_ownership_to_last_successful_installer(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _isolated.work / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text(
            json.dumps(
                {
                    "custom": {"keep": True},
                    "hooks": {
                        "Stop": [
                            {"hooks": [{"type": "command", "command": "custom-stop"}]}
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        asyncio.run(ss.install_lead_wake())
        host_b = procinfo.ProcessInfo(202, 1, "claude.exe")
        monkeypatch.setattr(
            ss.procinfo,
            "resolve_nearest_host",
            lambda: procinfo.HostResolution(chain=(host_b,), host=host_b),
        )
        monkeypatch.setattr(
            ss.process_manager_module, "creation_token", lambda _pid: "token-b"
        )

        asyncio.run(ss.install_lead_wake())

        config = json.loads(settings.read_text(encoding="utf-8"))
        wake = next(c for c in _stop_commands(config) if "lead_wake" in c)
        assert "token-b" in wake
        assert '"202"' in wake
        assert "token-a" not in wake
        assert '"101"' not in wake
        assert "custom-stop" in _stop_commands(config)
        assert config["custom"] == {"keep": True}

    def test_failed_atomic_replace_preserves_original(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asyncio.run(ss.install_lead_wake())
        settings = _isolated.work / ".claude" / "settings.json"
        original = settings.read_bytes()
        original_replace = Path.replace
        replace_failure = OSError("replace failed")

        def _fail_replace(path: Path, target: Path) -> Path:
            if target == settings:
                raise replace_failure
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", _fail_replace)

        result = asyncio.run(ss.install_lead_wake())

        assert result == {"success": False, "reason": "settings_write_failed"}
        assert settings.read_bytes() == original
        assert json.loads(original)
        assert list(settings.parent.glob(f"{settings.name}.*.tmp")) == []

    def test_merged_scopes_keep_private_and_distinct_shared_groups(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        session = tmp_path / "session"
        session.mkdir()
        shared_a = ss.hooks._wake_hook_matcher(
            session,
            "team-lead",
            owner_mode="bound",
            owner_host_pid=101,
            owner_host_token="a",
        )
        shared_b = ss.hooks._wake_hook_matcher(
            session,
            "team-lead",
            owner_mode="bound",
            owner_host_pid=202,
            owner_host_token="b",
        )
        private = ss.hooks._wake_hook_matcher(session, "worker", owner_mode="private")
        project = ss._install_wake_hook({}, shared_a, remove=False)
        user = ss._install_wake_hook({}, shared_b, remove=False)
        explicit = {"hooks": {"Stop": [private]}}
        merged = [
            *project["hooks"]["Stop"],
            *user["hooks"]["Stop"],
            *explicit["hooks"]["Stop"],
        ]

        commands = [group["hooks"][0]["command"] for group in merged]
        assert len(commands) == 3
        assert any('"101"' in command and '"a"' in command for command in commands)
        assert any('"202"' in command and '"b"' in command for command in commands)
        assert any('"private"' in command for command in commands)

        monkeypatch.setattr(lead_wake, "_resolve_session_dir", lambda _arg: session)
        monkeypatch.setattr(
            lead_wake, "_live_subagent_names", lambda *_args: ["worker"]
        )
        monkeypatch.setattr(
            lead_wake.procinfo, "resolve_nearest_host", lambda: _host(101)
        )
        monkeypatch.setattr(
            lead_wake.process_manager, "creation_token", lambda _pid: "a"
        )
        owner = lead_wake.evaluate(
            {},
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=101,
            owner_host_token="a",
        )
        foreign = lead_wake.evaluate(
            {},
            reader_arg="team-lead",
            owner_mode="bound",
            owner_host_pid=202,
            owner_host_token="b",
        )
        private_result = lead_wake.evaluate(
            {}, reader_arg="worker", owner_mode="private"
        )
        assert (owner.code, owner.action) == ("D5", "block")
        assert (foreign.code, foreign.action) == ("D0b", "allow")
        assert (private_result.code, private_result.action) == ("D5", "block")

        project_removed = ss._install_wake_hook(project, shared_a, remove=True)
        assert project_removed == {}
        assert user["hooks"]["Stop"] == [shared_b]
        assert explicit["hooks"]["Stop"] == [private]

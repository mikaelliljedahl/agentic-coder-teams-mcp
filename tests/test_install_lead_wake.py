"""Tests for the ``install_lead_wake`` MCP tool and its pure upsert helper."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    return SimpleNamespace(base=base, home=home, work=work)


def _stop_commands(config: dict) -> list[str]:
    return [g["hooks"][0]["command"] for g in config["hooks"]["Stop"]]


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
        assert Path(result["path"]) == settings
        config = json.loads(settings.read_text(encoding="utf-8"))
        cmds = _stop_commands(config)
        assert any("claude_teams.lead_wake" in c for c in cmds)
        assert any("--reader" in c and "team-lead" in c for c in cmds)

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

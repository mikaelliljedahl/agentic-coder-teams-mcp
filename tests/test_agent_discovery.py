"""Tests for shared agent/profile discovery helpers.

Covers the defensive/error branches in
``claude_teams.backends._agent_discovery`` that the main happy-path tests do
not reach: malformed Codex ``config.toml`` inputs, project-shadows-home
collision resolution, and the env-var-driven Goose recipe scan.
"""

import os
from pathlib import Path

import pytest

from claude_teams.backends._agent_discovery import (
    discover_claude_agents,
    discover_codex_style_agents,
    discover_goose_recipes,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at an empty tmp dir so discovery never reads the
    real user's ``~/.claude`` or ``~/.codex`` config (hermetic isolation)."""
    home = tmp_path / "_clean_home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


# --------------------------------------------------------------------------
# discover_claude_agents — project shadows home on name collision
# --------------------------------------------------------------------------


def test_discover_claude_agents_no_dirs_returns_empty(tmp_path):
    # Neither project-local nor home ``.claude/agents`` exists -> the
    # ``not root.is_dir()`` skip fires for both roots.
    assert discover_claude_agents(str(tmp_path / "empty_proj")) == []


def test_discover_claude_agents_project_shadows_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    _write(home / ".claude" / "agents" / "shared.md", "home version")
    _write(cwd / ".claude" / "agents" / "shared.md", "project version")
    _write(home / ".claude" / "agents" / "home_only.md", "home only")

    profiles = {p.name: p for p in discover_claude_agents(str(cwd))}

    assert set(profiles) == {"shared", "home_only"}
    # Project-local entry wins (setdefault sees the project root first).
    assert profiles["shared"].path == str(cwd / ".claude" / "agents" / "shared.md")


# --------------------------------------------------------------------------
# discover_codex_style_agents — error / skip branches (lines 81-92)
# --------------------------------------------------------------------------


def test_codex_agents_skips_undecodable_toml(tmp_path):
    cwd = tmp_path / "proj"
    _write(cwd / ".codex" / "config.toml", "this is = not valid toml [[[")

    assert discover_codex_style_agents(str(cwd), "codex") == []


def test_codex_agents_skips_non_dict_agents_table(tmp_path):
    cwd = tmp_path / "proj"
    # ``agents`` is a scalar, not a table.
    _write(cwd / ".codex" / "config.toml", 'agents = "nope"\n')

    assert discover_codex_style_agents(str(cwd), "codex") == []


def test_codex_agents_skips_non_dict_entry(tmp_path):
    cwd = tmp_path / "proj"
    # ``agents.foo`` is a string value, not a sub-table.
    _write(cwd / ".codex" / "config.toml", '[agents]\nfoo = "bar"\n')

    assert discover_codex_style_agents(str(cwd), "codex") == []


def test_codex_agents_skips_entry_without_config_file(tmp_path):
    cwd = tmp_path / "proj"
    _write(
        cwd / ".codex" / "config.toml",
        # Missing config_file, empty config_file, and non-string config_file.
        '[agents.missing]\nmodel = "x"\n'
        '[agents.empty]\nconfig_file = ""\n'
        "[agents.nonstr]\nconfig_file = 42\n",
    )

    assert discover_codex_style_agents(str(cwd), "codex") == []


def test_codex_agents_yields_valid_entry(tmp_path):
    cwd = tmp_path / "proj"
    _write(
        cwd / ".codex" / "config.toml",
        '[agents.reviewer]\nconfig_file = "/personas/reviewer.md"\n',
    )

    profiles = discover_codex_style_agents(str(cwd), "codex")

    assert [(p.name, p.path) for p in profiles] == [
        ("reviewer", "/personas/reviewer.md")
    ]


def test_codex_agents_project_overrides_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    _write(
        home / ".codex" / "config.toml",
        '[agents.shared]\nconfig_file = "/home/shared.md"\n',
    )
    _write(
        cwd / ".codex" / "config.toml",
        '[agents.shared]\nconfig_file = "/proj/shared.md"\n',
    )

    profiles = {p.name: p for p in discover_codex_style_agents(str(cwd), "codex")}

    # Project config is read last and overwrites the home entry.
    assert profiles["shared"].path == "/proj/shared.md"


# --------------------------------------------------------------------------
# discover_goose_recipes — env-driven scan (lines 112-128)
# --------------------------------------------------------------------------


def test_goose_recipes_empty_env_returns_empty(monkeypatch):
    monkeypatch.delenv("GOOSE_RECIPE_PATH", raising=False)
    assert discover_goose_recipes("/anything") == []

    monkeypatch.setenv("GOOSE_RECIPE_PATH", "")
    assert discover_goose_recipes("/anything") == []


def test_goose_recipes_skips_blank_and_missing_dirs(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist"
    # A blank segment and a non-existent directory are both skipped.
    monkeypatch.setenv("GOOSE_RECIPE_PATH", os.pathsep.join(["", str(missing)]))

    assert discover_goose_recipes("/anything") == []


def test_goose_recipes_discovers_supported_extensions(tmp_path, monkeypatch):
    recipes = tmp_path / "recipes"
    _write(recipes / "a.yaml", "name: a")
    _write(recipes / "b.yml", "name: b")
    _write(recipes / "c.json", "{}")
    _write(recipes / "ignore.txt", "not a recipe")

    monkeypatch.setenv("GOOSE_RECIPE_PATH", str(recipes))

    names = {p.name for p in discover_goose_recipes("/anything")}
    assert names == {"a", "b", "c"}


def test_goose_recipes_first_dir_wins_on_collision(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(first / "shared.yaml", "from first")
    _write(second / "shared.yaml", "from second")

    monkeypatch.setenv("GOOSE_RECIPE_PATH", os.pathsep.join([str(first), str(second)]))

    profiles = {p.name: p for p in discover_goose_recipes("/anything")}
    # setdefault keeps the first directory's entry.
    assert profiles["shared"].path == str(first / "shared.yaml")

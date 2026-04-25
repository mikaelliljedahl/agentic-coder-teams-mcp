"""Tests for bundled skills and custom vendor providers."""

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Client

from claude_teams import skill_providers as sp
from claude_teams.skill_providers import (
    _CUSTOM_PROVIDER_ROOTS,
    build_custom_skills_providers,
)
from tests._server_support import _content_text

pytest_plugins = ["tests._server_support"]

# ---------------------------------------------------------------------------
# Bundled skills served via MCP resources
# ---------------------------------------------------------------------------


async def test_bundled_skills_discoverable(client: Client) -> None:
    """Bundled skills appear as MCP resources with skill:// URIs."""
    resources = await client.list_resources()
    uris = [str(r.uri) for r in resources]
    assert "skill://team-orchestration/SKILL.md" in uris


async def test_bundled_skill_manifest_present(client: Client) -> None:
    """Each bundled skill exposes a synthetic _manifest resource."""
    resources = await client.list_resources()
    uris = [str(r.uri) for r in resources]
    assert "skill://team-orchestration/_manifest" in uris


async def test_bundled_skill_content_readable(client: Client) -> None:
    """The team-orchestration skill content is readable and has frontmatter."""
    result = await client.read_resource("skill://team-orchestration/SKILL.md")
    text = _content_text(result[0])
    assert text.startswith("---")
    assert "name: team-orchestration" in text
    assert "description:" in text


async def test_bundled_skill_content_has_sections(client: Client) -> None:
    """The team-orchestration skill covers the key workflow sections."""
    result = await client.read_resource("skill://team-orchestration/SKILL.md")
    text = _content_text(result[0])
    assert "## Quick start" in text
    assert "## Team lifecycle" in text
    assert "## Coordination patterns" in text
    assert "## Troubleshooting" in text


# ---------------------------------------------------------------------------
# Bundled SKILL.md files on disk
# ---------------------------------------------------------------------------

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "src" / "claude_teams" / "skills"


def test_skills_directory_exists() -> None:
    """The bundled skills directory exists inside the package."""
    assert _SKILLS_DIR.is_dir()


def test_at_least_one_skill_bundled() -> None:
    """At least one SKILL.md is present in the bundled skills directory."""
    skills = list(_SKILLS_DIR.rglob("SKILL.md"))
    assert len(skills) >= 1


@pytest.mark.parametrize("skill_md", list(_SKILLS_DIR.rglob("SKILL.md")))
def test_skill_frontmatter_valid(skill_md: Path) -> None:
    """Every bundled SKILL.md has required frontmatter fields."""
    content = skill_md.read_text()
    assert content.startswith("---"), f"{skill_md} missing frontmatter delimiter"
    parts = content.split("---", 2)
    assert len(parts) >= 3, f"{skill_md} frontmatter not closed"
    frontmatter = parts[1]
    assert "name:" in frontmatter, f"{skill_md} missing 'name' field"
    assert "description:" in frontmatter, f"{skill_md} missing 'description' field"


# ---------------------------------------------------------------------------
# Custom vendor providers (data-driven table in skill_providers.py)
# ---------------------------------------------------------------------------

# Backend name -> path fragments that MUST appear in at least one root.
# Keep this in sync with ``_CUSTOM_PROVIDER_ROOTS``; the coverage assertion
# below tightly couples the two so drift fails loudly.
_EXPECTED_ROOT_FRAGMENTS: dict[str, list[str]] = {}


def test_custom_provider_table_covers_expected_backends() -> None:
    """Provider table and the test expectations must agree on backend coverage."""
    assert set(_CUSTOM_PROVIDER_ROOTS) == set(_EXPECTED_ROOT_FRAGMENTS)


@pytest.mark.parametrize(
    ("backend", "expected_fragments"),
    list(_EXPECTED_ROOT_FRAGMENTS.items()),
    ids=list(_EXPECTED_ROOT_FRAGMENTS.keys()),
)
def test_custom_provider_roots_include_expected_paths(
    backend: str, expected_fragments: list[str]
) -> None:
    """Each backend's root list contains its conventional skill paths."""
    root_strs = [str(p) for p in _CUSTOM_PROVIDER_ROOTS[backend]]
    for fragment in expected_fragments:
        assert any(fragment in r for r in root_strs), (
            f"{backend}: expected '{fragment}' in roots, got {root_strs}"
        )


def test_factory_builds_one_provider_per_backend() -> None:
    """The factory returns one SkillsDirectoryProvider per table entry."""
    providers = build_custom_skills_providers()
    assert set(providers) == set(_CUSTOM_PROVIDER_ROOTS)


def test_factory_wires_skill_md_as_main_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory passes ``main_file_name='SKILL.md'`` to every construction.

    Recorded via constructor monkey-patch rather than inspecting provider
    attributes so the test stays coupled to the factory's public contract
    (what it passes to ``SkillsDirectoryProvider``) and not to whatever
    private field FastMCP happens to store the main file name in.
    """
    seen: list[dict[str, object]] = []

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(sp, "SkillsDirectoryProvider", _Recorder)
    sp.build_custom_skills_providers()

    assert len(seen) == len(sp._CUSTOM_PROVIDER_ROOTS)
    if not seen:
        return
    assert {call["main_file_name"] for call in seen} == {"SKILL.md"}


def test_factory_preserves_root_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory forwards each backend's roots list in declared order.

    Uses a constructor monkey-patch to capture each ``roots`` argument in
    call order, then asserts the captured sequence matches the dict's
    declared iteration order. Relies on Python's guaranteed dict insertion
    order and avoids touching provider internals.
    """
    seen: list[dict[str, object]] = []

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(sp, "SkillsDirectoryProvider", _Recorder)
    sp.build_custom_skills_providers()

    expected_roots = [list(roots) for roots in sp._CUSTOM_PROVIDER_ROOTS.values()]
    actual_roots = [list(cast(Iterable[Path], call["roots"])) for call in seen]
    assert actual_roots == expected_roots

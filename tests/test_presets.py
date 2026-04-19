"""Unit tests for the team preset registry."""

import pytest

from claude_teams import presets
from claude_teams.presets import PresetMemberSpec, TeamPreset


@pytest.fixture(autouse=True)
def _reset_registry():
    """Restore seeded built-in presets after each test.

    Tests mutate the module-level registry; seeding on teardown keeps
    order-independence without forcing every test to re-add all
    built-ins.
    """
    yield
    presets._seed_builtin_presets()


class TestBuiltInPresets:
    def test_seeded_presets_are_present(self):
        names = {p.name for p in presets.list_presets()}
        assert {"review-and-fix", "docs-pair"} <= names

    def test_list_presets_is_sorted(self):
        names = [p.name for p in presets.list_presets()]
        assert names == sorted(names)

    def test_list_names_matches_list_presets(self):
        assert presets.list_names() == [p.name for p in presets.list_presets()]

    def test_seeded_presets_have_non_empty_members(self):
        for preset in presets.list_presets():
            assert preset.members, f"{preset.name} has no members"

    def test_seeded_member_prompts_are_non_empty(self):
        for preset in presets.list_presets():
            for member in preset.members:
                assert member.prompt, f"{preset.name}/{member.name} has empty prompt"


class TestRegister:
    def test_register_adds_new_preset(self):
        p = TeamPreset(
            name="custom",
            description="Custom",
            members=(PresetMemberSpec(name="solo", prompt="do work"),),
        )
        presets.register_preset(p)
        assert presets.get_preset("custom") is p

    def test_register_overwrites_existing(self):
        first = TeamPreset(
            name="dup",
            description="First",
            members=(PresetMemberSpec(name="x", prompt="old"),),
        )
        second = TeamPreset(
            name="dup",
            description="Second",
            members=(PresetMemberSpec(name="x", prompt="new"),),
        )
        presets.register_preset(first)
        presets.register_preset(second)
        got = presets.get_preset("dup")
        assert got.description == "Second"
        assert got.members[0].prompt == "new"

    def test_register_preserves_frozen_dataclass_semantics(self):
        # Regression guard: TeamPreset must remain @dataclass(frozen=True).
        # Inspecting __dataclass_params__ verifies the runtime contract
        # (frozen=True → FrozenInstanceError on mutation) without the test
        # itself having to attempt a mutation the type system forbids.
        assert TeamPreset.__dataclass_params__.frozen is True


class TestGetPreset:
    def test_returns_registered_preset(self):
        assert presets.get_preset("review-and-fix").name == "review-and-fix"

    def test_raises_key_error_for_unknown_name(self):
        with pytest.raises(KeyError, match="not-registered"):
            presets.get_preset("not-registered")


class TestUnregister:
    def test_removes_registered_preset(self):
        presets.register_preset(TeamPreset(name="temp", description="x"))
        presets.unregister_preset("temp")
        with pytest.raises(KeyError):
            presets.get_preset("temp")

    def test_silent_on_unknown_name(self):
        # Documented no-op contract.
        presets.unregister_preset("not-there")


class TestPresetMemberSpecDefaults:
    def test_optional_fields_default_to_none(self):
        m = PresetMemberSpec(name="x", prompt="do")
        assert m.template is None
        assert m.backend is None
        assert m.model is None
        assert m.subagent_type is None
        assert m.reasoning_effort is None
        assert m.agent_profile is None
        assert m.cwd is None
        assert m.plan_mode_required is None
        assert m.permission_mode is None

    def test_provided_values_survive_round_trip(self):
        m = PresetMemberSpec(
            name="x",
            prompt="do",
            template="code-reviewer",
            backend="claude-code",
            model="balanced",
            subagent_type="reviewer",
            reasoning_effort="high",
            agent_profile="senior",
            cwd="/abs/work",
            plan_mode_required=True,
            permission_mode="require_approval",
        )
        assert m.template == "code-reviewer"
        assert m.backend == "claude-code"
        assert m.model == "balanced"
        assert m.subagent_type == "reviewer"
        assert m.reasoning_effort == "high"
        assert m.agent_profile == "senior"
        assert m.cwd == "/abs/work"
        assert m.plan_mode_required is True
        assert m.permission_mode == "require_approval"


class TestForwardCompatMetadata:
    def test_skill_roots_default_empty_tuple(self):
        p = TeamPreset(name="x", description="y")
        assert p.skill_roots == ()

    def test_mcp_servers_default_empty_tuple(self):
        p = TeamPreset(name="x", description="y")
        assert p.mcp_servers == ()

    def test_skill_roots_survive_round_trip(self):
        # Forward-compat metadata must stay readable even though the
        # expansion path ignores it today — otherwise Feature G's
        # consumer code will silently see empty lists after a schema
        # refactor.
        p = TeamPreset(
            name="g-ready",
            description="x",
            skill_roots=("/abs/skills/team",),
        )
        presets.register_preset(p)
        got = presets.get_preset("g-ready")
        assert got.skill_roots == ("/abs/skills/team",)

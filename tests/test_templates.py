"""Unit tests for the agent template registry."""

import pytest

from claude_teams import templates
from claude_teams.templates import AgentTemplate


@pytest.fixture(autouse=True)
def _reset_registry():
    """Restore the seeded built-in templates after each test.

    Tests mutate the module-level registry; seeding on teardown keeps
    order-independence without forcing every test to re-add all
    built-ins.
    """
    yield
    templates._seed_builtin_templates()


class TestBuiltInTemplates:
    def test_seeded_templates_are_present(self):
        names = {t.name for t in templates.list_templates()}
        assert {
            "code-reviewer",
            "debugger",
            "executor",
            "test-engineer",
            "writer",
        } <= names

    def test_seeded_templates_all_carry_role_prompt(self):
        for template in templates.list_templates():
            assert template.role_prompt, f"{template.name} has empty role_prompt"

    def test_list_templates_is_sorted(self):
        names = [t.name for t in templates.list_templates()]
        assert names == sorted(names)

    def test_list_names_matches_list_templates(self):
        assert templates.list_names() == [t.name for t in templates.list_templates()]


class TestRegister:
    def test_register_adds_new_template(self):
        t = AgentTemplate(
            name="custom", description="Custom role", role_prompt="Do the thing"
        )
        templates.register_template(t)
        assert templates.get_template("custom") is t

    def test_register_overwrites_existing(self):
        first = AgentTemplate(
            name="dup", description="First version", role_prompt="old"
        )
        second = AgentTemplate(
            name="dup", description="Second version", role_prompt="new"
        )
        templates.register_template(first)
        templates.register_template(second)
        assert templates.get_template("dup").role_prompt == "new"

    def test_register_preserves_frozen_dataclass_semantics(self):
        # Regression guard: AgentTemplate must remain @dataclass(frozen=True).
        # Inspecting __dataclass_params__ verifies the runtime contract
        # (frozen=True → FrozenInstanceError on mutation) without the test
        # itself having to attempt a mutation the type system forbids.
        assert AgentTemplate.__dataclass_params__.frozen is True


class TestGetTemplate:
    def test_returns_registered_template(self):
        assert templates.get_template("code-reviewer").name == "code-reviewer"

    def test_raises_key_error_for_unknown_name(self):
        with pytest.raises(KeyError, match="not-registered"):
            templates.get_template("not-registered")


class TestUnregister:
    def test_removes_registered_template(self):
        templates.register_template(AgentTemplate(name="temp", description="x"))
        templates.unregister_template("temp")
        with pytest.raises(KeyError):
            templates.get_template("temp")

    def test_silent_on_unknown_name(self):
        # Should not raise — documented no-op contract.
        templates.unregister_template("not-there")


class TestDefaults:
    def test_defaults_are_none_when_omitted(self):
        t = AgentTemplate(name="bare", description="x")
        assert t.default_backend is None
        assert t.default_model is None
        assert t.default_subagent_type is None
        assert t.default_reasoning_effort is None
        assert t.default_agent_profile is None
        assert t.default_permission_mode is None
        assert t.default_plan_mode_required is None

    def test_defaults_capture_provided_values(self):
        t = AgentTemplate(
            name="configured",
            description="x",
            default_backend="codex",
            default_model="powerful",
            default_subagent_type="planner",
            default_reasoning_effort="high",
            default_permission_mode="require_approval",
            default_plan_mode_required=True,
        )
        assert t.default_backend == "codex"
        assert t.default_model == "powerful"
        assert t.default_subagent_type == "planner"
        assert t.default_reasoning_effort == "high"
        assert t.default_permission_mode == "require_approval"
        assert t.default_plan_mode_required is True


class TestForwardCompatMetadata:
    def test_skill_roots_default_empty_tuple(self):
        t = AgentTemplate(name="x", description="y")
        assert t.skill_roots == ()

    def test_mcp_servers_default_empty_tuple(self):
        t = AgentTemplate(name="x", description="y")
        assert t.mcp_servers == ()

    def test_skill_roots_survive_round_trip(self):
        # Forward-compat metadata must stay readable even though the spawn
        # path ignores it today — otherwise Feature G's consumer code will
        # silently see empty lists after a schema refactor.
        t = AgentTemplate(
            name="g-ready",
            description="x",
            skill_roots=("/abs/skills/reviewer",),
        )
        templates.register_template(t)
        got = templates.get_template("g-ready")
        assert got.skill_roots == ("/abs/skills/reviewer",)

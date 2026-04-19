"""Team preset registry for declarative team compositions.

A preset expands, through the validated ``team_create`` +
``spawn_teammate`` path, into a ready-to-work team. Each preset carries:

- a team-identity layer (``description``, ``team_description``) used when
  the lead session calls ``create_team_from_preset``,
- a tuple of ``PresetMemberSpec`` entries that fan out to one
  ``spawn_teammate`` call each,
- optional forward-compat metadata (``skill_roots``, ``mcp_servers``) for
  Feature G's per-team injection support.

Presets do not bypass auth: they run through the same lead-capability
gate as manual ``team_create`` and ``spawn_teammate`` calls. Their value
is composition, not privilege elevation — a preset is a shorthand for
a sequence a lead could execute by hand.

Relationship to templates
-------------------------
A ``PresetMemberSpec.template`` resolves to an ``AgentTemplate`` at
spawn time via the existing ``apply_template`` helper in
``orchestration``. The preset's
per-member fields follow the same precedence rule: explicit preset
fields override template defaults.
"""

from dataclasses import dataclass, field
from typing import Literal

PermissionModeOpt = Literal["default", "require_approval", "bypass"]


@dataclass(frozen=True)
class PresetMemberSpec:
    """Declarative teammate entry used inside a ``TeamPreset``.

    Mirrors the non-identity fields of ``SpawnOptions`` plus the
    ``template`` selector and the ``prompt`` body that the preset author
    ships alongside the role. Everything except ``name`` and ``prompt``
    is optional — defaults flow through the preset → template → backend
    resolution chain in that order.

    Attributes:
        name: Member name passed to ``spawn_teammate``.
        prompt: Initial task prompt for the member. Required because a
            preset with an empty prompt would still create a teammate
            that has nothing to do — surface the omission at preset
            definition time instead of at spawn time.
        template: Optional registered ``AgentTemplate`` name. When set,
            the template's role-prompt is prepended and its defaults
            fill in any field the preset left unset.
        backend: Explicit backend identifier; ``None`` accepts the
            registry default (or the template's default if one is set).
        model: Generic tier (``fast``/``balanced``/``powerful``) or a
            backend-specific model identifier.
        subagent_type: Subagent label recorded in the team config.
        reasoning_effort: Per-backend effort selector.
        agent_profile: Per-backend persona profile name.
        cwd: Absolute working directory override; ``None`` inherits the
            server's cwd.
        plan_mode_required: Whether the teammate must enter plan mode
            before executing work.
        permission_mode: Permission override; ``None`` inherits backend
            default.

    """

    name: str
    prompt: str
    template: str | None = None
    backend: str | None = None
    model: str | None = None
    subagent_type: str | None = None
    reasoning_effort: str | None = None
    agent_profile: str | None = None
    cwd: str | None = None
    plan_mode_required: bool | None = None
    permission_mode: PermissionModeOpt | None = None


@dataclass(frozen=True)
class TeamPreset:
    """Named declarative team composition.

    Attributes:
        name: Unique identifier used by callers to select the preset.
        description: Human-readable summary for ``list_presets``.
        team_description: Description forwarded to ``team_create`` when
            the preset expands. Distinct from ``description`` — the
            latter summarizes the preset itself while this one becomes
            part of the created team's persistent metadata.
        members: Ordered tuple of ``PresetMemberSpec`` entries. Order
            determines spawn order, which in turn determines color
            assignment (``_assign_color`` walks teammates in config
            order).
        skill_roots: Forward-compat metadata for Feature G; absolute
            filesystem paths injected as per-team skill roots once
            backend-layer support lands.
        mcp_servers: Forward-compat metadata for Feature G; MCP server
            config fragments to inject per-team.

    """

    name: str
    description: str
    team_description: str = ""
    members: tuple[PresetMemberSpec, ...] = ()
    skill_roots: tuple[str, ...] = ()
    mcp_servers: tuple[dict[str, object], ...] = field(default_factory=tuple)


# Module-level registry. Tests reset via their ``_reset_registry`` fixtures;
# production code only ever reads or registers.
_registry: dict[str, TeamPreset] = {}


def register_preset(preset: TeamPreset) -> None:
    """Register or overwrite a preset by name.

    Overwrite semantics match ``templates.register_template`` so
    plugin-provided presets can supersede built-ins without an explicit
    unregister step.

    Args:
        preset: Preset to register.

    """
    _registry[preset.name] = preset


def unregister_preset(name: str) -> None:
    """Remove a preset from the registry.

    Silent no-op when the name is not registered, matching the template
    registry's contract so tests do not need to guard each call.

    Args:
        name: Preset name to remove.

    """
    _registry.pop(name, None)


def get_preset(name: str) -> TeamPreset:
    """Return the preset registered under ``name``.

    Args:
        name: Preset identifier.

    Returns:
        The registered ``TeamPreset``.

    Raises:
        KeyError: If no preset is registered under the given name.
            Callers at the MCP boundary translate this into
            ``UnknownPresetToolError`` with the discoverable-names list.

    """
    if name not in _registry:
        raise KeyError(name)
    return _registry[name]


def list_presets() -> list[TeamPreset]:
    """Return all registered presets sorted by name.

    Stable ordering keeps ``list_presets`` MCP output deterministic for
    clients that hash or diff the response.
    """
    return [_registry[name] for name in sorted(_registry)]


def list_names() -> list[str]:
    """Return the sorted list of registered preset names."""
    return sorted(_registry)


def _seed_builtin_presets() -> None:
    """Populate the registry with shipped built-in presets.

    The seeded presets are deliberately minimal — a single
    review-and-fix duo and a write-and-edit pair — to demonstrate the
    composition pattern without prescribing a heavyweight workflow.
    Ship-time presets should feel like starter kits, not opinions.

    """
    _registry.clear()

    register_preset(
        TeamPreset(
            name="review-and-fix",
            description=(
                "Small duo that reviews a change and produces the fix. "
                "Pairs the ``code-reviewer`` template with a focused "
                "executor so a single prompt triggers both passes."
            ),
            team_description=(
                "Review-and-fix duo: reviewer surfaces findings, "
                "executor applies the agreed changes."
            ),
            members=(
                PresetMemberSpec(
                    name="reviewer",
                    prompt=(
                        "Review the current change. Cite file:line for "
                        "every finding and rate severity. Do not edit "
                        "code — hand findings to the executor."
                    ),
                    template="code-reviewer",
                ),
                PresetMemberSpec(
                    name="executor",
                    prompt=(
                        "Wait for review findings from the reviewer, "
                        "then apply fixes. Run lint and tests before "
                        "reporting completion."
                    ),
                    template="executor",
                ),
            ),
        )
    )

    register_preset(
        TeamPreset(
            name="docs-pair",
            description=(
                "Two-person documentation pair. Writer drafts, "
                "reviewer polishes prose and verifies examples."
            ),
            team_description=("Docs-pair duo: writer drafts, reviewer hardens."),
            members=(
                PresetMemberSpec(
                    name="writer",
                    prompt=(
                        "Draft the requested documentation. Keep prose "
                        "tight and match the project's existing voice."
                    ),
                    template="writer",
                ),
                PresetMemberSpec(
                    name="reviewer",
                    prompt=(
                        "Review the writer's draft for accuracy, "
                        "clarity, and coverage. Verify every example "
                        "against the code."
                    ),
                    template="code-reviewer",
                ),
            ),
        )
    )


_seed_builtin_presets()


__all__ = [
    "PermissionModeOpt",
    "PresetMemberSpec",
    "TeamPreset",
    "get_preset",
    "list_names",
    "list_presets",
    "register_preset",
    "unregister_preset",
]

"""Agent template registry for reusable role-context layers.

Templates are lightweight named bundles of:

- a role-level prompt header prepended to the spawn prompt,
- optional backend/model/subagent defaults that fill in only when the
  caller left the matching ``SpawnOptions`` field unset.

Templates are not personas or prompt-construction DSLs: backend-required
system wrappers remain authoritative, and task-specific prompts layered
above the template take precedence. See ``apply_template`` in
``orchestration`` for the resolution order.

Forward-compat note
-------------------
``skill_roots`` and ``mcp_servers`` are reserved metadata fields for
Feature G (per-team / per-agent skills and MCP injection). The spawn
path ignores them today so the declarative shape can stabilise before
the per-backend injection work lands. Adding them now means Feature G
can layer on without schema churn.
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

PermissionModeOpt = Literal["default", "require_approval", "bypass"]


class McpServerConfig(TypedDict, total=False):
    """Structural shape of an MCP server config fragment.

    Forward-compat type for Feature G's per-team / per-agent MCP
    injection. Defined with ``total=False`` because the MCP server
    config grammar evolves (stdio vs. sse vs. http transports each use
    different keys) and this type only has to be permissive enough for
    the Feature G consumer to narrow further at use time. Using a
    TypedDict instead of ``dict[str, object]`` keeps the "this is a
    config fragment, not arbitrary bag" intent visible in signatures.

    Attributes:
        command: Executable path for stdio-transport servers.
        args: Argv passed to ``command``.
        env: Environment variables overlaid on the server process.
        type: Transport discriminator (``"stdio"``/``"sse"``/``"http"``).
        url: Endpoint URL for sse/http transports.

    """

    command: str
    args: list[str]
    env: dict[str, str]
    type: str
    url: str


@dataclass(frozen=True)
class AgentTemplate:
    """Reusable role-context layer applied at ``spawn_teammate`` time.

    Attributes:
        name: Unique identifier used by callers to select the template.
        description: Human-readable summary surfaced by ``list_templates``.
        role_prompt: Text prepended to the spawn prompt with a blank
            line separator; empty string means "no role prefix."
        default_backend: Backend name to use when the caller did not
            explicitly set ``options.backend``.
        default_model: Model tier or backend-specific identifier, applied
            only when the caller left ``options.model`` unset.
        default_subagent_type: Subagent label recorded in the team config
            when the caller did not pick one.
        default_reasoning_effort: Effort preset for backends exposing a
            reasoning dial; ignored by backends that do not.
        default_agent_profile: Backend-specific persona profile name for
            backends that support selection.
        default_permission_mode: Permission override (default /
            require_approval / bypass). ``None`` inherits the backend
            default.
        default_plan_mode_required: Whether the teammate must enter plan
            mode before executing work.
        skill_roots: Forward-compat metadata for Feature G; absolute
            filesystem paths that will be injected as per-worker skill
            roots once backend-layer support lands.
        mcp_servers: Forward-compat metadata for Feature G; MCP server
            config fragments to inject per-worker.

    """

    name: str
    description: str
    role_prompt: str = ""
    default_backend: str | None = None
    default_model: str | None = None
    default_subagent_type: str | None = None
    default_reasoning_effort: str | None = None
    default_agent_profile: str | None = None
    default_permission_mode: PermissionModeOpt | None = None
    default_plan_mode_required: bool | None = None
    skill_roots: tuple[str, ...] = ()
    mcp_servers: tuple[McpServerConfig, ...] = ()


# Module-level registry. Tests reset via ``_reset_registry`` in their
# fixtures; production code only ever reads or registers.
_registry: dict[str, AgentTemplate] = {}


def register_template(template: AgentTemplate) -> None:
    """Register or overwrite a template by name.

    Args:
        template: Template to register. Overwrites any existing entry
            with the same name — matches the backend registry convention
            and lets plugin-provided templates supersede built-ins.

    """
    _registry[template.name] = template


def unregister_template(name: str) -> None:
    """Remove a template from the registry.

    No-ops silently if the name is not registered so callers (primarily
    tests) do not need to guard each call with a membership check.

    Args:
        name: Template name to remove.

    """
    _registry.pop(name, None)


def get_template(name: str) -> AgentTemplate:
    """Return the template registered under ``name``.

    Args:
        name: Template identifier.

    Returns:
        The registered ``AgentTemplate``.

    Raises:
        KeyError: If no template is registered under the given name.
            Callers at the MCP boundary translate this into
            ``UnknownTemplateToolError`` with the discoverable-names list.

    """
    if name not in _registry:
        raise KeyError(name)
    return _registry[name]


def list_templates() -> list[AgentTemplate]:
    """Return all registered templates sorted by name.

    Stable ordering keeps ``list_templates`` MCP output deterministic for
    clients that hash or diff the response.
    """
    return [_registry[name] for name in sorted(_registry)]


def list_names() -> list[str]:
    """Return the sorted list of registered template names."""
    return sorted(_registry)


def _seed_builtin_templates() -> None:
    """Populate the registry with the shipped built-in templates.

    The five seeded roles — ``code-reviewer``, ``debugger``, ``executor``,
    ``test-engineer``, ``writer`` — cover the most common teammate shapes
    a lead spawns during a work session. The role prompts are short
    headers, not full personas: they set posture without pre-empting the
    task-specific prompt that layers on top.

    """
    _registry.clear()

    register_template(
        AgentTemplate(
            name="code-reviewer",
            description=(
                "Reviews code for correctness, security, style, and "
                "maintainability. Produces severity-rated findings."
            ),
            role_prompt=(
                "You are acting as a code reviewer. Favour evidence over "
                "opinion: cite file:line for every finding. Rate each "
                "issue by severity (blocker / high / medium / low / nit). "
                "Do not rewrite code — recommend changes and let the "
                "author apply them."
            ),
            default_subagent_type="code-reviewer",
        )
    )

    register_template(
        AgentTemplate(
            name="debugger",
            description=(
                "Investigates failures via reproduction, stack-trace "
                "analysis, and evidence-driven hypothesis testing."
            ),
            role_prompt=(
                "You are acting as a debugger. Establish reproduction "
                "first, then form hypotheses with explicit evidence for "
                "and against each. State uncertainty plainly and propose "
                "the next probe when the root cause is not yet proven."
            ),
            default_subagent_type="debugger",
        )
    )

    register_template(
        AgentTemplate(
            name="executor",
            description=(
                "Implements focused tasks against a known plan. "
                "Verifies work with tests and lint before reporting done."
            ),
            role_prompt=(
                "You are acting as an executor. Stay within the scope "
                "you were handed. Run the project's lint and tests "
                "before declaring completion. Surface blockers rather "
                "than silently widening scope."
            ),
            default_subagent_type="executor",
        )
    )

    register_template(
        AgentTemplate(
            name="test-engineer",
            description=(
                "Writes and hardens tests: unit, integration, edge "
                "cases, flaky-test triage."
            ),
            role_prompt=(
                "You are acting as a test engineer. Prefer tests that "
                "fail for one clear reason. Cover the happy path, "
                "boundary cases, and realistic error modes. Mark "
                "intentionally skipped cases with a reason."
            ),
            default_subagent_type="test-engineer",
        )
    )

    register_template(
        AgentTemplate(
            name="writer",
            description=(
                "Produces and refines technical documentation: README, "
                "API docs, inline comments."
            ),
            role_prompt=(
                "You are acting as a technical writer. Keep prose tight "
                "and factual. Match the project's existing voice. "
                "Document behaviour that a future reader cannot derive "
                "from well-named identifiers."
            ),
            default_subagent_type="writer",
        )
    )


_seed_builtin_templates()


# Re-exported for type-narrowing convenience at the MCP layer where a
# validator wants to treat the registry as read-only.
__all__ = [
    "AgentTemplate",
    "McpServerConfig",
    "PermissionModeOpt",
    "get_template",
    "list_names",
    "list_templates",
    "register_template",
    "unregister_template",
]

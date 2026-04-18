"""Reusable validated parameter types for FastMCP tools and prompts.

Each alias wraps a primitive type with a ``pydantic.Field`` that carries:
- an LLM-visible ``description`` used in the generated MCP tool manifest,
- validation constraints (length, pattern, range) that fire before the
  tool body runs, replacing prior hand-written checks.

Centralizing these keeps parameter semantics consistent across the five
registration modules and makes FastMCP's generated JSON schema uniform
for clients that introspect the tool surface.
"""

from typing import Annotated, Literal

from pydantic import Field

# Shared regex — mirrors ``claude_teams.teams._VALID_NAME_RE`` so that
# Field-validated parameters and the underlying storage layer agree on
# what characters are permitted in an identifier.
_SAFE_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
_SAFE_NAME_MAX = 64


TeamName = Annotated[
    str,
    Field(
        description="Team identifier.",
        min_length=1,
        max_length=_SAFE_NAME_MAX,
        pattern=_SAFE_NAME_PATTERN,
    ),
]
"""Name of a team. Enforces the same safe-name grammar as the storage layer."""

AgentName = Annotated[
    str,
    Field(
        description="Agent identifier within a team.",
        min_length=1,
        max_length=_SAFE_NAME_MAX,
        pattern=_SAFE_NAME_PATTERN,
    ),
]
"""Name of a teammate agent. Same grammar as ``TeamName``."""

TaskId = Annotated[
    str,
    Field(
        description="Task identifier assigned by ``task_create``.",
        min_length=1,
        max_length=128,
    ),
]
"""Identifier of a task within a team."""

RequestId = Annotated[
    str,
    Field(
        description="Request identifier returned from a previous message.",
        max_length=128,
    ),
]
"""Identifier tying a response message to its originating request."""

Capability = Annotated[
    str,
    Field(
        description=(
            "Optional capability token for out-of-session authentication; "
            "leave empty when the MCP session is already attached."
        ),
        # ``min_length=0`` is made explicit so "" remains a valid "no token"
        # sentinel even if pydantic ever tightens its default handling of
        # constrained strings. The resolver treats an empty string as
        # absence, so the length gate must stay at zero.
        min_length=0,
        max_length=512,
    ),
]
"""Opaque capability string used to authenticate without an attached session."""

Description = Annotated[
    str,
    Field(
        description="Free-form human-readable description.",
        max_length=4096,
    ),
]
"""Description field for teams and tasks."""

Subject = Annotated[
    str,
    Field(
        description="Short one-line subject for a task.",
        min_length=1,
        max_length=512,
    ),
]
"""Subject line of a task."""

ActiveForm = Annotated[
    str,
    Field(
        description=(
            'Present-continuous form of the task subject (e.g. "writing tests").'
        ),
        max_length=512,
    ),
]
"""Present-continuous phrasing used in status displays."""

Prompt = Annotated[
    str,
    Field(
        description="Initial prompt text delivered to a newly spawned teammate.",
        min_length=1,
        max_length=32768,
    ),
]
"""Spawn-time prompt body."""

MessageContent = Annotated[
    str,
    Field(
        description="Body text of a team message.",
        max_length=32768,
    ),
]
"""Body of a message."""

MessageSummary = Annotated[
    str,
    Field(
        description="One-line summary shown in inbox listings.",
        max_length=512,
    ),
]
"""Summary shown in terse inbox views."""

Hint = Annotated[
    str,
    Field(
        description="Optional guidance from the team lead.",
        max_length=2048,
    ),
]
"""Free-form lead-provided hint attached to a prompt."""

Cwd = Annotated[
    str,
    Field(
        description=(
            "Absolute working directory for a spawned teammate; "
            "empty string means inherit the server's cwd."
        ),
        max_length=4096,
    ),
]
"""Working directory for a spawned backend process."""

ModelName = Annotated[
    str,
    Field(
        description=(
            "Generic tier (``fast``/``balanced``/``powerful``) or a "
            "backend-specific model identifier. Call ``list_backends`` to "
            "discover each backend's ``default_model`` and ``supported_models`` "
            "before choosing a non-generic value."
        ),
        min_length=1,
        max_length=128,
    ),
]
"""Model selector accepted by the backend registry."""

BackendName = Annotated[
    str,
    Field(
        description=(
            "Explicit backend name; empty string selects the default backend. "
            "Call ``list_backends`` to enumerate installed backends with their "
            "availability and supported models."
        ),
        max_length=64,
    ),
]
"""Spawner backend identifier."""

SubagentType = Annotated[
    str,
    Field(
        description=(
            "Claude Code subagent type recorded in the team config. Common "
            "values include specialized agent types from installed plugins."
        ),
        min_length=1,
        max_length=128,
        examples=["general-purpose", "executor", "code-reviewer", "debugger"],
    ),
]
"""Subagent type string written to ``config.json``."""

PermissionModeOpt = Literal["default", "require_approval", "bypass"]
"""Permission behavior override accepted by ``spawn_teammate``."""

Limit = Annotated[
    int,
    Field(
        description="Page size for paginated listings.",
        ge=1,
        le=500,
    ),
]
"""Bounded page size."""

Offset = Annotated[
    int,
    Field(
        description="Zero-based offset into a paginated listing.",
        ge=0,
    ),
]
"""Non-negative page offset."""

TimeoutMs = Annotated[
    int,
    Field(
        description="Poll timeout in milliseconds (maximum five minutes).",
        ge=0,
        le=300_000,
    ),
]
"""Inbox-poll timeout, bounded to avoid wedging a session."""

OutputLines = Annotated[
    int,
    Field(
        description="Lines of recent terminal output to capture.",
        ge=1,
        le=120,
    ),
]
"""Window size for terminal captures."""

MaxMessages = Annotated[
    int,
    Field(
        description="Maximum inbox messages to return from ``check_teammate``.",
        ge=1,
        le=20,
    ),
]
"""Upper bound on messages returned from a check."""

SenderName = Annotated[
    str,
    Field(
        description=("Logical sender identity; use ``team-lead`` when acting as lead."),
        min_length=1,
        max_length=_SAFE_NAME_MAX,
    ),
]
"""Sender identity for send_message."""

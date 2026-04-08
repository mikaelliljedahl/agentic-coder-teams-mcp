"""Shared Pydantic models for team orchestration state."""

from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator
from pydantic.alias_generators import to_camel


def _to_camel(name: str) -> str:
    """Convert snake_case to camelCase, stripping trailing underscores for reserved words.

    Args:
        name (str): Field name in snake_case (may have trailing underscore).

    Returns:
        str: Field name converted to camelCase.

    """
    return to_camel(name.rstrip("_"))


COLOR_PALETTE: list[str] = [
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "cyan",
    "red",
]

TaskMetadataValue: TypeAlias = str | int | float | bool | None
TaskMetadata: TypeAlias = dict[str, TaskMetadataValue]
MessageRoutingValue: TypeAlias = str | None
MessageRouting: TypeAlias = dict[str, MessageRoutingValue]


class LeadMember(BaseModel):
    """Serialized team-lead member record."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    agent_id: str
    name: str
    agent_type: str
    model: str
    joined_at: int
    tmux_pane_id: str = ""
    cwd: str
    subscriptions: list[str] = Field(default_factory=list)


class TeammateMember(BaseModel):
    """Serialized teammate member record."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    agent_id: str
    name: str
    agent_type: str
    model: str
    prompt: str
    color: str
    plan_mode_required: bool = False
    joined_at: int
    tmux_pane_id: str
    cwd: str
    subscriptions: list[str] = Field(default_factory=list)
    backend_type: str = "claude-code"
    is_active: bool = False
    process_handle: str = ""

    @model_validator(mode="before")
    @classmethod
    def _sync_process_handle(cls, data: object) -> object:
        """Copy tmuxPaneId to processHandle during deserialization if absent.

        Args:
            data (object): Raw data dict or object being validated.

        Returns:
            object: Data with synchronized tmuxPaneId and processHandle fields.

        """
        if isinstance(data, dict):
            raw_dict = cast(dict[str, object], data)
            pane = raw_dict.get("tmuxPaneId") or raw_dict.get("tmux_pane_id") or ""
            handle = (
                raw_dict.get("processHandle") or raw_dict.get("process_handle") or ""
            )
            if pane and not handle:
                raw_dict["processHandle"] = pane
            elif handle and not pane:
                raw_dict["tmuxPaneId"] = handle
            return raw_dict
        return data


def _discriminate_member(v: object) -> str:
    if isinstance(v, dict):
        return "teammate" if "prompt" in v else "lead"
    if isinstance(v, TeammateMember):
        return "teammate"
    return "lead"


MemberUnion = Annotated[
    Annotated[LeadMember, Tag("lead")] | Annotated[TeammateMember, Tag("teammate")],
    Discriminator(_discriminate_member),
]


class TeamConfig(BaseModel):
    """Persistent team configuration."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    name: str
    description: str = ""
    created_at: int
    lead_agent_id: str
    lead_session_id: str
    members: list[MemberUnion]


class TaskFile(BaseModel):
    """Persistent task record."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    subject: str
    description: str
    active_form: str = ""
    status: Literal["pending", "in_progress", "completed", "deleted"] = "pending"
    blocks: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    owner: str | None = None
    metadata: TaskMetadata | None = None


class InboxMessage(BaseModel):
    """Persistent inbox message record."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    from_: str
    text: str
    timestamp: str
    read: bool = False
    summary: str | None = None
    color: str | None = None


class PaginatedTaskList(BaseModel):
    """Paginated task-list response payload."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    items: list[TaskFile]
    total_count: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None = None


class PaginatedInboxMessages(BaseModel):
    """Paginated inbox response payload."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    items: list[InboxMessage]
    total_count: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None = None


class IdleNotification(BaseModel):
    """Structured idle notification message."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    type: Literal["idle_notification"] = "idle_notification"
    from_: str
    timestamp: str
    idle_reason: str = "available"


class TaskAssignment(BaseModel):
    """Structured task-assignment message."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    type: Literal["task_assignment"] = "task_assignment"
    task_id: str
    subject: str
    description: str
    assigned_by: str
    timestamp: str


class ShutdownRequest(BaseModel):
    """Structured shutdown-request message."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    type: Literal["shutdown_request"] = "shutdown_request"
    request_id: str
    from_: str
    reason: str
    timestamp: str


class ShutdownApproved(BaseModel):
    """Structured graceful-shutdown approval message."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    type: Literal["shutdown_approved"] = "shutdown_approved"
    request_id: str
    from_: str
    timestamp: str
    pane_id: str
    backend_type: str
    process_handle: str = ""


class TeamCreateResult(BaseModel):
    """Result payload returned after team creation."""

    team_name: str
    team_file_path: str
    lead_agent_id: str
    lead_capability: str | None = None


class TeamAttachResult(BaseModel):
    """Result payload returned after team attachment."""

    team_name: str
    principal_name: str
    principal_role: Literal["lead", "agent"]


class TeamDeleteResult(BaseModel):
    """Result payload returned after team deletion."""

    success: bool
    message: str
    team_name: str


class SpawnResult(BaseModel):
    """Result payload returned after teammate spawn."""

    agent_id: str
    name: str
    team_name: str
    message: str = "The agent is now running and will receive instructions via mailbox."


class BackendInfo(BaseModel):
    """Backend availability and model metadata."""

    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    name: str
    binary: str
    available: bool
    default_model: str
    supported_models: list[str]


class SendMessageResult(BaseModel):
    """Result payload returned after a messaging action."""

    success: bool
    message: str
    routing: MessageRouting | None = None
    request_id: str | None = None
    target: str | None = None

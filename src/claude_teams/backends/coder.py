"""Coder backend integration."""

from typing import ClassVar

from claude_teams.backends._agent_discovery import discover_codex_style_agents
from claude_teams.backends.base import (
    AgentProfile,
    AgentSelectSpec,
    BaseBackend,
    ReasoningEffortSpec,
    SpawnRequest,
)


class CoderBackend(BaseBackend):
    """Backend adapter for Just Every Code CLI (Codex fork)."""

    _name = "coder"
    _binary_name = "coder"

    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "claude-haiku-4.5",
        "balanced": "claude-sonnet-4.5",
        "powerful": "claude-opus-4.6",
    }

    _REASONING_EFFORT_SPEC: ClassVar[ReasoningEffortSpec] = ReasoningEffortSpec(
        flag="-c",
        value_template="model_reasoning_effort={value}",
        options=frozenset({"low", "medium", "high", "xhigh"}),
    )

    _AGENT_SELECT_SPEC: ClassVar[AgentSelectSpec] = AgentSelectSpec(
        flag="-c",
        value_template='agents.{name}.config_file="{path}"',
    )

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Coder inherits Codex's ``-c`` config override for reasoning effort."""
        return self._REASONING_EFFORT_SPEC

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Coder selects an agent via the same ``-c`` override as Codex."""
        return self._AGENT_SELECT_SPEC

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Parse ``[agents.*]`` tables from ``~/.coder/config.toml`` and project."""
        return discover_codex_style_agents(cwd, "coder")

    def supported_models(self) -> list[str]:
        """Return supported Coder model names.

        Coder supports any model via ``-m``; these are the common ones.

        Returns:
            list[str]: Curated list of supported model identifiers.

        """
        return [
            "claude-haiku-4.5",
            "claude-sonnet-4.5",
            "claude-opus-4.6",
            "gpt-5.2-codex",
            "gpt-5.2",
            "o3",
        ]

    def default_model(self) -> str:
        """Return the default Coder model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "claude-sonnet-4.5"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Coder model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Coder model identifier.

        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def default_permission_args(self) -> list[str]:
        """Return default permission-bypass arguments for Coder."""
        return ["--full-auto"]

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Coder CLI command for non-interactive execution.

        Uses the ``exec`` subcommand with ``--full-auto`` for low-friction
        sandboxed automatic execution.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.

        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "exec",
            "-m",
            model,
            *self.permission_args(request),
        ]
        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))
        cmd.extend(self._agent_args(request))
        cmd.append(request.prompt)
        return cmd

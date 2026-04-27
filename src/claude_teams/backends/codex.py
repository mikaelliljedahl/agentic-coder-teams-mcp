"""Codex backend integration."""

from typing import ClassVar

from claude_teams.backends._agent_discovery import discover_codex_style_agents
from claude_teams.backends.base import (
    AgentProfile,
    AgentSelectSpec,
    BaseBackend,
    ReasoningEffortSpec,
    SpawnRequest,
)


class CodexBackend(BaseBackend):
    """Backend adapter for OpenAI Codex CLI."""

    _name = "codex"
    _binary_name = "codex"

    @property
    def is_interactive(self) -> bool:
        """Codex runs interactively so its configured MCP servers are started.

        Returns:
            bool: Always True.

        """
        return True

    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "gpt-5.1-codex-mini",
        "balanced": "gpt-5.3-codex",
        "powerful": "gpt-5.1-codex-max",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
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
        """Codex sets reasoning effort via a ``-c`` config override."""
        return self._REASONING_EFFORT_SPEC

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Codex selects an agent via a ``-c 'agents.<name>.config_file'`` override."""
        return self._AGENT_SELECT_SPEC

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Parse ``[agents.*]`` tables from ``~/.codex/config.toml`` and project."""
        return discover_codex_style_agents(cwd, "codex")

    def supported_models(self) -> list[str]:
        """Return supported Codex model names.

        Returns:
            list[str]: Curated list of supported model identifiers.

        """
        return ["gpt-5.3-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini"]

    def default_model(self) -> str:
        """Return the default Codex model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "gpt-5.3-codex"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Codex model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Codex model identifier.

        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def default_permission_args(self) -> list[str]:
        """Return default permission-bypass arguments for Codex."""
        return ["--full-auto"]

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Codex CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.

        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "--model",
            model,
            *self.permission_args(request),
            "-C",
            request.cwd,
        ]

        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))

        cmd.extend(self._agent_args(request))

        cmd.append(request.prompt)
        return cmd

"""Claude Code backend integration."""

from typing import ClassVar

from claude_teams.backends.base import BaseBackend, SpawnRequest
from claude_teams.errors import UnsupportedBackendModelError


class ClaudeCodeBackend(BaseBackend):
    """Backend adapter for Claude Code CLI."""

    _name = "claude-code"
    _binary_name = "claude"

    @property
    def is_interactive(self) -> bool:
        """Claude Code runs as an interactive MCP client with native team messaging.

        Returns:
            bool: Always True.

        """
        return True

    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "haiku",
        "balanced": "sonnet",
        "powerful": "opus",
        "haiku": "haiku",
        "sonnet": "sonnet",
        "opus": "opus",
    }

    def supported_models(self) -> list[str]:
        """Return supported Claude Code model short-names.

        Returns:
            list[str]: Curated list of supported model identifiers.

        """
        return ["haiku", "sonnet", "opus"]

    def default_model(self) -> str:
        """Return the default Claude Code model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "sonnet"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Claude Code model.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Claude Code model identifier.

        Raises:
            UnsupportedBackendModelError: For unsupported model names.

        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        raise UnsupportedBackendModelError(
            generic_name, "claude-code", self.supported_models()
        )

    def bypass_permission_args(self) -> list[str]:
        """Use Claude Code's explicit bypass permission mode."""
        return ["--permission-mode", "bypassPermissions"]

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Claude Code CLI command.

        Produces the canonical Claude Code worker launch command for this project.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.

        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "--agent-id",
            request.agent_id,
            "--agent-name",
            request.name,
            "--team-name",
            request.team_name,
            "--agent-color",
            request.color,
            "--parent-session-id",
            request.lead_session_id,
            "--agent-type",
            request.agent_type,
            "--model",
            model,
            *self.permission_args(request),
        ]
        if request.plan_mode_required:
            cmd.append("--plan-mode-required")
        return cmd

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Claude Code environment variables.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Dict with CLAUDECODE and CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS.

        """
        env = {
            "CLAUDECODE": "1",
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        }
        agent_capability = (request.extra or {}).get("agent_capability")
        if agent_capability:
            env["CLAUDE_TEAMS_AGENT_CAPABILITY"] = agent_capability
        return env

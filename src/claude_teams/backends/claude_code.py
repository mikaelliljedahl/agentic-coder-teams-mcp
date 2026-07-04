"""Claude Code backend integration."""

import json
import os
import shutil
from pathlib import Path
from typing import ClassVar

from claude_teams.backends._agent_discovery import discover_claude_agents
from claude_teams.backends.base import (
    AgentProfile,
    AgentSelectSpec,
    BaseBackend,
    ReasoningEffortSpec,
    SpawnRequest,
)
from claude_teams.backends.contracts import (
    BackendBinaryNotFoundError,
    UnsupportedBackendModelError,
)

_LOCAL_PATH_CLS = type(Path.cwd())


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

    _REASONING_EFFORT_SPEC: ClassVar[ReasoningEffortSpec] = ReasoningEffortSpec(
        flag="--effort",
        value_template="{value}",
        options=frozenset({"low", "medium", "high", "xhigh", "max"}),
    )

    _AGENT_SELECT_SPEC: ClassVar[AgentSelectSpec] = AgentSelectSpec(
        flag="--agent",
        value_template="{name}",
    )

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Claude Code exposes reasoning effort via a dedicated ``--effort`` flag."""
        return self._REASONING_EFFORT_SPEC

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Claude Code accepts a profile by name via ``--agent <name>``."""
        return self._AGENT_SELECT_SPEC

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Scan ``.claude/agents/*.md`` under the project and user-home roots."""
        return discover_claude_agents(cwd)

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

    def supports_resume(self) -> bool:
        """Claude Code supports native session resume."""
        return True

    def discover_binary(self) -> str:
        """Resolve ``claude`` to the native Windows binary when available."""
        shim = shutil.which(self._binary_name)
        if shim is None:
            raise BackendBinaryNotFoundError(self._binary_name, self._name)
        native = self._resolve_native_claude(shim)
        return native or shim

    @staticmethod
    def _resolve_native_claude(shim_path: str) -> str | None:
        """Locate npm's bundled ``claude.exe`` behind the shim."""
        if os.name != "nt":
            return None
        shim_dir = _LOCAL_PATH_CLS(shim_path).parent
        exe = (
            shim_dir
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        return str(exe) if exe.exists() else None

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
        mcp_config_path = (request.extra or {}).get("mcp_config_path")
        if mcp_config_path:
            cmd.extend(["--mcp-config", mcp_config_path])
        if request.plan_mode_required:
            cmd.append("--plan-mode-required")
        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))
        cmd.extend(self._agent_args(request))
        cmd.extend(self._hooks_settings_args(request))
        cmd.extend(self._disallowed_tools_args())
        cmd.append("--")
        cmd.append(self._prompt_arg(request))
        return cmd

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        """Build the Claude Code CLI command for a native session resume."""
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "--resume",
            backend_session_id,
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
        mcp_config_path = (request.extra or {}).get("mcp_config_path")
        if mcp_config_path:
            cmd.extend(["--mcp-config", mcp_config_path])
        if request.plan_mode_required:
            cmd.append("--plan-mode-required")
        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))
        cmd.extend(self._agent_args(request))
        cmd.extend(self._hooks_settings_args(request))
        cmd.extend(self._disallowed_tools_args())
        cmd.append("--")
        cmd.append(self._prompt_arg(request))
        return cmd

    @staticmethod
    def _disallowed_tools_args() -> list[str]:
        """Return ``--disallowed-tools`` args for a spawned team worker.

        Spawned agents run in Claude Code's native agent-teams mode (see
        :meth:`build_env`), where ``AskUserQuestion`` is routed as a permission
        request to the *team leader* session. In this system that "leader" is
        the MCP orchestrator that spawned the agent; it runs no native approval
        queue, so an ``AskUserQuestion`` call hangs forever on "Waiting for team
        lead approval". These workers are autonomous with no interactive user,
        so the tool is disabled outright -- a decision that needs the lead is
        escalated via ``send_message`` to ``lead`` instead.
        """
        return ["--disallowed-tools", "AskUserQuestion"]

    @staticmethod
    def _hooks_settings_args(request: SpawnRequest) -> list[str]:
        """Return ``--settings <path>`` args when state hooks are enabled.

        Guarded by env ``WIN_AGENT_TEAMS_STATE_HOOKS`` (default on). The path
        itself comes from ``request.extra["hooks_settings_path"]``, populated
        by the server before spawn/resume via
        :func:`claude_teams.hooks.write_claude_settings`.
        """
        if os.environ.get("WIN_AGENT_TEAMS_STATE_HOOKS", "1").strip() == "0":
            return []
        settings_path = (request.extra or {}).get("hooks_settings_path")
        if not settings_path:
            return []
        return ["--settings", settings_path]

    def _prompt_arg(self, request: SpawnRequest) -> str:
        """Return the initial Claude prompt argument.

        Multi-line tasks are carried as a single JSON string argument to avoid
        CLI/TUI line-boundary handling differences on Windows.
        """
        if "\n" not in request.prompt and "\r" not in request.prompt:
            return request.prompt
        return (
            "Decode this JSON string as your complete task prompt, then follow "
            f"the decoded text exactly: {json.dumps(request.prompt)}"
        )

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

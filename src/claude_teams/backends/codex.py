"""Codex backend integration."""

import json
from typing import ClassVar

from claude_teams.agent_output import codex_correlation_token
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
    _MCP_SERVER_NAME = "win-agent-teams"

    @property
    def is_interactive(self) -> bool:
        """Codex runs interactively so its configured MCP servers are started.

        Returns:
            bool: Always True.

        """
        return True

    # Verified against `codex debug models` (Codex CLI 0.130.0). Only
    # API-usable, listed slugs are mapped; `gpt-5.3-codex-spark`
    # (supported_in_api=false) and hidden `codex-auto-review` are excluded.
    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "gpt-5.4-mini",
        "balanced": "gpt-5.4",
        "powerful": "gpt-5.5",
        "gpt-5.5": "gpt-5.5",
        "gpt-5.4": "gpt-5.4",
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.2": "gpt-5.2",
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
        return ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2"]

    def default_model(self) -> str:
        """Return the default Codex model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "gpt-5.5"

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
        return ["--dangerously-bypass-approvals-and-sandbox"]

    def supports_resume(self) -> bool:
        """Codex supports native session resume."""
        return True

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Codex CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.

        """
        binary = self.discover_binary()
        cmd = [
            binary,
            *self.permission_args(request),
            "-C",
            request.cwd,
            *self._mcp_identity_args(request),
        ]

        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))

        cmd.extend(self._agent_args(request))

        cmd.append(self._prompt_arg(request, self._correlated_prompt(request)))
        return cmd

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        """Build the Codex CLI command for a native session resume."""
        binary = self.discover_binary()
        cmd = [
            binary,
            *self.permission_args(request),
            "-C",
            request.cwd,
            *self._mcp_identity_args(request),
        ]

        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))

        cmd.extend(self._agent_args(request))
        cmd.extend(["resume", backend_session_id, self._prompt_arg(request)])
        return cmd

    def _mcp_identity_args(self, request: SpawnRequest) -> list[str]:
        """Build a per-spawn ``-c`` override carrying this agent's identity.

        Codex does not propagate process env to MCP servers; it reads their
        ``env`` from config. Writing identity into the shared, machine-global
        ``~/.codex/config.toml`` is racy: a concurrent spawn from another
        session can overwrite it before Codex reads the file at startup,
        permanently mis-binding this agent's MCP server to the wrong session.

        Passing the env as a per-process ``-c`` override (highest config
        precedence) keeps identity bound to this exact Codex process via its
        own argv, with no shared mutable file and no race window.
        """
        env = {
            "CLAUDE_TEAMS_PERMISSION_MODE": "bypass",
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
        }
        pairs = ", ".join(
            f"{key} = {self._toml_literal(value)}" for key, value in env.items()
        )
        return ["-c", f"mcp_servers.{self._MCP_SERVER_NAME}.env={{ {pairs} }}"]

    @staticmethod
    def _toml_literal(value: str) -> str:
        """Render ``value`` as a TOML single-quoted literal string.

        Single-quoted literals avoid Windows ``CreateProcess`` double-quote
        escaping issues for the single argv token. TOML literal strings cannot
        contain a single quote (no escaping), so reject one rather than emit a
        corrupt override. ``AGENT_NAME`` is validated against ``[A-Za-z0-9_-]+``
        and ``AGENT_SESSION_ID`` is a uuid, so this is a defensive guard only.
        """
        if "'" in value or "\n" in value or "\r" in value:
            msg = f"value not representable as a TOML literal: {value!r}"
            raise ValueError(msg)
        return f"'{value}'"

    def _prompt_arg(self, request: SpawnRequest, prompt: str | None = None) -> str:
        """Return the Codex prompt argument for ``prompt`` (default: request).

        Codex's interactive prompt handling can truncate multi-line argv prompts
        in Windows consoles, so multi-line tasks are carried as a single JSON
        string argument and decoded by the agent from the initial instruction.
        """
        text = request.prompt if prompt is None else prompt
        if "\n" not in text and "\r" not in text:
            return text
        return (
            "Decode this JSON string as your complete task prompt, then follow "
            f"the decoded text exactly: {json.dumps(text)}"
        )

    def _correlated_prompt(self, request: SpawnRequest) -> str:
        """Append a per-agent correlation marker to the initial prompt.

        The marker lets ``read_codex_output`` bind this agent's rollout file
        deterministically when two agents are spawned in the same ``cwd`` at
        nearly the same time (before Codex's own session id is known). Only
        used for the initial spawn; resume already has the backend session id.
        """
        token = codex_correlation_token(request.agent_id)
        return (
            f"{request.prompt}\n\n"
            f"[win-agent-teams correlation id: {token} "
            "— internal marker, ignore this line]"
        )

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Pass agent identity so MCP servers inherit session context."""
        return {
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
        }

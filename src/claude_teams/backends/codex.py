"""Codex backend integration."""

import os
import platform
import shutil
from pathlib import Path
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
from claude_teams.backends.contracts import BackendBinaryNotFoundError

# Windows arch (``platform.machine()``) -> Codex npm platform-package suffix
# and Rust target triple, mirroring the dispatch table in the npm wrapper's
# ``bin/codex.js``.
_WINDOWS_NATIVE_TARGET: dict[str, tuple[str, str]] = {
    "AMD64": ("x64", "x86_64-pc-windows-msvc"),
    "ARM64": ("arm64", "aarch64-pc-windows-msvc"),
}


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

    def discover_binary(self) -> str:
        """Resolve ``codex`` to the native binary, bypassing the npm shim.

        On Windows ``shutil.which("codex")`` resolves to ``codex.cmd`` (an npm
        batch shim). Launching a ``.cmd`` runs it through ``cmd.exe``, which
        applies metacharacter parsing (``< > | & ^ ! ( )``) to the prompt argv
        token and kills the process before Codex starts. Resolving straight to
        the bundled native ``codex.exe`` removes both the ``cmd.exe`` and the
        ``node`` layers, so the argv passes verbatim through ``CreateProcess``
        to the Rust binary's standard ``CommandLineToArgvW`` parsing.

        Falls back to the shim path when the native binary cannot be located
        (non-npm install, unknown arch, or non-Windows), preserving prior
        behavior.
        """
        shim = shutil.which(self._binary_name)
        if shim is None:
            raise BackendBinaryNotFoundError(self._binary_name, self._name)
        native = self._resolve_native_codex(shim)
        return native[0] if native else shim

    @staticmethod
    def _resolve_native_codex(shim_path: str) -> tuple[str, str] | None:
        """Locate the native ``codex.exe`` and its arch vendor root.

        Mirrors the resolution in the npm wrapper's ``bin/codex.js``: the
        platform package ``@openai/codex-win32-<arch>`` (hoisted or nested) or
        the package's local ``vendor`` fallback, under
        ``vendor/<triple>/codex/codex.exe``.

        Returns ``(exe_path, arch_root)`` or ``None`` when not resolvable.
        """
        if os.name != "nt":
            return None
        target = _WINDOWS_NATIVE_TARGET.get(platform.machine().upper())
        if target is None:
            return None
        arch_suffix, triple = target
        shim_dir = Path(shim_path).parent
        codex_pkg = shim_dir / "node_modules" / "@openai" / "codex"
        platform_pkg = f"codex-win32-{arch_suffix}"
        rel = Path("vendor") / triple / "codex" / "codex.exe"
        bases = [
            codex_pkg / "node_modules" / "@openai" / platform_pkg,
            shim_dir / "node_modules" / "@openai" / platform_pkg,
            codex_pkg,
        ]
        for base in bases:
            exe = base / rel
            if exe.is_file():
                return str(exe), str(base / "vendor" / triple)
        return None

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

        Passed verbatim as a single argv token. Because the native binary is
        launched directly (see :meth:`discover_binary`), ``CreateProcess`` and
        the binary's ``CommandLineToArgvW`` parsing round-trip arbitrary text
        — metacharacters and newlines included — so no escaping or wrapping is
        needed.
        """
        return request.prompt if prompt is None else prompt

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
        """Pass agent identity and replicate the npm shim's runtime env.

        When the native binary is launched directly the npm ``bin/codex.js``
        wrapper is bypassed, so its two runtime effects are reproduced here:
        prepending the vendored ``path`` dir (bundled ``rg.exe``) to ``PATH``
        and marking the install as npm-managed.
        """
        env = {
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
        }
        shim = shutil.which(self._binary_name)
        native = self._resolve_native_codex(shim) if shim else None
        if native:
            _, arch_root = native
            env["CODEX_MANAGED_BY_NPM"] = "1"
            path_dir = Path(arch_root) / "path"
            if path_dir.is_dir():
                current = os.environ.get("PATH", "")
                env["PATH"] = f"{path_dir}{os.pathsep}{current}"
        return env

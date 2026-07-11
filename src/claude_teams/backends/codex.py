"""Codex backend integration."""

import json
import os
import platform
import shutil
import subprocess
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
from claude_teams.backends.contracts import (
    BackendBinaryNotFoundError,
    BackendModelUnavailableError,
)
from claude_teams.backends.process_manager import process_manager

_LOCAL_PATH_CLS = type(Path.cwd())

# Windows arch (``platform.machine()``) -> Codex npm platform-package suffix
# and Rust target triple, mirroring the dispatch table in the npm wrapper's
# ``bin/codex.js``.
_WINDOWS_NATIVE_TARGET: dict[str, tuple[str, str]] = {
    "AMD64": ("x64", "x86_64-pc-windows-msvc"),
    "ARM64": ("arm64", "aarch64-pc-windows-msvc"),
}

# Seconds to wait for ``codex debug models`` before giving up on live model
# discovery and using the static fallback list.
_MODEL_DISCOVERY_TIMEOUT_S = 20.0

# Process-lifetime cache of discovered model slugs, keyed by the resolved codex
# binary path. Populated lazily on first ``supported_models``/tier resolution so
# a spawn pays at most one ``codex debug models`` subprocess per process.
_MODEL_SLUG_CACHE: dict[str, list[str]] = {}


def _discover_codex_model_slugs(binary: str) -> list[str]:
    """Return the API-usable, list-visible model slugs from ``codex debug models``.

    Runs the codex CLI's own model registry dump and keeps only models that are
    both ``supported_in_api`` and ``visibility == "list"`` (hidden helpers like
    ``codex-auto-review`` are dropped). Cached per binary for the process
    lifetime. Any failure (binary missing, timeout, non-JSON, schema drift)
    yields ``[]`` so callers fall back to the static list rather than crash.
    """
    if binary in _MODEL_SLUG_CACHE:
        return _MODEL_SLUG_CACHE[binary]
    slugs: list[str] = []
    try:
        proc = subprocess.run(  # noqa: S603 - binary resolved from PATH, fixed argv.
            [binary, "debug", "models"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_MODEL_DISCOVERY_TIMEOUT_S,
            check=False,
        )
        data = json.loads(proc.stdout)
        for model in data.get("models", []):
            if not isinstance(model, dict):
                continue
            slug = model.get("slug")
            if (
                model.get("supported_in_api")
                and model.get("visibility") == "list"
                and isinstance(slug, str)
                and slug
            ):
                slugs.append(slug)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        slugs = []
    _MODEL_SLUG_CACHE[binary] = slugs
    return slugs


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

    # The ONLY model interface exposed to the MCP caller: five capability
    # tiers, each bundling a concrete GPT-5.6 model with a reasoning effort,
    # as an ascending cost/quality ladder. Names mirror the effort words the
    # caller already reasons in (low..ultra) rather than raw model slugs, which
    # the caller has no basis to choose between. All targets are GPT-5.6; if the
    # target model is not available on this codex install the spawn errors (no
    # silent downgrade). A tier fully determines model + effort; a caller-
    # supplied ``reasoning_effort`` is silently ignored for tiers.
    #   low    -> Terra @ medium   (dirt cheap, quick/low-stakes)
    #   medium -> Sol   @ low       (token-efficient general default)
    #   high   -> Sol   @ medium   (backend dev, code review)
    #   xhigh  -> Sol   @ high     (genuinely hard problems)
    #   ultra  -> Sol   @ xhigh    (top; higher still is diminishing returns)
    _TIER_LAUNCH: ClassVar[dict[str, tuple[str, str]]] = {
        "low": ("gpt-5.6-terra", "medium"),
        "medium": ("gpt-5.6-sol", "low"),
        "high": ("gpt-5.6-sol", "medium"),
        "xhigh": ("gpt-5.6-sol", "high"),
        "ultra": ("gpt-5.6-sol", "xhigh"),
    }

    _REASONING_EFFORT_SPEC: ClassVar[ReasoningEffortSpec] = ReasoningEffortSpec(
        flag="-c",
        value_template="model_reasoning_effort={value}",
        # Union across models; Terra adds ``max``/``ultra`` and Luna adds
        # ``max`` over the older 4-level set. Codex validates the actual
        # model+effort combination at launch, so this stays a superset.
        options=frozenset({"low", "medium", "high", "xhigh", "max", "ultra"}),
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
        """Return the capability tiers the MCP caller may choose from.

        Deliberately the tier names (``low``..``ultra``), not raw model slugs:
        the caller reasons about how much capability a task needs, not about
        GPT-5.6 model identities. Each tier maps to a concrete model+effort in
        :attr:`_TIER_LAUNCH`.

        Returns:
            list[str]: Selectable tier names, cheapest first.

        """
        return list(self._TIER_LAUNCH)

    def default_model(self) -> str:
        """Return the default capability tier.

        ``medium`` (Sol @ low) is the token-efficient general-purpose default.

        Returns:
            str: Default tier name.

        """
        return "medium"

    def resolve_model(self, generic_name: str) -> str:
        """Map a tier or raw slug to a Codex model slug (no availability check).

        A tier name resolves to its bundled model slug; a blank string stays
        blank (defer to codex config); any other value passes through
        unchanged. Effort is not considered here, and availability is not
        validated — use :meth:`resolve_launch` for the launch-time pair.

        Args:
            generic_name: Tier name or direct model slug.

        Returns:
            Codex model slug (possibly empty for blank input).

        """
        key = generic_name.strip()
        if not key:
            return ""
        tier = self._TIER_LAUNCH.get(key.lower())
        return tier[0] if tier else key

    def resolve_launch(
        self, model: str, reasoning_effort: str | None
    ) -> tuple[str, str | None]:
        """Resolve caller ``(model, effort)`` into a concrete Codex launch pair.

        - Blank ``model`` -> ``("", effort)``: no ``-c model`` override is
          emitted and codex uses its own ``config.toml`` default (escape hatch).
        - A capability tier (``low``/``medium``/``high``/``xhigh``/``ultra``)
          resolves to its bundled ``(GPT-5.6 slug, effort)``; a caller-supplied
          ``reasoning_effort`` is silently ignored (the tier owns the effort).
        - Any other value is treated as a raw model slug and passes through
          (``reasoning_effort`` still applies here and to blank models).

        The resolved model is validated against the models this codex install
        actually exposes; an unavailable GPT-5.6 model raises
        :class:`BackendModelUnavailableError` rather than downgrading silently.
        """
        key = model.strip()
        if not key:
            return "", reasoning_effort
        tier = self._TIER_LAUNCH.get(key.lower())
        if tier is not None:
            slug, tier_effort = tier
            self._require_available(slug)
            return slug, tier_effort
        self._require_available(key)
        return key, reasoning_effort

    def _available_model_slugs(self) -> list[str]:
        """Return the model slugs this codex install exposes (empty if unknown).

        Sourced from live ``codex debug models`` discovery (cached per process).
        Empty when the binary or its model dump cannot be read — callers treat
        empty as "cannot determine" and skip validation rather than false-error.
        """
        try:
            binary = self.discover_binary()
        except BackendBinaryNotFoundError:
            return []
        return _discover_codex_model_slugs(binary)

    def _require_available(self, slug: str) -> None:
        """Raise if ``slug`` is known to be unavailable on this codex install.

        Validation is skipped when discovery yields nothing (cannot determine),
        so a discovery hiccup never blocks a spawn; codex itself would then
        reject an genuinely-missing model at launch.
        """
        available = self._available_model_slugs()
        if available and slug not in available:
            raise BackendModelUnavailableError(slug, self._name, available)

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
        ``vendor/<triple>/{bin,codex}/codex.exe``.

        Returns ``(exe_path, arch_root)`` or ``None`` when not resolvable.
        """
        if os.name != "nt":
            return None
        target = _WINDOWS_NATIVE_TARGET.get(platform.machine().upper())
        if target is None:
            return None
        arch_suffix, triple = target
        shim_dir = _LOCAL_PATH_CLS(shim_path).parent
        codex_pkg = shim_dir / "node_modules" / "@openai" / "codex"
        platform_pkg = f"codex-win32-{arch_suffix}"
        vendor_rel = _LOCAL_PATH_CLS("vendor") / triple
        bases = [
            codex_pkg / "node_modules" / "@openai" / platform_pkg,
            shim_dir / "node_modules" / "@openai" / platform_pkg,
            codex_pkg,
        ]
        # The native exe sat under ``vendor/<triple>/codex/`` in older Codex
        # releases and moved to ``vendor/<triple>/bin/`` in newer ones. Probe
        # both, newest layout first, so a layout bump doesn't silently fall
        # back to the ``.cmd`` shim (which routes argv through ``cmd.exe`` and
        # truncates the prompt at the first newline).
        for base in bases:
            arch_root = base / vendor_rel
            for exe_subdir in ("bin", "codex"):
                exe = arch_root / exe_subdir / "codex.exe"
                if exe.is_file():
                    return str(exe), str(arch_root)
        return None

    def _headless(self) -> bool:
        """Return whether Codex must run its head-less ``exec`` entrypoint.

        The interactive TUI is strongly preferred (visible session the user
        can watch and type into), but as of codex-cli 0.142.5 it aborts
        immediately with ``Error: stdin is not a terminal`` when the spawned
        process has no real TTY — it then dies before any state hook fires
        and no marker is written. That only happens on the process manager's
        non-console spawn path (e.g. ``WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0``
        on Windows); Windows Terminal tabs, ``CREATE_NEW_CONSOLE``, tmux panes
        and Linux terminal windows all provide a real TTY where the TUI works
        (verified live in PR #14/#18). ``codex exec`` still starts configured
        MCP servers and runs injected state hooks, and accepts all the flags
        we pass — it just has no interactive UI.
        """
        return not process_manager.provides_tty(
            self._name, is_interactive=self.is_interactive
        )

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Codex CLI command.

        Uses the interactive TUI (``codex [OPTIONS] [PROMPT]``) when the
        agent will be attached to a real TTY, and the non-interactive
        ``codex exec`` entrypoint otherwise — see :meth:`_headless`.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.

        """
        binary = self.discover_binary()
        via_cmd_shim = self._launches_via_cmd_shim(binary)
        cmd = [
            binary,
            *(["exec"] if self._headless() else []),
            *self.permission_args(request),
            "-C",
            request.cwd,
            *self._model_args(request),
            *self._mcp_identity_args(request),
            *self._hook_override_args(request),
        ]

        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))

        cmd.extend(self._agent_args(request))

        cmd.append(
            self._prompt_arg(
                request, self._correlated_prompt(request), via_cmd_shim=via_cmd_shim
            )
        )
        return cmd

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        """Build the Codex CLI command for a native session resume.

        Uses the interactive ``codex [OPTIONS] resume <session-id> <prompt>``
        form when the agent gets a real TTY, and the non-interactive
        ``codex exec resume <session-id> [OPTIONS] <prompt>`` entrypoint
        otherwise — see :meth:`_headless`.
        """
        binary = self.discover_binary()
        via_cmd_shim = self._launches_via_cmd_shim(binary)
        headless = self._headless()
        cmd = [
            binary,
            *(["exec", "resume", backend_session_id] if headless else []),
            *self.permission_args(request),
            "-C",
            request.cwd,
            *self._model_args(request),
            *self._mcp_identity_args(request),
            *self._hook_override_args(request),
        ]

        if request.reasoning_effort:
            cmd.extend(self._REASONING_EFFORT_SPEC.build_args(request.reasoning_effort))

        cmd.extend(self._agent_args(request))
        if not headless:
            cmd.extend(["resume", backend_session_id])
        cmd.append(self._prompt_arg(request, via_cmd_shim=via_cmd_shim))
        return cmd

    @staticmethod
    def _hook_override_args(request: SpawnRequest) -> list[str]:
        """Return the Codex ``-c`` hook-override argv (plus trust-bypass flag).

        Enabled by default (confirmed via runtime spike on codex-cli 0.142.5:
        the ``-c`` inline hooks-table override is accepted and works,
        provided ``--dangerously-bypass-hook-trust`` is also passed — without
        it Codex silently skips injected hooks). Honors the global
        ``WIN_AGENT_TEAMS_STATE_HOOKS=0`` kill switch and the Codex-specific
        ``WIN_AGENT_TEAMS_STATE_HOOKS_CODEX=0`` override.

        ``request.extra["hook_overrides"]`` carries the JSON-encoded argv
        list produced by :func:`claude_teams.hooks.codex_hook_overrides`. The
        trust-bypass flag is only appended alongside actual ``-c`` hook args
        (i.e. not when ``hook_overrides`` is absent/empty), since there is
        nothing to trust otherwise.
        """
        if not CodexBackend._codex_hooks_enabled():
            return []
        raw = (request.extra or {}).get("hook_overrides")
        if not raw:
            return []
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(args, list) or not args or not all(
            isinstance(a, str) for a in args
        ):
            return []
        return [*args, "--dangerously-bypass-hook-trust"]

    @staticmethod
    def _codex_hooks_enabled() -> bool:
        """Return whether Codex state hooks are enabled for this process.

        Honors the global ``WIN_AGENT_TEAMS_STATE_HOOKS=0`` kill switch and
        the Codex-specific ``WIN_AGENT_TEAMS_STATE_HOOKS_CODEX=0`` override;
        enabled by default otherwise.
        """
        if os.environ.get("WIN_AGENT_TEAMS_STATE_HOOKS", "1").strip() == "0":
            return False
        return os.environ.get("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "1").strip() != "0"

    def _model_args(self, request: SpawnRequest) -> list[str]:
        """Build the ``-c model=<slug>`` override, if a model was selected.

        A blank ``request.model`` emits nothing so codex falls back to its own
        ``config.toml`` default. The slug is rendered as a TOML single-quoted
        literal (via :meth:`_toml_literal`) because slugs contain ``.``/``-``
        and would not be valid bare TOML values, and single quotes avoid
        Windows ``CreateProcess`` double-quote escaping for the argv token.
        """
        model = request.model.strip() if request.model else ""
        if not model:
            return []
        return ["-c", f"model={self._toml_literal(model)}"]

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
            "AGENT_PARENT_NAME": request.lead_session_id,
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

    @staticmethod
    def _launches_via_cmd_shim(binary: str) -> bool:
        """Return whether ``binary`` is a Windows batch shim run through ``cmd.exe``.

        ``cmd.exe`` truncates an argv token at the first newline (and mangles
        ``< > | & ^ ! ( )``), so a multi-line prompt cannot survive the npm
        ``codex.cmd`` shim verbatim. The native ``.exe`` and direct POSIX
        launches don't go through ``cmd.exe`` and take the prompt as-is.
        """
        return binary.lower().endswith((".cmd", ".bat"))

    def _prompt_arg(
        self,
        request: SpawnRequest,
        prompt: str | None = None,
        *,
        via_cmd_shim: bool = False,
    ) -> str:
        """Return the Codex prompt argument for ``prompt`` (default: request).

        Normally passed verbatim as a single argv token: the native binary is
        launched directly (see :meth:`discover_binary`), so ``CreateProcess``
        and ``CommandLineToArgvW`` round-trip arbitrary text — metacharacters
        and newlines included.

        When the native exe can't be resolved and we fall back to the npm
        ``codex.cmd`` shim (``via_cmd_shim``), ``cmd.exe`` would truncate a
        multi-line prompt at the first newline. As a fallback the prompt is
        then carried as a single-line JSON string and decoded by the agent
        from the leading instruction — no real newlines reach ``cmd.exe``.
        """
        text = request.prompt if prompt is None else prompt
        if via_cmd_shim and ("\n" in text or "\r" in text):
            return (
                "Decode this JSON string as your complete task prompt, then "
                f"follow the decoded text exactly: {json.dumps(text)}"
            )
        return text

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
        prepending the vendored bundled-tools dir (``rg.exe``) to ``PATH``
        and marking the install as npm-managed.
        """
        env = {
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
            "AGENT_PARENT_NAME": request.lead_session_id,
        }
        shim = shutil.which(self._binary_name)
        native = self._resolve_native_codex(shim) if shim else None
        if native:
            _, arch_root = native
            env["CODEX_MANAGED_BY_NPM"] = "1"
            # Bundled-tools dir was ``path`` in older Codex releases and
            # ``codex-path`` in newer ones; prepend whichever exists.
            for tools_subdir in ("codex-path", "path"):
                path_dir = _LOCAL_PATH_CLS(arch_root) / tools_subdir
                if path_dir.is_dir():
                    current = os.environ.get("PATH", "")
                    env["PATH"] = f"{path_dir}{os.pathsep}{current}"
                    break
        return env

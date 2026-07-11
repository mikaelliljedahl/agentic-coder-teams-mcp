"""Pi coding-agent backend integration.

Adapter for the ``pi`` CLI (``@earendil-works/pi-coding-agent``). Pi has, by
design, no built-in MCP support; a spawned pi agent reaches the win-agent-teams
tools through the official ``pi-mcp-adapter`` package (configured via a
generated ``mcp.json`` — see ``server_simple._ensure_pi_mcp_config``) and
reports lifecycle state through a small bundled pi extension
(``pi-extensions/win-agent-teams-state`` — passed via ``-e``).
"""

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from claude_teams.backends.base import (
    BaseBackend,
    SpawnRequest,
)
from claude_teams.backends.contracts import BackendBinaryNotFoundError
from claude_teams.backends.process_manager import process_manager

_LOCAL_PATH_CLS = type(Path.cwd())

# Provider that backs a ChatGPT Plus/Pro (Codex) login. Model tiers below are
# expressed against this provider; if the user is logged into a different
# provider these slugs are absent and the launch falls back to pi's own default
# model (see ``resolve_launch``) rather than erroring.
_PI_PROVIDER = "openai-codex"

# ``pi``'s npm bin script hands off to this entry under the package root. We
# resolve and launch it via ``node`` directly (bypassing the ``pi.cmd`` shim),
# so the prompt argv survives verbatim through ``CreateProcess`` /
# ``CommandLineToArgvW`` — the shim would route it through ``cmd.exe`` and
# truncate a multi-line prompt at the first newline (the same class of bug the
# Codex backend avoids by resolving ``codex.exe`` directly).
_PI_ENTRY_REL = _LOCAL_PATH_CLS(
    "node_modules"
) / "@earendil-works" / "pi-coding-agent" / "dist" / "cli.js"

# Seconds to wait for ``pi --list-models`` before giving up on live model
# discovery and treating the model set as unknown (skip validation).
_MODEL_DISCOVERY_TIMEOUT_S = 20.0

# ``pi --list-models`` rows have at least "<provider> <model>" columns.
_MIN_MODEL_COLUMNS = 2

# Process-lifetime cache of discovered pi model ids, keyed by the resolved
# launcher key. Populated lazily on first tier resolution.
_MODEL_ID_CACHE: dict[str, list[str]] = {}


def _discover_pi_model_ids(launcher: list[str]) -> list[str]:
    """Return the model ids ``pi --list-models`` reports (cached per launcher).

    ``pi --list-models`` prints a whitespace-aligned table whose second column
    is the model id::

        provider      model                context  max-out  thinking  images
        openai-codex  gpt-5.6-sol          372K     128K     yes       yes

    Only the model column is kept. Any failure (binary missing, timeout, not
    logged in) yields ``[]`` so callers treat the set as unknown and skip
    validation rather than crashing or false-erroring.
    """
    cache_key = "\x00".join(launcher)
    if cache_key in _MODEL_ID_CACHE:
        return _MODEL_ID_CACHE[cache_key]
    ids: list[str] = []
    try:
        proc = subprocess.run(  # noqa: S603 - launcher resolved from PATH, fixed argv.
            [*launcher, "--list-models"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_MODEL_DISCOVERY_TIMEOUT_S,
            check=False,
        )
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < _MIN_MODEL_COLUMNS:
                continue
            provider, model = parts[0], parts[1]
            # Skip the header row and any non-table noise.
            if provider in {"provider", "No"} or model in {"model"}:
                continue
            ids.append(model)
    except (OSError, subprocess.SubprocessError):
        ids = []
    _MODEL_ID_CACHE[cache_key] = ids
    return ids


class PiBackend(BaseBackend):
    """Backend adapter for the ``pi`` coding agent CLI."""

    _name = "pi"
    _binary_name = "pi"

    @property
    def is_interactive(self) -> bool:
        """Pi runs interactively so its adapter-configured MCP servers start.

        Returns:
            bool: Always True.

        """
        return True

    # The model interface exposed to the MCP caller: five capability tiers,
    # each bundling a concrete model with a ``--thinking`` level, as an
    # ascending cost/quality ladder. Names mirror the effort words the caller
    # already reasons in (low..ultra), matching the Codex backend's ladder so a
    # coordinator can pick a tier without knowing pi's model slugs. Unlike
    # Codex, an unavailable tier model does NOT error — it falls back to pi's
    # own default model (see ``resolve_launch``), because the user may be logged
    # into a provider whose catalog does not contain these slugs.
    #   low    -> Terra @ medium
    #   medium -> Sol   @ low     (token-efficient general default)
    #   high   -> Sol   @ medium
    #   xhigh  -> Sol   @ high
    #   ultra  -> Sol   @ xhigh
    _TIER_LAUNCH: ClassVar[dict[str, tuple[str, str]]] = {
        "low": ("gpt-5.6-terra", "medium"),
        "medium": ("gpt-5.6-sol", "low"),
        "high": ("gpt-5.6-sol", "medium"),
        "xhigh": ("gpt-5.6-sol", "high"),
        "ultra": ("gpt-5.6-sol", "xhigh"),
    }

    _THINKING_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
    )

    def supported_models(self) -> list[str]:
        """Return the capability tiers the MCP caller may choose from.

        Deliberately the tier names (``low``..``ultra``), not raw model slugs.

        Returns:
            list[str]: Selectable tier names, cheapest first.

        """
        return list(self._TIER_LAUNCH)

    def default_model(self) -> str:
        """Return the default capability tier (``medium``)."""
        return "medium"

    def resolve_model(self, generic_name: str) -> str:
        """Map a tier (or raw slug) to a pi model id (no availability check).

        A tier resolves to its bundled model slug; blank stays blank; any other
        value passes through unchanged. Availability is validated only in
        :meth:`resolve_launch`.
        """
        key = generic_name.strip()
        if not key:
            return ""
        tier = self._TIER_LAUNCH.get(key.lower())
        return tier[0] if tier else key

    def resolve_launch(
        self, model: str, reasoning_effort: str | None
    ) -> tuple[str, str | None]:
        """Resolve caller ``(model, effort)`` into a concrete pi launch pair.

        - Blank ``model`` -> ``("", effort)``: no ``--model`` override, pi uses
          its own default model.
        - A capability tier resolves to its bundled ``(slug, thinking)``; the
          caller-supplied ``reasoning_effort`` is ignored (the tier owns it).
        - Any other value is treated as a raw model id and passes through.

        The resolved slug is checked against the models this pi login actually
        exposes (``pi --list-models``). **Unlike Codex, an unavailable model is
        NOT an error**: it degrades to ``("", thinking)`` so pi falls back to
        its default model. Discovery returning nothing (offline / not logged in)
        skips validation entirely.
        """
        key = model.strip()
        if not key:
            return "", reasoning_effort
        tier = self._TIER_LAUNCH.get(key.lower())
        if tier is not None:
            slug, tier_effort = tier
            return (slug if self._model_available(slug) else ""), tier_effort
        return (key if self._model_available(key) else ""), reasoning_effort

    def _model_available(self, model: str) -> bool:
        """Return whether ``model`` is exposed by this pi login (soft check).

        Returns True when discovery yields nothing ("cannot determine" — never
        block a launch on a discovery hiccup) or when the id (with any
        ``provider/`` prefix stripped) is in the discovered set.
        """
        available = self._available_model_ids()
        if not available:
            return True
        bare = model.split("/", 1)[-1]
        return bare in available

    def _available_model_ids(self) -> list[str]:
        """Return the model ids this pi login exposes (empty if unknown)."""
        try:
            launcher = self._launcher()
        except BackendBinaryNotFoundError:
            return []
        return _discover_pi_model_ids(launcher)

    # ------------------------------------------------------------------
    # Launcher resolution (node + cli.js, bypassing the pi.cmd shim)
    # ------------------------------------------------------------------

    def discover_binary(self) -> str:
        """Return the resolved pi launcher's first token (``node`` or the shim).

        Kept for protocol/compat; :meth:`_launcher` is the full argv prefix.
        """
        return self._launcher()[0]

    def _launcher(self) -> list[str]:
        """Return the argv prefix that runs pi.

        Prefers ``[node, <pkg>/dist/cli.js]`` so the prompt argv bypasses the
        ``pi.cmd`` shim / ``cmd.exe`` and survives verbatim. Falls back to the
        bare ``pi`` shim path when either ``node`` or the entry script cannot be
        located (non-npm layout, unusual install).
        """
        shim = shutil.which(self._binary_name)
        if shim is None:
            raise BackendBinaryNotFoundError(self._binary_name, self._name)
        entry = self._resolve_entry(shim)
        node = shutil.which("node")
        if entry and node:
            return [node, str(entry)]
        return [shim]

    @staticmethod
    def _resolve_entry(shim_path: str) -> Path | None:
        """Locate ``dist/cli.js`` next to the npm ``pi`` shim."""
        entry = _LOCAL_PATH_CLS(shim_path).parent / _PI_ENTRY_REL
        return entry if entry.is_file() else None

    def _launches_via_shim(self) -> bool:
        """Return whether we fell back to the ``pi.cmd`` shim (cmd.exe path)."""
        return len(self._launcher()) == 1

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def default_permission_args(self) -> list[str]:
        """Trust project-local files so an autonomous agent never blocks on it.

        Pi has no permission popups; ``-a`` opts into loading project-local
        extensions/skills for this run without an interactive trust prompt.
        """
        return ["-a"]

    def supports_resume(self) -> bool:
        """Pi supports session resume via ``--session-id``/``--continue``."""
        return True

    def _headless(self) -> bool:
        """Return whether pi must run headless (``-p --mode json``).

        The interactive TUI is preferred (a visible tab the user can watch), but
        it needs a real TTY. On the process manager's non-console spawn path
        there is none, so we fall back to non-interactive print mode, which
        still loads the ``-e`` extension and the adapter's MCP servers.
        """
        return not process_manager.provides_tty(
            self._name, is_interactive=self.is_interactive
        )

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the pi launch command for a spawn request."""
        cmd = [
            *self._launcher(),
            *(["-p", "--mode", "json"] if self._headless() else []),
            *self.permission_args(request),
            "--session-dir",
            self._pi_session_dir(request),
            "--session-id",
            request.name,
            *self._model_args(request),
            *self._extension_args(request),
        ]
        cmd.extend(self._prompt_args(request))
        return cmd

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        """Build the pi command that resumes a prior session.

        The session id we set at spawn (``request.name``) is stable and
        directory-scoped, so resume re-targets the same ``--session-dir`` and
        ``--session-id`` and continues it. ``backend_session_id`` (pi's own
        header id, discovered from disk) is accepted for symmetry but the
        deterministic name binding is authoritative.
        """
        _ = backend_session_id
        cmd = [
            *self._launcher(),
            *(["-p", "--mode", "json"] if self._headless() else []),
            *self.permission_args(request),
            "--session-dir",
            self._pi_session_dir(request),
            "--session-id",
            request.name,
            "--continue",
            *self._model_args(request),
            *self._extension_args(request),
        ]
        cmd.extend(self._prompt_args(request))
        return cmd

    def _pi_session_dir(self, request: SpawnRequest) -> str:
        """Return the per-agent pi session storage dir under the team session dir.

        ``extra["session_dir"]`` is the win-agent-teams session directory
        (``~/.claude/agent-sessions/<session-id>``), passed by the server. Pi
        sessions live in a dedicated ``pi-sessions/<agent>`` subdir so
        :func:`claude_teams.agent_output.read_pi_output` can find exactly this
        agent's rollout.
        """
        base = (request.extra or {}).get("session_dir") or request.cwd
        return str(_LOCAL_PATH_CLS(base) / "pi-sessions" / request.name)

    def _model_args(self, request: SpawnRequest) -> list[str]:
        """Build ``--model``/``--thinking`` args for the resolved launch.

        A blank ``request.model`` emits no ``--model`` (pi's default model). A
        bare slug is qualified with the ``openai-codex/`` provider so pi selects
        the intended model; a value already containing ``/`` passes through.
        ``--thinking`` is emitted whenever a valid level was resolved, and is
        provider-agnostic, so it is kept even on the default-model fallback.
        """
        args: list[str] = []
        model = (request.model or "").strip()
        if model:
            qualified = model if "/" in model else f"{_PI_PROVIDER}/{model}"
            args.extend(["--model", qualified])
        effort = (request.reasoning_effort or "").strip()
        if effort in self._THINKING_OPTIONS:
            args.extend(["--thinking", effort])
        return args

    @staticmethod
    def _extension_args(request: SpawnRequest) -> list[str]:
        """Build the ``-e <path>`` arg loading the state-reporting extension."""
        ext = (request.extra or {}).get("pi_state_extension_path")
        return ["-e", str(ext)] if ext else []

    def _prompt_args(self, request: SpawnRequest) -> list[str]:
        """Return the initial-prompt argv for pi.

        Direct ``node`` launch takes the prompt verbatim as a single positional
        argument. On the shim fallback (``cmd.exe`` would mangle newlines) the
        prompt is delivered as a ``@<file>`` include instead, when the server
        wrote one, with a short plain-ASCII directive.
        """
        prompt_file = (request.extra or {}).get("prompt_file_path")
        if self._launches_via_shim() and prompt_file:
            return [f"@{prompt_file}", "Complete the task in the attached file."]
        return [request.prompt]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Pass agent identity and the state-marker target dir to pi.

        ``AGENT_*`` are consumed both by the ``pi-mcp-adapter`` mcp.json
        (``${AGENT_NAME}`` interpolation → the win-agent-teams MCP server's
        identity) and by the state extension. ``WIN_AGENT_TEAMS_SESSION_DIR``
        tells the extension where to write ``state-<agent>.json``.
        """
        env = {
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
            "AGENT_PARENT_NAME": request.lead_session_id,
        }
        session_dir = (request.extra or {}).get("session_dir")
        if session_dir:
            env["WIN_AGENT_TEAMS_SESSION_DIR"] = str(session_dir)
        return env

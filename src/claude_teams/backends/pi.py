"""Pi coding-agent backend integration.

Adapter for the ``pi`` CLI (``@earendil-works/pi-coding-agent``). Pi has, by
design, no built-in MCP support; a spawned pi agent reaches the win-agent-teams
tools through the official ``pi-mcp-adapter`` package (configured via a
generated ``mcp.json`` — see ``server_simple._ensure_pi_mcp_config``) and
reports lifecycle state through a small bundled pi extension
(``pi-extensions/win-agent-teams-state`` — passed via ``-e``).
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from claude_teams.agent_output import CORRELATION_FIELD, correlated_prompt
from claude_teams.backends.base import (
    BaseBackend,
    SpawnRequest,
)
from claude_teams.backends.contracts import (
    BackendBinaryNotFoundError,
    prefer_luna_tiers,
)
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
_PI_ENTRY_REL = (
    _LOCAL_PATH_CLS("node_modules")
    / "@earendil-works"
    / "pi-coding-agent"
    / "dist"
    / "cli.js"
)

# Seconds to wait for ``pi --list-models`` before giving up on live model
# discovery and treating the model set as unknown (skip validation).
_MODEL_DISCOVERY_TIMEOUT_S = 20.0

# ``pi --list-models`` rows have at least "<provider> <model>" columns.
_MIN_MODEL_COLUMNS = 2

# Process-lifetime cache of discovered pi model ids, keyed by the resolved
# launcher key. Populated lazily on first tier resolution.
_MODEL_ID_CACHE: dict[str, list[str]] = {}

logger = logging.getLogger(__name__)

# Tool names that ask a human a question and wait for the answer. Pi *core*
# registers none of them, but a user-global extension can (the common one is
# ``ask_user``, from a personal ``~/.pi/agent/settings.json`` extension). Such
# an extension self-disables when ``ctx.mode !== "tui"`` — and on Windows we
# deliberately run pi in its TUI, in a tab nobody is watching, so the question
# renders and the agent blocks forever. ``--exclude-tools`` removes the tool
# from the model's tool list outright. Pi tolerates names that are not
# registered, so one static list is safe across installs.
_DEFAULT_EXCLUDED_TOOLS = "ask_user,ask_question,ask_human,request_input"

# Override for the deny list above (comma-separated). An explicitly empty value
# omits ``--exclude-tools`` entirely — the debugging escape hatch for an
# operator who wants the interactive tool back.
_EXCLUDE_TOOLS_ENV = "WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS"

# Belt-and-braces companion to the deny list: policy the model reads, so it
# escalates instead of merely failing to find a question tool. Appended
# verbatim via ``--append-system-prompt``; kept to three lines because every
# turn of every pi agent pays for it. Note pi treats an ``--append-system-prompt``
# value that names an existing file as that file's contents — this text cannot
# be mistaken for a path.
_ESCALATION_POLICY = (
    "You run non-interactively: no human is watching your terminal, so never "
    "ask a human a question and never wait for one to answer.\n"
    "When you are blocked or need a decision, call the MCP tool send_message "
    "(exposed here as win_agent_teams_send_message); it defaults to the parent "
    "agent that spawned you.\n"
    "Then either stop, or continue under an assumption you state explicitly in "
    "your final message."
)

# Characters pi resolves from the FIRST character of the prompt token: ``@``
# makes it a file include, ``/`` an extension command / skill / prompt-template
# invocation, ``-`` a CLI flag. The same characters on any later line are inert.
_GUARDED_LEADING_CHARS = ("@", "/", "-")

# Above this many characters the prompt no longer travels as a command-line
# argument on the headless path. Windows' command line breaks somewhere past
# ~32 KB (``[WinError 206]``); 24 KB leaves room for the rest of the argv. The
# TUI path bakes argv into a ``.ps1`` wrapper and has no such ceiling.
MAX_ARGV_PROMPT_CHARS = 24 * 1024


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

    # The model interface exposed to the MCP caller: six capability tiers,
    # each bundling a concrete model with a ``--thinking`` level, as an
    # ascending cost/quality ladder. Names mirror the effort words the caller
    # already reasons in (cheapest..max), matching the Codex backend's ladder so a
    # coordinator can pick a tier without knowing pi's model slugs. Unlike
    # Codex, an unavailable tier model does NOT error — it falls back to pi's
    # own default model (see ``resolve_launch``), because the user may be logged
    # into a provider whose catalog does not contain these slugs.
    #   cheapest -> Luna @ medium
    #   low    -> Luna  @ high
    #   medium -> Luna  @ xhigh   (token-efficient general default)
    #   high   -> Sol   @ medium
    #   xhigh  -> Sol   @ high
    #   max    -> Sol   @ xhigh
    # The cheap end runs Luna rather than Terra or Sol for the quota reasons
    # documented on the Codex ladder; Terra is off the ladder but still
    # reachable as a raw slug.
    _TIER_LAUNCH: ClassVar[dict[str, tuple[str, str]]] = {
        "cheapest": ("gpt-5.6-luna", "medium"),
        "low": ("gpt-5.6-luna", "high"),
        "medium": ("gpt-5.6-luna", "xhigh"),
        "high": ("gpt-5.6-sol", "medium"),
        "xhigh": ("gpt-5.6-sol", "high"),
        "max": ("gpt-5.6-sol", "xhigh"),
    }

    # Opt-in ladder, selected by ``WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS=1``
    # (see :func:`claude_teams.backends.contracts.prefer_luna_tiers`). Identical
    # to the Codex backend's opt-in ladder, for the same reason the default
    # ladders match: a coordinator picks a tier without knowing which backend
    # will run it, so the two must not diverge.
    #   cheapest -> Luna @ medium   (unchanged)
    #   low      -> Luna @ high     (unchanged)
    #   medium   -> Luna @ xhigh    (unchanged)
    #   high     -> Luna @ max      (was Sol @ medium)
    #   xhigh    -> Sol  @ medium   (was Sol @ high)
    #   max      -> Sol  @ high     (was Sol @ xhigh)
    _TIER_LAUNCH_PREFER_LUNA: ClassVar[dict[str, tuple[str, str]]] = {
        "cheapest": ("gpt-5.6-luna", "medium"),
        "low": ("gpt-5.6-luna", "high"),
        "medium": ("gpt-5.6-luna", "xhigh"),
        "high": ("gpt-5.6-luna", "max"),
        "xhigh": ("gpt-5.6-sol", "medium"),
        "max": ("gpt-5.6-sol", "high"),
    }

    def _tier_launch(self) -> dict[str, tuple[str, str]]:
        """Return the tier ladder currently in force.

        The Luna-preferring ladder when
        ``WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS=1``, otherwise the
        default. Both ladders carry the same tier names in the same order.
        """
        if prefer_luna_tiers():
            return self._TIER_LAUNCH_PREFER_LUNA
        return self._TIER_LAUNCH

    _THINKING_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
    )

    def supported_models(self) -> list[str]:
        """Return the capability tiers the MCP caller may choose from.

        Deliberately the tier names (``cheapest``..``max``), not raw model slugs.

        Returns:
            list[str]: Selectable tier names, cheapest first.

        """
        return list(self._tier_launch())

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
        tier = self._tier_launch().get(key.lower())
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
        tier = self._tier_launch().get(key.lower())
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
            *self._mcp_config_args(request),
            *self._model_args(request),
            *self._extension_args(request),
            *self._autonomy_args(),
        ]
        cmd.extend(self._prompt_args(request))
        return cmd

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        """Build the pi command that resumes a prior session.

        Resume re-targets the same per-agent ``--session-dir`` (which holds this
        agent's single rollout) and passes ``--continue``. It must NOT also pass
        ``--session-id``: pi's CLI rejects that combination
        (``--session-id cannot be combined with --continue``) and exits
        immediately, which the delivery layer reports as
        ``resume_not_confirmed``. The directory scope makes ``--continue``
        unambiguous. ``backend_session_id`` (pi's own header id, discovered from
        disk) is accepted for symmetry but unused.
        """
        _ = backend_session_id
        cmd = [
            *self._launcher(),
            *(["-p", "--mode", "json"] if self._headless() else []),
            *self.permission_args(request),
            "--session-dir",
            self._pi_session_dir(request),
            "--continue",
            *self._mcp_config_args(request),
            *self._model_args(request),
            *self._extension_args(request),
            *self._autonomy_args(),
        ]
        cmd.extend(self._prompt_args(request))
        return cmd

    @staticmethod
    def _autonomy_args() -> list[str]:
        """Build the args that stop a spawned pi agent waiting on a human.

        Two layers, deliberately redundant:

        - ``--exclude-tools`` removes the human-question tools from the model's
          tool list, so the deadlock is unreachable rather than discouraged.
        - ``--append-system-prompt`` tells the model what to do *instead* —
          escalate to the parent through the MCP ``send_message`` tool — because
          a model that merely cannot ask may otherwise stall or guess silently.

        Emitted on spawn and on resume alike: a resumed agent is launched with a
        fresh argv, and a policy that lapses on resume is no policy at all.
        """
        args: list[str] = []
        excluded = os.environ.get(_EXCLUDE_TOOLS_ENV, _DEFAULT_EXCLUDED_TOOLS).strip()
        if excluded:
            args.extend(["--exclude-tools", excluded])
        args.extend(["--append-system-prompt", _ESCALATION_POLICY])
        return args

    @staticmethod
    def _mcp_config_args(request: SpawnRequest) -> list[str]:
        """Build the ``--mcp-config <path>`` args for the per-agent pi MCP file.

        The server writes ``<session_dir>/mcp/<agent>.pi.mcp.json`` with LITERAL
        ``AGENT_*`` identity (see ``server_simple._write_pi_mcp_config``) and
        passes its path in ``extra["pi_mcp_config_path"]``. Passing it via
        ``--mcp-config`` redirects the pi-mcp-adapter's pi-global config source
        to this file, so identity is not left to ``${AGENT_*}`` interpolation
        (which a lower-precedence empty ``env`` block can clobber). Emitted as a
        discrete argv token so a Windows path with spaces survives the launch.
        Omitted entirely when the path is absent.
        """
        path = (request.extra or {}).get("pi_mcp_config_path")
        return ["--mcp-config", str(path)] if path else []

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
        """Build the ``-e <path>`` args loading the bundled pi extensions.

        Two extensions are loaded, each via its own ``-e``:

        - the state-reporting extension (``pi_state_extension_path``), and
        - the inbox-wake extension (``pi_wake_extension_path``), so a spawned Pi
          agent that becomes a lead for level-2 children is woken when they post
          to its own inbox (nested orchestration; guarded off the injected
          ``WIN_AGENT_TEAMS_SESSION_DIR`` inside the extension).

        Either is omitted when its path is absent from ``request.extra``.
        """
        extra = request.extra or {}
        args: list[str] = []
        for key in ("pi_state_extension_path", "pi_wake_extension_path"):
            ext = extra.get(key)
            if ext:
                args.extend(["-e", str(ext)])
        return args

    @staticmethod
    def _guard_leading_char(prompt: str) -> str:
        """Return ``prompt`` defused against pi's leading-character dispatch.

        Pi decides ``@file`` / ``/command`` / ``-flag`` from the first character
        of the argv token and never from a later line, so one leading newline
        makes the whole prompt ordinary text again. Chosen over stripping or
        escaping because it loses no byte of the caller's prompt — pi hands the
        text to the model unchanged, leading whitespace included.
        """
        if prompt.startswith(_GUARDED_LEADING_CHARS):
            return f"\n{prompt}"
        return prompt

    @staticmethod
    def _correlated_prompt(request: SpawnRequest) -> str:
        """Return the prompt with this spawn's correlation marker appended.

        Pi takes the prompt verbatim as a single positional argument on the
        ``node`` launch path, so — like Codex, and unlike Claude Code — the
        marker can travel in the prompt itself and the backend appends it. The
        server therefore leaves a pi prompt alone in ``_materialize_prompt``;
        marking in both places would give pi two markers.

        The id comes from ``extra`` and is never derived here. A record that
        predates correlation carries no id, and inventing one would produce a
        marker no existing transcript can contain.
        """
        correlation_id = (request.extra or {}).get(CORRELATION_FIELD)
        prompt = PiBackend._guard_leading_char(request.prompt)
        if not correlation_id:
            return prompt
        return correlated_prompt(prompt, str(correlation_id), single_line=False)

    def _prompt_args(self, request: SpawnRequest) -> list[str]:
        """Return the initial-prompt argv for pi.

        Direct ``node`` launch takes the prompt verbatim as a single positional
        argument, which is the only fully faithful transport (pi wraps a
        ``@file`` include in ``<file name="…">…</file>`` rather than inlining it
        verbatim). The ``@<file>`` sidecar is therefore used only where argv
        cannot carry the prompt intact, and only when the server wrote one:

        - the ``pi.cmd`` shim fallback, where ``cmd.exe`` truncates the argv at
          the first newline; and
        - a very long prompt on the headless path, where the real command line
          runs into the Windows length ceiling. The TUI path bakes argv into a
          ``.ps1`` wrapper and is exempt.

        A shim launch with a multi-line prompt and no sidecar is the one case
        with no safe transport left; it is logged rather than silently truncated
        downstream.
        """
        prompt = self._correlated_prompt(request)
        prompt_file = (request.extra or {}).get("prompt_file_path")
        via_shim = self._launches_via_shim()
        oversize = self._headless() and len(prompt) > MAX_ARGV_PROMPT_CHARS
        if prompt_file and (via_shim or oversize):
            return [f"@{prompt_file}", "Complete the task in the attached file."]
        if via_shim and "\n" in prompt:
            logger.warning(
                "pi agent %s launches via the pi.cmd shim with a multi-line "
                "prompt and no prompt sidecar: cmd.exe will truncate the prompt "
                "at the first newline",
                request.name,
            )
        return [prompt]

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

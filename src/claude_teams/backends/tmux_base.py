"""Shared tmux-backed backend implementation helpers."""

import shlex
import shutil
from typing import cast

from claude_code_tools.tmux_cli_controller import TmuxCLIController

from claude_teams.backends.contracts import (
    _SAFE_ENV_KEY,
    AgentProfile,
    AgentSelectSpec,
    CaptureResult,
    HealthStatus,
    ReasoningEffortSpec,
    SpawnRequest,
    SpawnResult,
)
from claude_teams.errors import (
    BackendBinaryNotFoundError,
    InvalidEnvVarNameError,
    PermissionBypassUnsupportedValueError,
    TmuxPaneCreationError,
)


class BaseBackend:
    """Convenience base class with shared tmux lifecycle management."""

    _name: str
    _binary_name: str

    def __init__(self) -> None:
        """Initialize the lazy tmux controller handle."""
        self._controller: TmuxCLIController | None = None

    @property
    def controller(self) -> TmuxCLIController:
        """Return the lazily-created tmux controller."""
        if self._controller is None:
            self._controller = TmuxCLIController()
        return self._controller

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return self._name

    @property
    def binary_name(self) -> str:
        """Return the backend CLI binary name."""
        return self._binary_name

    @property
    def is_interactive(self) -> bool:
        """Return whether the backend handles native team messaging."""
        return False

    def retain_pane_after_exit(self, handle: str) -> None:
        """Keep a tmux pane alive after the process exits.

        Args:
            handle (str): Tmux pane identifier.

        """
        self.controller._run_tmux_command(
            ["set-option", "-p", "-t", handle, "remain-on-exit", "on"]
        )

    def is_available(self) -> bool:
        """Return whether the backend binary is on PATH."""
        return shutil.which(self._binary_name) is not None

    def discover_binary(self) -> str:
        """Return the full path to the backend binary.

        Returns:
            str: Resolved binary path.

        Raises:
            BackendBinaryNotFoundError: If the binary is not on PATH.

        """
        path = shutil.which(self._binary_name)
        if path is None:
            raise BackendBinaryNotFoundError(self._binary_name, self._name)
        return path

    def default_permission_args(self) -> list[str]:
        """Return this backend's default permission-related CLI args."""
        return []

    def bypass_permission_args(self) -> list[str]:
        """Return CLI args that explicitly request permission bypass."""
        return self.default_permission_args()

    def supports_permission_bypass(self) -> bool:
        """Return whether this backend supports explicit bypass mode."""
        return bool(self.bypass_permission_args())

    def permission_args(self, request: SpawnRequest) -> list[str]:
        """Resolve permission-related CLI args for the request mode.

        Args:
            request (SpawnRequest): Spawn request.

        Returns:
            list[str]: Permission-related CLI args.

        Raises:
            ValueError: If bypass mode is requested but unsupported.

        """
        if request.permission_mode == "require_approval":
            return []
        if request.permission_mode == "bypass":
            args = self.bypass_permission_args()
            if not args:
                raise PermissionBypassUnsupportedValueError(self.name)
            return args
        return self.default_permission_args()

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn the agent in a new tmux pane.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            SpawnResult: Spawn result with tmux handle.

        Raises:
            ValueError: If an environment variable name is invalid.
            RuntimeError: If the tmux pane could not be created.

        """
        cmd_parts = self.build_command(request)
        env_vars = self.build_env(request)

        for key in env_vars:
            if not _SAFE_ENV_KEY.match(key):
                raise InvalidEnvVarNameError(key)

        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in env_vars.items()
        )
        cmd_str = " ".join(shlex.quote(part) for part in cmd_parts)
        full_cmd = (
            f"cd {shlex.quote(request.cwd)} && {env_prefix} {cmd_str}"
            if env_prefix
            else f"cd {shlex.quote(request.cwd)} && {cmd_str}"
        )

        pane_id = self.controller.launch_cli(full_cmd)
        if pane_id is None:
            raise TmuxPaneCreationError(request.name)
        return SpawnResult(process_handle=pane_id, backend_type=self._name)

    def health_check(self, handle: str) -> HealthStatus:
        """Check whether a spawned agent is still running.

        Args:
            handle (str): Tmux pane identifier.

        Returns:
            HealthStatus: Liveness result.

        """
        panes = self.controller.list_panes()
        pane_exists = any(
            handle in (pane.get("id", ""), pane.get("formatted_id", ""))
            for pane in panes
        )
        if not pane_exists:
            return HealthStatus(alive=False, detail="tmux pane not found")

        output, code = self.controller._run_tmux_command(
            ["display-message", "-t", handle, "-p", "#{pane_dead}"]
        )
        if code == 0 and output.strip() == "1":
            return HealthStatus(alive=False, detail="process exited (pane retained)")

        return HealthStatus(alive=True, detail="tmux pane check")

    def kill(self, handle: str) -> None:
        """Force-kill a spawned agent pane.

        Args:
            handle (str): Tmux pane identifier.

        """
        self.controller.kill_pane(pane_id=handle)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Attempt graceful shutdown with Ctrl+C and idle wait.

        Args:
            handle (str): Tmux pane identifier.
            timeout_s (float): Maximum seconds to wait.

        Returns:
            bool: Whether the pane became idle within the timeout.

        """
        self.controller.send_interrupt(pane_id=handle)
        return self.controller.wait_for_idle(
            pane_id=handle,
            idle_time=1.0,
            timeout=int(timeout_s),
        )

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture current tmux pane output.

        Args:
            handle (str): Tmux pane identifier.
            lines (int | None): Optional line limit.

        Returns:
            str: Captured pane output.

        """
        return self.controller.capture_pane(pane_id=handle, lines=lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send text input to a tmux pane.

        Args:
            handle (str): Tmux pane identifier.
            text (str): Text to send.
            enter (bool): Whether to press Enter after the text.

        """
        self.controller.send_keys(text, pane_id=handle, enter=enter)

    def wait_idle(
        self,
        handle: str,
        idle_time: float = 2.0,
        timeout: int | None = None,
    ) -> bool:
        """Wait until a pane's output stabilizes.

        Args:
            handle (str): Tmux pane identifier.
            idle_time (float): Stable duration threshold.
            timeout (int | None): Maximum seconds to wait.

        Returns:
            bool: Whether the pane became idle.

        """
        return self.controller.wait_for_idle(
            pane_id=handle,
            idle_time=idle_time,
            timeout=timeout,
        )

    def execute_in_pane(
        self,
        handle: str,
        command: str,
        timeout: int = 30,
    ) -> CaptureResult:
        """Execute a shell command in a pane and return its result.

        Args:
            handle (str): Tmux pane identifier.
            command (str): Shell command to execute.
            timeout (int): Maximum seconds to wait.

        Returns:
            CaptureResult: Output and exit code.

        """
        return cast(
            CaptureResult,
            self.controller.execute(command, pane_id=handle, timeout=timeout),
        )

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the backend command.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts suitable for tmux execution.

        Raises:
            NotImplementedError: Always, subclasses must override.

        """
        raise NotImplementedError

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Return the backend's reasoning-effort spec.

        Default is ``None`` (backend does not expose a reasoning-effort knob).
        Supporting backends override this to publish their flag syntax and the
        fixed enum of accepted values.

        Returns:
            Spec describing flag + accepted values, or ``None``.

        """
        return None

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Return the backend's agent-selection spec.

        Default is ``None`` (backend does not support launching a named agent
        profile). Supporting backends override this to publish the flag and
        value template their CLI accepts.

        Returns:
            Spec describing flag + value template, or ``None``.

        """
        return None

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Enumerate agent profiles available to this backend.

        Default is an empty list. Backends that support profile selection
        override this to scan the filesystem (``.claude/agents/*.md``), parse
        configuration (``~/.codex/config.toml``), or walk env-var path lists
        (``$GOOSE_RECIPE_PATH``). ``cwd`` anchors project-local discovery so
        per-project profiles surface alongside user-global ones.

        Args:
            cwd: Absolute path anchoring project-local discovery.

        Returns:
            Discovered profiles, possibly empty.

        """
        _ = cwd
        return []

    def _agent_args(self, request: SpawnRequest) -> list[str]:
        """Build ``agent_profile``-related CLI args, if selection is active.

        Prefers a pre-resolved path from ``request.extra['agent_profile_path']``
        when the server-layer validator already discovered and matched the
        profile — the common path, and skips the filesystem/TOML scan that
        validation just performed. Falls back to re-running
        ``discover_agents(request.cwd)`` so direct callers (notably backend
        unit tests constructing ``SpawnRequest`` without routing through
        ``spawn_teammate``) keep working. A missing match produces an empty
        list rather than raising so the build stays pure.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Two-element flag/value pair for the resolved profile, or an
            empty list if no profile is set, the backend lacks a spec, or
            no discovered profile matches.

        """
        if not request.agent_profile:
            return []
        spec = self.agent_select_spec()
        if spec is None:
            return []
        cached_path = (request.extra or {}).get("agent_profile_path")
        if cached_path:
            return spec.build_args(
                AgentProfile(name=request.agent_profile, path=cached_path)
            )
        for profile in self.discover_agents(request.cwd):
            if profile.name == request.agent_profile:
                return spec.build_args(profile)
        return []

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Build additional environment variables for the spawned backend.

        Default implementation returns an empty dict. Subclasses override
        only when they need to export custom environment variables. The
        ``request`` parameter is unused in the default but is part of the
        contract so overrides can customize based on spawn inputs (e.g.,
        reading ``request.extra``).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Environment mapping. Empty by default.

        """
        _ = request
        return {}

    def supported_models(self) -> list[str]:
        """Return supported model identifiers.

        Returns:
            list[str]: Curated model identifiers.

        Raises:
            NotImplementedError: Always, subclasses must override.

        """
        raise NotImplementedError

    def default_model(self) -> str:
        """Return the default model identifier.

        Returns:
            str: Default model identifier.

        Raises:
            NotImplementedError: Always, subclasses must override.

        """
        raise NotImplementedError

    def resolve_model(self, generic_name: str) -> str:
        """Resolve a generic model name to a backend-specific identifier.

        Args:
            generic_name (str): Generic or backend-specific model name.

        Returns:
            str: Backend-specific model identifier.

        Raises:
            NotImplementedError: Always, subclasses must override.

        """
        raise NotImplementedError

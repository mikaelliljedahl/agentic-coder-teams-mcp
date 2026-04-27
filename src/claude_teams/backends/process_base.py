"""Shared Windows process-backed backend implementation helpers."""

import os
import shutil
import subprocess

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
from claude_teams.backends.process_manager import process_manager
from claude_teams.errors import (
    BackendBinaryNotFoundError,
    InvalidEnvVarNameError,
    PermissionBypassUnsupportedValueError,
)


class BaseBackend:
    """Convenience base class with shared Windows process lifecycle management."""

    _name: str
    _binary_name: str

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
        """No-op compatibility hook for one-shot process backends."""
        _ = handle

    def is_available(self) -> bool:
        """Return whether the backend binary is on PATH."""
        return shutil.which(self._binary_name) is not None

    def discover_binary(self) -> str:
        """Return the full path to the backend binary."""
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
        """Resolve permission-related CLI args for the request mode."""
        if request.permission_mode == "require_approval":
            return []
        if request.permission_mode == "bypass":
            args = self.bypass_permission_args()
            if not args:
                raise PermissionBypassUnsupportedValueError(self.name)
            return args
        return self.default_permission_args()

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn the agent as a Windows-native process."""
        cmd_parts = self.build_command(request)
        env_vars = self.build_env(request)

        for key in env_vars:
            if not _SAFE_ENV_KEY.match(key):
                raise InvalidEnvVarNameError(key)

        return process_manager.spawn_process(
            request, cmd_parts, env_vars, self._name, is_interactive=self.is_interactive
        )

    def health_check(self, handle: str) -> HealthStatus:
        """Check whether a spawned agent is still running."""
        alive, detail = process_manager.health_check(handle)
        return HealthStatus(alive=alive, detail=detail)

    def kill(self, handle: str) -> None:
        """Force-kill a spawned agent process."""
        process_manager.kill_process(handle)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Attempt graceful shutdown for a spawned process."""
        return process_manager.graceful_shutdown(handle, timeout_s=timeout_s)

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture stdout/stderr log output for a process."""
        return process_manager.capture(handle, lines=lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send text input to a process stdin pipe."""
        process_manager.send(handle, text, enter=enter)

    def wait_idle(
        self,
        handle: str,
        idle_time: float = 2.0,
        timeout: int | None = None,
    ) -> bool:
        """Return true when the process has exited before ``timeout``."""
        _ = idle_time
        status = self.health_check(handle)
        if not status.alive:
            return True
        if timeout is None:
            return False
        try:
            subprocess.run(  # noqa: S603 - backend binary path is resolved from PATH.
                [self.discover_binary(), "--version"],
                timeout=timeout,
                check=False,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return not self.health_check(handle).alive

    def execute_in_pane(
        self,
        handle: str,
        command: str,
        timeout: int = 30,
    ) -> CaptureResult:
        """Execute a shell command in the current process working directory."""
        _ = handle
        shell_cmd = (
            ["cmd", "/c", command] if os.name == "nt" else ["sh", "-lc", command]
        )
        completed = subprocess.run(  # noqa: S603 - command is explicit backend API input.
            shell_cmd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return {
            "output": completed.stdout + completed.stderr,
            "exit_code": completed.returncode,
        }

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the backend command."""
        raise NotImplementedError

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Return the backend's reasoning-effort spec."""
        return None

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Return the backend's agent-selection spec."""
        return None

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Enumerate agent profiles available to this backend."""
        _ = cwd
        return []

    def _agent_args(self, request: SpawnRequest) -> list[str]:
        """Build ``agent_profile``-related CLI args, if selection is active."""
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
        """Build additional environment variables for the spawned backend."""
        _ = request
        return {}

    def supported_models(self) -> list[str]:
        """Return supported model identifiers."""
        raise NotImplementedError

    def default_model(self) -> str:
        """Return the default model identifier."""
        raise NotImplementedError

    def resolve_model(self, generic_name: str) -> str:
        """Resolve a generic model name to a backend-specific identifier."""
        raise NotImplementedError

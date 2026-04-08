"""Shared backend contracts and request/result types."""

import re
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable


class CaptureResult(TypedDict):
    """Result of executing a command in a tmux pane."""

    output: str
    exit_code: int


@dataclass(frozen=True)
class SpawnRequest:
    """Backend-agnostic spawn parameters."""

    agent_id: str
    name: str
    team_name: str
    prompt: str
    model: str
    agent_type: str
    color: str
    cwd: str
    lead_session_id: str
    permission_mode: Literal["default", "require_approval", "bypass"] = "default"
    plan_mode_required: bool = False
    extra: dict[str, str] | None = None


@dataclass(frozen=True)
class SpawnResult:
    """What a backend returns after spawning."""

    process_handle: str
    backend_type: str


@dataclass(frozen=True)
class HealthStatus:
    """Health check result."""

    alive: bool
    detail: str = ""


@runtime_checkable
class Backend(Protocol):
    """Protocol that all spawner backends must satisfy."""

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        ...

    @property
    def is_interactive(self) -> bool:
        """Return whether the backend handles native team messaging."""
        ...

    def retain_pane_after_exit(self, handle: str) -> None:
        """Keep the backend pane available after the child process exits."""
        ...

    @property
    def binary_name(self) -> str:
        """Return the backend CLI binary name."""
        ...

    def is_available(self) -> bool:
        """Return whether the backend is available on this machine."""
        ...

    def discover_binary(self) -> str:
        """Return the resolved path to the backend binary."""
        ...

    def supported_models(self) -> list[str]:
        """Return the curated model identifiers supported by the backend."""
        ...

    def default_model(self) -> str:
        """Return the backend's default model identifier."""
        ...

    def resolve_model(self, generic_name: str) -> str:
        """Resolve a generic model alias to a backend-specific identifier."""
        ...

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the backend command line for a spawn request."""
        ...

    def supports_permission_bypass(self) -> bool:
        """Return whether explicit permission bypass is supported."""
        ...

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Build environment variables for a spawn request."""
        ...

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn a backend worker and return its process handle."""
        ...

    def health_check(self, handle: str) -> HealthStatus:
        """Check whether the backend worker identified by ``handle`` is alive."""
        ...

    def kill(self, handle: str) -> None:
        """Force-stop the backend worker identified by ``handle``."""
        ...

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Attempt a graceful shutdown for the backend worker."""
        ...

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture backend output for the worker identified by ``handle``."""
        ...

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send text input to the backend worker."""
        ...

    def wait_idle(
        self,
        handle: str,
        idle_time: float = 2.0,
        timeout: int | None = None,
    ) -> bool:
        """Wait until the backend worker becomes idle."""
        ...

    def execute_in_pane(
        self,
        handle: str,
        command: str,
        timeout: int = 30,
    ) -> CaptureResult:
        """Execute a shell command inside the backend worker context."""
        ...


_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

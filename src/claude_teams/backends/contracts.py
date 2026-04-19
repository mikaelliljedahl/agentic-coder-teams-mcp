"""Shared backend contracts and request/result types."""

import re
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable


class CaptureResult(TypedDict):
    """Result of executing a command in a tmux pane."""

    output: str
    exit_code: int


@dataclass(frozen=True)
class ReasoningEffortSpec:
    """Describes how a backend accepts a reasoning-effort selection.

    Each supporting backend returns a spec that captures both the CLI flag
    syntax used to set effort and the fixed enum of accepted values. Effort
    semantics differ across backends: ``claude`` has a dedicated ``--effort``
    flag, ``codex``/``coder`` set it via a ``-c`` override, and ``amp``/
    ``kimi`` express effort through the ``-m`` mode/model selector. Holding
    both parts in one frozen spec keeps the per-backend differences discoverable
    to callers without leaking shape decisions into shared code.

    Attributes:
        flag: CLI flag name (e.g. ``"--effort"``, ``"-c"``, ``"-m"``).
        value_template: How the value is rendered into the arg following
            ``flag`` — ``"{value}"`` for flags that take the raw value, or a
            ``"key={value}"`` form for configuration-style overrides.
        options: Fixed enum of accepted effort values for this backend.

    """

    flag: str
    value_template: str
    options: frozenset[str]

    def build_args(self, value: str) -> list[str]:
        """Render the flag/value pair as argv entries.

        Args:
            value: Effort value (already validated against ``options``).

        Returns:
            Two-element list suitable for splicing into a command vector.

        """
        return [self.flag, self.value_template.format(value=value)]


@dataclass(frozen=True)
class AgentProfile:
    """A discovered agent/persona definition a backend can select at launch.

    ``name`` is the identifier the user passes (matches the flag value for
    CLIs that take a name, or the stem of the discovered file when selection
    is path-based). ``path`` is the filesystem location of the underlying
    definition — needed when a backend's CLI accepts a path rather than a
    name (e.g. ``goose --recipe <path>``) or composes both (e.g. Codex's
    ``-c 'agents.<name>.config_file="<path>"'`` override).

    Attributes:
        name: Stable identifier used to select the profile.
        path: Absolute filesystem path to the profile definition.

    """

    name: str
    path: str


@dataclass(frozen=True)
class AgentSelectSpec:
    """Describes how a backend accepts an agent/profile selection.

    Supporting backends return a spec that captures the CLI flag and how
    to render the value. ``value_template`` may reference ``{name}`` and/or
    ``{path}`` placeholders — ``claude``/``claudish`` use the bare name,
    ``goose`` uses the bare path, and ``codex``/``coder`` use both in a
    compound ``-c`` override. Discovery of the actual profiles is handled
    separately via ``Backend.discover_agents`` since mechanisms vary widely
    (dir scans, TOML parsing, env-var path lists).

    Attributes:
        flag: CLI flag name (e.g. ``"--agent"``, ``"-c"``, ``"--recipe"``).
        value_template: How the value is rendered into the arg following
            ``flag``. May contain ``{name}`` and/or ``{path}`` placeholders.

    """

    flag: str
    value_template: str

    def build_args(self, profile: AgentProfile) -> list[str]:
        """Render the flag/value pair as argv entries for a given profile.

        Args:
            profile: Discovered profile to inject.

        Returns:
            Two-element list suitable for splicing into a command vector.

        """
        return [
            self.flag,
            self.value_template.format(name=profile.name, path=profile.path),
        ]


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
    reasoning_effort: str | None = None
    agent_profile: str | None = None
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

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Return the backend's reasoning-effort spec, or ``None`` if unsupported."""
        ...

    def agent_select_spec(self) -> AgentSelectSpec | None:
        """Return the backend's agent-selection spec, or ``None`` if unsupported."""
        ...

    def discover_agents(self, cwd: str) -> list[AgentProfile]:
        """Enumerate agent profiles this backend can launch from ``cwd``."""
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

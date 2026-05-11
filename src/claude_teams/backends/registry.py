"""Backend discovery and registry management."""

import importlib
import logging
from collections.abc import Iterator

from claude_teams.backends.base import Backend
from claude_teams.backends.contracts import BackendNotRegisteredError, NoBackendsAvailableError

logger = logging.getLogger(__name__)

_BUILTIN_BACKENDS: dict[str, str] = {
    "claude-code": "claude_teams.backends.claude_code.ClaudeCodeBackend",
    "codex": "claude_teams.backends.codex.CodexBackend",
}

class BackendRegistry:
    """Discovers and manages available spawner backends.

    Backends are loaded lazily on first access. Built-in backends are registered
    if their binary is found on PATH. Third-party backends can register via
    entry points or manual ``register()`` calls.
    """

    def __init__(self) -> None:
        """Initialize an empty lazy-loading backend registry."""
        self._backends: dict[str, Backend] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazily load built-in and entry-point backends."""
        if self._loaded:
            return
        self._loaded = True

        for name, dotted_path in _BUILTIN_BACKENDS.items():
            try:
                module_path, class_name = dotted_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                instance = cls()
                if instance.is_available():
                    self._backends[name] = instance
                    logger.info("Registered built-in backend: %s", name)
                else:
                    logger.debug("Backend %s not available (binary not found)", name)
            except Exception:
                logger.debug("Failed to load backend %s", name, exc_info=True)

    def register(self, name: str, backend: Backend) -> None:
        """Manually register a backend instance.

        Args:
            name: Backend identifier.
            backend: Backend instance satisfying the Backend protocol.

        """
        self._backends[name] = backend

    def get(self, name: str) -> Backend:
        """Get a backend by name.

        Args:
            name: Backend identifier.

        Returns:
            The Backend instance.

        Raises:
            BackendNotRegisteredError: If no backend with the given name is
                registered.

        """
        self._ensure_loaded()
        if name not in self._backends:
            raise BackendNotRegisteredError(name, self._backends.keys())
        return self._backends[name]

    def list_available(self) -> list[str]:
        """Return sorted names of all available backends.

        Returns:
            list[str]: Sorted list of registered backend names.

        """
        self._ensure_loaded()
        return sorted(self._backends.keys())

    def default_backend(self) -> str:
        """Return the name of the default backend.

        Returns:
            'claude-code' if available, else the first available backend.

        Raises:
            NoBackendsAvailableError: If no backends are available.

        """
        self._ensure_loaded()
        if "claude-code" in self._backends:
            return "claude-code"
        available = self.list_available()
        if available:
            return available[0]
        raise NoBackendsAvailableError()

    def __iter__(self) -> Iterator[tuple[str, Backend]]:
        """Yield (name, backend) tuples for all registered backends."""
        self._ensure_loaded()
        yield from self._backends.items()


registry = BackendRegistry()

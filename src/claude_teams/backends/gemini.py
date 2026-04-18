"""Gemini backend integration."""

from typing import ClassVar

from claude_teams.backends.base import BaseBackend, SpawnRequest


class GeminiBackend(BaseBackend):
    """Backend adapter for Google Gemini CLI."""

    _name = "gemini"
    _binary_name = "gemini"

    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "gemini-2.5-flash",
        "balanced": "gemini-2.5-pro",
        "powerful": "gemini-2.5-pro",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
    }

    def supported_models(self) -> list[str]:
        """Return supported Gemini model names.

        Returns:
            list[str]: Curated list of supported model identifiers.

        """
        return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

    def default_model(self) -> str:
        """Return the default Gemini model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "gemini-2.5-flash"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Gemini model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Gemini model identifier.

        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def default_permission_args(self) -> list[str]:
        """Return default permission-bypass arguments for Gemini."""
        return ["--yolo"]

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Gemini CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.

        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "--prompt",
            request.prompt,
            "--model",
            model,
            *self.permission_args(request),
        ]

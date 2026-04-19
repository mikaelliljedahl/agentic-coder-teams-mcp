"""Kimi backend integration."""

from typing import ClassVar

from claude_teams.backends.base import BaseBackend, ReasoningEffortSpec, SpawnRequest


class KimiBackend(BaseBackend):
    """Backend adapter for Kimi Code CLI (Moonshot AI)."""

    _name = "kimi"
    _binary_name = "kimi"

    _MODEL_MAP: ClassVar[dict[str, str]] = {
        "fast": "kimi-k2",
        "balanced": "kimi-k2-thinking",
        "powerful": "kimi-k2-thinking-turbo",
    }

    _REASONING_EFFORT_SPEC: ClassVar[ReasoningEffortSpec] = ReasoningEffortSpec(
        flag="-m",
        value_template="{value}",
        options=frozenset({"kimi-k2", "kimi-k2-thinking", "kimi-k2-thinking-turbo"}),
    )

    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        """Kimi expresses effort through the ``-m`` model variant selector.

        Setting ``reasoning_effort`` overrides whatever ``model`` would resolve
        to — on Kimi the thinking variants ARE the effort ladder.
        """
        return self._REASONING_EFFORT_SPEC

    def supported_models(self) -> list[str]:
        """Return supported Kimi model names.

        Kimi supports any model via ``-m``; these are the well-known ones.

        Returns:
            list[str]: Curated list of supported model identifiers.

        """
        return [
            "kimi-k2",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        ]

    def default_model(self) -> str:
        """Return the default Kimi model.

        Returns:
            str: Default model identifier for this backend.

        """
        return "kimi-k2-thinking"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Kimi model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Kimi model identifier.

        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Kimi CLI command for non-interactive execution.

        Uses ``--print`` which activates non-interactive mode and implicitly
        enables ``--yolo`` (auto-approve all actions).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.

        """
        binary = self.discover_binary()
        model = request.reasoning_effort or self.resolve_model(request.model)
        return [
            binary,
            "--print",
            "-p",
            request.prompt,
            "-m",
            model,
        ]

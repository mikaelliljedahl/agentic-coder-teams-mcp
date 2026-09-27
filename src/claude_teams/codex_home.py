"""Resolve the Codex home used by child launches and rollout readers."""

import os
from pathlib import Path


def codex_home() -> Path:
    """Use the server's cwd for a relative CODEX_HOME, as child launches do."""
    configured = os.environ.get("CODEX_HOME")
    return Path(configured).resolve() if configured else Path.home() / ".codex"

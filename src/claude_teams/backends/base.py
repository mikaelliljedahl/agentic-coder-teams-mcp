"""Compatibility exports for backend base types."""

import claude_teams.backends.tmux_base as tmux_base_module
from claude_teams.backends.contracts import (
    AgentProfile,
    AgentSelectSpec,
    Backend,
    CaptureResult,
    HealthStatus,
    ReasoningEffortSpec,
    SpawnRequest,
    SpawnResult,
)
from claude_teams.backends.tmux_base import BaseBackend

shutil = tmux_base_module.shutil

__all__ = [
    "AgentProfile",
    "AgentSelectSpec",
    "Backend",
    "BaseBackend",
    "CaptureResult",
    "HealthStatus",
    "ReasoningEffortSpec",
    "SpawnRequest",
    "SpawnResult",
    "shutil",
]

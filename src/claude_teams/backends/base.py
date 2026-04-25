"""Compatibility exports for backend base types."""

import shutil

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
from claude_teams.backends.process_base import BaseBackend

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

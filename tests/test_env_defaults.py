"""Feature H / Slice H.2 — env-var precedence tests.

Locks the ``direct → CLAUDE_TEAMS_*`` precedence chain for every dimension
catalogued in ``CONFIG_PARITY_AUDIT.md``. Coverage is organized by resolver
so failures point directly at the layer that regressed:

- ``TestResolveSpawnCwd`` / ``TestResolveCapability`` / ``TestResolveDescription``
  / ``TestResolveBackendName`` cover the scalar resolvers.
- ``TestParseBoolEnv`` covers the shared bool-env parser.
- ``TestApplySpawnEnvDefaults`` covers the unified ``SpawnOptions`` resolver,
  including the "direct wins over env" and "env wins over template" ordering.
- ``TestMcpIntegration`` exercises the env layer through the MCP tool surface
  so a missed wire-up in a tool body fails here (not just in a unit test).
"""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

from claude_teams import teams
from claude_teams.backends import registry
from claude_teams.errors import (
    CwdNotAbsoluteError,
    InvalidEnvVarValueError,
    InvalidPermissionModeError,
)
from claude_teams.models import SpawnOptions
from claude_teams.server_runtime import (
    _parse_bool_env,
    _resolve_backend_name,
    _resolve_capability,
    _resolve_description,
    _resolve_spawn_cwd,
    apply_spawn_env_defaults,
)
from tests._server_support import _data

# ---------------------------------------------------------------------------
# Scalar resolvers
# ---------------------------------------------------------------------------


class TestResolveSpawnCwd:
    """``_resolve_spawn_cwd`` — direct → env → ``Path.cwd()``."""

    def test_direct_value_takes_precedence_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(other))
        # Caller passed an explicit value — env is ignored even when set.
        assert _resolve_spawn_cwd(str(tmp_path)) == str(tmp_path)

    def test_env_used_when_direct_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        assert _resolve_spawn_cwd("") == str(tmp_path)

    def test_falls_back_to_process_cwd_when_neither_set(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_CWD", raising=False)
        # Whatever Path.cwd() returns is fine — the contract is "non-empty,
        # current working directory" and not a specific value.
        assert _resolve_spawn_cwd("") == str(Path.cwd())

    def test_env_value_still_validated(self, monkeypatch: pytest.MonkeyPatch):
        # Relative paths are rejected regardless of whether they came from
        # the direct arg or the env — env should not be a bypass route for
        # validation the direct arg also fails.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", "not/absolute")
        with pytest.raises(CwdNotAbsoluteError):
            _resolve_spawn_cwd("")


class TestResolveCapability:
    """``_resolve_capability`` — direct → ``CLAUDE_TEAMS_CAPABILITY`` → ``""``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_CAPABILITY", "from-env")
        assert _resolve_capability("from-direct") == "from-direct"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_CAPABILITY", "from-env")
        assert _resolve_capability("") == "from-env"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_CAPABILITY", raising=False)
        assert _resolve_capability("") == ""


class TestResolveDescription:
    """``_resolve_description`` — direct → ``CLAUDE_TEAMS_DEFAULT_DESCRIPTION``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "env desc")
        assert _resolve_description("direct desc") == "direct desc"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "env desc")
        assert _resolve_description("") == "env desc"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", raising=False)
        assert _resolve_description("") == ""


class TestResolveBackendName:
    """``_resolve_backend_name`` — direct → ``CLAUDE_TEAMS_DEFAULT_BACKEND``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        assert _resolve_backend_name("claude-code") == "claude-code"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        assert _resolve_backend_name("") == "codex"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_BACKEND", raising=False)
        assert _resolve_backend_name("") == ""


# ---------------------------------------------------------------------------
# Bool-env parser
# ---------------------------------------------------------------------------


class TestParseBoolEnv:
    """Shared truthy/falsy coercion used by the bool-typed env vars."""

    @pytest.mark.parametrize(
        "raw", ["true", "True", "TRUE", " true ", "1", "yes", "YES", "on"]
    )
    def test_truthy_values(self, raw: str):
        assert _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw) is True

    @pytest.mark.parametrize(
        "raw", ["false", "False", "FALSE", " false ", "0", "no", "NO", "off"]
    )
    def test_falsy_values(self, raw: str):
        assert _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw) is False

    def test_unrecognized_raises_typed_error(self):
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "maybe")
        message = str(exc_info.value)
        # Error carries the env-var name and the offending value so an
        # operator can grep logs for the exact misconfiguration.
        assert "CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED" in message
        assert "'maybe'" in message


# ---------------------------------------------------------------------------
# Unified SpawnOptions resolver
# ---------------------------------------------------------------------------


class TestApplySpawnEnvDefaults:
    """``apply_spawn_env_defaults`` — per-field SpawnOptions env fallback."""

    def test_no_env_returns_same_object(self, monkeypatch: pytest.MonkeyPatch):
        # When no env overrides apply, the resolver short-circuits and
        # returns the same object — avoids a wasted ``model_copy``.
        for key in (
            "CLAUDE_TEAMS_DEFAULT_CWD",
            "CLAUDE_TEAMS_DEFAULT_MODEL",
            "CLAUDE_TEAMS_DEFAULT_BACKEND",
            "CLAUDE_TEAMS_DEFAULT_SUBAGENT_TYPE",
            "CLAUDE_TEAMS_DEFAULT_REASONING_EFFORT",
            "CLAUDE_TEAMS_DEFAULT_AGENT_PROFILE",
            "CLAUDE_TEAMS_DEFAULT_TEMPLATE",
            "CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED",
            "CLAUDE_TEAMS_PERMISSION_MODE",
            "CLAUDE_TEAMS_CAPABILITY",
        ):
            monkeypatch.delenv(key, raising=False)
        opts = SpawnOptions()
        assert apply_spawn_env_defaults(opts) is opts

    @pytest.mark.parametrize(
        ("env_var", "field", "value"),
        [
            ("CLAUDE_TEAMS_DEFAULT_MODEL", "model", "powerful"),
            ("CLAUDE_TEAMS_DEFAULT_BACKEND", "backend", "codex"),
            ("CLAUDE_TEAMS_DEFAULT_SUBAGENT_TYPE", "subagent_type", "code-reviewer"),
            ("CLAUDE_TEAMS_CAPABILITY", "capability", "cap-from-env"),
            ("CLAUDE_TEAMS_DEFAULT_REASONING_EFFORT", "reasoning_effort", "high"),
            ("CLAUDE_TEAMS_DEFAULT_AGENT_PROFILE", "agent_profile", "reviewer"),
            ("CLAUDE_TEAMS_DEFAULT_TEMPLATE", "template", "executor"),
        ],
    )
    def test_env_fills_unset_string_fields(
        self,
        env_var: str,
        field: str,
        value: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(env_var, value)
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert getattr(resolved, field) == value
        # Env-filled fields enter ``model_fields_set`` so the downstream
        # template layer treats them as explicit.
        assert field in resolved.model_fields_set

    def test_cwd_env_fills_unset_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.cwd == str(tmp_path)

    def test_direct_value_wins_over_env(self, monkeypatch: pytest.MonkeyPatch):
        # Explicit caller value must survive even when every env var is set
        # — "direct" is the top of the precedence chain.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        opts = SpawnOptions(model="fast", backend="claude-code")
        resolved = apply_spawn_env_defaults(opts)
        assert resolved.model == "fast"
        assert resolved.backend == "claude-code"

    def test_permission_mode_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_PERMISSION_MODE", "require_approval")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.permission_mode == "require_approval"

    def test_invalid_permission_mode_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_PERMISSION_MODE", "nonsense-mode")
        # Bad env value surfaces the same typed error as a bad direct arg —
        # operators see a consistent contract regardless of input surface.
        with pytest.raises(InvalidPermissionModeError):
            apply_spawn_env_defaults(SpawnOptions())

    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
    def test_plan_mode_required_env(
        self, raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw)
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.plan_mode_required is expected

    def test_invalid_plan_mode_required_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "nope")
        with pytest.raises(InvalidEnvVarValueError):
            apply_spawn_env_defaults(SpawnOptions())

    def test_empty_env_value_does_not_override(self, monkeypatch: pytest.MonkeyPatch):
        # An env var set to "" is the same as absence — the env layer must
        # not clobber the pydantic default with an empty string. Prevents
        # a shell's ``export CLAUDE_TEAMS_DEFAULT_MODEL=`` from silently
        # stripping the "balanced" default.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.model == "balanced"

    def test_multiple_env_vars_compose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Setting several env vars at once must fill every corresponding
        # unset field in one pass — not the first one encountered.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "true")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.cwd == str(tmp_path)
        assert resolved.model == "powerful"
        assert resolved.backend == "codex"
        assert resolved.plan_mode_required is True


# ---------------------------------------------------------------------------
# Integration: env fallback through the MCP tool surface
# ---------------------------------------------------------------------------


class TestMcpIntegration:
    """End-to-end: env fallback reaches the backend call through MCP tools.

    Unit tests above prove the resolvers work in isolation; these tests
    prove the wrappers actually invoke them. A future refactor that removes
    ``apply_spawn_env_defaults`` from a dep factory, or a tool body that
    forgets to resolve its scalar, fails here rather than silently dropping
    the env fallback in production.
    """

    async def test_spawn_teammate_picks_up_env_model_default(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        mock = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool("team_create", {"team_name": "env-spawn-team"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "env-spawn-team",
                "name": "worker",
                "prompt": "do a thing",
            },
        )

        spawn_request = mock.spawn.call_args_list[0].args[0]
        # ``"powerful"`` resolves to ``"opus"`` via the mock backend's map,
        # so the env default actually flowed all the way to the backend.
        assert spawn_request.model == "opus"

    async def test_spawn_teammate_direct_arg_still_wins_over_env(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        mock = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool("team_create", {"team_name": "env-direct-win"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "env-direct-win",
                "name": "worker",
                "prompt": "do a thing",
                "options": {"model": "fast"},
            },
        )

        spawn_request = mock.spawn.call_args_list[0].args[0]
        assert spawn_request.model == "haiku"

    async def test_team_create_description_env_fallback(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            "CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "Baseline description from env"
        )

        await client.call_tool("team_create", {"team_name": "env-desc-team"})

        cfg = await teams.read_config("env-desc-team")
        assert cfg.description == "Baseline description from env"

    async def test_team_create_direct_description_wins(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "from env")

        await client.call_tool(
            "team_create",
            {"team_name": "env-desc-direct", "description": "from direct arg"},
        )

        cfg = await teams.read_config("env-desc-direct")
        assert cfg.description == "from direct arg"

    async def test_list_agents_backend_env_fallback(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "claude-code")

        result = _data(await client.call_tool("list_agents", {"backend_name": ""}))

        assert result["backend"] == "claude-code"

    async def test_list_agents_cwd_env_fallback(
        self,
        client: Client,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))

        result = _data(
            await client.call_tool("list_agents", {"backend_name": "claude-code"})
        )

        assert result["cwd"] == str(tmp_path)

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.amp import AmpBackend
from claude_teams.backends.base import SpawnRequest


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="smart",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestAmpProperties:
    def test_name_is_amp(self):
        backend = AmpBackend()
        assert backend.name == "amp"

    def test_binary_name_is_amp_cli(self):
        backend = AmpBackend()
        assert backend.binary_name == "amp-cli"


class TestAmpSupportedModels:
    def test_returns_expected_modes(self):
        backend = AmpBackend()
        models = backend.supported_models()
        assert "rush" in models
        assert "smart" in models
        assert "free" in models

    def test_returns_list(self):
        backend = AmpBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestAmpDefaultModel:
    def test_returns_smart(self):
        backend = AmpBackend()
        assert backend.default_model() == "smart"


class TestAmpResolveModel:
    def test_resolves_fast_to_rush(self):
        backend = AmpBackend()
        assert backend.resolve_model("fast") == "rush"

    def test_resolves_balanced_to_smart(self):
        backend = AmpBackend()
        assert backend.resolve_model("balanced") == "smart"

    def test_resolves_powerful_to_smart(self):
        backend = AmpBackend()
        assert backend.resolve_model("powerful") == "smart"

    def test_passes_through_direct_mode(self):
        backend = AmpBackend()
        assert backend.resolve_model("free") == "free"

    def test_passes_through_unknown_name(self):
        backend = AmpBackend()
        assert backend.resolve_model("custom") == "custom"


class TestAmpBuildCommand:
    def test_produces_execute_command(self, _make_request):
        backend = AmpBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/amp-cli"
        assert "-x" in cmd
        assert "--dangerously-allow-all" in cmd

    def test_includes_prompt_after_x_flag(self, _make_request):
        backend = AmpBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-x")
        assert cmd[idx + 1] == "fix the bug"

    def test_includes_mode_flag_for_known_mode(self, _make_request):
        backend = AmpBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "rush"

    def test_omits_mode_flag_for_unknown_model(self, _make_request):
        backend = AmpBackend()
        request = _make_request(model="some-unknown-model")

        cmd = backend.build_command(request)

        assert "-m" not in cmd


class TestAmpBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = AmpBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestAmpAvailability:
    def test_available_when_binary_found(self):
        backend = AmpBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = AmpBackend()
        assert backend.is_available() is False


class TestAmpReasoningEffort:
    def test_spec_advertises_m_flag_and_mode_options(self):
        backend = AmpBackend()
        spec = backend.reasoning_effort_spec()
        assert spec is not None
        assert spec.flag == "-m"
        assert spec.options == frozenset({"free", "rush", "smart"})

    def test_effort_overrides_resolved_model_on_m_flag(self, _make_request):
        backend = AmpBackend()
        request = _make_request(model="fast", reasoning_effort="free")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "free"

    def test_without_effort_falls_back_to_resolved_model(self, _make_request):
        backend = AmpBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "rush"

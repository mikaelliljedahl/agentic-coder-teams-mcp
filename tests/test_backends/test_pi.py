"""Unit tests for the pi backend."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.agent_output import read_pi_output
from claude_teams.backends import pi as pi_module
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.pi import PiBackend

_ALL_MODELS = [
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
]


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="gpt-5.6-sol",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="team-lead",
        reasoning_effort="low",
        extra={"session_dir": str(tmp_path / "sess")},
    )

    def factory(**overrides: object) -> SpawnRequest:
        return replace(default, **overrides)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def _direct_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the direct node+cli.js launcher (no shim, no PATH dependency)."""
    monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["node", "cli.js"])


@pytest.fixture
def _tty(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    def apply(has_tty: bool) -> None:
        monkeypatch.setattr(
            pi_module.process_manager,
            "provides_tty",
            lambda *a, **k: has_tty,
        )

    apply(True)
    return apply


@pytest.fixture
def _models(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], None]:
    def apply(ids: list[str]) -> None:
        monkeypatch.setattr(
            pi_module, "_discover_pi_model_ids", lambda launcher: list(ids)
        )

    apply(_ALL_MODELS)
    return apply


class TestPiProperties:
    def test_name(self):
        assert PiBackend().name == "pi"

    def test_binary_name(self):
        assert PiBackend().binary_name == "pi"

    def test_is_interactive(self):
        assert PiBackend().is_interactive is True

    def test_supports_resume(self):
        assert PiBackend().supports_resume() is True

    def test_default_permission_trusts_project(self):
        assert PiBackend().default_permission_args() == ["-a"]


class TestPiModels:
    def test_supported_models_are_tiers(self):
        assert PiBackend().supported_models() == [
            "low",
            "medium",
            "high",
            "xhigh",
            "ultra",
        ]

    def test_default_model_is_medium(self):
        assert PiBackend().default_model() == "medium"

    def test_resolve_model_tier_to_slug(self):
        assert PiBackend().resolve_model("medium") == "gpt-5.6-sol"

    def test_resolve_model_passthrough(self):
        assert PiBackend().resolve_model("some-model") == "some-model"

    def test_resolve_model_blank(self):
        assert PiBackend().resolve_model("") == ""


class TestPiResolveLaunch:
    def test_tier_maps_to_model_and_thinking(self, _models):
        assert PiBackend().resolve_launch("medium", None) == ("gpt-5.6-sol", "low")
        assert PiBackend().resolve_launch("low", None) == ("gpt-5.6-terra", "medium")

    def test_tier_owns_thinking_ignoring_caller(self, _models):
        assert PiBackend().resolve_launch("high", "max") == ("gpt-5.6-sol", "medium")

    def test_blank_defers_to_pi_default(self):
        assert PiBackend().resolve_launch("", None) == ("", None)

    def test_soft_fallback_when_tier_model_absent(self, _models):
        # Logged into a provider without the gpt-5.6 catalog: keep the tier's
        # thinking level but drop the model so pi uses its own default.
        _models(["claude-sonnet-4", "claude-opus-4"])
        assert PiBackend().resolve_launch("medium", None) == ("", "low")

    def test_raw_slug_passthrough_when_available(self, _models):
        assert PiBackend().resolve_launch("gpt-5.5", "high") == ("gpt-5.5", "high")

    def test_raw_slug_dropped_when_unavailable(self, _models):
        assert PiBackend().resolve_launch("nope", "high") == ("", "high")

    def test_skips_validation_when_discovery_empty(self, _models):
        _models([])
        assert PiBackend().resolve_launch("medium", None) == ("gpt-5.6-sol", "low")


class TestPiBuildCommand:
    def test_interactive_when_tty(self, _make_request, _direct_launch, _tty, _models):
        cmd = PiBackend().build_command(_make_request())
        assert "-p" not in cmd
        assert "--mode" not in cmd

    def test_headless_when_no_tty(self, _make_request, _direct_launch, _tty, _models):
        _tty(False)
        cmd = PiBackend().build_command(_make_request())
        assert cmd[2:5] == ["-p", "--mode", "json"]

    def test_session_binding_flags(self, _make_request, _direct_launch, _tty, _models):
        cmd = PiBackend().build_command(_make_request())
        assert "--session-id" in cmd
        assert cmd[cmd.index("--session-id") + 1] == "worker"
        sdir = cmd[cmd.index("--session-dir") + 1]
        assert sdir.endswith(str(Path("pi-sessions") / "worker"))

    def test_model_and_thinking_args(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_command(_make_request(model="gpt-5.6-sol"))
        assert cmd[cmd.index("--model") + 1] == "openai-codex/gpt-5.6-sol"
        assert cmd[cmd.index("--thinking") + 1] == "low"

    def test_no_model_arg_when_blank(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_command(_make_request(model="", reasoning_effort="low"))
        assert "--model" not in cmd
        # thinking is provider-agnostic and kept on the default-model fallback
        assert cmd[cmd.index("--thinking") + 1] == "low"

    def test_extension_loaded(self, _make_request, _direct_launch, _tty, _models):
        req = _make_request(
            extra={
                "session_dir": "S",
                "pi_state_extension_path": r"C:\ext\wat-state",
            }
        )
        cmd = PiBackend().build_command(req)
        assert cmd[cmd.index("-e") + 1] == r"C:\ext\wat-state"

    def test_wake_extension_loaded_alongside_state(
        self, _make_request, _direct_launch, _tty, _models
    ):
        # A spawned Pi agent (worker or nested subagent-as-lead) must load BOTH
        # the state extension and the wake extension, each via its own -e.
        req = _make_request(
            extra={
                "session_dir": "S",
                "pi_state_extension_path": r"C:\ext\wat-state",
                "pi_wake_extension_path": r"C:\ext\wat-wake",
            }
        )
        cmd = PiBackend().build_command(req)
        e_values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
        assert r"C:\ext\wat-state" in e_values
        assert r"C:\ext\wat-wake" in e_values

    def test_wake_extension_absent_when_not_provided(
        self, _make_request, _direct_launch, _tty, _models
    ):
        req = _make_request(
            extra={"session_dir": "S", "pi_state_extension_path": r"C:\ext\wat-state"}
        )
        cmd = PiBackend().build_command(req)
        e_values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
        assert e_values == [r"C:\ext\wat-state"]

    def test_prompt_verbatim_last_arg(
        self, _make_request, _direct_launch, _tty, _models
    ):
        req = _make_request(prompt="line1\nline2 < > | & ^ ( )")
        cmd = PiBackend().build_command(req)
        assert cmd[-1] == "line1\nline2 < > | & ^ ( )"

    def test_require_approval_drops_trust_flag(
        self, _make_request, _direct_launch, _tty, _models
    ):
        req = _make_request(permission_mode="require_approval")
        cmd = PiBackend().build_command(req)
        assert "-a" not in cmd

    def test_shim_fallback_uses_file_include(
        self, _make_request, monkeypatch, _tty, _models
    ):
        monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["pi.cmd"])
        req = _make_request(
            prompt="multi\nline",
            extra={"session_dir": "S", "prompt_file_path": r"C:\p\worker.txt"},
        )
        cmd = PiBackend().build_command(req)
        assert cmd[-2] == r"@C:\p\worker.txt"
        assert "multi\nline" not in cmd


class TestPiBuildResume:
    def test_resume_adds_continue(self, _make_request, _direct_launch, _tty, _models):
        cmd = PiBackend().build_resume_command(_make_request(), "sid-abc")
        assert "--continue" in cmd
        assert cmd[cmd.index("--session-id") + 1] == "worker"


class TestPiBuildEnv:
    def test_identity_and_state_dir(self, _make_request):
        env = PiBackend().build_env(_make_request(extra={"session_dir": r"C:\sess\S1"}))
        assert env["AGENT_NAME"] == "worker"
        assert env["AGENT_SESSION_ID"] == "team"
        assert env["AGENT_PARENT_NAME"] == "team-lead"
        assert env["WIN_AGENT_TEAMS_SESSION_DIR"] == r"C:\sess\S1"


def _write_pi_session(directory: Path, session_id: str, assistant_text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"2026-07-11T19-20-18-932Z_{session_id}.jsonl"
    lines = [
        {"type": "session", "version": 3, "id": session_id, "cwd": "C:/x"},
        {"type": "model_change", "id": "a1"},
        {
            "type": "message",
            "id": "b2",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        },
        {
            "type": "message",
            "id": "c3",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


class TestReadPiOutput:
    def test_reads_header_id_and_last_assistant_text(self, tmp_path: Path):
        _write_pi_session(tmp_path, "worker", "the answer is 42")
        out = read_pi_output(str(tmp_path))
        assert out is not None
        assert out.backend_session_id == "worker"
        assert out.last_message == "the answer is 42"

    def test_prefers_file_matching_expected_id(self, tmp_path: Path):
        _write_pi_session(tmp_path / "a", "worker", "correct")
        # A stray session file in the same dir with a different id.
        _write_pi_session(tmp_path / "a", "other", "wrong")
        out = read_pi_output(str(tmp_path / "a"), expected_session_id="worker")
        assert out is not None
        assert out.last_message == "correct"

    def test_missing_dir_returns_none(self, tmp_path: Path):
        assert read_pi_output(str(tmp_path / "nope")) is None

    def test_no_jsonl_returns_none(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        assert read_pi_output(str(tmp_path / "empty")) is None

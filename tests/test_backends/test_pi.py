"""Unit tests for the pi backend."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.agent_output import read_pi_output
from claude_teams.backends import pi as pi_module
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import PREFER_LUNA_ENV
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


@pytest.fixture(autouse=True)
def _default_tier_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test to the default tier ladder unless it opts in itself."""
    monkeypatch.delenv(PREFER_LUNA_ENV, raising=False)


@pytest.fixture
def _prefer_luna(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Return a helper that sets the Luna-preferring opt-in env var."""

    def apply(value: str = "1") -> None:
        monkeypatch.setenv(PREFER_LUNA_ENV, value)

    return apply


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
            "cheapest",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]

    def test_default_model_is_medium(self):
        assert PiBackend().default_model() == "medium"

    def test_resolve_model_tier_to_slug(self):
        backend = PiBackend()
        assert backend.resolve_model("cheapest") == "gpt-5.6-luna"
        assert backend.resolve_model("low") == "gpt-5.6-luna"
        assert backend.resolve_model("medium") == "gpt-5.6-luna"
        assert backend.resolve_model("high") == "gpt-5.6-sol"
        assert backend.resolve_model("max") == "gpt-5.6-sol"

    def test_resolve_model_passthrough(self):
        assert PiBackend().resolve_model("some-model") == "some-model"

    def test_resolve_model_blank(self):
        assert PiBackend().resolve_model("") == ""


class TestPiResolveLaunch:
    def test_tier_maps_to_model_and_thinking(self, _models):
        backend = PiBackend()
        assert backend.resolve_launch("cheapest", None) == (
            "gpt-5.6-luna",
            "medium",
        )
        assert backend.resolve_launch("low", None) == ("gpt-5.6-luna", "high")
        assert backend.resolve_launch("medium", None) == ("gpt-5.6-luna", "xhigh")
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("xhigh", None) == ("gpt-5.6-sol", "high")
        assert backend.resolve_launch("max", None) == ("gpt-5.6-sol", "xhigh")

    def test_old_ultra_name_uses_raw_slug_behavior(self, _models):
        _models(["ultra"])
        assert PiBackend().resolve_launch("ultra", None) == ("ultra", None)

    def test_tier_owns_thinking_ignoring_caller(self, _models):
        assert PiBackend().resolve_launch("high", "max") == ("gpt-5.6-sol", "medium")

    def test_blank_defers_to_pi_default(self):
        assert PiBackend().resolve_launch("", None) == ("", None)

    def test_soft_fallback_when_tier_model_absent(self, _models):
        # Logged into a provider without the gpt-5.6 catalog: keep the tier's
        # thinking level but drop the model so pi uses its own default.
        _models(["claude-sonnet-4", "claude-opus-4"])
        assert PiBackend().resolve_launch("medium", None) == ("", "xhigh")

    def test_partial_catalog_drops_model_rather_than_substituting(self, _models):
        # Sol present but Luna absent: the Luna-backed tiers do NOT silently
        # fall back to Sol (pi has no model-substitution rule) — they drop the
        # model and keep the tier's thinking level, while the Sol-backed tiers
        # are unaffected.
        _models(["gpt-5.6-sol"])
        backend = PiBackend()
        assert backend.resolve_launch("low", None) == ("", "high")
        assert backend.resolve_launch("medium", None) == ("", "xhigh")
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("xhigh", None) == ("gpt-5.6-sol", "high")
        assert backend.resolve_launch("max", None) == ("gpt-5.6-sol", "xhigh")

    def test_raw_slug_passthrough_when_available(self, _models):
        assert PiBackend().resolve_launch("gpt-5.5", "high") == ("gpt-5.5", "high")


class TestPiPreferLunaTiers:
    """Pi mirrors the Codex ladder, opt-in included, so a tier means the same
    thing on both backends."""

    def test_top_tiers_shift_toward_luna(self, _models, _prefer_luna):
        _prefer_luna()
        backend = PiBackend()
        assert backend.resolve_launch("high", None) == ("gpt-5.6-luna", "max")
        assert backend.resolve_launch("xhigh", None) == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("max", None) == ("gpt-5.6-sol", "high")

    def test_cheap_tiers_unchanged(self, _models, _prefer_luna):
        _prefer_luna()
        backend = PiBackend()
        assert backend.resolve_launch("cheapest", None) == ("gpt-5.6-luna", "medium")
        assert backend.resolve_launch("low", None) == ("gpt-5.6-luna", "high")
        assert backend.resolve_launch("medium", None) == ("gpt-5.6-luna", "xhigh")

    def test_matches_codex_ladder(self, _models, _prefer_luna):
        # The two backends must not diverge: a coordinator picks a tier without
        # knowing which backend will run it.
        _prefer_luna()
        assert PiBackend()._tier_launch() == CodexBackend()._tier_launch()

    def test_resolve_model_follows_opt_in(self, _prefer_luna):
        _prefer_luna()
        backend = PiBackend()
        assert backend.resolve_model("high") == "gpt-5.6-luna"
        assert backend.resolve_model("xhigh") == "gpt-5.6-sol"
        assert backend.resolve_model("max") == "gpt-5.6-sol"

    def test_tier_names_and_order_unchanged(self, _prefer_luna):
        _prefer_luna()
        backend = PiBackend()
        assert backend.supported_models() == [
            "cheapest",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert backend.default_model() == "medium"

    @pytest.mark.parametrize("value", ["0", "", "true", "2"])
    def test_only_exactly_one_opts_in(self, _models, _prefer_luna, value):
        _prefer_luna(value)
        assert PiBackend().resolve_launch("high", None) == ("gpt-5.6-sol", "medium")

    def test_soft_fallback_still_applies_to_shifted_tier(self, _models, _prefer_luna):
        # ``high`` needs Luna under the opt-in; without it pi drops the model
        # and keeps the tier's thinking level rather than erroring.
        _models(["gpt-5.6-sol"])  # no Luna
        _prefer_luna()
        assert PiBackend().resolve_launch("high", None) == ("", "max")

    def test_read_per_call_not_at_import(self, _models, _prefer_luna):
        backend = PiBackend()
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")
        _prefer_luna()
        assert backend.resolve_launch("high", None) == ("gpt-5.6-luna", "max")

    def test_raw_slug_dropped_when_unavailable(self, _models):
        assert PiBackend().resolve_launch("nope", "high") == ("", "high")

    def test_skips_validation_when_discovery_empty(self, _models):
        _models([])
        assert PiBackend().resolve_launch("medium", None) == ("gpt-5.6-luna", "xhigh")


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

    def test_medium_tier_launch_reaches_argv(
        self, _make_request, _direct_launch, _tty, _models
    ):
        # The default tier's resolved (slug, thinking) must survive into argv:
        # tuple-level resolve_launch assertions alone don't prove that.
        backend = PiBackend()
        model, thinking = backend.resolve_launch("medium", None)
        cmd = backend.build_command(
            _make_request(model=model, reasoning_effort=thinking)
        )
        assert cmd[cmd.index("--model") + 1] == "openai-codex/gpt-5.6-luna"
        assert cmd[cmd.index("--thinking") + 1] == "xhigh"

    def test_cheapest_tier_launch_reaches_argv(
        self, _make_request, _direct_launch, _tty, _models
    ):
        backend = PiBackend()
        model, thinking = backend.resolve_launch("cheapest", None)
        cmd = backend.build_command(
            _make_request(model=model, reasoning_effort=thinking)
        )
        assert cmd[cmd.index("--model") + 1] == "openai-codex/gpt-5.6-luna"
        assert cmd[cmd.index("--thinking") + 1] == "medium"

    def test_medium_tier_fallback_keeps_thinking_without_model(
        self, _make_request, _direct_launch, _tty, _models
    ):
        _models(["gpt-5.6-sol"])  # partial catalog: Sol present, Luna absent
        backend = PiBackend()
        model, thinking = backend.resolve_launch("medium", None)
        cmd = backend.build_command(
            _make_request(model=model, reasoning_effort=thinking)
        )
        assert "--model" not in cmd
        assert cmd[cmd.index("--thinking") + 1] == "xhigh"

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


class TestPiInteractiveToolLockdown:
    """A spawned pi agent must never be able to block on a human question.

    On Windows we run pi in its TUI, so a user-global extension such as
    ``ask_user`` finds ``ctx.mode === "tui"`` and renders a prompt in a tab
    nobody is watching -- the agent then waits forever. Two layers guard it: a
    hard ``--exclude-tools`` deny list and a soft ``--append-system-prompt``
    escalation policy. Both must be on spawn *and* resume, since a resumed
    agent runs with a fresh argv.
    """

    def test_build_command_excludes_interactive_tools(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_command(_make_request())
        assert cmd[cmd.index("--exclude-tools") + 1] == (
            "ask_user,ask_question,ask_human,request_input"
        )

    def test_resume_command_excludes_interactive_tools(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_resume_command(_make_request(), "sid-abc")
        assert cmd[cmd.index("--exclude-tools") + 1] == (
            "ask_user,ask_question,ask_human,request_input"
        )

    def test_exclude_tools_env_override(
        self, _make_request, _direct_launch, _tty, _models, monkeypatch
    ):
        monkeypatch.setenv("WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS", "ask_user,my_tool")
        cmd = PiBackend().build_command(_make_request())
        assert cmd[cmd.index("--exclude-tools") + 1] == "ask_user,my_tool"

    def test_empty_exclude_tools_env_omits_flag(
        self, _make_request, _direct_launch, _tty, _models, monkeypatch
    ):
        # An explicit empty value is the documented debugging escape hatch: no
        # deny list at all, rather than the default one.
        monkeypatch.setenv("WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS", "")
        cmd = PiBackend().build_command(_make_request())
        assert "--exclude-tools" not in cmd

    def test_build_command_appends_escalation_policy(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_command(_make_request())
        policy = cmd[cmd.index("--append-system-prompt") + 1]
        assert "send_message" in policy
        assert "never" in policy.lower()

    def test_resume_command_appends_escalation_policy(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_resume_command(_make_request(), "sid-abc")
        policy = cmd[cmd.index("--append-system-prompt") + 1]
        assert "send_message" in policy

    def test_escalation_policy_stays_short(self):
        # Every pi turn pays for this text; keep it to ~3 lines.
        assert len(pi_module._ESCALATION_POLICY.splitlines()) <= 3


class TestPiPromptTransport:
    """Prompt hazards pi resolves from the *first* character or from length."""

    @pytest.mark.parametrize("lead", ["@", "/", "-"])
    def test_leading_character_is_guarded(
        self, _make_request, _direct_launch, _tty, _models, lead
    ):
        # ``@x`` is a CLI file include, ``/x`` an extension/skill command and
        # ``-x`` a flag -- all decided from the token's first character only, so
        # a leading newline defuses them without losing a byte of the prompt.
        cmd = PiBackend().build_command(_make_request(prompt=f"{lead}do stuff"))
        assert cmd[-1] == f"\n{lead}do stuff"

    def test_ordinary_prompt_is_not_guarded(
        self, _make_request, _direct_launch, _tty, _models
    ):
        cmd = PiBackend().build_command(_make_request(prompt="do stuff"))
        assert cmd[-1] == "do stuff"

    def test_oversize_prompt_uses_sidecar_on_headless_path(
        self, _make_request, _direct_launch, _tty, _models
    ):
        # Headless (``-p --mode json``) puts the prompt on a real command line,
        # which Windows rejects past ~32 KB; the sidecar is the escape.
        _tty(False)
        req = _make_request(
            prompt="x" * (pi_module.MAX_ARGV_PROMPT_CHARS + 1),
            extra={"session_dir": "S", "prompt_file_path": r"C:\p\worker.txt"},
        )
        cmd = PiBackend().build_command(req)
        assert cmd[-2] == r"@C:\p\worker.txt"

    def test_oversize_prompt_stays_in_argv_on_wrapper_path(
        self, _make_request, _direct_launch, _tty, _models
    ):
        # The TUI path bakes argv into a .ps1, so there is no command-line
        # ceiling and the verbatim prompt is preferable to pi's ``<file>`` wrap.
        prompt = "x" * (pi_module.MAX_ARGV_PROMPT_CHARS + 1)
        req = _make_request(
            prompt=prompt,
            extra={"session_dir": "S", "prompt_file_path": r"C:\p\worker.txt"},
        )
        cmd = PiBackend().build_command(req)
        assert cmd[-1] == prompt

    def test_shim_without_sidecar_warns_about_multiline_argv(
        self, _make_request, monkeypatch, _tty, _models, caplog
    ):
        # The shim routes argv through cmd.exe, which truncates at the first
        # newline. We cannot fix it here, but it must not fail silently.
        monkeypatch.setattr(PiBackend, "_launcher", lambda self: ["pi.cmd"])
        req = _make_request(prompt="multi\nline", extra={"session_dir": "S"})
        with caplog.at_level("WARNING"):
            cmd = PiBackend().build_command(req)
        assert cmd[-1] == "multi\nline"
        assert any("truncat" in rec.message.lower() for rec in caplog.records)


class TestPiBuildResume:
    def test_resume_uses_continue_without_session_id(
        self, _make_request, _direct_launch, _tty, _models
    ):
        # pi's CLI rejects ``--session-id`` combined with ``--continue`` and
        # exits 1 immediately. The per-agent ``--session-dir`` already scopes
        # resume to this agent's single session, so ``--continue`` alone is
        # unambiguous.
        cmd = PiBackend().build_resume_command(_make_request(), "sid-abc")
        assert "--continue" in cmd
        assert "--session-id" not in cmd
        # Resume still targets the per-agent session dir.
        sdir = cmd[cmd.index("--session-dir") + 1]
        assert sdir.endswith("pi-sessions/worker") or sdir.endswith(
            "pi-sessions\\worker"
        )


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

"""Plan §2.7 / test 12: children inherit the native flags as values, never implied."""

import itertools
import json
import tomllib

import pytest

from claude_teams import native_wake
from claude_teams import server_simple as ss
from claude_teams.agent_output import CORRELATION_FIELD
from claude_teams.backends import process_manager as process_manager_module
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_base import BaseBackend, process_manager

MASTER = "WIN_AGENT_TEAMS_NATIVE_WAKE"
DOWNSTREAM = "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM"
CLAUDE = "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE"
CODEX = "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX"
FLAGS = (MASTER, DOWNSTREAM, CLAUDE, CODEX)
IDENTITY_PREFIX = "mcp_servers.win-agent-teams.env="

# None = absent. Values are passed through as-is, never normalised to "1".
_SUB_VALUES = (None, "1", "0")
COMBOS = [
    dict(zip((DOWNSTREAM, CLAUDE, CODEX), values, strict=True))
    for values in itertools.product(_SUB_VALUES, repeat=3)
]


def _set_flags(monkeypatch, master, subs):
    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    if master is not None:
        monkeypatch.setenv(MASTER, master)
    for key, value in subs.items():
        if value is not None:
            monkeypatch.setenv(key, value)


def _expected(subs):
    expected = {MASTER: "1"}
    expected.update({k: v for k, v in subs.items() if v is not None})
    return expected


def _request(tmp_path, **extra):
    return SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="",
        agent_type="",
        color="",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
        extra={CORRELATION_FIELD: "corr-test", **extra},
    )


def _codex_env_table(cmd):
    token = next(arg for arg in cmd if arg.startswith(IDENTITY_PREFIX))
    return tomllib.loads("x=" + token.partition("=")[2])["x"]


@pytest.fixture(autouse=True)
def _no_launcher_env(monkeypatch):
    for key in process_manager_module._NESTED_LINUX_LAUNCHER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestPropagatedEnv:
    @pytest.mark.parametrize("subs", COMBOS)
    def test_master_on_passes_every_present_value(self, monkeypatch, subs):
        _set_flags(monkeypatch, "1", subs)
        assert native_wake.propagated_env() == _expected(subs)

    @pytest.mark.parametrize("master", [None, "0", "true", ""])
    def test_master_off_passes_nothing(self, monkeypatch, master):
        _set_flags(monkeypatch, master, {DOWNSTREAM: "1", CLAUDE: "1", CODEX: "1"})
        assert native_wake.propagated_env() == {}

    def test_effective_master_is_normalised(self, monkeypatch):
        _set_flags(monkeypatch, " 1 ", {})
        assert native_wake.propagated_env() == {MASTER: "1"}


class TestClaudeMcpConfig:
    @pytest.fixture
    def session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "_SESSION_BASE", tmp_path)
        (tmp_path / "sid" / "mcp").mkdir(parents=True)
        return "sid"

    def _env(self, session):
        path = ss._write_mcp_config(session, "worker", "team-lead")
        return json.loads(path.read_text(encoding="utf-8"))["mcpServers"][
            "win-agent-teams"
        ]["env"]

    @pytest.mark.parametrize("subs", COMBOS)
    def test_flag_on_writes_values(self, monkeypatch, session, subs):
        _set_flags(monkeypatch, "1", subs)
        env = self._env(session)
        assert {k: v for k, v in env.items() if k in FLAGS} == _expected(subs)

    @pytest.mark.parametrize("master", [None, "0", "true", ""])
    def test_flag_off_config_unchanged(self, monkeypatch, session, master):
        _set_flags(monkeypatch, None, {})
        baseline = self._env(session)
        _set_flags(monkeypatch, master, {DOWNSTREAM: "1", CLAUDE: "0", CODEX: "1"})
        assert self._env(session) == baseline
        assert baseline == {
            "AGENT_SESSION_ID": session,
            "AGENT_NAME": "worker",
            "AGENT_PARENT_NAME": "team-lead",
        }


class TestCodexOverride:
    @pytest.mark.parametrize("operation", ["spawn", "resume"])
    @pytest.mark.parametrize("nested_lead", [False, True])
    @pytest.mark.parametrize("subs", COMBOS)
    def test_flag_on_override_carries_values(
        self, monkeypatch, tmp_path, subs, operation, nested_lead
    ):
        _set_flags(monkeypatch, "1", subs)
        extra = {"enable_spawned_lead_wake": "1"} if nested_lead else {}
        request = _request(tmp_path, **extra)
        backend = CodexBackend()
        cmd = (
            backend.build_command(request)
            if operation == "spawn"
            else backend.build_resume_command(request, "thread-1")
        )
        table = _codex_env_table(cmd)
        assert {k: v for k, v in table.items() if k in FLAGS} == _expected(subs)
        # Identity and the single table override are preserved.
        assert table["AGENT_NAME"] == "worker"
        assert sum(arg.startswith(IDENTITY_PREFIX) for arg in cmd) == 1

    def test_odd_values_are_valid_toml(self, monkeypatch, tmp_path):
        _set_flags(monkeypatch, "1", {DOWNSTREAM: 'it\'s "x"\n'})
        table = _codex_env_table(CodexBackend().build_command(_request(tmp_path)))
        assert table[DOWNSTREAM] == 'it\'s "x"\n'

    @pytest.mark.parametrize("operation", ["spawn", "resume"])
    @pytest.mark.parametrize("master", ["0", "true", ""])
    def test_flag_off_argv_byte_identical(
        self, monkeypatch, tmp_path, master, operation
    ):
        def build():
            backend = CodexBackend()
            request = _request(tmp_path)
            if operation == "spawn":
                return backend.build_command(request)
            return backend.build_resume_command(request, "thread-1")

        _set_flags(monkeypatch, None, {})
        baseline = build()
        _set_flags(monkeypatch, master, {DOWNSTREAM: "1", CLAUDE: "1", CODEX: "1"})
        assert build() == baseline
        assert not set(_codex_env_table(baseline)) & set(FLAGS)


class TestProcessEnv:
    def _observe(self, monkeypatch, backend_cls, operation, tmp_path):
        instance = backend_cls()
        monkeypatch.setattr(instance, "build_env", lambda _: {"EXAMPLE": "value"})
        monkeypatch.setattr(instance, "build_command", lambda _: ["fake"])
        monkeypatch.setattr(instance, "build_resume_command", lambda *a: ["fake"])
        observed = []
        monkeypatch.setattr(
            process_manager,
            "spawn_process",
            lambda request, argv, env_vars, *a, **k: observed.append(env_vars),
        )
        request = _request(tmp_path)
        if operation == "spawn":
            BaseBackend.spawn(instance, request)
        else:
            BaseBackend.resume(instance, request, "thread")
        return observed[0]

    @pytest.mark.parametrize("backend_cls", [ClaudeCodeBackend, CodexBackend])
    @pytest.mark.parametrize("operation", ["spawn", "resume"])
    @pytest.mark.parametrize("subs", COMBOS)
    def test_flag_on_process_env(
        self, monkeypatch, tmp_path, backend_cls, operation, subs
    ):
        _set_flags(monkeypatch, "1", subs)
        monkeypatch.setenv("CLAUDE_CODE_MESSAGING_SOCKET", "/lead.sock")
        monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "lead-token")
        env = self._observe(monkeypatch, backend_cls, operation, tmp_path)
        assert env == {
            "EXAMPLE": "value",
            **_expected(subs),
            # The lead's host channel is still scrubbed.
            "CLAUDE_CODE_MESSAGING_SOCKET": "",
            "CLAUDE_CODE_MESSAGING_TOKEN": "",
        }

    @pytest.mark.parametrize("backend_cls", [ClaudeCodeBackend, CodexBackend])
    @pytest.mark.parametrize("operation", ["spawn", "resume"])
    @pytest.mark.parametrize("master", [None, "0", "true", ""])
    def test_flag_off_process_env_unchanged(
        self, monkeypatch, tmp_path, backend_cls, operation, master
    ):
        _set_flags(monkeypatch, master, {DOWNSTREAM: "1", CLAUDE: "1", CODEX: "1"})
        env = self._observe(monkeypatch, backend_cls, operation, tmp_path)
        assert env == {"EXAMPLE": "value"}

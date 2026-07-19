"""Tests for ready-to-run watch command discovery."""

import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from claude_teams import cli, server_simple
from claude_teams.cli import app


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _quiet_dir(tmp_path: Path) -> Path:
    quiet = tmp_path / "quiet path O'Brien"
    quiet.mkdir()
    return quiet


def _shell_probe(argv: list[str]) -> bool:
    try:
        return _run(argv).returncode == 0
    except OSError:
        return False


def test_watch_argv_starts_with_current_interpreter_module_and_watch() -> None:
    argv = server_simple._watch_argv("session")

    assert argv[:4] == [sys.executable, "-m", "claude_teams.cli", "watch"]


def test_watch_argv_keeps_session_dir_with_spaces_as_one_token() -> None:
    session_dir = "C:\\session root\\one session"

    argv = server_simple._watch_argv(session_dir)

    assert argv[4] == session_dir
    assert len(argv) == 5


def test_watch_argv_omits_timeout_when_none() -> None:
    argv = server_simple._watch_argv("session", timeout=None)

    assert "--timeout" not in argv


def test_watch_argv_appends_numeric_timeout() -> None:
    argv = server_simple._watch_argv("session", timeout=2.5)

    assert argv[-2:] == ["--timeout", "2.5"]


def test_watch_command_bash_round_trips_every_token() -> None:
    session_dir = "C:\\odd path\\O'Brien\\$`%!&\\café\\"
    argv = server_simple._watch_argv(session_dir, timeout=3)

    command = server_simple._watch_command_bash(session_dir, timeout=3)

    assert shlex.split(command) == argv


def test_watch_command_powershell_uses_call_operator_and_doubles_quotes() -> None:
    session_dir = "C:\\Users\\O'Brien\\s"

    command = server_simple._watch_command_powershell(session_dir)

    assert command.startswith("& ")
    assert "'C:\\Users\\O''Brien\\s'" in command
    assert command.count("'") % 2 == 0


def test_shell_renderings_differ_for_embedded_quote() -> None:
    session_dir = "C:\\Users\\O'Brien\\s"

    bash = server_simple._watch_command_bash(session_dir)
    powershell = server_simple._watch_command_powershell(session_dir)

    assert bash != powershell
    assert "O''Brien" in powershell


def test_watch_argv_executes_and_times_out_quietly(tmp_path: Path) -> None:
    result = _run(server_simple._watch_argv(tmp_path, timeout=1))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == ""


def test_watch_command_bash_executes_and_times_out_quietly(tmp_path: Path) -> None:
    bash = "bash"
    if not _shell_probe([bash, "-c", "exit 0"]):
        pytest.skip("bash -c is not usable")
    quiet = _quiet_dir(tmp_path)

    result = _run([bash, "-c", server_simple._watch_command_bash(quiet, timeout=1)])

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == ""


def test_watch_command_powershell_executes_and_times_out_quietly(
    tmp_path: Path,
) -> None:
    powershell = "powershell"
    if not _shell_probe([powershell, "-NoProfile", "-Command", "exit 0"]):
        pytest.skip("powershell -Command is not usable")
    quiet = _quiet_dir(tmp_path)
    command = server_simple._watch_command_powershell(quiet, timeout=1)

    result = _run(
        [
            powershell,
            "-NoProfile",
            "-Command",
            f"{command}; exit $LASTEXITCODE",
        ]
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == ""


def test_watch_argv_runs_from_unrelated_cwd_without_pythonpath(tmp_path: Path) -> None:
    quiet = _quiet_dir(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = _run(server_simple._watch_argv(quiet, timeout=1), cwd=unrelated, env=env)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == ""


def test_watch_wakes_after_snapshot_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_taken = threading.Event()
    original_snapshot = cli._snapshot_mtimes
    result_box: list[Result] = []

    def _snapshot_with_barrier(
        session_dir: Path, pattern: str
    ) -> dict[str, tuple[int, int]]:
        snapshot = original_snapshot(session_dir, pattern)
        baseline_taken.set()
        return snapshot

    def _run_watch() -> None:
        result_box.append(
            CliRunner().invoke(
                app,
                ["watch", str(tmp_path), "--timeout", "3", "--no-inbox"],
            )
        )

    monkeypatch.setattr(cli, "_snapshot_mtimes", _snapshot_with_barrier)
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.01)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    worker = threading.Thread(target=_run_watch)
    worker.start()
    assert baseline_taken.wait(timeout=2)
    marker = tmp_path / "state-worker.json"
    marker.write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 1.0}),
        encoding="utf-8",
    )
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert len(result_box) == 1
    result = result_box[0]
    assert result.exit_code == 0
    records = result.stdout.splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["reason"] == "waiting"

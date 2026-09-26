"""Shared ``codex queue`` runner: enqueue proof vs. uncertainty (plan v2 §2.2.3)."""

import subprocess
from pathlib import Path

import pytest

from claude_teams import native_wake as nw

THREAD = "01a0ddc6-7a2e-7052-a4ff-7e8730fcc140"
SUBMISSION = "01a0ddd0-2200-7373-a65f-8b58834c87bb"


def runner(returncode=0, stdout="", stderr="", raises=None, calls=None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return run


def test_success_parses_submission_id_and_keeps_home_and_cwd(monkeypatch):
    monkeypatch.setenv("AGENT_NAME", "x")
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "t")
    calls: list = []
    outcome = nw.codex_queue(
        "codex",
        THREAD,
        "/h/.codex",
        "multi\nline (x) & <y>",
        runner=runner(
            stdout=f"Queued message {SUBMISSION} for thread {THREAD}.\n", calls=calls
        ),
    )
    assert outcome.started
    assert outcome.exit == 0
    assert outcome.submission_id == SUBMISSION
    assert outcome.enqueued
    ((argv, kwargs),) = calls
    assert argv == [
        "codex",
        "queue",
        "--thread",
        THREAD,
        "--message",
        "multi\nline (x) & <y>",
    ]
    assert kwargs["env"]["CODEX_HOME"] == "/h/.codex"
    assert kwargs["cwd"] == Path.home()
    assert "AGENT_NAME" not in kwargs["env"]
    assert "CLAUDE_CODE_MESSAGING_TOKEN" not in kwargs["env"]


@pytest.mark.parametrize(
    ("kwargs", "started", "exit_code", "timed_out"),
    [
        ({"stdout": "done"}, True, 0, False),  # exit 0 but no parsable id
        ({"returncode": 1, "stderr": "boom"}, True, 1, False),
        ({"raises": subprocess.TimeoutExpired("codex", 1)}, True, None, True),
        ({"raises": subprocess.SubprocessError("late")}, True, None, False),
    ],
)
def test_everything_after_spawn_without_an_id_is_uncertain(
    kwargs, started, exit_code, timed_out
):
    outcome = nw.codex_queue("codex", THREAD, "/h", "m", runner=runner(**kwargs))
    assert (outcome.started, outcome.exit, outcome.timed_out) == (
        started,
        exit_code,
        timed_out,
    )
    assert not outcome.enqueued
    assert not outcome.provably_not_enqueued


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError()])
def test_exec_failure_proves_nothing_was_enqueued(error):
    outcome = nw.codex_queue("codex", THREAD, "/h", "m", runner=runner(raises=error))
    assert not outcome.started
    assert outcome.provably_not_enqueued

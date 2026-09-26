"""Idle/turn sequences on the state marker (plan v3 §2.3.4, review R2-5).

A latest-value ``state`` cannot prove a running→waiting edge: a short turn can
write both between two polls. The marker therefore carries monotonic counters
and the host's own session id, written read-modify-write under a lock.
"""

import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from claude_teams import hooks


@pytest.fixture(autouse=True)
def _no_inherited_epoch(monkeypatch):
    monkeypatch.delenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", raising=False)


def _emit(tmp_path: Path, event: str, session_id: str | None = "s1") -> None:
    payload: dict = {"hook_event_name": event}
    if session_id is not None:
        payload["session_id"] = session_id
    old = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        hooks.emit(session_dir=tmp_path, agent="worker")
    finally:
        sys.stdin = old


def _read(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "state-worker.json").read_text(encoding="utf-8"))


def test_first_markers_start_counters_at_their_first_event(tmp_path):
    _emit(tmp_path, "SessionStart")
    marker = _read(tmp_path)
    assert (marker["idle_seq"], marker["turn_seq"]) == (0, 0)
    assert marker["backend_session_id"] == "s1"
    _emit(tmp_path, "Stop")
    assert _read(tmp_path)["idle_seq"] == 1


def test_a_whole_turn_between_polls_still_advances_idle_seq(tmp_path):
    _emit(tmp_path, "Stop")
    seen = _read(tmp_path)["idle_seq"]
    _emit(tmp_path, "UserPromptSubmit")
    _emit(tmp_path, "PreToolUse")
    _emit(tmp_path, "Stop")
    marker = _read(tmp_path)
    assert marker["state"] == "waiting"
    assert marker["idle_seq"] == seen + 1
    assert marker["turn_seq"] == 1


def test_running_events_keep_idle_seq(tmp_path):
    _emit(tmp_path, "Stop")
    _emit(tmp_path, "PostToolUse")
    assert _read(tmp_path)["idle_seq"] == 1


def test_new_host_session_starts_a_new_namespace(tmp_path):
    _emit(tmp_path, "Stop", session_id="s1")
    _emit(tmp_path, "UserPromptSubmit", session_id="s1")
    _emit(tmp_path, "Stop", session_id="s1")
    _emit(tmp_path, "Stop", session_id="s2")
    marker = _read(tmp_path)
    assert marker["backend_session_id"] == "s2"
    assert (marker["idle_seq"], marker["turn_seq"]) == (1, 0)


def test_duplicate_stop_in_one_idle_period_does_not_rearm(tmp_path):
    _emit(tmp_path, "Stop")
    _emit(tmp_path, "Stop")
    assert _read(tmp_path)["idle_seq"] == 1


def test_epoch_from_environment_is_recorded_and_new_epoch_resets(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", "3")
    _emit(tmp_path, "Stop")
    _emit(tmp_path, "UserPromptSubmit")
    _emit(tmp_path, "Stop")
    assert (_read(tmp_path)["dispatch_epoch"], _read(tmp_path)["idle_seq"]) == (3, 2)
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", "4")
    _emit(tmp_path, "Stop")
    marker = _read(tmp_path)
    assert (marker["dispatch_epoch"], marker["idle_seq"], marker["turn_seq"]) == (
        4,
        1,
        0,
    )


def test_late_hook_from_an_older_epoch_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", "5")
    _emit(tmp_path, "Stop")
    before = (tmp_path / "state-worker.json").read_bytes()
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", "4")
    _emit(tmp_path, "UserPromptSubmit")
    assert (tmp_path / "state-worker.json").read_bytes() == before


@pytest.mark.parametrize("raw", ["", "x", "-2"])
def test_invalid_epoch_is_zero(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", raw)
    _emit(tmp_path, "Stop")
    assert _read(tmp_path)["dispatch_epoch"] == 0


def test_missing_session_id_is_empty(tmp_path):
    _emit(tmp_path, "Stop", session_id=None)
    assert _read(tmp_path)["backend_session_id"] == ""


def test_corrupt_prior_marker_restarts_counters(tmp_path):
    (tmp_path / "state-worker.json").write_text("{not json", encoding="utf-8")
    _emit(tmp_path, "Stop")
    assert _read(tmp_path)["idle_seq"] == 1


@pytest.mark.parametrize("workers", [8])
def test_concurrent_prompts_never_lose_an_increment(tmp_path, workers, monkeypatch):
    real = hooks._next_marker

    def slow(prior, *args, **kwargs):
        time.sleep(0.02)  # widen the read-modify-write window
        return real(prior, *args, **kwargs)

    monkeypatch.setattr(hooks, "_next_marker", slow)

    def run():
        hooks._record_event(tmp_path, "worker", "UserPromptSubmit", "s1")

    threads = [threading.Thread(target=run) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert _read(tmp_path)["turn_seq"] == workers


def test_blank_epoch_is_no_epoch(tmp_path, monkeypatch):
    """The launcher's blank override (no minted epoch) reads as epoch 0.

    It must never raise, and a host launched with it behaves exactly like one
    launched without the variable: its hooks are accepted at epoch 0.
    """
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", "")
    assert hooks._dispatch_epoch() == 0
    _emit(tmp_path, "Stop")
    _emit(tmp_path, "UserPromptSubmit")
    blank = _read(tmp_path)
    assert (blank["dispatch_epoch"], blank["state"], blank["turn_seq"]) == (
        0,
        "running",
        1,
    )
    monkeypatch.delenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH")
    _emit(tmp_path, "Stop")
    assert (_read(tmp_path)["dispatch_epoch"], _read(tmp_path)["idle_seq"]) == (0, 2)

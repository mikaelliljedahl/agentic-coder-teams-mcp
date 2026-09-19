"""``SubagentStop`` writes no marker; ``Stop`` markers carry a generation id.

Claude Code can fire ``SubagentStop`` in an agent that already ended its turn
(observed: an "away summary" 16 minutes after ``Stop``). Mapping it to
``waiting`` overwrote the parked ``Stop`` marker with one the watch ignores as
churn, so a finished agent could never wake its coordinator. The event is now
inert in the emitter, which also removes the false ``waiting`` that
``agent_status`` reported for an agent whose Task subagent had just finished.
"""

import io
import json
import sys
from pathlib import Path

import pytest

from claude_teams import hooks


def _emit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: str) -> None:
    payload = json.dumps({"hook_event_name": event, "session_id": "s1"})
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    hooks.emit(session_dir=tmp_path, agent="worker")


def _marker(tmp_path: Path) -> Path:
    return tmp_path / "state-worker.json"


def _read(tmp_path: Path) -> dict:
    return json.loads(_marker(tmp_path).read_text(encoding="utf-8"))


class TestSubagentStopIsInert:
    def test_over_parked_stop_leaves_bytes_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "Stop")
        before = _marker(tmp_path).read_bytes()

        _emit(tmp_path, monkeypatch, "SubagentStop")

        assert _marker(tmp_path).read_bytes() == before
        assert json.loads(before)["event"] == "Stop"

    def test_over_running_leaves_bytes_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "PostToolUse")
        before = _marker(tmp_path).read_bytes()

        _emit(tmp_path, monkeypatch, "SubagentStop")

        assert _marker(tmp_path).read_bytes() == before
        assert json.loads(before)["state"] == "running"

    def test_without_prior_marker_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "SubagentStop")

        assert not _marker(tmp_path).exists()
        assert list(tmp_path.iterdir()) == []

    def test_running_event_over_stop_still_downgrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "Stop")

        _emit(tmp_path, monkeypatch, "UserPromptSubmit")

        assert _read(tmp_path)["state"] == "running"


class TestGeneration:
    def test_stop_marker_carries_a_hex_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "Stop")

        marker = _read(tmp_path)
        assert marker["state"] == "waiting"
        assert marker["event"] == "Stop"
        assert isinstance(marker["gen"], str)
        assert len(marker["gen"]) == 32
        int(marker["gen"], 16)

    def test_two_parks_get_distinct_generations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "Stop")
        first = _read(tmp_path)["gen"]
        _emit(tmp_path, monkeypatch, "PreToolUse")
        _emit(tmp_path, monkeypatch, "Stop")

        assert _read(tmp_path)["gen"] != first

    def test_running_markers_also_carry_a_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _emit(tmp_path, monkeypatch, "PreToolUse")

        assert len(_read(tmp_path)["gen"]) == 32

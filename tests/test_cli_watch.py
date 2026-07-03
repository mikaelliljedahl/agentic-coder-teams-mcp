"""Tests for the `win-agent-teams watch` CLI helper (item 5)."""

import threading
import time
from pathlib import Path

from typer.testing import CliRunner

from claude_teams.cli import app

runner = CliRunner()


def test_watch_exits_2_on_timeout_with_no_change(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["watch", str(tmp_path), "--timeout", "1"]
    )

    assert result.exit_code == 2


def test_watch_exits_0_and_prints_path_when_file_created(tmp_path: Path) -> None:
    target = tmp_path / "state-worker.json"

    def _create_after_delay() -> None:
        time.sleep(0.3)
        target.write_text('{"state": "waiting", "event": "Stop", "ts": 1.0}')

    thread = threading.Thread(target=_create_after_delay)
    thread.start()
    try:
        result = runner.invoke(
            app, ["watch", str(tmp_path), "--timeout", "5"]
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_exits_0_and_prints_path_when_file_mtime_changes(tmp_path: Path) -> None:
    target = tmp_path / "state-worker.json"
    target.write_text('{"state": "running", "event": "Start", "ts": 1.0}')

    def _touch_after_delay() -> None:
        time.sleep(0.3)
        # Ensure the mtime actually advances on filesystems with coarse
        # resolution by bumping it explicitly rather than relying on wall time.
        new_mtime = target.stat().st_mtime + 5
        target.write_text('{"state": "waiting", "event": "Stop", "ts": 2.0}')
        import os

        os.utime(target, (new_mtime, new_mtime))

    thread = threading.Thread(target=_touch_after_delay)
    thread.start()
    try:
        result = runner.invoke(
            app, ["watch", str(tmp_path), "--timeout", "5"]
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_detects_same_mtime_rewrite(tmp_path: Path) -> None:
    import os

    target = tmp_path / "state-worker.json"
    target.write_text('{"state": "running", "event": "Start", "ts": 1.0}')
    original_mtime = target.stat().st_mtime

    def _rewrite_same_mtime_after_delay() -> None:
        time.sleep(0.3)
        # Rewrite with different (larger) content but pin the mtime back to
        # its original value, simulating an atomic-replace/rewrite that
        # preserves the second-resolution mtime.
        target.write_text(
            '{"state": "waiting", "event": "Stop", "ts": 2.0, "extra": "padding"}'
        )
        os.utime(target, (original_mtime, original_mtime))

    thread = threading.Thread(target=_rewrite_same_mtime_after_delay)
    thread.start()
    try:
        result = runner.invoke(
            app, ["watch", str(tmp_path), "--timeout", "5"]
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "state-worker.json" in result.stdout


def test_watch_respects_custom_pattern(tmp_path: Path) -> None:
    target = tmp_path / "output-report.md"

    def _create_after_delay() -> None:
        time.sleep(0.3)
        target.write_text("done")

    thread = threading.Thread(target=_create_after_delay)
    thread.start()
    try:
        result = runner.invoke(
            app,
            [
                "watch",
                str(tmp_path),
                "--timeout",
                "5",
                "--pattern",
                "output-*.md",
            ],
        )
    finally:
        thread.join()

    assert result.exit_code == 0
    assert "output-report.md" in result.stdout


def test_watch_default_pattern_ignores_non_matching_files(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("noise")

    result = runner.invoke(app, ["watch", str(tmp_path), "--timeout", "1"])

    assert result.exit_code == 2

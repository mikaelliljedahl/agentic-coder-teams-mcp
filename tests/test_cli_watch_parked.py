"""``watch`` wakes for markers that were already parked before it started.

A coordinator that arms its watch after a worker parked used to sit until
timeout, because the watch only reacted to edges. Each park is now delivered
once per reader, tracked by a generation id in ``.watch/ack-<reader>.json``.
Delivery is at-least-once: the wake is printed first, then the ack is written,
and an ack failure never suppresses a wake.
"""

import hashlib
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from claude_teams import cli
from claude_teams.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_WATCH_POLL_SECONDS", 0.02)
    monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.0)
    monkeypatch.delenv("AGENT_NAME", raising=False)


def _park(session: Path, agent: str = "worker", gen: str | None = "g1") -> Path:
    marker = session / f"state-{agent}.json"
    body: dict = {"state": "waiting", "event": "Stop", "ts": 1.0}
    if gen is not None:
        body["gen"] = gen
    marker.write_text(json.dumps(body), encoding="utf-8")
    return marker


def _ack(session: Path, reader: str = "team-lead") -> dict:
    path = session / ".watch" / f"ack-{reader}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _watch(session: Path, *extra: str, timeout: str = "1"):
    return runner.invoke(app, ["watch", str(session), "--timeout", timeout, *extra])


def _expect_wake(result, session: Path, agent: str = "worker") -> None:
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "reason": "waiting",
        "agent": agent,
        "path": str(session / f"state-{agent}.json"),
    }


def _expect_timeout(result) -> None:
    assert result.exit_code == 2, result.output
    assert result.stdout == ""


class TestParkedAtStart:
    def test_parked_marker_wakes_and_is_acked(self, tmp_path: Path) -> None:
        _park(tmp_path)

        result = _watch(tmp_path)

        _expect_wake(result, tmp_path)
        assert _ack(tmp_path) == {"acked": {"state-worker.json": "g1"}}

    def test_acked_generation_does_not_wake_again(self, tmp_path: Path) -> None:
        _park(tmp_path)
        _watch(tmp_path)

        _expect_timeout(_watch(tmp_path, timeout="0.15"))

    def test_new_generation_wakes_again(self, tmp_path: Path) -> None:
        _park(tmp_path, gen="g1")
        _watch(tmp_path)
        _park(tmp_path, gen="g2")

        _expect_wake(_watch(tmp_path), tmp_path)
        assert _ack(tmp_path)["acked"]["state-worker.json"] == "g2"

    def test_legacy_marker_is_fingerprinted_by_bytes(self, tmp_path: Path) -> None:
        marker = _park(tmp_path, gen=None)
        digest = hashlib.sha256(marker.read_bytes()).hexdigest()

        _expect_wake(_watch(tmp_path), tmp_path)
        assert _ack(tmp_path)["acked"]["state-worker.json"] == digest
        _expect_timeout(_watch(tmp_path, timeout="0.15"))

        marker.write_text('{"state":"waiting","event":"Stop","ts":2.0}', "utf-8")
        _expect_wake(_watch(tmp_path), tmp_path)

    def test_legacy_subagentstop_marker_is_not_actionable(self, tmp_path: Path) -> None:
        (tmp_path / "state-worker.json").write_text(
            '{"state":"waiting","event":"SubagentStop","gen":"g1"}', "utf-8"
        )

        _expect_timeout(_watch(tmp_path, timeout="0.15"))
        assert not (tmp_path / ".watch").exists()

    def test_two_parked_markers_are_delivered_one_per_run(self, tmp_path: Path) -> None:
        _park(tmp_path, "a", "ga")
        _park(tmp_path, "b", "gb")

        first = _watch(tmp_path)
        second = _watch(tmp_path)
        third = _watch(tmp_path, timeout="0.15")

        assert first.exit_code == 0
        assert second.exit_code == 0
        woken = {json.loads(first.stdout)["agent"], json.loads(second.stdout)["agent"]}
        assert woken == {"a", "b"}
        assert _ack(tmp_path)["acked"] == {
            "state-a.json": "ga",
            "state-b.json": "gb",
        }
        _expect_timeout(third)

    def test_pattern_filters_parked_scan(self, tmp_path: Path) -> None:
        _park(tmp_path)

        _expect_timeout(_watch(tmp_path, "--pattern", "out-*.md", timeout="0.15"))


class TestNoParked:
    def test_no_parked_restores_edge_only_behavior(self, tmp_path: Path) -> None:
        _park(tmp_path)

        _expect_timeout(_watch(tmp_path, "--no-parked", timeout="0.15"))
        assert not (tmp_path / ".watch").exists()

    def test_no_parked_still_wakes_on_a_new_edge(self, tmp_path: Path) -> None:
        _park(tmp_path, gen="g1")
        marker = tmp_path / "state-worker.json"

        def _repark() -> None:
            time.sleep(0.08)
            marker.write_text(
                '{"state":"waiting","event":"Stop","ts":2.0,"gen":"g2"}', "utf-8"
            )

        thread = threading.Thread(target=_repark)
        thread.start()
        try:
            result = _watch(tmp_path, "--no-parked")
        finally:
            thread.join()

        _expect_wake(result, tmp_path)


class TestPriorityAndSettle:
    def test_unread_message_wins_and_marker_is_not_acked(self, tmp_path: Path) -> None:
        _park(tmp_path)
        (tmp_path / "inbox-team-lead.jsonl").write_text(
            json.dumps({"from": "worker", "text": "hi", "ts": 1.0}) + "\n", "utf-8"
        )

        result = _watch(tmp_path)

        assert result.exit_code == 0
        assert json.loads(result.stdout)["reason"] == "message"
        assert not (tmp_path / ".watch").exists()

    def test_output_edge_wins_while_parked_marker_settles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 0.3)
        _park(tmp_path)
        out = tmp_path / "report.md"

        def _write_output() -> None:
            time.sleep(0.06)
            out.write_text("{}", "utf-8")

        thread = threading.Thread(target=_write_output)
        thread.start()
        try:
            result = _watch(tmp_path, "--pattern", "*")
        finally:
            thread.join()

        assert result.exit_code == 0
        assert json.loads(result.stdout) == {"reason": "output", "path": str(out)}

    def test_timeout_before_settle_does_not_ack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "_WATCH_SETTLE_SECONDS", 1.0)
        _park(tmp_path)

        _expect_timeout(_watch(tmp_path, timeout="0.2"))
        assert not (tmp_path / ".watch").exists()

    def test_edge_triggered_wait_is_acked_too(self, tmp_path: Path) -> None:
        marker = tmp_path / "state-worker.json"

        def _park_later() -> None:
            time.sleep(0.08)
            _park(tmp_path, gen="edge")

        thread = threading.Thread(target=_park_later)
        thread.start()
        try:
            result = _watch(tmp_path)
        finally:
            thread.join()

        _expect_wake(result, tmp_path)
        assert _ack(tmp_path)["acked"]["state-worker.json"] == "edge"
        assert marker.exists()
        _expect_timeout(_watch(tmp_path, timeout="0.15"))


class TestAckStore:
    def test_corrupt_ack_file_is_treated_as_empty_and_rewritten(
        self, tmp_path: Path
    ) -> None:
        _park(tmp_path)
        ack = tmp_path / ".watch" / "ack-team-lead.json"
        ack.parent.mkdir()
        ack.write_bytes(b"{not json")

        _expect_wake(_watch(tmp_path), tmp_path)
        assert _ack(tmp_path) == {"acked": {"state-worker.json": "g1"}}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
    def test_ack_write_failure_never_suppresses_the_wake(self, tmp_path: Path) -> None:
        _park(tmp_path)
        ack_dir = tmp_path / ".watch"
        ack_dir.mkdir()
        ack_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            result = _watch(tmp_path)
        finally:
            ack_dir.chmod(stat.S_IRWXU)

        _expect_wake(result, tmp_path)
        assert "ack" in result.stderr.lower()
        _expect_wake(_watch(tmp_path), tmp_path)  # redelivered: at-least-once

    def test_ack_files_are_never_an_output_edge(self, tmp_path: Path) -> None:
        """A broad-pattern watch with nothing actionable must sleep through
        another reader's ack write inside ``.watch/``."""
        (tmp_path / ".watch").mkdir()
        written = threading.Event()

        def _ack_from_another_reader() -> None:
            time.sleep(0.08)
            (tmp_path / ".watch" / "ack-other.json").write_text('{"acked":{}}')
            written.set()

        thread = threading.Thread(target=_ack_from_another_reader)
        thread.start()
        try:
            result = _watch(tmp_path, "--pattern", "*", timeout="0.4")
        finally:
            thread.join()

        assert written.is_set()
        _expect_timeout(result)


class TestReaders:
    def test_each_reader_wakes_once(self, tmp_path: Path) -> None:
        _park(tmp_path)

        _expect_wake(_watch(tmp_path, "--reader", "a"), tmp_path)
        _expect_wake(_watch(tmp_path, "--reader", "b"), tmp_path)
        _expect_timeout(_watch(tmp_path, "--reader", "a", timeout="0.15"))
        assert _ack(tmp_path, "a") == _ack(tmp_path, "b")

    def test_ambient_agent_name_selects_the_ack_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_NAME", "child-lead")
        _park(tmp_path)

        _expect_wake(_watch(tmp_path), tmp_path)
        assert (tmp_path / ".watch" / "ack-child-lead.json").exists()
        assert not (tmp_path / ".watch" / "ack-team-lead.json").exists()

    def test_unsafe_ambient_agent_name_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_NAME", "../evil")
        _park(tmp_path)

        result = _watch(tmp_path)

        assert result.exit_code == 1
        assert "unsafe reader" in result.stderr
        assert not (tmp_path / ".watch").exists()


class TestAckHelper:
    def test_concurrent_same_reader_acks_are_merged(self, tmp_path: Path) -> None:
        """Two same-reader watchers acking different agents must not lose an
        entry: the locked read-modify-write serialises the merge."""
        ack = tmp_path / ".watch" / "ack-team-lead.json"
        markers = [_park(tmp_path, f"a{i}", f"g{i}") for i in range(8)]
        barrier = threading.Barrier(len(markers))
        errors: list[BaseException] = []

        def _ack_one(marker: Path, gen: str) -> None:
            try:
                barrier.wait(timeout=5)
                cli._acknowledge(ack, cli._Parked(marker.stem[6:], gen, marker))
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=_ack_one, args=(m, f"g{i}"))
            for i, m in enumerate(markers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert _ack(tmp_path)["acked"] == {
            f"state-a{i}.json": f"g{i}" for i in range(8)
        }
        assert [p.name for p in ack.parent.iterdir() if p.suffix == ".tmp"] == []

    def test_failed_replace_leaves_old_ack_and_no_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ack = tmp_path / ".watch" / "ack-team-lead.json"
        ack.parent.mkdir()
        ack.write_text('{"acked": {"state-old.json": "g0"}}', "utf-8")
        original = Path.replace

        def _fail(path: Path, target: Path) -> Path:
            if target == ack:
                raise OSError("denied")
            return original(path, target)

        monkeypatch.setattr(Path, "replace", _fail)
        marker = _park(tmp_path)

        with pytest.raises(OSError, match="denied"):
            cli._acknowledge(ack, cli._Parked("worker", "g1", marker))

        assert _ack(tmp_path) == {"acked": {"state-old.json": "g0"}}
        assert [p.name for p in ack.parent.iterdir() if p.suffix == ".tmp"] == []

    def test_temp_name_is_unique_per_process_and_call(self, tmp_path: Path) -> None:
        ack = tmp_path / ".watch" / "ack-team-lead.json"
        marker = _park(tmp_path)

        cli._acknowledge(ack, cli._Parked("worker", "g1", marker))
        cli._acknowledge(ack, cli._Parked("worker", "g2", marker))

        assert _ack(tmp_path) == {"acked": {"state-worker.json": "g2"}}
        assert sorted(p.name for p in ack.parent.iterdir()) == [
            "ack-team-lead.json",
            "ack-team-lead.json.lock",
        ]


class TestAckedGenerationSuppression:
    """Finding 1 of the implementation review: suppression must apply to
    later edges too, not only to the start-up scan."""

    def test_touching_an_acked_marker_does_not_redeliver(self, tmp_path: Path) -> None:
        marker = _park(tmp_path, gen="g1")
        _expect_wake(_watch(tmp_path), tmp_path)

        def _touch_same_gen() -> None:
            time.sleep(0.08)
            marker.write_text(
                '{"state":"waiting","event":"Stop","ts":9.0,"gen":"g1"}', "utf-8"
            )

        thread = threading.Thread(target=_touch_same_gen)
        thread.start()
        try:
            result = _watch(tmp_path, timeout="0.4")
        finally:
            thread.join()

        _expect_timeout(result)

    def test_identical_legacy_bytes_rewrite_does_not_redeliver(
        self, tmp_path: Path
    ) -> None:
        marker = _park(tmp_path, gen=None)
        body = marker.read_bytes()
        _expect_wake(_watch(tmp_path), tmp_path)

        def _rewrite_same_bytes() -> None:
            time.sleep(0.08)
            marker.write_bytes(body + b"")

        thread = threading.Thread(target=_rewrite_same_bytes)
        thread.start()
        try:
            result = _watch(tmp_path, timeout="0.4")
        finally:
            thread.join()

        _expect_timeout(result)

    def test_new_generation_edge_still_wakes(self, tmp_path: Path) -> None:
        marker = _park(tmp_path, gen="g1")
        _expect_wake(_watch(tmp_path), tmp_path)

        def _repark() -> None:
            time.sleep(0.08)
            marker.write_text(
                '{"state":"waiting","event":"Stop","ts":9.0,"gen":"g2"}', "utf-8"
            )

        thread = threading.Thread(target=_repark)
        thread.start()
        try:
            result = _watch(tmp_path)
        finally:
            thread.join()

        _expect_wake(result, tmp_path)
        assert _ack(tmp_path)["acked"]["state-worker.json"] == "g2"

    def test_no_parked_bypasses_the_ack_on_edges(self, tmp_path: Path) -> None:
        marker = _park(tmp_path, gen="g1")
        _expect_wake(_watch(tmp_path), tmp_path)

        def _touch_same_gen() -> None:
            time.sleep(0.08)
            marker.write_text(
                '{"state":"waiting","event":"Stop","ts":9.0,"gen":"g1"}', "utf-8"
            )

        thread = threading.Thread(target=_touch_same_gen)
        thread.start()
        try:
            result = _watch(tmp_path, "--no-parked")
        finally:
            thread.join()

        _expect_wake(result, tmp_path)


class TestEmitParkedWakeOrdering:
    """Platform-independent pins for print-then-ack and non-fatal ack failure."""

    def _parked(self, tmp_path: Path) -> cli._Parked:
        return cli._Parked("worker", "g1", tmp_path / "state-worker.json")

    def test_wake_is_printed_before_the_ack_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        monkeypatch.setattr(cli, "_emit_wake", lambda _r: order.append("print"))
        monkeypatch.setattr(cli, "_acknowledge", lambda _a, _p: order.append("ack"))

        with pytest.raises(typer.Exit) as raised:
            cli._emit_parked_wake(
                tmp_path / ".watch" / "ack.json", self._parked(tmp_path)
            )

        assert raised.value.exit_code == 0
        assert order == ["print", "ack"]

    @pytest.mark.parametrize(
        "failure", [OSError("disk full"), cli.FileLockTimeoutError("lock held")]
    )
    def test_ack_failure_keeps_exit_0_and_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        failure: BaseException,
    ) -> None:
        printed: list[dict] = []
        monkeypatch.setattr(cli, "_emit_wake", printed.append)

        def _raise(_a: Path, _p: cli._Parked) -> None:
            if isinstance(failure, cli.FileLockTimeoutError):
                raise OSError(str(failure))  # what _acknowledge converts it to
            raise failure

        monkeypatch.setattr(cli, "_acknowledge", _raise)

        with pytest.raises(typer.Exit) as raised:
            cli._emit_parked_wake(
                tmp_path / ".watch" / "ack.json", self._parked(tmp_path)
            )

        assert raised.value.exit_code == 0
        assert printed == [
            {
                "reason": "waiting",
                "agent": "worker",
                "path": str(tmp_path / "state-worker.json"),
            }
        ]
        assert "could not write watch ack" in capsys.readouterr().err

    def test_lock_timeout_inside_acknowledge_becomes_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _timeout(*_a: object, **_k: object):
            raise cli.FileLockTimeoutError("held")

        monkeypatch.setattr(cli, "file_lock", _timeout)
        ack = tmp_path / ".watch" / "ack-team-lead.json"

        with pytest.raises(OSError, match="held"):
            cli._acknowledge(ack, self._parked(tmp_path))

        assert [p.name for p in ack.parent.iterdir() if p.suffix == ".tmp"] == []

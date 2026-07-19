"""Unit tests for the pure inbox-protocol helpers in ``messaging``."""

import json
from pathlib import Path

from claude_teams.messaging import (
    purge_sender_from_inbox,
    read_inbox_by_sender,
)


def _line(sender: str, text: str) -> str:
    return json.dumps({"from": sender, "text": text, "ts": "t"})


class TestPurgeSenderFromInbox:
    def test_removes_only_target_senders_lines(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox-lead.jsonl"
        inbox.write_text(
            "\n".join(
                [
                    _line("worker", "a"),
                    _line("other", "b"),
                    _line("worker", "c"),
                    _line("other", "d"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        purge_sender_from_inbox(inbox, "worker")

        by_sender = read_inbox_by_sender(inbox)
        assert "worker" not in by_sender
        assert [m["text"] for _, m in by_sender["other"]] == ["b", "d"]

    def test_missing_inbox_is_noop(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox-lead.jsonl"
        purge_sender_from_inbox(inbox, "worker")
        assert not inbox.exists()

    def test_preserves_malformed_and_foreign_lines(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox-lead.jsonl"
        inbox.write_text(
            "\n".join(
                [
                    "not json",
                    _line("worker", "gone"),
                    _line("other", "kept"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        purge_sender_from_inbox(inbox, "worker")

        remaining = inbox.read_text(encoding="utf-8").splitlines()
        assert "not json" in remaining
        assert not any('"worker"' in line for line in remaining)
        assert any('"other"' in line for line in remaining)

    def test_purging_last_sender_leaves_empty_file(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox-lead.jsonl"
        inbox.write_text(_line("worker", "a") + "\n", encoding="utf-8")

        purge_sender_from_inbox(inbox, "worker")

        assert inbox.read_text(encoding="utf-8") == ""

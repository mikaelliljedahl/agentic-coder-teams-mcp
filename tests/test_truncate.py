"""Tests for the shared ``_truncate`` helper (R4 bounded, signalled truncation)."""

from claude_teams import server_simple


class TestTruncate:
    def test_truncate_under_limit_no_flag(self) -> None:
        clipped, truncated, full_len = server_simple._truncate("hello", 10)
        assert clipped == "hello"
        assert truncated is False
        assert full_len == 5

    def test_truncate_exact_limit_no_flag(self) -> None:
        clipped, truncated, full_len = server_simple._truncate("hello", 5)
        assert clipped == "hello"
        assert truncated is False
        assert full_len == 5

    def test_truncate_over_limit_sets_full_len(self) -> None:
        text = "x" * 100
        clipped, truncated, full_len = server_simple._truncate(text, 10)
        assert len(clipped) == 10
        assert truncated is True
        assert full_len == 100

    def test_truncate_none_maxchars_returns_full_text(self) -> None:
        text = "x" * 500
        clipped, truncated, full_len = server_simple._truncate(text, None)
        assert clipped == text
        assert truncated is False
        assert full_len == 500

    def test_truncate_zero_maxchars_returns_empty(self) -> None:
        clipped, truncated, full_len = server_simple._truncate("hello", 0)
        assert clipped == ""
        assert truncated is True
        assert full_len == 5

    def test_truncate_empty_text(self) -> None:
        clipped, truncated, full_len = server_simple._truncate("", 10)
        assert clipped == ""
        assert truncated is False
        assert full_len == 0

    def test_truncate_none_text_returns_empty(self) -> None:
        clipped, truncated, full_len = server_simple._truncate(None, 10)
        assert clipped == ""
        assert truncated is False
        assert full_len == 0

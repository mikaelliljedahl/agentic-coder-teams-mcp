"""R4 — PID-reuse-safe liveness: creation tokens + fail-closed ownership.

The invariant under test: no destructive PID operation may run unless we are
confident the live PID is still *our* process. That confidence comes from a
per-PID creation token (Windows GetProcessTimes creation FILETIME / Linux
/proc/<pid>/stat starttime). A tokenless record, a token mismatch, or an
unreadable token must all resolve to "not owned" (fail closed).

These tests use the current interpreter PID (guaranteed alive, stable token,
and NOT in the manager's in-memory registry — so they exercise the pure
token path, i.e. the post-restart code path).
"""

import os

from claude_teams.backends import process_manager as pm

# A PID that is implausibly high and (essentially) never live on a test host.
_DEAD_PID = "2147480000"


class TestCreationToken:
    def test_stable_for_live_pid(self) -> None:
        handle = str(os.getpid())
        first = pm.creation_token(handle)
        second = pm.creation_token(handle)
        assert first is not None
        assert first == second

    def test_none_for_dead_pid(self) -> None:
        assert pm.creation_token(_DEAD_PID) is None

    def test_none_for_non_numeric_handle(self) -> None:
        assert pm.creation_token("not-a-pid") is None


class TestOwnsProcess:
    def test_true_on_token_match(self) -> None:
        handle = str(os.getpid())
        token = pm.creation_token(handle)
        assert pm.process_manager.owns_process(handle, token) is True

    def test_false_on_token_mismatch(self) -> None:
        handle = str(os.getpid())
        assert pm.process_manager.owns_process(handle, "wrong-token") is False

    def test_false_when_tokenless(self) -> None:
        handle = str(os.getpid())
        assert pm.process_manager.owns_process(handle, None) is False
        assert pm.process_manager.owns_process(handle, "") is False

    def test_false_for_dead_pid_even_with_token(self) -> None:
        assert pm.process_manager.owns_process(_DEAD_PID, "whatever") is False


class TestTokenAwareHealthCheck:
    def test_reports_dead_on_token_mismatch(self) -> None:
        handle = str(os.getpid())
        alive, _ = pm.process_manager.health_check(handle, expected_token="wrong")  # noqa: S106
        assert alive is False

    def test_reports_alive_on_token_match(self) -> None:
        handle = str(os.getpid())
        token = pm.creation_token(handle)
        alive, _ = pm.process_manager.health_check(handle, expected_token=token)
        assert alive is True

    def test_backward_compat_no_token(self) -> None:
        handle = str(os.getpid())
        alive, _ = pm.process_manager.health_check(handle)
        assert alive is True

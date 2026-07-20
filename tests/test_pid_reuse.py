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
from types import SimpleNamespace

from claude_teams import server_simple
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


class _StubOwnershipManager(pm._PidOwnershipMixin):
    """Minimal manager to exercise the in-memory ownership shortcut."""

    def __init__(self, handle: str, *, tracked_alive: bool) -> None:
        self._processes = {handle: object()}
        self._tracked = tracked_alive

    def _pid_alive(self, handle: str) -> bool:
        return True  # a reused PID would satisfy bare liveness

    def _tracked_alive(self, info: object) -> bool:
        return self._tracked


class TestInMemoryOwnershipIsReuseSafe:
    def test_stale_entry_with_dead_child_is_not_owned(self) -> None:
        # B1: the PID is "in _processes" and bare-alive, but OUR tracked
        # child/pane is dead → must NOT be treated as owned (a reused PID).
        mgr = _StubOwnershipManager(_DEAD_PID, tracked_alive=False)
        assert mgr.owns_process(_DEAD_PID, None) is False
        assert mgr.owns_process(_DEAD_PID, "mismatch") is False

    def test_live_tracked_entry_is_owned(self) -> None:
        mgr = _StubOwnershipManager(_DEAD_PID, tracked_alive=True)
        assert mgr.owns_process(_DEAD_PID, None) is True

    def test_linux_terminal_ownership_ignores_reused_sidecar_pid(self) -> None:
        # A reused sidecar PID must NOT make a Linux-terminal record "owned":
        # ownership is proven by our own terminal launcher child, not a bare PID.
        mgr = pm.LinuxTerminalProcessManager()
        dead_launcher = SimpleNamespace(
            terminal_process=SimpleNamespace(poll=lambda: 0)
        )
        mgr._processes = {"launcher": dead_launcher}
        assert mgr._tracked_alive(dead_launcher) is False
        assert mgr.owns_process("launcher", None) is False
        assert mgr.owns_process("launcher", "mismatch") is False

        live_launcher = SimpleNamespace(
            terminal_process=SimpleNamespace(poll=lambda: None)
        )
        mgr._processes = {"launcher": live_launcher}
        assert mgr.owns_process("launcher", None) is True


class TestLeaseHolderLivenessIsReuseSafe:
    """A4b — the lease holder is identified by ``(pid, create_token)``.

    A lease left behind by an EARLIER incarnation of the server whose PID has
    since been recycled onto this process must not be treated as live just
    because the numbers match. Skipping the token check for our own PID makes
    that stale lease permanently unreclaimable: nothing else can ever prove the
    holder is gone.
    """

    def test_our_own_pid_still_has_to_pair_with_a_matching_token(self) -> None:
        handle = str(os.getpid())
        token = pm.creation_token(handle)
        assert server_simple._lease_holder_live(os.getpid(), token) is True
        assert server_simple._lease_holder_live(os.getpid(), "wrong-token") is False
        assert server_simple._lease_holder_live(os.getpid(), None) is False

    def test_a_foreign_live_pid_with_a_stale_token_is_not_ours(self) -> None:
        assert server_simple._lease_holder_live(int(_DEAD_PID), "whatever") is False


class TestTokenAwareHealthCheck:
    def test_reports_dead_on_token_mismatch(self) -> None:
        handle = str(os.getpid())
        alive, _ = pm.process_manager.health_check(handle, expected_token="wrong")
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


class TestOwnershipProbe:
    """Round-2 critical 4: uncertainty must not be reclaimed as death.

    ``owns_process`` collapses "provably gone" and "could not tell" into one
    ``False``. That is correct for a destructive gate — never kill what you
    cannot prove is yours — and wrong for a **reclaim** gate, where the same
    ``False`` authorizes a second caller to resume a conversation the first is
    still working. ``ownership_probe`` keeps the three answers apart.
    """

    def test_ours_when_the_live_token_matches(self) -> None:
        pid = str(os.getpid())
        assert (
            pm.process_manager.ownership_probe(pid, pm.creation_token(pid))
            == pm.OWNERSHIP_OURS
        )

    def test_not_ours_on_a_token_mismatch(self) -> None:
        assert (
            pm.process_manager.ownership_probe(str(os.getpid()), "not-the-token")
            == pm.OWNERSHIP_NOT_OURS
        )

    def test_not_ours_for_a_dead_pid(self) -> None:
        """A PID that is not alive is settled: the holder really is gone."""
        assert (
            pm.process_manager.ownership_probe(_DEAD_PID, "whatever")
            == pm.OWNERSHIP_NOT_OURS
        )

    def test_indeterminate_when_a_live_pid_s_token_is_unreadable(
        self, monkeypatch
    ) -> None:
        """Access denied against a LIVE process is uncertainty, not death.

        This is the one case ``owns_process`` cannot express, and the one that
        matters: reclaiming here hands a second caller a live delivery.
        """
        monkeypatch.setattr(pm, "creation_token", lambda handle: None)

        assert (
            pm.process_manager.ownership_probe(str(os.getpid()), "expected")
            == pm.OWNERSHIP_INDETERMINATE
        )

    def test_owns_process_still_refuses_everything_but_ours(self, monkeypatch) -> None:
        """The destructive gate is unchanged: indeterminate is still False."""
        monkeypatch.setattr(pm, "creation_token", lambda handle: None)

        assert pm.process_manager.owns_process(str(os.getpid()), "expected") is False

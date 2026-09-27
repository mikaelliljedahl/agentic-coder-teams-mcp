"""trust_cwd x native carrier: the boundaries chosen when the two features merged.

``trust_cwd`` guards a Codex *relaunch* only. A native ``codex queue`` carrier
submits to the live session and never restarts Codex, so it skips the trust
preflight; every path that falls back to resume (stage-2 eligibility loss, a
provably-not-enqueued native attempt) must re-enter ``_prepare`` and meet the
preflight, and the commit-time recheck still guards the relaunch itself.

Built on the real delivery/lease stores and the real ``codex queue`` runner
from ``tests/test_native_codex_dispatch.py``; only the processes are fake.
"""

import pytest

from claude_teams import delivery_store as ds
from claude_teams import leases, native_wake, server_simple
from tests import test_native_selection as _sel
from tests.test_native_codex_dispatch import (
    BINARY,
    _agent,
    _codex_target,
    _CodexResumeBackend,
    _install,
    _Queue,
)
from tests.test_native_selection import AGENT, KEY, SESSION, _FakeRegistry, _row

env = _sel.env

NEW_BINARY = "C:/codex/new/codex.exe"


class _TrustCodexBackend(_CodexResumeBackend):
    """Codex-shaped resume backend that also answers binary discovery."""

    def __init__(self, rollout, binaries: list[str]) -> None:
        super().__init__(rollout)
        self.binaries = list(binaries)
        self.discovered = 0

    def discover_binary(self) -> str:
        self.discovered += 1
        index = min(self.discovered, len(self.binaries)) - 1
        return self.binaries[index]


def _trusted_codex_target(env, *, interactive_now: bool, binaries=(BINARY,)):
    """An eligible codex_queue target that was spawned with ``trust_cwd``.

    ``interactive_now`` is what the server's launch mode reports today; the
    record says the agent was launched interactive, so ``False`` makes any
    resume fail the trust preflight with ``trust_cwd_launch_mode_changed``.
    """
    case = _codex_target(env)
    _sel._set_agent(env, trust_cwd=True, launch_interactive=True)
    backend = _TrustCodexBackend(env.transcript, list(binaries))
    env.monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    env.codex = backend
    env.monkeypatch.setattr(
        server_simple.process_manager,
        "provides_tty",
        lambda *a, **k: interactive_now,
    )
    return case


def _forbid_stopping_the_old_child(env) -> None:
    def fail(*args, **kwargs):
        pytest.fail("the old child was stopped")

    env.monkeypatch.setattr(server_simple.process_manager, "graceful_shutdown", fail)
    env.monkeypatch.setattr(server_simple.process_manager, "kill_process", fail)


def _assert_old_child_untouched(before: dict) -> None:
    after = _agent()
    for field in ("pid", "create_token", "spawned_at", "dispatch_epoch"):
        assert after[field] == before[field], field
    assert after["trust_cwd"] is True
    assert after["launch_interactive"] is True
    assert server_simple.PENDING_DELIVERY_FIELD not in after


def _assert_lease_released() -> None:
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


def _spy_resume_builds(env) -> list:
    """Record resume-request builds: a _prepare refusal happens before any."""
    built: list = []
    real = server_simple._build_resume_request

    def spy(*args, **kwargs):
        built.append(args)
        return real(*args, **kwargs)

    env.monkeypatch.setattr(server_simple, "_build_resume_request", spy)
    return built


# ==========================================================================
# a. native delivery despite a resume environment the preflight would refuse
# ==========================================================================


@pytest.mark.parametrize("unsafe", ["launch_mode_changed", "direct_launch"])
@pytest.mark.asyncio
async def test_trusted_native_delivery_skips_the_relaunch_preflight(
    env, unsafe: str
) -> None:
    _trusted_codex_target(env, interactive_now=unsafe != "launch_mode_changed")
    if unsafe == "direct_launch":
        # What WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1 means on Windows; patched
        # so the case also runs on the Linux CI host.
        env.monkeypatch.setattr(
            server_simple.process_manager_module,
            "codex_direct_launch_enabled",
            lambda: True,
        )
        assert (
            server_simple._trust_cwd_preflight("codex", str(env.work), True, BINARY)
            == "trust_cwd_unsafe_transport"
        )
    _forbid_stopping_the_old_child(env)
    queue = _install(env, _Queue(env.transcript))
    before = _agent()

    result = await server_simple.follow_up_agent(AGENT, "next step", KEY)

    assert result["status"] == ds.STATUS_DELIVERED, result
    assert result["method"] == ds.METHOD_CODEX_QUEUE
    assert result["pid"] == 123
    assert len(queue.calls) == 1
    assert env.codex.resume_calls == []
    assert env.codex.discovered == 0, "no relaunch preflight for a native carrier"
    row = _row()
    assert row["status"] == ds.STATUS_DELIVERED
    assert row["attempts"] == 1
    _assert_old_child_untouched(before)
    _assert_lease_released()


# ==========================================================================
# b. stage-2 eligibility loss, then a trust refusal on the resume retry
# ==========================================================================


@pytest.mark.asyncio
async def test_stage_two_loss_then_trust_refusal_never_resumes(env) -> None:
    _trusted_codex_target(env, interactive_now=False)
    _forbid_stopping_the_old_child(env)
    queue = _install(env, _Queue(env.transcript))
    calls = {"n": 0}

    def verified_once(home, thread):
        calls["n"] += 1
        return (True, "") if calls["n"] == 1 else (False, "archived")

    env.monkeypatch.setattr(native_wake, "verify_codex_thread", verified_once)
    built = _spy_resume_builds(env)
    before = _agent()

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert calls["n"] >= 2, "stage 2 re-evaluated eligibility"
    assert result["reason"] == "trust_cwd_launch_mode_changed", result
    assert built == [], "refused by the _prepare preflight, before a resume build"
    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_PENDING
    assert result["retriable"] is True
    assert queue.calls == []
    assert env.codex.resume_calls == []
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["attempts"] == 0, "stage-2 loss is never marked sent"
    _assert_old_child_untouched(before)
    _assert_lease_released()


# ==========================================================================
# c. provably-not-enqueued native attempt, then a trust refusal
# ==========================================================================


@pytest.mark.asyncio
async def test_not_enqueued_native_attempt_then_trust_refusal(env) -> None:
    _trusted_codex_target(env, interactive_now=False)
    _forbid_stopping_the_old_child(env)
    queue = _install(env, _Queue(env.transcript, "ctor"))
    built = _spy_resume_builds(env)
    before = _agent()

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == "trust_cwd_launch_mode_changed", result
    assert built == [], "refused by the _prepare preflight, before a resume build"
    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_PENDING
    assert queue.calls == [], "Popen construction failed: nothing ran"
    assert env.codex.resume_calls == []
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["reason"] == "native_not_enqueued"
    assert row["attempts"] == 1, "only the reverted native attempt is counted"
    assert not ds.is_unresolved_native(row)
    _assert_old_child_untouched(before)
    _assert_lease_released()


# ==========================================================================
# d. successful resume fallback keeps the trust override and mints an epoch
# ==========================================================================


@pytest.mark.asyncio
async def test_resume_fallback_carries_trust_extras_and_a_new_epoch(env) -> None:
    _trusted_codex_target(env, interactive_now=True)
    _install(env, _Queue(env.transcript, "ctor"))
    old_epoch = _agent()["dispatch_epoch"]

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED, result
    assert result["method"] == ds.METHOD_RESUME
    ((request, _),) = env.codex.resume_calls
    extra = request.extra or {}
    assert extra["codex_trust_cwd"] == "1"
    assert extra["codex_trust_binary"] == BINARY
    assert extra["codex_trust_interactive"] == "1"
    new_epoch = int(extra["dispatch_epoch"])
    assert new_epoch > old_epoch
    row = _row()
    assert row["attempts"] == 2
    assert row[ds.METHOD_FIELD] == ds.METHOD_RESUME
    after = _agent()
    assert after["trust_cwd"] is True
    assert after["launch_interactive"] is True
    assert after["dispatch_epoch"] == new_epoch
    assert str(after["pid"]) == "789", "the resumed child replaced the old one"
    _assert_lease_released()


# ==========================================================================
# e. the Codex binary changes between the retry's preflight and the commit
# ==========================================================================


@pytest.mark.asyncio
async def test_changed_binary_at_fallback_commit_is_refused(env) -> None:
    _trusted_codex_target(env, interactive_now=True, binaries=[BINARY, NEW_BINARY])
    _forbid_stopping_the_old_child(env)
    _install(env, _Queue(env.transcript, "ctor"))
    before = _agent()

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == "trust_cwd_binary_changed", result
    assert result["retriable"] is True
    assert env.codex.discovered == 2, "pinned in _prepare, rechecked at commit"
    assert env.codex.resume_calls == []
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["attempts"] == 1, "the refused resume was never marked sent"
    _assert_old_child_untouched(before)
    _assert_lease_released()

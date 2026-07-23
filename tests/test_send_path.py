"""C3/R5 — ``send_message`` classifies its recipient instead of rerouting.

R5 forbids an accept-then-drop path reachable as a general-purpose send. The
behaviour this file pins down is the classification that replaces the old
"unknown name is routed to the lead with a warning" rule:

===================  ==================================================
Recipient class      Behaviour
===================  ==================================================
own child            guaranteed (Phase B) path — never the inbox
own spawner          inbox + watcher, unchanged (this is what R3 rests on)
sibling              refused; sibling messaging is an explicit non-goal
root lead            as spawner
unknown / typo       refused, and specifically NOT sent upstream
===================  ==================================================

The last row is the point of the whole change: a typo must never become a
silent upstream message.

Identity spoofing is out of scope for the same reason it is in
``test_direction_guard``: ``IDENTITY`` is read by the caller's own process.
This is an accident guard.

The registry is flat — every agent shares one ``agents.json`` — so parentage is
a field, which is why "sibling" is a distinct class from "child". The fixture
below is reused from ``test_direction_guard``: ``team-lead`` spawned ``lead-a``,
which spawned ``worker-b`` and ``worker-c``.
"""

import json

import pytest

from claude_teams import delivery_store, server_simple
from claude_teams.delivery import DELIVERY_MARKER_PREFIX
from tests import test_direction_guard as _guard
from tests.test_direction_guard import (
    SESSION,
    _as,
    _backend_session,
    _install,
    _user_record,
)
from tests.test_follow_up_delivery import _append, _FakeResumeBackend

#: The flat lead/siblings/orphan session, reused rather than rebuilt: C3 and C2
#: classify the same relationships, so they must agree about the same fixture.
env = _guard.env


def _inbox(name: str):
    return server_simple._inbox_file(SESSION, name)


def _inbox_lines(name: str) -> list[dict]:
    path = _inbox(name)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _receipt_writer(env, name: str):
    """Play the CLI's part: record the delivery nonce in the target transcript."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcripts[name],
            _user_record(
                _backend_session(name),
                f"the message {DELIVERY_MARKER_PREFIX}{nonce}",
            ),
        )

    return write_receipt


def _alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (handle == "789", "x"),
    )


# ==========================================================================
# Own child -> the guaranteed path
# ==========================================================================


@pytest.mark.asyncio
async def test_send_to_own_child_uses_the_guaranteed_path(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend(_receipt_writer(env, "worker-b"))
    _install(monkeypatch, backend)
    _alive(monkeypatch)
    _as(monkeypatch, "lead-a")

    result = await server_simple.send_message(
        "the message", to="worker-b", idempotency_key="c3-child"
    )

    assert result["success"] is True
    assert result["status"] == delivery_store.STATUS_DELIVERED
    assert backend.resume_calls, "a downstream send must resume, not append"


@pytest.mark.asyncio
async def test_root_lead_send_to_own_child_uses_the_guaranteed_path(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root lead is a spawner too; its children get the same treatment."""
    backend = _FakeResumeBackend(_receipt_writer(env, "lead-a"))
    _install(monkeypatch, backend)
    _alive(monkeypatch)
    _as(monkeypatch, "team-lead")

    result = await server_simple.send_message(
        "the message", to="lead-a", idempotency_key="c3-root-child"
    )

    assert result["success"] is True
    assert result["status"] == delivery_store.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_guaranteed_send_does_not_double_present(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 must survive C3: the same text must not be both resumed and readable.

    If a guaranteed-path message also landed in the actionable inbox, a polling
    worker would act on it twice — once from its resumed context and once from
    ``read_messages``.
    """
    _install(monkeypatch, _FakeResumeBackend(_receipt_writer(env, "worker-b")))
    _alive(monkeypatch)
    _as(monkeypatch, "lead-a")

    await server_simple.send_message(
        "the message", to="worker-b", idempotency_key="c3-no-double"
    )

    assert _inbox_lines("worker-b") == []
    audit = delivery_store.list_for_sender(
        server_simple._deliveries_file(SESSION), "lead-a", "worker-b"
    )
    assert [row["idempotency_key"] for row in audit] == ["c3-no-double"]


@pytest.mark.asyncio
async def test_downstream_send_without_a_key_is_refused_with_a_pointer(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guaranteed path is only guaranteed if the outcome is recoverable."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _alive(monkeypatch)
    _as(monkeypatch, "lead-a")

    result = await server_simple.send_message("the message", to="worker-b")

    assert result["success"] is False
    assert result["reason"] == delivery_store.KEY_REQUIRED
    assert backend.resume_calls == []
    assert _inbox_lines("worker-b") == []


# ==========================================================================
# Own spawner -> inbox + watcher, unchanged (R3 rests on this)
# ==========================================================================


@pytest.mark.asyncio
async def test_send_to_spawner_by_alias_still_writes_the_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "worker-b")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "lead-a")

    result = await server_simple.send_message("status update", to="team-lead")

    assert result["success"] is True
    assert result["to"] == "lead-a"
    assert [line["text"] for line in _inbox_lines("lead-a")] == ["status update"]


@pytest.mark.asyncio
async def test_send_to_spawner_by_name_still_writes_the_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the spawner explicitly is the same upstream path as the alias.

    Load-bearing: ``lead-a`` was spawned by ``team-lead``, not by ``worker-b``,
    so a classifier that only asked "did I spawn this?" would call it a sibling
    and refuse the one path R3 depends on.
    """
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "worker-b")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "lead-a")

    result = await server_simple.send_message("status update", to="lead-a")

    assert result["success"] is True
    assert result["to"] == "lead-a"
    assert [line["from"] for line in _inbox_lines("lead-a")] == ["worker-b"]


@pytest.mark.asyncio
async def test_root_lead_alias_resolves_to_its_own_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "team-lead")

    result = await server_simple.send_message("note to self", to="team-lead")

    assert result["success"] is True
    assert result["to"] == "team-lead"
    assert [line["text"] for line in _inbox_lines("team-lead")] == ["note to self"]


# ==========================================================================
# Sibling -> refused, not rerouted
# ==========================================================================


@pytest.mark.asyncio
async def test_sibling_send_is_refused_and_writes_nothing(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "worker-b")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "lead-a")

    result = await server_simple.send_message("do my work", to="worker-c")

    assert result["success"] is False
    assert result["reason"] == "recipient_not_addressable"
    assert result["recipient_class"] == "sibling"
    assert _inbox_lines("worker-c") == []
    # The refusal is a refusal, not a redirect.
    assert _inbox_lines("lead-a") == []


@pytest.mark.asyncio
async def test_sibling_refusal_names_the_shared_spawner_route(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "worker-b")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "lead-a")

    result = await server_simple.send_message("do my work", to="worker-c")

    assert "worker-c" in result["detail"]
    assert "lead-a" in result["detail"]
    assert result["retriable"] is False


@pytest.mark.asyncio
async def test_lead_send_to_a_grandchild_is_refused(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested parentage: ``worker-b`` is ``team-lead``'s grandchild, not child."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _alive(monkeypatch)
    _as(monkeypatch, "team-lead")

    result = await server_simple.send_message(
        "skip a level", to="worker-b", idempotency_key="c3-grandchild"
    )

    assert result["success"] is False
    assert result["reason"] == "recipient_not_addressable"
    assert backend.resume_calls == []
    assert _inbox_lines("worker-b") == []


# ==========================================================================
# Unknown / typo -> refused, and NOT silently upstream
# ==========================================================================


@pytest.mark.asyncio
async def test_typo_recipient_is_refused_and_never_reaches_the_lead(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The specific failure R5 exists to remove."""
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "worker-b")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "lead-a")

    result = await server_simple.send_message("urgent", to="leed")

    assert result["success"] is False
    assert result["reason"] == "recipient_not_addressable"
    assert result["recipient_class"] == "unknown"
    assert "leed" in result["detail"]
    assert _inbox_lines("lead-a") == [], "a typo must not become an upstream message"
    assert not _inbox("leed").exists()


@pytest.mark.asyncio
async def test_unknown_recipient_refusal_lists_no_stray_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    _as(monkeypatch, "team-lead")

    result = await server_simple.send_message("hello", to="ghost")

    assert result["success"] is False
    assert not _inbox("ghost").exists()
    assert _inbox_lines("team-lead") == []


# ==========================================================================
# The classifier itself
# ==========================================================================


@pytest.mark.parametrize(
    ("identity", "parent", "to", "expected_class", "expected_name"),
    [
        ("lead-a", "team-lead", "worker-b", "child", "worker-b"),
        ("lead-a", "team-lead", "team-lead", "spawner", "team-lead"),
        ("lead-a", "team-lead", "orchestrator", "spawner", "team-lead"),
        ("worker-b", "lead-a", "lead-a", "spawner", "lead-a"),
        ("worker-b", "lead-a", "team-lead", "spawner", "lead-a"),
        ("worker-b", "lead-a", "worker-c", "sibling", "worker-c"),
        ("worker-b", "lead-a", "nope", "unknown", "nope"),
        ("team-lead", "", "lead-a", "child", "lead-a"),
        # A grandchild is refused like a sibling, but is not one: reporting it
        # as "sibling" would be a lie in a client-visible field.
        ("team-lead", "", "worker-b", "unrelated", "worker-b"),
        ("team-lead", "", "team-lead", "spawner", "team-lead"),
    ],
)
def test_classify_recipient(
    env,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
    parent: str,
    to: str,
    expected_class: str,
    expected_name: str,
) -> None:
    _as(monkeypatch, identity)
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", parent)

    assert server_simple._classify_recipient(to, SESSION) == (
        expected_class,
        expected_name,
    )

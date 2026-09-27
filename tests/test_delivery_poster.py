"""B — the child's ``DeliveryPoster`` (plan v3.1 §2.3.4, §2.9 R3-2, §4 tests 8, 12).

The poster runs in the Claude child's OWN MCP server and presents entries the
lead offered in its delivery mailbox to its own host channel. What is pinned:

- the idle gate: ``waiting``, same dispatch epoch, same backend session and an
  ``idle_seq`` above the epoch's consumption; one post per idle sequence;
- validation at take AND begin (epoch, backend session, host identity);
- ``finish`` with ok / pre-write failure (rollback, re-arm) / uncertain;
- a replacement poster never replays ``taken`` or ``posting``;
- the capability marker it heartbeats under its owner lock;
- it runs only with master + downstream + the ``_CLAUDE`` half on, read from
  its own environment.

The mailbox is the real store; the channel post and the hook marker are fakes
(plus one run through the real hook writer).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import delivery_mailbox as mb
from claude_teams import hooks, native_wake
from claude_teams.delivery_poster import DeliveryPoster, PosterFacts

SESSION = "sess"
CHILD = "worker"
BS = "backend-session"
EPOCH = 3
HOST = (123, "tok-123")
FLAGS = (
    "WIN_AGENT_TEAMS_NATIVE_WAKE",
    "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX",
)


def _flags(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for suffix, value in values.items():
        monkeypatch.setenv(f"WIN_AGENT_TEAMS_{suffix}", value)


class _Channel:
    """A fake host channel: records posts, answers with a scripted result."""

    def __init__(self) -> None:
        self.posts: list[str] = []
        self.results: list[native_wake.PostResult] = []
        self.on_post = None

    def __call__(
        self, channel: native_wake.ClaudeChannel, text: str
    ) -> native_wake.PostResult:
        self.posts.append(text)
        if self.on_post is not None:
            self.on_post()
        if self.results:
            return self.results.pop(0)
        return native_wake.PostResult(True, "", True)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    _flags(monkeypatch, NATIVE_WAKE="1", NATIVE_DOWNSTREAM="1")
    session_dir = tmp_path / SESSION
    session_dir.mkdir()
    assert mb.ensure_initialised(session_dir, CHILD).done
    state = SimpleNamespace(
        marker={
            "state": "waiting",
            "event": "Stop",
            "idle_seq": 1,
            "turn_seq": 1,
            "backend_session_id": BS,
            "dispatch_epoch": EPOCH,
        },
        markers=[],  # scripted successive marker reads, then ``marker``
        binding=mb.HostBinding(EPOCH, BS, *HOST),
        host=HOST,
        epoch=EPOCH,
    )

    def read_marker(session: str, name: str) -> dict | None:
        assert (session, name) == (SESSION, CHILD)
        if state.markers:
            return state.markers.pop(0)
        return dict(state.marker) if state.marker is not None else None

    def current_binding(session: str, name: str, host) -> mb.HostBinding | None:
        assert (session, name) == (SESSION, CHILD)
        return state.binding

    state.read_marker = read_marker
    state.current_binding = current_binding
    world = SimpleNamespace(
        tmp_path=tmp_path,
        session_dir=session_dir,
        state=state,
        monkeypatch=monkeypatch,
        posters=[],
    )
    yield world
    for poster in world.posters:
        poster.close()


def _poster(
    world: SimpleNamespace,
    channel: _Channel | None = None,
    *,
    pid: int | None = None,
    token: str = "me",
    reason: str = "available",
) -> tuple[DeliveryPoster, _Channel]:
    state = world.state
    channel = channel or _Channel()
    facts = PosterFacts(
        host=lambda: state.host,
        current_binding=state.current_binding,
        read_marker=state.read_marker,
        identity=mb.PosterIdentity(pid if pid is not None else os.getpid(), token),
        epoch=lambda: state.epoch,
    )
    poster = DeliveryPoster(
        facts,
        get_target=lambda: (SESSION, CHILD),
        session_dir=lambda sid: world.tmp_path / sid,
        channel=native_wake.ClaudeChannel(reason, "/fake", "secret", True, HOST[0]),
        post=channel,
        poll=1.0,
    )
    world.posters.append(poster)
    return poster, channel


def _offer(world: SimpleNamespace, nonce: str, *, epoch: int = EPOCH, ts=None) -> None:
    result = mb.publish(
        world.session_dir,
        CHILD,
        nonce,
        operation_id=f"op-{nonce}",
        sender="team-lead",
        key=f"key-{nonce}",
        dispatch_epoch=epoch,
        backend_session_id=BS,
        text=f"text for {nonce}",
        row_is_current=lambda: True,
        now=ts,
    )
    assert result.done


def _state(world: SimpleNamespace, nonce: str) -> str | None:
    return mb.read_entry(world.session_dir, CHILD, nonce).state


def _consumed(world: SimpleNamespace) -> dict:
    doc = mb.read_mailbox(world.session_dir, CHILD)
    assert doc is not None
    return doc["consumed"]


# ==========================================================================
# The happy path and the idle gate
# ==========================================================================


def test_waiting_child_gets_the_offer_posted_once(world) -> None:
    _offer(world, "a" * 32)
    poster, channel = _poster(world)

    poster.tick()
    poster.tick()

    assert channel.posts == [f"text for {'a' * 32}"]
    assert _state(world, "a" * 32) == mb.STATE_POSTED
    assert _consumed(world) == {str(EPOCH): {"seq": 1, "nonce": "a" * 32}}


def test_a_whole_turn_between_ticks_still_gives_exactly_one_post(world) -> None:
    """``idle_seq`` +2 between ticks (running and waiting both missed)."""
    _offer(world, "a" * 32, ts=1.0)
    poster, channel = _poster(world)
    poster.tick()
    _offer(world, "b" * 32, ts=2.0)
    world.state.marker["idle_seq"] = 3  # a full turn happened unseen

    poster.tick()
    poster.tick()

    assert channel.posts == [f"text for {'a' * 32}", f"text for {'b' * 32}"]
    assert _consumed(world)[str(EPOCH)] == {"seq": 3, "nonce": "b" * 32}


def test_restart_on_the_same_waiting_marker_posts_once_then_never(world) -> None:
    world.state.marker["idle_seq"] = 5
    _offer(world, "a" * 32, ts=1.0)
    first, channel = _poster(world, token="first")
    first.tick()
    first.close()
    _offer(world, "b" * 32, ts=2.0)
    second, _ = _poster(world, channel, token="second")

    second.tick()
    second.tick()

    assert channel.posts == [f"text for {'a' * 32}"], "idle_seq 5 was consumed"
    assert _state(world, "b" * 32) == mb.STATE_OFFERED


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"state": "running", "event": "UserPromptSubmit"}, "busy"),
        ({"backend_session_id": "predecessor"}, "stale predecessor marker"),
        ({"dispatch_epoch": EPOCH - 1}, "old epoch"),
        ({"idle_seq": 0}, "never idle in this namespace"),
    ],
)
def test_marker_gate_refuses(world, change: dict, why: str) -> None:
    world.state.marker.update(change)
    _offer(world, "a" * 32)
    poster, channel = _poster(world)

    poster.tick()

    assert channel.posts == [], why
    assert _state(world, "a" * 32) == mb.STATE_OFFERED, "the entry is not touched"


def test_missing_marker_refuses(world) -> None:
    world.state.marker = None
    _offer(world, "a" * 32)
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


def test_same_idle_seq_never_posts_a_second_entry(world) -> None:
    _offer(world, "a" * 32, ts=1.0)
    _offer(world, "b" * 32, ts=2.0)
    poster, channel = _poster(world)

    poster.tick()
    poster.tick()

    assert channel.posts == [f"text for {'a' * 32}"], "oldest first, one per seq"
    assert _state(world, "b" * 32) == mb.STATE_OFFERED
    world.state.marker["idle_seq"] = 2
    poster.tick()
    assert channel.posts[-1] == f"text for {'b' * 32}"


def test_offers_of_another_epoch_are_ignored(world) -> None:
    _offer(world, "a" * 32, epoch=EPOCH + 1)
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


def test_unavailable_channel_never_takes(world) -> None:
    _offer(world, "a" * 32)
    poster, channel = _poster(world, reason="no_socket")
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


# ==========================================================================
# finish: ok, pre-write failure (rollback and re-arm), uncertain
# ==========================================================================


def test_pre_write_failure_rolls_back_and_the_next_offer_reuses_the_seq(
    world,
) -> None:
    """begin -> pre-write failure -> rollback -> retry (R3-2)."""
    _offer(world, "a" * 32, ts=1.0)
    channel = _Channel()
    channel.results = [native_wake.PostResult(False, "refused", False)]
    poster, _ = _poster(world, channel)

    poster.tick()

    assert _state(world, "a" * 32) == mb.STATE_FAILED_BEFORE_WRITE
    assert _consumed(world) == {}, "the idle sequence is not spent"
    # The lead retries under a new nonce; the same idle period suffices.
    _offer(world, "b" * 32, ts=2.0)
    poster.tick()
    assert channel.posts == [f"text for {'a' * 32}", f"text for {'b' * 32}"]
    assert _state(world, "b" * 32) == mb.STATE_POSTED
    assert _consumed(world)[str(EPOCH)] == {"seq": 1, "nonce": "b" * 32}


def test_failure_after_the_write_started_is_uncertain_and_never_retried(
    world,
) -> None:
    _offer(world, "a" * 32)
    channel = _Channel()
    channel.results = [native_wake.PostResult(False, "timeout", True)]
    poster, _ = _poster(world, channel)

    poster.tick()
    world.state.marker["idle_seq"] = 2
    poster.tick()

    assert channel.posts == [f"text for {'a' * 32}"]
    assert _state(world, "a" * 32) == mb.STATE_UNCERTAIN
    assert _consumed(world)[str(EPOCH)]["seq"] == 1


def test_a_raising_post_is_treated_as_uncertain(world) -> None:
    _offer(world, "a" * 32)
    channel = _Channel()

    def boom() -> None:
        raise RuntimeError

    channel.on_post = boom
    poster, _ = _poster(world, channel)
    poster.tick()
    assert _state(world, "a" * 32) == mb.STATE_UNCERTAIN


# ==========================================================================
# Validation at take and at begin
# ==========================================================================


def test_waiting_to_running_between_take_and_begin_leaves_it_taken(world) -> None:
    _offer(world, "a" * 32)
    running = {**world.state.marker, "state": "running", "event": "PreToolUse"}
    world.state.markers = [dict(world.state.marker), running]
    poster, channel = _poster(world)

    poster.tick()

    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_TAKEN
    # The same poster resumes its own taken entry at the next idle edge.
    world.state.marker["idle_seq"] = 2
    poster.tick()
    assert channel.posts == [f"text for {'a' * 32}"]
    assert _state(world, "a" * 32) == mb.STATE_POSTED


def test_epoch_bump_before_take_is_rejected_untouched(world) -> None:
    _offer(world, "a" * 32)
    world.state.binding = mb.HostBinding(EPOCH + 1, BS, *HOST)
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


def test_epoch_bump_between_take_and_begin_stops_the_poster_at_begin(world) -> None:
    _offer(world, "a" * 32)

    def bump_after_take(session: str, name: str) -> dict:
        if _state(world, "a" * 32) == mb.STATE_TAKEN:
            world.state.binding = mb.HostBinding(EPOCH + 1, BS, *HOST)
        return dict(world.state.marker)

    world.state.read_marker = bump_after_take
    poster, channel = _poster(world)

    poster.tick()

    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_TAKEN, "begin refused, never posted"
    assert _consumed(world) == {}


def test_host_mismatch_is_rejected(world) -> None:
    _offer(world, "a" * 32)
    world.state.binding = mb.HostBinding(EPOCH, BS, 456, "tok-456")
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


def test_unknown_host_never_takes(world) -> None:
    _offer(world, "a" * 32)
    world.state.host = None
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == []
    assert _state(world, "a" * 32) == mb.STATE_OFFERED


@pytest.mark.parametrize("stage", ["taken", "posting"])
def test_replacement_poster_never_replays(world, stage: str) -> None:
    _offer(world, "a" * 32)
    dead = mb.PosterIdentity(999_999, "dead")
    binding = mb.HostBinding(EPOCH, BS, *HOST)
    assert mb.take(
        world.session_dir,
        CHILD,
        "a" * 32,
        poster=dead,
        binding=binding,
        current_binding=lambda: binding,
    ).done
    if stage == "posting":
        assert mb.begin(
            world.session_dir,
            CHILD,
            "a" * 32,
            poster=dead,
            binding=binding,
            current_binding=lambda: binding,
            idle=mb.IdleProof(EPOCH, BS, 1),
        ).done
    world.state.marker["idle_seq"] = 7
    poster, channel = _poster(world)

    poster.tick()

    assert channel.posts == []
    assert _state(world, "a" * 32) == stage


def test_lead_death_does_not_stop_a_surviving_poster(world) -> None:
    """The poster never consults the lead: only the child's record binding."""
    _offer(world, "a" * 32)
    poster, channel = _poster(world)
    poster.tick()
    assert channel.posts == [f"text for {'a' * 32}"]


# ==========================================================================
# Capability marker and owner lock
# ==========================================================================


def _capability(world: SimpleNamespace) -> dict:
    path = world.session_dir / f"native-delivery-{CHILD}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_capability_marker_is_written_and_heartbeats(world, monkeypatch) -> None:
    now = [1_000.0]
    monkeypatch.setattr("claude_teams.delivery_poster.time.time", lambda: now[0])
    poster, _ = _poster(world)

    poster.tick()
    first = _capability(world)
    now[0] = 1_005.0
    poster.tick()

    assert first == {
        "pid": os.getpid(),
        "create_token": "me",
        "host_pid": HOST[0],
        "host_create_token": HOST[1],
        "backend_session_id": BS,
        "dispatch_epoch": EPOCH,
        "channel": "available",
        "heartbeat_ts": 1_000.0,
    }
    assert _capability(world)["heartbeat_ts"] == 1_005.0
    assert poster.owns()


def test_capability_marker_takes_the_session_only_from_this_epochs_marker(
    world,
) -> None:
    world.state.marker["dispatch_epoch"] = EPOCH - 1
    poster, _ = _poster(world)
    poster.tick()
    assert _capability(world)["backend_session_id"] == ""


def test_capability_reports_an_unavailable_channel(world) -> None:
    poster, _ = _poster(world, reason="socket_missing")
    poster.tick()
    assert _capability(world)["channel"] == "socket_missing"


def test_second_poster_cannot_own_the_same_identity(world) -> None:
    _offer(world, "a" * 32)
    owner, _ = _poster(world, token="owner")
    owner.tick()
    world.state.marker["idle_seq"] = 2
    _offer(world, "b" * 32)
    other, other_channel = _poster(world, token="other")

    other.tick()

    assert not other.owns()
    assert other_channel.posts == []
    assert _capability(world)["create_token"] == "owner"


def test_closing_releases_the_owner_lock(world) -> None:
    first, _ = _poster(world, token="first")
    first.tick()
    first.close()
    second, _ = _poster(world, token="second")
    second.tick()
    assert second.owns()
    assert _capability(world)["create_token"] == "second"


# ==========================================================================
# Flags, read from the poster's own environment (§4 test 12)
# ==========================================================================


@pytest.mark.parametrize(
    "combo",
    [
        {},
        {"NATIVE_WAKE": "1"},
        {"NATIVE_DOWNSTREAM": "1"},
        {"NATIVE_WAKE": "1", "NATIVE_DOWNSTREAM": "1", "NATIVE_WAKE_CLAUDE": "0"},
    ],
)
def test_poster_is_inert_unless_master_downstream_and_claude_half_are_on(
    world, combo: dict
) -> None:
    _flags(world.monkeypatch, **combo)
    _offer(world, "a" * 32)
    poster, channel = _poster(world)

    poster.tick()

    assert channel.posts == []
    assert not poster.owns()
    assert not (world.session_dir / f"native-delivery-{CHILD}.json").exists()
    assert not (world.session_dir / f"native-delivery-{CHILD}.lock").exists()


def test_codex_half_off_does_not_stop_the_claude_poster(world) -> None:
    _flags(
        world.monkeypatch,
        NATIVE_WAKE="1",
        NATIVE_DOWNSTREAM="1",
        NATIVE_WAKE_CODEX="0",
    )
    _offer(world, "a" * 32)
    poster, channel = _poster(world)
    poster.tick()
    assert len(channel.posts) == 1


def test_flags_turned_off_later_release_the_lock(world) -> None:
    poster, _ = _poster(world)
    poster.tick()
    assert poster.owns()
    _flags(world.monkeypatch, NATIVE_WAKE="1")
    poster.tick()
    assert not poster.owns()


def test_no_identity_no_poster(world) -> None:
    poster, channel = _poster(world)
    poster.get_target = lambda: None
    _offer(world, "a" * 32)
    poster.tick()
    assert channel.posts == []
    assert not poster.owns()


# ==========================================================================
# Wiring: a second target kind in the notifier, and the real hook marker
# ==========================================================================


def test_notifier_ticks_the_poster_independently_of_the_inbox_channel(world) -> None:
    _offer(world, "a" * 32)
    poster, channel = _poster(world)
    notifier = native_wake.NativeWakeNotifier(
        get_target=lambda: None,
        session_dir=lambda sid: world.tmp_path / sid,
        member_alive=lambda sid, name: False,
        channel=native_wake.ClaudeChannel("no_socket"),
        delivery=poster,
    )
    try:
        notifier.tick()
    finally:
        notifier.close()

    assert channel.posts == [f"text for {'a' * 32}"]
    assert not poster.owns(), "the notifier releases the poster's lock on close"


def test_real_hook_marker_gates_the_poster(world, monkeypatch) -> None:
    """Hooks write the marker; the poster reads it without a lock."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_DISPATCH_EPOCH", str(EPOCH))

    def read_marker(session: str, name: str) -> dict | None:
        path = world.session_dir / f"state-{name}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            return None

    world.state.read_marker = read_marker
    _offer(world, "a" * 32, ts=1.0)
    poster, channel = _poster(world)

    hooks._record_event(world.session_dir, CHILD, "SessionStart", BS)
    poster.tick()
    assert channel.posts == [], "running: no post"

    hooks._record_event(world.session_dir, CHILD, "Stop", BS)
    hooks._record_event(world.session_dir, CHILD, "Stop", BS)  # duplicate Stop
    poster.tick()
    _offer(world, "b" * 32, ts=2.0)
    poster.tick()
    assert channel.posts == [f"text for {'a' * 32}"], "one post per idle period"

    hooks._record_event(world.session_dir, CHILD, "UserPromptSubmit", BS)
    hooks._record_event(world.session_dir, CHILD, "Stop", BS)
    poster.tick()
    assert channel.posts[-1] == f"text for {'b' * 32}"

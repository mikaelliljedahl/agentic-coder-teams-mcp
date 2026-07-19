# Requirement: Directional, non-dropping message delivery

Status: draft. Open questions from the first draft are now resolved (see
"Findings"); both changed the requirement, one materially.

## Problem

Reaching a running agent has two paths, and each fails structurally.

### 1. There is no way to reach a busy worker

`follow_up_agent` refuses a live busy agent with `reason="agent_busy"`
(`src/claude_teams/server_simple.py:1629-1636`). `send_message` appends to
`inbox-{name}.jsonl` and never wakes the target process — the recipient sees it
only if it calls `read_messages` itself, and a busy agent does not. A Codex
worker never polls at all.

Both tools fail on the same input. The choice between them only matters when the
target is idle. In the case that actually matters — the worker is mid-task and
the lead needs to correct or unblock it — the protocol offers no correct option:
one tool returns a refusal, the other returns success without delivering.

This is not a prompting problem. `send_message`'s own docstring says it is "not a
push/resume mechanism" and directs the caller to `follow_up_agent`
(`server_simple.py:1314-1316`); the orchestration skill repeats it
(`.claude/skills/agent-orchestration/SKILL.md:28`). Leads still pick wrong,
repeatedly, because the guidance points at a tool that then refuses.

### 2. Resume misidentifies its target, and cannot self-correct

`follow_up_agent` is not a stdin write. It kills the target process (when
`replace_if_idle=True`, the default, and ownership is provable) and spawns a
**new OS process** via `backend.resume()` against a `backend_session_id` scraped
from the backend's rollout log.

That id is chosen once, by heuristic, and then pinned. On first observation
`read_claude_output` picks candidate transcripts by "jsonl started after
`spawned_at`" in the cwd's project dir and takes **max mtime**
(`agent_output.py:97-108`). With any other concurrent Claude session in the same
project dir — the coordinator itself, a sibling, a sidechain — that file can
belong to a different conversation. Whatever it picks is persisted by
`_sync_backend_session_id` (`server_simple.py:980-987`), and from then on the
reader filters *exclusively* to the stored id (`agent_output.py:99-102`). The
mechanism that would have to observe a corrected id is already keyed to the
wrong one. **Once wrong, permanently wrong.**

Three consequences follow, all observed together in the field: three consecutive
`follow_up_agent` calls each returned success with a new PID, the worker
behaved throughout as if nothing arrived, and a later `send_message` was read
and acted on immediately.

- **Nothing verifies the resumed child survived.** `follow_up_agent` takes
  `new_pid` from the process handle and returns `success: True`
  (`server_simple.py:1703-1736`) with no liveness check, no exit code, no stdout
  inspection. `claude --resume <unknown-id>` exits within a second with "no
  conversation found" — and still yields a fresh PID and success.
- **The real worker is never killed.** The kill is gated fail-closed on
  `process_manager.owns_process` (`server_simple.py:1656-1659`), which returns
  `False` without in-memory ownership or a matching creation token. So the
  original worker keeps running untouched while `resume` starts a separate
  process against a different conversation.
- **Its inbox still works.** `send_message` is keyed by agent *name*
  (`server_simple.py:1324-1333`), independent of `backend_session_id`, so the
  still-running worker picks it up on its next `read_messages`.

The liveness gate cannot save this: with a pinned wrong transcript,
`last_activity_at` never advances, so the agent always looks idle and the
`_FOLLOW_UP_IDLE_SECONDS` check always passes. Busy/idle detection is downstream
of the same fragile identification step.

Separately, the prompt-file indirection races. `_prompt_file` is deterministic
per agent (`server_simple.py:228-230`) and overwritten unconditionally
(`:1135-1137`), so consecutive calls to one agent name share a path; the last
write wins and earlier prompts are lost. This did not cause the observed
incident — it delivers the *wrong* prompt, not silence — but it is a real defect.

### 3. Resume is unguarded against upstream calls

`_do_follow_up` performs no caller check. A spawned worker can call
`follow_up_agent` on its own lead; if the lead looks idle, the default
`replace_if_idle=True` kills the lead's process and respawns it. A subordinate
can terminate the orchestrator mid-run, reachable by accident.

Session resume is a downstream capability. Upstream traffic stays inbox-based,
which is sound because the lead runs `win-agent-teams watch` — the property
workers lack.

## Findings that shaped these requirements

**Caller identity is self-asserted and cannot be authenticated.** `IDENTITY` is a
module-level constant read from the `AGENT_NAME` environment variable at import
time (`server_simple.py:49-58`). There is one MCP server process *per agent*,
launched from a config file the agent itself can rewrite
(`_write_mcp_config`, `server_simple.py:1107-1124`), with `permission_mode`
defaulting to `bypass`. A worker can change its identity by editing that config,
by launching `AGENT_NAME=team-lead python -m claude_teams.server_simple`, or by
exporting the var into processes it spawns. The "server" enforcing any check is
the caller's own process.

**There is no recorded parentage.** The agent record
(`server_simple.py:1275-1289`) has no spawner field. `_message_recipient`
(`server_simple.py:777-809`) resolves lead aliases from `_AGENT_PARENT_NAME` —
the caller's own env var — not from a store. An earlier draft of this document
asserted that parentage existed and could be looked up; that was wrong.

**Consequence for R2.** A direction rule cannot be an authorization boundary
here. Making it one would require a single shared broker process, an unforgeable
per-agent credential minted at spawn time, and a spawner field persisted
somewhere workers cannot write — a redesign of the trust model, not a check.
That is out of scope. The stated threat is a confused agent, not a hostile one,
so a non-security guard is an appropriate fit — but it must be labelled as one
and not described as security.

## Requirements

Ordered by dependency. R8 is a prerequisite for R1.

**R8 — Resume must correctly identify its target and prove it worked.**
No compatibility exception: an agent whose binding cannot be verified MUST NOT be
resumed, including records predating correlation. Plan review 3 showed the
alternative — allowing legacy records through on an unverifiable stored id —
would let a nonce be confirmed in the wrong conversation and reported as
`delivered`, which is the original bug with a false receipt attached. The
accepted cost is that agents spawned before the upgrade cannot be followed up and
must be killed and respawned; the refusal MUST say so.

1. `backend_session_id` MUST be verifiable, not merely pinned. Selection MUST
   NOT rely on max-mtime among candidate transcripts, and a stored id MUST be
   re-validated rather than trusted permanently. The reader MUST NOT be
   structurally incapable of observing a corrected id.
2. Resume MUST confirm the child survived and attached — a returned PID is not
   evidence. A backend exiting with "no conversation found" MUST surface as a
   failure.
3. Prompt files MUST NOT collide across calls to the same agent.

**R6 — Delivery is confirmed at the recipient, not at the transport.**
A resume MUST NOT report success on the basis that a process started.
`delivered` is set only when the prompt is observably present in the target's
context.

Non-confirmation MUST distinguish two cases. An earlier draft collapsed both into
`failed`; plan review 4 showed that would misdescribe reality, because a
transcript write buffered past the bound can arrive afterwards.

- **Definite non-delivery** — child dead, no receipt record → `failed`. Terminal.
- **Live uncertainty** — bound expired while the child is still alive →
  non-terminal, reported as still in flight. It MUST NOT be reported as
  delivered, and MUST NOT be reported as terminally failed; either is a claim the
  system cannot support. Any retry MUST reconcile the prior attempt first.

Neither case may be reported as success.

**R1 — Downstream delivery is guaranteed within the call, with an honest tail.**
A spawner addressing an agent it spawned MUST NOT receive a dead end.
`agent_busy` ceases to be a refusal: the call waits, bounded, for the target to
reach a resumable point, then resumes and confirms per R6.

- Within the bound, the call MUST return a settled outcome: `delivered` or
  `failed`. This is the common case and it is genuinely guaranteed — no queue,
  no dependency on anyone coming back.
- If the bound expires the call MUST return `queued` **together with the
  explicit obligation on the sender** to call `deliver_pending`, and the message
  MUST remain durably queried-able per R4. It MUST NOT be reported as delivered.

Delivery MUST NOT depend on the recipient calling `read_messages` in either
case. The tail is cooperative and is documented as such — the requirement is
that a sender is never misled about which of the two it got, not that a passive
sender still achieves delivery.

Rationale for not requiring more: guaranteeing the tail needs a persistent
dispatcher, i.e. a daemon. This project deliberately has none — each agent runs
its own MCP server and the watcher is a one-shot edge detector. Adding one was
considered and rejected as disproportionate; see Non-goals.

**This requirement is not implementable before R8** — it rides on the same
resume primitive that currently misidentifies its target.

**R7 — No silently unreachable recipient.**
If no delivery path exists for a target in its current state, the call MUST say
so explicitly (e.g. `no_delivery_path`, naming the state) rather than one path
returning success and the other a refusal pointing back at the first.

**R4 — Observable delivery state, both directions.**
Every message carries a status the sender can query — `queued` / `delivered` /
`failed` — with recipient, timestamp, and the sender's idempotency key.
`delivered` is recipient-confirmed per R6. Transport progress is exposed as a
`phase` beneath `queued` (e.g. `phase="sent"`), never as a fourth public status,
so "sent" can never be read as "arrived". A sender MUST be able to answer "did
that land?" without asking the recipient, **and after losing the original
response** — hence the idempotency key, which the sender chooses before the call
rather than receiving from it.

**R5 — No accept-then-drop path reachable as a general-purpose send.**
Once R1 exists, a downstream `send_message` either routes through the guaranteed
path or is refused with a pointer to it.

**Scope of `read_messages`:** it covers **inbox-delivered and upstream messages
only**. Guaranteed-path downstream messages reach the recipient by resume and are
recorded in the sender's delivery/audit store; they deliberately never enter the
actionable inbox, so the recipient cannot re-read them via `read_messages`. That
is what removes the double-presentation risk, and it is the intended trade. If
recipient-visible history for those messages is wanted later, it MUST be a
separate non-actionable query — never an addition to unread delivery.

Within that scope `read_messages` remains for draining
history and re-reading; it stops being how delivery happens.

**R2 — Session resume is downstream-only (accident guard, not authorization).**
`follow_up_agent` MUST reject callers that are not the recorded spawner of the
target, and MUST name the direction rule and point to `send_message`.
Prerequisite: a spawner field MUST be persisted on the agent record, which does
not exist today. The check MUST be documented — in code comment and in the
protocol reference — as preventing accidental upstream resume, **not** as a
security control: identity here is self-asserted and a determined worker can
bypass it. Shipping it is still worthwhile; describing it as a boundary is not.

**R3 — Upstream stays inbox + watcher, and the watcher is load-bearing.**
A spawned agent reaches its lead only via `send_message`. The guarantee comes
from the lead's watcher, not from the lead remembering to poll. The watcher
therefore becomes a protocol component with an obligation: every inbox write
MUST produce an actionable wake within a bounded time, and no message may be
lost between wakes. Today's wake-without-consume and cursor clamping satisfy
this; it needs stating as a contract. `watch` exiting 2 on timeout MUST NOT
strand an unread message.

## Non-goals

- **Authenticating agent identity.** Requires a shared broker, minted
  credentials, and worker-unwritable state — a trust-model redesign. R2 is
  scoped as an accident guard instead.
- **A persistent delivery dispatcher.** The only design that guarantees R1's
  tail for a passive sender, but it means adding a daemon with its own
  lifecycle, locking, and restart recovery to a project built around per-agent
  MCP servers and a one-shot watcher. Rejected as disproportionate; R1 is scoped
  to bounded in-call delivery with a declared cooperative tail instead.
- Caller-identity enforcement on `kill_agent`. Same hazard class and root cause
  as R2, but lifecycle control rather than message delivery. Separate follow-up.
- `resume_session` — takes a `session_id` for session recovery, not an agent
  name; not an agent-to-agent operation.
- Sibling/peer messaging. Undefined until there is a use case; route through the
  shared spawner.
- Changing the watcher's wake-priority or settle-window logic. The
  `before = after` output-edge coupling at `cli.py:280` is load-bearing and out
  of scope.

## Acceptance criteria

1. Lead sends to a worker mid-tool-call **which becomes resumable within the
   call bound**; the worker acts on it, with no `read_messages` call in its
   transcript. The qualifier is required: R1 guarantees delivery within the
   bound, not for a target that never reaches a resumable point.
2. Same, to a Codex worker (which never polls) — delivered.
2b. A target that does not become resumable within the bound → the call returns
   `queued` with the sender obligation stated, the message is queryable by the
   sender's idempotency key, and it is never reported as delivered. A sender
   whose response was lost can recover the outcome by that key.
3. A resume whose prompt does not reach the target's context never returns
   success with a new PID. It returns `failed` when the child is dead with no
   receipt, and a non-terminal in-flight state when the bound expired with the
   child still alive (per R6). The sender can tell the two apart.
4. Three resumes in a row against the same target: three settled or explicitly
   in-flight outcomes. No silent drops, and no two calls sharing a prompt file.
5. A resume against an agent whose stored `backend_session_id` does not match a
   live transcript for that agent fails or re-resolves — it does not attach to a
   different conversation.
6. With a second unrelated Claude session running in the same project dir, a
   spawned agent's session id resolves to its own transcript.
7. A spawned worker calls `follow_up_agent` on its lead → error naming the
   direction rule; the lead's PID is unchanged.
8. A worker calls `follow_up_agent` on an agent it itself spawned → succeeds.
   The rule is parentage, not depth.
9. Target states are distinguished, and none of them produces a false success:
   - dead but still in the registry with a valid backend session → resumed and
     confirmed per R6. This is intentionally supported today
     (`server_simple.py:1767-1778`) and must keep working.
   - killed/removed, or no resumable backend session → explicit error naming the
     state (`no_delivery_path`), never success.
   An earlier draft demanded an error for any "exited" agent, which conflicts
   with the dead-but-resumable lifecycle.
10. Worker `send_message`s the lead mid-turn → the watcher wakes and the message
    is consumed exactly once.

## Suggested sequencing

1. **R8 + R6** — make resume correct and truthful. Everything else is built on
   this, and these are also the highest-value fixes standalone: today a
   misidentified session silently disables both follow-up *and* busy detection.
2. **R2** — persist the spawner and enforce the direction guard. Moved ahead of
   R1 after plan review: queue-on-busy must not exist before the downstream-only
   guard, or a worker can queue an upstream resume to its lead.
3. **R1 + R7 + R4** — guaranteed downstream delivery on the now-trustworthy
   primitive, behind the guard.
4. **R5 + R3** — send-path consolidation and watcher contract.

## Evidence

Source survey of `src/claude_teams/server_simple.py`, `cli.py`, `hooks.py`,
`messaging.py`, `agent_output.py`, and `backends/`. Current protocol behaviour is
documented separately in `docs/reference/agent-messaging-protocol.md`.

The prompt-loss observation is a single field report from one agent, which hedged
its own explanation. Only the observable is relied on — success with a new PID,
silence at the recipient, and a later `send_message` succeeding. The mechanism
described above was derived independently from the source and explains all three
facts; the defects it identifies are worth fixing regardless of whether they
caused that particular incident.

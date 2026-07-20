# Requirement — Pi-lead real-time wake via win-agent-teams extension

Status: **Agreed requirement** (pre-plan). Authors: coordinator + user, 2026-07-19.
Branch: `feature/pi-lead-inbox-wake`.

## 1. Problem statement

When a Pi agent is used as the orchestrator (lead) and a spawned worker sends a
reply — especially one containing a **question** — that reply should *drive the
lead to act*. Today a lead has to remember to block on a watcher
(`win-agent-teams watch …`) to notice incoming messages. That is the wrong place
for the responsibility: the lead should not have to start or babysit a watcher.

The wish (inspired by *Pi-to-Pi Agent-to-Agent Communication* in
`disler/pi-vs-claude-code`) is a real-time signal where the **harness layer**, not
the lead agent's own reasoning loop, wakes the lead with a new turn when a worker
replies.

## 2. Actors and assumptions

- **Lead** = the `pi` session the user starts manually. "The session you start =
  the lead." No external process wraps it.
- **Workers** = already-spawned agents (claude-code / codex / pi) via the
  `win-agent-teams` MCP. Already work; **out of scope** for this requirement.
- **Data channel** = the existing inbox: `send_message` writes
  `inbox-<agent>.jsonl`, `read_messages` reads a delta against a per-sender
  watermark. The MCP server and CLI are the source of truth.

## 3. Chosen solution shape (mechanism)

A **Pi extension/package** (`win-agent-teams` Pi plugin) loaded in the lead session
that:

1. On startup runs a **background watcher** reusing the existing
   `win-agent-teams watch <session_dir>` (semantics `message > output > waiting`,
   settle window, inbox) — unchanged.
2. On wake, **injects a turn** into the running Pi session via Pi's extension API
   (`pi.sendMessage(..., {triggerTurn: true, deliverAs: "steer" | "followUp"})`,
   `steer` when idle for an immediate response, `followUp` when mid-turn) that
   drives the **lead itself** to call `read_messages` and drain its inbox.
3. The data channel (inbox / `send_message` / `read_messages` / MCP) is **entirely
   unchanged**, and the **lead remains the single cursor owner** — the extension
   never advances the cursor. The whole delta is the Pi-side extension plus a small
   read-only CLI surface it shells out to.

This is symmetric with how workers are already woken (`follow_up_agent` /
`backend.send`), except the bridge lives on the Pi side instead of in an external
wrapper.

> **Revision (post plan-review-1):** the Pi extension API exposes tool *metadata*
> only (`getAllTools()`), not callable MCP tools, and provides `pi.exec()` for
> subprocesses. The extension therefore **cannot call `read_messages` itself** —
> it shells out to the CLI and drives the lead to read, keeping the lead as the
> sole cursor writer (a second writer would break the process-local single-writer
> assumption). Step 2 above reflects this; it replaces an earlier draft where the
> extension called `read_messages` directly.

## 4. Functional requirements

- **FR1** A Pi lead started as a bare `pi` session + loaded extension is woken
  automatically when a worker sends a message to the lead's inbox.
- **FR2** The wake is owned by the extension (harness), never by the lead agent's
  loop; the lead never starts a watcher itself.
- **FR3** An incoming worker reply is injected as a new turn so the lead is
  actually driven to act (not merely notified). Idle → immediate turn.
- **FR4** The mechanism reuses the existing watch semantics and `read_messages`
  watermark without re-delivering or dropping messages.

## 5. Non-requirements / scope boundary

- **NR1** No ACP (Agent Client Protocol). Deliberately ruled out, see §7.
- **NR2** No new spawn backend; no change to server or CLI semantics.
- **NR3** No regression for existing claude-code/codex lead or worker flows.
- **NR4** Pi as a worker is unaffected.
- **NR5** Symmetric peer-to-peer topology (Pi coms-net) is out of scope; the
  requirement keeps the hierarchical lead↔worker model.

## 6. Open questions for the plan phase (not requirement-blocking)

- Choice of `steer()` vs `followUp()` per situation, and how "idle" is decided.
- How the extension obtains `session_dir` and `AGENT_NAME` (default `team-lead`).
- Watch CLI as a subprocess vs re-implementing the loop in TS in the extension.
- How a **question** is recognized vs an ordinary status update (whether that
  should affect injection mode/priority).
- Where the Pi extension lives (this repo as a package vs separate) and how it is
  distributed/loaded.
- Whether/how output events (`reason=output`) are handled vs only `message`.

## 7. Investigation behind the rejected options

Fact-check done 2026-07-19:

- **ACP = Agent *Client* Protocol** (Zed) is supported by Pi via `pi-acp`, and
  supports multi-turn (`session/prompt` repeatedly, mapped to Pi `steer`). BUT:
  (a) it is editor↔agent, asymmetric — it requires Pi to run *under* an ACP
  client, which conflicts with "the session you start = the lead"; (b) the current
  adapter is early/community and **"MCP capabilities cannot pass through
  properly"**, which would break our MCP data channel; (c) native `--mode acp` is
  only proposed, not merged. → Wrong role + immature. Kept as an *optional future*
  wake adapter, not the baseline.
- **Pi Event Bus (RFC #2715)** — a design RFC, closed, no code shipped; injecting
  into the LLM context was an open question. Not usable now.
- **Pi coms-net (HTTP+SSE)** — true push but Pi-proprietary; claude/codex cannot
  speak it without a bridge. Out of scope.
- **Pi extension API** — has the primitives we need: `steer()` (immediate turn if
  idle), `followUp()`, `pi.sendMessage()`, lifecycle hooks (`before_agent_start`,
  `turn_start`/`turn_end`). Confirms an extension can *drive* the session, not just
  notify.

Sources:
- disler/pi-vs-claude-code; DeepWiki Multi-Agent Orchestration
- pi-acp adapter (nyanshak/pi-extensions), earendil-works/pi discussion #4444
- earendil-works/pi issue #2715 (Agent Event Bus RFC)
- earendil-works/pi `packages/coding-agent/docs/extensions.md`; DeepWiki
  Extension API & Lifecycle Events

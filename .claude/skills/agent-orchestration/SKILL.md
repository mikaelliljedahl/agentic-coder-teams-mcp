---
name: agent-orchestration
description: Spawn and drive one or more worker agents (Claude Code or Codex) from a coordinator agent (Claude Code or Codex) through the win-agent-teams MCP. Backend-agnostic — works Claude→Codex, Codex→Claude, Codex→Codex, and Claude→Claude. Covers the spawn → watch → read → resume → retire loop, restart recovery, and post-run verification. Use whenever a workflow needs a second agent to review, implement, research, or run a task in a separate process.
---

# Agent orchestration (win-agent-teams MCP)

A **coordinator** agent spawns and drives one or more **worker** agents through the `win-agent-teams` MCP. Each worker is a real, separate OS process (its own model, its own context) that the coordinator talks to over MCP tools — never a subagent inside the coordinator's own process.

This skill is **backend-agnostic**. The coordinator can be Claude Code or Codex; each worker can be Claude Code or Codex. The mechanics below are the same in all four directions; the only thing that changes is **how the coordinator waits** (see the direction matrix). Adapt the *prompt body* to your task (review, implement, research, test) — the *mechanics never change*.

> This is a generic, teachable baseline for the open-source `win-agent-teams` MCP. It contains nothing project-specific — drop in your own repo paths, backends, and prompt bodies.

## Why spawn via the MCP (not a raw CLI in a subagent)

Spawn workers with `spawn_agent`, not by shelling out to `claude ...` or `codex exec ...` from inside a coordinator subagent. A raw CLI worker launched from a harnessed subagent can have its flags blocked, run without the MCP wired in, or silently no-op while reporting "done". The MCP spawner launches a real process with the team MCP injected (so the worker can message back) and a lifecycle you can observe, resume, and kill. That observability is the entire point of the server.

## The direction matrix

|                       | **Worker = Claude Code** | **Worker = Codex** |
|-----------------------|--------------------------|--------------------|
| **Coordinator = Claude Code** | Claude→Claude | Claude→Codex |
| **Coordinator = Codex**       | Codex→Claude | Codex→Codex |

Two axes drive every difference:

- **Who coordinates → how you wait.**
  - **Claude Code coordinator** can idle-wake: run the marker-watch as a **background** command; the harness wakes the coordinator when it exits. Between wakes the coordinator uses no tokens.
  - **Codex coordinator** has *no* idle-wake between turns: run the marker-watch as a **bounded foreground** command *inside the current turn*, looped, until it signals completion. Never end the turn "waiting" — nothing will wake you.
- **Who works → how you reach it.** A spawned worker does **not** reliably poll its own inbox (a Codex worker never does). So `send_message` *to* a worker may go unread — the reliable way to push a follow-up to any worker is **`follow_up_agent`**. Because the worker is otherwise silent and the inbox is pull-only, put an explicit **reporting protocol** in the spawn prompt so the worker signals completion.

Everything else — spawn params, marker-watch, delta reads, terminal kill, restart recovery, output verification — is identical across all four cells.

## Caller-supplied parameters

The invoking workflow provides:

- `<REPO-PATH>` — absolute path used as the worker's `cwd`.
- `<OUTPUT-PATH>` — absolute path the worker must write its deliverable to (e.g. a review/report/summary file).
- `<AGENT-NAME>` — a stable logical name for the worker, reused across follow-ups (e.g. `reviewer-1`, `impl-auth`).
- `<BACKEND>` — `"claude"` or `"codex"` (confirm it exists in pre-flight).
- The prompt body — the task instructions.

## The core loop (every direction)

```
list_backends            # pre-flight: confirm the worker backend exists
spawn_agent(...)         # launch the worker; KEEP the returned handles
── wait ──────────────── # watch the state marker, don't tight-poll
  Claude coordinator:  background  win-agent-teams watch <session_dir> ...
  Codex   coordinator:  foreground win-agent-teams watch <session_dir> ...  (looped, this turn)
── on change ──────────
agent_status / check_agent          # coarse state (cheap; no bodies)
read_messages(from_agent=<name>)    # delta; the DONE/FAILED line lands here
── verify ─────────────
(check the OUTPUT file exists + is real before trusting "done")
── continue / retire ──
follow_up_agent(<name>, prompt)     # resume a dead/idle worker (NOT send_message, NOT a fresh spawn)
kill_agent(<name>)                  # terminal: removes the record for good
── after a coordinator restart, if list_agents is empty ──
session_info() → resume_session('<session_id>')
```

## 1. Pre-flight

Call `mcp__win-agent-teams__list_backends` and confirm `<BACKEND>` is present. If it is absent, **stop and tell the user** — do not silently fall back to running the worker yourself or via a raw CLI. (`list_backends` may report a stale `default_model`; the backend's own config is authoritative — prefer omitting `model` and letting the config default apply.)

## 2. Spawn

Call `mcp__win-agent-teams__spawn_agent` with:

- `backend`: `<BACKEND>` (`"claude"` or `"codex"`)
- `name`: `<AGENT-NAME>`
- `cwd`: `<REPO-PATH>`
- `reasoning_effort` (Codex): `"low" | "medium" | "high"` for the task's difficulty. Omit `model` unless you deliberately need to pin one.
- `permission_mode`: leave default unless the task needs otherwise.
- `prompt`: the task body, then **as the final block, an absolute-path write instruction plus a reporting protocol**:

  ```
  <prompt body>

  WRITE YOUR FULL OUTPUT TO <OUTPUT-PATH>

  REPORTING PROTOCOL (mandatory):
  1. The file above is the deliverable. Your chat output is not read as the report.
  2. When the file is written, call the win-agent-teams MCP tool send_message with
     to="lead" and exactly one line:  DONE: <one-line result/verdict>
  3. If you cannot complete, send_message to="lead":  FAILED: <reason>
  Do not finish without sending one of these two messages.
  ```

Notes:
- The prompt is a structured MCP argument, **not** a shell string — no HEREDOC or quoting concerns; non-ASCII text is safe inline.
- **Always use an absolute path** in the write instruction. A worker's working directory is unreliable; relative paths land in surprising places.
- Spawning is **non-blocking** — the call returns once the process launches, not when the worker finishes. **Keep the returned `state_marker_path`, `session_dir`, `session_id`, and `lead_token`**: the first two drive the watch step; the latter two are your restart-recovery handles.

## 3. Wait — watch, don't tight-poll

Do **not** spin on `check_agent`/`list_agents`. Each worker writes a `state-<agent>.json` marker under its `session_dir` on every lifecycle transition (`{ "state": "running" | "waiting" | ..., "event", "ts" }`); `agent_watch_paths()` rediscovers the paths if you lose them. Block on the marker with the CLI:

```
win-agent-teams watch <session_dir> --timeout 60 --pattern "state-*.json"
```

- **Claude Code coordinator** → run the watch as a **background** command. Its completion wakes you; then call `agent_status`/`check_agent` for the delta. You spend no tokens while it blocks.
- **Codex coordinator** (no idle-wake) → run the watch as a **bounded foreground** command in the **current turn**, looped: on each exit, read the marker JSON (and/or call `agent_status`); if not done and the worker is still alive, watch again. Longer tasks take several rounds — that is normal. Never end the turn in a "waiting" state.

On a change, check liveness with `agent_status` (or `check_agent` for one, `list_agents` for all). These are **compact by default** — a coarse `state` (`running`/`waiting`/`idle`/`dead`) plus `heartbeat_age_s` and `stalled`, with no transcript bodies. Use `stalled: true` to detect "alive but hung". Pass `full=True` only when you actually need `last_message` / `backend_session_id`.

## 4. Read the result

Read the worker's messages with `mcp__win-agent-teams__read_messages` (`from_agent: <AGENT-NAME>`) — this is where the `DONE`/`FAILED` line lands. It is **delta-by-default** with per-sender cursors: it returns `{messages, cursors, seq, unread_count, has_more}` and does **not** re-send history — advance via the returned `seq`/`cursors` (`full=True` returns everything; `limit=0` is a no-body watermark check).

Caveats:
- The inbox is **pull-only** — nothing wakes you when a message arrives. That is why you block on the **marker-watch**, not on the inbox.
- `check_agent.last_message` is only a **truncated tail** of the worker's output (a progress hint), never the deliverable. The deliverable is the **file** (see verification).

## 5. Reach or resume a worker

- **Use `follow_up_agent`, not `send_message`, to reach a worker.** `send_message` *to* a spawned worker may go unread (a Codex worker never polls its inbox). To send clarification or the next iteration to the **same logical worker** (e.g. "fix and re-review", "continue — requirement X is unmet"), use `mcp__win-agent-teams__follow_up_agent` (`name: <AGENT-NAME>`, `prompt: ...`). It preserves the logical worker (so it can reference its own prior work) instead of spawning a fresh one — and is the reliable way to push new input to it.
- A **naturally dead or idle** worker stays listable and resumable via `follow_up_agent`; a **live busy** worker is refused with `agent_busy`. Set `replace_if_idle: true` only when you intentionally want to override an idle run.
- Repeat the **reporting protocol** in every follow-up prompt (with the new deliverable name), so the resumed worker signals completion again.
- `send_message` *is* the direction a **worker** uses to report **to** the coordinator (`to: "lead"`) — that is what the reporting protocol relies on.

## 6. Retire — `kill_agent` is terminal

`mcp__win-agent-teams__kill_agent(name)` is **terminal removal**: it deletes the agent record (the worker disappears from `list_agents`; a later `follow_up_agent` returns `agent_not_found`) and cleans its marker/inbox. It is **not** a pause/park.

- Use it only to retire a worker for good, or to free a name for a clean respawn.
- To *pause and later continue* a worker, do nothing and resume later with `follow_up_agent` — a naturally-dead worker stays resumable until you kill it.
- **Don't kill the MCP server to stop a worker.** On Windows, workers break away from the server's job object and survive server death (the server restarts on the next tool call, markers persist on disk). Stop a worker with `kill_agent`.

## 7. Restart recovery

After the **coordinator's own** process restarts (crash, host reboot, context reset), an unexpectedly **empty `list_agents` does not mean "no workers"**. Recover:

1. Call `mcp__win-agent-teams__session_info()` → `{session_id, identity, cwd, agent_count, lead_token, recoverable_sessions}`.
2. If `recoverable_sessions` is non-empty, adopt the prior session with `mcp__win-agent-teams__resume_session('<session_id>')`; its workers reappear and become resumable via `follow_up_agent`.
3. Retain the `session_id` / `lead_token` from spawn/`session_info` as your stable recovery handle for long-running orchestrations.

A single-coordinator workspace auto-adopts the newest session; if **several coordinators** ran in the same folder, auto-adopt is off and you must pick explicitly via `resume_session`. Session dirs are auto-pruned after a retention window (`WIN_AGENT_TEAMS_RETENTION_DAYS`, default 30); live sessions are never removed.

## 8. Post-run verification (every time, before trusting output)

"Worker reports done" ≠ "deliverable written correctly". Before you read/act on the output:

```
ls -la <OUTPUT-PATH> && head -5 <OUTPUT-PATH>
```

Reject and re-run (via `follow_up_agent`, or a fresh spawn) if any of:

- **File doesn't exist** → the worker hit a path/permission failure, or wrote to the wrong place (relative path). Read its messages for the real error.
- **File is suspiciously small / empty** → the worker aborted mid-run.
- **File doesn't start with the expected content** (e.g. a heading) → the worker emitted chat noise instead of the deliverable.

For implementation tasks, additionally `git diff` and **grep the diff for the change's signature markers** — a green build is not proof the intended change was made. Only after the file passes, read/consume it yourself.

## Anti-patterns (each seen in real runs)

- **Raw CLI worker from a coordinator subagent** → flags blocked / MCP not wired in → silent no-op that claims "completed". Spawn via `spawn_agent`.
- **Codex coordinator ending its turn "waiting"** → nothing wakes it; the run stalls forever. Codex coordinators must watch in a bounded foreground loop within the turn.
- **Tight-polling `check_agent` in a spin loop** → wastes tokens and is server-dependent. Block on the marker-watch.
- **`send_message` to a worker and waiting for a reply** → may never be read (a Codex worker never polls its inbox). Use `follow_up_agent`.
- **Treating `check_agent.last_message` as the report** → it's a truncated tail. The file is the deliverable.
- **Spawn prompt without the reporting protocol** → a silent worker; you can't tell if it finished and the inbox stays empty.
- **Spawning a fresh worker each iteration instead of `follow_up_agent`** → loses continuity; the re-run can't reference its own prior work.
- **`kill_agent` to "pause"** → it's terminal; the worker is gone. Just resume later with `follow_up_agent`.
- **Empty `list_agents` after a restart read as "team gone"** → run `session_info()` → `resume_session(...)` first.
- **Trusting "done" without checking the output file** → the dominant silent-failure mode.

## Tool quick reference

| Tool | Use |
|------|-----|
| `list_backends` | Pre-flight: confirm the worker backend exists. |
| `spawn_agent` | Launch a worker; returns `state_marker_path`, `session_dir`, `session_id`, `lead_token`. |
| `agent_watch_paths` | Rediscover a worker's marker/session paths. |
| `win-agent-teams watch <session_dir>` (CLI) | Block until a state marker changes (background for Claude, foreground-looped for Codex). |
| `agent_status` / `check_agent` / `list_agents` | Compact liveness (`state`, `heartbeat_age_s`, `stalled`); `full=True` for bodies. |
| `read_messages` | Delta inbox read; where `DONE`/`FAILED` lands. Advance via `seq`/`cursors`. |
| `follow_up_agent` | Resume a dead/idle worker with a new prompt (the reliable way to reach a worker). |
| `send_message` | Worker→coordinator reporting channel (`to: "lead"`); unreliable coordinator→worker. |
| `kill_agent` | Terminal removal of a worker. Not a pause. |
| `session_info` / `resume_session` | Restart recovery after the coordinator's process restarts. |

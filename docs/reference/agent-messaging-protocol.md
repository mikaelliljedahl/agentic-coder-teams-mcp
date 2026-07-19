# Agent messaging and lifecycle protocol

Reference for the `win-agent-teams` MCP server as it behaves today. Audience: an
agent that must spawn, message, and retire other agents **without reading the
source**.

This document describes current behavior only. It proposes nothing. Known
defects are stated as facts in [Sharp edges](#sharp-edges).

## How to read the citations

**Symbol names are authoritative; line numbers are not.** Claims cite
`path:line`, but `server_simple.py` in particular churns fast — a verification
pass found 112 of its 113 line citations had drifted by 9–50 lines within a
handful of merged PRs, while every named constant, function, field and value
was still correct. Search for the named symbol rather than jumping to the
number, and treat a line that does not contain what the claim describes as
drift, not as a contradiction.

If a **claim** turns out to be false, that is a defect in this document: fix it
here in the same change that alters the behaviour. A reference that is trusted
and wrong is worse than no reference.

---

## 1. Actors and identity

There are three actors.

| Actor | What it is |
|-------|------------|
| **Lead / spawner** | The process whose MCP client calls `spawn_agent`. May itself be a spawned worker. |
| **Worker** | An OS process (`claude` or `codex`) started by the lead, with its own MCP server instance. |
| **Watcher** | A short-lived CLI process, `win-agent-teams watch <session_dir>`, that blocks until something actionable happens and then exits. |

### Identity is process-global and read once at import

The MCP server module reads three environment variables at import time and
never re-reads them (`src/claude_teams/server_simple.py:49-51`):

- `AGENT_NAME` → `_AGENT_NAME`
- `AGENT_SESSION_ID` → `_AGENT_SESSION_ID`
- `AGENT_PARENT_NAME` → `_AGENT_PARENT_NAME`

`IDENTITY` is `AGENT_NAME` when set, otherwise the constant `"team-lead"`
(`src/claude_teams/server_simple.py:57-58`). `IDENTITY` is *the* inbox name this
process reads (`src/claude_teams/server_simple.py:1397`) and *the* `from` field
stamped on every message it sends (`src/claude_teams/server_simple.py:1326`).

A root lead — one launched by a human, with no `AGENT_NAME` in its environment —
is therefore always `team-lead`.

### How a worker receives its identity

The two backends inject identity differently, because Codex does not propagate
process environment to its MCP servers.

- **claude-code**: the server writes a per-agent MCP config file containing
  `AGENT_SESSION_ID` / `AGENT_NAME` / `AGENT_PARENT_NAME` in its `env` block
  (`src/claude_teams/server_simple.py:1107-1124`), passed as
  `--mcp-config <path>` (`src/claude_teams/backends/claude_code.py:173-175`).
  The same three variables are also set in the child process environment
  (`src/claude_teams/backends/claude_code.py:283-289`).
- **codex**: identity is passed as a per-process config override,
  `-c mcp_servers.win-agent-teams.env={ ... }`
  (`src/claude_teams/backends/codex.py:465-487`). The comment there is explicit
  that writing to the shared `~/.codex/config.toml` would be racy. The same
  variables also go into the child's own environment
  (`src/claude_teams/backends/codex.py:558-570`).

`AGENT_PARENT_NAME` is set from `request.lead_session_id`, which `spawn_agent`
fills with the spawner's `IDENTITY` (`src/claude_teams/server_simple.py:1261`).

### Parentage is recorded only as a routing hint, not as a tree

`agents.json` records no parent field. A spawned agent's record is
`{name, pid, backend, session_id, status, spawned_at, cwd, model,
permission_mode, reasoning_effort, create_token}`
(`src/claude_teams/server_simple.py:1275-1289`). The only parent information
that exists is the `AGENT_PARENT_NAME` env var inside the child process, used to
resolve the `"team-lead"` alias when that child sends a message
(`src/claude_teams/server_simple.py:792-796`).

**All agents in a session share one flat registry.** `AGENT_SESSION_ID` is
propagated verbatim to every descendant, and `_active_session_id` returns it
before doing any recovery (`src/claude_teams/server_simple.py:567-568`). A
worker that spawns its own workers writes into the *same* `agents.json` as the
root lead. There are no sub-sessions.

### Session directory resolution

`_SESSION_BASE` is `~/.claude/agent-sessions`
(`src/claude_teams/server_simple.py:77`); the session dir is
`_SESSION_BASE / <uuid>` (`src/claude_teams/server_simple.py:198-199`).

For a root lead with no `AGENT_SESSION_ID`, the session is recovered via a
binding file keyed by `sha256(identity + parent_pid + cwd)`
(`src/claude_teams/server_simple.py:267-278`), with a cwd+identity fallback that
auto-adopts a single candidate only when this workspace has exactly one bound
session (`src/claude_teams/server_simple.py:555-599`). Otherwise the session is
left unresolved and a `recoverable_sessions` nudge is merged into subsequent
dict-returning tool results (`src/claude_teams/server_simple.py:602-624`).

---

## 2. Tools

`_annotate` merges the recovery nudge into the return value of `spawn_agent`,
`send_message`, `read_messages`, `check_agent`, and `follow_up_agent` only
(`src/claude_teams/server_simple.py:1302`, `1339`, `1504`, `1559`, `1738`).
`kill_agent`, `resume_session`, `session_info`, `list_agents`, `agent_status`,
`agent_watch_paths`, and `list_backends` are not annotated.

Every tool body runs on a worker thread via `run_blocking`
(`src/claude_teams/async_utils.py:7-21`).

`_with_disk_note` appends `_DISK_CONTRACT_NOTE` — the state-marker schema and
the `win-agent-teams watch` recipe — to the *registered* descriptions of
`check_agent`, `list_agents`, `agent_status` and `agent_watch_paths`. That
appended text, not the raw docstring in the source, is what a calling agent
actually reads, so a change there changes the contract those agents see. It must
sit below the `@mcp.tool()` decorator to take effect.

### `spawn_agent(prompt, name, backend, model, cwd, permission_mode, reasoning_effort, expected_outputs)`

**Mechanically**, under the `agents.json` file lock
(`src/claude_teams/server_simple.py:1227`):

1. Creates the session if none exists (`create=True`,
   `src/claude_teams/server_simple.py:1226`).
2. De-duplicates the requested name by appending `-2`, `-3`, …
   (`src/claude_teams/server_simple.py:360-374`). **The name you asked for is
   not necessarily the name you got — read `result["name"]`.**
3. Resolves backend (default `claude-code` when available, else the first
   available, `src/claude_teams/backends/registry.py:92-108`) and resolves
   `(model, effort)` together (`src/claude_teams/server_simple.py:1238`).
4. Generates a fresh per-spawn correlation id (`new_correlation_id`) **before**
   the backend is called, because the id has to be inside the final prompt.
5. Writes the per-agent MCP config, materializes the final prompt and its
   transport (`_materialize_prompt`, see §4), and puts the correlation id, the
   optional prompt sidecar path, and the hook wiring into `SpawnRequest.extra`.
6. Calls `backend.spawn(request)`, which builds argv and starts **a new OS
   process** (`src/claude_teams/backends/process_base.py:82-105`).
7. Captures a PID creation token from the just-live child and appends the
   registry record, including `correlation_id`.

**Returns** `{name, pid, backend, session_id, state_marker_path, session_dir,
watch_argv, watch_command_bash, watch_command_powershell, expected_outputs}`
(`src/claude_teams/server_simple.py:1338-1349`). The three `watch_*` fields are
ready-to-run renderings of the watcher invocation for this session, so a
coordinator does not have to assemble the command itself.

`expected_outputs` is echoed back verbatim and is **not** validated, watched, or
acted upon anywhere.

**What the return does not prove:** only that `CreateProcess` (or equivalent)
succeeded and returned a PID. It does not prove the CLI parsed its argv, that
the model was accepted, that hooks fired, or that the prompt was received. The
first real evidence of life is the appearance of `state-{name}.json`.

**Failure modes** are exceptions, not result fields: `BackendNotRegisteredError`
(`src/claude_teams/backends/registry.py:78-79`),
`BackendBinaryNotFoundError` (`src/claude_teams/backends/claude_code.py:121`),
`BackendModelUnavailableError` for a Codex tier whose GPT-5.6 model this install
does not expose (`src/claude_teams/backends/codex.py:237-246`) — there is no
silent downgrade.

### `send_message(text, to="team-lead")`

Appends one JSON line to `inbox-{recipient}.jsonl`
(`src/claude_teams/server_simple.py:1324-1333`). The line is
`{"from": IDENTITY, "text": ..., "ts": <ISO-8601 UTC>}`.

Recipient resolution (`src/claude_teams/server_simple.py:777-809`):

- Any of `""`, `team-lead`, `lead`, `orchestrator`, `parent`, `boss`, `manager`,
  `up`, `supervisor` (case-insensitive, `src/claude_teams/server_simple.py:63-75`)
  resolves to `AGENT_PARENT_NAME` for a worker, or to `team-lead` for the root
  lead.
- A name present in `agents.json` is used verbatim.
- **Anything else is silently rerouted to the lead** with a `warning` string in
  the result. A typo'd recipient never fails — it lands in the lead's inbox.

**Returns** `{"success": true, "to": <resolved recipient>}` plus optional
`warning`, or `{"success": false, "to": ..., "reason": "session_not_found"}`
when no session resolved (`src/claude_teams/server_simple.py:1322`).

**`success: true` proves only that a line was appended to a file.** There is no
delivery, no push, no wake of the recipient process. The recipient sees it only
if it later calls `read_messages`. The docstring states this outright
(`src/claude_teams/server_simple.py:1314-1316`).

### `read_messages(from_agent, since_seq, full, limit, max_chars)`

Reads **the caller's own** inbox — `inbox-{IDENTITY}.jsonl` — never anyone
else's (`src/claude_teams/server_simple.py:1397`). Serialized per inbox name by
an in-process lock only; there is deliberately no cross-process file lock
(`src/claude_teams/server_simple.py:245-260`, `1399`).

Mechanics per call:

1. Load cursors; group inbox lines by `from`, keeping global file position
   (`src/claude_teams/messaging.py:37-61`). Malformed lines, non-dict entries,
   and entries without a non-empty string `from` are skipped silently.
2. **Clamp** every stored cursor down to the observed message count for that
   sender, including senders absent from the current snapshot (clamped to 0)
   (`src/claude_teams/server_simple.py:1413-1415`).
3. Select unread entries, sort by global file order
   (`src/claude_teams/server_simple.py:1443-1448`), clip to `limit` (default 50
   unless `full=True`, `src/claude_teams/server_simple.py:1440-1442`).
4. Advance and persist cursors (`src/claude_teams/server_simple.py:1458-1475`).

**Returns** `{messages, cursors, seq, unread_count, has_more}`. Each message is
`{from, text, ts, seq}` where `seq` is the sender's 1-based per-sender count —
plus `truncated` and `full_len` on each message when `max_chars` is set.
`unread_count` is measured *before* limit clipping, so it is the true backlog,
not the returned batch size (`src/claude_teams/server_simple.py:1449-1452`).
`cursors` is the map when `from_agent` is unset; `seq` is a scalar and `cursors`
is `None` when `from_agent` is set (`src/claude_teams/server_simple.py:1496-1501`).

Preconditions and refusals: `since_seq` without `from_agent` raises `ValueError`
(`src/claude_teams/server_simple.py:1379-1381`); negative `limit` raises
`ValueError` (`src/claude_teams/server_simple.py:1382-1384`). `limit=0` is a
non-consuming watermark peek. With no session, an empty result is returned
rather than an error (`src/claude_teams/server_simple.py:1389-1396`).

**This call is destructive by default** — it advances the persisted cursor, so
messages are drained. Use `limit=0` to peek.

### `check_agent(name, full=False, max_chars=200)`

Reads the registry record, does a liveness check, and scans the backend rollout
log for the agent's last assistant message
(`src/claude_teams/server_simple.py:1540-1548`). Persists a newly discovered
`backend_session_id` back into `agents.json` as a side effect
(`src/claude_teams/server_simple.py:1546-1547`).

**Returns** `{name, state, alive, pid, backend, last_activity_at, unread_count,
last_line, seq, truncated, full_len, heartbeat_age_s, stalled}`
(`src/claude_teams/server_simple.py:1061-1075`). `full=True` adds
`last_message` and `backend_session_id`
(`src/claude_teams/server_simple.py:1550-1556`).

`state` resolution precedence (`src/claude_teams/server_simple.py:154-182`):

1. Not alive → `"dead"` (liveness gates everything; a stale `running` marker
   never wins).
2. Marker state in `{"running", "waiting"}` → that value.
3. Otherwise activity recency: `< _idle_seconds()` → `"running"`, else
   `"idle"`. `_idle_seconds()` defaults to 60 s, overridable via
   `WIN_AGENT_TEAMS_IDLE_SECONDS` (`src/claude_teams/server_simple.py:104-112`).

`stalled` is `True` only when alive, state is neither `waiting` nor `dead`, and
heartbeat age exceeds `WIN_AGENT_TEAMS_STALL_SECONDS` (default 300 s)
(`src/claude_teams/server_simple.py:118-148`).

**`unread_count` and `seq` count messages *from* this agent *to the caller***,
read out of the caller's own inbox — not the agent's inbox
(`src/claude_teams/server_simple.py:1051-1052`, `1020-1031`).

An unknown agent name returns a stable all-zero payload with `state: "dead"`,
`alive: false` — **not** an error (`src/claude_teams/server_simple.py:883-903`,
`1543`). You cannot distinguish "never existed" from "killed" from this tool.

### `follow_up_agent(name, prompt, replace_if_idle=True)`

See [section 4](#4-follow_up_agent-in-detail).

### `kill_agent(name)`

Terminal and destructive (`src/claude_teams/server_simple.py:1782-1798`):

1. Signals the OS process **only** when `owns_process` proves it is still ours —
   live in-memory ownership or a matching creation token
   (`src/claude_teams/server_simple.py:1790-1793`,
   `src/claude_teams/backends/process_manager.py:253-267`). A reused or foreign
   PID is left running.
2. Removes the record from `agents.json`.
3. Deletes `state-{name}.json`, `prompts/{name}.prompt.txt`,
   `inbox-{name}.jsonl`, and `inbox-{name}.pos.json`
   (`src/claude_teams/server_simple.py:1749-1755`) — **any unread messages
   addressed to that agent are destroyed**.
4. Deletes the killed agent's sender entry from the caller's own cursor file
   (`src/claude_teams/server_simple.py:1759-1764`).

**Returns** `{"success": true, "name": name}`; `{"success": false, "name":
name}` with **no `reason` field** when the agent is not in the registry
(`src/claude_teams/server_simple.py:1788`); `{"success": false, "name": name,
"reason": "session_not_found"}` with no session.

`success: true` does not prove the process died — step 1 is conditional.

A naturally-dead agent is **not** auto-removed; it stays listable and resumable
until killed (`src/claude_teams/server_simple.py:1777-1778`).

### `resume_session(session_id)`

Re-binds this lead to a prior session id after a restart
(`src/claude_teams/server_simple.py:1817-1846`). Validates the id is UUID-shaped
(blocking path traversal) and that its directory resolves directly under
`_SESSION_BASE` (`src/claude_teams/server_simple.py:1825-1834`).

**Returns** `{success, session_id, agent_count, lead_token}`, or `{success:
false, reason}` where `reason` is `session_id_required`, `invalid_session_id`,
or `session_not_found`.

### `session_info()`

Returns `{session_id, identity, cwd, agent_count, lead_token,
recoverable_sessions}` (`src/claude_teams/server_simple.py:1882-1889`).
`recoverable_sessions` excludes the current session
(`src/claude_teams/server_simple.py:1866-1868`) and lists only bindings whose
`agents.json` still holds a non-`killed` agent within the retention window
(`src/claude_teams/server_simple.py:466-521`).

### `list_agents(full=False)`

Compact rows `{name, state, alive, pid, backend, last_activity_at,
unread_count}` (`src/claude_teams/server_simple.py:1916-1924`). `full=True`
returns the raw registry record plus `last_line`, `truncated`, `full_len`
(`src/claude_teams/server_simple.py:1954-1967`). Empty list when no session
resolved (`src/claude_teams/server_simple.py:1945-1946`) — indistinguishable
from a session with no agents; call `session_info()` to tell them apart.

### `agent_status(names=None)`

The cheap path. Rows are exactly `{name, state, last_activity_ts, unread_count,
seq, heartbeat_age_s, stalled}` (`src/claude_teams/server_simple.py:1990-1998`).
Cost is one marker read + one cursor read + one liveness check per agent; a
rollout-log scan happens **only** as a fallback when no marker exists
(`src/claude_teams/server_simple.py:1977-1981`). Unknown names are silently
skipped (`src/claude_teams/server_simple.py:2036-2038`).

### `agent_watch_paths(names=None)`

Returns `{has_session, session_dir, watch_argv, watch_command_bash,
watch_command_powershell, agents: [{name, state_marker_path}]}`
(`src/claude_teams/server_simple.py:2121-2135`) — no longer a bare list. Pure
path computation: it does not check that any file exists. Use it to rediscover
what to watch when a `spawn_agent` return was not retained; the `watch_*` fields
give the same ready-to-run command `spawn_agent` returns.

### `list_backends()`

`[{name, binary, default_model, supported_models}]` for backends whose binary is
on `PATH` (`src/claude_teams/server_simple.py:2088-2100`,
`src/claude_teams/backends/registry.py:82-90`).

---

## 3. On-disk contract

Everything lives under `~/.claude/agent-sessions/<session-uuid>/`
(`src/claude_teams/server_simple.py:77`, `198-199`).

| Path | Written by | Read by |
|------|-----------|---------|
| `agents.json` | server, under an exclusive file lock (`src/claude_teams/server_simple.py:316-346`) | server |
| `agents.lock` | lock file only (`src/claude_teams/server_simple.py:81`, `206-207`) | server |
| `session.json` | server, on binding (`src/claude_teams/server_simple.py:423-426`) | server recovery |
| `inbox-{name}.jsonl` | any agent's `send_message` (append) (`src/claude_teams/server_simple.py:1332-1333`) | the owner's `read_messages`; watcher (`src/claude_teams/cli.py:223`) |
| `inbox-{name}.pos.json` | the owner's `read_messages` (`src/claude_teams/server_simple.py:1475`) | owner; watcher (read-only) |
| `state-{name}.json` | the **worker's own** lifecycle hook process (`src/claude_teams/hooks.py:88-89`) | server status tools; watcher |
| `prompts/{name}.prompt.txt` | server, before a claude-code spawn/resume (`src/claude_teams/server_simple.py:1135-1137`) | the worker itself, as a file read |
| `mcp/{name}.mcp.json` | server (`src/claude_teams/server_simple.py:1122-1123`) | the claude CLI |
| `hooks-{name}.settings.json` | server (`src/claude_teams/hooks.py:147-148`) | the claude CLI |
| `codex-hook-{name}.cmd` | server, Windows only (`src/claude_teams/hooks.py:182-184`) | Codex's hook runner |

Two directories are created eagerly at session creation, `mcp/` and `logs/`
(`src/claude_teams/server_simple.py:1100-1101`).

### Inbox format

One JSON object per line, appended:
`{"from": <sender IDENTITY>, "text": <str>, "ts": <ISO-8601 UTC>}`
(`src/claude_teams/server_simple.py:1325-1331`). Append-only; nothing ever
rewrites or compacts it. The file is deleted only by `kill_agent`.

### Cursor semantics

`inbox-{name}.pos.json` is a **per-sender consumed-count**, not a byte or line
offset: `{"<sender>": <int count of that sender's messages already read>}`.

- Loading discards non-string keys, non-int values, booleans, and negatives
  (`src/claude_teams/messaging.py:19-26`).
- Saving is atomic via a uniquely named temp file and `replace`
  (`src/claude_teams/messaging.py:29-34`).
- **Clamping** happens on every read: a stored count larger than the observed
  message count for that sender is reduced to the observed count, including
  senders with zero observed messages
  (`src/claude_teams/server_simple.py:1413-1415`; also in
  `unread_sender_counts`, `src/claude_teams/messaging.py:70-71`). This is what
  prevents a corrupt forward cursor from permanently swallowing a sender's
  future messages.
- Unread for a sender is `len(messages) - min(cursor, len(messages))`, and only
  positive counts are returned (`src/claude_teams/messaging.py:64-75`).

### State marker schema

`{"state": "running" | "waiting", "event": "<hook event name>", "ts": <float
epoch seconds>}` (`src/claude_teams/hooks.py:88`). Written atomically via a
temp file + `replace` (`src/claude_teams/hooks.py:53-58`).

Readers tolerate a missing or corrupt file by returning `None`
(`src/claude_teams/server_simple.py:233-242`, `src/claude_teams/cli.py:162-165`).
A marker state outside `{"running", "waiting"}` is treated as absent and falls
through to the activity heuristic (`src/claude_teams/server_simple.py:151`,
`174-177`).

### Backend rollout logs (outside the session dir)

The server does not capture agent transcripts. It reads the CLIs' own logs.

- **claude-code**: `~/.claude/projects/<cwd with `\`/`:` replaced by `-`>/*.jsonl`.
  Matched by `backend_session_id` when known, else by start time, then narrowed
  to the transcript carrying the agent's correlation marker when one is
  persisted (`read_claude_output`).
- **codex**: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`. Matched by the
  rollout's `session_meta` `cwd` plus start time, then narrowed the same way
  (`read_codex_output`, `_matching_codex_rollouts`).

Both readers fall back to the unnarrowed candidate set when no log carries the
marker yet — e.g. the CLI has not flushed the prompt.

#### Correlation ids

The correlation id is **generated once per spawn** (`new_correlation_id`) and
persisted as `correlation_id` on the agent record. It is deliberately *not*
derived from agent name + session id: a killed agent's name can be reused once
its record is removed, so a derived token could identify two different
conversations.

`_read_agent_output` loads the persisted id and passes
`correlation_marker_token(id)` to both readers. It is **never re-derived** when
the record does not carry one. `classify_correlation` distinguishes:

| Record state | Classification | Meaning |
|---|---|---|
| `correlation_id` key absent | `legacy` | Record predates correlation. Compatibility case. |
| Present but empty/blank/wrong type | `unverified` | Corrupt. |
| Present, non-blank string | `valid` | Bindable. |

Both non-`valid` cases currently read as if no marker existed, which reproduces
the pre-correlation behaviour. Consumers that must act differently on the two
(notably the follow-up gate) land with the validation ladder.

The id survives resume: `follow_up_agent` reads it off the record, passes it
back in `SpawnRequest.extra`, and writes it back on the updated record. Dropping
it there would silently reclassify a live agent as `legacy`.

`last_message` is the last assistant message, capped at 1000 chars with a
truncation marker (`src/claude_teams/agent_output.py:19`, `386-407`).
`last_activity_at` is the rollout file's mtime, not a semantic timestamp
(`src/claude_teams/agent_output.py:67`, `71-72`).

Separately, per-agent stdout/stderr logs go to
`~/.claude/teams/<session-id>/logs/<agent>.log`, overridable with
`WIN_AGENT_TEAMS_LOG_DIR` (`src/claude_teams/backends/process_manager.py:698-707`).

---

## 4. `follow_up_agent` in detail

### It starts a NEW OS PROCESS

This is the single most important fact about this tool. `follow_up_agent` does
**not** write to the running agent's stdin, does not attach to its terminal, and
does not send it a message. It calls `backend.resume(...)`
(`src/claude_teams/server_simple.py:1703`), which builds a resume command and
hands it to `process_manager.spawn_process`
(`src/claude_teams/backends/process_base.py:89-93`) — a brand-new process, with
a new PID recorded over the old one
(`src/claude_teams/server_simple.py:1710-1726`).

Continuity comes purely from the backend CLI's own `--resume` /
`resume <id>` mechanism replaying the prior conversation
(`src/claude_teams/backends/claude_code.py:193-196`,
`src/claude_teams/backends/codex.py:387-404`).

### The liveness gate

Evaluated in this order, in `_do_follow_up`:

| Order | Condition | Refusal reason |
|-------|-----------|----------------|
| 1 | No session resolved | `session_not_found` |
| 2 | `name` not in `agents.json` | `agent_not_found` |
| 3 | Backend not loadable, or `supports_resume()` false | `backend_not_supported` |
| 4 | No `backend_session_id` known | `backend_session_missing` |
| — | *`idle_by_marker` resolved here; a `waiting` marker skips 5–7* | |
| 5 | Alive, **not idle by marker**, and `last_message is None` | `agent_busy` |
| 6 | Alive, **not idle by marker**, and `last_activity_at is None` | `agent_state_unknown` |
| 7 | Alive, **not idle by marker**, and last activity < 60 s ago | `agent_busy` |
| 8 | Alive, judged idle, and `replace_if_idle=False` | `agent_idle_but_alive` |
| 9 | `backend.resume(...)` raised | `resume_failed` |

Checks 5–7 all sit **behind** the marker resolution. An earlier revision of this
document listed 5 and 6 as unconditional on `alive`, and carried an extra row for
an `output.busy_hint` check; that field and its branch no longer exist, and the
marker now precedes the transcript-derived checks rather than following two of
them.

Both `claude-code` and `codex` return `supports_resume() == True`
(`src/claude_teams/backends/claude_code.py:113-115`,
`src/claude_teams/backends/codex.py:252-254`), so reason 3 in practice means the
backend binary vanished from `PATH`.

**The authoritative signal is the `waiting` marker.** `idle_by_marker` is
computed by re-running `_resolve_agent_state` and testing for `"waiting"`. When
it is true, checks 5, 6 and 7 — the entire transcript-derived heuristic block —
are skipped. The comment above it explains why the ordering matters: an agent
parked at a Stop hook before emitting any assistant text has no `last_message`,
so checking that first reported `agent_busy` for precisely the case we know is
idle.

The inactivity heuristic (check 8) compares against the module constant
`_FOLLOW_UP_IDLE_SECONDS = 60.0` **directly**
(`src/claude_teams/server_simple.py:85`, `1642`) — not via `_idle_seconds()`, so
`WIN_AGENT_TEAMS_IDLE_SECONDS` does **not** affect this gate even though it does
affect the `state` field reported by `check_agent` / `agent_status`.

Failure payloads carry `{success: false, name, reason}` plus `alive`,
`backend_session_id`, `last_activity_at`, and the unbounded `last_message` when
a status was computed (`src/claude_teams/server_simple.py:1078-1094`).

### Kill-and-respawn under `replace_if_idle`

When the agent is alive and judged idle and `replace_if_idle=True` (the
default), the existing process is shut down before the resume
(`src/claude_teams/server_simple.py:1656-1659`):

```
if process_manager.owns_process(str(pid), _agent_create_token(agent)) \
   and not process_manager.graceful_shutdown(str(pid), timeout_s=5.0):
    process_manager.kill_process(str(pid))
```

The **creation-token ownership check fails closed**
(`src/claude_teams/backends/process_manager.py:253-267`): it returns `True` only
when this manager still holds a live in-memory child for that handle, or the
live PID's creation token equals the stored one. A tokenless recovered record,
an unreadable token, or a mismatch all yield `False`, and the process is left
untouched — the resume then proceeds anyway against `backend_session_id`,
potentially leaving the old process running.

Tokens are captured at spawn time from the just-live child
(`src/claude_teams/server_simple.py:1269-1273`) and re-captured after every
resume (`src/claude_teams/server_simple.py:1711`).

### `backend_session_id` scraping

The server never receives a session id from the backend. It **scrapes** it out
of the CLI's rollout log:

- claude-code: the first line's `sessionId` field
  (`src/claude_teams/agent_output.py:286-304`, `109`).
- codex: the `session_meta` record's `payload.id`
  (`src/claude_teams/agent_output.py:134-140`).

`_sync_backend_session_id` persists it onto the registry record the first time
it is observed, and both `check_agent` and `follow_up_agent` write the registry
back when it changes — `follow_up_agent` at every one of its refusal exits, so a
newly discovered id is not lost when the call goes on to refuse
(`src/claude_teams/server_simple.py:1546-1547`, `1603`).

Consequence: **`follow_up_agent` fails with `backend_session_missing` until the
target CLI has flushed enough of its rollout file to be discovered.** A freshly
spawned agent is not immediately followable-up.

### Prompt materialization and transport (claude-code)

The **server** builds the final prompt string and chooses its transport
(`_materialize_prompt`), because the backend cannot do it: the sidecar is
written before `backend.spawn`/`backend.resume`, and
`ClaudeCodeBackend._prompt_arg` then replaces argv with a fixed instruction.

Two rules, both load-bearing:

1. **Transport is decided from the user prompt alone.** `_needs_prompt_file`
   tests the *unmarked* user prompt against `_CLAUDE_PROMPT_FILE_CHARS`
   (`'`, `"`, `\n`, `\r`) **before** the correlation marker is appended. Testing
   the marked prompt would route every Claude spawn through a file read, since
   the multi-line marker form always adds a newline.
2. **The marker form differs per transport.** The argv branch appends a
   **single-line** marker joined by a space, which introduces no CLI-sensitive
   character — so the safety rule is respected, not bypassed. The sidecar branch
   appends the newline-delimited form. `_CLAUDE_PROMPT_FILE_CHARS` is unchanged.

When the sidecar is used, the server writes the **marked** prompt to
`prompts/{name}.prompt.txt`, passes `prompt_file_path` in `extra`, and the
backend puts only this fixed instruction on the command line
(`src/claude_teams/backends/claude_code.py:258-271`):

> `Read your complete task prompt from UTF-8 file path <path> then follow the file contents exactly.`

**The worker must actually read that file.** The real prompt never reaches the
model directly — and neither does the marker, until the read lands in context.

This applies identically to `spawn_agent` and `follow_up_agent`, and the file
path is the same for both — a follow-up **overwrites** the spawn prompt file.

Codex has no prompt-file path: `_materialize_prompt` returns the prompt
unchanged for any non-`claude-code` backend. Codex relies on passing the prompt
as a verbatim argv token via the native `.exe`, falling back to a JSON-encoded
single-line form only when it is forced through the `codex.cmd` npm shim
(`src/claude_teams/backends/codex.py`, `_prompt_arg`). Its marker is appended by
`CodexBackend._correlated_prompt`, which **consumes** the server-issued id from
`extra["correlation_id"]` rather than deriving one — so a Codex prompt carries
exactly one marker, never two. With no id in `extra` the prompt goes out
unmarked; no id is ever invented.

---

## 5. The watcher

`win-agent-teams watch <session_dir> [--timeout 60] [--pattern state-*.json] [--inbox/--no-inbox]`
(`src/claude_teams/cli.py:182-199`).

### Reader identity

The watcher determines whose inbox to watch from its **own process
environment**: `AGENT_NAME`, falling back to `team-lead`
(`src/claude_teams/cli.py:222`). It watches `inbox-{reader}.jsonl` and
`inbox-{reader}.pos.json` (`src/claude_teams/cli.py:223-224`). Both backends do
set `AGENT_NAME` in the child environment, so a worker running the watcher gets
its own inbox.

### Poll loop

- Poll interval: **0.5 s** (`src/claude_teams/cli.py:19`, `309`).
- Default timeout: **60 s** (`src/claude_teams/cli.py:186`).
- File identity is `(mtime_ns, size)`, not bare mtime, so a same-tick atomic
  replace that changes size is still detected
  (`src/claude_teams/cli.py:109-126`).

### Wake priority: message > output > waiting

1. **Message** is checked at the top of each iteration, and once before the loop
   starts against the current unread state (`src/claude_teams/cli.py:227-237`,
   `254-264`).
2. **Output** is emitted next (`src/claude_teams/cli.py:292-294`).
3. **Waiting** is emitted last, and only after settling
   (`src/claude_teams/cli.py:300-306`).

### The actionable-edge rule

A changed `state-*.json` wakes the watcher only when its `state` is `"waiting"`
**and** its `event` is not in `_NON_ACTIONABLE_WAITING_EVENTS`
(`src/claude_teams/cli.py:150-174`). That set contains exactly `SubagentStop`
(`src/claude_teams/cli.py:26`) — an agent's *own* built-in Task subagent
finishing, while the agent itself is still mid-task.

`running` markers are never actionable. A marker with no `event` field, or a
non-string `event`, is treated as actionable for backward compatibility
(`src/claude_teams/cli.py:168-173`).

### Settle window

An actionable waiting marker is registered with its first-seen monotonic time
(`src/claude_teams/cli.py:284-285`) and only wakes once it has **persisted** as
waiting for `_WATCH_SETTLE_SECONDS` (`src/claude_teams/cli.py:304`). Each marker
path settles independently, so overlapping waits across several agents do not
drop each other (`src/claude_teams/cli.py:239-244`). A candidate that flips back
to `running` inside the window is dropped
(`src/claude_teams/cli.py:301-303`).

Default is **15.0 s** (`_WATCH_SETTLE_DEFAULT_SECONDS`), overridable via
`WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS`
(`src/claude_teams/cli.py:28`, `31-56`). A malformed, `NaN`, infinite, or
negative override falls back to the default; `0` is a valid value meaning
"settle disabled".

### Wake-without-consume for the inbox

The watcher calls `unread_sender_counts`, which is explicitly documented as *not*
advancing cursors (`src/claude_teams/messaging.py:64-65`) and never calls
`save_inbox_cursors`. **A `reason="message"` wake leaves the message unread; you
must still call `read_messages`.** Consequently, an unread message that is never
drained will re-wake every subsequent watch immediately.

### Exit codes and output

- **Exit 0** — one JSON object printed on stdout
  (`src/claude_teams/cli.py:177-179`), one of:
  - `{"reason":"message","from":[<senders>],"path":"<inbox path>"}`
  - `{"reason":"output","path":"<changed path>"}`
  - `{"reason":"waiting","agent":"<name>","path":"<marker path>"}`
- **Exit 2** — deadline reached, **nothing printed**
  (`src/claude_teams/cli.py:307-308`). Exit 2 does not mean nothing happened:
  the docstring notes a waiting transition may precede the initial snapshot, or
  a genuine edge may still be inside its settle window at the deadline
  (`src/claude_teams/cli.py:214-216`). Re-check status after exit 2.

### `reason="output"` requires a non-default `--pattern`

Classification splits changed paths: anything named `state-*.json` goes to the
waiting bucket, everything else to outputs (`src/claude_teams/cli.py:269-276`).
The default pattern is `state-*.json` (`src/claude_teams/cli.py:20`), so with
default arguments the outputs bucket is **always empty** and `reason="output"`
can never fire. It only becomes reachable with a widened `--pattern`.

---

## 6. Lifecycle hooks

The marker is written by a hook command the server injects into the worker's
CLI configuration. The hook runs `python -m claude_teams.hooks emit
--session-dir <dir> --agent <name>` and reads the event JSON from stdin
(`src/claude_teams/hooks.py:92-112`, `297-304`).

### Event → state mapping

| Hook event | Marker state |
|-----------|--------------|
| `SessionStart` | `running` |
| `UserPromptSubmit` | `running` |
| `PreToolUse` | `running` |
| `PostToolUse` | `running` |
| `Stop` | `waiting` |
| `SubagentStop` | `waiting` |

(`src/claude_teams/hooks.py:19-22`, `44-50`.)

Any unrecognized event name writes **nothing** and leaves the prior marker
intact, as does empty, non-JSON, or non-dict stdin
(`src/claude_teams/hooks.py:61-89`).

Note the asymmetry that the watcher exists to handle: `SubagentStop` writes a
`waiting` marker on disk, but the watcher filters it out as non-actionable
(section 5). Status tools do **not** filter it — `check_agent` and
`agent_status` will report `state: "waiting"` for a `SubagentStop` marker,
because `_resolve_agent_state` looks only at `state`, never at `event`
(`src/claude_teams/server_simple.py:154-182`).

### Wiring per backend

- **claude-code**: a settings file `hooks-{name}.settings.json` is written with
  every event mapped to the emit command
  (`src/claude_teams/hooks.py:136-149`), passed as `--settings <path>`
  (`src/claude_teams/backends/claude_code.py:242-256`). Suppressed by
  `WIN_AGENT_TEAMS_STATE_HOOKS=0`.
- **codex**: a list of `-c hooks.<Event>=[...]` overrides
  (`src/claude_teams/hooks.py:188-227`), plus
  `--dangerously-bypass-hook-trust` — without which Codex silently skips
  injected hooks (`src/claude_teams/backends/codex.py:407-437`). Suppressed by
  `WIN_AGENT_TEAMS_STATE_HOOKS=0` or `WIN_AGENT_TEAMS_STATE_HOOKS_CODEX=0`
  (`src/claude_teams/backends/codex.py:439-449`).
- **codex on Windows** additionally needs a `.cmd` launcher referenced as a
  single bare path in `commandWindows`, because Codex's `cmd /C` argv escaping
  corrupts a multi-token quoted command
  (`src/claude_teams/hooks.py:152-185`, `236-265`).

Hook wiring is regenerated on **every** spawn and every follow-up
(`src/claude_teams/server_simple.py:1249`, `1685`).

---

## Sharp edges

Facts about current behavior. No remedies are proposed here.

### A busy worker is reachable by neither tool

`send_message` writes to a file and nothing more; the docstring states the agent
sees it only if it calls `read_messages` (its own docstring says so).
`follow_up_agent` refuses a busy agent with `agent_busy` — two sites in
`_do_follow_up`, the no-`last_message` check and the inactivity timer, both
behind the marker resolution. There is therefore
no mechanism at all — no push, no stdin write, no signal — to reach a worker
that is mid-task and not polling. Coordination with a busy worker is possible
only after it parks (marker `waiting`) or voluntarily polls.

### Codex workers do not poll `read_messages`

Nothing in the codex spawn path arranges for polling. The prompt is the caller's
text plus the server-issued correlation marker
(`CodexBackend._correlated_prompt`); the environment is
`AGENT_NAME` / `AGENT_SESSION_ID` / `AGENT_PARENT_NAME` only
(`src/claude_teams/backends/codex.py:566-570`). There is no Codex equivalent of
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, which the claude-code backend sets to
put its workers in Claude Code's native team-messaging mode
(`src/claude_teams/backends/claude_code.py:284-285`). Messages sent to a Codex
worker's inbox therefore accumulate unread unless the caller's own prompt text
instructs the agent to poll.

### `follow_up_agent` has no caller-identity check

The tool looks the target up by name in the session-wide `agents.json`
(`src/claude_teams/server_simple.py:1586`) and never compares the target against
`IDENTITY` or `_AGENT_PARENT_NAME`. Because all descendants share one flat
registry (section 1), **any agent can kill-and-respawn any other agent in the
session, including its own lead**, provided that lead is itself a spawned agent
record. The same holds for `kill_agent`, which performs no caller check either
(`src/claude_teams/server_simple.py:1786`).

### `before = after` is load-bearing and ordering-sensitive

At `src/claude_teams/cli.py:280` the snapshot is advanced unconditionally —
including for ignored `running` and corrupt markers — so a non-ready write
cannot be rediscovered forever. Because that assignment has already consumed the
current output edge, the outputs emit **must** come before the settled-wait
emit; a wait matures over several polls and can settle in the same tick an
output arrives. Waking on the wait first would drop that output permanently.
The code carries this reasoning inline
(`src/claude_teams/cli.py:277-291`). Reordering these three statements silently
loses wake events without failing loudly.

### The only non-marker busy heuristic is a 60-second timer

When no `waiting` marker is available, busy/idle rests entirely on
`_FOLLOW_UP_IDLE_SECONDS` — was there transcript activity in the last 60 s.
There is no signal derived from what the agent is actually doing.

(A vestigial `AgentOutput.busy_hint` field once suggested otherwise. It was
never set `True` by any code path, so its branch was unreachable; both were
removed. Any document or memory describing it is stale.)

### `WIN_AGENT_TEAMS_IDLE_SECONDS` does not affect the follow-up gate

It is read by `_idle_seconds()` (`src/claude_teams/server_simple.py:104-112`),
which is used only by `_resolve_agent_state`
(`src/claude_teams/server_simple.py:180`). The `follow_up_agent` inactivity
check uses the bare constant (`src/claude_teams/server_simple.py:1642`). Tuning
the env var changes reported `state` but not who gets refused as `agent_busy`.

### `kill_agent` destroys the target's unread inbox

`inbox-{name}.jsonl` and its cursor are deleted outright
(`src/claude_teams/server_simple.py:1751-1752`). Any message sent to that agent
and not yet read is gone.

### A follow-up overwrites the spawn prompt file

`_prompt_file` is keyed by agent name only
(`src/claude_teams/server_simple.py:228-230`), so a claude-code follow-up whose
prompt contains a quote or newline rewrites the same
`prompts/{name}.prompt.txt` the original spawn used. If the worker re-reads that
path later, it sees the follow-up prompt, not the original.

### A spawned agent may not have the name you requested

`_unique_agent_name` appends `-2`, `-3`, … on collision and truncates the base to
fit 64 characters (`src/claude_teams/server_simple.py:360-374`). Kill does free
the name for reuse, since it removes the record
(`src/claude_teams/server_simple.py:1794-1795`).

### An unknown recipient is silently rerouted, not rejected

`send_message` to a name that is neither a lead alias nor a known agent returns
`success: true` with the message written to the *lead's* inbox and only a
`warning` field to indicate it
(`src/claude_teams/server_simple.py:805-809`). Callers that check only
`success` will not notice.

### `check_agent` cannot distinguish "unknown" from "dead"

An unrecognized name returns the same `state: "dead"`, `alive: false` payload as
a genuinely dead agent (`src/claude_teams/server_simple.py:883-903`, `1543`).

### Inbox cursor writes are not cross-process locked

The lock guarding load/advance/save is a `threading.Lock` scoped to one Python
process, deliberately so (`src/claude_teams/server_simple.py:245-249`). The
invariant relied upon is that only the inbox owner's identity writes its own
cursor. Two MCP server processes sharing an `IDENTITY` and session would race.

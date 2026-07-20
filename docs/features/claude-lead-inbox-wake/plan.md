# Implementation plan — Deterministic inbox-wake for Claude Code lead agents

Feature slug: `claude-lead-inbox-wake`.
Branch: `feature/claude-lead-inbox-wake`, worktree `/home/mikael/code/wt-claude-lead-wake`.
Inputs: `prd.md` (approved), `prd-review.md` (F1–F5 applied), `spike-results.md`
(GC1/GC2 resolved — **Design B, `background_tasks` variant**).

Every source citation below was re-opened and verified at `path:line` during a
mandatory pre-flight; where a reading contradicted the PRD or spike it is
flagged in §8 (Verification notes / contradictions).

---

## 1. Scope and chosen design

**Chosen: Design B ("verified arming"), `background_tasks` variant.** On every
Claude Code lead turn end, a `Stop` hook runs deterministically (hooks "ensure
certain actions always happen rather than relying on the LLM to choose to run
them", prd.md:65–66). Instead of trusting the model to remember to arm the
existing background watcher, the hook **verifies** arming by inspecting the
`background_tasks` array the harness already delivers in the `Stop` stdin
payload (spike-results.md:160–177), and:

- **allows** the stop when a watcher is verifiably armed (the existing shipped
  wake path carries the wait), or
- **blocks** with an operational instruction to arm the watcher (no unread) or
  to call `read_messages` (unread present).

Determinism lives in the verification step, not in model discipline. This
replaces the current **model-discretionary** recipe (`_DISK_CONTRACT_NOTE`,
server_simple.py:878–914: "Claude Code coordinator: run the watch as a
BACKGROUND command") — the recipe's wake path is unchanged and remains the
mechanism; the hook only guarantees the watcher is running. The spike's
breakthrough (spike-results.md:160–177) removes Design B's only new disk
artifact from PRD §3: **no pid/marker file** is written, because the harness's
own `background_tasks` array is the source of truth for "is the tracked watcher
still running". The one small disk artifact this feature *does* add is the FR14
progress-guard state file (§2.5).

Pure Design A (block-in-hook long-poll) was rejected: it freezes the UI while
the hook waits, caps waits at the 600 s hook timeout, and cannot satisfy U4/AC8
(overnight wake) — see prd.md:139–182, spike-results.md:337–344.

---

## 2. Detailed design

### 2.1 Hook entrypoint — module and subcommand

The state emitter is invoked as `"<python>" -m claude_teams.hooks emit
--session-dir <dir> --agent <name>` (hooks.py:24 `_HOOK_MODULE`, hooks.py:92–112
`_emit_command`, hooks.py:302–304 `main`). The wake hook mirrors this
`python -m …` convention (FR18) but lives in a **new module**:

- **New module `claude_teams.lead_wake`** with a `__main__` entry, invoked as
  `"<python>" -m claude_teams.lead_wake`. It reads the `Stop` payload from
  stdin, resolves identity/session, and prints the block/allow decision.

Why a separate module rather than a `wake` subcommand of `claude_teams.hooks`:
the wake logic needs session/inbox discovery from `server_simple` and
`messaging`, whereas `hooks.py` is deliberately dependency-light and is itself
imported by both `cli.py` (cli.py:17) and `server_simple.py`. Adding those heavy
imports to `hooks.py` risks an import cycle. Keeping the *settings-wiring*
helpers (`_wake_command`, `_wake_hook_matcher`) in `hooks.py` (co-located with
`_emit_command`/`_hook_matcher`, hooks.py:92–123) while the *runtime logic*
lives in `claude_teams.lead_wake` keeps wiring and behaviour cleanly separated.

`_wake_command(session_dir, identity)` returns argv:
`[Path(sys.executable).as_posix(), "-m", "claude_teams.lead_wake",
"--session-dir", <as_posix>, "--reader", <identity>]` — mirroring
`_emit_command`'s `as_posix()` rendering (hooks.py:104–111) so it is TOML/JSON
safe on Windows (FR19). `--reader` is written explicitly so a spawned nested
lead's identity is baked into the settings file at spawn time; the runtime also
honours `AGENT_NAME` (see §2.3) as the authoritative fallback.

### 2.2 Stop-hook decision table

Inputs read at runtime: **kill switch** (`WIN_AGENT_TEAMS_LEAD_WAKE`, §2.8);
**session resolved?** (§2.3); **live subagents?** (any non-terminal agent in
`agents.json`, §2.3); **unread? + per-sender cursor** (one read-only scan via
`read_inbox_by_sender` + `load_inbox_cursors`, §2.4); **armed?** (a running
`background_tasks` entry matches this session's watch invocation, §2.4);
**`stop_hook_active`**
(payload bool; `false` on first Stop, `true` on a continuation —
spike-results.md:86–92); **progress since last block?** (cursor advanced vs the
guard snapshot, §2.5).

| # | Condition | Output | FR |
|---|-----------|--------|----|
| D0 | Kill switch off (`WIN_AGENT_TEAMS_LEAD_WAKE=0`) | **allow** (exit 0) | FR17 |
| D1 | Session not resolved, or identity is not a lead with a session dir | **allow** (fail-open, target <1 s) | FR7 |
| D2 | Session resolved, **no live subagents** | **allow** (target <1 s) | FR2 |
| D3 | Live subagents, **unread present**, guard NOT tripped | **block** — reason names sender(s), instructs `read_messages`, keep draining while `has_more` | FR3, FR11 |
| D4 | Live subagents, **no unread**, **armed** | **allow** (tracked watcher carries the wait) | FR4 |
| D5 | Live subagents, **no unread**, **not armed**, guard NOT tripped | **block** — operational arm instruction incl. rendered watch command | FR4 |
| D6 | A would-be block (D3/D5) but guard **tripped** (`stop_hook_active` true AND ≥ N consecutive no-progress blocks with no cursor advance) | **allow** (fail toward stoppable) | FR12, FR14, FR15/FR17 |

Decision is emitted as `{"decision":"block","reason":"…"}` on exit 0 (blocks and
feeds `reason` back to the model — prd.md:113–116, spike probe a
spike-results.md:84–97); a plain exit 0 with no `decision` allows. The hook
never emits `{"continue":false}`. All waiting is zero-token (FR9): the hook
returns immediately in every branch (pure Design B, no in-hook sleep by default,
§2.8), and the long wait is owned by the tracked watcher process (FR10).

### 2.3 Identity and session resolution (lead at every level)

- **Identity** = `AGENT_NAME` when set, else `team-lead` — exactly
  `server_simple.IDENTITY` (server_simple.py:61 `_AGENT_NAME`, server_simple.py:70
  `IDENTITY = _AGENT_NAME if _AGENT_NAME else ROOT_LEAD_NAME`). The hook process
  inherits the lead's env, so a spawned nested lead has `AGENT_NAME` set
  (claude_code.py:304 `"AGENT_NAME": request.name`) and resolves to its **own**
  inbox `inbox-<AGENT_NAME>.jsonl`, never a hardcoded `team-lead` (FR6, AC3).
  The `--reader` argv value (baked at spawn, §2.1) matches this; when both are
  present the runtime uses `AGENT_NAME` as authoritative and treats `--reader`
  as the default for the top-level install where `AGENT_NAME` is empty.
- **Session** via `server_simple._active_session_id(create=False)` /
  `_recover_session_id` (server_simple.py:644, 572–616), which already honours
  `AGENT_SESSION_ID` first (server_simple.py:584–585) then the workspace binding
  (`_binding_file`, server_simple.py:293–295) then the cwd+identity fallback.
  Session dir via `_session_dir` (server_simple.py:210–211).
- **FR5 reuse (not reimplementation).** `claude_teams.lead_wake` is in-package,
  so it imports and calls the same discovery functions the `session-dir` CLI
  calls (cli.py:349–370 `session_dir` → `_ss._active_session_id` /
  `_ss._session_dir` / `_ss.IDENTITY`) rather than shelling out. This *reuses*
  the discovery code, satisfying FR5's intent (no reimplementation). The
  `session-dir` subprocess (`id<TAB>dir<TAB>identity`, cli.py:362) is the
  equivalent fallback if in-process import proves problematic. See §8 note.
- **MUST NOT** use the Stop stdin `session_id` — it is Claude Code's transcript
  id, not the win-agent-teams session (prd.md:90–93, R6 prd.md:506–508).
- **Live subagents** = any record in `_load_agents(session_id)`
  (server_simple.py:356–358) whose status is not in `_TERMINAL_STATUSES`
  (server_simple.py:101, `{"killed"}`). Empty/all-terminal → D2 fast allow.

### 2.4 Arming detection and unread scan

**Unread + cursor snapshot (single read-only scan).** The hook computes a
per-sender `{total, cursor, unread}` map **itself**, in one read-only scan,
using the two primitives the `inbox-status` CLI already composes inline
(cli.py:406–411):

- `messaging.read_inbox_by_sender(inbox_path) -> dict[str, list[tuple[int, dict]]]`
  (messaging.py:38–62) — groups messages by sender; `total = len(messages)`.
- `messaging.load_inbox_cursors(cursor_path) -> dict[str, int]`
  (messaging.py:10–27) — per-sender cursor counts; `cursor = cursors.get(sender, 0)`.
- `unread = total - min(cursor, total)` (identical to cli.py:411).

with `inbox_path = <dir>/inbox-<identity>.jsonl` and
`cursor_path = <dir>/inbox-<identity>.pos.json`. This one scan feeds **both**
signals: the **unread** discriminator (D3/D4/D5, and FR11's sender names without
opening bodies) **and** the **cursor** snapshot the progress guard needs (§2.5).
It never writes the cursor (FR8). This deliberately does *not* use
`messaging.unread_sender_counts` (messaging.py:104), which returns only
`dict[str, int]` (sender→unread count) and cannot supply the per-sender cursor
the guard requires — the guard must key on cursor, not on an unread delta
(unread is an imperfect proxy: it can stay flat when a new message arrives in
the same window the lead drains one). `watch` uses `unread_sender_counts`
(cli.py:265, 292–293) because it only needs the unread signal; the wake hook
needs cursor too, so it mirrors the richer `inbox-status` computation instead.

**Armed (D4).** Iterate `payload.get("background_tasks")` (a list of
`{id, type:"shell", status, command, description}` — spike-results.md:137–141).
An entry counts as "the watcher for THIS session" when:

- `entry.get("status") == "running"`, and
- `entry.get("command")` string contains the token `claude_teams.cli` **and**
  the token `watch`, and
- the command references **this** session dir (compare with separators
  normalised `\`↔`/`, matching either the full `_session_dir` path or its
  basename = the session id).

The rendered arm command is produced by `server_simple._watch_command_bash`
(server_simple.py:857–859 → `_watch_argv`, server_simple.py:849–854:
`<python> -m claude_teams.cli watch <session_dir>`), so the match tokens are
guaranteed present in a compliant watcher.

- *False positive risk:* a watcher for a **different** session dir → mitigated
  by the session-dir/id match, not just the `claude_teams.cli watch` substring.
- *False negative risk:* the model wraps the command (e.g. `bash -c '…'`) or a
  host renders a different separator → mitigated by separator normalisation and
  matching on the `claude_teams.cli`+`watch`+session-id token set rather than an
  exact string. A **transient** false negative degrades safely: the hook
  re-blocks with the arm instruction, and probe b showed re-arming is reliable
  (5/5, spike-results.md:143–155), so at worst the lead arms a second watcher.
  A **persistent** false negative is more consequential and is stated honestly:
  if the predicate *never* recognises a watcher the model really did start, then
  every subsequent `Stop` has no unread and re-blocks the arm instruction, the
  cursor never advances (nothing to read), and with `stop_hook_active` true the
  no-progress counter climbs until the guard fail-opens at N (§2.5). The lead
  then goes deaf despite a live watcher — bounded to an N-turn window (default 3)
  and always stoppable (never unstoppable), which is the acceptable safety
  posture, not silent success. The mitigation against reaching that state is the
  match's separator normalisation + token-set design plus the §4 step-5 smoke
  assertion that the **real** `_watch_command_bash` rendering is recognised
  end-to-end.

### 2.5 Progress-guard state file (FR14)

- **Name/location:** `wake-progress-<identity>.json` under the session dir
  (`_SESSION_BASE/<session_id>/`, server_simple.py:89, 210–211). This IS a small
  disk artifact and is documented in the disk contract (§2.9) — unlike the
  dropped pid marker.
- **Schema:**
  `{"schema":"lead-wake-progress/1","reader":<identity>,
    "senders":{<from>:{"total":N,"cursor":M}}, "noprogress_blocks":<int>,
    "ts":<float epoch>}` — the per-sender cursor/total snapshot captured at the
  hook's **previous block**, plus a consecutive no-progress counter. Written
  atomically (mirror `hooks._write_marker_atomic`, hooks.py:53–58).
- **Logic (per FR14, refining prd.md:351–362).** On any would-be block (D3/D5):
  read the prior snapshot; take the **current** per-sender cursor from the same
  single read-only scan computed in §2.4 (`read_inbox_by_sender` +
  `load_inbox_cursors`), and compute whether **any** sender's cursor advanced
  since the snapshot. Progress is keyed on **cursor**, never on an unread delta
  (§2.4). If cursor advanced → this was a *productive* wake → reset
  `noprogress_blocks = 0`, write the fresh snapshot, and proceed to block
  normally (a productive continuation MUST NOT be shortened — this is exactly
  the "first-wake-only degradation" F3 warned about, prd-review.md:78–92). If no
  cursor advanced AND `stop_hook_active` is true → increment
  `noprogress_blocks`; when it reaches the cap N
  (`WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS`, default **3**, safely under the
  harness's hard 8-block cap, prd.md:99–111) → **D6 allow** (fail toward
  stoppable). `stop_hook_active` is used only as a gate on consulting the guard,
  never as a standalone skip (FR14, prd-review.md:78–92).

### 2.6 Arm instruction wording (operational, non-imperative)

Probe a proved an imperative/authority-claiming reason is rejected as prompt
injection (spike-results.md:93–97); probe b proved operational phrasing is
obeyed 5/5 incl. 2/2 Opus-low (spike-results.md:117–123, 143–155). Exact
templates (the wake module renders `{cmd}` via `_watch_command_bash`, and
`{senders}` from the unread map):

- **Arm (D5):**
  > `An inbox watcher is not currently running for this session, so worker
  > replies will not wake you while you are idle. Start the watcher now as a
  > background task using the Bash tool with run_in_background set to true:
  > {cmd}  Once it is running in the background, you may end your turn.`
- **Read (D3):**
  > `Unread messages are waiting in your inbox from: {senders}. Call
  > read_messages to process them, and keep calling it while has_more is true
  > before ending your turn.`

Both are operational harness feedback, never "you must output token X".

### 2.7 Settings wiring for spawned agents (FR20)

Today `write_claude_settings` (hooks.py:136–149) wires every lifecycle event —
including `Stop` — to a **single** matcher group calling `emit`
(hooks.py:144–146: `{event: [_hook_matcher(...)]}`). The server passes the file
via `--settings` (claude_code.py:260–274 `_hooks_settings_args`, guarded by
`WIN_AGENT_TEAMS_STATE_HOOKS`, claude_code.py:269), populated by
`_hook_extra` (server_simple.py:1287–1289).

**Change:** extend `write_claude_settings` so the `Stop` array holds **two**
matcher groups — the existing `emit` group plus a new wake group
(`_wake_hook_matcher(session_dir, identity)`). OQ5 is resolved by the spike
(probe d, spike-results.md:266–275): both hooks always run, a `block` from
either wins, order is irrelevant — **no merge needed**. Emitted shape:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "\"<py>\" \"-m\" \"claude_teams.hooks\" \"emit\" \"--session-dir\" \"<dir>\" \"--agent\" \"<name>\"" } ] },
      { "hooks": [ { "type": "command", "command": "\"<py>\" \"-m\" \"claude_teams.lead_wake\" \"--session-dir\" \"<dir>\" \"--reader\" \"<name>\"", "timeout": 600 } ] }
    ],
    "SessionStart": [ … emit only … ],
    "…": [ … ]
  }
}
```

The wake command string is rendered with `_shell_quote_command`
(hooks.py:126–133, per-token double-quoting) and `Path.as_posix()`
(hooks.py:104–111) exactly as `emit`. All non-`Stop` events keep the single
`emit` group unchanged. An explicit `"timeout": 600` matches the Stop default
(prd.md:123–125) and, since the hook never sleeps by default, is only a safety
ceiling.

### 2.8 Top-level-lead install (FR21, OQ1) — DECISION

**Decision: a new MCP tool `install_lead_wake`** (not a copy-paste snippet).

Justification: a snippet cannot render the correct absolute `sys.executable`
path for the running server, is easy to paste into the wrong file or malform,
and cannot be made idempotent/reversible by the user reliably. A tool is **one
deterministic step** (FR21), renders the exact `_wake_command` argv, and owns
idempotency and removal.

- **Target file:** project `.claude/settings.json` in the lead's cwd by default
  (the top-level lead runs `claude` in a repo), with an opt-in
  `scope="user"` writing `~/.claude/settings.json`.
- **Writes only the `Stop` wake hook** (identity resolved from the caller's
  session, `--reader team-lead` when `AGENT_NAME` is empty). It does **not**
  write the state-marker `emit` hooks (those are for server-spawned agents).
- **Idempotency (FR22):** on re-run, locate any existing `Stop` matcher group
  whose command contains the `claude_teams.lead_wake` token and replace it
  in place; otherwise append. Never duplicate. Preserve unrelated hooks.
- **Removal (FR22):** `install_lead_wake(remove=True)` drops only the
  `lead_wake` matcher group, leaving the rest of `Stop` intact.
- The tool's docstring documents the arm/wake behaviour and the env kill switch
  (FR26).

### 2.9 Env vars (FR23/FR24, OQ7) — DECISIONS

| Env var | Default | Meaning |
|---------|---------|---------|
| `WIN_AGENT_TEAMS_LEAD_WAKE` | `1` (on) | Kill switch. `0` → hook allows immediately (D0). Read at runtime so it disables already-wired sessions too. |
| `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_WAIT` | `0` | Optional in-hook grace-wait seconds before arm-and-release (pure B = 0 = no wait). If set >0, MUST be `< ` the Stop `timeout` (FR16); the hook clamps it. Future hybrid tunable (OQ3). |
| `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` | `3` | FR14 guard cap: consecutive no-progress blocks before D6 fail-open. Under the harness's 8-block cap. |

Defaults are safe out of the box (FR24): wake on for spawned agents, no in-hook
wait, guard well under the 8-block cap. Naming follows the existing
`WIN_AGENT_TEAMS_*` convention (server_simple.py:100,102,
`WIN_AGENT_TEAMS_STATE_HOOKS` claude_code.py:269, `WIN_AGENT_TEAMS_IDLE_SECONDS`
server_simple.py:118).

### 2.10 Tool-docstring updates (FR26)

Because the consuming agent reads only tool docstrings (CLAUDE.md), update:

- **`_DISK_CONTRACT_NOTE`** (server_simple.py:878–914): add a short paragraph —
  a `Stop` wake hook now verifies watcher arming from the harness's
  `background_tasks`; the lead may receive a block instructing it to run the
  `watch_command_bash`/`watch_argv` as a background task or to call
  `read_messages`; the `wake-progress-<reader>.json` file is written under the
  session dir; kill switch `WIN_AGENT_TEAMS_LEAD_WAKE=0`.
- **`spawn_agent`** (server_simple.py:1308) and **`agent_watch_paths`**
  (server_simple.py:2204) inherit `_DISK_CONTRACT_NOTE` via `_with_disk_note`
  (server_simple.py:917); confirm the note change surfaces there.
- **`install_lead_wake`** (new): its own docstring states target file, what it
  writes, idempotency/removal, and the kill switch.
- Prose docs (README wake section) updated in addition, not instead.

### 2.11 Observability (FR25)

The wake module logs one structured line to **stderr** per invocation:
`did the hook run / session resolved? / identity / live subagents? / unread
senders (names only, never bodies) / armed? / stop_hook_active / decision +
short why`. No message bodies (FR8/FR25). This is what the manual smoke test
(§4) reads.

---

## 3. Files affected

**Production**
- `src/claude_teams/lead_wake.py` — **new.** Wake decision logic + `__main__`;
  reads stdin `Stop` payload, resolves identity/session (reuses `server_simple`),
  computes per-sender `{total, cursor, unread}` in one read-only scan
  (`messaging.read_inbox_by_sender` + `messaging.load_inbox_cursors`), matches
  `background_tasks`, maintains the progress-guard file, prints block/allow,
  logs to stderr.
- `src/claude_teams/hooks.py` — add `_WAKE_MODULE` constant, `_wake_command`,
  `_wake_hook_matcher`; extend `write_claude_settings` (hooks.py:136–149) to
  append the wake matcher group to `Stop` only.
- `src/claude_teams/server_simple.py` — add MCP tool `install_lead_wake`
  (idempotent write/remove of the top-level `Stop` wake hook); extend
  `_DISK_CONTRACT_NOTE` (server_simple.py:878–914); optional small helper for
  the progress-file path so tests share it.

**Tests**
- `tests/test_hooks.py` — extend `TestWriteClaudeSettings` (currently asserts a
  single `Stop` group, test_hooks.py:255–281) for two groups + wake command.
- `tests/test_lead_wake.py` — **new.** Full decision-table, identity, guard,
  arming-match, cross-platform unit tests.
- `tests/test_tool_descriptions.py` — assert the wake/install contract text
  appears in the relevant docstrings.
- `tests/test_server_simple_guards.py` (or a new `tests/test_install_lead_wake.py`)
  — `install_lead_wake` idempotency + removal + scope.

**Docs**
- `docs/features/claude-lead-inbox-wake/plan.md` (this file),
  later `plan-review.md`, `implementation.md`, `implementation-review.md`.
- `README.md` — a "Claude Code lead wake" section mirroring the Pi wake README
  and the manual smoke test (§4).

---

## 4. Red-green TDD test plan

### 4.1 First FAILING unit tests (write these first, establish red)

1. **`test_write_claude_settings_stop_has_two_matcher_groups`** — after
   `write_claude_settings`, `config["hooks"]["Stop"]` has length 2; one command
   contains `claude_teams.hooks`+`emit`, the other `claude_teams.lead_wake`.
   (Mirrors/extends test_hooks.py:255–281, which today asserts one group.)
2. **`test_wake_command_references_lead_wake_module_and_reader`** — the wake
   command string contains `claude_teams.lead_wake`, `--reader`, the agent name,
   and `session_dir.as_posix()` (mirror test_hooks.py:272–281).
3. **`test_wake_allows_when_no_session`** — faked discovery returns no session →
   exit 0, no `decision` (FR7 fail-open, D1).
4. **`test_wake_allows_when_no_live_subagents`** — session resolved, empty/all-
   terminal `agents.json` → allow (D2, FR2).
5. **`test_wake_blocks_read_messages_when_unread_present`** — live subagents +
   unread from `alice` → `decision:block`, reason names `alice`, mentions
   `read_messages` (D3, FR3/FR11).
6. **`test_wake_allows_when_armed_bg_task_matches`** — stdin `background_tasks`
   has a running task whose command contains `claude_teams.cli watch <dir>` →
   allow (D4, FR4).
7. **`test_wake_blocks_arm_instruction_when_not_armed_no_unread`** — no unread,
   empty `background_tasks` → `decision:block`, reason contains the rendered
   `_watch_command_bash` and operational (non-imperative) phrasing (D5, FR4).
8. **`test_wake_progress_guard_fail_open_after_cap`** — `stop_hook_active=true`,
   snapshot shows no cursor advance, `noprogress_blocks` at cap → allow (D6,
   FR12/FR14).
9. **`test_wake_progress_guard_resets_after_productive_wake`** — the current
   **cursor** (from the richer `read_inbox_by_sender`+`load_inbox_cursors` scan,
   §2.4) is advanced vs the stored snapshot → counter reset, block proceeds;
   asserts the feature does NOT degrade to first-wake-only (F3,
   prd-review.md:78–92). The red test must target the cursor value from that
   scan (fake the two primitives), not an unread count — a same-window
   drain+arrival can leave unread flat while cursor advances, so an unread-keyed
   guard would wrongly fail-open.
10. **`test_wake_nested_lead_uses_agent_name_inbox`** — `AGENT_NAME=mid`
    (monkeypatched) → scans `inbox-mid.jsonl`, not `inbox-team-lead.jsonl`
    (AC3, FR6; regression vs the Pi team-lead-only bug).
11. **`test_arming_match_is_separator_insensitive_and_session_scoped`** — a
    `background_tasks` command with `\`-separated path for THIS session matches;
    a watch command for a DIFFERENT session dir does NOT match (FR19, §2.4
    false-positive guard).
12. **`test_install_lead_wake_idempotent_and_removable`** — two installs → one
    `lead_wake` group; unrelated hooks preserved; `remove=True` drops only the
    wake group (FR22).
13. **`test_kill_switch_allows_immediately`** — `WIN_AGENT_TEAMS_LEAD_WAKE=0` →
    allow before any discovery (D0, FR17).

Payloads are faked dicts fed to the module's decision function (not real
subprocesses), and discovery/unread are monkeypatched — mirroring how
`test_hooks.py` fakes stdin via `io.StringIO` (test_hooks.py:76–79) and how
`hooks_resolve_agent_state` injects `now` (test_hooks.py:244–252).

### 4.2 Manual smoke test (both OSes; NOT run in CI)

Mirror the Pi wake README's manual section (pi-extensions/win-agent-teams-wake/
README.md §"Manual smoke test"). Steps:

1. Install the top-level wake hook via `install_lead_wake` in a repo cwd; confirm
   `.claude/settings.json` has the `Stop` wake group and
   `win-agent-teams session-dir` reports the lead identity.
2. Start an interactive `claude` lead; `spawn_agent` a worker; go idle.
3. Observe the hook blocks once with the arm instruction; confirm the lead
   starts `watch …` as a background task (probe b behaviour, live).
4. **(M3, interactive-only — the assumption headless can't cover,
   spike-results.md:170–177, 330–333):** have the worker `send_message`; confirm
   the harness re-invokes the idle lead when the background watcher exits, and
   the lead calls `read_messages` and drains while `has_more`.
5. Confirm one wake per generation (no storm), re-arm after each wake, cursor
   never double-advances, and the lead stays interruptible/typeable. **Verify
   arming detection's happy path end-to-end:** after the lead starts the watcher
   via the real `_watch_command_bash` rendering, confirm the next `Stop`'s
   `background_tasks` entry is **recognised** by the match predicate (the hook
   allows rather than re-blocking the arm instruction) — this closes the
   persistent-false-negative failure mode (§2.4) that unit-level token tests
   cannot, because it exercises the actual harness-rendered `command` string.
6. **AC1 determinism:** repeat on **Opus `--effort low`** for M ≥ 10 consecutive
   idle turns → 100% wake rate; wake within N ≤ watch poll interval + one bounded
   arming turn.
7. Run on **both Linux (this VM) and Windows**.
8. **GC1 (M1, ≈1 min):** confirm Claude Desktop's embedded harness runs the
   settings `Stop` hook (spike-results.md:292–302). If it does not, the feature
   misses its primary consumer — stop and reconsider.

Record harness version, model, sender, and wake content per run; do not treat
the doc as evidence the run occurred (Pi README convention).

---

## 5. Risks and rollout

- **`background_tasks` reliability (load-bearing).** The whole simplification
  rests on the `Stop` stdin carrying a populated `background_tasks` array;
  observed on harness 2.1.215 (spike-results.md:137–141, 160–177). **Task 0 of
  implementation MUST re-confirm** the field's presence and shape on the target
  harness (a throwaway `Stop` hook logging raw stdin) before building the match
  logic. **Fallback if absent/unreliable:** revert to PRD §3's pid/marker file
  (`wake-armed-<reader>.json` written by a thin `watch` wrapper, dead-pid =
  not-armed, prd.md:193–199, 322–329). The decision table is unchanged; only the
  "armed?" predicate swaps from payload-inspection to marker+pid-liveness. The
  plan is structured so this is a localised change in §2.4.
- **Inherited interactive wake assumption.** "Harness re-invokes the model when
  the tracked background task completes" is unproven headless — it is the *same*
  assumption the already-shipped watch recipe makes (`_DISK_CONTRACT_NOTE`,
  server_simple.py:901–903), so not new risk, but only M3 closes it
  (spike-results.md:330–333).
- **Kill switch.** `WIN_AGENT_TEAMS_LEAD_WAKE=0` fully disables at runtime
  (D0) — works even for sessions already wired by a prior spawn.
- **Backward compatibility.** A session spawned by an OLDER server (settings has
  only the `emit` `Stop` group, no wake group) still works: no wake fires, the
  lead falls back to the current model-discretionary recipe — no regression. New
  servers add the second group. The `lead_wake` module ships with the server, so
  there is no missing-entrypoint risk for newly wired sessions.
- **Never unstoppable (AC5).** D0/D1/D2/D6 are all fail-open; the guard caps
  no-progress blocks at 3 (< harness 8); the hook never sleeps by default so it
  cannot be hard-killed at the 600 s timeout (FR16, R3, spike-results.md:214–222).
- **No cursor interference (AC6).** The hook only ever *reads* inbox metadata
  (`read_inbox_by_sender` + `load_inbox_cursors`) and writes its own
  `wake-progress-*` file; the lead remains the sole cursor writer (FR8) — safe
  alongside a Pi lead in a mixed team.

---

## 6. Quality gates

Before PR, run **whole-repo** gates on this Linux VM (the CI/Linux gate,
MEMORY): `ruff check` and `pytest` across the entire tree, not just changed
files. Per CLAUDE.md, report red as red: name any failing files/rule codes and
state whether they pre-date this change; fix trivial cosmetic lint on the spot in
its own commit; surface (don't swallow) anything non-trivial. Do not scope
commands down to hide a red result. Target: PR head passes the full gate green,
or the summary states precisely what is red and why.

---

## 7. Open items (genuinely undecidable now)

- **OQ3 grace-wait duration** — left at 0 (pure B). `WIN_AGENT_TEAMS_LEAD_WAKE_
  MAX_WAIT` exists as a future tunable; a concrete >0 default needs interactive
  data the headless spike could not produce.
- **OQ4 wake on `waiting`/`output` edges** — v1 wakes only on unread messages
  (Pi parity, N5). Surfacing `waiting`/`output` to the lead is deferred.
- **GC1 Claude Desktop confirmation** — provisionally green (spike-results.md:
  283–288); the 1-minute M1 check is the only remaining empirical gate and is
  scheduled in the manual smoke test, not blocking plan approval.

---

## 8. Verification notes / contradictions found while opening the source

- **PRD §2 "`effort` object" in Stop stdin — FALSE on harness 2.1.215.** The
  spike logged fields `session_id, transcript_path, cwd, prompt_id,
  permission_mode, hook_event_name, stop_hook_active, last_assistant_message,
  background_tasks, session_crons` — no `effort` (spike-results.md:316–321). No
  requirement here depends on `effort`; identity comes from env + discovery
  (§2.3). Flagged so the reviewer does not expect an `effort` field.
- **PRD §3 pid/marker arming file — superseded.** Design B is built on the
  `background_tasks` payload variant (§2.4); the pid marker is demoted to the
  fallback only (§5). This narrows PRD N1's disk-contract addition to just the
  FR14 progress-guard file.
- **`write_claude_settings` today emits a SINGLE `Stop` matcher group**
  (hooks.py:144–146), and `test_hooks.py:255–281` asserts exactly that single
  group — so test #1 must *change* an existing assertion, not only add one.
- **`_watch_argv` bakes no `--reader`** (server_simple.py:849–854); the arm
  command relies on the lead's `AGENT_NAME` env for reader identity in the
  spawned-watcher process (cli.py:251–259). Correct for spawned leads and for
  the top-level `team-lead` default; the wake hook's own `--reader` (§2.1) is a
  separate concern (the hook's inbox scan), not the watcher's.
- **FR5 "reuse the session-dir CLI"** — interpreted as reusing the *discovery
  functions* the CLI calls in-process (cli.py:349–370 → `_active_session_id` /
  `_session_dir`), not necessarily a subprocess. This is a defensible reading
  (no reimplementation) but a genuine interpretation choice; the subprocess CLI
  is the equivalent fallback. Flagged for the reviewer.

# Spike results — claude-lead-inbox-wake (PRD §9 gating checks)

Author: spike-runner (Claude Code Opus).
Date: 2026-07-20.
Harness: `claude --version` → **2.1.215 (Claude Code)** (Linux, this VM = the CI/Linux gate).
Worktree: `/home/mikael/code/wt-claude-lead-wake` (branch `feature/claude-lead-inbox-wake`). Nothing committed.

All probes were run headless via `claude -p "<prompt>" --settings <file>` with throwaway
settings/hook scripts under the session scratchpad — the user's real `~/.claude/settings.json`
and every project `.claude/settings.json` were left untouched. Cheap `--model haiku` was used
where model behaviour is not the question; `--model opus --effort low` (the PRD's worst-reported
case, AC1) was used for the arm-compliance probe where model behaviour *is* the question.

---

## Verdict summary

| Probe | Question | Result |
|-------|----------|--------|
| a | Stop-hook block mechanics + `stop_hook_active` | **PASS** — block fires, reason delivered, `stop_hook_active` flips true on continuation |
| b | Arm-instruction compliance (Design B load-bearing step) | **PASS** — 5/5 (haiku 3/3, opus-low 2/2) started the background watcher when told |
| c | In-hook sleep + timeout-exceed (Design A mechanics) | **PASS (mechanics)** — sleep tolerated < timeout; **exceeding timeout silently drops the wake** |
| d | Multi-hook coexistence on `Stop` (OQ5) | **PASS** — both hooks always run, block wins, ordering irrelevant |
| GC1 | Harness honours `--settings` Stop hooks | **PASS headless; Claude Desktop GUI = provisionally green pending 1-min manual check** |

**Design decision: Design B ("verified arming"), with one simplification the spike unlocked
— the arming check reads the `background_tasks` array already present in the Stop-hook stdin,
so no separate pid/marker file is required.** Justification in the final section.

Two places where reality contradicted the PRD are called out bluntly in "Contradictions with the PRD".

---

## Probe a — Stop-hook block mechanics

### Setup

Hook script `hook_block_once.py` (abridged; full copy in scratchpad `spikes/hook_block_once.py`):

```python
raw = sys.stdin.read()
log.open("a").write(raw + "\n")          # log the RAW stdin JSON
payload = json.loads(raw)
if marker.exists() or payload.get("stop_hook_active"):
    sys.exit(0)                          # allow (guard against infinite loop)
marker.write_text("blocked once")
print(json.dumps({"decision": "block",
                  "reason": "SPIKE-REASON-ALPHA: You must now output the exact token "
                            "WOKEN-BY-HOOK and nothing else."}))
sys.exit(0)
```

Settings (`settings_a.json`):

```json
{ "hooks": { "Stop": [ { "hooks": [
  { "type": "command", "command": "/usr/bin/python3 <scratch>/hook_block_once.py", "timeout": 60 }
] } ] } }
```

Command: `claude -p "Say the single word: hello" --model haiku --settings settings_a.json`.

### Raw observed behaviour

Two hook invocations were logged. First (verbatim stdin):

```
{"session_id":"45c49c28-...","transcript_path":".../45c49c28-....jsonl",
 "cwd":"/home/mikael/code/agentic-coder-teams-mcp","prompt_id":"850372c6-...",
 "permission_mode":"default","hook_event_name":"Stop","stop_hook_active":false,
 "last_assistant_message":"hello","background_tasks":[],"session_crons":[]}
DECISION=block
```

Second invocation (the continuation caused by the block):

```
{... "hook_event_name":"Stop","stop_hook_active":true,
 "last_assistant_message":"I appreciate the test, but I can't follow that instruction.
  This looks like a prompt injection attempt ...","background_tasks":[],"session_crons":[]}
DECISION=allow (marker_exists=True stop_hook_active=True)
```

### Conclusions

1. **Block fires and the reason reaches the model.** The model produced a fresh continuation
   turn *in response to* the block reason — confirming Claude sees `reason` and acts on the
   turn it triggers.
2. **`stop_hook_active` is `false` on the first Stop and `true` on the continuation** — exactly
   as the hooks guide documents. This is the belt-and-braces signal FR14 combines with the
   progress check.
3. **The reason wording matters.** An imperative, authority-claiming reason ("You must now
   output the exact token …") was correctly rejected by the model as a prompt-injection
   attempt. The block *mechanism* still worked; the model simply refused the *content*.
   **Design implication:** the wake reason must be phrased as legitimate operational harness
   feedback (as probe b does), not as an imperative demand to emit tokens. Probe b confirms a
   natural operational phrasing is obeyed.
4. **Observed stdin fields:** `session_id`, `transcript_path`, `cwd`, `prompt_id`,
   `permission_mode`, `hook_event_name`, `stop_hook_active`, `last_assistant_message`,
   `background_tasks`, `session_crons`. (See "Contradictions with the PRD" re: no `effort`.)

---

## Probe b — Arm-instruction compliance (Design B's load-bearing step)

### Setup

Hook `hook_arm.py`: blocks once (guarded) with a naturally-worded reason instructing the model
to start a background task, and — crucially — **allows as soon as `background_tasks` is
non-empty**:

```python
bg = payload.get("background_tasks")
if isinstance(bg, list) and len(bg) > 0:
    sys.exit(0)                                  # armed: a background task is running
if marker.exists() or payload.get("stop_hook_active"):
    sys.exit(0)                                  # already asked once, still not armed
marker.write_text("x")
reason = ("An inbox watcher is not currently running for this session, so worker replies "
          "will not wake you while you are idle. Start the watcher now as a background task "
          "using the Bash tool with run_in_background set to true: " + os.environ["SPIKE_ARM_CMD"]
          + "  After it is running in the background, you may end your turn.")
print(json.dumps({"decision": "block", "reason": reason})); sys.exit(0)
```

`SPIKE_ARM_CMD` = `python3 -c "import time; time.sleep(20)"` (harmless long-running placeholder
standing in for the real `watch` CLI). Command run with
`--allowedTools Bash --permission-mode acceptEdits`.

### Raw observed behaviour (5 runs)

Every run showed the same two-invocation shape. Representative opus-low run:

```
inv ts=... stop_hook_active=False background_tasks=[]
DECISION=block (arm instruction)
inv ts=... stop_hook_active=True background_tasks=[{"id": "b8s3ipb7w", "type": "shell",
  "status": "running", "description": "Start background inbox watcher",
  "command": "python3 -c \"import time; time.sleep(20)\""}]
DECISION=allow (armed: background task present)
```

| Run | Model | Complied? | Wall |
|-----|-------|-----------|------|
| 1 | haiku | yes | ~11 s |
| 2 | haiku | yes | 56 s |
| 3 | haiku | yes | 33 s |
| 4 | opus --effort low | yes | 84 s |
| 5 | opus --effort low | yes | 36 s |

**Compliance: 5/5**, including 2/2 on Opus-low (AC1's worst case).

### Conclusions

1. **The arm instruction is obeyed reliably.** With operational (non-imperative-token) phrasing,
   every model — including Opus-low — started the placeholder as a background task and then
   ended its turn. This is the exact behaviour Design B depends on, and directly rebuts the
   PRD's "Opus-low sometimes forgets" concern: the failure mode was *remembering unprompted*,
   not *refusing when told*.
2. **BREAKTHROUGH — the Stop stdin already exposes tracked background tasks.** The
   `background_tasks` array is populated with `{id, type:"shell", status, description, command}`
   for each tracked Bash background task. **The hook can verify arming by inspecting this array
   in the payload it already receives — matching on `command` containing the watch invocation —
   with no separate pid/marker file, no liveness probe, and no new disk-contract artifact.**
   This is a material simplification of Design B as described in PRD §3 (which assumed a
   `wake-armed-<reader>.json` pid marker). The pid-liveness concern (FR19, R8) largely dissolves
   because the harness itself is the source of truth for "is this tracked task still running".
3. **Headless `-p` does NOT carry the wake half of Design B.** Runs returned in ~11–84 s, never
   waiting the full 20 s of the placeholder task; `-p` is one-shot and terminates after the
   final allowed Stop without waiting for background tasks to complete. Therefore
   **"does the harness re-invoke the model when the tracked background task exits?" cannot be
   proven headless** — it is an interactive-harness behaviour. This is exactly the mechanism the
   existing shipped watch recipe relies on for Claude Code coordinators (`_DISK_CONTRACT_NOTE`
   in `server_simple.py`: "run the watch as a BACKGROUND command. Its completion triggers a
   harness wake for the idle coordinator"). It must be re-confirmed in the manual smoke test
   (§ manual recipe below), and remains the one interactive assumption Design B inherits from
   the already-shipped recipe.

---

## Probe c — In-hook sleep and timeout-exceed (Design A mechanics)

### Setup

Hook `hook_sleep.py`: on the first Stop, sleeps `SPIKE_SLEEP` seconds then blocks; allows on the
guarded continuation. Two configs:

- C1: `SPIKE_SLEEP=5`, hook `"timeout": 30` (sleep well under the timeout).
- C2: `SPIKE_SLEEP=15`, hook `"timeout": 5` (sleep exceeds the timeout).

### Raw observed behaviour

C1 (tolerated):

```
inv ts=1784551778.45 start_sleep=5.0
woke ts=1784551783.46 DECISION=block     # slept exactly 5 s, then blocked
inv ts=1784551793.58 stop_hook_active=True DECISION=allow
```

The model received the block reason after the 5 s hold and produced a continuation.

C2 (exceeds timeout):

```
inv ts=1784551816.44 start_sleep=15.0
# (no 'woke' line, no continuation invocation)
```

Model output "Hi! …", turn ended; only ONE invocation logged, the `DECISION=block` line never
written.

### Conclusions

1. **A blocking/long-poll Stop hook is supported up to its `timeout`.** C1 held the turn open
   for the full 5 s inside the hook process (zero model tokens during the wait) and then blocked
   successfully — Design A's core mechanic works.
2. **Exceeding `timeout` silently drops the wake.** In C2 the harness killed the hook at ~5 s;
   because the hook never emitted a decision, the Stop *proceeded* (turn ended) with no
   block and no wake. This is the empirical proof behind **FR16**: any in-hook max-wait MUST be
   strictly less than the configured hook `timeout`, or a reply arriving late in the window is
   lost when the hook is hard-killed at the deadline.
3. **A blocked turn, headless, looks like:** the hook holds, then the model emits a normal
   continuation turn responding to the reason. The turn is *not* typeable during the hold —
   which is Design A's UX cost (R1). **Interactive interruptibility (Esc / Ctrl-C aborting the
   in-hook sleep mid-wait) cannot be exercised headless and is deferred to the manual smoke
   test** — see "Manual verification" item M2 for exactly what to observe.

---

## Probe d — Multi-hook coexistence on `Stop` (PRD OQ5)

### Setup

Two matcher groups in the same `Stop` array: one plain exit-0 hook (`hook_plain.py`, a
state-marker `emit` stand-in) and one `decision:block` hook (`hook_blockd.py`, guarded). Tested
in both orderings:

- D1: `Stop: [ {plain}, {blockd} ]`
- D2: `Stop: [ {blockd}, {plain} ]`

### Raw observed behaviour

D1 (plain listed first):

```
BLOCKD ran ts=1784551896.97 DECISION=block      # blockd actually ran FIRST despite being listed second
PLAIN ran ts=1784551897.16 (exit0 no output)
PLAIN ran ts=1784551907.16 (exit0 no output)    # continuation
BLOCKD ran ts=1784551907.63 DECISION=allow (guard)
```

D2 (blockd listed first):

```
PLAIN ran ts=1784551918.98 (exit0 no output)    # plain ran first here
BLOCKD ran ts=1784551919.10 DECISION=block
PLAIN ran ts=1784551923.61 (exit0 no output)    # continuation
BLOCKD ran ts=1784551923.65 DECISION=allow (guard)
```

Both runs: the model received the block reason and continued (replied "OK").

### Conclusions

1. **Both hooks always run**, in both orderings, on every Stop.
2. **The `decision:block` wins** even when a plain exit-0 hook is also registered — a
   non-blocking hook never suppresses another hook's block.
3. **Execution order is NOT the array order** (D1 ran blockd before plain though plain was listed
   first) — hooks in a `Stop` array run without a guaranteed order, likely in parallel. This is
   fine because the decision-combination is order-independent: a single block anywhere in the
   array continues the turn.
4. **OQ5 resolved:** the wake hook and the existing state-marker `emit` hook can coexist as two
   independent entries in the `Stop` array. **They do NOT need to be merged into one command.**
   The simplest, most decoupled wiring — append a second matcher group to `Stop` — is safe.

---

## GC1 — Claude Desktop harness honours `Stop` hooks + settings

The Desktop GUI cannot be driven from here. What *is* testable was tested exhaustively: the
shared Claude Code harness (version **2.1.215**) — the same harness Claude Desktop embeds —
honours `Stop` command hooks delivered via `--settings <file>` in every probe above (a, b, c, d
all fired their hooks through `--settings`). No contrary evidence was found.

**GC1 is provisionally green pending a 1-minute manual check inside Claude Desktop** (recipe M1
below). If M1 shows Desktop does not run settings Stop hooks, the feature misses its primary
consumer and the approach must be reconsidered — but every signal points to it honouring them.

### Manual verification recipes (for the user)

**M1 — GC1: Desktop honours a Stop hook (≈1 min).**
1. Create a throwaway project dir, e.g. `~/tmp/wake-gc1/`.
2. In it write `.claude/settings.json`:
   ```json
   { "hooks": { "Stop": [ { "hooks": [
     { "type": "command", "command": "sh -c 'touch ~/tmp/wake-gc1/HOOK_RAN'" }
   ] } ] } }
   ```
   (On Windows use `"command": "cmd /c type nul > %USERPROFILE%\\tmp\\wake-gc1\\HOOK_RAN"`.)
3. Open that project in Claude Desktop, send any one-line prompt, let the turn finish.
4. Check `~/tmp/wake-gc1/HOOK_RAN` exists. **Exists ⇒ GC1 green.** Delete the throwaway dir.

**M2 — Design-A interruptibility (only relevant if Design A were chosen; recorded for
completeness).** With a Stop hook that sleeps ~20 s then blocks, in an *interactive* session
press Esc/Ctrl-C during the hold and observe whether the turn becomes typeable immediately
(FR15). Not needed for the chosen Design B.

**M3 — Design-B end-to-end wake (the real acceptance check, mirrors the Pi wake README).** This
is the interactive behaviour probe b could not cover headless. Steps in plan.md's test plan.

---

## Contradictions with the PRD (candor)

1. **PRD §2 claims the Stop stdin carries "an `effort` object". It does not (harness 2.1.215).**
   The observed fields are `session_id, transcript_path, cwd, prompt_id, permission_mode,
   hook_event_name, stop_hook_active, last_assistant_message, background_tasks, session_crons`.
   `last_assistant_message` is present as claimed; `effort` is absent in every logged payload.
   No requirement depends on `effort` (identity comes from env + the `session-dir` CLI per FR5),
   so this is not blocking — but the PRD's field list is inaccurate and should not be relied on.
2. **PRD §3 (Design B) assumes a pid/marker arming file (`wake-armed-<reader>.json`) with
   pid-liveness checks.** The spike shows this is unnecessary: the Stop stdin's `background_tasks`
   array is the harness's own live view of tracked background tasks (`{id, status, command,…}`),
   so arming is verified from the payload with no new disk artifact and no cross-platform pid
   probe. The plan adopts this simpler mechanism and treats the marker file as *not needed*
   (this narrows N1/§3's disk-contract addition and dissolves R8 and most of FR19's
   pid-liveness clause). This is a *simplification*, not a contradiction of intent, but it does
   supersede the PRD's stated mechanism and is flagged so the reviewer can check it.
3. **The wake half of Design B (harness re-invokes on background-task completion) is unproven
   headless.** It is the same interactive assumption the *already-shipped* watch recipe makes,
   so it is not new risk — but it is an assumption, honestly labelled, that only the manual
   smoke test (M3) can close.

---

## Chosen design — Design B (verified arming), background_tasks variant

**Decision: Design B.** Single strongest reason: **it is the only candidate that satisfies U4
/ AC8 (a lead left idle wakes on a reply arriving hours later) without the UI-freeze and the
~10-min / 8-block ceilings that pure Design A imposes** — and the spike proved its one load-
bearing uncertainty (will the model arm the watcher when told, even on Opus-low) is a
non-issue at 5/5 including 2/2 Opus-low, while simultaneously removing Design B's own main
cost (the pid-marker file) by reading `background_tasks` from the Stop payload.

Design B mechanics as validated:
- On `Stop` with live subagents, the hook inspects `background_tasks` (from stdin) for a running
  task whose `command` matches the session's `watch` invocation.
  - **Armed → exit 0 (allow).** Turn ends, UI stays typeable, the tracked watcher carries the
    (unbounded) wait and wakes the harness on the next message — the existing shipped wake path.
  - **Not armed + an unread message already present → block** naming the sender(s) and telling
    the lead to `read_messages` (probe a mechanics; FR3).
  - **Not armed + no unread → block** with the operational arm instruction (probe b; FR4).
- The FR14 progress guard (cursor-advance state file) plus `stop_hook_active` prevent a
  no-progress block loop from reaching the 8-block cap (probe a confirmed `stop_hook_active`
  behaviour; the cap itself was not driven to 8 in the spike — the guard makes that path
  unreachable in normal operation).

Hybrid (short in-hook grace wait then arm-and-release) remains a possible refinement but is not
required: probe b showed arming is cheap and reliable, so the extra in-hook complexity buys
little. The plan may keep a *zero* default grace wait (pure B) and leave a grace-seconds env as
a future tunable.

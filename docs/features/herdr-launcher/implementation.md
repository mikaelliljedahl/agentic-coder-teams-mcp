# Herdr launcher — implementation

Implements `docs/features/herdr-launcher/plan.md` (revision 4, approved in
`plan-review-4.md`).

## What was built

`HerdrProcessManager` in `src/claude_teams/backends/process_manager.py`, a third
Linux launcher that gives each spawned agent its own Herdr tab. Selected only by
`WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`.

Supporting pieces: `HerdrCommandError`, `HerdrServerUnavailableError`,
`HerdrSpawnError`, `_HerdrProbe`, `HerdrProcessInfo`,
`_require_creation_token`, and the pure `_select_linux_manager` that replaced
the inline if/elif launcher chain.

No backend adapter, hook, protocol or disk-contract file was touched.

## Red/green evidence

Built in six slices, each red before green:

| Slice | Red | Green |
| --- | --- | --- |
| Selection + constructor purity + session routing | `AttributeError: module ... has no attribute 'HerdrProcessManager'` | 16 passed |
| `_run_herdr` protocol boundary | 9 failed, 16 passed | 25 passed |
| Spawn | 6 errors (`no attribute '_ensure_server'`) | 31 passed |
| Probe + projections | 19 failed, 31 passed | 50 passed |
| Kill + graceful shutdown | 7 failed, 50 passed | 57 passed |
| Bootstrap, rebinding, capture | 1 failed, 71 passed | 73 passed |

Two failures in the spawn slice were **test** bugs, not production bugs, and are
worth recording: the first asserted a command ending in `exec claude --print`
while `claude-code` correctly appends `--debug-file` (mirroring the tmux
manager), and the second forgot to record the cleanup call it was asserting on.
Both were fixed in the test.

One production behaviour was added without a prior failing test of its own: the
server bootstrap (`_ensure_server`) was written during the spawn slice because
spawn depends on it. Its seven tests were written immediately afterwards and
passed on first run. They are genuine tests, but they did not drive the design.

## Deviations from the plan

1. **`errors="strict"` on `_run_herdr`.** Not in the plan. The repository's own
   guard test (`tests/test_subprocess_decoding.py`) failed the first full-suite
   run, and it was right to: Herdr JSON is a machine protocol, so
   `errors="replace"` would smuggle U+FFFD into text we parse and act on.
   Undecodable output is now a `malformed` command error, with a test.
2. **`docs/reference/agent-messaging-protocol.md` was not modified.** The plan
   listed it, but that document describes the messaging protocol and has no
   launcher section; adding one would have been unrelated scope. The launcher is
   documented in `README.md` instead.
3. **Plan test numbers are not 1:1 with test names.** The suite covers the
   planned behaviours; some planned cases merged into one parametrized test
   (e.g. the launcher-value table), and the `send`/`capture` gating tests are
   parametrized across probe states rather than written out individually.

## Validation

All four gates, whole repository, on Linux:

```
uv run ruff format --check .   # 80 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # see the final tally below
```

The launcher's own tests are counted in the final tally below, not here.

## Live verification — and the four real bugs it caught

A live smoke run against a real Herdr 0.8.2 server (disposable named session
`actest`, deleted afterwards) found **four defects that every unit test had
passed over**, because each was a wrong assumption about the CLI that the fakes
faithfully reproduced:

1. **`status --json` is not enveloped.** It answers with a bare object
   (`{"status":"running","running":true,"socket":...}`), not
   `{"id",...,"result":{"type":...}}`. Validating it as an envelope made every
   server probe look malformed, so the launcher never found a running server and
   timed out starting a new one. Fixed with a separate `_run_herdr_raw` seam.
2. **A fresh headless server has no workspace at all.** `workspaces: []`, and
   `tab create` fails `workspace_not_found: no active workspace`. The earlier
   manual probe only worked because the default session already had a workspace
   from prior use. The first agent now creates the workspace (same options, same
   `root_pane`) and renames the resulting tab back to `<agent>@<team>`.
3. **`pane run` succeeds silently.** Exit 0, no payload. Demanding an envelope
   turned every successful spawn into `malformed`. Silence is now accepted as
   success only when the exit code agrees.
4. **`pane read` prints plain text, not JSON.** Capture was parsing an envelope
   that never existed; it now reads stdout through a `_run_herdr_text` seam.

Each fix was driven by a failing unit test written first, and two pre-existing
tests were corrected because they had encoded the wrong assumption.

### End-to-end result

```
ensure_server -> /home/mikael/.config/herdr/sessions/actest/herdr.sock
spawned handle = 1472390        tab/pane = w1:t4 w1:p4
probe      = _HerdrProbe.OWNED
health     = (True, 'herdr pane w1:p4 alive')
capture    = "... SMOKE_ALIVE_smoke ..."     # --env injection confirmed live
after send = "... # hello from send"         # send reached the pane
tracked after kill = False                   # tab closed, process gone
```

`owns_process(handle, "bogus-token")` returns `True` here, which is correct and
not a hole: `ownership_probe` treats in-memory ownership of a live, fully proven
pane as authority without a second token read (`process_manager.py:332-334`) —
the same contract the tmux manager relies on. The dangerous case, a recycled
PID, is covered by `test_ownership_probe_is_not_fooled_by_a_recycled_pid`.

### Still unproven

The smoke run used a plain `bash` process, not the three real agent CLIs, and it
did not exercise `pane move`, `herdr --handoff` (either shape), capture with an
attached client, or a follow-up/resume. Those remain outstanding from the plan's
live matrix.


## Implementation review round 1 — 0 blockers, 8 majors, all fixed

`implementation-review-1.md` (Codex) found no blockers but eight real defects.
The two that mattered most were both cases of the code not doing what its own
documentation claimed:

1. **False death and false life (MAJOR 1).** `_probe`'s docstring said PID/token
   identity was settled before the pane was consulted. It was not: the pane was
   read first, so one transiently unreadable creation token classified a healthy
   agent `IDENTITY_MISMATCH` and `health_check` reported a **live agent dead** —
   while a dead PID plus a slow CLI reported "alive, pid still ours". Local
   identity is now triaged first (`_local_identity`), which also made
   `IDENTITY_MISMATCH` mean exactly one thing (a recycled PID), and that in turn
   fixed MAJOR 8 for free.
2. **Abandoned agents (MAJOR 7).** `kill_process` dropped the in-memory record
   before the stop had settled. If the token became unreadable at that moment no
   signal was sent, yet `server_simple.kill_agent` still deleted the durable
   record — a live agent with nothing left managing it. It now raises
   `HerdrOwnershipUnprovenError` and keeps the record.

The rest: rebinding was unreachable in production because the endpoint was
cached forever (MAJOR 2, fixed by revalidating per spawn); silent success was
generalised from `pane run` to every command and only one stream was parsed
(MAJOR 3); any status-query failure authorised auto-start, and the named-session
socket path was guessed wrongly — the real layout is
`<config>/sessions/<name>/herdr.sock` (MAJOR 4); a server that missed its
readiness deadline was leaked and retained children were never reaped (MAJOR 5);
a partial create response could raise past cleanup, and a failing provenance
write could leave a live agent the caller believed never started (MAJOR 6).
Minors: refusals are now logged, and the start lock lives under Herdr's config
directory honouring `HERDR_CONFIG_PATH`.

**On test quality.** The review's sharpest point was that several tests "fake
below the behaviour they claim to establish" — the fake server child had only
`poll()`, so it *could not* express cleanup, and the rebinding test set the
cached field by hand instead of scripting discovery. That criticism is correct
and is the same lesson the live run taught. The fakes were rebuilt accordingly;
the suite grew accordingly.

## Nested live test

See `nested-live-test.md`. A spawned agent drove the real server spawn path with
the Herdr launcher and spawned a codex child into a Herdr tab. The first run
failed two checks, both defects in the test driver rather than the launcher; the
corrected rerun passed everything, including the decisive evidence — a fresh
`state-<agent>.json` carrying `event: "Stop"`, which only a hook running
*inside* the spawned agent can write, plus the codex TUI status bar visible in
the pane and the child's message arriving in the lead's inbox.

## Final validation

All four gates, whole repository, on Linux:

```
uv run ruff format --check .   # 81 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1493 passed, 4 skipped
```

105 of those tests are new and specific to this launcher.

## Surfaced, not fixed here

Implementation review round 2 raised one blocker that lives **outside** this
work: `server_simple.kill_agent` collapses `INDETERMINATE` ownership to a
boolean, skips the kill (correctly) but deletes the durable record anyway,
abandoning a live agent. It dates from July (PR #36) and affects every backend.
Recorded in `docs/features/ownership-indeterminate-kill/finding.md` and left for
its own feature, by the repository owner's decision, rather than expanding a
launcher PR into shared lifecycle semantics.

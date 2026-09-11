# Independent implementation review: Herdr launcher

## Summary

The implementation has the right overall shape and preserves the existing launch contract: Herdr remains explicit opt-in, construction is pure, finite CLI calls are bounded, agent commands still come from the backend adapters, the shell command ends in `exec`, creation tokens are required at spawn, and pane input/capture plus tab destruction are ownership-gated.

It is not ready to merge. I found no path that deliberately signals a known foreign PID—the destructive PID fallbacks re-read the non-null creation token—but several state-machine projections do not match the approved design. In particular, `_probe` can report a live agent dead when its token is temporarily unreadable and can report a dead or PID-reused agent alive when Herdr is unavailable. Endpoint rebinding is only simulated by tests and cannot occur through production control flow. There are also lifecycle leaks around failed server startup, failed spawn cleanup, and record removal before a kill has actually been authorized.

The reported quality gates are credible. I independently ran the focused suite and static gates:

- `uv run pytest tests/test_backends/test_process_manager_herdr.py -q`: **80 passed**.
- `uv run ruff check ...`: passed.
- `uv run ruff format --check ...`: passed.
- `uv run ty check`: passed.
- `git diff --check main..HEAD`: passed.

## Findings

### MAJOR 1 — `_probe` does not establish PID/token state before projecting pane state

**Location:** `src/claude_teams/backends/process_manager.py:2619-2648`; `health_check` at `:2679-2693`; approved plan §“Ownership, liveness, kill, capture, send,” especially lines 307-337.

The implementation reads the token first, but only calls `_pid_alive` in the pane-not-found branch. That produces three wrong classifications:

1. If the pane still reports the stored PID but `creation_token` is temporarily unreadable, line 2647 returns `IDENTITY_MISMATCH`. `health_check` then reports a live agent dead. The same happens for a missing/moved pane with an unreadable but live PID because the conditional expression at lines 2640-2644 maps it to `IDENTITY_MISMATCH`, not `INDETERMINATE`.
2. If the PID is dead or reused and `pane process-info` times out, line 2645 returns `INDETERMINATE` without considering the already-observed null/mismatching token. `health_check` consequently reports it alive and even says the PID “still ours.”
3. A token mismatch plus any non-not-found Herdr failure likewise becomes degraded-alive, although the local immutable process identity already proves that the stored process is gone.

This contradicts the plan’s rule that `INDETERMINATE` is degraded-alive only while PID and token still match, and that PID/token liveness is settled before pane projection. `_tracked_alive` remains fail-closed, and the later signal paths revalidate the token, so I did not find a new foreign-PID signal here. The bug is false liveness/death and inconsistency with inherited `ownership_probe`: the two APIs can give opposite answers for the same record.

**Concrete fix:** make local process identity the first explicit triage: dead/non-live PID → `PID_GONE`; live PID with unreadable token → `INDETERMINATE`; readable mismatching token → `IDENTITY_MISMATCH`; only a matching non-null token may proceed to `pane process-info`. A non-not-found pane error can then safely mean `INDETERMINATE`, because local ownership was already re-proven. Add tests for both pane-present and pane-absent unreadable live tokens, and for dead/reused PIDs combined with a Herdr timeout.

### MAJOR 2 — Production never discovers a changed endpoint, so rebinding is unreachable

**Location:** `_ensure_server` at `src/claude_teams/backends/process_manager.py:2457-2478`; `_probe` at `:2619-2656`; tests at `tests/test_backends/test_process_manager_herdr.py:1004-1030`; approved plan lines 177-217 and 307-310.

`_ensure_server` freezes `self.socket_endpoint` and returns it forever once non-null. `_probe` never asks Herdr for the current endpoint. Its rebinding block merely compares the record with that same cached manager field. No production code changes the manager field after the first spawn.

The positive test manually assigns `manager.socket_endpoint = "/run/herdr-new.sock"`; it therefore tests the final assignment block, not endpoint discovery or a handoff. In a real handoff, commands may transparently reach the replacement via the immutable session selector, `_probe` may validate pane/PID/token, and the code will return `OWNED` while both cached endpoint fields still contain the old path. The promised endpoint member of the `OWNED` conjunction, rebind log, and stored-endpoint update never occur.

The pane/PID/non-null-token conjunction still prevents accidental adoption of a different process, so this is not by itself a foreign-object blocker. It is nevertheless a material omission from the approved ownership model.

**Concrete fix:** perform one bounded current-endpoint resolution during probing (or through a separately cached/refreshable endpoint resolver). Keep the candidate local; if it differs, query the pane through the same immutable selector and update the manager/record only after pane ID, shell PID, and token all validate. Resolution failure must remain `INDETERMINATE`. Replace the manual-attribute test with one that scripts old status → new status and proves both successful and rejected rebinding.

### MAJOR 3 — Silent-success handling is global and error-envelope validation is not actually “either stream”

**Location:** `_run_herdr`/`_validated_result` at `src/claude_teams/backends/process_manager.py:2355-2455`; `_parse_envelope` at `:2915-2939`; plan lines 384-390.

The live finding proves that **`pane run`** may succeed silently. The implementation generalizes that fact to every enveloped command: any exit-0 response with both streams empty is converted at lines 2444-2446 into the caller’s requested result type. Thus an empty response from `tab create`, `pane process-info`, `tab close`, `send-text`, or `send-keys` is asserted to have exactly the expected semantics without evidence. Some callers later notice missing fields, but `tab close` and send operations silently accept the fabricated success.

In addition, `_parse_envelope` returns the first dictionary it can parse. If stdout contains a success object and stderr contains an error envelope, stderr is never inspected. That does not meet the stated “reject an error envelope on either stream” boundary; the current parametrized test places the error on only one stream at a time.

**Concrete fix:** add an explicit command-specific `allow_empty_success` policy and enable it only for commands whose silence was observed (currently `pane run`). Parse both non-empty streams, give a structured error envelope precedence, and reject contradictory/multiple responses. Add tests for empty responses from each response-bearing command and for simultaneous stdout-success/stderr-error.

### MAJOR 4 — Server discovery converts protocol and availability failures into “no server”

**Location:** `_run_herdr_raw` at `src/claude_teams/backends/process_manager.py:2395-2412`; `_server_socket` at `:2480-2495`; `_ensure_server` at `:2457-2478`.

`_server_socket` catches every `HerdrCommandError` and returns `None`. A timeout against a busy live server, invalid JSON, a CLI usage regression, an undecodable response, or a missing binary therefore all authorize the auto-start path. The second status check is inside the project lock, but that lock does not coordinate with a user-started Herdr server, so it does not make “failed to query” equivalent to “confirmed absent.” A missing binary also falls through to raw `Popen`, whose `OSError` is not mapped to the named server-unavailable error.

Raw status is not schema-validated either. `{}` is treated as not running, string `"false"` is truthy, and a missing socket is replaced with `_session_socket_hint`. For a named session that hint is demonstrably wrong relative to the live path in `implementation.md:100-102`: it constructs `~/.config/herdr/<name>.sock`, while the observed path is `~/.config/herdr/sessions/<name>/herdr.sock`. It also ignores `HERDR_CONFIG_PATH`.

Finally, after first success `_ensure_server` never probes again. If the server dies, later spawns reuse the cached string and fail at `tab create` instead of applying the advertised “reuse or start” policy.

**Concrete fix:** return “absent” only from a well-formed status object that explicitly reports `running: false`; propagate timeout/malformed/unavailable states. Require boolean `running` and an absolute string socket when running, resolving via `session list --json` if that is the authoritative fallback rather than guessing. Wrap `Popen` failures in `HerdrServerUnavailableError`. Revalidate cached reachability on each spawn (with a bounded retry/restart policy that cannot restart merely on an indeterminate query).

### MAJOR 5 — A server that misses the readiness deadline is leaked, and retained children are never reaped later

**Location:** `_start_server` and `_popen_herdr_server` at `src/claude_teams/backends/process_manager.py:2497-2533`; approved plan lines 248-253 and 391-394.

On timeout, `_start_server` raises while the child may still be running. It neither terminates nor retains that child. The lock is then released, so another process can retry while the first untracked server is still starting. On successful startup the child is assigned to a dynamically created `_server_child`, but no later code polls or waits it; if the server subsequently exits, it can remain a zombie until the MCP host exits. The plan explicitly required retention **for reaping**.

The fake child has only `poll`, so the tests cannot express timeout cleanup or later reaping.

**Concrete fix:** initialize the child field in the pure constructor, retain the child immediately after `Popen`, and provide bounded cleanup on failed readiness for the exact child this manager started (`terminate`/wait, then force only if necessary). Poll/reap the retained child opportunistically on later ensure/probe operations without ever stopping a healthy server merely because the manager exits. Extend the Popen seam/fake with `wait`, `terminate`, and required state assertions.

### MAJOR 6 — Failed spawn cleanup is incomplete and can leave an unregistered live tab/agent

**Location:** `spawn_process` at `src/claude_teams/backends/process_manager.py:2574-2617`; `_close_tab_quietly` at `:2895-2898`; approved plan lines 279-291.

The cleanup `try` starts only after both `root_pane` and `tab` have been validated. If Herdr returns a real `tab.tab_id` but a missing/malformed root pane, lines 2577-2584 raise without closing the known-created tab. That is especially relevant after the live tests established that response assumptions were unreliable.

There is a second orphan path after registration: opening/writing the provenance log at lines 2612-2616 can raise. The caller sees spawn failure and therefore does not persist an agent record, while the tab, agent process, and manager’s private record remain live. Cleanup errors are also suppressed rather than attached to the original exception, contrary to the approved plan.

**Concrete fix:** establish a cleanup scope immediately after `_create_tab`, extracting any usable tab ID before validating the remainder. Make provenance logging best-effort or complete it inside the rollback scope before committing `_processes`. If rollback fails, preserve the original exception and attach/log the cleanup failure explicitly. Tests should cover malformed partial create responses, log-write failure, and tab-close failure during rollback.

### MAJOR 7 — `kill_process` forgets the only management record before its PID fallback is authorized

**Location:** `src/claude_teams/backends/process_manager.py:2720-2748`; caller at `src/claude_teams/server_simple.py:5876-5895`.

Both branches pop `_processes` before the operation has settled. A concrete `PANE_GONE` sequence is: `_probe` reads the matching token; the second token read in `_kill_if_still_ours` is transiently unreadable; no signal is sent; `kill_process` returns normally with the in-memory record gone. The outer `kill_agent` then deletes the durable record because `kill_process` has no failure result. The still-live moved agent is now wholly unmanaged. The same can happen after an `OWNED` tab-close failure (which is suppressed) followed by an unreadable fallback token.

Token mismatch must continue to withhold signalling because it proves the numeric PID is no longer the original process. An unreadable token while the PID remains live is different: it is indeterminate, not successful destruction.

**Concrete fix:** make the fallback return a tri-state/raise a named indeterminate error. Remove the in-memory record only after the original PID is proven gone/reused or an authorized stop operation has at least been issued; retain it and raise when a live token is unreadable so `server_simple.kill_agent` cannot commit durable deletion. Test the token changing to `None` between probe and fallback, tab-close failure plus unreadable token, and successful retry.

### MAJOR 8 — `graceful_shutdown` treats any object identity mismatch as proof the process exited

**Location:** `src/claude_teams/backends/process_manager.py:2750-2773`; follow-up caller at `src/claude_teams/server_simple.py:4518-4527`.

The polling loop returns `True` for `IDENTITY_MISMATCH`. That state can mean a token mismatch (the original process is gone), but it can also mean the pane reports a different shell PID while the original PID still exists with the stored token. In the latter case the function reports successful shutdown of a process that is demonstrably still alive. The follow-up path interprets `True` as “do not call force kill” and immediately starts the resumed agent, allowing old and resumed workers to overlap.

The moved-pane test calls `graceful_shutdown` but does not assert its return value, and no test covers a matching original PID/token combined with a later pane identity mismatch.

**Concrete fix:** determine shutdown completion from the original PID identity, not from Herdr object state: success when the PID is dead or its readable token differs (the original is gone); continue polling/return false while PID+token still match; treat an unreadable live token as indeterminate, never success. Add the missing transition tests and assert the moved-pane timeout result.

### MINOR 1 — Non-owned `send`/`capture` do not record the reason promised by the plan

**Location:** `src/claude_teams/backends/process_manager.py:2695-2718`; plan lines 361-377.

Gating is correct, but `_owned_info` discards the probe state and callers silently no-op/return `""`. The approved design required a recorded reason so an operator can distinguish unknown handle, moved pane, control-plane outage, and identity mismatch.

**Concrete fix:** have the gate return or log the state and write a bounded diagnostic to the normal logger or agent log. Do not change the public signatures.

### MINOR 2 — The startup lock does not follow the approved canonical config-directory policy

**Location:** `src/claude_teams/backends/process_manager.py:2535-2551`; plan lines 231-246.

The implementation correctly reuses `claude_teams.filelock.file_lock`, but places the lock under `XDG_RUNTIME_DIR` or the global temporary directory rather than under Herdr’s config directory. It is keyed by session name, not the resolved/configured target. Processes with inconsistent runtime environments can therefore use different locks for the same Herdr endpoint, and the predictable `/tmp` fallback is weaker than a user-owned config location. The implementation also ignores the documented `HERDR_CONFIG_PATH` override shown by `herdr --help`.

**Concrete fix:** derive one user-owned pre-server lock namespace from Herdr’s effective config path and the validated immutable selector, as the plan specified. Add a test showing two processes with different unrelated team state but the same Herdr config/session derive the same path.

### MINOR 3 — Strict UTF-8 decoding is applied to terminal display text as if it were JSON

**Location:** shared `_invoke` at `src/claude_teams/backends/process_manager.py:2369-2393`; `_run_herdr_text` at `:2414-2425`; neighboring `read_log_tail` at `:2220-2228`.

Strict decoding is appropriate for JSON control data, and the repository guard rightly prevented an implicit/default decoding policy. But `_run_herdr_text` uses the same strict boundary for arbitrary pane output. A single undecodable byte in terminal content makes capture raise `malformed`; neighboring display/log capture explicitly uses replacement decoding. The implementation comment at lines 2381-2383 describes all output as a machine protocol, which is no longer true after adding the plain-text seam.

**Concrete fix:** keep strict UTF-8 for raw/enveloped JSON and give the text seam an explicit display policy (`errors="replace"`, or bytes plus a deliberate decoder). Test invalid pane-output bytes independently from invalid JSON bytes.

### MINOR 4 — README overstates session visibility, and implementation evidence has stale test counts

**Location:** `README.md:280-283`; `docs/features/herdr-launcher/implementation.md:59-70`; session-policy plan lines 223-229.

The code never calls `status client`, so it cannot establish that a reused server is “the session you are sitting in” or that new tabs are visible. With no pinned name it selects Herdr’s default session, which need not be the caller’s named attached session. The implementation document also says 73 tests are new while the focused file currently executes 80 cases after the second commit.

**Concrete fix:** either implement the approved attached/headless distinction and report visibility from evidence, or soften README to “reuses the selected/default server; tabs are visible if a client is attached to that server.” Update the evidence count to 80.

## Fidelity audit

Implemented faithfully:

- explicit `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` selection with no `HERDR_ENV` auto-detection;
- pure import-time construction and immutable validated session name;
- top-level `--session` routing for every subprocess path;
- lazy first-spawn server resolution and reuse of the shared `file_lock` primitive;
- backend-built argv, environment identity variables, correlation/prompt transport, model/reasoning arguments, and a PTY-preserving shell command;
- non-null creation-token enforcement with rollback on the tested post-create failures;
- pane PID as the handle, strict `OWNED` requirement for `_tracked_alive`, and token revalidation before PID signals;
- `send` and `capture` gated on `OWNED`; tab close withheld outside `OWNED`;
- `lines=None` implemented by omitting `--lines`.

Not implemented as approved, or materially different without being recorded as a deviation:

- PID/token-first probe projection (MAJOR 1);
- discoverable/proven endpoint rebinding (MAJOR 2);
- command-specific strict response validation (MAJOR 3);
- confirmed-absence server startup and recovery after server death (MAJOR 4);
- retained child reaping and failed-start cleanup (MAJOR 5);
- rollback of every post-create failure with cleanup outcome attached (MAJOR 6);
- non-leaking kill record lifecycle and truthful graceful completion (MAJOR 7-8);
- recorded non-owned send/capture reasons (MINOR 1);
- config-directory canonical lock path (MINOR 2);
- attached-client/headless distinction through `status client` (MINOR 4).

`resolve_agent_pid` correctly remains inherited. For a successful launch, `_build_posix_shell_command` ends in the shell builtin `exec`, so the pane shell PID is replaced by the agent without changing PID. Selecting a foreground child would be less stable. This does not prove that an asynchronously injected command successfully reached `exec`; the live smoke used bash, and the suite still lacks real Claude Code/Codex/pi startup and resume coverage.

## Test-quality assessment and remaining live risk

The current suite would catch exact reintroductions of all four live-found defects:

1. Bare status object: `test_status_json_is_read_as_a_bare_object_not_an_envelope`.
2. Empty fresh server workspace: `test_spawn_creates_a_workspace_on_a_fresh_server`.
3. Silent successful `pane run`: `test_run_herdr_accepts_a_silent_success`.
4. Plain-text `pane read`: `test_capture_reads_plain_text_not_json`.

That is meaningful improvement, but many tests fake below the behavior they claim to establish. `_Herdr` maps the requested `expect` value directly to a matching response, so it restates the implementation’s assumption instead of validating a real command contract. Probe projection tests omit the cross-product that exposes MAJOR 1. The rebinding test mutates the cache instead of faking endpoint discovery. Kill/graceful tests replace `_probe` wholesale, and the moved graceful test does not inspect completion. The Popen fake cannot test reaping or cleanup. No test combines both output streams, a malformed raw status, a partial tab-created response, a failed provenance write, or a token becoming unreadable between proof and signal.

The live smoke also leaves important integration properties unproven:

- all three real agent CLI interactive forms under the Herdr PTY, including immediate CLI failure and Herdr agent detection;
- lossless multiline/quoted prompt plus correlation marker transport for Codex and pi, and Claude’s prompt sidecar path;
- model/reasoning-tier argv in a real spawn;
- pane move followed by health, send/capture gating, graceful stop, and force stop;
- both forms of `--handoff`, including actual endpoint-path behavior;
- an attached client versus a headless selected session;
- server death followed by health checks and a later spawn;
- native follow-up/resume, ensuring the old agent is truly stopped before the replacement starts.

These live gaps do not independently prove more code defects, but they are precisely where the faked tests cannot validate Herdr 0.8.2 behavior and should be exercised after the major fixes.

## Verdict

VERDICT: CHANGES REQUIRED — 0 BLOCKERS

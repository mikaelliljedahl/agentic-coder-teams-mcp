# Independent plan review: Herdr launcher

## Summary

The launcher is feasible, and using `pane run` with the backend-produced argv is the right general direction: it leaves model selection, prompt transport, correlation-marker insertion, MCP configuration, and lifecycle-hook arguments in the existing backend layer. The plan is not ready to implement, however. Two design holes can make lifecycle operations address the wrong Herdr session or authorize a PID operation from pane liveness that does not prove the numeric PID is still the original process.

The review used only the requested repository files and read-only CLI discovery (`herdr --help`, `herdr --skill`, and relevant `--help` subcommands). No Herdr server/status/control or mutating command was run.

## Findings

### BLOCKER 1 — Named-session routing is not specified for every command

**Concern:** `docs/features/herdr-launcher/plan.md:118-132` (Session policy), `:139-168` (Spawn/lifecycle), and the proposed `HerdrProcessInfo` at `:136-137`.

The plan says `WIN_AGENT_TEAMS_HERDR_SESSION` pins a named session and that auto-start “uses that name,” but every illustrated control invocation is an unqualified `herdr status server`, `herdr tab ...`, or `herdr pane ...`. Installed CLI help defines session selection as a top-level prefix:

```text
herdr --session <name> [options]
```

Therefore starting `herdr --session foo server` and then issuing `herdr tab create ...` does not establish that the latter addresses `foo`. It can fail against the default socket or, worse, address an independently running default session. Since tab and pane IDs are only meaningful within the selected session, an incorrectly routed `tab close` has user-data blast radius. The proposed process info also has no session identity with which to verify or construct later calls.

**Suggested fix:** Define one `_herdr_argv(*args)` helper that prepends `[herdr, "--session", configured_name]` whenever a session is pinned, and require every status, start, tab, pane, capture, send, and cleanup call to go through it. Make session identity immutable per manager (and preferably record it in `HerdrProcessInfo` for diagnostics). Specify validation of the configured session name and test exact argv for every operation. Do not infer that starting a named server changes later unqualified CLI routing.

### BLOCKER 2 — Pane liveness alone does not prove ownership of the PID handle

**Concern:** `docs/features/herdr-launcher/plan.md:136-164` and `:194-196`; `_PidOwnershipMixin` in `src/claude_teams/backends/process_manager.py:245-386`.

The mixin treats `_tracked_alive(info)` as conclusive ownership and bypasses `expected_token` when it returns true (`ownership_probe`, lines 332-345). The plan proposes that `_tracked_alive` prove only that the pane ID is live, while the public handle is the pane shell PID. Those are not equivalent invariants. A live pane can only prove that this is the tracked terminal object; it does not, by itself, prove that its current `shell_pid` still equals `info.pid`, nor that the numeric PID has not been reused. This becomes dangerous in `kill_process`: after `tab close`, the plan waits for the original PID and then calls `_kill_pid`. If the original PID exited while the pane remained/recovered and the number was reused, pane liveness can make `owns_process` return `OURS`, and the fallback can signal a foreign process.

The proposed `HerdrProcessInfo` does not store a creation token, so it cannot close this gap in-memory. The assertion that closed pane IDs are never reused is insufficient: it protects pane addressing, not the separate PID namespace. It is also not established for IDs across a server death/restart; auto-starting a fresh server while stale in-memory entries remain makes that distinction important.

**Suggested fix:** Make `_tracked_alive` require all of: the expected session, exact pane identity, a valid `process-info` result, `reported shell_pid == info.pid`, and a stored creation token matching the current PID. Add `creation_token` (and session identity) to `HerdrProcessInfo`. Fail closed on malformed or ambiguous process data. Before any PID fallback after closing a tab, revalidate the token; alternatively, never issue a bare-PID fallback for tracked Herdr tabs. Explicitly define behavior after the Herdr server generation changes, and test pane-live/PID-changed, token mismatch, PID reuse, and server-restart ID collision scenarios.

### MAJOR 1 — `HERDR_ENV=1` auto-detection is a behavior change, contrary to the plan

**Concern:** `docs/features/herdr-launcher/plan.md:104-116`; import-time selection at `src/claude_teams/backends/process_manager.py:2249-2254`.

Today every Linux process that does not explicitly select tmux gets `LinuxTerminalProcessManager`. After this change, any existing MCP server launched from a Herdr-managed pane with `HERDR_ENV=1` and `herdr` on `PATH` silently switches launchers. That is exactly an existing-deployment behavior change, despite line 116. It can also affect tests or developer shells run inside Herdr. `HERDR_ENV` establishes caller context, not user consent to make Herdr the win-agent-teams launcher.

Import-time manager construction is safe only if `HerdrProcessManager.__init__` is pure. If construction probes or starts the server, merely importing the module becomes a subprocess/state mutation and selection tests become order-dependent. Even a pure `shutil.which` decision is frozen for the process lifetime.

**Suggested fix:** Make Herdr strictly opt-in through `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`; use `HERDR_ENV` only as context for choosing the current session after explicit selection. If auto-detection is a firm product requirement, call it a compatibility change, provide an opt-out, distinguish truly unset from blank/unknown explicit values, and test all those cases. Keep construction pure and defer binary/server checks to `spawn_process`. Prefer a pure selector function that tests can call without reloading a module-level singleton.

### MAJOR 2 — “Running server” is not the same as an open user session, and server ownership is underspecified

**Concern:** `docs/features/herdr-launcher/plan.md:118-132` and Risks `:182-193`.

`herdr status server` establishes server reachability, not that a UI client is attached or that newly created tabs will be visible in the user’s window. The plan’s “reuse the user's open session” and “tabs land in their window” claims are therefore stronger than the probe described. CLI help exposes separate `status server` and `status client` commands, reinforcing that they are distinct states.

The auto-start policy also does not specify process supervision: exact `Popen` flags, stdio redirection, reaping, whether the child is retained, or what happens if the launching MCP server exits. Two MCP processes can both observe “not running” and start; “re-probe after failed start” does not prevent duplicate launch attempts, leaked `Popen` children, or ambiguity about which process owns cleanup. Reusing the default user server couples every spawned agent to user actions such as an update, session stop/delete, or server crash. “We never call server stop” does not mitigate those failure modes. If a subsequent spawn auto-starts a new empty server after the old one died, stale agent records and stale pane IDs remain in the process manager.

**Suggested fix:** Separate and document three states: reachable server with attached client, reachable headless server, and no server. Decide whether visibility is required or merely best effort. Prefer an explicitly selected/dedicated named session for auto-started service state, or require the operator to run Herdr rather than silently owning a daemon. If auto-start remains, serialize startup across processes, launch with non-inherited stdio and a defined lifetime, retain/reap the child, detect early exit, and never treat a newly started server as continuity for old pane records. Document that server loss kills all panes and forces all affected agent records dead; add tests for that transition.

### MAJOR 3 — `resolve_agent_pid` chooses a transient process without a stable rule

**Concern:** `docs/features/herdr-launcher/plan.md:160-161`; `resolve_agent_pid` contract at `src/claude_teams/backends/process_manager.py:278-290` and Linux terminal precedent at `:1892-1908`.

The plan chooses `foreground_processes[0].pid`, but neither the plan nor CLI help establishes ordering or that the first foreground process is the long-lived agent. Interactive agents routinely run hook, tool, shell, and helper processes in the foreground. Returning one of those transient PIDs can make server liveness reporting flip to dead while the pane and agent remain alive. The design’s own shell command ends with `exec <agent>`, so the shell PID should become the agent PID without changing PID; that makes the persisted handle more stable than a foreground snapshot.

**Suggested fix:** Return the handle by default. Override only if Herdr provides a documented, stable field that identifies the pane’s root agent process, with explicit validation and fallback. If foreground resolution is retained, specify selection semantics for empty/multiple lists, require a stable match across polls or ancestry to the shell/root, and test transient helper processes and malformed entries.

### MAJOR 4 — The Herdr JSON/error protocol and `capture` return semantics are underdefined

**Concern:** `docs/features/herdr-launcher/plan.md:50-88`, `:144-168`, and Risk 3 at `:189-191`; the existing `capture` behavior in `src/claude_teams/backends/process_manager.py:1513-1534` and `:1960-1965`.

“All parsing goes through one helper” is not enough to define success. The CLI uses JSON on stdout for success, JSON on stderr for server errors, exit 1 for runtime errors, and exit 2 for syntax errors. The plan does not say whether the helper first checks the return code, rejects an `error` envelope on either stream, validates the response type/id, or bounds subprocess time. Those distinctions matter because malformed output in ownership checks must fail closed, while mutating calls must surface actionable errors.

`capture` also does not define which JSON result field becomes the returned text, behavior on a nonzero first read, or `lines=None`. The existing tmux manager interprets `None` as the full available history. Falling back from empty `recent-unwrapped` to `visible` may yield only a viewport, not equivalent full history, and an empty successful pane is indistinguishable from a source bug unless the response shape is validated.

**Suggested fix:** Specify a single timeout-bounded `_run_herdr` result parser with exact envelope/type/field validation and a named exception carrying sanitized stderr. Define per-call policy for runtime errors (not-found means dead only for liveness; other errors are indeterminate/failure). Record real `pane read` and `process-info` fixtures, define `capture(None)` versus `capture(N)`, and test empty output separately from malformed/nonzero responses.

### MAJOR 5 — The test plan misses the safety properties and cannot validate several live assumptions

**Concern:** `docs/features/herdr-launcher/plan.md:198-228`.

Faking the subprocess boundary is appropriate for unit tests and can cover most argv/result parsing without a live server. It is structurally consistent with this module’s use of `subprocess.run` plus a separate long-lived `Popen` for server startup. For testability, the Herdr implementation should centralize those boundaries (analogous to the Windows manager’s `_popen` seam) rather than globally monkeypatching unrelated subprocess calls. The claim that this mirrors the existing Windows test suite is not substantiated by the files this review was instructed to read.

The 14 cases omit the highest-risk behaviors. Add at least:

- `creation_token`, `owns_process`, and three-valued `ownership_probe` tests: tracked pane with changed PID, matching/mismatching/unreadable token, dead PID, reused PID, and malformed `process-info`.
- Named-session prefix propagation to every command, including cleanup, capture, send, status, and server start; prove a configured session can never address the default session.
- Unknown/blank/case/whitespace launcher values, binary absent, `HERDR_ENV` absent/present, non-Linux selection, and proof that import/constructor launches no subprocess.
- Success/error-envelope validation, invalid JSON, missing/wrong-typed IDs, non-positive `shell_pid`, runtime exit 1, syntax exit 2, and subprocess timeout.
- Failure at each spawn phase: tab-created response partially malformed, `pane run` failure, process-info failure, cleanup-close failure, and preservation of the original exception plus cleanup context.
- Concurrent auto-start outcomes: winner/loser, early server exit, never-ready server, existing headless server, attached-client distinction, and old-server-death followed by a new generation.
- `kill_process` when tab close reports already gone, when close fails, when PID exits, when PID does not exit, and when the PID token changes before fallback; `graceful_shutdown` timeout and CLI-error behavior.
- `capture(lines=None)`, nonzero/invalid first and fallback reads, an actually empty pane, and JSON text extraction; send of empty, multiline, quote-heavy, leading-dash, and non-ASCII text.
- `resolve_agent_pid` with zero/one/multiple foreground entries and transient helpers.
- Backend integration/characterization for claude-code, codex, and pi proving the final argv token and environment survive shell quoting byte-for-byte, including prompt sidecar instruction, correlation marker, model/effort flags, and hook/MCP arguments. Include resume, which creates a new OS process through the same manager.

Some properties cannot be established by a fake CLI: whether `tab create` returns only after the shell accepts input, whether `shell_pid` remains stable through `exec`, pane behavior after agent exit, ID behavior across server restart, PTY behavior for all three interactive backends, and capture behavior with/without an attached client. Add an isolated named-session integration test or documented manual matrix for those. The smoke plan should include pi, use a disposable named Herdr session rather than the user's default session, and verify spawn plus follow-up/resume, not only inbox exchange and kill.

### MINOR 1 — The documented manager interface omits public methods the codebase relies on

**Concern:** `docs/features/herdr-launcher/plan.md:33-43`; `_PidOwnershipMixin` at `src/claude_teams/backends/process_manager.py:259-365`; backend convention at `ADDING-A-BACKEND.md:131-135`.

The list omits `provides_tty`, `creation_token`, `owns_process`, and `ownership_probe`. The proposed class will inherit them, so this is not an implementation-interface blocker once `_processes`, `_pid_alive`, and `_tracked_alive` are correct, but these are not optional details: `provides_tty` selects interactive versus headless backend command shapes, `creation_token` populates the registry, `owns_process` gates destruction, and `ownership_probe` gates lease/claim reclaim. Conversely, `_tracked_alive` and `_pid_alive` are internal subclass obligations, not methods normal callers should use.

**Suggested fix:** Replace the informal list with the complete public interface and a separate list of mixin subclass obligations. Add direct tests for all inherited public behavior, especially `provides_tty` before command construction and the three ownership methods.

### MINOR 2 — Environment-key validation is attributed to the wrong helper/layer

**Concern:** `docs/features/herdr-launcher/plan.md:141-143`; `_validate_safe_name` at `src/claude_teams/backends/process_manager.py:68-74`; `ADDING-A-BACKEND.md:142-147`.

The “existing safe-name check” in `process_manager.py` validates team/agent identifiers and raises `ValueError`; it is not the environment-key validator and imposes a 64-character name policy. Repository convention says env keys are validated upstream by `process_base._spawn_with_command` using `_SAFE_ENV_KEY`, with `InvalidEnvVarNameError`. A direct manager unit test expecting that exception is therefore not justified unless the new manager deliberately adds a second validation boundary.

**Suggested fix:** State that `spawn_process` receives already validated env keys from `BaseBackend`, or introduce/reuse the actual env-key validator if defense in depth is desired. Do not use `_validate_safe_name` for environment variables. Test values containing spaces, quotes, newlines, `=`, and non-ASCII as well as invalid keys.

### MINOR 3 — Several “verified” or risk claims need narrower wording

**Concern:** `docs/features/herdr-launcher/plan.md:50-100`, `:122-128`, `:185-196`.

The observed commands support the basic CLI strategy, but the plan promotes point observations into broader guarantees: a running server is called an open user session; `visible` returning full content once is treated as a general capture fallback; pane-ID non-reuse is applied as if it proves PID ownership and spans server generations; and a named-session socket path is promised in errors although only the default socket path is recorded. CLI help verifies command grammar, not those lifetime/ordering guarantees.

**Suggested fix:** Separate “observed on 0.8.2 in one run” from documented invariants. Record exact versioned JSON fixtures and identify assumptions requiring isolated integration tests. Discover/report the active socket/session from authoritative CLI output rather than hardcoding a named-session path.

## Contract assessment

Subject to the findings above, `pane run` is preferable to `herdr agent start` for this launcher. The existing backend builds `cmd` before `spawn_process`; that argv already contains resolved executable paths, model/effort and permission flags, hook/MCP wiring, prompt-sidecar instructions where applicable, and the per-spawn correlation marker. `_build_posix_shell_command` uses `shlex.join` and ends in `exec`, so passing that complete string as the one `COMMAND` token accepted by `herdr pane run <PANE_ID> <COMMAND>...` can preserve argv semantics and a real pane PTY. Herdr `tab create --env K=V` can preserve process identity variables, while the shell export is redundant defense in depth.

The plan should not claim this contract is proven solely by manager-level fake tests. Add backend integration characterization for spawn and resume and isolated live verification of PTY/readiness. The on-disk messaging contract itself need not change: state markers, inboxes, per-agent MCP identity, correlation binding, receipt confirmation, and prompt files remain backend/server responsibilities rather than Herdr responsibilities.

VERDICT: CHANGES REQUIRED — 2 BLOCKERS

# Herdr launcher for Linux

> Revision 4. Revised after `plan-review-1.md` (2 blockers),
> `plan-review-2.md` (1 new blocker) and `plan-review-3.md` (0 blockers,
> 3 majors). Every finding's disposition is recorded in
> "Review disposition".

## Scope

Add a third Linux launcher, `HerdrProcessManager`, that spawns each agent into
its own [Herdr](https://herdr.dev) tab, alongside `TmuxProcessManager` and
`LinuxTerminalProcessManager`. Herdr is a terminal workspace manager for coding
agents shipped with Omarchy (`local/herdr 0.8.2-1`, `/usr/bin/herdr`).

Goal: on a native Linux desktop, reproduce the Windows Terminal "one tab per
agent" experience that `WindowsProcessManager._spawn_in_terminal_tab` gives
today, without changing the messaging protocol or the on-disk session contract.

Out of scope: the disk contract, the delivery protocol, hooks, and every backend
adapter (`claude_code.py`, `codex.py`, `pi.py`). No Windows changes. Adding
`foot` to `LinuxTerminalProcessManager._discover_terminal` is a separate fix and
deliberately not in this branch.

## Current behavior

`process_manager.py:2249-2254` selects the manager at import time:

```python
if os.name == "nt":
    process_manager = WindowsProcessManager()
elif os.environ.get(_LINUX_LAUNCHER_ENV, "").strip().lower() == _TMUX_LAUNCHER_VALUE:
    process_manager = TmuxProcessManager()
else:
    process_manager = LinuxTerminalProcessManager()
```

`_LINUX_LAUNCHER_ENV` is `WIN_AGENT_TEAMS_LINUX_LAUNCHER` (`:55`), values
`terminal` (`:57`) and `tmux` (`:58`).

### The interface a manager must satisfy

Public methods callers use (inherited from `_PidOwnershipMixin`, `:245`, unless
the manager defines them):

| Method | Source | Notes |
| --- | --- | --- |
| `spawn_process(request, cmd, env, backend_type, *, is_interactive=False)` | manager | returns `SpawnResult` |
| `health_check(handle, expected_token=None)` | manager | `(alive, detail)` |
| `kill_process(handle, timeout_s=10.0)` | manager | |
| `graceful_shutdown(handle, timeout_s=10.0)` | manager | |
| `capture(handle, lines=None)` | manager | |
| `send(handle, text, *, enter=True)` | manager | |
| `log_path(team_name, agent_name)` | manager | |
| `provides_tty(backend_type, *, is_interactive=False)` | mixin `:263` | default `True`; gates TUI vs head-less command shape |
| `creation_token(handle)` | mixin `:259` | persisted into the agent registry |
| `owns_process(handle, expected_token)` | mixin `:352` | fail-closed **destruction** gate |
| `ownership_probe(handle, expected_token)` | mixin `:311` | three-valued **reclaim** gate |
| `resolve_agent_pid(handle, team, agent)` | mixin `:278` | default returns `handle` |

Subclass obligations (not caller API): `_processes`, `_pid_alive(handle)`,
`_tracked_alive(info)`.

The critical invariant, verified by reading `ownership_probe` (`:332-334`):

```python
if self._has_live_registry_entry(handle):
    return OWNERSHIP_OURS
```

In-memory tracking short-circuits the creation-token check entirely. So
`_tracked_alive` **is** the ownership proof, and it must be PID-reuse-safe on
its own — as its own docstring (`:292-299`) demands.

Also verified: `_build_posix_shell_command` (`:85-89`) ends in
`exec {shlex.join(cmd)}`, so the pane's shell PID *becomes* the agent process.
And env keys are validated upstream in `process_base.py:100` against
`_SAFE_ENV_KEY`, raising `InvalidEnvVarNameError` — `_validate_safe_name`
(`:68`) is for team/agent names only.

## Observed Herdr behavior

Observed on **herdr 0.8.2, one run, on this machine**, against a disposable
headless server that was stopped afterwards. These are point observations, not
documented invariants; the ones this design depends on are re-verified by the
integration matrix below.

- Success is JSON on stdout (`{"id": ..., "result": {...}}`); server errors are
  JSON on stderr with exit 1 (`{"id": ..., "error": {"code", "message"}}`);
  CLI syntax errors exit 2.
- `herdr tab create --cwd <dir> --label <text> --env K=V --no-focus` returned:

  ```json
  {"id":"cli:tab:create","result":{"root_pane":{"cwd":"/tmp","pane_id":"w3:p2",
   "tab_id":"w3:t2","terminal_id":"term_65b360f726a6b2","workspace_id":"w3"},
   "tab":{"label":"alice@team1","tab_id":"w3:t2"},"type":"tab_created"}}
  ```

  `--env` repeats. **Injection observed**: a pane created with
  `--env AGENT_NAME=alice --env AGENT_SESSION_ID=sess123 --cwd /tmp` reported
  `NAME=[alice] SESS=[sess123] PWD=[/tmp]`.
- `herdr pane run <pane_id> <command...>` sent command text plus Enter; exit 0,
  no stdout payload.
- `herdr pane process-info --pane <id>` returned `shell_pid` and
  `foreground_processes[]` (`pid`, `name`, `argv`, `cwd`).
- `herdr pane read <pane_id> --source <visible|recent|recent-unwrapped|detection>
  --lines N`. **Caveat observed:** against a headless server with no attached
  client, `recent` and `recent-unwrapped` were empty while `visible` returned
  the pane content. Whether that is "no attached client" or "nothing has
  scrolled yet" is *not* established; the design treats it as "any one source
  may be empty", not as a rule about clients.
- `herdr tab close <tab_id>` returned `{"result":{"type":"ok"}}` and the tab left
  `tab list`. `herdr session list` printed `name status directory socket` with
  `default` on `~/.config/herdr/herdr.sock`. `herdr status server` reports
  `status: running|not running`; `herdr status client` is a *separate* command.
  With no server, control commands fail `{"error":{"code":"server_not_running"}}`.
- Session selection is a **top-level prefix**, per `herdr --help`:
  `herdr --session <name> [options]`. It is not a per-subcommand flag.
- Panes are real PTYs, so the inherited `provides_tty` default (`True`) is
  correct and is deliberately not overridden.

### Why `pane run`, not `herdr agent start`

`herdr agent start <name> --kind <pi|claude|codex|...> --pane <id>` would give
Herdr's own lifecycle states, but only accepts a *kind* plus native args after
`--`, launching the canonical executable itself. It cannot carry the command
line the backend already built — resolved executable path, model/effort flags,
permission flags, hook/MCP wiring, prompt-sidecar instruction, and the per-spawn
correlation marker. `pane run` takes our `_build_posix_shell_command` string as
its one `COMMAND` token, preserving argv semantics and the PTY; Herdr still
auto-detects the agent in the pane, so its lifecycle UI comes along anyway.

This preserves the contract at the *manager* boundary only. Proof that the final
argv and env survive byte-for-byte is a **backend integration** obligation
(tests 25-26), not something manager-level fakes can establish.

## Proposed design

### Launcher selection — strictly opt-in

Add `_HERDR_LAUNCHER_VALUE = "herdr"` and a **pure** selector function so tests
need not reload a module-level singleton:

```python
def _select_linux_manager(launcher: str) -> ...:  # pure, no subprocess
```

Herdr is chosen **only** by `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`.
`HERDR_ENV=1` **neither selects the launcher nor supplies a session name** — it
is a boolean context marker and nothing more. Rationale: auto-detection would
silently switch launchers for anyone whose MCP server happens to start inside a
Herdr pane, which is a behavior change for existing deployments, and `HERDR_ENV`
signals caller context, not consent.

`HerdrProcessManager.__init__` is **pure**: no `shutil.which`, no server probe,
no subprocess. Binary and server checks happen in `spawn_process`, so importing
the module never mutates state and selection tests are order-independent.

### Session routing

Session selection is a **top-level prefix** (`herdr --session <name>`), never a
per-subcommand flag. The manager resolves its session **once**, at construction,
and treats it as immutable. Every invocation goes through:

```python
def _herdr_argv(self, *args: str) -> list[str]:
    base = [self._binary]
    if self._session:                      # pinned by configuration
        base += ["--session", self._session]
    return [*base, *args]
```

No command may build argv any other way — including `status`, server start,
`tab`, `pane`, capture, send, and cleanup. A configured session must never be
able to address the default session: pane/tab IDs are only meaningful within a
session, and a misrouted `tab close` would destroy the user's own tab.

**Identity is the resolved socket endpoint, not an env name.** Herdr 0.8.2
documents only `HERDR_ENV` (a boolean context marker) plus
`HERDR_WORKSPACE_ID`/`HERDR_TAB_ID`/`HERDR_PANE_ID`; it does not document a
`HERDR_SESSION` name variable, so one must not be invented. The manager
therefore:

- takes an optional pinned **name** from `WIN_AGENT_TEAMS_HERDR_SESSION` only.
  The **constructor is pure**: it captures and validates that name and nothing
  else — no `which`, no probe, no subprocess.
- **Lazily**, during the first `spawn_process` server-ensure step, resolves that
  name (or the unpinned default) to a canonical **socket path** via
  `herdr session list --json` / `herdr status server --json` (both `--json`
  flags verified present in 0.8.2 `--help`), and freezes it for the manager's
  lifetime.
- stores `session_name` (may be `None`) and the canonical **socket path** in
  `HerdrProcessInfo`.

**Stable identity is the canonical socket path / session selector — not the
socket inode.** Storing `(st_dev, st_ino)` and demanding equality would make
`herdr --handoff`, a supported continuity event that replaces the endpoint while
keeping panes and PIDs alive, turn every existing record permanently
non-`OWNED`: capture, input and `tab close` would all be lost even though the
agent is demonstrably the same process. That is the removed generation guard
returning by the back door.

Instead, endpoint replacement triggers **rebinding**, which is proven rather
than assumed: if the newly resolved endpoint reports the stored `pane_id` with
the same `shell_pid` and the same non-null creation token, the record is the
same agent and the stored endpoint is updated. Anything less does not rebind.

Three constraints keep that from becoming an adoption hole:

- The candidate endpoint must be resolved from the **same immutable session
  selector** the manager was constructed with. The implementation must never
  scan `session list` rows and adopt whichever one reports a matching-looking
  tuple — a second session presenting the same pane/PID text is not our agent.
- Rebinding is **logged** with the old and new canonical paths and the proof
  that succeeded; it is a real lifecycle event and must be observable.
- The proof is a **bounded** set of `_run_herdr` calls, never an unbounded
  retry loop, and the stored endpoint is updated only after *every* field
  validates. A failed or ambiguous proof leaves the record non-owned.

`HERDR_ENV=1` is never used as a session name and never selects the launcher.

### Session policy

Three distinct states, distinguished with `herdr status server --json` **and**
`herdr status client --json`:

1. **Reachable server with an attached client** — the user's live session. Reuse
   it; new tabs are visible in their window.
2. **Reachable headless server** — reuse it, but do not claim visibility.
3. **No server** — auto-start one, then proceed.

Auto-start reuses **`claude_teams.filelock.file_lock`** — the single
cross-process advisory lock this project has (`src/claude_teams/filelock.py`),
already shared by the agent registry and the delivery store. A second, subtly
different lock is not introduced.

The lock path is **canonical per target Herdr endpoint**, not per win-agent-teams
session: two independent team sessions targeting the same Herdr server must
contend for the *same* lock, so it is derived from the configured session name
(or `default`) under the Herdr config directory, and exists before the server
does. Inside the critical section the manager re-checks server status, then
performs the start, and releases in `finally`.

Honest about the primitive: on POSIX `file_lock` blocks in `fcntl.flock` and its
`timeout_s` is effective only on Windows (`filelock.py:10-14, 43-59`). Lock
acquisition is therefore **not** described as bounded; the readiness poll after
it is bounded.

The server child is launched through a dedicated `_popen_herdr_server` seam:
detached, non-inherited stdio (`DEVNULL`), retained for reaping, checked for
early exit, and polled for socket readiness to a deadline. If it never becomes
ready, raise a named error reporting the socket path **when discoverable** and
otherwise the selected session plus the authoritative `herdr` command — a
brand-new session that failed during startup may have no `session list` row yet.

**No "server generation" guard.** Round 2 established that no Herdr CLI field
documents a server generation, start time, or boot id, so implementing against a
guessed JSON field would be speculation — and `herdr --handoff` can legitimately
replace the endpoint while pane processes survive, which would make a naive
generation check destroy live agents' records. It is also unnecessary: the
conjunction of session endpoint, pane identity, `shell_pid` equality and a
non-null Linux creation token (a `/proc` start time) cannot be satisfied by a
different process on a fresh server. Endpoint-replacement behavior is recorded in
the live matrix instead of guarded by invented state.

### Spawn

`HerdrProcessInfo`: `pid`, `creation_token` (**non-empty, enforced**),
`session_name`, `socket_endpoint`, `name`, `agent_id`, `team_name`, `backend`,
`tab_id`, `pane_id`, `workspace_id`, `log_path`, `started_at`.

`spawn_process`:
1. For `claude-code`, apply `_with_debug_file` as tmux does (`:1421`).
2. Resolve/ensure the server per the session policy.
3. `tab create --cwd <request.cwd> --label "<name>@<team>" --no-focus` plus one
   `--env K=V` per entry of `env`. Keys arrive **already validated** by
   `process_base._spawn_with_command`; the manager does not re-validate and
   never applies `_validate_safe_name` to env keys. Values may contain spaces,
   quotes and non-ASCII and are passed as single argv tokens.
4. Parse `root_pane.pane_id` / `.tab_id` / `.workspace_id`.
5. `pane run <pane_id> <_build_posix_shell_command(cwd, cmd, env)>` — env is set
   both by `--env` and by the shell export prefix (defense in depth).
6. `pane process-info` → `shell_pid`; handle is `str(shell_pid)`.
   `creation_token(handle)` is captured **at spawn**. It returns `str | None`
   (`process_manager.py:228-242`), and a `None` token is a **spawn failure**:
   close the tab and raise. Registering a null token would make the later
   equality check `None == None` — trivially true — and hand `ownership_probe`
   a false `OURS` with no PID-reuse proof.
7. Append a `[herdr] session=… tab=… pane=… pid=…` provenance line to the log.

If any step after the tab exists fails, close the tab and re-raise the original
exception with the cleanup outcome attached, so a failed spawn leaves no orphan.

### Ownership, liveness, kill, capture, send

A boolean cannot faithfully encode what a control-plane failure means, so the
manager has an **internal richer probe** and explicit per-caller projections:

```python
class _HerdrProbe(Enum):
    OWNED              # endpoint + pane + shell_pid + non-null token all match
    PANE_GONE          # pane absent, but the PID+token still match a live process
    PID_GONE           # the process itself is dead
    IDENTITY_MISMATCH  # pane live but shell_pid or token differs -> someone else's
    INDETERMINATE      # control plane unavailable (timeout / non-not-found error)
```

`OWNED` requires **all** of: the resolved endpoint matches the stored one (or
rebinds, as above); `pane process-info` succeeds and is well-formed;
`shell_pid == info.pid`; and `creation_token(str(info.pid))` is **non-null** and
equals the **non-null** stored token. Two null tokens are never equal here.

**Pane absence is not process death.** Herdr's own documentation states that a
pane moved to another workspace receives a new workspace-qualified pane ID while
its process keeps running, so a stored pane ID can vanish with the agent very
much alive. PID/token liveness is therefore evaluated *before* projecting: a
missing pane with a matching live PID+token is `PANE_GONE` (degraded-alive but
**unmanaged**), never `PID_GONE`. Only a dead PID or a token mismatch reports
process death. Collapsing the two would let `kill_agent` drop the record while
the real agent kept running — and would contradict the inherited
`ownership_probe`, which returns `OURS` whenever PID+token match.

Pane liveness alone is explicitly **not** ownership: it proves the terminal
object, not the PID identity that `ownership_probe` would hand to a killer.

Projections, because the right trade-off differs by caller:

- **`_tracked_alive`** → `True` only for `OWNED`. This is the mixin's in-memory
  shortcut (`ownership_probe:332-334` returns `OURS` without consulting the
  token), so it must be the strict one.
- **`health_check`** (read-only) → must not turn a control-plane hiccup, or a
  moved pane, into a false death. `OWNED` is alive. `INDETERMINATE` and
  `PANE_GONE` are **alive with a degraded detail** when the PID is live and its
  token still matches. Only `PID_GONE` or `IDENTITY_MISMATCH` reports dead.
  Untracked handles keep the inherited `_pid_health_with_token` tail.
- **Reclaim (`ownership_probe`)** — inherited unchanged. With a real non-null
  expected token its semantics are already right: matching live token → `OURS`,
  unreadable live token → `INDETERMINATE`, mismatch/dead → `NOT_OURS`. Note the
  inherited API does *not* map a Herdr error to `INDETERMINATE`; the manager
  therefore never claims it does.
- **Every mutating Herdr object operation** — `tab close`, `pane send-keys`,
  `pane send-text` — requires `OWNED`. A matching creation token authorizes
  signalling *that PID*, but never an operation on a Herdr object, which needs
  proven object identity. On `INDETERMINATE` or `PANE_GONE` the object command
  is **withheld** and only a token-revalidated PID operation is permitted; on
  `IDENTITY_MISMATCH` or `PID_GONE`, nothing is touched at all. This applies
  uniformly to the public `send` as much as to shutdown and kill — gating only
  the kill paths would leave the interface inconsistent and still risk writing
  input into a stranger's pane after a restart or handoff.

Concretely:

- `graceful_shutdown`: on `OWNED`, `pane send-keys <pane> ctrl+c` and poll to the
  deadline; on `INDETERMINATE`/`PANE_GONE`, no pane command — fall back to a
  token-checked `SIGINT` to the PID; on `IDENTITY_MISMATCH`/`PID_GONE`, do
  nothing and report.
- `kill_process`: on `OWNED`, `tab close <tab_id>`, then wait for PID exit, then
  force-kill **only** after re-validating the token. On `INDETERMINATE`/
  `PANE_GONE`, skip `tab close` and use the token-revalidated PID path — so a
  moved pane still stops the agent rather than orphaning it. A token that no
  longer matches means the PID was reused: leave it untouched.
- `send`: requires `OWNED`. `pane send-text`, then `pane send-keys <pane> enter`
  when `enter`. In every other state it sends nothing and records the reason;
  the signature stays `-> None`, matching the other managers, which also no-op
  on an unknown handle.
- `resolve_agent_pid`: returns `handle` (inherited default). Because the shell
  command ends in `exec` (`:89`), the shell PID *is* the agent; picking
  `foreground_processes[0]` would latch onto a transient hook/tool helper and
  make liveness flap. Not overridden.
- `capture` (read-only, but still identity-gated so it cannot silently read an
  unrelated object): `pane read` with `recent-unwrapped`, falling back to
  `visible` only on an **empty but successful** response; nonzero/malformed
  responses raise rather than silently returning `""`. `lines=None` is defined
  **mechanically** — omit `--lines` and return whatever snapshot Herdr 0.8.2
  supplies; `herdr pane read --help` does not promise that omission means full
  history, so no tmux-equivalence is claimed until the live matrix measures it.
  `lines <= 0` returns `""` without a call. Non-`OWNED` states return `""` with
  a recorded reason rather than reading.
- `log_path`: identical to tmux's.

### Subprocess seams

Two seams, because one cannot cover both shapes:

- **`_run_herdr(*args, timeout)`** — every finite control command. It enforces a
  timeout; checks the return code; rejects an `error` envelope on **either**
  stream; validates response `type` and required field presence/types; and
  raises `HerdrCommandError` carrying the error `code` and sanitized stderr.
  Per-call policy: a not-found error means *gone* for liveness only; every other
  failure is `INDETERMINATE` at the manager's own probe level — never silently
  "dead".
- **`_popen_herdr_server()`** — the long-running detached daemon. `subprocess.run`
  would block for the server's lifetime and yields no JSON to validate, so this
  seam returns a `Popen`-like object whose `poll`/`wait` drive early-exit and
  reaping tests.

Tests additionally inject the `creation_token` / `_pid_alive` boundary, the
clock/sleep used by readiness polling, and the lock, rather than monkeypatching
`subprocess` globally.

### Files affected

- `src/claude_teams/backends/process_manager.py` — constants, `HerdrCommandError`,
  `_HerdrProbe`, `HerdrProcessInfo`, `HerdrProcessManager`, pure
  `_select_linux_manager`. Imports `claude_teams.filelock.file_lock`; that
  module is **reused unchanged**, not modified.
- `tests/test_backends/test_process_manager_herdr.py` — new.
- `README.md` — Linux launcher table gains the `herdr` row.
- `docs/reference/agent-messaging-protocol.md` — note the launcher option.
- `docs/features/herdr-launcher/` — plan, reviews, implementation.

## Risks

1. **Server lifetime.** The Herdr server owns every pane process; losing it kills
   every agent. Not solved, and deliberately not papered over with an invented
   generation field: an endpoint change is reported through the probe as
   `PANE_GONE`/`PID_GONE`/`INDETERMINATE` per caller, or resolved by proven
   rebinding, rather than silently re-addressed. Recorded
   as an operational property; `herdr server stop` is never called. Herdr **object**
   destruction is scoped to proven-owned tabs, while degraded or unmanaged agents
   are stopped with token-revalidated PID signalling.
2. **Ownership.** Addressed by the `OWNED` conjunction with **non-null** tokens.
   Herdr's pane-ID-non-reuse guarantee is used only for pane *addressing* and
   never as PID-identity proof.
3. **Auto-start races.** Serialized by the existing `file_lock` on a canonical
   per-endpoint path, with a status re-check inside the critical section,
   early-exit detection and a bounded readiness poll. POSIX lock acquisition
   itself is blocking, and is not claimed otherwise.
4. **False deaths from a flaky control plane.** The strict `OWNED` check would,
   on its own, report a live agent dead on any CLI timeout. Mitigated by the
   `health_check` projection (degraded-live when PID+token still match), which is
   why the probe is richer than a bool.
5. **Capture completeness.** `visible` may be a viewport rather than full
   history, so capture is a diagnostic aid; `agent_output.py`'s file-based
   readers remain the primary transcript source.
6. **Pre-1.0 CLI drift.** Herdr states "the binary is the authority". All parsing
   is centralized and validating; fixtures record the exact 0.8.2 shapes.

## Test cases

Unit tests drive the two seams above plus injected token/PID, clock and lock boundaries (no live server).

Selection and construction: 1 `=herdr` selects; 2 unknown/blank/whitespace/case
values; 3 `HERDR_ENV=1` alone does **not** select; 4 explicit `tmux`/`terminal`
still win; 5 non-Linux unaffected; 6 import and `__init__` launch no subprocess
(**constructor purity**), while endpoint resolution happens on first spawn.

Session routing: 7 every command (status, server start, tab, pane, capture, send,
close) carries `--session` when pinned; 8 unpinned omits it; 9 a pinned session
can never emit argv addressing the default; 10 invalid session name rejected.

Spawn: 11 `tab create` argv exact (`--cwd`, `--no-focus`, one `--env` per var,
`name@team` label); 12 parses recorded JSON → `SpawnResult(str(shell_pid))`;
13 `pane run` carries the posix shell command string; 14 env values with spaces,
quotes, newlines, `=` and non-ASCII survive as single tokens; 15 keys are not
re-validated at this layer; 16 failure at each phase (malformed tab response,
`pane run` failure, `process-info` failure) closes the tab and preserves the
original exception; 17 cleanup-close failure is reported, not swallowed.

Ownership: 18 pane live but `shell_pid` changed → `IDENTITY_MISMATCH`;
19 token mismatch → `IDENTITY_MISMATCH`; 20 **stored token `None`** is rejected
at spawn (tab closed, spawn fails); 21 **live token `None`** never equals a
stored token → not `OWNED`; 22 malformed `process-info` → `INDETERMINATE`, and
no pane/tab command is issued; 23 socket endpoint differs → not `OWNED`;
24 `owns_process`/`ownership_probe`/`creation_token` end-to-end;
25 `provides_tty` is `True` before command construction.

Protocol: 26 error envelope on stdout and on stderr; 27 invalid JSON;
28 missing or wrong-typed fields; 29 non-positive `shell_pid`; 30 exit 1 vs
exit 2; 31 subprocess timeout.

Caller projections: 32 `INDETERMINATE` + live PID + matching token →
`health_check` reports **alive, degraded**, not dead; 33 **pane not found +
matching live PID/token → `PANE_GONE`, degraded-alive, record not removed**;
34 a moved pane (new workspace-qualified ID) behaves as 33; 35 `PID_GONE` and
`IDENTITY_MISMATCH` report dead; 36 `INDETERMINATE`/`PANE_GONE` withhold
`tab close`/`send-keys` and use the token-revalidated PID path, so a moved pane
is still stopped rather than orphaned; 37 `IDENTITY_MISMATCH`/`PID_GONE` touch
nothing; 38 **`send` and `capture` issue no command in any non-`OWNED` state**;
39 endpoint replacement **rebinds** when pane+`shell_pid`+token match, does
**not** rebind otherwise, and **never** considers a different session selector
even when its response claims the same tuple.

Lifecycle: 40 `kill_process` when close reports already-gone, when close fails,
when the PID exits, when it does not, and when the token changed before fallback
(no signal sent); 41 `graceful_shutdown` success, timeout, and CLI error;
42 `capture(None)` omits `--lines` vs `capture(N)` vs `lines<=0`,
empty-but-successful fallback to `visible`, and nonzero/malformed reads raising;
43 `send` of empty, multiline, quote-heavy, leading-dash and non-ASCII text, and
`enter=False`.

Auto-start (via the `_popen_herdr_server` seam and an injected lock/clock):
44 running server reused and no server spawned; 45 attached-client vs headless
distinction; 46 two managers targeting the same endpoint derive the **same**
lock path and only one starts a server; 47 status re-check inside the lock
finds a server started by the winner; 48 server exits early; 49 never becomes
ready → named error reporting the discovered socket path, or the session plus
command when no row exists yet.

Backend command construction (characterization, no live server): 50 for
claude-code, codex and pi, the exact `pane run` argv and `--env` list reaching
the seam preserve the prompt-sidecar instruction, correlation marker, and
model/effort and permission flags; 51 the same for a resume/follow-up spawn.
These prove *construction*; the shell round-trip itself is proven only by the
live matrix below.

### Live matrix (cannot be faked; recorded in `implementation.md`)

Run in a **disposable named session**, never the user's default: whether
`tab create` returns only once the shell accepts input; whether `shell_pid`
survives `exec`; pane behavior after the agent exits; ID behavior across a server
restart; **both inode-preserving and inode-replacing `herdr --handoff`, and
whether rebinding succeeds across each**; whether `pane move` leaves the process
untouched as documented; what `pane read` without `--lines` actually returns
(and how it differs from tmux's full history); PTY acceptance for all three
backends; and capture with and without an attached client. Smoke: spawn claude-code, codex **and pi**, confirm
`state-<agent>.json` and `inbox-<agent>.jsonl` appear, exchange a message, do a
`follow_up_agent` resume, then `kill_agent` and confirm the tab closes.

## Review disposition

### Round 1 (`plan-review-1.md`) — all accepted

| Finding | Disposition |
| --- | --- |
| BLOCKER 1 session routing | **Accepted** — mandatory `_herdr_argv`; session immutable, resolved to a socket endpoint; tests 7-10. Round 2: RESOLVED. |
| BLOCKER 2 PID ownership | **Accepted** — compound `OWNED` check + stored token; token-revalidated kill. Round 2: partially resolved → closed by round-2 BLOCKER 1 below. |
| MAJOR 1 `HERDR_ENV` auto-detect | **Accepted** — strictly opt-in, pure constructor/selector. Round 2: RESOLVED. |
| MAJOR 2 server/session states | **Accepted** — three states via `--json` status. Round 2: partial → closed by round-2 MAJOR 2/3. |
| MAJOR 3 `resolve_agent_pid` | **Accepted** — not overridden (`exec`). Round 2: RESOLVED and independently confirmed. |
| MAJOR 4 protocol/capture | **Accepted** — validating `_run_herdr`. Round 2: partial → closed by round-2 BLOCKER 1 projections. |
| MAJOR 5 test plan | **Accepted** — expanded. Round 2: partial → closed by round-2 MAJOR 4 (second seam). |
| MINOR 1 interface list | **Accepted** — full table. Round 2: RESOLVED. |
| MINOR 2 env validation layer | **Accepted** — verified `process_base.py:100`. Round 2: RESOLVED. |
| MINOR 3 wording | **Accepted** — "Observed", one-run scoping. Round 2: partial → closed by round-2 MINOR 1. |

### Round 2 (`plan-review-2.md`) — all accepted

| Finding | Disposition |
| --- | --- |
| BLOCKER 1 nullable tokens + mixin fallback | **Accepted.** Verified `creation_token` returns `None` for a dead *or* unreadable PID (`:228-242`), so `None == None` would forge ownership. Spawn now **fails** on a null token (tab closed); equality requires both sides non-null; added `_HerdrProbe` with explicit per-caller projections, and the plan no longer claims the inherited `ownership_probe` maps Herdr errors to `INDETERMINATE`. Tests 20-23, 32-35. |
| MAJOR 1 strict `_tracked_alive` cuts both ways | **Accepted** — `health_check` no longer aliases `_tracked_alive`: degraded-live on `INDETERMINATE` with matching PID+token; pane/tab ops withheld unless `OWNED`. Tests 32-35. |
| MAJOR 2 "server generation" undefined | **Accepted — removed entirely.** No CLI field documents it, and `herdr --handoff` can replace the endpoint while panes survive, so the guard could have destroyed live records. The pane+PID+non-null-token conjunction already covers PID safety. Handoff moved to the live matrix. |
| MAJOR 3 locking | **Accepted** — reuse `claude_teams.filelock.file_lock` (verified as the project's single advisory lock); canonical per-endpoint lock path that exists before the server; status re-check inside the lock; POSIX acquisition explicitly not claimed bounded. Tests 42-43. |
| MAJOR 4 seams | **Accepted** — added `_popen_herdr_server` alongside `_run_herdr`, plus injected token/PID, clock and lock boundaries; tests 46-47 reworded as construction characterization. |
| MINOR 1 session/socket speculation | **Accepted** — identity is the resolved socket endpoint from `session list --json`/`status server --json`; `HERDR_ENV` is treated as a boolean marker only; startup-failure errors degrade to session + command when no row exists. |

### Round 3 (`plan-review-3.md`) — 0 blockers; all accepted

| Finding | Disposition |
| --- | --- |
| MAJOR 1 `GONE` conflates pane absence with process death | **Accepted.** Herdr documents that `pane move` gives a pane a new ID while the process runs, so pane absence never proved death. Split into `PANE_GONE` (degraded-alive, unmanaged) and `PID_GONE`; PID/token liveness is evaluated before projecting; kill still stops a moved-pane agent via the PID path rather than orphaning it. Tests 33-37. |
| MAJOR 2 socket inode identity breaks handoff | **Accepted.** Inode equality was the generation guard returning by the back door: a supported `--handoff` would have made every record permanently unmanaged. Identity is now the canonical socket path/session selector, with **proven rebinding** (same pane + `shell_pid` + non-null token) on endpoint replacement. Test 39; both handoff shapes in the live matrix. |
| MAJOR 3 `send`/`capture` ungated | **Accepted.** The `OWNED` requirement now applies to *every* mutating object operation including the public `send`, and `capture` is identity-gated so it cannot read a stranger's pane. Test 38. |
| MINOR 1 constructor purity contradiction | **Accepted.** The contradiction was real: the constructor is pure and only captures/validates the session *name*; the socket endpoint is resolved and frozen lazily on first spawn. Test 6. |
| MINOR 2 `capture(lines=None)` overclaim | **Accepted.** Defined mechanically as "omit `--lines`"; no tmux equivalence claimed until the live matrix measures 0.8.2. Test 42. |

## Quality gates

`uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
`uv run pytest` — all four, whole repo, before the PR.

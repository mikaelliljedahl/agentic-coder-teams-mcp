# Implementation review — lead-wake conversation scoping

Reviewer: Claude Opus (independent post-implementation review; implementer was
Codex). Reviewed at `C:\code\github\win-agent-teams-mcp\wt-lead-wake-scoping`,
branch `feat/lead-wake-conversation-scoping`, uncommitted working tree
(`git diff` + untracked `src/claude_teams/procinfo.py`, `tests/test_procinfo.py`).

## Verdict

**APPROVED WITH CHANGES** — findings 1 and 2 are merge blockers.

The core fix is sound and it does fix the reported bug: a bystander Claude
conversation running the shared bound group resolves its own nearest Claude host,
mismatches the baked pid+token, and short-circuits at `D0b` before any session
resolution, registry read, inbox scan, guard write, or stdout. I traced the
ordering in code (`lead_wake.py:417-431`), not from comments: nothing but the
master kill switch precedes `_owner_decision`, and every non-owner branch returns
a `WakeDecision("allow", "D0b", …)` whose only side effect is the stderr log line
emitted by `main`. Both quality gates are genuinely green (see "Gates" below).

Two defects remain in the *host-identification* rule that spike F6 declared
blocking, and the Linux half of the release gate is not just "unverified" — the
chosen naming source (`/proc/<pid>/comm`) is likely to fail there for the same
reason as finding 1.

---

## Findings

### 1. BLOCKER — the `pi` entry in the host set is dead; a Pi mid-level lead can bind its parent Claude lead's ownership

`src/claude_teams/procinfo.py:12` — `_HOST_NAMES = frozenset({"claude", "codex", "pi"})`.

Pi is never launched as an image named `pi`. `backends/pi.py:235-246`
deliberately resolves `[node, <pkg>/dist/cli.js]` to bypass the `pi.cmd` shim, so
a Pi agent's process image is `node.exe` / `node`. `is_host("node")` is `False`,
so the two-step rule (nearest host over the full set, then require Claude)
degenerates to a **Claude-only search** for any Pi-nested agent — precisely the
failure spike F6 documents and the plan calls "blocking":

```
pi agent MCP server: python -> node.exe(pi) -> … -> claude.exe(LEAD) -> claude.exe(shell)
                                                     ^ selected as "nearest host"
```

Concrete damage. `CLAUDE.md` states that "lead is a role at every nesting level"
and that win-agent-teams supplies the hierarchy for **all** backends including
Pi, so a Pi mid-level lead calling `install_lead_wake` is a designed use case.
That install succeeds and bakes the *parent Claude lead's* `host.pid` and
`creation_token` together with `--reader <pi-agent-name>` and the Pi agent's
`--session-dir`, into the shared project settings — replacing the real lead's
group (`_install_wake_hook` keys only on the module token). The Claude lead's own
Stop hook then **matches** ownership at `D0b` (pid and token are its own), and
`_resolve_identity` falls back to `reader_arg` because `AGENT_NAME` is empty in
the lead's host — so the top-level lead is now nagged to drain
`inbox-<pi-agent>` and arm a watcher for the Pi agent's session. That is worse
than the bug being fixed.

Recommendation: identify hosts from argv, not the image name — on Linux read
`/proc/<pid>/cmdline` and match the basename of `argv[0]` (and, for node
launchers, the script path) in addition to `comm`; on Windows pair the Toolhelp
name with the process command line, or add `node` to `_HOST_NAMES` gated on a
`dist/cli.js`-style cmdline match. Add a `procinfo` test whose chain has a
`node`-named pi layer under a `claude.exe`, asserting the selected host is not
Claude. If you instead choose to accept the gap, it must be re-dispositioned
explicitly against spike F6, not left implicit in a name that never matches.

### 2. BLOCKER (release gate) — Linux naming source is likely to fail, and the R2-3 Linux capture is still open

`src/claude_teams/procinfo.py:75-99` reads the name from `/proc/<pid>/comm`.
`comm` reports the *thread name*, truncated to 15 bytes, and for an npm/node
installation of Claude Code (`claude` is a `#!/usr/bin/env node` shim) it reports
`node`, not `claude`. If that is what the VM shows, then on Linux
`install_lead_wake` refuses with `host_not_found` **always** and the feature is
entirely unusable, while the hook silently allows every conversation. This is not
a hypothetical: it is the same root cause as finding 1, and one fix covers both.

`implementation.md` honestly discloses the open gate, and repo policy
(`CLAUDE.md`: "Test on Linux too … before calling a branch green") makes it a
merge gate. Recommendation: run the paired capture on the Lubuntu VM before the
PR, record all three required shapes, and derive the supported basenames from the
capture as plan v3 requires ("not guessed"). Prefer the cmdline/argv0 fix from
finding 1 over adding `node` to a name set.

### 3. MAJOR — the tool docstring does not carry the new contract

`src/claude_teams/server_simple.py:6009-6037`. Per `CLAUDE.md`, the consuming
agent reads only the tool docstring. The docstring gained the binding-lifetime
and handoff paragraphs but still omits everything an orchestrator needs to react:

- **The new hard precondition.** `remove=False` now requires a concrete active
  session (`_active_session_id(create=False)` + `is_dir()`,
  `server_simple.py:6081-6089`). `_active_session_id(create=False)` never
  creates one (`server_simple.py:1254-1272`), so a fresh top-level lead that
  installs the wake hook **before** its first `spawn_agent` — a natural ordering,
  and the ordering the old `_SESSION_BASE` fallback tolerated — is now refused
  with `no_active_session` and no remediation text. Document it ("spawn an agent
  first, or call a tool that creates the session"), or pass `create=True`.
- **The refusal shape.** `{"success": false, "reason": …, "chain": […]}` with the
  four stable reasons is undocumented, and unlike this module's other error path
  it carries no `error` key, so a caller matching on `error`/`action` sees
  neither. The success dict has no `"success": true` counterpart — asymmetric.
- `WIN_AGENT_TEAMS_LEAD_WAKE_OWNER=0` is documented in
  `docs/reference/agent-messaging-protocol.md` but not in the docstring, unlike
  the `WIN_AGENT_TEAMS_LEAD_WAKE` master switch right next to it.

### 4. MAJOR — `evaluate(owner_mode="private")` defaults the gate open, and that is why the pre-existing decision tests never traverse it

`src/claude_teams/lead_wake.py:410`. The default makes the *safe* value the one a
forgetful caller gets, in a function whose whole purpose is a containment gate.
It also silently satisfies plan test 11 the wrong way: the plan asked that "every
existing decision test passes **with a matching owner injected**"; instead the
~40 existing tests keep calling `evaluate` with no owner arguments and take the
`private` fast path at `lead_wake.py:106`, so they never execute
`_current_owner_identity`. Recommendation: default `owner_mode=None` (legacy →
allow), then either inject a matching bound owner in the existing decision tests
or have them pass `owner_mode="private"` explicitly. `main` already passes
`args.owner_mode` (argparse default `None`), so the CLI path is unaffected.

### 5. MAJOR — the hook-side Claude-host requirement is untested; deleting it leaves the suite green

`src/claude_teams/lead_wake.py:88`. I mutated
`if host is None or not procinfo.is_claude_host(host.name):` to `if host is None:`
and ran `tests/test_lead_wake.py tests/test_procinfo.py
tests/test_install_lead_wake.py` → **80 passed**. (File restored; `git diff
--stat` unchanged at 108 insertions.) The install side has
`test_nearest_codex_host_refuses_instead_of_using_outer_claude`
(`tests/test_install_lead_wake.py:288`) but the hook side has no equivalent — no
`lead_wake` test ever passes a non-Claude nearest host; `_host()`
(`tests/test_lead_wake.py`) always builds a single-row `claude.exe` chain.
This is the same guard that finding 1 breaks, so it needs a real assertion:
nearest host `codex.exe`/`node` → `D0b`, `why="owner-unknown"`, session
resolution never called.

### 6. MINOR — `main` can still exit non-zero via a shutdown-time stdout flush

`src/claude_teams/lead_wake.py:533-537`. The block JSON is written inside
`suppress(BaseException)` but never flushed inside the boundary. A write that
lands in the buffer and then fails to flush at interpreter shutdown makes CPython
print "Exception ignored" and exit **120** — outside the `try`. The subprocess
tests only cover streams that raise on `write` (`sys.stdout = None`), not a pipe
that breaks after the write. Impact is low (Claude Code treats a non-2 non-zero
Stop exit as a non-blocking error, not a block), but the plan's contract is
"never exits non-zero". Fix: `sys.stdout.flush()` inside the same `suppress`, and
optionally `os._exit(0)` after flushing.

### 7. MINOR — no retry on transient `CreateToolhelp32Snapshot` failure

`src/claude_teams/procinfo.py:135-137`. Toolhelp snapshots legitimately fail
transiently (`ERROR_BAD_LENGTH`) while the process table churns; the standard
remedy is one retry. Today that raises `OSError` → the hook fails open (a lost
nag, acceptable) but `install_lead_wake` refuses with `host_walk_failed`, which
the user must diagnose for no reason. Retry once before raising.

The rest of the ctypes surface is correct: `PROCESSENTRY32W` field types and
order match `tlhelp32.h` (`ULONG_PTR` heap id as `c_size_t`, `LONG`
`pcPriClassBase`, `WCHAR[260]`), `dwSize` is set before `Process32FirstW`, the
`INVALID_HANDLE_VALUE` comparison against `ctypes.c_void_p(-1).value` matches the
unsigned int a `c_void_p` restype yields, the failure path raises *before*
acquiring the handle so there is no leak, and `CloseHandle` is in a `finally`.
Walk termination is correct: visited-set cycle guard, `pid <= 0` orphan stop,
`reader is None` (vanished process) stop, 64-level ceiling — and
`tests/test_procinfo.py` asserts all four including `len(chain) == 64`.

### 8. MINOR — install-side session selection is still auto-adoption, so `--session-dir` is not conversation-scoped

`server_simple.py:6081` → `_active_session_id(create=False)` →
`_recover_session_id()`, which auto-adopts a single cwd+identity candidate. A
fresh conversation in a folder can therefore bake a *previous* conversation's
session dir. Harmless in practice — the hook re-resolves the session at runtime
and only falls back to the baked value — but the result's
`{"scope": "conversation"}` claim is about *ownership*, not about which session
was selected. One sentence in the docstring/reference would close the gap.

### 9. MINOR — one vacuous assertion and thin chain coverage in the new hook tests

`tests/test_lead_wake.py`, `test_foreign_owner_short_circuits_before_all_session_work`:
`assert not (tmp_path / "wake-progress-team-lead.json").exists()` cannot fail —
`tmp_path` is never wired as the session dir in that test, and
`_resolve_session_dir` is monkeypatched to raise. The meaningful assertions
(resolution/`_scan_senders`/`_write_guard` raise on call) are present, so the
test is not worthless, just carrying dead weight. Separately, every hook-side
owner test uses a one-row chain, so no hook test exercises a multi-level walk.

### 10. NIT — two patch seams for the same function

`server_simple.py:6074` calls the module-level `creation_token` (imported at
`server_simple.py:54`) while `lead_wake.py:91` calls
`process_manager.creation_token`. Tests must therefore patch two different
attributes for the same behavior (`monkeypatch.setattr(ss, "creation_token", …)`
vs `monkeypatch.setattr(lead_wake.process_manager, "creation_token", …)`). Align
on one form.

### 11. NIT — `install_lead_wake` raises on a settings-write failure instead of returning a reason

`server_simple.py:5997-6006` / `tests/test_install_lead_wake.py:398`
(`pytest.raises(OSError)`). Every other failure in this tool returns a dict. A
`settings_write_failed` reason would keep the contract uniform. The atomicity
itself is correct — unique sibling temp, `Path.replace`, temp cleaned in
`finally`, prior bytes provably intact.

---

## Verified behaviors (checked in code, not taken from the diff summary)

- **D0b ordering.** `lead_wake.py:417-431`: kill switch → `_owner_decision` →
  `_resolve_identity` → `_resolve_session_dir`. No non-owner path reaches
  `server_simple._active_session_id`, `_load_agents`, `_scan_senders`,
  `_apply_guard`, or stdout. Enforced by tests that monkeypatch
  `_resolve_session_dir`, `server_simple._active_session_id`, `_scan_senders`
  and `_write_guard` to raise.
- **Owner-mode matrix.** `private` + no values → skip; `bound` + valid pid/token
  → compare; anything else (absent mode, `bound` missing a value, `private`
  *with* values, unknown mode) → allow `owner-unknown`
  (`lead_wake.py:106-112`, parametrized test covers all six rows).
- **Malformed baked values.** `_valid_owner_pid` excludes `bool` before `int()`
  and rejects `<= 0`; the token must be a non-empty `str`. `("x","t")`,
  `(-1,"t")`, `(True,"t")`, `(1,"")` all → `owner-unknown`, never raise.
- **PID reuse.** Token compared as well as pid; `creation_token` is
  `GetProcessTimes` creation FILETIME on Windows and `stat` field 22 on Linux
  (`process_manager.py:124-190`), both immutable per process. Test 10 covers
  pid-match/token-mismatch.
- **Fail-open entrypoint.** `_parse_args` is inside the `try`, the guard catches
  `BaseException` (so argparse's `SystemExit` too), and the stderr/stdout writes
  are each independently suppressed. Subprocess tests assert exit 0 and no block
  JSON for `--unknown`, a missing `--owner-mode` value, `evaluate` raising,
  `_log_line` raising, and broken stderr/stdout. Residual hole in finding 6 only.
- **Guard contract.** `owner_generation` is an optional trailing parameter;
  `member_wake` (`member_wake.py:211,226`) never passes it, so member guard files
  keep filename `wake-progress-member-<name>.json` and exactly the five original
  keys — `tests/test_member_wake.py:363` asserts the key *set* and
  `"owner_generation" not in guard`. Owner change (and an old, owner-less lead
  guard) resets `noprogress_blocks` rather than tipping into D6; both cases are
  tested against a seeded `noprogress_blocks=2`.
- **Install refusal contract.** Host walk → Claude requirement → token → active
  session, all before `_lead_wake_settings_path` is even called; the tests prove
  it by monkeypatching that function to raise. Existing settings bytes are
  compared with `read_bytes()` (including CRLF), and an absent `.claude`
  directory is asserted not to appear, for all four reasons.
- **`remove=True`** resolves neither host nor session (asserted by making both
  resolvers raise) and preserves the member group.
- **Handoff and multi-scope merge** behave as planned; the merged-scope test
  additionally proves the foreign bound group goes silent while the private group
  keeps D5.

## implementation.md claims — audit

| Claim | Result |
|---|---|
| `ruff check` → exit 0, "All checks passed!" | **True** (reproduced) |
| `pytest tests/ -q` → `1246 passed, 2 skipped` | **True** (reproduced: 1246 passed, 2 skipped, 99.06s) |
| D0b runs before session/inbox work; no guard write on non-owner paths | **True** |
| Refusals leave settings byte-identical, create no directory, 4 stable reasons | **True** |
| `_SESSION_BASE` is no longer a fallback session dir | **True** |
| Atomic temp + `Path.replace`; failed replace leaves prior bytes valid; temp cleaned | **True** |
| `remove=True` performs neither owner nor session resolution | **True** |
| Member guard filename, `lead-wake-progress/1` schema, and absence of `owner_generation` unchanged | **True** (asserted, not assumed) |
| Install returns `{"binding":{"scope":"conversation","survives_restart":false}}` + rerun note; reference doc updated | **True** |
| Entrypoint is `BaseException`-safe including independently failing stdout/stderr | **True for write failures**; incomplete for a shutdown-time flush failure (finding 6) |
| "It selects the nearest host from `claude`, `codex`, and `pi`" | **False in effect** — `pi` never matches a real Pi process, which is launched as `node` (finding 1) |
| "There is no intentional production-design deviation from approved plan v3" | **Mostly true**, with two unflagged gaps: the pi half of the plan's two-step host rule does not function (finding 1), and plan test 11's "matching owner injected" was satisfied by a permissive `evaluate` default rather than injection (finding 4) |
| Windows paired ancestry relies on `spike.md`; Linux gate open; manual two-conversation smoke not run | **True and correctly disclosed** — but per repo policy the Linux run gates the PR (finding 2) |

## Gates (run independently by this reviewer)

```powershell
$env:PYTHONPATH="C:\code\github\win-agent-teams-mcp\wt-lead-wake-scoping\src"
& "C:\code\github\win-agent-teams-mcp\agentic-coder-teams-mcp\.venv\Scripts\python.exe" -m ruff check
# exit 0 — All checks passed!

& "…\.venv\Scripts\python.exe" -m pytest tests/ -q
# 1246 passed, 2 skipped in 99.06s
```

Whole-repo, not scoped down. The two skips are the platform-conditional tests;
this Windows run is **not** Linux verification.

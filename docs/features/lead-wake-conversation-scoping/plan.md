# Lead-wake conversation scoping (install-time owner binding)

> **v3.** v1 proposed a disk *ownership claim* written by the MCP server during
> session resolution. Codex rejected it — see
> [plan-review-codex.md](plan-review-codex.md). Findings 1–3 are accepted in
> full and killed that approach (reasoning under "Why v1 was abandoned"). v2
> adopted the reviewer's own finding-9 option 3: bind the owner **at install
> time**, in the hook command itself. The round-2 review
> ([plan-review-codex-2.md](plan-review-codex-2.md)) returned **APPROVED WITH
> CHANGES**; all eleven findings are folded into this v3 and tracked in
> "Round-2 dispositions" at the end.

## Problem

The lead-wake `Stop` hook (`claude_teams.lead_wake`) nags Claude Code
conversations that never used the win-agent-teams MCP: at turn end they are
blocked with "arm an inbox watcher" / "read your inbox" instructions they
cannot act on meaningfully.

### Root cause

The D2 guard ("no live subagents → allow") exists, but is evaluated per
**workspace**, not per **conversation**:

1. `install_lead_wake` writes the wake hook into a *shared* settings file
   (project `.claude/settings.json` or `~/.claude/settings.json`), so every
   Claude conversation in that folder — or in that user profile — runs it.
2. A conversation that never used the MCP resolves identity `team-lead` and
   finds the session by cwd+identity, with single-candidate auto-adopt
   (`server_simple.py:1170`). Conversations in one folder are indistinguishable,
   so a bystander adopts the session another conversation populated.
3. `_live_subagent_names` then sees that other lead's live records
   (`_TERMINAL_STATUSES` is only `{"killed"}`), so the bystander lands in D3/D5
   and is blocked up to the no-progress cap.

Same shape sequentially: a finished conversation whose workers were never
killed leaves live-looking records that nag the next conversation in the folder.

**The blast radius is exactly the shared-settings install.** Server-spawned
agents get a *private* per-agent settings file
(`hooks.write_claude_settings` → `claude --settings <path>`), which is already
conversation-scoped. So the fix belongs at the shared-install boundary.

## Goal

The shared wake hook may block only the conversation that installed it. Any
other conversation running the same hook — same folder, same user scope — fails
silent (allow).

## Non-goals

- Changing when agent records become terminal. Owner scoping makes record
  staleness harmless for bystanders; retiring records is a separate feature.
- `member_wake` behavior. It shares helpers with `lead_wake`, so it constrains
  this change (see finding 10) but gains no owner row here.
- Codex/pi hosts. The lead-wake Stop hook is Claude-only wiring.

## Why v1 was abandoned (accepted findings 1–3)

- **F1, self-defeating.** `lead_wake._resolve_session_dir` calls
  `server_simple._active_session_id`. A claim written "whenever the server
  resolves a session" would therefore be written **by the hook process itself**,
  stamping the bystander's own host PID and making D7 match. The bug survives.
  "In the server process" is not an enforceable boundary when both entrypoints
  import the same function.
- **F2, resolution ≠ ownership.** `session_info`/`list_agents`/`check_agent`
  all resolve sessions, so any passive read from a second conversation would
  steal ownership and silently disarm the real lead.
- **F3, no funnel.** `_active_session_id` short-circuits on the module
  `_session_id`; spawned agents start pre-seeded; exact-binding recovery does
  not persist; only auto-adopt does; `resume_session` persists directly. And
  `install_lead_wake(remove=True)` resolves a session while it must claim
  nothing.

Install-time binding has none of these: there is exactly one write site, it is
an explicit user-invoked tool, and the hook never writes.

## Proposed design

### Bind the owner into the hook command at install time

`install_lead_wake` runs inside the **owner conversation's** MCP server
process. It resolves that conversation's host process and bakes it into the
argv it writes to the shared settings file:

```
python -m claude_teams.lead_wake --session-dir <dir> --reader <name>
       --owner-mode bound --owner-host-pid <pid> --owner-host-token <token>
```

`--owner-mode` is explicit and required-by-construction (R2-2). The per-agent
settings written by `hooks.write_claude_settings` emit `--owner-mode private`
and no pid/token: that file is passed via `--settings` to exactly one spawned
agent, so it is already conversation-scoped and needs no gate. Absence of
`--owner-mode` means a **legacy** shared install, which fails silent — a log
hint is not containment, and "keep today's behavior" would leave every
upgraded-but-not-reinstalled hook nagging bystanders forever. Any inconsistent
combination (bound without both values, private with values, unknown mode) is
rejected → allow.

- `owner_host_pid` — the **nearest host-process ancestor** of the server
  process: walk ancestors, take the first whose image basename (extension
  stripped, case-insensitive) is a Claude host (`claude`). Not `os.getppid()`:
  the repo `.venv` python re-execs into the uv interpreter, so our code runs a
  level below what the host launched (spike F3).
- `owner_host_token` — `process_manager.creation_token(pid)`, the repo's
  existing cross-platform process-identity token (`server_simple.py:1526` uses
  the same pid+token idiom for watcher owner-binding). This closes the PID-reuse
  hole (finding 4); PID equality alone is not identity.
- **Refusal contract (R2-4).** `procinfo` returns a diagnostic result — the
  bounded chain plus the selected host — not a bare tuple, so the refusal can
  report what it saw. `install_lead_wake(remove=False)` resolves the host,
  validates the pid, captures the token, **and** requires a concrete active
  session dir, all *before* it creates a directory or reads/writes the settings
  file. Any failure returns a stable reason —
  `host_not_found` / `host_token_unavailable` / `host_walk_failed` /
  `no_active_session` — plus the sanitized chain, leaving the settings file
  byte-identical and creating no directory.
- **No `_SESSION_BASE` fallback (R2-9).** Today `install_lead_wake` bakes
  `_SESSION_BASE` when no session is resolved. That is not a session dir, and it
  would let hook-side auto-adopt later pick an old cwd/identity candidate. For
  `remove=False` a real session dir is now required; `remove=True` stays
  independent of both host and session.
- **Install is an ownership handoff (R2-10).** Re-installing in the same scope
  removes A's command and arms B — last successful install wins, per settings
  file. This is now observable, so it is documented, and the settings write
  becomes an atomic temp+replace that preserves unrelated groups (today it is a
  direct `write_text`, which can lose settings or leave invalid JSON under
  concurrent installs).

Nothing else in the server changes. No claim file, no new write sites, no
change to session resolution.

### Hook-side check — new row D0b, before session resolution

The ownership comparison needs no session directory, so it runs **immediately
after D0 (kill switch) and before D1**, not after it (R2-1). This matters:
`_resolve_session_dir` calls `server_simple._active_session_id(create=False)`,
which scans bindings and registries and — on single-candidate auto-adopt —
calls `_persist_session_binding`, *writing and pruning binding files*. v2's
claim that a post-D1 row runs "before any registry read" was simply false; a
bystander would still mutate another conversation's binding state before being
allowed.

1. `--owner-mode private` → skip the gate (already conversation-scoped).
2. No `--owner-mode` (legacy shared install), or an inconsistent combination →
   allow, code `D0b`, `why="owner-unknown"`.
3. `--owner-mode bound` → resolve **this hook process's own** nearest Claude
   host and its creation token. Owner iff pid **and** token both match.
   Mismatch → allow, `why="not-owner"`.
4. Host unresolvable / token unavailable / walk error → allow,
   `why="owner-unknown"`.

In every non-owner branch: no session resolution, no registry read, no inbox
scan, no guard write, no stdout. The existing codes D0–D6 stay untouched and
`D0b` is inserted between D0 and D1, so log parsing and tests asserting
`code == "D3"` remain valid.

**Equality of the nearest host, never chain intersection.** Every desktop-app
conversation shares the Electron shell and `explorer.exe` as ancestors (spike
F2), so an intersection test matches every bystander and fixes nothing. For the
same reason the walk must stop at the *nearest* host: the outer Electron
`claude.exe` is a common ancestor of all conversations.

### Ancestor walk (`claude_teams/procinfo.py`, new)

Shared by `lead_wake` and `install_lead_wake`, hence its own import-light
module (`hooks.py` does not use it, so its import-light constraint is intact).

- Windows: one `CreateToolhelp32Snapshot` per walk → pid→(ppid, image name).
- Linux: `/proc/<pid>/status` (`PPid:`) plus `/proc/<pid>/comm` for the name —
  **not** naïve `/proc/<pid>/stat` field splitting, which breaks on process
  names containing spaces or parentheses (finding 5).
- Walk toward the root with a visited-set cycle guard and a generous ceiling of
  **64** levels, not 8 — an extra launcher (`env`, a shim, a container init)
  must not silently disable the real lead (finding 5).
- Returns a diagnostic result: the bounded chain plus the selected
  `(pid, name)`, or a "no host" outcome with the chain retained for the refusal
  message.
- **Two-step host selection, not a Claude-only search** (spike F6). Resolve the
  nearest host over the **full** host set (`claude`, `codex`, and the pi host),
  *then* require the selected host to be a Claude one. A Claude-only search
  walks straight past an intervening `codex.exe` and lands on the lead's
  conversation — measured live: a nested Codex agent's own MCP server resolves
  to `claude.exe(18352)`, its **lead's** host, and could bind or match the
  lead's ownership. With the two-step rule the nearest host is `codex.exe` →
  not Claude → refuse (install) / skip (hook).
- Names are matched exactly on the stripped basename, so near-misses
  (`claude-helper`, unknown wrappers) do not match.

### Fail-open boundary (finding 7 / R2-7)

The whole entrypoint, not just `evaluate`, must be incapable of a non-zero exit:

- `_parse_args` runs *inside* the boundary. `argparse` raises `SystemExit` on
  unknown/malformed argv, and `except Exception` does **not** catch it — so the
  guard catches `BaseException` (or `SystemExit` explicitly) around parse,
  evaluation, log rendering, log writing, and decision writing.
- A failing fallback log attempt must not raise again.
- Baked values are validated: `int`, non-`bool`, positive; token must be a
  non-empty string. (`isinstance(True, int)` is true in Python, so `bool` must
  be excluded explicitly.)

### Progress guard (finding 6 / R2-6)

Non-owners never reach `_apply_guard`, so they write no guard file. The owner
generation must **not** leak into the shared guard contract: `member_wake`
imports `_apply_guard`/`_scan_senders`/`_is_armed`/`WakeDecision`, and stamping
lead-owner fields into member guard files would silently change a module this
feature promises not to touch. So the counter machinery stays shared and
schema-identical, and the owner generation is passed as an **optional
parameter used only by lead wake**; on owner change lead wake resets
`noprogress_blocks` to 0 (otherwise a post-reinstall owner inherits a stale
count and fails open at D6 too early). The member guard filename and serialized
schema are unchanged, and that is asserted, not assumed.

### Restart of the owner conversation

A restarted lead is a new host process, so the baked pid/token no longer match
and the hook goes silent (`why="not-owner"`). The safety direction is right — a
new process must not inherit the old one's authority — but recovery is weak
(R2-8): a stderr line may never reach the model or user, and a tool docstring is
not shown proactively at exactly the moment nothing blocks any more. So:

- `install_lead_wake` **returns** an explicit binding-lifetime warning in its
  result (`"binding": {"scope": "conversation", "survives_restart": false}` plus
  a human-readable note), not only prose in the docstring.
- `docs/reference/agent-messaging-protocol.md` records it as a behavioral
  limitation of the wake hook.
- Stated plainly rather than papered over: **wake enforcement does not survive a
  conversation restart**; re-run `install_lead_wake` after restarting the lead.

### Settings-scope dependency (R2-5)

The design relies on documented Claude Code behavior: settings supplied via
`--settings` **merge** with file-based settings, and array-valued settings
concatenate across scopes
([settings precedence](https://code.claude.com/docs/en/settings#settings-precedence)).
Consequences to hold in mind, and to test rather than assume:

- A spawned agent can carry *both* its private `Stop` group and a shared
  project/user group. Intended outcome: the foreign bound group goes silent at
  D0b, the private group behaves exactly as today.
- More than two lead-wake groups can exist (user + project + local +
  `--settings`). Removing the group from one scope does **not** remove it from
  another, and two shared groups from different scopes must not be conflated.

### Kill switch

`WIN_AGENT_TEAMS_LEAD_WAKE_OWNER=0` disables only the D0b row (restores current
behavior) for diagnosis. The existing `WIN_AGENT_TEAMS_LEAD_WAKE=0` master
switch stays first.

## Alternatives considered

1. **Disk ownership claim written by the server** (v1) — rejected, findings
   1–3 above.
2. **Per-session settings file + relaunch command** (finding 9 option 1) — the
   cleanest containment, but it requires the human to relaunch `claude
   --settings <path>`, which defeats the point of a tool that wires an
   already-running interactive lead. Rejected on UX; revisit if install-time
   binding proves fragile.
3. **Refuse shared/user installs outright** (finding 9 option 2) — leaves the
   feature with no working path for the top-level lead.
4. **Transcript-tail ownership marker** (finding 9 option 4) — the Stop payload
   carries `transcript_path`, and the owner's transcript necessarily mentions
   the session id. No process semantics at all, but it depends on transcript
   format and compaction, and reads a file on every Stop. Kept as fallback.

## Risks

- **Restart silence** — see above. Accepted, documented, logged.
- **Unrecognised host** — a lead launched through a wrapper whose image name is
  not a Claude host: install refuses loudly instead of installing something
  unscoped. Better than silent over-reach, but it is a hard failure the user
  must report; the host set must be extended when the CLI binary name changes.
- **Both settings groups active for a spawned agent** — `--settings` merges
  with file-based settings and array settings concatenate across scopes, so a
  spawned agent in an installed folder runs both its private wake hook and the
  shared one. Today both nag it; after this change the shared one rejects it at
  D0b (foreign owner) and only its private group acts. Improvement, but it must
  be proven by the multi-scope test, not assumed.
- **Legacy shared installs go quiet** — the deliberate consequence of failing
  silent on a missing `--owner-mode`. Anyone upgrading must re-run
  `install_lead_wake` once. Preferred over leaving the original bug in place for
  every folder that is never reinstalled.
- **`ctypes` in a hook** — new low-level surface, confined to `procinfo` behind
  the fail-open boundary.

## Release gate — paired ancestry evidence (R2-3)

**Windows: discharged.** See [spike.md](spike.md) — conversation A's server and
hook chains independently select the same nearest Claude host (6956), other
conversations select distinct hosts, and the capture surfaced spike F6 (nested
Codex agent resolving to its lead's host), which changed the host-selection
rule. **Linux: still open** — repeat the same capture on the Lubuntu VM.

The requirement, for the remaining platform, and recorded in
`implementation.md`:

- the server-side chain and the hook-side chain for the **same** top-level
  Claude conversation → both must select the **same** nearest host pid;
- the same pair for a **second** simultaneous conversation → must select a
  **different** pid;
- the same pair for a **server-spawned** Claude agent.

The failure modes this rules out are concrete: if the server reaches only the
outer Electron `claude.exe` while the hook selects the inner conversation
process, the real owner goes silent immediately; if either side selects a shared
ancestor, bystanders match and nothing is fixed. The captured shapes become
fixtures (`server_chain_A` / `hook_chain_A` / `…_B`) asserting same-pid for A
and different-pid for B. The exact supported basenames are derived from these
captures, not guessed.

## Files affected

- `src/claude_teams/procinfo.py` — **new**: bounded ancestor walk
  (Toolhelp32 / `/proc`), nearest-host resolution with diagnostic chain,
  Claude host-name set.
- `src/claude_teams/lead_wake.py` — `--owner-mode` / `--owner-host-pid` /
  `--owner-host-token` args, D0b row before session resolution, owner-scoped
  guard reset, `BaseException`-safe entrypoint.
- `src/claude_teams/hooks.py` — `_wake_command` / `_wake_hook_matcher` render
  the owner args; `write_claude_settings` emits `--owner-mode private`.
- `src/claude_teams/server_simple.py` — `install_lead_wake` resolves the owner,
  requires an active session, refuses with stable reasons, writes the settings
  file atomically, and returns the binding-lifetime warning. **No change to
  session resolution.**
- `src/claude_teams/member_wake.py` — no behavior change; the shared guard
  contract is held constant and regression-tested.
- `tests/test_procinfo.py` (new), `tests/test_lead_wake.py`,
  `tests/test_install_lead_wake.py`, `tests/test_member_wake.py`,
  `tests/test_hooks.py`.
- `docs/reference/agent-messaging-protocol.md` — owner binding, D0b, and the
  restart limitation.

## Test cases (red first)

Walk helper (`tests/test_procinfo.py`, new):
1. Synthetic pid→(ppid, name) map: the **nearest** host wins over an outer host
   in the same chain (the Electron-shell case). Explicit — finding 8(a).
2. Chain `python → python → shell → host` resolves the host (venv/uv + shell).
3. Cycle (pid is its own parent) and orphan (`ppid=0`) terminate.
4. No host in chain → `None`; chain longer than the 64 ceiling → `None`.
5. Name matching is case-insensitive and extension-stripped
   (`Claude.EXE` == `claude`).
6. Real-OS smoke: on Windows the Toolhelp walk from the test process returns a
   plausible chain; on Linux the `/proc` walk does. Skipped per-platform, not
   replaced by the synthetic map (finding 8g).
7. Linux `/proc` name with spaces/parentheses parses correctly (via
   `comm`/`status`, not `stat`).

8. Host-name false positives: `claude-helper`, `codex`, and unknown wrappers
   must **not** match the Claude-only set (R2-11d).

Hook side (`tests/test_lead_wake.py`):
9. Foreign owner → allow, `D0b`, `why="not-owner"`, **and** session resolution
   is never called: monkeypatch `_resolve_session_dir` *and* `_active_session_id`
   to raise (R2-1). Also assert no stdout, no guard file, no registry read.
10. Baked pid matches but token differs (PID reuse) → allow, not-owner.
11. Matching owner → D1 **is** reached and D2/D3/D5 behave exactly as today;
    every existing decision test passes with a matching owner injected.
12. Owner-mode matrix (R2-2): `private` → gate skipped, today's behavior;
    absent mode (legacy shared) → allow `owner-unknown`; `bound` without both
    values, `private` *with* values, unknown mode → allow `owner-unknown`.
13. Malformed baked values: non-numeric, negative, `bool`-ish, empty token →
    allow, never raises.
14. Walk raises / host disappears mid-walk → allow, `why="owner-unknown"`.
15. `WIN_AGENT_TEAMS_LEAD_WAKE_OWNER=0` → gate skipped.
16. Guard reset on owner change: a guard carrying a different owner and
    `noprogress_blocks=2` resets to 0 rather than tipping into D6. Plus an
    explicit migration test for an **old-format** lead guard file.
17. Subprocess-level never-unstoppable (R2-7): unknown argument, missing
    argument value, `evaluate` raising, stderr write failing, stdout write
    failing → exit 0, no block JSON, in every case.

Install side (`tests/test_install_lead_wake.py`):
18. Install bakes mode+pid+token; idempotent re-install replaces the group and
    re-bakes. Argv round-trips through the shell quoting on Windows and POSIX
    (R2-11a).
19. Refusals leave the settings file **byte-identical** and create no directory,
    for each stable reason: `host_not_found`, `host_token_unavailable`,
    `host_walk_failed`, `no_active_session` (R2-4, R2-9). Covers both an
    existing settings file and an absent settings directory, host exit between
    walk and token read, and access-denied token reads.
20. `remove=True` never resolves a host, never requires a session, never binds
    (assert the resolver is not called).
21. Ownership handoff (R2-10): A installs, B re-installs → A's command gone,
    B's present, unrelated groups preserved; a failed replacement leaves the
    original bytes valid and intact.
22. Restart sequence (R2-8): A blocks; A restarts as A2 → the old command allows
    **without any disk read**; reinstall from A2 → A2 blocks and A stays silent.

Cross-cutting:
23. Two-conversation regression at unit level: owner A's baked values + a hook
    resolving host B → allow; same hook resolving host A → blocks.
24. Settings-scope merge (R2-5): user + project + `--settings` all loaded — all
    expected groups active, the foreign bound group silent, the explicit-private
    group keeping D2–D6, two shared groups from different scopes not conflated,
    and removal from one scope leaving the other intact.
25. `member_wake`: M0–M5 decisions, **counter values, D6 timing and serialized
    guard schema** unchanged across the change — seeded with a pre-existing
    member progress file, not only decision labels (R2-6).

Manual smoke (supplemental, not the proof): two `claude` conversations in one
folder; A installs and spawns a worker; B ends a turn → B not blocked; A still
gets D3/D5; restart A and confirm the mismatch is diagnosable in Claude Code.
Windows and the Lubuntu VM, reported as **separate** suites — a synthetic walk
suite is not Linux verification. Record in `implementation.md`.

## Validation commands

```bash
ruff check
python -m pytest tests/ -x -q
```

Full suite on Windows **and** the Lubuntu VM before the PR (repo policy), which
also discharges the "Linux unverified" half of finding 5.

## Round-2 dispositions

| # | Finding | Disposition |
|---|---------|-------------|
| 1 | D1b too late; discovery is impure and can mutate bindings | **Accepted** — moved to D0b, before session resolution; proven by a test that makes resolution raise |
| 2 | Missing owner args must fail silent; add explicit private mode | **Accepted** — `--owner-mode private/bound`, legacy absence → allow |
| 3 | Host resolution unproven for the real topology | **Accepted** — paired-capture release gate on Windows + Linux, captures become fixtures |
| 4 | Refusal contract / race-free unchanged-settings | **Accepted** — diagnostic resolver result, stable reasons, resolve-before-touch |
| 5 | Settings merge supports the design but test 22 didn't prove it | **Accepted** — dependency cited, multi-scope config test added |
| 6 | Owner generation must stay out of the shared member guard | **Accepted** — optional lead-only parameter, member schema/filename frozen and asserted |
| 7 | Entrypoint boundary underspecified (`SystemExit`, log/stdout) | **Accepted** — `BaseException`-safe entrypoint, subprocess-level tests |
| 8 | Restart silence not operationally recoverable | **Accepted** — warning in the tool result + reference doc + sequence test; limitation stated plainly |
| 9 | Active-session gate dropped; `_SESSION_BASE` fallback | **Accepted** — required for `remove=False`, fallback removed |
| 10 | Install is a handoff; non-atomic settings write | **Accepted** — documented semantics, atomic temp+replace |
| 11 | Test seams over duplicated unit comparisons | **Accepted** — test list restructured around the seams |

Nothing is dispositioned as "won't fix".

## Workflow

Feature branch `feat/lead-wake-conversation-scoping` in its own worktree;
this plan (v3) → red-green TDD → paired-ancestry capture (release gate) →
Claude Opus post-implementation review → PR to
`mikaelliljedahl/agentic-coder-teams-mcp`.

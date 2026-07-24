# External-member wake — implementation record

Branch: `feature/external-agent-join`. Implements
[`plan.md`](plan.md) v2 (with all §8 review-1 dispositions) on the basis of
[`design.md`](design.md). Strict red-green-refactor TDD; implementer: Claude
(Fable 5) subagent, 2026-07-24.

## 1. What was built

- **NEW `src/claude_teams/member_wake.py`** — Stop-hook module, CLI
  `python -m claude_teams.member_wake --joined-session-dir <dir> --member
  <name>`. `evaluate_member(payload, *, member, joined_session_dir)` implements
  the M0→M5 (+M2b) decision path. Reused **by import** from `lead_wake`:
  `WakeDecision`, `_scan_senders`, `_is_armed`, `_command_matches_session`
  (transitively via `_is_armed`), `_read_guard`/`_write_guard`/
  `_cursor_advanced` (transitively via `_apply_guard`), `_apply_guard`,
  `_read_payload`. New member-specific pieces (lead-hardcoded in `lead_wake`,
  per review-1 Major 3): `_member_kill_switch_on()`
  (`WIN_AGENT_TEAMS_MEMBER_WAKE`, on unless `"0"`, unset/blank → falls back to
  `WIN_AGENT_TEAMS_LEAD_WAKE`), a member `_log_line` prefixed
  `win-agent-teams/member-wake`, `_member_read_reason` (names
  `external_read(member_token=...)`), `_member_arm_reason` (renders
  `_watch_command_bash(joined_dir, reader=member)` + the `leave_team` escape
  hatch). Guard identity `member-<member>` → file
  `wake-progress-member-<member>.json` (verified against
  `lead_wake._guard_file`, lead_wake.py:232). Never-unstoppable `main()`
  identical to `lead_wake.main` (exit 0 always; block prints
  `{"decision":"block","reason":...}`; one stderr log line always).
- **`src/claude_teams/hooks.py`** — `_MEMBER_WAKE_MODULE =
  "claude_teams.member_wake"`, `_member_wake_command(joined_session_dir,
  member)`, `_member_wake_hook_matcher(...)` mirroring the lead-wake pair
  (`as_posix()` rendering, same `_WAKE_HOOK_TIMEOUT_SECONDS`).
- **`src/claude_teams/server_simple.py`** — `_watch_command_bash` extended to
  `(session_dir, timeout=None, *, reader=None)` and forwards `reader` to
  `_watch_argv` (existing callers unchanged); `_group_has_member_wake_token` +
  `_install_member_wake_hook` mirroring `_install_wake_hook` but keyed on
  `_MEMBER_WAKE_MODULE` (coexists with, never clobbers, the lead-wake group);
  `@_register_tool() async install_member_wake(joined_session_id, member_name,
  remove=False, scope="user")` with the full contract docstring (harness Stop
  hook requirement, no-credential guarantee, pull-only note, `leave_team` /
  `remove=True` off-ramp, kill switch). Validates via
  `_validate_join_session_id` (returns its refusal dicts verbatim) and rejects
  empty `member_name` / bad `scope` before touching disk. Returns
  `{action, path, member, joined_session_dir, scope}`.
- **NEW `tests/test_member_wake.py`** — full plan §5 matrix, cases 1–20
  (39 tests): kill-switch truth table, all fail-open gates (M1/M2/M2b),
  decision core (M3/M4/M5, external_read-not-read_messages, reader-scoped arm
  command with leave_team hatch, armed near-miss), progress guard (shared
  `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` cap, cursor-advance reset,
  `stop_hook_active` gating, `team-lead`-named-member guard-file isolation),
  main() entrypoint (member-wake log prefix, allow prints nothing, garbage
  stdin / missing `--member` never raise), install tool (user-scope default,
  project scope, idempotency, remove, validation errors), lead/member
  coexistence both directions, credential absence in settings + argv, hooks
  wiring shapes, reader-aware `_watch_command_bash`.
- **Docs** — `design.md` status flipped to implemented; README tool-table row +
  member-wake paragraph; `docs/reference/agent-messaging-protocol.md` external
  members section gained the member-wake paragraph.

## 2. Red evidence

Tests were authored first; `uv run pytest -q tests/test_member_wake.py`
before any production change:

```text
ImportError while importing test module '.../tests/test_member_wake.py'.
tests/test_member_wake.py:21: in <module>
    from claude_teams import hooks, member_wake
E   ImportError: cannot import name 'member_wake' from 'claude_teams'
    (/home/mikael/code/agentic-coder-teams-mcp/src/claude_teams/__init__.py)
ERROR tests/test_member_wake.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.49s
```

## 3. Green evidence

After the minimal implementation, `uv run pytest -q tests/test_member_wake.py`:

```text
.......................................                                  [100%]
39 passed in 1.66s
```

Refactor pass (ruff autofix + format on the touched files, `# noqa: PLR0911`
on `evaluate_member` matching the existing `_parse_member_token` precedent)
kept all 39 green; final focused rerun: `39 passed in 1.32s`.

## 4. Deviations from plan v2 (with justification)

1. **M2b activity scan includes `agents.json`** in addition to the plan's
   `state-*.json` + `inbox-*.jsonl`. A just-joined quiet team can have neither
   a state marker nor an inbox yet; scanning only the plan's pair would make
   M2b fail open (silently disarming the wake) on a perfectly live, freshly
   joined team. `agents.json` always exists for a valid session and is
   rewritten on membership changes, so it is the correct freshness floor. The
   plan's intent (fail open on *abandoned* teams) is preserved; test 13 aged
   all three patterns.
2. **Missing/blank `--member` fails open at M1** (allow) rather than being an
   argparse-required error. `argparse` `required=True` exits non-zero on a
   mis-baked hook, violating the never-unstoppable contract; folding "no
   member" into the M1 no-joined-session allow keeps exit 0 always. Covered by
   `test_main_missing_member_arg_never_raises`.
3. `_apply_guard` fail-open at the cap returns code `D6` (reused verbatim from
   `lead_wake` per the plan's no-duplication instruction), so a member-wake
   guard fail-open logs `D6` rather than an `M`-prefixed code. Tests assert on
   `action == "allow"` for that cell.

Everything else — including every §8 accepted change (M2b TTL + leave_team
hatch, `scope="user"` default + harness-requirement docstring,
`_member_kill_switch_on` / member log prefix / documented+tested shared cap
env, `wake-progress-member-` prefix, `running`/`left`/`killed` vocabulary,
tests 13–20) — is implemented as planned.

## 5. Whole-repo gate outputs (Linux, `uv run`)

`uv run ruff format --check`:

```text
75 files already formatted
```

`uv run ruff check`:

```text
All checks passed!
```

`uv run ty check`:

```text
All checks passed!
```

`uv run pytest -q` (whole repo):

```text
FAILED tests/test_agent_output.py::test_spawn_agent_persists_output_lookup_metadata
1 failed, 1162 passed, 3 skipped in 48.66s
```

The single failure is **pre-existing and unrelated**: with all member-wake
changes stashed (`git stash -u`), the same test fails identically on the clean
tree (`1 failed in 2.81s`). It is the locally-flaky
`test_spawn_agent_persists_output_lookup_metadata` (live-PID sensitivity)
already known on this machine. The tree is otherwise green; this change adds
no red.

## 6. Validation commands

```bash
uv run pytest -q tests/test_member_wake.py   # 39 passed
uv run ruff format --check                   # clean
uv run ruff check                            # clean
uv run ty check                              # clean
uv run pytest -q                             # 1 pre-existing failure (above)
```

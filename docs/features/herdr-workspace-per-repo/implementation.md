# Herdr: one workspace per repo — implementation

## What changed

`HerdrProcessManager` now routes each agent's tab into a workspace named after
**its repository** instead of dropping every agent into whatever workspace was
active.

`src/claude_teams/backends/process_manager.py`:

- `_WorkspaceLookup` (`MATCH` / `EMPTY` / `UNKNOWN`) — "no such workspace" and
  "could not read the listing" demand opposite actions, so they are separate
  states rather than one falsy answer.
- `_workspace_label` / `_repo_name` / `_git_common_dir` / `_sanitise_label` —
  derive the label from `git rev-parse --path-format=absolute --git-common-dir`
  (the **main** checkout, so worktrees share their repo's workspace), falling
  back to the folder name and sanitising the result. Never raises.
- `_resolve_workspace` — one bounded `herdr workspace list` per spawn; picks the
  lowest `(number, workspace_id)` among duplicate labels; anything unreadable —
  including a matching-label entry with an unusable id — is `UNKNOWN`, never
  absence.
- `_create_tab` — `EMPTY` creates the repo's workspace directly (falling through
  to an unqualified `tab create` is exactly the bug being fixed); `MATCH` passes
  `--workspace`; `UNKNOWN` keeps today's unqualified call *including* the
  `_HERDR_NO_WORKSPACE_CODES` catch, so a fresh headless server still works.
- `_create_in_new_workspace` / `_workspace_lock_path` — the create path
  re-lists under a per-session, per-label `file_lock`
  (`win-agent-teams-<session>.ws-<sha256(label)[:16]>.lock`), so parallel first
  spawns for one repo cannot each create a workspace. An inner `UNKNOWN` does
  not create.
- `_workspace_create` — labels the **workspace** with the repo (today's code
  mislabelled it `<agent>@<team>`) while the tab keeps the agent label.
- `_configured_workspace` + `WIN_AGENT_TEAMS_HERDR_WORKSPACE` — pin one label,
  or `-` to restore the pre-per-repo behaviour. An explicit override is
  validated strictly; a derived label is sanitised instead, because a repo
  cannot be renamed just to be spawnable.
- `_write_provenance` — logs `workspace=<id> requested_label=<label>`.

Rollback is **deliberately unchanged**. Review round 2 showed that closing a
workspace we created races with another spawn legitimately joining it; the live
probe then showed Herdr already closes a workspace with its last tab, so
"close the agent's tab" is correct in both orderings with no bookkeeping.

`README.md` documents the grouping and the new setting.

## Red → green evidence

Tests were written first against the not-yet-existing API and failed on it:

```
AttributeError: <...HerdrProcessManager object...> has no attribute '_workspace_label'
25 passed, 1 error
```

After implementation:

```
tests/test_backends/test_process_manager_herdr.py .......... 151 passed
```

Two of my own new cases were wrong rather than the code: a blank
`WIN_AGENT_TEAMS_HERDR_WORKSPACE` means *unset* (as for the session name), and a
NUL byte cannot exist in an environment variable at all. Both were corrected —
blank now has its own test asserting "unset", and the invalid-override set uses
real control characters.

## Live verification (Herdr 0.8.2, isolated `--session watest3`)

Three agents spawned into two throwaway repos plus a worktree of the first:

```
a1 /tmp/.../repo-a  label=repo-a ws=w3
a2 /tmp/.../wt-a    label=repo-a ws=w3     <- worktree joins its repo
b1 /tmp/.../repo-b  label=repo-b ws=w4
  w3 label=repo-a tabs=2
  w4 label=repo-b tabs=1
```

The CLI contract itself was probed the same way before design (see plan.md
"Verified Herdr behaviour"), including the load-bearing one: closing a
workspace's last tab closes the workspace. Every probe ran in a throwaway
session that was stopped and deleted afterwards; the user's live session was
never touched.

## Deviations from the plan

None in behaviour. Two mechanical differences:

- `_config_root()` was extracted from `_start_lock_path()` so both lock paths
  share the `HERDR_CONFIG_PATH` handling.
- The label predicate is `not label.isprintable()` rather than an explicit
  whitespace/control test — a plain space is printable and legitimate, and
  everything a terminal would choke on is not.

## Reviews

- `plan-review-1.md` — 2 BLOCKER, 6 SHOULD, 1 NIT. All dispositioned in plan.md.
- `plan-review-2.md` — 3 BLOCKER, 5 SHOULD. The rollback blocker changed the
  design (no `workspace close`); the rest tightened the trichotomy, the lock
  path and the parsing rules.
- `plan-review-3.md` — 2 SHOULD, both documentation-vs-design contradictions
  that would have misled implementation. Fixed before coding.
- `implementation-review-1.md` — 0 BLOCKER, 3 SHOULD, 2 NIT, all accepted and
  fixed: `.git` stripped with `removesuffix` (not `Path.stem`, which ate any
  suffix), the git probe decodes strictly so undecodable output falls back
  instead of becoming a label of U+FFFD, plus three test gaps.
- `implementation-review-2.md` — 2 SHOULD, both about tests proving less than
  they looked like they proved. Fixed: `_git_says` now asserts the probe's
  decode policy, and the concurrency test forces the interleaving through an
  instrumented `file_lock` (the loser signals the moment it reaches the lock)
  instead of a 0.5 s sleep.

### Mutation evidence for the concurrency test

The first version of that test passed even with the lock removed — exactly the
vacuity the reviewer predicted. It is now checked both ways:

```
with file_lock      -> 1 passed in 1.29s
with nullcontext    -> assert queued == [True]  ->  assert [False, False] == [True]
```

## Validation commands

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

All four green: `1564 passed, 4 skipped`.

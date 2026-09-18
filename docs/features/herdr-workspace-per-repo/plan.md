# Herdr: one workspace per repo

## Scope

Spawned agents currently all land in whatever Herdr workspace happens to be
active. Route each agent's tab into a workspace derived from **its repository**
instead, so a Herdr session with agents working on three repos shows three
workspaces rather than one wall of tabs.

Out of scope: Windows/tmux/terminal launchers, moving already-running tabs,
workspace cleanup when the last agent of a repo exits.

## Current behaviour

`HerdrProcessManager.spawn_process` (`src/claude_teams/backends/process_manager.py`)
calls `_create_tab`, which runs:

```
herdr tab create --cwd <request.cwd> --label <name>@<team> --no-focus --env ...
```

with **no `--workspace`**, so Herdr puts the tab in the active workspace. Only
when the server has no workspace at all (`workspace_not_found` /
`no_active_workspace`, `_HERDR_NO_WORKSPACE_CODES`) does it fall back to
`herdr workspace <same args>` and rename the workspace's root tab to the agent
label.

## Verified Herdr behaviour (0.8.2, protocol 20)

Probed live in an isolated `--session watest` server, then torn down:

- `herdr workspace list` emits JSON unconditionally (it rejects `--json`):
  `{"result":{"type":"workspace_list","workspaces":[{workspace_id,label,number,
  tab_count,...}]}}`. Entries carry **no cwd** — the label is the only join key.
- `herdr tab create --workspace <id> ...` works and answers `tab_created` with
  `root_pane.workspace_id` set to that workspace.
- An unknown id answers `{"error":{"code":"workspace_not_found"}}` — the same
  code as "no workspace at all", so the existing fallback already covers a
  workspace that was closed between `list` and `create`.
- `herdr workspace create --cwd X --label L --no-focus` labels the **workspace**
  `L` and its root tab `1` (the existing rename of the root tab stays correct).
- Duplicate labels are allowed: two `workspace create --label myrepo` produced
  `w1` and `w2`, both labelled `myrepo`. Selection must therefore be
  deterministic, not "the first one Herdr happens to list".
- **Closing a workspace's last tab closes the workspace.** Probed in a second
  isolated session: `workspace create` → `tab close w1:t1` → `workspace list`
  is `[]`. This is what makes rollback safe (see below).

## Design

### Workspace label

`_workspace_label(cwd) -> str`, decided with the user:

1. `git -C <cwd> rev-parse --path-format=absolute --git-common-dir`. The
   answer is the **main checkout's** git dir, so:
   - it ends in `/.git` → the label is the **parent** directory's basename
     (a worktree under `…/agentic-coder-teams-mcp/.claude/worktrees/<slug>`
     therefore joins the `agentic-coder-teams-mcp` workspace, not one per
     branch);
   - otherwise (a bare repo, `…/foo.git`) → the label is that directory's own
     basename with a single trailing `.git` stripped.
2. Anything that fails — not a repo, no `git` on PATH, nonzero exit, timeout,
   undecodable or empty/whitespace output — falls back to `Path(cwd).name`
   (the user explicitly accepted the folder name as the fallback). An empty
   basename (`/`) falls back to `agents`.
3. The result is **sanitised, never rejected** (review-2 finding 7): control
   characters are dropped, leading `-` stripped, the rest truncated to 128
   characters, and anything left empty becomes `agents`. A repo legitimately
   named `-weird` or holding a newline must not fail a spawn. Only the
   *explicit override* is validated strictly and may raise (below).

The git call is bounded (`_HERDR_CALL_TIMEOUT_SECONDS`) and never raises: a
broken git must not stop a spawn.

### Workspace resolution — three outcomes, not two

Review finding 3: "no match" and "could not ask" are different states with
different correct actions. `_resolve_workspace(label)` returns

| outcome | meaning | action |
| --- | --- | --- |
| `(MATCH, id)` | listing parsed, a workspace carries this label | `tab create --workspace id` |
| `(EMPTY, None)` | listing parsed, authoritatively no such label | `workspace create --label <repo>` **directly** |
| `(UNKNOWN, None)` | `workspace list` failed or was malformed | legacy path: unqualified `tab create` (today's behaviour), never a failed spawn |

Parsing rules, stated exhaustively (review-1 finding 7, review-2 finding 6):

- `workspaces` missing or not a list → `UNKNOWN`.
- Each entry must be an object; a **non-object entry, or an entry whose
  `label` matches the wanted label but whose `workspace_id` is missing or not
  a non-empty string, makes the whole result `UNKNOWN`** — the one candidate
  we cannot read is exactly the one that must not be mistaken for absence and
  duplicated.
- Among usable matches, order by `(number_key, workspace_id)` where
  `number_key` is `number` when it is an `int` and not a `bool`, else
  `sys.maxsize`. So an absent, boolean or string `number` sorts last instead
  of raising `TypeError`, and equal numbers are broken by id. Lowest wins.
- A well-formed list with no matching label → `EMPTY`.

No caching: one bounded `workspace list` per spawn is cheap next to launching a
CLI, and a cached id goes stale the moment the user closes a workspace.

### `_create_tab`

Review finding 1: the old pseudocode kept an unqualified `tab create` first, so
a repo with no workspace of its own would silently land in whatever workspace
happened to be active — exactly the bug being fixed. `EMPTY` must therefore go
straight to `workspace create`:

```
label       = _workspace_label(request.cwd)     # None when the "-" sentinel is set
outcome, ws = (UNKNOWN, None) if label is None else _resolve_workspace(label)

if outcome is EMPTY:                            # review-1 finding 1
    return _create_in_new_workspace(request, env, label)

try:                                            # MATCH -> --workspace ws
    return tab create [--workspace ws] …        # UNKNOWN -> today's unqualified call
except HerdrCommandError as exc:
    if exc.code not in _HERDR_NO_WORKSPACE_CODES: raise
# Either the matched workspace was closed underneath us, or (legacy path) the
# server has no workspace at all -- review-2 finding 3: this catch must survive
# on the UNKNOWN branch too, or a fresh server would fail the spawn.
return _create_in_new_workspace(request, env, label)


def _create_in_new_workspace(request, env, label):
    if label is None:                           # sentinel: no lock, no re-list
        return _workspace_create(request, env, label=agent label)   # as today
    with file_lock(_workspace_lock_path(label)):            # review-1 finding 2
        outcome, ws = _resolve_workspace(label)             # re-list under the lock
        if outcome is not EMPTY:                # MATCH or UNKNOWN
            try:    return tab create [--workspace ws] …
            except HerdrCommandError as exc:
                if exc.code not in _HERDR_NO_WORKSPACE_CODES: raise
                # fall through: proven to have no workspace
        created = workspace create --cwd <cwd> --label <repo label> --no-focus --env …
        rename created.tab -> "<name>@<team>"               # unchanged
        return created
```

Review-2 finding 2: an inner `UNKNOWN` does **not** create. It retries the
unqualified `tab create` (today's behaviour) and only creates when Herdr itself
says there is no workspace — so a transient list failure can never manufacture
a duplicate.

The lock is per session **and** per label, a sibling of the existing
`_start_lock_path()` under Herdr's config dir, so two MCP servers spawning into
the same repo at the same moment cannot both create a workspace. It is taken
only on the create path, never on the common reuse path. Review-2 finding 5:
the full path is `<config dir>/win-agent-teams-<session>.ws-<sha256(label)[:16]>.lock`
-- the digest keeps `/`, `..`, spaces and Unicode out of the filename, and the
session is encoded in the name exactly as `_start_lock_path()` does it
(review-3 finding 2), so two Herdr sessions with the same repo do not
serialise against each other. `file_lock` blocks indefinitely on
POSIX flock, which is the only platform this launcher runs on, so there is no
unlocked-create fallback to document.

`workspace create` is labelled with the **repo**, while the tab keeps the agent
label — today's code passes the agent label to `workspace create`, mislabelling
the workspace as `worker@team`. Fixing that is part of this change.

### Rollback stays "close the tab" (review-2 finding 1)

The earlier draft had rollback close a workspace this spawn created. Review-2
showed why that is unsafe: the creation lock is released when `_create_tab`
returns, so another spawn can legitimately add a tab to that workspace before
this one fails, and closing the workspace would take the other agent's tab with
it.

The probe above settles it without any ownership bookkeeping: **Herdr closes a
workspace when its last tab closes.** So rollback keeps doing exactly what it
does today — close the agent's own tab — and the outcome is automatically
correct in both cases: a workspace nobody else joined disappears with the tab,
and one another agent joined survives with that agent's tab intact. No
`workspace close`, no `created.workspace` extraction (review-2 finding 4), no
new race.

### Escape hatch

`WIN_AGENT_TEAMS_HERDR_WORKSPACE=<label>` pins every agent of this process to
one label, and `WIN_AGENT_TEAMS_HERDR_WORKSPACE=-` restores the old
"active workspace" behaviour (no `--workspace`, no `list`). Finding 6: labels
are free-form text in a single argv token, so the narrow `_HERDR_SESSION_RE`
would wrongly reject ordinary labels like `repo.name` or `my repo`. The guard
is label-appropriate instead: non-empty after strip, at most 128 characters, no
control characters, and no leading `-` (reserved for the sentinel and
unambiguous against any CLI parser). Review-2 finding 7: an invalid **override**
raises at construction, exactly like the session name, while a **derived** label
is sanitised into validity instead — a repo named `-weird` must spawn fine.

### Provenance (review-1 finding 9, review-2 finding 8)

`_write_provenance` gains `workspace=<id> requested_label=<label|->`. The id is
the authoritative `root_pane.workspace_id` actually returned; the label is what
*we asked for*, which on the `UNKNOWN`/sentinel paths is deliberately not the
active workspace's own label — naming the field `requested_label` keeps the
record honest.

## Files affected

- `src/claude_teams/backends/process_manager.py` — `_workspace_label`,
  `_resolve_workspace`, `_workspace_lock_path`, `_tab_create_args`,
  `_create_tab`, `_write_provenance`, new constants. Rollback is deliberately
  **unchanged** (review-3 finding 1).
- `tests/test_backends/test_process_manager_herdr.py` — the `_Herdr` scripted
  fixture gains a default `workspace_list` reply (finding 5), plus the cases
  below.
- `README.md:272-298` — the Herdr launcher section documents per-repo
  workspaces and `WIN_AGENT_TEAMS_HERDR_WORKSPACE`.
  (`docs/reference/agent-messaging-protocol.md` does not describe launcher
  behaviour and is left alone.)

## Risks

| Risk | Mitigation |
| --- | --- |
| Agents land in a human workspace that merely shares the folder name | The user chose reuse-by-label; the `-` escape hatch restores old behaviour |
| `workspace list` slows every spawn | One bounded call (15 s cap) per spawn, failure is non-fatal (`UNKNOWN` → legacy path) |
| Two parallel spawns both create a repo workspace | Per-session, per-label `file_lock` around re-list + create; lowest-`number` rule converges anything that still slips through |
| Duplicate labels flap between workspaces | Deterministic `(number, workspace_id)` ordering |
| A workspace closes between list and create | `_HERDR_NO_WORKSPACE_CODES` falls through to the guarded create path |
| A failed spawn leaves an orphan workspace | Rollback closes only the agent's tab; Herdr removes the workspace iff that was its last tab, so a workspace another agent joined survives |
| `git` invoked on an attacker-controlled cwd | Fixed argv, no shell, bounded timeout, all exceptions swallowed |

## Test cases (red first)

Label derivation (finding 8):

1. Worktree cwd → the main checkout's basename (common dir ends in `/.git`).
2. Bare repo common dir → own basename, one trailing `.git` stripped.
3. Fallback to the folder basename for: nonzero exit, `FileNotFoundError`
   (no git), `TimeoutExpired`, and empty/whitespace output.
4. `/` (empty basename, no git) → `agents`.
4b. Sanitisation, never rejection: a derived `-weird`, a name with a control
    character, and a 300-character name all yield a usable label and a
    successful spawn (review-2 finding 7).

Routing:

5. Matching label → `tab create --workspace <id>`, and the stored
   `HerdrProcessInfo.workspace_id` is that workspace (finding 5).
6. Duplicates → lowest `(number_key, workspace_id)` wins; plus mixed-validity
   listings (review-2 finding 6): a non-object entry, a matching-label entry
   with an empty/missing id, a boolean `number`, a missing `number`, and two
   equal numbers.
7. **Authoritative no-match while an unrelated workspace is active** → NO
   unqualified `tab create`; `workspace create --label <repo>` is issued and
   the root tab is renamed `worker@team` (finding 1).
8. `UNKNOWN` (each of: `HerdrCommandError` from list, `cli_usage`, timeout,
   malformed payload, `workspaces` not a list) → unqualified `tab create`,
   spawn succeeds (finding 3).
8b. `UNKNOWN` **and** a server with no workspace: the unqualified `tab create`
    raises `workspace_not_found` and the spawn still succeeds through the
    guarded create (review-2 finding 3) — the existing fresh-server test must
    keep passing.
8c. `EMPTY` outside the lock but `UNKNOWN` on the re-list inside it → no
    `workspace create` unless Herdr says there is no workspace (review-2
    finding 2).
9. A non-workspace error from a targeted `tab create` propagates unchanged.
10. A stale `--workspace` id answering `workspace_not_found` re-lists under the
    lock and falls back to `workspace create`, returning a usable tab.
11. Two concurrent no-match spawns (threads through a barrier, one shared lock
    file) create **one** workspace and both target it (review-1 finding 2).
12. Rollback closes the agent's tab, and only that, for both a reused and a
    freshly created workspace (review-2 finding 1) — no `workspace close` is
    ever issued.
12b. `_workspace_lock_path` maps `/`, `..`, spaces and Unicode inside the
     intended directory; equal labels share one path; and the same label under
     two different sessions does not (review-2 finding 5, review-3 finding 2).

Override:

13. `WIN_AGENT_TEAMS_HERDR_WORKSPACE=-` issues no `workspace list` and no
    `--workspace`.
14. `WIN_AGENT_TEAMS_HERDR_WORKSPACE=repo.name` pins the label regardless of
    cwd — a value the session-name regex would reject (finding 6).
15. An empty / control-character / leading-dash / over-long override is
    rejected at construction, like the session name.

Plus the full existing Herdr suite, `ruff format --check`, `ruff check`,
`ty check`, `pytest`.

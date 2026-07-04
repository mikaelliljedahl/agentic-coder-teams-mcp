# Development workflow

How we take a change from idea to merged PR in this repo. The loop is:

```
plan  ->  implement (TDD)  ->  Codex code review  ->  smoke test  ->  PR
   ^                                      |
   +--------------- iterate <-------------+
```

Every non-trivial change goes through all five stages. Small mechanical fixes may
collapse plan/review, but the **Codex review** and at least a **smoke check** are
never skipped for anything touching process spawning, hooks, or the MCP surface.

## 0. Ground rules

- **`main` stays on `origin/main`.** Never commit or leave uncommitted work on the
  main working tree. All work happens on a branch, and branches live in their own
  **git worktree** (`git worktree add ../wt-<slug> -b <branch> main`) so multiple
  agents can run in parallel without disturbing each other's tree.
- **`docs/` is gitignored.** Planning/ticket artifacts (plan, QA, test plan, review
  reports) live in the outer, non-git `docs/tickets/<ticket>-<slug>/` folder shared
  across worktrees — they are working notes, not shipped files. Only tracked docs
  (like this file) go at the repo root.
- **The consuming agent only reads tool descriptions**, never the README. Any disk
  contract or behavioral note an orchestrating agent must know goes in the MCP
  **tool docstring/description**, not just prose docs.
- **Test on Linux too.** The suite is developed on Windows but platform-dependent
  bugs (path separators, shell quoting) hide there. Run the suite on the Lubuntu VM
  before calling a branch green. Example we hit: single-quoted shell commands work
  under `sh` but fail under `cmd.exe`.

## 1. Plan

Follow the `/portal-ticket` strategy, minus Jira. Write to
`docs/tickets/<ticket>-<slug>/` (outer non-git folder):

- `plan.md` — scope, requirements (R1..Rn), design, files to touch, risks.
- `qa.md` — acceptance criteria per requirement.
- `testplan.md` — the concrete test rows (§1 unit, §2 integration/safety). These
  are **deliverables**, not suggestions — name each test so the implementer can't
  quietly defer them.

Set up the worktree for the branch at this point.

## 2. Implement (TDD red -> green)

- Use **Claude** for implementation, in the branch's worktree.
- Strict **red -> green**: write the test first, run it, confirm it fails for the
  right reason, then implement to green. Capture the red output.
- Keep Linux managers / cross-platform code paths in mind; don't hard-code Windows
  assumptions.
- Run the full suite locally (Windows) before handing off to review.

## 3. Codex code review (cross-review, separation of duties)

The opposite tool reviews the work: **Claude implements, Codex reviews** (and vice
versa). See the `codex-agent-orchestration` skill for the exact mechanics. The hard
rules:

- **Spawn Codex as a real process via the `win-agent-teams` MCP**
  (`spawn_agent`, `backend: "codex"`, `reasoning_effort: "high"`, default model
  `gpt-5.5`). **Never** run `codex exec` from the Bash tool or a harness subagent —
  the harness blocks the bypass flag and Codex silently no-ops while falsely
  reporting "completed".
- Give Codex an **absolute** `WRITE YOUR FULL REVIEW TO <docs>/<artifact>-<N>.md`
  instruction. Its working directory is unreliable.
- **Verify the file before reading it**: it must exist, be > 500 bytes, and start
  with a markdown heading. A "done" signal is not proof the file was written.
- Iterate with `follow_up_agent` (same logical agent) for re-reviews so it can
  reference its own prior findings. Address every finding, then re-review, until
  clean.

## 4. Smoke test

Verify the change end-to-end against a live/real environment, not just unit tests:

- Run the suite on the **Lubuntu VM** (Linux) as well as Windows.
- For orchestration/spawning changes, run a **spawn-chain smoke** through the live
  MCP (e.g. `claude -> Codex -> claude`, relaying a magic phrase back to `lead`) to
  confirm nested spawning and messaging still work across both backends.
- The VM runs Claude Desktop spawning agents in separate terminals (not tmux).

## 5. PR

- Push the branch (`origin` needs `gh auth switch --user mikaelliljedahl` first;
  the git repo is the `agentic-coder-teams-mcp/` subfolder).
- Open the PR with `gh pr create --repo mikaelliljedahl/agentic-coder-teams-mcp`.
- One branch = one coherent story. Don't bundle unrelated fixes into the same PR
  (e.g. keep a spawn-survival fix and an unrelated doc change on separate branches).
- Mark ready for review once smoke + review are green.

## Roles at a glance

| Stage      | Who       | Where                                   |
|------------|-----------|-----------------------------------------|
| Plan       | Claude    | `docs/tickets/<ticket>/` (outer, non-git) |
| Implement  | Claude    | `wt-<slug>/` worktree, TDD red->green   |
| Review     | **Codex** | spawned via `win-agent-teams` MCP, report to `docs/tickets/<ticket>/` |
| Smoke      | Claude    | Lubuntu VM + live MCP spawn-chain       |
| PR         | Claude    | `gh pr create --repo mikaelliljedahl/agentic-coder-teams-mcp` |

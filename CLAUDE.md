# Repository workflow

## Isolation and branches

- Do feature work on a dedicated feature branch in a dedicated Git worktree.
- Create the worktree from `main` (after confirming its relation to `origin/main`).
- Never implement in the shared primary worktree: other agents may be using it.
- Before editing, report the current branch, worktree, and working-tree status.
- Keep unrelated or pre-existing *feature* changes out of the feature branch. This is not a licence to ignore pre-existing breakage — see "Quality gates and pre-existing breakage".

## Required feature workflow

Use this sequence for every non-trivial feature or bug fix:

1. **Plan** — write `docs/features/<feature>/plan.md` before implementation. Include scope, current behavior, proposed design, files affected, risks, and test cases.
2. **Independent plan review** — ask an agent from the opposite model family to review the written plan. When the implementer is GPT/Codex, use Claude Code Opus; when the implementer is Claude, use a capable GPT/Codex model. Save the review under the same feature directory and resolve or explicitly disposition every finding before coding.
3. **Implement with red-green-refactor TDD** — the GPT/Codex implementation agent owns the code by default. Add focused failing tests first and run them to establish red; make the smallest production change that turns them green; then refactor without changing behavior. Run focused tests followed by the full suite.
4. **Independent post-implementation review** — ask Claude Code Opus to review the implementation against the approved plan, tests, and final diff. Save the review under the feature directory, address accepted findings, and rerun tests.
5. **Pull request** — run all four quality gates (see below), then commit the scoped changes, push the feature branch, and create a PR with `gh`. This fork has multiple remotes, so use `--repo mikaelliljedahl/agentic-coder-teams-mcp` explicitly.

Do not skip a review because the change appears small. If an external reviewer is unavailable, stop and report the blocker rather than silently self-approving.

## Quality gates and pre-existing breakage

**Run all four CI gates locally before opening a PR.** CI (`.github/workflows`) runs these on Linux; running only some of them locally and reporting "gates green" is how a PR lands red:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

`ty check` is easy to forget because `ruff` and `pytest` can be green while it is red — and when it fails, CI never reaches the test step, so a Linux test result you were counting on never gets produced. On Windows, `ty` also reports a small number of platform-only diagnostics that do not appear in CI; compare against the CI log before treating one as yours.

Run quality gates (lint, type-check, tests, coverage) across the **whole repository**, not only the files you changed. When a gate comes back red, "it was already broken" / "not my code" / "someone else's fault" is **never** an acceptable disposition on its own. Specifically:

- **Never report a gate as green when it is red.** If `ruff`, `pytest`, or any gate fails anywhere in the tree, say so plainly in your summary, name the failing files and rule codes, and state whether they pre-date your change. Do not scope the command down to hide a red result, and do not describe "green on my files" as if the repo is green.
- **Fix trivial, behaviour-preserving breakage on the spot.** Pre-existing lint that is purely cosmetic — import sorting, missing docstrings, line length, formatting — has no functional risk. Prefer fixing it over stepping around it. Group such fixes into their own commit so the diff stays reviewable, and note them in the PR.
- **Surface, don't swallow, anything non-trivial.** If a pre-existing failure would require a real behaviour change, a risky edit, or significant scope to fix, stop and report it to the user with a recommendation (fix here, separate PR, or accept) instead of silently leaving it or silently absorbing it.
- **Ownership is the repo's, not the author's.** Touching an area makes its quality gates your responsibility to at least report accurately. The goal is that `main` and every PR head pass the full gate; a change that leaves the tree no worse but still red must say so.

## Feature documentation

Keep feature-specific artifacts together:

```text
docs/features/<feature>/
  plan.md
  plan-review.md
  implementation.md
  implementation-review.md
```

`implementation.md` should summarize the red/green evidence, final design, deviations from the plan, and validation commands. Broader reorganization of existing documentation should be performed as a separate follow-up feature unless it is required by the current change.

# Repository orientation

Guidance for agents working in this repo (win-agent-teams: an MCP server for
spawning and messaging Claude Code, Codex, and Pi agents on Windows/Linux).

## Orientation

- **Package**: `src/claude_teams/` — MCP server (`server_simple.py`), backend
  adapters (`backends/`), state hooks (`hooks.py`), output-fallback readers
  (`agent_output.py`).
- **Dev workflow** (plan → implement TDD → Codex cross-review → smoke → PR):
  see [DEVELOPMENT.md](DEVELOPMENT.md). Work on a branch in its own git
  **worktree** (`git worktree add ../wt-<slug> -b <branch> main`); never leave
  work on the `main` tree. `docs/` (including `docs/features/` review records)
  is tracked and committed to the feature branch.
- **The consuming agent only reads MCP tool descriptions**, never this file or the
  README. Any disk contract or behavior an orchestrating agent must know goes in
  the **tool docstring**, not only in prose docs.
- **Test on Linux too**: run the suite on the Lubuntu VM before calling a branch
  green — Windows-only runs hide path/quoting bugs.

## Nested orchestration: lead is a role, not an identity

- **"Lead" is a role at every nesting level, not a fixed agent.** Any spawned
  agent can itself spawn children, so a subagent is the lead for the level
  below it. This is nested orchestration — the equivalent of Claude Code's
  built-in subagents, which **Pi lacks natively**, so win-agent-teams supplies
  the hierarchy for **all** backends (claude-code, codex, pi).
- **Identity & inboxes.** Each agent's own inbox is `inbox-<identity>.jsonl`,
  where identity is the `AGENT_NAME` env when set, else `team-lead` (the
  root/human-launched coordinator). Children `send_message` to their parent's
  inbox; a parent `read_messages` from its own. So the root lead watches
  `inbox-team-lead`; a mid-level lead watches `inbox-<its AGENT_NAME>`.
- **Consequence for tooling.** Anything acting on "the lead" (the
  `watch`/`inbox-status` CLI, the Pi wake extension) must use the agent's OWN
  identity — never assume `team-lead`. Identity flows via
  `AGENT_NAME`/`AGENT_SESSION_ID`/`AGENT_PARENT_NAME` (+
  `WIN_AGENT_TEAMS_SESSION_DIR` for spawned agents). Designed follow-up:
  `docs/features/pi-lead-autoload/design.md`.

## Adding or changing a backend

Full guide: **[ADDING-A-BACKEND.md](ADDING-A-BACKEND.md)**. The essentials to keep
in mind whenever you touch `backends/`:

- **Disk contract is the one hard dependency.** A spawned agent participates via
  files under `~/.claude/agent-sessions/<session-id>/`: the `state-<agent>.json`
  marker (`{state: running|waiting, event, ts}`) and `inbox-<agent>.jsonl`.
  Identity flows through `AGENT_NAME`/`AGENT_SESSION_ID`/`AGENT_PARENT_NAME`.
  Messaging is **pull**, never pushed.
- **Windows launch gotcha.** npm `.cmd` shims route through `cmd.exe`, which
  truncates argv at the first newline and mangles `< > | & ^`. Bypass the shim
  (launch the native exe / `node <entry>` directly) or transport the prompt in a
  file — never pass a multi-line prompt through the shim.
- **Model tiers**, not raw slugs: `low/medium/high/xhigh/ultra` bundling model +
  effort. Codex hard-fails on a missing model; Pi soft-falls-back to the CLI
  default. Discover installed models live and skip validation when discovery is
  empty.
- **A new backend is not done** until: registered in
  `registry._BUILTIN_BACKENDS`; server glue wires MCP identity + `_hook_extra`;
  a `read_<name>_output` reader + `_read_agent_output` branch exist; it writes
  the `state-<agent>.json` marker; and `tests/test_backends/test_<name>.py` +
  the README backend table are updated. See the checklist in the guide.

## Conventions

- Lint with `ruff check`; keep new/changed files clean (pre-existing debt in
  untouched files is out of scope).
- Match the surrounding code's docstring density and naming.

## Reference documentation

- `docs/reference/agent-messaging-protocol.md` documents how spawn, messaging, follow-up, the lifecycle hooks, the on-disk session contract, and the `watch` CLI actually behave. Read it instead of re-deriving the protocol from `server_simple.py`, `cli.py`, `hooks.py`, and the backends.

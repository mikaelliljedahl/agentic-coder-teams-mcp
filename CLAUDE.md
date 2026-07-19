# Repository workflow

## Isolation and branches

- Do feature work on a dedicated feature branch in a dedicated Git worktree.
- Create the worktree from `main` (after confirming its relation to `origin/main`).
- Never implement in the shared primary worktree: other agents may be using it.
- Before editing, report the current branch, worktree, and working-tree status.
- Keep unrelated or pre-existing changes out of the feature branch.

## Required feature workflow

Use this sequence for every non-trivial feature or bug fix:

1. **Plan** — write `docs/features/<feature>/plan.md` before implementation. Include scope, current behavior, proposed design, files affected, risks, and test cases.
2. **Independent plan review** — ask an agent from the opposite model family to review the written plan. When the implementer is GPT/Codex, use Claude Code Opus; when the implementer is Claude, use a capable GPT/Codex model. Save the review under the same feature directory and resolve or explicitly disposition every finding before coding.
3. **Implement with red-green-refactor TDD** — the GPT/Codex implementation agent owns the code by default. Add focused failing tests first and run them to establish red; make the smallest production change that turns them green; then refactor without changing behavior. Run focused tests followed by the full suite.
4. **Independent post-implementation review** — ask Claude Code Opus to review the implementation against the approved plan, tests, and final diff. Save the review under the feature directory, address accepted findings, and rerun tests.
5. **Pull request** — commit the scoped changes, push the feature branch, and create a PR with `gh`. This fork has multiple remotes, so use `--repo mikaelliljedahl/agentic-coder-teams-mcp` explicitly.

Do not skip a review because the change appears small. If an external reviewer is unavailable, stop and report the blocker rather than silently self-approving.

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

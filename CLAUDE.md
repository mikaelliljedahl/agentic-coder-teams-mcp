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
5. **Pull request** — commit the scoped changes, push the feature branch, and create a PR with `gh`. This fork has multiple remotes, so use `--repo mikaelliljedahl/agentic-coder-teams-mcp` explicitly.

Do not skip a review because the change appears small. If an external reviewer is unavailable, stop and report the blocker rather than silently self-approving.

## Quality gates and pre-existing breakage

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

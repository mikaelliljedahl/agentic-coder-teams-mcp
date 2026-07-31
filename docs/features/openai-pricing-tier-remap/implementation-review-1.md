# Implementation review 1: OpenAI pricing tier remap

## Findings

### MAJOR — The documented “default spawn” Luna dependency is not the live MCP default path

The revised plan says that because `medium` is the default tier, an install without Luna “fails on a default spawn” (`plan.md:74-76`). The live `spawn_agent` entrypoint instead defaults `model` to `""` and passes that blank value directly to the selected backend (`src/claude_teams/server_simple.py:2890-2895`, `src/claude_teams/server_simple.py:2961-2966`). Both changed backends deliberately treat blank as an escape hatch:

- Codex returns `("", effort)` and defers to its `config.toml` model without running Luna availability validation (`src/claude_teams/backends/codex.py:207-225`).
- Pi returns `("", effort)` and defers to Pi's own default model (`src/claude_teams/backends/pi.py:177-195`).

The existing unit tests explicitly preserve this behavior (`tests/test_backends/test_codex.py:126-130`, `tests/test_backends/test_pi.py:134-135`). Therefore the implementation has two different notions of “default”:

1. `default_model()` advertises the tier name `medium`, which now means Luna @ xhigh.
2. Omitting `model` from the MCP tool does not select `medium`; it bypasses the tier table and does not acquire the new Luna dependency.

This does not make the table edit incorrect, but it makes the plan's central availability-risk statement false and leaves consumers unable to tell whether “default” means advertised tier or omitted tool argument. Decide and document the intended contract. If blank must remain the escape hatch, state that only callers which explicitly use the advertised `default_model()` value depend on Luna, and add a tool-path test documenting omitted-model behavior. If an omitted model is meant to use `medium`, that requires an implementation change and a deliberate replacement for the current escape hatch.

### MAJOR — Pi's soft-fallback coverage does not exercise the new partial-catalog failure mode

The implementation correctly maps `medium` to Luna @ xhigh (`src/claude_teams/backends/pi.py:124-137`) and retains Pi's exact-membership fallback: any unavailable tier slug becomes `("", tier_effort)` (`src/claude_teams/backends/pi.py:183-206`). Consequently, a Pi login that exposes Sol but not Luna also resolves `medium` to `("", "xhigh")`; Pi does not retain the formerly working Sol model or substitute the Sol-backed `high` tier.

The plan says this fallback “only triggers on a login without the GPT-5.6 catalog” (`plan.md:82-87`), and both new fallback tests use Claude-only catalogs (`tests/test_backends/test_pi.py:137-141`, `tests/test_backends/test_pi.py:192-202`). That is narrower than the code and misses the most relevant migration case: a catalog containing `gpt-5.6-sol` but not `gpt-5.6-luna`. In that case the old `medium` tier worked as Sol @ low, while the new `medium` silently drops the explicit model and asks an unspecified Pi default for xhigh thinking.

Add tuple- and command-level coverage with `_models(["gpt-5.6-sol"])`, asserting the intended behavior. Update the risk/decision text to acknowledge partial GPT-5.6 catalogs and explicitly accept that the advertised default tier can lose its model selection even while Sol remains available. This is needed to fully disposition the prior fallback concern; removing `max` eliminates the highest-effort variant, but not the model-loss behavior.

### MINOR — The revised plan's verification commands are not executable in this worktree

The plan records `.venv/Scripts/python.exe ...` as the focused and full gates (`plan.md:117-123`), but that path does not exist in this worktree. `implementation.md:65-72` correctly explains that the virtual environment lives in the primary checkout and that `PYTHONPATH` must point at this worktree's `src`; otherwise the editable install can test the wrong source tree.

The prior “no executable gate” finding is therefore only partly dispositioned: the implementation report has the right recipe, while the authoritative revised plan still contains commands that fail here or, from another directory, risk importing the primary checkout. Replace the plan commands with the actual worktree-safe form used by the implementation.

### NIT — One external-compatibility sentence still carries the abandoned `max` tier case

The revised plan correctly states that no tier introduces `max` (`plan.md:80-89`), but the next paragraph still describes validating “Luna @ high/xhigh/max” (`plan.md:91-95`). Luna @ max is historical/raw-slug context rather than part of this implementation. Remove `max` from that tier-focused compatibility statement or explicitly label it as raw-slug-only.

## Prior-review disposition

- Sol availability coverage: fixed for Codex. Sol-missing/Luna-present errors on `high`, Luna-backed tiers work without Sol, Luna-missing/Sol-present errors on `medium`, and the Sol-backed tier still works (`tests/test_backends/test_codex.py:140-161`).
- Pi `("", "max")` fallback: obsolete as stated because `high` remains Sol @ medium and no tier uses `max`. The lower-effort fallback still has the partial-catalog gap above.
- External model/effort compatibility: reasonably documented as a launch-time precondition, with direct command-construction coverage for Luna @ xhigh on both backends (`tests/test_backends/test_codex.py:283-295`, `tests/test_backends/test_pi.py:179-190`).
- Low-tier rationale: fixed in `plan.md:37-48`.
- Executable verification gate: only partly fixed, as described above.

## Diff, stale-text, and validation assessment

The production diff matches the revised five-row table exactly:

- `low` is Luna @ high.
- `medium` is Luna @ xhigh.
- `high`, `xhigh`, and `ultra` remain Sol @ medium/high/xhigh.
- Terra remains raw-slug reachable.

No active README, source comment, or backend docstring still presents the old Terra/Sol tier table. `server_simple.py:2962-2965` remains correct because its example is `high -> Sol @ medium`. The old pairs in the plan's “Current behavior,” `implementation.md` before/after table, and `plan-review-1.md` are historical context, not stale operational guidance. The separate resume-session document's explicit `gpt-5.6-sol --thinking low` command is a raw launch example, not a tier mapping.

Independent validation performed for this review:

- Focused backend suite: `98 passed in 2.52s`.
- Full repository suite: `1176 passed, 2 skipped in 67.36s`.
- Ruff: `All checks passed!`.
- `git diff --check`: clean.

The implementation report's recorded green counts are reproducible. No source files were edited by this review.

CHANGES-REQUIRED

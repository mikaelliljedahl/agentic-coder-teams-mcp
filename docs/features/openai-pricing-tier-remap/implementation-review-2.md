# Implementation review 2: OpenAI pricing tier remap

## Findings

No BLOCKER, MAJOR, MINOR, or NIT findings.

## Round-1 disposition verification

### MAJOR #1 — default tier versus omitted model: resolved

The revised plan now states the live contract accurately:

- `default_model()` advertises `medium`, which maps to Luna @ xhigh and therefore depends on Luna when a caller explicitly passes that tier (`plan.md:84-91`).
- Omitting `model` from `spawn_agent` supplies `""`, bypasses the tier table, and defers to the backend's configured default without Luna validation (`plan.md:92-97`).

This matches the implementation. `spawn_agent` defaults `model` to `""` and passes it unchanged to `resolve_launch` (`src/claude_teams/server_simple.py:2890-2895`, `src/claude_teams/server_simple.py:2961-2966`). Codex and Pi both preserve blank as their documented escape hatch (`src/claude_teams/backends/codex.py:207-225`, `src/claude_teams/backends/pi.py:177-195`), with direct tests at `tests/test_backends/test_codex.py:126-130` and `tests/test_backends/test_pi.py:134-135`.

I accept the docs-only disposition. A new tool-path test is not required for this change: the server performs a direct, branch-free handoff of the argument, neither the server path nor blank behavior was modified, and the backend boundary that gives blank its semantics is already pinned. The revised documentation no longer claims that omission selects the advertised tier.

### MAJOR #2 — Pi partial catalog: resolved

`test_partial_catalog_drops_model_rather_than_substituting` now uses a Sol-only catalog and proves all relevant branches:

- Luna-backed `low` becomes `("", "high")`.
- Luna-backed `medium` becomes `("", "xhigh")`.
- Sol-backed `high` remains `("gpt-5.6-sol", "medium")`.

This is asserted at `tests/test_backends/test_pi.py:143-152`. The command-level fallback test also uses a Sol-only catalog and proves that resolved `medium` emits no `--model` while retaining `--thinking xhigh` (`tests/test_backends/test_pi.py:203-213`).

The plan now describes partial-catalog model loss explicitly, rejects silent Sol substitution as a deliberate product decision, and ties that decision to both tuple- and command-level coverage (`plan.md:99-110`, `plan.md:129-134`). This genuinely closes the earlier gap.

### MINOR — worktree-safe verification commands: resolved

The plan now records that the virtual environment belongs to the primary checkout, requires this worktree's `src` in `PYTHONPATH`, and provides focused/full/Ruff commands using that arrangement (`plan.md:140-150`). This addresses the editable-install risk identified in round 1 and agrees with `implementation.md:83-96`.

### NIT — stale `max` compatibility wording: resolved

The compatibility statement now names only Luna @ high/xhigh (`plan.md:112-116`). `max` remains mentioned only where historically or technically relevant, not as a tier introduced by this implementation.

## Diff and regression assessment

The implementation still matches the revised ladder exactly in both backends:

- `low` → Luna @ high.
- `medium` → Luna @ xhigh.
- `high` → Sol @ medium.
- `xhigh` → Sol @ high.
- `ultra` → Sol @ xhigh.

No production behavior changed in round 2. The only test delta is focused characterization of the already-selected Pi fallback semantics. The Codex two-direction availability matrix, raw Terra passthrough, tier-owned effort, command emission, and discovery-empty behavior remain covered.

## Independent validation

- Focused backend suite: `99 passed in 6.78s`.
- Full repository run: `1 failed, 1176 passed, 2 skipped`; the sole failure was the already-documented load-sensitive `tests/test_watch_command_discovery.py::test_watch_command_powershell_executes_and_times_out_quietly` subprocess timeout.
- The failing watch-command test emitted `1 passed in 19.46s` when run alone, confirming it is unrelated to the pricing-tier diff.
- Ruff: `All checks passed!`.
- `git diff --check`: clean.

The implementation report's updated focused/full counts are consistent with the added test. The transient full-gate failure matches the pre-existing flake already disclosed in `implementation.md:98-101`; it is not evidence of a regression in this change.

APPROVED

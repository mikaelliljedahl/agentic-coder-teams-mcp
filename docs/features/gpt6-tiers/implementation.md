# GPT-6 tier remap — implementation

## Final ladders

| Tier | codex | pi |
| --- | --- | --- |
| `cheapest` | gpt-6-luna @ medium | gpt-6-luna @ medium |
| `low` | gpt-6-luna @ high | gpt-6-luna @ high |
| `medium` | gpt-6-luna @ xhigh | gpt-6-luna @ xhigh |
| `medium-fast` | — | gpt-6-sol @ low |
| `high` | gpt-6-sol @ medium | gpt-6-luna @ max |
| `high-fast` | — | gpt-6-sol @ medium |
| `xhigh` | gpt-6-astra @ low | gpt-6-astra @ low |
| `max` | gpt-6-astra @ medium | gpt-6-astra @ medium |

## Red / green evidence

- **Red:** after updating tests only (`test_codex.py`, `test_pi.py`,
  `test_tool_descriptions.py`, `test_pi_fast_subtier_resume.py`):
  `uv run pytest -q <those 4 files>` → **32 failed, 131 passed**.
- **Green:** after the `_TIER_LAUNCH` tables, `_PI_UPGRADE_HINT` and the
  `spawn_agent` docstring were updated: **164 passed** for the same files.
- New tests: codex GPT-5.6-only catalog fails `cheapest/low/medium/high` naming
  the GPT-6 slug + upgrade hint; pi 0.87.0-shaped catalog fails all six
  GPT-6 Luna/Sol tiers naming the slug and `0.87.1`; raw `gpt-5.6-terra` still
  passes through on pi; the registered `spawn_agent` description names
  `gpt-6-luna`/`gpt-6-sol`/`gpt-6-astra` and `0.87.1` and contains no `gpt-5.6`
  and no `3-4x`.

## Design (as implemented)

- Straight effort-preserving substitution, unbenchmarked (plan D1).
- Terra (no GPT-6 successor) replaced by `gpt-6-sol @ low` at pi `medium-fast`.
- `_PI_UPGRADE_HINT` now says GPT-6 Sol/Luna need pi >= 0.87.1.
- The 5.6-era "3-4x faster" multiplier removed everywhere.
- Codex rationale: 272k default context window (was "262k").

## Deviations from the plan

- Open question 1 was answered by the user mid-implementation: the 1M Luna
  window is their local pi setting and this MCP does not need to care about
  context windows. Pi `high` stays `gpt-6-luna @ max`, and all pi
  context-window wording (and the planned `models.json` override snippet) was
  dropped from code comments, README, ADDING-A-BACKEND and the docstring
  (implementation-review F1).
- `tests/test_resume_session_dir.py` left unchanged (plan-review F6).

## Validation

```bash
uv run ruff format --check .   # 86 files already formatted
uv run ruff check .            # All checks passed
uv run ty check                # All checks passed
uv run pytest                  # 1688 passed, 4 skipped
```

Run on Linux (Omarchy). `pi 0.87.1` catalog verified with
`npx -y @earendil-works/pi-coding-agent@0.87.1 --list-models` (global install
untouched); `codex debug models` on codex-cli 0.155.1.

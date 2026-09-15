# Implementation: pi-only fast subtiers

## Final design

`PiBackend._TIER_LAUNCH` gains two entries, interleaved at their capability point:

```python
"medium-fast": ("gpt-5.6-terra", "high"),   # after "medium"
"high-fast":   ("gpt-5.6-sol", "medium"),   # after "high"
```

`supported_models()` therefore advertises, cheapest first:
`cheapest, low, medium, medium-fast, high, high-fast, xhigh, max`.

No logic changed. `resolve_model`, `resolve_launch`, `_model_available` and
`build_command` are entirely table-driven, so the subtiers inherit tier
semantics for free: the tier owns its thinking level (a caller
`reasoning_effort` is ignored), and a discovery catalog that lacks the slug
hard-errors with `BackendModelUnavailableError` instead of downgrading.

Codex is untouched — it keeps exactly six tiers. The cross-backend invariant
test was rewritten from "same keys, differ at high" to "codex's keys are a
strict subset, shared pairs differ only at `high`, the extras are exactly the
two subtiers, and the shared tiers keep codex's order".

## Red / green evidence

Red (tests written first, against unmodified `pi.py`):

```
8 failed, 51 passed   # tests/test_backends/test_pi.py
3 failed, 19 passed   # tests/test_tool_descriptions.py + tests/test_pi_fast_subtier_resume.py
```

Failures were exactly the new assertions: the 8-name ordered ladder, both
`resolve_model` mappings, both `resolve_launch` pairs, the rewritten
cross-backend invariant, the pi-only assertion, subtier-owns-thinking,
subtier-model-absent errors, the argv-level launch, the registered-description
contract, and the concrete-pair resolution.

Green after the `_TIER_LAUNCH` + docstring change: `1514 passed, 4 skipped`.

## Deviations from the plan

- **Context-window framing dropped.** The plan (and plan-review F1) required
  every contract surface to name Terra 272k / Sol 372k against Luna's 1M. The
  requester pointed out those limits are provider/config state on the machine
  running pi, not a property this MCP can know, so hard-coding them would be
  wrong on any other install. The contract now states only the speed/quality
  trade (~3-4x faster at about the same benchmark quality) and `pi.py` says
  explicitly that the adapter makes no claim about context limits. The rest of
  F1 — dropping the misleading unqualified "same capability" wording — stands.
- All other plan-review findings (F2–F5) implemented as dispositioned.

## Validation

```
uv run ruff format --check .   # 82 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1514 passed, 4 skipped
```

Plus a live smoke spawn on the `pi` backend at `medium-fast` (see below).

## Implementation-review dispositions (see `implementation-review.md`)

- **F1 (required) — accepted, adjusted.** "same job" / "latency vs depth only"
  overstated interchangeability. Every surface now says the subtiers benchmark
  *close to* their neighbour and are "a different model, not a drop-in on every
  input". The reviewer's suggested context-capacity caveat was NOT added: per
  the repo owner, context limits are local provider/config state this MCP has
  no business asserting, in either direction.
- **F2 (required) — accepted.** Two real contradictions fixed: the docstring's
  "the codex and pi ladders differ only at `high`" is now scoped to "across the
  six tiers both backends share", and the self-contradicting "this adapter makes
  no claim about context limits" sentence (added beside pre-existing 1M/262k
  prose in the same comment) was removed rather than the prose rewritten.
- **F3 (required) — accepted.** The registered-description test now locates the
  line for each tier and asserts model and effort *on that line*, so a swapped
  or misattached effort fails.
- **F4 (nice-to-have) — accepted.** `tests/test_pi_fast_subtier_resume.py` was
  rewritten onto the real server path: it calls `ss.spawn_agent(model="<tier>",
  backend="pi")` against the real `PiBackend` with only `spawn` stubbed, then
  asserts the concrete pair at the spawn request, the persisted `agents.json`
  record, the resume request, and the resume argv.

### Note on the reviewer's suite run

The review reported one failure in `tests/test_follow_up_delivery.py`
(`test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`
returning `agent_busy` instead of `resume_not_confirmed`), and it still
reproduced for the reviewer in round 2, standalone.

It does not reproduce here — green standalone (41 passed), green in the full
suite before and after these fixes, and green on `main` in this worktree's
environment. The test hard-codes a record PID of `123`
(`tests/test_follow_up_delivery.py:139-157`); PID 123 on this host is the
kernel thread `kcompactd0`, so whether the delivery path sees that PID as a
live agent depends on host process state and on how the runner's sandbox is
allowed to probe it. That is a **pre-existing latent host-sensitivity** in a
file this branch does not touch, not a regression from this change — but it is
real, it is reproducible in at least one environment on this machine, and it
should be fixed separately by making the test use a PID it owns.

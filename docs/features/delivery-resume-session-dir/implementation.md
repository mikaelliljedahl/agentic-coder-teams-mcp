# Implementation: restore the dropped `session_dir` in the delivery resume request

## Summary

`server_simple._build_resume_request` (delivery branch) omitted the
`"session_dir"` key from the resume request's `extra`, which the spawn path
sets. The pi backend then fell back to `request.cwd` for `--session-dir`, so a
resumed pi session was launched with `--continue` against a base that has no
rollout; pi exited immediately and guaranteed delivery settled
`failed(resume_not_confirmed)`. Fix: re-add the one key so resume mirrors spawn.

## Root cause

See `plan.md`. The key was dropped when request construction was extracted from
`follow_up_agent` into `_build_resume_request`. `_hook_extra` does not add it
back. On `main` the (un-extracted) resume path still sets it, so `main` is
unaffected; the defect is specific to `feature/message-delivery-protocol`.

## Change

`src/claude_teams/server_simple.py`, `_build_resume_request` `extra` dict — one
added key (mirroring the spawn path, byte-identical expression):

```python
"session_dir": str(_session_dir(session_id)),
```

`src/claude_teams/backends/pi.py` is unchanged — the fix is server-side at the
authoritative construction site (`server_simple` owns `_SESSION_BASE`), per the
Codex plan review.

## Red → green evidence

New test `tests/test_resume_session_dir.py` calls the real production
constructor `_build_resume_request` (not a hand-built `SpawnRequest`), with an
`agent_cwd` deliberately distinct from the session base.

- **RED** (unfixed delivery-branch source):
  - `test_resume_request_carries_authoritative_session_dir` — `KeyError`/missing
    `extra["session_dir"]`.
  - `test_resumed_pi_session_dir_is_the_session_base_not_cwd` — pi resolved
    `<cwd>/repo/pi-sessions/worker-pi` instead of
    `<session_dir>/pi-sessions/worker-pi`.
  - Observed: `2 failed`.
- **GREEN** (with the fix, `PYTHONPATH=$PWD/src`): `2 passed`.

The test asserts: the resume `extra["session_dir"]` equals
`str(_session_dir(SESSION))` and differs from cwd; the pi resume command has
exactly one `--session-dir` equal to
`<session_dir>/pi-sessions/<name>` and independent of cwd; and
`build_env` restores `WIN_AGENT_TEAMS_SESSION_DIR`.

## Deviations from the plan

- The plan's optional third test (an explicit spawn-vs-resume parity comparison,
  ideally via a spy through `spawn_agent`/`follow_up_agent`) was **not** added.
  The spawn `extra` is built inline inside the `_do_spawn` closure (not a
  callable helper), so a faithful parity test would require either a
  process-level integration test with a fake registry/backend or refactoring the
  spawn path — both out of scope for a one-key fix. The two sites use the
  byte-identical expression `str(_session_dir(session_id))`, and the resume test
  already pins the resume value to exactly that expression. The Codex
  post-implementation review rated the omission **minor / non-blocking**.
  Disposition: **accepted** with this rationale.

## Reviews

- `plan-review.md` — Codex (opposite family) plan review; CHANGES-REQUESTED.
  Its central findings (main already threads `session_dir`; the true defect is a
  server-side dropped key; a backend→server fallback/import is wrong layering)
  were adopted — the plan was rewritten and the fix moved server-side.
- `implementation-review.md` — Codex independent post-implementation review;
  **APPROVED** (one minor coverage note, dispositioned above). A Claude Code
  Opus reviewer was attempted first but stalled without producing a transcript
  binding; it was retired and the independent review re-run on Codex.

## Validation commands

Run from the worktree (no editable install here, so `PYTHONPATH=$PWD/src`):

- `PYTHONPATH=$PWD/src python -m pytest tests/test_resume_session_dir.py -q` → 2 passed
- `PYTHONPATH=$PWD/src python -m pytest -q` → 1073 passed, 3 skipped
- `ruff check .` → All checks passed
- `ty check` → All checks passed

## Landing

Per the maintainer, the verified fix is pushed onto
`feature/message-delivery-protocol` (which has an open PR against `main`),
updating that PR; no separate PR is opened. It is marked merge-ready after a
successful live message-delivery smoke test (pi resume / `follow_up_agent`).

# Implementation review: restore delivery resume `session_dir`

## Verdict

**APPROVED**

The implementation fixes the identified production defect at its source. The
resume request now carries the same authoritative session base as the spawn
request, so pi constructs the same per-agent `--session-dir` on resume and
restores `WIN_AGENT_TEAMS_SESSION_DIR`. No blocker or major finding was found.

## Findings

### Minor — the planned explicit spawn/resume parity test is absent

`tests/test_resume_session_dir.py` exercises
`server_simple._build_resume_request`, the actual production request
constructor, and verifies both the exact server-owned base and pi's downstream
command/environment behavior. It is therefore a faithful regression for the
dropped-key bug, not a synthetic hand-built `SpawnRequest`.

However, the approved plan also specified a third test that compares the
resume value with the value assigned by the spawn path for the same session.
The implemented tests compare the resume value with `_session_dir(session_id)`
but do not invoke or spy on the spawn construction. The production source
currently makes parity clear—both sites use the identical expression—so this is
a non-blocking coverage gap rather than a correctness issue. Adding an
integration/spy assertion through `spawn_agent` would more strongly protect
against the two construction sites drifting independently.

## Evaluation

### 1. Correctness

The root cause is fully resolved. `_build_resume_request` now adds:

```python
"session_dir": str(_session_dir(session_id)),
```

This is byte-for-byte the expression used by the spawn path. `PiBackend`
consumes this value in `_pi_session_dir`, appending
`pi-sessions/<request.name>`, and `build_env` uses the same extra value for
`WIN_AGENT_TEAMS_SESSION_DIR`. Because `agent_cwd` is deliberately distinct in
the regression test, the test proves the corrected path no longer falls back
to `request.cwd`.

A tree search found only two production `SpawnRequest` construction sites
(spawn and `_build_resume_request`) and only one production call to
`backend.resume`, in the delivery follow-up path. No other reachable
resume/delivery request constructor still omits `session_dir`.

### 2. Plan and earlier review

The implementation matches the revised approved plan:

- The fix is server-side at the production construction site.
- `src/claude_teams/backends/pi.py` is unchanged.
- There is no backend-to-server import or `team_name`-derived fallback.
- The authoritative, monkeypatchable `_SESSION_BASE` remains owned by
  `server_simple`.
- The earlier review's central concern about a hand-built incomplete request is
  resolved by calling `_build_resume_request` directly.

The only deviation is the omitted explicit spawn-parity test noted above.

### 3. Test quality

The test is a valid RED-to-GREEN reproduction: on the base implementation,
`request.extra["session_dir"]` is absent at the exact production constructor,
so the first assertion raises/fails; with the fix it passes. The pi test then
checks:

- exactly one `--session-dir`;
- the full exact expected path;
- independence from the stored agent cwd; and
- restoration of `WIN_AGENT_TEAMS_SESSION_DIR`.

The test hermetically redirects `_SESSION_BASE` and stubs pi's launcher and
real-home configuration side effect. The principal gap is that it does not
exercise the spawn constructor for the planned explicit parity comparison.

### 4. Regression risk

Risk is very low. Spawn code is untouched. Claude Code and Codex may receive
the restored extra key on resume but do not interpret it incompatibly. Pi's
existing missing-key fallback remains unchanged; only server-built resume
requests now honor the same explicit contract as server-built spawn requests.

### 5. Scope hygiene

The implementation content is minimal: one request-extra key plus the focused
regression test. There are no unrelated source/backend changes and
`git diff --check` reports no whitespace errors.

Review-state note: `fix/delivery-resume-session-dir` currently points to the
same commit as `feature/message-delivery-protocol`
(`77559bda2ef4a4b1a9a69ae8e82730d8b114a2c7`), while the source fix and test are
uncommitted worktree changes. Consequently, the literal requested
`git diff feature/message-delivery-protocol...HEAD -- src/ tests/` is empty.
This does not change the code verdict, but the reviewed changes must be
committed before the branch/PR contains the approved implementation.

## Verification

- Requested literal command could not run because this environment has no
  `python` executable and system `python3` lacks pytest.
- Equivalent repository-environment run:
  `uv run --frozen python -m pytest tests/test_resume_session_dir.py -q` —
  **2 passed**.
- `uv run --frozen ruff check .` — **passed**.
- `uv run --frozen ty check` — **passed**.

# Plan: restore the dropped `session_dir` in the delivery resume request

> **Revised after the Codex plan review** (`plan-review.md`). The original plan
> mislocated the fix in the `pi` backend and proposed a `team_name`-derived
> fallback with a backend→server import. Investigation confirmed the review's
> central point: `main` already threads `session_dir` on resume, the true defect
> is a **dropped dict key** in the delivery branch's server-side resume builder,
> and the clean fix is server-side (the server owns `_SESSION_BASE`). This plan
> supersedes the original.

## Scope

Branch: `fix/delivery-resume-session-dir`, based on
`feature/message-delivery-protocol` (the branch where the defect exists and
which carries the delivery protocol; `main` is unaffected). The verified fix is
pushed onto `feature/message-delivery-protocol`, updating its open PR against
`main`; no separate PR is opened. Change is isolated to one key in
`server_simple._build_resume_request`.

## Root cause

The delivery protocol refactored `follow_up_agent` and extracted request
construction into `server_simple._build_resume_request`
(`src/claude_teams/server_simple.py:2716`). Its `extra` dict is:

```python
extra = {
    "mcp_config_path": str(mcp_config_path),
    "agent_capability": "",
    **_correlation_extra(correlation_id),
    **prompt_extra,
    **_hook_extra(session_id, agent_name, backend_name),
}
```

The spawn path (`server_simple.py:2332`) builds the parallel dict **with**
`"session_dir": str(_session_dir(session_id))`. The resume builder dropped that
key during extraction. `_hook_extra` does not add it back (verified).

Consequences of the missing key on resume:

1. `PiBackend._pi_session_dir` does `base = extra.get("session_dir") or
   request.cwd`. With the key gone it falls back to `request.cwd`, so
   `--session-dir` points at `<cwd>/pi-sessions/<name>` — a different,
   non-existent base than spawn used. Launched with `--continue`, pi finds no
   session, exits immediately, and guaranteed delivery settles
   `failed(resume_not_confirmed)`. This is the live smoke failure on
   `feature/message-delivery-protocol` (worker-pi.log).
2. `PiBackend.build_env` only sets `WIN_AGENT_TEAMS_SESSION_DIR` when
   `session_dir` is present, so the pi state-marker target is also lost on
   resume — same dropped-key cause.

On `main` the resume path retains the key, so neither symptom reproduces there;
`main` needs no change.

## Why the fix is server-side (not in pi.py)

`server_simple` owns `_SESSION_BASE`, active-session resolution, and request
construction. Threading `session_dir` is the clean ownership boundary and
preserves any nonstandard/test session base exactly (a `team_name`-derived
backend fallback could not). The pi backend legitimately reads
`extra["session_dir"]`; the contract it depends on was simply not honored on the
resume path. Restoring the key makes spawn and resume symmetric again — which is
how every other backend already behaves (codex uses `request.cwd` on both paths;
claude-code uses `extra["mcp_config_path"]` on both).

## Change

`server_simple._build_resume_request`, add one key to `extra` (mirroring spawn):

```python
extra = {
    "mcp_config_path": str(mcp_config_path),
    "agent_capability": "",
    # Mirror the spawn path: pi resolves its per-agent session dir and the
    # state-marker env from this; dropping it makes resume fall back to cwd,
    # so --continue finds no session (resume_not_confirmed).
    "session_dir": str(_session_dir(session_id)),
    **_correlation_extra(correlation_id),
    **prompt_extra,
    **_hook_extra(session_id, agent_name, backend_name),
}
```

## Files affected

- `src/claude_teams/server_simple.py` — one added key in `_build_resume_request`.
- `tests/test_resume_session_dir.py` — new failing-then-green regression.
- `docs/features/delivery-resume-session-dir/` — this plan, the Codex review,
  implementation notes, post-implementation review.

## Test cases (TDD)

Faithful to the production construction site (the review's key ask — do not test
only a hand-built `SpawnRequest`):

1. **RED — resume request carries the authoritative `session_dir`.** Set up a
   tmp session base (monkeypatch `server_simple._SESSION_BASE`, create
   `<base>/<session>/mcp`). Call `server_simple._build_resume_request(...)` with
   an `agent_cwd` deliberately different from the session base. Assert the
   returned `SpawnRequest.extra["session_dir"] ==
   str(server_simple._session_dir(session_id))`. Fails today (key absent).
2. **Pipeline effect on pi.** Feed that request to `PiBackend`: assert
   `build_resume_command`'s single `--session-dir` equals
   `str(_session_dir(session_id) / "pi-sessions" / agent_name)`, is independent
   of `agent_cwd`, and that `build_env` sets `WIN_AGENT_TEAMS_SESSION_DIR` to the
   session dir. (pi launcher/headless/model discovery stubbed as in
   `tests/test_backends/test_pi.py`.)
3. **Parity with spawn.** Assert the resume `extra["session_dir"]` equals the
   value the spawn path assigns for the same `session_id`.

## Risks

- Very low. Adds a key that spawn already sets and that `main`'s resume path
  already sets; no other backend reads it differently. `_write_mcp_config`
  already requires the session dir to exist, so no new filesystem assumption.

## Validation commands

- `python -m pytest tests/test_resume_session_dir.py -q`
- `python -m pytest -q` (whole suite)
- `ruff check .`
- `ty check` (whole repo; per CI QA gate — Linux)

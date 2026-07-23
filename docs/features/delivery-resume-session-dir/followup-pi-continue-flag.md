# Follow-up: pi resume passed `--session-id` with `--continue`

## How it surfaced

During the live smoke test of the `session_dir` fix, `follow_up_agent` to a
waiting pi worker still returned `resume_not_confirmed`. The `--session-dir` was
now correct (`<session_dir>/pi-sessions/<name>`, matching spawn — the first fix
worked, and no stray `pi-sessions/` appeared under cwd), yet the resumed pi
process (pid) exited immediately.

## Root cause (second, independent defect)

`PiBackend.build_resume_command` emitted **both** `--session-id <name>` and
`--continue`. The installed pi CLI rejects that combination:

```
$ pi -p --mode json --session-dir <dir> --session-id smoke-pi --continue 'ping'
Error: --session-id cannot be combined with --continue
(exit 1)
```

So resume exited before attaching, and the delivery layer honestly reported
`resume_not_confirmed`. This is independent of the `session_dir` key and
pre-exists on `main` (same two flags in `build_resume_command`).

## Verification that `--continue` alone resumes

The per-agent `--session-dir` holds exactly one rollout (id `smoke-pi`), so
`--continue` is unambiguous:

```
$ pi -p --mode json --session-dir <dir> --continue --model openai-codex/gpt-5.6-sol --thinking low 'Reply with exactly: PONG-CONTINUE'
{"type":"session","id":"smoke-pi",...}
... assistant -> "PONG-CONTINUE" ...
{"type":"agent_settled"}
(exit 0)
```

## Fix

`src/claude_teams/backends/pi.py`, `build_resume_command`: drop the
`--session-id`/`request.name` tokens; keep `--session-dir` + `--continue`.

Test updated: `tests/test_backends/test_pi.py::TestPiBuildResume` now asserts
`--continue` is present and `--session-id` is absent (RED before the fix, since
the command still carried `--session-id`).

## Gates

- `PYTHONPATH=$PWD/src python -m pytest -q` → 1073 passed, 3 skipped
- `ruff check .` → passed; `ty check` → passed

## Both fixes are required

The `session_dir` fix (server) and this `--continue` fix (pi backend) are
independent; pi guaranteed delivery needs both. They land on
`feature/message-delivery-protocol` (PR #36) as two commits.

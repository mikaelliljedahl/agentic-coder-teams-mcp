# Herdr launcher — implementation

Implements `docs/features/herdr-launcher/plan.md` (revision 4, approved in
`plan-review-4.md`).

## What was built

`HerdrProcessManager` in `src/claude_teams/backends/process_manager.py`, a third
Linux launcher that gives each spawned agent its own Herdr tab. Selected only by
`WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`.

Supporting pieces: `HerdrCommandError`, `HerdrServerUnavailableError`,
`HerdrSpawnError`, `_HerdrProbe`, `HerdrProcessInfo`,
`_require_creation_token`, and the pure `_select_linux_manager` that replaced
the inline if/elif launcher chain.

No backend adapter, hook, protocol or disk-contract file was touched.

## Red/green evidence

Built in six slices, each red before green:

| Slice | Red | Green |
| --- | --- | --- |
| Selection + constructor purity + session routing | `AttributeError: module ... has no attribute 'HerdrProcessManager'` | 16 passed |
| `_run_herdr` protocol boundary | 9 failed, 16 passed | 25 passed |
| Spawn | 6 errors (`no attribute '_ensure_server'`) | 31 passed |
| Probe + projections | 19 failed, 31 passed | 50 passed |
| Kill + graceful shutdown | 7 failed, 50 passed | 57 passed |
| Bootstrap, rebinding, capture | 1 failed, 71 passed | 73 passed |

Two failures in the spawn slice were **test** bugs, not production bugs, and are
worth recording: the first asserted a command ending in `exec claude --print`
while `claude-code` correctly appends `--debug-file` (mirroring the tmux
manager), and the second forgot to record the cleanup call it was asserting on.
Both were fixed in the test.

One production behaviour was added without a prior failing test of its own: the
server bootstrap (`_ensure_server`) was written during the spawn slice because
spawn depends on it. Its seven tests were written immediately afterwards and
passed on first run. They are genuine tests, but they did not drive the design.

## Deviations from the plan

1. **`errors="strict"` on `_run_herdr`.** Not in the plan. The repository's own
   guard test (`tests/test_subprocess_decoding.py`) failed the first full-suite
   run, and it was right to: Herdr JSON is a machine protocol, so
   `errors="replace"` would smuggle U+FFFD into text we parse and act on.
   Undecodable output is now a `malformed` command error, with a test.
2. **`docs/reference/agent-messaging-protocol.md` was not modified.** The plan
   listed it, but that document describes the messaging protocol and has no
   launcher section; adding one would have been unrelated scope. The launcher is
   documented in `README.md` instead.
3. **Plan test numbers are not 1:1 with test names.** The suite covers the
   planned behaviours; some planned cases merged into one parametrized test
   (e.g. the launcher-value table), and the `send`/`capture` gating tests are
   parametrized across probe states rather than written out individually.

## Validation

All four gates, whole repository, on Linux:

```
uv run ruff format --check .   # 80 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1461 passed, 4 skipped
```

73 of those tests are new and specific to this launcher.

## Not yet verified (live matrix still outstanding)

Everything above is unit-level against faked seams. The plan's live matrix —
real spawns of claude-code, codex and pi into real Herdr tabs, `pane move`
behaviour, `shell_pid` survival through `exec`, both handoff shapes, and capture
with and without an attached client — has **not** been run yet. Until it is, the
launcher should be treated as unproven against a real Herdr server.

# Implementation — nested Linux launcher environment

## Outcome

Nested Claude, Codex, and pi MCP servers now receive the allowlisted Linux
launcher policy that selected terminal, tmux, or Herdr for their parent. A
second-level spawn can therefore select the same launcher instead of falling
back to a standalone terminal.

## Changes

- Added a POSIX-only launcher environment helper covering terminal, tmux, and
  Herdr selection/context values, including `TMUX`.
- Merged that mapping into generated Claude and pi per-agent MCP configs before
  authoritative identity values.
- Added the mapping to Codex's per-process MCP override for both spawn and
  resume.
- Added TOML-safe encoding for arbitrary launcher values: JSON-compatible
  escaping, explicit DEL escaping, literal Unicode scalars, and a named failure
  for unrepresentable lone surrogates.
- Kept the global pi MCP config launcher-agnostic and documented why.
- Updated the messaging protocol reference to describe the launcher carrier.

## Test evidence

The regression tests were written around the observed failure: generated
per-agent configs lacked `WIN_AGENT_TEAMS_LINUX_LAUNCHER`, so a nested server
selected the terminal backend. They now cover inheritance and omission for
Claude/pi, Codex identity preservation, tmux context, special TOML characters,
astral Unicode, lone-surrogate rejection, and execution with an inherited
Herdr environment.

Final automated gates:

- `uv run ruff format --check .`: pass (86 files)
- `uv run ruff check .`: pass
- `uv run ty check`: pass
- `uv run pytest`: 1683 passed, 4 skipped before the final review-only test
- `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr uv run pytest`: 1684 passed, 4 skipped

## Review response

The first independent Claude Opus code review identified broad test isolation,
unsafe TOML literal encoding, missing `TMUX`, inaccurate rationale, and missing
documentation. Each finding was addressed. Its follow-up plan review also found
that ASCII-only JSON escaping produces TOML-invalid surrogate pairs for astral
characters; the final encoder retains scalar Unicode literally and tests U+1F680.

The final review approved the implementation and listed five non-blocking
improvements. All five were closed: resume now asserts launcher propagation, the
Windows gate has a direct test, touched protocol citations and docstrings were
updated, and the tmux smoke result is recorded below.

## Process note

Diagnosis and an initial implementation preceded discovery of this repository's
required plan-first workflow. The plan and independent plan review were added
before finalizing the design, and all review findings were resolved before the
final code review and smoke test. This records the actual order rather than
claiming strict red/green sequencing that did not occur.

## Live smoke test

A parent Claude was started with the worktree's interpreter in dedicated Herdr
session `nestedfixsmoke`. Through its own nested MCP server it spawned:

- `nested-smoke-child` (`claude-code`) in Herdr tab `w1:t2`
- `nested-smoke-codex` (`codex`) in Herdr tab `w1:t4`

Both registry records have `spawned_by: nested-smoke-parent`, and both launch
logs begin with `session=nestedfixsmoke`, proving this was the second-level MCP
path rather than a direct root spawn or standalone terminal fallback.

An additional isolated tmux server (`-L watnestedfix`, session `smoke`) hosted
`nested-tmux-parent` and its MCP-spawned `nested-tmux-child` in panes `1.3` and
`1.2`. The child registry record has `spawned_by: nested-tmux-parent`, confirming
that inherited `TMUX` and target context kept the second-level spawn on the same
server.

## Follow-up: Windows CI fix

`tests-windows` failed six Linux-propagation tests (Codex `-c` env override and
the Claude/Pi MCP config writers). `nested_linux_launcher_env()` is deliberately
empty on Windows, so those tests were asserting POSIX-host behaviour on a
Windows runner. The shared TOML escaping and config-writer paths are
platform-independent, so rather than skipping them on Windows, the host check
moved behind a private `_launcher_host_is_windows()` seam and the tests pin it
via a `posix_launcher_host` fixture (`tests/conftest.py`). The Windows-disabled
test now patches the same seam instead of `os.name`, which would also flip
unrelated call-time checks such as `filelock`. Red: the fixture raised
`AttributeError` before the seam existed; green: 90/90 focused, full suite
1684 passed / 4 skipped, and all four gates clean on Linux.

# Nested Linux launcher environment

## Problem

An agent spawned through the Herdr launcher starts its own win-agent-teams MCP
server when it delegates again. The generated MCP configuration currently
carries only agent identity. A server started in a Herdr pane may therefore
miss the launcher's environment and select the default Linux terminal backend,
so second-level agents open in separate terminal windows
(`server_simple._write_mcp_config`, `src/claude_teams/server_simple.py`).

## Scope

- Carry the explicit Linux launcher policy into generated Claude and pi MCP
  configurations.
- Carry the same policy in Codex's per-process MCP environment override.
- Include Herdr, terminal, and tmux selection/context variables, including
  `TMUX`, but do not copy arbitrary process environment values. The inherited
  `TMUX` value deliberately keeps descendants on the parent's tmux server; a
  stale value is allowed to fail normally rather than silently selecting a
  detached session.
- Keep the shared global pi MCP configuration launcher-agnostic because it is
  used by unrelated sessions.
- Apply the carrier only on POSIX. Windows keeps the existing single-quoted
  Codex argv representation and does not use these Linux launchers.

## Design

1. Define one allowlist and helper beside Linux launcher selection in
   `src/claude_teams/backends/process_manager.py` for the variables required by
   a nested MCP server.
2. Merge that mapping before `AGENT_*` identity values in each per-agent MCP
   configuration in `src/claude_teams/server_simple.py`, preserving
   authoritative identity fields on both spawn and resume paths.
3. Encode arbitrary launcher values as TOML basic strings in Codex overrides;
   keep strict literal-string validation for identity values. Encoding uses
   JSON-compatible escapes, which TOML accepts, for quotes, backslashes and C0
   controls; escape U+007F explicitly and retain non-ASCII Unicode scalars as
   literals so astral characters are not converted to TOML-invalid surrogate
   pairs. Reject lone UTF-16 surrogates with a key-specific hard failure because
   TOML cannot represent them and they cannot be passed safely in process argv.
4. Add tests for inheritance, omission, identity preservation, tmux context,
   and TOML-special characters.

## Files affected

- `src/claude_teams/backends/process_manager.py`: allowlist and POSIX-only
  environment helper used by nested servers.
- `src/claude_teams/server_simple.py`: generated Claude configuration used by
  spawn and resume, generated pi configuration used through `_hook_extra`, and
  a rationale for leaving the shared global pi config unchanged.
- `src/claude_teams/backends/codex.py`: per-process MCP override shared by
  `build_command` and `build_resume_command`.
- `tests/test_backends/test_codex.py`: inheritance, identity, encoding, resume,
  and inherited-environment isolation.
- `tests/test_pi_worker_identity.py`: Claude/pi config inheritance and omission.
- `docs/features/herdr-nested-launcher-env/`: plan, reviews, and implementation
  record.
- `README.md` and MCP tool docstrings need no change: the public launcher
  variables and tool contract are unchanged. Update
  `docs/reference/agent-messaging-protocol.md` to record the additional
  per-agent environment carrier.

## Test cases (red first)

1. Generated Claude and pi configs inherit every present allowlisted value,
   including `TMUX`, while preserving `AGENT_*` identity.
2. They omit every allowlisted value when none is set; the test clears values
   locally so an inherited Herdr shell remains valid.
3. Codex's spawn and resume override includes launcher policy and identity.
4. Codex round-trips apostrophe, newline, U+0001, U+007F, BMP text, and an
   astral U+1F680 character through `tomllib`; a lone surrogate raises a
   diagnostic naming the key.
5. Existing exact-mapping tests explicitly clear launcher values instead of
   assuming the runner is outside Herdr.

## Verification

- Focused unit tests for Claude/pi generated configs and Codex overrides.
- Full formatting, lint, type-check, and pytest gates.
- Run the full suite both with and without
  `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` inherited.
- Live two-level Claude and Codex spawns where the child is verified in the
  same dedicated Herdr session instead of a standalone terminal.
- Exercise a two-level tmux spawn when tmux is available; otherwise record the
  environment limitation and retain unit coverage of its selection context.

## Risks

- Over-propagation could leak unrelated environment data. The fixed allowlist
  limits the carrier to launcher policy only.
- TOML quoting mistakes could prevent Codex startup. Parsing the generated
  override in tests covers controls, Unicode, apostrophes, and newlines.
- Tests run from inside Herdr may inherit launcher variables. Omission tests
  clear the allowlist locally instead of mutating global test setup.
- A configured Herdr workspace intentionally pins the descendant tree to its
  parent's workspace even if a child changes repository.
- Invalid inherited Herdr settings are validated when the nested server imports
  its process manager and can therefore fail that server early.

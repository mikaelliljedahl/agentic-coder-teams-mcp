# Implementation — Pi spawn: `ask_user` deadlock and prompt corruption

Implements [plan.md](plan.md) on branch `claude/pi-agent-spawn-issues-4cf9a5`
(worktree `.claude/worktrees/pi-agent-spawn-issues-4cf9a5`). The plan review was
explicitly waived by the user; steps 3–5 of CLAUDE.md otherwise apply.

## Red → green evidence

Tests were written first and run to failure before any production edit.

Red run (`pytest -k "wrapper_does_not or Lockdown or PromptTransport or
pi_multiline or pi_short"`): **14 failed, 3 passed**.

| Test | Red failure |
|---|---|
| `test_build_command_excludes_interactive_tools` | `ValueError: '--exclude-tools' is not in list` |
| `test_resume_command_excludes_interactive_tools` | same |
| `test_exclude_tools_env_override` | same |
| `test_build_command_appends_escalation_policy` | `'--append-system-prompt' is not in list` |
| `test_resume_command_appends_escalation_policy` | same |
| `test_escalation_policy_stays_short` | `AttributeError: _ESCALATION_POLICY` |
| `test_leading_character_is_guarded[@ / -]` (3) | prompt reached argv with its leading char intact |
| `test_oversize_prompt_uses_sidecar_on_headless_path` | `AttributeError: MAX_ARGV_PROMPT_CHARS` |
| `test_oversize_prompt_stays_in_argv_on_wrapper_path` | same |
| `test_shim_without_sidecar_warns_about_multiline_argv` | no WARNING record emitted |
| `test_wrapper_does_not_rewrite_newlines_inside_the_prompt` | prompt literal contained `\r\n` in the raw bytes |
| `test_pi_multiline_prompt_gets_a_sidecar` | `KeyError: 'prompt_file_path'` |

The three that passed red are the negative controls, correct before and after:
`test_empty_exclude_tools_env_omits_flag`, `test_ordinary_prompt_is_not_guarded`,
`test_pi_short_single_line_prompt_stays_in_argv`.

Green run of the same selection after the production edits: **154 passed** across
`test_pi.py`, `test_base_runtime.py`, `test_correlation_transport.py`.

**PYTHONPATH trap.** The shared `.venv` can resolve `claude_teams` to the primary
worktree. `uv run` here created a worktree-local `.venv`, and resolution was
verified explicitly before the red run:

```
uv run python -c "import claude_teams.backends.pi as m; print(m.__file__)"
→ ...\worktrees\pi-agent-spawn-issues-4cf9a5\src\claude_teams\backends\pi.py
```

All runs additionally set `PYTHONPATH=<worktree>/src`.

## Final design

### `src/claude_teams/backends/pi.py`

- `_autonomy_args()` (new, static) emits both autonomy layers and is spliced into
  `build_command` **and** `build_resume_command` — a resumed agent gets a fresh
  argv, so a policy that lapsed on resume would be no policy.
  - `--exclude-tools` with `_DEFAULT_EXCLUDED_TOOLS =
    "ask_user,ask_question,ask_human,request_input"`, overridable via
    `WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS`; an empty/whitespace value omits the flag.
  - `--append-system-prompt _ESCALATION_POLICY` — three lines: never wait for a
    human; call `send_message` (`win_agent_teams_send_message`) to the parent when
    blocked; then stop or continue under a stated assumption. A test pins the
    three-line budget so the per-turn cost cannot drift.
- `_guard_leading_char()` (new, static) prefixes one newline when the prompt
  starts with `@`, `/` or `-`. Applied inside `_correlated_prompt`, so every argv
  path gets it. Lossless: pi hands leading whitespace to the model unchanged.
- `_prompt_args()` now selects the `@file` sidecar when a sidecar exists **and**
  either the launch fell back to the `pi.cmd` shim or the prompt exceeds
  `MAX_ARGV_PROMPT_CHARS` (24 KB) on the headless path. The TUI path bakes argv
  into a `.ps1` and is exempt, since verbatim argv beats pi's `<file name="…">`
  wrapping. The one case with no safe transport left — shim + multi-line prompt +
  no sidecar — is now logged at WARNING instead of being silently truncated by
  `cmd.exe`.
- `MAX_ARGV_PROMPT_CHARS` is public so `server_simple` can share the threshold
  instead of duplicating it.

### `src/claude_teams/backends/process_manager.py`

`_write_tab_wrapper` writes `("\r\n".join(lines) + "\r\n").encode("utf-8-sig")`
with `write_bytes` instead of `write_text`. Text mode was translating **every**
LF to CRLF, including the ones inside the single-quoted prompt literal, so the
agent received a prompt the caller never sent. The explicit `\r\n` join already
supplies PowerShell's line endings, and the `utf-8-sig` BOM PowerShell 5.1 needs
is preserved. Shared by the claude-code and codex tab launches, hence the
dedicated wrapper-fidelity test asserting on **raw bytes** (reading the file back
as text hides the bug, because the reader translates CRLF away again).

### `src/claude_teams/server_simple.py`

- `_pi_needs_prompt_file()` (new): multi-line, or longer than
  `MAX_ARGV_PROMPT_CHARS`.
- `_materialize_prompt` writes a sidecar for `pi` when that predicate holds,
  making `PiBackend._prompt_args`' previously dead `@file` branch reachable. The
  prompt itself stays **un-marked** for pi — the backend appends the single
  correlation marker — so pi never gets two.
- `_write_prompt_file()` (new) factors out the write shared by the Claude
  transport and the pi fallback transport, so both spell encoding and directory
  creation once.

### Docs

`README.md` (pi setup section) and `docs/reference/agent-messaging-protocol.md`
(§ prompt materialization) document the deny list, the env override, the
escalation policy, the pi sidecar as a *fallback transport*, and the
leading-character guard.

## Deviations from the plan

1. **Plan test case 5, second half** ("without a sidecar it must not silently
   emit a multi-line argv token") is implemented as a **WARNING log**, not an
   exception. The plan's stated fix was the sidecar (`2b`, first option) and
   explicitly rejected the hard-error alternative as breaking shim-only installs;
   the log is the "not silently" half without that breakage.
2. **`_prompt_transport` was left unchanged.** It reports `"sidecar"` whenever
   `prompt_file_path` is present, so a multi-line pi spawn is now recorded as
   sidecar transport even when the backend actually used argv. The server cannot
   know which launcher the backend will resolve, and the sidecar label only grants
   the binding ladder's gate-0 grace period — the safe direction. Noted here
   rather than silently changed.
3. **`MAX_ARGV_PROMPT_CHARS` was made public** (the plan implied a private
   constant) so the threshold has one owner across `pi.py` and `server_simple.py`.
4. Plan item 2e (stdin transport) was not adopted, as the plan itself directed.

## Validation

Run from the worktree root with `PYTHONPATH=<worktree>/src`, whole repo, nothing
scoped down:

| Command | Result |
|---|---|
| `uv run ruff format --check .` | `78 files already formatted` |
| `uv run ruff check .` | `All checks passed!` |
| `uv run ty check` | `Found 1 diagnostic` — see below |
| `uv run pytest` | `1285 passed, 2 skipped in 106.80s` |

`ty check` reports one diagnostic, **pre-existing and not from this change**:

```
error[unresolved-attribute]: Object of type `BaseContext` has no attribute `Process`
 --> tests\test_join_team.py:730:9
```

`tests/test_join_team.py` is untouched by this branch, and the diagnostic is the
known Windows-only `multiprocessing` context typing difference that CI (Linux)
does not report. Compare against the CI log before treating it as new.

Still outstanding per the plan's Validation section, deliberately not done here:
the **Lubuntu VM** run of all four gates and the **live pi smoke test** (10 KB
multi-line prompt with `—`, `åäö`, a code fence and a `;`, asserting the pi
rollout's first user message matches byte-for-byte modulo the correlation
marker). The user runs the Linux smoke test before this branch is committed.

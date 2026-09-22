# Implementation review 2 — final independent code review

Reviewer: Claude Code (Opus 5), independent of the implementer. Date: 2026-09-22.
Subject: the complete uncommitted diff on `fix/herdr-nested-env` (6 modified files,
165 insertions / 18 deletions) plus the untracked `docs/features/herdr-nested-launcher-env/`.
No source, test, plan or implementation document was modified during this review.

## Verdict

**APPROVE.** Every finding from implementation review 1 and from the two plan reviews is
resolved, the corrected TOML encoder is sound against 21 adversarial input classes, all
four gates are green in both environment states, and the recorded live smoke test is
independently corroborated on disk. The five findings below are all low severity and
none blocks the PR.

## Earlier findings — verification

| # | Finding (review 1) | Status |
|---|---|---|
| 1 | High: arbitrary launcher values through the restrictive TOML literal encoder aborted every codex spawn | **Fixed** — `_toml_basic_string` (`backends/codex.py:516-524`) selected per key by `render_value` (`codex.py:505-508`); identity keys keep `_toml_literal`. Re-verified below |
| 2 | Medium: global test setup cleared the launcher after the singleton was selected | **Fixed** — `tests/conftest.py` is no longer modified; the affected test clears the allowlist locally (`test_codex.py:421`) using the real constant, so there is no second copy to drift |
| 3 | Medium: `TMUX` omitted, nested delegation could fall to a detached server | **Fixed** — `TMUX` is in the allowlist (`process_manager.py:106`) and the precedence decision is recorded in `plan.md:17-21` |
| 4 | Low/med: rationale blamed MCP env replacement rather than daemon-created panes | **Fixed** — `process_manager.py:98-100` now states the daemon mechanism, which matches `HerdrProcessManager.spawn_process` (`process_manager.py:2946-2967`) |
| 5 | Low: global pi config needed an explicit rationale | **Fixed** — `server_simple.py:2801-2804` |
| 6 | Tests and feature documentation incomplete | **Fixed** — see the test table below; `plan.md`, `plan-review.md`, `implementation.md`, `implementation-review-1.md` all present; `docs/reference/agent-messaging-protocol.md:55-72` updated |
| B1 | Plan review: ASCII-only JSON escaping emits TOML-invalid surrogate pairs for astral scalars | **Fixed** — non-ASCII scalars stay literal, DEL is escaped explicitly (`codex.py:522-524`) |

Also confirmed resolved from the first review's lower-severity items: the helper is now
Linux-named **and** Linux-gated (`nested_linux_launcher_env`, `process_manager.py:113-120`),
the omission test asserts the whole tuple via `isdisjoint` (`test_pi_worker_identity.py:155`),
the inheritance tests assert identity survives the merge (`test_codex.py:464-466`), and
the constant block placement nit is addressed.

## The corrected TOML encoder — independent verification

Round-tripped every class through `CodexBackend._toml_basic_string` and `tomllib`,
asserting both value equality and that no raw newline or CR can reach the argv token.
**21 classes, 0 failures**: apostrophe, `\n`, `\r`, tab, U+0001, U+007F, `"`, `\`,
Latin-1, CJK, astral U+1F680, a C1 control, U+2028, empty string, and the
TOML-structural characters `=`, `,`, `}`.

Because `_toml_basic_string` post-processes `json.dumps` output with a `.replace`, I
specifically attacked that seam — all six cases round-trip exactly and none introduces a
key:

```
literal-escape-text  '\u007f' (6 chars)   -> "\\u007f"                ok, no extra keys
backslash+DEL                             -> "\\\u007f"               ok, no extra keys
DEL+backslash                             -> "\u007f\\"               ok, no extra keys
quote+DEL                                 -> "\"\u007f\""             ok, no extra keys
brace-injection  '", EVIL = "pwned'       -> "\", EVIL = \"pwned"     ok, no extra keys
inline-table-close '}, OTHER = {x = 1'    -> "}, OTHER = {x = 1"      ok, no extra keys
```

A lone surrogate is rejected with a key-naming diagnostic
(`launcher environment HERDR_CONFIG_PATH contains a non-Unicode scalar value`), which is
the right call: `os.environ` yields lone surrogates for undecodable bytes on POSIX, and
they cannot be represented in TOML or encoded into argv.

Note the split is well chosen: `WIN_AGENT_TEAMS_HERDR_WORKSPACE` is validated upstream
(`_configured_workspace`, `process_manager.py:2453-2482` — I confirmed it rejects control
characters at import), but `HERDR_CONFIG_PATH`, `WIN_AGENT_TEAMS_LINUX_TERMINAL` and
`TMUX` are validated nowhere, so the encoder genuinely has to be total for them.

## End-to-end behaviour verified

- **Claude and pi configs** round-trip a legal-but-awkward environment
  (`WIN_AGENT_TEAMS_HERDR_WORKSPACE="Mikael's räksmörgås 🚀"`,
  `HERDR_CONFIG_PATH=/home/o'brien/.config/herdr/herdr.toml`) with identity intact.
- **Codex spawn *and* resume** both emit the same override and parse back correctly with
  those values.
- **Recursion**: a level-2 server constructed from a generated config re-emits the
  identical allowlist.
- **Stale `TMUX` fails loudly rather than silently detaching**, as `plan.md:19-21`
  claims: `TmuxProcessManager.spawn_process` runs the tmux argv with `check=True`
  (`process_manager.py:1495-1503`), so a dead server raises instead of falling through
  to the detached-session branch.

### Live smoke test — corroborated independently

`implementation.md:57-67` is accurate. I checked the artifacts directly:

- `mcp/nested-smoke-child.mcp.json` and `mcp/nested-smoke-codex.mcp.json` were written by
  the **nested** server (`AGENT_PARENT_NAME: nested-smoke-parent`) and each carries
  `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` + `WIN_AGENT_TEAMS_HERDR_SESSION=nestedfixsmoke`.
- The launch logs show `session=nestedfixsmoke tab=w1:t2` (claude child) and
  `tab=w1:t4` (codex child), same `workspace=w1` as the parent's `tab=w1:t1` — so the
  second-level spawns landed in the parent's Herdr session, not a standalone terminal.
  That is the exact failure this branch fixes, demonstrated end to end for both backends.

## Gates (this worktree, both environment states)

```
uv run ruff format --check .   86 files already formatted
uv run ruff check .            All checks passed
uv run ty check                All checks passed
uv run pytest                  1683 passed, 4 skipped   (WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr)
uv run pytest                  1683 passed, 4 skipped   (launcher/HERDR_* unset, as CI runs)
```

The environment-sensitivity problem from review 1 is gone: the suite is now green in
both states, which CI alone can never establish.

## Findings (all low, none blocking)

1. **`build_resume_command` has no launcher assertion.** `plan.md:69` promises "Codex's
   spawn **and resume** override includes launcher policy and identity", but
   `test_build_resume_command_injects_identity_env_override` (`test_codex.py:442-450`)
   still asserts only the identity substring. Risk is genuinely low — `_mcp_identity_args`
   is shared by both paths and I verified resume manually — but the promised assertion is
   one line.
2. **The POSIX gate is untested and never exercised.** `nested_linux_launcher_env`
   returns `{}` when `os.name == "nt"` (`process_manager.py:114-115`); no test names the
   helper, and CI is Linux-only, so that branch has never run. A
   `monkeypatch.setattr(process_manager_module.os, "name", "nt")` test would close it.
3. **Stale line citations in the section the change edits.** `agent-messaging-protocol.md`
   cites `server_simple.py:1107-1124` for `_write_mcp_config` (actually `:2595`) and
   `codex.py:465-487` / `:558-570` for `_mcp_identity_args` / `build_env` (actually
   `:483` / `:601`). These were already stale at `HEAD` — not introduced here — but the
   change rewrites those exact bullets, and per `CLAUDE.md` trivial behaviour-preserving
   breakage in touched lines is worth fixing on the spot.
4. **Docstring density below the surrounding bar.** `_toml_basic_string`
   (`codex.py:516-517`) is documented as rendering "arbitrary launcher configuration",
   yet it rejects lone surrogates — the real contract lives only in the inline comment,
   while `_toml_literal` right below it documents its refusal in the docstring.
   Similarly `_write_mcp_config`'s one-line docstring (`server_simple.py:2596`) no longer
   describes what the file now carries.
5. **tmux was dropped from the smoke record.** `plan.md:83-84` requires exercising a
   two-level tmux spawn "when tmux is available; otherwise record the environment
   limitation". The rewritten `implementation.md:57-67` covers Herdr for claude and codex
   but says nothing about tmux either way. One sentence would discharge it.

## Residual risks (accepted, worth knowing)

- **`TMUX` now lands in on-disk configs and in codex argv** (`ps`-visible). It is not a
  secret — it names a socket path, server pid and session id that are already
  user-scoped — but it is the first handle-like value the allowlist carries, so future
  additions deserve the same scrutiny the fixed allowlist currently earns.
- **A pinned `WIN_AGENT_TEAMS_HERDR_WORKSPACE` pins the whole descendant tree**, even for
  a child working in another repository (recorded at `plan.md:94-95`).
- **Invalid inherited Herdr settings now fail the nested server at import**, because
  `HerdrProcessManager.__init__` validates at module import (`process_manager.py:3549`).
  Symmetric with the parent, and recorded at `plan.md:96-97`.
- **Scope is clean**: the diff touches only what `plan.md:44-61` lists, plus the two
  whitespace-only hunks around the relocated constant block. No unrelated changes.

---

## Addendum — post-review verification of the five findings (2026-09-22)

Reviewer: Claude Code (Opus 5). Only the five deltas were inspected. No file other than
this review was modified.

**All five confirmed. Approval stands, unconditionally.**

1. **Resume assertion — confirmed.** `test_build_resume_command_injects_identity_env_override`
   (`tests/test_backends/test_codex.py:442-453`) now asserts
   `'WIN_AGENT_TEAMS_LINUX_LAUNCHER = "herdr"' in token`. Better than what the finding
   asked for: matching the double-quoted form also pins the *basic-string* encoding on
   the resume path, not just the key's presence.
2. **POSIX gate — confirmed.** `test_nested_launcher_env_is_disabled_on_windows`
   (`tests/test_pi_worker_identity.py:160-164`) sets the launcher, patches
   `process_manager_module.os.name` to `"nt"` and asserts `{}`. The previously unreachable
   branch is now covered on Linux CI.
3. **Citations — confirmed.** All three stale references in the rewritten bullets are
   corrected and I re-resolved each one: `server_simple.py:2595` → `def _write_mcp_config`,
   `codex.py:483` → `def _mcp_identity_args`, `codex.py:601` → `def build_env`. Two
   citations in the same claude-code bullet remain slightly off (`claude_code.py:173-175`
   vs the `--mcp-config` extend at `:177`; `:283-289` vs the `AGENT_NAME` assignment at
   `:294`). They pre-date this branch, were not part of finding 3, and are noted only so
   the record is complete.
4. **Docstrings — confirmed.** `_toml_basic_string` (`codex.py:517`) now reads "Render
   launcher config as TOML, rejecting non-scalar Unicode", so the refusal is in the
   contract rather than only an inline comment, matching `_toml_literal` below it.
   `_write_mcp_config` (`server_simple.py:2596`) now says it writes identity **and
   launcher policy**.
5. **tmux smoke — confirmed, and independently corroborated on disk.** The isolated
   server `-L watnestedfix` is still live and its state matches the record exactly:

   ```
   smoke:1.2 pid=2132552 cmd=claude      <- nested-tmux-child
   smoke:1.3 pid=2129271 cmd=claude      <- nested-tmux-parent
   ```

   Both PIDs match the registry (`spawned_by: nested-tmux-parent` for the child), and
   `mcp/nested-tmux-child.mcp.json` — written by the **nested** server
   (`AGENT_PARENT_NAME: nested-tmux-parent`) — carries
   `WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux`, `WIN_AGENT_TEAMS_TMUX_TARGET=smoke` and
   `TMUX=/tmp/tmux-1000/watnestedfix,2123027,0`. The second-level spawn therefore landed
   on the parent's isolated tmux server rather than in a detached
   `win-agent-teams-<session>` session. That is the `TMUX` propagation decision
   (review 1 finding 3 / plan correction C4) demonstrated live, which until now rested on
   unit assertions alone. With the Herdr runs above, all three launchers and both
   backends are now covered by real nested spawns.

### Gates after the delta

```
uv run ruff format --check .   86 files already formatted
uv run ruff check .            All checks passed
uv run ty check                All checks passed
uv run pytest                  1684 passed, 4 skipped   (WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr)
uv run pytest                  1684 passed, 4 skipped   (launcher/HERDR_* unset, as CI runs)
```

One net new test (1683 → 1684) plus the resume assertion strengthened in place, green in
both environment states. No findings remain; the branch is ready to commit and open as a PR.

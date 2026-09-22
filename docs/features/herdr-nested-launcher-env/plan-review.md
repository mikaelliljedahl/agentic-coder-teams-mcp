# Plan review — Nested Linux launcher environment

Reviewer: Claude Code (Opus 5), independent of the implementer.
Reviewed: `docs/features/herdr-nested-launcher-env/plan.md` (46 lines) against
`CLAUDE.md`, the repository architecture, and the existing `docs/features/*/plan.md`
precedent. No code was modified. Working-tree state is cited only as corroboration.

## Verdict

**Approve the design; the document is not yet an approvable plan.** The diagnosis is
correct, the fix is in the right place, and the scope boundaries are the right ones.
But the plan omits two of the six sections `CLAUDE.md` mandates ("files affected",
"test cases"), and one design line — item 3, "encode arbitrary launcher values as TOML
basic strings" — is under-specified in a way that has already produced two encoding
holes in the working tree. Resolve C1–C4 before this counts as reviewed; C5–C8 are
should-fix.

## What the plan gets right

- **The diagnosis is architecturally correct.** Herdr panes are created by the Herdr
  daemon (`HerdrProcessManager.spawn_process` → `pane run`,
  `src/claude_teams/backends/process_manager.py:2946-2967`), so the agent inherits only
  the explicit `export` prefix from `_build_posix_shell_command`
  (`process_manager.py:135-140`) — never the MCP server's environment. The generated
  per-agent MCP config is therefore the only carrier, which is what the plan chooses.
- **Design item 2** (merge the allowlist *before* `AGENT_*`) is the right precedence:
  identity stays authoritative and an inherited value can never shadow it.
- **Design item 1** puts the allowlist in `backends/process_manager.py`, which already
  owns launcher selection (`_select_linux_manager`, `process_manager.py:3525-3545`) and
  every launcher env constant. Correct home.
- **Scope's exclusion of the global pi config is well-reasoned.** `_ensure_pi_mcp_config`
  writes the machine-global `~/.pi/agent/mcp.json` (`server_simple.py:2793-2831`) shared
  by unrelated sessions; pinning it to one launcher would be wrong.
- **A fixed allowlist rather than a wildcard copy** is the right security shape: nothing
  secret can reach an on-disk config or codex argv (world-visible in `ps`).

## Gaps

### G1 — "Encode arbitrary launcher values as TOML basic strings" is not a specification
Design item 3 states the goal but no encoding contract, and TOML basic strings are not
JSON strings. TOML forbids *raw* U+0000–U+0008, U+000A–U+001F and **U+007F** inside a
basic string, and requires every escape to denote a valid Unicode scalar. Verified
against the current tree, where `_toml_basic_string` is
`json.dumps(value, ensure_ascii=False)` (`backends/codex.py:511-514`):

```
'a\x7fb'  -> '"a\x7fb"'  -> tomllib: Illegal character '\x7f' (at line 1, column 7)
'bad\udcffpath' -> '"bad\udcffpath"' -> UnicodeEncodeError when the argv is encoded
```

The second case is reachable on POSIX without any exotic input: `os.environ` decodes
undecodable bytes with `surrogateescape`, so a `HERDR_CONFIG_PATH` containing non-UTF-8
bytes yields a lone surrogate and the codex spawn dies on argv encoding — the same
failure class the plan set out to remove. The plan must state the contract, not delegate
it to an encoder that does not meet it.

### G2 — `CLAUDE.md`'s mandated sections are missing
`CLAUDE.md` requires "scope, current behavior, proposed design, **files affected**,
risks, and **test cases**". The plan has no *Files affected* section, and "test cases"
is one line (design item 4) naming five categories with no test names, no expected
outcomes and no red-first ordering. Every sibling plan in this repo carries both as
first-class sections — e.g. `docs/features/subprocess-stdin-devnull/plan.md:78-117` and
`docs/features/herdr-workspace-per-repo/plan.md:201-283`. This one is below the
established bar, and the omission is not cosmetic: nothing in the plan records that
**both spawn and resume** paths must carry the env (`server_simple.py:3297`/`:3967`,
`_hook_extra` at `:3324`/`:4008`, and codex `build_command`/`build_resume_command`).

### G3 — `TMUX` is neither named nor justified
Scope says "tmux selection/context variables". `TMUX` is the one variable that decides
*which tmux server is addressed* and whether the manager splits the user's session at
all (`_inside_tmux` is literally `bool(os.environ.get("TMUX"))`, `process_manager.py:1819-1820`;
the three-way branch is at `:1640-1721`). It is also the one key whose value is
authoritative from a **different source**: tmux sets an accurate `TMUX` in each pane, so
copying the parent's value into the config *overrides* the pane-accurate one, and a
stale value addresses a dead server. That is a deliberate precedence decision and the
plan does not make it. (The working tree added `TMUX` to the allowlist —
`process_manager.py:101-110` — with no recorded rationale.)

### G4 — Risk 3 covers only the new tests, not the pre-existing ones
The plan correctly refuses to mutate global test setup, but the reciprocal obligation is
unstated: **existing tests that assert an exact env mapping must be made
launcher-independent**, or they fail for exactly the users this fix targets. Verified in
the current tree:

```
FAILED tests/test_backends/test_codex.py::TestCodexMcpIdentity::test_build_command_injects_identity_env_override
  Left contains 1 more item: {'WIN_AGENT_TEAMS_LINUX_LAUNCHER': 'herdr'}
1 failed, 1681 passed, 4 skipped
```

Green on CI (which does not set the variable), red in any shell running the Herdr
launcher. The plan's verification section must require the suite to be green in *both*
environment states, because CI can only ever prove one of them.

### G5 — The helper is Linux-named but not Linux-gated
The plan is titled "Nested **Linux** launcher environment" and scopes itself to Linux
launcher policy, but nothing in it says the carrier is skipped on Windows. In the
current tree `nested_linux_launcher_env()` (`process_manager.py:113-120`) reads
`os.environ` unconditionally, so a Windows user with `TMUX` or `HERDR_CONFIG_PATH` set
would newly place a double-quoted basic string into a codex argv token — precisely what
`_toml_literal`'s existing rationale avoids ("Single-quoted literals avoid Windows
`CreateProcess` double-quote escaping issues for the single argv token",
`codex.py:516-524`), and what `cmd.exe` mangles on the `codex.cmd` shim path
(`codex.py:531-540`). Either gate the helper on `os.name != "nt"` or state the risk as
accepted.

### G6 — Two risks are unrecorded
- Propagating `WIN_AGENT_TEAMS_HERDR_WORKSPACE` pins the whole agent *tree*, not just
  this process, to one label. That interacts with the per-repo workspace feature (#62):
  a nested agent working in a different repo is forced into the parent's pinned
  workspace, instead of `_workspace_label`'s per-repo derivation
  (`process_manager.py:2818-2829`). Intended, probably — but it changes #62's behaviour
  for nested agents and must be written down.
- Nested servers now inherit **validation failures**. `_configured_session` /
  `_configured_workspace` raise at `HerdrProcessManager.__init__`
  (`process_manager.py:2441-2482`), which runs at import (`process_manager.py:3545`), so
  an invalid inherited value fails the nested server at import rather than at spawn.
  Symmetric with the parent, but worth one line.

### G7 — The live verification does not cover the backend that changed
Verification names "a live two-level **Claude** spawn". Claude's config is plain JSON and
was never at risk; the only serialization that changed is codex's `-c` override. The
live check must include a two-level **codex** spawn under Herdr, and — given G3 — a
two-level spawn under `WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux`.

### G8 — No documentation delta, and no file:line citations
Two user-facing documents describe exactly what this change alters and are not in scope:
`README.md:249-289` (launcher env table) and
`docs/reference/agent-messaging-protocol.md:54-68` ("How a worker receives its
identity"), which `CLAUDE.md` names as the thing to read instead of re-deriving the
protocol. State the updates, or state explicitly that none are needed — including for
MCP tool docstrings, which `CLAUDE.md` singles out as the only thing a consuming agent
reads (nothing here changes an orchestrator-visible contract, but the plan should say
so rather than leave it inferred). Separately, the plan cites no `file:line` anywhere
except a bare module name; sibling plans quote exact signatures and line ranges.

## Required corrections

1. **C1 (G1)** — Specify the TOML encoding contract in the plan: which code points must
   be escaped (U+0000–U+0008, U+000A–U+001F, U+007F), what happens to a lone surrogate
   from `surrogateescape`, and the chosen mechanism (e.g. `ensure_ascii=True` plus an
   explicit rejection or sanitisation rule for non-scalar values). Add a test case that
   round-trips each class through `tomllib`.
2. **C2 (G2)** — Add `## Files affected` and `## Test cases (red first)` sections in the
   house format. Files affected must name both the spawn and resume call sites.
3. **C3 (G4)** — Add to Verification: the full suite must be green both with and without
   `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` set, and list the pre-existing exact-mapping
   tests that must be relaxed (at minimum
   `test_codex.py::TestCodexMcpIdentity::test_build_command_injects_identity_env_override`).
4. **C4 (G3)** — Name `TMUX` in Scope and record the precedence decision: copied parent
   value vs. the pane-accurate value tmux sets itself, and what happens when the copied
   value is stale.
5. **C5 (G5)** — State whether the carrier is skipped on Windows; if not, record why the
   double-quote/argv rationale in `_toml_literal` does not apply.
6. **C6 (G6)** — Add the two missing risks (tree-wide workspace pinning vs. #62;
   inherited validation failures raising at nested import).
7. **C7 (G7)** — Extend the live verification to a two-level **codex** spawn under Herdr
   and a two-level spawn under the tmux launcher.
8. **C8 (G8)** — Record the doc delta (or its absence, with reasons) for `README.md`,
   `docs/reference/agent-messaging-protocol.md` and MCP tool docstrings; add `file:line`
   citations to Problem and Design.

## Process note

`CLAUDE.md` orders the workflow plan → independent plan review → implement. The
implementation is already present in the working tree (`git status`: five modified
source/test files) while `plan.md` is still untracked and this review is only happening
now, so steps 1–3 ran out of order. Not a defect in the design, but the record should
say so in `implementation.md` rather than imply the sequence was followed. For
completeness of the current state: `ruff format --check` and `ty check` pass;
`ruff check` reports 2 findings (E501 at `codex.py:506`, I001 in
`tests/test_pi_worker_identity.py:13`) and `pytest` is 1 failed / 1681 passed as quoted
in G4 — all four gates must be green before the PR per `CLAUDE.md`.

---

## Addendum — re-review of the corrected plan (2026-09-22)

Re-reviewed `plan.md` against C1–C8. Reviewer: Claude Code (Opus 5). No code or plan
modified.

**Verdict: not yet approved — one blocking issue (C1).** Six of the eight corrections
are fully resolved and two more are resolved in substance; the remaining blocker is a
reachable encoding hole that the corrected plan asserts does not exist.

| | Status |
|---|---|
| C1 TOML encoding contract | **Blocking — incomplete** (below) |
| C2 Files affected / test cases | Resolved (`plan.md:44-73`), both sections present and naming spawn *and* resume |
| C3 Both env states | Resolved (`plan.md:79-80`, `:72-73`); see minor note |
| C4 `TMUX` named + precedence | Resolved (`plan.md:17-21`) — the decision is stated, not merely the key |
| C5 POSIX gating | Resolved (`plan.md:24-25`) |
| C6 Missing risks | Resolved (`plan.md:94-97`) |
| C7 Live codex + tmux | Resolved (`plan.md:81-84`) |
| C8 Doc delta / citations | Doc delta resolved (`plan.md:58-61`); citations still thin, non-blocking |

### B1 (blocking) — the C1 remedy fails on non-BMP characters

`plan.md:36-38` now asserts:

> Encoding uses JSON-compatible ASCII escapes, which TOML accepts, so controls including
> U+0000–U+0008, U+000A–U+001F, and U+007F round-trip.

True for controls, U+007F, apostrophes, newlines, Latin-1 and CJK — I re-verified each
through `tomllib`. It is **false for any character outside the BMP**, because
`ensure_ascii=True` emits UTF-16 **surrogate pairs** and TOML rejects each half as a
non-scalar escape. Reproduced end to end against the current tree with
`WIN_AGENT_TEAMS_HERDR_WORKSPACE="🚀 agents"`:

```
argv token: { ... WIN_AGENT_TEAMS_HERDR_WORKSPACE = "🚀 agents", ... }
CODEX WOULD REJECT -> TOMLDecodeError: Escaped character is not a Unicode scalar value (at line 1, column 88)
```

This input is legal today by design: `_configured_workspace`
(`backends/process_manager.py:2453-2482`) accepts any printable label up to 128
characters precisely because a workspace label is *display text*, and `"🚀 agents"`
satisfies `isprintable()`. So an ordinary emoji in a display label breaks **every** codex
spawn — the same failure mode C1 was raised to remove, moved from `'` to U+1F680.

The planned tests would not catch it: test case 4 (`plan.md:70-71`) says "apostrophe,
newline, U+0001, U+007F, and non-ASCII text", and the natural reading of "non-ASCII
text" (é, CJK) passes.

**Required to clear B1:**
1. Correct the claim in design item 3. JSON's ASCII escaping is not a TOML-safe encoder
   for the astral planes; say which mechanism is used instead. Two verified options:
   emit TOML's 8-digit `\UXXXXXXXX` for non-BMP scalars, or keep non-ASCII literal
   (TOML basic strings permit it) and escape only `"`, `\`, the C0 controls and U+007F.
   Both round-trip all seven classes I tested.
2. Make test case 4 name an **astral-plane** character explicitly (e.g. U+1F680)
   alongside the BMP ones, so the gap cannot reappear silently.

Note also that rejecting lone surrogates (`plan.md:38-40`) still hard-fails the spawn,
just with a better message. That is defensible for undecodable bytes, but the plan
should say so deliberately — the alternative (drop the offending key with a warning and
still spawn) keeps the agent usable, and the plan currently does not weigh the two.

### Non-blocking residue

- **C3** — Verification requires the suite to be green in both environment states, and
  test case 5 requires existing exact-mapping tests to clear launcher values, but no test
  is named. `test_codex.py::TestCodexMcpIdentity::test_build_command_injects_identity_env_override`
  is the one that is red today in a Herdr shell; naming it removes the ambiguity.
- **C8** — Only one citation was added (`plan.md:10`), and it is a symbol plus a bare
  path with no line numbers; sibling plans quote exact ranges.
- **Problem statement** still reads "may therefore miss the launcher's environment"
  (`plan.md:7-8`). The mechanism is settled, not conditional — Herdr and tmux panes are
  created by long-lived daemons, so the agent inherits only the `export` prefix from
  `_build_posix_shell_command` (`process_manager.py:135-140`). The implementation comment
  (`process_manager.py:98-100`) already states this correctly; the plan should match it.
- `plan.md:38` lists U+0000 among the controls that round-trip. A NUL cannot occur in a
  POSIX environment value; harmless, but it overstates the tested surface.

Clear B1 and this plan is approved; the residue above can be folded into
`implementation.md` rather than gating another review round.

---

## Final check — B1 cleared, plan approved (2026-09-22)

Reviewer: Claude Code (Opus 5). No code or `plan.md` modified.

**Approved. No blocking issues remain.**

Design item 3 (`plan.md:35-41`) now specifies the encoding instead of delegating it:
JSON-compatible escapes for quotes, backslashes and C0 controls, an explicit escape for
U+007F, non-ASCII scalars kept literal so astral characters are never converted to
TOML-invalid surrogate pairs, and a key-naming hard failure for lone surrogates. Test
case 4 (`plan.md:70-72`) now names the astral U+1F680 case that the previous wording
would have missed.

Verified against the implementation (`backends/codex.py:515-524`), all 14 classes
round-trip through `tomllib` with zero failures — apostrophe, newline, CR, tab, U+0001,
U+007F, `"`, `\`, Latin-1, CJK, U+1F680, a C1 control, U+2028, and TOML-structural
characters (`=`, `,`, `}`) — and no raw newline or CR can reach the argv token. A lone
surrogate is rejected with `launcher environment HERDR_CONFIG_PATH contains a
non-Unicode scalar value`. The B1 reproducer is now green end to end:

```
WIN_AGENT_TEAMS_HERDR_WORKSPACE = "🚀 agents"   ->  parsed: {'WIN_AGENT_TEAMS_HERDR_WORKSPACE': '🚀 agents', ...}
```

Gates in this worktree, both environment states as the plan requires:

```
ruff format --check .  86 files already formatted
ruff check .           All checks passed
ty check               All checks passed
pytest                 1683 passed, 4 skipped   (WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr)
pytest                 1683 passed, 4 skipped   (variable unset, as CI runs)
```

The non-blocking residue from the previous addendum stands as written: fold it into
`implementation.md` (which still owes the red/green evidence, the out-of-order
plan/implement sequence, and the `docs/reference/agent-messaging-protocol.md` update the
plan commits to at `plan.md:58-61`) rather than another review round.

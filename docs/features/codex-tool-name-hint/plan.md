# Codex team-tool name hint — implementation plan

Branch `fix/codex-tool-name-hint` (worktree `wt-codex-tool-name-hint`, from
`origin/main` `471a175`). Codex source: `rust-v0.156.1` (`codex-rs/…`). Every
citation was opened at `path:line`; in `server_simple.py` the symbol wins.

## 0. Scope

**Goal.** Every instruction win-agent-teams injects into a **Codex** agent names
the team tools by their exact model-visible name, and steers away from Codex's
built-in multi-agent tools that share the bare names.

In scope: the Codex spawn prompt, the Codex follow-up (resume) prompt, the
`create_join_ticket` `join_prompt`, and the Codex-only `codex queue` wake notice.
Out of scope: Pi (already names its adapter spelling, §1.4), Claude Code (no
collision), a human-launched Codex root lead (we inject nothing into it; README
note only), tool docstrings (unchanged, so the `tools` golden is untouched).

## 1. Current behaviour

**1.1 Observed failure (2026-09-26).** Rollout
`~/.codex/sessions/2026/09/26/rollout-2026-09-26T15-22-53-01a0dde2-07ed-7281-ae81-6228959e415c.jsonl`:
line 4 is Codex's own developer text ("You are `/root`, the primary agent in a
team of agents"); line 9 is our prompt "Call send_message(text='SMOKE-UP-1') to
your lead"; line 13 calls `send_message` in namespace `collaboration`
(`target:"lead"`); line 15 returns "live agent path `/root/lead` not found";
line 19 calls `collaboration.list_agents` (only `/root`); line 26 reports "no
lead agent available". After the corrective follow-up (line 34), line 44 runs
code-mode `exec` with `tools.mcp__win_agent_teams__send_message({text:…})` and
line 47 returns `{"success":true,"to":"team-lead"}`.

**1.2 Codex built-ins that collide.** Multi-agent v2 registers `send_message`,
`followup_task`, `interrupt_agent`, `list_agents` (+ optional `wait_agent`)
under a namespace (`codex-rs/core/src/tools/spec_plan.rs:671-690`), default
`"collaboration"` (`codex-rs/core/src/config/mod.rs:256`); `spawn_agent` is in
the same namespace (`codex-rs/core/src/tools/router.rs:46-49`). So our
`send_message`, `list_agents` and `spawn_agent` collide by bare name;
`read_messages` and the `external_*` tools do not.

**1.3 How Codex names MCP tools.** The callable namespace is the raw config key
(`codex-rs/codex-mcp/src/rmcp_client.rs:850`), passed through
`sanitize_responses_api_tool_name` — every char outside `[A-Za-z0-9_]` becomes
`_` (`codex-rs/codex-mcp/src/mcp/mod.rs:563-578`) — then prefixed `mcp__`
(`codex-rs/codex-mcp/src/tools.rs:139-146`, `:228-233`) and joined with `__`
and exposed to code mode as `tools.<name>` (`codex-rs/tools/src/code_mode.rs:184-195`).
Key `win-agent-teams` ⇒ `mcp__win_agent_teams__send_message` **in Codex's
standard configuration**: `NonPrefixedMcpToolNames` / per-server
`non_prefixed_mcp_tool_names` (`core/src/config/mod.rs:857-858, 1837-1840,
3354-3365`) and collision suffixing (`codex-mcp/src/tools.rs:153-209`) can
change it (plan-review F3). Claude Code keeps the hyphens:
`mcp__win-agent-teams__send_message` (`.claude/skills/agent-orchestration/SKILL.md:64`).

**1.4 What we inject today.**
- Spawn: `CodexBackend.build_command` appends
  `_prompt_arg(request, _correlated_prompt(request))`
  (`src/claude_teams/backends/codex.py:394-398`) — caller text + correlation
  marker only (`codex.py:591-609`). No tool guidance.
- Follow-up: the server appends the delivery marker for non-claude backends
  (`server_simple._materialize_prompt`, `server_simple.py:2830-2838`), then
  `build_resume_command` appends `_prompt_arg(request)` (`codex.py:429-431`).
- Join prompt: `_build_join_prompt` (`server_simple.py:832-866`) uses bare
  `external_read`/`external_send`/`leave_team`, and (flag on only) step 6 bare
  `external_set_wake`. `create_join_ticket(name, note)` (`server_simple.py:3049`)
  has no member-backend parameter: **the join prompt is backend-agnostic**.
- Codex member wake notice: `"… call external_read with your member_token"`
  (`src/claude_teams/native_wake.py:597-600`), built only for `codex queue`.
- Pi already appends `--append-system-prompt` naming
  `win_agent_teams_send_message` (`src/claude_teams/backends/pi.py:91-99`,
  `:430-449`) — Pi has no native subagents, so no collision. **No Pi change.**

**1.5 Server key.** The server cannot learn the client-side config key: MCP does
not transmit it, and `FastMCP(name="win-agent-teams")`
(`server_simple.py:354-356`) is a self-declared name, not the key. The Codex
backend already hard-depends on the key being `win-agent-teams`:
`_MCP_SERVER_NAME` (`codex.py:96`) drives the identity override
`-c mcp_servers.win-agent-teams.env=…` (`codex.py:494-524`). A different key
already breaks Codex identity. (The converse does not hold: identity can work
while a name-changing Codex option makes the hint inexact — §1.3, F3.)

## 2. Design

**2.1 Name derivation (Codex backend).** Add a module-level helper in
`backends/codex.py`:
`codex_mcp_tool_name(tool, server=CodexBackend._MCP_SERVER_NAME) -> str` =
`"mcp__" + re.sub(r"[^A-Za-z0-9_]", "_", server) + "__" + tool`, a documented
mirror of the Codex sanitizer (§1.3). **Decision: reuse the existing hardcoded
key** rather than add an env var — one constant already gates identity; an
override knob would need to change both together (Q1). The spelling is exact
for the standard Codex configuration only; README says so (F3).

**2.2 Codex prompt hint.** A constant-built single paragraph
`_TEAM_TOOL_HINT`, appended as the **last** paragraph (`"\n\n"` separator) of
the prompt text on both spawn and resume, before `_prompt_arg` (so the cmd-shim
JSON fallback still applies):

spawn wraps `self._correlated_prompt(request)`, resume wraps `request.prompt`,
each via `self._with_team_tool_hint(text)`.

Draft text (ASCII, no `< > | & ^ ! ( )`, no newline inside):

> win-agent-teams: your lead and teammates are reachable only through the
> win-agent-teams MCP tools. Message your lead with
> mcp__win_agent_teams__send_message. Work from your lead arrives as a new
> prompt, not in an inbox; mcp__win_agent_teams__read_messages reads only
> messages sent to you by agents you spawned yourself. spawn_agent, list_agents
> and follow_up_agent use the same mcp__win_agent_teams__ prefix. In code mode call
> them as tools.mcp__win_agent_teams__send_message. Do not use Codex's built-in
> collaboration tools such as collaboration.send_message or
> collaboration.list_agents for this: they only reach Codex-internal
> subagents, and your lead is not one.

Hint after the markers is safe: receipt matching `findall`s the whole user
record (`src/claude_teams/delivery.py:198-236`). Why the prompt and not
`-c developer_instructions=…` (`codex-rs/config/src/config_toml.rs:243`): a `-c`
override **replaces** a user's own `developer_instructions`; the prompt route is
the one proven by §1.1 line 34.

**2.3 Join prompt (backend-agnostic).** The lead does not know the member's
harness, so add one line right after `Join protocol:` naming every tool the
protocol uses (F2), keeping the literal argument examples in steps 1-5:

> `Tool names: in Codex call mcp__win_agent_teams__join_team,
> mcp__win_agent_teams__external_read, mcp__win_agent_teams__external_send and
> mcp__win_agent_teams__leave_team; in Claude Code the same tools use the prefix
> mcp__win-agent-teams__ instead.`

The `join_team` response `instructions` (`server_simple.py:938-941`) gets the
same sentence, since it restates the member tools (F2: decided yes).

Justification: `external_*` has no bare-name collision, but a Codex member in
code mode only sees MCP tools through `ALL_TOOLS` (§1.1 line 38 had to search
for them); naming the full spelling removes that discovery step for one line of
cost. Both spellings, not a heuristic, because we cannot know the client. Step 6
(flag-on, already Codex-specific) names `mcp__win_agent_teams__external_set_wake`.
Names come from the §2.1 helper; the Claude spelling uses the raw key.

**2.4 Codex queue notice.** `native_wake.py:597-600`: replace
`call external_read` with `call mcp__win_agent_teams__external_read`. The name is
`[a-z_]` only, so the shim-safety comment at `native_wake.py:596` still holds.
Flag-on only.

**2.5 Native-wake flag-off identity.** `test_join_prompt_golden`
(`tests/test_native_wake_flag_off.py:46-57`) compares against
`tests/fixtures/native_wake/flag_off.json` `"prompt"`. **This golden must
change** (the §2.3 line). This is an intentional contract change, not
preservation of the old baseline (F5): capture the red golden failure first,
update only `"prompt"`, review the exact diff, and record in
`implementation.md` that the flag-off prompt now changes while flag-on still
adds only its wake step. The `"tools"` golden is untouched
(no docstring edits). The spawn/resume env test (`:94-133`) is unaffected —
argv is stubbed and env is unchanged.

## 3. Files affected

- `src/claude_teams/backends/codex.py` (helper, hint, two call sites);
  `server_simple.py` (`_build_join_prompt`); `native_wake.py` (queue notice).
- Tests: `tests/test_backends/test_codex.py`, `test_join_team.py`,
  `test_codex_member_wake.py`, `test_follow_up_delivery.py`; fixture
  `tests/fixtures/native_wake/flag_off.json` (`"prompt"` regenerated).
- `docs/reference/agent-messaging-protocol.md`: "Prompt materialization and
  transport" (~1515-1524), sharp edge "Codex workers do not poll
  `read_messages`" (~1777-1791), External members (~317). `README.md` Codex
  setup (~96-128): the hint, and "keep the key `win-agent-teams`; identity and
  the hint both assume it", and that Codex options that rename MCP tools (F3)
  make the hint inexact; fix the bare `send_message` walkthrough at
  `README.md:682-683` (F6).

## 4. Risks

- **Model still picks the built-in.** Mitigated by the explicit negative; only
  the §5 smoke can show it. **Key assumption:** a renamed key breaks identity
  already (§1.5); documented in README.
- **Codex drift.** `tool_namespace` is configurable
  (`codex-rs/core/src/config/mod.rs:3093`) and code-mode spelling may change;
  text says "such as", pinned by source citation to 0.156.1.
- **Repeated hint.** ~90 tokens per follow-up; Pi accepts the same (`pi.py:86-88`).
- **Existing exact-argv tests.** `test_codex.py:829`, `:843`, `:854` assert
  resume argv equality; they must be updated. `:843` changes shape: a one-line
  prompt now gains `\n\n` and is JSON-wrapped via the shim — intended.

## 5. Test cases (red first)

1. `test_codex_mcp_tool_name_mirrors_codex_sanitizer` — `win-agent-teams` →
   `mcp__win_agent_teams__send_message`; `a.b-c` → `mcp__a_b_c__x`.
2. `test_spawn_prompt_names_full_team_tools` — `build_command(...)[-1]` contains
   `mcp__win_agent_teams__send_message`, `mcp__win_agent_teams__read_messages`,
   `collaboration.send_message`; starts with the caller text; correlation token
   occurs exactly once.
3. `test_resume_prompt_names_full_team_tools` — same on `build_resume_command`,
   delivery marker in `request.prompt` preserved; hint does NOT tell the worker
   to read lead replies via `read_messages` (asserts the "arrives as a new
   prompt" wording, F1); update `:829/843/854`.
4. `test_team_tool_hint_is_shim_safe` — hint has no `\n`, `\r`, or any of
   `<>|&^!()`; via the shim the resume argv is JSON-wrapped and decoding the
   *complete* prompt yields caller text + one delivery marker + the hint (F6).
5. `test_hint_is_codex_only` — Claude Code / Pi argv lack the Codex hint.
6. Follow-up end to end (F4): `test_follow_up_delivery.py` uses
   `_FakeResumeBackend`, which never builds argv, so it asserts only the
   materialized prompt (delivery marker); the final argv (marker + hint) is
   asserted red-first in `test_codex.py` via `CodexBackend.build_resume_command`.
7. `tests/test_join_team.py`: flag-off `join_prompt` contains the Codex full
   names of `join_team`, `external_read`, `external_send`, `leave_team` (first
   and last action, F2) and the `mcp__win-agent-teams__` prefix; flag-on step 6
   contains `mcp__win_agent_teams__external_set_wake`; the `join_team` response
   `instructions` contains `mcp__win_agent_teams__external_read`. Then regenerate `flag_off.json`
   `"prompt"`; the golden diff is exactly the §2.3 line.
8. `tests/test_codex_member_wake.py`: queued notice argv contains
   `mcp__win_agent_teams__external_read`.

Then all four gates: `ruff format --check`, `ruff check`, `ty check`, `pytest`.

**Smoke (Linux).** Re-run §1.1: `spawn_agent(backend='codex')` with "Call send_message(text='SMOKE-UP-1') to your lead";
pass = lead inbox receives it on the first turn and the rollout shows no
`collaboration.send_message` call. Record in `implementation.md`.

## 6. Open questions

- **Q1.** Add a `WIN_AGENT_TEAMS_MCP_SERVER_KEY` override driving both
  `_MCP_SERVER_NAME` and the hint? Proposed: no (YAGNI; separate change).
- **Q2.** Keep the join-prompt line (golden churn, no collision)? Proposed: keep.
- **Q3.** Skip the hint when the caller's prompt already names
  `mcp__win_agent_teams__`? Proposed: always add — simpler, testable.

## 7. Review dispositions (plan-review.md)

All six findings accepted and folded in above: F1 hint wording (§2.2, test 3);
F2 all join-protocol tools + `join_team` response (§2.3, test 7); F3
standard-config scoping (§1.3, §1.5, §2.1, README); F4 argv asserted in
`test_codex.py` (test 6); F5 golden treated as intentional contract change
(§2.5); F6 full JSON round-trip, code-mode citation, README walkthrough.
Q1 no override, Q2 keep, Q3 always append — reviewer agrees.

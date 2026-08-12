# Plan — Pi spawn: `ask_user` deadlock and prompt corruption

Companion research: [pi-cli-findings.md](pi-cli-findings.md) (written by a spawned
pi agent that inspected its own installed `dist/`, `pi --help`, and ran runtime
experiments). Every claim below cites it or a local measurement.

## Scope

Two independent defects on the pi spawn path:

1. A spawned pi agent can block forever on an `ask_user`-style interactive tool
   instead of escalating to its parent.
2. A long / multi-line / non-ASCII prompt is mutated or truncated on the way in.

Out of scope: the `ctx is stale after session replacement` crash observed at
`agent_settled` from a user-global pi extension (reproduced below, unrelated to
this feature — report separately).

---

## Problem 1 — `ask_user` blocks the agent

### Current behavior (verified)

- Pi **core** ships no human-question tool at all (`read, bash, edit, write,
  grep, find, ls`); "No permission popups" is explicit in its README
  (findings A1).
- The blocking tool comes from a **user-global extension** listed in
  `~/.pi/agent/settings.json` — here `…\my-pi-setup\extensions\ask-user\index.ts`,
  registering tool name `ask_user` (findings A1).
- That extension self-disables when `ctx.mode !== "tui"`, returning
  *"No interactive UI is available…"* immediately (findings A2).
- **But on Windows we run pi in TUI mode.** `PiBackend.is_interactive` is
  `True`, so `WindowsProcessManager._should_use_interactive_console` →
  `provides_tty` → `True`, so `PiBackend._headless()` is `False` and
  `-p --mode json` is **not** passed
  ([pi.py:275](../../../src/claude_teams/backends/pi.py#L275),
  [process_manager.py:867](../../../src/claude_teams/backends/process_manager.py#L867)).
  The agent runs the full TUI in a Windows Terminal tab, `ctx.mode === "tui"`,
  and `ask_user` renders a prompt in a tab nobody is watching. It never returns.

So this is a **Windows/TUI-only** deadlock; the Linux/headless path already
degrades safely.

### Fix

Two layers, belt and braces:

1. **Hard block — `--exclude-tools`.** Emit `--exclude-tools <names>` on both
   `build_command` and `build_resume_command`. Default deny list:
   `ask_user,ask_question,ask_human,request_input`.
   Verified locally that pi tolerates names that are not registered
   (`--exclude-tools ask_user,definitely_not_a_tool` ran clean), so a static
   list is safe across installs.
   Overridable via `WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS` (comma list; empty string
   disables the flag entirely for debugging).
2. **Soft policy — `--append-system-prompt`.** Append a short escalation rule:
   never wait for a human; when blocked on a decision, call the MCP tool
   `send_message` (exposed to pi as `win_agent_teams_send_message`, recipient
   defaults to the parent) and then stop or continue under a stated assumption.
   `--append-system-prompt` is repeatable and reads the value as file contents
   when it names an existing file (findings A5) — we pass literal text.

**Rejected:** `--no-extensions -e …`. It would also drop the
settings-discovered `pi-mcp-adapter`, which is how a pi agent reaches the
win-agent-teams MCP tools at all. Same for the `PI_CODING_AGENT_DIR`
full-isolation recipe in findings A4 — correct for a sandbox, fatal for us.

Also **do not** switch pi to headless just to dodge this: the visible tab is a
deliberate feature, and headless would only mask the extension rather than
route the question to the lead.

---

## Problem 2 — prompt truncation / mutation

### Pi is not the culprit (verified)

- `dist/cli/args.js` pushes each argv token as one message entry; there is **no
  newline split** and no length cap on the initial prompt (findings B2, B3).
- `initial-message.js` joins stdin/file/first-message with `parts.join("")`;
  `print-mode.js:94` and `interactive-mode.js:630` each call
  `session.prompt(initialMessage)` **once**. Confirmed by reading both modes.
- A 21.2 KB multi-line Unicode argv token round-tripped byte-identical by
  SHA-256 (findings B5).

Every remaining defect is in **our launcher**.

### 2a. LF → CRLF rewrite in the Windows Terminal wrapper (measured)

`_write_tab_wrapper` bakes the argv into a `.ps1` and writes it with
`Path.write_text(...)`
([process_manager.py:1132](../../../src/claude_teams/backends/process_manager.py#L1132)).
`write_text` opens in **text mode**, so Python's universal-newline translation
rewrites *every* LF — including the ones inside the prompt literal — to CRLF.

Measured on this machine: a 140-char prompt with 7 newlines arrived at `node`
as 147 chars, different SHA-256. The em dash and `åäö` survived (the
`utf-8-sig` BOM handles that correctly for PowerShell 5.1).

**Fix:** write the wrapper with newline translation disabled
(`wrapper_path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8-sig"))`,
or `open(..., "w", encoding="utf-8-sig", newline="")`). The explicit `\r\n`
join already supplies the line endings PowerShell needs; the implicit
translation only corrupts payload.

This affects **claude-code and codex too** — same wrapper — so it is a shared
fix, not a pi fix.

### 2b. The `pi.cmd` shim fallback truncates at the first newline (dead safety net)

`PiBackend._prompt_args` already knows the shim path is dangerous and falls back
to `@<prompt_file>` — but only *"when the server wrote one"*
([pi.py:429](../../../src/claude_teams/backends/pi.py#L429)).
The server never writes one for pi: `_materialize_prompt` returns `{}` for every
backend that is not `claude-code`
([server_simple.py:2503](../../../src/claude_teams/server_simple.py#L2503)).

So the fallback is unreachable, and if `node` or `dist/cli.js` cannot be resolved
(non-npm layout, bun install, PATH oddity) the whole prompt goes through
`cmd.exe` and is truncated at the first newline — silently. This is the most
likely explanation for a user-visible "long prompt got cut".

**Fix (pick one, plan proposes the first):**

- Extend `_materialize_prompt` to write a prompt sidecar for pi when the prompt
  needs one, so the existing `@file` path becomes live. Note `@file` is not
  verbatim — pi wraps it as `<file name="…">…</file>` (findings B1/B5) — so keep
  the short plain-ASCII directive that already accompanies it.
- *Alternative:* make the shim fallback a hard `BackendBinaryNotFoundError`.
  Simpler and louder, but breaks any install where only the shim exists.

### 2c. Leading-character hazards

Pi interprets the **first** character of the prompt token, never later lines
(findings B4):

| leading token | pi behavior |
|---|---|
| `@…` (whole argv token) | CLI file include |
| `/…` | extension command / skill / prompt-template expansion |
| `-…` / `--…` as a separate token | parsed as a flag |
| `!…` | shell, **interactive editor only** — not our path |

`#`, backticks and `${…}` are inert. Direct `CreateProcess` (no shell) means no
interpolation.

**Fix:** in `_correlated_prompt`, if the final prompt starts with `@`, `/` or
`-`, prepend a single newline (or a one-line ASCII preamble). Cheap and
lossless.

### 2d. argv length ceiling on the headless path

The TUI path bakes argv into the `.ps1`, so no ceiling. The headless
(`-p --mode json`) path passes the prompt on a real command line: findings B5
hit `[WinError 206]` past ~32 KB.

**Fix:** if `len(prompt)` exceeds a conservative threshold (24 KB) on the
non-wrapper path, route via the sidecar from 2b.

### 2e. Not adopted

Piped **stdin** is UTF-8 but pi applies `.trim()` to it (findings B5/B6), and
our spawn already uses `stdin=PIPE`/`DEVNULL` for lifecycle reasons. Not worth
the churn given 2a–2d cover the observed failures.

---

## Files affected

| File | Change |
|---|---|
| `src/claude_teams/backends/pi.py` | `--exclude-tools`, `--append-system-prompt`, leading-char guard, sidecar threshold |
| `src/claude_teams/backends/process_manager.py` | `_write_tab_wrapper` newline-safe write |
| `src/claude_teams/server_simple.py` | `_materialize_prompt`: prompt sidecar for pi |
| `tests/test_backends/test_pi.py` | new cases |
| `tests/test_backends/test_process_manager*.py` | wrapper fidelity case |
| `README.md` / `docs/reference/agent-messaging-protocol.md` | document the pi escalation policy + env override |

## Test cases (red first)

1. `build_command` / `build_resume_command` contain `--exclude-tools` with the
   default deny list, and honour `WIN_AGENT_TEAMS_PI_EXCLUDE_TOOLS`
   (including empty = omit the flag).
2. Both commands contain `--append-system-prompt` with the escalation text.
3. `_write_tab_wrapper` round-trip: write a wrapper for a prompt containing
   `\n`, `'`, `—`, `åäö`; read the file **as bytes** and assert the prompt
   literal still contains bare `\n` (not `\r\n`) and correct UTF-8 — the current
   code fails this.
4. `_materialize_prompt("pi", …)` returns a `prompt_file_path` for a multi-line
   prompt and none for a short single-line one.
5. `_prompt_args` on the shim path with a sidecar present returns
   `["@<path>", "<directive>"]`; without a sidecar it must not silently emit a
   multi-line argv token.
6. Prompt starting with `/`, `@` or `-` is guarded before argv.
7. Existing pi tests stay green (model tiers, `-e` extensions, `--mcp-config`,
   `--continue` without `--session-id`).

## Risks

- `--exclude-tools` is an allowlist-by-omission for *custom* tools too; a future
  win-agent-teams pi extension must not reuse a denied name.
- The newline fix in `_write_tab_wrapper` touches all three backends. Mitigate
  with a wrapper-fidelity test and a Linux + Windows suite run before PR.
- `--append-system-prompt` adds tokens to every pi turn; keep it to ~3 lines.

## Validation

Per CLAUDE.md, all four gates on the whole repo, on Windows **and** the Lubuntu
VM, plus a live smoke: spawn a pi agent with a 10 KB multi-line prompt
containing `—`, `åäö`, a code fence and a `;`, and assert the pi rollout's first
user message matches the input byte-for-byte (modulo the correlation marker).

## Workflow

Steps 2–5 of CLAUDE.md still apply: this plan goes to an opposite-family
reviewer (Codex/GPT) before implementation.

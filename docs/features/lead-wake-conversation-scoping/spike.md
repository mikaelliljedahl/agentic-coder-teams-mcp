# Spike: process ancestry as a conversation key (Windows)

Ran 2026-08-10 on Windows 11, Claude Desktop 1.26832.0.0 + claude-code
2.1.222, codex-cli 0.147.0. Probe script: an ancestor walk via
`CreateToolhelp32Snapshot` (ctypes), executed from a tool-invoked shell —
the same launch shape Claude Code uses for hook commands.

## What was measured

1. Ancestor chain of a process launched the way a hook is launched.
2. The real process tree of every live `claude.exe` / `server_simple` /
   `lead_wake` / `cli watch` process on the machine.

## Findings

### F1 — the ctypes ancestor walk works

Observed chain from a tool-launched python:

```
L0 python.exe 50628
L1 pwsh.exe   50272
L2 claude.exe 25992     <- conversation process
L3 claude.exe 17548     <- Electron app shell
L4 explorer.exe 12028
```

So a hook can reach its conversation process, but **not** via
`os.getppid()` — a shell layer sits in between.

### F2 — each conversation IS its own process (the design's premise holds)

Under the single Electron shell `17548` there are many sibling
`claude.exe … --output-format stream-json` processes (25992, 42040, 23912,
18352, 41720, 44704, …) — one per conversation. Conversations in the same
folder are therefore distinguishable by PID. This is the fact the whole
feature rests on, and it is confirmed.

### F3 — `os.getppid()` on the SERVER side is also wrong (new)

`server_simple` is launched through the repo `.venv` python, which re-execs
into the uv-managed interpreter, producing a two-level python chain:

```
15676 .venv\python.exe -m claude_teams.server_s…   (parent: 1988)
48660   uv cpython -m claude_teams.serv…           (child of 15676)
```

The same doubling appears for `lead_wake` (25272 → 42192) and `cli watch`.
So the process actually running our code may be a grandchild of the host.
`os.getppid()` would record the shim, not the conversation.

### F4 — the MCP host is not always `claude.exe` (new)

Live `server_simple` parents included **`codex.exe`** (1988, 29960, 48104)
as well as `claude.exe` (32888, 41880 — spawned CLI agents). A field named
`claude_pid` is wrong; the concept is *host process*.

### F5 — hook chains can be four levels deep

A `lead_wake` instance was observed under `bash.exe` (40548, Git-bash `-c`),
and with the venv/uv doubling that is
`python(uv) → python(shim) → bash → host`. A 5-level cap is too tight for
comfort.

## Consequences for the plan

- Both sides must resolve **the nearest host-process ancestor**, not a raw
  parent: walk ancestors and take the first whose image name is in the host
  set (`claude.exe`, `codex.exe`, and the pi host once it exists).
- A plain chain *intersection* test is unusable: every conversation shares
  the Electron shell and `explorer.exe` as ancestors, so intersection is
  always true. Equality of the *nearest host ancestor* is the correct test.
- Rename the claim field `claude_pid` → `host_pid`, and record
  `host_name` alongside it for diagnosis.
- Raise the walk cap to 8 levels.
- Non-Windows: the same walk over `/proc/<pid>/stat`; the host-name set must
  match POSIX image names (`claude`, `codex`) — verify on the Lubuntu VM
  during implementation.

## Paired capture — the R2-3 release gate (Windows)

Captured live with both shapes measured at once.

**Conversation A (Claude Desktop), server side:**

```
python(36936) -> python(6940) -> win-agent-teams.exe(39364)
              -> claude.exe(6956) -> claude.exe(17548) -> explorer.exe
```

**Conversation A, hook side:**

```
python -> pwsh(51876) -> claude.exe(6956) -> claude.exe(17548) -> explorer.exe
```

Both select **the same nearest Claude host, 6956** — the premise the whole
feature rests on. ✅ (Note the previously unseen `win-agent-teams.exe` console
shim between the server and the host: a third level, which is why the ceiling is
64 and not a small number.)

**Conversation B and others:** `claude.exe(18352)`, `claude.exe(42040)`,
`claude.exe(36584)`, `claude.exe(39504)` — distinct hosts per conversation,
including CLI leads under `powershell -> WindowsTerminal`. ✅

### F6 — a nested Codex agent resolves to its LEAD's Claude host (new, blocking)

```
server(41596): python -> codex.exe(17664) -> python -> python
            -> win-agent-teams.exe(43284) -> claude.exe(18352) -> claude.exe(17548)
```

A Codex agent spawned by a Claude lead runs its own MCP server, and a
**Claude-only** nearest-host search walks straight past `codex.exe` and lands on
the *lead's* conversation. That agent could therefore bind (or match) the lead's
ownership. Fix, now folded into the plan: resolve the nearest host over the
**full** host set (`claude`, `codex`, and the pi host), then require the selected
host to be a Claude one. For 41596 the nearest host is `codex.exe(17664)` → not
Claude → refuse/skip, which is correct.

## Not measured (deferred to implementation)

- Linux behavior (no VM access in this session) — the R2-3 gate is discharged
  for Windows only; the same paired capture must be repeated on the Lubuntu VM.
- Whether the shell layer ever `exec`s away (POSIX `sh -c` often does),
  which only shortens the chain and is therefore harmless.

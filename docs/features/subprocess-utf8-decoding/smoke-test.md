# Smoke test: captured subprocess output decodes without raising

Manual verification for `fix/subprocess-utf8-decoding`. The unit tests prove
the kwargs and guard the class; this proves the real behaviour against a real
process table, a real PowerShell, and a real console code page — the three
things a monkeypatched `subprocess.run` cannot exercise.

Budget: ~5 minutes on Windows, ~5 more on Linux if tmux is available.

## What you are looking for

Not a crash. **The pre-fix failure is silent**, in two ways.

*Mis-decode.* PowerShell writes a redirected stream in the console code page. A
planted argv of `≥→é中` comes back as `=\x1a‚?`, and `\x1a` (SUB) is a control
character, which `json.loads` refuses inside a string. So **one** process with
a non-ASCII command line discards the whole 170 KB table:
`_windows_command_lines` returns `{}` and every ancestry entry gets `argv=()`.

*Undecodable byte.* The reader thread dies, `stdout` comes back `None`, and the
helper's own `.strip()` raises `AttributeError` past a guard written for a
different failure.

Either way, everything built on argv — the node-shim rule that tells a
`node.exe` running Pi apart from one running Claude Code — degrades to a guess,
and nothing in the logs says why.

So every check below asks the same two questions: **did the table come back at
all, and is the argv in it the argv that was really there?** Phase 4 asks a
third: where the decoded text is a *machine protocol*, does an unreadable
answer fail closed instead of passing for a good one?

## Phase 0 — the tree

```bash
cd C:/code/github/win-agent-teams-mcp/wt-subprocess-encoding
git log --oneline -1        # expect the fix commit
```

Gates are the four in CLAUDE.md; CI runs them on Linux. Two results are
**pre-existing and Windows-only** — they fail on `main` too and are not yours:
`test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`, and the
`ty` diagnostic `unresolved-attribute` at `tests/test_join_team.py:730`.

## Phase 1 — Windows: an argv cp1252 cannot represent

This is the whole defect in one script. It starts a child whose argv carries
`≥ → é 中`, none of which survive a cp1252 round-trip, then asks `procinfo` to
read the live process table back.

```bash
cat > /tmp/probe_argv.py <<'PY'
import subprocess, sys, time
from claude_teams import procinfo

MARKER = "win-agent-teams-smoke-\u2265\u2192\u00e9\u4e2d"
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)", MARKER],
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
try:
    time.sleep(2)
    table = procinfo._windows_command_lines()
    argv = table.get(child.pid)
    print("rows        :", len(table))
    print("argv seen   :", argv)
    print("MARKER kept :", bool(argv) and MARKER in argv)
finally:
    child.kill()
PY
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe /tmp/probe_argv.py
```

`PYTHONIOENCODING=utf-8` is for the *print*, not the code under test — without
it your own console raises on `≥` and hides the result.

**PASS**: `rows` is in the hundreds, `argv seen` shows the marker intact, and
`MARKER kept : True`.

**FAIL (the pre-fix behaviour)**: `rows : 0`, `argv seen : None`,
`MARKER kept : False`, with a `UnicodeDecodeError` traceback from
`Thread-N (_readerthread)` printed above it. Note that the call **returned
normally** — it did not raise. That is the point.

To see the pre-fix behaviour for yourself, run the same script against `main`'s
sources without checking anything out:

```bash
PYTHONPATH=C:/code/github/win-agent-teams-mcp/agentic-coder-teams-mcp/src \
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe /tmp/probe_argv.py
```

## Phase 2 — Windows: the degradation the empty table causes

An empty table is not an abstract loss. `procinfo` uses argv to stop the
ancestry walk at the right host: a bare `node.exe` is only recognisable as Pi
or as Claude Code by the script path in its argv.

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import subprocess, sys, time
from claude_teams import procinfo
MARKER = 'win-agent-teams-smoke-\u2265'
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', MARKER],
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
time.sleep(2)
result = procinfo.resolve_nearest_host(child.pid)
print('host      :', result.host)
print('argv empty:', [e.name for e in result.chain if not e.argv])
child.kill()
"
```

**PASS**: the chain's entries carry argv; `argv empty` is short or empty.

**FAIL (pre-fix)**: every entry has `argv=()`, so `host_kind` cannot tell a
node-launched Pi from a node-launched Claude Code, and it says so nowhere.

## Phase 3 — Windows: no BOM, no blank line

The fix prepends `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;` to
the PowerShell command. If that emitted a BOM or a leading newline, `json.loads`
would fail and the helper would fall back to `{}` — the same silent empty table,
for a new reason. Prove the payload is clean:

```bash
.venv/Scripts/python.exe -c "
import inspect, re, subprocess
from claude_teams import procinfo
src = inspect.getsource(procinfo._windows_command_lines)
cmd = re.search(r'\"\[Console\].*?Compress\",', src, re.S).group(0)
print('command contains UTF8 pin:', 'OutputEncoding' in cmd)
"
```

then confirm end-to-end that the JSON parses — phase 1 already does: a
non-zero `rows` is only reachable through `json.loads`.

## Phase 4 — Linux: tmux, and the two protocols that must NOT be lenient

`errors="replace"` is a *don't crash* policy, not a *decode correctly* one. For
text that is only displayed or ignored that is the right trade. For the two
tmux **machine protocols** it is not, and the review round caught both: a
replacement character must never become a valid-looking answer.

### 4a. An unreadable pane status must not prove ownership

`#{pane_dead}` answers exactly `0` or `1`. `_tracked_alive` treats a `True`
here as proof that a pane is *ours*, so anything else has to fail closed. This
one needs no tmux:

```bash
uv run python -c "
from unittest.mock import MagicMock
from claude_teams.backends import process_manager as pm
m = pm.TmuxProcessManager()
pm.subprocess.run = MagicMock(return_value=MagicMock(stdout='?', stderr='', returncode=0))
print(m._pane_alive('%42'))
"
```

**PASS**: `(False, "unreadable tmux pane status: ...")`.

**FAIL (the pre-review behaviour)**: `(True, 'tmux pane running')` — an
unreadable byte just claimed ownership of a pane.

### 4b. A corrupt spawn id must fail the spawn

```bash
uv run python -c "
from claude_teams.backends import process_manager as pm
m = pm.TmuxProcessManager()
for bad in ['@?\t%42\t4242', '@7\t%?\t4242', '@7\t%42\t-1']:
    try:
        print('accepted:', m._parse_tmux_spawn_output(bad))
    except RuntimeError as exc:
        print('refused :', exc)
"
```

**PASS**: all three are refused. **FAIL**: any is accepted — that registers an
agent whose pane can never be health-checked, signalled or killed.

### 4c. And the ordinary path still works

Skip if `tmux` is absent.

```bash
tmux new-session -d -s smoke-utf8 -n "fönster-≥" "sleep 60"
uv run python -c "
from claude_teams.backends import process_manager as pm
print(pm.TmuxProcessManager().list_sessions() if hasattr(pm.TmuxProcessManager, 'list_sessions') else 'inspect manually')
"
tmux kill-session -t smoke-utf8
```

**PASS**: the call returns, and the window name round-trips or is replaced with
`\ufffd` — either is acceptable, because `errors="replace"` is deliberately a
*don't crash* policy here, not a *decode correctly* one. Only `procinfo`'s
PowerShell query got a real encoding contract.

**FAIL**: `UnicodeDecodeError`.

## Phase 5 — the guard holds

```bash
uv run pytest tests/test_subprocess_decoding.py -q
```

**PASS**: all files pass. Then prove the guard actually bites, by running it
against `main`'s sources:

```bash
PYTHONPATH=C:/code/github/win-agent-teams-mcp/agentic-coder-teams-mcp/src \
  uv run pytest tests/test_subprocess_decoding.py -q
```

**PASS**: exactly `procinfo.py`, `process_base.py` and `process_manager.py`
fail. A guard that passes against `main` is not guarding anything.

## Out of scope

`Popen(..., stdin=PIPE, text=True)` in `process_manager.py` encodes what we
*write* to a child with the locale encoding. That is the same root cause on the
write side and the likely source of the em-dash corruption seen in the Pi
messaging work, but changing it changes spawn behaviour. It needs its own branch
and its own smoke test.

## Reporting

If a PASS does not hold, capture: the full script output including any
`Thread-N (_readerthread)` traceback, `python -c "import locale;
print(locale.getencoding())"`, and `$PSVersionTable.PSVersion` from the
PowerShell the helper resolved.

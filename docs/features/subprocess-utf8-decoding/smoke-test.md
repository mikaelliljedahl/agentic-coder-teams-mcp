# Smoke test: captured subprocess output decodes without raising

Manual verification for `fix/subprocess-utf8-decoding`. The unit tests prove
the kwargs and guard the class; this proves the real behaviour against a real
process table, a real PowerShell, and a real console code page — the three
things a monkeypatched `subprocess.run` cannot exercise.

Budget: ~5 minutes on Windows, ~5 more on Linux if tmux is available.

**Run every command in Git Bash**, the Windows phases included. They use `cat`,
`/tmp` and backslash line continuations — they are not PowerShell.

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

**PASS**: `MARKER kept : True`. That single line is the invariant — the exact
marker round-trips.

**FAIL**: anything else. The pre-fix behaviour has more than one shape and you
should not wait for a particular one:

* `rows : 0` and `argv seen : None` — the mis-decode put a control character in
  the JSON and the whole table was discarded. Observed on 2026-08-30.
* `rows` in the hundreds but `MARKER kept : False` — the table survived and the
  argv is mojibake (`≥` shows as `â‰¥`). Equally a failure, and the quieter one.
* a `UnicodeDecodeError` traceback from `Thread-N (_readerthread)` above an
  otherwise normal return — the call did **not** raise; the stream was lost.

Which shape you get depends on the active code page and on what happens to be
running, so judge on `MARKER kept`, not on the row count.

To see the pre-fix behaviour for yourself, run the same script against `main`'s
sources without checking anything out:

```bash
PYTHONPATH=C:/code/github/win-agent-teams-mcp/agentic-coder-teams-mcp/src \
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe /tmp/probe_argv.py
```

## Phase 2 — Windows: a non-ASCII script path survives in argv

This is an argv round-trip probe, not a classification test — it does not
launch node and does not call `host_kind`. What it establishes is the
precondition classification depends on: `procinfo` stops the ancestry walk at
the right host by reading the script path out of a bare `node.exe`'s argv, so
if a non-ASCII path cannot survive that read, the classification downstream
cannot be right either. `test_node_launched_pi_stops_before_outer_claude` and
`test_linux_node_shim_cmdline_identifies_host` cover the classifier itself.

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import subprocess, sys, time
from claude_teams import procinfo
SCRIPT = '/opt/smoke-\u2265/node_modules/@earendil-works/pi-coding-agent/dist/cli.js'
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', SCRIPT],
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
time.sleep(2)
argv = procinfo._windows_command_lines().get(child.pid, ())
print('script argv intact:', SCRIPT in argv)
print('argv              :', argv)
child.kill()
"
```

**PASS**: `script argv intact: True`.

**FAIL**: `False` — whether because the table came back empty or because the
path came back mangled. Both are the same defect, and both leave `host_kind`
without the argv it needs to tell a node-launched Pi from a node-launched
Claude Code. Do **not** accept a merely non-empty argv as a pass: the mangled
case has a non-empty argv, and that is the shape `main` produced here on
2026-08-30.

## Phase 3 — Windows: no BOM, no blank line

The fix prepends `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;` to
the PowerShell command. If that emitted a BOM or a leading newline, `json.loads`
would fail and the helper would fall back to `{}` — the same silent empty table,
for a new reason. Run the real command and look at the real first bytes:

```bash
.venv/Scripts/python.exe -c "
import subprocess
cmd = ('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; '
       'Get-CimInstance Win32_Process | '
       'Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress')
out = subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd],
                     capture_output=True, timeout=30).stdout
print('BOM        :', out.startswith(b'\xef\xbb\xbf'))
print('first byte :', out[:1])
print('starts JSON:', out[:1] in (b'[', b'{'))
print('first bytes:', out[:8])
"
```

**PASS**: `BOM : False` and `starts JSON: True` — the *first* byte, with no
`lstrip()` softening it, because a leading blank line is exactly one of the
things this phase exists to catch.

**FAIL**: a BOM, or a first byte that is neither `[` nor `{`.

This is the only phase that inspects the child's raw output bytes. Phase 1
covers the same ground end to end — a non-zero row count is reachable only
through a successful `json.loads` — but a BOM failure there would be
indistinguishable from any other empty table, which is why this phase is
separate.

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

### 4c. A real pane, through the real decode policy

This one needs tmux; skip it if `tmux` is absent. It drives the production
`_pane_alive` against a pane that really exists, in a window whose name carries
characters the C locale cannot represent — so the decode policy is exercised
for real rather than mocked.

```bash
tmux new-session -d -s smoke-utf8 -n "fönster-≥" "sleep 60"
PANE=$(tmux list-panes -t smoke-utf8 -F '#{pane_id}' | head -1)
uv run python -c "
import sys
from claude_teams.backends import process_manager as pm
m = pm.TmuxProcessManager()
print('live pane :', m._pane_alive(sys.argv[1]))
print('bogus pane:', m._pane_alive('%999999'))
" "$PANE"
tmux kill-session -t smoke-utf8
```

**PASS**: the live pane reports `(True, 'tmux pane running')` and the bogus one
reports `(False, <tmux's own error text>)`.

**FAIL**: `UnicodeDecodeError` from either call, or the bogus pane reporting
`True`. A *live* pane reporting `unreadable tmux pane status` would mean this
tmux build does not answer `#{pane_dead}` with `0`/`1` — report that rather
than working around it, because the strict check in 4a depends on it.

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

"""PowerShell single-quoted literal hardening.

PowerShell's tokenizer treats FIVE characters as a single quote -- ASCII ``'``
plus U+2018, U+2019, U+201A and U+201B -- both as literal delimiters and as the
doubled escape inside one. Quoting that only doubles ASCII ``'`` lets a value
such as ``x<U+2019>; calc`` close the literal early and run as PowerShell.

``_ps_scan_single_quoted`` below models ``ScanStringLiteral`` from
PowerShell's tokenizer (v7.4.6, ``tokenizer.cs``): on any quote char, peek the
next char; if it is also a quote char, consume both and keep the SECOND,
otherwise the literal ends. It is regression evidence that runs on Linux CI,
not a proof; the optional ``pwsh`` parser test and the manual Windows check in
``docs/features/powershell-quote-hardening`` are the independent checks.
"""

from __future__ import annotations

import base64
import itertools
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from claude_teams import server_simple
from claude_teams.backends import process_manager as process_manager_mod
from claude_teams.backends.process_manager import _powershell_quote

PS_QUOTES = "'\u2018\u2019\u201a\u201b"
EXTRA_QUOTES = "\u2018\u2019\u201a\u201b"


def _ps_scan_single_quoted(text: str, start: int) -> tuple[str, int]:
    """Scan a PowerShell single-quoted literal at ``start``.

    Returns ``(value, end)`` where ``end`` is the index just past the closing
    quote. Raises ``ValueError`` if the literal is unterminated.
    """
    if text[start] not in PS_QUOTES:
        raise ValueError(f"no opening quote at {start}")  # noqa: TRY003
    out: list[str] = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch in PS_QUOTES:
            if i + 1 < len(text) and text[i + 1] in PS_QUOTES:
                out.append(text[i + 1])
                i += 2
                continue
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise ValueError("unterminated literal")  # noqa: TRY003


def _scan_whole(literal: str) -> str:
    value, end = _ps_scan_single_quoted(literal, 0)
    assert end == len(literal), f"literal ended early at {end}: {literal!r}"
    return value


def _hostile_corpus() -> list[str]:
    corpus = ["", "plain", "x\u2019; calc; \u2019", "a'\u2018\u2019\u201a\u201bb"]
    for q in PS_QUOTES:
        corpus += [q * n for n in range(1, 5)]
        corpus += [f"{q}start", f"end{q}", f"mid{q}dle", f"{q}; calc; {q}"]
    corpus += ["".join(pair) for pair in itertools.product(PS_QUOTES, repeat=2)]
    corpus += [
        "nul\x00byte",
        "bom\ufeffinside",
        "cr\rlf\ncrlf\r\n",
        "line\n'@\nnext",
        'line\n"@\nnext',
        "$(calc)",
        "`n`t`'",
        "\u201cdouble\u201d \u201elow",
        "C:\\Users\\O\u2019Brien\\proj; Write-Host PWNED",
    ]
    return list(dict.fromkeys(corpus))


# --- _powershell_quote: exact output ---------------------------------------


@pytest.mark.parametrize("quote", list(PS_QUOTES))
def test_each_single_quote_char_is_doubled_with_itself(quote):
    assert _powershell_quote(f"a{quote}b") == f"'a{quote}{quote}b'"


def test_empty_string_is_empty_literal():
    assert _powershell_quote("") == "''"


def test_breakout_payload_exact_literal():
    assert _powershell_quote("x\u2019; calc; \u2019") == (
        "'x\u2019\u2019; calc; \u2019\u2019'"
    )


def test_mixed_quote_run_exact_literal():
    assert _powershell_quote("a'\u2018\u2019\u201a\u201bb") == (
        "'a''\u2018\u2018\u2019\u2019\u201a\u201a\u201b\u201bb'"
    )


def test_double_quote_lookalikes_are_untouched():
    value = "\u201cq\u201d\u201e"
    assert _powershell_quote(value) == f"'{value}'"


# --- _powershell_quote: round-trip through the scanner model ---------------


@pytest.mark.parametrize("value", _hostile_corpus())
def test_quoted_value_scans_back_to_itself(value):
    assert _scan_whole(_powershell_quote(value)) == value


@pytest.mark.parametrize("quote", EXTRA_QUOTES)
def test_old_ascii_only_quoting_was_breakable(quote):
    # Guards the scanner model itself: with ASCII-only doubling the literal
    # must end early on each extra quote char, or the model proves nothing.
    naive = "'" + f"x{quote}; calc".replace("'", "''") + "'"
    _, end = _ps_scan_single_quoted(naive, 0)
    assert end < len(naive)


# --- _write_tab_wrapper -----------------------------------------------------

HOSTILE = "x'\u2018\u2019\u201a\u201b; Write-Host PWNED; \u2019y"


def _expect_statement(text: str, prefix: str, suffix: str) -> str:
    """Find ``prefix``, scan one literal after it, require ``suffix`` next."""
    start = text.index(prefix) + len(prefix)
    value, end = _ps_scan_single_quoted(text, start)
    assert text[end : end + len(suffix)] == suffix, text[end : end + 40]
    return value


def test_wrapper_quotes_every_field_against_all_quote_chars(tmp_path):
    manager = process_manager_mod.WindowsProcessManager()
    sidecar = tmp_path / f"side{HOSTILE}.pid"
    wrapper = tmp_path / "w.launch.ps1"
    cwd = f"C:\\proj{HOSTILE}"
    cmd = [f"C:\\bin{HOSTILE}\\claude.exe", "-C", cwd, "--", f"prompt {HOSTILE}"]

    manager._write_tab_wrapper(wrapper, cwd, cmd, {"AGENT_NAME": HOSTILE}, sidecar)

    text = wrapper.read_bytes().decode("utf-8-sig")
    assert _expect_statement(text, "$env:AGENT_NAME = ", "\r\n") == HOSTILE
    assert _expect_statement(text, "Set-Location -LiteralPath ", "\r\n") == cwd
    assert _expect_statement(
        text, "Out-File -FilePath ", " -Encoding ascii\r\n"
    ) == str(sidecar)
    # The call line: ``& `` then literals separated by single spaces, then EOL.
    pos = text.index("\r\n& ") + len("\r\n& ")
    parsed: list[str] = []
    while True:
        value, pos = _ps_scan_single_quoted(text, pos)
        parsed.append(value)
        if text[pos : pos + 2] == "\r\n":
            break
        assert text[pos] == " "
        pos += 1
    assert parsed == cmd


@pytest.mark.parametrize(
    "key",
    ["", "A;B", "A}", "A:B", "A B", "A\n", "A\r", "A\u2019", "1A", "A-B"],
)
def test_wrapper_rejects_malformed_env_key(tmp_path, key):
    manager = process_manager_mod.WindowsProcessManager()
    wrapper = tmp_path / "w.launch.ps1"

    with pytest.raises(ValueError, match="environment variable name"):
        manager._write_tab_wrapper(
            wrapper, "C:\\proj", ["claude"], {key: "v"}, tmp_path / "w.pid"
        )

    assert not wrapper.exists()


@pytest.mark.parametrize(
    "key", ["AGENT_NAME", "PATH", "WIN_AGENT_TEAMS_SESSION_DIR", "_X1", "a"]
)
def test_wrapper_accepts_well_formed_env_key(tmp_path, key):
    manager = process_manager_mod.WindowsProcessManager()
    wrapper = tmp_path / "w.launch.ps1"

    manager._write_tab_wrapper(
        wrapper, "C:\\proj", ["claude"], {key: "v"}, tmp_path / "w.pid"
    )

    assert f"$env:{key} = 'v'" in wrapper.read_bytes().decode("utf-8-sig")


# --- _watch_command_powershell ---------------------------------------------


def test_watch_command_powershell_quotes_all_quote_chars():
    session_dir = f"C:\\Users\\O{PS_QUOTES}Brien\\s; calc"

    command = server_simple._watch_command_powershell(session_dir, timeout=3)

    assert command.startswith("& ")
    pos = 2
    parsed: list[str] = []
    while True:
        value, pos = _ps_scan_single_quoted(command, pos)
        parsed.append(value)
        if pos == len(command):
            break
        assert command[pos] == " "
        pos += 1
    assert parsed == server_simple._watch_argv(session_dir, 3)


# --- _open_windows_terminal_tail -------------------------------------------


def test_terminal_tail_quotes_log_path(monkeypatch, tmp_path):
    manager = process_manager_mod.WindowsProcessManager()
    popen_mock = MagicMock()
    monkeypatch.delenv("USE_WINDOWS_TERMINAL", raising=False)
    monkeypatch.setattr(
        process_manager_mod.shutil,
        "which",
        lambda name: "C:\\WindowsApps\\wt.exe" if name == "wt.exe" else None,
    )
    monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
    log_path = tmp_path / "O'Br\u2019ien" / "worker.log"

    manager._open_windows_terminal_tail("team", "worker", log_path)

    script = popen_mock.call_args.args[0][-1]
    prefix = "Get-Content -LiteralPath "
    assert script.startswith(prefix)
    value, end = _ps_scan_single_quoted(script, len(prefix))
    assert value == str(log_path)
    assert script[end:] == " -Wait -Tail 80"


# --- optional: real PowerShell parser --------------------------------------

_PWSH = shutil.which("pwsh") or shutil.which("powershell")

# One PowerShell launch parses every source (each Base64 UTF-8) and prints one
# ``OK:<base64 value>`` or ``FAIL:<reason>`` line per source, in order. The
# ``OK:`` prefix keeps an empty value distinguishable from missing output.
_AST_PROBE = r"""
foreach ($b in @({sources})) {{
  $errors = $null; $tokens = $null
  $src = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b))
  $ast = [System.Management.Automation.Language.Parser]::ParseInput(
      $src, [ref]$tokens, [ref]$errors)
  if ($errors.Count -ne 0) {{ 'FAIL:parse'; continue }}
  $strings = @($ast.FindAll({{ param($n)
      $n -is [System.Management.Automation.Language.StringConstantExpressionAst]
  }}, $true))
  $cmds = @($ast.FindAll({{ param($n)
      $n -is [System.Management.Automation.Language.CommandAst]
  }}, $true))
  if ($strings.Count -ne 1 -or $cmds.Count -ne 0) {{ 'FAIL:shape'; continue }}
  'OK:' + [Convert]::ToBase64String(
      [Text.Encoding]::UTF8.GetBytes($strings[0].Value))
}}
"""


@pytest.mark.skipif(_PWSH is None, reason="no pwsh/powershell on PATH")
def test_real_powershell_parser_reads_single_literal():
    # NUL is parser-safe but cannot travel through a command line, so the
    # scanner-model tests above cover it instead.
    values = [v for v in _hostile_corpus() if "\x00" not in v]
    sources = ",".join(
        "'" + base64.b64encode(f"$v = {_powershell_quote(v)}".encode()).decode() + "'"
        for v in values
    )
    probe = _AST_PROBE.format(sources=sources)
    # ``-EncodedCommand`` (Base64 UTF-16LE), not ``-Command -``: fed over stdin,
    # PowerShell buffers multi-line statements like an interactive prompt and
    # silently drops them at EOF, so nothing would be evaluated.
    encoded = base64.b64encode(probe.encode("utf-16-le")).decode("ascii")
    assert _PWSH is not None
    result = subprocess.run(  # noqa: S603 - fixed probe, data passed as Base64.
        [_PWSH, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith(("OK:", "FAIL:"))]
    assert len(lines) == len(values), result.stdout + result.stderr
    for value, line in zip(values, lines, strict=True):
        assert line.startswith("OK:"), f"{value!r}: {line}"
        assert base64.b64decode(line[3:]).decode("utf-8") == value

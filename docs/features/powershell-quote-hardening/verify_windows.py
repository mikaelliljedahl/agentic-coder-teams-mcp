"""Manual Windows verification for PowerShell single-quote hardening.

Run by hand on Windows from the repo root with the project venv active:

    uv run python docs/features/powershell-quote-hardening/verify_windows.py

It is not a pytest test (CI is Linux). For each available shell
(``powershell.exe`` 5.1 and ``pwsh`` 7) it checks, with real PowerShell:

1. Literal round-trip: ``$v = <_powershell_quote(value)>`` parses with no
   errors into exactly one string constant, and ``$v`` equals the value
   (compared as Base64 of UTF-8, computed independently in Python).
2. No execution: break-out payloads try to create a sentinel file named by
   ``$env:PSQ_SENTINEL``; the file must not exist afterwards. A control run
   with the OLD ASCII-only quoting must create it, proving the oracle works.
3. Wrapper argv: a real ``_write_tab_wrapper`` output whose "agent" is a
   PowerShell script stub that writes each received argument as Base64. This
   verifies PowerShell script argument binding, not native-exe argv
   marshalling (outside this patch's guarantee).

Everything lives in a private temp dir that is removed afterwards. Exit code
is 0 only if every check passed on every shell found.
"""

# ruff: noqa: T201 - a manual CLI verifier reports on stdout.

from __future__ import annotations

import base64
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from claude_teams.backends.process_manager import (
    WindowsProcessManager,
    _powershell_quote,
)

QUOTES = [chr(0x27), chr(0x2018), chr(0x2019), chr(0x201A), chr(0x201B)]
BREAKOUT = "; Set-Content -LiteralPath $env:PSQ_SENTINEL -Value 1; "


def b64(text: str) -> str:
    """Return Base64 of the UTF-8 bytes of ``text``."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def corpus() -> list[str]:
    """Return the hostile values every shell must round-trip."""
    values = ["plain", "", "cr\rlf\ncrlf\r\n", "$(calc)", "`n`t", "line\n'@\nx"]
    values += [q * n for q in QUOTES for n in range(1, 5)]
    values += [a + b for a, b in itertools.product(QUOTES, repeat=2)]
    values += [f"x{q}{BREAKOUT}{q}y" for q in QUOTES]
    values.append("x" + "".join(QUOTES) + BREAKOUT + "".join(QUOTES))
    return values


LITERAL_PROBE = """
$errors = $null; $tokens = $null
$src = [IO.File]::ReadAllText({script}, [Text.Encoding]::UTF8)
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $src, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ 'PARSE-ERROR'; exit 2 }}
$n = @($ast.FindAll({{ param($x)
    $x -is [System.Management.Automation.Language.StringConstantExpressionAst]
}}, $true)).Count
if ($n -ne 1) {{ "STRING-COUNT $n"; exit 3 }}
. {script}
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($v))
"""

STUB = (
    "$args | ForEach-Object { "
    "[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes([string]$_)) } "
    "| Set-Content -LiteralPath $env:PSQ_OUT -Encoding ascii\r\n"
)


def run(
    shell: str, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run ``shell`` non-interactively with extra env vars."""
    return subprocess.run(  # noqa: S603 - local manual verifier.
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def check_shell(shell: str, work: Path) -> list[str]:
    """Run every check under ``shell`` and return the failure messages."""
    failures: list[str] = []
    version = run(shell, ["-Command", "$PSVersionTable.PSVersion.ToString()"], {})
    print(f"== {shell}  PSVersion {version.stdout.strip()}")
    sentinel = work / "sentinel dir" / "pwned.txt"
    sentinel.parent.mkdir(exist_ok=True)
    env = {"PSQ_SENTINEL": str(sentinel)}

    # Control: old ASCII-only quoting must be exploitable, else the oracle is blind.
    control = work / "control.ps1"
    naive = "'" + f"x{QUOTES[2]}{BREAKOUT}{QUOTES[2]}y".replace("'", "''") + "'"
    control.write_bytes(f"$v = {naive}\r\n".encode("utf-8-sig"))
    sentinel.unlink(missing_ok=True)
    run(shell, ["-File", str(control)], env)
    if not sentinel.exists():
        failures.append("control: old quoting did NOT execute payload (oracle blind)")
    sentinel.unlink(missing_ok=True)

    for i, value in enumerate(corpus()):
        script = work / f"lit{i}.ps1"
        script.write_bytes(f"$v = {_powershell_quote(value)}\r\n".encode("utf-8-sig"))
        probe = work / f"probe{i}.ps1"
        probe.write_bytes(
            LITERAL_PROBE.format(script=_powershell_quote(str(script))).encode(
                "utf-8-sig"
            )
        )
        result = run(shell, ["-File", str(probe)], env)
        lines = result.stdout.strip().splitlines()
        got = lines[-1] if lines else ""
        if result.returncode != 0 or got != b64(value):
            failures.append(
                f"literal {value!r}: rc={result.returncode} out={result.stdout!r}"
            )
        if sentinel.exists():
            failures.append(f"literal {value!r}: payload EXECUTED")
            sentinel.unlink()

    # Wrapper: hostile cwd, env value, sidecar path and argv through a stub.
    hostile = "x" + "".join(QUOTES) + "; Write-Host PWNED; " + QUOTES[2] + "y"
    cwd = work / f"cwd {hostile}"
    cwd.mkdir(exist_ok=True)
    stub = work / f"stub {QUOTES[1]}.ps1"
    stub.write_bytes(STUB.encode("utf-8-sig"))
    out = work / "argv.txt"
    out.unlink(missing_ok=True)
    argv = [hostile, f"-C {hostile}", "multi\nline " + QUOTES[3]]
    wrapper = work / "w.launch.ps1"
    WindowsProcessManager()._write_tab_wrapper(
        wrapper,
        str(cwd),
        [str(stub), *argv],
        {"PSQ_OUT": str(out), "AGENT_NAME": hostile},
        work / f"side {hostile}.pid",
    )
    result = run(shell, ["-File", str(wrapper)], env)
    received = out.read_text(encoding="ascii").split() if out.exists() else None
    if received != [b64(a) for a in argv]:
        failures.append(f"wrapper argv mismatch: {received!r} stdout={result.stdout!r}")
    if "PWNED" in result.stdout:
        failures.append("wrapper: payload EXECUTED")
    if not (work / f"side {hostile}.pid").exists():
        failures.append("wrapper: sidecar not written at the hostile path")
    return failures


def main() -> int:
    """Verify on each installed Windows PowerShell; exit 0 only if all pass."""
    if os.name != "nt":
        print("Windows only.")
        return 2
    shells = [s for s in (shutil.which("powershell.exe"), shutil.which("pwsh")) if s]
    all_failures: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="psq-verify-"))
    try:
        for shell in shells:
            failures = check_shell(shell, work)
            for f in failures:
                print(f"  FAIL {f}")
            verdict = "PASS" if not failures else "FAIL"
            print(f"  {verdict} ({len(corpus())} literals + wrapper)")
            all_failures += failures
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if shells and not all_failures else 1


if __name__ == "__main__":
    sys.exit(main())

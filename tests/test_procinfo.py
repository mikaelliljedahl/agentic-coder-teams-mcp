"""Tests for bounded, cross-platform process ancestry discovery."""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from claude_teams import procinfo


def _snapshot(*rows: tuple[int, int, str]) -> dict[int, procinfo.ProcessInfo]:
    return {
        pid: procinfo.ProcessInfo(pid=pid, ppid=ppid, name=name)
        for pid, ppid, name in rows
    }


def test_nearest_host_wins_over_outer_electron_host() -> None:
    result = procinfo.resolve_from_snapshot(
        10,
        _snapshot(
            (10, 20, "python.exe"),
            (20, 30, "claude.exe"),
            (30, 40, "claude.exe"),
            (40, 0, "explorer.exe"),
        ),
    )

    assert result.host == procinfo.ProcessInfo(20, 30, "claude.exe")
    assert [entry.pid for entry in result.chain] == [10, 20]


def test_venv_uv_and_shell_chain_resolves_host() -> None:
    result = procinfo.resolve_from_snapshot(
        1,
        _snapshot(
            (1, 2, "python"),
            (2, 3, "python"),
            (3, 4, "bash"),
            (4, 0, "Claude.EXE"),
        ),
    )

    assert result.host is not None
    assert result.host.pid == 4
    assert procinfo.is_claude_host(result.host.name)


@pytest.mark.parametrize(
    ("snapshot", "start"),
    [
        (_snapshot((1, 1, "python")), 1),
        (_snapshot((1, 0, "python")), 1),
    ],
)
def test_cycle_and_orphan_terminate(
    snapshot: dict[int, procinfo.ProcessInfo], start: int
) -> None:
    result = procinfo.resolve_from_snapshot(start, snapshot)

    assert result.host is None
    assert len(result.chain) == 1


def test_no_host_and_ceiling_terminate() -> None:
    ordinary = procinfo.resolve_from_snapshot(
        1, _snapshot((1, 2, "python"), (2, 0, "explorer"))
    )
    long = _snapshot(*[(pid, pid + 1, "python") for pid in range(1, 70)])

    assert ordinary.host is None
    assert procinfo.resolve_from_snapshot(1, long).host is None
    assert len(procinfo.resolve_from_snapshot(1, long).chain) == 64


def test_host_names_are_exact_case_insensitive_and_extension_stripped() -> None:
    assert procinfo.is_claude_host("Claude.EXE")
    assert procinfo.is_host("CODEX.exe")
    assert procinfo.is_host("pi")
    assert not procinfo.is_claude_host("claude-helper")
    assert not procinfo.is_claude_host("codex")
    assert not procinfo.is_claude_host("unknown-wrapper")


def test_full_host_set_stops_at_codex_before_outer_claude() -> None:
    result = procinfo.resolve_from_snapshot(
        1,
        _snapshot(
            (1, 2, "python"),
            (2, 3, "codex.exe"),
            (3, 0, "claude.exe"),
        ),
    )

    assert result.host == procinfo.ProcessInfo(2, 3, "codex.exe")
    assert result.host is not None
    assert not procinfo.is_claude_host(result.host)


def test_node_launched_pi_stops_before_outer_claude() -> None:
    pi = procinfo.ProcessInfo(
        2,
        3,
        "node.exe",
        (
            "C:/Program Files/nodejs/node.exe",
            "C:/npm/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
        ),
    )
    result = procinfo.resolve_from_snapshot(
        1,
        {
            1: procinfo.ProcessInfo(1, 2, "python.exe"),
            2: pi,
            3: procinfo.ProcessInfo(3, 0, "claude.exe"),
        },
    )

    assert result.host == pi
    assert result.host is not None
    assert not procinfo.is_claude_host(result.host)


@pytest.mark.parametrize(
    ("script", "expected_kind"),
    [
        ("/usr/lib/node_modules/@anthropic-ai/claude-code/cli.js", "claude"),
        (
            "/opt/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
            "pi",
        ),
    ],
)
def test_linux_node_shim_cmdline_identifies_host(
    tmp_path: Path, script: str, expected_kind: str
) -> None:
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    (process_dir / "cmdline").write_bytes(
        b"/usr/bin/node\0" + script.encode("utf-8") + b"\0--flag\0"
    )
    (process_dir / "comm").write_text("node\n", encoding="utf-8")
    (process_dir / "status").write_text("PPid:\t42\n", encoding="utf-8")

    entry = procinfo._read_linux_process(123, proc_root=tmp_path)

    assert entry == procinfo.ProcessInfo(
        123, 42, "node", ("/usr/bin/node", script, "--flag")
    )
    assert entry is not None
    assert procinfo.host_kind(entry) == expected_kind


def test_linux_cmdline_handles_argv0_with_spaces_and_parentheses(
    tmp_path: Path,
) -> None:
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    argv0 = "/opt/Odd (worker) name"
    (process_dir / "cmdline").write_bytes(argv0.encode("utf-8") + b"\0")
    (process_dir / "comm").write_text("odd (worker) name\n", encoding="utf-8")
    (process_dir / "status").write_text(
        "Name:\todd (worker) name\nState:\tS (sleeping)\nPPid:\t42\n",
        encoding="utf-8",
    )

    assert procinfo._read_linux_process(123, proc_root=tmp_path) == (
        procinfo.ProcessInfo(123, 42, "Odd (worker) name", (argv0,))
    )


def test_linux_comm_is_fallback_when_cmdline_is_unavailable(tmp_path: Path) -> None:
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    (process_dir / "comm").write_text("fallback name\n", encoding="utf-8")
    (process_dir / "status").write_text("PPid:\t42\n", encoding="utf-8")

    assert procinfo._read_linux_process(123, proc_root=tmp_path) == (
        procinfo.ProcessInfo(123, 42, "fallback name")
    )


def test_toolhelp_snapshot_retries_error_bad_length_once() -> None:
    invalid: int = ctypes.c_void_p(-1).value or 0

    class Kernel32:
        def __init__(self) -> None:
            self.calls = 0

        def CreateToolhelp32Snapshot(  # noqa: N802 - Win32 API test double.
            self, _flags: int, _pid: int
        ) -> int:
            self.calls += 1
            return invalid if self.calls == 1 else 123

    kernel32 = Kernel32()
    errors = iter((procinfo.ERROR_BAD_LENGTH, 0))

    assert (
        procinfo._create_toolhelp_snapshot(
            kernel32, get_last_error=lambda: next(errors)
        )
        == 123
    )
    assert kernel32.calls == 2


def test_real_os_walk_returns_a_plausible_chain() -> None:
    result = procinfo.resolve_nearest_host(os.getpid())

    assert result.chain
    assert result.chain[0].pid == os.getpid()
    assert all(entry.pid > 0 and entry.name for entry in result.chain)


def test_windows_command_lines_asks_the_child_for_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both halves of the encoding contract are present in the call itself.

    This is a call-shape assertion only - it proves nothing about decoding.
    The behaviour is covered by the real-subprocess test below; this one exists
    so that dropping *either* half fails loudly and separately.
    """
    argv: list[str] = []
    options: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        argv.extend(command)
        options.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "[]", "")

    monkeypatch.setattr(procinfo.subprocess, "run", fake_run)

    procinfo._windows_command_lines()

    # The parent decodes as UTF-8...
    assert options["encoding"] == "utf-8"
    assert options["errors"] == "replace"
    # ...and the child is told to emit it, or that decode is a guess.
    assert "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;" in argv[-1]


def _shim(tmp_path: Path, payload: bytes, *, require_utf8_prefix: bool = True) -> Path:
    """A stand-in for powershell.exe that writes ``payload`` as raw bytes."""
    script = tmp_path / "shim.py"
    guard = (
        "if '[Console]::OutputEncoding' not in sys.argv[-1]:\n"
        "    sys.stderr.write('child was never told to emit UTF-8')\n"
        "    raise SystemExit(1)\n"
        if require_utf8_prefix
        else ""
    )
    script.write_text(
        f"import sys\n{guard}sys.stdout.buffer.write({payload!r})\n",
        encoding="utf-8",
    )
    return script


def _run_helper_against_shim(
    monkeypatch: pytest.MonkeyPatch, shim: Path
) -> dict[int, tuple[str, ...]]:
    """Drive ``_windows_command_lines`` with a REAL child, on any platform.

    ``subprocess.run`` stays real - that is the whole point, since the defect
    lives in how subprocess decodes the stream. Only the executable and the
    Windows-only argv splitter are substituted.
    """
    real_run = subprocess.run

    def run_shim(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return real_run([sys.executable, str(shim), *command[1:]], **kwargs)

    monkeypatch.setattr(procinfo.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(procinfo.subprocess, "run", run_shim)
    monkeypatch.setattr(
        procinfo, "_split_windows_command_line", lambda line: tuple(line.split())
    )
    return procinfo._windows_command_lines()


def test_windows_command_lines_decodes_real_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-ASCII argv survives the round trip through a real child process.

    Pre-fix this is where the damage was invisible: the bytes decoded through
    the console code page instead, `≥` became `=\\x1a`, the SUB control
    character made ``json.loads`` reject the whole 170 KB payload, and the
    helper answered "the process table is empty".
    """
    # The UTF-8 bytes of "≥中é", written out so the test does not depend on
    # this source file's own encoding.
    non_ascii = b"\xe2\x89\xa5\xe4\xb8\xad\xc3\xa9"
    payload = (
        b'[{"ProcessId": 7, "CommandLine": "C:/pi.exe --name ' + non_ascii + b'"}]'
    )

    result = _run_helper_against_shim(monkeypatch, _shim(tmp_path, payload))

    assert result[7] == ("C:/pi.exe", "--name", "≥中é")


def test_windows_command_lines_survives_an_undecodable_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A byte that is not valid UTF-8 costs one character, not the whole table.

    Without ``errors="replace"`` the decode fails, and how it fails depends on
    the platform: on Windows the reader thread dies and ``stdout`` comes back
    ``None`` (which the helper now degrades to an empty table); elsewhere the
    ``UnicodeDecodeError`` reaches the caller. Either way the row below is
    lost, which is what this asserts.
    """
    payload = b'[{"ProcessId": 7, "CommandLine": "C:/pi.exe --name \x8f"}]'

    result = _run_helper_against_shim(monkeypatch, _shim(tmp_path, payload))

    assert result[7] == ("C:/pi.exe", "--name", "�")


#: Forces a nested interpreter's default text encoding away from UTF-8, so that
#: dropping ``encoding="utf-8"`` provably changes the result. On Linux the C
#: locale gives ASCII once UTF-8 mode and locale coercion are both off; Windows
#: is already on its own code page. If this ever stops working the test FAILS —
#: see the assertion below for why it must not skip.
_NON_UTF8_LOCALE_ENV = {
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LC_CTYPE": "C",
    "LANG": "C",
}


def test_explicit_encoding_is_what_decodes_utf8_not_the_host_locale(
    tmp_path: Path,
) -> None:
    """Prove the ``encoding="utf-8"`` keyword, not a lucky UTF-8 runner.

    The tests above pass on a UTF-8 Linux runner even without the keyword,
    because the locale default happens to agree. This one re-runs the same
    scenario in a nested interpreter whose default text encoding is not UTF-8.

    The mutation removes ``encoding`` and **nothing else**: ``errors="replace"``
    stays, so the stripped run cannot differ merely by raising. The only
    variable is which decoder subprocess picks, which is exactly the claim.

    A runner where the non-UTF-8 default cannot be forced FAILS rather than
    skips. This is the only behavioural proof of the encoding half of the fix;
    a silent skip would quietly delete it.
    """
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import sys\n"
        'sys.stdout.buffer.write(b\'[{"ProcessId": 7, "CommandLine": '
        '\\"pi \\xe2\\x89\\xa5\\"}]\')\n',
        encoding="utf-8",
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json, locale, subprocess, sys\n"
        "from claude_teams import procinfo\n"
        "strip_encoding = sys.argv[1] == 'strip'\n"
        "real_run = subprocess.run\n"
        "def run_shim(command, **kwargs):\n"
        "    if strip_encoding:\n"
        # ONLY the encoding. errors='replace' stays, so a difference cannot be
        # explained by strict decoding raising instead of choosing a decoder.
        "        kwargs.pop('encoding', None)\n"
        "        kwargs['text'] = True\n"
        f"    return real_run([sys.executable, {str(shim)!r}], **kwargs)\n"
        "procinfo.shutil.which = lambda _n: sys.executable\n"
        "procinfo.subprocess.run = run_shim\n"
        "procinfo._split_windows_command_line = lambda line: tuple(line.split())\n"
        "try:\n"
        "    table = procinfo._windows_command_lines()\n"
        "except UnicodeDecodeError:\n"
        "    table = {}\n"
        "print(json.dumps({'encoding': locale.getencoding(),\n"
        "                  'argv': list(table.get(7, ()))}))\n",
        encoding="utf-8",
    )

    def drive(mode: str) -> dict[str, object]:
        completed = subprocess.run(  # noqa: S603 - fixed argv, our own script.
            [sys.executable, str(driver), mode],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **_NON_UTF8_LOCALE_ENV},
            check=True,
        )
        return json.loads(completed.stdout)

    kept = drive("keep")
    stripped = drive("strip")

    nested_encoding = str(kept["encoding"]).lower().replace("-", "_")
    assert nested_encoding not in {"utf_8", "utf8", "cp65001"}, (
        f"the nested interpreter still defaults to {nested_encoding!r}, so this "
        f"test cannot tell the explicit encoding from the host locale. Fix the "
        f"forcing in _NON_UTF8_LOCALE_ENV rather than skipping: this is the only "
        f"behavioural proof that encoding='utf-8' is what decodes the stream."
    )
    # With the keyword: the real characters. Without it, and with errors=
    # unchanged: the locale decoder's answer, which is not the real characters.
    assert kept["argv"] == ["pi", "≥"]
    assert stripped["argv"] != kept["argv"]

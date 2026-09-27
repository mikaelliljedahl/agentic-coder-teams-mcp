"""Windows Terminal (``wt.exe``) command-line hardening.

The models below follow ``microsoft/terminal`` @ ``8c0a234f``
(``TerminalApp/AppCommandlineArgs.cpp``, ``TerminalApp/Commandline.cpp``,
``TerminalConnection/ConptyConnection.cpp``); see
``docs/features/wt-semicolon-hardening/plan.md``. Stages, kept separate:

1. ``_cmdline_to_argv`` -- wt's own argv (``CommandLineToArgvW`` rules) from
   the command line Python's ``Popen`` builds with ``list2cmdline``.
2. ``_wt_split`` -- ``_addCommandsForArg``: every argv element is split on
   ``^;|[^\\];`` (quotes are already gone), then ``Commandline::AddArg``
   turns each ``\\;`` back into ``;``.
3. ``_wt_rebuild`` -- ``_getNewTerminalArgs``: args after ``--`` are joined
   with spaces, wrapped in ``"..."`` only when they contain a space, with no
   other escaping.
4. ``_expand`` -- ``ExpandEnvironmentStringsW`` over the whole rebuilt child
   command line, before ``CreateProcessW``.
5. ``_crt_argv`` -- the child's own argv parsing (modern CRT / Rust rules).

They are regression evidence on Linux, not proof; the manual Windows verifier
in the plan's feature directory is the real-wt check.
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from claude_teams.backends import process_manager as process_manager_mod
from claude_teams.backends.process_manager import (
    _powershell_quote,
    _wt_argv,
    _wt_child_arg,
    _wt_escape_delimiters,
    _wt_may_expand,
)
from tests.test_backends.test_powershell_quote import _ps_scan_single_quoted

_DELIMITER = re.compile(r"^;|[^\\];")


# --- models -----------------------------------------------------------------


def _parse_windows_args(cmdline: str, *, modern_double_quote: bool) -> list[str]:
    """Split a Windows command line; argv[0] uses the program-name rule."""
    args: list[str] = []
    i, n = 0, len(cmdline)
    # argv[0]: quoted up to the next quote, else up to whitespace; no escapes.
    if i < n and cmdline[i] == '"':
        end = cmdline.find('"', i + 1)
        end = n if end == -1 else end
        args.append(cmdline[i + 1 : end])
        i = end + 1
    else:
        end = i
        while end < n and cmdline[end] not in " \t":
            end += 1
        args.append(cmdline[i:end])
        i = end
    while True:
        while i < n and cmdline[i] in " \t":
            i += 1
        if i >= n:
            return args
        buf: list[str] = []
        in_quotes = False
        while i < n:
            ch = cmdline[i]
            if ch == "\\":
                run = 0
                while i < n and cmdline[i] == "\\":
                    run += 1
                    i += 1
                if i < n and cmdline[i] == '"':
                    buf.append("\\" * (run // 2))
                    if run % 2:
                        buf.append('"')
                        i += 1
                else:
                    buf.append("\\" * run)
                continue
            if ch == '"':
                if in_quotes and i + 1 < n and cmdline[i + 1] == '"':
                    # ``""`` inside quotes: a literal quote. The modern CRT
                    # stays in quote mode; CommandLineToArgvW leaves it.
                    buf.append('"')
                    i += 2
                    in_quotes = modern_double_quote
                    continue
                in_quotes = not in_quotes
                i += 1
                continue
            if ch in " \t" and not in_quotes:
                break
            buf.append(ch)
            i += 1
        args.append("".join(buf))


def _cmdline_to_argv(cmdline: str) -> list[str]:
    return _parse_windows_args(cmdline, modern_double_quote=False)


def _crt_argv(cmdline: str) -> list[str]:
    return _parse_windows_args(cmdline, modern_double_quote=True)


def _add_arg(arg: str) -> str:
    """``Commandline::AddArg``: replace each ``\\;`` with ``;``, left to right."""
    out = arg
    pos = out.find("\\;")
    while pos != -1:
        out = out[:pos] + ";" + out[pos + 2 :]
        pos = out.find("\\;", pos + 1)
    return out


def _wt_split(argv: list[str]) -> list[list[str]]:
    """``BuildCommands``: split every argv element on unescaped ``;``."""
    commands: list[list[str]] = [[]]
    for arg in argv:
        remaining = arg
        match = _DELIMITER.search(remaining)
        while True:
            if match is None:
                commands[-1].append(_add_arg(remaining))
                break
            matched_first = len(match.group(0)) == 1
            cut = match.start() if matched_first else match.start() + 1
            if cut:
                commands[-1].append(_add_arg(remaining[:cut]))
            commands.append(["wt.exe"])
            remaining = remaining[match.end() :]
            if not remaining:
                break
            match = _DELIMITER.search(remaining)
    return commands


def _wt_rebuild(child: list[str]) -> str:
    """``_getNewTerminalArgs``: space-join, quote only args with a space."""
    return " ".join(f'"{a}"' if " " in a else a for a in child)


def _expand(cmdline: str, env: dict[str, str]) -> str:
    """``ExpandEnvironmentStringsW`` for the variables in ``env``."""
    lowered = {k.lower(): v for k, v in env.items()}
    return re.sub(
        r"%([^%]+)%", lambda m: lowered.get(m.group(1).lower(), m.group(0)), cmdline
    )


def _decode(wt_argv: list[str], env: dict[str, str] | None = None) -> list[dict]:
    """Run a Popen argv through every wt stage; one dict per sub-command."""
    argv = _cmdline_to_argv(subprocess.list2cmdline(wt_argv))
    decoded = []
    for command in _wt_split(argv):
        args = command[1:]
        if "--" in args:
            split_at = args.index("--")
            options, child = args[:split_at], args[split_at + 1 :]
            child_line = _expand(_wt_rebuild(child), env or {})
            decoded.append({"options": options, "child": _crt_argv(child_line)})
        else:
            decoded.append({"options": args, "child": None})
    return decoded


def _option(options: list[str], flag: str) -> str:
    return options[options.index(flag) + 1]


# --- corpora ----------------------------------------------------------------

SEMICOLON_VALUES = [
    ";",
    ";;",
    ";start",
    "end;",
    "a;b;c",
    "\\;",
    "\\\\;",
    "\\\\\\;",
    "trailing\\",
    "C:\\dir\\;x",
    "C:\\p;calc.exe",
    "C:\\a b;x",
]

CHILD_VALUES = [
    "plain",
    "",
    "has space",
    "tab\there",
    'quote"inside',
    'quote "and space"',
    '\\"',
    "a b\\",
    "a b\\\\",
    'a\\"b c',
    'a\\\\"b',
    "semi;colon",
    'mixed "q"; next',
    'x;"y z";\\',
    "tab\t;semi",
    "it's \u2019smart\u2019",
    "multi\nline",
    'Fix the "parser"; then run tests\\',
    ";",
    "\\;",
]


# --- delimiter escaping -----------------------------------------------------


@pytest.mark.parametrize("value", SEMICOLON_VALUES)
def test_escaped_value_is_one_wt_arg_and_round_trips(value):
    commands = _wt_split(["wt.exe", _wt_escape_delimiters(value)])
    assert commands == [["wt.exe", value]]


def test_unescaped_semicolon_is_a_new_sub_command():
    # Negative control: the model must see the bug the escape fixes.
    commands = _wt_split(["wt.exe", "-d", "C:\\p;calc.exe", "--", "codex.exe"])
    assert len(commands) == 2
    assert commands[1][:2] == ["wt.exe", "calc.exe"]


# --- child argv encoding ----------------------------------------------------


@pytest.mark.parametrize("value", CHILD_VALUES)
def test_child_arg_round_trips_through_every_wt_stage(value):
    wt = _wt_argv("C:\\wt.exe", ["nt"], ["C:\\bin\\codex.exe", value, "tail"])
    decoded = _decode(wt)
    assert len(decoded) == 1
    assert decoded[0]["child"] == ["C:\\bin\\codex.exe", value, "tail"]


def test_child_arg_combinations_round_trip():
    child = ["C:\\bin\\codex.exe", *CHILD_VALUES, *reversed(CHILD_VALUES)]
    decoded = _decode(_wt_argv("C:\\wt.exe", ["nt"], child))
    assert len(decoded) == 1
    assert decoded[0]["child"] == child


def test_old_semicolon_only_passthrough_corrupts_quoted_prompt():
    # Negative control: escaping only ``;`` lets wt's naive rebuild re-split a
    # prompt that contains ``"``.
    prompt = 'Fix the "parser" bug'
    old = ["C:\\wt.exe", "nt", "--", "C:\\codex.exe", prompt.replace(";", r"\;")]
    assert _decode(old)[0]["child"] != ["C:\\codex.exe", prompt]


def test_child_arg_leaves_plain_tokens_unchanged():
    assert _wt_child_arg("--model") == "--model"
    assert _wt_child_arg("C:\\bin\\codex.exe") == "C:\\bin\\codex.exe"
    assert _wt_child_arg("Implement the parser; run tests") == (
        r"Implement the parser\; run tests"
    )


# --- full argv --------------------------------------------------------------


@pytest.mark.parametrize("cwd", ["C:\\p;calc.exe", "C:\\a b;x", "C:\\p\\;q", "C:\\"])
def test_hostile_starting_directory_stays_one_sub_command(cwd):
    child = ["C:\\codex.exe", "-C", cwd, 'say "hi"; bye']
    decoded = _decode(_wt_argv("C:\\wt.exe", ["nt", "-d", cwd], child))
    assert len(decoded) == 1
    assert _option(decoded[0]["options"], "-d") == cwd
    assert decoded[0]["child"] == child


def test_wt_argv_does_not_encode_the_wt_executable():
    assert _wt_argv("C:\\Program Files\\wt;x\\wt.exe", [], [])[0] == (
        "C:\\Program Files\\wt;x\\wt.exe"
    )


# --- environment expansion (stage 4) ---------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        (["no percent"], False),
        (["50% done"], False),
        (["%A%"], True),
        (["50%", "then 80%"], True),
        (["%", "%"], True),
    ],
)
def test_may_expand_counts_percent_across_all_values(values, expected):
    assert _wt_may_expand(values) is expected


def test_expansion_can_resplit_child_argv():
    # Negative control: a variable whose value holds ``"`` changes argv
    # boundaries even when every stage before it was encoded exactly.
    child = ["C:\\codex.exe", "prompt %WTV%"]
    decoded = _decode(_wt_argv("C:\\wt.exe", ["nt"], child), {"WTV": 'a" "b'})
    assert decoded[0]["child"] != child


def test_expansion_can_break_the_old_tail_literal():
    # Negative control for the old ``-Command`` tail: a ``'`` in an expanded
    # variable ends the PowerShell literal ``_powershell_quote`` produced.
    path = "C:\\logs\\%WTV%\\w.log"
    script = f"Get-Content -LiteralPath {_powershell_quote(path)} -Wait -Tail 80"
    decoded = _decode(
        _wt_argv("C:\\wt.exe", ["nt"], ["powershell", "-Command", script]),
        {"WTV": "x'; calc; '"},
    )
    expanded = decoded[0]["child"][2]
    prefix = "Get-Content -LiteralPath "
    _, end = _ps_scan_single_quoted(expanded, len(prefix))
    assert expanded[end:] != " -Wait -Tail 80"


# --- call sites -------------------------------------------------------------


def _prep(monkeypatch, tmp_path, *, log_dir: Path | None = None, direct=False):
    manager = process_manager_mod.WindowsProcessManager()
    log_dir = log_dir or tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(log_dir))
    monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
    monkeypatch.delenv("WIN_AGENT_TEAMS_NO_WT_TABS", raising=False)
    monkeypatch.setenv("WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS", "0")
    if direct:
        monkeypatch.setenv(process_manager_mod._CODEX_DIRECT_LAUNCH_ENV, "1")
    else:
        monkeypatch.delenv(process_manager_mod._CODEX_DIRECT_LAUNCH_ENV, raising=False)
    monkeypatch.setattr(
        manager,
        "_should_use_interactive_console",
        lambda backend_type, *, is_interactive=False: True,
    )
    monkeypatch.setattr(
        process_manager_mod.shutil,
        "which",
        lambda name: "C:\\wt.exe" if name == "wt.exe" else None,
    )
    popen_mock = MagicMock(return_value=MagicMock(pid=999))
    monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(manager, "_await_tab_pid", lambda sidecar: 4242)
    monkeypatch.setattr(manager, "_await_codex_tab_pid", lambda token: 7777)
    monkeypatch.setattr(manager, "_pid_alive", lambda handle: False)
    monkeypatch.setattr(
        process_manager_mod, "creation_token", lambda handle: f"tok-{handle}"
    )
    return manager, popen_mock, log_dir


def _spawn_codex(manager, make_request, cwd: str, cmd: list[str]):
    return manager.spawn_process(
        make_request(cwd=cwd, extra={"correlation_id": "corr-1"}),
        cmd,
        {"PATH": "x"},
        "codex",
        is_interactive=True,
    )


def test_direct_launch_encodes_cwd_and_prompt(
    _make_spawn_request, monkeypatch, tmp_path
):
    manager, popen_mock, _ = _prep(monkeypatch, tmp_path, direct=True)
    cwd = str(tmp_path / "proj;calc.exe")
    cmd = ["C:\\codex.exe", "-C", cwd, 'Fix the "parser"; run tests wat-corr:corr-1']

    _spawn_codex(manager, _make_spawn_request, cwd, cmd)

    decoded = _decode(popen_mock.call_args.args[0])
    assert len(decoded) == 1
    assert _option(decoded[0]["options"], "-d") == cwd
    assert decoded[0]["child"] == cmd


def _assert_wrapper_launch(popen_mock, log_dir: Path, prompt: str) -> None:
    wt_cmd = popen_mock.call_args.args[0]
    decoded = _decode(wt_cmd)
    assert len(decoded) == 1
    child = decoded[0]["child"]
    assert child[0] == "powershell"
    assert child[-2] == "-File"
    wrapper = Path(child[-1])
    assert wrapper.parent == log_dir / "team"
    assert prompt in wrapper.read_bytes().decode("utf-8-sig")
    assert "-d" not in decoded[0]["options"]


@pytest.mark.parametrize(
    ("cwd_name", "prompt"),
    [
        ("proj", "use %USERPROFILE% here wat-corr:corr-1"),
        ("proj%A%", "plain prompt wat-corr:corr-1"),
        ("proj", "50% then 80% wat-corr:corr-1"),
    ],
)
def test_direct_launch_with_percent_pair_uses_wrapper_tab(
    _make_spawn_request, monkeypatch, tmp_path, cwd_name, prompt
):
    manager, popen_mock, log_dir = _prep(monkeypatch, tmp_path, direct=True)
    cwd = str(tmp_path / cwd_name)

    _spawn_codex(
        manager, _make_spawn_request, cwd, ["C:\\codex.exe", "-C", cwd, prompt]
    )

    _assert_wrapper_launch(popen_mock, log_dir, prompt)
    log = (log_dir / "team" / "worker.log").read_text(encoding="utf-8")
    assert "[wt direct launch skipped]" in log


def test_direct_launch_with_batch_shim_uses_wrapper_tab(
    _make_spawn_request, monkeypatch, tmp_path
):
    manager, popen_mock, log_dir = _prep(monkeypatch, tmp_path, direct=True)
    cwd = str(tmp_path)
    prompt = "p wat-corr:corr-1"

    _spawn_codex(
        manager, _make_spawn_request, cwd, ["C:\\npm\\codex.CMD", "-C", cwd, prompt]
    )

    _assert_wrapper_launch(popen_mock, log_dir, prompt)


def test_wrapper_path_with_semicolon_and_space_is_exact(
    _make_spawn_request, monkeypatch, tmp_path
):
    manager, popen_mock, log_dir = _prep(
        monkeypatch, tmp_path, log_dir=tmp_path / "lo g;calc.exe"
    )

    manager.spawn_process(
        _make_spawn_request(),
        ["claude", "--", "do stuff"],
        {},
        "claude-code",
        is_interactive=True,
    )

    decoded = _decode(popen_mock.call_args.args[0])
    assert len(decoded) == 1
    assert decoded[0]["child"][-1] == str(log_dir / "team" / "worker.launch.ps1")


def test_wrapper_path_with_percent_pair_refuses_before_writing(
    _make_spawn_request, monkeypatch, tmp_path
):
    manager, popen_mock, _ = _prep(monkeypatch, tmp_path, log_dir=tmp_path / "l%A%%B%")
    request = _make_spawn_request()
    log_path = manager.log_path(request.team_name, request.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = log_path.with_name("worker.pid")
    sidecar.write_text("stale", encoding="utf-8")

    with (
        log_path.open("a", encoding="utf-8") as log_handle,
        pytest.raises(process_manager_mod.WindowsTerminalTabUnsafeCommandLineError),
    ):
        manager._spawn_in_terminal_tab(
            request,
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            log_path,
            log_handle,
            wt="C:\\wt.exe",
            creationflags=0,
        )

    popen_mock.assert_not_called()
    assert not log_path.with_name("worker.launch.ps1").exists()
    assert sidecar.read_text(encoding="utf-8") == "stale"


def test_percent_pair_in_direct_and_wrapper_falls_back_to_classic_console(
    _make_spawn_request, monkeypatch, tmp_path
):
    # Composition: direct launch is skipped (``%`` pair across two args), and
    # the wrapper path it falls back to has a ``%`` pair too -> one classic
    # console launch, no wt.
    manager, popen_mock, log_dir = _prep(
        monkeypatch, tmp_path, log_dir=tmp_path / "l%A%", direct=True
    )
    cwd = str(tmp_path)
    cmd = ["C:\\codex.exe", "-C", cwd, "50%", "80% wat-corr:corr-1"]

    _spawn_codex(manager, _make_spawn_request, cwd, cmd)

    assert popen_mock.call_count == 1
    launched = popen_mock.call_args.args[0]
    assert launched[0] != "C:\\wt.exe"
    assert launched == cmd
    assert not (log_dir / "team" / "worker.launch.ps1").exists()
    log = (log_dir / "team" / "worker.log").read_text(encoding="utf-8")
    assert "[wt direct launch skipped]" in log
    assert "[fallback] launching agent in a new console window" in log


def test_tail_puts_no_log_path_on_the_command_line(monkeypatch, tmp_path):
    manager = process_manager_mod.WindowsProcessManager()
    popen_mock = MagicMock()
    monkeypatch.delenv("USE_WINDOWS_TERMINAL", raising=False)
    monkeypatch.setattr(
        process_manager_mod.shutil,
        "which",
        lambda name: "C:\\wt.exe" if name == "wt.exe" else None,
    )
    monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
    log_path = tmp_path / "t;a%USERNAME%il'\u2019s" / "worker.log"

    manager._open_windows_terminal_tail("team", "worker", log_path)

    wt_cmd = popen_mock.call_args.args[0]
    decoded = _decode(wt_cmd, {"USERNAME": "x'; calc; '"})
    assert len(decoded) == 1
    child = decoded[0]["child"]
    assert child[:3] == ["powershell", "-NoExit", "-EncodedCommand"]
    payload = child[3]
    assert re.fullmatch(r"[A-Za-z0-9+/=]+", payload)
    script = base64.b64decode(payload).decode("utf-16-le")
    assert script == (
        f"Get-Content -LiteralPath {_powershell_quote(str(log_path))} -Wait -Tail 80"
    )

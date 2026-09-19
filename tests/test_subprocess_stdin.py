"""Every ``subprocess.run`` says what its child's stdin is.

This package runs as a stdio MCP server: the process' own stdin is the
JSON-RPC pipe the host keeps open. A child started without ``stdin=`` inherits
that pipe. On Windows ``git`` then blocks until the call's timeout (found with
``install_lead_wake``: a 10 s stall of that one call - it runs in a
``run_blocking`` worker thread, so the event loop kept going - and a bogus
``git_ignore: "failed"``), and any child that actually reads would steal
protocol bytes. None of it shows up in unit tests, an in-memory client, or on
Linux - only under a real stdio server - so the call sites have to be pinned
statically.

**Scope, stated plainly.** Like ``test_subprocess_decoding.py`` this inspects
calls spelled literally ``subprocess.run``. It demands one of exactly two
shapes, because only those are *statically* provable detachment:

* ``stdin=subprocess.DEVNULL`` spelled literally, or
* ``input=<anything but a literal None>`` - subprocess then opens the pipe and
  writes it, while ``input=None`` means it does not and stdin is inherited.

Everything else fails, including shapes that may well be correct at runtime:
``stdin=handle``, ``stdin=sys.stdin``, ``stdin=0``, a conditional expression,
and a ``**kwargs`` splat. That is deliberate. A guard that accepts "not the
literal ``None``" accepts ``stdin=0`` (that *is* fd 0) and
``stdin=PIPE if flag else None``, i.e. it stops enforcing the thing it is named
after. If a call ever needs one of those, widen this allowlist to that exact
shape and say why, rather than loosening the rule.

``subprocess.Popen`` is not covered. Its five sites each choose stdin
deliberately, and two of them deliberately inherit: the Windows
``CREATE_NEW_CONSOLE`` agent launch and the Linux terminal-emulator launcher
pass ``stdin=None`` so the interactive CLI gets a real console. Forcing
``DEVNULL`` there would break that contract. Those are pinned by their own
tests in ``tests/test_backends/test_base_runtime.py``, as is the one Popen this
change did detach (the Windows Terminal log tail).
"""

import ast
from pathlib import Path

import pytest

import claude_teams

PACKAGE_ROOT = Path(claude_teams.__file__).parent


def _is_subprocess_run(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "run"
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def _is_literal_none(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def _is_subprocess_devnull(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "DEVNULL"
        and isinstance(value.value, ast.Name)
        and value.value.id == "subprocess"
    )


def _detaches_stdin(node: ast.Call) -> bool:
    """True only for the two shapes that *prove* the child's stdin statically.

    ``stdin=subprocess.DEVNULL`` is the detachment itself. ``input=`` makes
    subprocess open a pipe and write it - except ``input=None``, which is the
    default and leaves stdin inherited. Anything else (a name, an attribute, a
    conditional, a ``**kwargs`` splat) is not evidence and fails; see the
    module docstring for why the rule is deliberately this narrow.
    """
    for keyword in node.keywords:
        if keyword.arg == "input" and not _is_literal_none(keyword.value):
            return True
        if keyword.arg == "stdin" and _is_subprocess_devnull(keyword.value):
            return True
    return False


def _stdin_inheriting_run_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_subprocess_run(node)
        and not _detaches_stdin(node)
    ]


@pytest.mark.parametrize(
    "path", sorted(PACKAGE_ROOT.rglob("*.py")), ids=lambda p: p.name
)
def test_every_subprocess_run_names_stdin(path: Path) -> None:
    offenders = _stdin_inheriting_run_calls(path)

    assert not offenders, (
        f"{path.name} starts a child that inherits the MCP server's stdin at "
        f"line(s) {offenders}. Pass stdin=subprocess.DEVNULL (or input=...) so "
        f"the child can neither block on nor read the JSON-RPC pipe."
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("subprocess.run(a)", id="bare"),
        pytest.param(
            "subprocess.run(a, capture_output=True, text=True, errors='replace')",
            id="captured-but-stdin-inherited",
        ),
        # subprocess' own spelling for "inherit mine" - the keyword is present
        # but the child still gets the JSON-RPC pipe.
        pytest.param("subprocess.run(a, stdin=None)", id="stdin-none"),
        # input=None is the default: no pipe is opened, stdin stays inherited.
        pytest.param("subprocess.run(a, input=None)", id="input-none"),
        # 0 and False *are* fd 0 - the parent's stdin, not a detachment.
        pytest.param("subprocess.run(a, stdin=0)", id="stdin-fd-zero"),
        pytest.param("subprocess.run(a, stdin=False)", id="stdin-false"),
        # Explicitly handing over the protocol pipe.
        pytest.param("subprocess.run(a, stdin=sys.stdin)", id="stdin-sys-stdin"),
        # One branch inherits, so the expression proves nothing.
        pytest.param(
            "subprocess.run(a, stdin=subprocess.PIPE if flag else None)",
            id="stdin-conditional",
        ),
        # A splat may or may not carry stdin; it is not evidence either way.
        pytest.param("subprocess.run(a, **kw)", id="kwargs-splat"),
        pytest.param(
            "subprocess.run(a, **{'stdin': subprocess.DEVNULL})",
            id="kwargs-splat-literal-dict",
        ),
    ],
)
def test_guard_catches_the_shapes_it_claims_to(tmp_path: Path, source: str) -> None:
    module = tmp_path / "sample.py"
    module.write_text(f"import subprocess\n{source}\n", encoding="utf-8")

    assert _stdin_inheriting_run_calls(module) == [2]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("subprocess.run(a, stdin=subprocess.DEVNULL)", id="devnull"),
        pytest.param("subprocess.run(a, input='x', text=True)", id="input"),
        pytest.param("subprocess.run(a, input=payload)", id="input-name"),
        pytest.param("other.run(a)", id="not-subprocess"),
    ],
)
def test_guard_does_not_cry_wolf(tmp_path: Path, source: str) -> None:
    module = tmp_path / "sample.py"
    module.write_text(f"import subprocess\n{source}\n", encoding="utf-8")

    assert _stdin_inheriting_run_calls(module) == []

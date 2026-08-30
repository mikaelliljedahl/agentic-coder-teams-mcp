"""Every captured subprocess stream must decode without raising.

``text=True`` decodes a child's output with the locale encoding — cp1252 on a
Swedish Windows — and an undecodable byte raises ``UnicodeDecodeError`` from
subprocess' reader thread. That exception surfaces from the ``subprocess.run``
call itself, so the ``(OSError, SubprocessError)`` guards these call sites use
do not catch it: one non-ASCII byte in a process' command line took down a
whole ``deliver_pending`` call. A captured stream therefore has to name an
``errors`` policy.
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


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in node.keywords if kw.arg == name), None)


def _is_true(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def _captures_output(node: ast.Call) -> bool:
    if _is_true(_keyword(node, "capture_output")):
        return True
    stdout = _keyword(node, "stdout")
    return (
        isinstance(stdout, ast.Attribute)
        and stdout.attr == "PIPE"
        and isinstance(stdout.value, ast.Name)
        and stdout.value.id == "subprocess"
    )


def _decodes_text(node: ast.Call) -> bool:
    return (
        _is_true(_keyword(node, "text"))
        or _is_true(_keyword(node, "universal_newlines"))
        or _keyword(node, "encoding") is not None
    )


def _decoding_run_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_subprocess_run(node)
        and _captures_output(node)
        and _decodes_text(node)
        and _keyword(node, "errors") is None
    ]


@pytest.mark.parametrize(
    "path", sorted(PACKAGE_ROOT.rglob("*.py")), ids=lambda p: p.name
)
def test_captured_subprocess_output_names_an_errors_policy(path: Path) -> None:
    offenders = _decoding_run_calls(path)

    assert not offenders, (
        f"{path.name} decodes captured output without errors= at line(s) "
        f"{offenders}; add errors='replace' so an undecodable byte cannot "
        f"raise UnicodeDecodeError out of subprocess.run."
    )

"""Every ``subprocess.run`` that decodes a captured stream names an errors policy.

``text=True`` decodes a child's output with the *locale* encoding - cp1252 on a
Swedish Windows - and an undecodable byte kills subprocess' reader thread. What
the caller then sees is not a clean exception: ``completed.stdout`` comes back
``None``, and the next ``.strip()`` on it raises ``AttributeError`` from inside
a helper whose ``except (OSError, subprocess.SubprocessError)`` guard was
written for a different failure. So a captured, decoded stream has to name an
``errors`` policy.

**Scope, stated plainly.** This guard inspects calls spelled literally
``subprocess.run``, and nothing else. It deliberately does NOT cover:

* ``subprocess.Popen``. No Popen in this package decodes a captured stream
  today - their outputs are inherited, discarded, or redirected straight to a
  file. The ``stdin=PIPE, text=True`` Popen sites are a *write*-encoding
  concern: same root cause, different risk, separate change.
* an aliased or ``from``-imported ``run``/``PIPE``.

Those are accepted false negatives. This is a ratchet against the obvious
regression, not a proof. Nor is ``errors="replace"`` automatically the right
answer: for a machine protocol such as tmux's ``#{pane_dead}`` the right answer
is an explicit policy *plus* validation of the decoded value - see
``TmuxProcessManager._pane_alive``.
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


def _not_false(value: ast.expr | None) -> bool:
    """True unless the keyword is absent or a literal ``False``.

    A non-literal (``text=some_flag``) counts as a candidate rather than being
    waved through, so refactoring a literal into a variable cannot silently
    escape the guard.
    """
    if value is None:
        return False
    return not (isinstance(value, ast.Constant) and value.value is False)


def _is_pipe(value: ast.expr | None) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "PIPE"
        and isinstance(value.value, ast.Name)
        and value.value.id == "subprocess"
    )


def _captures_output(node: ast.Call) -> bool:
    return (
        _not_false(_keyword(node, "capture_output"))
        or _is_pipe(_keyword(node, "stdout"))
        or _is_pipe(_keyword(node, "stderr"))
    )


def _decodes_text(node: ast.Call) -> bool:
    return (
        _not_false(_keyword(node, "text"))
        or _not_false(_keyword(node, "universal_newlines"))
        or _keyword(node, "encoding") is not None
    )


def _undefended_run_calls(path: Path) -> list[int]:
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
def test_captured_subprocess_run_names_an_errors_policy(path: Path) -> None:
    offenders = _undefended_run_calls(path)

    assert not offenders, (
        f"{path.name} decodes a captured stream without an errors= policy at "
        f"line(s) {offenders}. Choose one deliberately: errors='replace' where "
        f"the text is only displayed or ignored, and an explicit policy plus "
        f"validation of the decoded value where it is a machine protocol."
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "subprocess.run(a, stderr=subprocess.PIPE, text=True)",
            id="stderr-only-pipe",
        ),
        pytest.param(
            "subprocess.run(a, capture_output=flag, text=True)",
            id="non-literal-capture_output",
        ),
        pytest.param(
            "subprocess.run(a, capture_output=True, text=flag)",
            id="non-literal-text",
        ),
        pytest.param(
            "subprocess.run(a, capture_output=True, encoding='utf-8')",
            id="encoding-without-errors",
        ),
    ],
)
def test_guard_catches_the_shapes_it_claims_to(tmp_path: Path, source: str) -> None:
    """A guard that only recognises the spelling we happened to use guards nothing."""
    module = tmp_path / "sample.py"
    module.write_text(f"import subprocess\n{source}\n", encoding="utf-8")

    assert _undefended_run_calls(module) == [2]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "subprocess.run(a, capture_output=True, text=True, errors='replace')",
            id="policy-named",
        ),
        pytest.param("subprocess.run(a, capture_output=True)", id="bytes-not-decoded"),
        pytest.param("subprocess.run(a, text=True)", id="not-captured"),
    ],
)
def test_guard_does_not_cry_wolf(tmp_path: Path, source: str) -> None:
    module = tmp_path / "sample.py"
    module.write_text(f"import subprocess\n{source}\n", encoding="utf-8")

    assert _undefended_run_calls(module) == []

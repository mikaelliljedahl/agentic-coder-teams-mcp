"""Shared fixtures for backend unit tests.

The ``_which_on_path`` autouse fixture replaces the ~125 per-test
``@patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/X")``
decorators that used to live in every backend test file. Having one central
default means:

* Adding a new backend test file doesn't require repeating the pattern.
* Tests that need the binary to be *missing* opt in explicitly by overriding
  the fixture's patch — the intent is then visible in the test body, not
  buried in a decorator stack.
* The global-but-overridable default mirrors the real PATH model: binaries
  "exist" unless a test says otherwise.
"""

import pytest


@pytest.fixture(autouse=True)
def _which_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``shutil.which`` report every binary as present at ``/usr/bin/<name>``.

    This is the default expectation for BaseBackend tests: ``discover_binary``
    should succeed and return a usable path. Tests that need the unavailable
    branch (``shutil.which -> None``) must override this fixture explicitly,
    e.g. via a nested ``monkeypatch.setattr(...)`` or ``unittest.mock.patch``
    context — ``@patch`` decorators and context managers win over the autouse
    because they activate inside the test function, after fixture setup.
    """
    monkeypatch.setattr(
        "claude_teams.backends.base.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

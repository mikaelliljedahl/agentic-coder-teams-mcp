from pathlib import Path

import pytest

pytest_plugins = ["tests._server_support", "tests.test_backends._base_support"]


@pytest.fixture
def tmp_claude_dir(tmp_path: Path) -> Path:
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    return tmp_path

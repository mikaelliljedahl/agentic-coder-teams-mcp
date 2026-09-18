"""Project-scope wake hooks live in ``.claude/settings.local.json``.

The wake ``Stop`` group bakes machine-specific absolute paths and (for the
lead) one process's PID/token, so it must never land in the checked-in
``.claude/settings.json``. These tests pin the new location, the migration of
a group left behind in the legacy file, and the best-effort git exclusion.
"""

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import hooks, procinfo
from claude_teams import server_simple as ss

MEMBER = "qa-member"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Relocate session base + home + cwd off real state."""
    base = tmp_path / "sessions"
    base.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(ss, "_SESSION_BASE", base)
    monkeypatch.setattr(ss, "_AGENT_SESSION_ID", "")
    monkeypatch.setattr(ss, "IDENTITY", ss.ROOT_LEAD_NAME)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    sid = str(uuid.uuid4())
    session = base / sid
    session.mkdir()
    (session / "agents.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(ss, "_session_id", sid)
    host = procinfo.ProcessInfo(101, 1, "claude.exe")
    monkeypatch.setattr(
        ss.procinfo,
        "resolve_nearest_host",
        lambda: procinfo.HostResolution(chain=(host,), host=host),
    )
    monkeypatch.setattr(
        ss.process_manager_module, "creation_token", lambda _pid: "token-a"
    )
    return SimpleNamespace(
        base=base,
        home=home,
        work=work,
        sid=sid,
        session=session,
        local=work / ".claude" / "settings.local.json",
        legacy=work / ".claude" / "settings.json",
    )


def _group(command: str) -> dict:
    return {"hooks": [{"type": "command", "command": command}]}


def _foreign_lead_group(session_dir: Path) -> dict:
    return hooks._wake_hook_matcher(
        session_dir,
        "team-lead",
        owner_mode="bound",
        owner_host_pid=19412,
        owner_host_token="foreign",
    )


def _commands(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return [
        h["command"]
        for g in config.get("hooks", {}).get("Stop", [])
        for h in g["hooks"]
    ]


def _write_legacy(path: Path, groups: list[dict], **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**extra, "hooks": {"Stop": groups}}, indent=2), encoding="utf-8"
    )


class TestLeadWakeLocation:
    def test_project_install_writes_local_file_only(
        self, _isolated: SimpleNamespace
    ) -> None:
        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert Path(result["path"]) == _isolated.local
        assert any("claude_teams.lead_wake" in c for c in _commands(_isolated.local))
        assert not _isolated.legacy.exists()
        assert "migrated_from" not in result

    def test_user_scope_leaves_both_project_files_untouched(
        self, _isolated: SimpleNamespace
    ) -> None:
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])
        before = _isolated.legacy.read_bytes()

        result = asyncio.run(ss.install_lead_wake(scope="user"))

        assert Path(result["path"]) == _isolated.home / ".claude" / "settings.json"
        assert _isolated.legacy.read_bytes() == before
        assert not _isolated.local.exists()
        assert "migrated_from" not in result


class TestLeadWakeMigration:
    def test_install_strips_legacy_group_and_keeps_unrelated_content(
        self, _isolated: SimpleNamespace
    ) -> None:
        _write_legacy(
            _isolated.legacy,
            [_group("custom-stop"), _foreign_lead_group(_isolated.session)],
            custom={"keep": True},
        )

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert Path(result["migrated_from"]) == _isolated.legacy
        assert _commands(_isolated.legacy) == ["custom-stop"]
        legacy = json.loads(_isolated.legacy.read_text(encoding="utf-8"))
        assert legacy["custom"] == {"keep": True}
        wake = [c for c in _commands(_isolated.local) if "lead_wake" in c]
        assert len(wake) == 1
        assert "token-a" in wake[0]

    def test_legacy_without_wake_group_stays_byte_identical(
        self, _isolated: SimpleNamespace
    ) -> None:
        _isolated.legacy.parent.mkdir()
        original = b'{"custom":  true, "hooks": {"Stop": []}}\r\n'
        _isolated.legacy.write_bytes(original)

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert "migrated_from" not in result
        assert _isolated.legacy.read_bytes() == original

    def test_corrupt_legacy_file_is_left_alone(
        self, _isolated: SimpleNamespace
    ) -> None:
        _isolated.legacy.parent.mkdir()
        _isolated.legacy.write_bytes(b"{not json")

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert _isolated.legacy.read_bytes() == b"{not json"

    def test_remove_strips_group_from_both_files(
        self, _isolated: SimpleNamespace
    ) -> None:
        asyncio.run(ss.install_lead_wake())
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])

        result = asyncio.run(ss.install_lead_wake(remove=True))

        assert result["action"] == "removed"
        assert Path(result["migrated_from"]) == _isolated.legacy
        assert _commands(_isolated.local) == []
        assert _commands(_isolated.legacy) == []

    def test_refusal_touches_neither_file(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])
        before = _isolated.legacy.read_bytes()
        monkeypatch.setattr(ss, "_session_id", "")
        monkeypatch.setattr(ss, "_active_session_id", lambda **_kw: "")

        result = asyncio.run(ss.install_lead_wake())

        assert result["reason"] == "no_active_session"
        assert _isolated.legacy.read_bytes() == before
        assert not _isolated.local.exists()

    def test_local_write_failure_leaves_legacy_bytes(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])
        before = _isolated.legacy.read_bytes()
        original_replace = Path.replace

        def _fail(path: Path, target: Path) -> Path:
            if target == _isolated.local:
                raise OSError("denied")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", _fail)

        result = asyncio.run(ss.install_lead_wake())

        assert result == {"success": False, "reason": "settings_write_failed"}
        assert _isolated.legacy.read_bytes() == before

    def test_legacy_cleanup_failure_is_reported_and_retry_repairs(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])
        before = _isolated.legacy.read_bytes()
        original_replace = Path.replace

        def _fail(path: Path, target: Path) -> Path:
            if target == _isolated.legacy:
                raise OSError("read-only")
            return original_replace(path, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", _fail)
            failed = asyncio.run(ss.install_lead_wake())

        assert failed["success"] is True
        assert failed["legacy_cleanup"] == "failed"
        assert Path(failed["legacy_path"]) == _isolated.legacy
        assert "migrated_from" not in failed
        assert _isolated.legacy.read_bytes() == before
        assert list(_isolated.legacy.parent.glob("*.tmp")) == []

        retried = asyncio.run(ss.install_lead_wake())

        assert Path(retried["migrated_from"]) == _isolated.legacy
        assert _commands(_isolated.legacy) == []
        wake = [c for c in _commands(_isolated.local) if "lead_wake" in c]
        assert len(wake) == 1


class TestMemberWakeProjectScope:
    def test_install_writes_local_and_migrates_only_member_group(
        self, _isolated: SimpleNamespace
    ) -> None:
        lead = _foreign_lead_group(_isolated.session)
        member = hooks._member_wake_hook_matcher(_isolated.session, MEMBER)
        _write_legacy(_isolated.legacy, [lead, member])

        result = asyncio.run(
            ss.install_member_wake(_isolated.sid, MEMBER, scope="project")
        )

        assert result["action"] == "installed"
        assert "success" not in result  # member result shape is unchanged
        assert Path(result["path"]) == _isolated.local
        assert Path(result["migrated_from"]) == _isolated.legacy
        legacy = _commands(_isolated.legacy)
        assert len(legacy) == 1
        assert "claude_teams.lead_wake" in legacy[0]
        local = _commands(_isolated.local)
        assert len([c for c in local if "claude_teams.member_wake" in c]) == 1

    def test_lead_install_leaves_legacy_member_group(
        self, _isolated: SimpleNamespace
    ) -> None:
        member = hooks._member_wake_hook_matcher(_isolated.session, MEMBER)
        _write_legacy(_isolated.legacy, [member])
        before = _isolated.legacy.read_bytes()

        result = asyncio.run(ss.install_lead_wake())

        assert "migrated_from" not in result
        assert _isolated.legacy.read_bytes() == before

    def test_remove_works_after_joined_session_is_gone(
        self, _isolated: SimpleNamespace
    ) -> None:
        asyncio.run(ss.install_member_wake(_isolated.sid, MEMBER, scope="project"))
        shutil.rmtree(_isolated.session)

        result = asyncio.run(
            ss.install_member_wake(_isolated.sid, MEMBER, remove=True, scope="project")
        )

        assert result["action"] == "removed"
        assert _commands(_isolated.local) == []

    def test_remove_still_rejects_a_malformed_session_id(
        self, _isolated: SimpleNamespace
    ) -> None:
        result = asyncio.run(
            ss.install_member_wake("not-a-uuid", MEMBER, remove=True, scope="project")
        )

        assert result["reason"] == "invalid_session_id"

    def test_member_write_is_atomic(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asyncio.run(ss.install_member_wake(_isolated.sid, MEMBER, scope="project"))
        before = _isolated.local.read_bytes()
        monkeypatch.setattr(
            Path, "replace", lambda _p, _t: (_ for _ in ()).throw(OSError("nope"))
        )

        result = asyncio.run(
            ss.install_member_wake(_isolated.sid, MEMBER, scope="project")
        )

        assert result == {"success": False, "reason": "settings_write_failed"}
        assert _isolated.local.read_bytes() == before
        assert list(_isolated.local.parent.glob("*.tmp")) == []


class TestGitIgnore:
    def test_not_a_repo(self, _isolated: SimpleNamespace) -> None:
        result = asyncio.run(ss.install_lead_wake())

        assert result["git_ignore"] == "not_a_repo"

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
    def test_unignored_repo_gets_a_local_exclude(
        self, _isolated: SimpleNamespace
    ) -> None:
        subprocess.run(["git", "init", "-q"], cwd=_isolated.work, check=True)  # noqa: S607

        first = asyncio.run(ss.install_lead_wake())
        second = asyncio.run(ss.install_lead_wake())

        assert first["git_ignore"] == "excluded"
        assert second["git_ignore"] == "already_ignored"
        exclude = (_isolated.work / ".git" / "info" / "exclude").read_text(
            encoding="utf-8"
        )
        assert exclude.count(".claude/settings.local.json") == 1
        assert not (_isolated.work / ".gitignore").exists()

    def test_user_scope_has_no_git_ignore_field(
        self, _isolated: SimpleNamespace
    ) -> None:
        result = asyncio.run(ss.install_lead_wake(scope="user"))

        assert "git_ignore" not in result


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _is_ignored(cwd: Path, path: Path) -> bool:
    done = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", str(path)],  # noqa: S607
        cwd=cwd,
        check=False,
    )
    return done.returncode == 0


class TestReviewFindings:
    """Cases added from the post-implementation review."""

    @pytest.mark.parametrize("hooks_value", [[], "nope", {"Stop": "nope"}])
    def test_malformed_legacy_shape_is_left_alone(
        self, _isolated: SimpleNamespace, hooks_value: object
    ) -> None:
        _isolated.legacy.parent.mkdir()
        _isolated.legacy.write_text(json.dumps({"hooks": hooks_value}), "utf-8")
        before = _isolated.legacy.read_bytes()

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert "migrated_from" not in result
        assert _isolated.legacy.read_bytes() == before

    def test_unreadable_legacy_file_reports_cleanup_failure(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_legacy(_isolated.legacy, [_foreign_lead_group(_isolated.session)])
        original_read = Path.read_text

        def _deny(path: Path, encoding: str | None = None) -> str:
            if path == _isolated.legacy:
                raise PermissionError
            return original_read(path, encoding=encoding)

        monkeypatch.setattr(Path, "read_text", _deny)

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert result["legacy_cleanup"] == "failed"
        assert Path(result["legacy_path"]) == _isolated.legacy

    def test_first_ever_project_remove_creates_no_file(
        self, _isolated: SimpleNamespace
    ) -> None:
        lead = asyncio.run(ss.install_lead_wake(remove=True))
        member = asyncio.run(
            ss.install_member_wake(_isolated.sid, MEMBER, remove=True, scope="project")
        )

        assert lead["action"] == "removed"
        assert member["action"] == "removed"
        assert not _isolated.local.exists()

    def test_member_remove_after_session_loss_also_migrates_legacy(
        self, _isolated: SimpleNamespace
    ) -> None:
        asyncio.run(ss.install_member_wake(_isolated.sid, MEMBER, scope="project"))
        _write_legacy(
            _isolated.legacy,
            [hooks._member_wake_hook_matcher(_isolated.session, MEMBER)],
        )
        shutil.rmtree(_isolated.session)

        result = asyncio.run(
            ss.install_member_wake(_isolated.sid, MEMBER, remove=True, scope="project")
        )

        assert Path(result["migrated_from"]) == _isolated.legacy
        assert _commands(_isolated.local) == []
        assert _commands(_isolated.legacy) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
class TestGitIgnoreLayouts:
    def test_tracked_local_file_is_reported_not_excluded(
        self, _isolated: SimpleNamespace
    ) -> None:
        _git(_isolated.work, "init", "-q")
        _isolated.local.parent.mkdir()
        _isolated.local.write_text("{}", encoding="utf-8")
        _git(_isolated.work, "add", "-f", str(_isolated.local))

        result = asyncio.run(ss.install_lead_wake())

        assert result["success"] is True
        assert result["git_ignore"] == "tracked"

    def test_nested_cwd_with_spaces_is_ignored(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git(_isolated.work, "init", "-q")
        nested = _isolated.work / "sub dir" / "app"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        result = asyncio.run(ss.install_lead_wake())

        local = nested / ".claude" / "settings.local.json"
        assert Path(result["path"]) == local
        assert result["git_ignore"] == "excluded"
        assert _is_ignored(nested, local)
        # Anchored: a same-named file elsewhere in the repo is not swallowed.
        other = _isolated.work / ".claude" / "settings.local.json"
        assert not _is_ignored(_isolated.work, other)

    def test_linked_worktree_is_ignored(
        self, _isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git(_isolated.work, "init", "-q")
        _git(_isolated.work, "commit", "-q", "--allow-empty", "-m", "init")
        linked = _isolated.work.parent / "linked"
        _git(_isolated.work, "worktree", "add", "-q", str(linked))
        monkeypatch.chdir(linked)

        result = asyncio.run(ss.install_lead_wake())

        assert result["git_ignore"] == "excluded"
        assert _is_ignored(linked, linked / ".claude" / "settings.local.json")

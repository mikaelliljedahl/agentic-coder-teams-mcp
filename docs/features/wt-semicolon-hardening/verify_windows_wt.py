"""Manual Windows verification for Windows Terminal command-line hardening.

Run by hand on Windows, with Windows Terminal installed, from the repo root:

    uv run python docs/features/wt-semicolon-hardening/verify_windows_wt.py

Not a pytest test (CI is Linux). Every case uses its own nonce and output
file; a missing output is a FAILURE, never "protected". Each result is
labelled with the boundary it exercised:

- ``wt->native(py)`` / ``wt->native(rust)``: wt split + unescape + re-join,
  then a native child's argv parser (Python; Rust if ``rustc`` is on PATH,
  since codex is Rust). ``-d`` start directory included.
- ``wt->ps-file``: the real tab spawn (wrapper ``.ps1`` via ``-File``) under a
  hostile log dir; the "agent" is a PowerShell script stub, so this covers the
  wrapper path and PowerShell literal/parameter binding -- NOT PowerShell's
  later native-argv marshalling.
- ``console-fallback``: a ``%`` pair in the log dir must make the real spawn
  skip wt and start the agent in a classic console.
- ``wt->tail``: the real tail ``-EncodedCommand`` script, with ``Get-Content``
  stubbed to record the path it was given.
- ``inject``: ``x;cmd.exe /c mk.cmd`` -- the raw (old) token must run
  ``mk.cmd`` (positive control), the encoded one must not.
- ``expand``: only with an ISOLATED Terminal process (none running at start),
  started with ``WTV_PROBE`` in its environment: the raw ``%WTV_PROBE%`` must
  be rewritten by wt (proves the stage exists) and the production guard must
  refuse it. Reported NOT EXECUTED otherwise.

Cases run in two window modes: a fresh named window per case, and an existing
named window kept alive by an anchor tab. Exit code 0 only if every executed
case passed.
"""

# ruff: noqa: T201, S603, S607 - manual CLI verifier: prints, runs local tools.

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from claude_teams.backends import process_manager as pm
from claude_teams.backends.contracts import SpawnRequest

WAIT_SECONDS = 20
RESULTS: list[tuple[str, str, str]] = []  # (boundary, case, PASS|FAIL|NOT EXECUTED)

RECORDER_PY = r"""
import json, os, sys
out = sys.argv[1]
data = {"argv": sys.argv[2:], "cwd": os.getcwd()}
with open(out + ".tmp", "w", encoding="utf-8") as fh:
    json.dump(data, fh)
os.replace(out + ".tmp", out)
"""

RECORDER_RS = r"""
use std::io::Write;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut s = String::new();
    for a in &args[2..] {
        for b in a.as_bytes() { s.push_str(&format!("{:02x}", b)); }
        s.push('\n');
    }
    let tmp = format!("{}.tmp", &args[1]);
    std::fs::File::create(&tmp).unwrap().write_all(s.as_bytes()).unwrap();
    std::fs::rename(&tmp, &args[1]).unwrap();
}
"""

ANCHOR_PY = r"""
import pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text("ready", encoding="utf-8")
time.sleep(900)
"""

PS_STUB = (
    "$lines = @($args | ForEach-Object { "
    "[BitConverter]::ToString([Text.Encoding]::UTF8.GetBytes([string]$_)) })\r\n"
    "$lines += 'CWD ' + [BitConverter]::ToString("
    "[Text.Encoding]::UTF8.GetBytes((Get-Location).Path))\r\n"
    "Set-Content -LiteralPath $env:WTV_OUT -Value $lines -Encoding ascii\r\n"
)

HOSTILE_ARG_SETS = [
    ["plain", "has space", "tab\there", ""],
    ['quote"inside', 'quote "and space"', 'a\\"b c', 'a\\\\"b'],
    ["a b\\", "a b\\\\", "trailing\\", 'x;"y z";\\'],
    [";", ";;", "\\;", "semi;colon", "tab\t;semi"],
    ['Fix the "parser"; then run tests\\', "it's \u2019smart\u2019", "50% done"],
]
HOSTILE_DIR_NAMES = ["a;b", "a b;c", ";y", "sp ace", "x'\u2019q"]


def record(boundary: str, case: str, ok: bool | None, detail: str = "") -> None:
    """Record and print one case result (``None`` = not executed)."""
    verdict = "NOT EXECUTED" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((boundary, case, verdict))
    print(f"  [{verdict}] {boundary}: {case}" + (f" -- {detail}" if detail else ""))


def unhex(dashed: str) -> str:
    """Decode a ``BitConverter.ToString`` line (``41-42``) as UTF-8."""
    return bytes.fromhex(dashed.replace("-", "")).decode("utf-8")


def wait_for(path: Path, seconds: float = WAIT_SECONDS) -> bool:
    """Wait until ``path`` exists."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.25)
    return False


def terminal_running() -> bool:
    """Whether any WindowsTerminal.exe process exists."""
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq WindowsTerminal.exe", "/NH"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    ).stdout
    return "WindowsTerminal.exe" in out


def describe_environment(wt: str) -> None:
    """Print the wt, Terminal, PowerShell and Python versions exercised."""
    print(f"wt.exe resolved: {wt}")
    ps = shutil.which("powershell.exe") or "powershell.exe"
    probe = (
        "Get-Process WindowsTerminal -ErrorAction SilentlyContinue | "
        "ForEach-Object { $_.Path + ' ' + $_.MainModule.FileVersionInfo.FileVersion }"
        "; 'PS ' + $PSVersionTable.PSVersion"
    )
    out = subprocess.run(
        [ps, "-NoProfile", "-Command", probe],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    print(f"WindowsTerminal.exe / PowerShell: {out}")
    print(f"Python: {sys.version.split()[0]}")


def launch(argv: list[str], env: dict[str, str] | None = None) -> None:
    """Start a wt argv the way production does (Popen of the list)."""
    subprocess.Popen(
        argv,
        env={**os.environ, **(env or {})},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Verifier:
    """Holds the scratch dir, recorders and window names for one run."""

    def __init__(self, wt: str, work: Path) -> None:
        """Prepare recorders and a space-free dir for the batch probe."""
        self.wt = wt
        self.work = work
        self.py = sys.executable
        self.rec_py = work / "rec.py"
        self.rec_py.write_text(RECORDER_PY, encoding="utf-8")
        self.anchor_py = work / "anchor.py"
        self.anchor_py.write_text(ANCHOR_PY, encoding="utf-8")
        self.rec_rs: Path | None = None
        rustc = shutil.which("rustc")
        if rustc:
            src = work / "rec.rs"
            src.write_text(RECORDER_RS, encoding="utf-8")
            exe = work / "rec_rs.exe"
            built = subprocess.run(
                [rustc, "-O", "-o", str(exe), str(src)],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            self.rec_rs = exe if built.returncode == 0 else None
        # mk.cmd must sit on a space-free, ';'-free path to run as an
        # injected fragment.
        plain = work if " " not in str(work) else None
        if plain is None:
            drive = os.environ.get("SYSTEMDRIVE", "C:")
            plain = Path(f"{drive}\\wtv-{secrets.token_hex(4)}")
            plain.mkdir()
        self.plain = plain
        self.window = f"wtv-{secrets.token_hex(3)}"

    def out(self, label: str) -> Path:
        """Return a fresh, unique output path."""
        return self.work / f"out-{label}-{secrets.token_hex(4)}"

    def start_anchor(self, env: dict[str, str] | None = None) -> bool:
        """Open the anchor tab that keeps ``self.window`` alive."""
        ready = self.out("anchor")
        launch(
            pm._wt_argv(
                self.wt,
                ["-w", self.window, "nt", "--title", "wtv-anchor"],
                [self.py, str(self.anchor_py), str(ready)],
            ),
            env,
        )
        return wait_for(ready)

    def native_case(self, mode: str, dir_name: str, args: list[str], rust: bool):
        """Round-trip ``args`` and a hostile ``-d`` through wt to a recorder."""
        start = self.work / "dirs" / dir_name
        start.mkdir(parents=True, exist_ok=True)
        out = self.out("native")
        window = self.window if mode == "existing" else f"wtv-f-{secrets.token_hex(3)}"
        if rust and self.rec_rs is not None:
            child = [str(self.rec_rs), str(out), *args]
        else:
            child = [self.py, str(self.rec_py), str(out), *args]
        launch(pm._wt_argv(self.wt, ["-w", window, "nt", "-d", str(start)], child))
        boundary = "wt->native(rust)" if rust else "wt->native(py)"
        case = f"{mode} -d {dir_name!r} args={args!r}"
        if not wait_for(out):
            record(boundary, case, ok=False, detail="no recorder output")
            return
        if rust:
            got = [
                bytes.fromhex(line).decode("utf-8")
                for line in out.read_text(encoding="ascii").split("\n")[:-1]
            ]
            record(boundary, case, got == args, f"got {got!r}")
            return
        data = json.loads(out.read_text(encoding="utf-8"))
        ok = data["argv"] == args and Path(data["cwd"]) == start
        record(boundary, case, ok, "" if ok else f"got {data!r}")

    def injection_probe(self) -> None:
        """Positive control (raw token runs mk.cmd) and protected run."""
        for protected in (False, True):
            sentinel = self.plain / f"pwned-{secrets.token_hex(4)}"
            mk = self.plain / "mk.cmd"
            mk.write_text(f'@echo 1> "{sentinel}"\r\n', encoding="ascii")
            out = self.out("inject")
            args = ["x;cmd.exe", "/c", str(mk)]
            child = [self.py, str(self.rec_py), str(out), *args]
            options = ["-w", self.window, "nt"]
            if protected:
                argv = pm._wt_argv(self.wt, options, child)
            else:
                argv = [self.wt, *options, "--", *child]  # the old, raw form
            launch(argv)
            if not protected:
                ok = wait_for(sentinel)
                record(
                    "inject",
                    "positive control: raw ';' runs mk.cmd",
                    ok,
                    "" if ok else "oracle blind: sentinel never appeared",
                )
            else:
                got_out = wait_for(out)
                time.sleep(3)
                ok = got_out and not sentinel.exists()
                if got_out:
                    ok = ok and json.loads(out.read_text("utf-8"))["argv"] == args
                record("inject", "encoded ';' stays data", ok)
            mk.unlink(missing_ok=True)
            sentinel.unlink(missing_ok=True)

    def wrapper_case(self) -> None:
        """Real tab spawn with a hostile log dir and cwd, PowerShell stub agent."""
        log_dir = self.work / "l;o g's"
        cwd = self.work / "dirs" / "c;w d'\u2019"
        cwd.mkdir(parents=True, exist_ok=True)
        stub = self.work / "stub.ps1"
        stub.write_bytes(PS_STUB.encode("utf-8-sig"))
        out = self.out("wrapper")
        args = ["a;b", 'say "hi"', "x'\u2019y", "tab\tz"]
        ok = self._spawn(log_dir, cwd, [str(stub), *args], {"WTV_OUT": str(out)})
        if not ok or not wait_for(out):
            record("wt->ps-file", "real tab spawn, hostile log dir", ok=False)
            return
        lines = out.read_text(encoding="ascii").splitlines()
        got = [unhex(line) for line in lines[:-1]]
        got_cwd = unhex(lines[-1].removeprefix("CWD "))
        log = (log_dir / "wtv" / "worker.log").read_text("utf-8", errors="replace")
        route_ok = "[wt command]" in log and "[fallback]" not in log
        passed = route_ok and got == args and Path(got_cwd) == cwd
        record("wt->ps-file", "real tab spawn, hostile log dir", passed, f"{got!r}")

    def console_fallback_case(self) -> None:
        """Check that a '%' pair in the log dir makes the spawn skip wt."""
        log_dir = self.work / "l%WTV_A%%WTV_B%"
        out = self.out("fallback")
        args = ["x", "y z"]
        ok = self._spawn(
            log_dir, self.work, [self.py, str(self.rec_py), str(out), *args], {}
        )
        log = (log_dir / "wtv" / "worker.log").read_text("utf-8", errors="replace")
        route_ok = "[fallback]" in log and "'%' pair" in log
        ok = ok and route_ok and "[wt command]" not in log and wait_for(out)
        ok = ok and json.loads(out.read_text("utf-8"))["argv"] == args
        record("console-fallback", "'%' pair in log dir", ok)

    def _spawn(self, log_dir: Path, cwd: Path, cmd: list[str], env: dict) -> bool:
        os.environ["WIN_AGENT_TEAMS_LOG_DIR"] = str(log_dir)
        os.environ["WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS"] = "0"
        os.environ.pop(pm._CODEX_DIRECT_LAUNCH_ENV, None)
        request = SpawnRequest(
            agent_id="worker@wtv",
            name="worker",
            team_name="wtv",
            prompt="",
            model="default",
            agent_type="general-purpose",
            color="blue",
            cwd=str(cwd),
            lead_session_id="wtv",
        )
        try:
            pm.WindowsProcessManager().spawn_process(
                request, cmd, env, "codex", is_interactive=True
            )
        except Exception as err:  # reported as a failure
            print(f"    spawn raised {err!r}")
            return False
        return True

    def tail_case(self, window: str, label: str) -> None:
        """Real tail script, Get-Content stubbed to record its path."""
        log_path = self.work / "t;a%USERNAME%il'\u2019s" / "worker.log"
        captured = mock.MagicMock()
        with (
            mock.patch.object(pm.shutil, "which", return_value=self.wt),
            mock.patch.object(pm.subprocess, "Popen", captured),
        ):
            pm.WindowsProcessManager()._open_windows_terminal_tail("wtv", "w", log_path)
        argv = captured.call_args.args[0]
        script = base64.b64decode(argv[-1]).decode("utf-16-le")
        out = self.out("tail")
        stub = (
            "function Get-Content { param([string]$LiteralPath, [switch]$Wait, "
            "[int]$Tail) [IO.File]::WriteAllText("
            f"{pm._powershell_quote(str(out))}, $LiteralPath) }}\r\n"
        )
        child = ["powershell", "-NoProfile", "-EncodedCommand"]
        child.append(pm._powershell_encoded(stub + script))
        launch(pm._wt_argv(self.wt, ["-w", window, "nt"], child))
        ok = wait_for(out) and out.read_text(encoding="utf-8-sig") == str(log_path)
        record("wt->tail", f"{label}: path exact, %USERNAME% not expanded", ok)

    def expansion_probe(self) -> None:
        """Raw %WTV_PROBE% is rewritten and re-split by wt; the guard refuses it."""
        out = self.out("expand")
        # wt quotes this arg as "x %WTV_PROBE% y"; with WTV_PROBE = a" "b the
        # expansion yields "x a" "b y", so the recorder must see TWO args.
        arg = "x %WTV_PROBE% y"
        child = [self.py, str(self.rec_py), str(out), arg]
        launch(pm._wt_argv(self.wt, ["-w", self.window, "nt"], child))
        got = json.loads(out.read_text("utf-8"))["argv"] if wait_for(out) else None
        record(
            "expand",
            "wt expands %WTV_PROBE% and re-splits argv",
            got == ["x a", "b y"],
            f"got {got!r}",
        )
        guard = pm.WindowsProcessManager._codex_direct_launch_blocker(
            ["C:\\codex.exe", arg], str(self.work)
        )
        record("expand", "direct-launch guard refuses the '%' pair", guard is not None)


def main() -> int:
    """Run every case; exit 0 only if all executed cases passed."""
    if os.name != "nt":
        print("Windows only.")
        return 2
    wt = shutil.which("wt.exe")
    if wt is None:
        print("wt.exe not found.")
        return 2
    isolated = not terminal_running()
    work = Path(tempfile.mkdtemp(prefix="wtv-"))
    verifier = Verifier(wt, work)
    try:
        probe_env = {"WTV_PROBE": 'a" "b'} if isolated else None
        if not verifier.start_anchor(probe_env):
            print("anchor tab never became ready; aborting")
            return 1
        describe_environment(wt)
        print(
            f"isolated Terminal process: {isolated}; rust recorder: "
            f"{verifier.rec_rs is not None}"
        )
        for mode in ("fresh", "existing"):
            for name, args in zip(HOSTILE_DIR_NAMES, HOSTILE_ARG_SETS, strict=True):
                verifier.native_case(mode, name, args, rust=False)
                if verifier.rec_rs is not None:
                    verifier.native_case(mode, name, args, rust=True)
        if verifier.rec_rs is None:
            record("wt->native(rust)", "rustc not on PATH", None)
        verifier.injection_probe()
        verifier.wrapper_case()
        verifier.console_fallback_case()
        verifier.tail_case(verifier.window, "named window")
        if isolated:
            verifier.tail_case("0", "-w 0 (isolated process)")
            verifier.expansion_probe()
        else:
            record("wt->tail", "-w 0 needs an isolated Terminal process", None)
            record("expand", "needs an isolated Terminal process", None)
    finally:
        print("Close the wtv-* Terminal windows, then delete:", work)
        if verifier.plain != work:
            shutil.rmtree(verifier.plain, ignore_errors=True)
    failed = [r for r in RESULTS if r[2] == "FAIL"]
    skipped = [r for r in RESULTS if r[2] == "NOT EXECUTED"]
    print(
        f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} not executed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

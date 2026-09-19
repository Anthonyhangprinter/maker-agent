"""Real-process tests for lab/gpu_window.sh (Task 3a, 2026-09-19).

These drive the ACTUAL shell script with subprocess, never a look-alike helper, because
every defect this file guards against lives in process mechanics the script cannot fake:
which process group a signal reaches, which file descriptor a descendant inherited, and
which systemd unit the EXIT trap starts.

Nothing here touches the real box. The script's side-effecting commands are env seams
(`GPU_WINDOW_SYSTEMCTL`, `GPU_WINDOW_NVIDIA_SMI`, `CAD_BUILD_LOCK_FILE`, the same var
cad_v5/config.py's BUILD_LOCK_FILE honours), and every test points them at stub scripts in
a tmp dir that append their argv to a log file, plus a tmp lock file. No systemctl, no
nvidia-smi, no GPU, no real lock.

The defects, all of which have a failing case below when run against the pre-Task-3a
script (set GPU_WINDOW_SCRIPT=/path/to/old.sh to reproduce):

  D1 the KILL escalation hit `timeout` only, so the real job and its grandchildren survived
     as orphans still holding the build lock through the inherited fd 9
     (test_term_kills_a_stubborn_child_and_grandchild, test_max_sec_expiry_*)
  D2 the TERM-to-KILL grace was a hardcoded 20s (test_grace_sec_is_honoured[*])
  D3 the EXIT trap restored "whatever was active at entry", i.e. a stray maker
     (test_restore_always_targets_the_resident[*])
  D4 the lock wait was a hardcoded 3600s with exit code 3 (test_lock_contention_*)
"""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("GPU_WINDOW_SCRIPT", str(HERE / "lab" / "gpu_window.sh")))

SYSTEMCTL_STUB = """#!/usr/bin/env bash
# Records its argv, answers `is-active maker-server` from $STUB_MAKER_STATE, and (only when
# $STUB_TERM_PARENT_ON_RESTORE is set) sends its caller a SIGTERM on the SECOND
# "stop maker-server" -- i.e. the one the EXIT trap's restore makes, never the eviction.
printf '%s\\n' "$*" >> "$STUB_LOG"
if [ "$*" = "--user is-active maker-server" ]; then
    state="${STUB_MAKER_STATE:-inactive}"
    printf '%s\\n' "$state"
    [ "$state" = "active" ] || exit 3
    exit 0
fi
if [ "$*" = "--user stop maker-server" ] && [ -n "${STUB_TERM_PARENT_ON_RESTORE:-}" ]; then
    n=$(grep -c -- '--user stop maker-server' "$STUB_LOG" || true)
    if [ "$n" -ge 2 ]; then
        kill -TERM "$PPID" 2>/dev/null || true
        sleep 1
    fi
fi
exit 0
"""

NVIDIA_SMI_STUB = """#!/usr/bin/env bash
printf '%s\\n' "${STUB_VRAM:-0}"
"""

# A job that refuses to die on TERM and takes a grandchild with it. `trap '' TERM` is
# SIG_IGN, which its own children inherit, so only a KILL to the whole process group
# clears this tree.
STUBBORN_JOB = """#!/usr/bin/env bash
DIR="$1"
trap '' TERM
bash -c 'trap "" TERM; echo $$ > "$1/grand.pid"; while :; do sleep 0.2; done' _ "$DIR" &
echo $$ > "$DIR/child.pid"
while :; do sleep 0.2; done
"""

# A well-behaved job that needs 5s to shut down cleanly (the real jobs' own arm restore
# waits on a server health check, which is the whole reason the grace had to grow).
SLOW_CLEANUP_JOB = """#!/usr/bin/env bash
DIR="$1"
cleanup() { sleep 5; echo "cleanup done" > "$DIR/cleanup.marker"; exit 0; }
trap cleanup TERM
echo $$ > "$DIR/child.pid"
while :; do sleep 0.2; done
"""

ENV_REPORT_JOB = """#!/usr/bin/env bash
echo "out CAD_GPU_WINDOW=${CAD_GPU_WINDOW:-unset} PYTHONUTF8=${PYTHONUTF8:-unset}"
echo "err line" >&2
exit 5
"""


def _write_exe(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _alive(pid: int) -> bool:
    """True when the pid exists AND is not a zombie. A killed process whose parent has not
    reaped it yet still answers kill(pid, 0), so existence alone would be a false positive."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return False
    return stat.rsplit(") ", 1)[-1].split()[0] != "Z"


def _wait_gone(pid: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def _wait_for_file(path: Path, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared (or stayed empty) within {timeout}s")


class Window:
    """One isolated gpu_window.sh harness: stub dir, tmp lock, tmp log, launcher."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.systemctl = _write_exe(self.bin / "systemctl", SYSTEMCTL_STUB)
        self.nvidia_smi = _write_exe(self.bin / "nvidia-smi", NVIDIA_SMI_STUB)
        self.stubborn = _write_exe(self.bin / "stubborn_job.sh", STUBBORN_JOB)
        self.slow_cleanup = _write_exe(self.bin / "slow_cleanup_job.sh", SLOW_CLEANUP_JOB)
        self.env_report = _write_exe(self.bin / "env_report_job.sh", ENV_REPORT_JOB)
        self.lock = tmp_path / "cad-build.lock"
        self.log = tmp_path / "systemctl.log"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.procs: list[subprocess.Popen] = []
        self.watched_pids: list[int] = []

    def launch(self, cmd: list[str], **overrides: str) -> subprocess.Popen:
        env = dict(os.environ)
        env.update({
            "GPU_WINDOW_SYSTEMCTL": str(self.systemctl),
            "GPU_WINDOW_NVIDIA_SMI": str(self.nvidia_smi),
            "CAD_BUILD_LOCK_FILE": str(self.lock),
            "STUB_LOG": str(self.log),
            "HOME": str(self.home),
            # Deliberately hostile defaults: the window must set these for its child itself.
            "CAD_GPU_WINDOW": "",
            "PYTHONUTF8": "0",
        })
        env.update(overrides)
        p = subprocess.Popen(
            [str(SCRIPT), *cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", start_new_session=True,
        )
        self.procs.append(p)
        return p

    def read_pid(self, name: str) -> int:
        pid = int(_wait_for_file(self.tmp / name))
        self.watched_pids.append(pid)
        return pid

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return [ln for ln in self.log.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def lock_is_free(self) -> bool:
        """True when nothing (no orphaned descendant holding an inherited fd 9) still holds
        the build lock."""
        with open(self.lock, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(fh, fcntl.LOCK_UN)
        return True

    def cleanup(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            for stream in (p.stdout, p.stderr):
                if stream:
                    stream.close()
        for pid in self.watched_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def win(tmp_path):
    w = Window(tmp_path)
    yield w
    w.cleanup()


# --------------------------------------------------------------------------- D1: the group

def test_term_kills_a_stubborn_child_and_grandchild(win):
    """D1. A job that ignores TERM, and a grandchild that also ignores it, must BOTH be gone
    once the window has been TERMed and exited -- and nothing may still hold the build lock,
    which is what proves no descendant inherited fd 9."""
    p = win.launch([str(win.stubborn), str(win.tmp)], GPU_WINDOW_GRACE_SEC="2")
    child = win.read_pid("child.pid")
    grand = win.read_pid("grand.pid")
    assert _alive(child) and _alive(grand)

    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)

    assert rc == 143, f"expected the TERM exit code, got {rc}"
    assert _wait_gone(child), f"the job {child} outlived the window"
    assert _wait_gone(grand), f"the grandchild {grand} outlived the window"
    assert win.lock_is_free(), "the build lock is still held after the window exited"
    assert win.calls()[-2:] == ["--user stop maker-server", "--user start qwen38-server"]


def test_max_sec_expiry_kills_a_stubborn_child_and_grandchild_and_restores(win):
    """D1/D2. The wall-clock cap must reach the whole group, not just `timeout`'s direct
    child, and the restore must still run afterwards. --kill-after rides on the same grace
    knob, so a 2s grace makes this fast."""
    p = win.launch([str(win.stubborn), str(win.tmp)],
                   GPU_WINDOW_MAX_SEC="2", GPU_WINDOW_GRACE_SEC="2")
    child = win.read_pid("child.pid")
    grand = win.read_pid("grand.pid")

    rc = p.wait(timeout=60)

    # 124 = timeout's own "timed out" code; 137 when the job ignored the TERM and coreutils'
    # group-wide SIGKILL took `timeout` itself down with the group (measured: this case).
    assert rc in (124, 137), f"expected a timeout exit code, got {rc}"
    assert _wait_gone(child), f"the job {child} survived the wall-clock cap"
    assert _wait_gone(grand), f"the grandchild {grand} survived the wall-clock cap"
    assert win.lock_is_free()
    assert win.calls()[-2:] == ["--user stop maker-server", "--user start qwen38-server"]


# ---------------------------------------------------------------------------- D2: the grace

@pytest.mark.parametrize("grace,expect_marker", [("10", True), ("1", False)])
def test_grace_sec_is_honoured(win, grace, expect_marker):
    """D2. A job that traps TERM and needs 5s to clean up finishes when the grace is 10, and
    is killed mid-cleanup when the grace is 1. Both halves are needed: the first alone would
    pass against the old hardcoded 20s."""
    p = win.launch([str(win.slow_cleanup), str(win.tmp)], GPU_WINDOW_GRACE_SEC=grace)
    win.read_pid("child.pid")
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=60)

    marker = win.tmp / "cleanup.marker"
    if expect_marker:
        assert marker.exists(), "the job was killed before its 5s cleanup finished"
        assert marker.read_text(encoding="utf-8").strip() == "cleanup done"
    else:
        assert not marker.exists(), "the job finished a 5s cleanup inside a 1s grace"


# -------------------------------------------------------------------------- D3: the restore

@pytest.mark.parametrize("maker_state", ["inactive", "active"])
@pytest.mark.parametrize("how", ["ok", "fail", "term"])
def test_restore_always_targets_the_resident(win, maker_state, how):
    """D3. Whatever was active at entry, and however the window ends, the last two systemctl
    calls are stop maker-server then start qwen38-server. A stray maker at entry used to make
    the trap restart the MAKER, on top of a resident the job's own bookend had just started."""
    if how == "term":
        cmd = [str(win.slow_cleanup), str(win.tmp)]
    elif how == "fail":
        cmd = ["bash", "-c", "exit 7"]
    else:
        cmd = ["bash", "-c", "exit 0"]

    p = win.launch(cmd, STUB_MAKER_STATE=maker_state, GPU_WINDOW_GRACE_SEC="10")
    if how == "term":
        win.read_pid("child.pid")
        p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)

    assert rc == {"ok": 0, "fail": 7, "term": 143}[how]
    calls = win.calls()
    assert calls[0] == "--user is-active maker-server"
    assert calls[-2:] == ["--user stop maker-server", "--user start qwen38-server"], calls
    assert "--user start maker-server" not in calls


def test_restore_maker_escape_hatch(win):
    """D3's documented escape hatch: GPU_WINDOW_RESTORE=maker restarts the maker arm instead
    of the resident. Nothing on the box uses it today."""
    p = win.launch(["bash", "-c", "exit 0"], GPU_WINDOW_RESTORE="maker",
                   STUB_MAKER_STATE="inactive")
    assert p.wait(timeout=60) == 0
    calls = win.calls()
    assert calls[-1] == "--user start maker-server", calls
    assert "--user start qwen38-server" not in calls


def test_a_second_term_during_the_restore_does_not_abort_it(win):
    """A SIGTERM landing while the EXIT trap is restoring the box must not abandon the
    restore half way: the stub sends one from inside the restore's own `stop maker-server`
    call, and `start qwen38-server` must still happen."""
    p = win.launch(["bash", "-c", "exit 0"], STUB_TERM_PARENT_ON_RESTORE="1")
    rc = p.wait(timeout=60)
    calls = win.calls()
    assert calls[-1] == "--user start qwen38-server", calls
    assert rc == 0, f"the ignored TERM changed the exit code to {rc}"


# ------------------------------------------------------------------------ D4: the lock wait

def test_lock_contention_exits_75_without_evicting_anything(win):
    """D4. When the build lock is held elsewhere, the window waits GPU_WINDOW_LOCK_WAIT_SEC
    and then exits 75 (EX_TEMPFAIL) so a systemd unit logs a skip rather than a failure -- and
    it must not have touched a single service on the way, since the eviction only ever happens
    behind the lock."""
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,sys,time\n"
         "f=open(sys.argv[1],'a')\n"
         "fcntl.flock(f, fcntl.LOCK_EX)\n"
         "sys.stdout.write('held\\n'); sys.stdout.flush()\n"
         "time.sleep(120)\n", str(win.lock)],
        stdout=subprocess.PIPE, encoding="utf-8")
    try:
        assert holder.stdout.readline().strip() == "held"
        p = win.launch(["bash", "-c", "exit 0"], GPU_WINDOW_LOCK_WAIT_SEC="1")
        start = time.monotonic()
        rc = p.wait(timeout=30)
        waited = time.monotonic() - start
        assert rc == 75, f"expected EX_TEMPFAIL 75, got {rc}"
        assert waited < 20, f"the lock wait knob was ignored ({waited:.1f}s)"
        assert win.calls() == [], "a service was touched before the lock was held"
        assert "exit 75" in (p.stderr.read() if p.stderr else "")
    finally:
        holder.kill()
        holder.wait(timeout=10)
        if holder.stdout:
            holder.stdout.close()


# ------------------------------------------------------------- the child's own environment

def test_child_environment_exit_code_and_output_passthrough(win):
    """The job sees CAD_GPU_WINDOW=1 (what lab/ship.py's verify gate checks) and
    PYTHONUTF8=1 (OpenCascade's Mesher().write() resets the process locale to C, after which
    unencoded reads decode as ASCII), its exit code is this script's exit code, and its
    stdout/stderr pass through. The parent environment deliberately sets PYTHONUTF8=0 and an
    empty CAD_GPU_WINDOW, so both values can only come from the window itself."""
    p = win.launch([str(win.env_report)])
    out, err = p.communicate(timeout=60)
    assert p.returncode == 5
    assert "out CAD_GPU_WINDOW=1 PYTHONUTF8=1" in out, out
    assert "err line" in err, err
    assert "== gpu_window: start" in out and "== gpu_window: done" in out

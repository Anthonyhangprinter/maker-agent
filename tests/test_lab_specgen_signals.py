"""Real-subprocess tests for lab/specgen.py's SIGTERM/SIGHUP handling (Task 2 fix
round 2, finding 1).

These spawn an actual child process, send it a real signal, and check what actually
happened on disk and in its exit code -- not a mock of signal.signal(). The original
review confirmed by an isolated /tmp reproduction that Python's DEFAULT SIGTERM
disposition terminates a process immediately with no exception raised and no `finally`
executed (unlike SIGINT, which the runtime converts to KeyboardInterrupt); a test that
only asserts "signal.signal was called with the right handler" would not catch a
regression where the handler itself is wrong, or where something re-installs the
default disposition later. All three tests here are offline (no network, no GPU, no
services) and complete in well under 5 seconds total.
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# A tiny standalone script: installs the real handler from lab/specgen.py, then mimics
# main()'s try/finally shape (a long sleep standing in for run_batch/run_total, a
# finally that marks cleanup started, optionally sleeps to simulate slow cleanup work,
# then writes a marker file). `cleanup_delay` gives the "second signal during cleanup"
# test a real window to send its second signal while the finally block is still running.
_HELPER = """
import sys, time
sys.path.insert(0, {repo!r})
from lab import specgen

marker = sys.argv[1]
cleanup_delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

specgen._install_signal_handlers()
rc = 0
try:
    time.sleep(10)
except specgen.SpecgenAborted:
    rc = 1
finally:
    specgen._mark_cleanup_started()
    if cleanup_delay:
        time.sleep(cleanup_delay)
    with open(marker, "w") as f:
        f.write("done")
sys.exit(rc)
"""


def _spawn(marker_path: Path, cleanup_delay: float = 0.0) -> subprocess.Popen:
    code = _HELPER.format(repo=str(HERE))
    return subprocess.Popen([sys.executable, "-c", code, str(marker_path), str(cleanup_delay)])


def _wait_and_cleanup(proc: subprocess.Popen, timeout: float = 4.0) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise


def test_sigterm_runs_the_finally_block_and_exits_nonzero(tmp_path):
    """The core regression test: without _install_signal_handlers(), this would fail --
    the child would be killed outright, the marker file would never be written, and the
    exit code would be the raw signal-terminated code with no SpecgenAborted involved.
    """
    marker = tmp_path / "marker.txt"
    proc = _spawn(marker)
    try:
        time.sleep(0.4)   # let the child install its handlers and enter the long sleep
        proc.send_signal(signal.SIGTERM)
        rc = _wait_and_cleanup(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert marker.exists(), "finally block did not run: SIGTERM did not reach it"
    assert marker.read_text() == "done"
    assert rc != 0


def test_sighup_also_runs_the_finally_block(tmp_path):
    marker = tmp_path / "marker.txt"
    proc = _spawn(marker)
    try:
        time.sleep(0.4)
        proc.send_signal(signal.SIGHUP)
        rc = _wait_and_cleanup(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert marker.exists(), "finally block did not run: SIGHUP did not reach it"
    assert marker.read_text() == "done"
    assert rc != 0


def test_second_sigterm_during_cleanup_does_not_abort_the_finally_block(tmp_path):
    """A second SIGTERM arriving WHILE the finally block is already running (its
    simulated slow-cleanup sleep) must not interrupt it: _mark_cleanup_started() makes
    the handler a no-op from that point on, so the marker still gets written and the
    process still exits non-zero, rather than being terminated mid-cleanup by the
    second signal."""
    marker = tmp_path / "marker.txt"
    proc = _spawn(marker, cleanup_delay=1.0)
    try:
        time.sleep(0.4)              # child is in its long sleep
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)              # first signal handled; now inside the cleanup_delay sleep
        proc.send_signal(signal.SIGTERM)   # must be ignored, not interrupt the cleanup
        rc = _wait_and_cleanup(proc, timeout=4.0)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert marker.exists(), "cleanup was interrupted by the second signal"
    assert marker.read_text() == "done"
    assert rc != 0

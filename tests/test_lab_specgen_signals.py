"""Real-subprocess tests for lab/specgen.py's SIGTERM/SIGHUP handling (Task 2 fix
round 2, finding 1).

These spawn an actual child process, send it a real signal, and check what actually
happened on disk and in its exit code -- not a mock of signal.signal(). The original
review confirmed by an isolated /tmp reproduction that Python's DEFAULT SIGTERM
disposition terminates a process immediately with no exception raised and no `finally`
executed (unlike SIGINT, which the runtime converts to KeyboardInterrupt); a test that
only asserts "signal.signal was called with the right handler" would not catch a
regression where the handler itself is wrong, or where something re-installs the
default disposition later. The first three tests below exercise the signal-handling
primitives in isolation; the fix-round-3 tests further down drive the REAL
lab.specgen.main()/run_total() in a real subprocess instead, which is what actually
caught the two regressions that shipped with the primitive-only tests green. All tests
here are offline (no network, no GPU, no services) and the whole file completes in well
under 15 seconds.
"""
import json
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


# ---------------------------------------------------------------------------
# Fix round 3: real signals against the REAL lab.specgen.main()/run_total(), not a
# look-alike helper. The fix-round-2 tests above give high confidence in the signal-to-
# exception mechanism as an isolated primitive; they never called main() or run_total()
# at all, which is exactly the gap that let two real regressions ship with those tests
# green (a signal mid-batch was silently absorbed as an ordinary call failure; a signal
# in the first ~100-150ms of the `use` call caused a restore that wrote
# maker.enabled: false with no marker ever having been written). Only `_run_arms` and
# `run_batch` are monkeypatched here (inside the spawned subprocess itself, since a
# separate process cannot be monkeypatched from the test process) -- no real arms.py
# subprocess, no model call, no GPU, no service. A temp CAD_CONFIG_FILE/MAKER_ENV point
# `_pre_arm_marker_paths()` (which imports scripts/arms.py's own CAD_JSON/ENV_PATH) at a
# throwaway directory, so nothing under the real ~/.openclaw/ is ever touched.
# ---------------------------------------------------------------------------

# {repo!r} = this repo's root (for sys.path); {cad_json!r}/{env_path!r} = the temp
# CAD_CONFIG_FILE/MAKER_ENV paths for this test's isolated .pre-arm marker checks.
# `scenario` picks what the faked `_run_arms("use", ...)` does before `main()`'s
# SIGTERM/SIGHUP handler can interrupt it:
#   mid_batch         -- "use" returns immediately (writes the marker first, like a
#                         real completed `arms.py use` would have), so the signal lands
#                         while the faked run_batch is asleep, mid-generation.
#   use_before_marker -- "use" sleeps for 3s and NEVER writes the marker; the signal
#                         must land inside that sleep, before any marker exists.
#   use_after_marker  -- "use" writes the marker FIRST, then sleeps for 3s (standing in
#                         for the rest of cmd_use's own systemctl/health-wait work); the
#                         signal lands inside that second sleep, after the marker exists.
_MAIN_HELPER = """
import json, os, sys, time
sys.path.insert(0, {repo!r})
os.environ["CAD_CONFIG_FILE"] = {cad_json!r}
os.environ["MAKER_ENV"] = {env_path!r}
from lab import specgen
import arms as arms_cli

call_log = sys.argv[1]
scenario = sys.argv[2]

def log(entry):
    with open(call_log, "a") as f:
        f.write(json.dumps(entry) + "\\n")

pre_cad, pre_env = arms_cli._pre_arm_paths(arms_cli.CAD_JSON, arms_cli.ENV_PATH)

def fake_run_arms(*args):
    log(list(args))
    if args[0] == "use":
        if scenario == "use_before_marker":
            time.sleep(3)
        elif scenario == "use_after_marker":
            pre_cad.write_text("{{}}")
            time.sleep(3)
        else:
            pre_cad.write_text("{{}}")

    class R:
        returncode = 0
    return R()

def fake_run_batch(group, n, dry_run=False):
    log(["batch_start", group])
    time.sleep(5)
    log(["batch_done", group])
    return {{"accepted": 1, "refused": [], "group": group, "requested": n,
            "generated": 1, "seconds": 0.1, "prompt_tokens": 1, "completion_tokens": 1}}

def fake_ensure_resident_up():
    log(["ensure_resident_up"])

specgen._run_arms = fake_run_arms
specgen.run_batch = fake_run_batch
specgen._ensure_resident_up = fake_ensure_resident_up
specgen._default_arm = lambda: "test-arm"

sys.argv = ["specgen.py", "--total", "1000000", "--max-batches", "3"]
rc = specgen.main()
log(["exit", rc])
sys.exit(rc)
"""


def _spawn_main(call_log: Path, cad_json: Path, env_path: Path, scenario: str) -> subprocess.Popen:
    code = _MAIN_HELPER.format(repo=str(HERE), cad_json=str(cad_json), env_path=str(env_path))
    return subprocess.Popen([sys.executable, "-c", code, str(call_log), scenario],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _read_tags(call_log: Path) -> list:
    if not call_log.exists():
        return []
    return [json.loads(line)[0] for line in call_log.read_text().splitlines() if line.strip()]


def _read_entries(call_log: Path) -> list:
    return [json.loads(line) for line in call_log.read_text().splitlines() if line.strip()]


def test_sigterm_mid_batch_stops_the_run_promptly_with_one_restore_call(tmp_path):
    """(a) the CRITICAL regression from fix round 3's re-review: a single SIGTERM
    landing while a batch is generating must stop the run on that one signal -- no
    further batches, restore called exactly once, non-zero exit, reason on the last
    printed line."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "mid_batch")
    try:
        time.sleep(0.4)   # "use" already returned; child is asleep inside fake run_batch
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "batch_start", "restore", "exit"], tags
    entries = _read_entries(call_log)
    assert entries[-1][1] != 0   # logged exit code is non-zero
    assert rc != 0
    last_line = [ln for ln in err.decode().splitlines() if ln.strip()][-1]
    assert last_line.startswith("specgen abort:")
    assert "signal" in last_line


def test_sighup_mid_batch_stops_the_run_the_same_way(tmp_path):
    """(d) same as (a), SIGHUP instead of SIGTERM."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "mid_batch")
    try:
        time.sleep(0.4)
        proc.send_signal(signal.SIGHUP)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "batch_start", "restore", "exit"], tags
    assert rc != 0


def test_signal_before_use_writes_its_marker_skips_restore_and_leaves_cad_json_alone(tmp_path):
    """(b) the launch-window trap: a signal landing while `use` is still in progress and
    has not yet written its .pre-arm marker must result in NO restore call and an
    untouched cad.json -- calling restore here would hit cmd_restore's no-marker
    fallback and write maker.enabled: false, corrupting a configuration this run never
    touched."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    placeholder = '{"untouched": true}'
    cad_json.write_text(placeholder)
    proc = _spawn_main(call_log, cad_json, env_path, "use_before_marker")
    try:
        time.sleep(0.4)   # inside "use"'s 3s sleep, well before any marker is written
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "ensure_resident_up", "exit"], tags
    assert "restore" not in tags
    assert rc != 0
    assert cad_json.read_text() == placeholder
    assert not (cad_json.with_suffix(".json.pre-arm")).exists()


def test_signal_after_use_writes_its_marker_restores_exactly_once(tmp_path):
    """(c) a signal landing during `use`, but AFTER its .pre-arm marker already exists
    on disk, must still result in exactly one restore call -- the marker is real
    evidence that something was changed and needs undoing."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "use_after_marker")
    try:
        time.sleep(0.4)   # marker already written; inside "use"'s remaining sleep
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "restore", "exit"], tags
    assert tags.count("restore") == 1
    assert rc != 0

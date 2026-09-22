"""Tests for lab/_armwindow.py -- the arm bookend factored out of lab/specgen.py (Task 3
ruling "One implementation of the arm bookend", 2026-09-19).

Three layers, same shape as tests/test_lab_specgen_signals.py (which this module's copied
code was proven correct by, four review rounds deep):

1. Primitive signal-handling tests against a tiny standalone helper script -- the same
   isolated proof specgen's fix-round-2 tests gave: without _install_signal_handlers(),
   SIGTERM/SIGHUP kill the process outright with no `finally` executed.
2. In-process tests of arm_window()'s CAD_KEEP_MAKER and restore-vs-ensure-resident-up
   branching, with _run_arms/_ensure_resident_up/_pre_arm_marker_exists monkeypatched
   directly (no subprocess needed: no signal is involved in these).
3. Real-subprocess tests driving the REAL lab._armwindow.arm_window() context manager in a
   separate process, with only _run_arms and _ensure_resident_up patched (inside the
   spawned subprocess itself) and a real SIGTERM/SIGHUP sent from this test process --
   look-alike helpers are not accepted per the Task 3 rulings ("that is how two defects
   got through a review round on Task 2"), so this drives arm_window() itself, not a
   restatement of what it is supposed to do.

All tests here are offline (no network, no GPU, no services) and complete in well under 15
seconds total.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from lab import _armwindow as aw  # noqa: E402


def _child_env() -> dict:
    env = dict(os.environ)
    env["CAD_GPU_WINDOW"] = "1"
    return env


# ---------------------------------------------------------------------------
# Layer 1: signal-handling primitives (mirrors test_lab_specgen_signals.py's fix-round-2
# tests, against lab._armwindow instead of lab.specgen).
# ---------------------------------------------------------------------------

_HELPER = """
import sys, time
sys.path.insert(0, {repo!r})
from lab import _armwindow as aw

marker = sys.argv[1]
cleanup_delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

aw._install_signal_handlers()
rc = 0
try:
    time.sleep(10)
except aw.SpecgenAborted:
    rc = 1
finally:
    aw._mark_cleanup_started()
    if cleanup_delay:
        time.sleep(cleanup_delay)
    with open(marker, "w") as f:
        f.write("done")
sys.exit(rc)
"""


def _spawn(marker_path: Path, cleanup_delay: float = 0.0) -> subprocess.Popen:
    code = _HELPER.format(repo=str(HERE))
    return subprocess.Popen([sys.executable, "-c", code, str(marker_path), str(cleanup_delay)],
                            env=_child_env())


def _wait_and_cleanup(proc: subprocess.Popen, timeout: float = 4.0) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise


def test_sigterm_runs_the_finally_block_and_exits_nonzero(tmp_path):
    marker = tmp_path / "marker.txt"
    proc = _spawn(marker)
    try:
        time.sleep(0.4)
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
    marker = tmp_path / "marker.txt"
    proc = _spawn(marker, cleanup_delay=1.0)
    try:
        time.sleep(0.4)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)   # must be ignored, not interrupt the cleanup
        rc = _wait_and_cleanup(proc, timeout=4.0)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert marker.exists(), "cleanup was interrupted by the second signal"
    assert marker.read_text() == "done"
    assert rc != 0


# ---------------------------------------------------------------------------
# Layer 2: in-process arm_window() branching (no signal involved -- pure control flow).
# ---------------------------------------------------------------------------

def test_arm_window_normal_exit_restores_when_marker_exists(tmp_path, monkeypatch):
    calls = []

    def fake_run_arms(*args):
        calls.append(list(args))
        class R:
            returncode = 0
        return R()

    def fake_marker_exists():
        return True

    def fake_ensure_resident_up():
        calls.append(["ensure_resident_up"])

    monkeypatch.setattr(aw, "_run_arms", fake_run_arms)
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", fake_marker_exists)
    monkeypatch.setattr(aw, "_ensure_resident_up", fake_ensure_resident_up)

    with aw.arm_window("gemma-4-31b") as arm:
        assert arm == "gemma-4-31b"
        calls.append(["body"])

    assert calls == [["use", "gemma-4-31b"], ["body"], ["restore"]]


def test_arm_window_normal_exit_ensures_resident_when_no_marker(tmp_path, monkeypatch):
    calls = []

    def fake_run_arms(*args):
        calls.append(list(args))
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(aw, "_run_arms", fake_run_arms)
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: calls.append(["ensure_resident_up"]))

    with aw.arm_window("gemma-4-31b"):
        pass

    assert calls == [["use", "gemma-4-31b"], ["ensure_resident_up"]]
    assert "restore" not in [c[0] for c in calls]


def test_arm_window_uses_default_arm_when_none_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(aw, "_default_arm", lambda: "default-arm")

    def fake_run_arms(*args):
        seen["arm"] = args[1] if args[0] == "use" else seen.get("arm")
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(aw, "_run_arms", fake_run_arms)
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: None)

    with aw.arm_window() as arm:
        assert arm == "default-arm"
    assert seen["arm"] == "default-arm"


def test_arm_window_raises_when_use_fails_and_still_cleans_up(monkeypatch):
    calls = []

    def fake_run_arms(*args):
        calls.append(list(args))
        class R:
            returncode = 1
        return R()

    monkeypatch.setattr(aw, "_run_arms", fake_run_arms)
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: calls.append(["ensure_resident_up"]))

    with pytest.raises(RuntimeError, match="scripts/arms.py use bad-arm failed"):
        with aw.arm_window("bad-arm"):
            calls.append(["body should never run"])

    assert calls == [["use", "bad-arm"], ["ensure_resident_up"]]


def test_arm_window_body_exception_propagates_after_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(aw, "_run_arms", lambda *a: calls.append(list(a)) or
                        type("R", (), {"returncode": 0})())
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: True)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: None)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with aw.arm_window("x"):
            raise Boom("body failed")

    assert ["restore"] in calls


def test_arm_window_restores_cad_keep_maker_env(monkeypatch):
    monkeypatch.setattr(aw, "_run_arms", lambda *a: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: None)
    monkeypatch.delenv("CAD_KEEP_MAKER", raising=False)

    with aw.arm_window("x"):
        assert os.environ.get("CAD_KEEP_MAKER") == "1"
    assert os.environ.get("CAD_KEEP_MAKER") is None


def test_arm_window_leaves_a_pre_existing_cad_keep_maker_alone(monkeypatch):
    monkeypatch.setattr(aw, "_run_arms", lambda *a: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(aw, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(aw, "_ensure_resident_up", lambda: None)
    monkeypatch.setenv("CAD_KEEP_MAKER", "already-set")

    with aw.arm_window("x"):
        pass
    assert os.environ.get("CAD_KEEP_MAKER") == "already-set"


# ---------------------------------------------------------------------------
# Layer 3: real-subprocess tests driving the REAL arm_window() context manager, with only
# _run_arms/_ensure_resident_up/_default_arm patched inside the spawned subprocess. A
# temp CAD_CONFIG_FILE/MAKER_ENV point _pre_arm_marker_paths() (which imports scripts/
# arms.py's own CAD_JSON/ENV_PATH) at a throwaway directory, so nothing under the real
# ~/.openclaw/ is ever touched.
# ---------------------------------------------------------------------------

_MAIN_HELPER = """
import json, os, sys, time
sys.path.insert(0, {repo!r})
os.environ["CAD_CONFIG_FILE"] = {cad_json!r}
os.environ["MAKER_ENV"] = {env_path!r}
from lab import _armwindow as aw
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
        elif scenario == "use_fails":
            class R:
                returncode = 1
            return R()
        else:
            pre_cad.write_text("{{}}")

    class R:
        returncode = 0
    return R()

def fake_ensure_resident_up():
    log(["ensure_resident_up"])

aw._run_arms = fake_run_arms
aw._ensure_resident_up = fake_ensure_resident_up
aw._default_arm = lambda: "test-arm"

rc = 0
try:
    with aw.arm_window("test-arm"):
        log(["body_start"])
        time.sleep(5)
        log(["body_done"])
except aw.SpecgenAborted:
    rc = 1
except Exception as e:
    log(["error", str(e)])
    rc = 2
log(["exit", rc])
sys.exit(rc)
"""


def _spawn_main(call_log: Path, cad_json: Path, env_path: Path, scenario: str) -> subprocess.Popen:
    code = _MAIN_HELPER.format(repo=str(HERE), cad_json=str(cad_json), env_path=str(env_path))
    return subprocess.Popen([sys.executable, "-c", code, str(call_log), scenario],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_child_env())


def _read_tags(call_log: Path) -> list:
    if not call_log.exists():
        return []
    return [json.loads(line)[0] for line in call_log.read_text().splitlines() if line.strip()]


def test_sigterm_mid_body_stops_promptly_with_one_restore_call(tmp_path):
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "mid_body")
    try:
        time.sleep(0.4)   # "use" already returned; child is asleep inside the body
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "body_start", "restore", "exit"], tags
    assert rc != 0


def test_sighup_mid_body_stops_the_same_way(tmp_path):
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "mid_body")
    try:
        time.sleep(0.4)
        proc.send_signal(signal.SIGHUP)
        out, err = proc.communicate(timeout=4.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
    tags = _read_tags(call_log)
    assert tags == ["use", "body_start", "restore", "exit"], tags
    assert rc != 0


def test_signal_before_use_marker_skips_restore_and_leaves_cad_json_alone(tmp_path):
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


def test_signal_after_use_marker_restores_exactly_once(tmp_path):
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


def test_use_returncode_nonzero_raises_and_ensures_resident(tmp_path):
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    proc = _spawn_main(call_log, cad_json, env_path, "use_fails")
    out, err = proc.communicate(timeout=4.0)
    rc = proc.returncode
    tags = _read_tags(call_log)
    assert tags == ["use", "ensure_resident_up", "error", "exit"], tags
    assert rc != 0

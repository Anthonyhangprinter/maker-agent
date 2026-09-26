"""GPU-busy graceful handling for Satine (integration/cad-telegram.py). See
~/.openclaw/gpu/DESIGN-gpu-busy.md #6: on enqueue, a GPU busy with someone else gets an
instant notice; a "starting now" ping fires only once the engine actually gets the build
lock; a timeout while still waiting names the holder.

Every network call (Telegram, the notice server, the engine subprocess) is faked — no
real bot token, no GPU, no build lock. Common-context rule exercised here: ANY failure
fetching /state fails OPEN to {"state": "ok"}, never raises, never blocks.

Run: python3 -m pytest tests/test_cad_telegram_gpu_busy.py -q
"""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "cad_telegram", HERE / "integration" / "cad-telegram.py")
ct = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ct
_spec.loader.exec_module(ct)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


# ── _fetch_gpu_state: fail-open ─────────────────────────────────────────────────

def test_fetch_gpu_state_fails_open_on_connection_error(monkeypatch):
    def boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(ct.urllib.request, "urlopen", boom)
    assert ct._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_timeout(monkeypatch):
    def boom(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ct.urllib.request, "urlopen", boom)
    assert ct._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_bad_json(monkeypatch):
    monkeypatch.setattr(ct.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(b"not json"))
    assert ct._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_missing_state_key(monkeypatch):
    monkeypatch.setattr(ct.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(
                            json.dumps({"holder": "lab"}).encode()))
    assert ct._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_passes_through_a_real_busy_payload(monkeypatch):
    payload = {"state": "busy", "holder": "lab", "label": "a lab job",
               "since_hhmm": "17:29", "progress": {"done": 163, "total": 226, "pct": 72},
               "message": "The home GPU is busy with a lab job."}
    monkeypatch.setattr(ct.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(json.dumps(payload).encode()))
    assert ct._fetch_gpu_state() == payload


def test_fetch_gpu_state_uses_the_common_context_2s_timeout(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse(json.dumps({"state": "ok"}).encode())

    monkeypatch.setattr(ct.urllib.request, "urlopen", fake_urlopen)
    ct._fetch_gpu_state()
    assert seen["timeout"] == ct.GPU_STATE_TIMEOUT_S <= 2


# ── message-text helpers (pure) ──────────────────────────────────────────────────

def test_gpu_busy_notice_names_label_and_since():
    msg = ct._gpu_busy_notice({"state": "busy", "holder": "lab", "label": "a lab job",
                               "since_hhmm": "17:29"})
    assert msg == ("GPU busy with a lab job since 17:29. You're queued, I'll message you "
                  "when your build starts.")
    assert "—" not in msg


def test_gpu_busy_notice_degrades_without_since():
    msg = ct._gpu_busy_notice({"state": "busy", "holder": "lab", "label": "a lab job"})
    assert msg == "GPU busy with a lab job. You're queued, I'll message you when your build starts."


def test_gpu_busy_notice_falls_back_when_label_missing():
    msg = ct._gpu_busy_notice({"state": "busy", "holder": "lab"})
    assert "another job" in msg


def test_gpu_timeout_notice_names_the_holder():
    msg = ct._gpu_timeout_notice({"state": "busy", "holder": "lab", "label": "a lab job"})
    assert msg == "Build timed out while the GPU was busy with a lab job."
    assert "—" not in msg


# ── enqueue(): notice only for a non-Satine holder ───────────────────────────────

def _patch_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "JOURNAL_FILE", str(tmp_path / "journal.json"))


def test_enqueue_sends_notice_when_gpu_busy_with_another_frontend(monkeypatch, tmp_path):
    _patch_journal(monkeypatch, tmp_path)
    monkeypatch.setattr(ct, "_fetch_gpu_state", lambda: {
        "state": "busy", "holder": "lab", "label": "a lab job", "since_hhmm": "17:29"})
    sent = []
    monkeypatch.setattr(ct, "send", lambda token, chat_id, text: sent.append(text))
    job = {"id": 1, "chat_id": 42, "kind": "build", "text": "a gear"}
    ct.enqueue("tok", job)
    assert job["gpu_notice_sent"] is True
    assert any("GPU busy with a lab job" in t for t in sent)


def test_enqueue_no_notice_when_gpu_ok(monkeypatch, tmp_path):
    _patch_journal(monkeypatch, tmp_path)
    monkeypatch.setattr(ct, "_fetch_gpu_state", lambda: {"state": "ok"})
    sent = []
    monkeypatch.setattr(ct, "send", lambda token, chat_id, text: sent.append(text))
    job = {"id": 2, "chat_id": 42, "kind": "build", "text": "a gear"}
    ct.enqueue("tok", job)
    assert "gpu_notice_sent" not in job
    assert not any("GPU busy" in t for t in sent)


def test_enqueue_no_notice_when_holder_is_another_satine_build(monkeypatch, tmp_path):
    """Satine's OWN "Queued — N request(s) ahead" message already covers this case; the
    GPU-busy notice is for a NON-CAD frontend (lab/web/cli/benchmark), not another Satine
    build queued behind itself."""
    _patch_journal(monkeypatch, tmp_path)
    monkeypatch.setattr(ct, "_fetch_gpu_state", lambda: {
        "state": "busy", "holder": "telegram", "label": "a Satine CAD build"})
    sent = []
    monkeypatch.setattr(ct, "send", lambda token, chat_id, text: sent.append(text))
    job = {"id": 3, "chat_id": 42, "kind": "build", "text": "a gear"}
    ct.enqueue("tok", job)
    assert "gpu_notice_sent" not in job
    assert not any("GPU busy" in t for t in sent)


# ── run_v5_build: streaming wait/start detection ─────────────────────────────────

class _FakeProc:
    """Stands in for subprocess.Popen: stderr yields the given lines, stdout/communicate
    return the given result once "started"."""
    def __init__(self, stderr_lines, stdout_text, delay=0.0):
        self._stderr_lines = stderr_lines
        self.stderr = iter(stderr_lines)
        self._stdout_text = stdout_text
        self._delay = delay

    def communicate(self, timeout=None):
        if self._delay:
            time.sleep(self._delay)
        return self._stdout_text, ""

    def kill(self):
        pass


def test_run_v5_build_fires_on_started_after_waiting_for_lock(monkeypatch):
    result_json = json.dumps({"ok": True, "url": "https://cad.onshape.com/x"})
    stderr = ["waiting for build lock", "waiting for build lock", "now building"]
    monkeypatch.setattr(ct.subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(stderr, result_json + "\n"))
    fired = []
    result, err, timed_out_waiting = ct.run_v5_build(
        "a gear", "auto", on_started=lambda: fired.append(True))
    assert result == {"ok": True, "url": "https://cad.onshape.com/x"}
    assert timed_out_waiting is False
    assert fired == [True]


def test_run_v5_build_fires_on_started_even_when_it_never_waited(monkeypatch):
    """The common case: the GPU was free, the build never printed the wait line at all —
    on_started still fires once, at the end, so a caller that unconditionally wants to know
    the build ran gets a call (Satine only wires this up when notify_start is True)."""
    result_json = json.dumps({"ok": True})
    monkeypatch.setattr(ct.subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(["building now"], result_json + "\n"))
    fired = []
    result, err, timed_out_waiting = ct.run_v5_build(
        "a gear", "auto", on_started=lambda: fired.append(True))
    assert result == {"ok": True}
    assert timed_out_waiting is False
    assert fired == [True]


def test_run_v5_build_never_fires_on_started_without_a_callback(monkeypatch):
    result_json = json.dumps({"ok": True})
    monkeypatch.setattr(ct.subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(["waiting for build lock"], result_json + "\n"))
    result, err, timed_out_waiting = ct.run_v5_build("a gear", "auto")  # on_started=None
    assert result == {"ok": True}


def test_run_v5_build_reports_timed_out_waiting_true_on_timeout_while_waiting(monkeypatch):
    class _HangingProc(_FakeProc):
        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    monkeypatch.setattr(ct.subprocess, "Popen",
                        lambda *a, **kw: _HangingProc(["waiting for build lock"] * 5, ""))
    result, err, timed_out_waiting = ct.run_v5_build("a gear", "auto", timeout=1)
    assert result is None
    assert err == "Build timed out."
    assert timed_out_waiting is True


def test_run_v5_build_on_started_exception_never_propagates(monkeypatch):
    """A caller's callback failing (e.g. a dead Telegram send) must not break the build."""
    result_json = json.dumps({"ok": True})
    monkeypatch.setattr(ct.subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(["waiting for build lock", "go"],
                                                   result_json + "\n"))

    def boom():
        raise RuntimeError("telegram is down")

    result, err, timed_out_waiting = ct.run_v5_build("a gear", "auto", on_started=boom)
    assert result == {"ok": True}

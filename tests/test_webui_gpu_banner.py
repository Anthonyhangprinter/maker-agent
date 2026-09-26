"""GPU-busy banner tests for webui/app.py (see ~/.openclaw/gpu/DESIGN-gpu-busy.md).

Covers the one pure helper this change adds, `_fetch_gpu_state()`, and the `/api/gpu`
route that wraps it. Every network call is faked — no real notice server, no GPU, no
build lock. Common-context rule: ANY failure fetching /state (connection refused,
timeout, bad JSON, missing/wrong-shaped keys) must fail OPEN to {"state": "ok"}, never
raise, never block, never show busy on a hunch.

Run: python3 -m pytest tests/test_webui_gpu_banner.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app.app)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_fetch_gpu_state_fails_open_on_connection_error(monkeypatch):
    def boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(app.urllib.request, "urlopen", boom)
    assert app._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_timeout(monkeypatch):
    def boom(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(app.urllib.request, "urlopen", boom)
    assert app._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_bad_json(monkeypatch):
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(b"not json"))
    assert app._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_missing_state_key(monkeypatch):
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(json.dumps(
                            {"holder": "lab", "message": "busy"}).encode()))
    assert app._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_fails_open_on_a_wrong_shaped_state_value(monkeypatch):
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(json.dumps(
                            {"state": "definitely-busy"}).encode()))
    assert app._fetch_gpu_state() == {"state": "ok"}


def test_fetch_gpu_state_passes_through_a_real_busy_payload(monkeypatch):
    payload = {"state": "busy", "holder": "lab", "label": "a lab job",
               "spec": None, "since": "2026-09-26T17:29:39+10:00",
               "since_hhmm": "17:29", "progress": {"done": 163, "total": 226, "pct": 72},
               "message": "The home GPU is busy with a lab job (since 17:29, 72% done). "
                          "Chat is paused and will come back on its own.",
               "checked": "2026-09-26T17:31:00+10:00"}
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(json.dumps(payload).encode()))
    assert app._fetch_gpu_state() == payload


def test_fetch_gpu_state_uses_a_short_timeout(monkeypatch):
    """Common-context rule: a 2 s timeout, never the app's own longer HTTP defaults —
    a wedged notice server must not stall a page load or the title worker."""
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse(json.dumps({"state": "ok"}).encode())

    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
    app._fetch_gpu_state()
    assert seen["timeout"] == app.GPU_STATE_TIMEOUT_S <= 2


def test_api_gpu_route_returns_ok_on_failure(monkeypatch):
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: (_ for _ in ()).throw(OSError("refused")))
    r = client.get("/api/gpu")
    assert r.status_code == 200
    assert r.json() == {"state": "ok"}


def test_api_gpu_route_passes_through_busy(monkeypatch):
    payload = {"state": "busy", "holder": "lab", "label": "a lab job",
               "since_hhmm": "17:29", "progress": None,
               "message": "GPU busy with a lab job since 17:29."}
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(json.dumps(payload).encode()))
    r = client.get("/api/gpu")
    assert r.status_code == 200
    assert r.json() == payload

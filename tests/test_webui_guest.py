"""Guest hardening + owner pings through Morai's bot (webui/app.py).

`tailscale serve` stamps Tailscale-User-Login; loopback requests carry no header and are the
owner. Guests (any login not in OWNER_LOGINS) see and own only their own jobs, get no lab
endpoints and are capped at GUEST_MAX_PENDING pending builds. Everything engine-touching is
faked: jobs are seeded straight into app._jobs and the build queues are drained/never run.

Run: python3 -m pytest tests/test_webui_guest.py -q
"""
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))


class _FakeStdout:
    def read(self):
        return json.dumps({"ok": True})


class _FakePopen:
    def __init__(self, *a, **kw):
        self.stderr = iter([])
        self.stdout = _FakeStdout()

    def wait(self, timeout=None):
        return 0

    def communicate(self, timeout=None):
        return json.dumps({"ok": True}), ""

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_REAL_POPEN = subprocess.Popen


def setup_module(_module):
    subprocess.Popen = _FakePopen


def teardown_module(_module):
    subprocess.Popen = _REAL_POPEN


import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app.app)

ALICE = {"Tailscale-User-Login": "Alice@Example.com", "Tailscale-User-Name": "Alice"}
BOB = {"Tailscale-User-Login": "bob@example.com", "Tailscale-User-Name": "Bob"}
OWNER = {"Tailscale-User-Login": "anthonyromanelli10@gmail.com"}


def setup_function(_fn):
    app._jobs.clear()
    app._gpu_busy.clear()
    while not app._queue.empty():
        app._queue.get_nowait()
    while not app._mesh_queue.empty():
        app._mesh_queue.get_nowait()


def _seed(job_id, login, status="done", build_dir=None, kind=None):
    job = {
        "id": job_id, "spec": f"spec {job_id}", "coder": "auto", "image": None,
        "user": login or "local", "guest": bool(login), "login": login,
        "status": status, "log": deque(maxlen=300), "result": None, "error": None,
        "created_at": time.time(), "updated_at": time.time(),
    }
    if kind:
        job["kind"] = kind
    if build_dir:
        job["result"] = {"build_dir_fs": str(build_dir)}
    app._jobs[job_id] = job
    return job


# ── listing / reading / deleting ─────────────────────────────────────────────────

def test_owner_sees_all_guest_sees_own():
    _seed("a1", "alice@example.com")
    _seed("b1", "bob@example.com")
    _seed("o1", "")
    assert {j["id"] for j in client.get("/api/jobs").json()} == {"a1", "b1", "o1"}
    assert {j["id"] for j in client.get("/api/jobs", headers=OWNER).json()} == {"a1", "b1", "o1"}
    assert {j["id"] for j in client.get("/api/jobs", headers=ALICE).json()} == {"a1"}


def test_guest_get_other_job_is_404():
    _seed("b1", "bob@example.com")
    _seed("o1", "")
    assert client.get("/api/jobs/b1", headers=ALICE).status_code == 404
    assert client.get("/api/jobs/o1", headers=ALICE).status_code == 404
    assert client.get("/api/jobs/b1", headers=BOB).status_code == 200
    assert client.get("/api/jobs/b1").status_code == 200


def test_guest_delete_other_403_own_ok():
    _seed("b1", "bob@example.com")
    _seed("a1", "alice@example.com")
    assert client.delete("/api/jobs/b1", headers=ALICE).status_code == 403
    assert "b1" in app._jobs
    assert client.delete("/api/jobs/a1", headers=ALICE).status_code == 200
    assert "a1" not in app._jobs
    assert client.delete("/api/jobs/b1").status_code == 200      # owner can delete anything


def test_guest_chat_and_rescale_on_foreign_job_403():
    _seed("b1", "bob@example.com", build_dir="/nonexistent/b")
    _seed("o1", "", build_dir="/nonexistent/o")
    for jid in ("b1", "o1"):
        r = client.post("/api/chat", json={"job_id": jid, "message": "hi"}, headers=ALICE)
        assert r.status_code == 403
        r = client.post("/api/rescale", json={"job_id": jid, "params": {"x": 1}}, headers=ALICE)
        assert r.status_code == 403
    r = client.post("/api/rescale", json={"build_id": "scad_123", "params": {}}, headers=ALICE)
    assert r.status_code == 403


# ── lab ──────────────────────────────────────────────────────────────────────────

def test_lab_endpoints_forbidden_to_guests():
    for path in ("/api/lab/card", "/api/lab/benchcad", "/api/lab/lift"):
        assert client.get(path, headers=ALICE).status_code == 403
        assert client.get(path).status_code in (200, 404)        # owner path unchanged
        assert client.get(path, headers=OWNER).status_code in (200, 404)


# ── artifacts / meshes ───────────────────────────────────────────────────────────

def test_guest_artifacts_only_for_own_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "BUILDS_DIR", tmp_path.resolve())
    for bid in ("build_a", "build_b"):
        (tmp_path / bid).mkdir()
        (tmp_path / bid / "build.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    _seed("a1", "alice@example.com", build_dir=tmp_path / "build_a")
    _seed("b1", "bob@example.com", build_dir=tmp_path / "build_b")
    assert client.get("/artifacts/build_b/build.png", headers=ALICE).status_code == 404
    assert client.get("/artifacts/build_a/build.png", headers=ALICE).status_code == 200
    assert client.get("/artifacts/build_b/build.png").status_code == 200
    assert client.get("/artifacts/build_b/build.png", headers=OWNER).status_code == 200


def test_guest_meshes_only_for_own_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(app.meshmod, "MESHES_DIR", tmp_path.resolve())
    for jid in ("m_a", "m_b"):
        (tmp_path / jid).mkdir()
        (tmp_path / jid / "mesh.glb").write_bytes(b"glTF")
    _seed("m_a", "alice@example.com", kind="mesh")
    _seed("m_b", "bob@example.com", kind="mesh")
    assert client.get("/meshes/m_b/mesh.glb", headers=ALICE).status_code == 404
    assert client.get("/meshes/m_a/mesh.glb", headers=ALICE).status_code == 200
    assert client.get("/meshes/m_b/mesh.glb").status_code == 200


# ── pending cap ──────────────────────────────────────────────────────────────────

def test_guest_pending_cap_returns_429_on_fourth(monkeypatch):
    # nothing may run: the worker thread would otherwise pick the id up
    monkeypatch.setattr(app._queue, "put", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "_notify", lambda *a, **k: None)
    for i in range(3):
        r = client.post("/api/build", data={"spec": f"box {i}"}, headers=ALICE)
        assert r.status_code == 200, r.text
    r = client.post("/api/build", data={"spec": "box 4"}, headers=ALICE)
    assert r.status_code == 429
    assert "3 builds already pending" in r.json()["detail"]
    # another guest and the owner are unaffected
    assert client.post("/api/build", data={"spec": "x"}, headers=BOB).status_code == 200
    for i in range(5):
        assert client.post("/api/build", data={"spec": f"o{i}"}).status_code == 200
    # the stored login is lowercased for guests, empty for the owner
    logins = {j["login"] for j in app._jobs.values()}
    assert logins == {"alice@example.com", "bob@example.com", ""}


def test_guest_rescale_respects_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(app._queue, "put", lambda *_a, **_k: None)
    (tmp_path / "bd").mkdir()
    (tmp_path / "bd" / "build_source.py").write_text("W = 10\n")
    _seed("a1", "alice@example.com", build_dir=tmp_path / "bd")
    for i in range(3):
        _seed(f"q{i}", "alice@example.com", status="queued")
    r = client.post("/api/rescale", json={"job_id": "a1", "params": {"W": 12}}, headers=ALICE)
    assert r.status_code == 429


# ── notifications ────────────────────────────────────────────────────────────────

class _Sync:
    """Run the _notify thread inline so the test can inspect what was sent."""
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


def _capture(monkeypatch):
    sent, bus = [], []
    monkeypatch.setattr(app.threading, "Thread", _Sync)

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.data))

    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(app, "_bus_post", lambda text: bus.append(text))
    return sent, bus


def test_notify_prefers_morai_token(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\n\nTELEGRAM_BOT_TOKEN="morai-tok"\nTELEGRAM_HOME_CHANNEL=4242\n')
    cfg = tmp_path / "openclaw.json"
    cfg.write_text(json.dumps({"channels": {"telegram": {"accounts": {
        "cad": {"botToken": "satine-tok"}}}}}))
    monkeypatch.setattr(app, "HERMES_ENV", env)
    monkeypatch.setattr(app, "OPENCLAW_CONFIG", cfg)
    sent, bus = _capture(monkeypatch)
    app._notify("CAD web (guest): hello")
    assert len(sent) == 1
    assert "botmorai-tok/sendMessage" in sent[0][0]
    assert b"chat_id=4242" in sent[0][1]
    assert bus == ["CAD web (guest): hello"]


def test_notify_falls_back_to_satine(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=morai-tok\n")          # chat id missing
    cfg = tmp_path / "openclaw.json"
    cfg.write_text(json.dumps({"channels": {"telegram": {"accounts": {
        "cad": {"botToken": "satine-tok"}}}}}))
    monkeypatch.setattr(app, "HERMES_ENV", env)
    monkeypatch.setattr(app, "OPENCLAW_CONFIG", cfg)
    sent, bus = _capture(monkeypatch)
    app._notify("hello")
    assert "botsatine-tok/sendMessage" in sent[0][0]
    assert f"chat_id={app.OWNER_CHAT_ID}".encode() in sent[0][1]
    assert bus == ["hello"]
    # no env file at all and no config: nothing sent, bus note still recorded, no raise
    monkeypatch.setattr(app, "HERMES_ENV", tmp_path / "missing.env")
    monkeypatch.setattr(app, "OPENCLAW_CONFIG", tmp_path / "missing.json")
    sent.clear()
    app._notify("again")
    assert sent == [] and bus[-1] == "again"


def test_bus_post_uses_topic_cad_to_morai(tmp_path, monkeypatch):
    lib = tmp_path / "fake_bus_lib.py"
    out = tmp_path / "calls.json"
    lib.write_text("import json\n"
                   "def post(from_agent, topic, text, to=None):\n"
                   f"    open({str(out)!r}, 'w').write(json.dumps([from_agent, topic, text, to]))\n")
    monkeypatch.setattr(app, "BUS_LIB", lib)
    app._bus_post("note")
    assert json.loads(out.read_text()) == ["cad-web", "cad", "note", "morai"]
    monkeypatch.setattr(app, "BUS_LIB", tmp_path / "nope.py")
    app._bus_post("ignored")                                  # missing lib: silent
    lib.write_text("raise RuntimeError('boom')\n")
    monkeypatch.setattr(app, "BUS_LIB", lib)
    app._bus_post("ignored")                                  # broken lib: silent


def test_guest_messages_are_prefixed(monkeypatch):
    msgs = []
    monkeypatch.setattr(app, "_notify", lambda text, photo=None: msgs.append(text))
    monkeypatch.setattr(app._queue, "put", lambda *_a, **_k: None)
    client.post("/api/build", data={"spec": "a bracket"}, headers=ALICE)
    assert msgs and msgs[0].startswith("CAD web (guest):")
    app._ping_outcome({"status": "done", "user": "x", "spec": "s"})
    assert msgs[-1].startswith("CAD web (guest):")

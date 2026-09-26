"""API-shape tests for the design assistant's web endpoints (/api/assist, /api/build's new
`parameters` field, /api/rescale's new build123d job_id path). FastAPI's TestClient, every
engine-touching call faked — no subprocess ever actually runs cad_engine, no network, no
GPU, no build lock.

webui/app.py starts its worker threads as an import-time side effect (existing design,
unrelated to this change): a queued job is picked up by a background thread that calls the
real `subprocess.Popen`. `subprocess.Popen` is process-global (there is only one `subprocess`
module), so patching it for the DURATION OF THIS FILE'S TESTS only, via setup_module/
teardown_module, matters: other test files (test_fluid_rescale.py in particular) run real
build123d subprocesses in the SAME pytest session and must get the real Popen back the
moment this module's tests are done, or their "real geometry" assertions silently pass
against a fake.

Run: python3 -m pytest tests/test_webui_assist.py -q
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))


class _FakeStdout:
    """`_run_build`'s stdout-draining thread only ever calls `.read()` once (fix round 1:
    it no longer shares `proc.communicate()` with the stderr thread)."""
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


class _FakePopen:
    """A real (not Mock-based) stand-in for subprocess.Popen, context-manager-correct,
    so the worker thread's `_run_build` never shells out to the real engine while active."""
    def __init__(self, *a, **kw):
        self.stderr = iter([])
        self.stdout = _FakeStdout(json.dumps({"ok": True}))

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


def setup_function(_fn):
    app._jobs.clear()
    app._gpu_busy.clear()
    while not app._queue.empty():
        app._queue.get_nowait()


# ── /api/assist ──────────────────────────────────────────────────────────────────

def test_assist_requires_a_spec():
    r = client.post("/api/assist", json={"spec": "  "})
    assert r.status_code == 400


def test_assist_rejects_an_unknown_mode():
    r = client.post("/api/assist", json={"spec": "a bracket", "mode": "sometimes"})
    assert r.status_code == 400


def test_assist_returns_409_while_the_gpu_is_busy():
    app._gpu_busy.set()
    try:
        r = client.post("/api/assist", json={"spec": "a bracket", "mode": "always"})
        assert r.status_code == 409
    finally:
        app._gpu_busy.clear()


def test_assist_returns_409_while_a_build_is_queued(monkeypatch):
    # A real _queue.put() races against the live worker thread, which drains it near-
    # instantly with the fake Popen above — patch `empty()` instead for a deterministic
    # "something is queued" signal.
    monkeypatch.setattr(app._queue, "empty", lambda: False)
    r = client.post("/api/assist", json={"spec": "a bracket", "mode": "always"})
    assert r.status_code == 409


def test_assist_passes_through_the_subprocess_proposal(monkeypatch):
    proposal = {"proposed": True, "mode": "always", "title": "Bracket", "summary": "x",
                "parameters": [{"key": "wall", "label": "Wall", "value": 2, "unit": "mm",
                                 "min": 1, "max": 6, "step": 0.5, "source": "user", "why": "x"}],
                "features": []}

    class _Proc:
        stdout = json.dumps(proposal)
        stderr = ""

    monkeypatch.setattr(app.subprocess, "run", lambda *a, **kw: _Proc())
    r = client.post("/api/assist", json={"spec": "a bracket with 2mm walls", "mode": "always"})
    assert r.status_code == 200
    assert r.json() == proposal


def test_assist_502s_on_unparsable_subprocess_output(monkeypatch):
    class _Proc:
        stdout = "not json"
        stderr = "traceback goes here"

    monkeypatch.setattr(app.subprocess, "run", lambda *a, **kw: _Proc())
    r = client.post("/api/assist", json={"spec": "a bracket", "mode": "always"})
    assert r.status_code == 502


# ── /api/build with `parameters` ─────────────────────────────────────────────────

def test_build_rejects_non_json_parameters():
    r = client.post("/api/build", data={"spec": "a bracket", "parameters": "not json"})
    assert r.status_code == 400


def test_build_rejects_parameters_that_are_not_a_list():
    r = client.post("/api/build", data={"spec": "a bracket",
                                        "parameters": json.dumps({"wall": 2})})
    assert r.status_code == 400


def test_build_composes_the_spec_when_parameters_are_given():
    params = [{"key": "wall", "label": "Wall Thickness", "value": 2.0, "unit": "mm",
               "source": "user"}]
    r = client.post("/api/build", data={"spec": "a bracket",
                                        "parameters": json.dumps(params)})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    job = app._jobs[job_id]
    assert job["spec"] == "a bracket"                  # original, unchanged, for display
    assert job["composed_spec"] is not None
    assert "Wall Thickness: 2 mm" in job["composed_spec"]
    assert job["parameters"] == params


def test_build_without_parameters_leaves_composed_spec_unset():
    r = client.post("/api/build", data={"spec": "a plain bracket"})
    assert r.status_code == 200
    job = app._jobs[r.json()["job_id"]]
    assert job["composed_spec"] is None
    assert job["parameters"] is None


# ── /api/rescale (build123d job_id path) ─────────────────────────────────────────

def _done_job(tmp_path, build_source: str = "wall_mm = 2.0\nresult = None\n"):
    build_dir = tmp_path / "build_1"
    build_dir.mkdir()
    (build_dir / "build_source.py").write_text(build_source, encoding="utf-8")
    job = {
        "id": "job1", "spec": "x", "coder": "auto", "image": None,
        "candidates": "", "fewshots": True, "lang": "build123d", "engine_mode": "fluid",
        "composed_spec": None, "parameters": None, "user": "local", "guest": False,
        "status": "done", "log": [], "error": None,
        "result": {"build_dir_fs": str(build_dir), "build_id": "build_1"},
        "created_at": time.time(), "updated_at": time.time(),
    }
    app._jobs["job1"] = job
    return job


def test_rescale_requires_build_id_or_job_id():
    r = client.post("/api/rescale", json={"params": {"wall_mm": 3}})
    assert r.status_code == 400


def test_rescale_404s_for_an_unknown_job():
    r = client.post("/api/rescale", json={"job_id": "nope", "params": {"wall_mm": 3}})
    assert r.status_code == 404


def test_rescale_400s_for_a_mesh_job(tmp_path):
    job = _done_job(tmp_path)
    job["kind"] = "mesh"
    r = client.post("/api/rescale", json={"job_id": "job1", "params": {"wall_mm": 3}})
    assert r.status_code == 400


def test_rescale_409s_when_the_job_has_no_completed_build(tmp_path):
    _done_job(tmp_path)
    app._jobs["job1"]["status"] = "running"
    r = client.post("/api/rescale", json={"job_id": "job1", "params": {"wall_mm": 3}})
    assert r.status_code == 409


def test_rescale_400s_when_not_a_build123d_build(tmp_path):
    build_dir = tmp_path / "no_source"
    build_dir.mkdir()
    app._jobs["job1"] = {
        "id": "job1", "status": "done", "kind": "cad",
        "result": {"build_dir_fs": str(build_dir)},
    }
    r = client.post("/api/rescale", json={"job_id": "job1", "params": {"wall_mm": 3}})
    assert r.status_code == 400


def test_rescale_400s_with_no_valid_numeric_params(tmp_path):
    _done_job(tmp_path)
    r = client.post("/api/rescale", json={"job_id": "job1", "params": {"wall_mm": "abc"}})
    assert r.status_code == 400


def test_rescale_queues_a_pending_rescale_turn(tmp_path):
    _done_job(tmp_path)
    r = client.post("/api/rescale", json={"job_id": "job1", "params": {"wall_mm": 3, "$bad name": 1}})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "job_id": "job1"}


# ── Fix round 2 (GPU-busy re-review): _run_build's non-timeout exception cleanup ────

def test_run_build_non_timeout_exception_kills_and_reaps_the_child(monkeypatch):
    """_run_build had the identical gap as integration/cad-telegram.py's run_v5_build:
    a proc.wait() failure that isn't a timeout had no except branch at all here, so it
    propagated into _worker()'s outer catch-all — which also never kills/reaps the child
    or joins the pump threads. A build subprocess (holding the machine-wide GPU build
    lock) could be left running untracked. Calls _run_build directly, bypassing the
    queue/worker thread, so the fix can be asserted synchronously."""
    class _BrokenPipeProc:
        def __init__(self):
            self.stderr = iter(["waiting for build lock"])
            self.stdout = _FakeStdout("{}")
            self._raised_once = False
            self.kill_calls = 0
            self.wait_after_kill_calls = 0

        def wait(self, timeout=None):
            if not self._raised_once:
                self._raised_once = True
                raise OSError("broken pipe")
            self.wait_after_kill_calls += 1
            return 0

        def kill(self):
            self.kill_calls += 1

    holder = {}

    def fake_popen(*a, **kw):
        p = _BrokenPipeProc()
        holder["proc"] = p
        return p

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    job = {"id": "cleanup-test", "status": "queued", "spec": "a cube", "coder": "auto",
          "image": None, "log": [], "created_at": 0.0, "result": {}}
    app._run_build(job)

    proc = holder["proc"]
    assert proc.kill_calls == 1                # proc.kill() was called from the except branch
    assert proc.wait_after_kill_calls == 1      # the best-effort reap ran after the kill
    assert job["status"] == "error"
    assert job["error"] == "broken pipe"

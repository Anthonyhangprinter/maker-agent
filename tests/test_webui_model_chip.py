"""Model-identity chip tests for webui/app.py (owner request, 2026-09-27): every creation
and every turn must say which model built it, because past builds used different CAD
coders and a revise today can ride a different arm than the one that built the original.

Covers the pure helpers this feature adds (`_mesh_model_chip`, `_job_model_chip`,
`_snapshot_turn`'s model capture, `_turns_public`), `_result_public`'s "model" passthrough,
and the `/api/jobs` + `/api/jobs/{id}` routes that expose it. No subprocess, no real build,
no GPU — pure in-memory job-dict assertions, same style as test_webui_history_persist.py.

Run: python3 -m pytest tests/test_webui_model_chip.py -q
"""
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app.app)

_FLUID_MODEL = {"code_model": "local:gemma-4-31b", "maker_enabled": True,
                "maker_arm": "gemma-4-31b", "engine_version": "5.0",
                "label": "gemma-4-31b (maker)"}


def setup_function(_fn):
    app._jobs.clear()


def _base_job(job_id="job1", **overrides):
    job = {"id": job_id, "spec": "a bracket", "coder": "strong", "image": None,
          "status": "done", "kind": "cad", "log": deque(maxlen=300),
          "result": None, "error": None, "user": "local",
          "created_at": time.time(), "updated_at": time.time()}
    job.update(overrides)
    return job


# ── _mesh_model_chip ────────────────────────────────────────────────────────────

def test_mesh_model_chip_triposr():
    d = app._mesh_model_chip({"provider": "triposr-local"})
    assert d["label"] == "TripoSR (mesh)"
    assert d["code_model"] is None


def test_mesh_model_chip_meshy():
    assert app._mesh_model_chip({"provider": "meshy"})["label"] == "Meshy (mesh)"


def test_mesh_model_chip_unknown_provider_falls_back_to_the_raw_name():
    d = app._mesh_model_chip({"provider": "some-new-backend"})
    assert d["label"] == "some-new-backend (mesh)"


def test_mesh_model_chip_missing_provider():
    d = app._mesh_model_chip({})
    assert "unknown provider" in d["label"]


# ── _job_model_chip ──────────────────────────────────────────────────────────────

def test_job_model_chip_fluid_job_reads_result_model():
    job = _base_job(result={"model": _FLUID_MODEL})
    assert app._job_model_chip(job) == _FLUID_MODEL


def test_job_model_chip_mesh_job_derives_from_provider_not_a_stored_model_key():
    job = _base_job(kind="mesh", result={"provider": "triposr-local"})
    assert app._job_model_chip(job)["label"] == "TripoSR (mesh)"


def test_job_model_chip_none_before_a_result_exists():
    job = _base_job(status="queued", result=None)
    assert app._job_model_chip(job) is None


def test_job_model_chip_mesh_job_none_before_a_result_exists():
    job = _base_job(kind="mesh", status="queued", result=None)
    assert app._job_model_chip(job) is None


# ── _job_light / _job_public expose the chip ────────────────────────────────────

def test_job_light_includes_model():
    job = _base_job(result={"model": _FLUID_MODEL})
    assert app._job_light(job)["model"] == _FLUID_MODEL


def test_job_public_includes_model():
    job = _base_job(result={"model": _FLUID_MODEL}, chat=[])
    assert app._job_public(job)["model"] == _FLUID_MODEL


# ── _snapshot_turn captures the model of the version being archived ─────────────

def test_snapshot_turn_captures_the_current_result_model(tmp_path):
    (tmp_path / "build.png").write_bytes(b"fake-png")
    job = _base_job(result={"model": _FLUID_MODEL, "build_dir_fs": str(tmp_path)})
    app._snapshot_turn(job, str(tmp_path), "make it wider")
    assert len(job["turns"]) == 1
    assert job["turns"][0]["model"] == _FLUID_MODEL
    assert job["turns"][0]["note"] == "make it wider"


def test_snapshot_turn_model_is_none_when_no_prior_result(tmp_path):
    (tmp_path / "build.png").write_bytes(b"fake-png")
    job = _base_job(result=None)
    app._snapshot_turn(job, str(tmp_path), "first revise")
    assert job["turns"][0]["model"] is None


def test_turns_public_surfaces_the_model_per_turn(tmp_path):
    job = _base_job(result={"model": {"label": "later-model"}, "build_id": "b1"})
    job["turns"] = [{"n": 1, "ts": 1.0, "note": "v1", "files": {},
                     "model": _FLUID_MODEL}]
    out = app._turns_public(job)
    assert out[0]["model"] == _FLUID_MODEL


# ── _result_public passes "model" through ───────────────────────────────────────

def test_result_public_keeps_the_model_dict():
    out = app._result_public({"ok": True, "model": _FLUID_MODEL, "code_model": "local:gemma-4-31b"})
    assert out["model"] == _FLUID_MODEL


def test_result_public_omits_model_key_when_absent():
    out = app._result_public({"ok": True})
    assert "model" not in out


# ── HTTP surface: /api/jobs (rail) and /api/jobs/{id} (thread) ──────────────────

def test_api_jobs_rail_includes_model_chip():
    app._jobs["job1"] = _base_job(result={"model": _FLUID_MODEL})
    r = client.get("/api/jobs")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["model"] == _FLUID_MODEL


def test_api_jobs_rail_mesh_row_shows_provider_derived_chip():
    app._jobs["mesh1"] = _base_job("mesh1", kind="mesh", result={"provider": "meshy"})
    r = client.get("/api/jobs")
    assert r.json()[0]["model"]["label"] == "Meshy (mesh)"


def test_api_job_detail_includes_model():
    app._jobs["job1"] = _base_job(result={"model": _FLUID_MODEL}, chat=[])
    r = client.get("/api/jobs/job1")
    assert r.status_code == 200
    assert r.json()["model"] == _FLUID_MODEL

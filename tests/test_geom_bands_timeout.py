"""Unit tests for the hard per-candidate scoring timeout (lab task step 0, 2026-09-26):
pair cfb20363 hung gate/scoring for 40+ minutes on a 54MB STL because perform_geometry_checks
has no internal timeout. geom_bands.score_against_reference now runs the check in its own
subprocess (see _run_check_bounded / _check_worker_main) bounded by a wall-clock timeout, and
records an expiry as band "crash" / reason "score_timeout" rather than hanging forever.

These tests stub subprocess.run itself (no real geometry, no real subprocess spawned) so they
run in well under a second and exercise exactly the timeout-handling logic, independent of the
real cadqueryeval checker's behaviour.
"""
import json
import subprocess
import sys
from pathlib import Path

import trimesh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import geom_bands as gb  # noqa: E402


def _box(tmp_path, name, extents=(10.0, 10.0, 10.0)):
    p = tmp_path / name
    trimesh.creation.box(extents=extents).export(p)
    return p


def test_score_against_reference_reports_score_timeout(tmp_path, monkeypatch):
    """A candidate that hangs the checker past `timeout` is scored 'crash' / reason
    'score_timeout', never left to block the caller indefinitely."""
    cand = _box(tmp_path, "cand.stl")
    ref = _box(tmp_path, "ref.stl")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(gb.subprocess, "run", fake_run)
    out = gb.score_against_reference(cand, ref, timeout=0.01)
    assert out["band"] == "crash"
    assert out["reason"] == "score_timeout"
    assert "errors" in out and out["errors"]


def test_score_against_reference_passes_through_worker_result(tmp_path, monkeypatch):
    """The normal (fast) path is unaffected: a worker subprocess that returns cleanly still
    produces the same band/fields score_against_reference always has."""
    cand = _box(tmp_path, "cand.stl")
    ref = _box(tmp_path, "ref.stl")

    payload = {
        "all_passed": True,
        "is_watertight": True,
        "is_single_component": True,
        "bbox_accurate": True,
        "chamfer_distance": 0.01,
        "hausdorff_95p": 0.01,
        "reference_volume": 1000.0,
        "generated_volume": 1000.0,
        "errors": [],
    }

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(gb.subprocess, "run", fake_run)
    out = gb.score_against_reference(cand, ref, timeout=5)
    assert out["band"] == "match"
    assert out["chamfer_mm"] == 0.01
    assert "reason" not in out


def test_score_against_reference_reports_worker_crash(tmp_path, monkeypatch):
    """A worker subprocess that exits non-zero (a real crash, not a hang) still lands the
    candidate in the ordinary 'fail' band via the existing outer exception handler, not the
    new timeout path -- the two failure modes stay distinguishable."""
    cand = _box(tmp_path, "cand.stl")
    ref = _box(tmp_path, "ref.stl")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(gb.subprocess, "run", fake_run)
    out = gb.score_against_reference(cand, ref, timeout=5)
    assert out["band"] == "fail"
    assert out.get("reason") != "score_timeout"


def test_run_check_bounded_raises_score_timeout_directly(monkeypatch):
    """_run_check_bounded itself raises ScoreTimeout (not just score_against_reference's
    handling of it) on a subprocess.TimeoutExpired."""
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(gb.subprocess, "run", fake_run)
    try:
        gb._run_check_bounded(Path("a.stl"), Path("b.stl"), 1, 0.01)
        assert False, "expected ScoreTimeout"
    except gb.ScoreTimeout:
        pass

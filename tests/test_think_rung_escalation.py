"""Task 1b (2026-09-19): the agent loop's auto-escalation ladder is now two rungs on one
loaded model: the strong rung, then the SAME arm with thinking on (CODE_MODEL_THINK). This
exercises build() end to end with every LLM/subprocess touchpoint monkeypatched (same pattern
as tests/test_n1_offline.py), forcing the deterministic gate to fail every turn so `fails`
crosses ESCALATE_AFTER, then checks:

- coder="auto" with no cad.json pin: the active rung actually climbs from CODE_MODEL_STRONG
  to CODE_MODEL_THINK, and the build result's code_model records the rung that produced the
  final code (the think rung, since it never recovers in this synthetic scenario).
- coder="auto" WITH a cad.json code_model pin: the active rung never leaves the pin, even
  though the same number of turns fail past ESCALATE_AFTER, pinning disables auto_escalate
  entirely (cad_engine.build's `elif pinned:` branch), not just the ladder walk.

Run: python3 -m pytest tests/test_think_rung_escalation.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cad_engine as v4  # noqa: E402


def _minimal_brief():
    return {
        "name": "t", "description": "a test part", "dimensions": {}, "features": [],
        "notes": [], "helper": "",
        "expected": {"solids": 1, "min_holes": 0, "min_through_holes": 0},
    }


def _patch_common(monkeypatch, tmp_path, load_config_result):
    # Required fix (2026-09-24): build() takes a real fcntl.flock on BUILD_LOCK_FILE before
    # any of the mocks below ever run (_acquire_build_lock, called at the top of build()).
    # cad_engine binds BUILD_LOCK_FILE from cad_v5.config at IMPORT time to the real,
    # machine-wide ~/.openclaw/cad-build.lock -- this test never touched that binding, so
    # a run here could block for real (and did: it hung for the full length of a live
    # harvest run holding that exact lock) instead of exercising the escalation ladder.
    # Point it at a private tmp_path file instead, same pattern test_lab_harvest.py and
    # test_gpu_window.py already use for the same lock.
    monkeypatch.setattr(v4, "BUILD_LOCK_FILE", tmp_path / "cad-build.lock")
    monkeypatch.setattr(v4, "preflight", lambda: None)
    monkeypatch.setattr(v4, "_load_config", lambda: load_config_result)
    monkeypatch.setattr(v4, "build_brief", lambda spec: _minimal_brief())
    monkeypatch.setattr(v4, "verify_questions", lambda spec, brief: [])
    monkeypatch.setattr(v4, "generate_code", lambda brief, spec="": "result = 1\n")
    monkeypatch.setattr(v4, "_new_build_dir", lambda spec: tmp_path)
    monkeypatch.setattr(v4, "STEP_OUT", tmp_path / "cad-last-build.step")
    monkeypatch.setattr(v4, "STL_OUT", tmp_path / "cad-last-build.stl")
    monkeypatch.setattr(v4, "DXF_OUT", tmp_path / "cad-last-build.dxf")
    monkeypatch.setattr(v4, "SESSION_FILE", tmp_path / "cad-session.json")
    monkeypatch.setattr(v4, "CONTRACT_FILE", tmp_path / "cad-contract.json")
    monkeypatch.setattr(v4, "run_stl", lambda step_out, stl_out: stl_out)
    monkeypatch.setattr(v4, "run_dxf",
                        lambda step_out, dxf_out: (_ for _ in ()).throw(RuntimeError("no flat face")))
    monkeypatch.setattr(v4, "visual_critique", lambda *a, **k: None)
    monkeypatch.setattr(v4, "decide_or_edit", lambda *a, **k: ("done", None))
    monkeypatch.setattr(v4, "_write_session", lambda data: None)
    monkeypatch.setattr(v4, "parse_facts", lambda output: {"solids": 1})
    # The gate fails EVERY turn: the one deterministic way to drive `fails` past
    # ESCALATE_AFTER without depending on the critic/decide path at all.
    monkeypatch.setattr(v4, "verify_expected",
                        lambda facts, expected, spec="": (["synthetic hard fail"], []))
    monkeypatch.setattr(v4, "revise_script",
                        lambda spec, code, problem, state="": "result = 1  # revised\n")

    step_counter = {"n": 0}

    def fake_run_step(code, work_dir):
        step_counter["n"] += 1
        out_step = work_dir / f"build_output_{step_counter['n']}.step"
        out_step.write_text("fake step content")
        return out_step, "build123d ok"

    monkeypatch.setattr(v4, "run_step", fake_run_step)
    monkeypatch.setattr(v4, "run_inspect", lambda step_path: {
        "valid": True, "output": "FACTS_JSON: {}", "errors": [], "warnings": []})


def test_build_escalates_from_strong_to_think_when_unpinned(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path, load_config_result={"cad": {}})

    result = v4.build("a test part", coder="auto", use_fewshots=False,
                      do_upload=False, final_render=False)

    assert v4.MAX_TURNS > v4.ESCALATE_AFTER, "test assumes room for at least one escalation"
    assert result["code_model"] == v4.CODE_MODEL_THINK
    assert result["converged"] is False   # the synthetic gate never lets a turn pass


def test_pinned_code_model_never_escalates_to_think(tmp_path, monkeypatch):
    pin = v4.CODE_MODEL_STRONG
    _patch_common(monkeypatch, tmp_path, load_config_result={"cad": {"code_model": pin}})

    result = v4.build("a test part", coder="auto", use_fewshots=False,
                      do_upload=False, final_render=False)

    assert result["code_model"] == pin
    assert result["code_model"] != v4.CODE_MODEL_THINK

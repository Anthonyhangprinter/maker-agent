"""N1 offline test — end-to-end build() with every LLM/subprocess touchpoint monkeypatched, so
it runs with zero Ollama calls and zero real build123d/subprocess invocations. Exercises the
inline auto-fix micro-loop: run_step raises once, then succeeds on the inline retry, all within
the SAME turn. Run: python3 -m pytest tests/test_n1_offline.py -q
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


def _patch_common(monkeypatch, tmp_path):
    """Wire every LLM/subprocess-touching call to an offline stub; leave the pure/deterministic
    functions (reconcile_expected, wants_section, _check_syntax) real."""
    monkeypatch.setattr(v4, "preflight", lambda: None)
    monkeypatch.setattr(v4, "build_brief", lambda spec: _minimal_brief())
    monkeypatch.setattr(v4, "verify_questions", lambda spec, brief: [])
    monkeypatch.setattr(v4, "generate_code", lambda brief: "result = 1\n")
    monkeypatch.setattr(v4, "_new_build_dir", lambda spec: tmp_path)
    monkeypatch.setattr(v4, "STEP_OUT", tmp_path / "cad-last-build.step")
    monkeypatch.setattr(v4, "STL_OUT", tmp_path / "cad-last-build.stl")
    monkeypatch.setattr(v4, "DXF_OUT", tmp_path / "cad-last-build.dxf")
    monkeypatch.setattr(v4, "SESSION_FILE", tmp_path / "cad-session.json")
    monkeypatch.setattr(v4, "CONTRACT_FILE", tmp_path / "cad-contract.json")
    monkeypatch.setattr(v4, "run_stl", lambda step_out, stl_out: stl_out)
    monkeypatch.setattr(v4, "run_dxf",
                        lambda step_out, dxf_out: (_ for _ in ()).throw(RuntimeError("no flat face")))
    monkeypatch.setattr(v4, "parse_facts", lambda output: {"solids": 1, "cyl_faces": 0})
    monkeypatch.setattr(v4, "verify_expected", lambda facts, expected, spec="": ([], []))
    monkeypatch.setattr(v4, "visual_critique", lambda *a, **k: None)
    monkeypatch.setattr(v4, "decide_or_edit", lambda *a, **k: ("done", None))
    monkeypatch.setattr(v4, "_write_session", lambda data: None)
    monkeypatch.setattr(v4, "run_inspect", lambda step_path: {
        "valid": True, "output": "FACTS_JSON: {}", "errors": [], "warnings": []})


def test_n1_recovers_inline_without_burning_a_turn(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path)

    run_step_calls = {"n": 0}

    def fake_run_step(code, work_dir):
        run_step_calls["n"] += 1
        if run_step_calls["n"] == 1:
            raise RuntimeError("NameError: name 'undefined_var' is not defined")
        out_step = work_dir / "build_output.step"
        out_step.write_text("fake step content")
        return out_step, "build123d ok"

    monkeypatch.setattr(v4, "run_step", fake_run_step)

    revise_calls = {"n": 0}

    def fake_revise_script(spec, code, problem, state=""):
        revise_calls["n"] += 1
        return "result = 1  # auto-fixed\n"

    monkeypatch.setattr(v4, "revise_script", fake_revise_script)

    result = v4.build("a test part", coder="fast", use_fewshots=False,
                      do_upload=False, final_render=False)

    # Recovered on the FIRST inline retry — only one run_step failure, one revise_script call,
    # then a second (successful) run_step call, all inside turn 1.
    assert run_step_calls["n"] == 2, "expected exactly one retry (2 run_step calls total)"
    assert revise_calls["n"] == 1, "expected exactly one inline auto-fix revise"
    assert result["n1_autofixes"] == 1
    assert result["converged"] is True
    assert result["ok"] is True


def test_n1_exhausts_retries_and_increments_fails_once(tmp_path, monkeypatch):
    """When every inline retry also fails, the turn is recorded as exactly ONE failure (not one
    per inline attempt) and the build proceeds to the next turn — proven here by observing the
    turn actually converges on turn 2 after N1_RETRIES+1 failed attempts on turn 1."""
    _patch_common(monkeypatch, tmp_path)

    run_step_calls = {"n": 0}
    # Attempts 1..(N1_RETRIES+1) fail (all of turn 1's inline budget); attempt N1_RETRIES+2
    # (the start of turn 2) succeeds.
    fail_budget = v4.N1_RETRIES + 1

    def fake_run_step(code, work_dir):
        run_step_calls["n"] += 1
        if run_step_calls["n"] <= fail_budget:
            raise RuntimeError(f"boom #{run_step_calls['n']}")
        out_step = work_dir / "build_output.step"
        out_step.write_text("fake step content")
        return out_step, "build123d ok"

    monkeypatch.setattr(v4, "run_step", fake_run_step)
    monkeypatch.setattr(v4, "revise_script",
                        lambda spec, code, problem, state="": "result = 1  # revised\n")

    result = v4.build("a test part", coder="fast", use_fewshots=False,
                      do_upload=False, final_render=False)

    # N1_RETRIES inline retries were consumed on turn 1 (all failed), then the turn-level
    # failure counter did its normal continue -> turn 2, which succeeds outright.
    assert result["n1_autofixes"] == v4.N1_RETRIES
    assert result["converged"] is True
    assert run_step_calls["n"] == fail_budget + 1

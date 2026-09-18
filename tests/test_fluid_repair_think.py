"""Task 1b (2026-09-19): fluid mode's `repair_think` knob. Default OFF: the first codegen
attempt is never touched either way; when on (CAD_REPAIR_THINK=1 or cad.json's `repair_think`),
the ONE crash-salvage turn and the ONE gate-repair turn in _materialize_with_salvage ride the
think rung instead of whatever the active rung already is, and the result records which rung
actually produced the winning repair in `repair_rung`.

Everything LLM/subprocess-touching is monkeypatched. Run:
    python3 -m pytest tests/test_fluid_repair_think.py -q
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("fluid_gen", HERE / "scripts" / "fluid_gen.py")
fg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fg
_spec.loader.exec_module(fg)


def _clean_materialize_result(error=None, gate_hard=None, gate_spec=None):
    return {"facts": {}, "instruments": [], "gate_hard": gate_hard or [],
            "gate_spec": gate_spec or [], "gate_adv": [], "error": error}


def test_repair_think_off_by_default_leaves_the_active_rung_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(fg, "repair_think_enabled", lambda: False)
    monkeypatch.setattr(fg.engine, "_ACTIVE_CODE_MODEL", "local:gemma-4-31b")

    seen_rung_during_call = {}

    def fake_revise_script(spec, code, problem, state=""):
        seen_rung_during_call["rung"] = fg.engine._ACTIVE_CODE_MODEL
        return "fixed = 1\n"

    monkeypatch.setattr(fg.engine, "revise_script", fake_revise_script)

    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return _clean_materialize_result(error="boom")
        return _clean_materialize_result(error=None)

    monkeypatch.setattr(fg, "_materialize", fake_materialize)

    m = fg._materialize_with_salvage("a cube", "code0", tmp_path)

    assert seen_rung_during_call["rung"] == "local:gemma-4-31b"   # unchanged during the call
    assert fg.engine._ACTIVE_CODE_MODEL == "local:gemma-4-31b"    # unchanged after
    assert m["salvaged"] is True
    assert m["repair_rung"] is None


def test_repair_think_on_routes_crash_salvage_to_the_think_rung(monkeypatch, tmp_path):
    monkeypatch.setattr(fg, "repair_think_enabled", lambda: True)
    monkeypatch.setattr(fg.engine, "_ACTIVE_CODE_MODEL", "local:gemma-4-31b")
    monkeypatch.setattr(fg.engine, "CODE_MODEL_THINK", "local:gemma-4-31b+think")

    seen_rung_during_call = {}

    def fake_revise_script(spec, code, problem, state=""):
        seen_rung_during_call["rung"] = fg.engine._ACTIVE_CODE_MODEL
        return "fixed = 1\n"

    monkeypatch.setattr(fg.engine, "revise_script", fake_revise_script)

    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return _clean_materialize_result(error="boom")
        return _clean_materialize_result(error=None)

    monkeypatch.setattr(fg, "_materialize", fake_materialize)

    m = fg._materialize_with_salvage("a cube", "code0", tmp_path)

    assert seen_rung_during_call["rung"] == "local:gemma-4-31b+think"   # switched during the call
    assert fg.engine._ACTIVE_CODE_MODEL == "local:gemma-4-31b"         # restored after
    assert m["salvaged"] is True
    assert m["repair_rung"] == "local:gemma-4-31b+think"


def test_repair_think_on_routes_gate_repair_to_the_think_rung(monkeypatch, tmp_path):
    monkeypatch.setattr(fg, "repair_think_enabled", lambda: True)
    monkeypatch.setattr(fg.engine, "_ACTIVE_CODE_MODEL", "local:gemma-4-31b")
    monkeypatch.setattr(fg.engine, "CODE_MODEL_THINK", "local:gemma-4-31b+think")
    (tmp_path / "build_source.py").write_text("code0")
    (tmp_path / "build.step").write_bytes(b"step")

    seen_rung_during_call = {}

    def fake_revise_script(spec, code, problem, state=""):
        seen_rung_during_call["rung"] = fg.engine._ACTIVE_CODE_MODEL
        return "fixed = 1\n"

    monkeypatch.setattr(fg.engine, "revise_script", fake_revise_script)

    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return _clean_materialize_result(error=None, gate_hard=["bad envelope"])
        return _clean_materialize_result(error=None, gate_hard=[])   # the repair improves

    monkeypatch.setattr(fg, "_materialize", fake_materialize)

    m = fg._materialize_with_salvage("a cube", "code0", tmp_path, gate_repair=True)

    assert seen_rung_during_call["rung"] == "local:gemma-4-31b+think"
    assert fg.engine._ACTIVE_CODE_MODEL == "local:gemma-4-31b"
    assert m["gate_repaired"] is True
    assert m["repair_rung"] == "local:gemma-4-31b+think"


def test_no_repair_needed_records_no_rung(monkeypatch, tmp_path):
    """The knob must never fire when there is nothing to repair: a clean first attempt
    keeps repair_rung None regardless of the knob."""
    monkeypatch.setattr(fg, "repair_think_enabled", lambda: True)
    monkeypatch.setattr(fg.engine, "_ACTIVE_CODE_MODEL", "local:gemma-4-31b")

    def fail_revise(*a, **k):
        raise AssertionError("revise_script should not run when the first attempt is clean")

    monkeypatch.setattr(fg.engine, "revise_script", fail_revise)
    monkeypatch.setattr(fg, "_materialize",
                        lambda code, build_dir, spec="": _clean_materialize_result())

    m = fg._materialize_with_salvage("a cube", "code0", tmp_path)

    assert m["error"] is None
    assert m["repair_rung"] is None

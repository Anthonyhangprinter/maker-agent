"""Task 2 (best-of-N in one-shot mode): fluid_gen.cmd_build should route the initial
codegen through engine._pick_first_turn_candidate whenever a caller opts in
(CAD_CANDIDATES env or cad.json's `candidates` key) AND the resulting n > 1, and the
first candidate is not a domain-helper short-circuit.

Deviation from the task-2 brief's literal wording, recorded in task-2-report.md: the
brief's Step 1 test patches only `first_turn_candidates` and expects the hook to fire
unconditionally. first_turn_candidates() defaults to 3 (CANDIDATES_DEFAULT in
cad_v5/config.py) whenever CAD_CANDIDATES is unset and cad.json has no `candidates` key,
so calling it unconditionally would turn best-of-N on for every fluid build with no
opt-in, silently changing the single-shot baseline the Phase 0 A/B measured against.
So the hook is gated on an explicit opt-in (env or config), verified below alongside
the "opted in and n>1 => hook fires" case the brief specifies.

Everything LLM/subprocess-touching is monkeypatched so this runs with zero Ollama calls
and zero real build123d invocations. Run: python3 -m pytest tests/test_fluid_bon.py -q
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("fluid_gen", HERE / "scripts" / "fluid_gen.py")
fg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fg
_spec.loader.exec_module(fg)


def _fake_args(**overrides):
    base = dict(spec="A cube 10 mm", coder="strong", image=None, no_fewshots=True, json=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_common(monkeypatch, tmp_path, recorded):
    monkeypatch.setattr(fg, "BUILDS_DIR", tmp_path)
    monkeypatch.delenv("CAD_CANDIDATES", raising=False)
    monkeypatch.setattr(fg, "load_config", lambda: {})
    monkeypatch.setattr(fg.engine, "triage_ambiguity", lambda spec: None)
    monkeypatch.setattr(fg.engine, "retrieval_notes_for",
                         lambda spec, use_fewshots=True: [])
    monkeypatch.setattr(fg.engine, "spec_helper", lambda spec: "")
    monkeypatch.setattr(fg.engine, "generate_code_raw",
                         lambda spec, notes, temperature=0.15: "code0")

    def fake_pick(brief, spec, first_code, work_dir, n):
        recorded["brief"] = brief
        recorded["spec"] = spec
        recorded["first_code"] = first_code
        recorded["work_dir"] = work_dir
        recorded["n"] = n
        return "best"

    monkeypatch.setattr(fg.engine, "_pick_first_turn_candidate", fake_pick)

    def fake_materialize(spec, code, build_dir, gate_repair=True):
        recorded["materialize_code"] = code
        return {"error": None, "facts": {}, "instruments": [], "gate_hard": [],
                "gate_spec": [], "gate_adv": [], "salvaged": False,
                "gate_repaired": False}

    monkeypatch.setattr(fg, "_materialize_with_salvage", fake_materialize)
    monkeypatch.setattr(fg.engine, "_code_model", lambda: "test-model")
    monkeypatch.setattr(fg.engine, "reset_usage", lambda: None)
    monkeypatch.setattr(fg.engine, "_USAGE_TOTAL", {}, raising=False)
    monkeypatch.setattr(fg.engine, "_LAST_USAGE", None, raising=False)


def test_bon_hook_invokes_candidate_picker_when_opted_in_via_env(monkeypatch, tmp_path):
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setenv("CAD_CANDIDATES", "3")
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 3)

    result = fg.cmd_build(_fake_args())

    assert recorded["n"] == 3
    assert recorded["first_code"] == "code0"
    assert recorded["work_dir"].exists()
    assert str(tmp_path) in str(recorded["work_dir"])
    assert result["candidates"] == 3
    assert recorded["materialize_code"] == "best"


def test_bon_hook_invokes_candidate_picker_when_opted_in_via_config(monkeypatch, tmp_path):
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setattr(fg, "load_config", lambda: {"cad": {"candidates": 3}})
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 3)

    result = fg.cmd_build(_fake_args())

    assert recorded["n"] == 3
    assert result["candidates"] == 3


def test_bon_hook_not_invoked_without_opt_in_even_if_default_is_3(monkeypatch, tmp_path):
    """No CAD_CANDIDATES env, no cad.json `candidates` key: the hook must stay off even
    though first_turn_candidates() itself defaults to 3. This is the single-shot
    baseline the best-of-N A/B needs to stay comparable against."""
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 3)

    def fail_pick(*a, **k):
        raise AssertionError("_pick_first_turn_candidate should not run without opt-in")

    monkeypatch.setattr(fg.engine, "_pick_first_turn_candidate", fail_pick)

    result = fg.cmd_build(_fake_args())

    assert result["candidates"] == 1
    assert recorded["materialize_code"] == "code0"


def test_bon_hook_skipped_when_opted_in_but_n_is_1(monkeypatch, tmp_path):
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setenv("CAD_CANDIDATES", "1")
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 1)

    def fail_pick(*a, **k):
        raise AssertionError("_pick_first_turn_candidate should not be called when n == 1")

    monkeypatch.setattr(fg.engine, "_pick_first_turn_candidate", fail_pick)

    result = fg.cmd_build(_fake_args())

    assert result["candidates"] == 1
    assert recorded["materialize_code"] == "code0"


def test_bon_hook_skipped_for_helper_short_circuit(monkeypatch, tmp_path):
    """spec_helper() itself returning empty means the outer helper branch is skipped, but
    the raw coder can still hand back a bare domain-helper call (result = spur_gear(...)),
    so the best-of-N hook must recognise that via _HELPER_RESULT_RE and skip sampling,
    since a correct-by-construction call has nothing to gain from more candidates."""
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setenv("CAD_CANDIDATES", "3")
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 3)
    monkeypatch.setattr(fg.engine, "generate_code_raw",
                         lambda spec, notes, temperature=0.15: "result = spur_gear(20, 2, 15)\n")

    def fail_pick(*a, **k):
        raise AssertionError("_pick_first_turn_candidate should not run over a helper result")

    monkeypatch.setattr(fg.engine, "_pick_first_turn_candidate", fail_pick)

    result = fg.cmd_build(_fake_args())

    assert result["candidates"] == 1
    assert recorded["materialize_code"] == "result = spur_gear(20, 2, 15)\n"


def test_bon_hook_skipped_for_true_helper_branch(monkeypatch, tmp_path):
    """spec_helper() matching (e.g. a fully-pinned bolt/gear spec) short-circuits codegen
    entirely: generate_code_raw / the candidate picker never run, and candidates is 1."""
    recorded = {}
    _patch_common(monkeypatch, tmp_path, recorded)
    monkeypatch.setenv("CAD_CANDIDATES", "3")
    monkeypatch.setattr(fg, "first_turn_candidates", lambda: 3)
    monkeypatch.setattr(fg.engine, "spec_helper", lambda spec: "spur_gear(20, 2, 15)")
    monkeypatch.setattr(fg.engine, "generate_code",
                         lambda brief, spec: "result = spur_gear(20, 2, 15)\n")

    def fail_pick(*a, **k):
        raise AssertionError("_pick_first_turn_candidate should not run for a true helper")

    monkeypatch.setattr(fg.engine, "_pick_first_turn_candidate", fail_pick)

    result = fg.cmd_build(_fake_args())

    assert result["candidates"] == 1
    assert recorded["materialize_code"] == "result = spur_gear(20, 2, 15)\n"

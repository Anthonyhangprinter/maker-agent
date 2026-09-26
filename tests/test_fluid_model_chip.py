"""Owner request (2026-09-27): every build and every revise turn must record which model
actually built it — the resolved code_model string, the maker arm (or 'resident' when the
maker is disabled), and the engine VERSION — so the web UI's model chip stays honest even
after cad.json's maker arm moves on. This covers scripts/fluid_gen.py's `build`/`revise`
subcommands: the "model" key on the JSON result and the same key written into the build
dir's fluid.json sidecar.

Everything LLM/subprocess-touching is monkeypatched (same pattern as test_fluid_bon.py) so
this runs with zero Ollama calls and zero real build123d invocations.

Run: python3 -m pytest tests/test_fluid_model_chip.py -q
"""
import importlib
import importlib.util
import json
import os
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


def _patch_build_common(monkeypatch, tmp_path, code_model="local:gemma-4-31b"):
    monkeypatch.setattr(fg, "BUILDS_DIR", tmp_path)
    monkeypatch.delenv("CAD_CANDIDATES", raising=False)
    monkeypatch.setattr(fg, "load_config", lambda: {})
    monkeypatch.setattr(fg.engine, "triage_ambiguity", lambda spec: None)
    monkeypatch.setattr(fg.engine, "retrieval_notes_for", lambda spec, use_fewshots=True: [])
    monkeypatch.setattr(fg.engine, "spec_helper", lambda spec: "")
    monkeypatch.setattr(fg.engine, "generate_code_raw",
                        lambda spec, notes, temperature=0.15: "code0")

    def fake_materialize(spec, code, build_dir, gate_repair=True):
        return {"error": None, "facts": {}, "instruments": [], "gate_hard": [],
                "gate_spec": [], "gate_adv": [], "salvaged": False, "gate_repaired": False}

    monkeypatch.setattr(fg, "_materialize_with_salvage", fake_materialize)
    monkeypatch.setattr(fg.engine, "_code_model", lambda: code_model)
    monkeypatch.setattr(fg.engine, "reset_usage", lambda: None)
    monkeypatch.setattr(fg.engine, "_USAGE_TOTAL", {}, raising=False)
    monkeypatch.setattr(fg.engine, "_LAST_USAGE", None, raising=False)
    # _model_for() would otherwise resolve the real CODE_MODEL_STRONG from cad.json; a
    # build/revise's OWN model identity must come from engine._code_model() (patched
    # above), so make _model_for a no-op rather than fighting it.
    monkeypatch.setattr(fg, "_model_for", lambda coder: None)


def _reload_config_with(tmp_path, cad_json):
    p = tmp_path / "cad.json"
    p.write_text(json.dumps(cad_json))
    os.environ["CAD_CONFIG_FILE"] = str(p)
    import cad_v5.config as cfg
    return importlib.reload(cfg)


def test_build_result_carries_the_model_identity(monkeypatch, tmp_path):
    cfg = _reload_config_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                                    "alias": "gemma-4-31b"}})
    _patch_build_common(monkeypatch, tmp_path, code_model="local:gemma-4-31b")
    res = fg.cmd_build(_fake_args())
    assert res["ok"] is True
    assert res["model"]["code_model"] == "local:gemma-4-31b"
    assert res["model"]["maker_arm"] == "gemma-4-31b"
    assert res["model"]["label"] == "gemma-4-31b (maker)"
    assert res["model"]["engine_version"] == cfg.VERSION


def test_build_writes_the_model_into_fluid_json(monkeypatch, tmp_path):
    """Self-describing artefact (owner request): the build dir's own fluid.json must say
    what built it, independent of the sessions.json row the web UI keeps separately."""
    _reload_config_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                             "alias": "gemma-4-31b-cad-r1"}})
    _patch_build_common(monkeypatch, tmp_path, code_model="local:gemma-4-31b-cad-r1")
    res = fg.cmd_build(_fake_args())
    build_dir = Path(res["build_dir"])
    meta = json.loads((build_dir / "fluid.json").read_text(encoding="utf-8"))
    assert meta["model"]["maker_arm"] == "gemma-4-31b-cad-r1"
    assert meta["model"]["label"] == "gemma-4-31b-cad-r1 (maker)"


def test_build_on_the_resident_labels_it_resident_not_maker(monkeypatch, tmp_path):
    _reload_config_with(tmp_path, {})     # maker disabled -> resident
    _patch_build_common(monkeypatch, tmp_path, code_model="local:qwen3.8-27b")
    res = fg.cmd_build(_fake_args())
    assert res["model"]["maker_enabled"] is False
    assert res["model"]["label"] == "qwen3.8-27b (resident)"


def test_revise_records_a_per_turn_model_in_history_and_updates_top_level(monkeypatch, tmp_path):
    """cad.json's maker arm can move on between a build and a later revise — history[i]
    must keep saying what built THAT turn, while the top-level fluid.json `model` tracks
    whichever turn most recently touched the build dir (per the owner's "continuing an old
    build today uses today's model" rule)."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "build_source.py").write_text("from build123d import *\nresult = Box(10, 10, 10)\n")
    original_model = {"code_model": "local:gemma-4-31b-cad-spike", "maker_enabled": True,
                      "maker_arm": "gemma-4-31b-cad-spike", "engine_version": "5.0",
                      "label": "gemma-4-31b-cad-spike (maker)"}
    (build_dir / "fluid.json").write_text(json.dumps(
        {"spec": "a cube", "history": [], "model": original_model}), encoding="utf-8")

    _reload_config_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                             "alias": "gemma-4-31b"}})   # arm swapped since
    monkeypatch.setattr(fg, "_model_for", lambda coder: None)
    monkeypatch.setattr(fg.engine, "_code_model", lambda: "local:gemma-4-31b")
    monkeypatch.setattr(fg.engine, "reset_usage", lambda: None)
    monkeypatch.setattr(fg.engine, "_USAGE_TOTAL", {}, raising=False)
    monkeypatch.setattr(fg.engine, "_LAST_USAGE", None, raising=False)
    monkeypatch.setattr(fg.engine, "run_inspect", lambda p: {"valid": False, "output": ""})
    monkeypatch.setattr(fg.engine, "revise_script",
                        lambda spec, code, feedback, state="": "from build123d import *\nresult = Box(20, 10, 10)\n")

    def fake_materialize(spec, code, build_dir, gate_repair=True):
        return {"error": None, "facts": {}, "instruments": [], "gate_hard": [],
                "gate_spec": [], "gate_adv": [], "salvaged": False, "gate_repaired": False}

    monkeypatch.setattr(fg, "_materialize_with_salvage", fake_materialize)

    res = fg.cmd_revise(SimpleNamespace(build_dir=str(build_dir), feedback="make it wider",
                                        coder="", image=None, no_fewshots=True, json=True))
    assert res["ok"] is True
    assert res["model"]["maker_arm"] == "gemma-4-31b"

    meta = json.loads((build_dir / "fluid.json").read_text(encoding="utf-8"))
    assert len(meta["history"]) == 1
    turn = meta["history"][0]
    assert turn["feedback"] == "make it wider"
    assert turn["model"]["maker_arm"] == "gemma-4-31b"          # today's arm, not the spike
    # the top-level `model` field now tracks the turn that most recently built this dir
    assert meta["model"]["maker_arm"] == "gemma-4-31b"


def teardown_module(module):
    os.environ.pop("CAD_CONFIG_FILE", None)
    import cad_v5.config as cfg
    importlib.reload(cfg)

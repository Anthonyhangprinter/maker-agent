"""Offline tests for cad_v5/design_assistant.py — the pre-build parameter proposal and the
build123d post-build slider extraction/substitution. Everything model-facing is monkeypatched
(engine._ollama); no network, no GPU, no build lock. Real generated-script fixtures live in
tests/fixtures/build123d_*.py, copied read-only from a Phase 3 harvest build dir.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cad_engine as engine  # noqa: E402
from cad_v5 import design_assistant as da  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fake(payload: dict):
    def _call(*a, **kw):
        return json.dumps(payload)
    return _call


# ── propose_parameters / _sanitize_proposal ──────────────────────────────────────

def test_user_number_survives_even_when_model_drops_it(monkeypatch):
    monkeypatch.setattr(engine, "_ollama", _fake({
        "title": "Rod", "summary": "A rod.", "features": [],
        "parameters": [
            {"key": "length", "label": "Length", "value": 100, "unit": "mm",
             "min": 50, "max": 150, "step": 1, "source": "user", "why": "x"},
        ],
    }))
    prop = da.propose_parameters("a 100mm rod, 8mm diameter")
    values = [(p["value"], p["unit"], p["source"]) for p in prop["parameters"]]
    assert (100.0, "mm", "user") in values
    assert (8.0, "mm", "user") in values, values


def test_user_number_restored_in_place_when_model_alters_it(monkeypatch):
    """The model changed the user's "2mm walls" to 3mm but still claimed source=user —
    it must be corrected back to 2, in place, not duplicated as a second parameter."""
    monkeypatch.setattr(engine, "_ollama", _fake({
        "title": "Enclosure", "summary": "x", "features": [],
        "parameters": [
            {"key": "length", "label": "Length", "value": 100, "unit": "mm",
             "min": 50, "max": 150, "step": 1, "source": "user", "why": "x"},
            {"key": "width", "label": "Width", "value": 60, "unit": "mm",
             "min": 30, "max": 90, "step": 1, "source": "user", "why": "x"},
            {"key": "depth", "label": "Depth", "value": 20, "unit": "mm",
             "min": 10, "max": 30, "step": 1, "source": "user", "why": "x"},
            {"key": "wall_thickness", "label": "Wall Thickness", "value": 3, "unit": "mm",
             "min": 1, "max": 6, "step": 0.5, "source": "user", "why": "x"},
        ],
    }))
    prop = da.propose_parameters("a 100x60x20mm enclosure with 2mm walls")
    assert len(prop["parameters"]) == 4, "must repair in place, never insert a duplicate"
    wall = next(p for p in prop["parameters"] if p["key"] == "wall_thickness")
    assert wall["value"] == 2.0
    assert wall["source"] == "user"


def test_min_value_max_repaired_when_model_gives_a_broken_range(monkeypatch):
    monkeypatch.setattr(engine, "_ollama", _fake({
        "title": "x", "summary": "x", "features": [],
        "parameters": [
            {"key": "wall", "label": "Wall", "value": 2, "unit": "mm",
             "min": 5, "max": 1, "step": 0, "source": "assumed", "why": "x"},
            {"key": "holes", "label": "Holes", "value": 4, "unit": "count",
             "min": 4, "max": 4, "step": 0, "source": "assumed", "why": "x"},
        ],
    }))
    prop = da.propose_parameters("a plate")
    for p in prop["parameters"]:
        assert p["min"] < p["value"] < p["max"], p


def test_count_parameters_are_integers_with_min_at_least_one(monkeypatch):
    monkeypatch.setattr(engine, "_ollama", _fake({
        "title": "x", "summary": "x", "features": [],
        "parameters": [
            {"key": "cyl_count", "label": "Cylinder Count", "value": 9.0, "unit": "count",
             "min": -2, "max": 20, "step": 1, "source": "assumed", "why": "x"},
        ],
    }))
    prop = da.propose_parameters("a manifold with cylinders")
    p = prop["parameters"][0]
    assert isinstance(p["value"], int) and p["value"] == 9
    assert p["min"] >= 1


def test_more_than_14_parameters_are_capped_keeping_user_values(monkeypatch):
    params = [{"key": f"p{i}", "label": f"P{i}", "value": i + 1, "unit": "mm",
               "min": 0, "max": 100, "step": 1, "source": "assumed", "why": "x"}
              for i in range(20)]
    params[17]["source"] = "user"   # a user-stated value buried among the assumed ones
    monkeypatch.setattr(engine, "_ollama", _fake(
        {"title": "x", "summary": "x", "features": [], "parameters": params}))
    prop = da.propose_parameters("18")
    assert len(prop["parameters"]) <= da.MAX_PARAMETERS
    assert any(p["source"] == "user" for p in prop["parameters"])


def test_malformed_json_degrades_to_no_proposal(monkeypatch):
    monkeypatch.setattr(engine, "_ollama", lambda *a, **kw: "not json at all, sorry")
    prop = da.propose_parameters("a bracket")
    assert prop == {"title": "", "summary": "", "parameters": [], "features": []}


def test_model_exception_degrades_to_no_proposal_without_raising(monkeypatch):
    def _raise(*a, **kw):
        raise RuntimeError("server unreachable")
    monkeypatch.setattr(engine, "_ollama", _raise)
    prop = da.propose_parameters("a bracket")
    assert prop["parameters"] == []


def test_empty_spec_short_circuits_without_calling_the_model(monkeypatch):
    called = []
    monkeypatch.setattr(engine, "_ollama", lambda *a, **kw: called.append(1) or "{}")
    prop = da.propose_parameters("   ")
    assert prop["parameters"] == []
    assert not called


# ── compose_spec ──────────────────────────────────────────────────────────────

def test_compose_spec_keeps_user_words_verbatim_and_first():
    composed = da.compose_spec("a 100x60x20mm enclosure with 2mm walls", [
        {"key": "wall", "label": "Wall Thickness", "value": 2.0, "unit": "mm", "source": "user"},
        {"key": "clear", "label": "Lid Clearance", "value": 0.3, "unit": "mm", "source": "assumed"},
    ])
    assert composed.startswith("a 100x60x20mm enclosure with 2mm walls")
    assert "Wall Thickness: 2 mm" in composed
    assert "Lid Clearance: 0.3 mm (assumed)" in composed


def test_compose_spec_passthrough_with_no_parameters():
    assert da.compose_spec("just the spec", []) == "just the spec"
    assert da.compose_spec("just the spec", None) == "just the spec"


# ── mode logic ──────────────────────────────────────────────────────────────────

def test_mode_defaults_to_auto(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {})
    assert da.design_assistant_mode() == "auto"


def test_mode_reads_cad_json(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {"cad": {"design_assistant": "always"}})
    assert da.design_assistant_mode() == "always"


def test_mode_explicit_request_overrides_config(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {"cad": {"design_assistant": "always"}})
    assert da.design_assistant_mode("off") == "off"


def test_cad_bench_forces_off_even_when_requested_always(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {"cad": {"design_assistant": "always"}})
    monkeypatch.setenv("CAD_BENCH", "1")
    assert da.design_assistant_mode("always") == "off"
    assert da.design_assistant_mode() == "off"


def test_invalid_config_value_falls_back_to_auto(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {"cad": {"design_assistant": "banana"}})
    assert da.design_assistant_mode() == "auto"


# ── build123d AST extraction / substitution ──────────────────────────────────────

def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_extract_simple_plate_all_substitutable_numeric():
    src = _read_fixture("build123d_simple_plate.py")
    params = da.extract_build123d_params(src)
    names = {p["name"]: p for p in params}
    assert set(names) == {"plate_size", "plate_thickness", "cutout_size", "chamfer_size"}
    for p in params:
        assert p["substitutable"] is True
        assert p["expr"] is None
        assert p["min"] < p["value"] < p["max"]


def test_extract_negative_number_and_expression_constant():
    src = _read_fixture("build123d_negative_and_expr.py")
    params = {p["name"]: p for p in da.extract_build123d_params(src)}
    assert params["center_2"]["value"] == -20.0
    assert params["center_2"]["substitutable"] is True
    # z_offset_1 = (l1 / 2) - (l2 / 2) is pure arithmetic over already-known params —
    # kept read-only for display, not offered as a slider.
    assert params["z_offset_1"]["substitutable"] is False
    assert params["z_offset_1"]["expr"]
    assert params["z_offset_1"]["value"] == (40.0 / 2) - (60.0 / 2)


def test_extract_integer_counts_and_expression_constants():
    src = _read_fixture("build123d_int_count_and_expr.py")
    params = {p["name"]: p for p in da.extract_build123d_params(src)}
    assert params["rib_count"]["value"] == 6
    assert params["rib_count"]["is_int"] is True
    assert params["rib_count"]["unit"] == "count"
    assert params["bolt_count"]["is_int"] is True
    # flange_z / stub_z / bore_h are coordinate calculations, not sliders.
    for derived in ("flange_z", "stub_z", "bore_h"):
        assert params[derived]["substitutable"] is False


def test_substitute_changes_only_the_named_constant_by_position():
    src = _read_fixture("build123d_simple_plate.py")
    new = da.substitute_build123d_params(src, {"plate_size": 80, "chamfer_size": 2.5})
    assert "plate_size = 80.0" in new
    assert "chamfer_size = 2.5" in new
    assert "plate_thickness = 3.0" in new      # untouched
    assert "cutout_size = 20.0" in new         # untouched
    # the comment mentioning the old half-extent must survive unmangled (proves this
    # isn't a whole-file regex substitution)
    assert "-27.5 to 27.5" in new


def test_substitute_ignores_expression_constants_and_unknown_names():
    src = _read_fixture("build123d_int_count_and_expr.py")
    new = da.substitute_build123d_params(src, {"rib_count": 4, "flange_z": 999,
                                                "totally_unknown": 1})
    assert "rib_count = 4" in new
    assert "flange_z = -60.5 + (flange_t / 2)" in new   # expression, not substitutable


def test_substitute_preserves_int_vs_float_literal_type():
    src = _read_fixture("build123d_int_count_and_expr.py")
    new = da.substitute_build123d_params(src, {"bolt_count": 8, "flange_d": 100})
    assert "bolt_count = 8" in new and "bolt_count = 8.0" not in new
    assert "flange_d = 100.0" in new


def test_extract_on_unparsable_source_returns_empty_list():
    assert da.extract_build123d_params("this is not ( python") == []


def test_substitute_with_no_matching_params_returns_source_unchanged():
    src = _read_fixture("build123d_simple_plate.py")
    assert da.substitute_build123d_params(src, {"nope": 1}) == src
    assert da.substitute_build123d_params(src, {}) == src

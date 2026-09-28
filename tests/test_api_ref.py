"""Tests for the introspected API reference (b123d/api_ref.json + api_ref.py) and its wiring
into cad_engine.inject_retrieval_notes (the shared harvest / fluid / teacher prompt path)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import api_ref  # noqa: E402


def test_json_matches_installed_package():
    pytest.importorskip("build123d")
    import gen_api_ref
    on_disk = json.loads((ROOT / "b123d" / "api_ref.json").read_text(encoding="utf-8"))
    assert gen_api_ref.build() == on_disk, "stale: run scripts/gen_api_ref.py"


def test_every_group_key_has_an_entry():
    entries = api_ref._load()["entries"]
    for _name, _s, _c, keys in api_ref.GROUPS:
        for k in keys:
            assert k in entries, k


def test_helper_signatures_are_the_real_ones():
    e = api_ref._load()["entries"]
    assert e["spur_gear"]["sig"].startswith("spur_gear(teeth, module_mm, width_mm")
    assert e["bolt_circle"]["import"] == "from b123d.domain import bolt_circle"
    assert "X Y Z" in e["Vector"]["fact"].replace(".", "")


def test_plain_part_gets_no_reference():
    assert api_ref.api_reference_note("a rectangular mounting plate 60x90x12mm with one "
                                      "central 16mm through hole") == ""


def test_fillet_spec_gets_edge_group_only():
    keys = api_ref.select_keys("a 40x40x10 block with a 2mm fillet on the top edges")
    assert "fillet" in keys and "Edge.center" in keys and "Vector" in keys
    assert "bolt_circle" not in keys and "revolve" not in keys


def test_bolt_circle_spec_gets_helper_and_import_line():
    note = api_ref.api_reference_note("a disc 60mm with 3 holes equally spaced on a 44mm "
                                      "bolt circle")
    assert "bolt_circle(count, bolt_circle_d, hole_d, depth" in note
    assert "from b123d.domain import bolt_circle" in note
    assert note.startswith("API REFERENCE")


def test_fewshot_code_can_trigger_a_group():
    assert api_ref.select_keys("a small plate") == []
    keys = api_ref.select_keys("a small plate", ["result -= cross_bore(5, 40)"])
    assert keys == ["cross_bore"]


def test_gear_spec_gets_spur_gear():
    assert "spur_gear" in api_ref.select_keys("a small spur gear, 12 teeth, module 1.0")


def test_wired_into_inject_retrieval_notes(monkeypatch):
    import cad_engine as engine
    monkeypatch.setattr(engine, "_load_fewshots", lambda spec: [])
    monkeypatch.setattr(engine, "cad_retrieval", None)
    spec = "a flange with 4 holes on a 50mm bolt circle and a 1mm chamfer"
    notes = engine.retrieval_notes_for(spec)
    assert any(n.startswith("API REFERENCE") and "bolt_circle(" in n for n in notes)
    monkeypatch.setenv("CAD_API_REF", "0")
    assert not any(n.startswith("API REFERENCE") for n in engine.retrieval_notes_for(spec))


def test_plain_spec_adds_no_note(monkeypatch):
    import cad_engine as engine
    monkeypatch.setattr(engine, "_load_fewshots", lambda spec: [])
    monkeypatch.setattr(engine, "cad_retrieval", None)
    assert engine.retrieval_notes_for("a 20mm cube") == []


def test_edge_group_not_triggered_by_fewshot_code_alone():
    assert api_ref.select_keys("a small plate", ["result = fillet(r.edges(), radius=1)"]) == []

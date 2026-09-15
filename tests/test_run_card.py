import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("run_card", HERE / "scripts" / "run_card.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_geometry_of_oneshot_maps_bbox_to_bbox_mm():
    # fluid_gen's `facts` dict IS cad_engine.parse_facts's output — its size key is
    # `bbox`, not `bbox_mm` (score_acceptance's key). This mapping is the whole point of
    # Task 8's "adapt the key mappings to the real interfaces" instruction.
    res = {"facts": {"solids": 1, "bbox": [80.0, 60.0, 20.0], "faces": 10,
                      "cyl_faces": 4, "through_holes": 4}}
    geom = rc.geometry_of(res, "oneshot")
    assert geom == {"solids": 1, "bbox_mm": [80.0, 60.0, 20.0], "faces": 10,
                     "cyl_faces": 4, "through_holes": 4}


def test_geometry_of_oneshot_missing_facts_is_all_none():
    geom = rc.geometry_of({"ok": False, "error": "boom"}, "oneshot")
    assert geom == {"solids": None, "bbox_mm": None, "faces": None,
                     "cyl_faces": None, "through_holes": None}


def test_geometry_of_agent_missing_step_is_all_none():
    # cad_v5's --once --json result carries no top-level geometry facts at all; with no
    # step_local (a build that never produced a STEP) geometry_of must not explode trying
    # to inspect a nonexistent file.
    geom = rc.geometry_of({"ok": False}, "agent")
    assert geom == {"solids": None, "bbox_mm": None, "faces": None,
                     "cyl_faces": None, "through_holes": None}


def test_step_of_oneshot_picks_latest_step_in_build_dir(tmp_path):
    (tmp_path / "build.step").write_text("x")
    res = {"build_dir": str(tmp_path)}
    assert rc.step_of(res, "oneshot") == tmp_path / "build.step"


def test_step_of_agent_uses_step_local():
    res = {"step_local": "/tmp/does-not-need-to-exist/build.step"}
    assert rc.step_of(res, "agent") == Path("/tmp/does-not-need-to-exist/build.step")


def test_contamination_flags_slug_overlap(tmp_path, monkeypatch):
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": "a 100x60x20mm enclosure with 2mm walls"}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    specs_by_suite = {"text-to-cad": [{"id": "01", "spec": "a 100x60x20mm enclosure with 2mm walls"}]}
    clashes = rc.contamination(specs_by_suite)
    assert len(clashes) == 1 and clashes[0].startswith("text-to-cad/01:")


def test_contamination_clean_when_no_overlap(tmp_path, monkeypatch):
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": "a totally different gizmo"}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    specs_by_suite = {"text-to-cad": [{"id": "01", "spec": "a 100x60x20mm enclosure with 2mm walls"}]}
    assert rc.contamination(specs_by_suite) == []


def test_load_suite_unwraps_benchmarks_key(tmp_path, monkeypatch):
    suite_dir = tmp_path / "some-suite"
    suite_dir.mkdir()
    (suite_dir / "specs.json").write_text(json.dumps({"benchmarks": [{"id": "a", "spec": "x"}]}))
    monkeypatch.setattr(rc, "BENCH", tmp_path)
    specs, acc = rc.load_suite("some-suite")
    assert specs == [{"id": "a", "spec": "x"}]
    assert acc == {}


def test_load_suite_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "BENCH", tmp_path)
    specs, acc = rc.load_suite("nope")
    assert specs == [] and acc == {}


def test_run_row_ok_is_always_a_real_bool_not_the_solids_count(monkeypatch):
    # bool(x) and y returns y verbatim when x is truthy (Python's `and` short-circuits to
    # the operand, not to a bool). Found live in the smoke run's rows.jsonl: oneshot rows
    # came out with ok=1 (the int solids count) instead of ok=True. run_row must coerce.
    res = {"ok": True, "facts": {"solids": 1, "bbox": [10, 10, 10], "faces": 6}, "build_dir": "/tmp/nope"}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout: (res, 1.0, ""))
    monkeypatch.setattr(rc, "score_acceptance", lambda geom, crit: {"passed": 0, "total": 0})
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, {}, "oneshot", 60)
    assert row["ok"] is True and type(row["ok"]) is bool


def test_run_row_ok_false_stays_false(monkeypatch):
    res = {"ok": False, "error": "boom"}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout: (res, 1.0, "err"))
    monkeypatch.setattr(rc, "score_acceptance", lambda geom, crit: {"passed": 0, "total": 0})
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, {}, "oneshot", 60)
    assert row["ok"] is False and type(row["ok"]) is bool

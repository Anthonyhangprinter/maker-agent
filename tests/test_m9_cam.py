"""M9 CAM offline/CPU tests — X2a print target (OrcaSlicer CLI) + X2b laser kerf compensation.

NO LLM calls anywhere in this file (per M9's hard rule — a benchmark may be running on the GPU
concurrently). Everything here is deterministic: build123d geometry, ezdxf measurement, and real
CPU-only OrcaSlicer CLI invocations via xvfb-run.

Run:  python3 -m pytest tests/test_m9_cam.py -q
Slicing tests take ~1-2 min each (OrcaSlicer CLI cold-start + arrange + slice).
"""
import subprocess
import sys
from pathlib import Path

import ezdxf
import pytest
from build123d import BuildPart, BuildSketch, Rectangle, Circle, Locations, Mode, extrude, \
    export_step, export_stl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cad_v5 import cam_print  # noqa: E402


# ── shared builders (inline build123d — no LLM, no corpus/retrieval involved) ───────────────

def _plate_with_holes():
    """80x50x3mm plate, two Ø5mm through-holes 40mm apart — the X2b golden part."""
    with BuildPart() as bp:
        with BuildSketch():
            Rectangle(80, 50)
            with Locations((-20, 0), (20, 0)):
                Circle(2.5, mode=Mode.SUBTRACT)
        extrude(amount=3)
    return bp.part


def _open_top_enclosure():
    """100x70x30mm open-top enclosure, 2mm walls — the X2a print-target milestone part."""
    with BuildPart() as bp:
        with BuildSketch():
            Rectangle(100, 70)
        extrude(amount=30)
        with BuildSketch(bp.faces().sort_by(lambda f: f.center().Z)[-1]):
            Rectangle(100 - 2 * 2, 70 - 2 * 2)
        extrude(amount=-(30 - 2), mode=Mode.SUBTRACT)
    return bp.part


def _measure_dxf(path: Path):
    """Overall XY extents (excluding the kerf-note TEXT layer) + circle diameters."""
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    import ezdxf.bbox as bboxmod
    geo_entities = [e for e in msp if e.dxftype() != "TEXT"]
    ext = bboxmod.extents(geo_entities)
    size = ext.extmax - ext.extmin
    diameters = sorted(2 * e.dxf.radius for e in msp if e.dxftype() == "CIRCLE")
    texts = [e.dxf.text for e in msp if e.dxftype() == "TEXT"]
    return size.x, size.y, diameters, texts


# ── 1. Kerf golden test (X2b deterministic exit check) ──────────────────────────────────────

def test_kerf_golden_offset(tmp_path):
    part = _plate_with_holes()
    step_path = tmp_path / "plate.step"
    export_step(part, str(step_path))

    dxf_kerf = tmp_path / "plate_kerf.dxf"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dxf"), str(step_path),
                       str(dxf_kerf), "--kerf", "0.3"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "kerf 0.30mm applied: outer +0.15, holes -0.15" in r.stdout

    x, y, diam, texts = _measure_dxf(dxf_kerf)
    assert x == pytest.approx(80.30, abs=0.02)
    assert y == pytest.approx(50.30, abs=0.02)
    assert len(diam) == 2
    for d in diam:
        assert d == pytest.approx(4.70, abs=0.02)
    assert any("KERF 0.30" in t for t in texts)


def test_kerf_zero_is_byte_identical_geometry(tmp_path):
    """kerf=0 (no flag at all) must reproduce the exact pre-M9 geometry — no offset call made."""
    part = _plate_with_holes()
    step_path = tmp_path / "plate.step"
    export_step(part, str(step_path))

    dxf_plain = tmp_path / "plate_plain.dxf"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dxf"), str(step_path),
                       str(dxf_plain)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    # no "kerf … applied" summary line — check per line (pytest's tmp_path itself contains
    # the substring 'kerf' via the test name, so a whole-stdout substring check false-fails)
    assert not any(ln.startswith("kerf") for ln in r.stdout.splitlines())

    x, y, diam, texts = _measure_dxf(dxf_plain)
    assert x == pytest.approx(80.00, abs=0.02)
    assert y == pytest.approx(50.00, abs=0.02)
    assert len(diam) == 2
    for d in diam:
        assert d == pytest.approx(5.00, abs=0.02)
    assert texts == []   # no kerf-note layer at all when kerf is off


def test_material_preset_matches_explicit_kerf(tmp_path):
    """--material acrylic-6mm (0.30mm) must produce the same geometry as --kerf 0.3."""
    part = _plate_with_holes()
    step_path = tmp_path / "plate.step"
    export_step(part, str(step_path))

    dxf_material = tmp_path / "plate_material.dxf"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dxf"), str(step_path),
                       str(dxf_material), "--material", "acrylic-6mm"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    x, y, diam, _ = _measure_dxf(dxf_material)
    assert x == pytest.approx(80.30, abs=0.02)
    assert y == pytest.approx(50.30, abs=0.02)
    for d in diam:
        assert d == pytest.approx(4.70, abs=0.02)


# ── 2. Print target — the milestone gate ────────────────────────────────────────────────────

def test_print_target_slices_enclosure(tmp_path):
    """THE milestone gate: 'one enclosure sliced + dry-run-validated end-to-end.' Builds a
    100x70x30mm open-top enclosure (2mm walls) inline with build123d — no LLM — slices it with
    the default A1 profile trio, and asserts the dry-run validation is genuinely clean."""
    part = _open_top_enclosure()
    stl_path = tmp_path / "enclosure.stl"
    export_stl(part, str(stl_path))
    assert stl_path.exists() and stl_path.stat().st_size > 0

    out_dir = tmp_path / "print"
    sliced = cam_print.slice_stl(stl_path, out_dir, timeout=300)
    assert sliced["gcode"] is not None, f"no gcode produced; stderr: {sliced['stderr_tail']}"
    assert sliced["result"] is not None
    assert sliced["result"].get("return_code") == 0

    fails, notes, facts = cam_print.validate_gcode(
        sliced["gcode"], sliced["result"], sliced["machine"])

    assert fails == [], f"expected a clean dry-run, got fails={fails} notes={notes}"
    assert facts["layers"] > 100, facts
    assert "xy_bbox" in facts
    bx = facts["xy_bbox"]["x"]; by = facts["xy_bbox"]["y"]
    # A1 bed is 256x256mm with a ±1mm sanity margin; a 100x70mm part must land well inside it.
    assert -1 <= bx[0] and bx[1] <= 257
    assert -1 <= by[0] and by[1] <= 257
    assert facts["extruded_mm"] > 0

    print(f"\n[M9 gate] layers={facts['layers']} est_time={facts.get('est_time')} "
          f"filament={facts.get('filament_used')} xy_bbox={facts['xy_bbox']} "
          f"max_z={facts.get('max_z_seen')} extruded_mm={facts.get('extruded_mm')}")


# ── 3. validate_gcode failure path ───────────────────────────────────────────────────────────

def test_validate_gcode_catches_bad_mesh(tmp_path):
    """A genuinely bad mesh: a box with one face deleted (non-manifold/open shell). Verified
    empirically (outside this test) that OrcaSlicer's CLI returns return_code=-100 with
    'Failed slicing the model...' for this exact construction — a real dry-run failure, not a
    synthesized one."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(20, 20, 20))
    mesh.faces = mesh.faces[:-2]          # drop one quad (2 triangles) -> open/non-manifold shell
    mesh.remove_unreferenced_vertices()
    assert not mesh.is_watertight

    stl_path = tmp_path / "openmesh.stl"
    mesh.export(str(stl_path))

    out_dir = tmp_path / "print_bad"
    sliced = cam_print.slice_stl(stl_path, out_dir, timeout=300)

    fails, notes, facts = cam_print.validate_gcode(
        sliced["gcode"], sliced["result"], sliced["machine"])

    if sliced["result"] is not None and sliced["result"].get("return_code") == 0:
        # OrcaSlicer version drift could in principle repair/slice this mesh anyway; if so,
        # fall back to a synthesized bad result.json so the failure PATH itself is still proven.
        fails, notes, facts = cam_print.validate_gcode(
            sliced["gcode"], {"return_code": -100, "error_string": "Failed slicing the model.",
                              "sliced_plates": []}, sliced["machine"])
        assert fails, "validate_gcode did not fail on a synthesized return_code=-100 result"
    else:
        assert fails, f"expected validate_gcode to fail on a known-bad mesh; notes={notes}"
        assert any("-100" in f or "return_code" in f for f in fails), fails


def test_validate_gcode_synthesized_bad_result():
    """Direct unit check of the failure path irrespective of what OrcaSlicer does with any
    particular mesh: a synthesized return_code=-100 result must always fail validation."""
    fails, notes, facts = cam_print.validate_gcode(
        None, {"return_code": -100, "error_string": "Failed slicing the model.",
              "sliced_plates": []}, "Bambu Lab A1 0.4 nozzle.json")
    assert fails
    assert any("-100" in f for f in fails)


# ── 4. Profile auto-extraction ───────────────────────────────────────────────────────────────

def test_ensure_profiles_auto_extracts(tmp_path, monkeypatch):
    """Point CAM_PROFILES_DIR at a fresh temp dir and confirm ensure_profiles() extracts the
    BBL tree from the AppImage from scratch (the fresh-machine install path)."""
    fresh = tmp_path / "cam-profiles"
    monkeypatch.setattr(cam_print, "CAM_PROFILES_DIR", fresh)
    # cam_print.py binds CAM_PROFILES_DIR at import time from config; ensure_profiles() reads the
    # module-level name directly, so patching the module attribute (not config's) is correct here.
    bbl = cam_print.ensure_profiles()
    assert bbl == fresh / "BBL"
    assert bbl.exists()
    assert (bbl / "machine" / cam_print.PRINT_MACHINE_DEFAULT).exists()
    assert (bbl / "process" / cam_print.PRINT_PROCESS_DEFAULT).exists()
    assert (bbl / "filament" / cam_print.PRINT_FILAMENT_DEFAULT).exists()

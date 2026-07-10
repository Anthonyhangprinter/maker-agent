"""M11 CNC 2.5D toolpath gate test (X2c, DIRECTION.md Part 2).

NO LLM calls anywhere in this file. Everything here is deterministic: build123d geometry for the
gate part, real FreeCAD 1.1.1 AppImage console-mode CAM/Path calls (via cad_v5.cam_cnc), and pure
Python gcode re-parsing for verification. A benchmark may be running on the GPU concurrently —
this file is CPU-only (FreeCAD's Job.recompute() for a part this small is a few seconds; the
AppImage cold-start, ~10-20s, dominates the wall time of each `generate_toolpath` call).

The milestone gate part: a 120x80x10mm plate with a 60x40x5mm-deep rectangular pocket centred on
top, and 4x Ø6mm through-holes at the corners (15mm insets from each edge) --
  half-width  - inset = 60 - 15 = 45  ->  hole X = ±45
  half-height - inset = 40 - 15 = 25  ->  hole Y = ±25
so the four known hole centres are (±45, ±25) — asserted explicitly below, not merely trusted
from the detector (per the standing "verify before claiming" discipline).

Empirical result (see cad_v5/cam_cnc.py's module docstring for the full API writeup): BOTH
Drilling and Pocket run headlessly in FreeCAD 1.1.1 -- the milestone brief's worry that Pocket
might be GUI-bound did not hold up. The gate below therefore exercises pocket+drill together, not
drilling alone.

Run:  python3 -m pytest tests/test_m11_cnc.py -q
FreeCAD runs take ~10-30s each (AppImage cold start + recompute).
"""
import sys
from pathlib import Path

import pytest
from build123d import BuildPart, BuildSketch, Rectangle, Circle, Locations, Mode, extrude, \
    export_step

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cad_v5 import cam_cnc  # noqa: E402

# The four known hole centres, computed from the spec (not just trusted from the detector).
KNOWN_HOLES = [
    {"x": -45.0, "y": -25.0},
    {"x": -45.0, "y": 25.0},
    {"x": 45.0, "y": -25.0},
    {"x": 45.0, "y": 25.0},
]


def _gate_plate():
    """120x80x10mm plate, 60x40x5mm-deep centred pocket, 4x Ø6mm through-holes at 15mm insets."""
    with BuildPart() as bp:
        with BuildSketch():
            Rectangle(120, 80)
        extrude(amount=10)
        top = bp.faces().sort_by(lambda f: f.center().Z)[-1]
        with BuildSketch(top):
            Rectangle(60, 40)
        extrude(amount=-5, mode=Mode.SUBTRACT)
        top2 = bp.faces().sort_by(lambda f: f.center().Z)[-1]
        with BuildSketch(top2):
            with Locations((45, 25), (-45, 25), (45, -25), (-45, -25)):
                Circle(3)
        extrude(amount=-10, mode=Mode.SUBTRACT)
    return bp.part


@pytest.fixture(scope="module")
def gate_build(tmp_path_factory):
    """Build the gate part + its toolpath ONCE (an AppImage cold start is ~10-20s; tests 1-3 all
    exercise the same generated gcode, only test 3 substitutes a fake stock)."""
    tmp_path = tmp_path_factory.mktemp("m11_gate")
    part = _gate_plate()
    step_path = tmp_path / "gate_plate.step"
    export_step(part, str(step_path))

    out_dir = tmp_path / "cnc"
    result = cam_cnc.generate_toolpath(step_path, out_dir)
    return result, step_path


# ── 1 + 2. generate_toolpath + verify_toolpath — the milestone gate ─────────────────────────────

def test_gate_part_generates_and_verifies_clean(gate_build):
    result, step_path = gate_build

    assert result["error"] is None, result["error"]
    assert result["gcode"] is not None and Path(result["gcode"]).exists()
    job_facts = result["job_facts"]

    # job_facts show the drilling op found all 4 holes AND the pocket op (pocket+drill, per the
    # gate's naming) -- not drilling alone, since Pocket proved headless-viable.
    assert job_facts["n_holes_detected"] == 4, job_facts
    assert job_facts["n_holes_drilled"] == 4, job_facts
    assert job_facts["pocket"] is not None, "expected the pocket floor to be auto-detected"
    assert set(job_facts["ops"]) == {"Pocket", "Drilling"}, job_facts["ops"]

    # Cross-check the hole count against scripts/inspect's independent bore detection on the
    # SAME STEP -- two different code paths (FreeCAD's Drillable vs build123d's cylindrical-face
    # classifier) must agree.
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect"), str(step_path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Through-holes: 4" in r.stdout, r.stdout

    fails, notes, facts = cam_cnc.verify_toolpath(
        Path(result["gcode"]), {"holes": KNOWN_HOLES}, job_facts["stock"])

    assert fails == [], f"expected a clean verification, got fails={fails} notes={notes}"
    assert facts["n_drills"] == 4, facts

    # Cut envelope must sit within the stock (120+2*2 x 80+2*2 x -11..top).
    env = facts["cut_envelope"]
    assert -62.0 <= env["x"][0] and env["x"][1] <= 62.0, env
    assert -42.0 <= env["y"][0] and env["y"][1] <= 42.0, env
    assert -11.0 <= env["z"][0], env

    # Every drilled XY within 0.1mm of the four KNOWN hole centres (explicit, not just "detector
    # said so" -- verify_toolpath's own matching already asserts this via `fails == []`, but the
    # facts are re-checked here directly against the coordinates computed from the spec).
    drilled_xy = [(h["x"], h["y"]) for h in job_facts["holes"]]
    for known in KNOWN_HOLES:
        match = min(drilled_xy, key=lambda xy: (xy[0]-known["x"])**2 + (xy[1]-known["y"])**2)
        dist = ((match[0] - known["x"]) ** 2 + (match[1] - known["y"]) ** 2) ** 0.5
        assert dist <= 0.1, f"known hole {known} has no drill within 0.1mm (closest {match})"

    print(f"\n[M11 gate] holes={job_facts['holes']} pocket={job_facts['pocket']} "
          f"cut_envelope={env} cut_length_mm={facts['cut_length_mm']}")


# ── 3. Deliberate failure case: a shrunken fake stock must report an envelope FAIL ──────────────

def test_fake_shrunken_stock_reports_envelope_fail(gate_build):
    result, _ = gate_build
    job_facts = result["job_facts"]

    # Claim the stock is only 60x40mm (real part is 120x80mm) -- the ±45/±25mm corner holes sit
    # well outside that + tool-radius envelope, so the cutting moves must be reported out of bounds.
    fake_stock = dict(job_facts["stock"])
    fake_stock["min"] = [-30.0, -20.0, job_facts["stock"]["min"][2]]
    fake_stock["max"] = [30.0, 20.0, job_facts["stock"]["max"][2]]

    fails, notes, facts = cam_cnc.verify_toolpath(
        Path(result["gcode"]), {"holes": KNOWN_HOLES}, fake_stock)

    assert fails, "expected verify_toolpath to report an envelope failure against a shrunken stock"
    assert any("envelope" in f for f in fails), fails
    print(f"\n[M11 deliberate-failure] fails={fails}")


# ── 4. Drilling-alone path (a plate with holes but no pocket) ───────────────────────────────────

def test_plate_with_holes_only_drilling(tmp_path):
    """A plain plate (no pocket) must still produce a clean drilling-only toolpath -- confirms
    the pocket-detection path degrades gracefully (pocket: None) rather than failing the build
    when a part genuinely has nothing to pocket."""
    with BuildPart() as bp:
        with BuildSketch():
            Rectangle(120, 80)
            with Locations((45, 25), (-45, 25), (45, -25), (-45, -25)):
                Circle(3, mode=Mode.SUBTRACT)
        extrude(amount=10)
    step_path = tmp_path / "plate.step"
    export_step(bp.part, str(step_path))

    result = cam_cnc.generate_toolpath(step_path, tmp_path / "cnc")
    assert result["error"] is None, result["error"]
    job_facts = result["job_facts"]
    assert job_facts["pocket"] is None
    assert job_facts["n_holes_drilled"] == 4, job_facts
    assert job_facts["ops"] == ["Drilling"], job_facts["ops"]

    fails, notes, facts = cam_cnc.verify_toolpath(
        Path(result["gcode"]), {"holes": KNOWN_HOLES}, job_facts["stock"])
    assert fails == [], fails

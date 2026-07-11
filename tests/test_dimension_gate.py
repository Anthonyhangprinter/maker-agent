"""Dimension gate (v5.1): hole_groups-based bolt_circle verification.

Reproduces the 2026-07-11 flange failure: the 7B cut the four bolt holes at Ø20
instead of Ø5, the critic passed it, and the gate couldn't see it (unique-diameter
bores list + total-cyl-face bolt_circle count were both satisfied). With measured
hole groups the gate must hard-fail that build and pass the correct one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import cad_engine as eng  # noqa: E402

SPEC = ("a 60mm diameter, 8mm thick circular flange with a 20mm central through-bore, "
        "four 5mm through-holes equally spaced on a 42mm bolt circle, and a 1mm chamfer "
        "on the top outer edge")

EXPECTED = {
    "solids": 1, "min_holes": 5, "min_through_holes": 5, "bores_mm": [20, 5],
    "feature_checks": [
        {"kind": "bore", "d_mm": 20, "orientation": "axial"},
        {"kind": "bolt_circle", "count": 4, "d_mm": 5, "circle_d_mm": 42},
    ],
}

# Measured by scripts/inspect on the actual broken build (cad-builds/20260711-113736-*)
BROKEN_FACTS = {
    "solids": 1, "cyl_faces": 8, "through_holes": 5, "blind_holes": 0,
    "bores": [20.0, 60.0],
    "hole_groups": [{"d": 20.0, "n": 1, "through": 1, "circle_d": 0.0},
                    {"d": 20.0, "n": 4, "through": 4, "circle_d": 42.0}],
    "bore_axes": [(20.0, "axial")],
    "volume": 10218.6, "bbox": [59.1, 59.1, 8.0],
}

# Measured on the correct part (text-to-cad plugin build, flange_before.step)
CORRECT_FACTS = {
    "solids": 1, "cyl_faces": 6, "through_holes": 5, "blind_holes": 0,
    "bores": [5.0, 20.0, 60.0],
    "hole_groups": [{"d": 5.0, "n": 4, "through": 4, "circle_d": 42.0},
                    {"d": 20.0, "n": 1, "through": 1, "circle_d": 0.0}],
    "bore_axes": [(20.0, "axial"), (5.0, "axial")],
    "volume": 19341.0, "bbox": [59.96, 59.98, 8.0],
}


def test_broken_flange_hard_fails():
    hard, _soft = eng.verify_expected(BROKEN_FACTS, EXPECTED, spec=SPEC)
    assert any("4× Ø5mm" in h for h in hard), f"expected the wrong-size bolt holes to hard-fail, got hard={hard}"
    # the failure message must quote the measured groups so the coder can fix the radius
    assert any("Ø20×4" in h for h in hard)


def test_correct_flange_passes():
    hard, soft = eng.verify_expected(CORRECT_FACTS, EXPECTED, spec=SPEC)
    assert hard == [], f"correct part must not hard-fail: {hard}"
    assert not any("Ø5" in s and "asks for" in s for s in soft), f"no missing-hole advisory expected: {soft}"


def test_uncorroborated_count_stays_advisory():
    # Brief hallucinates a 6-hole circle the spec never asked for → advisory, never a block
    exp = {"solids": 1,
           "feature_checks": [{"kind": "bolt_circle", "count": 6, "d_mm": 5, "circle_d_mm": 42}]}
    hard, soft = eng.verify_expected(BROKEN_FACTS, exp,
                                     spec="a flange with some holes")  # no 6, no 5 in spec
    assert hard == [], f"uncorroborated check must not hard-fail: {hard}"
    assert soft, "should still advise"


def test_wrong_circle_diameter_is_advisory():
    facts = dict(CORRECT_FACTS)
    facts["hole_groups"] = [{"d": 5.0, "n": 4, "through": 4, "circle_d": 30.0},
                            {"d": 20.0, "n": 1, "through": 1, "circle_d": 0.0}]
    hard, soft = eng.verify_expected(facts, EXPECTED, spec=SPEC)
    assert hard == []
    assert any("Ø30" in s and "Ø42" in s for s in soft), f"expected circle-Ø advisory: {soft}"


def test_no_hole_groups_falls_back_to_cyl_faces():
    facts = {"solids": 1, "cyl_faces": 2}
    hard, soft = eng.verify_expected(facts, EXPECTED, spec=SPEC)
    assert hard == []
    assert any("bolt circle" in s for s in soft)


BROKEN_CHAMFER_SRC = """\
from build123d import *
from b123d.domain import bolt_circle
chamfer = 1
result = Cylinder(radius=30, height=8)
try:
    result = chamfer(result.edges().filter_by(GeomType.CIRCLE).group_by(Axis.Z)[-1], length=chamfer)
except:
    pass
"""


def test_fix_feature_guards_shadow_and_unguard():
    # Real 30B output shape (2026-07-11): `chamfer = 1` shadows the function AND the call is
    # guarded — the feature dies silently. The AST fixer must repair both.
    fixed = eng._fix_feature_guards(BROKEN_CHAMFER_SRC, wants=frozenset({"chamfer"}))
    assert "chamfer_mm = 1" in fixed, fixed
    assert "length=chamfer_mm" in fixed, fixed
    assert "try:" not in fixed, fixed
    assert "chamfer(result" in fixed  # the call still targets the real function
    compile(fixed, "<fixed>", "exec")  # stays valid python


def test_fix_feature_guards_keeps_decorative_guard():
    # A fillet the spec never asked for may stay guarded (wants excludes it).
    src = "result = 1\ntry:\n    result = fillet(x, radius=3)\nexcept Exception as e:\n    print(e)\n"
    out = eng._fix_feature_guards(src, wants=frozenset({"chamfer"}))
    assert "try:" in out


def test_fix_feature_guards_no_change_passthrough():
    src = "result = Cylinder(radius=30, height=8)\n"
    assert eng._fix_feature_guards(src, wants=frozenset({"chamfer"})) == src


def test_missing_chamfer_is_advisory():
    facts = dict(CORRECT_FACTS)
    facts["cone_faces"] = 0
    hard, soft = eng.verify_expected(facts, EXPECTED, spec=SPEC)
    assert hard == []
    assert any("chamfer" in s for s in soft), f"expected chamfer advisory: {soft}"
    # and it goes quiet when a cone face exists
    facts["cone_faces"] = 1
    _hard, soft2 = eng.verify_expected(facts, EXPECTED, spec=SPEC)
    assert not any("chamfer" in s for s in soft2)

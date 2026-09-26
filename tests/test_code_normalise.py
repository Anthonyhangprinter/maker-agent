"""Unit tests for code_normalise.normalise_api_spelling (spelling/import normalisation run
inside cad_engine._patch_code). Every rewrite has a positive case taken from a real Gemma
baseline crash (benchmarks/results/card/codefirst-scale-2026-09-25) and at least one
negative case where the rewrite must NOT fire."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import code_normalise as cn  # noqa: E402

HDR = "from build123d import *\n"


def norm(src: str) -> str:
    return cn.normalise_api_spelling(src)[0]


# ── (a) Vector .x/.y/.z -> .X/.Y/.Z ─────────────────────────────────────────────────────

def test_vector_case_on_center_call():
    # cfs0473: AttributeError: 'Vector' object has no attribute 'z'
    src = HDR + "sel = r.edges().filter_by(lambda e: abs(e.center().z - 5) < 0.1)\n"
    assert "e.center().Z - 5" in norm(src)


def test_vector_case_on_constructor_position_and_bbox():
    src = HDR + ("v = Vector(1, 2, 3)\na = v.x\nb = loc.position.y\n"
                 "c = part.bounding_box().max.z\nbb = part.bounding_box()\nd = bb.size.x\n")
    out = norm(src)
    assert "a = v.X" in out and "loc.position.Y" in out
    assert "bounding_box().max.Z" in out and "bb.size.X" in out


def test_vector_case_follows_name_bound_to_vector():
    src = HDR + "c = e.center()\nh = c.z + 1\n"
    assert "h = c.Z + 1" in norm(src)


def test_vector_case_leaves_user_variables_alone():
    src = HDR + ("class P:\n    pass\np = P()\np.x = 3\nq = p.x\n"
                 "pt = (1, 2)\nm = {'x': 1}\n"
                 "c = e.center() if flag else other\nh = c.z\n"   # c not provably a Vector
                 "x = 5\ny = x + 1\n")
    assert norm(src) == src


def test_vector_case_ignores_user_vector_class():
    src = "class Vector:\n    x = 1\nv = Vector()\nprint(v.x)\n"
    assert norm(src) == src


def test_uppercase_already_correct_is_untouched():
    src = HDR + "z = e.center().Z\n"
    assert norm(src) == src


# ── (b) method used as attribute ────────────────────────────────────────────────────────

def test_method_as_attribute_center_and_start_point():
    # cfs0251 / cfs0551: 'function' object has no attribute 'z' / 'x'
    src = HDR + ("a = r.edges().filter_by(lambda e: abs(e.center.z - 15) < 0.1)\n"
                 "b = r.edges().filter_by(lambda e: e.start_point.x < -34)\n"
                 "c = e.end_point.Y\n")
    out = norm(src)
    assert "e.center().Z - 15" in out
    assert "e.start_point().X < -34" in out
    assert "e.end_point().Y" in out


def test_method_as_attribute_not_applied_to_other_attributes():
    src = HDR + ("a = obj.centre.x\nb = e.start.x\nc = e.center\nd = e.center()\n"
                 "f = e.radius\n")
    assert norm(src) == src


# ── (c) auto-import ─────────────────────────────────────────────────────────────────────

def test_auto_import_domain_helpers_after_build123d_line():
    # cfs0036 / cfs0366 / cfb20425: NameError on bolt_circle / cross_bore / counterbore_cutter
    src = HDR + ("result = Box(10, 10, 10)\nresult -= bolt_circle(4, 50, 6, 20)\n"
                 "result -= cross_bore(5, 60)\n")
    out = norm(src)
    lines = out.splitlines()
    assert lines[0] == "from build123d import *"
    assert lines[1] == "from b123d.domain import bolt_circle, cross_bore"


def test_auto_import_math_names():
    # cfs0722 / cfs0759: NameError: name 'cos' is not defined
    src = HDR + "x = r * cos(radians(a)) + sin(pi)\n"
    assert "from math import cos, pi, radians, sin" in norm(src)


def test_auto_import_skips_bound_and_already_imported_names():
    src = ("from build123d import *\nfrom b123d.domain import bolt_circle\nimport math\n"
           "def cross_bore(d, l):\n    return None\n"
           "cos = 3\nr = bolt_circle(4, 50, 6, 20)\nq = cross_bore(1, 2)\nw = cos + 1\n"
           "e_val = e if False else 0\n")   # bare `e` is NOT a whitelisted math name
    assert norm(src) == src


def test_auto_import_skips_unknown_star_import():
    src = "from build123d import *\nfrom numpy import *\nx = cos(1)\n"
    assert norm(src) == src


def test_auto_import_respects_math_star():
    src = "from build123d import *\nfrom math import *\nx = cos(1)\n"
    assert norm(src) == src


def test_auto_import_without_build123d_line_goes_after_docstring():
    src = '"""doc"""\nx = cos(1)\n'
    out = norm(src)
    assert out.splitlines()[1] == "from math import cos"


# ── (d) helper keyword missing its _mm suffix ───────────────────────────────────────────

def test_helper_kwarg_suffix():
    # cfs0446: spur_gear() got an unexpected keyword argument 'module'
    src = (HDR + "from b123d.domain import spur_gear\n"
           "gear = spur_gear(teeth=teeth, module=module, width=width, bore_mm=bore_d)\n")
    out = norm(src)
    assert "spur_gear(teeth=teeth, module_mm=module, width_mm=width, bore_mm=bore_d)" in out


def test_helper_kwarg_not_touched_on_user_function_or_correct_names():
    src = (HDR + "def spur_gear(teeth, module, width):\n    return None\n"
           "g = spur_gear(teeth=1, module=2, width=3)\n")
    assert norm(src) == src
    ok = HDR + "from b123d.domain import spur_gear\ng = spur_gear(12, module_mm=2, width_mm=5)\n"
    assert norm(ok) == ok


def test_helper_kwarg_not_duplicated_when_target_already_passed():
    src = (HDR + "from b123d.domain import spur_gear\n"
           "g = spur_gear(12, module_mm=2, module=2, width_mm=5)\n")
    assert norm(src) == src


# ── general ─────────────────────────────────────────────────────────────────────────────

def test_syntax_error_returns_unchanged():
    src = "def broken(:\n  e.center().z\n"
    assert norm(src) == src


def test_idempotent_and_preserves_comments():
    src = HDR + "# keep me\nz = e.center.z  # and me\nr = bolt_circle(3, 40, 4, 10)\n"
    once = norm(src)
    assert norm(once) == once
    assert "# keep me" in once and "# and me" in once


def test_non_ascii_columns():
    # AST col offsets are UTF-8 byte offsets; a non-ASCII char earlier on the line must not
    # shift the edit.
    src = HDR + "s = 'Ø20'; z = e.center().z\n"
    assert "s = 'Ø20'; z = e.center().Z" in norm(src)


def test_patch_code_calls_normaliser(monkeypatch):
    import cad_engine as engine
    src = HDR + "z = e.center().z\n"
    assert "e.center().Z" in engine._patch_code(src)
    monkeypatch.setenv("CAD_NORMALISE", "0")
    assert "e.center().z" in engine._patch_code(src)

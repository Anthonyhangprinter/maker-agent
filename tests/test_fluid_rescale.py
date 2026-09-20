"""fluid_gen.py's `rescale` subcommand — the build123d counterpart of the OpenSCAD
Customizer sliders. Zero LLM (asserted below via engine._USAGE_TOTAL / no _ollama calls),
but these tests DO run the real execute+inspect+render pipeline on tiny fixture scripts
(CPU-only build123d, no GPU, no model server, no build lock) — the fastest way to prove
"restores the previous version on a failed rescale" actually restores real bytes, not a
mocked stand-in. Run: python3 -m pytest tests/test_fluid_rescale.py -q
"""
import importlib.util
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("fluid_gen", HERE / "scripts" / "fluid_gen.py")
fg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fg
_spec.loader.exec_module(fg)


def _build_dir(tmp_path: Path, fixture: str, spec: str) -> Path:
    d = tmp_path / "build"
    d.mkdir()
    shutil.copy(FIXTURES / fixture, d / "build_source.py")
    (d / "fluid.json").write_text(json.dumps({"spec": spec, "history": []}), encoding="utf-8")
    return d


def test_rescale_changes_geometry_and_reports_new_params(tmp_path):
    d = _build_dir(tmp_path, "build123d_simple_plate.py",
                   "a 55mm plate with a 20mm square cutout")
    ns = Namespace(build_dir=str(d), params=json.dumps({"plate_size": 80}))
    res = fg.cmd_rescale(ns)
    assert res["ok"] is True
    assert res["rescaled"] is True
    assert res["facts"]["bbox"][0] == 80.0
    names = {p["name"]: p for p in res["params"]}
    assert names["plate_size"]["value"] == 80.0
    assert "build_source.py" in [p.name for p in d.iterdir()]
    assert "plate_size = 80.0" in (d / "build_source.py").read_text(encoding="utf-8")


def test_rescale_ignores_expression_constants(tmp_path):
    d = _build_dir(tmp_path, "build123d_int_count_and_expr.py", "a manifold")
    ns = Namespace(build_dir=str(d), params=json.dumps({"rib_count": 3, "flange_z": 999}))
    res = fg.cmd_rescale(ns)
    assert res["ok"] is True
    src = (d / "build_source.py").read_text(encoding="utf-8")
    assert "rib_count = 3" in src
    assert "flange_z = -60.5" in src   # expression constant, never substituted


def test_rescale_with_no_substitutable_match_is_a_clean_no_op_error(tmp_path):
    d = _build_dir(tmp_path, "build123d_simple_plate.py", "a plate")
    original = (d / "build_source.py").read_text(encoding="utf-8")
    ns = Namespace(build_dir=str(d), params=json.dumps({"totally_unknown_name": 5}))
    res = fg.cmd_rescale(ns)
    assert res["ok"] is False
    assert "no matching" in res["error"] or "substitutable" in res["error"]
    assert (d / "build_source.py").read_text(encoding="utf-8") == original


def test_rescale_zero_llm(tmp_path, monkeypatch):
    """Rescale must never call the model — flip _ollama into a hard failure and confirm a
    normal rescale still succeeds untouched."""
    d = _build_dir(tmp_path, "build123d_simple_plate.py", "a plate")

    def _boom(*a, **kw):
        raise AssertionError("rescale must be zero-LLM")
    monkeypatch.setattr(fg.engine, "_ollama", _boom)
    ns = Namespace(build_dir=str(d), params=json.dumps({"plate_thickness": 5}))
    res = fg.cmd_rescale(ns)
    assert res["ok"] is True


def test_rescale_that_fails_to_build_restores_the_previous_version(tmp_path):
    d = _build_dir(tmp_path, "build123d_simple_plate.py", "a plate")
    # First, a good rescale establishes a known-good build.step/.png/.stl to restore to.
    good = fg.cmd_rescale(Namespace(build_dir=str(d), params=json.dumps({"plate_size": 60})))
    assert good["ok"] is True
    before_src = (d / "build_source.py").read_bytes()
    before_step = (d / "build.step").read_bytes()

    # A negative box dimension makes build123d's Solid.make_box raise.
    bad = fg.cmd_rescale(Namespace(build_dir=str(d), params=json.dumps({"plate_size": -5})))
    assert bad["ok"] is False
    assert "restored the previous version" in bad["error"]
    assert (d / "build_source.py").read_bytes() == before_src
    assert (d / "build.step").read_bytes() == before_step

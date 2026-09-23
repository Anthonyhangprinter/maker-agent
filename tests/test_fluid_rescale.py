"""fluid_gen.py's `rescale` subcommand — the build123d counterpart of the OpenSCAD
Customizer sliders. Zero LLM (asserted below via engine._USAGE_TOTAL / no _ollama calls),
but these tests DO run the real execute+inspect+render pipeline on tiny fixture scripts
(CPU-only build123d, no GPU, no model server, no build lock) — the fastest way to prove
"restores the previous version on a failed rescale" actually restores real bytes, not a
mocked stand-in. Run: python3 -m pytest tests/test_fluid_rescale.py -q
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
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


def test_materialize_round_trips_a_unicode_degree_symbol_under_an_ascii_locale(tmp_path):
    """Regression test (Required fix #4, 2026-09-24). scripts/fluid_gen.py's _materialize()
    used to write build_source.py via Path.write_text(code) with no explicit encoding --
    exactly the bug lab/harvest.py's own _store_system already hit and fixed (see its
    docstring): "writing it via Path.write_text() with no explicit encoding raised
    UnicodeEncodeError under PYTHONUTF8=0 in a C/POSIX locale". OCCT/OCP native code is
    documented (CLAUDE.md's "process trap") to reset a build process's live locale to C the
    same way; either path lands the caller in a C/ASCII-preferred-encoding process, and any
    code containing 'Ø' (a real "Ø20mm bore" request) then raised. This test reproduces the
    ASCII-locale condition directly with the same harness test_engine_locale.py already
    established (env vars, not a live OCCT flip, which is the reliable way to force it),
    then drives the REAL _materialize() (real build123d, no mocks) with Ø in both the code
    and the spec."""
    unicode_code = "from build123d import *\nresult = Box(10, 10, 10)\n# a Ø20mm bore on the centre axis\n"

    # The child script goes to a FILE, not a `python3 -c "..."` argv string: under the
    # forced ASCII locale below, the interpreter cannot even decode a non-ASCII command
    # LINE (a separate, earlier failure mode than the one this test targets). A .py file is
    # read by Python's own source-decoding (UTF-8 by default, PEP 3120), which is
    # independent of the process locale, so 'Ø' survives the trip to the child regardless.
    #
    # engine.run_step is stubbed out: it shells out to scripts/step, which does its OWN
    # unrelated (and, as of this branch, still unfixed -- out of scope: only fluid_gen.py
    # and lab/harvest.py were in scope for this pass) unencoded Path.write_text of the same
    # code. Stubbing it isolates the one write this test is actually about: _materialize's
    # OWN `(build_dir / "build_source.py").write_text(code, encoding="utf-8")` at the top
    # of the function, which runs (and, before this fix, could raise) before run_step is
    # ever called.
    child_src = HERE / "tests" / "fixtures" / "_ascii_locale_materialize_child.py"
    child_src.write_text(textwrap.dedent(f"""
        import json, locale, sys
        from pathlib import Path
        sys.path.insert(0, {str(HERE)!r})
        sys.path.insert(0, {str(HERE / "scripts")!r})
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fluid_gen", {str(HERE / "scripts" / "fluid_gen.py")!r})
        fg = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fg
        spec.loader.exec_module(fg)

        d = Path({str(tmp_path / "build2")!r})
        d.mkdir(parents=True, exist_ok=True)
        fg.engine.run_step = lambda code, build_dir: (build_dir / "build.step", "")

        out = {{"preferred_encoding": locale.getpreferredencoding(False)}}
        m = fg._materialize({unicode_code!r}, d, {"a plate with a Ø20mm hole"!r})
        out["error"] = m["error"]
        out["written"] = (d / "build_source.py").read_text(encoding="utf-8")
        print(json.dumps(out))
        """), encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C",
        "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0",
    })
    try:
        result = subprocess.run(
            [sys.executable, str(child_src)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120, env=env,
        )
    finally:
        child_src.unlink(missing_ok=True)
    assert result.returncode == 0, (
        f"child crashed (rc={result.returncode}):\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, f"child produced no output; stderr:\n{result.stderr}"
    payload = json.loads(lines[-1])
    assert payload["preferred_encoding"].upper() in ("ANSI_X3.4-1968", "ASCII", "US-ASCII"), (
        f"harness failed to force an ASCII child locale: {payload}"
    )
    assert payload["error"] is None, f"_materialize raised under an ASCII locale: {payload}"
    assert "Ø" in payload["written"]

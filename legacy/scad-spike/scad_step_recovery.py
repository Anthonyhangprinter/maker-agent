"""
scad_step_recovery.py — mesh -> B-rep STEP recovery for the OpenSCAD backend spike (M8).

Pipeline: scripts/scad <part.scad> -o part.csg  (OpenSCAD's own pure-CSG dump, no meshing) then
a FreeCAD console script that hands the .csg to FreeCAD's bundled OpenSCAD-import module
(`importCSG`) — which re-parses the CSG primitive tree and rebuilds it with REAL OpenCASCADE
Part::Box/Part::Cylinder/Part::Cut/Part::MultiFuse features (exact BRep booleans, not meshed
approximations), then exports whatever is left at the top of that feature tree to STEP.

Run this ONCE on the final accepted geometry, never per turn (it is a whole extra process
launch + FreeCAD's own CSG parser, on the order of several seconds — fine once, wasteful as a
per-turn gate). Validation is by re-measurement, not by trusting FreeCAD's own success message:
the recovered STEP is re-imported with build123d and its volume/bbox are diffed against the
STL's own trimesh numbers. Only a build123d-confirmed match sets `step_recovered=True`.

`_freecad_invoke()` below is copied VERBATIM (see its docstring) from
`cad_v5/freecad_export.py::_freecad_invoke` — that module is off-limits to edit in this spike
(concurrent branch), so the ~6-line invocation helper is duplicated here rather than imported,
per the task's ADD-ONLY rule.
"""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = _HERE / "scripts"

FREECAD_TIMEOUT = 300
VOL_TOL = 0.02     # |dV|/V, per the task spec ("tolerance 2%")
BBOX_TOL_ABS = 1.0   # mm
BBOX_TOL_REL = 0.02  # 2%


def _freecad_invoke():
    """DUPLICATED from cad_v5/freecad_export.py::_freecad_invoke (that file is off-limits to
    edit in this spike — see this module's docstring). Keep behaviour identical: prefer a
    `freecadcmd`/`freecad.cmd` on PATH, else the newest ~/Applications/FreeCAD*.AppImage run in
    console mode (`-c`), else None (caller must treat that as "FreeCAD not found")."""
    cmd = shutil.which("freecadcmd") or shutil.which("freecad.cmd")
    if cmd:
        return [cmd]
    apps = sorted(Path.home().glob("Applications/FreeCAD*.AppImage"))
    if apps:
        return [str(apps[-1]), "-c"]
    return None


_FC_SCRIPT_TEMPLATE = """\
import FreeCAD, importCSG, Part
doc = FreeCAD.newDocument({name!r})
importCSG.insert({csg_path!r}, doc.Name)
doc.recompute()

# Top-level = objects nobody else consumes as a Base/Tool (importCSG's tree keeps every
# intermediate primitive/boolean as its own object; only the un-consumed leaf/leaves are the
# actual final result — verified empirically: a Part::Cut's own inputs otherwise look like
# valid solids too and would be mistaken for extra top-level bodies).
tops = [o for o in doc.Objects
        if hasattr(o, 'Shape') and o.Shape.Solids and not o.InList]

if not tops:
    print('FC_RECOVER_FAIL no top-level solid found after importCSG.insert', flush=True)
    import sys; sys.exit(0)

if len(tops) == 1:
    shape = tops[0].Shape
else:
    shape = Part.makeCompound([o.Shape for o in tops])

if shape.isNull() or shape.Volume <= 0:
    print('FC_RECOVER_FAIL reconstructed shape is null/zero-volume', flush=True)
    import sys; sys.exit(0)

shape.exportStep({step_out!r})
print('FC_RECOVER_OK', shape.Volume, flush=True)
import sys; sys.exit(0)
"""


def _run_freecad_recover(csg_path: Path, step_out: Path, work_dir: Path) -> Optional[str]:
    """Runs the FreeCAD console script. Returns None on success, or an error string."""
    invoke = _freecad_invoke()
    if not invoke:
        return "FreeCAD not found (freecadcmd/freecad.cmd on PATH, or ~/Applications/FreeCAD*.AppImage)"
    script = work_dir / "fc_recover.py"
    script.write_text(_FC_SCRIPT_TEMPLATE.format(
        name="scad_recov", csg_path=str(csg_path), step_out=str(step_out)))
    try:
        r = subprocess.run(invoke + [str(script)], capture_output=True, text=True,
                           timeout=FREECAD_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"FreeCAD console timed out after {FREECAD_TIMEOUT}s"
    finally:
        script.unlink(missing_ok=True)
    out = r.stdout or ""
    if "FC_RECOVER_OK" in out:
        if not step_out.exists():
            return "FreeCAD reported success but no STEP file was written"
        return None
    if "FC_RECOVER_FAIL" in out:
        line = next((ln for ln in out.splitlines() if "FC_RECOVER_FAIL" in ln), out)
        return line.strip()
    return (r.stderr or out or "FreeCAD console produced no recognizable output").strip()[-400:]


def _mesh_facts(stl_path: Path) -> dict:
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh", process=True)
    return {"volume": float(mesh.volume),
            "bbox": [float(x) for x in mesh.bounding_box.extents]}


def _step_facts(step_path: Path) -> dict:
    from build123d import import_step
    shape = import_step(str(step_path))
    bb = shape.bounding_box()
    return {"volume": float(shape.volume),
            "bbox": [bb.size.X, bb.size.Y, bb.size.Z]}


def _bbox_close(a: list[float], b: list[float]) -> bool:
    if len(a) != 3 or len(b) != 3:
        return False
    for x, y in zip(sorted(a), sorted(b)):
        if abs(x - y) > max(BBOX_TOL_ABS, BBOX_TOL_REL * max(x, y, 1e-9)):
            return False
    return True


def recover_step(scad_path, stl_path, out_step_path, work_dir=None) -> dict:
    """Run the mesh->STEP recovery pipeline once and VALIDATE it by re-measurement.

    Returns a dict:
      {"step_recovered": bool, "step_local": str|None, "step_recovery_error": str|None,
       "recovered_facts": dict|None}
    On any failure step_recovered is False and step_recovery_error explains exactly why
    (never silently swallowed) — the STL remains the deliverable either way.
    """
    scad_path = Path(scad_path)
    stl_path = Path(stl_path)
    out_step_path = Path(out_step_path)
    own_work_dir = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(
        __import__("tempfile").mkdtemp(prefix="scad_recov_"))
    try:
        csg_path = work_dir / (scad_path.stem + ".csg")
        scad_bin = [sys.executable, str(SCRIPTS_DIR / "scad")]
        r = subprocess.run(scad_bin + [str(scad_path), "-o", str(csg_path)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not csg_path.exists():
            return {"step_recovered": False, "step_local": None,
                    "step_recovery_error": f"CSG export failed: {(r.stderr or r.stdout)[-400:]}",
                    "recovered_facts": None}

        err = _run_freecad_recover(csg_path, out_step_path, work_dir)
        if err:
            return {"step_recovered": False, "step_local": None,
                    "step_recovery_error": f"FreeCAD CSG import/export failed: {err}",
                    "recovered_facts": None}

        try:
            mesh_facts = _mesh_facts(stl_path)
            step_facts = _step_facts(out_step_path)
        except Exception as e:
            return {"step_recovered": False, "step_local": None,
                    "step_recovery_error": f"post-recovery re-measurement failed: {e}",
                    "recovered_facts": None}

        v0, v1 = mesh_facts["volume"], step_facts["volume"]
        vol_ok = v0 > 0 and abs(v1 - v0) / v0 <= VOL_TOL
        bbox_ok = _bbox_close(mesh_facts["bbox"], step_facts["bbox"])
        if not (vol_ok and bbox_ok):
            return {"step_recovered": False, "step_local": None,
                    "step_recovery_error": (
                        f"recovered STEP failed validation vs the STL — "
                        f"volume STL={v0:.2f} STEP={v1:.2f} (ok={vol_ok}), "
                        f"bbox STL={mesh_facts['bbox']} STEP={step_facts['bbox']} (ok={bbox_ok})"),
                    "recovered_facts": {"stl": mesh_facts, "step": step_facts}}

        return {"step_recovered": True, "step_local": str(out_step_path),
                "step_recovery_error": None,
                "recovered_facts": {"stl": mesh_facts, "step": step_facts}}
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

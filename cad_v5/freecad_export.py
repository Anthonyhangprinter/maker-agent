"""E2 — native FreeCAD document with a real, editable feature tree.

Strategy (honest by construction): reconstruct the part as parametric FreeCAD Part
primitives + booleans (Box/Cylinder → Cut chain), with every dimension bound by expression
to a Params spreadsheet — so editing a number in the FreeCAD GUI regenerates the model.
The reconstruction is derived from MEASURED geometry (scripts/inspect facts + hole
positions), then VERIFIED: the generated document re-exports a STEP whose volume/bbox are
diffed against the agent's own STEP. Only a verified reconstruction is called parametric;
anything else falls back to a plain STEP import into a native .FCStd (still opens in
FreeCAD, no editable tree) with `parametric: False` in the result.

v1 scope (deliberately): box-based parts — solid blocks, plates, open-top shells — with
Z-axis through-holes. Round/complex parts take the fallback path until v2.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import log, SCRIPTS_DIR

VOL_TOL = 0.06     # |dV|/V — chamfers/fillets we don't reproduce cost a few percent
BBOX_TOL = 0.02    # per-axis relative


def _freecad_invoke():
    cmd = shutil.which("freecadcmd") or shutil.which("freecad.cmd")
    if cmd:
        return [cmd]
    apps = sorted(Path.home().glob("Applications/FreeCAD*.AppImage"))
    if apps:
        return [str(apps[-1]), "-c"]
    return None


def inspect_step(step_path: Path) -> dict:
    """FACTS_JSON + hole positions from scripts/inspect (no engine import needed)."""
    r = subprocess.run([sys.executable, str(SCRIPTS_DIR / "inspect"), str(step_path)],
                       capture_output=True, text=True, timeout=180)
    out = r.stdout
    m = re.search(r"^FACTS_JSON:\s*(\{.*\})\s*$", out, re.M)
    facts = json.loads(m.group(1)) if m else {}
    holes = []
    m = re.search(r"^Hole detail:\s*(.+)$", out, re.M)
    if m:
        for seg in m.group(1).split(";"):
            hm = re.search(r"r([\d.]+)@\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\):(THROUGH|BLIND)",
                           seg.strip())
            if hm:
                holes.append({"r": float(hm.group(1)), "x": float(hm.group(2)),
                              "y": float(hm.group(3)), "z": float(hm.group(4)),
                              "through": hm.group(5) == "THROUGH"})
    facts["hole_detail"] = holes
    return facts


def _fc_script(name: str, plan: dict, out_fcstd: Path, out_step: Path) -> str:
    """FreeCAD python: Params spreadsheet + Part primitives bound by expressions."""
    L, W, H = plan["bbox"]
    lines = [
        "import FreeCAD, Part",
        f"doc = FreeCAD.newDocument({name!r})",
        "sheet = doc.addObject('Spreadsheet::Sheet', 'Params')",
    ]
    aliases = {"length": L, "width": W, "height": H}
    if plan.get("wall"):
        aliases["wall"] = plan["wall"]
    for h in plan["holes"]:
        aliases[f"hole{h['i']}_d"] = round(2 * h["r"], 3)
    for row, (k, v) in enumerate(aliases.items(), start=1):
        lines += [f"sheet.set('A{row}', {k!r})", f"sheet.set('B{row}', '{v}')",
                  f"sheet.setAlias('B{row}', {k!r})"]
    lines += [
        "doc.recompute()",
        "outer = doc.addObject('Part::Box', 'Outer')",
        "outer.setExpression('Length', 'Params.length')",
        "outer.setExpression('Width', 'Params.width')",
        "outer.setExpression('Height', 'Params.height')",
        f"outer.Placement.Base = FreeCAD.Vector({-L/2}, {-W/2}, {plan['zmin']})",
        "solid = outer",
    ]
    if plan.get("wall"):
        lines += [
            "inner = doc.addObject('Part::Box', 'Cavity')",
            "inner.setExpression('Length', 'Params.length - 2*Params.wall')",
            "inner.setExpression('Width', 'Params.width - 2*Params.wall')",
            "inner.setExpression('Height', 'Params.height')",
            "inner.setExpression('Placement.Base.x', '-(Params.length - 2*Params.wall)/2')",
            "inner.setExpression('Placement.Base.y', '-(Params.width - 2*Params.wall)/2')",
            f"inner.Placement.Base = FreeCAD.Vector(0, 0, {plan['zmin'] + plan['wall']})",
            "cut = doc.addObject('Part::Cut', 'Hollow')",
            "cut.Base = solid; cut.Tool = inner",
            "solid = cut",
        ]
    for h in plan["holes"]:
        i = h["i"]
        lines += [
            f"c{i} = doc.addObject('Part::Cylinder', 'Hole{i}')",
            f"c{i}.setExpression('Radius', 'Params.hole{i}_d / 2')",
            f"c{i}.Height = {H + 20}",
            f"c{i}.Placement.Base = FreeCAD.Vector({h['x']}, {h['y']}, {plan['zmin'] - 10})",
            f"k{i} = doc.addObject('Part::Cut', 'CutHole{i}')",
            f"k{i}.Base = solid; k{i}.Tool = c{i}",
            f"solid = k{i}",
        ]
    lines += [
        "doc.recompute()",
        "assert not solid.Shape.isNull() and solid.Shape.Volume > 0, 'reconstruction degenerate'",
        f"doc.saveAs({str(out_fcstd)!r})",
        f"solid.Shape.exportStep({str(out_step)!r})",
        "print('FC_EXPORT_OK', solid.Shape.Volume, flush=True)",
        "import sys; sys.exit(0)",   # console mode stays interactive otherwise — never exits
    ]
    return "\n".join(lines) + "\n"


def convert(step_path: Path, name: str) -> dict:
    """STEP -> parametric .FCStd when the reconstruction verifies; import-only otherwise."""
    invoke = _freecad_invoke()
    if not invoke:
        raise RuntimeError("FreeCAD not found (freecadcmd or ~/Applications/FreeCAD*.AppImage).")
    out = step_path.with_suffix(".FCStd")
    facts = inspect_step(step_path)
    plan = None
    holes = [h for h in facts.get("hole_detail", []) if h["through"]]
    if (facts.get("solids") == 1 and facts.get("bbox")
            and len(holes) == len(facts.get("hole_detail", []))
            and facts.get("cone_faces", 0) <= 8):
        wall = min(facts["walls"]) if facts.get("walls") else None
        plan = {"bbox": facts["bbox"], "zmin": -facts["bbox"][2] / 2,
                "wall": wall if (wall and wall < min(facts["bbox"][:2]) / 4) else None,
                "holes": [{**h, "i": i} for i, h in enumerate(holes)]}

    if plan:
        re_step = step_path.with_suffix(".fc_reexport.step")
        script = step_path.with_suffix(".fc_build.py")
        script.write_text(_fc_script(name, plan, out, re_step))
        try:
            r = subprocess.run(invoke + [str(script)], capture_output=True, text=True,
                               timeout=300, stdin=subprocess.DEVNULL)
            if "FC_EXPORT_OK" in r.stdout and re_step.exists():
                got = inspect_step(re_step)
                v0, v1 = facts.get("volume", 0), got.get("volume", 0)
                bb0, bb1 = facts.get("bbox", []), got.get("bbox", [])
                vol_ok = v0 and abs(v1 - v0) / v0 <= VOL_TOL
                bb_ok = (len(bb0) == 3 == len(bb1) and
                         all(abs(a - b) <= max(0.2, BBOX_TOL * a) for a, b in zip(bb0, bb1)))
                holes_ok = got.get("through_holes", -1) >= len(plan["holes"])
                if vol_ok and bb_ok and holes_ok:
                    log.info("[v5] FreeCAD parametric tree VERIFIED (dV %.1f%%, %d holes): %s",
                             100 * abs(v1 - v0) / v0, len(plan["holes"]), out)
                    return {"fcstd_local": str(out), "parametric": True,
                            "params": sorted({"length", "width", "height"}
                                             | ({"wall"} if plan.get("wall") else set())
                                             | {f"hole{h['i']}_d" for h in plan["holes"]}),
                            "verified": {"dV_pct": round(100 * abs(v1 - v0) / max(v0, 1), 2)}}
                log.warning("[v5] FreeCAD reconstruction failed verification "
                            "(vol_ok=%s bbox_ok=%s holes_ok=%s) — falling back to import.",
                            vol_ok, bb_ok, holes_ok)
            else:
                log.warning("[v5] FreeCAD reconstruction errored — falling back to import: %s",
                            (r.stderr or r.stdout).strip()[-200:])
        finally:
            script.unlink(missing_ok=True)
            re_step.unlink(missing_ok=True)

    script = step_path.with_suffix(".fc_import.py")
    script.write_text(
        "import FreeCAD, Import\n"
        f"doc = FreeCAD.newDocument({name!r})\n"
        f"Import.insert({str(step_path)!r}, doc.Name)\n"
        f"doc.saveAs({str(out)!r})\n"
        "print('FC_IMPORT_OK', flush=True)\n"
        "import sys; sys.exit(0)\n")
    try:
        r = subprocess.run(invoke + [str(script)], capture_output=True, text=True, timeout=300,
                           stdin=subprocess.DEVNULL)
        if "FC_IMPORT_OK" not in r.stdout or not out.exists():
            raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    finally:
        script.unlink(missing_ok=True)
    log.info("[v5] FreeCAD document (import-only): %s", out)
    return {"fcstd_local": str(out), "parametric": False}

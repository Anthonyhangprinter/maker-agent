"""M11 — CNC 2.5D toolpaths via FreeCAD's CAM/Path workbench (X2c, DIRECTION.md Part 2).

Deterministic, no LLM. One FreeCAD console script drives the whole pipeline: import STEP ->
create a Job with a box Stock (part bbox + margins) -> auto-detect drillable holes and a pocket
floor from the MEASURED geometry -> add Drilling (+ Pocket, if the part has one) operations ->
post-process to gcode. `verify_toolpath()` then re-parses that gcode in pure Python (no FreeCAD)
and checks it against the stock envelope and the part's known hole centres — the same
"don't trust a tool's silent success, MEASURE the actual output" religion as `cam_print.py` (M9)
and `scripts/inspect`.

Framing (required everywhere this module's output is surfaced): the result is a VERIFIED
SIMULATED toolpath. Nothing here cuts real material — simulated, not cut.

── EMPIRICAL FINDINGS (FreeCAD 1.1.1 AppImage, console mode `-c script.py`, probed 2026-07-10) ──
FreeCAD 1.x renamed the classic "Path" workbench to "CAM" and gutted the old PathScripts.PathJob /
PathPocket / PathProfile / PathDrilling GUI-facing modules (`PathScripts` now only ships
PathPropertyBag/PathUtils). The real headless-scriptable API lives under the `Path` package:

  Path.Main.Job          .Create(name, base)              base = [doc_object, ...] (the part solids)
  Path.Main.Stock        .CreateBox(job, extent=Vector, placement=Placement)
  Path.Tool.toolbit      .ToolBit.from_shape_id("drill.fcstd" | "endmill.fcstd" | ...),
                          .set_diameter(FreeCAD.Units.Quantity(mm, Units.Length)), .attach_to_doc(doc)
  Path.Tool.Controller   .Create(name=, tool=<attached tool obj>, toolNumber=N) -> TC object
                          (job.Proxy.addToolController(tc) -- NOT job.addToolController, that
                          method lives on the Job's Proxy, not the FeaturePython doc object itself)
  Path.Base.Drillable    .getDrillableTargets(obj, ToolDiameter=, vector=Vector(0,0,1))
                          -> [(obj, "FaceN"), ...] for every Z-oriented cylindrical face big
                          enough for the given tool -- this IS FreeCAD's hole auto-detection,
                          reused by Path.Op.Drilling.ObjectDrilling.findAllHoles internally.
  Path.Op.Drilling       .Create(name, parentJob=job) -- creates the op AND immediately calls
                          findAllHoles() itself; no separate detection call needed on our side.
  Path.Op.Pocket         .Create(name, parentJob=job) -- unlike Drilling, Pocket has NO "find all
                          pockets" analog. Its `.Base` must be set MANUALLY to [(part_obj,
                          ["FaceN"])] naming the pocket FLOOR face. We auto-detect that face
                          ourselves (see `_is_pocket_floor` in the generated script): a planar
                          face, normal ~+Z, whose Z sits strictly between the stock's top and
                          bottom, whose XY footprint does NOT touch the part's outer XY boundary
                          (an enclosed cavity floor, not an edge notch/step).
  Path.Post.scripts.linuxcnc_post / grbl_post
                          both ship inside the AppImage as plain `export(objectslist, filename,
                          argstring) -> str` functions (the pre-1.x classic post-processor API,
                          still present alongside the newer class-based Path.Post.Processor).
                          linuxcnc_post is used here: it emits native G81/G82/G83 canned drill
                          cycles, which `verify_toolpath` parses back deterministically.

VERIFIED HEADLESS RESULT: contrary to the milestone brief's worry that Pocket might be GUI-bound,
**Pocket runs headlessly with zero GUI** -- proven on the exact gate part (120x80x10mm plate,
60x40x5mm pocket, 4x Ø6mm through-holes): the Pocket op generated a 114-command raster-stepover
toolpath (Z6->Z5, i.e. exactly the pocket's top-to-floor cut), and Drilling generated 4 G81 canned
cycles at the 4 measured hole centres. Both operations post-process together in one job (pocket
first, then drilling) with `Path.Post.scripts.linuxcnc_post.export([pocket_op, drill_op], ...)`.

UNDOCUMENTED HEADLESS BUG (the real technical risk, found empirically): `Path.Op.Base.__init__`
calls `self.setDefaultValues(obj)`, which calls `PathUtils.findToolController(obj, self)` with
`name=None` always -- there is no way to pass which tool controller a new op should default to.
That function's logic is:
    if len(controllers) == 1: tc = controllers[0]                    # fine
    elif name is not None: tc = [...][0]                             # fine (never hit -- name=None)
    elif UserInput: tc = UserInput.chooseToolController(controllers) # GUI-only
    return tc
In console mode `PathScripts.PathUtils.UserInput is None` (no GUI), so when a Job has MORE THAN
ONE ToolController and name is None, NONE of the three branches assign `tc` -> raw
`UnboundLocalError: cannot access local variable 'tc' where it is not associated with a value`
inside FreeCAD's own PathUtils.py. This bites immediately: `Path.Main.Job.Create()` ALWAYS
auto-adds a default "TC: 5mm Endmill" tool controller (even with `templateFile=None`), so a job
has >= 2 tool controllers the moment a second one is added for drilling/milling. WORKAROUND
(`_scoped_tool_controller` in the generated script): temporarily narrow `job.Tools.Group` down to
the ONE tool controller intended for the upcoming op, call that op's `*.Create()`, then restore
the full list. This sidesteps the ambiguous-selection path entirely (len(controllers) == 1 always
holds at Create()-time) and needs no GUI, no monkeypatching, and no FreeCAD source changes.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path as _Path

from .config import log, CNC_TIMEOUT
from .freecad_export import _freecad_invoke

# Retained for readability at call sites (this module always uses the pathlib.Path type).
Path = _Path


# ── FreeCAD console script generation ────────────────────────────────────────────

def _fc_script(step_path: _Path, gcode_path: _Path, facts_path: _Path, *,
               stock_margin_xy: float, stock_margin_z: float,
               tool_dia: float, drill_dia, safe_z: float) -> str:
    """Build the self-contained FreeCAD python script. All caller values are simple numbers/paths
    interpolated with repr() -- no user-controlled strings ever reach this (step_path/out_dir come
    from the agent's own build pipeline, never raw user text)."""
    drill_dia_literal = "None" if drill_dia is None else repr(float(drill_dia))
    return f'''
import sys, json
import FreeCAD as App
import Part, Import

STEP_PATH = {str(step_path)!r}
GCODE_PATH = {str(gcode_path)!r}
FACTS_PATH = {str(facts_path)!r}
MARGIN_XY = {float(stock_margin_xy)!r}
MARGIN_Z = {float(stock_margin_z)!r}
TOOL_DIA = {float(tool_dia)!r}
DRILL_DIA_PARAM = {drill_dia_literal}
SAFE_Z = {float(safe_z)!r}

doc = App.newDocument("cnc_build")
Import.insert(STEP_PATH, doc.Name)
doc.recompute()

part_obj = None
for o in doc.Objects:
    if hasattr(o, "Shape") and o.Shape.Solids:
        part_obj = o
        break
if part_obj is None:
    print("CNC_BUILD_ERROR: no solid body found in the STEP", flush=True)
    sys.exit(1)

bb = part_obj.Shape.BoundBox

# ── Pocket floor auto-detection: planar, +Z-facing, strictly between stock top/bottom, and
# fully enclosed (its XY footprint does not touch the part's outer XY boundary) -- distinguishes
# a real pocket cavity floor from a step/shelf open to the edge. ──────────────────────────────
def _is_pocket_floor(face, bbox, margin=0.5):
    surf = face.Surface
    if not isinstance(surf, Part.Plane):
        return False
    if abs(surf.Axis.z) < 0.99:
        return False
    z = face.CenterOfMass.z
    if abs(z - bbox.ZMin) < margin or abs(z - bbox.ZMax) < margin:
        return False
    fbb = face.BoundBox
    if fbb.XMin <= bbox.XMin + margin or fbb.XMax >= bbox.XMax - margin:
        return False
    if fbb.YMin <= bbox.YMin + margin or fbb.YMax >= bbox.YMax - margin:
        return False
    return True

pocket_candidates = []
for i, f in enumerate(part_obj.Shape.Faces, start=1):
    if _is_pocket_floor(f, bb):
        pocket_candidates.append((i, f.Area, f.CenterOfMass.z))
pocket_candidates.sort(key=lambda c: -c[1])
pocket_face_idx = pocket_candidates[0][0] if pocket_candidates else None
pocket_floor_z = pocket_candidates[0][2] if pocket_candidates else None

# ── Hole auto-detection: ALL Z-oriented cylindrical faces regardless of size (ToolDiameter=0.01
# so nothing is filtered out yet) -- this is the ground truth used to size the drill bit. ─────
import Path.Base.Drillable as Drillable
all_targets = Drillable.getDrillableTargets(part_obj, ToolDiameter=0.01, vector=App.Vector(0, 0, 1))
hole_info = []
for (obj_, fname) in all_targets:
    face = obj_.getSubObject(fname)
    try:
        c = face.Surface.Center
        r = face.Surface.Radius
    except Exception:
        continue
    hole_info.append({{"face": fname, "x": round(c.x, 4), "y": round(c.y, 4), "d": round(2 * r, 4)}})

drill_dia = DRILL_DIA_PARAM
if drill_dia is None and hole_info:
    # 95% of the smallest detected hole so the bit always physically fits every hole found.
    drill_dia = round(min(h["d"] for h in hole_info) * 0.95, 3)

# ── Job / Stock ────────────────────────────────────────────────────────────────────────────
import Path.Main.Job as Job
job = Job.Create("Job", [part_obj])

import Path.Main.Stock as Stock
stock_extent = App.Vector(bb.XLength + 2 * MARGIN_XY, bb.YLength + 2 * MARGIN_XY, bb.ZLength + 2 * MARGIN_Z)
stock_origin = App.Vector(bb.XMin - MARGIN_XY, bb.YMin - MARGIN_XY, bb.ZMin - MARGIN_Z)
stock = Stock.CreateBox(job, extent=stock_extent,
                         placement=App.Placement(stock_origin, App.Rotation()))
job.Stock = stock

stock_min = [stock_origin.x, stock_origin.y, stock_origin.z]
stock_max = [stock_origin.x + stock_extent.x, stock_origin.y + stock_extent.y,
             stock_origin.z + stock_extent.z]
clearance_z = stock_max[2] + SAFE_Z
safe_height = stock_max[2] + SAFE_Z / 2.0

# ── Tool controllers ───────────────────────────────────────────────────────────────────────
import Path.Tool.toolbit as ToolBitMod
import Path.Tool.Controller as Controller
ToolBit = ToolBitMod.ToolBit

def _make_tc(shape_id, dia, name, number):
    tb = ToolBit.from_shape_id(shape_id)
    tb.set_diameter(App.Units.Quantity(dia, App.Units.Length))
    tool_obj = tb.attach_to_doc(doc)
    tc = Controller.Create(name=name, tool=tool_obj, toolNumber=number)
    job.Proxy.addToolController(tc)
    return tc

tc_mill = _make_tc("endmill.fcstd", TOOL_DIA, "TC: mill", 1)
tc_drill = _make_tc("drill.fcstd", drill_dia, "TC: drill", 2) if (drill_dia and hole_info) else None

full_tools = list(job.Tools.Group)

def _scoped_create(tc, create_fn):
    """WORKAROUND for the FreeCAD 1.1.1 headless multi-ToolController bug documented in this
    module's docstring: narrow job.Tools.Group to just `tc` for the duration of the op's
    Create() call (so PathUtils.findToolController's len(controllers)==1 branch always fires,
    with no GUI needed), then restore the full tool list."""
    job.Tools.Group = [tc]
    try:
        return create_fn()
    finally:
        job.Tools.Group = full_tools

ops = []
pocket_facts = None
if pocket_face_idx is not None:
    import Path.Op.Pocket as Pocket
    def _mk_pocket():
        op = Pocket.Create("Pocket", parentJob=job)
        op.Base = [(part_obj, [f"Face{{pocket_face_idx}}"])]
        return op
    pocket_op = _scoped_create(tc_mill, _mk_pocket)
    ops.append(pocket_op)
    pocket_facts = {{"face": f"Face{{pocket_face_idx}}", "z": round(pocket_floor_z, 3),
                    "area": round(pocket_candidates[0][1], 1)}}

drill_op = None
drilled_holes = []
if tc_drill is not None:
    import Path.Op.Drilling as Drilling
    def _mk_drill():
        return Drilling.Create("Drilling", parentJob=job)
    drill_op = _scoped_create(tc_drill, _mk_drill)
    ops.append(drill_op)
    # Recover the ACTUAL drilled hole set from the op's own Base selection RIGHT NOW -- empirically,
    # doc.recompute() can renumber/clear this Base reference afterward (topological naming), so it
    # must be read immediately after Create(), before any further recompute. NOTE: op.Base groups
    # ALL matched face names for a base object into ONE tuple -- [(obj, ('Face5','Face6',...))] --
    # not one (obj, name) pair per face, so the names need an inner loop.
    for (obj_, fnames) in (drill_op.Base or []):
        for fname in fnames:
            face = obj_.getSubObject(fname)
            try:
                c = face.Surface.Center
                r = face.Surface.Radius
            except Exception:
                continue
            drilled_holes.append({{"face": fname, "x": round(c.x, 4), "y": round(c.y, 4),
                                  "d": round(2 * r, 4)}})

for op in ops:
    if hasattr(op, "ClearanceHeight"):
        op.ClearanceHeight = clearance_z
    if hasattr(op, "SafeHeight"):
        op.SafeHeight = safe_height

if not ops:
    print("CNC_BUILD_ERROR: no pocket and no holes detected -- nothing to machine", flush=True)
    sys.exit(1)

doc.recompute()

# ── Post-process: LinuxCNC post -- ships in the AppImage, runs with zero GUI, native
# G81/G82/G83 canned cycles (easiest to parse back deterministically). ─────────────────────────
import Path.Post.scripts.linuxcnc_post as linuxcnc_post
gcode_str = linuxcnc_post.export(ops, GCODE_PATH, "")

facts = {{
    "stock": {{"min": stock_min, "max": stock_max, "tool_dia": TOOL_DIA, "drill_dia": drill_dia}},
    "part_bbox": [bb.XLength, bb.YLength, bb.ZLength],
    "pocket": pocket_facts,
    "holes": drilled_holes,
    "n_holes_detected": len(hole_info),
    "n_holes_drilled": len(drilled_holes),
    "ops": [o.Name for o in ops],
    "clearance_z": round(clearance_z, 3),
    "safe_height": round(safe_height, 3),
}}
with open(FACTS_PATH, "w") as f:
    json.dump(facts, f)

print("CNC_BUILD_OK", flush=True)
sys.exit(0)
'''


def generate_toolpath(step_path: _Path, out_dir: _Path, *, stock_margin_xy: float = 2.0,
                       stock_margin_z: float = 1.0, tool_dia: float = 6.0,
                       drill_dia: float | None = None, safe_z: float = 5.0) -> dict:
    """Drive one FreeCAD console script: import the STEP, build a Job (Stock = part bbox +
    margins), auto-detect drillable holes (+ a pocket floor, if the part has one), add
    Drilling (+ Pocket) operations, post-process to gcode. Deterministic, no LLM.

    Returns {"gcode": Path|None, "job_facts": {...}, "error": str|None, "log": str}.
    All FreeCAD stdout/stderr is captured to `<out_dir>/freecad_cnc.log` for debugging.
    """
    invoke = _freecad_invoke()
    if not invoke:
        return {"gcode": None, "job_facts": {},
                "error": "FreeCAD not found (freecadcmd or ~/Applications/FreeCAD*.AppImage)."}

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gcode_path = out_dir / "toolpath.ngc"
    facts_path = out_dir / "job_facts.json"
    log_path = out_dir / "freecad_cnc.log"
    script_path = out_dir / "cnc_build.py"

    script_path.write_text(_fc_script(
        step_path, gcode_path, facts_path, stock_margin_xy=stock_margin_xy,
        stock_margin_z=stock_margin_z, tool_dia=tool_dia, drill_dia=drill_dia, safe_z=safe_z))

    try:
        r = subprocess.run(invoke + [str(script_path)], capture_output=True, text=True,
                           timeout=CNC_TIMEOUT, stdin=subprocess.DEVNULL)
        stdout, stderr = r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        stdout, stderr = (e.stdout or "") if isinstance(e.stdout, str) else "", \
                          (e.stderr or "") if isinstance(e.stderr, str) else ""
        log_path.write_text(stdout + "\\n--- STDERR ---\\n" + stderr + "\\n--- TIMEOUT ---\\n")
        return {"gcode": None, "job_facts": {},
                "error": f"FreeCAD CNC build timed out after {CNC_TIMEOUT}s",
                "log": str(log_path)}
    finally:
        script_path.unlink(missing_ok=True)

    log_path.write_text(stdout + "\n--- STDERR ---\n" + stderr)

    if "CNC_BUILD_OK" not in stdout or not gcode_path.exists():
        err = (stderr or stdout).strip()[-600:]
        log.warning("[v5] CNC: FreeCAD toolpath build failed: %s", err)
        return {"gcode": None, "job_facts": {}, "error": f"FreeCAD CNC build failed: {err}",
                "log": str(log_path)}

    job_facts = {}
    if facts_path.exists():
        try:
            job_facts = json.loads(facts_path.read_text())
        except Exception as e:
            log.warning("[v5] CNC: job_facts.json unreadable (%s)", e)

    log.info("[v5] CNC: toolpath generated (%s holes, pocket=%s) -> %s",
             job_facts.get("n_holes_drilled", "?"), bool(job_facts.get("pocket")), gcode_path)
    return {"gcode": gcode_path, "job_facts": job_facts, "error": None, "log": str(log_path)}


# ── Pure-python gcode motion parsing (no FreeCAD) ────────────────────────────────

_TOKEN_RE = re.compile(r"([A-Za-z])(-?\d+\.?\d*)")
_MOTION_G = {0, 1, 2, 3}
_DRILL_G = {81, 82, 83}


def _parse_gcode_motion(gcode_path: _Path):
    """One pass over the gcode: returns (moves, drills, min_z_seen, cut_length_mm).
    moves: list of {"x","y","z","rapid": bool} for every G0/G1/G2/G3/canned-cycle motion.
    drills: list of {"x","y","z"} -- one per canned-cycle (G81/G82/G83) hole, including the
    classic modal-repeat form (a bare X/Y line after a canned cycle is another hole at that XY,
    same Z/R) so posts other than linuxcnc_post (which always reissues the full G81 line) still
    parse correctly."""
    cur_x = cur_y = cur_z = 0.0
    canned_r = None
    canned_active = False
    moves = []
    drills = []
    min_z = float("inf")
    cut_len = 0.0
    prev_cut = None

    with gcode_path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.split(";", 1)[0].split("(", 1)[0].strip()
            if not line:
                continue
            toks = _TOKEN_RE.findall(line)
            if not toks:
                continue
            words: dict[str, float] = {}
            gwords: list[int] = []
            for letter, num in toks:
                letter = letter.upper()
                if letter == "G":
                    try:
                        gwords.append(int(float(num)))
                    except ValueError:
                        pass
                elif letter in ("X", "Y", "Z", "R"):
                    try:
                        words[letter] = float(num)
                    except ValueError:
                        pass

            new_x = words.get("X", cur_x)
            new_y = words.get("Y", cur_y)
            target_z = words.get("Z", cur_z)   # Z explicitly commanded on this line (drill BOTTOM
                                                # for a canned cycle -- not the tool's resting Z)

            is_cancel = 80 in gwords
            is_drill_g = any(g in _DRILL_G for g in gwords)
            is_rapid = 0 in gwords
            is_cut_g = any(g in (1, 2, 3) for g in gwords)

            if is_cancel:
                canned_active = False
            if is_drill_g:
                canned_active = True
            elif canned_active and not gwords and ("X" in words or "Y" in words):
                is_drill_g = True   # modal-repeat drill point

            if "R" in words:
                canned_r = words["R"]

            if is_drill_g:
                drills.append({"x": new_x, "y": new_y, "z": target_z})
                moves.append({"x": new_x, "y": new_y, "z": target_z, "rapid": False})
                min_z = min(min_z, target_z)
                # A canned cycle RETRACTS after each hole -- to the R plane at minimum (G99) or
                # higher, back to the pre-cycle Z (G98). We don't track which mode is active, so
                # the tool's position for subsequent modal lines is taken as R (a conservative
                # floor: using it only ever makes the crash-check MORE cautious, never less --
                # the real G98 retract, if active, is even higher above the material).
                post_z = canned_r if canned_r is not None else target_z
            elif is_rapid:
                moves.append({"x": new_x, "y": new_y, "z": target_z, "rapid": True})
                min_z = min(min_z, target_z)
                post_z = target_z
            elif is_cut_g:
                moves.append({"x": new_x, "y": new_y, "z": target_z, "rapid": False})
                min_z = min(min_z, target_z)
                post_z = target_z
            else:
                post_z = target_z   # non-motion line (e.g. a bare G90/G21) -- modal Z unchanged

            if (is_drill_g or is_cut_g) and prev_cut is not None:
                dx = new_x - prev_cut[0]
                dy = new_y - prev_cut[1]
                dz = target_z - prev_cut[2]
                cut_len += (dx * dx + dy * dy + dz * dz) ** 0.5
            if is_drill_g or is_cut_g:
                prev_cut = (new_x, new_y, target_z)

            cur_x, cur_y, cur_z = new_x, new_y, post_z

    return moves, drills, (min_z if min_z != float("inf") else 0.0), cut_len


def verify_toolpath(gcode_path: _Path, part_facts: dict, stock: dict) -> tuple:
    """Pure-python verification (no FreeCAD) of a generated toolpath. Returns (fails, notes, facts):
      fails — hard problems: cutting moves outside the stock envelope, Z below stock bottom, no
              cutting moves, rapid plunges into material, drill mismatches vs the known holes.
      notes — soft caveats.
      facts — n_drills, cut_envelope, stock, estimated cut-move length.

    `stock`      {"min": [x,y,z], "max": [x,y,z], "tool_dia": float}  (as produced by
                 generate_toolpath's job_facts["stock"], or supplied directly by a caller/test).
    `part_facts` ground truth to check drills against: {"holes": [{"x":..,"y":..}, ...]}.
    Never raises -- a validator that can crash is worse than useless (same discipline as
    cam_print.validate_gcode)."""
    fails: list[str] = []
    notes: list[str] = []
    facts: dict = {}

    gcode_path = _Path(gcode_path)
    if not gcode_path.exists():
        fails.append(f"gcode file not found: {gcode_path}")
        return fails, notes, facts

    moves, drills, min_z_seen, cut_len = _parse_gcode_motion(gcode_path)
    facts["n_drills"] = len(drills)
    facts["stock"] = stock

    if not moves:
        fails.append("no motion commands found in the gcode")
        return fails, notes, facts

    smin = stock.get("min", [0.0, 0.0, 0.0])
    smax = stock.get("max", [0.0, 0.0, 0.0])
    tool_dia = float(stock.get("tool_dia") or 6.0)
    tool_r = tool_dia / 2.0

    cutting = [m for m in moves if not m["rapid"]]
    if not cutting:
        fails.append("no cutting moves (all motion was rapid G0) — nothing would be machined")

    x_lo, x_hi = smin[0] - tool_r, smax[0] + tool_r
    y_lo, y_hi = smin[1] - tool_r, smax[1] + tool_r
    oob = 0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    max_z_cut = float("-inf")
    for m in cutting:
        min_x, max_x = min(min_x, m["x"]), max(max_x, m["x"])
        min_y, max_y = min(min_y, m["y"]), max(max_y, m["y"])
        max_z_cut = max(max_z_cut, m["z"])
        if m["x"] < x_lo or m["x"] > x_hi or m["y"] < y_lo or m["y"] > y_hi:
            oob += 1
    if oob:
        fails.append(f"{oob} cutting move(s) exit the stock XY envelope "
                     f"({x_lo:.1f}..{x_hi:.1f} x {y_lo:.1f}..{y_hi:.1f}mm, incl. "
                     f"{tool_r:.1f}mm tool radius)")

    stock_bottom = smin[2]
    if min_z_seen < stock_bottom - 0.1:
        fails.append(f"toolpath reaches Z={min_z_seen:.2f}mm, below the stock bottom "
                     f"({stock_bottom:.2f}mm, -0.1mm tolerance)")

    # Rapid-plunge crash check: a G0 dropping below the stock's TOP surface while its XY sits
    # over the stock's footprint is a basic collision signature (our own generator always
    # retracts well above stock top first, so this should never fire on a healthy toolpath).
    stock_top = smax[2]
    rapid_plunges = 0
    for m in moves:
        if m["rapid"] and m["z"] < stock_top - 0.05 \
                and smin[0] <= m["x"] <= smax[0] and smin[1] <= m["y"] <= smax[1]:
            rapid_plunges += 1
    if rapid_plunges:
        fails.append(f"{rapid_plunges} rapid (G0) move(s) plunge below the stock top "
                     f"({stock_top:.2f}mm) while over the stock footprint — crash risk")

    if min_x != float("inf"):
        facts["cut_envelope"] = {"x": [round(min_x, 2), round(max_x, 2)],
                                  "y": [round(min_y, 2), round(max_y, 2)],
                                  "z": [round(min_z_seen, 2), round(max_z_cut, 2)]}
    facts["cut_length_mm"] = round(cut_len, 1)

    # Drill verification against the known/expected hole set.
    known = list(part_facts.get("holes") or [])
    if known:
        remaining = list(range(len(known)))
        unmatched_drills = []
        for d in drills:
            best_idx, best_dist = None, None
            for idx in remaining:
                k = known[idx]
                dist = ((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2) ** 0.5
                if best_idx is None or dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx is not None and best_dist <= 0.1:
                remaining.remove(best_idx)
            else:
                unmatched_drills.append(d)
        if unmatched_drills:
            fails.append(f"{len(unmatched_drills)} drilled position(s) don't match any known "
                         f"hole within 0.1mm: {unmatched_drills}")
        if remaining:
            missed = [known[i] for i in remaining]
            fails.append(f"{len(remaining)} known hole(s) were never drilled: {missed}")
    elif drills:
        notes.append("no known hole list supplied — drilled positions unverified against intent")

    return fails, notes, facts

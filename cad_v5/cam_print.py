"""CAM: FDM print target (M9/X2a) — STL → OrcaSlicer CLI → dry-run-validated gcode.

Deterministic, no LLM. Mirrors the discipline of every other stage in this agent: don't trust a
tool's silent success, MEASURE the actual output. OrcaSlicer's own `result.json` is the first
signal (return_code / warning_message), but a slicer can also happily emit gcode for geometry
that isn't really printable on the target machine (out-of-bed, all-air, no extrusion) — so
`validate_gcode()` re-reads the gcode itself: the slicer's own header facts, plus a streamed
motion sanity scan against the machine's bed envelope.

Bambu system profiles (machine/process/filament trios) ship INSIDE the OrcaSlicer AppImage and
are extracted once to `~/.openclaw/cam-profiles/BBL/` — see `ensure_profiles()`. That directory,
once populated, is the install path for a fresh machine; nothing here re-extracts on every call.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config
from .config import log, CAM_PROFILES_DIR, PRINT_MACHINE_DEFAULT, PRINT_PROCESS_DEFAULT, \
    PRINT_FILAMENT_DEFAULT, SLICE_TIMEOUT

# The A1's bed if a machine profile's inherits chain never surfaces the real numbers — logged as
# a note, never silently assumed to be something else.
_FALLBACK_BED_XY = 256.0
_FALLBACK_BED_Z  = 256.0
_BED_MARGIN      = 1.0     # mm of slack either side of the nominal bed for the sanity scan


# ── OrcaSlicer binary discovery ──────────────────────────────────────────────────

def _orca_bin() -> Path:
    """$ORCA_SLICER env → `orca-slicer` on PATH → newest ~/Applications/OrcaSlicer*.AppImage."""
    import os
    env = os.environ.get("ORCA_SLICER")
    if env and Path(env).exists():
        return Path(env)
    which = shutil.which("orca-slicer")
    if which:
        return Path(which)
    candidates = sorted(Path.home().glob("Applications/OrcaSlicer*.AppImage"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "OrcaSlicer not found — set $ORCA_SLICER to a binary/AppImage path, install "
        "`orca-slicer` on PATH, or place an OrcaSlicer*.AppImage under ~/Applications/.")


# ── Profile install (extract-once) ───────────────────────────────────────────────

def ensure_profiles() -> Path:
    """Return `~/.openclaw/cam-profiles/BBL`, auto-extracting the Bambu system profile tree
    from the OrcaSlicer AppImage on first use (a few seconds; CAM_PROFILES_DIR is checked first
    so this is a true one-time cost per machine)."""
    bbl_dir = CAM_PROFILES_DIR / "BBL"
    if bbl_dir.exists() and any(bbl_dir.iterdir()):
        return bbl_dir

    orca = _orca_bin()
    CAM_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("[v5] CAM: extracting Bambu profiles from %s (first use only)...", orca.name)
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([str(orca), "--appimage-extract", "resources/profiles/BBL*"],
                           cwd=td, capture_output=True, text=True, timeout=120)
        extracted_root = Path(td) / "squashfs-root" / "resources" / "profiles"
        extracted_bbl = extracted_root / "BBL"
        if r.returncode != 0 or not extracted_bbl.exists():
            raise RuntimeError(
                f"OrcaSlicer profile extraction failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout or '').strip()[:400]}")
        shutil.copytree(extracted_bbl, bbl_dir, dirs_exist_ok=True)
        bbl_json = extracted_root / "BBL.json"
        if bbl_json.exists():
            shutil.copy(bbl_json, CAM_PROFILES_DIR / "BBL.json")
    log.info("[v5] CAM: profiles installed at %s", bbl_dir)
    return bbl_dir


def _resolve_profile(kind: str, name: str, bbl_dir: Path) -> Path:
    """Bare profile name → `<bbl_dir>/<kind>/<name>`; an absolute path that exists passes
    straight through (lets `~/.openclaw/cad.json` cad.print point at a custom profile)."""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    cand = bbl_dir / kind / name
    if cand.exists():
        return cand
    raise RuntimeError(f"CAM profile not found: {kind}/{name!r} (looked in {bbl_dir / kind})")


# ── Slicing ───────────────────────────────────────────────────────────────────

def slice_stl(stl_path: Path, out_dir: Path, machine: str | None = None,
              process: str | None = None, filament: str | None = None,
              timeout: int = SLICE_TIMEOUT) -> dict:
    """Slice `stl_path` with OrcaSlicer's headless CLI (xvfb-run is required even for CLI-only
    use — verified). Returns {"gcode": Path|None, "result": dict|None, "stderr_tail": str,
    "machine"/"process"/"filament": the resolved profile names actually used}. Never raises for
    a bad-geometry slice (that's `validate_gcode`'s job); DOES raise for infra problems (no
    OrcaSlicer binary, no xvfb-run, unresolvable profile name) — the caller (targets.print_target)
    turns that into a graceful target_error."""
    bbl_dir = ensure_profiles()
    machine  = machine  or PRINT_MACHINE_DEFAULT
    process  = process  or PRINT_PROCESS_DEFAULT
    filament = filament or PRINT_FILAMENT_DEFAULT
    m_path = _resolve_profile("machine", machine, bbl_dir)
    p_path = _resolve_profile("process", process, bbl_dir)
    f_path = _resolve_profile("filament", filament, bbl_dir)

    if shutil.which("xvfb-run") is None:
        raise RuntimeError("xvfb-run not found on PATH — required to run OrcaSlicer headless "
                           "(install the `xvfb` package).")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orca = _orca_bin()
    cmd = ["xvfb-run", "-a", str(orca),
           "--load-settings", f"{m_path};{p_path}",
           "--load-filaments", str(f_path),
           "--slice", "0", "--arrange", "1",
           "--outputdir", str(out_dir), str(stl_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"gcode": None, "result": None,
                "stderr_tail": f"slice timed out after {timeout}s",
                "machine": machine, "process": process, "filament": filament}

    gcode = out_dir / "plate_1.gcode"
    result_path = out_dir / "result.json"
    result = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except Exception as e:
            log.warning("[v5] CAM: result.json unreadable (%s)", e)

    return {"gcode": gcode if gcode.exists() else None, "result": result,
            "stderr_tail": ((proc.stderr or "") + (proc.stdout or ""))[-800:],
            "machine": machine, "process": process, "filament": filament}


# ── Bed envelope (best-effort — walks a bounded `inherits` chain) ───────────────

def _bed_envelope(machine_path: Path, bbl_dir: Path) -> tuple[float, float, float, list[str]]:
    """printable_area/printable_height often live on a PARENT profile (`inherits`), not the leaf
    machine json — walk a few levels up. Falls back to the A1's 256x256x256mm with a note; never
    raises (a lookup miss degrades to the fallback, not a build failure)."""
    notes: list[str] = []
    area = None
    height = None
    path = machine_path
    seen: set[Path] = set()
    for _ in range(6):
        if path is None or path in seen or not path.exists():
            break
        seen.add(path)
        try:
            d = json.loads(path.read_text())
        except Exception:
            break
        if area is None and d.get("printable_area"):
            area = d["printable_area"]
        if height is None and d.get("printable_height"):
            height = d["printable_height"]
        if area is not None and height is not None:
            break
        parent = d.get("inherits")
        path = (bbl_dir / "machine" / f"{parent}.json") if parent else None

    if area:
        xs, ys = [], []
        for pt in area:
            x, y = pt.split("x")
            xs.append(float(x)); ys.append(float(y))
        bed_x, bed_y = max(xs), max(ys)
    else:
        bed_x, bed_y = _FALLBACK_BED_XY, _FALLBACK_BED_XY
        notes.append(f"printable_area not found in {machine_path.name}'s inherits chain — "
                     f"assumed {bed_x:.0f}x{bed_y:.0f}mm (A1 default)")
    if height:
        bed_z = float(height)
    else:
        bed_z = _FALLBACK_BED_Z
        notes.append(f"printable_height not found in {machine_path.name}'s inherits chain — "
                     f"assumed {bed_z:.0f}mm (A1 default)")
    return bed_x, bed_y, bed_z, notes


# ── gcode header parse + motion sanity scan ─────────────────────────────────────

def _parse_header(gcode_path: Path) -> dict:
    """OrcaSlicer's `; key: value[; key2: value2]` HEADER_BLOCK — bounded read, not a full-file
    load. Some lines pack multiple key:value pairs separated by `; `."""
    header: dict[str, str] = {}
    with gcode_path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 400:      # header block is always near the top; this is generous slack
                break
            if not line.startswith(";"):
                continue
            body = line[1:].strip()
            if "HEADER_BLOCK_END" in body:
                break
            for part in body.split("; "):
                if ":" not in part:
                    continue
                k, _, v = part.partition(":")
                header[k.strip().lower()] = v.strip()
    return header


def _parse_filament_used(gcode_path: Path) -> str | None:
    """The `; filament used [..]` lines sit at the very end of the file (after
    EXECUTABLE_BLOCK_END) — seek to the tail (a few KB) instead of loading the whole gcode."""
    try:
        size = gcode_path.stat().st_size
        with gcode_path.open("rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode(errors="replace")
    except Exception:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        for key in ("filament used [g]", "filament used [cm3]", "filament used [mm]"):
            prefix = f"; {key} ="
            if line.lower().startswith(prefix):
                return line.split("=", 1)[1].strip() + f" {key.split('[')[1][:-1]}"
    return None


def validate_gcode(gcode_path: Path | None, result: dict | None,
                    machine_name: str) -> tuple[list[str], list[str], dict]:
    """Deterministic dry-run validation of a slice. Returns (fails, notes, facts):
      fails  — hard problems (bad geometry, out of bounds, no extrusion, slicer error)
      notes  — soft caveats (a lookup fell back to a default) that don't fail the build
      facts  — measured numbers (layers, est_time, filament_used, xy_bbox, max_z_seen)
    Never raises — a validation function that can crash is worse than useless."""
    fails: list[str] = []
    notes: list[str] = []
    facts: dict = {}

    if result is not None:
        rc = result.get("return_code")
        if rc not in (0, None):
            fails.append(f"slicer return_code={rc}: "
                         f"{result.get('error_string') or '(no error_string)'}")
        plates = result.get("sliced_plates")
        if not plates and rc in (0, None):
            fails.append("result.json has no sliced_plates — slicer produced nothing")
        for plate in plates or []:
            warn = (plate.get("warning_message") or "").strip()
            if warn:
                fails.append(f"slicer warning: {warn}")
    else:
        notes.append("no result.json to check (an infra issue, not necessarily a geometry one)")

    if gcode_path is None or not Path(gcode_path).exists():
        fails.append("no gcode file produced")
        return fails, notes, facts
    gcode_path = Path(gcode_path)

    header = _parse_header(gcode_path)
    layers = None
    for key in ("total layer number", "total_layer_number"):
        if key in header:
            try:
                layers = int(header[key])
            except ValueError:
                pass
            break
    if not layers:
        fails.append("gcode header missing 'total layer number'")
    else:
        facts["layers"] = layers

    time_str = header.get("model printing time") or header.get("total estimated time")
    if time_str:
        facts["est_time"] = time_str
    else:
        notes.append("gcode header missing a printing-time estimate")

    if "max_z_height" in header:
        try:
            facts["max_z_height"] = float(header["max_z_height"])
        except ValueError:
            pass

    filament_used = _parse_filament_used(gcode_path)
    if filament_used:
        facts["filament_used"] = filament_used
    else:
        notes.append("filament-used line not found (tail of gcode)")

    # Bed envelope — best-effort; a lookup miss degrades to a note, not a failure.
    bed_x = bed_y = bed_z = None
    try:
        bbl_dir = ensure_profiles()
        m_path = _resolve_profile("machine", machine_name, bbl_dir)
        bed_x, bed_y, bed_z, bed_notes = _bed_envelope(m_path, bbl_dir)
        notes.extend(bed_notes)
    except Exception as e:
        notes.append(f"could not resolve bed envelope ({e}) — motion sanity scan skipped")

    if bed_x is not None:
        # Bounds are enforced on EXTRUDING moves only — the A1's stock start/change gcode
        # legitimately travels outside the printable area (X-48.2 wipe shakes, X267 filament
        # cut; measured on a clean test slice: 134 out-of-bed points, ALL travel, 0 extruding).
        # What must stay on the bed is the deposited material: any G0/G1 that both moves in XY
        # and extrudes. xy_bbox/max_z therefore describe the PART's real printed footprint.
        lo = -_BED_MARGIN
        hi_x, hi_y, hi_z = bed_x + _BED_MARGIN, bed_y + _BED_MARGIN, bed_z + _BED_MARGIN
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        max_z_print = 0.0
        cur_z = 0.0
        oob = 0
        cur_x = cur_y = None
        relative_e = True          # Bambu profiles default to M83 (relative extrusion)
        last_abs_e = 0.0
        extruded = 0.0
        with gcode_path.open("r", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("M83"):
                    relative_e = True
                    continue
                if stripped.startswith("M82"):
                    relative_e = False
                    continue
                if not (line.startswith("G0") or line.startswith("G1")):
                    if line.startswith("G92") and "E" in line:
                        for tok in line.split()[1:]:
                            if tok and tok[0] in ("E", "e"):
                                try:
                                    last_abs_e = float(tok[1:])
                                except ValueError:
                                    pass
                    continue
                x = y = z = e = None
                for tok in line.split(";", 1)[0].split()[1:]:
                    if not tok:
                        continue
                    c, rest = tok[0], tok[1:]
                    try:
                        val = float(rest)
                    except ValueError:
                        continue
                    if c in ("X", "x"): x = val
                    elif c in ("Y", "y"): y = val
                    elif c in ("Z", "z"): z = val
                    elif c in ("E", "e"): e = val
                if x is not None:
                    cur_x = x
                if y is not None:
                    cur_y = y
                if z is not None:
                    cur_z = z
                de = 0.0
                if e is not None:
                    de = max(0.0, e) if relative_e else max(0.0, e - last_abs_e)
                    if not relative_e:
                        last_abs_e = e
                    extruded += de
                if de > 0 and (x is not None or y is not None) \
                        and cur_x is not None and cur_y is not None:
                    if cur_x < lo or cur_x > hi_x or cur_y < lo or cur_y > hi_y:
                        oob += 1
                    min_x, max_x = min(min_x, cur_x), max(max_x, cur_x)
                    min_y, max_y = min(min_y, cur_y), max(max_y, cur_y)
                    max_z_print = max(max_z_print, cur_z)
        if oob:
            fails.append(f"{oob} extruding move(s) outside the {bed_x:.0f}x{bed_y:.0f}mm bed "
                         f"(±{_BED_MARGIN:.0f}mm margin)")
        if max_z_print > hi_z:
            fails.append(f"printed max Z {max_z_print:.1f}mm exceeds printable height "
                         f"{bed_z:.0f}mm (+{_BED_MARGIN:.0f}mm margin)")
        if extruded <= 0:
            fails.append("no positive extrusion (E) found in the gcode — nothing would print")
        if min_x != float("inf"):
            facts["xy_bbox"] = {"x": [round(min_x, 1), round(max_x, 1)],
                                "y": [round(min_y, 1), round(max_y, 1)]}
        facts["max_z_seen"] = round(max_z_print, 2)
        facts["extruded_mm"] = round(extruded, 1)

    return fails, notes, facts

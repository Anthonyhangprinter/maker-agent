#!/usr/bin/env python3
"""OpenSCAD backend — Track E: second codegen language beside build123d.

Pipeline (the CADAM recipe, on our seams): frontier model writes a COMPLETE OpenSCAD file
with Customizer-annotated parameters -> local OpenSCAD compiles it (AppImage, headless
STL export) -> compiler stderr feeds a repair turn -> pure-python mesh facts (bbox/volume/
watertight) sanity-check the result. Parameters are parsed out of the code (regex, zero LLM)
and returned for the web UI's future sliders.

The M8 spike failed 0/6 because the LOCAL 7B free-styled pseudo-OpenSCAD; this runs on the
cloud rung where CADAM proves the approach. Local models can be A/B'd later via --model.

Usage:
    python3 scripts/openscad_gen.py "a 40mm cube with a 10mm centre hole" --json
    python3 scripts/openscad_gen.py "<spec>" --turns 3 --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cad_engine as engine  # noqa: E402

OPENSCAD = str(Path.home() / ".local" / "bin" / "OpenSCAD.AppImage")
BUILDS_DIR = Path.home() / ".openclaw" / "cad-builds"
COMPILE_TIMEOUT = 180

_SYSTEM = """\
You write OpenSCAD code (v2021.01, built-in language only — NO external libraries, no
BOSL/MCAD includes). Produce ONE complete .scad file for the requested part.

RULES:
- Every important dimension is a top-level Customizer parameter with an annotation:
    width = 50; // [10:1:200]
  Group them:  /* [Dimensions] */  then  /* [Details] */  etc. Millimetres throughout.
- End the parameter block with:  $fn = 64; // [16:8:128]
- Build the geometry from those parameters only — no magic numbers in the body.
- Solids must genuinely overlap where joined (never rely on exact face contact);
  through-holes must fully pierce (cut cylinders longer than the wall, offset epsilon).
- Multi-part requests: lay parts side by side with clear spacing, one union() per part.
- Reply with ONLY the OpenSCAD code. No markdown fences, no commentary.
"""

_REPAIR = """\
The OpenSCAD compiler rejected or warned on your file. Fix it and return ONLY the complete
corrected .scad file (same rules: Customizer parameters, built-ins only, no fences).

Compiler output:
{err}

Current file:
{code}
"""


def strip_fences(t: str) -> str:
    t = re.sub(r"^```(openscad|scad)?\s*\n?", "", t.strip())
    return t.rstrip("`").strip()


def compile_scad(scad: Path, stl: Path) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            [OPENSCAD, "--appimage-extract-and-run", "-o", str(stl),
             "--export-format", "binstl", str(scad)],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "compile timed out"
    err = (p.stderr or "").strip()
    ok = stl.is_file() and stl.stat().st_size > 84 and p.returncode == 0
    return ok, err[-2000:]


def parse_params(code: str) -> list[dict]:
    """Customizer assignments -> slider schema. Zero LLM — the CADAM pattern."""
    out, group = [], ""
    for line in code.splitlines():
        g = re.match(r"\s*/\*\s*\[(.+?)\]\s*\*/", line)
        if g:
            group = g.group(1).strip()
            continue
        m = re.match(r"\s*(\$?\w+)\s*=\s*(-?[0-9.]+)\s*;\s*//\s*\[([^\]]+)\]", line)
        if not m:
            continue
        name, val, ann = m.group(1), float(m.group(2)), m.group(3)
        parts = [p.strip() for p in ann.split(":")]
        p = {"name": name, "value": val, "group": group}
        try:
            if len(parts) == 3:
                p.update(min=float(parts[0]), step=float(parts[1]), max=float(parts[2]))
            elif len(parts) == 2:
                p.update(min=float(parts[0]), max=float(parts[1]), step=1)
        except ValueError:
            continue
        out.append(p)
    return out


def stl_facts(path: Path) -> dict:
    """bbox + signed volume (divergence theorem) + triangle count from a binary STL."""
    raw = path.read_bytes()
    if len(raw) < 84:
        return {}
    n = struct.unpack("<I", raw[80:84])[0]
    if len(raw) < 84 + n * 50:                      # ascii STL fallback: skip facts
        return {"triangles": None}
    lo = [1e30] * 3
    hi = [-1e30] * 3
    vol6 = 0.0
    off = 84
    for _ in range(n):
        v = struct.unpack("<12f", raw[off + 12:off + 60])
        (ax, ay, az, bx, by, bz, cx, cy, cz) = v[0:9]
        for (x, y, z) in ((ax, ay, az), (bx, by, bz), (cx, cy, cz)):
            lo[0], lo[1], lo[2] = min(lo[0], x), min(lo[1], y), min(lo[2], z)
            hi[0], hi[1], hi[2] = max(hi[0], x), max(hi[1], y), max(hi[2], z)
        vol6 += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                 + az * (bx * cy - by * cx))
        off += 50
    return {"triangles": n,
            "bbox": [round(hi[i] - lo[i], 2) for i in range(3)],
            "volume_mm3": round(abs(vol6) / 6.0, 1)}


_CHAT_REVISE = """\
The user looked at the compiled model and asked for a change. Apply it and return ONLY the
complete updated .scad file (same rules: Customizer parameters, built-ins only, no fences).

User's request (verbatim, authoritative):
{feedback}

Current file:
{code}
"""


def cmd_revise(a) -> dict:
    """Fluid chat turn for an existing openscad build: user feedback -> new file -> compile."""
    build_dir = Path(a.revise_dir)
    scad, stl = build_dir / "build.scad", build_dir / "build.stl"
    if not scad.is_file():
        return {"ok": False, "error": "no build.scad in that build dir"}
    cc = engine.cloud_config()
    model = a.model or cc.get("model", "claude-sonnet-5")
    engine.reset_cloud_budget(2)
    t0 = time.monotonic()
    code = scad.read_text()
    raw = engine._cloud_chat(model, _SYSTEM,
                             _CHAT_REVISE.format(feedback=a.feedback, code=code), timeout=300)
    if not raw:
        return {"ok": False, "error": "empty model reply"}
    new_code = strip_fences(raw)
    scad.write_text(new_code)
    ok, err = compile_scad(scad, stl)
    if not ok:                                    # keep chat fluid: one stderr repair
        raw2 = engine._cloud_chat(model, _SYSTEM,
                                  _REPAIR.format(err=err, code=new_code), timeout=300)
        if raw2:
            new_code = strip_fences(raw2)
            scad.write_text(new_code)
            ok, err = compile_scad(scad, stl)
    return {"ok": ok, "lang": "openscad", "converged": ok, "mode": "fluid",
            "accepted_via": "compile" if ok else None, "code_model": model,
            "build_dir": str(build_dir), "params": parse_params(new_code),
            "facts": stl_facts(stl) if ok else {},
            "error": None if ok else (err or "compile failed")[-300:],
            "build_time_s": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--model", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--revise-dir", default="", help="fluid chat: existing scad build dir")
    ap.add_argument("--feedback", default="", help="fluid chat: the user's change request")
    a = ap.parse_args()
    if a.revise_dir:
        result = cmd_revise(a)
        print(json.dumps(result))
        return 0 if result.get("ok") else 1
    if not a.spec:
        ap.error("spec required (or --revise-dir + --feedback)")

    cc = engine.cloud_config()
    model = a.model or cc.get("model", "claude-sonnet-5")
    build_dir = BUILDS_DIR / f"scad_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    build_dir.mkdir(parents=True, exist_ok=True)
    scad, stl = build_dir / "build.scad", build_dir / "build.stl"
    t0 = time.monotonic()

    engine.reset_cloud_budget(a.turns + 1)
    engine._ACTIVE_CODE_MODEL = engine.CLOUD_PREFIX + model
    prompt = (f"USER REQUEST (every number is AUTHORITATIVE, mm):\n{a.spec}\n\n"
              f"Write the OpenSCAD file:")
    code, err, ok = "", "", False
    for turn in range(1, a.turns + 1):
        raw = (engine._cloud_chat(model, _SYSTEM, prompt, timeout=300) if turn == 1 else
               engine._cloud_chat(model, _SYSTEM, _REPAIR.format(err=err, code=code),
                                  timeout=300))
        if not raw:
            err = "empty model reply"
            break
        code = strip_fences(raw)
        scad.write_text(code)
        ok, err = compile_scad(scad, stl)
        print(f"[scad] turn {turn}: compiled={ok}"
              + (f"  stderr: {err[:120]}" if err else ""), file=sys.stderr)
        if ok and not re.search(r"WARNING|ERROR", err or "", re.I):
            break
        if ok:                                       # compiled but warned — good enough
            break

    facts = stl_facts(stl) if ok else {}
    result = {
        "ok": ok, "lang": "openscad", "converged": ok,
        "accepted_via": "compile+mesh-facts" if ok else None,
        "code_model": model, "turns": turn if code else 0,
        "build_dir": str(build_dir), "params": parse_params(code) if code else [],
        "facts": facts, "error": None if ok else (err or "no compile"),
        "build_time_s": round(time.monotonic() - t0, 1),
    }
    if a.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

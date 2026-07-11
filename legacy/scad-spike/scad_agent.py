#!/usr/bin/env python3
"""
scad_agent.py — M8 spike: OpenSCAD second-backend CAD agent.

DIRECTION.md Part 2 X1: OpenSCAD is the most abundant code-CAD language in LLM training data
(Thingiverse customizer culture) — the bet is that a small local coder writes it more fluently
than build123d, the mirror image of the v1 lesson ("meet the model where its training data is").
This file is the measured spike that decides it: same brief/verify/critique religion as
`cad_agent_v4.py`, a different codegen target and a MESH-first verification gate (trimesh, not
a B-rep kernel), with best-effort STEP recovery via FreeCAD's OpenSCAD-import bridge so a
build that happens to be pure-CSG-representable ALSO gets a parametric-adjacent B-rep artifact.

Shared plumbing (imported, not duplicated) from `cad_agent_v4.py`:
  build_brief, reconcile_expected, verify_questions — brief generation is backend-agnostic language
  (BRIEF_MODEL/qwen3:8b, JSON-schema constrained) and identical for both backends.
  _ollama, _extract_json — the raw Ollama call + JSON-from-reply helper.
  _CRITIC_QA_SYSTEM, _CRITIC_SYSTEM, _ANSWERS_SCHEMA, CRITIC_MODEL, CRITIC_TIMEOUT — the
  question-based critic call is reused VERBATIM (see `scad_critique()` below for why it's a
  thin local wrapper rather than a call to `visual_critique()` itself: that function insists on
  rendering its own PNG from a STEP via `scripts/render`, which only understands build123d
  STEP/STL, not a pre-made two-panel PNG from `scripts/scad`).
  _new_build_dir — same `~/.openclaw/cad-builds/` tree, same KEEP_BUILDS=200 retention; called
  with a "scad "-prefixed spec so this backend's dirs are visually tagged (`...-scad-<slug>`)
  without needing a signature change to a function we are not allowed to edit.
Everything else here (codegen system prompt, revise/decide prompts, the run/gate/render loop
shape) is NEW: OpenSCAD is a different enough language, and the mesh gate a different enough
verification story, that translating v4's build123d-flavored prompts word-for-word would leave
stale API references (fillet()/Pos()/Box()/Mode.SUBTRACT — none of which exist in OpenSCAD).

NO FEW-SHOTS in this spike, deliberately: the entire hypothesis under test is that OpenSCAD
fluency already lives in the base model's training data and needs no retrieval crutch (contrast
with the build123d path, where `cad_retrieval.py`'s semantic few-shot retrieval measurably lifted
a fail to a converge — see PROJECT.md). `--no-fewshots` is accepted for CLI-contract parity with
`run_benchmarks.py` but is always a no-op here: there is nothing to disable.

NO LLM calls happen at import time or for `--help` — only inside `run_build()`.

CLI contract (matches `scripts/run_benchmarks.py`'s legacy-agent invocation exactly):
    python3 scad_agent.py build "<spec>" --coder fast --no-upload [--json] [--no-fewshots]
Exactly one JSON line is printed on stdout at the very end; all progress/diagnostic output goes
through the shared `cad_v5.config.log` logger, which writes to stderr + `~/.openclaw/cad-agent.log`
(same file v4/v5 use) — never to stdout, so a machine consumer parsing stdout never sees noise.
"""
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
SCRIPTS_DIR = _HERE / "scripts"

from cad_v5.config import (  # noqa: E402
    CODE_MODEL_FAST, CODE_MODEL_STRONG, CODE_TIMEOUT,
    MAX_TURNS, ESCALATE_AFTER, BUILD_TIMEOUT, DONE_SENTINEL, log,
)
from cad_v5.diagnose import diagnose  # noqa: E402  (B3 taxonomy — reused for the run-error histogram)

from cad_agent_v4 import (  # noqa: E402
    build_brief, reconcile_expected, verify_questions, _new_build_dir,
    _ollama, _extract_json, _CRITIC_QA_SYSTEM, _CRITIC_SYSTEM, _ANSWERS_SCHEMA,
    CRITIC_MODEL, CRITIC_TIMEOUT,
)

from scad_mesh_gate import gate as mesh_gate
from scad_step_recovery import recover_step

VERSION = "m8-spike-0.1"
BACKEND = "openscad"

N1_INLINE_RETRIES = 2          # per turn, same coder, raw stderr — never consumes a turn
SCAD_RUN_TIMEOUT = 120         # openscad -o stl
SCAD_RENDER_TIMEOUT = 60       # openscad -o png, per panel


# ── OpenSCAD codegen system prompt (the spike's core) ──────────────────────────

_OPENSCAD_REF = """\
Correct OpenSCAD idioms:
  Units: millimetres. Set $fn = 64; once near the top for smooth circles/cylinders/spheres.
  Primitives: cube([L,W,H], center=true)   cylinder(h=H, r=R, center=true)   sphere(r=R)
  Position: translate([x,y,z]) <shape>     Rotate: rotate([rx,ry,rz]) <shape>   (degrees)
  Combine: union() { a; b; }   difference() { base; cutter1; cutter2; }   intersection() { a; b; }
  A THROUGH-HOLE cutter must clear BOTH faces it pierces: give it height >= material thickness
  plus a few mm of overshoot on EACH side. Two safe patterns:
    translate([x, y, -1]) cylinder(h = thickness + 2, r = r);              // corner-anchored
    translate([x, y, 0])  cylinder(h = thickness + 4, r = r, center = true); // mid-plane centred
  A cutter that stops exactly at a face, or only overshoots one side, leaves a blind hole or a
  complete no-op — the single most common OpenSCAD mistake.
  Avoid minkowski() and hull() unless the part genuinely requires them: both are slow to render
  and easy to misuse; prefer explicit primitives and boolean chains.
  BOSL2 is installed and available via `include <BOSL2/std.scad>` for gears/threads/rounding/
  patterns, but plain primitives + booleans are equally valid for anything expressible that way."""

_CODE_SYSTEM_SCAD = """\
You are an OpenSCAD expert. Write a complete, runnable OpenSCAD script.

RULES:
1. Units are MILLIMETRES. Set `$fn = 64;` once near the top.
2. Centre the part at the origin (use `center=true` on cube()/cylinder(), or translate() an
   off-centre primitive back to centre) so the whole part sits symmetrically around (0,0,0)
   unless the request implies otherwise.
3. Model EXACTLY ONE part, unless the request explicitly describes an assembly of multiple
   distinct, separately-mating components. For an ordinary single-part request, the script's
   FINAL top-level geometry must be ONE object: if you build it from more than one unioned
   piece, wrap the whole thing in a single top-level `union() { ... }` so nothing is left as a
   separate, disconnected body.
4. PARAMETERS: define every key dimension as a named variable near the TOP of the file (e.g.
   `wall = 2; bore_d = 30;`) and use those names in the geometry — never repeat a raw literal
   dimension in two places. This is what makes `-D name=value` overrides work without touching
   the code (the params/regen contract).
5. THROUGH-HOLES: any cutter meant to pass fully through the part must extend past BOTH faces —
   height >= material_thickness + at least 2mm of overshoot on each side. NEVER give a
   through-hole cutter a height that just reaches a face; that produces a blind hole or a
   complete no-op (the #1 mistake — check every cutter's height and position against the
   material it should pierce).
6. Prefer plain difference()/union()/intersection() booleans over minkowski()/hull() — both are
   slow and easy to misuse; reach for them only when the shape genuinely cannot be expressed as
   a boolean of primitives.
7. BOSL2 is installed and importable with `include <BOSL2/std.scad>` for gears, threads, rounded
   primitives, and patterns (bolt circles, linear/radial arrays) when it is a clear fit — plain
   cube()/cylinder()/sphere() + booleans are just as valid for anything a boolean chain expresses.
8. Return ONLY OpenSCAD code — no prose, no explanation. Markdown fences are tolerated (``` or
   ```openscad — they get stripped) but are not required.

""" + _OPENSCAD_REF + """

EXAMPLES (study the style, then write ONE script):

// 60x40x25mm box with a 12mm hole through the top
$fn = 64;
length = 60; width = 40; height = 25; hole_d = 12;
difference() {
    cube([length, width, height], center = true);
    cylinder(h = height + 4, r = hole_d / 2, center = true);
}

// 100x60x20mm enclosure, 2mm walls, open top
$fn = 64;
length = 100; width = 60; height = 20; wall = 2;
difference() {
    cube([length, width, height], center = true);
    translate([0, 0, wall])
        cube([length - 2*wall, width - 2*wall, height], center = true);
}

// 80x50x5mm bracket with two M4 clearance holes (r=2.25), 50mm apart
$fn = 64;
length = 80; width = 50; thickness = 5; hole_r = 2.25;
difference() {
    cube([length, width, thickness], center = true);
    translate([-25, 0, 0]) cylinder(h = thickness + 4, r = hole_r, center = true);
    translate([ 25, 0, 0]) cylinder(h = thickness + 4, r = hole_r, center = true);
}

// 100x60x30mm box with four 4mm holes THROUGH the top face, near the corners (10mm inset)
$fn = 64;
length = 100; width = 60; height = 30; hole_d = 4; inset = 10;
difference() {
    cube([length, width, height], center = true);
    for (x = [-(length/2 - inset), (length/2 - inset)])
        for (y = [-(width/2 - inset), (width/2 - inset)])
            translate([x, y, 0]) cylinder(h = height + 4, r = hole_d / 2, center = true);
}

// Ø80x10mm flange, Ø30 centre bore, 6x Ø8 bolt-circle holes at r=30mm
$fn = 64;
od = 80; thickness = 10; bore_d = 30; bc_r = 30; bolt_d = 8; bolt_n = 6;
difference() {
    cylinder(h = thickness, r = od / 2, center = true);
    cylinder(h = thickness + 4, r = bore_d / 2, center = true);
    for (i = [0 : bolt_n - 1]) {
        a = i * 360 / bolt_n;
        translate([bc_r * cos(a), bc_r * sin(a), 0])
            cylinder(h = thickness + 4, r = bolt_d / 2, center = true);
    }
}"""

_REVISE_SYSTEM_SCAD = """\
You are debugging OpenSCAD code. Fix the problem described. Return ONLY the corrected, complete
OpenSCAD script — no explanation, no markdown fences.

""" + _OPENSCAD_REF

_DECIDE_SYSTEM_SCAD = f"""\
You are reviewing an OpenSCAD part you generated, against the user's request.
You are given the code, a numeric geometry report (from a trimesh MESH measurement — see below
for what it can and cannot tell you), and a visual critique of a render with TWO panels: an
isometric view and a top-down view.

The geometry report's hole/bore count is a topological GENUS sum — a LOWER BOUND on through-hole
count. It can UNDER-count (a blind hole/pocket/counterbore never punctures the surface, so it
adds zero genus) but it never invents a hole that isn't there, so trust it when it confirms a
hole IS present; a low count alone is not proof a hole is missing unless the render agrees.

BE STRICT. Before approving, check EVERY requested feature against the code:
- Is each hole/cutout actually subtracted, and does its cutter's position + height genuinely
  overlap the material along its full intended depth? A cutter that only touches one face (or
  misses the solid) is a silent no-op or a blind hole.
- Do the dimensions in the code match the request?
- Does the reported volume look consistent with the features (holes should reduce it)?

Only if every requested feature is genuinely present and correctly placed, reply with EXACTLY:
{DONE_SENTINEL}
Otherwise return the COMPLETE corrected OpenSCAD script (no explanation, no fences).

{_OPENSCAD_REF}"""


# ── Fence-strip / patch (language-specific — v4's are Python/build123d-specific and would
#    mishandle a ```scad fence or "True"/"False" Python-isms a coder slips into by habit) ──────

def _strip_fences_scad(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:openscad|scad)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


_CODE_PATCHES_SCAD = [
    (r"\bTrue\b", "true"),    # a coder slipping into Python-boolean habit
    (r"\bFalse\b", "false"),
]


def _patch_code_scad(code: str) -> str:
    for pat, repl in _CODE_PATCHES_SCAD:
        code = re.sub(pat, repl, code)
    return code


def generate_code_scad(brief: dict, model: str) -> str:
    dim_str = json.dumps(brief.get("dimensions", {}), indent=2)
    feat_str = "\n".join(f"- {f}" for f in brief.get("features", []))
    notes_str = "\n".join(f"- {n}" for n in brief.get("notes", []))
    prompt = (
        f"Part: {brief.get('description', brief.get('name', ''))}\n"
        f"Dimensions (mm):\n{dim_str}\n"
        + (f"Features:\n{feat_str}\n" if feat_str else "")
        + (f"Notes:\n{notes_str}\n" if notes_str else "")
        + "\nWrite the OpenSCAD code:"
    )
    raw = _ollama(model, _CODE_SYSTEM_SCAD, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    return _patch_code_scad(_strip_fences_scad(raw))


def revise_scad(spec: str, code: str, problem: str, model: str, state: str = "") -> str:
    state_block = f"\nGeometry report:\n{state}\n" if state else ""
    prompt = (
        f"Target part: {spec}\n\n"
        f"Problem to fix:\n{problem}\n"
        f"{state_block}\n"
        f"Current code:\n```openscad\n{code}\n```\n\n"
        f"Return the complete fixed script:"
    )
    raw = _ollama(model, _REVISE_SYSTEM_SCAD, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    return _patch_code_scad(_strip_fences_scad(raw))


def decide_or_edit_scad(spec: str, code: str, state: str,
                        critique: Optional[str], model: str) -> tuple[str, Optional[str]]:
    crit_block = critique if critique else "(visual critique unavailable — judge from code + geometry report)"
    prompt = (
        f"User request: {spec}\n\n"
        f"Current code:\n```openscad\n{code}\n```\n\n"
        f"Geometry report:\n{state}\n\n"
        f"Visual critique:\n{crit_block}\n\n"
        f"Reply {DONE_SENTINEL} if correct, otherwise return the corrected full script:"
    )
    raw = _ollama(model, _DECIDE_SYSTEM_SCAD, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    has_code = "```" in raw or re.search(
        r"^\s*(?:include|use|module|difference|union|intersection|cube|cylinder|sphere)\b",
        raw, re.M)
    if DONE_SENTINEL in raw and not has_code:
        return "done", None
    return "edit", _patch_code_scad(_strip_fences_scad(raw))


# ── Critic (thin local wrapper — see module docstring for why visual_critique() itself can't
#    be called: it insists on rendering its own PNG from a STEP via scripts/render) ────────────

def scad_critique(png_path: Path, spec: str, state: str,
                  questions: Optional[list[str]]) -> Optional[str]:
    """Same model, same question-based prompt structure as cad_agent_v4.visual_critique — the
    prompt CONSTANTS are imported straight from there (not copied) so tuning stays in sync; only
    the "get a PNG" step differs (a pre-made two-panel render from scripts/scad, not a STEP file
    rendered on demand via scripts/render)."""
    try:
        img_b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
        if questions:
            qlist = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
            raw = _ollama(CRITIC_MODEL, _CRITIC_QA_SYSTEM,
                         f"Requested part: {spec}\n\nGeometry facts (authoritative for "
                         f"hidden/through features):\n{state}\n\n"
                         f"Verification questions:\n{qlist}\n\nAnswer each:",
                         timeout=CRITIC_TIMEOUT, images=[img_b64], fmt=_ANSWERS_SCHEMA)
            answers = (_extract_json(raw) or {}).get("answers") or []
            if answers:
                noes = [a for a in answers if str(a.get("answer", "")).lower() == "no"]
                unclear = [a for a in answers if str(a.get("answer", "")).lower() == "unclear"]
                if not noes:
                    note = (" (unverifiable from renders: "
                            + "; ".join(a.get("question", "")[:50] for a in unclear) + ")"
                            if unclear else "")
                    return ("PASS: verification questions confirmed" + note + " — "
                            + "; ".join(a.get("question", "")[:60] for a in answers
                                        if a not in unclear))
                lines = [f"{a.get('question','?')} -> NO"
                        + (f" ({a.get('evidence','')})" if a.get("evidence") else "")
                        for a in noes]
                return ("Verification questions FAILED:\n" + "\n".join(lines)
                        + "\nFix these specific issues.")
        prompt = (
            f"Requested part: {spec}\n\n"
            f"Geometry facts (authoritative for hidden/through features):\n{state}\n\n"
            f"Critique the render:"
        )
        return _ollama(CRITIC_MODEL, _CRITIC_SYSTEM, prompt,
                       timeout=CRITIC_TIMEOUT, images=[img_b64]).strip()
    except Exception as e:
        log.warning("[scad] visual critique unavailable: %s", e)
        return None


# ── Run / render helpers ────────────────────────────────────────────────────────

def _run_scad_stl(code: str, work_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    """Write code to work_dir/part.scad, run scripts/scad -> STL. Returns (stl_path, None) on
    success or (None, error_text) on failure — mirrors cad_agent_v4.run_step's contract."""
    src = work_dir / "part.scad"
    stl = work_dir / "part.stl"
    src.write_text(code)
    try:
        stl.unlink()
    except FileNotFoundError:
        pass
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "scad"), str(src), "-o", str(stl)],
            capture_output=True, text=True, timeout=SCAD_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"openscad timed out after {SCAD_RUN_TIMEOUT}s"
    if proc.returncode != 0 or not stl.exists() or stl.stat().st_size == 0:
        return None, (proc.stderr or proc.stdout or "scripts/scad exited non-zero, no output").strip()
    return stl, None


def _render_two_panel(scad_src: Path, out_dir: Path) -> Optional[Path]:
    """iso + top PNGs via scripts/scad, hstacked with PIL into one render.png (mirrors the shape
    of cad_agent_v4.run_render's two-panel critic image, using this backend's own renderer)."""
    iso = out_dir / "_iso.png"
    top = out_dir / "_top.png"
    try:
        for out, cam in ((iso, "iso"), (top, "top")):
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "scad"), str(scad_src), "-o", str(out),
                 "--camera", cam],
                capture_output=True, text=True, timeout=SCAD_RENDER_TIMEOUT,
            )
            if proc.returncode != 0 or not out.exists():
                log.warning("[scad] render (%s) failed: %s", cam,
                           (proc.stderr or proc.stdout)[-200:])
                return None
        from PIL import Image
        a, b = Image.open(iso), Image.open(top)
        h = max(a.height, b.height)
        combo = Image.new("RGB", (a.width + b.width, h), (255, 255, 255))
        combo.paste(a.convert("RGB"), (0, 0))
        combo.paste(b.convert("RGB"), (a.width, 0))
        render_path = out_dir / "render.png"
        combo.save(render_path)
        return render_path
    except Exception as e:
        log.warning("[scad] two-panel render failed: %s", e)
        return None


def _facts_summary(facts: dict, hard: list[str], soft: list[str]) -> str:
    lines = [
        f"Loadable: {facts.get('loadable')}",
        f"Watertight: {facts.get('watertight')} (repaired={facts.get('repaired')})",
        f"Volume: {facts.get('volume_mm3')} mm^3",
        f"Bbox: {facts.get('bbox_mm')} mm",
        f"Body count: {facts.get('body_count')}",
        f"Genus sum (through-hole LOWER BOUND — cannot see blind holes): {facts.get('genus_sum')}",
    ]
    if hard:
        lines.append("HARD FAILS: " + "; ".join(hard))
    if soft:
        lines.append("ADVISORIES: " + "; ".join(soft))
    return "\n".join(lines)


def _gate_category(msg: str) -> str:
    """Deterministic categorisation of OUR OWN mesh-gate messages (we already know which check
    failed — no need to regex-sniff our own text the way diagnose() sniffs free-form tracebacks)."""
    m = msg.lower()
    if "failed to load" in m:
        return "gate_unloadable"
    if "mesh is empty" in m:
        return "gate_empty"
    if "near-zero" in m:
        return "gate_zero_volume"
    if "watertight" in m:
        return "gate_watertight"
    if "body/bodies" in m:
        return "gate_body_count"
    if "overall size" in m:
        return "gate_bbox"
    return "gate_other"


def _initial_model(coder: str) -> tuple[str, bool]:
    """(model, auto_escalate). This spike's ladder is fast<->strong only (2 rungs, same models
    cad_v5.config already pins for the build123d path) — no 14B, no cloud, no LLM triage call
    (kept simple; the spike is measuring backend choice, not routing sophistication)."""
    if coder == "strong":
        return CODE_MODEL_STRONG, False
    if coder in ("mid", "cloud"):
        log.warning("[scad] --coder %s is not on this backend's ladder (fast/strong only) — "
                    "using fast.", coder)
        return CODE_MODEL_FAST, False
    if coder == "auto":
        return CODE_MODEL_FAST, True
    return CODE_MODEL_FAST, False  # "fast" (default) or anything unrecognised


# ── Main build loop ─────────────────────────────────────────────────────────────

def run_build(spec: str, coder: str = "fast", timeout: Optional[int] = None) -> dict:
    t0 = time.monotonic()
    build_timeout = timeout or BUILD_TIMEOUT
    log.info("=" * 60)
    log.info("[scad] build (%s): %s", VERSION, spec)

    brief = build_brief(spec)
    reconcile_expected(brief, spec)
    questions = verify_questions(spec, brief)
    expected = brief.get("expected") if isinstance(brief.get("expected"), dict) else {}

    active_model, auto_escalate = _initial_model(coder)
    code = generate_code_scad(brief, active_model)

    stl_path: Optional[Path] = None
    gate_passed = False
    done = False
    accepted_via: Optional[str] = None
    fails = 0
    n1_autofixes = 0
    failure_categories: dict = {}
    last_state = ""
    last_critique: Optional[str] = None
    turns_used = 0

    def note(cat: str) -> None:
        failure_categories[cat] = failure_categories.get(cat, 0) + 1

    build_dir: Optional[Path] = None
    stl_out: Optional[Path] = None
    render_out: Optional[Path] = None
    step_local: Optional[str] = None
    step_recovered = False
    step_recovery_error: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="scadagent_") as work_str:
        work_dir = Path(work_str)
        best_stl: Optional[Path] = None
        best_code: Optional[str] = None

        for turn in range(1, MAX_TURNS + 1):
            turns_used = turn
            if time.monotonic() - t0 > build_timeout:
                log.warning("[scad] wall-clock budget hit before turn %d", turn)
                break

            if auto_escalate and fails >= ESCALATE_AFTER and active_model != CODE_MODEL_STRONG:
                log.info("[scad] escalating fast->strong after %d failed turn(s).", fails)
                active_model = CODE_MODEL_STRONG
                code = generate_code_scad(brief, active_model)
                fails = 0

            log.info("[scad] -- turn %d/%d (%s) --", turn, MAX_TURNS, active_model)

            # 1. Run, with up to N1_INLINE_RETRIES silent same-model retries on the raw error.
            #    These retries do NOT consume a turn; only exhausting all of them counts as one
            #    failed turn (mirrors N1's semantics from DIRECTION.md Part 1).
            run_err: Optional[str] = None
            for attempt in range(N1_INLINE_RETRIES + 1):
                stl_path, run_err = _run_scad_stl(code, work_dir)
                if stl_path is not None:
                    break
                if attempt < N1_INLINE_RETRIES:
                    n1_autofixes += 1
                    log.info("[scad] N1 inline autofix %d/%d (same model, raw error).",
                             attempt + 1, N1_INLINE_RETRIES)
                    code = revise_scad(spec, code,
                                       f"The OpenSCAD script failed to run:\n{run_err}",
                                       active_model)
            if stl_path is None:
                cat, hint = diagnose(run_err or "")
                note(cat)
                fails += 1
                log.warning("[scad] run failed after %d inline retries: %s",
                           N1_INLINE_RETRIES, (run_err or "")[:200])
                if turn < MAX_TURNS:
                    extra = f"\n\n{hint}" if hint else ""
                    code = revise_scad(
                        spec, code,
                        f"The OpenSCAD script still fails to run after retries:\n{run_err}{extra}",
                        active_model)
                continue

            # 2. Deterministic mesh gate (authoritative for structural integrity; see
            #    scad_mesh_gate.py's docstring for exactly what it can/cannot verify).
            hard, soft, facts = mesh_gate(stl_path, expected)
            last_state = _facts_summary(facts, hard, soft)
            if hard:
                for h in hard:
                    note(_gate_category(h))
                gate_passed = False
                stl_path = None  # never treat gate-failing geometry as a valid deliverable
                fails += 1
                log.warning("[scad] mesh gate FAILED: %s", "; ".join(hard))
                if turn < MAX_TURNS:
                    code = revise_scad(
                        spec, code,
                        "The build is geometrically WRONG. These deterministic checks failed "
                        "(authoritative — they measure the actual mesh):\n"
                        + "\n".join(f"- {h}" for h in hard)
                        + "\nFix the source so every check passes.",
                        active_model, state=last_state)
                continue

            gate_passed = True
            try:
                snap = work_dir / f"best_turn{turn}.stl"
                shutil.copy(stl_path, snap)
                best_stl, best_code = snap, code
            except Exception as e:
                log.warning("[scad] best-build snapshot failed: %s", e)

            # 3. Render + critique
            render_path = _render_two_panel(work_dir / "part.scad", work_dir)
            critique = scad_critique(render_path, spec, last_state, questions) if render_path else None
            if critique:
                last_critique = critique
                log.info("[scad] critic: %s", critique.replace("\n", " ")[:300])

            # 4. Decide: done, or edit and re-observe
            action, new_code = decide_or_edit_scad(spec, code, last_state, critique, active_model)
            if action == "done":
                log.info("[scad] model satisfied — finalizing.")
                done = True
                accepted_via = "critic"
                break
            if (new_code or "").strip() == code.strip():
                fails += 1
                if gate_passed:
                    log.info("[scad] coder stuck but the mesh gate passed — accepting via gate "
                            "(not escalating a verified-correct build).")
                    done = True
                    accepted_via = "gate"
                    break
                if turn < MAX_TURNS:
                    code = revise_scad(
                        spec, code,
                        f"A reviewer says this is WRONG: {critique or '(no critique available)'}\n"
                        f"Fix it so every requested feature is present and correctly placed.",
                        active_model, state=last_state)
                continue
            log.info("[scad] model chose to keep editing.")
            code = new_code

        # A failed/regressed final edit must never discard an earlier gate-verified mesh.
        if not done and best_stl is not None and (stl_path is None or not gate_passed):
            log.warning("[scad] restoring the last gate-verified build instead of discarding it.")
            stl_path = best_stl
            code = best_code or code
            gate_passed = True
        if not done and gate_passed and stl_path is not None:
            log.info("[scad] turn budget exhausted but the last build passed the mesh gate — "
                     "accepting as converged via gate.")
            done = True
            accepted_via = "gate"

        # Every 'gate' acceptance path above (stuck-but-passed, restored snapshot, or turn-budget
        # exhaustion right after a "keep editing" decision) must export the CODE THAT ACTUALLY
        # PRODUCED stl_path, not whatever `code` holds — a "keep editing" turn reassigns `code`
        # to an untested edit before the loop can re-run it, so on the very last permitted turn
        # that edit is never executed/verified. best_code is snapshotted at the exact moment its
        # matching stl_path passed the gate, so it's always the trustworthy pairing here.
        if accepted_via == "gate" and best_code is not None:
            code = best_code

        ok = stl_path is not None and stl_path.exists()
        if ok:
            build_dir = _new_build_dir("scad " + spec)   # "scad "-prefixed slug, shared retention
            stl_out = build_dir / "build.stl"
            shutil.copy(stl_path, stl_out)
            src_out = build_dir / "build_source.scad"
            try:
                src_out.write_text(code)
            except Exception as e:
                log.warning("[scad] could not save build_source.scad: %s", e)

            render_path = _render_two_panel(src_out, build_dir)
            if render_path:
                render_out = render_path

            step_out_path = build_dir / "build.step"
            try:
                rec = recover_step(src_out, stl_out, step_out_path)
            except Exception as e:
                rec = {"step_recovered": False, "step_local": None,
                      "step_recovery_error": f"recovery pipeline raised: {e}"}
            step_recovered = bool(rec.get("step_recovered"))
            step_recovery_error = rec.get("step_recovery_error")
            if step_recovered:
                step_local = rec.get("step_local")
                log.info("[scad] STEP recovery VERIFIED: %s", step_local)
            else:
                log.info("[scad] STEP recovery not available: %s", step_recovery_error)
        else:
            log.warning("[scad] build failed after %d turn(s) with no gate-passing geometry.",
                       turns_used)

    result = {
        "ok": bool(stl_out and Path(stl_out).exists()),
        "converged": bool(done),
        "accepted_via": accepted_via,
        "code_model": active_model,
        "build_time_s": round(time.monotonic() - t0, 1),
        "turns": turns_used,
        "n1_autofixes": n1_autofixes,
        "stl_local": str(stl_out) if stl_out else "",
        "render_local": str(render_out) if render_out else "",
        "failure_categories": failure_categories,
        "backend": BACKEND,
        "step_recovered": step_recovered,
        "step_recovery_error": step_recovery_error,
        "spec": spec,
        "brief": brief,
        "code": code,
        "build_dir": str(build_dir) if build_dir else "",
        "last_critique": last_critique,
        "built_at": datetime.now().astimezone().isoformat(),
    }
    if step_recovered and step_local:
        result["step_local"] = step_local
    if not result["ok"]:
        result["error"] = f"build failed after {turns_used} turn(s) — no gate-passing mesh produced"
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(
        prog="scad_agent.py",
        description="M8 spike: OpenSCAD second-backend CAD agent (mesh deliverable + "
                    "best-effort STEP recovery via FreeCAD's OpenSCAD-import bridge).")
    sub = p.add_subparsers(dest="command")

    b = sub.add_parser("build", help="build a part from a natural-language spec")
    b.add_argument("spec", help="what to build")
    b.add_argument("--coder", choices=["auto", "fast", "mid", "strong", "cloud"], default="fast",
                  help="fast=qwen2.5-coder:7b, strong=qwen3-coder:30b, auto=fast+escalate; "
                       "mid/cloud are not on this backend's ladder and fall back to fast")
    b.add_argument("--no-upload", action="store_true",
                  help="no-op — this backend never uploads anywhere (kept for CLI-contract "
                       "parity with cad_agent_v4 / scripts/run_benchmarks.py)")
    b.add_argument("--json", action="store_true",
                  help="no-op — stdout is already exactly one JSON result line")
    b.add_argument("--no-fewshots", action="store_true",
                  help="no-op — this spike never injects few-shots (see module docstring)")
    b.add_argument("--timeout", type=int, default=None,
                  help="override the build wall-clock budget in seconds (default: cad_v5.config.BUILD_TIMEOUT)")

    a = p.parse_args(argv)
    if a.command != "build":
        p.print_help()
        return 0 if a.command is None else 1

    result = run_build(a.spec, coder=a.coder, timeout=a.timeout)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

# CAD Agent v5 — User Guide

Turn a plain-English description into a real, validated 3D CAD model, then refine it by
chatting — all locally on your machine.

---

## The one thing to know

**There is one flow: describe → build → refine.** Every build drops you straight into a
conversation where you give feedback in plain English and it rebuilds. There is no separate
"one-shot" mode — if the first result is already right, you just type `done`.

```
$ cad "a 120x80x40mm enclosure with 2mm walls and a 20mm cable hole"

  …building (brief → code → check → view)…
  ✓ Built. Opening in CAD Viewer: http://127.0.0.1:4178/?dir=…&file=cad-last-build.step
    120 × 80 × 40 mm · 1 solid · Ø20 bore · 2.0 mm walls · STEP/STL/DXF saved

  Feedback? (describe a change, or: done / rate 5 / onshape / show)
> make the walls 3mm and add four M3 mounting holes in the floor

  …rebuilding with your feedback…
  ✓ Updated. Refresh the viewer tab.
    Δ vs previous: +4 cylindrical faces, −2,140 mm³  · walls now 3.0 mm

> done
  Saved. STEP: ~/.openclaw/cad-last-build.step  (STL, DXF alongside)
```

That's the whole experience. Keep talking until it's right; type `done` when it is.

---

## Running it

```bash
cad "<what you want>"                 # build, then refine interactively
cad                                   # start with no spec; describe it at the first prompt
cad --image photo.jpg "<spec>"        # build FROM a reference photo/sketch (jpg/png/webp)
```

(Equivalently `python3 ~/.openclaw/skills/cad-builder/cad_v5 "<spec>"`. From Telegram/Satine the
same loop applies: send a description, then reply with changes.)

### Building from a photo (`--image`)

Give the agent a reference photo or sketch and it uses the image two ways: a local vision
model (gemma4:e4b) first describes the part's form, features, and proportions into the build
brief, and then judges every draft render against your photo until they match.

**The rule: shape and proportions come from the image; absolute sizes come from your text.**
A photo has no scale, so the agent never invents millimetres from it — say the envelope
("..., 80mm wide") in the spec, or have real dimension annotations legible in the image.
From Telegram, caption a photo with your spec (or send the photo bare, then describe the part
within 30 minutes). The analysis is cached per photo, and refine turns keep the reference.

### Things you can type at the feedback prompt

| You type | What happens |
|---|---|
| *any description of a change* | merges your feedback into the spec and rebuilds |
| `done` (or `q`, Enter on empty) | stops and keeps the current model |
| `rate 5` *(1–5) [note]* | rates the build; 4–5★ are stored as good examples it learns from |
| `show` | re-opens / prints the current CAD Viewer link |
| `onshape` | also uploads the current model to Onshape (returns an editable Part Studio link) |
| `undo` | reverts to the previous build |

---

## Writing good prompts

The more of these four things a spec states, the better the first draft — and the less often
the agent has to stop and ask (see "If your spec is missing something" below).

| The 4 S's | What it means | Example |
|---|---|---|
| **Size** | overall envelope in mm | "120x80x40mm" |
| **Specs** | counts + diameters | "4x M3 through-holes", "a 20mm bore" |
| **Surfaces** | which faces features live on | "holes in the floor", "a groove on the outside" |
| **Symmetry** | patterns/spacing | "6 holes equally spaced on a 60mm bolt circle" |

**Clearances** — say which fit you want and the agent applies the right offset:

| Fit | Clearance |
|---|---|
| Push-fit | 0.0–0.1mm |
| Slip-fit | 0.2mm |
| Loose-fit | 0.5–1.0mm |

**Named hardware is understood** — "M3", "608ZZ bearing", "2020 V-slot extrusion" and similar
standard designations resolve to their real dimensions without you spelling them out.

Example prompts (copy/adapt these):
```
cad "a 120x80x40mm enclosure, 2mm walls, 4x M3 mounting holes in the floor"
cad "a flange: 80mm OD, 10mm thick, 30mm through-bore, 6x M6 bolt holes on a 60mm bolt circle"
cad "a shaft 12mm dia x 100mm long with a 4mm cross-hole 20mm from one end"
```

**Organic and pattern parts** — the agent isn't limited to boxy engineering shapes: it can also
build smoothly-varying and decorative forms — vases whose radius swells and shrinks along the
height, twisted prisms, wavy/scalloped edges, and repeating perforation patterns (honeycomb
grilles, diamond-pierced shells). The same rules apply: give every dimension explicitly, and for
waves/twists/patterns say *how many* and *how big* — e.g.
`cad "a vase 120mm tall, round, radius swelling from 30mm at the base to 40mm at mid-height and back to 30mm at the top, 2.5mm walls, closed floor, open top"`,
`cad "a hexagonal vase 60mm across flats, 150mm tall, twisted 90 degrees over its height, 2.5mm walls"`, or
`cad "a 100x60x3mm grille with hexagonal holes 7mm across flats, 2mm webs, inside a solid 6mm border"`.

### If your spec is missing something

Before the first build, the agent runs a quick check: does this carry enough (an overall size,
and the part's basic shape) for a competent first draft? If a reasonable default covers what's
missing (e.g. wall thickness defaults to ~2mm), it just builds — no interruption. Only when a
dimension or the basic form is genuinely unknowable does it stop and ask 2–3 short questions,
each with a suggested default; press Enter to accept the defaults and build anyway.

---

## Editing dimensions without the AI (`params` / `regen`)

Every build saves its recipe with the key dimensions as named values at the top. You can change
a number and rebuild in seconds — no model, no waiting, and the result is re-verified:

```bash
cad params            # list the editable dimensions of the last build
cad regen wall=3      # rebuild with a new value (several at once works too)
```

`regen` writes a fresh build folder and updates `cad-last-build.*`; if the new values produce
broken geometry it says so and leaves your previous files untouched. Use `--source <folder>` to
regenerate an older build from `~/.openclaw/cad-builds/`.

---

## Where your model goes (output targets)

By default everything is **local** — no cloud round-trip:

- **CAD Viewer** (default): a browser link opens the model; rotate, section, measure, click faces.
- Every build also writes, in `~/.openclaw/`:
  - `cad-last-build.step` — the master CAD file (import into any CAD tool)
  - `cad-last-build.stl` — sliceable mesh for 3D printing
  - `cad-last-build.dxf` — flat pattern for laser / sheet metal (when the part has a flat face)

**Onshape is opt-in.** Type `onshape` at the prompt (or run with `--onshape`) to also get a cloud
Part Studio. **FreeCAD is opt-in too** — `--target freecad` writes a parametric `.FCStd` (a Params
spreadsheet you can edit in the FreeCAD GUI to regenerate; box-grammar parts only, else a plain STEP
import). Change the default target permanently in `~/.openclaw/cad.json` (NOT `openclaw.json` — the
gateway's strict schema rejects a top-level `cad` key):

```json
{ "output_target": "cad-viewer" }   // or "onshape", "freecad", "print", "fstl", "file"
```

---

## From model to machine (CAM)

Two production processes are wired end-to-end — the output isn't just geometry-true, it's
machine-ready and **checked**.

### 3D printing — `--target print`

```bash
cad --target print "a 100x70x30mm enclosure with 2mm walls"
```

Slices the build's STL with OrcaSlicer (headless, Bambu Lab A1 profile by default) and
**dry-run-validates** the gcode before you ever send it to a printer:

- the slicer's own verdict (`return_code`, per-plate warning messages — the "empty layer /
  unprintable geometry" signal),
- the gcode header (layer count > 0, a time estimate, max Z height),
- a motion scan of the actual gcode: every **extruding** move must land on the machine's bed
  (envelope read from the machine profile; travel/wipe moves may legitimately leave it),
  printed height within the machine limit, and total extrusion > 0.

The result reports `print_ok` with `print_fails` (problems), `print_notes` (soft caveats) and
`print_facts` (layers, estimated time, filament use, printed-footprint bbox). Gcode lands in a
`print/` folder next to the build's STEP. First use extracts the Bambu system profiles from the
OrcaSlicer AppImage into `~/.openclaw/cam-profiles/` automatically (one-time, a few seconds).

Change the machine/process/filament with a `print` block in `~/.openclaw/cad.json` (NOT
openclaw.json — same rule as always). Bare names resolve inside the extracted profile tree's
`machine/` / `process/` / `filament/` subdirs; absolute paths pass through:

```json
{ "print": { "machine":  "Bambu Lab A1 0.4 nozzle.json",
             "process":  "0.20mm Standard @BBL A1.json",
             "filament": "Bambu PLA Basic @BBL A1.json" } }
```

Prerequisites: an OrcaSlicer AppImage under `~/Applications/` (or `$ORCA_SLICER` / `orca-slicer`
on PATH) and `xvfb-run` (package `xvfb`) — the slicer CLI needs a virtual display headlessly.

### Laser cutting — kerf-compensated DXF

```bash
python3 scripts/dxf ~/.openclaw/cad-last-build.step --kerf 0.3
python3 scripts/dxf ~/.openclaw/cad-last-build.step --material acrylic-3mm
```

A laser removes a kerf-wide strip **centred on the cut path**, so an uncompensated DXF cuts
parts that measure small and holes that measure big. With `--kerf` (or a `--material` preset)
the geometry is pre-offset so the finished part measures right:

| Feature | Drawn | Why |
|---|---|---|
| Outer boundary | **grows** by kerf/2 | the beam eats kerf/2 back inward → lands on the true edge |
| Interior holes | **shrink** by kerf/2 | the beam eats kerf/2 outward → lands on the true hole edge |

Verified deterministically: an 80×50mm plate with two Ø5mm holes at `--kerf 0.3` produces a DXF
measuring 80.30×50.30mm with Ø4.70mm holes. Compensated output uses `cut-outer` / `cut-holes`
layers plus a `kerf-note` text annotation recording the applied kerf; without a kerf flag the
output is unchanged from before (single `profile` layer, true dimensions).

Material presets (typical values — tune per machine): `acrylic-3mm` 0.20, `acrylic-6mm` 0.30,
`plywood-3mm` 0.30, `plywood-6mm` 0.40, `mdf-3mm` 0.30, `steel-1mm-fiber` 0.15.
`--kerf` wins if both flags are given.

### CNC milling — 2.5D toolpaths (`scripts/cnc`)

```bash
python3 scripts/cnc ~/.openclaw/cad-last-build.step --tool 6 --drill auto
python3 scripts/cnc ~/.openclaw/cad-last-build.step -o /tmp/mypart-cnc --margin-xy 3 --safe-z 8
```

Drives FreeCAD's CAM/Path workbench headlessly (no GUI, no LLM) to generate a Drilling operation
against the part's own through-holes, plus a Pocket operation if the part has a closed cavity
(a plate with only holes has none — that's fine, drilling alone is reported honestly). Both
operations are auto-detected from MEASURED geometry: holes via FreeCAD's own cylindrical-face
hole finder, the pocket floor via a planar-face classifier (a face inset from every outer edge,
sitting strictly between the stock's top and bottom). Posts to gcode with the LinuxCNC
post-processor (native G81/G82/G83 canned drill cycles).

The gcode is then **re-verified in pure Python** (no FreeCAD) against the stock envelope and the
part's own measured hole centres:

- every cutting move stays within the stock's XY footprint (+ tool radius) and above its bottom,
- at least one real cutting move exists (not just rapids),
- no rapid move plunges below the stock's top surface while over its footprint (basic crash check),
- every detected hole gets drilled, no extras, each within 0.1mm of its measured centre.

**This is a VERIFIED SIMULATED toolpath — simulated, not cut.** Nothing here machines real
material; `scripts/cnc` prints `VERIFICATION: PASS`/`FAIL` and exits nonzero on any FAIL.
Output lands in `<step_dir>/cnc/` (`toolpath.ngc`, `job_facts.json`, a FreeCAD build log).
See `cad_v5/cam_cnc.py`'s module docstring for the full empirical API writeup — including a
documented FreeCAD 1.1.1 headless-only bug (crashes when a Job has >1 tool controller and no GUI
is present) and its workaround.

---

## Speed & quality controls

- **The default is auto-routing** (`--coder auto`): it starts on the fast `qwen3:8b`
  (~seconds/turn, GPU) and only escalates to the strong `qwen3-coder:30b` (minutes/turn, CPU
  offload) after repeated failures — or immediately when the triage step judges the spec hard. No
  coder is pinned right now, so the ladder is live.
- Force it per build: **`--coder fast`** (the 8B, quick) or **`--coder strong`** (the 30B, for
  hard parts). There is no mid rung — the 14B was measured out and removed (2026-07-16).
- Pin a coder permanently with `cad.code_model` in `~/.openclaw/cad.json` (disables auto-escalation).
- From Telegram (Satine), prefix the spec with `fast:` or `strong:`.

---

## How it avoids bad CAD (the short version)

It does **not** drive Onshape/Fusion over an API and trust the result. It generates locally with
build123d (real OCCT geometry kernel → exact STEP solids), then **independently checks the actual
solid** before showing it to you:

1. **Brief** — your words → a structured spec with target dimensions.
2. **Code** — a coder model writes build123d Python (the editable source of truth).
3. **Run** — the kernel builds a real solid and exports STEP.
4. **Verify against intent (deterministic)** — the brief lists the part's features as structured
   checks (e.g. *a radial Ø20 bore*, *three ring grooves*), and the system confirms the actual solid
   exhibits each one. It measures solids/holes, bore diameters, wall thickness, and **bore
   orientation** — so it catches a hole that silently didn't cut, an unfused body, a wrong size, or a
   bore that went through the *top* when the spec said through the *side*. A mismatch **forces another
   edit** — the model can't declare itself done over a feature it got wrong.
5. **See (visual)** — renders isometric + top-down + (for hollow parts) a **cut-through section**,
   and a vision model critiques shape/features.
6. **Edit & re-check** — fixes the smallest thing, regenerates, and **diffs** the change to confirm
   it did what was intended — looping until correct.

Your feedback re-enters at step 1, so refining is the same trustworthy loop, not a fragile patch.

**Why this works where "just ask a bigger model" doesn't:** local models can't reliably reason in
3D — they write plausible code that's geometrically wrong (a "groove" cut full-depth into a bore, a
side hole left pointing up). So the system doesn't *trust* the model's spatial reasoning. It gives
the model a library of **correct-by-construction feature helpers** (`cross_bore` for a side hole,
`ring_groove` for a surface groove, `bolt_circle` for a hole pattern…) and **verifies the result
against the brief's stated intent**, generically. The model's job shrinks to *recognise the feature
and call the helper* — which it can do — instead of *computing 3D transforms* — which it can't.

### What it can build

Single parts (enclosures, brackets, plates, shafts, housings, pistons…), standard catalog parts
(I/H/C-beams, gears, bolts, nuts, bearings, flanges — via verified parametric libraries), and
**assemblies** (multiple distinct parts positioned to mate, exported as a multi-part model).

---

## Files & recovery

- Models: `~/.openclaw/cad-last-build.{step,stl,dxf}` (overwritten each build).
- Logs: `~/.openclaw/cad-agent.log`.
- Ratings/examples it learns from: `~/.openclaw/cad-examples.jsonl`,
  pitfalls: `~/.openclaw/cad-lessons.jsonl`.
- Rollback: the previous generation, `cad_engine.py` (v4.3), still runs unchanged if ever needed.

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama not reachable" | `ollama serve` isn't running, or models aren't pulled (`ollama pull qwen3:8b`). |
| Viewer link won't open | The local viewer needs the text-to-cad plugin installed; or set `CAD_VIEWER_BACKEND`. Falls back to writing files. |
| A feature looks wrong | Just say so at the prompt ("the hole should go through the side, not the floor") — it rebuilds. |
| Build is slow | It escalated to the 30B coder. Force `--coder fast` for speed, `strong` only for hard parts. |

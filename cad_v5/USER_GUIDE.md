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
```

(Equivalently `python3 ~/.openclaw/skills/cad-builder/cad_v5 "<spec>"`. From Telegram/Satine the
same loop applies: send a description, then reply with changes.)

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
Part Studio. Change the default permanently in `~/.openclaw/openclaw.json`:

```json
{ "cad": { "output_target": "cad-viewer" } }   // or "onshape", "fstl", "file"
```

---

## Speed & quality controls

- **The default coder is now `qwen3-coder:30b`** (pinned in `~/.openclaw/openclaw.json` →
  `cad.code_model`). It's the strongest local coder; the trade-off is **minutes per turn** (it
  offloads to CPU on the 8GB box) even for simple parts. To go fast on easy parts, pass
  **`--coder fast`** (the 7B, ~seconds/turn). Remove the `cad.code_model` pin to restore the old
  auto-escalation behaviour.
- From Telegram, prefix the spec with `fast:` or `strong:`.

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
- Rollback: the previous generation, `cad_agent_v4.py` (v4.3), still runs unchanged if ever needed.

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama not reachable" | `ollama serve` isn't running, or models aren't pulled (`ollama pull qwen3:8b`). |
| Viewer link won't open | The local viewer needs the text-to-cad plugin installed; or set `CAD_VIEWER_BACKEND`. Falls back to writing files. |
| A feature looks wrong | Just say so at the prompt ("the hole should go through the side, not the floor") — it rebuilds. |
| Build is slow | It escalated to the 30B coder. Force `--coder fast` for speed, `strong` only for hard parts. |

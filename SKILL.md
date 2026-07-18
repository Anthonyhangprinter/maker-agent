---
name: cad-builder
description: "Build CAD models from natural-language specs. build123d agentic observe-edit loop (v5: cad_v5 package + cad_engine.py core): brief (qwen3:8b) → auto-routed 2-rung coder ladder (qwen3:8b → qwen3-coder:30b; 14b manual-only) → run/inspect/render → gemma4:e4b visual critic → edit, until it matches. Outputs: local CAD Viewer (default via `cad`), Onshape, STEP/STL/DXF/FCStd. Handles enclosures/boxes, brackets, plates, holes/pockets, fillets, structural sections, gears, bolts. Use when: user asks to build/modify a part, or check a CAD model. All inference local via Ollama — no Claude API calls."
metadata:
  {
    "openclaw":
      {
        "emoji": "🔩",
        "requires": { "bins": ["python3"] },
      },
  }
---

# CAD Builder Skill (v5: cad_v5 package + cad_engine.py core)

Build and iterate on CAD models from text specs. All inference runs locally via Ollama
(qwen3:8b for brief+code, qwen3-coder:30b escalation, 14b manual-only, gemma4:e4b critic).
**No Claude API calls.** Default output target is the local CAD Viewer (via the `cad` launcher);
Onshape is opt-in.

The agent writes **build123d** Python (algebra mode), runs it → STEP, inspects the geometry,
renders a 2-panel PNG (isometric + top-down), has a multimodal critic judge it against the spec,
then edits and re-observes until it converges (`###DONE###`) or hits the turn/time budget. It then
translates the STEP into a real Onshape **Part Studio** (viewable/editable, not a blob).

## Layout — how this directory is filed

```
cad-builder/
├── cad               # launcher (`cad "<spec>"` — symlinked into ~/.local/bin, works anywhere)
├── cad_engine.py     # THE core engine: brief → codegen → gate → critic loop (one big module)
├── cad_retrieval.py  # few-shot / lesson semantic retrieval (nomic-embed)
├── cad_v5/           # the v5 package: cli, loop (refine/undo), engine wrapper, targets,
│                     #   config.py (SINGLE SOURCE OF TRUTH: models, timeouts, VERSION),
│                     #   cam_print.py / cam_cnc.py, freecad_export.py, USER_GUIDE.md, EXPLAINER.md
├── b123d/            # deterministic domain helpers: structural sections, gears (bd_warehouse), gussets
├── scripts/          # runners the loop shells out to: step, inspect, render, dxf, cnc, stl,
│                     #   run_benchmarks.py
├── benchmarks/       # suites (text-to-cad/, organic/) + results/ (json + artifacts per run)
├── tests/            # offline unit/behavioral tests (pytest)
├── integration/      # cad-telegram.py — Satine bot SOURCE (deployed: ~/.openclaw/cad-telegram.py
│                     #   is a symlink to it; restart cad-telegram.service after editing)
├── docs/             # PROJECT.md (design+results) · ROADMAP.md · DIRECTION.md · BUILDS.md
│                     #   (build journal) · HANDOFF.md
├── legacy/           # v1-v3 agents, scad-spike/ (M8, rejected), bak-20260627/ — rollback only
└── SKILL.md          # this file — usage (stays at root: skill convention)
```

Runtime state lives in `~/.openclaw/` (cad.json, cad-examples.jsonl, cad-lessons.jsonl,
cad-session.json, cad-builds/, cad-agent.log) — see "Scripts" below.

## When to Use

✅ **USE when:**
- "make / build / create a 100x60x30 enclosure with 2mm walls" — new build
- "make it 200mm longer", "add a lid", "3mm walls" — refine the last build
- "what did we just build?", "show session" — recall last build state
- "rate the last build 5" — save a good build as a future few-shot example

❌ **DON'T use when:**
- User just wants the Onshape URL opened in a browser (give them the URL)
- Deleting an Onshape document (Onshape UI only)

## Scripts

```
cad "<spec>"                                              # v5 interactive entry (local CAD Viewer)
cad --image photo.jpg "<spec>"                            # image-conditioned build (reference photo)
SCRIPT=~/.openclaw/skills/cad-builder/cad_engine.py       # core ENGINE (also a standalone CLI)
# Legacy / rollback only (moved to legacy/): cad_agent_v3.py, cad_agent_v2.py, onshape_cad_agent.py
```

**Reference images (`--image`, 2026-07-17; image-only mode 2026-07-18):** a gemma4:e4b vision
pre-pass describes the photo (shape family, features, proportions, `suggested_spec`) into an
`IMAGE ANALYSIS` addendum the brief reads, and the critic receives the photo as a SECOND image
each turn to judge form/features/proportions against (two-image attention verified 2026-07-17).
**Image-only ('fluid') builds:** an image with NO spec is a full request — `spec_from_image()`
turns the analysis into the spec, the brief CHOOSES and states sensible proportion-consistent
sizes (in-image annotations honoured), the ambiguity gate is skipped, and the result carries
`image_only: true` + the derived `spec` (frontends echo "I read the image as…"). With text
present, user mm still always win over the photo. Analysis cached at `<photo>.analysis.json`
(old caches without `suggested_spec` auto-refresh); downscaled reference saved per-build as
`reference.jpg`; result JSON carries `image` + `image_analysis`. Satine: bare photo = immediate
image-only build (mid-session: refines toward the new photo); captioned photo = spec+reference.

**Build lock (2026-07-17):** `engine.build()` holds an exclusive flock on
`~/.openclaw/cad-build.lock` — builds from ALL frontends (CLI/Satine/web/benchmarks) serialize
machine-wide (one model fits the GPU). The wall-clock budget starts after acquisition; frontends
surface the `waiting for build lock` stderr line as a queue state. Holder info (pid/frontend/spec)
is written into the lock file; `CAD_FRONTEND` env names the frontend.

Config & creds: `~/.openclaw/openclaw.json` (`env` ONSHAPE keys + telegram token) and
**`~/.openclaw/cad.json`** (`cad.*` agent settings incl. the `code_model` pin — kept out of
openclaw.json because the gateway's strict schema rejects unknown root keys).
Log: `~/.openclaw/cad-agent.log`. Corpus: `~/.openclaw/cad-examples.jsonl` (gold + rated few-shots);
lessons: `~/.openclaw/cad-lessons.jsonl` (semantic-deduped).
Session: `~/.openclaw/cad-session.json`. Per-build artifacts: `~/.openclaw/cad-builds/<ts>-<slug>/`
(last `KEEP_BUILDS` = 200 kept); `~/.openclaw/cad-last-build.{step,stl,dxf}` are convenience copies of the latest.

## Commands

```bash
python3 $SCRIPT build "a 100x60x30mm enclosure with 2mm walls"        # → Onshape URL
python3 $SCRIPT build "W200x100 I-beam 1500mm"                        # uses structural_section()
python3 $SCRIPT build "<spec>" --coder strong                        # force the 30B coder
python3 $SCRIPT build "<spec>" --coder fast                          # force the fast rung (qwen3:8b)
python3 $SCRIPT session                                              # last build state (JSON)
python3 $SCRIPT rate <1-5> [comment]                                 # store ≥4★ as a few-shot
python3 $SCRIPT brief "<spec>"                                       # debug: structured brief only
python3 $SCRIPT code  "<spec>"                                       # debug: generated code, no build
python3 $SCRIPT refine "<orig>" "<feedback>" [history_json]          # Satine spec-merge helper
python3 $SCRIPT inspect "<onshape_url>"                              # describe an existing doc
```

## The Loop (one build)

1. **Brief** (`qwen3:8b`, temp 0.2) — NL spec → structured JSON `{name, dimensions, features,
   notes, helper}`. Interprets intent: box/case/enclosure/container/tray/bin ⇒ **hollow, open-top,
   ~2mm walls** unless "solid"/"block"/"plate" etc. is said.
2. **Coder triage** (`qwen3:8b`) — decides if the spec is hard enough to SKIP the fast rung and start
   directly on the strong 30B rung.
3. **Codegen** (auto-routed coder, temp 0.15) — build123d algebra-mode script.
4. **Run → STEP** (`scripts/step`), **inspect** (`scripts/inspect`), **render** (`scripts/render`,
   2 panels).
5. **Visual critic** (`gemma4:e4b`) — describes BOTH panels; the top-down panel is where top-face
   holes/pockets show.
6. **Decide / edit** — `###DONE###` if every requested feature is genuinely present (verified
   against code + geometry, not just the render), else edit and loop. Max `MAX_TURNS` (4) /
   `BUILD_TIMEOUT` (1800s).

**Fast paths:** structural sections / gears / bolts come from `b123d/domain.py` helpers, are
correct by construction, and **bypass codegen and the critic** (~60s).

## Coder routing

Escalation ladder is **2 rungs: qwen3:8b → 30B** (`CODE_MODEL_LADDER = [CODE_MODEL_FAST,
CODE_MODEL_STRONG]`, weakest first, one rung per escalation). Default fast rung is **`qwen3:8b`**
(2026-07-12 A/B: 15/22 acceptance vs the code-tuned 7B's 8/22 under identical prompts — it also
keeps one model warm across brief/triage/codegen/decide); triage makes hard specs SKIP it and
start on the strong `qwen3-coder:30b` (~7min/call, CPU offload) — the last resort, reached by
triage or escalation. There is NO mid rung: the 14B was measured out twice and the model + the
`--coder mid` alias were REMOVED 2026-07-16 (as were qwen2.5-coder:7b, qwen3.5:4b/9b, phi3).
- Force per build: `--coder auto|fast|strong`.
- Telegram (Satine): prefix the message `fast: <spec>` or `strong: <spec>`.
- Pin permanently: set `cad.code_model` in `~/.openclaw/cad.json` (disables auto-climbing).
- The chosen model is recorded as `code_model` in the session/result.

## Telegram (Satine — `~/.openclaw/cad-telegram.py`, `cad-telegram.service`)

- Send a spec as plain text (or `/build <spec>`) to build; reply with changes to refine.
- `fast:`/`strong:` prefix forces a coder. `/rate 1-5`, `/plan`, `/done`, `/help`.
- Satine shells out to `cad_engine.py` per request, so agent edits are live without a restart;
  **restart `cad-telegram.service` only after editing `cad-telegram.py` itself.**

## Honesty & verification

The loop NEVER reports `converged: true` unless the model confirmed the part. A non-converged
build still uploads the last valid geometry but is flagged with `converged: false` + a `warning` +
`last_critique`. Verify geometry against intent (volume / face counts / render) — don't assume a
feature exists just because the code mentions it.

## Learning loop

Small-model quality comes from retrieval + memory, not parameters:
- **Few-shot corpus** `~/.openclaw/cad-examples.jsonl` — `gold` (verified) + `rated` builds. Semantic
  retrieval (`cad_retrieval.py`, `nomic-embed-text`) injects the closest known-good build123d code
  into the brief. `rate ≥4★` adds the current build. Measured: it flipped a flange variant the 7B
  couldn't build (`--no-fewshots`) into a clean converged build.
- **Fail→fix memory** `~/.openclaw/cad-lessons.jsonl` — a recovered build auto-distills one concrete
  pitfall (qwen3:8b), retrieved + injected as "PITFALLS to avoid" on similar future specs.
- **Measure the lift:** `build --no-fewshots` (or the runner's `--no-fewshots`) disables retrieval so
  any claimed improvement is A/B-checkable. If a piece shows no benchmark lift, remove it.

```bash
python3 cad_retrieval.py "a flange with a bolt circle"      # inspect what retrieval returns
```

## CAM — from model to machine (M9, M11)

```bash
cad --target print "<spec>"                       # slice STL → dry-run-validated gcode (OrcaSlicer,
                                                  # Bambu A1 profiles; result: print_ok/print_fails/
                                                  # print_facts; gcode in <build>/print/)
python3 scripts/dxf <part.step> --kerf 0.3        # laser-ready DXF: outer +kerf/2, holes −kerf/2
python3 scripts/dxf <part.step> --material acrylic-3mm   # kerf preset (see scripts/dxf header)
python3 scripts/cnc <part.step> --tool 6 --drill auto    # CNC 2.5D: Pocket+Drilling toolpath,
                                                          # verified vs stock envelope + hole centres
```

Print target config: `print` block in `~/.openclaw/cad.json` ({machine, process, filament} — bare
names resolve in `~/.openclaw/cam-profiles/BBL/`, auto-extracted from the OrcaSlicer AppImage on
first use). Needs `xvfb-run` + an OrcaSlicer AppImage in `~/Applications/` (or `$ORCA_SLICER`).
Validation is deterministic (no LLM): slicer return_code/warnings, gcode header (layers/time),
and a motion scan — extruding moves must stay on the bed, positive extrusion required. Kerf
semantics: the beam removes kerf-width centred on the path, so the drawn outer boundary GROWS by
kerf/2 and holes SHRINK by kerf/2 — the cut part then measures true (verified: 80×50 plate +
2×Ø5 holes at kerf 0.3 → DXF 80.30×50.30, holes Ø4.70). Tests: `tests/test_m9_cam.py`.

CNC (`cad_v5/cam_cnc.py`) drives FreeCAD's CAM/Path workbench headlessly (no GUI, no LLM):
auto-detects through-holes (Drilling) and a closed-pocket floor if the part has one (Pocket —
proven headless-viable empirically, contra the milestone's worry it might be GUI-bound), posts to
gcode via the LinuxCNC post, then re-verifies the gcode in pure Python against the stock envelope
and the part's measured hole centres (0.1mm position tolerance). Output is a VERIFIED SIMULATED
toolpath — simulated, not cut. Full empirical API writeup (module names, a documented FreeCAD
1.1.1 headless-only multi-tool-controller bug + its workaround) lives in `cam_cnc.py`'s docstring.
Tests: `tests/test_m11_cnc.py`.

## Writing good prompts

State the **4 S's** — Size (overall envelope, mm), Specs (counts + diameters, "4x M3"),
Surfaces (which face a feature is on), Symmetry (patterns/spacing) — plus a clearance word
(push-fit 0.0-0.1mm / slip-fit 0.2mm / loose-fit 0.5-1.0mm) when it matters. Named hardware (M3,
608ZZ bearing, 2020 V-slot) is understood as-is. Full rubric + examples: `cad --help` or
`cad_v5/USER_GUIDE.md`. A spec missing a critical dimension or basic form gets asked about
(2-3 short questions with defaults) instead of silently guessed.

## Tuning knobs (see docs/PROJECT.md for detail)

- **`~/.openclaw/cad.json`**: `code_model` (pin coder), `public_uploads`. Creds: openclaw.json `env.ONSHAPE_*`.
- **Constants** (`cad_v5/config.py` — single source of truth): models, `MAX_TURNS`, `ESCALATE_AFTER`, `*_TIMEOUT`.
- **Prompts** (highest leverage): `_BRIEF_SYSTEM` (intent), `_CODE_SYSTEM` (coordinate rules +
  examples), `_COMPLEXITY_SYSTEM` (triage), `_CRITIC_SYSTEM`/`_DECIDE_SYSTEM`. Sampling temps are
  inline (codegen/revise/decide 0.15, brief 0.2, triage 0.0).

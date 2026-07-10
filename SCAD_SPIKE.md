# M8 — OpenSCAD second-backend spike

*2026-07-09. Implements DIRECTION.md Part 2 item X1. This is a SPIKE: it exists to produce a
measured verdict, not to ship a finished feature. Nothing here is wired into the live `cad`
CLI, Satine, or the benchmark's default agent — it is exercised only via `--agent scad_agent.py`.*

## Hypothesis

OpenSCAD is (by a wide margin) the most abundant code-CAD language in LLM pretraining data —
Thingiverse's "Customizer" culture has produced more public OpenSCAD source than build123d has
ever had users. The bet: a small local coder (`qwen2.5-coder:7b`) writes fluent, correct OpenSCAD
more reliably than it writes build123d, simply because it has seen vastly more of the former
during training. printpal.io runs an entire commercial product on OpenSCAD-WASM + BOSL2, which is
existence proof the approach scales *somewhere* — the open question is whether it helps *this*
project's specific model/hardware combination.

This runs directly counter to this project's own v1 lesson: v1 used Onshape FeatureScript and
was retired specifically because FeatureScript is a rare, poorly-represented language the coder
constantly hallucinated APIs for — the lesson written down at the time was "meet the model where
its training data is," which is exactly why the project moved to build123d (a thin, popular,
pure-Python API) in the first place. X1 asks a legitimate question: does that lesson generalize
*past* build123d to an even-more-abundant target, or was build123d already close enough to the
ceiling that OpenSCAD's abundance premium doesn't move the needle on a 7B model? Only a same-spec,
same-coder, same-verification-discipline benchmark run answers that — hence the spike, gated
exactly like the 14B coder was (measured OUT of the ladder on real numbers, not vibes).

## Architecture — shared vs. new

**Shared with `cad_agent_v4.py` (imported, not duplicated):**
- `build_brief()` + `reconcile_expected()` — brief generation is backend-agnostic: `qwen3:8b`,
  schema-constrained JSON, same `expected` block shape (`bbox_mm`, `solids`, `min_holes`, …).
- `verify_questions()` — the CADCodeVerify-style binary-question generation, also backend-agnostic.
- The critic MODEL and its two prompt constants (`_CRITIC_QA_SYSTEM`, `_CRITIC_SYSTEM`,
  `_ANSWERS_SCHEMA`, `gemma4:e4b`) — reused verbatim via a thin local wrapper (`scad_critique()`
  in `scad_agent.py`) because `visual_critique()` itself insists on rendering its own PNG from a
  STEP via `scripts/render`, and this backend's renders come from `scripts/scad` instead.
- `_new_build_dir()` — the same `~/.openclaw/cad-builds/` tree and `KEEP_BUILDS=200` retention,
  called with a `"scad "`-prefixed spec so directories read `<ts>-scad-<slug>` without needing a
  signature change to a function this spike is not allowed to edit.
- `cad_v5/config.py` constants — `MAX_TURNS`, `ESCALATE_AFTER`, `BUILD_TIMEOUT`, `DONE_SENTINEL`,
  `CODE_MODEL_FAST`/`CODE_MODEL_STRONG`, the shared `log`.
- `cad_v5/diagnose.py`'s taxonomy — used for the run-error histogram; most of its regexes are
  Python/build123d-flavored and will often fall through to `"unknown"` for genuine OpenSCAD
  parser errors. That's an honest gap, not silently patched over (see Deviations below).

**New for this spike:**
- `scripts/scad` — headless OpenSCAD runner: locates the binary (`$OPENSCAD_BIN` → `openscad` on
  PATH → newest `~/Applications/OpenSCAD-*.AppImage`), exports STL/CSG directly, and renders PNGs
  with two EMPIRICALLY VERIFIED camera presets (see below — the docs' phrasing did not obviously
  predict which rotation is "top").
- `scad_mesh_gate.py` — pure-`trimesh` deterministic gate (no LLM), the mesh-backend counterpart
  to `cad_agent_v4.verify_expected()`.
- `scad_step_recovery.py` — `scripts/scad → .csg` then a FreeCAD console script using FreeCAD's
  bundled `importCSG` module to rebuild the CSG tree as real OpenCASCADE Part::Box/Cylinder/Cut/
  MultiFuse features, exported to STEP, then RE-MEASURED (never just trusted) against the STL.
- `scad_agent.py` — the loop itself: an all-new OpenSCAD codegen system prompt (the spike's core
  artifact — see below), OpenSCAD-flavored revise/decide prompts (v4's use "fillet()"/"Pos()"/
  "Mode.SUBTRACT" — none of which exist in OpenSCAD, so word-for-word reuse would be actively
  wrong), and the run/gate/render/critique/decide loop shape mirrored turn-for-turn from `build()`.

## Camera preset verification (methodology, not assumption)

`scripts/scad`'s PNG mode needs a rotation triple for "iso" and "top". Rather than trust the
`--camera=x,y,z,rx,ry,rz,d` docs' phrasing, both presets were verified empirically with a
purpose-built axes probe (three colored arms along +X/+Y/+Z) and an asymmetric marker part (a
plate with a small cube glued to one top corner), rendered at several candidate rotations and
inspected visually (Read tool on the resulting PNGs). Findings:
- **`iso` → `--camera=0,0,0,55,0,25,0`**: confirmed genuinely isometric (Z up, X and Y both
  receding at an angle) — matches the value the task brief suggested.
- **`top` → `--camera=0,0,0,0,0,0,0`** (i.e. **no rotation at all**): OpenSCAD's gimbal camera at
  zero rotation looks straight down −Z. This was NOT obvious in advance — a naive guess of
  `90,0,0` (which reads like "tip the camera 90° to look down") instead produces a **front
  elevation** (X horizontal, Z vertical) and visibly misplaces the corner marker. The unrotated
  camera is also OpenSCAD's own default view with no `--camera` flag at all (confirmed identical
  render), which is consistent with "isometric-first" being the GUI's actual home view and `top`
  being a deliberate, different, unrotated view.
Both presets were then re-confirmed on the real `block_holes.scad` golden: all 4 corner holes are
clearly visible in both panels (dark corner dots isometric, clean circles top-down) — see the
render test in `tests/test_scad_spike.py` and the note recorded there.

## What the mesh gate can (and cannot) verify — honestly

The B-rep gate (`scripts/inspect` + `cad_agent_v4.verify_expected()`) has privileged topological
information: it walks a cylindrical face's actual axis and asks "is there material past this
point?" to tell a THROUGH hole from a BLIND one, deterministically, per-hole.

`scad_mesh_gate.py` has none of that. All it has is the **Euler characteristic** of each closed,
watertight body (`mesh.split(only_watertight=True)`), from which it derives a **topological
genus**: `genus = (2 - euler_number) // 2`. For a simply-connected part, each clean through-hole
raises genus by exactly 1 (same topology as an N-holed torus) — **measured, not assumed**, on all
three goldens: a plain block is genus 0, the 4-through-hole block is genus 4, and the flange (one
centre bore + 6 bolt-circle holes) is genus 7, exactly matching the hand-worked prediction in
`flange.scad`'s own comment (bore alone → genus 1, six more independent through-holes → +6).

The catch: **genus is a lower bound, never an exact count, and it is structurally blind to blind
features.** A blind hole, pocket, or counterbore never punctures the surface — it changes zero
topology — so a part with ten blind pockets and zero through-holes still measures genus 0. This
is why `scad_mesh_gate.gate()` treats a genus shortfall vs. a spec's `min_holes` as an **advisory,
always**, never a hard fail (`test_min_holes_shortfall_is_advisory_only` asserts exactly this).
Everything else the gate measures — loadability, non-emptiness, non-zero volume, watertightness,
volume, bbox, and body count — is EXACT from the mesh. But exact *measurement* doesn't make the
*target* reliable: the `expected` block fed to the gate comes from the brief (`brief["expected"]`,
a qwen3:8b guess), exactly as in the B-rep agent. So the gate mirrors `verify_expected()`'s
2026-06-26 redesign: **body-count mismatch is a hard fail** (structural integrity — unfused or
fragmented geometry), while a **bbox mismatch is an advisory** at the same max(5mm, 10%) tolerance
the B-rep gate uses. Hard-failing bbox here would have biased the spike's A/B against OpenSCAD.
The external `run_benchmarks.py` scorer still applies its tighter max(2mm, 5%) acceptance rule to
the final geometry, so honesty of the *scoring* is unaffected.

## STEP recovery — what it proves and what it doesn't

`scad_step_recovery.py` never trusts FreeCAD's own success message. It re-imports the recovered
STEP with build123d and diffs volume (≤2%) and bbox (≤max(1mm, 2%) per axis) against the STL's own
`trimesh` numbers; only a build123d-confirmed match sets `step_recovered=True`. All three goldens
(plain block, 4-hole block, flange with 7 holes) recovered and validated cleanly — FreeCAD's
`importCSG` module handled every plain-primitive CSG tree in this spike without incident (no BOSL2
was exercised in the goldens; the task anticipated `importCSG` might choke on non-BOSL2 primitives,
but in practice the opposite risk — BOSL2's own generated primitive soup — is the untested one; see
Deviations). This is real evidence for the "OpenSCAD builds can ALSO carry a parametric-adjacent
B-rep artifact" half of X1's framing, at least for CSG expressible in plain primitives.

## How to run the spike benchmark

```bash
cd /home/theultimatecunt/.openclaw/skills/cad-builder
python3 scripts/run_benchmarks.py --tiers 1,2 --coder fast --agent scad_agent.py
```

Caveat for reading the results: `run_benchmarks.py`'s `derive_geometry()`/`score_acceptance()`
are STEP-based (`build123d.import_step` + `scripts/inspect`'s through/blind detector). A
`scad_agent.py` build only carries `step_local` when STEP recovery **succeeded and validated**
(per this spike's result-dict contract); when it didn't, the legacy scorer sees no geometry and
scores that benchmark 0/N on acceptance even though a perfectly valid STL was produced. Read the
acceptance number together with `stl_local`/`step_recovered` in the raw result, not in isolation —
an honest reading of "did this backend build the part" is `ok`/`converged`, not the STEP-only
acceptance score. A future iteration could teach the benchmark scorer to fall back to
`scad_mesh_gate` facts when `step_local` is absent; out of scope for this add-only spike.

## Decision rubric (DIRECTION.md X1) — verdict: **(c) REJECTED, with numbers** (2026-07-10)

The benchmark run above has not been executed as part of this spike (running it means burning the
single GPU on `qwen2.5-coder:7b` for real builds, which this task's hard rules reserve for the
supervisor's own benchmark pass — this spike may not call any LLM, including local Ollama). The
three possible verdicts, restated from DIRECTION.md so the next session can fill this in directly
from the run's numbers:

- **(a) Organic/pattern/artistic backend** — if OpenSCAD+BOSL2 measurably outperforms build123d on
  gear/thread/rounding-heavy specs (where BOSL2's primitives are a natural fit) even without being
  better on plain boxes/brackets.
- **(b) RESCUE rung** — if 7B+OpenSCAD converges on specs where 7B+build123d fails, at a
  speed/quality tradeoff better than escalating straight to 30B+build123d.
- **(c) Rejected** — if neither (a) nor (b) shows a measurable lift over the existing build123d
  ladder, the same way the 14B coder was measured OUT (`m1_14b_tiers12.json`) rather than kept
  around unused.

**Verdict: (c) REJECTED — measured out exactly like the 14B.**
`m8_scad_7b_tiers12.json` (2026-07-10, tiers 1–2, `--coder fast`, same specs/scorer as the
build123d runs): **0/6 converged, 0/6 geometry, 0/22 acceptance** vs build123d-7B's bracketing
runs of 2/6–5/6 (baseline 4/6). Not an infrastructure failure — verified by hand: the runner,
mesh gate, genus counting, renders, and CSG→STEP recovery all pass on hand-written .scad goldens
(a hand-unioned stepped shaft gates clean at 1 body). The failure is the coder itself:
qwen2.5-coder:7b writes **Python-shaped pseudo-OpenSCAD** — `block = cube(...);`,
`result = difference() { ... };`, a bare `result;` — assignments that instantiate no geometry,
leaving only loose cutter cylinders as real bodies (hence the repeated "3 watertight bodies"
gate failures), plus wrong dimensions (Ø3 holes for a Ø8 spec). Reproduced deterministically
outside the benchmark with the exact spike prompt. The v1 lesson stands after all: this model's
build123d-Python is far stronger than its OpenSCAD, *despite* OpenSCAD's training-data abundance
— fluency in a language's syntax is not fluency in producing valid geometry with it.

Consequences: **(b) rescue rung is answered NO** (0/6 can't rescue anything); **M10's "winner"
is build123d** (organic/pattern expansion happens there, not on BOSL2); the **M6′ fine-tune data
mix** question is answered — build123d-only. The OpenSCAD infrastructure (runner, mesh gate,
STEP recovery) is kept: it is verified, tested, and reusable for any future coder that actually
writes valid OpenSCAD (e.g. a fine-tuned model or a stronger base), and the mesh-gate machinery
is backend-agnostic groundwork for any future mesh backend. Re-opening the verdict requires a
NEW measured run, not enthusiasm.

## Spec deviations (and why)

- **No section-cut third panel.** `cad_agent_v4`'s render has an optional mid-plane cross-section
  panel for hollow/internal specs (`wants_section()`). This spike's renderer produces exactly the
  two panels the task specified (iso + top) — a section cut through a *mesh* would need its own
  boolean-against-a-half-space step in OpenSCAD/trimesh, which is real additional scope the task
  didn't ask for. Noted here rather than silently reproduced or silently dropped.
- **No LLM triage / no 14B rung / no cloud rung.** The build123d path has an LLM complexity triage
  call and a 3-way manual override (`fast|mid|strong|cloud`); this spike's ladder is deliberately
  just `fast ↔ strong` (`qwen2.5-coder:7b` / `qwen3-coder:30b`), matching exactly what the task
  asked for. `--coder mid` and `--coder cloud` are accepted (CLI-contract parity with
  `run_benchmarks.py`'s `--coder` choices) but fall back to `fast` with a logged warning rather
  than erroring — never a crash, never a cloud call.
- **`cad_v5/diagnose.py` reused as-is despite being build123d-flavored.** Its regexes will often
  return `"unknown"` for genuine OpenSCAD parser errors (different error vocabulary entirely). It
  is still wired in (rather than skipped) for run-time failures because (1) it is off-limits to
  edit in this spike anyway, (2) some categories (`timeout`, `degenerate_input`) are language-
  agnostic enough to still fire correctly, and (3) an honest `"unknown"` histogram entry is more
  useful signal than inventing a parallel, untested OpenSCAD-specific taxonomy no benchmark run
  has exercised yet. Mesh-GATE failures (as opposed to run-time errors) use a small local
  `_gate_category()` instead, because those messages are ours and we already know exactly which
  check failed — no sniffing needed.
- **`importCSG` risk did not materialize.** The task flagged a real risk ("if importCSG chokes on
  BOSL2-free primitives it should not — investigate and fix"); all three goldens use only plain
  primitives (cube/cylinder/difference) and `importCSG` recovered every one cleanly on the first
  try, with exact BRep boolean volumes (confirmed by hand-computation, e.g. the 4-hole block's
  Part::Cut volume of 118429.2mm³ matches `120000 - 4 × (20mm thickness × π×2.5²mm² hole area)`
  exactly). BOSL2-generated primitive soup (attachments, rounding, thread helpers) was never
  exercised — that remains an open risk for whatever benchmark specs actually invoke BOSL2.

# DIRECTION — from CAD agent to Maker Agent

*2026-07-09. The post-M5 charter: where this goes after the original roadmap. Grounded in a
competitor-methodology study (printpal.io — the strongest documented reference; Zoo Design
Studio; polysolid.ai turned out to be an empty shell publicly). ROADMAP.md remains the record
of what shipped; this file is the wishlist, direction, and sequencing for what's next.*

**North star:** a local-first *maker agent* — from plain-English idea to a made object across
processes (3D print, laser, CNC), with today's verified-parametric pipeline as the trust core.
New capabilities are added AROUND that pipeline, never by replacing it.

**The strongest external validation:** Zoo publicly tried direct text-to-geometry generation
and *abandoned it* for "LLM writes code, code executes, tools verify" — exactly this
architecture. printpal independently converged on brief→validate→critique. We are on the
consensus path; the job is depth, not pivot.

---

## Part 1 — Methodology upgrades (near-term, highest value-per-effort)

From the printpal/Zoo study, ranked for a local agent on small models. Each has an exit
criterion; nothing ships without a measured or behavioral test (standing rule).

**N1. Traceback auto-fix sub-loop** *(cheapest, biggest win)*
On a run exception, silently re-prompt the SAME coder with the raw traceback (1–2 retries)
before anything reaches the gate, critic, or escalation ladder. Tracebacks are the 7B's
dominant failure mode and need zero visual judgment; today each one burns a full loop turn.
Fits inside the run stage of `cad_agent_v4.py`'s loop; the diagnose() hint already computed
there rides along. *Exit: benchmark turns-to-converge drops on tiers 1–2; no convergence
regression.*

**N2. Ambiguity gate — ask, don't guess** *(printpal's "2–5 quick questions")*
A cheap pre-brief triage (qwen3:8b, schema output) classifies the spec: enough concrete
numbers to build, or not. If not, STOP and ask 2–3 bounded questions via Satine/CLI instead of
guessing. One chat turn costs seconds; a wrong build cycle costs minutes (worse if it triages
to the 30B). Pairs with surfacing the brief itself: print the checklist (envelope, features,
assumptions) BEFORE codegen so the user can veto a misread for free.
*Exit: a deliberately vague prompt set ("a wall mount for my router") produces questions, not
builds; specific prompts pass through untouched.*

**N3. Brief as a diffable contract** *(printpal's checklist-diff refine)*
The brief already carries `expected` targets; make the whole checklist a persistent JSON
object that refine turns PATCH rather than regenerate, and show users only the delta
("wall: 2mm→3mm; holes: +4×M3"). We already diff geometry between edits — this diffs *intent*.
Frontends read it directly (Satine message, CLI line). *Exit: refine round-trip shows a
field-level delta and the unchanged fields provably don't churn.*

**N4. API-doc retrieval for the coder** *(Zoo's "reads documentation as it works")*
Second retrieval index over build123d/bd_warehouse API docs — distinct from the existing
solved-examples corpus. Targets the specific measurable failure the histogram already shows:
`api_misuse` (hallucinated signatures). Injected only when triggered (error mentions an API
name) to keep prompts lean. *Exit: api_misuse category count drops across a benchmark run.*

**N5. Reference-anchored critique** *(unlocks image-conditioned builds)*
Satine already downloads and stashes reference photos. When one exists, the critic's question
pass compares render vs reference ("does the render match the drawing?") — a more constrained,
easier judgment for gemma than open critique. This is the on-ramp printpal uses: image →
clarifying questions → build → 4-view sheet → "does this match the reference?"
*Exit: one photo-conditioned build end-to-end via Telegram.*

**N6. Prompting rubric at the surface** *(near-zero cost)*
printpal pushes a "4 S's" doctrine (Size/Specs/Surfaces/Symmetry) + a clearance table
(push-fit 0.0–0.1mm, slip 0.2, loose 0.5–1.0) + named-hardware recognition (M3, 608ZZ, 2020
V-slot). Ours goes in `/help`, `cad --help`, and USER_GUIDE — plus example chips with real
dimensions. Reduces how often N2 even triggers. *Exit: docs updated; help text shows it.*

**N7. Feature tags for refines**
Let refine turns address named features ("@hole2 → 5mm") mapping to build123d selectors /
brief checklist entries. Complements N3. *Exit: a tagged refine changes only the tagged
feature (geometry diff proves it).*

## Part 2 — Capability expansion (the maker tracks)

**X1. Mesh/organic backend — a second codegen target, not a replacement.**
LLMs are genuinely stronger at mesh-style modelling (CSG trees, signed-distance fields,
procedural scripts) for organic/sculptural/artistic parts where B-rep is weak. Design: the
brief gains a backend router — *mechanical/toleranced → build123d (unchanged); organic/
freeform → mesh backend* (SDF/implicit code or trimesh/CSG scripts; the installed text-to-cad
plugin's implicit-CAD skill is a local precedent). Same religion applies: run the code,
measure the mesh (watertight, volume, bbox, min wall), gate, render, critique.
**Boundary (from Zoo's abandonment + printpal's separate lossy image-to-3D product): never
route functional/dimensioned parts to the mesh backend** — it can't hold tolerances, and
that's the system's core promise. *Exit: 5-spec organic mini-benchmark (vase, knurled grip,
low-poly figure, lattice panel, filleted blob) with its own honest scorer.*

**X2. CAM — close the loop from model to machine.**
- **X2a Print:** the gcode skill (slicer CLIs) + Bambu LAN skill already exist in the plugin
  ecosystem; wire a `print` target/command: STL → profiled slice → dry-run-validated gcode
  (+ optional printer handoff). *Exit: one enclosure sliced and dry-run-validated end-to-end.*
- **X2b Laser:** DXF export exists; add kerf compensation + material presets (cut/engrave
  layers) as parameters, so output is cut-ready, not just geometry-true. *Exit: kerf-offset
  DXF measures correct after offset (deterministic check).*
- **X2c CNC 2.5D:** the E2 FreeCAD bridge makes this reachable — FreeCAD's Path/CAM workbench
  is scriptable, and we already generate native `.FCStd` with known features. Generate Path
  jobs (facing, pocket, profile, drill) from the same measured features the gate uses; verify
  toolpath bounds against part bbox + stock, and gcode via dry-run checks. *Exit: one plate
  part → pocket+drill toolpath whose bounds check out; simulated, not cut.*

**X3. Assemblies validation** — the `Compound` path exists untested end-to-end (open item).
**X4. Edit-existing-model** — STEP in → measured facts → recipe recovery → refine loop.
**X5. Print-photo → slicer advisor** — kept from the roadmap horizon; the genuinely novel
half (defect → concrete setting deltas) still has no incumbent.
**X6. Local web UI** — printpal's polish (editor + chat + render side-by-side, example chips,
free deterministic ops) around our CAD Viewer. UX multiplier, deliberately after capability.

## Part 3 — What we will NOT do (bright lines)

1. **No direct text-to-mesh/point-cloud generation for functional parts** — Zoo tried, Zoo
   quit; lossy non-parametric geometry breaks the tolerance promise. Mesh backend is for
   organics only, and still code-driven.
2. **No unbounded auto-iterate.** SaaS can bill runaway loops; a single 8GB GPU cannot. Turn
   and wall-clock budgets stay hard.
3. **No cloud-by-default.** The dormant cloud rung activates only by explicit user config
   with a per-build call cap. Zero spend without opt-in.
4. **No rule-engines replacing LLM judgment** (standing design philosophy) — but also no LLM
   call where a pure assertion suffices (printpal's free-deterministic principle; our helper
   bypass already embodies it).
5. **No unmeasured claims.** Every N/X item lands with its exit criterion demonstrated.

## Part 4 — Sequencing proposal

| Milestone | Contents | Rationale |
|---|---|---|
| M7 | N1 + N2 + N3 (+N6 docs) | reliability + interaction; all small-model leverage, no new deps |
| M8 | X2a print + X2b laser kerf | completes idea→object for the two processes users have today |
| M6′ | fine-tune (whenever budget approved) | unchanged keystone; histogram data keeps accruing |
| M9 | X1 mesh backend + its mini-benchmark | first genuinely new modelling domain |
| M10 | X2c CNC via FreeCAD Path; then X3/X4/X5 | deepest capability, builds on E2 |

N4/N5/N7 slot into whichever milestone touches their files (N5 naturally with X5's photo work).

## Part 5 — Execution notes (token economy)

Per HANDOFF.md's session protocol. Delegation guide: N1, N4, N6, X2b, and doc updates are
**delegate-safe** (well-specified, mechanically verifiable — hand to Opus/Sonnet with exact
specs and exit criteria). N2, N3, X1's router/boundary, and X2c's design need **judgment**
(architecture decisions, prompt design, safety boundaries) — Fable or careful review. Every
delegated change: worktree in `.worktrees/`, offline tests, benchmark gate before merge.

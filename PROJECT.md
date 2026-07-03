# CAD Agent — Project Notes (v4.3 engine + v5 package)

> **Current layout (2026-07-04):** `cad_agent_v4.py` (v4.3) is the ENGINE — brief/codegen/gate/
> critic/loop. `cad_v5/` wraps it: pluggable output targets (local CAD Viewer default, Onshape
> opt-in), the interactive describe→build→refine→undo loop, and `cad_v5/config.py` as the single
> source of truth for models/timeouts/loop constants (v4 imports them from there). The `cad`
> launcher runs `python3 -m cad_v5`; Satine still shells the v4 CLI directly. Agent settings
> (`cad.*` incl. the `code_model` pin) live in **`~/.openclaw/cad.json`** — NOT openclaw.json,
> whose strict gateway schema rejects unknown root keys. Per-build artifacts land in
> `~/.openclaw/cad-builds/<ts>-<slug>/` (last 20 kept) with `cad-last-build.*` as convenience
> copies. Legacy v1/v2/v3 agents live in `legacy/`.

## What This Is

A single-agent, fully-local **build123d agentic observe-edit loop** that turns a plain-English
spec into a live Onshape 3D model. The model writes build123d Python; the system runs it, inspects
the geometry, renders it, has a multimodal critic judge it, and the coder edits until it matches —
then the STEP is translated into a real Onshape Part Studio. Modeled on the Claude↔Fusion MCP
operating pattern, but every model runs locally via Ollama — **no Claude API calls**.

```
python3 cad_agent_v4.py build "a 100x60x30mm enclosure with 2mm walls"
python3 cad_agent_v4.py build "W200x100 I-beam 1500mm" --coder strong
```

Supersedes v3 (pre-loop build123d), v2 (planner/executor), v1 (FeatureScript) — all kept only as
rollback. No v2 delegation, no routing regex.

---

## Architecture

```
spec
 │
 ▼  brief — qwen3:8b (temp 0.2)
{name, dimensions, features, notes, helper}   ← interprets intent (box ⇒ hollow container)
 │
 ▼  coder triage — qwen3:8b  →  start FAST (7B) or STRONG (30B)?
 │
 ▼  codegen — auto-routed coder (temp 0.15), build123d ALGEBRA mode
 │
 ▼  ── per-turn observe-edit loop (≤ MAX_TURNS / BUILD_TIMEOUT) ──────────────
 │   run (scripts/step → STEP) → inspect (scripts/inspect) → render (scripts/render, 2 panels)
 │      → visual critic (gemma4:e4b, sees isometric + top-down) → decide_or_edit
 │      • DONE  → finalize          • edit → next turn          • stuck on 7B → escalate to 30B
 ▼
STEP → translate → Onshape Part Studio (URL)   [public fallback on free accounts]
```

Domain parts (I/H/C sections, spur gears, hex bolts) come from `b123d/domain.py` helpers, are
correct by construction, and **bypass codegen + the critic** (a single foreshortened section view
false-negatives otherwise). ~60s vs ~150s+ for freeform.

---

## File Map

| File | Role |
|------|------|
| `cad_agent_v4.py` | **Current agent** — brief, triage, codegen, loop, Onshape upload, CLI |
| `b123d/domain.py` | Deterministic helpers: `structural_section`, `spur_gear`, `hex_bolt` |
| `scripts/step` `scripts/inspect` `scripts/render` | build123d→STEP, geometry validation, 2-panel PNG |
| `SKILL.md` | Operator-facing usage (commands, loop, routing) |
| `~/.openclaw/cad-examples.jsonl` | Few-shot corpus — gold + ≥4★ rated build123d examples |
| `~/.openclaw/cad-lessons.jsonl` | Fail→fix lessons (build123d pitfalls) |
| `~/.openclaw/cad-embeddings.json` | nomic-embed cache for retrieval |
| `~/.openclaw/cad-session.json` | Last build result (for `rate`, `session`) |
| `~/.openclaw/cad-last-build.step` | Last STEP — recoverable if upload fails |
| `~/.openclaw/openclaw.json` | `env` creds (ONSHAPE_*, Telegram token) + `cad.*` config |
| `~/.openclaw/cad-telegram.py` | Satine — Telegram frontend (shells out to this agent) |

---

## Coder routing (v4.2)

Default **fast** `qwen2.5-coder:7b-instruct-q5_k_m`. Triage (`spec_needs_strong_coder`, an LLM
call via qwen3:8b — **not** keyword rules) starts hard specs on **`qwen3-coder:30b`**. The loop
auto-escalates 7B→30B after `ESCALATE_AFTER` (2) failed/stuck turns and regenerates fresh.

- **Manual:** `build --coder auto|fast|strong`; Satine accepts a `strong:`/`fast:` prefix.
- **Pin:** `cad.code_model` in `openclaw.json` disables auto-switching.
- **Recorded:** chosen model stored as `code_model` in the session/result.
- Implemented via a module-global `_ACTIVE_CODE_MODEL` read by `_code_model()` (safe: one build
  per process; Satine shells out per request).

---

## Key Design Decisions (non-obvious)

- **Centred-origin is the coder's hardest trap.** build123d primitives are centred (`Box(_,_,H)`
  spans z −H/2..+H/2); LLMs instinctively use corner/bottom origin and place cuts outside the
  solid → silent no-op holes. Fix: explicit THROUGH-HOLE rule + worked examples in `_CODE_SYSTEM`
  (`stock - Pos(x,y,0)*Cylinder(radius=r, height=2*thickness)`, z=0). With a targeted corner-holes
  example, the **7B now gets the four-corner-holes case right**.
- **The critic prompt must match the render.** The render is **two panels** (isometric + top-down).
  An earlier prompt said "single isometric render", so gemma described only the iso panel and missed
  small top-face holes → false "missing" → decide tried to "fix" correct code → stuck → false
  `converged:false`. Fixed: critic inspects BOTH panels; decide won't reject a feature that code +
  geometry confirm. **gemma is a CHECKER, not the primary source** — semantic interpretation
  (e.g. box ⇒ container) belongs in the brief/codegen stage, never the critic.
- **Intent interpretation in the brief.** `_BRIEF_SYSTEM` reads the noun: box/case/enclosure/
  housing/container/tray/bin ⇒ hollow open-top ~2mm walls; solid/block/filled/plate/panel/bracket
  ⇒ solid. This is interpreting meaning, kept distinct from the "don't invent extra features" rule.
- **Honesty over false success.** The loop never reports `converged:true` unless confirmed; a
  non-converged build uploads the last valid geometry but is flagged (`converged:false` + warning +
  last_critique). Verify against geometry (volume/faces), not just the render.
- **Determinism for small models.** codegen/revise/decide at temp 0.15, brief 0.2, triage 0.0.
- **Algebra mode + auto-patches.** `Box - Cylinder` style is far more robust for a 7B than
  BuildSketch/plane offsets; `_CODE_PATCHES` auto-fix common API hallucinations.
- **Upload TRANSLATES the STEP into a Part Studio** (`blobelements ... translate=true` → poll
  `translations/{id}` → `resultElementIds[0]`), not a downloadable blob. Free Onshape accounts are
  public-only → tries private first, falls back to public with a warning (`cad.public_uploads`).

---

## Hardware & Models

- **Machine:** HP Z2 Tower G4 · AMD RX 6600 8GB VRAM (ROCm) · `OLLAMA_MAX_LOADED_MODELS=1`.
- **brief / triage / refine:** `qwen3:8b` (~35 tok/s GPU).
- **coder:** `qwen2.5-coder:7b-instruct-q5_k_m` (default, full GPU ~16s) ⇄ `qwen3-coder:30b`
  (30.5B MoE, CPU offload ~7min/call). `qwen2.5-coder:14b` rejected (slow ~217s, not better).
- **critic:** `gemma4:e4b` (multimodal; text on GPU, vision encoder on CPU).
- One model in VRAM at a time → model swaps cost cold-load time; the helper bypass and fast-coder
  default keep common builds quick.

---

## Tuning Surface (where to change behavior)

1. **`openclaw.json` `cad.*`** — `code_model` (pin coder), `public_uploads`; `env.ONSHAPE_*`.
2. **Constants** (top of `cad_agent_v4.py`) — model names, `MAX_TURNS`, `ESCALATE_AFTER`,
   `BUILD_TIMEOUT`, `CODE_TIMEOUT`, `CRITIC_TIMEOUT`, etc.
3. **Prompts** (highest leverage) — `_BRIEF_SYSTEM`, `_CODE_SYSTEM`, `_COMPLEXITY_SYSTEM`,
   `_CRITIC_SYSTEM`, `_DECIDE_SYSTEM`; inline sampling temps.
4. **Per build** — `--coder` flag / `strong:`/`fast:` Telegram prefix.
5. **By use** — `rate ≥4★` to grow the few-shot store.

---

## Open Follow-ups

- **Model upgrade (coder generation) — tracked 2026-06-26.** The agent still runs older coders:
  fast = `qwen2.5-coder:7b` (2024), strong = `qwen3-coder:30b-a3b`. Newer generation is available and
  pulled: `qwen3.6:35b-a3b` (A3B MoE, ~93% HumanEval). Plan: (1) **runtime** — move the strong coder
  off Ollama onto **llama.cpp `--n-cpu-moe`** (keeps the 3B active path + attention on GPU, offloads
  dormant experts to RAM) to kill the 30B timeouts (cases 04/08/10 hit the 35-min cap); (2) **model**
  — A/B `qwen3-coder:30b-a3b` (coder-tuned) vs `qwen3.6:35b-a3b` (newer, general) on tiers 1–2,
  scoring pass-rate AND wall-time (HumanEval ≠ build123d, so measure). Coder-tuning usually beats a
  bigger general model for code, but qwen3.6 is the newer lineage — let the A/B decide. See
  [[project_model_landscape_2026]].
- Richer / auto-oriented render so the critic can judge long foreshortened sections (helpers still
  bypass the critic for this reason).
- Wrap as a Hermes plugin (`~/.hermes/plugins/cad/`) over the clean `build()` API.
- STL/3MF/GLB export + a CAD viewer (ideas mined from `earthtojake/text-to-cad`, MIT) for a future
  3D-print path. That repo is NOT a replacement — it is a Claude-Code/Codex-driven, file-output
  toolkit with no Onshape and no local-Ollama loop — but its build123d benchmark specs were adopted
  (see below).

---

## Learning loop (v4.2) — make a small model punch above its weight

The lever for a local 7B isn't parameters, it's **retrieval of known-good examples + memory of
past mistakes**. Anti-vaporware rule: every stored item must be *consumed* by generation, and the
lift must be *measured* (benchmark with `--no-fewshots` vs default) — anything that doesn't move the
numbers gets removed, not left dormant.

- **Corpus** `~/.openclaw/cad-examples.jsonl` — `{spec, code, source: gold|rated|auto, verified}`.
  Gold = hand-authored + geometry-verified; rated = ≥4★ via `rate`; (auto-promote = future Stage C).
- **Retrieval** `cad_retrieval.py` — `nomic-embed-text` cosine similarity (cached in
  `cad-embeddings.json`), word-overlap fallback. `_load_fewshots` injects the top matches into the
  brief as "SIMILAR SUCCESSFUL BUILDS"; `build(use_fewshots=...)` / `--no-fewshots` toggles it.
- **Fail→fix memory** `~/.openclaw/cad-lessons.jsonl` — when a build recovers from an early failure,
  `distill_lesson` (qwen3:8b) writes one concrete build123d pitfall; `retrieve_lessons` injects the
  relevant ones as "PITFALLS to avoid" on similar future specs.

**Measured (2026-06-04):** spec = "circular flange 100mm OD 12mm, 40mm bore, 8x M7 on a 75mm bolt
circle, 2mm outer fillet", coder=fast(7B). `--no-fewshots` → FAILED all 4 turns (bad fillet-edge
selection). With retrieval → CONVERGED (255s, bbox 100x100x12, 10 cyl faces) by adapting the gold
flange's fillet pattern + bolt-circle trig. Same model/spec/coder; retrieval flipped fail→success.
The run also auto-learned a lesson ("bolt-hole cylinders should be ~2x the thickness for full
penetration"). Stale v1/v2 stores (`cad-feedback.jsonl`, `cad-learnings.json`) archived as
`.v2-archived` — they were the vaporware (accumulated but never consumed).

## Benchmark suite

`benchmarks/text-to-cad/` holds the 10 `earthtojake/text-to-cad` benchmark specs (MIT,
calibration block → centrifugal impeller → planetary gear stage) as `specs.json`, plus
`scripts/run_benchmarks.py` which builds each via the agent and records converged / volume / bbox /
faces / time / code_model / url to `benchmarks/results/`. (Benchmarks are IDs 01–10 across THREE
tiers: 1 = 01–03 simple, 2 = 04–06 moderate, 3 = 07–10 hard; tier 3 — radial engine cylinder,
impeller, spiral staircase, planetary gear — mainly stresses the 30B path on 8GB VRAM.)
Run: `python3 scripts/run_benchmarks.py [--tiers N,M] [--only IDs] [--coder fast|strong] [--agent PATH]`.

**Honest scoring (2026-07-04):** a build that produces no geometry scores 0/N on every applicable
criterion (previously 0/0, which inflated suite scores); bbox check is axis-invariant like the
runtime gate. **Measured baselines (pre-bug-fix engine, honest scorer, in
`benchmarks/results/baseline_*.json`):** pinned qwen3-coder:30b, full suite — **7/10 converged,
acceptance 21/31 (68%)**, ~720s/build. 7B floor (`--coder fast`, tiers 1–2) — **4/6 converged,
13/22 (59%)**, 3–6× faster per build.

---

## History

- **2026-07-04 — reliability pass (9 commits, from a 3-way adversarial code audit):**
  1. *Benchmark honesty:* failed builds score 0/N not 0/0; dead `built` metric removed;
     `--agent` flag; axis-invariant bbox scoring. Honest baselines recorded (30B 7/10; 7B floor 4/6 on tiers 1–2).
  2. *Never lose a verified build:* every gate-passing build is snapshotted (`best_step`); a broken
     final edit now exports the verified geometry instead of raising. Dead `gate_conflict` removed
     (`accepted_via` is the audit signal).
  3. *Brief robustness:* balanced-JSON extractor (raw_decode) + one `format=json` retry + loud
     fallback warning; triage uses the same extractor.
  4. *Gate false hard-fails:* brief-hallucinated bores (dimension not in the spec text) and
     detector-blind/ambiguous orientations demote to advisories; inspect reports an explicit
     `ambiguous` orientation band (dot 0.55–0.85); done-detection no longer misreads prose
     containing "import(ant)" as an edit.
  5. *Toolchain:* `coarse_pitch` fractional-key fix (M2.5 got M2's pitch); `scripts/step` prefers
     an explicit `result` var, warns loudly on discovery-scan export, and no longer exports CLASSES
     from the star-import; atomic corpus/lesson writes.
  6. *Config unification:* `cad_v5/config.py` is the single source of truth; v4 imports constants.
  7. *Satine:* request journal (ack-after-handling — restarts replay instead of dropping builds),
     worker thread + queue (bot stays responsive during 30-min builds), photo download + session
     stash, session pruning, `/rate` exact-prefix, refine revise timeout 60→300s.
  8. *Artifacts:* per-build dirs + retention; loud config errors; `cad.*` settings moved to
     `~/.openclaw/cad.json` (top-level `cad` key made the OpenClaw gateway REFUSE TO START under
     its strict schema — latent since the last gateway restart).
  9. *Hygiene:* legacy → `legacy/`; semantic lesson dedup (cosine ≥0.90) + one-time store cleanup;
     v5 `undo` implemented (instant revert, no rebuild); docs truth pass.
- **2026-06-25/26 (v4.3) — gate hardening, found by *looking at the model*:** a benchmark flange
  scored 3/3 on the gate yet was defective — its "30mm central through-bore" came out BLIND (cut
  ~5mm into a 10mm part) because the gate counted cylindrical faces and a blind hole has one just
  like a through hole. Fixes, all validated end-to-end (the same spec now builds a true 7-through-
  hole flange, converged):
  1. **Through/blind detection** in `scripts/inspect` (`Through-holes`/`Blind-holes`/`Hole detail`):
     per cylinder, classify hole-vs-wall (material outside the cylinder?), then walk the axis —
     through = no material on axis, blind = uncut remainder leaves material. Gate + benchmark scorer
     use `min_through_holes`; `acceptance.json` updated (case 05 = 0, its standoffs are blind by spec).
  2. **`forbid_blind_holes`** (`reconcile_expected()`): if the spec has no blind/pocket/counterbore/
     standoff language (`_BLIND_HOLE_TERMS`), every hole must pass through — catches a blind bore
     WITHOUT trusting the brief's unreliable hole counts.
  3. **Axis-invariant bbox gate:** compare the sorted set of extents, not per-axis — the brief gets
     the three magnitudes right but mis-assigns axes (flange came back [80,10,80] vs [80,80,10]),
     which was hard-failing correct parts. Orientation mismatch is now a soft advisory.
  4. **Blind-bore lesson rewritten** with explicit WRONG `Pos(0,0,-10)*Cylinder` vs RIGHT centred
     `Cylinder` code; the coder now centres through-cuts.
  5. **Upload guard:** `import_step_to_onshape` verifies the translated Part Studio has ≥1 body
     before reporting success; public uploads default true (free Onshape accounts are public-only).
  6. **Reverted** an unsafe "gate overrides critic" auto-accept (premise disproven by the flange);
     gate/critic disagreement now surfaces as `gate_conflict` + `accepted_via` (critic|gate|helper).
  Open thread: now that the gate is trustworthy, accepting a gate-passing build instead of letting
  the critic drive a regressive edit may be worth re-introducing (in the validated re-run, turn 1
  passed the gate but the critic's edit regressed it before the 30B recovered).
- **2026-06-03 (v4.2):** consolidation + naming bump to 4.2. Fast 7B default + LLM coder triage +
  7B→30B escalation + `--coder` / `strong:`/`fast:` manual override; two-panel critic prompt fix
  (gemma was missing small holes); brief interprets box ⇒ hollow container. Retired `cad-build.sh`'s
  v1 dependency (now forwards to v4); refreshed SKILL.md/PROJECT.md/BUILDS.md off the stale v2 era.
  Added the text-to-cad benchmark suite + runner. Verified end-to-end (box-with-holes and open-top
  container both converge on the 7B).
- Earlier v4.x: build123d agentic loop, domain-helper bypass, STEP→Part Studio translation,
  honesty/non-convergence flagging.

# CAD Agent — Project Notes (v4.3 engine + v5 package)

> **Current layout (2026-07-04):** `cad_agent_v4.py` (v4.3) is the ENGINE — brief/codegen/gate/
> critic/loop. `cad_v5/` wraps it: pluggable output targets (local CAD Viewer default, Onshape
> opt-in), the interactive describe→build→refine→undo loop, and `cad_v5/config.py` as the single
> source of truth for models/timeouts/loop constants (v4 imports them from there). The `cad`
> launcher runs `python3 -m cad_v5`; Satine still shells the v4 CLI directly. Agent settings
> (`cad.*` incl. the `code_model` pin) live in **`~/.openclaw/cad.json`** — NOT openclaw.json,
> whose strict gateway schema rejects unknown root keys. Per-build artifacts land in
> `~/.openclaw/cad-builds/<ts>-<slug>/` (last `KEEP_BUILDS` = 200 kept) with `cad-last-build.*` as convenience
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
 ▼  coder triage — qwen3:8b  →  start FAST (7B) or STRONG (30B)?  (14B is off the ladder)
 │
 ▼  codegen — auto-routed coder (temp 0.15), build123d ALGEBRA mode
 │
 ▼  ── per-turn observe-edit loop (≤ MAX_TURNS / BUILD_TIMEOUT) ──────────────
 │   run (scripts/step → STEP) → inspect (scripts/inspect) → render (scripts/render, 2 panels)
 │      → visual critic (gemma4:e4b, sees isometric + top-down) → decide_or_edit
 │      • DONE  → finalize          • edit → next turn          • stuck → climb one ladder rung
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
| `~/.openclaw/openclaw.json` | `env` creds (ONSHAPE_*, Telegram token) — schema-validated |
| `~/.openclaw/cad.json` | `cad.*` agent settings (`code_model` pin, `public_uploads`) |
| `~/.openclaw/cad-telegram.py` | Satine — Telegram frontend (shells out to this agent) |

---

## Coder routing (escalation ladder, 2026-07-04)

`CODE_MODEL_LADDER` in `cad_v5/config.py` is **2 rungs: fast → strong** (`[CODE_MODEL_FAST,
CODE_MODEL_STRONG]`), weakest first. Default fast rung is `qwen3:8b` (GPU, ~35 tok/s; won the
2026-07-12 A/B 15/22 vs the code-tuned 7B's 8/22 — it repairs fillet/chamfer crashes the 7B
spirals on, and keeps ONE model warm across brief/triage/codegen/decide);
escalation climbs ONE rung per trigger (`ESCALATE_AFTER` = 2 failed/stuck turns, or
stuck-with-gate-unverified) and regenerates fresh. Triage (`spec_needs_strong_coder`, an LLM call
via qwen3:8b — **not** keyword rules) makes hard specs SKIP the fast rung and start directly on
`CODE_MODEL_LADDER[1]` = the STRONG 30B (~7min/call, CPU offload) — deliberately the last resort on
8GB VRAM, reached either by triage or by escalation.

There is **no mid rung**: the 14B was measured out twice (`m1_14b_tiers12.json`: 3/6 converged at
583–804s/build, slower than the 30B MoE and weaker than the 7B), and on 2026-07-16 the model +
the `--coder mid` alias / `mid:` Telegram prefix were removed outright (with the retired
qwen2.5-coder:7b, qwen3.5:4b/9b and phi3 — ~28GB freed). The right middle rung for weak hardware
is the cloud (B4/M3).

- **Manual:** `build --coder auto|fast|strong`; Satine accepts `fast:`/`strong:` prefixes.
- **Pin:** `cad.code_model` in `~/.openclaw/cad.json` disables auto-climbing (currently UNPINNED
  — the ladder is live routing).
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
- **coder:** `qwen3:8b` (fast rung, full GPU, 35.2 tok/s measured 2026-07-16) ⇄ `qwen3-coder:30b`
  (30.5B MoE, CPU offload 64/36, 11.2 tok/s gen measured 2026-07-16). The 14B mid rung and the
  old qwen2.5-coder:7b were rejected by measurement and REMOVED 2026-07-16 (see Coder routing).
- **RAM ceiling rule (measured 2026-07-16):** a strong-rung model must be ≤ ~18GB on disk.
  `qwen3.6:35b-a3b` q4_K_M (23GB, ~19GiB CPU-side) cannot even LOAD beside the live desktop on
  31GB RAM — 98% memory pressure, runner never came up. Q3_K_M (~17GB) is the fit-this-box quant.
- **critic:** `gemma4:e4b` (multimodal; text on GPU, vision encoder on CPU).
- One model in VRAM at a time → model swaps cost cold-load time; the helper bypass and fast-coder
  default keep common builds quick.

---

## Tuning Surface (where to change behavior)

1. **`~/.openclaw/cad.json`** — `code_model` (pin coder), `public_uploads`; creds stay in openclaw.json `env.ONSHAPE_*`.
2. **Constants** (`cad_v5/config.py` — single source of truth) — model names, `MAX_TURNS`, `ESCALATE_AFTER`,
   `BUILD_TIMEOUT`, `CODE_TIMEOUT`, `CRITIC_TIMEOUT`, etc.
3. **Prompts** (highest leverage) — `_BRIEF_SYSTEM`, `_CODE_SYSTEM`, `_COMPLEXITY_SYSTEM`,
   `_CRITIC_SYSTEM`, `_DECIDE_SYSTEM`; inline sampling temps.
4. **Per build** — `--coder` flag / `strong:`/`fast:` Telegram prefix.
5. **By use** — `rate ≥4★` to grow the few-shot store.

---

## Open Follow-ups

- **Model upgrade (coder generation) — rung-1 SETTLED 2026-07-16, strong rung pending.**
  Rung-1 shootout ran (tiers 1–2, pinned, honest scorer; `refresh-20260716.log` +
  `run_20260716_*.json`): **qwen3:8b keeps the rung** — 3/6 conv 15/22 acc @173s vs
  granite4:7b-a1b-h 2/6 10/22 @224s (80 tok/s raw but more repair turns ⇒ slower per PART),
  granite3.3:8b 0/6 7/22, qwen3:4b 0/6 0/22 (emits unterminated-string Python — fails even a
  20mm cube; reproduced standalone). Token speed ≠ part speed: convergence rate dominates wall
  time. Strong-rung A/B `qwen3-coder:30b` (4/6, 15/22, 699s baseline) vs `qwen3.6:35b-a3b`
  Q3_K_M is BLOCKED on the Ollama 0.20.5→0.32.x upgrade (sudo): the hf.co GGUF pull downloads
  but model-create fails `Error: 400` on this runtime; blobs are cached, retry is instant
  post-upgrade. q4_K_M separately CANNOT LOAD (23GB > ~18GB RAM ceiling, see Hardware).
  Survey rejects: qwen3-coder-next (52GB), Laguna XS 2.1 (20GB q4 — marginal, revisit at q3);
  no small qwen3-coder exists. llama.cpp `--n-cpu-moe` stays open after the model verdict.
  See [[project_model_landscape_2026]].
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

**Held-out suite (2026-07-16):** `benchmarks/heldout-cqe/` — the 25 danwahl/cadqueryeval tasks
(MIT) ported by `scripts/port_cqe_heldout.py` (15 tier-1, 10 tier-2; NL spec + solids/bbox
acceptance + a reference STL each). Run: `run_benchmarks.py --suite heldout-cqe`; then
`scripts/score_heldout.py <run_tag>` adds registration-aligned reference-STL fidelity checks
(watertight / components / bbox 1mm / volume 2% / chamfer 1mm / hausdorff95 1mm, via
cadqueryeval's own open3d RANSAC+ICP code at `~/repos/cadqueryeval`; scorer chain verified
against a hand-built reference part: 0.10mm chamfer, 6/6 checks). **HELD-OUT CONTRACT: these
specs must never be rated into cad-examples.jsonl or distilled into cad-lessons.jsonl** — they
exist so retrieval lift can't inflate scores.

**Honest scoring (2026-07-04):** a build that produces no geometry scores 0/N on every applicable
criterion (previously 0/0, which inflated suite scores); bbox check is axis-invariant like the
runtime gate. **Measured baselines (pre-bug-fix engine, honest scorer, in
`benchmarks/results/baseline_*.json`):** pinned qwen3-coder:30b, full suite — **7/10 converged,
acceptance 21/31 (68%)**, ~720s/build. 7B floor (`--coder fast`, tiers 1–2) — **4/6 converged,
13/22 (59%)**, 3–6× faster per build.

---

## History

- **2026-07-18 — Stage C shipped (auto-promotion, the fast-rung follow-up):**
  `cad_retrieval.promote_build()` + engine hook — a build that converged AND was accepted
  (gate/critic) auto-appends to `cad-examples.jsonl` as source=stage-c, rating 3 (below
  hand-curated gold/seed; `retrieve()` now adds +0.02/rating-point above 3 as a tie-break so
  curation outranks automation at equal similarity). Spec-normalized dedup, 60-entry cap
  (prune before more), never raises. Benchmark runs are EXCLUDED via `CAD_BENCH=1` set by
  run_benchmarks.py — a suite must not feed its own answers into the corpus it is scored
  with. Seeded the 7B-q4's verified enclosure build (part 05, 3/3) with a caveat-carrying
  `teaches`; its part-01 build was NOT promoted (it chamfered a hole rim, not the perimeter —
  full acceptance, wrong idiom: acceptance doesn't measure chamfers). Verified end-to-end:
  organic spacer build → converged via critic, turn 1 → "Stage C: promoted to corpus (10)".
- **2026-07-17 (evening) — FAST RUNG REVERSED to qwen2.5-coder:7b-instruct-q4_K_M (user
  decision + same-day 2×2 A/B):** tiers 1–2, same engine, four pinned legs — 7b-q4 WITH
  few-shots **13/22 (59%)** / without 8/22; qwen3:8b with few-shots 11/22 (50%) / without
  11/22. The few-shot corpus (distilled from qwen3 builds) lifts the specialist +23pts and
  the incumbent ZERO — retrieval transfers idioms into the model that lacks them; the July-12
  "7B loses 8/22 vs 15/22" result was the old q5 quant on the pre-N1, pre-style-only-fewshot
  engine and is superseded. Trade-off accepted knowingly: brief(qwen3:8b)→coder swap costs a
  VRAM reload per build (7B suite 2068s vs 8B 1164s wall). `CODE_MODEL_FAST` updated in
  cad_v5/config.py; brief/triage/critic unchanged. Follow-up: Stage-C style auto-promotion
  should now rate/distil 7B-q4 builds into the corpus so the few-shot lift compounds.
- **2026-07-17 (later) — heldout baseline + etj pipeline measurements:** (1) *Held-out fidelity
  baseline* (25-part heldout-cqe, qwen3:8b unpinned ladder): 9/25 converged, 17/25 geometry,
  25/50 runner acceptance; `score_heldout.py` strict reference-STL checks pass 4/17 — misses
  are dimensional (volume/hausdorff), not topological. (2) *earthtojake-pipeline A/B* (etj_agent,
  tiers 1–2): the same qwen3:8b that scores 13/22 through the engine scores **0/22 bare-prompt,
  0/22 with the plugin's modeling reference inlined, 4/22 after adding a worked algebra-mode
  example** (and its one pass is the part shaped like the example — few-shot imitation, not
  doc comprehension). granite4/granite3.3 via etj: 0/22 each. This is the cleanest measured
  evidence yet for the engine's scaffolding thesis: brief + retrieved few-shots + critic loop
  ≈ 3× acceptance for small local coders; reference docs alone do nothing for them. (3) The
  first fixed strong A/B attempt (1200s call cap) still died: 7/7 parts hit the 1800s per-part
  cap — engine model-rotation makes a 23GB coder pay a ~6.5min reload per turn; `strong` mode
  now takes a per-part timeout arg (3600s recommended for 35B-class). qwen3.6:35b-a3b etj legs
  timed out per-call at 1200s (empty replies — under instrumented probe, see below).
  (4) *STRONG-RUNG VERDICT — qwen3-coder:30b keeps the rung.* Instrumented probe: a single
  qwen3.6:35b-a3b q4 `/api/generate` with a realistic CAD prompt (2.5–3K-token system+spec)
  completed in **35m32s** (journal, 13:01:45). Cause is prompt PREFILL through CPU-resident
  experts (5.6/23GB on GPU): short prompts generate fine (~66s gen in the 07-16 smoke), long
  prompts take tens of minutes before the first token. At ~35min/call no loop is viable; the
  30B (18GB, ~11GB offloaded) sustains ~7min/call. Capability was never measurable at sane
  budgets on this box — practicality eliminates the q4 35B first. This confirms and sharpens
  the ≤~18GB strong-rung ceiling (2026-07-16). Follow-up recorded, NOT run: unsloth
  `hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M` (16.6GB — inside the ceiling) and the MTP
  variant. The 3600s engine A/B was deliberately skipped: 2 calls/part × ~35min already
  exceeds it; running it would only manufacture more timeout zeros. (1) *The 2026-07-16 "system crashes" were systemd-oomd memory-pressure kills*
  (gnome-shell shot at 20:58 → Wayland session death), triggered by CPU-offloaded 30B/35B
  models thrashing reclaim on 8G swap; several benchmark parts died mid-run (rc=1), tainting
  that day's scores. Fixed + verified live (35B offload at ~0% PSI): swap 8→24G, oomd 50%/20s
  → 80%/60s, and `scripts/run_refresh.sh` — a generalized driver that detaches into a
  `MemoryHigh=12G` systemd unit, gates stages on `/proc/pressure/memory`, and refuses
  strong-rung legs under 20G RAM+swap headroom. (2) *Clean rung-1 shootout (tiers 1–2):*
  qwen3:8b **13/22 keeps the fast rung**; granite4:7b-a1b-h 6/22 (bimodal — 3/3 on BOTH
  tier-2 parts incl. the never-solved stepped-shaft-keyway, but tier-1 build123d API flubs
  it never recovers from: `SyntaxError`/`float(Vector)` through all N1 retries + 4 turns);
  granite3.3:8b 2/22; qwen3:4b 0/22. The recurring rc=1/model=None rows were NOT an engine
  bug — build() correctly returns a no-result failure when codegen never runs; the runner
  now preserves the CLI's error JSON + stderr tail (`stderr_tail`) so this is diagnosable
  from the run JSON. (3) *earthtojake pipeline adapter* (user request): `agents/etj_agent.py`
  drives any Ollama model through the text-to-cad plugin's own contract (`gen_step()` source
  → their `scripts/step` → `scripts/inspect --facts` feedback loop, no brief/few-shots/critic/
  gate) and emits the runner JSON — `run_refresh.sh etj <models…>` benches model+their-toolchain
  as an independent second pipeline under identical acceptance. Selftest green. (4) Stage-1
  Q3 pull failure explained: unsloth tags are `UD-Q3_K_M` (16.6G), not `Q3_K_M`; strong A/B
  runs on the local official q4 instead (quant-matched vs the 30B incumbent). Strong-rung
  qwen3.6:35b-a3b full-suite A/B + heldout qwen3:8b baseline + etj legs running overnight.
- **2026-07-16 — Coder-model refresh pass (Claude Code session):** (1) *Cleanup:* removed
  qwen3.5:9b/4b, phi3, qwen2.5-coder:7b/14b (~28GB) and the `--coder mid` alias everywhere
  (config/engine/cli/runner/Satine); openclaw.json cad agent repointed to qwen3:8b; gateway +
  Satine restarted clean. (2) *Measured:* speed table (gen tok/s, same CAD prompt) — granite4:
  7b-a1b-h 80.0 · qwen3:4b 58.5 · qwen3:8b 35.2 · granite3.3:8b 34.8 · qwen3-coder:30b 11.2;
  and the qwen3.6:35b-a3b q4 load failure → the ≤~18GB strong-rung RAM-ceiling rule. (3) *Suite
  hardening:* heldout-cqe suite + score_heldout.py (see Benchmark suite). (4) *Facts corrected:*
  there is NO small qwen3.6 (27b dense / 35b-a3b only); qwen3.6 has CONTROLLABLE thinking (the
  engine's `think:false` applies) and flagship coding scores — the "3.5/3.6 never for text" rule
  is 3.5-only pending the A/B. Open: Q3_K_M pull, rung-1 shootout, strong-rung A/B, Ollama
  upgrade. Operational rule learned: ONE heavy job at a time on this box (31GB RAM + desktop),
  monitor /proc/pressure/memory between stages.
- **2026-07-10 — M7 methodology trio + M8 OpenSCAD spike (code) + M9 CAM shipped (DIRECTION.md sequencing):**
  1. *M7/N1 traceback auto-fix:* syntax/run failures now retry INLINE inside the turn
     (`N1_RETRIES=2` in `cad_v5/config.py`) with the raw error re-prompted to the same coder;
     recovery falls through to inspect without burning the turn, exhausted retries count one
     failed turn so the escalation ladder still works; `turns` + `n1_autofixes` recorded per
     benchmark row. **Measured honestly:** two tier-1/2 7B runs bracket the 4/6 baseline —
     `m7_7b_tiers12_run1.json` 2/6 (7/22) vs `m7_7b_tiers12_run2.json` **5/6 (15/22)**; variance
     dominates at n=6. N1 never fired on converging builds (all ≤2 turns) and did not rescue the
     persistent-error builds (the 7B loops on one failure category 10-12×: edge_selection,
     api_misuse, fillet_chamfer) — recorded as *no measured lift; kept because it is zero-cost
     when idle*; follow-ups: strategy-change on final retry, N4 API-doc retrieval for api_misuse.
  2. *M7/N2 ambiguity gate:* `triage_ambiguity()` (qwen3:8b, schema) asks 2-3 bounded questions
     instead of guessing when a spec lacks its basic size/form. Interactive CLI asks inline;
     `--json` mode is opt-in via `--ask` (benchmark path provably unchanged); Satine asks in-chat
     and resolves the next plain message as the answer. Behavioral test: 3/3 vague prompts →
     questions, 3/3 tier-1 specs → pass-through (`tests/behavioral_n2.py`).
  3. *M7/N3 brief-as-contract:* the brief is a persistent, PATCHABLE contract
     (`~/.openclaw/cad-contract.json`); refine turns run `patch_brief()` → deterministic
     `apply_brief_patch()` (untouched fields byte-identical) → `build(brief_override=…)`, and
     surface a field-level `Δ intent:` line (CLI + Satine). Full-regeneration fallback on any
     patch failure. Satine cross-subprocess delta remains a known gap (fresh `--once` per turn).
  4. *M7/N6:* 4 S's prompting rubric + clearance table + named-hardware note in `cad --help`,
     USER_GUIDE, SKILL.md, Satine `/help`.
  5. *M8 OpenSCAD spike (code merged, verdict pending measurement):* `scripts/scad` runner
     (AppImage, verified camera presets — "top" is the UNROTATED camera, not 90,0,0),
     `scad_mesh_gate.py` (trimesh: watertight/volume/bbox/body-count; genus as a through-hole
     lower bound — advisory, blind features invisible to topology), `scad_step_recovery.py`
     (.csg → FreeCAD importCSG → STEP, revalidated by build123d volume/bbox diff — all three
     goldens recovered ≤2%), `scad_agent.py` (legacy benchmark CLI contract, no few-shots — the
     training-data hypothesis stands alone). Gate parity: bbox demoted to advisory to mirror the
     2026-06-26 B-rep gate redesign (the expected block is a brief guess in both backends).
  6. *M9 CAM:* `print` output target — OrcaSlicer 2.4.2 AppImage headless (xvfb-run), Bambu
     profiles auto-extracted to `~/.openclaw/cam-profiles/`, deterministic gcode dry-run
     validation (result.json + header + extruding-moves-within-bed read from the profile's
     `inherits` chain). Gate met: 100×70×30 enclosure → 150 layers, in-bed, validated; open-shell
     mesh genuinely fails. `scripts/dxf --kerf/--material`: outer +kerf/2, holes −kerf/2
     (`Kind.INTERSECTION`), measured exact via ezdxf (80.30×50.30 / Ø4.70 at kerf 0.3).
  7. *M10 organic mini-benchmark (measured) + M11 CNC:* organic suite (5 specs, `--suite` runner
     flag, `min_faces` criterion) first run: **2/5 conv, 6/13 acc, all specs → 30B via triage**
     (`m10_organic_auto.json`); min_faces immediately caught the critic accepting two
     under-perforated parts. M11: FreeCAD CAM headless pocket+drill toolpaths with a pure-python
     gcode envelope/crash/drill-position verifier (`cad_v5/cam_cnc.py`, `scripts/cnc`), gate part
     verified simulated — the FreeCAD 1.x CAM API findings + the headless `findToolController`
     bug workaround are documented in the module.
- **2026-07-06 — M4.1 / M5 / E2 shipped (see `ROADMAP.md` §8 for milestone detail):**
  1. *M4.1 question-based critique:* `verify_questions` + `_CRITIC_QA_SYSTEM` in `cad_agent_v4.py`
     turn the critic into per-feature binary Q&A (CADCodeVerify pattern); the gate never asserts a
     bore orientation the spec text doesn't state (`_spec_states_orientation`); `edge_selection`
     failure category added to `cad_v5/diagnose.py`; `scripts/render --section` cuts across the
     LONGER of X/Y (a hardcoded mid-Y cut slivered parts lying along X). Result:
     `m41_7b_tiers12.json` 4/6 converged, 13/22 acceptance, zero critic false-blocks.
  2. *M5 v5 migration (B7):* `scripts/run_benchmarks.py` defaults to the v5 `--json` entry
     (`DEFAULT_AGENT="v5"`); Satine (`~/.openclaw/cad-telegram.py`) uses `run_v5_build()`
     single-JSON-line parsing and `send_build_files()` to send the render photo + STEP/STL in-chat;
     Onshape upload is single-sourced in `cad_v5/targets.py` (v4's `import_step_to_onshape`
     delegates to it).
  3. *E2 FreeCAD target:* `cad_v5/freecad_export.py` produces a verified parametric `.FCStd` for
     box-grammar parts (Params spreadsheet + Cut tree), verified by re-exporting STEP and diffing
     volume/bbox; `freecad` target registered in `targets.py`; runs via the FreeCAD 1.1.1 AppImage
     at `~/Applications` (no `freecadcmd` on PATH).
  4. *Retention:* `KEEP_BUILDS = 200`; benchmark artifacts persist per-run under
     `benchmarks/results/artifacts/<run_tag>/`.
  5. *Showcase:* full 30B run (`--coder strong`, current stricter engine) —
     `showcase_30b_full.json`: 6/10 converged, 9/10 geometry, 22/31 acceptance (71%).
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
- **2026-07-17/18 (image-conditioned builds + web UI + build lock):** CADAM-pattern `--image`:
  gemma4:e4b pre-pass → structured analysis (proportions/features; absolute mm ONLY from user
  text or in-image annotations; cached per photo) merged into the brief, and the critic receives
  the reference as a SECOND image each turn (two-image attention verified: distinguished clevis
  vs 4-hole block, 110s cold). Machine-wide `fcntl.flock` in `engine.build()` serializes all
  frontends on the single GPU (verified live: queued 9 min behind a running benchmark; budget
  starts post-acquisition). Satine: captioned photo = spec+reference (routes build OR refine);
  bare photo = 30-min single-use stash. Web UI `webui/` (FastAPI, cad-web.service,
  127.0.0.1:8090, tailnet-only via tailscale serve :8443; RUNBOOK.md). **Measured (1 of 3 A/B
  cases, results/image_ab_20260717.json):** stdout purity held both legs; image cut turns 4→1
  (503s→332s) and steered form (through-holes+gusset vs plain bar) — but NO quality lift
  claimed: BOTH legs ignored the explicit "80mm long" (form-vague spec → brief emits
  bbox_mm=null → gate has no envelope check; open follow-up: parse "NNmm long/wide/tall" into
  a [spec] advisory), and the ref-aware critic's 1-turn accept is a leniency watch item.
  Remaining A/B cases: rerun via a MemoryHigh-capped unit (runner can escalate to the 30B rung).
- **2026-07-19 (GIFT adoption — arXiv 2603.27448; all measured):** the paper's data-bootstrapping
  recipe implemented end-to-end, no-money scope. **Best-of-N first turn** (`--candidates`,
  `CAD_CANDIDATES`, cad.json `candidates`; deterministic execute→inspect→gate selection, temps
  0.15/0.45/0.7): A/B tiers 1-2 fast rung, same engine both legs, capped unit, 0 oomd kills —
  **N=3 acc 10/22 + wall 1221s vs N=1 acc 5/22 + wall 2097s** (run_20260719_191156/193217.json);
  default now 3. Convergence 1/6 both legs — the same-day gate hardening refuses measurably
  incomplete parts the old engine passed, so the 2026-07-17 13/22 is non-comparable history.
  **Gate hardening** (3 new [spec] checks, all unit-tested, each from an observed live failure):
  spec-text axis dims ("80mm long" — closes the image-A/B follow-up), hole OVERcount ("a 10mm
  hole" shipped with five), hole spacing/bolt-circle via hole_groups circle Ø ("34mm apart"
  shipped at 24mm — a render cannot measure distance). Overcount-only for counts: inspect's
  through-hole classifier misses non-axial bores (gold L-bracket's radial holes measure 0 —
  logged as an open inspect gap). **M6′ harvest live:** per-build `turns/tN.py|step|png` +
  `build_meta.json` persist; converged organic builds append good + GIFT-FAIL pairs to
  `~/.openclaw/cad-sftpairs.jsonl` (CAD_BENCH excluded). **Retro census** (harvest_census.py):
  2 clean pairs from 154 dirs — 100 bench, 40 no-source, and 5/7 log-converged organic builds
  FAILED offline geometry checks (brick-as-box, 100/168mm "80mm" brackets, 5×Ø10 for 1, 4×Ø6
  for 2). **gift_sample.py** (corpus-only ground truth, per-spec build lock, N1-style repair
  retry): first full run 13 specs × K=8 → 3 match + 12 valid + 20 near_miss = 35 pairs, 75 min
  (gift_sample_20260719_204634.json); corpus audit then purged 2 poisoned stage-c rows (168mm
  bracket, 5-hole plate) + their 8 derived pairs → **dataset 30 clean pairs day one, corpus 11
  (10/11 pass audit; the solid "box with holes" gold is a flagged user call)**. Fine-tune
  itself: GPU rental pending (UNSW).

## 2026-07-30 — Teacher distillation built; a user CAD review reset the quality bar

**Teacher pipeline (M6′ data side).** `scripts/teacher_gen.py` distils a Claude teacher into
verified build123d pairs: production-identical prompts captured from `engine._LAST_PROMPT`
(never re-derived), deterministic accept/reject with no LLM in the verifier, one repair call per
failure to harvest fail→fix pairs, a hard contamination guard (exit 2) against text-to-cad /
organic / heldout-cqe, a per-call spend ledger and a `--budget` cap that stops BEFORE the spec
that would breach it. Pre-flight estimator projected $2.23 for the 12-mechanism run against
$2.29 actual (97%).

**Measured runs.** Pilot 40 specs: Sonnet 5 **31/40**, Opus 5 **33/40** — 3 genuine outcome
differences, a tie inside the noise floor at n=40, Opus 2x faster wall-clock at 2.4x the price.
Verdict: Sonnet for volume. Tier-4 mechanisms (12 specs, $2.29, $0.191/spec): gate said 6/11.

**The correction that matters.** A user review of all 11 mechanisms in the CAD viewer returned
**3/11 acceptable** (bearing, crankshaft, cam+follower). The other eight were dimensionally
clean and mechanically impossible: a connecting rod through a piston crown, gear teeth
interpenetrating (twice — centre distance was right, tooth PHASE was not), impeller fins
unattached to the hub, a Geneva mechanism whose parts sat 20.7mm apart, a worm whose swept
helical thread contributed nothing (volume measured EXACTLY a plain r8x60 cylinder — a boolean
with a failed operand silently returns the original solid). **The gate measured conformance; the
user measured whether the object could exist.**

**Five checks derived from that review** took gate/human agreement from 3/11 to **9/11**:
`interference` (HARD), `part_gaps` (>1mm = not assembled), bare-primitive face count,
per-part fragmentation, and mesh rules in `_CODE_SYSTEM`. Residual misses are honest: a
bearing's 8 balls legitimately ARE 8 bodies (flags, asks), and a geometrically valid part that
simply does not resemble a universal joint needs a human or a vision critic.

**Gate corrections (all regression-tested against the 2026-07-19 audit failures, which still
fail).** `corroborate_expected()` — the brief (qwen3:8b) no longer grades the teacher; assembly
overcount demoted to advisory; multi-component specs skip envelope-dim checks. A replay showed
the pre-revision gate had rejected 5 of 11 correct-by-measurement mechanisms.

**Brief retired from the prompt path, measured not assumed.** A/B on the local 7B, tiers 1-2,
same engine: WITH brief 3/6 converged / 14/22; BRIEF-LESS **6/6 converged / 15/22**. Acceptance
is flat at n=6, convergence is not. Production defaults brief-less; `cad.use_brief=true` reverts.

**Regression found and fixed the same day:** extracting the few-shot injection into
`inject_retrieval_notes()` dropped a local `fewshots` list that `result["fewshots_used"]` still
read — every live build completed its work then died at finalisation with a NameError. It hid
because `teacher_gen` bypasses `_build_impl`. Returning the rows (not counts) fixed it.

**Dataset:** 136 pairs; 11 mechanisms human-reviewed (3 accept / 8 reject) with reasons in
`~/.openclaw/cad-review-decisions.jsonl`. **GPU: UNSW Katana**, not rental.

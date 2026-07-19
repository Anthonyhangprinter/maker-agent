# CAD Agent Roadmap — Local-First Text-to-CAD

*2026-07-04. Written after the Phase-1 reliability pass (9 commits) and grounded in an external
survey of established text-to-CAD systems: Zoo's Zookeeper agent, CADCodeVerify (ICLR 2025),
CADSmith, CADDesigner/ECIP, SeekCAD, and the Text2CAD/CADmium fine-tuning line.*

---

## 1. Thesis

**The bottleneck is the LLM interface, not model size.** The honest baseline proves it: the 7B
coder — 18× smaller than the 30B, fully GPU-resident on 8GB — already converges 4/6 on tiers 1–2
at 3–6× the speed. What separates it from the 30B is not intelligence, it's *support*: how
reliably we constrain its output format, how precisely we diagnose its failures, how sharp the
verification signal we feed back is, and how good the retrieved examples are. Every roadmap track
below invests in that interface, because interface work is portable — it benefits an 8GB laptop, a
24GB workstation, and a cloud model equally, while a bigger-model dependency benefits only the
machines that can run it.

This matches where the field landed: Zookeeper "behaves like an engineer" not by being huge but by
*executing the model frequently, inspecting geometry, and reviewing multi-view snapshots* — the
same observe-edit loop this agent already runs. CADDesigner's benchmark found **build123d has the
highest Pass@1 and lowest latency of the CAD code libraries** — validating the library choice.
The remaining gap to the published systems is structured discipline around the model, not scale.

## 2. Where the system stands (measured, 2026-07-04)

Honest scorer (failed builds count 0/N; axis-invariant bbox):

| Config | Suite | Converged | Acceptance | Speed |
|---|---|---|---|---|
| pinned qwen3-coder:30B (CPU offload) | all 10 | **7/10** | 21/31 (68%) | ~720 s/build |
| qwen2.5-coder:7B (GPU) | tiers 1–2 (6) | **4/6** | 13/22 (59%) | ~250 s/build |

Pipeline: spec → **brief** (qwen3:8b, JSON) → few-shot + lesson retrieval (nomic-embed) → **coder**
(7B ⇄ 30B routing) → execute (`scripts/step`) → **inspect** (deterministic OCP geometry:
solids/through-vs-blind/bores/walls) → **gate** (structural integrity hard-fails; spec guesses are
advisories) → render (3-panel) → **visual critic** (gemma4:e4b) → decide/edit, ≤4 turns. v5
package wraps the engine with output targets (local CAD Viewer default / Onshape / file) and the
describe→build→refine→**undo** loop. Frontends: `cad` CLI, Satine (Telegram; journaled queue).

Phase 1 (done) fixed the trust layer: verified builds can no longer be lost, the benchmark can no
longer flatter, hallucinated feature-checks can no longer block convergence, and Satine can no
longer silently drop requests. Details: `PROJECT.md` History 2026-07-04.

## 3. Track A — Reliability floor (DONE, keep enforced)

The rules Phase 1 established, kept as invariants:

- **Never lose verified geometry** — every gate-passing build is snapshotted and is the export
  fallback (`best_step`).
- **Honest measurement** — failed builds score 0/N; `converged` is the headline number; benchmark
  before/after every engine change (`scripts/run_benchmarks.py`, baselines in
  `benchmarks/results/baseline_*.json`).
- **Absence of evidence ≠ evidence of absence** — a detector that can't measure demotes its gate
  check to advisory; only spec-corroborated dimensions may hard-fail.
- **No silent degradation** — config/parse failures log loudly; every store write is atomic.

## 4. Track B — LLM interface (the core track)

Priority order; each item names its published precedent. Exit criterion for the track: **7B ≥
30B-baseline convergence on tiers 1–2** — then the 30B pin (`cad.json code_model`) is retired.

**B1. Structured outputs everywhere** *(highest leverage; Ollama JSON-schema `format`)*
Brief, triage, and the decide step move from free-text + regex parsing to schema-constrained
decoding (`{"action": "done"|"edit", "reason", "code"}` for decide). Small models benefit
disproportionately from grammar constraints — no capacity wasted on formatting. Phase 1's
`format=json` retry was the down-payment; this makes it the primary path. Keep the extractor as
fallback for non-Ollama backends.

**B2. Deterministic toolchain contracts** *(CADSmith's "programmatic validation with structured
diagnostics")*
(a) `scripts/step` requires `result =` (the discovery scan becomes an error whose message teaches
the reviser; update the 6 gold corpus entries first). (b) `scripts/inspect --json` emits a machine
block; `parse_facts` reads JSON with regex fallback — removes the byte-exact coupling between the
gate and inspect's print strings.

**B3. Error taxonomy → targeted repair hints** *(CADSmith)*
Formalize the ad-hoc fillet hint into a table-driven `diagnose(error) → (category, repair_hint)`:
syntax / build123d-API misuse / invalid geometry / gate-dimension / gate-feature / visual. Log the
category per turn — **the histogram tells the fine-tune track (D) exactly what to train on**, and
structured hints are the cheapest way to make a 7B recover instead of flail.

**B4. Cloud escalation rung** *(explicit user requirement)*
`_llm()` provider dispatch: Ollama | OpenRouter | Anthropic (all plain HTTP chat calls, no SDKs).
Routing becomes `fast → mid → strong → cloud` with a per-build cloud-call cap; on low-VRAM machines the
30B rung is config-skipped so it's `7B → cloud` directly. Config: `cad.json → cloud: {provider,
model, api_key_env, max_calls_per_build}`. This is the portability mechanism: the same agent runs
on any low-end machine with the local rungs it can afford and a paid escape hatch it controls.

**B5. Question-based visual critique** *(CADCodeVerify — measured +7.3% geometric accuracy over
free-form critique)*
Two-pass critic: (1) generate 3–6 binary verification questions from the spec ("Is there a
through-hole in the top face?"), (2) answer each from the render panels → JSON
`{question, answer, evidence}` list. Converts gemma's vague prose into per-feature signals the
decide step can act on. Prompt-level change; same render pipeline.

**B6. Retrieval hygiene — DONE (2026-07-04, measured)** *(SeekCAD-style RAG discipline)*
Semantic lesson dedup live; thresholds set from measured score distributions (genuine ≥0.77,
CAD-noise 0.58–0.61 → floors raised to 0.65 examples / 0.6 lessons); lessons capped at 3.
**Few-shot lift re-measured on the M1 engine (m2_7b_nofewshots.json): WITH retrieval 5/6 conv
16/22 acc vs WITHOUT 3/6 conv 12/22 acc — the learning loop decisively earns its keep** (the
flange fails outright without its gold example; the L-bracket loses its vertical arm).

**B7. v5 migration completion** *(strangler pattern)*
`--json` result output on the v5 CLI → benchmark `--agent` default flips to v5 → Satine switches
to the v5 entry and drops stdout scraping → Onshape upload dedups into `cad_v5/targets.py`. Each
B-item extracts its module (`brief.py`, `geometry.py`, `diagnose.py`, `critic.py`) behind
`engine.py`'s stable names, benchmark-parity checked.

**B8. Renderer truthfulness** *(lowest priority — the critic is advisory)*
matplotlib's painter's algorithm can mis-occlude concave parts. Interim: brief-driven section
plane (not hardcoded mid-Y) + finer subdivision. Full: optional pyrender/EGL offscreen backend
with matplotlib kept as the guaranteed software fallback — **no GPU/GL hard dependency**, ever
(portability rule).

## 4b. Track E — Editable models (feature-tree handoff)

*User requirement (2026-07-04): adjust a finished model in CAD software without re-running the
agent.* Constraint: STEP cannot carry a feature tree (all history formats are proprietary). But the
generated build123d script IS the parametric source — so:

- **E1 (near-term, rides with B1/M1): named parameters + `regen`.** Codegen contract requires every
  key dimension as a named constant at the top of the script (`wall = 2.0`). New commands:
  `cad params` (list them), `cad regen wall=3` (re-execute with overrides — seconds, no LLM,
  geometry re-verified by the same gate). The saved `build_source.py` per build-dir is the input.
- **E2 v1 DONE (2026-07-06, live-verified): native parametric FreeCAD document target.**
  `--target freecad` is REGISTERED in `cad_v5/targets.py` and implemented in
  `cad_v5/freecad_export.py`. For **box-grammar parts** it builds a genuine parametric `.FCStd`: a
  `Params` spreadsheet holds the key dimensions (length/width/height/wall/hole diameters) and the
  outer box, inner cavity, and hole cutters bind to those cells by expression, so editing a number
  in the FreeCAD GUI regenerates the model. Non-box parts fall back to a plain STEP import into a
  native `.FCStd` (still opens/edits, no feature tree). It runs via the **FreeCAD 1.1.1 AppImage** at
  `~/Applications` — `_freecad_invoke()` locates it (there is no `freecadcmd` on PATH). Verification
  is the same honesty gate as everything else: `convert()` re-exports STEP from the generated `.FCStd`
  and diffs volume/bbox against the agent's own STEP (verified ΔV ~0.71% on the calibration block).
  E2 v2 (open): extend the grammar to cylinder-based parts (flanges, shafts), blind holes/pockets,
  and fillet/chamfer as real Part-Design features.
- **E3 (long-term): Onshape FeatureScript emission, revisited.** v1's approach, retired because
  small models couldn't write the niche language — worth revisiting only after Track D, and only as
  a post-verification EXPORT step (build+verify locally in build123d first, then emit FS).

## 5. Track C — Portability tiers

The tier model the codebase should encode explicitly (auto-detected, overridable in `cad.json`):

| Tier | Hardware | Coder rungs | Critic | Notes |
|---|---|---|---|---|
| **T0 floor** | CPU-only / ≤4GB | 7B q4 (slow) → cloud | off (gate-only) | degraded but functional |
| **T1 (now)** | 8GB VRAM | qwen3:8b GPU → 30B† → cloud | gemma4:e4b | 14B measured out and removed 2026-07-16 (with `--coder mid`); † 30B = last resort. RAM-ceiling rule: strong rung ≤ ~18GB on disk (31GB RAM + desktop) |
| **T2** | 16–24GB | fine-tuned 7B → 30B GPU → cloud | larger VLM | 30B becomes resident |
| **T3** | 96–128GB | 70B-class local | large VLM | no cloud needed |
| **Cloud burst** | any | OpenRouter/Claude | cloud VLM possible | per-build cost cap |

Design rules that make this real: single `_llm()` seam (B4); graceful degradation already in the
loop (no critic → gate-only; no embeddings → word-overlap retrieval); Ollama the only hard
dependency; all tier differences expressed as config, never code branches.

## 6. Track D — Fine-tune (make the 7B rung sufficient)

The single biggest proven lever for small-model CAD quality (Text-to-CadQuery: consistent gains
fine-tuning on 170k pairs; CADmium: Qwen2.5-Coder-14B on JSON CAD sequences; CAD-Coder: geometric-
reward RL on top).

- **Base:** the current fast rung (`qwen2.5-coder:7b-instruct-q4_K_M` since the 2026-07-17
  few-shot 2×2; QLoRA on a rented A100/4090, a day or two).
- **Data (methodology adopted 2026-07-19 from GIFT, arXiv 2603.27448 — see the maker-agent
  packet's GIFT-UPGRADE-PLAN.md):** the harvest machinery is BUILT and live —
  `~/.openclaw/cad-sftpairs.jsonl` accumulates (spec, code, render) good pairs from converged
  organic builds and GIFT-FAIL pairs (wrong-turn render + bad code → correct code);
  `scripts/gift_sample.py` amplifies the corpus by sampling K candidates per spec and
  band-scoring them against the stored verified geometry (`scripts/geom_bands.py`);
  `scripts/harvest_census.py` mines/audits history. **Census 2026-07-19: history is thin (2
  clean pairs from 154 dirs — 5/7 log-converged organic builds failed offline geometry
  verification, which forced the [spec] gate-hardening the same day).** So the mix is:
  sampler amplification + go-forward capture + Text-to-CadQuery 170k adapted toward build123d
  idioms (+ optionally a GenCAD-Code-derived image set for the --image path). B3's error
  histogram picks the failure modes to oversample.
- **Eval:** the 10-spec suite (`--coder fast`, honest scorer) + the `heldout-cqe` suite (25
  cadqueryeval tasks with reference-STL scoring, live since 2026-07-16 — see PROJECT.md).
  Success = fine-tuned 7B ≥ stock-30B baseline (7/10) on the full suite.
- **Deploy:** GGUF quant → Ollama tag (e.g. `cad-coder:7b`) → swap `CODE_MODEL_FAST` in
  `cad_v5/config.py`. Zero code changes — that's the point of the config unification.

## 7. Horizon — toward a general local CAD copilot

Explicitly out of scope until Tracks B+D land, in intended order:

1. **Assemblies validation** — the v4.3 `Compound` path exists but is untested end-to-end; one
   2-part assembly benchmark spec, then gate tuning.
2. **Image-conditioned builds** — Satine already downloads and stashes reference photos (Phase 1);
   wire them into the brief via gemma4:e4b image description → spec enrichment ("build a bracket
   like this photo"). Zookeeper lists multimodal input as its own next step.
3. **Photo-of-print → slicer-settings advisor** — the long-term differentiator. Established
   precedent for the *detection* half (Obico's failure detection, first-layer taxonomies:
   under/over-extrusion, adhesion, warping, stringing); the *recommendation* half (defect → profile
   deltas: temps, flow, speed, retraction) is greenfield. Sketch: VLM defect classifier →
   deterministic mapping table (the domain-helper philosophy applied to slicer profiles) → optional
   LLM explanation. Needs: labeled defect photos, printer/profile context, G-code awareness (the
   `cad:gcode`/Bambu skills are the on-ramp).
4. **Edit-existing-model** — inspect an uploaded STEP → recover a parametric build123d
   approximation → refine loop on it (Zookeeper's "reverse engineering" direction).

## 8. Sequencing & exit criteria

| Milestone | Items | Exit criterion (measured) |
|---|---|---|
| M0 (done) | Phase-1 reliability, honest baselines | 9 commits; baselines recorded |
| M1 | B1 structured outputs + B2 contracts + E1 params/regen | 0 parse-failure fallbacks in a full suite run; 7B tiers 1–2 ≥ 5/6; `cad regen wall=3` round-trips |
| M2 | B3 taxonomy + B6 retrieval hygiene | **DONE** — histogram in every result; lift measured: retrieval = +2 converged, +4 acc points |
| M3 | B4 cloud rung | **BUILT** (offline-verified both providers; live validation deferred — no API credit). Enable: add `cloud` block to cad.json |
| M4 | B5 question critique + B8 interim render | **DONE (as M4.1)** — first validation regressed 2/6 and the postmortem found two real bugs (brief-hallucinated bore orientation steering the coder wrong; unclassified edge-selection crashes); fixed, re-validated 4/6 with ZERO critic false-blocks. The gate is now stricter AND honest: unmet spec-corroborated features block, hallucinated ones cannot |
| M5 | B7 v5 migration + E2 FreeCAD target | **DONE**: B7 — benchmark defaults to the v5 --json entry, Satine parses one JSON line (no stdout scraping), Onshape upload single-sourced in `cad_v5/targets.py` (66 dup lines gone), results carry target URLs/errors. E2 v1 — box-grammar parametric `.FCStd`, verified by STEP re-export (E2 v2 = cylinder parts/blind holes/fillets, open) |
| M6 | Track D fine-tune | fine-tuned 7B ≥ 7/10 full suite; 30B pin retired |
| M7–M10 | superseded — see `DIRECTION.md` Part 4 | N1–N3 interaction/reliability → CAM print+laser → mesh backend → CNC; each item carries its own exit criterion |

Standing rule: every milestone ends with a full honest benchmark run committed to
`benchmarks/results/`, compared against `baseline_*.json`. No claimed improvement without a
measured delta.

## 9. Addendum — coder-model refresh (2026-07-16)

- **Rung 1 — SETTLED (measured, `run_20260716_*.json`): qwen3:8b keeps the rung.**
  3/6 conv, 15/22 acc, 173s/build vs granite4:7b-a1b-h 2/6, 10/22, 224s (80 tok/s raw — fast
  tokens, slow parts: convergence rate dominates wall time), granite3.3:8b 0/6, 7/22, and
  qwen3:4b 0/6, 0/22 (persistent unterminated-string syntax errors, fails even a cube).
- **Strong rung — BLOCKED on the Ollama 0.20.5→0.32.x upgrade:** the Q3_K_M GGUF (the only
  quant that fits the RAM ceiling) downloads but fails model-create with `Error: 400` on the
  old runtime; blobs cached, retry instant post-upgrade. Baseline to beat: 30B 4/6, 15/22, 699s.
- **Suite:** decisions use tiers 1–2 + the new `heldout-cqe` suite; held-out specs never enter
  the few-shot corpus, so retrieval can't flatter the numbers.
- Deferred: qwen3.6 MTP variant (post-upgrade), Laguna XS 2.1 at q3.

---

## Sources

- Zoo — Zookeeper conversational CAD agent: https://zoo.dev/research/zookeeper · Text-to-CAD/ML-ephant: https://zoo.dev/machine-learning-api
- CADCodeVerify (VLM question-based verification, ICLR 2025): https://arxiv.org/abs/2410.05340
- CADSmith (multi-agent + programmatic geometric validation): https://arxiv.org/pdf/2603.26512
- CADDesigner / ECIP (explicit-context CAD API for LLMs; build123d Pass@1 finding): https://arxiv.org/abs/2508.01031
- Text2CAD 660k dataset (NeurIPS 2024): https://sadilkhan.github.io/text2cad-project/
- CADmium (fine-tuning code LLMs for CAD): https://arxiv.org/pdf/2507.09792
- Text2CAD-Bench: https://arxiv.org/pdf/2605.18430 · Zero-to-CAD (synthetic agentic data): https://arxiv.org/pdf/2604.24479
- Obico AI failure detection (photo→defect precedent): https://www.obico.io/blog/ai-failure-detection-in-3d-printing/

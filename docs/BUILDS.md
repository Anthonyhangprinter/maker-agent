# CAD Agent — Build Verification List

Verification targets for the build123d agentic loop (v4.3 engine). The primary suite is the
`earthtojake/text-to-cad` benchmarks (MIT), reproduced in `benchmarks/text-to-cad/specs.json` and
run by `scripts/run_benchmarks.py`. There is no separate "plan" step in v4 — the agent plans,
builds, inspects, and self-critiques in one loop, so just `build` and check `converged`.

Mark each line: ⬜ untested | ✅ converged | ⚠️ built but not converged | ❌ failed

---

## Quick smoke tests (Tier 0 — should converge fast on the 7B)

| Spec | Expect | Status |
|---|---|---|
| `a solid cylinder 60mm diameter 80mm tall` | one Cylinder | ⬜ |
| `a 100x60x20mm plate with a 10mm hole in the centre` | Box − Cylinder | ⬜ |
| `a 100x60x30mm enclosure with 2mm walls, open top` | hollow box | ✅ (verified 2026-06-03) |
| `a 100x60x30mm box with four 4mm holes through the top near the corners` | Box − 4 Cyl | ✅ (verified 2026-06-03) |
| `W200x100 I-beam 1500mm` | `structural_section()` helper, bypasses critic | ⬜ |
| `spur gear 20 teeth module 2 width 15mm` | `spur_gear()` helper | ⬜ |
| `M12 hex bolt 80mm long` | `hex_bolt()` helper | ⬜ |

---

## Benchmark suite (text-to-cad) — `python3 scripts/run_benchmarks.py`

Tier 1 = simple, 2 = moderate, 3 = hard (mainly exercise the 30B-escalation path on 8GB VRAM).
Full specs in `benchmarks/text-to-cad/specs.json`. Results land in `benchmarks/results/`.

Status from the 2026-07-04 honest baselines (`benchmarks/results/baseline_30b_full.json` = pinned
30B; `baseline_7b_tiers12.json` = `--coder fast` on tiers 1–2):

| # | Benchmark | Tier | Key challenge | 30B (acc) | 7B (acc) |
|---|---|---|---|---|---|
| 01 | Rectangular calibration block | 1 | 4 through-holes + top-edge chamfer | ✅ 3/4 | ✅ 4/4 |
| 02 | Circular flange | 1 | central bore + 6-hole bolt circle + edge fillet | ✅ 3/4 | ✅ 2/4 |
| 03 | L-bracket | 1 | two plates + gussets + holes + fillet | ✅ 2/4 | ✅ 2/4 |
| 04 | Stepped shaft with keyway | 2 | 3 coaxial steps along X + keyway slot + chamfers | ✅ 3/3 | ❌ 0/3 |
| 05 | Open-top enclosure with bosses | 2 | hollow shell + 4 standoffs + blind holes + corner fillets | ✅ 3/3 | ✅ 3/3 |
| 06 | Clevis bracket w/ lightening cutouts | 2 | symmetric lugs + clevis hole + ribs + many fillets | ❌ 0/4 | ⚠️ 2/4 |
| 07 | Radial engine cylinder | 3 | 12 cooling fins pattern + flange + angled spark-plug boss | ⚠️ 2/3 | — |
| 08 | Centrifugal impeller | 3 | 12 backward-curved swept blades + fillets (very hard) | ✅ 3/4 | — |
| 09 | Spiral staircase | 3 | 20 rotated treads + helical handrail + balusters | ✅ 1/1 | — |
| 10 | Planetary gear stage | 3 | sun + 3 planets + internal-tooth ring + carrier | ⚠️ 1/1 | — |

Headline (baselines, pre-bug-fix engine): 30B full suite **7/10 converged, 21/31 acceptance (68%)**;
7B tiers 1–2 **4/6, 13/22 (59%)** at 3–6× the speed. (⚠️ = geometry produced but not converged.)

### Current measured numbers (2026-07-06, stricter honest engine)

| Run | Config | Suite | Converged | Acceptance | Result json |
|---|---|---|---|---|---|
| M4.1 | `--coder fast` (7B) | tiers 1–2 (6) | **4/6** | 13/22 (59%) | `m41_7b_tiers12.json` |
| Showcase | `--coder strong` (30B) | all 10 | **6/10** | 22/31 (71%) | `showcase_30b_full.json` |
| M7 run 1 (2026-07-09) | `--coder fast` (7B), post-N1/N2/N3 engine | tiers 1–2 (6) | 2/6 | 7/22 (32%) | `m7_7b_tiers12_run1.json` |
| M7 run 2 (2026-07-10) | same engine, variance re-run | tiers 1–2 (6) | **5/6** | 15/22 (68%) | `m7_7b_tiers12_run2.json` |
| M8 OpenSCAD spike (2026-07-10) | `scad_agent.py`, 7B, no few-shots | tiers 1–2 (6) | **0/6** | 0/22 (0%) | `m8_scad_7b_tiers12.json` |
| M10 organic (2026-07-10) | `--suite organic --coder auto` (all → 30B) | organic (5) | **2/5** | 6/13 (46%) | `m10_organic_auto.json` |

The showcase's 6/10 vs the baseline 7/10 reflects the stricter honest gate shipped in M4.1, not a
regression (acceptance rose 21→22). Artifacts: `benchmarks/results/artifacts/20260706_085550_strong/`.

The M7 pair is the honest variance picture at n=6 with a temperature-sampled 7B: the two runs
bracket the 4/6 baseline (failures are single-category loops — the 7B repeats one mistake all
turns — not engine crashes; run 1's four `rc=1` rows are the pre-existing raise-on-zero-geometry
path). N1's inline retries fired only on builds that ultimately failed anyway; converging builds
needed ≤2 turns with zero autofixes. Per-row `turns` + `n1_autofixes` are recorded from these runs
onward.

```bash
SCRIPT=~/.openclaw/skills/cad-builder
python3 $SCRIPT/scripts/run_benchmarks.py                 # all 10, --coder auto
python3 $SCRIPT/scripts/run_benchmarks.py --tiers 1,2     # the achievable ones
python3 $SCRIPT/scripts/run_benchmarks.py --only 01,02,03 # specific ids
python3 $SCRIPT/scripts/run_benchmarks.py --coder strong  # force the 30B coder
```

The runner records, per benchmark: `converged`, `code_model`, wall time, and re-derived
`volume_mm3` / `faces` / `bbox_mm` from the produced STEP, to `benchmarks/results/run_<ts>.json`
and `latest.json`.

### Organic mini-benchmark (M10) — `python3 scripts/run_benchmarks.py --suite organic`

First non-prismatic domain, on **build123d** (the M8-measured winner — the OpenSCAD backend
went 0/6, `SCAD_SPIKE.md`). Five specs in `benchmarks/organic/specs.json` +
`acceptance.json`: smoothly-varying revolves, twisted lofts, wavy 2D boundaries, and
perforation patterns. New acceptance criterion `min_faces` (total B-rep face count ≥ N) is the
deterministic proxy for "the perforation pattern actually exists" where hex/diamond cutouts
are invisible to the cylindrical-face hole detector (see the acceptance `_meta` block).

**Measurement note (recorded honestly):** two verified gold examples teaching the o1/o2
idioms (sine-profile revolve; twisted loft) were seeded into `~/.openclaw/cad-examples.jsonl`
as part of M10 — the mini-benchmark therefore measures the system WITH those idioms
retrievable, by design (M10 = expansion on measured footing). `--no-fewshots` remains the A/B
switch to measure the lift.

First run (2026-07-10, `--coder auto`, `m10_organic_auto.json`): **2/5 converged, 4/5
geometry, 6/13 acceptance (46%)**, 58 min — the complexity triage routed every organic spec to
the 30B, correctly. `min_faces` earned its place immediately: it caught the critic accepting
two under-perforated parts (o4/o5 converged with the pattern only partially cut). The seeded
gold idioms did not rescue o1 (sine vase produced no geometry) — organic revolves remain the
domain's hard edge. This is the honest baseline the domain improves against.

| # | Benchmark | Tier | Key challenge | Result (acc) |
|---|---|---|---|---|
| o1 | Sine-wave vase | 1 | revolve a sampled sine profile, hollow w/ offset revolve | ❌ 0/2 (no geometry) |
| o2 | Twisted hex vase | 2 | loft rotated hex sections, hollow w/ inner offset stack | ⚠️ 0/2 (2 solids, 227mm tall) |
| o3 | Scalloped coaster | 1 | sinusoidal polar boundary + blind recess | ⚠️ 2/3 (bbox ✓ recess ✓, 2 solids) |
| o4 | Hex-perforated grille | 2 | honeycomb hole pattern inside a solid border | ✅ 2/3 (min_faces ✗ — pattern under-cut) |
| o5 | Diamond lampshade | 2 | staggered diamond cutouts through a curved wall | ✅ 2/3 (min_faces ✗ — pattern under-cut) |

---

## Rating good builds (grows the few-shot store)

```bash
python3 ~/.openclaw/skills/cad-builder/cad_agent_v4.py rate 5 "clean geometry, matches spec"
```

≥4★ builds are appended to `~/.openclaw/cad-examples.jsonl` (deduped by spec) and injected as
"SIMILAR SUCCESSFUL BUILDS" few-shots into the brief for similar future specs. Tier 1–2 benchmark
builds that converge cleanly are good candidates to rate up.

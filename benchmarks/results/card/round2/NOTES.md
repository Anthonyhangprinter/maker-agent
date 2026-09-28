# Maker Agent round 2: running notes (2026-09-23 to 26)

## Data
| Source | Pairs | Stock Gemma solves | Hard (Gemma fails) | Spend |
|---|---|---|---|---|
| Owner references (18 SolidWorks parts) | held out, eval only | 1/18 first try | - | - |
| Opus 5.5 pilot (code-first) | 31 | - | - | $4.78 |
| Batch 1 (800 seeds, Batch API) | 443 | 81% | 90 | $39.41 |
| Batch 2 (600 seeds, Gemma-fail + 3D families) | 201 | 62% | 76 | $33.09 |
| **Total** | **673** | 76% | **166** | ~$77 |

Method: teacher designs a part as build123d code, it is built and measured, a spec is written
from the measurements, a blind rebuild from that spec alone must match (band=match, clean gate,
equal hole/fillet/chamfer counts). Verified by construction.

## Owner review (FreeCAD overlays)
- Batch 1 pilot: 8 reviewed, all sensible; 3 missing corner fillets -> measurer fixed, keep rule tightened.
- Batch 1 scale: 12 reviewed, 11 good (ratchet pawl missed a corner cut; accepted as tolerable).
- Batch 2: 15 reviewed, 15 good.
- Scorer calibrated to owner labels on the 18 references: agreement 13/24 -> 17/24.

## Findings
- Specs that under-determine the part were the root problem: Opus 5.5 7/18 vs Gemma 1/18 on the
  owner parts once specs were fully determined (the earlier "teacher not worth it" was an artefact).
- Stock Gemma already solves most teacher data; the value is in the misses (hard-example mining).
- Gemma's misses are mostly crashes (112), and over half are mechanical: `.z` vs `.Z` (~31),
  missing helper imports (~31), wrong helper signatures (~15).

## Next (in progress)
1. Engine fixes: API-casing normaliser + helper auto-import + build123d API docs retrieval, each
   A/B-measured on Gemma's 166 hard pairs.
2. Round 2 fine-tune on the 673 pairs (hard ones weighted up), prompts matching the fixed engine.
3. Ship/no-ship vs stock on public 85 + owner 18 (held out). Rule: beat stock by more than
   paired-flip noise with no validity regression.

## Engine fixes, measured 2026-09-26 (commit 76f8cda, no training)
| Set | Stock | + normalise | + normalise + API docs |
|---|---|---|---|
| 166 pairs stock Gemma failed | 0 match+valid (112 crash) | 87 | **101** (22 crash) |
| 60 random pairs stock solved | 60 | 52 | 56 (re-sampling noise; 0 of 561 solved codes are changed by the patch) |
- Offline replay of the 112 crashes: 63 now match/valid with the normaliser alone, 0 regressions on 561 non-crash builds.
- API docs vs none: paired flips +42/-24 (sign test p~0.04), ~200-270 extra prompt tokens when it fires.
- Found: one pair (cfb20363) hung the gate 40+ min on a 54 MB STL; needs a scorer timeout.
- Consequence: the ship test must compare fine-tune+fixes vs stock+fixes (the fixes are credited separately).

## Owner review, batch 3 (2026-09-26)
- 15 pairs across 15 families reviewed in FreeCAD: owner verdict "these are tough, nice work" (all accepted).
- Future families requested: bearings and similar multi-body/standard mechanical components (assemblies; the gate already supports multi-part via bd_warehouse).
- Verified examples are being added to the web UI as an Examples gallery (inspiration + demo to peers).

## Step 1: fixed-engine Gemma baseline, full live pool (2026-09-27)

Owner approved training on the full verified set (35 sampled pairs reviewed, all good) --
no `--approved-ids` gate this round. Live pool grew since this file's "Data" table above (a
batch-2 budget-priority rebuild kept 41 more pairs mid-run, and batch 3's 443 pairs are
counted here for the first time): **1,164 kept pairs** across scale (443) + pilot (29) +
batch2 (241) + batch2/pilot (8) + batch3 (443). Ran `lab/gemma_baseline.py` with fixes ON
(defaults: `CAD_NORMALISE=1 CAD_API_REF=1`, commit 76f8cda) over every pair, reusing 226
rows already baselined this way (`~/lab-scratch/normalise-2026-09-26/step4.jsonl`) plus 938
new ones -- `benchmarks/results/card/round2/gemma_fixed_baseline.jsonl` (gitignored,
per-build artefacts too, ~446MB of build dirs). Zero `score_timeout` crashes this run (the
new hard scorer timeout from step 0 never fired -- no pathological mesh in this batch).

| | match+valid | / n | % |
|---|---|---|---|
| **Overall** | 965 | 1164 | 82.9% |
| tier 1 | 153 | 153 | 100.0% |
| tier 2 | 463 | 546 | 84.8% |
| tier 3 | 270 | 341 | 79.2% |
| tier 4 | 79 | 124 | 63.7% |
| codefirst-scale-2026-09-25 | 400 | 443 | 90.3% |
| codefirst-pilot-2026-09-24 | 23 | 29 | 79.3% |
| codefirst-batch2-2026-09-25 | 195 | 241 | 80.9% |
| codefirst-batch2-2026-09-25/pilot | 3 | 8 | 37.5% |
| codefirst-batch3-2026-09-26 | 344 | 443 | 77.7% |

Bands overall: match 853, valid 112, near_miss 134, fail 17, crash 48. The 199 rows NOT
match/valid (near_miss + fail + crash) are round 2's oversampling target (3x weighting, see
`lab/compile_codefirst.py`'s `oversample()`). Tier confirms the expected gradient (tier 1
trivially solved, tier 4 the genuine hard tail) -- consistent with the fixes already having
closed most of the EASY misses (mechanical crashes) the pre-fix 2026-09-26 measurement
found, leaving geometry-level near-misses as the dominant remaining failure mode on tiers
3-4.

## Steps 2-5: compile, train, ship, eval -- verdict NO-SHIP (2026-09-27 to 28)

**Compile.** `lab/compile_codefirst.py` over all 5 pair sources (1,164 pairs, 0 dropped),
notes/API-reference block now injected into the rendered prompt to match serve time (a
train/serve mismatch this round fixed -- verified a rendered row contains the
`API REFERENCE` block when relevant). Weighting changed to 3x any row whose fixed-Gemma
band is NOT match/valid, 1x the rest (previously any non-"match" row, which wrongly
included "valid" builds). Before weighting: train 1106 / val 58 (frozen 5% spec-level
split), 187 hard train rows. After 3x weighting: **train 1480**, val unchanged at 58.

**Train.** `lab/round1.sh --data lab/rounds/round2 --epochs 1` (epochs dropped from round
1's 2: 1480 rows is ~6.1x round 1's 242, and the hard tail already gets 3x exposure per
epoch via oversampling). Rank 16/alpha 16/lr 1e-4/max-seq 5120 unchanged. 1409/1480 train
rows kept (71 dropped over max-seq), 54/58 val rows kept. 353 steps, 9h9m wall, peak VRAM
21.7GB. **train_loss 0.1899, eval_loss 0.1563** (single epoch, computed once at the end).

**Ship.** Merged, converted, quantized (Q4_K_M + Unsloth imatrix), registered as
`gemma-4-31b-cad-r2` (`benchmarks/arms.json`, `role: "comparison"`, never the default --
`~/.openclaw/cad.json` untouched). `ship.py verify` failed its first three attempts with a
false negative: its throwaway smoke server hardcodes port 8093, which collides with the
always-on `gpu-notice.service` (Casa AI's maintenance-notice stub) -- every failing attempt
was actually talking to the notice stub, not the model. A manual diagnostic on a free port
proved the arm writes correct `from build123d import *` code; re-verifying with
`--port 8096` passed cleanly (cube/bracket/text all ok).

**Eval.** Three-arm comparison (stock-no-fixes / stock-with-fixes / r2-with-fixes) on round
1's own 85 public specs (cadprompt/text2cadquery/heldout-cqe, phase1 subset) plus the 18
owner references, one sample each, run inside a single continuous `gpu_window.sh` hold
(materialize -> build r2 -> build stock+fixes -> build stock-no-fixes -> score, all in one
process) so no other CAD frontend could slip a build in and evict the maker arm mid-run.
Found and fixed a real bug in `lab/eval_round.py` along the way: `MAIN_CHECKOUT` pointed
one directory too high for `run_card.py --suite-root` (which replaces `BENCH` wholesale),
so the very first attempt silently built 0 public specs for every arm (n=0 tables, no
error) while the owner-refs-18 rows built fine. Fixed the constant, re-ran ONLY the missing
public-85 builds for all 3 arm-configs (owner-refs rows reused untouched), and added a hard
row-count sanity gate (exactly 85 public / 18 owner per arm, `SystemExit` otherwise) before
ever scoring again.

### r2+fixes vs stock+fixes (the ship/no-ship table, round 1's rule)

| suite | arm | n | invalid | match | flips (invalid) | flips (match) |
|---|---|---|---|---|---|---|
| public-85 | gemma-4-31b (baseline) | 85 | 8.2% | 31.8% | - | - |
| public-85 | gemma-4-31b-cad-r2 | 85 | 11.8% | 28.2% | +2/-5 | +2/-5 |
| owner-refs-18 | gemma-4-31b (baseline) | 18 | 50.0% | 11.1% | - | - |
| owner-refs-18 | gemma-4-31b-cad-r2 | 18 | 38.9% | 22.2% | +3/-1 | +2/-0 |

**Verdict: NO-SHIP.** r2 genuinely improves the harder, hand-verified owner set (invalid
50%->38.9%, match doubles 11.1%->22.2%, flips net positive both ways) -- exactly the hard
tail this round targeted -- but regresses on the easier public-85 suite (invalid
8.2%->11.8%, match 31.8%->28.2%, flips net negative both ways). Round 1's rule requires no
regression on BOTH tables, so this fails it. `~/.openclaw/cad.json` stays on stock
`gemma-4-31b`; `gemma-4-31b-cad-r2` is registered for comparison only.

### Engine-fix gain alone (stock+fixes vs stock-no-fixes, same two suites)

| suite | arm | n | invalid | match | flips (invalid) |
|---|---|---|---|---|---|
| public-85 | gemma-4-31b-nofix (baseline) | 85 | 9% | 34% | - |
| public-85 | gemma-4-31b (fixes on) | 85 | 8% | 32% | +6/-5 |
| owner-refs-18 | gemma-4-31b-nofix (baseline) | 18 | 39% | 17% | - |
| owner-refs-18 | gemma-4-31b (fixes on) | 18 | 50% | 11% | +1/-3 |

Roughly a wash to slightly negative on both suites at n=85/18 -- consistent with the
2026-09-26 finding that the fixes matter most on the genuinely hard tail (166 pairs stock
Gemma failed outright), not on suites where stock already solves most specs one-shot; not
a contradiction of that finding, just a different, easier population.

Full tables: `DECISION.md` (the invalid first attempt is kept at `DECISION.invalid-0135.md`
for the record), `LIFT.md`/`OWNER-REFS-LIFT.md`, `FIXGAIN-LIFT.md`/
`FIXGAIN-OWNER-REFS-LIFT.md`.

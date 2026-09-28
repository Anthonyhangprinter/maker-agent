# Maker Agent 1.0 findings, for Andrew

Status: final, 2026-09-22. Tag v1.0 on master.

## 1. Summary

Maker Agent 1.0 set out to fine-tune a CAD-coding model on a local RTX 3090: shootout a base
model, measure which agent-loop levers help it, prove a local QLoRA pipeline end to end, then
feed it enough verified data to beat the stock base. We built and measured all of that, and we ran
the fine-tune. The pipeline works: a 31B QLoRA fine-tune trains, merges, converts and serves on
this card with no rented compute. Trained on 242 of its own verified parts it came out level with
the stock model (match 34.1 % vs 34.9 %, paired flips +2/-2) and slightly less valid (5.9 % vs
2.4 % invalid, flips +0/-3), so it was not promoted. That is a cleaner result than Phase 2, where
older data made the model worse, but it is not a gain. The single most important finding: the
bottleneck is not the model, the language, or the training mechanics, it is verified-good CAD
data. Our best local coder gets fewer than half of hard (tier 3) parts gate-clean on one attempt,
and even among the ones the gate calls "good," a hand check found roughly a third wrong. A Phase
2 fine-tune on our older 7B-era pairs made the base model measurably worse, because that data was
below the model's own quality. Phase 3 built the harvest that produces
better data and measured its limit: 281 verified pairs in 22 GPU-hours, but only 33 of them hard
parts, and the model's own agreement with itself was wrong three times out of four on hard parts.

## 2. What we built

A benchmark card over public and internal CAD-text suites with geometry-aware scoring; a
swappable CAD model server (`maker-server`, one GGUF at a time, conflicts with the household chat
model); a local QLoRA train/merge/quantise/serve pipeline (`lab/`, own venv, one 3090); a
gate-verified harvest that samples, verifies and grades confirmation strength; 301 Claude-rebuilt
reference parts for cross-model checking; a hardened GPU window script; and two doors on the GPU
proxy so family chat gets a "busy until HH:MM" notice instead of a timeout during a harvest.

| Piece | Key measured numbers |
|---|---|
| Benchmark card | reproduced BenchCAD's published Gemma Code-QA score (0.663 vs 0.664) |
| Maker server (:8088) | one GPU, conflicts with the resident chat model by design |
| QLoRA pipeline | 21.7 GB peak VRAM, 78 s/step, 1 h 54 min/epoch over 348 rows |
| Gate-verified harvest | 2,514-spec bank, 281 good pairs, 12.1 good pairs/GPU-hour over 22.3 GPU-hours |
| Reference intake | 301 of 414 teacher specs admitted under today's gate |
| GPU busy doors | human door answers in about 7 ms with an ETA during a unit |
| Round 1 fine-tune | 242 rows, lr 1e-4, 2 epochs, 2 h 56 min, eval loss 0.148; ship chain 63 min; not promoted |
| Design assistant | vague request to an editable parameter panel before build, plus live sliders after; shipped, lift not yet measured |

## 3. Findings

**Phase 0, base model.** Gemma-4-31B-it won the seven-arm shootout: 3% invalid versus
Qwen3.8-27B's 18%, 42 matches versus 23. Caveat: one-shot, thinking off, no critic, so this
measures the base model, not the loop; BenchCAD favours the 27B on two of three tasks, a note
about edit-style training data, not a reason to change the pick.

**Phase 1, agent-lift levers.** Retrieval matters, stays on (off cost 7 of 85 specs). Best-of-3
and the full critic loop gave no geometry lift over one-shot at 2.4x-4.2x the time; one-shot
stays default. Thinking on gave a small lift (45% vs 40% match) at 5x time, 7x tokens: an
escalation option, not a default. The coder judging its own render beat the old Ollama critic on
match and time, so that critic was dropped. Caveat: n is 40-85, one sample per lever; read the
paired flips, not the percentages.

**Phase 2, training spike.** Pipeline GO, model NO-SHIP. The adapter, trained on 353 pairs from
the old 7B-era pipeline (short, idiom-heavy, masked dimensions), made Gemma write shorter,
faster, measurably worse code (invalid 12% vs 2%, match 24% vs 35%, flips +2/-11).

**Claude vs Gemma, 20 specs.** After one repair, all four models (Gemma, Sonnet, Opus, Fable)
build 19 or 20 of 20: no reliability gap. On exact match Claude leads 6 to 3, a lead all three
Claude models agree on, arguing against luck but not against shared lineage. Noise floor: two
Gemma runs on the same 20 specs disagreed on 2 of 7. The harvest stays local: Claude's edge only
shows where a reference proves it, and metered cost (10k-14k tokens/spec) rules it out at scale.

**Hard-part hit rate.** On teacher-suite tier 3 specs (written and solvable by stronger models),
the gate calls about 35% of one-shot attempts gate-clean; a hand check of 7 of those found 2
wrong (a lip cutter shearing the whole top off instead of a rim; a vertical cylinder standing in
for a horizontal through-hole). On tier 3 specs the model wrote for itself, one attended unit
produced zero pairs from 3 specs: it cannot yet set and pass its own hard homework.

**Same-model agreement.** When the harvest confirms a pair by two same-spec samples agreeing
(`same_pass`), the programs are usually the same one written twice: code similarity measured
0.66-0.94. That is evidence against sampling noise, not against a shared misconception, so every
pair records `confirm_strength` (`reference`/`cross_pass`/`same_pass`) plus code similarity, and
the audit decides which strengths are trainable.

**References.** 301 of 414 teacher specs now have a Claude-built reference passing today's gate;
as of Sunday night only 51 of those (about one in six) were solved and confirmed (the live table
below has a more recent total but no unique-spec breakdown). Separately, 19 of 321 (about 6%) of
the August Claude-teacher parts the old gate accepted fail today's stricter gate.

**Language A/B.** Across two models and three usable languages, exact match stayed in a
10%-17.5% band: Gemma+build123d 15% (no few-shots 15%), Gemma+CadQuery 12.5% (42.5% of attempts
fail on API misuse), Qwen(resident)+build123d 10% (no few-shots 17.5%), Qwen+CadQuery 17.5%.
OpenSCAD always compiled but produced no STEP file. The Qwen OpenSCAD row (32.5% built) is not
real: a harness fault returned an empty reply for 17 of 40 specs. Match barely moves across
languages, so spec interpretation is the bottleneck, not the language.

**Qwen as judge.** Precision of a "correct" verdict was 30.8% (text) and 36.0% (with a render)
against a 27% base rate: barely better than guessing. Caveat: the label was "exact match," a
stricter bar than "valid part," so some "wrong" verdicts may be reasonable readings. On three
fully dimensioned fixtures it was sharp, naming the exact error on two known-wrong parts, but it
also flagged the third (labelled correct) for a real floor-thickness discrepancy, so our own
label may need a second look.

## 4. Defects the process caught

- OCCT's mesh writer resets the process locale to ASCII mid-run, breaking UTF-8 decoding of child
  output in three tools; fixed with explicit decoding everywhere.
- Fluid mode's production gate never sets the expected solid count and silently accepts an
  unmeasured candidate; caught in the harvest's draft code, not yet fixed live.
- The harvest's first draft graded a candidate good whenever inspection raised an exception
  instead of failing closed; fixed before any pair was trusted.
- A stop signal mid-batch was swallowed by a generic exception handler, so the generator kept
  running after being told to stop.
- The GPU window's kill escalation hit the wrong process, and its exit handler could restore the
  CAD arm instead of the chat model after an abort; fixed with group signalling.
- A scheduler meant to escalate hard specs to thinking-on units instead locked onto thinking-only
  units for hours, starving normal sampling.
- The gate misread "a hole at each end" as one hole and vetoed correct multi-hole parts.
- A reviewer's test called the model-serving path unpatched and silently swapped the household
  chat model out for 1 h 43 min; the rule that followed: such tests must patch the model-call
  helpers, and status checks read one service at a time.

## 5. What is not done

No fine-tuned model is in production: round 1 was trained, shipped and benchmarked and did not
beat stock, so `cad.json` stays on stock Gemma-4-31B and the round-1 arm is kept for comparison.
Fluid mode (the web UI's default path) still runs the weaker gate described above; the harvest
uses its own stricter gate, and the port back is a v2 item because changing the live gate would
make the benchmark cards non-comparable. The design assistant is deployed but its on/off lift on
vague requests is unmeasured (about 2 GPU-hours plus a blind pick by the owner). The one
long-standing test failure (`tests/test_n1_offline.py`) is unchanged. Everything else is merged to
master and tagged v1.0.

## 5a. Hand audit and round 1

The audit rebuilt 32 sampled pairs on CPU (re-measured geometry matched the stored numbers in all
32) and read each against its spec with volume arithmetic and section renders.

| Confirmation grade | Wrong / audited |
|---|---|
| Matched a Claude-built reference | 0 / 10 |
| No-think and think samples agreed | 0 / 4 |
| Two same-pass samples agreed, tiers 1 and 2 | 1 / 10 |
| Two same-pass samples agreed, tiers 3 and 4 | 6 / 8 |

The six wrong hard parts were all no-op features that a dimensional gate cannot see: a gasket
groove and an opening cut inside already-empty space, keyways centred on the bore axis so they
remove no material, two cooling jackets with no water gap, a lead screw whose thread was never
modelled. In each case both samples shared the mistake, which is the predicted failure of
same-model agreement. The audit also overturned one of our own labels: the "correct" V066 divider
part has a 4.5 mm floor where the spec says 3 mm, which the Qwen judge had said and we had not
believed. Same-pass tier 3-4 pairs were excluded wholesale; the rest compiled to 242 train and 21
validation rows from 11 held-out specs.

| Round 1 vs stock, 85 public specs, one shot | Stock Gemma-4-31B | Round 1 |
|---|---|---|
| Invalid | 2.4 % | 5.9 % (flips +0/-3) |
| Gate-clean | 89.4 % | 88.2 % |
| Match vs reference | 34.9 % | 34.1 % (flips +2/-2) |
| Median time, output tokens | 31 s, 542 | 33 s, 522 |

Confounds are the same as Phase 2's: the round-1 GGUF is Q4_K_M with the Unsloth imatrix while
stock is UD-Q4_K_XL, the baseline rows are reused from the Phase 1 card, and n is 85 with one
sample per arm, so the paired flips are the evidence. Full numbers:
`benchmarks/results/card/round1/DECISION.md`.

## 6. Version 2 plan

Bring in public CAD-code datasets with a real lineage (DeepCAD, Text2CAD, Text-to-CadQuery,
CAD-Recode), held out against every card suite by today's contamination guard. Use the owner's
own CAD models, plus scraped or service-generated meshes, as geometry confirmers the way Claude
references are used now, since a dimensioned reference is our strongest verifier. Add
documentation retrieval aimed at the build123d API-misuse failures the A/B surfaced (it will not
fix spec interpretation, the real ceiling). Explore a cross-model rejector-style judge on fully
dimensioned specs, where Qwen was sharp despite being unreliable generally. Measure the design
assistant shipped in 1.0 (a vague request becomes an editable parameter panel before codegen,
with the user's own numbers locked): the literature (Self-planning, ClarifyGPT, TICoder) shows
real gains from clarify-before-code in code generation generally, but no clean CAD ablation
exists, so an on/off measurement on vague prompts would be a new result.

## 7. Reproduce

```
# Benchmark card (subset used for the Phase 1/2 lift tables)
PYTHONUTF8=1 python3 scripts/run_card.py --subset phase1 --arms gemma-4-31b

# Harvest status (read-only, live numbers)
PYTHONUTF8=1 python3 lab/harvest.py --status

# Language A/B (must run inside a GPU window)
lab/gpu_window.sh python3 benchmarks/lang-ab/lang_ab.py --arms b123d,b123d-nofs,cadquery,openscad --specs 40 --run-id <id>

# Judge eval
cd benchmarks/judge-eval && python3 run_judge.py && python3 report.py
```

## Harvest numbers (final, measurement ended 2026-09-21 07:00 AEST)

| metric | value |
|---|---|
| Spec bank | 2,514 (tier 1/2/3/4 = 71/632/1,576/235) |
| Run | Sat 23:07 to Mon 07:00, 61 units, 1,739 candidates, 22.3 GPU-hours |
| Good pairs (gate-clean and confirmed) | 281 |
| Good by tier | 97 / 151 / 29 / 4 (tier 3-4 share 11.7 %) |
| Good by confirmation grade | reference 101, cross-pass 4, same-pass 176 |
| Yield | 12.1 good pairs per GPU-hour overall; 12.2 before the gate fix, 4.4 during the think-pass starvation, 14.0 after the scheduler fix |
| Reference pool | 51 of 301 reference specs solved by Sunday night; the pool was spent, so the run was not extended |
| Left over | 107 unconfirmed candidates, 21 specs with disagreeing samples, 353 cold specs, 3 exhausted |

## Postscript: Round 2 (2026-09-24/28), NO-SHIP

This report's own "Next steps" (owner reference geometry as a confirmer, API-misuse
documentation retrieval, measuring vague-prompt lift) is exactly what round 2 tried. Result:
1,164 code-first verified teacher pairs (~$120), an engine normaliser + API-reference fix that
recovered 101/166 of Gemma's own hard teacher-pair failures with no training, and a QLoRA
fine-tune (`gemma-4-31b-cad-r2`) that came in level with stock on both public-85 and the 18
owner references, same result as round 1. `cad.json` stays on stock `gemma-4-31b`. The
bottleneck has moved from data volume to verification and eval coverage: 1,164 pairs did not
move the needle where only 18 real held-out parts and under-specified public suites exist to
measure against. Full numbers: `docs/PROJECT.md`'s 2026-09-24/28 section and
`benchmarks/results/card/round2/DECISION.md`.

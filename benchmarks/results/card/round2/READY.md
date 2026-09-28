# Round 2: ready-to-run command sequence (NOT executed -- prep only, 2026-09-25)

Round 2a (training on the raw codefirst-scale + codefirst-pilot pool, 472 teacher-verified
pairs) was cancelled by the owner before any GPU time was spent: `gemma_baseline.jsonl`
(live at compile time, 345 of the 472 ids baselined) already put stock `gemma-4-31b` at
288/345 = 83.5% band=="match" on that pool (crash 23, valid 19, near_miss 13, fail 2 make
up the rest) -- training on a pool the stock coder already solves ~90% of the time (owner's
number, in the same range once the baseline finishes) is a predictable no-ship, the same
finding round 1 already made at a smaller scale (round1/DECISION.md: "the limit is the
number of confirmed HARD parts the harvest can produce"). The owner wants harder pairs,
verified by them, first.

This file is the exact command sequence for round 2 once that owner review exists: an
`approved_ids.json` naming which codefirst pairs actually train. Nothing below has been
run. What HAS been run (CPU only, no GPU, no model calls) is recorded at the bottom under
"What was actually verified today".

## Precondition

`benchmarks/results/card/round2/approved_ids.json` must exist: either a bare JSON list of
pair ids (`["cfs0031", "cfs0142", ...]`, ids as they appear in the codefirst pairs.jsonl
files) or `{"approved_ids": [...]}`. Compiling without it (`--approved-ids` omitted) is
informational only -- see `lab/compile_codefirst.py`'s own module docstring -- and is not
what should train.

## 1. Compile (CPU)

```bash
python3 lab/compile_codefirst.py \
    --pairs benchmarks/results/card/codefirst-scale-2026-09-25/claude-opus-5-5/pairs.jsonl \
    --pairs benchmarks/results/card/codefirst-pilot-2026-09-24/claude-opus-5-5/pairs.jsonl \
    --gemma-baseline benchmarks/results/card/codefirst-scale-2026-09-25/gemma_baseline.jsonl \
    --approved-ids benchmarks/results/card/round2/approved_ids.json \
    --oversample-gemma-fail 2 \
    --out-dir lab/rounds/round2
```

Writes `lab/rounds/round2/{train.jsonl, val.jsonl, val_specs.json, data_meta.json}`. Run it
with `--dry-run` first and read `data_meta.json`'s `counts`/`dropped_by_reason` before the
real compile -- same habit `lab/compile.py`'s own README section models.

Knobs and why:
- `--oversample-gemma-fail 2`: doubles a TRAIN row whose gemma_baseline band != "match"
  (stock genuinely failed it), never applied to val. Chosen because the whole point of
  round 2 is the hard tail stock does not already solve; 2x is the same factor the
  original round-2a brief asked for, kept here as the default rather than re-derived,
  since nothing about the owner's approval step changes what "hard" means for a pair that
  does get approved. Re-tune only if the approved pool's gemma-fail share turns out to be
  a small minority (little to gain) or a large majority (2x may already be too gentle) --
  one line of justification in this file if changed.
- `--exclude-gemma-match-tier1` NOT passed by default: the approved-ids gate already does
  the real filtering (the owner reviews and picks hard pairs); this flag is available if a
  later approved-ids list still lets through easy tier-1 rows stock already matches and a
  second, automatic backstop is wanted.
- Owner references (18, `~/CAD/references/`) and every card suite are held out
  unconditionally regardless of approved-ids -- see the script's own docstring.
- `--val-frac 0.05` (default): spec-level, tier-stratified, frozen once
  `lab/rounds/round2/val_specs.json` is first written (a real, non-dry-run compile) -- a
  later re-compile with a wider approved-ids list reuses the same split, same reasoning as
  round 1's own frozen `val_specs.json`.

Read `data_meta.json`'s `counts.train.total` before training: if it is well under round
1's 242 train rows, training may not be worth a ~3-hour GPU window at all -- say so instead
of running it. If it is far larger (many hundreds), see the epoch note below.

## 2. Train (GPU, inside a window)

Hyperparameters: **unchanged from round 1** (`lab/round1.sh`'s own defaults) --
rank 16, alpha 16 (`lora_alpha=rank`), lr 1e-4, epochs 2, max-seq 5120, save-steps 25,
batch 1 x grad-accum 4, language layers only, adamw_8bit, bf16.

Justification for keeping them: round 1 (242 rows, this exact preset) was level with
stock and did NOT damage the model -- the round 1 verdict's failure mode was the DATA
(mostly tier 1-2, self-verified, nothing hard), not the optimizer setup. Round 2's pairs
are teacher-verified (Claude Opus, geometry-checked against a blind rebuild) rather than
self-verified, and oversampled toward the harder ones stock fails, which argues for
running the same preset again and reading whether ROW QUALITY moves the needle round 1's
row COUNT could not, rather than changing two variables (data and hyperparameters) in the
same experiment. Only reconsider if `lab/rounds/round2/data_meta.json`'s
`counts.train.total` lands far outside round 1's order of magnitude (roughly 100-400):
notably smaller and 2 epochs on so little data risks overfitting sooner, notably larger
(many hundreds to low thousands, plausible once several approval rounds accumulate) and 1
epoch may already be enough -- either way, change ONE knob and write the one-line reason
in this file before launching, per the standing "plain plan paragraph" rule.

```bash
mkdir -p ~/lab-scratch
GPU_WINDOW_MAX_SEC=28800 setsid nohup lab/round1.sh --data lab/rounds/round2 \
    > ~/lab-scratch/round2-train.log 2>&1 &
```

`lab/round1.sh` is reused unmodified: it already forwards `--data` (and any other
`train.py` flag) verbatim and only uses its own `--data`/`--out` parsing for the printed
wall-time estimate, so `--data lab/rounds/round2` is the only override needed. It prints
the adapter output path (`$HOME/lab-scratch/round1-adapter-<timestamp>/adapter` -- the
script's own `$OUT` naming is a leftover of round 1's name, not a bug to fix here: it does
not affect where round 2's data comes from) and logs to a fresh
`~/lab-scratch/round1-train-<timestamp>-<pid>.log` in addition to the redirect above.
`GPU_WINDOW_MAX_SEC=28800` (8h) is generous headroom over round 1's measured 2h56m; raise
it further only if `data_meta.json` shows several times round 1's row count.

`setsid nohup ... &` detaches the job into its own session so it survives the launching
shell exiting -- the same launch shape `lab/README.md`'s own specgen.py section uses for
an unattended multi-hour job. `lab/gpu_window.sh` (which `round1.sh` calls) already queues
for the CAD build lock, evicts the resident, and restores it on any exit path including a
clean SIGTERM -- see its own README section, in particular "the SIGKILL limitation", before
ever sending `kill -9` to this job. To stop it cleanly:

```bash
cat ~/.openclaw/cad-build.lock        # {"pid": ..., "child_pid": N, "pgid": N, ...}
kill -TERM -- -N                      # N = the pgid from the lock file, never -9
```

## 3. Ship the adapter as a GGUF arm (mixed CPU/GPU, per `lab/README.md`'s own round-1
   sequencing -- each intermediate deleted as soon as the next stage has consumed it,
   since the root SSD cannot hold the bf16 base + merged copy + F16 GGUF at once)

```bash
ADAPTER=~/lab-scratch/round1-adapter-<timestamp>/adapter   # printed by step 2's launch

# merge: adapter + bf16 base -> merged HF dir (CPU, ~33 min at round 1's size)
lab/.venv/bin/python lab/ship.py merge --adapter "$ADAPTER"

# convert: merged HF dir -> F16 GGUF (CPU, ~39 min)
lab/.venv/bin/python lab/ship.py convert --model-name gemma-4-31b-cad-r2

rm -rf ~/lab-scratch/merged                                  # reclaim ~62GB before quantizing

# imatrix + quantize straight onto NVMe (CPU, imatrix seconds, quantize ~19-21 min)
lab/.venv/bin/python lab/ship.py fetch-imatrix
mkdir -p ~/lab-scratch/rounds/round2
lab/.venv/bin/python lab/ship.py quantize \
    --out ~/lab-scratch/rounds/round2/gemma-4-31b-cad-r2-Q4_K_M.gguf

rm -f ~/lab-scratch/gemma-4-31b-cad-F16.gguf*                 # reclaim ~62GB

# verify: GPU step -- MUST run inside a window (~1 min)
lab/gpu_window.sh lab/.venv/bin/python lab/ship.py verify \
    --gguf ~/lab-scratch/rounds/round2/gemma-4-31b-cad-r2-Q4_K_M.gguf

# register: refuses without a passing verify.ok. role "comparison", NEVER the default arm --
# do not edit ~/.openclaw/cad.json.
lab/.venv/bin/python lab/ship.py register \
    --gguf ~/lab-scratch/rounds/round2/gemma-4-31b-cad-r2-Q4_K_M.gguf \
    --name gemma-4-31b-cad-r2 --adapter "$ADAPTER" \
    --store ~/lab-scratch/rounds/round2-store
```

After `register`, hand-edit `benchmarks/arms.json`'s new `gemma-4-31b-cad-r2` entry to
carry `"role": "comparison"` if `ship.py register` does not already set one (check against
how `gemma-4-31b-cad-r1` is declared there) -- the same rule round 1 followed: never the
default arm until a card actually clears the ship bar.

## 4. Evaluate vs stock and write DECISION.md (GPU for the builds, CPU for the scoring)

```bash
python3 lab/eval_round.py --arm gemma-4-31b-cad-r2 --baseline gemma-4-31b \
    --out-dir benchmarks/results/card/round2
```

One command: builds both arms on round 1's own 85 public specs (`cadprompt`,
`text2cadquery`, `heldout-cqe`, `--subset phase1 --mode oneshot`, read from the main
checkout via `--suite-root` since this worktree carries no public-suite `specs.json` of
its own) plus the 18 owner references (`benchmarks/owner-refs/`, materialised from
`~/CAD/references/` on the same call), scores every build with
`scripts/geom_bands.py`'s `score_against_reference` via `scripts/lift_report.py`'s
`lift_table`, and writes `LIFT.json`/`LIFT.md` (public, byte-identical in shape to round
1's own), `OWNER-REFS-LIFT.json`/`.md` (the 18-reference table), and `DECISION.md` with
match %, invalid %, paired flips for both, plus a SHIP/NO-SHIP verdict by round 1's own
rule applied to BOTH tables (see `lab/eval_round.py`'s `_verdict()`).

## What was actually verified today (2026-09-25, CPU only, no GPU, no model calls)

- `lab/compile_codefirst.py --dry-run` against the two real pairs files + the live
  `gemma_baseline.jsonl` (345 of 472 ids baselined at read time): 472 pairs loaded, 0
  dropped (no owner-reference collisions, no card-suite contamination, no duplicate ids),
  18/472 val (5%, tier-stratified), 504 train rows at `--oversample-gemma-fail 2` (449
  without oversampling) -- see the task report for the full breakdown.
- `--approved-ids`/`--exclude-gemma-match-tier1` composition smoke-tested with a 3-id
  fixture: correctly cut the pool to the one approved id whose gemma band was not a
  tier-1 match.
- `lab/eval_round.py --materialize-only`: built `benchmarks/owner-refs/` (18 specs, 0
  skipped) from `~/CAD/references/`, idempotent on a second run (0 materialized, 18
  reused).
- `scripts/run_card.py`'s own `load_suite("owner-refs")` + `apply_subset(..., "phase1")` +
  `contamination(...)` against the new suite: 18 specs load, uncapped by the phase1
  subset (as intended -- "owner-refs" is not a `SUBSETS` key), zero contamination clashes
  against the training corpora.
- `lab/eval_round.py --score-existing`: `geom_bands.score_against_reference` against a
  real reference (an owner reference's own `model.step` vs its materialised `.stl`,
  `band: near_miss`, chamfer 0.18mm -- the STEP->STL round trip is not bit-identical, so
  not "match", but close, as expected) and against an unrelated candidate (a codefirst
  build STEP scored against a different reference, `band: fail`, chamfer 8.8mm) --
  discriminates correctly in both directions.
- `lab/eval_round.py --score-only` against a synthetic `rows.jsonl` (both suites, both
  arms, no real builds): `LIFT.json`/`OWNER-REFS-LIFT.json`/`DECISION.md` generated
  correctly, verdict computed and matched the planted (worse) synthetic data (`NO-SHIP`,
  both tables' match-flip reasons named correctly).
- No GPU job was started, no arm was switched, the resident was never touched.

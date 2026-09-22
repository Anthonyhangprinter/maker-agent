# Round 1 numbers (2026-09-21)

Adapter: `~/lab-scratch/round1-adapter-20260921-092522/adapter`. Arm: `gemma-4-31b-cad-r1`
(role `round1` in `benchmarks/arms.json`). Base: `gemma-4-31b` (stock `gemma-4-31B-it`
UD-Q4_K_XL).

## Training set composition (`lab/rounds/round1/data_meta.json`)

Command: `python3 lab/compile.py --round 1 --strengths reference,cross_pass,same_pass --exclude-ids lab/rounds/round1/exclude_ids.txt --out-dir lab/rounds/round1`

- Strengths: `cross_pass`, `reference`, `same_pass`. `max_per_spec`: 2. `val_frac`: 0.08. `seed`: 1.
- Gate version: `gv1-2c11a923` (263 rows seen at this version).
- `rows_total_seen`: 494. `dropped_total`: 231 (`kind-not-good`: 213, `excluded-by-audit`: 18).
- `val_specs_count`: 11 (frozen, tier-stratified).
- Rows over `max_seq` 5120: train 0, val 0.

Train rows: 242 total.

| tier | strength | rows |
|---|---|---|
| 1 | reference | 22 |
| 1 | same_pass | 68 |
| 2 | reference | 59 |
| 2 | same_pass | 76 |
| 2 | cross_pass | 2 |
| 3 | reference | 15 |

Val rows: 21 total.

| tier | strength | rows |
|---|---|---|
| 1 | reference | 1 |
| 1 | same_pass | 6 |
| 2 | reference | 4 |
| 2 | same_pass | 8 |
| 3 | cross_pass | 2 |

## Training run (`train_meta.json`)

- Base checkpoint: `/mnt/nvme-apps/LinuxModels/gemma-4-31B-it-unsloth-bnb-4bit`.
- Hyperparameters: rank 16, alpha 16 (`lora_alpha=rank`), epochs 2, lr 1e-4, max-seq 5120,
  save-steps 25, batch 1 x accum 4, language layers only.
- Dataset actually used by the trainer: train_kept 242, train_dropped 0, val_kept 20, val_dropped 1.
- Trainable parameters: language 122,429,440, vision 0.
- Peak VRAM: 21.588 GB.
- Wall time: 10,577.05 s (2 h 56 min 17 s). Steps: 122.
- Train loss: 0.1070387452291172.
- Eval loss (completion-only): 0.14786268608086442, over 6,801 tokens.
- Versions: torch 2.12.1+cu130, unsloth 2026.9.5, transformers 5.5.0, trl 0.24.0, peft 0.21.0, bitsandbytes 0.50.2.

## Ship timings and sizes (`~/lab-scratch/ship_log.jsonl`)

| step | elapsed | output |
|---|---|---|
| merge | 1157.98 s (19.3 min) | `~/lab-scratch/merged` |
| convert | 1356.24 s (22.6 min) | `~/lab-scratch/gemma-4-31b-cad-F16.gguf` |
| quantize | 1276.20 s (21.3 min) | `~/lab-scratch/rounds/round1/gemma-4-31b-cad-r1-Q4_K_M.gguf` |
| verify | 57.52 s | 3/3 prompts passed |
| register | 0.0 s (copy already present) | `~/lab-scratch/rounds/round1-store/gemma-4-31b-cad-r1-Q4_K_M.gguf` |
| clean | 0.0 s | reclaimed 115.5 GB (merged dir + F16 GGUF) |

Final quantized GGUF: 18,687,064,160 bytes = 17.404 GiB = 18.687 GB (stock UD-Q4_K_XL is 18.82 GB).

Verify tok/s (smoke numbers, `-ngl 99 -c 8192 -np 1`, not the arm's serving speed): cube 29.36,
bracket 28.46, text 29.00.

## Card (`benchmarks/results/card/round1/card.json` meta)

Mode oneshot, subset phase1, arms `gemma-4-31b-cad-r1`, 127 builds. Started
2026-09-21T14:30:48, finished 2026-09-21T16:24:45 (1 h 53 min 57 s).

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-cad-r1 | cad-arena | 12 | 11 | 8% | 9 | 78% | 0 | 0 | 0 | 0 | 28 | 6353 | 1 |
| gemma-4-31b-cad-r1 | cadprompt | 30 | 30 | 0% | 30 | 83% | 16 | 5 | 6 | 3 | 31 | 11967 | 0 |
| gemma-4-31b-cad-r1 | hard-eval | 15 | 11 | 27% | 5 | - | 0 | 0 | 0 | 0 | 89 | 36955 | 0 |
| gemma-4-31b-cad-r1 | heldout-cqe | 25 | 21 | 16% | 17 | 72% | 13 | 3 | 4 | 1 | 39 | 18838 | 0 |
| gemma-4-31b-cad-r1 | organic | 5 | 3 | 40% | 1 | 31% | 0 | 0 | 0 | 0 | 76 | 6825 | 0 |
| gemma-4-31b-cad-r1 | text-to-cad | 10 | 5 | 50% | 4 | 48% | 0 | 0 | 0 | 0 | 92 | 14664 | 0 |
| gemma-4-31b-cad-r1 | text2cadquery | 30 | 29 | 3% | 28 | 80% | 0 | 2 | 19 | 8 | 32 | 13605 | 0 |
| gemma-4-31b (baseline, reused from the Phase 1 card) | cad-arena | 12 | 12 | 0% | 10 | 78% | 0 | 0 | 0 | 0 | 28 | 6097 | 1 |
| gemma-4-31b | cadprompt | 30 | 30 | 0% | 30 | 83% | 15 | 7 | 6 | 2 | 30 | 13832 | 0 |
| gemma-4-31b | hard-eval | 15 | 12 | 20% | 8 | - | 0 | 0 | 0 | 0 | 78 | 34154 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 17 | 68% | 14 | 2 | 6 | 1 | 39 | 19133 | 0 |
| gemma-4-31b | organic | 5 | 4 | 20% | 1 | 85% | 0 | 0 | 0 | 0 | 72 | 6539 | 0 |
| gemma-4-31b | text-to-cad | 10 | 6 | 40% | 4 | 52% | 0 | 0 | 0 | 0 | 96 | 15884 | 0 |
| gemma-4-31b | text2cadquery | 30 | 30 | 0% | 29 | 83% | 0 | 2 | 18 | 10 | 28 | 13125 | 0 |

## Lift vs stock (`LIFT.json`, public suites only: cadprompt, text2cadquery, heldout-cqe, n=85)

| arm | n | ref n | invalid | Δ invalid | gate clean | Δ | acceptance | Δ | match | Δ | flips (invalid) | flips (match) | median s | Δ | tokens/build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b (baseline) | 85 | 83 | 2.35% | - | 89.41% | - | 76.36% | - | 34.94% | - | +0/-0 | +0/-0 | 31.1 | - | 542.24 |
| gemma-4-31b-cad-r1 | 85 | 85 | 5.88% | +3.53 pts | 88.24% | -1.18 pts | 77.27% | +0.91 pts | 34.12% | -0.82 pts | +0/-3 | +2/-2 | 33.2 | +2.1 s | 522.47 |

## Confounds

1. `gemma-4-31b-cad-r1` is quantized plain Q4_K_M with the Unsloth importance matrix; the
   baseline `gemma-4-31b` is the stock Unsloth UD-Q4_K_XL dynamic quant. A quant-recipe
   difference, not the adapter itself, same as Phase 2's confound 1.
2. The 85 baseline (`gemma-4-31b`) rows are reused from the Phase 1 card (same arm, same
   one-shot code path, same subset) rather than re-run alongside round 1's own card.
3. n = 85 public specs, one sample per arm: the paired flips columns (+0/-3 invalid, +2/-2
   match) are the evidence at this sample size, not the percentages.
4. The LoRA delta was learned against nf4-dequantised weights and merged into bf16, same as
   Phase 2.
5. Round 1's training set (242 train / 21 val rows, `reference`+`cross_pass`+`same_pass`
   strengths, harvested from the maker arm itself under the current gate) has different
   provenance than the Phase 2 spike's 353 7B-era pairs.

## Verdict (owner/controller)

Do not promote. Round 1 is level with stock on match (34.1 % vs 34.9 %, paired flips +2/-2)
and slightly worse on validity (5.9 % vs 2.4 % invalid, flips +0/-3). Unlike the Phase 2 spike
it did not damage the model, but about 240 mostly tier 1-2 self-verified pairs at lr 1e-4 for
2 epochs do not move a 31B model on this card. cad.json stays on the stock gemma-4-31b arm, and
the round-1 arm is registered for comparison only. The limit is the number of confirmed HARD
parts the harvest can produce (33 of 281 pairs at tier 3-4, and same-model agreement was 75 %
wrong there), not the training pipeline.

# Phase 2 decision: training spike on Gemma-4-31B (2026-09-17)

## Verdict

**Pipeline: GO.** A 31B QLoRA fine-tune trains, merges, converts, quantises and serves on this box, locally, with no offload and no rented GPU.

**Spike model: NO-SHIP.** The adapter trained on the existing 353 gate-verified pairs makes Gemma-4-31B measurably worse on the public suites. `cad.json` keeps the stock `gemma-4-31b` arm. The spike arm `gemma-4-31b-cad-spike` stays registered for comparison only.

The spike measured the pipeline, not the data. The data is now the bottleneck, which is exactly what Phase 3 (data engine) exists for.

## Measured (RTX 3090, 24 GB, 64 GB RAM, all local)

| step | result |
|---|---|
| fit | Unsloth QLoRA on the pre-quantised 4-bit checkpoint, r 16, language layers only (122.4M trainable, vision 0), 5120-token window, batch 1 x accum 4: peak VRAM 21.7 GB, no offload needed |
| speed | 78 s per optimizer step (4 rows), 182 tokens/s; one epoch over 348 rows = 87 steps = 1 h 54 min |
| loss | train 0.358 (mean of per-step batch means), completion-only eval 0.391 over 6,973 tokens (the same eval read 0.953 on the 3-step smoke adapter; no untrained-base eval was run, the smoke adapter is the nearest proxy since LoRA B starts at zero) |
| merge | streaming per-tensor LoRA merge into the bf16 base, 1188 tensors / 410 deltas, 20 to 33 min, under 12 GB RAM (a full-model load needs 62.5 GB and cannot fit) |
| convert | llama.cpp Gemma 4 converter, F16 GGUF 61.4 GB, 31 to 39 min |
| quantise | Q4_K_M with the Unsloth importance matrix, 18.69 GB (stock UD-Q4_K_XL is 18.82 GB), 19 to 26 min |
| serve | llama-server with the stock vision projector, three-prompt verify passed, 28 to 30 tok/s on the verify flags (the arm itself is served with the maker-server flags; not the same number) |
| card | 127 builds, one-shot, phase1 subset, 1 h 25 min |
| disk | root SSD peak about 185 GB (bf16 base 59 + merged 62.5 + F16 61.4); the bf16 base is kept at `~/lab-scratch/gemma-4-31B-it-bf16` for later rounds |

## Lift vs the stock arm (public suites, like for like, 85 specs)

| arm | invalid | gate clean | acceptance | match | flips (match) | median s | tokens/build |
|---|---|---|---|---|---|---|---|
| gemma-4-31b (stock) | 2% | 89% | 76% | 35% | | 31 | 542 |
| gemma-4-31b-cad-spike | 12% (+9) | 79% (-11) | 71% (-5) | 24% (-11) | +2 / -11 | 24 (-7) | 343 |

Per suite (ok builds / geometry matches): CADPrompt 25 vs 30 ok, 12 vs 15 matches; heldout-cqe 21 vs 23 ok, 8 vs 14 matches; Text-to-CadQuery 29 vs 30 ok; internal suites flat to slightly worse (text-to-cad 5 vs 6, organic 2 vs 4, hard-eval 11 vs 12).

What the fine-tune did: outputs got about 40% shorter (343 vs 542 tokens) and faster, and the extra failures are build123d API misuse (`Plane.sketch`, `Polyline.make_face`, `RegularPolygon` kwargs, `Arc` undefined, a `GroupBy` attribute) plus more near-misses on geometry. That is the signature of the training set: 353 pairs written for and by the Qwen-era 7B pipeline (short, idiom-heavy scripts against a `<DIM>`-masked reference block), one epoch at lr 2e-4 on a model that already wrote correct build123d first try. The model moved toward the data, and the data is below the model.

Confounds, stated: (1) the spike is Q4_K_M + imatrix, the baseline is the stock UD-Q4_K_XL (a quant difference, not expected to explain an 11-point match drop); (2) the 127 baseline rows are reused from the Phase 1 card (same arm, same one-shot code path, same subset); (3) the LoRA delta was learned against nf4-dequantised weights and merged into bf16; (4) n = 85 public specs, one sample per arm, so read the paired flips (+2 / -11) rather than the percentages.

## What this changes for Phase 3 and 4

1. **Data before training.** Phase 3 must produce pairs that are above the model: rejection sampling from Gemma itself (expert iteration) on new specs, gate-verified and reference-scored where a reference exists, contamination-guarded against every card suite with the exact-key and per-suite near-duplicate rule, with the old 7B-era pairs dropped or re-verified under the Phase 0 gate. Training on the current corpus again would repeat this result.
2. **Round budget (measured):** about 2 h per epoch per 350 rows, so a 2,000-pair round is roughly 11 h of training plus 1.5 h of ship chain plus 1.4 h of card. One round fits an overnight window; three fixed rounds (Phase 4) are three nights.
3. **Recipe knobs to A/B in Phase 4, not now:** lower LR (5e-5 to 1e-4), fewer target modules (attention only), rank 8, and an untrained-base eval as the true baseline for the eval loss.
4. **Ship rule:** an arm is promoted only when its card beats the stock arm on the public suites by the section 6 rule (invalid ratio first, then match share) with the flips column in its favour. The spike fails that rule and stays unpromoted.

## Provenance

`benchmarks/results/card/phase2/spike/`: `train_meta.json` and `train_meta.jsonl` (per-step VRAM, loss, tokens/s), `adapter_config.json`, the smoke run's meta, and `provenance.json` (sha256 of the source and rendered datasets, adapter, chat template, Q4 GGUF and imatrix; train row ids; the ship log; git head). The spike trained on the `c9e78e8` version of `lab/train.py`; later commits changed no numerics (re-rendering the dataset after the fix round was byte-identical). Adapter: `lab/runs/spike1/adapter` (git-ignored). GGUF: `/mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf`.

## Follow-ups parked from the final review (all Low)

`register --force` also skips the verify gate (split the flags); the gpu_window holder JSON does not escape control characters; `QUANTIZE_DISK_FACTOR` 0.35 vs measured 0.30; the merge template canary checks the adapter's saved template rather than the exact `--template` data.py used (write a `data_meta.json`); `run_card.contamination` extracts specs fail-open per row while `lab/data.py` fails closed; the verify tok/s is a smoke number on different server flags; `_wait_health` duplicates `arms.py:_wait`; the shipped GGUF's `general.name` reads "Merged" (re-convert with `--model-name` next time).

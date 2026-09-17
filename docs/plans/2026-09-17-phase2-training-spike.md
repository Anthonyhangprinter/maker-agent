# Phase 2 "Training spike" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, with numbers, that Gemma-4-31B can be QLoRA fine-tuned on this RTX 3090 and shipped back as a llama.cpp GGUF the maker server can serve, or produce a documented no-go.

**Architecture:** A new `lab/` package holds the training side: `lab/train.py` (Unsloth QLoRA on the pre-quantised 4-bit checkpoint, vision tower frozen, checkpoint every N steps, resume), `lab/data.py` (the existing 353/16 ChatML pairs re-rendered with the Gemma 4 non-thinking template, contamination-guarded), and `lab/ship.py` (merge the adapter into the true bf16 base, convert to GGUF with llama.cpp's Gemma 4 converter, quantise Q4_K_M with the stock importance matrix, reuse the stock mmproj, register the file as a maker arm). Training runs under the same eviction discipline as the card: `lab/gpu_window.sh` holds the CAD build lock, stops the resident, runs one bounded job, restores. The spike model is then measured on the Phase 1 baseline subset with `run_card.py` so the go/no-go is a like-for-like number.

**Tech Stack:** Unsloth (pip, 2026.6+ with Gemma 4 support) in `lab/.venv` (uv, Python 3.12, torch 2.11 cu130 already on the box), bitsandbytes, TRL/PEFT, llama.cpp CUDA build at `~/llama.cpp-cuda-src` (`conversion/` package + `llama-quantize`), systemd user units, pytest.

**Spec:** `docs/MAKER-1.0-CAMPAIGN.md` sections 4.4, 5 (Phase 2 row), 6, 8; decisions `benchmarks/results/card/phase0/DECISION.md`, `phase1/DECISION.md`; research record in the 2026-09-17 session.

## Global Constraints

- All local. RunPod is not on the ladder; the offload ladder is: plain Unsloth QLoRA at 4k ctx, then 2k ctx, then Unsloth activation offload (already on), then DeepSpeed ZeRO-3 CPU offload with `bnb_4bit_quant_storage_dtype="bfloat16"`.
- One GPU: every training or conversion job runs inside `lab/gpu_window.sh` (build lock held, resident stopped, maker stopped, restored in a trap). Never two GPU jobs at once. The family health watcher already treats a held build lock as a deliberate eviction.
- Disk: permanent model files on `/mnt/nvme-apps/LinuxModels/` (about 61GB free until the owner deletes the losing GGUFs); TEMPORARY merge and convert intermediates (about 125GB) go on the root SSD under `~/lab-scratch/` (211GB free) and are deleted by `ship.py` after the quantised GGUF is verified. Ruling recorded in the ledger.
- Contamination: `lab/data.py` refuses any training row whose spec matches a card suite via `harvest_census.suite_keys()` + the near-duplicate rule.
- Never `pkill -f` a pattern in your own command line; long jobs launch from a script file with `setsid nohup`; no git stash/checkout during runs; no em dashes in user-facing copy; commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; tests `python3 -m pytest tests/ -q` keep the single pre-existing failure only.
- The spike trains on the EXISTING 353 pairs (Qwen-era teacher data, gate-verified). It is a feasibility run, not the Phase 4 training; no new data is generated in Phase 2.

---

## File map

| File | Responsibility |
|---|---|
| `lab/__init__.py`, `lab/README.md` | package + how to run the spike |
| `lab/gpu_window.sh` (create) | hold `~/.openclaw/cad-build.lock` (flock), stop `qwen38-server` and `maker-server`, run `"$@"`, restore in a trap; refuses if VRAM > 1.5GB used after stops |
| `lab/data.py` (create) | `render_pairs(src_jsonl, out_dir, template="gemma-4")` -> `train.jsonl`/`val.jsonl` of `{"text": ...}` rows rendered with the Gemma 4 non-thinking template; contamination refusal; stats (rows, tokens p50/p95) |
| `lab/train.py` (create) | Unsloth QLoRA: `--base`, `--data`, `--out`, `--rank`, `--epochs`, `--max-seq`, `--save-steps`, `--resume`; logs VRAM peak, tokens/s, s/step; writes `train_meta.json` |
| `lab/ship.py` (create) | `merge` (adapter + bf16 base -> merged dir on scratch), `convert` (llama.cpp -> F16 GGUF), `quantize` (Q4_K_M with `--imatrix`), `verify` (llama-server smoke on :8093, 3 prompts), `register` (copy GGUF to the store, add arm `gemma-4-31b-cad-spike` to arms.json, reuse stock mmproj), `clean` (delete scratch) |
| `benchmarks/arms.json` (modify) | the spike arm |
| `tests/test_lab_data.py`, `tests/test_lab_ship.py` | offline tests (template rendering, contamination refusal, argv construction, size/path checks with stubs) |
| `benchmarks/results/card/phase2/` (generated) | spike card + `DECISION.md` (go/no-go) |

Disk plan (verified sizes): 4-bit checkpoint 19.1GB (NVMe, permanent); adapter <1GB; bf16 base 62.5GB (scratch, deleted after merge); merged bf16 62.5GB (scratch); F16 GGUF about 60GB (scratch); Q4_K_M 18.3GB (NVMe, permanent); mmproj reused (1.2GB, already present).

---

### Task 1: lab venv and gpu_window

**Files:** `lab/README.md`, `lab/gpu_window.sh`, `lab/.venv` (uv, git-ignored), `.gitignore`

- [ ] `uv venv lab/.venv --python 3.12 && uv pip install --python lab/.venv/bin/python --upgrade "unsloth" unsloth_zoo bitsandbytes transformers accelerate trl peft datasets` (torch: reuse the system cu130 wheel if uv resolves it; otherwise `torch==2.11.*` cu130 from the PyTorch index). Record `pip show unsloth` version in README (must be >= 2026.6.7 for Gemma 4).
- [ ] `lab/gpu_window.sh`:

```bash
#!/usr/bin/env bash
# One bounded GPU job with the resident and maker evicted; restores both states in a trap.
set -euo pipefail
LOCK="$HOME/.openclaw/cad-build.lock"
exec 9>"$LOCK"; flock -w 3600 9 || { echo "gpu_window: build lock busy for 1h" >&2; exit 3; }
echo "{\"pid\": $$, \"frontend\": \"lab\", \"spec\": \"$*\", \"started\": \"$(date -Is)\"}" >&9
MAKER_WAS=$(systemctl --user is-active maker-server || true)
restore() { systemctl --user stop maker-server 2>/dev/null || true; systemctl --user start qwen38-server || true; }
trap restore EXIT
systemctl --user stop maker-server 2>/dev/null || true; systemctl --user stop qwen38-server || true
for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 2; done
[ "$used" -lt 1500 ] || { echo "gpu_window: VRAM still ${used} MiB after eviction" >&2; exit 4; }
echo "== gpu_window: start $(date +%T) :: $*"; "$@"; echo "== gpu_window: done $(date +%T)"
```

- [ ] Test by hand (30 s of GPU): `lab/gpu_window.sh nvidia-smi --query-gpu=memory.used --format=csv` shows < 1500 MiB inside, and the resident is active after. Commit `lab: venv + gpu_window`.

### Task 2: data rendering

**Files:** `lab/data.py`, `tests/test_lab_data.py`

- [ ] `render_pairs`: read `~/.openclaw/cad-sft-train.jsonl` / `cad-sft-val.jsonl` (ChatML `messages`), drop any row whose user spec hits `harvest_census.suite_keys()` or the near-duplicate slug rule (reuse `run_card.contamination` logic via import), render with `unsloth.chat_templates.get_chat_template(tokenizer, "gemma-4")` when the venv is available, else a pure-Python renderer of the same `<|turn>role\n...<turn|>` framing for tests; write `{"text": ...}` JSONL; print row counts, dropped ids, token p50/p95 (tokenizer from the 4-bit checkpoint when present, whitespace tokens otherwise).
- [ ] Tests: framing exact (`<|turn>user\n...<turn|>\n<|turn>model\n...<turn|>`), contamination refusal drops a planted card spec, stats shape.
- [ ] Run it for real once the checkpoint is down (Task 3) and record counts. Commit.

### Task 3: checkpoint download + train.py + the spike run

**Files:** `lab/train.py`, `lab/spike.sh`

- [ ] Download `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (19.1GB, ungated) to `/mnt/nvme-apps/LinuxModels/gemma-4-31B-it-unsloth-bnb-4bit/` with `hf download`.
- [ ] `lab/train.py` (Unsloth): `FastModel.from_pretrained(base, load_in_4bit=True, max_seq_length=args.max_seq, dtype=None)`; `FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", target_modules="all-linear", use_gradient_checkpointing="unsloth")`; TRL `SFTTrainer` with `per_device_train_batch_size=1, gradient_accumulation_steps=4, num_train_epochs=args.epochs, learning_rate=2e-4, bf16=True, save_steps=args.save_steps, resume_from_checkpoint=args.resume`; a callback logging `torch.cuda.max_memory_allocated()`, tokens/s and s/step every 10 steps to `train_meta.json`; assert at start that the LoRA touches language layers only and prints trainable parameter counts per module type.
- [ ] `lab/spike.sh`: `lab/gpu_window.sh lab/.venv/bin/python lab/train.py --base <4bit dir> --data lab/data/ --out lab/runs/spike1 --rank 16 --epochs 1 --max-seq 4096 --save-steps 25`; launch with `setsid nohup`, log to `lab/runs/spike1/train.log`.
- [ ] Offload ladder if OOM at 4096: rerun at `--max-seq 2048`; if still OOM, add `--zero3-offload` (accelerate config with `zero_stage 3`, `offload_param_device cpu`, `offload_optimizer_device cpu`, `bnb_4bit_quant_storage_dtype bfloat16`) and record the slowdown.
- [ ] Exit numbers to record: peak VRAM, s/step, tokens/s, wall time for one epoch over 353 rows, final train/val loss. Commit code (not runs).

### Task 4: ship.py (merge, convert, quantise, verify, register)

**Files:** `lab/ship.py`, `tests/test_lab_ship.py`, `benchmarks/arms.json`

- [ ] `merge`: download `google/gemma-4-31B-it` bf16 (62.5GB, ungated) to `~/lab-scratch/gemma-4-31B-it-bf16/` (root SSD, temporary), load in bf16 on CPU with PEFT, `merge_and_unload()`, save to `~/lab-scratch/merged/` (CPU RAM 64GB: load with `low_cpu_mem_usage=True` and `device_map={"": "cpu"}`; if it does not fit, fall back to Unsloth `save_pretrained_merged(save_method="merged_16bit")` from the 4-bit base and record that the merge is against dequantised weights).
- [ ] `convert`: `python ~/llama.cpp-cuda-src/convert_hf_to_gguf.py ~/lab-scratch/merged --outfile ~/lab-scratch/gemma-4-31b-cad-F16.gguf --outtype f16` (the Gemma4 converter is in the tree's `conversion/` package; no `--mmproj` run needed).
- [ ] `quantize`: fetch `imatrix_unsloth.gguf_file` from `unsloth/gemma-4-31B-it-GGUF`; `llama-quantize --imatrix <it> <F16> <out Q4_K_M> Q4_K_M`; output to `/mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf`.
- [ ] `verify`: start a throwaway `llama-server` on :8093 inside `gpu_window` with the new GGUF + the stock `mmproj-F16.gguf`, three prompts (a cube, an L-bracket, an image critic call), check a solid builds through `scripts/fluid_gen.py` with `CAD_CRITIC_MODEL`/`LOCAL_CODER_URL` pointed at :8093 via env (or simply register the arm and run `arms.py use`).
- [ ] `register`: add arm `gemma-4-31b-cad-spike` (same ctx/extra_args as `gemma-4-31b`, mmproj = the stock file) to `benchmarks/arms.json`; `clean` removes `~/lab-scratch/*` after `verify` passes.
- [ ] Tests with stubs: argv construction for each step, refusal to `clean` before `verify`, arm entry shape.

### Task 5: measure the spike and decide

- [ ] `run_card.py --arms gemma-4-31b,gemma-4-31b-cad-spike --subset phase1 --mode oneshot --out benchmarks/results/card/phase2 --allow-contaminated` then `lift_report.py benchmarks/results/card/phase2 --baseline gemma-4-31b`.
- [ ] `benchmarks/results/card/phase2/DECISION.md`: go/no-go per spec section 5 Phase 2 exit: peak VRAM, hours per epoch, offload route used (if any), merge route (bf16 vs dequantised), GGUF size, serving tok/s vs stock, the spike's lift table (expected near zero or slightly negative: the data is small and Qwen-era; the point is the pipeline), and the Phase 3/4 plan implications (rounds of about 2,000 pairs, hours per round).
- [ ] Optional side-spike, only if time allows: Gemma MTP for serving speed (`mtp-gemma-4-31B-it.gguf` from the stock repo with `--spec-type draft-mtp`), measured tok/s on the stock arm.

---

## Self-review

- Spec 4.4 (train.py, ship.py, offload ladder, MTP measure): Tasks 3, 4, 5. MTP for Gemma is optional because the stock arm runs without it today.
- Spec 5 Phase 2 exit (go/no-go with numbers; offload route recorded): Task 5.
- Spec 8 risks: fit (Task 3 ladder), untrained DeltaNet layers (not applicable: Gemma is standard attention; Task 3 prints per-module trainable counts anyway), MTP miscalibration (Gemma serves without MTP), llama.cpp LoRA path (not used; merge then requantise), contamination (Task 2), GPU contention (gpu_window).
- Disk ruling: temporary intermediates on the root SSD; permanent files on the NVMe.

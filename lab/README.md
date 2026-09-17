# Phase 2 training lab

Own `uv` venv for the Maker Agent training spike (QLoRA fine-tune of Gemma-4-31B), kept separate
from the rest of the repo's tooling so training deps (torch, unsloth, bitsandbytes, trl, peft)
never collide with the CAD engine's runtime deps or the system torch.

## Venv

Path: `lab/.venv` (git-ignored, along with `lab/runs/` and `lab/data/`, which are job output, not
source).

Created with:

```bash
uv venv lab/.venv --python 3.12
uv pip install --python lab/.venv/bin/python --upgrade \
    "unsloth" unsloth_zoo bitsandbytes transformers accelerate trl peft datasets
```

`uv` resolved its own torch (a cu13x PyPI wheel, not the system's cu130 build at
`/usr/bin/python3.12`'s `2.11.0+cu130`) because `unsloth`/`bitsandbytes`/`xformers` need matching
wheels for each other; letting `uv` pick avoided a hand-pinned mismatch. Verified working:

```bash
lab/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# True 13.0
```

### Resolved versions (2026-09-17, install log: `~/lab-scratch/uv_install_1.log`)

| package | version |
|---|---|
| torch | 2.12.1+cu130 |
| unsloth | 2026.9.5 |
| unsloth-zoo | 2026.9.4 |
| transformers | 5.5.0 |
| trl | 0.24.0 |
| peft | 0.21.0 |
| bitsandbytes | 0.50.2 |
| xformers | 0.0.35 |
| triton | 3.7.1 |
| accelerate | 1.15.0 |
| datasets | 4.3.0 |

`unsloth` 2026.9.5 is above the 2026.6.7 floor needed for Gemma 4 support, so no
`--upgrade --prerelease=allow` / GitHub-main install was required.

## gpu_window.sh

One bounded GPU job with the resident (`qwen38-server`) and the CAD maker arm (`maker-server`)
evicted for its duration, restoring whichever one was actually active before the job started.
It holds the same machine-wide flock (`~/.openclaw/cad-build.lock`) every CAD frontend uses, so a
training run and a CAD build can never contend for the GPU at once.

Usage:

```bash
lab/gpu_window.sh <command> [args...]
```

Example (the hand test, no training involved):

```bash
lab/gpu_window.sh nvidia-smi --query-gpu=memory.used --format=csv
```

What it does, in order:

1. Opens the lock file in APPEND mode and takes the lock (`flock -w 3600`, i.e. queues up to 1h
   rather than erroring immediately). Only once the lock is held does it truncate the file and
   write its own `{"pid", "child_pid", "frontend": "lab", "spec", "started"}` holder line, so a
   lab job queueing behind another frontend never wipes that frontend's holder line (it used to:
   `exec 9>` truncates at open time, before `flock` returns).
2. Records whether `maker-server` was active before touching anything (`MAKER_WAS`).
3. Stops both `maker-server` and `qwen38-server`, then polls `nvidia-smi` for up to 60s
   (30 x 2s) for VRAM to drop under 1500 MiB before proceeding; exits 4 if it never drops.
4. Runs the given command as a BACKGROUND child, with `CAD_GPU_WINDOW=1` exported for it and
   under `timeout --signal=TERM --kill-after=60 ${GPU_WINDOW_MAX_SEC:-36000}` (a 10h dead-man
   cap, the same idea as `maker-server.service`'s `RuntimeMaxSec`), then rewrites the holder
   line with the child's PID and `wait`s on it. The child's exit code is this script's exit
   code.
5. On `SIGINT`/`SIGTERM`: kills the child first (TERM, then KILL after 20s) and only then runs
   the EXIT trap. This ordering is the whole point of the background child: with a foreground
   job, bash runs the EXIT trap on signal while the job is still live, so the resident (about
   23GB) was started on top of a running training job (21.7GB peak) and one of them died of a
   CUDA OOM.
6. On exit (success, failure, or handled signal, via `trap ... EXIT`): if `maker-server` was
   active at entry, restarts only `maker-server` (its unit `Conflicts=qwen38-server.service`, so
   systemd stops the resident as a side effect); otherwise restarts `qwen38-server`. The two
   units are never started together, matching the `Conflicts=` declared between them.

Exit codes: `3` = build lock busy for over an hour, `4` = VRAM did not clear after eviction,
`124` = the job hit the wall-clock cap, `130`/`143` = interrupted/terminated, anything else =
the job's own exit code.

### The SIGKILL limitation (read this before using kill -9)

`SIGKILL` (`kill -9`) cannot be trapped by any shell, so killing the wrapper that way skips
everything above: the child keeps running on the GPU, and because it inherits fd 9 the build
lock stays held by that orphan. `health-watch.sh` treats a held build lock as a deliberate
eviction, so it will not restore the resident either, and the box then sits with no resident
until someone intervenes. That is why the holder line carries the child's PID:

```bash
cat ~/.openclaw/cad-build.lock     # {"pid": ..., "child_pid": 12345, ...}
kill 12345                         # then, if needed: systemctl --user start qwen38-server
```

Send `SIGTERM` (plain `kill`, or Ctrl-C in the foreground) instead: that path is handled.

## Runbook

The whole spike, in the order it was actually run on 2026-09-17. Every GPU step goes through
`gpu_window.sh`; every CPU step (data rendering, merge, convert, quantize) does not.

### 1. Render the dataset (CPU)

```bash
python3 lab/data.py --src-train ~/.openclaw/cad-sft-train.jsonl \
                    --src-val ~/.openclaw/cad-sft-val.jsonl --out lab/data
```

Reads the ChatML SFT rows, extracts each row's verbatim spec, drops any row colliding with a
card suite (exact key, or a near-duplicate slug that identifies exactly one spec in its own
suite), and renders the survivors through the checkpoint's OWN `chat_template.jinja` into
`{"prompt", "completion", "id", "kind"}` rows under `lab/data/`. Measured on this corpus: 353
train rows kept, 0 dropped for a missing spec header, 0 contaminated against the 262 suite
specs; 15 val rows.

The checkpoint template is mandatory: if `--template` does not exist the script refuses rather
than quietly rendering through its built-in stand-in framing. `--allow-fallback-template` is the
explicit opt-in (tests use it). `--tokenizer <checkpoint dir>` adds real token-count stats.

### 2. Train (GPU, inside a window)

```bash
lab/spike.sh                                                      # 1 epoch, 353 rows
lab/spike.sh --max-steps 3 --out lab/runs/smoke --save-steps 2    # smoke test
```

`spike.sh` is a thin wrapper: it creates the `--out` directory (its own default, or whatever
`--out` is passed in the overrides) and `exec`s `gpu_window.sh lab/.venv/bin/python lab/train.py`
with the spike's hyperparameters (`--base` the 4-bit Gemma-4-31B checkpoint, `--data lab/data`,
`--rank 16 --epochs 1 --max-seq 5120 --save-steps 25`). Extra arguments are forwarded verbatim,
so any `train.py` flag can be overridden without editing the script.

Measured for the real epoch: 87 steps at about 78 s/step, 1 h 54 min total, peak VRAM 21.7GB,
train loss 0.358, eval loss 0.391, 348 of 353 rows kept at `--max-seq 5120` (5 dropped as too
long, never truncated), 122.4M trainable parameters, 0 of them vision.

Resume a killed run with `--resume` (bare flag = latest checkpoint in `--out`). Re-score any
adapter without training:

```bash
lab/gpu_window.sh lab/.venv/bin/python lab/train.py --eval-only \
    --data lab/data --out lab/runs/spike1 --adapter lab/runs/spike1/adapter
```

`--eval-only` writes `eval_meta.json` beside the adapter and takes `--base` from the adapter's
own `adapter_config.json` when it is omitted.

### 3. Ship the adapter as a GGUF arm

`ship.py` is one subcommand per stage, each idempotent (a completeness marker, not bare path
existence) and each logging a JSON line to `~/lab-scratch/ship_log.jsonl`. The step ORDER below
is not the same as `ship.py all`: the root SSD cannot hold the bf16 base, the merged copy and
the F16 GGUF at once, so each intermediate is deleted as soon as the next stage has consumed it.
That sequencing is what was actually run.

```bash
# merge: adapter + bf16 base -> merged HF dir            (about 33 min, CPU, streaming)
lab/.venv/bin/python lab/ship.py merge --adapter lab/runs/spike1/adapter

# convert: merged HF dir -> F16 GGUF                     (about 39 min, CPU)
lab/.venv/bin/python lab/ship.py convert --model-name gemma-4-31b-cad-spike

# the merged dir has been consumed: delete it before quantizing (about 62GB back)
rm -rf ~/lab-scratch/merged

# imatrix (a few seconds) then quantize straight onto the NVMe store (about 19 min)
lab/.venv/bin/python lab/ship.py fetch-imatrix
lab/.venv/bin/python lab/ship.py quantize \
    --out /mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf

# the F16 has been consumed: delete it (about 62GB back)
rm -f ~/lab-scratch/gemma-4-31b-cad-F16.gguf*

# verify: GPU step, so it MUST run inside a window      (about 1 min)
lab/gpu_window.sh lab/.venv/bin/python lab/ship.py verify \
    --gguf /mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf

# register the arm in benchmarks/arms.json (refuses without a passing verify.ok)
lab/.venv/bin/python lab/ship.py register \
    --gguf /mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf \
    --adapter lab/runs/spike1/adapter

# clean: scratch merged dir + F16 GGUF. The bf16 base is KEPT (see below).
lab/.venv/bin/python lab/ship.py clean
```

Measured wall times for the real run (smoke4 run in brackets): merge 33 min (20 min), convert
39 min (31 min), quantize 19 min (26 min), verify 1 min, register instant.

Notes that matter:

- **`verify` refuses to run outside a GPU window.** It loads an 18.7GB GGUF at `-ngl 99`; run
  bare while the 23GB resident is up on a 24GB card it fights the resident for VRAM and takes no
  build lock. `gpu_window.sh` exports `CAD_GPU_WINDOW=1` for its child, which is what `verify`
  checks. `--i-know-the-gpu-is-free` is the manual override for a human who has already evicted
  everything by hand. Its tok/s figure is a smoke number, not the arm's serving speed: it uses
  `-ngl 99 -c 8192` and no KV quantisation, while `maker-server` serves the arm with
  `-ngl 999 -c 16384 --cache-type-k q8_0 --cache-type-v q8_0` plus the arm's `extra_args`.
- **The bf16 base at `~/lab-scratch/gemma-4-31B-it-bf16` is deliberately kept** (about 59GB).
  Re-downloading `google/gemma-4-31B-it` costs about 80 minutes per later phase, so `clean`
  leaves it alone; `clean --include-base` is the explicit opt-in to delete it anyway.
- **`ship.py all` stops after quantize on purpose.** verify needs a human at the GPU, register
  touches the shared `benchmarks/arms.json`, and clean is destructive: none of the three should
  fire unattended as part of a chained run.
- **merge refuses on a template mismatch.** The merged dir takes its chat template from the bf16
  base, and that is what ends up in the GGUF and frames every request at serve time, while the
  training rows were rendered with the checkpoint's own template. `merge` renders a canary
  conversation through both and refuses on a difference (`--skip-template-check` overrides).
- **merge, convert and quantize precheck free disk space** on the filesystem their output lands
  on and refuse up front with both numbers named, rather than failing half an hour into writing a
  62GB file.
- **`register` refuses without a passing `~/lab-scratch/verify.ok`** (`--force` overrides), the
  same gate `clean` has always had.

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
training run and a CAD build can never contend for the GPU at once; other frontends' JSON holder
lines in that lock file are left alone.

Usage:

```bash
lab/gpu_window.sh <command> [args...]
```

Example (the hand test, no training involved):

```bash
lab/gpu_window.sh nvidia-smi --query-gpu=memory.used --format=csv
```

What it does, in order:

1. Takes the build lock (`flock -w 3600`, i.e. queues up to 1h rather than erroring immediately),
   and writes a `{"pid", "frontend": "lab", "spec", "started"}` line into it so other frontends
   can see a lab job is holding the lock.
2. Records whether `maker-server` was active before touching anything (`MAKER_WAS`).
3. Stops both `maker-server` and `qwen38-server`, then polls `nvidia-smi` for up to 60s
   (30 x 2s) for VRAM to drop under 1500 MiB before proceeding; exits 4 if it never drops.
4. Runs the given command with the GPU otherwise idle.
5. On exit (success, failure, or signal, via `trap ... EXIT`): if `maker-server` was active at
   entry, restarts only `maker-server` (its unit `Conflicts=qwen38-server.service`, so systemd
   stops the resident as a side effect); otherwise restarts `qwen38-server`. The two units are
   never started together, matching the `Conflicts=` declared between them.

Exit codes: `3` = build lock busy for over an hour, `4` = VRAM did not clear after eviction.

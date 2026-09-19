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
evicted for its duration, always restoring the RESIDENT afterwards. It holds the same
machine-wide flock (`~/.openclaw/cad-build.lock`) every CAD frontend uses, so a training run and
a CAD build can never contend for the GPU at once.

Usage:

```bash
lab/gpu_window.sh <command> [args...]
```

Example (the hand test, no training involved):

```bash
lab/gpu_window.sh nvidia-smi --query-gpu=memory.used --format=csv
```

What it does, in order:

1. Opens the lock file in APPEND mode and takes the lock (`flock -w ${GPU_WINDOW_LOCK_WAIT_SEC:-3600}`,
   i.e. queues up to 1h by default rather than erroring immediately). Only once the lock is held
   does it truncate the file and write its own
   `{"pid", "child_pid", "pgid", "frontend": "lab", "spec", "started"}` holder line, so a lab job
   queueing behind another frontend never wipes that frontend's holder line (it used to:
   `exec 9>` truncates at open time, before `flock` returns). If the lock never comes free it
   exits **75** (`EX_TEMPFAIL`) and touches no service at all.
2. Records whether `maker-server` was active before touching anything (`MAKER_WAS`), for the log
   line only.
3. Stops both `maker-server` and `qwen38-server`, then polls `nvidia-smi` for up to 60s
   (30 x 2s) for VRAM to drop under 1500 MiB before proceeding; exits 4 if it never drops.
4. Runs the given command as a BACKGROUND child in its OWN process group (`set -m`, so the
   group's PGID is the child's PID), with `CAD_GPU_WINDOW=1` and `PYTHONUTF8=1` exported for it,
   with the build lock fd closed for it (`9>&-`), and under
   `timeout --signal=TERM --kill-after=${GPU_WINDOW_GRACE_SEC:-180} ${GPU_WINDOW_MAX_SEC:-36000}`
   (a 10h dead-man cap, the same idea as `maker-server.service`'s `RuntimeMaxSec`), then rewrites
   the holder line with the child's PID and PGID and `wait`s on it. The child's exit code is this
   script's exit code.
5. On `SIGINT`/`SIGTERM`: signals the job's whole PROCESS GROUP first (TERM, then KILL after
   `GPU_WINDOW_GRACE_SEC`, default 180s) and only then runs the EXIT trap. Two things matter
   here. The background child is what makes the ordering possible at all: with a foreground job,
   bash runs the EXIT trap on signal while the job is still live, so the resident (about 23GB)
   was started on top of a running training job (21.7GB peak) and one of them died of a CUDA
   OOM. And the signal goes to the group, not to the PID, because the PID is `timeout`: killing
   it alone left the real job running as an orphan, still on the GPU and still holding the build
   lock through the inherited fd 9, while the trap started the resident on top of it.
   `timeout` is deliberately not given `--foreground`, so its own expiry signal also goes to the
   whole group and reaches grandchildren.
6. On exit (success, failure, or handled signal, via `trap ... EXIT`): stops `maker-server` and
   starts `qwen38-server`, ALWAYS. The maker arm is per-build, never a resting state, so a stray
   maker at entry used to make this trap restart the maker after the job's own bookend had
   correctly restored the resident, leaving the household with no chat model.
   `GPU_WINDOW_RESTORE=maker` is the escape hatch for a caller that really wants the maker arm
   left running; nothing uses it today. A second `SIGTERM` arriving during the restore is
   ignored, so it cannot abandon it half way.

Knobs (all optional, defaults in brackets):

| Env var | Default | What it does |
|---|---|---|
| `GPU_WINDOW_MAX_SEC` | 36000 | wall-clock cap for the job |
| `GPU_WINDOW_GRACE_SEC` | 180 | TERM to KILL grace, and `timeout --kill-after` |
| `GPU_WINDOW_LOCK_WAIT_SEC` | 3600 | how long to queue for the build lock before exiting 75 |
| `GPU_WINDOW_RESTORE` | resident | `maker` restores the maker arm instead of the resident |
| `GPU_WINDOW_SYSTEMCTL` / `GPU_WINDOW_NVIDIA_SMI` | `systemctl` / `nvidia-smi` | test seams |
| `CAD_BUILD_LOCK_FILE` | `~/.openclaw/cad-build.lock` | the same var `cad_v5/config.py` honours |

The grace default is 180s because that is what a real job needs: a build step in flight can take
about 120s to return before the job even sees its abort flag, and the job's own bookend then
cold-loads the resident (25-40s). `lab/harvest_unit.sh` sets `GPU_WINDOW_LOCK_WAIT_SEC=120`, so a
timer tick that would merely queue skips instead (exit 75, declared `SuccessExitStatus=` in
`deploy/lab-harvest.service`) rather than being killed by that unit's own `RuntimeMaxSec`.

Exit codes: `75` = build lock busy for the whole wait (a skip, nothing was touched), `4` = VRAM
did not clear after eviction, `124` = the job hit the wall-clock cap (`137` when the job ignored
that TERM and the group-wide KILL took `timeout` down with it), `130`/`143` =
interrupted/terminated, anything else = the job's own exit code.

Tests: `tests/test_gpu_window.py` drives the real script against stub `systemctl`/`nvidia-smi`
binaries and a temp lock file, including the stubborn-child, grandchild, grace, restore-target,
lock-contention and child-environment cases.

### The SIGKILL limitation (read this before using kill -9)

`SIGKILL` (`kill -9`) cannot be trapped by any shell, so killing the WRAPPER that way still skips
the restore, and the JOB SURVIVES and keeps the GPU. The job no longer holds the build lock
(its fd is closed), so the lock is released with the wrapper; `health-watch.sh` then sees a
free lock with no maker-server active, calls it a real failure and restarts the resident ON
TOP of the orphaned job within about 5 minutes. So after a `kill -9` of the wrapper, kill the
job's group yourself first; the group is named in the holder line. Under the harvest unit,
`ExecStopPost=` starts the resident after systemd has SIGKILLed the whole cgroup, which is safe.
Stop the window with TERM, never KILL:

```bash
cat ~/.openclaw/cad-build.lock     # {"pid": ..., "child_pid": 12345, "pgid": 12345, ...}
kill -TERM -- -12345               # then, if needed: systemctl --user start qwen38-server
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
train rows and 16 val rows kept, 0 dropped for a missing spec header, 0 contaminated against
the 262 suite specs. Re-running it after the fix round reproduced both files byte-identically.

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
long, never truncated) and 15 of the 16 val rows, 122.4M trainable parameters, 0 of them
vision.

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

## Spec generation (Phase 3, `lab/specgen.py`)

`lab/specgen.py` writes new CAD part specs by prompting the maker arm, family by family, until
the spec bank has enough rows at enough tier 3-4 share. It is a one-time, multi-hour,
unattended job, and like `ship.py verify` it **refuses to run outside a GPU window**: it
switches the maker arm with `scripts/arms.py use <arm>` and then talks to it, so without the
window's build lock it would take the GPU out from under any CAD build, benchmark card or lab
job already running. `--i-know-the-gpu-is-free` is the manual override, same name and meaning
as `verify`'s.

The only supported launch, from a script file (not pasted into a shell), with a fresh log every
time:

```bash
setsid nohup lab/gpu_window.sh python3 lab/specgen.py \
    --total 2500 --target-tier34 0.45 \
    > ~/lab-scratch/specgen-$(date +%Y%m%d-%H%M).log 2>&1 &
```

The smoke shape is the same wrapper with one batch:

```bash
lab/gpu_window.sh python3 lab/specgen.py --once --group plate --n 20
```

- **While it runs, every CAD build on the box waits on the build lock** ("waiting for GPU"), so
  run it at night. The window holds `~/.openclaw/cad-build.lock` for the whole run.
- **Bounds:** `--max-hours` (default 4) is checked between batches; `GPU_WINDOW_MAX_SEC`
  (default 10h) is the window's own dead-man cap. `run_total`'s circuit breakers (3 consecutive
  failed model calls, 8 consecutive zero-accept batches) stop a sick run sooner.
- **To stop it, send ONE SIGTERM to the `gpu_window.sh` process** (`kill <pid>`, never `-9`).
  The window TERMs specgen's whole process group first; specgen stops on that one signal, runs
  its `arms.py restore` bookend (it has `GPU_WINDOW_GRACE_SEC`, 180s by default, to do it), and
  only then does the window restore the box. A second SIGTERM during that cleanup is ignored on
  purpose, by both.
- **Recovery check after any hard kill:**

  ```bash
  ls ~/.openclaw/cad.json.pre-arm ~/.openclaw/maker.env.pre-arm
  python3 scripts/arms.py restore    # only if either file exists
  ```

  A healthy-looking `qwen38-server` says nothing about whether `cad.json` still points at the
  run's arm, so run the check even then.
- **The double restore is harmless.** specgen's own bookend restores the pre-run `maker` block
  and starts the resident; the window's EXIT trap then does its own restore, which is a no-op
  stop of an already-stopped maker plus a start of an already-running resident. Since Task 3a
  the window's restore always targets the resident, so the two can no longer disagree (they
  used to: a stray maker at entry made the window restart the maker over specgen's resident).

## Teacher reference geometry (Phase 3 Task 3d1, `lab/teacher_refs.py`)

The harvest confirms a candidate as a training pair either by agreement between two samples
of the local model, or by matching a REFERENCE geometry (`reference_stl` on a bank row,
scored by `geom_bands.score_against_reference`) -- the stronger of the two. Only
owner-supplied references existed until now. `lab/teacher_refs.py` promotes the 414
`source: "teacher-suite"` bank rows' own August solves as references: `~/.openclaw/
cad-sftpairs.jsonl` already holds accepted build123d code for most of them (`source`
"teacher" or "teacher-human-accepted"), written by a stronger teacher model back when the
gate was weaker. That code is used ONLY as geometry to check the local model's own code
against -- it never becomes training text.

```bash
python3 lab/specbank.py import-teacher-refs --dry-run                # counts only, writes nothing
python3 lab/specbank.py import-teacher-refs --limit 10                # a real, bounded run
python3 lab/specbank.py import-teacher-refs                           # the full remaining set
python3 lab/specbank.py stats                                         # now reports with_reference
```

CPU only, no model calls, no service starts/stops: `cad_engine._ollama` is patched to raise
on import, and every part is built/inspected through `cad_engine.run_step`/`run_inspect`
(local subprocesses) plus a fresh child interpreter for build123d's STEP->STL conversion and
for `lab/harvest.py`'s `strict_envelope_check` (a separate process per call, deliberately,
so a bad moment in a file being edited elsewhere never takes the whole run down -- see the
module docstring). Each matched teacher solve is RE-BUILT and RE-GATED under TODAY's rules
(the engine's deterministic gate plus the strict envelope check); several suites clean in
August now fail on the same numeric mismatches the gate was hardened to catch since, and
those are reported, never admitted. Up to 3 candidates build in parallel
(`--workers`, clamped to 3). Admitted rows gain `reference_stl` (cached at
`lab/state/refs/<key[:16]>.stl`, git-ignored), `reference_source: "teacher-claude"`,
`reference_facts` (solids/faces/volume/bbox/bores/hole_groups) and `reference_added`.
Owner references always win and a second run is a no-op: any row that already carries
`reference_stl` is left alone. A full run writes `lab/state/teacher_refs_report.json`
(git-ignored) with counts by tier/suite for every stage (read, rejected by review, matched,
crashed, failed-gate, admitted) plus the non-admitted list with reasons.

`specbank.apply_reference_updates` is the bank-mutation half: it takes the SAME flock
`add_items`/`import_teacher`/`import_references` already use, rewrites the file in place
(never via a temp-file + rename -- a rename would swap the path to a new inode while a
concurrent appender already blocked on this lock still holds an fd to the old one, so its
write would succeed and still be lost forever once orphaned), and leaves every row it does
not touch byte-identical to how it was written.

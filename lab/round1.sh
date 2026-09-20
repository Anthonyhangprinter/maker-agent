#!/usr/bin/env bash
# lab/round1.sh -- launch the Phase 3 round 1 QLoRA training run inside a GPU window.
#
# Same shape as lab/spike.sh (a thin wrapper around lab/gpu_window.sh + lab/train.py, the
# resident/maker CAD models evicted for the duration, the CAD build lock held so no CAD
# frontend can contend for the GPU mid-run), but a GENTLER preset for a small,
# self-generated round instead of the Phase 2 spike's 353-row corpus:
#
#   epochs 2 (spike: 1), everything else unchanged from the spike (r16 / alpha 16 via
#   train.py's own lora_alpha=rank / language layers only / batch 1 x accum 4 /
#   max-seq 5120 / adamw_8bit / bf16), data from lab/rounds/round1/ (see lab/compile.py
#   and the "Compile and train a round" section of lab/README.md for how that directory
#   is produced), output under ~/lab-scratch/round1-adapter-<timestamp>.
#
# Learning rate 1e-4 (the spike used 2e-4): lab/train.py gained --lr on 2026-09-21.
# Usage:
#   lab/round1.sh                                            # full round 1 run
#   lab/round1.sh --out ~/lab-scratch/round1-try2             # explicit output dir
#   lab/round1.sh --max-steps 3 --out lab/runs/round1-smoke --save-steps 2   # smoke test
#
# Extra arguments are forwarded to lab/train.py verbatim (same as spike.sh), so any
# train.py flag can be overridden without editing this script. THIS SCRIPT DOES NOT RUN
# ITSELF -- it only launches lab/gpu_window.sh, which does real GPU work; nothing here
# starts, stops, or restarts any systemd unit on its own.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAMP="$(date +%Y%m%d-%H%M%S)"

# Create ONLY the output directory this run will actually use (same reasoning as
# spike.sh's own comment, finding 25 there): a smoke run that overrides --out used to
# create an empty default dir beside its real output. train.py creates --out itself;
# this mkdir just means a caller sees the directory immediately.
OUT="$HOME/lab-scratch/round1-adapter-$STAMP"
prev=""
for arg in "$@"; do
    case "$arg" in
        --out=*) OUT="${arg#--out=}" ;;
    esac
    if [ "$prev" = "--out" ]; then OUT="$arg"; fi
    prev="$arg"
done
mkdir -p "$OUT"

# Same lookup for --data, only so the wall-time estimate below can find the right
# train.jsonl; the actual --data value train.py uses is still the literal default plus
# "$@" forwarding, same last-value-wins logic as --out.
DATA="lab/rounds/round1"
prev=""
for arg in "$@"; do
    case "$arg" in
        --data=*) DATA="${arg#--data=}" ;;
    esac
    if [ "$prev" = "--data" ]; then DATA="$arg"; fi
    prev="$arg"
done

# Fresh log name per launch (timestamp + PID, so two launches in the same second never
# collide), never overwritten by a later run.
LOG_DIR="$HOME/lab-scratch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/round1-train-$STAMP-$$.log"

echo "lab/round1.sh: logging this launch to $LOG"
echo "lab/round1.sh: gentle preset, lr 1e-4 (spike: 2e-4), 2 epochs"

# Expected wall time from the row count (informational only; the spike measured 78s per
# optimizer step at accum 4, i.e. 4 rows/step): steps = ceil(rows/4) * epochs. Silent if
# train.jsonl is not there yet (run lab/compile.py first) or --epochs is overridden below
# in a way this estimate does not track.
if [ -f "$DATA/train.jsonl" ]; then
    ROWS=$(grep -c . "$DATA/train.jsonl" 2>/dev/null || echo 0)
    if [ "${ROWS:-0}" -gt 0 ]; then
        STEPS=$(( (ROWS + 3) / 4 * 2 ))
        SECONDS_EST=$(( STEPS * 78 ))
        echo "lab/round1.sh: $ROWS training rows in $DATA/train.jsonl -> about $STEPS" \
             "optimizer steps at 2 epochs -> about $((SECONDS_EST / 60)) min at 78s/step" \
             "(the spike's own measured rate; --epochs overrides below are not reflected here)"
    fi
fi

# --out lab/runs/round1 below is the literal DEFAULT train.py sees; if the caller's own
# "$@" also carries --out (forwarded last), argparse's last-value-wins semantics let the
# caller's value win, same pattern as spike.sh.
# The training venv is untracked and lives in the MAIN checkout; a worktree has none.
LAB_PY="${LAB_PY:-lab/.venv/bin/python}"
[ -x "$LAB_PY" ] || LAB_PY="$HOME/.openclaw/skills/cad-builder/lab/.venv/bin/python"
[ -x "$LAB_PY" ] || { echo "lab/round1.sh: no training venv found ($LAB_PY)" >&2; exit 2; }
lab/gpu_window.sh "$LAB_PY" lab/train.py \
    --base /mnt/nvme-apps/LinuxModels/gemma-4-31B-it-unsloth-bnb-4bit \
    --data lab/rounds/round1 \
    --out "$OUT" \
    --rank 16 \
    --epochs 2 \
    --lr 1e-4 \
    --max-seq 5120 \
    --save-steps 25 \
    "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"

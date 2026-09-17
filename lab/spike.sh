#!/usr/bin/env bash
# lab/spike.sh -- launch the Phase 2 QLoRA training spike inside a GPU window.
#
# Runs one epoch of lab/train.py over the full 353-row training set on the
# Phase 0 winner (Gemma-4-31B-it, 4-bit), with the resident/maker CAD models
# evicted for the duration (lab/gpu_window.sh) and the CAD build lock held so
# no CAD frontend can contend for the GPU mid-run.
#
# Usage:
#   lab/spike.sh                         # full spike1 run (1 epoch, 353 rows)
#   lab/spike.sh --max-steps 3 --out lab/runs/smoke --save-steps 2   # smoke test
#
# Extra arguments are forwarded to lab/train.py verbatim, so a smoke test can
# override --out/--max-steps/--save-steps without editing this script.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p lab/runs/spike1

exec lab/gpu_window.sh lab/.venv/bin/python lab/train.py \
    --base /mnt/nvme-apps/LinuxModels/gemma-4-31B-it-unsloth-bnb-4bit \
    --data lab/data \
    --out lab/runs/spike1 \
    --rank 16 \
    --epochs 1 \
    --max-seq 5120 \
    --save-steps 25 \
    "$@"

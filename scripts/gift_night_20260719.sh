#!/bin/bash
# gift_night_20260719.sh — GIFT upgrade verdict night (one heavy job at a time, capped unit):
#   leg A: tiers 1-2, fast rung, N=1 first-turn candidates (fresh baseline — the gate grew
#          new [spec] checks today, so the 2026-07-17 13/22 is no longer comparable)
#   leg B: tiers 1-2, fast rung, N=3 (the best-of-N A/B leg)
#   then:  full GIFT-REJECT/FAIL sampler over the corpus (K=8)
# Detaches into a MemoryHigh-capped transient systemd unit per SESSION PROTOCOL (the
# 2026-07-16 oomd lesson). Log: benchmarks/results/gift-night-20260719.log — finished only
# when it ends "=== ALL DONE ===".
set -u
CB=/home/theultimatecunt/.openclaw/skills/cad-builder
LOG="$CB/benchmarks/results/gift-night-20260719.log"

if [ -z "${GIFT_DETACHED:-}" ]; then
  exec systemd-run --user --collect --unit="gift-night-$(date +%H%M%S)" \
    -p MemoryHigh=12G -p Nice=10 \
    --setenv=GIFT_DETACHED=1 "$(readlink -f "$0")"
fi

{
  echo "=== [$(date '+%F %T')] leg A: tiers 1-2 fast N=1 ==="
  CAD_CANDIDATES=1 python3 "$CB/scripts/run_benchmarks.py" --tiers 1,2 --coder fast
  echo "=== [$(date '+%F %T')] leg B: tiers 1-2 fast N=3 ==="
  CAD_CANDIDATES=3 python3 "$CB/scripts/run_benchmarks.py" --tiers 1,2 --coder fast
  echo "=== [$(date '+%F %T')] gift_sample full K=8 ==="
  python3 "$CB/scripts/gift_sample.py" --k 8
  echo "=== [$(date '+%F %T')] oomd taint check ==="
  journalctl --since today 2>/dev/null | grep -i "oomd.*Killed" || echo "no oomd kills today"
  echo "=== ALL DONE ==="
} >> "$LOG" 2>&1

#!/bin/bash
# refresh_20260716.sh — staggered coder-model refresh sequence (ONE heavy job at a time).
# Run as a detached systemd unit (SESSION PROTOCOL #4), never as a harness background task.
# Log: benchmarks/results/refresh-20260716.log ; finished only when it ends "=== ALL DONE ===".
set -u
CB=/home/theultimatecunt/.openclaw/skills/cad-builder
CADJSON=/home/theultimatecunt/.openclaw/cad.json
LOG_TS() { echo "[$(date '+%F %T')] $*"; }

# Always leave the ladder unpinned, whatever happens.
trap 'echo "{}" > "$CADJSON"; LOG_TS "EXIT: cad.json unpinned"' EXIT

# Memory-pressure gate: wait up to 10 min for avg10(some) < 40%; else skip the stage.
pressure_ok() {
  for _ in $(seq 1 20); do
    P=$(awk -F'avg10=' '/^some/{split($2,a," "); print int(a[1])}' /proc/pressure/memory)
    [ "${P:-100}" -lt 40 ] && return 0
    LOG_TS "memory pressure ${P}% >= 40% — waiting 30s"; sleep 30
  done
  LOG_TS "SKIPPING stage: pressure never settled (box busy — noted, moving on)"
  return 1
}

run_tiers12() {  # $1 = model to pin
  printf '{"code_model": "%s"}\n' "$1" > "$CADJSON"
  LOG_TS "== tiers 1-2 pinned to $1"
  (cd "$CB" && python3 scripts/run_benchmarks.py --tiers 1,2 --timeout 1500) 2>&1 | tail -20
  echo "{}" > "$CADJSON"
  sleep 60   # let the box settle between stages
}

LOG_TS "=== STAGE 1: pull qwen3.6 Q3_K_M (alone, nothing else running) ==="
pressure_ok && ollama pull hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:Q3_K_M 2>&1 | tail -2
sleep 30

LOG_TS "=== STAGE 2: rung-1 shootout (tiers 1-2, sequential) ==="
for M in granite3.3:8b granite4:7b-a1b-h qwen3:4b; do
  pressure_ok && run_tiers12 "$M"
done

LOG_TS "=== STAGE 3: strong rung — qwen3.6 Q3_K_M tiers 1-2 (vs v52_qwen3-coder-30b_tiers12.json) ==="
if ollama list | grep -q "Qwen3.6-35B-A3B-GGUF"; then
  Q3=$(ollama list | awk '/Qwen3.6-35B-A3B-GGUF/{print $1; exit}')
  pressure_ok && run_tiers12 "$Q3"
else
  LOG_TS "Q3 model not present (stage 1 failed/skipped) — stage 3 skipped, reason logged"
fi

LOG_TS "=== ALL DONE ==="

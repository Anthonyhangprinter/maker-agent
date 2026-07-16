#!/bin/bash
# run_refresh.sh — generalized staggered coder-model refresh driver (ONE heavy job at a time).
# Successor to refresh_20260716.sh after the 2026-07-16 oomd crashes: self-detaches into a
# memory-capped transient systemd unit (SESSION PROTOCOL #4) so a runaway build gets throttled
# instead of the desktop getting killed. NOTE the cap governs the bench-side processes only —
# ollama lives in the system slice; the real RAM protection is the 24G swap + relaxed oomd.
#
# Usage:
#   run_refresh.sh rung1  <model> [model...]   # tiers 1-2 per model, sequential
#   run_refresh.sh strong <model>              # full text-to-cad suite (pinned, headroom-gated)
#   run_refresh.sh heldout <model>             # 25-part heldout-cqe suite (pinned) — hours
#   run_refresh.sh baseline                    # unpinned auto-ladder full suite
#   run_refresh.sh etj <model> [model...]      # tiers 1-2 via the earthtojake-plugin
#                                              # adapter (agents/etj_agent.py) per model
#   run_refresh.sh night-20260716              # tonight's chain: rung1 granites -> strong
#                                              # qwen3.6:35b-a3b -> heldout qwen3:8b
# Log: benchmarks/results/refresh-<date>.log ; finished only when it ends "=== ALL DONE ===".
set -u
CB=/home/theultimatecunt/.openclaw/skills/cad-builder
CADJSON=/home/theultimatecunt/.openclaw/cad.json
LOG="$CB/benchmarks/results/refresh-$(date +%Y%m%d).log"
LOG_TS() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Self-detach: re-exec inside a capped transient unit unless already there.
if [ -z "${REFRESH_DETACHED:-}" ]; then
  UNIT="cad-refresh-$(date +%H%M%S)"
  exec systemd-run --user --collect --unit="$UNIT" \
    -p MemoryHigh=12G -p Nice=10 \
    --setenv=REFRESH_DETACHED=1 "$(readlink -f "$0")" "$@"
fi

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

# Strong-rung preflight: refuse to start when RAM+swap headroom can't hold the offload.
# 23G weights - ~6G GPU ≈ 17G resident; want that plus slack across MemAvailable+SwapFree.
headroom_ok() {
  local need_gb=${1:-20}
  local avail_kb swap_kb total_gb
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  swap_kb=$(awk '/SwapFree/{print $2}' /proc/meminfo)
  total_gb=$(( (avail_kb + swap_kb) / 1024 / 1024 ))
  if [ "$total_gb" -lt "$need_gb" ]; then
    LOG_TS "ABORT stage: only ${total_gb}G RAM+swap headroom (< ${need_gb}G) — close apps or grow swap"
    return 1
  fi
  LOG_TS "headroom ok: ${total_gb}G RAM+swap available"
}

run_suite() {  # $1=model|auto  $2=suite  $3=tiers(""=all)  $4=timeout  [$5=agent path]
  local model=$1 suite=$2 tiers=$3 timeout=$4 args=()
  if [ "$model" != "auto" ]; then
    printf '{"code_model": "%s"}\n' "$model" > "$CADJSON"
  fi
  [ -n "$tiers" ] && args+=(--tiers "$tiers")
  [ -n "${5:-}" ] && args+=(--agent "$5")
  LOG_TS "== suite=$suite tiers=${tiers:-all} pinned to $model"
  (cd "$CB" && python3 scripts/run_benchmarks.py --suite "$suite" "${args[@]}" \
      --timeout "$timeout") 2>&1 | tee -a "$LOG" | tail -20
  echo "{}" > "$CADJSON"
  sleep 60   # let the box settle between stages
}

MODE=${1:?usage: run_refresh.sh rung1|strong|baseline [models...]}; shift || true
LOG_TS "=== run_refresh.sh mode=$MODE models=$* (unit=${UNIT:-inherited}) ==="

case "$MODE" in
  rung1)
    for M in "$@"; do
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500
    done
    ;;
  strong)
    M=${1:?strong needs a model}
    headroom_ok 20 || exit 1
    pressure_ok && run_suite "$M" text-to-cad "" 1800
    ;;
  heldout)
    M=${1:?heldout needs a model}
    pressure_ok && run_suite "$M" heldout-cqe "" 1500
    ;;
  baseline)
    pressure_ok && run_suite auto text-to-cad "" 1500
    ;;
  etj)
    for M in "$@"; do
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500 "$CB/agents/etj_agent.py"
    done
    ;;
  day-20260717)
    # Strong A/B re-run with the size-based codegen timeout fix (85b6865), then the
    # earthtojake-pipeline legs (etj keeps one model loaded — no brief/critic rotation).
    if headroom_ok 20; then
      pressure_ok && run_suite "qwen3.6:35b-a3b" text-to-cad "" 1800
    fi
    for M in qwen3:8b granite4:7b-a1b-h granite3.3:8b qwen3.6:35b-a3b; do
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500 "$CB/agents/etj_agent.py"
    done
    ;;
  night-20260716)
    for M in granite4:7b-a1b-h granite3.3:8b; do
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500
    done
    if headroom_ok 20; then
      pressure_ok && run_suite "qwen3.6:35b-a3b" text-to-cad "" 1800
    fi
    pressure_ok && run_suite "qwen3:8b" heldout-cqe "" 1500
    ;;
  *) LOG_TS "unknown mode $MODE"; exit 2 ;;
esac

LOG_TS "=== ALL DONE ==="

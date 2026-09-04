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

# Always leave the ladder unpinned, whatever happens. Also tear down the critic-ab GPU
# mmproj drop-in if a run died mid-leg — production default is CPU offload.
MMPROJ_DROPIN="$HOME/.config/systemd/user/qwen36-server.service.d/mmproj-gpu.conf"
trap 'echo "{}" > "$CADJSON"; LOG_TS "EXIT: cad.json unpinned";
      if [ -f "$MMPROJ_DROPIN" ]; then rm -f "$MMPROJ_DROPIN"; systemctl --user daemon-reload;
        systemctl --user restart qwen36-server; LOG_TS "EXIT: mmproj GPU drop-in removed"; fi' EXIT

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

run_suite() {  # $1=model|auto  $2=suite  $3=tiers(""=all)  $4=timeout  [$5=agent path]  [$6=nofs]
  local model=$1 suite=$2 tiers=$3 timeout=$4 args=()
  if [ "$model" != "auto" ]; then
    printf '{"code_model": "%s"}\n' "$model" > "$CADJSON"
  fi
  [ -n "$tiers" ] && args+=(--tiers "$tiers")
  [ -n "${5:-}" ] && args+=(--agent "$5")
  [ "${6:-}" = "nofs" ] && args+=(--no-fewshots)
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
    T=${2:-1800}   # per-part cap; a rotating 35B needs ~3600 for 2+ edit turns (2026-07-17)
    headroom_ok 20 || exit 1
    pressure_ok && run_suite "$M" text-to-cad "" "$T"
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
  ab7b-20260717)
    # 2x2 user-requested A/B: qwen2.5-coder:7b Q4 (user's research pick) vs qwen3:8b incumbent,
    # each with few-shots ON and OFF — separates model quality from scaffolding lift (the
    # corpus/pitfalls were distilled mostly from qwen3 builds, a real affinity confound).
    for M in qwen2.5-coder:7b-instruct-q4_K_M qwen3:8b; do
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500
      pressure_ok && run_suite "$M" text-to-cad "1,2" 1500 "" nofs
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
  critic-ab-20260815)
    # Critic A/B (user request 2026-08-15): gemma4:e4b vs the resident 35B (now that
    # qwen36-server carries an mmproj). Coder PINNED to the 7B fast rung in both legs so
    # only the critic varies; per-build critic model-call seconds land in critic_secs in
    # the run JSON. Timeout 2100 both legs — the 35B critic pays CPU image-encode plus a
    # server restart per turn and must not be timeout-clipped into a gate-only leg.
    M=qwen2.5-coder:7b-instruct-q4_K_M
    pressure_ok && run_suite "$M" text-to-cad "1,2" 2100
    # Leg B runs the 35B critic in its END-STATE config: mmproj on GPU (user call
    # 2026-08-16 — CPU-encode numbers would skew the speed comparison; smoke reference:
    # 270s/call on CPU). The drop-in survives the engine's own stop/start cycles and is
    # removed after the leg (and by the EXIT trap on any crash).
    mkdir -p "$(dirname "$MMPROJ_DROPIN")"
    printf '[Service]\nEnvironment=MMPROJ_OFFLOAD=gpu\n' > "$MMPROJ_DROPIN"
    systemctl --user daemon-reload
    systemctl --user restart qwen36-server
    export CAD_CRITIC_MODEL=local:qwen3.6-35b-a3b CAD_CRITIC_TIMEOUT=480
    pressure_ok && run_suite "$M" text-to-cad "1,2" 2100
    unset CAD_CRITIC_MODEL CAD_CRITIC_TIMEOUT
    rm -f "$MMPROJ_DROPIN"; systemctl --user daemon-reload
    systemctl --user restart qwen36-server
    LOG_TS "critic-ab: mmproj restored to CPU offload"
    ;;
  *) LOG_TS "unknown mode $MODE"; exit 2 ;;
esac

LOG_TS "=== ALL DONE ==="

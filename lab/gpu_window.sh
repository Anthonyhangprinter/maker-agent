#!/usr/bin/env bash
# One bounded GPU job with the resident and maker evicted; restores both states in a trap.
set -euo pipefail
LOCK="$HOME/.openclaw/cad-build.lock"
exec 9>"$LOCK"; flock -w 3600 9 || { echo "gpu_window: build lock busy for 1h" >&2; exit 3; }
echo "{\"pid\": $$, \"frontend\": \"lab\", \"spec\": \"$*\", \"started\": \"$(date -Is)\"}" >&9
MAKER_WAS=$(systemctl --user is-active maker-server || true)
restore() {
    # maker-server and qwen38-server Conflict= each other, so only one is ever
    # started here: if maker-server was active at entry, restore it (systemd
    # stops qwen38-server as a side effect of the Conflicts relationship);
    # otherwise restore the resident.
    if [ "$MAKER_WAS" = "active" ]; then
        systemctl --user start maker-server || echo "gpu_window: WARNING maker-server failed to restart" >&2
    else
        systemctl --user stop maker-server 2>/dev/null || true
        systemctl --user start qwen38-server || echo "gpu_window: WARNING qwen38-server failed to restart, GPU has no resident" >&2
    fi
}
trap restore EXIT
systemctl --user stop maker-server 2>/dev/null || true; systemctl --user stop qwen38-server || true
for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 2; done
[ "$used" -lt 1500 ] || { echo "gpu_window: VRAM still ${used} MiB after eviction" >&2; exit 4; }
echo "== gpu_window: start $(date +%T) :: $*"; "$@"; echo "== gpu_window: done $(date +%T)"

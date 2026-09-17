#!/usr/bin/env bash
# One bounded GPU job with the resident and maker evicted; restores both states in a trap.
#
# The job runs as a BACKGROUND child that this wrapper `wait`s on, so a SIGINT/SIGTERM to the
# wrapper kills the child FIRST and only then restores the resident (fix round 3, finding 2:
# with a foreground child, bash ran the EXIT trap on signal while the job kept running, and
# restore() then started a 23GB resident on top of a live 21.7GB training job).
#
# SIGKILL (kill -9) to this wrapper cannot be handled by any shell: the traps never run, the
# child keeps the GPU, and because the child inherits fd 9 the build lock stays held by the
# orphan. The child's PID is written into the lock holder line precisely so a human can find
# and kill it in that case: `cat ~/.openclaw/cad-build.lock`, then kill the child_pid.
set -euo pipefail
LOCK="$HOME/.openclaw/cad-build.lock"
# Open append-only: `exec 9>"$LOCK"` truncated the file BEFORE flock returned, wiping the
# live holder's JSON line while this process was still queueing behind it (finding 3).
exec 9>>"$LOCK"; flock -w 3600 9 || { echo "gpu_window: build lock busy for 1h" >&2; exit 3; }
# Only now that the lock is held may the holder line be replaced. fd 9 is O_APPEND, so every
# write below lands at the (post-truncation) end of file.
SPEC="$*"
# JSON-escape the spec so the holder line stays parseable for cad_engine's
# "waiting for build lock (held by: ...)" diagnostic even when the command contains quotes.
SPEC_JSON=${SPEC//\\/\\\\}; SPEC_JSON=${SPEC_JSON//\"/\\\"}
STARTED="$(date -Is)"
: > "$LOCK"
echo "{\"pid\": $$, \"child_pid\": null, \"frontend\": \"lab\", \"spec\": \"$SPEC_JSON\", \"started\": \"$STARTED\"}" >&9
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
CHILD=""
kill_child() {
    [ -n "$CHILD" ] || return 0
    kill -0 "$CHILD" 2>/dev/null || return 0
    echo "gpu_window: signalled, sending TERM to child $CHILD" >&2
    kill -TERM "$CHILD" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "$CHILD" 2>/dev/null || return 0
        sleep 1
    done
    echo "gpu_window: child $CHILD still alive after 20s, sending KILL" >&2
    kill -KILL "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
}
trap restore EXIT
trap 'kill_child; exit 143' TERM
trap 'kill_child; exit 130' INT
systemctl --user stop maker-server 2>/dev/null || true; systemctl --user stop qwen38-server || true
for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 2; done
[ "$used" -lt 1500 ] || { echo "gpu_window: VRAM still ${used} MiB after eviction" >&2; exit 4; }
echo "== gpu_window: start $(date +%T) :: $SPEC"
# Wall-clock cap (finding 4), the same dead-man idea as maker-server.service's RuntimeMaxSec:
# TERM at GPU_WINDOW_MAX_SEC (default 10h), KILL 60s later. Exit code 124 = timed out.
# CAD_GPU_WINDOW=1 is what lab/ship.py's `verify` checks before it starts a GPU server.
CAD_GPU_WINDOW=1 timeout --signal=TERM --kill-after=60 "${GPU_WINDOW_MAX_SEC:-36000}" "$@" &
CHILD=$!
# Rewrite the holder line now that the child exists: a SIGKILLed wrapper leaves this line as
# the only record of which PID is still holding the GPU (see the header note).
: > "$LOCK"
echo "{\"pid\": $$, \"child_pid\": $CHILD, \"frontend\": \"lab\", \"spec\": \"$SPEC_JSON\", \"started\": \"$STARTED\"}" >&9
rc=0
wait "$CHILD" || rc=$?
echo "== gpu_window: done $(date +%T) (exit $rc)"
exit "$rc"

#!/usr/bin/env bash
# One bounded GPU job with the resident and the maker arm evicted; the EXIT trap always
# puts the RESIDENT back.
#
# The job runs as a BACKGROUND child that this wrapper `wait`s on, so a SIGINT/SIGTERM to the
# wrapper kills the job FIRST and only then restores the resident (fix round 3, finding 2:
# with a foreground child, bash ran the EXIT trap on signal while the job kept running, and
# restore() then started a 23GB resident on top of a live 21.7GB training job).
#
# Task 3a (2026-09-19) fixed four defects in that design:
#
# D1 PROCESS GROUP. `$CHILD` is the `timeout` process, not the job, so the old
#    `kill -KILL "$CHILD"` after the grace killed the supervisor and left the real job
#    running as an orphan: still on the GPU, and still holding this script's fd 9 (the build
#    lock) by inheritance, while the trap started the 23GB resident on top of it. The job is
#    now launched under bash job control (`set -m`), which puts it in its OWN process group
#    whose PGID equals `$CHILD`, and every signal this script sends goes to the GROUP
#    (`kill -TERM -- -PGID`, then `kill -KILL -- -PGID`), so grandchildren die with it. The
#    child command also closes fd 9 (`9>&-`), so no descendant can hold the build lock even
#    if one did escape. `timeout` is deliberately NOT given `--foreground`: in its default
#    mode it makes itself a process-group leader and its own expiry signal goes to the whole
#    group (`kill(0, sig)` in coreutils' cleanup()), which is what makes the wall-clock cap
#    reach grandchildren too. `set -m` is what guarantees that group exists from the instant
#    of the fork, rather than from whenever `timeout` gets round to its own setpgid() call,
#    and it makes the PGID knowable here (it is `$!`). Verified at runtime from /proc below.
#    NOTE (measured): because `timeout` is itself in that group, a group TERM reaches it too
#    and it relays a second TERM to the group. Jobs here are built for that (lab/specgen.py
#    and lab/_armwindow.py ignore a repeat signal once their cleanup has started); a group
#    signal is preferred over signalling `timeout` alone because it does not depend on any
#    coreutils version's relay behaviour.
# D2 GRACE. The TERM-to-KILL grace was 20s, below the jobs' own clean shutdown: a build step
#    in flight can take ~120s to return before the job even sees its abort flag, and the
#    job's own restore then cold-loads the resident (25-40s). `GPU_WINDOW_GRACE_SEC` now
#    governs it, default 180s, and is also passed to `timeout --kill-after` so the wall-clock
#    cap escalates on the same schedule.
# D3 RESTORE TARGET. The trap used to restore "whatever was active at entry", so a stray
#    maker-server at entry (it happens: a maker arm is per-build, never a resting state) made
#    the trap restart the MAKER after the job's own bookend had correctly started the
#    resident, leaving the household with no chat model. The trap now ALWAYS restores the
#    resident. `GPU_WINDOW_RESTORE=maker` is the documented escape hatch for a caller that
#    really wants the maker arm left up; nothing uses it today.
# D4 LOCK WAIT. The lock wait was a fixed 3600s, so a timer tick could spend an hour merely
#    queueing behind an interactive build while the next ticks pile up behind it.
#    `GPU_WINDOW_LOCK_WAIT_SEC` now governs it (default 3600, unchanged), and a run that
#    does not get the lock exits 75 (EX_TEMPFAIL) so a unit can log it as a skip.
#
# SIGKILL (kill -9) to THIS WRAPPER still cannot be handled by any shell: the traps never
# run, so the resident is never restored. The job itself no longer survives such a kill
# unnoticed (its PGID is in the holder line below, and it no longer holds fd 9), but stop
# this script with SIGTERM (plain `kill`, or Ctrl-C), never with -9.
#
# Every side-effecting command is env-overridable so the whole script can be tested against
# stubs with no GPU, no services and no real lock file (tests/test_gpu_window.py):
#   GPU_WINDOW_SYSTEMCTL   (default: systemctl)
#   GPU_WINDOW_NVIDIA_SMI  (default: nvidia-smi)
#   CAD_BUILD_LOCK_FILE    (default: ~/.openclaw/cad-build.lock; the SAME env var
#                           cad_v5/config.py's BUILD_LOCK_FILE honours, so a test or a probe
#                           that redirects one redirects both)
#   GPU_WINDOW_CURL        (default: curl)
#   GPU_WINDOW_PROXY_STATUS_URL (default: http://127.0.0.1:8087/)
#   GPU_WINDOW_DRAIN_SEC   (default: 120; 0 disables the drain entirely)
#
# D4 DRAIN BEFORE EVICTION (2026-09-20). Stopping the resident while it is mid-generation
# cut the client's connection, and llama.cpp does not exit on SIGTERM inside
# qwen38-server.service's TimeoutStopSec, so systemd SIGKILLed it and left the unit
# `failed` (cosmetic, the next start works) while the caller died with RemoteDisconnected.
# harvest.py's pre-flight could not see this: it reads only the gpu-proxy's `waiting`
# counter, which counts requests queued while the backend is DOWN and is therefore 0
# whenever the resident is up and busy. So AFTER the lock is held and BEFORE any unit is
# stopped, this script now polls the proxy's `active` counter and lets in-flight work
# finish, bounded by GPU_WINDOW_DRAIN_SEC. The drain fails OPEN on any error (proxy down,
# non-JSON, missing key): the build lock is the safety boundary, not this probe, and an
# unreachable proxy must never wedge the nightly window shut.
set -euo pipefail

SYSTEMCTL="${GPU_WINDOW_SYSTEMCTL:-systemctl}"
NVIDIA_SMI="${GPU_WINDOW_NVIDIA_SMI:-nvidia-smi}"
LOCK="${CAD_BUILD_LOCK_FILE:-$HOME/.openclaw/cad-build.lock}"
LOCK_WAIT_SEC="${GPU_WINDOW_LOCK_WAIT_SEC:-3600}"
GRACE_SEC="${GPU_WINDOW_GRACE_SEC:-180}"
MAX_SEC="${GPU_WINDOW_MAX_SEC:-36000}"
RESTORE_TO="${GPU_WINDOW_RESTORE:-resident}"
CURL="${GPU_WINDOW_CURL:-curl}"
PROXY_STATUS_URL="${GPU_WINDOW_PROXY_STATUS_URL:-http://127.0.0.1:8087/}"
DRAIN_SEC="${GPU_WINDOW_DRAIN_SEC:-120}"

mkdir -p "$(dirname "$LOCK")"
# Open append-only: `exec 9>"$LOCK"` truncated the file BEFORE flock returned, wiping the
# live holder's JSON line while this process was still queueing behind it (finding 3).
exec 9>>"$LOCK"
flock -w "$LOCK_WAIT_SEC" 9 || {
    echo "gpu_window: build lock still busy after ${LOCK_WAIT_SEC}s, skipping this run (exit 75)" >&2
    exit 75
}
# Only now that the lock is held may the holder line be replaced. fd 9 is O_APPEND, so every
# write below lands at the (post-truncation) end of file.
SPEC="$*"
# JSON-escape the spec so the holder line stays parseable for cad_engine's
# "waiting for build lock (held by: ...)" diagnostic even when the command contains quotes.
SPEC_JSON=${SPEC//\\/\\\\}; SPEC_JSON=${SPEC_JSON//\"/\\\"}
STARTED="$(date -Is)"
: > "$LOCK"
echo "{\"pid\": $$, \"child_pid\": null, \"frontend\": \"lab\", \"spec\": \"$SPEC_JSON\", \"started\": \"$STARTED\"}" >&9

MAKER_WAS=$("$SYSTEMCTL" --user is-active maker-server || true)

restore() {
    # A second SIGTERM/SIGINT must never abandon a restore half way through (the same rule
    # lab/_armwindow.py's _signal_handler follows), so they are ignored from here on.
    trap '' TERM INT HUP
    if [ "$RESTORE_TO" = "maker" ]; then
        echo "gpu_window: restore: maker-server was '${MAKER_WAS:-unknown}' at entry; GPU_WINDOW_RESTORE=maker, so starting maker-server" >&2
        "$SYSTEMCTL" --user start maker-server || echo "gpu_window: WARNING maker-server failed to restart" >&2
    else
        # The maker arm is per-build, never a resting state: whatever was up at entry, the
        # box's resting state is the resident (D3). maker-server and qwen38-server Conflict=
        # each other, so the stop is belt and braces rather than strictly required.
        echo "gpu_window: restore: maker-server was '${MAKER_WAS:-unknown}' at entry; restoring the resident qwen38-server" >&2
        "$SYSTEMCTL" --user stop maker-server 2>/dev/null || true
        "$SYSTEMCTL" --user start qwen38-server || echo "gpu_window: WARNING qwen38-server failed to restart, GPU has no resident" >&2
    fi
}

CHILD=""
CHILD_PGID=""
proc_pgid() {
    # Field 5 of /proc/<pid>/stat is the process group id. Field 2 (comm) can contain spaces
    # and parentheses, so everything up to the LAST ") " is dropped first; the remaining
    # fields are state, ppid, pgrp.
    local pid="$1" rest=""
    [ -r "/proc/$pid/stat" ] || return 1
    rest=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null) || return 1
    printf '%s\n' "$rest" | awk '{print $3}' | grep -E '^[0-9]+$' || return 1
}
SELF_PGID=$(proc_pgid $$ || true)

signal_job() {
    # Signal the whole process group when one was confirmed, never a bare PID: the bare PID
    # is `timeout`, and killing it leaves the real job orphaned on the GPU (D1).
    local sig="$1"
    if [ -n "$CHILD_PGID" ]; then
        kill -"$sig" -- "-$CHILD_PGID" 2>/dev/null || true
    else
        kill -"$sig" "$CHILD" 2>/dev/null || true
    fi
}

kill_job() {
    [ -n "$CHILD" ] || return 0
    if kill -0 "$CHILD" 2>/dev/null; then
        echo "gpu_window: signalled, sending TERM to the job's process group ${CHILD_PGID:-none (pid $CHILD only)}, grace ${GRACE_SEC}s" >&2
        signal_job TERM
        local i=0
        while [ "$i" -lt "$GRACE_SEC" ]; do
            kill -0 "$CHILD" 2>/dev/null || break
            sleep 1
            i=$((i + 1))
        done
        if kill -0 "$CHILD" 2>/dev/null; then
            echo "gpu_window: job still alive after ${GRACE_SEC}s, sending KILL to the process group" >&2
            signal_job KILL
        fi
    fi
    # Final sweep: the supervisor can be gone while a grandchild it never reaped is still on
    # the GPU. An empty group makes this a no-op.
    [ -n "$CHILD_PGID" ] && kill -KILL -- "-$CHILD_PGID" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
}

trap restore EXIT
trap 'kill_job; exit 143' TERM
trap 'kill_job; exit 130' INT
# HUP (dropped ssh, closed terminal): without this trap bash still runs the EXIT trap on the
# fatal signal, so the restore would start the resident ON TOP of a live job. nohup/systemd
# launches ignore HUP already; an interactive run does not.
trap 'kill_job; exit 129' HUP

proxy_active() {
    # Echoes the gpu-proxy's `active` count, or nothing at all when the answer cannot be
    # trusted (proxy unreachable, non-JSON body, key absent). "Nothing" is the fail-open
    # signal the caller treats as "stop waiting", never as "0 requests in flight".
    local body="" n=""
    body=$("$CURL" -sf -m 3 "$PROXY_STATUS_URL" 2>/dev/null) || return 0
    n=$(printf '%s' "$body" | grep -o '"active"[[:space:]]*:[[:space:]]*[0-9][0-9]*' | head -1 \
        | grep -o '[0-9][0-9]*$') || return 0
    [ -n "$n" ] && printf '%s\n' "$n"
    return 0
}

drain_resident() {
    # Bounded wait for in-flight resident work to finish before the eviction cuts it off
    # (D4). Never touches a service and never holds anything: the lock is already held, so
    # no new CAD or lab job can start behind our back, and a chat turn that arrives during
    # the drain simply extends it up to the bound.
    [ "$DRAIN_SEC" -gt 0 ] || { echo "gpu_window: drain disabled (GPU_WINDOW_DRAIN_SEC=0)" >&2; return 0; }
    local waited=0 n=""
    while [ "$waited" -lt "$DRAIN_SEC" ]; do
        n=$(proxy_active)
        if [ -z "$n" ]; then
            echo "gpu_window: drain: gpu-proxy status unreadable at ${PROXY_STATUS_URL}, proceeding (fail-open) after ${waited}s" >&2
            return 0
        fi
        if [ "$n" -eq 0 ]; then
            echo "gpu_window: drain: resident idle after ${waited}s, evicting" >&2
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "gpu_window: drain: ${n} request(s) still in flight after ${DRAIN_SEC}s, evicting anyway" >&2
    return 0
}

drain_resident
"$SYSTEMCTL" --user stop maker-server 2>/dev/null || true
"$SYSTEMCTL" --user stop qwen38-server || true
for i in $(seq 1 30); do used=$("$NVIDIA_SMI" --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 2; done
[ "$used" -lt 1500 ] || { echo "gpu_window: VRAM still ${used} MiB after eviction" >&2; exit 4; }
echo "== gpu_window: start $(date +%T) :: $SPEC"
# CAD_GPU_WINDOW=1 is what lab/ship.py's `verify` (and every other lab entry point) checks
# before it starts a GPU server. PYTHONUTF8=1 is not cosmetic: OpenCascade's Mesher().write()
# resets the process locale to C inside any Python process that builds geometry, so after the
# first build every unencoded read_text()/open() in that process decodes as ASCII; UTF-8 mode
# makes Python ignore the locale for file and subprocess text.
export CAD_GPU_WINDOW=1
export PYTHONUTF8=1
# Wall-clock cap (finding 4), the same dead-man idea as maker-server.service's RuntimeMaxSec:
# TERM at GPU_WINDOW_MAX_SEC (default 10h), KILL one grace later. Exit code 124 = timed out
# (137 when the job ignored the TERM and coreutils' own KILL took the whole group down,
# `timeout` included).
# `set -m` puts the job in its own process group with PGID == $! (D1); `9>&-` keeps the build
# lock out of the job's file descriptors; `</dev/null` preserves the stdin a background job
# gets without job control.
set -m
timeout --signal=TERM --kill-after="$GRACE_SEC" "$MAX_SEC" "$@" </dev/null 8>&- 9>&- &
CHILD=$!
set +m
CHILD_PGID=$(proc_pgid "$CHILD" || true)
if [ -z "$CHILD_PGID" ] || [ "$CHILD_PGID" = "$SELF_PGID" ]; then
    # Never signal our own process group: that would take this wrapper (and its restore)
    # down with the job. Fall back to the single PID and say so.
    echo "gpu_window: WARNING could not confirm a separate process group for the job (pgid '${CHILD_PGID:-unknown}', mine '${SELF_PGID:-unknown}'); signals will go to pid $CHILD only" >&2
    CHILD_PGID=""
fi
# Rewrite the holder line now that the job exists: a SIGKILLed wrapper leaves this line as
# the only record of which process group is still holding the GPU (see the header note).
: > "$LOCK"
echo "{\"pid\": $$, \"child_pid\": $CHILD, \"pgid\": ${CHILD_PGID:-null}, \"frontend\": \"lab\", \"spec\": \"$SPEC_JSON\", \"started\": \"$STARTED\"}" >&9
rc=0
wait "$CHILD" || rc=$?
echo "== gpu_window: done $(date +%T) (exit $rc)"
exit "$rc"

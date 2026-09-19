#!/usr/bin/env bash
# lab/harvest_unit.sh -- the timer's own launcher for the Phase 3 harvest unit (Task 3).
#
# Serialises concurrent timer fires with a non-blocking flock on lab/state/harvest.lock:
# if a previous unit is somehow still running when the next OnCalendar tick lands, this
# one exits immediately (exit 0, not an error) rather than queueing behind it -- the next
# tick, 30 minutes later, tries again. This script does NOT touch
# ~/.openclaw/cad-build.lock itself; it CALLS lab/gpu_window.sh (frozen, unmodified) and
# lets that be the one thing that ever HOLDS the CAD build lock, exactly like every other
# GPU-heavy lab entry point (lab/specgen.py, lab/ship.py verify). No second locking
# scheme (Task 3 ruling).
#
# Fix round 1 (2026-09-19, H4): `--check-gate` runs the cheap paused/night-window/budget/
# gpu-proxy/bank-exhausted/build-lock checks WITHOUT a GPU window first -- a tick that is
# going to skip anyway (the overwhelming majority, since the timer used to fire all day
# and the window is only 22:00-07:00) no longer pays for a full resident eviction and
# cold arm restart just to find that out. Only a --check-gate exit 0 proceeds to the real
# GPU window, where --unit re-runs the same gates (minus the lock probe, which would be
# circular there: this process's own ancestor already holds the lock by that point).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$HERE/lab/state/harvest.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "harvest_unit: a unit is already running, skipping this tick" >&2; exit 0; }
if ! python3 "$HERE/lab/harvest.py" --check-gate; then
    exit 0   # quiet skip; --check-gate already printed the reason to stderr
fi
exec "$HERE/lab/gpu_window.sh" python3 "$HERE/lab/harvest.py" --unit

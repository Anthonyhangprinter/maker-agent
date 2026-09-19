#!/usr/bin/env bash
# lab/harvest_unit.sh -- the timer's own launcher for the Phase 3 harvest unit (Task 3).
#
# Serialises concurrent timer fires with a non-blocking flock on lab/state/harvest.lock:
# if a previous unit is somehow still running when the next OnCalendar tick lands, this
# one exits immediately (exit 0, not an error) rather than queueing behind it -- the next
# tick, 30 minutes later, tries again. This script does NOT touch
# ~/.openclaw/cad-build.lock itself; it CALLS lab/gpu_window.sh (frozen, unmodified) and
# lets that be the one thing that ever takes the CAD build lock, exactly like every other
# GPU-heavy lab entry point (lab/specgen.py, lab/ship.py verify). No second locking
# scheme (Task 3 ruling).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$HERE/lab/state/harvest.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "harvest_unit: a unit is already running, skipping this tick" >&2; exit 0; }
exec "$HERE/lab/gpu_window.sh" python3 "$HERE/lab/harvest.py" --unit

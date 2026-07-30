#!/usr/bin/env bash
# Free A/B: does removing the qwen3:8b brief hurt the LOCAL 7B in production?
#
# The teacher clearly does better without it, but the live loop runs a 7B, and the brief was
# built FOR a small coder. Measure before trusting the change on a working tool. Same engine,
# same suite, same coder rung — only cad.use_brief differs.
#
#   bash scripts/brief_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CAD_JSON="$HOME/.openclaw/cad.json"
BACKUP="$(mktemp)"
cp "$CAD_JSON" "$BACKUP"
# Always put the user's config back, however this exits.
trap 'cp "$BACKUP" "$CAD_JSON"; echo "[ab] restored $CAD_JSON"' EXIT

CLOUD=$(python3 -c "import json,pathlib;d=json.loads(pathlib.Path('$CAD_JSON').read_text());print(json.dumps(d.get('cloud',{})))")

run_leg () {                       # $1 = true|false, $2 = label
  python3 - "$1" <<'PY'
import json, pathlib, sys, os
p = pathlib.Path(os.path.expanduser("~/.openclaw/cad.json"))
d = json.loads(p.read_text())
d["use_brief"] = (sys.argv[1] == "true")
p.write_text(json.dumps(d, indent=1))
print(f"  cad.use_brief = {d['use_brief']}")
PY
  echo "[ab] === leg: $2 (use_brief=$1) ==="
  python3 scripts/run_benchmarks.py --tiers 1,2 --coder fast --timeout 1500 2>&1 \
    | grep -E "^\[|acceptance|converged|Benchmark|=== " | tail -25
}

echo "[ab] cloud block preserved: $CLOUD"
run_leg true  "WITH brief (legacy)"
sleep 45                            # let the GPU settle between legs
run_leg false "BRIEF-LESS (new default)"

echo
echo "[ab] compare the two newest runs:"
python3 scripts/compare_runs.py --latest 2 2>&1 | tail -20

#!/usr/bin/env bash
# setup_cq_env.sh -- build an ISOLATED venv for the CadQuery arm of the lang-ab harness.
#
# Why isolated: the live CAD engine (cad_engine.py, the whole cad-builder production path)
# depends on the SYSTEM python's build123d 0.10.0 + OCP 7.8.1.1. `pip install --user
# cadquery` (or any system/user-wide install) could pull in a different OCP build and
# silently break every other CAD build on this box. cadquery vendors its own OCP wheel, so
# it must live in a venv that `uv` never lets touch site-packages.
#
# Usage:
#   bash benchmarks/lang-ab/setup_cq_env.sh
#
# What it does, in order:
#   1. records the SYSTEM python3's build123d/OCP versions (before)
#   2. creates benchmarks/lang-ab/.venv-cq with `uv venv --python 3.12`
#   3. installs cadquery 2.8.x into THAT venv only
#   4. verifies .venv-cq/bin/python can `import cadquery`
#   5. re-checks the SYSTEM python3's build123d/OCP versions are unchanged (after) and
#      refuses (non-zero exit) if they moved
#
# Network access is required (step 3 downloads from PyPI). Safe to re-run: `uv venv`
# recreates the directory and the version check is idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-cq"

if ! command -v uv >/dev/null 2>&1; then
    echo "setup_cq_env.sh: uv not found on PATH" >&2
    exit 1
fi

echo "== recording system build123d/OCP versions before touching anything =="
BEFORE="$(python3 -c 'import build123d, OCP; print(build123d.__version__, OCP.__version__)' 2>&1)" || {
    echo "setup_cq_env.sh: WARNING: system python3 could not import build123d/OCP before setup: $BEFORE" >&2
    BEFORE=""
}
echo "before: ${BEFORE:-<unavailable>}"

echo "== creating isolated venv at $VENV (python 3.12) =="
uv venv --python 3.12 "$VENV"

echo "== installing cadquery (2.8.x) into the isolated venv only =="
uv pip install --python "$VENV/bin/python" "cadquery>=2.8,<2.9"

echo "== verifying the isolated venv can import cadquery =="
"$VENV/bin/python" -c "import cadquery; print('cadquery', cadquery.__version__)"

echo "== verifying the SYSTEM build123d/OCP install is unchanged =="
AFTER="$(python3 -c 'import build123d, OCP; print(build123d.__version__, OCP.__version__)')"
echo "after:  $AFTER"
if [ -n "$BEFORE" ] && [ "$BEFORE" != "$AFTER" ]; then
    echo "setup_cq_env.sh: REFUSING: system build123d/OCP changed ($BEFORE -> $AFTER)." >&2
    echo "  This should be impossible (uv venv never touches site-packages), but the check" >&2
    echo "  exists because a broken production CAD engine is a much worse outcome than a" >&2
    echo "  failed setup script. Investigate before using this venv." >&2
    exit 1
fi

echo "setup_cq_env.sh: OK. isolated venv at $VENV, system build123d/OCP unchanged ($AFTER)."

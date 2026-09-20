#!/usr/bin/env python3
"""Design assistant CLI — the pre-build "Parameters" panel, run as its own subprocess so
the web UI's tiny FastAPI venv never has to import cad_engine (same reasoning as
fluid_gen.py/openscad_gen.py: this runs under the SYSTEM interpreter).

Prints ONE JSON line: {"proposed": bool, "mode": "off"|"auto"|"always", "reason": str?,
"title": str, "summary": str, "parameters": [...], "features": [...]}. "off" and an
auto-mode spec that triage_ambiguity finds buildable-as-typed both return proposed=false
with no model call for the parameters themselves — the caller (webui/app.py) then just
builds the spec unchanged.

    python3 scripts/assist_gen.py "<spec>" --mode auto --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cad_engine as engine  # noqa: E402
from cad_v5 import design_assistant as da  # noqa: E402


def run(spec: str, mode_req: str) -> dict:
    mode = da.design_assistant_mode(mode_req or None)
    empty = {"proposed": False, "mode": mode, "title": "", "summary": "",
             "parameters": [], "features": []}
    if not spec.strip():
        return {**empty, "reason": "empty spec"}
    if mode == "off":
        return {**empty, "reason": "design assistant is off"}
    if mode == "auto":
        try:
            questions = engine.triage_ambiguity(spec)
        except Exception as e:
            questions = []
            print(f"[assist] triage failed ({e}) — treating as buildable as typed",
                  file=sys.stderr)
        if not questions:
            return {**empty, "reason": "spec looks buildable as typed"}
    proposal = da.propose_parameters(spec)
    proposed = bool(proposal.get("parameters"))
    out = {"proposed": proposed, "mode": mode, **proposal}
    if not proposed:
        out["reason"] = "no usable proposal"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--mode", default="", choices=["", "off", "auto", "always"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    result = run(a.spec, a.mode)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

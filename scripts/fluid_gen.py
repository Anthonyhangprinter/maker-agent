#!/usr/bin/env python3
"""Fluid mode — conversational CAD with NO gate vetoes (user direction 2026-08-03).

The verification gate exists for unattended runs (benchmarks, training harvests) where
nothing else can catch a bad build. In a live conversation the HUMAN is the gate: this
driver does ONE codegen (or one revise from the user's words), builds it, renders it, and
returns whatever happened — measurements included as information, never as a veto. Fast
(~single model call + build) so the chat loop stays fluid.

    fluid_gen.py build "a 40mm cube with a 10mm hole" [--coder fast|strong|cloud] --json
    fluid_gen.py revise <build_dir> "make the walls 3mm and the hole bigger" --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cad_engine as engine  # noqa: E402

BUILDS_DIR = Path.home() / ".openclaw" / "cad-builds"


def _model_for(coder: str) -> None:
    if coder == "strong":
        engine._ACTIVE_CODE_MODEL = engine.CODE_MODEL_STRONG
    elif coder == "cloud":
        cc = engine.cloud_config()
        engine._ACTIVE_CODE_MODEL = engine.CLOUD_PREFIX + cc.get("model", "claude-sonnet-5")
        engine.reset_cloud_budget(2)
    else:
        engine._ACTIVE_CODE_MODEL = engine.CODE_MODEL_FAST


def _materialize(code: str, build_dir: Path) -> dict:
    """Run + render + measure. Never vetoes — failures are reported, not fatal."""
    out = {"facts": {}, "instruments": [], "error": None}
    (build_dir / "build_source.py").write_text(code)
    try:
        step, _ = engine.run_step(code, build_dir)
        target = build_dir / "build.step"
        if Path(step) != target:
            target.write_bytes(Path(step).read_bytes())
    except Exception as e:
        # The exception LINE lives at the END of a traceback — keep the tail, not the head.
        msg = str(e).strip()
        out["error"] = "the script failed to run: " + \
            ("…" + msg[-380:] if len(msg) > 400 else msg)
        return out
    try:
        insp = engine.run_inspect(build_dir / "build.step")
        if insp["valid"]:
            out["facts"] = engine.parse_facts(insp["output"])
            # Instruments, not judges: same measurements the gate uses, shown as info.
            hard, notes = engine.verify_expected(out["facts"], {}, spec="")
            out["instruments"] = (hard or []) + (notes or [])
    except Exception:
        pass
    try:
        subprocess.run(["/usr/bin/python3", str(HERE / "scripts" / "render"),
                        str(build_dir / "build.step"), str(build_dir / "build.png")],
                       timeout=120, capture_output=True)
    except Exception:
        pass
    try:
        from build123d import Mesher, import_step
        shape = import_step(str(build_dir / "build.step"))
        m2 = Mesher()
        m2.add_shape(shape)
        m2.write(str(build_dir / "build.stl"))
    except Exception:
        pass
    return out


def _materialize_with_salvage(spec: str, code: str, build_dir: Path) -> dict:
    """One build attempt + ONE automatic crash-salvage turn. A runtime error is the local
    coder's dominant failure mode; a chat that dead-ends every other message isn't fluid."""
    m = _materialize(code, build_dir)
    if m["error"]:
        try:
            fixed = engine.revise_script(spec, code, m["error"])
            m2 = _materialize(fixed, build_dir)
            if not m2["error"]:
                m2["salvaged"] = True
                return m2
        except Exception:
            pass
    return m


def cmd_build(a) -> dict:
    _model_for(a.coder)
    t0 = time.monotonic()
    # Correct-by-construction first: bolts/gears the user fully pinned down never need an
    # LLM at all (spec_helper is deterministic from the spec text; generate_code
    # materializes the helper call with its imports and no model call).
    helper = engine.spec_helper(a.spec)
    if helper:
        code = engine.generate_code({"helper": helper, "notes": [], "expected": {}}, a.spec)
    else:
        notes = engine.retrieval_notes_for(a.spec)
        code = engine.generate_code_raw(a.spec, notes)
    build_dir = BUILDS_DIR / f"fluid_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "fluid.json").write_text(json.dumps(
        {"spec": a.spec, "coder": a.coder, "history": []}))
    m = _materialize_with_salvage(a.spec, code, build_dir)
    return {"ok": m["error"] is None, "mode": "fluid", "build_dir": str(build_dir),
            "code_model": engine._code_model(), "facts": m["facts"],
            "instruments": m["instruments"][:6], "error": m["error"],
            "build_time_s": round(time.monotonic() - t0, 1)}


def cmd_revise(a) -> dict:
    build_dir = Path(a.build_dir)
    src = build_dir / "build_source.py"
    meta_f = build_dir / "fluid.json"
    if not src.is_file():
        return {"ok": False, "error": "no build_source.py in that build dir"}
    meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {"spec": "", "history": []}
    _model_for(a.coder or meta.get("coder", "fast"))
    t0 = time.monotonic()
    code = src.read_text()
    state = ""
    try:
        insp = engine.run_inspect(build_dir / "build.step")
        if insp["valid"]:
            state = insp["output"][:1500]
    except Exception:
        pass
    # The user's words ARE the problem statement — verbatim, no paraphrase.
    new_code = engine.revise_script(meta.get("spec", ""), code, a.feedback, state=state)
    m = _materialize_with_salvage(meta.get("spec", ""), new_code, build_dir)
    meta["history"].append({"feedback": a.feedback,
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "ok": m["error"] is None})
    meta_f.write_text(json.dumps(meta))
    return {"ok": m["error"] is None, "mode": "fluid", "build_dir": str(build_dir),
            "code_model": engine._code_model(), "facts": m["facts"],
            "instruments": m["instruments"][:6], "error": m["error"],
            "turns": len(meta["history"]) + 1,
            "build_time_s": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("spec")
    b.add_argument("--coder", default="fast", choices=["fast", "strong", "cloud"])
    b.add_argument("--json", action="store_true")
    r = sub.add_parser("revise")
    r.add_argument("build_dir")
    r.add_argument("feedback")
    r.add_argument("--coder", default="")
    r.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = cmd_build(a) if a.cmd == "build" else cmd_revise(a)
    print(json.dumps(res))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

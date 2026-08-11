#!/usr/bin/env python3
"""Fluid mode — conversational CAD with gates ON but the human in charge (2026-08-11).

Originally fluid mode blindfolded the gate (verify_expected called with an empty spec, so
only 4 of ~28 checks could fire). The 2026-08-11 direction: conversationality and
measurement are independent axes. This driver keeps the one-turn chat flow — ONE codegen
(or one revise from the user's words), build, render, return — but the gate now measures
against the REAL working spec, severity preserved:

- initial build: hard fails / [spec] contradictions earn ONE automatic repair turn; if the
  repair doesn't improve, the original build is restored (artifacts are always kept — the
  user is the final critic, honesty gates the claim, never the output)
- revise turns: gates measure and display only, never repair — the user's feedback may
  deliberately override the stored spec, and an auto-repair re-asserting old words would
  fight the user
- expansion rung: a vague spec (per triage_ambiguity) gets its missing parameterization
  written by the strong model BEFORE codegen; the expansion becomes the working spec, its
  assumptions are surfaced for correction, and the [spec] checks enforce its numbers

There is still no convergence verdict and no multi-turn loop — the chat stays fluid.

    fluid_gen.py build "a propeller" [--coder fast|strong|cloud] [--image photo.jpg] --json
    fluid_gen.py revise <build_dir> "make it two blades" --json
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cad_engine as engine  # noqa: E402
from cad_v5.diagnose import diagnose  # noqa: E402

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


def _materialize(code: str, build_dir: Path, spec: str = "") -> dict:
    """Run + render + measure. Failures are reported, not fatal. The gate runs against the
    real working spec (gates ON, 2026-08-11): hard/[spec]/advisory severity is preserved so
    callers and the UI can treat them differently. forbid_blind_holes is the one
    expected-field derivable without a brief (deterministic from spec text)."""
    out = {"facts": {}, "instruments": [], "gate_hard": [], "gate_spec": [],
           "gate_adv": [], "error": None}
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
            expected = ({"forbid_blind_holes":
                         not bool(engine._BLIND_HOLE_TERMS.search(spec))} if spec else {})
            hard, notes = engine.verify_expected(out["facts"], expected, spec=spec)
            out["gate_hard"] = hard or []
            out["gate_spec"] = [n for n in (notes or []) if n.startswith("[spec]")]
            out["gate_adv"] = [n for n in (notes or []) if not n.startswith("[spec]")]
            out["instruments"] = out["gate_hard"] + (notes or [])
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


def _materialize_with_salvage(spec: str, code: str, build_dir: Path,
                              gate_repair: bool = True) -> dict:
    """One build attempt + at most ONE automatic recovery turn. Two triggers:
    - crash salvage (always): a runtime error is the local coder's dominant failure mode;
      a chat that dead-ends every other message isn't fluid
    - gate repair (initial non-helper builds only): hard fails / [spec] contradictions get
      one revise. The repair is kept only if it IMPROVES (fewer findings, still builds);
      otherwise the original artifacts are restored — never ship the regression."""
    m = _materialize(code, build_dir, spec)
    if m["error"]:
        try:
            # Same failure taxonomy the full loop uses — a fillet crash gets the targeted
            # "wrap it in try/except, don't repeat the call" hint, not just the traceback.
            _, hint = diagnose(m["error"])
            problem = m["error"] + (f"\nRepair hint: {hint}" if hint else "")
            fixed = engine.revise_script(spec, code, problem)
            m2 = _materialize(fixed, build_dir, spec)
            if not m2["error"]:
                m2["salvaged"] = True
                return m2
        except Exception:
            pass
        return m
    findings = m["gate_hard"] + m["gate_spec"]
    if gate_repair and findings:
        # Snapshot the working build first — _materialize overwrites the dir in place.
        snap = {}
        for name in ("build_source.py", "build.step", "build.stl", "build.png"):
            p = build_dir / name
            if p.is_file():
                snap[name] = p.read_bytes()
        try:
            problem = ("Deterministic measurements of the built solid contradict the "
                       "request (authoritative — they measure the actual geometry):\n- "
                       + "\n- ".join(findings))
            fixed = engine.revise_script(spec, code, problem)
            m2 = _materialize(fixed, build_dir, spec)
            if not m2["error"] and \
                    len(m2["gate_hard"] + m2["gate_spec"]) < len(findings):
                m2["gate_repaired"] = True
                return m2
        except Exception:
            pass
        for name, data in snap.items():
            try:
                (build_dir / name).write_bytes(data)
            except Exception:
                pass
    return m


def _result(m: dict, extra: dict, t0: float) -> dict:
    res = {"ok": m["error"] is None, "mode": "fluid",
           "code_model": engine._code_model(), "facts": m["facts"],
           "instruments": m["instruments"][:6],
           "gate_hard": m["gate_hard"], "gate_spec": m["gate_spec"],
           "gate_adv": m["gate_adv"][:6],
           "salvaged": m.get("salvaged", False),
           "gate_repaired": m.get("gate_repaired", False),
           "error": m["error"],
           "build_time_s": round(time.monotonic() - t0, 1)}
    res.update(extra)
    return res


def cmd_build(a) -> dict:
    _model_for(a.coder)
    t0 = time.monotonic()
    spec, expansion, image_only = a.spec or "", None, False
    if a.image:
        # Same vision pre-pass the full loop uses (gemma coexists with the resident
        # server; cached per-photo). Image-only: the analysis IS the request.
        analysis = engine.analyze_reference_image(a.image)
        if not spec:
            spec = engine.spec_from_image(analysis) or "the part in the reference image"
            image_only = True
        if analysis:
            spec = spec + "\n" + engine.image_analysis_text(analysis, image_only=image_only)
    helper = engine.spec_helper(spec)
    if helper:
        code = engine.generate_code({"helper": helper, "notes": [], "expected": {}}, spec)
    else:
        if not a.image:
            # Expansion rung: a vague spec fails at parameterization, not geometry — the
            # strong model decides the missing criticals BEFORE codegen and states them as
            # correctable assumptions. Runs in the pre-codegen window while the resident
            # server is still up (codegen on the fast rung evicts it). Image builds skip
            # this — the vision analysis already plays the expansion's role.
            questions = engine.triage_ambiguity(spec)
            if questions:
                expansion = engine.expand_spec(spec, questions)
                if expansion:
                    spec = expansion["expanded_spec"]
        notes = engine.retrieval_notes_for(spec, use_fewshots=not a.no_fewshots)
        code = engine.generate_code_raw(spec, notes)
    build_dir = BUILDS_DIR / f"fluid_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    build_dir.mkdir(parents=True, exist_ok=True)
    if a.image:
        try:  # downscaled copy so the UI can show what conditioned the build
            (build_dir / "reference.jpg").write_bytes(
                base64.b64decode(engine._prep_image_b64(Path(a.image))))
        except Exception:
            pass
    meta = {"spec": spec, "user_spec": a.spec or "", "coder": a.coder, "history": []}
    if expansion:
        meta["assumptions"] = expansion["assumptions"]
    (build_dir / "fluid.json").write_text(json.dumps(meta))
    # Helper builds are correct by construction — a 7B "repair" of a bd_warehouse call
    # could only degrade it, so gate findings there are display-only.
    m = _materialize_with_salvage(spec, code, build_dir, gate_repair=not helper)
    extra = {"build_dir": str(build_dir)}
    if expansion:
        extra["expanded_spec"] = expansion["expanded_spec"]
        extra["assumptions"] = expansion["assumptions"]
    if image_only:
        extra["image_only"], extra["spec"] = True, spec
    return _result(m, extra, t0)


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
    spec = meta.get("spec", "")
    # The user's words ARE the problem statement — verbatim, no paraphrase.
    new_code = engine.revise_script(spec, code, a.feedback, state=state)
    # Gate display measures against the spec + every revision so far (latest words
    # included). gate_repair stays OFF: feedback may deliberately contradict the stored
    # spec, and an auto-repair re-asserting the old words would fight the user.
    gate_text = spec + "".join(
        f"\nRevision: {h['feedback']}" for h in meta.get("history", [])) \
        + f"\nRevision: {a.feedback}"
    m = _materialize_with_salvage(gate_text, new_code, build_dir, gate_repair=False)
    meta.setdefault("history", []).append(
        {"feedback": a.feedback, "ts": datetime.now(timezone.utc).isoformat(),
         "ok": m["error"] is None})
    meta_f.write_text(json.dumps(meta))
    extra = {"build_dir": str(build_dir), "turns": len(meta["history"]) + 1}
    if meta.get("assumptions"):
        extra["assumptions"] = meta["assumptions"]
    return _result(m, extra, t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("spec", nargs="?", default="")
    b.add_argument("--coder", default="fast", choices=["fast", "strong", "cloud"])
    b.add_argument("--image", default=None)
    b.add_argument("--no-fewshots", action="store_true")
    b.add_argument("--json", action="store_true")
    r = sub.add_parser("revise")
    r.add_argument("build_dir")
    r.add_argument("feedback")
    r.add_argument("--coder", default="")
    r.add_argument("--image", default=None)        # accepted for frontend symmetry; the
    r.add_argument("--no-fewshots", action="store_true")  # reference already lives in the dir
    r.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        res = cmd_build(a) if a.cmd == "build" else cmd_revise(a)
    finally:
        # Fluid runs bypass engine.build(), so the default-server resume (the 35B
        # evicted to make VRAM room for the fast coder) must happen here too.
        engine._resume_default_server()
    print(json.dumps(res))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

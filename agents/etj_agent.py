#!/usr/bin/env python3
"""etj_agent.py — benchmark adapter that drives a local Ollama model through the
earthtojake/text-to-cad plugin's own workflow (2026-07-17, user request).

Loop: model writes a `gen_step()` build123d source (per the plugin's SKILL.md
contract) → the plugin's `scripts/step` generates/validates the STEP → the
plugin's `scripts/inspect refs --facts` reports measured geometry → the model
revises or declares DONE. No brief stage, no few-shot retrieval, no visual
critic, no deterministic gate — this measures model + earthtojake toolchain,
as a second, independent pipeline next to cad_engine's.

Runner contract (run_benchmarks.py --agent <this file>):
    python3 etj_agent.py build "<spec>" --coder fast|strong|auto|<model> [--no-upload] [--no-fewshots]
prints exactly one JSON object on stdout. Child-tool output is captured, never
echoed (parse_result takes the first JSON-looking line).

Model pin: ~/.openclaw/cad.json {"code_model": ...} wins over --coder, same as
cad_engine, so scripts/run_refresh.sh stages work unchanged.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

ETJ = Path.home() / "repos" / "text-to-cad" / "skills" / "cad"
CADJSON = Path.home() / ".openclaw" / "cad.json"
BUILDS = Path.home() / ".openclaw" / "cad-builds"
OLLAMA = "http://localhost:11434/api/generate"
FAST, STRONG = "qwen3:8b", "qwen3-coder:30b"
MAX_TURNS = 4
LLM_TIMEOUT = 1200

SYSTEM = """You are a mechanical CAD engineer generating parametric parts with build123d
(Python). Follow the earthtojake/text-to-cad source contract exactly:

- Define `def gen_step():` that builds the part and RETURNS the final solid or
  labeled Compound. No file I/O, no show(), no prints, no output paths.
- Units are millimeters. Base plane XY, extrusion/up axis +Z. Origin at the
  center of the part unless the spec says otherwise.
- Output must be closed, positive-volume solid geometry: one solid, a compound
  of solids, or a labeled assembly compound.
- Defaults when unspecified: small enclosure walls 2.0-3.0 mm; cosmetic fillets
  1.0-3.0 mm only where locally safe; M3/M4/M5 clearance holes 3.4/4.5/5.5 mm.
- Import from build123d; use algebra mode (e.g. `part = Box(...) - Cylinder(...)`)
  or builder mode, whichever fits.

Reply with ONE ```python code block containing the complete source, nothing else."""

FEEDBACK_TMPL = """The toolchain ran your source. Result:

{report}

If the geometry satisfies the spec, reply exactly DONE.
Otherwise reply with ONE corrected complete ```python code block (full source, not a diff)."""


def resolve_model(coder: str) -> str:
    try:
        pin = json.loads(CADJSON.read_text()).get("code_model")
        if pin:
            return pin
    except Exception:
        pass
    return {"fast": FAST, "strong": STRONG, "auto": FAST}.get(coder, coder)


def ollama(model: str, prompt: str, system: str = SYSTEM) -> str:
    payload = {"model": model, "stream": False, "system": system, "prompt": prompt,
               "options": {"num_ctx": 16384}, "think": False}
    req = urllib.request.Request(OLLAMA, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        return json.loads(r.read())["response"]


def extract_code(reply: str) -> str | None:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    if m:
        return m[-1].strip()
    return reply.strip() if "def gen_step" in reply else None


def run_tool(args: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                       cwd=cwd, timeout=600)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def gen_and_inspect(src: Path, step: Path, cwd: Path) -> tuple[bool, str]:
    """Run the plugin's step generation; on success append its geometry facts."""
    rc, out = run_tool([str(ETJ / "scripts" / "step"), f"{src.name}={step.name}"], cwd)
    if rc != 0 or not step.exists() or step.stat().st_size == 0:
        return False, f"STEP GENERATION FAILED (rc={rc}):\n{out[-1500:]}"
    rc2, facts = run_tool([str(ETJ / "scripts" / "inspect"), "refs", step.name,
                           "--facts", "--planes", "--positioning"], cwd)
    report = f"STEP generated OK: {step.name}\n\nGEOMETRY FACTS:\n{facts[-2500:]}" \
        if rc2 == 0 else f"STEP generated OK but inspect failed (rc={rc2}):\n{facts[-800:]}"
    return True, report


SELFTEST_CODE = """from build123d import *

def gen_step():
    part = Box(40, 20, 10) - Cylinder(radius=3, height=20)
    return part
"""


def build(spec: str, coder: str, selftest: bool = False) -> dict:
    model = resolve_model(coder)
    t0 = time.monotonic()
    slug = re.sub(r"[^a-z0-9]+", "-", spec.lower())[:40].strip("-")
    bdir = BUILDS / f"{datetime.now():%Y%m%d-%H%M%S}-etj-{slug}"
    bdir.mkdir(parents=True, exist_ok=True)
    src, step = bdir / "gen_source.py", bdir / "build.step"

    turns, converged, ok, last_report, error = 0, False, False, "", None
    prompt = f"Specification:\n{spec}"
    for turn in range(1, MAX_TURNS + 1):
        turns = turn
        reply = SELFTEST_CODE if selftest else ollama(model, prompt)
        if not selftest and turn > 1 and reply.strip().upper().startswith("DONE"):
            converged = True
            break
        code = SELFTEST_CODE if selftest else extract_code(reply)
        if not code:
            last_report = "Your reply contained no python code block. Reply with ONE ```python block."
            prompt = f"Specification:\n{spec}\n\n{last_report}"
            continue
        src.write_text(code)
        ok, last_report = gen_and_inspect(src, step, bdir)
        if selftest:
            converged = ok
            break
        prompt = (f"Specification:\n{spec}\n\nYour current source:\n```python\n{code}\n```\n\n"
                  + FEEDBACK_TMPL.format(report=last_report))
    if not ok:
        error = last_report[-300:]

    return {
        "ok": ok, "converged": converged,
        "accepted_via": "etj-facts" if converged else None,
        "code_model": model if ok else None,
        "agent": "etj", "turns": turns,
        "build_time_s": round(time.monotonic() - t0, 1),
        "step_local": str(step) if step.exists() else None,
        "build_dir": str(bdir),
        "failure_categories": {} if ok else {"etj_pipeline": 1},
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "selftest"])
    ap.add_argument("spec", nargs="?", default="a 40x20x10mm block with a 6mm through hole")
    ap.add_argument("--coder", default="auto")
    ap.add_argument("--no-upload", action="store_true")   # runner passes it; nothing to do
    ap.add_argument("--no-fewshots", action="store_true") # this pipeline has no retrieval
    a = ap.parse_args()
    res = build(a.spec, a.coder, selftest=(a.cmd == "selftest"))
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

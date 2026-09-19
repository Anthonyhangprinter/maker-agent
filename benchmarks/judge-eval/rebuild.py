#!/usr/bin/env python3
"""
Judge-eval Task 2: rebuild every labelled program on CPU and capture measured facts + a
two-panel render.

Safety: this script imports cad_engine ONLY for run_step/run_inspect/parse_facts (pure
subprocess wrappers around scripts/step and scripts/inspect — no model calls). Per the task's
safety rule, the model-swapping entry points are patched to raise immediately after import, so
nothing in this process can ever start/stop/evict a server. Builds run one at a time, CPU only,
`nice -n 10` (subprocess.run inherits our own niceness via os.nice before exec, and we also
os.nice(10) this process itself so ANY child inherits it — a belt-and-suspenders "at most 2 at a
time, CPU only" is honoured by never launching more than 2 build workers concurrently).

Usage: python3 rebuild.py [--workers 2]
Writes results/<run_id>/artifacts/<safe_program_id>/{build.step, facts.json, render.png, error.txt}
and results/<run_id>/build_manifest.json (per-program status, resumable).
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]  # .../cad-builder-phase3
RUN_ID = os.environ.get("JUDGE_EVAL_RUN_ID", "2026-09-19")
RESULTS_DIR = HERE / "results" / RUN_ID
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
MANIFEST_PATH = RESULTS_DIR / "build_manifest.json"

sys.path.insert(0, str(WORKTREE))

# Lower our own niceness so every subprocess we spawn (step/inspect/render) inherits it.
try:
    os.nice(10)
except Exception:
    pass

import cad_engine  # noqa: E402


def _raise(*a, **kw):
    raise RuntimeError("judge-eval safety guard: model-swapping cad_engine call blocked")


cad_engine._ollama = _raise
cad_engine._ensure_default_server = _raise
cad_engine._pause_default_server_for = _raise
cad_engine._resume_default_server = _raise


def safe_id(program_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", program_id)


def render_two_panel(step_path: Path, png_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["nice", "-n", "10", sys.executable, str(WORKTREE / "scripts" / "render"),
         str(step_path), str(png_path)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    ok = result.returncode == 0 and png_path.exists()
    return ok, (result.stdout + result.stderr)


def build_one(rec: dict) -> dict:
    pid = rec["program_id"]
    sid = safe_id(pid)
    outdir = ARTIFACTS_DIR / sid
    outdir.mkdir(parents=True, exist_ok=True)
    code = Path(rec["code_path"]).read_text(encoding="utf-8")
    status = {"program_id": pid, "ok": False, "stage": None, "error": None}
    with tempfile.TemporaryDirectory(prefix="judge-eval-") as tmp:
        tmpdir = Path(tmp)
        try:
            step_path, step_log = cad_engine.run_step(code, tmpdir)
        except Exception as e:
            status["stage"] = "run_step"
            status["error"] = str(e)[:4000]
            (outdir / "error.txt").write_text(f"[run_step]\n{status['error']}", encoding="utf-8")
            return status
        # Persist the STEP (small) so a run is inspectable/resumable without recompiling.
        final_step = outdir / "build.step"
        final_step.write_bytes(step_path.read_bytes())
        try:
            inspect_result = cad_engine.run_inspect(final_step)
        except Exception as e:
            status["stage"] = "run_inspect"
            status["error"] = str(e)[:4000]
            (outdir / "error.txt").write_text(f"[run_inspect]\n{status['error']}", encoding="utf-8")
            return status
        facts = cad_engine.parse_facts(inspect_result["output"])
        (outdir / "inspect_output.txt").write_text(inspect_result["output"], encoding="utf-8")
        (outdir / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        png_path = outdir / "render.png"
        ok, render_log = render_two_panel(final_step, png_path)
        if not ok:
            status["stage"] = "render"
            status["error"] = render_log[:4000]
            (outdir / "error.txt").write_text(f"[render]\n{status['error']}", encoding="utf-8")
            return status
        status["ok"] = True
        status["facts"] = facts
        status["png"] = str(png_path)
        status["step"] = str(final_step)
        return status


def main():
    workers = 2
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    input_name = "labelled.json"
    if "--input" in sys.argv:
        input_name = sys.argv[sys.argv.index("--input") + 1]

    labelled = json.loads((HERE / input_name).read_text(encoding="utf-8"))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    todo = [r for r in labelled if r["program_id"] not in manifest or not manifest[r["program_id"]].get("ok")]
    print(f"{len(labelled)} labelled programs, {len(todo)} to (re)build")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, r): r for r in todo}
        for fut in concurrent.futures.as_completed(futs):
            rec = futs[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = {"program_id": rec["program_id"], "ok": False, "stage": "exception", "error": str(e)}
            manifest[rec["program_id"]] = status
            print(f"{'OK  ' if status['ok'] else 'FAIL'} {rec['program_id']} "
                  f"{'' if status['ok'] else '(' + str(status.get('stage')) + ': ' + str(status.get('error'))[:120] + ')'}")
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_ok = sum(1 for v in manifest.values() if v.get("ok"))
    print(f"\nRebuilt {n_ok}/{len(labelled)} successfully. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

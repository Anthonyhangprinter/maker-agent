#!/usr/bin/env python3
"""Official BenchCAD harness (benchcad.com, MIT) against each arm, through the maker server.

  run_benchcad.py --arms all --tasks codeedit,codeqa,vision2code --num 150 --seed 42 --out benchmarks/results/card/phase0

Writes <out>/benchcad.json = {arm: {task: {"score", "n", "exec_rate", "raw"}}}. Vision2Code runs
only for arms with an mmproj (text-only arms get null for it). Progress is not resumed within a
single arm/task pair — rerunning a card re-does every task listed in --tasks for every arm named
in --arms; the `results[name].setdefault` / `todo` skip below only avoids redoing an (arm, task)
pair that is ALREADY in a previously-written benchcad.json (e.g. a --out dir reused after a crash).
The resident is restored in a finally, whatever happens mid-run.

Harness facts (verified against the actual clone 2026-09-15 — see
benchmarks/external/benchcad/README.md for the full note): the top-level `benchcad.py --task`
keys are `codeedit` / `qa` / `vision2code` (NOT `codeqa` — our own --tasks flag keeps the name
`codeqa` for readability/consistency with the plan, and is mapped to the harness's `qa` key
below). Each task run always uses `configs/prod.yaml` (data_dir `data/`, downloaded from HF on
demand; out_dir `results_prod/`) — there's no CLI flag to redirect out_dir, so this script deletes
that file before each (arm, task) run so a fresh, uncontaminated `results.jsonl` is guaranteed
(every arm calls the harness as the same synthetic model id `"local"` — see local_adapter.py — so
without this the next arm's rows would silently overwrite the previous arm's under the harness's
own (model, record_id) dedup key).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HARNESS = HERE / "benchmarks" / "external" / "benchcad" / "BenchCAD-main"
sys.path.insert(0, str(HERE / "scripts"))
import arms as arms_mod  # noqa: E402

# our --tasks name -> (harness `benchcad.py --task` key, harness task subdirectory)
TASKS = {
    "codeedit": ("codeedit", "CodeEdit"),
    "codeqa": ("qa", "QA"),
    "vision2code": ("vision2code", "Vision2Code"),
}
SCORE_FIELD = {"codeedit": "norm_iou", "codeqa": "qa_score", "vision2code": "score"}
# `execute_cq_to_step` succeeded (a STEP was produced) for status "ok" (fully scored) and
# "score_fail" (STEP produced but the IoU scorer itself threw); "api_fail"/"no_code"/"exec_fail"
# mean no STEP. QA has no execution step at all (text-only in, numbers out).
EXEC_OK_STATUSES = {"ok", "score_fail"}


def template_kwargs(arm: dict) -> str:
    m = re.search(r"--chat-template-kwargs\s+(\S+)", arm.get("extra_args", "") or "")
    return m.group(1) if m else "{}"


def results_path(task: str) -> Path:
    _, subdir = TASKS[task]
    return HARNESS / subdir / "results_prod" / "results.jsonl"


def read_results(path: Path, task: str) -> dict:
    if not path.exists():
        return {"score": None, "n": 0, "exec_rate": None,
                "raw": {"results_path": str(path), "error": "no results.jsonl written"}}
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("model") == "local"]
    field = SCORE_FIELD[task]
    scores = [r[field] for r in rows if r.get(field) is not None]
    score = round(sum(scores) / len(scores), 4) if scores else None
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1
    exec_rate = None
    if task != "codeqa":  # QA is text-only: no CadQuery execution to rate
        exec_rate = round(sum(status_counts.get(s, 0) for s in EXEC_OK_STATUSES) / len(rows), 4) if rows else None
    total_tokens = sum(r.get("total_tokens") or 0 for r in rows)
    return {
        "score": score, "n": len(rows), "exec_rate": exec_rate,
        "raw": {"results_path": str(path.relative_to(HERE)) if HERE in path.parents else str(path),
                "status_counts": status_counts, "total_tokens": total_tokens},
    }


def run_task(arm: dict, task: str, num: int, seed: int) -> dict:
    env = {**os.environ, "BENCHCAD_BASE_URL": "http://127.0.0.1:8088/v1",
           "BENCHCAD_MODEL": arm["alias"], "BENCHCAD_TEMPLATE_KWARGS": template_kwargs(arm)}
    harness_task, _ = TASKS[task]
    rp = results_path(task)
    rp.unlink(missing_ok=True)  # avoid the next arm's rows dedup-colliding into the last arm's file
    cmd = ["uv", "run", "python", "benchcad.py", "--task", harness_task,
           "--num", str(num), "--seed", str(seed), "--model", "local"]
    p = subprocess.run(cmd, cwd=HARNESS, env=env, timeout=6 * 3600)
    result = read_results(rp, task)
    if p.returncode != 0:
        result["raw"]["returncode"] = p.returncode
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--tasks", default="codeedit,codeqa,vision2code")
    ap.add_argument("--num", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    all_arms = arms_mod.load_arms()
    names = list(all_arms) if ns.arms == "all" else ns.arms.split(",")
    tasks = ns.tasks.split(",")
    out = Path(ns.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "benchcad.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    try:
        for name in names:
            arm = all_arms[name]
            if arm.get("skip"):
                print(f'== arm {name}: skipped ("skip": true in benchmarks/arms.json)', flush=True)
                continue
            results.setdefault(name, {})
            todo = [t for t in tasks if t not in results[name] and (t != "vision2code" or arm.get("mmproj"))]
            if "vision2code" in tasks and not arm.get("mmproj") and "vision2code" not in results[name]:
                results[name]["vision2code"] = None
            if not todo:
                continue
            arms_mod.cmd_use(arm)
            for t in todo:
                print(f"== {name} / {t}", flush=True)
                results[name][t] = run_task(arm, t, ns.num, ns.seed)
                res_path.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        arms_mod.cmd_restore()
        res_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

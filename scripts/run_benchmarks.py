#!/usr/bin/env python3
"""
run_benchmarks.py — run the text-to-cad benchmark suite against a CAD agent (default: v4).

For each selected benchmark it shells out to `<agent> build`, parses the result JSON,
re-derives geometry (volume / bbox / faces) from the produced STEP, and records everything to
benchmarks/results/. A build is "converged" only if the agent confirmed it (not just uploaded).

Usage:
  python3 scripts/run_benchmarks.py                       # all 10, --coder auto
  python3 scripts/run_benchmarks.py --tiers 1,2           # only tiers 1 and 2
  python3 scripts/run_benchmarks.py --only 01,05,08       # specific ids
  python3 scripts/run_benchmarks.py --coder fast          # force a coder for the whole run
  python3 scripts/run_benchmarks.py --timeout 2400        # per-build wall-clock cap (s)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE        = Path(__file__).resolve().parent.parent          # .../cad-builder
DEFAULT_AGENT = "v5"   # the cad_v5 --json entry; pass --agent <path/to/script.py> for legacy v4
SPECS_FILE  = HERE / "benchmarks" / "text-to-cad" / "specs.json"
ACCEPT_FILE = HERE / "benchmarks" / "text-to-cad" / "acceptance.json"
RESULTS_DIR = HERE / "benchmarks" / "results"


def derive_geometry(step_path: str) -> dict:
    """Re-derive volume / bbox / solid + face counts from the STEP the agent just produced."""
    try:
        from build123d import import_step, GeomType
        shape = import_step(step_path)
        bb = shape.bounding_box()
        faces = shape.faces()
        try:
            cyl = len(faces.filter_by(GeomType.CYLINDER))
        except Exception:
            cyl = -1
        geom = {
            "volume_mm3": round(shape.volume, 1),
            "solids": len(shape.solids()),
            "faces": len(faces),
            "cyl_faces": cyl,
            "bbox_mm": [round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)],
        }
        # Through/blind hole counts come from the canonical scripts/inspect detector — a blind
        # bore has a cyl face like a through one, so cyl_faces alone scores it a false positive.
        try:
            insp = subprocess.run([sys.executable, str(HERE / "scripts" / "inspect"), step_path],
                                  capture_output=True, text=True, timeout=120).stdout
            import re as _re
            for key, pat in (("through_holes", r"^Through-holes:\s*(-?\d+)"),
                             ("blind_holes",   r"^Blind-holes:\s*(-?\d+)")):
                m = _re.search(pat, insp, _re.M)
                if m:
                    geom[key] = int(m.group(1))
        except Exception:
            pass
        return geom
    except Exception as e:
        return {"geom_error": str(e)[:200]}


def score_acceptance(geom: dict, criteria: dict | None) -> dict:
    """Score derived geometry against a benchmark's auto-checkable acceptance criteria.
    Only non-null criteria count; returns per-check pass/fail plus a fraction.

    A build that produced NO geometry fails every applicable criterion (0/N, not 0/0) —
    otherwise total failures vanish from the denominator and the suite-wide acceptance
    score overstates capability. A criterion is skipped (not failed) only when geometry
    exists but its detector couldn't measure it (-1 sentinel)."""
    if not criteria:
        return {"score": None, "checks": {}, "passed": 0, "total": 0}
    built = bool(geom) and "geom_error" not in geom
    checks: dict[str, bool] = {}
    exp_solids = criteria.get("solids")
    if isinstance(exp_solids, int):
        checks["solids"] = built and (geom.get("solids") == exp_solids)
    exp_bbox = criteria.get("bbox_mm")
    if isinstance(exp_bbox, list) and len(exp_bbox) == 3:
        if built and geom.get("bbox_mm"):
            # Axis-invariant, matching the runtime gate: a correct part re-oriented on
            # another axis is still correct, so compare sorted extent triples.
            ok = True
            for e, g in zip(sorted(exp_bbox), sorted(geom["bbox_mm"])):
                if isinstance(e, (int, float)) and e > 0 and abs(g - e) > max(2.0, 0.05 * e):
                    ok = False
            checks["bbox"] = ok
        else:
            checks["bbox"] = False
    exp_holes = criteria.get("min_holes")
    if isinstance(exp_holes, int):
        cyl = geom.get("cyl_faces", -1)
        if not built:
            checks["holes_cut"] = False
        elif cyl >= 0:
            checks["holes_cut"] = (cyl >= exp_holes)
    exp_through = criteria.get("min_through_holes")
    if isinstance(exp_through, int) and exp_through > 0:
        thru = geom.get("through_holes", -1)
        if not built:
            checks["through"] = False
        elif thru >= 0:
            checks["through"] = (thru >= exp_through)
    passed = sum(checks.values())
    total = len(checks)
    return {"score": round(passed / total, 3) if total else None,
            "checks": checks, "passed": passed, "total": total}


def parse_result(stdout: str) -> dict | None:
    """The agent prints the result dict as a JSON block; grab the last valid JSON object."""
    # Try the whole-block first, then line-by-line, then a brace slice.
    for candidate in (stdout, *[l for l in stdout.splitlines() if l.strip().startswith("{")]):
        try:
            return json.loads(candidate.strip())
        except Exception:
            pass
    start, end = stdout.find("{"), stdout.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stdout[start:end + 1])
        except Exception:
            pass
    return None


def run_one(bm: dict, coder: str, timeout: int, no_fewshots: bool, criteria: dict | None = None,
            agent=DEFAULT_AGENT, run_tag: str = "") -> dict:
    spec = bm["spec"]
    print(f"\n{'='*70}\n[{bm['id']}] {bm['name']}  (tier {bm['tier']}, --coder {coder}"
          f"{', no-fewshots' if no_fewshots else ''})\n{'='*70}", flush=True)
    if str(agent) == "v5":
        # v5 contract: exactly one JSON line on stdout; file target keeps benchmarks headless.
        cmd = [sys.executable, "-m", "cad_v5", spec, "--once", "--json",
               "--coder", coder, "--target", "file"]
    else:
        cmd = [sys.executable, str(agent), "build", spec, "--coder", coder, "--no-upload"]
    if no_fewshots:
        cmd.append("--no-fewshots")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", f"timed out after {timeout}s", 124
    wall = round(time.monotonic() - t0, 1)

    res = parse_result(stdout) or {}
    row = {
        "id": bm["id"], "name": bm["name"], "tier": bm["tier"],
        "ok": bool(res.get("ok")) and rc == 0,
        "converged": bool(res.get("converged")),
        "accepted_via": res.get("accepted_via"),
        "code_model": res.get("code_model"),
        "agent_build_time_s": res.get("build_time_s"),
        "turns": res.get("turns"),
        "n1_autofixes": res.get("n1_autofixes"),
        "wall_time_s": wall,
        "url": res.get("url", ""),
        "warning": res.get("warning"),
        "failures": res.get("failure_categories") or {},
        "rc": rc,
    }
    step_local = res.get("step_local")
    geom = {}
    if step_local and Path(step_local).exists():
        geom = derive_geometry(step_local)
        row.update(geom)
        # Preserve this build's STEP + render so the models can be viewed after the run.
        # Per-run subdir: successive runs used to OVERWRITE each other's artifacts, which is
        # why the gallery lost every older model.
        art = RESULTS_DIR / "artifacts" / (run_tag or "untagged")
        art.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.copy(step_local, art / f"{bm['id']}.step")
            rl = res.get("render_local")
            if rl and Path(rl).exists():
                shutil.copy(rl, art / f"{bm['id']}.png")
                row["render"] = str(art / f"{bm['id']}.png")
            row["step"] = str(art / f"{bm['id']}.step")
        except Exception as e:
            row["artifact_error"] = str(e)[:150]
    if rc != 0 and not res:
        row["error"] = (stderr or stdout or "no output")[-300:]

    acc = score_acceptance(geom, criteria)
    row["acceptance"] = acc
    # Geometry produced = a STEP exists and parsed, regardless of upload (the runner always
    # passes --no-upload, so url is meaningless as a "built" signal).
    row["geometry"] = bool(geom) and "geom_error" not in geom

    via = f" via {row['accepted_via']}" if row.get("accepted_via") else ""
    status = (f"✅ converged{via}" if row["converged"]
              else ("⚠️ geometry, not converged" if row["geometry"] else "❌ failed"))
    acc_str = (f"acc {acc['passed']}/{acc['total']} " + ",".join(f"{k}={'✓' if v else '✗'}"
               for k, v in acc["checks"].items())) if acc["total"] else "acc n/a"
    print(f"--> {status}  | model={row['code_model']} | wall={wall}s | {acc_str} | "
          f"solids={row.get('solids','?')} cyl={row.get('cyl_faces','?')} bbox={row.get('bbox_mm','?')} | "
          f"{row['url'] or row.get('error','')}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", help="comma list of tiers to run, e.g. 1,2")
    ap.add_argument("--only", help="comma list of benchmark ids, e.g. 01,05")
    ap.add_argument("--coder", choices=["auto", "fast", "mid", "strong", "cloud"], default="auto")
    ap.add_argument("--no-fewshots", action="store_true",
                    help="disable retrieval — run the baseline to measure the few-shot lift")
    ap.add_argument("--timeout", type=int, default=2100, help="per-build wall-clock cap (s)")
    ap.add_argument("--agent", default=DEFAULT_AGENT,
                    help='"v5" (default, the cad_v5 --json entry) or a path to a legacy agent '
                         'script accepting: build <spec> --coder X --no-upload')
    a = ap.parse_args()
    if str(a.agent) != "v5" and not Path(a.agent).exists():
        print(f"Agent not found: {a.agent}"); sys.exit(1)

    data = json.loads(SPECS_FILE.read_text())
    benches = data["benchmarks"]
    acceptance = {}
    if ACCEPT_FILE.exists():
        acceptance = {k: v for k, v in json.loads(ACCEPT_FILE.read_text()).items()
                      if not k.startswith("_")}
    if a.tiers:
        keep = {int(t) for t in a.tiers.split(",")}
        benches = [b for b in benches if b["tier"] in keep]
    if a.only:
        keep = {i.strip() for i in a.only.split(",")}
        benches = [b for b in benches if b["id"] in keep]
    if not benches:
        print("No benchmarks match the filter."); sys.exit(1)

    print(f"Running {len(benches)} benchmark(s): {', '.join(b['id'] for b in benches)}  "
          f"(--coder {a.coder}, per-build timeout {a.timeout}s)")
    t0 = time.monotonic()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{a.coder}"
    rows = [run_one(b, a.coder, a.timeout, a.no_fewshots, acceptance.get(b["id"]),
                    agent=a.agent, run_tag=run_tag) for b in benches]
    total = round(time.monotonic() - t0, 1)

    n_conv = sum(r["converged"] for r in rows)
    n_geom = sum(r["geometry"] for r in rows)
    acc_passed = sum(r["acceptance"]["passed"] for r in rows)
    acc_total = sum(r["acceptance"]["total"] for r in rows)
    acc_score = round(acc_passed / acc_total, 3) if acc_total else None
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "agent": str(a.agent), "coder": a.coder, "fewshots": not a.no_fewshots, "count": len(rows),
        "converged": n_conv, "geometry": n_geom, "total_time_s": total,
        "acceptance_passed": acc_passed, "acceptance_total": acc_total, "acceptance_score": acc_score,
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"run_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "latest.json").write_text(json.dumps(summary, indent=2))

    acc_pct = f"{acc_score*100:.0f}%" if acc_score is not None else "n/a"
    print(f"\n{'='*78}\nSUMMARY  —  {n_conv}/{len(rows)} converged, {n_geom}/{len(rows)} produced geometry, "
          f"acceptance {acc_passed}/{acc_total} ({acc_pct}), {total}s total\n{'='*78}")
    print(f"{'id':<3} {'tier':<4} {'status':<14} {'acc':<7} {'model':<26} {'wall_s':<7} {'name'}")
    for r in rows:
        st = "converged" if r["converged"] else ("not-converged" if r["geometry"] else "failed")
        ac = r["acceptance"]
        acs = f"{ac['passed']}/{ac['total']}" if ac["total"] else "-"
        print(f"{r['id']:<3} {r['tier']:<4} {st:<14} {acs:<7} {str(r.get('code_model') or '-'):<26} "
              f"{str(r.get('wall_time_s')):<7} {r['name']}")
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()

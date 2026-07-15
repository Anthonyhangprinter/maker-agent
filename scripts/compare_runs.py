#!/usr/bin/env python3
"""compare_runs.py — side-by-side comparison of benchmark result JSONs.

Usage:
  python3 scripts/compare_runs.py results/run_A.json results/run_B.json ...
  python3 scripts/compare_runs.py --latest 4          # newest N run_*.json

One row per run: coder model(s), converged, acceptance, mean/median wall time, and a
per-benchmark status string (C=converged, a=built-not-converged, F=no geometry) so two
runs can be diffed build-by-build at a glance.
"""
from pathlib import Path
import argparse, json, statistics, sys

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def load(p: Path) -> dict:
    d = json.loads(p.read_text())
    rows = d.get("results") or []
    walls = [r["wall_time_s"] for r in rows if r.get("wall_time_s")]
    models = {str(r.get("code_model")) for r in rows if r.get("code_model")}

    def mark(r):
        if r.get("converged"):
            return "C"
        return "a" if r.get("solids") else "F"

    return {
        "file": p.name,
        "suite": d.get("suite", "?"),
        "models": ",".join(sorted(models)) or "?",
        "n": d.get("count", len(rows)),
        "conv": d.get("converged", sum(1 for r in rows if r.get("converged"))),
        "acc": f"{d.get('acceptance_passed', '?')}/{d.get('acceptance_total', '?')}",
        "wall_mean": round(statistics.mean(walls)) if walls else None,
        "wall_med": round(statistics.median(walls)) if walls else None,
        "per": " ".join(f"{r.get('id','?')}:{mark(r)}" for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--latest", type=int, help="compare the newest N run_*.json instead")
    a = ap.parse_args()

    if a.latest:
        files = sorted(RESULTS.glob("run_*.json"),
                       key=lambda p: p.stat().st_mtime)[-a.latest:]
    else:
        files = [Path(f) for f in a.files]
    if not files:
        sys.exit("no result files given (pass paths or --latest N)")

    rows = [load(Path(f)) for f in files]
    print(f"\n{'file':<30}{'suite':<14}{'model(s)':<26}{'conv':<7}{'acc':<9}"
          f"{'wall x̄/med':<13}per-benchmark")
    for r in rows:
        w = f"{r['wall_mean']}/{r['wall_med']}s" if r["wall_mean"] else "-"
        print(f"{r['file']:<30}{r['suite']:<14}{r['models']:<26}"
              f"{str(r['conv'])+'/'+str(r['n']):<7}{r['acc']:<9}{w:<13}{r['per']}")
    print("\nC=converged  a=built-not-converged  F=no geometry")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""score_heldout.py — reference-STL scoring for the held-out heldout-cqe suite.

Usage:
  python3 scripts/score_heldout.py <run_tag>            # e.g. 20260716_113000_auto
  python3 scripts/score_heldout.py <run_tag> --suite heldout-cqe

For every <id>.step in benchmarks/results/artifacts/<run_tag>/, converts to STL and runs
danwahl/cadqueryeval's registration-based checks (watertight, components, bbox 1mm,
volume 2%, chamfer 1mm, hausdorff95 1mm — RANSAC+ICP aligned, so orientation-free)
against the suite's reference STL. Writes heldout_scores.json next to the artifacts and
prints a table. This AUGMENTS run_benchmarks.py's solids/bbox acceptance; it never
replaces it — the runner stays the convergence verdict, this is the fidelity verdict.
"""
from pathlib import Path
import argparse, json, sys, tempfile

HERE = Path(__file__).resolve().parent.parent
_GEOM = Path.home() / "repos" / "cadqueryeval" / "src" / "cadqueryeval" / "geometry.py"

# Load geometry.py directly by path — the cadqueryeval package __init__ imports
# inspect_ai (its eval harness), which we neither have nor need.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("cqe_geometry", _GEOM)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
perform_geometry_checks = _mod.perform_geometry_checks


def step_to_stl(step: Path, stl: Path) -> None:
    from build123d import import_step, export_stl
    export_stl(import_step(str(step)), str(stl))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--suite", default="heldout-cqe")
    a = ap.parse_args()

    suite_dir = HERE / "benchmarks" / a.suite
    accept = json.loads((suite_dir / "acceptance.json").read_text())
    art = HERE / "benchmarks" / "results" / "artifacts" / a.run_tag
    if not art.is_dir():
        sys.exit(f"no artifacts dir: {art}")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for step in sorted(art.glob("*.step")):
            bid = step.stem
            crit = accept.get(bid)
            if not crit or "reference_stl" not in crit:
                continue
            ref = suite_dir / crit["reference_stl"]
            gen_stl = Path(td) / f"{bid}.stl"
            row = {"id": bid}
            try:
                step_to_stl(step, gen_stl)
                r = perform_geometry_checks(
                    gen_stl, ref, expected_components=int(crit.get("solids", 1)))
                row.update({
                    "watertight": r.is_watertight,
                    "components": r.is_single_component,
                    "bbox": r.bbox_accurate,
                    "volume": r.volume_passed,
                    "chamfer_mm": r.chamfer_distance,
                    "chamfer": r.chamfer_passed,
                    "hausdorff95_mm": r.hausdorff_95p,
                    "hausdorff": r.hausdorff_passed,
                    "passed": bool(r.all_passed),
                    "errors": (r.errors or [])[:3],
                })
            except Exception as e:  # a broken STEP is a scored failure, not a crash
                row.update({"passed": False, "errors": [str(e)[:150]]})
            rows.append(row)

    out = art / "heldout_scores.json"
    out.write_text(json.dumps(
        {"run_tag": a.run_tag, "suite": a.suite, "rows": rows}, indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))  # numpy scalars

    def fmt(v, mm=False):
        if v is None:
            return "-"
        if mm:
            return f"{v:.2f}"
        return "PASS" if v else "fail"

    print(f"\n{'id':<5}{'wtight':<8}{'comps':<8}{'bbox':<8}{'vol':<8}"
          f"{'chamf(mm)':<11}{'haus95(mm)':<12}{'PASSED'}")
    for r in rows:
        print(f"{r['id']:<5}{fmt(r.get('watertight')):<8}{fmt(r.get('components')):<8}"
              f"{fmt(r.get('bbox')):<8}{fmt(r.get('volume')):<8}"
              f"{fmt(r.get('chamfer_mm'), mm=True):<11}{fmt(r.get('hausdorff95_mm'), mm=True):<12}"
              f"{'PASS' if r.get('passed') else 'FAIL'}")
    n = sum(1 for r in rows if r.get("passed"))
    print(f"\n{n}/{len(rows)} passed all reference-STL checks -> {out}")


if __name__ == "__main__":
    main()

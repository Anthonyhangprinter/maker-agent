#!/usr/bin/env python3
"""port_cqe_heldout.py — regenerate the held-out benchmark suite from danwahl/cadqueryeval.

Converts the 25 MIT-licensed CadQueryEval tasks (~/repos/cadqueryeval) into
benchmarks/heldout-cqe/{specs.json,acceptance.json} + reference/ STLs, in the exact
format run_benchmarks.py --suite heldout-cqe consumes.

HELD-OUT CONTRACT: these specs must NEVER be rated into ~/.openclaw/cad-examples.jsonl
or distilled into cad-lessons.jsonl — the whole point is that retrieval cannot have
seen them, so few-shot lift stays honest. Deterministic acceptance (solids + bbox)
comes from the task YAML; richer reference-STL metrics (volume/chamfer/hausdorff via
registration) are computed post-hoc by scripts/score_heldout.py.

Tier mapping from cadqueryeval's manual_operations complexity proxy:
  <=3 ops -> tier 1, 4-6 -> tier 2, >=7 -> tier 3.
"""
from pathlib import Path
import json, shutil, sys

import yaml

CQE = Path.home() / "repos" / "cadqueryeval" / "src" / "cadqueryeval" / "data"
OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "heldout-cqe"


def tier_for(ops: int) -> int:
    return 1 if ops <= 3 else (2 if ops <= 6 else 3)


def main() -> None:
    tasks = sorted(CQE.glob("tasks/task*.yaml"),
                   key=lambda p: int(p.stem.replace("task", "")))
    if len(tasks) != 25:
        sys.exit(f"expected 25 task YAMLs, found {len(tasks)} — check {CQE}")

    (OUT / "reference").mkdir(parents=True, exist_ok=True)
    specs, accept = [], {
        "_meta": {
            "source": "danwahl/cadqueryeval (MIT) — https://github.com/danwahl/cadqueryeval",
            "checks": "solids (expected_component_count) + bbox_mm (sorted-axis, runner tol "
                      "max(2mm,5%)). Reference-STL metrics (volume 2%, chamfer 1mm, "
                      "hausdorff95 1mm, after RANSAC+ICP registration) via score_heldout.py.",
            "heldout": "NEVER promote these specs into cad-examples.jsonl / cad-lessons.jsonl.",
        }
    }
    for p in tasks:
        t = yaml.safe_load(p.read_text())
        n = int(p.stem.replace("task", ""))
        bid = f"H{n:02d}"
        specs.append({
            "id": bid,
            "name": f"cqe {t['task_id']}",
            "tier": tier_for(int(t.get("manual_operations", 3))),
            "spec": t["description"].strip(),
        })
        req = t.get("requirements", {})
        entry = {}
        if req.get("bounding_box"):
            entry["bbox_mm"] = [float(x) for x in req["bounding_box"]]
        comp = (req.get("topology_requirements") or {}).get("expected_component_count")
        if comp:
            entry["solids"] = int(comp)
        entry["reference_stl"] = f"reference/{p.stem}.stl"
        accept[bid] = entry
        shutil.copy2(CQE / "reference" / f"{p.stem}.stl", OUT / "reference" / f"{p.stem}.stl")

    (OUT / "specs.json").write_text(json.dumps({
        "source": "danwahl/cadqueryeval (MIT) — 25 NL->CAD tasks with reference STLs",
        "note": "HELD-OUT suite: excluded from the few-shot corpus by contract (see acceptance _meta).",
        "benchmarks": specs,
    }, indent=1))
    (OUT / "acceptance.json").write_text(json.dumps(accept, indent=1))
    tiers = [s["tier"] for s in specs]
    print(f"wrote {len(specs)} specs -> {OUT}")
    print(f"tiers: 1={tiers.count(1)} 2={tiers.count(2)} 3={tiers.count(3)}")


if __name__ == "__main__":
    main()

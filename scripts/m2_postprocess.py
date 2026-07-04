#!/usr/bin/env python3
"""Archive the no-fewshots A/B run and write the measured few-shot lift summary."""
import glob
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

for f in sorted(glob.glob(str(R / "run_2026*.json"))):
    d = json.load(open(f))
    if d["coder"] == "fast" and not d.get("fewshots", True):
        Path(f).rename(R / "m2_7b_nofewshots.json")

ab, base = R / "m2_7b_nofewshots.json", R / "m1_7b_tiers12.json"
if ab.exists() and base.exists():
    a, b = json.load(open(ab)), json.load(open(base))
    lines = [
        "FEW-SHOT LIFT A/B (7B, tiers 1-2, M1 engine):",
        f"  WITH retrieval:    {b['converged']}/{b['count']} conv, "
        f"{b['acceptance_passed']}/{b['acceptance_total']} acc, {b['total_time_s']:.0f}s",
        f"  WITHOUT retrieval: {a['converged']}/{a['count']} conv, "
        f"{a['acceptance_passed']}/{a['acceptance_total']} acc, {a['total_time_s']:.0f}s",
    ]
    (R / "m2-lift-summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
else:
    print(f"missing inputs: ab={ab.exists()} base={base.exists()}")

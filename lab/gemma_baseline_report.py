#!/usr/bin/env python3
"""lab/gemma_baseline_report.py -- read lab/gemma_baseline.py's own output jsonl and print
the hard-example-mining picture batch 2 needs: where the stock Gemma-4-31B arm already
solves a teacher-verified spec, and where it does not.

CPU-only, no model calls, no GPU window needed -- reads gemma_baseline.jsonl and prints.

Usage:
    python3 lab/gemma_baseline_report.py
    python3 lab/gemma_baseline_report.py path/to/gemma_baseline.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_PATH = (HERE / "benchmarks" / "results" / "card" /
                "codefirst-scale-2026-09-25" / "gemma_baseline.jsonl")

# match/valid = Gemma solved it; near_miss/fail = wrong geometry; crash = never reached a
# geometry comparison at all (codegen raised, the build crashed, or no build.step came
# out). All three of the latter are "not solved" for hard-example mining purposes.
BANDS = ("match", "valid", "near_miss", "fail", "crash")
FAIL_BANDS = ("near_miss", "fail", "crash")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def _tier_sort_key(t: str):
    # numeric tiers sort numerically; "None"/anything non-numeric sorts last.
    try:
        return (0, int(t))
    except (TypeError, ValueError):
        return (1, t)


def print_band_histogram(rows: list[dict]) -> None:
    total = len(rows)
    overall = Counter(r.get("band") or "?" for r in rows)
    print(f"\n== Gemma band histogram, overall (n={total}) ==")
    for b in BANDS:
        n = overall.get(b, 0)
        print(f"  {b:10s} {n:4d}  {_pct(n, total)}")
    other = total - sum(overall.get(b, 0) for b in BANDS)
    if other:
        print(f"  {'(other)':10s} {other:4d}  {_pct(other, total)}")

    by_tier: dict = defaultdict(Counter)
    tier_totals: Counter = Counter()
    for r in rows:
        t = str(r.get("tier"))
        by_tier[t][r.get("band") or "?"] += 1
        tier_totals[t] += 1
    print("\n== Gemma band histogram, by tier ==")
    for t in sorted(by_tier, key=_tier_sort_key):
        tt = tier_totals[t]
        counts = "  ".join(f"{b}={by_tier[t].get(b, 0)}" for b in BANDS)
        ok = by_tier[t].get("match", 0) + by_tier[t].get("valid", 0)
        print(f"  tier {t:>4s} (n={tt:3d}, match+valid={_pct(ok, tt)}): {counts}")


def print_family_match_rate(rows: list[dict]) -> None:
    by_family: dict = defaultdict(Counter)
    for r in rows:
        fam = r.get("family") or "(unknown)"
        by_family[fam]["total"] += 1
        if r.get("band") in ("match", "valid"):
            by_family[fam]["ok"] += 1
    ranked = sorted(by_family.items(),
                    key=lambda kv: (kv[1]["ok"] / kv[1]["total"], -kv[1]["total"], kv[0]))
    print(f"\n== match+valid rate by family, lowest first (n={len(ranked)} families) ==")
    for family, c in ranked:
        rate = c["ok"] / c["total"] if c["total"] else 0.0
        print(f"  {rate * 100:5.1f}%  ({c['ok']:2d}/{c['total']:2d})  {family}")


def print_fail_counts(rows: list[dict]) -> None:
    fails = [r for r in rows if r.get("band") in FAIL_BANDS]
    by_tier = Counter(str(r.get("tier")) for r in fails)
    by_family = Counter(r.get("family") or "(unknown)" for r in fails)
    print(f"\n== Gemma FAILS (band in {FAIL_BANDS}): {len(fails)}/{len(rows)} pairs "
         "-- the batch 2 targeting list ==")
    print("-- by tier --")
    for t, n in sorted(by_tier.items(), key=lambda kv: _tier_sort_key(kv[0])):
        print(f"  tier {t:>4s}: {n}")
    print("-- by family, most fails first --")
    for family, n in by_family.most_common():
        print(f"  {n:3d}  {family}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    a = ap.parse_args()
    path = Path(a.path)
    if not path.exists():
        print(f"gemma_baseline_report: no such file: {path}", file=sys.stderr)
        return 1
    rows = load_rows(path)
    if not rows:
        print(f"gemma_baseline_report: {path} has no rows yet", file=sys.stderr)
        return 0
    print_band_histogram(rows)
    print_family_match_rate(rows)
    print_fail_counts(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

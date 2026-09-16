#!/usr/bin/env python3
"""Lift table: variant deltas vs a baseline arm on the public suites (Task 5, Phase 1 agent lift).

Reads a card run's rows.jsonl (scripts/run_card.py's row schema, see that module's docstring)
and computes, per labelled arm, the handful of numbers that answer "is this variant better than
the baseline arm": invalid ratio, gate-clean rate, pooled acceptance, match rate against a
reference (where one exists), median wall time, and mean output tokens. Everything here is
computed ONLY over the public benchmark suites (cadprompt, text2cadquery, heldout-cqe) so an
internal suite's easier or harder specs never leak into a lift claim, and helper rows (a
bd_warehouse gear/bolt/bearing built correct-by-construction, never touching the model) are
always excluded, same rationale as card_report.summarise's helper column.

  python3 scripts/lift_report.py benchmarks/results/card/phase1          # default baseline
  python3 scripts/lift_report.py benchmarks/results/card/phase1 --baseline gemma-4-31b

Kept dependency-free (stdlib only), same house style as scripts/card_report.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import mean, median

PUBLIC_SUITES = ("cadprompt", "text2cadquery", "heldout-cqe")
DELTA_METRICS = ("invalid_ratio", "gate_clean_rate", "acceptance", "match_rate", "median_wall_s")


def _metrics_for(rows: list[dict]) -> dict:
    """The six numbers for one arm's already-filtered (public suite, non-helper) rows."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "invalid_ratio": None, "gate_clean_rate": None, "acceptance": None,
                "match_rate": None, "median_wall_s": None, "tokens_per_build": None}
    ok_n = sum(1 for r in rows if r["ok"])
    gate_clean_n = sum(1 for r in rows if r["ok"] and r["gate_hard"] == 0 and r["gate_spec"] == 0)
    acc_p = sum(r["acc_passed"] for r in rows)
    acc_t = sum(r["acc_total"] for r in rows)
    banded = [r for r in rows if r.get("band")]
    tokens = [r["tokens_out"] for r in rows if r.get("tokens_out") is not None]
    return {
        "n": n,
        "invalid_ratio": (n - ok_n) / n,
        "gate_clean_rate": gate_clean_n / n,
        "acceptance": (acc_p / acc_t) if acc_t else None,
        "match_rate": (sum(1 for r in banded if r["band"] == "match") / len(banded)) if banded else None,
        "median_wall_s": median(r["wall_s"] for r in rows),
        "tokens_per_build": mean(tokens) if tokens else None,
    }


def lift_table(rows: list[dict], baseline: str, public_suites=PUBLIC_SUITES) -> dict:
    """Per-arm metrics + delta-vs-baseline, over `public_suites` only, helper rows excluded.

    Every arm mentioned anywhere in `rows` (plus `baseline` itself) is present in the result,
    even one with zero qualifying rows (n=0, every other metric None). The baseline's own
    `delta` is all zeros unconditionally; every other arm's delta is `value - baseline_value`
    per metric, None when either side is None."""
    arms = sorted({r["arm"] for r in rows} | {baseline})
    raw = {}
    for arm in arms:
        filtered = [r for r in rows if r["arm"] == arm and r["suite"] in public_suites and not r.get("helper")]
        raw[arm] = _metrics_for(filtered)

    baseline_metrics = raw[baseline]
    table = {}
    for arm, metrics in raw.items():
        if arm == baseline:
            delta = {k: 0 for k in DELTA_METRICS}
        else:
            delta = {}
            for k in DELTA_METRICS:
                v, b = metrics[k], baseline_metrics[k]
                delta[k] = (v - b) if (v is not None and b is not None) else None
        table[arm] = {**metrics, "delta": delta}
    return table


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _num(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def _delta_pts(x) -> str:
    if x is None:
        return "-"
    d = round(100 * x)
    return "0" if d == 0 else f"{d:+d}"


def _delta_secs(x) -> str:
    if x is None:
        return "-"
    d = round(x)
    return "0" if d == 0 else f"{d:+d}"


def render_lift_md(table: dict, baseline: str) -> str:
    """Render the lift table: baseline row first, the rest sorted by arm name."""
    lines = [
        "# Lift vs baseline",
        "",
        f"Baseline arm: `{baseline}`. Computed over the public suites ({', '.join(PUBLIC_SUITES)}) "
        "only, helper rows excluded: invalid = no solid produced, gate clean = solid with zero "
        "hard and zero [spec] findings, acceptance = pooled checks, match = band==match rate "
        "among rows with a band, deltas (delta / Δ) are percentage points vs the baseline "
        "except median s, which is seconds; tokens/build is the mean output tokens.",
        "",
        "| variant | n | invalid | Δ | gate clean | Δ | acceptance | Δ | match | Δ | median s | Δ | tokens/build |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    names = [baseline] + sorted(a for a in table if a != baseline)
    for arm in names:
        r = table[arm]
        d = r["delta"]
        lines.append(
            f"| {arm} | {r['n']} | {_pct(r['invalid_ratio'])} | {_delta_pts(d['invalid_ratio'])} | "
            f"{_pct(r['gate_clean_rate'])} | {_delta_pts(d['gate_clean_rate'])} | "
            f"{_pct(r['acceptance'])} | {_delta_pts(d['acceptance'])} | "
            f"{_pct(r['match_rate'])} | {_delta_pts(d['match_rate'])} | "
            f"{_num(r['median_wall_s'])} | {_delta_secs(d['median_wall_s'])} | "
            f"{_num(r['tokens_per_build'])} |"
        )
    return "\n".join(lines) + "\n"


def _default_baseline(rows: list[dict]) -> str:
    """The unlabelled arm name present in the rows (no "+"), error if not exactly one."""
    unlabelled = sorted({r["arm"] for r in rows if "+" not in r["arm"]})
    if len(unlabelled) != 1:
        raise SystemExit(f"ambiguous baseline: unlabelled arms = {unlabelled}; pass --baseline")
    return unlabelled[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--baseline", default="", help="arm name to diff against (default: the "
                     "one unlabelled arm present in rows.jsonl, error if that's ambiguous)")
    ns = ap.parse_args()

    out = Path(ns.out_dir)
    rows_path = out / "rows.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"{rows_path} not found")
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]

    baseline = ns.baseline or _default_baseline(rows)
    table = lift_table(rows, baseline)
    (out / "LIFT.json").write_text(json.dumps(table, indent=2) + "\n")
    md = render_lift_md(table, baseline)
    (out / "LIFT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

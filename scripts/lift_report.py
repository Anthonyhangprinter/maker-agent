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

Every delta is LIKE FOR LIKE: a variant run on a smaller subset (phase1think is a strict
subset of phase1) is diffed against the baseline's rows for that variant's own specs, not
against the baseline's whole-suite average. `base n` in the table is that restricted count.
--baseline-for lets one variant be diffed against another (agent-self vs agent-gemma4)
instead of against the run's single baseline arm.

  python3 scripts/lift_report.py benchmarks/results/card/phase1          # default baseline
  python3 scripts/lift_report.py benchmarks/results/card/phase1 --baseline gemma-4-31b
  python3 scripts/lift_report.py OUT --baseline gemma-4-31b \
      --baseline-for gemma-4-31b+agent-self=gemma-4-31b+agent-gemma4

Kept dependency-free (stdlib only), same house style as scripts/card_report.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import mean, median

PUBLIC_SUITES = ("cadprompt", "text2cadquery", "heldout-cqe")
DELTA_METRICS = ("invalid_ratio", "gate_clean_rate", "acceptance", "match_rate", "median_wall_s")
# The metrics a paired flip count is reported for: both are per-spec yes/no outcomes, so
# "how many specs did this variant fix, and how many did it break" is meaningful for them
# and meaningless for a pooled ratio or a median.
FLIP_METRICS = ("invalid_ratio", "match_rate")


def _key(r: dict) -> tuple:
    return (r["suite"], r["id"])


def _ref_ids(rows: list[dict]) -> set:
    """The (suite, id) specs that have a band reference, for rows written before run_card
    recorded `has_ref`.

    A row carries a band only when its suite's acceptance entry named a reference_stl AND
    the build produced a STEP, so "carries a band in ANY arm" is the best available proxy
    for "this spec has a reference": a spec no arm ever built still looks reference-less,
    but every spec any arm scored is counted for all of them."""
    return {_key(r) for r in rows if r.get("band")}


def _has_ref(r: dict, ref_ids: set) -> bool:
    return bool(r["has_ref"]) if "has_ref" in r else _key(r) in ref_ids


def _metrics_for(rows: list[dict], ref_ids: set) -> dict:
    """The numbers for one arm's already-filtered (public suite, non-helper) rows.

    match_rate's denominator is every row that HAS a reference, not every row that produced
    a band: a build that failed outright has no band, and counting it out of the denominator
    scored an arm on only the specs it managed to build, so failing more could raise the
    match rate. An invalid build is a non-match."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "ref_n": 0, "invalid_ratio": None, "gate_clean_rate": None,
                "acceptance": None, "match_rate": None, "median_wall_s": None,
                "tokens_per_build": None}
    ok_n = sum(1 for r in rows if r["ok"])
    gate_clean_n = sum(1 for r in rows if r["ok"] and r["gate_hard"] == 0 and r["gate_spec"] == 0)
    acc_p = sum(r["acc_passed"] for r in rows)
    acc_t = sum(r["acc_total"] for r in rows)
    reffed = [r for r in rows if _has_ref(r, ref_ids)]
    tokens = [r["tokens_out"] for r in rows if r.get("tokens_out") is not None]
    return {
        "n": n,
        "ref_n": len(reffed),
        "invalid_ratio": (n - ok_n) / n,
        "gate_clean_rate": gate_clean_n / n,
        "acceptance": (acc_p / acc_t) if acc_t else None,
        "match_rate": (sum(1 for r in reffed if r.get("band") == "match") / len(reffed)) if reffed else None,
        "median_wall_s": median(r["wall_s"] for r in rows),
        "tokens_per_build": mean(tokens) if tokens else None,
    }


def _flips(arm_rows: dict, base_rows: dict, ref_ids: set) -> dict:
    """Paired improved/worsened counts over the specs present in BOTH arms.

    A 3-point delta on n=40 is one build. The delta alone cannot tell "one spec flipped"
    from "eleven improved and ten regressed", and those are different findings, so both
    per-spec outcomes are counted spec by spec against the same spec in the other arm."""
    out = {}
    shared = sorted(set(arm_rows) & set(base_rows))
    for metric in FLIP_METRICS:
        improved = worsened = 0
        for k in shared:
            a, b = arm_rows[k], base_rows[k]
            if metric == "invalid_ratio":
                a_good, b_good = bool(a["ok"]), bool(b["ok"])
            else:
                if not (_has_ref(a, ref_ids) or _has_ref(b, ref_ids)):
                    continue
                a_good = a.get("band") == "match"
                b_good = b.get("band") == "match"
            improved += int(a_good and not b_good)
            worsened += int(b_good and not a_good)
        out[metric] = {"improved": improved, "worsened": worsened}
    return out


def lift_table(rows: list[dict], baseline: str, public_suites=PUBLIC_SUITES,
               baseline_for: dict | None = None) -> dict:
    """Per-arm metrics + like-for-like delta, over `public_suites` only, helpers excluded.

    Every arm mentioned anywhere in `rows` (plus `baseline` itself) is present in the result,
    even one with zero qualifying rows (n=0, every other metric None).

    An arm's delta is against its OWN reference arm's rows for its OWN specs: `baseline_for`
    maps an arm to a reference arm other than `baseline` (so agent-self can be diffed against
    agent-gemma4), and the reference metrics are always recomputed over the reference arm's
    rows restricted to the (suite, id) set the arm itself covers. Without that restriction a
    variant run on a strict subset of the baseline's specs (phase1think inside phase1) would
    be diffed against a different, easier or harder, population. `base` and `base_n` in each
    entry name the reference arm and that restricted count.

    The reference arm's own delta is all zeros unconditionally."""
    baseline_for = dict(baseline_for or {})
    arms = sorted({r["arm"] for r in rows} | {baseline} | set(baseline_for) | set(baseline_for.values()))
    pub = [r for r in rows if r["suite"] in public_suites and not r.get("helper")]
    ref_ids = _ref_ids(rows)
    by_arm: dict[str, dict] = {a: {} for a in arms}
    for r in pub:
        by_arm.setdefault(r["arm"], {})[_key(r)] = r

    table = {}
    for arm in arms:
        own = by_arm.get(arm, {})
        metrics = _metrics_for(list(own.values()), ref_ids)
        base_name = baseline_for.get(arm, baseline)
        base_all = by_arm.get(base_name, {})
        # Like for like: the reference arm's rows for exactly the specs this arm covers.
        base_own = {k: v for k, v in base_all.items() if k in own} if own else {}
        base_metrics = _metrics_for(list(base_own.values()), ref_ids)
        if arm == base_name:
            delta = {k: 0 for k in DELTA_METRICS}
            flips = {k: {"improved": 0, "worsened": 0} for k in FLIP_METRICS}
        else:
            delta = {}
            for k in DELTA_METRICS:
                v, b = metrics[k], base_metrics[k]
                delta[k] = (v - b) if (v is not None and b is not None) else None
            flips = _flips(own, base_all, ref_ids)
        table[arm] = {**metrics, "base": base_name, "base_n": base_metrics["n"],
                      "base_metrics": base_metrics, "delta": delta, "flips": flips}
    return table


def build_report(rows: list[dict], baseline: str, baseline_for: dict | None = None,
                 public_suites=PUBLIC_SUITES) -> dict:
    """The LIFT.json document: which baseline the deltas are against, any per-variant
    overrides, and the table itself. Readers (the Lab view in webui/static/index.html) take
    the baseline name from here instead of re-deriving it and getting a different answer."""
    return {"baseline": baseline, "baseline_for": dict(baseline_for or {}),
            "table": lift_table(rows, baseline, public_suites, baseline_for)}


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


def _flip(f) -> str:
    return "-" if not f else f"+{f['improved']}/-{f['worsened']}"


def render_lift_md(table: dict, baseline: str) -> str:
    """Render the lift table: baseline row first, the rest sorted by arm name."""
    lines = [
        "# Lift vs baseline",
        "",
        f"Baseline arm: `{baseline}`. Computed over the public suites ({', '.join(PUBLIC_SUITES)}) "
        "only, helper rows excluded: invalid = no solid produced, gate clean = solid with zero "
        "hard and zero [spec] findings, acceptance = pooled checks, match = band==match over the "
        "rows that have a reference (ref n; an invalid build is a non-match), deltas (delta / Δ) "
        "are percentage points vs the baseline except median s, which is seconds; tokens/build is "
        "the mean output tokens.",
        "",
        "Each delta is like for like: `base` names the arm it is measured against and `base n` is "
        "that arm's row count restricted to this variant's own specs, so a variant run on a subset "
        "is never compared against the baseline's whole-suite average.",
        "",
        "With n around 40 to 85 a 3-point delta is 1 to 3 builds, which is inside the run-to-run "
        "noise of a single sample: read the flips column (+improved/-worsened, paired spec by spec "
        "against the same spec in the base arm) before calling any of these a result.",
        "",
        "Acceptance is near-redundant with the invalid ratio on cadprompt and text2cadquery, whose "
        "acceptance entries carry the single `solids` criterion: on those suites a build that "
        "produced a solid passes and one that did not fails, so the column mostly restates invalid.",
        "",
        "| variant | base | n | base n | invalid | Δ | flips | gate clean | Δ | acceptance | Δ | "
        "match | ref n | Δ | flips | median s | Δ | tokens/build |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    names = [baseline] + sorted(a for a in table if a != baseline)
    for arm in names:
        r = table[arm]
        d = r["delta"]
        f = r.get("flips") or {}
        lines.append(
            f"| {arm} | {r.get('base', baseline)} | {r['n']} | {r.get('base_n', 0)} | "
            f"{_pct(r['invalid_ratio'])} | {_delta_pts(d['invalid_ratio'])} | "
            f"{_flip(f.get('invalid_ratio'))} | "
            f"{_pct(r['gate_clean_rate'])} | {_delta_pts(d['gate_clean_rate'])} | "
            f"{_pct(r['acceptance'])} | {_delta_pts(d['acceptance'])} | "
            f"{_pct(r['match_rate'])} | {r.get('ref_n', 0)} | {_delta_pts(d['match_rate'])} | "
            f"{_flip(f.get('match_rate'))} | "
            f"{_num(r['median_wall_s'])} | {_delta_secs(d['median_wall_s'])} | "
            f"{_num(r['tokens_per_build'])} |"
        )
    return "\n".join(lines) + "\n"


def _default_baseline(rows: list[dict]) -> str:
    """The unlabelled arm name present in the rows (no "+"), exit 2 if not exactly one.

    Exit 2, not a bare SystemExit (which exits 1): "I could not work out what to compare
    against" is a usage error, and a caller scripting this needs to tell it apart from a
    crash. The message always names the candidates and the flag that settles it."""
    arms = sorted({r["arm"] for r in rows})
    unlabelled = [a for a in arms if "+" not in a]
    if len(unlabelled) > 1:
        print(f"ambiguous baseline: unlabelled arms = {unlabelled}; "
              f"pass --baseline with one of them", file=sys.stderr)
        sys.exit(2)
    if not unlabelled:
        print(f"no unlabelled arm in rows.jsonl to use as a baseline; pass --baseline with "
              f"one of: {arms}", file=sys.stderr)
        sys.exit(2)
    return unlabelled[0]


def _parse_baseline_for(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            print(f"--baseline-for wants VARIANT=ARM, got {p!r}", file=sys.stderr)
            sys.exit(2)
        variant, arm = p.split("=", 1)
        out[variant.strip()] = arm.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--baseline", default="", help="arm name to diff against (default: the "
                     "one unlabelled arm present in rows.jsonl, error if that's ambiguous)")
    ap.add_argument("--baseline-for", action="append", default=[], metavar="VARIANT=ARM",
                     help="diff VARIANT against ARM instead of against --baseline, e.g. "
                          "gemma-4-31b+agent-self=gemma-4-31b+agent-gemma4 (repeatable)")
    ns = ap.parse_args()

    out = Path(ns.out_dir)
    rows_path = out / "rows.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"{rows_path} not found")
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]

    baseline = ns.baseline or _default_baseline(rows)
    baseline_for = _parse_baseline_for(ns.baseline_for)
    report = build_report(rows, baseline, baseline_for)
    (out / "LIFT.json").write_text(json.dumps(report, indent=2) + "\n")
    md = render_lift_md(report["table"], baseline)
    (out / "LIFT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

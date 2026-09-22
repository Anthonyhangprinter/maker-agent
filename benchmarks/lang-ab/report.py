#!/usr/bin/env python3
"""report.py -- turn a lang_ab.py run's rows.jsonl into REPORT.md + report.json.

Usage:
    python3 benchmarks/lang-ab/report.py <run_dir>          # writes <run_dir>/REPORT.md + report.json
    python3 benchmarks/lang-ab/report.py <run_dir> --print   # also prints the markdown

Library:
    from report import summarize, paired_vs_baseline, render_md, write_report
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

BASELINE_ARM = "b123d"


def load_rows(run_dir: Path) -> list[dict]:
    rows_path = run_dir / "rows.jsonl"
    if not rows_path.exists():
        return []
    rows = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _median(values: list) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return round((vals[mid - 1] + vals[mid]) / 2.0, 2)


def _error_signature(row: dict) -> str:
    """A short, dedupe-able label for one failing row's traceback tail: the last
    `SomeError: message` line when one is present (truncated), else a class-derived
    fallback so every failing row still contributes to the top-10 signature count."""
    if row.get("ok"):
        return ""
    tb = row.get("traceback_tail") or ""
    last_exc = None
    for line in tb.splitlines():
        if ":" in line and line[:1].isalpha() and ("Error" in line.split(":")[0] or "Exception" in line.split(":")[0]):
            last_exc = line.strip()
    if last_exc:
        return last_exc[:100]
    ec = row.get("error_class", "other")
    if ec == "no_code":
        return "no_code: empty model reply"
    if ec == "timeout":
        return "timeout"
    stripped = tb.strip()
    if stripped:
        return stripped.splitlines()[-1][:100]
    return ec


def summarize(rows: list[dict]) -> dict:
    """Per-arm: n, built %, error-class %, band match %, match-or-valid %, median
    codegen/build seconds, median tokens out, and the 10 most common error signatures."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    summary: dict[str, dict] = {}
    for arm, arm_rows in by_arm.items():
        n = len(arm_rows)
        built = sum(1 for r in arm_rows if r.get("ok"))
        fail_rows = [r for r in arm_rows if not r.get("ok")]
        err_counts = Counter(r.get("error_class", "other") for r in fail_rows)
        match = sum(1 for r in arm_rows if r.get("band") == "match")
        match_or_valid = sum(1 for r in arm_rows if r.get("band") in ("match", "valid"))
        codegen_s = [r.get("seconds_codegen") for r in arm_rows]
        build_s = [r.get("seconds_build") for r in arm_rows]
        tok_out = [r.get("tokens_out") for r in arm_rows]
        sigs = Counter(_error_signature(r) for r in fail_rows)
        summary[arm] = {
            "n": n,
            "built_pct": round(100.0 * built / n, 1) if n else 0.0,
            "error_class_pct": {k: round(100.0 * v / n, 1) for k, v in sorted(err_counts.items())},
            "match_pct": round(100.0 * match / n, 1) if n else 0.0,
            "match_or_valid_pct": round(100.0 * match_or_valid / n, 1) if n else 0.0,
            "median_codegen_s": _median(codegen_s),
            "median_build_s": _median(build_s),
            "median_tokens_out": _median(tok_out),
            "top_error_signatures": sigs.most_common(10),
        }
    return summary


def paired_vs_baseline(rows: list[dict], baseline_arm: str = BASELINE_ARM) -> dict:
    """Paired comparison of every other arm against `baseline_arm` on the SAME (suite,
    spec_id) pairs only. "Flips" +improved/-worsened on two booleans (built, match=="band
    == match"), overall and broken down by tier. A spec missing from either arm's rows is
    excluded from that arm's pairing (n_pairs records how many pairs were actually compared)."""
    idx: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        idx[(r["suite"], r["spec_id"])][r["arm"]] = r

    other_arms = sorted({r["arm"] for r in rows} - {baseline_arm})
    out: dict[str, dict] = {}
    for arm in other_arms:
        match_flips = {"improved": 0, "worsened": 0}
        built_flips = {"improved": 0, "worsened": 0}
        by_tier: dict = defaultdict(lambda: {"n": 0, "match_improved": 0, "match_worsened": 0})
        n_pairs = 0
        for (suite, spec_id), m in idx.items():
            if baseline_arm not in m or arm not in m:
                continue
            n_pairs += 1
            base, cand = m[baseline_arm], m[arm]
            base_match, cand_match = base.get("band") == "match", cand.get("band") == "match"
            base_built, cand_built = bool(base.get("ok")), bool(cand.get("ok"))
            if cand_match and not base_match:
                match_flips["improved"] += 1
            elif base_match and not cand_match:
                match_flips["worsened"] += 1
            if cand_built and not base_built:
                built_flips["improved"] += 1
            elif base_built and not cand_built:
                built_flips["worsened"] += 1
            tier = base.get("tier", 0)
            t = by_tier[tier]
            t["n"] += 1
            if cand_match and not base_match:
                t["match_improved"] += 1
            elif base_match and not cand_match:
                t["match_worsened"] += 1
        out[arm] = {
            "n_pairs": n_pairs,
            "match_flips": match_flips,
            "built_flips": built_flips,
            "by_tier": {k: v for k, v in sorted(by_tier.items())},
        }
    return out


def render_md(summary: dict, paired: dict, meta: Optional[dict] = None) -> str:
    lines = ["# lang-ab report", ""]
    if meta:
        lines.append(f"Run: {meta.get('run_id', '?')}  ")
        lines.append(f"Specs: {meta.get('n_specs', '?')}  Seed: {meta.get('seed', '?')}  "
                     f"Arms: {', '.join(meta.get('arms', []))}")
        lines.append("")

    lines.append("## Per-arm summary")
    lines.append("")
    lines.append("| arm | n | built % | match % | match-or-valid % | median codegen s | "
                 "median build s | median tokens out |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in sorted(summary):
        s = summary[arm]
        lines.append(
            f"| {arm} | {s['n']} | {s['built_pct']} | {s['match_pct']} | "
            f"{s['match_or_valid_pct']} | {s['median_codegen_s']} | {s['median_build_s']} | "
            f"{s['median_tokens_out']} |"
        )
    lines.append("")

    lines.append("## Error classes (% of that arm's attempts)")
    lines.append("")
    all_classes = sorted({c for s in summary.values() for c in s["error_class_pct"]})
    header = "| arm | " + " | ".join(all_classes) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(all_classes))
    for arm in sorted(summary):
        pct = summary[arm]["error_class_pct"]
        row = " | ".join(str(pct.get(c, 0.0)) for c in all_classes)
        lines.append(f"| {arm} | {row} |")
    lines.append("")

    lines.append(f"## Paired comparison vs {BASELINE_ARM} (same specs only)")
    lines.append("")
    lines.append("| arm | n pairs | match +improved | match -worsened | built +improved | built -worsened |")
    lines.append("|---|---|---|---|---|---|")
    for arm in sorted(paired):
        p = paired[arm]
        lines.append(
            f"| {arm} | {p['n_pairs']} | {p['match_flips']['improved']} | "
            f"{p['match_flips']['worsened']} | {p['built_flips']['improved']} | "
            f"{p['built_flips']['worsened']} |"
        )
    lines.append("")

    lines.append(f"## Paired comparison vs {BASELINE_ARM}, by tier")
    lines.append("")
    for arm in sorted(paired):
        by_tier = paired[arm]["by_tier"]
        if not by_tier:
            continue
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| tier | n | match +improved | match -worsened |")
        lines.append("|---|---|---|---|")
        for tier, t in by_tier.items():
            lines.append(f"| {tier} | {t['n']} | {t['match_improved']} | {t['match_worsened']} |")
        lines.append("")

    lines.append("## Top error signatures per arm")
    lines.append("")
    for arm in sorted(summary):
        sigs = summary[arm]["top_error_signatures"]
        lines.append(f"### {arm}")
        lines.append("")
        if not sigs:
            lines.append("(no failures)")
        else:
            for sig, count in sigs:
                lines.append(f"- ({count}x) {sig}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_report(run_dir: Path) -> str:
    """Read run_dir/rows.jsonl, write run_dir/REPORT.md + run_dir/report.json, return the
    markdown text. Safe to call on a partial (still-running) run: an empty rows.jsonl
    produces an empty-but-valid report rather than raising."""
    rows = load_rows(run_dir)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summary = summarize(rows)
    paired = paired_vs_baseline(rows)
    (run_dir / "report.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "paired_vs_baseline": paired}, indent=2) + "\n",
        encoding="utf-8",
    )
    md = render_md(summary, paired, meta)
    (run_dir / "REPORT.md").write_text(md, encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args()
    md = write_report(Path(a.run_dir))
    if a.do_print:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

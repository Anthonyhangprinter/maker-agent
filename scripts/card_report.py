#!/usr/bin/env python3
"""Pure summarise/render functions for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md).

This module is built incrementally across the phase0-instruments plan:
  - Task 5b: `render_benchcad_md` — the official BenchCAD harness table (benchcad.com, run via
    scripts/run_benchcad.py) plus the published leaderboard reference rows.
  - Task 7: `summarise` / `render_md` — the internal-suite card table, consumed by run_card.py
    (Task 8) and the Lab view (Task 9).

Kept dependency-free (stdlib only) so it can be imported by scripts, tests, and (eventually) the
Lab web view without dragging in the rest of the CAD engine.
"""
from __future__ import annotations
from statistics import median

BANDS = ("match", "valid", "near_miss", "fail")

# Published leaderboard rows (benchcad.com LEADERBOARD.md, read 2026-09-15) — quoted alongside our
# own runs in render_benchcad_md so a card reader can see where a local arm sits against the field.
# Columns: (name, CodeEdit norm_iou, Code-QA accuracy, Vision2Code IoU) — None where not published.
BENCHCAD_REFERENCE = [
    ("Gemma-4-31B-it (published)", None, 0.664, None),
    ("gpt-oss-120b (published)", 0.561, 0.689, None),
    ("GPT-4o (published)", None, 0.726, 0.1823),
    ("Gemini 3.1 Pro (published)", 0.837, 0.838, 0.2890),
]


def _f(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_benchcad_md(results: dict) -> str:
    """Render the official BenchCAD harness results (scripts/run_benchcad.py's benchcad.json).

    `results` is `{arm: {task: {"score": float, "n": int, "exec_rate": float|None, "raw": ...} | None}}`
    (task keys: codeedit / codeqa / vision2code; a task entry is `None` when it didn't run —
    e.g. a text-only arm has no mmproj, so Vision2Code never ran for it).
    """
    lines = ["## Official BenchCAD (benchcad.com harness, local adapter)", "",
             "| arm | CodeEdit | Code-QA | Vision2Code IoU |", "|---|---|---|---|"]
    for arm in sorted(results):
        r = results[arm] or {}

        def g(t):
            v = r.get(t)
            return v.get("score") if isinstance(v, dict) else None

        lines.append(f"| {arm} | {_f(g('codeedit'))} | {_f(g('codeqa'))} | {_f(g('vision2code'))} |")
    for name, ce, qa, v2c in BENCHCAD_REFERENCE:
        lines.append(f"| {name} | {_f(ce)} | {_f(qa)} | {_f(v2c)} |")
    return "\n".join(lines) + "\n"


def summarise(rows: list[dict]) -> dict:
    """Group internal-suite build rows (Task 8's run_card.py schema — see the module docstring
    above) by arm|suite and roll each group up into the card's per-row stats: validity, gate
    cleanliness, pooled acceptance, unit-normalised Chamfer bands, wall time, and token spend."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['arm']}|{r['suite']}", []).append(r)
    out = {}
    for key, rs in groups.items():
        n = len(rs)
        valid = sum(1 for r in rs if r["ok"])
        acc_p = sum(r["acc_passed"] for r in rs)
        acc_t = sum(r["acc_total"] for r in rs)
        out[key] = {
            "arm": rs[0]["arm"], "suite": rs[0]["suite"], "n": n, "valid": valid,
            "invalid_ratio": (n - valid) / n if n else None,
            "gate_clean": sum(1 for r in rs if r["ok"] and r["gate_hard"] == 0 and r["gate_spec"] == 0),
            "acceptance": acc_p / acc_t if acc_t else None,
            "bands": {b: sum(1 for r in rs if r.get("band") == b) for b in BANDS},
            "median_wall_s": median(r["wall_s"] for r in rs) if rs else None,
            "tokens_out": sum(r["tokens_out"] or 0 for r in rs),
        }
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def render_md(summary: dict, meta: dict) -> str:
    """Render the internal-suite card table: one row per arm and suite."""
    lines = [f"# Maker Agent card {meta.get('stamp', '')}", "",
             f"Mode: {meta.get('mode', '')}. One row per arm and suite. invalid = no solid produced, "
             "gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, "
             "bands = unit-normalised Chamfer vs reference where one exists.", "",
             "| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(summary):
        s = summary[key]
        b = s["bands"]
        lines.append(f"| {s['arm']} | {s['suite']} | {s['n']} | {s['valid']} | {_pct(s['invalid_ratio'])} | "
                     f"{s['gate_clean']} | {_pct(s['acceptance'])} | {b['match']} | {b['valid']} | {b['near_miss']} | "
                     f"{b['fail']} | {s['median_wall_s']:.0f} | {s['tokens_out']} |")
    return "\n".join(lines) + "\n"

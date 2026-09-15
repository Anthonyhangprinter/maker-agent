#!/usr/bin/env python3
"""Pure summarise/render functions for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md).

This module is built incrementally across the phase0-instruments plan:
  - Task 5b (this pass): `render_benchcad_md` — the official BenchCAD harness table
    (benchcad.com, run via scripts/run_benchcad.py) plus the published leaderboard reference rows.
  - Task 7 (later): `summarise` / `render_md` — the internal-suite card table, consumed by
    run_card.py (Task 8) and the Lab view (Task 9).

Kept dependency-free (stdlib only) so it can be imported by scripts, tests, and (eventually) the
Lab web view without dragging in the rest of the CAD engine.
"""
from __future__ import annotations

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

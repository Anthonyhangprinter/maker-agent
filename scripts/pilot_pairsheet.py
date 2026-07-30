#!/usr/bin/env python3
"""Stitch two teacher runs into one paired comparison sheet: same part, side by side.

Reads the per-model PNGs that pilot_review.py already regenerated (out/<model-slug>/<id>.png),
so it inherits that script's guarantee that each image came from its own run's code. One row
per spec, left = model A, right = model B, labelled with the spec id and each run's turn count.

    python3 scripts/pilot_pairsheet.py A.json B.json [--out ~/CAD/teacher-pilot]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DEFAULT = Path.home() / "CAD" / "teacher-pilot"
PANEL_W = 560          # width of one model's cropped isometric panel
LABEL_H = 26


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")


def iso_panel(path: Path) -> Image.Image | None:
    """Crop the leftmost (isometric) panel out of a 2- or 3-panel render."""
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    panels = 3 if im.width > im.height * 2.2 else 2
    iso = im.crop((0, 0, im.width // panels, im.height))
    iso.thumbnail((PANEL_W, PANEL_W))
    return iso


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs=2)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    runs = []
    for p in args.summaries:
        d = json.loads(Path(p).read_text())
        runs.append({"model": d.get("model", "?"), "slug": slug(d.get("model", "")),
                     "recs": {r["id"]: r for r in d["per_spec"]}})

    ids = sorted(set(runs[0]["recs"]) | set(runs[1]["recs"]))
    rows = []
    for sid in ids:
        panels = [iso_panel(out / r["slug"] / f"{sid}.png") for r in runs]
        if not any(panels):
            continue                       # neither model produced geometry — nothing to compare
        rows.append((sid, panels))
    if not rows:
        print("no rendered parts found — run pilot_review.py first")
        return 1

    cell_h = max(p.height for _, ps in rows for p in ps if p)
    row_h = cell_h + LABEL_H
    sheet = Image.new("RGB", (PANEL_W * 2, row_h * len(rows) + LABEL_H), "white")
    d = ImageDraw.Draw(sheet)

    for i, r in enumerate(runs):                      # column headers
        d.text((10 + i * PANEL_W, 7), r["model"], fill="black")

    y = LABEL_H
    for sid, panels in rows:
        for i, (r, p) in enumerate(zip(runs, panels)):
            rec = r["recs"].get(sid) or {}
            turns = rec.get("turns", 0)
            oc = rec.get("outcome", "—")
            label = f"{sid}  {oc}  ({turns} turn{'s' if turns != 1 else ''})"
            d.text((10 + i * PANEL_W, y + 5), label, fill="#333")
            if p:
                sheet.paste(p, (i * PANEL_W, y + LABEL_H - 4))
            else:
                d.text((10 + i * PANEL_W, y + LABEL_H + 20), "no accepted geometry",
                       fill="#a02020")
        d.line([(0, y), (PANEL_W * 2, y)], fill="#ddd")
        y += row_h

    dst = out / "comparison-sheet.png"
    sheet.save(dst)
    print(f"paired sheet -> {dst}  ({len(rows)} parts, {sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

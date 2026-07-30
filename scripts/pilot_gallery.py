#!/usr/bin/env python3
"""Collect the accepted parts from a teacher_gen run into a browsable gallery.

Copies each accepted STEP into ~/CAD/teacher-pilot/<model>/<id>.step (double-clickable
into the CAD Viewer from the Files bookmark) and tiles the per-part renders into a single
contact sheet so the whole run can be eyeballed at once.

    python3 scripts/pilot_gallery.py                     # newest run
    python3 scripts/pilot_gallery.py --summary <path>    # a specific run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "benchmarks" / "results"
DEST_ROOT = Path.home() / "CAD" / "teacher-pilot"


def newest_summary() -> Path:
    runs = sorted(RESULTS.glob("teacher_pilot_*.json"))
    if not runs:
        sys.exit("no teacher_pilot_*.json runs found")
    return runs[-1]


def build_step(code: str, work: Path) -> Path | None:
    """Rebuild the STEP from the row's own stored code.

    Do NOT hunt /tmp for build_output.step: every leg globs to the same
    /tmp/teacher_gen_*/<id>/t*/ pattern, so a newest-mtime pick silently returns ANOTHER
    run's geometry. That made a Sonnet-vs-Opus comparison show byte-identical volumes for
    two demonstrably different programs (2026-07-30). The code is the source of truth.
    """
    import subprocess
    work.mkdir(parents=True, exist_ok=True)
    src, out = work / "src.py", work / "out.step"
    src.write_text(code)
    r = subprocess.run([sys.executable, str(HERE / "scripts" / "step"), str(src), str(out)],
                       capture_output=True, text=True, timeout=180)
    return out if out.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    path = Path(args.summary) if args.summary else newest_summary()
    run = json.loads(path.read_text())
    model = run.get("model") or "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    dest = DEST_ROOT / slug
    dest.mkdir(parents=True, exist_ok=True)

    accepted = [r for r in run["per_spec"] if r.get("outcome", "").startswith("accepted")]
    print(f"run     : {path.name}")
    print(f"model   : {model}")
    print(f"accepted: {len(accepted)}/{run['specs']}")

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="pilot_gallery_"))
    copied, pngs = 0, []
    for rec in accepted:
        sid = rec["id"]
        rows = rec.get("rows") or []
        if rows:
            step = build_step(rows[0]["code"], tmp / sid)
            if step:
                shutil.copy(step, dest / f"{sid}.step")
                (dest / f"{sid}.py").write_text(rows[0]["code"])
                copied += 1
            img = rows[0].get("image") or ""
            if img and Path(img).exists() and img not in pngs:
                pngs.append(img)
        (dest / f"{sid}.txt").write_text(f"{sid}\n{rec['spec']}\n\noutcome: {rec['outcome']}\n")

    print(f"STEP files -> {dest}  ({copied} copied)")

    if not pngs:
        print("no renders to tile")
        return 0

    try:
        from PIL import Image
    except ImportError:
        print("pillow not available — skipping contact sheet; open the PNGs individually")
        return 0

    # Tile the isometric panel of each render (leftmost third of the 2- or 3-panel image).
    thumbs = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        panels = 3 if im.width > im.height * 2.2 else 2
        iso = im.crop((0, 0, im.width // panels, im.height))
        iso.thumbnail((420, 420))
        thumbs.append(iso)

    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    tw = max(t.width for t in thumbs)
    th = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * tw, rows_n * th), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    out = dest / "contact-sheet.png"
    sheet.save(out)
    print(f"contact sheet -> {out}  ({len(thumbs)} parts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

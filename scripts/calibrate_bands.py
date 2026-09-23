#!/usr/bin/env python3
"""scripts/calibrate_bands.py — before/after scorer calibration harness (2026-09-24,
branch scorer-calibration-2026-09-24).

Runs BOTH the OLD scorer (scripts/geom_bands.py as committed on `master`, loaded via
`git show master:scripts/geom_bands.py` into an isolated temp module — read-only, never
touches this worktree's checkout) and the NEW scorer (this branch's working tree) against
the real candidate STEP file for each of the 24 owner-labelled builds in
benchmarks/results/card/opus55-pilot-2026-09-23/human_verdicts*.json (18 Opus 5.5 + 6
Gemma harvest-round-1). Everything read is under the MAIN checkout's
benchmarks/results/... and lab/state/{specs.jsonl,ledger.jsonl,refs/} — read-only, never
written to.

Open3D's RANSAC-based registration has run-to-run variance (confirmed during manual
calibration: the same candidate/reference pair can land a few tenths of a mm apart on
chamfer between runs). Each scorer therefore runs --runs times (default 3) per build; the
reported band is the MODE across those runs, with a `flip` flag when the runs didn't all
agree.

Candidate selection:
  Opus set:  the highest-numbered attemptN/ dir that exists for that slug (attempt2 when
             present, else attempt1) — matches every recorded "scorer" value in
             human_verdicts.json exactly (verified by hand before writing this script:
             e.g. cad-exam-3d-part's recorded near_miss matches only attempt2, not
             attempt1's fail).
  Gemma set: candidate "0" of the EARLIEST (round-1) unit for that spec_id in
             lab/state/ledger.jsonl — matches every recorded "scorer" value in
             human_verdicts_gemma_round1.json exactly (candidate "1" differs on 2 of 6
             parts and does not match).

Owner targets:
  Opus set:  human_verdicts.json's "human" field, used as-is.
  Gemma set: human_verdicts_gemma_round1.json's "measured" field is free text ("the
             owner's eye was fooled by the overlay" per the task brief); GEMMA_TARGETS
             below is the adjudicated band each note's own text concludes with, decided by
             hand from that text (quoted in each entry's comment) since there's no
             structured field to parse.

Usage:
  python3 scripts/calibrate_bands.py [--runs 3] [--json out.json]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# Labels, build artifacts and the live harvest's ledger/spec bank are untracked state that
# only exists in the MAIN checkout (this worktree's own lab/state/ is a separate, static
# snapshot with no benchmarks/results/ at all) -- read-only per this branch's rules, never
# written to below. `git show master:...` for the OLD scorer, further down, is a git
# object-database read and does not need this: it works from any worktree unchanged.
MAIN_CHECKOUT = Path("/home/theultimatecunt/.openclaw/skills/cad-builder")
PILOT_DIR = MAIN_CHECKOUT / "benchmarks" / "results" / "card" / "opus55-pilot-2026-09-23"
LEDGER_FILE = MAIN_CHECKOUT / "lab" / "state" / "ledger.jsonl"
SPECS_FILE = MAIN_CHECKOUT / "lab" / "state" / "specs.jsonl"

# Gemma round-1 targets: hand-adjudicated from human_verdicts_gemma_round1.json's free-text
# "measured" field (owner_target, quoted source text).
GEMMA_TARGETS = {
    "air-engine-bush": ("near_miss",
        "vol +19%, step at 4mm not 2mm (overlay hid it) -> near_miss is correct"),
    "air-engine-conrod": ("fail", "3 separate solids -> should be fail"),
    "air-engine-crank-pin": ("near_miss", "25mm vs 20mm long, vol +32% -> near_miss"),
    "air-engine-cylinder": ("valid/match",
        "vol within 2mm3, same bbox; ref bore is also blind -> valid/match"),
    "air-engine-flywheel": ("match", "vol within 15mm3; hole angle unspecified -> match"),
    "cad-exam-3d-part": ("fail", "2 solids, bbox off up to 14mm -> fail"),
}


def load_master_geom_bands():
    """The OLD scorer, exactly as committed on master — never the working tree, never a
    checkout: `git show` reads the blob straight out of the object database, so this
    worktree's HEAD/branch/index is untouched."""
    src = subprocess.run(["git", "show", "master:scripts/geom_bands.py"], cwd=HERE,
                         capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
    tf = tempfile.NamedTemporaryFile(suffix="_geom_bands_old.py", delete=False,
                                     mode="w", encoding="utf-8")
    tf.write(src)
    tf.close()
    spec = importlib.util.spec_from_file_location("geom_bands_old", tf.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_new_geom_bands():
    sys.path.insert(0, str(HERE / "scripts"))
    import geom_bands as new_mod
    return new_mod


def _opus_candidate_step(slug: str) -> Path | None:
    d = PILOT_DIR / "builds" / "owner" / f"owner-reference__{slug}"
    if not d.is_dir():
        return None
    attempts = sorted((p for p in d.iterdir() if p.is_dir() and p.name.startswith("attempt")),
                      key=lambda p: int(p.name.replace("attempt", "")))
    for p in reversed(attempts):
        if (p / "build.step").is_file():
            return p / "build.step"
    return None


def _reference_stl_for(spec_id: str) -> Path | None:
    if not SPECS_FILE.is_file():
        return None
    with open(SPECS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("id") == spec_id and row.get("reference_stl"):
                return Path(row["reference_stl"])
    return None


def _gemma_round1_candidate(slug: str) -> tuple[Path | None, Path | None]:
    """(candidate step, reference stl) for candidate "0" of the EARLIEST unit for
    owner-reference:<slug> in the ledger."""
    spec_id = f"owner-reference:{slug}"
    best = None  # (ts, row)
    with open(LEDGER_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("spec_id") != spec_id or row.get("candidate") != "0":
                continue
            if row.get("pass") != "student":
                continue
            ts = row.get("ts") or ""
            if best is None or ts < best[0]:
                best = (ts, row)
    if best is None:
        return None, None
    row = best[1]
    bdir = row.get("build_dir")
    ref = row.get("ref")
    step = Path(bdir) / "build.step" if bdir else None
    if step and not step.is_file():
        step = None
    return step, (Path(ref) if ref else None)


def build_cases() -> list[dict]:
    cases = []
    verdicts = json.loads((PILOT_DIR / "human_verdicts.json").read_text(encoding="utf-8"))["verdicts"]
    for slug, v in verdicts.items():
        step = _opus_candidate_step(slug)
        ref = _reference_stl_for(f"owner-reference:{slug}")
        cases.append({"slug": slug, "model": "opus-5.5", "owner": v["human"],
                     "step": str(step) if step else None,
                     "ref": str(ref) if ref else None})
    gverdicts = json.loads(
        (PILOT_DIR / "human_verdicts_gemma_round1.json").read_text(encoding="utf-8"))["verdicts"]
    for slug in gverdicts:
        target, _src_note = GEMMA_TARGETS[slug]
        step, ref = _gemma_round1_candidate(slug)
        cases.append({"slug": slug, "model": "gemma-4-31b-r1", "owner": target,
                     "step": str(step) if step else None,
                     "ref": str(ref) if ref else None})
    return cases


def score_n(mod, step: str, ref: str, runs: int) -> dict:
    bands, metrics = [], []
    for _ in range(runs):
        r = mod.score_against_reference(Path(step), Path(ref))
        bands.append(r.get("band"))
        metrics.append(r)
    mode_band = Counter(bands).most_common(1)[0][0]
    flipped = len(set(bands)) > 1
    # Representative metrics: the run whose band matches the mode (first such run).
    rep = next((m for m, b in zip(metrics, bands) if b == mode_band), metrics[0])
    return {"bands": bands, "mode": mode_band, "flipped": flipped,
           "chamfer_mm": rep.get("chamfer_mm"), "hausdorff95_mm": rep.get("hausdorff95_mm"),
           "volume_diff_pct": rep.get("volume_diff_pct"),
           "bbox_near_miss_ok": rep.get("bbox_near_miss_ok"),
           "single_component": rep.get("single_component"),
           "watertight": rep.get("watertight")}


def agrees(owner: str, band: str) -> bool:
    """"valid/match" (the one hedged Gemma target) counts as agreement with either band."""
    targets = owner.split("/")
    return band in targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    old_mod = load_master_geom_bands()
    new_mod = load_new_geom_bands()
    cases = build_cases()

    rows = []
    for c in cases:
        if not c["step"] or not c["ref"]:
            rows.append({**c, "error": "missing step or reference file", "old": None, "new": None})
            print(f"SKIP {c['slug']} ({c['model']}): missing step={c['step']} ref={c['ref']}",
                  file=sys.stderr)
            continue
        old = score_n(old_mod, c["step"], c["ref"], a.runs)
        new = score_n(new_mod, c["step"], c["ref"], a.runs)
        rows.append({**c, "old": old, "new": new})

    header = (f"{'slug':26s} {'model':16s} {'owner':12s} {'OLD':10s} {'NEW':10s} "
             f"{'chamf_old':>9s} {'chamf_new':>9s} {'vol%_old':>9s} {'vol%_new':>9s} "
             f"{'bbox_new':>8s} {'solid_new':>9s} {'flip_old':>8s} {'flip_new':>8s}")
    print(header)
    print("-" * len(header))
    n_before = n_after = n_total = 0
    for r in rows:
        if r.get("old") is None:
            continue
        n_total += 1
        old_agree = agrees(r["owner"], r["old"]["mode"])
        new_agree = agrees(r["owner"], r["new"]["mode"])
        n_before += old_agree
        n_after += new_agree
        print(f"{r['slug']:26s} {r['model']:16s} {r['owner']:12s} "
              f"{r['old']['mode']:10s} {r['new']['mode']:10s} "
              f"{str(round(r['old']['chamfer_mm'],3) if r['old']['chamfer_mm'] is not None else None):>9s} "
              f"{str(round(r['new']['chamfer_mm'],3) if r['new']['chamfer_mm'] is not None else None):>9s} "
              f"{str(r['old']['volume_diff_pct']):>9s} {str(r['new']['volume_diff_pct']):>9s} "
              f"{str(r['new']['bbox_near_miss_ok']):>8s} {str(r['new']['single_component']):>9s} "
              f"{str(r['old']['flipped']):>8s} {str(r['new']['flipped']):>8s}"
              f"{'  <-- DISAGREE(old)' if not old_agree else ''}"
              f"{'  <-- DISAGREE(new)' if not new_agree else ''}")
    opus_rows = [r for r in rows if r.get("old") and r["model"] == "opus-5.5"]
    gemma_rows = [r for r in rows if r.get("old") and r["model"] != "opus-5.5"]
    opus_before = sum(agrees(r["owner"], r["old"]["mode"]) for r in opus_rows)
    opus_after = sum(agrees(r["owner"], r["new"]["mode"]) for r in opus_rows)
    gemma_before = sum(agrees(r["owner"], r["old"]["mode"]) for r in gemma_rows)
    gemma_after = sum(agrees(r["owner"], r["new"]["mode"]) for r in gemma_rows)
    print()
    print(f"Opus:  {opus_before}/{len(opus_rows)} before -> {opus_after}/{len(opus_rows)} after")
    print(f"Gemma: {gemma_before}/{len(gemma_rows)} before -> {gemma_after}/{len(gemma_rows)} after")
    print(f"Total: {n_before}/{n_total} before -> {n_after}/{n_total} after")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

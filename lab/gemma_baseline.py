#!/usr/bin/env python3
"""lab/gemma_baseline.py -- hard-example mining: which of the verified teacher pairs does
the STOCK local CAD model (Gemma-4-31B, the maker arm) already solve on its own?

WHY (owner request, 2026-09-25): lab/teacher_codefirst.py's code-first Opus pipeline
produced 443 verified pairs (benchmarks/results/card/codefirst-scale-2026-09-25/
claude-opus-5-5/pairs.jsonl) plus ~31 from the earlier pilot (codefirst-pilot-2026-09-24) --
each one a (spec, build123d code) pair proven correct against a reference geometry the
teacher itself designed and measured (scripts/measure_part.py). Training the maker arm on
every one of those wastes budget re-teaching it specs it can already build; the pairs it
FAILS on are the real hard-example mining target for a training round (batch 2). This
script measures exactly that split, one sample per spec, through the SAME production path
the harvest's student pass uses: engine.retrieval_notes_for + engine.generate_code_raw
(thinking off, temperature 0.2 -- see lab/harvest.py's own _student_generate), built and
gated the harvest way (fluid_gen._materialize + harvest._regate), then scored against the
SAME reference.stl the teacher pipeline already wrote for that pair
(scripts/geom_bands.score_against_reference).

Known-weak input, flagged up front per the "plain plan paragraph" rule: one sample at
temperature 0.2 is a single draw, not best-of-N. Phase 1
(benchmarks/results/card/phase1/DECISION.md) already found best-of-3 does not beat
one-shot on geometry match for this arm, so a single draw mirrors production rather than
under-measuring it -- but it can still misclassify a spec that sits right on the model's
edge (a near_miss that a second sample would have landed as "valid"). Batch 2's targeting
should read a "fail" row here as "very likely hard for this arm", not "provably
unsolvable".

Success criterion: every pair in both input files gets exactly one row in
gemma_baseline.jsonl (id/tier/family/band/chamfer_mm/gate findings/seconds), resumable
(skip ids already written), so lab/gemma_baseline_report.py can rank tiers and families by
Gemma's failure rate.

What could go wrong: a spec whose reference.stl the original teacher run somehow never
wrote (regenerated here from that pair's own design_build/build.step, see
ensure_reference()); a crashed/killed run losing the in-flight pair's row (resumable design
absorbs this -- it is simply re-attempted next run, never double-counted since a row is
only appended after the full build+score completes); the maker arm itself being unreachable
mid-run (propagates as SpecgenAborted/an ordinary exception, caught per-pair as band
"crash" so one bad pair never aborts the whole overnight run -- only a real SIGTERM does
that, and on purpose, via lab._armwindow's signal handling).

GPU: needs the maker arm resident on the GPU for the whole run (~474 specs x ~45s ~= 6
GPU-hours). MUST run inside exactly one lab/gpu_window.sh window, with
lab._armwindow.arm_window() switching the arm ONCE for the whole run, never per spec (see
lab/README.md's "Teacher reference geometry" section and lab/harvest.py's own _run_in_window
for the pattern this mirrors).

Usage:
    lab/gpu_window.sh python3 -X utf8 lab/gemma_baseline.py --limit 3       # smoke
    lab/gpu_window.sh python3 -X utf8 lab/gemma_baseline.py                 # full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("CAD_BENCH", "1")  # never let a good build self-promote into
                                          # cad-examples.jsonl / cad-sftpairs.jsonl

# ── OCCT/OCP locale trap workaround (see ~/CLAUDE.md's "process trap" note, and
# lab/teacher_codefirst.py's own copy of this fix): force utf-8 as the DEFAULT for every
# Path.write_text/read_text call in this process, before any build123d import happens
# anywhere below. ───────────────────────────────────────────────────────────────────────
import pathlib as _pathlib  # noqa: E402
_orig_write_text = _pathlib.Path.write_text
_orig_read_text = _pathlib.Path.read_text


def _write_text_utf8(self, data, encoding=None, errors=None, newline=None):
    return _orig_write_text(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)


def _read_text_utf8(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors or "replace")


_pathlib.Path.write_text = _write_text_utf8
_pathlib.Path.read_text = _read_text_utf8

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "lab"))

import cad_engine as engine              # noqa: E402
import fluid_gen                         # noqa: E402
from lab import harvest as lab_harvest   # noqa: E402  (_regate, _check_code_model_pin)
from lab import ship                     # noqa: E402
from lab._armwindow import SpecgenAborted, abort_requested, arm_window  # noqa: E402
import geom_bands                        # noqa: E402
import measure_part                      # noqa: E402

CARD_ROOT = HERE / "benchmarks" / "results" / "card"
INPUT_PAIR_FILES = [
    CARD_ROOT / "codefirst-scale-2026-09-25" / "claude-opus-5-5" / "pairs.jsonl",
    CARD_ROOT / "codefirst-pilot-2026-09-24" / "claude-opus-5-5" / "pairs.jsonl",
]
OUT_DIR = CARD_ROOT / "codefirst-scale-2026-09-25"
OUT_JSONL = OUT_DIR / "gemma_baseline.jsonl"
BUILD_ROOT = OUT_DIR / "gemma_baseline_builds"
TEMPERATURE = 0.2

GPU_WINDOW_HINT = (
    "gemma_baseline.py samples the maker arm once per spec across the whole teacher-pair "
    "bank: it must run inside a GPU window: `lab/gpu_window.sh python3 -X utf8 "
    "lab/gemma_baseline.py ...`. Without the lock this run would take the GPU out from "
    "under any CAD build, benchmark card or lab job already using it. Pass "
    "--i-know-the-gpu-is-free only when the GPU is already free by hand.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_pairs() -> list[dict]:
    """Every kept pair row from both input files, each tagged with its own source build
    dir (where reference.stl already lives, written by lab/teacher_codefirst.py itself) so
    later stages never have to guess which run produced it. Refuses on a duplicate id
    across the two files -- the two seed banks (cfNN vs cfsNNNN) are prefixed differently
    on purpose and should never collide; a collision means something upstream changed."""
    rows = []
    for f in INPUT_PAIR_FILES:
        if not f.exists():
            print(f"gemma_baseline: input file missing, skipping: {f}", file=sys.stderr)
            continue
        build_dir_root = f.parent / "builds"
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = str(f)
            row["_ref_dir"] = build_dir_root / row["id"]
            rows.append(row)
    ids = [r["id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RuntimeError(f"gemma_baseline: duplicate pair ids across input files: {dupes}")
    return rows


def family_of(idea: str) -> str:
    """The idea text up to (not including) the first " with " -- e.g. "a rectangular
    mounting plate 60x90x12mm with one central 16mm through hole" -> "a rectangular
    mounting plate 60x90x12mm". Falls back to the whole idea when there is no " with "."""
    return (idea or "").split(" with ", 1)[0].strip()


# ---------------------------------------------------------------------------
# Resumable output
# ---------------------------------------------------------------------------

def done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line).get("id"))
        except Exception:
            continue
    return done


def append_row(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# Reference geometry (reuse what lab/teacher_codefirst.py already wrote)
# ---------------------------------------------------------------------------

def ensure_reference(ref_dir: Path) -> Optional[Path]:
    """The teacher pipeline already wrote reference.stl for every kept pair, in its own
    build dir -- reuse it directly, that geometry IS the ground truth this whole
    measurement scores against. Regenerated (from that same pair's own design_build/
    build.step + measurements.json) only if somehow missing on disk, via the exact sidecar
    writer lab/teacher_codefirst.py itself used
    (measure_part.write_reference_volume_sidecar), so a regenerated reference is the same
    kind of artefact the original run would have produced. Returns None (never raises) when
    neither the reference nor the STEP it would be derived from exists."""
    ref_stl = ref_dir / "reference.stl"
    if ref_stl.exists():
        return ref_stl
    step_path = ref_dir / "design_build" / "build.step"
    if not step_path.exists():
        return None
    measurements = None
    measurements_path = ref_dir / "measurements.json"
    if measurements_path.exists():
        try:
            measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        except Exception:
            measurements = None
    sidecar = ref_dir / "reference_volume.json"
    try:
        measure_part.write_reference_volume_sidecar(step_path, ref_stl, sidecar, measurements)
    except Exception as e:
        print(f"gemma_baseline: could not regenerate reference for {ref_dir}: {e}",
              file=sys.stderr)
        return None
    return ref_stl


# ---------------------------------------------------------------------------
# Build + gate (CPU, the harvest way -- see lab/teacher_codefirst.py's build_and_gate,
# which this mirrors byte-for-byte)
# ---------------------------------------------------------------------------

def build_and_gate(code: str, build_dir: Path, spec: str) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    m = fluid_gen._materialize(code, build_dir, spec)
    return lab_harvest._regate(m, build_dir, spec)


# ---------------------------------------------------------------------------
# Per-pair pipeline
# ---------------------------------------------------------------------------

def run_pair(pair: dict, arm: str) -> dict:
    """band is one of match/valid/near_miss/fail (geom_bands.score_against_reference's own
    bands) or "crash" (codegen raised, the build itself crashed, or no build.step was ever
    produced -- three distinct failure points, folded into one band because none of them
    reached a geometry comparison at all)."""
    sid = pair["id"]
    tier = pair.get("tier")
    idea = pair.get("idea", "")
    spec = pair["spec"]
    family = family_of(idea)
    row: dict = {"id": sid, "tier": tier, "family": family, "arm": arm,
                "source_file": pair["_source_file"], "ts": _now()}
    t0 = time.monotonic()

    ref_stl = ensure_reference(pair["_ref_dir"])
    if ref_stl is None:
        row.update(band="crash", error="no reference geometry available",
                  seconds=round(time.monotonic() - t0, 1))
        return row

    notes = engine.retrieval_notes_for(spec, use_fewshots=True)
    try:
        code = engine.generate_code_raw(spec, notes, temperature=TEMPERATURE)
    except Exception as e:
        row.update(band="crash", error=f"codegen: {str(e)[:300]}",
                  seconds=round(time.monotonic() - t0, 1))
        return row

    build_dir = BUILD_ROOT / sid
    try:
        gate = build_and_gate(code, build_dir, spec)
    except Exception as e:
        row.update(band="crash", error=f"build: {str(e)[:300]}",
                  seconds=round(time.monotonic() - t0, 1))
        return row

    row["gate_hard"] = gate.get("gate_hard")
    row["gate_spec"] = gate.get("gate_spec")
    row["gate_adv"] = gate.get("gate_adv")
    row["unscored_reason"] = gate.get("unscored_reason")

    step_path = build_dir / "build.step"
    if gate.get("error") or not step_path.exists():
        row.update(band="crash", error=gate.get("error") or "no build.step produced",
                  seconds=round(time.monotonic() - t0, 1))
        return row

    score = geom_bands.score_against_reference(step_path, ref_stl)
    row.update(band=score.get("band"), chamfer_mm=score.get("chamfer_mm"),
              volume_diff_pct=score.get("volume_diff_pct"),
              seconds=round(time.monotonic() - t0, 1))
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process this many NOT-YET-DONE pairs (smoke runs)")
    ap.add_argument("--arm", default=None,
                    help="maker arm for this run (default: cad.json's maker block, i.e. "
                        "the stock gemma-4-31b arm)")
    ap.add_argument("--i-know-the-gpu-is-free", action="store_true",
                    help="run outside a GPU window (only when the GPU was freed by hand)")
    a = ap.parse_args()

    # Before ANYTHING that touches a model or a service (same pattern as lab/harvest.py
    # and lab/specgen.py): no arm switch, no signal handlers, no model call.
    ship.require_gpu_window(a, GPU_WINDOW_HINT)

    pin_refusal = lab_harvest._check_code_model_pin()
    if pin_refusal:
        print(f"gemma_baseline: refusing ({pin_refusal})", file=sys.stderr)
        return 1

    pairs = load_pairs()
    print(f"gemma_baseline: loaded {len(pairs)} pairs from "
         f"{sum(1 for f in INPUT_PAIR_FILES if f.exists())} input file(s)", file=sys.stderr)

    rc = 0
    try:
        with arm_window(a.arm) as resolved_arm:
            already = done_ids(OUT_JSONL)
            todo = [p for p in pairs if p["id"] not in already]
            if a.limit is not None:
                todo = todo[:a.limit]
            print(f"gemma_baseline: arm={resolved_arm} already_done={len(already)} "
                 f"todo={len(todo)}", file=sys.stderr)
            for i, pair in enumerate(todo, 1):
                if abort_requested():
                    raise SpecgenAborted("aborted by signal before processing next pair")
                row = run_pair(pair, resolved_arm)
                append_row(OUT_JSONL, row)
                print(f"[{i}/{len(todo)}] {row['id']} tier={row.get('tier')} "
                     f"band={row.get('band')} {row.get('seconds')}s "
                     f"{row.get('error') or ''}", file=sys.stderr)
    except SpecgenAborted as e:
        rc = 1
        print(f"gemma_baseline abort: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())

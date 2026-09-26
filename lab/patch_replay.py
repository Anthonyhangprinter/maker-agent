#!/usr/bin/env python3
"""lab/patch_replay.py -- OFFLINE (CPU, no model) replay of the Gemma baseline's generated
code through the CURRENT cad_engine._patch_code, rebuilt, gated and scored exactly the way
lab/gemma_baseline.py does (fluid_gen._materialize + harvest._regate +
geom_bands.score_against_reference against the same reference.stl).

WHY (2026-09-26): measures what the deterministic spelling/import normalisation
(code_normalise.py) buys on the 112 baseline crashes, and proves it costs nothing on the
561 builds that did not crash. A row whose code the patch leaves byte-identical is not
rebuilt: the build is deterministic, so its band is unchanged by construction (reported as
`unchanged_code`). Every row whose code changed IS rebuilt and re-scored.

Input code is recovered from each baseline build's build_source.py (engine.run_step writes
it as a 3-line sys.path prefix + the already-patched code). Re-applying _patch_code is
idempotent for the old patches, so the only difference is the new normalisation.

Usage:
    python3 -X utf8 lab/patch_replay.py --out ~/lab-scratch/normalise-2026-09-26 [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("CAD_BENCH", "1")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lab"))

from lab import gemma_baseline as gb  # noqa: E402  (also installs the utf-8 Path patch)
import cad_engine as engine            # noqa: E402

BAND_RANK = {"match": 4, "valid": 3, "near_miss": 2, "fail": 1, "crash": 0}


def recover_code(build_dir: Path) -> str:
    src = (build_dir / "build_source.py").read_text(encoding="utf-8")
    prefix_end = src.find("\n\n")
    if src.startswith("import sys as _sys\n_sys.path.insert(") and prefix_end != -1:
        return src[prefix_end + 2:]
    return src


def crash_category(err: str) -> str:
    e = err or ""
    if "'Vector' object has no attribute" in e:
        return "vector_case"
    if "'function' object has no attribute" in e:
        return "method_as_attr"
    if "NameError" in e:
        name = e.split("name '", 1)[-1].split("'", 1)[0] if "name '" in e else ""
        from code_normalise import helper_exports, MATH_NAMES
        if name in helper_exports():
            return "name_helper"
        if name in MATH_NAMES:
            return "name_math"
        return "name_other"
    if "unexpected keyword argument" in e:
        return "helper_signature"
    if "fillet" in e.lower() and "radius" in e.lower():
        return "fillet_too_large"
    if "no build.step" in e or "result =" in e:
        return "no_result"
    return "other_api"


def replay_one(base: dict, pair: dict, out_root: str) -> dict:
    sid = base["id"]
    spec = pair["spec"]
    code = recover_code(gb.BUILD_ROOT / sid)
    patched = engine._patch_code(code, wants=engine._wanted_edge_features(spec))
    row = {"id": sid, "tier": base.get("tier"), "band_before": base["band"],
           "error_before": (base.get("error") or "")[-300:]}
    if patched == code:
        row.update(unchanged_code=True, band_after=base["band"])
        return row
    row["unchanged_code"] = False
    _, fixes = __import__("code_normalise").normalise_api_spelling(code)
    row["fixes"] = sorted(set(fixes))
    t0 = time.monotonic()
    ref_stl = gb.ensure_reference(pair["_ref_dir"])
    bdir = Path(out_root) / "builds" / sid
    try:
        gate = gb.build_and_gate(patched, bdir, spec)
    except Exception as e:
        row.update(band_after="crash", error_after=f"build: {str(e)[:300]}")
        return row
    step = bdir / "build.step"
    if gate.get("error") or not step.exists() or ref_stl is None:
        row.update(band_after="crash",
                   error_after=(gate.get("error") or "no build.step produced")[-300:])
        return row
    score = gb.geom_bands.score_against_reference(step, ref_stl)
    row.update(band_after=score.get("band"), chamfer_mm=score.get("chamfer_mm"),
               volume_diff_pct=score.get("volume_diff_pct"),
               gate_hard=gate.get("gate_hard"), gate_spec=gate.get("gate_spec"),
               seconds=round(time.monotonic() - t0, 1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", choices=["all", "crash", "noncrash"], default="all")
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    pairs = {p["id"]: p for p in gb.load_pairs()}
    base = [json.loads(l) for l in gb.OUT_JSONL.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if a.only == "crash":
        base = [b for b in base if b["band"] == "crash"]
    elif a.only == "noncrash":
        base = [b for b in base if b["band"] != "crash"]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(replay_one, b, pairs[b["id"]], str(out)): b for b in base}
        for i, f in enumerate(as_completed(futs), 1):
            b = futs[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"id": b["id"], "band_before": b["band"], "band_after": "crash",
                     "error_after": f"replay: {e}"[:300], "unchanged_code": False}
            if b["band"] == "crash":
                r["category"] = crash_category(b.get("error") or "")
            rows.append(r)
            if not r.get("unchanged_code"):
                print(f"[{i}/{len(base)}] {r['id']} {r['band_before']} -> {r['band_after']} "
                      f"{r.get('fixes', '')} {(r.get('error_after') or '')[-120:]}",
                      file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["id"])
    (out / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                      encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out / 'replay.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

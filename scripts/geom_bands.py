#!/usr/bin/env python3
"""geom_bands.py — GIFT-style band scoring of a candidate geometry against a reference.

The GIFT paper (arXiv 2603.27448) buckets sampled CAD programs by voxel IoU against ground
truth: exact (>=0.99), "diverse valid" (0.9-0.99, kept as extra training pairs), "near miss"
(0.5-0.9, rendered back as fail->fix pairs), else discarded. We have no voxel-IoU tooling but
already ship danwahl/cadqueryeval's registration-based checker (Chamfer / Hausdorff-95 /
volume / bbox, RANSAC+ICP aligned — orientation-free), so the bands are translated into that
metric space:

  match      all strict checks pass (bbox 1mm, volume 2%, chamfer 1mm, hausdorff95 1mm)
  valid      watertight single solid, chamfer <= VALID_CHAMFER_MM, volume within VALID_VOL_PCT
             -> GIFT-REJECT band: a correct-but-differently-written part, worth keeping as an
                extra (spec, code) SFT pair
  near_miss  chamfer <= NEAR_MISS_DIAG_FRAC of the reference bbox diagonal (scale-aware — a
             2mm miss on a 20mm part is not a 2mm miss on a 500mm beam), volume within
             NEAR_MISS_VOL_PCT when measurable
             -> GIFT-FAIL band: recognisably the intended part built wrong; its render paired
                with the CORRECT code is a geometric-denoising training pair
  fail       everything else (including geometry that does not execute/tessellate)

Usage:
  python3 scripts/geom_bands.py <candidate.step|.stl> <reference.stl> [--components N]
Library:
  from geom_bands import score_against_reference   # returns dict incl. "band"
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import tempfile

_GEOM = Path.home() / "repos" / "cadqueryeval" / "src" / "cadqueryeval" / "geometry.py"

# Load geometry.py directly by path — the cadqueryeval package __init__ imports inspect_ai
# (its eval harness), which we neither have nor need. (Same trick as score_heldout.py.)
_spec = importlib.util.spec_from_file_location("cqe_geometry", _GEOM)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
perform_geometry_checks = _mod.perform_geometry_checks

# Band thresholds (mm / percent). VALID is deliberately just outside the strict gate: the
# strict checks already define "match", so VALID only has to admit parts a human would call
# the same part with cosmetic deviation. NEAR_MISS is scale-aware via the bbox diagonal.
VALID_CHAMFER_MM    = 2.5
VALID_VOL_PCT       = 10.0
NEAR_MISS_DIAG_FRAC = 0.08
NEAR_MISS_VOL_PCT   = 50.0


def _ref_diagonal_mm(ref_stl: Path) -> float:
    import trimesh
    mesh = trimesh.load(str(ref_stl), force="mesh")
    lo, hi = mesh.bounds
    return float(((hi - lo) ** 2).sum() ** 0.5)


def step_to_stl(step: Path, stl: Path) -> None:
    from build123d import import_step, export_stl
    export_stl(import_step(str(step)), str(stl))


def _vol_diff_pct(r) -> float | None:
    if not r.reference_volume or r.generated_volume is None:
        return None
    return abs(r.generated_volume - r.reference_volume) / r.reference_volume * 100.0


def band_of(r, ref_diag_mm: float) -> str:
    """Bucket a GeometryCheckResult into match/valid/near_miss/fail."""
    if r.all_passed:
        return "match"
    vol = _vol_diff_pct(r)
    if (r.is_watertight and r.is_single_component
            and r.chamfer_distance is not None and r.chamfer_distance <= VALID_CHAMFER_MM
            and vol is not None and vol <= VALID_VOL_PCT):
        return "valid"
    if (r.chamfer_distance is not None
            and r.chamfer_distance <= NEAR_MISS_DIAG_FRAC * ref_diag_mm
            and (vol is None or vol <= NEAR_MISS_VOL_PCT)):
        return "near_miss"
    return "fail"


def score_against_reference(candidate: Path, reference_stl: Path,
                            expected_components: int = 1) -> dict:
    """Score a candidate STEP/STL against a reference STL. Never raises: a candidate that
    fails to convert or crashes the checker is a scored 'fail', not an exception — samplers
    call this in bulk and one broken solid must not kill the run."""
    candidate, reference_stl = Path(candidate), Path(reference_stl)
    out: dict = {"candidate": str(candidate), "reference": str(reference_stl), "band": "fail"}
    try:
        ref_diag = _ref_diagonal_mm(reference_stl)
        out["ref_diag_mm"] = round(ref_diag, 2)
        with tempfile.TemporaryDirectory() as td:
            gen_stl = candidate
            if candidate.suffix.lower() in (".step", ".stp"):
                gen_stl = Path(td) / "candidate.stl"
                step_to_stl(candidate, gen_stl)
            r = perform_geometry_checks(gen_stl, reference_stl,
                                        expected_components=expected_components)
        vol = _vol_diff_pct(r)
        out.update({
            "band": band_of(r, ref_diag),
            "all_passed": bool(r.all_passed),
            "watertight": r.is_watertight,
            "single_component": r.is_single_component,
            "bbox": r.bbox_accurate,
            "chamfer_mm": r.chamfer_distance,
            "hausdorff95_mm": r.hausdorff_95p,
            "volume_diff_pct": round(vol, 2) if vol is not None else None,
            "errors": (r.errors or [])[:3],
        })
    except Exception as e:
        out["errors"] = [str(e)[:200]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="generated .step or .stl")
    ap.add_argument("reference", help="reference .stl (ground truth)")
    ap.add_argument("--components", type=int, default=1)
    a = ap.parse_args()
    r = score_against_reference(Path(a.candidate), Path(a.reference),
                                expected_components=a.components)
    print(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    sys.exit(0 if r["band"] in ("match", "valid") else 1)


if __name__ == "__main__":
    main()

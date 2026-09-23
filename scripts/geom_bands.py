#!/usr/bin/env python3
"""geom_bands.py — GIFT-style band scoring of a candidate geometry against a reference.

The GIFT paper (arXiv 2603.27448) buckets sampled CAD programs by voxel IoU against ground
truth: exact (>=0.99), "diverse valid" (0.9-0.99, kept as extra training pairs), "near miss"
(0.5-0.9, rendered back as fail->fix pairs), else discarded. We have no voxel-IoU tooling but
already ship danwahl/cadqueryeval's registration-based checker (Chamfer / Hausdorff-95 /
volume / bbox, RANSAC+ICP aligned — orientation-free), so the bands are translated into that
metric space:

  match      all strict checks pass (bbox 1mm, volume 2%, chamfer 1mm, hausdorff95 1mm) --
             the reference side of the volume check uses an authoritative STEP-solid volume
             when one is cached (see analytic_volume_from_step / write_reference_volume_
             sidecar) instead of the reference MESH's volume, because a mesh re-tessellated
             from STEP can be non-watertight at a fine feature (a drill-point apex, a small
             fillet) even though the solid itself is perfectly valid -- trimesh's mesh-based
             volume check then fails for a reason that has nothing to do with the candidate
             (calibration finding, 2026-09-24: air-engine-piston and air-engine-cylinder)
  valid      watertight single solid, chamfer <= VALID_CHAMFER_MM, volume within VALID_VOL_PCT
             -> GIFT-REJECT band: a correct-but-differently-written part, worth keeping as an
                extra (spec, code) SFT pair
  near_miss  a single solid (2026-09-24: more than one solid is always fail -- a candidate
             that never got fused into one body is not "recognisably the right part built
             wrong", it is unfinished), bbox extents within NEAR_MISS_BBOX_ABS_MM or
             NEAR_MISS_BBOX_REL_FRAC of the reference (whichever is looser) on every axis,
             chamfer <= NEAR_MISS_DIAG_FRAC of the reference bbox diagonal (scale-aware -- a
             2mm miss on a 20mm part is not a 2mm miss on a 500mm beam), volume within
             NEAR_MISS_VOL_PCT when measurable
             -> GIFT-FAIL band: recognisably the intended part built wrong; its render paired
                with the CORRECT code is a geometric-denoising training pair
  fail       everything else (including geometry that does not execute/tessellate)

Calibrated 2026-09-24 against 24 owner FreeCAD-overlay labels (18 Opus 5.5 + 6 Gemma
harvest-round-1 builds, benchmarks/results/card/opus55-pilot-2026-09-23/human_verdicts*.json)
-- see that folder for the label files and docs/FINDINGS-1.0.md's calibration note for the
full before/after table. Two things a single global chamfer/volume threshold could NOT do,
confirmed by measurement, not assumed: (1) separate "identical bbox, wrong hole pattern"
(air-engine-base-plate, still wrongly near_miss after this pass) from "correct shape, one
feature moved" (air-engine-upright/bush-housing, still wrongly valid) -- no metric here is
local enough to see a small missing/misplaced feature on a big part; occasionally left
unresolved on purpose rather than adding a per-part rule. (2) use one relative bbox tolerance
that both excludes air-engine-end-plate/cad-exam-3d-part (grossly wrong dimensions marked
near_miss before this fix) AND keeps air-engine-crank-pin (25mm vs the reference's 20mm, a
25% miss the owner still calls near_miss) -- NEAR_MISS_BBOX_ABS_MM=5.0/REL_FRAC=0.10 is the
widest reasonable per-axis tolerance that still excludes both bad cases; crank-pin clears it
only by a coincidence of exactly matching numbers, not a rule built around it.

Usage:
  python3 scripts/geom_bands.py <candidate.step|.stl> <reference.stl> [--components N]
Library:
  from geom_bands import score_against_reference   # returns dict incl. "band"
"""
from __future__ import annotations
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
MATCH_VOLUME_PCT = _mod.DEFAULT_VOLUME_THRESHOLD_PERCENT  # 2.0 -- the strict "match" cap

# Band thresholds (mm / percent). VALID is deliberately just outside the strict gate: the
# strict checks already define "match", so VALID only has to admit parts a human would call
# the same part with cosmetic deviation. NEAR_MISS is scale-aware via the bbox diagonal.
VALID_CHAMFER_MM    = 2.5
VALID_VOL_PCT       = 10.0
NEAR_MISS_DIAG_FRAC = 0.08
# Raised 50 -> 60 (2026-09-24): air-engine-spring is a thin coiled wire whose total volume is
# small, so a proportionally large percentage swing (57%) comes from a physically small
# absolute feature (the unmodelled variable-pitch closed ends) -- the owner still calls this
# near_miss. A volume-percent cap this loose only matters for parts the bbox+single-solid
# gates below don't already exclude; it is not, by itself, a reliable "is this the right
# part" signal for solid/flat parts (see the base-plate note above), so it stays permissive.
NEAR_MISS_VOL_PCT   = 60.0
# New 2026-09-24 (Required fix #1): per-axis bounding-box tolerance for near_miss, whichever
# of the absolute floor or the relative fraction is looser. 5mm / 10% is the widest tolerance
# that still excludes every known "wrong-size" near_miss false positive (air-engine-end-plate,
# cad-exam-3d-part in both the Opus and Gemma sets) while keeping the one confirmed
# true-near_miss case with a large single-axis bbox miss (air-engine-crank-pin, 25mm built vs
# a 20mm reference).
NEAR_MISS_BBOX_ABS_MM   = 5.0
NEAR_MISS_BBOX_REL_FRAC = 0.10


def _ref_diagonal_mm(ref_stl: Path) -> float:
    import trimesh
    mesh = trimesh.load(str(ref_stl), force="mesh")
    lo, hi = mesh.bounds
    return float(((hi - lo) ** 2).sum() ** 0.5)


def step_to_stl(step: Path, stl: Path) -> None:
    from build123d import import_step, export_stl
    export_stl(import_step(str(step)), str(stl))


def analytic_volume_from_step(step_path: Path) -> float | None:
    """The solid's exact volume straight from OCCT (build123d's Shape.volume), never a
    tessellated mesh. Used to cache an authoritative reference volume that is immune to the
    STL-watertightness defects a mesh reconversion can introduce at a fine feature -- see the
    module docstring's "match" entry. Returns None on any failure (never raises: this is a
    best-effort accuracy improvement, not a required input)."""
    try:
        from build123d import import_step
        shape = import_step(str(step_path))
        vol = float(shape.volume)
        return abs(vol) if vol else None
    except Exception:
        return None


def _reference_volume_sidecar_path(reference_stl: Path) -> Path:
    return Path(reference_stl).with_suffix(".volume.json")


def write_reference_volume_sidecar(step_path: Path, reference_stl: Path) -> bool:
    """Cache the reference's analytic STEP-solid volume next to its cached STL, so a later
    score_against_reference() call (which only ever sees the STL) can pick it up by
    convention. Called once, at reference-materialisation time, by
    lab/specbank.py's _materialize_reference_stl -- never at scoring time, so a broken/slow
    STEP never costs a harvest unit anything. Returns True iff a sidecar was written."""
    vol = analytic_volume_from_step(step_path)
    if vol is None:
        return False
    try:
        _reference_volume_sidecar_path(reference_stl).write_text(
            json.dumps({"volume_mm3": vol, "source": str(step_path)}), encoding="utf-8")
        return True
    except Exception:
        return False


def _reference_analytic_volume(reference_stl: Path) -> float | None:
    sidecar = _reference_volume_sidecar_path(reference_stl)
    if not sidecar.is_file():
        return None
    try:
        return float(json.loads(sidecar.read_text(encoding="utf-8"))["volume_mm3"])
    except Exception:
        return None


def _volume_diff_pct_and_passed(r, ref_vol_override: float | None) -> tuple[float | None, bool | None]:
    """(volume_diff_pct, volume_passed) -- substitutes an authoritative reference volume
    (see analytic_volume_from_step) for the mesh-derived one when a sidecar is available.
    Only ever the REFERENCE side is substituted; the candidate's own generated_volume (from
    a mesh this run just tessellated itself, not a cached asset with a possible defect) is
    always trusted as-is -- this is the "fix the reference conversion, not the candidate
    check" rule."""
    if ref_vol_override and r.generated_volume is not None:
        diff = abs(r.generated_volume - ref_vol_override) / ref_vol_override * 100.0
        return diff, diff <= MATCH_VOLUME_PCT
    if not r.reference_volume or r.generated_volume is None:
        return None, r.volume_passed
    diff = abs(r.generated_volume - r.reference_volume) / r.reference_volume * 100.0
    return diff, r.volume_passed


def _bbox_near_miss_ok(gen_stl: Path, reference_stl: Path) -> bool:
    """Per-axis bounding-box tolerance for the near_miss band (Required fix #1) -- looser
    than the strict 1mm "match" bbox check, computed independently of it (raw sorted AABB
    extents, no ICP registration) since we need the actual per-axis differences, not just a
    pass/fail bit. Always compares real millimetres (pre-normalize, if the caller is scoring
    a normalized public-suite pair) since the tolerance constants are physical mm. Fails
    OPEN (True) on any loading error -- an unrelated bbox-measurement hiccup should not by
    itself veto a near_miss the chamfer/volume/solid-count checks already support."""
    try:
        import trimesh
        gen_dims = sorted(trimesh.load(str(gen_stl), force="mesh")
                          .bounding_box.primitive.extents.tolist())
        ref_dims = sorted(trimesh.load(str(reference_stl), force="mesh")
                          .bounding_box.primitive.extents.tolist())
        return all(abs(g - rr) <= max(NEAR_MISS_BBOX_ABS_MM, NEAR_MISS_BBOX_REL_FRAC * rr)
                   for g, rr in zip(gen_dims, ref_dims))
    except Exception:
        return True


def band_of(r, ref_diag_mm: float, bbox_near_miss_ok: bool = True,
           ref_vol_override: float | None = None) -> str:
    """Bucket a GeometryCheckResult into match/valid/near_miss/fail."""
    vol, volume_passed = _volume_diff_pct_and_passed(r, ref_vol_override)
    all_passed = (bool(r.is_watertight) and bool(r.is_single_component)
                 and bool(r.bbox_accurate) and bool(volume_passed)
                 and bool(r.chamfer_passed) and bool(r.hausdorff_passed))
    if all_passed:
        return "match"
    if (r.is_watertight and r.is_single_component
            and r.chamfer_distance is not None and r.chamfer_distance <= VALID_CHAMFER_MM
            and vol is not None and vol <= VALID_VOL_PCT):
        return "valid"
    if (r.is_single_component and bbox_near_miss_ok
            and r.chamfer_distance is not None
            and r.chamfer_distance <= NEAR_MISS_DIAG_FRAC * ref_diag_mm
            and (vol is None or vol <= NEAR_MISS_VOL_PCT)):
        return "near_miss"
    return "fail"


def normalize_stl(src: Path, dst: Path, target_diag: float = 100.0) -> float:
    """Scale a mesh so its bounding-box diagonal is target_diag (about the origin). Returns the
    scale factor applied. Used to put public suites given in DeepCAD-style normalised units
    (bbox diagonal ~1-2) on the same footing as mm-scale candidates before Chamfer/volume bands."""
    import trimesh
    m = trimesh.load(str(src), force="mesh")
    ext = m.bounding_box.primitive.extents
    diag = float((ext @ ext) ** 0.5)
    f = target_diag / diag if diag > 0 else 1.0
    m.apply_scale(f)
    m.export(str(dst))
    return f


def score_against_reference(candidate: Path, reference_stl: Path, expected_components: int = 1,
                            normalize: bool = False) -> dict:
    """Score a candidate STEP/STL against a reference STL. Never raises: a candidate that
    fails to convert or crashes the checker is a scored 'fail', not an exception — samplers
    call this in bulk and one broken solid must not kill the run.

    normalize=True scales BOTH meshes to a bounding-box diagonal of 100 before the existing
    checks, so absolute-mm thresholds (Chamfer/volume bands) stay meaningful for suites given
    in DeepCAD-style normalised units rather than millimetres. Applied AFTER STEP->STL
    conversion so the candidate is always a concrete mesh when it's rescaled."""
    candidate, reference_stl = Path(candidate), Path(reference_stl)
    out: dict = {"candidate": str(candidate), "reference": str(reference_stl), "band": "fail"}
    extra: dict = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            gen_stl = candidate
            if candidate.suffix.lower() in (".step", ".stp"):
                gen_stl = Path(td) / "candidate.stl"
                step_to_stl(candidate, gen_stl)
            # Bbox tolerance (Required fix #1) always compares real millimetres, so it's
            # measured here against the pre-normalize STL/reference regardless of whether
            # `normalize` later rescales both for the chamfer/volume comparison below.
            bbox_near_miss_ok = _bbox_near_miss_ok(gen_stl, reference_stl)
            # Reference-volume override (Required fix #2) is real-mm too, and normalize's
            # rescale of both meshes for public-suite comparisons is out of scope for the
            # owner-reference sidecar this covers -- only applied on the un-normalized path.
            ref_vol_override = None if normalize else _reference_analytic_volume(reference_stl)
            cmp_reference_stl = reference_stl
            if normalize:
                norm_cand = Path(td) / "norm_cand.stl"
                norm_ref = Path(td) / "norm_ref.stl"
                fc = normalize_stl(gen_stl, norm_cand)
                fr = normalize_stl(reference_stl, norm_ref)
                gen_stl, cmp_reference_stl = norm_cand, norm_ref
                extra = {"normalized": True, "scale_candidate": fc, "scale_reference": fr}
            ref_diag = _ref_diagonal_mm(cmp_reference_stl)
            out["ref_diag_mm"] = round(ref_diag, 2)
            r = perform_geometry_checks(gen_stl, cmp_reference_stl,
                                        expected_components=expected_components)
        vol, _ = _volume_diff_pct_and_passed(r, ref_vol_override)
        out.update({
            "band": band_of(r, ref_diag, bbox_near_miss_ok=bbox_near_miss_ok,
                            ref_vol_override=ref_vol_override),
            "all_passed": bool(r.all_passed),
            "watertight": r.is_watertight,
            "single_component": r.is_single_component,
            "bbox": r.bbox_accurate,
            "bbox_near_miss_ok": bbox_near_miss_ok,
            "chamfer_mm": r.chamfer_distance,
            "hausdorff95_mm": r.hausdorff_95p,
            "volume_diff_pct": round(vol, 2) if vol is not None else None,
            "errors": (r.errors or [])[:3],
        })
    except Exception as e:
        out["errors"] = [str(e)[:200]]
    return {**out, **extra}


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

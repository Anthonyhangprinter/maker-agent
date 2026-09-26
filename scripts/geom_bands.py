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
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_SCORE_TIMEOUT_SEC = 300

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


class ScoreTimeout(Exception):
    """Raised (and always caught, inside score_against_reference itself) when the
    registration checker does not return within its wall-clock budget."""


# Fields of cadqueryeval's GeometryCheckResult that band_of()/_vol_diff_pct() actually
# read, in the shape re-buildable from a subprocess's stdout (see _run_check_bounded).
_CHECK_RESULT_FIELDS = (
    "is_watertight", "is_single_component", "bbox_accurate", "chamfer_distance",
    "hausdorff_95p", "reference_volume", "generated_volume", "errors",
)


def _check_worker_main(argv: list) -> None:
    """Internal subprocess entry point, dispatched from __main__ below before argparse
    ever runs -- never called directly by a human. Computes perform_geometry_checks on
    the two mesh paths given and prints ONE line of JSON with the fields
    _run_check_bounded needs, so the parent process can bound the call with a real
    wall-clock timeout (perform_geometry_checks itself has no timeout knob, and its
    RANSAC+ICP registration can spin forever on a pathological mesh)."""
    gen_stl, ref_stl, components = Path(argv[0]), Path(argv[1]), int(argv[2])
    r = perform_geometry_checks(gen_stl, ref_stl, expected_components=components)
    payload = {"all_passed": bool(r.all_passed)}
    for f in _CHECK_RESULT_FIELDS:
        v = getattr(r, f, None)
        payload[f] = (v[:3] if f == "errors" and v else v)
    print(json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _run_check_bounded(gen_stl: Path, ref_stl: Path, expected_components: int,
                       timeout: float):
    """Runs perform_geometry_checks in its OWN subprocess (this same script, re-invoked
    with a hidden `--check-worker` mode), bounded by `timeout` seconds wall clock.

    Why: measured 2026-09-26, pair cfb20363 hung gate/scoring for 40+ minutes on a 54MB
    candidate STL -- the registration checker has no internal timeout and a
    pathologically large/degenerate mesh can make its ICP alignment spin indefinitely. A
    candidate that large is itself evidence of a bad build, not a correct-but-slow one, so
    past `timeout` this raises ScoreTimeout (caught by score_against_reference's own
    caller below, which records band 'crash' / reason 'score_timeout') rather than
    blocking a whole overnight sampling run on one pair. Runs via subprocess.run(...,
    encoding="utf-8", errors="replace") per the repo's OCP/OCCT locale-trap rule, never
    text=True."""
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--check-worker",
          str(gen_stl), str(ref_stl), str(expected_components)]
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise ScoreTimeout(f"scorer exceeded {timeout}s wall clock") from e
    if proc.returncode != 0:
        raise RuntimeError(f"check-worker failed (exit {proc.returncode}): "
                          f"{proc.stderr[-500:]}")
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.strip()), None)
    if line is None:
        raise RuntimeError(f"check-worker produced no output: stderr={proc.stderr[-500:]!r}")
    return SimpleNamespace(**json.loads(line))


def score_against_reference(candidate: Path, reference_stl: Path, expected_components: int = 1,
                            normalize: bool = False,
                            timeout: float = DEFAULT_SCORE_TIMEOUT_SEC) -> dict:
    """Score a candidate STEP/STL against a reference STL. Never raises: a candidate that
    fails to convert or crashes the checker is a scored 'fail' (or, past `timeout` seconds,
    a scored 'crash' with reason 'score_timeout' -- see _run_check_bounded), never an
    exception -- samplers call this in bulk and one broken or hung solid must not kill the
    run.

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
            try:
                r = _run_check_bounded(gen_stl, cmp_reference_stl, expected_components, timeout)
            except ScoreTimeout as e:
                out["band"] = "crash"
                out["reason"] = "score_timeout"
                out["errors"] = [str(e)]
                return {**out, **extra}
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
    if len(sys.argv) > 1 and sys.argv[1] == "--check-worker":
        _check_worker_main(sys.argv[2:])
        sys.exit(0)
    main()

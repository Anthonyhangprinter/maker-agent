"""Unit tests for the 2026-09-24 scorer calibration in scripts/geom_bands.py (branch
scorer-calibration-2026-09-24). Calibrated against 24 owner FreeCAD-overlay labels --
see the module docstring and benchmarks/results/card/opus55-pilot-2026-09-23/
human_verdicts*.json. Covers the two structural additions to Required fix #1 (a near_miss
candidate must be a single solid; bbox extents must be within an absolute+relative
tolerance of the reference) and Required fix #2 (an authoritative reference volume cached
from the STEP solid, immune to a re-tessellated mesh's watertightness defects).

Run: python3 -m pytest tests/test_geom_bands.py -q
"""
import sys
from pathlib import Path

import pytest
import trimesh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import geom_bands as gb  # noqa: E402


def _box_stl(path: Path, extents) -> Path:
    trimesh.creation.box(extents=extents).export(str(path))
    return path


def _result(**kw):
    """A fabricated GeometryCheckResult -- band_of() is a pure function of this object plus
    a couple of extra scalars, so testing it directly (no RANSAC/ICP registration) is fast
    and deterministic."""
    fields = dict(is_watertight=True, is_single_component=True, bbox_accurate=True,
                  volume_passed=True, chamfer_passed=True, hausdorff_passed=True,
                  chamfer_distance=0.1, hausdorff_95p=0.1, reference_volume=1000.0,
                  generated_volume=1000.0, errors=[])
    fields.update(kw)
    return gb._mod.GeometryCheckResult(**fields)


# --- Required fix #1a: a near_miss candidate must be a single solid -----------------------

def test_band_of_multi_solid_candidate_is_fail_not_near_miss():
    """The exact Gemma conrod regression: 3 separate bodies, but chamfer/volume alone were
    small enough (the pieces sit close to their intended positions) to read as near_miss.
    "Anything with more than one solid is fail" (Required fix #1)."""
    r = _result(is_single_component=False, is_watertight=False, bbox_accurate=False,
               volume_passed=False, chamfer_passed=False, hausdorff_passed=False,
               chamfer_distance=0.2, reference_volume=1000.0, generated_volume=1010.0)
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=True) == "fail"


def test_band_of_single_solid_is_near_miss_all_else_equal():
    """Control for the test above: flip only is_single_component back to True and the same
    candidate DOES qualify as near_miss -- proves the multi-solid check is what's excluding
    it above, not some other field."""
    r = _result(is_single_component=True, is_watertight=False, bbox_accurate=False,
               volume_passed=False, chamfer_passed=False, hausdorff_passed=False,
               chamfer_distance=0.2, reference_volume=1000.0, generated_volume=1010.0)
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=True) == "near_miss"


def test_band_of_multi_solid_also_excludes_valid():
    """"valid" already required is_single_component before this fix -- confirm that's
    still true (not something this pass accidentally loosened)."""
    r = _result(is_single_component=False, chamfer_distance=0.1,
               reference_volume=1000.0, generated_volume=1005.0,
               volume_passed=False, chamfer_passed=False, hausdorff_passed=False,
               bbox_accurate=False)
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=True) in ("fail",)


# --- Required fix #1b: near_miss requires the bbox to be within tolerance -----------------

def test_band_of_near_miss_requires_bbox_near_miss_ok():
    """band_of() itself, given bbox_near_miss_ok=False, must not grant near_miss even when
    every other metric (chamfer, volume, single solid) is fine -- the caller
    (score_against_reference) is responsible for actually measuring the flag; this proves
    band_of() honours it."""
    r = _result(is_watertight=False, bbox_accurate=False, volume_passed=False,
               chamfer_passed=False, hausdorff_passed=False, chamfer_distance=0.5,
               reference_volume=1000.0, generated_volume=1010.0)
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=False) == "fail"
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=True) == "near_miss"


def test_bbox_near_miss_ok_uses_absolute_floor_on_a_small_axis(tmp_path):
    """Base case: NEAR_MISS_BBOX_ABS_MM=5.0 dominates when 10% of the reference's own
    dimension would be tighter than 5mm (the reference's smallest axis here is 12mm, so
    10% would be 1.2mm -- the 5mm absolute floor is what actually applies)."""
    ref = _box_stl(tmp_path / "ref.stl", (170.0, 100.0, 12.0))
    ok = _box_stl(tmp_path / "gen_ok.stl", (170.0, 100.0, 16.5))     # 4.5mm off < 5mm floor
    bad = _box_stl(tmp_path / "gen_bad.stl", (170.0, 100.0, 19.0))   # 7mm off > 5mm floor
    assert gb._bbox_near_miss_ok(ok, ref) is True
    assert gb._bbox_near_miss_ok(bad, ref) is False


def test_bbox_near_miss_ok_uses_relative_fraction_on_a_large_axis(tmp_path):
    """On a large part, NEAR_MISS_BBOX_REL_FRAC=0.10 (10% of that axis) is the looser, and
    therefore controlling, bound -- confirms the "whichever is looser" rule, not just the
    absolute floor, actually widens the tolerance for big parts."""
    ref = _box_stl(tmp_path / "ref.stl", (170.0, 100.0, 12.0))
    ok = _box_stl(tmp_path / "gen_ok.stl", (185.0, 100.0, 12.0))     # 15mm off, 10% = 17mm
    bad = _box_stl(tmp_path / "gen_bad.stl", (195.0, 100.0, 12.0))   # 25mm off > 17mm
    assert gb._bbox_near_miss_ok(ok, ref) is True
    assert gb._bbox_near_miss_ok(bad, ref) is False


def test_bbox_near_miss_ok_fails_open_on_a_missing_file(tmp_path):
    """A bbox-measurement hiccup must not by itself veto a near_miss the other checks
    already support -- fails OPEN (True), per the function's own docstring."""
    ref = _box_stl(tmp_path / "ref.stl", (50.0, 50.0, 50.0))
    assert gb._bbox_near_miss_ok(tmp_path / "does_not_exist.stl", ref) is True


# --- Required fix #2: an authoritative reference volume overrides the mesh volume ---------

def test_volume_diff_pct_and_passed_uses_the_analytic_override():
    """When a sidecar volume is available, it replaces the mesh-derived reference_volume
    for both the reported percentage AND the strict pass/fail bit -- this is what lets a
    candidate reach "match" against a reference whose cached STL is non-watertight at a
    fine feature (air-engine-piston's drill-point apex) even though the solid itself, and
    the candidate, are both fine."""
    r = _result(reference_volume=None, generated_volume=1000.0)  # mesh volume unmeasurable
    diff, passed = gb._volume_diff_pct_and_passed(r, ref_vol_override=1000.0)
    assert diff == pytest.approx(0.0)
    assert passed is True

    diff2, passed2 = gb._volume_diff_pct_and_passed(r, ref_vol_override=2000.0)
    assert diff2 == pytest.approx(50.0)
    assert passed2 is False


def test_volume_diff_pct_and_passed_falls_back_to_mesh_volume_without_an_override():
    """No sidecar (the common case: an old cache, or a reference with no source STEP) ->
    identical behaviour to before this fix, straight off the mesh-derived fields."""
    r = _result(reference_volume=1000.0, generated_volume=1020.0, volume_passed=False)
    diff, passed = gb._volume_diff_pct_and_passed(r, ref_vol_override=None)
    assert diff == pytest.approx(2.0)
    assert passed is False


def test_reference_volume_sidecar_round_trips(tmp_path):
    from build123d import Box, export_step
    box = Box(10, 10, 10)
    step_path = tmp_path / "model.step"
    export_step(box, str(step_path))
    ref_stl = tmp_path / "ref.stl"
    gb.step_to_stl(step_path, ref_stl)

    assert gb._reference_analytic_volume(ref_stl) is None  # no sidecar yet
    assert gb.write_reference_volume_sidecar(step_path, ref_stl) is True
    assert gb._reference_analytic_volume(ref_stl) == pytest.approx(1000.0, rel=1e-6)


def test_reference_volume_sidecar_is_best_effort_on_a_missing_step(tmp_path):
    ref_stl = _box_stl(tmp_path / "ref.stl", (10.0, 10.0, 10.0))
    assert gb.write_reference_volume_sidecar(tmp_path / "no_such.step", ref_stl) is False
    assert gb._reference_analytic_volume(ref_stl) is None


# --- band_of(): raised NEAR_MISS_VOL_PCT still requires the other near_miss checks --------

def test_band_of_still_requires_chamfer_within_diag_frac_even_with_loose_volume():
    """Raising NEAR_MISS_VOL_PCT (to admit the thin-walled air-engine-spring case) must not
    turn it into the ONLY near_miss gate -- a candidate with a huge chamfer relative to the
    reference diagonal still fails, volume tolerance or not."""
    r = _result(is_watertight=False, bbox_accurate=False, volume_passed=False,
               chamfer_passed=False, hausdorff_passed=False,
               chamfer_distance=50.0,  # >> 0.08 * 100mm diag
               reference_volume=1000.0, generated_volume=1050.0)
    assert gb.band_of(r, ref_diag_mm=100.0, bbox_near_miss_ok=True) == "fail"

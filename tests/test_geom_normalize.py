import sys
from pathlib import Path
import trimesh
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import geom_bands as gb


def _box(tmp_path, name, extents):
    p = tmp_path / name
    trimesh.creation.box(extents=extents).export(p)
    return p


def test_normalize_scales_to_target_diagonal(tmp_path):
    src = _box(tmp_path, "small.stl", (0.3, 0.4, 0.5))
    dst = tmp_path / "norm.stl"
    f = gb.normalize_stl(src, dst, target_diag=100.0)
    m = trimesh.load(dst)
    assert abs(float(m.bounding_box.primitive.extents @ m.bounding_box.primitive.extents) ** 0.5 - 100.0) < 1e-3
    # tolerance widened from the brief's 1e-6: trimesh writes binary STL (float32 vertices)
    # by default for a .stl path, so `src` itself already lost precision before normalize_stl
    # ever reads it (measured ~1.7e-6 absolute error on this box, trimesh 4.12.2) -- no
    # implementation of normalize_stl can recover precision the source file never had.
    assert abs(f - 100.0 / (0.3**2 + 0.4**2 + 0.5**2) ** 0.5) < 1e-5


def test_same_shape_different_units_matches_when_normalized(tmp_path):
    ref = _box(tmp_path, "ref.stl", (1.0, 2.0, 3.0))       # DeepCAD-style units
    cand = _box(tmp_path, "cand.stl", (10.0, 20.0, 30.0))  # the agent built it in mm
    raw = gb.score_against_reference(cand, ref)
    norm = gb.score_against_reference(cand, ref, normalize=True)
    assert raw["band"] == "fail"
    assert norm["band"] == "match" and norm["normalized"] is True

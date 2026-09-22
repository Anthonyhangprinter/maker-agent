"""The [spec] hole-overcount check must not read a distributive count as a total."""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
# Scoped to this import only -- a bare os.environ.setdefault here used to leak CAD_BENCH=1
# for the rest of the pytest session (module-level env writes are never undone), which made
# tests/test_design_assistant.py's mode tests see CAD_BENCH set and force "off" when they
# expected "auto"/"always" (caught running the full suite together after the master merge,
# 2026-09-22). verify_expected() itself never reads CAD_BENCH, so nothing in this file
# actually needs it kept set past the import line.
_CAD_BENCH_WAS_SET = "CAD_BENCH" in os.environ
os.environ.setdefault("CAD_BENCH", "1")
import cad_engine as engine  # noqa: E402
if not _CAD_BENCH_WAS_SET:
    os.environ.pop("CAD_BENCH", None)


def _facts(groups):
    return {"solids": 1, "volume": 1000.0, "bbox": [110.0, 35.0, 8.0], "faces": 10,
            "cyl_faces": len(groups), "cone_faces": 0, "hole_groups": groups,
            "bores": [g["d"] for g in groups], "through_holes": sum(g["n"] for g in groups),
            "blind_holes": 0}


def _overcount_notes(spec, groups):
    hard, notes = engine.verify_expected(_facts(groups), {}, spec=spec)
    return [n for n in (notes or []) if "hole(s) of that size" in n]


@pytest.mark.parametrize("spec,groups", [
    ("a 110x35x8mm tie plate with a 14mm hole at each end",
     [{"d": 14.0, "n": 2, "through": 2, "circle_d": 80.0}]),
    ("a 90x60x5mm plate with a 15mm through hole at each corner and a 25mm central through hole",
     [{"d": 15.0, "n": 4, "through": 4, "circle_d": 90.0},
      {"d": 25.0, "n": 1, "through": 1, "circle_d": 0.0}]),
    ("four 8mm tall bosses at the corners, each with a 2mm blind hole",
     [{"d": 2.0, "n": 4, "through": 0, "circle_d": 60.0}]),
    ("an L bracket with two 6mm holes in each leg",
     [{"d": 6.0, "n": 4, "through": 4, "circle_d": 50.0}]),
])
def test_distributive_hole_counts_are_not_overcounts(spec, groups):
    assert _overcount_notes(spec, groups) == []


def test_a_plain_total_is_still_enforced():
    spec = "a 60x60x10mm plate with a 10mm central hole"
    five = [{"d": 10.0, "n": 5, "through": 5, "circle_d": 40.0}]
    assert len(_overcount_notes(spec, five)) == 1
    spec2 = "a flange with two 6mm through holes 34mm apart"
    assert len(_overcount_notes(spec2, [{"d": 6.0, "n": 5, "through": 5, "circle_d": 34.0}])) == 1


def test_a_nearby_each_that_belongs_to_another_feature_does_not_disable_the_check():
    spec = "a barrel with a 66mm through bore, twelve fins each 3mm thick"
    assert len(_overcount_notes(spec, [{"d": 66.0, "n": 3, "through": 3, "circle_d": 0.0}])) == 1

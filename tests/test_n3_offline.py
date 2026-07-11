"""N3 offline unit tests — apply_brief_patch() is pure/deterministic (no LLM, no subprocess),
so it's fully testable without Ollama. Run: python3 -m pytest tests/test_n3_offline.py -q
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cad_engine as v4  # noqa: E402


def _brief():
    return {
        "name": "Test enclosure",
        "description": "a simple hollow enclosure",
        "dimensions": {"length": 100, "width": 60, "wall": 2},
        "features": ["hollow", "open top", "lid"],
        "notes": ["some note"],
        "expected": {"solids": 1, "min_holes": 0, "min_through_holes": 0, "wall_mm": 2},
    }


def test_dimension_change():
    brief = _brief()
    new_brief, delta = v4.apply_brief_patch(
        brief, [{"field": "dimensions.wall", "value": "3"}], [], [])
    assert new_brief["dimensions"]["wall"] == 3
    assert any(d.startswith("wall: 2 → 3") for d in delta), delta


def test_new_expected_key():
    brief = _brief()
    assert "bore_mm" not in brief["expected"]
    new_brief, delta = v4.apply_brief_patch(
        brief, [{"field": "expected.bore_mm", "value": "20"}], [], [])
    assert new_brief["expected"]["bore_mm"] == 20
    assert any("bore_mm" in d for d in delta)


def test_feature_add_and_remove():
    brief = _brief()
    new_brief, delta = v4.apply_brief_patch(
        brief, [], ["4x M3 mounting holes"], ["lid"])
    assert "4x M3 mounting holes" in new_brief["features"]
    assert not any("lid" in f.lower() for f in new_brief["features"])
    assert any(d == "features: +4x M3 mounting holes" for d in delta), delta
    assert any(d.startswith("features: −") and "lid" in d for d in delta), delta


def test_feature_remove_is_case_insensitive_substring():
    brief = _brief()
    new_brief, _ = v4.apply_brief_patch(brief, [], [], ["LID"])
    assert "lid" not in [f.lower() for f in new_brief["features"]]


def test_malformed_field_skipped():
    brief = _brief()
    new_brief, delta = v4.apply_brief_patch(
        brief, [{"field": "bogus_top_level", "value": "x"}], [], [])
    assert new_brief["dimensions"] == brief["dimensions"]
    assert any(d == "(?) bogus_top_level ignored" for d in delta), delta


def test_malformed_dotted_field_with_empty_key_skipped():
    brief = _brief()
    new_brief, delta = v4.apply_brief_patch(
        brief, [{"field": "dimensions.", "value": "x"}], [], [])
    assert new_brief["dimensions"] == brief["dimensions"]
    assert any("ignored" in d for d in delta), delta


def test_numeric_coercion_int_and_float():
    brief = _brief()
    new_brief, _ = v4.apply_brief_patch(
        brief,
        [{"field": "dimensions.wall", "value": "3"},
         {"field": "dimensions.length", "value": "101.5"}],
        [], [])
    assert new_brief["dimensions"]["wall"] == 3 and isinstance(new_brief["dimensions"]["wall"], int)
    assert new_brief["dimensions"]["length"] == 101.5 and isinstance(new_brief["dimensions"]["length"], float)


def test_non_numeric_string_value_kept_as_string():
    brief = _brief()
    new_brief, _ = v4.apply_brief_patch(
        brief, [{"field": "name", "value": "M3 Bracket"}], [], [])
    assert new_brief["name"] == "M3 Bracket"


def test_original_dict_never_mutated():
    brief = _brief()
    original_copy = copy.deepcopy(brief)
    v4.apply_brief_patch(
        brief,
        [{"field": "dimensions.wall", "value": "3"}, {"field": "name", "value": "Changed"}],
        ["new feature"], ["lid"])
    assert brief == original_copy, "the original brief dict must never be mutated"


def test_untouched_fields_identical_in_new_copy():
    brief = _brief()
    new_brief, _ = v4.apply_brief_patch(
        brief, [{"field": "dimensions.wall", "value": "3"}], [], [])
    # Only `wall` changed; everything else in dimensions and the rest of the brief is identical.
    assert new_brief["dimensions"]["length"] == brief["dimensions"]["length"]
    assert new_brief["dimensions"]["width"] == brief["dimensions"]["width"]
    assert new_brief["description"] == brief["description"]
    assert new_brief["notes"] == brief["notes"]
    assert new_brief["name"] == brief["name"]


def test_no_changes_is_a_no_op_copy():
    brief = _brief()
    new_brief, delta = v4.apply_brief_patch(brief, [], [], [])
    assert new_brief == brief
    assert delta == []
    assert new_brief is not brief   # still a deep copy, not the same object

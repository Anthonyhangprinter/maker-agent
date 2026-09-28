"""Negation-aware feature-word gate checks (2026-09-24).

Several of cad_engine.verify_expected's ADVISORY checks fire purely on "does this word
appear in the spec text" (chamfer presence vs measured cone_faces; thread/groove/tooth
presence vs a bare-primitive part). A bare `re.search` cannot tell a request from its
negation, and this bit a real blind-rebuild pair in the code-first teacher pipeline
(lab/teacher_codefirst.py, smoke test 2026-09-24): a spec reading "No chamfers, fillets, or
other features are required" tripped the chamfer check even though the rebuild matched its
reference exactly (band "match", volume_diff_pct 0.0). Fixed via cad_engine._feature_requested,
a simple whole-clause negation check: a negator word ("no", "without", "never", ...) ANYWHERE
in the same clause as the feature word negates every feature mention in that clause.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import cad_engine as eng  # noqa: E402

NO_CONE_FACTS = {"solids": 1, "cone_faces": 0, "cyl_faces": 1, "through_holes": 1,
                 "blind_holes": 0, "bores": [15.0], "volume": 50000, "bbox": [100, 60, 10]}


# ── _feature_requested: pure unit tests ─────────────────────────────────────────────────────
def test_feature_requested_true_for_plain_mention():
    assert eng._feature_requested("a 1mm chamfer on the top edge", r"\bchamfer") is True


def test_feature_requested_false_for_no_x():
    assert eng._feature_requested("No chamfers are required", r"\bchamfer") is False


def test_feature_requested_false_for_no_x_or_y_list():
    """The exact phrasing that tripped this in the wild: a negator governing a list of
    nouns joined by commas/or must negate every noun in that clause, not just the first."""
    spec = "No chamfers, fillets, or other features are required — just the plate with the through-hole."
    assert eng._feature_requested(spec, r"\bchamfer") is False
    assert eng._feature_requested(spec, r"\bfillet") is False


def test_feature_requested_false_for_without_x():
    assert eng._feature_requested("a plain bracket, without any chamfers", r"\bchamfer") is False


def test_feature_requested_false_for_suffix_free():
    assert eng._feature_requested("a chamfer-free block", r"\bchamfer") is False


def test_feature_requested_true_when_negation_is_a_different_clause():
    """A negator in an EARLIER, unrelated clause must not blank out a later affirmative
    request -- clause-scoped, not whole-spec."""
    spec = "A bracket with no mounting holes. Add a 2mm chamfer on the top edge."
    assert eng._feature_requested(spec, r"\bchamfer") is True


def test_feature_requested_alternation_pattern():
    """The thread/groove/... sibling check passes a multi-word alternation, not a single
    word -- the helper must handle that shape too."""
    pattern = r"thread|groove|tooth|teeth|knurl|flute|spline|vane|fin"
    assert eng._feature_requested("a shaft with a helical thread", pattern) is True
    assert eng._feature_requested("a shaft with no threads or grooves", pattern) is False


# ── verify_expected: the two cases the task asked for, plus the thread sibling check ────────
def test_no_chamfers_required_gives_no_finding():
    spec = ("I need a flat plate measuring 100mm x 60mm x 10mm thick. Please machine a "
            "single 15mm diameter hole straight through the plate's thickness, centered at "
            "the middle of the plate (50mm from each side edge, 30mm from each end edge). "
            "No chamfers, fillets, or other features are required — just the plate with the "
            "through-hole as described.")
    _hard, soft = eng.verify_expected(NO_CONE_FACTS, {}, spec=spec)
    assert not any("chamfer" in s.lower() for s in soft), soft


def test_chamfer_with_no_conical_face_still_finds():
    spec = "a 1mm chamfer on the top edge"
    _hard, soft = eng.verify_expected(NO_CONE_FACTS, {}, spec=spec)
    assert any("[spec] the spec asks for a chamfer" in s for s in soft), soft


def test_thread_sibling_check_negation_aware():
    bare_primitive_facts = {"solids": 1, "parts": [{"label": "shaft", "faces": 2, "solids": 1}],
                            "volume": 1000, "bbox": [10, 10, 50]}
    _hard, soft_negated = eng.verify_expected(
        bare_primitive_facts, {}, spec="a plain shaft with no threads or grooves")
    assert not any("thread/teeth/grooves" in s for s in soft_negated), soft_negated

    _hard, soft_affirmed = eng.verify_expected(
        bare_primitive_facts, {}, spec="a shaft with a helical thread along its length")
    assert any("thread/teeth/grooves" in s for s in soft_affirmed), soft_affirmed

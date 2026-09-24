"""Offline tests for the code-first teacher-data runner (lab/teacher_codefirst.py) and its
measurement module (scripts/measure_part.py).

NO API CALLS ANYWHERE in this file -- every Anthropic call site
(call_model/run_batch_stage/client()) is either untouched or only its pure, non-networked
helpers (_call_kwargs, budget arithmetic) are exercised. The one thing that DOES run real
geometry is scripts/measure_part.measure(): a tiny real build123d box-with-a-hole is built and
exported to STEP via cad_engine.run_step (a local subprocess, no network, no GPU, same pattern
tests/test_lab_teacher_refs.py already uses), because the whole point of a measurement
function is to prove it reads real geometry correctly, not a shape it is merely told exists.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "lab"))

os.environ.setdefault("PYTHONUTF8", "1")

import cad_engine  # noqa: E402
import measure_part  # noqa: E402
import teacher_codefirst as tc  # noqa: E402


# ── real geometry fixtures (local subprocess, no network) ──────────────────────────────────
BOX_WITH_THROUGH_HOLE = (
    "from build123d import *\n"
    "with BuildPart() as bp:\n"
    "    Box(40, 30, 6)\n"
    "    Cylinder(radius=5, height=6, mode=Mode.SUBTRACT)\n"
    "result = bp.part\n"
)

BOX_WITH_BLIND_HOLE = (
    "from build123d import *\n"
    "with BuildPart() as bp:\n"
    "    Box(40, 30, 10)\n"
    "    with Locations((0, 0, 5)):\n"
    "        Cylinder(radius=4, height=6, align=(Align.CENTER, Align.CENTER, Align.MAX),\n"
    "                 mode=Mode.SUBTRACT)\n"
    "result = bp.part\n"
)

TWO_SOLIDS_CODE = (
    "from build123d import *\n"
    "b1 = Box(10, 10, 10)\n"
    "b2 = Pos(40, 0, 0) * Box(10, 10, 10)\n"
    "result = Compound(children=[b1, b2])\n"
)


@pytest.fixture()
def through_hole_step(tmp_path) -> Path:
    work = tmp_path / "work1"
    work.mkdir(parents=True, exist_ok=True)
    step, _log = cad_engine.run_step(BOX_WITH_THROUGH_HOLE, work)
    return step


@pytest.fixture()
def blind_hole_step(tmp_path) -> Path:
    work = tmp_path / "work2"
    work.mkdir(parents=True, exist_ok=True)
    step, _log = cad_engine.run_step(BOX_WITH_BLIND_HOLE, work)
    return step


# 2026-09-24 fillet-detection fix: a box with its 4 VERTICAL (straight, prismatic) edges
# filleted -- exactly cf13/cf14/cf44's failure mode (measure_part.py used to silently drop
# these as "fillet/blend slivers" instead of reporting them at all).
BOX_WITH_VERTICAL_EDGE_FILLETS = (
    "from build123d import *\n"
    "with BuildPart() as bp:\n"
    "    Box(40, 30, 10)\n"
    "    fillet(bp.edges().filter_by(Axis.Z), radius=3)\n"
    "result = bp.part\n"
)

# An L-notch cut into a box, then its ONE concave (interior) vertical edge filleted -- the
# other classification cell: partial arc + CONCAVE = internal fillet/round, not a hole.
BOX_WITH_INTERNAL_CONCAVE_FILLET = (
    "from build123d import *\n"
    "with BuildPart() as bp:\n"
    "    Box(40, 30, 10)\n"
    "    with Locations((10, 7.5, 0)):\n"
    "        Box(20, 15, 10, mode=Mode.SUBTRACT)\n"
    "    fillet(bp.edges().filter_by(Axis.Z), radius=2)\n"
    "result = bp.part\n"
)


@pytest.fixture()
def vertical_fillets_step(tmp_path) -> Path:
    work = tmp_path / "work3"
    work.mkdir(parents=True, exist_ok=True)
    step, _log = cad_engine.run_step(BOX_WITH_VERTICAL_EDGE_FILLETS, work)
    return step


@pytest.fixture()
def internal_concave_fillet_step(tmp_path) -> Path:
    work = tmp_path / "work4"
    work.mkdir(parents=True, exist_ok=True)
    step, _log = cad_engine.run_step(BOX_WITH_INTERNAL_CONCAVE_FILLET, work)
    return step


# ── scripts/measure_part.py ─────────────────────────────────────────────────────────────────
def test_measure_box_with_through_hole(through_hole_step):
    m = measure_part.measure(through_hole_step)
    assert m["n_solids"] == 1
    assert m["size_mm"] == pytest.approx([40.0, 30.0, 6.0], abs=0.01)
    # volume = box - cylinder, both exact: 40*30*6 - pi*5^2*6
    expected_vol = 40 * 30 * 6 - 3.141592653589793 * 25 * 6
    assert m["volume_mm3"] == pytest.approx(expected_vol, rel=0.01)
    assert len(m["holes"]) == 1
    hole = m["holes"][0]
    assert hole["diameter_mm"] == pytest.approx(10.0, abs=0.05)
    assert hole["through"] is True
    assert hole["kind"] == "through hole"
    assert m["shafts"] == []


def test_measure_box_with_blind_hole(blind_hole_step):
    m = measure_part.measure(blind_hole_step)
    assert m["n_solids"] == 1
    assert len(m["holes"]) == 1
    hole = m["holes"][0]
    assert hole["diameter_mm"] == pytest.approx(8.0, abs=0.05)
    assert hole["through"] is False
    assert hole["kind"] == "blind hole"
    assert hole["depth_mm"] == pytest.approx(6.0, abs=0.1)


def test_measure_to_json_roundtrip(through_hole_step):
    """The measurement dict must be plain-JSON-serialisable -- it is sent straight into the
    SPEC call's prompt (build_spec_prompt -> json.dumps)."""
    m = measure_part.measure(through_hole_step)
    dumped = json.dumps(m)
    reloaded = json.loads(dumped)
    assert reloaded["n_solids"] == 1
    assert reloaded["holes"][0]["diameter_mm"] == m["holes"][0]["diameter_mm"]


def test_write_reference_volume_sidecar(through_hole_step, tmp_path):
    ref_stl = tmp_path / "reference.stl"
    sidecar_json = tmp_path / "reference_volume.json"
    m = measure_part.measure(through_hole_step)
    sidecar = measure_part.write_reference_volume_sidecar(through_hole_step, ref_stl,
                                                           sidecar_json, m)
    assert ref_stl.exists() and ref_stl.stat().st_size > 0
    assert sidecar_json.exists()
    on_disk = json.loads(sidecar_json.read_text(encoding="utf-8"))
    assert on_disk["volume_mm3"] == m["volume_mm3"]
    assert on_disk["reference_stl"] == str(ref_stl)
    assert sidecar == on_disk


def test_measure_box_with_vertical_edge_fillets_reports_4_fillets(vertical_fillets_step):
    """The exact case that caught cf13/cf14/cf44: fillet(edges().filter_by(Axis.Z)) on a box
    leaves 4 CYLINDRICAL (not torus) faces, each a 90 deg partial arc -- must be reported as
    4 grouped fillets of the requested radius, and must NOT be counted as holes."""
    m = measure_part.measure(vertical_fillets_step)
    assert m["n_solids"] == 1
    assert m["holes"] == []
    assert m["shafts"] == []
    assert len(m["fillets"]) == 1, m["fillets"]  # one group: same radius, same kind
    group = m["fillets"][0]
    assert group["count"] == 4
    assert group["radius_mm"] == pytest.approx(3.0, abs=0.05)
    assert group["kind"] == "edge fillet"


def test_measure_plate_with_through_hole_reports_zero_fillets(through_hole_step):
    m = measure_part.measure(through_hole_step)
    assert len(m["holes"]) == 1
    assert m["fillets"] == []
    assert m["chamfers"] == []


def test_measure_internal_concave_fillet_not_a_hole(internal_concave_fillet_step):
    m = measure_part.measure(internal_concave_fillet_step)
    # 5 outer/notch-corner CONVEX fillets + 1 concave (reflex) corner of the L-notch itself
    fillet_kinds = {g["kind"]: g["count"] for g in m["fillets"]}
    assert fillet_kinds.get("internal fillet/round") == 1, m["fillets"]
    assert fillet_kinds.get("edge fillet") == 5, m["fillets"]
    assert m["holes"] == []
    assert m["shafts"] == []


# ── compare_features (lab/teacher_codefirst.py's tightened keep rule) ──────────────────────
def test_compare_features_true_when_identical():
    design = {"holes": [{"diameter_mm": 15.0}], "shafts": [],
              "fillets": [{"radius_mm": 3.0, "count": 4}], "chamfers": []}
    rebuild = {"holes": [{"diameter_mm": 15.05}], "shafts": [],
               "fillets": [{"radius_mm": 2.95, "count": 4}], "chamfers": []}
    ok, problems = measure_part.compare_features(design, rebuild)
    assert ok is True
    assert problems == []


def test_compare_features_catches_dropped_fillets():
    """The cf13/cf14/cf44 case: the design has 4 corner fillets, the rebuild has none."""
    design = {"holes": [], "shafts": [], "fillets": [{"radius_mm": 3.0, "count": 4}],
              "chamfers": []}
    rebuild = {"holes": [], "shafts": [], "fillets": [], "chamfers": []}
    ok, problems = measure_part.compare_features(design, rebuild)
    assert ok is False
    assert any("fillets count differs" in p for p in problems)


def test_compare_features_catches_radius_drift_beyond_tolerance():
    design = {"holes": [], "shafts": [], "fillets": [{"radius_mm": 3.0, "count": 1}],
              "chamfers": []}
    rebuild = {"holes": [], "shafts": [], "fillets": [{"radius_mm": 3.5, "count": 1}],
               "chamfers": []}
    ok, problems = measure_part.compare_features(design, rebuild, radius_tol_mm=0.2)
    assert ok is False
    assert any("fillets size mismatch" in p for p in problems)


def test_compare_features_tolerates_differently_grouped_same_multiset():
    """Design groups 4 identical fillets in one entry; rebuild's floating-point radii split
    into two groups of 2 -- still the same 4 physical fillets, must compare equal."""
    design = {"holes": [], "shafts": [], "fillets": [{"radius_mm": 3.0, "count": 4}],
              "chamfers": []}
    rebuild = {"holes": [], "shafts": [],
               "fillets": [{"radius_mm": 2.96, "count": 2}, {"radius_mm": 3.04, "count": 2}],
               "chamfers": []}
    ok, problems = measure_part.compare_features(design, rebuild)
    assert ok is True, problems


# ── keep/reject rule (lab/teacher_codefirst.py) -- no API calls, no builds ─────────────────
def test_design_accept_ok():
    gate = {"error": None, "unscored_reason": None, "gate_hard": [], "facts": {"solids": 1}}
    ok, reason = tc.design_accept(gate)
    assert ok is True
    assert reason == ""


@pytest.mark.parametrize("gate,expect_substr", [
    ({"error": "boom", "facts": {}}, "crash"),
    ({"error": None, "unscored_reason": "inspect raised: x", "facts": {}}, "unscored"),
    ({"error": None, "unscored_reason": None, "gate_hard": ["not watertight"],
      "facts": {"solids": 1}}, "gate_hard"),
    ({"error": None, "unscored_reason": None, "gate_hard": [], "facts": {"solids": 2}},
     "n_solids"),
    ({"error": None, "unscored_reason": None, "gate_hard": [], "facts": {}}, "n_solids"),
])
def test_design_accept_rejects(gate, expect_substr):
    ok, reason = tc.design_accept(gate)
    assert ok is False
    assert expect_substr in reason


CLEAN_GATE = {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}
SAME_MEASUREMENTS = {"holes": [{"diameter_mm": 15.0}], "shafts": [], "fillets": [], "chamfers": []}


def test_keep_pair_true_only_on_match_and_clean_gate_and_same_features():
    kept, problems = tc.keep_pair("match", CLEAN_GATE, SAME_MEASUREMENTS, SAME_MEASUREMENTS)
    assert kept is True
    assert problems == []


@pytest.mark.parametrize("band,gate", [
    ("valid", CLEAN_GATE),
    ("near_miss", CLEAN_GATE),
    ("fail", CLEAN_GATE),
    (None, CLEAN_GATE),
    ("match", {"error": "crashed", "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": "x", "gate_hard": [], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": None, "gate_hard": ["bad"], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": ["[spec] no"]}),
])
def test_keep_pair_false_on_band_or_gate(band, gate):
    kept, _problems = tc.keep_pair(band, gate, SAME_MEASUREMENTS, SAME_MEASUREMENTS)
    assert kept is False


def test_keep_pair_false_when_a_feature_is_missing_from_the_rebuild():
    """The cf13/cf14/cf44 regression test: band == "match" and the gate is clean, but the
    rebuild dropped the design's 4 corner fillets entirely -- must not be kept."""
    design_m = {"holes": [], "shafts": [], "fillets": [{"radius_mm": 3.0, "count": 4}],
               "chamfers": []}
    rebuild_m = {"holes": [], "shafts": [], "fillets": [], "chamfers": []}
    kept, problems = tc.keep_pair("match", CLEAN_GATE, design_m, rebuild_m)
    assert kept is False
    assert any("fillets count differs" in p for p in problems)


# ── model call-shape rules (pure, no network) ───────────────────────────────────────────────
def test_call_kwargs_opus_omits_thinking():
    kw = tc._call_kwargs("claude-opus-5-5")
    assert "thinking" not in kw
    assert kw["output_config"] == {"effort": "high"}
    assert "temperature" not in kw and "top_p" not in kw and "top_k" not in kw


def test_call_kwargs_sonnet_sets_adaptive_thinking():
    kw = tc._call_kwargs("claude-sonnet-5")
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}


def test_call_kwargs_rejects_unknown_model():
    with pytest.raises(ValueError):
        tc._call_kwargs("claude-haiku-4-5")


# ── batch id resumability (no API calls -- pure file I/O) ──────────────────────────────────
def test_find_resumable_batch_none_when_no_file(tmp_path):
    assert tc._find_resumable_batch(tmp_path, "design", ["cf01", "cf02"]) is None


def test_record_then_find_resumable_batch(tmp_path):
    tc._record_batch_id(tmp_path, "design", "batch_abc123", ["cf01", "cf02"])
    found = tc._find_resumable_batch(tmp_path, "design", ["cf01", "cf02"])
    assert found == "batch_abc123"
    # order of custom_ids must not matter -- it's a SET match
    found2 = tc._find_resumable_batch(tmp_path, "design", ["cf02", "cf01"])
    assert found2 == "batch_abc123"


def test_find_resumable_batch_none_for_different_stage_or_ids(tmp_path):
    tc._record_batch_id(tmp_path, "design", "batch_abc123", ["cf01", "cf02"])
    assert tc._find_resumable_batch(tmp_path, "spec", ["cf01", "cf02"]) is None
    assert tc._find_resumable_batch(tmp_path, "design", ["cf01", "cf03"]) is None


def test_load_api_key_prefers_process_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert tc._load_api_key() == "sk-from-env"


def test_load_api_key_falls_back_to_cad_teacher_key_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    key_file = tmp_path / "cad-teacher.key"
    key_file.write_text("sk-from-file\n", encoding="utf-8")
    monkeypatch.setattr(tc, "CAD_TEACHER_KEY_FILE", key_file)
    assert tc._load_api_key() == "sk-from-file"


# ── seed bank sanity ─────────────────────────────────────────────────────────────────────
def test_seed_bank_is_50_unique_ids_with_valid_tiers():
    assert len(tc.SEEDS) == 50
    ids = [s["id"] for s in tc.SEEDS]
    assert len(set(ids)) == 50
    for s in tc.SEEDS:
        assert s["tier"] in (1, 2, 3, 4)
        assert isinstance(s["idea"], str) and len(s["idea"]) > 10


def test_seed_bank_passes_contamination_guard():
    """Real specbank.contamination_sets() -- if this ever trips, a seed collides with a real
    eval suite and must be reworded before any spend happens against it."""
    for s in tc.SEEDS:
        assert tc.check_contamination(s["idea"]) is None, s["id"]


# ── budget arithmetic (pure) ────────────────────────────────────────────────────────────────
def test_budget_check_raises_before_a_call_that_would_breach_the_cap():
    from datetime import datetime, timezone
    state = {"total_usd": 0.59, "calls": 3, "max_call_usd": 0.20}
    with pytest.raises(tc.BudgetStop):
        tc.budget_check("claude-sonnet-5", datetime.now(timezone.utc), 0.60, state)


def test_budget_check_noop_when_budget_is_zero():
    from datetime import datetime, timezone
    state = {"total_usd": 999.0, "calls": 1, "max_call_usd": 999.0}
    tc.budget_check("claude-sonnet-5", datetime.now(timezone.utc), 0.0, state)  # must not raise


def test_budget_check_per_call_estimate_override_avoids_worst_case_blowup():
    """The bug this guards: with reserve_calls=800 and the old "assume every call hits
    MAX_TOKENS at synchronous list price" fallback, the pre-flight estimate alone was
    hundreds of dollars -- enough to refuse to start a batch whose REAL cost was a fraction
    of that. per_call_estimate_usd must be used INSTEAD of that formula when given."""
    from datetime import datetime, timezone
    state = {"total_usd": 0.0, "calls": 0, "max_call_usd": 0.0}
    # a small, realistic per-call estimate x 800 requests must fit a modest budget
    tc.budget_check("claude-opus-5-5", datetime.now(timezone.utc), 48.0, state,
                    reserve_calls=800, per_call_estimate_usd=0.02)  # must not raise
    # the SAME reserve_calls with no override uses the old worst-case formula and must blow
    # straight through that same budget
    with pytest.raises(tc.BudgetStop):
        tc.budget_check("claude-opus-5-5", datetime.now(timezone.utc), 48.0, state,
                        reserve_calls=800)


def test_estimate_batch_call_cost_uses_batch_pricing_not_synchronous():
    """Batch price is half of list (BATCH_DISCOUNT); the estimate must use it, not the full
    synchronous MODEL_PRICES rate, or every batch pre-flight check would be needlessly (2x)
    more conservative than the real bill."""
    est = tc.estimate_batch_call_cost("claude-opus-5-5", "design")
    pin, pout = tc.batch_price("claude-opus-5-5")
    sync_pin, sync_pout = tc.MODEL_PRICES["claude-opus-5-5"]
    assert pin == pytest.approx(sync_pin / 2)
    assert pout == pytest.approx(sync_pout / 2)
    assert est > 0


def test_record_spend_batch_true_uses_half_the_synchronous_cost(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "SPEND_LEDGER", tmp_path / "spend.jsonl")  # never touch the real ledger

    class _Usage:
        input_tokens = 1000
        output_tokens = 1000
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    state_sync = {"total_usd": 0.0, "calls": 0, "max_call_usd": 0.0}
    state_batch = {"total_usd": 0.0, "calls": 0, "max_call_usd": 0.0}
    m1 = tc.record_spend("claude-opus-5-5", _Usage(), 0.0, state_sync, batch=False)
    m2 = tc.record_spend("claude-opus-5-5", _Usage(), 0.0, state_batch, batch=True)
    assert m2["cost"] == pytest.approx(m1["cost"] * tc.BATCH_DISCOUNT)

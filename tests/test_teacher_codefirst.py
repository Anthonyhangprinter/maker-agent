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


def test_keep_pair_true_only_on_match_and_clean_gate():
    clean_gate = {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}
    assert tc.keep_pair("match", clean_gate) is True


@pytest.mark.parametrize("band,gate", [
    ("valid", {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    ("near_miss", {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    ("fail", {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    (None, {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    ("match", {"error": "crashed", "unscored_reason": None, "gate_hard": [], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": "x", "gate_hard": [], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": None, "gate_hard": ["bad"], "gate_spec": []}),
    ("match", {"error": None, "unscored_reason": None, "gate_hard": [], "gate_spec": ["[spec] no"]}),
])
def test_keep_pair_false(band, gate):
    assert tc.keep_pair(band, gate) is False


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

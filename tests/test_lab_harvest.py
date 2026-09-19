"""Offline tests for lab/harvest.py (Task 3, fix round 1 2026-09-19): verdict table
(including the new "unscored" verdict and the independent re-gating that replaced
trusting fluid_gen._materialize's own gate), dedup/max-pairs, teacher promotion +
think-unavailable skip, tier-first scheduling, unit-gate refusals (including
--check-gate), ledger/pair/status row shapes, pass rates by pass/tier, aborts and infra
errors mid-candidate, the code-model pin refusal, and real-subprocess tests that drive
the ACTUAL lab.harvest.main() (not a look-alike helper -- Task 3 rulings: "that is how
two defects got through a review round on Task 2").

**Gating seam (fix round 1).** `sample_spec`/`_execute_with_salvage` now call
`fluid_gen._materialize` for EXECUTION side effects only, then independently re-gate via
`harvest._regate` (which calls `engine.run_inspect`/`engine.parse_facts`/
`engine.reconcile_expected`/`engine.verify_expected` directly). Orchestration-focused
tests (dedup, max-pairs, scheduling, ledger writing) mock `harvest._regate` itself via
`_patch_regate_passthrough` below, rather than faking a real build.step + run_inspect
round trip for every test -- `_regate`'s OWN correctness (H1/H2) is tested directly,
separately, against `engine.run_inspect` mocks.

Everything model/GPU-touching in the offline tests is monkeypatched at the same seam
test_fluid_repair_think.py already established for this codebase (fluid_gen._materialize,
engine._ollama / generate_code_raw) -- no network, no GPU, no services. The real-subprocess
tests near the bottom of this file patch ONLY the model call (cad_engine._ollama, or
cad_engine.run_step for the H3 abort-during-execution test) and the arm switch
(lab._armwindow._run_arms/_ensure_resident_up), inside a spawned subprocess, and drive
lab.harvest.main() for real.
"""
import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from lab import harvest  # noqa: E402
from lab import specbank  # noqa: E402
from lab import _armwindow as aw  # noqa: E402
import cad_engine as engine  # noqa: E402
import fluid_gen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own empty state dir and spec bank -- never the real
    lab/state/*.jsonl or the real 434-row bank. Also resets the module-level abort flag
    (Task 3 fix H3) -- it is never reset in production (one abort per process lifetime),
    but tests share one interpreter across many cases."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setattr(harvest, "STATE_DIR", d)
    monkeypatch.setattr(harvest, "LEDGER_FILE", d / "ledger.jsonl")
    monkeypatch.setattr(harvest, "PAIRS_FILE", d / "pairs.jsonl")
    monkeypatch.setattr(harvest, "REVIEW_FILE", d / "review.jsonl")
    monkeypatch.setattr(harvest, "CANDIDATES_FILE", d / "candidates.jsonl")
    # Fix round 3, H2b: isolate the confirm_strength upgrade log too -- omitting this
    # would let a test's UPGRADES_FILE row land in the REAL lab/state/upgrades.jsonl.
    monkeypatch.setattr(harvest, "UPGRADES_FILE", d / "upgrades.jsonl")
    monkeypatch.setattr(harvest, "PROGRESS_FILE", d / "progress.json")
    monkeypatch.setattr(harvest, "STATUS_FILE", d / "status.json")
    monkeypatch.setattr(harvest, "PAUSED_FILE", d / "paused")
    monkeypatch.setattr(harvest, "BUILDS_DIR", d / "builds")
    monkeypatch.setattr(harvest, "SYSTEMS_DIR", d / "systems")
    monkeypatch.setattr(harvest, "BUILD_LOCK_FILE", d / "cad-build.lock")
    monkeypatch.setattr(specbank, "SPECS_FILE", d / "specs.jsonl")
    monkeypatch.setattr(harvest, "_current_arm_alias", lambda: "test-arm")
    monkeypatch.setattr(aw, "_ABORT_REQUESTED", False)
    return d


def _plant_bank(rows: list[dict]) -> None:
    specbank.SPECS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(specbank.SPECS_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _spec_row(id_, tier=2, group="plate", **extra) -> dict:
    row = {"id": id_, "spec": f"spec text for {id_}", "tier": tier, "group": group,
          "source": "teacher-suite", "key": f"key-{id_}", "added": "2026-09-19T00:00:00Z"}
    row.update(extra)
    return row


def _default_cfg(**overrides) -> dict:
    cfg = {"night_start": "22:00", "night_end": "07:00", "day_allowed": True,
          "hours_per_day": 12, "unit_minutes": 25, "candidates": 3,
          "temps": [0.2, 0.5, 0.8],
          # Fix round 2, Section C: tier 3-4 defaults, mirrored from
          # cad_v5.config._LAB_DEFAULTS so tests exercise the real shape.
          "candidates_tier34": 5, "temps_tier34": [0.2, 0.35, 0.5, 0.65, 0.8],
          "max_pairs_per_spec": 2, "teacher_passes": ["think"],
          "attempt_caps": {"student": 2, "teacher": 5},
          "agreement": {"volume_tol_pct": 0.05, "bbox_tol_mm": 0.05, "bore_round_mm": 0.01},
          "strict": {"envelope_tol_mm": 0.2},
          "keep_builds": 500}
    cfg.update(overrides)
    return cfg


# Fix round 2, Section A: a simple-box default so DIFFERENT fake codes sampled for the
# SAME spec, all regated via _patch_regate_passthrough's default facts, naturally
# AGREE on signature -- matching how two genuinely-correct-but-differently-written
# programs behave in production. Tests that need to exercise DISAGREEMENT pass an
# explicit `facts=` override.
_DEFAULT_CLEAN_FACTS = {"solids": 1, "faces": 6, "cyl_faces": 0, "cone_faces": 0,
                        "volume": 1000.0, "bbox": [10.0, 10.0, 10.0], "bores": []}


def _clean_m(gate_hard=None, gate_spec=None, gate_adv=None, error=None, facts=None,
            unscored_reason=None) -> dict:
    """A fully-measured, gate-clean `m` dict -- the shape _regate() itself would produce
    for a valid, fully-inspected candidate. Must include solids/volume/bbox (Task 3 fix
    H2: _classify now requires all three before it will even consider "good") plus
    faces/cyl_faces/cone_faces (fix round 2: signature() also fails closed on those)."""
    return {"facts": facts if facts is not None else dict(_DEFAULT_CLEAN_FACTS),
           "error": error, "gate_hard": gate_hard or [], "gate_spec": gate_spec or [],
           "gate_adv": gate_adv or [], "unscored_reason": unscored_reason}


def _confirm_via_reference(monkeypatch, row_id: str = "s1") -> dict:
    """Fix round 2: a single-candidate test that only cares about some OTHER mechanic
    (system_sha1 dedup, model-string propagation, salvage row shape...) needs its lone
    candidate to actually become a pair. A reference-band "match" confirms on its own
    (Section A, path 1), same as before this fix round, without needing a second
    agreeing candidate. Returns the spec row to use (carries reference_stl)."""
    monkeypatch.setattr(harvest, "score_against_reference",
                        lambda *a, **k: {"band": "match", "reference": "ref.stl"})
    return _spec_row(row_id, reference_stl="ref.stl")


def _patch_regate_passthrough(monkeypatch, facts=None):
    """Bypasses the real engine.run_inspect/parse_facts/reconcile_expected/verify_expected
    round trip _regate() does: a crash (m_fluid["error"] set) still reports as a crash;
    otherwise returns a clean, fully-measured `m`. Lets orchestration tests (dedup,
    max-pairs, scheduling) exercise the real sample_spec/_execute_with_salvage control
    flow without needing a real build.step file on disk. _regate's OWN correctness is
    tested directly and separately below (test_regate_*)."""
    def fake_regate(m_fluid, build_dir, spec):
        if m_fluid.get("error"):
            return {"error": m_fluid["error"], "facts": {}, "gate_hard": [],
                    "gate_spec": [], "gate_adv": [], "unscored_reason": None}
        return _clean_m(facts=facts)
    monkeypatch.setattr(harvest, "_regate", fake_regate)


def _patch_generate_sequence(monkeypatch, codes: list[str], mode: str = "student"):
    calls = {"n": 0}

    def fake_generate(spec, notes, temperature):
        code = codes[calls["n"]]
        calls["n"] += 1
        return code, {"model": "local:test-arm", "system": "SYS",
                     "prompt": f"PROMPT for {code}",
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    attr = "_student_generate" if mode == "student" else "_teacher_generate"
    monkeypatch.setattr(harvest, attr, fake_generate)
    return calls


# ---------------------------------------------------------------------------
# Verdict table (_classify) -- pure function, given an already-gated `m`
# ---------------------------------------------------------------------------

def test_classify_good_with_no_reference(tmp_path):
    verdict, band = harvest._classify(_clean_m(), None, tmp_path)
    assert verdict == "good"
    assert band == {}


def test_classify_none_when_build_crashed(tmp_path):
    verdict, band = harvest._classify(_clean_m(error="boom"), None, tmp_path)
    assert verdict == "none"
    assert band == {}


def test_classify_unscored_when_marked_unscored(tmp_path):
    """Task 3 fix H2: an unscored `m` (inspect invalid/raised, or a fact missing) is
    NEVER "good", regardless of what its (necessarily empty) gate_hard/gate_spec/band
    happen to look like -- this was the exact shape of bug the review found."""
    verdict, band = harvest._classify(
        _clean_m(facts={}, unscored_reason="inspect reported the STEP invalid"),
        None, tmp_path)
    assert verdict == "unscored"
    assert band == {}


@pytest.mark.parametrize("missing_key", ["solids", "volume", "bbox"])
def test_classify_unscored_when_a_required_fact_is_missing(tmp_path, missing_key):
    """Task 3 fix H2: "good" requires solids, volume AND bbox measured -- _classify
    itself refuses to call anything good without them, independent of _regate."""
    facts = {"solids": 1, "volume": 1000.0, "bbox": [10.0, 10.0, 10.0]}
    del facts[missing_key]
    verdict, _ = harvest._classify(_clean_m(facts=facts), None, tmp_path)
    assert verdict == "unscored"


def test_classify_none_when_gate_hard_present(tmp_path):
    verdict, _ = harvest._classify(_clean_m(gate_hard=["no solid produced"]), None, tmp_path)
    assert verdict == "none"


def test_classify_silver_when_gate_spec_present(tmp_path):
    verdict, _ = harvest._classify(_clean_m(gate_spec=["[spec] wrong length"]), None, tmp_path)
    assert verdict == "silver"


def test_classify_gate_adv_alone_is_still_good(tmp_path):
    """Advisory-only notes never block "good" -- only [spec]-tagged gate_spec findings do."""
    verdict, _ = harvest._classify(_clean_m(gate_adv=["a cosmetic note"]), None, tmp_path)
    assert verdict == "good"


def test_classify_good_with_reference_match_band(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "score_against_reference",
                        lambda *a, **k: {"band": "match", "reference": "ref.stl"})
    verdict, band = harvest._classify(_clean_m(), "ref.stl", tmp_path)
    assert verdict == "good"
    assert band["band"] == "match"


def test_classify_none_with_reference_valid_band(monkeypatch, tmp_path):
    """"valid" (not "match") with a reference present is neither good nor a fail pair --
    Task 3 rulings: owner-reference rows use the STRICT rule, only "match" counts good."""
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {"band": "valid"})
    verdict, _ = harvest._classify(_clean_m(), "ref.stl", tmp_path)
    assert verdict == "none"


def test_classify_fail_with_reference_near_miss_band(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {
        "band": "near_miss", "chamfer_mm": 5.2, "volume_diff_pct": 12.0})
    verdict, band = harvest._classify(_clean_m(), "ref.stl", tmp_path)
    assert verdict == "fail"
    assert band["band"] == "near_miss"


def test_classify_none_with_reference_fail_band(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {"band": "fail"})
    verdict, _ = harvest._classify(_clean_m(), "ref.stl", tmp_path)
    assert verdict == "none"


def test_classify_silver_takes_priority_over_near_miss_band(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {"band": "near_miss"})
    verdict, _ = harvest._classify(_clean_m(gate_spec=["[spec] wrong"]), "ref.stl", tmp_path)
    assert verdict == "silver"


def test_classify_unscored_when_reference_scoring_raises(monkeypatch, tmp_path):
    """Task 3 rulings: "Any exception inside scoring/classification = unscored, never
    good." score_against_reference itself never raises per its own contract, but
    _classify must not assume that forever."""
    def boom(*a, **k):
        raise RuntimeError("trimesh blew up")
    monkeypatch.setattr(harvest, "score_against_reference", boom)
    verdict, _ = harvest._classify(_clean_m(), "ref.stl", tmp_path)
    assert verdict == "unscored"


def test_row_is_good_requires_confirmation_not_just_classify_good(tmp_path):
    """Fix round 2, Section A: _classify's "good" (gate-clean) is necessary but no
    longer sufficient for _row_is_good (used for pass-rate stats) -- only a CONFIRMED
    good counts ("reference" or "agreement"), matching the new rule that a lone
    gate-clean candidate with no partner is "unconfirmed", not good (the exact class of
    defect this fix round exists to catch: V064/V066 were both gate-clean and wrong)."""
    m = _clean_m()
    verdict, _ = harvest._classify(m, None, tmp_path)
    assert verdict == "good"
    unconfirmed_row = harvest._ledger_row(unit_id="u1", spec_id="s1", tier=2, mode="student",
                              candidate="0", temperature=0.2, m=m, band_info={},
                              usage={}, model="local:x", build_dir=None, seconds=1.0,
                              agreement="unconfirmed")
    assert harvest._row_is_good(unconfirmed_row) is False
    assert harvest._row_is_good({**unconfirmed_row, "agreement": "agreement"}) is True
    assert harvest._row_is_good({**unconfirmed_row, "agreement": "reference"}) is True
    assert harvest._row_is_good({**unconfirmed_row, "agreement": "split"}) is False


def test_row_is_good_is_false_for_an_unscored_row_even_with_empty_gate_lists():
    """Task 3 fix H2, the exact regression: ok=True, empty gate_hard/gate_spec, band=None
    used to satisfy _is_good by accident for an unscored row. unscored_reason must veto
    it regardless."""
    row = {"ok": True, "gate_hard": [], "gate_spec": [], "band": None,
          "unscored_reason": "inspect reported the STEP invalid"}
    assert harvest._row_is_good(row) is False


# ---------------------------------------------------------------------------
# _regate: independent re-gating (Task 3 fix H1/H2), against engine.run_inspect mocks
# ---------------------------------------------------------------------------

def _inspect_output(solids=1, faces=10, volume=1000.0, bbox=(10.0, 10.0, 10.0)) -> str:
    return (f"Solids: {solids}\nFaces: {faces}\nVolume: {volume}\n"
           f"Bbox: {bbox[0]} x {bbox[1]} x {bbox[2]}\n")


def test_regate_passes_through_a_real_crash(tmp_path):
    m = harvest._regate({"error": "the script failed to run: boom"}, tmp_path, "a spec")
    assert m["error"] == "the script failed to run: boom"
    assert m["unscored_reason"] is None
    assert harvest._classify(m, None, tmp_path)[0] == "none"


def test_regate_catches_unfused_bodies_that_fluid_gate_missed(monkeypatch, tmp_path):
    """H1 (repro1b.py): fluid_gen.expected_for never sets expected.solids, so its own
    gate never fires the unfused-bodies HARD check on a part that came back as several
    disconnected bodies -- confirmed against the real engine.verify_expected/
    fluid_gen.expected_for on 94caffe (3 solids, gate_hard == []). _regate must catch it
    independently via engine.reconcile_expected, which DOES set expected.solids."""
    monkeypatch.setattr(engine, "run_inspect",
                        lambda p: {"valid": True, "output": _inspect_output(
                            solids=3, faces=24, volume=12000.0, bbox=(80.0, 40.0, 40.0))})
    spec = "an L-bracket 80mm long with a stiffening gusset"

    # Reproduce the pre-fix behaviour for comparison: fluid_gen's own weak gate.
    old_expected = fluid_gen.expected_for(spec)
    old_hard, _ = engine.verify_expected(
        {"solids": 3, "faces": 24, "volume": 12000.0, "bbox": [80.0, 40.0, 40.0]},
        old_expected, spec=spec)
    assert old_hard == [], "fluid's own gate should still miss this (documents the bug)"

    m = harvest._regate({"error": None}, tmp_path, spec)
    assert m["error"] is None
    assert m["unscored_reason"] is None
    assert any("separate bodies" in h or "fused solid" in h for h in m["gate_hard"])
    assert harvest._classify(m, None, tmp_path)[0] != "good"


def test_regate_unscored_when_inspect_reports_invalid(monkeypatch, tmp_path):
    """H2 (repro2_inspect_swallowed.py): fluid_gen._materialize swallows an
    insp["valid"] is False result (bare except Exception: pass), reporting error=None
    with empty facts/gates -- which the OLD _classify called "good". _regate must
    independently detect this and mark it unscored."""
    monkeypatch.setattr(engine, "run_inspect",
                        lambda p: {"valid": False, "output": "ERROR bad step"})
    m = harvest._regate({"error": None}, tmp_path, "an 80x50x6mm plate with four M4 holes")
    assert m["error"] is None
    assert m["unscored_reason"] == "inspect reported the STEP invalid"
    assert harvest._classify(m, None, tmp_path)[0] == "unscored"


def test_regate_unscored_when_inspect_raises(monkeypatch, tmp_path):
    """H2, the second repro2 case: inspect itself raising (e.g. a timeout) is ALSO
    swallowed by fluid_gen._materialize's bare except."""
    def raiser(p):
        raise RuntimeError("inspect timed out")
    monkeypatch.setattr(engine, "run_inspect", raiser)
    m = harvest._regate({"error": None}, tmp_path, "an 80x50x6mm plate with four M4 holes")
    assert m["error"] is None
    assert "inspect timed out" in m["unscored_reason"]
    assert harvest._classify(m, None, tmp_path)[0] == "unscored"


def test_regate_unscored_when_a_required_fact_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "run_inspect",
                        lambda p: {"valid": True, "output": "Solids: 1\nFaces: 6\n"})
    m = harvest._regate({"error": None}, tmp_path, "a 10x10x10mm cube")
    assert m["unscored_reason"] is not None
    assert "volume" in m["unscored_reason"] or "bbox" in m["unscored_reason"]


def test_regate_signal_during_inspect_propagates_not_swallowed(monkeypatch, tmp_path):
    def raiser(p):
        raise aw.SpecgenAborted("terminated by signal 15 (SIGTERM)")
    monkeypatch.setattr(engine, "run_inspect", raiser)
    with pytest.raises(aw.SpecgenAborted):
        harvest._regate({"error": None}, tmp_path, "a spec")


def test_regate_good_candidate_still_passes(monkeypatch, tmp_path):
    """Sanity: a correctly measured, gate-clean candidate is still "good" post-fix."""
    monkeypatch.setattr(engine, "run_inspect",
                        lambda p: {"valid": True, "output": _inspect_output()})
    m = harvest._regate({"error": None}, tmp_path, "a 10x10x10mm cube")
    assert m["unscored_reason"] is None
    assert m["gate_hard"] == []
    assert harvest._classify(m, None, tmp_path)[0] == "good"


# ---------------------------------------------------------------------------
# Geometric signature + independent agreement (fix round 2, Section A). Fixtures are
# the REAL facts/spec text of the two reviewer-caught cases, saved to
# tests/fixtures/harvest_agreement_fixtures.json before the pre-round-2 harvest state
# was moved aside (see the fixture-writing commit for provenance).
# ---------------------------------------------------------------------------

_FIXTURES_PATH = HERE / "tests" / "fixtures" / "harvest_agreement_fixtures.json"


def _load_fixtures() -> dict:
    return json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))


_AGREEMENT_CFG = {"volume_tol_pct": 0.05, "bbox_tol_mm": 0.05, "bore_round_mm": 0.01}


def test_signature_v066_two_temperatures_disagree_on_face_counts():
    """The real regression this fix round exists for: t:teacher-batch2:V066 sampled at
    T=0.2 makes a slot (Z-axis cylinder against an X-thin divider) where T=0.5 makes the
    stated circular hole -- the SAME spec, two wrong-vs-right answers, both gate-clean
    under the pre-fix-round-2 gate. The separating field, checked here explicitly: total
    face count (19 vs 17) -- cylindrical face count (2 vs 1) and volume (0.35% apart,
    over the 0.05% tolerance) also independently disagree, but total faces is the first
    field signatures_agree checks after solids, so it is the one that actually trips."""
    fx = _load_fixtures()
    sig_a = harvest.signature(fx["v066_t02_wrong_slot"]["facts"], _AGREEMENT_CFG)
    sig_b = harvest.signature(fx["v066_t05_correct_hole"]["facts"], _AGREEMENT_CFG)
    assert sig_a is not None and sig_b is not None
    assert sig_a["faces"] == 19 and sig_b["faces"] == 17
    assert sig_a["faces"] != sig_b["faces"]
    assert sig_a["cyl_faces"] != sig_b["cyl_faces"]
    assert not harvest.signatures_agree(sig_a, sig_b, _AGREEMENT_CFG)


def test_signature_v066_agrees_with_itself():
    """Sanity companion to the disagreement test above: a signature always agrees with
    an identical copy of itself (reflexivity), so the disagreement above is a real
    tolerance/field mismatch, not a bug that makes signatures_agree() always False."""
    fx = _load_fixtures()
    sig = harvest.signature(fx["v066_t05_correct_hole"]["facts"], _AGREEMENT_CFG)
    assert harvest.signatures_agree(sig, dict(sig), _AGREEMENT_CFG)


def test_signature_v064_missing_no_field_but_is_still_a_real_signature():
    """V064's own facts are fully measured (unlike V066's disagreement, this candidate's
    rejection comes from the strict spec checks below, not from a signature mismatch) --
    documented here so a reader can see signature() does not itself reject it."""
    fx = _load_fixtures()
    sig = harvest.signature(fx["v064_wrong_sheared_lip"]["facts"], _AGREEMENT_CFG)
    assert sig is not None
    assert sig["bbox_sorted"] == [53.0, 130.0, 180.0]


@pytest.mark.parametrize("missing_key", ["solids", "faces", "cyl_faces", "cone_faces"])
def test_signature_fails_closed_on_missing_required_fact(missing_key):
    facts = dict(_DEFAULT_CLEAN_FACTS)
    del facts[missing_key]
    assert harvest.signature(facts) is None


@pytest.mark.parametrize("sentinel_key", ["cyl_faces", "cone_faces"])
def test_signature_fails_closed_on_the_inspect_unavailable_sentinel(sentinel_key):
    """scripts/inspect reports -1 for cyl_faces/cone_faces when classification was
    unavailable (older build123d) -- two candidates that both failed to classify must
    never look like they "agree" by both carrying the same -1 sentinel."""
    facts = dict(_DEFAULT_CLEAN_FACTS)
    facts[sentinel_key] = -1
    assert harvest.signature(facts) is None


def test_signatures_agree_is_false_when_either_signature_is_none():
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    assert not harvest.signatures_agree(None, sig)
    assert not harvest.signatures_agree(sig, None)
    assert not harvest.signatures_agree(None, None)


def test_signatures_agree_within_bbox_and_volume_tolerance():
    a = harvest.signature(_DEFAULT_CLEAN_FACTS)
    close_facts = dict(_DEFAULT_CLEAN_FACTS, bbox=[10.04, 10.0, 10.0], volume=1000.4)
    b = harvest.signature(close_facts)
    assert harvest.signatures_agree(a, b, _AGREEMENT_CFG)


def test_signatures_disagree_outside_bbox_tolerance():
    a = harvest.signature(_DEFAULT_CLEAN_FACTS)
    far_facts = dict(_DEFAULT_CLEAN_FACTS, bbox=[10.1, 10.0, 10.0])
    b = harvest.signature(far_facts)
    assert not harvest.signatures_agree(a, b, _AGREEMENT_CFG)


def test_signatures_disagree_outside_volume_tolerance():
    a = harvest.signature(_DEFAULT_CLEAN_FACTS)
    far_facts = dict(_DEFAULT_CLEAN_FACTS, volume=1002.0)   # 0.2% off, over 0.05%
    b = harvest.signature(far_facts)
    assert not harvest.signatures_agree(a, b, _AGREEMENT_CFG)


def test_signatures_agree_bore_diameters_equal_after_rounding():
    a = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, bores=[12.001]))
    b = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, bores=[12.004]))
    assert harvest.signatures_agree(a, b, _AGREEMENT_CFG)   # both round to 12.00
    c = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, bores=[12.02]))
    assert not harvest.signatures_agree(a, c, _AGREEMENT_CFG)


def test_signatures_never_agree_on_zero_or_negative_volume():
    """Fix round 3, LOW2: the old `avg_vol <= 0: return a["volume"] == b["volume"]`
    branch let two candidates that BOTH measured a broken zero (or negative) volume
    "agree" with each other on the strength of that shared brokenness -- exactly
    backwards, since a non-positive volume means the measurement itself is untrustworthy,
    never that it matches."""
    zero_a = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=0.0))
    zero_b = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=0.0))
    assert not harvest.signatures_agree(zero_a, zero_b, _AGREEMENT_CFG)
    neg = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=-5.0))
    assert not harvest.signatures_agree(zero_a, neg, _AGREEMENT_CFG)
    assert not harvest.signatures_agree(neg, neg, _AGREEMENT_CFG)


def _pool_item(sig, temperature=0.2, origin="new", **extra) -> dict:
    item = {"origin": origin, "signature": sig, "temperature": temperature}
    item.update(extra)
    return item


def test_resolve_agreement_empty_pool_is_unconfirmed():
    assert harvest.resolve_agreement([], _AGREEMENT_CFG) == ([], "unconfirmed")


def test_resolve_agreement_single_candidate_is_unconfirmed():
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    pool = [_pool_item(sig)]
    winners, tag = harvest.resolve_agreement(pool, _AGREEMENT_CFG)
    assert winners == [] and tag == "unconfirmed"


def test_resolve_agreement_two_agreeing_candidates_confirm_each_other():
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    a, b = _pool_item(sig, temperature=0.2), _pool_item(sig, temperature=0.5)
    winners, tag = harvest.resolve_agreement([a, b], _AGREEMENT_CFG)
    assert tag == "agreement"
    assert sorted(id(w) for w in winners) == sorted([id(a), id(b)])


def test_resolve_agreement_v066_real_pair_never_confirms():
    """The real V066 case end to end through resolve_agreement, not just
    signatures_agree: two gate-clean candidates for one spec that disagree produce
    exactly the "split" the brief calls for -- nobody confirmed."""
    fx = _load_fixtures()
    sig_a = harvest.signature(fx["v066_t02_wrong_slot"]["facts"], _AGREEMENT_CFG)
    sig_b = harvest.signature(fx["v066_t05_correct_hole"]["facts"], _AGREEMENT_CFG)
    pool = [_pool_item(sig_a, temperature=0.2), _pool_item(sig_b, temperature=0.5)]
    winners, tag = harvest.resolve_agreement(pool, _AGREEMENT_CFG)
    assert winners == []
    assert tag == "split"


def test_resolve_agreement_largest_cluster_must_be_strictly_larger():
    """A 2-2 tie between two clusters confirms nobody (Task 3 ruling, verbatim: "the
    largest cluster wins only if it has at least 2 members AND is strictly larger than
    every other cluster")."""
    sig1 = harvest.signature(_DEFAULT_CLEAN_FACTS)
    sig2 = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=5000.0))
    pool = [_pool_item(sig1, 0.2), _pool_item(sig1, 0.35),
           _pool_item(sig2, 0.5), _pool_item(sig2, 0.65)]
    winners, tag = harvest.resolve_agreement(pool, _AGREEMENT_CFG)
    assert winners == [] and tag == "split"


def test_resolve_agreement_larger_cluster_wins_over_a_smaller_one():
    sig1 = harvest.signature(_DEFAULT_CLEAN_FACTS)
    sig2 = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=5000.0))
    a, b, c = (_pool_item(sig1, 0.2), _pool_item(sig1, 0.35), _pool_item(sig1, 0.5))
    d = _pool_item(sig2, 0.65)
    winners, tag = harvest.resolve_agreement([a, b, c, d], _AGREEMENT_CFG)
    assert tag == "agreement"
    assert sorted(id(w) for w in winners) == sorted([id(a), id(b), id(c)])


def test_resolve_agreement_existing_anchor_confirms_a_lone_new_candidate():
    """An already-confirmed pair's signature (an "existing_anchor" pool item, from
    PairIndex.signatures_for) confirms a lone new candidate on its own -- no second NEW
    sample required, matching production's cross-unit confirmation path."""
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    anchor = {"origin": "existing", "existing_anchor": True, "signature": sig,
             "fingerprint": None, "temperature": 0.0}
    new = _pool_item(sig, temperature=0.2)
    winners, tag = harvest.resolve_agreement([anchor, new], _AGREEMENT_CFG)
    assert tag == "agreement"
    assert winners == [new]   # the anchor itself is never a "winner" to promote


def test_resolve_agreement_existing_anchor_wins_even_alone():
    """A single new candidate matching an anchor is enough (size 1 is fine when
    anchored) -- unlike two new candidates, where size 1 is "unconfirmed"."""
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    anchor = {"origin": "existing", "existing_anchor": True, "signature": sig,
             "fingerprint": None, "temperature": 0.0}
    new = _pool_item(sig, temperature=0.2)
    winners, tag = harvest.resolve_agreement([anchor, new], _AGREEMENT_CFG)
    assert tag == "agreement" and winners == [new]


def test_resolve_agreement_anchored_cluster_with_no_new_members_is_split_not_unconfirmed():
    """Fix round 3, LOW1: when the anchor's own cluster gains no new member (a new
    candidate this round disagreed with it), the outcome is "split" -- a real
    disagreement occurred, not merely "nobody to compare against yet". Before this fix,
    resolve_agreement returned `([], "agreement")` here (an empty-winners "agreement" is
    nonsensical -- nothing was actually confirmed), which then made the caller label the
    disagreeing new candidate "unconfirmed" instead of "split"."""
    sig_anchor = harvest.signature(_DEFAULT_CLEAN_FACTS)
    sig_other = harvest.signature(dict(_DEFAULT_CLEAN_FACTS, volume=5000.0))
    anchor = {"origin": "existing", "existing_anchor": True, "signature": sig_anchor,
             "fingerprint": None, "temperature": 0.0}
    new = _pool_item(sig_other, temperature=0.2)
    winners, tag = harvest.resolve_agreement([anchor, new], _AGREEMENT_CFG)
    assert winners == [] and tag == "split"


def test_cluster_by_signature_is_order_independent():
    """Complete-link clustering (fix round 3, M2, replacing fix round 2's union-find
    design): shuffling the pool must never change which candidates end up in the same
    cluster -- processing order is derived from each item's own stable key, never from
    its position in the caller's list."""
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    pool = [_pool_item(sig, t) for t in (0.2, 0.35, 0.5)]
    import random
    shuffled = list(pool)
    random.Random(7).shuffle(shuffled)
    c1 = harvest._cluster_by_signature(pool, _AGREEMENT_CFG)
    c2 = harvest._cluster_by_signature(shuffled, _AGREEMENT_CFG)
    assert len(c1) == len(c2) == 1
    assert len(c1[0]) == len(c2[0]) == 3


def test_cluster_by_signature_complete_link_rejects_a_tolerance_chain():
    """Fix round 3, M2, the reviewer's own counter-example: three candidates whose bbox
    (100.00/100.05/100.10mm on one axis) and volume (1000.00/1000.25/1000.50) each drift
    from their IMMEDIATE neighbour by exactly one tolerance step, but the two EXTREMES
    (100.00 vs 100.10, a 0.10mm gap over the 0.05mm bbox_tol_mm) do not themselves agree.
    Single-link (union-find) transitivity fuses all three into one cluster via the A~B,
    B~C chain even though A and C disagree -- complete-link must never do that: at most
    a 2-member cluster may form here, never a 3-member one."""
    cfg = {"volume_tol_pct": 0.05, "bbox_tol_mm": 0.05, "bore_round_mm": 0.01}
    facts_a = dict(_DEFAULT_CLEAN_FACTS, bbox=[100.00, 50.0, 20.0], volume=1000.00)
    facts_b = dict(_DEFAULT_CLEAN_FACTS, bbox=[100.05, 50.0, 20.0], volume=1000.25)
    facts_c = dict(_DEFAULT_CLEAN_FACTS, bbox=[100.10, 50.0, 20.0], volume=1000.50)
    sig_a, sig_b, sig_c = (harvest.signature(f, cfg) for f in (facts_a, facts_b, facts_c))
    # Sanity: neighbours agree, extremes do not (confirms the fixture is a real chain).
    assert harvest.signatures_agree(sig_a, sig_b, cfg)
    assert harvest.signatures_agree(sig_b, sig_c, cfg)
    assert not harvest.signatures_agree(sig_a, sig_c, cfg)

    pool = [_pool_item(sig_a, 0.2), _pool_item(sig_b, 0.35), _pool_item(sig_c, 0.5)]
    clusters = harvest._cluster_by_signature(pool, cfg)
    assert all(len(c) < 3 for c in clusters)
    assert sum(len(c) for c in clusters) == 3   # every item still accounted for


# ---------------------------------------------------------------------------
# Strict spec checks (fix round 2, Section B) -- harvest-local, fail closed, additional
# to and independent of the engine's own gate.
# ---------------------------------------------------------------------------

def test_strict_envelope_check_rejects_v064():
    """The real regression: V064's stated 180x130x55mm envelope measures 180x130x53mm
    (a mis-sized lip cutter sheared 2mm off the top) -- within the engine gate's own
    2.75mm axis tolerance, but over this harvest-local check's 0.2mm one."""
    fx = _load_fixtures()
    v064 = fx["v064_wrong_sheared_lip"]
    reason = harvest.strict_envelope_check(v064["spec"], v064["facts"])
    assert reason is not None
    assert "strict_envelope" in reason
    assert "2.00mm" in reason


def test_strict_envelope_check_passes_a_matching_envelope():
    fx = _load_fixtures()
    v066 = fx["v066_t02_wrong_slot"]
    assert harvest.strict_envelope_check(v066["spec"], v066["facts"]) is None


def test_strict_envelope_check_none_when_spec_states_no_envelope():
    assert harvest.strict_envelope_check("a bracket with two M4 holes",
                                         {"bbox": [1.0, 2.0, 3.0]}) is None


def test_strict_envelope_check_skips_assembly_specs():
    spec = "an assembly of two identical 40x25x5mm ears and a 60mm shaft"
    # A wildly wrong bbox would fail if this were treated as a single-part envelope --
    # it must not be, since _ASSEMBLY_SPEC_RE matches "two identical".
    assert harvest.strict_envelope_check(spec, {"bbox": [1.0, 1.0, 1.0]}) is None


@pytest.mark.parametrize("phrasing", [
    "a housing 100x50x20mm with a lid",
    "a housing 100 x 50 x 20 mm with a lid",
    "a housing 100mm x 50mm x 20mm with a lid",
    "a housing 100 by 50 by 20mm with a lid",
])
def test_spec_envelope_dims_mm_accepts_spaced_and_by_forms(phrasing):
    assert harvest._spec_envelope_dims_mm(phrasing) == [100.0, 50.0, 20.0]


def test_spec_envelope_dims_mm_none_when_ambiguous_multiple_matches():
    spec = "a base 60x40x6mm flange and a 6x40x60mm upright"
    assert harvest._spec_envelope_dims_mm(spec) is None


def test_strict_through_holes_check_no_verdict_on_v064_real_facts():
    """Fix round 3, M1: the OLD undercount test rejected V064 because its aggregate
    `through_holes` measured 0 against a stated count of 1 -- but that classifier only
    recognises one hole orientation, so a bare low/zero AGGREGATE count is not
    trustworthy evidence either way (Task 3 ruling, verbatim: "not reject, not
    unscored"). The new check requires POSITIVE evidence: a `hole_groups` entry whose
    diameter matches the spec's stated 3.2mm through-hole diameter, measured blind.
    V064's REAL facts (tests/fixtures/harvest_agreement_fixtures.json) carry per-hole
    diameter grouping in general, but the one group present is d=9.0 (the boss body) --
    there is no hole_groups entry anywhere near 3.2mm, so there is nothing to check
    positive evidence against: this check now gives NO verdict on this fixture. V064
    still ends up rejected overall -- via strict_envelope_check alone (its stated
    180x130x55mm envelope measures 180x130x53mm), tested separately above."""
    fx = _load_fixtures()
    v064 = fx["v064_wrong_sheared_lip"]
    assert v064["facts"]["hole_groups"] == [{"d": 9.0, "n": 4, "through": 0, "circle_d": 199.7}]
    assert harvest.strict_through_holes_check(v064["spec"], v064["facts"]) is None


def test_strict_through_holes_check_rejects_on_positive_blind_evidence():
    """Fix round 3, M1: the check DOES fire when a hole_groups entry's diameter matches
    a spec-stated through-hole diameter (within 0.1mm) and that group measured at least
    one hole of that diameter blind."""
    spec = "a bracket with a 5mm through hole"
    facts = {"hole_groups": [{"d": 5.02, "n": 1, "through": 0, "circle_d": 0.0}]}
    reason = harvest.strict_through_holes_check(spec, facts)
    assert reason is not None
    assert "strict_through_holes" in reason
    assert "5mm" in reason   # the SPEC's stated diameter, not the measured group's 5.02


def test_strict_through_holes_check_no_false_reject_on_a_correct_orientation():
    """Fix round 3, M1's own false-rejection test: a part whose through hole is real and
    correct, but whose orientation the aggregate through/blind classifier does not
    recognise (through_holes: 0, blind_holes: 0, bore_axes axial -- exactly how
    scripts/inspect really reports a correctly-drilled axial hole it could not resolve
    to a through/blind verdict) must PASS, not be rejected, because there is no
    hole_groups entry (positive evidence) naming that diameter as blind."""
    spec = "a shaft with a 10mm through hole"
    facts = {"through_holes": 0, "blind_holes": 0, "bore_axes": [(10.0, "axial")],
            "hole_groups": []}
    assert harvest.strict_through_holes_check(spec, facts) is None


def test_strict_through_holes_check_none_when_spec_states_no_count():
    fx = _load_fixtures()
    v066 = fx["v066_t02_wrong_slot"]
    assert harvest.strict_through_holes_check(v066["spec"], v066["facts"]) is None


def test_strict_through_holes_check_passes_when_count_is_met():
    spec = "a plate with four 5mm through holes"
    clean = {"hole_groups": [{"d": 5.0, "n": 4, "through": 4, "circle_d": 0.0}]}
    assert harvest.strict_through_holes_check(spec, clean) is None


def test_strict_through_holes_check_none_without_a_stated_diameter():
    """A count with no accompanying diameter has nothing to positively match against a
    hole_groups entry (fix round 3, M1) -- no verdict, not a reject or an unscored."""
    spec = "a bracket with 6 through-holes"
    assert harvest._spec_through_hole_diameter_mm(spec) is None
    assert harvest.strict_through_holes_check(
        spec, {"hole_groups": [{"d": 5.0, "n": 6, "through": 0, "circle_d": 0.0}]}) is None


@pytest.mark.parametrize("phrasing,expected", [
    ("a plate with four 5mm through holes", 4),
    ("a bracket with 6 through-holes", 6),
    ("a bracket with a 5mm through hole", 1),
    ("a plate with twelve through holes", 12),
])
def test_spec_through_hole_count_word_and_digit_forms(phrasing, expected):
    assert harvest._spec_through_hole_count(phrasing) == expected


def test_strict_through_holes_check_no_verdict_when_hole_groups_missing_entirely():
    """Fix round 3, M1: there is no "unscored" branch any more -- a missing/empty
    hole_groups field (no per-hole detail at all) gives no verdict, same as one present
    but with no matching diameter."""
    spec = "a bracket with four 5mm through holes"
    assert harvest.strict_through_holes_check(spec, {}) is None
    assert harvest.strict_through_holes_check(spec, {"hole_groups": []}) is None


def test_regate_and_strict_pushes_v064_to_silver_via_gate_spec():
    """End to end through the actual wiring: _regate_and_strict augments gate_spec with
    the strict_envelope finding, which _classify then reads as "silver", never "good" --
    this is the exact mechanism that rejects V064 in production. Fix round 3, M1:
    strict_through_holes_check no longer ALSO fires on this real fixture (see
    test_strict_through_holes_check_no_verdict_on_v064_real_facts) -- envelope alone
    carries the rejection now, which this test asserts explicitly."""
    fx = _load_fixtures()
    v064 = fx["v064_wrong_sheared_lip"]
    monkeypatch_facts = v064["facts"]

    def fake_regate(m_fluid, build_dir, spec):
        return {"error": None, "facts": monkeypatch_facts, "gate_hard": [],
               "gate_spec": [], "gate_adv": [], "unscored_reason": None}

    import unittest.mock
    with unittest.mock.patch.object(harvest, "_regate", fake_regate):
        m = harvest._regate_and_strict({"error": None}, Path("/tmp"), v064["spec"],
                                       _default_cfg())
    assert any("strict_envelope" in n for n in m["gate_spec"])
    assert not any("strict_through_holes" in n for n in m["gate_spec"])
    verdict, _ = harvest._classify(m, None, Path("/tmp"))
    assert verdict == "silver"


def test_regate_and_strict_leaves_error_and_unscored_rows_untouched():
    def fake_regate_error(m_fluid, build_dir, spec):
        return {"error": "boom", "facts": {}, "gate_hard": [], "gate_spec": [],
               "gate_adv": [], "unscored_reason": None}

    import unittest.mock
    with unittest.mock.patch.object(harvest, "_regate", fake_regate_error):
        m = harvest._regate_and_strict({"error": "boom"}, Path("/tmp"), "spec", _default_cfg())
    assert m["error"] == "boom"
    assert m["gate_spec"] == []


# ---------------------------------------------------------------------------
# Attempt-cap exhaustion (fix round 2, Section C)
# ---------------------------------------------------------------------------

def test_eligible_pools_drops_a_teacher_exhausted_spec_from_both_pools():
    bank = [_spec_row("a")]
    progress = {"a": {"student_attempts": 2, "teacher_attempts": 5, "pairs": 0}}
    cfg = _default_cfg(max_pairs_per_spec=2, attempt_caps={"student": 2, "teacher": 5})
    student, teacher = harvest._eligible_pools(bank, progress, cfg)
    assert student == [] and teacher == []


def test_eligible_pools_keeps_a_spec_under_its_teacher_cap_in_the_teacher_pool():
    bank = [_spec_row("a")]
    progress = {"a": {"student_attempts": 2, "teacher_attempts": 4, "pairs": 0}}
    cfg = _default_cfg(max_pairs_per_spec=2, attempt_caps={"student": 2, "teacher": 5})
    student, teacher = harvest._eligible_pools(bank, progress, cfg)
    assert [r["id"] for r in teacher] == ["a"]


def test_exhausted_specs_lists_only_specs_over_the_teacher_cap_and_short_of_pairs():
    bank = [_spec_row("a"), _spec_row("b"), _spec_row("c")]
    progress = {
        "a": {"student_attempts": 2, "teacher_attempts": 5, "pairs": 0},   # exhausted
        "b": {"student_attempts": 2, "teacher_attempts": 5, "pairs": 2},   # done, not exhausted
        "c": {"student_attempts": 2, "teacher_attempts": 1, "pairs": 0},   # still eligible
    }
    cfg = _default_cfg(max_pairs_per_spec=2, attempt_caps={"student": 2, "teacher": 5})
    exhausted = harvest._exhausted_specs(bank, progress, cfg)
    assert [r["id"] for r in exhausted] == ["a"]


# ---------------------------------------------------------------------------
# sample_spec: max pairs per spec, dedup by AST fingerprint, prefer lower temperature
# ---------------------------------------------------------------------------

def test_max_pairs_per_spec_stops_once_enough_good_candidates_are_found(monkeypatch, tmp_path):
    """Temps are cycled ascending, so once max_pairs_per_spec distinct good candidates
    exist, sampling stops early -- a later (higher-temperature) candidate could only ever
    lose the "prefer lower temperature" tie-break, never win it, so there is nothing left
    to gain from spending more GPU time on this spec. Every candidate that WAS sampled is
    still ledgered."""
    _patch_generate_sequence(monkeypatch, ["code-A", "code-B", "code-C"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 2   # candidate index 2 (code-C) never attempted: quota already met
    assert all(r["ok"] for r in ledger)
    assert all(r["model"] == "local:test-arm" for r in ledger)

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 2
    kept_codes = {p["code"] for p in pairs}
    assert kept_codes == {"code-A", "code-B"}   # temp 0.2 and 0.5 kept
    assert progress["s1"]["pairs"] == 2
    # Fix round 3, LOW3: attempt caps count ROUNDS (one sample_spec call), not
    # candidates -- this was a single round, however many of its candidates ran.
    assert progress["s1"]["student_attempts"] == 1


def test_candidates_exceed_slots_when_duplicates_force_extra_sampling(monkeypatch, tmp_path):
    """When a duplicate eats a slot's worth of sampling, the loop keeps going up to the
    full `candidates` budget looking for a distinct replacement -- the early-stop above
    only fires once enough DISTINCT good codes exist."""
    _patch_generate_sequence(monkeypatch, ["code-A", "code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 3   # all three candidates attempted: the repeat didn't fill a slot
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert sorted(p["code"] for p in pairs) == ["code-A", "code-B"]
    assert progress["s1"]["pairs"] == 2


def test_dedup_by_ast_fingerprint_skips_repeated_candidate(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A", "code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    # code-A (temp 0.2) kept, the repeated code-A (temp 0.5) skipped as a duplicate,
    # code-B (temp 0.8) kept to fill the second slot.
    assert sorted(p["code"] for p in pairs) == ["code-A", "code-B"]
    assert progress["s1"]["pairs"] == 2


def test_dedup_across_units_via_pair_index_seeded_from_disk(monkeypatch, tmp_path):
    """Task 3 fix M5: a PairIndex built from an existing pairs.jsonl (a previous unit's
    output) must dedup against it too, not just within one sample_spec call. Fix round 2:
    the seeded pair also carries a "signature" matching _clean_m()'s default facts, so
    code-B (a genuinely new candidate) is confirmed by AGREEING with that already-
    confirmed pair (Section A: an existing pair is trusted evidence, not only a sibling
    sampled in the same round) -- without a signature to agree with, a lone new gate-
    clean candidate would stay "unconfirmed" and never become a pair at all."""
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "spec_id": "s1", "tier": 2, "kind": "good", "code": "code-A",
        "signature": harvest.signature(_clean_m()["facts"])})
    _patch_generate_sequence(monkeypatch, ["code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    pair_index = harvest.PairIndex()
    assert pair_index.good_total == 1
    harvest.sample_spec(row, cfg, "student", "unit1", progress, pair_index)

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    # code-A already existed on disk: only code-B should be newly appended.
    new_codes = [p["code"] for p in pairs if p.get("unit_id") == "unit1"]
    assert new_codes == ["code-B"]


# ---------------------------------------------------------------------------
# gate_version + promotion re-check (fix round 3, M3)
# ---------------------------------------------------------------------------

def test_gate_version_changes_with_effective_agreement_or_strict_config():
    cfg1 = _default_cfg()
    cfg2 = _default_cfg(agreement={"volume_tol_pct": 0.1, "bbox_tol_mm": 0.05,
                                   "bore_round_mm": 0.01})
    assert harvest.gate_version(cfg1) != harvest.gate_version(cfg2)
    assert harvest.gate_version(cfg1) == harvest.gate_version(_default_cfg())


def test_candidate_index_excludes_a_stale_gate_version_row():
    """Fix round 3, M3: a candidate written under a DIFFERENT gate_version than the one
    in effect now is excluded outright -- it can neither confirm anything nor be
    promoted (see the module docstring's own note on why this happens at CandidateIndex
    level, not only at the final promotion step)."""
    harvest._append_jsonl(harvest.CANDIDATES_FILE, {
        "id": "c:s1:x", "spec_id": "s1", "status": "unconfirmed",
        "fingerprint": "fp1", "gate_version": "gv1-deadbeef"})
    idx = harvest.CandidateIndex(harvest.PairIndex(), current_gate_version="gv1-somethingelse")
    assert idx.unconfirmed_for("s1") == []
    # Matching gate_version: not excluded.
    idx2 = harvest.CandidateIndex(harvest.PairIndex(), current_gate_version="gv1-deadbeef")
    assert len(idx2.unconfirmed_for("s1")) == 1


def test_promotion_recheck_rejects_a_carried_candidate_that_fails_strict_today(
        monkeypatch, tmp_path):
    """Fix round 3, M3, the reviewer's own scenario: a carried candidate whose STORED
    facts show a stated-diameter through hole measuring blind (the M1 positive-evidence
    check) must not be allowed to confirm -- or be promoted alongside -- a new clean
    sample that agrees with it on geometric signature. `hole_groups` is not part of
    `signature()` at all, so the two rows' signatures agree even though only the
    carried one's facts fail today's strict check."""
    spec_text = "a bracket with a 5mm through hole"
    row = _spec_row("s1", spec=spec_text)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    gv = harvest.gate_version(cfg)
    bad_facts = dict(_DEFAULT_CLEAN_FACTS,
                     hole_groups=[{"d": 5.0, "n": 1, "through": 0, "circle_d": 0.0}])
    sig = harvest.signature(bad_facts)
    harvest._append_jsonl(harvest.CANDIDATES_FILE, {
        "id": "c:s1:carried1", "spec_id": "s1", "spec": spec_text, "tier": 2,
        "group": "plate", "source": "student", "kind": "candidate", "turn": "first",
        "code": "carried-code", "facts": bad_facts, "signature": sig,
        "fingerprint": "carriedfp", "temperature": 0.2, "status": "unconfirmed",
        "gate_version": gv, "confirm_strength": None})

    _patch_generate_sequence(monkeypatch, ["new-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    good_facts = dict(_DEFAULT_CLEAN_FACTS,
                      hole_groups=[{"d": 5.0, "n": 1, "through": 1, "circle_d": 0.0}])
    _patch_regate_passthrough(monkeypatch, facts=good_facts)
    # Same geometric signature as the carried row (hole_groups is not part of it).
    assert harvest.signature(good_facts) == sig

    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    by_id = {}
    for r in harvest._read_jsonl(harvest.CANDIDATES_FILE):
        by_id[r["id"]] = r
    assert by_id["c:s1:carried1"]["status"] == "rejected_at_promotion"
    assert "strict_through_holes" in by_id["c:s1:carried1"]["rejected_reason"]


def test_ast_fingerprint_collapses_comment_only_differences():
    a = "from build123d import *\n# a comment\nresult = Box(1, 1, 1)\n"
    b = "from build123d import *\n# a DIFFERENT comment\nresult = Box(1, 1, 1)\n"
    assert harvest._code_fingerprint(a) == harvest._code_fingerprint(b)


def test_ast_fingerprint_collapses_a_pure_variable_rename():
    """Fix round 3, H2a: the old Task 3 fix L7 accepted that a renamed variable would
    NOT collapse ("catching that needs a heavier alpha-renaming normalisation this fix
    does not attempt"). This fix round IS that heavier normalisation: two programs
    identical except for which name they gave the same locally-bound value now collapse
    to the same fingerprint (both alpha-rename to `_v0 = Box(1, 1, 1); _v1 = _v0`)."""
    a = "from build123d import *\nslot = Box(1, 1, 1)\nresult = slot\n"
    b = "from build123d import *\nslot_cutter = Box(1, 1, 1)\nresult = slot_cutter\n"
    assert harvest._code_fingerprint(a) == harvest._code_fingerprint(b)


def test_ast_fingerprint_does_not_collapse_an_extra_intermediate_statement():
    """Fix round 3, H2a: a genuinely different PROGRAM STRUCTURE -- not merely a rename
    -- must still NOT collapse. "result = Box(20, 20, 20)" is one statement;
    "cube = Box(20, 20, 20); result = cube" is two (an extra assignment indirection).
    Alpha-renaming both (result->_v0 in the first; cube->_v0, result->_v1 in the second)
    still leaves the second with one more statement than the first, so they remain
    genuinely different code -- this is also the exact pair the real-subprocess
    end-to-end test below (test_agreement_end_to_end_real_subprocess) samples as its
    "alone" spec's two distinct-but-agreeing-on-geometry codes, and this test confirms
    directly, at the fingerprint level, why that test still expects 2 DISTINCT pairs
    rather than one being silently deduplicated as "the same code" under this fix."""
    a = "from build123d import *\nresult = Box(20, 20, 20)\n"
    b = "from build123d import *\ncube = Box(20, 20, 20)\nresult = cube\n"
    assert harvest._code_fingerprint(a) != harvest._code_fingerprint(b)


def test_ast_fingerprint_collapses_numeric_literal_int_vs_float():
    """Fix round 3, H2a: `5` and `5.0` normalise to the same Constant value."""
    a = "from build123d import *\nresult = Box(5, 5, 5)\n"
    b = "from build123d import *\nresult = Box(5.0, 5.0, 5.0)\n"
    assert harvest._code_fingerprint(a) == harvest._code_fingerprint(b)


def test_ast_fingerprint_never_renames_builtins_attributes_or_kwargs():
    """Fix round 3, H2a: a builtin name used as a Load-context call (`sum`, `len`, ...),
    an attribute access (`x.faces`), and a keyword-argument name (`radius=`) must never
    be folded into the identifier-renaming map -- only genuinely NEW locally-bound names
    are renamed. Swapping the keyword-argument NAME (radius vs height, an actual
    semantic difference, not a rename of a bound value) must still change the
    fingerprint; swapping which bound local holds the same call result must not."""
    a = "from build123d import *\nx = Cylinder(radius=5, height=len([1, 2]))\nresult = x.faces\n"
    b = "from build123d import *\ny = Cylinder(radius=5, height=len([1, 2]))\nresult = y.faces\n"
    assert harvest._code_fingerprint(a) == harvest._code_fingerprint(b)
    c = "from build123d import *\nx = Cylinder(height=5, radius=len([1, 2]))\nresult = x.faces\n"
    assert harvest._code_fingerprint(a) != harvest._code_fingerprint(c)


# ---------------------------------------------------------------------------
# code_similarity + confirm_strength (fix round 3, H2b)
# ---------------------------------------------------------------------------

def test_code_similarity_pure_function():
    a = "from build123d import *\nresult = Box(10, 10, 10)\n"
    b = "from build123d import *\nbox_ = Box(10, 10, 10)\nresult = box_\n"
    sim = harvest.code_similarity(a, [b])
    assert 0.0 < sim < 1.0
    assert sim == round(sim, 3)
    assert harvest.code_similarity(a, []) == 0.0   # nothing to compare against
    assert harvest.code_similarity(a, [a]) == 1.0   # identical to itself


def test_confirm_strength_for_cluster_labels():
    """Fix round 3, H2b: "cross_pass" requires the agreeing set to span BOTH a
    student-pass and a think-pass member; any set drawn from only one pass, however
    many members, is "same_pass". `_item_pass_label` reads a carried/existing item's
    own historical `source`, and a "new" item's pass from the CURRENT round's mode."""
    new_item = {"origin": "new"}
    carried_student = {"origin": "carried", "cand_row": {"source": "student"}}
    carried_think = {"origin": "carried", "cand_row": {"source": "teacher:think"}}
    existing_student = {"origin": "existing", "source": "student"}
    existing_think = {"origin": "existing", "source": "teacher:think"}

    assert harvest._confirm_strength_for_cluster([new_item, new_item], "student") == "same_pass"
    assert harvest._confirm_strength_for_cluster(
        [new_item, carried_think], "student") == "cross_pass"
    assert harvest._confirm_strength_for_cluster(
        [existing_student, new_item], "student") == "same_pass"
    assert harvest._confirm_strength_for_cluster(
        [existing_think, new_item], "student") == "cross_pass"
    assert harvest._confirm_strength_for_cluster(
        [carried_student, carried_student], "think") == "same_pass"


def test_confirm_strength_same_pass_when_both_new_candidates_share_a_pass(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 2
    assert all(p["confirm_strength"] == "same_pass" for p in pairs)
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    confirmed = [r for r in ledger if r["agreement"] == "agreement"]
    assert confirmed and all(r["confirm_strength"] == "same_pass" for r in confirmed)


def test_confirm_strength_cross_pass_when_a_new_candidate_agrees_with_a_think_pass_anchor(
        monkeypatch, tmp_path):
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "id": "p:s1:anchor1", "spec_id": "s1", "tier": 2, "kind": "good",
        "code": "anchor-code", "signature": sig, "source": "teacher:think",
        "confirm_strength": "reference"})
    _patch_generate_sequence(monkeypatch, ["new-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    new_pair = [p for p in pairs if p["code"] == "new-code"][0]
    assert new_pair["confirm_strength"] == "cross_pass"
    # The anchor was already "reference" (owner ground truth) -- never "upgraded" just
    # because a same-geometry sample landed under a different pass.
    assert harvest._read_jsonl(harvest.UPGRADES_FILE) == []


def test_confirm_strength_upgrade_appended_without_rewriting_pairs_jsonl(monkeypatch, tmp_path):
    """Fix round 3, H2b part 2: a later think-pass sample agreeing with an anchor that
    was originally confirmed same_pass appends an UPGRADES_FILE row for that anchor's
    pair_id -- pairs.jsonl itself (append-only) is never rewritten; a FRESH PairIndex
    picks up the upgrade via anchors_for."""
    sig = harvest.signature(_DEFAULT_CLEAN_FACTS)
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "id": "p:s1:anchor1", "spec_id": "s1", "tier": 2, "kind": "good",
        "code": "anchor-code", "signature": sig, "source": "student",
        "confirm_strength": "same_pass"})
    _patch_generate_sequence(monkeypatch, ["new-code"], mode="think")
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "think", "unit1", progress, harvest.PairIndex())

    upgrades = harvest._read_jsonl(harvest.UPGRADES_FILE)
    assert len(upgrades) == 1
    assert upgrades[0]["pair_id"] == "p:s1:anchor1"
    assert upgrades[0]["confirm_strength"] == "cross_pass"
    anchor_row = [p for p in harvest._read_jsonl(harvest.PAIRS_FILE)
                 if p["id"] == "p:s1:anchor1"][0]
    assert anchor_row["confirm_strength"] == "same_pass"   # pairs.jsonl untouched
    fresh_index = harvest.PairIndex()
    anchors = fresh_index.anchors_for("s1")
    assert any(a["id"] == "p:s1:anchor1" and a["confirm_strength"] == "cross_pass"
              for a in anchors)


def test_reference_match_pair_has_zero_code_similarity_and_reference_confirm_strength(
        monkeypatch, tmp_path):
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert pairs[0]["confirm_strength"] == "reference"
    assert pairs[0]["code_similarity"] == 0.0
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert ledger[0]["confirm_strength"] == "reference"


def test_pairs_stats_reports_good_by_confirm_strength(monkeypatch, tmp_path):
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "spec_id": "s1", "tier": 2, "kind": "good", "code": "a",
        "id": "p:s1:a", "confirm_strength": "same_pass"})
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "spec_id": "s2", "tier": 2, "kind": "good", "code": "b",
        "id": "p:s2:b", "confirm_strength": "reference"})
    harvest._append_jsonl(harvest.UPGRADES_FILE, {
        "pair_id": "p:s1:a", "confirm_strength": "cross_pass", "ts": "2026-09-19T00:00:00Z"})
    stats = harvest._pairs_stats()
    assert stats["good_by_confirm_strength"] == {"cross_pass": 1, "reference": 1}


def test_ast_fingerprint_falls_back_to_raw_hash_on_unparseable_code():
    import hashlib
    bad = "this is not ( python"
    assert harvest._code_fingerprint(bad) == hashlib.sha1(bad.encode()).hexdigest()


def test_spec_already_at_pair_cap_is_never_sampled(monkeypatch, tmp_path):
    calls = _patch_generate_sequence(monkeypatch, ["should-not-be-called"])
    row = _spec_row("s1")
    cfg = _default_cfg(max_pairs_per_spec=2)
    progress = {"s1": {"student_attempts": 5, "teacher_attempts": 0, "pairs": 2}}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())
    assert calls["n"] == 0
    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []


def test_generation_exception_writes_a_failed_ledger_row_and_continues(monkeypatch, tmp_path):
    def fake_generate(spec, notes, temperature):
        raise ValueError("the model returned unparseable output")

    monkeypatch.setattr(harvest, "_student_generate", fake_generate)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 2
    assert all(not r["ok"] for r in ledger)
    assert all("codegen failed" in r["error"] for r in ledger)
    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    # Fix round 3, LOW3: one round (one sample_spec call) is one attempt, whatever the
    # candidate width -- both candidates here failed inside the SAME round.
    assert progress["s1"]["student_attempts"] == 1


def test_crash_salvage_produces_two_ledger_rows_and_a_pair_on_recovery(monkeypatch, tmp_path):
    """Fix round 2: the salvage's own single candidate needs a reference match to
    become a pair on its own (Section A path 1) -- otherwise a lone salvaged candidate
    would only ever reach "unconfirmed", which is a separate concern from what this test
    actually exercises (the salvage mechanics themselves)."""
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["broken-code"])
    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "the script failed to run: boom"}
        return {"error": None}   # the salvage attempt succeeds

    monkeypatch.setattr(fluid_gen, "_materialize", fake_materialize)
    _patch_regate_passthrough(monkeypatch)
    monkeypatch.setattr(harvest, "diagnose", lambda err: ("generic", "try wrapping it"))
    monkeypatch.setattr(fluid_gen, "_revise_on_repair_rung",
                        lambda spec, code, problem: ("fixed-code", None))

    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 2
    assert ledger[0]["candidate"] == "0"
    assert ledger[0]["ok"] is False
    assert ledger[1]["candidate"] == "0-salvage"
    assert ledger[1]["ok"] is True

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 1
    assert pairs[0]["code"] == "fixed-code"
    assert pairs[0]["kind"] == "good"
    assert pairs[0]["turn"] == "salvage"   # Task 3 fix M8


def test_salvage_candidates_never_confirm_each_other(monkeypatch, tmp_path):
    """Fix round 3, H1: a candidate with turn=="salvage" never enters the agreement
    pool. Both candidates here crash on their first turn and recover identically via
    salvage (same facts, no reference); before this fix the two salvages would
    geometrically agree with each other and become 2 confirmed good pairs -- exactly the
    loophole a reviewer found (a repair call is more likely than a first try to produce
    a plausible-but-wrong shape, so two of them agreeing is weak evidence at best)."""
    _patch_generate_sequence(monkeypatch, ["broken-A", "broken-B"])
    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec):
        calls["n"] += 1
        if calls["n"] % 2 == 1:   # every candidate's FIRST turn crashes
            return {"error": "the script failed to run: boom"}
        return {"error": None}   # every candidate's salvage turn recovers

    monkeypatch.setattr(fluid_gen, "_materialize", fake_materialize)
    _patch_regate_passthrough(monkeypatch)
    monkeypatch.setattr(harvest, "diagnose", lambda err: ("generic", "try wrapping it"))
    monkeypatch.setattr(fluid_gen, "_revise_on_repair_rung",
                        lambda spec, code, problem: (f"fixed-{code}", None))

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    assert harvest._read_jsonl(harvest.CANDIDATES_FILE) == []   # never entered the pool
    review = harvest._read_jsonl(harvest.REVIEW_FILE)
    assert len(review) == 2
    assert all(r["turn"] == "salvage" for r in review)


def test_salvage_must_not_confirm_a_first_turn_candidate(monkeypatch, tmp_path):
    """Fix round 3, H1: a lone first-turn candidate with nobody else to agree with stays
    "unconfirmed" even when a LATER candidate's salvage attempt lands on the identical
    geometric signature -- a salvage is never allowed to BE the confirming partner."""
    _patch_generate_sequence(monkeypatch, ["good-code", "broken-code"])

    def fake_materialize(code, build_dir, spec):
        if code == "broken-code":
            return {"error": "the script failed to run: boom"}
        return {"error": None}   # "good-code" (first turn) and "fixed-code" (salvage)

    monkeypatch.setattr(fluid_gen, "_materialize", fake_materialize)
    _patch_regate_passthrough(monkeypatch)
    monkeypatch.setattr(harvest, "diagnose", lambda err: ("generic", "try wrapping it"))
    monkeypatch.setattr(fluid_gen, "_revise_on_repair_rung",
                        lambda spec, code, problem: ("fixed-code", None))

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    candidates = harvest._read_jsonl(harvest.CANDIDATES_FILE)
    assert len(candidates) == 1
    assert candidates[0]["code"] == "good-code"
    assert candidates[0]["agreement"] == "unconfirmed"
    review = harvest._read_jsonl(harvest.REVIEW_FILE)
    assert len(review) == 1
    assert review[0]["turn"] == "salvage"
    assert review[0]["code"] == "fixed-code"


def test_owner_reference_near_miss_writes_a_fail_pair_with_null_code(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["wrong-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {
        "band": "near_miss", "reference": "ref.stl", "chamfer_mm": 3.4, "volume_diff_pct": 8.0})

    row = _spec_row("s1", reference_stl="/tmp/does-not-need-to-exist.stl")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["kind"] == "fail"
    assert p["code"] is None
    assert p["bad_code"] == "wrong-code"
    assert "near-miss" in p["problem"]
    assert p["turn"] == "first"
    assert progress["s1"]["pairs"] == 0   # a fail pair does not count against the good quota


def test_silver_writes_a_review_row_not_a_pair(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["mostly-right-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    monkeypatch.setattr(harvest, "_execute_with_salvage", lambda spec, code, build_dir,
                        prompt_info, gen_seconds, cfg, deadline=None: [{
                            "code": code, "prompt_info": prompt_info, "turn": "first",
                            "seconds": gen_seconds,
                            "m": _clean_m(gate_spec=["[spec] length is 90mm, spec said 80mm"])}])

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    review = harvest._read_jsonl(harvest.REVIEW_FILE)
    assert len(review) == 1
    assert review[0]["code"] == "mostly-right-code"
    assert any("80mm" in n for n in review[0]["notes"])
    assert review[0]["turn"] == "first"


def test_unscored_candidate_writes_a_ledger_row_never_a_pair_or_review(monkeypatch, tmp_path):
    """Task 3 fix H2: unscored is a third bucket -- not silver, not discarded silently,
    just never a pair/review row."""
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    monkeypatch.setattr(harvest, "_regate", lambda m_fluid, build_dir, spec: {
        "error": None, "facts": {}, "gate_hard": [], "gate_spec": [], "gate_adv": [],
        "unscored_reason": "inspect reported the STEP invalid"})

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    assert harvest._read_jsonl(harvest.REVIEW_FILE) == []
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 1
    assert ledger[0]["unscored_reason"] == "inspect reported the STEP invalid"
    assert ledger[0]["build_dir"] is None   # unscored builds are not persisted either


# ---------------------------------------------------------------------------
# Aborts and infra errors mid-candidate (Task 3 fix H3/L3)
# ---------------------------------------------------------------------------

def test_abort_flag_undoes_the_attempt_and_writes_no_ledger_row(monkeypatch, tmp_path):
    """H3 (repro3_abort_swallowed.py): a SIGTERM landing during execution is swallowed by
    fluid_gen._materialize's own bare except, but the module-level abort flag survives
    that. sample_spec must raise SpecgenAborted, write no ledger row for the interrupted
    candidate, and undo its attempt-counter increment."""
    _patch_generate_sequence(monkeypatch, ["code-A", "code-B", "code-C"])

    def run_step_aborted(code, build_dir):
        aw._ABORT_REQUESTED = True
        raise aw.SpecgenAborted("terminated by signal 15 (SIGTERM)")

    monkeypatch.setattr(engine, "run_step", run_step_aborted)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    with pytest.raises(aw.SpecgenAborted):
        harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())

    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []
    assert progress["s1"]["student_attempts"] == 0
    assert not harvest.PROGRESS_FILE.exists()


def test_abort_between_candidates_stops_before_the_next_one(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_generate(spec, notes, temperature):
        calls["n"] += 1
        if calls["n"] == 2:
            aw._ABORT_REQUESTED = True
        return f"code-{calls['n']}", {"model": "local:x", "system": "SYS", "prompt": "P",
                                      "usage": {}}

    monkeypatch.setattr(harvest, "_student_generate", fake_generate)
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=5, temps=[0.2], max_pairs_per_spec=5)
    progress: dict = {}
    with pytest.raises(aw.SpecgenAborted):
        harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())
    # Candidate 1 completed and was ledgered/paired before the flag was set (set as a
    # side effect of generating candidate 2); candidate 2 itself is abandoned entirely.
    assert calls["n"] == 2
    assert len(harvest._read_jsonl(harvest.LEDGER_FILE)) == 1


def test_infra_error_aborts_the_unit_and_does_not_count_as_an_attempt(monkeypatch, tmp_path):
    """Task 3 fix L3: a dead/unreachable model server must abort the whole unit rather
    than being recorded as a per-candidate student failure."""
    import urllib.error

    def fake_generate(spec, notes, temperature):
        raise urllib.error.URLError(ConnectionRefusedError("connection refused"))

    monkeypatch.setattr(harvest, "_student_generate", fake_generate)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    with pytest.raises(harvest._InfraError):
        harvest.sample_spec(row, cfg, "student", "unit1", progress, harvest.PairIndex())
    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []
    assert progress["s1"]["student_attempts"] == 0


def test_is_infra_error_recognises_connection_and_timeout_failures():
    import socket
    import urllib.error
    assert harvest._is_infra_error(urllib.error.URLError("x")) is True
    assert harvest._is_infra_error(ConnectionRefusedError()) is True
    assert harvest._is_infra_error(socket.timeout()) is True
    assert harvest._is_infra_error(TimeoutError()) is True
    assert harvest._is_infra_error(ValueError("not an infra problem")) is False


def test_run_in_window_reports_infra_error(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "arm_window",
                        lambda arm: _NullCtx())

    def boom():
        raise harvest._InfraError("model server unreachable: connection refused")

    rc = harvest._run_in_window(None, boom)
    assert rc == 1


class _NullCtx:
    def __enter__(self):
        return "test-arm"

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Contamination re-check (Task 3 fix L1)
# ---------------------------------------------------------------------------

def test_contaminated_spec_is_never_sampled(monkeypatch, tmp_path):
    calls = _patch_generate_sequence(monkeypatch, ["should-not-be-called"])
    monkeypatch.setattr(harvest, "_is_contaminated", lambda spec: True)
    row = _spec_row("s1")
    cfg = _default_cfg()
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex())
    assert calls["n"] == 0
    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []


def test_is_contaminated_matches_a_real_card_suite_spec():
    """Uses the real, unmocked default_contamination_sets() against a spec pulled
    straight from a card suite file, so this test would fail if the wiring to
    lab.data/harvest_census ever broke."""
    import json as _json
    specs_path = HERE / "benchmarks" / "text-to-cad" / "specs.json"
    if not specs_path.exists():
        pytest.skip("benchmarks/text-to-cad/specs.json not present in this checkout")
    raw = _json.loads(specs_path.read_text(encoding="utf-8"))
    items = raw["benchmarks"] if isinstance(raw, dict) else raw
    assert harvest._is_contaminated(items[0]["spec"]) is True


def test_default_contamination_sets_survives_a_c_locale_real_subprocess():
    """Regression test for a real bug found via a real --unit smoke (fix round 1):
    benchmarks/text-to-cad/specs.json's own "source" field carries a real em dash, and
    scripts/harvest_census.py's _suite_specs_by_suite() read it via Path.read_text()
    with no explicit encoding, which raised UnicodeDecodeError under the C/POSIX locale
    lab/gpu_window.sh's launch environment uses. lab/harvest.py's own fail-closed L1
    check then caught that exception and refused EVERY spec it examined -- the entire
    bank looked "contaminated". Fixed with encoding="utf-8" in harvest_census.py (not a
    frozen file). Driven as a real subprocess with LC_ALL=C/LANG=C so this actually
    exercises Python's locale-dependent default encoding, which a UTF-8 dev shell would
    never trigger (this exact test, without the forced C locale, would not have caught
    the original bug)."""
    specs_path = HERE / "benchmarks" / "text-to-cad" / "specs.json"
    if not specs_path.exists():
        pytest.skip("benchmarks/text-to-cad/specs.json not present in this checkout")
    code = (
        f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        f"sys.path.insert(0, {str(HERE / 'scripts')!r})\n"
        "from lab.data import default_contamination_sets\n"
        "keys, slugs = default_contamination_sets()\n"
        "print(len(keys), len(slugs))\n"
    )
    env = dict(os.environ)
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"})
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env,
                          timeout=15)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    n_keys, n_slugs = map(int, proc.stdout.decode().split())
    assert n_keys > 0
    assert n_slugs > 0


def test_is_contaminated_fails_closed_when_sets_cannot_be_computed(monkeypatch):
    def boom():
        raise RuntimeError("suite files missing")
    monkeypatch.setattr(harvest, "default_contamination_sets", boom)
    assert harvest._is_contaminated("anything at all") is True


def test_is_contaminated_computation_failure_prints_a_distinct_message(monkeypatch, capsys):
    """A computation failure (e.g. the real UnicodeDecodeError this fix round found)
    must be visibly distinguishable in the logs from an actual key/slug match -- both
    return True, but conflating them under one message cost real debugging time."""
    def boom():
        raise RuntimeError("suite files missing")
    monkeypatch.setattr(harvest, "default_contamination_sets", boom)
    harvest._is_contaminated("anything at all")
    err = capsys.readouterr().err
    assert "computation failure" in err
    assert "suite files missing" in err


# ---------------------------------------------------------------------------
# Model-string capture + code-model pin refusal (Task 3 fix M3)
# ---------------------------------------------------------------------------

def test_ledger_and_pair_rows_carry_the_recorded_model_string(monkeypatch, tmp_path):
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex())
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert ledger[0]["model"] == "local:test-arm"
    assert pairs[0]["model"] == "local:test-arm"


def test_check_code_model_pin_refuses_on_a_stale_pin(monkeypatch):
    monkeypatch.setattr(engine, "_load_config",
                        lambda: {"cad": {"code_model": "local:some-other-arm"}})
    reason = harvest._check_code_model_pin()
    assert reason is not None
    assert "some-other-arm" in reason


def test_check_code_model_pin_clears_and_forces_active_model(monkeypatch):
    monkeypatch.setattr(engine, "_load_config", lambda: {"cad": {}})
    monkeypatch.setattr(engine, "_ACTIVE_CODE_MODEL", None)
    reason = harvest._check_code_model_pin()
    assert reason is None
    assert engine._ACTIVE_CODE_MODEL == harvest.CODE_MODEL_STRONG


def test_check_code_model_pin_allows_a_pin_that_matches_the_maker_arm(monkeypatch):
    monkeypatch.setattr(engine, "_load_config",
                        lambda: {"cad": {"code_model": harvest.CODE_MODEL_STRONG}})
    assert harvest._check_code_model_pin() is None


# ---------------------------------------------------------------------------
# Student vs think prompt parity (Task 3 fix M4)
# ---------------------------------------------------------------------------

def test_student_and_think_passes_send_byte_identical_prompts(monkeypatch):
    """M4: the think pass now reuses generate_code_raw's own prompt-construction code
    (via _ThinkInjector) instead of a hand-built duplicate string -- the two passes must
    send byte-identical system/user text and model, differing ONLY in the think kwarg."""
    captured = []

    def spy(model, system, prompt, *a, **kw):
        captured.append({"model": model, "system": system, "prompt": prompt,
                        "kw": dict(kw)})
        return "from build123d import *\nresult = Box(1, 1, 1)\n"

    monkeypatch.setattr(engine, "_ollama", spy)
    monkeypatch.setattr(engine, "_ACTIVE_CODE_MODEL", "local:test-arm")

    harvest._student_generate("a spec text", ["note1"], 0.2)
    harvest._teacher_generate("a spec text", ["note1"], 0.2)

    assert len(captured) == 2
    assert captured[0]["system"] == captured[1]["system"]
    assert captured[0]["prompt"] == captured[1]["prompt"]
    assert captured[0]["model"] == captured[1]["model"] == "local:test-arm"
    assert not captured[0]["kw"].get("think")
    assert captured[1]["kw"].get("think") is True


def test_teacher_generate_result_records_think_true_in_prompt_info(monkeypatch):
    def spy(model, system, prompt, *a, **kw):
        return "from build123d import *\nresult = Box(1, 1, 1)\n"
    monkeypatch.setattr(engine, "_ollama", spy)
    monkeypatch.setattr(engine, "_ACTIVE_CODE_MODEL", "local:test-arm")
    code, prompt_info = harvest._teacher_generate("a spec", [], 0.2)
    assert prompt_info["model"] == "local:test-arm"
    assert "result = Box" in code


def test_no_duplicated_prompt_string_no_em_dash():
    """M4's other requirement: removing the hand-built think prompt also removed the one
    added line that carried a literal em dash (a copy of cad_engine.py's own
    "USER REQUEST (verbatim ...)" prompt text). Checked against _teacher_generate's own
    source specifically, not the whole file -- this module's docstrings legitimately
    quote that phrase as documentation of what USED to be there."""
    import inspect
    src = inspect.getsource(harvest._teacher_generate)
    assert "USER REQUEST" not in src
    assert "\u2014" not in src


# ---------------------------------------------------------------------------
# system_sha1 dedup (Task 3 fix M5)
# ---------------------------------------------------------------------------

def test_store_system_writes_once_and_returns_a_stable_hash(tmp_path):
    h1 = harvest._store_system("a long system prompt")
    h2 = harvest._store_system("a long system prompt")
    assert h1 == h2
    assert (harvest.SYSTEMS_DIR / f"{h1}.txt").read_text() == "a long system prompt"


def test_store_system_returns_none_for_empty_string():
    assert harvest._store_system("") is None


def test_good_pair_row_carries_system_sha1_not_the_raw_system_text(monkeypatch, tmp_path):
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert "system" not in pairs[0]
    assert pairs[0]["system_sha1"] is not None
    assert (harvest.SYSTEMS_DIR / f"{pairs[0]['system_sha1']}.txt").read_text() == "SYS"


def test_pair_index_reads_pairs_file_only_once(monkeypatch, tmp_path):
    """Task 3 fix M5: PairIndex is built once and reused, not re-read per spec."""
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "spec_id": "s1", "tier": 2, "kind": "good", "code": "existing-code"})
    read_calls = {"n": 0}
    real_read_jsonl = harvest._read_jsonl

    def counting_read_jsonl(path):
        if path == harvest.PAIRS_FILE:
            read_calls["n"] += 1
        return real_read_jsonl(path)

    monkeypatch.setattr(harvest, "_read_jsonl", counting_read_jsonl)
    pair_index = harvest.PairIndex()
    assert read_calls["n"] == 1
    pair_index.has("s1", "somehash")
    pair_index.add("s2", "otherhash", 3)
    assert read_calls["n"] == 1   # no further reads from .has()/.add()


# ---------------------------------------------------------------------------
# Teacher promotion + think-unavailable skip
# ---------------------------------------------------------------------------

def test_eligible_pools_promotes_after_two_student_attempts():
    bank = [_spec_row("a"), _spec_row("b"), _spec_row("c")]
    progress = {
        "a": {"student_attempts": 0, "teacher_attempts": 0, "pairs": 0},
        "b": {"student_attempts": 2, "teacher_attempts": 0, "pairs": 0},
        "c": {"student_attempts": 5, "teacher_attempts": 3, "pairs": 2},   # done
    }
    cfg = _default_cfg(max_pairs_per_spec=2)
    student, teacher = harvest._eligible_pools(bank, progress, cfg)
    assert [r["id"] for r in student] == ["a"]
    assert [r["id"] for r in teacher] == ["b"]


def test_choose_mode_is_student_below_20_teacher_specs(monkeypatch):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    cfg = _default_cfg()
    pool = [_spec_row(f"t{i}") for i in range(19)]
    assert harvest._choose_mode(pool, cfg) == "student"


def test_choose_mode_is_think_at_20_teacher_specs_when_available(monkeypatch):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    cfg = _default_cfg()
    pool = [_spec_row(f"t{i}") for i in range(20)]
    assert harvest._choose_mode(pool, cfg) == "think"


def test_choose_mode_stays_student_when_think_unavailable(monkeypatch):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: False)
    cfg = _default_cfg()
    pool = [_spec_row(f"t{i}") for i in range(50)]
    assert harvest._choose_mode(pool, cfg) == "student"


def test_choose_mode_stays_student_when_not_configured(monkeypatch):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    cfg = _default_cfg(teacher_passes=[])
    pool = [_spec_row(f"t{i}") for i in range(50)]
    assert harvest._choose_mode(pool, cfg) == "student"


def test_status_think_pass_says_why_it_is_skipped(monkeypatch):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: False)
    _plant_bank([_spec_row(f"t{i}") for i in range(25)])
    progress = {f"t{i}": {"student_attempts": 2, "teacher_attempts": 0, "pairs": 0}
               for i in range(25)}
    harvest._save_progress(progress)
    status = harvest._compute_status()
    assert status["think_pass"]["configured"] is True
    assert status["think_pass"]["available"] is False
    assert status["think_pass"]["eligible_specs"] == 25
    assert "think_rung_available" in status["think_pass"]["note"]


# ---------------------------------------------------------------------------
# run_once honours teacher_passes config (Task 3 fix L4)
# ---------------------------------------------------------------------------

def test_run_once_stays_student_when_think_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    _plant_bank([_spec_row("s1")])
    harvest._save_progress({"s1": {"student_attempts": 2, "teacher_attempts": 0, "pairs": 0}})
    result = harvest.run_once("s1", _default_cfg(candidates=1, temps=[0.2], teacher_passes=[]))
    assert result["mode"] == "student"


def test_run_once_uses_think_after_two_failed_student_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    _patch_generate_sequence(monkeypatch, ["code-A"], mode="think")
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    _plant_bank([_spec_row("s1")])
    progress = {"s1": {"student_attempts": 2, "teacher_attempts": 0, "pairs": 0}}
    harvest._save_progress(progress)
    result = harvest.run_once("s1", _default_cfg(candidates=1, temps=[0.2]))
    assert result["mode"] == "think"


def test_run_once_unknown_spec_id_raises_system_exit(monkeypatch):
    _plant_bank([_spec_row("a")])
    with pytest.raises(SystemExit):
        harvest.run_once("does-not-exist", _default_cfg())


# ---------------------------------------------------------------------------
# Tier-first scheduling
# ---------------------------------------------------------------------------

def test_order_specs_prefers_tier34_first_when_share_is_low():
    pool = [_spec_row("p1", tier=1), _spec_row("p2", tier=2),
           _spec_row("h1", tier=3), _spec_row("h2", tier=4)]
    ordered = harvest._order_specs(pool, {}, prefer_tier34=True)
    assert set(r["id"] for r in ordered[:2]) == {"h1", "h2"}


def test_order_specs_round_robins_by_attempts_when_share_is_high():
    pool = [_spec_row("a", tier=1), _spec_row("b", tier=1), _spec_row("c", tier=1)]
    progress = {"a": {"student_attempts": 3}, "b": {"student_attempts": 0},
               "c": {"student_attempts": 1}}
    ordered = harvest._order_specs(pool, progress, prefer_tier34=False)
    assert [r["id"] for r in ordered] == ["b", "c", "a"]


def test_order_specs_tier34_bucket_itself_is_attempt_ordered():
    progress = {"h1": {"student_attempts": 2}, "h2": {"student_attempts": 0}}
    pool = [_spec_row("h1", tier=3), _spec_row("h2", tier=4), _spec_row("p1", tier=1)]
    ordered = harvest._order_specs(pool, progress, prefer_tier34=True)
    assert [r["id"] for r in ordered] == ["h2", "h1", "p1"]


# ---------------------------------------------------------------------------
# Unit gate refusals, including --check-gate (Task 3 fix H4)
# ---------------------------------------------------------------------------

def test_unit_gate_paused(monkeypatch):
    harvest.PAUSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    harvest.PAUSED_FILE.write_text("")
    reason = harvest._unit_gate(_default_cfg())
    assert reason is not None and "paused" in reason


def test_unit_gate_outside_night_window(monkeypatch):
    import datetime as dt

    class FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 19, 12, 0, 0)   # noon: well outside 22:00-07:00

    monkeypatch.setattr(harvest, "datetime", FixedDatetime)
    _plant_bank([_spec_row("a")])
    reason = harvest._unit_gate(_default_cfg(day_allowed=False))
    assert reason is not None and "night window" in reason


def test_unit_gate_hours_budget_exhausted(monkeypatch):
    monkeypatch.setattr(harvest, "_hours_today", lambda now_local=None: 999.0)
    _plant_bank([_spec_row("a")])
    reason = harvest._unit_gate(_default_cfg(day_allowed=True))
    assert reason is not None and "hours_today" in reason


def test_unit_gate_gpu_proxy_queued(monkeypatch):
    monkeypatch.setattr(harvest, "_gpu_proxy_waiting", lambda timeout=2.0: 3)
    _plant_bank([_spec_row("a")])
    reason = harvest._unit_gate(_default_cfg(day_allowed=True))
    assert reason is not None and "gpu-proxy" in reason


def test_unit_gate_bank_exhausted(monkeypatch):
    monkeypatch.setattr(harvest, "_gpu_proxy_waiting", lambda timeout=2.0: 0)
    _plant_bank([_spec_row("a")])
    progress = {"a": {"student_attempts": 5, "teacher_attempts": 5, "pairs": 2}}
    harvest._save_progress(progress)
    reason = harvest._unit_gate(_default_cfg(day_allowed=True, max_pairs_per_spec=2))
    assert reason is not None and "exhausted" in reason


def test_unit_gate_passes_when_nothing_blocks(monkeypatch):
    monkeypatch.setattr(harvest, "_gpu_proxy_waiting", lambda timeout=2.0: 0)
    _plant_bank([_spec_row("a")])
    reason = harvest._unit_gate(_default_cfg(day_allowed=True))
    assert reason is None


def test_build_lock_free_true_when_uncontended(tmp_path):
    assert harvest._build_lock_free() is True


def test_build_lock_free_false_when_another_process_holds_it(tmp_path):
    import fcntl
    lock_path = harvest.BUILD_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        # A second, independent open (a different open file description) must see it
        # as held -- flock() is per-open-file-description, matching a real second
        # process's view of the same lock file.
        assert harvest._build_lock_free() is False
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def test_check_gate_reports_the_unit_gate_reason_first(monkeypatch):
    harvest.PAUSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    harvest.PAUSED_FILE.write_text("")
    reason = harvest._check_gate()
    assert reason is not None and "paused" in reason


def test_check_gate_reports_lock_held_when_unit_gate_passes(monkeypatch):
    monkeypatch.setattr(harvest, "_unit_gate", lambda cfg: None)
    monkeypatch.setattr(harvest, "_build_lock_free", lambda: False)
    reason = harvest._check_gate()
    assert reason == "the CAD build lock is currently held"


def test_check_gate_none_when_everything_is_clear(monkeypatch):
    monkeypatch.setattr(harvest, "_unit_gate", lambda cfg: None)
    monkeypatch.setattr(harvest, "_build_lock_free", lambda: True)
    assert harvest._check_gate() is None


def test_check_gate_cli_needs_no_gpu_window(monkeypatch):
    """Task 3 fix H4: --check-gate must work with no CAD_GPU_WINDOW set at all."""
    monkeypatch.delenv("CAD_GPU_WINDOW", raising=False)
    harvest.PAUSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    harvest.PAUSED_FILE.write_text("")
    monkeypatch.setattr(sys, "argv", ["harvest.py", "--check-gate"])
    rc = harvest.main()
    assert rc == 3


def test_check_gate_cli_returns_0_on_go(monkeypatch):
    monkeypatch.delenv("CAD_GPU_WINDOW", raising=False)
    monkeypatch.setattr(harvest, "_check_gate", lambda: None)
    monkeypatch.setattr(sys, "argv", ["harvest.py", "--check-gate"])
    assert harvest.main() == 0


# ---------------------------------------------------------------------------
# --unit-minutes / --allow-day manual overrides (for a bounded manual --unit smoke
# without redirecting scripts/arms.py's writes away from the real cad.json)
# ---------------------------------------------------------------------------

def test_unit_minutes_and_allow_day_cli_overrides_reach_run_unit(monkeypatch):
    monkeypatch.setenv("CAD_GPU_WINDOW", "1")
    captured_cfg = {}

    def fake_run_unit(cfg):
        captured_cfg.update(cfg)
        return {"unit_id": "u1", "mode": "student", "specs_processed": 0,
               "builds_pruned": 0, "status": {}}

    monkeypatch.setattr(harvest, "run_unit", fake_run_unit)
    monkeypatch.setattr(harvest, "_unit_gate", lambda cfg: None)
    monkeypatch.setattr(harvest, "_check_code_model_pin", lambda: None)

    class _NullArmWindow:
        def __enter__(self):
            return "test-arm"

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(harvest, "arm_window", lambda arm: _NullArmWindow())
    monkeypatch.setattr(sys, "argv",
                        ["harvest.py", "--unit", "--unit-minutes", "10", "--allow-day",
                         "--i-am-the-owner"])
    rc = harvest.main()
    assert rc == 0
    assert captured_cfg["unit_minutes"] == 10.0
    assert captured_cfg["day_allowed"] is True


def test_unit_minutes_override_is_seen_by_the_gate_check_too(monkeypatch):
    """The gate check and run_unit must see the SAME overridden cfg object -- a bug
    where main() re-reads lab_config() fresh for run_unit would silently drop the CLI
    override the gate had just been evaluated against."""
    monkeypatch.setenv("CAD_GPU_WINDOW", "1")
    monkeypatch.setattr(harvest, "_check_code_model_pin", lambda: None)
    seen_by_gate = {}

    def fake_unit_gate(cfg):
        seen_by_gate.update(cfg)
        return None

    monkeypatch.setattr(harvest, "_unit_gate", fake_unit_gate)
    monkeypatch.setattr(harvest, "run_unit", lambda cfg: {
        "unit_id": "u1", "mode": "student", "specs_processed": 0, "builds_pruned": 0,
        "status": {}})

    class _NullArmWindow:
        def __enter__(self):
            return "test-arm"

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(harvest, "arm_window", lambda arm: _NullArmWindow())
    monkeypatch.setattr(sys, "argv", ["harvest.py", "--unit", "--unit-minutes", "10", "--i-am-the-owner"])
    harvest.main()
    assert seen_by_gate["unit_minutes"] == 10.0


# ---------------------------------------------------------------------------
# Ledger / pair / status row shapes
# ---------------------------------------------------------------------------

LEDGER_KEYS = {"ts", "unit_id", "spec_id", "tier", "arm", "model", "pass", "candidate",
              "temperature", "ok", "gate_hard", "gate_spec", "gate_adv", "band", "ref",
              "chamfer_mm", "tokens_in", "tokens_out", "seconds", "build_dir", "error",
              "unscored_reason",
              # Fix round 2, Section A: the code fingerprint (when generated) and the
              # confirmation outcome (None/"reference"/"agreement"/"unconfirmed"/"split").
              "fingerprint", "agreement",
              # Fix round 3, M3/H2c: the effective gate configuration this row was
              # judged under, and (for a confirmed good row) how strongly.
              "gate_version", "confirm_strength"}

GOOD_PAIR_KEYS = {"id", "spec_id", "spec", "tier", "group", "source", "kind", "turn",
                  "band", "arm", "model", "temperature", "system_sha1", "prompt", "code",
                  "bad_code", "problem", "facts", "verified", "ts", "unit_id",
                  # Fix round 2, Section A: the geometric signature and why this pair
                  # was trusted ("reference" or "agreement").
                  "signature", "confirmed_by",
                  # Fix round 3, H2b/M3: how much weight the confirmation can bear, how
                  # similar this code is to the rest of its agreeing set, and the
                  # effective gate configuration it was judged under.
                  "confirm_strength", "code_similarity", "gate_version"}


def test_ledger_row_has_the_documented_shape():
    row = harvest._ledger_row(unit_id="u1", spec_id="s1", tier=2, mode="student",
                              candidate="0", temperature=0.2, m=_clean_m(),
                              band_info={"band": "match", "reference": "r.stl",
                                        "chamfer_mm": 0.1},
                              usage={"prompt_tokens": 10, "completion_tokens": 5},
                              model="local:test-arm", build_dir="/tmp/somewhere",
                              seconds=12.3)
    assert set(row.keys()) == LEDGER_KEYS
    assert row["ok"] is True
    assert row["band"] == "match"
    assert row["tokens_in"] == 10
    assert row["tokens_out"] == 5
    assert row["model"] == "local:test-arm"
    assert row["unscored_reason"] is None


def test_pair_row_has_the_documented_shape(monkeypatch, tmp_path):
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 1
    assert set(pairs[0].keys()) == GOOD_PAIR_KEYS
    assert pairs[0]["source"] == "student"
    assert pairs[0]["turn"] == "first"
    assert pairs[0]["confirmed_by"] == "reference"


def test_pair_row_source_is_teacher_think_for_the_think_pass(monkeypatch, tmp_path):
    row = _confirm_via_reference(monkeypatch, "s1")
    _patch_generate_sequence(monkeypatch, ["code-A"], mode="think")
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "think", "unit1", {}, harvest.PairIndex())
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert pairs[0]["source"] == "teacher:think"
    assert pairs[0]["arm"] == "test-arm"


def test_status_shape_and_pairs_stats(monkeypatch, tmp_path):
    _plant_bank([_spec_row("a", tier=3), _spec_row("b", tier=1)])
    harvest._append_jsonl(harvest.PAIRS_FILE, {"spec_id": "a", "tier": 3, "source": "student",
                                              "kind": "good"})
    harvest._append_jsonl(harvest.PAIRS_FILE, {"spec_id": "b", "tier": 1, "source": "student",
                                              "kind": "good"})
    status = harvest._compute_status()
    assert status["specs"]["total"] == 2
    assert status["specs"]["by_tier"] == {"3": 1, "1": 1}
    assert status["pairs"]["total"] == 2
    assert status["pairs"]["tier34_share"] == 0.5
    # Task 3 fix M6: good-only numbers reported alongside the all-kinds ones.
    assert status["pairs"]["good_total"] == 2
    assert status["pairs"]["good_tier34_share"] == 0.5
    for key in ("updated", "specs", "pairs", "ledger", "think_pass", "budget",
               "nights_completed", "timer_active"):
        assert key in status


def test_status_pairs_good_total_excludes_fail_kind_pairs(monkeypatch, tmp_path):
    _plant_bank([_spec_row("a", tier=3), _spec_row("b", tier=1)])
    harvest._append_jsonl(harvest.PAIRS_FILE, {"spec_id": "a", "tier": 3, "source": "student",
                                              "kind": "good"})
    harvest._append_jsonl(harvest.PAIRS_FILE, {"spec_id": "b", "tier": 1, "source": "student",
                                              "kind": "fail"})
    status = harvest._compute_status()
    assert status["pairs"]["total"] == 2
    assert status["pairs"]["good_total"] == 1
    assert status["pairs"]["good_tier34_share"] == 1.0   # the only good pair is tier 3


def test_status_ledger_unscored_count(tmp_path):
    """Task 3 fix H2: "counted separately in status"."""
    harvest._append_jsonl(harvest.LEDGER_FILE, {"pass": "student", "tier": 2, "ok": True,
                                               "gate_hard": [], "gate_spec": [], "band": None,
                                               "unscored_reason": "inspect timed out"})
    harvest._append_jsonl(harvest.LEDGER_FILE, {"pass": "student", "tier": 2, "ok": True,
                                               "gate_hard": [], "gate_spec": [], "band": None,
                                               "unscored_reason": None})
    status = harvest._compute_status()
    assert status["ledger"]["unscored"] == 1
    assert status["ledger"]["attempts"] == 2


def test_pass_rate_by_pass_and_tier(tmp_path):
    """Fix round 2: pass rates are computed on CONFIRMED goods (Section A) -- the
    "good" row below is given an explicit agreement="agreement" to represent a
    confirmed candidate, and the plain gate_hard failure stays unconfirmed (agreement
    defaults to None), same 0.5/1.0 split the pre-fix-round-2 test encoded."""
    rows = [
        harvest._ledger_row(unit_id="u1", spec_id="s1", tier=3, mode="student", candidate="0",
                            temperature=0.2, m=_clean_m(), band_info={}, usage={},
                            model="local:x", build_dir=None, seconds=1.0,
                            agreement="agreement"),
        harvest._ledger_row(unit_id="u1", spec_id="s1", tier=3, mode="student", candidate="1",
                            temperature=0.5, m=_clean_m(gate_hard=["bad"]), band_info={},
                            usage={}, model="local:x", build_dir=None, seconds=1.0),
        harvest._ledger_row(unit_id="u1", spec_id="s2", tier=1, mode="think", candidate="0",
                            temperature=0.2, m=_clean_m(), band_info={}, usage={},
                            model="local:x", build_dir=None, seconds=1.0,
                            agreement="reference"),
    ]
    for r in rows:
        harvest._append_jsonl(harvest.LEDGER_FILE, r)
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    by_pass = harvest._pass_rate(ledger, lambda r: r["pass"])
    by_tier = harvest._pass_rate(ledger, lambda r: str(r["tier"]))
    assert by_pass == {"student": 0.5, "think": 1.0}
    assert by_tier == {"3": 0.5, "1": 1.0}


# ---------------------------------------------------------------------------
# run_unit: wall-clock deadline (Task 3 fix M1: checked inside the candidate loop too)
# ---------------------------------------------------------------------------

def test_run_unit_stops_between_specs_at_the_real_deadline(monkeypatch, tmp_path):
    """Task 3 fix M7: assert the REAL stop point deterministically, not just an upper
    bound (the old test asserted `specs_processed <= 5` on a 5-spec bank, which passes
    even if the deadline check does nothing at all). `sample_spec` itself is mocked out
    here (its own deadline handling is covered separately, above) so only run_unit's
    between-specs deadline arithmetic is under test: a fake clock that advances by
    exactly 1.0 per spec "processed", with a budget of 2.5, must process exactly 3 specs
    (0.0 < 2.5, 1.0 < 2.5, 2.0 < 2.5, then 3.0 >= 2.5 stops the 4th)."""
    _plant_bank([_spec_row(f"s{i}") for i in range(5)])

    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    processed_ids = []

    def fake_sample_spec(row, cfg, mode, unit_id, progress, pair_index, deadline=None):
        processed_ids.append(row["id"])
        fake_now["t"] += 1.0

    monkeypatch.setattr(harvest.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(harvest, "sample_spec", fake_sample_spec)

    cfg = _default_cfg(unit_minutes=2.5 / 60)
    result = harvest.run_unit(cfg)
    assert processed_ids == ["s0", "s1", "s2"]
    assert result["specs_processed"] == 3


def test_run_unit_deadline_is_also_checked_inside_the_candidate_loop(monkeypatch, tmp_path):
    """Task 3 fix M1: the deadline must be checked before each codegen call inside
    sample_spec, not only between specs -- a spec configured with several candidates must
    stop mid-spec once the deadline passes."""
    calls = {"n": 0}

    def fake_generate(spec, notes, temperature):
        calls["n"] += 1
        return f"code-{calls['n']}", {"model": "local:x", "system": "SYS", "prompt": "P",
                                      "usage": {}}

    monkeypatch.setattr(harvest, "_student_generate", fake_generate)
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    row = _spec_row("s1")
    # A deadline already in the past: no candidate should even be attempted.
    cfg = _default_cfg(candidates=5, temps=[0.2], max_pairs_per_spec=5)
    harvest.sample_spec(row, cfg, "student", "unit1", {}, harvest.PairIndex(),
                        deadline=time.monotonic() - 1)
    assert calls["n"] == 0
    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []


# ---------------------------------------------------------------------------
# Retention (Task 3 fix L9)
# ---------------------------------------------------------------------------

def test_prune_builds_keeps_the_newest_by_mtime(tmp_path):
    import time as _time
    harvest.BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    dirs = []
    for i in range(5):
        d = harvest.BUILDS_DIR / f"build{i}"
        d.mkdir()
        (d / "build.step").write_text("x")
        os.utime(d, (i, i))   # deterministic, ascending mtimes
        dirs.append(d)
    removed = harvest._prune_builds(keep=2)
    assert removed == 3
    remaining = sorted(p.name for p in harvest.BUILDS_DIR.iterdir())
    assert remaining == ["build3", "build4"]   # the two newest survive


def test_prune_builds_no_op_when_under_the_cap(tmp_path):
    harvest.BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    (harvest.BUILDS_DIR / "build0").mkdir()
    assert harvest._prune_builds(keep=500) == 0


def test_prune_builds_missing_dir_is_a_no_op():
    assert harvest._prune_builds(keep=10) == 0


def test_run_unit_prunes_builds_at_the_end(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(harvest, "_prune_builds", lambda keep: calls.append(keep) or 0)
    _plant_bank([])   # empty bank -> nothing to sample, still prunes at the end
    result = harvest.run_unit(_default_cfg(keep_builds=123))
    assert calls == [123]
    assert result["builds_pruned"] == 0


# ---------------------------------------------------------------------------
# progress.json reconciliation against pairs.jsonl (fix round 3, M4)
# ---------------------------------------------------------------------------

def test_reconcile_progress_repairs_a_stale_low_pairs_count(monkeypatch, tmp_path):
    """Fix round 3, M4: a crash between a pair append and the next _save_progress call
    leaves progress.json's own "pairs" counter stale-LOW; pairs.jsonl is the truth, and
    run_unit must reconcile against it BEFORE deciding whether a spec is still eligible,
    or a later unit would sample straight past max_pairs_per_spec. Simulated here as:
    2 good pairs already really exist on disk, but progress.json (never saved after the
    simulated crash) does not exist at all -- the worst-case stale value, 0."""
    row = _spec_row("s1")
    _plant_bank([row])
    cfg = _default_cfg(max_pairs_per_spec=2, candidates=1, temps=[0.2])
    for i in range(2):
        harvest._append_jsonl(harvest.PAIRS_FILE, {
            "spec_id": "s1", "tier": 2, "kind": "good", "code": f"existing-code-{i}",
            "id": f"p:s1:existing{i}"})
    assert not harvest.PROGRESS_FILE.exists()

    calls = _patch_generate_sequence(monkeypatch, ["should-not-be-called"] * 5)
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: {"error": None})
    _patch_regate_passthrough(monkeypatch)

    harvest.run_unit(cfg)
    # Reconciled BEFORE pool-eligibility was checked: s1 was already full, so it was
    # never even sampled.
    assert calls["n"] == 0
    assert harvest._load_progress()["s1"]["pairs"] == 2

    # A second run_unit call is equally safe (idempotent reconciliation) -- never more
    # than max_pairs_per_spec good pairs for the spec, across the crash and both reruns.
    harvest.run_unit(cfg)
    assert calls["n"] == 0
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len([p for p in pairs if p["spec_id"] == "s1"]) == 2


def test_reconcile_progress_leaves_attempt_counters_untouched():
    """pairs.jsonl can tell "pairs", never student_attempts/teacher_attempts (fix round
    3, M4) -- reconciliation must not reset or otherwise touch those."""
    bank = [_spec_row("s1")]
    progress = {"s1": {"student_attempts": 3, "teacher_attempts": 1, "pairs": 0}}
    harvest._append_jsonl(harvest.PAIRS_FILE, {
        "spec_id": "s1", "tier": 2, "kind": "good", "code": "x", "id": "p:s1:x"})
    harvest._reconcile_progress(progress, bank, harvest.PairIndex())
    assert progress["s1"]["pairs"] == 1
    assert progress["s1"]["student_attempts"] == 3
    assert progress["s1"]["teacher_attempts"] == 1


# ---------------------------------------------------------------------------
# Real-subprocess tests: drive the ACTUAL lab.harvest.main(), with ONLY the model call
# (cad_engine._ollama, or cad_engine.run_step for the H3 test) and the arm switch
# (lab._armwindow._run_arms/_ensure_resident_up) patched, inside the spawned subprocess
# itself. PYTHONUTF8=0 per the Task 3 rulings (decoding must stay correct without the
# interpreter's own UTF-8 mode masking a locale bug). A temp CAD_CONFIG_FILE/MAKER_ENV
# point _pre_arm_marker_paths() (which imports scripts/arms.py's own CAD_JSON/ENV_PATH)
# at a throwaway directory, and a temp specs.jsonl stands in for the bank -- nothing
# under the real ~/.openclaw/ or the real lab/state/ is ever touched.
# ---------------------------------------------------------------------------

# {repo!r} = this repo's root; {cad_json!r}/{env_path!r} = temp CAD_CONFIG_FILE/MAKER_ENV;
# {state_dir!r} = temp lab/state equivalent; {specs_file!r} = temp spec bank.
# `scenario` picks what the faked calls do:
#   fast_then_slow -- (patches cad_engine._ollama) the FIRST call returns a valid box
#                     program immediately, the SECOND call sleeps 5s.
#   use_before_marker -- "use" sleeps 3s and NEVER writes the marker.
#   run_step_blocks -- (patches cad_engine.run_step, Task 3 fix H3) the model call
#                       returns instantly with valid code; run_step itself sleeps 5s,
#                       simulating a signal landing during EXECUTION rather than
#                       generation -- fluid_gen._materialize would otherwise swallow it.
_HARVEST_MAIN_HELPER = """
import json, os, sys, time
sys.path.insert(0, {repo!r})
os.environ["CAD_CONFIG_FILE"] = {cad_json!r}
os.environ["MAKER_ENV"] = {env_path!r}
os.environ["CAD_GPU_WINDOW"] = "1"
from pathlib import Path
from lab import harvest
from lab import _armwindow as aw
from lab import specbank
import cad_engine as engine
import arms as arms_cli

call_log = sys.argv[1]
scenario = sys.argv[2]

def log(entry):
    with open(call_log, "a") as f:
        f.write(json.dumps(entry) + "\\n")

state = Path({state_dir!r})
harvest.STATE_DIR = state
harvest.LEDGER_FILE = state / "ledger.jsonl"
harvest.PAIRS_FILE = state / "pairs.jsonl"
harvest.REVIEW_FILE = state / "review.jsonl"
harvest.PROGRESS_FILE = state / "progress.json"
harvest.STATUS_FILE = state / "status.json"
harvest.PAUSED_FILE = state / "paused"
harvest.BUILDS_DIR = state / "builds"
harvest.SYSTEMS_DIR = state / "systems"
specbank.SPECS_FILE = Path({specs_file!r})

pre_cad, pre_env = arms_cli._pre_arm_paths(arms_cli.CAD_JSON, arms_cli.ENV_PATH)

def fake_run_arms(*args):
    log(list(args))
    if args[0] == "use":
        if scenario == "use_before_marker":
            time.sleep(3)
        else:
            pre_cad.write_text("{{}}")
    class R:
        returncode = 0
    return R()

def fake_ensure_resident_up():
    log(["ensure_resident_up"])

aw._run_arms = fake_run_arms
aw._ensure_resident_up = fake_ensure_resident_up

calls = {{"n": 0}}
def fake_ollama(model, system, prompt, *a, **kw):
    calls["n"] += 1
    log(["ollama_call", calls["n"]])
    if scenario == "run_step_blocks":
        return "from build123d import *\\nresult = Box(10, 10, 10)\\n"
    if calls["n"] == 1:
        return "from build123d import *\\nresult = Box(10, 10, 10)\\n"
    time.sleep(5)
    return "from build123d import *\\nresult = Box(20, 20, 20)\\n"

engine._ollama = fake_ollama

if scenario == "run_step_blocks":
    real_run_step = engine.run_step
    def blocking_run_step(code, build_dir):
        log(["run_step_start"])
        time.sleep(5)
        log(["run_step_done"])
        return real_run_step(code, build_dir)
    engine.run_step = blocking_run_step

sys.argv = ["harvest.py", "--once", "--spec-id", "s1", "--arm", "test-arm"]
rc = harvest.main()
log(["exit", rc])
sys.exit(rc)
"""


def _write_temp_bank(specs_file: Path) -> None:
    specs_file.write_text(json.dumps({
        "id": "s1", "spec": "an 80x50x6mm plate", "tier": 2, "group": "plate",
        "source": "teacher-suite", "key": "k1", "added": "2026-09-19T00:00:00Z",
    }) + "\n")


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"
    return env


def _spawn_harvest_main(call_log: Path, cad_json: Path, env_path: Path, state_dir: Path,
                        specs_file: Path, scenario: str) -> subprocess.Popen:
    code = _HARVEST_MAIN_HELPER.format(
        repo=str(HERE), cad_json=str(cad_json), env_path=str(env_path),
        state_dir=str(state_dir), specs_file=str(specs_file))
    return subprocess.Popen([sys.executable, "-c", code, str(call_log), scenario],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_child_env())


def _read_tags(call_log: Path) -> list:
    if not call_log.exists():
        return []
    return [json.loads(line)[0] for line in call_log.read_text().splitlines() if line.strip()]


def _read_entries(call_log: Path) -> list:
    if not call_log.exists():
        return []
    return [json.loads(line) for line in call_log.read_text().splitlines() if line.strip()]


def _wait_for_entry(call_log: Path, entry: list, timeout: float = 25.0,
                    poll: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if entry in _read_entries(call_log):
            return True
        time.sleep(poll)
    return False


def test_sigterm_mid_spec_stops_promptly_ledger_stays_clean_restore_called_once(tmp_path):
    """candidate #1 runs to completion for real (a genuine build123d execute/inspect/
    render/gate pass, which costs several real seconds -- polled for rather than
    guessed, since it is not a fixed-latency operation); the SIGTERM lands inside
    candidate #2's model call, which is patched to sleep, so there is no ambiguity about
    where in the pipeline the signal landed."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    _write_temp_bank(specs_file)

    proc = _spawn_harvest_main(call_log, cad_json, env_path, state_dir, specs_file,
                               "fast_then_slow")
    try:
        assert _wait_for_entry(call_log, ["ollama_call", 2], timeout=30.0), (
            "second candidate's model call never started: " + str(_read_entries(call_log)))
        time.sleep(0.3)   # inside its 5s sleep, not just at the very start of it
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=15.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()

    tags = _read_tags(call_log)
    assert tags[0] == "use"
    assert "ollama_call" in tags
    assert tags.count("restore") == 1
    assert tags[-1] == "exit"
    assert rc != 0

    ledger_file = state_dir / "ledger.jsonl"
    if ledger_file.exists():
        lines = [ln for ln in ledger_file.read_text().splitlines() if ln.strip()]
        for ln in lines:
            json.loads(ln)   # every line parses cleanly -- no half-written JSON
        # At most the first candidate's row(s) -- the second candidate's generation was
        # interrupted before _materialize (or even the codegen call) ever returned, so it
        # contributed nothing, complete or partial.
        assert len(lines) <= 2

    err_text = err.decode(errors="replace")
    last_line = [ln for ln in err_text.splitlines() if ln.strip()][-1]
    assert last_line.startswith("harvest abort:")
    assert "signal" in last_line


def test_sigterm_during_execution_not_generation_leaves_no_bogus_failure_row(tmp_path):
    """Task 3 fix H3, the exact regression (repro3_abort_swallowed.py): a SIGTERM landing
    DURING cad_engine.run_step (execution, not the model call) used to be swallowed by
    fluid_gen._materialize's bare except and recorded as an ordinary build failure. This
    drives the REAL lab.harvest.main() with cad_engine.run_step itself patched to block,
    proving the abort flag catches it even though the exception it also raised never
    reaches sample_spec directly."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    _write_temp_bank(specs_file)

    proc = _spawn_harvest_main(call_log, cad_json, env_path, state_dir, specs_file,
                               "run_step_blocks")
    try:
        assert _wait_for_entry(call_log, ["run_step_start"], timeout=30.0), (
            "run_step was never reached: " + str(_read_entries(call_log)))
        time.sleep(0.3)   # inside run_step's 5s sleep
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=15.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()

    tags = _read_tags(call_log)
    assert tags[0] == "use"
    assert "run_step_start" in tags
    assert "run_step_done" not in tags   # interrupted mid-sleep, never completed
    assert tags.count("restore") == 1
    assert tags[-1] == "exit"
    assert rc != 0

    # No ledger row at all for the interrupted candidate: NOT recorded as a build
    # failure ("terminated by signal 15" must never look like the model's fault).
    ledger_file = state_dir / "ledger.jsonl"
    if ledger_file.exists():
        lines = [ln for ln in ledger_file.read_text().splitlines() if ln.strip()]
        for ln in lines:
            row = json.loads(ln)
            assert "signal" not in (row.get("error") or "")
    progress_file = state_dir / "progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        assert progress.get("s1", {}).get("student_attempts", 0) == 0

    err_text = err.decode(errors="replace")
    last_line = [ln for ln in err_text.splitlines() if ln.strip()][-1]
    assert last_line.startswith("harvest abort:")


def test_signal_before_use_marker_skips_restore_and_leaves_cad_json_alone(tmp_path):
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    _write_temp_bank(specs_file)
    placeholder = '{"untouched": true}'
    cad_json.write_text(placeholder)

    proc = _spawn_harvest_main(call_log, cad_json, env_path, state_dir, specs_file,
                               "use_before_marker")
    try:
        time.sleep(0.4)   # inside "use"'s 3s sleep, well before any marker is written
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=6.0)
        rc = proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()

    tags = _read_tags(call_log)
    assert tags == ["use", "ensure_resident_up", "exit"], tags
    assert "restore" not in tags
    assert "ollama_call" not in tags   # never reached the model call at all
    assert rc != 0
    assert cad_json.read_text() == placeholder
    assert not (cad_json.with_suffix(".json.pre-arm")).exists()
    assert not (state_dir / "ledger.jsonl").exists()


def test_refusal_outside_gpu_window_touches_nothing(tmp_path):
    """No CAD_GPU_WINDOW, no --i-know-the-gpu-is-free: main() must refuse before ANYTHING
    -- no signal handler installed, no arm switch, no model call, no state files."""
    call_log = tmp_path / "calls.jsonl"
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    _write_temp_bank(specs_file)

    code = _HARVEST_MAIN_HELPER.format(
        repo=str(HERE), cad_json=str(cad_json), env_path=str(env_path),
        state_dir=str(state_dir), specs_file=str(specs_file))
    # Same helper, but with CAD_GPU_WINDOW stripped right back out before exec -- the
    # helper itself sets it unconditionally, so patch it out of the launched env instead.
    code = code.replace('os.environ["CAD_GPU_WINDOW"] = "1"\n', "")
    env = _child_env()
    env.pop("CAD_GPU_WINDOW", None)
    proc = subprocess.run([sys.executable, "-c", code, str(call_log), "fast_then_slow"],
                          capture_output=True, env=env, timeout=10.0)

    assert proc.returncode != 0
    assert _read_tags(call_log) == []   # nothing was ever called
    # state_dir itself may exist (this test file's autouse fixture pre-creates a
    # same-named tmp dir for the OUTER pytest process, unrelated to the subprocess) --
    # what matters is the subprocess never wrote anything into it.
    assert not (state_dir / "ledger.jsonl").exists()
    assert not (state_dir / "progress.json").exists()
    assert "GPU window" in proc.stderr.decode(errors="replace")


def test_check_gate_real_subprocess_needs_no_gpu_window(tmp_path):
    """Task 3 fix H4, real-subprocess proof: `harvest.py --check-gate` (no
    CAD_GPU_WINDOW) exits 3 (skip) or 0 (go) -- never the GPU-window refusal -- and
    touches no state."""
    cad_json = tmp_path / "cad.json"
    env_path = tmp_path / "maker.env"
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    _write_temp_bank(specs_file)
    code = f"""
import sys
sys.path.insert(0, {str(HERE)!r})
from pathlib import Path
from lab import harvest
from lab import specbank
harvest.STATE_DIR = Path({str(state_dir)!r})
harvest.LEDGER_FILE = harvest.STATE_DIR / "ledger.jsonl"
harvest.PAUSED_FILE = harvest.STATE_DIR / "paused"
harvest.BUILD_LOCK_FILE = harvest.STATE_DIR / "cad-build.lock"
specbank.SPECS_FILE = Path({str(specs_file)!r})
sys.argv = ["harvest.py", "--check-gate"]
sys.exit(harvest.main())
"""
    env = dict(os.environ)
    env.pop("CAD_GPU_WINDOW", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env,
                          timeout=10.0)
    assert proc.returncode in (0, 3)
    assert not (state_dir / "ledger.jsonl").exists()


# ---------------------------------------------------------------------------
# Real-subprocess proof of the agreement layer end to end (fix round 2, Section E):
# drives the ACTUAL lab.harvest.main() across several invocations sharing one on-disk
# state dir, with ONLY the model call (cad_engine._ollama), materialize
# (fluid_gen._materialize) and the arm switch (lab._armwindow._run_arms/
# _ensure_resident_up) patched inside the spawned subprocess -- harvest._regate is also
# patched, to a SCRIPTED facts sequence, since driving real build123d geometry to
# produce two specific agreeing-or-disagreeing signatures on demand is not practical
# here; _regate is the exact seam this module's own test suite already treats as safe
# to fake for orchestration-focused coverage (see _patch_regate_passthrough above).
# ---------------------------------------------------------------------------

_AGREEMENT_SUBPROCESS_HELPER = """
import json, sys
sys.path.insert(0, {repo!r})
import os
os.environ["CAD_CONFIG_FILE"] = {cad_json!r}
os.environ["MAKER_ENV"] = {env_path!r}
os.environ["CAD_GPU_WINDOW"] = "1"
from pathlib import Path
from lab import harvest
from lab import _armwindow as aw
from lab import specbank
import cad_engine as engine
import fluid_gen
import arms as arms_cli

script_file = sys.argv[1]
spec_id = sys.argv[2]

state = Path({state_dir!r})
harvest.STATE_DIR = state
harvest.LEDGER_FILE = state / "ledger.jsonl"
harvest.PAIRS_FILE = state / "pairs.jsonl"
harvest.REVIEW_FILE = state / "review.jsonl"
harvest.CANDIDATES_FILE = state / "candidates.jsonl"
harvest.PROGRESS_FILE = state / "progress.json"
harvest.STATUS_FILE = state / "status.json"
harvest.PAUSED_FILE = state / "paused"
harvest.BUILDS_DIR = state / "builds"
harvest.SYSTEMS_DIR = state / "systems"
specbank.SPECS_FILE = Path({specs_file!r})

pre_cad, pre_env = arms_cli._pre_arm_paths(arms_cli.CAD_JSON, arms_cli.ENV_PATH)

def fake_run_arms(*args):
    if args[0] == "use":
        pre_cad.write_text("{{}}")
    class R:
        returncode = 0
    return R()

aw._run_arms = fake_run_arms
aw._ensure_resident_up = lambda: None

script = json.loads(Path(script_file).read_text())
calls = {{"n": 0}}

def fake_ollama(model, system, prompt, *a, **kw):
    i = calls["n"]
    calls["n"] += 1
    return script["codes"][i]

engine._ollama = fake_ollama

def fake_materialize(code, build_dir, spec):
    return {{"error": None}}

fluid_gen._materialize = fake_materialize

facts_iter = iter(script["facts"])

def fake_regate(m_fluid, build_dir, spec):
    facts = next(facts_iter)
    return {{"error": None, "facts": facts, "gate_hard": [], "gate_spec": [],
             "gate_adv": [], "unscored_reason": None}}

harvest._regate = fake_regate

sys.argv = ["harvest.py", "--once", "--spec-id", spec_id, "--arm", "test-arm"]
rc = harvest.main()
print("RC", rc)
sys.exit(rc)
"""


def _write_agreement_cad_json(path: Path, candidates: int, temps: list, max_pairs: int) -> None:
    harvest_block = {
        "candidates": candidates, "temps": temps,
        "candidates_tier34": candidates, "temps_tier34": temps,
        "max_pairs_per_spec": max_pairs, "teacher_passes": [],
        "agreement": {"volume_tol_pct": 0.05, "bbox_tol_mm": 0.05,
                     "bore_round_mm": 0.01},
        "strict": {"envelope_tol_mm": 0.2},
    }
    path.write_text(json.dumps({"cad": {"lab": {"harvest": harvest_block}}}))


def _write_agreement_bank(specs_file: Path) -> None:
    # No envelope/through-hole phrasing in any of these specs: the strict checks
    # (Section B) must stay out of this test's way entirely, which is exercised
    # separately above (test_strict_*).
    rows = [
        {"id": "agree", "spec": "a simple test widget for the agreement suite",
         "tier": 1, "group": "test", "source": "unit-test", "key": "k-agree",
         "added": "2026-09-19T00:00:00Z"},
        {"id": "alone", "spec": "a lone test widget for the agreement suite",
         "tier": 1, "group": "test", "source": "unit-test", "key": "k-alone",
         "added": "2026-09-19T00:00:00Z"},
        {"id": "split", "spec": "a disputed test widget for the agreement suite",
         "tier": 1, "group": "test", "source": "unit-test", "key": "k-split",
         "added": "2026-09-19T00:00:00Z"},
    ]
    specs_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


_FACTS_SIMPLE = {"solids": 1, "faces": 6, "cyl_faces": 0, "cone_faces": 0,
                 "volume": 1000.0, "bbox": [10.0, 10.0, 10.0], "bores": [],
                 "through_holes": 0, "blind_holes": 0}
_FACTS_DIFFERENT = {"solids": 1, "faces": 10, "cyl_faces": 2, "cone_faces": 0,
                    "volume": 5000.0, "bbox": [30.0, 20.0, 15.0], "bores": [8.0],
                    "through_holes": 1, "blind_holes": 0}


def _run_agreement_subprocess(repo: Path, cad_json: Path, env_path: Path,
                              state_dir: Path, specs_file: Path, script_file: Path,
                              spec_id: str) -> subprocess.CompletedProcess:
    code = _AGREEMENT_SUBPROCESS_HELPER.format(
        repo=str(repo), cad_json=str(cad_json), env_path=str(env_path),
        state_dir=str(state_dir), specs_file=str(specs_file))
    return subprocess.run([sys.executable, "-c", code, str(script_file), spec_id],
                          capture_output=True, env=_child_env(), timeout=60.0)


def _pairs_for(state_dir: Path, spec_id: str) -> list:
    path = state_dir / "pairs.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [r for r in rows if r.get("spec_id") == spec_id]


def _candidates_for(state_dir: Path, spec_id: str) -> list:
    path = state_dir / "candidates.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [r for r in rows if r.get("spec_id") == spec_id]


def _ledger_for(state_dir: Path, spec_id: str) -> list:
    path = state_dir / "ledger.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [r for r in rows if r.get("spec_id") == spec_id]


def test_agreement_end_to_end_real_subprocess(tmp_path):
    """Task 3 fix round 2, Section E, the required real-subprocess proof, driving the
    ACTUAL lab.harvest.main() across several invocations of a shared on-disk state:
      - two agreeing distinct codes in ONE unit -> 2 good pairs
      - one gate-clean candidate alone -> 0 pairs, 1 candidates.jsonl row
      - a later unit's agreeing sample -> BOTH promoted, no duplicates on a third run
      - a split (disagreeing signatures) -> nothing confirmed
    """
    repo = HERE
    state_dir = tmp_path / "state"
    specs_file = tmp_path / "specs.jsonl"
    env_path = tmp_path / "maker.env"
    _write_agreement_bank(specs_file)

    # -- 1. "agree": two distinct codes, identical signature, in one unit --------------
    cad_multi = tmp_path / "cad_multi.json"
    _write_agreement_cad_json(cad_multi, candidates=2, temps=[0.2, 0.5], max_pairs=2)
    script_agree = tmp_path / "script_agree.json"
    script_agree.write_text(json.dumps({
        "codes": ["from build123d import *\nresult = Box(10, 10, 10)\n",
                 "from build123d import *\nbox_ = Box(10, 10, 10)\nresult = box_\n"],
        "facts": [dict(_FACTS_SIMPLE), dict(_FACTS_SIMPLE)],
    }))
    proc = _run_agreement_subprocess(repo, cad_multi, env_path, state_dir, specs_file,
                                     script_agree, "agree")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    agree_pairs = _pairs_for(state_dir, "agree")
    assert len(agree_pairs) == 2
    assert all(p["kind"] == "good" and p["confirmed_by"] == "agreement"
              for p in agree_pairs)
    assert len({p["code"] for p in agree_pairs}) == 2   # both distinct codes kept
    agree_ledger = _ledger_for(state_dir, "agree")
    assert all(r["agreement"] == "agreement" for r in agree_ledger if r.get("ok"))

    # -- 2. "alone": one gate-clean candidate, no partner yet ---------------------------
    cad_single = tmp_path / "cad_single.json"
    _write_agreement_cad_json(cad_single, candidates=1, temps=[0.2], max_pairs=2)
    script_alone_1 = tmp_path / "script_alone_1.json"
    script_alone_1.write_text(json.dumps({
        "codes": ["from build123d import *\nresult = Box(20, 20, 20)\n"],
        "facts": [dict(_FACTS_SIMPLE)],
    }))
    proc = _run_agreement_subprocess(repo, cad_single, env_path, state_dir, specs_file,
                                     script_alone_1, "alone")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _pairs_for(state_dir, "alone") == []
    alone_candidates = _candidates_for(state_dir, "alone")
    assert len(alone_candidates) == 1
    assert alone_candidates[0]["agreement"] == "unconfirmed"
    assert alone_candidates[0]["status"] != "promoted"
    alone_ledger_1 = _ledger_for(state_dir, "alone")
    assert any(r["agreement"] == "unconfirmed" for r in alone_ledger_1)

    # -- 3. next unit's agreeing sample for "alone" -> BOTH promoted --------------------
    script_alone_2 = tmp_path / "script_alone_2.json"
    script_alone_2.write_text(json.dumps({
        "codes": ["from build123d import *\ncube = Box(20, 20, 20)\nresult = cube\n"],
        "facts": [dict(_FACTS_SIMPLE)],
    }))
    proc = _run_agreement_subprocess(repo, cad_single, env_path, state_dir, specs_file,
                                     script_alone_2, "alone")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    alone_pairs = _pairs_for(state_dir, "alone")
    assert len(alone_pairs) == 2
    assert all(p["confirmed_by"] == "agreement" for p in alone_pairs)
    assert len({p["code"] for p in alone_pairs}) == 2
    # the first candidate's original row is superseded by a "promoted" marker row --
    # CandidateIndex folds by id, last write wins.
    alone_candidates_after = _candidates_for(state_dir, "alone")
    by_id = {}
    for r in alone_candidates_after:
        by_id[r["id"]] = r
    assert any(r.get("status") == "promoted" for r in by_id.values())

    # -- 4. a third run must never duplicate the pair (max_pairs_per_spec already met) -
    script_alone_3 = tmp_path / "script_alone_3.json"
    script_alone_3.write_text(json.dumps({"codes": [], "facts": []}))
    proc = _run_agreement_subprocess(repo, cad_single, env_path, state_dir, specs_file,
                                     script_alone_3, "alone")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert len(_pairs_for(state_dir, "alone")) == 2   # unchanged, no duplicates

    # -- 5. "split": two gate-clean candidates that disagree -> nothing confirmed ------
    script_split = tmp_path / "script_split.json"
    script_split.write_text(json.dumps({
        "codes": ["from build123d import *\nresult = Box(10, 10, 10)\n",
                 "from build123d import *\nresult = Cylinder(radius=15, height=20)\n"],
        "facts": [dict(_FACTS_SIMPLE), dict(_FACTS_DIFFERENT)],
    }))
    proc = _run_agreement_subprocess(repo, cad_multi, env_path, state_dir, specs_file,
                                     script_split, "split")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _pairs_for(state_dir, "split") == []
    split_candidates = _candidates_for(state_dir, "split")
    assert len(split_candidates) == 2
    assert all(r["agreement"] == "split" for r in split_candidates)
    split_ledger = _ledger_for(state_dir, "split")
    assert all(r["agreement"] == "split" for r in split_ledger if r.get("ok"))

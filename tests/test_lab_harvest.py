"""Offline tests for lab/harvest.py (Task 3): verdict table, dedup/max-pairs, teacher
promotion + think-unavailable skip, tier-first scheduling, unit-gate refusals, ledger/
pair/status row shapes, pass rates by pass/tier, and real-subprocess signal tests that
drive the ACTUAL lab.harvest.main() (not a look-alike helper -- Task 3 rulings: "that is
how two defects got through a review round on Task 2").

Everything model/GPU-touching in the offline tests is monkeypatched at the same seam
test_fluid_repair_think.py already established for this codebase (fluid_gen._materialize,
engine._ollama / generate_code_raw) -- no network, no GPU, no services. The real-subprocess
tests near the bottom of this file patch ONLY the model call (cad_engine._ollama) and the
arm switch (lab._armwindow._run_arms/_ensure_resident_up), inside a spawned subprocess,
and drive lab.harvest.main() for real.
"""
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
import cad_engine as engine  # noqa: E402
import fluid_gen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own empty state dir and spec bank -- never the real
    lab/state/*.jsonl or the real 434-row bank."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setattr(harvest, "STATE_DIR", d)
    monkeypatch.setattr(harvest, "LEDGER_FILE", d / "ledger.jsonl")
    monkeypatch.setattr(harvest, "PAIRS_FILE", d / "pairs.jsonl")
    monkeypatch.setattr(harvest, "REVIEW_FILE", d / "review.jsonl")
    monkeypatch.setattr(harvest, "PROGRESS_FILE", d / "progress.json")
    monkeypatch.setattr(harvest, "STATUS_FILE", d / "status.json")
    monkeypatch.setattr(harvest, "PAUSED_FILE", d / "paused")
    monkeypatch.setattr(harvest, "BUILDS_DIR", d / "builds")
    monkeypatch.setattr(specbank, "SPECS_FILE", d / "specs.jsonl")
    monkeypatch.setattr(harvest, "_current_arm_alias", lambda: "test-arm")
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
          "temps": [0.2, 0.5, 0.8], "max_pairs_per_spec": 2, "teacher_passes": ["think"]}
    cfg.update(overrides)
    return cfg


def _clean_m(gate_hard=None, gate_spec=None, gate_adv=None, error=None, facts=None) -> dict:
    return {"facts": facts or {"solids": 1}, "instruments": [], "error": error,
           "gate_hard": gate_hard or [], "gate_spec": gate_spec or [],
           "gate_adv": gate_adv or []}


# ---------------------------------------------------------------------------
# Verdict table (_classify)
# ---------------------------------------------------------------------------

def test_classify_good_with_no_reference(tmp_path):
    verdict, band = harvest._classify(_clean_m(), None, tmp_path)
    assert verdict == "good"
    assert band == {}


def test_classify_none_when_build_crashed(tmp_path):
    verdict, band = harvest._classify(_clean_m(error="boom"), None, tmp_path)
    assert verdict == "none"
    assert band == {}


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


def test_row_is_good_matches_classify(tmp_path):
    """_row_is_good (used for pass-rate stats) must agree with _classify's own "good"
    predicate -- one definition of "good", read time and write time."""
    m = _clean_m()
    verdict, _ = harvest._classify(m, None, tmp_path)
    row = harvest._ledger_row("u1", "s1", 2, "student", "0", 0.2, m, {}, {}, None, 1.0)
    assert (verdict == "good") == harvest._row_is_good(row)


# ---------------------------------------------------------------------------
# sample_spec: max pairs per spec, dedup by code hash, prefer lower temperature
# ---------------------------------------------------------------------------

def _patch_generate_sequence(monkeypatch, codes: list[str], mode: str = "student"):
    calls = {"n": 0}

    def fake_generate(spec, notes, temperature):
        code = codes[calls["n"]]
        calls["n"] += 1
        return code, {"system": "SYS", "prompt": f"PROMPT for {code}", "usage": {
            "prompt_tokens": 10, "completion_tokens": 5}}

    attr = "_student_generate" if mode == "student" else "_teacher_generate"
    monkeypatch.setattr(harvest, attr, fake_generate)
    return calls


def test_max_pairs_per_spec_stops_once_enough_good_candidates_are_found(monkeypatch, tmp_path):
    """Temps are cycled ascending, so once max_pairs_per_spec distinct good candidates
    exist, sampling stops early -- a later (higher-temperature) candidate could only ever
    lose the "prefer lower temperature" tie-break, never win it, so there is nothing left
    to gain from spending more GPU time on this spec. Every candidate that WAS sampled is
    still ledgered."""
    _patch_generate_sequence(monkeypatch, ["code-A", "code-B", "code-C"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 2   # candidate index 2 (code-C) never attempted: quota already met
    assert all(r["ok"] for r in ledger)

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 2
    kept_codes = {p["code"] for p in pairs}
    assert kept_codes == {"code-A", "code-B"}   # temp 0.2 and 0.5 kept
    assert progress["s1"]["pairs"] == 2
    assert progress["s1"]["student_attempts"] == 2


def test_candidates_exceed_slots_when_duplicates_force_extra_sampling(monkeypatch, tmp_path):
    """When a duplicate eats a slot's worth of sampling, the loop keeps going up to the
    full `candidates` budget looking for a distinct replacement -- the early-stop above
    only fires once enough DISTINCT good codes exist."""
    _patch_generate_sequence(monkeypatch, ["code-A", "code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 3   # all three candidates attempted: the repeat didn't fill a slot
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert sorted(p["code"] for p in pairs) == ["code-A", "code-B"]
    assert progress["s1"]["pairs"] == 2


def test_dedup_by_code_hash_skips_repeated_candidate(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A", "code-A", "code-B"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=3, temps=[0.2, 0.5, 0.8], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    # code-A (temp 0.2) kept, the repeated code-A (temp 0.5) skipped as a duplicate,
    # code-B (temp 0.8) kept to fill the second slot.
    assert sorted(p["code"] for p in pairs) == ["code-A", "code-B"]
    assert progress["s1"]["pairs"] == 2


def test_spec_already_at_pair_cap_is_never_sampled(monkeypatch, tmp_path):
    calls = _patch_generate_sequence(monkeypatch, ["should-not-be-called"])
    row = _spec_row("s1")
    cfg = _default_cfg(max_pairs_per_spec=2)
    progress = {"s1": {"student_attempts": 5, "teacher_attempts": 0, "pairs": 2}}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)
    assert calls["n"] == 0
    assert harvest._read_jsonl(harvest.LEDGER_FILE) == []


def test_generation_exception_writes_a_failed_ledger_row_and_continues(monkeypatch, tmp_path):
    def fake_generate(spec, notes, temperature):
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(harvest, "_student_generate", fake_generate)
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=2, temps=[0.2, 0.5], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    assert len(ledger) == 2
    assert all(not r["ok"] for r in ledger)
    assert all("codegen failed" in r["error"] for r in ledger)
    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    assert progress["s1"]["student_attempts"] == 2


def test_crash_salvage_produces_two_ledger_rows_and_a_pair_on_recovery(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["broken-code"])
    calls = {"n": 0}

    def fake_materialize(code, build_dir, spec):
        calls["n"] += 1
        if calls["n"] == 1:
            return _clean_m(error="the script failed to run: boom")
        return _clean_m()   # the salvage attempt succeeds

    monkeypatch.setattr(fluid_gen, "_materialize", fake_materialize)
    monkeypatch.setattr(harvest, "diagnose", lambda err: ("generic", "try wrapping it"))
    monkeypatch.setattr(fluid_gen, "_revise_on_repair_rung",
                        lambda spec, code, problem: ("fixed-code", None))

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

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


def test_owner_reference_near_miss_writes_a_fail_pair_with_null_code(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["wrong-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())
    monkeypatch.setattr(harvest, "score_against_reference", lambda *a, **k: {
        "band": "near_miss", "reference": "ref.stl", "chamfer_mm": 3.4, "volume_diff_pct": 8.0})

    row = _spec_row("s1", reference_stl="/tmp/does-not-need-to-exist.stl")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["kind"] == "fail"
    assert p["code"] is None
    assert p["bad_code"] == "wrong-code"
    assert "near-miss" in p["problem"]
    assert progress["s1"]["pairs"] == 0   # a fail pair does not count against the good quota


def test_silver_writes_a_review_row_not_a_pair(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["mostly-right-code"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m(
        gate_spec=["[spec] length is 90mm, spec said 80mm"]))

    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    progress: dict = {}
    harvest.sample_spec(row, cfg, "student", "unit1", progress)

    assert harvest._read_jsonl(harvest.PAIRS_FILE) == []
    review = harvest._read_jsonl(harvest.REVIEW_FILE)
    assert len(review) == 1
    assert review[0]["code"] == "mostly-right-code"
    assert any("80mm" in n for n in review[0]["notes"])


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
# Tier-first scheduling
# ---------------------------------------------------------------------------

def test_order_specs_prefers_tier34_first_when_share_is_low():
    pool = [_spec_row("p1", tier=1), _spec_row("p2", tier=2),
           _spec_row("h1", tier=3), _spec_row("h2", tier=4)]
    ordered = harvest._order_specs(pool, {}, prefer_tier34=True)
    assert [r["id"] for r in ordered[:2]] == ["h1", "h2"] or \
        set(r["id"] for r in ordered[:2]) == {"h1", "h2"}


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
# Unit gate refusals
# ---------------------------------------------------------------------------

def test_unit_gate_paused(monkeypatch):
    harvest.PAUSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    harvest.PAUSED_FILE.write_text("")
    reason = harvest._unit_gate(_default_cfg())
    assert reason is not None and "paused" in reason


def test_unit_gate_outside_night_window(monkeypatch):
    import datetime as dt
    monkeypatch.setattr(harvest, "datetime", dt.datetime)   # no-op, keep real class

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


# ---------------------------------------------------------------------------
# Ledger / pair / status row shapes
# ---------------------------------------------------------------------------

LEDGER_KEYS = {"ts", "unit_id", "spec_id", "tier", "arm", "pass", "candidate", "temperature",
              "ok", "gate_hard", "gate_spec", "gate_adv", "band", "ref", "chamfer_mm",
              "tokens_in", "tokens_out", "seconds", "build_dir", "error"}

GOOD_PAIR_KEYS = {"id", "spec_id", "spec", "tier", "group", "source", "kind", "band", "arm",
                  "temperature", "system", "prompt", "code", "bad_code", "problem", "facts",
                  "verified", "ts", "unit_id"}


def test_ledger_row_has_the_documented_shape():
    row = harvest._ledger_row("u1", "s1", 2, "student", "0", 0.2,
                              _clean_m(), {"band": "match", "reference": "r.stl",
                                          "chamfer_mm": 0.1},
                              {"prompt_tokens": 10, "completion_tokens": 5},
                              "/tmp/somewhere", 12.3)
    assert set(row.keys()) == LEDGER_KEYS
    assert row["ok"] is True
    assert row["band"] == "match"
    assert row["tokens_in"] == 10
    assert row["tokens_out"] == 5


def test_pair_row_has_the_documented_shape(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A"])
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "student", "unit1", {})
    pairs = harvest._read_jsonl(harvest.PAIRS_FILE)
    assert len(pairs) == 1
    assert set(pairs[0].keys()) == GOOD_PAIR_KEYS
    assert pairs[0]["source"] == "student"


def test_pair_row_source_is_teacher_think_for_the_think_pass(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A"], mode="think")
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())
    row = _spec_row("s1")
    cfg = _default_cfg(candidates=1, temps=[0.2], max_pairs_per_spec=2)
    harvest.sample_spec(row, cfg, "think", "unit1", {})
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
    for key in ("updated", "specs", "pairs", "ledger", "think_pass", "budget",
               "nights_completed", "timer_active"):
        assert key in status


def test_pass_rate_by_pass_and_tier(tmp_path):
    rows = [
        harvest._ledger_row("u1", "s1", 3, "student", "0", 0.2, _clean_m(), {}, {}, None, 1.0),
        harvest._ledger_row("u1", "s1", 3, "student", "1", 0.5,
                            _clean_m(gate_hard=["bad"]), {}, {}, None, 1.0),
        harvest._ledger_row("u1", "s2", 1, "think", "0", 0.2, _clean_m(), {}, {}, None, 1.0),
    ]
    for r in rows:
        harvest._append_jsonl(harvest.LEDGER_FILE, r)
    ledger = harvest._read_jsonl(harvest.LEDGER_FILE)
    by_pass = harvest._pass_rate(ledger, lambda r: r["pass"])
    by_tier = harvest._pass_rate(ledger, lambda r: str(r["tier"]))
    assert by_pass == {"student": 0.5, "think": 1.0}
    assert by_tier == {"3": 0.5, "1": 1.0}


# ---------------------------------------------------------------------------
# run_once / run_unit smoke (fully mocked generation+materialize)
# ---------------------------------------------------------------------------

def test_run_once_unknown_spec_id_raises_system_exit(monkeypatch):
    _plant_bank([_spec_row("a")])
    with pytest.raises(SystemExit):
        harvest.run_once("does-not-exist", _default_cfg())


def test_run_once_uses_think_after_two_failed_student_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "think_rung_available", lambda: True)
    _patch_generate_sequence(monkeypatch, ["code-A"], mode="think")
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())
    _plant_bank([_spec_row("s1")])
    progress = {"s1": {"student_attempts": 2, "teacher_attempts": 0, "pairs": 0}}
    harvest._save_progress(progress)
    result = harvest.run_once("s1", _default_cfg(candidates=1, temps=[0.2]))
    assert result["mode"] == "think"


def test_run_unit_stops_within_the_wall_clock_budget(monkeypatch, tmp_path):
    _patch_generate_sequence(monkeypatch, ["code-A"] * 10)
    monkeypatch.setattr(fluid_gen, "_materialize", lambda code, build_dir, spec: _clean_m())
    _plant_bank([_spec_row(f"s{i}") for i in range(5)])

    real_monotonic = time.monotonic
    call_count = {"n": 0}

    def fake_monotonic():
        call_count["n"] += 1
        # Let the deadline check trip after the first spec has been processed.
        return real_monotonic() + (1000 if call_count["n"] > 20 else 0)

    monkeypatch.setattr(harvest.time, "monotonic", fake_monotonic)
    result = harvest.run_unit(_default_cfg(candidates=1, temps=[0.2], unit_minutes=25))
    assert result["specs_processed"] <= 5


# ---------------------------------------------------------------------------
# Real-subprocess tests: drive the ACTUAL lab.harvest.main(), with ONLY the model call
# (cad_engine._ollama) and the arm switch (lab._armwindow._run_arms/_ensure_resident_up)
# patched, inside the spawned subprocess itself. PYTHONUTF8=0 per the Task 3 rulings
# (decoding must stay correct without the interpreter's own UTF-8 mode masking a locale
# bug). A temp CAD_CONFIG_FILE/MAKER_ENV point _pre_arm_marker_paths() (which imports
# scripts/arms.py's own CAD_JSON/ENV_PATH) at a throwaway directory, and a temp
# specs.jsonl stands in for the bank -- nothing under the real ~/.openclaw/ or the real
# lab/state/ is ever touched.
# ---------------------------------------------------------------------------

# {repo!r} = this repo's root; {cad_json!r}/{env_path!r} = temp CAD_CONFIG_FILE/MAKER_ENV;
# {state_dir!r} = temp lab/state equivalent; {specs_file!r} = temp spec bank.
# `scenario` picks what the faked model call (cad_engine._ollama) does:
#   fast_then_slow -- the FIRST call returns a valid box program immediately (so one
#                     clean candidate/ledger row is written), the SECOND call sleeps 5s
#                     (the signal must land inside that sleep, before candidate #2 has
#                     produced anything).
#   use_before_marker -- "use" sleeps 3s and NEVER writes the marker; the signal must
#                         land inside that sleep, before any marker exists, and the
#                         model call must never even be reached.
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
    if calls["n"] == 1:
        return "from build123d import *\\nresult = Box(10, 10, 10)\\n"
    time.sleep(5)
    return "from build123d import *\\nresult = Box(20, 20, 20)\\n"

engine._ollama = fake_ollama

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

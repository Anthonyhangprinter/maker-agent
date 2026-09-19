"""Offline tests for lab/teacher_refs.py (Phase 3 Task 3d1) and lab/specbank.py's
apply_reference_updates.

CPU only, no model calls anywhere: cad_engine.py is imported directly (its module-level
import has no side effects -- verified against the real file), and lab/teacher_refs.py
patches cad_engine._ollama to raise the instant it imports cad_engine, which is asserted
below rather than assumed. Building two or three tiny real build123d parts (a box, a box
with a through-hole, a box built to the wrong size) IS wanted here: run_step/run_inspect
are local subprocesses with no network and no GPU, and the whole point of build_and_gate
is to prove it re-gates real geometry under today's rules, not a rule the caller trusts
blindly.

Every test gets its own tmp_path bank (specbank.SPECS_FILE/REFS_DIR monkeypatched, same
fixture shape as tests/test_lab_specbank.py's own `_isolated_bank`) and its own tmp
cad-sftpairs.jsonl / cad-review-decisions.jsonl -- no test may touch the real
lab/state/ or ~/.openclaw files.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

os.environ.setdefault("PYTHONUTF8", "1")

import cad_engine  # noqa: E402
import harvest_census as hc  # noqa: E402
from lab import specbank  # noqa: E402
from lab import teacher_refs  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    """Own empty specs.jsonl / refs dir per test -- never the real bank."""
    specs_file = tmp_path / "specs.jsonl"
    refs_dir = tmp_path / "refs"
    monkeypatch.setattr(specbank, "SPECS_FILE", specs_file)
    monkeypatch.setattr(specbank, "REFS_DIR", refs_dir)
    yield specs_file


def _write_bank(rows: list[dict]) -> None:
    specbank.SPECS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(specbank.SPECS_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _teacher_bank_row(suite: str, orig_id: str, spec: str, tier: int = 2) -> dict:
    return {"id": f"t:{suite}:{orig_id}", "spec": spec, "tier": tier, "group": "",
           "source": "teacher-suite", "key": hc._key(spec), "added": "2026-09-19T00:00:00+00:00"}


def _pair(spec: str, code: str, source: str = "teacher", teacher_spec_id: str = "P01",
         timestamp: str = "2026-07-29T12:00:00+00:00") -> dict:
    return {"spec": spec, "code": code, "source": source, "kind": "good",
           "teacher_spec_id": teacher_spec_id, "timestamp": timestamp,
           "code_model": "claude-teacher", "verified": {}}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# Real, tiny build123d sources -- CPU-only, no network, matched to spec text below so the
# envelope/gate checks below are exercising the real numbers, not a mock.
BOX_20_CODE = "from build123d import *\nresult = Box(20, 20, 10)\n"
BOX_HOLE_CODE = (
    "from build123d import *\n"
    "with BuildPart() as bp:\n"
    "    Box(20, 20, 10)\n"
    "    Cylinder(radius=3, height=20, mode=Mode.SUBTRACT)\n"
    "result = bp.part\n"
)
# States 55mm long in the spec text below; actually builds 53mm -- the exact "clean but
# wrong" shape strict_envelope_check exists to catch.
BOX_WRONG_SIZE_CODE = "from build123d import *\nresult = Box(53, 20, 10)\n"
CRASH_CODE = "raise RuntimeError('the teacher code itself is broken')\n"


# ---------------------------------------------------------------------------
# Safety: the model-call guard is real, not assumed
# ---------------------------------------------------------------------------

def test_ollama_is_patched_to_raise():
    with pytest.raises(RuntimeError, match="must never call a model"):
        cad_engine._ollama("local:x", "sys", "prompt")


# ---------------------------------------------------------------------------
# Loading + source/review filters
# ---------------------------------------------------------------------------

def test_load_teacher_pairs_filters_by_source(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _write_jsonl(pairs_path, [
        _pair("spec a", BOX_20_CODE, source="teacher"),
        _pair("spec b", BOX_20_CODE, source="teacher-human-accepted"),
        _pair("spec c", BOX_20_CODE, source="teacher-repair"),
        _pair("spec d", BOX_20_CODE, source="gift-fail"),
        {"spec": "spec e", "code": BOX_20_CODE},   # no source at all
    ])
    rows = teacher_refs.load_teacher_pairs(pairs_path)
    assert {r["spec"] for r in rows} == {"spec a", "spec b"}


def test_load_teacher_pairs_missing_file_is_empty(tmp_path):
    assert teacher_refs.load_teacher_pairs(tmp_path / "nope.jsonl") == []


def test_load_review_verdicts_latest_line_wins(tmp_path):
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, [
        {"id": "C02", "verdict": "accept", "note": "first pass"},
        {"id": "C02", "verdict": "reject", "note": "re-reviewed, actually broken"},
    ])
    v = teacher_refs.load_review_verdicts(decisions_path)
    assert v["C02"]["verdict"] == "reject"
    assert v["C02"]["note"] == "re-reviewed, actually broken"


def test_exclude_reviewed_rejects_matches_by_teacher_spec_id_only():
    verdicts = {"C02": {"verdict": "reject"}, "C01": {"verdict": "accept"}}
    rows = [
        _pair("rejected spec", BOX_20_CODE, teacher_spec_id="C02"),
        _pair("accepted spec", BOX_20_CODE, teacher_spec_id="C01"),
        _pair("no verdict at all", BOX_20_CODE, teacher_spec_id="C99"),
    ]
    kept, rejected = teacher_refs.exclude_reviewed_rejects(rows, verdicts)
    assert [r["spec"] for r in rejected] == ["rejected spec"]
    assert {r["spec"] for r in kept} == {"accepted spec", "no verdict at all"}


# ---------------------------------------------------------------------------
# Duplicate resolution + spec-key matching (not short id)
# ---------------------------------------------------------------------------

def test_preferred_ranks_human_accepted_over_plain_teacher():
    rows = [
        _pair("s", BOX_20_CODE, source="teacher", timestamp="2026-07-29T14:00:00+00:00"),
        _pair("s", BOX_20_CODE, source="teacher-human-accepted",
             timestamp="2026-07-29T10:00:00+00:00"),
    ]
    chosen = teacher_refs._preferred(rows)
    assert chosen["source"] == "teacher-human-accepted"


def test_preferred_falls_back_to_newest_timestamp_within_the_same_source():
    rows = [
        _pair("s", BOX_20_CODE, source="teacher", timestamp="2026-07-29T10:00:00+00:00"),
        _pair("s", BOX_20_CODE, source="teacher", timestamp="2026-07-29T14:00:00+00:00"),
        _pair("s", BOX_20_CODE, source="teacher", timestamp="2026-07-29T12:00:00+00:00"),
    ]
    chosen = teacher_refs._preferred(rows)
    assert chosen["timestamp"] == "2026-07-29T14:00:00+00:00"


def test_match_to_bank_uses_spec_key_not_the_colliding_short_id():
    """Two different suites both number their specs "P01" -- teacher_spec_id alone would
    be ambiguous; the bank's own sha1-of-spec-text key is not."""
    spec_a = "a 20x20x10mm block with a 6mm through hole"
    spec_b = "an entirely different 40x30x8mm plate with four 5mm holes"
    bank = [_teacher_bank_row("teacher-pilot", "P01", spec_a),
           _teacher_bank_row("teacher-mech2", "P01", spec_b)]
    bank_by_key = teacher_refs.bank_teacher_rows_by_key(bank)
    pairs = [_pair(spec_a, BOX_HOLE_CODE, teacher_spec_id="P01"),
            _pair(spec_b, BOX_20_CODE, teacher_spec_id="P01")]
    matched, unmatched = teacher_refs.match_to_bank(pairs, bank_by_key)
    assert not unmatched
    assert len(matched) == 2
    got_specs = {v["bank_row"]["spec"] for v in matched.values()}
    assert got_specs == {spec_a, spec_b}


def test_match_to_bank_reports_unmatched_rows():
    bank_by_key = {}
    pairs = [_pair("an orphan spec with no bank row", BOX_20_CODE)]
    matched, unmatched = teacher_refs.match_to_bank(pairs, bank_by_key)
    assert matched == {}
    assert len(unmatched) == 1


def test_suite_of_reads_the_bank_row_id():
    row = _teacher_bank_row("teacher-complex", "C04", "spec text")
    assert teacher_refs._suite_of(row) == "teacher-complex"
    assert teacher_refs._suite_of({"id": "g:specgen:abc"}) == "unknown"


# ---------------------------------------------------------------------------
# build_and_gate: real geometry, real gate, real re-gating under today's rules
# ---------------------------------------------------------------------------

def test_build_and_gate_admits_a_clean_box_with_a_hole():
    spec = "a 20x20x10mm block with a 6mm through hole"
    result = teacher_refs.build_and_gate(spec, BOX_HOLE_CODE)
    try:
        assert result["verdict"] == "admitted", result
        assert result["facts"]["solids"] == 1
        assert result["facts"]["bbox"] == [20.0, 20.0, 10.0]
        assert result["step_path"].exists()
    finally:
        import shutil
        shutil.rmtree(result["work_dir"], ignore_errors=True)


def test_build_and_gate_rejects_a_teacher_part_that_built_the_wrong_size():
    """The exact 'states 55mm, builds 53mm' class of defect: the code runs clean, inspect
    measures fine, but the stated envelope does not match -- strict_envelope_check must
    catch it and the candidate must not be admitted."""
    spec = "a 55x20x10mm block"
    result = teacher_refs.build_and_gate(spec, BOX_WRONG_SIZE_CODE)
    try:
        assert result["verdict"] == "failed-gate", result
        assert "strict_envelope" in result["reason"]
        assert "55" in result["reason"] and "53" in result["reason"]
    finally:
        import shutil
        shutil.rmtree(result["work_dir"], ignore_errors=True)


def test_build_and_gate_reports_a_crash_as_crashed_not_failed_gate():
    result = teacher_refs.build_and_gate("anything", CRASH_CODE)
    try:
        assert result["verdict"] == "crashed"
        assert "run_step" in result["reason"]
    finally:
        import shutil
        shutil.rmtree(result["work_dir"], ignore_errors=True)


def test_reference_facts_from_carries_the_expected_subset():
    facts = {"solids": 1, "faces": 7, "cyl_faces": 1, "cone_faces": 0, "volume": 3717.257,
            "bbox": [20.0, 20.0, 10.0], "bores": [6.0],
            "hole_groups": [{"d": 6.0, "n": 1, "through": 1, "circle_d": 0.0}],
            "through_holes": 1, "blind_holes": 0}
    ref = teacher_refs.reference_facts_from(facts)
    assert ref == {"solids": 1, "volume": 3717.257, "bbox": [20.0, 20.0, 10.0],
                   "faces": 7, "cyl_faces": 1, "cone_faces": 0, "bores": [6.0],
                   "hole_groups": [{"d": 6.0, "n": 1, "through": 1, "circle_d": 0.0}]}


def test_reason_bucket_collapses_specific_messages_to_a_category():
    assert teacher_refs._reason_bucket("strict_envelope: stated 55x20x10mm vs measured "
                                      "53x20x10mm") == "strict_envelope"
    assert teacher_refs._reason_bucket("run_step: boom") == "run_step"
    assert teacher_refs._reason_bucket(None) == ""


# ---------------------------------------------------------------------------
# apply_reference_updates: crash-safe, lock-safe, byte-identical untouched rows
# ---------------------------------------------------------------------------

def test_apply_reference_updates_attaches_the_four_fields():
    row = {"id": "t:teacher-pilot:P01", "spec": "s", "tier": 2, "group": "",
          "source": "teacher-suite", "key": "keyone", "added": "t"}
    _write_bank([row])
    r = specbank.apply_reference_updates({"keyone": {
        "reference_stl": "/refs/keyone.stl", "reference_source": "teacher-claude",
        "reference_facts": {"solids": 1}, "reference_added": "2026-09-19T00:00:00+00:00"}})
    assert r["applied"] == ["keyone"]
    bank = specbank.load_bank()
    assert bank[0]["reference_stl"] == "/refs/keyone.stl"
    assert bank[0]["reference_source"] == "teacher-claude"
    assert bank[0]["reference_facts"] == {"solids": 1}
    assert bank[0]["reference_added"] == "2026-09-19T00:00:00+00:00"


def test_apply_reference_updates_leaves_untouched_rows_byte_identical():
    """A row with unusual formatting (unicode, a long float) must reappear EXACTLY as
    written -- json.dumps(json.loads(line)) is not guaranteed to reproduce it."""
    untouched_raw = ('{"id": "t:x:1", "spec": "a bore \\u00d812mm plate", "tier": 2, '
                     '"group": "", "source": "teacher-suite", '
                     '"key": "untouchedkey", "added": "t", "volume_seen": 1234.123456789012}')
    touched_row = {"id": "t:x:2", "spec": "s2", "tier": 2, "group": "",
                  "source": "teacher-suite", "key": "touchedkey", "added": "t"}
    specbank.SPECS_FILE.parent.mkdir(parents=True, exist_ok=True)
    specbank.SPECS_FILE.write_text(untouched_raw + "\n" + json.dumps(touched_row) + "\n",
                                   encoding="utf-8")
    specbank.apply_reference_updates({"touchedkey": {"reference_stl": "/x.stl"}})
    lines = specbank.SPECS_FILE.read_text(encoding="utf-8").splitlines()
    assert lines[0] == untouched_raw
    assert json.loads(lines[1])["reference_stl"] == "/x.stl"


def test_apply_reference_updates_never_overwrites_an_owner_reference():
    row = {"id": "owner-reference:bearing", "spec": "s", "tier": 3, "group": "owner-reference",
          "source": "owner-reference", "key": "ownerkey", "added": "t",
          "reference_stl": "/owner/bearing.stl"}
    _write_bank([row])
    r = specbank.apply_reference_updates({"ownerkey": {"reference_stl": "/refs/should-not-land.stl",
                                                       "reference_source": "teacher-claude"}})
    assert r["skipped_already_has_reference"] == ["ownerkey"]
    assert r["applied"] == []
    bank = specbank.load_bank()
    assert bank[0]["reference_stl"] == "/owner/bearing.stl"
    assert "reference_source" not in bank[0]


def test_apply_reference_updates_is_idempotent_on_a_second_run():
    row = {"id": "t:x:1", "spec": "s", "tier": 2, "group": "", "source": "teacher-suite",
          "key": "keyone", "added": "t"}
    _write_bank([row])
    update = {"keyone": {"reference_stl": "/refs/keyone.stl", "reference_source": "teacher-claude"}}
    r1 = specbank.apply_reference_updates(update)
    assert r1["applied"] == ["keyone"]
    r2 = specbank.apply_reference_updates(update)
    assert r2["applied"] == []
    assert r2["skipped_already_has_reference"] == ["keyone"]
    bank = specbank.load_bank()
    assert bank[0]["reference_stl"] == "/refs/keyone.stl"


def test_apply_reference_updates_reports_not_found():
    _write_bank([{"id": "t:x:1", "spec": "s", "tier": 2, "group": "",
                 "source": "teacher-suite", "key": "keyone", "added": "t"}])
    r = specbank.apply_reference_updates({"nosuchkey": {"reference_stl": "/x.stl"}})
    assert r["not_found"] == ["nosuchkey"]
    assert r["applied"] == []


def test_apply_reference_updates_on_a_missing_bank_reports_not_found_and_writes_nothing(tmp_path):
    assert not specbank.SPECS_FILE.exists()
    r = specbank.apply_reference_updates({"keyone": {"reference_stl": "/x.stl"}})
    assert r == {"applied": [], "skipped_already_has_reference": [], "not_found": ["keyone"]}
    assert not specbank.SPECS_FILE.exists()


def test_apply_reference_updates_sees_a_row_appended_before_it_started(monkeypatch):
    """Sequential correctness: a concurrent add_items() that completed and released the
    lock before apply_reference_updates ever opens the file must be visible to it (the
    function always re-reads fresh under its own lock, never a stale earlier snapshot)."""
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    row = {"id": "t:x:1", "spec": "s", "tier": 2, "group": "", "source": "teacher-suite",
          "key": "keyone", "added": "t"}
    _write_bank([row])
    specbank.add_items([{"spec": "a brand new specgen spec", "tier": 1}], source="specgen")
    specbank.apply_reference_updates({"keyone": {"reference_stl": "/x.stl"}})
    bank = specbank.load_bank()
    sources = {r["source"] for r in bank}
    assert "specgen" in sources
    assert {r["reference_stl"] for r in bank if r["key"] == "keyone"} == {"/x.stl"}


def test_apply_reference_updates_does_not_lose_a_row_an_appender_writes_while_blocked():
    """The dangerous ordering the module docstring describes: an appender is ALREADY
    blocked, waiting on the SAME flock, when apply_reference_updates finishes and
    releases it. A rename-based (temp file + os.replace) implementation would let that
    appender proceed against the OLD, now file-unreachable inode and lose its row
    forever; writing into the same already-locked fd (what this function actually does)
    must not."""
    row = {"id": "t:x:1", "spec": "s", "tier": 2, "group": "", "source": "teacher-suite",
          "key": "keyone", "added": "t"}
    _write_bank([row])

    orig_flock = specbank.fcntl.flock
    updater_locked = threading.Event()
    appender_waiting = threading.Event()

    def tracking_flock(fd, op):
        r = orig_flock(fd, op)
        if op == specbank.fcntl.LOCK_EX:
            updater_locked.set()
            appender_waiting.wait(timeout=5)
            time.sleep(0.25)  # let the appender's flock() call actually enter the wait queue
        return r

    def run_update():
        specbank.fcntl.flock = tracking_flock
        try:
            specbank.apply_reference_updates({"keyone": {"reference_stl": "/x.stl"}})
        finally:
            specbank.fcntl.flock = orig_flock

    def run_appender():
        updater_locked.wait(timeout=5)
        with open(specbank.SPECS_FILE, "a+") as f:
            appender_waiting.set()
            orig_flock(f, specbank.fcntl.LOCK_EX)   # blocks until the updater unlocks
            f.write(json.dumps({"id": "g:specgen:2", "spec": "a concurrently appended spec",
                               "tier": 1, "group": "", "source": "specgen",
                               "key": "keytwo", "added": "t"}) + "\n")
            f.flush()
            orig_flock(f, specbank.fcntl.LOCK_UN)

    t1 = threading.Thread(target=run_update)
    t2 = threading.Thread(target=run_appender)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()

    bank = specbank.load_bank()
    by_key = {r["key"]: r for r in bank}
    assert by_key["keyone"]["reference_stl"] == "/x.stl"
    assert "keytwo" in by_key, "the concurrently appended row was lost"


# ---------------------------------------------------------------------------
# import_teacher_refs: end to end (dry-run writes nothing; a real small run builds,
# gates, admits, updates the bank, writes the report)
# ---------------------------------------------------------------------------

def _seed_end_to_end(tmp_path):
    spec_good = "a 20x20x10mm block with a 6mm through hole"
    spec_bad_gate = "a 55x20x10mm block"
    spec_rejected = "a spec whose human review says reject"
    spec_crash = "a spec whose teacher code crashes"
    bank = [
        _teacher_bank_row("teacher-pilot", "P01", spec_good, tier=1),
        _teacher_bank_row("teacher-complex", "C02", spec_bad_gate, tier=2),
        _teacher_bank_row("teacher-mech2", "M01", spec_rejected, tier=4),
        _teacher_bank_row("teacher-hard", "H001", spec_crash, tier=3),
    ]
    _write_bank(bank)
    pairs_path = tmp_path / "sftpairs.jsonl"
    _write_jsonl(pairs_path, [
        _pair(spec_good, BOX_HOLE_CODE, teacher_spec_id="P01"),
        _pair(spec_bad_gate, BOX_WRONG_SIZE_CODE, teacher_spec_id="C02"),
        _pair(spec_rejected, BOX_20_CODE, teacher_spec_id="M01"),
        _pair(spec_crash, CRASH_CODE, teacher_spec_id="H001"),
    ])
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, [{"id": "M01", "verdict": "reject", "note": "no good"}])
    return pairs_path, decisions_path


def test_import_teacher_refs_dry_run_writes_nothing(tmp_path):
    pairs_path, decisions_path = _seed_end_to_end(tmp_path)
    report_path = teacher_refs.report_file_path()
    assert not report_path.exists()
    r = teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                         dry_run=True)
    assert r["dry_run"] is True
    assert r["counts"]["teacher_rows_read"]["total"] == 4
    assert r["counts"]["rejected_by_review"]["total"] == 1
    assert r["counts"]["matched_to_bank"] == 3
    assert r["counts"]["would_attempt"] == 3
    assert not report_path.exists()
    assert not specbank.REFS_DIR.exists() or not list(specbank.REFS_DIR.glob("*.stl"))
    bank = specbank.load_bank()
    assert not any(row.get("reference_stl") for row in bank)


def test_import_teacher_refs_real_run_admits_gates_and_reports(tmp_path):
    pairs_path, decisions_path = _seed_end_to_end(tmp_path)
    report = teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                              dry_run=False, workers=2)

    counts = report["counts"]
    assert counts["teacher_rows_read"]["total"] == 4
    assert counts["rejected_by_review"]["total"] == 1
    assert counts["matched_to_bank"] == 3
    assert counts["attempted"] == 3
    assert counts["admitted"]["total"] == 1
    assert counts["failed_gate"]["total"] == 1
    assert counts["crashed"]["total"] == 1

    bank = {r["id"]: r for r in specbank.load_bank()}
    admitted_row = bank["t:teacher-pilot:P01"]
    assert admitted_row["reference_source"] == "teacher-claude"
    assert Path(admitted_row["reference_stl"]).exists()
    assert Path(admitted_row["reference_stl"]).stat().st_size > 0
    assert admitted_row["reference_facts"]["solids"] == 1
    assert "reference_added" in admitted_row

    assert "reference_stl" not in bank["t:teacher-complex:C02"]
    assert "reference_stl" not in bank["t:teacher-hard:H001"]
    assert "reference_stl" not in bank["t:teacher-mech2:M01"]

    non_admitted_ids = {r["id"]: r for r in report["non_admitted"]}
    assert non_admitted_ids["t:teacher-complex:C02"]["verdict"] == "failed-gate"
    assert non_admitted_ids["t:teacher-hard:H001"]["verdict"] == "crashed"
    assert "t:teacher-mech2:M01" not in non_admitted_ids  # excluded before attempting

    report_path = teacher_refs.report_file_path()
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["counts"]["admitted"]["total"] == 1


def test_import_teacher_refs_second_run_is_idempotent(tmp_path):
    pairs_path, decisions_path = _seed_end_to_end(tmp_path)
    teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                     dry_run=False, workers=2)
    stl_path_1 = next(r["reference_stl"] for r in specbank.load_bank()
                     if r.get("reference_stl"))
    report2 = teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                               dry_run=False, workers=2)
    # Second run: the one admitted spec already has a reference, so nothing is attempted.
    assert report2["counts"]["already_has_reference"] == 1
    assert report2["counts"]["attempted"] == 2  # the two still-unreferenced (bad-gate, crash)
    bank2 = specbank.load_bank()
    stl_path_2 = next(r["reference_stl"] for r in bank2 if r.get("reference_stl"))
    assert stl_path_1 == stl_path_2


def test_import_teacher_refs_respects_limit(tmp_path):
    pairs_path, decisions_path = _seed_end_to_end(tmp_path)
    report = teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                              dry_run=False, workers=2, limit=1)
    assert report["counts"]["attempted"] == 1


def test_import_teacher_refs_never_overwrites_an_owner_reference(tmp_path):
    spec_good = "a 20x20x10mm block with a 6mm through hole"
    row = _teacher_bank_row("teacher-pilot", "P01", spec_good, tier=1)
    row["reference_stl"] = "/owner/already-there.stl"
    row["reference_source"] = "owner-reference"
    _write_bank([row])
    pairs_path = tmp_path / "sftpairs.jsonl"
    _write_jsonl(pairs_path, [_pair(spec_good, BOX_HOLE_CODE, teacher_spec_id="P01")])
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, [])

    report = teacher_refs.import_teacher_refs(pairs_path=pairs_path, decisions_path=decisions_path,
                                              dry_run=False, workers=1)
    assert report["counts"]["already_has_reference"] == 1
    assert report["counts"]["attempted"] == 0
    bank = specbank.load_bank()
    assert bank[0]["reference_stl"] == "/owner/already-there.stl"
    assert bank[0]["reference_source"] == "owner-reference"

"""Offline tests for lab/specbank.py and lab/specgen.py's pure/stubbed pieces.

specbank.py is imported normally (`from lab import specbank`) -- it only pulls in
harvest_census and lab.data (jinja2), both lightweight, same as production callers
(scripts/run_card.py, lab/ship.py already `from lab.data import ...` this way).
specgen.py additionally imports cad_engine, which existing tests (test_maker_config.py)
already import directly in the plain test suite, so no stub/heavy-dep gymnastics are
needed here either -- engine._ollama itself is monkeypatched per test.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from lab import specbank  # noqa: E402
from lab import specgen  # noqa: E402
import cad_engine  # noqa: E402
import harvest_census as hc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    """Every test gets its own empty specs.jsonl / refs dir / specgen log -- never the
    real bank or the real lab/state/specgen_log.jsonl.

    Also cleans up CAD_KEEP_MAKER unconditionally after every test: specgen.main()
    calls os.environ.setdefault("CAD_KEEP_MAKER", "1") directly (a real env mutation),
    and monkeypatch.delenv(..., raising=False) is a NO-OP with nothing recorded to undo
    when the key is already absent (verified against _pytest.monkeypatch's own
    delitem/undo source) -- so it would only clean up a leak retroactively, at the
    NEXT test's setup, and never across a file boundary (test_maker_config.py doesn't
    use this fixture at all). A plain `os.environ.pop` after `yield` has no such gap."""
    specs_file = tmp_path / "specs.jsonl"
    refs_dir = tmp_path / "refs"
    monkeypatch.setattr(specbank, "SPECS_FILE", specs_file)
    monkeypatch.setattr(specbank, "REFS_DIR", refs_dir)
    monkeypatch.setattr(specgen, "SPECGEN_LOG", tmp_path / "specgen_log.jsonl")
    yield specs_file
    os.environ.pop("CAD_KEEP_MAKER", None)


# ---------------------------------------------------------------------------
# import-teacher against the REAL benchmarks/teacher-*/specs.json files
# ---------------------------------------------------------------------------

def test_import_teacher_counts_414_from_the_real_files():
    r = specbank.import_teacher()
    assert r["accepted"] == 414
    assert r["refused"] == 0
    assert sum(v["found"] for v in r["per_suite"].values()) == 414
    assert r["per_suite"]["teacher-pilot"]["found"] == 40
    assert r["per_suite"]["teacher-complex"]["found"] == 12
    assert r["per_suite"]["teacher-hard"]["found"] == 141
    assert r["per_suite"]["teacher-mech2"]["found"] == 25
    assert r["per_suite"]["teacher-batch2"]["found"] == 196


def test_import_teacher_writes_tagged_rows_to_the_bank(_isolated_bank):
    specbank.import_teacher()
    bank = specbank.load_bank()
    assert len(bank) == 414
    row = bank[0]
    assert set(row.keys()) >= {"id", "spec", "tier", "group", "source", "key", "added"}
    assert row["source"] == "teacher-suite"
    assert row["id"].startswith("t:teacher-pilot:")


def test_import_teacher_dry_run_reports_but_does_not_write(_isolated_bank):
    r = specbank.import_teacher(dry_run=True)
    assert r["accepted"] == 414
    assert specbank.load_bank() == []
    assert not _isolated_bank.exists()


def test_import_teacher_is_idempotent_second_run_all_duplicates(_isolated_bank):
    specbank.import_teacher()
    r2 = specbank.import_teacher()
    assert r2["accepted"] == 0
    assert r2["refused"] == 414
    reasons = {d["reason"] for d in r2["refused_detail"]}
    assert reasons == {"duplicate-in-bank"}
    assert len(specbank.load_bank()) == 414   # nothing appended twice


# ---------------------------------------------------------------------------
# refusal_reason / add_items: contamination + duplicate
# ---------------------------------------------------------------------------

def test_add_items_refuses_a_planted_card_suite_spec(monkeypatch):
    planted = "a 100x60x30mm enclosure with 2mm walls"
    monkeypatch.setattr(specbank, "contamination_sets",
                        lambda: ({hc._key(planted)}, set()))
    r = specbank.add_items(
        [{"spec": planted, "tier": 2, "group": "enclosure"},
         {"spec": "a clean 20mm cube with a 5mm centre hole", "tier": 1, "group": "plate"}],
        source="specgen")
    assert r["accepted"] == 1
    assert len(r["refused"]) == 1
    assert r["refused"][0]["reason"] == "suite-exact-match"
    assert r["refused"][0]["spec"] == planted


def test_add_items_refuses_a_unique_suite_slug_match(monkeypatch):
    spec = "a bracket with a 40mm slot for mounting hardware here, then some more words"
    slug = hc._slug(spec, 40)
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), {slug}))
    r = specbank.add_items([{"spec": spec, "tier": 2, "group": "bracket"}], source="specgen")
    assert r["accepted"] == 0
    assert r["refused"][0]["reason"] == "suite-unique-slug-match"


def test_add_items_refuses_a_duplicate_already_in_the_bank(monkeypatch):
    # Fix round 1, finding 6's practical regression test: two SEQUENTIAL add_items()
    # calls (not concurrent, but each doing its own full read-decide-append under
    # _locked_bank's single flock) must still see each other's writes -- the second
    # call's read happens after the first call's lock has been released and its row
    # flushed, so the duplicate is caught exactly as it would be under real contention.
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    spec = "a stepped shaft 12mm to 8mm with a 3mm keyway 20mm long"
    r1 = specbank.add_items([{"spec": spec, "tier": 2, "group": "shaft"}], source="specgen")
    assert r1["accepted"] == 1
    r2 = specbank.add_items([{"spec": spec, "tier": 2, "group": "shaft"}], source="specgen")
    assert r2["accepted"] == 0
    assert r2["refused"][0]["reason"] == "duplicate-in-bank"


def test_add_items_holds_one_lock_spanning_read_and_write(monkeypatch):
    """The read-decide-append window must run under a SINGLE flock acquisition, not a
    separate one for the read and another for the final write (which would reopen the
    exact race fix round 1's finding 6 closes)."""
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    lock_ops = []
    real_flock = specbank.fcntl.flock

    def counting_flock(fd, op):
        lock_ops.append(op)
        return real_flock(fd, op)

    monkeypatch.setattr(specbank.fcntl, "flock", counting_flock)
    specbank.add_items([{"spec": "a spec for the lock-shape test", "tier": 1}],
                       source="specgen")
    assert lock_ops.count(specbank.fcntl.LOCK_EX) == 1
    assert lock_ops.count(specbank.fcntl.LOCK_UN) == 1


def test_add_items_refuses_a_duplicate_within_the_same_batch(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    spec = "a flange 60mm outside diameter with a 20mm bore and four 6mm bolt holes"
    r = specbank.add_items(
        [{"spec": spec, "tier": 2, "group": "flange"},
         {"spec": spec, "tier": 2, "group": "flange"}],
        source="specgen")
    assert r["accepted"] == 1
    assert len(r["refused"]) == 1
    assert r["refused"][0]["reason"] == "duplicate-in-bank"


def test_add_items_refuses_empty_spec(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    r = specbank.add_items([{"spec": "  ", "tier": 2}], source="specgen")
    assert r["accepted"] == 0
    assert r["refused"][0]["reason"] == "empty-spec"


def test_add_items_default_id_shape(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    spec = "a small pattern plate with a 3x3 grid of 4mm holes on 15mm centres"
    r = specbank.add_items([{"spec": spec, "tier": 2, "group": "pattern"}], source="specgen")
    assert r["accepted"] == 1
    row = specbank.load_bank()[0]
    assert row["id"] == f"g:specgen:{hc._key(spec)[:16]}"
    assert row["group"] == "pattern"
    assert row["source"] == "specgen"


# ---------------------------------------------------------------------------
# stats: tier / source / tier34-share maths
# ---------------------------------------------------------------------------

def test_stats_tier_and_source_maths(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    rows = [
        {"spec": "spec one tier one", "tier": 1, "group": "plate"},
        {"spec": "spec two tier two", "tier": 2, "group": "bracket"},
        {"spec": "spec three tier three", "tier": 3, "group": "surface"},
        {"spec": "spec four tier four", "tier": 4, "group": "mechanism"},
    ]
    specbank.add_items(rows, source="teacher-suite")
    st = specbank.stats()
    assert st["total"] == 4
    assert st["by_tier"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    assert st["by_source"] == {"teacher-suite": 4}
    assert st["tier34_share"] == 0.5


def test_stats_on_empty_bank_does_not_divide_by_zero():
    st = specbank.stats()
    assert st == {"total": 0, "by_tier": {}, "by_source": {}, "by_group": {},
                  "tier34_share": 0.0}


# ---------------------------------------------------------------------------
# import-references: the empty-folder case (the only one on disk today) plus the
# populated-folder shapes, all against a temp references root.
# ---------------------------------------------------------------------------

def test_import_references_zero_folders_reports_zero_and_is_not_an_error(tmp_path):
    refs_dir = tmp_path / "references"   # never created -- mirrors "not on this machine"
    r = specbank.import_references(refs_dir)
    assert r == {"imported": 0, "refused": 0, "skipped": 0, "folders_found": 0}


def test_import_references_empty_dir_with_only_a_readme(tmp_path):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    (refs_dir / "README.md").write_text("put reference folders here")
    r = specbank.import_references(refs_dir)
    assert r == {"imported": 0, "refused": 0, "skipped": 0, "folders_found": 0}


def test_import_references_skips_a_folder_missing_the_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    refs_dir = tmp_path / "references"
    part = refs_dir / "widget"
    part.mkdir(parents=True)
    (part / "spec.txt").write_text("a 50mm widget")
    r = specbank.import_references(refs_dir)
    assert r["folders_found"] == 1
    assert r["imported"] == 0
    assert r["skipped"] == 1


def test_import_references_reads_tier_prefix_and_copies_stl(tmp_path, monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    refs_dir = tmp_path / "references"
    part = refs_dir / "gizmo"
    part.mkdir(parents=True)
    (part / "spec.txt").write_text("tier: 4\na gizmo assembly with two mating parts")
    (part / "model.stl").write_bytes(b"fake stl bytes")
    r = specbank.import_references(refs_dir)
    assert r == {"imported": 1, "refused": 0, "skipped": 0, "folders_found": 1}
    row = specbank.load_bank()[0]
    assert row["tier"] == 4
    assert row["spec"] == "a gizmo assembly with two mating parts"
    assert row["source"] == "owner-reference"
    assert Path(row["reference_stl"]).read_bytes() == b"fake stl bytes"


def test_import_references_defaults_tier_3_without_a_tier_line(tmp_path, monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    refs_dir = tmp_path / "references"
    part = refs_dir / "plain"
    part.mkdir(parents=True)
    (part / "spec.txt").write_text("a plain reference part, no tier line")
    (part / "model.stl").write_bytes(b"x")
    r = specbank.import_references(refs_dir)
    assert r["imported"] == 1
    assert specbank.load_bank()[0]["tier"] == 3


def test_import_references_refuses_a_contaminated_reference_spec(tmp_path, monkeypatch):
    spec = "a contaminated reference spec"
    monkeypatch.setattr(specbank, "contamination_sets", lambda: ({hc._key(spec)}, set()))
    refs_dir = tmp_path / "references"
    part = refs_dir / "bad"
    part.mkdir(parents=True)
    (part / "spec.txt").write_text(spec)
    (part / "model.stl").write_bytes(b"x")
    r = specbank.import_references(refs_dir)
    assert r == {"imported": 0, "refused": 1, "skipped": 0, "folders_found": 1}
    assert specbank.load_bank() == []


def test_import_references_dry_run_does_not_write_stl_or_bank(tmp_path, monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    refs_dir = tmp_path / "references"
    part = refs_dir / "gizmo"
    part.mkdir(parents=True)
    (part / "spec.txt").write_text("a dry-run gizmo part")
    (part / "model.stl").write_bytes(b"x")
    r = specbank.import_references(refs_dir, dry_run=True)
    assert r["imported"] == 1
    assert specbank.load_bank() == []
    assert not specbank.REFS_DIR.exists()


# ---------------------------------------------------------------------------
# refusal_reason: the ordering/precedence contract add_items/import_teacher rely on
# ---------------------------------------------------------------------------

def test_refusal_reason_checks_bank_duplicate_before_suite_sets():
    key = "deadbeef"
    reason = specbank.refusal_reason(key, "irrelevant spec text",
                                     suite_keys={key}, suite_slugs=set(),
                                     bank_keys={key})
    assert reason == "duplicate-in-bank"


def test_refusal_reason_none_when_clean():
    assert specbank.refusal_reason("k", "a clean unseen spec", set(), set(), set()) is None


# ---------------------------------------------------------------------------
# specgen._normalize_items / gen_family: JSON parse + one repair-retry + robust
# item-shape handling (engine stubbed)
# ---------------------------------------------------------------------------

def test_specgen_parses_a_clean_json_array(monkeypatch):
    calls = []

    def fake_ollama(model, system, prompt, **kw):
        calls.append((model, kw.get("temperature"), kw.get("no_think")))
        return json.dumps([{"tier": 2, "spec": "a clean 30mm plate with two 5mm holes"}])

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    items, skipped = specgen.gen_family("plate", 1, "tier 1-2 guidance", [])
    assert items == [{"tier": 2, "group": "plate", "spec": "a clean 30mm plate with two 5mm holes"}]
    assert skipped == {}
    assert len(calls) == 1
    model, temperature, no_think = calls[0]
    assert model == cad_engine.CODE_MODEL_STRONG
    assert temperature == 0.8
    assert no_think is False   # thinking stays ON for spec writing


def test_specgen_strips_markdown_fences():
    raw = "```json\n[{\"tier\": 1, \"spec\": \"a fenced spec\"}]\n```"
    assert specgen._parse_json_array(raw) == [{"tier": 1, "spec": "a fenced spec"}]


def test_specgen_retries_once_on_malformed_json_then_succeeds(monkeypatch):
    calls = []

    def fake_ollama(model, system, prompt, **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json at all, sorry"
        return json.dumps([{"tier": 3, "spec": "a repaired spec after retry"}])

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    items, skipped = specgen.gen_family("surface", 1, "tier 3 guidance", [])
    assert items == [{"tier": 3, "group": "surface", "spec": "a repaired spec after retry"}]
    assert skipped == {}
    assert len(calls) == 2
    assert calls[1].endswith(specgen._REPAIR_SUFFIX)


def test_specgen_raises_when_the_repair_retry_also_fails(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw: "still not json")
    with pytest.raises(ValueError):
        specgen.gen_family("plate", 1, "guidance", [])


def test_specgen_defaults_missing_tier_from_group_table(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps([{"spec": "no tier given here"}]))
    items, skipped = specgen.gen_family("mechanism", 1, "guidance", [])
    assert items[0]["tier"] == 4   # _TIER_BY_GROUP["mechanism"]
    assert skipped == {}


def test_specgen_drops_items_without_a_spec(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps([{"tier": 2}, {"tier": 2, "spec": "kept spec here"}]))
    items, skipped = specgen.gen_family("plate", 2, "guidance", [])
    assert len(items) == 1
    assert items[0]["spec"] == "kept spec here"
    assert skipped == {"missing-spec": 1}


def test_normalize_items_skips_non_dict_items_with_counted_reason():
    items, skipped = specgen._normalize_items(
        ["a bare string", {"tier": 1, "spec": "kept"}], "plate")
    assert items == [{"tier": 1, "group": "plate", "spec": "kept"}]
    assert skipped == {"not-a-dict": 1}


def test_normalize_items_skips_items_with_uncoercible_tier():
    items, skipped = specgen._normalize_items(
        [{"tier": "not-a-number", "spec": "bad tier spec"},
         {"tier": 2, "spec": "good spec"}], "plate")
    assert items == [{"tier": 2, "group": "plate", "spec": "good spec"}]
    assert skipped == {"bad-tier": 1}


def test_normalize_items_accepts_a_missing_tier_key_as_default_not_a_skip():
    # A PRESENT-but-uncoercible tier is a skip; an ABSENT tier key just defaults.
    items, skipped = specgen._normalize_items([{"spec": "no tier key at all"}], "plate")
    assert items == [{"tier": 1, "group": "plate", "spec": "no tier key at all"}]
    assert skipped == {}


def test_gen_family_never_raises_on_an_all_string_json_array(monkeypatch):
    """Fix round 1, finding 5: a valid JSON array whose items are not dicts (plausible
    temperature-0.8 drift) must never reach an unhandled AttributeError. Here it is
    skipped cleanly (every item counted as not-a-dict) rather than forcing a repair
    retry -- only one _ollama call is made."""
    calls = []

    def fake_ollama(model, system, prompt, **kw):
        calls.append(prompt)
        return json.dumps(["just a string", "another string"])

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    items, skipped = specgen.gen_family("plate", 2, "guidance", [])
    assert items == []
    assert skipped == {"not-a-dict": 2}
    assert len(calls) == 1


def test_seed_specs_falls_back_to_teacher_suite_then_to_empty(monkeypatch):
    monkeypatch.setattr(specbank, "load_bank", lambda: [])
    assert specgen._seed_specs("plate") == []

    monkeypatch.setattr(specbank, "load_bank", lambda: [
        {"spec": "a teacher spec", "group": "bracket", "source": "teacher-suite"}])
    seeds = specgen._seed_specs("plate")   # no "plate" rows -> falls back to teacher-suite
    assert seeds == ["a teacher spec"]


# ---------------------------------------------------------------------------
# run_batch: shape, per-batch log row, token usage capture, failure path
# ---------------------------------------------------------------------------

def test_run_batch_reports_shape(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    monkeypatch.setattr(specgen, "_seed_specs", lambda group, n=5: [])
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps([{"tier": 1, "spec": "a run_batch smoke spec"}]))
    r = specgen.run_batch("plate", 1)
    assert r["accepted"] == 1
    assert r["group"] == "plate"
    assert r["requested"] == 1
    assert r["generated"] == 1
    assert "seconds" in r
    assert "prompt_tokens" in r
    assert "completion_tokens" in r


def test_run_batch_unknown_group_raises():
    with pytest.raises(SystemExit):
        specgen.run_batch("not-a-real-group", 1)


def test_run_batch_captures_token_usage_and_logs_one_row(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    monkeypatch.setattr(specgen, "_seed_specs", lambda group, n=5: [])

    def fake_ollama(model, system, prompt, **kw):
        cad_engine._LAST_USAGE = {"prompt_tokens": 40, "completion_tokens": 15}
        cad_engine._USAGE_TOTAL["prompt_tokens"] += 40
        cad_engine._USAGE_TOTAL["completion_tokens"] += 15
        cad_engine._USAGE_TOTAL["calls"] += 1
        return json.dumps([{"tier": 1, "spec": "a usage-tracked spec"}])

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    cad_engine.reset_usage()
    r = specgen.run_batch("plate", 1)
    assert r["prompt_tokens"] == 40
    assert r["completion_tokens"] == 15

    rows = [json.loads(line) for line in specgen.SPECGEN_LOG.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) >= {"ts", "group", "tier", "requested", "generated", "accepted",
                               "refused", "reasons", "seconds", "prompt_tokens",
                               "completion_tokens"}
    assert row["group"] == "plate"
    assert row["accepted"] == 1
    assert row["prompt_tokens"] == 40
    assert row["completion_tokens"] == 15
    cad_engine.reset_usage()


def test_run_batch_logs_reasons_from_refusals_and_shape_skips(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets",
                        lambda: ({hc._key("a duplicate spec on the suite")}, set()))
    monkeypatch.setattr(specgen, "_seed_specs", lambda group, n=5: [])
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps(["a bare string",
                                   {"tier": 1, "spec": "a duplicate spec on the suite"},
                                   {"tier": 1, "spec": "a clean kept spec"}]))
    r = specgen.run_batch("plate", 3)
    assert r["accepted"] == 1
    rows = [json.loads(line) for line in specgen.SPECGEN_LOG.read_text().splitlines()]
    assert rows[0]["reasons"] == {"not-a-dict": 1, "suite-exact-match": 1}


def test_run_batch_logs_and_reraises_on_a_model_call_failure(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    monkeypatch.setattr(specgen, "_seed_specs", lambda group, n=5: [])

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(specgen.engine, "_ollama", boom)
    with pytest.raises(OSError, match="connection refused"):
        specgen.run_batch("plate", 1)

    rows = [json.loads(line) for line in specgen.SPECGEN_LOG.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["accepted"] == 0
    assert rows[0]["generated"] == 0
    assert rows[0]["error"] == "connection refused"


# ---------------------------------------------------------------------------
# run_total: circuit breakers (fix round 1, findings 2/3) and defaults
# ---------------------------------------------------------------------------

def test_run_total_default_max_batches_is_200():
    assert specgen.MAX_BATCHES_DEFAULT == 200


def test_run_total_reraises_specgen_aborted_immediately_not_as_a_call_failure(monkeypatch):
    """Fix round 3, finding 1 (CRITICAL): SpecgenAborted IS an Exception subclass, so
    the generic `except Exception` used to catch it too, count it as one ordinary call
    failure, and keep looping -- meaning a single SIGTERM/SIGHUP landing mid-batch did
    NOT stop the run. It must propagate on the very first occurrence."""
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    calls = {"n": 0}

    def run_batch_signal_then_ok(group, n, dry_run=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise specgen.SpecgenAborted("terminated by signal 15 (SIGTERM)")
        return {"accepted": 1, "refused": [], "group": group, "requested": n,
                "generated": 1, "seconds": 0.1, "prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(specgen, "run_batch", run_batch_signal_then_ok)
    with pytest.raises(specgen.SpecgenAborted, match="terminated by signal"):
        specgen.run_total(total=100000, target_tier34=0.0, batch_size=5, max_batches=50)
    assert calls["n"] == 1   # never reached a second batch after the signal


def test_gen_family_reraises_specgen_aborted_from_parse_without_a_repair_retry(monkeypatch):
    """Fix round 3, finding 1 audit: the JSON-repair retry's bare `except Exception`
    must not treat a signal-triggered SpecgenAborted raised during parsing as an
    ordinary parse failure and burn a second model call on the repair prompt."""
    calls = []

    def fake_ollama(model, system, prompt, **kw):
        calls.append(prompt)
        return "irrelevant, _parse_json_array is stubbed below to raise directly"

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    monkeypatch.setattr(specgen, "_parse_json_array",
                        lambda raw: (_ for _ in ()).throw(
                            specgen.SpecgenAborted("terminated by signal 15 (SIGTERM)")))
    with pytest.raises(specgen.SpecgenAborted):
        specgen.gen_family("plate", 1, "guidance", [])
    assert len(calls) == 1   # no repair-retry call was made


# ---------------------------------------------------------------------------
# _pre_arm_marker_paths / _pre_arm_marker_exists / _ensure_resident_up
# (fix round 3, finding 2)
# ---------------------------------------------------------------------------

def test_pre_arm_marker_paths_delegate_to_the_real_arms_module():
    """Imported directly from scripts/arms.py, not re-derived, so this can never drift
    from what the real arms.py subprocess itself checks."""
    import arms as arms_cli
    expected = arms_cli._pre_arm_paths(arms_cli.CAD_JSON, arms_cli.ENV_PATH)
    assert specgen._pre_arm_marker_paths() == expected


def test_pre_arm_marker_exists_false_when_neither_file_present(monkeypatch, tmp_path):
    import arms as arms_cli
    monkeypatch.setattr(arms_cli, "CAD_JSON", tmp_path / "cad.json")
    monkeypatch.setattr(arms_cli, "ENV_PATH", tmp_path / "maker.env")
    assert specgen._pre_arm_marker_exists() is False


def test_pre_arm_marker_exists_true_when_cad_json_marker_present(monkeypatch, tmp_path):
    import arms as arms_cli
    monkeypatch.setattr(arms_cli, "CAD_JSON", tmp_path / "cad.json")
    monkeypatch.setattr(arms_cli, "ENV_PATH", tmp_path / "maker.env")
    (tmp_path / "cad.json.pre-arm").write_text("{}")
    assert specgen._pre_arm_marker_exists() is True


def test_pre_arm_marker_exists_true_when_only_the_maker_env_marker_is_present(monkeypatch, tmp_path):
    """_save_pre_arm writes both markers back to back, not atomically as a pair -- "at
    least one" (per the fix instructions) is the correct test, not "both"."""
    import arms as arms_cli
    monkeypatch.setattr(arms_cli, "CAD_JSON", tmp_path / "cad.json")
    monkeypatch.setattr(arms_cli, "ENV_PATH", tmp_path / "maker.env")
    (tmp_path / "maker.env.pre-arm").write_text('{"existed": false, "content": null}')
    assert specgen._pre_arm_marker_exists() is True


def test_ensure_resident_up_starts_qwen38_server_and_touches_nothing_else(monkeypatch):
    calls = []
    monkeypatch.setattr(specgen.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    specgen._ensure_resident_up()
    assert calls == [["systemctl", "--user", "start", "qwen38-server"]]


def test_run_total_aborts_after_3_consecutive_call_failures(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    calls = {"n": 0}

    def failing_run_batch(group, n, dry_run=False):
        calls["n"] += 1
        raise RuntimeError("connection refused")

    monkeypatch.setattr(specgen, "run_batch", failing_run_batch)
    with pytest.raises(specgen.SpecgenAborted, match="circuit breaker"):
        specgen.run_total(total=1000, target_tier34=0.0, batch_size=5, max_batches=50)
    assert calls["n"] == specgen.MAX_CONSECUTIVE_CALL_FAILURES


def test_run_total_aborts_after_8_consecutive_no_progress_batches(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    calls = {"n": 0}

    def zero_accept_run_batch(group, n, dry_run=False):
        calls["n"] += 1
        return {"accepted": 0, "refused": [{"spec": "x", "reason": "duplicate-in-bank"}],
                "group": group, "requested": n, "generated": 1, "seconds": 0.1,
                "prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(specgen, "run_batch", zero_accept_run_batch)
    with pytest.raises(specgen.SpecgenAborted, match="no-progress guard"):
        specgen.run_total(total=100000, target_tier34=0.0, batch_size=5, max_batches=50)
    assert calls["n"] == specgen.MAX_CONSECUTIVE_NO_PROGRESS


def test_run_total_call_failure_counter_resets_on_a_healthy_batch(monkeypatch):
    """2 failures, then a healthy batch, then 2 more failures must NOT trip the
    3-consecutive breaker -- the counter is consecutive, not cumulative."""
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    outcomes = ["fail", "fail", "ok", "fail", "fail"]

    def run_batch_seq(group, n, dry_run=False):
        outcome = outcomes.pop(0)
        if outcome == "fail":
            raise RuntimeError("timeout")
        return {"accepted": 1, "refused": [], "group": group, "requested": n,
                "generated": 1, "seconds": 0.1, "prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(specgen, "run_batch", run_batch_seq)
    result = specgen.run_total(total=100000, target_tier34=0.0, batch_size=5, max_batches=5)
    assert result["batches"] == 5   # ran to completion, never tripped the breaker


def test_run_total_respects_the_wall_clock_cap(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    real_time = specgen.time.time
    call_count = {"n": 0}

    def fake_time():
        call_count["n"] += 1
        # Jump an hour forward on every call after the first (the `start` snapshot),
        # so a tiny max_hours trips on the loop's very first elapsed-time check.
        return real_time() + call_count["n"] * 3600

    monkeypatch.setattr(specgen.time, "time", fake_time)
    monkeypatch.setattr(specgen, "run_batch",
                        lambda group, n, dry_run=False: pytest.fail("must not run a batch"))
    with pytest.raises(specgen.SpecgenAborted, match="wall-clock cap"):
        specgen.run_total(total=100000, target_tier34=0.0, batch_size=5, max_hours=0.01)


def test_run_total_usage_total_sums_across_batches(monkeypatch):
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    batches = {"n": 0}

    def run_batch_ok(group, n, dry_run=False):
        batches["n"] += 1
        return {"accepted": 1, "refused": [], "group": group, "requested": n,
                "generated": 1, "seconds": 0.1, "prompt_tokens": 10, "completion_tokens": 4}

    monkeypatch.setattr(specgen, "run_batch", run_batch_ok)
    result = specgen.run_total(total=100000, target_tier34=0.0, batch_size=5, max_batches=3)
    assert result["batches"] == 3
    assert result["usage_total"] == {"prompt_tokens": 30, "completion_tokens": 12}


# ---------------------------------------------------------------------------
# _default_arm / _run_arms / main(): the arms.py use-then-restore bookend
# (fix round 1, finding 3 -- replaces the old hardcoded LOCK_ALIAS/_relock_maker)
# ---------------------------------------------------------------------------

def test_default_arm_reads_the_currently_configured_maker_block(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_load_config", lambda: {
        "cad": {"maker": {"enabled": True, "arm": "devstral-small-2",
                          "alias": "devstral-small-2"}}})
    assert specgen._default_arm() == "devstral-small-2"


def test_default_arm_falls_back_when_no_maker_block_is_configured(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_load_config", lambda: {})
    assert specgen._default_arm() == specgen.DEFAULT_ARM_FALLBACK


# NOTE (fix round 3): main()'s finally now gates the arms.py restore call on whether a
# real .pre-arm marker file exists on disk (see _pre_arm_marker_exists()), not on a flag
# computed earlier in the function -- a signal can interrupt `use` before any such flag
# would even be set. These tests stub `_run_arms` entirely (no real arms.py subprocess),
# so they also stub `_pre_arm_marker_exists`/`_ensure_resident_up` explicitly rather than
# depending on whatever real marker files do or do not happen to exist on the machine
# running the suite. The REAL marker-existence behaviour, end to end with a real signal
# landing at a real, controlled instant, is exercised in
# tests/test_lab_specgen_signals.py's real-subprocess tests instead.

def test_main_bookend_calls_use_then_restore_and_nothing_else(monkeypatch):
    arms_calls = []

    def fake_run_arms(*args):
        arms_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(specgen, "_run_arms", fake_run_arms)
    monkeypatch.setattr(specgen, "_default_arm", lambda: "gemma-4-31b")
    monkeypatch.setattr(specgen, "_pre_arm_marker_exists", lambda: True)
    monkeypatch.setattr(specgen, "run_batch",
                        lambda group, n, dry_run=False: {
                            "accepted": 1, "refused": [], "group": group, "requested": n,
                            "generated": 1, "seconds": 0.1, "prompt_tokens": 5,
                            "completion_tokens": 5})
    monkeypatch.setattr(sys, "argv", ["specgen.py", "--once", "--group", "plate", "--n", "1"])
    rc = specgen.main()
    assert rc == 0
    assert arms_calls == [("use", "gemma-4-31b"), ("restore",)]


def test_main_still_restores_when_the_batch_raises(monkeypatch):
    arms_calls = []

    def fake_run_arms(*args):
        arms_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(specgen, "_run_arms", fake_run_arms)
    monkeypatch.setattr(specgen, "_default_arm", lambda: "gemma-4-31b")
    monkeypatch.setattr(specgen, "_pre_arm_marker_exists", lambda: True)

    def boom(group, n, dry_run=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(specgen, "run_batch", boom)
    monkeypatch.setattr(sys, "argv", ["specgen.py", "--once", "--group", "plate"])
    rc = specgen.main()
    assert rc == 1
    assert arms_calls == [("use", "gemma-4-31b"), ("restore",)]


def test_main_still_restores_when_a_circuit_breaker_trips(monkeypatch, capsys):
    arms_calls = []

    def fake_run_arms(*args):
        arms_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(specgen, "_run_arms", fake_run_arms)
    monkeypatch.setattr(specgen, "_default_arm", lambda: "gemma-4-31b")
    monkeypatch.setattr(specgen, "_pre_arm_marker_exists", lambda: True)
    monkeypatch.setattr(specgen, "run_batch",
                        lambda group, n, dry_run=False: (_ for _ in ()).throw(
                            RuntimeError("connection refused")))
    monkeypatch.setattr(sys, "argv", ["specgen.py", "--total", "1000"])
    rc = specgen.main()
    assert rc == 1
    assert arms_calls == [("use", "gemma-4-31b"), ("restore",)]
    err = capsys.readouterr().err
    # The abort reason must be the true LAST line printed (after the restore bookend's
    # own forwarded output), so a supervisor can grep the tail of the log for it.
    last_line = [ln for ln in err.splitlines() if ln.strip()][-1]
    assert last_line.startswith("specgen abort:")
    assert "circuit breaker" in last_line


def test_main_aborts_without_generating_when_arms_use_fails(monkeypatch):
    ensure_resident_calls = []

    def fake_run_arms(*args):
        rc = 1 if args and args[0] == "use" else 0
        return subprocess.CompletedProcess(args, rc, stdout="", stderr="model missing")

    monkeypatch.setattr(specgen, "_run_arms", fake_run_arms)
    monkeypatch.setattr(specgen, "_default_arm", lambda: "gemma-4-31b")
    # An ordinary (non-signal) `use` failure never gets as far as writing a marker
    # (cmd_use's own model-path check runs before _save_pre_arm), so the real
    # _pre_arm_marker_exists() would already return False here -- stubbed explicitly so
    # this test does not depend on the ambient state of any real file on the machine
    # running the suite, and _ensure_resident_up is stubbed so no real `systemctl`
    # call happens in a test.
    monkeypatch.setattr(specgen, "_pre_arm_marker_exists", lambda: False)
    monkeypatch.setattr(specgen, "_ensure_resident_up",
                        lambda: ensure_resident_calls.append(1))
    monkeypatch.setattr(specgen, "run_batch",
                        lambda *a, **k: pytest.fail("must not generate when use failed"))
    monkeypatch.setattr(sys, "argv", ["specgen.py", "--once", "--group", "plate"])
    rc = specgen.main()
    assert rc == 1
    assert ensure_resident_calls == [1]


def test_no_relock_flag_skips_the_arms_bookend_entirely(monkeypatch):
    monkeypatch.setattr(specgen, "_run_arms",
                        lambda *a: pytest.fail("must not call arms.py under --no-relock"))
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    monkeypatch.setattr(specgen, "_seed_specs", lambda group, n=5: [])
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps([{"tier": 1, "spec": "a no-relock smoke spec"}]))
    monkeypatch.setattr(sys, "argv",
                        ["specgen.py", "--once", "--group", "plate", "--no-relock"])
    rc = specgen.main()
    assert rc == 0

"""Offline tests for lab/specbank.py and lab/specgen.py's pure/stubbed pieces.

specbank.py is imported normally (`from lab import specbank`) -- it only pulls in
harvest_census and lab.data (jinja2), both lightweight, same as production callers
(scripts/run_card.py, lab/ship.py already `from lab.data import ...` this way).
specgen.py additionally imports cad_engine, which existing tests (test_maker_config.py)
already import directly in the plain test suite, so no stub/heavy-dep gymnastics are
needed here either -- engine._ollama itself is monkeypatched per test.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from lab import specbank  # noqa: E402
import harvest_census as hc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    """Every test gets its own empty specs.jsonl / refs dir -- never the real bank."""
    specs_file = tmp_path / "specs.jsonl"
    refs_dir = tmp_path / "refs"
    monkeypatch.setattr(specbank, "SPECS_FILE", specs_file)
    monkeypatch.setattr(specbank, "REFS_DIR", refs_dir)
    return specs_file


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
    monkeypatch.setattr(specbank, "contamination_sets", lambda: (set(), set()))
    spec = "a stepped shaft 12mm to 8mm with a 3mm keyway 20mm long"
    r1 = specbank.add_items([{"spec": spec, "tier": 2, "group": "shaft"}], source="specgen")
    assert r1["accepted"] == 1
    r2 = specbank.add_items([{"spec": spec, "tier": 2, "group": "shaft"}], source="specgen")
    assert r2["accepted"] == 0
    assert r2["refused"][0]["reason"] == "duplicate-in-bank"


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
# specgen.gen_family: the JSON-array parse + one repair-retry path (engine stubbed)
# ---------------------------------------------------------------------------

import cad_engine  # noqa: E402
from lab import specgen  # noqa: E402


def test_specgen_parses_a_clean_json_array(monkeypatch):
    calls = []

    def fake_ollama(model, system, prompt, **kw):
        calls.append((model, kw.get("temperature"), kw.get("no_think")))
        return json.dumps([{"tier": 2, "spec": "a clean 30mm plate with two 5mm holes"}])

    monkeypatch.setattr(specgen.engine, "_ollama", fake_ollama)
    items = specgen.gen_family("plate", 1, "tier 1-2 guidance", [])
    assert items == [{"tier": 2, "group": "plate", "spec": "a clean 30mm plate with two 5mm holes"}]
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
    items = specgen.gen_family("surface", 1, "tier 3 guidance", [])
    assert items == [{"tier": 3, "group": "surface", "spec": "a repaired spec after retry"}]
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
    items = specgen.gen_family("mechanism", 1, "guidance", [])
    assert items[0]["tier"] == 4   # _TIER_BY_GROUP["mechanism"]


def test_specgen_drops_items_without_a_spec(monkeypatch):
    monkeypatch.setattr(specgen.engine, "_ollama",
                        lambda model, system, prompt, **kw:
                        json.dumps([{"tier": 2}, {"tier": 2, "spec": "kept spec here"}]))
    items = specgen.gen_family("plate", 2, "guidance", [])
    assert len(items) == 1
    assert items[0]["spec"] == "kept spec here"


def test_seed_specs_falls_back_to_teacher_suite_then_to_empty(monkeypatch):
    monkeypatch.setattr(specbank, "load_bank", lambda: [])
    assert specgen._seed_specs("plate") == []

    monkeypatch.setattr(specbank, "load_bank", lambda: [
        {"spec": "a teacher spec", "group": "bracket", "source": "teacher-suite"}])
    seeds = specgen._seed_specs("plate")   # no "plate" rows -> falls back to teacher-suite
    assert seeds == ["a teacher spec"]


def test_run_batch_reports_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(specbank, "SPECS_FILE", tmp_path / "specs.jsonl")
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


def test_run_batch_unknown_group_raises():
    with pytest.raises(SystemExit):
        specgen.run_batch("not-a-real-group", 1)

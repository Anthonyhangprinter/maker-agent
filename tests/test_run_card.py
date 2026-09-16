import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("run_card", HERE / "scripts" / "run_card.py")
rc = importlib.util.module_from_spec(_spec)
# Registered in sys.modules before exec: the dataclasses module (used by run_card.Knobs)
# resolves annotations via sys.modules[cls.__module__] at class-definition time, which
# raises AttributeError on a module that was exec'd but never registered.
sys.modules[_spec.name] = rc
_spec.loader.exec_module(rc)


def test_geometry_of_oneshot_maps_bbox_to_bbox_mm():
    # fluid_gen's `facts` dict IS cad_engine.parse_facts's output — its size key is
    # `bbox`, not `bbox_mm` (score_acceptance's key). This mapping is the whole point of
    # Task 8's "adapt the key mappings to the real interfaces" instruction.
    res = {"facts": {"solids": 1, "bbox": [80.0, 60.0, 20.0], "faces": 10,
                      "cyl_faces": 4, "through_holes": 4}}
    geom = rc.geometry_of(res, "oneshot")
    assert geom == {"solids": 1, "bbox_mm": [80.0, 60.0, 20.0], "faces": 10,
                     "cyl_faces": 4, "through_holes": 4}


def test_geometry_of_oneshot_missing_facts_is_empty():
    # {} , not a dict of Nones: score_acceptance reads "did this build produce geometry?" off
    # bool(geom), and a dict of Nones both looks like geometry and raises TypeError on
    # `cyl >= 0` once a min_holes criterion is in play (finding C1).
    assert rc.geometry_of({"ok": False, "error": "boom"}, "oneshot") == {}


def test_geometry_of_agent_missing_step_is_empty():
    # cad_v5's --once --json result carries no top-level geometry facts at all; with no
    # step_local (a build that never produced a STEP) geometry_of must not explode trying
    # to inspect a nonexistent file.
    assert rc.geometry_of({"ok": False}, "agent") == {}


def test_step_of_oneshot_picks_latest_step_in_build_dir(tmp_path):
    (tmp_path / "build.step").write_text("x")
    res = {"build_dir": str(tmp_path)}
    assert rc.step_of(res, "oneshot") == tmp_path / "build.step"


def test_step_of_agent_uses_step_local():
    res = {"step_local": "/tmp/does-not-need-to-exist/build.step"}
    assert rc.step_of(res, "agent") == Path("/tmp/does-not-need-to-exist/build.step")


def test_contamination_flags_slug_overlap(tmp_path, monkeypatch):
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": "a 100x60x20mm enclosure with 2mm walls"}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    specs_by_suite = {"text-to-cad": [{"id": "01", "spec": "a 100x60x20mm enclosure with 2mm walls"}]}
    clashes = rc.contamination(specs_by_suite)
    assert len(clashes) == 1 and clashes[0].startswith("text-to-cad/01:")


def test_contamination_clean_when_no_overlap(tmp_path, monkeypatch):
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": "a totally different gizmo"}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    specs_by_suite = {"text-to-cad": [{"id": "01", "spec": "a 100x60x20mm enclosure with 2mm walls"}]}
    assert rc.contamination(specs_by_suite) == []


def test_load_suite_unwraps_benchmarks_key(tmp_path, monkeypatch):
    suite_dir = tmp_path / "some-suite"
    suite_dir.mkdir()
    (suite_dir / "specs.json").write_text(json.dumps({"benchmarks": [{"id": "a", "spec": "x"}]}))
    monkeypatch.setattr(rc, "BENCH", tmp_path)
    specs, acc = rc.load_suite("some-suite")
    assert specs == [{"id": "a", "spec": "x"}]
    assert acc == {}


def test_load_suite_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "BENCH", tmp_path)
    specs, acc = rc.load_suite("nope")
    assert specs == [] and acc == {}


def test_run_row_ok_is_always_a_real_bool_not_the_solids_count(monkeypatch):
    # bool(x) and y returns y verbatim when x is truthy (Python's `and` short-circuits to
    # the operand, not to a bool). Found live in the smoke run's rows.jsonl: oneshot rows
    # came out with ok=1 (the int solids count) instead of ok=True. run_row must coerce.
    res = {"ok": True, "facts": {"solids": 1, "bbox": [10, 10, 10], "faces": 6}, "build_dir": "/tmp/nope"}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None: (res, 1.0, ""))
    monkeypatch.setattr(rc, "score_acceptance", lambda geom, crit: {"passed": 0, "total": 0})
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, {}, "oneshot", 60)
    assert row["ok"] is True and type(row["ok"]) is bool


def test_run_row_ok_false_stays_false(monkeypatch):
    res = {"ok": False, "error": "boom"}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None: (res, 1.0, "err"))
    monkeypatch.setattr(rc, "score_acceptance", lambda geom, crit: {"passed": 0, "total": 0})
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, {}, "oneshot", 60)
    assert row["ok"] is False and type(row["ok"]) is bool


def test_acceptance_denominator_ignores_non_criterion_keys(monkeypatch):
    """C1: ok and failed rows must share one denominator. The failed branch used to count raw
    criteria keys, so a suite entry carrying reference_stl/normalized/source inflated a
    failure's total past what score_acceptance would ever have scored."""
    crit = {"solids": 1, "reference_stl": "refs/x.stl", "normalized": True,
            "source": "CADPrompt/0001", "bbox_notes": "emergent", "min_holes_nulls": True}
    assert rc.criteria_of(crit) == {"solids": 1}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None: ({"ok": False, "error": "boom"}, 1.0, ""))
    failed = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, crit, "oneshot", 60)
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None:
                        ({"ok": True, "facts": {"solids": 1, "bbox": [1, 1, 1], "faces": 6}}, 1.0, ""))
    ok = rc.run_row("arm", "suite", {"id": "01", "spec": "x"}, {"solids": 1}, "oneshot", 60)
    assert failed["acc_total"] == ok["acc_total"] == 1
    assert failed["acc_passed"] == 0 and ok["acc_passed"] == 1


def test_no_geometry_scores_zero_of_n_instead_of_raising(monkeypatch):
    """geometry_of must return {} (score_acceptance's "no geometry" contract), not a dict of
    Nones: with a min_holes criterion the None cyl_faces raised TypeError on `cyl >= 0`."""
    assert rc.geometry_of({"ok": False, "error": "boom"}, "oneshot") == {}
    assert rc.geometry_of({"ok": False}, "agent") == {}
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None: ({"ok": False, "error": "boom"}, 1.0, ""))
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "x"},
                     {"solids": 1, "min_holes": 4, "reference_stl": "refs/x.stl"}, "oneshot", 60)
    assert row["acc_passed"] == 0 and row["acc_total"] == 2


def test_band_normalisation_follows_the_suite_not_the_runner(monkeypatch, tmp_path):
    """I8: normalize=True was applied to every suite, including the mm-specified heldout-cqe,
    where rescaling would forgive a part built at half the specified size."""
    seen = {}

    def fake_score(step, ref, normalize=False):
        seen["normalize"] = normalize
        return {"band": "match"}

    monkeypatch.setattr(rc.geom_bands, "score_against_reference", fake_score)
    rc.band_of(tmp_path / "a.step", tmp_path / "ref.stl", {"reference_stl": "refs/x.stl"})
    assert seen["normalize"] is False
    rc.band_of(tmp_path / "a.step", tmp_path / "ref.stl",
               {"reference_stl": "refs/x.stl", "normalized": True})
    assert seen["normalize"] is True


def test_helper_flag_rides_the_row(monkeypatch):
    monkeypatch.setattr(rc, "build_once", lambda spec, mode, timeout, knobs=None:
                        ({"ok": True, "helper": True, "facts": {"solids": 1}}, 1.0, ""))
    row = rc.run_row("arm", "suite", {"id": "01", "spec": "a spur gear"}, {}, "oneshot", 60)
    assert row["helper"] is True


def test_internal_and_external_lists_derive_from_card_suites():
    """M6: the card's suite lists and the contamination guard's must not drift apart."""
    import harvest_census as hc
    assert rc.INTERNAL + rc.EXTERNAL == hc.CARD_SUITES


def test_contamination_uses_the_exact_key_not_a_truncated_slug(tmp_path, monkeypatch):
    """I2: two specs sharing a 40-char opening are different specs. Slug-keyed, one training
    row would have condemned every CADPrompt spec."""
    shared = "Create a 3D model by extruding a circular sketch. "
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": shared + "Radius 0.75 units."}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    clashes = rc.contamination({"cadprompt": [
        {"id": "cp-1", "spec": shared + "Radius 0.75 units."},
        {"id": "cp-2", "spec": shared + "Radius 2.5 units."}]})
    assert [c.split(":")[0] for c in clashes] == ["cadprompt/cp-1"]


def test_run_card_skips_an_arm_that_will_not_load(tmp_path, monkeypatch):
    """I7: one unloadable arm used to abort the whole --arms all card."""
    def boom(arm, **kw):
        if arm["name"] == "bad":
            raise SystemExit("gguf missing")

    monkeypatch.setattr(rc.arms_mod, "cmd_use", boom)
    monkeypatch.setattr(rc.arms_mod, "cmd_restore", lambda: None)
    monkeypatch.setattr(rc.arms_mod, "load_arms", lambda: {
        "bad": {"name": "bad"}, "good": {"name": "good"}})
    monkeypatch.setattr(rc, "load_suite", lambda name: ([{"id": "01", "spec": "x", "tier": 1}], {}))
    monkeypatch.setattr(rc, "contamination", lambda specs: [])
    rows = []

    def fake_row(arm, suite, spec, crit, mode, timeout, knobs=None):
        rows.append(arm)
        return {"arm": arm, "suite": suite, "id": spec["id"], "tier": 1, "ok": True,
                "gate_hard": 0, "gate_spec": 0, "acc_passed": 0, "acc_total": 0, "band": None,
                "helper": False, "wall_s": 1.0, "tokens_out": 0, "build_dir": "", "error": None,
                "stderr_tail": ""}

    monkeypatch.setattr(rc, "run_row", fake_row)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_card.py", "--suites", "text-to-cad", "--out", str(out)])
    rc.main()
    assert rows == ["good"]                       # the second arm still ran
    meta = json.loads((out / "card.json").read_text())["meta"]
    assert "gguf missing" in meta["skipped_arms"]["bad"]


def test_run_card_honours_the_skip_key(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(rc.arms_mod, "cmd_use", lambda arm, **kw: used.append(arm["name"]))
    monkeypatch.setattr(rc.arms_mod, "cmd_restore", lambda: None)
    monkeypatch.setattr(rc.arms_mod, "load_arms", lambda: {
        "parked": {"name": "parked", "skip": True}, "live": {"name": "live", "skip": False}})
    monkeypatch.setattr(rc, "load_suite", lambda name: ([{"id": "01", "spec": "x", "tier": 1}], {}))
    monkeypatch.setattr(rc, "contamination", lambda specs: [])
    monkeypatch.setattr(rc, "run_row", lambda *a, **k: {
        "arm": a[0], "suite": a[1], "id": "01", "tier": 1, "ok": True, "gate_hard": 0,
        "gate_spec": 0, "acc_passed": 0, "acc_total": 0, "band": None, "helper": False,
        "wall_s": 1.0, "tokens_out": 0, "build_dir": "", "error": None, "stderr_tail": ""})
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_card.py", "--suites", "text-to-cad", "--out", str(out)])
    rc.main()
    assert used == ["live"]
    assert json.loads((out / "card.json").read_text())["meta"]["skipped_arms"]["parked"]


def test_rescore_fixes_stale_denominators_and_rewrites_the_card(tmp_path, monkeypatch):
    out = tmp_path / "run"; out.mkdir()
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"arm": "a", "suite": "s", "id": "01", "tier": 1, "ok": False, "gate_hard": 1,
         "gate_spec": 0, "acc_passed": 0, "acc_total": 6, "band": None, "wall_s": 5.0,
         "tokens_out": 10, "build_dir": "", "error": "boom", "stderr_tail": ""},
        {"arm": "a", "suite": "s", "id": "02", "tier": 1, "ok": True, "gate_hard": 0,
         "gate_spec": 0, "acc_passed": 1, "acc_total": 1, "band": "match", "wall_s": 6.0,
         "tokens_out": 20, "build_dir": "", "error": None, "stderr_tail": ""}]) + "\n")
    monkeypatch.setattr(rc, "load_suite", lambda name: (
        [], {"01": {"solids": 1, "reference_stl": "refs/x.stl", "normalized": True,
                    "source": "upstream"},
             "02": {"solids": 1}}))
    md = rc.rescore(out)
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["acc_total"] == 1 and rows[0]["acc_passed"] == 0   # was 0/6
    assert rows[1]["acc_total"] == 1                                   # untouched
    assert all("helper" in r for r in rows)
    assert (out / "rows.jsonl.bak").exists()
    assert "| helper |" in md and (out / "card.md").exists()
    assert json.loads((out / "card.json").read_text())["meta"]["rescored"]


def test_latest_symlink_only_for_runs_inside_card_dir(tmp_path, monkeypatch):
    """M6: `latest` points at a sibling by name, so an --out elsewhere used to leave it dangling."""
    monkeypatch.setattr(rc, "CARD_DIR", tmp_path / "card")
    (tmp_path / "card").mkdir()
    elsewhere = tmp_path / "other" / "run"; elsewhere.mkdir(parents=True)
    rc.write_card(elsewhere, [], {"stamp": "x", "mode": "oneshot"})
    assert not (tmp_path / "card" / "latest").exists()
    inside = tmp_path / "card" / "20260915-1200"; inside.mkdir()
    rc.write_card(inside, [], {"stamp": inside.name, "mode": "oneshot"})
    assert (tmp_path / "card" / "latest").is_symlink()


def test_contamination_reports_a_reworded_clash_as_a_near_duplicate(tmp_path, monkeypatch):
    """The real known clash (text-to-cad/05) is the same part REWORDED, so exact-key matching
    alone would miss it. A slug match counts, but only when that slug identifies exactly one
    spec in its suite."""
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({
        "spec": "a solid open-top electronics enclosure box 100x70x30mm with 3mm walls"}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    clashes = rc.contamination({"text-to-cad": [
        {"id": "05", "spec": "A solid open-top electronics enclosure base. Outer box 100mm in X, "
                             "70mm in Y, 30mm tall in Z, with 3mm walls and a 3mm floor."},
        {"id": "06", "spec": "A flanged shaft with three bolt holes."}]})
    assert len(clashes) == 1
    assert clashes[0].startswith("text-to-cad/05 (near duplicate")


def test_contamination_near_duplicate_ignores_degenerate_slug_buckets(tmp_path, monkeypatch):
    """A slug shared by many specs in the same suite is not evidence about any of them: that
    is the degeneracy that made the old slug-only key condemn whole public suites."""
    shared_opening = "Create a 3D model by extruding a circular sketch and then "
    train_file = tmp_path / "cad-examples.jsonl"
    train_file.write_text(json.dumps({"spec": shared_opening + "something else entirely."}) + "\n")
    monkeypatch.setattr(rc, "TRAIN_FILES", [train_file])
    specs = [{"id": f"cp-{i}", "spec": shared_opening + f"cutting {i} holes."} for i in range(5)]
    assert rc.contamination({"cadprompt": specs}) == []


def test_subset_phase1_caps_each_suite(monkeypatch):
    import run_card as rc
    specs = {"cadprompt": ([{"id": f"cp-{i}", "spec": "x", "tier": 0} for i in range(100)], {}),
             "cad-arena": ([{"id": f"a-{i}", "spec": "y", "tier": 1} for i in range(12)], {})}
    cut = rc.apply_subset(specs, "phase1")
    assert len(cut["cadprompt"][0]) == 30 and [s["id"] for s in cut["cadprompt"][0]][:3] == ["cp-0", "cp-1", "cp-2"]
    assert len(cut["cad-arena"][0]) == 12
    assert rc.apply_subset(specs, "full") == specs


def test_variant_labels_arm_and_env(monkeypatch):
    import run_card as rc
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd; seen["env"] = kw["env"]
        class P: stdout = '{"ok": true, "facts": {"solids": 1}, "gate_hard": [], "gate_spec": [], "usage": {"completion_tokens": 5}, "build_dir": ""}'; stderr = ""; returncode = 0
        return P()
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    row = rc.run_row("gemma-4-31b", "cad-arena", {"id": "a-1", "spec": "A cube", "tier": 1}, {"solids": 1}, "oneshot", 60,
                     knobs=rc.Knobs(variant="bo3", candidates=3, no_fewshots=True, critic="local:minicpm-v"))
    assert row["arm"] == "gemma-4-31b+bo3"
    assert seen["env"]["CAD_CANDIDATES"] == "3" and seen["env"]["CAD_CRITIC_MODEL"] == "local:minicpm-v"
    assert "--no-fewshots" in seen["cmd"]

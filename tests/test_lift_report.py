"""lift_table / render_lift_md (Task 5, scripts/lift_report.py): variant deltas vs a baseline
arm on the public suites only (cadprompt, text2cadquery, heldout-cqe), excluding helper rows.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import lift_report as lr

# Baseline arm "a": 4 public (cadprompt) rows + 1 internal row (text-to-cad, must be ignored)
# + 1 helper row (cadprompt, helper=True, must be ignored despite being a public suite).
BASELINE_ROWS = [
    {"arm": "a", "suite": "cadprompt", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 10.0, "tokens_out": 100, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a", "suite": "cadprompt", "id": "2", "tier": 1, "ok": True, "gate_hard": 1, "gate_spec": 0,
     "acc_passed": 1, "acc_total": 2, "band": "fail", "wall_s": 20.0, "tokens_out": 200, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a", "suite": "cadprompt", "id": "3", "tier": 1, "ok": False, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 0, "acc_total": 2, "band": None, "wall_s": 30.0, "tokens_out": None, "helper": False,
     "build_dir": "", "error": "no STEP"},
    {"arm": "a", "suite": "cadprompt", "id": "4", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 40.0, "tokens_out": 400, "helper": False,
     "build_dir": "", "error": None},
    # internal suite: must not affect the baseline's numbers at all
    {"arm": "a", "suite": "text-to-cad", "id": "99", "tier": 1, "ok": False, "gate_hard": 1, "gate_spec": 1,
     "acc_passed": 0, "acc_total": 9, "band": "fail", "wall_s": 999.0, "tokens_out": 9999, "helper": False,
     "build_dir": "", "error": "internal, must be ignored"},
    # helper row on a public suite: must also not affect the baseline's numbers
    {"arm": "a", "suite": "cadprompt", "id": "98", "tier": 1, "ok": False, "gate_hard": 1, "gate_spec": 1,
     "acc_passed": 0, "acc_total": 9, "band": "fail", "wall_s": 999.0, "tokens_out": 9999, "helper": True,
     "build_dir": "", "error": "helper row, must be ignored"},
]

# Variant "a+v": same shape, tuned so every delta is a clean number.
VARIANT_ROWS = [
    {"arm": "a+v", "suite": "cadprompt", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 8.0, "tokens_out": 90, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a+v", "suite": "cadprompt", "id": "2", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 18.0, "tokens_out": 180, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a+v", "suite": "cadprompt", "id": "3", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 1,
     "acc_passed": 1, "acc_total": 2, "band": "valid", "wall_s": 28.0, "tokens_out": 280, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a+v", "suite": "cadprompt", "id": "4", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 38.0, "tokens_out": 380, "helper": False,
     "build_dir": "", "error": None},
    # internal suite: must not affect the variant's numbers
    {"arm": "a+v", "suite": "organic", "id": "99", "tier": 1, "ok": False, "gate_hard": 1, "gate_spec": 1,
     "acc_passed": 0, "acc_total": 9, "band": "fail", "wall_s": 999.0, "tokens_out": 9999, "helper": False,
     "build_dir": "", "error": "internal, must be ignored"},
    # helper row on a public suite: must not affect the variant's numbers
    {"arm": "a+v", "suite": "text2cadquery", "id": "98", "tier": 1, "ok": False, "gate_hard": 1, "gate_spec": 1,
     "acc_passed": 0, "acc_total": 9, "band": "fail", "wall_s": 999.0, "tokens_out": 9999, "helper": True,
     "build_dir": "", "error": "helper row, must be ignored"},
]

# Second variant: no rows on any public suite at all (only internal + helper).
EMPTY_ROWS = [
    {"arm": "a+empty", "suite": "organic", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 1, "acc_total": 1, "band": None, "wall_s": 5.0, "tokens_out": 50, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a+empty", "suite": "cadprompt", "id": "2", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 1, "acc_total": 1, "band": "match", "wall_s": 5.0, "tokens_out": 50, "helper": True,
     "build_dir": "", "error": None},
]

ALL_ROWS = BASELINE_ROWS + VARIANT_ROWS + EMPTY_ROWS


def test_baseline_metrics_public_only_no_helper():
    table = lr.lift_table(ALL_ROWS, baseline="a")
    a = table["a"]
    assert a["n"] == 4
    assert abs(a["invalid_ratio"] - 0.25) < 1e-9          # (4-3)/4
    assert abs(a["gate_clean_rate"] - 0.5) < 1e-9          # rows 1 and 4
    assert abs(a["acceptance"] - 5 / 8) < 1e-9             # pooled 5/8
    assert a["ref_n"] == 4                                 # all four cadprompt rows have a reference
    assert abs(a["match_rate"] - 0.5) < 1e-9               # 2 match of 4 reffed, the failure is a non-match
    assert a["median_wall_s"] == 25.0                      # median(10,20,30,40)
    assert abs(a["tokens_per_build"] - 700 / 3) < 1e-9     # mean(100,200,400)


def test_baseline_delta_is_all_zero():
    table = lr.lift_table(ALL_ROWS, baseline="a")
    assert table["a"]["delta"] == {
        "invalid_ratio": 0, "gate_clean_rate": 0, "acceptance": 0, "match_rate": 0, "median_wall_s": 0,
    }


def test_variant_metrics_and_hand_computed_deltas():
    table = lr.lift_table(ALL_ROWS, baseline="a")
    v = table["a+v"]
    assert v["n"] == 4
    assert v["invalid_ratio"] == 0.0
    assert abs(v["gate_clean_rate"] - 0.75) < 1e-9
    assert abs(v["acceptance"] - 0.875) < 1e-9
    assert v["ref_n"] == 4
    assert abs(v["match_rate"] - 0.75) < 1e-9
    assert v["median_wall_s"] == 23.0
    assert abs(v["tokens_per_build"] - 232.5) < 1e-9

    d = v["delta"]
    assert abs(d["invalid_ratio"] - (-0.25)) < 1e-9
    assert abs(d["gate_clean_rate"] - 0.25) < 1e-9
    assert abs(d["acceptance"] - 0.25) < 1e-9
    assert abs(d["match_rate"] - 0.25) < 1e-9              # 0.75 - 0.50
    assert abs(d["median_wall_s"] - (-2.0)) < 1e-9


def test_variant_with_zero_public_rows_is_present_with_none_metrics():
    table = lr.lift_table(ALL_ROWS, baseline="a")
    e = table["a+empty"]
    assert e["n"] == 0
    for key in ("invalid_ratio", "gate_clean_rate", "acceptance", "match_rate", "median_wall_s", "tokens_per_build"):
        assert e[key] is None
    for key in ("invalid_ratio", "gate_clean_rate", "acceptance", "match_rate", "median_wall_s"):
        assert e["delta"][key] is None


def test_acceptance_none_when_zero_total():
    rows = [{"arm": "z", "suite": "cadprompt", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
             "acc_passed": 0, "acc_total": 0, "band": None, "wall_s": 1.0, "tokens_out": None, "helper": False,
             "build_dir": "", "error": None}]
    table = lr.lift_table(rows, baseline="z")
    assert table["z"]["acceptance"] is None


def test_render_lift_md_baseline_first_and_percentages():
    table = lr.lift_table(ALL_ROWS, baseline="a")
    md = lr.render_lift_md(table, baseline="a")
    lines = [l for l in md.splitlines() if l.startswith("|") and not l.startswith("|---")]
    header, rows_out = lines[0], lines[1:]
    assert "variant" in header and ("delta" in header.lower() or "Δ" in header)
    # baseline row first, others sorted by name
    assert rows_out[0].startswith("| a |")
    names = [l.split("|")[1].strip() for l in rows_out]
    assert names[0] == "a"
    assert names[1:] == sorted(names[1:])
    # baseline percentages and zero deltas (62% = round-half-to-even of 62.5). The baseline's
    # own base/base n are itself and its own n; match is 2 of the 4 rows that have a reference
    # (row 3 failed, so it has no band and counts as a non-match).
    assert "| a | a | 4 | 4 | 25% | 0 | +0/-0 | 50% | 0 | 62% | 0 | 50% | 4 | 0 | +0/-0 | 25 | 0 | 233 |" in md
    # variant deltas, signed points, wall seconds as ints; base n is the baseline restricted
    # to the variant's own four specs, so still 4 here
    assert "| a+v | a | 4 | 4 | 0% | -25 | +1/-0 | 75% | +25 | 88% | +25 | 75% | 4 | +25 | +1/-0 | 23 | -2 | 232 |" in md
    # zero-public-rows variant renders dashes
    assert "| a+empty | a | 0 | 0 | - | - | +0/-0 | - | - | - | - | - | 0 | - | +0/-0 | - | - | - |" in md


def test_cli_writes_lift_json_and_md(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("w") as f:
        for r in ALL_ROWS:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path), "--baseline", "a"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, result.stderr
    lift_json = json.loads((tmp_path / "LIFT.json").read_text())
    assert lift_json["baseline"] == "a" and lift_json["baseline_for"] == {}
    assert lift_json["table"]["a"]["n"] == 4
    lift_md = (tmp_path / "LIFT.md").read_text()
    assert "a+v" in lift_md and "a+empty" in lift_md
    assert lift_md.strip() in result.stdout.strip() or result.stdout.strip() in lift_md.strip()


def test_cli_default_baseline_is_the_unlabelled_arm(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("w") as f:
        for r in ALL_ROWS:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path)],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, result.stderr
    lift_json = json.loads((tmp_path / "LIFT.json").read_text())
    assert lift_json["baseline"] == "a"
    assert lift_json["table"]["a"]["delta"]["invalid_ratio"] == 0


def test_cli_no_unlabelled_arm_errors(tmp_path):
    """Every arm labelled: there is no default baseline, so exit 2 naming the candidates."""
    rows_path = tmp_path / "rows.jsonl"
    rows = [dict(r, arm="a+x") for r in BASELINE_ROWS] + [dict(r, arm="a+y") for r in VARIANT_ROWS]
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path)],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 2                     # exit 2, not a bare SystemExit's 1
    assert "--baseline" in result.stderr and "a+x" in result.stderr
    assert not (tmp_path / "LIFT.json").exists()


def test_cli_two_unlabelled_arms_errors_naming_both(tmp_path):
    """Two unlabelled arms: exit 2 naming both, rather than silently picking one (M9)."""
    rows_path = tmp_path / "rows.jsonl"
    rows = BASELINE_ROWS + [dict(r, arm="b") for r in VARIANT_ROWS]
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path)],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 2
    assert "ambiguous baseline" in result.stderr and "'a'" in result.stderr and "'b'" in result.stderr
    assert "--baseline" in result.stderr
    assert not (tmp_path / "LIFT.json").exists()


# --- C1: like-for-like baseline restriction ------------------------------------------------
# "a+sub" covers only specs 1 and 2 of the baseline's four, the shape phase1think has inside
# phase1. Its delta must be against the baseline's rows 1 and 2, not the baseline's average
# over all four (which includes the failure on spec 3 and the slow spec 4).
SUBSET_ROWS = [
    {"arm": "a+sub", "suite": "cadprompt", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 9.0, "tokens_out": 90, "helper": False,
     "build_dir": "", "error": None},
    {"arm": "a+sub", "suite": "cadprompt", "id": "2", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0,
     "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 9.0, "tokens_out": 90, "helper": False,
     "build_dir": "", "error": None},
]


def test_subset_variant_is_diffed_against_the_restricted_baseline():
    table = lr.lift_table(BASELINE_ROWS + SUBSET_ROWS, baseline="a")
    s = table["a+sub"]
    assert s["n"] == 2 and s["base"] == "a" and s["base_n"] == 2
    # restricted baseline over specs 1+2: invalid 0, gate clean 1/2, acceptance 3/4,
    # match 1/2, median wall 15. The UNRESTRICTED baseline is 25%/50%/62%/50%/25s.
    b = s["base_metrics"]
    assert b["n"] == 2 and b["invalid_ratio"] == 0.0 and b["median_wall_s"] == 15.0
    assert abs(b["acceptance"] - 0.75) < 1e-9 and abs(b["match_rate"] - 0.5) < 1e-9
    d = s["delta"]
    assert d["invalid_ratio"] == 0.0                   # would be -0.25 against the whole baseline
    assert abs(d["median_wall_s"] - (-6.0)) < 1e-9     # would be -16 against the whole baseline
    assert abs(d["gate_clean_rate"] - 0.5) < 1e-9
    assert abs(d["match_rate"] - 0.5) < 1e-9


def test_baseline_for_diffs_one_variant_against_another():
    """agent-self vs agent-gemma4: the interesting comparison is variant to variant, not
    each variant to the one-shot baseline."""
    p = [dict(r, arm="a+p") for r in SUBSET_ROWS]
    q = [dict(r, arm="a+q", band="fail", wall_s=20.0) for r in SUBSET_ROWS]
    table = lr.lift_table(BASELINE_ROWS + p + q, baseline="a", baseline_for={"a+q": "a+p"})
    assert table["a+p"]["base"] == "a"
    assert table["a+q"]["base"] == "a+p" and table["a+q"]["base_n"] == 2
    # a+p matched both specs, a+q neither: -100 points against a+p, and two worsened flips
    assert abs(table["a+q"]["delta"]["match_rate"] - (-1.0)) < 1e-9
    assert table["a+q"]["flips"]["match_rate"] == {"improved": 0, "worsened": 2}
    assert abs(table["a+q"]["delta"]["median_wall_s"] - 11.0) < 1e-9


# --- C2: match denominator + paired flips --------------------------------------------------
def _row(arm, rid, **kw):
    base = {"arm": arm, "suite": "cadprompt", "id": rid, "tier": 1, "ok": True, "gate_hard": 0,
            "gate_spec": 0, "acc_passed": 1, "acc_total": 1, "band": None, "wall_s": 10.0,
            "tokens_out": 10, "helper": False, "build_dir": "", "error": None}
    base.update(kw)
    return base


def test_match_rate_counts_invalid_builds_as_non_matches():
    """An invalid build has no band. Dropping it from the denominator scored an arm only on
    the specs it managed to build, so failing more could raise the match rate."""
    rows = [_row("b", "1", has_ref=True, band="match"),
            _row("b", "2", has_ref=True, band="fail"),
            _row("b", "3", has_ref=True, ok=False, band=None, error="no STEP")]
    table = lr.lift_table(rows, baseline="b")
    assert table["b"]["ref_n"] == 3
    assert abs(table["b"]["match_rate"] - 1 / 3) < 1e-9   # not 1/2 over the two that banded


def test_has_ref_false_rows_are_out_of_the_match_denominator():
    rows = [_row("b", "1", has_ref=True, band="match"),
            _row("b", "2", has_ref=False)]          # no reference for this spec at all
    table = lr.lift_table(rows, baseline="b")
    assert table["b"]["n"] == 2 and table["b"]["ref_n"] == 1
    assert table["b"]["match_rate"] == 1.0


def test_legacy_rows_without_has_ref_fall_back_to_the_band_union():
    """Rows written before run_card recorded has_ref: a spec that carries a band in ANY arm
    is treated as having a reference in every arm, so a failure is still a non-match."""
    rows = [_row("b", "1", band="match"), _row("b", "2", ok=False, band=None),
            _row("b+v", "1", band="match"), _row("b+v", "2", band="match")]
    table = lr.lift_table(rows, baseline="b")
    assert table["b"]["ref_n"] == 2                       # spec 2 banded under b+v
    assert abs(table["b"]["match_rate"] - 0.5) < 1e-9
    assert table["b+v"]["ref_n"] == 2 and table["b+v"]["match_rate"] == 1.0


def test_flips_are_paired_per_spec_and_count_both_directions():
    """A net delta of zero can be one improvement and one regression, which is a different
    finding from nothing changing."""
    rows = [_row("b", "1", has_ref=True, band="match"), _row("b", "2", has_ref=True, band="fail"),
            _row("b", "3", has_ref=True, ok=False, band=None),
            _row("b+v", "1", has_ref=True, band="fail"), _row("b+v", "2", has_ref=True, band="match"),
            _row("b+v", "3", has_ref=True, ok=False, band=None)]
    table = lr.lift_table(rows, baseline="b")
    v = table["b+v"]
    assert v["delta"]["match_rate"] == 0.0                       # 1/3 both sides
    assert v["flips"]["match_rate"] == {"improved": 1, "worsened": 1}
    assert v["flips"]["invalid_ratio"] == {"improved": 0, "worsened": 0}
    assert table["b"]["flips"]["match_rate"] == {"improved": 0, "worsened": 0}


def test_flips_ignore_specs_the_other_arm_never_ran():
    rows = [_row("b", "1", has_ref=True, band="fail"), _row("b", "2", has_ref=True, band="fail"),
            _row("b+v", "1", has_ref=True, band="match")]
    table = lr.lift_table(rows, baseline="b")
    assert table["b+v"]["flips"]["match_rate"] == {"improved": 1, "worsened": 0}
    assert table["b+v"]["flips"]["invalid_ratio"] == {"improved": 0, "worsened": 0}


def test_invalid_flips_count_builds_fixed_and_broken():
    rows = [_row("b", "1", ok=False), _row("b", "2", ok=True), _row("b", "3", ok=True),
            _row("b+v", "1", ok=True), _row("b+v", "2", ok=False), _row("b+v", "3", ok=True)]
    table = lr.lift_table(rows, baseline="b")
    assert table["b+v"]["delta"]["invalid_ratio"] == 0.0
    assert table["b+v"]["flips"]["invalid_ratio"] == {"improved": 1, "worsened": 1}


def test_cli_baseline_for_lands_in_lift_json(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    p = [dict(r, arm="a+p") for r in SUBSET_ROWS]
    q = [dict(r, arm="a+q", band="fail") for r in SUBSET_ROWS]
    with rows_path.open("w") as f:
        for r in BASELINE_ROWS + p + q:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path),
         "--baseline", "a", "--baseline-for", "a+q=a+p"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, result.stderr
    doc = json.loads((tmp_path / "LIFT.json").read_text())
    assert doc["baseline"] == "a" and doc["baseline_for"] == {"a+q": "a+p"}
    assert doc["table"]["a+q"]["base"] == "a+p"
    assert "| a+q | a+p |" in (tmp_path / "LIFT.md").read_text()


def test_cli_rejects_malformed_baseline_for(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("w") as f:
        for r in BASELINE_ROWS:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path),
         "--baseline", "a", "--baseline-for", "a+q"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 2 and "VARIANT=ARM" in result.stderr


def test_duplicate_rows_are_deduped_keeping_the_last():
    """A run_card invocation without --resume rebuilds specs already in the dir and appends a
    second row for each. The lift table must measure one build per spec, the most recent."""
    rows = [_row("b", "1", ok=False, has_ref=True),
            _row("b", "2", ok=False, has_ref=True),
            _row("b", "1", ok=True, has_ref=True, band="match")]   # rerun of spec 1
    table = lr.lift_table(rows, baseline="b")
    assert table["b"]["n"] == 2                     # two specs, not three rows
    assert table["b"]["invalid_ratio"] == 0.5       # the later, successful row 1 wins
    assert abs(table["b"]["match_rate"] - 0.5) < 1e-9

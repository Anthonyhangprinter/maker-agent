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
    assert abs(a["match_rate"] - 2 / 3) < 1e-9             # 2 match of 3 banded
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
    assert abs(v["match_rate"] - 0.75) < 1e-9
    assert v["median_wall_s"] == 23.0
    assert abs(v["tokens_per_build"] - 232.5) < 1e-9

    d = v["delta"]
    assert abs(d["invalid_ratio"] - (-0.25)) < 1e-9
    assert abs(d["gate_clean_rate"] - 0.25) < 1e-9
    assert abs(d["acceptance"] - 0.25) < 1e-9
    assert abs(d["match_rate"] - (2 / 3 * -1 + 0.75)) < 1e-9  # 0.75 - 2/3
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
    # baseline percentages and zero deltas (62% = round-half-to-even of 62.5)
    assert "| a | 4 | 25% | 0 | 50% | 0 | 62% | 0 | 67% | 0 | 25 | 0 | 233 |" in md
    # variant deltas, signed points, wall seconds as ints
    assert "| a+v | 4 | 0% | -25 | 75% | +25 | 88% | +25 | 75% | +8 | 23 | -2 | 232 |" in md
    # zero-public-rows variant renders dashes
    assert "| a+empty | 0 | - | - | - | - | - | - | - | - | - | - | - |" in md


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
    assert lift_json["a"]["n"] == 4
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
    assert lift_json["a"]["delta"]["invalid_ratio"] == 0


def test_cli_ambiguous_baseline_errors(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = [dict(r, arm="a+x") for r in BASELINE_ROWS] + [dict(r, arm="a+y") for r in VARIANT_ROWS]
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "lift_report.py"), str(tmp_path)],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode != 0
    assert not (tmp_path / "LIFT.json").exists()

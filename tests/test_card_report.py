"""Pure summarise/render functions for the Maker Agent card (docs/MAKER-1.0-CAMPAIGN.md).

Task 5b adds only the BenchCAD renderer here (`render_benchcad_md`); Task 7 extends this file
with `summarise`/`render_md` for the internal-suite card table.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import card_report as cr


def test_render_benchcad_md_rows_and_reference():
    res = {"a": {"codeedit": {"score": 0.5, "n": 10, "exec_rate": None}, "codeqa": {"score": 0.6, "n": 10, "exec_rate": None}, "vision2code": None}}
    md = cr.render_benchcad_md(res)
    assert "| a | 0.500 | 0.600 | - |" in md and "Gemma-4-31B-it" in md and "0.664" in md


ROWS = [
    {"arm": "a", "suite": "s", "id": "1", "tier": 1, "ok": True, "gate_hard": 0, "gate_spec": 0, "acc_passed": 2, "acc_total": 2, "band": "match", "wall_s": 10.0, "tokens_out": 100, "build_dir": "", "error": None},
    {"arm": "a", "suite": "s", "id": "2", "tier": 1, "ok": True, "gate_hard": 1, "gate_spec": 0, "acc_passed": 1, "acc_total": 2, "band": "fail", "wall_s": 30.0, "tokens_out": 300, "build_dir": "", "error": None},
    {"arm": "a", "suite": "s", "id": "3", "tier": 2, "ok": False, "gate_hard": 0, "gate_spec": 0, "acc_passed": 0, "acc_total": 2, "band": None, "wall_s": 20.0, "tokens_out": None, "build_dir": "", "error": "no STEP"},
]


def test_summarise_counts():
    s = cr.summarise(ROWS)["a|s"]
    assert s["n"] == 3 and s["valid"] == 2
    assert abs(s["invalid_ratio"] - 1/3) < 1e-9
    assert s["gate_clean"] == 1                       # ok and gate_hard==0 and gate_spec==0
    assert abs(s["acceptance"] - 3/6) < 1e-9          # pooled checks, like run_benchmarks
    assert s["bands"] == {"match": 1, "valid": 0, "near_miss": 0, "fail": 1}
    assert s["median_wall_s"] == 20.0 and s["tokens_out"] == 400


def test_render_md_has_one_line_per_arm_suite():
    md = cr.render_md(cr.summarise(ROWS), {"stamp": "t", "mode": "oneshot"})
    assert "| a | s | 3 |" in md and "invalid" in md.lower()


def test_summarise_counts_helper_rows_and_render_shows_the_column():
    rows = [dict(ROWS[0], helper=True), dict(ROWS[1], helper=False), dict(ROWS[2], helper=False)]
    s = cr.summarise(rows)["a|s"]
    assert s["helper_rows"] == 1
    md = cr.render_md(cr.summarise(rows), {"stamp": "t", "mode": "oneshot"})
    header = [l for l in md.splitlines() if l.startswith("| arm |")][0]
    assert header.rstrip().endswith("| helper |")
    assert md.splitlines()[-1].rstrip().endswith("| 1 |")


def test_summarise_tolerates_rows_written_before_the_helper_column():
    assert cr.summarise(ROWS)["a|s"]["helper_rows"] == 0

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

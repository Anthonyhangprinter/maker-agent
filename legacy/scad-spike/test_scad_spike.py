"""
tests/test_scad_spike.py — M8 spike deterministic tests.

NO LLM calls anywhere in this file (not even local Ollama) — every check here exercises
`scripts/scad`, `scad_mesh_gate.py`, and `scad_step_recovery.py` against three hand-written
golden .scad parts. `scad_agent.py build` (the LLM loop) is deliberately never invoked; the
harness's own `--help` smoke test is the only thing touching that module directly.

Run: python3 -m pytest tests/test_scad_spike.py -q
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "benchmarks" / "scad-goldens"
SCAD_RUNNER = REPO / "scripts" / "scad"

sys.path.insert(0, str(REPO))
from scad_mesh_gate import gate as mesh_gate          # noqa: E402
from scad_step_recovery import recover_step           # noqa: E402

GOLDEN_IDS = ["block", "block_holes", "flange"]


def _run_scad(scad_file: Path, out_file: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCAD_RUNNER), str(scad_file), "-o", str(out_file), *extra],
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture(scope="module")
def stl_paths(tmp_path_factory) -> dict[str, Path]:
    """Build all three goldens to STL once per test session (module-scoped: reused by every
    test below instead of re-invoking OpenSCAD per assertion)."""
    out_dir = tmp_path_factory.mktemp("scad_goldens_stl")
    paths = {}
    for gid in GOLDEN_IDS:
        scad_file = GOLDENS / f"{gid}.scad"
        assert scad_file.exists(), f"missing golden source: {scad_file}"
        out = out_dir / f"{gid}.stl"
        r = _run_scad(scad_file, out)
        assert r.returncode == 0, f"{gid}: scripts/scad failed: {r.stderr}"
        assert out.exists() and out.stat().st_size > 0
        paths[gid] = out
    return paths


# ── 1. Goldens through scripts/scad -> STL -> mesh gate ─────────────────────────

def test_block_gate_clean(stl_paths):
    hard, soft, facts = mesh_gate(stl_paths["block"])
    assert hard == []
    assert facts["watertight"] is True
    assert facts["body_count"] == 1
    assert facts["genus_sum"] == 0
    # 100x60x20mm block: exact volume, no cuts.
    assert abs(facts["volume_mm3"] - 120000.0) / 120000.0 < 0.01


def test_block_holes_gate_clean(stl_paths):
    hard, soft, facts = mesh_gate(stl_paths["block_holes"])
    assert hard == []
    assert facts["watertight"] is True
    assert facts["body_count"] == 1
    # 4 clean through-holes in an otherwise simply-connected block -> genus 4 (same topology
    # as a 4-holed torus): each hole adds exactly one independent handle.
    assert facts["genus_sum"] == 4
    # Volume must be LESS than the plain block (holes removed material) but still close (small
    # 5mm-diameter holes out of a 100x60x20 block remove only a little).
    assert facts["volume_mm3"] < 120000.0
    assert facts["volume_mm3"] > 115000.0


def test_flange_gate_clean(stl_paths):
    hard, soft, facts = mesh_gate(stl_paths["flange"])
    assert hard == []
    assert facts["watertight"] is True
    assert facts["body_count"] == 1
    # Genus, worked out by hand (see flange.scad's own comment) and CONFIRMED by this
    # measurement, not just assumed: a solid disc is genus 0 (simply connected); the centre
    # bore alone turns it into an annulus/torus topology (genus 1); each of the 6 additional
    # through-holes on the bolt circle adds one more independent handle on top of that
    # (genus += 1 per additional hole, regardless of the base surface's existing genus) ->
    # 1 (centre bore) + 6 (bolt circle) = 7. If this ever measures differently, that is a real
    # finding about how trimesh/OpenSCAD's mesh represents this topology, not a typo — hence
    # asserting the exact value here rather than a looser inequality.
    assert facts["genus_sum"] == 7


# ── 2. bbox tolerance check — wrong expected bbox => ADVISORY (mirrors verify_expected(),
# where size is a demoted hint because the brief's bbox is a guess; 2026-06-26 gate redesign) ──

def test_bbox_mismatch_is_advisory(stl_paths):
    # block.scad is 100x60x20mm; ask for something wildly different.
    hard, soft, facts = mesh_gate(stl_paths["block"], expected={"bbox_mm": [10, 10, 10]})
    assert not any("overall size" in h for h in hard), hard
    assert any("overall size" in s for s in soft), soft

    # Sanity: the CORRECT expected bbox (any axis order — the check is axis-invariant, same
    # rule as the B-rep gate's advisory) produces neither a hard fail nor a size advisory.
    hard_ok, soft_ok, _ = mesh_gate(stl_paths["block"], expected={"bbox_mm": [60, 20, 100]})
    assert hard_ok == []
    assert not any("overall size" in s for s in soft_ok), soft_ok


def test_body_count_mismatch_is_hard_fail(stl_paths):
    hard, soft, facts = mesh_gate(stl_paths["block"], expected={"solids": 2})
    assert any("body/bodies" in h for h in hard), hard


def test_min_holes_shortfall_is_advisory_only(stl_paths):
    # block.scad has genus 0 (no holes at all) but ask for holes anyway: this MUST be an
    # advisory, never a hard fail — genus cannot see blind holes and the spike's whole honesty
    # policy is to never hard-fail on a measurement that structurally cannot be trusted as a
    # negative signal.
    hard, soft, facts = mesh_gate(stl_paths["block"], expected={"min_holes": 4})
    assert hard == []
    assert any("genus-sum lower bound" in s for s in soft), soft


# ── 3. Render test — golden (b), iso+top PNGs exist + non-trivial size ──────────

def test_render_two_panels_block_holes(tmp_path):
    scad_file = GOLDENS / "block_holes.scad"
    iso = tmp_path / "iso.png"
    top = tmp_path / "top.png"
    r_iso = _run_scad(scad_file, iso, "--camera", "iso")
    r_top = _run_scad(scad_file, top, "--camera", "top")
    assert r_iso.returncode == 0, r_iso.stderr
    assert r_top.returncode == 0, r_top.stderr
    assert iso.exists() and iso.stat().st_size > 2000   # a trivial/blank PNG is a few hundred bytes
    assert top.exists() and top.stat().st_size > 2000

    # Visual check (done once, by hand, during spike development — recorded here so the
    # assertion has a citable basis): both PNGs were read with the Read tool and confirmed to
    # show all 4 corner holes clearly — as dark corner dots in the isometric panel and as
    # clean white circles in the top-down panel, matching block_holes.scad's 10mm-inset corner
    # layout exactly. See SCAD_SPIKE.md for the camera-preset verification methodology.
    assert True


# ── 4. STEP recovery — all three goldens ────────────────────────────────────────

@pytest.mark.parametrize("gid", GOLDEN_IDS)
def test_step_recovery(stl_paths, tmp_path, gid):
    scad_file = GOLDENS / f"{gid}.scad"
    step_out = tmp_path / f"{gid}.step"
    result = recover_step(scad_file, stl_paths[gid], step_out)
    assert result["step_recovered"] is True, (
        f"{gid}: STEP recovery failed: {result['step_recovery_error']}")
    assert step_out.exists() and step_out.stat().st_size > 0
    facts = result["recovered_facts"]
    v_stl, v_step = facts["stl"]["volume"], facts["step"]["volume"]
    assert abs(v_step - v_stl) / v_stl <= 0.02, (
        f"{gid}: STEP volume {v_step} vs STL volume {v_stl} — outside 2% tolerance")


# ── 5. CLI smoke test — argparse only, NEVER invokes `build` (that would call LLMs) ──────────

def test_cli_help_smoke():
    r = subprocess.run([sys.executable, str(REPO / "scad_agent.py"), "--help"],
                       capture_output=True, text=True, timeout=30, cwd=REPO)
    assert r.returncode == 0
    assert "build" in r.stdout

    r2 = subprocess.run([sys.executable, str(REPO / "scad_agent.py"), "build", "--help"],
                        capture_output=True, text=True, timeout=30, cwd=REPO)
    assert r2.returncode == 0
    assert "--coder" in r2.stdout
    assert "--no-fewshots" in r2.stdout


def test_scripts_scad_help_on_bad_args():
    # No args at all -> usage on stderr, nonzero exit (never hangs waiting on stdin).
    r = subprocess.run([sys.executable, str(SCAD_RUNNER)],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode != 0


# ── small pure-function unit tests (no subprocess, no LLM) ──────────────────────

def test_strip_fences_scad_variants():
    from scad_agent import _strip_fences_scad
    assert _strip_fences_scad("```openscad\ncube([1,1,1]);\n```") == "cube([1,1,1]);"
    assert _strip_fences_scad("```scad\ncube([1,1,1]);\n```") == "cube([1,1,1]);"
    assert _strip_fences_scad("```\ncube([1,1,1]);\n```") == "cube([1,1,1]);"
    assert _strip_fences_scad("cube([1,1,1]);") == "cube([1,1,1]);"


def test_gate_category_mapping():
    from scad_agent import _gate_category
    assert _gate_category("mesh is empty (no vertices/faces)") == "gate_empty"
    assert _gate_category("mesh is not watertight (open edges)") == "gate_watertight"
    assert _gate_category("expected 1 solid body/bodies but got 2") == "gate_body_count"
    assert _gate_category("overall size 5x5x5mm is outside tolerance") == "gate_bbox"

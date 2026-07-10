"""M10 organic mini-benchmark deterministic tests.

NO LLM calls anywhere in this file (not even local Ollama, not even nomic-embed) — everything
here is: JSON parsing of the suite files, pure-function unit tests of the runner's scorer and
suite resolution, and deterministic build123d re-runs of the two M10 gold corpus examples via
`scripts/step` + `scripts/inspect` (that re-run IS the "verified" in verified-gold).

The actual mini-benchmark (`run_benchmarks.py --suite organic`, which invokes the LLM agent)
is deliberately never run here.

Run: python3 -m pytest tests/test_m10_suite.py -q
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ORGANIC = REPO / "benchmarks" / "organic"
TEXT_TO_CAD = REPO / "benchmarks" / "text-to-cad"
CORPUS = Path.home() / ".openclaw" / "cad-examples.jsonl"

# Import the runner as a module (scripts/ has no package __init__; load by path).
_spec = importlib.util.spec_from_file_location("run_benchmarks", REPO / "scripts" / "run_benchmarks.py")
run_benchmarks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmarks)

ORGANIC_IDS = ["o1", "o2", "o3", "o4", "o5"]


# ── 1. Suite files parse + mirror the text-to-cad format ────────────────────────

@pytest.fixture(scope="module")
def organic_specs() -> dict:
    return json.loads((ORGANIC / "specs.json").read_text())


@pytest.fixture(scope="module")
def organic_accept() -> dict:
    return json.loads((ORGANIC / "acceptance.json").read_text())


def test_specs_mirror_format(organic_specs):
    ref = json.loads((TEXT_TO_CAD / "specs.json").read_text())
    # same top-level shape as the reference suite
    assert set(organic_specs.keys()) == set(ref.keys()) == {"source", "note", "benchmarks"}
    benches = organic_specs["benchmarks"]
    assert [b["id"] for b in benches] == ORGANIC_IDS
    ref_keys = set(ref["benchmarks"][0].keys())
    for b in benches:
        assert set(b.keys()) == ref_keys == {"id", "name", "tier", "spec"}
        assert b["tier"] in (1, 2)
        assert len(b["spec"]) > 50, f"{b['id']}: spec too thin to be buildable-precise"
        assert "mm" in b["spec"], f"{b['id']}: no explicit dimensions"
    # exactly two tier-1 (the easiest), rest tier-2
    assert sum(1 for b in benches if b["tier"] == 1) == 2


def test_every_id_has_acceptance(organic_specs, organic_accept):
    ids = {b["id"] for b in organic_specs["benchmarks"]}
    accept_ids = {k for k in organic_accept if not k.startswith("_")}
    assert accept_ids == ids
    assert "_meta" in organic_accept  # the honest what-is/isn't-checked block


def test_acceptance_criteria_shape(organic_accept):
    for bid in ORGANIC_IDS:
        crit = organic_accept[bid]
        assert crit["solids"] == 1
        bbox = crit["bbox_mm"]
        assert isinstance(bbox, list) and len(bbox) == 3
        assert all(isinstance(v, (int, float)) and v > 0 for v in bbox)
        # null = skipped-not-failed convention: keys must be present, value int or None
        for k in ("min_holes", "min_through_holes", "min_faces"):
            assert k in crit
            assert crit[k] is None or isinstance(crit[k], int)
    # the design decisions this suite documents in _meta must hold in the data:
    # hex/diamond cutouts are invisible to cyl-face hole detection -> min_holes null, min_faces set
    assert organic_accept["o4"]["min_holes"] is None
    assert organic_accept["o5"]["min_holes"] is None
    assert isinstance(organic_accept["o4"]["min_faces"], int)
    assert isinstance(organic_accept["o5"]["min_faces"], int)
    # the o3 recess leaves a cylindrical face -> the one checkable min_holes
    assert organic_accept["o3"]["min_holes"] == 1


# ── 2. Runner --suite resolution (no builds) ────────────────────────────────────

def test_suite_resolution_default_is_byte_identical():
    specs, accept = run_benchmarks.suite_files()  # no arg = default suite
    assert specs == run_benchmarks.SPECS_FILE
    assert accept == run_benchmarks.ACCEPT_FILE
    specs2, accept2 = run_benchmarks.suite_files("text-to-cad")
    assert (specs2, accept2) == (specs, accept)


def test_suite_resolution_organic():
    specs, accept = run_benchmarks.suite_files("organic")
    assert specs == ORGANIC / "specs.json"
    assert accept == ORGANIC / "acceptance.json"
    assert specs.exists() and accept.exists()
    # and both parse to the runner's expected shapes
    data = json.loads(specs.read_text())
    assert len(data["benchmarks"]) == 5
    acc = {k: v for k, v in json.loads(accept.read_text()).items() if not k.startswith("_")}
    assert set(acc) == set(ORGANIC_IDS)


def test_suite_flag_exists_in_cli():
    src = (REPO / "scripts" / "run_benchmarks.py").read_text()
    assert '"--suite"' in src and 'default=DEFAULT_SUITE' in src


# ── 3. min_faces scoring unit tests (fake geometry dicts, no builds) ────────────

def test_min_faces_pass():
    geom = {"solids": 1, "faces": 152, "bbox_mm": [100.0, 60.0, 3.0]}
    acc = run_benchmarks.score_acceptance(geom, {"min_faces": 100})
    assert acc["checks"] == {"min_faces": True}
    assert (acc["passed"], acc["total"]) == (1, 1)


def test_min_faces_fail_solid_plate():
    geom = {"solids": 1, "faces": 6, "bbox_mm": [100.0, 60.0, 3.0]}  # uncut plate
    acc = run_benchmarks.score_acceptance(geom, {"min_faces": 100})
    assert acc["checks"] == {"min_faces": False}
    assert (acc["passed"], acc["total"]) == (0, 1)


def test_min_faces_null_is_skipped_not_failed():
    geom = {"solids": 1, "faces": 6}
    acc = run_benchmarks.score_acceptance(geom, {"min_faces": None})
    assert "min_faces" not in acc["checks"]
    assert acc["total"] == 0 and acc["score"] is None


def test_min_faces_failed_build_scores_zero_of_n():
    # failed-build 0/N semantics: no geometry -> every applicable criterion fails
    acc = run_benchmarks.score_acceptance({}, {"solids": 1, "bbox_mm": [100, 60, 3], "min_faces": 100})
    assert acc["checks"]["min_faces"] is False
    assert acc["passed"] == 0 and acc["total"] == 3


def test_min_faces_alongside_existing_criteria():
    geom = {"solids": 1, "faces": 210, "cyl_faces": 1, "bbox_mm": [90.0, 90.0, 6.0]}
    acc = run_benchmarks.score_acceptance(
        geom, {"solids": 1, "bbox_mm": [90, 90, 6], "min_holes": 1, "min_faces": 100})
    assert acc["checks"] == {"solids": True, "bbox": True, "holes_cut": True, "min_faces": True}
    assert (acc["passed"], acc["total"]) == (4, 4)


# ── 4. Verified-gold: the two M10 corpus examples re-build deterministically ────

GOLD_MARKERS = {
    "sine_vase": "outer radius varies sinusoidally along the height",
    "twisted_hex": "twists 90 degrees over the height",
}


@pytest.fixture(scope="module")
def m10_gold_entries() -> dict:
    assert CORPUS.exists(), f"corpus missing: {CORPUS}"
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    found = {}
    for key, marker in GOLD_MARKERS.items():
        hits = [r for r in rows if marker in r.get("spec", "")]
        assert hits, f"M10 gold entry not found in corpus (marker: {marker!r})"
        found[key] = hits[-1]
    return found


def test_gold_entries_schema(m10_gold_entries):
    # exact same field set as the existing gold entries — retrieval treats them identically
    for key, e in m10_gold_entries.items():
        assert set(e.keys()) == {"spec", "code", "source", "rating", "verified", "timestamp", "teaches"}
        assert e["source"] == "gold" and e["rating"] == 5
        assert e["verified"]["solids"] == 1
        assert len(e["teaches"]) > 40, "teaches must describe the idiom, not name the part"


def _build_and_inspect(code: str, tmp_path: Path, name: str) -> dict:
    """Run gold code through scripts/step then scripts/inspect; return parsed FACTS_JSON."""
    src = tmp_path / f"{name}.py"
    step = tmp_path / f"{name}.step"
    src.write_text(code + "\n")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "step"), str(src), str(step)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"scripts/step failed for {name}: {r.stderr}"
    assert step.exists() and step.stat().st_size > 0
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "inspect"), str(step)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"scripts/inspect rejected {name}: {r.stdout}\n{r.stderr}"
    m = re.search(r"^FACTS_JSON: (\{.*\})$", r.stdout, re.M)
    assert m, f"no FACTS_JSON in inspect output for {name}"
    return json.loads(m.group(1))


@pytest.mark.parametrize("key", list(GOLD_MARKERS))
def test_gold_code_rebuilds_and_matches_verified(key, m10_gold_entries, tmp_path):
    entry = m10_gold_entries[key]
    facts = _build_and_inspect(entry["code"], tmp_path, key)
    assert facts["solids"] == 1
    exp_bbox = entry["verified"]["bbox_mm"]
    for got, exp in zip(sorted(facts["bbox"]), sorted(exp_bbox)):
        assert abs(got - exp) <= max(0.5, 0.01 * exp), \
            f"{key}: bbox {facts['bbox']} drifted from verified {exp_bbox}"
    exp_vol = entry["verified"]["volume_mm3"]
    assert abs(facts["volume"] - exp_vol) <= 0.01 * exp_vol, \
        f"{key}: volume {facts['volume']} drifted from verified {exp_vol}"

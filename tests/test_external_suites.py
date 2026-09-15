import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import fetch_external as fx


def _fake_cadprompt(root: Path, n: int = 3):
    for i in range(n):
        d = root / f"{i:08d}"; d.mkdir(parents=True)
        (d / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt").write_text(
            f"Create a cylinder of radius {0.5+i} units and height 1 unit.")
        (d / "Natural_Language_Descriptions_Prompt.txt").write_text("Create a cylinder.")
        (d / "Ground_Truth.stl").write_bytes(b"solid x\nendsolid x\n")
        (d / "Ground_Truth.json").write_text(json.dumps({"Number_of_Faces": 3, "Is_Solid": True}))
    return root


def test_cadprompt_to_suite_builds_specs_and_refs(tmp_path):
    src = _fake_cadprompt(tmp_path / "CADPrompt")
    out = tmp_path / "suite"
    n = fx.cadprompt_to_suite(src, out, limit=2, seed=7)
    specs = json.loads((out / "specs.json").read_text())
    acc = json.loads((out / "acceptance.json").read_text())
    assert n == 2 and len(specs) == 2
    s = specs[0]
    assert set(s) >= {"id", "name", "tier", "spec", "source"} and "units" in s["spec"]
    assert acc[s["id"]]["reference_stl"] == f"refs/{s['id']}.stl"
    assert (out / "refs" / f"{s['id']}.stl").exists()


def test_cadprompt_sampling_is_deterministic(tmp_path):
    src = _fake_cadprompt(tmp_path / "CADPrompt", n=6)
    a = fx.cadprompt_to_suite(src, tmp_path / "a", limit=3, seed=1)
    b = fx.cadprompt_to_suite(src, tmp_path / "b", limit=3, seed=1)
    ids = lambda p: [s["id"] for s in json.loads((p / "specs.json").read_text())]
    assert a == b == 3 and ids(tmp_path / "a") == ids(tmp_path / "b")


def test_arena_has_twelve_public_prompts_with_tiers():
    specs = fx.arena_specs()
    assert len(specs) == 12 and sorted({s["tier"] for s in specs}) == [1, 2, 3, 4]
    assert any("spur gear" in s["spec"] for s in specs)


def test_suite_slugs_cover_external_dirs():
    import harvest_census as hc
    assert {"cadprompt", "cad-arena", "text2cadquery"} <= set(hc.CARD_SUITES)


# Real upstream wording (benchmarks/external/cache/CAD_Code_Generation/CADPrompt/00000007/):
# every CADPrompt prompt is written for a code-generation task, not as a CAD spec.
_UPSTREAM = ("Write Python code using CADQuery to create a 3D model by extruding a circular "
             "sketch. The circle should have a radius of 0.75 units and the extrusion should "
             "be 0.20923 units high.")


def test_cadprompt_strips_the_cadquery_preamble():
    out = fx.strip_cadprompt_preamble(_UPSTREAM)
    assert not out.lower().startswith("write python code")
    assert out.startswith("Create a 3D model by extruding a circular sketch.")
    assert "radius of 0.75 units" in out           # the rest of the sentence is verbatim


def test_cadprompt_preamble_stripper_handles_the_article_and_case():
    assert fx.strip_cadprompt_preamble("write a python code using cadquery to make a box.") == "Make a box."
    # A prompt that never had the preamble is returned unchanged apart from stripping.
    assert fx.strip_cadprompt_preamble("  Create a cylinder. ") == "Create a cylinder."


def test_cadprompt_suite_specs_have_no_preamble_and_are_normalised(tmp_path):
    src = tmp_path / "CADPrompt"
    for i in range(2):
        d = src / f"{i:08d}"; d.mkdir(parents=True)
        (d / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt").write_text(_UPSTREAM)
        (d / "Ground_Truth.stl").write_bytes(b"solid x\nendsolid x\n")
    out = tmp_path / "suite"
    fx.cadprompt_to_suite(src, out, limit=2, seed=7)
    specs = json.loads((out / "specs.json").read_text())
    acc = json.loads((out / "acceptance.json").read_text())
    assert all(not s["spec"].lower().startswith("write python code") for s in specs)
    # normalized: DeepCAD units, not mm, so run_card scores these bands shape-only.
    assert all(v["normalized"] is True for v in acc.values())
    assert "Write Python code using CADQuery to" in (out / "README.md").read_text()


def test_generated_public_suites_are_clean_and_normalised():
    """The committed regeneration itself, not just the converter."""
    for suite in ("cadprompt", "text2cadquery"):
        d = HERE / "benchmarks" / suite
        if not (d / "specs.json").exists():
            continue
        specs = json.loads((d / "specs.json").read_text())
        acc = json.loads((d / "acceptance.json").read_text())
        assert specs and all(not s["spec"].lower().startswith("write python code") for s in specs)
        assert acc and all(v.get("normalized") is True for v in acc.values())


def test_suite_keys_are_distinct_per_spec_where_slugs_collapse():
    """The reason the contamination key moved off _slug: CADPrompt's 100 specs shared an
    opening, so 40-char slugs collapsed them into a couple of buckets."""
    import harvest_census as hc
    d = HERE / "benchmarks" / "cadprompt" / "specs.json"
    if not d.exists():
        return
    specs = json.loads(d.read_text())
    assert len({hc._key(s["spec"]) for s in specs}) == len(specs)


def test_key_is_whitespace_and_case_insensitive_but_not_truncating():
    import harvest_census as hc
    assert hc._key("A  Cube\n20mm") == hc._key("a cube 20mm")
    long_a = "a plate " + "x" * 200 + " with one hole"
    long_b = "a plate " + "x" * 200 + " with two holes"
    assert hc._slug(long_a, 40) == hc._slug(long_b, 40)    # the old key could not tell them apart
    assert hc._key(long_a) != hc._key(long_b)

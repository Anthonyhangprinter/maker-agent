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

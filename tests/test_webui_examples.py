"""Tests for the Examples gallery: 24 curated, owner-reviewed parts served as static assets
under webui/static/examples/. No engine call, no build lock, no subprocess — this is a pure
static-file contract: the index and every asset it points at must actually be there and be
internally consistent, or the page's gallery renders broken cards.

Run: python3 -m pytest tests/test_webui_examples.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app.app)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "webui" / "static" / "examples"
REQUIRED_KEYS = {"id", "title", "tier", "family", "difficulty", "spec", "dims_mm", "source"}
VALID_DIFFICULTY = {"starter", "intermediate", "advanced"}


def _load_index():
    return json.loads((EXAMPLES_DIR / "index.json").read_text())


def test_index_file_exists_and_has_24_entries():
    rows = _load_index()
    assert len(rows) == 24


def test_every_row_has_the_expected_fields():
    for row in _load_index():
        missing = REQUIRED_KEYS - row.keys()
        assert not missing, f"{row.get('id')} missing {missing}"
        assert row["tier"] in (1, 2, 3, 4)
        assert row["difficulty"] in VALID_DIFFICULTY
        assert row["spec"].strip()
        assert isinstance(row["dims_mm"], list) and len(row["dims_mm"]) == 3


def test_ids_are_unique():
    ids = [row["id"] for row in _load_index()]
    assert len(ids) == len(set(ids))


def test_every_row_ships_render_and_step_on_disk():
    for row in _load_index():
        d = EXAMPLES_DIR / row["id"]
        assert (d / "render.png").is_file(), row["id"]
        assert (d / "part.step").is_file(), row["id"]
        assert (d / "spec.txt").is_file(), row["id"]


MAX_STL_BYTES = 2 * 1024 * 1024   # "~2 MB" budget per example, so the 3D viewer stays snappy


def test_every_row_ships_a_part_stl_under_the_size_budget():
    # The example detail view uses the SAME three.js viewer as a creation thread, which
    # only knows how to load STL/GLB — every example needs its own part.stl alongside
    # part.step for that to work.
    for row in _load_index():
        stl = EXAMPLES_DIR / row["id"] / "part.stl"
        assert stl.is_file(), row["id"]
        assert 0 < stl.stat().st_size <= MAX_STL_BYTES, \
            f"{row['id']}: {stl.stat().st_size} bytes over the {MAX_STL_BYTES} budget"


def test_spec_txt_matches_the_index_entry():
    for row in _load_index():
        on_disk = (EXAMPLES_DIR / row["id"] / "spec.txt").read_text().strip()
        assert on_disk == row["spec"].strip(), row["id"]


# ── served through the app's own /static mount, the same route the page's JS fetches ──

def test_static_route_serves_the_index():
    r = client.get("/static/examples/index.json")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 24


def test_static_route_serves_a_render_and_a_step_for_every_id():
    rows = client.get("/static/examples/index.json").json()
    for row in rows:
        rid = row["id"]
        png = client.get(f"/static/examples/{rid}/render.png")
        assert png.status_code == 200, rid
        assert png.headers["content-type"].startswith("image/")
        step = client.get(f"/static/examples/{rid}/part.step")
        assert step.status_code == 200, rid


def test_static_route_serves_a_part_stl_for_every_id():
    # The same /static mount the page's three.js viewer fetches part.stl from.
    rows = client.get("/static/examples/index.json").json()
    for row in rows:
        rid = row["id"]
        stl = client.get(f"/static/examples/{rid}/part.stl")
        assert stl.status_code == 200, rid
        assert len(stl.content) <= MAX_STL_BYTES, rid


def test_static_route_404s_for_an_unknown_example():
    r = client.get("/static/examples/not-a-real-id/render.png")
    assert r.status_code == 404


def test_family_and_difficulty_cover_a_diverse_set():
    rows = _load_index()
    families = {row["family"] for row in rows}
    # the owner asked for these specific families to be represented among the 24
    for fam in ("gear", "cam", "yoke", "flange", "bracket", "pipe", "heat_sink",
                "turned_shaft", "housing"):
        assert fam in families, f"missing family {fam}"
    difficulties = {row["difficulty"] for row in rows}
    assert difficulties == VALID_DIFFICULTY, "expected a mix of starter/intermediate/advanced"

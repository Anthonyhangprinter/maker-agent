import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("lab_data", HERE / "lab" / "data.py")
ld = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ld
_spec.loader.exec_module(ld)


def _user_msg(spec: str) -> dict:
    return {
        "role": "user",
        "content": (
            "USER REQUEST (verbatim, every number here is AUTHORITATIVE):\n"
            f"{spec}\n\nNotes:\n- some technique references go here"
        ),
    }


def _row(spec: str, code: str = "from build123d import *\nresult = Box(10, 10, 10)",
         kind: str = "good") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a build123d expert."},
            _user_msg(spec),
            {"role": "assistant", "content": code},
        ],
        "kind": kind,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_extract_spec_pulls_text_between_header_and_blank_line():
    spec = ld.extract_spec(_user_msg("a 40x30x15mm open-top box with 2mm walls")["content"])
    assert spec == "a 40x30x15mm open-top box with 2mm walls"


def test_framing_matches_checkpoint_shape_exactly():
    # load_template(None) forces the built-in fallback so this test does not depend on the
    # checkpoint being downloaded on the machine running it. The fallback is verified (by
    # hand, against the real chat_template.jinja) to render this exact shape.
    template, from_checkpoint = ld.load_template(None)
    assert not from_checkpoint

    prompt = template.render(
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        bos_token="", add_generation_prompt=True,
    )
    assert prompt == (
        "<|turn>system\nSYS<turn|>\n"
        "<|turn>user\nUSER<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>"
    )


def test_render_pairs_output_rows_have_four_keys(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [_row("a plain 10mm cube")])
    template, _ = ld.load_template(None)

    kept, dropped = ld.render_pairs(src, tmp_path / "out.jsonl", keys=set(), slugs=set(),
                                     template=template, tag="t")

    assert dropped == []
    assert len(kept) == 1
    row = kept[0]
    assert set(row.keys()) == {"prompt", "completion", "id", "kind"}
    assert row["id"] == "t-0000"
    assert row["kind"] == "good"
    assert row["prompt"].endswith("<|channel>thought\n<channel|>")
    assert row["completion"] == "from build123d import *\nresult = Box(10, 10, 10)<turn|>\n"

    written = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert written == kept


def test_contamination_exact_key_drops_planted_spec_keeps_clean_one():
    template, _ = ld.load_template(None)
    contaminated_spec = "a 100x60x30mm enclosure with 2mm walls"
    clean_spec = "a 20mm diameter 15mm tall cylinder"
    src_rows = [_row(contaminated_spec), _row(clean_spec)]

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        src = tdp / "train.jsonl"
        _write_jsonl(src, src_rows)

        planted_keys = {ld.hc._key(contaminated_spec)}
        kept, dropped = ld.render_pairs(src, tdp / "out.jsonl", keys=planted_keys, slugs=set(),
                                         template=template, tag="t")

    assert len(kept) == 1
    assert len(dropped) == 1
    dropped_id, reason = dropped[0]
    assert dropped_id == "t-0000"
    assert "exact suite match" in reason
    assert contaminated_spec in reason


def test_near_duplicate_slug_drops_only_when_unique_in_its_suite():
    template, _ = ld.load_template(None)
    spec_a = "a bracket with a 40mm slot for mounting hardware here"
    spec_b = "a bracket with a 40mm slot for mounting hardware elsewhere"  # same 40-char slug
    # Both specs share a 40-character opening, so the slug seen at render time is the same
    # for both. If that slug were still ambiguous (identifying more than one suite spec) it
    # would not be trusted as evidence; here the injected `slugs` set represents a slug that
    # the suites confirm identifies exactly ONE spec, so any training row sharing that
    # opening is treated as a near duplicate and dropped.
    slug = ld.hc._slug(spec_a, 40)
    assert slug == ld.hc._slug(spec_b, 40)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        src = tdp / "train.jsonl"
        _write_jsonl(src, [_row(spec_a)])
        kept, dropped = ld.render_pairs(src, tdp / "out.jsonl", keys=set(), slugs={slug},
                                         template=template, tag="t")
        assert kept == []
        assert len(dropped) == 1
        assert "near-duplicate" in dropped[0][1]

        # Same spec, but the slug is NOT in the (now empty) unique-slug set: kept.
        kept2, dropped2 = ld.render_pairs(src, tdp / "out2.jsonl", keys=set(), slugs=set(),
                                           template=template, tag="t")
        assert len(kept2) == 1
        assert dropped2 == []


def test_stats_shape_percentile_helper():
    assert ld._pctl([], 50) == 0.0
    assert ld._pctl([10], 50) == 10.0
    assert ld._pctl([1, 2, 3, 4, 5], 50) == 3.0
    p95 = ld._pctl(list(range(1, 101)), 95)
    assert 94 <= p95 <= 96


def test_scan_special_tokens_flags_rows_that_already_contain_framing(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [_row("clean spec"), _row("a spec with <|turn> already in it")])
    hits = ld.scan_special_tokens(src)
    assert hits == [f"{src.name}#1"]

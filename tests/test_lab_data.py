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


def test_extract_spec_recognises_target_part_header():
    # revise_script's GIFT-FAIL repair prompt (cad_engine.py:1892): spec is inline, on the
    # same line as the header, not on a following line.
    content = (
        "Target part: a 100x60x30mm enclosure with 2mm walls\n\n"
        "Problem to fix:\nresult produced no output\n\n"
        "Current code:\n```python\nfrom build123d import *\n```\n\n"
        "Return the complete fixed script:"
    )
    assert ld.extract_spec(content) == "a 100x60x30mm enclosure with 2mm walls"


def test_extract_spec_recognises_user_request_lowercase_header():
    # decide_or_edit's agent-loop revise turn (cad_engine.py:1934): also inline.
    content = (
        "User request: a 20mm diameter 15mm tall cylinder\n\n"
        "Current code:\n```python\nfrom build123d import *\n```\n\n"
        "Geometry report:\nsolids=1\n\n"
        "Visual critique:\n(visual critique unavailable)\n\n"
        "Reply DONE if correct, otherwise return the corrected full script:"
    )
    assert ld.extract_spec(content) == "a 20mm diameter 15mm tall cylinder"


def test_extract_spec_returns_empty_for_unrecognised_header():
    assert ld.extract_spec("Some other prompt shape entirely:\nno known header here") == ""


def test_render_pairs_fails_closed_when_no_header_matches(tmp_path):
    row = {
        "messages": [
            {"role": "system", "content": "You are a build123d expert."},
            {"role": "user", "content": "Some other prompt shape entirely:\nno known header"},
            {"role": "assistant", "content": "from build123d import *\nresult = Box(1, 1, 1)"},
        ],
        "kind": "good",
    }
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [row])
    template, _ = ld.load_template(None, allow_fallback=True)

    kept, dropped = ld.render_pairs(src, tmp_path / "out.jsonl", keys=set(), slugs=set(),
                                     template=template, tag="t")

    # Fail closed: a row whose spec cannot be extracted is dropped, never kept unchecked.
    assert kept == []
    assert dropped == [("t-0000", "no-spec-header")]
    assert (tmp_path / "out.jsonl").read_text() == ""


def test_framing_matches_checkpoint_shape_exactly():
    # load_template(None, allow_fallback=True) forces the built-in fallback so this test does
    # not depend on the checkpoint being downloaded on the machine running it (the flag is
    # mandatory since finding 9: a missing checkpoint template is an error by default). The fallback is verified (by
    # hand, against the real chat_template.jinja) to render this exact shape.
    template, from_checkpoint = ld.load_template(None, allow_fallback=True)
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
    template, _ = ld.load_template(None, allow_fallback=True)

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
    template, _ = ld.load_template(None, allow_fallback=True)
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
    template, _ = ld.load_template(None, allow_fallback=True)
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


def test_make_completion_does_not_double_append_turn_end():
    assert ld._make_completion("CODE HERE") == "CODE HERE<turn|>\n"
    # Reproduces the review's exact repro case: content already ending in the marker must
    # not get a second one appended.
    assert ld._make_completion("CODE HERE<turn|>\n") == "CODE HERE<turn|>\n"
    assert ld._make_completion("CODE HERE<turn|>") == "CODE HERE<turn|>\n"
    # Trailing whitespace between the code and an existing marker is also collapsed away,
    # not preserved as a gap before the single closing marker.
    assert ld._make_completion("CODE HERE  <turn|>  \n") == "CODE HERE<turn|>\n"


def test_render_pairs_completion_does_not_double_append_turn_end(tmp_path):
    row = _row("a plain 10mm cube", code="from build123d import *\nresult = Box(10, 10, 10)<turn|>\n")
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [row])
    template, _ = ld.load_template(None, allow_fallback=True)

    kept, dropped = ld.render_pairs(src, tmp_path / "out.jsonl", keys=set(), slugs=set(),
                                     template=template, tag="t")

    assert dropped == []
    assert kept[0]["completion"] == "from build123d import *\nresult = Box(10, 10, 10)<turn|>\n"
    assert kept[0]["completion"].count("<turn|>") == 1


def test_scan_special_tokens_flags_rows_that_already_contain_framing(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [_row("clean spec"), _row("a spec with <|turn> already in it")])
    hits = ld.scan_special_tokens(src)
    assert hits == [f"{src.name}#1"]


def test_load_template_refuses_the_fallback_unless_it_is_asked_for(tmp_path):
    """finding 9: a moved/renamed/never-downloaded checkpoint template used to fall back to
    the built-in framing with one printed line, so an unattended rerun could train a whole
    corpus on a framing the checkpoint never used."""
    import pytest

    with pytest.raises(SystemExit, match="chat template not found"):
        ld.load_template(tmp_path / "missing" / "chat_template.jinja")
    template, from_checkpoint = ld.load_template(tmp_path / "missing" / "chat_template.jinja",
                                                 allow_fallback=True)
    assert from_checkpoint is False
    assert template is not None


def test_load_template_reads_a_real_checkpoint_template(tmp_path):
    path = tmp_path / "chat_template.jinja"
    path.write_text("{{ messages[0]['content'] }}")
    template, from_checkpoint = ld.load_template(path)
    assert from_checkpoint is True
    assert template.render(messages=[{"role": "user", "content": "HI"}]) == "HI"


def test_contamination_slug_uniqueness_is_counted_per_suite(monkeypatch):
    """finding 13: run_card.contamination() counts slug uniqueness inside each suite, so a
    slug appearing once in suite A and once in suite B is evidence in BOTH. The old global
    count called it ambiguous and let such a row through, i.e. this guard was more permissive
    than the one it claims to mirror."""
    shared = "a-bracket-slug"
    monkeypatch.setattr(ld.hc, "suite_keys", lambda: {"deadbeef"})
    monkeypatch.setattr(ld.hc, "suite_slug_counts", lambda n=40: {
        "suite-a": {shared: 1, "degenerate": 28},
        "suite-b": {shared: 1},
    })
    keys, slugs = ld.default_contamination_sets()
    assert keys == {"deadbeef"}
    assert shared in slugs
    assert "degenerate" not in slugs

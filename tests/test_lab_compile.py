"""Offline tests for lab/compile.py (Task 4, the Phase 3 compiler).

Everything here uses temp dirs for pairs.jsonl/upgrades.jsonl/systems/val_specs.json/
out-dir -- never lab/state/ itself (a GPU harvest may be appending to the real
lab/state/pairs.jsonl while this suite runs; these tests must not read or touch it).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

_spec = importlib.util.spec_from_file_location("lab_compile", HERE / "lab" / "compile.py")
lc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lc
_spec.loader.exec_module(lc)

_data_spec = importlib.util.spec_from_file_location("lab_data_for_compile_tests",
                                                     HERE / "lab" / "data.py")
ld = importlib.util.module_from_spec(_data_spec)
sys.modules[_data_spec.name] = ld
_data_spec.loader.exec_module(ld)

_train_spec = importlib.util.spec_from_file_location("lab_train_for_compile_tests",
                                                      HERE / "lab" / "train.py")
lt = importlib.util.module_from_spec(_train_spec)
sys.modules[_train_spec.name] = lt
_train_spec.loader.exec_module(lt)


CURRENT_GV = lc.current_gate_version()


import re

_SPECIAL_TOKEN_RE = re.compile(r"(<\|turn>|<turn\|>|<\|channel>|<channel\|>)")


class _FakeTokenizer:
    """Stand-in for the real HF tokenizer, same contract lab/train.py's own tests use
    (tests/test_lab_train.py's FakeTokenizer), extended to treat the fallback
    template's own framing tokens (<|turn>, <turn|>, <|channel>, <channel|>) as atomic,
    the way a real tokenizer's added_special_tokens always split them regardless of
    surrounding whitespace -- needed here because the template's own
    add_generation_prompt block ends "<channel|>" with no boundary space before the
    completion begins, which a plain whitespace-split tokenizer would otherwise glue
    into one token and wrongly report the prompt as not a prefix of prompt+completion.
    Defined locally (not imported cross-file) so this test module stays self-contained."""

    bos_token_id = 2

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id_for(self, piece: str) -> int:
        return self._vocab.setdefault(piece, 100 + len(self._vocab))

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = []
        for piece in _SPECIAL_TOKEN_RE.split(text):
            if not piece:
                continue
            if _SPECIAL_TOKEN_RE.fullmatch(piece):
                ids.append(self._id_for(piece))
            else:
                ids.extend(self._id_for(w) for w in piece.split())
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_system(systems_dir: Path, text: str) -> str:
    systems_dir.mkdir(parents=True, exist_ok=True)
    sha1 = hashlib.sha1(text.encode()).hexdigest()
    (systems_dir / f"{sha1}.txt").write_text(text, encoding="utf-8")
    return sha1


SYS_TEXT = "You are a build123d expert, write ALGEBRA MODE code."


def _pair(id_, spec_id, spec, tier, confirm_strength, system_sha1, *, kind="good",
         turn="first", code="from build123d import *\nresult = Box(1, 1, 1)",
         gate_version=None, ts="2026-09-19T00:00:00+00:00") -> dict:
    return {
        "id": id_, "spec_id": spec_id, "spec": spec, "tier": tier, "group": "misc",
        "source": "student", "kind": kind, "turn": turn, "band": "match",
        "arm": "gemma-4-31b", "model": "local:gemma-4-31b", "temperature": 0.2,
        "system_sha1": system_sha1,
        "prompt": (f"USER REQUEST (verbatim, every number here is AUTHORITATIVE):\n"
                  f"{spec}\n\nNotes:\n- none"),
        "code": code, "bad_code": None, "problem": None, "facts": {}, "signature": {},
        "confirmed_by": "reference" if confirm_strength == "reference" else "agreement",
        "confirm_strength": confirm_strength, "code_similarity": 0.0,
        "gate_version": gate_version or CURRENT_GV, "verified": {}, "ts": ts, "unit_id": "u1",
    }


@pytest.fixture()
def lab_state(tmp_path):
    """A temp lab/state look-alike: pairs.jsonl, upgrades.jsonl, systems/, val_specs.json
    (not yet created), plus a fresh out-dir. sys1 is pre-written so most tests don't need
    to think about system rehydration."""
    state = tmp_path / "state"
    systems_dir = state / "systems"
    sys1 = _write_system(systems_dir, SYS_TEXT)
    return {
        "dir": state,
        "pairs_file": state / "pairs.jsonl",
        "upgrades_file": state / "upgrades.jsonl",
        "systems_dir": systems_dir,
        "val_specs_file": state / "val_specs.json",
        "out_dir": tmp_path / "out",
        "sys1": sys1,
    }


def _compile(lab_state, pairs, *, strengths=("reference", "cross_pass"), upgrades=None,
            exclude_ids_file=None, max_per_spec=2, val_frac=0.0, seed=1,
            keys=None, slugs=None, dry_run=False, allow_fallback_template=True):
    _write_jsonl(lab_state["pairs_file"], pairs)
    if upgrades is not None:
        _write_jsonl(lab_state["upgrades_file"], upgrades)
    template, _ = ld.load_template(None, allow_fallback=True)
    return lc.compile_round(
        round_n=1, out_dir=lab_state["out_dir"], strengths=list(strengths),
        pairs_file=lab_state["pairs_file"], upgrades_file=lab_state["upgrades_file"],
        systems_dir=lab_state["systems_dir"], val_specs_file=lab_state["val_specs_file"],
        exclude_ids_file=exclude_ids_file, max_per_spec=max_per_spec, val_frac=val_frac,
        seed=seed, keys=keys if keys is not None else set(),
        slugs=slugs if slugs is not None else set(),
        template_path=None if allow_fallback_template else ld.DEFAULT_TEMPLATE,
        allow_fallback_template=allow_fallback_template, dry_run=dry_run,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Strength filter + upgrades
# ---------------------------------------------------------------------------

def test_strength_filter_honours_an_upgraded_row(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:aaa", "s1", "a 10mm cube", 1, "same_pass", sys1),
        _pair("p:s2:bbb", "s2", "a 20mm cube", 1, "same_pass", sys1,
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    upgrades = [{"pair_id": "p:s1:aaa", "confirm_strength": "cross_pass"}]
    meta = _compile(lab_state, pairs, strengths=("reference", "cross_pass"), upgrades=upgrades)

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert [r["pair_id"] for r in train] == ["p:s1:aaa"]
    assert train[0]["confirm_strength"] == "cross_pass"
    assert meta["dropped_by_reason"].get("strength-excluded") == 1


# ---------------------------------------------------------------------------
# Salvage turn exclusion
# ---------------------------------------------------------------------------

def test_salvage_turn_rows_are_excluded(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:aaa", "s1", "a 10mm cube", 1, "reference", sys1, turn="first"),
        _pair("p:s2:bbb", "s2", "a 20mm cube", 1, "reference", sys1, turn="salvage",
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    meta = _compile(lab_state, pairs, strengths=("reference",))

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert [r["pair_id"] for r in train] == ["p:s1:aaa"]
    assert meta["dropped_by_reason"].get("turn-not-first") == 1


# ---------------------------------------------------------------------------
# --exclude-ids
# ---------------------------------------------------------------------------

def test_exclude_ids_file_is_honoured(lab_state, tmp_path):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:aaa", "s1", "a 10mm cube", 1, "reference", sys1),
        _pair("p:s2:bbb", "s2", "a 20mm cube", 1, "reference", sys1,
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("# hand-audit rejections\np:s2:bbb  # geometrically wrong\n", encoding="utf-8")

    meta = _compile(lab_state, pairs, strengths=("reference",), exclude_ids_file=exclude)

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert [r["pair_id"] for r in train] == ["p:s1:aaa"]
    assert meta["dropped_by_reason"].get("excluded-by-audit") == 1


# ---------------------------------------------------------------------------
# Missing system file
# ---------------------------------------------------------------------------

def test_missing_system_file_is_dropped_and_reported(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:aaa", "s1", "a 10mm cube", 1, "reference", sys1),
        _pair("p:s2:bbb", "s2", "a 20mm cube", 1, "reference", "deadbeefdeadbeefdeadbeef",
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    meta = _compile(lab_state, pairs, strengths=("reference",))

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert [r["pair_id"] for r in train] == ["p:s1:aaa"]
    assert meta["dropped_by_reason"].get("missing-system-file") == 1


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------

def test_contamination_drop_on_planted_card_suite_spec(lab_state):
    sys1 = lab_state["sys1"]
    contaminated_spec = "a 100x60x30mm enclosure with 2mm walls"
    clean_spec = "a 20mm diameter 15mm tall cylinder"
    pairs = [
        _pair("p:s1:aaa", "s1", contaminated_spec, 1, "reference", sys1),
        _pair("p:s2:bbb", "s2", clean_spec, 1, "reference", sys1,
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    planted_keys = {ld.hc._key(contaminated_spec)}
    meta = _compile(lab_state, pairs, strengths=("reference",), keys=planted_keys, slugs=set())

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert [r["pair_id"] for r in train] == ["p:s2:bbb"]
    assert meta["dropped_by_reason"].get("contamination") == 1


# ---------------------------------------------------------------------------
# Dedup by code fingerprint
# ---------------------------------------------------------------------------

def test_dedup_by_normalised_code_fingerprint(lab_state):
    sys1 = lab_state["sys1"]
    code_a = "from build123d import *\nx = Box(5, 5, 5)\nresult = x"
    code_b = "from build123d import *\ny = Box(5.0, 5.0, 5.0)\nresult = y"  # same modulo rename + float
    pairs = [
        _pair("p:s1:aaa", "s1", "a 5mm cube", 1, "reference", sys1, code=code_a,
             ts="2026-09-19T00:00:00+00:00"),
        _pair("p:s1:bbb", "s1", "a 5mm cube", 1, "reference", sys1, code=code_b,
             ts="2026-09-19T00:00:01+00:00"),
    ]
    meta = _compile(lab_state, pairs, strengths=("reference",), max_per_spec=5)

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert len(train) == 1
    assert train[0]["pair_id"] == "p:s1:aaa"  # earlier ts wins the dedup keep
    assert meta["dropped_by_reason"].get("dedup") == 1


# ---------------------------------------------------------------------------
# Per-spec cap
# ---------------------------------------------------------------------------

def test_max_per_spec_cap(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair(f"p:s1:{i}", "s1", "a 5mm cube", 1, "reference", sys1,
             code=f"from build123d import *\nresult = Box(5, 5, {5 + i})",
             ts=f"2026-09-19T00:00:0{i}+00:00")
        for i in range(3)
    ]
    meta = _compile(lab_state, pairs, strengths=("reference",), max_per_spec=2)

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    assert len(train) == 2
    assert meta["dropped_by_reason"].get("max-per-spec-cap") == 1


# ---------------------------------------------------------------------------
# Spec-level split, no leakage, tier stratification
# ---------------------------------------------------------------------------

def test_spec_level_split_no_leakage_and_tier_stratified(lab_state):
    sys1 = lab_state["sys1"]
    pairs = []
    for t in (1, 2):
        for i in range(10):
            spec_id = f"s{t}-{i}"
            pairs.append(_pair(f"p:{spec_id}:x", spec_id, f"a tier {t} spec {i}", t,
                               "reference", sys1,
                               code=f"from build123d import *\nresult = Box(1, 1, {t}0 + {i})"))
    meta = _compile(lab_state, pairs, strengths=("reference",), val_frac=0.2, seed=7)

    train = _read_jsonl(lab_state["out_dir"] / "train.jsonl")
    val = _read_jsonl(lab_state["out_dir"] / "val.jsonl")
    train_specs = {r["spec_id"] for r in train}
    val_specs = {r["spec_id"] for r in val}
    assert train_specs.isdisjoint(val_specs)
    assert len(val_specs) == 4  # 20% of 10 specs per tier, both tiers represented
    val_tiers = {r["tier"] for r in val}
    assert val_tiers == {1, 2}
    assert meta["val_specs_count"] == 4


# ---------------------------------------------------------------------------
# Frozen val reuse
# ---------------------------------------------------------------------------

def test_val_specs_frozen_and_reused_across_runs(lab_state):
    sys1 = lab_state["sys1"]
    round1_pairs = [
        _pair(f"p:s{i}:x", f"s{i}", f"a spec {i}", 1, "reference", sys1,
             code=f"from build123d import *\nresult = Box(1, 1, {i})")
        for i in range(10)
    ]
    _compile(lab_state, round1_pairs, strengths=("reference",), val_frac=0.3, seed=3)
    first_val_specs = json.loads(lab_state["val_specs_file"].read_text(encoding="utf-8"))

    # A later run adds new specs; the frozen file must be read verbatim, never redrawn.
    more_pairs = round1_pairs + [
        _pair(f"p:s{i}:x", f"s{i}", f"a spec {i}", 1, "reference", sys1,
             code=f"from build123d import *\nresult = Box(1, 1, {i})")
        for i in range(10, 15)
    ]
    meta2 = _compile(lab_state, more_pairs, strengths=("reference",), val_frac=0.3, seed=3)
    second_val_specs = json.loads(lab_state["val_specs_file"].read_text(encoding="utf-8"))

    assert first_val_specs == second_val_specs
    assert meta2["val_specs_source"] == "reused"
    # every new spec (10..14) defaults to train, never silently promoted into val
    val = _read_jsonl(lab_state["out_dir"] / "val.jsonl")
    val_specs = {r["spec_id"] for r in val}
    assert val_specs == set(first_val_specs["val_specs"])
    assert not (val_specs & {f"s{i}" for i in range(10, 15)})


# ---------------------------------------------------------------------------
# data_meta counts, exact
# ---------------------------------------------------------------------------

def test_data_meta_counts_are_exact(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:a", "s1", "a 10mm cube", 1, "reference", sys1),
        _pair("p:s2:a", "s2", "a 20mm cube", 2, "reference", sys1,
             code="from build123d import *\nresult = Box(2, 2, 2)"),
        _pair("p:s3:a", "s3", "a 30mm cube", 3, "same_pass", sys1,
             code="from build123d import *\nresult = Box(3, 3, 3)"),
    ]
    meta = _compile(lab_state, pairs, strengths=("reference", "cross_pass"), val_frac=0.0)

    assert meta["counts"]["train"]["total"] == 2
    assert meta["counts"]["train"]["by_tier_strength"] == {"1|reference": 1, "2|reference": 1}
    assert meta["counts"]["val"]["total"] == 0
    assert meta["dropped_by_reason"] == {"strength-excluded": 1}
    assert meta["rows_total_seen"] == 3


# ---------------------------------------------------------------------------
# Output accepted downstream: lab/data.py's own renderer (used internally), and
# lab/train.py's real load_rows/mask_example/build_dataset on the actual output file.
# ---------------------------------------------------------------------------

def test_output_rows_are_accepted_by_the_real_downstream_loader(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [
        _pair("p:s1:a", "s1", "a 10mm cube", 1, "reference", sys1),
        _pair("p:s2:a", "s2", "a 20mm cube", 2, "reference", sys1,
             code="from build123d import *\nresult = Box(2, 2, 2)"),
    ]
    _compile(lab_state, pairs, strengths=("reference",), val_frac=0.0)

    train_path = lab_state["out_dir"] / "train.jsonl"
    rows = _read_jsonl(train_path)
    assert len(rows) == 2
    for r in rows:
        assert set(r) >= {"prompt", "completion", "id", "kind", "spec_id", "tier",
                          "confirm_strength", "pair_id", "source"}
        assert r["prompt"].endswith("<|channel>thought\n<channel|>")
        assert r["completion"].endswith("<turn|>\n")
        assert r["source"] == "harvest-round1"

    # The real lab/train.py loader/masker (Phase 2's own trainer) accepts these rows
    # unmodified -- this is the "accepted by lab/data.py's loader" proof end to end,
    # through the actual consumer the compiled round is built for.
    loaded = lt.load_rows(train_path)
    assert len(loaded) == 2

    tok = _FakeTokenizer()
    kept, dropped = lt.build_dataset(loaded, tok, max_seq=5120)
    assert dropped == 0
    assert len(kept) == 2
    for ex in kept:
        assert ex["input_ids"][0] == tok.bos_token_id
        assert any(l != -100 for l in ex["labels"])  # the completion is unmasked


def test_dry_run_writes_nothing(lab_state):
    sys1 = lab_state["sys1"]
    pairs = [_pair("p:s1:a", "s1", "a 10mm cube", 1, "reference", sys1)]
    meta = _compile(lab_state, pairs, strengths=("reference",), dry_run=True)

    assert meta["dry_run"] is True
    assert not lab_state["out_dir"].exists()
    assert not lab_state["val_specs_file"].exists()
    assert meta["counts"]["train"]["total"] == 1

"""Offline tests for lab/train.py's pure functions (no torch/unsloth import).

Loaded by file path (not `import lab.train`) so this test never needs
lab/.venv's training deps on the machine running the plain repo test suite --
only the module's pure functions (load_rows, mask_example, build_dataset,
count_trainable_params) are exercised; main() and its heavy imports are never
called here.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("lab_train", HERE / "lab" / "train.py")
lt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lt
_spec.loader.exec_module(lt)


class FakeTokenizer:
    """A minimal stand-in for the real HF tokenizer: whitespace-splits text
    into word ids from a fixed vocabulary, and prepends exactly one BOS id
    when add_special_tokens=True -- the one behaviour mask_example() relies on
    (see lab/train.py's module docstring for why the real Gemma 4 tokenizer
    needs `add_bos_token = True` set explicitly to do the same thing).
    """

    bos_token_id = 2

    def __init__(self, double_bos: bool = False):
        self._vocab: dict[str, int] = {}
        self._double_bos = double_bos

    def _id_for(self, word: str) -> int:
        # ids start at 100 so they never collide with bos_token_id (2)
        return self._vocab.setdefault(word, 100 + len(self._vocab))

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        # split() on runs of whitespace (never an empty trailing token) --
        # callers must leave a boundary space between prompt and completion
        # (real prompt strings end with "\n", same idea) or two words glue
        # into one token, same as a real BPE tokenizer merging across a join
        # with no separator.
        ids = [self._id_for(w) for w in text.split()]
        if add_special_tokens:
            prefix = [self.bos_token_id, self.bos_token_id] if self._double_bos else [self.bos_token_id]
            ids = prefix + ids
        return {"input_ids": ids}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# load_rows
# ---------------------------------------------------------------------------


def test_load_rows_reads_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"prompt": "p1", "completion": "c1", "id": "a-0000", "kind": "good"}) + "\n"
        "\n"
        + json.dumps({"prompt": "p2", "completion": "c2", "id": "a-0001", "kind": "good"}) + "\n"
    )
    rows = lt.load_rows(path)
    assert [r["id"] for r in rows] == ["a-0000", "a-0001"]


# ---------------------------------------------------------------------------
# mask_example
# ---------------------------------------------------------------------------


def test_mask_example_masks_prompt_and_keeps_completion_ids():
    tok = FakeTokenizer()
    prompt = "system says do the thing "
    completion = "here is the code"
    ex = lt.mask_example(prompt, completion, tok, max_seq=100)

    prompt_ids = tok(prompt)["input_ids"]  # includes BOS, matches what mask_example computes
    full_ids = tok(prompt + completion)["input_ids"]

    assert ex["input_ids"] == full_ids
    assert ex["attention_mask"] == [1] * len(full_ids)
    # every prompt-region label is masked...
    assert ex["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    # ...and every completion-region label is the real token id, not masked
    assert ex["labels"][len(prompt_ids):] == full_ids[len(prompt_ids):]
    assert -100 not in ex["labels"][len(prompt_ids):]
    # exactly one BOS, at position 0
    assert ex["input_ids"][0] == tok.bos_token_id
    assert ex["input_ids"][1] != tok.bos_token_id


def test_mask_example_drops_rows_over_max_seq():
    tok = FakeTokenizer()
    full_len = len(tok("prompt words here " + "more completion words", add_special_tokens=True)["input_ids"])
    assert lt.mask_example("prompt words here ", "more completion words", tok, max_seq=full_len) is not None
    assert lt.mask_example("prompt words here ", "more completion words", tok, max_seq=full_len - 1) is None


def test_mask_example_raises_on_double_bos():
    tok = FakeTokenizer(double_bos=True)
    with pytest.raises(ValueError, match="second BOS"):
        lt.mask_example("a b c", "d e f", tok, max_seq=100)


def test_mask_example_raises_when_prompt_not_a_prefix():
    class DriftingTokenizer(FakeTokenizer):
        def __call__(self, text, add_special_tokens=True):
            # Simulate a boundary re-merge: tokenizing the joined string yields
            # different ids for the prompt region than tokenizing it alone.
            out = super().__call__(text, add_special_tokens=add_special_tokens)
            if "DRIFT" in text and add_special_tokens:
                out["input_ids"][1] = 999999
            return out

    tok = DriftingTokenizer()
    with pytest.raises(ValueError, match="prefix"):
        lt.mask_example("DRIFT prompt", "completion", tok, max_seq=100)


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_reports_kept_and_dropped_counts():
    tok = FakeTokenizer()
    rows = [
        {"prompt": "short prompt ", "completion": "short completion", "id": "r0", "kind": "good"},
        {"prompt": "a much longer prompt with many many more words in it than the other one ",
         "completion": "and a much longer completion too, with plenty more words padded in as well",
         "id": "r1", "kind": "good"},
    ]
    # max_seq sized to fit only the first (short) row
    short_len = len(tok(rows[0]["prompt"] + rows[0]["completion"])["input_ids"])
    kept, dropped = lt.build_dataset(rows, tok, max_seq=short_len)

    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["id"] == "r0"
    assert kept[0]["kind"] == "good"
    assert set(kept[0].keys()) == {"input_ids", "attention_mask", "labels", "id", "kind"}


def test_build_dataset_keeps_all_rows_when_none_exceed_max_seq():
    tok = FakeTokenizer()
    rows = [
        {"prompt": "one two three ", "completion": "four five six", "id": "r0", "kind": "good"},
        {"prompt": "seven eight ", "completion": "nine ten", "id": "r1", "kind": "fail"},
    ]
    kept, dropped = lt.build_dataset(rows, tok, max_seq=1000)
    assert dropped == 0
    assert [ex["id"] for ex in kept] == ["r0", "r1"]


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------


class _FakeParam:
    def __init__(self, n: int, requires_grad: bool):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def __init__(self, named: list[tuple[str, _FakeParam]]):
        self._named = named

    def named_parameters(self):
        return iter(self._named)


def test_count_trainable_params_splits_vision_from_language():
    model = _FakeModel([
        ("model.language_model.layers.0.q_proj.lora_A", _FakeParam(100, True)),
        ("model.language_model.layers.0.q_proj.lora_B", _FakeParam(50, True)),
        ("model.vision_tower.blocks.0.proj.lora_A", _FakeParam(10, True)),
        ("model.multi_modal_projector.lora_A", _FakeParam(5, True)),
        ("model.language_model.layers.0.base_weight", _FakeParam(9999, False)),  # frozen, excluded
    ])
    counts = lt.count_trainable_params(model)
    assert counts == {"vision": 15, "language": 150}


def test_count_trainable_params_all_zero_when_nothing_trainable():
    model = _FakeModel([("anything", _FakeParam(1234, False))])
    assert lt.count_trainable_params(model) == {"vision": 0, "language": 0}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_argparser_defaults():
    args = lt.build_argparser().parse_args(["--base", "B", "--data", "D", "--out", "O"])
    assert args.rank == 16
    assert args.epochs == 1
    assert args.max_seq == 5120
    assert args.save_steps == 25
    assert args.resume is None
    assert args.max_steps is None


def test_argparser_resume_bare_flag_is_true():
    args = lt.build_argparser().parse_args(["--base", "B", "--data", "D", "--out", "O", "--resume"])
    assert args.resume is True


def test_argparser_resume_with_path():
    args = lt.build_argparser().parse_args(
        ["--base", "B", "--data", "D", "--out", "O", "--resume", "lab/runs/spike1/checkpoint-25"]
    )
    assert args.resume == "lab/runs/spike1/checkpoint-25"


def test_argparser_max_steps_override():
    args = lt.build_argparser().parse_args(
        ["--base", "B", "--data", "D", "--out", "O", "--max-steps", "3"]
    )
    assert args.max_steps == 3

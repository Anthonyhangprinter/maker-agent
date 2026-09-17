"""Offline tests for lab/train.py's pure functions (no unsloth/transformers/trl import).

Loaded by file path (not `import lab.train`) so this test never needs
lab/.venv's training deps on the machine running the plain repo test suite --
only the module's pure functions (load_rows, mask_example, build_dataset,
count_trainable_params) are exercised without any heavy import at all;
main() and its unsloth/transformers/trl imports are never called here.

chunked_completion_loss() is the one exception: it needs bare torch (CPU is
fine, no unsloth/transformers/trl/GPU) to verify the chunked loss maths
against a full-sequence reference computation. Bare torch is normally
available on this machine's system Python (see CLAUDE.md), but those tests
are still skipped via pytest.importorskip rather than failing hard if it
ever isn't, so the rest of this file's genuinely offline tests are unaffected.
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


def test_argparser_eval_only_and_adapter():
    args = lt.build_argparser().parse_args(
        ["--base", "B", "--data", "D", "--out", "O", "--eval-only", "--adapter", "lab/runs/spike1/adapter"]
    )
    assert args.eval_only is True
    assert args.adapter == "lab/runs/spike1/adapter"


def test_argparser_eval_only_defaults_false_and_adapter_none():
    args = lt.build_argparser().parse_args(["--base", "B", "--data", "D", "--out", "O"])
    assert args.eval_only is False
    assert args.adapter is None


# ---------------------------------------------------------------------------
# chunked_completion_loss -- bare torch only (CPU), no unsloth/transformers/trl.
# Verifies the chunked loss exactly reproduces a full, unchunked cross-entropy
# computed the way Gemma4ForConditionalGeneration.forward() computes its own
# (shift by one position, ignore_index=-100, optional cap*tanh(x/cap)
# softcapping applied before the loss) -- see eval_loss()'s docstring for why
# lab/train.py never gets to run that full computation directly on the real
# 248k-vocab model without OOMing.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")


def _reference_shifted_cross_entropy(hidden, weight, labels, softcap=None):
    """The full, unchunked computation chunked_completion_loss() must match."""
    shift_labels = labels[1:]
    logits = F.linear(hidden[:-1], weight).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    loss = F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")
    tokens = int((shift_labels != -100).sum().item())
    return loss.item(), tokens


def test_chunked_completion_loss_matches_full_cross_entropy():
    torch.manual_seed(0)
    seq_len, hidden_size, vocab = 37, 8, 13  # seq_len not a multiple of chunk_size, on purpose
    hidden = torch.randn(seq_len, hidden_size)
    weight = torch.randn(vocab, hidden_size)  # nn.Linear-shaped weight (out_features, in_features)
    head = lambda h: F.linear(h, weight)  # noqa: E731
    labels = torch.randint(0, vocab, (seq_len,))
    labels[:5] = -100  # mask a prefix, like mask_example() would for the prompt region

    chunk_loss, chunk_tokens = lt.chunked_completion_loss(hidden, head, labels, chunk_size=8)
    ref_loss, ref_tokens = _reference_shifted_cross_entropy(hidden, weight, labels)

    assert chunk_tokens == ref_tokens
    assert abs(chunk_loss - ref_loss) < 1e-4


def test_chunked_completion_loss_matches_full_cross_entropy_with_softcap():
    torch.manual_seed(1)
    seq_len, hidden_size, vocab = 20, 6, 9
    hidden = torch.randn(seq_len, hidden_size)
    weight = torch.randn(vocab, hidden_size)
    head = lambda h: F.linear(h, weight)  # noqa: E731
    labels = torch.randint(0, vocab, (seq_len,))
    cap = 30.0

    chunk_loss, chunk_tokens = lt.chunked_completion_loss(hidden, head, labels, chunk_size=6, softcap=cap)
    ref_loss, ref_tokens = _reference_shifted_cross_entropy(hidden, weight, labels, softcap=cap)

    assert chunk_tokens == ref_tokens
    assert abs(chunk_loss - ref_loss) < 1e-4


def test_chunked_completion_loss_excludes_all_ignored_positions():
    torch.manual_seed(2)
    seq_len, hidden_size, vocab = 10, 4, 5
    hidden = torch.randn(seq_len, hidden_size)
    weight = torch.randn(vocab, hidden_size)
    head = lambda h: F.linear(h, weight)  # noqa: E731
    labels = torch.full((seq_len,), -100, dtype=torch.long)

    loss, tokens = lt.chunked_completion_loss(hidden, head, labels, chunk_size=4)

    assert tokens == 0
    assert loss == 0.0


def test_chunked_completion_loss_ignores_only_masked_positions_within_a_chunk():
    # A chunk that mixes real and -100 labels must count/score only the real ones.
    torch.manual_seed(3)
    seq_len, hidden_size, vocab = 12, 4, 6
    hidden = torch.randn(seq_len, hidden_size)
    weight = torch.randn(vocab, hidden_size)
    head = lambda h: F.linear(h, weight)  # noqa: E731
    labels = torch.randint(0, vocab, (seq_len,))
    # Mask every other post-shift label so every chunk mixes real and ignored.
    labels[1::2] = -100

    chunk_loss, chunk_tokens = lt.chunked_completion_loss(hidden, head, labels, chunk_size=3)
    ref_loss, ref_tokens = _reference_shifted_cross_entropy(hidden, weight, labels)

    assert chunk_tokens == ref_tokens
    assert abs(chunk_loss - ref_loss) < 1e-4

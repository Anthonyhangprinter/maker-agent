#!/usr/bin/env python3
"""lab/train.py -- Unsloth QLoRA fine-tune of the CAD coder base model.

Phase 2 training spike (Maker Agent 1.0): a feasibility QLoRA fine-tune of
Gemma-4-31B-it (the Phase 0 shootout winner) on the gate-verified SFT corpus
lab/data.py rendered (lab/data/train.jsonl + val.jsonl, {"prompt", "completion",
"id", "kind"} rows, already in the checkpoint's own Gemma 4 chat-template framing).

Tokenization contract (important, do not "simplify" this):
  - lab/data.py rendered `prompt` WITHOUT the literal `<bos>` text (the checkpoint's
    chat_template.jinja emits it itself at generation time via `{{ bos_token }}`);
    this script must therefore make the *tokenizer* add exactly one BOS when it
    tokenizes the raw prompt/completion strings, not apply a chat template again.
  - The checkpoint's tokenizer_config.json has NO `add_bos_token` key at all, so
    plain `tokenizer(text)` adds nothing (finding 20: an earlier version of this
    note claimed the file shipped `add_bos_token=False`, which it does not).
    Setting `tokenizer.add_bos_token = True` after load works because this is a
    Gemma tokenizer class, whose `add_bos_token` setter rewrites the
    post-processor; it makes `tokenizer(text, add_special_tokens=True)` prepend
    exactly one BOS. Nothing here trusts that silently: mask_example() asserts it
    per row, so a base model whose tokenizer class lacks that setter fails loudly
    on row 1 instead of training on unframed text.
  - `prompt + completion` is tokenized as ONE string so the tokenizer sees the
    real token boundary (tokenizing prompt and completion separately and
    concatenating ids can split a token across the join, e.g. mid-BPE-merge,
    which would silently feed the model junk at every prompt/completion seam).
  - Loss is completion-only: labels are -100 over every prompt token, and the
    real token ids over the completion (mask_example() builds this by finding
    len(prompt_ids) as a prefix of the full sequence and slicing).
  - Rows whose full tokenized length exceeds --max-seq are DROPPED, never
    truncated -- truncating a completion mid-script would train the model to
    stop emitting code partway through, exactly the failure mode this spike is
    trying to fix, not reproduce. build_dataset() reports the drop count so it
    is visible, not silent.

Pure, torch-free functions (load_rows, mask_example, build_dataset,
count_trainable_params) are importable and unit-testable without unsloth/torch
installed -- see tests/test_lab_train.py. Every unsloth/torch/trl import is
deferred into main() so importing this module never requires the training venv.

Usage:
    lab/.venv/bin/python lab/train.py --base <4bit checkpoint dir> --data lab/data \
        --out lab/runs/spike1 --rank 16 --epochs 1 --max-seq 5120 --save-steps 25

Smoke test (a few optimizer steps, not a real epoch):
    lab/.venv/bin/python lab/train.py --base <dir> --data lab/data --out lab/runs/smoke \
        --max-steps 3 --save-steps 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure functions -- no torch/unsloth/transformers import here on purpose, so
# tests/test_lab_train.py can exercise them on a machine/venv without the
# training deps installed (a "fake tokenizer" duck-types the one method used).
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    """Load a {"prompt", "completion", "id", "kind"} JSONL file into a list of dicts.

    Blank lines are skipped (lab/data.py never writes them, but a hand-edited
    file might).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def base_from_adapter_config(adapter_dir: str | Path) -> str:
    """Read `base_model_name_or_path` out of `<adapter_dir>/adapter_config.json`.

    This is the exact same field Unsloth's `FastModel.from_pretrained()`
    itself resolves the real base model from when `model_name` points at a
    saved LoRA adapter (see `unsloth/models/loader.py`'s `is_peft` branch,
    which reads a `PeftConfig` and uses `peft_config.base_model_name_or_path`
    to find and load the base before attaching the adapter). Used only to
    default `--base` for `--eval-only`'s own recorded metadata when the
    caller omits it -- the actual model load never needs this value passed
    explicitly, Unsloth resolves it internally from the same file.

    Raises FileNotFoundError / KeyError with the normal, informative
    messages if `adapter_dir` is not a saved PEFT adapter directory.
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def mask_example(prompt: str, completion: str, tokenizer: Any, max_seq: int) -> dict | None:
    """Tokenize prompt+completion as one string and mask the prompt out of the loss.

    Returns {"input_ids", "attention_mask", "labels"} (plain python lists, not
    tensors -- the caller wraps them in a datasets.Dataset), or None if the
    tokenized length exceeds max_seq (the row must be dropped, never truncated).

    `tokenizer` only needs to support `tokenizer(text, add_special_tokens=True)
    -> {"input_ids": [...]}` and a `bos_token_id` attribute, so a lightweight
    fake stands in for the real HF tokenizer in tests.

    Raises ValueError if the tokenizer does not add exactly one BOS at position
    0, or if the prompt's own tokenization is not a token-level prefix of the
    full prompt+completion tokenization -- both would mean the completion-only
    mask below is wrong, so this fails loudly rather than silently training on
    a misaligned label sequence.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]

    if not full_ids or full_ids[0] != tokenizer.bos_token_id:
        raise ValueError("expected exactly one BOS token at position 0 of the tokenized sequence")
    if len(full_ids) > 1 and full_ids[1] == tokenizer.bos_token_id:
        raise ValueError("found a second BOS token at position 1 -- tokenizer is double-adding BOS")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt tokenization is not a token-level prefix of prompt+completion "
            "tokenization; the completion-only label mask below would be misaligned"
        )

    if len(full_ids) > max_seq:
        return None

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_dataset(rows: list[dict], tokenizer: Any, max_seq: int) -> tuple[list[dict], int]:
    """Tokenize+mask every row, dropping any that exceed max_seq.

    Returns (kept, dropped_count). `kept` rows carry `id`/`kind` alongside the
    tokenized fields so a later inspection can trace a training example back to
    its source spec; SFTConfig(remove_unused_columns=True) strips them before
    the forward pass.
    """
    kept = []
    dropped = 0
    for row in rows:
        ex = mask_example(row["prompt"], row["completion"], tokenizer, max_seq)
        if ex is None:
            dropped += 1
            continue
        ex["id"] = row.get("id")
        ex["kind"] = row.get("kind")
        kept.append(ex)
    return kept, dropped


_VISION_KEYWORDS = ("vision", "visual", "tower", "multi_modal")


def count_trainable_params(model: Any) -> dict[str, int]:
    """Sum trainable parameter counts, split by whether the module name looks
    like a vision component. Used to assert the LoRA touched language layers
    only (finetune_vision_layers=False), since a silent regression there would
    burn VRAM/compute on parameters this spike has no vision training data for.
    """
    counts = {"vision": 0, "language": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bucket = "vision" if any(k in name.lower() for k in _VISION_KEYWORDS) else "language"
        counts[bucket] += param.numel()
    return counts


def compute_metric_row(step: int, loss: float, learning_rate: float | None, elapsed: float,
                        tokens_seen: int, last_step: int, last_tokens: int,
                        last_time: float | None, now: float, peak_vram_gb: float,
                        reserved_vram_gb: float) -> dict:
    """Pure delta/cumulative throughput math for one MetaLogger.on_log() row.

    Split out of MetaLogger itself (a nested TrainerCallback inside main(),
    since it needs live torch.cuda.* VRAM figures every call and so cannot be
    exercised outside lab/.venv on real hardware) so the actual arithmetic --
    the part review findings #4/#5 were about -- is independently testable
    with plain floats/ints, no CUDA context needed.

    `step`/`tokens_seen` are the trainer's own cumulative counters
    (state.global_step / logs["num_input_tokens_seen"]), which keep counting
    across a --resume (they are restored from the checkpoint). `last_step` /
    `last_tokens` / `last_time` are the values from the PREVIOUS on_log call
    -- or, for the first call after a fresh start OR a --resume, whatever
    on_train_begin seeded them to (0/0/t0 on a fresh run; the checkpoint's own
    resumed global_step/num_input_tokens_seen and t0 on a resume).

    Returns a row with both:
    - `s_per_step` / `tokens_per_s`: DELTA-based, i.e. divided by
      `now - last_time` and `step - last_step` / `tokens_seen - last_tokens`.
      Correct immediately after a --resume, when step/tokens_seen are large
      pre-resume-inclusive counts but this invocation's own elapsed time is
      small -- a plain elapsed/step there would read as a nonsensical burst
      of steps in a few seconds.
    - `s_per_step_cum` / `tokens_per_s_cum`: the previous (pre-fix) behaviour,
      divided by `elapsed`/`step`/`tokens_seen` since THIS invocation's own
      start -- kept for continuity/comparison, but wrong (reads low) for the
      first several rows after any resume.
    """
    step_delta = step - last_step
    token_delta = tokens_seen - last_tokens
    time_delta = (now - last_time) if last_time is not None else elapsed
    return {
        "step": step,
        "loss": loss,
        "learning_rate": learning_rate,
        "elapsed_s": elapsed,
        "s_per_step": (time_delta / step_delta) if step_delta > 0 else None,
        "tokens_per_s": (token_delta / time_delta) if time_delta > 0 else None,
        "s_per_step_cum": (elapsed / step) if step else None,
        "tokens_per_s_cum": (tokens_seen / elapsed) if elapsed > 0 else None,
        "peak_vram_gb": peak_vram_gb,
        "reserved_vram_gb": reserved_vram_gb,
    }


def chunked_completion_loss(hidden_states: Any, head: Any, labels: Any, chunk_size: int,
                             softcap: float | None = None) -> tuple[float, int]:
    """Shifted causal cross-entropy over `hidden_states`, computed a chunk of
    positions at a time so the (chunk, vocab) fp32 logits tensor stays small
    regardless of sequence length -- the point of chunking is to never
    materialise a (seq_len, vocab) fp32 tensor at once (a ~5000-token row at
    Gemma 4's 248k vocab needs ~4.3GB for that in one shot, which is exactly
    what OOM'd both the plain trainer.evaluate() path and a bare
    labels-forward call on this 24GB card -- see eval_loss()'s docstring).

    hidden_states: (T, H) tensor (bf16 or fp32), a single sequence, no batch
        dimension.
    head: callable taking (n, H) -> (n, V) logits, e.g. an nn.Linear
        (model.get_output_embeddings()) or a tied-embedding fallback.
    labels: (T,) int64 tensor, -100 at positions to ignore (mask_example()'s
        convention).
    chunk_size: number of (post-shift) positions to score per chunk.
    softcap: Gemma 4's `final_logit_softcapping` value, or None. Applied as
        `cap * tanh(x / cap)` before the loss -- the exact formula
        Gemma4ForConditionalGeneration.forward() applies to its own logits --
        so this is numerically the same loss the model's own labels-forward
        would compute, just chunked.

    Returns (summed_loss, token_count) -- SUMMED not meaned, so callers can
    token-weight-average across multiple sequences of different lengths.
    Verified against a full, unchunked cross-entropy on random tensors in
    tests/test_lab_train.py (agrees within 1e-4; also verifies -100 positions
    never contribute to either the loss or the token count).
    """
    import torch
    import torch.nn.functional as F

    seq_len = hidden_states.shape[0]
    shift_labels = labels[1:]
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, seq_len - 1, chunk_size):
        end = min(start + chunk_size, seq_len - 1)
        label_chunk = shift_labels[start:end]
        n_chunk = int((label_chunk != -100).sum().item())
        if n_chunk == 0:
            continue
        h_chunk = hidden_states[start:end]
        logits = head(h_chunk).float()
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        loss = F.cross_entropy(logits, label_chunk, ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += n_chunk
        del logits, loss, h_chunk
    return total_loss, total_tokens


def eval_loss(model: Any, tokenizer: Any, val_rows: list[dict], max_seq: int,
              chunk_size: int = 512) -> tuple[float | None, int]:
    """Token-weighted mean completion-only loss over the val split.

    This is a full replacement for `trainer.evaluate()`, not a tuned-up
    version of it -- two prior smoke tests confirmed evaluate() OOMs on this
    24GB card (it re-wraps the model for its own eval pass via
    accelerate, which upcasts the full-vocab logits SFTTrainer's own
    compute_loss always materialises to fp32), and a third confirmed that
    calling the bare model directly with `labels=` still OOMs too -- Gemma 4's
    248k vocab makes even ONE fp32 (seq_len, vocab) logits tensor ~4.3GB for a
    ~5000-token row, and torch.cuda.empty_cache() beforehand does not help
    (the process legitimately holds ~21GB of real training-time allocations,
    there is nothing spare to reclaim). So this function never asks the model
    for logits at all: it pulls hidden states from the text decoder directly
    (no LM head applied), then applies the head and cross-entropy itself in
    `chunk_size`-position slices via chunked_completion_loss() -- the same
    numbers, at a small fraction of the peak memory.

    `model` must be the raw, unwrapped model (e.g. `trainer.model` right
    after `trainer.train()`, or the model FastModel.from_pretrained() /
    PeftModel.from_pretrained() hands back for --eval-only) -- never
    `trainer.model_wrapped` or anything passed through trainer.evaluate().

    Resolves the text-only decoder and output head defensively rather than
    hardcoding Gemma4ForConditionalGeneration's attribute names, since `model`
    may be wrapped in a PEFT LoraModel: tries `model.get_decoder()` first
    (PreTrainedModel's generic lookup, forwarded through PEFT's
    attribute-proxying __getattr__ chain, correctly resolves to the
    Gemma4TextModel instance for this architecture -- verified by reading
    transformers/models/gemma4/modeling_gemma4.py's base_model_prefix="model"
    and Gemma4Model.language_model), and falls back to explicit attribute
    paths if that ever returns the model itself (a sign the generic lookup
    failed silently). Both paths print which one was used. The output head
    similarly prefers `model.get_output_embeddings()`, which returns Gemma4's
    real `lm_head` nn.Linear (its weight tensor is tied to the input
    embeddings via `_tied_weights_keys`, but it is still the actual module
    the model's own forward calls, so using it here matches exactly, LoRA or
    not -- our own LoRA config does not target lm_head anyway) and falls back
    to `F.linear` against the input embedding weight only if
    get_output_embeddings() is ever None (fully tied architectures with no
    separate head module at all).

    Returns (mean_loss_or_None, total_completion_tokens). Rows mask_example()
    drops for exceeding max_seq are skipped, same as build_dataset() would.
    """
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    def _resolve_text_decoder(m):
        top = m
        try:
            decoder = m.get_decoder()
            if decoder is not top:
                print(f"[eval] text decoder via model.get_decoder() -> {type(decoder).__name__}")
                return decoder
            print("[eval] model.get_decoder() returned the model itself (unhelpful); trying explicit paths")
        except Exception as exc:
            print(f"[eval] model.get_decoder() raised {exc!r}; trying explicit paths")

        candidates = [
            ("base_model.model.model.language_model", lambda x: x.base_model.model.model.language_model),
            ("base_model.model.language_model", lambda x: x.base_model.model.language_model),
            ("base_model.model.model", lambda x: x.base_model.model.model),
            ("model.language_model", lambda x: x.model.language_model),
            ("model.model", lambda x: x.model.model),
        ]
        for name, getter in candidates:
            try:
                decoder = getter(top)
            except AttributeError:
                continue
            if decoder is not top:
                print(f"[eval] text decoder via {name} -> {type(decoder).__name__}")
                return decoder
        raise RuntimeError("eval_loss: could not resolve a text-only decoder module from the model")

    def _resolve_output_head(m):
        head = m.get_output_embeddings()
        if head is not None:
            print(f"[eval] output head via model.get_output_embeddings() -> {type(head).__name__}")
            return head
        print("[eval] get_output_embeddings() is None (tied embeddings); using input embedding weight via F.linear")
        weight = m.get_input_embeddings().weight
        return lambda h: torch.nn.functional.linear(h, weight)

    decoder = _resolve_text_decoder(model)
    head = _resolve_output_head(model)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    print(f"[eval] final_logit_softcapping={softcap}")

    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for row in val_rows:
                ex = mask_example(row["prompt"], row["completion"], tokenizer, max_seq)
                if ex is None:
                    continue
                labels_list = ex["labels"]
                if all(label == -100 for label in labels_list):
                    continue
                input_ids = torch.tensor([ex["input_ids"]], device=device)
                attention_mask = torch.tensor([ex["attention_mask"]], device=device)
                labels = torch.tensor(labels_list, device=device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                hidden = out.last_hidden_state[0]
                del out

                row_loss, row_tokens = chunked_completion_loss(hidden, head, labels, chunk_size, softcap)
                total_loss += row_loss
                total_tokens += row_tokens
                del hidden, labels
    finally:
        if was_training:
            model.train()
    mean_loss = (total_loss / total_tokens) if total_tokens else None
    return mean_loss, total_tokens


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=None,
                    help="path to the 4-bit base checkpoint dir; required for training, "
                         "optional for --eval-only (defaults from the adapter's own "
                         "adapter_config.json base_model_name_or_path when omitted)")
    p.add_argument("--data", required=True, help="dir containing train.jsonl and val.jsonl")
    p.add_argument("--out", required=True, help="output dir for checkpoints, adapter, and metrics")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (lora_alpha is set equal to it)")
    p.add_argument("--epochs", type=float, default=1, help="num_train_epochs")
    p.add_argument("--max-seq", type=int, default=5120,
                    help="rows whose tokenized prompt+completion exceed this are DROPPED, not truncated")
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume training: bare flag resumes from the latest checkpoint in --out, "
                         "or pass an explicit checkpoint path")
    p.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke-test use only; overrides --epochs)")
    p.add_argument("--eval-only", action="store_true",
                    help="skip training: load --adapter over the base model and just run eval_loss(), "
                         "writing <adapter parent dir>/eval_meta.json")
    p.add_argument("--adapter", default=None,
                    help="adapter dir (from a previous run's <out>/adapter) to load for --eval-only")
    return p


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    args = build_argparser().parse_args(argv)
    if not args.eval_only and not args.base:
        raise SystemExit("--base is required (unless --eval-only is used with --adapter)")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred to here so the pure functions above stay importable
    # (and unit-testable) without the lab/.venv training deps installed.
    # unsloth must be imported before trl/transformers/peft (it patches them at
    # import time) -- import it first among the heavy deps, per unsloth's own
    # startup warning.
    import importlib.metadata as importlib_metadata

    from unsloth import FastModel

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if args.eval_only:
        if not args.adapter:
            raise SystemExit("--eval-only requires --adapter <dir>")
        # Unsloth's FastModel.from_pretrained() auto-detects an adapter_config.json
        # in model_name: it reads the PeftConfig, resolves the real base model from
        # its base_model_name_or_path, loads that, then attaches the adapter via
        # PeftModel.from_pretrained(..., is_trainable=True) itself (verified by
        # reading unsloth/models/loader.py's is_peft branch) -- so pointing
        # model_name at the adapter dir is the whole load; --base is never read
        # for loading in this branch. It is optional here and, when omitted,
        # is resolved from the adapter's own adapter_config.json purely so
        # eval_meta.json has a real value in its "base" field instead of null.
        if not args.base:
            args.base = base_from_adapter_config(args.adapter)
            print(f"[eval-only] --base not given; resolved from {args.adapter}/adapter_config.json: {args.base}")
        print(f"[eval-only] loading base+adapter from: {args.adapter}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=args.max_seq,
            load_in_4bit=True,
            dtype=None,
        )
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        tokenizer.add_bos_token = True

        data_dir = Path(args.data)
        val_rows = load_rows(data_dir / "val.jsonl")
        # Report kept/dropped the same way the training path does (via
        # build_dataset(), the exact same mask_example()-per-row predicate
        # eval_loss() itself uses to skip a row): run 5 reported "rows: 16"
        # here against training's "val rows: kept=15 dropped=1" for the same
        # val.jsonl and --max-seq, because this printed len(val_rows) (every
        # row loaded from disk, unfiltered) instead of how many rows actually
        # contributed to eval_loss's sum. eval_loss()'s own per-row loop was
        # never wrong -- it already skips a row mask_example() drops -- only
        # this reporting line was inconsistent with the training path's.
        val_kept, val_dropped = build_dataset(val_rows, tokenizer, args.max_seq)
        print(f"[eval-only] val rows: kept={len(val_kept)} dropped={val_dropped} (max_seq={args.max_seq})")

        t_eval0 = time.time()
        eval_loss_val, eval_tokens = eval_loss(model, tokenizer, val_rows, args.max_seq)
        eval_seconds = time.time() - t_eval0

        eval_meta = {
            "adapter": str(Path(args.adapter).resolve()),
            "base": args.base,
            "max_seq": args.max_seq,
            "eval_loss": eval_loss_val,
            "eval_tokens": eval_tokens,
            "rows": len(val_kept),
            "rows_dropped": val_dropped,
            "seconds": eval_seconds,
        }
        eval_meta_path = Path(args.adapter).resolve().parent / "eval_meta.json"
        eval_meta_path.write_text(json.dumps(eval_meta, indent=2))
        print(f"[eval-only] wrote {eval_meta_path}")
        print(json.dumps(eval_meta, indent=2))
        return

    print(f"[train] loading base model: {args.base}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    # Gemma 4 is multimodal; from_pretrained can hand back a processor rather than
    # a bare tokenizer. Unwrap it -- everything here tokenizes plain text, no images.
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    # See the module docstring: the checkpoint's tokenizer config ships
    # add_bos_token=False, but lab/data.py's rendered prompts assume the
    # tokenizer will add it. mask_example() re-verifies this per row, so a
    # regression here fails loudly at the first row rather than silently
    # training on unprefixed sequences.
    tokenizer.add_bos_token = True

    # target_modules is deliberately left at unsloth's default (None), NOT
    # "all-linear" as a literal reading of the original brief suggested.
    # unsloth/models/vision.py's FastModel.get_peft_model has a special case
    # that fires on the literal string target_modules == "all-linear" and
    # unconditionally forces finetune_vision_layers = finetune_language_layers
    # = finetune_attention_modules = finetune_mlp_modules =
    # finetune_audio_layers = True, overriding whatever the caller passes for
    # those flags. Passing target_modules="all-linear" together with
    # finetune_vision_layers=False here would have that flag silently flipped
    # back to True, defeating the language-only requirement (and burning
    # VRAM/compute on the vision tower with no vision training data at all).
    # Leaving target_modules=None takes vision.py's target_modules-is-None
    # branch instead, which respects the caller's finetune_* flags as given --
    # functionally equivalent to "all-linear" but correctly scoped to language
    # layers. The `assert param_counts["vision"] == 0` right below is the
    # guard for this: if a future unsloth version ever changes this behaviour
    # (or this comment's understanding of it is wrong), that assert fails
    # loudly the first time this runs, instead of silently training the
    # vision tower.
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
    )

    param_counts = count_trainable_params(model)
    print(f"[train] trainable params: language={param_counts['language']:,} vision={param_counts['vision']:,}")
    assert param_counts["vision"] == 0, (
        "LoRA touched vision-tower parameters despite finetune_vision_layers=False "
        f"(counted {param_counts['vision']:,} trainable vision params)"
    )

    data_dir = Path(args.data)
    train_rows = load_rows(data_dir / "train.jsonl")
    val_rows = load_rows(data_dir / "val.jsonl")
    train_kept, train_dropped = build_dataset(train_rows, tokenizer, args.max_seq)
    val_kept, val_dropped = build_dataset(val_rows, tokenizer, args.max_seq)
    print(f"[train] train rows: kept={len(train_kept)} dropped={train_dropped} (max_seq={args.max_seq})")
    print(f"[train] val rows: kept={len(val_kept)} dropped={val_dropped}")
    if not train_kept:
        raise SystemExit("no training rows survived the --max-seq filter -- nothing to train on")

    train_ds = Dataset.from_list(train_kept)
    val_ds = Dataset.from_list(val_kept) if val_kept else None

    sft_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        # SFTConfig defaults per_device_eval_batch_size to 8; sequences here run
        # up to ~5000 tokens, so an 8-row eval batch OOMs even though training
        # itself (batch size 1) fits comfortably -- match the train batch size.
        # Found the hard way: the first smoke test trained all 3 steps cleanly
        # then OOMed inside trainer.evaluate() on the unset default.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim="adamw_8bit",
        max_length=args.max_seq,
        seed=3407,
        dataloader_num_workers=0,
        # The dataset is already tokenized+masked by build_dataset() above; tell
        # SFTTrainer not to re-run its own chat-template/tokenization pipeline on
        # it (it would also auto-detect this from the `input_ids` column, but
        # being explicit documents the intent).
        dataset_kwargs={"skip_prepare_dataset": True},
        # Lets the built-in Trainer.log() report cumulative non-padding tokens
        # seen (logs["num_input_tokens_seen"]), which MetaLogger below turns
        # into a running tokens/s figure -- real tokens actually fed to the
        # model, not raw batch x seq_len including padding.
        include_num_input_tokens_seen="non_padding",
    )
    if args.max_steps:
        sft_kwargs["max_steps"] = args.max_steps
    config = SFTConfig(**sft_kwargs)

    meta_path = out_dir / "train_meta.jsonl"

    class MetaLogger(TrainerCallback):
        """Appends one JSON line per on_log call (logging_steps=1, so per
        optimizer step): step, loss, lr, elapsed time, DELTA-based s/step and
        tokens/s (since the previous logged row, or since this invocation's
        own start for the first row), the same figures as CUMULATIVE-since-
        this-invocation for continuity, plus current VRAM.

        Resume-safe (review findings #4/#5 on an earlier version of this
        class): a --resume run's state.global_step and
        state.num_input_tokens_seen are restored from the checkpoint and
        continue their pre-resume-inclusive cumulative counts, but this
        process's own elapsed clock (self.t0) starts fresh at zero. A plain
        elapsed/step or tokens_seen/elapsed on those cumulative counters would
        divide a small per-invocation elapsed time by a large pre-resume
        count, producing bogus throughput numbers on the first several rows
        after any resume -- that is why s_per_step/tokens_per_s below are
        deltas since the previous on_log call (seeded from the checkpoint's
        own resumed step/token counts in on_train_begin, so even the FIRST
        post-resume row's delta is correct, not measured against zero).
        """

        def __init__(self, path: Path, resuming: bool):
            self.path = path
            self.t0: float | None = None
            self.last_step = 0
            self.last_tokens = 0
            self.last_time: float | None = None
            # A --resume run into the same --out dir must not discard the
            # metrics rows recorded before the resume point -- only truncate
            # a stale train_meta.jsonl on a genuinely fresh run.
            if not resuming:
                self.path.write_text("")

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            self.last_time = self.t0
            # Seed the delta trackers from the checkpoint's own resumed counts
            # (0 on a fresh run, since global_step/num_input_tokens_seen both
            # start at 0) so the first on_log after a resume computes its
            # delta against this session's own progress, not against zero.
            self.last_step = state.global_step
            self.last_tokens = state.num_input_tokens_seen

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or self.t0 is None:
                return
            now = time.time()
            elapsed = now - self.t0
            step = state.global_step
            tokens_seen = logs.get("num_input_tokens_seen", 0)

            row = compute_metric_row(
                step=step,
                loss=logs["loss"],
                learning_rate=logs.get("learning_rate"),
                elapsed=elapsed,
                tokens_seen=tokens_seen,
                last_step=self.last_step,
                last_tokens=self.last_tokens,
                last_time=self.last_time,
                now=now,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                reserved_vram_gb=torch.cuda.memory_reserved() / 2**30,
            )
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")

            self.last_step = step
            self.last_tokens = tokens_seen
            self.last_time = now

    resume = args.resume if args.resume else None

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[MetaLogger(meta_path, resuming=resume is not None)],
    )

    train_result = trainer.train(resume_from_checkpoint=resume)

    # Save the adapter and write a first cut of train_meta.json BEFORE eval,
    # so a from-scratch custom eval implementation crashing (or OOMing) can
    # never cost the trained adapter or the training-side metrics that are
    # already known good -- only eval_loss/eval_tokens are filled in later.
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] adapter saved to {adapter_dir}")

    def _version(pkg: str) -> str:
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return "unknown"

    meta = {
        "args": vars(args),
        "dataset": {
            "train_kept": len(train_kept),
            "train_dropped": train_dropped,
            "val_kept": len(val_kept),
            "val_dropped": val_dropped,
        },
        "trainable_params": param_counts,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.time() - t_start,
        "steps": trainer.state.global_step,
        "train_loss": getattr(train_result, "training_loss", None),
        "eval_loss": None,
        "eval_tokens": 0,
        "versions": {
            "torch": torch.__version__,
            "unsloth": _version("unsloth"),
            "transformers": _version("transformers"),
            "trl": _version("trl"),
            "peft": _version("peft"),
            "bitsandbytes": _version("bitsandbytes"),
        },
    }
    meta_json_path = out_dir / "train_meta.json"
    meta_json_path.write_text(json.dumps(meta, indent=2))
    print(f"[train] wrote pre-eval {meta_json_path} (adapter + training metrics are now safe on disk)")

    if val_rows:
        # trainer.evaluate() is deliberately not used here -- see eval_loss()'s
        # docstring for the full diagnosis (it OOMs on this 24GB card by
        # re-wrapping the model for its own eval pass, which upcasts the
        # full-vocab logits SFTTrainer always computes to fp32). Pass
        # trainer.model explicitly: it is the raw, unwrapped model even though
        # `model` (this function's own local variable) already refers to the
        # same object, to make the "must not be the wrapped model" contract
        # obvious at the call site.
        print("[train] running custom completion-only eval loss (bypassing trainer.evaluate())")
        eval_loss_val, eval_tokens = eval_loss(trainer.model, tokenizer, val_rows, args.max_seq)
        meta["eval_loss"] = eval_loss_val
        meta["eval_tokens"] = eval_tokens
        meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        meta_json_path.write_text(json.dumps(meta, indent=2))
        print(f"[train] eval_loss={eval_loss_val} eval_tokens={eval_tokens}")
    else:
        print("[train] no val rows -- skipping eval_loss")

    print(f"[train] wrote {meta_json_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

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
  - The checkpoint's tokenizer config ships `add_bos_token=False` (verified against
    the on-disk tokenizer_config.json), so plain `tokenizer(text)` adds nothing.
    Setting `tokenizer.add_bos_token = True` after load makes `tokenizer(text,
    add_special_tokens=True)` prepend exactly one BOS -- verified empirically
    against all 369 real train+val rows (see mask_example()'s assertions, which
    re-check this on every row rather than trusting it silently).
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


def eval_loss(model: Any, tokenizer: Any, val_rows: list[dict], max_seq: int) -> tuple[float | None, int]:
    """Token-weighted mean completion-only loss over the val split.

    Calls `model` directly (the caller must pass the raw, unwrapped model --
    e.g. `trainer.model` right after `trainer.train()`, never
    `trainer.model_wrapped` or `trainer.evaluate()`) so accelerate's bf16
    forward wrapper never runs. That wrapper upcasts every forward call's
    output tensors -- including the full-vocab (248k) logits SFTTrainer's own
    compute_loss always materialises, to log entropy/token accuracy -- to
    fp32; a single ~5000-token row needs ~4.3GB of scratch for that alone.
    `trainer.evaluate()` re-wraps the model internally for its own eval pass
    and OOMs on exactly that allocation once training has left the CUDA
    caching allocator fragmented near the 24GB ceiling (confirmed by
    traceback across two separate smoke tests, one before and one after
    matching per_device_eval_batch_size to the train batch size, and a third
    where torch.cuda.empty_cache() first was tried and did not help -- the
    process legitimately holds ~21GB at that point, there is nothing spare to
    reclaim). Calling the bare model directly is the same forward path
    training already proved fits.

    Uses this module's own mask_example() so the val loss is computed with
    exactly the same completion-only masking as training, not the previous
    trainer.evaluate() path (which used the same masking but the wrapped
    model). Requires torch to already be imported into this module's globals
    (main() does `global torch; import torch` before calling this).

    Returns (mean_loss_or_None, total_completion_tokens). Rows that mask_seq
    drops for exceeding max_seq are skipped, same as build_dataset() would.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for row in val_rows:
                ex = mask_example(row["prompt"], row["completion"], tokenizer, max_seq)
                if ex is None:
                    continue
                n_tokens = sum(1 for label in ex["labels"] if label != -100)
                if n_tokens == 0:
                    continue
                input_ids = torch.tensor([ex["input_ids"]], device=device)
                attention_mask = torch.tensor([ex["attention_mask"]], device=device)
                labels = torch.tensor([ex["labels"]], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += out.loss.item() * n_tokens
                total_tokens += n_tokens
                del out
    finally:
        if was_training:
            model.train()
    mean_loss = (total_loss / total_tokens) if total_tokens else None
    return mean_loss, total_tokens


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="path to the 4-bit base checkpoint dir")
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
    return p


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred to here so the pure functions above stay importable
    # (and unit-testable) without the lab/.venv training deps installed.
    # unsloth must be imported before trl/transformers/peft (it patches them at
    # import time) -- import it first among the heavy deps, per unsloth's own
    # startup warning.
    import importlib.metadata as importlib_metadata

    from unsloth import FastModel

    # `global` so the module-level eval_loss() (called below with the raw,
    # unwrapped model) can see `torch` too -- it is defined outside main() so
    # its signature/behaviour is easy to find and reuse, but it still must
    # not require torch at module import time for tests/test_lab_train.py.
    global torch
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

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
        optimizer step): step, loss, lr, elapsed time, and RUNNING (cumulative
        average, not instantaneous) s/step and tokens/s, plus current VRAM.
        """

        def __init__(self, path: Path):
            self.path = path
            self.t0: float | None = None
            # Truncate any stale file from a previous run in the same --out dir.
            self.path.write_text("")

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or self.t0 is None:
                return
            elapsed = time.time() - self.t0
            step = state.global_step
            tokens_seen = logs.get("num_input_tokens_seen", 0)
            row = {
                "step": step,
                "loss": logs["loss"],
                "learning_rate": logs.get("learning_rate"),
                "elapsed_s": elapsed,
                "s_per_step": (elapsed / step) if step else None,
                "tokens_per_s": (tokens_seen / elapsed) if elapsed > 0 else None,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_vram_gb": torch.cuda.memory_reserved() / 2**30,
            }
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[MetaLogger(meta_path)],
    )

    resume = args.resume if args.resume else None
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

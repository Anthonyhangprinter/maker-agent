#!/usr/bin/env python3
"""M6' QLoRA fine-tune of the CAD fast rung — runs ON THE RENTED GPU (RunPod), not this box.

Student:  Qwen2.5-Coder-7B-Instruct (4-bit base via Unsloth)
Data:     cad-sft-train.jsonl / cad-sft-val.jsonl (ChatML `messages` rows from
          scripts/compile_sft.py — production-identical prompts, incl. revise turns)
Loss:     response-only (everything before the assistant turn is masked — the ~2.7k-token
          system prompt must not dominate the loss)
Output:   ./out/adapter (LoRA), ./out/merged (fp16), ./cad-coder-q4_k_m.gguf

See docs/RUNPOD_RUNBOOK.md for the pod setup + transfer steps.

    python3 train_qlora.py --train cad-sft-train.jsonl --val cad-sft-val.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="cad-sft-train.jsonl")
    ap.add_argument("--val", default="cad-sft-val.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--no-gguf", action="store_true", help="skip the GGUF export step")
    args = ap.parse_args()

    # Imports deferred so --help works on a box without unsloth installed.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    # ChatML — the same template Ollama's qwen2.5-coder tag applies at inference.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def to_text(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]]}

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    ds = ds.map(to_text, batched=True, remove_columns=ds["train"].column_names)
    n_train = len(ds["train"])
    print(f"train={n_train}  val={len(ds['val'])}  max_seq={args.max_seq}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=SFTConfig(
            output_dir=args.out,
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            seed=args.seed,
            report_to="none",
        ),
    )
    # Mask everything before the assistant turn: system + user tokens carry no loss.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    print(f"train loss {stats.training_loss:.4f}")
    metrics = trainer.evaluate()
    print(f"val loss {metrics.get('eval_loss'):.4f}")

    out = Path(args.out)
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    if not args.no_gguf:
        # Unsloth drives llama.cpp's convert + quantize itself.
        model.save_pretrained_gguf(str(out / "gguf"), tokenizer,
                                   quantization_method="q4_k_m")
    (out / "run_meta.json").write_text(json.dumps({
        "base": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "rank": args.rank, "alpha": args.rank * 2, "lr": args.lr,
        "epochs": args.epochs, "seed": args.seed, "max_seq": args.max_seq,
        "n_train": n_train, "train_loss": stats.training_loss,
        "val_loss": metrics.get("eval_loss")}, indent=1))
    print("done — see", out)


if __name__ == "__main__":
    main()

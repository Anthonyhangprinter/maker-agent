# RunPod QLoRA Runbook — M6′ fine-tune of `cad-coder:7b`

One training run, start to finish. Expected cost **$3–6** (1–3 h on a 24 GB card).
The pod bills per minute while it exists — **terminate it when the GGUF is downloaded.**

## 1. Account + pod (human step)

1. runpod.io → sign up → Billing → add ~$10 credit.
2. Deploy → GPU Pod → pick **RTX 4090 (24 GB)** — usually $0.35–0.70/hr on Community Cloud.
   (A100 80GB works too but costs more; only pick it if no 4090 is available.)
3. Template: **RunPod PyTorch 2.x** (any recent CUDA 12 PyTorch template).
4. Disk: 60 GB volume (base model ~15 GB + merged fp16 ~15 GB + GGUF ~4.7 GB + headroom).
5. Deploy, wait for Running, note the **SSH command** from the Connect tab.

## 2. Upload data + script (from the Z2 box)

```bash
# paths on this machine
cd ~/.openclaw/skills/cad-builder
scp -P <PORT> ~/.openclaw/cad-sft-train.jsonl ~/.openclaw/cad-sft-val.jsonl \
    scripts/train_qlora.py root@<POD_IP>:/workspace/
```

## 3. Train (on the pod)

```bash
cd /workspace
pip install unsloth            # pulls torch-matched wheels, trl, peft, bitsandbytes
python3 train_qlora.py --train cad-sft-train.jsonl --val cad-sft-val.jsonl
```

- ~45–90 min for 3 epochs over ~450 examples on a 4090.
- Watch the loss lines; val loss printed at each epoch end.
- Output: `/workspace/out/adapter/`, `/workspace/out/merged/`,
  `/workspace/out/gguf/…q4_k_m.gguf`, `/workspace/out/run_meta.json`.
- If the GGUF step fails (llama.cpp build issue on the pod), rerun with `--no-gguf`,
  download `out/merged/` instead, and convert locally:
  `python3 llama.cpp/convert_hf_to_gguf.py merged --outfile cad-coder-f16.gguf`
  then `llama-quantize cad-coder-f16.gguf cad-coder-q4_k_m.gguf Q4_K_M`.

## 4. Download + verify (from the Z2 box)

```bash
scp -P <PORT> root@<POD_IP>:/workspace/out/gguf/*q4_k_m.gguf ~/cad-coder-q4_k_m.gguf
scp -P <PORT> root@<POD_IP>:/workspace/out/run_meta.json ~/cad-coder-run_meta.json
# checksum both sides — a truncated 4.7 GB scp looks fine until Ollama rejects it
ssh -p <PORT> root@<POD_IP> "sha256sum /workspace/out/gguf/*q4_k_m.gguf"
sha256sum ~/cad-coder-q4_k_m.gguf
```

Optionally also download `out/adapter/` (small, ~200 MB) so a re-quantize or a
continued-training run never needs the pod again.

## 5. TERMINATE THE POD

RunPod console → My Pods → Terminate. Stopped-but-not-terminated pods keep billing
for the volume. Verify billing shows $0/hr after.

## 6. Deploy into Ollama (Phase 7)

```bash
ollama show qwen2.5-coder:7b-instruct-q4_K_M --modelfile > /tmp/stock.Modelfile
# Build the new Modelfile: FROM ~/cad-coder-q4_k_m.gguf + the stock TEMPLATE/PARAMETER
# lines copied VERBATIM (template fidelity is a ship-blocker).
ollama create cad-coder:7b -f Modelfile
```

Then pin via `~/.openclaw/cad.json` `"code_model": "cad-coder:7b"` for the eval;
`CODE_MODEL_FAST` in `cad_v5/config.py` changes only at ship (see the plan).

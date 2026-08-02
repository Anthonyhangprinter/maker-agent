#!/bin/bash
# M6' eval chain: waits for baseline run 2, then fine-tuned evals, then stock held-out.
cd ~/.openclaw/skills/cad-builder
L=logs
{
echo "=== waiting for baseline run 2 ==="
until grep -q BASELINE2-DONE $L/baseline_run2.log 2>/dev/null; do sleep 60; done
echo "=== smoke test: cad-coder:7b via env override ==="
CAD_CODE_MODEL_FAST=cad-coder:7b python3 cad_engine.py build "a 80x50x6mm plate with a 12mm centre hole" --coder fast --no-upload > $L/smoke_ft.json 2>$L/smoke_ft.err
grep -o '"code_model": *"[^"]*"' $L/smoke_ft.json || { echo SMOKE-FAILED; exit 1; }
grep -q 'cad-coder:7b' $L/smoke_ft.json || { echo SMOKE-WRONG-MODEL; exit 1; }
echo "=== FT suite run A ==="
CAD_CODE_MODEL_FAST=cad-coder:7b python3 scripts/run_benchmarks.py --coder fast --timeout 1500 > $L/ft_run1.log 2>&1
echo "=== FT suite run B ==="
CAD_CODE_MODEL_FAST=cad-coder:7b python3 scripts/run_benchmarks.py --coder fast --timeout 1500 > $L/ft_run2.log 2>&1
echo "=== FT heldout-cqe ==="
CAD_CODE_MODEL_FAST=cad-coder:7b python3 scripts/run_benchmarks.py --coder fast --timeout 1500 --suite heldout-cqe > $L/ft_heldout.log 2>&1
echo "=== stock heldout-cqe ==="
python3 scripts/run_benchmarks.py --coder fast --timeout 1500 --suite heldout-cqe > $L/stock_heldout.log 2>&1
echo EVAL-CHAIN-DONE
} >> logs/eval_chain.log 2>&1

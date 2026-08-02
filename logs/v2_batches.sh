#!/bin/bash
cd ~/.openclaw/skills/cad-builder
{
echo "=== v2 batch 1: hard singles, Sonnet, accept-soft ==="
python3 scripts/teacher_batch.py --specs benchmarks/teacher-hard/specs.json \
  --only "$(cat logs/v2_rest_ids.txt)" --budget 10 --accept-soft --max-tokens 24000
echo "=== v2 batch 2: mechanisms, Opus 32k, accept-soft ==="
python3 scripts/teacher_batch.py --specs benchmarks/teacher-hard/specs.json \
  --only "$(cat logs/v2_mech_ids.txt)" --model claude-opus-5 --budget 8 --accept-soft --max-tokens 32000
echo V2-BATCHES-DONE
} >> logs/v2_batches.log 2>&1

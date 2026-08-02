#!/bin/bash
# M6' overnight chain 2026-07-31: mechanism redos -> review queue -> GIFT amplification.
# Serialized on purpose: two writers must never append cad-sftpairs.jsonl concurrently.
set -x
cd ~/.openclaw/skills/cad-builder

echo "=== redo mechanisms at 32k tokens ==="
python3 scripts/teacher_batch.py --specs benchmarks/teacher-mech2/specs.json \
    --only M05,M08,M10,M17,M23,M25 --model claude-opus-5 --max-tokens 32000 --budget 3

echo "=== redo V150 (server-side error) ==="
python3 scripts/teacher_batch.py --specs benchmarks/teacher-batch2/specs.json \
    --only V150 --max-tokens 32000 --budget 1

echo "=== queue accepted parts for human review (no-render; .step opens in CAD Viewer) ==="
for f in $(ls -t benchmarks/results/teacher_pilot_2026073*.json | head -3); do
    python3 scripts/review_queue.py --queue-run "$f" --which accepted --no-render
done

echo "=== GIFT amplification (free, local GPU) ==="
python3 scripts/gift_sample.py --k 6

echo "=== preview dataset compile (report only) ==="
python3 scripts/compile_sft.py --report-only

echo "OVERNIGHT DONE"

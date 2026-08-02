#!/bin/bash
cd ~/.openclaw/skills/cad-builder
{
until grep -q V2-BATCHES-DONE logs/v2_batches.log 2>/dev/null; do sleep 60; done
echo "=== v2 recovery: unrepaired + empty hard singles at 32k ==="
IDS=$(python3 - <<'PY'
import json, glob
runs = sorted(glob.glob('benchmarks/results/teacher_pilot_*.json'))
for f in reversed(runs):
    d = json.load(open(f))
    if d.get('specs') == 103 and 'sonnet' in (d.get('model') or ''):
        bad = [r['id'] for r in d['per_spec']
               if r.get('outcome') in ('repair-skipped-budget', 'error')]
        print(','.join(bad)); break
PY
)
echo "recovering: $IDS"
python3 scripts/teacher_batch.py --specs benchmarks/teacher-hard/specs.json \
  --only "$IDS" --budget 10 --accept-soft --max-tokens 32000
echo V2-RECOVERY-DONE
} >> logs/v2_batches.log 2>&1

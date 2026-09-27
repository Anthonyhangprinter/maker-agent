# gemma-4-31b-cad-r2 vs gemma-4-31b -- round evaluation

Arm: `gemma-4-31b-cad-r2`. Baseline: `gemma-4-31b`. Harness: `scripts/run_card.py` `--subset phase1 --mode oneshot` (round 1's own harness, unchanged); scorer: `scripts/geom_bands.py score_against_reference` via `scripts/lift_report.py`'s `lift_table` (unchanged). One sample per arm per spec at the engine's fixed serving temperature -- the same 'one sample each' round 1 used.

## Public suites (cadprompt, text2cadquery, heldout-cqe; round 1's own 85 specs)

| arm | n | invalid | gate clean | acceptance | match (ref n) | flips (invalid) | flips (match) | median s |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31b (baseline) | 85 | 8.2% | 85.9% | 77.3% | 31.8% (85) | - | - | 34.6 |
| gemma-4-31b-cad-r2 | 85 | 11.8% | 82.4% | 75.5% | 28.2% (85) | +2/-5 | +2/-5 | 28.9 |

## Owner references (18 parts, ~/CAD/references, hand-verified, never trained on)

| arm | n | invalid | gate clean | acceptance | match (ref n) | flips (invalid) | flips (match) | median s |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31b (baseline) | 18 | 50.0% | 33.3% | 44.4% | 11.1% (18) | - | - | 99.0 |
| gemma-4-31b-cad-r2 | 18 | 38.9% | 22.2% | 50.0% | 22.2% (18) | +3/-1 | +2/-0 | 72.75 |

## Verdict: NO-SHIP

Round 1's rule (invalid ratio first, then match share, paired flips in favour, on BOTH tables) was not met:
- public-85: invalid ratio regressed (flips +2/-5)
- public-85: match share did not improve net of flips (+2/-5)

# Lift vs baseline

Baseline arm: `gemma-4-31b`. Computed over the public suites (cadprompt, text2cadquery, heldout-cqe) only, helper rows excluded: invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, match = band==match over the rows that have a reference (ref n; an invalid build is a non-match), deltas (delta / Δ) are percentage points vs the baseline except median s, which is seconds; tokens/build is the mean output tokens.

Each delta is like for like: `base` names the arm it is measured against and `base n` is that arm's row count restricted to this variant's own specs, so a variant run on a subset is never compared against the baseline's whole-suite average.

With n around 40 to 85 a 3-point delta is 1 to 3 builds, which is inside the run-to-run noise of a single sample: read the flips column (+improved/-worsened, paired spec by spec against the same spec in the base arm) before calling any of these a result.

Acceptance is near-redundant with the invalid ratio on cadprompt and text2cadquery, whose acceptance entries carry the single `solids` criterion: on those suites a build that produced a solid passes and one that did not fails, so the column mostly restates invalid.

| variant | base | n | base n | invalid | Δ | flips | gate clean | Δ | acceptance | Δ | match | ref n | Δ | flips | median s | Δ | tokens/build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | gemma-4-31b | 85 | 85 | 2% | 0 | +0/-0 | 89% | 0 | 76% | 0 | 35% | 84 | 0 | +0/-0 | 31 | 0 | 542 |
| gemma-4-31b+agent-gemma4 | gemma-4-31b | 40 | 40 | 8% | +8 | +0/-3 | 90% | -5 | 92% | +8 | 35% | 40 | -5 | +1/-3 | 129 | +99 | - |
| gemma-4-31b+agent-self | gemma-4-31b+agent-gemma4 | 40 | 40 | 8% | 0 | +0/-0 | 90% | 0 | 92% | 0 | 40% | 40 | +5 | +2/-0 | 100 | -29 | - |
| gemma-4-31b+bo3 | gemma-4-31b | 85 | 85 | 5% | +2 | +0/-2 | 89% | 0 | 75% | -1 | 36% | 85 | +2 | +3/-1 | 76 | +44 | 1406 |
| gemma-4-31b+nofs | gemma-4-31b | 85 | 85 | 8% | +6 | +1/-6 | 88% | -1 | 69% | -7 | 28% | 85 | -6 | +2/-7 | 30 | -1 | 540 |
| gemma-4-31b-think | gemma-4-31b | 40 | 40 | 0% | 0 | +0/-0 | 98% | +3 | 86% | +2 | 45% | 40 | +5 | +3/-1 | 150 | +120 | 3838 |

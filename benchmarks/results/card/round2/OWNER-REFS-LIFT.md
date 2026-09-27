# Lift vs baseline

Baseline arm: `gemma-4-31b`. Computed over the public suites (cadprompt, text2cadquery, heldout-cqe) only, helper rows excluded: invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, match = band==match over the rows that have a reference (ref n; an invalid build is a non-match), deltas (delta / Δ) are percentage points vs the baseline except median s, which is seconds; tokens/build is the mean output tokens.

Each delta is like for like: `base` names the arm it is measured against and `base n` is that arm's row count restricted to this variant's own specs, so a variant run on a subset is never compared against the baseline's whole-suite average.

With n around 40 to 85 a 3-point delta is 1 to 3 builds, which is inside the run-to-run noise of a single sample: read the flips column (+improved/-worsened, paired spec by spec against the same spec in the base arm) before calling any of these a result.

Acceptance is near-redundant with the invalid ratio on cadprompt and text2cadquery, whose acceptance entries carry the single `solids` criterion: on those suites a build that produced a solid passes and one that did not fails, so the column mostly restates invalid.

| variant | base | n | base n | invalid | Δ | flips | gate clean | Δ | acceptance | Δ | match | ref n | Δ | flips | median s | Δ | tokens/build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | gemma-4-31b | 18 | 18 | 50% | 0 | +0/-0 | 33% | 0 | 44% | 0 | 11% | 18 | 0 | +0/-0 | 99 | 0 | 2752 |
| gemma-4-31b-cad-r2 | gemma-4-31b | 18 | 18 | 39% | -11 | +3/-1 | 22% | -11 | 50% | +6 | 22% | 18 | +11 | +2/-0 | 73 | -26 | 1307 |

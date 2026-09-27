# Maker Agent card round2

Mode: oneshot. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-cad-r2 | cadprompt | 30 | 25 | 17% | 25 | 63% | 13 | 5 | 3 | 4 | 26 | 8187 | 0 |
| gemma-4-31b-cad-r2 | heldout-cqe | 25 | 21 | 16% | 18 | 78% | 11 | 5 | 4 | 1 | 34 | 11058 | 0 |
| gemma-4-31b-cad-r2 | owner-refs | 18 | 11 | 39% | 4 | 50% | 4 | 3 | 2 | 2 | 73 | 23519 | 0 |
| gemma-4-31b-cad-r2 | text2cadquery | 30 | 29 | 3% | 27 | 83% | 0 | 1 | 17 | 11 | 25 | 8428 | 0 |
| gemma-4-31b | cadprompt | 30 | 27 | 10% | 27 | 73% | 12 | 6 | 7 | 2 | 32 | 14718 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 19 | 78% | 15 | 2 | 4 | 2 | 41 | 18573 | 0 |
| gemma-4-31b | owner-refs | 18 | 9 | 50% | 6 | 44% | 2 | 0 | 6 | 1 | 99 | 49533 | 0 |
| gemma-4-31b | text2cadquery | 30 | 28 | 7% | 27 | 80% | 0 | 1 | 20 | 7 | 31 | 14251 | 0 |

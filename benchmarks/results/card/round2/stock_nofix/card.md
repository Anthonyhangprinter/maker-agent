# Maker Agent card stock_nofix

Mode: oneshot. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | cadprompt | 30 | 28 | 7% | 28 | 77% | 15 | 5 | 7 | 1 | 31 | 13546 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 20 | 20% | 17 | 76% | 14 | 5 | 1 | 0 | 40 | 17173 | 0 |
| gemma-4-31b | owner-refs | 18 | 11 | 39% | 3 | 56% | 3 | 1 | 6 | 1 | 89 | 35433 | 0 |
| gemma-4-31b | text2cadquery | 30 | 29 | 3% | 28 | 80% | 0 | 2 | 17 | 10 | 31 | 13289 | 0 |

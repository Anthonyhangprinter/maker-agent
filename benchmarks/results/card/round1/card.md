# Maker Agent card round1

Mode: oneshot. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-cad-r1 | cad-arena | 12 | 11 | 8% | 9 | 78% | 0 | 0 | 0 | 0 | 28 | 6353 | 1 |
| gemma-4-31b-cad-r1 | cadprompt | 30 | 30 | 0% | 30 | 83% | 16 | 5 | 6 | 3 | 31 | 11967 | 0 |
| gemma-4-31b-cad-r1 | hard-eval | 15 | 11 | 27% | 5 | - | 0 | 0 | 0 | 0 | 89 | 36955 | 0 |
| gemma-4-31b-cad-r1 | heldout-cqe | 25 | 21 | 16% | 17 | 72% | 13 | 3 | 4 | 1 | 39 | 18838 | 0 |
| gemma-4-31b-cad-r1 | organic | 5 | 3 | 40% | 1 | 31% | 0 | 0 | 0 | 0 | 76 | 6825 | 0 |
| gemma-4-31b-cad-r1 | text-to-cad | 10 | 5 | 50% | 4 | 48% | 0 | 0 | 0 | 0 | 92 | 14664 | 0 |
| gemma-4-31b-cad-r1 | text2cadquery | 30 | 29 | 3% | 28 | 80% | 0 | 2 | 19 | 8 | 32 | 13605 | 0 |
| gemma-4-31b | cad-arena | 12 | 12 | 0% | 10 | 78% | 0 | 0 | 0 | 0 | 28 | 6097 | 1 |
| gemma-4-31b | cadprompt | 30 | 30 | 0% | 30 | 83% | 15 | 7 | 6 | 2 | 30 | 13832 | 0 |
| gemma-4-31b | hard-eval | 15 | 12 | 20% | 8 | - | 0 | 0 | 0 | 0 | 78 | 34154 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 17 | 68% | 14 | 2 | 6 | 1 | 39 | 19133 | 0 |
| gemma-4-31b | organic | 5 | 4 | 20% | 1 | 85% | 0 | 0 | 0 | 0 | 72 | 6539 | 0 |
| gemma-4-31b | text-to-cad | 10 | 6 | 40% | 4 | 52% | 0 | 0 | 0 | 0 | 96 | 15884 | 0 |
| gemma-4-31b | text2cadquery | 30 | 30 | 0% | 29 | 83% | 0 | 2 | 18 | 10 | 28 | 13125 | 0 |

# Maker Agent card phase2

Mode: oneshot. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-cad-spike | cad-arena | 12 | 12 | 0% | 9 | 78% | 0 | 0 | 0 | 0 | 26 | 3632 | 1 |
| gemma-4-31b-cad-spike | cadprompt | 30 | 25 | 17% | 25 | 73% | 12 | 4 | 7 | 2 | 23 | 8888 | 0 |
| gemma-4-31b-cad-spike | hard-eval | 15 | 11 | 27% | 3 | - | 0 | 0 | 0 | 0 | 69 | 32474 | 0 |
| gemma-4-31b-cad-spike | heldout-cqe | 25 | 21 | 16% | 15 | 60% | 8 | 3 | 10 | 0 | 32 | 10730 | 0 |
| gemma-4-31b-cad-spike | organic | 5 | 2 | 60% | 0 | 31% | 0 | 0 | 0 | 0 | 42 | 3435 | 0 |
| gemma-4-31b-cad-spike | text-to-cad | 10 | 5 | 50% | 3 | 39% | 0 | 0 | 0 | 0 | 52 | 9759 | 0 |
| gemma-4-31b-cad-spike | text2cadquery | 30 | 29 | 3% | 27 | 87% | 0 | 2 | 19 | 8 | 22 | 9522 | 0 |
| gemma-4-31b | cad-arena | 12 | 12 | 0% | 10 | 78% | 0 | 0 | 0 | 0 | 28 | 6097 | 1 |
| gemma-4-31b | cadprompt | 30 | 30 | 0% | 30 | 83% | 15 | 7 | 6 | 2 | 30 | 13832 | 0 |
| gemma-4-31b | hard-eval | 15 | 12 | 20% | 8 | - | 0 | 0 | 0 | 0 | 78 | 34154 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 17 | 68% | 14 | 2 | 6 | 1 | 39 | 19133 | 0 |
| gemma-4-31b | organic | 5 | 4 | 20% | 1 | 85% | 0 | 0 | 0 | 0 | 72 | 6539 | 0 |
| gemma-4-31b | text-to-cad | 10 | 6 | 40% | 4 | 52% | 0 | 0 | 0 | 0 | 96 | 15884 | 0 |
| gemma-4-31b | text2cadquery | 30 | 30 | 0% | 29 | 83% | 0 | 2 | 18 | 10 | 28 | 13125 | 0 |

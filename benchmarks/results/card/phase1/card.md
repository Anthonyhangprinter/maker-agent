# Maker Agent card phase1

Mode: agent. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b+agent-gemma4 | cad-arena | 6 | 6 | 0% | 5 | 100% | 0 | 0 | 0 | 0 | 98 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | cadprompt | 15 | 12 | 20% | 12 | 80% | 7 | 3 | 2 | 0 | 122 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | hard-eval | 5 | 5 | 0% | 4 | - | 0 | 0 | 0 | 0 | 362 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | heldout-cqe | 10 | 10 | 0% | 9 | 95% | 7 | 3 | 0 | 0 | 168 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | organic | 3 | 2 | 33% | 0 | 71% | 0 | 0 | 0 | 0 | 393 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | text-to-cad | 5 | 4 | 20% | 2 | 82% | 0 | 0 | 0 | 0 | 335 | 0 | 0 |
| gemma-4-31b+agent-gemma4 | text2cadquery | 15 | 15 | 0% | 15 | 100% | 0 | 1 | 10 | 4 | 136 | 0 | 0 |
| gemma-4-31b+agent-self | cad-arena | 6 | 6 | 0% | 5 | 100% | 0 | 0 | 0 | 0 | 175 | 0 | 0 |
| gemma-4-31b+agent-self | cadprompt | 15 | 12 | 20% | 12 | 80% | 8 | 3 | 1 | 0 | 94 | 0 | 0 |
| gemma-4-31b+agent-self | hard-eval | 5 | 5 | 0% | 4 | - | 0 | 0 | 0 | 0 | 409 | 0 | 0 |
| gemma-4-31b+agent-self | heldout-cqe | 10 | 10 | 0% | 9 | 95% | 8 | 1 | 1 | 0 | 102 | 0 | 0 |
| gemma-4-31b+agent-self | organic | 3 | 2 | 33% | 0 | 71% | 0 | 0 | 0 | 0 | 266 | 0 | 0 |
| gemma-4-31b+agent-self | text-to-cad | 5 | 4 | 20% | 2 | 82% | 0 | 0 | 0 | 0 | 387 | 0 | 0 |
| gemma-4-31b+agent-self | text2cadquery | 15 | 15 | 0% | 15 | 100% | 0 | 1 | 13 | 1 | 128 | 0 | 0 |
| gemma-4-31b+bo3 | cad-arena | 12 | 11 | 8% | 9 | 89% | 0 | 0 | 0 | 0 | 62 | 17372 | 1 |
| gemma-4-31b+bo3 | cadprompt | 30 | 30 | 0% | 30 | 83% | 15 | 6 | 7 | 2 | 73 | 41735 | 0 |
| gemma-4-31b+bo3 | hard-eval | 15 | 14 | 7% | 8 | - | 0 | 0 | 0 | 0 | 162 | 52565 | 0 |
| gemma-4-31b+bo3 | heldout-cqe | 25 | 21 | 16% | 17 | 70% | 16 | 1 | 4 | 0 | 96 | 45161 | 0 |
| gemma-4-31b+bo3 | organic | 5 | 4 | 20% | 1 | 62% | 0 | 0 | 0 | 0 | 134 | 13455 | 0 |
| gemma-4-31b+bo3 | text-to-cad | 10 | 7 | 30% | 5 | 58% | 0 | 0 | 0 | 0 | 173 | 32802 | 0 |
| gemma-4-31b+bo3 | text2cadquery | 30 | 30 | 0% | 29 | 77% | 0 | 2 | 19 | 9 | 68 | 32603 | 0 |
| gemma-4-31b+nofs | cad-arena | 12 | 11 | 8% | 9 | 89% | 0 | 0 | 0 | 0 | 25 | 9476 | 1 |
| gemma-4-31b+nofs | cadprompt | 30 | 30 | 0% | 30 | 77% | 13 | 5 | 9 | 3 | 28 | 13463 | 0 |
| gemma-4-31b+nofs | hard-eval | 15 | 12 | 20% | 4 | - | 0 | 0 | 0 | 0 | 84 | 39794 | 0 |
| gemma-4-31b+nofs | heldout-cqe | 25 | 19 | 24% | 17 | 58% | 11 | 2 | 6 | 0 | 40 | 19117 | 0 |
| gemma-4-31b+nofs | organic | 5 | 3 | 40% | 1 | 54% | 0 | 0 | 0 | 0 | 77 | 7477 | 0 |
| gemma-4-31b+nofs | text-to-cad | 10 | 5 | 50% | 4 | 45% | 0 | 0 | 0 | 0 | 89 | 14489 | 0 |
| gemma-4-31b+nofs | text2cadquery | 30 | 29 | 3% | 28 | 80% | 0 | 2 | 16 | 11 | 28 | 13311 | 0 |
| gemma-4-31b-think | cad-arena | 6 | 6 | 0% | 5 | 75% | 0 | 0 | 0 | 0 | 171 | 33565 | 0 |
| gemma-4-31b-think | cadprompt | 15 | 15 | 0% | 15 | 80% | 10 | 2 | 2 | 1 | 84 | 44506 | 0 |
| gemma-4-31b-think | hard-eval | 5 | 5 | 0% | 5 | - | 0 | 0 | 0 | 0 | 313 | 44558 | 0 |
| gemma-4-31b-think | heldout-cqe | 10 | 10 | 0% | 9 | 90% | 8 | 1 | 1 | 0 | 137 | 36653 | 0 |
| gemma-4-31b-think | organic | 3 | 3 | 0% | 1 | 100% | 0 | 0 | 0 | 0 | 473 | 36856 | 0 |
| gemma-4-31b-think | text-to-cad | 5 | 2 | 60% | 1 | 41% | 0 | 0 | 0 | 0 | 296 | 45362 | 0 |
| gemma-4-31b-think | text2cadquery | 15 | 15 | 0% | 15 | 87% | 0 | 1 | 12 | 2 | 205 | 72366 | 0 |
| gemma-4-31b | cad-arena | 12 | 12 | 0% | 10 | 78% | 0 | 0 | 0 | 0 | 28 | 6097 | 1 |
| gemma-4-31b | cadprompt | 30 | 30 | 0% | 30 | 83% | 15 | 7 | 6 | 2 | 30 | 13832 | 0 |
| gemma-4-31b | hard-eval | 15 | 12 | 20% | 8 | - | 0 | 0 | 0 | 0 | 78 | 34154 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 17 | 68% | 14 | 2 | 6 | 1 | 39 | 19133 | 0 |
| gemma-4-31b | organic | 5 | 4 | 20% | 1 | 85% | 0 | 0 | 0 | 0 | 72 | 6539 | 0 |
| gemma-4-31b | text-to-cad | 10 | 6 | 40% | 4 | 52% | 0 | 0 | 0 | 0 | 96 | 15884 | 0 |
| gemma-4-31b | text2cadquery | 30 | 30 | 0% | 29 | 83% | 0 | 2 | 18 | 10 | 28 | 13125 | 0 |

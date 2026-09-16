# Maker Agent card phase0

Mode: oneshot. One row per arm and suite. invalid = no solid produced, gate clean = solid with zero hard and zero [spec] findings, acceptance = pooled checks, bands = Chamfer vs reference where one exists (unit-normalised only for suites whose acceptance entries say so), helper = builds a correct-by-construction helper produced instead of the model.

| arm | suite | n | valid | invalid | gate clean | acceptance | match | valid band | near miss | fail | median s | tokens out | helper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| devstral-small-2 | cad-arena | 12 | 11 | 8% | 9 | 78% | 0 | 0 | 0 | 0 | 22 | 5787 | 1 |
| devstral-small-2 | cadprompt | 100 | 85 | 15% | 77 | 71% | 22 | 17 | 22 | 24 | 20 | 31681 | 0 |
| devstral-small-2 | hard-eval | 15 | 13 | 13% | 5 | - | 0 | 0 | 0 | 0 | 44 | 14298 | 0 |
| devstral-small-2 | heldout-cqe | 25 | 19 | 24% | 14 | 54% | 6 | 4 | 6 | 3 | 23 | 10276 | 0 |
| devstral-small-2 | organic | 5 | 4 | 20% | 2 | 23% | 0 | 0 | 0 | 0 | 42 | 4343 | 0 |
| devstral-small-2 | text-to-cad | 10 | 4 | 60% | 1 | 39% | 0 | 0 | 0 | 0 | 42 | 9549 | 0 |
| devstral-small-2 | text2cadquery | 95 | 90 | 5% | 80 | 76% | 4 | 5 | 33 | 48 | 21 | 28573 | 0 |
| gemma-4-31b | cad-arena | 12 | 10 | 17% | 8 | 67% | 0 | 0 | 0 | 0 | 26 | 7628 | 1 |
| gemma-4-31b | cadprompt | 100 | 94 | 6% | 92 | 78% | 39 | 14 | 31 | 10 | 34 | 68416 | 0 |
| gemma-4-31b | hard-eval | 15 | 14 | 7% | 7 | - | 0 | 0 | 0 | 0 | 82 | 36258 | 0 |
| gemma-4-31b | heldout-cqe | 25 | 23 | 8% | 18 | 82% | 17 | 2 | 3 | 1 | 42 | 19587 | 0 |
| gemma-4-31b | organic | 5 | 5 | 0% | 2 | 85% | 0 | 0 | 0 | 0 | 66 | 6107 | 0 |
| gemma-4-31b | text-to-cad | 10 | 6 | 40% | 4 | 52% | 0 | 0 | 0 | 0 | 86 | 16005 | 0 |
| gemma-4-31b | text2cadquery | 95 | 95 | 0% | 91 | 77% | 3 | 5 | 43 | 44 | 29 | 42483 | 0 |
| glm-4.7-flash | cad-arena | 12 | 10 | 17% | 8 | 100% | 0 | 0 | 0 | 0 | 19 | 6218 | 1 |
| glm-4.7-flash | cadprompt | 100 | 71 | 29% | 66 | 53% | 10 | 13 | 30 | 18 | 16 | 143577 | 0 |
| glm-4.7-flash | hard-eval | 15 | 8 | 47% | 3 | - | 0 | 0 | 0 | 0 | 24 | 32845 | 0 |
| glm-4.7-flash | heldout-cqe | 25 | 16 | 36% | 13 | 48% | 4 | 2 | 7 | 3 | 16 | 42869 | 0 |
| glm-4.7-flash | organic | 5 | 1 | 80% | 0 | 23% | 0 | 0 | 0 | 0 | 23 | 20578 | 0 |
| glm-4.7-flash | text-to-cad | 10 | 5 | 50% | 1 | 42% | 0 | 0 | 0 | 0 | 24 | 25715 | 0 |
| glm-4.7-flash | text2cadquery | 95 | 61 | 36% | 59 | 41% | 1 | 3 | 32 | 25 | 17 | 107134 | 0 |
| gpt-oss-20b | cad-arena | 12 | 12 | 0% | 9 | 89% | 0 | 0 | 0 | 0 | 20 | 10497 | 1 |
| gpt-oss-20b | cadprompt | 100 | 81 | 19% | 78 | 63% | 26 | 13 | 27 | 15 | 16 | 105382 | 0 |
| gpt-oss-20b | hard-eval | 15 | 14 | 7% | 3 | - | 0 | 0 | 0 | 0 | 30 | 33941 | 0 |
| gpt-oss-20b | heldout-cqe | 25 | 16 | 36% | 13 | 54% | 7 | 4 | 4 | 1 | 18 | 30936 | 0 |
| gpt-oss-20b | organic | 5 | 1 | 80% | 0 | 15% | 0 | 0 | 0 | 0 | 20 | 9936 | 0 |
| gpt-oss-20b | text-to-cad | 10 | 2 | 80% | 1 | 10% | 0 | 0 | 0 | 0 | 19 | 19922 | 0 |
| gpt-oss-20b | text2cadquery | 95 | 89 | 6% | 79 | 72% | 2 | 5 | 38 | 44 | 15 | 66700 | 0 |
| qwen2.5-coder-7b | cad-arena | 12 | 10 | 17% | 7 | 67% | 0 | 0 | 0 | 0 | 20 | 5644 | 1 |
| qwen2.5-coder-7b | cadprompt | 100 | 56 | 44% | 56 | 39% | 9 | 8 | 23 | 16 | 14 | 47047 | 0 |
| qwen2.5-coder-7b | hard-eval | 15 | 10 | 33% | 4 | - | 0 | 0 | 0 | 0 | 22 | 12692 | 0 |
| qwen2.5-coder-7b | heldout-cqe | 25 | 18 | 28% | 11 | 44% | 3 | 3 | 5 | 7 | 18 | 27676 | 0 |
| qwen2.5-coder-7b | organic | 5 | 0 | 100% | 0 | 0% | 0 | 0 | 0 | 0 | 14 | 4186 | 0 |
| qwen2.5-coder-7b | text-to-cad | 10 | 5 | 50% | 2 | 35% | 0 | 0 | 0 | 0 | 19 | 8923 | 0 |
| qwen2.5-coder-7b | text2cadquery | 95 | 48 | 49% | 48 | 42% | 1 | 0 | 24 | 23 | 15 | 65867 | 0 |
| qwen3-coder-30b-a3b | cad-arena | 12 | 11 | 8% | 10 | 100% | 0 | 0 | 0 | 0 | 18 | 4575 | 1 |
| qwen3-coder-30b-a3b | cadprompt | 100 | 77 | 23% | 74 | 60% | 18 | 14 | 24 | 21 | 14 | 42595 | 0 |
| qwen3-coder-30b-a3b | hard-eval | 15 | 11 | 27% | 5 | - | 0 | 0 | 0 | 0 | 22 | 13389 | 0 |
| qwen3-coder-30b-a3b | heldout-cqe | 25 | 20 | 20% | 15 | 56% | 5 | 6 | 7 | 2 | 15 | 34835 | 0 |
| qwen3-coder-30b-a3b | organic | 5 | 4 | 20% | 0 | 54% | 0 | 0 | 0 | 0 | 21 | 3743 | 0 |
| qwen3-coder-30b-a3b | text-to-cad | 10 | 3 | 70% | 1 | 32% | 0 | 0 | 0 | 0 | 18 | 8560 | 0 |
| qwen3-coder-30b-a3b | text2cadquery | 95 | 82 | 14% | 76 | 65% | 1 | 3 | 39 | 39 | 15 | 76787 | 0 |
| qwen3.8-27b-nothink | cad-arena | 12 | 11 | 8% | 10 | 89% | 0 | 0 | 0 | 0 | 22 | 7696 | 1 |
| qwen3.8-27b-nothink | cadprompt | 100 | 71 | 29% | 69 | 62% | 22 | 13 | 20 | 16 | 23 | 79599 | 0 |
| qwen3.8-27b-nothink | hard-eval | 15 | 14 | 7% | 5 | - | 0 | 0 | 0 | 0 | 54 | 30761 | 0 |
| qwen3.8-27b-nothink | heldout-cqe | 25 | 17 | 32% | 14 | 58% | 9 | 3 | 4 | 1 | 27 | 21865 | 0 |
| qwen3.8-27b-nothink | organic | 5 | 2 | 60% | 0 | 31% | 0 | 0 | 0 | 0 | 56 | 18598 | 0 |
| qwen3.8-27b-nothink | text-to-cad | 10 | 6 | 40% | 5 | 58% | 0 | 0 | 0 | 0 | 49 | 20603 | 0 |
| qwen3.8-27b-nothink | text2cadquery | 95 | 88 | 7% | 82 | 68% | 1 | 2 | 36 | 49 | 21 | 45976 | 0 |

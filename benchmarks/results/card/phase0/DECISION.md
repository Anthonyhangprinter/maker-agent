# Phase 0 decision: the base model to train (final, 2026-09-16)

Rule applied: docs/MAKER-1.0-CAMPAIGN.md section 6. Public suites rank first (invalid ratio on
CADPrompt and Text-to-CadQuery, then the Chamfer match share on them), the internal suites break
ties, and inside a 3-point band the faster and easier-to-train arm wins. BenchCAD (official
harness) is run separately and appended below when its stage finishes.

## Shootout conditions

- One-shot mode: one codegen call, one automatic salvage turn, deterministic gate, no critic.
  This measures the base model, not the agent loop.
- Thinking off for every arm where the template exposes a switch. The 27B thinking arm was
  measured at 223 s per build (about 16 hours per arm) and Gemma-4-31B at 196 s with its
  template default on (21k tokens on one spec); both are out of scope for a base-model
  shootout. Thinking depth is a Phase 1 A/B on the winner.
- 262 builds per arm: text-to-cad 10, organic 5, hard-eval 15, heldout-cqe 25, cadprompt 100,
  text2cadquery 95, cad-arena 12. Card: card.md and card.json beside this file; rows.jsonl has
  every build.
- Excluded from ranking: text-to-cad/05 (the enclosure spec is also a seed example in the
  retrieval corpus; Phase 3 must drop it from one side). Helper rows (one per arm on cad-arena,
  the pinned spur gear) are byte-identical across arms and carry no signal.

## Ranking (public suites first)

| rank | arm | public invalid (cadprompt + t2cq, 195) | public matches | heldout invalid / matches (25) | internal gate-clean (29) | median s (cadprompt) | tokens out (all suites) |
|---|---|---|---|---|---|---|---|
| 1 | gemma-4-31b | 3% | 42 | 8% / 17 | 12 | 34 | 195,276 |
| 2 | devstral-small-2 | 10% | 26 | 24% / 6 | 8 | 20 | 103,719 |
| 3 | gpt-oss-20b | 13% | 28 | 36% / 7 | 4 | 16 | 275,825 |
| 4 | qwen3.8-27b-nothink (control) | 18% | 23 | 32% / 9 | 9 | 23 | 224,652 |
| 5 | qwen3-coder-30b-a3b | 18% | 19 | 20% / 5 | 6 | 14 | 183,703 |
| 6 | glm-4.7-flash | 32% | 11 | 36% / 4 | 4 | 16 | 378,093 |
| 7 | qwen2.5-coder-7b (floor) | 47% | 10 | 28% / 3 | 6 | 14 | 171,302 |

Skipped: qwen3.8-27b-think (see conditions).

## Decision

**Base model for training: Gemma-4-31B-it.** It wins outside the 3-point band on every
reference-scored column: a third of the next arm's invalid ratio on the public suites, 1.6x its
geometry matches, and twice the control's held-out matches. It also wins the internal gate-clean
count. The tie-breaks (speed, trainability, licence) do not apply because the margin is not
within the band.

Costs to carry into Phase 2: dense 31B (Unsloth quotes 22GB for its QLoRA with the vision tower
frozen; standard attention, no DeltaNet kernels), the Gemma licence for the 1.0 model card
(fine-tuning permitted; redistribution under Gemma terms with attribution), 50% more wall time
and about twice the tokens per build than the 27B, and the whole card at 22.9GB in use, so the
visual critic runs after the coder rather than beside it (Phase 1 measures that).

**Runner-up: Devstral Small 2** (Apache 2.0, 14.5GB, dense 24B): second on the public suites at
the fastest wall time among the dense arms. It is the fallback if Gemma's QLoRA does not fit
24GB in the Phase 2 spike, and a teacher-arm candidate for specs Gemma fails.

**The control stays the resident.** Qwen3.8-27B remains Morai's and Casa AI's model; the maker
server carries Gemma for CAD work (spec 4.2).

**Teacher arms for Phase 3:** Gemma (student) is also the strongest generator; Devstral and
gpt-oss-20b are the fast second opinions for specs the student fails twice. Flash-Next stays the
untrained reference.

## Arm files

Kept on disk: gemma-4-31b (winner), devstral-small-2 (runner-up, teacher), gpt-oss-20b
(teacher, cheap), the resident's file. Deletion candidates (about 39GB): glm-4.7-flash,
qwen3-coder-30b-a3b, qwen2.5-coder-7b GGUFs. Not deleted automatically; the owner decides after
reading this card.

## BenchCAD (official harness, 60 items per task, seed 42, finished 2026-09-16 20:20)

Run through the harness's own scorer with a local chat-completions adapter; the 27B CodeEdit
row was rerun cleanly after stale rows from a killed 150-item attempt leaked into its first
score. Published rows are the leaderboard values read on 2026-09-15.

| arm | CodeEdit (headroom IoU) | Code-QA (accuracy) | Vision2Code (voxel IoU) |
|---|---|---|---|
| qwen3.8-27b-nothink (control) | 0.800 | 0.685 | 0.166 |
| gemma-4-31b | 0.732 | 0.663 | 0.172 |
| devstral-small-2 | 0.797 | 0.557 | 0.114 |
| gpt-oss-20b | 0.692 | 0.639 | not vision |
| qwen3-coder-30b-a3b | 0.702 | 0.524 | not vision |
| glm-4.7-flash | 0.703 | 0.477 | not vision |
| qwen2.5-coder-7b | 0.703 | 0.170 | not vision |
| Gemma-4-31B-it (published) | - | 0.664 | - |
| gpt-oss-120b (published) | 0.561 | 0.689 | - |
| GPT-4o (published) | - | 0.726 | 0.182 |
| Gemini 3.1 Pro (published) | 0.837 | 0.838 | 0.289 |

Reading: Gemma's Code-QA reproduces its published score (0.663 vs 0.664), which validates the
adapter. On BenchCAD the 27B and Gemma trade places (27B ahead on CodeEdit and Code-QA, Gemma
ahead on Vision2Code); Devstral is a surprise on CodeEdit. None of it moves the decision: the
spec ranks the geometry-scored public suites first, and there Gemma's margin is decisive.
BenchCAD says the 27B remains the better editor of existing CadQuery programs, which is a
Phase 3 note (edit-style training data) rather than a base-model reason. Sample cut from 150 to
60 on 2026-09-16 (150 measured at about an hour per task on the 27B, roughly 14 hours across
the arms); the finalists get 150 in a later phase.

## Status

Final. Base model for Phase 2: **Gemma-4-31B-it**. Runner-up and fallback: Devstral Small 2.

# Phase 1 decision: agent-lift levers on Gemma-4-31B (final, 2026-09-17)

Rule applied: docs/plans/2026-09-16-phase1-agent-lift.md Task 7. A lever is locked in as the default
only if it improves the invalid ratio or the match rate on the public suites by 3 points or more,
with neither metric worsening, at no more than 2x the baseline wall time; ties go to the cheaper
setting. Every comparison is like for like: the baseline is restricted to the variant's own specs,
and the flips column counts paired spec-by-spec changes (see LIFT.md for the full table).

## Conditions

- Winner from Phase 0: Gemma-4-31B on the maker server (ctx 16384, thinking off unless stated).
- Subsets: `phase1` (127 builds, 85 public rows), `phase1think` (59 builds, tier-stratified, 40
  public rows) for the expensive legs. Mode one-shot unless stated. One sample per lever, so a
  3-point delta is 1 to 3 builds; the flips are the evidence, not the percentages alone.
- Agent mode note: `cad_engine.build()` always samples `first_turn_candidates()` (default 3) on
  its first turn, so agent-mode legs are best-of-3 by construction while one-shot legs are
  single-shot; the agent legs are therefore compared against each other for the critic question
  and against the one-shot baseline only for the mode question.

## Results (public suites: CADPrompt, Text-to-CadQuery, held-out; helper rows excluded)

| lever | n | invalid (base) | flips | match (base) | flips | median s (base) | verdict |
|---|---|---|---|---|---|---|---|
| best-of-3 (one-shot) | 85 | 5% (2%) | +0/-2 | 36% (35%) | +3/-1 | 76 (31) | no lift, 2.4x time: OFF |
| retrieval off | 85 | 8% (2%) | +1/-6 | 28% (35%) | +2/-7 | 30 (31) | retrieval matters: stays ON |
| thinking on (template default) | 40 | 0% (2%) | +0/-0 | 45% (40%) | +3/-1 | 150 (31) | small lift at 5x time, 7x tokens: not the default; kept as an escalation arm |
| agent mode, self-critic | 40 | 8% (0%) | +0/-3 | 40% (40%) | +2/-2 | 100 (29) | no geometry win over one-shot at 3.4x time; acceptance +8 (the loop clears spec advisories) |
| agent mode, Ollama gemma4:e4b critic (stock) | 40 | 8% (0%) | +0/-3 | 35% (40%) | +1/-3 | 129 (29) | the pre-campaign default; worse than one-shot on geometry at 4.2x time |
| self-critic vs Ollama critic (paired) | 40 | 8% (8%) | +0/-0 | 40% (35%) | +2/-0 | 100 (129) | self-critic wins on match and time: LOCKED IN |

## Decisions (draft)

1. **One-shot with retrieval on, thinking off, single candidate stays the build mode.**
   `candidates` stays unset for fluid mode (the engine's own agent loop keeps its best-of-3
   first turn, unchanged). The full loop is not a geometry win over one-shot on Gemma (both
   critics: invalid 8% vs 0%, match at best equal) and costs 3.4x to 4.2x the time; it remains
   the interactive refine path, not the benchmark or batch default.
2. **Thinking depth becomes an escalation option, not a default**: the `gemma-4-31b-think` arm
   stays in `benchmarks/arms.json` for Phase 3 teacher duty on specs the fast setting fails, and
   for hard tiers where five minutes per part is acceptable.
3. **Best-of-N is off for the strong rung.** The Phase 0 GIFT result (best-of-3 doubling the
   7B's acceptance) does not transfer to a model that is already reliable on the first sample.
4. **Retrieval is confirmed as a measured lift on the strong rung** (7 of 85 public specs flip
   to failure without it), which also validates the Phase 3 corpus work.
5. **Critic locked in: the coder judges its own render** (`CRITIC_MODEL` defaults to
   `CODE_MODEL_STRONG`; `CAD_CRITIC_MODEL` still overrides). Paired on the same 40 specs the
   self-critic beat the stock Ollama gemma4:e4b critic on match (40% vs 35%, two improved, none
   worsened) and on time (100 s vs 129 s), with no VRAM cost, and it removes the last Ollama
   model from the loop. The small-critic server variant (MiniCPM-V beside the coder) was ruled
   out by VRAM (21.4GB used at ctx 16384, about 3GB free) and stays a follow-up with an
   8k-context arm.
7. **Production defaults changed on this branch**: fluid mode's default coder is now `strong`
   (the 7B fast rung measured 44% invalid on CADPrompt against Gemma's 6%), and `cad.json`
   `maker` is enabled with the `gemma-4-31b` arm, so a CAD build swaps the resident out for the
   maker server for its duration (spec 4.2; the resident comes back in the build's finally).
   `arms.py restore` returns to the resident-only regime.
6. **Fluid auto-escalation** (spec Phase 1 lever) is deliberately deferred to Phase 3 as a
   teacher-arm question: switching arms mid-build costs a server swap per failing spec.

## Defects found by the runs (fixed on this branch)

- `cad_engine._new_build_dir` pruned build directories by name, so every date-named agent build
  deleted its own directory on creation and failed at the STEP copy: fixed (mtime, never the
  new dir, KEEP_BUILDS 5000). The first agent leg was rerun after the fix.
- The lift table had to restrict the baseline to the variant's own specs and count invalid
  builds as non-matches; without those two fixes the subset legs carried a 13-point artefact.

## Confirmation run

Not needed: the locked defaults are the baseline configuration already measured (127 builds,
117 valid, 31 s median on the public suites), so the baseline row is the confirmation.

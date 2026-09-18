# Claude (via the Claude Code subscription) vs local Gemma-4-31B, 20 specs, 2026-09-19

Owner question: are the Anthropic models better CAD coders than the local arm, and should Claude be a teacher?

Method: 20 reference-scored public specs (12 CADPrompt, 8 Text2CadQuery, picked at random). Every model got the byte-identical prompt the engine sends for a one-shot build (`scripts/dump_prompts.py`: the engine's 9.8k system prompt, the verbatim spec, retrieval notes). One attempt per spec. Every script went through the identical scorer (`scripts/score_external.py`: the engine's own step, inspect and gate path plus the Chamfer band against the reference). Builds that crashed got the engine's own repair prompt once, kept only if it improved. Claude ran as fresh Claude Code subagents (no API key, no metered spend); Gemma ran on the maker arm, thinking off, temperature 0.15.

| model | first attempt: built | match | after one repair: built | match |
|---|---|---|---|---|
| gemma-4-31b (local) | 17/20 | 3 | 19/20 | 3 |
| claude-sonnet | 19/20 | 6 | 19/20 | 6 |
| claude-opus | 19/20 | 6 | 20/20 | 6 |
| claude-fable | 20/20 | 6 | 20/20 | 6 |

Reading it:
- After one repair all four build 19 or 20 of 20. On reliability there is no gap. The repair turn moved builds, never a match.
- On exact geometry match Claude leads 6 to 3. That lead is 3 specs. All three Claude models agree, which makes luck less likely, but they share a lineage, so: suggestive, not proven. Noise floor measured the same night: two Gemma baselines on the same specs disagreed on 2 of 7.
- Sonnet, Opus and Fable tie on match. Sonnet is sufficient for any Claude batch.
- 13 of 20 specs were matched by no model, and 5 gave the identical outcome for all four: those specs are under-determined (missing bore, missing wall, contradictory wording). The reachable ceiling on this subset is about 7 to 9, not 20.
- Gemma's 17/20 built equals its Phase 0 count on the same 20 specs, an independent check that the scorer measures correctly.

Decision: the harvest loop stays local (the verifier is the teacher: Gemma x3 samples, then the same model with thinking on, which is a per-request switch). For specs with no reference the gate is the only judge and it cannot see a wrong shape, so a Claude teacher adds nothing measurable there. One frozen, small Claude (Sonnet) batch is worth making only for reference-backed specs, where the match check proves the pair: the owner's own reference models first. Cost through the subscription was about 10k to 14k tokens per spec, so a batch means hundreds of specs, not the whole bank.

Compliance notes: builders made 43 to 46 tool calls for 20 reads and 20 writes (consistent with one attempt each, no iteration); each repair agent made exactly 3. Sonnet ran `ls` twice against the no-shell rule (listing only); Opus and Fable read `prompts/INDEX.json` to learn the filenames. No reference, acceptance or other model's output was read by any of them, by their own report.

Harness bug found and worked around (nearly reported as a result): the scorer runs builds and scoring in one process; something in a build stage resets the process locale to C, after which `cad_engine.run_step`'s `subprocess.run(text=True)` cannot decode the child's `Volume: ... mm³` line and the build is recorded as "the script failed to run". Symptom: every model scored exactly 1 of 20. Proven by a per-stage locale trace and fixed for this run with `python3 -X utf8`. The 80 artefact rows are kept in `rows.locale-bug-artifact.jsonl`. None of the 2,646 Phase 0, 1, 2 rows carries the error (`run_card` builds in a child process). Engine hardening (explicit UTF-8 decoding) is on the phase3 branch.

Files: `prompts/` (the 20 prompts), `code/<model>/` (first attempts), `repair_prompts/`, `repairs/`, `rows.jsonl` (first attempt and `+repair` arms, plus 7 seeded three-day-old baseline rows kept for reference only).

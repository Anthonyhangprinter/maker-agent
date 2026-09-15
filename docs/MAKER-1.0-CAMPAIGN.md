# Maker Agent 1.0 Campaign (spec, 2026-09-15)

Status: DRAFT awaiting owner review. Supersedes ROADMAP.md Track D (fine-tune the 7B) with a
top-down plan: train and ship the STRONG rung first on the local RTX 3090, low rung later.

## 1. Goal and definition of done

The RTX 3090 was bought to train the Maker Agent locally and keep the GPU busy doing it. The
campaign ends when all three of these exist, at which point the project is closed and only a
scheduled loop remains:

1. **A benchmark card**: one command reproduces a table of local models (stock and trained)
   on the internal suites and at least two public text-to-CAD suites, with geometry metrics.
2. **A trained strong rung shipped as a separate CAD model** ("maker" model), swapped in per
   build, with a measured lift over its stock base on the card and no held-out regression.
3. **The data-and-training loop runs unattended** under Hermes on the Bridge (Lab tile,
   kanban approvals, ledgers), and the public README describes the real hardware story
   (developed on 24GB, 16GB ship target later).

Tag: `v1.0` on github.com/Anthonyhangprinter/maker-agent.

## 2. Decisions already made (owner, 2026-09-15)

- The trained model is a **separate CAD-only model**, loaded via the existing eviction hooks
  (resident `qwen38-server` stopped, maker server started, restored after). Morai and Casa AI
  keep the stock resident.
- **Training is local.** Use the 64GB system RAM (offload routes) before anything else.
  RunPod only if truly blocked and only with the owner's explicit say-so.
- **Teachers are local** (the stronger shootout arms and Qwen3.8-Flash-Next). Frontier cloud teachers are not
  worth it: the CAD-Coder result (fine-tuned 7B 1.45% invalid vs GPT-4o 93%) shows the gate
  and the data matter more than the teacher's brand.
- **The base model is not fixed.** Phase 0 is a shootout of every open model that fits the
  card and plausibly writes CAD code (Qwen3.8-27B, Gemma-4-31B, Qwen3-Coder-30B-A3B and the
  rest of the list in section 4.1); the winner is trained. The goal is the best CAD
  capability, not loyalty to one family.
- **UI work this campaign is a Lab tab** in the web UI: card, runs, review queue, job controls.

## 3. Principles

- **All local at runtime.** No cloud calls in any lane, timer, or button.
- **Measure, never claim.** Every lift is an A/B on the card. Stock vs trained, same suite,
  same seed, same settings.
- **Contamination is a hard rule.** Any spec on the card (internal or public) never enters a
  training set. The guard runs at compile time and at harvest time.
- **One GPU, pre-emptible.** Idle-time jobs are short units that yield the GPU when chat
  demand appears; long jobs (training rounds) are explicitly approved and scheduled.
- **Every capability ships with its Lab-tab control** (webui-is-the-testbed rule).
- **Bounded.** Fixed phase count, fixed round count, fixed exit gates. No open-ended loop.

## 4. Architecture

### 4.1 The card (benchmark harness)

Extends `scripts/run_benchmarks.py`, `score_heldout.py` and `geom_bands.py`. New:

- `benchmarks/external/` loaders: **Text2CAD test slice** (200 prompts sampled with a fixed
  seed from the MIT-licensed test split; references rebuilt from the DeepCAD JSON to STL),
  **CADPrompt** (200 expert prompts with ground-truth STLs), **BenchCAD** (benchcad.com, the
  owner's pick, adopted per the 2026-09-15 investigation), **CAD Arena**
  (20 prompts, 4 tiers, human-facing smoke set). Each loader yields `(id, prompt, reference
  STL or None, tier)`.
- Metrics per arm and suite: converged rate, acceptance, **invalid ratio** (no STEP produced),
  Chamfer band distribution (match / valid / near-miss / fail) where a reference exists,
  median wall time, output tokens.
- Arms are declared in `benchmarks/arms.json`: model descriptor (GGUF, mmproj, ctx, port,
  reasoning setting, candidates N). Phase 0 shootout arms (final list in section 4.1a):
  `qwen3.8-27b` (thinking off and on), `gemma-4-31b`, `qwen3-coder-30b-a3b`, plus up to
  three more from the survey, `flash-next` (reference only), `qwen2.5-coder-7b` (floor).
  Each arm is one GGUF download (18 to 23GB); losers' files are deleted after the card so
  the winner's bf16 base fits on the NVMe.
#### 4.1a Phase 0 shootout arms (survey 2026-09-15)

| Arm | Params | Arch | Licence | Q4 on disk | Why it is in |
|---|---|---|---|---|---|
| `qwen3.8-27b` (control) | 27B dense hybrid | 16 attention + 48 DeltaNet layers | Apache 2.0 | 17.9GB, on disk | Current strong rung; strongest published agentic-coding scores; hardest to fine-tune (flash-linear-attention kernels) |
| `gemma-4-31b` | 31B dense | standard attention | Gemma licence (custom) | 18.8GB, on disk (22.9GB in use) | Only arm with a measured local CAD win (correct build123d first try, 14/14 probes); simplest to fine-tune; licence must be checked before publishing a fine-tune |
| `qwen3-coder-30b-a3b` | 30B total, 3B active | MoE | Apache 2.0 | ~18GB, download | Purpose-built coder; fast (3B active); Unsloth QLoRA quoted at 17.5GB |
| `glm-4.7-flash` | 31B total, 3B active | MoE | MIT | ~19GB, download | Newer and reportedly stronger than the Qwen coder on agentic benchmarks; QLoRA VRAM unconfirmed |
| `devstral-small-2` | 24B dense | standard attention | Apache 2.0 | 14.5GB, download | Agentic-coding specialist close to the observe-edit loop; smallest footprint, most room for few-shots |
| `gpt-oss-20b` | 21B total MoE | native MXFP4 | Apache 2.0 | ~12GB, re-download | Cheapest training loop (QLoRA 14GB, no bf16 download); measured 184 tok/s here |
| `flash-next` (reference) | 125B total, 6B active | MoE hybrid | Apache 2.0 | on disk | Quality-per-token ceiling on this box; not trainable here, teacher arm only |
| `qwen2.5-coder-7b` (floor) | 7B dense | standard | Apache 2.0 | on disk | Current fast rung; shows what the low rung will later have to reach |

Excluded on size or fit: Qwen3-Coder-Next 80B, Devstral 2 123B, Kimi K2.7, Nemotron 3 Super.
Excluded on evidence: Codestral 22B (completion-tuned, mid-tier), DeepSeek-Coder-V2-Lite
(2024-era), Gemma-4-26B-A4B (no coding scores found), Seed-Coder-8B (fast-rung size).
New downloads total about 64GB; the NVMe has 125GB free. Losers are deleted after the card.

- Output: `benchmarks/results/card/<date>.json` and `card.md`; the Lab tab renders the latest.
- Terms: **Chamfer distance** is the average nearest-point distance between two surfaces, the
  field's fidelity metric. **Invalid ratio** is the share of prompts producing no solid.

### 4.2 The maker server

A second llama-server user unit, `maker-server`, started on demand by the CAD engine's swap
hook (generalised from `_pause_default_server_for`): stop `qwen38-server`, start
`maker-server` with the configured GGUF at a CAD-sized context (16k), run the build, restore.
If the maker model leaves room (the 27B at 16k does, a dense 31B does not) the visual critic
coexists; otherwise the critic runs sequentially after the coder, which Phase 1 measures. `~/.openclaw/cad.json` gains a `maker` block: `gguf`, `mmproj`,
`ctx`, `port`, `spec` (MTP on or off), `version`. Rollback is a one-line change to a previous
GGUF. The strong rung's `local:` prefix keeps working; it now targets the maker port.

### 4.3 The data engine (expert iteration)

New package `lab/` in the skill dir. **Expert iteration** (also called rejection-sampling
fine-tuning, ReST-EM, STaR): sample N programs per spec, keep those the deterministic gate
passes, fine-tune on the winners, repeat. Components:

- **Spec bank** `lab/state/specs.jsonl`: every teacher-suite spec already on disk (~450),
  local specgen from the maker model (existing `teacher_specgen.py` retargeted to local),
  and specs mined from BenchCAD and Zero-to-CAD programs (licence checked per source).
  Each spec carries tier, source, and a contamination hash set against every card suite.
- **Sampler**: best-of-N from the maker model; for specs it fails twice, a **teacher arm**
  generates candidates: the other shootout arms that scored higher on that tier, then
  Flash-Next. Teacher GGUFs that earn pairs are kept; the rest are deleted after Phase 0. Teacher output is verified exactly
  like the student's. Thinking on for codegen; temperature ladder as in GIFT.
- **Verifier**: the existing gate (`verify_expected`) plus `geom_bands` where a reference
  exists. Hard-pass and clean [spec] checks promote to gold; disputes go to the silver review
  queue (existing `review_queue.py`).
- **Ledger** `lab/state/ledger.jsonl`: one row per attempt (spec id, arm, gate result, band,
  tokens, seconds). The Lab tab and the Bridge tile read only this.
- **Compile**: `compile_sft.py` retargeted (ChatML for the chosen base's template, tier
  weighting, dedup, contamination guard, held-out split written once and frozen).
- **Scheduler** `lab-harvest.timer`: runs when the build lock is free and the gpu-proxy
  queue is empty; each unit is one spec batch (about 10 minutes); between units it yields
  if demand appeared. Bounded by a daily budget of GPU hours set in `cad.json`.

### 4.4 The trainer and shipper

- `lab/train.py`: Unsloth QLoRA on the chosen base, text-only (vision tower frozen), rank
  16 then 32, 4k context, batch 1 with gradient accumulation, checkpoint every N steps to
  `lab/runs/<round>/`, resume on restart. Offload ladder if 24GB is exceeded: Unsloth
  activation offload to RAM, then DeepSpeed ZeRO-3 offload (slow but local). RunPod is not
  on the ladder.
- `lab/ship.py`: merge the adapter into the bf16 base, convert to GGUF, quantise with an
  importance matrix to the same class as the stock file, smoke-build, measure MTP draft
  acceptance (27B only), register the file in `cad.json` with a version tag. The previous
  GGUF is kept as rollback.
- Training rounds are kanban cards: the scheduler files "review-required: round N ready
  (M new pairs)"; the owner approves from the Bridge or the Lab tab; the round runs at night.

### 4.5 The Lab tab (web UI)

Fourth surface beside the build page in `webui/`: the latest card as a table with arm
comparison, training runs with status and loss curve, the review queue (accept / reject /
gate-bug with render and code side by side), and job controls (run card, run harvest unit,
start approved round) behind the existing confirm pattern. Routes under `/api/lab/*`, reading
the ledgers and results files; no new state store.

### 4.6 Hermes and the Bridge

A **Lab** department tile (`~/bridge/bridge/collectors/lab.py`) reading the ledger, the latest
card and the run status; kanban cards for round approvals; a Morai tool `lab_status`
(same plugin pattern as `gpu_status`). The loop itself is deterministic timers and scripts,
not an LLM worker, per the "one bounded question per task" rule. The lead name comes from
`~/agent-names.md`, chosen by the owner at Phase 5.

## 5. Phases and exit gates

| Phase | Work | Exit gate |
|---|---|---|
| **0 Instruments** | Re-measure the strong rung on the internal suites on the 3090; external loaders; arms file; Phase-0 card with the four arms; stale labels and README claims fixed; maker-server unit and swap hook (stock GGUF) | Card reproduced by one command with 3 local models on 2+ public suites; base model chosen by rule (section 6) |
| **1 Agent lift** | Cheap wins, each A/B'd on the card: thinking on for strong-rung codegen, best-of-N on the strong rung, Gemma-4-12B (or MiniCPM-V) as critic within the VRAM budget, strong-rung auto-escalation in fluid, retrieval on/off | Best config is the default; lift table in the card; no change to weights yet |
| **2 Training spike** | bf16 base download (NVMe), Unsloth stack in `lab/.venv`, one epoch on the existing 353 pairs at 4k ctx; adapter to GGUF via ship.py; VRAM, hours, MTP acceptance before and after; card on the spike model | Go / no-go with numbers; offload route recorded if used |
| **3 Data engine** | Spec bank, sampler, verifier, ledger, scheduler, compile; Lab tab review queue; Bridge tile | 2,000 verified pairs, at least 40% tier 3-4, ledger complete, harvest ran 3 nights unattended |
| **4 Rounds** | Three fixed train-eval-ship rounds | Ship rule per round (section 7); stop after round 3 or when a round gains under 2 points |
| **5 Release** | README and CITATION rewrite, GitHub push, model card, Hermes cron and kanban lane own the loop, `/doc-refresh` | `v1.0` tagged; loop runs one week unattended with no owner intervention |

Sessions, not calendar dates, are the unit; each phase is at least one session and its exit
gate is verified with pasted output before the next begins.

## 6. Base model selection rule (end of Phase 0)

Public suites rank first (owner rule 2026-09-15: official benchmarks, not our own gradings).
Rank arms by invalid ratio on the public suites (CADPrompt, BenchCAD where adopted,
Text-to-CadQuery), then by the Chamfer match-band share on those suites; the internal
suites only break remaining ties. Then apply two tie-breaks inside a 3-point band: prefer the
faster arm (a 3B-active MoE at 100+ tok/s over a dense 31B at 25 tok/s) and the easier-to-train
arm (standard attention over DeltaNet; Apache or MIT over the Gemma licence, because 1.0
publishes a model card). An arm displaces the 27B control only if it wins outside the band or
ties inside it while being faster and easier to train. Only the winner's bf16 base is
downloaded (54 to 61GB; one at a time on the NVMe). The rule is applied in writing on the
card before Phase 2 starts, so the choice is auditable.

## 7. Ship rule (each round in Phase 4)

A round ships only if all hold on the card, same seed and settings as the stock base:

- Internal acceptance up by at least 3 points.
- External invalid ratio not worse, Chamfer match band not worse.
- Held-out (`heldout-cqe` plus the frozen val split) not worse.
- For the 27B: MTP draft acceptance at or above 60%, else the maker server runs with MTP off
  and the wall-time cost is recorded.

## 8. Risks and how each is retired

| Risk | Retired by |
|---|---|
| The winner's QLoRA does not fit in 24GB at 4k ctx | Phase 2 spike; offload ladder; shorter ctx (build123d programs are under 2k tokens) |
| DeltaNet layers untrained (missing fla / causal-conv1d) | Spike asserts LoRA targets include the linear-attention modules; count trainable params per layer type |
| MTP head miscalibrated after merge | Measured in ship.py; MTP-off fallback |
| llama.cpp LoRA on qwen35 layers | Not used; merge-then-requantise only |
| Public suite unavailable or licence unclear | Loader skips and the card notes it; at least two suites required |
| Training data contaminated by card specs | Hash guard at harvest and compile; card refuses to run on a set that overlaps |
| GPU contention with Morai and family | Pre-emptible units; daily GPU-hour budget; rounds only at night after approval |

## 9. Non-goals

Low-rung (7B) training, GRPO or any online RL, the 16GB ship target, new CAD languages,
Onshape or mesh features, and any cloud teacher. On-policy distillation with the gate as
verifier (OPDVR-style) is a candidate for a later campaign, not this one.

## 10. Sources

CAD-Coder (arXiv 2505.19713), CADPrompt/CADCodeVerify (2410.05340), Text2CAD (2409.17106),
BenchCAD (2605.10865), CAD Arena (cadarena.dev), Yue et al. RLVR limits (2504.13837),
ReST-EM, STaR, Unsloth requirements and Gemma 4 guide, Axolotl Qwen3.5 docs, OPDVR
(2608.24696). Research notes for this spec are in the 2026-09-15 session record.

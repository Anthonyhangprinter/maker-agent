# Phase 3 "Data engine" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the unattended expert-iteration loop that turns the maker arm's own verified builds into training pairs, and run it until the bank holds 2,000 verified pairs with at least 40% tier 3-4, with a ledger the Lab tab, the Bridge and Morai can read.

**Architecture:** Everything lives in `lab/` beside the Phase 2 trainer and shipper and runs on the maker arm through the existing engine, fully local. A **spec bank** (`lab/state/specs.jsonl`) is built once from the 414 teacher specs plus local spec generation on the maker arm, every row tier-tagged and contamination-keyed against every card suite. A **harvest unit** (`lab/harvest.py`, 10 minutes, one systemd timer) samples N candidates per spec in-process (the `gift_sample.py` loop generalised to the bank), verifies each candidate with the deterministic gate and, where a reference exists, the Chamfer bands, and writes one **ledger** row per attempt (`lab/state/ledger.jsonl`) and one **pair** per verified winner (`lab/state/pairs.jsonl`). Specs the student fails twice go to a **teacher pass** (the `gemma-4-31b-think` arm, then Devstral) verified the same way. A **compiler** (`lab/compile.py`) turns pairs into the Gemma 4 prompt/completion rows Phase 2's trainer already consumes, with tier weighting, dedup, the contamination guard and a frozen validation split. The Lab tab gains a harvest panel and the review queue; the Bridge gets a Lab tile and Morai a `lab_status` tool.

**Tech Stack:** Python 3.12 (system python for the engine side, `lab/.venv` only for training), the existing engine API (`cad_engine.generate_code_raw`, `retrieval_notes_for`, `verify_expected`, `parse_facts`, `_acquire_build_lock`, `_ollama` local branch), `scripts/geom_bands.py`, `scripts/harvest_census.py`, `lab/data.py`, systemd user units, FastAPI web UI, the Bridge collectors, pytest.

**Spec:** `docs/MAKER-1.0-CAMPAIGN.md` sections 4.3, 4.5, 4.6, 5 (Phase 3 row), 7, 8. Phase 2 decision `benchmarks/results/card/phase2/DECISION.md` (why the old pairs are below the model). Interface map from the 2026-09-19 scan (fluid result JSON fields, sftpairs format, compile_sft stages, gift_sample loop).

## Global Constraints

- All local. Spec generation, sampling, teacher passes and verification run on the maker arms through `cad_engine._ollama` `local:` models only. No cloud call anywhere in `lab/`; `_cloud_chat` is never called.
- One GPU. A harvest unit runs only when `~/.openclaw/cad-build.lock` is free and the gpu-proxy has no queued requests; it keeps the arm warm for its whole unit (`CAD_KEEP_MAKER=1`) and restores the resident at unit end. Night window and daily GPU-hour budget come from `~/.openclaw/cad.json` (`lab` block, defaults below).
- Contamination: every bank spec carries `key = harvest_census._key(spec)` and is refused if it matches `suite_keys()` or the per-suite unique-slug rule (`suite_slug_counts`); the compiler re-checks. `CAD_BENCH=1` is set for every harvest build so the engine never self-promotes into `cad-examples.jsonl` or `cad-sftpairs.jsonl` (the bank has its own files).
- The training pair records the EXACT system prompt and user prompt the candidate was generated from (retrieval notes included), so the compiler never has to reconstruct a prompt. Pairs are Gemma's own outputs or a teacher arm's, never the old 7B-era corpus (which is excluded by default and can be opted in per round for the ablation).
- Ledger and pairs are append-only JSONL; the Lab tab, the Bridge and Morai read only those files plus `lab/state/status.json` (rewritten by every unit).
- Never `pkill -f` a pattern in your own command line; long jobs launch from script files via `setsid nohup`; no git stash/checkout during runs; no em dashes in user-facing copy; commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `python3 -m pytest tests/ -q` keeps only the pre-existing `tests/test_n1_offline.py` failure, and full suites run only when no GPU job holds the CAD lock.
- `cad.json` `lab` block (read through `cad_v5.config.lab_config()`, defaults when absent): `{"harvest": {"night_start": "22:00", "night_end": "07:00", "day_allowed": true, "hours_per_day": 12, "unit_minutes": 10, "candidates": 3, "temps": [0.2, 0.5, 0.8], "max_pairs_per_spec": 2, "teacher_arms": ["gemma-4-31b-think", "devstral-small-2"]}}`.

---

## File map

| File | Responsibility |
|---|---|
| `scripts/arms.py` (modify) | `use` saves the pre-run `maker` block to `~/.openclaw/cad.json.pre-arm`; `restore` re-applies it (and `--disable` keeps today's behaviour) |
| `cad_v5/config.py` (modify) | `lab_config()` with the defaults above; `LAB_STATE = <skill>/lab/state` |
| `lab/specgen.py` (create) | local spec generation on the maker arm: families and tiers from `scripts/teacher_specgen.py` retargeted to `engine._ollama(CODE_MODEL_STRONG, ...)`, JSON-array output, per-family seeds, dedup, contamination refusal |
| `lab/specbank.py` (create) | builds/extends `lab/state/specs.jsonl`: rows `{id, spec, tier, group, source, key, added}`; imports the 414 teacher specs; `stats` per tier and source |
| `lab/harvest.py` (create) | the unit: pick specs, sample N candidates, execute + inspect + gate + band, write ledger and pairs, teacher passes, budget/window/lock checks, `--status`, `--once --spec-id` for tests and smoke |
| `lab/harvest_unit.sh` (create) | wrapper for the timer: `python3 lab/harvest.py --unit` under `flock` on `lab/state/harvest.lock` |
| `deploy/lab-harvest.service`, `deploy/lab-harvest.timer` (create) | user units, timer every 15 minutes; the script decides whether a unit may run |
| `lab/compile.py` (create) | pairs -> `lab/data/round<N>/{train,val}.jsonl` via `lab.data.render_pairs` on messages built from the stored prompts; tier weighting, dedup, contamination, frozen split `lab/state/val_specs.json`; `--report` |
| `webui/app.py`, `webui/static/index.html` (modify) | Lab tab: harvest status panel (`/api/lab/harvest`), review queue (`/api/lab/review`, POST decisions with the confirm header), controls (`run one unit`, `pause/resume timer`) |
| `~/bridge/bridge/collectors/lab.py`, tile wiring (create, outside repo) | Lab tile reading `lab/state/status.json` and the latest card |
| `~/.hermes/plugins/lab/` (create, outside repo) | Morai tool `lab_status` returning the status JSON as text |
| `tests/test_lab_specbank.py`, `tests/test_lab_harvest.py`, `tests/test_lab_compile.py`, `tests/test_arms.py` (modify) | offline tests with stubbed engine calls |
| `benchmarks/results/harvest/phase3/` (generated) | `STATUS.md` (pairs by tier/source, ledger completeness, nights run), `DECISION.md` |

Ledger row: `{ts, unit_id, spec_id, tier, arm, pass ("student"|"teacher1"|"teacher2"), candidate, temperature, ok, gate_hard, gate_spec, gate_adv, band, ref, chamfer_mm, tokens_in, tokens_out, seconds, build_dir, error}`.
Pair row: `{id, spec_id, spec, tier, group, source ("student"|"teacher:<arm>"), kind ("good"|"fail"), band, arm, temperature, system, prompt, code, bad_code, problem, facts, verified: {gate_hard: 0, gate_spec: 0, band}, ts, unit_id}`.
Status file: `{updated, specs: {total, by_tier}, pairs: {total, by_tier, by_source, tier34_share}, ledger: {attempts, pass_rate_by_tier, last_unit}, budget: {hours_today, hours_per_day, window, next_allowed}, nights_completed, timer_active}`.

---

### Task 1: arms.py restore re-applies the pre-run maker block

**Files:** `scripts/arms.py`, `tests/test_arms.py`

- [ ] `cmd_use` writes the current `maker` block AND a copy of `~/.openclaw/maker.env` to `~/.openclaw/cad.json.pre-arm` / `maker.env.pre-arm` before switching (2026-09-19: the Phase 2 card left maker.env pointing at the spike GGUF while cad.json said gemma-4-31b; both must travel together) (only if the file does not already exist, so nested cards keep the outermost lock-in). `cmd_restore` (also called by `run_card`'s finally) stops the maker, restarts the resident as today, then re-applies the saved block and the saved maker.env and deletes both markers; when no marker exists it falls back to today's behaviour (`enabled: false`). New flag `restore --disable` forces today's behaviour.
- [ ] Tests: use then restore leaves cad.json byte-identical; restore without a marker disables; `--disable` ignores the marker. Commit `arms: restore re-applies the pre-run maker block (Phase 2 trap)`.

### Task 2: spec bank and local spec generation

**Files:** `cad_v5/config.py`, `lab/specgen.py`, `lab/specbank.py`, `tests/test_lab_specbank.py`

- [ ] `lab_config()` in `cad_v5/config.py` with the defaults from Global Constraints (deep-merged with the `lab` block of cad.json).
- [ ] `lab/specbank.py`: `import-teacher` (reads the five `benchmarks/teacher-*/specs.json`, keeps `id` as `t:<suite>:<id>`, tier, group, source `teacher-suite`); `add` (from a JSON list, used by specgen); `stats`; every add computes `key`, refuses exact-key or per-suite-unique-slug clashes against the card suites and duplicates within the bank (by key), and appends to `lab/state/specs.jsonl` atomically.
- [ ] `lab/specgen.py`: the families and tier guidance copied from `scripts/teacher_specgen.py` (`FAMILIES`, `HARD_FAMILIES`, `_SYSTEM`) but the call is `engine._ollama(engine.CODE_MODEL_STRONG, system, prompt, no_think=False, temperature=0.8)` through the `local:` branch (thinking ON for spec writing: it is cheap and variety matters), JSON-array parsing with one repair retry, seeds = 5 random bank specs of the same group, `--target-tier34 0.45 --total 2500` loop that keeps generating families until the bank reaches the total with at least the tier 3-4 share; each batch is 20 specs; the maker arm is kept warm across the run (`CAD_KEEP_MAKER=1` plus `arms.py use gemma-4-31b`); on exit `arms.py restore`.
- [ ] Owner reference intake: `lab/specbank.py import-references [--dir ~/CAD/references]` walks `~/CAD/references/<name>/` folders containing `spec.txt` (one spec sentence, mm) and `model.step` or `model.stl`; each becomes a bank row with `source: owner-reference`, `tier` from `spec.txt`'s optional first line `tier: N` (default 3), and `reference_stl` (STEP converted to STL once via `scripts/step` tooling, stored under `lab/state/refs/<id>.stl`). Harvest treats reference rows with the strict rule (only `match` band pairs count as good; `near_miss` gives a fail pair). The folder is created with a README explaining the two files. Public datasets (Text2CadQuery, DeepCAD, Fusion 360 Gallery, ABC) may be imported the same way later, one importer per source, each with a licence line in the README and the card-suite contamination refusal; not in this plan.
- [ ] Tests (engine stubbed): teacher import counts 414, a planted card spec is refused, a duplicate is refused, tier share maths, the JSON repair path.
- [ ] Run for real (GPU, about 1-2 h): `lab/specbank.py import-teacher` then `lab/specgen.py --total 2500 --target-tier34 0.45`; paste `lab/specbank.py stats`. Commit code; `lab/state/` is git-ignored except `specs.jsonl` and `val_specs.json` which ARE committed (they are the reproducibility record).

### Task 3: the harvest unit, ledger, pairs, timer

**Files:** `lab/harvest.py`, `lab/harvest_unit.sh`, `deploy/lab-harvest.service`, `deploy/lab-harvest.timer`, `tests/test_lab_harvest.py`

- [ ] Sampling loop per spec, in-process (model `engine._ACTIVE_CODE_MODEL = "local:<arm alias>"`): `notes = engine.retrieval_notes_for(spec)`; for each of `candidates` temperatures: `code = engine.generate_code_raw(spec, notes, temperature=t)` (capture the exact `system` and `prompt` strings the engine sends: add a tiny `engine.last_prompt()` accessor if none exists, recorded in the pair), execute through the same path `gift_sample.py` uses (`scripts/step` -> STEP, `scripts/inspect` -> facts via `engine.parse_facts`, `engine.reconcile_expected` + `engine.verify_expected(facts, expected, spec=spec)`), band via `geom_bands.score_against_reference` when the bank row has `reference_stl` (teacher specs do not; card suites are never sampled), one salvage attempt on a crash exactly like fluid's (`engine.diagnose` hint + one revise) counted as a separate candidate row.
- [ ] Verdicts: `ok and gate_hard == 0 and gate_spec == 0 and (band in (None, "match"))` -> pair kind `good` (at most `max_pairs_per_spec` distinct codes per spec, prefer lower temperature, dedup by code hash); `ok and gate_hard == 0 and gate_spec > 0` -> silver review row in `lab/state/review.jsonl` (spec, code, notes, render path) not a pair; reference present and band `near_miss` -> pair kind `fail` (bad_code = candidate, code = reference code, problem string as gift_sample); else nothing. Every candidate writes a ledger row regardless.
- [ ] Spec scheduling: `lab/state/specs.jsonl` rows gain no fields; progress lives in `lab/state/progress.json` `{spec_id: {student_attempts, teacher_attempts, pairs}}`. A unit picks specs with `pairs < max_pairs_per_spec`, tier 3-4 first while `tier34_share < 0.40`, then round-robin; specs with `student_attempts >= 2` and no pair go to the teacher list; a unit is single-arm: it runs the student list on `gemma-4-31b`, or, when the teacher list has at least 20 specs, one teacher unit on the first teacher arm in `teacher_arms` (arm switch via `arms.py use <arm>` at unit start, `restore` at unit end); teacher output is verified identically and recorded `source: teacher:<arm>`.
- [ ] Gating a unit (`--unit`): refuse when the build lock is held, when `curl 127.0.0.1:8087` reports queued requests, when outside the window and `day_allowed` is false, when `hours_today >= hours_per_day` (hours counted from ledger seconds per calendar day), when the timer is paused (`lab/state/paused` file), or when the bank is exhausted; then run for `unit_minutes` wall clock (finish the current spec), restore, rewrite `status.json`. `--once --spec-id X` runs one spec now (smoke/tests) without the gates.
- [ ] Units: `deploy/lab-harvest.service` (`Type=oneshot`, `ExecStart=<skill>/lab/harvest_unit.sh`, `RuntimeMaxSec=1800`), `deploy/lab-harvest.timer` (`OnCalendar=*:0/15`, `Persistent=false`), install notes in `deploy/README.md`; `harvest_unit.sh` takes `flock -n lab/state/harvest.lock`.
- [ ] `--status` prints the status JSON; `nights_completed` counts calendar nights with at least `unit_minutes * 6` minutes of ledger time inside the window.
- [ ] Tests (engine stubbed, temp state dir): verdict table, max pairs per spec, teacher promotion after two failures, tier-first scheduling, window/budget/lock refusals, ledger and pair row shapes, status maths.
- [ ] Smoke (GPU): `python3 lab/harvest.py --once --spec-id <a tier-3 id>` then one real `--unit` of 10 minutes; paste the ledger tail and status. Install and enable the timer only after the smoke passes. Commit.

### Task 4: compiler

**Files:** `lab/compile.py`, `tests/test_lab_compile.py`, `lab/README.md`

- [ ] `lab/compile.py --round N [--include-legacy] [--tier12-cap 0.5] [--report]`: reads `lab/state/pairs.jsonl` (and, only with `--include-legacy`, the old `~/.openclaw/cad-sft-train.jsonl` rows re-verified through the gate), builds `messages` rows `{system, user: prompt, assistant: code}` from the stored prompt strings (fail pairs use the revise prompt shape exactly as `compile_sft.to_chatml` does for GIFT-FAIL rows), drops duplicates by `(key, sha1(code))`, caps tier 1-2 at the given share by random down-sampling with a fixed seed, applies the contamination guard, and renders through `lab.data.render_pairs` (prompt/completion, checkpoint template) into `lab/data/round<N>/`. The validation split is by spec key: `lab/state/val_specs.json` is written ONCE (5% of bank spec keys, seed 3407) and reused by every round; a spec in the val set never appears in train.
- [ ] `--report` writes `lab/data/round<N>/compile_report.json` (counts by tier/source/kind, dropped reasons, token p50/p95 from data.py) and `data_meta.json` (template path + sha256, row counts, source file sha256s: the Phase 2 review's finding 29).
- [ ] Tests: split frozen across two runs, tier cap, legacy excluded by default, contamination drop, fail-pair prompt shape.
- [ ] `lab/README.md`: a "Data engine" section (bank, harvest, timer, compile, the cad.json `lab` block, status and review files). Commit.

### Task 5: Lab tab, Bridge tile, Morai tool

**Files:** `webui/app.py`, `webui/static/index.html`, `tests/test_webui_lab.py`; outside the repo `~/bridge/bridge/collectors/lab.py` (+ tile registration and a test), `~/.hermes/plugins/lab/{plugin.yaml,__init__.py}`

- [ ] `/api/lab/harvest` returns `lab/state/status.json` plus the last 20 ledger rows; `/api/lab/review` lists `review.jsonl` rows with render URLs; `POST /api/lab/review/<id>` with `{verdict: accept|reject|gate-bug, note}` behind the existing confirm-header pattern appends to `lab/state/review_decisions.jsonl`, and `accept` promotes the row into `pairs.jsonl` as `source: human-accepted`; `POST /api/lab/harvest/unit` runs one unit now (spawns `harvest_unit.sh`), `POST /api/lab/harvest/pause|resume` toggles the `paused` file. The Lab view shows: bank and pairs counts by tier with the 40% tier 3-4 bar, budget for today, last unit, the review queue with render + code side by side and the three buttons, and the controls.
- [ ] Bridge: a `lab` collector reading `status.json` and the latest card (`benchmarks/results/card/latest/card.json`) into a Lab tile beside the others (follow `bridge/collectors/infra.py`'s shape and the tile registration in the floor config); a test on a fixture status file. Morai plugin `lab` with tool `lab_status` (returns the status JSON as compact text; same pattern as `gpu_status`). Restart `bridge.service` and `hermes-gateway` and verify (`curl 127.0.0.1:8091/api/state | grep -c lab`, `hermes tools` lists `lab_status`).
- [ ] Commit the repo side; the Bridge side commits in `~/bridge` on a branch `lab-tile`.

### Task 6: run the engine to the exit gate and decide

- [ ] Enable `lab-harvest.timer`; the controller supervises nightly (ScheduleWakeup heartbeat each morning: read `--status`, the ledger error tail, the resident state); fix anything that breaks and note it in the ledger of this plan.
- [ ] Exit gate check: `pairs.total >= 2000`, `tier34_share >= 0.40`, every ledger row complete (no null seconds/ok), `nights_completed >= 3`. Write `benchmarks/results/harvest/phase3/STATUS.md` (the status JSON rendered, per-tier pass rates for student and teacher passes, pairs per hour, GPU hours used) and `DECISION.md` (gate met or not, what to fix, the Phase 4 round plan: round 1 = compile round 1 without legacy, train with Phase 2's `spike.sh --data lab/data/round1`, ship, card, ship rule section 7).
- [ ] Run `python3 lab/compile.py --round 1 --report` so Phase 4 can start immediately; commit `lab/state/specs.jsonl`, `val_specs.json`, the results dir (`git add -f`).

---

## Self-review

- Spec 4.3: spec bank (Task 2), sampler with the teacher ladder (Task 3), verifier = gate + bands (Task 3), ledger (Task 3), compile (Task 4), scheduler timer with lock/queue/budget gating (Task 3). Zero-to-CAD and BenchCAD program mining is dropped by ruling: licences unchecked and the bank reaches 2,500 specs from teacher suites plus local specgen alone.
- Spec 4.5 Lab tab review queue and controls: Task 5. Spec 4.6 Bridge tile and `lab_status`: Task 5 (kanban round cards belong to Phase 4).
- Spec 5 exit gate: Task 6 measures all four conditions.
- Spec 8 risks: contamination (bank keys + compiler re-check + CAD_BENCH), GPU contention (lock, queue, window, budget, unit bound), self-training on wrong geometry (hard gate AND clean spec checks AND match band, silver queue for disputes), data below the model (student/teacher outputs only; legacy excluded by default).
- Phase 2 trap closed in Task 1 before any card runs again.

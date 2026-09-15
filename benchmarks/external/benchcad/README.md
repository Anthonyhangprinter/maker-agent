# BenchCAD (official harness)

`benchcad.com`, code MIT-licensed (`github.com/BenchCAD/BenchCAD-main`), data CC-BY-4.0 on the
HF Hub (`BenchCAD/BenchCAD`). We run its harness, unmodified apart from one new model adapter,
against each Maker Agent candidate model ("arm", see `scripts/arms.py`) served by our
`maker-server` (llama.cpp, OpenAI-compatible, `http://127.0.0.1:8088/v1`). No agent loop: the
harness calls the model directly with code/an instruction/a 4-view render and scores the raw
response. This is the primary "official, not home-made" yardstick per the owner's 2026-09-15
direction (see `.superpowers/sdd/2026-09-15-phase0-instruments/progress.md`, Task 5b ruling).

## Layout

- `BenchCAD-main/` — the clone (`git clone --depth 1 https://github.com/BenchCAD/BenchCAD-main`),
  **git-ignored** (`.gitignore`: `benchmarks/external/benchcad/BenchCAD-main/`). Its own `uv sync`
  venv (`.venv/`, pinned `cadquery==2.3.0`, `cadquery-ocp==7.9.3.0`, `numpy==1.26.4`, ...) lives
  entirely inside the clone — nothing from it is installed into the system or our project python.
- `local_adapter.py` (this dir) — our source of truth for the local model adapter. It is copied
  into the clone at `BenchCAD-main/benchcad_core/models/local_adapter.py` (also `git add`ed
  there... no — the clone itself is git-ignored, so only THIS copy is committed). **To recreate
  the clone**: `git clone`, `uv sync`, `cp .env.example .env`, then
  `cp benchmarks/external/benchcad/local_adapter.py benchmarks/external/benchcad/BenchCAD-main/benchcad_core/models/local_adapter.py`,
  and apply the two dispatcher edits in "Registering the adapter" below (small enough to keep as
  documented diffs rather than a patch file).
- `scripts/run_benchcad.py` (repo `scripts/`, not here) — our runner: switches arms via
  `scripts/arms.py`, invokes the harness per (arm, task), reads its real result artefact, writes
  our own summary JSON.

## Recreating the clone (Step 1, as run 2026-09-15)

```bash
cd benchmarks/external/benchcad && mkdir -p /dev/null 2>/dev/null  # (benchcad dir already exists in-repo for local_adapter.py + this README)
git clone --depth 1 https://github.com/BenchCAD/BenchCAD-main
cd BenchCAD-main && uv sync && cp .env.example .env
```

`uv sync` took ~50s cold, installed ~140 packages (vtk, cadquery/OCP, scipy/pandas, openai/
anthropic/google-genai SDKs, etc.) into `BenchCAD-main/.venv/` only.

## Facts verified against the clone (vs the 2026-09-15 web-investigation brief)

- **Repo/license/data**: all as briefed — MIT harness, `BenchCAD/BenchCAD` on HF, CC-BY-4.0.
  Config record counts not independently re-verified (would require a full pull); not needed for
  the offline smoke.
- **Task keys differ from the brief.** `benchcad.py --task` choices are `codeedit`, `qa`,
  `vision2code` (NOT `codeqa` — that name doesn't exist in the harness). Our own
  `scripts/run_benchcad.py --tasks` flag keeps the plan's `codeqa` spelling for readability and
  maps it internally to the harness's `qa`.
- **`--num`/`--model`/`--seed` match the brief.** `--num` is `all` or a positive int (default `5`
  = a quick smoke); `--model` takes one or more model ids; `--seed` makes an N-record sample
  reproducible.
- **Model dispatch is a prefix router (`benchcad_core/models/__init__.py:_route`), not a
  name→module registry file** — `openrouter_adapter.py` is not "registered" anywhere separate
  from that one function. Added: `if model == "local": return "local"` in `_route()`, and an
  `elif provider == "local": from .local_adapter import generate` branch in `call_model()`. Two
  small, surgical edits — well under the "≤3 harness files" stop-rule (only `__init__.py` and the
  new `local_adapter.py` itself were touched).
- **`openrouter_adapter.py` is a very close template** — OpenAI SDK, `chat.completions.create`,
  `image_url` content parts with base64 data URIs, `usage_from_openai()` helper already shared.
  Dropped the OpenRouter `:reasoning=<spec>` suffix parsing (not how this endpoint's reasoning
  knob works — our template-kwargs approach below replaces it) and the `openai_adapter.py`
  Responses-API path is confirmed unused for local (it would NOT work against llama-server, per
  the brief — untouched, not needed).
- **The real per-run result artefact is `<task_dir>/results_prod/results.jsonl`** (or
  `results_test/results.jsonl` for `configs/test.yaml`), JSON Lines, one row per record — NOT
  stdout scraping. `run_benchcad.py` reads this file directly (see "Result file format" below).
  There is no CLI flag on any task's `main.py` to redirect `out_dir`; `benchcad.py` always invokes
  `main.py --config configs/prod.yaml`, whose `out_dir: results_prod` is fixed in the YAML.
- **Gotcha found and worked around**: every arm is called through the harness as the same model
  id `"local"` (the real arm identity lives only in our `BENCHCAD_MODEL`/`BENCHCAD_TEMPLATE_KWARGS`
  env vars, invisible to the harness). The harness's own results.jsonl is keyed by
  `(model, record_id)` (or `(mode, model, record_id)` for CodeEdit) with overwrite-on-rerun
  semantics — so running arm B after arm A would silently clobber arm A's rows for any record_id
  the two runs share. `run_benchcad.py` deletes `results_prod/results.jsonl` immediately before
  each (arm, task) subprocess call, so every run starts from a clean file and its rows are
  unambiguously this arm's.
- **HF pull is real but small for a 3-record smoke**: the QA `--num 3` smoke pulled one 200-part
  chunk of the `qa_2400.parquet` config into `~/.cache/huggingface/hub/datasets--BenchCAD--BenchCAD/`
  — 31MB total, wrote 3 records / 36 QA pairs into `QA/data/records.jsonl` (28KB). Allowed to
  download per the task's stop-rule (CC-BY-4.0 data, small). A `--num all` pull for the full
  150-sample run (Task 12) will be considerably larger — not measured here.
- **`.env.example` exists and was copied to `.env`**; unused by the local adapter (no API key
  needed — `BENCHCAD_MODEL`/`BENCHCAD_BASE_URL`/`BENCHCAD_TEMPLATE_KWARGS` are read directly from
  the environment `run_benchcad.py` passes to the subprocess), but other adapters (openai/
  anthropic/gemini/openrouter) do need their key in it if ever exercised for a cross-check.

## Result file format

Each task's `main.py` writes `<out_dir>/results.jsonl` (JSON Lines, overwritten in place on
rerun by its dedup key). Row keys, verified live against the committed `test_data/` smoke:

| Task (harness key) | Score field | Status field | Status values (verified) |
|---|---|---|---|
| `qa` (our `codeqa`) | `qa_score` (ratio accuracy, 0-1) | `status` | `ok`, `no_qa`, `api_fail`, `parse_fail` |
| `codeedit` | `norm_iou` (headroom-normalised IoU improvement) | `status` | `ok`, `api_fail`, `no_code`, `exec_fail`, `score_fail` |
| `vision2code` | `score` (primary; `iou` also present raw) | `status` | same set as `codeedit` |

Common row fields across all three: `record_id`, `model`, `status`, `lat_s`, `prompt_tokens`,
`completion_tokens`, `reasoning_tokens`, `total_tokens`, `cost_usd` (null — the `local` model id
isn't in `pricing.yaml`, which the pricing module treats as "unknown price → null", not an
error), `err`. CodeEdit/Vision2Code additionally have `code_path`/`step_path`/`png_path` per-record
artifacts under `<out_dir>/outputs/...`.

`exec_rate` (our own derived field, not a harness field) = the fraction of rows whose `status` is
`ok` or `score_fail` — both mean CadQuery execution produced a STEP file; `score_fail` just means
the IoU scorer itself then threw. QA has no execution step, so its `exec_rate` is always `None`.

## Registering the adapter (for recreating a fresh clone)

In `benchcad_core/models/__init__.py`:
```python
def _route(model: str) -> str:
    if model == "local":
        return "local"
    ...  # unchanged

def call_model(...):
    ...
    elif provider == "local":
        from .local_adapter import generate
    ...  # unchanged
```

## Smoke test (2026-09-15, Task 5b — control arm, QA task, offline test_data)

```bash
python3 scripts/arms.py use qwen3.8-27b-nothink
cd benchmarks/external/benchcad/BenchCAD-main/QA
BENCHCAD_MODEL=qwen3.8-27b BENCHCAD_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  uv run python main.py --config configs/test.yaml --model local --limit 3
python3 ../../../../scripts/arms.py restore
```

Used the committed offline `test_data/` (4 records) via `configs/test.yaml` instead of the
brief's literal `--task codeqa --num 3` (that flag/value combination doesn't exist — see above);
this proves the adapter works end to end with **zero** network access. Output (verbatim):

```
config: configs/test.yaml
data:   test_data
out:    results_test
mode:   code
gen:    max_tokens=16000 timeout=600s
runs:   3 record(s) × 1 model(s)
  [local] 1/3 threaded_adapter_000240_s4420 ... ok         qa_score=0.917  (3.2s)
  [local] 2/3 motor_end_cap_000500_s4420 ... ok         qa_score=0.841  (2.0s)
  [local] 3/3 keyhole_plate_000358_s4420 ... ok         qa_score=0.595  (2.5s)

results  → results_test/results.jsonl  (3 rows)
summary  → mean qa_score per model:
    local  n=3   mean_qa=0.784
tokens   → 2,378 total
```

A second smoke ran the real `scripts/run_benchcad.py` itself (`--arms qwen3.8-27b-nothink --tasks
codeqa --num 3 --seed 42`), which goes through `benchcad.py --task qa` and therefore does pull
from HF (see above); it produced the same ~0.78 mean score (different 3-record sample, `prod.yaml`
+ `--seed 42` vs the offline `test_data`'s fixed 4) and correctly restored the resident afterward.
Vision2Code and the full 150-item sample are explicitly out of scope for this task (Task 12).

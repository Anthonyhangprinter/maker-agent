# lang-ab: which CAD language does the local model write best

A small A/B harness. Same spec pool, same model, four arms: `--model maker` (default) uses
whichever arm the GPU window has already loaded (currently `gemma-4-31b` per
`~/.openclaw/cad.json`'s `maker` block); `--model resident` uses the always-on Qwen3.8-27b
behind the :8085 gpu-proxy instead (see `--model resident` below).

- `b123d` -- the production path: `cad_engine.generate_code_raw`'s exact system prompt and
  retrieval few-shots, ON.
- `b123d-nofs` -- identical, few-shots OFF.
- `cadquery` -- a hand-written system prompt (`lang_ab.CADQUERY_SYSTEM`), no few-shots, run
  in an isolated venv (`setup_cq_env.sh`, `run_cq.py`).
- `openscad` -- `scripts/openscad_gen.py`'s own system prompt, one shot, no repair turn,
  compiled with the local OpenSCAD AppImage.

Every arm is scored the same way: `geom_bands.score_against_reference` (Chamfer /
Hausdorff-95 / volume / bbox, GIFT-style bands: match / valid / near_miss / fail) against
the reference geometry that ships with `benchmarks/cadprompt/` and
`benchmarks/text2cadquery/` -- the only two suites in this repo with reference STLs.

## The unfairness note

This is not a fair fight between languages. The `b123d` arm carries months of CAD-specific
engineering the other three arms get none of: a long, iterated system prompt
(`cad_engine._CODE_SYSTEM`), a retrieval few-shot corpus of verified builds
(`cad-examples.jsonl`), and a regex patch layer that silently fixes common API
hallucinations (`cad_engine._patch_code`) before the code is ever executed. `cadquery` and
`openscad` get a plain system prompt and nothing else.

That means:

- A **cadquery or openscad win** (built more, matched more) is real evidence the model
  writes that language more reliably out of the box -- conservative, since it happened
  despite carrying none of the b123d arm's scaffolding.
- A **cadquery or openscad loss** is **inconclusive** on the language itself: it may just
  be the missing scaffolding, not the language. `b123d-nofs` exists to separate the two --
  it is the like-for-like leg (same model, same lack of few-shots, same lack of
  hand-written repair patches for cadquery/openscad's failure modes) and is the fairer
  baseline for a cadquery/openscad comparison than plain `b123d`.

## Running it

The default `--model maker` leg needs a real model call per (arm, spec) against whichever
arm the GPU window has loaded, so it must run inside a GPU window -- it refuses otherwise
(`lab.ship.require_gpu_window`). It does not switch arms or evict services itself:
`lab._armwindow.arm_window()` (the same context manager `lab/harvest.py`/`lab/specgen.py`
use) does that, so the controller only needs to launch it inside `lab/gpu_window.sh`:

```
PYTHONUTF8=1 lab/gpu_window.sh python3 benchmarks/lang-ab/lang_ab.py \
    --arms b123d,b123d-nofs,cadquery,openscad --specs 40 --run-id 2026-09-19-langab
```

### `--model resident`

The same experiment against the RESIDENT model (Qwen3.8-27b), which is already up all day
behind the :8085 gpu-proxy independent of any maker arm. This leg is a plain chat client: no
GPU window, no `arm_window`, no service started or stopped anywhere. It checks
`GET http://127.0.0.1:8086/health` (the resident's own port) before the first call and exits
2 if that fails. A non-200 or connection error on an individual call is recorded as that
attempt's error and the run continues; 5 consecutive call failures abort the run (exit 3) --
the resident may have been taken down by another job.

```
PYTHONUTF8=1 python3 benchmarks/lang-ab/lang_ab.py --model resident \
    --arms b123d,b123d-nofs,cadquery,openscad --specs 40 --run-id 2026-09-19-langab-resident
```

Every row carries which model produced it (`gemma-4-31b` for `--model maker`'s default
config, `qwen3.8-27b` for `--model resident`); `REPORT.md`'s header and `report.json`'s
top-level `model` key both read it back from the rows, not from `meta.json`.

Before the first real run, build the isolated CadQuery venv once (network required):

```
bash benchmarks/lang-ab/setup_cq_env.sh
```

This installs cadquery 2.8.x into `benchmarks/lang-ab/.venv-cq`, never into the system
Python the production CAD engine depends on (`build123d 0.10.0` + `OCP 7.8.1.1`) -- the
script verifies both before and after that the system versions are unchanged.

Flags:

- `--arms` comma list of `b123d,b123d-nofs,cadquery,openscad`, or `all` (default: all four).
- `--specs N` number of specs drawn from the two public suites, stratified and seeded
  (default 40).
- `--seed N` the draw's seed (default fixed; change it to draw a different sample).
- `--run-id NAME` reuse a run-id to resume: a rerun with the same id skips every
  (arm, suite, spec) already in that run's `rows.jsonl`.
- `--suite-root DIR` point at a different `benchmarks/` root (used by the test suite's
  fixture suites; not needed for a real run).
- `--arm NAME` the maker arm name to load via `arm_window` (default: whatever
  `~/.openclaw/cad.json`'s `maker` block already names). Ignored under `--model resident`.
- `--model maker|resident` which model answers every call (default `maker`, unchanged
  behaviour). See `--model resident` above.

Output: `benchmarks/lang-ab/results/<run-id>/rows.jsonl` (one JSON row per attempt),
`meta.json`, `REPORT.md`, and `report.json`. Per-build artifacts (generated code, the
compiled STEP/STL) live under `results/<run-id>/builds/<arm>/<spec-id>/`.

To regenerate the report from an existing run without rebuilding anything:

```
python3 benchmarks/lang-ab/report.py benchmarks/lang-ab/results/<run-id> --print
```

## Reading the report

- **built %** -- the model's code ran and produced geometry at all (a much lower bar than
  match).
- **match %** -- `geom_bands` scored the result an exact match against the suite's
  reference (all of bbox/volume/chamfer/hausdorff-95 pass).
- **match-or-valid %** -- match plus "valid" (watertight, single solid, close but not exact
  -- GIFT's "diverse valid" band, a correct-but-differently-written part).
- **error classes** -- `none` / `syntax` / `import_or_name` / `api_misuse`
  (AttributeError/TypeError/ValueError raised from inside the CAD library, not the kernel)
  / `kernel` (OCP/Standard_/TopoDS/null-shape signatures: the geometry kernel itself
  rejected the operation) / `timeout` / `no_code` (empty model reply) / `other`.
- **paired comparison vs b123d** -- restricted to specs BOTH arms attempted, so it is never
  skewed by one arm having an easier subset. `+improved`/`-worsened` counts are the actual
  evidence at this sample size (dozens of specs); the raw percentages above are the
  headline numbers but the flips are what to trust when they disagree.
- **top error signatures** -- the 10 most common failure messages per arm, for spotting a
  single systemic bug (e.g. one hallucinated API call) versus genuinely varied failures.

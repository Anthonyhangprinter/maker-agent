# Phase 1 "Agent lift" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, one lever at a time on the Phase 0 winner (Gemma-4-31B on the maker server), the cheap capability levers that need no weight change, and lock the winning configuration as the default before Phase 2 touches training.

**Architecture:** Every lever is an A/B on the existing card: `scripts/run_card.py` gains variant labels and pass-through knobs so several configurations of the same arm coexist in one `rows.jsonl`; `scripts/fluid_gen.py` gains the engine's best-of-N first turn; the visual critic gets its own server URL so a small vision model can run beside the coder when VRAM allows, or the coder can judge its own render; the full-loop preflight stops requiring Ollama when every model is local. Levers run on a fixed stratified subset (about 130 builds, one-shot) so each A/B costs about an hour on Gemma; the lift table is written by a pure function next to `card_report.py` and shown in the Lab view.

**Tech Stack:** unchanged from Phase 0 (Python 3.12 system interpreter, llama.cpp CUDA build, systemd user units, pytest via `python3 -m pytest tests/ -q`).

**Spec:** `docs/MAKER-1.0-CAMPAIGN.md` section 5 (Phase 1 row), 4.1, 4.2, 4.5; decision `benchmarks/results/card/phase0/DECISION.md`.

## Global Constraints

- All local at runtime; no cloud calls.
- One GPU: the maker arm is Gemma-4-31B (`scripts/arms.py use gemma-4-31b`); the resident is down while a card runs; every runner restores it in a `finally`.
- Never `pkill -f` a pattern that appears in your own command line; never put a process pattern and a matching heredoc in one command; launch long jobs from a script file with `setsid nohup`.
- No git stash/checkout while a run reads the tree.
- Each lever is compared against a baseline produced with the SAME subset, seed and mode in the SAME rows file; a lift is claimed only from that table.
- Thinking-on runs are bounded (they are 5 to 6 times slower); never a full 262-build arm.
- No em dashes in user-facing copy. Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Tests: `1 failed (pre-existing tests/test_n1_offline.py), N passed` must hold, N only growing.
- Contamination guard unchanged; `text-to-cad/05` stays excluded from any ranking.

---

## File map

| File | Responsibility |
|---|---|
| `scripts/run_card.py` (modify) | `--variant LABEL`, `--candidates N`, `--no-fewshots`, `--critic MODEL`, `--subset NAME` (stratified per-suite counts); rows carry `arm = "<arm>+<variant>"` |
| `scripts/fluid_gen.py` (modify) | best-of-N first turn via `engine._pick_first_turn_candidate`; result gains `candidates` |
| `cad_v5/config.py` (modify) | `CRITIC_URL` (env `CAD_CRITIC_URL`, default = `LOCAL_CODER_URL`), `CRITIC_HEALTH` |
| `cad_engine.py` (modify) | `_ollama()` local branch posts critic calls to `CRITIC_URL`; `_installed_ollama_models()` returns an empty set with a warning instead of raising when every configured model is `local:`/`cloud/` |
| `deploy/critic-server`, `deploy/critic-server.service` (create) | optional second llama-server for a small vision critic (no `Conflicts=`), env `~/.openclaw/critic.env` |
| `scripts/arms.py` (modify) | `critic use <name> | off` writing `critic.env`, starting/stopping `critic-server`, and setting env `CAD_CRITIC_MODEL`/`CAD_CRITIC_URL` in `~/.openclaw/maker.env` companions; `benchmarks/arms.json` gains a `critics` list |
| `scripts/lift_report.py` (create) | pure: `lift_table(rows, baseline_variant, metric_keys) -> dict` and `render_lift_md(...)` |
| `webui/app.py`, `webui/static/index.html` (modify) | `GET /api/lab/lift`, lift table under the card in the Lab view |
| `benchmarks/results/card/phase1/` (generated) | `rows.jsonl`, `card.md`, `LIFT.md`, `DECISION.md` |
| tests: `tests/test_run_card.py`, `tests/test_fluid_bon.py`, `tests/test_maker_config.py`, `tests/test_arms.py`, `tests/test_lift_report.py` | offline coverage |

---

### Task 1: card variants and the stratified subset

**Files:**
- Modify: `scripts/run_card.py` (argparse in `main()`, `build_once`, `run_row`, the arm loop at ~:329)
- Test: `tests/test_run_card.py`

**Interfaces:**
- Produces: CLI flags `--variant LABEL` (default `""`), `--candidates N` (default 0 = engine default), `--no-fewshots`, `--critic MODEL` (default `""` = engine default), `--subset NAME` (`full` default; `phase1` = per-suite caps `{"cadprompt": 30, "text2cadquery": 30, "heldout-cqe": 25, "text-to-cad": 10, "organic": 5, "hard-eval": 15, "cad-arena": 12}`, first N of each suite in file order, deterministic). Rows: `arm` becomes `f"{name}+{variant}"` when a variant is given; `meta["variant"]`, `meta["knobs"]` record the knobs. `build_once` passes the knobs to the child: env `CAD_CANDIDATES=<N>` when N > 0, env `CAD_CRITIC_MODEL=<MODEL>` when given, argv `--no-fewshots` when set (both fluid and cad_v5 accept it). Resume keys use the labelled arm name.

- [ ] **Step 1: Failing tests**

```python
# append to tests/test_run_card.py
def test_subset_phase1_caps_each_suite(monkeypatch):
    import run_card as rc
    specs = {"cadprompt": ([{"id": f"cp-{i}", "spec": "x", "tier": 0} for i in range(100)], {}),
             "cad-arena": ([{"id": f"a-{i}", "spec": "y", "tier": 1} for i in range(12)], {})}
    cut = rc.apply_subset(specs, "phase1")
    assert len(cut["cadprompt"][0]) == 30 and [s["id"] for s in cut["cadprompt"][0]][:3] == ["cp-0", "cp-1", "cp-2"]
    assert len(cut["cad-arena"][0]) == 12
    assert rc.apply_subset(specs, "full") == specs


def test_variant_labels_arm_and_env(monkeypatch):
    import run_card as rc
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd; seen["env"] = kw["env"]
        class P: stdout = '{"ok": true, "facts": {"solids": 1}, "gate_hard": [], "gate_spec": [], "usage": {"completion_tokens": 5}, "build_dir": ""}'; stderr = ""; returncode = 0
        return P()
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    row = rc.run_row("gemma-4-31b", "cad-arena", {"id": "a-1", "spec": "A cube", "tier": 1}, {"solids": 1}, "oneshot", 60,
                     knobs=rc.Knobs(variant="bo3", candidates=3, no_fewshots=True, critic="local:minicpm-v"))
    assert row["arm"] == "gemma-4-31b+bo3"
    assert seen["env"]["CAD_CANDIDATES"] == "3" and seen["env"]["CAD_CRITIC_MODEL"] == "local:minicpm-v"
    assert "--no-fewshots" in seen["cmd"]
```

- [ ] **Step 2: Run, expect failure** (`AttributeError: apply_subset` / `Knobs`).

- [ ] **Step 3: Implement**

```python
# scripts/run_card.py additions
from dataclasses import dataclass

SUBSETS = {
    "full": None,
    "phase1": {"cadprompt": 30, "text2cadquery": 30, "heldout-cqe": 25, "text-to-cad": 10,
               "organic": 5, "hard-eval": 15, "cad-arena": 12},
}


@dataclass(frozen=True)
class Knobs:
    variant: str = ""
    candidates: int = 0
    no_fewshots: bool = False
    critic: str = ""

    def env(self) -> dict:
        e = {}
        if self.candidates > 0: e["CAD_CANDIDATES"] = str(self.candidates)
        if self.critic: e["CAD_CRITIC_MODEL"] = self.critic
        return e

    def argv(self) -> list[str]:
        return ["--no-fewshots"] if self.no_fewshots else []


def apply_subset(suites: dict, name: str) -> dict:
    caps = SUBSETS[name]
    if caps is None:
        return suites
    return {s: (specs[: caps.get(s, len(specs))], acc) for s, (specs, acc) in suites.items()}


def labelled(name: str, knobs: "Knobs") -> str:
    return f"{name}+{knobs.variant}" if knobs.variant else name
```

`build_once(spec, mode, timeout, knobs=Knobs())`: `env = {**os.environ, "CAD_BENCH": "1", "CAD_KEEP_MAKER": "1", **knobs.env()}`; append `knobs.argv()` to `cmd` for both modes. `run_row(..., knobs=Knobs())` sets `"arm": labelled(name, knobs)`. In `main()`: parse the five flags, build `knobs`, `suites = apply_subset(suites, ns.subset)` after loading and before the contamination check, `meta.update(variant=knobs.variant, knobs=dataclasses.asdict(knobs), subset=ns.subset)`, and use `labelled(name, knobs)` for the `done` lookups and the row label. The `--rescore` path is unchanged.

- [ ] **Step 4: Tests green, full suite, commit** `run_card: variants, knobs pass-through, stratified phase1 subset`.

---

### Task 2: best-of-N in one-shot mode

**Files:**
- Modify: `scripts/fluid_gen.py:~203-205` (the `generate_code_raw` call and `build_dir` creation)
- Test: `tests/test_fluid_bon.py`

**Interfaces:**
- Consumes: `engine._pick_first_turn_candidate(brief: dict, spec: str, first_code: str, work_dir: Path, n: int) -> str`, `first_turn_candidates()` from `cad_v5.config`, `engine._HELPER_RESULT_RE`.
- Produces: fluid `build` runs `n = first_turn_candidates()` candidates when `n > 1` and the code is not a helper short-circuit; result JSON gains `"candidates": n`.

- [ ] **Step 1: Failing test** (`tests/test_fluid_bon.py`): import `fluid_gen` by path, monkeypatch `engine.generate_code_raw` to return `"code0"`, `engine._pick_first_turn_candidate` to record its args and return `"best"`, `fluid_gen.first_turn_candidates` to return 3, `engine.triage_ambiguity` to return None, `engine.retrieval_notes_for` to return [], `engine.spec_helper` to return None, and `_materialize_with_salvage` to return a minimal `m` dict; run `cmd_build` with a fake args object (`spec="A cube 10 mm"`, `coder="strong"`, `image=None`, `no_fewshots=True`, `json=True`); assert the recorded `n == 3`, `first_code == "code0"`, the work_dir exists under `BUILDS_DIR`, and the printed result has `"candidates": 3`.

- [ ] **Step 2: Implement**

Move `build_dir` creation ABOVE the codegen block, then replace the `generate_code_raw` line with:

```python
        code = engine.generate_code_raw(spec, notes)
        n_cand = first_turn_candidates()
        if n_cand > 1 and not engine._HELPER_RESULT_RE.search(code):
            code = engine._pick_first_turn_candidate(
                {"helper": None, "notes": notes, "expected": {}, "_raw": True}, spec, code, build_dir, n_cand)
```

(`_raw: True` makes `generate_code` route the extra candidates through `generate_code_raw`, matching the first one; verify that branch in `cad_engine.generate_code` before relying on it, and if it keys on a different flag use that.) Import `first_turn_candidates` from `cad_v5.config`. Add `"candidates": n_cand if not helper else 1` to `_result`'s `extra` for `build`.

- [ ] **Step 3: Tests green, full suite, commit** `fluid: best-of-N first turn (engine candidate picker), result carries candidates`.

---

### Task 3: critic URL and Ollama-free preflight

**Files:**
- Modify: `cad_v5/config.py` (after the `LOCAL_CODER_*` block), `cad_engine.py` (`_ollama` local branch ~:579, `_installed_ollama_models` ~:811, `preflight` ~:837)
- Test: `tests/test_maker_config.py`

**Interfaces:**
- Produces: `CRITIC_URL = os.environ.get("CAD_CRITIC_URL", LOCAL_CODER_URL)`, `CRITIC_HEALTH` derived by replacing the path; in `_ollama()`'s local branch the POST goes to `CRITIC_URL` when `model == CRITIC_MODEL` (critic calls are the only ones with `images` today, but key on the model, not on images), else `LOCAL_CODER_URL`; `_installed_ollama_models()` returns `set()` and logs a warning instead of raising when Ollama is unreachable AND every model in `(BRIEF_MODEL, _code_model(), CRITIC_MODEL)` starts with `local:` or `cloud/`; otherwise unchanged. When `CRITIC_MODEL` is `local:` and `CRITIC_URL != LOCAL_CODER_URL`, `preflight()` also probes `CRITIC_HEALTH` once and raises a clear error if it does not answer.

- [ ] **Step 1: Failing tests**

```python
def test_critic_url_defaults_to_coder_url(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD_CRITIC_URL", raising=False)
    cfg = _reload_with(tmp_path, {})
    assert cfg.CRITIC_URL == cfg.LOCAL_CODER_URL


def test_critic_call_goes_to_critic_url(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_CRITIC_URL", "http://127.0.0.1:8089/v1/chat/completions")
    monkeypatch.setenv("CAD_CRITIC_MODEL", "local:minicpm-v")
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    urls = []
    class R:
        def __init__(s, url): urls.append(url)
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s): return b'{"choices":[{"message":{"content":"ok"}}]}'
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", lambda req, timeout=0: R(req.full_url))
    monkeypatch.setattr(cad_engine, "_ensure_default_server", lambda *a, **k: None)
    monkeypatch.setattr(cad_engine, "_unload_ollama_guests", lambda *a, **k: None)
    cad_engine._ollama("local:minicpm-v", "sys", "user", images=["QUJD"])
    cad_engine._ollama("local:gemma-4-31b", "sys", "user")
    assert urls == ["http://127.0.0.1:8089/v1/chat/completions", cfg.LOCAL_CODER_URL]


def test_preflight_tolerates_ollama_down_when_all_local(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_CRITIC_MODEL", "local:gemma-4-31b")
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    def boom(*a, **k): raise OSError("connection refused")
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", boom)
    monkeypatch.setattr(cad_engine, "_code_model", lambda: "local:gemma-4-31b")
    assert cad_engine._installed_ollama_models() == set()
```

- [ ] **Step 2: Implement** per the Interfaces block (the `_ollama` request builder already constructs a `urllib.request.Request`; select the URL variable there; the test patches `urlopen` and inspects `req.full_url`).

- [ ] **Step 3: Tests green, full suite, commit** `engine: critic URL seam; preflight no longer needs Ollama when every model is local`.

---

### Task 4: optional critic server and `arms.py critic`

**Files:**
- Create: `deploy/critic-server` (copy of `deploy/maker-server` reading `~/.openclaw/critic.env`, default `PORT=8089`? NO: 8089 is the embed server. Use `PORT=8090`? NO: 8090 is cad-web. Use `PORT=8092`), `deploy/critic-server.service` (no `Conflicts=`, `RuntimeMaxSec=12h`, `Restart=no`)
- Modify: `benchmarks/arms.json` (add `"critics": [{"name": "minicpm-v-4.6", "alias": "minicpm-v", "gguf": "MiniCPM-V-4.6-gguf/<file>", "mmproj": "MiniCPM-V-4.6-gguf/<mmproj>", "ctx": 8192, "port": 8092}, {"name": "gemma-4-12b", "alias": "gemma-4-12b", "gguf": "gemma-4-12B-it-GGUF/<file>", "mmproj": "gemma-4-12B-it-GGUF/<mmproj>", "ctx": 8192, "port": 8092}]`, exact filenames from `ls /mnt/nvme-apps/LinuxModels/MiniCPM-V-4.6-gguf/ gemma-4-12B-it-GGUF/`), `scripts/arms.py` (`critic use <name>` writes `~/.openclaw/critic.env` and starts the unit, `critic off` stops it; `load_critics()`), `deploy/README.md`
- Test: `tests/test_arms.py`

**Interfaces:**
- Produces: `critic-server.service` on :8092 coexisting with the maker; `arms.py critic use minicpm-v-4.6` prints the two env exports the card must use (`CAD_CRITIC_MODEL=local:minicpm-v CAD_CRITIC_URL=http://127.0.0.1:8092/v1/chat/completions`); `run_card --critic local:minicpm-v --critic-url http://127.0.0.1:8092/v1/chat/completions` (add `--critic-url` to Task 1's knobs: env `CAD_CRITIC_URL`).
- VRAM rule: `arms.py critic use` refuses (exit 2, message) if `nvidia-smi` reports less than the critic's `vram_gb` field free (add `vram_gb` to each critic entry: MiniCPM-V 2.5, Gemma-4-12B 9.2).

- [ ] Tests: `render_critic_env` keys; `load_critics` shape; the refusal path with a patched `free_vram_gb()`.
- [ ] Live check (GPU): `arms.py use gemma-4-31b`, `nvidia-smi` used MiB recorded into the report (the Gemma 16k figure Phase 0 never logged), then `arms.py critic use minicpm-v-4.6` (expect success only if the free VRAM allows), `curl :8092/health`, a one-image critic call through `cad_engine._ollama` with `CAD_CRITIC_URL` set, then `critic off`, `arms.py restore`.
- [ ] Commit `deploy: optional critic-server (:8092) + arms.py critic use|off with a VRAM check`.

---

### Task 5: lift table

**Files:**
- Create: `scripts/lift_report.py`
- Test: `tests/test_lift_report.py`

**Interfaces:**
- Produces: `lift_table(rows: list[dict], baseline: str) -> dict` keyed by labelled arm, each `{"n", "invalid_ratio", "gate_clean_rate", "acceptance", "match_rate", "median_wall_s", "tokens_per_build", "delta": {metric: value - baseline_value}}` computed over the PUBLIC suites + heldout (cadprompt, text2cadquery, heldout-cqe) only, excluding helper rows; `render_lift_md(table, baseline) -> str` with one row per variant, deltas as signed points, baseline first.

- [ ] Tests: two variants of one arm, hand-computed deltas; helper rows excluded; a variant with zero public rows renders "-".
- [ ] Commit `lift_report: variant deltas vs baseline on the public suites`.

---

### Task 6: the A/B runs (execution)

All on `benchmarks/results/card/phase1/` with `--subset phase1` (about 127 builds per variant; Gemma one-shot about 1.2 h, thinking-on about 6 h, agent mode 3 to 4 h). Launch each from a script file with `setsid nohup`, one at a time, resident restored between them; the owner's chat is down while they run.

- [ ] **6.1 baseline** `run_card.py --arms gemma-4-31b --subset phase1 --mode oneshot --out benchmarks/results/card/phase1 --allow-contaminated` (variant `""`).
- [ ] **6.2 best-of-3** `--variant bo3 --candidates 3` (Task 2).
- [ ] **6.3 no retrieval** `--variant nofs --no-fewshots`.
- [ ] **6.4 thinking on** an arms.json variant entry `gemma-4-31b-think` (same GGUF, `extra_args` without `enable_thinking:false`), `--arms gemma-4-31b-think --subset phase1` BUT with a tighter subset: add `"phase1think": {"cadprompt": 15, "text2cadquery": 15, "heldout-cqe": 10, "text-to-cad": 5, "organic": 3, "hard-eval": 5, "cad-arena": 6}` to `SUBSETS` and run the baseline again on that subset (`--variant think-base`) so the comparison is like for like.
- [ ] **6.5 agent mode, self-critic** `--mode agent --variant agent-self --critic local:gemma-4-31b` (the coder judges its own two-panel render; no second server).
- [ ] **6.6 agent mode, small critic** `--mode agent --variant agent-minicpm --critic local:minicpm-v --critic-url http://127.0.0.1:8092/v1/chat/completions` with `arms.py critic use minicpm-v-4.6` first (skip and record if the VRAM check refuses).
- [ ] **6.7 agent mode, Ollama gemma4:e4b** `--mode agent --variant agent-gemma4 --critic gemma4:e4b` (the per-turn eviction path; expect it to be slow; it is the pre-Phase-0 default and must be measured once).
- [ ] After each run: `python3 -c` over `lift_report.lift_table` and append to `LIFT.md`; the Lab view (Task 7) reads the same file.

---

### Task 7: Lab view lift table and the lock-in

**Files:**
- Modify: `webui/app.py` (`GET /api/lab/lift` reading `benchmarks/results/card/latest/LIFT.json`), `webui/static/index.html` (table under the card), `~/.openclaw/cad.json` (`candidates`, and `critic` if a critic won), `benchmarks/arms.json` (gemma `extra_args` if thinking won), `cad_v5/config.py` `CRITIC_MODEL` default (if a `local:` critic won), `benchmarks/results/card/phase1/DECISION.md`

- [ ] Write `DECISION.md`: the lift table, the rule (a lever is locked in only if it improves invalid ratio or match rate on the public suites by 3 points or more without a wall-time cost above 2x; ties go to the cheaper setting), and the chosen defaults.
- [ ] Apply the defaults; run `python3 scripts/run_card.py --arms gemma-4-31b --subset phase1 --variant locked` as the confirmation run; it must reproduce the winning variant's numbers within 2 points.
- [ ] Commit `phase1: lift table in the Lab view; defaults locked from the A/Bs`. Exit gate met when `LIFT.md`, `DECISION.md` and the confirmation run exist.

---

## Self-review

- Spec Phase 1 row: thinking on (6.4), best-of-N (6.2), critic within the VRAM budget (Tasks 3, 4, 6.5 to 6.7), retrieval on/off (6.3), best config locked as default with a lift table (Task 7). Fluid auto-escalation to a second arm is deliberately out (it needs a server swap per failing spec; recorded as a Phase 3 teacher-arm concern instead).
- Interfaces: `Knobs` (Task 1) is consumed by Task 6's commands; `--critic-url` is introduced in Task 4 and must be added to `Knobs` there; `lift_table` (Task 5) is consumed by Tasks 6 and 7; `CRITIC_URL` (Task 3) is consumed by Task 4's env exports.
- Ports: 8085 proxy, 8086 resident, 8088 maker, 8089 embed, 8090 cad-web, 8091 bridge, 8092 critic (new; confirm free with `ss -ltn` in Task 4).

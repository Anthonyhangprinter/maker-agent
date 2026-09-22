#!/usr/bin/env python3
"""lang_ab.py -- A/B harness: which CAD language does the local strong-rung model write best.

Four arms, same spec pool, same model (whichever arm the GPU window has already loaded),
one model call per (arm, spec):

  b123d        the production path -- engine.generate_code_raw's exact prompt shape
               (system prompt + retrieval few-shots ON), executed via engine.run_step.
  b123d-nofs   identical, retrieval few-shots OFF (the like-for-like leg vs cadquery/openscad,
               neither of which gets few-shots either).
  cadquery     a hand-written system prompt (CADQUERY_SYSTEM below), no few-shots, executed
               in an ISOLATED venv (.venv-cq, see setup_cq_env.sh) via run_cq.py.
  openscad     scripts/openscad_gen.py's own system prompt, one shot, no repair turn,
               compiled with the local OpenSCAD AppImage.

Every arm is scored the same way: geom_bands.score_against_reference (Chamfer/Hausdorff/
volume/bbox, GIFT-style match/valid/near_miss/fail bands) against the SAME public-suite
reference geometry (benchmarks/cadprompt, benchmarks/text2cadquery -- the only two suites in
this repo that carry reference STLs). See README.md for the unfairness note: the b123d arm
carries tuned few-shots and code patches, the other three do not, so this is not a fair
fight between languages -- it measures whether the SAME model, unassisted, does much worse
in a language it has had no CAD-specific engineering poured into.

SAFETY / SCOPE. This script does not switch models or evict services itself: main() calls
lab._armwindow.arm_window(), the SAME reviewed context manager lab/harvest.py and
lab/specgen.py use, so the maker-arm switch and resident restore are one implementation,
not a fourth copy. It refuses to run at all outside a GPU window
(lab.ship.require_gpu_window, same gate as every other lab entry point). Launch:

    PYTHONUTF8=1 lab/gpu_window.sh python3 benchmarks/lang-ab/lang_ab.py \\
        --arms b123d,b123d-nofs,cadquery,openscad --specs 40 --run-id <name>

Every model call goes through ONE function, `call_model(system, user, temperature)`
(module-level, replaceable): the default implementation rides engine._ollama against
whichever local: model is currently configured (cad_config.CODE_MODEL_STRONG), thinking
off. Tests replace this function wholesale with a fake that returns canned code, so no
test in tests/test_lang_ab.py ever reaches a real model endpoint, and `run_harness()`
(the actual work, below `main()`) never touches arm_window/require_gpu_window at all --
only main() does, so a test can call run_harness() directly and never risk a systemctl
or scripts/arms.py call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

LANG_AB_DIR = Path(__file__).resolve().parent
HERE = LANG_AB_DIR.parents[1]          # the cad-builder(-phase3) repo root
BENCH = HERE / "benchmarks"
RESULTS_DIR = LANG_AB_DIR / "results"
CQ_VENV_PY = LANG_AB_DIR / ".venv-cq" / "bin" / "python"
RUN_CQ_PY = LANG_AB_DIR / "run_cq.py"

for _p in (str(HERE), str(HERE / "scripts"), str(LANG_AB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every python process this harness itself spawns is told PYTHONUTF8=1 explicitly (see
# _run_cq_subprocess/_run below); this setdefault only covers THIS interpreter, in case the
# controller forgot to set it on the outer `python3 benchmarks/lang-ab/lang_ab.py` launch.
os.environ.setdefault("PYTHONUTF8", "1")
# Before cad_engine (or anything that might build) is even imported: this harness builds
# and scores in its own process exactly like a benchmark card run, and a good build must
# never self-promote into ~/.openclaw/cad-examples.jsonl / cad-sftpairs.jsonl (same guard
# lab/harvest.py and scripts/run_card.py use).
os.environ.setdefault("CAD_BENCH", "1")

import cad_engine as engine                # noqa: E402
from cad_v5 import config as cad_config    # noqa: E402
import geom_bands                          # noqa: E402
import openscad_gen                        # noqa: E402  (scripts/openscad_gen.py, read-only reuse)
from lab import ship                       # noqa: E402
from lab._armwindow import arm_window      # noqa: E402
import report                              # noqa: E402  (benchmarks/lang-ab/report.py)

ARMS = ("b123d", "b123d-nofs", "cadquery", "openscad")
STEP_ARMS = ("b123d", "b123d-nofs", "cadquery")   # arms that produce a STEP (facts via run_inspect)
DEFAULT_SPECS = 40
DEFAULT_SEED = 20260919
CQ_BUILD_TIMEOUT = 240
CODEGEN_TEMPERATURE = 0.2

GPU_WINDOW_HINT = (
    "lang_ab.py runs one model call per (arm, spec) against the currently loaded maker/"
    "resident model: it must run inside a GPU window: `PYTHONUTF8=1 lab/gpu_window.sh "
    "python3 benchmarks/lang-ab/lang_ab.py ...` (that holds ~/.openclaw/cad-build.lock, "
    f"evicts the resident and the maker arm first, and exports {ship.GPU_WINDOW_ENV}=1). "
    "Pass --i-know-the-gpu-is-free only when the GPU is already free by hand."
)

# --model resident (2026-09-19): the owner also wants this same experiment run against the
# RESIDENT model (Qwen3.8-27b), which is up all day behind the :8085 gpu-proxy independent of
# any maker arm. This leg is a plain chat client, not a benchmark-card runner: it never calls
# arm_window/require_gpu_window, never starts or stops qwen38-server/maker-server, and posts
# straight to RESIDENT_URL. See ResidentCaller below and main()'s --model branch. Every arm's
# codegen still goes through the ONE call_model_fn seam described above -- only WHICH callable
# main() hands to run_harness changes.

# ---------------------------------------------------------------------------
# The CadQuery system prompt (no few-shots; concise; same contract as build123d's).
# ---------------------------------------------------------------------------

CADQUERY_SYSTEM = """\
You are a CadQuery expert. Write a complete, runnable CadQuery script for the requested part.

RULES:
1. First line: import cadquery as cq
2. All dimensions in MILLIMETRES. Centre the part at the origin unless the request says
   otherwise.
3. Define every key dimension as a named constant (a plain float assignment, mm) near the
   top of the script, before it is used in geometry.
4. Build the part with the CadQuery Workplane/Sketch API and assign the FINAL object to a
   variable named `result`.
5. Unless the request explicitly describes more than one component (an assembly), `result`
   must be a SINGLE fused solid: union every feature into one Workplane chain (or one
   cq.Shape) -- never leave loose, unfused bodies.
6. For a genuine multi-part request, build each part as its own solid and combine them into
   a cq.Assembly (one .add() call per part, each with a distinct name), and assign that
   Assembly to `result`.
7. Do NOT call cq.exporters.export(...) or write any file -- the runner exports `result`.
8. Do NOT call show_object(...) -- assigning `result` is enough.
9. Reply with ONLY the Python code -- no prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Code-block extraction (one implementation, used by every arm).
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_+-]*[ \t]*\n?")


def extract_code(raw: Optional[str]) -> str:
    """Strip a leading/trailing markdown code fence from a model reply, tolerant of any
    language tag (```python, ```cadquery, ```openscad, ```scad, or a bare ```).

    Used uniformly by all four arms so a fence-extraction bug is fixed in one place rather
    than in four near-duplicate copies (engine._strip_fences only recognises ```python; a
    cadquery or openscad reply may come back tagged differently)."""
    if not raw:
        return ""
    text = raw.strip()
    text = _FENCE_OPEN_RE.sub("", text, count=1)
    text = text.rstrip("`").strip()
    return text


# ---------------------------------------------------------------------------
# Error classification.
# ---------------------------------------------------------------------------

ERROR_CLASSES = ("none", "syntax", "import_or_name", "api_misuse", "kernel", "timeout", "no_code", "other")

_SYNTAX_RE = re.compile(r"\b(SyntaxError|IndentationError|TabError)\b")
_IMPORT_NAME_RE = re.compile(r"\b(ImportError|ModuleNotFoundError|NameError)\b")
# OCP/OpenCascade kernel-level failures: distinguished from a plain Python API-misuse
# exception by carrying an OCP/TopoDS/BRep/Standard_ signature (a null/degenerate shape from
# a boolean op, an OCP-raised Standard_Failure, etc.) -- these indicate the KERNEL rejected
# the geometry, not that the model called the wrong Python method.
_KERNEL_RE = re.compile(
    r"(OCP\.[A-Za-z_.]*|Standard_Failure|Standard_\w+|BRep_API|BRepAlgoAPI|TopoDS_\w+|"
    r"StdFail_NotDone|[Nn]ull\s*[Ss]hape)"
)
_API_MISUSE_RE = re.compile(r"\b(AttributeError|TypeError|ValueError)\b")


def classify_error(text: Optional[str], timed_out: bool = False, no_code: bool = False) -> str:
    """Bucket a failure into one of ERROR_CLASSES from its traceback/stderr text.

    Order matters: no_code and timeout are decided by the caller's own flags (they may have
    no traceback text at all), then syntax/import_or_name (Python-level, unambiguous
    exception names) are checked before the OCP/kernel signature, and api_misuse
    (AttributeError/TypeError/ValueError with NO kernel signature) is checked last of the
    pattern-based buckets -- a plain AttributeError from calling a nonexistent CadQuery/
    build123d method is api_misuse, but the SAME exception type raised from inside OCP's own
    wrapper carries one of the kernel strings and is classified kernel instead."""
    if no_code:
        return "no_code"
    if timed_out:
        return "timeout"
    if not text:
        return "other"
    if _SYNTAX_RE.search(text):
        return "syntax"
    if _IMPORT_NAME_RE.search(text):
        return "import_or_name"
    if _KERNEL_RE.search(text):
        return "kernel"
    if _API_MISUSE_RE.search(text):
        return "api_misuse"
    return "other"


def _tail(text: Optional[str], n: int = 300) -> str:
    return (text or "")[-n:]


# ---------------------------------------------------------------------------
# Model call seam.
# ---------------------------------------------------------------------------

def call_model(system: str, user: str, temperature: float = CODEGEN_TEMPERATURE) -> str:
    """The ONE seam every arm's codegen call goes through.

    Default implementation rides engine._ollama against whichever local: model is currently
    configured (cad_config.CODE_MODEL_STRONG -- the maker arm on :8088 when cad.json's
    maker.enabled, else the resident on :8086), thinking off (no_think=True: a codegen call
    is not a judgement call). max_tokens is left at the server default, matching what a
    plain (non-think) engine.generate_code_raw call does.

    tests/test_lang_ab.py never calls this: every test passes its own `call_model_fn`
    (a fake returning canned code) into run_harness()/_codegen_for_arm(), so replacing this
    function is as simple as passing a different callable -- no monkeypatching needed."""
    return engine._ollama(
        cad_config.CODE_MODEL_STRONG, system, user,
        timeout=cad_config.CODE_TIMEOUT_STRONG, temperature=temperature, no_think=True,
    )


CallModelFn = Callable[[str, str, float], str]


# ---------------------------------------------------------------------------
# --model resident: a plain chat client against the always-on resident, no service control.
# ---------------------------------------------------------------------------

RESIDENT_URL = cad_config.RESIDENT_PROXY_URL           # :8085 gpu-proxy, OpenAI chat completions
RESIDENT_MODEL_ALIAS = cad_config.RESIDENT_ALIAS       # "qwen3.8-27b"
# The resident's OWN port, always 8086 regardless of cad.json's maker block -- unlike
# cad_config.LOCAL_CODER_HEALTH, which follows maker.enabled and would point at :8088 while a
# maker arm is loaded. This leg cares whether the RESIDENT answers, not whichever server the
# maker/default rung currently targets.
RESIDENT_HEALTH_URL = "http://127.0.0.1:8086/health"
RESIDENT_TIMEOUT = 600
RESIDENT_MAX_CONSECUTIVE_FAILURES = 5


class ResidentCallAbort(RuntimeError):
    """Raised by ResidentCaller once RESIDENT_MAX_CONSECUTIVE_FAILURES calls in a row have
    failed. A distinct type on purpose: run_one's broad `except Exception` around a single
    codegen call (which classifies ONE failed attempt as that row's error and lets the harness
    move on to the next spec) re-raises this one instead of swallowing it -- a resident that
    keeps failing is very likely down or held by another job, and grinding through the rest of
    the spec list one timeout at a time would just burn wall time for no data."""


class ResidentCaller:
    """A call_model_fn implementation for --model resident: posts directly to RESIDENT_URL
    (the :8085 gpu-proxy in front of the always-on qwen38-server), thinking off, no arm
    switch, no systemctl call, no lab._armwindow involvement at all -- the resident is simply
    used like any other OpenAI-compatible chat client for the duration of the run.

    Returns the raw reply text, same as call_model()'s default (maker) implementation, and
    sets cad_engine._LAST_USAGE from the response's `usage` object so run_one's existing
    tokens_in/tokens_out bookkeeping (which reads that global right after the call) works
    identically for both legs -- one row shape, whichever model answered it.

    Tracks consecutive HTTP failures across calls (state lives on the instance, one instance
    per run) so a resident that goes down mid-run raises ResidentCallAbort on the
    RESIDENT_MAX_CONSECUTIVE_FAILURES'th failure in a row instead of silently degrading into
    an all-errors run. A single failure (HTTP error, connection refused, malformed JSON) is a
    plain RuntimeError -- run_one records it as that attempt's error and moves on, per the
    task's continue-on-error contract."""

    def __init__(self) -> None:
        self.consecutive_failures = 0

    def __call__(self, system: str, user: str, temperature: float = CODEGEN_TEMPERATURE,
                 max_tokens: Optional[int] = None) -> str:
        body = {
            "model": RESIDENT_MODEL_ALIAS,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            RESIDENT_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=RESIDENT_TIMEOUT) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            self.consecutive_failures += 1
            if self.consecutive_failures >= RESIDENT_MAX_CONSECUTIVE_FAILURES:
                raise ResidentCallAbort(
                    f"{RESIDENT_MAX_CONSECUTIVE_FAILURES} consecutive resident-call failures "
                    f"against {RESIDENT_URL} -- the resident may be down or taken by another "
                    f"job. Last error: {e}"
                ) from e
            raise RuntimeError(f"resident call failed ({e})") from e
        self.consecutive_failures = 0
        usage = resp.get("usage")
        engine._LAST_USAGE = usage if isinstance(usage, dict) else None
        choices = resp.get("choices") or [{}]
        return (choices[0].get("message", {}).get("content") or "").strip()


def _resident_health_ok(timeout: int = 5) -> bool:
    """GET RESIDENT_HEALTH_URL and return True on any reply that parses as JSON. Checked
    once, before the first call, so an already-down resident fails fast with a clear message
    instead of burning RESIDENT_MAX_CONSECUTIVE_FAILURES timeouts to discover the same thing."""
    try:
        with urllib.request.urlopen(RESIDENT_HEALTH_URL, timeout=timeout) as r:
            json.loads(r.read().decode("utf-8"))
        return True
    except Exception:
        return False


def _neutralize_service_helpers_for_resident() -> None:
    """Belt-and-suspenders for --model resident. Reading cad_retrieval.py and
    inject_retrieval_notes() confirms the b123d/b123d-nofs arms' retrieval path (the only
    engine-import-dependent codepath either arm's codegen touches) calls only the CPU
    embed-server on :8089 -- nothing on it reaches _ensure_default_server or a model call. This
    still patches the service-eviction helper(s) to no-ops under --model resident so a future
    change to that path can never make this leg start or stop qwen38-server/maker-server; the
    resident leg is a plain chat client, full stop."""
    engine._ensure_default_server = lambda *a, **k: None
    if hasattr(engine, "_pause_default_server_for"):
        engine._pause_default_server_for = lambda *a, **k: None


# ---------------------------------------------------------------------------
# Suite loading + deterministic tier-stratified spec selection.
# ---------------------------------------------------------------------------

PUBLIC_SUITES = ("cadprompt", "text2cadquery")


def load_suite(suite_root: Path, name: str) -> tuple[list[dict], dict]:
    d = suite_root / name
    specs_path, acc_path = d / "specs.json", d / "acceptance.json"
    if not specs_path.exists():
        return [], {}
    data = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = data["benchmarks"] if isinstance(data, dict) and "benchmarks" in data else data
    acc = json.loads(acc_path.read_text(encoding="utf-8")) if acc_path.exists() else {}
    return specs, acc


def select_specs(suites: dict[str, list[dict]], n: int, seed: int) -> list[dict]:
    """`n` specs drawn deterministically (given `seed`) and stratified by (suite, tier)
    across ALL the given suites, so the SAME list of specs is used for every arm.

    Generalises scripts/run_card.py's `_stratified` (which stratifies by tier within one
    suite) across the whole pool: each (suite, tier) group is allotted ceil(n * group_size /
    total) specs, ceilings that sum past `n` are trimmed one at a time off whichever group
    currently holds the most (ties broken toward the group sorted first), and WHICH specs
    are taken from each group is a `random.Random(seed)` shuffle -- deterministic for a
    given seed, but not merely "the first N in file order" the way run_card's is (this
    harness's contract asks for "deterministically (seeded)", not "file order").

    Today both public suites carry only tier 0, so this reduces to an even, seeded split
    between the two suites; the (suite, tier) grouping is kept general so a suite gaining
    tiers later is still represented proportionally without a code change here."""
    pool = [dict(s, suite=suite) for suite, specs in suites.items() for s in specs]
    total = len(pool)
    if total == 0:
        return []
    if n >= total:
        return list(pool)  # every spec in every suite: nothing to select, file order kept
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(pool):
        groups.setdefault((s["suite"], s.get("tier", 0)), []).append(i)
    take = {g: min(len(idxs), -(-n * len(idxs) // total)) for g, idxs in groups.items()}
    while sum(take.values()) > n:
        g = max(sorted(take), key=lambda k: take[k])
        take[g] -= 1
        if take[g] == 0:
            del take[g]
    rng = random.Random(seed)
    chosen: set[int] = set()
    for g, count in take.items():
        idxs = groups[g][:]
        rng.shuffle(idxs)
        chosen.update(idxs[:count])
    # Stable order (file order across the concatenated pool) so rows.jsonl reads the same
    # way run after run; only WHICH specs are chosen is seed-dependent, not their order.
    return [pool[i] for i in range(total) if i in chosen]


# ---------------------------------------------------------------------------
# Per-arm codegen (all four routed through call_model / extract_code).
# ---------------------------------------------------------------------------

def _b123d_prompt(spec_text: str, notes: list) -> str:
    """Byte-for-byte the prompt shape cad_engine.generate_code_raw builds (its own
    docstring: "Codegen straight from the USER'S WORDS -- no brief, no paraphrase"),
    reproduced here rather than called directly so the model call itself can go through
    this harness's call_model seam instead of engine._ollama -- see call_model's docstring
    for why that matters for testability."""
    notes_str = "\n".join(f"- {n}" for n in (notes or []))
    return (
        f"USER REQUEST (verbatim, every number here is AUTHORITATIVE):\n{spec_text}\n\n"
        + (f"Notes:\n{notes_str}\n\n" if notes_str else "")
        + "Write the build123d code:"
    )


def _codegen_for_arm(arm: str, spec_text: str, call_model_fn: CallModelFn) -> tuple[str, str]:
    """Returns (code, raw_reply). Raises only on a genuine call_model_fn exception (network
    error, malformed fake, etc.) -- an EMPTY reply is not an exception, it is returned as
    empty `code` and classified `no_code` by the caller."""
    if arm in ("b123d", "b123d-nofs"):
        use_fewshots = arm == "b123d"
        notes = engine.retrieval_notes_for(spec_text, use_fewshots=use_fewshots)
        prompt = _b123d_prompt(spec_text, notes)
        raw = call_model_fn(engine._CODE_SYSTEM, prompt, CODEGEN_TEMPERATURE)
        code = engine._patch_code(extract_code(raw), wants=engine._wanted_edge_features(spec_text))
        return code, raw
    if arm == "cadquery":
        prompt = (f"USER REQUEST (verbatim, every number here is AUTHORITATIVE):\n{spec_text}\n\n"
                  "Write the CadQuery code:")
        raw = call_model_fn(CADQUERY_SYSTEM, prompt, CODEGEN_TEMPERATURE)
        return extract_code(raw), raw
    if arm == "openscad":
        prompt = f"USER REQUEST (every number is AUTHORITATIVE, mm):\n{spec_text}\n\nWrite the OpenSCAD file:"
        raw = call_model_fn(openscad_gen._SYSTEM, prompt, CODEGEN_TEMPERATURE)
        return extract_code(raw), raw
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# Per-arm execution.
# ---------------------------------------------------------------------------

def _run_cq_subprocess(code: str, work_dir: Path, timeout: int = CQ_BUILD_TIMEOUT):
    """Execute a CadQuery script via the isolated .venv-cq interpreter + run_cq.py.
    Returns (step_path_or_None, error_text_or_None, timed_out)."""
    if not CQ_VENV_PY.exists():
        return None, (f"cadquery venv not found at {CQ_VENV_PY}; run "
                       f"benchmarks/lang-ab/setup_cq_env.sh first"), False
    src = work_dir / "build_source.py"
    src.write_text(code, encoding="utf-8")
    step = work_dir / "build_output.step"
    try:
        step.unlink()
    except FileNotFoundError:
        pass
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        p = subprocess.run(
            [str(CQ_VENV_PY), str(RUN_CQ_PY), str(src), str(step)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as e:
        return None, str(e), True
    if p.returncode != 0 or not step.exists() or step.stat().st_size == 0:
        return None, (p.stderr or p.stdout or f"run_cq.py exited {p.returncode} with no output"), False
    return step, None, False


def _run_openscad(code: str, work_dir: Path):
    """Compile an OpenSCAD script, one shot, no repair (see module docstring).
    Returns (stl_path_or_None, error_text_or_None, timed_out)."""
    scad = work_dir / "build.scad"
    scad.write_text(code, encoding="utf-8")
    stl = work_dir / "build.stl"
    ok, err = openscad_gen.compile_scad(scad, stl)
    if not ok:
        timed_out = "timed out" in (err or "").lower()
        return None, (err or "compile failed"), timed_out
    return stl, None, False


def _run_b123d(code: str, work_dir: Path):
    """Execute build123d code via the production runner (engine.run_step).
    Returns (step_path_or_None, error_text_or_None, timed_out)."""
    try:
        step, _log = engine.run_step(code, work_dir)
    except subprocess.TimeoutExpired as e:
        return None, str(e), True
    except RuntimeError as e:
        return None, str(e), False
    return step, None, False


def _build_for_arm(arm: str, code: str, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    if arm in ("b123d", "b123d-nofs"):
        return _run_b123d(code, work_dir)
    if arm == "cadquery":
        return _run_cq_subprocess(code, work_dir)
    if arm == "openscad":
        return _run_openscad(code, work_dir)
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# One (arm, spec) attempt -> one row.
# ---------------------------------------------------------------------------

def run_one(run_id: str, arm: str, suite: str, spec: dict, crit: Optional[dict],
            suite_dir: Path, out_dir: Path, call_model_fn: CallModelFn,
            model_label: str = "gemma-4-31b") -> dict:
    spec_id, tier, spec_text = spec["id"], spec.get("tier", 0), spec["spec"]
    work_dir = out_dir / "builds" / arm / spec_id
    row = {
        "run_id": run_id, "arm": arm, "spec_id": spec_id, "suite": suite, "tier": tier,
        "model": model_label,
        "ok": False, "error_class": "none", "traceback_tail": "", "band": None,
        "chamfer_mm": None, "facts": {}, "tokens_in": None, "tokens_out": None,
        "seconds_codegen": None, "seconds_build": None, "code": "",
    }

    t0 = time.time()
    try:
        code, _raw = _codegen_for_arm(arm, spec_text, call_model_fn)
    except ResidentCallAbort:
        # A single call failure is this row's error and the run continues (below); five
        # in a row is an operational signal (the resident is down / taken by another job)
        # that must reach run_harness/main rather than be recorded as just another failed
        # spec, so it is re-raised past this row entirely -- see ResidentCallAbort's docstring.
        raise
    except Exception:
        row["seconds_codegen"] = round(time.time() - t0, 2)
        row["error_class"] = "other"
        row["traceback_tail"] = _tail(traceback.format_exc())
        return row
    row["seconds_codegen"] = round(time.time() - t0, 2)
    row["code"] = code
    usage = engine._LAST_USAGE if isinstance(engine._LAST_USAGE, dict) else {}
    row["tokens_in"] = usage.get("prompt_tokens")
    row["tokens_out"] = usage.get("completion_tokens")

    if not code.strip():
        row["error_class"] = "no_code"
        return row

    t1 = time.time()
    try:
        result_path, build_err, timed_out = _build_for_arm(arm, code, work_dir)
    except Exception:
        row["seconds_build"] = round(time.time() - t1, 2)
        row["error_class"] = classify_error(traceback.format_exc())
        row["traceback_tail"] = _tail(traceback.format_exc())
        return row
    row["seconds_build"] = round(time.time() - t1, 2)

    if build_err is not None or result_path is None:
        row["error_class"] = classify_error(build_err, timed_out=timed_out)
        row["traceback_tail"] = _tail(build_err)
        return row

    row["ok"] = True

    if arm in STEP_ARMS:
        try:
            insp = engine.run_inspect(result_path)
            if insp["valid"]:
                row["facts"] = engine.parse_facts(insp["output"])
        except Exception:
            pass  # facts are best-effort; a build that already succeeded still counts ok

    ref_name = (crit or {}).get("reference_stl")
    if ref_name:
        ref_path = suite_dir / ref_name
        try:
            score = geom_bands.score_against_reference(
                result_path, ref_path, normalize=bool((crit or {}).get("normalized")))
            row["band"] = score.get("band")
            row["chamfer_mm"] = score.get("chamfer_mm")
        except Exception as e:
            row["band"] = "fail"
            row["traceback_tail"] = _tail(f"band scoring error: {e}")

    return row


# ---------------------------------------------------------------------------
# Model label in the report. report.py is out of scope for this change (it does not know
# about the per-row "model" field), so this stamps the label into the files IT writes right
# after write_report() runs, instead of teaching report.py about the field itself.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_LABEL = "gemma-4-31b"


def _model_label_from_rows(rows_path: Path) -> str:
    """The model rows_path's rows were generated against, read from each row's own "model"
    field. An old rows.jsonl written before that field existed only ever came from the maker
    arm, "gemma-4-31b" for the whole time this harness has existed -- that is the fallback."""
    if not rows_path.exists():
        return DEFAULT_MODEL_LABEL
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line).get("model")
        if m:
            return m
    return DEFAULT_MODEL_LABEL


def _stamp_model_into_report(out_dir: Path, rows_path: Path) -> None:
    """Add a "Model: <label>" line to REPORT.md's header and a top-level "model" key to
    report.json, both already written by report.write_report(). Safe to call on the
    empty-rows report write_report itself already tolerates (no REPORT.md/report.json yet
    is a no-op here, not an error)."""
    model = _model_label_from_rows(rows_path)
    report_md = out_dir / "REPORT.md"
    if report_md.exists():
        lines = report_md.read_text(encoding="utf-8").splitlines()
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("Specs:"):
                insert_at = i + 1
                break
        else:
            for i, line in enumerate(lines):
                if line.startswith("# lang-ab report"):
                    insert_at = i + 1
                    break
        lines.insert(insert_at, f"Model: {model}")
        report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_json_path = out_dir / "report.json"
    if report_json_path.exists():
        data = json.loads(report_json_path.read_text(encoding="utf-8"))
        data["model"] = model
        report_json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The harness proper (no GPU-window / arm-switching here -- see module docstring).
# ---------------------------------------------------------------------------

def run_harness(args: argparse.Namespace, call_model_fn: Optional[CallModelFn] = None) -> Path:
    # --model resident (default "maker", byte-for-byte the old behaviour): getattr so a
    # hand-built argparse.Namespace from a test written before this flag existed (no `model`
    # attribute at all) still runs the maker path exactly as before.
    model_choice = getattr(args, "model", "maker")
    if model_choice not in ("maker", "resident"):
        raise SystemExit(f"lang_ab: --model must be 'maker' or 'resident' (got {model_choice!r})")
    if model_choice == "resident":
        _neutralize_service_helpers_for_resident()
        model_label = RESIDENT_MODEL_ALIAS
    else:
        model_label = cad_config.maker_config()["alias"]
    if call_model_fn is None:
        call_model_fn = ResidentCaller() if model_choice == "resident" else call_model

    suite_root = Path(args.suite_root) if args.suite_root else BENCH
    suites: dict[str, list[dict]] = {}
    accs: dict[str, dict] = {}
    for name in PUBLIC_SUITES:
        specs, acc = load_suite(suite_root, name)
        if specs:
            suites[name] = specs
            accs[name] = acc
        else:
            print(f"lang_ab: suite {name!r} has no specs.json under {suite_root} -- skipped",
                  file=sys.stderr)
    chosen = select_specs(suites, args.specs, args.seed)

    arms = list(ARMS) if args.arms in ("all", "") else [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"lang_ab: unknown arm {a!r} (choices: {', '.join(ARMS)})")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    done: set[tuple] = set()
    rows: list[dict] = []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done.add((r["arm"], r["suite"], r["spec_id"]))

    meta = {
        "run_id": run_id, "arms": arms, "n_specs": len(chosen), "seed": args.seed,
        "suite_root": str(suite_root), "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    try:
        for arm in arms:
            todo = [s for s in chosen if (arm, s["suite"], s["id"]) not in done]
            print(f"== arm {arm}: {len(todo)} builds (of {len(chosen)} total specs)")
            for spec in todo:
                suite = spec["suite"]
                crit = accs.get(suite, {}).get(spec["id"])
                suite_dir = suite_root / suite
                row = run_one(run_id, arm, suite, spec, crit, suite_dir, out_dir, call_model_fn,
                              model_label)
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {suite}/{spec['id']}: ok={row['ok']} error={row['error_class']} "
                      f"band={row['band']} codegen={row['seconds_codegen']}s build={row['seconds_build']}s")
    finally:
        try:
            report.write_report(out_dir)
            _stamp_model_into_report(out_dir, rows_path)
        except Exception as e:
            print(f"lang_ab: report generation failed ({e}); rows.jsonl is still complete", file=sys.stderr)

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS),
                     help=f"comma list of arms to run, or 'all' (default: all of {', '.join(ARMS)})")
    ap.add_argument("--specs", type=int, default=DEFAULT_SPECS,
                     help=f"number of specs drawn from the public suites (default: {DEFAULT_SPECS})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help=f"seed for the deterministic spec draw (default: {DEFAULT_SEED})")
    ap.add_argument("--suite-root", default="",
                     help="override the benchmarks/ root the two public suites are read from "
                          "(default: this repo's own benchmarks/; tests point this at a fixture dir)")
    ap.add_argument("--run-id", default="", help="reuse a run-id to resume (default: a UTC timestamp)")
    ap.add_argument("--arm", default=None,
                     help="maker arm for this run (default: cad.json's maker block) -- forwarded to "
                          "lab._armwindow.arm_window, never applied by hand")
    ap.add_argument("--i-know-the-gpu-is-free", action="store_true",
                     help="run outside a GPU window (only when the GPU was freed by hand)")
    ap.add_argument("--model", choices=("maker", "resident"), default="maker",
                     help="which model answers every call: 'maker' (default, unchanged "
                          "behaviour -- whichever local: model the GPU window/arm_window "
                          "brought up) or 'resident' (the always-on Qwen3.8-27b behind the "
                          "gpu-proxy at :8085, no GPU window, no arm switch, no service "
                          "control at all)")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.model == "resident":
        # The resident is up all day already: no GPU window, no arm switch, no systemctl
        # call anywhere in this branch. Just check it's actually answering before spending
        # the first spec on it.
        if not _resident_health_ok():
            print(f"lang_ab: resident not healthy at {RESIDENT_HEALTH_URL} -- aborting",
                  file=sys.stderr)
            return 2
        try:
            out_dir = run_harness(args)
        except ResidentCallAbort as e:
            print(f"lang_ab: {e}", file=sys.stderr)
            return 3
    else:
        # Before ANYTHING that touches a model or a service: no arm switch, no signal
        # handlers, no cad.json read, no model call. Same ordering rule as
        # lab/harvest.py's main().
        ship.require_gpu_window(args, GPU_WINDOW_HINT)
        with arm_window(args.arm):
            out_dir = run_harness(args)
    print(f"lang_ab: wrote {out_dir / 'rows.jsonl'} and {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared configuration: paths, models, loop constants, config/credential loaders, logging.

Extracted verbatim from cad_engine.py (v4.3) so behaviour is identical. The CAD builder root
(`_HERE`) resolves to the parent package dir so `scripts/`, `b123d/`, and `cad_retrieval` still
resolve exactly as before.
"""
import copy
import os
import sys
import json
import logging
from pathlib import Path

VERSION = "5.0"

# ── Paths ─────────────────────────────────────────────────────────────────────
# cad_v5/ lives inside the cad-builder skill dir; the skill root is its parent.
_PKG_DIR      = Path(__file__).resolve().parent
_HERE         = _PKG_DIR.parent                      # …/skills/cad-builder
_OPENCLAW     = Path.home() / ".openclaw"
LOG_FILE      = _OPENCLAW / "cad-agent.log"
FEEDBACK_FILE = _OPENCLAW / "cad-examples.jsonl"     # unified corpus: gold + rated + auto
SESSION_FILE  = _OPENCLAW / "cad-session.json"
CONFIG_FILE     = _OPENCLAW / "openclaw.json"
CAD_CONFIG_FILE = Path(os.environ.get("CAD_CONFIG_FILE", str(_OPENCLAW / "cad.json")))
                                           # agent settings (cad.*) — separate file because the
                                           # OpenClaw gateway's strict schema rejects unknown keys
                                           # (env override lets tests point at a temp file)
SCRIPTS_DIR   = _HERE / "scripts"
B123D_DIR     = _HERE / "b123d"
LAB_STATE     = _HERE / "lab" / "state"   # Phase 3 data engine: bank/ledger/pairs JSONL, git-ignored
                                           # except specs.jsonl + val_specs.json (see lab_config())

STEP_OUT      = _OPENCLAW / "cad-last-build.step"    # persisted so a failed export is recoverable
STL_OUT       = _OPENCLAW / "cad-last-build.stl"     # sliceable mesh, always exported alongside STEP
DXF_OUT       = _OPENCLAW / "cad-last-build.dxf"     # flat-pattern DXF (laser/sheet), best-effort

STL_VIEWER_CMD  = os.environ.get("CAD_STL_VIEWER", "fstl")        # local GUI STL viewer
CAD_VIEWER_PORT = int(os.environ.get("CAD_VIEWER_PORT", "4178"))  # browser CAD Viewer (text-to-cad)

# Make the skill root importable so `cad_retrieval`, `b123d.*`, and scripts resolve as before.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Semantic few-shot retrieval (graceful — any failure here must never break a build).
try:
    import cad_retrieval  # noqa: F401  (re-exported via learning.py)
except Exception:
    cad_retrieval = None

# ── Config / credentials (defined here, ABOVE the model constants, because
# maker_config() below needs it at import time) ────────────────────────────────
def load_config() -> dict:
    """openclaw.json (channels/env) overlaid with ~/.openclaw/cad.json as the `cad` block
    (kept separate — the OpenClaw gateway's strict schema rejects a top-level cad key)."""
    cfg: dict = {}
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.getLogger("cad_v5").warning(
            "[v5] openclaw.json unreadable (%s) — creds/telegram token unavailable.", e)
    try:
        with open(CAD_CONFIG_FILE) as f:
            cfg["cad"] = {**cfg.get("cad", {}), **json.load(f)}
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.getLogger("cad_v5").warning(
            "[v5] cad.json unreadable (%s) — cad.* settings ignored.", e)
    return cfg

# ── Models ────────────────────────────────────────────────────────────────────
# Strong rung: the "local:" prefix routes it through the OpenAI-schema branch in
# cad_engine._ollama(), against the resident (behind the gpu-proxy on :8085) or the
# maker server when cad.json maker.enabled. The health probe still targets the raw
# backend :8086 directly (LOCAL_CODER_HEALTH) — _ensure_default_server()/_wait_health()
# are the evictor and need backend truth fast (the :8085 gpu-proxy would happily hold
# the connection open for up to 40 min instead of failing fast while the resident is
# still starting). GPU-busy fix (2026-09-26, DESIGN-gpu-busy.md #8 "latent bypass"):
# the actual chat-completions calls (LOCAL_CODER_URL, every codegen/critique/utility
# request during a build) now ride the same :8085 agent door as every other agent
# caller (Morai/Hermes) when maker is disabled, instead of hitting :8086 directly —
# a build in flight while the resident is briefly down now waits it out gracefully
# through the proxy's hold-and-retry, rather than erroring with connection-refused.
# The maker-enabled case is untouched: maker-server is not behind the proxy at all.
# The always-there resident (qwen38-server.service on :8086, behind the gpu-proxy on :8085).
# Single-sourced here because the web UI's title worker needs the alias too, and the CAD
# rungs cannot name it once a maker arm is enabled.
RESIDENT_ALIAS = "qwen3.8-27b"
RESIDENT_PROXY_URL = "http://127.0.0.1:8085/v1/chat/completions"

def maker_config() -> dict:
    """The optional swappable CAD coder server ("maker" block in cad.json).

    Disabled (default): the strong rung is the resident, chat-completions calls ride
    the gpu-proxy agent door :8085, health checks the raw backend :8086 directly.
    Enabled: the strong rung is `maker-server` on `port`, serving `alias`, no proxy.

    A stale `alias` left behind by `scripts/arms.py restore` (which only flips `enabled`
    to false) must NOT keep routing strong-rung calls at a model the resident does not
    serve, so port, alias and unit are all read from the block only while enabled is true.

    `arm` is the entry NAME in benchmarks/arms.json (written by scripts/arms.py's
    apply_arm as `cad.json`'s maker.arm), which can differ from `alias` (the llama.cpp
    `--alias` the server actually answers as, e.g. the "qwen3.8-27b-nothink" arm serves
    alias "qwen3.8-27b") -- think_rung_available() below looks the card up by this name,
    not by alias. None when disabled (the resident is not a benchmarks/arms.json entry).
    """
    m = load_config().get("cad", {}).get("maker") or {}
    if not bool(m.get("enabled", False)):
        return {"enabled": False, "port": 8086, "alias": RESIDENT_ALIAS, "unit": "qwen38-server",
                "arm": None}
    alias = str(m.get("alias", RESIDENT_ALIAS))
    return {
        "enabled": True,
        "port": int(m.get("port", 8088)),
        "alias": alias,
        "unit": "maker-server",
        "arm": str(m.get("arm") or alias),
    }


def model_identity(code_model: str) -> dict:
    """Self-describing model identity for one build/turn — the web UI's per-creation and
    per-turn "model chip" (owner request, 2026-09-27: past builds used different arms, and
    a revise turn today rides whatever `cad.json`'s maker block currently names, which can
    differ from the arm that built the original). Reads maker_config() fresh (not the
    import-time `_MAKER` snapshot below) so this stays right if the maker arm was swapped
    mid-session (`scripts/arms.py use <arm>`).

    `code_model` is the resolved rung string a build/revise/rescale actually called
    (`cad_engine._code_model()`, e.g. "local:gemma-4-31b", "local:qwen3.8-27b+think",
    "cloud/claude-sonnet-5") — parsed directly rather than re-derived from maker_config()
    alone, so a manual override (CAD_CODE_MODEL_FAST) or the "+think" escalation rung still
    shows up correctly even if it does not match the currently configured maker alias.
    """
    mk = maker_config()
    label, maker_arm = code_model, None
    if code_model.startswith("cloud/"):
        label = f"{code_model[len('cloud/'):]} (cloud)"
    elif code_model.startswith("local:"):
        alias = code_model[len("local:"):]
        base_alias = alias[:-len(THINK_SUFFIX)] if alias.endswith(THINK_SUFFIX) else alias
        thinking = alias.endswith(THINK_SUFFIX)
        if mk["enabled"] and base_alias == mk["alias"]:
            maker_arm = mk["arm"]
            label = f"{maker_arm or base_alias} (maker)"
        elif base_alias == RESIDENT_ALIAS:
            label = f"{base_alias} (resident)"
        else:
            # An alias that matches neither the currently configured maker arm nor the
            # resident (e.g. cad.json's maker block moved on since this ran) — still show
            # the raw alias rather than guess a role for it.
            label = base_alias
        if thinking:
            label += " +think"
    return {"code_model": code_model, "maker_enabled": bool(mk["enabled"]),
            "maker_arm": maker_arm, "engine_version": VERSION, "label": label}


_MAKER = maker_config()
CODE_MODEL_STRONG  = "local:" + _MAKER["alias"]
LOCAL_CODER_PORT   = _MAKER["port"]
# Chat-completions target: the gpu-proxy agent door when the strong rung IS the
# resident (maker disabled — see the comment above), else the maker-server port
# directly (it has no proxy in front of it). Health always targets the raw port.
LOCAL_CODER_URL    = (RESIDENT_PROXY_URL if not _MAKER["enabled"]
                     else f"http://127.0.0.1:{LOCAL_CODER_PORT}/v1/chat/completions")
LOCAL_CODER_HEALTH = f"http://127.0.0.1:{LOCAL_CODER_PORT}/health"
# Utility-call model (brief/patch/lesson/questions/describe/refine — never the coder itself):
# was the Ollama qwen3:8b, deleted 2026-09-12 when Ollama started being retired. Riding
# CODE_MODEL_STRONG means these calls go through the local: OpenAI-schema branch of
# _ollama() against whichever strong-rung server is up (resident or maker arm) with
# thinking off (see _ollama()'s local: branch) — no separate Ollama model to keep alive.
BRIEF_MODEL        = CODE_MODEL_STRONG
# ── Fast rung: RETIRED 2026-09-19 ─────────────────────────────────────────────
# The fast rung was qwen2.5-coder:7b-instruct-q4_K_M on Ollama. Ollama is off the box
# (user rule: no fallback through it), and the rung had already lost its case on merit:
# Phase 0 measured it at 44% invalid on CADPrompt against Gemma-4-31B's 6%
# (benchmarks/results/card/phase0/card.md), which is why Phase 1 made fluid mode's
# default --coder "strong". CODE_MODEL_FAST keeps its NAME so scripts/fluid_gen.py,
# scripts/gift_sample.py and any pinned benchmark leg still import something real; it
# now resolves to the strong rung. The retired fast->strong escalation is gone; Task 1b
# (2026-09-19, below) adds the think rung back as the one escalation step that exists.
#
# CAD_CODE_MODEL_FAST survives as an A/B override, but ONLY for a local: alias — an
# Ollama tag would reach _ollama()'s hard error mid-build instead of failing here. A
# trailing "+think" (CODE_MODEL_THINK's suffix) passes this check unchanged: it is still
# a "local:..." string, and cad_engine._ollama() is what strips the suffix, not this check.
_FAST_OVERRIDE = os.environ.get("CAD_CODE_MODEL_FAST", "")
if _FAST_OVERRIDE and not _FAST_OVERRIDE.startswith(("local:", "cloud/")):
    raise RuntimeError(
        f"CAD_CODE_MODEL_FAST={_FAST_OVERRIDE!r} is not a local: model. Ollama was retired "
        "2026-09-19; the CAD coder rungs are llama.cpp servers. Use "
        "CAD_CODE_MODEL_FAST=local:<alias> (the alias your maker-server or the resident "
        "serves), or unset it to use the strong rung."
    )
CODE_MODEL_FAST    = _FAST_OVERRIDE or CODE_MODEL_STRONG
FAST_RUNG_RETIRED  = "fast rung retired 2026-09-19, using the strong rung"
# ── Think rung (Task 1b, 2026-09-19) ──────────────────────────────────────────
# Measured 2026-09-19: thinking is a PER-REQUEST switch on the same loaded maker arm. A
# request carrying chat_template_kwargs.enable_thinking=true produced 1,864 completion
# tokens (4,439 chars of reasoning) in 67s, against 127 tokens / 5s without it. No relaunch,
# no second server, no extra VRAM. So the second rung is a suffix on the SAME model string,
# not a new alias: cad_engine._ollama() strips "+think" before it ever reaches the server
# and turns enable_thinking on for that one call. This also means CAD_CODE_MODEL_FAST and
# any local: alias accept the suffix unchanged (the prefix check below never inspects it),
# and maker_config()/_ensure_default_server()/_maker_server_active() need no change at all:
# none of them resolve a server from the code-model string, only from cad.json's `maker`
# block, so "local:<alias>+think" is already, structurally, the same server as
# "local:<alias>". CRITIC_MODEL is a separate constant (never derived from CODE_MODEL_THINK
# or from cad_engine._code_model()), so the critic never inherits +think by construction.
THINK_SUFFIX       = "+think"
CODE_MODEL_THINK   = CODE_MODEL_STRONG + THINK_SUFFIX
# Reasoning needs headroom beyond a plain codegen call: a think call's max_tokens must cover
# the reasoning_content PLUS the code that follows it. _ollama() only sets this on a think
# call (no_think / plain calls are unaffected, same as before this rung existed).
CODE_MAX_TOKENS_THINK = int(os.environ.get("CAD_THINK_MAX_TOKENS", 12000))
# Card arms, used only to check thinking support. Env-overridable like CAD_CONFIG_FILE so a
# test can put a broken file in place BEFORE import (the ladder is decided at import time).
ARMS_FILE = Path(os.environ.get("CAD_ARMS_FILE", str(_HERE / "benchmarks" / "arms.json")))

def think_rung_available() -> bool:
    """Whether CODE_MODEL_THINK (the "+think" suffix) actually changes anything on the
    CURRENTLY ACTIVE strong-rung server, as opposed to being a no-op or a wrong-lever
    kwarg sent to a model that does not use it. Fix round 1 (2026-09-19), from a review
    finding: the think rung was originally built unconditionally from CODE_MODEL_STRONG,
    which is wrong whenever the strong rung resolves to the resident, not a maker arm.

    Measured 2026-09-19 on the two servers this engine can route the strong rung to:
    - the maker arms in benchmarks/arms.json launch llama.cpp with
      `--chat-template-kwargs {"enable_thinking":false}` (e.g. the gemma-4-31b arm's
      extra_args), so a per-request `enable_thinking: true` genuinely flips them from off
      to on -- this is the case the Task 1b live probe measured (67s / 1,864 completion
      tokens with reasoning vs 5s / 127 tokens without, on gemma-4-31b).
    - the resident qwen3.8-27b (maker.enabled=false) thinks by default; its depth lever
      is chat_template_kwargs.reasoning_effort. (CLAUDE.md said "enable_thinking is gone"
      for this model line; the measurement below shows enable_thinking:false still turns
      thinking OFF on this server, so that note is wrong and only the ON direction is moot.)
      Measured directly on the resident the same day: no kwarg = 37 completion tokens
      with reasoning; {"enable_thinking": false} = 2 tokens, no reasoning (no_think=True
      still works there, utility calls are fine); {"enable_thinking": true} = 37 tokens,
      IDENTICAL to the no-kwarg default, no error. So on the resident the plain rung
      already thinks, and "+think" is behaviourally the same request: offering it as a
      second, different escalation rung would be misleading, not merely redundant.

    Returns False (never raises) unless it can POSITIVELY confirm the active arm's
    benchmarks/arms.json entry launches with enable_thinking:false: maker disabled,
    arms.json missing/unreadable/malformed, the active arm absent from it, or its
    extra_args missing the flag all return False. benchmarks/arms.json is a card-only
    file with no guaranteed presence in every deployment, and this module is imported by
    the live engine, so a missing or broken file must degrade quietly, never crash import
    or a build.
    """
    # The WHOLE body is guarded, not just the JSON parse: CODE_MODEL_LADDER calls this at
    # import time, so any exception here would break `import cad_v5.config` and with it
    # every entry point (CLI, web UI, Satine, benchmarks). Fix round 2 (2026-09-19): a
    # review reproduced uncaught AttributeErrors on arms.json files that are VALID JSON of
    # the WRONG SHAPE ({"arms": "str"}, {"arms": {...}}, {"arms": ["x"]}, extra_args: 123).
    # The isinstance checks keep the normal path from leaning on the except.
    try:
        m = _MAKER
        if not isinstance(m, dict) or not m.get("enabled"):
            return False
        doc = json.loads(ARMS_FILE.read_text())
        arms = doc.get("arms") if isinstance(doc, dict) else None
        if not isinstance(arms, list):
            return False
        arm_name = m.get("arm")
        for arm in arms:
            if not isinstance(arm, dict) or arm.get("name") != arm_name:
                continue
            extra = arm.get("extra_args")
            if not isinstance(extra, str):
                return False
            # Whitespace-tolerant: extra_args is a shell-quoted string, not JSON, so
            # "enable_thinking": false vs "enable_thinking":false are both valid.
            return '"enable_thinking":false' in "".join(extra.split())
        return False
    except Exception:
        return False

# Two-rung ladder (local) ONLY where the second rung is a genuinely different request
# (think_rung_available() above); otherwise the ladder stays one rung, exactly as before
# Task 1b, rather than offering a rung that "escalates" to an identical call. A configured
# cad.json `cloud` block still appends a paid rung above whichever local ladder this is,
# at runtime (cad_engine._ladder()).
CODE_MODEL_LADDER  = ([CODE_MODEL_STRONG, CODE_MODEL_THINK] if think_rung_available()
                     else [CODE_MODEL_STRONG])
CODE_MODEL_DEFAULT = CODE_MODEL_STRONG
# CAD_CRITIC_MODEL env override exists for A/B evals (2026-08-15: gemma4 vs the resident 35B,
# now that the qwen36-server carries an mmproj) — same pattern as CAD_CODE_MODEL_FAST.
# "local:<name>" routes the critic through the resident llama.cpp server (images supported).
# Phase 1 lock-in (2026-09-17): the coder judges its own two-panel render ("self-critic") on the
# same llama.cpp server. Measured on Gemma-4-31B against the Ollama gemma4:e4b critic, paired on
# the same 40 public specs: match 40% vs 35% (+2/-0 flips), 22% faster, no VRAM cost, and no
# Ollama model left in the loop. CAD_CRITIC_MODEL still overrides, but only with another
# local: alias — Ollama came off the box 2026-09-19 and _ollama() rejects bare tags.
# NEVER point this at CODE_MODEL_THINK (or set CAD_CRITIC_MODEL to a "+think" string): a
# visual critique is a judgment call, not code, and _ollama() already forces thinking off
# for every images= call regardless of rung, so a +think critic would just pay the reasoning
# tokens for no effect. The default here is CODE_MODEL_STRONG, which never carries the
# suffix, so this is thinking-off by construction, not by convention.
CRITIC_MODEL       = os.environ.get("CAD_CRITIC_MODEL", CODE_MODEL_STRONG)
# CAD_CRITIC_URL lets the critic run on its own server (e.g. a dedicated vision model
# on a different port) instead of riding the coder server. Defaults to LOCAL_CODER_URL
# so an unset env var is a no-op change from the prior single-URL behaviour.
CRITIC_URL         = os.environ.get("CAD_CRITIC_URL", LOCAL_CODER_URL)
CRITIC_HEALTH      = CRITIC_URL.replace("/v1/chat/completions", "/health")
# Default per-call LLM timeout (was OLLAMA_TIMEOUT; the Ollama host/URL/tags constants
# were deleted 2026-09-19 with the Ollama rung itself).
LLM_TIMEOUT    = 300
CODE_TIMEOUT   = 600
# The strong rung pays a cold model swap (resident out, maker arm in) on the first call of a
# build; 600s killed a build at exactly +600s while a 30B was still loading (2026-07-11).
# Budget the swap + one slow generation. Since 2026-09-19 this is the ONLY local rung, so
# CODE_TIMEOUT below is only reached by a pinned cad.code_model.
CODE_TIMEOUT_STRONG = int(os.environ.get("CAD_CODE_TIMEOUT_STRONG", 1200))
# Env-overridable for critic A/B legs (a 35B critique pays CPU image-encode + a possible
# server restart; a timeout silently degrades the loop to gate-only, poisoning the leg).
CRITIC_TIMEOUT = int(os.environ.get("CAD_CRITIC_TIMEOUT", 200))
# Reference-image builds: two images through gemma's CPU-side vision encoder need more headroom
# than the single-render 200s (measured 2026-07-17: 110s cold for a two-image compare).
REF_CRITIC_TIMEOUT = 300
REF_IMAGE_MAX_PX   = 1024   # downscale reference photos to this long edge before base64/vision

# One build at a time across ALL frontends (CLI / Satine / web) — the RX 6600 fits one model.
# engine.build() takes an fcntl.flock on this file; flock self-releases on process death.
# Env-overridable (Task 3 fix H4, 2026-09-19) so lab/harvest.py's cheap `--check-gate`
# lock-free probe, and its tests, never touch the real lock file. Same pattern as
# CAD_CONFIG_FILE/MAKER_ENV/CAD_ARMS_FILE above.
BUILD_LOCK_FILE = Path(os.environ.get("CAD_BUILD_LOCK_FILE", str(_OPENCLAW / "cad-build.lock")))

# ── GIFT-style sampling + SFT-pair harvest (2026-07-19, arXiv 2603.27448) ─────
# Best-of-N first-turn sampling: draw N initial candidates at varied temperatures and let the
# deterministic gate pick the survivor — the GIFT paper's pass@N data says candidate diversity
# is worth double-digit accuracy (their pass@1->pass@10 gap was 15.5%). Default 3 per the
# 2026-07-19 A/B (tiers 1-2, fast rung, same engine both legs, capped unit, no oomd kills):
# N=3 doubled acceptance 5/22 -> 10/22 AND cut suite wall time 2097s -> 1221s — a good first
# candidate saves whole edit turns, so sampling pays for itself. gift-night-20260719.log +
# run_20260719_191156/193217.json. Resolution order: CAD_CANDIDATES env (CLI --candidates /
# benchmark legs) > cad.json `candidates` > this default. Read at build time, not import time,
# so the CLI flag can set the env var after modules load. (Default lives HERE, not cad.json —
# run_refresh.sh's exit trap rewrites cad.json to {}.)
CANDIDATE_TEMPS = [0.15, 0.45, 0.7]   # cycled across candidates; index 0 = the classic default
CANDIDATES_DEFAULT = 3

def first_turn_candidates() -> int:
    env = os.environ.get("CAD_CANDIDATES")
    try:
        if env:
            return max(1, int(env))
        return max(1, int(load_config().get("cad", {}).get("candidates", CANDIDATES_DEFAULT)))
    except Exception:
        return CANDIDATES_DEFAULT

# Fine-tune data harvest (feeds the budget-gated M6' QLoRA): organic builds record
# (spec, code) pairs and GIFT-FAIL pairs (render of a wrong turn + the final CORRECT code)
# here. Images are COPIED in — cad-builds/ rotates at KEEP_BUILDS and must not eat the
# dataset. Benchmark runs (CAD_BENCH=1) are excluded, same contamination rule as Stage C.
SFTPAIRS_DIR  = _OPENCLAW / "cad-sftpairs"
SFTPAIRS_FILE = _OPENCLAW / "cad-sftpairs.jsonl"

# ── Loop config ─────────────────────────────────────────────────────────────--
MAX_TURNS      = 4
ESCALATE_AFTER = 2
# N1 (2026-07-09): a syntax error or run exception is the 7B's dominant failure mode and needs
# zero visual judgment, so it gets an inline auto-fix micro-loop INSIDE the turn (same coder,
# raw error re-prompt) before the failure burns a full turn / touches the escalation ladder.
N1_RETRIES     = 2
BUILD_TIMEOUT  = int(os.environ.get("CAD_BUILD_TIMEOUT", 1800))
STEP_TIMEOUT   = 120
RENDER_TIMEOUT = 120
STL_TIMEOUT    = 120
INSPECT_TIMEOUT = 240   # was 60: sized for plates. A helical thread or a vaned
                        # impeller has orders more faces — the 2026-07-30 lead-screw
                        # spec (C10) timed out at 60s and was lost as an 'error'.
TRANSLATE_TIMEOUT = 120
BASE_URL       = "https://cad.onshape.com"
DONE_SENTINEL  = "###DONE###"

# ── CAM: print target (M9/X2a) ─────────────────────────────────────────────────
# Bambu system slicer profiles, extracted ONCE from the OrcaSlicer AppImage on first use
# (see cad_v5/cam_print.py::ensure_profiles) and cached here — this dir is the install path
# for a fresh machine, not just this session's scratch.
CAM_PROFILES_DIR       = _OPENCLAW / "cam-profiles"          # …/BBL/{machine,process,filament}/*.json
PRINT_MACHINE_DEFAULT  = "Bambu Lab A1 0.4 nozzle.json"
PRINT_PROCESS_DEFAULT  = "0.20mm Standard @BBL A1.json"
PRINT_FILAMENT_DEFAULT = "Bambu PLA Basic @BBL A1.json"
SLICE_TIMEOUT          = 300

# ── CAM: CNC 2.5D toolpaths via FreeCAD Path/CAM (M11/X2c) ─────────────────────
# A small plate's Pocket+Drilling recompute takes a few seconds; the AppImage cold-start
# (~10-20s) dominates the wall time. See cad_v5/cam_cnc.py for the full empirical writeup.
CNC_TIMEOUT = 300

def print_config() -> dict:
    """cad.json `print` block — CAM print-target overrides (same file/pattern as `public_uploads()`
    and `cloud_config()` above — NEVER put this in openclaw.json, only cad.json):
      {"machine": "Bambu Lab A1 0.4 nozzle.json", "process": "0.20mm Standard @BBL A1.json",
       "filament": "Bambu PLA Basic @BBL A1.json"}
    Bare names resolve inside the extracted BBL profile tree's machine/process/filament
    subdirs; absolute paths pass straight through. Missing block or missing keys fall back to
    the PRINT_*_DEFAULT trio above (measured to slice a test cube to 75 layers, return_code 0)."""
    return load_config().get("cad", {}).get("print") or {}

# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if logging.getLogger().handlers:   # already configured (e.g. v4 imported alongside)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stderr)],
    )

_setup_logging()
log = logging.getLogger("cad_v5")

# ── Credentials (load_config() itself lives above, before the model constants) ─
def creds() -> tuple[str, str]:
    cfg = load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

def use_brief() -> bool:
    """Should the generated brief shape the coder prompt? Default FALSE (2026-07-30).

    The brief existed to structure a vague request for a small coder, but it is authored by the
    weakest model in the chain, is non-deterministic, and makes prompt-wording experiments
    unattributable. Set cad.use_brief=true in ~/.openclaw/cad.json to restore the legacy path.
    """
    return bool(load_config().get("cad", {}).get("use_brief", False))


def repair_think_enabled() -> bool:
    """Should fluid mode's ONE crash-salvage turn and ONE gate-repair turn ride the think
    rung (CODE_MODEL_THINK) instead of the plain strong rung? Default FALSE (2026-09-19,
    Task 1b): a repair-only think turn has never been measured, see the Task 1b probe in
    docs/plans/2026-09-19-phase3-data-engine.md, and the controller, not this code, decides
    whether to flip the default after reviewing that data. The first codegen attempt is
    never affected either way. CAD_REPAIR_THINK=1 overrides cad.json's `repair_think` key.
    """
    if os.environ.get("CAD_REPAIR_THINK") == "1":
        return True
    return bool(load_config().get("cad", {}).get("repair_think", False))


def public_uploads() -> bool:
    # Free Onshape accounts can ONLY create public documents, so public is the default.
    return bool(load_config().get("cad", {}).get("public_uploads", True))

def cloud_config() -> dict:
    """cad.json `cloud` block — the paid escalation rung above the local ladder:
      {"provider": "anthropic"|"openrouter", "model": "...",
       "api_key_env": "ANTHROPIC_API_KEY", "max_calls_per_build": 4}
    Returns {} (rung disabled) unless provider+model are both set."""
    c = load_config().get("cad", {}).get("cloud") or {}
    return c if c.get("model") and c.get("provider") in ("anthropic", "openrouter") else {}

_LAB_DEFAULTS = {
    "harvest": {
        "night_start": "22:00",
        "night_end": "07:00",
        # Task 3 ruling (2026-09-19): night-only until the owner says otherwise. A unit
        # evicts and restores the resident (about 70s of model loads) and there is one
        # GPU, so daytime harvesting would compete with interactive CAD builds.
        "day_allowed": False,
        "hours_per_day": 12,
        # Task 3 ruling (2026-09-19): raised from 10 -> 25 (paired with the timer's own
        # 30-minute OnCalendar), so a unit's ~70s eviction/restore overhead is a smaller
        # fraction of its wall clock and consecutive units don't thrash the GPU swap.
        "unit_minutes": 25,
        "candidates": 3,
        "temps": [0.2, 0.5, 0.8],
        # Fix round 2 (2026-09-19), Section C: tier 3-4 specs sample wider before giving
        # up on finding a confirming (agreeing) partner -- a reviewer found the gate
        # passes structurally-clean geometry that names the wrong feature (V064, V066;
        # see tests/fixtures/harvest_agreement_fixtures.json), so a single gate-clean
        # candidate is no longer "good" on its own (see the "agreement" block below).
        # Tiers 1-2 keep the original 3-candidate budget -- their geometry is simple
        # enough that the old single-sample gate rarely disagreed with itself.
        "candidates_tier34": 5,
        "temps_tier34": [0.2, 0.35, 0.5, 0.65, 0.8],
        "max_pairs_per_spec": 2,
        # Fix round 2, Section C: the per-spec attempt budget before giving up. A spec
        # that has used its full teacher_attempt_cap of think-pass attempts and still
        # has fewer than max_pairs_per_spec confirmed pairs is "exhausted" -- dropped
        # from both the student and teacher pools (see _eligible_pools) so the unit
        # scheduler stops paying GPU time on it every night. Its unconfirmed candidates
        # are NOT deleted: they stay in lab/state/candidates.jsonl in case Phase 3's
        # later reference-scoring work can resolve them without a new sample.
        # student_attempt_cap unchanged at 2 (matches the pre-existing promotion rule);
        # teacher_attempt_cap of 5 gives one full think-pass round (candidates_tier34's
        # own width) before a spec is written off.
        "attempt_caps": {"student": 2, "teacher": 5},
        # Fix round 2, Section A: tolerances for signatures_agree() -- the cross-
        # candidate geometric-agreement check that replaced "gate-clean alone is good
        # enough". Bore diameters are compared after rounding to bore_round_mm inside
        # signature() itself (equal after rounding, not toleranced here).
        "agreement": {
            "volume_tol_pct": 0.05,
            "bbox_tol_mm": 0.05,
            "bore_round_mm": 0.01,
        },
        # Fix round 2, Section B: harvest-local strict spec checks, independent of and
        # in addition to the engine's own deterministic gate (engine.verify_expected).
        # envelope_tol_mm is deliberately tighter than the engine's own axis-dimension
        # tolerance (max(1, 5%) mm) -- V064 (180x130x55mm spec, 180x130x53mm measured,
        # 2mm off) passed the engine gate's 2.75mm tolerance but is visibly the wrong
        # part (a cutter sheared the whole top off); through-hole counting has no
        # engine-side tolerance concept at all, it is pass/fail against the stated count.
        "strict": {
            "envelope_tol_mm": 0.2,
        },
        # Task 1b (2026-09-19): teaching is a per-request think PASS on the SAME arm, not a
        # separate model or arm, so the key is named for what it now holds (fix round 1;
        # the plan's own Global Constraints block always called this teacher_passes, the
        # code lagged it). "gemma-4-31b-think" (a distinct server launch) is retired along
        # with the benchmarks/arms.json entry of that name. Task 3's harvest unit reads
        # this list to decide which passes to try when a spec fails the student twice, and
        # "think" means "the active rung with enable_thinking on" (only where
        # think_rung_available() agrees it is a real second rung), never a second arm.
        # devstral-small-2 is dropped too: Phase 0 measured it weaker than the student
        # (15% invalid / 22 matches vs Gemma-4-31B's 6% / 39 on CADPrompt), and the
        # Decisions section rules a weaker model out as a teacher regardless of gating.
        "teacher_passes": ["think"],
        # at most one unit in this many is a think unit while the student pool has work
        "think_unit_every": 4,
        # Task 3 fix L9 (2026-09-19): retention cap for lab/state/builds/, mtime-based
        # (never the newest), same discipline as cad_engine's own KEEP_BUILDS fix. A
        # pruned build dir may belong to an already-recorded pair -- fair game, since the
        # pair row holds the code itself; the build dir is only supplementary render/
        # mesh evidence.
        "keep_builds": 500,
        # D4 (2026-09-20): consecutive timer ticks that may yield to an in-flight
        # request on the resident before one proceeds anyway. 0 disables yielding.
        # The bound exists so a long chat turn cannot postpone harvesting for ever;
        # lab/gpu_window.sh still drains before it evicts, so the tick that does
        # proceed is not a hard cut-off. See lab/harvest.py _unit_gate().
        "max_busy_skips": 3,
        # Task 3c (2026-09-19, fix round for the first real unit's zero-pair result):
        # a round only samples this many candidates before checking whether ANY of them
        # was gate-clean; if none was, the round stops there rather than burning the
        # rest of the tier's candidate budget on a spec the model is currently unable to
        # build at all (see lab/harvest.py's own module docstring and sample_spec()).
        "probe_candidates": 2,
        # Task 3c: OFF by default -- the first real unit spent a repair codegen call
        # (plus a rebuild) on every crash, and a salvage candidate can neither confirm
        # nor be confirmed (fix round 3, H1), so with salvage on every one of those
        # calls was pure cost. Set true to restore the old always-salvage behaviour.
        "salvage": False,
        # Task 3c: a weighted round-robin over tiers, replacing "tier 3-4 first while
        # the pairs' tier34 share is under 0.40" (with zero pairs that condition never
        # stops being true, so the scheduler started on the hardest, mostly model-
        # written specs every single round). Tier 2 and 3 carry the most weight; tier 3
        # (the "hard but not extreme" band) still gets the most GPU time, but no longer
        # monopolises the front of the queue the way "tier 3-4 always first" did.
        "tier_weights": {"1": 1, "2": 3, "3": 4, "4": 1},
        # Task 3c: combined with the unit's own calendar date to seed the deterministic
        # shuffle _order_specs() uses to break ties within one scheduling group -- see
        # that function's own docstring. Two units on the same calendar day reproduce
        # the same order (useful for a smoke re-run); the next day's units do not repeat
        # it verbatim.
        "seed": 1,
    }
}


def _valid_tier_weights(tw) -> bool:
    """True when `tw` is a non-empty dict of tier -> a non-negative number with a
    strictly positive sum (Task 3c: "tier_weights as a list/string/negative/zero-sum
    must fall back to the default"). A list or a string is already caught upstream by
    `_sanitize_against_defaults`'s own "not a dict" fallback (tier_weights' own default
    is a dict, so a non-dict override there is replaced before this function ever sees
    it) -- this function exists for the narrower case that passes that generic check
    (a real dict of numbers) but is still unusable: every weight zero/negative, or the
    whole thing summing to zero, which would either stall the scheduler or divide by
    zero inside _tier_weight()'s own callers."""
    if not isinstance(tw, dict) or not tw:
        return False
    total = 0.0
    for v in tw.values():
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        if v < 0:
            return False
        total += v
    return total > 0


def _sanitize_against_defaults(defaults: dict, value, path: str = "lab") -> dict:
    """Recursively coerce `value` into the shape of `defaults`, so a malformed cad.json
    `lab` block degrades to safe defaults piece by piece instead of raising or silently
    handing a caller the wrong type deeper in the call stack (Task 3's harvest.py reads
    lab_config()["harvest"] as a dict on every unit).

    Fix round 1: `{"lab": "x"}`, `{"lab": ["x"]}` and `{"lab": 5}` used to reach
    `_deep_merge`'s unconditional `override.items()` and raise AttributeError; a
    malformed NESTED value (`{"lab": {"harvest": "oops"}}`) used to silently overwrite
    the whole `harvest` default with a non-dict, which is just the same bug one call
    deeper. Both cases now fall back to that subtree's default and log one warning
    naming the offending path (e.g. "lab" or "lab.harvest"), never crash, never mix a
    wrong type into an otherwise-good config.

    Fix round 2: every path returns a `copy.deepcopy` of the relevant defaults subtree,
    never the module-level `_LAB_DEFAULTS` object (or any of its nested dicts/lists) by
    reference. The old `dict(defaults)` was a SHALLOW copy: a caller that mutated a
    nested value in place (e.g. `lab_config()["harvest"]["temps"].append(...)`, an
    ordinary pattern for a caller composing a response dict) would silently corrupt the
    process-wide default for the rest of that process's life -- this module is imported
    by the live engine and, per the plan, will be called repeatedly from long-lived
    processes (the harvest unit, the web UI), so that corruption would outlive any one
    caller. Deep-copying `defaults` unconditionally, then overwriting only the keys the
    override actually names, means the returned dict never shares structure with
    `_LAB_DEFAULTS` no matter which branch is taken."""
    if not isinstance(value, dict):
        if value is not None:
            log.warning("[v5] cad.json's %s block is not an object (%r); using defaults.",
                        path, value)
        return copy.deepcopy(defaults)
    out = copy.deepcopy(defaults)
    for k, v in value.items():
        if isinstance(defaults.get(k), dict):
            out[k] = _sanitize_against_defaults(defaults[k], v, f"{path}.{k}")
        else:
            out[k] = v
    return out


def lab_config() -> dict:
    """cad.json `lab` block (Phase 3 "data engine", the harvest unit that samples the
    maker arm's own verified builds into training pairs), sanitized against the defaults
    below so a partial or malformed override keeps every default it doesn't name (and
    never crashes or leaks a wrong type on a malformed one):
      {"harvest": {"night_start", "night_end", "day_allowed", "hours_per_day",
                    "unit_minutes", "candidates", "temps", "candidates_tier34",
                    "temps_tier34", "max_pairs_per_spec", "attempt_caps",
                    "agreement", "strict", "teacher_passes", "keep_builds",
                    "max_busy_skips",
                    "probe_candidates", "salvage", "tier_weights", "seed"}}
    Same file/pattern as maker_config()/cloud_config()/print_config() above: never put
    this in openclaw.json, only cad.json. See docs/plans/2026-09-19-phase3-data-engine.md
    Global Constraints for where these defaults come from.

    `tier_weights` gets one extra pass beyond the generic dict-shape sanitizing above
    (Task 3c): a list or a string there already falls back to the default via the
    generic "not a dict" branch, but a real dict of negative or all-zero weights would
    pass that check and still be unusable, so `_valid_tier_weights` is checked
    explicitly and the whole sub-value (never a partial merge of it) falls back to
    `_LAB_DEFAULTS`'s own tier_weights, deep-copied, on any failure -- this function
    never raises regardless of what cad.json holds."""
    user = load_config().get("cad", {}).get("lab")
    out = _sanitize_against_defaults(_LAB_DEFAULTS, user)
    tw = out.get("harvest", {}).get("tier_weights")
    if not _valid_tier_weights(tw):
        log.warning("[v5] cad.json's lab.harvest.tier_weights is invalid (%r); using "
                    "defaults.", tw)
        out["harvest"]["tier_weights"] = copy.deepcopy(_LAB_DEFAULTS["harvest"]["tier_weights"])
    return out


def tg_token() -> str:
    cfg = load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# NOTE: per-build code-model routing (triage / escalation / manual --coder) lives in the
# engine (cad_engine._ACTIVE_CODE_MODEL / _code_model). A parallel holder here was dead
# code with no consumer and was removed; extract it for real when codegen.py is split out.

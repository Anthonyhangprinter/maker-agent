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
# cad_engine._ollama(), against the resident :8086, or the maker server when cad.json
# maker.enabled (the engine is the evictor, so its health probe must see backend truth —
# the :8085 gpu-proxy would happily queue it).
# The always-there resident (qwen38-server.service on :8086, behind the gpu-proxy on :8085).
# Single-sourced here because the web UI's title worker needs the alias too, and the CAD
# rungs cannot name it once a maker arm is enabled.
RESIDENT_ALIAS = "qwen3.8-27b"
RESIDENT_PROXY_URL = "http://127.0.0.1:8085/v1/chat/completions"

def maker_config() -> dict:
    """The optional swappable CAD coder server ("maker" block in cad.json).

    Disabled (default): the strong rung is the resident on :8086.
    Enabled: the strong rung is `maker-server` on `port`, serving `alias`.

    A stale `alias` left behind by `scripts/arms.py restore` (which only flips `enabled`
    to false) must NOT keep routing strong-rung calls at a model the resident does not
    serve, so port, alias and unit are all read from the block only while enabled is true.
    """
    m = load_config().get("cad", {}).get("maker") or {}
    if not bool(m.get("enabled", False)):
        return {"enabled": False, "port": 8086, "alias": RESIDENT_ALIAS, "unit": "qwen38-server"}
    return {
        "enabled": True,
        "port": int(m.get("port", 8088)),
        "alias": str(m.get("alias", RESIDENT_ALIAS)),
        "unit": "maker-server",
    }


_MAKER = maker_config()
CODE_MODEL_STRONG  = "local:" + _MAKER["alias"]
LOCAL_CODER_PORT   = _MAKER["port"]
LOCAL_CODER_URL    = f"http://127.0.0.1:{LOCAL_CODER_PORT}/v1/chat/completions"
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
# Two-rung ladder (local): the strong rung, then the same arm with thinking on. A configured
# cad.json `cloud` block still appends a paid rung above both at runtime (cad_engine._ladder()).
CODE_MODEL_LADDER  = [CODE_MODEL_STRONG, CODE_MODEL_THINK]
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
BUILD_LOCK_FILE = _OPENCLAW / "cad-build.lock"

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
        "day_allowed": True,
        "hours_per_day": 12,
        "unit_minutes": 10,
        "candidates": 3,
        "temps": [0.2, 0.5, 0.8],
        "max_pairs_per_spec": 2,
        # Task 1b (2026-09-19): teaching is a per-request think pass on the SAME arm, not a
        # separate model. "gemma-4-31b-think" (a distinct server launch) is retired along
        # with the benchmarks/arms.json entry of that name. Task 3's harvest unit reads
        # this list to decide which passes to try when a spec fails the student twice, and
        # "think" means "the active rung with enable_thinking on", never a second arm.
        # devstral-small-2 is dropped too: Phase 0 measured it weaker than the student
        # (15% invalid / 22 matches vs Gemma-4-31B's 6% / 39 on CADPrompt), and the
        # Decisions section rules a weaker model out as a teacher regardless of gating.
        "teacher_arms": ["think"],
    }
}


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
                    "unit_minutes", "candidates", "temps", "max_pairs_per_spec",
                    "teacher_arms"}}
    Same file/pattern as maker_config()/cloud_config()/print_config() above: never put
    this in openclaw.json, only cad.json. See docs/plans/2026-09-19-phase3-data-engine.md
    Global Constraints for where these defaults come from."""
    user = load_config().get("cad", {}).get("lab")
    return _sanitize_against_defaults(_LAB_DEFAULTS, user)


def tg_token() -> str:
    cfg = load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# NOTE: per-build code-model routing (triage / escalation / manual --coder) lives in the
# engine (cad_engine._ACTIVE_CODE_MODEL / _code_model). A parallel holder here was dead
# code with no consumer and was removed; extract it for real when codegen.py is split out.

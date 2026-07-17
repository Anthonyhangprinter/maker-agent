"""Shared configuration: paths, models, loop constants, config/credential loaders, logging.

Extracted verbatim from cad_engine.py (v4.3) so behaviour is identical. The CAD builder root
(`_HERE`) resolves to the parent package dir so `scripts/`, `b123d/`, and `cad_retrieval` still
resolve exactly as before.
"""
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
CAD_CONFIG_FILE = _OPENCLAW / "cad.json"   # agent settings (cad.*) — separate file because the
                                           # OpenClaw gateway's strict schema rejects unknown keys
SCRIPTS_DIR   = _HERE / "scripts"
B123D_DIR     = _HERE / "b123d"

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

# ── Models ────────────────────────────────────────────────────────────────────
BRIEF_MODEL        = "qwen3:8b"
# Fast rung = qwen3:8b (2026-07-12, tiers 1-2 A/B under identical v5.2 prompts):
# qwen3:8b 15/22 acceptance (68%), 5/6 valid geometry, 1052s — vs the code-tuned
# qwen2.5-coder:7b at 8/22 (36%), 3/6 geometry, 1242s. The general 8B repairs
# fillet/chamfer crashes the 7B spirals on, AND it keeps ONE model warm across
# brief/triage/codegen/decide (fewer VRAM swaps). Old 7B stays reachable via a
# cad.json code_model pin. Results: benchmarks/results/ (20260712 runs).
CODE_MODEL_FAST    = "qwen3:8b"
CODE_MODEL_STRONG  = "qwen3-coder:30b"                    # CPU offload, ~7min/call — last resort
# Escalation ladder, weakest first; failures climb one rung per trigger. There is no mid rung:
# the 14B was MEASURED OUT of the auto ladder (2026-07-04, m1_14b_tiers12.json): 3/6 converged
# at 583-804s/build — slower than the 30B MoE (dense 14B offloads worse than a 3B-active MoE)
# with worse results than the 7B (5/6 at ~250s), and the model itself was removed 2026-07-16.
# The right middle rung for weak hardware is the cloud (B4/M3).
CODE_MODEL_LADDER  = [CODE_MODEL_FAST, CODE_MODEL_STRONG]
CODE_MODEL_DEFAULT = CODE_MODEL_FAST
CRITIC_MODEL       = "gemma4:e4b"
OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_URL     = OLLAMA_HOST + "/api/generate"
OLLAMA_TAGS    = OLLAMA_HOST + "/api/tags"
OLLAMA_TIMEOUT = 300
CODE_TIMEOUT   = 600
# The strong 30B rung is CPU-offloaded (~7min/call measured) and pays a cold model swap when the
# ladder escalates mid-build; 600s killed a build at exactly +600s while the 30B was still loading
# (2026-07-11). Budget the swap + one slow generation.
CODE_TIMEOUT_STRONG = 1200
# A model whose weights exceed this (GB, from /api/tags) cannot sit fully in the RX 6600's 8GB
# VRAM → CPU offload + a ~6min reload whenever the brief/critic evicts it. Such models get
# CODE_TIMEOUT_STRONG even when pinned by name (2026-07-17: a pinned qwen3.6:35b-a3b got the
# 600s fast timeout, timed out on all 10 benchmark parts, and produced a 0/31 artifact score).
VRAM_RESIDENT_GB_MAX = 5.5
CRITIC_TIMEOUT = 200
# Reference-image builds: two images through gemma's CPU-side vision encoder need more headroom
# than the single-render 200s (measured 2026-07-17: 110s cold for a two-image compare).
REF_CRITIC_TIMEOUT = 300
REF_IMAGE_MAX_PX   = 1024   # downscale reference photos to this long edge before base64/vision

# One build at a time across ALL frontends (CLI / Satine / web) — the RX 6600 fits one model.
# engine.build() takes an fcntl.flock on this file; flock self-releases on process death.
BUILD_LOCK_FILE = _OPENCLAW / "cad-build.lock"

# ── Loop config ─────────────────────────────────────────────────────────────--
MAX_TURNS      = 4
ESCALATE_AFTER = 2
# N1 (2026-07-09): a syntax error or run exception is the 7B's dominant failure mode and needs
# zero visual judgment, so it gets an inline auto-fix micro-loop INSIDE the turn (same coder,
# raw error re-prompt) before the failure burns a full turn / touches the escalation ladder.
N1_RETRIES     = 2
BUILD_TIMEOUT  = 1800
STEP_TIMEOUT   = 120
RENDER_TIMEOUT = 120
STL_TIMEOUT    = 120
INSPECT_TIMEOUT = 60
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

# ── Config / credentials ──────────────────────────────────────────────────────
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
        log.warning("[v5] openclaw.json unreadable (%s) — creds/telegram token unavailable.", e)
    try:
        with open(CAD_CONFIG_FILE) as f:
            cfg["cad"] = {**cfg.get("cad", {}), **json.load(f)}
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("[v5] cad.json unreadable (%s) — cad.* settings ignored.", e)
    return cfg

def creds() -> tuple[str, str]:
    cfg = load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

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

def tg_token() -> str:
    cfg = load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# NOTE: per-build code-model routing (triage / escalation / manual --coder) lives in the
# engine (cad_engine._ACTIVE_CODE_MODEL / _code_model). A parallel holder here was dead
# code with no consumer and was removed; extract it for real when codegen.py is split out.

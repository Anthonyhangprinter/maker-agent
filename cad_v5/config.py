"""Shared configuration: paths, models, loop constants, config/credential loaders, logging.

Extracted verbatim from cad_agent_v4.py (v4.3) so behaviour is identical. The CAD builder root
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
CONFIG_FILE   = _OPENCLAW / "openclaw.json"
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
CODE_MODEL_FAST    = "qwen2.5-coder:7b-instruct-q5_k_m"
CODE_MODEL_STRONG  = "qwen3-coder:30b"
CODE_MODEL_DEFAULT = CODE_MODEL_FAST
CRITIC_MODEL       = "gemma4:e4b"
OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_URL     = OLLAMA_HOST + "/api/generate"
OLLAMA_TAGS    = OLLAMA_HOST + "/api/tags"
OLLAMA_TIMEOUT = 300
CODE_TIMEOUT   = 600
CRITIC_TIMEOUT = 200

# ── Loop config ─────────────────────────────────────────────────────────────--
MAX_TURNS      = 4
ESCALATE_AFTER = 2
BUILD_TIMEOUT  = 1800
STEP_TIMEOUT   = 120
RENDER_TIMEOUT = 120
STL_TIMEOUT    = 120
INSPECT_TIMEOUT = 60
TRANSLATE_TIMEOUT = 120
BASE_URL       = "https://cad.onshape.com"
DONE_SENTINEL  = "###DONE###"

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
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def creds() -> tuple[str, str]:
    cfg = load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

def public_uploads() -> bool:
    # Free Onshape accounts can ONLY create public documents, so public is the default.
    return bool(load_config().get("cad", {}).get("public_uploads", True))

def tg_token() -> str:
    cfg = load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# NOTE: per-build code-model routing (triage / escalation / manual --coder) lives in the
# engine (cad_agent_v4._ACTIVE_CODE_MODEL / _code_model). A parallel holder here was dead
# code with no consumer and was removed; extract it for real when codegen.py is split out.

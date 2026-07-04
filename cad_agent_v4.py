#!/usr/bin/env python3
"""
CAD Agent v4.3 — build123d agentic observe-edit loop + Onshape upload.

Modeled on how the Claude<->Fusion MCP connection operates: the model writes a
build123d script, the system RUNS it, INSPECTS the geometry, RENDERS a two-panel view
(isometric + top-down), and a multimodal CRITIC (gemma4:e4b) describes whether it matches
the spec. That observation is fed back so the coder model EDITS the script and re-observes —
iterating until it judges the part correct (replies ###DONE###) or the bounds are hit.

v4.2: the coder is auto-routed — fast qwen2.5-coder:7b by default, an LLM triage starts hard
specs on qwen3-coder:30b, and the loop auto-escalates 7B->30B after repeated failures (override
with --coder auto|fast|strong). The brief interprets intent (box => hollow container). Structural
sections / gears / bolts use the b123d domain helpers inside the same loop — no separate
parametric path, no routing regex.

v4.3: Onshape upload now VERIFIES the translated Part Studio contains >=1 body before reporting
success — a translation can report DONE yet yield an empty Part Studio when the STEP holds
invalid geometry (e.g. a fillet on a hole rim OCCT tolerates but Onshape silently drops). Public
uploads are the default (free Onshape accounts can only create public documents).

Usage:
    python3 cad_agent_v4.py build "a 100x60x20mm electronics enclosure with 2mm walls"
    python3 cad_agent_v4.py build "W200x100 I-beam 1500mm"
    python3 cad_agent_v4.py rate <1-5> [comment]
    python3 cad_agent_v4.py session
"""

import os
import sys
import json
import re
import ast
import time
import math
import base64
import shutil
import logging
import socket
import subprocess
import tempfile
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

VERSION = "4.3"

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).parent
_OPENCLAW     = Path.home() / ".openclaw"
LOG_FILE      = _OPENCLAW / "cad-agent.log"
FEEDBACK_FILE = _OPENCLAW / "cad-examples.jsonl"   # unified corpus: gold + rated + auto
SESSION_FILE  = _OPENCLAW / "cad-session.json"
CONFIG_FILE     = _OPENCLAW / "openclaw.json"
CAD_CONFIG_FILE = _OPENCLAW / "cad.json"   # agent settings (cad.*) — separate file because the
                                           # OpenClaw gateway's strict schema rejects unknown keys
SCRIPTS_DIR   = _HERE / "scripts"

# Semantic few-shot retrieval (graceful — any failure here must never break a build).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import cad_retrieval
except Exception:
    cad_retrieval = None
B123D_DIR     = _HERE / "b123d"
STEP_OUT      = _OPENCLAW / "cad-last-build.step"   # persisted so a failed upload is recoverable
STL_OUT       = _OPENCLAW / "cad-last-build.stl"    # always exported alongside STEP — sliceable for printing
DXF_OUT       = _OPENCLAW / "cad-last-build.dxf"    # flat-pattern DXF (laser/sheet) — best-effort, plate-like parts
STL_VIEWER_CMD  = os.environ.get("CAD_STL_VIEWER", "fstl")  # local GUI STL viewer the chat auto-opens
CAD_VIEWER_PORT = int(os.environ.get("CAD_VIEWER_PORT", "4178"))  # browser CAD Viewer (text-to-cad skill)

# ── Models + loop constants — single source of truth is cad_v5/config.py ──────
# (previously duplicated here; the two copies drifted independently. Model/timeout/loop
# tuning now happens in ONE place and both the v4 engine and the v5 package follow.)
from cad_v5.config import (        # noqa: E402
    BRIEF_MODEL, CODE_MODEL_FAST, CODE_MODEL_MID, CODE_MODEL_STRONG, CODE_MODEL_LADDER,
    CODE_MODEL_DEFAULT, CRITIC_MODEL,
    OLLAMA_HOST, OLLAMA_URL, OLLAMA_TAGS, OLLAMA_TIMEOUT, CODE_TIMEOUT, CRITIC_TIMEOUT,
    MAX_TURNS, ESCALATE_AFTER, BUILD_TIMEOUT, STEP_TIMEOUT, RENDER_TIMEOUT, STL_TIMEOUT,
    INSPECT_TIMEOUT, TRANSLATE_TIMEOUT, BASE_URL, DONE_SENTINEL,
)
from cad_v5.diagnose import diagnose  # noqa: E402  (B3 failure taxonomy)
from cad_v5.config import cloud_config  # noqa: E402  (B4 cloud rung)

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stderr)],
    )

_setup_logging()
log = logging.getLogger("cad_v4")

# ── Config / credentials ──────────────────────────────────────────────────────

def _load_config() -> dict:
    """openclaw.json (channels/env — schema-validated by the OpenClaw gateway) overlaid
    with ~/.openclaw/cad.json as the `cad` block. The agent's settings live in their own
    file because a top-level "cad" key in openclaw.json fails the gateway's strict config
    validation (Unrecognized key) and prevents it from starting."""
    cfg: dict = {}
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:
        # A corrupt openclaw.json silently disabled creds and the telegram token; loud.
        log.warning("[v4] openclaw.json unreadable (%s) — Onshape creds / telegram token "
                    "unavailable until it is fixed.", e)
    try:
        with open(CAD_CONFIG_FILE) as f:
            cfg["cad"] = {**cfg.get("cad", {}), **json.load(f)}
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("[v4] cad.json unreadable (%s) — model pin / cad.* settings ignored.", e)
    return cfg

BUILDS_DIR = _OPENCLAW / "cad-builds"
KEEP_BUILDS = 20   # retention: prune oldest per-build artifact dirs beyond this

def _new_build_dir(spec: str) -> Path:
    """Per-build artifact dir — concurrent builds must never clobber each other's output."""
    slug = re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-")[:40] or "build"
    d = BUILDS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        old = sorted(p for p in BUILDS_DIR.iterdir() if p.is_dir())[:-KEEP_BUILDS]
        for p in old:
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        log.warning("[v4] build-dir pruning failed: %s", e)
    return d

def _creds() -> tuple[str, str]:
    cfg = _load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

def _public_uploads() -> bool:
    # Free Onshape accounts can ONLY create public documents, so public is the default.
    # A paid account can opt back into private with cad.public_uploads: false.
    return bool(_load_config().get("cad", {}).get("public_uploads", True))

# Per-build override set by build() (complexity triage / escalation / manual --coder). A build
# runs as its own process (Satine shells out per request), so a module global is safe here.
_ACTIVE_CODE_MODEL: Optional[str] = None

def _code_model() -> str:
    """Code model for the current call. Priority:
    per-build override (triage/escalation/manual) > openclaw.json cad.code_model > CODE_MODEL_DEFAULT."""
    if _ACTIVE_CODE_MODEL:
        return _ACTIVE_CODE_MODEL
    return _load_config().get("cad", {}).get("code_model") or CODE_MODEL_DEFAULT

CLOUD_PREFIX = "cloud/"
_CLOUD_CALLS_LEFT = 0   # per-build cost cap, reset by build() from cad.json cloud.max_calls_per_build

def _ladder() -> list:
    """The live escalation ladder: local rungs + a paid cloud rung when configured."""
    rungs = list(CODE_MODEL_LADDER)
    cc = cloud_config()
    if cc:
        rungs.append(CLOUD_PREFIX + cc["model"])
    return rungs

def _next_code_model(current: str):
    """The next rung up the escalation ladder, or None at (or off) the top."""
    ladder = _ladder()
    try:
        i = ladder.index(current)
    except ValueError:
        return None
    return ladder[i + 1] if i + 1 < len(ladder) else None

def _cloud_key(cc: dict) -> str:
    env_name = cc.get("api_key_env") or ("ANTHROPIC_API_KEY" if cc["provider"] == "anthropic"
                                         else "OPENROUTER_API_KEY")
    return (os.environ.get(env_name)
            or _load_config().get("env", {}).get(env_name, ""))

def _cloud_chat(model: str, system: str, prompt: str,
                timeout: int = 120, temperature: Optional[float] = None) -> str:
    """One chat call to the configured cloud provider. Plain HTTP, no SDKs. Budget-capped
    per build so a stuck loop can never run up a bill."""
    global _CLOUD_CALLS_LEFT
    if _CLOUD_CALLS_LEFT <= 0:
        # Budget exhausted must never crash a build (or run up a bill): an empty reply
        # flows through the normal stuck-handling, which keeps the best verified build.
        log.warning("[v4] Cloud call budget exhausted — returning empty reply "
                    "(raise cad.json cloud.max_calls_per_build to allow more).")
        return ""
    cc = cloud_config()
    key = _cloud_key(cc)
    if not key:
        raise RuntimeError(f"cloud rung configured but no API key found "
                           f"({cc.get('api_key_env') or 'default env'} unset)")
    _CLOUD_CALLS_LEFT -= 1
    if cc["provider"] == "anthropic":
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model, "max_tokens": 4096, "system": system,
                "messages": [{"role": "user", "content": prompt}],
                **({"temperature": temperature} if temperature is not None else {}),
            }).encode(),
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        return "".join(b.get("text", "") for b in resp.get("content", [])).strip()
    # openrouter (OpenAI-schema chat completions)
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            **({"temperature": temperature} if temperature is not None else {}),
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"].strip()

def _tg_token() -> str:
    cfg = _load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# ── Ollama LLM call (optional images for multimodal) ───────────────────────────

def _ollama(model: str, system: str, prompt: str,
            timeout: int = OLLAMA_TIMEOUT, images: Optional[list[str]] = None,
            temperature: Optional[float] = None, fmt=None) -> str:
    # fmt: "json" or a JSON-schema dict — Ollama enforces the output grammar server-side.
    if model.startswith(CLOUD_PREFIX):
        # The paid rung rides the same seam every local call uses — nothing upstream
        # knows or cares which provider answered. fmt is ignored (cloud rung = coder only).
        return _cloud_chat(model[len(CLOUD_PREFIX):], system, prompt,
                           timeout=min(timeout, 300), temperature=temperature)
    options = {"num_ctx": 16384}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {
        "model":  model,
        "stream": False,
        "system": system,
        "prompt": prompt,
        "options": options,
        "think":  False,
    }
    if fmt:
        payload["format"] = fmt   # e.g. "json" — constrains decoding to valid JSON
    if images:
        payload["images"] = images
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    # Retry once on a transient connection blip (e.g. Ollama briefly busy swapping a model),
    # but never on a timeout — a slow model should not be hit twice.
    import socket
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            return resp["response"].strip()
        except urllib.error.URLError as e:
            if attempt == 0 and not isinstance(e.reason, socket.timeout):
                log.warning("[v4] Ollama connection issue (%s) — retrying once", e)
                time.sleep(3)
                continue
            raise

# ── Preflight ─────────────────────────────────────────────────────────────────

def preflight() -> None:
    """Fail fast with a clear message if Ollama or the required models are missing.
    The critic model is optional — its absence only disables visual critique."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS, timeout=10) as r:
            tags = json.loads(r.read())
    except Exception as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_HOST} ({e}). Is `ollama serve` running?"
        )
    have = {m.get("name", "") for m in tags.get("models", [])}
    missing = [m for m in (BRIEF_MODEL, _code_model())
               if m not in have and not m.startswith(CLOUD_PREFIX)]
    if missing:
        raise RuntimeError(
            "Missing required Ollama model(s): " + ", ".join(missing) +
            ". Pull with: " + "; ".join(f"ollama pull {m}" for m in missing)
        )
    if CRITIC_MODEL not in have:
        log.warning("[v4] Critic model %s not installed — visual critique disabled "
                    "(loop falls back to numeric geometry state).", CRITIC_MODEL)

# ── Onshape REST ──────────────────────────────────────────────────────────────

def _onshape(method: str, path: str, body=None) -> dict:
    ak, sk = _creds()
    creds  = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Accept":        "application/json;charset=UTF-8",
        "Content-Type":  "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()[:400]
        raise RuntimeError(f"Onshape {method} {path} → {e.code}: {body_text}")

# ── Brief generation ──────────────────────────────────────────────────────────

_BRIEF_SYSTEM = """\
You are a CAD specification analyst. Convert the description into a structured build brief.
Return ONLY valid JSON — no explanation, no markdown fences:
{
  "name": "short document name (max 40 chars)",
  "description": "one-sentence description of the part",
  "dimensions": {"key": "value in mm — include all relevant dimensions"},
  "features": ["ONLY features explicitly requested or clearly implied: holes, fillets, threads, etc."],
  "notes": ["any constraints or special requirements"],
  "helper": "exact build123d domain-helper call if this is a standard section/gear/bolt, else \"\"",
  "expected": {
    "bbox_mm": [X, Y, Z] or null,
    "solids": 1,
    "min_holes": 0,
    "min_through_holes": 0,
    "bores_mm": [],
    "wall_mm": null,
    "feature_checks": []
  }
}
All values in millimetres.

The "expected" block is a deterministic acceptance target the build is checked against — be careful and literal:
  • "bbox_mm": the part's overall bounding-box size [X, Y, Z] in mm, ONLY when it follows
    unambiguously from the dimensions (e.g. an 80x50x8 base plate with a plate rising 50mm above it
    → [80, 50, 58]; an 80mm-outer-diameter flange → [80, 80, thickness]). If the outer extent is
    not clearly derivable, use null — never guess.
  • "solids": the number of separate solid bodies in the finished part. This is 1 for any single
    fused part (the normal case, even with brackets/gussets/ribs — they fuse into one solid). Use >1
    ONLY when the spec explicitly asks for multiple loose/unconnected bodies or an assembly.
  • "min_holes": how many holes, bores, or through-cutouts are explicitly requested (count a bolt
    circle of 6 as 6, a central bore as 1; 0 if none). Counterbores/countersinks count too.
  • "min_through_holes": of those holes, how many must pass FULLY THROUGH the part — i.e. the spec
    says "through-hole", "through-bore", "drilled through", "clearance hole", or a hole whose only
    purpose is to receive a bolt/shaft (bolt holes are through unless stated blind). Do NOT count
    blind holes, pockets, counterbores that stop part-way, or holes the spec calls "blind". A
    central "through-bore" plus 6 bolt holes = 7 through. If unsure whether a hole is through, and
    the spec uses the word "through", count it; otherwise leave it out of this number.
  • "bores_mm": the DIAMETERS in mm of any holes/bores whose size the spec actually states (e.g.
    "20mm wrist-pin bore" → [20]; "M5 clearance holes" → [5.5]; "drill 8mm and 12mm holes" →
    [8, 12]; a bolt circle of 6× M6 → [6.6]). List each DISTINCT diameter once. Use metric
    clearance sizes for "MN clearance/bolt holes" (M3→3.4, M4→4.5, M5→5.5, M6→6.6, M8→9). Leave
    [] when no specific hole diameter is given.
  • "wall_mm": the wall thickness in mm for a HOLLOW part / enclosure / tube when the spec states
    or clearly implies one (e.g. "2mm walls" → 2; "enclosure" with no thickness → 2). Use null for
    solid parts, plates, or when there is no wall.
  • "feature_checks": a GENERAL, verifiable list of the part's distinct features — each an object the
    finished geometry is checked against. This is how the build is verified to MATCH THE INTENT, so
    be literal and only list what the spec actually asks for. Supported objects:
      {"kind": "bore", "d_mm": <diameter>, "orientation": "radial"|"axial"|null}
         orientation = "radial" when the hole goes through the SIDE / wall, perpendicular to the
           part's long axis (a wrist-pin bore through a piston/cylinder SKIRT, a cross-hole through a
           shaft, a hole through a tube wall). "axial" when it runs ALONG the long axis (a central
           bore straight down a cylinder). Use null for a hole in a flat plate where it doesn't apply.
         CAUTION — a part's OUTER/running diameter is NOT a bore feature. "an 80mm bore" / "80mm
           bore diameter" on a PISTON, CYLINDER, or shaft means the part's OUTER diameter (it runs in
           an 80mm cylinder bore) — that is the part's SIZE (bbox), NOT a hole through it. Only add a
           bore feature_check for holes/cavities that are actually cut INTO the part. So an "80mm-bore
           piston" contributes Ø80 to bbox/diameter, and ONLY its wrist-pin hole is a bore feature.
      {"kind": "groove", "count": <N>}   # N circumferential ring grooves (piston rings, snap-ring, O-ring)
      {"kind": "bolt_circle", "count": <N>, "d_mm": <hole diameter>}   # N equally-spaced holes on a circle
    Examples:
      "a piston ... three ring grooves ... 20mm wrist-pin bore through the skirt"
         → [{"kind":"bore","d_mm":20,"orientation":"radial"}, {"kind":"groove","count":3}]
      "a shaft with a central 10mm bore and a 5mm cross-hole"
         → [{"kind":"bore","d_mm":10,"orientation":"axial"}, {"kind":"bore","d_mm":5,"orientation":"radial"}]
      "a flange with six M6 bolt holes" → [{"kind":"bolt_circle","count":6,"d_mm":6.6}]
    Empty list for a plain plate/box/bracket with no such distinct features.

Build EXACTLY what is asked — do NOT invent extra features (mounting holes, bosses, fillets,
chamfers) that were not requested.

INTERPRET THE NOUN'S MEANING — this is understanding intent, NOT inventing features. A "box",
"case", "enclosure", "housing", "container", "tray", or "bin" means a HOLLOW part with walls that
can hold something — NOT a solid block. It is open-topped unless the description mentions a lid /
closed top, or features that imply a top exists (e.g. "holes through the top face"). Use ~2mm
walls if no thickness is given, and record "hollow" (and "open top" when applicable) in features.
Treat it as a SOLID block only if the user says "solid", "block", "filled", or calls it a
"plate"/"panel"/"bracket"/"blank". So "a box with an open top" is a hollow open-topped box and
nothing else; "a solid box" or "a 100x60x10 plate" stays solid.

ASSEMBLY — when the spec asks for MULTIPLE distinct parts that fit/mate together (e.g. "a housing
AND a lid", "a bolt and nut", "a shaft in a bearing", "a box with a separate press-fit cover", "a
two-part clamp"): set "expected.solids" to the NUMBER of separate bodies, name each part in
"features", and add "assembly" to "notes". The parts stay SEPARATE bodies positioned where they
sit — they are NOT fused into one solid. A single part that merely has gussets/ribs/bosses/standoffs
is NOT an assembly (those fuse → solids:1). Only mark an assembly when the parts are genuinely
distinct objects.

Set "helper" ONLY when the WHOLE part IS one standard catalog component, with CONCRETE NUMBERS
(no placeholders). These call verified parametric libraries (correct by construction):
  I/H/W/UB/UC beam or C-channel → "structural_section(depth, flange_width, flange_thickness, web_thickness, length)"
  spur / involute gear          → "gear(module, teeth, thickness, bore=0)"   # bore optional, mm
  metric screw / cap screw / bolt → "screw('M5-0.8', length, head='socket')"  # head: socket|hex|countersunk|pan|button
  hex nut                       → "nut('M5-0.8')"
  ball bearing                  → "ball_bearing('M8-22-7')"   # 'bore-OD-width' in mm
  ASME pipe flange              → "pipe_flange('2', 150)"     # nominal pipe size, class
For a W/UB/UC designation like W200x100: depth≈200, flange_width≈100, and typical
flange_thickness≈12, web_thickness≈8 unless stated. Example: "structural_section(200, 100, 12, 8, 1500)".
Use a helper ONLY when the entire requested object is that one component. For ANY assembly,
bracket, enclosure, plate, box, or part that merely CONTAINS holes/gears/threads as features
(not IS one), leave "helper": "" and let the modeller build it."""

# JSON schemas for Ollama's grammar-constrained decoding (`format`). Enforcing the shape
# server-side frees a small model from spending capacity on formatting — it can only emit
# valid JSON matching the schema. Content quality is still the model's job.
_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "dimensions": {"type": "object"},
        "features": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "helper": {"type": "string"},
        "expected": {
            "type": "object",
            "properties": {
                "bbox_mm": {"type": ["array", "null"], "items": {"type": "number"}},
                "solids": {"type": "integer"},
                "min_holes": {"type": "integer"},
                "min_through_holes": {"type": "integer"},
                "bores_mm": {"type": "array", "items": {"type": "number"}},
                "wall_mm": {"type": ["number", "null"]},
                "feature_checks": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["solids", "min_holes", "min_through_holes"],
        },
    },
    "required": ["name", "description", "dimensions", "features", "notes", "helper", "expected"],
}

_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {"hard": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["hard", "reason"],
}

def _extract_json(raw: str) -> Optional[dict]:
    """Parse the first JSON object found anywhere in an LLM reply. raw_decode handles
    braces inside strings and ignores trailing prose, so fenced or prose-wrapped JSON
    (which defeated the old start-anchored fence-strip) is recovered."""
    dec = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = dec.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = raw.find("{", idx + 1)
    return None

def build_brief(spec: str) -> dict:
    # Primary path: schema-constrained decoding — the server guarantees shape-valid JSON.
    brief = None
    try:
        raw = _ollama(BRIEF_MODEL, _BRIEF_SYSTEM, f"Spec: {spec}",
                      timeout=OLLAMA_TIMEOUT, temperature=0.2, fmt=_BRIEF_SCHEMA)
        brief = _extract_json(raw)
    except Exception as e:
        log.warning("[v4] Schema-constrained brief failed (%s) — falling back to free-form.", e)
    if brief is None:
        raw = _ollama(BRIEF_MODEL, _BRIEF_SYSTEM, f"Spec: {spec}",
                      timeout=OLLAMA_TIMEOUT, temperature=0.2)
        brief = _extract_json(raw)
    if brief is None:
        # Silent degradation here previously produced an unguided, feature-ungated build.
        log.warning("[v4] Brief failed twice — building UNGUIDED from the raw spec "
                    "(no dimensions/features/expected; gate limited to universal checks).")
        brief = {"name": spec[:40], "description": spec,
                 "dimensions": {}, "features": [], "notes": [], "helper": ""}
    return brief

# Words that signal a part legitimately HAS a hole/recess that stops part-way (a blind feature).
# If a spec contains none of these, every hole it asks for is meant to pass fully through, and the
# gate forbids blind holes deterministically — without trusting the brief's unreliable hole counts.
_BLIND_HOLE_TERMS = re.compile(
    r"\b(blind|counterbore|counter-bore|c'?bore|pocket|recess(?:ed)?|"
    r"boss(?:es)?|standoff|spotface|spot-face|"
    r"deep\b|does ?n.?t go through|not (?:going )?through|stops? (?:part|short)|"
    r"partial(?:ly)? (?:through|depth)|tapp?ed hole|threaded hole|sink hole)\b",
    re.IGNORECASE)

def reconcile_expected(brief: dict, spec: str) -> None:
    """Deterministically harden the brief's `expected` block in place. The brief (qwen3:8b)
    under-counts holes and mis-assigns bbox axes; rather than trust its numbers, derive what we
    can from the spec text. Here: if the spec describes no blind/pocket/recess feature, mark that
    NO hole may be blind, so a half-cut through-bore is caught regardless of the brief's count."""
    exp = brief.setdefault("expected", {})
    if not isinstance(exp, dict):
        exp = brief["expected"] = {}
    exp["forbid_blind_holes"] = not bool(_BLIND_HOLE_TERMS.search(spec or ""))

# ── Code-model triage (LLM decides, not keyword rules) ────────────────────────

_COMPLEXITY_SYSTEM = """\
You decide which code model should build a CAD part. A small FAST coder reliably handles simple
parts: a handful of primitives (Box/Cylinder/Sphere/Cone) combined with + - &, holes/pockets,
fillets/chamfers, hollow shells, simple linear/radial patterns. A large STRONG coder is only
worth its ~25x time cost for genuinely hard parts: many interacting features, revolved/swept/
lofted profiles, non-trivial maths or patterns, or unusual geometry a small model would likely
get wrong. When unsure, prefer FAST (the agent escalates automatically if the fast model fails).
Reply with ONLY JSON: {"hard": true|false, "reason": "<short>"}"""

def spec_needs_strong_coder(spec: str, brief: dict) -> bool:
    """LLM triage (not keyword rules) of whether a spec warrants the strong coder up front.
    Domain-helper parts bypass codegen entirely, so they never need it."""
    if (brief.get("helper") or "").strip():
        return False
    try:
        prompt = (f"Spec: {spec}\nFeatures: {brief.get('features', [])}\n"
                  f"Dimensions: {brief.get('dimensions', {})}\n\nDecide:")
        raw = _ollama(BRIEF_MODEL, _COMPLEXITY_SYSTEM, prompt,
                      timeout=OLLAMA_TIMEOUT, temperature=0.0, fmt=_TRIAGE_SCHEMA)
        verdict = _extract_json(raw) or {}
        hard = bool(verdict.get("hard"))
        log.info("[v4] Complexity triage: hard=%s — %s", hard, str(verdict.get("reason", ""))[:140])
        return hard
    except Exception as e:
        log.warning("[v4] Complexity triage failed (%s) — staying on fast coder.", e)
        return False

# ── Stage B: distil a fail->fix lesson from a build that recovered ─────────────

_LESSON_SYSTEM = """\
You are reviewing a build123d CAD build that initially FAILED, then was fixed and converged.
State the ONE concrete, reusable lesson — the specific mistake and the fix — as a single imperative
sentence a coder could apply next time (e.g. "For through-holes, place the cutting Cylinder at z=0
with height greater than the part thickness so it pierces fully"). It must be specific to build123d
geometry, not generic advice. If there is no concrete, generalizable lesson, reply exactly: NONE.
No preamble, one line only."""

def distill_lesson(spec: str, problem: str, final_code: str) -> Optional[str]:
    """Turn a recovered failure into one reusable pitfall, or None if nothing generalizable."""
    try:
        prompt = (f"Spec: {spec}\n\nWhat went wrong first:\n{problem}\n\n"
                  f"Final working code:\n{final_code[:1500]}\n\nThe one reusable lesson:")
        raw = _ollama(BRIEF_MODEL, _LESSON_SYSTEM, prompt,
                      timeout=OLLAMA_TIMEOUT, temperature=0.1).strip().strip('"')
        if not raw or raw.upper().startswith("NONE") or len(raw) < 15:
            return None
        return raw.splitlines()[0].strip()[:240]
    except Exception as e:
        log.warning("[v4] Lesson distillation failed: %s", e)
        return None

# ── Code generation ───────────────────────────────────────────────────────────

_CODE_SYSTEM = """\
You are a build123d expert. Write a complete, runnable build123d script using ALGEBRA MODE —
combine solid primitives with the + - & operators. Algebra mode is far simpler and more
reliable than BuildPart/BuildSketch; prefer it for everything.

RULES:
1. First line: from build123d import *
   (optionally) from b123d.domain import structural_section, spur_gear, hex_bolt
2. All dimensions in MILLIMETRES. Centre the part at the origin.
3. Build solids, combine them, and assign the final solid to `result`.
4. Do NOT call export_step() — the runner exports `result`.
5. Return ONLY Python code — no prose, no markdown fences.
6. PARAMETERS: define every key dimension as a named constant at the TOP of the script,
   directly after the imports (e.g. `wall = 2.0`, `cable_hole_d = 20.0`), and use those
   names in the geometry. This makes the part editable without an AI: the user can change
   a number and regenerate. Plain numeric assignments only — no expressions on those lines.

COORDINATE SYSTEM — CRITICAL: every primitive is CENTRED on the origin.
  Box(120, 80, 40) spans X −60..+60, Y −40..+40, Z −20..+20 — it does NOT start at 0.
  So the TOP face is at z = +height/2 (here +20), the bottom at −height/2 (here −20).
  Place every feature in these centred coordinates. A subtracted feature MUST overlap the
  solid or it removes nothing (a hole at z=35 on a box that ends at z=20 cuts NOTHING).

  VERTICAL THROUGH-HOLES (holes through the top/bottom face) — follow this EXACTLY:
    subtract  Pos(x, y, 0) * Cylinder(radius=r, height=BIG)
    • Z is 0 (centred), NOT the top-face height. Do NOT offset the cylinder up to the top.
    • height = at least 2× the part thickness, so it pierces the FULL thickness.
    • ONLY x and y position the hole (e.g. corners of a 100x60 part are near x=±45, y=±25).
  A hole at "z = height" is the #1 mistake — it makes a shallow blind hole or misses entirely.

PRIMITIVES (each centred at the origin):
  Box(length, width, height)   Cylinder(radius, height)   Sphere(radius)
  Cone(bottom_radius, top_radius, height)              # Cylinder axis is along +Z

POSITION / ROTATE by multiplying with a location (do NOT call .offset on a shape):
  Pos(x, y, z) * Cylinder(radius=6, height=40)         # translate
  Rotation(0, 90, 0) * Cylinder(radius=4, height=20)   # rotate, degrees about X,Y,Z

COMBINE:   a + b  (union)    a - b  (cut a hole)    a & b  (intersect)

ASSEMBLY (ONLY when the brief notes say "assembly" or expected has multiple bodies): build each
part as its OWN labelled solid, position it where it mates, and combine them into a Compound —
do NOT fuse distinct parts with '+'. Pattern:
  base = Box(60, 40, 10); base.label = "base"
  lid  = Pos(0, 0, 11) * (Box(60, 40, 6) - Pos(0, 0, 0) * Cylinder(radius=5, height=20))
  lid.label = "lid"
  result = Compound(children=[base, lid]); result.label = "assembly"
Position parts so they actually mate (a lid sits ON the base's top face; a shaft passes THROUGH a
bore). For a normal SINGLE part, ignore this and assign one fused solid to `result` as usual.

FILLET / CHAMFER operate on a solid's edges:
  result = fillet(result.edges().filter_by(Axis.Z), radius=3)        # vertical edges
  result = chamfer(result.edges().group_by(Axis.Z)[-1], length=2)    # top edges
A fillet/chamfer is COSMETIC — it must never crash the whole part. OCCT raises
"no suitable edges" if the selection is empty/already-filleted, or the radius is too big for the
geometry. So ALWAYS guard them and keep the unfilleted solid if it fails:
  try:
      result = fillet(result.edges().filter_by(Axis.Z), radius=3)
  except Exception:
      pass   # keep the solid; a missing cosmetic fillet beats a failed build
Keep radii small relative to the part (radius < half the thinnest adjacent wall), and select a
SPECIFIC edge set (filter_by / group_by) — never fillet result.edges() blindly.

Cylinder/Cone take keyword args: Cylinder(radius=R, height=H) — NOT r=/h=.
A hole/bore DIAMETER D means radius = D/2 (a 4mm-diameter hole → radius=2). An M3 bolt clearance
hole is radius≈1.75.

If the brief notes recommend a domain helper (e.g. "Use structural_section(...)"), you MUST call
that helper — never hand-build an I/H/C-beam, gear, or bolt from boxes/cylinders:
  structural_section(d_mm, bf_mm, tf_mm, tw_mm, length_mm, mitre=False)
  spur_gear(teeth, module_mm, width_mm, bore_mm=0, pressure_angle=20)
  hex_bolt(size_mm, length_mm, pitch_mm=None)

FEATURE HELPERS — PREFER THESE over hand-rolled geometry; they are verified-correct and compose
with algebra mode. Import what you use: `from b123d.domain import countersink_cutter, counterbore_cutter, bolt_circle, gusset, cross_bore, ring_groove`
  countersink_cutter(through_d, head_d, depth)        # SUBTRACT: countersunk (flat-head) hole
  counterbore_cutter(through_d, bore_d, bore_depth, depth)  # SUBTRACT: counterbored (socket-head) hole
  bolt_circle(count, bolt_circle_d, hole_d, depth)    # SUBTRACT: N holes equally spaced on a bolt circle (centred on origin)
  cross_bore(diameter, length, axis='x')              # SUBTRACT: a RADIAL hole through the SIDE wall (axis horizontal)
  ring_groove(part_radius, depth, width)              # SUBTRACT: a shallow circumferential surface groove (axis +Z)
  gusset(leg_h, leg_v, thickness)                     # ADD: right-triangle corner brace (legs +X and +Z)
  • Cutters open at local z=0 and cut DOWNWARD — place with Pos(x, y, top_z) and use depth ≥ part thickness:
      result -= Pos(25, 0, 5) * countersink_cutter(6, 12, 20)        # countersunk M6 in a part whose top is z=5
      result -= bolt_circle(6, 60, 6, 20)                            # 6-hole bolt circle (flange/hub), centred
  • gusset adds a brace; rotate/position it onto the inner corner, e.g. result += Pos(20, 0, 0) * gusset(25, 40, 8)
  Use bolt_circle for ANY ring of equally-spaced holes (flanges, hubs, engine cylinders); use the
  countersink/counterbore cutters for any chamfered/recessed screw hole; use gusset to brace L/T corners.

RADIAL / SIDE / CROSS BORES — a hole through the SIDE WALL (perpendicular to a cylinder's axis): a
wrist-pin bore through a piston/cylinder SKIRT, a cross-hole in a shaft, a port through a tube wall.
A plain Cylinder() points up +Z and would bore the TOP/BOTTOM — WRONG for a side hole. ALWAYS use
cross_bore (or an explicit Rotation), positioned at the bore's HEIGHT:
    result -= Pos(0, 0, 22) * cross_bore(20, 200)        # Ø20 RADIAL bore through the side at z=22
  Make length ≥ the part's full width so it pierces both walls. NEVER use a bare vertical Cylinder
  for a hole the spec says goes through the side / skirt / wall.

RING GROOVES — shallow circumferential grooves on a cylinder's OUTER surface (piston rings, snap-ring
/ circlip, O-ring glands). Each is a SHALLOW surface cut, NOT a deep bore. Subtracting a plain
Cylinder of nearly the body radius hollows the whole part (the #1 groove mistake). Use ring_groove,
one per groove height:
    body = Pos(0, 0, 30) * Cylinder(radius=40, height=60)   # base at z=0, top at z=60
    for z in (40, 47, 54):
        body -= Pos(0, 0, z) * ring_groove(40, 3, 3)        # three 3mm-deep x 3mm-wide surface grooves
    result = body

EXAMPLES (study the style, then write ONE script):

# 60x40x25 box with a 12mm hole through the top
from build123d import *
result = Box(60, 40, 25) - Cylinder(radius=6, height=30)

# 100x60x20 enclosure, 2mm walls, open top
from build123d import *
result = Box(100, 60, 20) - Pos(0, 0, 2) * Box(96, 56, 20)

# 80x50x5 bracket with two M4 clearance holes
from build123d import *
plate = Box(80, 50, 5)
result = plate - Pos(-25, 0, 0) * Cylinder(radius=2.25, height=10) \\
               - Pos( 25, 0, 0) * Cylinder(radius=2.25, height=10)

# 100x60x30 box with four 4mm holes THROUGH the top face, near the corners
# (centred coords: corners ≈ x=±45, y=±25; cutters at z=0, height>thickness → fully through)
from build123d import *
result = Box(100, 60, 30)
for x in (-45, 45):
    for y in (-25, 25):
        result -= Pos(x, y, 0) * Cylinder(radius=2, height=60)

# W200x100 I-beam, 1500mm — use the domain helper, don't hand-build the section
from build123d import *
from b123d.domain import structural_section
result = structural_section(200, 100, 10, 6, 1500)"""


_HELPER_RE    = re.compile(r"^(structural_section|spur_gear|hex_bolt)\s*\(.*\)\s*$")
# bd_warehouse (gumyr) correct-by-construction parts — see b123d/warehouse.py.
_WH_HELPER_RE = re.compile(r"^(gear|screw|nut|iso_thread|ball_bearing|pipe_flange)\s*\(.*\)\s*$")
_WH_IMPORT    = "from b123d.warehouse import gear, screw, nut, iso_thread, ball_bearing, pipe_flange\n"

def generate_code(brief: dict) -> str:
    # If the brief decided this is a standard section/gear/bolt/fastener, execute that decision
    # deterministically instead of hoping the coder model copies it from prose.
    helper = (brief.get("helper") or "").strip().rstrip(".")
    if _HELPER_RE.match(helper):
        log.info("[v4] Brief selected domain helper: %s", helper)
        return ("from build123d import *\n"
                "from b123d.domain import structural_section, spur_gear, hex_bolt\n"
                f"result = {helper}\n")
    if _WH_HELPER_RE.match(helper):
        log.info("[v4] Brief selected bd_warehouse helper: %s", helper)
        return ("from build123d import *\n" + _WH_IMPORT + f"result = {helper}\n")

    dim_str   = json.dumps(brief.get("dimensions", {}), indent=2)
    feat_str  = "\n".join(f"- {f}" for f in brief.get("features", []))
    notes_str = "\n".join(f"- {n}" for n in brief.get("notes", []))
    prompt = (
        f"Part: {brief.get('description', brief.get('name', ''))}\n"
        f"Dimensions (mm):\n{dim_str}\n"
        + (f"Features:\n{feat_str}\n" if feat_str else "")
        + (f"Notes:\n{notes_str}\n" if notes_str else "")
        + "\nWrite the build123d code:"
    )
    raw = _ollama(_code_model(), _CODE_SYSTEM, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    return _patch_code(_strip_fences(raw))

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```python\s*\n?", "", text)
    text = re.sub(r"^```\s*\n?",       "", text)
    return text.rstrip("`").strip()

# Common LLM hallucinations of the build123d API that are trivially fixable
_CODE_PATCHES = [
    (r"\bMode\.SUBTRACTION\b",  "Mode.SUBTRACT"),
    (r"\bMode\.REMOVE\b",       "Mode.SUBTRACT"),
    (r"\bMode\.CUT\b",          "Mode.SUBTRACT"),
    (r"\bMode\.DELETE\b",       "Mode.SUBTRACT"),
    (r"\.at_origin\(\)",        ""),                 # Plane.XY.at_origin() → Plane.XY
    (r"\.offset\(\s*z\s*=",     ".offset("),         # .offset(z=12) → .offset(12)
    (r"\.offset\(\s*amount\s*=", ".offset("),        # .offset(amount=12) → .offset(12)
    (r"\b(Cylinder|Cone)\(\s*r\s*=", r"\1(radius="),  # Cylinder(r= → Cylinder(radius=
    (r"\bSphere\(\s*r\s*=",     "Sphere(radius="),   # Sphere(r= → Sphere(radius=
    (r"(\bCylinder\([^)]*?),\s*h\s*=", r"\1, height="),  # Cylinder(..., h= → height=
    (r"^\s*\w+\.export_step\(.*$", "# export_step removed — the runner handles export"),
]

def _patch_code(code: str) -> str:
    for pattern, replacement in _CODE_PATCHES:
        code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
    return code

def _check_syntax(code: str) -> Optional[str]:
    """Return None if the code parses, else a short SyntaxError description."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"

# ── Revise / decide ───────────────────────────────────────────────────────────

# Correct-API reference shared by the repair + decide prompts so the model fixes
# AttributeError/TypeError hallucinations against the real build123d API.
_B123D_REF = """\
Correct build123d API — ALGEBRA MODE (build123d 0.10):
  Primitives (centred at origin): Box(l,w,h)  Cylinder(radius,height)  Sphere(r)  Cone(rb,rt,h)
  Position: Pos(x,y,z) * shape    Rotate: Rotation(rx,ry,rz) * shape   (degrees)
  Combine:  a + b (union)   a - b (cut)   a & b (intersect)
  Fillet/chamfer: result = fillet(result.edges().filter_by(Axis.Z), radius=r)
  Vertical through-hole: stock - Pos(x, y, 0) * Cylinder(radius=r, height=2*thickness).
    Z must be 0 (centred), NOT the top-face height; only x,y position it. A cutter offset up
    to z=height makes a shallow/missing hole — the most common mistake.
  Domain helpers: structural_section(d,bf,tf,tw,length), spur_gear(teeth,module,width,bore=0), hex_bolt(size,length).
  Do NOT use: Plane.at_origin(), Rectangle.offset(), or .offset() on a 2D shape — none exist.
  Position solids with Pos(...)*shape, not by offsetting sketches."""

_REVISE_SYSTEM = """\
You are debugging build123d Python code. Fix the problem described — if it is an
AttributeError or TypeError, you used an API that does not exist; use the correct one below.
Return ONLY the corrected, complete Python script — no explanation, no markdown fences.

""" + _B123D_REF

def revise_script(spec: str, code: str, problem: str, state: str = "") -> str:
    """Used when the script failed to parse / run / produce valid geometry."""
    state_block = f"\nGeometry report:\n{state}\n" if state else ""
    prompt = (
        f"Target part: {spec}\n\n"
        f"Problem to fix:\n{problem}\n"
        f"{state_block}\n"
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Return the complete fixed script:"
    )
    raw = _ollama(_code_model(), _REVISE_SYSTEM, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    return _patch_code(_strip_fences(raw))


_DECIDE_SYSTEM = f"""\
You are reviewing a build123d part you generated, against the user's request.
You are given the code, a numeric geometry report, and a visual critique of a render with
TWO views (an isometric view and a top-down view) produced by a vision model.

Trust the visual critique for OVERALL SHAPE, PROPORTIONS, and whether expected FEATURES
are present — including top-face holes/pockets, which are visible in the top-down view. But
it cannot see underside/hidden faces or internal cavities, and may miss whether a hole is
through vs blind — for those you must VERIFY THE CODE yourself. Do NOT reject a feature that
the geometry report and code confirm is present just because the critique overlooked it.

BE STRICT. Before approving, check EVERY requested feature against the code:
- Is each hole/cutout/fillet actually in the code, and does it OVERLAP the solid?
  Remember primitives are CENTRED at the origin: Box(L,W,H) spans z −H/2..+H/2. A cut placed
  outside that range (e.g. a hole at z=35 on a box ending at z=20) is a SILENT NO-OP — reject it.
- Do the dimensions in the code match the request?
- Does the geometry report's volume look consistent with the features (e.g. holes should reduce it)?
If ANY requested feature is missing, mis-placed, or a no-op, the part is WRONG — fix it.

Only if every requested feature is genuinely present and correctly placed, reply with EXACTLY:
{DONE_SENTINEL}
Otherwise return the COMPLETE corrected build123d script (no explanation, no fences).

{_B123D_REF}"""

def decide_or_edit(spec: str, code: str, state: str,
                   warnings: list[str], critique: Optional[str]) -> tuple[str, Optional[str]]:
    """Returns ('done', None) or ('edit', new_code)."""
    warn_block = ("\nWarnings:\n" + "\n".join(warnings)) if warnings else ""
    crit_block = critique if critique else "(visual critique unavailable — judge from code + geometry report)"
    prompt = (
        f"User request: {spec}\n\n"
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Geometry report:\n{state}{warn_block}\n\n"
        f"Visual critique:\n{crit_block}\n\n"
        f"Reply {DONE_SENTINEL} if correct, otherwise return the corrected full script:"
    )
    raw = _ollama(_code_model(), _DECIDE_SYSTEM, prompt, timeout=CODE_TIMEOUT, temperature=0.15)
    # A 'done' verdict is the sentinel, possibly with a short remark; a real edit contains
    # code. Detect code by a fence or an actual import STATEMENT — the old substring test
    # ("import" not in raw) misread prose like "all important dimensions match" as an edit.
    has_code = "```" in raw or re.search(r"^\s*(?:from|import)\s+\w+", raw, re.M)
    if DONE_SENTINEL in raw and not has_code:
        return "done", None
    return "edit", _patch_code(_strip_fences(raw))

# ── build123d runner ──────────────────────────────────────────────────────────

def run_step(code: str, work_dir: Path) -> tuple[Path, str]:
    """Write code to work_dir/build_source.py, run scripts/step → STEP. Returns (step_path, log)."""
    sys_path_inject = f"import sys as _sys\n_sys.path.insert(0, {str(_HERE)!r})\n\n"
    src  = work_dir / "build_source.py"
    step = work_dir / "build_output.step"
    src.write_text(sys_path_inject + code)
    # Remove any STEP from a previous turn so a failed run can't be mistaken for success.
    try:
        step.unlink()
    except FileNotFoundError:
        pass

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "step"), str(src), str(step)],
        capture_output=True, text=True, timeout=STEP_TIMEOUT,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or not step.exists():
        raise RuntimeError(output or "scripts/step exited non-zero with no output")
    return step, output

def run_inspect(step_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "inspect"), str(step_path)],
        capture_output=True, text=True, timeout=INSPECT_TIMEOUT,
    )
    output   = result.stdout + result.stderr
    valid    = result.returncode == 0
    errors   = [ln for ln in output.splitlines() if ln.startswith("ERROR")]
    warnings = [ln for ln in output.splitlines() if ln.startswith("WARNING")]
    return {"valid": valid, "output": output, "errors": errors, "warnings": warnings}

def run_diff(old_step: Path, new_step: Path) -> str:
    """Deterministic before→after geometry delta (what an edit changed): ΔSolids/Faces/Cyl/Volume/
    Bbox. Lets the loop confirm an EDIT did the intended thing (e.g. 'add a hole' ⇒ ΔCyl +1,
    ΔVolume < 0) instead of trusting the vision critic to notice. Returns '' on failure."""
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "inspect"), str(new_step), "--diff", str(old_step)],
            capture_output=True, text=True, timeout=INSPECT_TIMEOUT,
        )
        out = (result.stdout or "").strip()
        return "\n".join(ln for ln in out.splitlines() if ln.startswith(("Δ", "DIFF")))
    except Exception as e:
        log.warning("[v4] diff failed: %s", e)
        return ""

# ── Deterministic verification gate ────────────────────────────────────────────
# Discipline borrowed from earthtojake/text-to-cad (MIT): assert geometry FACTS against an
# expected target instead of trusting a vision model to "see" features. Holes that silently
# missed the solid, unfused plates, and gross dimensional errors are caught here, deterministically.

_FACT_INT_RES = {
    "solids":    re.compile(r"^Solids:\s*(\d+)",            re.M),
    "faces":     re.compile(r"^Faces:\s*(\d+)",             re.M),
    "cyl_faces": re.compile(r"^Cylindrical faces:\s*(-?\d+)", re.M),
    "cone_faces":re.compile(r"^Conical faces:\s*(-?\d+)",   re.M),
    "through_holes": re.compile(r"^Through-holes:\s*(-?\d+)", re.M),
    "blind_holes":   re.compile(r"^Blind-holes:\s*(-?\d+)",   re.M),
}
_FACT_VOL_RE  = re.compile(r"^Volume:\s*([\d.]+)", re.M)
_FACT_BBOX_RE = re.compile(r"^Bbox:\s*([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)", re.M)

def parse_facts(inspect_output: str) -> dict:
    """Pull the structured geometry facts out of scripts/inspect's report. Prefers the
    FACTS_JSON machine block (exact, wording-independent); falls back to the regex scrape
    of the human-readable lines for older inspect outputs."""
    m = re.search(r"^FACTS_JSON:\s*(\{.*\})\s*$", inspect_output, re.M)
    if m:
        try:
            data = json.loads(m.group(1))
            data["bore_axes"] = [(float(d), str(o)) for d, o in data.get("bore_axes", [])]
            return data
        except Exception as e:
            log.warning("[v4] FACTS_JSON parse failed (%s) — using regex fallback.", e)
    facts: dict = {}
    for key, rx in _FACT_INT_RES.items():
        m = rx.search(inspect_output)
        if m:
            facts[key] = int(m.group(1))
    m = _FACT_VOL_RE.search(inspect_output)
    if m:
        facts["volume"] = float(m.group(1))
    m = _FACT_BBOX_RE.search(inspect_output)
    if m:
        facts["bbox"] = [float(m.group(i)) for i in (1, 2, 3)]
    # Feature dimensions (measure): bore diameters + wall-thickness estimates.
    m = re.search(r"^Bore diameters:\s*([\d.,\s]+?)\s*mm", inspect_output, re.M)
    if m:
        facts["bores"] = [float(x) for x in re.findall(r"[\d.]+", m.group(1))]
    m = re.search(r"^Wall candidates \(est\):\s*([\d.,\s]+?)\s*mm", inspect_output, re.M)
    if m:
        facts["walls"] = [float(x) for x in re.findall(r"[\d.]+", m.group(1))]
    # Bore orientation (radial/axial) — drives the general geometry-matches-intent gate.
    m = re.search(r"^Bore axes:\s*(.+)$", inspect_output, re.M)
    if m:
        axes = []
        for seg in m.group(1).split(";"):
            mm = re.search(r"Ø([\d.]+)\s+(radial|axial|ambiguous)", seg)
            if mm:
                axes.append((float(mm.group(1)), mm.group(2)))
        facts["bore_axes"] = axes
    return facts

_RADIAL_TERMS = re.compile(r"\b(radial(?:ly)?|cross[- ]?(?:hole|bore|drill)|side (?:hole|bore)|"
                           r"through (?:the )?(?:side|wall|skirt))\b", re.I)
_AXIAL_TERMS = re.compile(r"\b(axial(?:ly)?|along (?:the|its) (?:long )?axis|central(?:ly)? "
                          r"(?:bored|drilled)|down (?:the|its) (?:centre|center|length))\b", re.I)

def _spec_states_orientation(spec: str, orient: str) -> bool:
    """Does the spec TEXT actually claim this bore orientation? The brief labels orientations
    and gets them wrong (a vertical blind standoff hole labelled 'radial' hard-failed every
    turn AND told the coder to bore sideways). Only a user-stated orientation may steer."""
    rx = _RADIAL_TERMS if orient == "radial" else _AXIAL_TERMS
    return bool(rx.search(spec or ""))

def _spec_mentions_dim(spec: str, d: float) -> bool:
    """Does the spec text corroborate dimension d (mm, as diameter or radius)? Guards the
    gate against brief-hallucinated feature dims: only a number the user actually wrote may
    hard-fail a build — an invented bore can never be satisfied and burns every turn."""
    for m in re.findall(r"\d+(?:\.\d+)?", spec or ""):
        v = float(m)
        if abs(v - d) < 0.05 or abs(v * 2 - d) < 0.05 or abs(v / 2 - d) < 0.05:
            return True
    return False

def verify_expected(facts: dict, expected: dict, spec: str = "") -> tuple[list[str], list[str]]:
    """Sanity-check the produced geometry. Returns (hard_fails, advisories).

    DESIGN (2026-06-26): the gate's job is to confirm the code produced a VALID, non-degenerate,
    single fused solid — a universal check that holds for any part type. It is NOT a spec-matcher.
    The brief (qwen3:8b) guesses the `expected` bbox / hole-counts / through-vs-blind policy and is
    frequently wrong for anything past a plate (it read "80mm bore, 60mm tall" as an 80×60×60 box,
    and condemned a piston's ring grooves as illegal blind holes). Enforcing those guesses as hard
    failures rejected correct parts AND short-circuited the visual critic (the loop skips the critic
    on any hard fail). So those spec-target checks are now ADVISORIES: surfaced to the coder and the
    visual critic, but they do not block convergence. Feature/dimension judgement belongs to the
    visual critic and the human, not to a guessed checklist.

    HARD (universal structural integrity, type-independent):
      • produced no solid at all
      • fragmented into multiple disconnected bodies when a single part was expected (unfused plates)
    ADVISORY (spec-derived, brief-fallible — guide, don't block):
      • overall size differs from the brief's bbox guess
      • fewer holes/cylindrical faces than the brief counted
      • a hole came out blind where the brief expected it through
    """
    hard: list[str] = []
    soft: list[str] = []
    if not isinstance(expected, dict):
        return hard, soft

    # ── HARD: structural integrity (true for every part type) ──────────────────────────────────
    # No solid produced — the code ran but built nothing usable.
    if isinstance(facts.get("solids"), int) and facts["solids"] == 0:
        hard.append("the build produced no solid body at all — nothing was created or everything "
                    "was cut away. Re-check that the base shape exists and cuts don't remove it.")

    # Fragmented: more bodies than expected. For a single part (expected 1) this means features
    # were placed but never unioned (unfused plates / floating gussets). For an assembly (expected
    # N>1) it means a part fragmented into extra pieces.
    exp_solids = expected.get("solids")
    got_solids = facts.get("solids")
    if (isinstance(exp_solids, int) and exp_solids >= 1 and isinstance(got_solids, int)
            and got_solids > max(exp_solids, 1)):
        if exp_solids <= 1:
            hard.append(
                f"the part has {got_solids} separate bodies but should be one fused solid — "
                f"features are not unioned together (unfused plates or floating gussets/bodies). "
                f"Combine them with '+' so the result is a single watertight solid.")
        else:
            hard.append(
                f"this assembly should have {exp_solids} parts but produced {got_solids} separate "
                f"bodies — a part has fragmented into extra pieces. Make each named part exactly one "
                f"solid, then put them in a Compound.")
    # ADVISORY: an assembly whose distinct parts got fused into TOO FEW bodies (lost separation).
    if (isinstance(exp_solids, int) and exp_solids > 1 and isinstance(got_solids, int)
            and 0 < got_solids < exp_solids):
        soft.append(
            f"this is an assembly of ~{exp_solids} parts but only {got_solids} body/bodies were "
            f"produced — distinct parts may have been fused; keep each part its own labelled solid "
            f"in a Compound (children=[...]), don't '+' them together.")

    # ── ADVISORY: spec-target hints (brief-derived, may be wrong — never block on these) ─────────
    exp_holes = expected.get("min_holes")
    cyl = facts.get("cyl_faces", -1)
    if isinstance(exp_holes, int) and exp_holes > 0 and cyl >= 0 and cyl < exp_holes:
        soft.append(
            f"the spec seems to ask for ~{exp_holes} hole(s)/bore(s) but the solid has only "
            f"{cyl} cylindrical face(s) — if holes are missing, ensure each cut overlaps the solid.")

    exp_through = expected.get("min_through_holes")
    thru  = facts.get("through_holes", -1)
    blind = facts.get("blind_holes", -1)
    if isinstance(exp_through, int) and exp_through > 0 and thru >= 0 and thru < exp_through:
        detail = (f" ({blind} blind)" if isinstance(blind, int) and blind > 0 else "")
        soft.append(
            f"the spec may want {exp_through} hole(s) to pass fully THROUGH but only {thru} do{detail} "
            f"— if a through-hole is required, give the cutter height ≥ 2× the thickness, centred.")

    if expected.get("forbid_blind_holes") and isinstance(blind, int) and blind > 0:
        soft.append(
            f"{blind} hole(s) came out blind — if the spec intended them to pass through (no "
            f"pocket/groove/counterbore was named), deepen and centre the cutter to clear both faces.")

    # Overall size — compared as a sorted set of extents (axis-permutation-invariant). Advisory:
    # the brief routinely mis-derives dimensions, so a mismatch is a hint to the critic, not a block.
    exp_bbox = expected.get("bbox_mm")
    got = facts.get("bbox")
    if (isinstance(exp_bbox, list) and len(exp_bbox) == 3
            and all(isinstance(e, (int, float)) and e > 0 for e in exp_bbox) and got):
        if any(abs(g - e) > max(5.0, 0.10 * e) for e, g in zip(sorted(exp_bbox), sorted(got))):
            soft.append(
                f"overall size differs from the brief's guess: extents are "
                f"{'×'.join(f'{d:.1f}' for d in sorted(got, reverse=True))}mm vs an expected "
                f"{'×'.join(f'{d:.1f}' for d in sorted(exp_bbox, reverse=True))}mm — confirm the "
                f"dimensions against the actual spec (the brief's size guess is often wrong).")

    # Bore/hole diameters — a RELIABLE feature-size check (cylindrical faces measure exactly). If
    # the spec names a target Ø and no cylindrical face of that size exists, the hole is missing or
    # mis-sized. Advisory (the brief may mis-read a diameter), but a strong signal.
    exp_bores = expected.get("bores_mm")
    got_bores = facts.get("bores")
    if isinstance(exp_bores, list) and got_bores is not None:
        missing = [d for d in exp_bores if isinstance(d, (int, float)) and d > 0
                   and not any(abs(d - g) <= max(0.6, 0.05 * d) for g in got_bores)]
        if missing:
            seen = ', '.join(f'{g:g}' for g in sorted({round(x, 1) for x in got_bores})) or "none"
            soft.append(
                f"the spec asks for bore/hole diameter(s) {', '.join(f'{d:g}' for d in missing)}mm "
                f"but no cylindrical face of that size was found (measured Ø: {seen}mm) — check the "
                f"hole radius and that each cut actually overlaps the solid.")

    # Wall thickness — NOISY estimate, advisory only. Fires only when the spec names a wall and NO
    # candidate is near it; noise only ADDS candidates, so this won't false-alarm on a correct wall.
    exp_wall = expected.get("wall_mm")
    got_walls = facts.get("walls")
    if isinstance(exp_wall, (int, float)) and exp_wall > 0 and got_walls:
        if not any(abs(exp_wall - w) <= max(0.6, 0.25 * exp_wall) for w in got_walls):
            cand = ', '.join(f'{w:g}' for w in sorted({round(x, 1) for x in got_walls}))
            soft.append(
                f"the spec implies a ~{exp_wall:g}mm wall but no wall near that thickness was "
                f"measured (candidates: {cand}mm) — if the walls look too thick/thin, adjust the "
                f"shell offset or cavity size.")

    # ── GENERAL geometry-matches-intent gate ────────────────────────────────────────────────────
    # One generic loop over the brief's structured `feature_checks` — NO part-specific code. The
    # same rule verifies a piston's wrist-pin bore, a shaft's cross-hole, or a manifold port: does
    # the geometry exhibit each intended feature? A bore with the wrong ORIENTATION (a side bore
    # that came out axial) is a deterministic, reliable miss → HARD fail, forcing an edit. Count
    # checks (grooves, bolt circles) are advisories (exact counts are harder to measure).
    checks = expected.get("feature_checks")
    axes = facts.get("bore_axes") or []          # [(diameter, "radial"|"axial"), ...]
    if isinstance(checks, list):
        for ch in checks:
            if not isinstance(ch, dict):
                continue
            kind = ch.get("kind")
            if kind == "bore":
                d = ch.get("d_mm")
                orient = ch.get("orientation")
                if not isinstance(d, (int, float)) or d <= 0:
                    continue
                # Only a spec-corroborated dimension may HARD-fail; a hallucinated feature
                # check must guide, not block (see _spec_mentions_dim).
                corroborated = _spec_mentions_dim(spec, d)
                tol = max(0.6, 0.06 * d)
                got = [o for (dd, o) in axes if abs(dd - d) <= tol]
                # The brief's orientation label only counts when the SPEC states it —
                # otherwise check presence only and never steer the coder directionally.
                if orient in ("radial", "axial") and not _spec_states_orientation(spec, orient):
                    orient = None
                if not got:
                    # Distinguish "measured absent" from "couldn't measure": cylinders exist
                    # but none were orientation-classified → the detector is blind here, and
                    # absence of evidence must not hard-fail the build.
                    detector_blind = (not axes and facts.get("cyl_faces", 0) != 0)
                    msg = (f"the spec needs a {orient + ' ' if orient in ('radial', 'axial') else ''}"
                           f"Ø{d:g}mm bore but no bore of that size was measured — add it "
                           f"(cut a Ø{d:g}mm cylinder into the part where the spec places it).")
                    if corroborated and not detector_blind:
                        hard.append(msg)
                    else:
                        soft.append(msg)
                    continue
                if orient in ("radial", "axial") and orient not in got:
                    if "ambiguous" in got:
                        soft.append(
                            f"the Ø{d:g}mm bore's axis is tilted/ambiguous — confirm it runs "
                            f"{orient} as the spec requires.")
                        continue
                    bucket = hard if corroborated else soft
                    have = "/".join(sorted(set(got)))
                    if orient == "radial":
                        bucket.append(
                            f"the Ø{d:g}mm bore must be RADIAL (through the SIDE wall) but it came "
                            f"out {have} (through the top/bottom). Use cross_bore({d:g}, <length>) — "
                            f"a plain Cylinder points up +Z and bores the wrong way.")
                    else:
                        bucket.append(
                            f"the Ø{d:g}mm bore must be AXIAL (along the part's long axis) but it "
                            f"came out {have}. Orient the cutting cylinder along that axis.")
            elif kind == "groove":
                n = ch.get("count")
                blind = facts.get("blind_holes", -1)
                if isinstance(n, int) and n > 0 and isinstance(blind, int) and 0 <= blind < n:
                    soft.append(
                        f"the spec asks for {n} ring groove(s) but only ~{blind} shallow "
                        f"circumferential cut(s) were detected — ring grooves are SHALLOW surface "
                        f"cuts (use ring_groove(part_r, depth, width) at each height), not one deep "
                        f"bore that hollows the part.")
            elif kind == "bolt_circle":
                n = ch.get("count")
                d = ch.get("d_mm")
                cyl = facts.get("cyl_faces", -1)
                if isinstance(n, int) and n > 0 and isinstance(cyl, int) and 0 <= cyl < n:
                    soft.append(
                        f"the spec asks for a {n}-hole bolt circle but only {cyl} cylindrical "
                        f"face(s) exist — use bolt_circle({n}, <circle Ø>, "
                        f"{d if d else '<hole Ø>'}, <depth>) so the holes are equally spaced.")
    return hard, soft

def run_render(step_path: Path, work_dir: Path, section: bool = False) -> Path:
    png = work_dir / "render.png"
    cmd = [sys.executable, str(SCRIPTS_DIR / "render"), str(step_path), str(png)]
    if section:
        cmd.append("--section")   # add a 3rd cut-through panel for hollow/internal parts
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=RENDER_TIMEOUT)
    if result.returncode != 0 or not png.exists():
        raise RuntimeError(result.stdout + result.stderr)
    return png

def run_stl(step_path: Path, out_path: Path) -> Path:
    """Export STEP → watertight binary STL (sliceable for printing). Returns the STL path."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "stl"), str(step_path), str(out_path)],
        capture_output=True, text=True, timeout=STL_TIMEOUT,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(result.stdout + result.stderr)
    return out_path

def run_dxf(step_path: Path, out_path: Path) -> tuple[Path, str]:
    """Export STEP → 2D flat-pattern DXF (laser/sheet). Returns (path, summary). Raises on no
    applicable flat face — caller treats that as 'not a sheet part', not a hard error."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "dxf"), str(step_path), str(out_path)],
        capture_output=True, text=True, timeout=STL_TIMEOUT,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(out.strip() or "scripts/dxf produced no file")
    summary = next((ln for ln in out.splitlines() if ln.startswith("Flat pattern:")), "")
    return out_path, summary

# ── Visual critic (graceful) ────────────────────────────────────────────────--

_QUESTIONS_SYSTEM = """\
You are a CAD verification planner. From the part request, write 3-6 short YES/NO questions a
reviewer can answer by LOOKING at renders of the finished part (isometric + top-down + optional
cross-section). Each question must check one requested feature or proportion, be answerable
visually, and be phrased so YES means correct. Do not ask about exact dimensions (cameras can't
measure) — ask about presence, count, placement, and proportion.
Examples: "Are there four holes near the corners of the top face?" · "Is the box hollow with an
open top?" · "Does the vertical plate stand perpendicular to the base plate?"
Reply ONLY JSON: {"questions": ["...", "..."]}"""

_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"},
                                 "minItems": 3, "maxItems": 6}},
    "required": ["questions"],
}

def verify_questions(spec: str, brief: dict) -> list[str]:
    """CADCodeVerify-style: binary verification questions generated ONCE per build, answered
    by the critic against the renders. Best-effort — [] falls back to free-form critique."""
    try:
        raw = _ollama(BRIEF_MODEL, _QUESTIONS_SYSTEM,
                      f"Part request: {spec}\nFeatures: {brief.get('features', [])}",
                      timeout=OLLAMA_TIMEOUT, temperature=0.2, fmt=_QUESTIONS_SCHEMA)
        qs = (_extract_json(raw) or {}).get("questions") or []
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:6]
        # Post-filter dimension questions the prompt forbids but small models still emit:
        # a camera cannot verify "2mm" — such questions only ever answer UNCLEAR (noise).
        qs = [q for q in qs if not re.search(r"\d+(?:\.\d+)?\s*mm", q, re.I)]
        if qs:
            log.info("[v4] %d verification questions: %s", len(qs), " | ".join(qs)[:200])
        return qs
    except Exception as e:
        log.warning("[v4] Question generation failed (%s) — free-form critique.", e)
        return []

_ANSWERS_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"question": {"type": "string"},
                       "answer": {"type": "string", "enum": ["yes", "no", "unclear"]},
                       "evidence": {"type": "string"}},
        "required": ["question", "answer"]}}},
    "required": ["answers"],
}

_CRITIC_QA_SYSTEM = (
    "You are a CAD reviewer looking at a render with TWO panels — LEFT isometric (overall "
    "shape), RIGHT top-down looking straight down -Z (top-face features show here as "
    "dots/circles even when edge-on invisible in the isometric) — and possibly a THIRD "
    "warm/orange panel: a real cross-section exposing walls, cavities, and bore depths. "
    "Answer each verification question strictly from what the panels show: 'yes' only when "
    "the render clearly confirms it, 'no' when it clearly contradicts it, 'unclear' when the "
    "views cannot tell. One short evidence phrase each. Reply ONLY JSON."
)

_CRITIC_SYSTEM = (
    "You are a CAD reviewer looking at a render of a 3D part that has TWO panels: a LEFT "
    "isometric view (for overall shape and proportions) and a RIGHT top-down view looking "
    "straight down the −Z axis (for features on the top face). ALWAYS inspect BOTH panels. "
    "Holes, pockets, and cutouts in the top face are usually invisible edge-on in the "
    "isometric view but show clearly in the top-down view — look there before concluding a "
    "top-face feature is missing (e.g. small corner holes appear as dots/circles in the "
    "top-down panel). "
    "A THIRD 'section' panel may be present (warm/orange): it is a real cross-section, the part "
    "cut through its middle, exposing internal structure. When present, use it to judge wall "
    "thickness, hollowness, cavities, internal bosses/ribs, and how deep bores/pockets actually go "
    "— these are otherwise invisible from outside. "
    "Describe what you see in 1-2 sentences, then state whether it matches the requested "
    "part and what (if anything) looks wrong with the overall shape, proportions, or "
    "missing/extra features. Be concise and concrete."
)

# Specs whose correctness lives INSIDE the part — render a section panel so the critic can see in.
_INTERNAL_FEATURE_TERMS = re.compile(
    r"\b(hollow|cavity|cavities|enclosure|housing|case|casing|container|tray|bin|shell|shelled|"
    r"wall(?:s|ed)?|tube|tubular|pipe|cylinder head|bore|counterbore|pocket|recess(?:ed)?|"
    r"internal|inside|interior|lid|open[- ]?top|piston|cup|vessel|tank)\b", re.IGNORECASE)

def wants_section(spec: str, brief: dict) -> bool:
    """True when the part's correctness is internal (hollow/walled/bored) → render a section view."""
    if _INTERNAL_FEATURE_TERMS.search(spec or ""):
        return True
    hay = " ".join(str(x) for x in (brief.get("features", []) + brief.get("notes", [])))
    if _INTERNAL_FEATURE_TERMS.search(hay):
        return True
    exp = brief.get("expected", {})
    return isinstance(exp, dict) and isinstance(exp.get("wall_mm"), (int, float))

def visual_critique(step_path: Path, spec: str, state: str, work_dir: Path,
                    section: bool = False, questions: Optional[list[str]] = None) -> Optional[str]:
    """Render the part and ask the multimodal model to judge it. With `questions`, runs the
    CADCodeVerify-style pass: the critic answers each binary question from the panels and the
    verdict is assembled deterministically (measured +7.3% geometric accuracy over free-form
    critique in the paper). Falls back to free-form critique without questions, and returns
    None on any failure so the loop degrades gracefully to numeric-state-only."""
    try:
        png = run_render(step_path, work_dir, section=section)
        img_b64 = base64.b64encode(png.read_bytes()).decode()
        if questions:
            try:
                qlist = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                raw = _ollama(CRITIC_MODEL, _CRITIC_QA_SYSTEM,
                              f"Requested part: {spec}\n\nGeometry facts (authoritative for "
                              f"hidden/through features):\n{state}\n\n"
                              f"Verification questions:\n{qlist}\n\nAnswer each:",
                              timeout=CRITIC_TIMEOUT, images=[img_b64], fmt=_ANSWERS_SCHEMA)
                answers = (_extract_json(raw) or {}).get("answers") or []
                if answers:
                    # Only a hard NO blocks: UNCLEAR means the views can't tell, and
                    # punishing that false-blocked correct parts in live testing.
                    noes = [a for a in answers if str(a.get("answer", "")).lower() == "no"]
                    unclear = [a for a in answers
                               if str(a.get("answer", "")).lower() == "unclear"]
                    if not noes:
                        note = (" (unverifiable from renders: "
                                + "; ".join(a.get("question", "")[:50] for a in unclear) + ")"
                                if unclear else "")
                        return ("PASS: verification questions confirmed" + note + " — "
                                + "; ".join(a.get("question", "")[:60] for a in answers
                                            if a not in unclear))
                    lines = [f"{a.get('question','?')} -> NO"
                             + (f" ({a.get('evidence','')})" if a.get("evidence") else "")
                             for a in noes]
                    return ("Verification questions FAILED:\n" + "\n".join(lines)
                            + "\nFix these specific issues.")
            except Exception as e:
                log.warning("[v4] QA critique failed (%s) — free-form fallback.", e)
        prompt = (
            f"Requested part: {spec}\n\n"
            f"Geometry facts (authoritative for hidden/through features):\n{state}\n\n"
            f"Critique the render:"
        )
        return _ollama(CRITIC_MODEL, _CRITIC_SYSTEM, prompt,
                       timeout=CRITIC_TIMEOUT, images=[img_b64]).strip()
    except Exception as e:
        log.warning("[v4] Visual critique unavailable: %s", e)
        return None

# ── Onshape STEP import (pluggable upload target) ──────────────────────────────

def _onshape_multipart(path: str, fields: dict, file_path: Path, filename: str) -> dict:
    """POST multipart/form-data (text fields + one file) to Onshape."""
    boundary = "----cadv4boundary"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    ak, sk = _creds()
    creds  = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    req = urllib.request.Request(
        BASE_URL + path, data=body, method="POST",
        headers={"Authorization": f"Basic {creds}",
                 "Accept": "application/json;charset=UTF-8",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def import_step_to_onshape(step_path: Path, name: str,
                           public: bool = False, upload_target=None) -> dict:
    """Upload the STEP and TRANSLATE it into a real Onshape Part Studio (a plain blob
    element is only a downloadable file — not a viewable/editable model).
    `upload_target` is a seam for a future local target (e.g. FreeCAD / file drop):
    a callable(step_path, name) -> result dict. Never reports a fake success."""
    if upload_target is not None:
        return upload_target(step_path, name)

    try:
        doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": public})
    except RuntimeError as e:
        # Free Onshape accounts can only create PUBLIC documents — fall back with a warning
        # rather than failing, but never silently downgrade a paid account's privacy choice.
        if "409" in str(e) and "public" in str(e).lower() and not public:
            log.warning("[v4] Onshape rejected a private document (free account) — "
                        "creating a PUBLIC document instead.")
            doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": True})
            public = True
        else:
            raise
    did = doc["id"]
    wid = doc["defaultWorkspace"]["id"]
    doc_url = f"{BASE_URL}/documents/{did}/w/{wid}"

    # Upload + translate STEP → Part Studio
    try:
        resp = _onshape_multipart(
            f"/api/blobelements/d/{did}/w/{wid}",
            {"translate": "true", "storeInDocument": "true",
             "flattenAssemblies": "false", "yAxisIsUp": "false"},
            step_path, step_path.name)
    except Exception as e:
        raise RuntimeError(f"Onshape upload failed ({e}). Empty document at {doc_url}")

    tid = resp.get("translationId")
    if not tid:
        raise RuntimeError(f"Onshape did not start a STEP translation: {str(resp)[:200]}. "
                           f"Empty document at {doc_url}")

    # Poll the translation job until the Part Studio is ready
    deadline = time.monotonic() + TRANSLATE_TIMEOUT
    state = None
    while time.monotonic() < deadline:
        t = _onshape("GET", f"/api/translations/{tid}")
        state = t.get("requestState")
        if state == "DONE":
            eids = t.get("resultElementIds") or []
            if not eids:
                raise RuntimeError(f"Translation finished with no Part Studio. Document at {doc_url}")
            eid = eids[0]
            # Onshape reports a translation DONE even when the STEP yields ZERO geometry
            # (e.g. a self-intersecting fillet OCCT tolerates but the importer silently drops).
            # A URL to an empty Part Studio is a fake success — verify a real body landed.
            try:
                bd = _onshape("GET", f"/api/partstudios/d/{did}/w/{wid}/e/{eid}/bodydetails")
                n_bodies = len(bd.get("bodies", []))
            except Exception as e:
                n_bodies = -1
                log.warning("[v4] Could not verify imported bodies (%s)", e)
            if n_bodies == 0:
                raise RuntimeError(
                    f"Onshape translation produced an EMPTY Part Studio (0 bodies) — the STEP "
                    f"imported with no geometry. This usually means an invalid feature in the "
                    f"build123d output (commonly a fillet/chamfer on a hole rim or a "
                    f"self-intersecting blend). Fix the geometry and re-export. Document at {doc_url}")
            log.info("[v4] Translated STEP → Part Studio %s (%s body/ies)", eid, n_bodies)
            return {"url": f"{doc_url}/e/{eid}", "did": did, "wid": wid,
                    "eid": eid, "uploaded": True, "public": public, "bodies": n_bodies}
        if state == "FAILED":
            raise RuntimeError(f"Onshape translation failed: {t.get('failureReason')}. "
                               f"STEP stored locally; document at {doc_url}")
        time.sleep(2)
    raise RuntimeError(f"Onshape translation timed out after {TRANSLATE_TIMEOUT}s "
                       f"(last state {state}). Document at {doc_url}")

# ── Onshape URL / document inspect (unchanged from v3) ─────────────────────────

_ONSHAPE_URL_RE = re.compile(
    r"https://cad\.onshape\.com/documents/([a-f0-9]+)/w/([a-f0-9]+)/e/([a-f0-9]+)",
    re.IGNORECASE,
)

def _parse_onshape_url(url: str) -> tuple[str, str, str]:
    m = _ONSHAPE_URL_RE.search(url)
    if not m:
        raise ValueError(f"Not a recognised Onshape URL: {url!r}")
    return m.group(1), m.group(2), m.group(3)

def _describe_document(did: str, wid: str, eid: str) -> str:
    try:
        doc_meta = _onshape("GET", f"/api/v9/documents/{did}")
        doc_name = doc_meta.get("name", "Unknown document")
    except Exception:
        doc_name = "Unknown document"
    try:
        features_resp = _onshape("GET", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features")
        feature_names = [
            f"{f.get('name', '?')} ({f.get('featureType', '?')})"
            for f in features_resp.get("features", []) if not f.get("suppressed")
        ]
    except Exception as e:
        feature_names = [f"(could not fetch features: {e})"]
    try:
        mass_resp = _onshape("GET", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties")
        bodies = mass_resp.get("bodies") or {}
        first = next(iter(bodies.values()), {}) if bodies else {}
        mass_g  = round(first.get("mass", [0])[0] * 1000, 1) if first else None
        vol_cm3 = round(first.get("volume", [0])[0] * 1e6, 2) if first else None
        mass_line = (f"Mass: {mass_g} g, Volume: {vol_cm3} cm³" if mass_g else "No solid bodies.")
    except Exception:
        mass_line = "Mass: unknown"
    feat_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(feature_names)) or "  (none)"
    doc_summary = (f"Document: {doc_name}\nFeatures ({len(feature_names)}):\n{feat_list}\n{mass_line}").strip()
    system = ("You are a CAD assistant. Summarise the Onshape model in 2-4 sentences: "
              "what it appears to be, its key features, and approximate size. Be concise.")
    try:
        return _ollama(BRIEF_MODEL, system, f"Describe this Onshape model:\n\n{doc_summary}",
                       timeout=OLLAMA_TIMEOUT)
    except Exception:
        return doc_summary

# ── Few-shot retrieval (kept: simple keyword overlap over rated builds) ─────────

def _load_fewshots(spec: str, n: int = 2) -> list[dict]:
    """Retrieve the most similar known-good build123d examples for `spec`.
    Semantic (nomic-embed cosine) via cad_retrieval; degrades to word-overlap on any failure."""
    if cad_retrieval is not None:
        try:
            hits = cad_retrieval.retrieve(spec, n=n)
            if hits:
                log.info("[v4] Few-shots (%s): %s", hits[0].get("_how", "?"),
                         ", ".join(f"{h['spec'][:40]}~{h.get('_score','?')}" for h in hits))
            return hits
        except Exception as e:
            log.warning("[v4] Semantic retrieval failed (%s) — falling back to word-overlap.", e)

    # Fallback: legacy word-overlap over the same corpus file.
    if not FEEDBACK_FILE.exists():
        return []
    spec_words = set(re.findall(r'\b[a-z]{3,}\b', spec.lower()))
    rows = []
    for line in FEEDBACK_FILE.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        row_words = set(re.findall(r'\b[a-z]{3,}\b', row.get("spec", "").lower()))
        overlap   = len(spec_words & row_words)
        if overlap >= 1 and row.get("rating", 0) >= 4:
            rows.append((overlap, row))
    rows.sort(key=lambda x: (-x[0], -x[1].get("rating", 0)))
    return [r for _, r in rows[:n]]

# ── Session ───────────────────────────────────────────────────────────────────

def _write_session(data: dict) -> None:
    SESSION_FILE.write_text(json.dumps(data, indent=2))

def _read_session() -> dict:
    try:
        return json.loads(SESSION_FILE.read_text())
    except Exception:
        return {}

# ── Main build — the agentic observe-edit loop ─────────────────────────────────

def build(spec: str, chat_id: Optional[str] = None, coder: str = "auto",
          use_fewshots: bool = True, do_upload: bool = True,
          final_render: bool = True) -> dict:
    """Returns result dict with at least: url, path, spec, has_bodies, build_time_s.

    coder: "auto" (triage the spec + escalate on failure, default), "fast" (force the 7B),
    or "strong" (force the 30B). An explicit cad.code_model pin also disables auto-switching.
    use_fewshots: inject retrieved known-good examples into codegen (set False to A/B the lift).
    do_upload: when False, render the final STEP to a local PNG and skip Onshape (used for
    benchmarking / when no Onshape creds are configured) — result carries render_local, url=""."""
    log.info("=" * 60)
    log.info("[v4.%s] build: %s", VERSION.split('.')[-1], spec)
    preflight()
    t0 = time.monotonic()

    brief    = build_brief(spec)
    reconcile_expected(brief, spec)   # deterministically harden the brief's expected block
    want_section = wants_section(spec, brief)   # render a cut-through panel for hollow/internal parts
    questions = verify_questions(spec, brief)   # B5: binary checks the critic answers per turn
    if want_section:
        log.info("[v4] Section view ON — critic gets a mid-plane cut-through panel.")
    name     = brief.get("name", spec[:40])
    fewshots = [fs for fs in _load_fewshots(spec) if fs.get("code")] if use_fewshots else []
    if fewshots:
        fs_note = (
            "TECHNIQUE REFERENCES — idioms from past *verified* builds, NOT answers to copy.\n"
            "This spec differs from every reference below. Study HOW each one performs its\n"
            "operation (edge selection, boolean, helper call) and reuse that idiom, then write\n"
            "ORIGINAL code for THIS spec. Copying a whole reference verbatim is wrong — the part\n"
            "you are building is different; adapt the technique, not the geometry.\n")
        for fs in fewshots:
            teaches = fs.get("teaches")
            fs_note += "\n"
            fs_note += (f"• Idiom: {teaches}\n" if teaches
                        else f"• From a build of: {fs['spec']}\n")
            fs_note += f"  how it's expressed in build123d:\n{fs['code'][:600]}\n"
        brief["notes"] = brief.get("notes", []) + [fs_note.strip()]

    # Stage B: inject pitfalls learned from past failures on similar parts.
    lessons = (cad_retrieval.retrieve_lessons(spec)
               if (use_fewshots and cad_retrieval is not None) else [])
    if lessons:
        brief["notes"] = brief.get("notes", []) + [
            "PITFALLS to avoid (learned from past builds of similar parts):\n"
            + "\n".join(f"- {l}" for l in lessons)]
        log.info("[v4] Injected %d learned pitfall(s).", len(lessons))

    # ── Code-model strategy: manual > config pin > auto (triage + escalation) ──
    global _ACTIVE_CODE_MODEL
    pinned = (_load_config().get("cad", {}).get("code_model") or "").strip()
    auto_escalate = False
    global _CLOUD_CALLS_LEFT
    _CLOUD_CALLS_LEFT = int(cloud_config().get("max_calls_per_build", 4)) if cloud_config() else 0
    if coder == "cloud":
        cc = cloud_config()
        if not cc:
            raise RuntimeError("--coder cloud requires a cad.json `cloud` block "
                               "({provider, model, api_key_env?, max_calls_per_build?})")
        _ACTIVE_CODE_MODEL = CLOUD_PREFIX + cc["model"]
        log.info("[v4] Code model: %s (manual --coder cloud, budget %d calls)",
                 _ACTIVE_CODE_MODEL, _CLOUD_CALLS_LEFT)
    elif coder in ("fast", "mid", "strong"):
        _ACTIVE_CODE_MODEL = {"fast": CODE_MODEL_FAST, "mid": CODE_MODEL_MID,
                              "strong": CODE_MODEL_STRONG}[coder]
        log.info("[v4] Code model: %s (manual --coder %s)", _ACTIVE_CODE_MODEL, coder)
    elif pinned:
        _ACTIVE_CODE_MODEL = pinned
        log.info("[v4] Code model: %s (pinned via cad.code_model)", _ACTIVE_CODE_MODEL)
    else:
        # Hard specs skip the first rung and start one step up the ladder; whatever the
        # ladder's top is stays escalation-only.
        _ACTIVE_CODE_MODEL = (CODE_MODEL_LADDER[1] if spec_needs_strong_coder(spec, brief)
                              and len(CODE_MODEL_LADDER) > 1 else CODE_MODEL_FAST)
        auto_escalate = True
        log.info("[v4] Code model: %s (auto%s)", _ACTIVE_CODE_MODEL,
                 ", may escalate" if auto_escalate else "")

    code = generate_code(brief)

    with tempfile.TemporaryDirectory(prefix="cadv4_") as work_str:
        work_dir = Path(work_str)
        step_path: Optional[Path] = None
        prev_step: Optional[Path] = None   # last good build, kept so an edit can be diffed against it
        best_step: Optional[Path] = None   # last GATE-PASSING build + its code — a later broken edit
        best_code: Optional[str] = None    # must never discard verified-correct geometry
        last_state = ""
        last_errors: list[str] = []
        last_critique: Optional[str] = None
        done = False
        gate_passed = False  # did the build currently held in step_path pass the deterministic gate?
        accepted_via: Optional[str] = None  # 'critic' | 'gate' | 'helper' — how convergence was reached
        fails = 0   # failed/stuck turns on the current coder (drives auto-escalation)
        failure_categories: dict = {}   # B3: per-build failure histogram (feeds the fine-tune track)
        def _note_failure(err_text: str) -> str:
            cat, hint = diagnose(err_text)
            failure_categories[cat] = failure_categories.get(cat, 0) + 1
            log.info("[v4] failure category: %s", cat)
            return hint
        recovered_problem: Optional[str] = None   # first failure overcome (drives Stage B lessons)

        for turn in range(1, MAX_TURNS + 1):
            if time.monotonic() - t0 > BUILD_TIMEOUT:
                log.warning("[v4] Wall-clock budget hit before turn %d", turn)
                break

            # Auto-escalate to the stronger coder once the fast one has failed a few times.
            nxt = _next_code_model(_ACTIVE_CODE_MODEL) if auto_escalate else None
            if nxt and fails >= ESCALATE_AFTER:
                log.info("[v4] Escalating one rung to %s after %d failed turn(s).", nxt, fails)
                _ACTIVE_CODE_MODEL = nxt
                code = generate_code(brief)   # fresh attempt with the stronger model
                fails = 0

            log.info("[v4] ── Turn %d/%d (%s) ──", turn, MAX_TURNS, _ACTIVE_CODE_MODEL)

            # 1. Syntax pre-flight (cheap — avoids burning a subprocess on a typo)
            syn = _check_syntax(code)
            if syn:
                log.warning("[v4] %s", syn)
                _note_failure("SyntaxError: " + syn)
                last_errors = [syn]
                fails += 1
                recovered_problem = recovered_problem or f"syntax error: {syn}"
                if turn < MAX_TURNS:
                    code = revise_script(spec, code, syn)
                continue

            # 2. Run build123d
            try:
                step_path, step_log = run_step(code, work_dir)
                log.info("[v4] build123d OK: %s", step_log.strip().replace("\n", " | "))
            except Exception as e:
                err = str(e)
                log.warning("[v4] Run failed: %s", err[:300])
                last_errors = [err]
                step_path = None
                fails += 1
                recovered_problem = recovered_problem or f"runtime error: {err[:200]}"
                # B3: classify the failure and use the category's targeted repair hint (the old
                # ad-hoc fillet hint is now one row of the taxonomy table in cad_v5/diagnose.py).
                cat_hint = _note_failure(err)
                hint = ("\n\n" + cat_hint) if cat_hint else ""
                if turn < MAX_TURNS:
                    code = revise_script(spec, code, f"The script failed to run:\n{err}{hint}")
                continue

            # 3. Inspect geometry
            inspection = run_inspect(step_path)
            last_state = inspection["output"]
            log.info("[v4] Inspect valid=%s", inspection["valid"])
            if not inspection["valid"]:
                last_errors = inspection["errors"]
                step_path = None  # don't upload invalid geometry
                _note_failure("; ".join(inspection["errors"]))
                fails += 1
                recovered_problem = recovered_problem or ("invalid geometry: " + "; ".join(inspection["errors"])[:200])
                if turn < MAX_TURNS:
                    code = revise_script(
                        spec, code,
                        "The geometry is invalid:\n" + "\n".join(inspection["errors"]),
                        state=inspection["output"],
                    )
                continue

            # 3a. If this turn is an EDIT of a previous good build, deterministically report what
            # changed (ΔSolids/Faces/Cyl/Volume/Bbox) so the coder/critic can confirm the edit did
            # the intended thing — not via the vision model.
            if prev_step and prev_step.exists():
                edit_diff = run_diff(prev_step, step_path)
                if edit_diff:
                    last_state += "\n[edit diff vs previous build]\n" + edit_diff
                    log.info("[v4] Edit diff: %s", edit_diff.replace("\n", " | ")[:220])

            # 3b. Deterministic verification gate — assert geometry facts vs the brief's expected
            # target (solid count, holes-actually-cut, overall size). This is the AUTHORITY for
            # feature presence; the vision critic below only judges shape/proportion.
            facts = parse_facts(inspection["output"])
            gate_fails, gate_notes = verify_expected(facts, brief.get("expected", {}), spec=spec)
            if gate_notes:
                last_state += "\n[advisory] " + "  ".join(gate_notes)
            if gate_fails:
                gate_passed = False
                msg = "; ".join(gate_fails)
                log.warning("[v4] Verification gate FAILED: %s", msg)
                _note_failure(msg)
                last_errors = gate_fails
                fails += 1
                recovered_problem = recovered_problem or ("verification failed: " + msg[:200])
                if turn < MAX_TURNS:
                    code = revise_script(
                        spec, code,
                        "The build is geometrically WRONG. These deterministic checks failed "
                        "(authoritative — they measure the actual solid):\n"
                        + "\n".join(f"- {f}" for f in gate_fails)
                        + "\nFix the source so every check passes. Remember primitives are centred at "
                          "the origin, so a cut must overlap the solid's actual position to remove material.",
                        state=inspection["output"])
                continue
            if facts:
                gate_passed = True
                log.info("[v4] Verification gate passed (solids=%s cyl_faces=%s bbox=%s).",
                         facts.get("solids"), facts.get("cyl_faces"), facts.get("bbox"))
                # Snapshot every gate-passing build so a broken later edit can never discard
                # verified-correct geometry (restored at finalize if the loop ends worse off).
                try:
                    best_step = work_dir / "best_output.step"
                    shutil.copy(step_path, best_step)
                    best_code = code
                except Exception as e:
                    log.warning("[v4] best-build snapshot failed: %s", e)
                    best_step = None
                # The gate only confirms a valid single fused solid — it does NOT verify features
                # or dimensions. So the visual critic IS the feature judge: it must check the part
                # actually has the requested shape and features, and look in the top-down panel for
                # holes/grooves/pockets before deciding.
                last_state += ("\n[gate] valid single fused solid produced — now judge whether the "
                               "shape, proportions, AND every requested feature (holes, grooves, "
                               "bores, bosses) are actually present and correct; check the top-down "
                               "panel for top-face features before concluding one is missing.")

            # Domain-helper geometry is correct by construction — accept it without
            # subjecting it to the visual critic (a single isometric view of a long
            # section foreshortens the cross-section and yields false negatives).
            if re.search(r"result\s*=\s*(structural_section|spur_gear|hex_bolt|"
                         r"gear|screw|nut|iso_thread|ball_bearing|pipe_flange)\s*\(", code):
                log.info("[v4] Trusted library-helper geometry is valid — accepting.")
                done = True
                accepted_via = "helper"
                break

            # 4 + 5. Render + multimodal critique (graceful if unavailable)
            critique = visual_critique(step_path, spec, last_state, work_dir,
                                        section=want_section, questions=questions)
            if critique:
                last_critique = critique
                log.info("[v4] Critic: %s", critique.replace("\n", " ")[:300])

            # 6. Decide: done, or edit and re-observe
            action, new_code = decide_or_edit(
                spec, code, last_state, inspection["warnings"], critique
            )
            if action == "done":
                log.info("[v4] Model satisfied — finalizing.")
                done = True
                accepted_via = "critic"
                break
            # The model wants to EDIT (it is NOT satisfied) but returned identical code. Try a
            # directive revise using the critic's specific complaint to break the stalemate.
            if (new_code or "").strip() == code.strip():
                fails += 1
                recovered_problem = recovered_problem or (
                    f"critic rejected: {critique}" if critique else "model stalled with no change")
                if critique and turn < MAX_TURNS:
                    log.info("[v4] decide stalled — attempting a directive revise from the critique.")
                    retry = revise_script(spec, code,
                                          f"A reviewer says this is WRONG: {critique}\n"
                                          f"Fix it so every requested feature is present and "
                                          f"correctly placed (primitives are centred at the origin).")
                    if retry.strip() != code.strip():
                        code = retry
                        continue
                # Coder is genuinely stuck (even a directive revise produced identical code).
                # Reaching this point means THIS turn's geometry PASSED the deterministic gate —
                # it is verified-correct on every measurable axis (solid count, axis-invariant
                # size, holes cut, through-vs-blind bores). Accept it via the gate.
                # Crucially, do NOT escalate a gate-passing build to the 30B: the strong coder
                # rewrites from scratch and was observed to DESTROY a correct build (the enclosure
                # benchmark passed the gate twice on the 7B, then a 30B escalation crashed on a bad
                # fillet → total failure). An un-actionable visual nitpick must not sink a
                # verified-correct part. (`accepted_via=gate` keeps every such accept auditable.)
                if gate_passed:
                    log.info("[v4] Coder stuck but the deterministic gate passed — accepting as "
                             "converged via gate (not escalating a verified-correct build).")
                    done = True
                    accepted_via = "gate"
                    break
                # Gate UNVERIFIED (geometry facts didn't parse) — there is no measurement to
                # trust, so escalate to the strong coder for fresh code before giving up.
                nxt = _next_code_model(_ACTIVE_CODE_MODEL) if auto_escalate else None
                if nxt and turn < MAX_TURNS:
                    log.info("[v4] Coder stuck (gate unverified) — escalating one rung to %s.", nxt)
                    _ACTIVE_CODE_MODEL = nxt
                    code = generate_code(brief)
                    fails = 0
                    continue
                log.warning("[v4] Model is stuck (no change) and gate unverified — "
                            "stopping NOT converged.")
                break
            log.info("[v4] Model chose to keep editing.")
            # Snapshot this good build (run_step overwrites build_output.step next turn) so the
            # edit can be diffed against it deterministically.
            try:
                snap = work_dir / "prev_output.step"
                shutil.copy(step_path, snap)
                prev_step = snap
            except Exception:
                prev_step = None
            code = new_code  # next turn runs the edit; step_path holds last good build

        # A failed or regressed final edit must never discard an earlier gate-verified build:
        # if the loop ended unaccepted holding nothing (or a gate-failing build) while a
        # gate-passing snapshot exists, restore the snapshot and export that instead.
        if (not done and best_step is not None and best_step.exists()
                and (step_path is None or not gate_passed)):
            log.warning("[v4] Loop ended with %s — restoring the last gate-verified build "
                        "instead of discarding it.",
                        "no geometry" if step_path is None else "a gate-failing build")
            step_path = best_step
            code = best_code or code
            gate_passed = True

        if not step_path or not step_path.exists():
            raise RuntimeError(
                f"Build failed after {MAX_TURNS} turn(s).\n" + "\n".join(last_errors)
            )

        # Turn budget exhausted without an explicit accept, but the last geometry PASSED the
        # deterministic gate → it is verified-correct. Accept it via the gate rather than discard
        # a correct part (consistent with the stuck-branch policy, now that the gate checks
        # solids/size/holes/through-vs-blind and can be trusted).
        if not done and gate_passed:
            log.info("[v4] Turn budget exhausted but the last build passed the deterministic "
                     "gate — accepting as converged via gate.")
            done = True
            accepted_via = "gate"

        # Persist artifacts into a per-build directory (concurrent builds previously
        # clobbered each other in the shared cad-last-build.* files), then refresh the
        # legacy cad-last-build.* convenience copies that Satine/v5/docs rely on.
        build_dir = _new_build_dir(spec)
        step_out = build_dir / "build.step"
        stl_out  = build_dir / "build.stl"
        dxf_out  = build_dir / "build.dxf"
        shutil.copy(step_path, step_out)
        # The recipe IS the parametric model — persist it so `cad params`/`cad regen`
        # can edit dimensions and rebuild without an LLM.
        try:
            (build_dir / "build_source.py").write_text(code)
        except Exception as e:
            log.warning("[v4] could not save build_source.py: %s", e)
        # Always export a sliceable STL alongside the STEP (printable mesh, mm units).
        try:
            run_stl(step_out, stl_out)
        except Exception as e:
            log.warning("[v4] STL export failed: %s", e)
        # Best-effort flat-pattern DXF (laser/sheet). Only meaningful for plate-like parts; a part
        # with no usable flat face just won't get one — not an error, so never fail the build.
        dxf_summary = ""
        try:
            _, dxf_summary = run_dxf(step_out, dxf_out)
            log.info("[v4] DXF flat pattern: %s", dxf_summary or dxf_out)
        except Exception as e:
            log.info("[v4] DXF flat pattern skipped: %s", str(e)[:120])
        # Legacy last-build copies (clear stale ones first so a missing export from THIS
        # build can never be mistaken for its output).
        for src, legacy in ((step_out, STEP_OUT), (stl_out, STL_OUT), (dxf_out, DXF_OUT)):
            try:
                legacy.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                log.warning("[v4] could not clear %s: %s", legacy, e)
            if src.exists():
                try:
                    shutil.copy(src, legacy)
                except Exception as e:
                    log.warning("[v4] last-build copy failed (%s): %s", legacy.name, e)

    # Stage B: if this build recovered from an early failure, distil one reusable lesson.
    if done and recovered_problem and use_fewshots and cad_retrieval is not None:
        lesson = distill_lesson(spec, recovered_problem, code)
        if lesson:
            try:
                cad_retrieval.store_lesson(spec, lesson, recovered_problem)
                log.info("[v4] Learned: %s", lesson)
            except Exception as e:
                log.warning("[v4] store_lesson failed: %s", e)

    base_result = {
        "spec":         spec,
        "path":         "code_v4",
        "version":      VERSION,
        "build_dir":    str(build_dir),
        "step_local":   str(step_out),
        "stl_local":    str(stl_out) if stl_out.exists() else "",
        "dxf_local":    str(dxf_out) if dxf_out.exists() else "",
        "code":         code,
        "brief":        brief,
        "model_done":   done,
        "converged":    done,
        "accepted_via": accepted_via,
        "code_model":   _ACTIVE_CODE_MODEL,
        "fewshots_used": [fs["spec"] for fs in fewshots],
        "failure_categories": failure_categories,
        "last_critique": last_critique,
        "build_time_s": round(time.monotonic() - t0, 1),
        "built_at":     datetime.now(timezone.utc).isoformat(),
    }
    if not done:
        base_result["warning"] = (
            f"Loop ended after {MAX_TURNS} turns without the model confirming the part "
            f"matches the spec — the uploaded geometry is the last valid build and may be "
            f"wrong. Last critique: {last_critique or 'n/a'}"
        )
        log.warning("[v4] Not converged — uploading last valid build as best effort.")

    if not do_upload:
        render_local = ""
        if final_render:   # chat skips this — fstl/CAD-Viewer show the model instead of a static PNG
            try:
                png  = run_render(step_out, build_dir, section=want_section)  # +section if internal
                dest = build_dir / "build.png"
                if Path(png) != dest:
                    shutil.copy(png, dest)
                shutil.copy(dest, STEP_OUT.with_suffix(".png"))   # legacy convenience copy
                render_local = str(dest)
            except Exception as e:
                log.warning("[v4] Final render failed: %s", e)
        result = {**base_result, "ok": True, "has_bodies": True, "url": "",
                  "render_local": render_local}
        _write_session(result)
        log.info("[v4] Done (no upload): step=%s render=%s", step_out, render_local)
        return result

    try:
        upload = import_step_to_onshape(step_out, name, public=_public_uploads())
    except Exception as e:
        fail = {**base_result, "ok": False, "has_bodies": True,
                "url": "", "error": str(e)}
        _write_session(fail)
        log.error("[v4] Upload failed: %s", e)
        raise RuntimeError(f"{e}  (local STEP saved at {step_out})")

    result = {**base_result, "ok": True, "has_bodies": True,
              "url": upload["url"], "did": upload["did"],
              "wid": upload["wid"], "eid": upload["eid"]}
    _write_session(result)
    log.info("[v4] Done: url=%s  time=%.1fs", result["url"], result["build_time_s"])
    return result

# ── Feedback store (kept) ─────────────────────────────────────────────────────

def store_feedback(result: dict, rating: int, comment: str = "") -> None:
    """Append a ≥4★ rated build to the corpus so it can be retrieved as a future few-shot.
    Gold (curated, geometry-verified) entries are always preserved; rated entries are deduped
    by spec and capped. Only stores builds that carry code — never empty scaffolding."""
    if rating < 4 or not result.get("code"):
        return
    gold, rated = [], []
    if FEEDBACK_FILE.exists():
        for line in FEEDBACK_FILE.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            (gold if row.get("source") == "gold" else rated).append(row)
    # Drop any prior rated entry for the same spec (gold is never dropped).
    rated = [r for r in rated if r.get("spec") != result.get("spec")]
    rated.append({
        "spec": result.get("spec", ""), "code": result.get("code", ""),
        "source": "rated", "rating": rating, "comment": comment,
        "url": result.get("url", ""), "code_model": result.get("code_model"),
        "converged": result.get("converged"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if len(rated) > 50:
        rated = rated[-50:]
    # Atomic replace — a crash mid-rewrite must not lose the gold corpus.
    tmp = FEEDBACK_FILE.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row in gold + rated:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, FEEDBACK_FILE)
    log.info("[v4] Stored %d★ to corpus (gold=%d rated=%d): %s",
             rating, len(gold), len(rated), result.get("spec", "")[:50])

# ── Telegram helper (now actually wired from _cmd_build) ───────────────────────

def _tg_send(chat_id: str, text: str) -> None:
    token = _tg_token()
    if not token or not chat_id:
        return
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
    except Exception as e:
        log.warning("[v4] Telegram send failed: %s", e)

# ── CLI commands ──────────────────────────────────────────────────────────────

def _cmd_build(argv):
    import argparse
    p = argparse.ArgumentParser(prog="build")
    p.add_argument("spec", nargs="+")
    p.add_argument("--chat-id", default=None)
    p.add_argument("--coder", choices=["auto", "fast", "mid", "strong", "cloud"], default="auto",
                   help="auto=triage spec + escalate on failure (default); "
                        "fast=force the 7B coder; strong=force the 30B coder")
    p.add_argument("--no-fewshots", action="store_true",
                   help="disable retrieved few-shot examples (to A/B-measure the retrieval lift)")
    p.add_argument("--no-upload", action="store_true",
                   help="skip Onshape upload; render the final STEP to a local PNG instead "
                        "(for benchmarking or when no Onshape creds are configured)")
    a = p.parse_args(argv)
    spec = " ".join(a.spec)

    try:
        result = build(spec, chat_id=a.chat_id, coder=a.coder,
                       use_fewshots=not a.no_fewshots, do_upload=not a.no_upload)
    except Exception as e:
        log.error("[v4] Build error: %s", e)
        _tg_send(a.chat_id or "", f"❌ CAD build failed: {spec}\n\n{str(e)[:300]}")
        print(json.dumps({"ok": False, "error": str(e), "spec": spec}))
        sys.exit(1)

    print(json.dumps(result, indent=2))
    url = result.get("url")
    if url:
        print(f"\n>>> URL: {url}")
        if result.get("converged"):
            _tg_send(a.chat_id or "", f"✅ CAD model ready!\n\n📐 {spec}\n🔗 {url}")
        else:
            _tg_send(a.chat_id or "",
                     f"⚠️ CAD model uploaded but may not match the spec (it iterated without "
                     f"confirming).\n\n📐 {spec}\n🔗 {url}\n\nReply with a correction to refine.")
    elif result.get("render_local"):
        print(f"\n>>> No upload — local STEP: {result.get('step_local')}  render: {result['render_local']}")
    else:
        print("\n>>> No URL — build failed")


def _cmd_rate(argv):
    if not argv or not argv[0].isdigit():
        print("Usage: rate <1-5> [comment...]"); sys.exit(1)
    rating  = int(argv[0])
    comment = " ".join(argv[1:]) if len(argv) > 1 else ""
    session = _read_session()
    if not session:
        print("No session found — run a build first."); sys.exit(1)
    store_feedback(session, rating, comment)
    print(f"Saved {'⭐' * rating} for: {session.get('spec', '')[:60]}")


def _cmd_session(argv):
    session = _read_session()
    print(json.dumps(session, indent=2) if session else "No session.")


def _cmd_brief(argv):
    print(json.dumps(build_brief(" ".join(argv)), indent=2))


def _cmd_code(argv):
    spec  = " ".join(argv)
    print(generate_code(build_brief(spec)))


def _cmd_inspect(argv):
    if not argv:
        print(json.dumps({"ok": False, "error": "Usage: inspect <onshape_url>"})); sys.exit(1)
    url = argv[0]
    try:
        did, wid, eid = _parse_onshape_url(url)
    except ValueError as e:
        print(json.dumps({"ok": False, "error": str(e)})); sys.exit(1)
    try:
        description = _describe_document(did, wid, eid)
        print(json.dumps({"ok": True, "description": description,
                          "url": url, "did": did, "wid": wid, "eid": eid}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)})); sys.exit(1)


def merge_spec(original: str, feedback: str, history: Optional[list] = None) -> str:
    """Merge a feedback message into the running spec → a single revised spec sentence.

    The shared conversational primitive: both `refine` and the interactive `chat` loop use
    this so a follow-up like "make the grooves deeper" turns into a complete, self-contained
    spec the stateless build123d regen can act on."""
    history = history or []
    history_str = ""
    if history:
        history_str = "\nPrevious iterations:\n" + "\n".join(
            f"  - {h.get('spec','')}" for h in history[-3:])
    system = ("You are a CAD specification editor. Merge the feedback into the original spec to "
              "produce a revised spec. Return ONLY the new spec as a single sentence — no explanation.")
    prompt = f"Original: {original}\nFeedback: {feedback}{history_str}\n\nRevised spec:"
    revised = _ollama(BRIEF_MODEL, system, prompt, timeout=OLLAMA_TIMEOUT).strip()
    if not revised or len(revised) < 5:
        revised = f"{original}, {feedback}"
    return revised


def _cmd_refine(argv):
    if len(argv) < 2:
        print("Usage: refine <original_spec> <feedback> [<history_json>]"); sys.exit(1)
    original, feedback = argv[0], argv[1]
    try:
        history = json.loads(argv[2]) if len(argv) > 2 else []
    except Exception:
        history = []
    print(merge_spec(original, feedback, history))


def _cmd_chat(argv):
    """Interactive back-and-forth: build, then iterate with plain-English feedback.

    Each turn rebuilds the part from the full evolving spec (build123d regen is stateless),
    saves STEP+STL+PNG locally, and shows the result. Type feedback to refine; /commands below."""
    import argparse
    p = argparse.ArgumentParser(prog="chat", add_help=False)
    p.add_argument("--coder", choices=["auto", "fast", "mid", "strong", "cloud"], default="auto")
    p.add_argument("--no-fewshots", action="store_true")
    p.add_argument("opening", nargs="*", help="optional first spec; otherwise you'll be prompted")
    a = p.parse_args(argv)

    HELP = (
        "\nCommands:\n"
        "  <text>          plain English — first message = the part, after that = feedback to refine\n"
        "  /spec           show the current full spec\n"
        "  /show           print STEP / STL paths (+ web viewer link if running)\n"
        f"  /view           (re)open the latest STL in {STL_VIEWER_CMD}\n"
        "  /coder <m>      switch coder: auto | fast | strong\n"
        "  /redo           rebuild the current spec unchanged (e.g. after a coder switch)\n"
        "  /upload         push the latest build to Onshape as a Part Studio\n"
        "  /help           this list\n"
        "  /quit           exit\n"
    )
    print(f"CAD chat — v{VERSION}.  Describe a part to start; refine it in plain English. /help, /quit.")
    viewer_note = f"auto-opens in {STL_VIEWER_CMD}" if shutil.which(STL_VIEWER_CMD) else f"{STL_VIEWER_CMD} not found — install it or set CAD_STL_VIEWER"
    print(f"Coder: {a.coder} | fewshots: {not a.no_fewshots} | STEP+STL each turn, {viewer_note} (/upload to push)")

    spec: Optional[str] = None
    history: list = []
    coder = a.coder
    last_result: dict = {}

    def _link(path: str) -> str:
        """Wrap an absolute path as an OSC 8 terminal hyperlink (ctrl-click in GNOME Terminal)."""
        if not path:
            return ""
        uri = "file://" + str(Path(path).resolve())
        return f"\033]8;;{uri}\033\\{path}\033]8;;\033\\"

    def _open_in_fstl(stl: str) -> bool:
        """Auto-launch the local GUI STL viewer (fstl) detached. Best-effort."""
        if not stl or not Path(stl).exists() or not shutil.which(STL_VIEWER_CMD):
            return False
        try:
            subprocess.Popen([STL_VIEWER_CMD, str(Path(stl).resolve())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            return True
        except Exception:
            return False

    def _web_url(path: str) -> str:
        """CAD Viewer (browser) URL for `path`, only if a viewer server is up on CAD_VIEWER_PORT."""
        if not path:
            return ""
        try:
            with socket.create_connection(("127.0.0.1", CAD_VIEWER_PORT), timeout=0.3):
                pass
        except OSError:
            return ""
        p = Path(path).resolve()
        return (f"http://127.0.0.1:{CAD_VIEWER_PORT}/?dir={p.parent}&file={p.name}")

    def _do_build(s: str):
        nonlocal last_result
        print(f"\n… building [{coder}]: {s}")
        try:
            last_result = build(s, coder=coder, use_fewshots=not a.no_fewshots,
                                do_upload=False, final_render=False)
        except Exception as e:
            print(f"  build failed: {e}")
            return
        conv = "converged ✓" if last_result.get("converged") else "NOT converged ✗"
        via  = last_result.get("accepted_via") or "—"
        print(f"  {conv} via {via} | {last_result.get('code_model','?')} | {last_result.get('build_time_s','?')}s")
        if last_result.get("last_critique"):
            print(f"  critic: {last_result['last_critique']}")
        if last_result.get("warning"):
            print(f"  ⚠ {last_result['warning']}")
        stl = last_result.get("stl_local", "")
        print(f"  STEP: {_link(last_result.get('step_local',''))}")
        print(f"  STL : {_link(stl) or '(export failed)'}")
        if _open_in_fstl(stl):
            print(f"  → opened in {STL_VIEWER_CMD}")
        web = _web_url(stl)
        if web:
            print(f"  Web : {web}")

    if a.opening:
        spec = " ".join(a.opening)
        history.append({"spec": spec})
        _do_build(spec)

    while True:
        try:
            line = input("\ncad> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); break
        if not line:
            continue
        if line in ("/quit", "/q", "/exit"):
            print("bye"); break
        if line in ("/help", "/h", "/?"):
            print(HELP); continue
        if line == "/spec":
            print(spec or "(no spec yet)"); continue
        if line == "/show":
            if last_result:
                print(f"  STEP: {_link(last_result.get('step_local',''))}")
                print(f"  STL : {_link(last_result.get('stl_local',''))}")
                web = _web_url(last_result.get("stl_local", ""))
                if web:
                    print(f"  Web : {web}")
            else:
                print("(nothing built yet)")
            continue
        if line in ("/view", "/v"):
            if last_result and _open_in_fstl(last_result.get("stl_local", "")):
                print(f"  → reopened in {STL_VIEWER_CMD}")
            else:
                print(f"(nothing to view, or {STL_VIEWER_CMD} not found)")
            continue
        if line.startswith("/coder"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in ("auto", "fast", "strong"):
                coder = parts[1]; print(f"  coder → {coder}")
            else:
                print("  usage: /coder auto|fast|strong")
            continue
        if line == "/redo":
            if spec:
                _do_build(spec)
            else:
                print("(no spec yet)")
            continue
        if line == "/upload":
            if not last_result or not Path(STEP_OUT).exists():
                print("(nothing to upload yet)"); continue
            try:
                up = import_step_to_onshape(STEP_OUT, "cad-chat-build", public=_public_uploads())
                print(f"  uploaded: {up['url']}")
            except Exception as e:
                print(f"  upload failed: {e}")
            continue
        if line.startswith("/rate"):
            print("  (rating is set aside for now — it isn't feeding the pipeline)")
            continue
        if line.startswith("/"):
            print(f"  unknown command {line!r} — /help"); continue

        # Plain text: first message sets the part; after that it's feedback to merge.
        if spec is None:
            spec = line
        else:
            print("  (merging feedback into spec …)")
            spec = merge_spec(spec, line, history)
            print(f"  revised spec: {spec}")
        history.append({"spec": spec})
        _do_build(spec)


CMDS = {
    "build": _cmd_build, "chat": _cmd_chat, "rate": _cmd_rate, "session": _cmd_session,
    "brief": _cmd_brief, "code": _cmd_code, "refine": _cmd_refine, "inspect": _cmd_inspect,
}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in CMDS:
        CMDS[sys.argv[1]](sys.argv[2:])
    else:
        print(f"CAD Agent v{VERSION}")
        print(f"Usage: {sys.argv[0]} <{'|'.join(CMDS)}> [args...]")
        print("\nCommands:")
        print("  build <spec> [--chat-id <id>] [--coder auto|fast|strong]")
        print("                                 — agentic observe-edit loop + Onshape upload")
        print("                                   (auto: triage spec + escalate 7B→30B on failure)")
        print("  chat [<spec>] [--coder ...] [--no-fewshots]")
        print("                                 — INTERACTIVE: build then refine in plain English")
        print("                                   (rebuilds local STEP+STL+PNG each turn; /help inside)")
        print("  rate  <1-5> [comment]          — rate last build (stores ≥4★)")
        print("  session                        — show last build session")
        print("  refine <orig> <feedback> [hist]— merge feedback into spec (Satine compat)")
        print("  brief <spec>                   — debug: show build brief only")
        print("  code  <spec>                   — debug: generate code only (no build)")
        sys.exit(1)

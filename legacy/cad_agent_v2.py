#!/usr/bin/env python3
"""
CAD Agent v2 — Two-stage planner/executor pipeline for Onshape.

Stage 1: phi3:latest (Planner)    — NL spec → ordered JSON plan
Stage 2: qwen2.5:14b (Executor)   — step-by-step execution with validation

Fallback triggers:
  - Planner outputs invalid JSON
  - >2 consecutive tool failures
  - Tool call wall-time >10 s (MCP timeout budget)

Usage:
  python3 cad_agent_v2.py build "spur gear 20 teeth module 2 meshing pinion 3:1 ratio"
  python3 cad_agent_v2.py build "W200x100 I-beam 2000mm" --chat-id 7788781234
  python3 cad_agent_v2.py plan  "some spec"   # dry-run planner only
"""

import os, sys, json, re, time, math, base64, logging, subprocess
import urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Optional RAG support (requires: pip install chromadb) ──────────────────────
try:
    import sys as _sys
    import os as _os
    _HERE_RAG = _os.path.dirname(_os.path.abspath(__file__))
    if _HERE_RAG not in _sys.path:
        _sys.path.insert(0, _HERE_RAG)
    from cad_rag import rag_query_tiered as _rag_query_tiered
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False
    def _rag_query_tiered(spec): return {"tier1": [], "tier2": [], "tier3": []}

# ── Paths & config ─────────────────────────────────────────────────────────────

_HERE         = Path(__file__).parent
_OPENCLAW     = Path.home() / ".openclaw"
LOG_FILE      = _OPENCLAW / "cad-agent.log"
FEEDBACK_FILE  = _OPENCLAW / "cad-feedback.jsonl"
FAILURES_FILE  = _OPENCLAW / "cad-failures.jsonl"
SESSION_FILE   = _OPENCLAW / "cad-session.json"
FALLBACK_SCRIPT = _HERE / "onshape_cad_agent.py"
CONFIG_FILE   = _OPENCLAW / "openclaw.json"

PLANNER_MODEL  = "qwen3:8b"            # fits fully on GPU, better reasoning than 2.5:14b
EXECUTOR_MODEL = "qwen2.5:14b"        # GPU — fallback: qwen2.5-coder:32b via CPU
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300   # seconds per LLM call (14b model needs headroom)
PLANNER_TIMEOUT = 600  # planner uses thinking mode — give it time

MAX_PLAN_STEPS    = 8
PLAN_TIMEOUT_S    = 120
MAX_CONSECUTIVE_FAILURES = 2
TOOL_TIMEOUT_S    = 10   # wall-clock budget for MCP tool calls before fallback
TOOL_TIMEOUT_DIRECT = 120  # budget for direct builders (create_gear/create_bolt — multiple API calls)

MAX_FEEDBACK_ROWS = 50
MAX_FEWSHOT       = 3
RAG_TOP_K         = 5
MASS_DEVIATION_PCT = 0.50   # repair if >50 % off expected

BASE_URL = "https://cad.onshape.com"

# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stderr),   # stderr so JSON stdout stays clean
        ],
    )

_setup_logging()
log = logging.getLogger("cad_v2")

# ── Credentials ────────────────────────────────────────────────────────────────

def _load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _creds():
    cfg = _load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

def _tg_token():
    cfg = _load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

# ── Onshape REST API ────────────────────────────────────────────────────────────

def _onshape(method, path, body=None):
    ak, sk = _creds()
    creds = base64.b64encode(f"{ak}:{sk}".encode()).decode()
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


# ── Onshape URL parser ──────────────────────────────────────────────────────────

_ONSHAPE_URL_RE = re.compile(
    r"https://cad\.onshape\.com/documents/([a-f0-9]+)/w/([a-f0-9]+)/e/([a-f0-9]+)",
    re.IGNORECASE,
)

def _parse_onshape_url(url: str) -> tuple[str, str, str]:
    """Return (did, wid, eid) from an Onshape URL, or raise ValueError."""
    m = _ONSHAPE_URL_RE.search(url)
    if not m:
        raise ValueError(f"Not a recognised Onshape part-studio URL: {url!r}")
    return m.group(1), m.group(2), m.group(3)


def _describe_document(did: str, wid: str, eid: str) -> str:
    """Fetch Onshape document data and return a plain-English description via LLM."""
    try:
        doc_meta = _onshape("GET", f"/api/v9/documents/{did}")
        doc_name = doc_meta.get("name", "Unknown document")
    except Exception:
        doc_name = "Unknown document"

    try:
        features_resp = _onshape("GET", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features")
        features = features_resp.get("features", [])
        feature_names = [
            f"{f.get('name', '?')} ({f.get('featureType', '?')})"
            for f in features
            if not f.get("suppressed")
        ]
    except Exception as e:
        feature_names = [f"(could not fetch features: {e})"]

    try:
        mass = _get_mass_props(did, wid, eid)
    except Exception:
        mass = {"has_bodies": False}

    try:
        bbox = _get_bounding_box(did, wid, eid) if mass.get("has_bodies") else {}
    except Exception:
        bbox = {}

    feat_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(feature_names)) or "  (none)"
    mass_line = (
        f"Mass: {mass['mass_g']} g, Volume: {mass['volume_cm3']} cm³"
        if mass.get("mass_g") else "No solid bodies yet."
    )
    bbox_line = ""
    if bbox:
        dx = round((bbox.get("max_x_mm", 0) - bbox.get("min_x_mm", 0)), 1)
        dy = round((bbox.get("max_y_mm", 0) - bbox.get("min_y_mm", 0)), 1)
        dz = round((bbox.get("max_z_mm", 0) - bbox.get("min_z_mm", 0)), 1)
        bbox_line = f"Bounding box: {dx} × {dy} × {dz} mm"

    doc_summary = (
        f"Document: {doc_name}\n"
        f"Features ({len(feature_names)}):\n{feat_list}\n"
        f"{mass_line}\n{bbox_line}"
    ).strip()

    system = (
        "You are a CAD assistant. The user has shared an Onshape part-studio. "
        "Summarise what the model shows in 2-4 sentences: what it appears to be, "
        "its key features, and its approximate size. Be concise and conversational."
    )
    try:
        return _ollama(PLANNER_MODEL, system, f"Describe this Onshape model:\n\n{doc_summary}",
                       timeout=60)
    except Exception:
        return doc_summary


# ── Web fetch helpers ───────────────────────────────────────────────────────────

def _html_strip(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&amp;","&"),("&lt;","<"),("&gt;",">"),("&nbsp;"," "),("&#xA0;"," ")]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()


def _web_fetch(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return _html_strip(raw)


def _search_section_props(designation: str) -> Optional[dict]:
    """
    Search the web for a structural section's d, bf, tf, tw dimensions (all mm).
    Uses DuckDuckGo JSON + HTML endpoints then extracts values with the LLM.
    Returns dict with d_mm, bf_mm, tf_mm, tw_mm as floats, or None on failure.
    """
    log.info("[Section] Searching web for %s properties ...", designation)
    search_text = ""

    # 1. DuckDuckGo JSON instant-answer API (fast, often has abstract for std sections)
    try:
        q = urllib.parse.quote(
            f"{designation} steel section depth flange web thickness mm dimensions"
        )
        ddg_json = _web_fetch(
            f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1",
            timeout=10,
        )
        ddg = json.loads(ddg_json)
        if ddg.get("AbstractText"):
            search_text += f"Abstract: {ddg['AbstractText']}\n"
        for rt in ddg.get("RelatedTopics", [])[:5]:
            if isinstance(rt, dict) and rt.get("Text"):
                search_text += rt["Text"] + "\n"
    except Exception as e:
        log.warning("[Section] DDG JSON failed: %s", e)

    # 2. DuckDuckGo HTML search — grab plain-text snippets
    try:
        q = urllib.parse.quote(
            f"{designation} section properties d bf tf tw mm steel datasheet"
        )
        html_text = _web_fetch(
            f"https://html.duckduckgo.com/html/?q={q}", timeout=15
        )
        search_text += html_text[:4000]
    except Exception as e:
        log.warning("[Section] DDG HTML failed: %s", e)

    if not search_text.strip():
        log.warning("[Section] No web content for %s", designation)
        return None

    log.info("[Section] %d chars retrieved for %s", len(search_text), designation)

    # 3. LLM extracts the numeric dimensions from whatever text we got
    system = (
        "You are extracting structural steel I-section dimensions from web search results. "
        "Return ONLY a JSON object with these numeric keys (all in millimetres): "
        "d_mm (overall depth), bf_mm (flange width), tf_mm (flange thickness), tw_mm (web thickness). "
        "Use the exact numbers from the source. Omit keys you cannot find. No other text."
    )
    prompt = (
        f"Find dimensions for section {designation!r} from this text:\n\n"
        f"{search_text[:5000]}"
    )
    try:
        raw  = _ollama(PLANNER_MODEL, system, prompt, timeout=90)
        text = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        dims = json.loads(text)
        if all(dims.get(k) for k in ("d_mm", "bf_mm", "tf_mm", "tw_mm")):
            dims = {k: float(dims[k]) for k in ("d_mm", "bf_mm", "tf_mm", "tw_mm")}
            log.info("[Section] %s → %s", designation, dims)
            return dims
        log.warning("[Section] LLM returned incomplete dims: %s", dims)
    except Exception as e:
        log.warning("[Section] LLM extraction failed: %s", e)

    return None


# ── LLM-optimised tool schemas ─────────────────────────────────────────────────

TOOL_SCHEMAS = {
    "create_document": {
        "when": "Always call first. Creates the Onshape document that holds all parts. "
                "Returns documentId and workspaceId needed by every later tool.",
        "params": {
            "name": "string — human-readable name, e.g. 'Spur Gear Assembly 20T'",
        },
        "returns": "documentId: str, workspaceId: str",
        "failure_modes": "None expected on fresh builds.",
    },
    "get_part_studio": {
        "when": "Call immediately after create_document to get the default Part Studio elementId. "
                "Required before any sketch or feature tool.",
        "params": {
            "documentId":  "string",
            "workspaceId": "string",
        },
        "returns": "elementId: str",
        "failure_modes": "Always present in new documents.",
    },
    "create_gear": {
        "when": "Call for any toothed rotary transmission component — spur gear, helical gear, pinion, "
                "sprocket, cog, or any meshing pair. If the spec describes something that transfers "
                "rotational motion through teeth, use this. "
                "For a meshing pair both gears must share the same module value. "
                "Place pinion at centerX = (gear_pitch_radius_in + pinion_pitch_radius_in) inches from gear centre.",
        "params": {
            "numTeeth":      "integer — MUST be integer, min 4. Gear ratio = N_gear / N_pinion.",
            "module":        "number — tooth size in mm. Must match meshing partner. Typical: 1, 1.5, 2, 2.5, 3.",
            "pressureAngle": "number — default 20 (degrees)",
            "thickness":     "number — face width in INCHES, default 0.5 (~12.7mm)",
            "boreDiameter":  "number — shaft hole diameter in INCHES, default 0.0",
            "centerX":       "number — X position in INCHES (NOT mm), default 0.0",
            "centerY":       "number — Y position in INCHES (NOT mm), default 0.0",
            "plane":         "string — Front | Top | Right, default Front",
            "name":          "string — e.g. 'Gear 20T' or 'Pinion 7T'",
        },
        "returns": "featureId: str",
        "failure_modes": "numTeeth < 4 causes undercut. Module mismatch breaks mesh.",
    },
    "create_sketch_circle": {
        "when": "Use whenever the cross-section of the part is round — tubes, rods, shafts, "
                "bosses, pins, discs, rings, cans, cups, or anything cylindrical. "
                "Prefer this over create_sketch_rectangle any time the profile is circular.",
        "params": {
            "center": "array [x, y] — circle centre in INCHES, usually [0, 0]",
            "radius": "number — radius in INCHES (e.g. 1.575 for 40mm radius / 80mm diameter)",
            "plane":  "string — Front | Top | Right, default Top (Top gives vertical cylinders)",
            "name":   "string",
        },
        "returns": "featureId: str (pass to create_extrude as sketchFeatureId)",
        "failure_modes": "Zero radius causes silent failure.",
    },
    "create_sketch": {
        "when": "Use when the cross-section has more than one shape in it — e.g. a tube (outer + inner "
                "circle), a slotted plate (rectangle + circle), or any profile that can't be captured "
                "by a single primitive. Supports lines, circles, and rectangles mixed freely.",
        "params": {
            "entities": "array of entity objects. Each has 'type': 'circle'|'line'|'rectangle'. "
                        "Circle: {type, center:[x,y], radius} — all in INCHES. "
                        "Line: {type, start:[x,y], end:[x,y]}. "
                        "Rectangle: {type, corner1:[x,y], corner2:[x,y]}. "
                        "Add isConstruction:true to any for a construction entity.",
            "plane":    "string — Front | Top | Right, default Top",
            "name":     "string",
        },
        "returns": "featureId: str",
        "failure_modes": "Invalid entity type or missing required fields ignored silently.",
    },
    "create_sketch_rectangle": {
        "when": "Use whenever the cross-section is rectangular or square — flat plates, bars, "
                "box sections, brackets, housings, keys, or any part with a straight-sided profile. "
                "For round cross-sections use create_sketch_circle instead.",
        "params": {
            "corner1": "array [x, y] — first corner in INCHES",
            "corner2": "array [x, y] — opposite corner in INCHES",
            "plane":   "string — Front | Top | Right, default Front",
            "name":    "string",
        },
        "returns": "featureId: str (pass to create_extrude as sketchFeatureId)",
        "failure_modes": "Zero or negative dimensions cause silent failure.",
    },
    "create_extrude": {
        "when": "Call after ANY sketch tool to make a solid body. "
                "operationType=NEW for first body; ADD to merge; REMOVE to cut material (pocket/hole/slot). "
                "Set oppositeDirection=true to extrude downward/backward (e.g. legs going below a Top-plane seat).",
        "params": {
            "sketchFeatureId":  "string — featureId from the preceding sketch step",
            "depth":            "number — extrude depth in INCHES (NOT mm). E.g. 3.937 for 100mm",
            "operationType":    "string — NEW | ADD | REMOVE | INTERSECT, default NEW",
            "oppositeDirection": "boolean — true to flip extrude direction (default false)",
            "name":             "string",
        },
        "returns": "featureId: str",
        "failure_modes": "REMOVE fails if no solid body exists yet. Sketch must succeed first.",
    },
    "create_revolve": {
        "when": "Revolve a closed sketch profile around an axis to create a solid of revolution. "
                "Use for any rotationally symmetric shape: bowl, cup, mug body, vase, wine glass, "
                "spoon bowl, knob, ring, jar, bottle, pulley, wheel hub. "
                "Sketch the RIGHT HALF of the profile only (x ≥ 0) on the Front plane. "
                "The Y-axis is the revolution axis — place the profile's centreline at x=0.",
        "params": {
            "sketchFeatureId":  "string — featureId from the preceding sketch step",
            "axis":             "string — Y | X | Z, default Y (use Y for Front-plane cup/bowl profiles)",
            "angle":            "number — revolution angle in degrees, default 360 (full solid of revolution)",
            "operationType":    "string — NEW | ADD | REMOVE, default NEW",
            "name":             "string",
        },
        "returns": "featureId: str",
        "failure_modes": "Profile must be entirely on one side of the axis (x ≥ 0 for Y-axis). "
                         "Closed profile required — open profiles produce a thin shell, not a solid.",
    },
    "create_hole": {
        "when": "Shorthand for a circular through-hole or blind bore. "
                "Use instead of create_sketch_circle + create_extrude(REMOVE) when the spec "
                "explicitly mentions hole, bore, drill, or through-hole.",
        "params": {
            "sketchFeatureId": "string — featureId of a circle sketch at the hole location",
            "depth":           "number — hole depth in INCHES (use a large value like 99 for through-holes)",
            "name":            "string — default 'Hole'",
        },
        "returns": "featureId: str",
        "failure_modes": "Requires a solid body to cut into — must come after body-creating steps.",
    },
    "create_stepped_extrude": {
        "when": "Use for counterbore holes — a stepped hole with multiple diameters at different depths. "
                "Typical use: M8 counterbore = large head clearance diameter at top, smaller thread diameter below.",
        "params": {
            "center": "array [x, y] — hole centre in INCHES",
            "radii":  "array of numbers — radii in INCHES, largest to smallest (e.g. [0.315, 0.157] for 16mm→8mm counterbore)",
            "depths": "array of numbers — CUMULATIVE depths in INCHES for each step, same length as radii",
            "plane":  "string — Front | Top | Right, default Top",
            "namePrefix": "string — prefix for step names, e.g. 'Counterbore'",
        },
        "returns": "featureId: str",
        "failure_modes": "radii and depths must be same length. Body must exist before calling.",
    },
    "create_thicken": {
        "when": "Use to give thickness to a surface or open sketch profile (shell-like result). "
                "Alternative to extrude when you need symmetric thickening or surface offset.",
        "params": {
            "sketchFeatureId": "string — featureId of sketch to thicken",
            "thickness":       "number — thickness in INCHES",
            "operationType":   "string — NEW | ADD | REMOVE | INTERSECT, default NEW",
            "midplane":        "boolean — true for symmetric thickness (half each side), default false",
            "name":            "string",
        },
        "returns": "featureId: str",
        "failure_modes": "Open profiles only — closed profiles should use create_extrude instead.",
    },
    "create_fillet": {
        "when": "Use when edges of an existing solid need to be softened, rounded, or relieved — "
                "whether the user says fillet, chamfer, radius, blend, or just describes a part "
                "that would naturally have rounded edges for strength or aesthetics. "
                "Requires edgeIds — call get_edges first to find valid IDs.",
        "params": {
            "edgeIds":   "array of edge deterministic ID strings — from get_edges",
            "radius":    "number — fillet radius in INCHES",
            "filletType": "string — EDGE | FACE | FULL_ROUND, default EDGE",
            "name":      "string — default 'Fillet'",
        },
        "returns": "featureId: str",
        "failure_modes": "Empty edgeIds or invalid IDs → feature error.",
    },
    "get_mass_properties": {
        "when": "Always include as the LAST step to verify the model was built. "
                "Empty bodies indicate a build failure upstream.",
        "params": {
            "documentId":  "string",
            "workspaceId": "string",
            "elementId":   "string",
        },
        "returns": "mass_g: float, volume_cm3: float, has_bodies: bool",
        "failure_modes": "Empty response if no solid body produced.",
    },
    "get_bounding_box": {
        "when": "Call during repair pass to measure part extents for deviation check.",
        "params": {
            "documentId":  "string",
            "workspaceId": "string",
            "elementId":   "string",
        },
        "returns": "min_x_mm, max_x_mm, min_y_mm, max_y_mm, min_z_mm, max_z_mm",
        "failure_modes": "Empty if no solid body.",
    },
    "create_section": {
        "when": "Call whenever spec mentions an I-beam, i-beam, H-beam, wide-flange, "
                "structural section, or any standard designation (e.g. 250UC72.9, W200x100). "
                "Use the PRE-COMPUTED d_mm/bf_mm/tf_mm/tw_mm values from the geometry hints — "
                "if no designation is given, IPE200 defaults are already pre-computed for you.",
        "params": {
            "d_mm":       "number — section depth in mm",
            "bf_mm":      "number — flange width in mm",
            "tf_mm":      "number — flange thickness in mm",
            "tw_mm":      "number — web thickness in mm",
            "length_mm":  "number — member length in mm, default 1000",
            "mitre_ends": "boolean — true to cut 45° isosceles right triangle at each end for L-joints",
            "plane":      "string — Front | Top | Right, default Front",
            "name":       "string — e.g. '250UC72.9'",
        },
        "returns": "featureId: str",
        "failure_modes": "Wrong dimensions produce incorrect section properties.",
    },
    "create_bolt": {
        "when": "Use for any threaded fastener you tighten with a wrench or driver — bolts, "
                "screws, cap screws, hex fasteners, studs, or threaded connectors. "
                "Builds hex head + cylindrical shank + ISO metric threads. "
                "Always prefer this over sketch+extrude for fasteners.",
        "params": {
            "size":        "number — nominal diameter in mm (e.g. 18 for M18, 8 for M8)",
            "pitch":       "number — thread pitch in mm. Defaults: M4=0.7 M6=1 M8=1.25 M10=1.5 M12=1.75 M16=2 M18=2.5 M20=2.5",
            "acrossFlats": "number — hex head across-flats in mm (default size×1.5)",
            "headHeight":  "number — hex head height in mm (default size×0.64)",
            "shankLength": "number — shank length in mm (default 80)",
            "plane":       "string — Front | Top | Right, default Top",
            "name":        "string — e.g. 'M18 Bolt'",
        },
        "returns": "featureId: str, threadStatus: str",
        "failure_modes": "ThreadCreator requires the plugin to be installed in the document's custom features.",
    },
    "mirror_part": {
        "when": "Use when the design has bilateral symmetry — one half is the reflection of the other, "
                "the part comes in a left/right or top/bottom pair, or the spec implies symmetry "
                "even without using the word 'mirror'. Build one half then mirror rather than "
                "modelling both halves independently.",
        "params": {
            "plane":         "string — Front | Right | Top. Default: Right",
            "operationType": "string — NEW (separate mirrored body) | ADD (boolean merge). Default: NEW",
            "name":          "string — feature name, e.g. 'Mirror Beam'",
        },
        "returns": "featureId: str",
        "failure_modes": "Fails if no solid bodies exist yet. Must come after body-creating steps.",
    },
    "create_thread": {
        "when": "Use when an existing cylindrical surface needs to be threaded — a nut bore, "
                "a threaded rod, a lid that screws onto a container, or any part that mates "
                "by screwing. Apply after the cylinder body exists. "
                "Internal (isExternal=false) for bores/nuts; external (isExternal=true) for bosses/rods.",
        "params": {
            "threadType":   "string — ISO metric thread designation e.g. 'M40', 'M30', 'M20', 'M8'. "
                            "Must match cylinder nominal diameter.",
            "isExternal":   "boolean — false for internal (bore/nut), true for external (boss/bolt)",
            "depth_mm":     "number — thread engagement depth in mm",
            "faceIndex":    "int — which cylindrical face to thread: 0=first/outer (default), 1=inner bore. "
                            "For hollow cylinder (container): use 1 to target the bore, 0 for lid outer boss.",
            "rightHand":    "boolean — true (default) for right-hand thread",
            "fullThread":   "boolean — true to thread full depth (default true)",
            "name":         "string — e.g. 'Internal Thread M40'",
        },
        "returns": "featureId: str",
        "failure_modes": "Cylindrical face must exist. threadType must match the cylinder diameter exactly. "
                         "Use faceIndex=1 to target inner bore of hollow cylinder.",
    },
}

TOOL_SCHEMA_TEXT = json.dumps(
    {name: {k: v for k, v in s.items() if k != "failure_modes"}
     for name, s in TOOL_SCHEMAS.items()},
    indent=2,
)

# ── MCP tool bridge — clarsbyte/onshape-mcp ────────────────────────────────────

_mcp_call_tool_fn = None  # lazy-loaded on first use
_mcp_event_loop = None    # persistent loop — asyncio.run() closes loop between calls


def _get_mcp_call_tool():
    global _mcp_call_tool_fn
    if _mcp_call_tool_fn is None:
        ak, sk = _creds()
        os.environ.setdefault("ONSHAPE_ACCESS_KEY", ak)
        os.environ.setdefault("ONSHAPE_SECRET_KEY", sk)
        mcp_repo = str(Path.home() / ".openclaw/skills/onshape-mcp")
        if mcp_repo not in sys.path:
            sys.path.insert(0, mcp_repo)
        import importlib
        mod = importlib.import_module("onshape_mcp.server")
        _mcp_call_tool_fn = mod.call_tool
    return _mcp_call_tool_fn


def _get_mcp_loop():
    """Return a persistent event loop. Creates a new one if current is closed."""
    global _mcp_event_loop
    import asyncio
    if _mcp_event_loop is None or _mcp_event_loop.is_closed():
        _mcp_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_mcp_event_loop)
        # Also reset the httpx client inside the MCP server so it runs on the new loop
        try:
            import importlib
            mod = importlib.import_module("onshape_mcp.server")
            mod.client._client = None  # force re-init on next use
        except Exception:
            pass
    return _mcp_event_loop


def _call_mcp_tool(name: str, arguments: dict) -> dict:
    """Call a tool from clarsbyte/onshape-mcp using a persistent event loop."""
    ct = _get_mcp_call_tool()
    loop = _get_mcp_loop()
    try:
        text_contents = loop.run_until_complete(ct(name, arguments))
    except Exception as e:
        raise RuntimeError(f"MCP tool {name!r} failed: {e}") from e

    result: dict = {"_responses": [tc.text for tc in text_contents]}
    for tc in text_contents:
        if "Error" in tc.text and "Feature ID" not in tc.text:
            raise RuntimeError(f"MCP tool {name!r} returned error: {tc.text[:300]}")
        m = re.search(r"Feature ID:\s*(\S+)", tc.text)
        if m and m.group(1) != "unknown":
            result["featureId"] = m.group(1)
    return result


# ── Onshape feature-status guard ───────────────────────────────────────────────

def _check_feature_status(resp: dict, label: str) -> str:
    """Raise RuntimeError if Onshape reports ERROR/INVALID on a feature POST. Returns featureId."""
    fid    = resp.get("feature", {}).get("featureId")
    status = resp.get("featureState", {}).get("featureStatus", "")
    notice = resp.get("featureState", {}).get("featureError", "")
    if status in ("ERROR", "INVALID"):
        raise RuntimeError(
            f"{label} featureStatus={status}: {notice or '(no detail)'} (fid={fid})"
        )
    if not fid:
        raise RuntimeError(f"{label} returned no featureId: {resp}")
    log.info("[Feature] %s fid=%s status=%s", label, fid, status or "OK")
    return fid


# ── Involute gear builder (direct BTM JSON, single connected polygon) ─────────

def _build_gear_direct(did: str, wid: str, eid: str, params: dict) -> dict:
    """
    Build a proper involute spur gear via direct Onshape BTM JSON.
    Computes the full gear polygon (outer profile + optional bore circle) in Python,
    submits as a single closed sketch (shared point IDs), then extrudes once.
    """
    N       = int(params["numTeeth"])
    mod_mm  = float(params.get("module", 2.0))
    pa_deg  = float(params.get("pressureAngle", 20.0))
    thick_in = float(params.get("thickness", 0.5))
    bore_in  = float(params.get("boreDiameter", 0.0))
    cx_in    = float(params.get("centerX", 0.0))
    cy_in    = float(params.get("centerY", 0.0))
    plane_nm = params.get("plane", "Front")
    label    = params.get("name", f"Gear {N}T")

    # Hardcoded standard plane IDs (same as the MCP server)
    plane_id = {"Front": "JCC", "Top": "JDC", "Right": "JEC"}.get(plane_nm, "JCC")

    # ── Gear geometry in meters (Onshape API uses SI units) ──────────────────
    alpha = math.radians(pa_deg)
    m     = mod_mm / 1000.0          # module in metres
    cx_m  = cx_in * 0.0254
    cy_m  = cy_in * 0.0254

    r_p = m * N / 2                  # pitch radius
    r_a = r_p + m                    # addendum (tip) radius
    r_f = max(r_p - 1.25 * m, 0.3 * m)  # dedendum (root) radius
    r_b = r_p * math.cos(alpha)      # base circle radius

    if r_b >= r_p:
        raise RuntimeError(
            f"Base circle radius ({r_b*1000:.2f} mm) >= pitch radius ({r_p*1000:.2f} mm). "
            f"Reduce pressureAngle or increase numTeeth."
        )

    r_inv_start = max(r_f, r_b)      # involute starts at base circle or root (whichever larger)
    tooth_angle = 2 * math.pi / N

    def inv(t: float):
        return (r_b * (math.cos(t) + t * math.sin(t)),
                r_b * (math.sin(t) - t * math.cos(t)))

    def t_at(r: float) -> float:
        return math.sqrt(max(0.0, (r / r_b) ** 2 - 1.0))

    t_root = t_at(r_inv_start)
    t_tip  = t_at(r_a)
    t_p    = t_at(r_p)
    x_p, y_p = inv(t_p)
    angle_at_pitch = math.atan2(y_p, x_p)
    half_tooth = math.pi / (2 * N)

    rot_right = -(half_tooth + angle_at_pitch)
    rot_left  =   half_tooth + angle_at_pitch

    def rot2d(x, y, theta):
        c, s = math.cos(theta), math.sin(theta)
        return x * c - y * s, x * s + y * c

    # Adaptive segment count — cap total sketch entities at ~800 (near Onshape's undocumented limit)
    N_INV = 8
    N_ARC = 5
    MAX_ENTITIES = 800
    segs_per_tooth = 2 * N_INV + 2 * N_ARC - 5
    while N > 4 and segs_per_tooth * N > MAX_ENTITIES:
        if N_INV > 4:
            N_INV -= 1
        elif N_ARC > 3:
            N_ARC -= 1
        else:
            break
        segs_per_tooth = 2 * N_INV + 2 * N_ARC - 5
    log.info("[Gear] N_INV=%d N_ARC=%d → %d segments for %dT gear",
             N_INV, N_ARC, segs_per_tooth * N, N)

    def involute_strip(t0, t1, rotation, flip_y=False):
        pts = []
        for i in range(N_INV):
            t = t0 + (t1 - t0) * i / (N_INV - 1)
            x, y = inv(t)
            if flip_y:
                y = -y
            pts.append(rot2d(x, y, rotation))
        return pts

    def arc_strip(r, a0, a1):
        pts = []
        for i in range(N_ARC):
            a = a0 + (a1 - a0) * i / (N_ARC - 1)
            pts.append((r * math.cos(a), r * math.sin(a)))
        return pts

    # ── Assemble all tooth points (single continuous polygon, CCW) ───────────
    # For each tooth i:
    #   right involute root→tip  →  tip arc  →  left involute tip→root  →  root arc to next tooth
    all_pts: list[tuple[float, float]] = []

    # Pre-compute per-tooth angles (in local frame, before rotation by ca)
    right_local = involute_strip(t_root, t_tip, rot_right)
    left_local  = involute_strip(t_tip, t_root, rot_left, flip_y=True)

    rr_local_angle = math.atan2(right_local[0][1], right_local[0][0])   # root of right flank
    lr_local_angle = math.atan2(left_local[-1][1], left_local[-1][0])   # root of left flank
    rt_local_angle = math.atan2(right_local[-1][1], right_local[-1][0]) # tip of right flank
    lt_local_angle = math.atan2(left_local[0][1],   left_local[0][0])   # tip of left flank

    for i in range(N):
        ca = i * tooth_angle

        # Right involute (root → tip)
        for x, y in right_local:
            all_pts.append(rot2d(x, y, ca))

        # Tip arc (right tip → left tip)
        a0_tip = rt_local_angle + ca
        a1_tip = lt_local_angle + ca
        # Tip arc goes counterclockwise (left_tip > right_tip in angle)
        while a1_tip < a0_tip:
            a1_tip += 2 * math.pi
        for pt in arc_strip(r_a, a0_tip, a1_tip)[1:]:   # skip dup
            all_pts.append(pt)

        # Left involute (tip → root)
        for x, y in left_local[1:]:                      # skip dup tip
            all_pts.append(rot2d(x, y, ca))

        # Root arc (left root of tooth i → right root of tooth i+1)
        a0_root = lr_local_angle + ca
        a1_root = rr_local_angle + (i + 1) * tooth_angle
        while a1_root < a0_root:
            a1_root += 2 * math.pi
        for pt in arc_strip(r_f, a0_root, a1_root)[1:-1]:  # skip both endpoints (shared)
            all_pts.append(pt)

    # Translate to gear centre (in metres)
    all_pts_m = [(cx_m + x, cy_m + y) for x, y in all_pts]

    # Drop near-duplicate consecutive points (zero-length segments crash Onshape)
    MIN_SEG = 1e-9  # metres
    clean: list[tuple[float, float]] = []
    for pt in all_pts_m:
        if not clean or math.hypot(pt[0]-clean[-1][0], pt[1]-clean[-1][1]) > MIN_SEG:
            clean.append(pt)
    # Also check closure (last → first)
    while clean and math.hypot(clean[-1][0]-clean[0][0], clean[-1][1]-clean[0][1]) < MIN_SEG:
        clean.pop()

    n_pts = len(clean)
    log.info("[Gear] Profile polygon: %d points for %dT gear", n_pts, N)

    # ── Build sketch entities (connected polygon — shared point IDs) ─────────
    pids = [f"gp{j}" for j in range(n_pts)]
    entities = []
    for j in range(n_pts):
        x1, y1 = clean[j]
        x2, y2 = clean[(j + 1) % n_pts]
        dx, dy  = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < MIN_SEG:
            continue
        entities.append({
            "btType": "BTMSketchCurveSegment-155",
            "entityId": f"gseg{j}",
            "startPointId": pids[j],
            "endPointId":   pids[(j + 1) % n_pts],
            "startParam": 0.0,
            "endParam":   seg_len,
            "geometry": {
                "btType": "BTCurveGeometryLine-117",
                "pntX": x1, "pntY": y1,
                "dirX": dx / seg_len, "dirY": dy / seg_len,
            },
            "isConstruction": False,
        })

    # Optional bore circle (as a separate inner loop — Onshape will cut it out)
    if bore_in > 0.0:
        bore_r_m = (bore_in / 2.0) * 0.0254
        entities.append({
            "btType": "BTMSketchCurve-4",
            "entityId": "gbore",
            "centerId": "gbore.center",
            "geometry": {
                "btType": "BTCurveGeometryCircle-115",
                "radius":  bore_r_m,
                "xCenter": cx_m,
                "yCenter": cy_m,
                "xDir": 1.0, "yDir": 0.0,
                "clockwise": False,
            },
            "isConstruction": False,
        })

    # ── Submit sketch ────────────────────────────────────────────────────────
    sk_data = {"feature": {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": f"{label} Profile",
        "suppressed": False,
        "parameters": [{"btType": "BTMParameterQueryList-148",
                         "queries": [{"btType": "BTMIndividualQuery-138",
                                       "deterministicIds": [plane_id]}],
                         "parameterId": "sketchPlane"}],
        "entities": entities,
        "constraints": [],
    }}
    sk_resp = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", sk_data)
    sk_fid  = _check_feature_status(sk_resp, "Gear sketch")

    # ── Submit extrude ───────────────────────────────────────────────────────
    filter_inner = bore_in > 0.0   # True → outer annular region (ring) only; False → solid disk
    ex_data = {"btType": "BTFeatureDefinitionCall-1406", "feature": {
        "btType": "BTMFeature-134",
        "featureType": "extrude",
        "name": label,
        "suppressed": False,
        "namespace": "",
        "parameters": [
            {
                "btType": "BTMParameterQueryList-148",
                "parameterId": "entities",
                "queries": [{
                    "btType": "BTMIndividualSketchRegionQuery-140",
                    "featureId": sk_fid,
                    "filterInnerLoops": filter_inner,
                    "queryString": f'query = qSketchRegion(id + "{sk_fid}", {str(filter_inner).lower()});',
                    "deterministicIds": [],
                }],
            },
            {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
             "value": "NEW", "enumName": "NewBodyOperationType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
             "expression": f"{thick_in} in", "value": thick_in,
             "units": "", "isInteger": False},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }}
    ex_resp = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", ex_data)
    ex_fid  = _check_feature_status(ex_resp, "Gear extrude")

    return {"featureId": ex_fid, "sketchFeatureId": sk_fid}


# ── ThreadCreator custom feature namespace (from user's reference bolt document) ─
_THREAD_CREATOR_NS = (
    "d940a64ae93da270e768d2c9a::v7efc504c97d9ea02bf3e8bae::"
    "eac49831e657b0b9f8e0cd5ad::mfd8ae6bb081d206849c83d6b"
)

# Filter used by ThreadCreator for face selection (from reference document BTM)
_THREAD_FACE_FILTER = {
    "btType": "BTAndFilter-110",
    "operand1": {
        "btType": "BTOrFilter-167",
        "operand1": {"btType": "BTGeometryFilter-130", "geometryType": "LINE"},
        "operand2": {"btType": "BTOrFilter-167",
            "operand1": {"btType": "BTGeometryFilter-130", "geometryType": "CIRCLE"},
            "operand2": {"btType": "BTOrFilter-167",
                "operand1": {"btType": "BTGeometryFilter-130", "geometryType": "ARC"},
                "operand2": {"btType": "BTOrFilter-167",
                    "operand1": {"btType": "BTGeometryFilter-130", "geometryType": "CYLINDER"},
                    "operand2": {"btType": "BTOrFilter-167",
                        "operand1": {"btType": "BTGeometryFilter-130", "geometryType": "CONE"},
                        "operand2": {"btType": "BTGeometryFilter-130", "geometryType": "REVOLVED"},
                    },
                },
            },
        },
    },
    "operand2": {"btType": "BTEntityTypeFilter-124", "entityType": "FACE"},
}


def _build_bolt_direct(did: str, wid: str, eid: str, params: dict) -> dict:
    """
    Build a standard metric hex-head bolt via direct Onshape BTM JSON.
    Steps: hex head sketch → extrude head → shank circle sketch
           → extrude shank (opposite direction) → ThreadCreator ISO threads.
    """
    m_size   = float(params.get("size", 18))
    pitch    = float(params.get("pitch", 2.5))
    af_mm    = float(params.get("acrossFlats", round(m_size * 1.5, 1)))
    head_h   = float(params.get("headHeight",  round(m_size * 0.64, 1)))
    shank_l  = float(params.get("shankLength", 80.0))
    label    = params.get("name", f"M{int(m_size)} Bolt")
    plane_nm = params.get("plane", "Top")
    plane_id = {"Front": "JCC", "Top": "JDC", "Right": "JEC"}.get(plane_nm, "JDC")

    # Geometry units
    r_shank    = m_size / 2000             # metres (Onshape SI)
    r_vtx      = (af_mm / 2) / math.cos(math.radians(30)) / 1000
    head_h_in  = head_h  / 25.4           # mm → inches (matches gear extrude path)
    shank_l_in = shank_l / 25.4

    def post_feature(body):
        return _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", body)

    # ── Hex head sketch ──────────────────────────────────────────────────────
    hex_pts = [(r_vtx * math.cos(math.radians(i * 60)),
                r_vtx * math.sin(math.radians(i * 60))) for i in range(6)]
    hex_ents = []
    for j in range(6):
        x1, y1 = hex_pts[j]
        x2, y2 = hex_pts[(j + 1) % 6]
        dx, dy  = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        hex_ents.append({
            "btType": "BTMSketchCurveSegment-155",
            "entityId": f"bh{j}", "startPointId": f"bhp{j}",
            "endPointId": f"bhp{(j+1)%6}",
            "startParam": 0.0, "endParam": L,
            "geometry": {"btType": "BTCurveGeometryLine-117",
                         "pntX": x1, "pntY": y1, "dirX": dx/L, "dirY": dy/L},
            "isConstruction": False,
        })
    r1 = post_feature({"feature": {
        "btType": "BTMSketch-151", "featureType": "newSketch",
        "name": f"{label} Head Profile", "suppressed": False,
        "parameters": [{"btType": "BTMParameterQueryList-148",
                         "queries": [{"btType": "BTMIndividualQuery-138",
                                       "deterministicIds": [plane_id]}],
                         "parameterId": "sketchPlane"}],
        "entities": hex_ents, "constraints": [],
    }})
    sk1_fid = _check_feature_status(r1, "Bolt head sketch")

    # ── Head extrude ─────────────────────────────────────────────────────────
    r2 = post_feature({"btType": "BTFeatureDefinitionCall-1406", "feature": {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": f"{label} Head", "suppressed": False, "namespace": "",
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                           "featureId": sk1_fid, "filterInnerLoops": False,
                           "queryString": f'query=qSketchRegion(id+"{sk1_fid}",false);',
                           "deterministicIds": []}]},
            {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
             "value": "NEW", "enumName": "NewBodyOperationType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
             "expression": f"{head_h_in} in", "value": head_h_in, "units": "", "isInteger": False},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }})
    ex1_fid = _check_feature_status(r2, "Bolt head extrude")

    # ── Shank sketch (circle on same plane) ───────────────────────────────────
    r3 = post_feature({"feature": {
        "btType": "BTMSketch-151", "featureType": "newSketch",
        "name": f"{label} Shank Profile", "suppressed": False,
        "parameters": [{"btType": "BTMParameterQueryList-148",
                         "queries": [{"btType": "BTMIndividualQuery-138",
                                       "deterministicIds": [plane_id]}],
                         "parameterId": "sketchPlane"}],
        "entities": [{
            "btType": "BTMSketchCurve-4", "entityId": "bsc",
            "centerId": "bsc.center",
            "geometry": {"btType": "BTCurveGeometryCircle-115",
                         "radius": r_shank, "xCenter": 0.0, "yCenter": 0.0,
                         "xDir": 1.0, "yDir": 0.0, "clockwise": False},
            "isConstruction": False,
        }],
        "constraints": [],
    }})
    sk2_fid = _check_feature_status(r3, "Bolt shank sketch")

    # ── Shank extrude (ADD, opposite direction to head) ───────────────────────
    r4 = post_feature({"btType": "BTFeatureDefinitionCall-1406", "feature": {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": f"{label} Shank", "suppressed": False, "namespace": "",
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                           "featureId": sk2_fid, "filterInnerLoops": False,
                           "queryString": f'query=qSketchRegion(id+"{sk2_fid}",false);',
                           "deterministicIds": []}]},
            {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
             "value": "ADD", "enumName": "NewBodyOperationType"},
            {"btType": "BTMParameterBoolean-144", "parameterId": "oppositeDirection", "value": True},
            {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
             "expression": f"{shank_l_in} in", "value": shank_l_in, "units": "", "isInteger": False},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }})
    ex2_fid = _check_feature_status(r4, "Bolt shank extrude")

    # ── ThreadCreator ─────────────────────────────────────────────────────────
    # refChoice: select cylindrical faces created specifically by the shank extrude
    major_cm = m_size / 10
    pitch_cm = pitch  / 10
    r5 = post_feature({"btType": "BTFeatureDefinitionCall-1406", "feature": {
        "btType": "BTMFeature-134",
        "namespace": _THREAD_CREATOR_NS,
        "featureType": "threadCreator",
        "name": f"M{int(m_size)}-{pitch} Thread",
        "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f"query=qGeometry(qCreatedBy(makeId(\"{ex2_fid}\"),EntityType.FACE),GeometryType.CYLINDER);",
                          "deterministicIds": []}],
             "filter": _THREAD_FACE_FILTER,
             "parameterId": "refChoice"},
            {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
             "enumName": "ThreadProfile", "value": "ISO_STD", "parameterId": "threadProfile"},
            {"btType": "BTMParameterBoolean-144", "value": False, "parameterId": "leftHanded"},
            {"btType": "BTMParameterString-149", "value": f"M{int(m_size)}-{pitch}",
             "parameterId": "screwfriendlyname"},
            {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
             "units": "", "expression": f"{major_cm} cm", "parameterId": "majorDiameter"},
            {"btType": "BTMParameterBoolean-144", "value": False, "parameterId": "internalThreads"},
            {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
             "enumName": "DiameterType", "value": "MAJOR", "parameterId": "diameterType"},
            {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
             "units": "", "expression": f"{pitch_cm} cm", "parameterId": "p"},
            {"btType": "BTMParameterBoolean-144", "value": False, "parameterId": "oppositeEnd"},
            {"btType": "BTMParameterQuantity-147", "isInteger": True, "value": 0.0,
             "units": "", "expression": "1", "parameterId": "numStarts"},
            {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
             "enumName": "LengthSelectionType", "value": "FULL", "parameterId": "lengthType"},
            {"btType": "BTMParameterBoolean-144", "value": True, "parameterId": "taperFirstEnd"},
            {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
             "units": "", "expression": "45 deg", "parameterId": "taperFirstEndAngle"},
            {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
             "units": "", "expression": "1", "parameterId": "leadInPitches"},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }})
    thread_fid = _check_feature_status(r5, "ThreadCreator thread")

    return {
        "featureId":       thread_fid,
        "headFeatureId":   ex1_fid,
        "shankFeatureId":  ex2_fid,
        "threadFeatureId": thread_fid,
    }


# ── Structural section builder (direct BTM JSON, I/H polygon) ─────────────────

def _build_section_direct(did: str, wid: str, eid: str, params: dict) -> dict:
    """
    Build a structural I/H-section (UC/UB/WB) via direct Onshape BTM JSON.
    Draws the 12-vertex I-profile as a single closed polygon, extrudes by length.
    Sketch on Front plane → extrudes in +Z (beam length direction).
    """
    d_mm  = float(params["d_mm"])
    bf_mm = float(params["bf_mm"])
    tf_mm = float(params["tf_mm"])
    tw_mm = float(params["tw_mm"])
    L_mm  = float(params.get("length_mm", 1000.0))
    label = params.get("name", "Section")
    plane_nm = params.get("plane", "Front")
    plane_id = {"Front": "JCC", "Top": "JDC", "Right": "JEC"}.get(plane_nm, "JCC")

    # All geometry in metres (Onshape SI)
    h  = d_mm  / 2 / 1000   # half depth
    fw = bf_mm / 2 / 1000   # half flange width
    ft = tf_mm / 1000        # flange thickness
    wt = tw_mm / 2 / 1000   # half web thickness
    L_in = L_mm / 25.4       # length in inches for extrude expression

    # 12-vertex I/H profile CCW (viewed from front): bottom-left → CW around outer, then web cut-ins
    pts = [
        (-fw, -h),       #  0 bottom-left  of bottom flange
        ( fw, -h),       #  1 bottom-right of bottom flange
        ( fw, -h + ft),  #  2 inner  bottom-right (flange/web junction)
        ( wt, -h + ft),  #  3 web    bottom-right
        ( wt,  h - ft),  #  4 web    top-right
        ( fw,  h - ft),  #  5 inner  top-right
        ( fw,  h),       #  6 top-right  of top flange
        (-fw,  h),       #  7 top-left   of top flange
        (-fw,  h - ft),  #  8 inner  top-left
        (-wt,  h - ft),  #  9 web    top-left
        (-wt, -h + ft),  # 10 web    bottom-left
        (-fw, -h + ft),  # 11 inner  bottom-left
    ]

    n    = len(pts)
    pids = [f"sp{i}" for i in range(n)]
    entities = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx, dy  = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        entities.append({
            "btType": "BTMSketchCurveSegment-155",
            "entityId": f"ss{i}",
            "startPointId": pids[i],
            "endPointId":   pids[(i + 1) % n],
            "startParam": 0.0, "endParam": seg_len,
            "geometry": {
                "btType": "BTCurveGeometryLine-117",
                "pntX": x1, "pntY": y1,
                "dirX": dx / seg_len, "dirY": dy / seg_len,
            },
            "isConstruction": False,
        })

    sk_resp = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
        "feature": {
            "btType": "BTMSketch-151", "featureType": "newSketch",
            "name": f"{label} Profile", "suppressed": False,
            "parameters": [{"btType": "BTMParameterQueryList-148",
                            "queries": [{"btType": "BTMIndividualQuery-138",
                                         "deterministicIds": [plane_id]}],
                            "parameterId": "sketchPlane"}],
            "entities": entities, "constraints": [],
        }
    })
    sk_fid = _check_feature_status(sk_resp, f"{label} sketch")

    ex_resp = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
        "btType": "BTFeatureDefinitionCall-1406", "feature": {
            "btType": "BTMFeature-134", "featureType": "extrude",
            "name": label, "suppressed": False, "namespace": "",
            "parameters": [
                {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
                 "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                              "featureId": sk_fid, "filterInnerLoops": False,
                              "queryString": f'query=qSketchRegion(id+"{sk_fid}",false);',
                              "deterministicIds": []}]},
                {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
                 "value": "NEW", "enumName": "NewBodyOperationType"},
                {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
                 "expression": f"{L_in} in", "value": L_in, "units": "", "isInteger": False},
            ],
            "subFeatures": [], "returnAfterSubfeatures": False,
        }
    })
    ex_fid = _check_feature_status(ex_resp, f"{label} extrude")

    if params.get("mitre_ends", False):
        _apply_mitre_cuts(did, wid, eid, label, d_mm, bf_mm, L_mm)

    log.info("[Section] %s built: d=%.1f bf=%.1f tf=%.1f tw=%.1f L=%.0f mm",
             label, d_mm, bf_mm, tf_mm, tw_mm, L_mm)
    return {
        "featureId": ex_fid, "sketchFeatureId": sk_fid,
        "d_mm": d_mm, "bf_mm": bf_mm, "tf_mm": tf_mm, "tw_mm": tw_mm, "length_mm": L_mm,
    }


def _apply_mitre_cuts(did: str, wid: str, eid: str,
                      label: str, d_mm: float, bf_mm: float, L_mm: float) -> None:
    """
    Cut a 45° isosceles right triangle from each end of a beam.

    Sketch on the RIGHT plane (JEC).  Onshape Right-plane sketch axes:
      pntX  →  world -Z  (along beam, reversed; yAxis = +X × (−Z) = +Y)
      pntY  →  world +Y  (beam depth, up)

    Extrude-REMOVE THROUGH_ALL in both X directions to cut through the full flange width.

    Triangle legs = d (full beam depth), giving a 45° cut in the side (YZ) view:
      Near end: right-angle at world (0, +h, 0)  — removes top-near corner
      Far  end: right-angle at world (0, -h, L)  — removes bottom-far corner (complementary)

    Two identical beams rotated 90° about Z will have mating diagonal end faces.
    """
    L_m   = L_mm / 1000
    d_m   = d_mm / 1000        # full depth in metres = triangle leg in Z
    h     = d_m / 2 + 0.005   # half depth + 5mm margin ensures full coverage

    # pntX = -Z_world, pntY = +Y_world
    # Proper saw-cut mitre: the cut plane runs from the top corner of the beam end
    # diagonally down to the bottom at d_m (= beam depth) into the beam.
    # This removes the full end-face wedge, leaving a clean 45° diagonal face.
    # Near and far cuts are mirror images about the beam midpoint (symmetric from side).
    cuts = [
        # Near end — removes wedge from near face to 45° cut plane
        # Hypotenuse (cut face): world(Z=0,Y=+h) → world(Z=d_m,Y=-h)  → slope -1 → 45°
        ("Near Mitre", [
            ( 0.0,  h),     # top-near corner (enter cut)
            ( 0.0, -h),     # bottom-near corner
            (-d_m, -h),     # bottom, d into beam (pntX=-d_m → Z=+d_m)
        ]),
        # Far end — mirror of near; removes wedge from far face to 45° cut plane
        # Hypotenuse (cut face): world(Z=L_m,Y=+h) → world(Z=L_m-d_m,Y=-h) → slope +1 → 45°
        ("Far Mitre", [
            (-L_m,        +h),   # top-far corner (enter cut)
            (-L_m,        -h),   # bottom-far corner
            (-L_m + d_m,  -h),   # bottom, d inward from far end
        ]),
    ]

    for cut_name, pts in cuts:
        tag  = cut_name[0].lower()
        n    = len(pts)
        pids = [f"mp{i}{tag}" for i in range(n)]
        ents = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            dx, dy  = x2 - x1, y2 - y1
            seg     = math.hypot(dx, dy)
            ents.append({
                "btType": "BTMSketchCurveSegment-155",
                "entityId": f"ms{i}{tag}",
                "startPointId": pids[i],
                "endPointId":   pids[(i + 1) % n],
                "startParam": 0.0, "endParam": seg,
                "geometry": {"btType": "BTCurveGeometryLine-117",
                             "pntX": x1, "pntY": y1,
                             "dirX": dx / seg, "dirY": dy / seg},
                "isConstruction": False,
            })

        sk_r = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
            "feature": {
                "btType": "BTMSketch-151", "featureType": "newSketch",
                "name": f"{label} {cut_name} Sketch",
                "suppressed": False,
                "parameters": [{"btType": "BTMParameterQueryList-148",
                                "queries": [{"btType": "BTMIndividualQuery-138",
                                             "deterministicIds": ["JEC"]}],  # Right plane (YZ)
                                "parameterId": "sketchPlane"}],
                "entities": ents, "constraints": [],
            }
        })
        cut_sk_fid = _check_feature_status(sk_r, f"{label} {cut_name} sketch")

        # Onshape sometimes needs a moment to index the sketch region before REMOVE extrude.
        # Retry up to 3 times with increasing delays before giving up.
        last_err = None
        for attempt in range(3):
            if attempt:
                time.sleep(0.5 * attempt)
            try:
                ex_r = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
                    "btType": "BTFeatureDefinitionCall-1406", "feature": {
                        "btType": "BTMFeature-134", "featureType": "extrude",
                        "name": f"{label} {cut_name}",
                        "suppressed": False, "namespace": "",
                        "parameters": [
                            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
                             "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                                          "featureId": cut_sk_fid, "filterInnerLoops": False,
                                          "queryString": f'query=qSketchRegion(id+"{cut_sk_fid}",false);',
                                          "deterministicIds": []}]},
                            {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
                             "value": "REMOVE", "enumName": "NewBodyOperationType"},
                            {"btType": "BTMParameterEnum-145", "parameterId": "endBound",
                             "value": "THROUGH_ALL", "enumName": "BoundingType"},
                            {"btType": "BTMParameterBoolean-144", "parameterId": "hasSecondDirection", "value": True},
                            {"btType": "BTMParameterEnum-145", "parameterId": "secondDirectionBound",
                             "value": "THROUGH_ALL", "enumName": "BoundingType"},
                        ],
                        "subFeatures": [], "returnAfterSubfeatures": False,
                    }
                })
                _check_feature_status(ex_r, f"{label} {cut_name} cut")
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                log.warning("[Section] %s attempt %d failed: %s", cut_name, attempt + 1, e)

        if last_err:
            log.error("[Section] %s failed after 3 attempts — section body still intact", cut_name)
        else:
            log.info("[Section] %s applied", cut_name)


# Standard Onshape plane deterministicIds (consistent across all documents)
_PLANE_IDS = {"Front": "JCC", "Right": "JEC", "Top": "JHC"}


def _apply_mirror(did: str, wid: str, eid: str,
                  name: str = "Mirror",
                  plane: str = "Right",
                  operation: str = "NEW",
                  body_feature_id: Optional[str] = None) -> str:
    """
    Mirror all solid bodies about a standard plane (Front/Right/Top).
    body_feature_id: if provided, selects only bodies created by that feature;
                     otherwise selects all bodies via qEverything.
    Returns the new feature's featureId.
    """
    plane_id = _PLANE_IDS.get(plane, plane)

    if body_feature_id:
        body_query = f'query=qCreatedBy(makeId("{body_feature_id}"), EntityType.BODY);'
    else:
        body_query = "query=qEverything(EntityType.BODY);"

    r = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": {
            "btType": "BTMFeature-134",
            "featureType": "mirror",
            "name": name,
            "suppressed": False,
            "namespace": "",
            "parameters": [
                {"btType": "BTMParameterEnum-145",   "parameterId": "patternType",
                 "enumName": "MirrorType",           "value": "PART"},
                {"btType": "BTMParameterEnum-145",   "parameterId": "operationType",
                 "enumName": "NewBodyOperationType", "value": operation},
                {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
                 "queries": [{"btType": "BTMIndividualQuery-138",
                              "queryStatement": None, "queryString": body_query,
                              "deterministicIds": []}]},
                {"btType": "BTMParameterQueryList-148", "parameterId": "mirrorPlane",
                 "queries": [{"btType": "BTMIndividualQuery-138",
                              "queryStatement": None, "queryString": "",
                              "deterministicIds": [plane_id]}]},
                {"btType": "BTMParameterBoolean-144", "parameterId": "defaultScope",   "value": False},
                {"btType": "BTMParameterQueryList-148", "parameterId": "booleanScope",
                 "queries": [{"btType": "BTMIndividualQuery-138",
                              "queryStatement": None, "queryString": body_query,
                              "deterministicIds": []}]},
                {"btType": "BTMParameterBoolean-144", "parameterId": "fullFeaturePattern", "value": False},
            ],
            "subFeatures": [], "returnAfterSubfeatures": False,
        }
    })
    return _check_feature_status(r, f"{name} mirror")


def _apply_thread(did: str, wid: str, eid: str, params: dict, ctx: dict | None = None) -> str:
    """Apply thread via the same threadCreator plugin used by the bolt builder.

    Uses qGeometry(qCreatedBy(...), CYLINDER) to target the cylindrical face
    produced by the most recent extrude in ctx.  Falls back to qCylindrical()
    if no extrude feature ID is available.
    """
    # Normalize: accept lowercase or camelCase keys, depth/depth_mm alias
    p = {k.lower(): v for k, v in params.items()}
    def _get(key_lower, default):
        return p.get(key_lower, default)

    thread_type  = _get("threadtype", "M20")
    is_internal  = not _get("isexternal", False)   # container bore = internal
    depth_mm     = float(_get("depth_mm", _get("depth", 10.0)))
    face_index   = int(_get("faceindex", 0))
    left_handed  = not _get("righthand", True)
    full_thread  = _get("fullthread", True)
    name         = _get("name", f"Thread {thread_type}")

    # Parse "M74" → size_mm=74, pitch from standard coarse table
    import re as _re
    m = _re.match(r"M(\d+(?:\.\d+)?)", thread_type)
    size_mm = float(m.group(1)) if m else 20.0
    _coarse_pitch = {1:0.25,1.2:0.25,1.6:0.35,2:0.4,2.5:0.45,3:0.5,4:0.7,5:0.8,
                     6:1.0,8:1.25,10:1.5,12:1.75,16:2.0,20:2.5,24:3.0,30:3.5,
                     36:4.0,42:4.5,48:5.0,56:5.5,64:6.0,72:6.0,76:6.0,80:6.0}
    pitch_mm = _coarse_pitch.get(size_mm, round(size_mm * 0.075, 2))
    major_cm = size_mm / 10
    pitch_cm = pitch_mm / 10

    # Target the cylindrical face from the last extrude (same as bolt shank approach)
    last_extrude_fid = (ctx or {}).get("last_extrude_fid")
    if last_extrude_fid:
        qs = (f"query=qGeometry("
              f"qCreatedBy(makeId(\"{last_extrude_fid}\"),EntityType.FACE),"
              f"GeometryType.CYLINDER);")
    elif face_index > 0:
        qs = f"query=qNthElement(qCylindrical(qAllSolidBodies()),{face_index});"
    else:
        qs = "query=qCylindrical(qAllSolidBodies());"

    length_type = "FULL" if full_thread else "DISTANCE"
    r = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": {
            "btType": "BTMFeature-134",
            "namespace": _THREAD_CREATOR_NS,
            "featureType": "threadCreator",
            "name": name,
            "suppressed": False,
            "parameters": [
                {"btType": "BTMParameterQueryList-148",
                 "queries": [{"btType": "BTMIndividualQuery-138",
                              "queryString": qs, "deterministicIds": []}],
                 "filter": _THREAD_FACE_FILTER,
                 "parameterId": "refChoice"},
                {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
                 "enumName": "ThreadProfile", "value": "ISO_STD", "parameterId": "threadProfile"},
                {"btType": "BTMParameterBoolean-144", "value": left_handed,   "parameterId": "leftHanded"},
                {"btType": "BTMParameterString-149",  "value": thread_type,   "parameterId": "screwfriendlyname"},
                {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
                 "units": "", "expression": f"{major_cm} cm",  "parameterId": "majorDiameter"},
                {"btType": "BTMParameterBoolean-144", "value": is_internal,   "parameterId": "internalThreads"},
                {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
                 "enumName": "DiameterType", "value": "MAJOR", "parameterId": "diameterType"},
                {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
                 "units": "", "expression": f"{pitch_cm} cm",  "parameterId": "p"},
                {"btType": "BTMParameterBoolean-144", "value": False,         "parameterId": "oppositeEnd"},
                {"btType": "BTMParameterQuantity-147", "isInteger": True, "value": 0.0,
                 "units": "", "expression": "1",               "parameterId": "numStarts"},
                {"btType": "BTMParameterEnum-145", "namespace": _THREAD_CREATOR_NS,
                 "enumName": "LengthSelectionType", "value": length_type, "parameterId": "lengthType"},
                {"btType": "BTMParameterBoolean-144", "value": True,          "parameterId": "taperFirstEnd"},
                {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
                 "units": "", "expression": "45 deg",          "parameterId": "taperFirstEndAngle"},
                {"btType": "BTMParameterQuantity-147", "isInteger": False, "value": 0.0,
                 "units": "", "expression": "1",               "parameterId": "leadInPitches"},
            ],
            "subFeatures": [], "returnAfterSubfeatures": False,
        }
    })
    fid    = r.get("feature", {}).get("featureId")
    status = r.get("featureState", {}).get("featureStatus", "?")
    log.info("[Thread] fid=%s  status=%s", fid, status)
    if status not in ("OK", "WARNING"):
        raise RuntimeError(f"threadCreator failed: status={status}")
    return fid or name


# ── Onshape helper functions (non-MCP tools) ────────────────────────────────────

def _create_document(name: str) -> tuple[str, str]:
    doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": True})
    return doc["id"], doc["defaultWorkspace"]["id"]

def _get_part_studio(did: str, wid: str) -> str:
    els = _onshape("GET", f"/api/v9/documents/d/{did}/w/{wid}/elements")
    for el in els:
        if el.get("elementType") == "PARTSTUDIO":
            return el["id"]
    raise RuntimeError("No Part Studio found in new document")

def _get_mass_props(did: str, wid: str, eid: str) -> dict:
    raw = _onshape("GET", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties")
    bodies = raw.get("bodies", {})
    result = {"has_bodies": bool(bodies), "mass_g": None, "volume_cm3": None}
    if bodies:
        b = list(bodies.values())[0]
        mass = b.get("mass", [None])[0]
        vol  = b.get("volume", [None])[0]
        result["mass_g"]     = round(mass * 1000, 2) if mass else None
        result["volume_cm3"] = round(vol * 1e6,   2) if vol  else None
    return result

def _get_bounding_box(did: str, wid: str, eid: str) -> dict:
    raw = _onshape("GET", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/boundingboxes")
    bb  = raw.get("bodies", {})
    if not bb:
        return {}
    b = list(bb.values())[0]
    to_mm = lambda v: round(v * 1000, 3) if v is not None else None
    return {
        "min_x_mm": to_mm(b.get("lowX")),  "max_x_mm": to_mm(b.get("highX")),
        "min_y_mm": to_mm(b.get("lowY")),  "max_y_mm": to_mm(b.get("highY")),
        "min_z_mm": to_mm(b.get("lowZ")),  "max_z_mm": to_mm(b.get("highZ")),
    }


# ── Direct BTM extrude ──────────────────────────────────────────────────────────

def _build_extrude_direct(did: str, wid: str, eid: str,
                          sk_fid: str, name: str, depth_in: float,
                          op_type: str = "NEW", opposite: bool = False) -> str:
    """
    Create an extrude via direct Onshape BTM API using the same queryString format
    that the mitre cuts use (known to work). Retries up to 3 times; on failure the
    errored feature is deleted before the next attempt to avoid duplicates.
    Returns featureId.
    """
    _path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
    body = {
        "btType": "BTFeatureDefinitionCall-1406",
        "feature": {
            "btType": "BTMFeature-134",
            "featureType": "extrude",
            "name": name,
            "suppressed": False,
            "namespace": "",
            "parameters": [
                {
                    "btType": "BTMParameterQueryList-148",
                    "parameterId": "entities",
                    "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                                 "queryString": f'query=qSketchRegion(makeId("{sk_fid}"),false);'}],
                },
                {
                    "btType": "BTMParameterEnum-145",
                    "namespace": "",
                    "enumName": "NewBodyOperationType",
                    "value": op_type,
                    "parameterId": "operationType",
                },
                {
                    "btType": "BTMParameterEnum-145",
                    "namespace": "",
                    "enumName": "BoundingType",
                    "value": "BLIND",
                    "parameterId": "endBound",
                },
                {
                    "btType": "BTMParameterQuantity-147",
                    "isInteger": False,
                    "value": depth_in,
                    "expression": f"{depth_in} in",
                    "parameterId": "depth",
                },
                {
                    "btType": "BTMParameterBoolean-144",
                    "value": opposite,
                    "parameterId": "oppositeDirection",
                },
            ],
        }
    }
    last_err: Optional[Exception] = None
    last_fid: Optional[str] = None
    for attempt in range(3):
        if attempt == 0:
            time.sleep(2.0)  # give Onshape time to commit the preceding sketch
        else:
            if last_fid:
                try:
                    _onshape("DELETE", f"{_path}/featureid/{last_fid}")
                    log.info("[Extrude] Deleted failed fid=%s before retry %d", last_fid, attempt)
                except Exception as _de:
                    log.warning("[Extrude] Delete fid=%s failed: %s", last_fid, _de)
            time.sleep(1.0 * attempt)
        r: dict = {}
        try:
            r = _onshape("POST", _path, body)
            last_fid = r.get("feature", {}).get("featureId")
            fid = _check_feature_status(r, name)
            log.info("[Extrude] %s fid=%s ok", name, fid)
            return fid
        except RuntimeError as _e:
            last_err = _e
            last_fid = r.get("feature", {}).get("featureId")
            log.warning("[Extrude] %s attempt %d failed: %s", name, attempt + 1, _e)
    raise last_err or RuntimeError(f"Extrude '{name}' failed after 3 attempts")


# Standard-plane axis queries used by _build_revolve_direct.
# Onshape's standard plane IDs: JCC=Front (XY), JDC=Top (XZ), JEC=Right (YZ).
# For Y-axis revolution (most common — cups/bowls on Front plane): use the Right-plane (JEC)
# as it contains the Y-axis as its defining linear entity.
_REVOLVE_AXIS_QUERIES: dict = {
    "Y": 'query=qGeometry(makeId("JEC"), GeometryType.LINE);',   # Y-axis (vertical, for Front-plane profiles)
    "X": 'query=qGeometry(makeId("JDC"), GeometryType.LINE);',   # X-axis (horizontal, for Top-plane profiles)
    "Z": 'query=qGeometry(makeId("JCC"), GeometryType.LINE);',   # Z-axis (depth, for Right-plane profiles)
}


def _build_revolve_direct(did: str, wid: str, eid: str,
                          sk_fid: str, name: str,
                          axis: str = "Y", angle_deg: float = 360.0,
                          op_type: str = "NEW") -> str:
    """
    Create a revolve feature via direct Onshape BTM API.
    Sketch profile must lie entirely on one side of the axis (x≥0 for Y-axis).
    Returns featureId.
    """
    axis_key = axis.upper()
    axis_query = _REVOLVE_AXIS_QUERIES.get(axis_key, _REVOLVE_AXIS_QUERIES["Y"])
    revolve_type = "FULL" if angle_deg >= 360 else "ONE_DIRECTION"

    _path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
    body = {
        "btType": "BTMFeatureDefinitionCall-1406",
        "feature": {
            "btType": "BTMFeature-134",
            "featureType": "revolve",
            "name": name,
            "suppressed": False,
            "namespace": "",
            "parameters": [
                {
                    "btType": "BTMParameterQueryList-148",
                    "parameterId": "entities",
                    "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                                 "queryString": f'query=qSketchRegion(makeId("{sk_fid}"),false);'}],
                },
                {
                    "btType": "BTMParameterQueryList-148",
                    "parameterId": "axis",
                    "queries": [{"btType": "BTMIndividualQuery-138",
                                 "queryString": axis_query}],
                },
                {
                    "btType": "BTMParameterEnum-145",
                    "namespace": "",
                    "enumName": "RevolveType",
                    "value": revolve_type,
                    "parameterId": "revolveType",
                },
                {
                    "btType": "BTMParameterEnum-145",
                    "namespace": "",
                    "enumName": "NewBodyOperationType",
                    "value": op_type,
                    "parameterId": "operationType",
                },
            ],
        }
    }
    if revolve_type == "ONE_DIRECTION":
        body["feature"]["parameters"].append({
            "btType": "BTMParameterQuantity-147",
            "isInteger": False,
            "value": angle_deg,
            "expression": f"{angle_deg} deg",
            "parameterId": "angle",
        })

    last_err: Optional[Exception] = None
    last_fid: Optional[str] = None
    for attempt in range(3):
        if attempt == 0:
            time.sleep(2.0)
        else:
            if last_fid:
                try:
                    _onshape("DELETE", f"{_path}/featureid/{last_fid}")
                    log.info("[Revolve] Deleted failed fid=%s before retry %d", last_fid, attempt)
                except Exception as _de:
                    log.warning("[Revolve] Delete fid=%s failed: %s", last_fid, _de)
            time.sleep(1.0 * attempt)
        r: dict = {}
        try:
            r = _onshape("POST", _path, body)
            last_fid = r.get("feature", {}).get("featureId")
            fid = _check_feature_status(r, name)
            log.info("[Revolve] %s fid=%s ok", name, fid)
            return fid
        except RuntimeError as _e:
            last_err = _e
            last_fid = r.get("feature", {}).get("featureId")
            log.warning("[Revolve] %s attempt %d failed: %s", name, attempt + 1, _e)
    raise last_err or RuntimeError(f"Revolve '{name}' failed after 3 attempts")


# ── Tool dispatcher ─────────────────────────────────────────────────────────────

# Tools that go through the clarsbyte/onshape-mcp MCP server
# NOTE: create_gear is NOT in this set — it uses _build_gear_direct() instead
# because the repo's gear builder produces disconnected segments (57 separate bodies).
_MCP_TOOLS = {
    "create_sketch", "create_sketch_rectangle", "create_sketch_line",
    "create_sketch_circle", "create_stepped_extrude",
    "create_thicken", "create_fillet", "create_hole",
    "get_edges", "find_circular_edges", "find_edges_by_feature",
    "get_variables", "set_variable", "get_features",
}
# create_extrude is handled via direct BTM (not MCP) for reliable qSketchRegion evaluation

_REAL_FID_RE = re.compile(r'^F[A-Za-z0-9]{6,}_\d+$')


def _is_real_fid(s: str) -> bool:
    """Return True only if s looks like a genuine Onshape featureId (e.g. FZWDPmuO3wuRTzm_0)."""
    return bool(s and _REAL_FID_RE.match(s))


def _resolve_sketch_fid(params: dict, ctx: dict, tool: str) -> str:
    """
    Return the best sketchFeatureId to use:
    - If params contains a genuine Onshape featureId, use it.
    - Otherwise fall back to last_sketch_fid from context.
    Raises RuntimeError if nothing is available.
    """
    sk_fid = params.get("sketchFeatureId", "")
    if not _is_real_fid(sk_fid):
        if sk_fid and sk_fid not in ("", "AUTO_FROM_CTX", "SKETCH_ID"):
            log.info("[Tool] %s: sketchFeatureId %r looks like placeholder, using last_sketch_fid", tool, sk_fid)
        sk_fid = ctx.get("last_sketch_fid", "")
    if not sk_fid:
        raise RuntimeError(f"{tool}: no valid sketchFeatureId in params or context")
    return sk_fid


def dispatch_tool(tool: str, params: dict, ctx: dict) -> dict:
    """
    Execute a tool call and return structured result.
    ctx is the shared build context (did, wid, eid, etc.).
    MCP tools (create_gear, create_extrude, etc.) go through the repo's server.
    Non-MCP tools use direct Onshape REST calls.
    Raises RuntimeError on failure.
    """
    log.info("[Tool] %s  params=%s", tool, json.dumps(params))
    t0 = time.monotonic()

    def _did():
        return params.get("documentId")  or ctx.get("did")
    def _wid():
        return params.get("workspaceId") or ctx.get("wid")
    def _eid():
        return params.get("elementId")   or ctx.get("eid")

    result: dict = {}

    if tool == "create_document":
        did, wid = _create_document(params["name"])
        ctx["did"] = did
        ctx["wid"] = wid
        ctx["doc_name"] = params["name"]
        result = {"documentId": did, "workspaceId": wid}

    elif tool == "get_part_studio":
        eid = _get_part_studio(_did(), _wid())
        ctx["eid"] = eid
        result = {"elementId": eid}

    elif tool == "create_gear":
        full_args = dict(params)
        if "numTeeth" in full_args:
            full_args["numTeeth"] = int(full_args["numTeeth"])
        full_args.setdefault("thickness", 0.5)
        full_args.setdefault("pressureAngle", 20.0)
        full_args.setdefault("boreDiameter", 0.0)
        full_args.setdefault("centerX", 0.0)
        full_args.setdefault("centerY", 0.0)
        full_args.setdefault("plane", "Front")
        full_args.setdefault("name", f"Gear {full_args.get('numTeeth', '?')}T")
        if full_args.get("numTeeth", 0) < 4:
            raise RuntimeError("numTeeth < 4 produces undercut — refusing")
        if full_args.get("thickness", 0) <= 0:
            full_args["thickness"] = 0.5
        result = _build_gear_direct(_did(), _wid(), _eid(), full_args)

    elif tool == "create_bolt":
        full_args = dict(params)
        full_args.setdefault("size", 18.0)
        full_args.setdefault("pitch", {4: 0.7, 6: 1.0, 8: 1.25, 10: 1.5,
                                        12: 1.75, 16: 2.0, 18: 2.5, 20: 2.5,
                                        24: 3.0}.get(int(float(full_args.get("size", 18))), 2.5))
        full_args.setdefault("plane", "Top")
        full_args.setdefault("name", f"M{int(float(full_args.get('size', 18)))} Bolt")
        result = _build_bolt_direct(_did(), _wid(), _eid(), full_args)

    elif tool == "create_extrude":
        sk_fid = _resolve_sketch_fid(params, ctx, "create_extrude")
        depth_in  = float(params.get("depth", 1.0))
        op_type   = params.get("operationType", "NEW")
        opposite  = bool(params.get("oppositeDirection", False))
        name      = params.get("name", "Extrude")
        fid = _build_extrude_direct(_did(), _wid(), _eid(), sk_fid, name, depth_in, op_type, opposite)
        ctx["last_body_fid"]    = fid
        ctx["last_extrude_fid"] = fid
        result = {"featureId": fid}

    elif tool == "create_revolve":
        sk_fid = _resolve_sketch_fid(params, ctx, "create_revolve")
        axis     = params.get("axis", "Y")
        angle    = float(params.get("angle", 360.0))
        op_type  = params.get("operationType", "NEW")
        name     = params.get("name", "Revolve")
        fid = _build_revolve_direct(_did(), _wid(), _eid(), sk_fid, name, axis, angle, op_type)
        ctx["last_body_fid"] = fid
        result = {"featureId": fid}

    elif tool == "create_section":
        full_args = dict(params)
        full_args.setdefault("length_mm", 1000.0)
        full_args.setdefault("plane", "Front")
        full_args.setdefault("name", full_args.get("designation", "Section"))
        if not all(full_args.get(k) for k in ("d_mm", "bf_mm", "tf_mm", "tw_mm")):
            raise RuntimeError("create_section: missing required dimension(s) d_mm/bf_mm/tf_mm/tw_mm")
        result = _build_section_direct(_did(), _wid(), _eid(), full_args)
        if result.get("featureId"):
            ctx["last_body_fid"] = result["featureId"]

    elif tool == "mirror_part":
        plane = params.get("plane", "Right")
        operation = params.get("operationType", params.get("operation_type", "NEW"))
        name = params.get("name", "Mirror")
        fid = _apply_mirror(_did(), _wid(), _eid(), name, plane, operation,
                            body_feature_id=ctx.get("last_body_fid"))
        result = {"featureId": fid}

    elif tool == "create_thread":
        fid = _apply_thread(_did(), _wid(), _eid(), params, ctx=ctx)
        result = {"featureId": fid}

    elif tool in _MCP_TOOLS:
        # Inject context IDs if not already present
        full_args = dict(params)
        if _did() and "documentId" not in full_args:
            full_args["documentId"] = _did()
        if _wid() and "workspaceId" not in full_args:
            full_args["workspaceId"] = _wid()
        if _eid() and "elementId" not in full_args:
            full_args["elementId"] = _eid()
        # Planner can't know the sketch featureId ahead of time — inject from ctx.
        _sid = full_args.get("sketchFeatureId", "")
        _needs_inject = (
            tool in ("create_extrude", "create_hole", "create_thicken")
            and not _is_real_fid(_sid)
        )
        if _needs_inject:
            if ctx.get("last_sketch_fid"):
                full_args["sketchFeatureId"] = ctx["last_sketch_fid"]
                log.info("[Tool] Injected sketchFeatureId=%s into %s", ctx["last_sketch_fid"], tool)
            else:
                raise RuntimeError(f"{tool}: sketchFeatureId missing and no preceding sketch in ctx")
        result = _call_mcp_tool(tool, full_args)
        # Track sketch featureId so the next create_extrude can pick it up
        if tool in ("create_sketch_rectangle", "create_sketch", "create_sketch_circle",
                    "create_sketch_line") and result.get("featureId"):
            ctx["last_sketch_fid"] = result["featureId"]
            log.info("[Tool] Stored last_sketch_fid=%s", result["featureId"])
        # Track body-creating feature so mirror_part / threadCreator can reference it
        if tool == "create_extrude" and result.get("featureId"):
            ctx["last_body_fid"]    = result["featureId"]
            ctx["last_extrude_fid"] = result["featureId"]

    elif tool == "get_mass_properties":
        result = _get_mass_props(_did(), _wid(), _eid())

    elif tool == "get_bounding_box":
        result = _get_bounding_box(_did(), _wid(), _eid())

    else:
        raise RuntimeError(f"Unknown tool: {tool}")

    elapsed = time.monotonic() - t0
    log.info("[Tool] %s done in %.2fs  result=%s", tool, elapsed, json.dumps(result))
    return result


# ── Ollama helper ───────────────────────────────────────────────────────────────

def _ollama(model: str, system: str, prompt: str,
            schema: Optional[dict] = None, timeout: int = OLLAMA_TIMEOUT,
            think: bool = False, num_ctx: int = 8192) -> str:
    payload: dict = {
        "model":  model,
        "stream": False,
        "system": system,
        "prompt": prompt,
        "options": {"num_ctx": num_ctx},
        "think":  think,
    }
    if schema:
        payload["format"] = schema
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp["response"].strip()


# ── Stage 1: Planner ────────────────────────────────────────────────────────────

def _load_fewshot(shape_type: str) -> list[dict]:
    """Load top MAX_FEWSHOT feedback examples matching shape_type."""
    if not FEEDBACK_FILE.exists():
        return []
    matches = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("shape_type") == shape_type:
                    matches.append(row)
            except Exception:
                pass
    matches.sort(key=lambda x: x.get("rating", 0), reverse=True)
    return matches[:MAX_FEWSHOT]


# ── Gear geometry pre-computation (keeps planner prompt clean) ─────────────────

def _gear_plan_params(spec_lower: str) -> Optional[dict]:
    """
    If spec mentions gears, parse tooth counts + module from common patterns
    and return pre-computed step params for the planner to embed.
    Returns None if no gear detected.
    """
    # Look for tooth count + module
    teeth_m  = re.search(r"(\d+)\s*(?:teeth|tooth|t\b)", spec_lower)
    module_m = re.search(r"module\s*([0-9.]+)", spec_lower)
    ratio_m  = re.search(r"(\d+)\s*:\s*1", spec_lower)

    if not (teeth_m and module_m):
        return None

    n_gear  = int(teeth_m.group(1))
    mod     = float(module_m.group(1))
    ratio   = int(ratio_m.group(1)) if ratio_m else 3

    # For N_gear:N_pinion = ratio:1  →  N_pinion = N_gear / ratio (rounded)
    # Try to get an integer: use N_pinion = round(N_gear / ratio)
    # 17T is the minimum undercut-free pinion at 20° pressure angle
    n_pinion = max(17, round(n_gear / ratio))
    # Recompute actual ratio hint
    n_gear_adj = n_pinion * ratio  # adjusted gear teeth for exact ratio

    r_gear   = mod * n_gear_adj / 2
    r_pinion = mod * n_pinion    / 2
    center   = r_gear + r_pinion  # center distance (mm)

    center_in = center / 25.4  # mm → inches for centerX

    return {
        "gear":   {"numTeeth": n_gear_adj, "module": mod,
                   "centerX": 0.0, "centerY": 0.0,
                   "thickness": 0.5,
                   "name": f"Gear {n_gear_adj}T"},
        "pinion": {"numTeeth": n_pinion, "module": mod,
                   "centerX": round(center_in, 4), "centerY": 0.0,
                   "thickness": 0.5,
                   "name": f"Pinion {n_pinion}T"},
        "shape_type": "gear",
    }


_GENERIC_BEAM_RE = re.compile(
    r'\b(i[\s-]?beam|i[\s-]?section|h[\s-]?beam|h[\s-]?section|wide[\s-]?flange)\b',
    re.IGNORECASE,
)

# IPE 200 defaults used when no designation code is supplied
_IPE200 = {"d_mm": 200.0, "bf_mm": 100.0, "tf_mm": 8.5, "tw_mm": 5.6}


def _structural_plan_params(spec: str) -> Optional[dict]:
    """
    Detect a structural section designation in spec (e.g. 250UC72.9, 310UB46.2).
    Also detects generic keywords like "i beam" and uses IPE 200 defaults.
    Returns a hint dict for the planner, or None if no beam detected.
    """
    m = re.search(r'\b(\d{2,3}[A-Z]{2,3}[\d.]+)\b', spec.upper())
    if m:
        designation = m.group(1)
        dims = _search_section_props(designation)
        if not dims:
            return None
    elif _GENERIC_BEAM_RE.search(spec):
        designation = "IPE200"
        dims = dict(_IPE200)
    else:
        return None

    # Extract beam length from spec
    len_mm_m = re.search(r'(\d+(?:\.\d+)?)\s*mm', spec)
    len_m_m  = re.search(r'(\d+(?:\.\d+)?)\s*m\b', spec)
    if len_mm_m:
        length_mm = float(len_mm_m.group(1))
    elif len_m_m:
        length_mm = float(len_m_m.group(1)) * 1000
    else:
        length_mm = 1000.0

    mitre_words = {"mitre", "miter", "45", "triangle", "mate", "joint", "corner", "butt"}
    mitre_ends = any(w in spec.lower() for w in mitre_words)

    return {
        "designation": designation,
        "length_mm":   length_mm,
        "mitre_ends":  mitre_ends,
        "shape_type":  "i_beam",
        **dims,
    }


_PARAM_ALIASES = {
    # Map LLM snake_case output → repo's camelCase param names
    "num_teeth":          "numTeeth",
    "num teeth":          "numTeeth",
    "numin teeth":        "numTeeth",
    "number_teeth":       "numTeeth",
    "numteeth":           "numTeeth",
    "teeth":              "numTeeth",
    "module_mm":          "module",
    "pressure_angle":     "pressureAngle",
    "pressure_angle_deg": "pressureAngle",
    "pressure angle":     "pressureAngle",
    "thickness_mm":       "thickness",
    "bore_diameter":      "boreDiameter",
    "bore_diameter_mm":   "boreDiameter",
    "bore":               "boreDiameter",
    "center_x":           "centerX",
    "center_x_mm":        "centerX",
    "centerx":            "centerX",
    "center_y":           "centerY",
    "center_y_mm":        "centerY",
    "centery":            "centerY",
    "sketch_feature_id":  "sketchFeatureId",
    "sketchfeatureid":    "sketchFeatureId",
    "sketch_id":          "sketchFeatureId",
    "operation_type":     "operationType",
    "operationtype":      "operationType",
    "fillet_type":        "filletType",
    "fillettype":         "filletType",
    "edge_ids":           "edgeIds",
    "edgeids":            "edgeIds",
    "corner_1":           "corner1",
    "corner_2":           "corner2",
    "borediameter":       "boreDiameter",
    "bore_diameter":      "boreDiameter",
    "bore_diameter_mm":   "boreDiameter",
    "pressureangle":      "pressureAngle",
    "pressure_angle":     "pressureAngle",
    "pressure_angle_deg": "pressureAngle",
    "numeeth":            "numTeeth",
    "depth_mm":           "depth",
    "radius_mm":          "radius",
    "oppositedirection":  "oppositeDirection",
    "opposite_direction": "oppositeDirection",
    "wall_thickness":     "wallThickness",
    "wallthickness":      "wallThickness",
}

_INT_PARAMS   = {"numTeeth", "numStarts"}
_ARRAY_PARAMS = {"corner1", "corner2", "edgeIds"}


def _sanitize_plan(raw: dict) -> dict:
    """
    Post-process planner output:
    - Remove garbage keys (containing ':' or starting with '#' or '/')
    - Fix phi3 mangled key names via alias table
    - Strip placeholder values like '<DOCUMENT_ID>'
    - Coerce string numerics to float
    """
    def clean_params(p: dict) -> dict:
        out: dict = {}
        for k, v in p.items():
            ks = k.strip()
            if ":" in ks or ks.startswith("#") or ks.startswith("/"):
                continue
            norm = ks.lower().replace("-", "_")
            norm = _PARAM_ALIASES.get(norm, norm)
            if isinstance(v, str):
                vs = v.strip()
                if re.match(r"^<[A-Z_]+>$", vs):
                    continue
                if not vs:
                    continue
                try:
                    v = float(vs)
                except ValueError:
                    v = vs
            # Enforce integer params (LLM often emits 20.0 instead of 20)
            if norm in _INT_PARAMS and isinstance(v, float):
                v = int(v)
            # Coerce array params from dict {x:..., y:...} or scalar to [x, y]
            if norm in _ARRAY_PARAMS:
                if isinstance(v, dict):
                    v = [float(v.get("x", v.get("0", 0.0))),
                         float(v.get("y", v.get("1", 0.0)))]
                elif isinstance(v, (int, float)):
                    v = [float(v), 0.0]
            if norm not in out:
                out[norm] = v
        return out

    steps = []
    for s in raw.get("steps", []):
        if not isinstance(s, dict) or "tool" not in s:
            continue
        steps.append({
            "tool":            s["tool"],
            "params":          clean_params(s.get("params") or {}),
            "validates_after": bool(s.get("validates_after", False)),
        })

    return {
        "steps":      steps,
        "max_calls":  min(int(raw.get("max_calls", MAX_PLAN_STEPS)), MAX_PLAN_STEPS),
        "timeout_s":  min(int(raw.get("timeout_s",  PLAN_TIMEOUT_S)),  PLAN_TIMEOUT_S),
        "shape_type": raw.get("shape_type", ""),
        "expected_mass_g": raw.get("expected_mass_g"),
        "expected_max_dim_mm": raw.get("expected_max_dim_mm"),
    }


_PARAMETRIC_KEYWORDS = frozenset([
    "gear", "pinion", "sprocket",
    "bolt", "screw", "nut", "fastener",
    "i-beam", "i beam", "ipe", "hea", "heb", "ucb", "c-channel", "c channel",
    "naca", "airfoil", "aerofoil",
])


def _plan_freeform_nl(spec: str) -> str:
    """
    Stage 1a: Ask the model to decompose the object in plain English before JSON.
    Returns a natural-language CAD plan that Stage 1b can convert to JSON.
    This sidesteps the think=True + structured-output incompatibility.
    """
    system = (
        "You are a CAD decomposition assistant. Given an object name, describe exactly how to "
        "build it in Onshape using ONLY these operations:\n"
        "  - create_sketch_rectangle(corner1, corner2, plane)  — flat rect profile\n"
        "  - create_sketch_circle(center, radius, plane)       — flat circle profile\n"
        "  - create_extrude(depth_in_inches, operationType)    — push sketch into 3D solid\n"
        "  - create_revolve(axis)                              — spin sketch 360° around Y-axis\n"
        "  - create_thicken(thickness, operationType=REMOVE)   — hollow a solid\n\n"
        "RULES:\n"
        "  1. Break the object into individual solid bodies (one sketch + one operation each).\n"
        "  2. For symmetric rounded shapes (bowl, cup, vase, ring): sketch the right-half profile "
        "on the Front plane (x ≥ 0) then create_revolve around Y.\n"
        "  3. For flat shapes (seat, shelf, leg, plate): sketch_rectangle on Top plane then extrude.\n"
        "  4. Legs/feet below a surface: extrude with oppositeDirection=true.\n"
        "  5. Give realistic dimensions in inches. Use whole numbers or simple fractions.\n"
        "  6. Be concrete and terse — list each body on one line with its operation.\n\n"
        "Example for 'spoon':\n"
        "  Bowl: sketch half-oval profile on Front plane from (0,0) to (0.75,0.4); revolve Y-axis.\n"
        "  Handle: sketch_rectangle on Front plane corner1=(-0.12,0.4) corner2=(0.12,4.5); "
        "extrude depth=0.15 NEW.\n"
    )
    log.info("[Planner-NL] Decomposing freeform spec: %r", spec)
    try:
        nl = _ollama(PLANNER_MODEL, system, f"Object: {spec}",
                     schema=None, timeout=120, think=False, num_ctx=4096)
        log.info("[Planner-NL] Got %d chars: %s", len(nl), nl[:300])
        return nl.strip()
    except Exception as exc:
        log.warning("[Planner-NL] Failed (%s) — continuing without NL pre-plan", exc)
        return ""


def run_planner(spec: str, shape_type_hint: str = "") -> dict:
    """
    Stage 1: phi3:latest → JSON plan.
    Returns sanitized plan dict or raises ValueError on invalid JSON.
    """
    spec_lower = spec.lower()

    # Fast-path for gears: pre-compute geometry, give the model a concrete template
    gear_params    = _gear_plan_params(spec_lower)
    section_params = _structural_plan_params(spec) if not gear_params else None

    fewshot = _load_fewshot(
        shape_type_hint
        or (gear_params    or {}).get("shape_type", "")
        or (section_params or {}).get("shape_type", "")
    )
    fewshot_text = ""
    if fewshot:
        fewshot_text = "\n\nSUCCESSFUL PAST EXAMPLES (rated 4-5★ by user):\n"
        for ex in fewshot[:2]:
            fewshot_text += f"spec: {ex.get('spec','')}\n"
            fewshot_text += f"steps: {json.dumps([s for s in ex.get('plan_steps',[])])}\n"
            # Show feedback chain if this example was refined through chat
            comment = ex.get("comment", "")
            if "iterations:" in comment:
                chain_part = comment.split("iterations:")[-1].strip()
                fewshot_text += f"refinement_history: {chain_part}\n"
            fewshot_text += "\n"

    # ── RAG retrieval (tiered) ─────────────────────────────────────────────────
    # Injection order: Tier3 (feedback/orientation) → Tier2 (internet) → Tier1 (primitives/closest to schema)
    rag_tier3 = rag_tier2 = rag_tier1 = ""
    if RAG_ENABLED:
        try:
            tiered = _rag_query_tiered(spec)

            def _fmt_tier(chunks, header):
                if not chunks:
                    return ""
                out = f"\n\n{header}\n"
                for c in chunks:
                    src  = c.get("metadata", {}).get("source", "?")
                    dist = c.get("distance", 0.0)
                    out += f"--- source={src}  similarity={1 - dist:.2f} ---\n"
                    out += c["text"].strip() + "\n"
                return out

            rag_tier3 = _fmt_tier(tiered.get("tier3", []),
                                  "REFERENCE BUILD (orientation only — do not copy blindly):")
            rag_tier2 = _fmt_tier(tiered.get("tier2", []),
                                  "INTERNET EXAMPLES (inspiration — verify params before use):")
            rag_tier1 = _fmt_tier(tiered.get("tier1", []),
                                  "FEATURE PRIMITIVES (high trust — use these exact BTM patterns):")
        except Exception:
            pass

    # For freeform objects (not gear/beam parametric), run NL pre-planning first.
    # This gives the JSON stage a concrete decomposition to translate rather than
    # having to reason about shape AND output structured JSON simultaneously.
    is_parametric = any(kw in spec_lower for kw in _PARAMETRIC_KEYWORDS)
    freeform_nl_hint = ""
    if not is_parametric and not gear_params and not section_params:
        nl_plan = _plan_freeform_nl(spec)
        if nl_plan:
            freeform_nl_hint = f"\n\nNL PRE-PLAN (convert this to JSON steps — use exact dimensions given):\n{nl_plan}\n"

    # Build geometry hints (pre-computed / web-fetched values)
    hints = ""
    if gear_params:
        g, p = gear_params["gear"], gear_params["pinion"]
        center_in = p["centerX"]
        center_mm = center_in * 25.4
        hints = (
            f"\nPRE-COMPUTED GEAR GEOMETRY (use these exact values — centerX is in INCHES):\n"
            f"  gear:   numTeeth={g['numTeeth']} (integer!), module={g['module']} (mm), "
            f"centerX={g['centerX']}, centerY={g['centerY']}, thickness={g['thickness']} (inches)\n"
            f"  pinion: numTeeth={p['numTeeth']} (integer!), module={p['module']} (mm), "
            f"centerX={center_in:.4f} (inches), centerY={p['centerY']}, thickness={p['thickness']} (inches)\n"
            f"  (center distance = {center_mm:.2f} mm = {center_in:.4f} inches)\n"
        )
    elif section_params:
        sp = section_params
        designation_note = (
            f"for {sp['designation']}"
            if sp["designation"] != "IPE200"
            else "(IPE200 defaults — user gave no designation)"
        )
        hints = (
            f"\nI-BEAM DETECTED — use create_section. "
            f"Geometry pre-computed {designation_note} (all values in mm):\n"
            f"  d_mm={sp['d_mm']}  bf_mm={sp['bf_mm']}  "
            f"tf_mm={sp['tf_mm']}  tw_mm={sp['tw_mm']}  "
            f"length_mm={sp['length_mm']}  mitre_ends={str(sp['mitre_ends']).lower()}\n"
            f"  name=\"{sp['designation']}\"\n"
            f"  shape_type: i_beam\n"
        )

    # ── Failure memory — inject what has already been tried and failed ───────────
    failure_hint = ""
    past_failures = _load_similar_failures(spec)
    if past_failures:
        failure_hint = "\n\nPREVIOUS ATTEMPTS THAT FAILED (do NOT repeat these approaches):\n"
        for i, f in enumerate(past_failures, 1):
            tools_tried = ", ".join(f.get("plan_steps", []))
            failed      = ", ".join(f.get("tools_failed", [])) or "no solid bodies produced"
            failure_hint += (
                f"  Attempt {i}: spec={f.get('spec','?')!r}  tools_tried=[{tools_tried}]\n"
                f"    FAILED because: {failed}\n"
            )
            if f.get("failed_details"):
                for d in f["failed_details"][:2]:
                    failure_hint += f"    - {d['tool']} params={d['params']} → {d['error'][:100]}\n"
        failure_hint += "  → Choose a DIFFERENT decomposition strategy than any of the above.\n"

    system = f"""You are a CAD planning engine. Output ONLY valid JSON — no markdown, no comments, no explanation.

AVAILABLE TOOLS (call in this order — always start with create_document then get_part_studio, always end with get_mass_properties):
{TOOL_SCHEMA_TEXT}

UNIT CONVERSION — ALL sketch/extrude params are in INCHES. Divide mm by 25.4:
  80mm OD   → radius = 40/25.4 = 1.575 in    (radius = half the OD)
  100mm tall → depth = 100/25.4 = 3.937 in
  3mm wall  → inner radius = (40-3)/25.4 = 1.457 in   (for 80mm OD container)
  50mm OD, 3mm wall → outer r=0.984 in, inner r=0.866 in
  200mm     → 7.874 in    150mm → 5.906 in    120mm → 4.724 in
  60mm      → 2.362 in     40mm → 1.575 in     30mm → 1.181 in
  20mm      → 0.787 in     10mm → 0.394 in      5mm → 0.197 in

CONSTRAINTS:
- max_calls: {MAX_PLAN_STEPS}
- timeout_s: {PLAN_TIMEOUT_S}
- validates_after: true for every step that creates geometry
- Do NOT include documentId/workspaceId/elementId in params — the executor injects these from context
- All numeric values must be numbers, not strings
- numTeeth MUST be an integer (not float) — use int values only
- centerX/centerY/thickness/boreDiameter/depth/radius are in INCHES (module is in mm)
- For create_extrude: use "sketchFeatureId" (not sketch_feature_id), depth in inches
- For create_sketch_rectangle: use "corner1" and "corner2" as [x,y] arrays in inches
- For create_sketch_circle: use "center" as [x,y] and "radius" in inches (NOT diameter)
- For create_sketch: use "entities" array with objects containing "type" and shape fields
- For cylinders/containers: create_sketch_circle on Top plane → create_extrude(NEW) for outer body, second create_sketch_circle(smaller radius) → create_extrude(REMOVE) for hollow interior
- shape_type: pick the closest match from: gear, i_beam, c_channel, hollow_rect, plate, round_bar, cylinder, container, naca_airfoil, bolt, freeform

SHAPE PATTERNS — identify the object's type and follow the matching recipe:

EXTRUSION PATTERN — flat profile swept to a depth (chair, table, shelf, frame, bracket, phone stand, box lid, flat plate):
  Each solid body = one sketch + one extrude. Top-plane coord: X=left/right, Y=front/back.
  Use oppositeDirection=true for any part that goes downward (legs, feet, pegs).
  Example — chair (seat 14"×12"×0.75", legs 1.5"×1.5"×17", back 14"×16"×0.75"):
    1. Seat:   create_sketch_rectangle corner1=[-7,-6] corner2=[7,6] plane=Top
               → create_extrude depth=0.75 operationType=NEW name="Seat"
    2. Leg FL: create_sketch_rectangle corner1=[-6.5,-5.5] corner2=[-5,-4] plane=Top
               → create_extrude depth=17 operationType=NEW oppositeDirection=true name="Leg_FL"
    3. Leg FR: create_sketch_rectangle corner1=[5,-5.5] corner2=[6.5,-4] plane=Top
               → create_extrude depth=17 operationType=NEW oppositeDirection=true name="Leg_FR"
    4. Leg BL: create_sketch_rectangle corner1=[-6.5,4] corner2=[-5,5.5] plane=Top
               → create_extrude depth=17 operationType=NEW oppositeDirection=true name="Leg_BL"
    5. Leg BR: create_sketch_rectangle corner1=[5,4] corner2=[6.5,5.5] plane=Top
               → create_extrude depth=17 operationType=NEW oppositeDirection=true name="Leg_BR"
    6. Back:   create_sketch_rectangle corner1=[-7,5] corner2=[7,5.75] plane=Front
               → create_extrude depth=16 operationType=NEW name="Back"

REVOLUTION PATTERN — half-profile on Front plane spun around Y-axis (cup, bowl, vase, mug body, spoon bowl, knob, ring, jar, bottle):
  RULE: sketch ONLY the right half of the profile (x ≥ 0). Y axis is the revolution axis.
  Front-plane coords: X=horizontal (radius), Y=vertical (height). All in INCHES.
  Use create_revolve with axis="Y" (default — no need to specify if Y).
  Example — simple cup (outer radius 1.5", height 3.5", wall 0.1"):
    1. create_sketch_rectangle corner1=[0,0] corner2=[1.5,3.5] plane=Front name="CupOuter"
       → create_revolve operationType=NEW name="CupBody"
    2. create_sketch_rectangle corner1=[0,0.1] corner2=[1.4,3.5] plane=Front name="CupInner"
       → create_revolve operationType=REMOVE name="CupHollow"
  Example — spoon (bowl radius 0.75", handle 4" long × 0.25" wide × 0.15" thick):
    1. Bowl:   create_sketch_rectangle corner1=[0,0] corner2=[0.75,0.5] plane=Front name="BowlProfile"
               → create_revolve operationType=NEW name="Bowl"
    2. Handle: create_sketch_rectangle corner1=[-0.125,-0.5] corner2=[0.125,4.5] plane=Front name="HandleProfile"
               → create_extrude depth=0.15 operationType=NEW name="Handle"

SHELL PATTERN — solid box hollowed to thin walls (phone case, enclosure, tray, open box):
  1. Extrude the full solid block first.
  2. Use create_thicken on the top face with operationType=REMOVE to scoop out the interior.
  Phone case example (3"×6"×0.5" block, 0.06" walls):
    1. create_sketch_rectangle corner1=[-1.5,-3] corner2=[1.5,3] plane=Top name="CaseBase"
       → create_extrude depth=0.5 operationType=NEW name="CaseBlock"
    2. create_thicken wallThickness=0.06 operationType=REMOVE name="CaseHollow"

COMBINATION — mix patterns for multi-component objects:
  mug: body=revolution + handle=extrude(ADD) rectangle on Front plane offset from centre
  teapot: body=revolution + spout=extrude + handle=extrude
  spoon: bowl=revolution + handle=extrude (shown above)

NEVER output an empty steps array — always plan at least one geometry step.
{hints}{fewshot_text}{rag_tier3}{rag_tier2}{rag_tier1}{freeform_nl_hint}{failure_hint}
OUTPUT SCHEMA:
{{"steps":[{{"tool":"<name>","params":{{...}},"validates_after":true|false}}],"max_calls":<int>,"timeout_s":<int>,"shape_type":"<str>","expected_mass_g":<number|null>}}"""

    prompt = f"Build plan for: {spec}"

    log.info("[Planner] Calling %s ...", PLANNER_MODEL)

    plan_schema = {
        "type": "object",
        "properties": {
            "steps":           {"type": "array"},
            "max_calls":       {"type": "integer"},
            "timeout_s":       {"type": "integer"},
            "shape_type":      {"type": "string"},
            "expected_mass_g": {"type": ["number", "null"]},
        },
        "required": ["steps", "max_calls", "timeout_s"],
    }
    raw_text = _ollama(PLANNER_MODEL, system, prompt, plan_schema,
                       timeout=PLANNER_TIMEOUT, num_ctx=16384)
    log.info("[Planner] Raw output length=%d", len(raw_text))

    text = re.sub(r"^```[a-z]*\n?", "", raw_text).rstrip("`").strip()
    raw  = json.loads(text)   # raises ValueError on invalid JSON → triggers fallback
    plan = _sanitize_plan(raw)

    # Drop consecutive steps with the same tool + name — planner sometimes emits duplicates
    deduped: list = []
    for step in plan.get("steps", []):
        if (not deduped
                or step.get("tool") != deduped[-1].get("tool")
                or step.get("params", {}).get("name") != deduped[-1].get("params", {}).get("name")):
            deduped.append(step)
    if len(deduped) < len(plan.get("steps", [])):
        log.warning("[Planner] Removed %d duplicate step(s)",
                    len(plan["steps"]) - len(deduped))
    plan["steps"] = deduped

    log.info("[Planner] Plan: %d steps, shape=%s",
             len(plan.get("steps", [])), plan.get("shape_type", "?"))
    log.info("[Planner] Full plan:\n%s", json.dumps(plan, indent=2))

    geometry_tools = {
        "create_sketch", "create_sketch_circle", "create_sketch_rectangle",
        "create_extrude", "create_revolve", "create_gear", "create_bolt", "create_section",
        "create_fillet", "create_thread", "create_hole", "create_thicken",
        "create_stepped_extrude", "mirror_part",
    }
    has_geometry = any(s["tool"] in geometry_tools for s in plan.get("steps", []))
    if not has_geometry:
        raise ValueError(
            f"Planner returned no geometry steps for shape_type={plan.get('shape_type')!r}. "
            "Try a more specific description with dimensions and shape details."
        )

    return plan


# ── Stage 2: Executor ───────────────────────────────────────────────────────────

_VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "ok":      {"type": "boolean"},
        "reason":  {"type": "string"},
        "adjusted_params": {"type": "object"},
    },
    "required": ["ok", "reason"],
}


def _validate_step(step: dict, result: dict) -> bool:
    """Fast heuristic validation — no LLM call needed for simple cases."""
    tool = step["tool"]

    # Structural checks
    if tool in ("create_document",) and not result.get("documentId"):
        return False
    if tool in ("get_part_studio",) and not result.get("elementId"):
        return False
    if tool in ("create_gear", "create_bolt", "create_section",
                "create_sketch_rectangle", "create_extrude", "create_revolve", "create_fillet"):
        if not result.get("featureId"):
            return False
    if tool == "get_mass_properties":
        if not result.get("has_bodies"):
            return False

    return True


def _llm_adjust_params(step: dict, error: str, ctx: dict) -> dict:
    """Ask qwen2.5:14b to suggest adjusted params for a retry."""
    system = (
        "You are a CAD debugging assistant. A tool call failed. "
        "Return ONLY a JSON object with the adjusted 'params' key containing the fixed parameters. "
        "Do not change the tool name."
    )
    prompt = (
        f"Tool: {step['tool']}\n"
        f"Original params: {json.dumps(step['params'])}\n"
        f"Error: {error}\n"
        f"Context: {json.dumps({k: v for k, v in ctx.items() if k in ('did','wid','eid')})}\n"
        "Return adjusted params as JSON: {{\"params\": {{...}}}}"
    )
    try:
        raw = _ollama(EXECUTOR_MODEL, system, prompt, timeout=30)
        parsed = json.loads(re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip())
        return parsed.get("params", step["params"])
    except Exception as e:
        log.warning("[Executor] LLM param adjustment failed: %s", e)
        return step["params"]


def run_executor(plan: dict, out_ctx: Optional[dict] = None) -> tuple[dict, dict, list]:
    """
    Stage 2: Execute plan step by step.
    Returns (final_result, build_context, step_trace).
    step_trace: list of per-step dicts with tool, success, error, elapsed_s, retry.
    Raises RuntimeError with fallback message if >MAX_CONSECUTIVE_FAILURES.
    out_ctx: if provided, populated in-place so caller can inspect did/wid after an exception.
    """
    ctx: dict         = out_ctx if out_ctx is not None else {}
    consecutive_fails = 0
    last_result: dict = {}
    step_trace: list  = []

    steps = plan.get("steps", [])

    # Guard: planner sometimes omits the mandatory setup steps
    tool_names = [s["tool"] for s in steps]
    if "create_document" not in tool_names:
        doc_name = plan.get("shape_type", "CAD Build").replace("_", " ").title() or "CAD Build"
        prefix = [
            {"tool": "create_document",  "params": {"name": doc_name}, "validates_after": True},
            {"tool": "get_part_studio",  "params": {},                 "validates_after": True},
        ]
        steps = prefix + [s for s in steps if s["tool"] != "get_part_studio"]
        log.warning("[Executor] Auto-prepended create_document + get_part_studio (planner omitted them)")

    if len(steps) > MAX_PLAN_STEPS:
        log.warning("[Executor] Plan has %d steps, capping at %d", len(steps), MAX_PLAN_STEPS)
        steps = steps[:MAX_PLAN_STEPS]

    for i, step in enumerate(steps):
        tool   = step["tool"]
        params = dict(step.get("params", {}))
        log.info("[Executor] Step %d/%d: %s", i + 1, len(steps), tool)

        # Propagate context into params if not already set
        for key in ("documentId", "workspaceId", "elementId"):
            ctx_key = {"documentId": "did", "workspaceId": "wid", "elementId": "eid"}.get(key)
            if ctx_key and ctx.get(ctx_key) and key not in params:
                params[key] = ctx[ctx_key]

        # Execute with wall-clock budget (direct builders get a larger budget)
        budget = TOOL_TIMEOUT_DIRECT if tool in ("create_gear", "create_bolt", "create_section") else TOOL_TIMEOUT_S
        t0 = time.monotonic()
        step_record = {"step": i + 1, "tool": tool, "params": params,
                       "success": False, "retry": False, "error": None, "elapsed_s": 0.0}
        try:
            result = dispatch_tool(tool, params, ctx)
            elapsed = time.monotonic() - t0
            step_record["elapsed_s"] = round(elapsed, 2)
            if elapsed > budget:
                raise RuntimeError(
                    f"FALLBACK: tool {tool!r} exceeded {budget}s wall-clock budget "
                    f"({elapsed:.1f}s)"
                )
            step_record["success"] = True
        except RuntimeError as e:
            step_record["elapsed_s"] = round(time.monotonic() - t0, 2)
            if str(e).startswith("FALLBACK:"):
                step_trace.append(step_record)
                raise
            step_record["error"] = str(e)
            log.warning("[Executor] Step %d failed: %s", i + 1, e)
            consecutive_fails += 1
            if consecutive_fails > MAX_CONSECUTIVE_FAILURES:
                step_trace.append(step_record)
                raise RuntimeError(
                    f"FALLBACK: {consecutive_fails} consecutive tool failures — "
                    f"last error on {tool!r}: {e}"
                )
            # Retry once with LLM-adjusted params
            log.info("[Executor] Retrying step %d with LLM-adjusted params...", i + 1)
            step_record["retry"] = True
            adjusted = _llm_adjust_params(step, str(e), ctx)
            for key in ("documentId", "workspaceId", "elementId"):
                ctx_key = {"documentId": "did", "workspaceId": "wid", "elementId": "eid"}.get(key)
                if ctx_key and ctx.get(ctx_key) and key not in adjusted:
                    adjusted[key] = ctx[ctx_key]
            t1 = time.monotonic()
            try:
                result = dispatch_tool(tool, adjusted, ctx)
                step_record["elapsed_s"] = round(time.monotonic() - t1, 2)
                step_record["success"] = True
                step_record["error"] = None
            except RuntimeError as e2:
                step_record["elapsed_s"] = round(time.monotonic() - t1, 2)
                step_record["error"] = str(e2)
                log.error("[Executor] Retry also failed: %s", e2)
                consecutive_fails += 1
                if consecutive_fails > MAX_CONSECUTIVE_FAILURES:
                    step_trace.append(step_record)
                    raise RuntimeError(
                        f"FALLBACK: {consecutive_fails} consecutive failures — "
                        f"retry of {tool!r} also failed: {e2}"
                    )
                result = {"error": str(e2), "skipped": True}
        else:
            consecutive_fails = 0  # reset on success

        step_trace.append(step_record)
        last_result = result

        # Validate step output if flagged
        if step.get("validates_after") and not _validate_step(step, result):
            log.warning("[Executor] Validation failed for step %d (%s): %s",
                        i + 1, tool, result)

    return last_result, ctx, step_trace


# ── Repair pass ─────────────────────────────────────────────────────────────────

def repair_pass(ctx: dict, plan: dict) -> dict:
    """
    After build: read back mass + bounding box, compare to expected from plan.
    If >MASS_DEVIATION_PCT deviation, attempt one rebuild of geometry steps.
    Returns {"repaired": bool, "mass_g": ..., "volume_cm3": ..., "deviation_pct": ...}
    """
    did = ctx.get("did"); wid = ctx.get("wid"); eid = ctx.get("eid")
    if not all([did, wid, eid]):
        return {"repaired": False, "error": "missing did/wid/eid"}

    log.info("[Repair] Reading mass properties...")
    mp = _get_mass_props(did, wid, eid)
    bb = _get_bounding_box(did, wid, eid)
    log.info("[Repair] mass=%s g  volume=%s cm³  bbox=%s",
             mp.get("mass_g"), mp.get("volume_cm3"), bb)

    expected_mass = plan.get("expected_mass_g")
    repaired = False

    if expected_mass and mp.get("mass_g") is not None:
        actual = mp["mass_g"]
        deviation = abs(actual - expected_mass) / max(expected_mass, 1e-9)
        log.info("[Repair] Expected mass: %.1f g  Actual: %.1f g  Deviation: %.0f%%",
                 expected_mass, actual, deviation * 100)

        if deviation > MASS_DEVIATION_PCT:
            log.warning("[Repair] Mass deviation %.0f%% > %.0f%% — attempting repair",
                        deviation * 100, MASS_DEVIATION_PCT * 100)
            # Delete features and re-run geometry steps
            feat_path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
            feat_data = _onshape("GET", feat_path)
            fids = [f["featureId"] for f in feat_data.get("features", []) if f.get("featureId")]
            for fid in reversed(fids):
                try:
                    _onshape("DELETE", f"{feat_path}/featureid/{fid}")
                except Exception:
                    pass
            time.sleep(1)
            log.info("[Repair] Cleared %d features, rebuilding...", len(fids))
            try:
                _, new_ctx, _ = run_executor(plan)
                ctx.update(new_ctx)
                mp2 = _get_mass_props(did, wid, eid)
                mp  = mp2
                repaired = True
                log.info("[Repair] Repaired — new mass: %s g", mp.get("mass_g"))
            except Exception as e:
                log.error("[Repair] Repair attempt failed: %s", e)
    else:
        deviation = None

    return {
        "repaired":      repaired,
        "mass_g":        mp.get("mass_g"),
        "volume_cm3":    mp.get("volume_cm3"),
        "has_bodies":    mp.get("has_bodies", False),
        "deviation_pct": round(deviation * 100, 1) if deviation is not None else None,
        "bounding_box":  bb,
    }


# ── Feedback store ──────────────────────────────────────────────────────────────

def _load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    rows = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def store_feedback(build_result: dict, rating: int, comment: str = "") -> None:
    """Store a 4-5 star result to cad-feedback.jsonl (cap 50, deduplicate by shape_type)."""
    if rating < 4:
        log.info("[Feedback] Rating %d < 4 — not stored", rating)
        return

    rows = _load_feedback()

    # Deduplicate: remove existing entry with same shape_type (keep newest)
    shape = build_result.get("shape_type", "unknown")
    rows = [r for r in rows if r.get("shape_type") != shape]

    rows.append({
        # ── Identity ──────────────────────────────────────────────────
        "shape_type":     shape,
        "spec":           build_result.get("spec", ""),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "rating":         rating,
        "comment":        comment,
        # ── Plan ─────────────────────────────────────────────────────
        "plan_steps":     build_result.get("plan_steps", []),
        "plan_full":      build_result.get("plan_full"),       # full planner JSON
        # ── Execution trace (RL training signal) ─────────────────────
        "executor_trace": build_result.get("executor_trace"),  # per-step success/error/elapsed
        "tools_called":   build_result.get("tools_called", build_result.get("plan_steps", [])),
        "tools_failed":   build_result.get("tools_failed", []),
        "fallback_used":  build_result.get("fallback_used", False),
        "build_time_s":   build_result.get("build_time_s"),
        # ── Outcome ───────────────────────────────────────────────────
        "mass_g":         build_result.get("mass_g"),
        "volume_cm3":     build_result.get("volume_cm3"),
        "has_bodies":     build_result.get("has_bodies"),
        "deviation_pct":  build_result.get("deviation_pct"),
        "url":            build_result.get("url", ""),
    })

    # Cap at MAX_FEEDBACK_ROWS, drop oldest
    if len(rows) > MAX_FEEDBACK_ROWS:
        rows = rows[-MAX_FEEDBACK_ROWS:]

    with open(FEEDBACK_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info("[Feedback] Stored %d★ example for shape_type=%s", rating, shape)


def store_failure(build_result: dict, rating: int = 0, comment: str = "") -> None:
    """Append a failed or low-rated build to cad-failures.jsonl."""
    tools_failed = build_result.get("tools_failed", [])
    executor_trace = build_result.get("executor_trace", [])
    # Summarise what each failed step tried
    failed_details = []
    for t in executor_trace:
        if not t.get("success"):
            failed_details.append({
                "tool":   t.get("tool"),
                "params": {k: v for k, v in (t.get("params") or {}).items()
                           if k not in ("documentId", "workspaceId", "elementId")},
                "error":  str(t.get("error", ""))[:200],
            })
    row = {
        "shape_type":    build_result.get("shape_type", "unknown"),
        "spec":          build_result.get("spec", ""),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "rating":        rating,
        "comment":       comment or ("auto-saved: no solid bodies" if not rating else ""),
        "plan_steps":    build_result.get("plan_steps", []),
        "tools_failed":  tools_failed,
        "failed_details": failed_details,
        "has_bodies":    build_result.get("has_bodies", False),
        "url":           build_result.get("url", ""),
    }
    with open(FAILURES_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    log.info("[Failure] Logged failure (rating=%s) for spec=%r shape=%s",
             rating or "auto", row["spec"][:60], row["shape_type"])


def _load_similar_failures(spec: str, limit: int = 3) -> list[dict]:
    """
    Return the most recent failures whose spec shares keywords with the current spec.
    Used to inform the planner about what has already been tried and failed.
    """
    if not FAILURES_FILE.exists():
        return []
    spec_words = set(re.findall(r'\b[a-z]{3,}\b', spec.lower()))
    rows: list[tuple[int, dict]] = []
    try:
        with open(FAILURES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                row_words = set(re.findall(r'\b[a-z]{3,}\b', row.get("spec", "").lower()))
                overlap = len(spec_words & row_words)
                if overlap >= 1:
                    rows.append((overlap, row))
    except Exception as exc:
        log.warning("[Failure] Could not read failures file: %s", exc)
        return []
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in rows[:limit]]


# ── Telegram feedback ───────────────────────────────────────────────────────────

def _tg(method: str, params: Optional[dict] = None) -> Optional[dict]:
    token = _tg_token()
    if not token:
        return None
    url  = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params or {}).encode() if params else None
    req  = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        log.warning("[Telegram] %s failed: %s", method, e)
        return None


def send_feedback_request(chat_id: str, build_result: dict) -> None:
    """
    Send build result + rate prompt. Warns clearly if no geometry was produced.
    """
    url        = build_result.get("url", "—")
    mass       = build_result.get("mass_g", "—")
    vol        = build_result.get("volume_cm3", "—")
    has_bodies = build_result.get("has_bodies", False)

    if not has_bodies:
        text = (
            f"Build ran but produced no geometry.\n"
            f"URL: {url}\n\n"
            f"The planner couldn't figure out how to build this with the available tools.\n"
            f"Try a more specific description — add dimensions, material, or break it into parts."
        )
    else:
        text = (
            f"Build complete!\n"
            f"URL: {url}\n"
            f"Mass: {mass} g   Volume: {vol} cm³\n\n"
            f"Rate this build:\n"
            f"⭐ /cad rate 1   ⭐⭐ /cad rate 2   ⭐⭐⭐ /cad rate 3\n"
            f"⭐⭐⭐⭐ /cad rate 4   ⭐⭐⭐⭐⭐ /cad rate 5"
        )
    _tg("sendMessage", {"chat_id": chat_id, "text": text})


# ── Fallback to v1 ──────────────────────────────────────────────────────────────

def run_fallback(spec: str, reason: str) -> dict:
    """
    Delegate to onshape_cad_agent.py (v1).
    Returns dict with url, did, wid, eid, and fallback_reason.
    """
    log.warning("[Fallback] Triggered: %s", reason)
    print(f"\n[CAD v2] Falling back to v1 agent: {reason}", flush=True)

    ak, sk = _creds()
    env = os.environ.copy()
    env["ONSHAPE_ACCESS_KEY"] = ak
    env["ONSHAPE_SECRET_KEY"] = sk

    try:
        proc = subprocess.run(
            [sys.executable, str(FALLBACK_SCRIPT), "build", spec],
            capture_output=True, text=True, timeout=300, env=env,
        )
        output = proc.stdout + proc.stderr
        log.info("[Fallback] v1 output:\n%s", output)

        url = did = wid = eid = None
        for line in output.splitlines():
            if line.startswith("URL="):
                url = line[4:]
            elif line.startswith("DID="):
                did = line[4:]
            elif line.startswith("WID="):
                wid = line[4:]
            elif line.startswith("EID="):
                eid = line[4:]

        return {
            "url":            url,
            "did":            did,
            "wid":            wid,
            "eid":            eid,
            "fallback_path":  "v1",
            "fallback_reason": reason,
        }
    except Exception as e:
        log.error("[Fallback] v1 also failed: %s", e)
        return {"error": str(e), "fallback_path": "v1", "fallback_reason": reason}


# ── Session writer ──────────────────────────────────────────────────────────────

def _write_session(data: dict) -> None:
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Main pipeline ───────────────────────────────────────────────────────────────

class CADAgentV2:
    """Orchestrates the two-stage planner/executor pipeline."""

    def build(self, spec: str, chat_id: Optional[str] = None) -> dict:
        log.info("=" * 60)
        log.info("[Build] spec=%s", spec)
        log.info("=" * 60)

        fallback_reason = None
        plan            = None
        build_t0        = time.monotonic()

        # ── Stage 1: Plan ───────────────────────────────────────────
        try:
            # Quick shape-type hint for fewshot matching
            hint = ""
            if any(w in spec.lower() for w in ("gear", "pinion", "sprocket")):
                hint = "gear"
            elif re.search(r'\b\d{2,3}[A-Z]{2,3}[\d.]+\b', spec.upper()):
                hint = "i_beam"
            elif any(w in spec.lower() for w in ("beam", "channel", "plate", "bar")):
                hint = "i_beam" if "beam" in spec.lower() else "structural"

            plan = run_planner(spec, shape_type_hint=hint)
        except (ValueError, json.JSONDecodeError) as e:
            fallback_reason = f"Planner invalid JSON: {e}"
        except Exception as e:
            fallback_reason = f"Planner error: {e}"

        if fallback_reason:
            result = run_fallback(spec, fallback_reason)
            result["spec"] = spec
            result["path"] = "v1_fallback"
            _write_session(result)
            if chat_id:
                self._send_result(chat_id, result)
            return result

        # ── Stage 2: Execute ─────────────────────────────────────────
        executor_ctx: dict = {}
        step_trace: list   = []
        try:
            final, ctx, step_trace = run_executor(plan, out_ctx=executor_ctx)
        except RuntimeError as e:
            # Clean up any partially-built document before delegating to fallback
            orphan_did = executor_ctx.get("did")
            if orphan_did:
                try:
                    _onshape("DELETE", f"/api/v9/documents/{orphan_did}")
                    log.info("[Build] Deleted orphaned document %s", orphan_did)
                except Exception as de:
                    log.warning("[Build] Could not delete orphaned doc %s: %s", orphan_did, de)
            reason = str(e)[9:].strip() if str(e).startswith("FALLBACK:") else str(e)
            result = run_fallback(spec, reason)
            result["spec"] = spec
            result["path"] = "v1_fallback"
            _write_session(result)
            if chat_id:
                self._send_result(chat_id, result)
            return result

        # ── Build URL ────────────────────────────────────────────────
        did = ctx.get("did"); wid = ctx.get("wid"); eid = ctx.get("eid")
        url = f"{BASE_URL}/documents/{did}/w/{wid}/e/{eid}" if all([did, wid, eid]) else None

        # ── Repair pass ──────────────────────────────────────────────
        repair = repair_pass(ctx, plan)
        log.info("[Build] Repair pass: %s", repair)

        tools_failed = [t["tool"] for t in step_trace if not t["success"]]
        result = {
            "spec":          spec,
            "path":          "v2",
            "url":           url,
            "did":           did,
            "wid":           wid,
            "eid":           eid,
            "mass_g":        repair.get("mass_g"),
            "volume_cm3":    repair.get("volume_cm3"),
            "has_bodies":    repair.get("has_bodies"),
            "repaired":      repair.get("repaired"),
            "deviation_pct": repair.get("deviation_pct"),
            "bounding_box":  repair.get("bounding_box"),
            "shape_type":    plan.get("shape_type", hint or "unknown"),
            "plan_steps":    [s["tool"] for s in plan.get("steps", [])],
            "plan_full":     plan,
            "executor_trace": step_trace,
            "tools_called":  [t["tool"] for t in step_trace],
            "tools_failed":  tools_failed,
            "fallback_used": False,
            "build_time_s":  round(time.monotonic() - build_t0, 1),
            "built_at":      datetime.now(timezone.utc).isoformat(),
        }

        _write_session(result)
        log.info("[Build] Done: url=%s  mass=%s g  repaired=%s",
                 url, result["mass_g"], result["repaired"])

        # ── Auto-save failures & retry with failure context ──────────
        if not repair.get("has_bodies"):
            store_failure(result)
            log.info("[Build] No solid bodies — stored failure, retrying with failure context")
            # One automatic retry — the planner will see the failure in FAILURES_FILE
            try:
                plan2 = run_planner(spec, shape_type_hint=hint)
                final2, ctx2, trace2 = run_executor(plan2, out_ctx={})
                did2 = ctx2.get("did"); wid2 = ctx2.get("wid"); eid2 = ctx2.get("eid")
                url2 = f"{BASE_URL}/documents/{did2}/w/{wid2}/e/{eid2}" if all([did2, wid2, eid2]) else None
                repair2 = repair_pass(ctx2, plan2)
                if repair2.get("has_bodies"):
                    log.info("[Build] Retry succeeded: url=%s", url2)
                    result.update({
                        "url": url2, "did": did2, "wid": wid2, "eid": eid2,
                        "mass_g": repair2.get("mass_g"),
                        "has_bodies": True, "path": "v2_retry",
                        "plan_steps": [s["tool"] for s in plan2.get("steps", [])],
                    })
                    _write_session(result)
                    if chat_id and url2:
                        send_feedback_request(chat_id, result)
                    return result
                else:
                    store_failure({**result, "url": url2 or url, "spec": spec,
                                   "plan_steps": [s["tool"] for s in plan2.get("steps", [])],
                                   "executor_trace": trace2,
                                   "tools_failed": [t["tool"] for t in trace2 if not t["success"]]})
                    log.info("[Build] Retry also produced no bodies")
            except Exception as exc:
                log.warning("[Build] Auto-retry failed: %s", exc)

        # ── Telegram feedback ────────────────────────────────────────
        if chat_id and url:
            send_feedback_request(chat_id, result)

        return result

    def _send_result(self, chat_id: str, result: dict) -> None:
        path = result.get("path", "?")
        url  = result.get("url")
        if url:
            msg = f"Build complete ({path})\n{url}"
            if result.get("mass_g"):
                msg += f"\nMass: {result['mass_g']} g"
            if result.get("fallback_reason"):
                msg += f"\n⚠️ Fallback: {result['fallback_reason']}"
        else:
            msg = f"Build failed: {result.get('error', 'unknown error')}"
        _tg("sendMessage", {"chat_id": chat_id, "text": msg})


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _cmd_build(argv):
    import argparse
    p = argparse.ArgumentParser(prog="build")
    p.add_argument("spec", nargs="+")
    p.add_argument("--chat-id", default=None)
    a = p.parse_args(argv)
    spec = " ".join(a.spec)
    agent = CADAgentV2()
    result = agent.build(spec, chat_id=a.chat_id)
    print(json.dumps(result, indent=2))
    url = result.get("url")
    if url:
        print(f"\n>>> URL: {url}")
    else:
        print("\n>>> No URL — build failed (has_bodies=False or executor error)")


def _cmd_plan(argv):
    spec = " ".join(argv) if argv else ""
    plan = run_planner(spec)
    print(json.dumps(plan, indent=2))


def _cmd_test(argv):
    """test  — run the spur gear assembly stress test from the spec."""
    spec = (
        "make a spur gear 20 teeth module 2 meshing with a pinion "
        "at 3:1 ratio"
    )
    print(f"[Test] Spec: {spec}\n")
    agent = CADAgentV2()
    result = agent.build(spec)
    print("\n[Test] Result:")
    print(json.dumps(result, indent=2))
    if result.get("url"):
        print(f"\n✅ URL: {result['url']}")
    else:
        print("\n❌ No URL — check log for errors")


def _cmd_rate(argv):
    """rate <1-5> [comment...]  — store feedback for the last build."""
    if not argv:
        print(json.dumps({"error": "Usage: rate <1-5> [comment...]"}))
        return
    try:
        rating = int(argv[0])
    except ValueError:
        print(json.dumps({"error": f"Invalid rating {argv[0]!r} — must be 1-5"}))
        return
    if not (1 <= rating <= 5):
        print(json.dumps({"error": f"Rating {rating} out of range — must be 1-5"}))
        return
    comment = " ".join(argv[1:]) if len(argv) > 1 else ""

    if not SESSION_FILE.exists():
        print(json.dumps({"error": "No session on record"}))
        return
    try:
        last = json.loads(SESSION_FILE.read_text())
    except Exception as e:
        print(json.dumps({"error": f"Cannot read session: {e}"}))
        return

    if rating >= 4:
        store_feedback(last, rating, comment)
    else:
        store_failure(last, rating, comment)
    stars = "*" * rating
    print(json.dumps({"ok": True, "rating": rating, "stars": stars,
                      "spec": last.get("spec", ""), "comment": comment,
                      "stored_as": "feedback" if rating >= 4 else "failure"}))


def _revise_spec(original_spec: str, history: list, feedback: str) -> str:
    """
    Ask the LLM to distill conversation history + feedback into a clean revised CAD spec.
    Returns a single concise spec string suitable for the planner.
    """
    history_text = "\n".join(
        f"  Attempt {i + 1}: {h['spec']}"
        for i, h in enumerate(history)
    )
    system = (
        "You are a CAD specification writer. Given an original CAD request, previous attempt "
        "descriptions, and user feedback, write a single concise revised CAD specification "
        "(1-2 sentences, no markdown). Output ONLY the revised spec string, nothing else."
    )
    prompt = (
        f"Original request: {original_spec}\n"
        f"Previous attempts:\n{history_text}\n"
        f"User feedback: {feedback}\n"
        "Revised spec:"
    )
    try:
        revised = _ollama(EXECUTOR_MODEL, system, prompt, timeout=30).strip().strip('"\'')
        log.info("[Chat] Revised spec: %s", revised)
        return revised or original_spec
    except Exception as e:
        log.warning("[Chat] Spec revision failed (%s) — using original spec", e)
        return original_spec


def _cmd_chat(argv):
    """chat [spec...]  — interactive build-and-refine loop in the terminal."""
    if argv:
        spec = " ".join(argv)
    else:
        print("What do you want to build?")
        spec = input("> ").strip()
        if not spec:
            print("No spec provided.")
            return

    agent  = CADAgentV2()
    original_spec = spec
    history: list = []   # list of {"spec": str, "url": str}

    while True:
        print(f"\n[CAD] Building: {spec}", flush=True)
        result = agent.build(spec)
        url  = result.get("url") or "—"
        mass = result.get("mass_g")
        print(f"\n[CAD] Done!")
        print(f"  URL:  {url}")
        if mass is not None:
            print(f"  Mass: {mass} g")
        history.append({"spec": spec, "url": url})

        print("\nWhat needs changing? (press Enter when satisfied)")
        feedback = input("> ").strip()
        if not feedback:
            break

        spec = _revise_spec(original_spec, history, feedback)

    # Rating prompt
    print("\nRate this build (1-5) to save it as a future example, or press Enter to skip:")
    rating_str = input("> ").strip()
    if rating_str.isdigit() and 1 <= int(rating_str) <= 5:
        rating = int(rating_str)
        print("Add a comment? (or press Enter to skip)")
        comment = input("> ").strip()

        # Encode the feedback chain into the comment for richer learning context
        if len(history) > 1:
            chain = " → ".join(h["spec"] for h in history)
            full_comment = f"{comment} | iterations: {chain}".strip(" |")
        else:
            full_comment = comment

        final_result = result
        final_result["spec"] = original_spec
        store_feedback(final_result, rating, full_comment)
        print(f"[CAD] Saved {'⭐' * rating} — will inform future builds of this type.")
    else:
        print("[CAD] Skipped — no example saved.")


def _cmd_index(argv):
    """index [--source capabilities|mcp|apidocs|github|feedback|all] [--verbose]"""
    import argparse
    p = argparse.ArgumentParser(prog="index")
    p.add_argument("--source", default="all",
                   choices=["capabilities", "mcp", "apidocs", "github", "feedback", "all"])
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    if not RAG_ENABLED:
        print("ERROR: chromadb not installed. Run: python3 -m pip install chromadb --break-system-packages")
        sys.exit(1)

    from cad_rag import index_all
    sources = ["capabilities", "mcp", "apidocs", "github", "feedback"] if a.source == "all" else [a.source]
    counts  = index_all(sources, verbose=a.verbose)
    print(json.dumps({"ok": True, "indexed": counts}, indent=2))


# ── Curriculum loop ─────────────────────────────────────────────────────────────
#
# Progressive build sessions — simple specs first, complex later.
# Each spec runs: plan → (user confirms) → build → rate → re-index feedback.
# Designed for RL data collection: run many builds, rate them, let the agent improve.
#
# Spec list is tiered: Tier 1 (single op) → Tier 2 (two ops) → Tier 3 (composition).
# Within each tier, specs are ordered easy → hard.
# Complex features (threads, mitre cuts) appear ONLY after their prerequisites pass.

_CURRICULUM = [
    # ── Tier 1: single operation ─────────────────────────────────────────────────
    ("solid cylinder 60mm diameter 80mm tall",                          "cylinder"),
    ("steel round bar 25mm diameter 500mm long",                        "round_bar"),
    ("flat rectangular plate 150x100mm 10mm thick",                     "plate"),
    ("spur gear 24 teeth module 2 12mm thick",                          "gear"),
    ("M12 hex bolt 80mm shank",                                         "bolt"),
    # ── Tier 2: two operations ───────────────────────────────────────────────────
    ("hollow tube 50mm OD 3mm wall 200mm long",                         "cylinder"),
    ("round disc 100mm diameter 15mm thick with central 20mm bore",     "cylinder"),
    ("rectangular box 120x80mm 60mm tall wall 4mm open top",            "container"),
    ("plate 200x150mm 8mm thick with 10mm diameter hole at centre",     "plate"),
    ("flat plate 100x100mm 5mm thick all edges filleted 3mm radius",    "plate"),
    ("L-bracket 80x60mm legs 5mm thick 150mm long mirrored",            "hollow_rect"),
    # ── Tier 3: three+ operations (RL core targets) ──────────────────────────────
    ("cylindrical boss 40mm diameter 30mm tall on 120x80mm plate 10mm thick", "plate"),
    ("plate 160x120mm 10mm thick with 4 corner holes 8mm diameter",     "plate"),
    ("symmetric bracket 100x40mm 5mm thick 2 bolt holes 8mm filleted corners 2mm", "hollow_rect"),
    ("gear shaft 20mm diameter 150mm long with 8mm counterbore 20mm deep at one end", "round_bar"),
    ("spur gear pair 3:1 ratio module 2 20mm face width",               "gear"),
    # ── Tier 4: complex / stress tests ───────────────────────────────────────────
    ("flanged cylinder 80mm OD 50mm bore 40mm tall with 10mm flange ring at base", "cylinder"),
    ("stepped shaft 30mm then 20mm then 15mm diameter 60mm each section", "round_bar"),
    ("plate 200x150mm 8mm thick with 6-hole 60mm bolt circle holes 8mm diameter", "plate"),
    ("symmetric H-frame two 40x40mm posts 200mm tall 80mm apart joined by 20mm crossbar", "freeform"),
    # ── Tier 5: requires prerequisites (thread, mitre) ───────────────────────────
    ("hollow cylindrical container 80mm OD 3mm wall 100mm tall open top", "container"),
    ("250UC72.9 I-beam 1000mm",                                          "i_beam"),
    ("250UC72.9 I-beam 1000mm mitre ends",                               "i_beam"),
]


def _cmd_curriculum(argv):
    """
    curriculum [--start N] [--tier 1|2|3|4|5] [--plan-only]

    Progressive build loop for RL data collection.
    For each spec: shows plan → you confirm → builds → you rate → re-indexes.
    Simple specs first. Thread/mitre cuts only after prerequisites pass.
    """
    import argparse
    p = argparse.ArgumentParser(prog="curriculum")
    p.add_argument("--start",     type=int, default=1,
                   help="Start at spec number N (1-indexed, see list in BUILDS.md)")
    p.add_argument("--tier",      type=int, default=0,
                   help="Run only a specific tier (1-5). 0 = all tiers")
    p.add_argument("--plan-only", action="store_true",
                   help="Show plans only — do not build")
    a = p.parse_args(argv)

    agent   = CADAgentV2()
    specs   = _CURRICULUM[a.start - 1:]
    passed  = 0
    failed  = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"  CAD Curriculum Loop — {len(specs)} specs from #{a.start}")
    print(f"  Commands at each step: y=build  n=skip  q=quit  p=plan only")
    print(f"{'='*60}\n")

    for idx, (spec, expected_shape) in enumerate(specs, start=a.start):
        print(f"\n[{idx}/{len(_CURRICULUM)}] {spec}")
        print(f"  expected shape_type: {expected_shape}")
        print("-" * 50)

        # Always show plan first
        try:
            plan = run_planner(spec)
            tools = [s["tool"] for s in plan.get("steps", [])]
            shape = plan.get("shape_type", "?")
            print(f"  Plan ({len(tools)} steps, shape={shape}):")
            for i, t in enumerate(tools, 1):
                print(f"    {i}. {t}")
        except Exception as e:
            print(f"  Plan FAILED: {e}")
            failed += 1
            cmd = input("\n  [n=skip  q=quit] > ").strip().lower()
            if cmd == "q":
                break
            skipped += 1
            continue

        if a.plan_only:
            cmd = input("\n  [Enter=next  q=quit] > ").strip().lower()
            if cmd == "q":
                break
            continue

        cmd = input("\n  Build this? [y=yes  n=skip  q=quit] > ").strip().lower()
        if cmd == "q":
            break
        if cmd != "y":
            skipped += 1
            print("  Skipped.")
            continue

        # Build
        print(f"\n  Building: {spec}")
        try:
            result = agent.build(spec)
        except Exception as e:
            print(f"  Build FAILED: {e}")
            failed += 1
            continue

        url      = result.get("url", "no url")
        mass     = result.get("mass_g")
        has_body = result.get("has_bodies")
        path     = result.get("path", "?")

        print(f"\n  Result ({path}):")
        print(f"    URL:   {url}")
        print(f"    Mass:  {mass} g   bodies={has_body}")
        if result.get("tools_failed"):
            print(f"    FAILED tools: {result['tools_failed']}")

        # Rating
        print("\n  Rate 1-5 (or Enter to skip):")
        r = input("  > ").strip()
        if r.isdigit() and 1 <= int(r) <= 5:
            rating = int(r)
            print("  Comment (or Enter to skip):")
            comment = input("  > ").strip()
            store_feedback(result, rating, comment)
            print(f"  Saved {'⭐' * rating}")
            if rating >= 4:
                passed += 1
                # Re-index feedback after each good rating
                if RAG_ENABLED:
                    try:
                        from cad_rag import index_feedback
                        n = index_feedback(verbose=False)
                        if n:
                            print(f"  RAG updated (+{n} feedback chunks)")
                    except Exception:
                        pass
            else:
                failed += 1
        else:
            skipped += 1
            print("  No rating saved.")

    print(f"\n{'='*60}")
    print(f"  Session complete: {passed} passed  {failed} failed  {skipped} skipped")
    print(f"  RL-ready examples added: {passed}")
    print(f"{'='*60}\n")


def _cmd_refine(argv):
    """
    refine <original_spec> <feedback> [<history_json>]
    Ask the LLM to blend feedback into the spec. Prints revised spec as plain text.
    history_json: JSON array of {"spec": str, "url": str} dicts.
    """
    if len(argv) < 2:
        print("Usage: refine <original_spec> <feedback> [<history_json>]")
        sys.exit(1)
    original = argv[0]
    feedback = argv[1]
    history  = json.loads(argv[2]) if len(argv) > 2 else [{"spec": original, "url": ""}]
    revised  = _revise_spec(original, history, feedback)
    print(revised)


def _cmd_inspect(argv):
    """
    inspect <onshape_url>
    Fetch document info from Onshape and return an LLM-generated description as JSON.
    Output: {"ok": true, "description": "...", "url": "...", "did": ..., "wid": ..., "eid": ...}
    """
    if not argv:
        print(json.dumps({"ok": False, "error": "Usage: inspect <onshape_url>"}))
        sys.exit(1)
    url = argv[0]
    try:
        did, wid, eid = _parse_onshape_url(url)
    except ValueError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
    try:
        description = _describe_document(did, wid, eid)
        print(json.dumps({
            "ok": True,
            "description": description,
            "url": url,
            "did": did, "wid": wid, "eid": eid,
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)


CMDS = {
    "build":       _cmd_build,
    "plan":        _cmd_plan,
    "test":        _cmd_test,
    "rate":        _cmd_rate,
    "chat":        _cmd_chat,
    "refine":      _cmd_refine,
    "inspect":     _cmd_inspect,
    "index":       _cmd_index,
    "curriculum":  _cmd_curriculum,
}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in CMDS:
        CMDS[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Usage: {sys.argv[0]} <{'|'.join(CMDS)}> [args...]")
        print("\nCommands:")
        print("  build <spec> [--chat-id <id>]  — full two-stage pipeline")
        print("  chat  [spec...]                — interactive build-and-refine loop")
        print("  plan  <spec>                   — dry-run planner only")
        print("  test                           — spur gear 3:1 stress test")
        print("  rate  <1-5> [comment...]       — rate last build (saves 4-5★ as examples)")
        print("  index      [--source feedback|mcp|apidocs|all]  — build RAG knowledge index")
        print("  curriculum [--start N] [--tier N] [--plan-only] — progressive build loop for RL")
        sys.exit(1)

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

# ── Paths & config ─────────────────────────────────────────────────────────────

_HERE         = Path(__file__).parent
_OPENCLAW     = Path.home() / ".openclaw"
LOG_FILE      = _OPENCLAW / "cad-agent.log"
FEEDBACK_FILE = _OPENCLAW / "cad-feedback.jsonl"
SESSION_FILE  = _OPENCLAW / "cad-session.json"
FALLBACK_SCRIPT = _HERE / "onshape_cad_agent.py"
CONFIG_FILE   = _OPENCLAW / "openclaw.json"

PLANNER_MODEL  = "qwen2.5:14b"         # same GPU model — structured JSON, reliable
EXECUTOR_MODEL = "qwen2.5:14b"        # GPU — fallback: qwen2.5-coder:32b via CPU
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300   # seconds per LLM call (14b model needs headroom)

MAX_PLAN_STEPS    = 8
PLAN_TIMEOUT_S    = 120
MAX_CONSECUTIVE_FAILURES = 2
TOOL_TIMEOUT_S    = 10   # wall-clock budget per tool call before fallback

MAX_FEEDBACK_ROWS = 50
MAX_FEWSHOT       = 3
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
    return cfg.get("channels", {}).get("telegram", {}).get("botToken", "")

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
        "when": "Call once per gear when spec mentions gear, pinion, sprocket, or cog. "
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
    "create_sketch_rectangle": {
        "when": "Call to create a rectangular profile before create_extrude. "
                "Use for box-like parts: plates, beams, housings, brackets.",
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
        "when": "Call after create_sketch_rectangle to make a solid body. "
                "Use operationType=NEW for first body, ADD to merge into existing.",
        "params": {
            "sketchFeatureId": "string — featureId from the preceding sketch step",
            "depth":           "number — extrude depth in INCHES (NOT mm). E.g. 0.787 for 20mm",
            "operationType":   "string — NEW | ADD | REMOVE | INTERSECT, default NEW",
            "name":            "string",
        },
        "returns": "featureId: str",
        "failure_modes": "Fails if sketchFeatureId references a failed sketch.",
    },
    "create_fillet": {
        "when": "Call after a solid body exists to blend edges. "
                "Use when spec says fillet, radius, chamfer, rounded, or blended. "
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
    "create_bolt": {
        "when": "Call when spec mentions bolt, screw, cap screw, hex bolt, or threaded fastener. "
                "Builds hex head + cylindrical shank + ISO metric threads (ThreadCreator). "
                "Always use this instead of sketch+extrude for bolts.",
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
        m = re.search(r"Feature ID:\s*([A-Za-z0-9_\-\.]+)", tc.text)
        if m and m.group(1) != "unknown":
            result["featureId"] = m.group(1)
    return result


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

    N_INV = 8   # points per involute flank (more → smoother teeth)
    N_ARC = 5   # points per arc

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
    sk_resp   = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", sk_data)
    sk_fid    = sk_resp.get("feature", {}).get("featureId")
    sk_status = sk_resp.get("featureState", {}).get("featureStatus", "?")
    if not sk_fid:
        raise RuntimeError(f"Gear sketch submission failed (no featureId): {sk_resp}")
    log.info("[Gear] Sketch featureId=%s status=%s", sk_fid, sk_status)

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
    ex_resp   = _onshape("POST", f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features", ex_data)
    ex_fid    = ex_resp.get("feature", {}).get("featureId")
    ex_status = ex_resp.get("featureState", {}).get("featureStatus", "?")
    log.info("[Gear] Extrude featureId=%s status=%s", ex_fid, ex_status)

    return {"featureId": ex_fid, "sketchFeatureId": sk_fid, "extrudeStatus": ex_status}


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

    # Metres (Onshape SI)
    r_shank  = m_size / 2000
    r_vtx    = (af_mm / 2) / math.cos(math.radians(30)) / 1000
    head_h_m = head_h  / 1000
    shank_l_m = shank_l / 1000

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
    sk1_fid = r1.get("feature", {}).get("featureId")
    if not sk1_fid:
        raise RuntimeError(f"Bolt hex head sketch failed: {r1}")
    log.info("[Bolt] Head sketch fid=%s", sk1_fid)

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
             "expression": f"{head_h_m} m", "value": head_h_m, "units": "", "isInteger": False},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }})
    ex1_fid = r2.get("feature", {}).get("featureId")
    if not ex1_fid:
        raise RuntimeError(f"Bolt head extrude failed: {r2}")
    log.info("[Bolt] Head extrude fid=%s  status=%s",
             ex1_fid, r2.get("featureState", {}).get("featureStatus"))

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
    sk2_fid = r3.get("feature", {}).get("featureId")
    if not sk2_fid:
        raise RuntimeError(f"Bolt shank sketch failed: {r3}")
    log.info("[Bolt] Shank sketch fid=%s", sk2_fid)

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
             "expression": f"{shank_l_m} m", "value": shank_l_m, "units": "", "isInteger": False},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False,
    }})
    ex2_fid = r4.get("feature", {}).get("featureId")
    log.info("[Bolt] Shank extrude fid=%s  status=%s",
             ex2_fid, r4.get("featureState", {}).get("featureStatus"))

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
                          "queryString": "query=qGeometry(qEverything(EntityType.FACE),GeometryType.CYLINDER);",
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
    thread_fid    = r5.get("feature", {}).get("featureId")
    thread_status = r5.get("featureState", {}).get("featureStatus", "?")
    log.info("[Bolt] Thread fid=%s  status=%s", thread_fid, thread_status)

    return {
        "featureId":       thread_fid or ex2_fid,
        "headFeatureId":   ex1_fid,
        "shankFeatureId":  ex2_fid,
        "threadFeatureId": thread_fid,
        "threadStatus":    thread_status,
    }


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


# ── Tool dispatcher ─────────────────────────────────────────────────────────────

# Tools that go through the clarsbyte/onshape-mcp MCP server
# NOTE: create_gear is NOT in this set — it uses _build_gear_direct() instead
# because the repo's gear builder produces disconnected segments (57 separate bodies).
_MCP_TOOLS = {
    "create_sketch", "create_sketch_rectangle", "create_sketch_line",
    "create_sketch_circle", "create_extrude", "create_stepped_extrude",
    "create_thicken", "create_fillet", "create_hole",
    "get_edges", "find_circular_edges", "find_edges_by_feature",
    "get_variables", "set_variable", "get_features",
}


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

    elif tool in _MCP_TOOLS:
        # Inject context IDs if not already present
        full_args = dict(params)
        if _did() and "documentId" not in full_args:
            full_args["documentId"] = _did()
        if _wid() and "workspaceId" not in full_args:
            full_args["workspaceId"] = _wid()
        if _eid() and "elementId" not in full_args:
            full_args["elementId"] = _eid()
        # Planner can't know the sketch featureId ahead of time — inject from ctx
        if tool == "create_extrude" and not full_args.get("sketchFeatureId"):
            if ctx.get("last_sketch_fid"):
                full_args["sketchFeatureId"] = ctx["last_sketch_fid"]
                log.info("[Tool] Injected sketchFeatureId=%s into create_extrude", ctx["last_sketch_fid"])
            else:
                raise RuntimeError("create_extrude: sketchFeatureId missing and no preceding sketch in ctx")
        result = _call_mcp_tool(tool, full_args)
        # Track sketch featureId so the next create_extrude can pick it up
        if tool in ("create_sketch_rectangle", "create_sketch", "create_sketch_circle",
                    "create_sketch_line") and result.get("featureId"):
            ctx["last_sketch_fid"] = result["featureId"]
            log.info("[Tool] Stored last_sketch_fid=%s", result["featureId"])

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
            schema: Optional[dict] = None, timeout: int = OLLAMA_TIMEOUT) -> str:
    payload: dict = {
        "model":  model,
        "stream": False,
        "system": system,
        "prompt": prompt,
        "options": {"num_ctx": 8192},
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
    n_pinion = max(7, round(n_gear / ratio))
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
}


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
            # Skip commentary keys phi3 sometimes emits
            ks = k.strip()
            if ":" in ks or ks.startswith("#") or ks.startswith("/"):
                continue
            # Normalise key
            norm = ks.lower().replace("-", "_")
            norm = _PARAM_ALIASES.get(norm, norm)
            # Strip placeholder strings
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
            if norm not in out:  # first occurrence wins
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


def run_planner(spec: str, shape_type_hint: str = "") -> dict:
    """
    Stage 1: phi3:latest → JSON plan.
    Returns sanitized plan dict or raises ValueError on invalid JSON.
    """
    spec_lower = spec.lower()

    # Fast-path for gears: pre-compute geometry, give phi3 a concrete template
    gear_params = _gear_plan_params(spec_lower)

    fewshot = _load_fewshot(shape_type_hint or (gear_params or {}).get("shape_type", ""))
    fewshot_text = ""
    if fewshot:
        fewshot_text = "\n\nSUCCESSFUL PAST EXAMPLES:\n"
        for ex in fewshot[:2]:
            fewshot_text += f"spec: {ex.get('spec','')}\n"
            fewshot_text += f"steps: {json.dumps([s for s in ex.get('plan_steps',[])])}\n\n"

    # Build geometry hints section (pre-computed values from Python math)
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

    system = f"""You are a CAD planning engine. Output ONLY valid JSON — no markdown, no comments, no explanation.

AVAILABLE TOOLS (call in this order — always start with create_document then get_part_studio, always end with get_mass_properties):
{TOOL_SCHEMA_TEXT}

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
- shape_type: pick the closest match from: gear, i_beam, c_channel, hollow_rect, plate, round_bar, naca_airfoil, bolt, freeform
{hints}{fewshot_text}
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
    raw_text = _ollama(PLANNER_MODEL, system, prompt, plan_schema)
    log.info("[Planner] Raw output length=%d", len(raw_text))

    text = re.sub(r"^```[a-z]*\n?", "", raw_text).rstrip("`").strip()
    raw  = json.loads(text)   # raises ValueError on invalid JSON → triggers fallback
    plan = _sanitize_plan(raw)

    log.info("[Planner] Plan: %d steps, shape=%s",
             len(plan.get("steps", [])), plan.get("shape_type", "?"))
    log.info("[Planner] Full plan:\n%s", json.dumps(plan, indent=2))
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
    if tool in ("create_gear", "create_bolt", "create_sketch_rectangle",
                "create_extrude", "create_fillet"):
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


def run_executor(plan: dict) -> tuple[dict, dict]:
    """
    Stage 2: Execute plan step by step.
    Returns (final_result, build_context).
    Raises RuntimeError with fallback message if >MAX_CONSECUTIVE_FAILURES.
    """
    ctx: dict        = {}
    consecutive_fails = 0
    last_result: dict = {}

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

        # Execute with wall-clock budget
        t0 = time.monotonic()
        try:
            result = dispatch_tool(tool, params, ctx)
            elapsed = time.monotonic() - t0
            if elapsed > TOOL_TIMEOUT_S:
                raise RuntimeError(
                    f"FALLBACK: tool {tool!r} exceeded {TOOL_TIMEOUT_S}s wall-clock budget "
                    f"({elapsed:.1f}s)"
                )
        except RuntimeError as e:
            if str(e).startswith("FALLBACK:"):
                raise
            log.warning("[Executor] Step %d failed: %s", i + 1, e)
            consecutive_fails += 1
            if consecutive_fails > MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"FALLBACK: {consecutive_fails} consecutive tool failures — "
                    f"last error on {tool!r}: {e}"
                )
            # Retry once with LLM-adjusted params
            log.info("[Executor] Retrying step %d with LLM-adjusted params...", i + 1)
            adjusted = _llm_adjust_params(step, str(e), ctx)
            for key in ("documentId", "workspaceId", "elementId"):
                ctx_key = {"documentId": "did", "workspaceId": "wid", "elementId": "eid"}.get(key)
                if ctx_key and ctx.get(ctx_key) and key not in adjusted:
                    adjusted[key] = ctx[ctx_key]
            try:
                result = dispatch_tool(tool, adjusted, ctx)
            except RuntimeError as e2:
                log.error("[Executor] Retry also failed: %s", e2)
                consecutive_fails += 1
                if consecutive_fails > MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"FALLBACK: {consecutive_fails} consecutive failures — "
                        f"retry of {tool!r} also failed: {e2}"
                    )
                result = {"error": str(e2), "skipped": True}
        else:
            consecutive_fails = 0  # reset on success

        last_result = result

        # Validate step output if flagged
        if step.get("validates_after") and not _validate_step(step, result):
            log.warning("[Executor] Validation failed for step %d (%s): %s",
                        i + 1, tool, result)

    return last_result, ctx


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
                _, new_ctx = run_executor(plan)
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
        "shape_type":  shape,
        "spec":        build_result.get("spec", ""),
        "plan_steps":  build_result.get("plan_steps", []),
        "rating":      rating,
        "comment":     comment,
        "mass_g":      build_result.get("mass_g"),
        "url":         build_result.get("url", ""),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })

    # Cap at MAX_FEEDBACK_ROWS, drop oldest
    if len(rows) > MAX_FEEDBACK_ROWS:
        rows = rows[-MAX_FEEDBACK_ROWS:]

    with open(FEEDBACK_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info("[Feedback] Stored %d★ example for shape_type=%s", rating, shape)


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
    Send build result + /cad rate prompt as plain text.
    Uses /cad rate N commands instead of inline keyboard callback_data,
    which conflicts with OpenClaw's own getUpdates polling.
    """
    url  = build_result.get("url", "—")
    mass = build_result.get("mass_g", "—")
    vol  = build_result.get("volume_cm3", "—")
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

        # ── Stage 1: Plan ───────────────────────────────────────────
        try:
            # Quick shape-type hint for fewshot matching
            hint = ""
            if any(w in spec.lower() for w in ("gear", "pinion", "sprocket")):
                hint = "gear"
            elif any(w in spec.lower() for w in ("beam", "channel", "plate", "bar")):
                hint = "structural"

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
        try:
            final, ctx = run_executor(plan)
        except RuntimeError as e:
            if str(e).startswith("FALLBACK:"):
                result = run_fallback(spec, str(e)[9:].strip())
            else:
                result = run_fallback(spec, str(e))
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

        result = {
            "spec":        spec,
            "path":        "v2",
            "url":         url,
            "did":         did,
            "wid":         wid,
            "eid":         eid,
            "mass_g":      repair.get("mass_g"),
            "volume_cm3":  repair.get("volume_cm3"),
            "has_bodies":  repair.get("has_bodies"),
            "repaired":    repair.get("repaired"),
            "deviation_pct": repair.get("deviation_pct"),
            "bounding_box":  repair.get("bounding_box"),
            "shape_type":  plan.get("shape_type", hint or "unknown"),
            "plan_steps":  [s["tool"] for s in plan.get("steps", [])],
            "built_at":    datetime.now(timezone.utc).isoformat(),
        }

        _write_session(result)
        log.info("[Build] Done: url=%s  mass=%s g  repaired=%s",
                 url, result["mass_g"], result["repaired"])

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

    store_feedback(last, rating, comment)
    stars = "⭐" * rating
    print(json.dumps({"ok": True, "rating": rating, "stars": stars,
                      "spec": last.get("spec", ""), "comment": comment}))


CMDS = {
    "build": _cmd_build,
    "plan":  _cmd_plan,
    "test":  _cmd_test,
    "rate":  _cmd_rate,
}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in CMDS:
        CMDS[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Usage: {sys.argv[0]} <{'|'.join(CMDS)}> [args...]")
        print("\nCommands:")
        print("  build <spec> [--chat-id <id>]  — full two-stage pipeline")
        print("  plan  <spec>                   — dry-run planner only")
        print("  test                           — spur gear 3:1 stress test")
        sys.exit(1)

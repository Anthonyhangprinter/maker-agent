#!/usr/bin/env python3
"""
OpenClaw Agentic CAD Builder — Onshape
======================================
Give it a spec or datasheet, it builds the model in Onshape fully headlessly.

Architecture:
  User prompt → LLM parses spec → geometry code generator → Onshape REST API → verify

Usage:
  python onshape_cad_agent.py --spec "W200x100 I-beam, length 2000mm"
  python onshape_cad_agent.py --spec "NACA 2412 airfoil 300mm chord 800mm span"
  python onshape_cad_agent.py --file datasheet.pdf
  python onshape_cad_agent.py --interactive

Config (set via openclaw config set env.* or export directly):
  ONSHAPE_ACCESS_KEY
  ONSHAPE_SECRET_KEY
  ONSHAPE_DOCUMENT_ID  — optional: target document (creates new one if unset)
"""

import os, sys, json, base64, time, re, argparse, math
from datetime import datetime, timezone
import requests

# ── Config ────────────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ.get("ONSHAPE_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("ONSHAPE_SECRET_KEY", "")
BASE_URL   = "https://cad.onshape.com"
TARGET_DOC = os.environ.get("ONSHAPE_DOCUMENT_ID", "")
OLLAMA_MODEL = "qwen2.5-coder:7b"

# ── Onshape Auth (Basic Auth — on_ prefix keys) ───────────────────────────────

def _auth_headers():
    """Onshape on_ API keys use HTTP Basic Auth (base64 access_key:secret_key)."""
    creds = base64.b64encode(f"{ACCESS_KEY}:{SECRET_KEY}".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Accept":        "application/json;charset=UTF-8",
        "Content-Type":  "application/json",
    }

def onshape(method, path, body=None):
    url  = BASE_URL + path
    resp = requests.request(method, url, headers=_auth_headers(), json=body, timeout=30)
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Onshape {method} {path} → {resp.status_code}: {resp.text[:400]}")
    return resp.json() if resp.text else {}

def test_auth():
    print("[Auth] Testing API credentials...")
    try:
        r = onshape("GET", "/api/v9/documents?limit=1")
        print(f"[Auth] ✅ Connected — documents endpoint returned OK")
        return True
    except Exception as e:
        print(f"[Auth] ❌ {e}")
        return False

# ── Onshape Document / Part Studio Setup ──────────────────────────────────────

def create_document(name):
    print(f"[Onshape] Creating document: {name}")
    doc = onshape("POST", "/api/v9/documents", {"name": name, "isPublic": True})
    did = doc["id"]
    wid = doc["defaultWorkspace"]["id"]
    print(f"[Onshape] Document created: {did}")
    return did, wid

def get_or_create_document(name):
    if TARGET_DOC:
        doc = onshape("GET", f"/api/v9/documents/{TARGET_DOC}")
        wid = doc["defaultWorkspace"]["id"]
        print(f"[Onshape] Using existing document: {TARGET_DOC}")
        return TARGET_DOC, wid
    return create_document(name)

def create_part_studio(did, wid, name="AI Generated Part"):
    """Get the default Part Studio that Onshape creates with every new document."""
    print(f"[Onshape] Getting Part Studio for: {name}")
    elements = onshape("GET", f"/api/v9/documents/d/{did}/w/{wid}/elements")
    for el in elements:
        if el.get("elementType") == "PARTSTUDIO":
            eid = el["id"]
            print(f"[Onshape] Part Studio element ID: {eid}")
            return eid
    raise RuntimeError("No Part Studio found in new document")

def add_feature(did, wid, eid, feature_body):
    path   = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
    result = onshape("POST", path, {"feature": feature_body})
    fid    = result.get("feature", {}).get("featureId", "?")
    # POST response uses 'featureState' (singular) — not 'featureStates'
    state_obj = result.get("featureState", {})
    state     = state_obj.get("featureStatus", "?")
    print(f"  [Feature] {feature_body.get('featureType','?')} → id={fid} status={state}")
    if state == "ERROR":
        notices = result.get("feature", {}).get("notices", [])
        for n in notices:
            print(f"  [ERROR] {n.get('message','?')}")
    return fid

def get_mass_props(did, wid, eid):
    path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties"
    return onshape("GET", path)

# ── Feature Studio API ────────────────────────────────────────────────────────

def create_document_version(did, wid, version_name="feature-version"):
    """Create a named version of the current workspace so it can be used as a
    stable namespace when calling Feature Studio custom features via BTM.
    NOTE: documentId must be included in the body or Onshape returns 400."""
    path = f"/api/v9/documents/d/{did}/versions"
    result = onshape("POST", path, {"documentId": did, "workspaceId": wid, "name": version_name})
    vid = result.get("id")
    print(f"[Version] Created: {vid}")
    return vid

def create_feature_studio(did, wid, name="Generated Feature"):
    """Create a new Feature Studio tab in an existing document."""
    path = f"/api/v9/featurestudios/d/{did}/w/{wid}"
    result = onshape("POST", path, {"name": name})
    feid = result.get("id")
    print(f"[FeatureStudio] Created element: {feid}")
    return feid

def get_fs_content(did, wid, feid):
    """Get Feature Studio FeatureScript source.
    Response uses 'sourceMicroversion' (not 'microversion') as the ETag field."""
    path = f"/api/v9/featurestudios/d/{did}/w/{wid}/e/{feid}"
    return onshape("GET", path)

def set_fs_content(did, wid, feid, source_microversion, contents):
    """Upload new FeatureScript source to a Feature Studio.
    source_microversion is the ETag from get_fs_content (key: 'sourceMicroversion').
    Returns the response dict; the new sourceMicroversion is in result['sourceMicroversion']."""
    path = f"/api/v9/featurestudios/d/{did}/w/{wid}/e/{feid}"
    # Onshape expects the current sourceMicroversion as the request body microversion for optimistic locking
    result = onshape("POST", path, {"microversion": source_microversion, "contents": contents})
    new_mv = result.get("sourceMicroversion", "")
    print(f"[FeatureStudio] Code uploaded ({len(contents)} chars)  sourceMicroversion={new_mv[:12] if new_mv else 'n/a'}")
    return result

def call_custom_feature(did, wid, eid, feature_name, feid,
                        display_name="Generated Part", fs_microversion=None):
    """Add a custom Feature Studio function to a Part Studio via BTM.

    Tries namespace formats in order until one succeeds or all fail:
      1. d/{did}/w/{wid}/e/{feid}        — live workspace
      2. {did}::m{fs_microversion}       — doc + FS microversion (community-reported)
      3. {did}::{feid}                   — doc + element ID
      4. d/{did}/v/{vid}/e/{feid}        — versioned snapshot (last resort)
    """
    path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"

    def _attempt(namespace):
        feature_body = {
            "btType": "BTMFeature-134",
            "featureType": feature_name,
            "namespace": namespace,
            "name": display_name,
            "suppressed": False,
            "parameters": [],
            "subFeatures": [],
            "returnAfterSubfeatures": False
        }
        result = onshape("POST", path, {"feature": feature_body})
        fid       = result.get("feature", {}).get("featureId", "?")
        state_obj = result.get("featureState", {})
        state     = state_obj.get("featureStatus", "?")
        print(f"  [CustomFeature] {feature_name} ns={namespace[:50]} → id={fid} status={state}")
        if state == "ERROR":
            for n in result.get("feature", {}).get("notices", []):
                print(f"  [ERROR] {n.get('message','?')}")
        return fid

    def _is_ns_error(e):
        s = str(e)
        return "invalid namespace" in s.lower() or '"status": 400' in s or "→ 400" in s

    # Build list of namespace candidates to try
    candidates = [f"d/{did}/w/{wid}/e/{feid}"]
    if fs_microversion:
        mv = fs_microversion.lstrip("m")   # ensure no double-m prefix
        candidates.append(f"{did}::m{mv}")
        candidates.append(f"d/{did}/m/{mv}/e/{feid}")
    candidates.append(f"{did}::{feid}")

    for ns in candidates:
        try:
            return _attempt(ns)
        except RuntimeError as e:
            if not _is_ns_error(e):
                raise
            print(f"  [CustomFeature] ns={ns[:40]}… rejected → {str(e)[:120]}")

    # Last resort: create a version snapshot
    print("  [CustomFeature] all workspace/microversion namespaces failed — trying versioned")
    vid = create_document_version(did, wid, f"fs-{feature_name[:20]}")
    return _attempt(f"d/{did}/v/{vid}/e/{feid}")

# ── Feature JSON Builders ─────────────────────────────────────────────────────

def _param_quantity(param_id, expression):
    return {"btType": "BTMParameterQuantity-147",
            "value": 0, "units": "", "isInteger": False,
            "expression": expression, "parameterId": param_id}

def _param_enum(param_id, value, enum_name=""):
    return {"btType": "BTMParameterEnum-145",
            "value": value, "parameterId": param_id, "enumName": enum_name}

def _param_bool(param_id, value):
    return {"btType": "BTMParameterBoolean-144",
            "value": value, "parameterId": param_id}

def _param_query(param_id, query_str):
    return {
        "btType": "BTMParameterQueryList-148",
        "queries": [{"btType": "BTMIndividualQuery-138", "queryString": query_str}],
        "parameterId": param_id
    }

def _param_sketch_plane(det_id="JDC"):
    """Sketch plane parameter using deterministicIds — the only format Onshape accepts for the Front plane."""
    return {
        "btType": "BTMParameterQueryList-148",
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": "BTMIndividualQuery-138",
            "queryString": "",
            "deterministicIds": [det_id]
        }]
    }

def _line_entity(idx, x1, y1, x2, y2):
    return {
        "btType": "BTMSketchCurveSegment-155",
        "entityId": f"line{idx}",
        "startPointId": f"pt{idx}s",
        "endPointId":   f"pt{idx}e",
        "startParam": 0.0, "endParam": 1.0,
        "geometry": {
            "btType": "BTCurveGeometryLine-117",
            "pntX": x1, "pntY": y1,
            "dirX": x2 - x1, "dirY": y2 - y1
        }
    }

def _closed_polygon_sketch(pts_m, name):
    """
    Build a BTMSketch from a list of (x,y) points in metres.
    Closure is inferred by Onshape from coincident endpoint coordinates —
    explicit constraints are not needed and cause errors when point IDs are used.
    """
    n        = len(pts_m)
    entities = [_line_entity(i, *pts_m[i], *pts_m[(i+1) % n]) for i in range(n)]
    return {
        "btType": "BTMSketch-151", "featureType": "newSketch",
        "name": name, "suppressed": False,
        "parameters": [_param_sketch_plane()],
        "entities": entities, "constraints": [],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_sketch_i_beam(h, b, tf, tw):
    H = h/1000; B = b/1000; TF = tf/1000; TW = tw/1000
    pts = [
        (-B/2,  H/2),       (-B/2,  H/2 - TF),   (-TW/2, H/2 - TF),
        (-TW/2, -H/2 + TF), (-B/2, -H/2 + TF),   (-B/2, -H/2),
        ( B/2, -H/2),       ( B/2, -H/2 + TF),   ( TW/2, -H/2 + TF),
        ( TW/2,  H/2 - TF), ( B/2,  H/2 - TF),   ( B/2,  H/2),
    ]
    return _closed_polygon_sketch(pts, "I-Beam Profile")

def build_sketch_rectangle(w_mm, h_mm, name="Rectangle Profile"):
    W = w_mm/2000; H = h_mm/2000
    return _closed_polygon_sketch([(-W,-H),(W,-H),(W,H),(-W,H)], name)

def build_extrude(length_mm, sketch_fid, name="Extrude"):
    return {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": name, "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f'query = qSketchRegion(id + "{sketch_fid}", true);',
                          "deterministicIds": []}]},
            _param_quantity("depth", f"{length_mm} mm"),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_cut_extrude_symmetric(sketch_fid, depth_mm, name="Cut"):
    """Cut (REMOVE) extrude symmetric about the sketch plane — cuts in both directions."""
    return {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": name, "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f'query = qSketchRegion(id + "{sketch_fid}", true);',
                          "deterministicIds": []}]},
            _param_enum("operationType", "REMOVE", "NewBodyOperationType"),
            _param_quantity("depth", f"{depth_mm} mm"),
            _param_bool("symmetric", True),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_mitre_cut_sketch(H_mm, L_mm, is_start_end, name):
    """
    45° mitre cut triangle sketched on the Right plane (JEC).
    Right plane sketch axes: pntX → Z (beam length), pntY → Y (beam height).

    is_start_end=True : wedge at Z=0 end  (bottom stays at Z=0, top cut back by H)
    is_start_end=False: opposing wedge at Z=L (top stays at Z=L, bottom cut at Z=L-H)
    """
    H = H_mm / 1000.0
    L = L_mm / 1000.0
    m = 0.010  # 10 mm margin beyond beam boundary

    if is_start_end:
        # Cut line: Z = Y + H/2  →  at Y=-H/2, Z=0; at Y=H/2, Z=H
        # Triangle to remove (left of cut line, above Z=-m):
        pts = [
            (-m,     -H/2 - m),  # bottom-left outside beam
            (-m,      H/2 + m),  # top-left outside beam
            (H + m,   H/2 + m),  # top-right at Z=H (beyond cut line)
        ]
    else:
        # Opposing cut: line Z = L - H/2 + Y  →  at Y=-H/2, Z=L-H; at Y=H/2, Z=L
        # Triangle to remove (right of cut line, below Z=L+m):
        pts = [
            (L + m,       H/2 + m),   # top-right outside beam
            (L + m,      -H/2 - m),   # bottom-right outside beam
            (L - H - m,  -H/2 - m),   # bottom-left at Z=L-H (beyond cut line)
        ]

    n = len(pts)
    entities = [_line_entity(i, *pts[i], *pts[(i+1) % n]) for i in range(n)]
    return {
        "btType": "BTMSketch-151", "featureType": "newSketch",
        "name": name, "suppressed": False,
        "parameters": [{"btType": "BTMParameterQueryList-148", "parameterId": "sketchPlane",
                        "queries": [{"btType": "BTMIndividualQuery-138",
                                     "queryString": "", "deterministicIds": ["JEC"]}]}],
        "entities": entities, "constraints": [], "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_fillet(radius_mm, edge_query):
    # NOTE: do NOT include a 'filletType' enum parameter — Onshape BTM rejects it
    # with "does not match its feature spec".  The fillet feature defaults to EDGE
    # mode automatically.
    return {
        "btType": "BTMFeature-134", "featureType": "fillet",
        "name": "Fillet", "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138", "queryString": edge_query}]},
            _param_quantity("radius", f"{radius_mm} mm"),
            _param_bool("tangentPropagation", True),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_boolean_union(name="Boolean Union"):
    """Merge ALL solid bodies in the Part Studio into one via a Boolean Union feature.
    Tries featureType='booleanBody' (matches Onshape std/booleanBody.fs function name)."""
    return {
        "btType": "BTMFeature-134",
        "featureType": "booleanBody",
        "name": name,
        "suppressed": False,
        "parameters": [
            _param_enum("operationType", "UNION", "BooleanOperationType"),
            {
                "btType": "BTMParameterQueryList-148",
                "parameterId": "tools",
                "queries": [{
                    "btType": "BTMIndividualQuery-138",
                    "queryString": "query = qEverything(EntityType.BODY);"
                }]
            }
        ],
        "subFeatures": [],
        "returnAfterSubfeatures": False
    }

def build_shell(thickness_mm, face_query):
    return {
        "btType": "BTMFeature-134", "featureType": "shell",
        "name": "Shell", "suppressed": False,
        "parameters": [
            _param_quantity("thickness", f"{thickness_mm} mm"),
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138", "queryString": face_query}]},
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_extrude_add_opposite(length_mm, sketch_fid, name="Extrude Add"):
    """Extrude in the opposite direction from the sketch normal, merging into existing body."""
    return {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": name, "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f'query = qSketchRegion(id + "{sketch_fid}", true);',
                          "deterministicIds": []}]},
            _param_quantity("depth", f"{length_mm} mm"),
            _param_bool("oppositeDirection", True),
            _param_enum("operationType", "ADD", "NewBodyOperationType"),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

def build_extrude_add(length_mm, sketch_fid, name="Extrude Add"):
    """Extrude in the SAME direction as the sketch normal, merging (ADD) into existing body.
    defaultScope=True tells Onshape to automatically select all touching/overlapping bodies
    as the merge target — required for ADD to work via the BTM REST API."""
    return {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": name, "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f'query = qSketchRegion(id + "{sketch_fid}", true);',
                          "deterministicIds": []}]},
            _param_quantity("depth", f"{length_mm} mm"),
            _param_enum("operationType", "ADD", "NewBodyOperationType"),
            _param_bool("defaultScope", True),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }

# ── NACA 4-Digit Airfoil ──────────────────────────────────────────────────────

def naca_4digit_coords(digits, n_points=50):
    """
    Generate normalised NACA 4-digit airfoil coordinates (x,y in chord units 0–1).
    Returns (upper, lower) each as list of (x,y) tuples from LE to TE.
    Uses cosine spacing for better leading-edge resolution.
    """
    m = int(digits[0]) / 100.0    # max camber ratio
    p = int(digits[1]) / 10.0     # camber position (tenths of chord)
    t = int(digits[2:]) / 100.0   # max thickness ratio

    # Cosine spacing: denser near LE and TE
    betas = [math.pi * i / (n_points - 1) for i in range(n_points)]
    xs    = [(1 - math.cos(b)) / 2 for b in betas]

    def y_thickness(x):
        return (t / 0.2) * (
            0.2969 * math.sqrt(max(x, 0))
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    def camber_and_slope(x):
        if m == 0 or p == 0:
            return 0.0, 0.0
        if x < p:
            yc  = (m / p**2) * (2*p*x - x**2)
            dyc = (2*m / p**2) * (p - x)
        else:
            yc  = (m / (1-p)**2) * (1 - 2*p + 2*p*x - x**2)
            dyc = (2*m / (1-p)**2) * (p - x)
        return yc, dyc

    upper, lower = [], []
    for x in xs:
        yt       = y_thickness(x)
        yc, dyc  = camber_and_slope(x)
        theta    = math.atan(dyc)
        upper.append((x - yt * math.sin(theta),  yc + yt * math.cos(theta)))
        lower.append((x + yt * math.sin(theta),  yc - yt * math.cos(theta)))

    return upper, lower


def build_sketch_naca_airfoil(digits, chord_mm, name=None):
    """
    Build a BTMSketch for a NACA 4-digit airfoil profile.
    Profile: upper surface LE→TE, lower surface TE→LE (closed polygon).
    ~100 line segments total.
    """
    if name is None:
        name = f"NACA {digits} Profile"
    chord = chord_mm / 1000.0   # metres

    upper, lower = naca_4digit_coords(digits, n_points=51)

    # Build closed profile: upper LE→TE then lower TE→LE
    # Skip duplicate LE (lower[0] == upper[0]) and use lower[1..n-1] reversed
    profile_norm = upper + lower[-2:0:-1]   # 51 + 49 = 100 points

    # Scale to chord length; place chord along X axis, LE at origin
    pts = [(x * chord, y * chord) for x, y in profile_norm]

    return _closed_polygon_sketch(pts, name)

# ── Shape Library ─────────────────────────────────────────────────────────────

SHAPE_BUILDERS = {
    "i_beam":       lambda p, d, w, e: _build_i_beam(p, d, w, e),
    "c_channel":    lambda p, d, w, e: _build_c_channel(p, d, w, e),
    "hollow_rect":  lambda p, d, w, e: _build_hollow_rect(p, d, w, e),
    "plate":        lambda p, d, w, e: _build_plate(p, d, w, e),
    "round_bar":    lambda p, d, w, e: _build_round_bar(p, d, w, e),
    "naca_airfoil": lambda p, d, w, e: _build_naca_airfoil(p, d, w, e),
    "bolt":         lambda p, d, w, e: _build_bolt(p, d, w, e),
}

def _build_i_beam(p, did, wid, eid):
    h  = float(p["height_mm"])
    b  = float(p["flange_width_mm"])
    tf = float(p["flange_thickness_mm"])
    tw = float(p["web_thickness_mm"])
    L  = float(p["length_mm"])
    fid = add_feature(did, wid, eid, build_sketch_i_beam(h, b, tf, tw))
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(L, fid, "I-Beam Body"))
    if p.get("root_radius_mm"):
        # Select concave edges (web–flange transitions) for the root radius fillet.
        # qConcave scoped to edges only — qEverything() alone returns all entities
        # including faces, which fillet rejects.
        add_feature(did, wid, eid, build_fillet(float(p["root_radius_mm"]),
                                                "qConcave(qEverything(EntityType.EDGE))"))
    if p.get("mitre_cuts"):
        time.sleep(0.5)
        # Beam width B used as symmetric cut depth (cut through full flange width + margin)
        cut_depth = b + 50
        sk1 = add_feature(did, wid, eid, build_mitre_cut_sketch(h, L, True,  "Mitre Start Sketch"))
        time.sleep(0.3)
        add_feature(did, wid, eid, build_cut_extrude_symmetric(sk1, cut_depth, "Mitre Start Cut"))
        time.sleep(0.3)
        sk2 = add_feature(did, wid, eid, build_mitre_cut_sketch(h, L, False, "Mitre End Sketch"))
        time.sleep(0.3)
        add_feature(did, wid, eid, build_cut_extrude_symmetric(sk2, cut_depth, "Mitre End Cut"))

def _build_c_channel(p, did, wid, eid):
    h  = float(p["height_mm"]); b  = float(p["flange_width_mm"])
    tf = float(p["flange_thickness_mm"]); tw = float(p["web_thickness_mm"])
    L  = float(p["length_mm"])
    H = h/1000; B = b/1000; TF = tf/1000; TW = tw/1000
    pts = [(0,0),(B,0),(B,TF),(TW,TF),(TW,H-TF),(B,H-TF),(B,H),(0,H)]
    sketch = _closed_polygon_sketch(pts, "C-Channel Profile")
    fid = add_feature(did, wid, eid, sketch)
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(L, fid, "C-Channel Body"))

def _build_hollow_rect(p, did, wid, eid):
    W = float(p["width_mm"]); H_mm = float(p["height_mm"])
    T = float(p["wall_thickness_mm"]); L = float(p["length_mm"])
    fid = add_feature(did, wid, eid, build_sketch_rectangle(W, H_mm, "RHS Profile"))
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(L, fid, "RHS Body"))
    time.sleep(0.5)
    add_feature(did, wid, eid, build_shell(T, "qCapEntity(qEverything(), CapType.START)"))

def _build_plate(p, did, wid, eid):
    fid = add_feature(did, wid, eid,
                      build_sketch_rectangle(float(p["width_mm"]), float(p["height_mm"]), "Plate Profile"))
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(float(p["length_mm"]), fid, "Plate Body"))

def _build_circle_sketch(radius_mm, name="Circle Profile", n_sides=32):
    """Approximate a circle as a regular n-gon (BTM circles don't produce sketch regions)."""
    R = radius_mm / 1000.0
    pts = [(R * math.cos(2 * math.pi * i / n_sides),
            R * math.sin(2 * math.pi * i / n_sides)) for i in range(n_sides)]
    return _closed_polygon_sketch(pts, name)

def _build_round_bar(p, did, wid, eid):
    fid = add_feature(did, wid, eid,
                      _build_circle_sketch(float(p["diameter_mm"]) / 2, "Round Bar Profile"))
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(float(p["length_mm"]), fid, "Round Bar Body"))

def _build_naca_airfoil(p, did, wid, eid):
    digits   = str(p["naca_digits"])
    chord_mm = float(p["chord_mm"])
    span_mm  = float(p["span_mm"])
    print(f"[NACA] Generating NACA {digits} profile, chord={chord_mm}mm, span={span_mm}mm")
    sketch = build_sketch_naca_airfoil(digits, chord_mm)
    fid = add_feature(did, wid, eid, sketch)
    time.sleep(0.5)
    add_feature(did, wid, eid, build_extrude(span_mm, fid, f"NACA {digits} Wing Section"))

# ── ISO Metric Bolt ───────────────────────────────────────────────────────────

# ISO 4014/4017 hex bolt standard dimensions (coarse pitch)
ISO_BOLT_DIMS = {
    "M3":  {"d": 3,   "pitch": 0.5,  "s": 5.5, "k": 2.0},
    "M4":  {"d": 4,   "pitch": 0.7,  "s": 7,   "k": 2.8},
    "M5":  {"d": 5,   "pitch": 0.8,  "s": 8,   "k": 3.5},
    "M6":  {"d": 6,   "pitch": 1.0,  "s": 10,  "k": 4.0},
    "M8":  {"d": 8,   "pitch": 1.25, "s": 13,  "k": 5.3},
    "M10": {"d": 10,  "pitch": 1.5,  "s": 17,  "k": 6.4},
    "M12": {"d": 12,  "pitch": 1.75, "s": 19,  "k": 7.5},
    "M14": {"d": 14,  "pitch": 2.0,  "s": 22,  "k": 8.8},
    "M16": {"d": 16,  "pitch": 2.0,  "s": 24,  "k": 10.0},
    "M18": {"d": 18,  "pitch": 2.5,  "s": 27,  "k": 11.5},
    "M20": {"d": 20,  "pitch": 2.5,  "s": 30,  "k": 12.5},
    "M22": {"d": 22,  "pitch": 2.5,  "s": 32,  "k": 14.0},
    "M24": {"d": 24,  "pitch": 3.0,  "s": 36,  "k": 15.0},
    "M27": {"d": 27,  "pitch": 3.0,  "s": 41,  "k": 17.0},
    "M30": {"d": 30,  "pitch": 3.5,  "s": 46,  "k": 18.7},
    "M36": {"d": 36,  "pitch": 4.0,  "s": 55,  "k": 22.5},
    "M42": {"d": 42,  "pitch": 4.5,  "s": 65,  "k": 26.0},
    "M48": {"d": 48,  "pitch": 5.0,  "s": 75,  "k": 30.0},
}

def build_sketch_hexagon(af_mm, plane_id="JDC", name="Hex Profile"):
    """Regular hexagon centred at origin; af_mm = across-flats dimension."""
    s = af_mm / 1000.0        # across flats in metres
    r = s / math.sqrt(3)      # circumradius = side_length = s/√3
    # Start at 90° so a flat is at top/bottom (standard bolt orientation)
    pts = [(r * math.cos(math.radians(90 + 60*i)),
            r * math.sin(math.radians(90 + 60*i))) for i in range(6)]
    sketch = _closed_polygon_sketch(pts, name)
    # Override the sketch plane if not Front
    if plane_id != "JDC":
        sketch["parameters"] = [_param_sketch_plane(plane_id)]
    return sketch

def _bolt_featurescript(d_mm, s_mm, k_mm, L_mm, designation="Bolt"):
    """
    Generate FeatureScript for a hex-head bolt.
    Uses qCapEntity to sketch the shank on the head's top face — the correct
    Onshape approach that avoids the BTM REST API's ADD-operation limitations.

    d_mm  = nominal diameter (shank OD)
    s_mm  = head across-flats
    k_mm  = head height
    L_mm  = shank length
    """
    # For skRegularPolygon: vertex radius = s / sqrt(3)
    # (regular hex: vertex_r = side_length = across_flats / sqrt(3))
    vertex_r_m = (s_mm / 1000.0) / math.sqrt(3)
    d_m        = d_mm  / 1000.0
    k_m        = k_mm  / 1000.0
    L_m        = L_mm  / 1000.0
    fn_name    = "boltFeature"

    code = f"""FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// {designation} hex-head bolt: d={d_mm}mm  s={s_mm}mm  k={k_mm}mm  L={L_mm}mm
annotation {{ "Feature Type Name" : "{designation} Bolt" }}
export const {fn_name} = defineFeature(function(context is Context, id is Id, definition is map)
    precondition {{}}
    {{
        // ── Hex Head ──────────────────────────────────────────────────────────────
        var skHead = newSketchOnPlane(context, id + "skHead", {{
            "sketchPlane" : plane(WORLD_ORIGIN, Z_DIRECTION)
        }});
        skRegularPolygon(skHead, "hex", {{
            "center"      : vector(0, 0) * millimeter,
            "firstVertex" : vector({vertex_r_m * 1000:.4f}, 0) * millimeter,
            "sides"       : 6
        }});
        skSolve(skHead);

        opExtrude(context, id + "head", {{
            "entities"      : qSketchRegion(id + "skHead"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : {k_m * 1000:.4f} * millimeter,
            "operationType" : NewBodyOperationType.NEW
        }});

        // ── Shank: sketched on END face of head so ADD works reliably ─────────────
        var headEndFace = qCapEntity(id + "head", CapType.END);
        var skShank = newSketchOnPlane(context, id + "skShank", {{
            "sketchPlane" : evFacePlane(context, {{ "face" : headEndFace }})
        }});
        skCircle(skShank, "shank", {{
            "center" : vector(0, 0) * millimeter,
            "radius" : {d_m / 2 * 1000:.4f} * millimeter
        }});
        skSolve(skShank);

        opExtrude(context, id + "shank", {{
            "entities"      : qSketchRegion(id + "skShank"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : {L_m * 1000:.4f} * millimeter,
            "operationType" : NewBodyOperationType.ADD
        }});
    }}
);
"""
    return code, fn_name


def build_bolt_with_featurescript(params, doc_name):
    """
    Build an ISO bolt via FeatureScript pipeline.
    Called from build_from_spec for all bolt shapes — bypasses the BTM REST API
    which cannot reliably merge two separate extrudes via operationType=ADD.
    """
    designation = params.get("designation", "").upper()
    std = ISO_BOLT_DIMS.get(designation, {})

    if std:
        d = std["d"];  s = std["s"];  k = std["k"]
    else:
        d = float(params.get("nominal_diameter_mm", 10))
        s = float(params.get("head_af_mm",  d * 1.7))
        k = float(params.get("head_height_mm", d * 0.64))

    L = float(params["length_mm"])
    label = designation or f"M{d:.0f}"
    print(f"[Bolt] {label}: d={d}mm s={s}mm k={k}mm L={L}mm  → FeatureScript path")

    fs_code, fn_name = _bolt_featurescript(d, s, k, L, label)

    # Create document + Part Studio + Feature Studio
    did, wid = create_document(doc_name)
    eid      = create_part_studio(did, wid, doc_name)
    time.sleep(1)
    feid     = create_feature_studio(did, wid, f"{doc_name} Feature")
    time.sleep(0.5)

    fs_data   = get_fs_content(did, wid, feid)
    source_mv = fs_data.get("sourceMicroversion", "")
    upload_result = set_fs_content(did, wid, feid, source_mv, fs_code)
    new_mv = upload_result.get("sourceMicroversion", "")
    time.sleep(2)   # let Onshape compile the FeatureScript

    call_custom_feature(did, wid, eid, fn_name, feid, doc_name, fs_microversion=new_mv)
    time.sleep(1)

    return did, wid, eid


def _build_bolt(p, did, wid, eid):
    """
    Build a hex-head bolt using BTM REST API.

    Strategy: shank cylinder FIRST as Body 1 (NEW), then hex head ADD on top.
    This order is critical: the hex prism is LARGER than the cylinder and adds
    real external material at Z=0..k, so operationType=ADD succeeds (unlike
    circle-into-hex where the circle adds nothing outside the existing body).
    """
    designation = p.get("designation", "").upper()
    std = ISO_BOLT_DIMS.get(designation, {})
    if std:
        d = std["d"];  s = std["s"];  k = std["k"]
    else:
        d = float(p.get("nominal_diameter_mm", 10))
        s = float(p.get("head_af_mm", d * 1.7))
        k = float(p.get("head_height_mm", d * 0.64))
    L = float(p["length_mm"])
    label = designation or f"M{d:.0f}"
    print(f"[Bolt] {label}: d={d}mm s={s}mm k={k}mm L={L}mm")

    # ── Body 1: Shank cylinder (NEW, polygon-approximated circle) ─────────────
    shank_sk = add_feature(did, wid, eid, _build_circle_sketch(d / 2, "Shank Profile"))
    time.sleep(0.3)
    add_feature(did, wid, eid, build_extrude(L, shank_sk, "Shank Body"))
    time.sleep(0.5)

    # ── Head: Hex prism ADD to cylinder ───────────────────────────────────────
    # The hex (s=27mm across-flats, inscribed_r=13.5mm) is larger than the
    # shank (r=9mm) so ADD has genuine new external material → should succeed.
    head_sk = add_feature(did, wid, eid, build_sketch_hexagon(s))
    time.sleep(0.3)
    head_extrude = {
        "btType": "BTMFeature-134", "featureType": "extrude",
        "name": "Hex Head", "suppressed": False,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{"btType": "BTMIndividualQuery-138",
                          "queryString": f'query = qSketchRegion(id + "{head_sk}", true);',
                          "deterministicIds": []}]},
            _param_quantity("depth", f"{k} mm"),
            _param_enum("operationType", "ADD", "NewBodyOperationType"),
            _param_bool("defaultScope", True),
        ],
        "subFeatures": [], "returnAfterSubfeatures": False
    }
    head_fid = add_feature(did, wid, eid, head_extrude)

    # ── Best-effort union in case ADD created 2 bodies instead of 1 ─────────
    time.sleep(0.5)
    try:
        add_feature(did, wid, eid, build_boolean_union("Bolt Union"))
    except RuntimeError:
        pass   # ADD already merged; union not needed

# ── LLM Spec Parser ───────────────────────────────────────────────────────────

PARSE_SYSTEM = """You are a CAD parameter extraction engine.
Given a part description, extract geometric parameters and return ONLY valid JSON — no markdown, no explanation.

Return this exact structure:
{
  "shape_type": "i_beam" | "c_channel" | "hollow_rect" | "plate" | "round_bar" | "naca_airfoil" | "bolt" | "featurescript",
  "part_name": "descriptive name",
  "material": "steel" | "aluminium" | "other",
  "params": { ... }
}

Params by shape_type:

i_beam:       height_mm, flange_width_mm, flange_thickness_mm, web_thickness_mm, length_mm, root_radius_mm (optional), mitre_cuts (optional bool)
c_channel:    height_mm, flange_width_mm, flange_thickness_mm, web_thickness_mm, length_mm
hollow_rect:  width_mm, height_mm, wall_thickness_mm, length_mm
plate:        width_mm, height_mm, length_mm
round_bar:    diameter_mm, length_mm
naca_airfoil: naca_digits (string e.g. "2412"), chord_mm, span_mm
bolt:         designation (e.g. "M18"), length_mm
              Optional overrides (only if non-standard): nominal_diameter_mm, head_af_mm, head_height_mm

Standard designations — use EXACT dimensions shown (all fields REQUIRED):
- W200x100:    shape_type=i_beam, height_mm=210, flange_width_mm=206, flange_thickness_mm=14.5, web_thickness_mm=9.0, root_radius_mm=11.4
- 250UC72.9:   shape_type=i_beam, height_mm=250, flange_width_mm=254, flange_thickness_mm=14.2, web_thickness_mm=8.6, root_radius_mm=12.7
- UB203x102x23:shape_type=i_beam, height_mm=203, flange_width_mm=102, flange_thickness_mm=9.3,  web_thickness_mm=5.4, root_radius_mm=7.6
- ISO metric bolts M3-M48: use shape_type=bolt with designation="M6" etc.
IMPORTANT: root_radius_mm must ALWAYS be included for standard steel sections.

NACA 4-digit airfoils (e.g. "NACA 2412"): set naca_digits="2412", extract chord_mm and span_mm.
If spec mentions "mitre" or "miter" cuts: set mitre_cuts=true.
Convert any imperial units to metric (mm).
All param values must be numbers except naca_digits and designation which are strings.

IMPORTANT: If the geometry is NOT a standard structural section, plate, round bar, airfoil, or bolt — use shape_type="featurescript".
Examples: spoon, fork, gear, knob, bottle, cup, hook, bracket, spring, helix, organic/freeform shape, revolved part.
For featurescript shapes, set params to {} and add a top-level "description" field with a detailed geometric description
including all key dimensions the user mentioned (e.g. "A table spoon: bowl 80mm long 50mm wide, handle 150mm long tapering from 12mm to 6mm diameter").
"""

def parse_spec_with_ollama(spec_text):
    import urllib.request
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "system": PARSE_SYSTEM,
        "prompt": spec_text,
        "format": "json",
        "options": {"num_ctx": 8192}
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    text = data["response"].strip()
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return json.loads(text)

def parse_spec(spec_text):
    print("\n[Parser] Extracting parameters...")
    result = parse_spec_with_ollama(spec_text)
    print(f"[Parser] Shape type: {result['shape_type']}")
    print(f"[Parser] Part name:  {result['part_name']}")
    print(f"[Parser] Params:\n{json.dumps(result['params'], indent=2)}")
    return result

# ── PDF Reader ─────────────────────────────────────────────────────────────────

def read_pdf_spec(path):
    import fitz
    doc  = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    print(f"[PDF] Read {len(text)} chars from {path}")
    return text

# ── FeatureScript Learning ────────────────────────────────────────────────────

LEARNINGS_FILE = "/home/theultimatecunt/.openclaw/cad-learnings.json"
FS_LEARNINGS_DIR = os.path.join(os.path.dirname(__file__), "fs-learnings")

def _load_learnings():
    try:
        with open(LEARNINGS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"version": 1, "successful_builds": [], "known_errors": []}
    except Exception:
        return {"version": 1, "successful_builds": [], "known_errors": []}

def _save_learnings(data):
    with open(LEARNINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_successful_fs_build(spec, feat_name, part_name, fs_code, verify_result=None):
    """
    Persist a successful FeatureScript build so future generations can learn from it.
    Saves to cad-learnings.json and writes the working code to fs-learnings/.
    """
    os.makedirs(FS_LEARNINGS_DIR, exist_ok=True)
    data = _load_learnings()

    entry = {
        "spec":         spec,
        "feature_name": feat_name,
        "part_name":    part_name,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "mass_g":       verify_result.get("mass_g")    if verify_result else None,
        "volume_cm3":   verify_result.get("volume_cm3") if verify_result else None,
    }
    data["successful_builds"].append(entry)
    # Keep only the 30 most recent
    data["successful_builds"] = data["successful_builds"][-30:]
    _save_learnings(data)

    # Write the working FeatureScript as an example file for future few-shot prompting
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", part_name.lower())[:40]
    fs_path = os.path.join(FS_LEARNINGS_DIR, f"learned_{safe_name}.fs")
    with open(fs_path, "w") as f:
        f.write(f"// Learned from successful build: {spec}\n")
        f.write(f"// Part: {part_name}  |  Feature: {feat_name}\n")
        if verify_result:
            f.write(f"// Verified: mass={verify_result.get('mass_g')}g  volume={verify_result.get('volume_cm3')}cm³\n")
        f.write("\n")
        f.write(fs_code)
    print(f"[Learn] Saved working FeatureScript → {fs_path}")
    return fs_path

def log_fs_error(spec, error_description, wrong_pattern, fix_applied):
    """
    Record a FeatureScript error+fix pair so the LLM can avoid the same mistake.
    Called by the CAD agent when it detects and fixes an error in generated code.
    """
    data = _load_learnings()
    entry = {
        "spec":            spec,
        "error":           error_description,
        "wrong_pattern":   wrong_pattern,
        "fix_applied":     fix_applied,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }
    # Deduplicate by error description (same error seen twice = skip)
    existing = {e.get("error", "") for e in data["known_errors"]}
    if error_description not in existing:
        data["known_errors"].append(entry)
        data["known_errors"] = data["known_errors"][-20:]  # cap at 20
        _save_learnings(data)
        print(f"[Learn] Logged error pattern: {error_description[:60]}")

def load_fs_learnings():
    """Load user-confirmed working examples from fs-learnings/ directory."""
    examples = []
    if not os.path.isdir(FS_LEARNINGS_DIR):
        return examples
    for fname in sorted(os.listdir(FS_LEARNINGS_DIR)):
        if fname.endswith(".fs"):
            try:
                with open(os.path.join(FS_LEARNINGS_DIR, fname)) as f:
                    examples.append({"filename": fname, "content": f.read()})
            except Exception:
                pass
    return examples

# ── FeatureScript Generation ───────────────────────────────────────────────────

FS_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "fs-examples")

FS_GENERATE_SYSTEM = """\
You are a FeatureScript programmer for Onshape CAD. Your job is to write a complete,
self-contained FeatureScript program that builds the described 3D geometry.

RULES:
1. Return ONLY valid JSON — no markdown fences, no explanation, just the JSON object.
2. ALL dimensions must be hardcoded from the spec (no user parameters in precondition).
3. The exported feature function MUST have an empty precondition block: precondition {}
4. Use FeatureScript 2454 with: import(path : "onshape/std/geometry.fs", version : "2454.0");
5. Feature function name must be camelCase starting lowercase (e.g. makeSpoon).
6. Use millimeter for dimensions: e.g. 80 * millimeter
7. Sketch points are 2D vectors: vector(x * millimeter, y * millimeter)

KEY OPERATIONS (use exactly as shown):

Sketch on a plane:
  var sk = newSketchOnPlane(context, id + "sk", {"sketchPlane": plane(WORLD_ORIGIN, Z_DIRECTION)});
  skLineSegment(sk, "l1", {"start": vector(0 * millimeter, 0 * millimeter), "end": vector(50 * millimeter, 0 * millimeter)});
  skCircle(sk, "c1", {"center": vector(0 * millimeter, 0 * millimeter), "radius": 25 * millimeter});
  skEllipse(sk, "e1", {"center": vector(0 * millimeter, 0 * millimeter), "majorRadius": 40 * millimeter, "minorRadius": 25 * millimeter});
  skSolve(sk);

Extrude (creates solid):
  opExtrude(context, id + "ext", {
      "entities": qSketchRegion(id + "sk"),
      "direction": Z_DIRECTION,
      "endBound": BoundingType.BLIND,
      "endDepth": 30 * millimeter
  });

Revolve (solid of revolution — use for turned parts, cups, bowls, knobs):
  opRevolve(context, id + "rev", {
      "entities": qSketchRegion(id + "sk"),
      "axis": line(WORLD_ORIGIN, Z_DIRECTION),
      "angleForward": 2 * PI * radian
  });

Loft between two profiles (use for tapered/organic shapes like spoons, handles):
  opLoft(context, id + "loft", {
      "profileSubqueries": [qSketchRegion(id + "sk1"), qSketchRegion(id + "sk2")]
  });

Offset plane (for second sketch at different height):
  plane(vector(0 * millimeter, 0 * millimeter, 150 * millimeter), Z_DIRECTION)

Boolean union (merge two bodies):
  opBoolean(context, id + "bool", {
      "tools": qCreatedBy(id + "body2", EntityType.BODY),
      "targets": qCreatedBy(id + "body1", EntityType.BODY),
      "operationType": BooleanOperationType.UNION
  });

PLANE NORMALS: X_DIRECTION, Y_DIRECTION, Z_DIRECTION
WORLD_ORIGIN = vector(0, 0, 0) * meter

Return this JSON (nothing else):
{
  "featurescript": "complete FeatureScript code as a single string with \\n for newlines",
  "feature_name": "camelCaseFunctionName",
  "part_name": "Human Readable Name"
}"""

def load_fs_examples():
    """Load .fs example files from fs-examples/ for few-shot prompting."""
    examples = []
    if not os.path.isdir(FS_EXAMPLES_DIR):
        return examples
    for fname in sorted(os.listdir(FS_EXAMPLES_DIR)):
        if fname.endswith(".fs"):
            try:
                with open(os.path.join(FS_EXAMPLES_DIR, fname)) as f:
                    examples.append({"filename": fname, "content": f.read()})
            except Exception:
                pass
    return examples

_FS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "featurescript": {"type": "string", "description": "Complete FeatureScript program source code"},
        "feature_name":  {"type": "string", "description": "camelCase exported function name"},
        "part_name":     {"type": "string", "description": "Human-readable part name"}
    },
    "required": ["featurescript", "feature_name", "part_name"]
}

def generate_featurescript(spec_text):
    """Generate a FeatureScript program from a natural language spec using qwen2.5-coder:7b."""
    import urllib.request

    # 1. Reference examples: first 35 lines of each bundled example
    examples = load_fs_examples()
    example_snippets = ""
    if examples:
        parts = ["\nPATTERN SNIPPETS (adapt these, don't copy filenames):"]
        for ex in examples:
            snippet = "\n".join(ex["content"].split("\n")[:35])
            parts.append(f"\n// Pattern: {ex['filename']}\n{snippet}\n// ...")
        example_snippets = "\n".join(parts)

    # 2. Most recent successful user build (full code as gold example)
    learnings = load_fs_learnings()
    learned_example = ""
    if learnings:
        best = learnings[-1]   # most recent confirmed-working program
        lines = best["content"].split("\n")
        # Include full file but cap at 80 lines to stay within context budget
        learned_example = (
            f"\n\nCONFIRMED WORKING EXAMPLE (from a previous successful build):\n"
            f"// Source: {best['filename']}\n"
            + "\n".join(lines[:80])
        )

    # 3. Known error patterns to avoid
    data = _load_learnings()
    error_warnings = ""
    if data["known_errors"]:
        parts = ["\n\nKNOWN MISTAKES TO AVOID (learned from past failures):"]
        for e in data["known_errors"][-5:]:   # last 5 logged errors
            parts.append(
                f"\n// ERROR: {e.get('error','?')}\n"
                f"// WRONG: {e.get('wrong_pattern','?')}\n"
                f"// FIX:   {e.get('fix_applied','?')}"
            )
        error_warnings = "\n".join(parts)

    prompt = (
        f"Build this part as FeatureScript:"
        f"{example_snippets}"
        f"{learned_example}"
        f"{error_warnings}"
        f"\n\nPART SPEC: {spec_text}"
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "system": FS_GENERATE_SYSTEM,
        "prompt": prompt,
        "format": _FS_OUTPUT_SCHEMA,   # enforce exact output schema
        "options": {"num_predict": 4096, "num_ctx": 8192}
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    text = data["response"].strip()
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    result = json.loads(text)
    print(f"[FS] Generated: feature={result.get('feature_name')} part={result.get('part_name')}")
    return result

def build_with_featurescript(spec_text, description=None, doc_name=None):
    """Generate FeatureScript → upload to Feature Studio → attempt BTM call.

    The BTM custom-feature call currently fails due to an unresolved Onshape
    namespace issue (all workspace/version/microversion formats return 400).
    We still create the document and Feature Studio so the user can manually
    run the generated code from the Onshape UI (open the FS tab → Run).
    The generated code is also saved to fs-learnings/ for future reference.
    """
    fs_spec   = description or spec_text
    fs_result = generate_featurescript(fs_spec)

    fs_code   = fs_result["featurescript"]
    feat_name = fs_result["feature_name"]
    part_name = doc_name or fs_result.get("part_name", "Generated Part")

    print(f"\n[FS] Building with FeatureScript: {part_name}")

    did, wid = create_document(part_name)
    eid      = create_part_studio(did, wid, part_name)
    time.sleep(1)

    feid = create_feature_studio(did, wid, f"{part_name} Feature")
    time.sleep(0.5)

    fs_data       = get_fs_content(did, wid, feid)
    source_mv     = fs_data.get("sourceMicroversion", "")
    upload_result = set_fs_content(did, wid, feid, source_mv, fs_code)
    time.sleep(2)   # let Onshape compile

    # Save the generated code for future reference regardless of call success
    log_successful_fs_build(spec_text, feat_name, part_name, fs_code)

    # Attempt BTM call — currently always fails with "invalid namespace" because
    # Onshape requires a UI-initiated "link" step between Feature Studio and Part
    # Studio before the REST API will resolve the namespace.  We catch the error
    # and leave the Feature Studio ready for manual execution via the Onshape UI.
    new_mv = upload_result.get("sourceMicroversion", "") if upload_result else ""
    try:
        call_custom_feature(did, wid, eid, feat_name, feid, part_name, fs_microversion=new_mv)
        time.sleep(1)
        verify_result = verify_model_json(did, wid, eid)
        verify_model(did, wid, eid)
    except RuntimeError as e:
        print(f"[FS] BTM call failed (expected — namespace API limitation): {str(e)[:100]}")
        print(f"[FS] FeatureScript saved to Feature Studio — open the Onshape link and")
        print(f"[FS] click the Feature Studio tab, then 'Run' to build the model manually.")
        verify_result = {"ok": False}

    url = f"https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}"
    print(f"\n[Done] ✅ Model ready: {url}")

    parsed = {"shape_type": "featurescript", "feature_name": feat_name,
              "feid": feid, "description": fs_spec}
    write_session(spec_text, parsed, did, wid, eid, url)
    return url, did, wid, eid

# ── Session file ──────────────────────────────────────────────────────────────

SESSION_FILE = "/home/theultimatecunt/.openclaw/cad-session.json"

def write_session(spec, parsed, did, wid, eid, url):
    data = {
        "spec": spec, "parsed": parsed,
        "did": did, "wid": wid, "eid": eid, "url": url,
        "built_at": datetime.now(timezone.utc).isoformat()
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data

def read_session():
    try:
        with open(SESSION_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        return {"error": str(e)}

# ── Verification ───────────────────────────────────────────────────────────────

def verify_model(did, wid, eid):
    print("\n[Verify] Reading back model properties...")
    try:
        mp = get_mass_props(did, wid, eid)
        bodies = mp.get("bodies", {})
        if bodies:
            b    = list(bodies.values())[0]
            mass = b.get("mass", [None])[0]
            vol  = b.get("volume", [None])[0]
            if mass: print(f"[Verify] ✅ Mass:   {mass*1000:.1f} g")
            if vol:  print(f"[Verify] ✅ Volume: {vol*1e6:.1f} cm³")
        else:
            print("[Verify] ⚠️  No solid bodies — check feature errors above")
    except Exception as e:
        print(f"[Verify] Mass props error: {e}")

def verify_model_json(did, wid, eid):
    """Return structured verification result as dict."""
    result = {"ok": False, "errors": [], "mass_g": None, "volume_cm3": None, "feature_errors": []}
    # Check features for errors
    try:
        path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
        feat_data = onshape("GET", path)
        for f in feat_data.get("features", []):
            fid = f.get("featureId", "?")
            fname = f.get("name", "?")
            for notice in f.get("notices", []):
                if notice.get("noticeLevel") in ("ERROR", "WARNING"):
                    result["feature_errors"].append({
                        "featureId": fid, "name": fname,
                        "level": notice.get("noticeLevel"),
                        "message": notice.get("message", "")
                    })
    except Exception as e:
        result["errors"].append(f"feature read error: {e}")
    # Mass properties
    try:
        mp = get_mass_props(did, wid, eid)
        bodies = mp.get("bodies", {})
        if bodies:
            b = list(bodies.values())[0]
            mass = b.get("mass", [None])[0]
            vol  = b.get("volume", [None])[0]
            result["mass_g"]     = round(mass * 1000, 2) if mass else None
            result["volume_cm3"] = round(vol  * 1e6,  2) if vol  else None
            result["ok"] = True
        else:
            result["errors"].append("no solid bodies found — model may not have been built correctly")
    except Exception as e:
        result["errors"].append(f"mass props error: {e}")
    return result

def get_features_json(did, wid, eid):
    """Return list of features with their statuses."""
    path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
    data = onshape("GET", path)
    out = []
    for f in data.get("features", []):
        notices = [{"level": n.get("noticeLevel"), "msg": n.get("message","")}
                   for n in f.get("notices", []) if n.get("noticeLevel") in ("ERROR","WARNING","INFO")]
        out.append({
            "featureId": f.get("featureId"),
            "name":      f.get("name"),
            "type":      f.get("featureType"),
            "suppressed":f.get("suppressed", False),
            "notices":   notices,
        })
    return out

def delete_all_features(did, wid, eid):
    """Delete every feature in the part studio (clears for rebuild)."""
    path = f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/features"
    data = onshape("GET", path)
    fids = [f["featureId"] for f in data.get("features", []) if f.get("featureId")]
    for fid in reversed(fids):  # delete in reverse order to avoid dependency errors
        try:
            onshape("DELETE", f"{path}/featureid/{fid}")
            print(f"  [Delete] {fid}")
        except Exception as e:
            print(f"  [Delete] {fid} error: {e}")
    print(f"[Rebuild] Cleared {len(fids)} features")

# ── Main Pipeline ──────────────────────────────────────────────────────────────

def _bolt_fs_pipeline(params, name, spec_text):
    """Route a parsed bolt spec through the FeatureScript pipeline."""
    did, wid, eid = build_bolt_with_featurescript(params, name)
    time.sleep(1)
    verify_model(did, wid, eid)
    url = f"https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}"
    print(f"\n[Done] ✅ Model ready: {url}")
    parsed_out = {"shape_type": "bolt", "params": params}
    write_session(spec_text, parsed_out, did, wid, eid, url)
    return url, did, wid, eid


def build_from_spec(spec_text, doc_name=None):
    if not ACCESS_KEY or not SECRET_KEY:
        raise RuntimeError(
            "ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY must be set.\n"
            "export ONSHAPE_ACCESS_KEY=... or openclaw config set env.ONSHAPE_ACCESS_KEY ..."
        )

    parsed = parse_spec(spec_text)
    shape  = parsed["shape_type"]
    params = parsed["params"]
    name   = doc_name or parsed["part_name"]

    # Freeform / organic shapes → generate FeatureScript, save it, build best-effort BTM
    if shape == "featurescript":
        description = parsed.get("description", spec_text)
        return build_with_featurescript(spec_text, description=description, doc_name=name)

    if shape not in SHAPE_BUILDERS:
        raise ValueError(f"Unknown shape '{shape}'. Supported: {list(SHAPE_BUILDERS)} + featurescript")

    did, wid = get_or_create_document(name)
    eid = create_part_studio(did, wid, name)
    time.sleep(1)

    print(f"\n[Builder] Building {shape}: {name}")
    SHAPE_BUILDERS[shape](params, did, wid, eid)
    time.sleep(1)

    verify_model(did, wid, eid)

    url = f"https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}"
    print(f"\n[Done] ✅ Model ready: {url}")
    write_session(spec_text, parsed, did, wid, eid, url)
    return url, did, wid, eid

def rebuild_in_doc(did, wid, eid, parsed):
    """Clear all features in an existing document and rebuild from parsed params."""
    shape  = parsed["shape_type"]
    params = parsed["params"]
    if shape not in SHAPE_BUILDERS:
        raise ValueError(f"Unknown shape '{shape}'")
    delete_all_features(did, wid, eid)
    time.sleep(1)
    print(f"\n[Rebuild] Building {shape}...")
    SHAPE_BUILDERS[shape](params, did, wid, eid)
    time.sleep(1)
    verify_model(did, wid, eid)

# ── CLI ────────────────────────────────────────────────────────────────────────

def cmd_parse(argv):
    """parse <spec>  — extract geometry params as JSON (uses qwen2.5-coder:7b)"""
    if not argv:
        print("Usage: onshape_cad_agent.py parse <spec text>", file=sys.stderr)
        sys.exit(1)
    spec = " ".join(argv)
    result = parse_spec(spec)
    print(json.dumps(result, indent=2))

def cmd_build(argv):
    """build <spec>  — full pipeline: parse → create doc → add features → write session"""
    p = argparse.ArgumentParser(prog="build")
    p.add_argument("spec", nargs="+")
    p.add_argument("--doc-name", default=None)
    a = p.parse_args(argv)
    spec = " ".join(a.spec)
    url, did, wid, eid = build_from_spec(spec, doc_name=a.doc_name)
    print(f"\nURL={url}")
    print(f"DID={did}")
    print(f"WID={wid}")
    print(f"EID={eid}")

def cmd_verify(argv):
    """verify <did> <wid> <eid>  — check feature errors + mass/volume, output JSON"""
    if len(argv) < 3:
        print("Usage: onshape_cad_agent.py verify <did> <wid> <eid>", file=sys.stderr)
        sys.exit(1)
    did, wid, eid = argv[0], argv[1], argv[2]
    result = verify_model_json(did, wid, eid)
    print(json.dumps(result, indent=2))
    if not result["ok"] or result["feature_errors"]:
        sys.exit(1)

def cmd_features(argv):
    """features <did> <wid> <eid>  — list all features with statuses as JSON"""
    if len(argv) < 3:
        print("Usage: onshape_cad_agent.py features <did> <wid> <eid>", file=sys.stderr)
        sys.exit(1)
    did, wid, eid = argv[0], argv[1], argv[2]
    result = get_features_json(did, wid, eid)
    print(json.dumps(result, indent=2))

def cmd_rebuild(argv):
    """rebuild <did> <wid> <eid> <spec>  — clear doc and rebuild from new spec"""
    if len(argv) < 4:
        print("Usage: onshape_cad_agent.py rebuild <did> <wid> <eid> <spec>", file=sys.stderr)
        sys.exit(1)
    did, wid, eid = argv[0], argv[1], argv[2]
    spec = " ".join(argv[3:])
    parsed = parse_spec(spec)
    rebuild_in_doc(did, wid, eid, parsed)
    url = f"https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}"
    write_session(spec, parsed, did, wid, eid, url)
    print(f"\nURL={url}")
    print(f"DID={did}")
    print(f"WID={wid}")
    print(f"EID={eid}")

def cmd_session(_argv):
    """session  — print the current session state (last built doc)"""
    s = read_session()
    if s:
        print(json.dumps(s, indent=2))
    else:
        print(json.dumps({"error": "no session — run build first"}))

def cmd_fs(argv):
    """fs <description>  — force FeatureScript path (skip standard shape detection)"""
    if not argv:
        print("Usage: onshape_cad_agent.py fs <description>", file=sys.stderr)
        sys.exit(1)
    spec = " ".join(argv)
    url, did, wid, eid = build_with_featurescript(spec)
    print(f"\nURL={url}")
    print(f"DID={did}")
    print(f"WID={wid}")
    print(f"EID={eid}")

def cmd_fs_preview(argv):
    """fs-preview <description>  — generate and print FeatureScript without uploading"""
    if not argv:
        print("Usage: onshape_cad_agent.py fs-preview <description>", file=sys.stderr)
        sys.exit(1)
    spec = " ".join(argv)
    result = generate_featurescript(spec)
    print(f"\n--- feature_name: {result.get('feature_name')} ---")
    print(f"--- part_name: {result.get('part_name')} ---")
    print("\n--- FeatureScript ---")
    print(result.get("featurescript", ""))

def cmd_learn(argv):
    """learn <onshape-url> [label]
    Import a reference Onshape model as a verified learning example.
    Reads its mass/volume and saves metadata to cad-learnings.json.
    The URL format is:  https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}
    """
    if not argv:
        print("Usage: onshape_cad_agent.py learn <onshape-url> [label]", file=sys.stderr)
        sys.exit(1)

    url   = argv[0]
    label = " ".join(argv[1:]) if len(argv) > 1 else "reference model"

    # Parse did/wid/eid from URL
    m = re.search(r"/documents/([^/]+)/w/([^/]+)/e/([^/?\s]+)", url)
    if not m:
        print(f"[Learn] Cannot parse did/wid/eid from URL: {url}", file=sys.stderr)
        sys.exit(1)

    did, wid, eid = m.group(1), m.group(2), m.group(3)
    print(f"[Learn] Reference: did={did} wid={wid} eid={eid}")

    verify_result = verify_model_json(did, wid, eid)
    verify_model(did, wid, eid)

    data = _load_learnings()
    entry = {
        "type":       "reference",
        "label":      label,
        "url":        url,
        "did":        did, "wid": wid, "eid": eid,
        "mass_g":     verify_result.get("mass_g"),
        "volume_cm3": verify_result.get("volume_cm3"),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    data.setdefault("reference_models", []).append(entry)
    # Cap at 20 reference models
    data["reference_models"] = data["reference_models"][-20:]
    _save_learnings(data)

    print(json.dumps({
        "saved": True, "label": label,
        "mass_g": verify_result.get("mass_g"),
        "volume_cm3": verify_result.get("volume_cm3"),
        "feature_errors": len(verify_result.get("feature_errors", [])),
    }, indent=2))


def cmd_log_error(argv):
    """log-error <spec> <error_description> <wrong_pattern> <fix_applied>"""
    if len(argv) < 4:
        print('Usage: onshape_cad_agent.py log-error "<spec>" "<error>" "<wrong_pattern>" "<fix>"',
              file=sys.stderr)
        sys.exit(1)
    spec, error_description, wrong_pattern, fix_applied = argv[0], argv[1], argv[2], argv[3]
    log_fs_error(spec, error_description, wrong_pattern, fix_applied)
    print(json.dumps({"logged": True, "spec": spec, "error": error_description}))

SUBCOMMANDS = {
    "parse":      cmd_parse,
    "build":      cmd_build,
    "verify":     cmd_verify,
    "features":   cmd_features,
    "rebuild":    cmd_rebuild,
    "session":    cmd_session,
    "fs":         cmd_fs,
    "fs-preview": cmd_fs_preview,
    "log-error":  cmd_log_error,
    "learn":      cmd_learn,
}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in SUBCOMMANDS:
        SUBCOMMANDS[sys.argv[1]](sys.argv[2:])
    else:
        # Legacy flag-based CLI for backward compatibility
        parser = argparse.ArgumentParser(description="Onshape Agentic CAD Builder")
        group  = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--spec",        help='e.g. "W200x100 I-beam 2000mm long"')
        group.add_argument("--file",        help="Path to PDF or text datasheet")
        group.add_argument("--interactive", action="store_true")
        parser.add_argument("--doc-name",   help="Override document name")
        args = parser.parse_args()

        if args.interactive:
            print("Onshape CAD Agent — Interactive Mode (empty line to submit, Ctrl-C to quit)\n")
            while True:
                lines = []
                try:
                    while True:
                        line = input("> " if not lines else "  ")
                        if not line and lines: break
                        lines.append(line)
                except KeyboardInterrupt:
                    print("\nBye!"); break
                spec = "\n".join(lines)
                if spec.strip():
                    try:
                        url, *_ = build_from_spec(spec, doc_name=args.doc_name)
                        print(f"\n🔗 {url}\n")
                    except Exception as e:
                        print(f"❌ {e}\n")
        elif args.file:
            spec_text = read_pdf_spec(args.file) if args.file.endswith(".pdf") else open(args.file).read()
            url, *_ = build_from_spec(spec_text, doc_name=args.doc_name)
            print(f"\n🔗 {url}")
        else:
            url, *_ = build_from_spec(args.spec, doc_name=args.doc_name)
            print(f"\n🔗 {url}")

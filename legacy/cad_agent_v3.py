#!/usr/bin/env python3
"""
CAD Agent v3 — build123d code-gen loop + Onshape upload.

Two paths:
  code — LLM writes build123d Python → run locally → STEP → Onshape blob upload
  api  — structural sections, gears, bolts → delegate to cad_agent_v2.py (keeps parametric history)

Usage:
    python3 cad_agent_v3.py build "a 100x60x20mm electronics enclosure with 2mm walls"
    python3 cad_agent_v3.py build "W200x100 I-beam 1500mm"       # routes to api path
    python3 cad_agent_v3.py rate <1-5> [comment]
    python3 cad_agent_v3.py session
"""

import os
import sys
import json
import re
import time
import math
import base64
import logging
import subprocess
import tempfile
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent
_OPENCLAW  = Path.home() / ".openclaw"
LOG_FILE   = _OPENCLAW / "cad-agent.log"
FEEDBACK_FILE = _OPENCLAW / "cad-feedback.jsonl"
SESSION_FILE  = _OPENCLAW / "cad-session.json"
CONFIG_FILE   = _OPENCLAW / "openclaw.json"
SCRIPTS_DIR   = _HERE / "scripts"
B123D_DIR     = _HERE / "b123d"

# ── Models ────────────────────────────────────────────────────────────────────

BRIEF_MODEL    = "qwen3:8b"
CODE_MODEL     = "qwen2.5-coder:7b-instruct-q5_k_m"
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300
CODE_TIMEOUT   = 240   # allow longer for complex geometry

# ── Build config ──────────────────────────────────────────────────────────────

MAX_REPAIR_ATTEMPTS = 4
BUILD_TIMEOUT       = 480   # seconds total wall-clock
BASE_URL            = "https://cad.onshape.com"

# Keywords that route to the v2 API path (preserves Onshape parametric feature tree)
_API_KEYWORDS = {
    "i-beam", "h-beam", "i beam", "h beam",
    "uc section", "ub section", "ipn", "ipe",
    "wide flange", "w-section", "structural section",
    "c-channel", "angle iron", "angle section",
    "spur gear", "helical gear", "gear", "pinion", "sprocket", "cog",
    "bolt", "hex bolt", "hex screw", "cap screw", "fastener",
}

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stderr),
        ],
    )

_setup_logging()
log = logging.getLogger("cad_v3")

# ── Config / credentials ──────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _creds() -> tuple[str, str]:
    cfg = _load_config()
    env = cfg.get("env", {})
    ak = env.get("ONSHAPE_ACCESS_KEY") or os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = env.get("ONSHAPE_SECRET_KEY") or os.environ.get("ONSHAPE_SECRET_KEY", "")
    return ak, sk

def _tg_token() -> str:
    cfg = _load_config()
    return (cfg.get("channels", {}).get("telegram", {})
               .get("accounts", {}).get("cad", {}).get("botToken", ""))

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

# ── Ollama LLM call ───────────────────────────────────────────────────────────

def _ollama(model: str, system: str, prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    payload = {
        "model":  model,
        "stream": False,
        "system": system,
        "prompt": prompt,
        "options": {"num_ctx": 16384},
        "think":  False,
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp["response"].strip()

# ── Path routing ──────────────────────────────────────────────────────────────

def _detect_path(spec: str) -> str:
    sl = spec.lower()
    # Standard section designations e.g. W200x100, 250UC72.9, IPE200
    if re.search(r'\b\d{2,3}[A-Z]{2,3}[\d.]+\b', spec.upper()):
        return "api"
    for kw in _API_KEYWORDS:
        if kw in sl:
            return "api"
    return "code"

# ── Brief generation ──────────────────────────────────────────────────────────

_BRIEF_SYSTEM = """\
You are a CAD specification analyst. Convert the description into a structured build brief.
Return ONLY valid JSON — no explanation, no markdown fences:
{
  "name": "short document name (max 40 chars)",
  "description": "one-sentence description of the part",
  "path": "code",
  "dimensions": {"key": "value in mm — include all relevant dimensions"},
  "features": ["list any important features: holes, fillets, threads, etc."],
  "notes": ["any constraints or special requirements"]
}
All values in millimetres. path is always "code" here."""

def build_brief(spec: str) -> dict:
    raw = _ollama(BRIEF_MODEL, _BRIEF_SYSTEM, f"Spec: {spec}", timeout=OLLAMA_TIMEOUT)
    text = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: minimal brief
        return {
            "name": spec[:40],
            "description": spec,
            "path": "code",
            "dimensions": {},
            "features": [],
            "notes": [],
        }

# ── Code generation ───────────────────────────────────────────────────────────

_CODE_SYSTEM = """\
You are a build123d Python expert. Write a complete, runnable build123d script.

STRICT RULES:
1. Start with:
   from build123d import *
   (optionally) from b123d.domain import structural_section, spur_gear, hex_bolt
2. All dimensions in MILLIMETRES.
3. Centre the part at the origin unless assembly logic says otherwise.
4. Extrude along +Z axis by default.
5. Assign the final Part/Compound to `result`:
   result = bp.part
6. Do NOT call export_step() — the runner handles export.
7. No if __name__ == ... block needed.
8. Do NOT add any text explanation — return ONLY Python code.

Available domain helpers (import from b123d.domain):
  structural_section(d_mm, bf_mm, tf_mm, tw_mm, length_mm, mitre=False)
  spur_gear(teeth, module_mm, width_mm, bore_mm=0, pressure_angle=20)
  hex_bolt(size_mm, length_mm, pitch_mm=None)

build123d quick reference:
  Box(l, w, h)  Cylinder(r, h)  Sphere(r)
  with BuildPart() as bp: ...  → extrude(amount=depth)
  with BuildSketch(Plane.XY): Rectangle(w, h) / Circle(r) / Polyline(pts) + make_face()
  fillet(bp.part.edges(), radius=r)
  chamfer(bp.part.edges(), length=c)
  Plane.XY / Plane.YZ / Plane.XZ  and Plane.XY.offset(z)

IMPORTANT — Mode enum has exactly three values:
  Mode.ADD        (join to existing solid)
  Mode.SUBTRACT   (cut from existing solid — NOT Mode.SUBTRACTION)
  Mode.INTERSECT
Default is Mode.ADD so you only need to specify Mode.SUBTRACT or Mode.INTERSECT."""


def generate_code(brief: dict) -> str:
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
    raw = _ollama(CODE_MODEL, _CODE_SYSTEM, prompt, timeout=CODE_TIMEOUT)
    return _patch_code(_strip_fences(raw))

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```python\s*\n?", "", text)
    text = re.sub(r"^```\s*\n?",       "", text)
    return text.rstrip("`").strip()

# Common LLM hallucinations that are trivially fixable
_CODE_PATCHES = [
    (r"\bMode\.SUBTRACTION\b",  "Mode.SUBTRACT"),
    (r"\bMode\.REMOVE\b",       "Mode.SUBTRACT"),
    (r"\bMode\.CUT\b",          "Mode.SUBTRACT"),
    (r"\bMode\.DELETE\b",       "Mode.SUBTRACT"),
    (r"\.export_step\(",        "# export_step is called by the runner — removed: export_step("),
]

def _patch_code(code: str) -> str:
    for pattern, replacement in _CODE_PATCHES:
        code = re.sub(pattern, replacement, code)
    return code

# ── Code repair ───────────────────────────────────────────────────────────────

_REPAIR_SYSTEM = """\
You are debugging build123d Python code. Fix the error shown.
Return ONLY the corrected Python code — no explanation, no markdown fences."""

def repair_code(code: str, errors: list[str], attempt: int) -> str:
    err_block = "\n".join(errors)
    prompt = (
        f"Errors (attempt {attempt}):\n{err_block}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Fixed code:"
    )
    raw = _ollama(CODE_MODEL, _REPAIR_SYSTEM, prompt, timeout=CODE_TIMEOUT)
    return _patch_code(_strip_fences(raw))

# ── build123d runner ──────────────────────────────────────────────────────────

def run_step(code: str, work_dir: Path) -> tuple[Path, str]:
    """Write code to work_dir/build_source.py, run scripts/step. Returns (step_path, log_output)."""
    # Ensure b123d is importable from the script
    sys_path_inject = (
        f"import sys as _sys\n"
        f"_sys.path.insert(0, {str(_HERE)!r})\n\n"
    )
    src  = work_dir / "build_source.py"
    step = work_dir / "build_output.step"
    src.write_text(sys_path_inject + code)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "step"), str(src), str(step)],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or not step.exists():
        raise RuntimeError(output or "scripts/step exited non-zero with no output")
    return step, output

def run_inspect(step_path: Path) -> dict:
    """Validate the STEP file. Returns dict with valid, output, errors."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "inspect"), str(step_path)],
        capture_output=True, text=True, timeout=60,
    )
    output  = result.stdout + result.stderr
    valid   = result.returncode == 0
    errors  = [line for line in output.splitlines() if line.startswith("ERROR")]
    warnings = [line for line in output.splitlines() if line.startswith("WARNING")]
    return {"valid": valid, "output": output, "errors": errors, "warnings": warnings}

# ── Onshape STEP import ───────────────────────────────────────────────────────

def import_step_to_onshape(step_path: Path, name: str) -> dict:
    """Upload STEP as a blob element in a new Onshape document. Returns result dict."""
    doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": True})
    did = doc["id"]
    wid = doc["defaultWorkspace"]["id"]

    step_bytes = step_path.read_bytes()
    filename   = step_path.name
    boundary   = "--------cadv3boundary"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + step_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    ak, sk = _creds()
    creds  = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v5/blobelements/d/{did}/w/{wid}",
        data=body,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            blob_resp = json.loads(r.read())
        eid = blob_resp.get("id", blob_resp.get("elementId", ""))
        url = f"{BASE_URL}/documents/{did}/w/{wid}/e/{eid}"
    except Exception as e:
        log.warning("[v3] Blob upload failed: %s — returning document URL only", e)
        url = f"{BASE_URL}/documents/{did}/w/{wid}"
        eid = ""

    return {"url": url, "did": did, "wid": wid, "eid": eid}

# ── Onshape URL / document inspect ───────────────────────────────────────────

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
            for f in features_resp.get("features", [])
            if not f.get("suppressed")
        ]
    except Exception as e:
        feature_names = [f"(could not fetch features: {e})"]

    try:
        mass_resp = _onshape("GET",
            f"/api/v9/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties")
        bodies = mass_resp.get("bodies") or {}
        first = next(iter(bodies.values()), {}) if bodies else {}
        mass_g  = round(first.get("mass", [0])[0] * 1000, 1) if first else None
        vol_cm3 = round(first.get("volume", [0])[0] * 1e6, 2) if first else None
        mass_line = (f"Mass: {mass_g} g, Volume: {vol_cm3} cm³"
                     if mass_g else "No solid bodies.")
    except Exception:
        mass_line = "Mass: unknown"

    feat_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(feature_names)) or "  (none)"
    doc_summary = (
        f"Document: {doc_name}\n"
        f"Features ({len(feature_names)}):\n{feat_list}\n"
        f"{mass_line}"
    ).strip()

    system = (
        "You are a CAD assistant. Summarise the Onshape model in 2-4 sentences: "
        "what it appears to be, its key features, and approximate size. Be concise."
    )
    try:
        return _ollama(BRIEF_MODEL, system,
                       f"Describe this Onshape model:\n\n{doc_summary}",
                       timeout=OLLAMA_TIMEOUT)
    except Exception:
        return doc_summary


# ── API path delegate ─────────────────────────────────────────────────────────

def build_api_path(spec: str, chat_id: Optional[str] = None) -> dict:
    """Route structural/gear/bolt specs to cad_agent_v2.py for parametric Onshape build."""
    v2_script = _HERE / "cad_agent_v2.py"
    ak, sk = _creds()
    env = os.environ.copy()
    env["ONSHAPE_ACCESS_KEY"] = ak
    env["ONSHAPE_SECRET_KEY"] = sk

    cmd = [sys.executable, str(v2_script), "build", spec]
    if chat_id:
        cmd += ["--chat-id", chat_id]

    log.info("[v3] Delegating to v2 API path: %s", spec)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=BUILD_TIMEOUT, env=env)

    # v2 outputs JSON followed by a ">>> URL: ..." line
    url = None
    result_dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith(">>> URL:"):
            url = line.replace(">>> URL:", "").strip()
    try:
        # v2 prints the result dict before the URL line
        raw_json = proc.stdout[:proc.stdout.rfind(">>> URL:")].strip()
        result_dict = json.loads(raw_json)
    except Exception:
        pass

    if not url:
        url = result_dict.get("url")
    if not url:
        raise RuntimeError(f"v2 build failed:\n{proc.stdout[-800:]}\n{proc.stderr[-400:]}")

    result_dict["url"]  = url
    result_dict["path"] = "api_v2"
    return result_dict

# ── Few-shot retrieval ────────────────────────────────────────────────────────

def _load_fewshots(spec: str, n: int = 2) -> list[dict]:
    """Return up to n similar rated examples from cad-feedback.jsonl.
    Uses simple keyword overlap (semantic search if nomic-embed-text available later).
    """
    if not FEEDBACK_FILE.exists():
        return []
    spec_words = set(re.findall(r'\b[a-z]{3,}\b', spec.lower()))
    rows = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
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

# ── Main build ────────────────────────────────────────────────────────────────

def build(spec: str, chat_id: Optional[str] = None) -> dict:
    """
    Full build pipeline.  Returns result dict with at least:
      url, path, spec, has_bodies, build_time_s
    """
    log.info("=" * 60)
    log.info("[v3] build: %s", spec)
    t0 = time.monotonic()

    path = _detect_path(spec)
    log.info("[v3] path=%s", path)

    # ── API path ──────────────────────────────────────────────────────────────
    if path == "api":
        result = build_api_path(spec, chat_id=chat_id)
        result["spec"] = spec
        result["build_time_s"] = round(time.monotonic() - t0, 1)
        _write_session(result)
        return result

    # ── Code path ─────────────────────────────────────────────────────────────
    brief      = build_brief(spec)
    name       = brief.get("name", spec[:40])
    fewshots   = _load_fewshots(spec)

    # Inject few-shot examples into code generation prompt if available
    if fewshots:
        fs_note = "\n\nSIMILAR SUCCESSFUL BUILDS (reference these patterns):\n"
        for fs in fewshots:
            if fs.get("code"):
                fs_note += f"# Spec: {fs['spec']}\n{fs['code'][:600]}\n\n"
        brief["notes"] = brief.get("notes", []) + [fs_note.strip()]

    code = generate_code(brief)

    with tempfile.TemporaryDirectory(prefix="cadv3_") as work_str:
        work_dir = Path(work_str)
        step_path: Optional[Path] = None
        last_errors: list[str] = []

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            elapsed = time.monotonic() - t0
            if elapsed > BUILD_TIMEOUT:
                raise TimeoutError(
                    f"Build timeout ({BUILD_TIMEOUT}s) exceeded after {attempt-1} attempt(s)"
                )

            log.info("[v3] Attempt %d/%d", attempt, MAX_REPAIR_ATTEMPTS)

            # Run build123d
            try:
                step_path, step_log = run_step(code, work_dir)
                log.info("[v3] build123d OK:\n%s", step_log)
            except Exception as e:
                err = str(e)
                log.warning("[v3] Attempt %d build failed: %s", attempt, err[:300])
                last_errors = [err]
                if attempt < MAX_REPAIR_ATTEMPTS:
                    code = repair_code(code, [err], attempt)
                continue

            # Inspect geometry
            inspection = run_inspect(step_path)
            log.info("[v3] Inspect: valid=%s  %s", inspection["valid"], inspection["output"].strip())

            if inspection["valid"]:
                break

            last_errors = inspection["errors"]
            log.warning("[v3] Attempt %d inspect failed: %s", attempt, last_errors)
            if attempt < MAX_REPAIR_ATTEMPTS:
                code = repair_code(code, last_errors, attempt)

        if not step_path or not step_path.exists():
            raise RuntimeError(
                f"Build failed after {MAX_REPAIR_ATTEMPTS} attempts.\n" +
                "\n".join(last_errors)
            )

        # Upload to Onshape
        log.info("[v3] Uploading STEP to Onshape: %s", name)
        upload = import_step_to_onshape(step_path, name)

    result = {
        "spec":        spec,
        "path":        "code_v3",
        "url":         upload["url"],
        "did":         upload["did"],
        "wid":         upload["wid"],
        "eid":         upload.get("eid", ""),
        "has_bodies":  True,
        "code":        code,
        "brief":       brief,
        "build_time_s": round(time.monotonic() - t0, 1),
        "built_at":    datetime.now(timezone.utc).isoformat(),
    }

    _write_session(result)
    log.info("[v3] Done: url=%s  time=%.1fs", result["url"], result["build_time_s"])
    return result

# ── Feedback store ────────────────────────────────────────────────────────────

def store_feedback(result: dict, rating: int, comment: str = "") -> None:
    if rating < 4:
        return
    rows = []
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows = [r for r in rows if r.get("spec") != result.get("spec")]
    rows.append({
        "spec":       result.get("spec", ""),
        "path":       result.get("path", ""),
        "code":       result.get("code", ""),
        "url":        result.get("url", ""),
        "rating":     rating,
        "comment":    comment,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })
    if len(rows) > 50:
        rows = rows[-50:]
    with open(FEEDBACK_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    log.info("[v3] Stored %d★ feedback for: %s", rating, result.get("spec", "")[:50])

# ── Telegram helper ───────────────────────────────────────────────────────────

def _tg_send(chat_id: str, text: str) -> None:
    token = _tg_token()
    if not token:
        return
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, data=data), timeout=15
        )
    except Exception as e:
        log.warning("[v3] Telegram send failed: %s", e)

# ── CLI commands ──────────────────────────────────────────────────────────────

def _cmd_build(argv):
    import argparse
    p = argparse.ArgumentParser(prog="build")
    p.add_argument("spec", nargs="+")
    p.add_argument("--chat-id", default=None)
    a = p.parse_args(argv)
    spec = " ".join(a.spec)

    try:
        result = build(spec, chat_id=a.chat_id)
    except Exception as e:
        log.error("[v3] Build error: %s", e)
        print(json.dumps({"ok": False, "error": str(e), "spec": spec}))
        sys.exit(1)

    print(json.dumps(result, indent=2))
    url = result.get("url")
    if url:
        print(f"\n>>> URL: {url}")
    else:
        print("\n>>> No URL — build failed")


def _cmd_rate(argv):
    if not argv or not argv[0].isdigit():
        print("Usage: rate <1-5> [comment...]")
        sys.exit(1)
    rating  = int(argv[0])
    comment = " ".join(argv[1:]) if len(argv) > 1 else ""
    session = _read_session()
    if not session:
        print("No session found — run a build first.")
        sys.exit(1)
    store_feedback(session, rating, comment)
    print(f"Saved {'⭐' * rating} for: {session.get('spec', '')[:60]}")


def _cmd_session(argv):
    session = _read_session()
    if session:
        print(json.dumps(session, indent=2))
    else:
        print("No session.")


def _cmd_brief(argv):
    """Debug: just run build_brief on a spec."""
    spec = " ".join(argv)
    brief = build_brief(spec)
    print(json.dumps(brief, indent=2))


def _cmd_code(argv):
    """Debug: generate code for a spec without building."""
    spec  = " ".join(argv)
    brief = build_brief(spec)
    code  = generate_code(brief)
    print(code)


def _cmd_inspect(argv):
    """
    inspect <onshape_url>
    Fetch document info from Onshape and return an LLM-generated description as JSON.
    Output: {"ok": true, "description": "...", "url": "...", "did":..., "wid":..., "eid":...}
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
            "ok": True, "description": description,
            "url": url, "did": did, "wid": wid, "eid": eid,
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)


def _cmd_refine(argv):
    """
    refine <original_spec> <feedback> [<history_json>]
    Merge feedback into spec and print revised spec (plain text).
    Compatible with Satine's refine call convention.
    """
    if len(argv) < 2:
        print("Usage: refine <original_spec> <feedback> [<history_json>]")
        sys.exit(1)
    original = argv[0]
    feedback = argv[1]
    try:
        history = json.loads(argv[2]) if len(argv) > 2 else []
    except Exception:
        history = []

    history_str = ""
    if history:
        history_str = "\nPrevious iterations:\n" + "\n".join(
            f"  - {h.get('spec','')}" for h in history[-3:]
        )

    system = (
        "You are a CAD specification editor. Merge the feedback into the original spec to "
        "produce a revised spec. Return ONLY the new spec as a single sentence — no explanation."
    )
    prompt = (
        f"Original: {original}\n"
        f"Feedback: {feedback}"
        + history_str
        + "\n\nRevised spec:"
    )
    revised = _ollama(BRIEF_MODEL, system, prompt, timeout=OLLAMA_TIMEOUT).strip()
    if not revised or len(revised) < 5:
        revised = f"{original}, {feedback}"
    print(revised)


CMDS = {
    "build":   _cmd_build,
    "rate":    _cmd_rate,
    "session": _cmd_session,
    "brief":   _cmd_brief,
    "code":    _cmd_code,
    "refine":  _cmd_refine,
    "inspect": _cmd_inspect,
}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in CMDS:
        CMDS[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Usage: {sys.argv[0]} <{'|'.join(CMDS)}> [args...]")
        print("\nCommands:")
        print("  build <spec> [--chat-id <id>]  — code-gen loop + Onshape upload")
        print("  rate  <1-5> [comment]          — rate last build")
        print("  session                        — show last build session")
        print("  refine <orig> <feedback> [hist]— merge feedback into spec (Satine compat)")
        print("  brief <spec>                   — debug: show build brief only")
        print("  code  <spec>                   — debug: generate code only (no build)")
        sys.exit(1)

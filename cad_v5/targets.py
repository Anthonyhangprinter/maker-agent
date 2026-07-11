"""Pluggable output targets.

A target is `callable(step_path: Path, name: str) -> dict` that delivers the finished STEP somewhere
the user can see it, returning a result dict with at least {"url": str, "uploaded": bool}.

v5 default: the LOCAL CAD Viewer (earthtojake text-to-cad skill) — instant, offline, local-first.
Onshape upload (ported verbatim from v4.3) is opt-in via `--onshape`. fstl and file targets round
it out. Resolve a target name/flag with `resolve_target()`.
"""
import os
import json
import time
import base64
import shutil
import socket
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Optional

from . import config
from .config import (log, creds, BASE_URL, TRANSLATE_TIMEOUT, CAD_VIEWER_PORT, STL_VIEWER_CMD,
                     _OPENCLAW, STEP_OUT, STL_OUT)

TargetFn = Callable[[Path, str], dict]

# ── Onshape REST (only used by the onshape target) ──────────────────────────────

def _onshape(method: str, path: str, body=None) -> dict:
    ak, sk = creds()
    cr = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    headers = {"Authorization": f"Basic {cr}", "Accept": "application/json;charset=UTF-8",
               "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Onshape {method} {path} → {e.code}: {e.read().decode()[:400]}")

def _onshape_multipart(path: str, fields: dict, file_path: Path, filename: str) -> dict:
    boundary = "----cadv5boundary"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    ak, sk = creds()
    cr = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    req = urllib.request.Request(
        BASE_URL + path, data=body, method="POST",
        headers={"Authorization": f"Basic {cr}", "Accept": "application/json;charset=UTF-8",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def onshape_target(step_path: Path, name: str, public: bool = True) -> dict:
    """Upload the STEP and TRANSLATE it into a real Onshape Part Studio, verifying ≥1 body landed
    (Onshape reports DONE even when an invalid STEP yields an empty studio). Ported from v4.3."""
    try:
        doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": public})
    except RuntimeError as e:
        if "409" in str(e) and "public" in str(e).lower() and not public:
            log.warning("[v5] Onshape rejected a private document (free account) — creating PUBLIC.")
            doc = _onshape("POST", "/api/v9/documents", {"name": name, "isPublic": True})
            public = True
        else:
            raise
    did = doc["id"]; wid = doc["defaultWorkspace"]["id"]
    doc_url = f"{BASE_URL}/documents/{did}/w/{wid}"
    try:
        resp = _onshape_multipart(
            f"/api/blobelements/d/{did}/w/{wid}",
            {"translate": "true", "storeInDocument": "true",
             "flattenAssemblies": "false", "yAxisIsUp": "false"}, step_path, step_path.name)
    except Exception as e:
        raise RuntimeError(f"Onshape upload failed ({e}). Empty document at {doc_url}")
    tid = resp.get("translationId")
    if not tid:
        raise RuntimeError(f"Onshape did not start a STEP translation: {str(resp)[:200]}. "
                           f"Empty document at {doc_url}")
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
            try:
                bd = _onshape("GET", f"/api/partstudios/d/{did}/w/{wid}/e/{eid}/bodydetails")
                n_bodies = len(bd.get("bodies", []))
            except Exception as e:
                n_bodies = -1
                log.warning("[v5] Could not verify imported bodies (%s)", e)
            if n_bodies == 0:
                raise RuntimeError(
                    f"Onshape translation produced an EMPTY Part Studio (0 bodies) — the STEP "
                    f"imported with no geometry (commonly an invalid fillet/chamfer or self-"
                    f"intersecting blend). Fix the geometry and re-export. Document at {doc_url}")
            log.info("[v5] Translated STEP → Part Studio %s (%s body/ies)", eid, n_bodies)
            return {"url": f"{doc_url}/e/{eid}", "did": did, "wid": wid, "eid": eid,
                    "uploaded": True, "public": public, "bodies": n_bodies, "target": "onshape"}
        if state == "FAILED":
            raise RuntimeError(f"Onshape translation failed: {t.get('failureReason')}. "
                               f"Document at {doc_url}")
        time.sleep(2)
    raise RuntimeError(f"Onshape translation timed out after {TRANSLATE_TIMEOUT}s "
                       f"(last state {state}). Document at {doc_url}")

# ── Local CAD Viewer (default) ──────────────────────────────────────────────────

def _viewer_backend() -> Optional[Path]:
    """Locate the earthtojake CAD Viewer node backend (server.mjs). Override with CAD_VIEWER_BACKEND."""
    env = os.environ.get("CAD_VIEWER_BACKEND")
    if env and Path(env).exists():
        return Path(env)
    base = Path.home() / ".claude/plugins/cache/text-to-cad/cad"
    if base.exists():
        # newest installed version that ships a viewer backend
        for ver in sorted(base.iterdir(), reverse=True):
            cand = ver / "skills/cad-viewer/scripts/viewer/backend/server.mjs"
            if cand.exists():
                return cand
    return None

def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def _ensure_viewer(serve_dir: Path, port: int) -> bool:
    """Start the CAD Viewer server on `port` serving `serve_dir` if nothing is listening. Returns
    True if a server is up (reused or freshly started), False if it could not be started."""
    if _port_open(port):
        return True   # reuse whatever is already there (a prior build's viewer)
    backend = _viewer_backend()
    if backend is None:
        log.warning("[v5] CAD Viewer backend not found (install the text-to-cad plugin, or set "
                    "CAD_VIEWER_BACKEND). Falling back to file target.")
        return False
    try:
        subprocess.Popen(
            ["node", str(backend), "--host", "127.0.0.1", "--port", str(port),
             "--dir", str(serve_dir), "--shutdown-after", "12h"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except FileNotFoundError:
        log.warning("[v5] `node` not found — cannot start CAD Viewer. Falling back to file target.")
        return False
    for _ in range(20):   # wait up to ~10s for the port to come up
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False

def cad_viewer_target(step_path: Path, name: str) -> dict:
    """Serve the STEP through the local CAD Viewer and return a browser link. Local + offline."""
    serve_dir = _OPENCLAW
    # The viewer serves a directory tree and `file=` takes a path RELATIVE to it — so anything
    # already under ~/.openclaw (cad-builds/<ts>/build.step, cad-last-build.step) is served in
    # place, keeping the per-build artifact dirs browsable in the viewer's picker. Only artifacts
    # from OUTSIDE the tree get copied in (the old behavior copied every build to the root, so
    # the bare viewer URL showed a pile of loose files and no build history).
    try:
        rel = step_path.resolve().relative_to(serve_dir.resolve())
    except ValueError:
        dest = serve_dir / step_path.name
        try:
            shutil.copy(step_path, dest)
        except Exception:
            dest = step_path
        step_path = dest
        rel = Path(step_path.name)
    # Drop any stale hidden GLB sidecar so the viewer regenerates from THIS step. The viewer caches
    # by mtime; a leftover newer-than-step .glb (e.g. from an earlier part) would otherwise be served
    # forever, immune to browser refresh.
    stale_glb = step_path.with_name("." + step_path.name + ".glb")
    try:
        stale_glb.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    up = _ensure_viewer(serve_dir, CAD_VIEWER_PORT)
    if not up:
        return file_target(step_path, name)
    url = (f"http://127.0.0.1:{CAD_VIEWER_PORT}/?dir={serve_dir}&file={rel}")
    log.info("[v5] CAD Viewer: %s", url)
    return {"url": url, "uploaded": False, "viewer": True, "target": "cad-viewer",
            "step_local": str(step_path)}

def freecad_target(step_path: Path, name: str) -> dict:
    """E2: native FreeCAD document. Verified parametric feature tree (Params spreadsheet +
    Part primitives/booleans) when the part fits the reconstructor's grammar; plain native
    import otherwise — result carries `parametric` so nothing overclaims."""
    from . import freecad_export
    r = freecad_export.convert(step_path, name)
    r.update({"url": "", "uploaded": False, "target": "freecad",
              "step_local": str(step_path)})
    return r

# ── fstl / file ─────────────────────────────────────────────────────────────────

def fstl_target(step_path: Path, name: str) -> dict:
    """Open the sliceable STL in the local fstl GUI (best-effort)."""
    stl = STL_OUT if STL_OUT.exists() else step_path.with_suffix(".stl")
    opened = False
    if shutil.which(STL_VIEWER_CMD) and stl.exists():
        try:
            subprocess.Popen([STL_VIEWER_CMD, str(stl)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            opened = True
        except Exception as e:
            log.warning("[v5] fstl launch failed: %s", e)
    return {"url": "", "uploaded": False, "target": "fstl", "opened": opened,
            "stl_local": str(stl) if stl.exists() else ""}

def file_target(step_path: Path, name: str) -> dict:
    """No viewer — just report where the artifacts landed on disk."""
    return {"url": "", "uploaded": False, "target": "file", "step_local": str(step_path)}

# ── print (M9/X2a) ────────────────────────────────────────────────────────────

def print_target(step_path: Path, name: str) -> dict:
    """Slice the build into dry-run-validated FDM gcode (OrcaSlicer CLI, Bambu profiles).
    Deterministic, no LLM. Graceful like every other target: infra failures (no slicer, no
    xvfb-run, unresolvable profile) set `target_error` and return; geometry/dry-run failures
    (bad mesh, out-of-bed, no extrusion) are NOT exceptions — they come back as a normal result
    with `print_ok: false` + `print_fails` so the caller can show them like any other advisory."""
    result = {"url": "", "uploaded": False, "target": "print", "step_local": str(step_path)}
    stl_path = STL_OUT if STL_OUT.exists() and step_path.resolve() == STEP_OUT.resolve() \
        else step_path.with_suffix(".stl")
    try:
        if not stl_path.exists():
            from . import engine
            stl_path = engine.run_stl(step_path, step_path.with_suffix(".stl"))
    except Exception as e:
        result["target_error"] = f"STL export failed (needed to slice): {e}"[:300]
        return result

    from . import cam_print
    pc = config.print_config()
    out_dir = step_path.parent / "print"
    try:
        sliced = cam_print.slice_stl(stl_path, out_dir, machine=pc.get("machine"),
                                      process=pc.get("process"), filament=pc.get("filament"))
    except Exception as e:
        result["target_error"] = f"slicing failed: {e}"[:300]
        return result

    machine_name = sliced.get("machine") or pc.get("machine") or config.PRINT_MACHINE_DEFAULT
    gcode = sliced.get("gcode")
    fails, notes, facts = cam_print.validate_gcode(gcode, sliced.get("result"), machine_name)
    result.update({
        "gcode_local": str(gcode) if gcode else "",
        "print_ok": not fails,
        "print_fails": fails,
        "print_notes": notes,
        "print_facts": facts,
    })
    if sliced.get("stderr_tail"):
        result["stderr_tail"] = sliced["stderr_tail"]
    return result

# ── Resolver ────────────────────────────────────────────────────────────────────

_TARGETS: dict[str, TargetFn] = {
    "cad-viewer": cad_viewer_target,
    "viewer":     cad_viewer_target,
    "onshape":    lambda p, n: onshape_target(p, n, public=config.public_uploads()),
    "freecad":    freecad_target,
    "fstl":       fstl_target,
    "file":       file_target,
    "print":      print_target,
}

def resolve_target(name: str) -> TargetFn:
    """Map a target name (config `cad.output_target`, a CLI flag, or the default) to its callable."""
    key = (name or "").strip().lower() or "cad-viewer"
    if key not in _TARGETS:
        log.warning("[v5] Unknown output target %r — using cad-viewer.", name)
        key = "cad-viewer"
    return _TARGETS[key]

def default_target_name() -> str:
    """v5 default is the local CAD Viewer; overridable via openclaw.json cad.output_target."""
    return (config.load_config().get("cad", {}).get("output_target") or "cad-viewer")

"""Mesh capability — CADAM's "Mesh" button for our web UI.

Text (and/or reference image) -> organic 3D mesh via a hosted image-to-3D provider.
This is the CREATIVE pipeline, separate from parametric CAD: no gate, no dimensions —
you get a sculpted mesh (GLB/OBJ), not engineering geometry.

Provider seam: first backend is Meshy (api.meshy.ai — own free tier, no fal.ai middleman).
  text-to-3d:  POST /openapi/v2/text-to-3d  {mode: "preview", prompt}   (~5 credits)
  image-to-3d: POST /openapi/v1/image-to-3d {image_url: data URI}
Poll the task, download model_urls.glb/.obj + thumbnail. Docs: docs.meshy.ai.

Key lookup mirrors the engine's cloud-key path: env MESHY_API_KEY, else the `env` block of
~/.openclaw/openclaw.json. No key -> /api/mesh answers 503 with setup instructions.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

MESHES_DIR = Path.home() / ".openclaw" / "cad-web" / "meshes"
LOCAL_RUNNER = Path.home() / ".openclaw" / "mesh-local" / "generate.sh"
_BASE = "https://api.meshy.ai"
POLL_S = 6
TASK_TIMEOUT = 15 * 60
LOCAL_TIMEOUT = 20 * 60          # CPU TripoSR takes minutes; be generous


def provider() -> str:
    """local-first: TripoSR on this box is the default; Meshy only by explicit opt-in."""
    if os.environ.get("MESH_PROVIDER") == "meshy":
        return "meshy"
    return "triposr-local" if LOCAL_RUNNER.is_file() else "meshy"


def api_key() -> str:
    if os.environ.get("MESHY_API_KEY"):
        return os.environ["MESHY_API_KEY"]
    try:
        cfg = json.loads((Path.home() / ".openclaw" / "openclaw.json").read_text())
        return cfg.get("env", {}).get("MESHY_API_KEY", "")
    except Exception:
        return ""


def _req(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        _BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _download(url: str, dst: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            dst.write_bytes(r.read())
        return True
    except Exception:
        return False


def _data_uri(image_path: str) -> str:
    import base64
    p = Path(image_path)
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"} \
        .get(p.suffix.lstrip(".").lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def run_mesh_local(job: dict) -> None:
    """LOCAL image->3D via TripoSR (~/.openclaw/mesh-local/generate.sh). CPU inference is
    minutes per mesh on this box — the log keeps the user informed while it grinds."""
    import subprocess
    log = job["log"].append
    job["status"] = "running"
    out = MESHES_DIR / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    log("mesh: local TripoSR starting (CPU — a few minutes; fully offline)")
    t0 = time.monotonic()
    try:
        p = subprocess.run([str(LOCAL_RUNNER), job["image"], str(out)],
                           capture_output=True, text=True, timeout=LOCAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        job["status"], job["error"] = "error", "local mesh timed out"
        return
    for line in (p.stderr or "").splitlines()[-8:]:
        if line.strip():
            log("triposr: " + line.strip()[:160])
    arts = {}
    for key, fname in (("glb", "model.glb"), ("obj", "model.obj"), ("render", "thumb.png")):
        if (out / fname).is_file() and (out / fname).stat().st_size > 0:
            arts[key] = f"/meshes/{job['id']}/{fname}"
    if not arts.get("glb") and not arts.get("obj"):
        tail = (p.stderr or p.stdout or "no output").strip()[-300:]
        job["status"], job["error"] = "error", f"TripoSR produced no mesh: …{tail}"
        return
    job["result"] = {"ok": True, "kind": "mesh", "artifacts": arts,
                     "provider": "triposr-local",
                     "build_time_s": round(time.monotonic() - t0, 1)}
    job["status"] = "done"


def run_mesh_job(job: dict) -> None:
    """Worker entry: drives one mesh generation start->files. Mutates job in place the same
    way _run_build does (status/log/result)."""
    if provider() == "triposr-local":
        run_mesh_local(job)
        return
    log = job["log"].append
    job["status"] = "running"
    prompt, image = job["spec"], job["image"]
    try:
        if image:
            log("mesh: submitting image-to-3d task")
            body = {"image_url": _data_uri(image), "should_texture": True,
                    "enable_pbr": False, "should_remesh": True}
            if prompt:
                body["texture_prompt"] = prompt
            task_id = _req("POST", "/openapi/v1/image-to-3d", body)["result"]
            poll_path = f"/openapi/v1/image-to-3d/{task_id}"
        else:
            log("mesh: submitting text-to-3d preview task")
            task_id = _req("POST", "/openapi/v2/text-to-3d", {
                "mode": "preview", "prompt": prompt, "should_remesh": True})["result"]
            poll_path = f"/openapi/v2/text-to-3d/{task_id}"
        log(f"mesh: task {task_id}")

        t0 = time.monotonic()
        task = None
        while time.monotonic() - t0 < TASK_TIMEOUT:
            task = _req("GET", poll_path)
            st, prog = task.get("status"), task.get("progress", 0)
            log(f"mesh: {st} {prog}%")
            if st in ("SUCCEEDED", "FAILED", "CANCELED"):
                break
            time.sleep(POLL_S)
        if not task or task.get("status") != "SUCCEEDED":
            err = (task or {}).get("task_error", {}).get("message") or \
                  f"mesh task ended: {(task or {}).get('status', 'timeout')}"
            job["status"], job["error"] = "error", err
            return

        out = MESHES_DIR / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        urls = task.get("model_urls") or {}
        arts = {}
        for key, fname in (("glb", "model.glb"), ("obj", "model.obj")):
            if urls.get(key) and _download(urls[key], out / fname):
                arts[key] = f"/meshes/{job['id']}/{fname}"
        if task.get("thumbnail_url") and _download(task["thumbnail_url"], out / "thumb.png"):
            arts["render"] = f"/meshes/{job['id']}/thumb.png"
        if not arts:
            job["status"], job["error"] = "error", "mesh succeeded but no files downloadable"
            return
        job["result"] = {"ok": True, "kind": "mesh", "artifacts": arts,
                         "provider": "meshy",
                         "build_time_s": round(time.monotonic() - t0, 1)}
        job["status"] = "done"
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:200]
        except Exception:
            pass
        job["status"], job["error"] = "error", f"meshy API {e.code}: {detail or e.reason}"
    except Exception as e:
        job["status"], job["error"] = "error", f"mesh failed: {e}"

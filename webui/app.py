"""CAD agent web UI — CADAM-style front door for the v5 engine, tailnet-only.

A deliberately small FastAPI app mirroring Satine's shape: one queue, one worker thread,
builds run as the same `python3 -m cad_v5 … --once --json` subprocess (stdout = exactly one
JSON line; live progress arrives on stderr and is streamed into the job's log for the browser
to poll). The engine's machine-wide flock serializes GPU use across frontends — this app's
queue only orders ITS OWN jobs; a build may additionally wait on the lock behind Satine/CLI,
surfaced as status "waiting_gpu".

Bind: 127.0.0.1:8090 (loopback only). Tailnet exposure is `tailscale serve --bg --https=8443
http://127.0.0.1:8090` — see RUNBOOK.md. Never bind a public interface.

v1 limitation (stated in the UI): jobs live in memory — a service restart loses in-flight
builds (the engine's per-build artifact dirs survive; only the job bookkeeping is lost).
"""
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

# The engine must run under the SYSTEM interpreter (build123d/OCP live there) — NOT this
# app's venv python (sys.executable here), which only carries fastapi/uvicorn. Using
# sys.executable propagated the venv into the engine's own scripts/step subprocesses and
# every build123d run died on ImportError (found live 2026-07-18).
ENGINE_PYTHON = "/usr/bin/python3"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

SKILL_ROOT  = Path(__file__).resolve().parent.parent          # …/skills/cad-builder
STATIC_DIR  = Path(__file__).resolve().parent / "static"
BUILDS_DIR  = (Path.home() / ".openclaw" / "cad-builds").resolve()
UPLOADS_DIR = Path.home() / ".openclaw" / "cad-web" / "uploads"
MAX_UPLOAD  = 10 * 1024 * 1024          # 10 MB
BUILD_TIMEOUT = 2 * 1860                # engine budget + a full lock wait (matches Satine)
LOG_TAIL    = 40
ARTIFACT_EXTS = {".step", ".stl", ".dxf", ".png", ".py", ".jpg"}
CODERS = {"auto", "fast", "strong"}

# magic bytes → extension (content decides, not the filename)
_MAGIC = [(b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"), (b"RIFF", ".webp")]

app = FastAPI(title="CAD agent", docs_url=None, redoc_url=None)

_jobs: dict[str, dict] = {}             # job_id -> job (in-memory, v1)
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()

_WAIT_RE = re.compile(r"waiting for build lock")


def _job_public(job: dict) -> dict:
    return {
        "id": job["id"], "spec": job["spec"], "coder": job["coder"],
        "has_image": bool(job["image"]), "status": job["status"],
        "queued_ahead": job.get("queued_ahead", 0),
        "log_tail": list(job["log"])[-LOG_TAIL:],
        "result": job.get("result"), "error": job.get("error"),
        "created_at": job["created_at"],
    }


def _sniff_ext(head: bytes) -> str | None:
    for magic, ext in _MAGIC:
        if head.startswith(magic):
            if ext == ".webp" and head[8:12] != b"WEBP":
                continue
            return ext
    return None


def _worker():
    while True:
        job_id = _queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            continue
        try:
            _run_build(job)
        except Exception as e:
            job["status"], job["error"] = "error", f"internal error: {e}"
        finally:
            _queue.task_done()


def _run_build(job: dict):
    job["status"] = "running"
    cmd = [ENGINE_PYTHON, "-m", "cad_v5", job["spec"], "--once", "--json", "--ask",
           "--coder", job["coder"], "--target", "file"]
    if job["image"]:
        cmd += ["--image", job["image"]]
    env = {**os.environ, "CAD_FRONTEND": "web"}
    proc = subprocess.Popen(cmd, cwd=str(SKILL_ROOT), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            job["log"].append(line)
            # The engine logs this exact line while blocked behind another frontend's build.
            if _WAIT_RE.search(line):
                job["status"] = "waiting_gpu"
            elif job["status"] == "waiting_gpu":
                job["status"] = "running"

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()
    try:
        stdout, _ = proc.communicate(timeout=BUILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        job["status"], job["error"] = "error", "build timed out"
        return
    t.join(timeout=5)

    result = None
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except Exception:
                pass
            break
    if result is None:
        job["status"] = "error"
        job["error"] = (list(job["log"])[-1] if job["log"] else "no output from engine")
        return
    _ensure_render(result)
    job["result"] = _result_public(result)
    if result.get("needs_clarification"):
        job["status"] = "needs_clarification"
    elif result.get("ok"):
        job["status"] = "done"
    else:
        job["status"], job["error"] = "error", result.get("error", "build failed")


def _ensure_render(result: dict):
    """--json builds skip the engine's final render (final_render=False in machine mode), so
    the web preview would be empty. Generate build.png here (CPU-only, ~10s) — best-effort."""
    build_dir = result.get("build_dir") or ""
    step = Path(build_dir) / "build.step" if build_dir else None
    if not step or not step.is_file() or (step.parent / "build.png").exists():
        return
    try:
        subprocess.run([ENGINE_PYTHON, str(SKILL_ROOT / "scripts" / "render"),
                        str(step), str(step.parent / "build.png")],
                       timeout=120, capture_output=True)
    except Exception:
        pass


def _result_public(result: dict) -> dict:
    """Strip the result to what the page needs, with artifact paths rewritten to URLs."""
    out = {k: result.get(k) for k in
           ("ok", "converged", "accepted_via", "code_model", "turns", "build_time_s",
            "last_critique", "warning", "image_analysis", "image_only",
            "needs_clarification", "questions", "spec", "error")}
    build_dir = result.get("build_dir") or ""
    if build_dir:
        bid = Path(build_dir).name
        arts = {}
        for key, fname in (("render", "build.png"), ("step", "build.step"),
                           ("stl", "build.stl"), ("dxf", "build.dxf"),
                           ("reference", "reference.jpg"), ("source", "build_source.py")):
            if (BUILDS_DIR / bid / fname).exists():
                arts[key] = f"/artifacts/{bid}/{fname}"
        out["artifacts"], out["build_id"] = arts, bid
    return {k: v for k, v in out.items() if v not in (None, "")}


@app.post("/api/build")
async def api_build(spec: str = Form(""), coder: str = Form("auto"),
                    image: UploadFile | None = File(None)):
    spec = spec.strip()
    if not spec and not (image is not None and image.filename):
        raise HTTPException(400, "provide a spec, an image, or both")
    if coder not in CODERS:
        raise HTTPException(400, f"coder must be one of {sorted(CODERS)}")
    image_path = None
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "image too large (max 10 MB)")
        ext = _sniff_ext(data[:16])
        if ext is None:
            raise HTTPException(400, "unsupported image type (jpg/png/webp)")
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        image_path = str(UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}")
        Path(image_path).write_bytes(data)
    job = {
        "id": uuid.uuid4().hex[:12], "spec": spec, "coder": coder, "image": image_path,
        "status": "queued", "log": deque(maxlen=300), "result": None, "error": None,
        "created_at": time.time(),
    }
    with _jobs_lock:
        job["queued_ahead"] = sum(1 for j in _jobs.values()
                                  if j["status"] in ("queued", "running", "waiting_gpu"))
        _jobs[job["id"]] = job
    _queue.put(job["id"])
    return {"job_id": job["id"], "queued_ahead": job["queued_ahead"]}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (jobs do not survive a server restart)")
    return _job_public(job)


@app.get("/api/jobs")
def api_jobs():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:20]
        return [_job_public(j) for j in jobs]


@app.get("/artifacts/{build_id}/{filename}")
def artifacts(build_id: str, filename: str):
    target = (BUILDS_DIR / build_id / filename).resolve()
    if not str(target).startswith(str(BUILDS_DIR) + os.sep):
        raise HTTPException(404, "not found")
    if target.suffix.lower() not in ARTIFACT_EXTS:
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found (old builds are pruned after 200 kept)")
    media = {".png": "image/png", ".jpg": "image/jpeg",
             ".py": "text/plain"}.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(target, media_type=media, filename=filename)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "queue": _queue.qsize()})


threading.Thread(target=_worker, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)

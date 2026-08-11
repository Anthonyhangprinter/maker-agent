"""CAD agent web UI — CADAM-style front door for the v5 engine, tailnet-only.

A deliberately small FastAPI app mirroring Satine's shape: one queue, one worker thread,
builds run as the same `python3 -m cad_v5 … --once --json` subprocess (stdout = exactly one
JSON line; live progress arrives on stderr and is streamed into the job's log for the browser
to poll). The engine's machine-wide flock serializes GPU use across frontends — this app's
queue only orders ITS OWN jobs; a build may additionally wait on the lock behind Satine/CLI,
surfaced as status "waiting_gpu".

Bind: 127.0.0.1:8090 (loopback only). Tailnet exposure is `tailscale serve --bg --https=8443
http://127.0.0.1:8090` — see RUNBOOK.md. Never bind a public interface.

Jobs ("creations") persist to ~/.openclaw/cad-web/sessions.json so the page's history rail
survives a restart. In-flight builds still cannot: their subprocess dies with the service, so
a job reloaded in a running state is marked errored rather than silently resumed.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import deque
from pathlib import Path

# The engine must run under the SYSTEM interpreter (build123d/OCP live there) — NOT this
# app's venv python (sys.executable here), which only carries fastapi/uvicorn. Using
# sys.executable propagated the venv into the engine's own scripts/step subprocesses and
# every build123d run died on ImportError (found live 2026-07-18).
ENGINE_PYTHON = "/usr/bin/python3"

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import mesh as meshmod

SKILL_ROOT  = Path(__file__).resolve().parent.parent          # …/skills/cad-builder
STATIC_DIR  = Path(__file__).resolve().parent / "static"
BUILDS_DIR  = (Path.home() / ".openclaw" / "cad-builds").resolve()
UPLOADS_DIR = Path.home() / ".openclaw" / "cad-web" / "uploads"
SESSIONS_FILE = Path.home() / ".openclaw" / "cad-web" / "sessions.json"
MAX_SESSIONS = 200                      # matches the engine's KEEP_BUILDS artifact rotation
MAX_UPLOAD  = 10 * 1024 * 1024          # 10 MB
BUILD_TIMEOUT = 2 * 1860                # engine budget + a full lock wait (matches Satine)
LOG_TAIL    = 40
ARTIFACT_EXTS = {".step", ".stl", ".dxf", ".png", ".py", ".jpg", ".scad"}
CODERS = {"auto", "fast", "strong"}
# Titles are named by the same small model the engine uses for its brief (cad_v5/config.py).
TITLE_MODEL = "qwen3:8b"
OLLAMA_GEN  = "http://localhost:11434/api/generate"
# Files copied aside before a revise overwrites them in place (fluid_gen revises the SAME dir).
TURN_FILES = (("render", "build.png"), ("step", "build.step"),
              ("stl", "build.stl"), ("source", "build_source.py"))

# magic bytes → extension (content decides, not the filename)
_MAGIC = [(b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"), (b"RIFF", ".webp")]

app = FastAPI(title="CAD agent", docs_url=None, redoc_url=None)

# ── Beta-tester identity + owner pings ─────────────────────────────────────────
# `tailscale serve` stamps every proxied request with the visitor's tailnet identity
# (Tailscale-User-Login / Tailscale-User-Name headers); loopback requests carry neither
# and are the owner at the keyboard. Jobs from anyone NOT in OWNER_LOGINS ping the
# owner's Telegram through Satine's bot (token read from openclaw.json at send time,
# never cached — the gateway owns that file).
OWNER_LOGINS  = {"anthonyromanelli10@gmail.com"}
OWNER_CHAT_ID = "7788781234"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


def _bot_token() -> str | None:
    try:
        cfg = json.loads(OPENCLAW_CONFIG.read_text())
        return cfg["channels"]["telegram"]["accounts"]["cad"]["botToken"]
    except Exception:
        return None


def _notify(text: str, photo: str | None = None):
    """Best-effort Telegram ping to the owner, off-thread — must never block or fail a job."""
    token = _bot_token()
    if not token:
        return

    def _send():
        try:
            if photo and Path(photo).is_file():
                boundary = uuid.uuid4().hex
                body = b""
                for k, v in (("chat_id", OWNER_CHAT_ID), ("caption", text[:1000])):
                    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                             f"name=\"{k}\"\r\n\r\n{v}\r\n").encode()
                body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; "
                         f"filename=\"build.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
                body += Path(photo).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendPhoto", data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            else:
                data = urllib.parse.urlencode(
                    {"chat_id": OWNER_CHAT_ID, "text": text[:4000]}).encode()
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendMessage", data=data)
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def _visitor(request: Request) -> tuple[str, bool]:
    """(display label, is_guest) for the request's tailnet identity."""
    login = request.headers.get("tailscale-user-login", "")
    name  = request.headers.get("tailscale-user-name", "")
    if not login:
        return "local", False
    label = f"{name} ({login})" if name and name != login else login
    return label, login not in OWNER_LOGINS


def _ping_outcome(job: dict):
    r = job.get("result") or {}
    status = {"done": "✅ finished", "error": "❌ failed",
              "needs_clarification": "❓ needs answers"}.get(job["status"], job["status"])
    line = (f"CAD web: {status} — {job.get('user', '?')}: "
            f"“{(job.get('spec') or 'image-only build')[:120]}”")
    if job.get("error"):
        line += f"\n{str(job['error'])[:200]}"
    photo = None
    bd = r.get("build_dir_fs")
    if bd and (Path(bd) / "build.png").is_file():
        photo = str(Path(bd) / "build.png")
    _notify(line, photo)

_jobs: dict[str, dict] = {}             # job_id -> job (in-memory, v1)
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
# Mesh jobs run remotely (no GPU, no build lock) — their own queue so a mesh generation
# never waits behind a CAD build or vice versa.
_mesh_queue: "queue.Queue[str]" = queue.Queue()
# Set while a job holds the box. The title worker waits on this: naming a creation swaps the
# model in VRAM (OLLAMA_MAX_LOADED_MODELS=1) and must never happen during a build.
_gpu_busy = threading.Event()
_title_ping = threading.Event()

_WAIT_RE = re.compile(r"waiting for build lock")

# ── History persistence ────────────────────────────────────────────────────────
# The page's creation rail is only useful if it outlives a restart, so the job store is
# mirrored to one JSON file. The live `log` deque is progress, not history — it is not
# persisted. Writes are atomic (tmp + os.replace) so a crash mid-write cannot truncate the
# file into unparseable JSON.
_PERSIST_SKIP = {"log", "pending_chat", "guest"}


def _persist():
    try:
        with _jobs_lock:
            jobs = sorted(_jobs.values(), key=lambda j: j["created_at"])[-MAX_SESSIONS:]
            rows = [{k: v for k, v in j.items() if k not in _PERSIST_SKIP} for j in jobs]
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SESSIONS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows))
        os.replace(tmp, SESSIONS_FILE)
    except Exception as e:                # history is a convenience; never fail a build over it
        print(f"[history] not saved — {type(e).__name__}: {e}", flush=True)


def _load():
    """Rehydrate the store at import, BEFORE the workers start."""
    if not SESSIONS_FILE.is_file():
        return
    try:
        rows = json.loads(SESSIONS_FILE.read_text())
    except Exception as e:
        print(f"[history] {SESSIONS_FILE} unreadable, starting empty — {e}", flush=True)
        return
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        row["log"] = deque(maxlen=300)
        # The page indexes these directly; a truncated or hand-edited file must not 500 it.
        for k, default in (("spec", ""), ("coder", "auto"), ("image", None),
                           ("status", "error"), ("created_at", 0.0)):
            row.setdefault(k, default)
        # A queued/running job's subprocess died with the service — it cannot be resumed,
        # and leaving it "running" would spin the browser's poller forever.
        if row.get("status") in ("queued", "running", "waiting_gpu"):
            row["status"] = "error"
            row["error"] = "interrupted by a server restart"
        _jobs[row["id"]] = row


def _fallback_title(job: dict) -> str:
    spec = (job.get("spec") or "").strip()
    if not spec:
        return "Image-only build"
    return spec[:45].rstrip() + ("…" if len(spec) > 45 else "")


def _title_for(job: dict):
    """Name the creation with the small local model — CADAM-style rail titles.

    Only ever called from the title worker while the GPU is idle: OLLAMA_MAX_LOADED_MODELS=1,
    so this call evicts the coder from VRAM and a cold qwen3:8b load costs ~30s. Running it
    inline after a build would push that cost onto the next queued build. Any failure leaves
    the truncated-spec fallback in place.
    """
    job.setdefault("title", _fallback_title(job))
    if job.get("titled") or not (job.get("spec") or "").strip():
        return
    prompt = ("Name this 3D CAD creation with a short title of 2 to 5 words in Title Case. "
              "Reply with the title only — no quotes, no punctuation, no explanation.\n\n"
              f"Description: {job['spec'][:400]}")
    data = json.dumps({"model": TITLE_MODEL, "prompt": prompt, "stream": False,
                       "think": False, "options": {"temperature": 0.2}}).encode()
    raw = ""
    for attempt in range(3):
        try:
            req = urllib.request.Request(OLLAMA_GEN, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read())
        except Exception as e:
            return _title_gave_up(job, f"{type(e).__name__}: {e}")
        # Ollama answers a request that arrives mid-model-swap with this instead of blocking.
        if payload.get("error"):
            if "loading model" in str(payload["error"]) and attempt < 2:
                time.sleep(10)
                continue
            return _title_gave_up(job, str(payload["error"]))
        raw = payload.get("response", "")
        break
    # qwen3 can still emit a think block even with think:false on older builds — strip it.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    title = raw.strip().strip('"\'' + "“”").splitlines()[0].strip() if raw.strip() else ""
    if not title or len(title) > 60 or title.lower().startswith(("i ", "here", "sure")):
        return _title_gave_up(job, f"unusable response {raw[:80]!r}")
    job["title"], job["titled"] = title[:48], True


def _title_gave_up(job: dict, why: str):
    """Keep the truncated-spec fallback, but say why in the journal — a title that silently
    never arrives is indistinguishable from one that is merely slow."""
    print(f"[title] {job['id']}: keeping fallback — {why}", flush=True)


def _title_worker():
    """Names finished creations, but only while nothing is building — the title call swaps
    the model in VRAM, so it must never contend with a build. Re-checks every 2 minutes,
    which also covers titles skipped because the box was busy."""
    while True:
        _title_ping.wait(timeout=120)
        _title_ping.clear()
        with _jobs_lock:
            pending = [j for j in _jobs.values()
                       if not j.get("titled")
                       and j["status"] in ("done", "error", "needs_clarification")
                       and (j.get("spec") or "").strip()]
        done_any = False
        for job in pending:
            if _gpu_busy.is_set() or not (_queue.empty() and _mesh_queue.empty()):
                break                            # a build wants the GPU — try again later
            _title_for(job)
            done_any = True
        if done_any:
            _persist()


def _turns_public(job: dict) -> list:
    """Past versions of this creation, newest snapshot last. Each revise overwrites the
    build dir in place, so the previous version is copied to turn<N>.* before it goes."""
    bid = ((job.get("result") or {}).get("build_id")
           or Path((job.get("result") or {}).get("build_dir_fs", "")).name)
    out = []
    for t in job.get("turns", []):
        out.append({"n": t["n"], "ts": t.get("ts"), "note": t.get("note", ""),
                    "artifacts": {k: f"/artifacts/{bid}/{v}"
                                  for k, v in (t.get("files") or {}).items()}})
    return out


def _job_public(job: dict) -> dict:
    return {
        "id": job["id"], "spec": job["spec"], "coder": job["coder"],
        "kind": job.get("kind", "cad"),
        "title": job.get("title") or _fallback_title(job),
        "has_image": bool(job["image"]), "status": job["status"],
        "user": job.get("user", "local"),
        "queued_ahead": job.get("queued_ahead", 0),
        "log_tail": list(job["log"])[-LOG_TAIL:],
        "result": job.get("result"), "error": job.get("error"),
        "chat": job.get("chat", []),
        "turns": _turns_public(job),
        "created_at": job["created_at"], "updated_at": job.get("updated_at", job["created_at"]),
    }


def _job_light(job: dict) -> dict:
    """Sidebar row. Deliberately excludes `result` — the rail must stay cheap no matter how
    many creations exist; the full payload is fetched only for the open thread."""
    return {
        "id": job["id"], "title": job.get("title") or _fallback_title(job),
        "spec": job["spec"], "status": job["status"], "kind": job.get("kind", "cad"),
        "has_image": bool(job["image"]), "turns": len(job.get("turns", [])),
        "user": job.get("user", "local"),
        "created_at": job["created_at"], "updated_at": job.get("updated_at", job["created_at"]),
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
        _gpu_busy.set()
        try:
            _run_build(job)
        except Exception as e:
            job["status"], job["error"] = "error", f"internal error: {e}"
        finally:
            _gpu_busy.clear()
            job["updated_at"] = time.time()
            _persist()
            _title_ping.set()            # name it once the box is free
            if job.get("guest"):
                _ping_outcome(job)
            _queue.task_done()


def _snapshot_turn(job: dict, prev_dir: str, note: str):
    """A revise rewrites the build dir in place, so copy the current version aside first."""
    d = Path(prev_dir)
    n = len(job.setdefault("turns", [])) + 1
    files = {}
    for key, fname in TURN_FILES:
        src = d / fname
        if not src.is_file():
            continue
        dst = d / f"turn{n}{src.suffix}"
        try:
            shutil.copy2(src, dst)
            files[key] = dst.name
        except Exception:
            pass
    if files:
        job["turns"].append({"n": n, "ts": time.time(), "note": note[:200], "files": files})


def _run_build(job: dict):
    job["status"] = "running"
    job["updated_at"] = time.time()
    chat_msg = job.pop("pending_chat", None)
    prev_dir = (job.get("result") or {}).get("build_dir_fs", "")
    if chat_msg and prev_dir:
        # Conversational revise turn on the existing build — the user is the gate.
        # The version about to be overwritten was produced by the PREVIOUS message
        # (chat[-1] is the one we are about to apply), or by the original spec.
        prior = [m["text"] for m in job.get("chat", []) if m.get("role") == "user"][:-1]
        _snapshot_turn(job, prev_dir, prior[-1] if prior else (job.get("spec") or "original"))
        if job.get("lang") == "openscad":
            cmd = [ENGINE_PYTHON, str(SKILL_ROOT / "scripts" / "openscad_gen.py"),
                   "--revise-dir", prev_dir, "--feedback", chat_msg, "--json"]
        else:
            cmd = [ENGINE_PYTHON, str(SKILL_ROOT / "scripts" / "fluid_gen.py"),
                   "revise", prev_dir, chat_msg, "--json"]
    elif job.get("lang") == "openscad":
        cmd = [ENGINE_PYTHON, str(SKILL_ROOT / "scripts" / "openscad_gen.py"),
               job["spec"], "--json"]
    elif job.get("engine_mode") == "loop":
        cmd = [ENGINE_PYTHON, "-m", "cad_v5", job["spec"], "--once", "--json", "--ask",
               "--coder", job["coder"], "--target", "file"]
    else:                                     # fluid: fast single turn, no gate vetoes
        cmd = [ENGINE_PYTHON, str(SKILL_ROOT / "scripts" / "fluid_gen.py"),
               "build", job["spec"], "--coder",
               job["coder"] if job["coder"] in ("fast", "strong", "cloud") else "fast",
               "--json"]
    if job["image"]:
        cmd += ["--image", job["image"]]
    if not job.get("fewshots", True):
        cmd += ["--no-fewshots"]
    env = {**os.environ, "CAD_FRONTEND": "web"}
    if job.get("candidates"):
        env["CAD_CANDIDATES"] = job["candidates"]      # best-of-N first-turn sampling
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
            "last_critique", "warning", "image_analysis", "image_only", "lang", "params",
            "needs_clarification", "questions", "spec", "error",
            # gates-on fluid mode (2026-08-11): severity-preserved findings + expansion rung
            "gate_hard", "gate_spec", "gate_adv", "gate_repaired", "salvaged",
            "expanded_spec", "assumptions")}
    build_dir = result.get("build_dir") or ""
    out["instruments"] = result.get("instruments")
    out["mode"] = result.get("mode")
    if build_dir:
        out["build_dir_fs"] = build_dir          # for chat revise turns
        bid = Path(build_dir).name
        arts = {}
        for key, fname in (("render", "build.png"), ("step", "build.step"),
                           ("stl", "build.stl"), ("dxf", "build.dxf"),
                           ("scad", "build.scad"),
                           ("reference", "reference.jpg"), ("source", "build_source.py")):
            if (BUILDS_DIR / bid / fname).exists():
                arts[key] = f"/artifacts/{bid}/{fname}"
        out["artifacts"], out["build_id"] = arts, bid
    return {k: v for k, v in out.items() if v not in (None, "")}


@app.post("/api/build")
async def api_build(request: Request, spec: str = Form(""), coder: str = Form("auto"),
                    candidates: str = Form(""), fewshots: str = Form("on"),
                    lang: str = Form("build123d"), engine_mode: str = Form("fluid"),
                    image: UploadFile | None = File(None)):
    spec = spec.strip()
    if candidates and candidates not in {"1", "3", "5"}:
        raise HTTPException(400, "candidates must be 1, 3 or 5")
    if lang not in {"build123d", "openscad"}:
        raise HTTPException(400, "lang must be build123d or openscad")
    if lang == "openscad" and image is not None and image.filename:
        raise HTTPException(400, "openscad backend is text-only for now")
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
    user, guest = _visitor(request)
    job = {
        "id": uuid.uuid4().hex[:12], "spec": spec, "coder": coder, "image": image_path,
        "candidates": candidates, "fewshots": fewshots != "off", "lang": lang,
        "engine_mode": engine_mode if engine_mode in ("fluid", "loop") else "fluid",
        "user": user, "guest": guest,
        "status": "queued", "log": deque(maxlen=300), "result": None, "error": None,
        "created_at": time.time(), "updated_at": time.time(),
    }
    job["title"] = _fallback_title(job)       # real title lands when the build finishes
    with _jobs_lock:
        job["queued_ahead"] = sum(1 for j in _jobs.values()
                                  if j["status"] in ("queued", "running", "waiting_gpu"))
        _jobs[job["id"]] = job
    _persist()
    _queue.put(job["id"])
    if guest:
        _notify(f"CAD web: 🛠 {user} queued a build — "
                f"“{(spec or 'image-only')[:150]}” ({job['queued_ahead']} ahead)")
    return {"job_id": job["id"], "queued_ahead": job["queued_ahead"]}


@app.post("/api/chat")
def api_chat(payload: dict, request: Request):
    """Fluid conversation: the user's words become a revise turn on the job's build."""
    job_id, msg = str(payload.get("job_id", "")), str(payload.get("message", "")).strip()
    if not msg:
        raise HTTPException(400, "empty message")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (jobs do not survive a restart)")
    if job.get("kind") == "mesh":
        raise HTTPException(400, "chat revise isn't wired for mesh jobs yet")
    if job["status"] not in ("done", "error") or not (job.get("result") or {}).get("build_dir_fs"):
        raise HTTPException(409, "job has no completed build to revise yet")
    user, guest = _visitor(request)
    job["user"], job["guest"] = user, guest        # attribute the revise turn to its author
    job.setdefault("chat", []).append({"role": "user", "text": msg, "ts": time.time()})
    job["pending_chat"] = msg
    job["status"] = "queued"
    job["updated_at"] = time.time()
    _persist()
    _queue.put(job_id)
    if guest:
        _notify(f"CAD web: 💬 {user} revising "
                f"“{(job.get('spec') or 'image-only build')[:80]}”: {msg[:200]}")
    return {"ok": True}


@app.post("/api/rescale")
def api_rescale(payload: dict):
    """The CADAM slider loop: substitute parameter values in a build's .scad and recompile
    locally. Zero LLM — pure text substitution + OpenSCAD. Runs in FastAPI's threadpool."""
    build_id = str(payload.get("build_id", ""))
    params = payload.get("params") or {}
    if not re.fullmatch(r"scad_[0-9_]+", build_id):
        raise HTTPException(400, "not an openscad build")
    build_dir = BUILDS_DIR / build_id
    scad = build_dir / "build.scad"
    if not scad.is_file():
        raise HTTPException(404, "build not found")
    code = scad.read_text()
    for name, val in params.items():
        if not re.fullmatch(r"\$?\w+", str(name)):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        vs = str(int(v)) if v == int(v) else f"{v:g}"
        code = re.sub(rf"^(\s*{re.escape(str(name))}\s*=\s*)-?[0-9.]+(\s*;)",
                      rf"\g<1>{vs}\g<2>", code, count=1, flags=re.M)
    scad.write_text(code)
    import importlib.util as ilu
    spec_ = ilu.spec_from_file_location("og", SKILL_ROOT / "scripts" / "openscad_gen.py")
    og = ilu.module_from_spec(spec_)
    spec_.loader.exec_module(og)
    ok, err = og.compile_scad(scad, build_dir / "build.stl")
    if not ok:
        raise HTTPException(422, f"recompile failed: {err[:300]}")
    return {"ok": True, "facts": og.stl_facts(build_dir / "build.stl"),
            "stl": f"/artifacts/{build_id}/build.stl",
            "params": og.parse_params(code)}


@app.post("/api/mesh")
async def api_mesh(request: Request, spec: str = Form(""),
                   image: UploadFile | None = File(None)):
    """CADAM-style Mesh mode: image -> organic mesh, LOCAL TripoSR by default.
    Meshy stays as a dormant cloud fallback (MESH_PROVIDER=meshy + key)."""
    spec = spec.strip()
    has_image = image is not None and image.filename
    if meshmod.provider() == "triposr-local" and not has_image:
        raise HTTPException(400, "Local mesh generation works from an image — attach a "
                                 "photo or sketch (text-to-3D needs an image model first; "
                                 "coming later).")
    if meshmod.provider() == "meshy" and not meshmod.api_key():
        raise HTTPException(503, "MESH_PROVIDER=meshy is set but no MESHY_API_KEY found.")
    if not spec and not has_image:
        raise HTTPException(400, "provide a prompt, an image, or both")
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
    user, guest = _visitor(request)
    job = {
        "id": uuid.uuid4().hex[:12], "spec": spec, "coder": "-", "image": image_path,
        "user": user, "guest": guest,
        "kind": "mesh", "status": "queued", "log": deque(maxlen=300),
        "result": None, "error": None,
        "created_at": time.time(), "updated_at": time.time(),
    }
    job["title"] = _fallback_title(job)
    with _jobs_lock:
        _jobs[job["id"]] = job
    _persist()
    _mesh_queue.put(job["id"])
    if guest:
        _notify(f"CAD web: 🛠 {user} queued a mesh — “{(spec or 'image-only')[:150]}”")
    return {"job_id": job["id"]}


@app.get("/meshes/{job_id}/{filename}")
def meshes(job_id: str, filename: str):
    target = (meshmod.MESHES_DIR / job_id / filename).resolve()
    if not str(target).startswith(str(meshmod.MESHES_DIR.resolve()) + os.sep):
        raise HTTPException(404, "not found")
    if target.suffix.lower() not in {".glb", ".obj", ".png"}:
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found")
    media = {".png": "image/png", ".glb": "model/gltf-binary"} \
        .get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(target, media_type=media, filename=filename)


def _mesh_worker():
    while True:
        job_id = _mesh_queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            continue
        _gpu_busy.set()
        try:
            meshmod.run_mesh_job(job)
        except Exception as e:
            job["status"], job["error"] = "error", f"internal error: {e}"
        finally:
            _gpu_busy.clear()
            job["updated_at"] = time.time()
            _persist()
            _title_ping.set()
            if job.get("guest"):
                _ping_outcome(job)
            _mesh_queue.task_done()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (jobs do not survive a server restart)")
    return _job_public(job)


@app.get("/api/jobs")
def api_jobs():
    """The creation rail: light rows only, newest activity first."""
    with _jobs_lock:
        jobs = sorted(_jobs.values(),
                      key=lambda j: j.get("updated_at", j["created_at"]), reverse=True)
        return [_job_light(j) for j in jobs]


@app.delete("/api/jobs/{job_id}")
def api_delete_job(job_id: str):
    """Remove a creation from the rail. Artifacts stay on disk for the engine to prune."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job["status"] in ("queued", "running", "waiting_gpu"):
            raise HTTPException(409, "can't delete a creation while it is building")
        _jobs.pop(job_id, None)
    _persist()
    return {"ok": True}


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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "queue": _queue.qsize()})


_load()                                  # history first, then the workers that mutate it
threading.Thread(target=_worker, daemon=True).start()
threading.Thread(target=_mesh_worker, daemon=True).start()
threading.Thread(target=_title_worker, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)

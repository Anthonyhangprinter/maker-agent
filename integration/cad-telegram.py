#!/usr/bin/env python3
"""
cad-telegram.py — Satine Telegram bot frontend for the CAD agent (v5, cad_engine.py).

Commands:
  /build <spec>  — build the spec in Onshape (or just send the spec as plain text)
  /plan <spec>   — dry-run planner, no build
  /rate <n> [comment]  — rate the last build 1-5 and save to feedback
  /done          — close current refinement session without rating
  /help          — show usage

Plain text after a build = refinement feedback (e.g. "make it 200mm longer").
Start a brand-new build by beginning with a verb: "make", "build", "create", etc.

Coder selection: the agent auto-picks the fast 7B or strong 30B coder per spec and escalates
on failure. Force one by prefixing the message: "strong: <spec>" / "fast: <spec>".

Architecture: the poll thread only ACKs Telegram and answers instant commands; every
LLM-touching request (build/refine/plan/inspect) is journaled to disk, then queued to a
single worker thread. Journaling before the offset ack means a crash or systemd restart
mid-build replays the request instead of silently dropping it; the single worker keeps
one build at a time on the GPU while other users still get instant replies.

M7 (N2/N3): builds are run with --ask — if the agent's ambiguity gate finds the spec is
missing a critical dimension or basic form, it asks 2-3 short questions instead of guessing;
the next plain message from that chat answers them and the build proceeds. A refine's result
may carry an `intent_delta` (the brief contract's field-level diff, e.g. "wall: 2 -> 3") which
is shown prefixed with "Δ " before the usual follow-up prompt.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ONSHAPE_URL_RE = re.compile(
    r"https://cad\.onshape\.com/documents/[a-f0-9]+/w/[a-f0-9]+/e/[a-f0-9]+",
    re.IGNORECASE,
)

CONFIG_FILE   = os.path.expanduser("~/.openclaw/openclaw.json")
OFFSET_FILE   = os.path.expanduser("~/.openclaw/telegram/update-offset-cad.json")
SESSIONS_FILE = os.path.expanduser("~/.openclaw/telegram/cad-sessions.json")
JOURNAL_FILE  = os.path.expanduser("~/.openclaw/telegram/cad-jobs-journal.json")
PHOTOS_DIR    = os.path.expanduser("~/.openclaw/telegram/cad-photos")
CAD_AGENT     = os.path.expanduser("~/.openclaw/skills/cad-builder/cad_engine.py")
POLL_TIMEOUT  = 30
MAX_MSG_LEN   = 4000
# Must exceed the agent's own wall-clock budget (cad_engine BUILD_TIMEOUT=1800)
# so the bot never kills a build that is still legitimately iterating.
BUILD_TIMEOUT = 1860
# Spec-revision LLM call budget: must survive a cold model swap on a busy single-GPU box
# (the old 60s silently degraded refines to a raw feedback restatement).
REFINE_TIMEOUT = 300
SESSION_TTL   = 1800   # seconds — session expires after 30 min of inactivity

# Words that signal the user is starting a NEW build, not refining the last one
NEW_BUILD_TRIGGERS = {
    "make", "build", "create", "design", "model", "generate",
    "draw", "fabricate", "produce", "construct",
}


# ── Config ─────────────────────────────────────────────────────────────────────

def load_credentials():
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    account = cfg["channels"]["telegram"]["accounts"]["cad"]
    token   = account["botToken"]
    allowed = set(str(u) for u in account.get("allowFrom", []))
    return token, allowed


# ── Telegram API ───────────────────────────────────────────────────────────────

def tg_request(token, method, params=None):
    url  = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params or {}).encode() if params else None
    req  = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=POLL_TIMEOUT + 5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[telegram] HTTP {e.code}: {e.read().decode()}", flush=True)
        return None
    except Exception as e:
        print(f"[telegram] Request failed: {e}", flush=True)
        return None


def send(token, chat_id, text):
    for chunk in [text[i:i + MAX_MSG_LEN] for i in range(0, len(text), MAX_MSG_LEN)]:
        tg_request(token, "sendMessage", {"chat_id": chat_id, "text": chunk})


def get_updates(token, offset):
    result = tg_request(token, "getUpdates", {
        "offset":          offset,
        "timeout":         POLL_TIMEOUT,
        "allowed_updates": '["message"]',
    })
    if result and result.get("ok"):
        return result["result"]
    return []


def tg_upload(token, method, chat_id, field, path, caption=""):
    """Send a local file (sendPhoto/sendDocument) via multipart — the model files should
    arrive IN the chat, not as a path the user has to go digging for."""
    import mimetypes
    import uuid
    boundary = uuid.uuid4().hex
    fname = os.path.basename(path)
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    parts = []
    for k, v in (("chat_id", str(chat_id)), ("caption", caption)):
        if v:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f"name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    with open(path, "rb") as f:
        blob = f.read()
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                 f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
                 + blob + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[telegram] upload failed ({method} {fname}): {e}", flush=True)
        return None


def send_build_files(token, chat_id, result: dict):
    """Deliver the build's artifacts directly in the chat: render photo + STEP/STL documents."""
    render = result.get("render_local") or ""
    if not (render and os.path.exists(render)):
        # Uploaded (Onshape-mode) builds skip the local PNG; render one best-effort so the
        # user always SEES the model in the chat, not just a link.
        step = result.get("step_local") or ""
        bdir = result.get("build_dir") or os.path.dirname(step)
        if step and os.path.exists(step) and bdir:
            render = os.path.join(bdir, "build.png")
            try:
                subprocess.run([sys.executable,
                                os.path.expanduser("~/.openclaw/skills/cad-builder/scripts/render"),
                                step, render], capture_output=True, timeout=120)
            except Exception as e:
                print(f"[telegram] render fallback failed: {e}", flush=True)
    if render and os.path.exists(render):
        tg_upload(token, "sendPhoto", chat_id, "photo", render,
                  caption=result.get("spec", "")[:200])
    for key, label in (("step_local", "STEP (CAD)"), ("stl_local", "STL (print-ready)")):
        p = result.get(key) or ""
        if p and os.path.exists(p):
            tg_upload(token, "sendDocument", chat_id, "document", p, caption=label)


# ── Offset ─────────────────────────────────────────────────────────────────────

def load_offset():
    try:
        return json.loads(open(OFFSET_FILE).read()).get("lastUpdateId", 0) + 1
    except Exception:
        return 0


def save_offset(update_id):
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, "w") as f:
        json.dump({"version": 1, "lastUpdateId": update_id}, f)


# ── Session storage ─────────────────────────────────────────────────────────────
# sessions: {chat_id_str: {original_spec, history: [{spec, url}], timestamp, photo?}}
# Shared between the poll thread (routing/photo stash) and the worker thread (handlers),
# so every access goes through _sessions_lock.

_sessions_lock = threading.RLock()


def load_sessions() -> dict:
    try:
        return json.loads(open(SESSIONS_FILE).read())
    except Exception:
        return {}


def save_sessions(sessions: dict):
    # Prune expired sessions on every save — they were previously only removed by an
    # explicit /rate or /done, so the file grew without bound.
    with _sessions_lock:
        now = time.time()
        for key in [k for k, s in sessions.items()
                    if now - s.get("timestamp", 0) >= SESSION_TTL]:
            sessions.pop(key, None)
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, "w") as f:
            json.dump(sessions, f, indent=2)


def get_session(sessions: dict, chat_id) -> dict | None:
    with _sessions_lock:
        s = sessions.get(str(chat_id))
        if s and (time.time() - s.get("timestamp", 0)) < SESSION_TTL:
            return s
    return None


def set_session(sessions: dict, chat_id, original_spec: str, history: list):
    with _sessions_lock:
        prev = sessions.get(str(chat_id)) or {}
        sessions[str(chat_id)] = {
            "original_spec": original_spec,
            "history":       history,
            "timestamp":     time.time(),
            **({"photo": prev["photo"]} if prev.get("photo") else {}),
        }
        save_sessions(sessions)


def clear_session(sessions: dict, chat_id):
    with _sessions_lock:
        sessions.pop(str(chat_id), None)
        save_sessions(sessions)


def stash_photo(sessions: dict, chat_id, path: str):
    """Remember the latest reference photo on the (possibly empty) session."""
    with _sessions_lock:
        s = sessions.setdefault(str(chat_id), {"original_spec": "", "history": []})
        s["photo"] = path
        s["timestamp"] = time.time()
        save_sessions(sessions)


# ── N2 pending clarification (in-memory — small and short-lived, no journal needed) ────────────
# {chat_id_str: spec} — a spec the ambiguity gate flagged as missing a critical dimension or
# basic form. The next plain message from that chat answers it; a /-command or an explicit
# new-build message supersedes and clears it instead.

_pending_lock = threading.RLock()
_pending_clarification: dict[str, str] = {}


def set_pending_clarification(chat_id, spec: str):
    with _pending_lock:
        _pending_clarification[str(chat_id)] = spec


def get_pending_clarification(chat_id):
    with _pending_lock:
        return _pending_clarification.get(str(chat_id))


def clear_pending_clarification(chat_id):
    with _pending_lock:
        _pending_clarification.pop(str(chat_id), None)


# ── Job journal + queue ─────────────────────────────────────────────────────────
# Every LLM-touching request is journaled to disk BEFORE the Telegram offset is
# advanced, then queued to the single worker. A crash / systemd restart mid-build
# re-enqueues the journal on startup instead of silently dropping the request
# (previously the offset was acked first, so a restart lost the build forever).

_jobs: "queue.Queue[dict]" = queue.Queue()
_journal_lock = threading.Lock()
_current_job: dict | None = None   # job the worker is executing right now (None = idle)


def _journal_read() -> list:
    try:
        return json.loads(open(JOURNAL_FILE).read()).get("jobs", [])
    except Exception:
        return []


def _journal_write(jobs: list):
    os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
    tmp = JOURNAL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": 1, "jobs": jobs}, f)
    os.replace(tmp, JOURNAL_FILE)


def journal_add(job: dict):
    with _journal_lock:
        _journal_write(_journal_read() + [job])


def journal_remove(job_id):
    with _journal_lock:
        _journal_write([j for j in _journal_read() if j.get("id") != job_id])


def enqueue(token, job: dict, journaled: bool = False):
    if not journaled:
        journal_add(job)
    waiting = _jobs.qsize() + (1 if _current_job is not None else 0)
    _jobs.put(job)
    if waiting:
        send(token, job["chat_id"],
             f"Queued — {waiting} request(s) ahead of yours. I'll start on it as soon "
             f"as the current build finishes.")


# ── Subprocess runner ──────────────────────────────────────────────────────────

def extract_coder(text: str):
    """Pull an optional leading coder directive off a message so the user can force a model
    from Telegram. Requires an explicit marker so normal specs aren't misread:
    "--strong <spec>" or "strong: <spec>" (likewise fast/auto). The agent decides automatically
    by default. Returns (coder, remaining_text) with coder in auto|fast|strong."""
    m = re.match(r"^\s*(?:--(strong|mid|fast|auto)\s+|(strong|mid|fast|auto):\s*)(.*)$",
                 text, re.IGNORECASE | re.DOTALL)
    if m:
        return (m.group(1) or m.group(2)).lower(), m.group(3).strip()
    return "auto", text.strip()


def run_agent(args, timeout=60):
    cmd = [sys.executable, CAD_AGENT] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "Build timed out.", 1
    except Exception as e:
        return "", str(e), 1


def run_v5_build(spec, coder, timeout=BUILD_TIMEOUT):
    """Build via the v5 --json entry: stdout is exactly one JSON line (B7 contract) —
    no more scraping the human output for the first '{' and hoping.

    --ask (M7/N2): the agent pre-checks the spec for a missing critical dimension or basic
    form; if it can't build a competent first draft, it replies with one JSON line
    {needs_clarification: true, questions: [...], spec} INSTEAD of building. The caller
    (handle_build/handle_refine below) surfaces those questions and remembers the spec so
    the user's next plain message can answer them."""
    cmd = [sys.executable, "-m", "cad_v5", spec, "--once", "--json", "--ask",
           "--coder", coder, "--target", "onshape"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=os.path.expanduser("~/.openclaw/skills/cad-builder"))
    except subprocess.TimeoutExpired:
        return None, "Build timed out."
    except Exception as e:
        return None, str(e)
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), ""
            except Exception:
                break
    return None, (proc.stderr or proc.stdout or "no output").strip()[-1500:]


# ── Build result parser ────────────────────────────────────────────────────────

def parse_build_result(stdout: str) -> dict | None:
    """Extract the JSON result dict from agent stdout."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    # stdout may be the whole JSON block
    try:
        return json.loads(stdout.strip())
    except Exception:
        return None


# ── Command handlers ───────────────────────────────────────────────────────────

def send_clarification(token, chat_id, spec: str, result: dict) -> None:
    """N2: surface the ambiguity gate's questions and remember the spec so the user's next
    plain message answers them instead of being read as an unrelated new build."""
    questions = result.get("questions") or []
    qtext = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    send(token, chat_id,
         f"Before I build — a couple of quick questions:\n{qtext}\n\n"
         f"Reply with your answers (or send anything else and I'll build with defaults).")
    set_pending_clarification(chat_id, result.get("spec") or spec)


def handle_build(token, chat_id, spec, sessions: dict):
    coder, spec = extract_coder(spec)
    if not spec.strip():
        send(token, chat_id, "Usage: /build <spec>  e.g. /build spur gear 20T")
        return
    note = "" if coder == "auto" else f"  (forcing {coder} coder)"
    send(token, chat_id, f"Looking at: {spec}{note}\nThis may take a few minutes (it iterates and self-checks)…")
    result, err = run_v5_build(spec, coder)
    if not result:
        send(token, chat_id, f"Build failed:\n{err[-MAX_MSG_LEN:]}")
        return
    if result.get("needs_clarification"):
        send_clarification(token, chat_id, spec, result)
        return

    url = result.get("url", "")
    send_build_files(token, chat_id, result)   # render photo + STEP/STL in the chat
    if result.get("target_error"):
        send(token, chat_id, f"(Onshape upload failed: {result['target_error'][:300]} — "
                             f"the files above are still your model.)")
    history = [{"spec": spec, "url": url}]
    set_session(sessions, chat_id, original_spec=spec, history=history)
    send(token, chat_id,
         "Reply with what to change (e.g. \"make it 200mm longer\") "
         "or /rate 1-5 when you're happy.")


def handle_refine(token, chat_id, feedback: str, sessions: dict):
    coder, feedback = extract_coder(feedback)
    session = get_session(sessions, chat_id)
    if not session:
        # Session expired — treat as new build
        handle_build(token, chat_id, feedback, sessions)
        return

    original = session["original_spec"]
    history  = session["history"]

    send(token, chat_id, "Revising…")
    # Get a revised spec from the LLM
    stdout, _, rc = run_agent(
        ["refine", original, feedback, json.dumps(history)],
        timeout=REFINE_TIMEOUT,
    )
    revised = stdout.strip() if rc == 0 and stdout.strip() else feedback
    if revised == original:
        revised = f"{original}, {feedback}"

    note = "" if coder == "auto" else f"  (forcing {coder} coder)"
    send(token, chat_id, f"Looking at revised: {revised}{note}\nThis may take a few minutes (it iterates and self-checks)…")
    result, err = run_v5_build(revised, coder)
    if not result:
        send(token, chat_id, f"Build failed:\n{err[-MAX_MSG_LEN:]}")
        return
    if result.get("needs_clarification"):
        send_clarification(token, chat_id, revised, result)
        return

    url = result.get("url", "")
    send_build_files(token, chat_id, result)   # render photo + STEP/STL in the chat
    history.append({"spec": revised, "url": url})
    set_session(sessions, chat_id, original_spec=original, history=history)
    # N3: the brief-contract patch delta, when the loop patched instead of regenerating.
    delta = result.get("intent_delta") or []
    prefix = ("Δ " + "; ".join(delta) + "\n\n") if delta else ""
    send(token, chat_id,
         prefix + "Reply with more changes, /rate 1-5 to save this, or /done to close.")


def handle_plan(token, chat_id, spec):
    if not spec.strip():
        send(token, chat_id, "Usage: /plan <spec>  e.g. /plan W200x100 I-beam 2000mm")
        return
    send(token, chat_id, f"Planning: {spec}…")
    stdout, stderr, rc = run_agent(["plan", spec], timeout=60)
    output = (stdout or stderr or "No output.").strip()
    send(token, chat_id, output[-MAX_MSG_LEN:])


def handle_rate(token, chat_id, rest: str, sessions: dict):
    parts = rest.strip().split(None, 1)
    if not parts:
        send(token, chat_id, "Usage: /rate <1-5> [comment]")
        return
    stars   = parts[0]
    comment = parts[1] if len(parts) > 1 else ""

    # Encode refinement history into comment for richer few-shot context
    session = get_session(sessions, chat_id)
    if session and len(session["history"]) > 1:
        chain = " → ".join(h["spec"] for h in session["history"])
        comment = f"{comment} | iterations: {chain}".strip(" |")

    args   = ["rate", stars] + ([comment] if comment else [])
    stdout, stderr, rc = run_agent(args, timeout=15)
    output = (stdout or stderr or "Done.").strip()

    try:
        data = json.loads(output)
        if data.get("ok"):
            stars_str = "⭐" * int(data["rating"])
            msg = f"{stars_str} Saved! This build will improve future {data.get('spec','')[:40]} results."
        else:
            msg = data.get("error", output)
    except Exception:
        msg = output

    send(token, chat_id, msg[-MAX_MSG_LEN:])
    clear_session(sessions, chat_id)


def handle_done(token, chat_id, sessions: dict):
    if get_session(sessions, chat_id):
        clear_session(sessions, chat_id)
        send(token, chat_id, "Session closed. Send a new spec whenever you're ready.")
    else:
        send(token, chat_id, "No active session.")


def handle_inspect(token, chat_id, url: str, extra_text: str, sessions: dict):
    """Inspect an Onshape URL. If extra_text is provided, immediately refine from it."""
    send(token, chat_id, f"Fetching document info…")
    stdout, stderr, rc = run_agent(["inspect", url], timeout=90)
    result = None
    if rc == 0:
        try:
            result = json.loads(stdout.strip())
        except Exception:
            pass

    if not result or not result.get("ok"):
        err = (result or {}).get("error") or (stderr or stdout or "Unknown error").strip()
        send(token, chat_id, f"Couldn't read that document:\n{err[:MAX_MSG_LEN]}")
        return

    description = result["description"]
    send(token, chat_id, f"Here's what I see:\n\n{description}")

    # Seed the session so the user can refine from this doc
    history = [{"spec": description, "url": url}]
    set_session(sessions, chat_id, original_spec=description, history=history)

    if extra_text.strip():
        # User sent a URL + feedback in one message — start refining immediately
        handle_refine(token, chat_id, extra_text.strip(), sessions)
    else:
        send(token, chat_id,
             "Reply with what to change, /rate 1-5 to save it, or /done to close.")


def is_new_build(text: str) -> bool:
    """Return True if the message looks like a fresh build request rather than refinement."""
    first = text.strip().split()[0].lower().rstrip(".,!") if text.strip() else ""
    return first in NEW_BUILD_TRIGGERS


def download_photo(token, msg) -> str | None:
    """Download the largest size of a photo message to PHOTOS_DIR. Returns local path."""
    photos = msg.get("photo") or []
    if not photos:
        return None
    file_id = max(photos, key=lambda p: p.get("file_size", 0)).get("file_id")
    info = tg_request(token, "getFile", {"file_id": file_id})
    if not info or not info.get("ok"):
        return None
    remote = info["result"].get("file_path", "")
    if not remote:
        return None
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    local = os.path.join(PHOTOS_DIR, f"{int(time.time())}-{os.path.basename(remote)}")
    try:
        with urllib.request.urlopen(
                f"https://api.telegram.org/file/bot{token}/{remote}", timeout=60) as r, \
             open(local, "wb") as f:
            f.write(r.read())
        return local
    except Exception as e:
        print(f"[telegram] photo download failed: {e}", flush=True)
        return None


# ── Main loop ──────────────────────────────────────────────────────────────────

HELP_TEXT = (
    "Satine — CAD Agent\n\n"
    "Just describe what you want to build and I'll make it in Onshape.\n\n"
    "After a build, reply with what to change — I'll refine it.\n\n"
    "Paste an Onshape URL and I'll read the document and describe it.\n"
    "You can also paste a URL + feedback in one message to inspect and refine.\n\n"
    "Photos: I save reference images with your session (building FROM a photo is coming); "
    "a caption on the photo is treated as your spec.\n\n"
    "Builds run one at a time — if I'm busy you'll be queued and told your position.\n\n"
    "I pick a coder automatically (fast 7B → 14B → strong 30B, climbing a rung when one "
    "struggles). To force one, prefix your spec: \"fast: <spec>\", \"mid: <spec>\" or "
    "\"strong: <spec>\".\n\n"
    "If a spec is missing something critical (overall size, or basic shape), I'll ask 2-3 short "
    "questions with suggested defaults before building — just reply, or send anything else to "
    "build with the defaults.\n\n"
    "GOOD PROMPTS — the 4 S's: Size (envelope in mm), Specs (counts + diameters, e.g. \"4x M3\"), "
    "Surfaces (which face a feature is on), Symmetry (patterns/spacing). Clearances: push-fit "
    "0.0-0.1mm · slip-fit 0.2mm · loose-fit 0.5-1.0mm. Named hardware (M3, 608ZZ bearing, 2020 "
    "V-slot) is understood as-is.\n"
    "e.g. \"a 120x80x40mm enclosure, 2mm walls, 4x M3 mounting holes in the floor\"\n\n"
    "/build <spec>         start a build\n"
    "/plan <spec>          dry-run planner only\n"
    "/rate <1-5> [comment] rate the last build (saves 4-5★ as future examples)\n"
    "/done                 close current session\n"
    "/help                 show this message"
)


def dispatch(token, job: dict, sessions: dict):
    """Worker thread: route a journaled message. Session-based build-vs-refine routing
    happens HERE (not at poll time) so a refinement queued behind its own build still
    sees the session that build creates."""
    chat_id, text = job["chat_id"], job["text"]
    if job["kind"] == "plan":
        handle_plan(token, chat_id, text)
        return

    # N2: a clarification is pending for this chat — the next plain message answers it, UNLESS
    # it's clearly a fresh build request or an Onshape URL, which supersede it (cleared either
    # way so it can never answer a later, unrelated message).
    if job["kind"] == "message":
        pending_spec = get_pending_clarification(chat_id)
        if pending_spec is not None:
            clear_pending_clarification(chat_id)
            if not is_new_build(text) and not ONSHAPE_URL_RE.search(text):
                handle_build(token, chat_id, f"{pending_spec} — {text}".strip(), sessions)
                return

    url_match = ONSHAPE_URL_RE.search(text)
    if url_match:
        url = url_match.group(0)
        extra = (text[:url_match.start()].strip() + " " + text[url_match.end():].strip()).strip()
        handle_inspect(token, chat_id, url, extra, sessions)
    elif job["kind"] == "build":
        handle_build(token, chat_id, text, sessions)
    elif get_session(sessions, chat_id) and not is_new_build(text):
        handle_refine(token, chat_id, text, sessions)
    else:
        handle_build(token, chat_id, text, sessions)


def worker(token, sessions: dict):
    global _current_job
    while True:
        job = _jobs.get()
        _current_job = job
        try:
            dispatch(token, job, sessions)
        except Exception as e:
            print(f"[cad-telegram] worker error on job {job.get('id')}: {e}", flush=True)
            try:
                send(token, job["chat_id"], f"Something went wrong handling that request: {e}")
            except Exception:
                pass
        finally:
            _current_job = None
            journal_remove(job.get("id"))
            _jobs.task_done()


def main():
    token, allowed = load_credentials()
    sessions = load_sessions()
    offset   = load_offset()

    # Replay any jobs journaled before a crash/restart — these were acked to Telegram
    # but never finished, and previously vanished without a trace.
    pending = _journal_read()
    for job in pending:
        print(f"[cad-telegram] Replaying journaled job {job.get('id')}: "
              f"{job.get('text', '')[:60]}", flush=True)
        _jobs.put(job)
    if pending:
        for chat_id in {j["chat_id"] for j in pending}:
            send(token, chat_id, "I was restarted — resuming your pending request(s) now.")

    threading.Thread(target=worker, args=(token, sessions), daemon=True).start()
    print(f"[cad-telegram] Polling (offset={offset}, journal={len(pending)} replayed)", flush=True)

    while True:
        updates = get_updates(token, offset)
        for upd in updates:
            msg     = upd.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            from_id = str(msg.get("from", {}).get("id", ""))
            text    = (msg.get("text") or "").strip()

            try:
                if not chat_id or from_id not in allowed:
                    continue

                # Photo messages: download + stash on the session so the upcoming
                # image-conditioned flow can use it (previously silently dropped).
                if msg.get("photo"):
                    path = download_photo(token, msg)
                    caption = (msg.get("caption") or "").strip()
                    if path:
                        stash_photo(sessions, chat_id, path)
                        send(token, chat_id,
                             "Got the image — I can't build FROM photos yet, but I've saved "
                             "it with your session. Describe what to build in words"
                             + (f" (I'll treat your caption as the spec)." if caption else "."))
                    else:
                        send(token, chat_id, "Couldn't download that image — please try again.")
                    if not caption:
                        continue
                    text = caption   # a captioned photo proceeds as a normal message

                if not text:
                    continue

                # N2: any explicit command supersedes a pending clarification question.
                if text.startswith("/"):
                    clear_pending_clarification(chat_id)

                # Instant commands stay on the poll thread; LLM-touching work is queued.
                if text in ("/help", "/start"):
                    send(token, chat_id, HELP_TEXT)
                elif text == "/done":
                    handle_done(token, chat_id, sessions)
                elif text == "/rate" or text.startswith("/rate "):
                    handle_rate(token, chat_id, text[5:].strip(), sessions)
                elif text in ("/build", "/plan"):
                    send(token, chat_id, f"Usage: {text} <spec>  e.g. {text} spur gear 20T")
                elif text.startswith("/build "):
                    enqueue(token, {"id": upd["update_id"], "chat_id": chat_id,
                                    "kind": "build", "text": text[7:].strip()})
                elif text.startswith("/plan "):
                    enqueue(token, {"id": upd["update_id"], "chat_id": chat_id,
                                    "kind": "plan", "text": text[6:].strip()})
                elif text.startswith("/"):
                    send(token, chat_id, "Unknown command. Send /help for usage.")
                else:
                    enqueue(token, {"id": upd["update_id"], "chat_id": chat_id,
                                    "kind": "message", "text": text})
            finally:
                # Ack AFTER journaling/handling so a crash replays instead of drops.
                offset = upd["update_id"] + 1
                save_offset(upd["update_id"])


if __name__ == "__main__":
    main()

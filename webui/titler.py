"""Creation-title request shape, kept out of app.py so it is importable without FastAPI.

The title worker names a finished creation with the RESIDENT model through the gpu-proxy on
:8085, not with the CAD coder server directly: the proxy queues a request that lands
mid-eviction instead of erroring, and titling must never be the thing that pulls a model into
VRAM. The old Ollama qwen3:8b call on :11434 was retired 2026-09-19 with the rest of Ollama
(the model itself had been deleted on 2026-09-12, so every title had been silently falling
back to the truncated spec since then).
"""
import re
import sys
from pathlib import Path

TITLE_URL = "http://127.0.0.1:8085/v1/chat/completions"
# max_tokens is deliberately tiny: a runaway answer must not cost a minute of GPU, and a
# 2-to-5-word title fits comfortably.
TITLE_MAX_TOKENS = 24
TITLE_PROMPT = ("Name this 3D CAD creation with a short title of 2 to 5 words in Title Case. "
                "Reply with the title only, no quotes, no punctuation, no explanation.")


def resident_alias() -> str:
    """The resident's served alias, single-sourced in the engine config. This app runs in its
    own venv, so the import is guarded: a missing cad_v5 degrades the title worker, it must
    never stop the server from booting."""
    try:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from cad_v5.config import RESIDENT_ALIAS
        return RESIDENT_ALIAS
    except Exception:
        return "qwen3.8-27b"


TITLE_MODEL = resident_alias()


def build_request(spec: str) -> tuple:
    """(url, body) for one title call: chat-completions against the resident, thinking off.
    reasoning_effort is a TOP-LEVEL field on this server (llama.cpp b10795 maps it into the
    chat template); a thinking preamble for a four-word title is pure latency."""
    return TITLE_URL, {
        "model": TITLE_MODEL,
        "messages": [{"role": "user",
                      "content": TITLE_PROMPT + f"\n\nDescription: {spec[:400]}"}],
        "reasoning_effort": "none",
        "temperature": 0.2,
        "max_tokens": TITLE_MAX_TOKENS,
        "stream": False,
    }


def parse_title(payload: dict) -> str:
    """The usable title in a chat-completions payload, or "" when there isn't one. A
    thinking model can still emit a think block, so strip it before judging the answer."""
    try:
        raw = payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if not raw.strip():
        return ""
    title = raw.strip().strip('"\'' + "“”").splitlines()[0].strip()
    if not title or len(title) > 60 or title.lower().startswith(("i ", "here", "sure")):
        return ""
    return title[:48]

#!/usr/bin/env python3
"""
Thin client for the resident model's OpenAI-compatible front door. ONE request in flight at a
time (the caller is responsible for that — this module makes no attempt to parallelize, on
purpose, since a family chat shares this server).
"""
import base64
import json
import time
import urllib.error
import urllib.request

ENDPOINT = "http://localhost:8085/v1/chat/completions"
MODEL = "qwen3.8-27b"


def image_data_url(png_path) -> str:
    data = open(png_path, "rb").read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_last_json_object(text: str):
    """Scan for balanced {...} objects and return the last one that parses AND has a
    'verdict' key. Returns None if nothing qualifies (== abstain)."""
    best = None
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            j = i
            in_str = False
            esc = False
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and "verdict" in obj:
                                    best = obj
                            except Exception:
                                pass
                            i = j
                            break
                j += 1
        i += 1
    return best


def call_judge(system_prompt: str, user_content, max_tokens=6000, reasoning_effort="medium",
               timeout=280, retries=1):
    """Returns dict: {ok, raw_response, content, reasoning_content, verdict_obj, usage,
    wall_s, error}."""
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
            wall_s = time.time() - t0
            resp = json.loads(raw)
            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            verdict_obj = _extract_last_json_object(content) or _extract_last_json_object(reasoning)
            usage = resp.get("usage", {})
            return {
                "ok": True,
                "content": content,
                "reasoning_content": reasoning,
                "verdict_obj": verdict_obj,
                "usage": usage,
                "wall_s": wall_s,
                "finish_reason": choice.get("finish_reason"),
                "error": None,
            }
        except Exception as e:
            last_err = str(e)
            wall_s = time.time() - t0
            if attempt < retries:
                time.sleep(2)
                continue
    return {
        "ok": False, "content": None, "reasoning_content": None, "verdict_obj": None,
        "usage": {}, "wall_s": wall_s, "finish_reason": None, "error": last_err,
    }

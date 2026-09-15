"""Local llama.cpp adapter (Maker Agent's maker-server, OpenAI-compatible).

Model id: exactly `local`. Unlike the cloud adapters, the local adapter does
not encode the real model in the `--model` slug (there's only ever one arm up
on the maker-server at a time — see `scripts/arms.py`). Instead:

    BENCHCAD_BASE_URL       default http://127.0.0.1:8088/v1 (maker-server)
    BENCHCAD_MODEL          the llama-server model alias (arm["alias"]), default "maker"
    BENCHCAD_TEMPLATE_KWARGS  JSON blob forwarded as chat_template_kwargs in
                              extra_body, e.g. {"enable_thinking": false} or
                              {"reasoning_effort": "low"} — same knob the arm's
                              own `--chat-template-kwargs` launch flag sets, so
                              a thinking/no-thinking arm is honoured here too.

Cloned from `openrouter_adapter.py` (same OpenAI-compatible chat.completions
shape, image_url content parts for Vision2Code) with the OpenRouter
`:reasoning=<spec>` suffix parsing dropped — that's not how this endpoint's
reasoning knob works.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import _img_b64, usage_from_openai


def generate(*, model: str, system: str, user_text: str,
             image_paths: list[Path], max_tokens: int, timeout: int) -> tuple[str, dict]:
    import openai

    base_url = os.environ.get("BENCHCAD_BASE_URL", "http://127.0.0.1:8088/v1")
    real_model = os.environ.get("BENCHCAD_MODEL", "maker")
    template_kwargs = json.loads(os.environ.get("BENCHCAD_TEMPLATE_KWARGS", "{}"))

    user_content: list = [{"type": "text", "text": user_text}]
    for p in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_img_b64(p)}"},
        })

    client = openai.OpenAI(
        api_key="local",
        base_url=base_url,
        timeout=max(timeout, 900),
    )
    resp = client.chat.completions.create(
        model=real_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": template_kwargs} if template_kwargs else None,
    )
    text = resp.choices[0].message.content or ""
    return text, usage_from_openai(resp)

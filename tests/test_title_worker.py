"""Title worker request shape — webui/titler.py.

The engine's own test suite must stay importable under the SYSTEM python (no FastAPI), which
is why the request shape lives in webui/titler.py rather than webui/app.py. Everything here
is offline: no HTTP, no model, no GPU.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "webui"))

import titler


def test_title_request_targets_the_resident_through_the_proxy():
    """Not Ollama's :11434/api/generate (retired 2026-09-19), and not the CAD coder server
    on :8088 either — the gpu-proxy queues a title that lands mid-eviction."""
    url, body = titler.build_request("a 100x60x20mm enclosure with 2mm walls")
    assert url == "http://127.0.0.1:8085/v1/chat/completions"
    assert "11434" not in url
    assert body["model"] == titler.resident_alias() == "qwen3.8-27b"
    assert body["reasoning_effort"] == "none"      # top-level, not chat_template_kwargs
    assert body["max_tokens"] == 24
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"
    assert "enclosure" in body["messages"][0]["content"]
    # chat-completions schema, not Ollama's /api/generate one
    assert "prompt" not in body and "options" not in body and "think" not in body
    json.dumps(body)   # must be serialisable as sent


def test_long_specs_are_truncated_into_the_prompt():
    _, body = titler.build_request("x" * 5000)
    assert len(body["messages"][0]["content"]) < 700


def test_parse_title_reads_a_chat_completion():
    p = {"choices": [{"message": {"content": '  "Vented Motor Bracket"\n'}}]}
    assert titler.parse_title(p) == "Vented Motor Bracket"


def test_parse_title_strips_a_think_block():
    p = {"choices": [{"message": {
        "content": "<think>the user wants a title</think>Hex Spacer Plate"}}]}
    assert titler.parse_title(p) == "Hex Spacer Plate"


def test_parse_title_rejects_chatter_and_malformed_payloads():
    """An empty answer must fall back to the truncated spec, never become the title."""
    assert titler.parse_title({"choices": [{"message": {"content":
                               "Sure, here is a title: Cube"}}]}) == ""
    assert titler.parse_title({"choices": [{"message": {"content": "  "}}]}) == ""
    assert titler.parse_title({"error": "model not found"}) == ""
    assert titler.parse_title({}) == ""

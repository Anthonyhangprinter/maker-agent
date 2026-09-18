import importlib, json, os, sys

import pytest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _reload_with(tmp_path, cad_json: dict):
    p = tmp_path / "cad.json"
    p.write_text(json.dumps(cad_json))
    os.environ["CAD_CONFIG_FILE"] = str(p)
    import cad_v5.config as cfg
    return importlib.reload(cfg)


def test_defaults_point_at_resident(tmp_path):
    cfg = _reload_with(tmp_path, {})
    m = cfg.maker_config()
    assert m == {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server"}
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8086/v1/chat/completions"
    assert cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"


def test_enabled_maker_rewrites_port_and_alias(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "gemma-4-31b"}})
    m = cfg.maker_config()
    assert m["enabled"] and m["port"] == 8088 and m["unit"] == "maker-server"
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8088/v1/chat/completions"
    assert cfg.LOCAL_CODER_HEALTH == "http://127.0.0.1:8088/health"
    assert cfg.CODE_MODEL_STRONG == "local:gemma-4-31b"
    assert cfg.CODE_MODEL_LADDER == ["local:gemma-4-31b"]


def test_ensure_hook_starts_maker_and_resume_restores(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    import cad_engine
    importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)
    cad_engine._ensure_default_server(timeout=1)
    assert ["systemctl", "--user", "stop", "qwen38-server"] in calls
    assert ["systemctl", "--user", "start", "maker-server"] in calls
    calls.clear()
    cad_engine._resume_default_server()
    assert calls == [["systemctl", "--user", "stop", "maker-server"],
                     ["systemctl", "--user", "start", "qwen38-server"]]


def test_keep_maker_reuses_warm_arm_without_systemctl(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    monkeypatch.setenv("CAD_KEEP_MAKER", "1")
    import cad_engine
    importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen",
                         lambda url, timeout=3: _FakeHealthResponse())
    cad_engine._ensure_default_server(timeout=1)
    assert calls == []
    cad_engine._resume_default_server()
    assert calls == []


def test_disabled_maker_ignores_a_stale_alias_and_port(tmp_path):
    """scripts/arms.py restore only flips enabled to false, leaving the last arm's alias and
    port behind. Those must not keep routing strong-rung calls at a model the resident does
    not serve, so a disabled block reads as the plain resident."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": False, "port": 8088, "alias": "gemma-4-31b",
                                             "arm": "gemma-4-31b"}})
    m = cfg.maker_config()
    assert m == {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server"}
    assert cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8086/v1/chat/completions"


def test_keep_maker_falls_through_to_start_when_probe_fails(tmp_path, monkeypatch):
    """The warm fast path is a probe, not a promise: when no arm answers,
    _ensure_default_server must fall through to the normal maker path (stop resident,
    start maker) rather than return as if an arm were up."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    monkeypatch.setenv("CAD_KEEP_MAKER", "1")
    import cad_engine
    importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)

    def dead_probe(url, timeout=3):
        raise OSError("connection refused")

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", dead_probe)
    cad_engine._ensure_default_server(timeout=1)
    assert ["systemctl", "--user", "stop", "qwen38-server"] in calls
    assert ["systemctl", "--user", "start", "maker-server"] in calls
    assert cad_engine._MAKER_STARTED is True


def test_local_usage_accumulates_across_calls(tmp_path, monkeypatch):
    """_LAST_USAGE holds only the final call. A build is many local: calls (triage, expansion,
    codegen, salvage, repair), so a card row's token count has to come from the running total."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)

    bodies = [{"choices": [{"message": {"content": "one"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
              {"choices": [{"message": {"content": "two"}}],
               "usage": {"prompt_tokens": 20, "completion_tokens": 7}}]

    def fake_urlopen(req, timeout=None):
        return _FakeChatResponse(json.dumps(bodies.pop(0)).encode())

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)
    cad_engine.reset_usage()
    cad_engine._ollama("local:qwen3.8-27b", "sys", "a")
    cad_engine._ollama("local:qwen3.8-27b", "sys", "b")
    assert cad_engine._USAGE_TOTAL == {"prompt_tokens": 30, "completion_tokens": 12, "calls": 2}
    assert cad_engine._LAST_USAGE == {"prompt_tokens": 20, "completion_tokens": 7}
    cad_engine.reset_usage()
    assert cad_engine._USAGE_TOTAL == {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def test_brief_model_is_the_local_strong_rung(tmp_path):
    cfg = _reload_with(tmp_path, {})
    assert cfg.BRIEF_MODEL == cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"


def test_preflight_accepts_local_and_cloud_models(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_code_model", lambda: "local:qwen3.8-27b")
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen",
                        lambda url, timeout=0: _FakeHealthResponse())
    cad_engine._preflight_models()   # must not raise
    cad_engine.preflight()           # nor this — no Ollama probe left in it


def test_critic_url_defaults_to_coder_url(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD_CRITIC_URL", raising=False)
    cfg = _reload_with(tmp_path, {})
    assert cfg.CRITIC_URL == cfg.LOCAL_CODER_URL


def test_critic_call_goes_to_critic_url(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_CRITIC_URL", "http://127.0.0.1:8089/v1/chat/completions")
    monkeypatch.setenv("CAD_CRITIC_MODEL", "local:minicpm-v")
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    urls = []

    class R:
        def __init__(s, url): urls.append(url)
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s): return b'{"choices":[{"message":{"content":"ok"}}]}'

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", lambda req, timeout=0: R(req.full_url))
    monkeypatch.setattr(cad_engine, "_ensure_default_server", lambda *a, **k: None)
    cad_engine._ollama("local:minicpm-v", "sys", "user", images=["QUJD"])
    cad_engine._ollama("local:gemma-4-31b", "sys", "user")
    assert urls == ["http://127.0.0.1:8089/v1/chat/completions", cfg.LOCAL_CODER_URL]


def test_preflight_never_touches_the_network_for_a_model_list(tmp_path, monkeypatch):
    """The Ollama /api/tags reachability check is gone (2026-09-19). A dead server must
    leave preflight advisory, not raise: _ensure_default_server() starts the unit."""
    monkeypatch.setenv("CAD_CRITIC_MODEL", "local:gemma-4-31b")
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)

    def boom(*a, **k): raise OSError("connection refused")

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", boom)
    monkeypatch.setattr(cad_engine, "_code_model", lambda: "local:gemma-4-31b")
    cad_engine.preflight()   # must not raise


def test_preflight_rejects_a_bare_ollama_tag(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_code_model", lambda: "qwen2.5-coder:7b-instruct-q4_K_M")
    with pytest.raises(RuntimeError, match="retired"):
        cad_engine.preflight()


class _FakeChatResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_no_think_is_by_intent_not_by_model_string(tmp_path, monkeypatch):
    """no_think must be a caller-declared kwarg, never inferred from `model == BRIEF_MODEL` —
    that string equality would also silently catch strong-rung codegen/revise/decide calls,
    which ride the exact same model string as BRIEF_MODEL (CODE_MODEL_STRONG) and must keep
    whatever reasoning_effort the server (resident or maker arm) is configured with."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    # Keep this offline: _ollama()'s local: branch calls _ensure_default_server() first,
    # which would otherwise shell out to real systemctl and probe a real health endpoint.
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)

    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data.decode()))
        return _FakeChatResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)

    cad_engine._ollama("local:qwen3.8-27b", "sys", "user")
    assert "chat_template_kwargs" not in captured[-1]

    cad_engine._ollama("local:qwen3.8-27b", "sys", "user", no_think=True)
    assert captured[-1]["chat_template_kwargs"] == {"enable_thinking": False}


class _FakeHealthResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"{}"


def teardown_module(module):
    os.environ.pop("CAD_CONFIG_FILE", None)
    os.environ.pop("CAD_KEEP_MAKER", None)
    os.environ.pop("CAD_CRITIC_URL", None)
    os.environ.pop("CAD_CRITIC_MODEL", None)
    import cad_v5.config as cfg
    importlib.reload(cfg)
    import cad_engine
    importlib.reload(cad_engine)


# ── Ollama retirement (2026-09-19) ─────────────────────────────────────────────

def test_ladder_is_one_local_rung(tmp_path, monkeypatch):
    """The 7B fast rung lived on Ollama. With Ollama off the box the local ladder is the
    strong rung alone, and CODE_MODEL_FAST keeps its name pointing at it so importers
    (fluid_gen, gift_sample, pinned benchmark legs) still resolve something real."""
    monkeypatch.delenv("CAD_CODE_MODEL_FAST", raising=False)
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    assert cfg.CODE_MODEL_LADDER == ["local:gemma-4-31b"]
    assert cfg.CODE_MODEL_FAST == cfg.CODE_MODEL_DEFAULT == cfg.CODE_MODEL_STRONG
    assert "ollama" not in cfg.CODE_MODEL_FAST.lower()


def test_coder_fast_resolves_to_the_strong_rung(tmp_path, monkeypatch):
    """--coder fast (and Satine's fast: prefix) must keep parsing — every saved command and
    benchmark leg uses it — but land on the only rung that exists."""
    monkeypatch.delenv("CAD_CODE_MODEL_FAST", raising=False)
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    sys.path.insert(0, str(HERE / "scripts"))
    import cad_engine; importlib.reload(cad_engine)
    import fluid_gen; importlib.reload(fluid_gen)
    for word in ("fast", "auto", "strong"):
        fluid_gen._model_for(word)
        assert fluid_gen.engine._ACTIVE_CODE_MODEL == "local:gemma-4-31b", word


def test_fast_override_refuses_an_ollama_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_CODE_MODEL_FAST", "qwen2.5-coder:7b-instruct-q4_K_M")
    with pytest.raises(RuntimeError, match="not a local: model"):
        _reload_with(tmp_path, {})
    monkeypatch.setenv("CAD_CODE_MODEL_FAST", "local:some-arm")
    cfg = _reload_with(tmp_path, {})
    assert cfg.CODE_MODEL_FAST == "local:some-arm"


def test_ollama_tag_raises_instead_of_calling_11434(tmp_path, monkeypatch):
    """No fallback through Ollama (user rule): a bare tag is a hard error, never an HTTP
    call to :11434."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)

    def no_http(*a, **k):
        pytest.fail("a bare Ollama tag must not reach the network")

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", no_http)
    with pytest.raises(RuntimeError, match="Ollama is retired"):
        cad_engine._ollama("qwen3:8b", "sys", "user")


def test_triage_is_a_free_no_op(tmp_path, monkeypatch):
    """With one local rung the triage answer is foregone; it must not spend a round trip."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_ollama",
                        lambda *a, **k: pytest.fail("triage must not call a model"))
    assert cad_engine.spec_needs_strong_coder("a 20mm cube", {}) is True


def test_vision_prepass_rides_the_local_rung_with_an_image_part(tmp_path, monkeypatch):
    """The gemma4:e4b pre-pass is gone: the analysis call must go to the strong rung's
    chat-completions endpoint, carry the photo as an OpenAI image part, and run thinking
    off with the same JSON contract."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_ensure_default_server", lambda *a, **k: None)
    monkeypatch.setattr(cad_engine, "_prep_image_b64", lambda p: "QUJD")
    sent = {}
    analysis = {"shape_family": "plate", "features": [], "proportions": "flat",
                "legible_dimensions_mm": [], "symmetry": "none",
                "suggested_spec": "a 50mm plate", "confidence": "low"}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        return _FakeChatResponse(json.dumps(
            {"choices": [{"message": {"content": json.dumps(analysis)}}]}).encode())

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)
    photo = tmp_path / "ref.jpg"
    photo.write_bytes(b"not-a-real-jpeg")
    assert cad_engine.analyze_reference_image(str(photo)) == analysis
    assert sent["url"] == "http://127.0.0.1:8088/v1/chat/completions"
    assert sent["body"]["model"] == "gemma-4-31b"
    parts = sent["body"]["messages"][1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].endswith("QUJD")
    assert sent["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent["body"]["response_format"]["type"] == "json_object"
    # the per-photo cache still short-circuits the second call
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("cached analysis must not re-call"))
    assert cad_engine.analyze_reference_image(str(photo)) == analysis

import importlib, json, logging, os, sys

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
    assert m == {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server",
                "arm": None}
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8086/v1/chat/completions"
    assert cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"


def test_enabled_maker_rewrites_port_and_alias(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "gemma-4-31b"}})
    m = cfg.maker_config()
    assert m["enabled"] and m["port"] == 8088 and m["unit"] == "maker-server"
    assert cfg.LOCAL_CODER_URL == "http://127.0.0.1:8088/v1/chat/completions"
    assert cfg.LOCAL_CODER_HEALTH == "http://127.0.0.1:8088/health"
    assert cfg.CODE_MODEL_STRONG == "local:gemma-4-31b"
    assert cfg.CODE_MODEL_LADDER == ["local:gemma-4-31b", "local:gemma-4-31b+think"]


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
    assert m == {"enabled": False, "port": 8086, "alias": "qwen3.8-27b", "unit": "qwen38-server",
                "arm": None}
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


# ── Task 1b: the think rung (2026-09-19) ───────────────────────────────────────

def _offline_ollama(monkeypatch, cad_engine, response_extra=None):
    """Common offline plumbing for the think-rung tests below: no systemctl, no health
    probe, no real network, just a captured request body per call."""
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data.decode()))
        msg = {"content": "ok"}
        if response_extra:
            msg.update(response_extra)
        return _FakeChatResponse(json.dumps(
            {"choices": [{"message": msg}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_think_suffix_sends_enable_thinking_true_to_the_same_alias_and_url(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "gemma-4-31b"}})
    import cad_engine; importlib.reload(cad_engine)
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append((req.full_url, json.loads(req.data.decode())))
        return _FakeChatResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(cad_engine.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)

    cad_engine._ollama("local:gemma-4-31b", "sys", "user")
    cad_engine._ollama("local:gemma-4-31b+think", "sys", "user")

    (url_plain, body_plain), (url_think, body_think) = captured
    assert url_plain == url_think == cfg.LOCAL_CODER_URL
    assert body_plain["model"] == body_think["model"] == "gemma-4-31b"   # suffix stripped
    assert "chat_template_kwargs" not in body_plain
    assert body_think["chat_template_kwargs"] == {"enable_thinking": True}
    assert body_think["max_tokens"] == cfg.CODE_MAX_TOKENS_THINK
    assert cfg.CODE_MAX_TOKENS_THINK >= 12000


def test_think_keyword_without_a_rung_suffix(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    captured = _offline_ollama(monkeypatch, cad_engine)
    cad_engine._ollama("local:qwen3.8-27b", "sys", "user", think=True)
    assert captured[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured[-1]["model"] == "qwen3.8-27b"


def test_no_think_wins_over_the_think_suffix(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    captured = _offline_ollama(monkeypatch, cad_engine)
    cad_engine._ollama("local:qwen3.8-27b+think", "sys", "user", no_think=True)
    assert captured[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured[-1]["model"] == "qwen3.8-27b"


def test_no_think_and_think_both_true_raises(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    with pytest.raises(ValueError):
        cad_engine._ollama("local:qwen3.8-27b", "sys", "user", no_think=True, think=True)


def test_images_call_never_sends_true_even_on_the_think_rung(tmp_path, monkeypatch):
    """The critic (and the image-analysis pre-pass) always pass images=. CRITIC_MODEL never
    carries "+think" by construction, but this must hold even if a caller somehow reached
    here with a think-suffixed model and an image, because the critic path must be provably
    safe, not merely unreachable in today's call graph."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    captured = _offline_ollama(monkeypatch, cad_engine)
    cad_engine._ollama("local:qwen3.8-27b+think", "sys", "user", images=["QUJD"])
    assert captured[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_schema_call_never_sends_true_even_on_the_think_rung(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    captured = _offline_ollama(monkeypatch, cad_engine)
    cad_engine._ollama("local:qwen3.8-27b+think", "sys", "user", fmt={"type": "object"})
    assert captured[-1]["chat_template_kwargs"]["enable_thinking"] is False


def test_reasoning_content_is_ignored_only_content_is_returned(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    _offline_ollama(monkeypatch, cad_engine,
                     response_extra={"reasoning_content": "a" * 4439, "content": "result = 1"})
    out = cad_engine._ollama("local:qwen3.8-27b+think", "sys", "user", think=True)
    assert out == "result = 1"


def test_usage_counts_completion_tokens_on_a_think_call(tmp_path, monkeypatch):
    """_LAST_USAGE / _USAGE_TOTAL must keep counting completion_tokens on a think call.
    The server's own usage block already includes reasoning tokens in that count, and
    _ollama() must not special-case the think rung when banking it."""
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cad_engine, "_wait_health", lambda url, timeout: None)

    def fake_urlopen(req, timeout=None):
        return _FakeChatResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"}}],
             "usage": {"prompt_tokens": 50, "completion_tokens": 1864}}).encode())

    monkeypatch.setattr(cad_engine.urllib.request, "urlopen", fake_urlopen)
    cad_engine.reset_usage()
    cad_engine._ollama("local:qwen3.8-27b+think", "sys", "user")
    assert cad_engine._USAGE_TOTAL["completion_tokens"] == 1864
    assert cad_engine._LAST_USAGE["completion_tokens"] == 1864


def test_code_timeout_uses_the_strong_cap_on_the_think_rung(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_code_model", lambda: cad_engine.CODE_MODEL_THINK)
    assert cad_engine._code_timeout() == cfg.CODE_TIMEOUT_STRONG


def test_ladder_climbs_from_strong_to_think_then_stops(tmp_path, monkeypatch):
    # gemma-4-31b's real benchmarks/arms.json entry launches with
    # enable_thinking:false, so think_rung_available() is True here (fix round 1: the
    # ladder only offers the think rung where it is a genuinely different request).
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b", "arm": "gemma-4-31b"}})
    import cad_engine; importlib.reload(cad_engine)
    assert cad_engine._next_code_model(cfg.CODE_MODEL_STRONG) == cfg.CODE_MODEL_THINK
    assert cad_engine._next_code_model(cfg.CODE_MODEL_THINK) is None


def test_fast_override_accepts_the_think_form(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_CODE_MODEL_FAST", "local:some-arm+think")
    cfg = _reload_with(tmp_path, {})
    assert cfg.CODE_MODEL_FAST == "local:some-arm+think"
    monkeypatch.setenv("CAD_CODE_MODEL_FAST", "qwen2.5-coder:7b-instruct-q4_K_M+think")
    with pytest.raises(RuntimeError, match="not a local: model"):
        _reload_with(tmp_path, {})


# ── Fix round 1: think_rung_available() ─────────────────────────────────────────

def test_think_rung_available_true_for_a_real_thinking_off_arm(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b", "arm": "gemma-4-31b"}})
    assert cfg.think_rung_available() is True


def test_think_rung_available_false_when_maker_disabled(tmp_path):
    cfg = _reload_with(tmp_path, {})
    assert cfg.think_rung_available() is False
    assert cfg.CODE_MODEL_LADDER == [cfg.CODE_MODEL_STRONG]


def test_think_rung_available_false_for_an_arm_without_the_thinking_off_flag(tmp_path):
    # devstral-small-2's real benchmarks/arms.json entry has extra_args="" -- it never
    # tells the server to launch with thinking off, so a "+think" request would not be
    # a meaningfully different call from the plain one.
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "devstral-small-2",
                                            "arm": "devstral-small-2"}})
    assert cfg.think_rung_available() is False
    assert cfg.CODE_MODEL_LADDER == [cfg.CODE_MODEL_STRONG]


def test_think_rung_available_false_and_no_exception_when_arms_json_unreadable(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b", "arm": "gemma-4-31b"}})
    monkeypatch.setattr(cfg, "ARMS_FILE", tmp_path / "does-not-exist.json")
    assert cfg.think_rung_available() is False


def test_think_rung_available_false_and_no_exception_when_arms_json_malformed(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b", "arm": "gemma-4-31b"}})
    bad = tmp_path / "arms.json"
    bad.write_text("not json")
    monkeypatch.setattr(cfg, "ARMS_FILE", bad)
    assert cfg.think_rung_available() is False


def test_think_rung_available_false_when_the_arm_is_absent_from_arms_json(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "some-unknown-alias",
                                            "arm": "some-unknown-arm"}})
    assert cfg.think_rung_available() is False


# ── Task 1b: repair_think config accessor ──────────────────────────────────────

def test_repair_think_defaults_off(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD_REPAIR_THINK", raising=False)
    cfg = _reload_with(tmp_path, {})
    assert cfg.repair_think_enabled() is False


def test_repair_think_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_REPAIR_THINK", "1")
    cfg = _reload_with(tmp_path, {})
    assert cfg.repair_think_enabled() is True


def test_repair_think_cad_json_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD_REPAIR_THINK", raising=False)
    cfg = _reload_with(tmp_path, {"repair_think": True})
    assert cfg.repair_think_enabled() is True


# ── lab_config() (Phase 3 Task 2, fix round 1: malformed-block tolerance) ──────

def test_lab_config_defaults(tmp_path):
    cfg = _reload_with(tmp_path, {})
    assert cfg.lab_config() == cfg._LAB_DEFAULTS
    assert cfg.lab_config() is not cfg._LAB_DEFAULTS   # never hand back the live default dict


def test_lab_config_partial_override_keeps_other_defaults(tmp_path):
    cfg = _reload_with(tmp_path, {"lab": {"harvest": {"hours_per_day": 6}}})
    lc = cfg.lab_config()
    assert lc["harvest"]["hours_per_day"] == 6
    assert lc["harvest"]["unit_minutes"] == 10   # untouched default survives
    assert lc["harvest"]["teacher_passes"] == cfg._LAB_DEFAULTS["harvest"]["teacher_passes"]


@pytest.mark.parametrize("bad_value", ["a string", ["a", "list"], 5])
def test_lab_config_malformed_top_level_falls_back_and_warns(tmp_path, bad_value, caplog):
    cfg = _reload_with(tmp_path, {"lab": bad_value})
    with caplog.at_level(logging.WARNING, logger="cad_v5"):
        lc = cfg.lab_config()
    assert lc == cfg._LAB_DEFAULTS
    assert any("lab" in r.message and "not an object" in r.message for r in caplog.records)


def test_lab_config_malformed_nested_block_falls_back_and_warns(tmp_path, caplog):
    """A malformed harvest sub-block ({"lab": {"harvest": "oops"}}) must fall back to
    just the harvest defaults, not silently store a non-dict one level deep."""
    cfg = _reload_with(tmp_path, {"lab": {"harvest": "oops"}})
    with caplog.at_level(logging.WARNING, logger="cad_v5"):
        lc = cfg.lab_config()
    assert lc["harvest"] == cfg._LAB_DEFAULTS["harvest"]
    assert any("lab.harvest" in r.message and "not an object" in r.message
              for r in caplog.records)


def test_lab_config_absent_lab_block_is_not_a_warning(tmp_path, caplog):
    cfg = _reload_with(tmp_path, {})
    with caplog.at_level(logging.WARNING, logger="cad_v5"):
        cfg.lab_config()
    assert caplog.records == []


def test_lab_config_returns_deep_copies_never_shared_with_the_defaults(tmp_path):
    """Fix round 2: a caller mutating a nested value in the dict lab_config() returns
    (a list append, an in-place key overwrite) must never corrupt the module-level
    _LAB_DEFAULTS -- this module is imported by the live engine and, per the plan, is
    called repeatedly from long-lived processes (the harvest unit, the web UI)."""
    cfg = _reload_with(tmp_path, {})
    lc1 = cfg.lab_config()
    lc1["harvest"]["temps"].append(999)
    lc1["harvest"]["unit_minutes"] = 999999
    lc1["harvest"]["teacher_passes"].append("some-new-pass")

    assert cfg._LAB_DEFAULTS["harvest"]["temps"] == [0.2, 0.5, 0.8]
    assert cfg._LAB_DEFAULTS["harvest"]["unit_minutes"] == 10
    assert cfg._LAB_DEFAULTS["harvest"]["teacher_passes"] == ["think"]

    lc2 = cfg.lab_config()
    assert lc2["harvest"]["temps"] == [0.2, 0.5, 0.8]
    assert lc2["harvest"]["unit_minutes"] == 10
    assert lc2 == cfg._LAB_DEFAULTS
    assert lc2 is not cfg._LAB_DEFAULTS
    assert lc2["harvest"] is not cfg._LAB_DEFAULTS["harvest"]


def test_lab_config_partial_override_also_returns_deep_copies(tmp_path):
    """The same guarantee holds on the merge path (a real override present), not just
    the all-defaults fallback path -- both branches of _sanitize_against_defaults
    deepcopy before returning."""
    cfg = _reload_with(tmp_path, {"lab": {"harvest": {"hours_per_day": 6}}})
    lc1 = cfg.lab_config()
    lc1["harvest"]["temps"].append(999)
    lc2 = cfg.lab_config()
    assert lc2["harvest"]["temps"] == [0.2, 0.5, 0.8]
    assert cfg._LAB_DEFAULTS["harvest"]["temps"] == [0.2, 0.5, 0.8]


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

def test_ladder_is_two_local_rungs(tmp_path, monkeypatch):
    """The 7B fast rung lived on Ollama. With Ollama off the box the local ladder is the
    strong rung, then the SAME arm thinking on (Task 1b, 2026-09-19, no second server), and
    CODE_MODEL_FAST keeps its name pointing at the strong rung so importers (fluid_gen,
    gift_sample, pinned benchmark legs) still resolve something real."""
    monkeypatch.delenv("CAD_CODE_MODEL_FAST", raising=False)
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    assert cfg.CODE_MODEL_LADDER == ["local:gemma-4-31b", "local:gemma-4-31b+think"]
    assert cfg.CODE_MODEL_THINK == "local:gemma-4-31b+think"
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

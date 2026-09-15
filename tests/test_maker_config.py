import importlib, json, os, sys
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
    assert cfg.CODE_MODEL_LADDER[1] == "local:gemma-4-31b"


def test_ensure_hook_starts_maker_and_resume_restores(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    import cad_engine
    importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine, "_unload_ollama_guests", lambda *_a, **_k: None)
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
    monkeypatch.setattr(cad_engine, "_unload_ollama_guests", lambda *_a, **_k: None)
    monkeypatch.setattr(cad_engine.urllib.request, "urlopen",
                         lambda url, timeout=3: _FakeHealthResponse())
    cad_engine._ensure_default_server(timeout=1)
    assert calls == []
    cad_engine._resume_default_server()
    assert calls == []


def test_brief_model_is_the_local_strong_rung(tmp_path):
    cfg = _reload_with(tmp_path, {})
    assert cfg.BRIEF_MODEL == cfg.CODE_MODEL_STRONG == "local:qwen3.8-27b"


def test_preflight_ignores_local_and_cloud_models(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {})
    import cad_engine; importlib.reload(cad_engine)
    monkeypatch.setattr(cad_engine, "_installed_ollama_models", lambda: set())  # nothing on Ollama
    monkeypatch.setattr(cad_engine, "_code_model", lambda: "local:qwen3.8-27b")
    cad_engine._preflight_models()   # must not raise


def test_pause_hook_stops_maker_when_enabled(tmp_path, monkeypatch):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088, "alias": "arm-x"}})
    import cad_engine; importlib.reload(cad_engine)
    calls = []
    monkeypatch.setattr(cad_engine.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    monkeypatch.setattr(cad_engine, "_model_size_gb", lambda m: 3.4)
    # The is-active probe goes through subprocess.run too (systemctl is-active <unit>); patch
    # it directly so the recorded calls list only carries the actual pause action, not the probe.
    monkeypatch.setattr(cad_engine, "_maker_server_active", lambda: True)
    cad_engine._PAUSED_DEFAULT_SERVER = False
    cad_engine._pause_default_server_for("gemma4:e4b")
    assert calls == [["systemctl", "--user", "stop", "maker-server"]]


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
    import cad_v5.config as cfg
    importlib.reload(cfg)
    import cad_engine
    importlib.reload(cad_engine)

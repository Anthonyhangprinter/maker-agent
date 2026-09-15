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


def teardown_module(module):
    os.environ.pop("CAD_CONFIG_FILE", None)
    import cad_v5.config as cfg
    importlib.reload(cfg)

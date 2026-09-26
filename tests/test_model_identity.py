"""cad_v5.config.model_identity() — the self-describing model chip the web UI shows on
every creation and every turn (owner request, 2026-09-27): past builds used different CAD
coders (Phase 0 shootout arms, the round1 fine-tune, the resident when maker is disabled),
and a revise turn today rides whatever cad.json's maker block currently names, which can
differ from the arm that built the original. This must read as "what actually built THIS",
not "whatever cad.json says right now" — see the docstring on model_identity() itself.

Run: python3 -m pytest tests/test_model_identity.py -q
"""
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _reload_with(tmp_path, cad_json: dict):
    p = tmp_path / "cad.json"
    p.write_text(json.dumps(cad_json))
    os.environ["CAD_CONFIG_FILE"] = str(p)
    import cad_v5.config as cfg
    return importlib.reload(cfg)


def test_resident_when_maker_disabled(tmp_path):
    cfg = _reload_with(tmp_path, {})
    d = cfg.model_identity("local:qwen3.8-27b")
    assert d == {"code_model": "local:qwen3.8-27b", "maker_enabled": False,
                "maker_arm": None, "engine_version": cfg.VERSION,
                "label": "qwen3.8-27b (resident)"}


def test_maker_arm_matching_current_config(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    d = cfg.model_identity("local:gemma-4-31b")
    assert d["maker_enabled"] is True
    assert d["maker_arm"] == "gemma-4-31b"
    assert d["label"] == "gemma-4-31b (maker)"
    assert d["engine_version"] == cfg.VERSION


def test_maker_arm_name_can_differ_from_alias(tmp_path):
    """maker_config()'s `arm` (the benchmarks/arms.json entry name) can differ from
    `alias` (the llama.cpp --alias the server answers as) — the label should use the arm
    name (what the owner recognises from the card), not the raw alias."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "qwen3.8-27b-nothink",
                                            "arm": "qwen3.8-27b-nothink-arm"}})
    d = cfg.model_identity("local:qwen3.8-27b-nothink")
    assert d["maker_arm"] == "qwen3.8-27b-nothink-arm"
    assert d["label"] == "qwen3.8-27b-nothink-arm (maker)"


def test_cloud_prefix(tmp_path):
    cfg = _reload_with(tmp_path, {})
    d = cfg.model_identity("cloud/claude-sonnet-5")
    assert d == {"code_model": "cloud/claude-sonnet-5", "maker_enabled": False,
                "maker_arm": None, "engine_version": cfg.VERSION,
                "label": "claude-sonnet-5 (cloud)"}


def test_think_suffix_is_flagged_in_the_label(tmp_path):
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    d = cfg.model_identity("local:gemma-4-31b+think")
    assert d["maker_arm"] == "gemma-4-31b"
    assert d["label"] == "gemma-4-31b (maker) +think"
    assert d["code_model"] == "local:gemma-4-31b+think"


def test_stale_alias_neither_current_maker_nor_resident_shows_raw_alias(tmp_path):
    """A build recorded under an arm that cad.json's maker block no longer names (the arm
    was swapped since) must not be mislabeled as the CURRENT arm or the resident — it
    should just show the raw alias plainly rather than guessing a role for it."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    d = cfg.model_identity("local:gemma-4-31b-cad-r1")
    assert d["maker_arm"] is None
    assert d["label"] == "gemma-4-31b-cad-r1"


def test_resident_alias_shown_even_when_maker_currently_enabled(tmp_path):
    """A build from before the maker arm was turned on (or one whose result explicitly
    resolved to the resident, e.g. an A/B leg) must still say "(resident)", even if
    cad.json's maker block is enabled right now."""
    cfg = _reload_with(tmp_path, {"maker": {"enabled": True, "port": 8088,
                                            "alias": "gemma-4-31b"}})
    d = cfg.model_identity("local:qwen3.8-27b")
    assert d["label"] == "qwen3.8-27b (resident)"
    assert d["maker_arm"] is None


def teardown_module(module):
    os.environ.pop("CAD_CONFIG_FILE", None)
    import cad_v5.config as cfg
    importlib.reload(cfg)

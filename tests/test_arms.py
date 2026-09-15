import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import arms


def test_arms_file_is_well_formed():
    a = arms.load_arms()
    assert {"qwen3.8-27b-nothink", "gemma-4-31b", "qwen3-coder-30b-a3b", "glm-4.7-flash",
            "devstral-small-2", "gpt-oss-20b", "qwen2.5-coder-7b"} <= set(a)
    for arm in a.values():
        assert arm["gguf"].endswith(".gguf") and arm["ctx"] >= 8192 and arm["alias"]


def test_render_env_has_every_key():
    arm = arms.load_arms()["gpt-oss-20b"]
    env = arms.render_env(arm)
    for key in ("MODEL=", "MMPROJ=", "CTX='16384'", "PORT='8088'", "ALIAS='gpt-oss-20b'", "EXTRA_ARGS="):
        assert key in env
    assert "/mnt/nvme-apps/LinuxModels/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf" in env


def test_render_env_values_are_single_quoted():
    # correction (controller ruling): the launcher `source`s this file, so every value
    # must be single-quoted or unquoted values containing spaces break bash.
    arm = arms.load_arms()["qwen3.8-27b-nothink"]
    env = arms.render_env(arm)
    for line in env.splitlines():
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        assert value.startswith("'") and value.endswith("'"), f"{key} value is not single-quoted: {value!r}"


def test_apply_arm_writes_env_and_cad_json(tmp_path):
    arm = arms.load_arms()["gemma-4-31b"]
    cad_json = tmp_path / "cad.json"; cad_json.write_text(json.dumps({"code_model": None}))
    env_path = tmp_path / "maker.env"
    arms.apply_arm(arm, cad_json, env_path)
    cfg = json.loads(cad_json.read_text())
    assert cfg["maker"] == {"enabled": True, "port": 8088, "alias": "gemma-4-31b", "arm": "gemma-4-31b"}
    assert cfg["code_model"] is None          # untouched keys survive
    assert "ALIAS='gemma-4-31b'" in env_path.read_text()


def test_restore_disables_maker(tmp_path):
    cad_json = tmp_path / "cad.json"
    cad_json.write_text(json.dumps({"maker": {"enabled": True, "port": 8088, "alias": "x", "arm": "x"}}))
    arms.disable_maker(cad_json)
    assert json.loads(cad_json.read_text())["maker"]["enabled"] is False

"""Offline tests for lab/ship.py (no torch/transformers/peft import, no network, no GPU).

Loaded by file path (not `import lab.ship`), same convention as test_lab_train.py, so this
test never depends on lab/.venv's training deps being on the machine running the plain repo
test suite -- every heavy import (torch, transformers, peft) lives inside merge_and_save()
in the module under test and is never touched here.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))  # `lab.data` (the chat-template canary's renderer) lives here

_spec = importlib.util.spec_from_file_location("lab_ship", HERE / "lab" / "ship.py")
ship = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ship
_spec.loader.exec_module(ship)


# ---------------------------------------------------------------------------
# scratch_paths / logging
# ---------------------------------------------------------------------------


def test_scratch_paths_are_all_under_scratch(tmp_path):
    paths = ship.scratch_paths(tmp_path)
    for key in ("bf16_base", "merged", "f16_gguf", "imatrix_dir", "imatrix_file", "log", "verify_ok"):
        assert str(paths[key]).startswith(str(tmp_path))
    assert paths["imatrix_file"] == paths["imatrix_dir"] / "imatrix_unsloth.gguf_file"
    assert paths["f16_gguf"].name == "gemma-4-31b-cad-F16.gguf"


def test_log_step_appends_json_lines(tmp_path):
    ship.log_step(tmp_path, "merge", "ok", 12.5, foo="bar")
    ship.log_step(tmp_path, "convert", "ok", 3.0)
    lines = ship.scratch_paths(tmp_path)["log"].read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["step"] == "merge" and row0["status"] == "ok" and row0["elapsed_s"] == 12.5 and row0["foo"] == "bar"


# ---------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------


def test_convert_argv():
    argv = ship.convert_argv("PY", Path("/m/merged"), Path("/o/out.gguf"))
    assert argv == [
        "PY", str(ship.CONVERT_SCRIPT), "/m/merged", "--outfile", "/o/out.gguf", "--outtype", "f16",
    ]


def test_quantize_argv_default_type():
    argv = ship.quantize_argv(Path("/i/im.gguf"), Path("/f/f16.gguf"), Path("/q/q4.gguf"))
    assert argv == [
        str(ship.LLAMA_QUANTIZE_BIN), "--imatrix", "/i/im.gguf", "/f/f16.gguf", "/q/q4.gguf", "Q4_K_M",
    ]


def test_quantize_argv_custom_type():
    argv = ship.quantize_argv(Path("/i/im.gguf"), Path("/f/f16.gguf"), Path("/q/q5.gguf"), "Q5_K_M")
    assert argv[-1] == "Q5_K_M"


def test_fetch_imatrix_argv_uses_hf_bin(monkeypatch):
    monkeypatch.setattr(ship, "_hf_bin", lambda: "FAKE_HF")
    argv = ship.fetch_imatrix_argv(Path("/scratch/imatrix"))
    assert argv == [
        "FAKE_HF", "download", "unsloth/gemma-4-31B-it-GGUF", "imatrix_unsloth.gguf_file",
        "--local-dir", "/scratch/imatrix",
    ]


# ---------------------------------------------------------------------------
# _hf_bin (fix round 1, INFO finding 4)
# ---------------------------------------------------------------------------


def test_hf_bin_prefers_the_pinned_path_when_present(tmp_path, monkeypatch):
    fake_hf = tmp_path / "hf"; fake_hf.write_text("x")
    monkeypatch.setattr(ship, "HF_BIN", fake_hf)
    assert ship._hf_bin() == str(fake_hf)


def test_hf_bin_falls_back_to_path_lookup_when_pinned_path_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ship, "HF_BIN", tmp_path / "does-not-exist" / "hf")
    assert ship._hf_bin() == "hf"


def test_verify_server_argv():
    argv = ship.verify_server_argv(Path("/g/model.gguf"), Path("/g/mmproj.gguf"), 8093)
    assert argv == [
        str(ship.LLAMA_SERVER_BIN),
        "-m", "/g/model.gguf",
        "--mmproj", "/g/mmproj.gguf",
        "-ngl", "99",
        "-c", "8192",
        "--port", "8093",
        "--host", "127.0.0.1",
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]


# ---------------------------------------------------------------------------
# pick_converter_python
# ---------------------------------------------------------------------------


def test_pick_converter_python_prefers_system():
    picked = ship.pick_converter_python(checker=lambda p: True)
    assert picked == ship.SYSTEM_PYTHON


def test_pick_converter_python_falls_back_to_lab_venv():
    picked = ship.pick_converter_python(checker=lambda p: p == str(ship.LAB_VENV_PYTHON))
    assert picked == str(ship.LAB_VENV_PYTHON)


def test_pick_converter_python_raises_when_neither_works():
    with pytest.raises(SystemExit):
        ship.pick_converter_python(checker=lambda p: False)


# ---------------------------------------------------------------------------
# MemAvailable guard
# ---------------------------------------------------------------------------


def test_read_mem_available_kb_parses_the_field(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65000000 kB\nMemAvailable:   52976668 kB\nMemFree: 1 kB\n")
    assert ship.read_mem_available_kb(meminfo) == 52976668


def test_read_mem_available_kb_raises_when_field_missing(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65000000 kB\n")
    with pytest.raises(ValueError):
        ship.read_mem_available_kb(meminfo)


def test_check_mem_available_raises_below_floor(monkeypatch):
    monkeypatch.setattr(ship, "read_mem_available_kb", lambda meminfo_path=None: 10 * 1024 * 1024)  # 10GB
    with pytest.raises(SystemExit, match="MemAvailable"):
        ship.check_mem_available(min_gb=40.0)


def test_check_mem_available_returns_gb_above_floor(monkeypatch):
    monkeypatch.setattr(ship, "read_mem_available_kb", lambda meminfo_path=None: 50 * 1024 * 1024)  # 50GB
    gb = ship.check_mem_available(min_gb=40.0)
    assert gb == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# copy_aux_files
# ---------------------------------------------------------------------------


def test_copy_aux_files_copies_every_non_weight_file_and_skips_existing(tmp_path):
    """Fix round 1, HIGH finding 1: copy_aux_files must copy EVERY top-level regular file
    except weight artifacts and repo plumbing, not an allow-list of filename patterns (the
    old pattern list silently missed special_tokens_map.json / added_tokens.json)."""
    base = tmp_path / "base"; base.mkdir()
    merged = tmp_path / "merged"; merged.mkdir()
    (base / "config.json").write_text('{"base": true}')
    (base / "generation_config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    (base / "tokenizer_config.json").write_text("{}")
    (base / "tokenizer.model").write_bytes(b"spm")
    (base / "special_tokens_map.json").write_text("{}")   # the exact gap the review found
    (base / "added_tokens.json").write_text("{}")          # same gap
    (base / "chat_template.jinja").write_text("{{ bos_token }}")
    (base / "processor_config.json").write_text("{}")
    (base / "preprocessor_config.json").write_text("{}")
    # weight artifacts and repo plumbing -- must NOT be copied
    (base / "model.safetensors").write_text("not aux")
    (base / "model-00001-of-00002.safetensors").write_text("not aux")
    (base / "model.safetensors.index.json").write_text("{}")
    (base / "some-mtp.gguf").write_text("not aux")
    (base / "pytorch_model.bin").write_text("not aux")
    (base / "adapter.pt").write_text("not aux")
    (base / ".gitattributes").write_text("*.bin filter=lfs")
    (base / "README.md").write_text("# model card")
    # merged already has its own config.json (as if save_pretrained wrote it) -- must survive
    (merged / "config.json").write_text('{"merged": true}')

    copied = ship.copy_aux_files(base, merged)

    assert "config.json" not in copied  # already present in merged, left alone
    assert json.loads((merged / "config.json").read_text()) == {"merged": True}
    assert set(copied) == {
        "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
        "processor_config.json", "preprocessor_config.json",
    }
    for weight_or_plumbing in (
        "model.safetensors", "model-00001-of-00002.safetensors", "model.safetensors.index.json",
        "some-mtp.gguf", "pytorch_model.bin", "adapter.pt", ".gitattributes", "README.md",
    ):
        assert not (merged / weight_or_plumbing).exists()


def test_copy_aux_files_missing_optional_files_are_fine(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "config.json").write_text("{}")
    merged = tmp_path / "merged"
    copied = ship.copy_aux_files(base, merged)
    assert copied == ["config.json"]


def test_copy_aux_files_never_overwrites_generation_config_either(tmp_path):
    """The coordinator's ruling calls out config.json AND generation_config.json by name as
    files that must never be overwritten -- both are just instances of the general
    never-clobber-an-existing-destination rule, exercised here explicitly."""
    base = tmp_path / "base"; base.mkdir()
    (base / "generation_config.json").write_text('{"base": true}')
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "generation_config.json").write_text('{"merged": true}')
    copied = ship.copy_aux_files(base, merged)
    assert "generation_config.json" not in copied
    assert json.loads((merged / "generation_config.json").read_text()) == {"merged": True}


# ---------------------------------------------------------------------------
# streaming per-tensor LoRA merge (fix round 2 -- replaces the full transformers+PEFT load)
# ---------------------------------------------------------------------------


def _make_base_checkpoint(tmp_path, w_target, w_other1, w_other2, target_key):
    """A 2-shard fp32 base checkpoint (mirrors the real Gemma-4-31B-it bf16 base's shape --
    multiple shards, an index.json, plain HF tensor keys -- just fp32 and tiny) with one
    targeted tensor plus two untouched ones."""
    from safetensors.torch import save_file

    base_dir = tmp_path / "base"; base_dir.mkdir()
    shard1 = {target_key: w_target, "model.embed_tokens.weight": w_other1}
    shard2 = {"model.language_model.layers.1.mlp.down_proj.weight": w_other2}
    save_file(shard1, str(base_dir / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(shard2, str(base_dir / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    total_size = sum(t.numel() * t.element_size() for t in {**shard1, **shard2}.values())
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )
    return base_dir


def _make_adapter(tmp_path, target_base_key, a, b, r, alpha, name="adapter", extra_config=None):
    from safetensors.torch import save_file

    adapter_dir = tmp_path / name; adapter_dir.mkdir()
    stem = target_base_key[: -len(".weight")]
    tensors = {
        f"base_model.model.{stem}.lora_A.weight": a,
        f"base_model.model.{stem}.lora_B.weight": b,
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = {"r": r, "lora_alpha": alpha, "fan_in_fan_out": False}
    if extra_config:
        config.update(extra_config)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    return adapter_dir


TARGET_KEY = "model.language_model.layers.0.mlp.down_proj.weight"


def test_base_key_for_adapter_key_maps_lora_a_and_b():
    key_a = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    key_b = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight"
    assert ship.base_key_for_adapter_key(key_a) == TARGET_KEY
    assert ship.base_key_for_adapter_key(key_b) == TARGET_KEY


def test_base_key_for_adapter_key_none_for_unrelated_key():
    assert ship.base_key_for_adapter_key("some.other.key.weight") is None
    assert ship.base_key_for_adapter_key("base_model.model.something.weight") is None  # no lora suffix


def test_lora_scale_default_is_alpha_over_r():
    assert ship.lora_scale({"r": 16, "lora_alpha": 16}) == 1.0
    assert ship.lora_scale({"r": 2, "lora_alpha": 4}) == 2.0


def test_lora_scale_rslora_uses_alpha_over_sqrt_r():
    import math
    assert ship.lora_scale({"r": 4, "lora_alpha": 8, "use_rslora": True}) == pytest.approx(8 / math.sqrt(4))


def test_build_lora_pairs_raises_on_unpaired_key():
    tensors = {"base_model.model.model.x.lora_A.weight": object()}
    with pytest.raises(ValueError, match="unpaired"):
        ship.build_lora_pairs(tensors)


def test_build_lora_pairs_pairs_up_a_and_b():
    tensors = {
        "base_model.model.model.x.lora_A.weight": "A",
        "base_model.model.model.x.lora_B.weight": "B",
        "irrelevant.key": "ignored",
    }
    pairs = ship.build_lora_pairs(tensors)
    assert pairs == {
        "model.x.weight": (
            "base_model.model.model.x.lora_A.weight", "base_model.model.model.x.lora_B.weight",
        )
    }


def test_base_model_index_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit, match="model.safetensors.index.json"):
        ship.base_model_index(tmp_path)


def test_base_model_index_reads_the_real_file(tmp_path):
    idx = {"metadata": {"total_size": 5}, "weight_map": {"a": "x.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(idx))
    assert ship.base_model_index(tmp_path) == idx


def test_shard_files_in_order_sorted_and_deduped():
    index = {"weight_map": {
        "a": "model-00002-of-00002.safetensors",
        "b": "model-00001-of-00002.safetensors",
        "c": "model-00001-of-00002.safetensors",
    }}
    assert ship.shard_files_in_order(index) == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    ]


def test_plan_output_shards_splits_when_over_the_byte_cap(tmp_path):
    import torch
    from safetensors.torch import save_file

    base_dir = tmp_path / "base"; base_dir.mkdir()
    t1 = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    t2 = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    save_file({"a": t1, "b": t2}, str(base_dir / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": 800},
        "weight_map": {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"},
    }
    key_to_shard, n_shards, total_bytes = ship.plan_output_shards(base_dir, index, max_shard_bytes=400)
    assert n_shards == 2
    assert key_to_shard == {"a": 0, "b": 1}
    assert total_bytes == 800


def test_load_adapter_reads_tensors_and_config(tmp_path):
    import torch

    adapter_dir = _make_adapter(
        tmp_path, TARGET_KEY, torch.randn(2, 4), torch.randn(8, 2), r=2, alpha=4,
    )
    tensors, config = ship.load_adapter(adapter_dir)
    assert set(tensors) == {
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
    }
    assert config["r"] == 2 and config["lora_alpha"] == 4


def test_streaming_merge_applies_lora_delta_and_leaves_others_untouched(tmp_path):
    import torch
    from safetensors import safe_open

    torch.manual_seed(0)
    w_target = torch.randn(8, 4, dtype=torch.float32)
    w_other1 = torch.randn(6, dtype=torch.float32)
    w_other2 = torch.randn(3, 5, dtype=torch.float32)
    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(8, 2, dtype=torch.float32)

    base_dir = _make_base_checkpoint(tmp_path, w_target, w_other1, w_other2, TARGET_KEY)
    adapter_dir = _make_adapter(tmp_path, TARGET_KEY, a, b, r=2, alpha=4)  # scale = 4/2 = 2

    out_dir = tmp_path / "merged"
    summary = ship.streaming_merge(adapter_dir, base_dir, out_dir)

    assert summary["tensor_count"] == 3
    assert summary["lora_pairs_applied"] == 1
    assert summary["n_shards"] == 1

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        TARGET_KEY, "model.embed_tokens.weight", "model.language_model.layers.1.mlp.down_proj.weight",
    }
    assert set(index["weight_map"].values()) == {"model-00001-of-00001.safetensors"}
    expected_total = (w_target.numel() + w_other1.numel() + w_other2.numel()) * 4  # fp32 = 4 bytes
    assert index["metadata"]["total_size"] == expected_total

    with safe_open(str(out_dir / "model-00001-of-00001.safetensors"), framework="pt") as f:
        merged_target = f.get_tensor(TARGET_KEY)
        merged_other1 = f.get_tensor("model.embed_tokens.weight")
        merged_other2 = f.get_tensor("model.language_model.layers.1.mlp.down_proj.weight")

    expected = w_target + 2.0 * (b @ a)  # scale = alpha/r = 4/2 = 2
    assert torch.allclose(merged_target, expected, atol=1e-6)
    assert torch.equal(merged_other1, w_other1)
    assert torch.equal(merged_other2, w_other2)


def test_streaming_merge_raises_on_missing_target(tmp_path):
    import torch

    base_dir = _make_base_checkpoint(
        tmp_path, torch.randn(8, 4), torch.randn(3), torch.randn(2, 2), TARGET_KEY,
    )
    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(8, 2, dtype=torch.float32)
    adapter_dir = _make_adapter(tmp_path, "model.does.not.exist.weight", a, b, r=2, alpha=4)

    with pytest.raises(ValueError, match="not found in the base model"):
        ship.streaming_merge(adapter_dir, base_dir, tmp_path / "merged")


def test_streaming_merge_raises_on_fan_in_fan_out(tmp_path):
    import torch

    base_dir = _make_base_checkpoint(
        tmp_path, torch.randn(8, 4), torch.randn(3), torch.randn(2, 2), TARGET_KEY,
    )
    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(8, 2, dtype=torch.float32)
    adapter_dir = _make_adapter(
        tmp_path, TARGET_KEY, a, b, r=2, alpha=4, extra_config={"fan_in_fan_out": True},
    )

    with pytest.raises(ValueError, match="fan_in_fan_out"):
        ship.streaming_merge(adapter_dir, base_dir, tmp_path / "merged")


def test_streaming_merge_raises_on_shape_mismatch(tmp_path):
    import torch

    base_dir = _make_base_checkpoint(
        tmp_path, torch.randn(8, 4), torch.randn(3), torch.randn(2, 2), TARGET_KEY,
    )
    a = torch.randn(2, 5, dtype=torch.float32)  # wrong in_features -> B@A won't be 8x4
    b = torch.randn(8, 2, dtype=torch.float32)
    adapter_dir = _make_adapter(tmp_path, TARGET_KEY, a, b, r=2, alpha=4)

    with pytest.raises(ValueError, match="shape"):
        ship.streaming_merge(adapter_dir, base_dir, tmp_path / "merged")


def test_cmd_merge_runs_the_streaming_merge_copies_aux_files_and_writes_marker(tmp_path, monkeypatch):
    import torch

    torch.manual_seed(1)
    w_target = torch.randn(8, 4, dtype=torch.float32)
    base_dir = _make_base_checkpoint(
        tmp_path, w_target, torch.randn(6), torch.randn(3, 5), TARGET_KEY,
    )
    (base_dir / "config.json").write_text('{"model_type": "gemma4"}')  # comes from the BASE now
    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(8, 2, dtype=torch.float32)
    adapter_dir = _make_adapter(tmp_path, TARGET_KEY, a, b, r=2, alpha=4)

    out_dir = tmp_path / "merged"
    monkeypatch.setattr(ship, "check_mem_available", lambda: 50.0)
    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), force=False,
        adapter=str(adapter_dir), base=str(base_dir), out=str(out_dir),
        train_template=None, skip_template_check=True,  # the fake base ships no chat template
    )
    ship.cmd_merge(args)

    assert (out_dir / "model.safetensors.index.json").exists()
    assert (out_dir / "config.json").read_text() == '{"model_type": "gemma4"}'  # copied from base
    assert ship.marker_path_for_dir(out_dir).exists()
    log_lines = ship.scratch_paths(tmp_path / "scratch")["log"].read_text().splitlines()
    assert json.loads(log_lines[-1])["step"] == "merge"


# ---------------------------------------------------------------------------
# dir_size_bytes
# ---------------------------------------------------------------------------


def test_dir_size_bytes_sums_files_recursively(tmp_path):
    d = tmp_path / "d"; (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"x" * 10)
    (d / "sub" / "b.bin").write_bytes(b"y" * 20)
    assert ship.dir_size_bytes(d) == 30


def test_dir_size_bytes_single_file(tmp_path):
    f = tmp_path / "f.bin"; f.write_bytes(b"z" * 7)
    assert ship.dir_size_bytes(f) == 7


def test_dir_size_bytes_missing_path_is_zero(tmp_path):
    assert ship.dir_size_bytes(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# completeness markers (fix round 1, HIGH finding 2)
# ---------------------------------------------------------------------------


def test_marker_path_for_dir_is_inside_the_directory():
    assert ship.marker_path_for_dir(Path("/a/b")) == Path("/a/b/.ship_done")


def test_marker_path_for_file_is_a_sibling_dot_done():
    assert ship.marker_path_for_file(Path("/a/b/out.gguf")) == Path("/a/b/out.gguf.done")


def test_dir_output_is_complete_false_until_marker_written(tmp_path):
    d = tmp_path / "d"; d.mkdir()
    assert ship.dir_output_is_complete(d) is False
    ship.write_dir_done_marker(d, "merge", 5.0)
    assert ship.dir_output_is_complete(d) is True


def test_write_dir_done_marker_records_expected_fields(tmp_path):
    d = tmp_path / "d"; d.mkdir()
    (d / "a.bin").write_bytes(b"x" * 100)
    marker = ship.write_dir_done_marker(d, "merge", 12.5)
    row = json.loads(marker.read_text())
    assert row["step"] == "merge" and row["elapsed_s"] == 12.5 and row["size_bytes"] == 100
    assert "ts" in row


def test_file_output_is_complete_false_without_marker(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    assert ship.file_output_is_complete(f) is False


def test_write_file_done_marker_records_expected_fields(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    marker = ship.write_file_done_marker(f, "convert", 3.5)
    row = json.loads(marker.read_text())
    assert row == {"step": "convert", "elapsed_s": 3.5, "size_bytes": 10, "ts": row["ts"]}


def test_file_output_is_complete_true_when_marker_size_matches(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    ship.write_file_done_marker(f, "convert", 3.0)
    assert ship.file_output_is_complete(f) is True


def test_file_output_is_complete_false_when_size_mismatches(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    ship.write_file_done_marker(f, "convert", 3.0)
    f.write_bytes(b"y" * 3)  # something replaced the file after the marker was written
    assert ship.file_output_is_complete(f) is False


def test_file_output_is_complete_false_when_file_missing(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    ship.write_file_done_marker(f, "convert", 3.0)
    f.unlink()
    assert ship.file_output_is_complete(f) is False


def test_file_output_is_complete_false_when_marker_is_corrupt(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 10)
    ship.marker_path_for_file(f).write_text("not json")
    assert ship.file_output_is_complete(f) is False


def test_should_skip_file_only_true_with_a_matching_marker(tmp_path):
    f = tmp_path / "f.gguf"; f.write_bytes(b"x" * 4)
    assert ship._should_skip(f, force=False, label="convert") is False   # exists, no marker
    ship.write_file_done_marker(f, "convert", 1.0)
    assert ship._should_skip(f, force=False, label="convert") is True    # marker matches
    assert ship._should_skip(f, force=True, label="convert") is False    # --force always proceeds


def test_should_skip_dir_only_true_with_a_marker(tmp_path):
    d = tmp_path / "d"; d.mkdir()
    assert ship._should_skip(d, force=False, label="merge", is_dir=True) is False
    ship.write_dir_done_marker(d, "merge", 1.0)
    assert ship._should_skip(d, force=False, label="merge", is_dir=True) is True


# ---------------------------------------------------------------------------
# idempotency: merge/convert/fetch-imatrix/quantize skip on completeness, not bare
# existence (fix round 1, HIGH finding 2) -- an existing-but-unmarked output must never be
# treated as done, a matching marker must skip, and a size mismatch against the marker must
# not skip.
# ---------------------------------------------------------------------------


def test_cmd_merge_does_not_skip_without_a_marker(tmp_path, monkeypatch):
    out_dir = tmp_path / "merged"; out_dir.mkdir()
    (out_dir / "config.json").write_text("{}")  # partial leftover from a killed run, no marker

    def boom(*a, **kw):
        raise AssertionError("must not reach check_mem_available -- proves it did not skip AND "
                              "did not incorrectly stop earlier for an unrelated reason")

    monkeypatch.setattr(ship, "check_mem_available", boom)
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, adapter=str(tmp_path / "nonexistent-adapter"),
        base=None, out=str(out_dir),
    )
    # Proceeds past the (non-)skip, then fails on the missing adapter dir -- proves the skip
    # path was NOT taken (a real skip returns silently with no exception at all).
    with pytest.raises(SystemExit, match="adapter dir not found"):
        ship.cmd_merge(args)


def test_cmd_merge_skips_when_marker_present(tmp_path, monkeypatch):
    out_dir = tmp_path / "merged"; out_dir.mkdir()
    ship.write_dir_done_marker(out_dir, "merge", 1.0)

    def boom(*a, **kw):
        raise AssertionError("merge_and_save must not run when the marker says it's already done")

    monkeypatch.setattr(ship, "merge_and_save", boom)
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, adapter=str(tmp_path / "nonexistent-adapter"),
        base=None, out=str(out_dir),
    )
    ship.cmd_merge(args)  # must not raise -- skipped before the (missing) adapter is checked


def test_cmd_convert_does_not_skip_without_a_marker(tmp_path):
    out_file = tmp_path / "out.gguf"; out_file.write_text("leftover partial junk, no marker")
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, merged=str(tmp_path / "nope"), out=str(out_file), python=None,
    )
    # Proceeds past the (non-)skip, then fails on the missing merged dir -- proves it did not skip.
    with pytest.raises(SystemExit, match="merged HF dir not found"):
        ship.cmd_convert(args)


def test_cmd_convert_skips_when_marker_matches_size(tmp_path, monkeypatch):
    out_file = tmp_path / "out.gguf"; out_file.write_bytes(b"x" * 10)
    ship.write_file_done_marker(out_file, "convert", 1.0)

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the marker matches")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, merged=str(tmp_path), out=str(out_file), python=None)
    ship.cmd_convert(args)  # must not raise


def test_cmd_convert_does_not_skip_when_marker_size_mismatches(tmp_path):
    out_file = tmp_path / "out.gguf"; out_file.write_bytes(b"x" * 10)
    ship.write_file_done_marker(out_file, "convert", 1.0)
    out_file.write_bytes(b"y" * 3)  # size changed since the marker was written

    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, merged=str(tmp_path / "nope"), out=str(out_file), python=None,
    )
    with pytest.raises(SystemExit, match="merged HF dir not found"):
        ship.cmd_convert(args)


def test_cmd_fetch_imatrix_does_not_skip_without_a_marker(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"; out_dir.mkdir()
    dest = out_dir / "imatrix_unsloth.gguf_file"
    dest.write_text("leftover, no marker")
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        dest.write_text("refetched")

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)
    assert calls, "fetch-imatrix must not have skipped when no marker was present"


def test_cmd_fetch_imatrix_skips_when_marker_matches_size(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"; out_dir.mkdir()
    dest = out_dir / "imatrix_unsloth.gguf_file"
    dest.write_bytes(b"x" * 20)
    ship.write_file_done_marker(dest, "fetch-imatrix", 1.0)

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the marker matches")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)  # must not raise


def test_cmd_fetch_imatrix_does_not_skip_when_marker_size_mismatches(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"; out_dir.mkdir()
    dest = out_dir / "imatrix_unsloth.gguf_file"
    dest.write_bytes(b"x" * 20)
    ship.write_file_done_marker(dest, "fetch-imatrix", 1.0)
    dest.write_bytes(b"y" * 5)  # size changed since the marker was written
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        dest.write_bytes(b"z" * 20)

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)
    assert calls, "fetch-imatrix must not have skipped on a marker/size mismatch"


def test_cmd_quantize_does_not_skip_without_a_marker(tmp_path):
    out_file = tmp_path / "q.gguf"; out_file.write_text("leftover junk, no marker")
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, f16=str(tmp_path / "nope.gguf"),
        imatrix=str(tmp_path / "nope2.gguf"), out=str(out_file), type="Q4_K_M",
    )
    with pytest.raises(SystemExit, match="F16 GGUF not found"):
        ship.cmd_quantize(args)


def test_cmd_quantize_skips_when_marker_matches_size(tmp_path, monkeypatch):
    out_file = tmp_path / "q.gguf"; out_file.write_bytes(b"x" * 10)
    ship.write_file_done_marker(out_file, "quantize", 1.0)

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the marker matches")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, f16=str(tmp_path / "f16.gguf"),
        imatrix=str(tmp_path / "im.gguf"), out=str(out_file), type="Q4_K_M",
    )
    ship.cmd_quantize(args)  # must not raise


def test_cmd_quantize_does_not_skip_when_marker_size_mismatches(tmp_path):
    out_file = tmp_path / "q.gguf"; out_file.write_bytes(b"x" * 10)
    ship.write_file_done_marker(out_file, "quantize", 1.0)
    out_file.write_bytes(b"yy" * 10)  # size changed since the marker was written

    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, f16=str(tmp_path / "nope.gguf"),
        imatrix=str(tmp_path / "nope2.gguf"), out=str(out_file), type="Q4_K_M",
    )
    with pytest.raises(SystemExit, match="F16 GGUF not found"):
        ship.cmd_quantize(args)


def test_cmd_convert_raises_when_merged_dir_missing(tmp_path):
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, merged=str(tmp_path / "nope"),
        out=str(tmp_path / "out.gguf"), python=None,
    )
    with pytest.raises(SystemExit, match="merged HF dir not found"):
        ship.cmd_convert(args)


def test_cmd_quantize_raises_when_inputs_missing(tmp_path):
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, f16=str(tmp_path / "nope.gguf"),
        imatrix=str(tmp_path / "nope2.gguf"), out=str(tmp_path / "out.gguf"), type="Q4_K_M",
    )
    with pytest.raises(SystemExit, match="F16 GGUF not found"):
        ship.cmd_quantize(args)


def test_cmd_convert_writes_to_a_partial_name_then_replaces_atomically(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    out_file = tmp_path / "out.gguf"
    partial = out_file.with_name(out_file.name + ".partial")
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        Path(argv[4]).write_text("f16 bytes")  # simulate convert_hf_to_gguf.py writing --outfile

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, merged=str(merged),
                              out=str(out_file), python="FAKE_PY", model_name=None)
    ship.cmd_convert(args)

    # argv named the .partial path, and carries the GGUF's general.name (finding 23)
    assert calls == [ship.convert_argv("FAKE_PY", merged, partial, "out")]
    assert not partial.exists()                                      # renamed away, never left behind
    assert out_file.read_text() == "f16 bytes"
    marker = ship.marker_path_for_file(out_file)
    assert marker.exists()
    assert json.loads(marker.read_text())["size_bytes"] == out_file.stat().st_size
    log_lines = ship.scratch_paths(tmp_path)["log"].read_text().splitlines()
    assert json.loads(log_lines[-1])["step"] == "convert"


def test_cmd_quantize_writes_to_a_partial_name_then_replaces_atomically(tmp_path, monkeypatch):
    f16 = tmp_path / "f16.gguf"; f16.write_text("x")
    imatrix = tmp_path / "im.gguf"; imatrix.write_text("x")
    out_file = tmp_path / "out.gguf"
    partial = out_file.with_name(out_file.name + ".partial")
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        Path(argv[4]).write_text("quantized bytes")  # simulate llama-quantize writing its output arg

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, f16=str(f16), imatrix=str(imatrix), out=str(out_file), type="Q4_K_M")
    ship.cmd_quantize(args)

    assert calls == [ship.quantize_argv(imatrix, f16, partial, "Q4_K_M")]
    assert not partial.exists()
    assert out_file.read_text() == "quantized bytes"
    marker = ship.marker_path_for_file(out_file)
    assert marker.exists()
    assert json.loads(marker.read_text())["size_bytes"] == out_file.stat().st_size


def test_cmd_fetch_imatrix_calls_subprocess_with_the_right_argv(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"
    dest = out_dir / "imatrix_unsloth.gguf_file"
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text("imatrix bytes")  # simulate `hf download` landing the file

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)

    assert calls == [ship.fetch_imatrix_argv(out_dir)]
    marker = ship.marker_path_for_file(dest)
    assert marker.exists()
    assert json.loads(marker.read_text())["size_bytes"] == dest.stat().st_size


# ---------------------------------------------------------------------------
# verify: pure checks
# ---------------------------------------------------------------------------


def test_build_chat_request_shape():
    req = ship.build_chat_request("hello", alias="my-arm")
    assert req["model"] == "my-arm"
    assert req["messages"] == [{"role": "user", "content": "hello"}]


def test_check_verify_reply_raises_on_empty():
    with pytest.raises(SystemExit, match="empty reply"):
        ship.check_verify_reply("cube", "", True)
    with pytest.raises(SystemExit, match="empty reply"):
        ship.check_verify_reply("cube", "   ", False)


def test_check_verify_reply_raises_when_build123d_missing():
    with pytest.raises(SystemExit, match="from build123d import"):
        ship.check_verify_reply("cube", "here is some code but no import", True)


def test_check_verify_reply_passes_with_build123d_import():
    ship.check_verify_reply("cube", "from build123d import *\nresult = Box(20, 20, 20)", True)


def test_check_verify_reply_text_only_does_not_need_build123d():
    ship.check_verify_reply("text", "OK", False)


def test_tokens_per_second_prefers_server_reported_value():
    assert ship._tokens_per_second({}, {"predicted_per_second": 12.3}) == 12.3


def test_tokens_per_second_derives_from_usage_and_timing():
    val = ship._tokens_per_second({"completion_tokens": 100}, {"predicted_ms": 2000})
    assert val == pytest.approx(50.0)


def test_tokens_per_second_none_when_no_data():
    assert ship._tokens_per_second({}, {}) is None


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def _sample_arms_data():
    return {
        "store": "/store",
        "arms": [
            {
                "name": "gemma-4-31b", "alias": "gemma-4-31b", "role": "candidate",
                "gguf": "gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf",
                "mmproj": "gemma-4-31B-it-GGUF/mmproj-F16.gguf", "ctx": 16384,
                "extra_args": '--chat-template-kwargs {"enable_thinking":false}',
                "hf": None, "notes": "dense 31B",
            },
        ],
    }


def test_build_spike_arm_clones_the_base_arm():
    data = _sample_arms_data()
    arm = ship.build_spike_arm(data, "gemma-4-31b-cad-spike", "lab/runs/spike1/adapter", "gemma-4-31B-cad-spike/out.gguf")
    assert arm["name"] == "gemma-4-31b-cad-spike"
    assert arm["alias"] == "gemma-4-31b-cad-spike"
    assert arm["role"] == "spike"
    assert arm["gguf"] == "gemma-4-31B-cad-spike/out.gguf"
    assert arm["mmproj"] == "gemma-4-31B-it-GGUF/mmproj-F16.gguf"
    assert arm["ctx"] == 16384
    assert arm["extra_args"] == '--chat-template-kwargs {"enable_thinking":false}'
    assert "Q4_K_M" in arm["notes"] and "lab/runs/spike1/adapter" in arm["notes"]
    assert "UD dynamic quant" in arm["notes"]


def test_build_spike_arm_raises_when_base_arm_missing():
    data = _sample_arms_data()
    with pytest.raises(StopIteration):
        ship.build_spike_arm(data, "x", "adapter", "x.gguf", base_arm_name="does-not-exist")


def test_register_arm_inserts_new_arm():
    data = _sample_arms_data()
    arm = ship.build_spike_arm(data, "gemma-4-31b-cad-spike", "adapter", "x.gguf")
    new_data = ship.register_arm(data, arm)
    names = [a["name"] for a in new_data["arms"]]
    assert names == ["gemma-4-31b", "gemma-4-31b-cad-spike"]
    assert data["arms"] == _sample_arms_data()["arms"]  # input untouched


def test_register_arm_refuses_duplicate_without_force():
    data = _sample_arms_data()
    dup = ship.build_spike_arm(data, "gemma-4-31b", "adapter", "x.gguf")  # collides with the base arm's own name
    with pytest.raises(SystemExit, match="already exists"):
        ship.register_arm(data, dup)


def test_register_arm_replaces_duplicate_with_force():
    data = _sample_arms_data()
    replacement = ship.build_spike_arm(data, "gemma-4-31b", "adapter", "x.gguf")
    new_data = ship.register_arm(data, replacement, force=True)
    assert len(new_data["arms"]) == 1
    assert new_data["arms"][0]["role"] == "spike"


def test_cmd_register_writes_temp_arms_file_and_copies_gguf(tmp_path):
    gguf = tmp_path / "src" / "gemma-4-31b-cad-spike-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"fake gguf bytes")

    store = tmp_path / "store"
    arms_data = _sample_arms_data()
    arms_data["store"] = str(store)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps(arms_data, indent=2) + "\n")

    scratch = tmp_path / "scratch"
    ship.scratch_paths(scratch)["verify_ok"].parent.mkdir(parents=True, exist_ok=True)
    ship.scratch_paths(scratch)["verify_ok"].write_text(json.dumps({"gguf": str(gguf)}) + "\n")

    args = argparse.Namespace(
        scratch=str(scratch), gguf=str(gguf), name="gemma-4-31b-cad-spike",
        adapter="lab/runs/spike1/adapter", store=str(store / "gemma-4-31B-cad-spike"),
        arms_file=str(arms_file), quant_type="Q4_K_M", force=False,
    )
    # The real benchmarks/arms.json must come back byte-identical: the point is that
    # cmd_register touched only the --arms-file it was given. (Asserting the spike arm is
    # ABSENT from it was wrong once b9cc26b legitimately registered that arm, and made the
    # branch fail its own suite -- final review, finding 1.)
    real_before = ship.ARMS_FILE.read_bytes()
    ship.cmd_register(args)
    assert ship.ARMS_FILE.read_bytes() == real_before

    dest = store / "gemma-4-31B-cad-spike" / "gemma-4-31b-cad-spike-Q4_K_M.gguf"
    assert dest.read_bytes() == b"fake gguf bytes"

    new_data = json.loads(arms_file.read_text())
    names = [a["name"] for a in new_data["arms"]]
    assert "gemma-4-31b-cad-spike" in names
    new_arm = next(a for a in new_data["arms"] if a["name"] == "gemma-4-31b-cad-spike")
    assert new_arm["gguf"] == "gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf"


def test_cmd_register_refuses_duplicate_without_force(tmp_path):
    gguf = tmp_path / "g.gguf"; gguf.write_bytes(b"x")
    store = tmp_path / "store"
    arms_data = _sample_arms_data()
    arms_data["store"] = str(store)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps(arms_data, indent=2) + "\n")

    scratch = tmp_path / "scratch"
    ship.scratch_paths(scratch)["verify_ok"].parent.mkdir(parents=True, exist_ok=True)
    ship.scratch_paths(scratch)["verify_ok"].write_text(json.dumps({"gguf": str(gguf)}) + "\n")

    args = argparse.Namespace(
        scratch=str(scratch), gguf=str(gguf), name="gemma-4-31b",  # collides
        adapter=None, store=str(store), arms_file=str(arms_file), quant_type="Q4_K_M", force=False,
    )
    with pytest.raises(SystemExit, match="already exists"):
        ship.cmd_register(args)


def test_cmd_register_refuses_without_a_passing_verify(tmp_path):
    """finding 24: register publishes into the shared arms.json, so it demands the same
    passing verify `clean` does."""
    gguf = tmp_path / "g.gguf"; gguf.write_bytes(b"x")
    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), gguf=str(gguf), name="gemma-4-31b-cad-spike",
        adapter=None, store=str(tmp_path / "store"), arms_file=str(tmp_path / "arms.json"),
        quant_type="Q4_K_M", force=False,
    )
    with pytest.raises(SystemExit, match="verify"):
        ship.cmd_register(args)


def test_check_verify_ok_for_warns_on_a_different_gguf(tmp_path, capsys):
    paths = ship.scratch_paths(tmp_path)
    paths["verify_ok"].write_text(json.dumps({"gguf": "/store/some-other-model.gguf"}) + "\n")
    ship.check_verify_ok_for(tmp_path, Path("/store/this-one.gguf"), force=False)
    assert "WARNING" in capsys.readouterr().out


def test_check_verify_ok_for_force_allows_a_missing_marker(tmp_path, capsys):
    ship.check_verify_ok_for(tmp_path, Path("/store/x.gguf"), force=True)
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_cmd_clean_refuses_without_verify_ok(tmp_path):
    args = argparse.Namespace(scratch=str(tmp_path))
    with pytest.raises(SystemExit, match="verify"):
        ship.cmd_clean(args)


def test_cmd_clean_deletes_targets_and_reports_reclaimed_gb(tmp_path):
    paths = ship.scratch_paths(tmp_path)
    paths["verify_ok"].write_text("{}")
    paths["bf16_base"].mkdir()
    (paths["bf16_base"] / "shard.bin").write_bytes(b"a" * (1024 * 1024))  # 1 MiB
    paths["merged"].mkdir()
    (paths["merged"] / "model.safetensors").write_bytes(b"b" * (2 * 1024 * 1024))  # 2 MiB
    paths["f16_gguf"].write_bytes(b"c" * (1024 * 1024))  # 1 MiB

    args = argparse.Namespace(scratch=str(tmp_path), include_base=False)
    ship.cmd_clean(args)

    # finding 6: the bf16 base is the ledger's keep-it decision, so a default clean leaves it.
    assert paths["bf16_base"].exists()
    assert not paths["merged"].exists()
    assert not paths["f16_gguf"].exists()
    assert paths["verify_ok"].exists()  # clean never removes the marker itself

    log_lines = ship.scratch_paths(tmp_path)["log"].read_text().splitlines()
    row = json.loads(log_lines[-1])
    assert row["step"] == "clean"
    expected_gb = (2 + 1) * 1024 * 1024 / (1024**3)  # 3 MiB reclaimed, base untouched
    assert row["reclaimed_gb"] == pytest.approx(round(expected_gb, 2))


def test_cmd_clean_include_base_deletes_the_bf16_base(tmp_path):
    paths = ship.scratch_paths(tmp_path)
    paths["verify_ok"].write_text("{}")
    paths["bf16_base"].mkdir()
    (paths["bf16_base"] / "shard.bin").write_bytes(b"a" * (1024 * 1024))

    ship.cmd_clean(argparse.Namespace(scratch=str(tmp_path), include_base=True))

    assert not paths["bf16_base"].exists()


def test_cmd_clean_handles_already_missing_targets(tmp_path):
    paths = ship.scratch_paths(tmp_path)
    paths["verify_ok"].write_text("{}")
    args = argparse.Namespace(scratch=str(tmp_path), include_base=True)
    ship.cmd_clean(args)  # must not raise even though bf16_base/merged/f16_gguf never existed


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_build_argparser_merge_requires_adapter():
    p = ship.build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args(["merge"])
    args = p.parse_args(["merge", "--adapter", "lab/runs/spike1/adapter"])
    assert args.adapter == "lab/runs/spike1/adapter"
    assert args.base is None and args.out is None and args.force is False


def test_build_argparser_quantize_defaults():
    p = ship.build_argparser()
    args = p.parse_args(["quantize"])
    assert args.type == "Q4_K_M"
    assert args.f16 is None and args.imatrix is None and args.out is None


def test_build_argparser_verify_requires_gguf_and_has_defaults():
    p = ship.build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args(["verify"])
    args = p.parse_args(["verify", "--gguf", "/g/model.gguf"])
    assert args.mmproj == str(ship.STOCK_MMPROJ)
    assert args.port == 8093
    assert args.i_know_the_gpu_is_free is False


def test_build_argparser_register_defaults():
    p = ship.build_argparser()
    args = p.parse_args(["register", "--gguf", "/g/model.gguf"])
    assert args.name == "gemma-4-31b-cad-spike"
    assert args.store == str(ship.DEFAULT_STORE_DIR)
    assert args.arms_file == str(ship.ARMS_FILE)


def test_build_argparser_scratch_default():
    p = ship.build_argparser()
    args = p.parse_args(["clean"])
    assert args.scratch == str(ship.DEFAULT_SCRATCH)


def test_build_argparser_all_requires_adapter():
    p = ship.build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args(["all"])
    args = p.parse_args(["all", "--adapter", "lab/runs/spike1/adapter"])
    assert args.quant_type == "Q4_K_M"


def test_dispatch_table_covers_every_subcommand():
    p = ship.build_argparser()
    sub_actions = [a for a in p._subparsers._group_actions if hasattr(a, "choices")]
    subcommands = set(sub_actions[0].choices.keys())
    assert subcommands == set(ship._DISPATCH.keys())


# ---------------------------------------------------------------------------
# fix round 3: disk precheck (finding 7), adapter validation (8), GPU window (5),
# template canary (10), GGUF model name (23)
# ---------------------------------------------------------------------------


def test_check_disk_free_passes_when_there_is_room(tmp_path):
    free = ship.check_disk_free(tmp_path / "not-created-yet" / "out.gguf", 1, "convert")
    assert free > 0


def test_check_disk_free_refuses_and_names_the_step_and_both_sizes(tmp_path):
    with pytest.raises(SystemExit, match="quantize"):
        ship.check_disk_free(tmp_path / "out.gguf", 10 ** 18, "quantize")


def test_existing_ancestor_walks_up_to_a_real_directory(tmp_path):
    assert ship._existing_ancestor(tmp_path / "a" / "b" / "c.gguf") == tmp_path.resolve()


def test_validate_adapter_config_accepts_the_real_spike_config():
    # The verified spike adapter_config.json shape: empty patterns, no dora, no extra modules.
    ship.validate_adapter_config({
        "r": 16, "lora_alpha": 16, "rank_pattern": {}, "alpha_pattern": {},
        "modules_to_save": None, "use_dora": False, "lora_bias": False,
    })


@pytest.mark.parametrize("key,value", [
    ("rank_pattern", {"layers.0.mlp": 32}),
    ("alpha_pattern", {"layers.0.mlp": 64}),
    ("modules_to_save", ["lm_head"]),
    ("use_dora", True),
    ("lora_bias", True),
    ("trainable_token_indices", [1, 2, 3]),
    ("fan_in_fan_out", True),
])
def test_validate_adapter_config_raises_on_unimplemented_features(key, value):
    with pytest.raises(ValueError):
        ship.validate_adapter_config({"r": 16, "lora_alpha": 16, key: value})


def test_assert_all_adapter_tensors_consumed_passes_on_clean_pairs():
    tensors = {
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": object(),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": object(),
    }
    pairs = ship.build_lora_pairs(tensors)
    ship.assert_all_adapter_tensors_consumed(tensors, pairs)


def test_assert_all_adapter_tensors_consumed_raises_and_lists_leftovers():
    tensors = {
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": object(),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": object(),
        "base_model.model.model.embed_tokens.lora_embedding_A": object(),
    }
    pairs = ship.build_lora_pairs(tensors)
    with pytest.raises(ValueError, match="lora_embedding_A"):
        ship.assert_all_adapter_tensors_consumed(tensors, pairs)


def test_in_gpu_window_reads_the_marker_env():
    assert ship.in_gpu_window({"CAD_GPU_WINDOW": "1"}) is True
    assert ship.in_gpu_window({}) is False


def test_require_gpu_window_refuses_outside_a_window(monkeypatch):
    monkeypatch.delenv("CAD_GPU_WINDOW", raising=False)
    with pytest.raises(SystemExit, match="gpu_window"):
        ship.require_gpu_window(argparse.Namespace(i_know_the_gpu_is_free=False))


def test_require_gpu_window_allows_inside_a_window(monkeypatch):
    monkeypatch.setenv("CAD_GPU_WINDOW", "1")
    ship.require_gpu_window(argparse.Namespace(i_know_the_gpu_is_free=False))


def test_require_gpu_window_allows_the_explicit_manual_override(monkeypatch):
    monkeypatch.delenv("CAD_GPU_WINDOW", raising=False)
    ship.require_gpu_window(argparse.Namespace(i_know_the_gpu_is_free=True))


def test_cmd_verify_refuses_before_starting_a_server(monkeypatch, tmp_path):
    monkeypatch.delenv("CAD_GPU_WINDOW", raising=False)

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("verify started a server outside a GPU window")

    monkeypatch.setattr(ship.subprocess, "Popen", boom)
    args = argparse.Namespace(scratch=str(tmp_path), gguf="/g/m.gguf", mmproj="/g/mm.gguf",
                              port=8093, i_know_the_gpu_is_free=False)
    with pytest.raises(SystemExit, match="GPU window"):
        ship.cmd_verify(args)


_TEMPLATE_A = (
    "{% if messages and messages[0]['role'] == 'system' %}"
    "<|turn>system\n{{ messages[0]['content'] | trim }}<turn|>\n"
    "{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}<|turn>{{ message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)
_TEMPLATE_B = _TEMPLATE_A.replace("<|turn>model", "<|start>model")


def test_check_template_match_passes_on_identical_framing(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "chat_template.jinja").write_text(_TEMPLATE_A)
    train = tmp_path / "train_chat_template.jinja"
    train.write_text("{# a different file, same rendering #}" + _TEMPLATE_A)
    rendered = ship.check_template_match(base, train)
    assert rendered.endswith("<|turn>model\n")


def test_check_template_match_refuses_on_a_serving_framing_that_differs(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "chat_template.jinja").write_text(_TEMPLATE_B)
    train = tmp_path / "train_chat_template.jinja"
    train.write_text(_TEMPLATE_A)
    with pytest.raises(SystemExit, match="chat template mismatch"):
        ship.check_template_match(base, train)


def test_check_template_match_refuses_when_the_base_has_no_template(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    train = tmp_path / "t.jinja"; train.write_text(_TEMPLATE_A)
    with pytest.raises(SystemExit, match="chat_template.jinja"):
        ship.check_template_match(base, train)


def test_training_template_path_prefers_the_adapters_own(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "chat_template.jinja").write_text(_TEMPLATE_A)
    assert ship.training_template_path(adapter) == adapter / "chat_template.jinja"


def test_training_template_path_falls_back_to_lab_data_default(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir()
    from lab.data import DEFAULT_TEMPLATE
    assert ship.training_template_path(adapter) == Path(DEFAULT_TEMPLATE)


def test_default_model_name_strips_the_precision_tail():
    assert ship.default_model_name(Path("/s/gemma-4-31b-cad-F16.gguf")) == "gemma-4-31b-cad"
    assert ship.default_model_name(Path("/s/gemma-4-31b-cad-F16.gguf.partial")) == "gemma-4-31b-cad"
    assert ship.default_model_name(
        Path("/s/gemma-4-31b-cad-spike-Q4_K_M.gguf")) == "gemma-4-31b-cad-spike"


def test_convert_argv_passes_model_name_when_given():
    argv = ship.convert_argv("PY", Path("/m"), Path("/o/out.gguf"), "gemma-4-31b-cad-spike")
    assert argv[-2:] == ["--model-name", "gemma-4-31b-cad-spike"]


def test_cmd_merge_refuses_when_the_serving_template_differs_from_training(tmp_path, monkeypatch):
    """finding 10: the merged dir (and therefore the GGUF) takes its chat template from the
    bf16 BASE, while the training data was rendered with the checkpoint's. Nothing checked
    that the two frame a conversation the same way."""
    import torch

    torch.manual_seed(2)
    w_target = torch.randn(8, 4, dtype=torch.float32)
    base_dir = _make_base_checkpoint(
        tmp_path, w_target, torch.randn(6), torch.randn(3, 5), TARGET_KEY,
    )
    (base_dir / "chat_template.jinja").write_text(_TEMPLATE_B)
    adapter_dir = _make_adapter(
        tmp_path, TARGET_KEY, torch.randn(2, 4), torch.randn(8, 2), r=2, alpha=4)
    (adapter_dir / "chat_template.jinja").write_text(_TEMPLATE_A)

    monkeypatch.setattr(ship, "check_mem_available", lambda: 50.0)
    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), force=False, adapter=str(adapter_dir),
        base=str(base_dir), out=str(tmp_path / "merged"),
        train_template=None, skip_template_check=False,
    )
    with pytest.raises(SystemExit, match="chat template mismatch"):
        ship.cmd_merge(args)
    assert not (tmp_path / "merged" / "model.safetensors.index.json").exists()

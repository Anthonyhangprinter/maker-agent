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


def test_fetch_imatrix_argv():
    argv = ship.fetch_imatrix_argv(Path("/scratch/imatrix"))
    assert argv == [
        "hf", "download", "unsloth/gemma-4-31B-it-GGUF", "imatrix_unsloth.gguf_file",
        "--local-dir", "/scratch/imatrix",
    ]


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


def test_copy_aux_files_copies_matching_patterns_and_skips_existing(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    merged = tmp_path / "merged"; merged.mkdir()
    (base / "config.json").write_text('{"base": true}')
    (base / "generation_config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    (base / "tokenizer_config.json").write_text("{}")
    (base / "chat_template.jinja").write_text("{{ bos_token }}")
    (base / "preprocessor_config.json").write_text("{}")
    (base / "model.safetensors").write_text("not aux")  # must NOT be copied
    # merged already has its own config.json (as if save_pretrained wrote it) -- must survive
    (merged / "config.json").write_text('{"merged": true}')

    copied = ship.copy_aux_files(base, merged)

    assert "config.json" not in copied  # already present in merged, left alone
    assert json.loads((merged / "config.json").read_text()) == {"merged": True}
    assert set(copied) == {
        "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "preprocessor_config.json",
    }
    assert not (merged / "model.safetensors").exists()


def test_copy_aux_files_missing_optional_files_are_fine(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "config.json").write_text("{}")
    merged = tmp_path / "merged"
    copied = ship.copy_aux_files(base, merged)
    assert copied == ["config.json"]


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
# idempotency: merge/convert/fetch-imatrix/quantize skip when their output exists
# ---------------------------------------------------------------------------


def test_cmd_merge_skips_when_out_config_exists(tmp_path, monkeypatch):
    out_dir = tmp_path / "merged"; out_dir.mkdir()
    (out_dir / "config.json").write_text("{}")

    def boom(*a, **kw):
        raise AssertionError("merge_and_save must not run when the output already exists")

    monkeypatch.setattr(ship, "merge_and_save", boom)
    monkeypatch.setattr(ship, "check_mem_available", boom)
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, adapter=str(tmp_path / "nonexistent-adapter"),
        base=None, out=str(out_dir),
    )
    ship.cmd_merge(args)  # must not raise


def test_cmd_convert_skips_when_out_file_exists(tmp_path, monkeypatch):
    out_file = tmp_path / "out.gguf"; out_file.write_text("x")

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the output already exists")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, merged=str(tmp_path), out=str(out_file), python=None)
    ship.cmd_convert(args)  # must not raise


def test_cmd_fetch_imatrix_skips_when_dest_exists(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"; out_dir.mkdir()
    (out_dir / "imatrix_unsloth.gguf_file").write_text("x")

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the output already exists")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)  # must not raise


def test_cmd_quantize_skips_when_out_file_exists(tmp_path, monkeypatch):
    out_file = tmp_path / "q.gguf"; out_file.write_text("x")

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called when the output already exists")

    monkeypatch.setattr(ship.subprocess, "run", boom)
    args = argparse.Namespace(
        scratch=str(tmp_path), force=False, f16=str(tmp_path / "f16.gguf"),
        imatrix=str(tmp_path / "im.gguf"), out=str(out_file), type="Q4_K_M",
    )
    ship.cmd_quantize(args)  # must not raise


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


def test_cmd_convert_calls_subprocess_with_the_right_argv(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    out_file = tmp_path / "out.gguf"
    calls = []
    monkeypatch.setattr(ship.subprocess, "run", lambda argv, check: calls.append(argv))
    args = argparse.Namespace(scratch=str(tmp_path), force=False, merged=str(merged), out=str(out_file), python="FAKE_PY")
    ship.cmd_convert(args)
    assert calls == [ship.convert_argv("FAKE_PY", merged, out_file)]
    log_lines = ship.scratch_paths(tmp_path)["log"].read_text().splitlines()
    assert json.loads(log_lines[-1])["step"] == "convert"


def test_cmd_quantize_calls_subprocess_with_the_right_argv(tmp_path, monkeypatch):
    f16 = tmp_path / "f16.gguf"; f16.write_text("x")
    imatrix = tmp_path / "im.gguf"; imatrix.write_text("x")
    out_file = tmp_path / "out.gguf"
    calls = []
    monkeypatch.setattr(ship.subprocess, "run", lambda argv, check: calls.append(argv))
    args = argparse.Namespace(scratch=str(tmp_path), force=False, f16=str(f16), imatrix=str(imatrix), out=str(out_file), type="Q4_K_M")
    ship.cmd_quantize(args)
    assert calls == [ship.quantize_argv(imatrix, f16, out_file, "Q4_K_M")]


def test_cmd_fetch_imatrix_calls_subprocess_with_the_right_argv(tmp_path, monkeypatch):
    out_dir = tmp_path / "imatrix"
    calls = []
    monkeypatch.setattr(ship.subprocess, "run", lambda argv, check: calls.append(argv))
    args = argparse.Namespace(scratch=str(tmp_path), force=False, out=str(out_dir))
    ship.cmd_fetch_imatrix(args)
    assert calls == [ship.fetch_imatrix_argv(out_dir)]


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

    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), gguf=str(gguf), name="gemma-4-31b-cad-spike",
        adapter="lab/runs/spike1/adapter", store=str(store / "gemma-4-31B-cad-spike"),
        arms_file=str(arms_file), quant_type="Q4_K_M", force=False,
    )
    ship.cmd_register(args)

    dest = store / "gemma-4-31B-cad-spike" / "gemma-4-31b-cad-spike-Q4_K_M.gguf"
    assert dest.read_bytes() == b"fake gguf bytes"

    new_data = json.loads(arms_file.read_text())
    names = [a["name"] for a in new_data["arms"]]
    assert "gemma-4-31b-cad-spike" in names
    new_arm = next(a for a in new_data["arms"] if a["name"] == "gemma-4-31b-cad-spike")
    assert new_arm["gguf"] == "gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf"

    # the REAL benchmarks/arms.json must never be touched by this test
    real = json.loads(ship.ARMS_FILE.read_text())
    assert "gemma-4-31b-cad-spike" not in [a["name"] for a in real["arms"]]


def test_cmd_register_refuses_duplicate_without_force(tmp_path):
    gguf = tmp_path / "g.gguf"; gguf.write_bytes(b"x")
    store = tmp_path / "store"
    arms_data = _sample_arms_data()
    arms_data["store"] = str(store)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps(arms_data, indent=2) + "\n")

    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), gguf=str(gguf), name="gemma-4-31b",  # collides
        adapter=None, store=str(store), arms_file=str(arms_file), quant_type="Q4_K_M", force=False,
    )
    with pytest.raises(SystemExit, match="already exists"):
        ship.cmd_register(args)


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

    args = argparse.Namespace(scratch=str(tmp_path))
    ship.cmd_clean(args)

    assert not paths["bf16_base"].exists()
    assert not paths["merged"].exists()
    assert not paths["f16_gguf"].exists()
    assert paths["verify_ok"].exists()  # clean never removes the marker itself

    log_lines = ship.scratch_paths(tmp_path)["log"].read_text().splitlines()
    row = json.loads(log_lines[-1])
    assert row["step"] == "clean"
    expected_gb = (1 + 2 + 1) * 1024 * 1024 / (1024**3)  # 4 MiB reclaimed
    assert row["reclaimed_gb"] == pytest.approx(round(expected_gb, 2))


def test_cmd_clean_handles_already_missing_targets(tmp_path):
    paths = ship.scratch_paths(tmp_path)
    paths["verify_ok"].write_text("{}")
    args = argparse.Namespace(scratch=str(tmp_path))
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

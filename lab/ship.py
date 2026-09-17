#!/usr/bin/env python3
"""lab/ship.py -- turn a trained LoRA adapter (Task 3's lab/runs/<run>/adapter/) into a
llama.cpp GGUF the maker server (scripts/arms.py) can serve.

One subcommand per pipeline stage, each idempotent (skips when its output already exists
unless --force) and each printing the argv/action it ran plus its elapsed time, and (merge/
convert/fetch-imatrix/quantize/verify/register/clean) appending one JSON line to
<scratch>/ship_log.jsonl:

  merge         PEFT-merge the adapter into the bf16 base on CPU, save the merged HF dir.
  convert       llama.cpp convert_hf_to_gguf.py: merged HF dir -> F16 GGUF.
  fetch-imatrix download the Unsloth importance matrix used to quantize the stock arm.
  quantize      llama-quantize --imatrix: F16 GGUF -> Q4_K_M GGUF.
  verify        throwaway llama-server + three chat-completions smoke prompts.
  register      copy the Q4_K_M GGUF into the permanent store and add an arm to
                benchmarks/arms.json.
  clean         delete the scratch bf16 base / merged dir / F16 GGUF -- refuses unless a
                passing `verify` has already written <scratch>/verify.ok.
  all           merge -> convert -> fetch-imatrix -> quantize, in that order. Deliberately
                stops there: verify needs a human to look at the GPU run, register touches
                the shared arms.json, and clean is destructive -- none of those three should
                ever fire unattended as part of a chained "all".

Paths (all overridable per-subcommand flag):
  scratch root:     ~/lab-scratch                          (root SSD, temporary)
  permanent store:  /mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/
  bf16 base:        <scratch>/gemma-4-31B-it-bf16/          (google/gemma-4-31B-it, ~62.5GB;
                     downloaded separately -- merge does NOT trigger that download itself,
                     it only checks the directory is there and fails with the download
                     command if it isn't)
  merged HF dir:    <scratch>/merged/
  F16 GGUF:         <scratch>/gemma-4-31b-cad-F16.gguf
  imatrix:          <scratch>/imatrix/imatrix_unsloth.gguf_file
                     (repo unsloth/gemma-4-31B-it-GGUF -- the filename was verified against
                     the live repo listing on 2026-09-17: it really is spelled
                     "imatrix_unsloth.gguf_file", including the literal ".gguf_file"
                     extension, not a typo for ".gguf")
  Q4_K_M GGUF:      /mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike/gemma-4-31b-cad-spike-Q4_K_M.gguf
                     (plain Q4_K_M built with an Unsloth imatrix -- NOT the same recipe as
                     the stock arm's gemma-4-31B-it-UD-Q4_K_XL.gguf, which is Unsloth's own
                     dynamic UD quant; register() notes this difference on the arm entry so
                     a card comparison between the two is never read as apples-to-apples)

Converter interpreter: verified 2026-09-17 that convert_hf_to_gguf.py's own deps (torch,
sentencepiece, plus its own gguf-py, which the script inserts onto sys.path itself) import
cleanly under lab/.venv/bin/python but NOT under the system python3 (missing sentencepiece).
pick_converter_python() re-checks this at run time rather than trusting that one-time
finding, in case either environment changes before this is actually run.

Heavy imports (torch, transformers, peft) live inside merge_and_save(), never at module
scope, so this module and every other function in it stay importable -- and unit-testable --
without the training venv on the machine running the plain repo test suite. Nothing in this
file talks to the GPU except verify's llama-server subprocess (started/stopped by the PID
this module itself spawned, never by a pkill pattern); merge/convert/quantize/fetch-imatrix
are CPU/network-only.

This module writes code only -- Task 4's brief is explicit that the heavy steps (merge,
convert, quantize, verify) are executed later by the controller once the adapter and the
bf16 base are actually on disk. Nothing in this dispatch ran any of them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parents[1]

DEFAULT_SCRATCH = Path.home() / "lab-scratch"
DEFAULT_STORE_DIR = Path("/mnt/nvme-apps/LinuxModels/gemma-4-31B-cad-spike")
STOCK_MMPROJ = Path("/mnt/nvme-apps/LinuxModels/gemma-4-31B-it-GGUF/mmproj-F16.gguf")
ARMS_FILE = HERE / "benchmarks" / "arms.json"

LLAMA_CPP_SRC = Path.home() / "llama.cpp-cuda-src"
CONVERT_SCRIPT = LLAMA_CPP_SRC / "convert_hf_to_gguf.py"
GGUF_PY_DIR = LLAMA_CPP_SRC / "gguf-py"
LLAMA_QUANTIZE_BIN = LLAMA_CPP_SRC / "build" / "bin" / "llama-quantize"
LLAMA_SERVER_BIN = LLAMA_CPP_SRC / "build" / "bin" / "llama-server"

SYSTEM_PYTHON = "python3"
LAB_VENV_PYTHON = HERE / "lab" / ".venv" / "bin" / "python"

IMATRIX_REPO = "unsloth/gemma-4-31B-it-GGUF"
# Verified against the live repo file listing 2026-09-17 -- this really is the exact name.
IMATRIX_FILENAME = "imatrix_unsloth.gguf_file"

DEFAULT_QUANT_TYPE = "Q4_K_M"
DEFAULT_ARM_NAME = "gemma-4-31b-cad-spike"
BASE_ARM_FOR_CLONE = "gemma-4-31b"

MIN_MEM_AVAILABLE_GB = 40.0

# Copied from the bf16 base dir into the merged dir after save_pretrained(); a file already
# present in the merged dir (e.g. config.json, which the model's own save_pretrained writes)
# is left alone -- copy_aux_files() never clobbers an existing destination file.
AUX_FILE_GLOBS = (
    "config.json", "generation_config.json", "tokenizer*", "chat_template.jinja",
    "processor_config.json", "preprocessor_config.json",
)

# (label, prompt, needs_build123d) -- the three verify smoke prompts. The first two must come
# back containing "from build123d import"; the third (text-only) only needs a non-empty reply.
VERIFY_PROMPTS: tuple[tuple[str, str, bool], ...] = (
    ("cube", "Write build123d Python code for a 20mm cube.", True),
    ("bracket", "Write build123d Python code for an L bracket 60x40x5mm with two 5mm through holes.", True),
    ("text", "Reply with the single word OK.", False),
)


# ---------------------------------------------------------------------------
# Paths and logging
# ---------------------------------------------------------------------------


def scratch_paths(scratch: Path) -> dict[str, Path]:
    """Every derived scratch-relative path in one place, so merge/convert/quantize/clean
    never disagree about where an intermediate artifact lives."""
    scratch = Path(scratch)
    return {
        "bf16_base": scratch / "gemma-4-31B-it-bf16",
        "merged": scratch / "merged",
        "f16_gguf": scratch / "gemma-4-31b-cad-F16.gguf",
        "imatrix_dir": scratch / "imatrix",
        "imatrix_file": scratch / "imatrix" / IMATRIX_FILENAME,
        "log": scratch / "ship_log.jsonl",
        "verify_ok": scratch / "verify.ok",
    }


def log_step(scratch: Path, step: str, status: str, elapsed_s: float, **extra: Any) -> None:
    """Append one JSON line to <scratch>/ship_log.jsonl. Callers ensure `scratch` exists
    (main() creates it once before dispatching to any subcommand)."""
    row = {"step": step, "status": status, "elapsed_s": round(elapsed_s, 3), "ts": time.time(), **extra}
    path = scratch_paths(scratch)["log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _should_skip(output_path: Path, force: bool, label: str) -> bool:
    if output_path.exists() and not force:
        print(f"[ship] {label}: {output_path} already exists, skipping (--force to redo)")
        return True
    return False


# ---------------------------------------------------------------------------
# MemAvailable guard
# ---------------------------------------------------------------------------


def read_mem_available_kb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    """Parse MemAvailable (kB) out of /proc/meminfo. Raises ValueError if the field is
    missing -- a kernel without MemAvailable (very old) must fail loudly, never be silently
    treated as "plenty free"."""
    text = Path(meminfo_path).read_text()
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            return int(parts[1])
    raise ValueError(f"MemAvailable not found in {meminfo_path}")


def check_mem_available(
    min_gb: float = MIN_MEM_AVAILABLE_GB, meminfo_path: Path = Path("/proc/meminfo")
) -> float:
    """Refuse (SystemExit) unless at least min_gb of RAM is free right now. Returns the
    measured GB so the caller can log it.

    The CPU bf16 merge loads the full ~62.5GB base (low_cpu_mem_usage lowers the peak but
    does not eliminate it) plus the merged copy transiently; 40GB is a floor that catches
    "something else is already holding memory on this box", not a tight sizing estimate.
    """
    kb = read_mem_available_kb(meminfo_path)
    gb = kb / (1024 * 1024)
    if gb < min_gb:
        raise SystemExit(
            f"MemAvailable is {gb:.1f}GB, below the {min_gb:.0f}GB floor for a CPU bf16 "
            f"merge; free up RAM (stop other heavy processes) before retrying"
        )
    return gb


# ---------------------------------------------------------------------------
# argv builders (pure -- no subprocess call here, just the list construction)
# ---------------------------------------------------------------------------


def convert_argv(python_bin: str, merged_dir: Path, out_file: Path) -> list[str]:
    return [
        str(python_bin), str(CONVERT_SCRIPT), str(merged_dir),
        "--outfile", str(out_file), "--outtype", "f16",
    ]


def quantize_argv(
    imatrix_file: Path, f16_file: Path, out_file: Path, qtype: str = DEFAULT_QUANT_TYPE
) -> list[str]:
    return [
        str(LLAMA_QUANTIZE_BIN), "--imatrix", str(imatrix_file), str(f16_file), str(out_file), qtype,
    ]


def fetch_imatrix_argv(out_dir: Path) -> list[str]:
    return ["hf", "download", IMATRIX_REPO, IMATRIX_FILENAME, "--local-dir", str(out_dir)]


def verify_server_argv(gguf_file: Path, mmproj_file: Path, port: int) -> list[str]:
    return [
        str(LLAMA_SERVER_BIN),
        "-m", str(gguf_file),
        "--mmproj", str(mmproj_file),
        "-ngl", "99",
        "-c", "8192",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]


# ---------------------------------------------------------------------------
# Converter interpreter selection
# ---------------------------------------------------------------------------


def _can_import_convert_deps(python_bin: str) -> bool:
    """Runtime check: can this interpreter import everything convert_hf_to_gguf.py needs
    (torch, sentencepiece, and its own gguf-py, inserted onto sys.path the same way the
    script inserts it onto its own)? Never raises -- a missing interpreter or an import
    crash both just mean "no"."""
    code = f"import sys; sys.path.insert(1, {str(GGUF_PY_DIR)!r}); import gguf, torch, sentencepiece"
    try:
        result = subprocess.run([python_bin, "-c", code], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def pick_converter_python(checker: Callable[[str], bool] = _can_import_convert_deps) -> str:
    """System python3 first, lab venv as fallback. Verified 2026-09-17: system python3
    lacks sentencepiece, so this currently always picks the lab venv -- re-checked at run
    time (not hardcoded to that finding) so a system python3 upgrade is picked up
    automatically. Raises SystemExit if neither interpreter has the deps."""
    for candidate in (SYSTEM_PYTHON, str(LAB_VENV_PYTHON)):
        if checker(candidate):
            return candidate
    raise SystemExit(
        "neither system python3 nor lab/.venv/bin/python can import torch+sentencepiece+gguf; "
        "install the missing deps in one of them before running `ship.py convert`"
    )


# ---------------------------------------------------------------------------
# merge (heavy step -- torch/transformers/peft imports live inside merge_and_save only)
# ---------------------------------------------------------------------------


def copy_aux_files(
    base_dir: Path, merged_dir: Path, patterns: tuple[str, ...] = AUX_FILE_GLOBS
) -> list[str]:
    """Copy tokenizer/processor/config files from the bf16 base dir into the merged dir,
    skipping any filename the merge's own save_pretrained() already wrote there (never
    clobbers an existing destination file, e.g. the merged model's own config.json). Returns
    the list of filenames actually copied."""
    copied = []
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in sorted(Path(base_dir).glob(pattern)):
            if not src.is_file():
                continue
            dst = merged_dir / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            copied.append(src.name)
    return copied


def _load_base_model(base_dir: Path, torch_module: Any) -> Any:
    """Load the bf16 base for merging. Gemma 4 is a vision-capable ForConditionalGeneration
    architecture (Task 3 loaded the 4-bit checkpoint through Unsloth's FastModel wrapper,
    which hides this); transformers exposes conditional-generation multimodal heads under
    AutoModelForImageTextToText in current versions, so try that first and fall back to
    AutoModelForCausalLM for older transformers or a text-only config export. This path has
    not been exercised against the real checkpoint in this dispatch (no heavy deps are
    supposed to run here) -- confirm on the first real merge."""
    import transformers

    kwargs = dict(low_cpu_mem_usage=True, device_map={"": "cpu"}, torch_dtype=torch_module.bfloat16)
    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(str(base_dir), **kwargs)
        except (ValueError, OSError):
            pass
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(base_dir), **kwargs)


def merge_and_save(adapter_dir: Path, base_dir: Path, out_dir: Path) -> list[str]:
    """The actual CPU bf16 PEFT merge. Heavy imports live here, not at module scope, so this
    module stays importable -- and every other function in it unit-testable -- without
    torch/transformers/peft installed. Not exercised by the offline test suite (there is no
    way to unit-test a real 62.5GB merge in CI); the pieces it is built from (copy_aux_files,
    check_mem_available, scratch_paths) are tested directly instead."""
    import torch
    from peft import PeftModel

    base = _load_base_model(base_dir, torch)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    return copy_aux_files(base_dir, out_dir)


def cmd_merge(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    base_dir = Path(args.base) if args.base else paths["bf16_base"]
    out_dir = Path(args.out) if args.out else paths["merged"]
    adapter_dir = Path(args.adapter)

    if _should_skip(out_dir / "config.json", args.force, "merge"):
        return
    if not adapter_dir.exists():
        raise SystemExit(f"adapter dir not found: {adapter_dir}")
    if not base_dir.exists():
        raise SystemExit(
            f"bf16 base not found at {base_dir}; download it first "
            f"(hf download google/gemma-4-31B-it --local-dir {base_dir}) -- merge does not "
            f"trigger that ~62.5GB download itself"
        )
    mem_gb = check_mem_available()

    adapter_config_path = adapter_dir / "adapter_config.json"
    base_ref = ""
    if adapter_config_path.exists():
        base_ref = json.loads(adapter_config_path.read_text()).get("base_model_name_or_path", "")
    is_4bit_ref = "4bit" in base_ref.lower() or "bnb" in base_ref.lower()

    print(
        f"[ship] merge: adapter={adapter_dir} base={base_dir} out={out_dir} "
        f"mem_available={mem_gb:.1f}GB adapter_base_ref={base_ref!r} "
        f"(loading the bf16 base regardless -- LoRA weights are dtype-independent)"
    )
    t0 = time.time()
    copied = merge_and_save(adapter_dir, base_dir, out_dir)
    elapsed = time.time() - t0
    print(f"[ship] merge: done in {elapsed:.1f}s, copied aux files: {copied}")
    log_step(
        scratch, "merge", "ok", elapsed,
        adapter=str(adapter_dir), base=str(base_dir), out=str(out_dir),
        base_model_name_or_path=base_ref, adapter_base_is_4bit=is_4bit_ref,
        mem_available_gb=round(mem_gb, 1), copied_aux_files=copied,
    )


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    merged_dir = Path(args.merged) if args.merged else paths["merged"]
    out_file = Path(args.out) if args.out else paths["f16_gguf"]

    if _should_skip(out_file, args.force, "convert"):
        return
    if not merged_dir.exists():
        raise SystemExit(f"merged HF dir not found: {merged_dir}; run `ship.py merge` first")

    python_bin = args.python or pick_converter_python()
    argv = convert_argv(python_bin, merged_dir, out_file)
    print(f"[ship] convert: {' '.join(argv)}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(argv, check=True)
    elapsed = time.time() - t0
    print(f"[ship] convert: done in {elapsed:.1f}s -> {out_file}")
    log_step(scratch, "convert", "ok", elapsed, python=python_bin, merged=str(merged_dir), out=str(out_file))


# ---------------------------------------------------------------------------
# fetch-imatrix
# ---------------------------------------------------------------------------


def cmd_fetch_imatrix(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    out_dir = Path(args.out) if args.out else paths["imatrix_dir"]
    dest = out_dir / IMATRIX_FILENAME

    if _should_skip(dest, args.force, "fetch-imatrix"):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = fetch_imatrix_argv(out_dir)
    print(f"[ship] fetch-imatrix: {' '.join(argv)}")
    t0 = time.time()
    subprocess.run(argv, check=True)
    elapsed = time.time() - t0
    print(f"[ship] fetch-imatrix: done in {elapsed:.1f}s -> {dest}")
    log_step(scratch, "fetch-imatrix", "ok", elapsed, out=str(dest))


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------


def cmd_quantize(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    f16_file = Path(args.f16) if args.f16 else paths["f16_gguf"]
    imatrix_file = Path(args.imatrix) if args.imatrix else paths["imatrix_file"]
    out_file = (
        Path(args.out) if args.out
        else DEFAULT_STORE_DIR / f"{DEFAULT_ARM_NAME}-{args.type}.gguf"
    )

    if _should_skip(out_file, args.force, "quantize"):
        return
    if not f16_file.exists():
        raise SystemExit(f"F16 GGUF not found: {f16_file}; run `ship.py convert` first")
    if not imatrix_file.exists():
        raise SystemExit(f"imatrix not found: {imatrix_file}; run `ship.py fetch-imatrix` first")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    argv = quantize_argv(imatrix_file, f16_file, out_file, args.type)
    print(f"[ship] quantize: {' '.join(argv)}")
    t0 = time.time()
    subprocess.run(argv, check=True)
    elapsed = time.time() - t0
    print(f"[ship] quantize: done in {elapsed:.1f}s -> {out_file}")
    log_step(
        scratch, "quantize", "ok", elapsed,
        f16=str(f16_file), imatrix=str(imatrix_file), out=str(out_file), type=args.type,
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def build_chat_request(prompt: str, alias: str = DEFAULT_ARM_NAME) -> dict:
    return {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }


def check_verify_reply(label: str, reply: str, needs_build123d: bool) -> None:
    """Raises SystemExit (not a bare assert -- this is a real acceptance check on a live
    model reply, not a debug aid `python -O` should be able to strip) when a verify prompt's
    reply fails its check."""
    if not reply or not reply.strip():
        raise SystemExit(f"verify: {label!r} prompt got an empty reply")
    if needs_build123d and "from build123d import" not in reply:
        raise SystemExit(f"verify: {label!r} prompt's reply did not contain 'from build123d import'")


def _tokens_per_second(usage: dict, timing: dict) -> float | None:
    """llama-server's OpenAI-compatible endpoint reports token counts in `usage` and, when
    `--verbose`/timings are enabled, a `timings` block; prefer the server's own
    predicted_per_second when present, else derive completion_tokens / predicted time.
    Returns None rather than guessing when neither is available."""
    if "predicted_per_second" in timing:
        return timing["predicted_per_second"]
    completion_tokens = usage.get("completion_tokens")
    predicted_ms = timing.get("predicted_ms")
    if completion_tokens and predicted_ms:
        return completion_tokens / (predicted_ms / 1000.0)
    return None


def _wait_health(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                json.loads(r.read())
                return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"{url} not healthy after {timeout}s")


def _post_chat(url: str, payload: dict, timeout: int = 300) -> tuple[str, dict, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    reply = body["choices"][0]["message"]["content"]
    return reply, body.get("usage", {}), body.get("timings", {})


def cmd_verify(args: argparse.Namespace) -> None:
    """Starts a throwaway llama-server on args.port, waits for /health, sends the three
    VERIFY_PROMPTS, and kills the server by the PID this call spawned (never a pkill
    pattern) whether the checks pass or not. Intentionally does NOT skip on a pre-existing
    verify.ok -- a stale pass must never stand in for testing the actual current GGUF, so
    every invocation runs the full check fresh and verify.ok is overwritten on the next
    success."""
    scratch = Path(args.scratch)
    gguf = Path(args.gguf)
    mmproj = Path(args.mmproj)
    port = args.port

    argv = verify_server_argv(gguf, mmproj, port)
    print(f"[ship] verify: starting {' '.join(argv)}")
    t0 = time.time()
    proc = subprocess.Popen(argv)
    results: dict[str, dict] = {}
    try:
        _wait_health(f"http://127.0.0.1:{port}/health", timeout=900)
        for label, prompt, needs_build123d in VERIFY_PROMPTS:
            reply, usage, timing = _post_chat(
                f"http://127.0.0.1:{port}/v1/chat/completions", build_chat_request(prompt)
            )
            check_verify_reply(label, reply, needs_build123d)
            toks_per_s = _tokens_per_second(usage, timing)
            results[label] = {"reply_chars": len(reply), "tokens_per_s": toks_per_s}
            print(f"[ship] verify: {label} ok" + (f" ({toks_per_s:.1f} tok/s)" if toks_per_s else ""))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    elapsed = time.time() - t0
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_paths(scratch)["verify_ok"].write_text(
        json.dumps({"ts": time.time(), "gguf": str(gguf), "results": results}) + "\n"
    )
    log_step(scratch, "verify", "ok", elapsed, gguf=str(gguf), mmproj=str(mmproj), port=port, results=results)
    print(f"[ship] verify: PASSED in {elapsed:.1f}s, wrote {scratch_paths(scratch)['verify_ok']}")


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def build_spike_arm(
    arms_data: dict, name: str, adapter_path: str, gguf_rel_path: str,
    base_arm_name: str = BASE_ARM_FOR_CLONE, quant_type: str = DEFAULT_QUANT_TYPE,
) -> dict:
    """Clone the base_arm_name entry (gemma-4-31b) into a new arm dict for the trained
    spike: same alias-shaped ctx/extra_args/mmproj (the adapter never touched vision, so the
    stock mmproj is reused unchanged), name/alias set to `name`, role "spike", gguf pointed
    at the spike's own store-relative path, and a notes string flagging the quant-recipe
    difference from the stock arm (plain Q4_K_M + Unsloth imatrix, not Unsloth's own UD
    dynamic quant) plus the adapter it was built from.

    Raises StopIteration (via next()) if base_arm_name is not in arms_data -- arms.json
    changed shape underneath this."""
    base = next(a for a in arms_data["arms"] if a["name"] == base_arm_name)
    return {
        "name": name,
        "alias": name,
        "role": "spike",
        "gguf": gguf_rel_path,
        "mmproj": base["mmproj"],
        "ctx": base["ctx"],
        "extra_args": base.get("extra_args", ""),
        "hf": None,
        "notes": (
            f"Phase 2 QLoRA training spike: {base_arm_name}'s base model with a LoRA "
            f"adapter merged in, quantized plain {quant_type} with the Unsloth importance "
            f"matrix (NOT the stock arm's Unsloth UD dynamic quant -- a different recipe, "
            f"note it when comparing card results). Adapter: {adapter_path or 'unrecorded'}."
        ),
    }


def register_arm(arms_data: dict, arm: dict, force: bool = False) -> dict:
    """Insert/replace `arm` in a copy of arms_data, refusing a duplicate name unless
    force=True. Returns a new dict; never mutates the input."""
    arms_data = json.loads(json.dumps(arms_data))
    names = [a["name"] for a in arms_data["arms"]]
    if arm["name"] in names and not force:
        raise SystemExit(f"arm {arm['name']!r} already exists in arms.json; pass --force to replace it")
    arms_data["arms"] = [a for a in arms_data["arms"] if a["name"] != arm["name"]]
    arms_data["arms"].append(arm)
    return arms_data


def cmd_register(args: argparse.Namespace) -> None:
    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"gguf not found: {gguf}")

    store_dir = Path(args.store)
    store_dir.mkdir(parents=True, exist_ok=True)
    dest = store_dir / gguf.name
    t0 = time.time()
    if dest.exists() and not args.force:
        print(f"[ship] register: {dest} already exists, skipping copy (--force to redo)")
    else:
        print(f"[ship] register: copying {gguf} -> {dest}")
        shutil.copy2(gguf, dest)
    copy_elapsed = time.time() - t0

    arms_file = Path(args.arms_file)
    data = json.loads(arms_file.read_text())
    rel = str(dest.relative_to(Path(data["store"])))
    arm = build_spike_arm(data, args.name, args.adapter or "", rel, quant_type=args.quant_type)
    new_data = register_arm(data, arm, force=args.force)
    arms_file.write_text(json.dumps(new_data, indent=2) + "\n")
    print(f"[ship] register: arm {args.name!r} added to {arms_file}")

    log_step(
        Path(args.scratch), "register", "ok", copy_elapsed,
        gguf=str(gguf), dest=str(dest), arms_file=str(arms_file), name=args.name,
    )


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cmd_clean(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    if not paths["verify_ok"].exists():
        raise SystemExit(
            f"{paths['verify_ok']} not found; run `ship.py verify` successfully first -- "
            f"clean refuses to delete the merge inputs/outputs before a passing verify"
        )

    targets = [paths["bf16_base"], paths["merged"], paths["f16_gguf"]]
    reclaimed = 0
    for target in targets:
        reclaimed += dir_size_bytes(target)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()

    reclaimed_gb = reclaimed / (1024**3)
    print(f"[ship] clean: reclaimed {reclaimed_gb:.1f}GB ({', '.join(str(t) for t in targets)})")
    log_step(scratch, "clean", "ok", 0.0, reclaimed_gb=round(reclaimed_gb, 2),
             targets=[str(t) for t in targets])


# ---------------------------------------------------------------------------
# all (merge -> convert -> fetch-imatrix -> quantize)
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    quant_out = (
        Path(args.quant_out) if args.quant_out
        else DEFAULT_STORE_DIR / f"{DEFAULT_ARM_NAME}-{args.quant_type}.gguf"
    )

    cmd_merge(argparse.Namespace(
        scratch=args.scratch, force=args.force,
        adapter=args.adapter, base=args.base, out=str(paths["merged"]),
    ))
    cmd_convert(argparse.Namespace(
        scratch=args.scratch, force=args.force,
        merged=str(paths["merged"]), out=str(paths["f16_gguf"]), python=args.python,
    ))
    cmd_fetch_imatrix(argparse.Namespace(
        scratch=args.scratch, force=args.force, out=str(paths["imatrix_dir"]),
    ))
    cmd_quantize(argparse.Namespace(
        scratch=args.scratch, force=args.force,
        f16=str(paths["f16_gguf"]), imatrix=str(paths["imatrix_file"]),
        out=str(quant_out), type=args.quant_type,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", default=str(DEFAULT_SCRATCH), help="scratch root (default: ~/lab-scratch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge")
    m.add_argument("--adapter", required=True, help="lab/runs/<run>/adapter dir (Task 3 output)")
    m.add_argument("--base", default=None, help="bf16 base dir (default: <scratch>/gemma-4-31B-it-bf16)")
    m.add_argument("--out", default=None, help="merged HF dir out (default: <scratch>/merged)")
    m.add_argument("--force", action="store_true")

    c = sub.add_parser("convert")
    c.add_argument("--merged", default=None, help="merged HF dir (default: <scratch>/merged)")
    c.add_argument("--out", default=None, help="F16 GGUF out (default: <scratch>/gemma-4-31b-cad-F16.gguf)")
    c.add_argument("--python", default=None, help="override the auto-picked converter interpreter")
    c.add_argument("--force", action="store_true")

    fi = sub.add_parser("fetch-imatrix")
    fi.add_argument("--out", default=None, help="download dir (default: <scratch>/imatrix)")
    fi.add_argument("--force", action="store_true")

    q = sub.add_parser("quantize")
    q.add_argument("--f16", default=None, help="F16 GGUF in (default: <scratch>/gemma-4-31b-cad-F16.gguf)")
    q.add_argument("--imatrix", default=None, help="imatrix file (default: <scratch>/imatrix/...)")
    q.add_argument("--out", default=None, help="quantized GGUF out (default: the permanent store)")
    q.add_argument("--type", default=DEFAULT_QUANT_TYPE)
    q.add_argument("--force", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("--gguf", required=True)
    v.add_argument("--mmproj", default=str(STOCK_MMPROJ))
    v.add_argument("--port", type=int, default=8093)

    r = sub.add_parser("register")
    r.add_argument("--gguf", required=True)
    r.add_argument("--name", default=DEFAULT_ARM_NAME)
    r.add_argument("--adapter", default=None, help="adapter path recorded in the arm's notes")
    r.add_argument("--store", default=str(DEFAULT_STORE_DIR))
    r.add_argument("--arms-file", default=str(ARMS_FILE))
    r.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    r.add_argument("--force", action="store_true")

    sub.add_parser("clean")

    a = sub.add_parser("all")
    a.add_argument("--adapter", required=True)
    a.add_argument("--base", default=None)
    a.add_argument("--python", default=None)
    a.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    a.add_argument("--quant-out", default=None)
    a.add_argument("--force", action="store_true")

    return p


_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "merge": cmd_merge,
    "convert": cmd_convert,
    "fetch-imatrix": cmd_fetch_imatrix,
    "quantize": cmd_quantize,
    "verify": cmd_verify,
    "register": cmd_register,
    "clean": cmd_clean,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    _DISPATCH[args.cmd](args)


if __name__ == "__main__":
    main()

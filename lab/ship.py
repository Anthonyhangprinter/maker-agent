#!/usr/bin/env python3
"""lab/ship.py -- turn a trained LoRA adapter (Task 3's lab/runs/<run>/adapter/) into a
llama.cpp GGUF the maker server (scripts/arms.py) can serve.

One subcommand per pipeline stage, each idempotent (skips when its output already exists
unless --force) and each printing the argv/action it ran plus its elapsed time, and (merge/
convert/fetch-imatrix/quantize/verify/register/clean) appending one JSON line to
<scratch>/ship_log.jsonl:

  merge         streaming per-tensor LoRA merge of the adapter into the bf16 base, save the
                merged HF dir (fix round 2: no full-model load -- see streaming_merge()).
                Refuses an adapter using features this merge does not implement, and refuses
                when the bf16 base's chat template renders the canary conversation
                differently from the template the training data was rendered with
                (--skip-template-check overrides).
  convert       llama.cpp convert_hf_to_gguf.py: merged HF dir -> F16 GGUF, with
                --model-name so the GGUF advertises the model, not the scratch dir name.
  fetch-imatrix download the Unsloth importance matrix used to quantize the stock arm.
  quantize      llama-quantize --imatrix: F16 GGUF -> Q4_K_M GGUF.
  verify        throwaway llama-server + three chat-completions smoke prompts. GPU step:
                refuses to run outside lab/gpu_window.sh (CAD_GPU_WINDOW=1) unless
                --i-know-the-gpu-is-free, and stops its server on SIGTERM/SIGINT too.
  register      copy the Q4_K_M GGUF into the permanent store and add an arm to
                benchmarks/arms.json -- refuses without a passing <scratch>/verify.ok
                unless --force.
  clean         delete the scratch merged dir / F16 GGUF -- refuses unless a passing
                `verify` has already written <scratch>/verify.ok. The bf16 base is KEPT
                unless --include-base (the ledger's ruling: re-downloading it costs about
                80 minutes per later phase).
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

Heavy imports (torch, safetensors) live inside streaming_merge()/load_adapter(), never at
module scope, so this module and every other function in it stay importable -- and
unit-testable -- without the training venv on the machine running the plain repo test suite.
Fix round 2 removed transformers and peft from this module entirely: the earlier merge
loaded the whole ~62.5GB bf16 checkpoint through transformers+PEFT, which does not fit this
box's 47GB MemAvailable; the streaming per-tensor merge (see streaming_merge()) reads and
writes one base tensor and one output shard buffer at a time instead, targeting well under
12GB peak RSS. Nothing in this file talks to the GPU except verify's llama-server subprocess
(started/stopped by the PID this module itself spawned, never by a pkill pattern);
merge/convert/quantize/fetch-imatrix are CPU/network-only. merge, convert and quantize each
refuse up front when the filesystem their output lands on does not have the free space the
step needs (fix round 3: RAM was guarded, disk was not, and a full disk fails late and
expensively).

Idempotency is completeness-based, not existence-based (fix round 1, 2026-09-17): merge/
convert/fetch-imatrix/quantize each write a marker only after their real work has already
succeeded (a `.ship_done` file inside the merged dir for merge's directory output; a sibling
`<file>.done` for convert/fetch-imatrix/quantize's single-file outputs), and `_should_skip()`
checks that marker, never bare path existence -- a path that exists because a previous run
was killed mid-write must never be mistaken for "done". convert and quantize additionally
write to a `<out>.partial` name and atomically `Path.replace()` it onto the real output path
only after their subprocess exits 0, so the final filename itself never names a truncated
file. The `.done` marker for a file output also records `size_bytes`, and a mismatch against
the file's current size on disk (e.g. something replaced the file out from under ship.py
after a genuine completion) is treated the same as "not done".

This module writes code only -- Task 4's brief is explicit that the heavy steps (merge,
convert, quantize, verify) are executed later by the controller once the adapter and the
bf16 base are actually on disk. Nothing in this dispatch ran any of them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
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

# Pinned per fix round 1: every other subprocess binary in this module is a Path.home()
# -anchored constant (LLAMA_QUANTIZE_BIN, LLAMA_SERVER_BIN, CONVERT_SCRIPT); `hf` was the one
# bare-PATH-lookup exception. Verified present on this box 2026-09-17.
HF_BIN = Path.home() / ".local" / "bin" / "hf"

IMATRIX_REPO = "unsloth/gemma-4-31B-it-GGUF"
# Verified against the live repo file listing 2026-09-17 -- this really is the exact name.
IMATRIX_FILENAME = "imatrix_unsloth.gguf_file"

DEFAULT_QUANT_TYPE = "Q4_K_M"
DEFAULT_ARM_NAME = "gemma-4-31b-cad-spike"
BASE_ARM_FOR_CLONE = "gemma-4-31b"

# Fix round 2: the streaming per-tensor merge peaks well under 12GB (one base tensor plus
# one output shard buffer), never the full 62.5GB checkpoint -- 40GB was sized for the
# now-removed full transformers+PEFT load and would refuse to run on this box's real 47GB
# MemAvailable for no reason. 8GB is a floor that still catches "something else already
# ate the RAM", not a tight sizing estimate.
MIN_MEM_AVAILABLE_GB = 8.0

# Free-space headroom each heavy step demands on the FILESYSTEM ITS OUTPUT LANDS ON, as a
# multiple of its own input size (fix round 3, finding 7: RAM was guarded, disk was not, and
# this session ran the root SSD down to 28GB free). merge writes a copy of the base
# checkpoint (same bytes plus the small aux files), convert writes an F16 GGUF of about the
# same size as the merged dir, quantize writes a Q4_K_M of about a third of the F16. The
# factors are deliberately loose: this is a "you will run out" precheck, not a sizing model.
MERGE_DISK_FACTOR = 1.05
CONVERT_DISK_FACTOR = 1.05
QUANTIZE_DISK_FACTOR = 0.35

# verify starts a real llama-server on the whole card. gpu_window.sh exports CAD_GPU_WINDOW=1
# for its child, which is the only evidence this module has that the resident/maker were
# evicted and the build lock is held (fix round 3, finding 5).
GPU_WINDOW_ENV = "CAD_GPU_WINDOW"

# adapter_config.json keys this merge does NOT implement. A non-empty/true value means the
# adapter carries trained state or a per-layer scale that base_key_for_adapter_key() and
# lora_scale() would silently drop or get wrong (fix round 3, finding 8).
UNSUPPORTED_ADAPTER_CONFIG_KEYS = ("rank_pattern", "alpha_pattern", "modules_to_save",
                                   "use_dora", "lora_bias", "trainable_token_indices")

# The canary conversation rendered through both chat templates in cmd_merge (finding 10): a
# system+user turn with add_generation_prompt=True and thinking off, i.e. exactly the shape
# lab/data.py rendered every training row in.
CANARY_MESSAGES = (
    {"role": "system", "content": "You are a build123d expert."},
    {"role": "user", "content": "Write build123d Python code for a 20mm cube."},
)

# copy_aux_files() copies every top-level regular file from the bf16 base dir into the merged
# dir EXCEPT these -- weight artifacts (already written by save_pretrained, or irrelevant to
# a merged checkpoint) and repo plumbing. This is an exclude-list, not an allow-list (fix
# round 1: an allow-list of tokenizer/config filenames silently missed
# special_tokens_map.json and added_tokens.json), so a checkpoint shipping one more small
# config/tokenizer file than expected still gets it copied.
AUX_EXCLUDE_SUFFIXES = (".safetensors", ".safetensors.index.json", ".gguf", ".bin", ".pt")
AUX_EXCLUDE_NAMES = frozenset({".gitattributes", "README.md"})

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


def marker_path_for_dir(out_dir: Path) -> Path:
    """A directory-shaped output's completeness marker lives INSIDE it, so deleting the
    directory (e.g. `clean`) removes the marker along with everything else -- there is never
    a marker pointing at a directory that no longer exists."""
    return Path(out_dir) / ".ship_done"


def marker_path_for_file(out_file: Path) -> Path:
    """A file-shaped output's completeness marker is a sibling `<file>.done`."""
    out_file = Path(out_file)
    return out_file.with_name(out_file.name + ".done")


def write_dir_done_marker(out_dir: Path, step: str, elapsed_s: float) -> Path:
    """Written only after the caller's real work (merge_and_save, including its aux-file
    copy) has already returned successfully. size_bytes is informational here (the total
    merged-dir size) -- unlike a file output, a directory's completeness is judged by the
    marker's mere existence, not a size comparison (see file_output_is_complete)."""
    marker = marker_path_for_dir(out_dir)
    marker.write_text(json.dumps({
        "step": step, "elapsed_s": round(elapsed_s, 3),
        "size_bytes": dir_size_bytes(out_dir), "ts": time.time(),
    }) + "\n")
    return marker


def write_file_done_marker(out_file: Path, step: str, elapsed_s: float) -> Path:
    """Written only after the caller's real work has already produced the final `out_file`
    (for convert/quantize, that means AFTER the `<out>.partial` -> `out_file` atomic
    replace). Records the file's real size so a later run can tell "done" from "something
    else replaced this file since"."""
    out_file = Path(out_file)
    marker = marker_path_for_file(out_file)
    marker.write_text(json.dumps({
        "step": step, "elapsed_s": round(elapsed_s, 3),
        "size_bytes": out_file.stat().st_size, "ts": time.time(),
    }) + "\n")
    return marker


def dir_output_is_complete(out_dir: Path) -> bool:
    """A directory output is "done" purely on its marker's existence -- unlike a file, there
    is no single size to compare against, and the marker is written last (after every file
    merge_and_save produces), so its existence already proves the whole directory landed."""
    return marker_path_for_dir(out_dir).exists()


def file_output_is_complete(out_file: Path) -> bool:
    """True only when BOTH the marker exists and its recorded size_bytes equals the file's
    current size on disk -- catches "never finished" (no marker, e.g. a killed run left a
    real but truncated/empty file at the target path) and "finished, then the file changed
    underneath ship.py" (size mismatch) as both "not done"."""
    out_file = Path(out_file)
    marker = marker_path_for_file(out_file)
    if not marker.exists() or not out_file.exists():
        return False
    try:
        recorded = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return recorded.get("size_bytes") == out_file.stat().st_size


def _should_skip(output_path: Path, force: bool, label: str, *, is_dir: bool = False) -> bool:
    """Idempotency check: skip only when the step's own completeness marker says the output
    is done, never on bare path existence -- a partial/interrupted run can leave a real file
    or directory at the target path with nothing actually finished inside it (fix round 1)."""
    output_path = Path(output_path)
    complete = dir_output_is_complete(output_path) if is_dir else file_output_is_complete(output_path)
    if complete and not force:
        marker = marker_path_for_dir(output_path) if is_dir else marker_path_for_file(output_path)
        print(f"[ship] {label}: {output_path} already complete ({marker}), skipping (--force to redo)")
        return True
    return False


def _hf_bin() -> str:
    """Prefer the pinned ~/.local/bin/hf (verified present 2026-09-17, same convention as
    every other subprocess binary this module hardcodes); fall back to a bare "hf" PATH
    lookup so this keeps working on a box where it only lives somewhere else (e.g.
    lab/.venv/bin/hf, or a future reinstall)."""
    return str(HF_BIN) if HF_BIN.exists() else "hf"


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

    Fix round 2: the streaming per-tensor merge (see streaming_merge()) never holds more than
    one base tensor plus one output shard buffer (target under 12GB peak, even against the
    real 62.5GB base), so the floor dropped from 40GB (sized for the removed full
    transformers+PEFT load) to 8GB -- still a floor that catches "something else is already
    holding memory on this box", not a tight sizing estimate.
    """
    kb = read_mem_available_kb(meminfo_path)
    gb = kb / (1024 * 1024)
    if gb < min_gb:
        raise SystemExit(
            f"MemAvailable is {gb:.1f}GB, below the {min_gb:.0f}GB floor for the merge; "
            f"free up RAM (stop other heavy processes) before retrying"
        )
    return gb


# ---------------------------------------------------------------------------
# Disk precheck (fix round 3, finding 7)
# ---------------------------------------------------------------------------


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing directory at or above `path` -- shutil.disk_usage needs a path
    that exists, and an output path usually does not yet."""
    path = Path(path).resolve()
    while not path.exists():
        parent = path.parent
        if parent == path:
            return path
        path = parent
    return path


def check_disk_free(out_path: Path, required_bytes: int, label: str) -> int:
    """Refuse (SystemExit) unless the filesystem `out_path` lands on has `required_bytes`
    free right now. Returns the measured free bytes so the caller can log/print them.

    The step is named in the message along with both numbers, because the whole point is
    that a full disk should fail in one second at the start instead of after the 30 minutes
    it takes to write most of a 62GB file."""
    free = shutil.disk_usage(_existing_ancestor(out_path)).free
    if free < required_bytes:
        raise SystemExit(
            f"{label}: needs about {required_bytes / 1024**3:.1f}GB free on the filesystem "
            f"holding {out_path}, but only {free / 1024**3:.1f}GB is free; free space (or "
            f"point --out at another filesystem) before retrying"
        )
    return free


# ---------------------------------------------------------------------------
# argv builders (pure -- no subprocess call here, just the list construction)
# ---------------------------------------------------------------------------


def default_model_name(out_file: Path) -> str:
    """The GGUF's `general.name`, derived from the output file stem with the quant/precision
    suffix dropped: "gemma-4-31b-cad-F16.gguf" -> "gemma-4-31b-cad" (fix round 3, finding 23
    -- without --model-name the converter names the model after the merged scratch DIRECTORY,
    which is how the shipped spike GGUF ended up advertising itself as "Merged")."""
    stem = Path(out_file).name
    for suffix in (".partial", ".gguf"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for tail in ("-F16", "-f16", "-BF16", "-Q4_K_M", "-Q5_K_M", "-Q8_0"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    return stem


def convert_argv(
    python_bin: str, merged_dir: Path, out_file: Path, model_name: str | None = None
) -> list[str]:
    argv = [
        str(python_bin), str(CONVERT_SCRIPT), str(merged_dir),
        "--outfile", str(out_file), "--outtype", "f16",
    ]
    if model_name:
        argv += ["--model-name", model_name]
    return argv


def quantize_argv(
    imatrix_file: Path, f16_file: Path, out_file: Path, qtype: str = DEFAULT_QUANT_TYPE
) -> list[str]:
    return [
        str(LLAMA_QUANTIZE_BIN), "--imatrix", str(imatrix_file), str(f16_file), str(out_file), qtype,
    ]


def fetch_imatrix_argv(out_dir: Path) -> list[str]:
    return [_hf_bin(), "download", IMATRIX_REPO, IMATRIX_FILENAME, "--local-dir", str(out_dir)]


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
# merge (fix round 2: streaming per-tensor LoRA merge -- torch/safetensors imports live
# inside streaming_merge/load_adapter only, and neither transformers nor peft is imported
# anywhere in this module any more)
# ---------------------------------------------------------------------------


def copy_aux_files(
    base_dir: Path, merged_dir: Path,
    exclude_suffixes: tuple[str, ...] = AUX_EXCLUDE_SUFFIXES,
    exclude_names: frozenset[str] = AUX_EXCLUDE_NAMES,
) -> list[str]:
    """Copy EVERY top-level regular file from the bf16 base dir into the merged dir except
    weight artifacts and repo plumbing (config.json, generation_config.json, tokenizer.json,
    tokenizer_config.json, tokenizer.model, special_tokens_map.json, added_tokens.json,
    chat_template.jinja, processor_config.json, preprocessor_config.json, and anything else
    HF ships alongside a checkpoint all come across this way -- fix round 1: the previous
    allow-list of filename patterns silently missed special_tokens_map.json/
    added_tokens.json).

    Fix round 2: the streaming merge never calls save_pretrained() (there is no HF model
    object to save), so these files now come from the base dir with NOTHING already written
    at the destination -- copy_aux_files() still never overwrites an existing destination
    file (harmless now, defensive against a future caller that pre-populates the merged dir).
    Returns the list of filenames actually copied.
    """
    copied = []
    base_dir = Path(base_dir)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(base_dir.iterdir()):
        if not src.is_file():
            continue
        name = src.name
        if name in exclude_names or any(name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        dst = merged_dir / name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


# Verified 2026-09-17 against lab/runs/smoke4/adapter/adapter_model.safetensors (820
# tensors, fp32, 0 unmapped): a PEFT LoRA key looks like
# "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight" and its base
# target is "model.language_model.layers.0.mlp.down_proj.weight" -- strip the
# "base_model.model." prefix, then replace the ".lora_A.weight"/".lora_B.weight" suffix with
# plain ".weight".
ADAPTER_KEY_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

# safetensors reports dtype as one of these short strings via PySafeSlice.get_dtype(); sizes
# in bytes per element. Used only to SIZE tensors (for shard planning) without materializing
# them -- see plan_output_shards().
SAFETENSORS_DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# Flush an output shard once its buffered bytes exceed this. Matches the brief's 5GB target;
# a single tensor larger than this still lands whole in its own shard (same convention HF's
# own sharded save uses -- the cap bounds normal packing, not any one tensor).
MAX_SHARD_BYTES = 5 * 1024 ** 3


def base_key_for_adapter_key(adapter_key: str) -> str | None:
    """Map a PEFT adapter tensor key to the base model tensor it targets, or None if the key
    is not a lora_A/lora_B weight (kept permissive rather than raising -- the verified real
    adapter has 0 such keys, but a future PEFT version adding e.g. an embedding delta should
    not crash the mapping step, only fail later if it turns out to matter)."""
    if not adapter_key.startswith(ADAPTER_KEY_PREFIX):
        return None
    stripped = adapter_key[len(ADAPTER_KEY_PREFIX):]
    for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)] + ".weight"
    return None


def validate_adapter_config(config: dict) -> None:
    """Raise ValueError on any adapter feature this merge does not implement (fix round 3,
    finding 8). base_key_for_adapter_key() only maps lora_A/lora_B weights and lora_scale()
    only reads the GLOBAL r/lora_alpha/use_rslora, so an adapter carrying per-layer ranks or
    alphas (rank_pattern/alpha_pattern), fully trained extra modules (modules_to_save,
    trainable_token_indices), a DoRA magnitude vector (use_dora) or trained LoRA biases
    (lora_bias) would merge SILENTLY WRONG -- wrong scale on some layers, or trained weights
    dropped outright. The verified spike adapter sets none of them; this is the guard for the
    Phase 3/4 reuse of the same path.

    fan_in_fan_out is checked here too (it was previously checked inline in streaming_merge):
    this key mapping assumes PEFT's default weight orientation, not the transposed one.
    """
    if config.get("fan_in_fan_out"):
        raise ValueError("fan_in_fan_out=True adapters are not supported by this streaming merge")
    unsupported = [k for k in UNSUPPORTED_ADAPTER_CONFIG_KEYS if config.get(k)]
    if unsupported:
        raise ValueError(
            f"adapter_config.json sets {', '.join(unsupported)}, which this merge does not "
            f"implement (it merges global-scale lora_A/lora_B pairs only); merge with PEFT's "
            f"own merge_and_unload() instead, or extend streaming_merge() first"
        )


def assert_all_adapter_tensors_consumed(tensors: dict, lora_pairs: dict) -> None:
    """Every tensor in the adapter file must belong to a merged A/B pair (fix round 3,
    finding 8). A leftover key means the adapter carries trained state this merge is about to
    throw away, so it raises and names the leftovers rather than merging a partial adapter.
    Verified against the real spike adapter: 820 tensors, 410 pairs, 0 leftovers."""
    used = {key for pair in lora_pairs.values() for key in pair}
    leftover = sorted(set(tensors) - used)
    if leftover:
        raise ValueError(
            f"{len(leftover)} adapter tensor(s) are not part of a lora_A/lora_B pair and "
            f"would be dropped by this merge: {leftover[:5]}"
            + (f" (+{len(leftover) - 5} more)" if len(leftover) > 5 else "")
        )


def load_adapter(adapter_dir: Path) -> tuple[dict, dict]:
    """Load every adapter tensor (small -- LoRA A/B matrices only, never the base model) plus
    adapter_config.json. Returns (tensors, config). Heavy import (safetensors) lives here,
    not at module scope."""
    from safetensors.torch import load_file

    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    return tensors, config


def lora_scale(config: dict) -> float:
    """alpha / r by default; alpha / sqrt(r) when the adapter was trained with rslora
    (`use_rslora: true` in adapter_config.json) -- PEFT's own two scaling conventions.
    lora_dropout has no effect at merge time (it only ever applied during training) and is
    not read here."""
    r = config["r"]
    alpha = config.get("lora_alpha", r)
    if config.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def build_lora_pairs(tensors: dict) -> dict[str, tuple[str, str]]:
    """{base_key: (lora_A_key, lora_B_key)} from a loaded adapter tensor dict. Raises
    ValueError if any base key has an A without a matching B or vice versa -- a real PEFT
    adapter never does this; if it happens, the mapping in base_key_for_adapter_key() has
    drifted from the actual key format and merging would silently apply half a delta."""
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for key in tensors:
        base_key = base_key_for_adapter_key(key)
        if base_key is None:
            continue
        if key.endswith(LORA_A_SUFFIX):
            a_keys[base_key] = key
        elif key.endswith(LORA_B_SUFFIX):
            b_keys[base_key] = key
    missing_b = set(a_keys) - set(b_keys)
    missing_a = set(b_keys) - set(a_keys)
    if missing_b or missing_a:
        raise ValueError(
            f"unpaired LoRA keys: {len(missing_b)} A-without-B, {len(missing_a)} B-without-A"
        )
    return {base_key: (a_key, b_keys[base_key]) for base_key, a_key in a_keys.items()}


def base_model_index(base_dir: Path) -> dict:
    """Read the base checkpoint's model.safetensors.index.json (shard map + total_size).
    Required: the verified real base (google/gemma-4-31B-it bf16, 2 shards, 1188 tensors)
    always ships one for a sharded checkpoint."""
    index_path = Path(base_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(
            f"no model.safetensors.index.json in {base_dir}; expected a sharded safetensors "
            f"checkpoint"
        )
    return json.loads(index_path.read_text())


def shard_files_in_order(index: dict) -> list[str]:
    """Every distinct base shard filename from an index's weight_map, in a stable
    (numerically sorted, since shard names are zero-padded) order."""
    return sorted(set(index["weight_map"].values()))


def plan_output_shards(
    base_dir: Path, index: dict, max_shard_bytes: int = MAX_SHARD_BYTES
) -> tuple[dict[str, int], int, int]:
    """First pass (fix round 2): decide which output shard bucket every base tensor key
    lands in, reading only shape+dtype via safe_open().get_slice() -- never a full tensor.
    Iterates shard files and keys in EXACTLY the order streaming_merge()'s real pass does, so
    the two agree on which key goes in which bucket.

    Returns (key_to_shard_index, n_shards, total_bytes), so the real merge pass can write
    each output shard under its FINAL filename ("model-NNNNN-of-MMMMM.safetensors") the first
    time -- no rename-after step needed."""
    from safetensors import safe_open

    key_to_shard: dict[str, int] = {}
    shard_bytes = [0]
    total_bytes = 0
    for shard_file in shard_files_in_order(index):
        with safe_open(str(Path(base_dir) / shard_file), framework="pt") as f:
            for key in f.keys():
                s = f.get_slice(key)
                nbytes = SAFETENSORS_DTYPE_SIZES[s.get_dtype()]
                for dim in s.get_shape():
                    nbytes *= dim
                if shard_bytes[-1] > 0 and shard_bytes[-1] + nbytes > max_shard_bytes:
                    shard_bytes.append(0)
                key_to_shard[key] = len(shard_bytes) - 1
                shard_bytes[-1] += nbytes
                total_bytes += nbytes
    return key_to_shard, len(shard_bytes), total_bytes


def streaming_merge(adapter_dir: Path, base_dir: Path, out_dir: Path) -> dict:
    """Stream every base tensor through, apply its LoRA delta if one targets it, and write
    straight to new output shards -- never holding more than one base tensor plus one output
    shard buffer in memory (peak target well under 12GB, even against the real 62.5GB base,
    so it can run beside a training job). Replaces the earlier full-model transformers+PEFT
    load (fix round 1), which does not fit this box's 47GB MemAvailable against a 62.5GB bf16
    checkpoint (2 shards, ~31GB each).

    delta = (B @ A).float() * scale, added in fp32 then cast back to the base tensor's own
    dtype -- matches PEFT's own merge math. Raises ValueError on: any adapter_config feature
    this merge does not implement, including fan_in_fan_out=True (see
    validate_adapter_config), an adapter tensor left out of every A/B pair (see
    assert_all_adapter_tensors_consumed), a LoRA target key not present in the base's weight_map, a delta
    shape that does not match its base tensor, or (should never happen if the code above is
    correct) a LoRA pair left unconsumed / a final tensor count or key set that disagrees
    with the base index -- those last three are self-consistency checks on this function's
    own bookkeeping, not normal user-input validation.

    Returns a small summary dict (tensor/pair counts, shard count, bytes, elapsed) for
    logging.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_tensors, adapter_config = load_adapter(adapter_dir)
    validate_adapter_config(adapter_config)
    scale = lora_scale(adapter_config)
    lora_pairs = build_lora_pairs(adapter_tensors)
    assert_all_adapter_tensors_consumed(adapter_tensors, lora_pairs)

    index = base_model_index(base_dir)
    base_keys = set(index["weight_map"].keys())
    unmapped = sorted(set(lora_pairs) - base_keys)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} LoRA target(s) not found in the base model: {unmapped[:5]}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_to_shard, n_shards, total_bytes = plan_output_shards(base_dir, index)

    weight_map: dict[str, str] = {}
    consumed: set[str] = set()
    buffer: dict = {}
    current_shard_idx = 0
    tensor_count = 0

    def flush(idx: int) -> None:
        if not buffer:
            return
        name = f"model-{idx + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(dict(buffer), str(out_dir / name), metadata={"format": "pt"})
        for k in buffer:
            weight_map[k] = name
        buffer.clear()

    t0 = time.time()
    for shard_file in shard_files_in_order(index):
        shard_t0 = time.time()
        shard_tensor_count = 0
        with safe_open(str(Path(base_dir) / shard_file), framework="pt") as f:
            for key in f.keys():
                target_shard = key_to_shard[key]
                if target_shard != current_shard_idx:
                    flush(current_shard_idx)
                    current_shard_idx = target_shard
                w = f.get_tensor(key)
                if key in lora_pairs:
                    a_key, b_key = lora_pairs[key]
                    a = adapter_tensors[a_key]
                    b = adapter_tensors[b_key]
                    delta = (b.float() @ a.float()) * scale
                    if tuple(delta.shape) != tuple(w.shape):
                        raise ValueError(
                            f"LoRA delta shape {tuple(delta.shape)} for {key!r} does not "
                            f"match base tensor shape {tuple(w.shape)}"
                        )
                    w = (w.float() + delta).to(w.dtype)
                    consumed.add(key)
                buffer[key] = w
                tensor_count += 1
                shard_tensor_count += 1
        print(f"[ship] merge: shard {shard_file} -> {shard_tensor_count} tensors in "
              f"{time.time() - shard_t0:.1f}s ({tensor_count} total so far)")
    flush(current_shard_idx)

    if len(consumed) != len(lora_pairs):
        missing = sorted(set(lora_pairs) - consumed)
        raise ValueError(f"{len(missing)} LoRA pair(s) were never applied: {missing[:5]}")
    if tensor_count != len(base_keys):
        raise ValueError(f"merged {tensor_count} tensors but the base index lists {len(base_keys)}")
    if set(weight_map) != base_keys:
        raise ValueError("merged tensor keys do not match the base index's key set")

    (out_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }, indent=2) + "\n")

    elapsed = time.time() - t0
    print(f"[ship] merge: streamed {tensor_count} tensors ({len(consumed)} with a LoRA delta) "
          f"into {n_shards} shard(s) in {elapsed:.1f}s")
    return {
        "tensor_count": tensor_count, "lora_pairs_applied": len(consumed),
        "n_shards": n_shards, "total_bytes": total_bytes, "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# serving-vs-training chat template canary (fix round 3, finding 10)
# ---------------------------------------------------------------------------


def _load_template_renderer(path: Path):
    """lab/data.py's own load_template, imported lazily so this module keeps working (and
    keeps importing) without jinja2 on the box. Deliberately NOT a second copy of the
    renderer: the whole point of the canary is that it renders the way data.py rendered the
    training rows."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from lab.data import load_template

    template, from_checkpoint = load_template(Path(path), allow_fallback=False)
    if not from_checkpoint:  # load_template raises on a missing file; belt and braces
        raise SystemExit(f"chat template not found: {path}")
    return template


def render_canary(template_path: Path) -> str:
    """Render CANARY_MESSAGES through one chat_template.jinja: a system+user conversation
    with add_generation_prompt=True and thinking off, the exact shape lab/data.py rendered
    every training row in."""
    template = _load_template_renderer(template_path)
    return template.render(messages=[dict(m) for m in CANARY_MESSAGES], bos_token="",
                           add_generation_prompt=True, enable_thinking=False)


def training_template_path(adapter_dir: Path) -> Path:
    """The template the TRAINING side used: the adapter's own chat_template.jinja when it
    saved one, else the 4-bit checkpoint lab/data.py renders against (lab.data's
    DEFAULT_TEMPLATE)."""
    adapter_template = Path(adapter_dir) / "chat_template.jinja"
    if adapter_template.exists():
        return adapter_template
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from lab.data import DEFAULT_TEMPLATE

    return Path(DEFAULT_TEMPLATE)


def check_template_match(base_dir: Path, train_template: Path) -> str:
    """Refuse (SystemExit) when the bf16 base's chat_template.jinja -- the one copy_aux_files
    puts in the merged dir, and therefore the one baked into the GGUF and used at serve time
    -- renders the canary conversation differently from the template the training data was
    rendered with.

    The spike shipped fine (the two templates differ only in a header comment and a tool-call
    branch, and render identically for this message shape), but nothing checked it: a future
    base/checkpoint pairing could silently ship a serving framing that disagrees with
    training, which is the one defect no benchmark would explain. Returns the rendered canary
    on a match, for logging."""
    base_template = Path(base_dir) / "chat_template.jinja"
    if not base_template.exists():
        raise SystemExit(
            f"no chat_template.jinja in the bf16 base dir {base_dir}; cannot check that the "
            f"serving framing matches training (pass --skip-template-check to merge anyway)"
        )
    train_template = Path(train_template)
    if not train_template.exists():
        raise SystemExit(
            f"training chat template not found: {train_template}; pass --train-template "
            f"<path> or --skip-template-check"
        )
    serve_render = render_canary(base_template)
    train_render = render_canary(train_template)
    if serve_render != train_render:
        import difflib

        diff = "\n".join(difflib.unified_diff(
            train_render.splitlines(), serve_render.splitlines(),
            fromfile=str(train_template), tofile=str(base_template), lineterm=""))
        raise SystemExit(
            "chat template mismatch: the bf16 base's template (which ends up in the GGUF and "
            "frames every request at serve time) renders the canary conversation differently "
            f"from the template the training data was rendered with.\n{diff}\n"
            "Fix the pairing, or pass --skip-template-check if the difference is understood "
            "and harmless for this data shape."
        )
    return serve_render


def merge_and_save(adapter_dir: Path, base_dir: Path, out_dir: Path) -> list[str]:
    """streaming_merge() + copy_aux_files(). Heavy imports live inside streaming_merge/
    load_adapter, not at module scope, so this module and every pure function in it stay
    importable without torch/safetensors installed."""
    out_dir = Path(out_dir)
    streaming_merge(adapter_dir, base_dir, out_dir)
    return copy_aux_files(base_dir, out_dir)


def cmd_merge(args: argparse.Namespace) -> None:
    scratch = Path(args.scratch)
    paths = scratch_paths(scratch)
    base_dir = Path(args.base) if args.base else paths["bf16_base"]
    out_dir = Path(args.out) if args.out else paths["merged"]
    adapter_dir = Path(args.adapter)

    if _should_skip(out_dir, args.force, "merge", is_dir=True):
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
    base_bytes = dir_size_bytes(base_dir)
    free_bytes = check_disk_free(out_dir, int(base_bytes * MERGE_DISK_FACTOR), "merge")

    if args.skip_template_check:
        print("[ship] merge: WARNING --skip-template-check, the serving template is NOT "
              "checked against the training template")
    else:
        train_template = Path(args.train_template) if args.train_template else training_template_path(adapter_dir)
        check_template_match(base_dir, train_template)
        print(f"[ship] merge: chat template canary ok (serve={base_dir}/chat_template.jinja "
              f"renders identically to train={train_template})")

    print(
        f"[ship] merge: adapter={adapter_dir} base={base_dir} out={out_dir} "
        f"mem_available={mem_gb:.1f}GB disk_free={free_bytes / 1024**3:.1f}GB "
        f"(streaming per-tensor merge -- never loads the full model into memory)"
    )
    t0 = time.time()
    copied = merge_and_save(adapter_dir, base_dir, out_dir)
    elapsed = time.time() - t0
    write_dir_done_marker(out_dir, "merge", elapsed)
    print(f"[ship] merge: done in {elapsed:.1f}s, copied aux files: {copied}")
    log_step(
        scratch, "merge", "ok", elapsed,
        adapter=str(adapter_dir), base=str(base_dir), out=str(out_dir),
        mem_available_gb=round(mem_gb, 1), disk_free_gb=round(free_bytes / 1024**3, 1),
        copied_aux_files=copied,
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
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged_bytes = dir_size_bytes(merged_dir)
    free_bytes = check_disk_free(out_file, int(merged_bytes * CONVERT_DISK_FACTOR), "convert")
    model_name = args.model_name or default_model_name(out_file)
    partial = out_file.with_name(out_file.name + ".partial")
    argv = convert_argv(python_bin, merged_dir, partial, model_name)
    print(f"[ship] convert: {' '.join(argv)}")
    t0 = time.time()
    subprocess.run(argv, check=True)
    partial.replace(out_file)  # atomic: out_file never names a truncated/in-progress file
    elapsed = time.time() - t0
    write_file_done_marker(out_file, "convert", elapsed)
    print(f"[ship] convert: done in {elapsed:.1f}s -> {out_file}")
    log_step(scratch, "convert", "ok", elapsed, python=python_bin, merged=str(merged_dir),
             out=str(out_file), model_name=model_name,
             disk_free_gb=round(free_bytes / 1024**3, 1))


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
    # `hf download` itself writes to a temp name and atomically renames onto `dest` (unlike
    # convert_hf_to_gguf.py/llama-quantize), so no local .partial dance is needed here -- just
    # the completeness marker so a later run's skip check is consistent with the other steps.
    elapsed = time.time() - t0
    write_file_done_marker(dest, "fetch-imatrix", elapsed)
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
    free_bytes = check_disk_free(
        out_file, int(f16_file.stat().st_size * QUANTIZE_DISK_FACTOR), "quantize")
    partial = out_file.with_name(out_file.name + ".partial")
    argv = quantize_argv(imatrix_file, f16_file, partial, args.type)
    print(f"[ship] quantize: {' '.join(argv)}")
    t0 = time.time()
    subprocess.run(argv, check=True)
    partial.replace(out_file)  # atomic: out_file never names a truncated/in-progress file
    elapsed = time.time() - t0
    write_file_done_marker(out_file, "quantize", elapsed)
    print(f"[ship] quantize: done in {elapsed:.1f}s -> {out_file}")
    log_step(
        scratch, "quantize", "ok", elapsed,
        f16=str(f16_file), imatrix=str(imatrix_file), out=str(out_file), type=args.type,
        disk_free_gb=round(free_bytes / 1024**3, 1),
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


def in_gpu_window(env: dict | None = None) -> bool:
    """True when this process is a child of lab/gpu_window.sh, which exports CAD_GPU_WINDOW=1
    for its child only after it has taken the CAD build lock and evicted both the resident and
    the maker arm."""
    env = os.environ if env is None else env
    return env.get(GPU_WINDOW_ENV) == "1"


def require_gpu_window(args: argparse.Namespace) -> None:
    """Refuse (SystemExit) to start a GPU server outside a GPU window (fix round 3, finding
    5). verify loads an 18.7GB GGUF at -ngl 99: run bare while the 23GB resident is up on a
    24GB card, it fights the resident for VRAM and takes no build lock, so a CAD frontend can
    start a build on top of it. --i-know-the-gpu-is-free is the manual escape hatch for a
    human who has already evicted everything by hand."""
    if getattr(args, "i_know_the_gpu_is_free", False) or in_gpu_window():
        return
    raise SystemExit(
        "verify starts a llama-server on the whole GPU, so it must run inside a GPU window: "
        "`lab/gpu_window.sh lab/.venv/bin/python lab/ship.py verify --gguf <path>` (that "
        "holds ~/.openclaw/cad-build.lock, evicts the resident and the maker arm, and exports "
        f"{GPU_WINDOW_ENV}=1). Pass --i-know-the-gpu-is-free only when the GPU is already "
        "free by hand."
    )


def _stop_server(proc: subprocess.Popen) -> None:
    """TERM, then KILL after 30s. Safe to call twice (a finished process just returns)."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _install_server_signal_handlers(proc: subprocess.Popen) -> dict:
    """On SIGTERM/SIGINT, stop the spawned llama-server before dying (fix round 3, finding 5:
    a `finally` alone does not run when the process is signalled, so a TERM to ship.py
    orphaned a server holding the whole card). Returns the previous handlers so the caller can
    restore them."""
    def handler(signum, _frame):
        print(f"[ship] verify: signal {signum}, stopping llama-server pid {proc.pid}")
        _stop_server(proc)
        raise SystemExit(128 + signum)

    return {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}


def cmd_verify(args: argparse.Namespace) -> None:
    """Starts a throwaway llama-server on args.port, waits for /health, sends the three
    VERIFY_PROMPTS, and kills the server by the PID this call spawned (never a pkill
    pattern) whether the checks pass, fail or the process is signalled. Refuses to run at all
    outside a GPU window (see require_gpu_window). Intentionally does NOT skip on a
    pre-existing verify.ok -- a stale pass must never stand in for testing the actual current
    GGUF, so every invocation runs the full check fresh and verify.ok is overwritten on the
    next success."""
    require_gpu_window(args)
    scratch = Path(args.scratch)
    gguf = Path(args.gguf)
    mmproj = Path(args.mmproj)
    port = args.port

    argv = verify_server_argv(gguf, mmproj, port)
    print(f"[ship] verify: starting {' '.join(argv)}")
    t0 = time.time()
    proc = subprocess.Popen(argv)
    previous_handlers = _install_server_signal_handlers(proc)
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
        _stop_server(proc)
        for sig, old in previous_handlers.items():
            signal.signal(sig, old)

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


def check_verify_ok_for(scratch: Path, gguf: Path, force: bool) -> None:
    """register publishes the arm into the shared benchmarks/arms.json that `arms.py use` and
    run_card.py read, so it now demands the same passing verify `clean` does (fix round 3,
    finding 24). A verify.ok naming a DIFFERENT GGUF is a loud warning, not a refusal: the
    common case is a re-quantized file at a new path, and the marker's own record makes what
    happened auditable in the log."""
    verify_ok = scratch_paths(scratch)["verify_ok"]
    if not verify_ok.exists():
        if force:
            print(f"[ship] register: WARNING {verify_ok} not found, registering anyway (--force)")
            return
        raise SystemExit(
            f"{verify_ok} not found; run `ship.py verify --gguf {gguf}` (inside a GPU window) "
            f"before registering the arm, or pass --force"
        )
    try:
        recorded = json.loads(verify_ok.read_text()).get("gguf", "")
    except (json.JSONDecodeError, OSError):
        recorded = ""
    if recorded and Path(recorded).name != Path(gguf).name:
        print(f"[ship] register: WARNING {verify_ok} records {recorded}, not {gguf}")


def cmd_register(args: argparse.Namespace) -> None:
    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"gguf not found: {gguf}")
    check_verify_ok_for(Path(args.scratch), gguf, args.force)

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

    # The bf16 base is KEPT by default (fix round 3, finding 6): the ledger's ruling is that
    # the ~59GB google/gemma-4-31B-it download stays on disk for the Phase 3/4 rounds, because
    # re-downloading it costs about 80 minutes per round. --include-base is the explicit opt-in
    # to delete it anyway.
    targets = [paths["merged"], paths["f16_gguf"]]
    if getattr(args, "include_base", False):
        targets.insert(0, paths["bf16_base"])
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
        train_template=args.train_template, skip_template_check=args.skip_template_check,
    ))
    cmd_convert(argparse.Namespace(
        scratch=args.scratch, force=args.force,
        merged=str(paths["merged"]), out=str(paths["f16_gguf"]), python=args.python,
        model_name=args.model_name,
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
    m.add_argument("--train-template", default=None,
                   help="chat_template.jinja the training data was rendered with (default: "
                        "the adapter's own, else lab/data.py's DEFAULT_TEMPLATE)")
    m.add_argument("--skip-template-check", action="store_true",
                   help="skip the serving-vs-training chat template canary")
    m.add_argument("--force", action="store_true")

    c = sub.add_parser("convert")
    c.add_argument("--merged", default=None, help="merged HF dir (default: <scratch>/merged)")
    c.add_argument("--out", default=None, help="F16 GGUF out (default: <scratch>/gemma-4-31b-cad-F16.gguf)")
    c.add_argument("--python", default=None, help="override the auto-picked converter interpreter")
    c.add_argument("--model-name", default=None,
                   help="GGUF general.name (default: derived from the --out file stem)")
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
    v.add_argument("--i-know-the-gpu-is-free", action="store_true",
                   help="run outside a GPU window (only when the GPU was freed by hand)")

    r = sub.add_parser("register")
    r.add_argument("--gguf", required=True)
    r.add_argument("--name", default=DEFAULT_ARM_NAME)
    r.add_argument("--adapter", default=None, help="adapter path recorded in the arm's notes")
    r.add_argument("--store", default=str(DEFAULT_STORE_DIR))
    r.add_argument("--arms-file", default=str(ARMS_FILE))
    r.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    r.add_argument("--force", action="store_true")

    cl = sub.add_parser("clean")
    cl.add_argument("--include-base", action="store_true",
                    help="also delete the bf16 base (kept by default: re-downloading it costs "
                         "about 80 minutes per later phase)")

    a = sub.add_parser("all")
    a.add_argument("--adapter", required=True)
    a.add_argument("--base", default=None)
    a.add_argument("--python", default=None)
    a.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    a.add_argument("--quant-out", default=None)
    a.add_argument("--model-name", default=None)
    a.add_argument("--train-template", default=None)
    a.add_argument("--skip-template-check", action="store_true")
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

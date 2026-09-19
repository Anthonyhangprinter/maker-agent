#!/usr/bin/env python3
"""run_cq.py -- exec a CadQuery script in a fresh namespace and export STEP.

Runs BY the isolated cq venv's python (benchmarks/lang-ab/.venv-cq/bin/python), never by
the system python: see setup_cq_env.sh for why (the production CAD engine depends on the
SYSTEM build123d 0.10.0 + OCP 7.8.1.1, and cadquery vendors its own OCP build that must
never touch site-packages).

Usage:
    .venv-cq/bin/python run_cq.py <code.py> <output.step>

Exit 0 and a non-empty STEP on success. Exit non-zero with the full traceback on stderr on
any failure (bad syntax, no result found, export failure). The caller enforces the
wall-clock timeout via subprocess.run(..., timeout=...); this script has none of its own.

Convention for finding "the result": a `result` variable in the executed script's
namespace wins if present (a cq.Workplane, cq.Assembly, cq.Shape, or anything exposing a
build123d/cadquery-style `.val()`); otherwise the last object passed to `show_object(...)`
is used (a stub is injected so scripts written in the interactive CQ-editor convention,
which call show_object instead of assigning `result`, still work).

Run with PYTHONUTF8=1 and this file itself is read/written as utf-8 throughout: OpenCascade
resets the process locale to C after a mesh export, and an unencoded read after that point
would decode as ASCII and corrupt anything non-ASCII in stdout/stderr capture.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _find_result(namespace: dict, recorded: list):
    result = namespace.get("result")
    if result is not None:
        return result
    if recorded:
        return recorded[-1]
    return None


def _export(result, output_path: Path) -> None:
    import cadquery as cq

    if not isinstance(result, (cq.Workplane, cq.Assembly, cq.Shape)) and not hasattr(result, "val"):
        raise TypeError(
            f"`result` is a {type(result).__name__}, not a cq.Workplane/cq.Assembly/cq.Shape "
            "(and has no .val())"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cq.exporters.export(result, str(output_path))
    except Exception:
        if isinstance(result, cq.Assembly):
            result.save(str(output_path), exportType=cq.exporters.ExportTypes.STEP)
        else:
            raise
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("export produced no STEP output (empty or missing file)")


def run(code_path: Path, output_path: Path) -> None:
    source = code_path.read_text(encoding="utf-8")
    recorded: list = []

    def show_object(obj, **_kwargs):
        recorded.append(obj)
        return obj

    namespace: dict = {"__name__": "__cq_lang_ab_script__", "show_object": show_object}
    exec(compile(source, str(code_path), "exec"), namespace)

    result = _find_result(namespace, recorded)
    if result is None:
        raise ValueError(
            "no `result` variable and no show_object(...) call found in the script"
        )
    _export(result, output_path)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: run_cq.py <code.py> <output.step>", file=sys.stderr)
        return 2
    code_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        run(code_path, output_path)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Regression test for the locale-dependent child-output decoding bug (2026-09-19).

cad_engine.run_step / run_inspect / run_diff (and the render/stl/dxf runners beside them)
used to call subprocess.run(..., capture_output=True, text=True). text=True with no explicit
encoding decodes the child's stdout/stderr using subprocess._text_encoding(), which -- when
Python's own UTF-8 mode is off -- resolves to locale.getencoding(): the CURRENT process-wide
C locale codeset, read live at call time, not the environment cad_engine started with.

scripts/step always prints a line like "Volume:  0.001 mm³" (the ³ is UTF-8 bytes
c2 b3). If something resets the CALLING process's live LC_CTYPE to the C (ASCII) locale --
in production this was very likely a native library (build123d's CAD kernel) calling
setlocale() internally -- by the time run_step's subprocess.run call happens, decoding that
line raises

    UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position ...

even though the child process itself wrote perfectly valid UTF-8 (it started fresh, so its
OWN locale setup was unaffected). scripts/fluid_gen.py's _materialize() catches that
exception and reports "the script failed to run: ...", so a build that actually succeeded
on disk gets recorded as a crash.

This test reproduces the exact split that made the bug easy to miss: the process that
imports cad_engine and calls run_step has its live LC_CTYPE forced to C (ASCII preferred
encoding, asserted below so the test cannot silently pass without the condition it means to
test), while the grandchild scripts/step process it spawns keeps a normal UTF-8 environment
and writes the mm³ line correctly. Everything runs in a real subprocess with real
build123d and the real scripts/step runner; nothing is mocked.

Run: python3 -m pytest tests/test_engine_locale.py -q
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent

_TRIVIAL_BOX = "from build123d import *\nresult = Box(10, 10, 10)\n"

_CHILD_SCRIPT = textwrap.dedent("""
    import json, locale, os, sys
    from pathlib import Path

    sys.path.insert(0, {here!r})

    out = {{
        "preferred_encoding": locale.getpreferredencoding(False),
        "utf8_mode": sys.flags.utf8_mode,
    }}

    # Give any subprocess THIS process spawns (scripts/step, via run_step) a normal UTF-8
    # environment, matching the real incident: a native library flips the CALLING process's
    # live LC_CTYPE mid-run, but a freshly spawned child still boots its own correct locale
    # from the environment, which was never touched. Mutating os.environ here does not change
    # this process's OWN already-established preferred encoding (see the assertions below) --
    # only a real subprocess launched afterwards reads it.
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
    os.environ.pop("LANGUAGE", None)
    os.environ.pop("PYTHONCOERCECLOCALE", None)
    os.environ.pop("PYTHONUTF8", None)

    try:
        import build123d  # noqa: F401
    except Exception as e:
        out["skip"] = "build123d unavailable: {{}}".format(e)
        print(json.dumps(out))
        sys.exit(0)

    step_runner = Path({here!r}) / "scripts" / "step"
    if not step_runner.exists():
        out["skip"] = "scripts/step not found at {{}}".format(step_runner)
        print(json.dumps(out))
        sys.exit(0)

    import cad_engine

    work_dir = Path({work_dir!r})
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        step_path, output = cad_engine.run_step({box_code!r}, work_dir)
        out["ok"] = True
        out["step_exists"] = step_path.exists()
        out["output"] = output
    except Exception as e:
        out["ok"] = False
        out["error"] = "{{}}: {{}}".format(type(e).__name__, e)

    print(json.dumps(out))
""")


def test_run_step_decodes_child_output_under_ascii_locale(tmp_path):
    """run_step must not crash, and must return the mm³ line intact, even when the
    CALLING process's own preferred encoding has been forced to ASCII."""
    work_dir = tmp_path / "build"
    script = _CHILD_SCRIPT.format(here=str(HERE), work_dir=str(work_dir), box_code=_TRIVIAL_BOX)

    env = dict(os.environ)
    # Force the launched process's OWN preferred encoding to real ASCII at interpreter
    # startup. LC_ALL=C LANG=C alone is not enough on modern CPython + glibc: PEP 538/540
    # auto-coerce a bare C/POSIX locale to a UTF-8 one (C.UTF-8) unless both the coercion and
    # the UTF-8-mode fallback are explicitly disabled, which would silently rescue the buggy
    # code and make this test meaningless. The process then restores a normal UTF-8
    # environment for its OWN children before calling run_step (see _CHILD_SCRIPT) so the
    # scripts/step grandchild writes normal UTF-8 output, matching the real incident: a
    # mid-process locale flip in the caller, not a broken environment for everyone.
    env.update({
        "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C",
        "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0",
    })

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, encoding="utf-8", errors="replace", timeout=120, env=env,
    )
    assert result.returncode == 0, (
        f"child interpreter crashed (rc={result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, f"child produced no output; stderr:\n{result.stderr}"
    payload = json.loads(lines[-1])

    # Prove the harness actually created the ASCII-locale condition the bug needs. If this
    # fails, the test is not exercising the bug at all, so it must not be allowed to pass.
    assert payload["utf8_mode"] == 0, f"child unexpectedly ran in UTF-8 mode: {payload}"
    assert payload["preferred_encoding"].upper() in ("ANSI_X3.4-1968", "ASCII", "US-ASCII"), (
        f"harness failed to force an ASCII child locale: {payload}"
    )

    if "skip" in payload:
        pytest.skip(payload["skip"])

    assert payload.get("ok"), f"run_step raised under an ASCII-locale caller: {payload}"
    assert payload["step_exists"], "run_step did not produce a STEP file"
    assert "mm³" in payload["output"], (
        f"run_step's returned output lost the mm³ unit text: {payload['output']!r}"
    )

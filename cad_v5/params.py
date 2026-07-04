"""E1 — parametric handoff without an LLM.

The build123d script the coder wrote IS the parametric model: every key dimension is a named
numeric constant at the top (enforced by the codegen contract). These commands edit those
numbers and re-run the recipe directly — seconds, deterministic, no model involved:

    cad params                # list the editable dimensions of the last build
    cad regen wall=3 depth=25 # rebuild with overrides (re-verified by scripts/inspect)
"""
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import log, SCRIPTS_DIR, STEP_OUT, STL_OUT, _OPENCLAW

BUILDS_DIR = _OPENCLAW / "cad-builds"

# A parameter is a top-level (unindented) `name = <number>` line, optionally commented.
_PARAM_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<val>-?\d+(?:\.\d+)?)\s*(?:#.*)?$",
                       re.M)
_RESERVED = {"result"}


def latest_source(explicit: str | None = None) -> Path:
    """The most recent build's saved recipe (build_source.py in its artifact dir)."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            p = p / "build_source.py"
        if not p.exists():
            raise FileNotFoundError(f"No recipe at {p}")
        return p
    sources = sorted(BUILDS_DIR.glob("*/build_source.py"))
    if not sources:
        raise FileNotFoundError(
            f"No saved recipes in {BUILDS_DIR} — run a build first (recipes are saved "
            f"per-build from now on).")
    return sources[-1]


def list_params(src: str) -> dict[str, float]:
    return {m["name"]: float(m["val"]) for m in _PARAM_RE.finditer(src)
            if m["name"] not in _RESERVED}


def apply_overrides(src: str, overrides: dict[str, float]) -> str:
    def sub(m):
        name = m["name"]
        if name in overrides:
            return f"{name} = {overrides[name]:g}"
        return m.group(0)
    return _PARAM_RE.sub(sub, src)


def _run(script: str, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS_DIR / script), *args],
                          capture_output=True, text=True, timeout=timeout)


def cmd_params(source: str | None = None) -> int:
    src_path = latest_source(source)
    params = list_params(src_path.read_text())
    print(f"Recipe: {src_path}")
    if not params:
        print("No editable parameters found — this recipe predates the parameter contract. "
              "Rebuild once with the agent and its recipe will carry named dimensions.")
        return 1
    width = max(len(k) for k in params)
    for k, v in params.items():
        print(f"  {k:<{width}} = {v:g}")
    print(f"\nEdit + rebuild without the AI:  cad regen {next(iter(params))}=<value>")
    return 0


def cmd_regen(assignments: list[str], source: str | None = None) -> int:
    overrides: dict[str, float] = {}
    for a in assignments:
        if "=" not in a:
            print(f"Not an assignment: {a!r} — use name=value, e.g. wall=3")
            return 2
        k, v = a.split("=", 1)
        try:
            overrides[k.strip()] = float(v)
        except ValueError:
            print(f"{k.strip()!r} needs a number, got {v!r}")
            return 2

    src_path = latest_source(source)
    src = src_path.read_text()
    known = list_params(src)
    unknown = [k for k in overrides if k not in known]
    if unknown:
        print(f"Unknown parameter(s): {', '.join(unknown)}")
        print(f"This recipe has: {', '.join(known) or '(none)'}")
        return 2

    new_src = apply_overrides(src, overrides)
    slug = "regen-" + "-".join(f"{k}{v:g}" for k, v in overrides.items())[:32]
    out_dir = BUILDS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / "build_source.py"
    new_path.write_text(new_src)
    step = out_dir / "build.step"

    print(f"Rebuilding with {', '.join(f'{k}={v:g}' for k, v in overrides.items())} … "
          f"(no AI — direct re-run)")
    r = _run("step", str(new_path), str(step))
    if r.returncode != 0 or not step.exists():
        print("Rebuild FAILED — the recipe no longer runs with those values:")
        print((r.stderr or r.stdout).strip()[-800:])
        return 1

    insp = _run("inspect", str(step))
    print(insp.stdout.strip())
    if insp.returncode != 0:
        print("⚠ geometry check failed — the new values produce a degenerate part; "
              "the previous build files are untouched.")
        return 1

    # Refresh the convenience copies only for a verified rebuild.
    shutil.copy(step, STEP_OUT)
    stl = out_dir / "build.stl"
    r = _run("stl", str(step), str(stl))
    if r.returncode == 0 and stl.exists():
        shutil.copy(stl, STL_OUT)
    print(f"\n✓ Regenerated.  STEP: {step}")
    print(f"  (also copied to {STEP_OUT.name}/{STL_OUT.name} in {_OPENCLAW})")
    return 0

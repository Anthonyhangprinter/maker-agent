#!/usr/bin/env python3
"""lab/_armwindow.py -- the arm bookend, factored out of lab/specgen.py (Phase 3 Task 3
ruling "One implementation of the arm bookend", 2026-09-19).

lab/specgen.py's signal handlers, `.pre-arm` marker check, `_ensure_resident_up`, `_run_arms`
and the CAD_GPU_WINDOW gate took four review rounds to get right (see specgen.py's own
module docstring and tests/test_lab_specgen_signals.py). lab/harvest.py (Task 3) needs the
exact same bookend -- refuse outside a GPU window, install SIGTERM/SIGHUP handlers, switch
the maker arm, and on exit restore ONLY if a `.pre-arm` marker exists, else just make sure
the resident is up -- so this module is a COPY (not a move) of those pieces, verbatim,
wrapped as one context manager (`arm_window()`) that both callers can share.

THIS IS A COPY, NOT A MOVE. lab/specgen.py keeps its own copies today (2026-09-19 hard
freeze: specgen.py must not change while a reviewed unattended run is scheduled against it
tonight). The plan is for specgen.py to import from here tomorrow and delete its own copies
-- for that migration to be a no-op, every name and every line of behaviour below is kept
byte-for-byte identical to lab/specgen.py's originals. Do not "clean up" anything here without
also updating specgen.py's copy in the same change, once the freeze is lifted.

Usage (see lab/harvest.py):

    from lab._armwindow import arm_window
    from lab import ship

    ship.require_gpu_window(args, MY_HINT)   # refuse before ANYTHING else touches a
                                              # model/service -- see arm_window()'s own
                                              # docstring for why this happens outside the
                                              # context manager, not inside it.
    with arm_window(arm_name) as arm:
        ...  # do the GPU work; SIGTERM/SIGHUP raise SpecgenAborted, which propagates
             # through this block (never swallowed) so the restore step below still runs.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

ARMS_PY = HERE / "scripts" / "arms.py"
MAKER_UNIT = "maker-server"

# Fallback only: used when cad.json has no maker block configured yet (a fresh box). Any
# box that has ever run a CAD build or a card already has a real maker.arm/alias on disk,
# which _default_arm() reads and uses instead -- this module never hardcodes which arm is
# the campaign's current standing pick.
DEFAULT_ARM_FALLBACK = "gemma-4-31b"


class SpecgenAborted(RuntimeError):
    """Raised when a signal (SIGTERM/SIGHUP) lands during the GPU window, and by any
    caller-side circuit breaker that wants the SAME cleanup guarantee. Name kept identical
    to lab/specgen.py's own exception (same class, copied verbatim) so that when specgen
    imports this module tomorrow, `except specgen.SpecgenAborted` (used by its own tests
    and call sites) keeps working with zero rename -- the name reaching specgen's namespace
    via `from lab._armwindow import SpecgenAborted` is the exact same class object."""


_CLEANUP_STARTED = False   # set by _mark_cleanup_started(); see _install_signal_handlers

# Fix round 1 (2026-09-19, Task 3 review H3): set unconditionally by _signal_handler,
# independent of _CLEANUP_STARTED and of whether the SpecgenAborted it also raises ever
# reaches the caller. lab/harvest.py's candidate execution goes through
# scripts/fluid_gen.py's _materialize(), which is frozen (this branch may not edit it)
# and wraps run_step/render/inspect in bare `except Exception: pass` blocks -- a signal
# landing during any of those is swallowed there, so the exception alone is not a
# reliable abort signal for a caller sitting on the other side of that call. This flag
# is: checked via abort_requested() immediately after any such call returns, so the
# caller can raise its own fresh SpecgenAborted from a point _materialize's swallowing
# cannot reach.
_ABORT_REQUESTED = False


def abort_requested() -> bool:
    """True once a SIGTERM/SIGHUP has been received by this process, even if the
    SpecgenAborted it also raised was caught and discarded by code this module does not
    control. Never reset -- once a signal has landed, every subsequent check must keep
    seeing it; there is exactly one abort per process lifetime."""
    return _ABORT_REQUESTED


def _signal_handler(signum, frame) -> None:
    """Raise SpecgenAborted so the caller's own try/finally (here: arm_window()'s) runs
    the restore step, mirroring lab/gpu_window.sh's trap-based restore-on-signal pattern.
    Idempotent by design: once cleanup has begun (_mark_cleanup_started() was called), a
    second SIGTERM/SIGHUP is ignored here rather than raising again -- raising a second
    time WHILE the restore subprocess call is itself blocked would interrupt that call via
    the same PEP 475 mechanism this handler relies on, which could abandon the restore
    mid-way and leave things in a worse state than either finishing it or never starting
    it. The module-level abort flag above is set FIRST, unconditionally, before that
    idempotency check -- a caller relying on abort_requested() (see its own docstring)
    must see the signal regardless of how many times this handler has already fired."""
    global _ABORT_REQUESTED
    _ABORT_REQUESTED = True
    if _CLEANUP_STARTED:
        return
    raise SpecgenAborted(f"terminated by signal {signum} ({signal.Signals(signum).name})")


def _install_signal_handlers() -> None:
    """SIGTERM (the default signal `kill <pid>` sends, and the one a detached
    `setsid nohup ...` process is stopped with) and SIGHUP (a terminal hangup) both
    otherwise terminate the process immediately with no exception raised and no `finally`
    executed -- confirmed empirically, not textbook assumption: Python's default
    disposition for SIGTERM runs no cleanup at all, unlike SIGINT, which the runtime
    converts to KeyboardInterrupt by default. Callers install this before any GPU-window
    setup step (the arm switch included), so even a signal arriving mid-switch is caught."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)


def _mark_cleanup_started() -> None:
    global _CLEANUP_STARTED
    _CLEANUP_STARTED = True


def _run_arms(*args: str) -> subprocess.CompletedProcess:
    """Run scripts/arms.py <args>, forwarding its stdout/stderr into ours so a
    supervisor's log still shows what it did, but returning the CompletedProcess so a
    non-zero exit is visible to the caller instead of being silently swallowed the way a
    bare `subprocess.run(..., check=False)` with no capture would leave it."""
    p = subprocess.run([sys.executable, str(ARMS_PY), *args], capture_output=True,
                       encoding="utf-8", errors="replace")
    if p.stdout:
        sys.stdout.write(p.stdout if p.stdout.endswith("\n") else p.stdout + "\n")
    if p.stderr:
        sys.stderr.write(p.stderr if p.stderr.endswith("\n") else p.stderr + "\n")
    return p


def _pre_arm_marker_paths() -> tuple[Path, Path]:
    """The exact (cad.json.pre-arm, maker.env.pre-arm) paths scripts/arms.py itself
    checks, imported directly from that module rather than re-derived here: `arms.CAD_JSON`/
    `arms.ENV_PATH` already honour the CAD_CONFIG_FILE/MAKER_ENV env vars, and
    `arms._pre_arm_paths()` is the exact function `cmd_use`/`cmd_restore` use, so this can
    never drift from what the real `arms.py` subprocess (launched by _run_arms, inheriting
    this same process's environment) would check."""
    import arms as arms_cli   # scripts/ is already on sys.path (see the top of this file)
    return arms_cli._pre_arm_paths(arms_cli.CAD_JSON, arms_cli.ENV_PATH)


def _pre_arm_marker_exists() -> bool:
    pre_cad, pre_env = _pre_arm_marker_paths()
    return pre_cad.exists() or pre_env.exists()


def _maker_unit_active() -> bool:
    """True when maker-server is running right now, via scripts/arms.py's own `unit_active`
    (imported the same way `_pre_arm_marker_paths` imports its path helpers, so there is one
    definition of "active" and it can never drift from what arms.py itself would decide).
    `unit_active` never raises. An import failure here is read as "not active": this helper
    only ever guards an extra refusal, and failing to import arms.py at all is a much louder
    problem than a resident that was started when it did not need to be."""
    try:
        import arms as arms_cli   # scripts/ is already on sys.path (see the top of this file)
    except Exception:
        return False
    return arms_cli.unit_active(MAKER_UNIT)


def _ensure_resident_up() -> None:
    """The cad.json-untouched fallback when no pre-arm marker exists: no function in
    scripts/arms.py isolates "just make sure the resident answers" from its own
    cad.json/maker.env read-modify-write logic, so calling `arms.py restore` here (which
    would hit `cmd_restore`'s no-marker fallback and write maker.enabled: false) or
    `arms.py restore --disable` (same effect, stated explicitly) are both wrong. This
    mirrors just the resident-starting systemctl call `cmd_restore` itself makes, directly,
    touching no config file at all -- an idempotent no-op in the overwhelmingly likely case
    that qwen38-server was never stopped in the first place, and a real (if best-effort)
    recovery in the unlikely case that it somehow was.

    With ONE exception: if maker-server is active and this run wrote no pre-arm marker,
    that maker is provably not this run's doing, so it belongs to something else -- a CAD
    build, a benchmark card, another lab job. Starting qwen38-server would stop it
    (maker-server.service declares Conflicts=qwen38-server.service, and systemd's
    documented behaviour is that starting either side stops the other), i.e. the cleanup
    path of a run that changed nothing would kill someone else's in-flight GPU work. Leave
    it alone and say so."""
    if _maker_unit_active():
        print("armwindow cleanup: maker-server is active and this run wrote no pre-arm "
              "marker: not ours, leaving it alone", file=sys.stderr)
        return
    subprocess.run(["systemctl", "--user", "start", "qwen38-server"], check=False)


def _default_arm() -> str:
    """The arm name currently configured in cad.json's maker block -- so a caller never
    has to hardcode which arm is the campaign's standing pick. Falls back to
    DEFAULT_ARM_FALLBACK only when cad.json has no maker block yet, or cad_engine cannot be
    imported at all (kept import-safe: a caller that always passes an explicit `arm` to
    arm_window() never needs cad_engine on the path)."""
    try:
        import cad_engine as engine   # heavy import; only needed for this fallback lookup
    except Exception:
        return DEFAULT_ARM_FALLBACK
    m = engine._load_config().get("cad", {}).get("maker") or {}
    return m.get("arm") or m.get("alias") or DEFAULT_ARM_FALLBACK


@contextmanager
def arm_window(arm: Optional[str] = None) -> Iterator[str]:
    """Install the signal handlers, switch to `arm` (or the cad.json default), run the
    caller's body, then restore -- with the exact marker-based decision specgen.py's main()
    uses: `.pre-arm` marker present -> `scripts/arms.py restore`; absent -> only make sure
    the resident is up, config untouched.

    Callers MUST call `ship.require_gpu_window(args, hint)` themselves BEFORE entering this
    context manager (see the module docstring) -- that check must happen before ANYTHING
    that touches a model or a service, including installing signal handlers, and this
    context manager's whole job starts one step later than that.

    The try/finally below spans the arm switch itself (not just the caller's body), which
    is what makes the marker-based decision safe against a signal landing WHILE `use` is
    still in flight: no marker on disk means `use` never got past its own first
    check-model-path/`_save_pre_arm()` step, so cad.json/maker.env were never touched and
    nothing needs undoing; a marker on disk is real evidence a restore is owed, regardless
    of exactly when the signal landed. This is a `@contextmanager` generator specifically
    so that an exception raised BEFORE the `yield` (i.e. during the `use` call) still runs
    the `finally` block: Python's ordinary try/finally semantics apply across the yield
    boundary just as they would in a plain function.

    A second SIGTERM/SIGHUP arriving while this cleanup is already running is ignored (see
    _signal_handler), so it cannot abandon a restore partway through.

    Any exception raised by the caller's body -- including SpecgenAborted from a signal --
    propagates out of this context manager unchanged after cleanup runs; it is never
    swallowed here."""
    resolved_arm = arm or _default_arm()
    _install_signal_handlers()   # before the `use` call: even a signal during the arm
                                  # switch itself must reach the cleanup below.
    keep_maker_prev = os.environ.get("CAD_KEEP_MAKER")
    try:
        use_p = _run_arms("use", resolved_arm)
        if use_p.returncode != 0:
            raise RuntimeError(
                f"scripts/arms.py use {resolved_arm} failed (exit {use_p.returncode})")
        os.environ.setdefault("CAD_KEEP_MAKER", "1")   # one warm arm across this window
        yield resolved_arm
    finally:
        _mark_cleanup_started()   # a second SIGTERM/SIGHUP from here on is ignored
        # The marker on disk, not a flag computed earlier, is the ONLY thing that decides
        # whether a restore is safe -- see the docstring above and lab/specgen.py's own
        # main() for the fix-round-3 finding this mirrors.
        if _pre_arm_marker_exists():
            print("armwindow cleanup: pre-arm marker found, restoring via "
                  "scripts/arms.py restore", file=sys.stderr)
            _run_arms("restore")
        else:
            print("armwindow cleanup: no pre-arm marker, leaving cad.json untouched; "
                  "ensuring qwen38-server is up", file=sys.stderr)
            _ensure_resident_up()
        # Restore CAD_KEEP_MAKER at the source rather than relying on the caller's own
        # environment to clean it up.
        if keep_maker_prev is None:
            os.environ.pop("CAD_KEEP_MAKER", None)
        else:
            os.environ["CAD_KEEP_MAKER"] = keep_maker_prev

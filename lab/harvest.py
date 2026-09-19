#!/usr/bin/env python3
"""lab/harvest.py -- the Phase 3 "data engine" harvest unit (Task 3).

Samples build123d code from the local CAD model (the maker arm, or the resident when the
maker is disabled) for specs in `lab/state/specs.jsonl` (the bank built by
`lab/specbank.py`), verifies every candidate, and keeps only verified winners as training
pairs. Every candidate, kept or not, gets one row in the ledger, so the record of what
was tried is never lossy.

**Execution and gating are two separate concerns (fix round 1, 2026-09-19).**
`scripts/fluid_gen.py._materialize` is reused for EXECUTION SIDE EFFECTS only (it writes
build_source.py/build.step/build.stl/build.png the same way a real fluid build does) --
its own `error`/`gate_hard`/`gate_spec`/`facts` are never trusted beyond a genuine crash,
because that file is frozen (this branch may not edit it) and has two defects that would
otherwise leak into training data: (1) its own `expected_for()` never sets
`expected.solids`, so `verify_expected`'s unfused-bodies HARD check never fires there; a
part that came back as several disconnected bodies could be classified "good" (H1); (2)
every one of its internal execute/inspect/render steps is wrapped in a bare
`except Exception: pass`, so an inspect failure (an invalid STEP, a timeout) is swallowed
silently and reported as `error=None` with empty facts and gate lists, which this module
would otherwise also classify "good" with nothing actually measured (H2). `_regate()`
below re-inspects the STEP directly and re-verifies from scratch against
`engine.reconcile_expected`'s `expected` block (the same `forbid_blind_holes` derivation
fluid's own `expected_for` uses, PLUS `expected.solids`, which fluid's never sets).

All local: sampling, execution and scoring all happen inside THIS process, on the arm the
GPU window already switched to (`lab._armwindow.arm_window`) -- no cloud call anywhere.
`CAD_BENCH=1` is set before `cad_engine` is even imported so a good build can never
self-promote into `~/.openclaw/cad-examples.jsonl` / `cad-sftpairs.jsonl` (the bank has its
own files: `ledger.jsonl` / `pairs.jsonl` / `review.jsonl`).

Two passes, never both in the same unit ("a unit never swaps arms", Task 3 ruling): a
"student" pass (`engine.generate_code_raw`, thinking off, the same call fluid mode's
strong rung makes) for specs that have not yet failed the student twice; a "think" pass
for specs that HAVE, only when `cad_v5.config.think_rung_available()` says the active
arm's request shape actually changes anything, and only once at least 20 specs are
waiting for it. The think pass reuses `generate_code_raw` unmodified too (fix round 1,
M4): `_ThinkInjector` wraps `engine._ollama` to force `think=True` onto the one call
`generate_code_raw` makes, composed with `_PromptRecorder` so the call actually recorded
is the real, think-injected one -- no separate, hand-built prompt string.

Verdict table (spec-derived contamination is already handled at bank-admission time --
see lab/specbank.py -- and re-checked here fail-closed before any pair is written, L1):
  ok, gate_hard==0, gate_spec==0, band in (None, "match")  -> pair, kind "good"
  ok, gate_hard==0, gate_spec>0                            -> silver review row, not a pair
  reference present (owner-reference bank rows only, none exist on disk as of 2026-09-19)
    and band == "near_miss"                                -> pair, kind "fail"
  the candidate ran but could not be measured/gated (inspect invalid/raised, or a
    required fact -- solids, volume, bbox -- is missing)   -> verdict "unscored": a
    ledger row with the reason, never a pair, never a review row, counted separately
    in status (H2; "good" REQUIRES a valid inspect with those facts present)
  everything else                                          -> nothing (still ledgered)
A "good" candidate whose reference row has NO source code of its own (an owner-reference
spec.txt/model.step pair has geometry, never a build123d program) cannot produce a "fail"
pair with a correct target: `code` is left null on that pair and `bad_code`/`problem`
carry the wrong candidate and its geometric diagnosis, same as gift_sample.py's GIFT-FAIL
rows. This branch is untested by real data today (the reference folder is empty) but is
exercised by a planted fixture in tests/test_lab_harvest.py.

**Aborts (fix round 1, H3).** A SIGTERM/SIGHUP mid-candidate must never be recorded as a
build failure and must never count as a student attempt. Because `_materialize`'s own
`except Exception: pass` blocks would otherwise swallow the SpecgenAborted exception a
signal raises, `lab._armwindow.abort_requested()` (a flag set unconditionally by the
signal handler, independent of whether that exception was ever seen by anyone) is
checked immediately after every `_materialize` call, after every salvage attempt, at the
top of every candidate iteration, and between specs; a candidate in flight when the flag
is found set is un-attempted (its attempt-counter increment is undone) and gets NO
ledger row, and `SpecgenAborted` is raised from outside any generic `except`.

Runs ONLY inside a GPU window, exactly like lab/specgen.py (`ship.require_gpu_window`,
checked before ANYTHING else -- no signal handler, no arm switch, no bank read):

    lab/harvest_unit.sh                                          # the timer's own launch
    lab/gpu_window.sh python3 lab/harvest.py --unit               # equivalent, direct
    lab/gpu_window.sh python3 lab/harvest.py --once --spec-id ID  # one spec now, no gates
    python3 lab/harvest.py --status                                # no GPU window needed
    python3 lab/harvest.py --check-gate                            # no GPU window needed

**`--check-gate` (fix round 1, H4).** `lab/harvest_unit.sh` used to enter
`lab/gpu_window.sh` (build lock + resident eviction + cold arm start) BEFORE any of the
cheap gates below could run, so a daytime tick paid a full resident bounce just to
discover it should skip. `--check-gate` runs the exact same `_unit_gate()` function
`--unit` re-checks once inside the window, PLUS a non-blocking probe of the CAD build
lock (`flock -n`, open-and-release, never waits -- unlike `lab/gpu_window.sh`'s own
`flock`, which is frozen today and waits up to an hour; that stays the real arbiter, this
is only a cheap "worth trying" signal), and exits 0 (go) or 3 (skip, reason on stderr).
`lab/harvest_unit.sh` runs it first and only enters the GPU window on a 0.

`--unit` additionally gates on (in order, same function `--check-gate` calls): a
`lab/state/paused` file; the night window (`cad.json`'s `lab.harvest` block --
night_start/night_end/day_allowed); today's GPU-hour budget (summed from the ledger's own
`seconds`, per LOCAL calendar day); the gpu-proxy's `waiting` counter at :8087 (an
unreachable/misshapen proxy status is read as 0 -- see `_gpu_proxy_waiting()`'s own
docstring for why that is a deliberate fail-open, not an oversight); and the bank being
exhausted (no spec has room for another pair). `--once --spec-id` skips all of those (the
plan's own words: "runs one spec now (smoke/tests) without the gates") but NOT the
GPU-window gate itself. There is no second locking scheme: `lab/harvest_unit.sh`
serialises concurrent timer fires with its own `flock` on `lab/state/harvest.lock`, and
`lab/gpu_window.sh` (frozen, unmodified) is the one thing that ever actually HOLDS
`~/.openclaw/cad-build.lock`.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

# Before cad_engine (or anything that might build) is even imported: the harvest builds
# and scores in its own process exactly like a benchmark card run, and must never let a
# converged/good build self-promote into cad-examples.jsonl / cad-sftpairs.jsonl -- the
# bank has its own files. Sourcing this exclusively from cad_engine's own CAD_BENCH guard
# (rather than duplicating the exclusion here) means the ONE place that decides "was this
# a benchmark/harvest run" never drifts between the two.
os.environ.setdefault("CAD_BENCH", "1")

import cad_engine as engine  # noqa: E402
import fluid_gen  # noqa: E402  (scripts/fluid_gen.py: _materialize, _revise_on_repair_rung)
import harvest_census as hc  # noqa: E402
from cad_v5.diagnose import diagnose  # noqa: E402
from cad_v5.config import (  # noqa: E402
    BUILD_LOCK_FILE, CODE_MODEL_STRONG, lab_config, maker_config, think_rung_available,
)
from geom_bands import score_against_reference  # noqa: E402
from lab import specbank  # noqa: E402
from lab import ship  # noqa: E402
from lab._armwindow import SpecgenAborted, abort_requested, arm_window  # noqa: E402
from lab.data import default_contamination_sets  # noqa: E402

STATE_DIR    = HERE / "lab" / "state"
LEDGER_FILE  = STATE_DIR / "ledger.jsonl"
PAIRS_FILE   = STATE_DIR / "pairs.jsonl"
REVIEW_FILE  = STATE_DIR / "review.jsonl"
PROGRESS_FILE = STATE_DIR / "progress.json"
STATUS_FILE  = STATE_DIR / "status.json"
PAUSED_FILE  = STATE_DIR / "paused"
BUILDS_DIR   = STATE_DIR / "builds"
SYSTEMS_DIR  = STATE_DIR / "systems"

GPU_PROXY_STATUS_URL = "http://127.0.0.1:8087/"

GPU_WINDOW_HINT = (
    "harvest samples the maker arm in a loop for a whole unit: it must run inside a GPU "
    "window: `lab/harvest_unit.sh` (the timer's own launcher) or directly "
    "`lab/gpu_window.sh python3 lab/harvest.py --unit` / `--once --spec-id ID` (that "
    "holds ~/.openclaw/cad-build.lock, evicts the resident and the maker arm first, and "
    f"exports {ship.GPU_WINDOW_ENV}=1). Without the lock this run would take the GPU out "
    "from under any CAD build, benchmark card or lab job already using it. Pass "
    "--i-know-the-gpu-is-free only when the GPU is already free by hand. `--check-gate` "
    "and `--status` need no GPU window at all."
)

_TIER34 = ("3", "4")


class _InfraError(RuntimeError):
    """A codegen failure caused by the model server itself being unreachable (Task 3 fix
    L3), not a per-candidate defect: connection refused, a network/DNS failure, a read
    timeout. Aborts the WHOLE unit rather than being counted as a failed student attempt
    -- a dead server teaches nothing about this spec, and letting it burn through every
    remaining candidate as ordinary "student failures" would wrongly promote specs toward
    the teacher pass and pollute the pass-rate stats with an outage, not a model result."""


# ---------------------------------------------------------------------------
# Small local helpers -- time, JSONL/JSON I/O, all under a flock or atomic rename so a
# signal landing mid-write can never leave a half-written line or a corrupt state file.
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_abort() -> None:
    """Raise SpecgenAborted the moment a signal has landed, even if the exception it also
    raised was swallowed somewhere this module does not control (Task 3 fix H3). Called
    from OUTSIDE any generic `except` -- a plain top-level statement, never itself wrapped
    in a try/except Exception that could eat it a second time."""
    if abort_requested():
        raise SpecgenAborted("aborted by signal (detected via the abort flag after a "
                             "call that may have swallowed the original exception)")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    """One complete JSON line per call, flushed under an exclusive flock -- a signal
    landing anywhere in this function either writes the WHOLE line or none of it; there is
    no code path that could leave a half-written line on disk. json.dumps defaults to
    ensure_ascii=True (every row is pure ASCII on disk regardless of locale), but
    encoding="utf-8" is still explicit here rather than relying on that being forever
    true of every future caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _atomic_write_json(path: Path, data) -> None:
    """write-to-temp + os.replace: the file at `path` is either the old complete version
    or the new complete version, never a partial write, whatever gets interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_progress(progress: dict) -> None:
    _atomic_write_json(PROGRESS_FILE, progress)


def _safe_id(spec_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", spec_id)


def _current_arm_alias() -> str:
    try:
        return maker_config().get("alias", "unknown")
    except Exception:
        return "unknown"


def _code_fingerprint(code: str) -> str:
    """AST-normalised dedup key (Task 3 fix L7): two candidates differing only in
    comments/whitespace collapse to the same fingerprint. A renamed variable does NOT
    collapse (ast.dump preserves identifier names) -- accepted; catching that needs a
    heavier alpha-renaming normalisation this fix does not attempt. Falls back to the raw
    sha1 of the source text when the code does not even parse (should not happen for a
    candidate that already executed successfully, but this must never raise)."""
    try:
        return hashlib.sha1(ast.dump(ast.parse(code)).encode()).hexdigest()
    except Exception:
        return hashlib.sha1(code.encode()).hexdigest()


def _store_system(system: str) -> Optional[str]:
    """Dedup the large, mostly-static system prompt across rows (Task 3 fix M5): write it
    once to lab/state/systems/<sha1>.txt (idempotent -- a second call with the same text
    is a no-op) and return the hash. Pair/review rows carry `system_sha1`, not the ~9.8KB
    string, inline. Task 4's compiler must read `system_sha1` off each row and load
    `lab/state/systems/<system_sha1>.txt` (encoding="utf-8" -- the prompt text itself
    contains real em dashes, not JSON-escaped ones, which is exactly what surfaced this
    fix: writing it via Path.write_text() with no explicit encoding raised
    UnicodeEncodeError under PYTHONUTF8=0 in a C/POSIX locale) to get the actual system
    prompt back before building a ChatML `messages` array -- see this module's own
    report for the exact contract."""
    if not system:
        return None
    h = hashlib.sha1(system.encode()).hexdigest()
    path = SYSTEMS_DIR / f"{h}.txt"
    if not path.exists():
        SYSTEMS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(system, encoding="utf-8")
        os.replace(tmp, path)
    return h


def _is_contaminated(spec: str) -> bool:
    """Fail-closed re-check before ANY pair is written (Task 3 fix L1): the bank already
    refuses a contaminated spec at admission time (lab/specbank.py's refusal_reason), but
    this is a second, independent gate at the point a spec is actually about to be
    sampled, using the exact same primitive lab.data.default_contamination_sets() (which
    scripts/run_card.py's contamination() also uses) rather than a re-derived copy.
    Any failure to even compute the contamination sets is read as contaminated (fail
    closed), never as clear."""
    try:
        suite_keys, suite_slugs = default_contamination_sets()
    except Exception:
        return True
    key = hc._key(spec)
    if key in suite_keys:
        return True
    return hc._slug(spec, 40) in suite_slugs


def _is_infra_error(e: Exception) -> bool:
    """See _InfraError's docstring. urllib.error.URLError is itself an OSError subclass
    (as are ConnectionError and socket.timeout/TimeoutError in modern Python), so this
    tuple is intentionally redundant for readability rather than coverage -- a reviewer
    should not have to know that fact to trust this check."""
    return isinstance(e, (urllib.error.URLError, ConnectionError, OSError,
                         socket.timeout, TimeoutError))


def _prune_builds(keep: int) -> int:
    """Retention cap for lab/state/builds/ (Task 3 fix L9): mtime-based, never the
    newest, same discipline as cad_engine's own KEEP_BUILDS fix (a name-based prune had
    been deleting the wrong directories there). A pruned build dir may belong to an
    already-recorded pair -- fair game, since the pair row holds the code itself and the
    build dir is only supplementary render/mesh evidence. Returns the count removed."""
    if not BUILDS_DIR.exists():
        return 0
    dirs = sorted((d for d in BUILDS_DIR.iterdir() if d.is_dir()),
                 key=lambda d: d.stat().st_mtime)
    removed = 0
    while len(dirs) > keep:
        oldest = dirs.pop(0)
        try:
            shutil.rmtree(oldest)
            removed += 1
        except Exception:
            pass
    return removed


# ---------------------------------------------------------------------------
# Unit gates
# ---------------------------------------------------------------------------

def _gpu_proxy_waiting(timeout: float = 2.0) -> int:
    """The gpu-proxy's :8087 status endpoint reports {"waiting": N, ...} (verified live,
    2026-09-19). Fail-open on any error (unreachable, non-JSON, missing key): the proxy is
    a convenience signal for "don't start a unit while an interactive chat is queued", not
    a safety boundary -- the GPU window and the build lock are the safety boundary, and
    they are unaffected by this check. A misconfigured or down proxy must not permanently
    wedge the nightly timer into skipping every unit forever."""
    try:
        with urllib.request.urlopen(GPU_PROXY_STATUS_URL, timeout=timeout) as r:
            body = json.loads(r.read())
        return int(body.get("waiting") or 0)
    except Exception:
        return 0


def _build_lock_free() -> bool:
    """Non-blocking probe (Task 3 fix H4): true when the CAD build lock (BUILD_LOCK_FILE,
    env-overridable so tests never touch the real lock) is currently uncontended. This is
    instant and NEVER waits, unlike lab/gpu_window.sh's own flock (frozen today, waits up
    to an hour) -- it exists purely so --check-gate can decide "worth trying" a real GPU
    window without paying for one. It is not a second locking scheme: nothing here ever
    holds the lock past the probe itself, and `--unit` inside a real window never calls
    this (by the time it runs, gpu_window.sh already holds the lock as this process's own
    ancestor)."""
    try:
        BUILD_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BUILD_LOCK_FILE, "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
                return True
    except Exception:
        return True   # fail-open: a probe that cannot even run must never permanently
                      # wedge the timer -- gpu_window.sh's own flock is the real boundary.


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def _in_night_window(now_local: datetime, cfg: dict) -> bool:
    start, end = _parse_hhmm(cfg["night_start"]), _parse_hhmm(cfg["night_end"])
    t = now_local.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end   # wraps midnight, e.g. 22:00-07:00


def _night_bucket(local_dt: datetime, cfg: dict) -> Optional[str]:
    """Which calendar night (keyed by the date the night STARTED) `local_dt` falls in, or
    None when it falls outside the configured night window entirely (a daytime `--once`
    run, or a day-allowed unit) -- those don't count toward `nights_completed`."""
    start, end = _parse_hhmm(cfg["night_start"]), _parse_hhmm(cfg["night_end"])
    t = local_dt.time()
    if start <= end:
        return local_dt.date().isoformat() if start <= t < end else None
    if t >= start:
        return local_dt.date().isoformat()
    if t < end:
        return (local_dt.date() - timedelta(days=1)).isoformat()
    return None


def _to_local(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    return dt.astimezone() if dt.tzinfo else dt


def _hours_today(now_local: Optional[datetime] = None) -> float:
    now_local = now_local or datetime.now()
    today = now_local.date()
    total = 0.0
    for r in _read_jsonl(LEDGER_FILE):
        local_dt = _to_local(r.get("ts", ""))
        secs = r.get("seconds")
        if local_dt is None or secs is None or local_dt.date() != today:
            continue
        try:
            total += float(secs)
        except (TypeError, ValueError):
            pass
    return total / 3600.0


def nights_completed(cfg: dict) -> int:
    """Calendar nights with at least `unit_minutes * 6` minutes of ledger time inside the
    configured night window (Task 3: "--status prints ... nights_completed counts calendar
    nights with at least unit_minutes * 6 minutes of ledger time inside the window")."""
    buckets: dict[str, float] = {}
    for r in _read_jsonl(LEDGER_FILE):
        local_dt = _to_local(r.get("ts", ""))
        secs = r.get("seconds")
        if local_dt is None or secs is None:
            continue
        bucket = _night_bucket(local_dt, cfg)
        if bucket is None:
            continue
        try:
            buckets[bucket] = buckets.get(bucket, 0.0) + float(secs)
        except (TypeError, ValueError):
            pass
    threshold = cfg["unit_minutes"] * 6 * 60
    return sum(1 for total in buckets.values() if total >= threshold)


def _eligible_pools(bank: list[dict], progress: dict, cfg: dict) -> tuple[list, list]:
    """(student_pool, teacher_pool): a spec with `pairs >= max_pairs_per_spec` is excluded
    from both (done); a spec with fewer than 2 student attempts is student-pool; one with
    2+ and still short of its pair quota graduates to the teacher pool. Pool membership is
    independent of whether a think pass is actually available this run -- that decision
    (and the >=20-spec threshold) belongs to the caller."""
    max_pairs = cfg["max_pairs_per_spec"]
    student, teacher = [], []
    for row in bank:
        entry = progress.get(row["id"], {})
        if entry.get("pairs", 0) >= max_pairs:
            continue
        if entry.get("student_attempts", 0) >= 2:
            teacher.append(row)
        else:
            student.append(row)
    return student, teacher


def _order_specs(pool: list[dict], progress: dict, prefer_tier34: bool) -> list[dict]:
    """Tier 3-4 first while the pairs' tier34 share is under 0.40, else round-robin.
    "Round robin" is implemented as sort-by-(attempts-so-far, id): a spec sampled this unit
    gains an attempt, which pushes it later in every future ordering, so repeated calls
    across units spread attempts evenly without needing a separate pointer/cursor file."""
    def attempts(row: dict) -> int:
        e = progress.get(row["id"], {})
        return e.get("student_attempts", 0) + e.get("teacher_attempts", 0)

    if prefer_tier34:
        hi = sorted((r for r in pool if str(r.get("tier")) in _TIER34),
                   key=lambda r: (attempts(r), r["id"]))
        lo = sorted((r for r in pool if str(r.get("tier")) not in _TIER34),
                   key=lambda r: (attempts(r), r["id"]))
        return hi + lo
    return sorted(pool, key=lambda r: (attempts(r), r["id"]))


def _unit_gate(cfg: dict) -> Optional[str]:
    """None when a --unit (or --check-gate) may proceed; otherwise the reason it was
    skipped. Does NOT check the build lock (see _build_lock_free, checked separately by
    _check_gate) -- this function alone is what --unit re-runs once already inside the
    GPU window, where a lock self-check would be circular (this process's own ancestor,
    lab/gpu_window.sh, already holds it)."""
    if PAUSED_FILE.exists():
        return "paused (lab/state/paused exists)"
    now_local = datetime.now()
    if not cfg["day_allowed"] and not _in_night_window(now_local, cfg):
        return f"outside the night window ({cfg['night_start']}-{cfg['night_end']})"
    hours_today = _hours_today(now_local)
    if hours_today >= cfg["hours_per_day"]:
        return (f"hours_today {hours_today:.2f} >= hours_per_day {cfg['hours_per_day']}")
    waiting = _gpu_proxy_waiting()
    if waiting:
        return f"gpu-proxy has {waiting} queued request(s)"
    bank = specbank.load_bank()
    progress = _load_progress()
    student_pool, teacher_pool = _eligible_pools(bank, progress, cfg)
    if not student_pool and not teacher_pool:
        return "bank exhausted (no spec has room for another pair)"
    return None


def _check_gate() -> Optional[str]:
    """The gate a timer tick should evaluate BEFORE paying for a GPU window at all (Task 3
    fix H4): _unit_gate()'s own checks, plus the cheap non-blocking build-lock probe.
    None means go; otherwise the skip reason."""
    cfg = lab_config()["harvest"]
    reason = _unit_gate(cfg)
    if reason:
        return reason
    if not _build_lock_free():
        return "the CAD build lock is currently held"
    return None


def _choose_mode(teacher_pool: list[dict], cfg: dict) -> str:
    """"student" or "think" -- never both in the same unit (Task 3 ruling: "a unit never
    swaps arms"). A think pass needs: the config to list it, the active arm's request
    shape to actually change under it (think_rung_available(), consulted here -- per
    _ollama()'s own contract for a caller passing think=True directly), and at least 20
    specs waiting (a teacher unit is not worth an eviction/restore cycle for one spec)."""
    if "think" in (cfg.get("teacher_passes") or []) and len(teacher_pool) >= 20:
        if think_rung_available():
            return "think"
    return "student"


def _check_code_model_pin() -> Optional[str]:
    """Refuses rather than silently harvesting from the wrong model (Task 3 fix M3): if
    cad.json pins `cad.code_model` to something other than the maker arm
    (CODE_MODEL_STRONG), that pin almost certainly outranks the maker arm inside
    `engine._code_model()`'s own resolution order (per-build override > cad.json pin >
    default) and this run would otherwise silently sample from whatever that pin names,
    under this run's `arm`-labelled rows. On a clear pin (or none), sets
    `engine._ACTIVE_CODE_MODEL = CODE_MODEL_STRONG` explicitly, matching gift_sample.py's
    own pattern, so every call in this process is guaranteed to resolve to the maker arm
    regardless of what cad.json says from this point on."""
    pinned = engine._load_config().get("cad", {}).get("code_model")
    if pinned and pinned != CODE_MODEL_STRONG:
        return (f"cad.json pins cad.code_model={pinned!r}, which is not the maker arm "
                f"({CODE_MODEL_STRONG!r}); refusing rather than harvesting from the "
                f"wrong model. Clear the pin or point it at {CODE_MODEL_STRONG!r}.")
    engine._ACTIVE_CODE_MODEL = CODE_MODEL_STRONG
    return None


# ---------------------------------------------------------------------------
# Prompt/usage recording -- a small recorder around engine._ollama (never reconstructed
# after the fact): every codegen path here (student, think, and the crash-salvage's own
# revise call) goes through _ollama exactly once per call, so wrapping it for the
# duration of one call captures the EXACT system/prompt/usage/model that call sent.
# ---------------------------------------------------------------------------

class _PromptRecorder:
    def __init__(self) -> None:
        self.last: Optional[dict] = None

    def __enter__(self) -> "_PromptRecorder":
        self._orig = engine._ollama

        def recording(model, system, prompt, *a, **kw):
            result = self._orig(model, system, prompt, *a, **kw)
            self.last = {"model": model, "system": system, "prompt": prompt,
                        "usage": dict(engine._LAST_USAGE or {})}
            return result

        engine._ollama = recording
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        engine._ollama = self._orig
        return False


class _ThinkInjector:
    """Wraps engine._ollama to force think=True on every call for the duration of one
    generate_code_raw() call (Task 3 fix M4), so the think pass reuses
    generate_code_raw's own prompt-construction code instead of duplicating it (which had
    left one line -- the literal "USER REQUEST (verbatim...)" string -- carrying a
    character copied verbatim from cad_engine.py's own prompt text). no_think always wins
    inside _ollama itself regardless of this injection, so this stays safe even if some
    future caller inside generate_code_raw's call chain ever passes no_think=True
    explicitly."""
    def __enter__(self) -> "_ThinkInjector":
        self._orig = engine._ollama

        def injected(model, system, prompt, *a, **kw):
            kw["think"] = True
            return self._orig(model, system, prompt, *a, **kw)

        engine._ollama = injected
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        engine._ollama = self._orig
        return False


def _student_generate(spec: str, notes: list, temperature: float) -> tuple[str, dict]:
    """The exact production call fluid mode's strong rung makes (thinking off, whatever
    the active arm's launch default already is)."""
    with _PromptRecorder() as rec:
        code = engine.generate_code_raw(spec, notes, temperature=temperature)
    return code, rec.last


def _teacher_generate(spec: str, notes: list, temperature: float) -> tuple[str, dict]:
    """The think pass (Task 3 fix M4): reuses engine.generate_code_raw's own prompt-
    construction code, composing _PromptRecorder (entered first, so it wraps the real
    _ollama) with _ThinkInjector (entered second, so it wraps the recorder and is the one
    generate_code_raw's call actually reaches first) -- the call chain is
    generate_code_raw -> injected (sets think=True) -> recording (captures
    model/system/prompt/usage) -> the real _ollama. think_rung_available() must already
    have been checked by the caller (see _choose_mode) before this is ever invoked, per
    _ollama()'s own documented contract for a caller passing think=True directly."""
    with _PromptRecorder() as rec, _ThinkInjector():
        code = engine.generate_code_raw(spec, notes, temperature=temperature)
    return code, rec.last


def _regate(m_fluid: dict, build_dir: Path, spec: str) -> dict:
    """Independent re-gating (Task 3 fix H1/H2): see this module's own docstring for why
    fluid_gen._materialize's own gate/facts cannot be trusted beyond a genuine crash.
    `m_fluid["error"]` alone is trusted (a real run_step exception); everything else is
    re-derived here: the STEP is re-inspected directly, "unscored" is returned whenever
    inspect is invalid/raises or a required fact (solids, volume, bbox) is missing, and
    only a fully measured candidate is verified against engine.reconcile_expected's
    `expected` block (fluid's own expected_for plus expected.solids, which fluid never
    sets)."""
    if m_fluid["error"]:
        return {"error": m_fluid["error"], "facts": {}, "gate_hard": [], "gate_spec": [],
               "gate_adv": [], "unscored_reason": None}
    step_path = build_dir / "build.step"
    try:
        insp = engine.run_inspect(step_path)
    except SpecgenAborted:
        raise
    except Exception as e:
        return {"error": None, "facts": {}, "gate_hard": [], "gate_spec": [], "gate_adv": [],
               "unscored_reason": f"inspect raised: {str(e)[:200]}"}
    if not insp.get("valid"):
        return {"error": None, "facts": {}, "gate_hard": [], "gate_spec": [], "gate_adv": [],
               "unscored_reason": "inspect reported the STEP invalid"}
    facts = engine.parse_facts(insp["output"])
    missing = [k for k in ("solids", "volume", "bbox") if not facts.get(k) and facts.get(k) != 0]
    if missing:
        return {"error": None, "facts": facts, "gate_hard": [], "gate_spec": [], "gate_adv": [],
               "unscored_reason": f"inspect did not measure: {', '.join(missing)}"}
    brief: dict = {}
    engine.reconcile_expected(brief, spec)
    hard, notes = engine.verify_expected(facts, brief["expected"], spec=spec)
    return {"error": None, "facts": facts,
           "gate_hard": hard or [],
           "gate_spec": [n for n in (notes or []) if n.startswith("[spec]")],
           "gate_adv": [n for n in (notes or []) if not n.startswith("[spec]")],
           "unscored_reason": None}


def _execute_with_salvage(spec: str, code: str, build_dir: Path, prompt_info: dict,
                          gen_seconds: float) -> list[dict]:
    """Execute through fluid_gen._materialize for its side effects only (the exact
    execute/inspect/render path a real fluid build uses), independently re-gate via
    _regate (Task 3 fix H1/H2) and, on a crash, ONE salvage attempt reusing fluid_gen's
    own diagnose()/_revise_on_repair_rung() -- never reimplemented, so the salvage logic
    itself can never drift from production's. A signal that landed anywhere inside either
    _materialize call (and was swallowed there -- see this module's docstring) is caught
    via _check_abort() immediately after each call returns.

    Returns one or two rows: {code, prompt_info, m, turn, seconds} -- `turn` is "first" or
    "salvage" (Task 3 fix M8). The first row's `seconds` includes the codegen call that
    produced `code` (gen_seconds, passed in) plus this materialize; the salvage row's
    `seconds` is its OWN codegen+materialize time only, so summing every row's `seconds`
    equals this candidate's total wall time exactly once, never double-counted."""
    t0 = time.monotonic()
    m_fluid = fluid_gen._materialize(code, build_dir, spec)
    _check_abort()
    m = _regate(m_fluid, build_dir, spec)
    out = [{"code": code, "prompt_info": prompt_info, "m": m, "turn": "first",
           "seconds": gen_seconds + (time.monotonic() - t0)}]
    if m["error"]:
        t1 = time.monotonic()
        try:
            _, hint = diagnose(m["error"])
            problem = m["error"] + (f"\nRepair hint: {hint}" if hint else "")
            with _PromptRecorder() as rec:
                fixed, _rung = fluid_gen._revise_on_repair_rung(spec, code, problem)
            m2_fluid = fluid_gen._materialize(fixed, build_dir, spec)
            _check_abort()
            m2 = _regate(m2_fluid, build_dir, spec)
            if not m2["error"]:
                m2["salvaged"] = True
            out.append({"code": fixed, "prompt_info": rec.last, "m": m2, "turn": "salvage",
                       "seconds": time.monotonic() - t1})
        except SpecgenAborted:
            raise
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Verdicts, ledger/pair/review row construction
# ---------------------------------------------------------------------------

def _is_good(ok: bool, gate_hard: list, gate_spec: list, band: Optional[str]) -> bool:
    return bool(ok) and not gate_hard and not gate_spec and band in (None, "match")


def _classify(m: dict, reference_stl: Optional[str], build_dir: Path) -> tuple[str, dict]:
    """(verdict, band_info) -- verdict in ("good", "silver", "fail", "unscored", "none").
    band_info is the geom_bands.score_against_reference() dict, or {} when there was no
    reference or nothing could be scored.

    Every default in this function returns "none" or "unscored", never "good" (Task 3
    fix H2: "audit every default: no branch may fall through to good") -- "good" is
    reachable through exactly one path, at the bottom of an explicit chain of checks that
    all must have passed."""
    if m.get("error") is not None:
        return "none", {}
    if m.get("unscored_reason"):
        return "unscored", {}
    facts = m.get("facts") or {}
    if not facts.get("solids") and facts.get("solids") != 0:
        return "unscored", {}
    if facts.get("volume") is None:
        return "unscored", {}
    if not facts.get("bbox"):
        return "unscored", {}
    ok = True
    band_info: dict = {}
    if reference_stl:
        try:
            band_info = score_against_reference(build_dir / "build.step", Path(reference_stl))
        except SpecgenAborted:
            raise
        except Exception as e:
            return "unscored", {"errors": [str(e)[:200]]}
    band = band_info.get("band")
    gate_hard = m.get("gate_hard") or []
    gate_spec = m.get("gate_spec") or []
    if _is_good(ok, gate_hard, gate_spec, band):
        return "good", band_info
    if not gate_hard and gate_spec:
        return "silver", band_info
    if reference_stl and band == "near_miss":
        return "fail", band_info
    return "none", band_info


def _row_is_good(row: dict) -> bool:
    """Recomputes the SAME "good" predicate _classify uses, from what a ledger row
    already stored -- used by the status/pass-rate aggregation, so there is one
    definition of "good" shared by write time and read time. An unscored row is never
    good regardless of what its (necessarily empty) gate/band fields happen to look like
    (Task 3 fix H2 -- this was the exact class of bug the review found: ok=True with
    empty gate lists and band=None used to satisfy _is_good by accident)."""
    if row.get("unscored_reason"):
        return False
    return _is_good(bool(row.get("ok")), row.get("gate_hard") or [],
                    row.get("gate_spec") or [], row.get("band"))


def _persist_build(build_dir: Path, unit_id: str, spec_id: str, label: str) -> str:
    """Only "useful" candidates (good/silver/fail) get their build artifacts copied out of
    the per-candidate temp dir into a durable home under lab/state/builds/ -- everything
    else (including "unscored" and "none") is discarded when the temp dir closes, so a
    night of mostly-rejected candidates does not balloon disk the way
    ~/.openclaw/cad-builds/ already has (2,123 dirs/829MB, a standing open item; harvest
    keeps its own artifacts out of that directory entirely, and prunes its own via
    _prune_builds)."""
    dest = BUILDS_DIR / f"{_safe_id(unit_id)}_{_safe_id(spec_id)}_{_safe_id(str(label))}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("build_source.py", "build.step", "build.stl", "build.png"):
        src = build_dir / name
        if src.is_file():
            (dest / name).write_bytes(src.read_bytes())
    return str(dest)


def _ledger_row(*, unit_id: str, spec_id: str, tier, mode: str, candidate: str,
                temperature: float, m: Optional[dict], band_info: Optional[dict],
                usage: Optional[dict], model: Optional[str], build_dir: Optional[str],
                seconds: Optional[float], error_text: Optional[str] = None,
                unscored_reason: Optional[str] = None) -> dict:
    m = m or {}
    band_info = band_info or {}
    error = error_text if error_text is not None else m.get("error")
    return {
        "ts": _now_utc(),
        "unit_id": unit_id,
        "spec_id": spec_id,
        "tier": tier,
        "arm": _current_arm_alias(),
        "model": model,   # Task 3 fix M3: the model string the recorder actually captured
        "pass": mode,
        "candidate": candidate,
        "temperature": temperature,
        "ok": error is None,
        "gate_hard": m.get("gate_hard") or [],
        "gate_spec": m.get("gate_spec") or [],
        "gate_adv": m.get("gate_adv") or [],
        "band": band_info.get("band"),
        "ref": band_info.get("reference"),
        "chamfer_mm": band_info.get("chamfer_mm"),
        "tokens_in": (usage or {}).get("prompt_tokens"),
        "tokens_out": (usage or {}).get("completion_tokens"),
        "seconds": round(seconds, 2) if seconds is not None else None,
        "build_dir": build_dir,
        "error": error,
        # Task 3 fix H2: "ledger row with the reason", counted separately in status.
        "unscored_reason": unscored_reason or m.get("unscored_reason"),
    }


def _pair_source(mode: str) -> str:
    return "student" if mode == "student" else "teacher:think"


def _good_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict,
                   turn: str) -> dict:
    prompt_info = prompt_info or {}
    return {
        "id": f"p:{row['id']}:{hashlib.sha1(code.encode()).hexdigest()[:12]}",
        "spec_id": row["id"], "spec": row["spec"], "tier": row.get("tier"),
        "group": row.get("group"),
        "source": _pair_source(mode), "kind": "good", "turn": turn,
        "band": band_info.get("band"),
        "arm": _current_arm_alias(), "model": prompt_info.get("model"),
        "temperature": temperature,
        "system_sha1": _store_system(prompt_info.get("system", "")),
        "prompt": prompt_info.get("prompt", ""),
        "code": code, "bad_code": None, "problem": None,
        "facts": m.get("facts") or {},
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _fail_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict,
                   turn: str) -> dict:
    """An owner-reference spec whose candidate near-misses the reference geometry.
    `code` is null: an owner reference is spec.txt + model.step/model.stl -- geometry
    only, never a build123d program -- so there is no verified CORRECT code to pair the
    wrong candidate against (unlike gift_sample.py's corpus rows, which always have one).
    `bad_code`/`problem` still carry the wrong candidate and its geometric diagnosis, same
    shape as gift_sample's GIFT-FAIL rows, so the row remains useful evidence even without
    a training target."""
    prompt_info = prompt_info or {}
    chamfer = band_info.get("chamfer_mm")
    vol = band_info.get("volume_diff_pct")
    problem = (f"near-miss geometry: chamfer {chamfer:.1f}mm from reference, "
              f"volume off {vol}%") if chamfer is not None else "near-miss geometry"
    return {
        "id": f"p:{row['id']}:{hashlib.sha1(code.encode()).hexdigest()[:12]}",
        "spec_id": row["id"], "spec": row["spec"], "tier": row.get("tier"),
        "group": row.get("group"),
        "source": _pair_source(mode), "kind": "fail", "turn": turn,
        "band": band_info.get("band"),
        "arm": _current_arm_alias(), "model": prompt_info.get("model"),
        "temperature": temperature,
        "system_sha1": _store_system(prompt_info.get("system", "")),
        "prompt": prompt_info.get("prompt", ""),
        "code": None, "bad_code": code, "problem": problem,
        "facts": m.get("facts") or {},
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _review_row(row: dict, unit_id: str, mode: str, temperature: float,
                prompt_info: Optional[dict], code: str, m: dict,
                render_path: Optional[str], turn: str) -> dict:
    prompt_info = prompt_info or {}
    notes = (m.get("gate_spec") or []) + (m.get("gate_adv") or [])
    return {
        "id": f"r:{row['id']}:{hashlib.sha1(code.encode()).hexdigest()[:12]}",
        "spec_id": row["id"], "spec": row["spec"], "tier": row.get("tier"),
        "code": code, "notes": notes, "render": render_path, "turn": turn,
        "arm": _current_arm_alias(), "model": prompt_info.get("model"),
        "system_sha1": _store_system(prompt_info.get("system", "")),
        "prompt": prompt_info.get("prompt", ""),
        "temperature": temperature, "pass": mode,
        "ts": _now_utc(), "unit_id": unit_id,
    }


class PairIndex:
    """Cache of pairs.jsonl state for the duration of one unit/once call (Task 3 fix M5):
    read once, updated in memory as pairs are appended, never re-read from disk mid-run.
    Dedup uses the AST-normalised fingerprint (_code_fingerprint, fix L7); tier34-share
    tracks GOOD pairs only (the scheduler's own priority signal, see run_unit)."""

    def __init__(self) -> None:
        self._hashes: dict[str, set] = {}
        self._good_by_tier: dict[str, int] = {}
        self._good_total = 0
        for r in _read_jsonl(PAIRS_FILE):
            code = r.get("code")
            if code:
                self._hashes.setdefault(r.get("spec_id"), set()).add(_code_fingerprint(code))
            if r.get("kind") == "good":
                self._good_total += 1
                t = str(r.get("tier"))
                self._good_by_tier[t] = self._good_by_tier.get(t, 0) + 1

    def has(self, spec_id: str, fingerprint: str) -> bool:
        return fingerprint in self._hashes.get(spec_id, set())

    def add(self, spec_id: str, fingerprint: str, tier) -> None:
        self._hashes.setdefault(spec_id, set()).add(fingerprint)
        self._good_total += 1
        t = str(tier)
        self._good_by_tier[t] = self._good_by_tier.get(t, 0) + 1

    @property
    def good_total(self) -> int:
        return self._good_total

    def good_tier34_share(self) -> float:
        if not self._good_total:
            return 0.0
        tier34 = sum(n for t, n in self._good_by_tier.items() if t in _TIER34)
        return tier34 / self._good_total


# ---------------------------------------------------------------------------
# Per-spec sampling
# ---------------------------------------------------------------------------

def sample_spec(row: dict, cfg: dict, mode: str, unit_id: str, progress: dict,
                pair_index: PairIndex, deadline: Optional[float] = None) -> None:
    """Sample up to cfg["candidates"] candidates for one bank spec, in the given pass
    ("student" thinking-off, "think" thinking-on), verify each, and write a ledger row for
    every one plus a pair/review row where the verdict earns it. Never raises except
    SpecgenAborted (a signal) or _InfraError (the model server itself unreachable), both
    of which propagate straight through so the caller's arm_window() bookend still runs
    its restore step; a candidate aborted or hit by an infra error mid-flight gets NO
    ledger row and its attempt-counter increment is undone (Task 3 fix H3/L3).

    `deadline` (monotonic seconds), when given, is checked before every codegen call and
    before every materialize call inside the per-candidate loop (Task 3 fix M1) -- not
    only between specs. A single already-in-flight model call cannot itself be cut short
    this way (engine.generate_code_raw resolves its own timeout internally and takes no
    override parameter, and editing cad_engine.py is out of scope for this fix), so a unit
    can still run somewhat past its budget on its LAST candidate; this bounds how often
    that happens to at most once per spec, not once per candidate."""
    spec_id, spec, tier = row["id"], row["spec"], row.get("tier")
    reference_stl = row.get("reference_stl")
    entry = progress.setdefault(
        spec_id, {"student_attempts": 0, "teacher_attempts": 0, "pairs": 0})
    max_pairs = cfg["max_pairs_per_spec"]
    if entry.get("pairs", 0) >= max_pairs:
        return
    if _is_contaminated(spec):
        print(f"harvest: spec {spec_id!r} matched a card-suite contamination key/slug at "
              "sample time; refusing (fail closed, L1)", file=sys.stderr)
        return

    notes = engine.retrieval_notes_for(spec)
    temps = cfg.get("temps") or [0.2]
    n = max(1, int(cfg.get("candidates", 1)))
    attempt_key = "student_attempts" if mode == "student" else "teacher_attempts"
    good_found: list[dict] = []
    seen_fps: set = set()

    for i in range(n):
        _check_abort()
        # Stop sampling this spec once it already has enough DISTINCT good candidates --
        # temps are cycled ascending, so whatever is already in good_found is already the
        # lowest-temperature run found so far; a later (higher-temperature) candidate can
        # only ever be dropped by the "prefer lower temperature" rule below, never chosen
        # over it, so there is nothing left to gain by spending more GPU time on this spec.
        if entry.get("pairs", 0) + len(good_found) >= max_pairs:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        temperature = temps[i % len(temps)]
        entry[attempt_key] = entry.get(attempt_key, 0) + 1
        candidate_label = str(i)
        t0 = time.monotonic()
        # The WHOLE candidate body lives inside this try, with ONE outer except that
        # undoes the attempt-counter increment above before re-raising (Task 3 fix H3/
        # L3: an aborted or infra-failed candidate must never count as a student/teacher
        # attempt). A normal codegen failure (`continue`) and a deadline cutoff (`break`)
        # both happen inside the inner try/if below and never reach this except, since
        # neither raises SpecgenAborted/_InfraError.
        try:
            try:
                if mode == "student":
                    code, prompt_info = _student_generate(spec, notes, temperature)
                else:
                    code, prompt_info = _teacher_generate(spec, notes, temperature)
            except SpecgenAborted:
                raise
            except Exception as e:
                if _is_infra_error(e):
                    raise _InfraError(f"model server unreachable: {str(e)[:300]}") from e
                _append_jsonl(LEDGER_FILE, _ledger_row(
                    unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                    candidate=candidate_label, temperature=temperature, m=None,
                    band_info=None, usage=None, model=None, build_dir=None,
                    seconds=time.monotonic() - t0,
                    error_text=f"codegen failed: {str(e)[:400]}"))
                _save_progress(progress)
                continue
            gen_seconds = time.monotonic() - t0
            _check_abort()
            if deadline is not None and time.monotonic() >= deadline:
                # The candidate already generated code but there is no time left to
                # materialize it; drop it rather than starting work we cannot finish,
                # and undo the attempt (it was never actually scored).
                entry[attempt_key] -= 1
                break

            with tempfile.TemporaryDirectory(prefix="harvest_") as td:
                build_dir = Path(td)
                attempts = _execute_with_salvage(spec, code, build_dir, prompt_info,
                                                gen_seconds)
                for attempt in attempts:
                    label = (candidate_label if attempt["turn"] == "first"
                            else f"{candidate_label}-salvage")
                    verdict, band_info = _classify(attempt["m"], reference_stl, build_dir)
                    persisted = (_persist_build(build_dir, unit_id, spec_id, label)
                                if verdict not in ("none", "unscored") else None)
                    usage = (attempt["prompt_info"] or {}).get("usage")
                    model = (attempt["prompt_info"] or {}).get("model")
                    _append_jsonl(LEDGER_FILE, _ledger_row(
                        unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                        candidate=label, temperature=temperature, m=attempt["m"],
                        band_info=band_info, usage=usage, model=model,
                        build_dir=persisted, seconds=attempt["seconds"]))
                    if verdict == "good":
                        fp = _code_fingerprint(attempt["code"])
                        # pair_index.has() only reflects pairs already on disk (from a
                        # previous unit) plus anything added THIS round via .add() below
                        # -- but .add() only runs at the very end of sample_spec, once
                        # the winners are chosen. seen_fps catches a duplicate found
                        # earlier in THIS SAME per-candidate loop, which pair_index alone
                        # would miss until the next spec/unit.
                        if pair_index.has(spec_id, fp) or fp in seen_fps:
                            continue
                        seen_fps.add(fp)
                        good_found.append({**attempt, "temperature": temperature,
                                           "band_info": band_info, "fingerprint": fp})
                    elif verdict == "silver":
                        render = str(Path(persisted) / "build.png") if persisted else None
                        _append_jsonl(REVIEW_FILE, _review_row(
                            row, unit_id, mode, temperature, attempt["prompt_info"],
                            attempt["code"], attempt["m"], render, attempt["turn"]))
                    elif verdict == "fail":
                        _append_jsonl(PAIRS_FILE, _fail_pair_row(
                            row, unit_id, mode, temperature, attempt["prompt_info"],
                            attempt["code"], attempt["m"], band_info, attempt["turn"]))
            _save_progress(progress)
            _check_abort()
        except (SpecgenAborted, _InfraError):
            entry[attempt_key] -= 1
            raise

    # Dedup already enforced above via pair_index; keep at most the remaining slots,
    # preferring lower temperature (Task 3: "at most max_pairs_per_spec distinct codes
    # per spec (dedup by code hash, prefer lower temperature)").
    good_found.sort(key=lambda g: g["temperature"])
    slots = max(0, max_pairs - entry.get("pairs", 0))
    for g in good_found[:slots]:
        _append_jsonl(PAIRS_FILE, _good_pair_row(
            row, unit_id, mode, g["temperature"], g["prompt_info"], g["code"], g["m"],
            g["band_info"], g["turn"]))
        pair_index.add(spec_id, g["fingerprint"], tier)
        entry["pairs"] = entry.get("pairs", 0) + 1
    _save_progress(progress)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _pass_rate(rows: list[dict], key_fn) -> dict:
    counts: dict = {}
    goods: dict = {}
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        counts[k] = counts.get(k, 0) + 1
        if _row_is_good(r):
            goods[k] = goods.get(k, 0) + 1
    return {k: round(goods.get(k, 0) / n, 4) for k, n in counts.items()}


def _pairs_stats() -> dict:
    rows = _read_jsonl(PAIRS_FILE)
    total = len(rows)
    by_tier: dict = {}
    by_source: dict = {}
    good_by_tier: dict = {}
    good_total = 0
    for r in rows:
        by_tier[str(r.get("tier"))] = by_tier.get(str(r.get("tier")), 0) + 1
        by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1
        if r.get("kind") == "good":
            good_total += 1
            good_by_tier[str(r.get("tier"))] = good_by_tier.get(str(r.get("tier")), 0) + 1
    tier34 = sum(n for t, n in by_tier.items() if t in _TIER34)
    good_tier34 = sum(n for t, n in good_by_tier.items() if t in _TIER34)
    return {
        "total": total, "by_tier": by_tier, "by_source": by_source,
        "tier34_share": round(tier34 / total, 4) if total else 0.0,
        # Task 3 fix M6: the Task 6 exit gate (2,000 pairs, 40% tier 3-4) counts GOOD
        # pairs only -- these are the authoritative numbers for that; the ones above are
        # the all-kinds context (good + fail) kept alongside, clearly named.
        "good_total": good_total, "good_by_tier": good_by_tier,
        "good_tier34_share": round(good_tier34 / good_total, 4) if good_total else 0.0,
    }


def _think_pass_status(cfg: dict, teacher_pool_len: int) -> dict:
    """Task 3 ruling: "when unavailable the teacher pass is skipped and the status says
    so." `configured` is whether cad.json's lab.harvest.teacher_passes even lists "think";
    `available` is think_rung_available() itself; `note` is None only when a think unit
    could actually run right now."""
    configured = "think" in (cfg.get("teacher_passes") or [])
    available = configured and think_rung_available()
    note = None
    if not configured:
        note = "\"think\" is not in cad.json's lab.harvest.teacher_passes"
    elif not available:
        note = ("think_rung_available() is False for the active arm (maker disabled, or "
                "its launch args do not use enable_thinking): teacher pass skipped")
    elif teacher_pool_len < 20:
        note = f"only {teacher_pool_len} spec(s) waiting on the teacher pass (needs 20)"
    return {"configured": configured, "available": available,
           "eligible_specs": teacher_pool_len, "note": note}


def _compute_status() -> dict:
    cfg = lab_config()["harvest"]
    bank = specbank.load_bank()
    specs_by_tier: dict = {}
    for r in bank:
        specs_by_tier[str(r.get("tier"))] = specs_by_tier.get(str(r.get("tier")), 0) + 1
    ledger_rows = _read_jsonl(LEDGER_FILE)
    now_local = datetime.now()
    progress = _load_progress()
    _, teacher_pool = _eligible_pools(bank, progress, cfg)
    unscored = sum(1 for r in ledger_rows if r.get("unscored_reason"))
    return {
        "updated": _now_utc(),
        "specs": {"total": len(bank), "by_tier": specs_by_tier},
        "pairs": _pairs_stats(),
        "ledger": {
            "attempts": len(ledger_rows),
            "unscored": unscored,   # Task 3 fix H2: counted separately
            "pass_rate_by_tier": _pass_rate(ledger_rows, lambda r: str(r.get("tier"))),
            "pass_rate_by_pass": _pass_rate(ledger_rows, lambda r: r.get("pass")),
            "last_unit": ledger_rows[-1].get("unit_id") if ledger_rows else None,
        },
        "think_pass": _think_pass_status(cfg, len(teacher_pool)),
        "budget": {
            "hours_today": round(_hours_today(now_local), 2),
            "hours_per_day": cfg["hours_per_day"],
            "window": f"{cfg['night_start']}-{cfg['night_end']}",
            "in_window": _in_night_window(now_local, cfg),
            "day_allowed": cfg["day_allowed"],
        },
        "nights_completed": nights_completed(cfg),
        "timer_active": not PAUSED_FILE.exists(),
    }


def _write_status() -> dict:
    status = _compute_status()
    _atomic_write_json(STATUS_FILE, status)
    return status


# ---------------------------------------------------------------------------
# Unit / once drivers
# ---------------------------------------------------------------------------

def run_unit(cfg: dict) -> dict:
    unit_id = _now_utc()
    deadline = time.monotonic() + cfg["unit_minutes"] * 60
    bank = specbank.load_bank()
    progress = _load_progress()
    student_pool, teacher_pool = _eligible_pools(bank, progress, cfg)
    mode = _choose_mode(teacher_pool, cfg)
    pool = teacher_pool if mode == "think" else student_pool
    pair_index = PairIndex()
    processed = 0
    if pool:
        prefer_tier34 = pair_index.good_tier34_share() < 0.40
        ordered = _order_specs(pool, progress, prefer_tier34)
        for row in ordered:
            _check_abort()   # between specs (Task 3 fix H3)
            if time.monotonic() >= deadline:
                break
            sample_spec(row, cfg, mode, unit_id, progress, pair_index, deadline=deadline)
            processed += 1
    pruned = _prune_builds(int(cfg.get("keep_builds", 500)))
    status = _write_status()
    return {"unit_id": unit_id, "mode": mode, "specs_processed": processed,
           "builds_pruned": pruned, "status": status}


def run_once(spec_id: str, cfg: dict) -> dict:
    unit_id = _now_utc()
    bank = specbank.load_bank()
    row = next((r for r in bank if r.get("id") == spec_id), None)
    if row is None:
        raise SystemExit(f"harvest --once: no spec with id {spec_id!r} in the bank")
    progress = _load_progress()
    entry = progress.get(spec_id, {})
    # Task 3 fix L4: honour teacher_passes the same way _choose_mode does, rather than
    # only checking think_rung_available().
    mode = "student"
    if (entry.get("student_attempts", 0) >= 2
            and "think" in (cfg.get("teacher_passes") or [])
            and think_rung_available()):
        mode = "think"
    pair_index = PairIndex()
    sample_spec(row, cfg, mode, unit_id, progress, pair_index)
    status = _write_status()
    return {"unit_id": unit_id, "spec_id": spec_id, "mode": mode, "status": status}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_in_window(arm: Optional[str], body) -> int:
    """Shared try/finally shape between --unit and --once: the abort/infra exceptions are
    caught here ONLY to report them and set a non-zero exit code -- arm_window()'s own
    finally has already run by the time this except fires, since it lives inside the
    `with` block."""
    rc, reason = 0, None
    try:
        with arm_window(arm):
            body()
    except SpecgenAborted as e:
        rc, reason = 1, str(e)
    except _InfraError as e:
        rc, reason = 1, f"infrastructure error: {e}"
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        reason = str(e.code) if not isinstance(e.code, int) else None
    except Exception as e:
        rc, reason = 1, f"unexpected error: {e}"
    if reason:
        print(f"harvest abort: {reason}", file=sys.stderr)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", action="store_true", help="run one gated unit")
    ap.add_argument("--once", action="store_true",
                    help="run exactly one spec now, no gates (smoke/tests)")
    ap.add_argument("--spec-id", default=None, help="required with --once")
    ap.add_argument("--status", action="store_true", help="print the status JSON and exit")
    ap.add_argument("--check-gate", action="store_true",
                    help="cheap pre-flight for the timer, no GPU window needed: "
                         "exit 0 = go, exit 3 = skip (reason on stderr)")
    ap.add_argument("--arm", default=None,
                    help="maker arm for this run (default: cad.json's maker block)")
    ap.add_argument("--i-know-the-gpu-is-free", action="store_true",
                    help="run outside a GPU window (only when the GPU was freed by hand)")
    a = ap.parse_args()

    if a.status:
        print(json.dumps(_compute_status(), indent=2))
        return 0

    if a.check_gate:
        reason = _check_gate()
        if reason:
            print(f"harvest --check-gate: skip ({reason})", file=sys.stderr)
            return 3
        print("harvest --check-gate: go", file=sys.stderr)
        return 0

    if not a.unit and not (a.once and a.spec_id):
        ap.error("pass --unit, --once --spec-id ID, --check-gate, or --status")

    # Before ANYTHING that touches a model or a service (fix pattern shared with
    # lab/specgen.py): no arm switch, no signal handlers, no cad.json read, no model call.
    ship.require_gpu_window(a, GPU_WINDOW_HINT)

    pin_refusal = _check_code_model_pin()
    if pin_refusal:
        print(f"harvest: refusing ({pin_refusal})", file=sys.stderr)
        return 1

    if a.unit:
        cfg = lab_config()["harvest"]
        skip_reason = _unit_gate(cfg)
        if skip_reason:
            print(f"harvest: unit skipped ({skip_reason})", file=sys.stderr)
            _write_status()
            return 0
        return _run_in_window(a.arm, lambda: run_unit(lab_config()["harvest"]))

    return _run_in_window(a.arm, lambda: run_once(a.spec_id, lab_config()["harvest"]))


if __name__ == "__main__":
    sys.exit(main())

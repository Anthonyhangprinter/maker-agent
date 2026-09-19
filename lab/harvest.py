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
import math
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
# Fix round 2, Section A: gate-clean candidates that were NOT confirmed this round (no
# reference match, no agreeing partner yet) -- never a pair until a later unit's
# agreeing sample promotes one, see CandidateIndex/_resolve_agreement.
CANDIDATES_FILE = STATE_DIR / "candidates.jsonl"
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
    closed), never as clear -- printed distinctly from an actual key/slug match (a real
    UnicodeDecodeError from a stale locale, found via a real --unit smoke, previously
    looked identical in the logs to "this spec really did collide with a card suite",
    which cost real time to diagnose)."""
    try:
        suite_keys, suite_slugs = default_contamination_sets()
    except Exception as e:
        print(f"harvest: default_contamination_sets() failed ({e}); refusing this spec "
              "fail-closed rather than risking a contaminated pair (this is a "
              "computation failure, not a confirmed match)", file=sys.stderr)
        return True
    key = hc._key(spec)
    if key in suite_keys:
        return True
    return hc._slug(spec, 40) in suite_slugs


def _is_infra_error(e: Exception) -> bool:
    """See _InfraError's docstring. Fix round 2, D2: catching bare OSError was too wide
    -- FileNotFoundError, PermissionError and other local filesystem OSErrors are not
    "the model server is unreachable", and treating them as an infra error would abort
    the whole unit over what might be a genuine local bug, hiding it as a false "server
    down" report. Only connection-level failures count: urllib.error.URLError (itself an
    OSError subclass, as are ConnectionError and socket.timeout/TimeoutError in modern
    Python -- listed explicitly anyway so a reviewer does not have to know that fact to
    trust this check), ConnectionError and its subclasses, and socket.timeout/
    TimeoutError."""
    return isinstance(e, (urllib.error.URLError, ConnectionError,
                         socket.timeout, TimeoutError))


def _prune_builds(keep: int) -> int:
    """Retention cap for lab/state/builds/ (Task 3 fix L9): mtime-based, never the
    newest, same discipline as cad_engine's own KEEP_BUILDS fix (a name-based prune had
    been deleting the wrong directories there). A pruned build dir may belong to an
    already-recorded pair -- fair game, since the pair row holds the code itself and the
    build dir is only supplementary render/mesh evidence. Returns the count removed.
    Fix round 2, D3: `keep` is floored at 1 -- a misconfigured or non-positive
    `keep_builds` must never let this prune every directory including the one a caller
    could still be reading from."""
    keep = max(1, keep)
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
# Geometric signature + independent agreement (fix round 2, Section A). A reviewer read
# the 7 tier-3 "good" pairs from the first real run by eye and found 2 that are WRONG
# geometry the deterministic gate passed: t:teacher-batch2:V064 (a cutter sheared the
# whole top off a hollow enclosure, and its "through holes" are blind) and
# t:teacher-batch2:V066 (one sample's cylinder bores the wrong axis, making a slot
# instead of a circular hole through a thin divider -- the OTHER sample of the SAME spec
# is correct). The gate measures structure and numbers, not whether a named feature
# exists. Real facts for both, taken from lab/state/ledger.jsonl / pairs.jsonl before
# this fix round's state reset, are fixtured at
# tests/fixtures/harvest_agreement_fixtures.json.
#
# Ruling: gate-clean is no longer enough for "good". A gate-clean, measured candidate
# becomes a confirmed "good" pair only when EITHER a reference exists and the band is
# "match" (as before -- an owner-reference bank row is ground truth, no cross-checking
# needed), OR at least one OTHER gate-clean candidate for the same spec_id, with a
# DIFFERENT code fingerprint, has the same geometric signature. "Other candidate" also
# includes an already-confirmed pair sitting on disk from an earlier unit (PairIndex
# carries signatures for exactly this purpose) -- a new sample that matches previously-
# vetted truth is just as confirmed as two new samples agreeing with each other.
# ---------------------------------------------------------------------------

_SIGNATURE_FACT_KEYS = ("solids", "faces", "cyl_faces", "cone_faces")


def signature(facts: dict, cfg: Optional[dict] = None) -> Optional[dict]:
    """A candidate's geometric fingerprint for cross-candidate agreement, or None when
    `facts` is missing a required field or carries scripts/inspect's own "couldn't
    classify" sentinel (a negative cyl_faces/cone_faces) -- fail closed: an incomplete
    signature can never be compared, never silently "agrees" by omission (two candidates
    whose classification both failed are NOT thereby the same geometry). `cfg` is the
    cad.json lab.harvest.agreement block (bore_round_mm); defaults match
    cad_v5.config._LAB_DEFAULTS when omitted, which every real caller passes."""
    cfg = cfg or {}
    bore_round = cfg.get("bore_round_mm", 0.01)
    if any(facts.get(k) is None for k in _SIGNATURE_FACT_KEYS):
        return None
    if facts.get("cyl_faces", -1) < 0 or facts.get("cone_faces", -1) < 0:
        return None
    if facts.get("volume") is None:
        return None
    bbox = facts.get("bbox")
    if not bbox or len(bbox) != 3:
        return None
    ndigits = max(0, -int(round(math.log10(bore_round)))) if bore_round else 2
    bores = facts.get("bores") or []
    return {
        "solids": facts["solids"], "faces": facts["faces"],
        "cyl_faces": facts["cyl_faces"], "cone_faces": facts["cone_faces"],
        "volume": float(facts["volume"]),
        "bbox_sorted": sorted(round(float(x), 6) for x in bbox),
        "bores_sorted": sorted(round(float(b), ndigits) for b in bores),
    }


def signatures_agree(a: Optional[dict], b: Optional[dict], cfg: Optional[dict] = None) -> bool:
    """True when two signatures describe the same physical geometry within the
    configured tolerances (cad.json lab.harvest.agreement: volume_tol_pct, bbox_tol_mm).
    Missing signatures never agree (fail closed, see signature()'s own docstring). Field
    order matters for which one a caller sees fail first on a real mismatch: solids,
    then total face count, then cylindrical/conical face counts, then bore diameters,
    then bbox, then volume -- topology before size, since two genuinely different builds
    of the same spec (V066's slot-vs-hole pair) usually diverge in face counts long
    before their volumes drift enough to notice."""
    if a is None or b is None:
        return False
    if a["solids"] != b["solids"] or a["faces"] != b["faces"]:
        return False
    if a["cyl_faces"] != b["cyl_faces"] or a["cone_faces"] != b["cone_faces"]:
        return False
    if a["bores_sorted"] != b["bores_sorted"]:
        return False
    if len(a["bbox_sorted"]) != len(b["bbox_sorted"]):
        return False
    cfg = cfg or {}
    bbox_tol = cfg.get("bbox_tol_mm", 0.05)
    for x, y in zip(a["bbox_sorted"], b["bbox_sorted"]):
        if abs(x - y) > bbox_tol:
            return False
    avg_vol = (a["volume"] + b["volume"]) / 2.0
    vol_tol_pct = cfg.get("volume_tol_pct", 0.05)
    if avg_vol <= 0:
        return a["volume"] == b["volume"]
    if abs(a["volume"] - b["volume"]) / avg_vol * 100.0 > vol_tol_pct:
        return False
    return True


def _cluster_by_signature(pool: list[dict], cfg: dict) -> list[list[dict]]:
    """Connected components of `pool` under signatures_agree (fix round 2, Section A):
    union-find over pairwise agreement, not a single-representative greedy scan, so
    membership is transitive by construction and never depends on sampling/insertion
    order. Each pool item must carry a "signature" key (possibly None -- signatures_agree
    already treats that as never-agreeing)."""
    n = len(pool)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if signatures_agree(pool[i].get("signature"), pool[j].get("signature"), cfg):
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, item in enumerate(pool):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


def resolve_agreement(pool: list[dict], cfg: dict) -> tuple[list[dict], str]:
    """(winners, tag) for a spec's confirmable candidate pool (fix round 2, Section A).
    `tag` is one of:
      "agreement"   a decisive cluster was found -- `winners` are its members, MINUS any
                    synthetic existing_anchor entries (an already-confirmed pair on disk
                    is trusted evidence, never itself something to "promote" again). A
                    cluster containing an existing_anchor wins outright regardless of
                    size (it agrees with already-vetted truth); otherwise the largest
                    cluster wins only when it has >=2 members AND is strictly larger than
                    every other cluster (Task 3 ruling, verbatim).
      "unconfirmed" exactly one candidate (or one cluster of size 1, no anchor) -- a lone
                    gate-clean candidate with nobody to agree with yet.
      "split"       2+ clusters exist, none anchored, and none wins outright (a tie, or
                    every candidate disagrees with every other) -- nobody is confirmed.
    An empty pool is "unconfirmed" with no winners (nothing to resolve)."""
    if not pool:
        return [], "unconfirmed"
    if len(pool) == 1 and not pool[0].get("existing_anchor"):
        return [], "unconfirmed"
    clusters = _cluster_by_signature(pool, cfg)
    anchored = [c for c in clusters if any(item.get("existing_anchor") for item in c)]
    if anchored:
        winners = [item for c in anchored for item in c if not item.get("existing_anchor")]
        return winners, "agreement"
    if len(clusters) == 1:
        only = clusters[0]
        return (only, "agreement") if len(only) >= 2 else ([], "unconfirmed")
    sizes = sorted((len(c) for c in clusters), reverse=True)
    if sizes[0] >= 2 and sizes[0] > sizes[1]:
        return max(clusters, key=len), "agreement"
    return [], "split"


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


_DEFAULT_ATTEMPT_CAPS = {"student": 2, "teacher": 5}


def _eligible_pools(bank: list[dict], progress: dict, cfg: dict) -> tuple[list, list]:
    """(student_pool, teacher_pool): a spec with `pairs >= max_pairs_per_spec` is excluded
    from both (done); a spec with fewer than attempt_caps["student"] student attempts is
    student-pool; one with more, and still short of its pair quota, graduates to the
    teacher pool. Fix round 2, Section C: a spec that has ALSO used its full
    attempt_caps["teacher"] think-pass attempts and is still short is EXHAUSTED -- dropped
    from both pools (see _exhausted_specs for the matching status/reporting view). One
    full think-pass round samples up to candidates_tier34 candidates in one call, so a
    default teacher cap of 5 gives a spec exactly one such round before it is written off;
    its unconfirmed candidates are not deleted, only no longer resampled (see
    lab/state/candidates.jsonl). Pool membership is independent of whether a think pass is
    actually available this run -- that decision (and the >=20-spec threshold) belongs to
    the caller."""
    max_pairs = cfg["max_pairs_per_spec"]
    caps = cfg.get("attempt_caps") or _DEFAULT_ATTEMPT_CAPS
    student, teacher = [], []
    for row in bank:
        entry = progress.get(row["id"], {})
        if entry.get("pairs", 0) >= max_pairs:
            continue
        if entry.get("teacher_attempts", 0) >= caps.get("teacher", _DEFAULT_ATTEMPT_CAPS["teacher"]):
            continue   # exhausted -- see _exhausted_specs
        if entry.get("student_attempts", 0) >= caps.get("student", _DEFAULT_ATTEMPT_CAPS["student"]):
            teacher.append(row)
        else:
            student.append(row)
    return student, teacher


def _exhausted_specs(bank: list[dict], progress: dict, cfg: dict) -> list[dict]:
    """Specs that have used their full teacher attempt budget (attempt_caps["teacher"])
    and still have fewer than max_pairs_per_spec confirmed pairs (fix round 2, Section
    C) -- excluded from _eligible_pools's own two pools, reported separately here so
    --status can show how many specs the bank has given up on for the night without
    silently losing count of them."""
    max_pairs = cfg["max_pairs_per_spec"]
    caps = cfg.get("attempt_caps") or _DEFAULT_ATTEMPT_CAPS
    teacher_cap = caps.get("teacher", _DEFAULT_ATTEMPT_CAPS["teacher"])
    out = []
    for row in bank:
        entry = progress.get(row["id"], {})
        if entry.get("pairs", 0) >= max_pairs:
            continue
        if entry.get("teacher_attempts", 0) >= teacher_cap:
            out.append(row)
    return out


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


# ---------------------------------------------------------------------------
# Strict spec checks (fix round 2, Section B) -- harvest-local, fail closed, additional
# to and independent of the engine's own deterministic gate (engine.verify_expected,
# run inside _regate above). V064 (a hollow enclosure sheared 2mm off its stated height
# by a mis-sized lip cutter, with its stated through holes measuring blind) passed the
# engine gate's own axis tolerance (max(1, 5%)mm = 2.75mm here) and is still visibly the
# wrong part -- these checks exist to catch exactly that class of "numerically close
# enough, structurally the wrong shape" defect. Both are pure functions of (spec, facts)
# so they are unit-testable directly against the real fixture rows, no build required.
# ---------------------------------------------------------------------------

# "180x130x55mm", "180 x 130 x 55 mm", "180mm x 130mm x 55mm", "180 by 130 by 55mm" --
# with or without spaces/units on the first two numbers, but the LAST number must carry
# an explicit "mm" (same discipline as cad_engine._DIMS3_SPEC_RE: a bare, unitless
# "180x130x55" out of context is too ambiguous to enforce). Extends _DIMS3_SPEC_RE
# (which only covers the no-space "NxNxN mm" form) with the "by" phrasing and optional
# per-number units the brief calls for; cad_engine has no such regex to reuse as-is.
_ENVELOPE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:x|by)\s*"
    r"(\d+(?:\.\d+)?)\s*mm\b", re.I)

_THROUGH_HOLE_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
# "four 3.2mm through holes", "6 through-holes", "a 3.2mm through hole" -- the count
# word (digit or number word up to twelve, "a"/"an" as one) directly followed by an
# OPTIONAL numeric-dimension filler ("3.2mm ") then "through[- ]hole(s)". The filler is
# deliberately narrow (a number+unit, not an arbitrary word): an earlier version allowed
# up to 3 arbitrary filler words and let a spec's leading "a" (from "a bracket with 6
# through-holes") jump all the way to a LATER "through-holes" over the real count "6" in
# between -- found via a real fixture (V064's OWN "a" would otherwise misfire on other
# specs' unrelated leading articles).
_THROUGH_HOLES_RE = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
    r"(?:\d+(?:\.\d+)?\s*(?:mm|cm)?\s+)?through[- ]?holes?\b", re.I)


def _spec_envelope_dims_mm(spec: str) -> Optional[list[float]]:
    """The ONE stated envelope AxBxC in mm the spec text names, or None when the spec
    states no envelope, states more than one (too ambiguous to enforce -- same discipline
    as cad_engine._spec_axis_dims's own multi-match handling), or is an assembly spec
    (engine._ASSEMBLY_SPEC_RE), whose bbox is emergent, not specified."""
    spec = spec or ""
    if engine._ASSEMBLY_SPEC_RE.search(spec):
        return None
    matches = _ENVELOPE_RE.findall(spec)
    if len(matches) != 1:
        return None
    return [float(x) for x in matches[0]]


def strict_envelope_check(spec: str, facts: dict, tol_mm: float = 0.2) -> Optional[str]:
    """None when the spec states no single envelope (or is an assembly spec), or the
    measured bbox matches the stated one within tol_mm axis-for-axis after sorting both
    (permutation-free, same reasoning as the engine's own axis check -- the spec does not
    say which axis is which). Otherwise a "strict_envelope: ..." reason string (fix round
    2, Section B1) naming the mismatch. `facts` is assumed already known to carry a valid
    bbox (the caller only runs this on a fully-measured candidate)."""
    dims = _spec_envelope_dims_mm(spec)
    if dims is None:
        return None
    bbox = facts.get("bbox")
    if not bbox or len(bbox) != 3:
        return "strict_envelope: spec states an envelope but no bbox was measured"
    stated = sorted(dims)
    measured = sorted(float(x) for x in bbox)
    worst = max(abs(s - m) for s, m in zip(stated, measured))
    if worst > tol_mm:
        return (f"strict_envelope: stated {dims[0]:g}x{dims[1]:g}x{dims[2]:g}mm vs "
                f"measured bbox {bbox[0]:g}x{bbox[1]:g}x{bbox[2]:g}mm (sorted {stated} "
                f"vs {measured}, worst axis off by {worst:.2f}mm > {tol_mm}mm)")
    return None


def _spec_through_hole_count(spec: str) -> Optional[int]:
    m = _THROUGH_HOLES_RE.search(spec or "")
    if not m:
        return None
    token = m.group(1).lower()
    if token in _THROUGH_HOLE_COUNT_WORDS:
        return _THROUGH_HOLE_COUNT_WORDS[token]
    return int(token) if token.isdigit() else None


def strict_through_holes_check(spec: str, facts: dict) -> Optional[str]:
    """None when the spec states no through-hole count, or the measured through-hole
    count meets it. When the spec DOES state a count but `facts` carries no usable
    through-hole measurement (scripts/inspect's own -1 "analysis unavailable" sentinel,
    or the key missing entirely), the check is UNSCORED, NOT PASS (fix round 2, Section
    B2, verbatim): a false accept is unacceptable here, so uncertainty blocks "good" the
    same way an outright short count does, distinguished only by the reason prefix."""
    stated = _spec_through_hole_count(spec)
    if stated is None:
        return None
    measured = facts.get("through_holes")
    if measured is None or measured < 0:
        return ("unscored:strict_through_holes: spec states "
                f"{stated} through hole(s) but through-hole classification is "
                "unavailable for this candidate")
    if measured < stated:
        return (f"strict_through_holes: spec states {stated} through hole(s), measured "
                f"only {measured}")
    return None


def _strict_check_notes(spec: str, facts: dict, cfg: dict) -> list[str]:
    """Both strict checks, formatted as "[strict] ..." notes ready to extend an `m`
    dict's gate_spec list (fix round 2): appending a non-empty result here makes
    _classify() fall through to "silver" exactly the way an engine [spec] finding does,
    reusing that path rather than adding a parallel verdict branch. Empty when neither
    check fires."""
    strict_cfg = cfg.get("strict") or {}
    tol_mm = strict_cfg.get("envelope_tol_mm", 0.2)
    notes = []
    for reason in (strict_envelope_check(spec, facts, tol_mm),
                  strict_through_holes_check(spec, facts)):
        if reason:
            notes.append(f"[strict] {reason}")
    return notes


def _regate_and_strict(m_fluid: dict, build_dir: Path, spec: str, cfg: dict) -> dict:
    """_regate() plus the strict spec checks above, applied only to a fully-measured
    candidate (no error, no unscored_reason) -- a candidate that already failed to
    execute or measure has nothing for these checks to examine. Returns a NEW dict (never
    mutates `_regate`'s own return value) with gate_spec extended by any strict finding,
    so a strict violation blocks "good" via the exact same _is_good/_classify path an
    engine [spec] finding already uses."""
    m = _regate(m_fluid, build_dir, spec)
    if m["error"] or m["unscored_reason"]:
        return m
    facts = m.get("facts") or {}
    if not facts.get("bbox") or facts.get("volume") is None:
        return m   # not enough measured to run a strict check meaningfully either
    strict_notes = _strict_check_notes(spec, facts, cfg)
    if not strict_notes:
        return m
    return {**m, "gate_spec": list(m.get("gate_spec") or []) + strict_notes}


def _execute_with_salvage(spec: str, code: str, build_dir: Path, prompt_info: dict,
                          gen_seconds: float, cfg: dict,
                          deadline: Optional[float] = None) -> list[dict]:
    """Execute through fluid_gen._materialize for its side effects only (the exact
    execute/inspect/render path a real fluid build uses), independently re-gate via
    _regate_and_strict (Task 3 fix H1/H2, fix round 2 Section B) and, on a crash, ONE
    salvage attempt reusing fluid_gen's own diagnose()/_revise_on_repair_rung() -- never
    reimplemented, so the salvage logic itself can never drift from production's. A
    signal that landed anywhere inside either _materialize call (and was swallowed there
    -- see this module's docstring) is caught via _check_abort() immediately after each
    call returns.

    `deadline` (monotonic seconds), when given and already passed, skips the salvage
    attempt entirely (fix round 2, D1): no salvage codegen call starts once the unit's
    time budget is spent, matching the same rule sample_spec already applies before every
    OTHER codegen call.

    Returns one or two rows: {code, prompt_info, m, turn, seconds} -- `turn` is "first" or
    "salvage" (Task 3 fix M8). The first row's `seconds` includes the codegen call that
    produced `code` (gen_seconds, passed in) plus this materialize; the salvage row's
    `seconds` is its OWN codegen+materialize time only, so summing every row's `seconds`
    equals this candidate's total wall time exactly once, never double-counted."""
    t0 = time.monotonic()
    m_fluid = fluid_gen._materialize(code, build_dir, spec)
    _check_abort()
    m = _regate_and_strict(m_fluid, build_dir, spec, cfg)
    out = [{"code": code, "prompt_info": prompt_info, "m": m, "turn": "first",
           "seconds": gen_seconds + (time.monotonic() - t0)}]
    if m["error"] and not (deadline is not None and time.monotonic() >= deadline):
        t1 = time.monotonic()
        try:
            _, hint = diagnose(m["error"])
            problem = m["error"] + (f"\nRepair hint: {hint}" if hint else "")
            with _PromptRecorder() as rec:
                fixed, _rung = fluid_gen._revise_on_repair_rung(spec, code, problem)
            m2_fluid = fluid_gen._materialize(fixed, build_dir, spec)
            _check_abort()
            m2 = _regate_and_strict(m2_fluid, build_dir, spec, cfg)
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
    """A ledger row counts as "good" for pass-rate stats only when it was CONFIRMED (fix
    round 2, Section A: "pass rates ... computed on CONFIRMED goods") -- gate-clean
    (_classify's own "good" verdict) is necessary but no longer sufficient, since a lone
    gate-clean candidate with no partner is "unconfirmed", not good (the exact class of
    defect this fix round exists to catch: V064/V066 were both gate-clean and wrong).
    `agreement` is set at ledger-write time to "reference" or "agreement" only when
    _classify said "good" AND a confirmation (a reference-band match, or a matching
    OTHER candidate/existing pair) was actually found -- every other row, including an
    unscored one (Task 3 fix H2's original regression), carries None here and is never
    good."""
    return row.get("agreement") in ("reference", "agreement")


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
                unscored_reason: Optional[str] = None,
                fingerprint: Optional[str] = None,
                agreement: Optional[str] = None) -> dict:
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
        # Fix round 2, Section A: the AST-normalised code fingerprint (when code was
        # generated for this attempt), so a ledger row can be cross-referenced against
        # pairs.jsonl/candidates.jsonl after the fact without storing the code itself
        # twice.
        "fingerprint": fingerprint,
        # Fix round 2, Section A: None unless this candidate was gate-clean AND
        # confirmed ("reference" or "agreement") -- or, for a gate-clean candidate that
        # was NOT confirmed this round, "unconfirmed" or "split" (see resolve_agreement).
        # _row_is_good reads exactly this field.
        "agreement": agreement,
    }


def _pair_source(mode: str) -> str:
    return "student" if mode == "student" else "teacher:think"


def _good_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict,
                   turn: str, confirmed_by: str) -> dict:
    """`confirmed_by` is "reference" (an owner-reference band match) or "agreement" (a
    matching other candidate or already-confirmed pair) -- fix round 2, Section A: every
    good pair now carries the reason it was trusted, never implicit. `signature` is
    stored too, so PairIndex can offer this pair as agreement evidence to a FUTURE
    candidate without re-deriving it from `facts`."""
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
        # signature()'s own default bore_round_mm (0.01) matches
        # cad_v5.config._LAB_DEFAULTS["harvest"]["agreement"]["bore_round_mm"] -- no cfg
        # threaded through here (this function has many call sites already); an owner
        # who customises that tuning knob away from the default would see a stored
        # signature rounded slightly differently from a live signatures_agree() check,
        # a narrow, documented edge case rather than a correctness bug for the default
        # configuration this ships with.
        "signature": signature(m.get("facts") or {}),
        "confirmed_by": confirmed_by,
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _candidate_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, fingerprint: str,
                   turn: str, agreement: str) -> dict:
    """A gate-clean candidate that was NOT confirmed this round (fix round 2, Section A:
    "a full row (same fields as a pair row plus signature)"). Same shape as a good pair
    row, distinguished by id prefix "c:" (never collides with a real "p:" pair id) and
    kind="candidate"; `agreement` is "unconfirmed" or "split". Never written to
    PAIRS_FILE -- only CANDIDATES_FILE."""
    base = _good_pair_row(row, unit_id, mode, temperature, prompt_info, code, m, {},
                          turn, confirmed_by=None)
    base["id"] = f"c:{row['id']}:{base['id'].rsplit(':', 1)[-1]}"
    base["kind"] = "candidate"
    base["fingerprint"] = fingerprint
    base["agreement"] = agreement
    base["status"] = "unconfirmed"
    return base


def _promote_candidate_row(cand: dict, unit_id: str, confirmed_by: str) -> dict:
    """Turn an already-recorded candidates.jsonl row into a real pairs.jsonl "good" row
    (fix round 2, Section A: cross-unit confirmation) without re-deriving anything the
    candidate row already carries -- its system_sha1 already points at the stored prompt
    text, and there is no raw system string left on the candidate row to re-hash (that
    would also silently rewrite lab/state/systems/ with a duplicate under a fresh
    _store_system call, which is exactly the redundancy that function exists to avoid)."""
    return {
        "id": f"p:{cand['spec_id']}:{cand['fingerprint'][:12]}",
        "spec_id": cand["spec_id"], "spec": cand["spec"], "tier": cand.get("tier"),
        "group": cand.get("group"),
        "source": cand.get("source"), "kind": "good", "turn": cand.get("turn"),
        "band": cand.get("band"),
        "arm": cand.get("arm"), "model": cand.get("model"),
        "temperature": cand.get("temperature"),
        "system_sha1": cand.get("system_sha1"),
        "prompt": cand.get("prompt", ""),
        "code": cand.get("code"), "bad_code": None, "problem": None,
        "facts": cand.get("facts") or {},
        "signature": cand.get("signature"),
        "confirmed_by": confirmed_by,
        "verified": cand.get("verified") or {},
        "ts": _now_utc(), "unit_id": unit_id,
        "promoted_from": cand.get("id"),
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
    tracks GOOD pairs only (the scheduler's own priority signal, see run_unit). Fix round
    2, Section A: also indexes each GOOD pair's stored "signature" per spec_id, so a NEW
    candidate can be confirmed by agreeing with an ALREADY-confirmed pair from an earlier
    unit, not only with a sibling sampled in the same round."""

    def __init__(self) -> None:
        self._hashes: dict[str, set] = {}
        self._good_by_tier: dict[str, int] = {}
        self._good_total = 0
        self._sigs: dict[str, list] = {}
        for r in _read_jsonl(PAIRS_FILE):
            code = r.get("code")
            if code:
                self._hashes.setdefault(r.get("spec_id"), set()).add(_code_fingerprint(code))
            if r.get("kind") == "good":
                self._good_total += 1
                t = str(r.get("tier"))
                self._good_by_tier[t] = self._good_by_tier.get(t, 0) + 1
                sig = r.get("signature")
                if sig:
                    self._sigs.setdefault(r.get("spec_id"), []).append(sig)

    def has(self, spec_id: str, fingerprint: str) -> bool:
        return fingerprint in self._hashes.get(spec_id, set())

    def add(self, spec_id: str, fingerprint: str, tier, signature: Optional[dict] = None) -> None:
        self._hashes.setdefault(spec_id, set()).add(fingerprint)
        self._good_total += 1
        t = str(tier)
        self._good_by_tier[t] = self._good_by_tier.get(t, 0) + 1
        if signature:
            self._sigs.setdefault(spec_id, []).append(signature)

    def signatures_for(self, spec_id: str) -> list:
        return list(self._sigs.get(spec_id, []))

    @property
    def good_total(self) -> int:
        return self._good_total

    def good_tier34_share(self) -> float:
        if not self._good_total:
            return 0.0
        tier34 = sum(n for t, n in self._good_by_tier.items() if t in _TIER34)
        return tier34 / self._good_total


class CandidateIndex:
    """Cache of candidates.jsonl state for the duration of one unit/once call (fix round
    2, Section A, same discipline as PairIndex): read once, folded by id (last write
    wins, so a "promoted" status row written after the original unconfirmed row correctly
    supersedes it in memory without ever rewriting or deleting the earlier line -- this
    is an append-only log). `pair_index` provides the belt-and-braces exclusion: a
    candidate whose fingerprint already has a real pair on disk is never offered again as
    "still unconfirmed", which is what actually makes a crash between promoting two
    candidates safe to re-run (see run_unit/sample_spec's own docstrings) -- the "promoted"
    status marker is written for human/tool legibility of candidates.jsonl, not as the
    correctness mechanism itself."""

    def __init__(self, pair_index: "PairIndex") -> None:
        self._pair_index = pair_index
        by_id: dict[str, dict] = {}
        for r in _read_jsonl(CANDIDATES_FILE):
            rid = r.get("id")
            if rid:
                by_id[rid] = r
        self._by_spec: dict[str, list] = {}
        for r in by_id.values():
            if r.get("status") == "promoted":
                continue
            fp = r.get("fingerprint")
            spec_id = r.get("spec_id")
            if fp and self._pair_index.has(spec_id, fp):
                continue
            self._by_spec.setdefault(spec_id, []).append(r)

    def unconfirmed_for(self, spec_id: str) -> list:
        return list(self._by_spec.get(spec_id, []))


# ---------------------------------------------------------------------------
# Per-spec sampling
# ---------------------------------------------------------------------------

def sample_spec(row: dict, cfg: dict, mode: str, unit_id: str, progress: dict,
                pair_index: PairIndex, deadline: Optional[float] = None) -> None:
    """Sample up to cfg["candidates"] (or cfg["candidates_tier34"] for a tier 3/4 spec,
    fix round 2 Section C) candidates for one bank spec, in the given pass ("student"
    thinking-off, "think" thinking-on), verify each, and write a ledger row for every
    one plus a pair/review/candidate row where the verdict earns it. Never raises except
    SpecgenAborted (a signal) or _InfraError (the model server itself unreachable), both
    of which propagate straight through so the caller's arm_window() bookend still runs
    its restore step; a candidate aborted or hit by an infra error mid-flight gets NO
    ledger row and its attempt-counter increment is undone (Task 3 fix H3/L3).

    Fix round 2, Section A: a gate-clean, measured candidate with no reference match is
    no longer immediately "good" -- it is held as a CANDIDATE until either an existing
    confirmed pair on disk, a candidate carried over from an earlier unit
    (CandidateIndex), or another candidate sampled THIS round agrees with it on geometric
    signature (see resolve_agreement). Reference-band-matched candidates are unaffected
    (still immediately good, as before): an owner-reference row is ground truth.

    `deadline` (monotonic seconds), when given, is checked before every codegen call and
    before every materialize call inside the per-candidate loop (Task 3 fix M1), and
    before the salvage codegen call too (fix round 2, D1, via _execute_with_salvage's own
    deadline param) -- not only between specs. A single already-in-flight model call
    cannot itself be cut short this way (engine.generate_code_raw resolves its own
    timeout internally and takes no override parameter, and editing cad_engine.py is out
    of scope for this fix), so a unit can still run somewhat past its budget on its LAST
    candidate; this bounds how often that happens to at most once per spec, not once per
    candidate."""
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
    is_tier34 = str(tier) in _TIER34
    if is_tier34:
        temps = cfg.get("temps_tier34") or cfg.get("temps") or [0.2]
        n = max(1, int(cfg.get("candidates_tier34", cfg.get("candidates", 3))))
    else:
        temps = cfg.get("temps") or [0.2]
        n = max(1, int(cfg.get("candidates", 1)))
    attempt_key = "student_attempts" if mode == "student" else "teacher_attempts"
    agreement_cfg = cfg.get("agreement") or {}

    candidate_index = CandidateIndex(pair_index)
    carried = [{"origin": "carried", "cand_row": c, "fingerprint": c.get("fingerprint"),
               "signature": c.get("signature"), "temperature": c.get("temperature") or 0.0}
              for c in candidate_index.unconfirmed_for(spec_id)]
    existing_anchors = [{"origin": "existing", "existing_anchor": True, "signature": sig,
                         "fingerprint": None, "temperature": 0.0}
                        for sig in pair_index.signatures_for(spec_id)]
    clean_candidates: list[dict] = []
    # seen_fps also seeded from carried candidates: a NEW sample identical to one already
    # sitting unconfirmed on disk is the SAME code, not a second opinion, and must not be
    # allowed to "confirm" it (Section A: "a DIFFERENT code fingerprint").
    seen_fps: set = {c["fingerprint"] for c in carried if c.get("fingerprint")}
    resolved = {"done": False}

    def _pool() -> list[dict]:
        return existing_anchors + carried + clean_candidates

    def _write_new_ledger(c: dict, agreement: Optional[str]) -> None:
        _append_jsonl(LEDGER_FILE, _ledger_row(
            unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
            candidate=c["candidate_label"], temperature=c["temperature"], m=c["m"],
            band_info={}, usage=c["usage"], model=c["model"], build_dir=c["build_dir"],
            seconds=c["seconds"], fingerprint=c["fingerprint"], agreement=agreement))

    def _finalize() -> bool:
        """Resolve the current pool (existing pairs + carried unconfirmed candidates +
        this round's new clean candidates) and commit the outcome: a winning cluster's
        NEW members become good pairs (ledger row written now, for the first time -- a
        clean candidate's ledger row is deferred exactly until this point, since its
        final agreement status is not known any earlier) up to the remaining
        max_pairs_per_spec slots (lower temperature preferred); a winning carried member
        is promoted (a real pair written from its already-recorded candidates.jsonl row,
        marked "promoted" there so it is never re-promoted -- see CandidateIndex); every
        other new candidate this round gets its (first) ledger row now too, tagged
        "unconfirmed" or "split"; an untouched carried candidate is left exactly as it
        already is on disk. A no-op (returns False) when no NEW candidate emerged this
        round (nothing to report -- a spec with only stale carried candidates is not
        re-litigated without new evidence, and the caller's own finally block uses this
        return value to skip a needless progress.json save on a round that changed
        nothing, preserving the pre-existing "an aborted first candidate never even
        creates progress.json" guarantee)."""
        if resolved["done"]:
            return False
        resolved["done"] = True
        if not clean_candidates:
            return False
        pool = _pool()
        winners, tag = resolve_agreement(pool, agreement_cfg)
        winner_ids = {id(w) for w in winners}
        if tag == "agreement":
            winners_sorted = sorted(winners, key=lambda c: c.get("temperature", 0.0))
            slots = max(0, max_pairs - entry.get("pairs", 0))
            chosen_ids = {id(c) for c in winners_sorted[:slots]}
            for c in winners_sorted:
                chosen = id(c) in chosen_ids
                if c["origin"] == "new":
                    _write_new_ledger(c, "agreement")
                    if chosen:
                        pair_row = _good_pair_row(row, unit_id, mode, c["temperature"],
                                                  c["prompt_info"], c["code"], c["m"], {},
                                                  c["turn"], confirmed_by="agreement")
                        _append_jsonl(PAIRS_FILE, pair_row)
                        pair_index.add(spec_id, c["fingerprint"], tier,
                                       pair_row.get("signature"))
                        entry["pairs"] = entry.get("pairs", 0) + 1
                    else:
                        _append_jsonl(CANDIDATES_FILE, _candidate_row(
                            row, unit_id, mode, c["temperature"], c["prompt_info"],
                            c["code"], c["m"], c["fingerprint"], c["turn"],
                            agreement="agreement"))
                elif chosen:   # carried, and this round has a slot for it
                    pair_row = _promote_candidate_row(c["cand_row"], unit_id, "agreement")
                    _append_jsonl(PAIRS_FILE, pair_row)
                    pair_index.add(spec_id, c["fingerprint"], tier,
                                   pair_row.get("signature"))
                    entry["pairs"] = entry.get("pairs", 0) + 1
                    _append_jsonl(CANDIDATES_FILE, {
                        **c["cand_row"], "status": "promoted",
                        "promoted_unit": unit_id, "promoted_ts": _now_utc()})
                # a carried, non-chosen winner is left exactly as it is on disk.
        # Every NEW candidate not already handled above (a losing-cluster member when
        # tag=="agreement", or every new candidate at all when tag is "unconfirmed"/
        # "split") gets its first-ever ledger + candidate row now.
        loser_tag = "split" if tag == "split" else "unconfirmed"
        for c in clean_candidates:
            if id(c) in winner_ids:
                continue
            _write_new_ledger(c, loser_tag)
            _append_jsonl(CANDIDATES_FILE, _candidate_row(
                row, unit_id, mode, c["temperature"], c["prompt_info"], c["code"],
                c["m"], c["fingerprint"], c["turn"], agreement=loser_tag))
        return True

    # The whole loop lives inside a try/finally so an abort or infra error propagating
    # out (SpecgenAborted/_InfraError, re-raised below after undoing the interrupted
    # candidate's own attempt count) still flushes whatever clean_candidates an EARLIER,
    # successfully-completed candidate this round already added -- without this, a
    # signal landing on candidate 2 would silently lose candidate 1's gate-clean result
    # forever, since its ledger/candidate row is deferred until _finalize() runs
    # (fix round 2: "every candidate ... gets one row in the ledger" must still hold
    # when the unit is interrupted, not only when it completes normally).
    try:
        for i in range(n):
            _check_abort()
            # Fix round 2, Section C: only an ALREADY-CONFIRMED pair count stops
            # sampling early -- a merely gate-clean, unconfirmed candidate must not (the
            # old early stop fired on exactly that, which is how a wrong candidate went
            # unchallenged).
            if entry.get("pairs", 0) >= max_pairs:
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
                                                    gen_seconds, cfg, deadline=deadline)
                    for attempt in attempts:
                        label = (candidate_label if attempt["turn"] == "first"
                                else f"{candidate_label}-salvage")
                        verdict, band_info = _classify(attempt["m"], reference_stl, build_dir)
                        persisted = (_persist_build(build_dir, unit_id, spec_id, label)
                                    if verdict not in ("none", "unscored") else None)
                        usage = (attempt["prompt_info"] or {}).get("usage")
                        model = (attempt["prompt_info"] or {}).get("model")
                        fp = (_code_fingerprint(attempt["code"])
                             if verdict != "unscored" else None)

                        if verdict == "good" and band_info.get("band") == "match":
                            # An owner-reference band match confirms on its own, exactly as
                            # before this fix round -- no cross-candidate agreement needed.
                            if pair_index.has(spec_id, fp) or fp in seen_fps:
                                _append_jsonl(LEDGER_FILE, _ledger_row(
                                    unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                    candidate=label, temperature=temperature, m=attempt["m"],
                                    band_info=band_info, usage=usage, model=model,
                                    build_dir=persisted, seconds=attempt["seconds"],
                                    fingerprint=fp, agreement=None))
                                continue
                            seen_fps.add(fp)
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement="reference"))
                            if entry.get("pairs", 0) < max_pairs:
                                pair_row = _good_pair_row(
                                    row, unit_id, mode, temperature, attempt["prompt_info"],
                                    attempt["code"], attempt["m"], band_info, attempt["turn"],
                                    confirmed_by="reference")
                                _append_jsonl(PAIRS_FILE, pair_row)
                                pair_index.add(spec_id, fp, tier, pair_row.get("signature"))
                                entry["pairs"] = entry.get("pairs", 0) + 1
                        elif verdict == "good":
                            # Gate-clean, no reference -- held for agreement (fix round 2,
                            # Section A). Its ledger row is written only once _finalize()
                            # knows this candidate's outcome.
                            if pair_index.has(spec_id, fp) or fp in seen_fps:
                                _append_jsonl(LEDGER_FILE, _ledger_row(
                                    unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                    candidate=label, temperature=temperature, m=attempt["m"],
                                    band_info=band_info, usage=usage, model=model,
                                    build_dir=persisted, seconds=attempt["seconds"],
                                    fingerprint=fp, agreement=None))
                                continue
                            seen_fps.add(fp)
                            clean_candidates.append({
                                "origin": "new", "code": attempt["code"],
                                "prompt_info": attempt["prompt_info"], "m": attempt["m"],
                                "temperature": temperature, "turn": attempt["turn"],
                                "fingerprint": fp,
                                "signature": signature(attempt["m"].get("facts") or {}),
                                "candidate_label": label, "usage": usage, "model": model,
                                "build_dir": persisted, "seconds": attempt["seconds"],
                            })
                            # Stop sampling once a CONFIRMED quota is reached (Section C: the
                            # old early stop fired on merely-unconfirmed goods). Re-resolve
                            # the whole pool fresh -- cheap at this scale, never order-
                            # dependent (see resolve_agreement/_cluster_by_signature).
                            winners_now, tag_now = resolve_agreement(_pool(), agreement_cfg)
                            if (tag_now == "agreement"
                                    and entry.get("pairs", 0) + len(winners_now) >= max_pairs):
                                _finalize()
                        elif verdict == "silver":
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement=None))
                            render = str(Path(persisted) / "build.png") if persisted else None
                            _append_jsonl(REVIEW_FILE, _review_row(
                                row, unit_id, mode, temperature, attempt["prompt_info"],
                                attempt["code"], attempt["m"], render, attempt["turn"]))
                        elif verdict == "fail":
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement=None))
                            _append_jsonl(PAIRS_FILE, _fail_pair_row(
                                row, unit_id, mode, temperature, attempt["prompt_info"],
                                attempt["code"], attempt["m"], band_info, attempt["turn"]))
                        else:   # "none" or "unscored"
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement=None))
                _save_progress(progress)
                _check_abort()
                if resolved["done"]:
                    break
            except (SpecgenAborted, _InfraError):
                entry[attempt_key] -= 1
                raise
    finally:
        if _finalize():
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


def _confirmation_status(ledger_rows: list[dict], bank: list[dict], progress: dict,
                         cfg: dict) -> dict:
    """Fix round 2, Section A: "Status gets unconfirmed, confirmed_by: {reference,
    agreement}, and split_specs counts" -- an aggregate view of how the confirmation
    layer is doing, alongside (not replacing) the per-pair confirmed_by field and the
    per-ledger-row agreement field this same fix round adds.
    `unconfirmed_candidates`: rows in candidates.jsonl not marked "promoted" (still
    waiting on a confirming partner, or permanently stuck if their spec is exhausted).
    `confirmed_by`: how many GOOD pairs on disk were confirmed each way.
    `split_specs`: distinct spec_ids with at least one ledger row recording a "split"
    this fix round found (2+ disagreeing clusters, nobody confirmed) -- a running total,
    not just this unit's, since a split spec needs a human's attention eventually.
    `exhausted_specs`: specs that used their full teacher attempt budget and are still
    short of max_pairs_per_spec (see _exhausted_specs)."""
    candidate_rows = _read_jsonl(CANDIDATES_FILE)
    latest_by_id: dict[str, dict] = {}
    for r in candidate_rows:
        rid = r.get("id")
        if rid:
            latest_by_id[rid] = r
    unconfirmed_candidates = sum(1 for r in latest_by_id.values()
                                if r.get("status") != "promoted")
    confirmed_by = {"reference": 0, "agreement": 0}
    for r in _read_jsonl(PAIRS_FILE):
        if r.get("kind") == "good" and r.get("confirmed_by") in confirmed_by:
            confirmed_by[r["confirmed_by"]] += 1
    split_specs = len({r.get("spec_id") for r in ledger_rows
                       if r.get("agreement") == "split" and r.get("spec_id")})
    return {
        "unconfirmed_candidates": unconfirmed_candidates,
        "confirmed_by": confirmed_by,
        "split_specs": split_specs,
        "exhausted_specs": len(_exhausted_specs(bank, progress, cfg)),
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
        "confirmation": _confirmation_status(ledger_rows, bank, progress, cfg),
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
    ap.add_argument("--unit-minutes", type=float, default=None,
                    help="override lab.harvest.unit_minutes for THIS run only (manual/"
                         "smoke use -- never persisted to cad.json). A temp "
                         "CAD_CONFIG_FILE would redirect scripts/arms.py's own writes "
                         "away from the real cad.json, which a manual --unit smoke must "
                         "not do, so this is the supported way to bound a manual run.")
    ap.add_argument("--allow-day", action="store_true",
                    help="override lab.harvest.day_allowed=true for THIS run only "
                         "(manual/smoke use, same rationale as --unit-minutes)")
    ap.add_argument("--i-am-the-owner", action="store_true",
                    help="required, together with an interactive TTY on stdin, to use "
                         "--unit-minutes/--allow-day (fix round 2, D4) -- the timer path "
                         "(lab/harvest_unit.sh) passes neither flag and never needs this")
    a = ap.parse_args()

    if a.unit_minutes is not None or a.allow_day:
        # Fix round 2, D4: loud on every use (these override the night-window/GPU-hour
        # budget gates a nightly, unattended run relies on), and refused outside an
        # interactive session unless the caller explicitly claims ownership -- the timer
        # path (lab/harvest_unit.sh) passes neither flag, so this never affects it.
        print("harvest: --unit-minutes/--allow-day override the night-window/budget "
              "gates -- manual/smoke use only, never the timer path.", file=sys.stderr)
        if not (sys.stdin.isatty() or a.i_am_the_owner):
            print("harvest: refusing --unit-minutes/--allow-day without an interactive "
                  "TTY on stdin or --i-am-the-owner", file=sys.stderr)
            return 1

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
        if a.unit_minutes is not None:
            cfg["unit_minutes"] = a.unit_minutes
        if a.allow_day:
            cfg["day_allowed"] = True
        skip_reason = _unit_gate(cfg)
        if skip_reason:
            print(f"harvest: unit skipped ({skip_reason})", file=sys.stderr)
            _write_status()
            return 0
        # `cfg` (with any --unit-minutes/--allow-day override already applied) is
        # captured by this closure rather than re-reading lab_config() fresh here -- a
        # second fresh read would silently drop the CLI override the gate check above
        # was just evaluated against.
        return _run_in_window(a.arm, lambda: run_unit(cfg))

    return _run_in_window(a.arm, lambda: run_once(a.spec_id, lab_config()["harvest"]))


if __name__ == "__main__":
    sys.exit(main())

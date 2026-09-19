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

**confirm_strength (fix round 3, H2).** In plain words: two samples from ONE model
agreeing with each other is evidence against sampling NOISE (a fluke bad turn), but it is
NOT evidence against a shared MISCONCEPTION -- if the model always misreads a spec the
same way, every sample it produces will make the same mistake and will happily "confirm"
itself. `confirm_strength` records how much weight that agreement can actually bear:
"reference" (an owner-reference band match -- ground truth, no model involved),
"cross_pass" (the agreeing set spans both a student-pass and a think-pass sample -- two
different generation conditions independently landed on the same geometry, which is
real corroboration), or "same_pass" (every agreeing sample came from the same pass --
still useful, but exactly the weaker kind of evidence described above). The compiler
(Task 4) and any human audit should weigh a "same_pass" pair more skeptically than a
"cross_pass" or "reference" one, not treat all three as equally proven.

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
`flock`, which now (Task 3a, merged onto this branch 2026-09-19) waits up to
`GPU_WINDOW_LOCK_WAIT_SEC` seconds (default 3600; `lab/harvest_unit.sh` sets 120 for the
timer's own tick) before giving up, exiting 75 (EX_TEMPFAIL) on a still-busy lock.
`lab/gpu_window.sh` stays the real arbiter; this probe is only a cheap "worth trying"
signal), and exits 0 (go) or 3 (skip, reason on stderr).
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
import builtins as _builtins_mod
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import socket
import sys
import tempfile
import time
import tokenize
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
# Fix round 3, H2b part 2: an append-only log of confirm_strength upgrades for pairs
# already on disk -- see gate_version()/_confirm_strength_for_cluster's own docstrings
# for why a pair's row is never rewritten in place.
UPGRADES_FILE = STATE_DIR / "upgrades.jsonl"
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


_BUILTIN_NAMES = frozenset(dir(_builtins_mod))


class _BindingCollector(ast.NodeVisitor):
    """First pass of the identifier normaliser (fix round 3, H2a): collects every LOCALLY
    BOUND name in the order it is first bound (textual/traversal order), so the renamer
    below can assign each one a canonical `_v<i>` placeholder. "Locally bound" means an
    `ast.Name` in Store context (an assignment target, a for-loop/comprehension/with-as
    target, an augmented assignment target), a function/lambda parameter (`ast.arg`), or
    an `except ... as name` binding -- deliberately NOT an imported name (imports never
    produce a Store-context Name), an attribute (`foo.bar`'s `bar` is a plain string, not
    a Name node), or a keyword-argument name (`Box(length=10)`'s `length` is also a plain
    string on `ast.keyword`, never a Name node) -- those three exclusions fall out of only
    ever looking at Name/arg/ExceptHandler nodes, no special-casing needed. Builtins
    (`sum`, `list`, ...) are excluded explicitly even if locally reassigned, so a script
    that shadows one is not treated as introducing a fresh local identity for it."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self._seen: set[str] = set()

    def _bind(self, name: str) -> None:
        if name in _BUILTIN_NAMES or name in self._seen:
            return
        self._seen.add(name)
        self.order.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._bind(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)


class _IdentifierAndLiteralNormaliser(ast.NodeTransformer):
    """Second pass (fix round 3, H2a): renames every Name/arg/ExceptHandler binding found
    by `_BindingCollector` to its canonical `_v<i>` placeholder (a Name in LOAD context
    that refers to one of these is renamed too, so a use site tracks its own definition),
    and normalises every non-bool numeric Constant to `float(value)` so `5` and `5.0`
    fingerprint identically (`isinstance(value, bool)` is checked FIRST -- `bool` is a
    subclass of `int` in Python, and True/False must never be coerced to 1.0/0.0 here)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._mapping:
            node.id = self._mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self._mapping:
            node.arg = self._mapping[node.arg]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if node.name and node.name in self._mapping:
            node.name = self._mapping[node.name]
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, bool):
            return node
        if isinstance(node.value, (int, float)):
            node.value = float(node.value)
        return node


def _normalized_ast_dump(code: str) -> str:
    """Raises on unparseable code (the caller falls back to a raw hash); never raises
    otherwise. Renaming is alpha-order-of-first-binding, so two candidates that differ
    ONLY in which literal identifier they happened to pick for the same role produce the
    identical dump."""
    tree = ast.parse(code)
    collector = _BindingCollector()
    collector.visit(tree)
    mapping = {name: f"_v{i}" for i, name in enumerate(collector.order)}
    tree = _IdentifierAndLiteralNormaliser(mapping).visit(tree)
    return ast.dump(tree)


def _code_fingerprint(code: str) -> str:
    """Identifier- and literal-normalised dedup key (fix round 3, H2a, floor of the old
    Task 3 fix L7): two candidates differing only in comments/whitespace, or ONLY in
    which name they gave the same locally-bound value, or ONLY in `5` vs `5.0`, collapse
    to the same fingerprint via `_normalized_ast_dump`. A candidate that restructures the
    computation (an extra intermediate assignment, a different statement count, a
    different formula) does NOT collapse -- that is a real difference, not a rename.
    Falls back to the raw sha1 of the source text when the code does not even parse
    (should not happen for a candidate that already executed successfully, but this must
    never raise)."""
    try:
        return hashlib.sha1(_normalized_ast_dump(code).encode()).hexdigest()
    except Exception:
        return hashlib.sha1(code.encode()).hexdigest()


_SIMILARITY_SKIP_TOKENS = frozenset({
    tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
    tokenize.ENCODING, tokenize.ENDMARKER,
})


def _similarity_tokens(code: str) -> list[str]:
    """Tokenises `code` for code_similarity (fix round 3, H2b): drops comments, blank-line
    NL tokens, and INDENT/DEDENT/ENCODING/ENDMARKER noise, keeping every other token's
    exact string -- a renamed identifier still shows up as a real token-level difference
    here (unlike `_code_fingerprint`'s identifier-blind equality test), which is exactly
    what makes this a SIMILARITY score rather than an equality test. Returns [] when the
    code does not even tokenize (never raises)."""
    try:
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        return [t.string for t in toks if t.type not in _SIMILARITY_SKIP_TOKENS]
    except Exception:
        return []


def code_similarity(code: str, others: list[str]) -> float:
    """Fix round 3, H2b: the max difflib.SequenceMatcher ratio between `code` and every
    OTHER code string in its confirming agreement set, computed on the tokenised source
    (not the identifier-normalised one -- this number is meant to show a human how close
    two INDEPENDENT programs really are, comments/whitespace aside, and a renamed
    variable is a real, visible difference for that purpose). 0.0 when `others` is empty
    (nothing to compare against) or `code` fails to tokenize. Rounded to 3 places."""
    my_tokens = _similarity_tokens(code)
    if not others or not my_tokens:
        return 0.0
    best = 0.0
    for other in others:
        if not other:
            continue
        other_tokens = _similarity_tokens(other)
        if not other_tokens:
            continue
        ratio = difflib.SequenceMatcher(None, my_tokens, other_tokens).ratio()
        best = max(best, ratio)
    return round(best, 3)


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


_GATE_VERSION_BASE = "gv1"


def gate_version(cfg: dict) -> str:
    """A short, stable label for the EFFECTIVE gate configuration in force right now
    (fix round 3, M3): a hand-bumped base string (`_GATE_VERSION_BASE` -- bump this BY
    HAND whenever the CHECKS THEMSELVES change, e.g. a new strict check is added or an
    existing one's logic changes; the config hash below only tracks TUNING, not logic)
    plus an 8-hex-char hash of the `agreement`/`strict` tuning blocks actually in effect.
    Stamped on every candidates/pairs/ledger row so a row written under an earlier gate
    configuration can be told apart from one written under today's -- see
    `_promotion_recheck_ok` and `CandidateIndex`'s own gate_version filtering, and
    `_confirmation_status`'s `stale_candidates` count."""
    payload = json.dumps({"agreement": cfg.get("agreement") or {},
                          "strict": cfg.get("strict") or {}}, sort_keys=True)
    return f"{_GATE_VERSION_BASE}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


def _promotion_recheck_ok(cand_row: dict, cfg: dict) -> Optional[str]:
    """Re-validates a CARRIED candidate's stored facts/spec against TODAY's strict
    checks and contamination sets (fix round 3, M3) before it is allowed to confirm
    anything or be promoted itself. A carried row on disk may be days old; the strict
    checks or the card-suite contamination list can change underneath it, and a stale
    "gate-clean" verdict must be re-proven, not grandfathered in. Returns None when it
    still holds up today, else a short reason string. Deliberately run BEFORE the
    candidate ever enters this round's confirmation pool (not only at the moment it
    would be written to pairs.jsonl): a candidate that fails this re-check must "confirm
    nobody" (Task 3 ruling, verbatim), and the only way to guarantee that is to keep it
    out of the pool in the first place -- checking only at the final promotion step would
    let an already-invalid carried candidate still act as the agreeing partner that
    confirms some OTHER (new) candidate this round, before its own invalidity is ever
    noticed."""
    spec = cand_row.get("spec") or ""
    if _is_contaminated(spec):
        return "contaminated at promotion re-check"
    facts = cand_row.get("facts") or {}
    notes = _strict_check_notes(spec, facts, cfg)
    if notes:
        return "; ".join(notes)
    return None


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
    # Fix round 3, LOW2: a zero or negative volume NEVER agrees with anything, including
    # another zero/negative volume -- the old `avg_vol <= 0: return a==b` branch let two
    # candidates that both measured (broken) zero volume "agree" with each other on the
    # strength of that shared brokenness, which is exactly backwards: a zero/negative
    # volume means the geometry measurement itself is untrustworthy, not that it matches.
    if a["volume"] <= 0 or b["volume"] <= 0:
        return False
    avg_vol = (a["volume"] + b["volume"]) / 2.0
    vol_tol_pct = cfg.get("volume_tol_pct", 0.05)
    if abs(a["volume"] - b["volume"]) / avg_vol * 100.0 > vol_tol_pct:
        return False
    return True


def _signature_sort_key(sig: Optional[dict]) -> tuple:
    """A total, deterministic ordering key for a (possibly None/empty) signature dict,
    used only to make `_cluster_by_signature`'s processing order independent of the
    caller's list order (fix round 3, M2). The leading element (0 vs 1) means a
    missing/empty signature never compares equal-shaped with a real one, so the two
    branches never collide even though their tuples have different lengths."""
    if not sig:
        return (0,)
    return (1, sig.get("solids"), sig.get("faces"), sig.get("cyl_faces"),
           sig.get("cone_faces"), tuple(sig.get("bores_sorted") or ()),
           tuple(sig.get("bbox_sorted") or ()), sig.get("volume"))


def _pool_item_stable_key(item: dict) -> tuple:
    """The full stable sort key for one pool item (fix round 3, M2): temperature first
    (the natural "which sample was this" axis for new/carried candidates), then the
    fingerprint (distinguishes same-temperature siblings), then the signature itself
    (distinguishes anchors, which carry no fingerprint), then origin as a final
    tie-break. Depends ONLY on the item's own fields -- never on its position in
    whatever list the caller happened to pass, which is what makes clustering below
    order-independent."""
    return (item.get("temperature", 0.0), item.get("fingerprint") or "",
           _signature_sort_key(item.get("signature")), item.get("origin", ""))


def _cluster_by_signature(pool: list[dict], cfg: dict) -> list[list[dict]]:
    """COMPLETE-link clusters of `pool` under signatures_agree (fix round 3, M2,
    replacing fix round 2's union-find): a candidate joins an existing cluster only when
    it agrees with EVERY member already in that cluster, not merely one of them --
    single-link (transitive) clustering let a tolerance CHAIN (A agrees with B, B agrees
    with C) fuse A and C into one cluster even when A and C do NOT themselves agree (the
    reviewer's counter-example: bbox 100.00/100.05/100.10mm, volume
    1000.00/1000.25/1000.50 -- each neighbouring pair is within tolerance, the two ends
    are not; single-link produced one 3-member cluster, which is wrong).

    Deterministic and order-independent: pool items are first sorted by a stable key
    derived only from each item's OWN fields (`_pool_item_stable_key`), never from its
    position in the caller's list, and then assigned to clusters in that fixed order. A
    candidate eligible to join more than one existing cluster joins the LARGEST one;
    ties are broken by the smallest stable key among that cluster's own members (again a
    property of the clusters' contents, not of processing order)."""
    n = len(pool)
    order = sorted(range(n), key=lambda i: _pool_item_stable_key(pool[i]))
    clusters: list[list[int]] = []

    for i in order:
        sig_i = pool[i].get("signature")
        eligible = [ci for ci, members in enumerate(clusters)
                   if all(signatures_agree(sig_i, pool[m].get("signature"), cfg)
                         for m in members)]
        if not eligible:
            clusters.append([i])
            continue
        best = min(eligible, key=lambda ci: (
            -len(clusters[ci]),
            min(_pool_item_stable_key(pool[m]) for m in clusters[ci])))
        clusters[best].append(i)

    return [[pool[i] for i in members] for members in clusters]


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
        if winners:
            return winners, "agreement"
        # Fix round 3, LOW1: the anchor's own cluster gained no new member -- every other
        # candidate this round disagreed with it (or there were none). Only the second
        # case ("no other candidate existed at all") is genuinely "unconfirmed"; the
        # first is a real disagreement and must be counted as "split", not silently
        # reported the same way as "nobody to compare against yet".
        if all(item.get("existing_anchor") for item in pool):
            return [], "unconfirmed"
        return [], "split"
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
    instant and NEVER waits, unlike lab/gpu_window.sh's own flock -- see the module
    docstring's own note on that wait (GPU_WINDOW_LOCK_WAIT_SEC, default 3600, exit 75 on
    a still-busy lock) -- it exists purely so --check-gate can decide "worth trying" a
    real GPU window without paying for one. It is not a second locking scheme: nothing
    here ever holds the lock past the probe itself, and `--unit` inside a real window
    never calls this (by the time it runs, gpu_window.sh already holds the lock as this
    process's own ancestor)."""
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


_DEFAULT_TIER_WEIGHTS = {"1": 1, "2": 3, "3": 4, "4": 1}


def _live_candidate_spec_ids() -> set:
    """spec_ids with at least one CANDIDATES_FILE row that is not yet terminal (not
    "promoted" or "rejected_at_promotion") -- Task 3c's "cheapest to confirm" scheduling
    signal: a live unconfirmed candidate is already sitting on disk waiting for an
    agreeing partner, so sampling that spec again might confirm it outright rather than
    needing agreement from scratch. Same fold-by-id-keep-last discipline as
    CandidateIndex (an append-only log; a later status row supersedes the original)."""
    by_id: dict = {}
    for r in _read_jsonl(CANDIDATES_FILE):
        rid = r.get("id")
        if rid:
            by_id[rid] = r
    return {r.get("spec_id") for r in by_id.values()
           if r.get("status") not in ("promoted", "rejected_at_promotion")}


def _tier_weight(tier_weights: dict, tier: str) -> float:
    try:
        w = float((tier_weights or {}).get(tier, 0))
    except (TypeError, ValueError):
        w = 0.0
    return w if w > 0 else 0.0


def _intra_tier_key(row: dict, progress: dict, carried_ids: set) -> tuple:
    """(source_bucket, status_bucket): the sort key that decides where a spec falls
    WITHIN its own tier (Task 3c, replacing "tier 3-4 always first"), before the seeded
    shuffle in _order_specs breaks ties inside one bucket.

    source_bucket: 0 for anything but a model-written `specgen` spec (teacher suites,
    owner references), 1 for `specgen` -- the first real unit's finding was that the
    scheduler started on exactly the specs the model mostly cannot build (source
    "specgen"), so those now pay LAST inside a tier, never first.

    status_bucket: 0 when a live unconfirmed candidate already sits on disk for this
    spec (carried_ids, see _live_candidate_spec_ids) -- cheapest to confirm; 1 when the
    spec has never been sampled (student_attempts == teacher_attempts == 0); 2 otherwise
    -- a "cold" spec, meaning a previous round was sampled and ended with nothing live
    to confirm (this is DERIVED, not a stored flag: a spec with attempts and no live
    candidate is, by construction, exactly a spec whose last round(s) never landed on a
    confirmable candidate)."""
    source_bucket = 1 if row.get("source") == "specgen" else 0
    entry = progress.get(row["id"], {})
    attempts = entry.get("student_attempts", 0) + entry.get("teacher_attempts", 0)
    if row["id"] in carried_ids:
        status_bucket = 0
    elif attempts == 0:
        status_bucket = 1
    else:
        status_bucket = 2
    return (source_bucket, status_bucket)


def _seeded_rng(cfg: dict, today: str) -> random.Random:
    """A deterministic RNG from `lab.harvest.seed` (default 1) combined with a calendar
    date: two _order_specs() calls on the SAME date reproduce the same tie-break order
    (useful for re-running a smoke test), while the next day's units do not repeat it
    verbatim -- see _order_specs's own docstring."""
    payload = f"{cfg.get('seed', 1)}:{today}"
    digest = hashlib.sha1(payload.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _order_specs(pool: list[dict], progress: dict, cfg: dict,
                 today: Optional[str] = None) -> list[dict]:
    """A weighted round-robin over tiers (Task 3c, replacing "tier 3-4 first while the
    pairs' tier34 share is under 0.40" -- with zero pairs that condition never stops
    being true, so the old scheduler started every single round on the hardest,
    mostly model-written specs). `cfg["tier_weights"]` (default
    cad_v5.config._LAB_DEFAULTS: {"1": 1, "2": 3, "3": 4, "4": 1}) sets each tier's
    share of the unit; tiers with no eligible spec, or a non-positive/missing weight,
    are skipped rather than stalling the interleave. The 40% tier 3-4 GOOD-pair share
    this replaced stays visible in `_pairs_stats()`'s own `good_tier34_share` as a
    reported goal, never again as a scheduling condition.

    Within one tier, specs are grouped by `_intra_tier_key` (non-specgen before specgen;
    inside that, a carried unconfirmed candidate first, then never-tried, then cold
    last) and groups are emitted in that key's ascending order. Ties WITHIN one group
    are broken by a seeded shuffle (`_seeded_rng`, keyed on `cfg["seed"]` and `today`,
    default today's calendar date) -- each group is sorted by spec id first so the
    shuffle's result depends only on (seed, date, the group's own membership), never on
    the caller's incoming `pool` order, which is what makes "one unit's order is
    reproducible" true regardless of how the pool was assembled upstream. The seed in
    effect is printed to stderr once per call."""
    if not pool:
        return []
    today = today or datetime.now().date().isoformat()
    rng = _seeded_rng(cfg, today)
    print(f"harvest: spec order seed={cfg.get('seed', 1)!r} date={today}", file=sys.stderr)
    carried_ids = _live_candidate_spec_ids()
    weights = cfg.get("tier_weights") or _DEFAULT_TIER_WEIGHTS

    by_tier: dict[str, list] = {}
    for row in pool:
        by_tier.setdefault(str(row.get("tier")), []).append(row)

    ordered_by_tier: dict[str, list] = {}
    for tier, rows in by_tier.items():
        groups: dict[tuple, list] = {}
        for row in rows:
            groups.setdefault(_intra_tier_key(row, progress, carried_ids), []).append(row)
        tier_order: list = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda r: r["id"])
            rng.shuffle(group)
            tier_order.extend(group)
        ordered_by_tier[tier] = tier_order

    idx = {t: 0 for t in ordered_by_tier}
    counts = {t: 0 for t in ordered_by_tier}
    total = sum(len(v) for v in ordered_by_tier.values())
    order: list[dict] = []
    while len(order) < total:
        active = [t for t in ordered_by_tier
                 if idx[t] < len(ordered_by_tier[t]) and _tier_weight(weights, t) > 0]
        if not active:
            # Every tier with specs left has a zero/missing configured weight -- fall
            # back to a fixed (tier id ascending) order rather than stalling forever.
            active = sorted(t for t in ordered_by_tier if idx[t] < len(ordered_by_tier[t]))
            if not active:
                break
        best = min(active, key=lambda t: ((counts[t] + 1) / (_tier_weight(weights, t) or 1.0), t))
        order.append(ordered_by_tier[best][idx[best]])
        idx[best] += 1
        counts[best] += 1
    return order


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
    r"(?:(\d+(?:\.\d+)?)\s*(?:mm|cm)?\s+)?through[- ]?holes?\b", re.I)


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


def _spec_through_hole_diameter_mm(spec: str) -> Optional[float]:
    """The stated diameter of the through hole(s) named by _THROUGH_HOLES_RE's own
    optional numeric filler group ("a 3.2mm through hole" -> 3.2), or None when the spec
    states a through-hole count with no accompanying diameter (fix round 3, M1: with no
    diameter to match against a measured `hole_groups` entry, there is nothing to check
    positive evidence against)."""
    m = _THROUGH_HOLES_RE.search(spec or "")
    if not m or not m.group(2):
        return None
    return float(m.group(2))


def strict_through_holes_check(spec: str, facts: dict) -> Optional[str]:
    """POSITIVE evidence only (fix round 3, M1, replacing the old undercount test).

    The old check compared the spec's stated through-hole COUNT against `facts`'s
    aggregate `through_holes` number -- but scripts/inspect's through/blind classifier
    only recognises a hole as "through" via one specific axis-walking probe (see
    scripts/inspect's own comments on that method); a genuinely correct part whose
    through hole simply isn't oriented the way that probe expects can measure the
    aggregate `through_holes: 0` and be rejected forever on a part that is entirely
    right. That is an UNDERCOUNT failure mode, and a bare low/zero aggregate count is not
    trustworthy evidence of anything (Task 3 ruling, verbatim: "a zero or low through-
    count alone gives NO verdict -- not reject, not unscored").

    This check now fires ONLY on POSITIVE evidence: `facts["hole_groups"]`
    (scripts/inspect's own per-diameter grouping, each entry `{d, n, through,
    circle_d}`) must contain a group whose diameter `d` is within 0.1mm of a diameter
    the spec explicitly states for a through hole, AND that group must have measured at
    least one hole of that diameter as BLIND (`through < n`). Traced against the real
    V064 fixture (tests/fixtures/harvest_agreement_fixtures.json): its facts DO carry
    per-hole diameter grouping in general, but the ONE group present measures the 9.0mm
    boss body, not the spec's stated 3.2mm through hole -- there is no hole_groups entry
    near 3.2mm at all in that candidate's real facts (whatever swallowed that hole out of
    the grouping is a scripts/inspect question, out of scope for this harvest-local fix;
    see the fix round 3 report). Under this rule that is "no positive evidence found",
    not "reject" -- V064 stays rejected on strict_envelope_check alone (its stated
    180x130x55mm envelope measures 180x130x53mm), which this fix leaves untouched.

    A missing `hole_groups` field, an empty list, or no diameter-matching entry all
    return None (no verdict) -- there is no "unscored" branch any more: with only
    positive evidence able to fire this check at all, silence and a genuine clean pass
    are indistinguishable, and both are correctly "no verdict"."""
    stated_count = _spec_through_hole_count(spec)
    if stated_count is None:
        return None
    stated_d = _spec_through_hole_diameter_mm(spec)
    if stated_d is None:
        return None
    for g in (facts.get("hole_groups") or []):
        try:
            d = float(g.get("d"))
        except (TypeError, ValueError):
            continue
        if abs(d - stated_d) > 0.1:
            continue
        n = g.get("n") or 0
        through = g.get("through") or 0
        blind = n - through
        if blind > 0:
            return (f"strict_through_holes: spec states a {stated_d:g}mm through hole "
                    f"but {blind} of {n} measured hole(s) of that diameter came out "
                    f"blind (through {through}/{n})")
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

    Salvage is gated on `cfg["lab.harvest.salvage"]` (Task 3c, default OFF): the first
    real unit spent one repair codegen call plus a rebuild on every crash, and a salvage
    candidate can neither confirm nor be confirmed (fix round 3, H1), so with salvage on
    every one of those calls was pure cost when the model is simply unable to build the
    spec at all. When off, a crash gets its one ledger row and nothing more -- no repair
    call, no second materialize.

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
    if (m["error"] and cfg.get("salvage", False)
            and not (deadline is not None and time.monotonic() >= deadline)):
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
                agreement: Optional[str] = None,
                gate_version: Optional[str] = None,
                confirm_strength: Optional[str] = None) -> dict:
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
        # Fix round 3, M3: the effective gate configuration this row was judged under
        # (see gate_version()'s own docstring).
        "gate_version": gate_version,
        # Fix round 3, H2c: mirrors the pair row's own confirm_strength ("reference" /
        # "cross_pass" / "same_pass") when this ledger row corresponds to a CONFIRMED
        # good pair; None otherwise (an unconfirmed/split/silver/fail/unscored row has
        # nothing to weigh here).
        "confirm_strength": confirm_strength,
    }


def _pair_source(mode: str) -> str:
    return "student" if mode == "student" else "teacher:think"


def _good_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict,
                   turn: str, confirmed_by: str, confirm_strength: Optional[str] = None,
                   code_similarity_: Optional[float] = None,
                   gate_version_: Optional[str] = None) -> dict:
    """`confirmed_by` is "reference" (an owner-reference band match) or "agreement" (a
    matching other candidate or already-confirmed pair) -- fix round 2, Section A: every
    good pair now carries the reason it was trusted, never implicit. `signature` is
    stored too, so PairIndex can offer this pair as agreement evidence to a FUTURE
    candidate without re-deriving it from `facts`. Fix round 3, H2b/M3:
    `confirm_strength` ("reference" / "cross_pass" / "same_pass") and `code_similarity`
    (see the `code_similarity` function's own docstring) record how much weight this
    confirmation can bear; `gate_version_` stamps the effective gate configuration (see
    `gate_version`'s own docstring). The trailing underscores on the last two parameters
    avoid shadowing the module-level `code_similarity`/`gate_version` functions inside
    this function's own body."""
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
        "confirm_strength": confirm_strength,
        "code_similarity": code_similarity_,
        "gate_version": gate_version_,
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _candidate_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, fingerprint: str,
                   turn: str, agreement: str, gate_version_: Optional[str] = None) -> dict:
    """A gate-clean candidate that was NOT confirmed this round (fix round 2, Section A:
    "a full row (same fields as a pair row plus signature)"). Same shape as a good pair
    row, distinguished by id prefix "c:" (never collides with a real "p:" pair id) and
    kind="candidate"; `agreement` is "unconfirmed" or "split". Never written to
    PAIRS_FILE -- only CANDIDATES_FILE. `gate_version_` (fix round 3, M3) is what lets a
    later unit tell whether this candidate is still valid under TODAY's gate
    configuration before ever letting it confirm or be confirmed (see CandidateIndex)."""
    base = _good_pair_row(row, unit_id, mode, temperature, prompt_info, code, m, {},
                          turn, confirmed_by=None, gate_version_=gate_version_)
    base["id"] = f"c:{row['id']}:{base['id'].rsplit(':', 1)[-1]}"
    base["kind"] = "candidate"
    base["fingerprint"] = fingerprint
    base["agreement"] = agreement
    base["status"] = "unconfirmed"
    return base


def _promote_candidate_row(cand: dict, unit_id: str, confirmed_by: str,
                           confirm_strength: Optional[str] = None,
                           code_similarity_: Optional[float] = None) -> dict:
    """Turn an already-recorded candidates.jsonl row into a real pairs.jsonl "good" row
    (fix round 2, Section A: cross-unit confirmation) without re-deriving anything the
    candidate row already carries -- its system_sha1 already points at the stored prompt
    text, and there is no raw system string left on the candidate row to re-hash (that
    would also silently rewrite lab/state/systems/ with a duplicate under a fresh
    _store_system call, which is exactly the redundancy that function exists to avoid).
    `cand`'s own `gate_version` is carried straight through (fix round 3, M3): by the
    time this is called, the caller has already confirmed it matches today's, via
    CandidateIndex's own filtering, so there is nothing to re-derive."""
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
        "confirm_strength": confirm_strength,
        "code_similarity": code_similarity_,
        "gate_version": cand.get("gate_version"),
        "verified": cand.get("verified") or {},
        "ts": _now_utc(), "unit_id": unit_id,
        "promoted_from": cand.get("id"),
    }


def _fail_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict,
                   turn: str, gate_version_: Optional[str] = None) -> dict:
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
        "gate_version": gate_version_,
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


def _load_upgrades() -> dict:
    """pair_id -> latest confirm_strength from lab/state/upgrades.jsonl (fix round 3,
    H2b part 2): an append-only log of confirm_strength upgrades for pairs already on
    disk, so a LATER cross-pass confirmation never rewrites pairs.jsonl itself (an
    append-only file). The last row for a given pair_id wins (append order == recency)."""
    out: dict = {}
    for r in _read_jsonl(UPGRADES_FILE):
        pid = r.get("pair_id")
        if pid and r.get("confirm_strength"):
            out[pid] = r["confirm_strength"]
    return out


class PairIndex:
    """Cache of pairs.jsonl state for the duration of one unit/once call (Task 3 fix M5):
    read once, updated in memory as pairs are appended, never re-read from disk mid-run.
    Dedup uses the identifier/literal-normalised fingerprint (_code_fingerprint, fix
    round 3 H2a, floor of the old Task 3 fix L7); tier34-share tracks GOOD pairs only
    (the scheduler's own priority signal, see run_unit). Fix round 2, Section A: also
    indexes each GOOD pair's stored "signature" per spec_id, so a NEW candidate can be
    confirmed by agreeing with an ALREADY-confirmed pair from an earlier unit, not only
    with a sibling sampled in the same round. Fix round 3, H2b: each anchor entry now
    also carries the underlying pair's `source` (pass) and `code` (so a NEW candidate's
    code_similarity can be computed against it too) and `id`/`confirm_strength` (so a
    later cross-pass confirmation can append a confirm_strength UPGRADE row for it --
    see `_confirm_strength_for_cluster` -- without ever rewriting pairs.jsonl)."""

    def __init__(self) -> None:
        self._hashes: dict[str, set] = {}
        self._good_by_tier: dict[str, int] = {}
        self._good_total = 0
        self._anchors: dict[str, list] = {}
        upgrades = _load_upgrades()
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
                    strength = upgrades.get(r.get("id"), r.get("confirm_strength"))
                    self._anchors.setdefault(r.get("spec_id"), []).append({
                        "signature": sig, "source": r.get("source"), "code": r.get("code"),
                        "id": r.get("id"), "confirm_strength": strength})

    def has(self, spec_id: str, fingerprint: str) -> bool:
        return fingerprint in self._hashes.get(spec_id, set())

    def good_count_for(self, spec_id: str) -> int:
        """The number of GOOD pairs pairs.jsonl actually holds for `spec_id` right now
        (fix round 3, M4) -- the source of truth `_reconcile_progress` stamps into
        progress.json's own "pairs" counter, which a crash between a pair append and the
        next `_save_progress` call can otherwise leave stale-LOW."""
        return len(self._hashes.get(spec_id, ()))

    def add(self, spec_id: str, fingerprint: str, tier, signature: Optional[dict] = None,
           source: Optional[str] = None, code: Optional[str] = None,
           id_: Optional[str] = None, confirm_strength: Optional[str] = None) -> None:
        self._hashes.setdefault(spec_id, set()).add(fingerprint)
        self._good_total += 1
        t = str(tier)
        self._good_by_tier[t] = self._good_by_tier.get(t, 0) + 1
        if signature:
            self._anchors.setdefault(spec_id, []).append({
                "signature": signature, "source": source, "code": code, "id": id_,
                "confirm_strength": confirm_strength})

    def anchors_for(self, spec_id: str) -> list:
        return list(self._anchors.get(spec_id, []))

    def signatures_for(self, spec_id: str) -> list:
        return [a["signature"] for a in self._anchors.get(spec_id, [])]

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
    wins, so a "promoted"/"rejected_at_promotion" status row written after the original
    unconfirmed row correctly supersedes it in memory without ever rewriting or deleting
    the earlier line -- this is an append-only log). `pair_index` provides the
    belt-and-braces exclusion: a candidate whose fingerprint already has a real pair on
    disk is never offered again as "still unconfirmed", which is what actually makes a
    crash between promoting two candidates safe to re-run (see run_unit/sample_spec's own
    docstrings) -- the "promoted" status marker is written for human/tool legibility of
    candidates.jsonl, not as the correctness mechanism itself. Fix round 3, M3:
    `current_gate_version`, when given, also excludes a candidate written under a
    DIFFERENT gate_version -- it can neither confirm nor be promoted (see
    _confirmation_status's own `stale_candidates` count, which reports these) -- and a
    row already marked "rejected_at_promotion" (fix round 3, M3's strict/contamination
    re-check) is excluded the same permanent way "promoted" is."""

    def __init__(self, pair_index: "PairIndex",
                current_gate_version: Optional[str] = None) -> None:
        self._pair_index = pair_index
        by_id: dict[str, dict] = {}
        for r in _read_jsonl(CANDIDATES_FILE):
            rid = r.get("id")
            if rid:
                by_id[rid] = r
        self._by_spec: dict[str, list] = {}
        for r in by_id.values():
            if r.get("status") in ("promoted", "rejected_at_promotion"):
                continue
            if (current_gate_version is not None
                    and r.get("gate_version") != current_gate_version):
                continue
            fp = r.get("fingerprint")
            spec_id = r.get("spec_id")
            if fp and self._pair_index.has(spec_id, fp):
                continue
            self._by_spec.setdefault(spec_id, []).append(r)

    def unconfirmed_for(self, spec_id: str) -> list:
        return list(self._by_spec.get(spec_id, []))


# ---------------------------------------------------------------------------
# confirm_strength (fix round 3, H2b) -- how much weight one round's agreement can bear.
# See this module's own docstring for the plain-words rationale.
# ---------------------------------------------------------------------------

def _find_full_cluster(pool: list[dict], cfg: dict, winners: list[dict]) -> list[dict]:
    """The complete agreeing CLUSTER (including any existing_anchor and any non-chosen
    member) that produced `winners` -- re-derived from the same pool/cfg
    `resolve_agreement` already clustered, rather than duplicating its cluster-selection
    logic here, so this can never drift from what actually got confirmed. `winners` is a
    non-empty subset of exactly one cluster by construction (see resolve_agreement's own
    contract); an empty `winners` (nothing confirmed) returns []."""
    if not winners:
        return []
    winner_ids = {id(w) for w in winners}
    for cluster in _cluster_by_signature(pool, cfg):
        if any(id(item) in winner_ids for item in cluster):
            return cluster
    return list(winners)   # defensive fallback; should not be reachable


def _item_pass_label(item: dict, mode: str) -> str:
    """"student" or "think" for one pool item (fix round 3, H2b): a "new" item was
    necessarily sampled under THIS round's `mode`; a "carried" or "existing" (anchor)
    item carries its own historical pass via its stored `source` field
    ("student"/"teacher:think", see `_pair_source`)."""
    origin = item.get("origin")
    if origin == "new":
        return "think" if mode == "think" else "student"
    if origin == "carried":
        src = (item.get("cand_row") or {}).get("source")
    elif origin == "existing":
        src = item.get("source")
    else:
        src = None
    return "think" if src == "teacher:think" else "student"


def _confirm_strength_for_cluster(cluster: list[dict], mode: str) -> str:
    """"cross_pass" when the agreeing set spans both a student-pass and a think-pass
    sample, else "same_pass" (fix round 3, H2b) -- see this module's own docstring for
    why that distinction matters. Reference-band confirmation is handled separately by
    its own call sites (a reference match needs no cluster at all)."""
    labels = {_item_pass_label(item, mode) for item in cluster}
    return "cross_pass" if len(labels) >= 2 else "same_pass"


def _cluster_codes(cluster: list[dict], exclude: dict) -> list[str]:
    """Every OTHER cluster member's code, for code_similarity (fix round 3, H2b) -- an
    existing_anchor carries its code via PairIndex (see PairIndex.add/anchors_for); a
    "new"/"carried" item carries it directly/on its stored candidate row."""
    out = []
    for item in cluster:
        if item is exclude:
            continue
        code = item.get("code")
        if code is None and item.get("origin") == "carried":
            code = (item.get("cand_row") or {}).get("code")
        if code:
            out.append(code)
    return out


# ---------------------------------------------------------------------------
# Adaptive sampling (Task 3c, 2026-09-19): the first real harvest unit spent 27 GPU
# minutes and produced ZERO pairs across 3 specs / 22 ledger rows, because every
# candidate of a tier 3-4 round was sampled even when the first ones all crashed --
# rejection sampling pays at the model's capability frontier, not above it. A round now
# PROBES a small number of candidates first and gives up early (a "cold" round) when
# none of them was even gate-clean, unless a candidate already carried over from an
# earlier round gives the spec something worth trying one more sample for.
# ---------------------------------------------------------------------------

def _probe_indices(n: int, is_tier34: bool, probe_candidates: int) -> tuple[list[int], list[int]]:
    """(probe, rest): the 0-based candidate indices (each maps to `temps[i % len(temps)]`
    exactly as the main loop always has) to try during a round's PROBE stage, and the
    remaining indices to try only once the probe finds at least one gate-clean candidate.

    Tier 3-4 spreads its probe across the tier's own temperature list (stride 2 -- with
    the shipped defaults, `candidates_tier34=5` and `probe_candidates=2`, that is indices
    0 and 2, i.e. temperatures 0.2 and 0.5) so the two probe samples cover more of the
    range than two adjacent low temperatures would; tiers 1-2 just take the first
    `probe_candidates` indices, unchanged from the old always-sequential order (their
    geometry is simple enough that the old single-probe-sized budget rarely needed a
    change). `probe_candidates` is floored at 1: a probe of zero candidates cannot ever
    decide anything."""
    probe_candidates = max(1, int(probe_candidates))
    if is_tier34:
        probe = list(range(0, n, 2))[:probe_candidates]
    else:
        probe = list(range(min(probe_candidates, n)))
    rest = [i for i in range(n) if i not in probe]
    return probe, rest


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
    candidate.

    Fix round 3, H1: a `turn == "salvage"` candidate (the crash-repair attempt) never
    enters the agreement pool -- it can neither confirm another candidate nor be
    confirmed by one. Two salvages independently "confirming" each other was the exact
    defect a reviewer found: a repair call is more likely than a first-try call to
    produce a plausible-but-wrong shape, and letting two of those agree with each other
    is the agreement mechanism's whole point turned against itself. A reference-band
    match on a salvage candidate is UNAFFECTED (ground truth needs no cross-candidate
    agreement, salvage or not); a gate-clean salvage with no reference match is routed to
    REVIEW_FILE instead, exactly like a silver verdict -- visible to a human, never
    silently promoted and never silently lost.

    Fix round 3, LOW3: `attempt_key` (student_attempts/teacher_attempts) is incremented
    ONCE per `sample_spec` CALL (a "round"), not once per candidate inside it -- the
    Task 3 plan's own intent ("tier 3-4 specs get 2 student rounds then the think
    pass") only holds if a round with `candidates_tier34=5` costs 1 attempt, not 5. The
    increment happens on the first iteration that actually starts generating a
    candidate (so a round that never gets that far, e.g. an already-past deadline, never
    spends an attempt at all), and is undone only if the round produces not even one
    fully-processed candidate before an abort/infra error cuts it short (see the
    `completed_any` flag below and this function's outer except)."""
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
    # Task 3c: probe/rest ordering for the adaptive-sampling stop below. `probe_order`
    # is what the main loop actually iterates; `candidate_label` (and hence the ledger's
    # own `candidate` field, and which temperature is used) always reflects the ORIGINAL
    # index `i`, never the probe-reordered position -- only the ORDER candidates are
    # tried in changes, not their identity.
    probe_idx, rest_idx = _probe_indices(n, is_tier34,
                                        cfg.get("probe_candidates", 2))
    probe_order = probe_idx + rest_idx
    attempt_key = "student_attempts" if mode == "student" else "teacher_attempts"
    agreement_cfg = cfg.get("agreement") or {}
    gv = gate_version(cfg)

    pair_index_anchors = pair_index.anchors_for(spec_id)
    candidate_index = CandidateIndex(pair_index, current_gate_version=gv)
    carried_raw = candidate_index.unconfirmed_for(spec_id)
    carried: list[dict] = []
    for c in carried_raw:
        # Fix round 3, M3: re-prove a carried candidate against TODAY's strict checks
        # and contamination sets BEFORE it ever gets a chance to confirm or be
        # confirmed -- see _promotion_recheck_ok's own docstring for why this happens
        # here, not only at the moment it would be written to pairs.jsonl.
        reason = _promotion_recheck_ok(c, cfg)
        if reason:
            _append_jsonl(CANDIDATES_FILE, {
                **c, "status": "rejected_at_promotion",
                "rejected_reason": reason, "rejected_ts": _now_utc()})
            continue
        carried.append({"origin": "carried", "cand_row": c, "fingerprint": c.get("fingerprint"),
                        "signature": c.get("signature"), "temperature": c.get("temperature") or 0.0})
    existing_anchors = [{"origin": "existing", "existing_anchor": True,
                         "signature": a["signature"], "source": a["source"],
                         "code": a["code"], "id": a["id"],
                         "confirm_strength": a["confirm_strength"],
                         "fingerprint": None, "temperature": 0.0}
                        for a in pair_index_anchors]
    clean_candidates: list[dict] = []
    # seen_fps also seeded from carried candidates: a NEW sample identical to one already
    # sitting unconfirmed on disk is the SAME code, not a second opinion, and must not be
    # allowed to "confirm" it (Section A: "a DIFFERENT code fingerprint").
    seen_fps: set = {c["fingerprint"] for c in carried if c.get("fingerprint")}
    resolved = {"done": False}

    def _pool() -> list[dict]:
        return existing_anchors + carried + clean_candidates

    def _write_new_ledger(c: dict, agreement: Optional[str],
                          confirm_strength: Optional[str] = None) -> None:
        _append_jsonl(LEDGER_FILE, _ledger_row(
            unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
            candidate=c["candidate_label"], temperature=c["temperature"], m=c["m"],
            band_info={}, usage=c["usage"], model=c["model"], build_dir=c["build_dir"],
            seconds=c["seconds"], fingerprint=c["fingerprint"], agreement=agreement,
            gate_version=gv, confirm_strength=confirm_strength))

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
        creates progress.json" guarantee).

        Fix round 3, H2b: every confirmed pair (and its ledger row) additionally gets
        `confirm_strength` (from the FULL agreeing cluster, including any anchor/losing
        member -- see `_find_full_cluster`) and `code_similarity` (against every OTHER
        coded member of that same cluster). When the cluster contains an anchor whose
        OWN stored confirm_strength is not already "cross_pass" but this event proves
        cross-pass evidence for it, an upgrade row is appended to UPGRADES_FILE rather
        than rewriting the anchor's own (append-only) pairs.jsonl row."""
        if resolved["done"]:
            return False
        resolved["done"] = True
        if not clean_candidates:
            return False
        pool = _pool()
        winners, tag = resolve_agreement(pool, agreement_cfg)
        winner_ids = {id(w) for w in winners}
        if tag == "agreement":
            full_cluster = _find_full_cluster(pool, agreement_cfg, winners)
            strength = _confirm_strength_for_cluster(full_cluster, mode)
            anchor_items = [item for item in full_cluster if item.get("existing_anchor")]
            if strength == "cross_pass":
                # Only a genuine same_pass -> cross_pass promotion is an "upgrade" --
                # an anchor already confirmed by "reference" (owner ground truth) is
                # not made MORE true by a same-geometry sample from a different pass,
                # so it is never touched here.
                for anchor in anchor_items:
                    if anchor.get("id") and anchor.get("confirm_strength") == "same_pass":
                        _append_jsonl(UPGRADES_FILE, {
                            "pair_id": anchor["id"], "confirm_strength": "cross_pass",
                            "ts": _now_utc()})
            winners_sorted = sorted(winners, key=lambda c: c.get("temperature", 0.0))
            slots = max(0, max_pairs - entry.get("pairs", 0))
            chosen_ids = {id(c) for c in winners_sorted[:slots]}
            for c in winners_sorted:
                chosen = id(c) in chosen_ids
                if c["origin"] == "new":
                    _write_new_ledger(c, "agreement",
                                      confirm_strength=strength if chosen else None)
                    if chosen:
                        sim = code_similarity(c["code"], _cluster_codes(full_cluster, c))
                        pair_row = _good_pair_row(row, unit_id, mode, c["temperature"],
                                                  c["prompt_info"], c["code"], c["m"], {},
                                                  c["turn"], confirmed_by="agreement",
                                                  confirm_strength=strength,
                                                  code_similarity_=sim, gate_version_=gv)
                        _append_jsonl(PAIRS_FILE, pair_row)
                        pair_index.add(spec_id, c["fingerprint"], tier,
                                       pair_row.get("signature"), source=pair_row["source"],
                                       code=c["code"], id_=pair_row["id"],
                                       confirm_strength=strength)
                        entry["pairs"] = entry.get("pairs", 0) + 1
                    else:
                        _append_jsonl(CANDIDATES_FILE, _candidate_row(
                            row, unit_id, mode, c["temperature"], c["prompt_info"],
                            c["code"], c["m"], c["fingerprint"], c["turn"],
                            agreement="agreement", gate_version_=gv))
                elif chosen:   # carried, and this round has a slot for it
                    cand_code = c["cand_row"].get("code")
                    sim = code_similarity(cand_code, _cluster_codes(full_cluster, c))
                    pair_row = _promote_candidate_row(c["cand_row"], unit_id, "agreement",
                                                      confirm_strength=strength,
                                                      code_similarity_=sim)
                    _append_jsonl(PAIRS_FILE, pair_row)
                    pair_index.add(spec_id, c["fingerprint"], tier,
                                   pair_row.get("signature"), source=pair_row["source"],
                                   code=cand_code, id_=pair_row["id"],
                                   confirm_strength=strength)
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
                c["m"], c["fingerprint"], c["turn"], agreement=loser_tag,
                gate_version_=gv))
        return True

    # The whole loop lives inside a try/finally so an abort or infra error propagating
    # out (re-raised below, after the outer except decides whether to undo this ROUND's
    # attempt count) still flushes whatever clean_candidates an EARLIER, successfully-
    # completed candidate this round already added -- without this, a signal landing on
    # candidate 2 would silently lose candidate 1's gate-clean result forever, since its
    # ledger/candidate row is deferred until _finalize() runs (fix round 2: "every
    # candidate ... gets one row in the ledger" must still hold when the unit is
    # interrupted, not only when it completes normally).
    #
    # Fix round 3, LOW3: `completed_any` tracks whether this ROUND has fully processed
    # at least one candidate (whatever its verdict) -- the outer except below undoes the
    # round's single attempt-key increment only when it is still False, i.e. the round
    # produced NOTHING (aborted/infra-errored before even one candidate finished).
    completed_any = False
    # Task 3c: `probe_found_clean` flips True the moment ANY attempt (first or salvage,
    # any turn) this round classifies "good" -- see _classify's own verdict table. A
    # carried unconfirmed candidate already sitting on disk for this spec is "there is
    # something to confirm" (Task 3c ruling, verbatim): it earns the round one extra try
    # beyond the plain probe size before giving up cold.
    probe_found_clean = False
    stop_threshold = min(len(probe_order), len(probe_idx) + (1 if carried else 0))
    try:
        try:
            for pos, i in enumerate(probe_order):
                _check_abort()
                # Task 3c: the adaptive-sampling stop. Checked at the TOP of the loop
                # (rather than after each candidate) so it also applies when the
                # PREVIOUS candidate's codegen call raised and `continue`d straight past
                # the bottom of the loop body -- a codegen failure still spent one of
                # the round's probe tries. `pos` at this point is exactly the count of
                # candidates already attempted (0-based), so `pos >= stop_threshold`
                # means the probe (plus any carried-candidate allowance) is fully spent.
                if pos >= stop_threshold and not probe_found_clean:
                    break
                # Fix round 2, Section C: only an ALREADY-CONFIRMED pair count stops
                # sampling early -- a merely gate-clean, unconfirmed candidate must not
                # (the old early stop fired on exactly that, which is how a wrong
                # candidate went unchallenged).
                if entry.get("pairs", 0) >= max_pairs:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if pos == 0:
                    # Fix round 3, LOW3: one attempt per ROUND (this whole sample_spec
                    # call), charged only once the round actually starts working, not
                    # once per candidate inside it.
                    entry[attempt_key] = entry.get(attempt_key, 0) + 1
                temperature = temps[i % len(temps)]
                candidate_label = str(i)
                t0 = time.monotonic()
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
                        error_text=f"codegen failed: {str(e)[:400]}", gate_version=gv))
                    completed_any = True
                    _save_progress(progress)
                    continue
                gen_seconds = time.monotonic() - t0
                _check_abort()
                if deadline is not None and time.monotonic() >= deadline:
                    # The candidate already generated code but there is no time left to
                    # materialize it; drop it rather than starting work we cannot finish.
                    # No attempt-count adjustment here any more (fix round 3, LOW3): the
                    # round already spent its one attempt at pos==0 regardless of exactly
                    # how many of its candidates got through before the deadline.
                    break

                with tempfile.TemporaryDirectory(prefix="harvest_") as td:
                    build_dir = Path(td)
                    attempts = _execute_with_salvage(spec, code, build_dir, prompt_info,
                                                    gen_seconds, cfg, deadline=deadline)
                    for attempt in attempts:
                        label = (candidate_label if attempt["turn"] == "first"
                                else f"{candidate_label}-salvage")
                        verdict, band_info = _classify(attempt["m"], reference_stl, build_dir)
                        if verdict == "good":
                            # Task 3c: a "good" verdict (of ANY turn -- first or salvage,
                            # confirmed or not yet) is proof the model CAN build this
                            # spec, which is exactly what the probe is checking for --
                            # never mind whether this particular sample goes on to be a
                            # duplicate, a lone unconfirmed candidate, or a full pair.
                            probe_found_clean = True
                        persisted = (_persist_build(build_dir, unit_id, spec_id, label)
                                    if verdict not in ("none", "unscored") else None)
                        usage = (attempt["prompt_info"] or {}).get("usage")
                        model = (attempt["prompt_info"] or {}).get("model")
                        fp = (_code_fingerprint(attempt["code"])
                             if verdict != "unscored" else None)

                        if verdict == "good" and band_info.get("band") == "match":
                            # An owner-reference band match confirms on its own, exactly as
                            # before this fix round -- no cross-candidate agreement needed,
                            # and UNAFFECTED by turn=="salvage" (fix round 3, H1's own
                            # docstring note: "a reference band match on a salvage
                            # candidate keeps today's behaviour").
                            if pair_index.has(spec_id, fp) or fp in seen_fps:
                                _append_jsonl(LEDGER_FILE, _ledger_row(
                                    unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                    candidate=label, temperature=temperature, m=attempt["m"],
                                    band_info=band_info, usage=usage, model=model,
                                    build_dir=persisted, seconds=attempt["seconds"],
                                    fingerprint=fp, agreement=None, gate_version=gv))
                                continue
                            seen_fps.add(fp)
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement="reference",
                                gate_version=gv, confirm_strength="reference"))
                            if entry.get("pairs", 0) < max_pairs:
                                pair_row = _good_pair_row(
                                    row, unit_id, mode, temperature, attempt["prompt_info"],
                                    attempt["code"], attempt["m"], band_info, attempt["turn"],
                                    confirmed_by="reference", confirm_strength="reference",
                                    code_similarity_=code_similarity(attempt["code"], []),
                                    gate_version_=gv)
                                _append_jsonl(PAIRS_FILE, pair_row)
                                pair_index.add(spec_id, fp, tier, pair_row.get("signature"),
                                              source=pair_row["source"], code=attempt["code"],
                                              id_=pair_row["id"], confirm_strength="reference")
                                entry["pairs"] = entry.get("pairs", 0) + 1
                        elif verdict == "good" and attempt["turn"] == "salvage":
                            # Fix round 3, H1: a salvage candidate with no reference match
                            # never enters the agreement pool -- it can neither confirm
                            # another candidate nor be confirmed by one (a repair call is
                            # more likely than a first try to produce a plausible-but-wrong
                            # shape). Routed to review, exactly like a silver verdict:
                            # visible to a human, never silently promoted, never lost.
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement=None, gate_version=gv))
                            render = str(Path(persisted) / "build.png") if persisted else None
                            _append_jsonl(REVIEW_FILE, _review_row(
                                row, unit_id, mode, temperature, attempt["prompt_info"],
                                attempt["code"], attempt["m"], render, attempt["turn"]))
                        elif verdict == "good":
                            # Gate-clean, no reference, first turn -- held for agreement
                            # (fix round 2, Section A). Its ledger row is written only
                            # once _finalize() knows this candidate's outcome.
                            if pair_index.has(spec_id, fp) or fp in seen_fps:
                                _append_jsonl(LEDGER_FILE, _ledger_row(
                                    unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                    candidate=label, temperature=temperature, m=attempt["m"],
                                    band_info=band_info, usage=usage, model=model,
                                    build_dir=persisted, seconds=attempt["seconds"],
                                    fingerprint=fp, agreement=None, gate_version=gv))
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
                                fingerprint=fp, agreement=None, gate_version=gv))
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
                                fingerprint=fp, agreement=None, gate_version=gv))
                            _append_jsonl(PAIRS_FILE, _fail_pair_row(
                                row, unit_id, mode, temperature, attempt["prompt_info"],
                                attempt["code"], attempt["m"], band_info, attempt["turn"],
                                gate_version_=gv))
                        else:   # "none" or "unscored"
                            _append_jsonl(LEDGER_FILE, _ledger_row(
                                unit_id=unit_id, spec_id=spec_id, tier=tier, mode=mode,
                                candidate=label, temperature=temperature, m=attempt["m"],
                                band_info=band_info, usage=usage, model=model,
                                build_dir=persisted, seconds=attempt["seconds"],
                                fingerprint=fp, agreement=None, gate_version=gv))
                completed_any = True
                _save_progress(progress)
                _check_abort()
                if resolved["done"]:
                    break
        except (SpecgenAborted, _InfraError):
            if not completed_any:
                entry[attempt_key] -= 1
            raise
    finally:
        if _finalize():
            _save_progress(progress)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _rate_by(rows: list[dict], key_fn, pred) -> dict:
    """{key: round(hits/total, 4)} for whatever `pred(row)` counts as a hit, grouped by
    `key_fn(row)` (None keys are dropped -- nothing to group them under). Generalises the
    old `_pass_rate` (confirmed-good pass rate) to also drive `_row_is_gate_clean` (Task
    3c's `gate_clean_rate_by_*`, the model's raw success rate BEFORE cross-candidate
    agreement is even considered)."""
    counts: dict = {}
    hits: dict = {}
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        counts[k] = counts.get(k, 0) + 1
        if pred(r):
            hits[k] = hits.get(k, 0) + 1
    return {k: round(hits.get(k, 0) / n, 4) for k, n in counts.items()}


def _pass_rate(rows: list[dict], key_fn) -> dict:
    return _rate_by(rows, key_fn, _row_is_good)


def _row_is_gate_clean(row: dict) -> bool:
    """True for a ledger row whose candidate was gate-clean (Task 3c: `_classify`'s own
    "good" verdict), regardless of whether it went on to be CONFIRMED -- the model's raw
    success rate, one step upstream of `_row_is_good`'s confirmed-good rate. Mirrors
    `_is_good`'s own condition directly off the ledger row's stored fields, since the
    ledger never stores the verdict string itself: `ok` (error is None) is not enough on
    its own -- an "unscored" row also has `ok=True` (see _ledger_row's own comment on
    why regate reports its error as None there), so it must be excluded explicitly."""
    if not row.get("ok") or row.get("unscored_reason"):
        return False
    if row.get("gate_hard") or row.get("gate_spec"):
        return False
    return row.get("band") in (None, "match")


def _spec_source_map(bank: list[dict]) -> dict:
    return {r.get("id"): r.get("source") for r in bank}


def _row_source_group(row: dict, spec_source: dict) -> str:
    """"specgen" vs "other" for a ledger row, joined against the bank by spec_id (the
    ledger itself never carries the spec's bank `source` -- Task 3c's whole point was
    that a model-written `specgen` spec behaves very differently from a teacher-suite/
    owner-reference one, so the status breakdown needs this distinction even though the
    ledger row predates it)."""
    return "specgen" if spec_source.get(row.get("spec_id")) == "specgen" else "other"


def _cold_specs_count(bank: list[dict], progress: dict, cfg: dict) -> int:
    """Count of currently-ELIGIBLE specs (student or teacher pool -- see _eligible_pools)
    sitting in the "cold" bucket _intra_tier_key uses for scheduling: sampled at least
    once, and with no live unconfirmed candidate on disk right now. Reported so the
    owner can see, at a glance, how many specs the harvest has tried and gotten nothing
    confirmable from yet -- distinct from `exhausted_specs` (used up its full teacher
    attempt budget and given up for good)."""
    student_pool, teacher_pool = _eligible_pools(bank, progress, cfg)
    spec_ids = {r["id"] for r in student_pool} | {r["id"] for r in teacher_pool}
    carried_ids = _live_candidate_spec_ids()
    count = 0
    for spec_id in spec_ids:
        entry = progress.get(spec_id, {})
        attempts = entry.get("student_attempts", 0) + entry.get("teacher_attempts", 0)
        if attempts > 0 and spec_id not in carried_ids:
            count += 1
    return count


def _yield_stats(ledger_rows: list[dict]) -> dict:
    """Confirmed good pairs per GPU-hour (Task 3c: the number the owner tunes
    `tier_weights` against), both cumulative (the whole ledger) and for just the most
    recently written unit_id. GPU-hours are the sum of ledger `seconds` -- the exact
    same field `_hours_today`/`nights_completed` already treat as authoritative wall
    time spent, converted to hours. `good_pairs_per_gpu_hour` is None (never a
    ZeroDivisionError, never a misleading 0.0) when there is no ledger time to divide
    by yet."""
    def _per_hour(good: int, seconds: float) -> Optional[float]:
        hours = seconds / 3600.0
        return round(good / hours, 4) if hours > 0 else None

    last_unit = ledger_rows[-1].get("unit_id") if ledger_rows else None
    total_seconds = last_seconds = 0.0
    total_good = last_good = 0
    for r in ledger_rows:
        try:
            secs = float(r.get("seconds") or 0.0)
        except (TypeError, ValueError):
            secs = 0.0
        good = _row_is_good(r)
        total_seconds += secs
        total_good += int(good)
        if r.get("unit_id") == last_unit:
            last_seconds += secs
            last_good += int(good)
    return {
        "cumulative": {"good_pairs": total_good,
                      "gpu_hours": round(total_seconds / 3600.0, 4),
                      "good_pairs_per_gpu_hour": _per_hour(total_good, total_seconds)},
        "last_unit": {"unit_id": last_unit, "good_pairs": last_good,
                     "gpu_hours": round(last_seconds / 3600.0, 4),
                     "good_pairs_per_gpu_hour": _per_hour(last_good, last_seconds)},
    }


def _pairs_stats() -> dict:
    rows = _read_jsonl(PAIRS_FILE)
    total = len(rows)
    by_tier: dict = {}
    by_source: dict = {}
    good_by_tier: dict = {}
    good_by_confirm_strength: dict = {}
    good_total = 0
    upgrades = _load_upgrades()
    for r in rows:
        by_tier[str(r.get("tier"))] = by_tier.get(str(r.get("tier")), 0) + 1
        by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1
        if r.get("kind") == "good":
            good_total += 1
            good_by_tier[str(r.get("tier"))] = good_by_tier.get(str(r.get("tier")), 0) + 1
            # Fix round 3, H2c: an UPGRADES_FILE row (a later cross-pass confirmation
            # for an anchor originally written as same_pass) takes priority over the
            # row's own stored confirm_strength, without ever rewriting pairs.jsonl.
            strength = upgrades.get(r.get("id"), r.get("confirm_strength")) or "unknown"
            good_by_confirm_strength[strength] = good_by_confirm_strength.get(strength, 0) + 1
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
        # Fix round 3, H2c: good pairs by confirm_strength, so the compiler/audit can see
        # at a glance how much of the corpus is only same_pass-confirmed (see this
        # module's own docstring for why that distinction matters).
        "good_by_confirm_strength": good_by_confirm_strength,
    }


def _confirmation_status(ledger_rows: list[dict], bank: list[dict], progress: dict,
                         cfg: dict) -> dict:
    """Fix round 2, Section A: "Status gets unconfirmed, confirmed_by: {reference,
    agreement}, and split_specs counts" -- an aggregate view of how the confirmation
    layer is doing, alongside (not replacing) the per-pair confirmed_by field and the
    per-ledger-row agreement field this same fix round adds.
    `unconfirmed_candidates`: rows in candidates.jsonl not marked "promoted" or
    "rejected_at_promotion" (still waiting on a confirming partner, or permanently
    stuck if their spec is exhausted).
    `confirmed_by`: how many GOOD pairs on disk were confirmed each way.
    `split_specs`: distinct spec_ids with at least one ledger row recording a "split"
    this fix round found (2+ disagreeing clusters, nobody confirmed) -- a running total,
    not just this unit's, since a split spec needs a human's attention eventually.
    `exhausted_specs`: specs that used their full teacher attempt budget and are still
    short of max_pairs_per_spec (see _exhausted_specs).
    `stale_candidates` (fix round 3, M3): unconfirmed candidates whose stored
    `gate_version` does not match today's effective one -- they can neither confirm nor
    be promoted until the bank/config catches up with them (see CandidateIndex's own
    gate_version filtering).
    `cold_specs` (Task 3c): see _cold_specs_count's own docstring."""
    cfg_gv = gate_version(cfg)
    candidate_rows = _read_jsonl(CANDIDATES_FILE)
    latest_by_id: dict[str, dict] = {}
    for r in candidate_rows:
        rid = r.get("id")
        if rid:
            latest_by_id[rid] = r
    live = [r for r in latest_by_id.values()
           if r.get("status") not in ("promoted", "rejected_at_promotion")]
    unconfirmed_candidates = len(live)
    stale_candidates = sum(1 for r in live if r.get("gate_version") != cfg_gv)
    confirmed_by = {"reference": 0, "agreement": 0}
    for r in _read_jsonl(PAIRS_FILE):
        if r.get("kind") == "good" and r.get("confirmed_by") in confirmed_by:
            confirmed_by[r["confirmed_by"]] += 1
    split_specs = len({r.get("spec_id") for r in ledger_rows
                       if r.get("agreement") == "split" and r.get("spec_id")})
    return {
        "unconfirmed_candidates": unconfirmed_candidates,
        "stale_candidates": stale_candidates,
        "confirmed_by": confirmed_by,
        "split_specs": split_specs,
        "exhausted_specs": len(_exhausted_specs(bank, progress, cfg)),
        "cold_specs": _cold_specs_count(bank, progress, cfg),
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
    spec_source = _spec_source_map(bank)
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
            # Task 3c: raw gate-clean rate (upstream of confirmation) and confirmed-good
            # rate, both by tier and by bank source (specgen vs the rest) -- the numbers
            # the owner tunes tier_weights against, alongside `yield` below.
            "gate_clean_rate_by_tier": _rate_by(ledger_rows, lambda r: str(r.get("tier")),
                                               _row_is_gate_clean),
            "gate_clean_rate_by_source": _rate_by(
                ledger_rows, lambda r: _row_source_group(r, spec_source), _row_is_gate_clean),
            "confirmed_rate_by_tier": _pass_rate(ledger_rows, lambda r: str(r.get("tier"))),
            "confirmed_rate_by_source": _rate_by(
                ledger_rows, lambda r: _row_source_group(r, spec_source), _row_is_good),
        },
        "yield": _yield_stats(ledger_rows),
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

def _reconcile_progress(progress: dict, bank: list[dict], pair_index: PairIndex) -> None:
    """pairs.jsonl is the source of truth for how many GOOD pairs a spec actually has
    (fix round 3, M4) -- progress.json's own "pairs" counter is bookkeeping a crash
    between a pair append (`_append_jsonl(PAIRS_FILE, ...)`) and the NEXT
    `_save_progress` call can leave stale-LOW, which then lets a later unit sample past
    `max_pairs_per_spec` before its own (already-real) pairs are reflected in
    progress.json. Called at the start of every --unit and --once run, before pool
    eligibility is computed: every bank spec's "pairs" entry is overwritten from
    `pair_index.good_count_for`; attempt counters (student_attempts/teacher_attempts),
    which pairs.jsonl cannot reconstruct, are left exactly as they are."""
    for row in bank:
        spec_id = row["id"]
        entry = progress.setdefault(
            spec_id, {"student_attempts": 0, "teacher_attempts": 0, "pairs": 0})
        entry["pairs"] = pair_index.good_count_for(spec_id)


def run_unit(cfg: dict) -> dict:
    unit_id = _now_utc()
    deadline = time.monotonic() + cfg["unit_minutes"] * 60
    bank = specbank.load_bank()
    progress = _load_progress()
    pair_index = PairIndex()
    _reconcile_progress(progress, bank, pair_index)
    _save_progress(progress)
    student_pool, teacher_pool = _eligible_pools(bank, progress, cfg)
    mode = _choose_mode(teacher_pool, cfg)
    pool = teacher_pool if mode == "think" else student_pool
    processed = 0
    if pool:
        # Task 3c: the weighted round-robin (see _order_specs's own docstring) replaces
        # the old "tier 3-4 first while good_tier34_share() < 0.40" rule -- that share
        # stays visible in status (_pairs_stats) as a reported goal only.
        ordered = _order_specs(pool, progress, cfg)
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
    pair_index = PairIndex()
    _reconcile_progress(progress, bank, pair_index)
    _save_progress(progress)
    entry = progress.get(spec_id, {})
    # Task 3 fix L4: honour teacher_passes the same way _choose_mode does, rather than
    # only checking think_rung_available().
    mode = "student"
    if (entry.get("student_attempts", 0) >= 2
            and "think" in (cfg.get("teacher_passes") or [])
            and think_rung_available()):
        mode = "think"
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

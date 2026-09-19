#!/usr/bin/env python3
"""lab/harvest.py -- the Phase 3 "data engine" harvest unit (Task 3).

Samples build123d code from the local CAD model (the maker arm, or the resident when the
maker is disabled) for specs in `lab/state/specs.jsonl` (the bank built by
`lab/specbank.py`), verifies every candidate the exact same way `scripts/fluid_gen.py`
verifies a real build (`_materialize`: run, inspect, gate against the spec; one crash
salvage, reusing `_materialize`'s own `diagnose`/`_revise_on_repair_rung` helpers), and
keeps only verified winners as training pairs. Every candidate -- kept or not -- gets one
row in the ledger, so the record of what was tried is never lossy.

All local: sampling, execution and scoring all happen inside THIS process, on the arm the
GPU window already switched to (`lab._armwindow.arm_window`) -- no cloud call anywhere.
`CAD_BENCH=1` is set before `cad_engine` is even imported so a good build can never
self-promote into `~/.openclaw/cad-examples.jsonl` / `cad-sftpairs.jsonl` (the bank has its
own files: `ledger.jsonl` / `pairs.jsonl` / `review.jsonl`).

Two passes, never both in the same unit ("a unit never swaps arms", Task 3 ruling): a
"student" pass (`engine.generate_code_raw`, thinking off, the same call fluid mode's
strong rung makes) for specs that have not yet failed the student twice; a "think" pass
(`engine._ollama(..., think=True)`, the SAME loaded arm, ONE request with thinking
switched on -- Task 1b) for specs that HAVE failed the student twice, only when
`cad_v5.config.think_rung_available()` says the active arm's request shape actually
changes anything, and only once at least 20 specs are waiting for it (a teacher unit is
not worth an eviction/restore cycle for one spec).

Verdict table (spec-derived contamination is already handled at bank-admission time --
see lab/specbank.py -- so nothing sampled here is a card-suite spec):
  ok, gate_hard==0, gate_spec==0, band in (None, "match")  -> pair, kind "good"
  ok, gate_hard==0, gate_spec>0                            -> silver review row, not a pair
  reference present (owner-reference bank rows only, none exist on disk as of 2026-09-19)
    and band == "near_miss"                                -> pair, kind "fail"
  everything else                                          -> nothing (still ledgered)
A "good" candidate whose reference row has NO source code of its own (an owner-reference
spec.txt/model.step pair has geometry, never a build123d program) cannot produce a "fail"
pair with a correct target: `code` is left null on that pair and `bad_code`/`problem`
carry the wrong candidate and its geometric diagnosis, same as gift_sample.py's GIFT-FAIL
rows. This branch is untested by real data today (the reference folder is empty) but is
exercised by a planted fixture in tests/test_lab_harvest.py.

Runs ONLY inside a GPU window, exactly like lab/specgen.py (`ship.require_gpu_window`,
checked before ANYTHING else -- no signal handler, no arm switch, no bank read):

    lab/harvest_unit.sh                                          # the timer's own launch
    lab/gpu_window.sh python3 lab/harvest.py --unit               # equivalent, direct
    lab/gpu_window.sh python3 lab/harvest.py --once --spec-id ID  # one spec now, no gates
    python3 lab/harvest.py --status                                # no GPU window needed

`--unit` additionally gates on (in order): a `lab/state/paused` file; the night window
(`cad.json`'s `lab.harvest` block -- night_start/night_end/day_allowed); today's GPU-hour
budget (summed from the ledger's own `seconds`, per LOCAL calendar day); the gpu-proxy's
`waiting` counter at :8087 (an unreachable/misshapen proxy status is read as 0 -- see
`_gpu_proxy_waiting()`'s own docstring for why that is a deliberate fail-open, not an
oversight); and the bank being exhausted (no spec has room for another pair). `--once
--spec-id` skips all of those (the plan's own words: "runs one spec now (smoke/tests)
without the gates") but NOT the GPU-window gate itself. There is no second locking scheme:
`lab/harvest_unit.sh` serialises concurrent timer fires with its own `flock` on
`lab/state/harvest.lock`, and `lab/gpu_window.sh` (frozen, unmodified) is the one thing
that ever takes `~/.openclaw/cad-build.lock`.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
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
from cad_v5.diagnose import diagnose  # noqa: E402
from cad_v5.config import (  # noqa: E402
    CODE_MODEL_STRONG, lab_config, maker_config, think_rung_available,
)
from geom_bands import score_against_reference  # noqa: E402
from lab import specbank  # noqa: E402
from lab import ship  # noqa: E402
from lab._armwindow import SpecgenAborted, arm_window  # noqa: E402

STATE_DIR    = HERE / "lab" / "state"
LEDGER_FILE  = STATE_DIR / "ledger.jsonl"
PAIRS_FILE   = STATE_DIR / "pairs.jsonl"
REVIEW_FILE  = STATE_DIR / "review.jsonl"
PROGRESS_FILE = STATE_DIR / "progress.json"
STATUS_FILE  = STATE_DIR / "status.json"
PAUSED_FILE  = STATE_DIR / "paused"
BUILDS_DIR   = STATE_DIR / "builds"

GPU_PROXY_STATUS_URL = "http://127.0.0.1:8087/"

GPU_WINDOW_HINT = (
    "harvest samples the maker arm in a loop for a whole unit: it must run inside a GPU "
    "window: `lab/harvest_unit.sh` (the timer's own launcher) or directly "
    "`lab/gpu_window.sh python3 lab/harvest.py --unit` / `--once --spec-id ID` (that "
    "holds ~/.openclaw/cad-build.lock, evicts the resident and the maker arm first, and "
    f"exports {ship.GPU_WINDOW_ENV}=1). Without the lock this run would take the GPU out "
    "from under any CAD build, benchmark card or lab job already using it. Pass "
    "--i-know-the-gpu-is-free only when the GPU is already free by hand."
)

_TIER34 = ("3", "4")


# ---------------------------------------------------------------------------
# Small local helpers -- time, JSONL/JSON I/O, all under a flock or atomic rename so a
# signal landing mid-write can never leave a half-written line or a corrupt state file.
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
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
    no code path that could leave a half-written line on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with open(path, "a") as f:
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
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(PROGRESS_FILE.read_text())
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
    """None when a --unit may proceed; otherwise the reason it was skipped. Checked AFTER
    the CAD_GPU_WINDOW gate (require_gpu_window is called by main() before this), which is
    always first and unconditional."""
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


# ---------------------------------------------------------------------------
# Prompt/usage recording -- a small recorder around engine._ollama (never reconstructed
# after the fact): every codegen path here (student, think, and the crash-salvage's own
# revise call) goes through _ollama exactly once per call, so wrapping it for the
# duration of one call captures the EXACT system/prompt/usage that call sent.
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


def _student_generate(spec: str, notes: list, temperature: float) -> tuple[str, dict]:
    """The exact production call fluid mode's strong rung makes (thinking off, whatever
    the active arm's launch default already is)."""
    with _PromptRecorder() as rec:
        code = engine.generate_code_raw(spec, notes, temperature=temperature)
    return code, rec.last


def _teacher_generate(spec: str, notes: list, temperature: float) -> tuple[str, dict]:
    """The think pass: engine._ollama(..., think=True) directly (Task 3 ruling), not the
    "+think" model-string suffix fluid_gen's repair_think knob uses -- this call is
    self-contained and does not depend on cad_engine's global _ACTIVE_CODE_MODEL for its
    thinking behaviour. Mirrors generate_code_raw's exact prompt shape (system/user
    layout) so the resulting training pair's prompt matches the student pass's shape
    byte-for-byte; think_rung_available() must already have been checked by the caller
    (see _choose_mode) before this is ever invoked, per _ollama()'s own documented
    contract for a caller passing think=True directly."""
    notes_str = "\n".join(f"- {n}" for n in (notes or []))
    prompt = (
        f"USER REQUEST (verbatim — every number here is AUTHORITATIVE):\n{spec}\n\n"
        + (f"Notes:\n{notes_str}\n\n" if notes_str else "")
        + "Write the build123d code:"
    )
    with _PromptRecorder() as rec:
        raw = engine._ollama(CODE_MODEL_STRONG, engine._CODE_SYSTEM, prompt,
                             timeout=engine._code_timeout(), temperature=temperature,
                             think=True)
    code = engine._patch_code(engine._strip_fences(raw),
                              wants=engine._wanted_edge_features(spec))
    return code, rec.last


def _execute_with_salvage(spec: str, code: str, build_dir: Path, prompt_info: dict,
                          gen_seconds: float) -> list[dict]:
    """Execute through fluid_gen._materialize (the exact execute/inspect/gate path a real
    fluid build uses) and, on a crash, ONE salvage attempt reusing fluid_gen's own
    diagnose()/_revise_on_repair_rung() -- never reimplemented, so the salvage logic
    itself can never drift from production's. Returns one or two rows:
    {code, prompt_info, m, seconds} -- the first row's `seconds` includes the codegen call
    that produced `code` (gen_seconds, passed in) plus this materialize; the salvage row's
    `seconds` is its OWN codegen+materialize time only, so summing every row's `seconds`
    equals this candidate's total wall time exactly once, never double-counted."""
    t0 = time.monotonic()
    m = fluid_gen._materialize(code, build_dir, spec)
    out = [{"code": code, "prompt_info": prompt_info, "m": m,
           "seconds": gen_seconds + (time.monotonic() - t0)}]
    if m["error"]:
        t1 = time.monotonic()
        try:
            _, hint = diagnose(m["error"])
            problem = m["error"] + (f"\nRepair hint: {hint}" if hint else "")
            with _PromptRecorder() as rec:
                fixed, _rung = fluid_gen._revise_on_repair_rung(spec, code, problem)
            m2 = fluid_gen._materialize(fixed, build_dir, spec)
            if not m2["error"]:
                m2["salvaged"] = True
            out.append({"code": fixed, "prompt_info": rec.last, "m": m2,
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
    """(verdict, band_info) -- verdict in ("good", "silver", "fail", "none"). band_info is
    the geom_bands.score_against_reference() dict, or {} when there was no reference (or
    the candidate never ran, so there is nothing to score)."""
    ok = m.get("error") is None
    band_info: dict = {}
    if ok and reference_stl:
        try:
            band_info = score_against_reference(build_dir / "build.step", Path(reference_stl))
        except Exception as e:
            band_info = {"band": "fail", "errors": [str(e)[:200]]}
    band = band_info.get("band")
    gate_hard = m.get("gate_hard") or []
    gate_spec = m.get("gate_spec") or []
    if _is_good(ok, gate_hard, gate_spec, band):
        return "good", band_info
    if ok and not gate_hard and gate_spec:
        return "silver", band_info
    if reference_stl and band == "near_miss":
        return "fail", band_info
    return "none", band_info


def _row_is_good(row: dict) -> bool:
    """Recomputes the SAME "good" predicate _classify uses, from what a ledger row
    already stored -- used by the status/pass-rate aggregation, so there is one
    definition of "good" shared by write time and read time."""
    return _is_good(bool(row.get("ok")), row.get("gate_hard") or [],
                    row.get("gate_spec") or [], row.get("band"))


def _persist_build(build_dir: Path, unit_id: str, spec_id: str, label: str) -> str:
    """Only "useful" candidates (good/silver/fail) get their build artifacts copied out of
    the per-candidate temp dir into a durable home under lab/state/builds/ -- everything
    else is discarded when the temp dir closes, so a night of mostly-rejected candidates
    does not balloon disk the way ~/.openclaw/cad-builds/ already has (2,123 dirs/829MB,
    a standing open item; harvest keeps its own artifacts out of that directory
    entirely)."""
    dest = BUILDS_DIR / f"{_safe_id(unit_id)}_{_safe_id(spec_id)}_{_safe_id(str(label))}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("build_source.py", "build.step", "build.stl", "build.png"):
        src = build_dir / name
        if src.is_file():
            (dest / name).write_bytes(src.read_bytes())
    return str(dest)


def _ledger_row(unit_id: str, spec_id: str, tier, mode: str, candidate: str,
                temperature: float, m: Optional[dict], band_info: Optional[dict],
                usage: Optional[dict], build_dir: Optional[str], seconds: Optional[float],
                error_text: Optional[str] = None) -> dict:
    m = m or {}
    band_info = band_info or {}
    error = error_text if error_text is not None else m.get("error")
    return {
        "ts": _now_utc(),
        "unit_id": unit_id,
        "spec_id": spec_id,
        "tier": tier,
        "arm": _current_arm_alias(),
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
    }


def _pair_source(mode: str) -> str:
    return "student" if mode == "student" else "teacher:think"


def _good_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict) -> dict:
    prompt_info = prompt_info or {}
    return {
        "id": f"p:{row['id']}:{hashlib.sha1(code.encode()).hexdigest()[:12]}",
        "spec_id": row["id"], "spec": row["spec"], "tier": row.get("tier"),
        "group": row.get("group"),
        "source": _pair_source(mode), "kind": "good",
        "band": band_info.get("band"),
        "arm": _current_arm_alias(), "temperature": temperature,
        "system": prompt_info.get("system", ""), "prompt": prompt_info.get("prompt", ""),
        "code": code, "bad_code": None, "problem": None,
        "facts": m.get("facts") or {},
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _fail_pair_row(row: dict, unit_id: str, mode: str, temperature: float,
                   prompt_info: Optional[dict], code: str, m: dict, band_info: dict) -> dict:
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
        "source": _pair_source(mode), "kind": "fail",
        "band": band_info.get("band"),
        "arm": _current_arm_alias(), "temperature": temperature,
        "system": prompt_info.get("system", ""), "prompt": prompt_info.get("prompt", ""),
        "code": None, "bad_code": code, "problem": problem,
        "facts": m.get("facts") or {},
        "verified": {"gate_hard": len(m.get("gate_hard") or []),
                    "gate_spec": len(m.get("gate_spec") or []),
                    "band": band_info.get("band")},
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _review_row(row: dict, unit_id: str, mode: str, temperature: float,
                prompt_info: Optional[dict], code: str, m: dict,
                render_path: Optional[str]) -> dict:
    prompt_info = prompt_info or {}
    notes = (m.get("gate_spec") or []) + (m.get("gate_adv") or [])
    return {
        "id": f"r:{row['id']}:{hashlib.sha1(code.encode()).hexdigest()[:12]}",
        "spec_id": row["id"], "spec": row["spec"], "tier": row.get("tier"),
        "code": code, "notes": notes, "render": render_path,
        "system": prompt_info.get("system", ""), "prompt": prompt_info.get("prompt", ""),
        "temperature": temperature, "arm": _current_arm_alias(), "pass": mode,
        "ts": _now_utc(), "unit_id": unit_id,
    }


def _existing_pair_code_hashes(spec_id: str) -> set:
    hashes = set()
    for r in _read_jsonl(PAIRS_FILE):
        if r.get("spec_id") == spec_id and r.get("code"):
            hashes.add(hashlib.sha1(r["code"].encode()).hexdigest())
    return hashes


# ---------------------------------------------------------------------------
# Per-spec sampling
# ---------------------------------------------------------------------------

def sample_spec(row: dict, cfg: dict, mode: str, unit_id: str, progress: dict) -> None:
    """Sample up to cfg["candidates"] candidates for one bank spec, in the given pass
    ("student" thinking-off, "think" thinking-on), verify each, and write a ledger row for
    every one plus a pair/review row where the verdict earns it. Never raises except
    SpecgenAborted (a signal), which propagates straight through so the caller's
    arm_window() bookend still runs its restore step."""
    spec_id, spec, tier = row["id"], row["spec"], row.get("tier")
    reference_stl = row.get("reference_stl")
    entry = progress.setdefault(
        spec_id, {"student_attempts": 0, "teacher_attempts": 0, "pairs": 0})
    max_pairs = cfg["max_pairs_per_spec"]
    if entry.get("pairs", 0) >= max_pairs:
        return

    notes = engine.retrieval_notes_for(spec)
    temps = cfg.get("temps") or [0.2]
    n = max(1, int(cfg.get("candidates", 1)))
    attempt_key = "student_attempts" if mode == "student" else "teacher_attempts"
    seen_hashes = _existing_pair_code_hashes(spec_id)
    good_found: list[dict] = []

    for i in range(n):
        # Stop sampling this spec once it already has enough DISTINCT good candidates --
        # temps are cycled ascending, so whatever is already in good_found is already the
        # lowest-temperature run found so far; a later (higher-temperature) candidate can
        # only ever be dropped by the "prefer lower temperature" rule below, never chosen
        # over it, so there is nothing left to gain by spending more GPU time on this spec.
        if entry.get("pairs", 0) + len(good_found) >= max_pairs:
            break
        temperature = temps[i % len(temps)]
        entry[attempt_key] = entry.get(attempt_key, 0) + 1
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
            _append_jsonl(LEDGER_FILE, _ledger_row(
                unit_id, spec_id, tier, mode, candidate_label, temperature, None, None,
                None, None, time.monotonic() - t0,
                error_text=f"codegen failed: {str(e)[:400]}"))
            _save_progress(progress)
            continue
        gen_seconds = time.monotonic() - t0

        with tempfile.TemporaryDirectory(prefix="harvest_") as td:
            build_dir = Path(td)
            attempts = _execute_with_salvage(spec, code, build_dir, prompt_info, gen_seconds)
            for idx, attempt in enumerate(attempts):
                label = candidate_label if idx == 0 else f"{candidate_label}-salvage"
                verdict, band_info = _classify(attempt["m"], reference_stl, build_dir)
                persisted = (_persist_build(build_dir, unit_id, spec_id, label)
                            if verdict != "none" else None)
                usage = (attempt["prompt_info"] or {}).get("usage")
                _append_jsonl(LEDGER_FILE, _ledger_row(
                    unit_id, spec_id, tier, mode, label, temperature, attempt["m"],
                    band_info, usage, persisted, attempt["seconds"]))
                if verdict == "good":
                    h = hashlib.sha1(attempt["code"].encode()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    good_found.append({**attempt, "temperature": temperature,
                                       "band_info": band_info})
                elif verdict == "silver":
                    render = str(Path(persisted) / "build.png") if persisted else None
                    _append_jsonl(REVIEW_FILE, _review_row(
                        row, unit_id, mode, temperature, attempt["prompt_info"],
                        attempt["code"], attempt["m"], render))
                elif verdict == "fail":
                    _append_jsonl(PAIRS_FILE, _fail_pair_row(
                        row, unit_id, mode, temperature, attempt["prompt_info"],
                        attempt["code"], attempt["m"], band_info))
        _save_progress(progress)

    # Dedup by code hash already enforced above; keep at most the remaining slots,
    # preferring lower temperature (Task 3: "at most max_pairs_per_spec distinct codes
    # per spec (dedup by code hash, prefer lower temperature)").
    good_found.sort(key=lambda g: g["temperature"])
    slots = max(0, max_pairs - entry.get("pairs", 0))
    for g in good_found[:slots]:
        _append_jsonl(PAIRS_FILE, _good_pair_row(
            row, unit_id, mode, g["temperature"], g["prompt_info"], g["code"], g["m"],
            g["band_info"]))
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
    for r in rows:
        by_tier[str(r.get("tier"))] = by_tier.get(str(r.get("tier")), 0) + 1
        by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1
    tier34 = sum(n for t, n in by_tier.items() if t in _TIER34)
    return {"total": total, "by_tier": by_tier, "by_source": by_source,
           "tier34_share": round(tier34 / total, 4) if total else 0.0}


def _good_pairs_tier34_share() -> float:
    """Used ONLY for spec-selection priority (Task 3: "tier 3-4 first while the pairs'
    tier34_share is under 0.40") -- deliberately GOOD pairs only, since scheduling cares
    about the training-signal balance the model will actually learn from, whereas the
    status file's reported pairs.tier34_share (see _pairs_stats) is the whole-dataset
    figure the Task 6 exit gate reads, over every kind."""
    rows = [r for r in _read_jsonl(PAIRS_FILE) if r.get("kind") == "good"]
    if not rows:
        return 0.0
    tier34 = sum(1 for r in rows if str(r.get("tier")) in _TIER34)
    return tier34 / len(rows)


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
    return {
        "updated": _now_utc(),
        "specs": {"total": len(bank), "by_tier": specs_by_tier},
        "pairs": _pairs_stats(),
        "ledger": {
            "attempts": len(ledger_rows),
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
    processed = 0
    if pool:
        prefer_tier34 = _good_pairs_tier34_share() < 0.40
        ordered = _order_specs(pool, progress, prefer_tier34)
        for row in ordered:
            if time.monotonic() >= deadline:
                break
            sample_spec(row, cfg, mode, unit_id, progress)
            processed += 1
    status = _write_status()
    return {"unit_id": unit_id, "mode": mode, "specs_processed": processed,
           "status": status}


def run_once(spec_id: str, cfg: dict) -> dict:
    unit_id = _now_utc()
    bank = specbank.load_bank()
    row = next((r for r in bank if r.get("id") == spec_id), None)
    if row is None:
        raise SystemExit(f"harvest --once: no spec with id {spec_id!r} in the bank")
    progress = _load_progress()
    entry = progress.get(spec_id, {})
    mode = "student"
    if entry.get("student_attempts", 0) >= 2 and think_rung_available():
        mode = "think"
    sample_spec(row, cfg, mode, unit_id, progress)
    status = _write_status()
    return {"unit_id": unit_id, "spec_id": spec_id, "mode": mode, "status": status}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_in_window(arm: Optional[str], body) -> int:
    """Shared try/finally shape between --unit and --once: the abort exception
    (SpecgenAborted, from a signal) is caught here ONLY to report it and set a non-zero
    exit code -- arm_window()'s own finally has already run by the time this except
    fires, since it lives inside the `with` block."""
    rc, reason = 0, None
    try:
        with arm_window(arm):
            body()
    except SpecgenAborted as e:
        rc, reason = 1, str(e)
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
    ap.add_argument("--arm", default=None,
                    help="maker arm for this run (default: cad.json's maker block)")
    ap.add_argument("--i-know-the-gpu-is-free", action="store_true",
                    help="run outside a GPU window (only when the GPU was freed by hand)")
    a = ap.parse_args()

    if a.status:
        print(json.dumps(_compute_status(), indent=2))
        return 0

    if not a.unit and not (a.once and a.spec_id):
        ap.error("pass --unit, --once --spec-id ID, or --status")

    # Before ANYTHING that touches a model or a service (fix pattern shared with
    # lab/specgen.py): no arm switch, no signal handlers, no cad.json read, no model call.
    ship.require_gpu_window(a, GPU_WINDOW_HINT)

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

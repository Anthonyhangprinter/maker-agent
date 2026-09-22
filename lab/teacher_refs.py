#!/usr/bin/env python3
"""lab/teacher_refs.py -- import-teacher-refs: turn August's teacher-solved code into
reference geometry for the Phase 3 spec bank (the 414 `source: "teacher-suite"` rows in
lab/state/specs.jsonl).

Today the harvest unit (lab/harvest.py) confirms a model-written program as a training pair
either by agreement between two samples of the same model (weak evidence) or by matching a
REFERENCE geometry (strong evidence: `reference_stl` on a bank row, consumed via
geom_bands.score_against_reference). Only owner-supplied references exist so far, and none
are on disk yet. But every `teacher-suite` spec was already solved once, in August, by a
stronger teacher model: `~/.openclaw/cad-sftpairs.jsonl` holds that accepted code (rows with
`source` "teacher" or "teacher-human-accepted"; fields include `spec`, `code`,
`teacher_spec_id`, `verified`). This module builds and RE-GATES that code under TODAY's rules
and, for anything that still passes, materialises it as reference geometry on the matching
bank row. The teacher code is used ONLY as geometry to check the local model's own code
against -- a cross-model check -- it never becomes training text itself.

CPU only, no model calls. `cad_engine._ollama` is patched to raise the instant this module
imports cad_engine, and `cad_engine._ensure_default_server` / `_pause_default_server_for`
are never called anywhere below -- every part is built and measured through
`cad_engine.run_step` / `run_inspect` / `parse_facts` (each a local subprocess) plus a
dedicated child interpreter for build123d's STEP->STL conversion.

Why lab/harvest.py's own `strict_envelope_check` runs in a SEPARATE subprocess rather than
being imported directly here: this branch's Task 3 harvest (lab/harvest.py) and its config
seam (cad_v5/config.py) are being actively edited by another agent on this SAME worktree
while this module is written and run. Importing lab.harvest in-process would tie this run's
liveness (it can take 1-2 hours) to whatever that file's on-disk state happens to be at any
one instant; a half-saved edit at the wrong moment would take the whole run down. A fresh
child interpreter re-imports it per call instead, so a bad moment fails (and is retried for)
one candidate, never the run.

    python3 lab/specbank.py import-teacher-refs [--dry-run] [--workers 3] [--limit N]
                                                [--pairs PATH] [--decisions PATH]

`--dry-run` only counts (teacher rows read / rejected by review / matched to the bank /
already-referenced) -- it never builds anything and never writes a file (no report, no STL,
no bank update). A real run (`--limit N` or unbounded) builds and gates every remaining
matched candidate up to `--workers` (max 3) at a time, admits the ones that pass today's gate,
converts each admitted STEP to STL, updates the bank rows and writes
lab/state/teacher_refs_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Every subprocess spawned below (including the ones cad_engine.run_step/run_inspect start
# internally) inherits this via os.environ -- OpenCascade resets the locale to C after the
# first mesh export, and an unencoded read then decodes as ASCII (see tests/test_engine_locale.py
# for the incident this guards against).
os.environ.setdefault("PYTHONUTF8", "1")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import cad_engine as engine  # noqa: E402
import harvest_census as hc  # noqa: E402
from lab import specbank  # noqa: E402


def _forbid_model_calls() -> None:
    """Patch cad_engine._ollama to raise immediately. This module never generates code --
    it only builds/inspects/gates code that already exists as text in
    ~/.openclaw/cad-sftpairs.jsonl. Idempotent (safe to call more than once)."""
    def _raise(*_a, **_kw):
        raise RuntimeError(
            "lab/teacher_refs.py must never call a model -- it only builds and re-gates "
            "code that already exists (the August teacher pass). Something tried to call "
            "cad_engine._ollama, which is a bug in this module, not a missing feature.")
    engine._ollama = _raise


_forbid_model_calls()

DEFAULT_PAIRS_FILE = Path.home() / ".openclaw" / "cad-sftpairs.jsonl"
DEFAULT_DECISIONS_FILE = Path.home() / ".openclaw" / "cad-review-decisions.jsonl"


def report_file_path() -> Path:
    """lab/state/teacher_refs_report.json, resolved from `specbank.REFS_DIR` at CALL time
    (not a module-level constant) -- so a test that monkeypatches `specbank.REFS_DIR` to a
    tmp directory (the same pattern tests/test_lab_specbank.py's own fixture uses) gets a
    report path under that same tmp directory, never the real lab/state/."""
    return specbank.REFS_DIR.parent / "teacher_refs_report.json"

# kind="good" is the only kind these two sources ever carry in practice (verified against
# the real file, 2026-09-19: 399/399 rows), but the filter is explicit anyway -- a fail->fix
# pair sharing one of these sources in the future would carry `bad_code`, not a clean solve.
TEACHER_SOURCES = ("teacher", "teacher-human-accepted")
ENVELOPE_TOL_MM = 0.2
REFERENCE_SOURCE_TAG = "teacher-claude"

_STRICT_CHECK_RETRIES = 3
_STRICT_CHECK_RETRY_DELAY_S = 2.0
_STRICT_CHECK_TIMEOUT_S = 60
_STEP_TO_STL_TIMEOUT_S = 120

# Fixed at import time (this module's own HERE never moves); only the per-call payload
# (spec/facts/tol_mm, or step/stl paths) travels over stdin as JSON, so no per-call string
# formatting or shell-escaping is needed for either child script.
_STRICT_CHECK_SCRIPT = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(HERE)!r})\n"
    f"sys.path.insert(0, {str(HERE / 'scripts')!r})\n"
    "import cad_engine as engine\n"
    "def _raise(*a, **k):\n"
    "    raise RuntimeError('lab/teacher_refs.py subprocess must never call a model')\n"
    "engine._ollama = _raise\n"
    "from lab.harvest import strict_envelope_check\n"
    "payload = json.loads(sys.stdin.read())\n"
    "result = strict_envelope_check(payload['spec'], payload['facts'],\n"
    "                               payload.get('tol_mm', 0.2))\n"
    "print(json.dumps({'result': result}))\n"
)

_STEP_TO_STL_SCRIPT = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(HERE / 'scripts')!r})\n"
    "from geom_bands import step_to_stl\n"
    "payload = json.loads(sys.stdin.read())\n"
    "step_to_stl(payload['step'], payload['stl'])\n"
)


# ---------------------------------------------------------------------------
# Loading + filtering the teacher pairs
# ---------------------------------------------------------------------------

def load_teacher_pairs(pairs_path: Path) -> list[dict]:
    """Every row of `pairs_path` (normally ~/.openclaw/cad-sftpairs.jsonl) whose source is
    "teacher" or "teacher-human-accepted" -- the August teacher pass's ACCEPTED code, never
    "teacher-repair" (an edit-turn pair), "gift-*"/"retro" (amplification pairs) or an unset
    source (unlabelled legacy rows)."""
    rows: list[dict] = []
    if not pairs_path.exists():
        return rows
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("source") in TEACHER_SOURCES:
                rows.append(d)
    return rows


def load_review_verdicts(decisions_path: Path) -> dict[str, dict]:
    """Latest human verdict per spec id -- later lines win. Mirrors
    scripts/compile_sft.py's latest_verdicts() exactly (same file, same rule), so a teacher
    row already excluded from SFT compilation for a human reject is excluded from becoming
    reference geometry too."""
    v: dict[str, dict] = {}
    if not decisions_path.exists():
        return v
    with open(decisions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("id"):
                v[d["id"]] = d
    return v


def exclude_reviewed_rejects(rows: list[dict], verdicts: dict) -> tuple[list[dict], list[dict]]:
    """(kept, rejected) using the exact same matching rule as
    scripts/compile_sft.py's exclude_rejected(): a row is dropped only when its OWN
    `teacher_spec_id` has a verdict of "reject" in `verdicts`. Copied rather than imported --
    compile_sft.py is a benchmark/compile-time script this module has no other reason to
    depend on."""
    kept, rejected = [], []
    for r in rows:
        sid = r.get("teacher_spec_id")
        if sid and verdicts.get(sid, {}).get("verdict") == "reject":
            rejected.append(r)
        else:
            kept.append(r)
    return kept, rejected


def _preferred(rows: list[dict]) -> dict:
    """Pick ONE row among several accepted teacher rows for the same spec: prefer
    "teacher-human-accepted" over plain "teacher", then the newest by `timestamp`
    (ISO-8601 strings from this same generator, lexically sortable)."""
    def rank(r: dict) -> tuple:
        return (r.get("source") == "teacher-human-accepted", r.get("timestamp") or "")
    return max(rows, key=rank)


def bank_teacher_rows_by_key(bank: list[dict]) -> dict[str, dict]:
    return {r["key"]: r for r in bank if r.get("source") == "teacher-suite"}


def match_to_bank(rows: list[dict], bank_by_key: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Group `rows` by the bank's own contamination key (sha1 of the spec text -- NOT
    `teacher_spec_id`, whose short ids like "P01" are only unique per suite file), resolve
    duplicates to one preferred row per key, and match against `bank_by_key`.

    Returns (key -> {"pair": chosen_row, "bank_row": ...}, unmatched_rows). `unmatched` is
    every teacher row whose spec text does not sha1-match any teacher-suite bank row
    (verified empty against the real files as of 2026-09-19; reported, never silently
    dropped, in case the bank or the pairs file ever drifts)."""
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(hc._key(r.get("spec", "")), []).append(r)
    matched: dict[str, dict] = {}
    unmatched: list[dict] = []
    for key, group in grouped.items():
        bank_row = bank_by_key.get(key)
        if bank_row is None:
            unmatched.extend(group)
            continue
        matched[key] = {"pair": _preferred(group), "bank_row": bank_row}
    return matched, unmatched


def _suite_of(bank_row: dict) -> str:
    rid = bank_row.get("id", "")
    if rid.startswith("t:"):
        parts = rid.split(":", 2)
        if len(parts) >= 2:
            return parts[1]
    return "unknown"


# ---------------------------------------------------------------------------
# Subprocess helpers -- strict envelope check (harvest.py, actively edited elsewhere) and
# STEP -> STL conversion (build123d/OCCT is not something to share across threads).
# ---------------------------------------------------------------------------

def strict_envelope_check_subprocess(spec: str, facts: dict, tol_mm: float = ENVELOPE_TOL_MM,
                                     retries: int = _STRICT_CHECK_RETRIES
                                     ) -> tuple[bool, Optional[str]]:
    """Run lab/harvest.py's strict_envelope_check in a fresh child interpreter (see the
    module docstring for why). Returns (ok, value): ok is False only when the subprocess
    itself could not be completed after `retries` attempts (import error, crash, timeout) --
    reported as a gate failure, never silently treated as a pass. When ok is True, `value`
    is strict_envelope_check's own return: None (no envelope stated, or it matches) or a
    "strict_envelope: ..." mismatch string."""
    payload = json.dumps({"spec": spec, "facts": facts, "tol_mm": tol_mm})
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    last_err = ""
    for _attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _STRICT_CHECK_SCRIPT],
                input=payload, capture_output=True, encoding="utf-8", errors="replace",
                timeout=_STRICT_CHECK_TIMEOUT_S, env=env,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(_STRICT_CHECK_RETRY_DELAY_S)
            continue
        if proc.returncode != 0:
            last_err = (proc.stderr or proc.stdout or "non-zero exit")[-500:]
            time.sleep(_STRICT_CHECK_RETRY_DELAY_S)
            continue
        out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not out_lines:
            last_err = f"no output; stderr={proc.stderr[-300:]}"
            time.sleep(_STRICT_CHECK_RETRY_DELAY_S)
            continue
        try:
            out = json.loads(out_lines[-1])
            return True, out.get("result")
        except Exception as e:
            last_err = f"unparseable output: {e}: {proc.stdout[-300:]}"
            time.sleep(_STRICT_CHECK_RETRY_DELAY_S)
            continue
    return False, f"strict_envelope_check subprocess failed after {retries} attempts: {last_err}"


def step_to_stl_subprocess(step_path: Path, stl_path: Path,
                          timeout: int = _STEP_TO_STL_TIMEOUT_S) -> None:
    """geom_bands.step_to_stl in a fresh child interpreter (build123d/OCCT has no
    documented thread-safety contract for concurrent modelling calls inside one
    interpreter, and up to 3 candidates build in parallel here)."""
    payload = json.dumps({"step": str(step_path), "stl": str(stl_path)})
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", _STEP_TO_STL_SCRIPT], input=payload,
                          capture_output=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)
    if proc.returncode != 0 or not stl_path.exists():
        raise RuntimeError(f"step_to_stl failed: {(proc.stderr or proc.stdout)[-500:]}")


# ---------------------------------------------------------------------------
# Build + re-gate one candidate
# ---------------------------------------------------------------------------

def reference_facts_from(facts: dict) -> dict:
    """The subset of a fully-measured `facts` dict worth caching on the bank row: solids,
    faces (+ per-type cylindrical/conical face counts when carried), volume, bbox, bores
    and hole_groups -- everything a future consumer (geom_bands scoring, a human audit)
    would want without re-inspecting the STEP."""
    out: dict = {"solids": facts.get("solids"), "volume": facts.get("volume"),
                "bbox": facts.get("bbox")}
    if "faces" in facts:
        out["faces"] = facts["faces"]
    for k in ("cyl_faces", "cone_faces"):
        if k in facts:
            out[k] = facts[k]
    if facts.get("bores"):
        out["bores"] = facts["bores"]
    if facts.get("hole_groups"):
        out["hole_groups"] = facts["hole_groups"]
    return out


def build_and_gate(spec: str, code: str, tol_mm: float = ENVELOPE_TOL_MM) -> dict:
    """Run one teacher program through execute -> inspect -> parse -> gate, using ONLY the
    stable half of lab/harvest.py's own `_regate` path -- cad_engine.run_step / run_inspect
    / parse_facts / reconcile_expected / verify_expected -- directly and in-process (pure
    subprocess wrappers and pure functions, none of them in a file being edited elsewhere on
    this worktree), plus the strict envelope check via a fresh subprocess (see above for
    why). `_regate` itself is not called: it lives in lab/harvest.py.

    Returns one of:
      {"verdict": "crashed", "reason": str, "work_dir": Path}
      {"verdict": "failed-gate", "reason": str, "facts": dict, "work_dir": Path}
      {"verdict": "admitted", "facts": dict, "step_path": Path, "work_dir": Path}

    The caller owns `work_dir` (and, for an admitted candidate, must copy `step_path`
    somewhere permanent before cleaning it up)."""
    work_dir = Path(tempfile.mkdtemp(prefix="teacher-ref-"))
    result: dict = {"work_dir": work_dir}
    try:
        step_path, _log = engine.run_step(code, work_dir)
    except Exception as e:
        result.update(verdict="crashed", reason=f"run_step: {str(e)[:300]}")
        return result
    try:
        insp = engine.run_inspect(step_path)
    except Exception as e:
        result.update(verdict="crashed", reason=f"run_inspect: {str(e)[:300]}")
        return result
    if not insp.get("valid"):
        result.update(verdict="failed-gate", reason="inspect reported the STEP invalid",
                     facts={})
        return result
    facts = engine.parse_facts(insp["output"])
    missing = [k for k in ("solids", "volume", "bbox")
              if not facts.get(k) and facts.get(k) != 0]
    if missing:
        result.update(verdict="failed-gate",
                     reason=f"inspect did not measure: {', '.join(missing)}", facts=facts)
        return result
    brief: dict = {}
    engine.reconcile_expected(brief, spec)
    hard, notes = engine.verify_expected(facts, brief["expected"], spec=spec)
    gate_spec = [n for n in (notes or []) if n.startswith("[spec]")]
    if hard:
        result.update(verdict="failed-gate", reason="gate_hard: " + "; ".join(hard)[:400],
                     facts=facts)
        return result
    if gate_spec:
        result.update(verdict="failed-gate",
                     reason="gate_spec: " + "; ".join(gate_spec)[:400], facts=facts)
        return result
    ok, strict_value = strict_envelope_check_subprocess(spec, facts, tol_mm)
    if not ok:
        result.update(verdict="failed-gate", reason=strict_value, facts=facts)
        return result
    if strict_value:
        result.update(verdict="failed-gate", reason=strict_value, facts=facts)
        return result
    result.update(verdict="admitted", facts=facts, step_path=step_path)
    return result


def _reason_bucket(reason: Optional[str]) -> str:
    """Collapse a specific reason string into a short category, so "most common reasons"
    means something -- every raw message is otherwise unique (it names exact mm/mesh
    values)."""
    reason = reason or ""
    for prefix in ("run_step:", "run_inspect:", "gate_hard:", "gate_spec:",
                  "strict_envelope:", "strict_through_holes:",
                  "inspect did not measure", "inspect reported the STEP invalid",
                  "strict_envelope_check subprocess failed"):
        if reason.startswith(prefix):
            return prefix.rstrip(":")
    return reason[:40]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_teacher_refs(pairs_path: Path = DEFAULT_PAIRS_FILE,
                        decisions_path: Path = DEFAULT_DECISIONS_FILE,
                        dry_run: bool = False, workers: int = 3,
                        limit: Optional[int] = None) -> dict:
    """The whole import-teacher-refs run. See the module docstring for the two modes
    (dry-run: counts only, writes nothing; real: builds, gates, admits, writes)."""
    workers = max(1, min(int(workers or 1), 3))

    all_rows = load_teacher_pairs(pairs_path)
    by_source: dict[str, int] = {}
    for r in all_rows:
        by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1

    verdicts = load_review_verdicts(decisions_path)
    kept, rejected = exclude_reviewed_rejects(all_rows, verdicts)

    bank = specbank.load_bank()
    bank_by_key = bank_teacher_rows_by_key(bank)
    matched, unmatched = match_to_bank(kept, bank_by_key)

    already_referenced = {k: v for k, v in matched.items() if v["bank_row"].get("reference_stl")}
    candidates = {k: v for k, v in matched.items() if k not in already_referenced}

    counts = {
        "teacher_rows_read": {"total": len(all_rows), "by_source": by_source},
        "rejected_by_review": {"total": len(rejected),
                               "ids": sorted({r.get("teacher_spec_id") for r in rejected})},
        "matched_to_bank": len(matched),
        "unmatched": len(unmatched),
        "already_has_reference": len(already_referenced),
    }

    if dry_run:
        return {"dry_run": True, "counts": {**counts, "would_attempt": len(candidates)}}

    ordered_keys = sorted(candidates.keys(), key=lambda k: candidates[k]["bank_row"].get("id", k))
    if limit is not None:
        ordered_keys = ordered_keys[:limit]
    counts["attempted"] = len(ordered_keys)

    admitted: dict[str, dict] = {}
    crashed: list[dict] = []
    failed_gate: list[dict] = []
    work_dirs: list[Path] = []

    def _run_one(key: str) -> tuple[str, dict]:
        entry = candidates[key]
        spec = entry["bank_row"]["spec"]
        code = entry["pair"]["code"]
        return key, build_and_gate(spec, code)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, key): key for key in ordered_keys}
        for fut in as_completed(futures):
            key, res = fut.result()
            work_dirs.append(res["work_dir"])
            bank_row = candidates[key]["bank_row"]
            tier = bank_row.get("tier")
            suite = _suite_of(bank_row)
            spec_id = bank_row.get("id", key)
            if res["verdict"] == "crashed":
                crashed.append({"id": spec_id, "tier": tier, "suite": suite,
                               "reason": res["reason"]})
            elif res["verdict"] == "failed-gate":
                failed_gate.append({"id": spec_id, "tier": tier, "suite": suite,
                                   "reason": res["reason"]})
            else:
                admitted[key] = {"bank_row": bank_row, "facts": res["facts"],
                                "step_path": res["step_path"], "work_dir": res["work_dir"]}

    # Convert admitted STEPs to STL (still up to `workers` at a time -- independent child
    # interpreters, no shared OCCT state) before any work_dir is cleaned up.
    specbank.REFS_DIR.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict] = {}
    conversion_failed: list[dict] = []

    def _convert_one(key: str) -> tuple[str, Optional[Exception]]:
        entry = admitted[key]
        dest = specbank.REFS_DIR / f"{key[:16]}.stl"
        try:
            if not dest.exists():
                step_to_stl_subprocess(entry["step_path"], dest)
            return key, None
        except Exception as e:
            return key, e

    if admitted:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_convert_one, key): key for key in admitted}
            for fut in as_completed(futures):
                key, err = fut.result()
                bank_row = admitted[key]["bank_row"]
                if err is not None:
                    conversion_failed.append({"id": bank_row.get("id", key), "tier": bank_row.get("tier"),
                                             "suite": _suite_of(bank_row),
                                             "reason": f"step_to_stl: {str(err)[:300]}"})
                    continue
                dest = specbank.REFS_DIR / f"{key[:16]}.stl"
                updates[key] = {
                    "reference_stl": str(dest),
                    "reference_source": REFERENCE_SOURCE_TAG,
                    "reference_facts": reference_facts_from(admitted[key]["facts"]),
                    "reference_added": _now(),
                }

    for wd in work_dirs:
        shutil.rmtree(wd, ignore_errors=True)

    apply_result = specbank.apply_reference_updates(updates) if updates else \
        {"applied": [], "skipped_already_has_reference": [], "not_found": []}

    def _bucket(rows: list[dict]) -> dict:
        by_tier: dict[str, int] = {}
        by_suite: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        for r in rows:
            t = str(r.get("tier"))
            by_tier[t] = by_tier.get(t, 0) + 1
            by_suite[r["suite"]] = by_suite.get(r["suite"], 0) + 1
            b = _reason_bucket(r.get("reason"))
            by_reason[b] = by_reason.get(b, 0) + 1
        return {"total": len(rows), "by_tier": by_tier, "by_suite": by_suite,
               "by_reason": by_reason}

    admitted_rows = [{"id": v["bank_row"].get("id", k), "tier": v["bank_row"].get("tier"),
                      "suite": _suite_of(v["bank_row"])} for k, v in admitted.items()
                     if k in updates]
    admitted_bucket = {"total": len(admitted_rows), "by_tier": {}, "by_suite": {}}
    for r in admitted_rows:
        t = str(r["tier"])
        admitted_bucket["by_tier"][t] = admitted_bucket["by_tier"].get(t, 0) + 1
        admitted_bucket["by_suite"][r["suite"]] = admitted_bucket["by_suite"].get(r["suite"], 0) + 1

    non_admitted = (
        [{"id": r["id"], "tier": r["tier"], "suite": r["suite"], "verdict": "crashed",
         "reason": r["reason"]} for r in crashed]
        + [{"id": r["id"], "tier": r["tier"], "suite": r["suite"], "verdict": "failed-gate",
           "reason": r["reason"]} for r in failed_gate]
        + [{"id": r["id"], "tier": r["tier"], "suite": r["suite"],
           "verdict": "stl-conversion-failed", "reason": r["reason"]}
          for r in conversion_failed]
    )

    report = {
        "dry_run": False,
        "generated": _now(),
        "counts": {
            **counts,
            "crashed": _bucket(crashed),
            "failed_gate": _bucket(failed_gate),
            "stl_conversion_failed": _bucket(conversion_failed),
            "admitted": admitted_bucket,
        },
        "non_admitted": non_admitted,
        "bank_update": apply_result,
    }
    report_path = report_file_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_FILE)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_FILE)
    a = ap.parse_args(argv)
    report = import_teacher_refs(pairs_path=a.pairs, decisions_path=a.decisions,
                                 dry_run=a.dry_run, workers=a.workers, limit=a.limit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""lab/specbank.py -- build and extend the Phase 3 spec bank (lab/state/specs.jsonl).

The bank is the pool the harvest unit (Task 3) samples from: the 414 existing teacher
specs (benchmarks/teacher-*/specs.json, hand-written or cloud-generated in earlier
phases) plus new specs written locally by the maker model (lab/specgen.py) plus, when
present, owner reference parts (~/CAD/references/<name>/spec.txt + model.step|model.stl).

Every row is tier-tagged and contamination-keyed against every card suite BEFORE it is
admitted: a spec is refused if its exact text matches a card-suite spec
(harvest_census.suite_keys()), if its 40-char slug uniquely identifies one spec in a card
suite (harvest_census.suite_slug_counts() -- the same per-suite-unique-slug rule
scripts/run_card.py and lab/data.py already use), or if it is already in the bank by
exact key. Refusal is silent per-item (the caller sees counts + reasons), never a hard
stop for the whole batch.

  python3 lab/specbank.py import-teacher [--dry-run]
  python3 lab/specbank.py import-references [--dir ~/CAD/references] [--dry-run]
  python3 lab/specbank.py import-teacher-refs [--dry-run] [--workers 3] [--limit N]
  python3 lab/specbank.py add specs.json --source specgen [--group plate] [--tier 2]
  python3 lab/specbank.py stats

Row shape: {id, spec, tier, group, source, key, added} (+ reference_stl/reference_source/
reference_facts/reference_added for a row with reference geometry -- an owner-reference row,
or a teacher-suite row promoted by `import-teacher-refs`, see lab/teacher_refs.py). New rows
are appended to lab/state/specs.jsonl under an exclusive flock so concurrent writers
(specgen batches now, the harvest unit later) never interleave partial lines; that append
path never rewrites the file wholesale. `apply_reference_updates` (used by
`import-teacher-refs` to attach reference geometry to EXISTING rows) is the one exception --
see its own docstring for why it still never renames the file even though it rewrites it.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))

import harvest_census as hc  # noqa: E402
from lab.data import default_contamination_sets  # noqa: E402 -- reuse, not a third copy

STATE_DIR = HERE / "lab" / "state"
SPECS_FILE = STATE_DIR / "specs.jsonl"
REFS_DIR = STATE_DIR / "refs"
REFERENCES_ROOT = Path.home() / "CAD" / "references"

# (suite tag, path) -- the five teacher/training spec files. Card (eval) suites are a
# different list (harvest_census.CARD_SUITES) and are never read here as a source.
TEACHER_FILES = [
    ("teacher-pilot", HERE / "benchmarks" / "teacher-pilot" / "specs.json"),
    ("teacher-complex", HERE / "benchmarks" / "teacher-complex" / "specs.json"),
    ("teacher-hard", HERE / "benchmarks" / "teacher-hard" / "specs.json"),
    ("teacher-mech2", HERE / "benchmarks" / "teacher-mech2" / "specs.json"),
    ("teacher-batch2", HERE / "benchmarks" / "teacher-batch2" / "specs.json"),
]


def _load_items(p: Path) -> list[dict]:
    d = json.loads(p.read_text())
    return d["benchmarks"] if isinstance(d, dict) else d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bank(path: Path | None = None) -> list[dict]:
    # `path` defaults dynamically to the module-level SPECS_FILE (looked up at CALL time,
    # not bound as a mutable default argument) so tests that monkeypatch
    # specbank.SPECS_FILE to a tmp path are honoured by every caller that omits `path`.
    path = path if path is not None else SPECS_FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_atomic(path: Path, rows: list[dict]) -> None:
    """Append `rows` to a JSONL file under an exclusive flock -- concurrent writers
    (this module and, later, lab/harvest.py) never interleave partial lines, and a
    crash mid-write loses at most the in-flight append, never corrupts earlier rows.

    Used by callers that already hold their own read-decide-append lock (see
    _locked_bank below) and just need a plain flocked append elsewhere (e.g. a future
    caller writing to a different JSONL file); add_items/import_teacher/
    import_references no longer call this for SPECS_FILE itself -- they write through
    the file handle _locked_bank yields, inside the SAME lock the decision was made
    under."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def _locked_bank():
    """Open lab/state/specs.jsonl for read-then-append under ONE exclusive flock that
    spans the whole read-decide-append window (fix round 1, finding 6): without this,
    add_items/import_teacher/import_references each read the whole bank once via
    load_bank(), decide what is a duplicate from that snapshot, and only locked the file
    for the final write -- so two concurrent callers (a human running `specbank.py
    import-references` while specgen.py's multi-hour run is also mid-add_items) could
    both read the bank before either wrote, both decide the same new spec is not yet a
    duplicate, and both append it.

    Yields (file handle positioned at EOF, ready to append; current bank rows as a list
    of dicts read fresh under this lock). Callers must build their bank_keys set from the
    yielded rows, not a separate load_bank() call, and write new rows straight to the
    yielded handle -- never open SPECS_FILE again inside the `with` block, which would
    deadlock against this process's own lock."""
    SPECS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pre_existed = SPECS_FILE.exists()
    with open(SPECS_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            rows = [json.loads(line) for line in f.read().splitlines() if line.strip()]
            f.seek(0, 2)   # back to EOF -- writes below must never overwrite what we just read
            yield f, rows
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    # Opening in "a+" mode creates the file even when nothing is ever written (a dry
    # run, or a batch where every item was refused) -- undo that side effect so a
    # bank that did not exist before this call still does not exist after it.
    if not pre_existed and SPECS_FILE.exists() and SPECS_FILE.stat().st_size == 0:
        SPECS_FILE.unlink()


def contamination_sets():
    """(exact suite keys, per-suite-unique suite slugs) -- straight from lab.data, the
    same primitive scripts/run_card.py's contamination() and the eventual compiler use.
    Not recomputed here: a second copy of this rule drifting from the first is exactly
    the kind of bug a shared contamination guard exists to prevent."""
    return default_contamination_sets()


def refusal_reason(key: str, spec: str, suite_keys: set, suite_slugs: set,
                    bank_keys: set) -> str | None:
    """None when `spec` (already reduced to its contamination `key`) may be admitted;
    otherwise the reason it is refused. Bank-duplicate is checked first since it is the
    cheapest and most common case once a bank exists."""
    if key in bank_keys:
        return "duplicate-in-bank"
    if key in suite_keys:
        return "suite-exact-match"
    if hc._slug(spec, 40) in suite_slugs:
        return "suite-unique-slug-match"
    return None


def add_items(items: list[dict], source: str, group_default: str | None = None,
              tier_default: int | None = None, dry_run: bool = False) -> dict:
    """Admit a list of {"spec", "tier"?, "group"?, "id"?} dicts (specgen's own output
    shape, or a hand-authored JSON file via the `add` CLI) into the bank. Returns
    {"accepted": int, "refused": [{"spec", "reason"}, ...]}.

    The whole read-decide-append window runs under _locked_bank's single flock (fix
    round 1, finding 6), including for a dry run -- a dry run never writes, but reading
    a consistent bank snapshot under the same lock a real run would use keeps its report
    honest against a concurrent writer."""
    with _locked_bank() as (f, bank):
        bank_keys = {r["key"] for r in bank}
        suite_keys, suite_slugs = contamination_sets()
        accepted: list[dict] = []
        refused: list[dict] = []
        now = _now()
        for it in items:
            spec = str(it.get("spec", "")).strip()
            if not spec:
                refused.append({"spec": spec, "reason": "empty-spec"})
                continue
            key = hc._key(spec)
            reason = refusal_reason(key, spec, suite_keys, suite_slugs, bank_keys)
            if reason:
                refused.append({"spec": spec, "reason": reason})
                continue
            tier = int(it.get("tier") or tier_default or 2)
            group = it.get("group") or group_default or "unspecified"
            row = {
                "id": it.get("id") or f"g:{source}:{key[:16]}",
                "spec": spec,
                "tier": tier,
                "group": group,
                "source": source,
                "key": key,
                "added": now,
            }
            if it.get("reference_stl"):
                row["reference_stl"] = it["reference_stl"]
            accepted.append(row)
            bank_keys.add(key)   # dedup within this batch too, not just against the disk bank
        if accepted and not dry_run:
            for row in accepted:
                f.write(json.dumps(row) + "\n")
            f.flush()
    return {"accepted": len(accepted), "refused": refused}


def import_teacher(dry_run: bool = False) -> dict:
    """Read the five benchmarks/teacher-*/specs.json files, tag each admitted row
    `id = "t:<suite>:<orig id>"`, `source = "teacher-suite"`, and append. Per-suite and
    total counts are returned so a stale/incomplete teacher corpus is visible, not
    silently short. Whole read-decide-append window under _locked_bank's single flock
    (fix round 1, finding 6), same as add_items."""
    with _locked_bank() as (f, bank):
        bank_keys = {r["key"] for r in bank}
        suite_keys, suite_slugs = contamination_sets()
        accepted: list[dict] = []
        refused: list[tuple[str, str, str]] = []
        per_suite: dict[str, dict] = {}
        now = _now()
        for suite, path in TEACHER_FILES:
            if not path.exists():
                per_suite[suite] = {"found": 0, "accepted": 0, "refused": 0}
                continue
            items = _load_items(path)
            suite_accepted = 0
            suite_refused = 0
            for it in items:
                spec = str(it.get("spec", "")).strip()
                orig_id = str(it.get("id", ""))
                if not spec:
                    refused.append((suite, orig_id, "empty-spec"))
                    suite_refused += 1
                    continue
                key = hc._key(spec)
                reason = refusal_reason(key, spec, suite_keys, suite_slugs, bank_keys)
                if reason:
                    refused.append((suite, orig_id, reason))
                    suite_refused += 1
                    continue
                row = {
                    "id": f"t:{suite}:{orig_id}",
                    "spec": spec,
                    "tier": int(it.get("tier") or 2),
                    "group": it.get("group", ""),
                    "source": "teacher-suite",
                    "key": key,
                    "added": now,
                }
                accepted.append(row)
                bank_keys.add(key)
                suite_accepted += 1
            per_suite[suite] = {"found": len(items), "accepted": suite_accepted,
                                "refused": suite_refused}
        if accepted and not dry_run:
            for row in accepted:
                f.write(json.dumps(row) + "\n")
            f.flush()
    return {
        "per_suite": per_suite,
        "accepted": len(accepted),
        "refused": len(refused),
        "refused_detail": [{"suite": s, "id": i, "reason": r} for s, i, r in refused],
    }


def _materialize_reference_stl(folder: Path, step_file: Path, stl_file: Path,
                               key: str) -> str | None:
    """Owner reference intake: a provided model.stl is copied as-is; a model.step is
    converted once via geom_bands.step_to_stl (build123d import_step/export_stl -- the
    same conversion the band scorer itself uses). Cached at lab/state/refs/<key>.stl so
    a rerun of import-references never reconverts an already-materialised part.
    build123d is only imported here, lazily, so a bank/stats/import-teacher run with no
    reference folders never needs it on the path."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REFS_DIR / f"{key[:16]}.stl"
    if dest.exists():
        return str(dest)
    try:
        if stl_file.exists():
            shutil.copyfile(stl_file, dest)
        else:
            sys.path.insert(0, str(HERE / "scripts"))
            import geom_bands  # noqa: PLC0415 -- lazy, build123d-heavy
            geom_bands.step_to_stl(step_file, dest)
        return str(dest)
    except Exception as e:
        print(f"WARNING: could not materialise reference STL for {folder.name}: {e}",
              file=sys.stderr)
        return None


def import_references(refs_dir: Path = REFERENCES_ROOT, dry_run: bool = False) -> dict:
    """Walk `refs_dir`/<name>/ folders, each holding spec.txt (one spec sentence, mm,
    optional leading "tier: N" line, default tier 3 -- these are real parts, not the
    tier-1-2 filler the family generator tends toward) plus model.step or model.stl.
    Zero folders (the common case today: the folder exists with only a README) reports
    0 imported and exits cleanly -- this is not an error state, and this early return is
    BEFORE _locked_bank is ever opened, so the common no-op path never touches the lock.

    Whole read-decide-append window (once folders are found) under _locked_bank's
    single flock (fix round 1, finding 6), same as add_items/import_teacher. This
    includes the (possibly slow, build123d-heavy) STEP->STL materialisation for an
    accepted row -- acceptable here since import-references is a rare, manual, one
    -folder-at-a-time operation, not the automated hot path specgen.py runs in a loop."""
    if not refs_dir.exists():
        return {"imported": 0, "refused": 0, "skipped": 0, "folders_found": 0}

    folders = sorted(p for p in refs_dir.iterdir() if p.is_dir())
    with _locked_bank() as (f, bank):
        bank_keys = {r["key"] for r in bank}
        suite_keys, suite_slugs = contamination_sets()
        accepted: list[dict] = []
        refused: list[dict] = []
        skipped: list[dict] = []
        now = _now()
        for folder in folders:
            spec_file = folder / "spec.txt"
            step_file = folder / "model.step"
            stl_file = folder / "model.stl"
            if not spec_file.exists() or not (step_file.exists() or stl_file.exists()):
                skipped.append({"folder": folder.name,
                                "reason": "missing spec.txt or model.step/model.stl"})
                continue
            text = spec_file.read_text().strip()
            lines = text.splitlines()
            tier = 3
            spec_lines = lines
            if lines and lines[0].strip().lower().startswith("tier:"):
                try:
                    tier = int(lines[0].split(":", 1)[1].strip())
                except Exception:
                    pass
                spec_lines = lines[1:]
            spec = "\n".join(spec_lines).strip()
            if not spec:
                skipped.append({"folder": folder.name, "reason": "empty spec.txt"})
                continue
            key = hc._key(spec)
            reason = refusal_reason(key, spec, suite_keys, suite_slugs, bank_keys)
            if reason:
                refused.append({"folder": folder.name, "reason": reason})
                continue
            ref_stl = None
            if not dry_run:
                ref_stl = _materialize_reference_stl(folder, step_file, stl_file, key)
            row = {
                "id": f"owner-reference:{folder.name}",
                "spec": spec,
                "tier": tier,
                "group": "owner-reference",
                "source": "owner-reference",
                "key": key,
                "added": now,
            }
            if ref_stl:
                row["reference_stl"] = ref_stl
            accepted.append(row)
            bank_keys.add(key)
        if accepted and not dry_run:
            for row in accepted:
                f.write(json.dumps(row) + "\n")
            f.flush()
    return {"imported": len(accepted), "refused": len(refused), "skipped": len(skipped),
            "folders_found": len(folders)}


def apply_reference_updates(updates: dict[str, dict]) -> dict:
    """Merge `updates` (bank `key` -> extra fields, e.g. reference_stl/reference_source/
    reference_facts/reference_added) into the matching bank rows. Used by
    lab/teacher_refs.py's import-teacher-refs to promote admitted teacher-suite geometry,
    and written to be safe for any future caller with the same shape of update.

    Crash-safe and lock-safe against a concurrent appender (add_items/import_teacher/
    import_references, all of which flock this SAME file via _locked_bank): this function
    takes that identical flock directly (open SPECS_FILE "r+", LOCK_EX) and holds it for
    the whole read-decide-rewrite window, exactly like _locked_bank's own contract.

    It deliberately does NOT reuse _locked_bank's own parsed `rows` as the thing it writes
    back, and it deliberately does NOT write via a temp file + os.replace:

    - Byte-identity (every row this function does not touch must reappear unchanged):
      json.dumps(json.loads(line)) is not guaranteed to reproduce the exact original bytes
      (key order, float formatting, unicode escaping all vary), so untouched rows are kept
      as their ORIGINAL raw line, verbatim -- only rows named in `updates` are re-serialised.

    - No rename: a temp-file + os.replace swap would give SPECS_FILE a NEW inode while this
      process still holds the flock on the OLD one. A concurrent appender (add_items) that
      had already called open(SPECS_FILE, "a+") and is blocked in flock() at that exact
      moment holds an fd to the OLD inode; once this function unlocks, that appender's
      flock() call returns and it writes its new row into the now-orphaned, path-unreachable
      inode -- a write that succeeds and is still lost forever, because nothing reachable by
      path points at that inode any more. Writing the fully-assembled new content into the
      SAME already-locked fd (truncate + write, no path change) makes that failure mode
      structurally impossible: whichever process gets the lock next always opens the one,
      current, live inode this path has ever pointed to.

    Owner references always winning, and idempotency, both fall out of one rule: a row that
    already carries `reference_stl` (an owner import, or a previous run of this function) is
    left completely alone and reported under `skipped_already_has_reference`, never
    overwritten.

    Returns {"applied": [key, ...], "skipped_already_has_reference": [key, ...],
    "not_found": [key, ...]} -- `not_found` is any update key with no matching bank row
    (reported, never silently swallowed)."""
    empty = {"applied": [], "skipped_already_has_reference": [], "not_found": []}
    if not updates:
        return empty
    if not SPECS_FILE.exists():
        return {**empty, "not_found": sorted(updates.keys())}

    applied: list[str] = []
    skipped: list[str] = []
    seen_keys: set[str] = set()
    with open(SPECS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw_lines = [ln for ln in f.read().split("\n") if ln.strip()]
            out_lines: list[str] = []
            for raw_line in raw_lines:
                row = json.loads(raw_line)
                key = row.get("key")
                seen_keys.add(key)
                extra = updates.get(key)
                if extra is None:
                    out_lines.append(raw_line)
                    continue
                if row.get("reference_stl"):
                    skipped.append(key)
                    out_lines.append(raw_line)
                    continue
                row.update(extra)
                out_lines.append(json.dumps(row))
                applied.append(key)
            new_content = "".join(line + "\n" for line in out_lines)
            f.seek(0)
            f.truncate()
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    not_found = sorted(k for k in updates if k not in seen_keys)
    return {"applied": applied, "skipped_already_has_reference": skipped,
            "not_found": not_found}


def stats(path: Path | None = None) -> dict:
    bank = load_bank(path)
    total = len(bank)
    by_tier: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_group: dict[str, int] = {}
    with_ref_by_tier: dict[str, int] = {}
    with_ref_by_source: dict[str, int] = {}
    for r in bank:
        by_tier[str(r.get("tier"))] = by_tier.get(str(r.get("tier")), 0) + 1
        by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1
        by_group[r.get("group", "")] = by_group.get(r.get("group", ""), 0) + 1
        if r.get("reference_stl"):
            tier_k = str(r.get("tier"))
            with_ref_by_tier[tier_k] = with_ref_by_tier.get(tier_k, 0) + 1
            src_k = r.get("reference_source") or r.get("source", "")
            with_ref_by_source[src_k] = with_ref_by_source.get(src_k, 0) + 1
    tier34 = sum(n for t, n in by_tier.items() if t in ("3", "4"))
    tier34_share = round(tier34 / total, 4) if total else 0.0
    return {"total": total, "by_tier": by_tier, "by_source": by_source,
            "by_group": by_group, "tier34_share": tier34_share,
            "with_reference": {"by_tier": with_ref_by_tier,
                               "by_reference_source": with_ref_by_source}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    it = sub.add_parser("import-teacher")
    it.add_argument("--dry-run", action="store_true")

    ir = sub.add_parser("import-references")
    ir.add_argument("--dir", type=Path, default=REFERENCES_ROOT)
    ir.add_argument("--dry-run", action="store_true")

    ad = sub.add_parser("add")
    ad.add_argument("file", type=Path)
    ad.add_argument("--source", required=True)
    ad.add_argument("--group", default=None)
    ad.add_argument("--tier", type=int, default=None)
    ad.add_argument("--dry-run", action="store_true")

    itr = sub.add_parser("import-teacher-refs")
    itr.add_argument("--dry-run", action="store_true")
    itr.add_argument("--workers", type=int, default=3)
    itr.add_argument("--limit", type=int, default=None)
    itr.add_argument("--pairs", type=Path, default=None)
    itr.add_argument("--decisions", type=Path, default=None)

    sub.add_parser("stats")

    a = ap.parse_args()
    if a.cmd == "import-teacher":
        r = import_teacher(dry_run=a.dry_run)
        print(json.dumps(r, indent=2))
    elif a.cmd == "import-references":
        r = import_references(a.dir, dry_run=a.dry_run)
        print(json.dumps(r, indent=2))
    elif a.cmd == "import-teacher-refs":
        # Lazy import: a plain bank/stats/import-teacher run never needs build123d or
        # cad_engine on the path, same reasoning as _materialize_reference_stl's own
        # lazy `import geom_bands` above.
        from lab import teacher_refs  # noqa: PLC0415
        pairs = a.pairs or teacher_refs.DEFAULT_PAIRS_FILE
        decisions = a.decisions or teacher_refs.DEFAULT_DECISIONS_FILE
        r = teacher_refs.import_teacher_refs(pairs_path=pairs, decisions_path=decisions,
                                             dry_run=a.dry_run, workers=a.workers,
                                             limit=a.limit)
        print(json.dumps(r, indent=2))
    elif a.cmd == "add":
        raw = json.loads(a.file.read_text())
        items = raw.get("benchmarks", raw.get("items", [])) if isinstance(raw, dict) else raw
        r = add_items(items, source=a.source, group_default=a.group,
                     tier_default=a.tier, dry_run=a.dry_run)
        print(json.dumps({"accepted": r["accepted"], "refused": len(r["refused"])}, indent=2))
        for item in r["refused"][:20]:
            print(f"  refused: {item}", file=sys.stderr)
    elif a.cmd == "stats":
        print(json.dumps(stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

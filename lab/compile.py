#!/usr/bin/env python3
"""lab/compile.py -- Task 4, the Phase 3 data engine compiler.

Turns `lab/state/pairs.jsonl` (the harvest's verified training pairs) into a training
round: `train.jsonl` and `val.jsonl` in the exact {"prompt", "completion", "id", "kind"}
shape `lab/train.py` consumes, plus `spec_id`/`tier`/`confirm_strength`/`pair_id`/`source`
carried alongside for provenance (`lab/train.py`'s `load_rows` only ever reads "prompt" and
"completion", so the extra keys ride along harmlessly).

Read-only against `lab/state/*` (pairs.jsonl, upgrades.jsonl, systems/, val_specs.json):
this module never appends to, rewrites, or deletes anything under `lab/state/`. A GPU
harvest may be appending to pairs.jsonl while this runs; every JSONL read here tolerates a
torn last line the same defensive way `lab/harvest.py`'s own `_read_jsonl` does (skip a
line that fails to parse, never raise).

Pipeline, per split (train/val):
  1. select rows: kind=="good", turn=="first" (a salvage turn never trains), effective
     confirm_strength (after upgrades.jsonl) in --strengths, id not in --exclude-ids,
     gate_version accepted (current, or in --accept-gate-versions; default: all, always
     reported by version).
  2. rehydrate the system prompt from lab/state/systems/<system_sha1>.txt (a missing file
     drops the row -- there is no prompt to train on without it).
  3. dedup by AST-normalised code fingerprint and cap at --max-per-spec, per spec_id,
     preferring the strongest confirm_strength then the earliest ts.
  4. split surviving specs into train/val by a FROZEN, spec-level assignment
     (--val-specs-file, stratified by tier, seeded by --seed): once that file exists it is
     read and reused verbatim, never redrawn, so round 2 stays comparable to round 1. A
     copy is also written into --out-dir for this round's own record.
  5. build one ChatML {"messages": [system, user, assistant]} row per surviving pair
     (user content is the pair's own recorded prompt, VERBATIM -- the same string
     cad_engine actually sent) and render every one of them through
     `lab.data.render_pairs`, the real function `lab/data.py` itself uses: this is what
     fail-closes contamination (a spec lab.data.extract_spec cannot find in the rendered
     user message, or one matching a card suite, is dropped there) and proves the shape
     compile.py builds is exactly the shape lab/data.py's own loader accepts -- there is
     no separate, hand-rolled contamination or rendering path in this file.

--dry-run runs the whole pipeline and prints data_meta, touching no file under --out-dir
and creating no new --val-specs-file (an existing one is still read, never redrawn).

Usage:
  python3 lab/compile.py --round 1 --strengths reference,cross_pass \\
      --out-dir lab/rounds/round1
  python3 lab/compile.py --round 1 --strengths reference,cross_pass,same_pass \\
      --out-dir lab/rounds/round1 --dry-run
"""
from __future__ import annotations

import argparse
import ast
import builtins as _builtins_mod
import hashlib
import json
import os
import random
import shlex
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from lab.data import (  # noqa: E402
    DEFAULT_TEMPLATE, default_contamination_sets, load_template, render_pairs,
)
from cad_v5.config import lab_config  # noqa: E402

STATE_DIR = HERE / "lab" / "state"
DEFAULT_PAIRS_FILE = STATE_DIR / "pairs.jsonl"
DEFAULT_UPGRADES_FILE = STATE_DIR / "upgrades.jsonl"
DEFAULT_SYSTEMS_DIR = STATE_DIR / "systems"
DEFAULT_VAL_SPECS_FILE = STATE_DIR / "val_specs.json"

# Hand-bumped alongside lab/harvest.py's own _GATE_VERSION_BASE (fix round 3, M3 there):
# bump this THE SAME DAY that constant changes, never independently -- a mismatch here
# would make every row report as a "stale" gate version it is not.
_GATE_VERSION_BASE = "gv1"

# Preference order when capping/dedup-ing a spec's pairs: lower rank = stronger evidence
# (mirrors lab/harvest.py's own confirm_strength docstring, "reference" > "cross_pass" >
# "same_pass"), so a cap always keeps the best-evidenced pairs first.
_STRENGTH_RANK = {"reference": 0, "cross_pass": 1, "same_pass": 2}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Small JSONL / JSON helpers, deliberately reimplemented (not imported from lab.harvest,
# which is frozen and pulls in cad_engine/fluid_gen/the whole GPU-facing import graph --
# nothing here needs any of that). Read side mirrors lab.harvest._read_jsonl exactly: a
# line that fails to parse (a concurrent harvest writer caught mid-line) is skipped, never
# raised on.
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
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


def load_upgrades(path: Path) -> dict[str, str]:
    """pair_id -> latest confirm_strength (mirrors lab.harvest._load_upgrades: an
    append-only log, last row for a given pair_id wins because append order is
    recency)."""
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        pid = row.get("pair_id")
        if pid and row.get("confirm_strength"):
            out[pid] = row["confirm_strength"]
    return out


def load_exclude_ids(path: Path | None) -> set[str]:
    """One pair id per line; a `#` starts a comment (a whole-line comment, or trailing
    after an id on the same line); blank lines ignored."""
    ids: set[str] = set()
    if path is None:
        return ids
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--exclude-ids file not found: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


# ---------------------------------------------------------------------------
# Code fingerprint -- a standalone reimplementation of lab.harvest._code_fingerprint's
# identifier/literal-normalised AST dump (per Task 4's own instructions: importing
# lab.harvest itself pulls in cad_engine/fluid_gen for no reason a pure compiler needs,
# so this is a deliberate, exact reimplementation, not a shortcut). Two candidates
# differing only in comments/whitespace, which locally-bound name they used for the
# same role, or `5` vs `5.0` fingerprint identically; a real difference in structure
# does not collapse.
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = frozenset(dir(_builtins_mod))


class _BindingCollector(ast.NodeVisitor):
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
    tree = ast.parse(code)
    collector = _BindingCollector()
    collector.visit(tree)
    mapping = {name: f"_v{i}" for i, name in enumerate(collector.order)}
    tree = _IdentifierAndLiteralNormaliser(mapping).visit(tree)
    return ast.dump(tree)


def code_fingerprint(code: str) -> str:
    try:
        return hashlib.sha1(_normalized_ast_dump(code).encode()).hexdigest()
    except Exception:
        return hashlib.sha1(code.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Gate version -- mirrors lab.harvest.gate_version() exactly (same base string, same
# hash inputs and formula); verified byte-for-byte against a real pairs.jsonl row's
# stored gate_version on this branch (both read gv1-2c11a923 for the live cad.json).
# ---------------------------------------------------------------------------

def current_gate_version() -> str:
    cfg = (lab_config() or {}).get("harvest") or {}
    payload = json.dumps({"agreement": cfg.get("agreement") or {},
                          "strict": cfg.get("strict") or {}}, sort_keys=True)
    return f"{_GATE_VERSION_BASE}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _tally_reasons(dropped: list[tuple[str, str]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for _pid, reason in dropped:
        cat = reason.split(":", 1)[0]
        tally[cat] = tally.get(cat, 0) + 1
    return tally


def select_eligible(pairs: list[dict], upgrades: dict[str, str], strengths: set[str],
                    exclude_ids: set[str], accept_gate_versions: set[str] | None,
                    current_gv: str) -> tuple[list[dict], list[tuple[str, str]], dict[str, int]]:
    """First filtering pass: kind, turn, effective confirm_strength, exclude-ids, gate
    version. Returns (eligible rows with an added "effective_strength" key, dropped
    (id, reason) pairs, gate_versions_seen tally over every row that reached the gate
    version check -- i.e. regardless of whether --accept-gate-versions then filters it
    out, so the report always shows the true distribution)."""
    eligible: list[dict] = []
    dropped: list[tuple[str, str]] = []
    gate_versions_seen: dict[str, int] = {}
    for row in pairs:
        pid = row.get("id", "")
        if row.get("kind") != "good":
            dropped.append((pid, "kind-not-good: fail/candidate rows never train"))
            continue
        if row.get("turn") != "first":
            dropped.append((pid, f"turn-not-first: {row.get('turn')!r} (salvage rows never train)"))
            continue
        effective = upgrades.get(pid, row.get("confirm_strength"))
        if effective not in strengths:
            dropped.append((pid, f"strength-excluded: {effective!r} not in {sorted(strengths)}"))
            continue
        if pid in exclude_ids:
            dropped.append((pid, "excluded-by-audit: id listed in --exclude-ids"))
            continue
        gv = row.get("gate_version")
        gate_versions_seen[gv] = gate_versions_seen.get(gv, 0) + 1
        if accept_gate_versions is not None and gv not in accept_gate_versions and gv != current_gv:
            dropped.append((pid, f"gate-version-excluded: {gv!r}"))
            continue
        row = dict(row)
        row["effective_strength"] = effective
        eligible.append(row)
    return eligible, dropped, gate_versions_seen


def rehydrate_systems(eligible: list[dict], systems_dir: Path,
                      dropped: list[tuple[str, str]]) -> list[dict]:
    """Attaches "system_text" to every row whose lab/state/systems/<sha1>.txt is present;
    a row whose system prompt cannot be found is dropped -- there is nothing to train the
    user/assistant turns against without it."""
    cache: dict[str, str] = {}
    kept: list[dict] = []
    for row in eligible:
        sha1 = row.get("system_sha1")
        if not sha1:
            dropped.append((row.get("id", ""), "missing-system-file: no system_sha1 on the row"))
            continue
        if sha1 not in cache:
            path = systems_dir / f"{sha1}.txt"
            if not path.exists():
                cache[sha1] = ""
            else:
                cache[sha1] = path.read_text(encoding="utf-8")
        if not cache[sha1]:
            dropped.append((row.get("id", ""), f"missing-system-file: {sha1}.txt not found"))
            continue
        row = dict(row)
        row["system_text"] = cache[sha1]
        kept.append(row)
    return kept


def dedup_and_cap(eligible: list[dict], max_per_spec: int,
                  dropped: list[tuple[str, str]]) -> list[dict]:
    """Per spec_id: sort by (strength rank, ts) so the best-evidenced, earliest pairs are
    considered first, drop an exact code-fingerprint repeat, then cap at max_per_spec."""
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_spec[row.get("spec_id", "")].append(row)

    kept: list[dict] = []
    for spec_id, rows in by_spec.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (_STRENGTH_RANK.get(r.get("effective_strength"), 9), r.get("ts") or ""),
        )
        seen_fp: set[str] = set()
        n_kept = 0
        for row in rows_sorted:
            fp = code_fingerprint(row.get("code") or "")
            if fp in seen_fp:
                dropped.append((row.get("id", ""),
                                f"dedup: duplicate code fingerprint within spec {spec_id!r}"))
                continue
            if n_kept >= max_per_spec:
                dropped.append((row.get("id", ""),
                                f"max-per-spec-cap: {spec_id!r} already has {max_per_spec}"))
                continue
            seen_fp.add(fp)
            n_kept += 1
            kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Frozen validation split, by spec, stratified by tier
# ---------------------------------------------------------------------------

def load_or_create_val_specs(path: Path, spec_tiers: dict[str, object], val_frac: float,
                             seed: int, persist: bool) -> tuple[set[str], bool]:
    """Returns (val_spec_ids, created). If `path` already exists it is read and reused
    VERBATIM -- spec_tiers is not consulted at all, so a later round's new specs never
    perturb an earlier round's split. Otherwise a fresh stratified-by-tier sample is
    drawn (deterministic from `seed`) and, when `persist` is true (never during
    --dry-run), written to `path`."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = data.get("val_specs") if isinstance(data, dict) else data
        return set(ids or []), False

    rng = random.Random(seed)
    by_tier: dict[str, list[str]] = defaultdict(list)
    for spec_id, tier in spec_tiers.items():
        by_tier[str(tier)].append(spec_id)

    val_ids: set[str] = set()
    for tier in sorted(by_tier):
        ids_sorted = sorted(by_tier[tier])
        rng.shuffle(ids_sorted)
        n = min(len(ids_sorted), round(len(ids_sorted) * val_frac))
        val_ids.update(ids_sorted[:n])

    # A round small enough that every tier rounds down to 0 should still hold out at
    # least one spec overall (when there is more than one spec to hold out from), taken
    # deterministically from the largest tier group. Never fires for an explicit
    # val_frac of 0 -- that means "no split wanted", not "rounding truncated it away".
    if val_frac > 0 and not val_ids and sum(len(v) for v in by_tier.values()) > 1:
        biggest_tier = max(by_tier, key=lambda t: (len(by_tier[t]), t))
        ids_sorted = sorted(by_tier[biggest_tier])
        rng.shuffle(ids_sorted)
        val_ids.add(ids_sorted[0])

    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"val_specs": sorted(val_ids), "seed": seed, "val_frac": val_frac,
                  "created": _now_utc()}
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return val_ids, True


# ---------------------------------------------------------------------------
# ChatML build + render (via lab.data.render_pairs, the real function -- see the module
# docstring for why this is not a hand-rolled second contamination/rendering path)
# ---------------------------------------------------------------------------

def _build_chatml_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": row["system_text"]},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row.get("code") or ""},
        ],
        "kind": "good",
    }


def render_split(rows: list[dict], *, tmp_dir: Path, tag: str, keys, slugs, template,
                 source_tag: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """Writes `rows` (already ordered) as ChatML to a scratch file under `tmp_dir`, runs
    them through the real lab.data.render_pairs, then merges spec_id/tier/
    confirm_strength/pair_id/source back onto each surviving rendered row by position
    (render_pairs's own row id is "<tag>-<i:04d>", i the 0-based line position in the
    file it read -- exactly the index of `rows` this function wrote it from)."""
    src = tmp_dir / f"{tag}.chatml.jsonl"
    scratch_out = tmp_dir / f"{tag}.rendered.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_build_chatml_row(row)) + "\n")

    rendered_kept, rendered_dropped = render_pairs(src, scratch_out, keys=keys, slugs=slugs,
                                                   template=template, tag=tag)

    merged: list[dict] = []
    for out_row in rendered_kept:
        i = int(out_row["id"].rsplit("-", 1)[-1])
        src_row = rows[i]
        merged.append({
            **out_row,
            "spec_id": src_row.get("spec_id"),
            "tier": src_row.get("tier"),
            "confirm_strength": src_row.get("effective_strength"),
            "pair_id": src_row.get("id"),
            "source": source_tag,
        })

    dropped: list[tuple[str, str]] = []
    for rid, reason in rendered_dropped:
        i = int(rid.rsplit("-", 1)[-1])
        pair_id = rows[i].get("id", rid)
        cat = "contamination" if ("suite match" in reason or "near-duplicate" in reason) \
            else "no-spec-header" if reason == "no-spec-header" \
            else "malformed-row"
        dropped.append((pair_id, f"{cat}: (lab.data.render_pairs) {reason}"))

    for tmp_file in (src, scratch_out):
        try:
            tmp_file.unlink()
        except OSError:
            pass

    return merged, dropped


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _counts_by_tier_strength(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        key = f"{r.get('tier')}|{r.get('confirm_strength')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _rows_over_length(rows: list[dict], max_seq: int = 5120, chars_per_token: float = 3.2) -> int:
    n = 0
    for r in rows:
        est_tokens = (len(r.get("prompt", "")) + len(r.get("completion", ""))) / chars_per_token
        if est_tokens > max_seq:
            n += 1
    return n


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    """Writes `rows` and returns the sha256 of the exact bytes written."""
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_text(rows: list[dict]) -> str:
    text = "".join(json.dumps(r) + "\n" for r in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_round(*, round_n: int, out_dir: Path, strengths: list[str],
                  pairs_file: Path = DEFAULT_PAIRS_FILE,
                  upgrades_file: Path = DEFAULT_UPGRADES_FILE,
                  systems_dir: Path = DEFAULT_SYSTEMS_DIR,
                  val_specs_file: Path = DEFAULT_VAL_SPECS_FILE,
                  exclude_ids_file: Path | None = None,
                  max_per_spec: int = 2, val_frac: float = 0.08, seed: int = 1,
                  accept_gate_versions: list[str] | None = None,
                  template_path: Path | None = None, allow_fallback_template: bool = False,
                  keys=None, slugs=None, dry_run: bool = False,
                  command: str = "") -> dict:
    """Runs the full Task 4 pipeline and returns data_meta. Writes train.jsonl,
    val.jsonl, val_specs.json and data_meta.json under out_dir unless dry_run."""
    strengths_set = set(strengths)
    current_gv = current_gate_version()
    accept_gv_set = set(accept_gate_versions) if accept_gate_versions is not None else None

    pairs = read_jsonl(pairs_file)
    upgrades = load_upgrades(upgrades_file)
    exclude_ids = load_exclude_ids(exclude_ids_file)

    dropped: list[tuple[str, str]] = []
    eligible, sel_dropped, gate_versions_seen = select_eligible(
        pairs, upgrades, strengths_set, exclude_ids, accept_gv_set, current_gv)
    dropped.extend(sel_dropped)

    eligible = rehydrate_systems(eligible, systems_dir, dropped)
    survivors = dedup_and_cap(eligible, max_per_spec, dropped)

    spec_tiers: dict[str, object] = {}
    for row in survivors:
        spec_tiers.setdefault(row.get("spec_id"), row.get("tier"))

    val_ids, val_specs_created = load_or_create_val_specs(
        val_specs_file, spec_tiers, val_frac, seed, persist=not dry_run)

    train_rows = [r for r in survivors if r.get("spec_id") not in val_ids]
    val_rows = [r for r in survivors if r.get("spec_id") in val_ids]

    if template_path is None:
        template_path = DEFAULT_TEMPLATE
    template, from_checkpoint = load_template(template_path, allow_fallback=allow_fallback_template)
    if keys is None or slugs is None:
        d_keys, d_slugs = default_contamination_sets()
        keys = d_keys if keys is None else keys
        slugs = d_slugs if slugs is None else slugs

    source_tag = f"harvest-round{round_n}"
    with tempfile.TemporaryDirectory(prefix=f"lab-compile-round{round_n}-") as tmp:
        tmp_path = Path(tmp)
        train_final, train_dropped = render_split(
            train_rows, tmp_dir=tmp_path, tag=f"round{round_n}train", keys=keys, slugs=slugs,
            template=template, source_tag=source_tag)
        val_final, val_dropped = render_split(
            val_rows, tmp_dir=tmp_path, tag=f"round{round_n}val", keys=keys, slugs=slugs,
            template=template, source_tag=source_tag)
    dropped.extend(train_dropped)
    dropped.extend(val_dropped)

    out_dir = Path(out_dir)
    if dry_run:
        train_sha = _sha256_text(train_final)
        val_sha = _sha256_text(val_final)
    else:
        train_sha = _write_jsonl(out_dir / "train.jsonl", train_final)
        val_sha = _write_jsonl(out_dir / "val.jsonl", val_final)
        (out_dir / "val_specs.json").write_text(
            json.dumps({"val_specs": sorted(val_ids)}, indent=2), encoding="utf-8")

    data_meta = {
        "round": round_n,
        "command": command,
        "timestamp": _now_utc(),
        "dry_run": dry_run,
        "strengths": sorted(strengths_set),
        "max_per_spec": max_per_spec,
        "val_frac": val_frac,
        "seed": seed,
        "current_gate_version": current_gv,
        "accept_gate_versions": sorted(accept_gv_set) if accept_gv_set is not None else None,
        "gate_versions_seen": gate_versions_seen,
        "val_specs_source": "reused" if not val_specs_created else "created",
        "val_specs_file": str(val_specs_file),
        "val_specs_count": len(val_ids),
        "counts": {
            "train": {"total": len(train_final), "by_tier_strength": _counts_by_tier_strength(train_final)},
            "val": {"total": len(val_final), "by_tier_strength": _counts_by_tier_strength(val_final)},
        },
        "rows_total_seen": len(pairs),
        "dropped_total": len(dropped),
        "dropped_by_reason": _tally_reasons(dropped),
        "rows_over_max_seq_5120": {
            "train": _rows_over_length(train_final),
            "val": _rows_over_length(val_final),
        },
        "sha256": {"train.jsonl": train_sha, "val.jsonl": val_sha},
        "template": {"path": str(template_path), "from_checkpoint": from_checkpoint},
        "out_dir": str(out_dir),
    }

    if not dry_run:
        tmp_meta = out_dir / f"data_meta.json.tmp{os.getpid()}"
        tmp_meta.write_text(json.dumps(data_meta, indent=2), encoding="utf-8")
        os.replace(tmp_meta, out_dir / "data_meta.json")

    return data_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--strengths", required=True,
                    help="comma-separated confirm_strength values to admit, e.g. "
                         "reference,cross_pass or reference,cross_pass,same_pass")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exclude-ids", type=Path, default=None,
                    help="file of pair ids to reject, one per line, # comments allowed")
    ap.add_argument("--max-per-spec", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--accept-gate-versions", default=None,
                    help="comma-separated extra gate_version values to accept besides "
                         "today's current one (default: accept every gate_version seen, "
                         "still reported by count)")
    ap.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS_FILE)
    ap.add_argument("--upgrades-file", type=Path, default=DEFAULT_UPGRADES_FILE)
    ap.add_argument("--systems-dir", type=Path, default=DEFAULT_SYSTEMS_DIR)
    ap.add_argument("--val-specs-file", type=Path, default=DEFAULT_VAL_SPECS_FILE)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--allow-fallback-template", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full pipeline and print data_meta; write nothing")
    a = ap.parse_args()

    strengths = [s.strip() for s in a.strengths.split(",") if s.strip()]
    accept_gv = None
    if a.accept_gate_versions:
        accept_gv = [s.strip() for s in a.accept_gate_versions.split(",") if s.strip()]

    command = "python3 lab/compile.py " + shlex.join(sys.argv[1:])

    data_meta = compile_round(
        round_n=a.round, out_dir=a.out_dir, strengths=strengths,
        pairs_file=a.pairs_file, upgrades_file=a.upgrades_file, systems_dir=a.systems_dir,
        val_specs_file=a.val_specs_file, exclude_ids_file=a.exclude_ids,
        max_per_spec=a.max_per_spec, val_frac=a.val_frac, seed=a.seed,
        accept_gate_versions=accept_gv, template_path=a.template,
        allow_fallback_template=a.allow_fallback_template, dry_run=a.dry_run,
        command=command,
    )

    if a.dry_run:
        print("DRY RUN -- nothing written")
    print(json.dumps(data_meta, indent=2))


if __name__ == "__main__":
    main()

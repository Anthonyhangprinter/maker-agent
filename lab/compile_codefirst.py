#!/usr/bin/env python3
"""lab/compile_codefirst.py -- round 2 compiler for teacher "codefirst" pairs.

PREP-ONLY as of 2026-09-25 (owner instruction): round 2a training was cancelled before it
started -- the stock coder already matches ~90% of the codefirst-scale pairs per
gemma_baseline.jsonl, so training on that pool now would be a predictable no-ship. The
owner wants harder pairs, verified by them, first. This script exists so a later, actually
-approved compile is one command; running it with no --approved-ids is informational only
(reports the pool's shape) and is NOT a recommendation to train on the unfiltered result.

Turns teacher-verified (spec, blind-rebuild code) pairs --
benchmarks/results/card/codefirst-*/*/pairs.jsonl, every row already band=="match" (the
teacher's own rebuild-vs-design check) -- into the exact {"prompt", "completion", "id",
"kind"} shape lab/train.py consumes: the SAME rendering path lab/compile.py's round 1 used
(lab.data.render_pairs -- the checkpoint's own chat template, the card-suite contamination
guard), so there is no separate, hand-rolled framing.

Unlike lab/compile.py (which reads the harvest's own lab/state/pairs.jsonl, one CAPTURED
system prompt per row via system_sha1), this source was never sent through cad_engine at
all -- it is teacher-authored code checked by geometry, not a harvested model call. Every
row here is rendered with the CURRENT production system prompt
(cad_engine._CODE_SYSTEM, imported live from the module, never a cached copy) and the
CURRENT production user-message shape for a brief-less oneshot build
(cad_engine.generate_code_raw's own "USER REQUEST (verbatim...)" framing -- reproduced in
_build_user_message, with _assert_user_template_matches_production() failing loudly if
cad_engine.py's own literal ever drifts from the copy here). This is what "same chat
template and system prompt as production, so there's no train/serve mismatch" means in
practice: round 1's rows and round 2's rows are framed identically at the string level.

Reusable knobs (added for the round 2 prep, not round 1):
  --oversample-gemma-fail N   duplicate (N total copies, N>=1) any TRAIN row whose
                               gemma_baseline band is present and NOT in ("match", "valid")
                               (stock Gemma did not build a correct part). Never applied to
                               val rows -- an eval split must stay one row per spec. A row
                               with NO gemma_baseline entry at all (not yet baselined) is
                               left at 1x: "unknown" is not "failed".
  --exclude-gemma-match-tier1 drop tier-1 rows stock Gemma already matched outright, before
                               the split -- the easy, already-solved slice.
  --approved-ids FILE          JSON file, either {"approved_ids": [...]} or a bare list of
                               pair ids; when given, ONLY those ids are eligible at all --
                               everything else is dropped as "not-owner-approved". THIS IS
                               THE GATE: a real training compile should always pass this.
                               Omitted, every band=="match" pair in --pairs is eligible
                               (today's dry run: no approved-ids file exists yet).

Held out, always, regardless of the knobs above:
  - kind: only rows with band=="match" in the source file are eligible at all (a defensive
    check -- every row in the two files this was written for already satisfies it).
  - any pair whose spec collides with a card suite (lab.data.render_pairs's own
    default_contamination_sets -- the identical guard lab/compile.py's round 1 used).
  - any pair whose spec collides, exactly or by unique near-duplicate slug, with one of the
    owner references under ~/CAD/references/<name>/spec.txt (18 on this box as of
    2026-09-25; see load_owner_reference_specs).

Usage:
  # dry run, no --approved-ids: reports the pool's shape, writes nothing
  python3 lab/compile_codefirst.py \\
      --pairs benchmarks/results/card/codefirst-scale-2026-09-25/claude-opus-5-5/pairs.jsonl \\
      --pairs benchmarks/results/card/codefirst-pilot-2026-09-24/claude-opus-5-5/pairs.jsonl \\
      --gemma-baseline benchmarks/results/card/codefirst-scale-2026-09-25/gemma_baseline.jsonl \\
      --oversample-gemma-fail 2 \\
      --out-dir lab/rounds/round2 --dry-run

  # a real, owner-approved compile (later, once approved_ids.json exists)
  python3 lab/compile_codefirst.py --pairs ... --gemma-baseline ... \\
      --oversample-gemma-fail 2 --approved-ids benchmarks/results/card/round2/approved_ids.json \\
      --out-dir lab/rounds/round2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from lab.compile import (  # noqa: E402
    read_jsonl, load_or_create_val_specs, _write_jsonl, _sha256_text, _tally_reasons,
)
import harvest_census as hc  # noqa: E402

# cad_engine is imported for its _CODE_SYSTEM constant only (the live production system
# prompt) -- it never generates code here. _ollama is patched to raise immediately, the
# same safety net lab/teacher_refs.py uses for the same reason: a bug that somehow reached
# a model call in a CPU-only compile script should crash loudly, not burn GPU time or send
# a request to a server this script has no business talking to.
import cad_engine as engine  # noqa: E402


def _raise_on_model_call(*a, **kw):
    raise RuntimeError(
        "lab/compile_codefirst.py called cad_engine._ollama -- this module never generates "
        "code, only reads engine._CODE_SYSTEM. This is a bug in this script, not a missing "
        "feature.")


engine._ollama = _raise_on_model_call

DEFAULT_REFS_DIR = Path.home() / "CAD" / "references"
DEFAULT_OUT_DIR = HERE / "lab" / "rounds" / "round2"
DEFAULT_VAL_SPECS_FILE = DEFAULT_OUT_DIR / "val_specs.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Production framing: the live system prompt (imported, never copied) and the exact
# user-message shape cad_engine.generate_code_raw builds for a brief-less oneshot build
# (fluid_gen.py's "fluid mode" -- see cad_engine.py's own docstring on generate_code_raw:
# "Codegen straight from the USER'S WORDS -- no brief, no 8B paraphrase in between").
# generate_code_raw itself also calls the model, so its prompt-building three lines are
# reproduced here rather than called; _assert_user_template_matches_production() is the
# drift guard that makes a silent divergence between this copy and cad_engine.py's own
# literal a hard failure instead of a quiet train/serve mismatch.
# ---------------------------------------------------------------------------

_USER_TEMPLATE_CANARY = "USER REQUEST (verbatim — every number here is AUTHORITATIVE):"


def _assert_user_template_matches_production() -> None:
    src = Path(engine.__file__).read_text(encoding="utf-8")
    if _USER_TEMPLATE_CANARY not in src:
        raise SystemExit(
            "lab/compile_codefirst.py: the literal "
            f"{_USER_TEMPLATE_CANARY!r} was not found in {engine.__file__}. "
            "cad_engine.generate_code_raw's user-message framing has changed since this "
            "script's _build_user_message was written -- update it to match before "
            "compiling, or every round 2 row trains on a shape production no longer sends.")


def _build_user_message(spec: str) -> str:
    """Mirrors cad_engine.generate_code_raw's prompt byte-for-byte, INCLUDING the Notes
    block (round 2 change, 2026-09-26): the codefirst pairs themselves carry no notes (they
    are teacher-authored, never sampled through the engine), but production always computes
    retrieval_notes_for(spec) before codegen -- few-shots, learned pitfalls, and (since
    commit 76f8cda) the introspected API reference. Omitting that block here would train on
    a prompt shape serving never sends. Notes are recomputed HERE, at compile time, against
    the CURRENT retrieval corpora (cad-examples.jsonl/cad-lessons.jsonl/b123d/api_ref.json)
    -- exactly what a live request for this spec would see today, which is the whole point
    of matching train prompts to serve prompts. retrieval_notes_for only calls the
    embed-server (nomic-embed, CPU) and static JSON lookups, never the code model, so this
    is safe under this script's _raise_on_model_call patch. See
    _assert_user_template_matches_production for the drift guard on the rest of the
    template."""
    notes = engine.retrieval_notes_for(spec, use_fewshots=True)
    notes_str = "\n".join(f"- {n}" for n in notes)
    return (f"{_USER_TEMPLATE_CANARY}\n{spec}\n\n"
           + (f"Notes:\n{notes_str}\n\n" if notes_str else "")
           + "Write the build123d code:")


# ---------------------------------------------------------------------------
# Owner references: the held-out 18-part set under ~/CAD/references/<name>/spec.txt. Read
# -only here -- this script never calls lab/specbank.py's import_references (that mutates
# the shared lab/state/specs.jsonl bank, which this script has no business touching for a
# hold-out check).
# ---------------------------------------------------------------------------

def load_owner_reference_specs(refs_dir: Path = DEFAULT_REFS_DIR) -> list[dict]:
    """One row per <name>/spec.txt found (skipping a bare README.md or a folder missing
    spec.txt): {"name", "spec", "tier"}. The optional leading "tier: N" line in spec.txt is
    stripped the same way lab/specbank.py's import_references reads it."""
    out: list[dict] = []
    if not refs_dir.exists():
        return out
    for folder in sorted(p for p in refs_dir.iterdir() if p.is_dir()):
        spec_file = folder / "spec.txt"
        if not spec_file.exists():
            continue
        lines = spec_file.read_text(encoding="utf-8").splitlines()
        tier = 3
        spec_lines = lines
        if lines and lines[0].strip().lower().startswith("tier:"):
            try:
                tier = int(lines[0].split(":", 1)[1].strip())
            except Exception:
                pass
            spec_lines = lines[1:]
        spec = "\n".join(spec_lines).strip()
        if spec:
            out.append({"name": folder.name, "spec": spec, "tier": tier})
    return out


def owner_reference_contamination_sets(refs_dir: Path = DEFAULT_REFS_DIR) -> tuple[set[str], set[str]]:
    """(keys, slugs) for the owner references, same identity functions
    (harvest_census._key / _slug) the card-suite contamination guard uses, so a collision
    check against this set is directly comparable. With 18 members a slug collision within
    the set itself is vanishingly unlikely, but the uniqueness rule is applied anyway for
    consistency with default_contamination_sets()'s own reasoning."""
    refs = load_owner_reference_specs(refs_dir)
    keys = {hc._key(r["spec"]) for r in refs}
    slug_counts: dict[str, int] = {}
    for r in refs:
        s = hc._slug(r["spec"], 40)
        slug_counts[s] = slug_counts.get(s, 0) + 1
    slugs = {s for s, n in slug_counts.items() if n == 1}
    return keys, slugs


# ---------------------------------------------------------------------------
# Gemma baseline join
# ---------------------------------------------------------------------------

def load_gemma_bands(paths: list[Path]) -> dict[str, str]:
    """id -> band, from one or more gemma_baseline.jsonl files (whatever rows exist when
    this runs -- the scale baseline was still appending live at compile time on
    2026-09-25). A later file's row for the same id overwrites an earlier one; within one
    file, read_jsonl's own tolerant parse (skip a torn last line, never raise) already
    handles a concurrent writer."""
    bands: dict[str, str] = {}
    for p in paths:
        for row in read_jsonl(p):
            rid = row.get("id")
            band = row.get("band")
            if rid and band:
                bands[rid] = band
    return bands


# ---------------------------------------------------------------------------
# Approved-ids gate
# ---------------------------------------------------------------------------

def load_approved_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get("approved_ids") if isinstance(data, dict) else data
    if not isinstance(ids, list):
        raise SystemExit(f"--approved-ids {path}: expected a JSON list or "
                         f"{{'approved_ids': [...]}}, got {type(ids)}")
    return {str(i) for i in ids}


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_pairs(paths: list[Path]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Concatenates --pairs files, keeping only band=="match" rows (defensive: every row
    in the two files this was written for already satisfies it) and deduping by id
    (first occurrence wins; a later duplicate is dropped and reported, never silently
    overwritten -- two pairs files should never share an id, so a collision is worth
    seeing)."""
    seen: dict[str, dict] = {}
    dropped: list[tuple[str, str]] = []
    for path in paths:
        for row in read_jsonl(path):
            rid = row.get("id", "")
            if row.get("band") != "match":
                dropped.append((rid, f"not-verified-match: band={row.get('band')!r}"))
                continue
            if rid in seen:
                dropped.append((rid, f"duplicate-id: already loaded from another --pairs file"))
                continue
            row = dict(row)
            row["_source_file"] = str(path)
            seen[rid] = row
    return list(seen.values()), dropped


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_pool(pairs: list[dict], *, gemma_bands: dict[str, str], approved_ids: set[str] | None,
                exclude_gemma_match_tier1: bool, owner_ref_keys: set[str],
                owner_ref_slugs: set[str]) -> tuple[list[dict], list[tuple[str, str]]]:
    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for row in pairs:
        rid = row.get("id", "")
        spec = row.get("spec", "")
        if approved_ids is not None and rid not in approved_ids:
            dropped.append((rid, "not-owner-approved"))
            continue
        key = hc._key(spec)
        slug = hc._slug(spec, 40)
        if key in owner_ref_keys:
            dropped.append((rid, "owner-reference-collision: exact spec match"))
            continue
        if slug in owner_ref_slugs:
            dropped.append((rid, "owner-reference-collision: near-duplicate slug"))
            continue
        gband = gemma_bands.get(rid)
        if exclude_gemma_match_tier1 and row.get("tier") == 1 and gband == "match":
            dropped.append((rid, "gemma-matched-tier1-excluded"))
            continue
        row = dict(row)
        row["_gemma_band"] = gband
        kept.append(row)
    return kept, dropped


# ---------------------------------------------------------------------------
# Render (ChatML -> lab.data.render_pairs, the real production rendering path)
# ---------------------------------------------------------------------------

def _build_chatml_row(row: dict, system_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": _build_user_message(row["spec"])},
            {"role": "assistant", "content": row.get("code") or ""},
        ],
        "kind": "good",
    }


def render_split(rows: list[dict], *, tmp_dir: Path, tag: str, keys, slugs, template,
                 system_text: str, source_tag: str) -> tuple[list[dict], list[tuple[str, str]]]:
    src = tmp_dir / f"{tag}.chatml.jsonl"
    scratch_out = tmp_dir / f"{tag}.rendered.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_build_chatml_row(row, system_text)) + "\n")

    rendered_kept, rendered_dropped = render_pairs(src, scratch_out, keys=keys, slugs=slugs,
                                                   template=template, tag=tag)

    merged: list[dict] = []
    for out_row in rendered_kept:
        i = int(out_row["id"].rsplit("-", 1)[-1])
        src_row = rows[i]
        merged.append({
            **out_row,
            "spec_id": src_row.get("id"),
            "tier": src_row.get("tier"),
            "gemma_band": src_row.get("_gemma_band"),
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


def oversample(rows: list[dict], factor: int) -> list[dict]:
    """factor total copies (factor=1: no-op) of every row whose gemma_band is present and
    NOT in ("match", "valid") -- round 2's weighting rule (2026-09-26 task brief): a fixed-
    engine Gemma sample that is band=="valid" already built a correct part (just not byte-
    identical to the teacher's own code), so it is not "hard" and stays at 1x same as a
    "match"; only near_miss/fail/crash rows -- genuinely hard for the stock coder -- get
    oversampled. A row with gemma_band None (never baselined) is also left at 1x: "unknown"
    is not "failed". Duplicate rows get a "-dupN" suffix on their rendered id so train.jsonl
    ids stay unique; every other field (prompt/completion/pair_id/...) is identical to the
    original, which is exactly the point -- the trainer sees the same (spec, code) pair
    `factor` times."""
    if factor <= 1:
        return rows
    out: list[dict] = []
    for row in rows:
        gband = row.get("_gemma_band")
        n = factor if (gband and gband not in ("match", "valid")) else 1
        for i in range(n):
            copy = dict(row)
            if i > 0:
                copy["id"] = f"{row['id']}-dup{i}"
            out.append(copy)
    return out


def _counts_by_tier_band(rows: list[dict], band_key: str = "_gemma_band") -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        key = f"tier{r.get('tier')}|gemma_{r.get(band_key) or 'none'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_codefirst(*, pairs_files: list[Path], gemma_baseline_files: list[Path],
                      out_dir: Path, oversample_gemma_fail: int = 1,
                      exclude_gemma_match_tier1: bool = False,
                      approved_ids_file: Path | None = None,
                      refs_dir: Path = DEFAULT_REFS_DIR,
                      val_specs_file: Path = DEFAULT_VAL_SPECS_FILE,
                      val_frac: float = 0.05, seed: int = 1,
                      template_path: Path | None = None, allow_fallback_template: bool = False,
                      dry_run: bool = False, command: str = "") -> dict:
    _assert_user_template_matches_production()

    dropped: list[tuple[str, str]] = []
    pairs, load_dropped = load_pairs(pairs_files)
    dropped.extend(load_dropped)

    gemma_bands = load_gemma_bands(gemma_baseline_files)
    approved_ids = load_approved_ids(approved_ids_file)
    owner_ref_keys, owner_ref_slugs = owner_reference_contamination_sets(refs_dir)
    owner_refs = load_owner_reference_specs(refs_dir)

    survivors, filter_dropped = filter_pool(
        pairs, gemma_bands=gemma_bands, approved_ids=approved_ids,
        exclude_gemma_match_tier1=exclude_gemma_match_tier1,
        owner_ref_keys=owner_ref_keys, owner_ref_slugs=owner_ref_slugs)
    dropped.extend(filter_dropped)

    spec_tiers = {row["id"]: row.get("tier") for row in survivors}
    val_ids, val_specs_created = load_or_create_val_specs(
        val_specs_file, spec_tiers, val_frac, seed, persist=not dry_run)

    train_rows = [r for r in survivors if r["id"] not in val_ids]
    val_rows = [r for r in survivors if r["id"] in val_ids]
    train_rows = oversample(train_rows, oversample_gemma_fail)
    # val stays exactly 1x, always -- oversample() never called on it.

    if template_path is None:
        template_path = DEFAULT_TEMPLATE
    template, from_checkpoint = load_template(template_path, allow_fallback=allow_fallback_template)
    keys, slugs = default_contamination_sets()
    system_text = engine._CODE_SYSTEM

    source_tag = "codefirst-round2"
    with tempfile.TemporaryDirectory(prefix="lab-compile-codefirst-") as tmp:
        tmp_path = Path(tmp)
        train_final, train_dropped = render_split(
            train_rows, tmp_dir=tmp_path, tag="round2train", keys=keys, slugs=slugs,
            template=template, system_text=system_text, source_tag=source_tag)
        val_final, val_dropped = render_split(
            val_rows, tmp_dir=tmp_path, tag="round2val", keys=keys, slugs=slugs,
            template=template, system_text=system_text, source_tag=source_tag)
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
        "round": "2 (prep)",
        "command": command,
        "timestamp": _now_utc(),
        "dry_run": dry_run,
        "pairs_files": [str(p) for p in pairs_files],
        "gemma_baseline_files": [str(p) for p in gemma_baseline_files],
        "oversample_gemma_fail": oversample_gemma_fail,
        "exclude_gemma_match_tier1": exclude_gemma_match_tier1,
        "approved_ids_file": str(approved_ids_file) if approved_ids_file else None,
        "approved_ids_count": len(approved_ids) if approved_ids is not None else None,
        "owner_references_dir": str(refs_dir),
        "owner_references_count": len(owner_refs),
        "val_frac": val_frac,
        "seed": seed,
        "val_specs_source": "reused" if not val_specs_created else "created",
        "val_specs_file": str(val_specs_file),
        "val_specs_count": len(val_ids),
        "pairs_total_loaded": len(pairs),
        "pool_after_filter": len(survivors),
        "counts": {
            "train": {"total": len(train_final), "by_tier_gemma_band": _counts_by_tier_band(train_rows)},
            "val": {"total": len(val_final), "by_tier_gemma_band": _counts_by_tier_band(val_rows)},
        },
        "gemma_baseline_coverage": {
            "pairs_with_gemma_row": sum(1 for r in survivors if r.get("_gemma_band")),
            "pairs_without_gemma_row": sum(1 for r in survivors if not r.get("_gemma_band")),
            "gemma_band_breakdown": _tally_reasons(
                [(r["id"], r["_gemma_band"]) for r in survivors if r.get("_gemma_band")]),
        },
        "dropped_total": len(dropped),
        "dropped_by_reason": _tally_reasons(dropped),
        "sha256": {"train.jsonl": train_sha, "val.jsonl": val_sha},
        "template": {"path": str(template_path), "from_checkpoint": from_checkpoint},
        "out_dir": str(out_dir),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_meta = out_dir / f"data_meta.json.tmp{os.getpid()}"
        tmp_meta.write_text(json.dumps(data_meta, indent=2), encoding="utf-8")
        os.replace(tmp_meta, out_dir / "data_meta.json")

    return data_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", action="append", required=True, type=Path,
                    help="a codefirst pairs.jsonl file; repeatable")
    ap.add_argument("--gemma-baseline", action="append", default=[], type=Path,
                    help="a gemma_baseline.jsonl file (id -> band); repeatable, optional")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--oversample-gemma-fail", type=int, default=1,
                    help="total copies (>=1) of a TRAIN row whose gemma band != match; "
                         "default 1 = no oversampling")
    ap.add_argument("--exclude-gemma-match-tier1", action="store_true")
    ap.add_argument("--approved-ids", type=Path, default=None,
                    help="JSON file gating which pair ids are eligible at all; omit for an "
                         "informational, unfiltered dry run")
    ap.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR)
    ap.add_argument("--val-specs-file", type=Path, default=DEFAULT_VAL_SPECS_FILE)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--allow-fallback-template", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import shlex
    command = "python3 lab/compile_codefirst.py " + shlex.join(sys.argv[1:])

    data_meta = compile_codefirst(
        pairs_files=a.pairs, gemma_baseline_files=a.gemma_baseline, out_dir=a.out_dir,
        oversample_gemma_fail=a.oversample_gemma_fail,
        exclude_gemma_match_tier1=a.exclude_gemma_match_tier1,
        approved_ids_file=a.approved_ids, refs_dir=a.refs_dir,
        val_specs_file=a.val_specs_file, val_frac=a.val_frac, seed=a.seed,
        template_path=a.template, allow_fallback_template=a.allow_fallback_template,
        dry_run=a.dry_run, command=command,
    )

    if a.dry_run:
        print("DRY RUN -- nothing written")
    print(json.dumps(data_meta, indent=2))


if __name__ == "__main__":
    main()

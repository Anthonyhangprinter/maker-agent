#!/usr/bin/env python3
"""Compile ~/.openclaw/cad-sftpairs.jsonl into a training-ready dataset (M6' QLoRA).

The jsonl is append-only and never rewritten: every exclusion here is compile-time
filtering, so a bad verdict can always be reversed by re-running the compiler.

Stages (each report-visible):
  1. exclude   — drop teacher rows whose teacher_spec_id carries a latest 'reject'
                 human verdict (~/.openclaw/cad-review-decisions.jsonl).
  2. revalidate— re-run every surviving row's code through the deterministic gate
                 (step -> inspect -> parse_facts -> verify_expected with
                 spec-corroborated expectations only). Rows generated before the
                 2026-07-30 gate hardening may no longer pass; they must not train.
                 Cached by sha1(spec+code) in ~/.openclaw/cad-sft-reverify-cache.json.
  3. dedup     — sha1(spec+code); prefer the row that persisted its production prompt.

Usage:
    python3 scripts/compile_sft.py --report-only    # hygiene report, writes no dataset
    python3 scripts/compile_sft.py                  # full compile (Phase 5 adds ChatML out)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import cad_engine as engine  # noqa: E402
from cad_v5.config import SFTPAIRS_FILE  # noqa: E402

DECISIONS_FILE = Path.home() / ".openclaw" / "cad-review-decisions.jsonl"
CACHE_FILE = Path.home() / ".openclaw" / "cad-sft-reverify-cache.json"
RESULTS_DIR = HERE / "benchmarks" / "results"


def _load_teacher_gen():
    spec = importlib.util.spec_from_file_location("tg", HERE / "scripts" / "teacher_gen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows() -> list[dict]:
    rows = []
    for i, line in enumerate(SFTPAIRS_FILE.read_text().splitlines(), 1):
        try:
            r = json.loads(line)
            r["_line"] = i
            rows.append(r)
        except Exception as e:
            print(f"  [load] line {i} unparseable: {e}", file=sys.stderr)
    return rows


def latest_verdicts() -> dict[str, dict]:
    """Latest human verdict per spec id — later lines win, so a re-review supersedes."""
    v: dict[str, dict] = {}
    if DECISIONS_FILE.exists():
        for line in DECISIONS_FILE.read_text().splitlines():
            try:
                d = json.loads(line)
                v[d["id"]] = d
            except Exception:
                continue
    return v


def exclude_rejected(rows: list[dict], verdicts: dict) -> tuple[list[dict], list[dict]]:
    keep, dropped = [], []
    for r in rows:
        sid = r.get("teacher_spec_id")
        if sid and verdicts.get(sid, {}).get("verdict") == "reject":
            r["_why"] = f"human reject: {verdicts[sid].get('note', '')[:120]}"
            dropped.append(r)
        else:
            keep.append(r)
    return keep, dropped


def _key(r: dict) -> str:
    """Verification cache key: geometry depends only on (spec, code)."""
    return hashlib.sha1((r.get("spec", "") + "\n" + r.get("code", "")).encode()).hexdigest()


def _dedup_key(r: dict) -> str:
    """Dedup key: a good row and its sibling fail->fix row share (spec, code) but are
    DIFFERENT training examples — kind and bad_code must participate or the edit-turn
    pairs (the highest-value data) silently collapse into their good twins."""
    return hashlib.sha1((r.get("spec", "") + "\n" + r.get("code", "") + "\n"
                         + (r.get("kind") or "") + "\n"
                         + (r.get("bad_code") or "")).encode()).hexdigest()


def revalidate(rows: list[dict], tg, tmp_root: Path) -> tuple[list[dict], list[dict]]:
    """Offline re-verify each row's code (the FIX for kind=fail rows) under the current,
    hardened gate. No model calls, no spend — pure geometry."""
    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            cache = {}
    keep, dropped = [], []
    for i, r in enumerate(rows, 1):
        k = _key(r)
        if k in cache:
            v = cache[k]
        else:
            graded = engine.corroborate_expected({}, r.get("spec", ""))
            full = tg.verify(r.get("code", ""), graded, r.get("spec", ""), tmp_root / k[:12])
            v = {"accepted": tg._accepted(full), "hard": full["hard"],
                 "spec_notes": full["spec_notes"], "error": full["error"]}
            cache[k] = v
            CACHE_FILE.write_text(json.dumps(cache))   # per-row: a crash keeps progress
            print(f"  [reverify {i}/{len(rows)}] {r.get('teacher_spec_id') or r.get('source')}"
                  f" -> {'ok' if v['accepted'] else 'FAIL'}", flush=True)
        if v["accepted"]:
            keep.append(r)
        else:
            r["_why"] = ("hardened gate: "
                         + "; ".join(v["hard"] or v["spec_notes"] or [v["error"]])[:200])
            dropped.append(r)
    return keep, dropped


def dedup(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Prefer the duplicate that persisted its production prompt (system+prompt keys)."""
    best: dict[str, dict] = {}
    for r in rows:
        k = _dedup_key(r)
        cur = best.get(k)
        if cur is None or (not cur.get("system") and r.get("system")):
            best[k] = r
    keep = list(best.values())
    kept_ids = {id(r) for r in keep}
    dropped = [r for r in rows if id(r) not in kept_ids]
    for r in dropped:
        r["_why"] = "duplicate (spec+code)"
    return keep, dropped


def _capture_prompt(call) -> dict:
    """Run an engine codegen function with the model call stubbed out and lift the exact
    (kind, system, prompt) it stashed — never copy the f-strings (teacher_gen rule #1)."""
    orig = engine._ollama
    engine._ollama = lambda *a, **k: ""
    try:
        call()
    finally:
        engine._ollama = orig
    return dict(engine._LAST_PROMPT)


def reconstruct_prompt(r: dict) -> dict:
    """For rows that predate prompt persistence (gift/retro/human-accepted). Retrieval notes
    are re-derived from TODAY'S corpus — the prompt shape is byte-identical to production,
    the notes content is current rather than historical. Flagged in the row."""
    spec = r.get("spec", "")
    if r.get("kind") == "fail":
        p = _capture_prompt(lambda: engine.revise_script(
            spec, r.get("bad_code", ""), r.get("problem", "geometry did not match"),
            state=""))
    else:
        p = _capture_prompt(lambda: engine.generate_code_raw(
            spec, engine.retrieval_notes_for(spec)))
    p["reconstructed"] = True
    return p


def to_chatml(r: dict) -> dict:
    """One training example in the ChatML `messages` shape the trainer consumes.
    Assistant target is the stored code VERBATIM, no fences — production strips fences from
    output and _REVISE_SYSTEM demands fence-less replies, so the target must model that.
    For kind=fail rows the example IS the revise turn (the fix as completion)."""
    if r.get("system") and r.get("prompt"):
        system, prompt = r["system"], r["prompt"]
        kind, recon = r.get("prompt_kind", ""), False
    else:
        p = reconstruct_prompt(r)
        system, prompt = p["system"], p["prompt"]
        kind, recon = p.get("kind", ""), True
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": r["code"]},
        ],
        "prompt_kind": kind, "kind": r.get("kind"), "source": r.get("source"),
        "band": r.get("band"), "teacher_spec_id": r.get("teacher_spec_id"),
        "spec_slug": _slug_of(r), "reconstructed": recon,
    }


_HC = None

def _slug_of(r: dict) -> str:
    global _HC
    if _HC is None:
        spec = importlib.util.spec_from_file_location(
            "hc", HERE / "scripts" / "harvest_census.py")
        _HC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HC)
    return _HC._slug(r.get("spec", ""), 40)


def split_rows(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """~95/5 grouped by spec slug — the same spec never appears in both sides."""
    train, val = [], []
    for ex in examples:
        h = int(hashlib.sha1(ex["spec_slug"].encode()).hexdigest(), 16)
        (val if h % 20 == 0 else train).append(ex)
    return train, val


def _est_tokens(ex: dict) -> int:
    # cheap char-based estimate (~3.2 chars/token for this mix of prose + code);
    # the real tokenizer check happens on the training box.
    return sum(len(m["content"]) for m in ex["messages"]) // 3


def brief_row(r: dict) -> dict:
    return {"line": r["_line"], "kind": r.get("kind"), "source": r.get("source"),
            "spec_id": r.get("teacher_spec_id"), "model": r.get("code_model"),
            "spec": (r.get("spec") or "")[:90], "why": r.get("_why", "")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile the M6' SFT dataset.")
    ap.add_argument("--report-only", action="store_true",
                    help="hygiene stages + report only; write no dataset files")
    args = ap.parse_args()

    tg = _load_teacher_gen()
    rows = load_rows()
    verdicts = latest_verdicts()
    print(f"[compile] {len(rows)} rows loaded, {len(verdicts)} human verdicts on file")

    t0 = time.monotonic()
    kept1, drop_rej = exclude_rejected(rows, verdicts)
    print(f"[exclude] -{len(drop_rej)} human-rejected -> {len(kept1)}")

    tmp_root = Path(tempfile.mkdtemp(prefix="sft_reverify_"))
    kept2, drop_gate = revalidate(kept1, tg, tmp_root)
    print(f"[reverify] -{len(drop_gate)} fail hardened gate -> {len(kept2)}")

    kept3, drop_dup = dedup(kept2)
    print(f"[dedup] -{len(drop_dup)} duplicates -> {len(kept3)}")

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "loaded": len(rows), "surviving": len(kept3),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "excluded_human_reject": [brief_row(r) for r in drop_rej],
        "excluded_hardened_gate": [brief_row(r) for r in drop_gate],
        "excluded_duplicate": [brief_row(r) for r in drop_dup],
        "surviving_by_source": {},
        "surviving_by_kind": {},
        "no_source_rows": [brief_row(r) for r in kept3 if not r.get("source")],
    }
    for r in kept3:
        report["surviving_by_source"][r.get("source") or "none"] = \
            report["surviving_by_source"].get(r.get("source") or "none", 0) + 1
        report["surviving_by_kind"][r.get("kind") or "?"] = \
            report["surviving_by_kind"].get(r.get("kind") or "?", 0) + 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"sft_compile_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n[report] {out}")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("loaded", "surviving", "surviving_by_source",
                               "surviving_by_kind")}, indent=1))
    if args.report_only:
        return 0

    # ── Phase 5: format -> split -> stats -> write ───────────────────────────
    suite = set()
    _slug_of({})            # force-load harvest_census
    suite = _HC.suite_slugs()
    examples = [to_chatml(r) for r in kept3]
    leaked = [e for e in examples if e["spec_slug"] in suite]
    if leaked:
        # Corpus gold seeds overlap the eval suites by construction (the L-bracket/enclosure
        # golds ARE suite parts), so gift rows sampled from them collide here. Drop them
        # LOUDLY — training must never see a held-out spec — but don't abort the compile.
        print(f"[compile] CONTAMINATION: dropping {len(leaked)} example(s) that collide "
              f"with eval suites:", file=sys.stderr)
        for e in leaked[:10]:
            print(f"  {e['spec_slug']} ({e['source']})", file=sys.stderr)
        examples = [e for e in examples if e["spec_slug"] not in suite]
        report["excluded_suite_collision"] = [
            {"slug": e["spec_slug"], "source": e["source"]} for e in leaked]

    train, val = split_rows(examples)
    lens = sorted(_est_tokens(e) for e in examples)
    too_long = [e["spec_slug"] for e in examples if _est_tokens(e) > 8192]
    stats = {
        "examples": len(examples), "train": len(train), "val": len(val),
        "by_kind": dict(collections.Counter(e["kind"] for e in examples)),
        "by_prompt_kind": dict(collections.Counter(e["prompt_kind"] for e in examples)),
        "by_source": dict(collections.Counter(e["source"] for e in examples)),
        "reconstructed_prompts": sum(1 for e in examples if e["reconstructed"]),
        "est_tokens_p50": lens[len(lens) // 2] if lens else 0,
        "est_tokens_max": lens[-1] if lens else 0,
        "over_8192_est": too_long,
    }
    report["format_stats"] = stats
    out.write_text(json.dumps(report, indent=1))

    train_f = Path.home() / ".openclaw" / "cad-sft-train.jsonl"
    val_f = Path.home() / ".openclaw" / "cad-sft-val.jsonl"
    with train_f.open("w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with val_f.open("w") as f:
        for e in val:
            f.write(json.dumps(e) + "\n")
    print(f"\n[compile] wrote {len(train)} -> {train_f}")
    print(f"[compile] wrote {len(val)} -> {val_f}")
    print(json.dumps(stats, indent=1))
    if too_long:
        print(f"[compile] WARNING: {len(too_long)} example(s) estimated over 8192 tokens "
              f"— verify with the real tokenizer on the training box.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

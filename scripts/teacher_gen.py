#!/usr/bin/env python3
"""M6' teacher distillation — generate build123d SFT pairs from a cloud teacher model.

Andrew Vassili's Q2 recommendation (2026-07-26): have a much larger model produce the
build123d code, run every candidate through the verification stack we already have, keep
only what passes, and for each FAILED candidate spend one repair call with the error text —
turning the failure stream into fail->fix edit-turn pairs. Those pairs are the data the
deployed model needs most, because it spends most of its life revising rather than writing
from scratch.

Design rules this script exists to enforce:

  * PRODUCTION-IDENTICAL PROMPTS. It does not hand-assemble a prompt. It points the engine's
    active coder at the cloud rung and calls engine.generate_code() / engine.revise_script()
    — the same functions the live loop calls — so the (system, user) strings are byte-identical
    to production by construction, and are captured from engine._LAST_PROMPT rather than
    re-derived. scripts/gift_sample.py hand-rolled its brief and skipped the retrieval
    injection entirely; every pair it wrote came from a prompt missing the TECHNIQUE
    REFERENCES and PITFALLS blocks. Do not repeat that.

  * NO LLM IN THE VERIFIER. Accept/reject is scripts/step -> scripts/inspect -> parse_facts
    -> verify_expected, the same deterministic gate the loop uses for accepted_via="gate":
    valid geometry, no hard fails, and no [spec]-tagged advisories (a [spec] note means a
    measurement contradicts a number the user literally wrote).

  * HELD-OUT CONTRACT. Aborts non-zero if any input spec collides with text-to-cad, organic,
    or heldout-cqe. A silent skip is how contamination gets in.

Usage:
    python3 scripts/teacher_gen.py --dry-run            # guard + brief only, no spend
    python3 scripts/teacher_gen.py --limit 5            # small paid smoke run
    python3 scripts/teacher_gen.py                      # full pilot
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import cad_engine as engine  # noqa: E402
from cad_v5.config import SFTPAIRS_DIR, SFTPAIRS_FILE  # noqa: E402

DEFAULT_SPECS = HERE / "benchmarks" / "teacher-pilot" / "specs.json"
RESULTS_DIR = HERE / "benchmarks" / "results"

# One codegen + one repair per spec. The budget is a hard stop on spend, not a target.
CALLS_PER_SPEC = 2


# Fallback USD/spec when there is no measured history yet. Derived from the 2026-07-29/30 runs:
# simple parts ~$0.075/call at ~1.3 calls/spec; tier-4 mechanisms measured $0.191/spec.
_FALLBACK_RATE = {1: 0.06, 2: 0.10, 3: 0.13, 4: 0.19}


def measured_rates() -> dict:
    """USD/spec per tier, learned from past run summaries. Beats a guess, and a guess is what
    made me quote $25-30 for a run that would really have cost ~$100 (2026-07-30)."""
    tot: dict = {}
    for f in sorted(RESULTS_DIR.glob("teacher_pilot_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        usd, done = d.get("usd_spent"), d.get("specs_done")
        if not usd or not done:
            continue                      # pre-ledger runs have no cost recorded
        for r in d.get("per_spec", []):
            t = r.get("tier")
            if t:
                a = tot.setdefault(t, [0.0, 0])
                a[0] += usd / done
                a[1] += 1
    return {t: v[0] / v[1] for t, v in tot.items() if v[1]}


def project_cost(specs: list) -> tuple[float, dict]:
    """(projected USD, rate table used). Blends measured rates with fallbacks per tier."""
    rates = {**_FALLBACK_RATE, **measured_rates()}
    total = sum(rates.get(s.get("tier") or 2, _FALLBACK_RATE[2]) for s in specs)
    return total, rates


def _load_census():
    """harvest_census.py is a script, not a package module — load it by path for suite_slugs."""
    spec = importlib.util.spec_from_file_location("hc", HERE / "scripts" / "harvest_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def guard_contamination(specs: list[dict]) -> None:
    """Abort if any pilot spec collides with an eval suite. Never skip silently."""
    hc = _load_census()
    suite = hc.suite_slugs()
    clashes = [s for s in specs if hc._slug(s["spec"], 40) in suite]
    if clashes:
        print("CONTAMINATION GUARD TRIPPED — these specs collide with an eval suite:",
              file=sys.stderr)
        for s in clashes:
            print(f"  {s.get('id', '?')}: {s['spec'][:80]}", file=sys.stderr)
        print("\nTraining specs must never derive from the held-out suites "
              "(benchmarks/heldout-cqe/acceptance.json._meta.heldout). Aborting.",
              file=sys.stderr)
        sys.exit(2)
    print(f"[guard] {len(specs)} spec(s) checked against {len(suite)} suite slug(s) — no collisions.")


REVIEW_FILE = SFTPAIRS_FILE.with_name("cad-sftpairs-review.jsonl")


def verify(code: str, expected: dict, spec: str, work: Path) -> dict:
    """Deterministic accept/reject. No model involved. Mirrors the loop's gate-accept rule."""
    out = {"ran": False, "valid": False, "hard": [], "spec_notes": [], "notes": [],
           "facts": {}, "state": "", "error": "", "step": None}
    work.mkdir(parents=True, exist_ok=True)   # run_step writes build_source.py in here
    try:
        step, _log = engine.run_step(code, work)
        out["ran"], out["step"] = True, step
    except Exception as e:
        out["error"] = str(e)[:400]
        return out
    try:
        insp = engine.run_inspect(step)
    except Exception as e:
        # A pathological solid can blow the inspect timeout (M17 spiral-bevel case,
        # 2026-07-31) — that is a failed candidate, not a crashed run.
        out["error"] = f"inspect failed: {str(e)[:350]}"
        return out
    out["valid"], out["state"] = insp["valid"], insp["output"]
    if not insp["valid"]:
        out["error"] = "; ".join(insp["errors"])[:400]
        return out
    facts = engine.parse_facts(insp["output"])
    hard, notes = engine.verify_expected(facts, expected, spec=spec)
    out["facts"] = facts
    out["hard"] = hard
    out["notes"] = notes
    out["spec_notes"] = [n for n in notes if n.startswith("[spec]")]
    return out


def _accepted(v: dict) -> bool:
    return bool(v["ran"] and v["valid"] and not v["hard"] and not v["spec_notes"])


def _problem_text(v: dict) -> str:
    """What to hand the repair call — deterministic measurements first, exactly as the
    loop's fail->fix harvest prioritises them."""
    if not v["ran"] or not v["valid"]:
        return v["error"] or "the script produced no valid geometry"
    return "; ".join(v["hard"] or v["spec_notes"] or v["notes"]) or "geometry did not match the request"


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")


def _keep_render(step_path: Path, work: Path, tag: str) -> str:
    """Render the accepted geometry so the row supports image->code SFT as well as text->code."""
    try:
        png = engine.run_render(step_path, work)
        if png and Path(png).exists():
            SFTPAIRS_DIR.mkdir(parents=True, exist_ok=True)
            dst = SFTPAIRS_DIR / f"teacher-{tag}.png"
            shutil.copy(png, dst)
            return str(dst)
    except Exception:
        pass
    return ""


def run_spec(item: dict, model: str, dry_run: bool, tmp_root: Path,
             use_brief: bool = False) -> dict:
    """Generate, verify, and (on failure) repair one spec. Returns a per-spec record."""
    spec = item["spec"]
    sid = item.get("id", "?")
    rec = {"id": sid, "tier": item.get("tier"), "group": item.get("group"),
           "spec": spec, "outcome": "", "turns": 0, "rows": []}
    t0 = time.monotonic()

    if use_brief:
        brief = engine.build_brief(spec)
        engine.reconcile_expected(brief, spec)
        fs_rows, lesson_rows = engine.inject_retrieval_notes(brief, spec, use_fewshots=True)
        rec["fewshots"], rec["lessons"] = len(fs_rows), len(lesson_rows)
    elif dry_run:
        brief = {"notes": [], "expected": {}}        # no GPU work for a free dry run
    else:
        # No 8B in the chain. Notes (retrieved idioms + pitfalls) still ride along; the spec
        # itself is what the teacher reads. forbid_blind_holes is derived from the spec text,
        # not guessed, so it is safe to keep.
        brief = {"notes": engine.retrieval_notes_for(spec), "expected": {}}
        engine.reconcile_expected(brief, spec)
        rec["fewshots"] = len(brief["notes"])
        rec["lessons"] = 0
        rec["no_brief"] = True
    # Grade against spec-corroborated expectations, NOT the brief's raw guesses. The brief is
    # written by qwen3:8b and would otherwise be examining a far stronger model. The brief
    # itself is left intact so the prompt stays byte-identical to production.
    graded = engine.corroborate_expected(brief.get("expected", {}), spec)
    rec["graded_expected"] = graded
    rec["brief_expected"] = brief.get("expected", {})

    helper = (brief.get("helper") or "").strip().rstrip(".")
    if helper and (engine._HELPER_RE.match(helper) or engine._WH_HELPER_RE.match(helper)):
        # Deterministic helper short-circuit: generate_code returns hand-written code with no
        # model call, so there is no prompt->code pair worth training on.
        rec["outcome"] = "skipped-helper"
        return rec

    if dry_run:
        rec["outcome"] = "dry-run"
        return rec

    work = tmp_root / sid
    work.mkdir(parents=True, exist_ok=True)
    engine.reset_cloud_budget(CALLS_PER_SPEC)
    prev_model = engine._ACTIVE_CODE_MODEL
    engine._ACTIVE_CODE_MODEL = engine.CLOUD_PREFIX + model
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        # --- turn 1: production codegen path, cloud rung ---
        code = (engine.generate_code(brief, spec) if use_brief
                else engine.generate_code_raw(spec, brief.get("notes")))
        p0 = dict(engine._LAST_PROMPT)
        rec["turns"] = 1
        v0 = verify(code, graded, spec, work / "t1")

        if _accepted(v0):
            rec["outcome"] = "accepted-first-try"
            rec["rows"].append({
                "kind": "good", "source": "teacher", "spec": spec, "code": code,
                "image": _keep_render(v0["step"], work / "t1", f"{_slugify(model)}-{sid}-t1"),
                "prompt_kind": p0.get("kind", ""), "system": p0.get("system", ""),
                "prompt": p0.get("prompt", ""),
                "code_model": model, "teacher_spec_id": sid,
                "verified": {"solids": v0["facts"].get("solids"),
                             "bbox_mm": v0["facts"].get("bbox"),
                             "volume_mm3": v0["facts"].get("volume"),
                             "advisories": v0["notes"]},
                "timestamp": stamp})
            return rec

        # --- turn 2: ONE repair call with the real error text (the fail->fix pair) ---
        problem = _problem_text(v0)
        rec["first_problem"] = problem[:300]
        code2 = engine.revise_script(spec, code, problem, state=v0["state"])
        p1 = dict(engine._LAST_PROMPT)
        rec["turns"] = 2
        v1 = verify(code2, graded, spec, work / "t2")

        if not _accepted(v1):
            # Distinguish "the teacher produced bad geometry" from "the deterministic gate
            # disagreed with the teacher about which number a dimension refers to". A part
            # that builds, is watertight, and clears every structural check but trips only a
            # [spec] advisory is usually a sub-feature-dimension read: "standoffs 8mm across"
            # measured against the 80x80x40 overall bbox. Those are worth a human's eyes, not
            # a silent discard — but they are NOT auto-written as training pairs either.
            only_spec = (v1["valid"] and not v1["hard"] and v1["spec_notes"])
            # If the repair drew the IDENTICAL complaint, the gate asked for something the
            # model cannot deliver — C01 was told twice to merge 8 balls into 3 bodies, at
            # $0.18. That is a gate defect, not a part defect: label it so, and queue it.
            same = (_problem_text(v0).strip() == _problem_text(v1).strip())
            rec["outcome"] = ("gate-suspect" if same else
                              "spec-vetoed" if only_spec else "failed-after-repair")
            rec["identical_complaint"] = same
            rec["second_problem"] = _problem_text(v1)[:300]
            # Keep BOTH attempts' code and measurements for EVERY non-accepted outcome.
            # Previously only spec-vetoed kept anything, so a failed-after-repair discarded the
            # one artefact needed to diagnose it — four failures sharing a single cause
            # (2026-07-30 "missing bore" on P05/P20, both models) could not be examined at all.
            rec["review"] = {
                "code": code2, "facts": v1["facts"],
                "spec_notes": v1["spec_notes"], "hard": v1["hard"], "notes": v1["notes"],
                "t1_code": code, "t1_facts": v0["facts"],
                "t1_hard": v0["hard"], "t1_spec_notes": v0["spec_notes"],
                "t1_error": v0["error"], "t2_error": v1["error"],
                # The prompt is half of any useful review: without it you cannot tell whether
                # the model misread the spec or the spec misled the model.
                "t1_prompt_kind": p0.get("kind", ""), "t1_prompt": p0.get("prompt", ""),
                "t2_prompt_kind": p1.get("kind", ""), "t2_prompt": p1.get("prompt", ""),
            }
            if only_spec or same:
                # SILVER tier: builds, watertight, no structural fault — only a spec-text check
                # disagreed. Keep it OUT of training but RECOVERABLE for human review. The first
                # pilot binned 11 parts later shown to be correct; a bin is not a decision.
                try:
                    with REVIEW_FILE.open("a") as f:
                        f.write(json.dumps({
                            "tier": "silver", "spec": spec, "code": code2,
                            "spec_notes": v1["spec_notes"], "facts": v1["facts"],
                            "code_model": model, "teacher_spec_id": sid,
                            "timestamp": stamp}) + "\n")
                    rec["review_queued"] = True
                except Exception as e:
                    print(f"    (could not queue for review: {e})", file=sys.stderr)
            return rec

        rec["outcome"] = "accepted-after-repair"
        img = _keep_render(v1["step"], work / "t2", f"{_slugify(model)}-{sid}-t2")
        verified = {"solids": v1["facts"].get("solids"), "bbox_mm": v1["facts"].get("bbox"),
                    "volume_mm3": v1["facts"].get("volume"), "advisories": v1["notes"]}
        # The corrected program, as a plain text->code example...
        rec["rows"].append({
            "kind": "good", "source": "teacher", "spec": spec, "code": code2, "image": img,
            "prompt_kind": p1.get("kind", ""), "system": p1.get("system", ""),
            "prompt": p1.get("prompt", ""), "code_model": model, "teacher_spec_id": sid,
            "verified": verified, "timestamp": stamp})
        # ...and the edit-turn pair: what was wrong, what it was measured to be, and the fix.
        rec["rows"].append({
            "kind": "fail", "source": "teacher-repair", "spec": spec,
            "code": code2, "bad_code": code, "image": img, "problem": problem[:300],
            "prompt_kind": p1.get("kind", ""), "system": p1.get("system", ""),
            "prompt": p1.get("prompt", ""),
            "bad_prompt_kind": p0.get("kind", ""), "bad_system": p0.get("system", ""),
            "bad_prompt": p0.get("prompt", ""),
            "code_model": model, "teacher_spec_id": sid,
            "verified": verified, "timestamp": stamp})
        return rec
    finally:
        engine._ACTIVE_CODE_MODEL = prev_model
        rec["elapsed_s"] = round(time.monotonic() - t0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate teacher-distilled build123d SFT pairs.")
    ap.add_argument("--specs", default=str(DEFAULT_SPECS), help="spec list JSON")
    ap.add_argument("--limit", type=int, default=0, help="max specs (0 = all)")
    ap.add_argument("--only", default="", help="comma-separated spec ids")
    ap.add_argument("--model", default="", help="override the cad.json cloud model")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the guard and the brief, make no paid calls, write nothing")
    ap.add_argument("--brief", action="store_true",
                    help="use the qwen3:8b brief to shape the prompt (legacy). Default is "
                         "brief-less: the teacher reads the user's words directly.")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="hard USD cap for THIS run; stops cleanly before exceeding it "
                         "(0 = uncapped). Priced at list rate, so the real bill is lower.")
    args = ap.parse_args()

    data = json.loads(Path(args.specs).read_text())
    specs = data["benchmarks"] if isinstance(data, dict) else data

    # Guard the WHOLE file before any filtering: --limit/--only must never let a
    # contaminated spec sit in the list unexamined for a later run to pick up.
    guard_contamination(specs)      # exits 2 on collision, before any spend

    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.get("id") in want]
    if args.limit:
        specs = specs[:args.limit]

    cc = engine.cloud_config()
    if not cc and not args.dry_run:
        print("No cloud rung configured — set cad.cloud in ~/.openclaw/cad.json.", file=sys.stderr)
        return 1
    model = args.model or cc.get("model", "")
    print(f"[teacher] model={model or '(dry-run)'}  specs={len(specs)}  "
          f"dry_run={args.dry_run}")

    tmp_root = Path(engine.tempfile.mkdtemp(prefix="teacher_gen_"))
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    projected, rates = project_cost(specs)
    shown = ", ".join(f"t{t}=${rates[t]:.3f}" for t in sorted(rates))
    print(f"[estimate] {len(specs)} spec(s) -> ~${projected:.2f} projected  ({shown})")
    if args.budget and projected > args.budget and not args.dry_run:
        print(f"[estimate] REFUSING TO START: projection ${projected:.2f} exceeds the "
              f"--budget ${args.budget:.2f} cap.\n           Raise --budget, cut --limit, "
              f"or accept a partial run by lowering the spec count.", file=sys.stderr)
        return 2

    prior_usd, prior_calls = engine.cloud_spend_total()
    print(f"[spend] ledger to date: ${prior_usd:.2f} over {prior_calls} call(s)"
          + (f"   |   cap for this run: ${args.budget:.2f}" if args.budget else "   |   uncapped"))

    records, rows, stopped = [], [], None
    for i, item in enumerate(specs, 1):
        # Stop BEFORE the spec that would breach the cap. Worst case per spec is one codegen
        # plus one repair, so reserve that much headroom rather than discovering the overrun
        # after paying for it. Rows already written stay written.
        if args.budget:
            spent = engine.cloud_spend_this_run()
            per_spec = (spent / max(1, i - 1)) if i > 1 else 0.0
            if spent + max(per_spec, 0.02) > args.budget:
                stopped = (f"budget cap ${args.budget:.2f} reached after {i - 1} spec(s) "
                           f"(${spent:.2f} spent, ~${per_spec:.3f}/spec) — stopping cleanly")
                print(f"\n[spend] {stopped}")
                break
        # Per-spec build lock: an interactive build queues one spec, not the whole run.
        # A dry run builds nothing, so it must NOT queue behind a live build — otherwise the
        # free estimate blocks for the length of a benchmark (found 2026-07-30).
        fh = None if args.dry_run else engine._acquire_build_lock(
            f"teacher_gen: {item['spec'][:60]}")
        try:
            rec = run_spec(item, model, args.dry_run, tmp_root, use_brief=args.brief)
        except Exception as e:                       # one bad spec must not kill the run
            rec = {"id": item.get("id"), "spec": item["spec"],
                   "outcome": "error", "error": str(e)[:300]}
        finally:
            if fh is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
        records.append(rec)
        # Append per spec, not in one batch at the end: a timeout or crash 40 specs deep
        # would otherwise discard every paid pair in the run.
        new_rows = rec.get("rows", [])
        if new_rows and not args.dry_run:
            with SFTPAIRS_FILE.open("a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
        rows.extend(new_rows)
        print(f"  [{i}/{len(specs)}] {rec.get('id'):>4}  {rec.get('outcome'):<22} "
              f"turns={rec.get('turns', 0)}  rows={len(new_rows)}  "
              f"{rec.get('elapsed_s', 0)}s", flush=True)

    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("outcome", "?")] = counts.get(r.get("outcome", "?"), 0) + 1

    # (rows were already appended per spec, above)
    run_usd = engine.cloud_spend_this_run()
    summary = {"ran_at": started, "model": model, "dry_run": args.dry_run,
               "specs": len(specs), "specs_done": len(records),
               "usd_spent": round(run_usd, 4), "budget": args.budget or None,
               "stopped_early": stopped,
               "elapsed_s": round(time.monotonic() - t0, 1),
               "outcomes": counts, "pairs_written": len(rows),
               "good": sum(1 for r in rows if r["kind"] == "good"),
               "fail_fix": sum(1 for r in rows if r["kind"] == "fail"),
               "per_spec": records}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"teacher_pilot_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=1))

    total_usd, total_calls = engine.cloud_spend_total()
    print(f"\noutcomes: {counts}")
    print(f"spend this run: ${run_usd:.2f}   |   ledger to date: ${total_usd:.2f} "
          f"over {total_calls} call(s)   [list-price estimate; Console is authoritative]")
    print(f"pairs written: {len(rows)} ({summary['good']} good, {summary['fail_fix']} fail->fix)"
          f"{' [DRY RUN — nothing written]' if args.dry_run else ''}")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

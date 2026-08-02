#!/usr/bin/env python3
"""Batch-mode teacher distillation — same pipeline as teacher_gen.py, half the price.

Uses the Anthropic Message Batches API (50% of list on all tokens). The serial loop in
teacher_gen.py becomes two batched rounds:

    round 1: batch ALL codegen calls -> poll -> verify each result locally (free)
    round 2: batch ONE repair call per failure -> poll -> verify again

Prompts stay production-identical the same way teacher_gen does it: we call the engine's own
generate_code_raw()/revise_script() with the model call stubbed out, and lift the exact
(system, prompt) strings from engine._LAST_PROMPT. Post-processing of replies uses the same
engine functions (_strip_fences, _patch_code) so a batched row is byte-for-byte what the
sync path would have written.

Spend is metered into the ledger at HALF list rate (the batch discount), with a batch flag.

Usage:
    python3 scripts/teacher_batch.py --specs benchmarks/teacher-batch2/specs.json --budget 10
    python3 scripts/teacher_batch.py --only V001,V002 --model claude-opus-5 --budget 5
"""
from __future__ import annotations

import argparse
import fcntl
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

import anthropic  # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402

RESULTS_DIR = HERE / "benchmarks" / "results"
BATCH_DISCOUNT = 0.5


def _load_tg():
    spec = importlib.util.spec_from_file_location("tg", HERE / "scripts" / "teacher_gen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture_prompt(call) -> dict:
    """Run an engine codegen function with the model call stubbed to "", and return the
    exact (kind, system, prompt) it stashed. Zero spend, prompts identical by construction."""
    orig = engine._ollama
    engine._ollama = lambda *a, **k: ""
    try:
        call()
    finally:
        engine._ollama = orig
    return dict(engine._LAST_PROMPT)


def record_spend(model: str, usage_d: dict, run_total: list) -> float:
    """Ledger a batched call at the 50% batch rate. Mirrors engine._record_cloud_spend."""
    cost = engine.cloud_call_cost(model, usage_d) * BATCH_DISCOUNT
    run_total[0] += cost
    with engine.CLOUD_SPEND_FILE.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(), "model": model,
            "in": usage_d.get("input_tokens"), "out": usage_d.get("output_tokens"),
            "usd": round(cost, 6), "batch": True}) + "\n")
    return cost


def safe_verify(tg, code: str, graded: dict, spec: str, work: Path) -> dict:
    """One pathological part must never kill a paid run — verify() guards its own
    subprocesses now, but anything else (parse_facts, disk) still gets caught here."""
    try:
        return tg.verify(code, graded, spec, work)
    except Exception as e:
        return {"ran": False, "valid": False, "hard": [], "spec_notes": [], "notes": [],
                "facts": {}, "state": "", "error": f"verify crashed: {str(e)[:300]}",
                "step": None}


def postprocess(raw: str, spec: str) -> str:
    return engine._patch_code(engine._strip_fences(raw),
                              wants=engine._wanted_edge_features(spec))


def submit_and_wait(client, requests: list, model: str, run_total: list,
                    poll_s: int = 20, resume_id: str = "", ledger: bool = True) -> dict:
    """Create a batch (or resume an already-paid one), poll to completion, return
    {custom_id: text}. Errored/expired entries are absent (caller treats as failures).
    In resume mode the spend was already ledgered by the crashed run — don't re-ledger."""
    if resume_id:
        class _B:  # duck-typed handle
            id = resume_id
        batch = _B()
        print(f"[batch] resuming {resume_id} (spend already ledgered)", flush=True)
    else:
        batch = client.messages.batches.create(requests=requests)
        print(f"[batch] {batch.id}  ({len(requests)} requests) submitted", flush=True)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f"[batch] {b.processing_status}: ok={c.succeeded} err={c.errored} "
              f"processing={c.processing}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(poll_s)
    out = {}
    for r in client.messages.batches.results(batch.id):
        if r.result.type != "succeeded":
            print(f"  [{r.custom_id}] batch result: {r.result.type}", flush=True)
            continue
        msg = r.result.message
        usage_d = {"input_tokens": msg.usage.input_tokens,
                   "output_tokens": msg.usage.output_tokens,
                   "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                   "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0}
        if ledger:
            record_spend(model, usage_d, run_total)
        if msg.stop_reason == "refusal":
            print(f"  [{r.custom_id}] refused by safety classifiers", flush=True)
            continue
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if not text:
            # Adaptive thinking can consume the ENTIRE max_tokens budget on hard specs
            # (M05/M10/M23, 2026-07-31): stop_reason=max_tokens with zero text. Treat as a
            # failed call, not as empty code to "verify".
            print(f"  [{r.custom_id}] EMPTY reply (stop_reason={msg.stop_reason}, "
                  f"out={msg.usage.output_tokens}tok) — raise --max-tokens", flush=True)
            continue
        out[r.custom_id] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched teacher distillation (50% price).")
    ap.add_argument("--specs", default=str(HERE / "benchmarks" / "teacher-pilot" / "specs.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="hard USD cap; refuses to start if the batch-rate projection exceeds "
                         "it, and skips the repair round if round 1 spent too much")
    ap.add_argument("--poll", type=int, default=20, help="poll interval seconds")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="override cad.json cloud.max_tokens (thinking + response share it; "
                         "hard tier-4 specs can need 32000)")
    ap.add_argument("--accept-soft", action="store_true",
                    help="v2 (2026-08-02): also keep parts that build, are watertight and "
                         "pass ALL structural checks but trip a [spec] dimension advisory — "
                         "rows tagged source=teacher-soft. Broken geometry still excluded.")
    ap.add_argument("--resume-batch", default="",
                    help="msgbatch id of an already-paid round-1 batch: skip submission, "
                         "fetch its results, and continue (spend not re-ledgered)")
    ap.add_argument("--ledger-resume", action="store_true",
                    help="with --resume-batch: DO ledger round-1 spend (use when the "
                         "original run died before fetching results)")
    args = ap.parse_args()

    tg = _load_tg()
    data = json.loads(Path(args.specs).read_text())
    specs = data["benchmarks"] if isinstance(data, dict) else data
    tg.guard_contamination(specs)          # exit 2 before any spend
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.get("id") in want]
    if args.limit:
        specs = specs[:args.limit]

    cc = engine.cloud_config()
    model = args.model or cc.get("model", "")
    max_tokens = args.max_tokens or int(cc.get("max_tokens", 16000) or 16000)
    projected, rates = tg.project_cost(specs)
    projected *= BATCH_DISCOUNT
    print(f"[teacher-batch] model={model} specs={len(specs)} "
          f"projected≈${projected:.2f} at batch rates")
    if args.budget and projected > args.budget:
        print(f"REFUSING TO START: batch-rate projection ${projected:.2f} exceeds "
              f"--budget ${args.budget:.2f}", file=sys.stderr)
        return 2

    # Key may live in openclaw.json's env block rather than the process env — resolve it
    # the same way the engine's own HTTP path does.
    client = anthropic.Anthropic(api_key=engine._cloud_key(cc))
    tmp_root = Path(tempfile.mkdtemp(prefix="teacher_batch_"))
    run_total = [0.0]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    stamp = started

    # ── round 1: build production-identical prompts, batch them ──────────────
    ctx: dict[str, dict] = {}       # custom_id -> per-spec working state
    requests = []
    for item in specs:
        sid, spec = item.get("id", "?"), item["spec"]
        brief = {"notes": engine.retrieval_notes_for(spec), "expected": {}}
        engine.reconcile_expected(brief, spec)
        graded = engine.corroborate_expected(brief.get("expected", {}), spec)
        helper = (brief.get("helper") or "").strip().rstrip(".")
        rec = {"id": sid, "tier": item.get("tier"), "group": item.get("group"),
               "spec": spec, "outcome": "", "turns": 0, "rows": [],
               "fewshots": len(brief["notes"]), "no_brief": True,
               "graded_expected": graded, "brief_expected": brief.get("expected", {})}
        if helper and (engine._HELPER_RE.match(helper) or engine._WH_HELPER_RE.match(helper)):
            rec["outcome"] = "skipped-helper"
            ctx[sid] = {"rec": rec, "spec": spec, "graded": graded, "skip": True}
            continue
        p0 = capture_prompt(lambda s=spec, n=brief["notes"]: engine.generate_code_raw(s, n))
        ctx[sid] = {"rec": rec, "spec": spec, "graded": graded, "skip": False, "p0": p0}
        requests.append(Request(
            custom_id=sid,
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=max_tokens,
                system=p0["system"],
                messages=[{"role": "user", "content": p0["prompt"]}])))

    replies = (submit_and_wait(client, requests, model, run_total, args.poll,
                               resume_id=args.resume_batch,
                               ledger=(not args.resume_batch) or args.ledger_resume)
               if (requests or args.resume_batch) else {})

    # ── verify round 1 locally, collect repairs ──────────────────────────────
    repair_reqs = []
    for sid, c in ctx.items():
        if c["skip"]:
            continue
        rec, spec, graded = c["rec"], c["spec"], c["graded"]
        raw = replies.get(sid)
        if raw is None:
            rec["outcome"] = "error"
            rec["error"] = "no batch result (errored/expired/refused)"
            continue
        code = postprocess(raw, spec)
        rec["turns"] = 1
        fh = engine._acquire_build_lock(f"teacher_batch verify: {spec[:60]}")
        try:
            v0 = safe_verify(tg, code, graded, spec, tmp_root / sid / "t1")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
        c["code"], c["v0"] = code, v0
        if tg._accepted(v0):
            rec["outcome"] = "accepted-first-try"
            rec["rows"].append({
                "kind": "good", "source": "teacher", "spec": spec, "code": code,
                "image": tg._keep_render(v0["step"], tmp_root / sid / "t1",
                                         f"{tg._slugify(model)}-{sid}-t1"),
                "prompt_kind": c["p0"].get("kind", ""), "system": c["p0"].get("system", ""),
                "prompt": c["p0"].get("prompt", ""),
                "code_model": model, "teacher_spec_id": sid,
                "verified": {"solids": v0["facts"].get("solids"),
                             "bbox_mm": v0["facts"].get("bbox"),
                             "volume_mm3": v0["facts"].get("volume"),
                             "advisories": v0["notes"]},
                "timestamp": stamp})
            continue
        problem = tg._problem_text(v0)
        rec["first_problem"] = problem[:300]
        p1 = capture_prompt(lambda s=spec, cd=code, pr=problem, st=v0["state"]:
                            engine.revise_script(s, cd, pr, state=st))
        c["p1"], c["problem"] = p1, problem
        repair_reqs.append(Request(
            custom_id=sid,
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=max_tokens,
                system=p1["system"],
                messages=[{"role": "user", "content": p1["prompt"]}])))

    # ── round 2: batched repairs (budget-gated) ──────────────────────────────
    skipped_repairs = False
    if repair_reqs and args.budget:
        # a repair call reads a bigger prompt than round 1; assume ~1.3x round-1 avg/spec
        done = max(1, len(replies))
        est = run_total[0] / done * 1.3 * len(repair_reqs)
        if run_total[0] + est > args.budget:
            print(f"[spend] skipping repair round: ${run_total[0]:.2f} spent + ~${est:.2f} "
                  f"projected repairs exceeds cap ${args.budget:.2f}", flush=True)
            skipped_repairs = True
    repairs = ({} if (not repair_reqs or skipped_repairs)
               else submit_and_wait(client, repair_reqs, model, run_total, args.poll))

    for sid, c in ctx.items():
        rec, spec, graded = c["rec"], c["spec"], c["graded"]
        if c.get("skip") or rec["outcome"] or "p1" not in c:
            continue
        raw2 = repairs.get(sid)
        if raw2 is None:
            rec["outcome"] = "failed-after-repair" if not skipped_repairs else "repair-skipped-budget"
            if not skipped_repairs:
                rec["error"] = "no repair batch result"
            continue
        code2 = postprocess(raw2, spec)
        rec["turns"] = 2
        fh = engine._acquire_build_lock(f"teacher_batch verify2: {spec[:60]}")
        try:
            v1 = safe_verify(tg, code2, graded, spec, tmp_root / sid / "t2")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
        v0, code, p0, p1, problem = c["v0"], c["code"], c["p0"], c["p1"], c["problem"]
        if not tg._accepted(v1):
            only_spec = (v1["valid"] and not v1["hard"] and v1["spec_notes"])
            if only_spec and args.accept_soft:
                rec["outcome"] = "accepted-soft"
                img = tg._keep_render(v1["step"], tmp_root / sid / "t2",
                                      f"{tg._slugify(model)}-{sid}-t2")
                rec["rows"].append({
                    "kind": "good", "source": "teacher-soft", "spec": spec, "code": code2,
                    "image": img, "prompt_kind": p1.get("kind", ""),
                    "system": p1.get("system", ""), "prompt": p1.get("prompt", ""),
                    "code_model": model, "teacher_spec_id": sid,
                    "verified": {"solids": v1["facts"].get("solids"),
                                 "bbox_mm": v1["facts"].get("bbox"),
                                 "volume_mm3": v1["facts"].get("volume"),
                                 "advisories": v1["notes"],
                                 "soft_spec_notes": v1["spec_notes"]},
                    "timestamp": stamp})
                continue
            same = (tg._problem_text(v0).strip() == tg._problem_text(v1).strip())
            rec["outcome"] = ("gate-suspect" if same else
                              "spec-vetoed" if only_spec else "failed-after-repair")
            rec["identical_complaint"] = same
            rec["second_problem"] = tg._problem_text(v1)[:300]
            rec["review"] = {
                "code": code2, "facts": v1["facts"], "spec_notes": v1["spec_notes"],
                "hard": v1["hard"], "notes": v1["notes"],
                "t1_code": code, "t1_facts": v0["facts"], "t1_hard": v0["hard"],
                "t1_spec_notes": v0["spec_notes"], "t1_error": v0["error"],
                "t2_error": v1["error"],
                "t1_prompt_kind": p0.get("kind", ""), "t1_prompt": p0.get("prompt", ""),
                "t2_prompt_kind": p1.get("kind", ""), "t2_prompt": p1.get("prompt", "")}
            if only_spec or same:
                with tg.REVIEW_FILE.open("a") as f:
                    f.write(json.dumps({
                        "tier": "silver", "spec": spec, "code": code2,
                        "spec_notes": v1["spec_notes"], "facts": v1["facts"],
                        "code_model": model, "teacher_spec_id": sid,
                        "timestamp": stamp}) + "\n")
                rec["review_queued"] = True
            continue
        rec["outcome"] = "accepted-after-repair"
        img = tg._keep_render(v1["step"], tmp_root / sid / "t2",
                              f"{tg._slugify(model)}-{sid}-t2")
        verified = {"solids": v1["facts"].get("solids"), "bbox_mm": v1["facts"].get("bbox"),
                    "volume_mm3": v1["facts"].get("volume"), "advisories": v1["notes"]}
        rec["rows"].append({
            "kind": "good", "source": "teacher", "spec": spec, "code": code2, "image": img,
            "prompt_kind": p1.get("kind", ""), "system": p1.get("system", ""),
            "prompt": p1.get("prompt", ""), "code_model": model, "teacher_spec_id": sid,
            "verified": verified, "timestamp": stamp})
        rec["rows"].append({
            "kind": "fail", "source": "teacher-repair", "spec": spec,
            "code": code2, "bad_code": code, "image": img, "problem": problem[:300],
            "prompt_kind": p1.get("kind", ""), "system": p1.get("system", ""),
            "prompt": p1.get("prompt", ""),
            "bad_prompt_kind": p0.get("kind", ""), "bad_system": p0.get("system", ""),
            "bad_prompt": p0.get("prompt", ""),
            "code_model": model, "teacher_spec_id": sid,
            "verified": verified, "timestamp": stamp})

    # ── persist rows + summary ───────────────────────────────────────────────
    records = [c["rec"] for c in ctx.values()]
    rows = [r for rec in records for r in rec.get("rows", [])]
    if rows:
        with SFTPAIRS_FILE.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("outcome", "?")] = counts.get(r.get("outcome", "?"), 0) + 1
    summary = {"ran_at": started, "model": model, "mode": "batch", "dry_run": False,
               "specs": len(specs), "specs_done": len(records),
               "usd_spent": round(run_total[0], 4), "budget": args.budget or None,
               "stopped_early": "repair round skipped (budget)" if skipped_repairs else None,
               "elapsed_s": round(time.monotonic() - t0, 1),
               "outcomes": counts, "pairs_written": len(rows),
               "good": sum(1 for r in rows if r["kind"] == "good"),
               "fail_fix": sum(1 for r in rows if r["kind"] == "fail"),
               "per_spec": records}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"teacher_pilot_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\noutcomes: {counts}")
    print(f"spend this run: ${run_total[0]:.2f} (batch rates, list-derived)")
    print(f"pairs written: {len(rows)} ({summary['good']} good, {summary['fail_fix']} fail->fix)")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

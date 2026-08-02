#!/usr/bin/env python3
"""OpenSCAD teacher distillation — the CADAM-recipe corpus for a local Qwen fine-tune.

Same two-round Batch API shape as teacher_batch.py, but the target language is OpenSCAD and
the verifier is a FREE LOCAL COMPILE (seconds/spec, no build lock, no GPU):

    round 1: batch all codegen calls (openscad_gen's exact production prompt)
    verify:  OpenSCAD AppImage compile -> stderr + STL mesh facts + params-present check
    round 2: batch ONE repair per failure (stderr verbatim) -> fail->fix pairs

Budget fix vs teacher_batch.py: the repair round is RESERVED — round 1 refuses to submit
more than 60% of --budget by projection, so repairs always have headroom (the v2 hard runs
lost their entire repair round twice to the old check).

Rows -> ~/.openclaw/cad-sftpairs-scad.jsonl (a SEPARATE corpus: different language).
Acceptance: compiles clean-ish + >=3 customizer params + mesh facts sane (nonzero volume,
bbox under 1m). No [spec] dimension gate v1 — OpenSCAD training value is the param
discipline; measured dimensional checks can come later.

Usage:
    python3 scripts/teacher_batch_scad.py --specs benchmarks/teacher-batch2/specs.json --budget 8
    python3 scripts/teacher_batch_scad.py --specs benchmarks/teacher-hard/specs.json --budget 7 --model claude-opus-5
"""
from __future__ import annotations

import argparse
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

import anthropic  # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402

SCAD_PAIRS = Path.home() / ".openclaw" / "cad-sftpairs-scad.jsonl"
RESULTS_DIR = HERE / "benchmarks" / "results"
BATCH_DISCOUNT = 0.5
ROUND1_SHARE = 0.6            # reserve the rest for the repair round


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_scad(og, code: str, work: Path) -> dict:
    scad, stl = work / "build.scad", work / "build.stl"
    work.mkdir(parents=True, exist_ok=True)
    scad.write_text(code)
    ok, err = og.compile_scad(scad, stl)
    facts = og.stl_facts(stl) if ok else {}
    params = og.parse_params(code)
    problems = []
    if not ok:
        problems.append(f"compile failed:\n{err[-800:]}" if err else "compile produced no STL")
    else:
        if len([p for p in params if p['name'] != '$fn']) < 3:
            problems.append("fewer than 3 Customizer parameters — every important dimension "
                            "must be an annotated top-level parameter")
        v = facts.get("volume_mm3")
        if v is not None and v < 1:
            problems.append("mesh volume is ~zero — the model produced no real solid")
        bb = facts.get("bbox")
        if bb and max(bb) > 1000:
            problems.append(f"bbox {bb} exceeds 1000mm — check units/scale")
        if err and "WARNING" in err.upper():
            problems.append(f"compiler warnings:\n{err[-500:]}")
    return {"ok": ok and not problems, "compiled": ok, "problems": problems,
            "facts": facts, "params": params, "stderr": err}


def submit_and_wait(client, requests, model, run_total, poll_s=20):
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
            print(f"  [{r.custom_id}] {r.result.type}", flush=True)
            continue
        msg = r.result.message
        usage_d = {"input_tokens": msg.usage.input_tokens,
                   "output_tokens": msg.usage.output_tokens}
        cost = engine.cloud_call_cost(model, usage_d) * BATCH_DISCOUNT
        run_total[0] += cost
        with engine.CLOUD_SPEND_FILE.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                "model": model, **usage_d,
                                "usd": round(cost, 6), "batch": True,
                                "lang": "openscad"}) + "\n")
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if not text:
            print(f"  [{r.custom_id}] EMPTY (stop={msg.stop_reason}) — dropped", flush=True)
            continue
        out[r.custom_id] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--budget", type=float, required=True,
                    help="hard USD cap covering BOTH rounds; round 1 submits at most 60%% "
                         "of it by projection so repairs are never starved")
    ap.add_argument("--max-tokens", type=int, default=16000)
    a = ap.parse_args()

    tg = _load("teacher_gen")
    og = _load("openscad_gen")
    data = json.loads(Path(a.specs).read_text())
    specs = data["benchmarks"] if isinstance(data, dict) else data
    tg.guard_contamination(specs)
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        specs = [s for s in specs if s.get("id") in want]
    if a.limit:
        specs = specs[:a.limit]

    cc = engine.cloud_config()
    model = a.model or cc.get("model", "claude-sonnet-5")
    # OpenSCAD replies are short (no 10k-token geometry epics) — project conservatively at
    # $0.05/spec batch until measured_rates has scad history.
    est_per = 0.05
    max_r1 = int((a.budget * ROUND1_SHARE) / est_per)
    if len(specs) > max_r1:
        print(f"[cap] trimming round 1 to {max_r1} of {len(specs)} specs so the repair "
              f"round keeps its reserved {int((1-ROUND1_SHARE)*100)}% of ${a.budget:.2f}")
        specs = specs[:max_r1]
    print(f"[scad-teacher] model={model} specs={len(specs)} budget=${a.budget:.2f}")

    client = anthropic.Anthropic(api_key=engine._cloud_key(cc))
    run_total = [0.0]
    stamp = datetime.now(timezone.utc).isoformat()
    tmp = Path(tempfile.mkdtemp(prefix="scad_teacher_"))

    def req(sid, prompt):
        return Request(custom_id=sid, params=MessageCreateParamsNonStreaming(
            model=model, max_tokens=a.max_tokens, system=og._SYSTEM,
            messages=[{"role": "user", "content": prompt}]))

    def user_prompt(spec):
        return (f"USER REQUEST (every number is AUTHORITATIVE, mm):\n{spec}\n\n"
                f"Write the OpenSCAD file:")

    r1 = submit_and_wait(client, [req(s["id"], user_prompt(s["spec"])) for s in specs],
                         model, run_total)

    rows, records, repair_reqs, ctx = [], [], [], {}
    for s in specs:
        sid, spec = s["id"], s["spec"]
        rec = {"id": sid, "tier": s.get("tier"), "group": s.get("group"), "spec": spec,
               "outcome": "error", "rows": []}
        records.append(rec)
        raw = r1.get(sid)
        if raw is None:
            continue
        code = og.strip_fences(raw)
        v = verify_scad(og, code, tmp / sid / "t1")
        ctx[sid] = {"code": code, "v": v, "rec": rec, "spec": spec}
        if v["ok"]:
            rec["outcome"] = "accepted-first-try"
            rec["rows"].append({
                "kind": "good", "source": "teacher-scad", "lang": "openscad",
                "spec": spec, "code": code,
                "prompt_kind": "scad-codegen", "system": og._SYSTEM,
                "prompt": user_prompt(spec), "code_model": model,
                "teacher_spec_id": sid, "params_n": len(v["params"]),
                "verified": v["facts"], "timestamp": stamp})
        else:
            problem = "\n".join(v["problems"])
            rec["first_problem"] = problem[:300]
            repair_reqs.append(req(sid, og._REPAIR.format(err=problem, code=code)))

    if repair_reqs and run_total[0] < a.budget * 0.95:
        r2 = submit_and_wait(client, repair_reqs, model, run_total)
        for sid, c in ctx.items():
            rec = c["rec"]
            if rec["outcome"] != "error" or sid not in r2:
                if rec["outcome"] == "error" and c.get("v"):
                    rec["outcome"] = "failed-no-repair-result" if sid not in r2 else rec["outcome"]
                continue
            code2 = og.strip_fences(r2[sid])
            v2 = verify_scad(og, code2, tmp / sid / "t2")
            if v2["ok"]:
                rec["outcome"] = "accepted-after-repair"
                base = {"lang": "openscad", "spec": c["spec"], "code": code2,
                        "prompt_kind": "scad-revise", "system": og._SYSTEM,
                        "code_model": model, "teacher_spec_id": sid,
                        "params_n": len(v2["params"]), "verified": v2["facts"],
                        "timestamp": stamp}
                problem = "\n".join(c["v"]["problems"])[:600]
                rec["rows"].append({**base, "kind": "good", "source": "teacher-scad",
                                    "prompt": og._REPAIR.format(err=problem, code=c["code"])})
                rec["rows"].append({**base, "kind": "fail", "source": "teacher-scad-repair",
                                    "bad_code": c["code"], "problem": problem,
                                    "prompt": og._REPAIR.format(err=problem, code=c["code"]),
                                    "bad_prompt": user_prompt(c["spec"])})
            else:
                rec["outcome"] = "failed-after-repair"
                rec["second_problem"] = "\n".join(v2["problems"])[:300]

    for rec in records:
        rows.extend(rec["rows"])
    if rows:
        with SCAD_PAIRS.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    counts = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    summary = {"ran_at": stamp, "model": model, "mode": "batch-scad", "lang": "openscad",
               "specs_done": len(records), "usd_spent": round(run_total[0], 4),
               "budget": a.budget, "outcomes": counts, "pairs_written": len(rows),
               "good": sum(1 for r in rows if r["kind"] == "good"),
               "fail_fix": sum(1 for r in rows if r["kind"] == "fail"),
               "per_spec": records}
    out = RESULTS_DIR / f"teacher_scad_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\noutcomes: {counts}")
    print(f"spend: ${run_total[0]:.2f} (batch rates)  pairs: {len(rows)} "
          f"({summary['good']} good, {summary['fail_fix']} fail->fix)")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""lab/finish_batch2_rebuild.py -- finish the seeds a codefirst --batch run's Stage C dropped
as "rebuild-skipped-budget-priority" (lab/teacher_codefirst.py's --rebuild-max-usd), WITHOUT
re-running design or spec at all.

WHY THIS EXISTS (not just another teacher_codefirst.py resume): a resume that gives the same
out_dir + the same seeds-file re-derives design+spec for real request-matching (see
run_batch_stage's resumability -- it only reuses a cached batch when the custom_id SET is
identical), which forces main()'s done_ids() to see ALL 580 seeds as already-attempted (every
one has a terminal results.jsonl row from the run that produced this situation) and skip the
whole thing. There is no way to ask run_batch_pipeline for "just these 100, but still reuse
design/spec" without also fighting its id-set-matching resumability contract. This script
sidesteps that: each of the ~100 ids already has its spec.txt on disk (written by the ORIGINAL
run's Stage B, before Stage C's priority trim ever ran), so it re-derives the SAME production
rebuild prompt locally (capture_codegen_prompt -- free, deterministic, no API call) and
submits ONE fresh Batch containing ONLY the rebuild-stage calls these ids still need. Design
and spec are never touched, so this cannot re-bill them (real cost = just the new rebuild
batch, once).

Usage:
    ANTHROPIC_API_KEY="$(cat ~/.openclaw/cad-teacher.key)" python3 -X utf8 \\
        lab/finish_batch2_rebuild.py \\
        --out-dir benchmarks/results/card/codefirst-batch2-2026-09-25/claude-opus-5-5 \\
        --model claude-opus-5-5 --outcome rebuild-skipped-budget-priority \\
        --max-usd 0.80 [--priority-families lab/teacher_seeds_batch2_families.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import pathlib as _pathlib  # noqa: E402
_orig_write_text = _pathlib.Path.write_text
_orig_read_text = _pathlib.Path.read_text


def _write_text_utf8(self, data, encoding=None, errors=None, newline=None):
    return _orig_write_text(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)


def _read_text_utf8(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors or "replace")


_pathlib.Path.write_text = _write_text_utf8
_pathlib.Path.read_text = _read_text_utf8

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "lab"))

import geom_bands              # noqa: E402
import measure_part             # noqa: E402
from lab import teacher_codefirst as tc  # noqa: E402


def find_target_ids(out_dir: Path, model: str, outcome: str) -> list[str]:
    results = out_dir / "results.jsonl"
    ids: list[str] = []
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("model") == model and r.get("outcome") == outcome:
            ids.append(r["id"])
    return ids


def already_finished_ids(out_dir: Path, model: str) -> set[str]:
    """ids that ALREADY got a real rebuild attempt from THIS script in an earlier, possibly
    interrupted run -- has a "rebuild_call" usage row (only this script and Stage C ever
    write one) rather than the placeholder outcome. Lets this script itself be re-run safely
    without re-submitting for ids it already finished."""
    results = out_dir / "results.jsonl"
    done: set[str] = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("model") == model and isinstance(r.get("rebuild_call"), dict):
            done.add(r["id"])
    return done


def measured_rebuild_cost_per_call(out_dir: Path, model: str) -> float:
    """Real average $/call from THIS out_dir's own already-completed rebuild_call rows (the
    462 that Stage C already ran) -- a far more accurate estimate than
    estimate_batch_call_cost's stale default-OUT_ROOT fallback, since it is this exact run's
    own measured tokens at the exact same Batch pricing."""
    tin = tout = n = 0
    results = out_dir / "results.jsonl"
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("model") != model:
            continue
        c = r.get("rebuild_call")
        if isinstance(c, dict) and c.get("tokens_in") is not None:
            tin += c["tokens_in"]; tout += c["tokens_out"]; n += 1
    if n == 0:
        return tc.estimate_batch_call_cost(model, "rebuild")
    pin, pout = tc.batch_price(model)
    return (tin / n * pin + tout / n * pout) / 1_000_000


def priority_key(sid: str, tier: int, priority_map: dict[str, str] | None) -> tuple:
    order = {"fail": 0, "construction": 1, "control": 2, "unknown": 3}
    bucket = "unknown"
    if priority_map is not None:
        fam = priority_map.get(sid, "")
        bucket = fam.split(":", 1)[0] if fam else "unknown"
    return (order.get(bucket, 3), -tier, sid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True, choices=sorted(tc.MODEL_PRICES))
    ap.add_argument("--outcome", default="rebuild-skipped-budget-priority")
    ap.add_argument("--max-usd", type=float, required=True,
                    help="hard cap on THIS script's own new spend (rebuild stage only)")
    ap.add_argument("--priority-families", default="")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    priority_map = None
    if a.priority_families:
        priority_map = json.loads(Path(a.priority_families).read_text(encoding="utf-8"))

    target_ids = find_target_ids(out_dir, a.model, a.outcome)
    finished = already_finished_ids(out_dir, a.model)
    target_ids = [i for i in target_ids if i not in finished]
    print(f"[finish] {len(target_ids)} ids with outcome={a.outcome!r} still need a rebuild "
         f"attempt ({len(finished)} already finished by an earlier run of this script)")
    if not target_ids:
        print("[finish] nothing to do")
        return 0

    # tier lookup: results.jsonl's design_gate rows don't carry tier, but the seed's own
    # idea/tier were captured on the SAME row as the "outcome" (design_call/tier fields are
    # set on the first row written for that id back in Stage A/B) -- reread results.jsonl
    # once more for {id: tier}.
    tiers: dict[str, int] = {}
    for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("model") == a.model and r.get("id") in target_ids and "tier" in r:
            tiers[r["id"]] = r["tier"]

    per_call = measured_rebuild_cost_per_call(out_dir, a.model) * 1.2  # same margin as budget_check
    print(f"[finish] measured per-call estimate (incl. 1.2x margin): ${per_call:.5f}")

    ordered = sorted(target_ids, key=lambda sid: priority_key(sid, tiers.get(sid, 0), priority_map))
    subset: list[str] = []
    running = 0.0
    for sid in ordered:
        nxt = running + per_call
        if nxt > a.max_usd:
            break
        subset.append(sid)
        running = nxt
    print(f"[finish] {len(subset)}/{len(target_ids)} ids fit under --max-usd ${a.max_usd:.4f} "
         f"(est ${running:.4f}); {len(target_ids) - len(subset)} left for a future top-up")
    if not subset:
        print("[finish] max-usd too small for even one call -- nothing submitted, $0 spent")
        return 0

    # local, free: re-derive each id's production rebuild prompt from its already-written
    # spec.txt (Stage B's own output, untouched by this script).
    prompts: dict[str, dict] = {}
    for sid in subset:
        spec_text = (out_dir / "builds" / sid / "spec.txt").read_text(encoding="utf-8")
        prompts[sid] = {"spec": spec_text, **tc.capture_codegen_prompt(spec_text)}

    since_ts = datetime.now(timezone.utc)
    state = {"total_usd": 0.0, "calls": 0, "max_call_usd": 0.0}
    tc.budget_check(a.model, since_ts, a.max_usd, state, reserve_calls=len(subset),
                    per_call_estimate_usd=per_call / 1.2)

    batch = tc.client().messages.batches.create(
        requests=[tc._batch_request(sid, a.model, prompts[sid]["system"], prompts[sid]["prompt"])
                 for sid in subset])
    print(f"[finish] submitted batch {batch.id} ({len(subset)} rebuild requests)")
    tc._record_batch_id(out_dir, "rebuild-finish", batch.id, subset)

    deadline = time.monotonic() + tc.BATCH_MAX_WAIT_S
    while True:
        b = tc.client().messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"batch {batch.id} did not end within {tc.BATCH_MAX_WAIT_S}s "
                                f"(status={b.processing_status})")
        time.sleep(tc.BATCH_POLL_S)

    results_by_id: dict[str, dict] = {}
    for entry in tc.client().messages.batches.results(batch.id):
        cid = entry.custom_id
        if entry.result.type != "succeeded":
            results_by_id[cid] = {"text": "", "stop_reason": f"batch-{entry.result.type}",
                                  "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
            continue
        msg = entry.result.message
        meter = tc.record_spend(a.model, msg.usage, 0.0, state, batch=True)  # REAL, new, once
        text = "".join(c.text for c in msg.content if getattr(c, "type", "") == "text")
        results_by_id[cid] = {"text": text, "stop_reason": msg.stop_reason, **meter}
    print(f"[finish] results read for {len(results_by_id)}/{len(subset)} requests")

    kept_n = 0
    for sid in subset:
        seed_tier = tiers.get(sid, 0)
        part_dir = out_dir / "builds" / sid
        spec_text = prompts[sid]["spec"]
        resp = results_by_id.get(sid) or {"stop_reason": "batch-missing", "text": "",
                                          "in": 0, "out": 0, "cost": 0.0, "wall_s": 0.0}
        row: dict = {"id": sid, "tier": seed_tier, "model": a.model,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "rebuild_call": tc._usage_row(resp)}
        if resp["stop_reason"] == "refusal" or not resp["text"].strip():
            row.update(outcome="rebuild-refused", reason=f"stop_reason={resp['stop_reason']}")
            tc.append_result(out_dir, row)
            print(f"  {sid}  rebuild-refused")
            continue
        rebuild_code = tc.to_code(resp["text"], spec_text)
        (part_dir / "rebuild_code.py").write_text(rebuild_code, encoding="utf-8")
        rebuild_build_dir = part_dir / "rebuild_build"
        gate2 = tc.build_and_gate(rebuild_code, rebuild_build_dir, spec_text)
        row["rebuild_gate"] = {"gate_hard": gate2.get("gate_hard"), "gate_spec": gate2.get("gate_spec"),
                              "error": gate2.get("error"), "unscored_reason": gate2.get("unscored_reason"),
                              "solids": gate2.get("facts", {}).get("solids")}
        rebuild_step = rebuild_build_dir / "build.step"
        if gate2.get("error") or not rebuild_step.exists():
            row.update(outcome="rebuild-crashed", reason=gate2.get("error") or "no build.step", band="fail")
            tc.append_result(out_dir, row)
            print(f"  {sid}  rebuild-crashed")
            continue
        ref_stl = part_dir / "reference.stl"
        score = geom_bands.score_against_reference(rebuild_step, ref_stl)
        row["score"] = score
        row["band"] = score.get("band")
        try:
            rebuild_measurements = measure_part.measure(rebuild_step)
        except Exception as e:
            row.update(outcome="rebuild-measure-failed", reason=str(e)[:300], kept=False)
            tc.append_result(out_dir, row)
            print(f"  {sid}  rebuild-measure-failed")
            continue
        (part_dir / "rebuild_measurements.json").write_text(
            json.dumps(rebuild_measurements, indent=2), encoding="utf-8")
        measurements = json.loads((part_dir / "measurements.json").read_text(encoding="utf-8"))
        kept, feature_problems = tc.keep_pair(score.get("band"), gate2, measurements, rebuild_measurements)
        row["kept"] = kept
        row["feature_check"] = feature_problems
        if kept:
            row["outcome"] = "kept"
            kept_n += 1
            idea = json.loads((part_dir / "design_prompt.json").read_text(encoding="utf-8")).get("seed", "")
            design_code = (part_dir / "design_code.py").read_text(encoding="utf-8")
            pair = {"id": sid, "tier": seed_tier, "model": a.model, "idea": idea, "spec": spec_text,
                   "code": rebuild_code, "design_code": design_code, "band": score.get("band"),
                   "chamfer_mm": score.get("chamfer_mm"), "volume_diff_pct": score.get("volume_diff_pct"),
                   "measurements": measurements, "rebuild_measurements": rebuild_measurements,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            with (out_dir / "pairs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(pair, default=str) + "\n")
        elif score.get("band") != "match":
            row["outcome"] = "rebuild-mismatch"
        elif gate2.get("gate_hard") or gate2.get("gate_spec") or gate2.get("error") or gate2.get("unscored_reason"):
            row["outcome"] = "rebuild-gate-dirty"
        else:
            row["outcome"] = "rebuild-feature-mismatch"
        tc.append_result(out_dir, row)
        print(f"  {sid}  {row['outcome']:<26} band={row.get('band')}  kept={kept}  "
             f"spend=${state['total_usd']:.4f}")

    print(f"\n[finish] TOTAL NEW SPEND: ${state['total_usd']:.4f} over {state['calls']} call(s)")
    print(f"[finish] kept: {kept_n} / {len(subset)} attempted this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())

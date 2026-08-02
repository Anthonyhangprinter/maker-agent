#!/usr/bin/env python3
"""gift_sample.py — GIFT-style candidate sampling + band bucketing (arXiv 2603.27448).

For every corpus spec whose stored code is the ground truth (gold + stage-c rows in
~/.openclaw/cad-examples.jsonl — NEVER benchmark/heldout specs, those must stay heldout),
sample K fresh programs from the fast coder at varied temperatures, execute each, and score
it against the reference geometry with scripts/geom_bands.py:

  match/valid  -> GIFT-REJECT: a correct-but-differently-written program; kept as an extra
                  (spec, code) SFT pair — teaches the model multiple valid expressions.
  near_miss    -> GIFT-FAIL: recognisably the part built wrong; its two-panel render + the
                  REFERENCE code become a geometric-denoising pair (image of the error ->
                  correct program).
  fail         -> discarded (counted).

Rows append to ~/.openclaw/cad-sftpairs.jsonl (source=gift-reject / gift-fail); a per-run
summary lands in benchmarks/results/gift_sample_<ts>.json. This is the local, small-scale
analog of the paper's 163k->370k amplification — the census decides whether it's enough or
public data (GenCAD-Code) must be adapted on top.

GPU etiquette: takes the machine-wide build flock PER SPEC (released between specs) so an
interactive Satine/web build queues briefly instead of colliding; run the whole thing inside
a MemoryHigh-capped unit per SESSION PROTOCOL (run via systemd-run, see run_refresh.sh).

Usage:
  python3 scripts/gift_sample.py                       # all corpus specs, K=8, fast coder
  python3 scripts/gift_sample.py --k 4 --limit 3       # quick trial
  python3 scripts/gift_sample.py --model qwen3:8b      # sample a different coder
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import cad_engine as engine                              # noqa: E402
from geom_bands import score_against_reference, step_to_stl   # noqa: E402
from cad_v5.config import SFTPAIRS_DIR, SFTPAIRS_FILE    # noqa: E402

TEMPS = [0.2, 0.4, 0.6, 0.8]


def sample_spec(row: dict, k: int, results: list, use_raw: bool = True) -> dict:
    spec, ref_code = row["spec"], row["code"]
    counts = {"match": 0, "valid": 0, "near_miss": 0, "fail": 0, "no_run": 0, "dup": 0}
    stats = {"spec": spec, "counts": counts}
    with tempfile.TemporaryDirectory(prefix="gift_") as td:
        work = Path(td)
        # Ground truth: execute the corpus row's stored (verified) code.
        try:
            (work / "ref").mkdir()
            ref_step, _ = engine.run_step(ref_code, work / "ref")
            ref_stl = work / "ref.stl"
            step_to_stl(ref_step, ref_stl)
        except Exception as e:
            stats["error"] = f"reference code failed: {str(e)[:150]}"
            return stats
        # Raw mode (default, matches the brief-less production path AND the teacher rows):
        # verbatim spec + retrieval notes, no qwen3:8b brief. --brief restores the old path.
        notes = None
        brief = None
        if use_raw:
            notes = engine.retrieval_notes_for(spec)
        else:
            try:
                brief = engine.build_brief(spec)
                engine.reconcile_expected(brief, spec)
            except Exception as e:
                stats["error"] = f"brief failed: {str(e)[:150]}"
                return stats

        seen: set = {hashlib.sha1(ref_code.encode()).hexdigest()}
        for i in range(k):
            t = TEMPS[i % len(TEMPS)]
            try:
                cand = (engine.generate_code_raw(spec, notes, temperature=t) if use_raw
                        else engine.generate_code(brief, spec, temperature=t))
            except Exception:
                counts["no_run"] += 1
                continue
            # The exact production prompt that produced this candidate — persisted with the
            # row so training never has to re-derive it (gift_sample's old rows had none).
            p = dict(engine._LAST_PROMPT)
            h = hashlib.sha1(cand.encode()).hexdigest()
            if h in seen:
                counts["dup"] += 1
                continue
            seen.add(h)
            if engine._check_syntax(cand):
                counts["no_run"] += 1
                continue
            cdir = work / f"cand{i}"
            cdir.mkdir(exist_ok=True)
            # One N1-style repair retry, mirroring the engine's inline auto-fix: a runtime
            # error is the 7B's dominant failure mode and salvaging the candidate costs one
            # call — without this most of the sampling budget dies before scoring.
            try:
                cstep, _ = engine.run_step(cand, cdir)
            except Exception as e:
                try:
                    cand = engine.revise_script(spec, cand,
                                                f"The script failed to run:\n{str(e)[:600]}")
                    p = dict(engine._LAST_PROMPT)   # the code now comes from the revise call
                    if engine._check_syntax(cand):
                        raise RuntimeError("still broken")
                    cstep, _ = engine.run_step(cand, cdir)
                except Exception:
                    counts["no_run"] += 1
                    continue
            score = score_against_reference(cstep, ref_stl)
            band = score["band"]
            counts[band] += 1
            ts = datetime.now(timezone.utc).isoformat()
            meta = {"chamfer_mm": score.get("chamfer_mm"),
                    "volume_diff_pct": score.get("volume_diff_pct")}
            if band in ("match", "valid"):
                results.append({"kind": "good", "source": "gift-reject", "band": band,
                                "spec": spec, "code": cand, "image": "",
                                "prompt_kind": p.get("kind", ""),
                                "system": p.get("system", ""), "prompt": p.get("prompt", ""),
                                "code_model": engine._code_model(), "timestamp": ts, **meta})
            elif band == "near_miss":
                png = ""
                try:
                    # NB: do not name this `p` — that's the candidate's prompt dict above.
                    render_path = engine.run_render(cstep, cdir)
                    SFTPAIRS_DIR.mkdir(parents=True, exist_ok=True)
                    dst = SFTPAIRS_DIR / f"gift-{h[:12]}.png"
                    dst.write_bytes(Path(render_path).read_bytes())
                    png = str(dst)
                except Exception:
                    pass
                problem = (f"near-miss geometry: chamfer {meta['chamfer_mm']:.1f}mm from "
                           f"reference, volume off {meta['volume_diff_pct']}%"
                           if meta["chamfer_mm"] is not None else "near-miss geometry")
                # The fix-turn (system, prompt) whose completion is ref_code was never sent
                # to a model, so only the bad side's prompt is real — persist it; the
                # compiler reconstructs the revise-turn prompt from (spec, bad_code, problem).
                results.append({"kind": "fail", "source": "gift-fail", "spec": spec,
                                "code": ref_code, "bad_code": cand, "image": png,
                                "problem": problem,
                                "bad_prompt_kind": p.get("kind", ""),
                                "bad_system": p.get("system", ""),
                                "bad_prompt": p.get("prompt", ""),
                                "code_model": engine._code_model(),
                                "timestamp": ts, **meta})
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="candidates per spec (default 8)")
    ap.add_argument("--limit", type=int, default=0, help="max specs (0 = all)")
    ap.add_argument("--model", default=None, help="coder to sample (default: fast rung)")
    ap.add_argument("--brief", action="store_true",
                    help="legacy qwen3:8b brief path (default is raw: verbatim spec + "
                         "retrieval notes, matching brief-less production)")
    a = ap.parse_args()

    engine._ACTIVE_CODE_MODEL = a.model or engine.CODE_MODEL_FAST
    corpus = [r for r in engine.cad_retrieval.load_corpus() if r.get("code")]
    if a.limit:
        corpus = corpus[:a.limit]
    print(f"gift_sample: {len(corpus)} specs × K={a.k} on {engine._code_model()}",
          file=sys.stderr)

    all_stats, results, t0 = [], [], time.monotonic()
    for n, row in enumerate(corpus, 1):
        # Per-spec build lock: an interactive build queues for one spec's worth of sampling
        # (minutes), not the whole run (hours).
        fh = engine._acquire_build_lock(f"gift_sample: {row['spec'][:60]}")
        try:
            st = sample_spec(row, a.k, results, use_raw=not a.brief)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        all_stats.append(st)
        note = f"  ERROR: {st['error']}" if "error" in st else ""
        print(f"[{n}/{len(corpus)}] {row['spec'][:60]} -> {st.get('counts')}{note}",
              file=sys.stderr)

    if results:
        with SFTPAIRS_FILE.open("a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = HERE / "benchmarks" / "results" / f"gift_sample_{tag}.json"
    tot = {b: sum(s["counts"].get(b, 0) for s in all_stats if "counts" in s)
           for b in ("match", "valid", "near_miss", "fail", "no_run", "dup")}
    out.write_text(json.dumps({
        "model": engine._code_model(), "k": a.k, "specs": len(corpus),
        "elapsed_s": round(time.monotonic() - t0, 1), "totals": tot,
        "pairs_written": len(results), "per_spec": all_stats}, indent=1))
    print(f"\ntotals {tot}  -> {len(results)} pairs appended to {SFTPAIRS_FILE}\n{out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

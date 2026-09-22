#!/usr/bin/env python3
"""
Judge-eval Task 4: turn judgements.jsonl + labelled.json into REPORT.md + report.json.

Definitions used throughout:
  - truth: "correct" (band == match) or "wrong" (built, any other band, or a known-bad fixture).
  - verdict: "correct" / "wrong" / "abstain" (no parseable JSON with a verdict key).
  - PRECISION of a "correct" verdict = P(truth==correct | verdict==correct). This is the number
    that matters for "can this judge confirm training data": if it says correct, how often is
    that true.
  - RECALL of "correct" = P(verdict==correct | truth==correct): of the parts that really are
    correct, how many does the judge confirm.
  - wrong-caught = P(verdict==wrong | truth==wrong): of the parts that are really wrong, how
    many does the judge flag (an abstain does NOT count as caught).
  - abstain rate = P(verdict==abstain).
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID = "2026-09-19"
RESULTS_DIR = HERE / "results" / RUN_ID


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def load_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def confusion(rows):
    """rows: list of {truth, verdict}. Returns dict of counts."""
    c = defaultdict(int)
    for r in rows:
        c[(r["truth"], r["verdict"])] += 1
    return dict(c)


def precision_recall(rows, conf_threshold=None):
    if conf_threshold is not None:
        # A "correct" verdict only counts if confidence clears the bar; otherwise treat it as
        # not-confirmed (folded into the denominator as a miss, not silently dropped).
        def v(r):
            if r["verdict"] == "correct" and (r["confidence"] is None or r["confidence"] < conf_threshold):
                return "unconfirmed"
            return r["verdict"]
    else:
        def v(r):
            return r["verdict"]

    tp = sum(1 for r in rows if v(r) == "correct" and r["truth"] == "correct")
    fp = sum(1 for r in rows if v(r) == "correct" and r["truth"] == "wrong")
    n_pred_correct = tp + fp
    n_truth_correct = sum(1 for r in rows if r["truth"] == "correct")
    n_truth_wrong = sum(1 for r in rows if r["truth"] == "wrong")
    wrong_caught = sum(1 for r in rows if v(r) == "wrong" and r["truth"] == "wrong")
    abstain = sum(1 for r in rows if r["verdict"] == "abstain")

    precision = tp / n_pred_correct if n_pred_correct else None
    recall = tp / n_truth_correct if n_truth_correct else None
    wrong_recall = wrong_caught / n_truth_wrong if n_truth_wrong else None
    ci = wilson_ci(tp, n_pred_correct) if n_pred_correct else (None, None)

    return {
        "n": len(rows), "tp": tp, "fp": fp, "n_pred_correct": n_pred_correct,
        "n_truth_correct": n_truth_correct, "n_truth_wrong": n_truth_wrong,
        "wrong_caught": wrong_caught, "abstain": abstain,
        "precision": precision, "precision_wilson95": ci,
        "recall": recall, "wrong_recall": wrong_recall,
        "abstain_rate": abstain / len(rows) if rows else None,
    }


def fmt_pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def fmt_ci(ci):
    lo, hi = ci
    if lo is None:
        return "n/a"
    return f"[{lo*100:.1f}%, {hi*100:.1f}%]"


def main():
    labelled = {r["program_id"]: r for r in json.loads((HERE / "labelled.json").read_text(encoding="utf-8"))}
    judgements = load_jsonl(RESULTS_DIR / "judgements.jsonl")
    manifest = json.loads((RESULTS_DIR / "build_manifest.json").read_text(encoding="utf-8"))

    n_labelled = len(labelled)
    n_rebuilt = sum(1 for m in manifest.values() if m.get("ok"))
    n_correct_truth = sum(1 for r in labelled.values() if r["truth"] == "correct")
    n_wrong_truth = sum(1 for r in labelled.values() if r["truth"] == "wrong")

    by_mode = defaultdict(list)
    for j in judgements:
        by_mode[j["mode"]].append(j)

    report = {
        "run_id": RUN_ID,
        "labelled_set_size": n_labelled,
        "rebuilt_ok": n_rebuilt,
        "ground_truth": {"correct": n_correct_truth, "wrong": n_wrong_truth},
        "modes": {},
    }

    md = []
    md.append(f"# Judge reliability: qwen3.8-27b scoring known-correct/known-wrong CAD parts\n")
    md.append(f"Run id: {RUN_ID}. Labelled set: {n_labelled} programs "
              f"({n_correct_truth} correct / {n_wrong_truth} wrong by reference-scored Chamfer "
              f"band or the named fixture verdicts). Rebuilt successfully: {n_rebuilt}. "
              f"Judge calls made: {len(judgements)} of a possible {2 * n_rebuilt}.\n")
    md.append("Ground truth: `band == \"match\"` = CORRECT; built with any other band = WRONG "
              "(near_miss/valid/fail all count as wrong here, since the question is exact "
              "match, not partial credit); a program that never built is excluded, a crash "
              "needs no judge. Three fixture rows (V064, V066 T=0.2, V066 T=0.5) are known-wrong "
              "/ known-wrong / known-correct by direct inspection of the code, not the card "
              "scorer, since they came from a live agreement check outside this card.\n")
    md.append("**n is small (18-40 per cell in most breakdowns). Every percentage below should "
              "be read with its Wilson 95% interval, not as a point estimate.**\n")

    headline_rows = []
    for mode in ("text", "vision"):
        rows = by_mode.get(mode, [])
        pr = precision_recall(rows)
        cm = confusion(rows)
        wall_times = [j["wall_s"] for j in rows if j.get("wall_s") is not None]
        tokens = [j["usage"].get("completion_tokens") for j in rows if j.get("usage") and j["usage"].get("completion_tokens") is not None]
        med_time = statistics.median(wall_times) if wall_times else None
        med_tok = statistics.median(tokens) if tokens else None

        by_writer = defaultdict(list)
        for j in rows:
            by_writer[j["writer_family"]].append(j)
        writer_stats = {fam: precision_recall(rs) for fam, rs in by_writer.items()}

        thresh_stats = {str(t): precision_recall(rows, conf_threshold=t) for t in (0.6, 0.8, 0.9)}

        report["modes"][mode] = {
            "n_calls": len(rows), "n_errors": sum(1 for j in rows if not j.get("ok")),
            "confusion": {f"{k[0]}->{k[1]}": v for k, v in cm.items()},
            "precision_recall": pr,
            "by_writer_family": writer_stats,
            "by_confidence_threshold": thresh_stats,
            "median_wall_s": med_time,
            "median_completion_tokens": med_tok,
        }

        headline_rows.append((mode, pr, med_time, med_tok))

        md.append(f"## Mode {mode.upper()} ({'text: spec + code + facts' if mode=='text' else 'vision: spec + code + facts + two-panel render'})\n")
        md.append("| truth \\ verdict | correct | wrong | abstain |")
        md.append("|---|---|---|---|")
        for t in ("correct", "wrong"):
            row = [str(cm.get((t, v), 0)) for v in ("correct", "wrong", "abstain")]
            md.append(f"| {t} | " + " | ".join(row) + " |")
        md.append("")
        md.append(f"- Precision of a \"correct\" verdict: **{fmt_pct(pr['precision'])}** "
                  f"({pr['tp']}/{pr['n_pred_correct']}), Wilson 95% CI {fmt_ci(pr['precision_wilson95'])}")
        md.append(f"- Recall of \"correct\" (of the {pr['n_truth_correct']} truly-correct parts, how many it confirmed): **{fmt_pct(pr['recall'])}**")
        md.append(f"- Wrong parts caught (of the {pr['n_truth_wrong']} truly-wrong parts): **{pr['wrong_caught']}/{pr['n_truth_wrong']}** = {fmt_pct(pr['wrong_recall'])}")
        md.append(f"- Abstain rate: {fmt_pct(pr['abstain_rate'])} ({pr['abstain']}/{pr['n']})")
        md.append(f"- Median seconds/judgement: {med_time:.1f}" if med_time else "- Median seconds/judgement: n/a")
        md.append(f"- Median completion tokens/judgement: {med_tok:.0f}" if med_tok else "- Median completion tokens/judgement: n/a")
        md.append("")
        md.append("By writer family:")
        md.append("| writer | n | precision | recall | wrong caught | abstain |")
        md.append("|---|---|---|---|---|---|")
        for fam, s in writer_stats.items():
            md.append(f"| {fam} | {s['n']} | {fmt_pct(s['precision'])} | {fmt_pct(s['recall'])} | "
                      f"{s['wrong_caught']}/{s['n_truth_wrong']} | {fmt_pct(s['abstain_rate'])} |")
        md.append("")
        md.append("By confidence threshold (a \"correct\" verdict only counts if confidence clears the bar):")
        md.append("| threshold | n confirmed correct | precision | Wilson 95% CI | recall of correct |")
        md.append("|---|---|---|---|---|")
        for t, s in thresh_stats.items():
            md.append(f"| >= {t} | {s['n_pred_correct']} | {fmt_pct(s['precision'])} | {fmt_ci(s['precision_wilson95'])} | {fmt_pct(s['recall'])} |")
        md.append("")

    # Headline table
    md_headline = ["## Headline\n", "| mode | precision(correct) | 95% CI | recall(correct) | wrong caught | abstain rate | median s | median tok |",
                   "|---|---|---|---|---|---|---|---|"]
    for mode, pr, mt, mtok in headline_rows:
        mt_str = f"{mt:.1f}" if mt is not None else "n/a"
        mtok_str = f"{mtok:.0f}" if mtok is not None else "n/a"
        md_headline.append(
            f"| {mode} | {fmt_pct(pr['precision'])} | {fmt_ci(pr['precision_wilson95'])} | "
            f"{fmt_pct(pr['recall'])} | {pr['wrong_caught']}/{pr['n_truth_wrong']} | "
            f"{fmt_pct(pr['abstain_rate'])} | {mt_str} | {mtok_str} |"
        )
    md = md[:2] + md_headline + [""] + md[2:]

    # Named cases
    md.append("## Named cases: V064 and both V066 rows\n")
    fixture_map = {
        "fixture::v064_wrong_sheared_lip": "V064 (WRONG — lip cutter shears the whole top off, not just the rim)",
        "fixture::v066_t02_wrong_slot": "V066 T=0.2 (WRONG — Z-axis cylinder makes a vertical slot, not a round hole through the divider)",
        "fixture::v066_t05_floor_4p5mm": "V066 T=0.5 (WRONG, corrected 2026-09-22 — cylinder rotated onto the divider's X thickness axis as intended, but the floor is 4.5mm where the spec calls for 3mm)",
    }
    jmap = {(j["program_id"], j["mode"]): j for j in judgements}
    for pid, label in fixture_map.items():
        md.append(f"### {label}\n")
        for mode in ("text", "vision"):
            j = jmap.get((pid, mode))
            md.append(f"**Mode {mode}:**")
            if not j:
                md.append("(no judgement recorded)\n")
                continue
            md.append(f"- verdict: `{j['verdict']}`, confidence: {j['confidence']}, problems: {j['problems']}")
            content = (j.get("content") or "").strip()
            md.append("- verbatim reply:")
            md.append("```")
            md.append(content if content else "(empty content)")
            md.append("```")
        md.append("")

    md_text = "\n".join(md)
    (RESULTS_DIR / "REPORT.md").write_text(md_text, encoding="utf-8")
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'REPORT.md'}")
    print(f"Wrote {RESULTS_DIR / 'report.json'}")


if __name__ == "__main__":
    main()

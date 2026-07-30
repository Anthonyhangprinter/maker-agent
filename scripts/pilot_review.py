#!/usr/bin/env python3
"""Build a browsable review page comparing teacher runs, part by part.

Answers the three questions you actually need when judging a run:
  * what exactly was the model answering  -> the verbatim production prompt, per part
  * how many attempts did it take         -> turn count + outcome, per part
  * how do the models differ             -> renders side by side, same row

Everything is regenerated from each row's own stored CODE. It never reads back the STEP or
PNG files a run left behind: those are written to shared paths (`/tmp/teacher_gen_*/`,
`cad-sftpairs/teacher-<id>-t1.png`) and a second run silently overwrites the first, which
made two different programs report byte-identical geometry (2026-07-30).

    python3 scripts/pilot_review.py A.json B.json [--out ~/CAD/teacher-pilot]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT_DEFAULT = Path.home() / "CAD" / "teacher-pilot"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")


def build_artifacts(code: str, work: Path, dest: Path, stem: str) -> tuple[str, str]:
    """code -> STEP + PNG in dest. Returns (step_name, png_name), '' where it failed."""
    work.mkdir(parents=True, exist_ok=True)
    src, step = work / "src.py", work / "out.step"
    src.write_text(code)
    subprocess.run([sys.executable, str(HERE / "scripts" / "step"), str(src), str(step)],
                   capture_output=True, text=True, timeout=240)
    if not step.exists():
        return "", ""
    (dest / f"{stem}.step").write_bytes(step.read_bytes())
    (dest / f"{stem}.py").write_text(code)
    png = ""
    try:
        subprocess.run([sys.executable, str(HERE / "scripts" / "render"),
                        str(step), str(work / "r.png")], capture_output=True, timeout=240)
        if (work / "r.png").exists():
            (dest / f"{stem}.png").write_bytes((work / "r.png").read_bytes())
            png = f"{stem}.png"
    except Exception:
        pass
    return f"{stem}.step", png


def load(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["_recs"] = {r["id"]: r for r in d["per_spec"]}
    d["_slug"] = slug(d.get("model", ""))
    return d


CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;
background:#fbfbfa;color:#1b1b1a}
h1{margin:0 0 4px}.sub{color:#666;margin-bottom:20px}
table.sum{border-collapse:collapse;margin-bottom:28px}
table.sum td,table.sum th{border:1px solid #ddd;padding:6px 12px;text-align:left}
.part{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:16px;margin-bottom:18px}
.spec{font-weight:600;margin-bottom:2px}
.meta{color:#777;font-size:13px;margin-bottom:12px}
.cols{display:flex;gap:16px;flex-wrap:wrap}
.col{flex:1 1 420px;min-width:340px;border:1px solid #eee;border-radius:6px;padding:10px}
.col h3{margin:0 0 6px;font-size:14px}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;font-weight:600}
.ok{background:#e5f4e8;color:#176b32}.rep{background:#fff4e0;color:#8a5300}
.bad{background:#fdeaea;color:#a02020}
.turns{background:#eef1f6;color:#38414f;margin-left:6px}
img{max-width:100%;border:1px solid #eee;border-radius:4px;background:#fff}
details{margin-top:8px}summary{cursor:pointer;font-size:13px;color:#444;user-select:none}
pre{background:#f6f6f4;border:1px solid #e8e8e4;border-radius:4px;padding:10px;
overflow-x:auto;font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
.nogeo{color:#a02020;font-size:13px}
.why{background:#fdf3f3;border-left:3px solid #d98080;padding:7px 10px;margin-top:8px;
font-size:13px;border-radius:0 4px 4px 0}
.why.warn{background:#fff8e6;border-left-color:#d9a640}
.cap{font-size:12px;color:#666;margin-top:4px}
.cap a{color:#2a5db0}
"""


def classify(rec: dict) -> str:
    """Which check actually stopped this part. Grouped so a run's weak spots are countable."""
    oc = rec.get("outcome", "")
    if oc.startswith("accepted") or oc == "skipped-helper":
        return oc
    txt = f"{rec.get('first_problem', '')} {rec.get('second_problem', '')}".lower()
    if "no build123d part/compound" in txt:      return "fail: never bound `result`"
    if "no axis of the part measures" in txt:    return "fail: overall size wrong"
    if "apart" in txt or "circle" in txt:        return "fail: hole placement/spacing"
    if "hole" in txt and "but the solid has" in txt: return "fail: wrong hole count"
    if "pass fully through" in txt:              return "fail: hole not through"
    if "bore" in txt:                            return "fail: missing bore"
    if "separate bodies" in txt:                 return "fail: unfused bodies"
    if "hollow" in txt:                          return "fail: solid where hollow wanted"
    if "error" in txt or "traceback" in txt:     return "fail: script error"
    return "fail: other"


def stats_html(runs: list[dict]) -> str:
    """Failure breakdown by cause and by tier — the two cuts that tell you where to aim."""
    causes = sorted({classify(r) for run in runs for r in run["per_spec"]})
    head = "".join(f"<th>{html.escape(m['model'])}</th>" for m in runs)
    body = ""
    for c in causes:
        cells = "".join(
            f"<td>{sum(1 for r in run['per_spec'] if classify(r) == c)}</td>" for run in runs)
        body += f"<tr><td>{html.escape(c)}</td>{cells}</tr>"

    tiers = sorted({r.get("tier") for run in runs for r in run["per_spec"] if r.get("tier")})
    tbody = ""
    for t in tiers:
        cells = ""
        for run in runs:
            ids = [r for r in run["per_spec"] if r.get("tier") == t]
            acc = sum(1 for r in ids if r.get("outcome", "").startswith("accepted"))
            cells += f"<td>{acc}/{len(ids)}</td>"
        tbody += f"<tr><td>tier {t}</td>{cells}</tr>"

    return (f'<h2>Why parts failed</h2><table class="sum"><tr><th>cause</th>{head}</tr>'
            f'{body}</table>'
            f'<h2>Accepted by difficulty tier</h2><table class="sum">'
            f'<tr><th></th>{head}</tr>{tbody}</table>')


def badge(outcome: str) -> str:
    cls = "ok" if outcome == "accepted-first-try" else \
          "rep" if outcome.startswith("accepted") else "bad"
    return f'<span class="badge {cls}">{html.escape(outcome or "—")}</span>'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--no-render", action="store_true",
                    help="reuse PNG/STEP already in --out (fast page rebuild)")
    args = ap.parse_args()

    runs = [load(Path(p)) for p in args.summaries]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pilot_review_"))

    ids = sorted({i for r in runs for i in r["_recs"]})
    system_seen = ""

    parts_html = []
    for sid in ids:
        first = next((r["_recs"][sid] for r in runs if sid in r["_recs"]), None)
        if not first:
            continue
        cols = []
        for run in runs:
            rec = run["_recs"].get(sid)
            if not rec:
                cols.append(f'<div class="col"><h3>{html.escape(run["model"])}</h3>'
                            f'<div class="nogeo">not in this run</div></div>')
                continue
            rows = rec.get("rows") or []
            review = rec.get("review") or {}
            mdir = out / run["_slug"]
            mdir.mkdir(parents=True, exist_ok=True)
            turns = rec.get("turns", 0)
            accepted = bool(rows)

            # A REJECTED attempt still produced geometry worth looking at — C01's bearing was
            # measurably correct (10 solids = 2 rings + 8 balls, both bores present) and was
            # binned only because the brief predicted 3 parts. Build and render it anyway;
            # a verdict you cannot inspect is a verdict you cannot correct.
            show_code = (rows[0]["code"] if accepted
                         else review.get("code") or review.get("t1_code") or "")
            stem = sid if accepted else f"{sid}-rejected"
            png = ""
            if show_code:
                if args.no_render and (mdir / f"{stem}.png").exists():
                    png = f"{stem}.png"
                else:
                    _, png = build_artifacts(show_code, tmp / run["_slug"] / stem, mdir, stem)
            if accepted and rows[0].get("system"):
                system_seen = system_seen or rows[0]["system"]

            body = [f'<h3>{html.escape(run["model"])}</h3>',
                    badge(rec.get("outcome", "")),
                    f'<span class="badge turns">{turns} turn{"s" if turns != 1 else ""}</span>']

            # Rejection reasons OPEN, not tucked behind a disclosure triangle: the reason is
            # the whole point of a failed row.
            if not accepted:
                for k, lbl in (("first_problem", "attempt 1 rejected"),
                               ("second_problem", "attempt 2 rejected")):
                    if rec.get(k):
                        body.append(f'<div class="why"><b>{lbl}:</b> '
                                    f'{html.escape(rec[k])}</div>')
                if rec.get("first_problem") and rec.get("first_problem") == rec.get("second_problem"):
                    body.append('<div class="why warn">both attempts got the IDENTICAL '
                                'complaint — the repair call could not satisfy it, so this '
                                'was almost certainly an impossible demand rather than a '
                                'fixable fault.</div>')

            if png:
                cap = "accepted geometry" if accepted else "geometry the gate REJECTED"
                body.append(f'<div style="margin-top:8px"><img src="{run["_slug"]}/{png}">'
                            f'<div class="cap">{cap} &middot; '
                            f'<a href="{run["_slug"]}/{stem}.step">{stem}.step</a> &middot; '
                            f'<a href="{run["_slug"]}/{stem}.py">source</a></div></div>')
            elif show_code:
                body.append('<div class="nogeo">code stored but would not build</div>')
            else:
                body.append('<div class="nogeo">no code retained for this attempt</div>')

            if accepted:
                r0 = rows[0]
                body.append(
                    f'<details><summary>prompt the model answered '
                    f'({html.escape(r0.get("prompt_kind", "?"))}, '
                    f'{len(r0.get("prompt", ""))} chars)</summary>'
                    f'<pre>{html.escape(r0.get("prompt", ""))}</pre></details>')
                body.append(f'<details><summary>code</summary>'
                            f'<pre>{html.escape(r0["code"])}</pre></details>')
                body.append(f'<details><summary>measured</summary>'
                            f'<pre>{html.escape(json.dumps(r0.get("verified") or {}, indent=1))}'
                            f'</pre></details>')
            else:
                if show_code:
                    body.append(f'<details><summary>rejected code</summary>'
                                f'<pre>{html.escape(show_code)}</pre></details>')
                facts = review.get("facts") or review.get("t1_facts") or {}
                if facts:
                    body.append(f'<details open><summary>what was actually measured</summary>'
                                f'<pre>{html.escape(json.dumps(facts, indent=1))}</pre></details>')
                if rec.get("brief_expected"):
                    body.append(f'<details><summary>what the brief PREDICTED (qwen3:8b)'
                                f'</summary><pre>'
                                f'{html.escape(json.dumps(rec["brief_expected"], indent=1))}'
                                f'</pre></details>')
            cols.append(f'<div class="col">{"".join(body)}</div>')

        parts_html.append(
            f'<div class="part"><div class="spec">{html.escape(sid)} — '
            f'{html.escape(first["spec"])}</div>'
            f'<div class="meta">tier {first.get("tier", "?")} · '
            f'{html.escape(str(first.get("group", "")))}</div>'
            f'<div class="cols">{"".join(cols)}</div></div>')

    sum_rows = "".join(
        f'<tr><td>{html.escape(r["model"])}</td>'
        f'<td>{sum(1 for x in r["per_spec"] if x.get("outcome","").startswith("accepted"))}/{r["specs"]}</td>'
        f'<td>{r["good"]}</td><td>{r["fail_fix"]}</td>'
        f'<td>{r["elapsed_s"] / 60:.0f} min</td></tr>' for r in runs)

    page = f"""<!doctype html><meta charset="utf-8">
<title>Teacher pilot review</title><style>{CSS}</style>
<h1>Teacher pilot — part-by-part review</h1>
<div class="sub">Every render, prompt and program below is regenerated from that run's own
stored code, so the two columns cannot be confused with each other.</div>
<table class="sum"><tr><th>model</th><th>accepted</th><th>good pairs</th>
<th>fail&rarr;fix pairs</th><th>wall clock</th></tr>{sum_rows}</table>
{stats_html(runs)}
{"".join(parts_html)}
<details><summary>the shared system prompt every codegen turn carried
({len(system_seen)} chars)</summary><pre>{html.escape(system_seen)}</pre></details>
"""
    idx = out / "index.html"
    idx.write_text(page)
    print(f"review page -> {idx}")
    print(f"parts: {len(ids)}   models: {', '.join(r['model'] for r in runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Triage the silver queue — parts the gate disputed but did not prove wrong.

The gate is a labeller, not a bin. A part that builds, is watertight, and has no structural
fault but trips one spec-text check is a QUESTION, not a verdict — and on 2026-07-30 five of
eleven such mechanisms turned out to be measurably correct. This tool puts a human in that
loop: it renders each disputed part, shows the check that objected, and records your call.

Your verdict does two things: `accept` moves the pair into the training set, and `gate-bug`
files the objection against the check that raised it, so the gate can be fixed rather than
argued with every run.

    python3 scripts/review_queue.py --list
    python3 scripts/review_queue.py --accept C01 --reject C12
    python3 scripts/review_queue.py --gate-bug C02 --note "140mm is the rod, not the envelope"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cad_v5.config import SFTPAIRS_FILE  # noqa: E402

QUEUE = SFTPAIRS_FILE.with_name("cad-sftpairs-review.jsonl")
DECISIONS = SFTPAIRS_FILE.with_name("cad-review-decisions.jsonl")
GATE_BUGS = SFTPAIRS_FILE.with_name("cad-gate-bugs.jsonl")
OUT = Path.home() / "CAD" / "review"


def load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    rows = []
    for line in QUEUE.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def decided() -> dict:
    out = {}
    if DECISIONS.exists():
        for line in DECISIONS.read_text().splitlines():
            try:
                d = json.loads(line)
                out[d["id"]] = d["verdict"]
            except Exception:
                continue
    return out


def render(row: dict, sid: str) -> str:
    """Build STEP + PNG so the part can be judged by eye as well as by numbers."""
    OUT.mkdir(parents=True, exist_ok=True)
    w = Path(tempfile.mkdtemp(prefix="review_"))
    (w / "s.py").write_text(row["code"])
    subprocess.run([sys.executable, str(HERE / "scripts" / "step"),
                    str(w / "s.py"), str(w / "o.step")], capture_output=True, timeout=300)
    if not (w / "o.step").exists():
        return ""
    (OUT / f"{sid}.step").write_bytes((w / "o.step").read_bytes())
    (OUT / f"{sid}.py").write_text(row["code"])
    try:
        subprocess.run([sys.executable, str(HERE / "scripts" / "render"),
                        str(w / "o.step"), str(w / "r.png")], capture_output=True, timeout=300)
        if (w / "r.png").exists():
            (OUT / f"{sid}.png").write_bytes((w / "r.png").read_bytes())
    except Exception:
        pass
    return str(OUT / f"{sid}.step")


def cmd_list(rows: list[dict], do_render: bool) -> None:
    done = decided()
    pending = [r for r in rows if r.get("teacher_spec_id") not in done]
    print(f"queue: {len(rows)} item(s), {len(pending)} awaiting your call")
    print(f"files: {OUT}\n")
    for r in rows:
        sid = r.get("teacher_spec_id", "?")
        mark = done.get(sid)
        print(f"{'─' * 74}")
        print(f"{sid}  [{mark or 'PENDING'}]  {r.get('code_model', '')}")
        print(f"  spec : {r['spec'][:160]}")
        for n in (r.get("spec_notes") or [])[:3]:
            print(f"  gate : {n[:150]}")
        f = r.get("facts") or {}
        print(f"  built: solids={f.get('solids')} bbox={f.get('bbox')} volume={f.get('volume')}")
        if do_render and not mark:
            p = render(r, sid)
            print(f"  view : {p or '(would not rebuild)'}")


def cmd_decide(rows: list[dict], verdict: str, ids: list[str], note: str) -> None:
    by_id = {r.get("teacher_spec_id"): r for r in rows}
    stamp = datetime.now(timezone.utc).isoformat()
    promoted = 0
    for sid in ids:
        row = by_id.get(sid)
        if not row:
            print(f"  {sid}: not in queue", file=sys.stderr)
            continue
        with DECISIONS.open("a") as f:
            f.write(json.dumps({"id": sid, "verdict": verdict, "note": note,
                                "spec": row["spec"], "timestamp": stamp}) + "\n")
        if verdict == "accept":
            # Promote into the training set. Marked so it is traceable to a human decision
            # rather than looking like something the gate passed on its own.
            with SFTPAIRS_FILE.open("a") as f:
                f.write(json.dumps({
                    "kind": "good", "source": "teacher-human-accepted",
                    "spec": row["spec"], "code": row["code"], "image": "",
                    "code_model": row.get("code_model"), "teacher_spec_id": sid,
                    "verified": {"facts": row.get("facts"),
                                 "gate_objected": row.get("spec_notes"),
                                 "accepted_by": "human review", "note": note},
                    "timestamp": stamp}) + "\n")
            promoted += 1
        if verdict == "gate-bug":
            # File the objection against the CHECK, so the gate gets fixed once instead of
            # being overruled every run.
            for n in (row.get("spec_notes") or ["(unspecified)"]):
                with GATE_BUGS.open("a") as f:
                    f.write(json.dumps({"check": n[:180], "spec": row["spec"],
                                        "id": sid, "note": note,
                                        "timestamp": stamp}) + "\n")
        print(f"  {sid}: {verdict}")
    if promoted:
        print(f"\npromoted {promoted} pair(s) into {SFTPAIRS_FILE.name}")


def queue_run(summary: Path, which: str) -> int:
    """Add a run's parts to the review queue so a HUMAN sees them — accepted ones included.

    The gate measures conformance, not function. On 2026-07-30 a user review of five
    gate-REJECTED mechanisms found four genuinely broken (rod through a piston crown, fins
    unattached, gears not meshing) — the gate had been right for the wrong reasons. The parts
    it ACCEPTS get no such scrutiny, and those are the ones that become training data. So
    review is a standard step, not an exception path.
    """
    d = json.loads(summary.read_text())
    seen = {r.get("teacher_spec_id") for r in load_queue()}
    added = 0
    with QUEUE.open("a") as f:
        for rec in d["per_spec"]:
            sid = rec["id"]
            acc = rec.get("outcome", "").startswith("accepted")
            if which == "accepted" and not acc:
                continue
            if which == "rejected" and acc:
                continue
            if sid in seen:
                continue
            rows = rec.get("rows") or []
            rev = rec.get("review") or {}
            code = rows[0]["code"] if rows else (rev.get("code") or rev.get("t1_code"))
            if not code:
                continue
            facts = (rows[0].get("verified") if rows else None) or rev.get("facts") or {}
            f.write(json.dumps({
                "tier": "gate-accepted" if acc else "silver",
                "spec": rec["spec"], "code": code,
                "spec_notes": [f"gate verdict: {rec.get('outcome')}"]
                              + (rev.get("spec_notes") or []),
                "facts": facts, "code_model": d.get("model"),
                "teacher_spec_id": sid, "prompt": (rows[0].get("prompt") if rows
                                                   else rev.get("t1_prompt", "")),
                "timestamp": d.get("ran_at")}) + "\n")
            added += 1
    print(f"queued {added} part(s) from {summary.name} ({which})")
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-run", default="", help="add a run summary's parts to the queue")
    ap.add_argument("--which", default="all", choices=["all", "accepted", "rejected"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="skip STEP/PNG export when listing")
    ap.add_argument("--accept", nargs="*", default=[])
    ap.add_argument("--reject", nargs="*", default=[])
    ap.add_argument("--gate-bug", nargs="*", default=[], dest="gate_bug")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.queue_run:
        queue_run(Path(args.queue_run), args.which)

    rows = load_queue()
    if not rows:
        print(f"review queue is empty ({QUEUE})")
        return 0
    for verdict, ids in (("accept", args.accept), ("reject", args.reject),
                         ("gate-bug", args.gate_bug)):
        if ids:
            cmd_decide(rows, verdict, ids, args.note)
    if args.list or not (args.accept or args.reject or args.gate_bug):
        cmd_list(rows, do_render=not args.no_render)
    return 0


if __name__ == "__main__":
    sys.exit(main())

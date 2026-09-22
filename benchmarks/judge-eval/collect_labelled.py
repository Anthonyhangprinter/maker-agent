#!/usr/bin/env python3
"""
Judge-eval Task 1: assemble the labelled set (program + spec + ground-truth verdict).

Two sources, both read-only:

1. The claude-sub-2026-09-19 card (MAIN checkout, not this worktree):
   ~/.openclaw/skills/cad-builder/benchmarks/results/card/claude-sub-2026-09-19/
   20 reference-scored public-suite specs x 4 writers (gemma-4-31b-sameprompt, claude-sonnet,
   claude-opus, claude-fable) = 80 programs, each scored against the suite's reference geometry
   with a Chamfer band. We use the "+repair" row as the row of record (repair could not move a
   *build* into a match without also changing code, and the file on disk after repair is what a
   judge would actually see), taking the repaired code file when repaired=true and the original
   otherwise. Programs that never built (ok=false) are excluded — a crash needs no judge.

2. Three known-wrong fixture rows from this worktree's
   tests/fixtures/harvest_agreement_fixtures.json: V064 (WRONG — a lip-cutter that shears the
   whole top off instead of just the rim), V066 T=0.2 (WRONG — a Z-axis cylinder makes a slot,
   not a round hole through the divider), V066 T=0.5 (WRONG, corrected 2026-09-22 — the
   cylinder is rotated onto the divider's thickness axis as intended, but the floor measures
   4.5mm where the 140x90x60mm/3mm-wall spec calls for 3mm, caught by hand audit).

Ground truth: band == "match" -> CORRECT. Built but any other band (valid/near_miss/fail) ->
WRONG. The three fixture rows carry their verdict directly from the task description (they
have no `band`, since they came from a live-harvest agreement check, not the card scorer).

Writes labelled.json: a list of records with spec, code, writer, suite/id, ground truth.
"""
import json
import sys
from pathlib import Path

MAIN_CARD = Path.home() / ".openclaw/skills/cad-builder/benchmarks/results/card/claude-sub-2026-09-19"
FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/harvest_agreement_fixtures.json"
OUT = Path(__file__).resolve().parent / "labelled.json"

WRITERS = ["gemma-4-31b-sameprompt", "claude-sonnet", "claude-opus", "claude-fable"]


def load_rows_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompts_index():
    prompts = {}
    for p in (MAIN_CARD / "prompts").glob("*.json"):
        if p.name == "INDEX.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        prompts[(d["suite"], d["id"])] = d["spec"]
    return prompts


def main():
    rows = load_rows_jsonl(MAIN_CARD / "rows.jsonl")
    prompts = load_prompts_index()

    # index +repair rows by (writer, suite, id)
    repair_rows = {}
    for r in rows:
        if r["arm"].endswith("+repair"):
            writer = r["arm"][: -len("+repair")]
            repair_rows[(writer, r["suite"], r["id"])] = r

    labelled = []
    excluded_no_build = []

    for writer in WRITERS:
        for (w, suite, sid), r in repair_rows.items():
            if w != writer:
                continue
            if not r.get("ok"):
                excluded_no_build.append({"writer": writer, "suite": suite, "id": sid, "reason": "ok=false (never built)"})
                continue
            band = r.get("band")
            if band is None:
                excluded_no_build.append({"writer": writer, "suite": suite, "id": sid, "reason": f"band=None (gate veto? gate_hard={r.get('gate_hard')})"})
                continue
            repaired = bool(r.get("repaired"))
            subdir = "repairs" if repaired else "code"
            code_path = MAIN_CARD / subdir / writer / f"{suite}__{sid}.py"
            if not code_path.exists():
                excluded_no_build.append({"writer": writer, "suite": suite, "id": sid, "reason": f"code file missing: {code_path}"})
                continue
            spec = prompts.get((suite, sid))
            if spec is None:
                excluded_no_build.append({"writer": writer, "suite": suite, "id": sid, "reason": "no prompt/spec found"})
                continue
            truth = "correct" if band == "match" else "wrong"
            labelled.append({
                "program_id": f"{writer}::{suite}__{sid}",
                "writer": writer,
                "writer_family": "gemma" if writer.startswith("gemma") else "claude",
                "suite": suite,
                "spec_id": sid,
                "spec": spec,
                "code_path": str(code_path),
                "band": band,
                "truth": truth,
                "source": "claude-sub-2026-09-19",
                "repaired": repaired,
            })

    # ---- Fixture rows (V064 / V066) ----
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    fixture_truth = {
        "v064_wrong_sheared_lip": "wrong",
        "v066_t02_wrong_slot": "wrong",
        "v066_t05_floor_4p5mm": "wrong",
    }
    fixture_dir = Path(__file__).resolve().parent / "fixture_code"
    fixture_dir.mkdir(exist_ok=True)
    for key, truth in fixture_truth.items():
        rec = fixtures[key]
        code_path = fixture_dir / f"{key}.py"
        code_path.write_text(rec["code"], encoding="utf-8")
        labelled.append({
            "program_id": f"fixture::{key}",
            "writer": rec.get("arm", "gemma-4-31b"),
            "writer_family": "gemma",
            "suite": "teacher-batch2-fixture",
            "spec_id": rec.get("spec_id", key),
            "spec": rec["spec"],
            "code_path": str(code_path),
            "band": rec.get("band"),
            "truth": truth,
            "source": "harvest_agreement_fixtures.json",
            "repaired": False,
            "fixture_key": key,
            "fixture_facts": rec.get("facts"),
            "temperature": rec.get("temperature"),
        })

    OUT.write_text(json.dumps(labelled, indent=2), encoding="utf-8")
    n_correct = sum(1 for r in labelled if r["truth"] == "correct")
    n_wrong = sum(1 for r in labelled if r["truth"] == "wrong")
    print(f"Labelled set: {len(labelled)} programs ({n_correct} correct / {n_wrong} wrong)")
    print(f"Excluded (did not build / no gate verdict): {len(excluded_no_build)}")
    for e in excluded_no_build:
        print("  ", e)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

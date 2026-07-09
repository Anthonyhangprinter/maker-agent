#!/usr/bin/env python3
"""behavioral_n2.py — live behavioral check of the N2 ambiguity gate (triage_ambiguity()).

NOT a pytest file — a runnable script. Calls qwen3:8b via Ollama (~6 short, schema-constrained
calls; cheap). Feeds 3 deliberately vague prompts and 3 specific tier-1 specs taken verbatim from
benchmarks/text-to-cad/specs.json, and prints a table of prompt -> questions (or PASS = no
questions). Expected: every vague prompt gets >=1 question, every specific prompt gets [].

Run:  python3 tests/behavioral_n2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cad_agent_v4 as v4  # noqa: E402

VAGUE = [
    "a wall mount for my router",
    "a box for my stuff",
    "something to hold my pens",
]


def _specific_from_specs() -> list[str]:
    specs_path = Path(__file__).resolve().parent.parent / "benchmarks" / "text-to-cad" / "specs.json"
    data = json.loads(specs_path.read_text())
    tier1 = [b["spec"] for b in data["benchmarks"] if b["tier"] == 1]
    return tier1[:3]


def main() -> int:
    specific = _specific_from_specs()
    rows = [("vague", p) for p in VAGUE] + [("specific (tier-1)", p) for p in specific]

    print(f"{'kind':<18} {'prompt':<70} {'result'}")
    print("-" * 130)
    all_ok = True
    for kind, prompt in rows:
        questions = v4.triage_ambiguity(prompt)
        expect_questions = (kind == "vague")
        ok = (bool(questions) == expect_questions)
        all_ok = all_ok and ok
        short = (prompt[:67] + "...") if len(prompt) > 70 else prompt
        if questions:
            result = "QUESTIONS:\n" + "\n".join(f"      {i}. {q}" for i, q in enumerate(questions, 1))
        else:
            result = "PASS (no questions)"
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] {kind:<13} {short:<70} {result}")

    print("-" * 130)
    print("ALL OK" if all_ok else "SOME MISMATCHES — see FAIL rows above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

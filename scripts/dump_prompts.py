#!/usr/bin/env python3
"""Dump the EXACT system + user prompt strings the local engine would send for a first-turn
one-shot build with retrieval on, for a chosen set of (suite, id) specs, with NO model call.

This is the same trick scripts/compile_sft.py's reconstruct_prompt() uses to recover a
historical prompt without re-calling a model: stub engine._ollama to a no-op, run the real
codegen function, and read back what it stashed into engine._LAST_PROMPT just before it would
have sent the request. engine.retrieval_notes_for(spec) DOES hit the local CPU embed server on
:8089 for semantic few-shot retrieval, that is not a model call in the sense this harness
cares about (no GPU, no coder/critic inference), and is exactly what production one-shot does
before it calls the coder.

The captured prompt is for engine.generate_code_raw(spec, notes) specifically: the brief-less,
first-turn codegen call scripts/fluid_gen.py's `build` command makes when the spec is not a
domain-helper shortcut and (for a text-only build) not ambiguous enough to trigger the
expansion rung. A vague spec that would trigger engine.triage_ambiguity()/expand_spec() in a
live one-shot build is NOT expanded here, this script captures only the codegen prompt for
the spec text as written in the suite's specs.json, which is what an outside model is being
asked to answer against.

Acquires no build lock and starts/stops no server: nothing here executes code or talks to a
coder model.

    python3 scripts/dump_prompts.py --ids FILE --out DIR

FILE is a JSON array of [suite, id] pairs, e.g. [["cadprompt", "cp-00000007"], ...].
Writes DIR/<suite>__<id>.json = {"suite", "id", "spec", "system", "prompt"} per pair, plus
DIR/INDEX.json (the ordered list of pairs actually written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import run_card             # noqa: E402  (load_suite, same suite/spec resolution run_card uses)
import cad_engine as engine  # noqa: E402


def _capture_prompt(call) -> dict:
    """Run an engine codegen function with the model call stubbed out and lift the exact
    (kind, system, prompt) it stashed, the same trick scripts/compile_sft.py's
    reconstruct_prompt() uses. No network, no model, no GPU."""
    orig = engine._ollama
    engine._ollama = lambda *a, **k: ""
    try:
        call()
    finally:
        engine._ollama = orig
    return dict(engine._LAST_PROMPT)


def prompt_for_spec(spec_text: str) -> dict:
    """The exact (system, prompt) engine.generate_code_raw() would send for `spec_text`,
    retrieval on, first turn, no brief, the same call scripts/fluid_gen.py's `build` command
    makes for a non-helper, non-image, already-parameterized spec."""
    notes = engine.retrieval_notes_for(spec_text, use_fewshots=True)
    p = _capture_prompt(lambda: engine.generate_code_raw(spec_text, notes))
    return {"system": p["system"], "prompt": p["prompt"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="JSON file: a list of [suite, id] pairs")
    ap.add_argument("--out", required=True, help="directory to write <suite>__<id>.json + INDEX.json into")
    args = ap.parse_args()

    pairs = [tuple(x) for x in json.loads(Path(args.ids).read_text())]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    suite_cache: dict[str, tuple[list[dict], dict]] = {}
    index: list[list[str]] = []
    for suite, sid in pairs:
        if suite not in suite_cache:
            suite_cache[suite] = run_card.load_suite(suite)
        specs, _acc = suite_cache[suite]
        spec = next((s for s in specs if s["id"] == sid), None)
        if spec is None:
            print(f"WARNING: {suite}/{sid} not found in benchmarks/{suite}/specs.json, skipped",
                  file=sys.stderr)
            continue
        p = prompt_for_spec(spec["spec"])
        row = {"suite": suite, "id": sid, "spec": spec["spec"],
              "system": p["system"], "prompt": p["prompt"]}
        (out_dir / f"{suite}__{sid}.json").write_text(json.dumps(row, indent=2) + "\n")
        digest = hashlib.sha256(p["prompt"].encode()).hexdigest()
        print(f"{suite}/{sid}: prompt sha256={digest}")
        index.append([suite, sid])
    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(index)} of {len(pairs)} requested prompt(s) to {out_dir}")


if __name__ == "__main__":
    main()

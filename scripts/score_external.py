#!/usr/bin/env python3
"""Score CAD code written by an OUTSIDE model (e.g. Claude, in a Claude Code session)
exactly the way scripts/run_card.py scores the local arm's one-shot builds, no GPU, no
model call of any kind, no network beyond the local CPU embed server that
scripts/dump_prompts.py already used to build the prompt (this script runs no prompts at
all: the code is already written).

Reuses the SAME runners one-shot uses, not a reimplementation of them:
  - fluid_gen._materialize(code, build_dir, spec), the exact function scripts/fluid_gen.py's
    `build` command calls for its single codegen turn: engine.run_step (scripts/step, CPU
    build123d execution), engine.run_inspect (scripts/inspect), engine.parse_facts, then
    engine.verify_expected(facts, fluid_gen.expected_for(spec), spec=spec) for the
    hard/[spec]/advisory split. It also renders a PNG and exports an STL, same as one-shot,
    though neither is needed for scoring.
  - run_card.geometry_of / run_card.criteria_of / run_card.score_acceptance / run_card.band_of
    for acceptance scoring and the Chamfer-band verdict against a suite's reference_stl.

Called with NO salvage/repair turn (`--arm NAME`), a row is exactly one _materialize() call:
"salvaged": false, "no_salvage": true. A missing code file or a script that raises is still a
row (ok: false, the error recorded), never a skipped spec, the reader must see every id that
was supposed to be scored.

Repair turns (owner-approved scope addition, the Claude arms MAY use one repair attempt, and
both numbers, before and after, are wanted):
  --emit-repairs writes the EXACT system+prompt string fluid_gen's own one repair attempt
    would send for a spec whose first attempt crashed or gated dirty: engine.diagnose()'s hint
    + engine.revise_script() for a crash (fluid's crash-salvage branch), or the joined hard+
    [spec] findings + engine.revise_script() for a gate-repair (fluid's gate-repair branch).
    Exactly one prompt per qualifying spec, matching fluid_gen's own "at most one repair
    attempt per build" rule, never both branches for the same spec.
  Scoring `--arm NAME+repair --code-dir DIR --repair-dir RDIR` re-scores the repaired code the
    same way and APPLIES A KEEP RULE per spec: the repair replaces the first attempt only if it
    measurably improves (fewer hard findings; or the same hard findings and fewer [spec]
    findings; or it now runs where the first attempt crashed), else the first-attempt row is
    carried forward unchanged. This is a more precise, hard-then-spec hierarchical rule than
    fluid_gen's own literal comparison (`len(hard)+len(spec) < len(old hard)+len(old spec)`,
    a combined count that a hard-finding improvement can lose to a spec-finding regression) ,
    the owner asked for the hierarchical version here, so the KEEP decision is intentionally
    NOT bit-identical to what a live fluid_gen build would decide on ties between hard and
    [spec] counts. The two REPAIR PROMPT STRINGS themselves (crash / gate) are byte-identical
    to what fluid_gen would send, via the same diagnose()/revise_script() calls it uses.

    scripts/score_external.py --arm NAME --code-dir DIR --ids FILE --out CARD_DIR [--tokens-json FILE]
    scripts/score_external.py --seed-baseline ARM --from ROWS --ids FILE --out CARD_DIR
    scripts/score_external.py --emit-repairs --arm NAME --code-dir DIR --ids FILE --out CARD_DIR
    scripts/score_external.py --arm NAME+repair --code-dir DIR --repair-dir RDIR --ids FILE --out CARD_DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import run_card                       # noqa: E402  (load_suite, geometry_of, criteria_of, band_of, score_acceptance, BENCH)
import fluid_gen                      # noqa: E402  (_materialize, the exact one-shot runner path)
import cad_engine as engine           # noqa: E402  (_ollama stub trick, revise_script)
from cad_v5.diagnose import diagnose  # noqa: E402  (B3 failure taxonomy, same hint fluid uses)

# The key set run_card.run_row() always produces (see that function's return statement).
# A score_external row must carry every one of these, so lift_report.py / card_report.py read
# it exactly like a local-arm row.
RUN_ROW_KEYS = {
    "arm", "suite", "id", "tier", "ok", "mode", "subset", "candidates",
    "gate_hard", "gate_spec", "acc_passed", "acc_total", "band", "has_ref", "helper",
    "wall_s", "tokens_out", "build_dir", "error", "stderr_tail",
}
# Added on top, to say WHERE the code came from and whether a repair turn was involved ,
# nothing here shadows a run_card key.
EXTRA_ROW_KEYS = {"code_model", "salvaged", "no_salvage"}

MODE = "oneshot"
SUBSET = "claude-sub"


# ── suite lookup, memoized (mirrors run_card's own load_suite call pattern) ────────────

_SUITE_CACHE: dict[str, tuple[list[dict], dict]] = {}


def _suite(suite: str) -> tuple[list[dict], dict]:
    if suite not in _SUITE_CACHE:
        _SUITE_CACHE[suite] = run_card.load_suite(suite)
    return _SUITE_CACHE[suite]


def spec_and_crit(suite: str, sid: str) -> tuple[dict | None, dict | None]:
    specs, acc = _suite(suite)
    spec = next((s for s in specs if s["id"] == sid), None)
    return spec, (acc.get(sid) if spec else None)


def load_ids(path: str) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text())
    return [tuple(x) for x in data]


# ── rows.jsonl I/O (resumable by (arm, suite, id), same contract as run_card) ──────────


def resume_done(rows_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["arm"], r["suite"], r["id"]))
    return done


def append_row(rows_path: Path, row: dict) -> None:
    with rows_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_rows(rows_path: Path) -> list[dict]:
    if not rows_path.exists():
        return []
    return [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]


# ── the one-shot scoring path itself ────────────────────────────────────────────────────


def _build_row(arm: str, suite: str, spec: dict, crit: dict | None, *, ok: bool,
               gate_hard: int, gate_spec: int, acc_passed: int, acc_total: int,
               band: str | None, wall_s: float, tokens_out: int | None, build_dir: str,
               error: str | None, stderr_tail: str, salvaged: bool = False,
               no_salvage: bool = True, code_model: str | None = None) -> dict:
    return {
        "arm": arm, "suite": suite, "id": spec["id"], "tier": spec.get("tier", 0),
        "ok": bool(ok), "mode": MODE, "subset": SUBSET, "candidates": 1,
        "gate_hard": gate_hard, "gate_spec": gate_spec,
        "acc_passed": acc_passed, "acc_total": acc_total, "band": band,
        "has_ref": bool((crit or {}).get("reference_stl")), "helper": False,
        "wall_s": round(wall_s, 1), "tokens_out": tokens_out,
        "build_dir": build_dir, "error": error,
        "stderr_tail": (stderr_tail or "")[-300:] if not ok else "",
        "code_model": code_model or f"claude-code-sub/{arm}",
        "salvaged": salvaged, "no_salvage": no_salvage,
    }


def score_code(code_path: Path, build_dir: Path, spec_text: str) -> tuple[dict, float]:
    """Run one script through fluid_gen's exact one-shot materializer: run_step -> run_inspect
    -> parse_facts -> verify_expected(facts, expected_for(spec), spec=spec). No salvage, no
    repair, fluid_gen._materialize (not _materialize_with_salvage) is exactly that: one
    execute-inspect-gate pass and nothing else."""
    build_dir.mkdir(parents=True, exist_ok=True)
    code = code_path.read_text()
    t0 = time.monotonic()
    m = fluid_gen._materialize(code, build_dir, spec_text)
    return m, time.monotonic() - t0


def finish_row(arm: str, suite: str, spec: dict, crit: dict | None, m: dict, wall: float,
              tokens_out: int | None, build_dir: Path, **extra) -> dict:
    facts = m.get("facts") or {}
    # Same non-bool-safe formula run_row uses: `bool(x) and y` returns y verbatim, so wrap it.
    ok = bool(bool(m.get("error") is None) and facts.get("solids", 1))
    gate_hard = len(m.get("gate_hard") or [])
    gate_spec = len(m.get("gate_spec") or [])
    geom = run_card.geometry_of({"facts": facts}, "oneshot") if ok else {}
    acc = run_card.score_acceptance(geom, run_card.criteria_of(crit))
    band = None
    ref = (crit or {}).get("reference_stl")
    step = build_dir / "build.step"
    if ok and ref and step.exists():
        try:
            band = run_card.band_of(step, run_card.BENCH / suite / ref, crit)
        except Exception:
            band = "fail"
    return _build_row(arm, suite, spec, crit, ok=ok, gate_hard=gate_hard, gate_spec=gate_spec,
                      acc_passed=acc.get("passed", 0), acc_total=acc.get("total", 0), band=band,
                      wall_s=wall, tokens_out=tokens_out, build_dir=str(build_dir),
                      error=m.get("error"), stderr_tail=m.get("error") or "", **extra)


def missing_row(arm: str, suite: str, spec: dict, crit: dict | None, tokens_out: int | None,
                reason: str) -> dict:
    """A missing file or an unreadable script: still a row, never a skipped spec."""
    acc = run_card.score_acceptance({}, run_card.criteria_of(crit))
    return _build_row(arm, suite, spec, crit, ok=False, gate_hard=0, gate_spec=0,
                      acc_passed=acc.get("passed", 0), acc_total=acc.get("total", 0), band=None,
                      wall_s=0.0, tokens_out=tokens_out, build_dir="", error=reason,
                      stderr_tail=reason)


def cmd_score(args) -> None:
    ids = load_ids(args.ids)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = resume_done(rows_path)
    tokens_map = json.loads(Path(args.tokens_json).read_text()) if args.tokens_json else {}
    code_dir = Path(args.code_dir)
    builds_root = out / "builds" / args.arm
    scored = skipped = 0
    for suite, sid in ids:
        if (args.arm, suite, sid) in done:
            continue
        spec, crit = spec_and_crit(suite, sid)
        if spec is None:
            print(f"WARNING: {suite}/{sid} not found in benchmarks/{suite}/specs.json, skipped")
            skipped += 1
            continue
        key = f"{suite}__{sid}"
        tk = tokens_map.get(key)
        code_path = code_dir / f"{key}.py"
        if not code_path.is_file():
            row = missing_row(args.arm, suite, spec, crit, tk, f"missing code file: {code_path}")
        else:
            build_dir = builds_root / key
            try:
                m, wall = score_code(code_path, build_dir, spec["spec"])
            except Exception as e:
                row = missing_row(args.arm, suite, spec, crit, tk,
                                  f"scorer raised while scoring {code_path}: {e}")
                append_row(rows_path, row)
                scored += 1
                continue
            row = finish_row(args.arm, suite, spec, crit, m, wall, tk, build_dir)
        append_row(rows_path, row)
        scored += 1
        print(f"  {suite}/{sid}: ok={row['ok']} hard={row['gate_hard']} spec={row['gate_spec']} "
              f"band={row['band']} {row['wall_s']}s")
    print(f"scored {scored} new row(s) ({len(done)} already done, {skipped} spec(s) not found) "
          f"-> {rows_path}")


# ── seed-baseline: copy a local arm's own rows for the same ids, unchanged ─────────────


def cmd_seed_baseline(args) -> None:
    src_rows = read_rows(Path(args.frm))
    ids = set(load_ids(args.ids))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = resume_done(rows_path)
    matched = [r for r in src_rows
              if r.get("arm") == args.seed_baseline and (r["suite"], r["id"]) in ids]
    written = 0
    for r in matched:
        key = (r["arm"], r["suite"], r["id"])
        if key in done:
            continue
        append_row(rows_path, r)
        done.add(key)
        written += 1
    covered = {(r["suite"], r["id"]) for r in matched}
    missing = sorted(ids - covered)
    print(f"seeded {written} baseline row(s) for arm {args.seed_baseline!r} from {args.frm} "
          f"-> {rows_path}")
    if missing:
        print(f"WARNING: {len(missing)} of {len(ids)} id(s) have NO row for arm "
              f"{args.seed_baseline!r} in {args.frm}: {missing}")


# ── emit-repairs: the exact repair prompt fluid_gen's one repair turn would send ───────


def _capture_prompt(call) -> dict:
    """Run an engine call with the model stubbed out and lift the (kind, system, prompt) it
    stashed, the same trick scripts/compile_sft.py's reconstruct_prompt uses. No network, no
    model, no GPU: `call()` runs to the point where it would send the request and stops there."""
    orig = engine._ollama
    engine._ollama = lambda *a, **k: ""
    try:
        call()
    finally:
        engine._ollama = orig
    return dict(engine._LAST_PROMPT)


def repair_prompt_for(spec_text: str, code: str, m: dict) -> tuple[dict, str] | None:
    """The (prompt, why) fluid_gen's ONE repair attempt would produce for this first-attempt
    result, or None if the build was already clean (no repair fires). Mirrors
    fluid_gen._materialize_with_salvage exactly: a crash gets diagnose()'s hint, a gate-dirty
    but running build gets the joined hard+[spec] findings, never both for the same spec,
    same as fluid_gen never falling through from one branch to the other."""
    if m.get("error"):
        _, hint = diagnose(m["error"])
        problem = m["error"] + (f"\nRepair hint: {hint}" if hint else "")
        why = f"crash: {m['error'].strip()[:160]}"
    else:
        findings = list(m.get("gate_hard") or []) + list(m.get("gate_spec") or [])
        if not findings:
            return None
        problem = ("Deterministic measurements of the built solid contradict the request "
                  "(authoritative, they measure the actual geometry):\n- "
                  + "\n- ".join(findings))
        why = (f"gate: {len(m.get('gate_hard') or [])} hard finding(s), "
              f"{len(m.get('gate_spec') or [])} [spec] finding(s)")
    prompt = _capture_prompt(lambda: engine.revise_script(spec_text, code, problem))
    return prompt, why


def cmd_emit_repairs(args) -> None:
    ids = load_ids(args.ids)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    repair_dir = out / "repair_prompts" / args.arm
    repair_dir.mkdir(parents=True, exist_ok=True)
    code_dir = Path(args.code_dir)
    written = clean = missing = 0
    for suite, sid in ids:
        spec, crit = spec_and_crit(suite, sid)
        if spec is None:
            continue
        code_path = code_dir / f"{suite}__{sid}.py"
        if not code_path.is_file():
            # No first-attempt code to revise, fluid_gen's own repair turn always has a
            # `code` argument (the script that just ran); with nothing written there is
            # nothing to hand revise_script(), so no repair prompt can be reproduced here.
            missing += 1
            continue
        code = code_path.read_text()
        scratch = Path(tempfile.mkdtemp(prefix="score-ext-emit-"))
        try:
            m, _ = score_code(code_path, scratch, spec["spec"])
            result = repair_prompt_for(spec["spec"], code, m)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if result is None:
            clean += 1
            continue
        prompt, why = result
        row = {"suite": suite, "id": sid, "spec": spec["spec"], "system": prompt["system"],
              "prompt": prompt["prompt"], "why": why}
        (repair_dir / f"{suite}__{sid}.json").write_text(json.dumps(row, indent=2) + "\n")
        written += 1
    print(f"wrote {written} repair prompt(s) to {repair_dir} "
          f"({clean} already clean, {missing} with no first-attempt code)")


# ── scoring the repaired code + the keep rule ──────────────────────────────────────────


def improves(first_crashed: bool, first_hard: int, first_spec: int,
            repaired_error: str | None, repaired_hard: int, repaired_spec: int) -> bool:
    """The owner's hierarchical keep rule: fewer hard findings; or the same hard findings and
    fewer [spec] findings; or it now runs where the first attempt crashed. Deliberately NOT
    fluid_gen's own literal rule (`len(hard)+len(spec) < len(old hard)+len(old spec)`, a
    combined count where a hard-finding win can be cancelled by a spec-finding loss), see the
    module docstring."""
    if first_crashed:
        return repaired_error is None
    if repaired_error is not None:
        return False
    if repaired_hard < first_hard:
        return True
    return repaired_hard == first_hard and repaired_spec < first_spec


def _carry_forward(arm: str, base_row: dict, tokens_out: int | None) -> dict:
    row = dict(base_row)
    row["arm"] = arm
    row["code_model"] = f"claude-code-sub/{arm}"
    row["salvaged"] = False
    if tokens_out is not None:
        row["tokens_out"] = tokens_out
    return row


def cmd_score_repair(args) -> None:
    base_arm = args.arm[:-len("+repair")] if args.arm.endswith("+repair") else args.arm
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    base_rows = {(r["suite"], r["id"]): r for r in read_rows(rows_path) if r["arm"] == base_arm}
    done = resume_done(rows_path)
    tokens_map = json.loads(Path(args.tokens_json).read_text()) if args.tokens_json else {}
    repair_dir = Path(args.repair_dir)
    builds_root = out / "builds" / args.arm
    scored = missing_base = 0
    for suite, sid in load_ids(args.ids):
        if (args.arm, suite, sid) in done:
            continue
        spec, crit = spec_and_crit(suite, sid)
        if spec is None:
            continue
        base_row = base_rows.get((suite, sid))
        if base_row is None:
            print(f"WARNING: no first-attempt row for arm {base_arm!r} on {suite}/{sid} "
                  f"in {rows_path}; run --arm {base_arm} first. Skipping.")
            missing_base += 1
            continue
        key = f"{suite}__{sid}"
        tk = tokens_map.get(key)
        repair_path = repair_dir / f"{key}.py"
        if not repair_path.is_file():
            row = _carry_forward(args.arm, base_row, tk)
            row["repaired"] = False
        else:
            build_dir = builds_root / key
            m, wall = score_code(repair_path, build_dir, spec["spec"])
            repaired_row = finish_row(args.arm, suite, spec, crit, m, wall, tk, build_dir,
                                      no_salvage=False)
            if improves(base_row.get("error") is not None,
                       base_row.get("gate_hard", 0), base_row.get("gate_spec", 0),
                       repaired_row["error"], repaired_row["gate_hard"], repaired_row["gate_spec"]):
                row = repaired_row
                row["repaired"] = True
            else:
                row = _carry_forward(args.arm, base_row, tk)
                row["repaired"] = False
        row["no_salvage"] = False
        append_row(rows_path, row)
        scored += 1
        print(f"  {suite}/{sid}: repaired={row['repaired']} ok={row['ok']} "
              f"hard={row['gate_hard']} spec={row['gate_spec']}")
    print(f"scored {scored} +repair row(s) ({missing_base} missing a first-attempt row) "
          f"-> {rows_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="")
    ap.add_argument("--code-dir", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tokens-json", default="")
    ap.add_argument("--seed-baseline", default="", metavar="ARM")
    ap.add_argument("--from", dest="frm", default="", metavar="ROWS")
    ap.add_argument("--emit-repairs", action="store_true")
    ap.add_argument("--repair-dir", default="")
    args = ap.parse_args()

    if args.seed_baseline:
        if not (args.frm and args.ids and args.out):
            ap.error("--seed-baseline needs --from, --ids and --out")
        cmd_seed_baseline(args)
    elif args.emit_repairs:
        if not (args.arm and args.code_dir and args.ids and args.out):
            ap.error("--emit-repairs needs --arm, --code-dir, --ids and --out")
        cmd_emit_repairs(args)
    elif args.repair_dir:
        if not (args.arm and args.ids and args.out):
            ap.error("scoring a repair pass needs --arm, --repair-dir, --ids and --out")
        cmd_score_repair(args)
    else:
        if not (args.arm and args.code_dir and args.ids and args.out):
            ap.error("scoring needs --arm, --code-dir, --ids and --out")
        cmd_score(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""lab/specgen.py -- local spec generation on the maker arm (Phase 3 Task 2).

Writes new CAD part specs by prompting the strong-rung coder (engine.CODE_MODEL_STRONG,
the maker arm on :8088 when cad.json's maker.enabled, else the resident on :8086) with the
same family/tier structure scripts/teacher_specgen.py used for its cloud teacher calls --
but entirely local. This module never imports or calls cad_engine._cloud_chat, and never
touches Ollama (retired 2026-09-19; cad_engine._ollama raises on anything but a local:/
cloud/ model tag).

Generated specs are admitted through lab.specbank.add_items(), which applies the same
contamination refusal (exact card-suite key + per-suite-unique-slug) and in-bank
duplicate check that import-teacher uses -- a spec that collides is silently dropped,
counted, never retried.

  python3 lab/specgen.py --once --group plate --n 20         # one family, one call (smoke)
  python3 lab/specgen.py --total 2500 --target-tier34 0.45   # full run (long; controller-launched)

Every invocation ends (success or failure) by restoring the resident and re-applying the
standing maker lock-in via scripts/arms.py -- see _relock_maker() below for why a bare
`arms.py restore` alone is not enough.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import cad_engine as engine  # noqa: E402
from lab import specbank  # noqa: E402

ARMS_PY = HERE / "scripts" / "arms.py"
LOCK_ALIAS = "gemma-4-31b"   # the Phase 1 lock-in this module must never leave disturbed

# ---------------------------------------------------------------------------
# Families and tier guidance -- copied from scripts/teacher_specgen.py (2026-09-19: that
# file's own calls go through cad_engine._cloud_chat/cloud_config, which is forbidden in
# lab/; only the family tables and the system prompt travel here, retargeted below to the
# local engine._ollama call).
# ---------------------------------------------------------------------------
FAMILIES = [
    ("plate",     26, "tier 1-2: flat plates with holes, slots, chamfered edges, counterbores"),
    ("bracket",   26, "tier 2: L/T/U brackets, gussets, mounting tabs with bolt holes"),
    ("enclosure", 26, "tier 2-3: hollow boxes/cases with wall thickness, lids, bosses, cutouts"),
    ("shaft",     24, "tier 2-3: stepped shafts, keyways, circlip grooves, flats, cross-holes"),
    ("flange",    20, "tier 2-3: pipe/mounting flanges with bolt circles, raised faces, hubs"),
    ("surface",   34, "tier 3: lofted/swept/revolved forms, fillets, drafted walls, shells"),
    ("pattern",   20, "tier 2-3: polar/linear feature patterns, grids of holes, vent slots"),
    ("assembly",  20, "tier 3: 2-3 part assemblies as separate solids that visibly mate"),
]

HARD_FAMILIES = [
    ("engine",    30, "tier 3: ONE body composing 4+ features -- finned cylinders with a "
                      "central bore, head flange and stud holes; cooling-jacketed sleeves; "
                      "ribbed exhaust stubs with mounting flanges"),
    ("housing",   30, "tier 3: ONE body composing 4+ features -- bearing housings with bore, "
                      "bolt flange, ribs and grease port; gearbox end covers with recesses, "
                      "bosses and seal grooves; motor mounts with slots, gussets and pads"),
    ("machined",  30, "tier 3: ONE body composing 4+ features -- brackets/blocks combining "
                      "angled faces, counterbored holes, keyed bores, T-slots, dovetails, "
                      "chamfered pockets, cross-drilled passages"),
    ("manifold",  25, "tier 3: ONE body -- flanged manifolds/elbows/tees with through "
                      "passages, bolt circles on each port face, wall thickness stated"),
    ("mechanism", 40, "tier 4: 2-4 separate solids that genuinely assemble, ALWAYS "
                      "including the fastening elements as their own solids (clevis pins, "
                      "wrist pins, pivot pins, shoulder bolts) sized to their holes with "
                      "running clearance; correct centre distances and phasing"),
]

_TIER_BY_GROUP = {g: 3 for g, _, _ in HARD_FAMILIES}
_TIER_BY_GROUP["mechanism"] = 4
_TIER_BY_GROUP.update({
    "plate": 1, "bracket": 2, "enclosure": 2, "shaft": 2, "flange": 2,
    "surface": 3, "pattern": 2, "assembly": 3,
})

FAMILY_GUIDANCE = {g: guide for g, _, guide in (*FAMILIES, *HARD_FAMILIES)}
FAMILY_N = {g: n for g, n, _ in (*FAMILIES, *HARD_FAMILIES)}
ALL_GROUPS = [g for g, _, _ in FAMILIES] + [g for g, _, _ in HARD_FAMILIES]

_SYSTEM = """\
You write CAD part specifications for a text-to-CAD training corpus. Each spec is one
plain-English sentence a mechanical engineer might type, describing ONE buildable part
(or small assembly when asked) with CONCRETE millimetre dimensions for every major feature.

Rules:
- Every spec self-consistent and physically buildable; features must fit inside the part.
- Vary dimensions, feature counts, and phrasing across specs -- no two alike.
- Do NOT phrase anything as a bare catalog part name (avoid leading with exactly
  "spur gear ...", "hex bolt ...", "W-section I-beam ..." -- describe the geometry instead).
- Return ONLY a JSON array of objects: [{"tier": <1-4>, "spec": "<sentence>"}, ...].
  No markdown fences, no commentary."""

_REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be parsed as a JSON array. Reply again with ONLY "
    "the JSON array, no markdown fences, no commentary, no explanation.")


def _parse_json_array(raw: str) -> list[dict]:
    raw = re.sub(r"^```(json)?\s*\n?", "", raw.strip())
    raw = raw.rstrip("`").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array in reply: {raw[:120]!r}")
    return json.loads(raw[start:end + 1])


def _seed_specs(group: str, n: int = 5) -> list[str]:
    """Up to `n` random bank specs from the same group as style exemplars. Falls back to
    any teacher-suite spec when the group has no bank rows yet (a brand-new group on an
    empty bank), and to an empty seed block (the prompt says so) when the bank itself is
    empty -- never crashes for want of examples."""
    bank = specbank.load_bank()
    pool = [r["spec"] for r in bank if r.get("group") == group]
    if not pool:
        pool = [r["spec"] for r in bank if r.get("source") == "teacher-suite"]
    if not pool:
        return []
    random.shuffle(pool)
    return pool[:n]


def gen_family(group: str, n: int, guidance: str, seeds: list[str],
               temperature: float = 0.8) -> list[dict]:
    """One local call to the strong rung, thinking left ON (no_think=False -- variety
    matters more than latency for spec writing, per the Phase 3 plan), with one
    JSON-repair retry on a parse failure before giving up."""
    seed_block = "\n".join(f"- {s}" for s in seeds[:5]) or "(no seed examples yet)"
    prompt = (
        f"Part family: {group} -- {guidance}\n\n"
        f"Style exemplars (match their voice and specificity, NOT their dimensions or "
        f"feature mix):\n{seed_block}\n\n"
        f"Write {n} NEW specs for this family as the JSON array described.")
    raw = engine._ollama(engine.CODE_MODEL_STRONG, _SYSTEM, prompt,
                         no_think=False, temperature=temperature)
    try:
        items = _parse_json_array(raw)
    except Exception:
        raw2 = engine._ollama(engine.CODE_MODEL_STRONG, _SYSTEM, prompt + _REPAIR_SUFFIX,
                              no_think=False, temperature=temperature)
        items = _parse_json_array(raw2)
    return [{"tier": int(x.get("tier") or _TIER_BY_GROUP.get(group, 2)), "group": group,
             "spec": str(x["spec"]).strip()}
            for x in items if x.get("spec")]


def _relock_maker(arm: str = LOCK_ALIAS) -> None:
    """Restore the resident and re-apply the standing Phase 1 maker lock-in
    (maker.enabled=true, arm=gemma-4-31b) via two arms.py calls.

    KNOWN TRAP: `scripts/arms.py restore` alone sets cad.json's maker.enabled to false --
    correct for a card run returning the box to its idle default, wrong here, since the
    production CAD lock-in (Phase 1, 2026-09-17) expects maker.enabled=true/arm=gemma-4-31b
    as the box's normal resting state. So every exit path from this module restores the
    resident (`arms.py restore`) and then immediately reasserts the lock-in with
    `arms.py use <arm> --no-start` (writes cad.json + maker.env only, does not touch any
    systemd unit -- the resident stays up from the restore step)."""
    subprocess.run([sys.executable, str(ARMS_PY), "restore"], check=False)
    subprocess.run([sys.executable, str(ARMS_PY), "use", arm, "--no-start"], check=False)


def run_batch(group: str, n: int, dry_run: bool = False) -> dict:
    guidance = FAMILY_GUIDANCE.get(group)
    if guidance is None:
        raise SystemExit(f"unknown group {group!r}; known: {', '.join(ALL_GROUPS)}")
    seeds = _seed_specs(group)
    t0 = time.time()
    items = gen_family(group, n, guidance, seeds)
    elapsed = time.time() - t0
    result = specbank.add_items(items, source="specgen", group_default=group, dry_run=dry_run)
    result["group"] = group
    result["requested"] = n
    result["generated"] = len(items)
    result["seconds"] = round(elapsed, 1)
    return result


def run_total(total: int, target_tier34: float, batch_size: int = 20,
              max_batches: int = 500) -> dict:
    """Keep generating batches -- tier 3-4 families first while the bank's tier34_share
    is under `target_tier34`, then round-robin every family -- until the bank has at
    least `total` rows AND at least `target_tier34` of them are tier 3-4, or
    `max_batches` is hit (a hard stop so a stuck loop can never run all night)."""
    tier12_groups = [g for g, _, _ in FAMILIES]
    tier34_groups = [g for g, _, _ in HARD_FAMILIES]
    all_groups = tier34_groups + tier12_groups
    batches_run = 0
    log: list[dict] = []
    while batches_run < max_batches:
        st = specbank.stats()
        if st["total"] >= total and st["tier34_share"] >= target_tier34:
            break
        pool = tier34_groups if st["tier34_share"] < target_tier34 else all_groups
        group = pool[batches_run % len(pool)]
        n = min(batch_size, FAMILY_N.get(group, batch_size))
        try:
            r = run_batch(group, n)
        except Exception as e:
            r = {"group": group, "error": str(e)}
        log.append(r)
        batches_run += 1
    return {"batches": batches_run, "final_stats": specbank.stats(), "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", help="single family/group name for a one-batch run")
    ap.add_argument("--n", type=int, default=20, help="specs requested in a one-batch run")
    ap.add_argument("--once", action="store_true",
                    help="run exactly one batch (--group required) -- the smoke-test shape")
    ap.add_argument("--total", type=int, default=0, help="full-run target bank size")
    ap.add_argument("--target-tier34", type=float, default=0.45)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-relock", action="store_true",
                    help="skip the arms.py restore/use --no-start bookend (debugging only "
                         "-- never pass this on a real run)")
    a = ap.parse_args()

    if not a.once and not a.total:
        ap.error("pass --once --group NAME (smoke) or --total N (full run)")

    os.environ.setdefault("CAD_KEEP_MAKER", "1")   # one warm arm across every call in this run
    rc = 0
    try:
        if a.once:
            if not a.group:
                ap.error("--once requires --group")
            result = run_batch(a.group, a.n, dry_run=a.dry_run)
            print(json.dumps(result, indent=2))
        else:
            result = run_total(a.total, a.target_tier34, a.batch_size)
            print(json.dumps({"batches": result["batches"],
                              "final_stats": result["final_stats"]}, indent=2))
    except Exception as e:
        rc = 1
        print(f"specgen failed: {e}", file=sys.stderr)
    finally:
        if not a.no_relock:
            _relock_maker()
    return rc


if __name__ == "__main__":
    sys.exit(main())

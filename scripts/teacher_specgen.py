#!/usr/bin/env python3
"""Generate teacher-training spec variations (Andrew's suggestion: have the teacher write
the spec variations too).

One cloud call per part-family, seeded with the hand-written pilot/complex specs as style
exemplars. Output goes to benchmarks/teacher-batch2/specs.json (tiers 1-3) and
benchmarks/teacher-mech2/specs.json (tier 4 mechanisms, including rewordings of the 8
human-rejected C mechanisms with the rejection reasons baked into the new wording).

Every generated spec is slugified and checked against (a) the eval-suite slugs
(contamination guard — same rule as teacher_gen) and (b) all existing teacher specs,
then deduped within the batch. Collisions are dropped, suite collisions loudly.

Cost: ~9 sync cloud calls, ~$1 total. Spend rides the normal engine ledger.

Usage:
    python3 scripts/teacher_specgen.py            # generate both files
    python3 scripts/teacher_specgen.py --dry-run  # show the plan, no calls
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import cad_engine as engine  # noqa: E402

PILOT = HERE / "benchmarks" / "teacher-pilot" / "specs.json"
COMPLEX = HERE / "benchmarks" / "teacher-complex" / "specs.json"
OUT_BATCH2 = HERE / "benchmarks" / "teacher-batch2" / "specs.json"
OUT_MECH2 = HERE / "benchmarks" / "teacher-mech2" / "specs.json"
DECISIONS = Path.home() / ".openclaw" / "cad-review-decisions.jsonl"

# (group, n, tier guidance)
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

_SYSTEM = """\
You write CAD part specifications for a text-to-CAD training corpus. Each spec is one
plain-English sentence a mechanical engineer might type, describing ONE buildable part
(or small assembly when asked) with CONCRETE millimetre dimensions for every major feature.

Rules:
- Every spec self-consistent and physically buildable; features must fit inside the part.
- Vary dimensions, feature counts, and phrasing across specs — no two alike.
- Do NOT phrase anything as a bare catalog part name (avoid leading with exactly
  "spur gear ...", "hex bolt ...", "W-section I-beam ..." — describe the geometry instead).
- Return ONLY a JSON array of objects: [{"tier": <1-4>, "spec": "<sentence>"}, ...].
  No markdown fences, no commentary."""


def _load_specs(p: Path) -> list[dict]:
    d = json.loads(p.read_text())
    return d["benchmarks"] if isinstance(d, dict) else d


def _load_census():
    spec = importlib.util.spec_from_file_location(
        "hc", HERE / "scripts" / "harvest_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_json_array(raw: str) -> list[dict]:
    raw = re.sub(r"^```(json)?\s*\n?", "", raw.strip())
    raw = raw.rstrip("`").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array in reply: {raw[:120]!r}")
    return json.loads(raw[start:end + 1])


def gen_family(group: str, n: int, guidance: str, seeds: list[str]) -> list[dict]:
    seed_block = "\n".join(f"- {s}" for s in seeds[:5])
    prompt = (
        f"Part family: {group} — {guidance}\n\n"
        f"Style exemplars (hand-written; match their voice and specificity, NOT their "
        f"dimensions or feature mix):\n{seed_block}\n\n"
        f"Write {n} NEW specs for this family as the JSON array described.")
    engine.reset_cloud_budget(1)
    raw = engine._cloud_chat(engine.cloud_config()["model"], _SYSTEM, prompt, timeout=300)
    items = _parse_json_array(raw)
    return [{"tier": int(x.get("tier") or 2), "group": group, "spec": str(x["spec"]).strip()}
            for x in items if x.get("spec")]


def gen_mechanisms(n: int) -> list[dict]:
    rejects = []
    if DECISIONS.exists():
        for line in DECISIONS.read_text().splitlines():
            d = json.loads(line)
            if d.get("verdict") == "reject":
                rejects.append(f"- SPEC: {d['spec']}\n  REJECTED BECAUSE: {d.get('note', '')}")
    reject_block = "\n".join(rejects)
    prompt = (
        "Part family: mechanism — tier 4 multi-part mechanisms (2-4 separate solids that "
        "must genuinely assemble: no interpenetrating parts, mating features aligned, "
        "meshing features at correct centre distances and phasing).\n\n"
        "A human engineer REJECTED these earlier attempts for the stated reasons:\n"
        f"{reject_block}\n\n"
        f"Write {n} NEW mechanism specs as the JSON array described. Include reworded "
        "versions of each rejected mechanism above whose wording makes the failure "
        "explicit as a requirement (e.g. state the correct centre distance, state that "
        "the rod must NOT intersect the piston crown, state that parts must not overlap), "
        "plus fresh mechanisms (cams, ratchets, linkages, clamps, hinges, pulleys). "
        "All tier 4.")
    engine.reset_cloud_budget(1)
    raw = engine._cloud_chat(engine.cloud_config()["model"], _SYSTEM, prompt, timeout=300)
    items = _parse_json_array(raw)
    return [{"tier": 4, "group": "mechanism", "spec": str(x["spec"]).strip()}
            for x in items if x.get("spec")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mech-n", type=int, default=30)
    args = ap.parse_args()

    hc = _load_census()
    suite = hc.suite_slugs()
    existing = {hc._slug(s["spec"], 40) for s in _load_specs(PILOT) + _load_specs(COMPLEX)}
    print(f"[specgen] {len(suite)} suite slugs, {len(existing)} existing teacher slugs")
    if args.dry_run:
        for g, n, guide in FAMILIES:
            print(f"  {g:<10} n={n}  {guide}")
        print(f"  mechanism  n={args.mech_n}  (tier 4, incl. rejected-mechanism rewords)")
        return 0

    pilot = _load_specs(PILOT)
    seeds_by_group: dict[str, list[str]] = {}
    for s in pilot:
        seeds_by_group.setdefault(s.get("group", ""), []).append(s["spec"])

    seen = set(existing)

    def admit(items: list[dict], label: str) -> list[dict]:
        out = []
        for it in items:
            slug = hc._slug(it["spec"], 40)
            if slug in suite:
                print(f"  [guard] DROPPED suite collision in {label}: {it['spec'][:70]}",
                      file=sys.stderr)
                continue
            if slug in seen:
                continue
            seen.add(slug)
            out.append(it)
        return out

    batch2 = []
    for group, n, guidance in FAMILIES:
        seeds = seeds_by_group.get(group) or [s["spec"] for s in pilot[:5]]
        try:
            items = admit(gen_family(group, n, guidance, seeds), group)
        except Exception as e:
            print(f"  [{group}] FAILED: {e}", file=sys.stderr)
            continue
        print(f"  [{group}] {len(items)} specs")
        batch2.extend(items)

    mech2 = admit(gen_mechanisms(args.mech_n), "mechanism")
    print(f"  [mechanism] {len(mech2)} specs")

    for i, it in enumerate(batch2, 1):
        it["id"] = f"V{i:03d}"
    for i, it in enumerate(mech2, 1):
        it["id"] = f"M{i:02d}"
    OUT_BATCH2.parent.mkdir(parents=True, exist_ok=True)
    OUT_MECH2.parent.mkdir(parents=True, exist_ok=True)
    OUT_BATCH2.write_text(json.dumps({
        "_meta": {"generated": "teacher_specgen.py", "note":
                  "teacher-training specs, NOT an eval suite; guarded against suite slugs"},
        "benchmarks": batch2}, indent=1))
    OUT_MECH2.write_text(json.dumps({
        "_meta": {"generated": "teacher_specgen.py", "note":
                  "tier-4 mechanism specs incl. rewords of human-rejected C mechanisms"},
        "benchmarks": mech2}, indent=1))
    usd, calls = engine.cloud_spend_total()
    print(f"\n[specgen] wrote {len(batch2)} -> {OUT_BATCH2}")
    print(f"[specgen] wrote {len(mech2)} -> {OUT_MECH2}")
    print(f"[specgen] ledger total now ${usd:.2f} over {calls} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())

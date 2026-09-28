#!/usr/bin/env python3
"""lab/gen_seeds_scale.py -- generate the 800-seed bank for the code-first teacher pipeline's
scale run (lab/teacher_seeds_scale.jsonl). NO API calls, NO spend -- this only writes text.

Weighted 15% tier 1 / 40% tier 2 / 35% tier 3 / 10% tier 4 (of 800: 120/320/280/80), diverse
across part families (each tier draws from several distinct family templates, not numeric
permutations of one shape), contamination-guarded against the eval suites
(lab/specbank.contamination_sets(), the same primitive lab/teacher_codefirst.py's own guard
uses) and against every seed already in lab/teacher_codefirst.py's 50-seed pilot bank (both an
exact-text check and the same sha1 contamination key, so a reworded near-duplicate is caught
too). Deterministic: a fixed RNG seed, so a re-run reproduces the same 800 lines byte-for-byte.

Usage:
    python3 lab/gen_seeds_scale.py                 # writes lab/teacher_seeds_scale.jsonl
    python3 lab/gen_seeds_scale.py --dry-run        # counts only, writes nothing
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import harvest_census as hc          # noqa: E402
from lab import specbank             # noqa: E402
from lab import teacher_codefirst as tc  # noqa: E402

OUT_FILE = HERE / "lab" / "teacher_seeds_scale.jsonl"
RNG_SEED = 20260924
TARGET_TOTAL = 800
TIER_WEIGHTS = {1: 0.15, 2: 0.40, 3: 0.35, 4: 0.10}


def _r(rng: random.Random, lo: float, hi: float, step: float = 1.0) -> float:
    """A "shop-plausible" round number in [lo, hi], snapped to `step` (mm are rarely
    specified to 3 decimal places by a person asking for a part)."""
    n = int(round((hi - lo) / step))
    v = lo + rng.randint(0, max(n, 1)) * step
    return round(v, 2)


def _ri(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


# ── family templates, one lambda per family: rng -> idea text ─────────────────────────────
# Each family is a distinct PART TYPE (not a numeric variant of a sibling family), so 800
# seeds read as 800 different kinds of mechanical parts, grouped by tier the same way the
# 50-seed pilot bank was, but never repeating its exact wording.

def _tier1_families():
    return [
        lambda r: (f"a rectangular mounting plate {_r(r,60,160,5)}x{_r(r,40,100,5)}x"
                  f"{_r(r,4,12,1)}mm with one central {_r(r,6,20,1)}mm through hole"),
        lambda r: (f"a flat rectangular spacer shim {_r(r,20,80,5)}x{_r(r,20,60,5)}x"
                  f"{_r(r,0.5,3,0.5)}mm with no holes"),
        lambda r: (f"a cylindrical standoff post {_r(r,8,25,1)}mm diameter {_r(r,10,60,5)}mm "
                  f"tall with a {_r(r,3,8,0.5)}mm through bore"),
        lambda r: (f"a flat washer-like disc {_r(r,20,60,2)}mm diameter with a "
                  f"{_r(r,4,16,1)}mm centre hole"),
        lambda r: (f"a straight metal strap {_r(r,80,220,10)}x{_r(r,15,40,5)}x"
                  f"{_r(r,2,6,1)}mm with two {_r(r,4,10,1)}mm through holes near each end"),
        lambda r: (f"a square panel {_r(r,50,140,10)}x{_r(r,50,140,10)}x{_r(r,3,10,1)}mm "
                  f"with a single {_r(r,20,60,5)}mm square cutout in the middle"),
        lambda r: f"a plain solid cylindrical shaft {_r(r,10,40,2)}mm diameter {_r(r,60,200,10)}mm long",
        lambda r: (f"a rectangular block {_r(r,40,100,5)}x{_r(r,30,80,5)}x{_r(r,10,30,2)}mm "
                  f"with one {_r(r,6,16,1)}mm counterbored hole"),
        lambda r: (f"a small square tab {_r(r,20,45,5)}x{_r(r,20,45,5)}x{_r(r,2,6,1)}mm with "
                  f"one {_r(r,4,8,0.5)}mm mounting hole and rounded corners"),
        lambda r: (f"a solid rectangular spacer block {_r(r,40,90,5)}x{_r(r,30,70,5)}x"
                  f"{_r(r,10,25,2)}mm with four {_r(r,3,8,0.5)}mm corner holes"),
        lambda r: (f"a short tube segment {_r(r,20,60,2)}mm outside diameter "
                  f"{_r(r,14,50,2)}mm inside diameter {_r(r,20,80,5)}mm long, open at both ends"),
        lambda r: (f"a flat disc plate {_r(r,50,120,5)}mm diameter with three equally spaced "
                  f"{_r(r,4,10,1)}mm mounting holes"),
    ]


def _tier2_families():
    return [
        lambda r: (f"an L-shaped angle bracket with a {_r(r,50,100,5)}x{_r(r,30,60,5)}mm "
                  f"horizontal leg and a {_r(r,30,60,5)}x{_r(r,30,60,5)}mm vertical leg, "
                  f"both {_r(r,3,8,1)}mm thick, with {_ri(r,2,4)} {_r(r,4,8,1)}mm mounting holes"),
        lambda r: (f"a gusseted corner bracket {_r(r,50,80,5)}x{_r(r,50,80,5)}x{_r(r,4,8,1)}mm "
                  f"with a triangular stiffening rib {_r(r,3,6,1)}mm thick"),
        lambda r: (f"a tie plate {_r(r,80,160,10)}x{_r(r,20,40,5)}x{_r(r,5,10,1)}mm with a "
                  f"{_r(r,8,16,1)}mm hole at each end and its outside corners rounded"),
        lambda r: (f"an open-top rectangular tray {_r(r,80,180,10)}x{_r(r,60,140,10)}x"
                  f"{_r(r,20,40,5)}mm with {_r(r,2,4,0.5)}mm walls and a flat floor"),
        lambda r: (f"a hollow electronics enclosure case {_r(r,60,150,10)}x{_r(r,50,120,10)}x"
                  f"{_r(r,20,45,5)}mm, open on top, with {_r(r,2,3,0.5)}mm walls and a "
                  f"mounting lip"),
        lambda r: (f"an open-top box {_r(r,60,120,10)}x{_r(r,60,120,10)}x{_r(r,25,50,5)}mm "
                  f"with {_ri(r,2,4)} cylindrical standoff bosses inside, each "
                  f"{_r(r,6,12,1)}mm across"),
        lambda r: (f"a straight rectangular duct segment {_r(r,100,250,10)}mm long, "
                  f"{_r(r,40,80,5)}x{_r(r,30,60,5)}mm outer cross-section, {_r(r,2,4,0.5)}mm "
                  f"walls"),
        lambda r: (f"a stepped shaft {_r(r,80,180,10)}mm long: {_r(r,20,40,2)}mm diameter for "
                  f"the first {_r(r,30,60,5)}mm, then {_r(r,12,25,1)}mm diameter for the rest"),
        lambda r: (f"a shaft {_r(r,25,50,2)}mm diameter {_r(r,60,140,10)}mm long with a "
                  f"{_r(r,6,10,1)}mm wide by {_r(r,3,6,0.5)}mm deep keyway cut along "
                  f"{_r(r,30,70,5)}mm of its length"),
        lambda r: (f"a shaft {_r(r,30,60,2)}mm diameter {_r(r,50,100,10)}mm long with a "
                  f"{_r(r,15,30,1)}mm central through bore and a {_r(r,2,4,0.5)}mm wide "
                  f"retaining-ring groove near one end"),
        lambda r: (f"a spindle {_r(r,20,45,2)}mm diameter {_r(r,50,110,10)}mm long with a "
                  f"{_ri(r,30,60)} degree chamfer {_r(r,2,4,0.5)}mm deep on both ends"),
        lambda r: (f"a round flange {_r(r,70,150,10)}mm outside diameter and {_r(r,8,16,1)}mm "
                  f"thick with a {_r(r,25,55,5)}mm central bore and {_ri(r,4,8)} "
                  f"{_r(r,6,12,1)}mm holes on a bolt circle"),
        lambda r: (f"a square flange {_r(r,60,110,5)}x{_r(r,60,110,5)}x{_r(r,8,16,1)}mm with "
                  f"a {_r(r,20,45,5)}mm central bore and four {_r(r,6,10,1)}mm holes on a "
                  f"bolt circle"),
        lambda r: (f"a blind flange {_r(r,80,150,10)}mm diameter and {_r(r,10,20,2)}mm thick "
                  f"with {_ri(r,6,10)} {_r(r,6,10,1)}mm holes on a bolt circle and no centre "
                  f"bore"),
        lambda r: (f"a pillow-block style bearing mount with a raised boss "
                  f"{_r(r,25,45,5)}mm diameter on an {_r(r,60,110,10)}x{_r(r,30,50,5)}mm base "
                  f"and {_ri(r,2,4)} {_r(r,5,9,1)}mm base holes"),
        lambda r: (f"a clevis fork with two parallel ears {_r(r,30,60,5)}x{_r(r,20,40,5)}x"
                  f"{_r(r,4,8,1)}mm, {_r(r,15,30,2)}mm apart, and a {_r(r,6,12,1)}mm pin hole "
                  f"through both"),
        lambda r: (f"a U-shaped mounting bracket {_r(r,40,90,5)}mm wide with "
                  f"{_r(r,30,60,5)}mm side walls and a {_r(r,4,8,1)}mm hole in each wall"),
        lambda r: (f"a stepped spacer sleeve with a {_r(r,20,40,2)}mm shoulder diameter, a "
                  f"{_r(r,12,25,1)}mm body diameter, {_r(r,20,50,5)}mm long, and a "
                  f"{_r(r,6,12,1)}mm through bore"),
        lambda r: (f"a flanged bushing with a {_r(r,20,40,2)}mm outside diameter head, a "
                  f"{_r(r,10,20,1)}mm body diameter, {_r(r,15,40,5)}mm long, and a "
                  f"{_r(r,5,10,1)}mm through bore"),
        lambda r: (f"a cable-clamp saddle bracket with a curved seat {_r(r,15,35,2)}mm across "
                  f"and two mounting feet each with a {_r(r,4,7,1)}mm hole"),
        lambda r: (f"a single-groove pulley wheel {_r(r,50,120,10)}mm diameter with a "
                  f"{_r(r,8,20,1)}mm central bore and a {_r(r,3,6,0.5)}mm set-screw hole"),
        lambda r: (f"a pipe coupling collar {_r(r,25,60,5)}mm outside diameter with a "
                  f"{_r(r,15,40,5)}mm bore through its {_r(r,20,50,5)}mm length and a split "
                  f"slot"),
        lambda r: (f"a bearing end cap {_r(r,40,90,5)}mm diameter with a {_r(r,15,35,2)}mm "
                  f"central bore and {_ri(r,4,6)} {_r(r,3,6,0.5)}mm holes on a bolt circle"),
        lambda r: (f"a slotted bar {_r(r,80,180,10)}x{_r(r,20,40,5)}x{_r(r,6,12,1)}mm with one "
                  f"{_r(r,8,16,1)}mm wide slot running {_r(r,40,100,10)}mm along its length"),
        lambda r: (f"a bored plate {_r(r,60,120,10)}x{_r(r,60,120,10)}x{_r(r,6,12,1)}mm with a "
                  f"{_r(r,15,35,2)}mm central bore and {_ri(r,4,6)} {_r(r,4,8,1)}mm holes on a "
                  f"bolt circle"),
    ]


def _tier3_families():
    return [
        lambda r: (f"a pipe flange {_r(r,90,180,10)}mm diameter {_r(r,12,22,2)}mm thick with a "
                  f"{_r(r,30,70,5)}mm bore, a {_r(r,2,4,0.5)}mm raised sealing face, and "
                  f"{_ri(r,6,8)} {_r(r,7,12,1)}mm holes on a bolt circle"),
        lambda r: (f"a two-part assembly: a {_r(r,50,90,5)}x{_r(r,50,90,5)}x{_r(r,6,10,1)}mm "
                  f"base plate and a separate {_r(r,20,40,2)}mm diameter "
                  f"{_r(r,30,60,5)}mm tall post that plugs into it"),
        lambda r: (f"a square-to-round transition duct {_r(r,60,140,10)}mm tall, square "
                  f"{_r(r,40,80,5)}x{_r(r,40,80,5)}mm at the base, blending to a "
                  f"{_r(r,25,55,5)}mm diameter circle at the top"),
        lambda r: (f"a shallow revolved bowl {_r(r,80,160,10)}mm across the rim, "
                  f"{_r(r,30,70,5)}mm deep, with a rounded rim"),
        lambda r: (f"a {_r(r,10,25,1)}mm diameter rod swept along a smooth S-shaped curved "
                  f"path {_r(r,100,220,10)}mm end to end"),
        lambda r: (f"a small spur gear, {_ri(r,10,24)} teeth, module {_r(r,1,2.5,0.5)}, "
                  f"{_r(r,10,25,2)}mm face width, with a {_r(r,6,14,1)}mm centre bore"),
        lambda r: (f"a hinge knuckle segment for a folding bracket, barrel "
                  f"{_r(r,15,30,2)}mm diameter {_r(r,20,40,5)}mm long, with a "
                  f"{_r(r,4,8,1)}mm pin bore through it"),
        lambda r: (f"a perforated disc {_r(r,80,160,10)}mm diameter {_r(r,3,7,1)}mm thick "
                  f"with a hexagonal grid of {_r(r,4,8,1)}mm holes on "
                  f"{_r(r,10,18,1)}mm centres"),
        lambda r: (f"a rectangular ventilation panel {_r(r,90,180,10)}x{_r(r,50,110,10)}x"
                  f"{_r(r,3,6,1)}mm with rows of rounded slots {_r(r,25,45,5)}mm long"),
        lambda r: (f"a cam plate {_r(r,60,120,10)}mm diameter {_r(r,6,12,1)}mm thick with an "
                  f"off-centre eccentric bore {_r(r,12,25,1)}mm diameter for a follower"),
        lambda r: (f"a ratchet pawl plate {_r(r,50,100,5)}x{_r(r,30,60,5)}x{_r(r,4,8,1)}mm "
                  f"with a {_r(r,6,12,1)}mm pivot bore and a hooked profile at one end"),
        lambda r: (f"a T-slot rail segment {_r(r,100,220,10)}mm long, {_r(r,20,40,5)}mm wide, "
                  f"with a captured T-shaped channel along its length"),
        lambda r: (f"a {_r(r,4,10,1)}mm diameter rod swept along a helix of "
                  f"{_r(r,20,50,5)}mm pitch, {_r(r,15,35,2)}mm radius, {_ri(r,2,4)} full turns"),
        lambda r: (f"a lofted blade {_r(r,60,120,10)}mm tall twisting {_ri(r,20,60)} degrees "
                  f"from root to tip, with an aerofoil-like cross-section"),
        lambda r: (f"a saddle-shaped surface panel {_r(r,80,140,10)}x{_r(r,80,140,10)}mm and "
                  f"{_r(r,3,6,1)}mm thick whose surface curves upward along one axis and down "
                  f"along the other"),
        lambda r: (f"a torus-like ring {_r(r,60,140,10)}mm across with a "
                  f"{_r(r,10,22,1)}mm circular cross-section, cut flat on one side to sit flush"),
        lambda r: (f"a smooth fillet-blended junction between a {_r(r,25,45,2)}mm diameter "
                  f"vertical cylinder {_r(r,40,80,5)}mm tall and a {_r(r,50,100,5)}mm square "
                  f"base plate"),
        lambda r: (f"a rounded organic handle {_r(r,15,28,1)}mm diameter section swept along "
                  f"a C-shaped spline arc {_r(r,90,160,10)}mm across"),
    ]


def _tier4_families():
    return [
        lambda r: (f"a three-knuckle piano hinge segment, each knuckle "
                  f"{_r(r,15,25,1)}mm diameter, with a pin bore running through all three"),
        lambda r: (f"a multi-lobe cam plate {_r(r,60,110,10)}mm diameter with {_ri(r,3,5)} "
                  f"lobes and a {_r(r,10,20,1)}mm centre bore"),
        lambda r: (f"a worm-gear housing boss {_r(r,60,120,10)}mm long with a "
                  f"{_r(r,25,45,2)}mm bore for the worm shaft and a mounting flange at one end"),
        lambda r: (f"a planetary-gear carrier plate {_r(r,60,110,10)}mm diameter with "
                  f"{_ri(r,3,4)} {_r(r,8,14,1)}mm planet-shaft bores equally spaced around a "
                  f"{_r(r,15,25,1)}mm centre bore"),
        lambda r: (f"a dovetail slide {_r(r,80,160,10)}mm long, {_r(r,20,40,5)}mm wide, with a "
                  f"{_ri(r,45,60)} degree dovetail rail along its length"),
        lambda r: (f"a two-link articulated bracket: two {_r(r,40,70,5)}x{_r(r,20,35,5)}x"
                  f"{_r(r,5,9,1)}mm arms joined by a {_r(r,6,10,1)}mm pivot bore, each arm "
                  f"also carrying a {_r(r,5,9,1)}mm mounting hole at its far end"),
        lambda r: (f"a spiral-bevel gear blank {_r(r,50,100,10)}mm pitch diameter with a "
                  f"{_ri(r,20,35)} degree cone angle and a {_r(r,10,20,1)}mm centre bore"),
        lambda r: (f"a cross-slot universal-joint yoke {_r(r,50,90,10)}mm across with two "
                  f"perpendicular {_r(r,8,14,1)}mm cross-bores and a {_r(r,15,25,2)}mm shaft "
                  f"bore"),
    ]


def generate(rng: random.Random) -> list[dict]:
    families = {1: _tier1_families(), 2: _tier2_families(), 3: _tier3_families(),
               4: _tier4_families()}
    targets = {t: round(TARGET_TOTAL * w) for t, w in TIER_WEIGHTS.items()}
    # rounding can drift the total by 1-2; fix it up on tier 2 (the largest bucket)
    drift = TARGET_TOTAL - sum(targets.values())
    targets[2] += drift

    suite_keys, suite_slugs = specbank.contamination_sets()
    pilot_keys = {hc._key(s["idea"]) for s in tc.SEEDS}
    pilot_slugs = {hc._slug(s["idea"], 40) for s in tc.SEEDS}

    out: list[dict] = []
    seen_keys: set[str] = set()
    n = 0
    for tier, target in targets.items():
        fams = families[tier]
        made = 0
        attempts = 0
        # round-robin the families so the tier's seeds are spread evenly across part types,
        # not front-loaded onto the first family before moving to the next
        fam_i = 0
        while made < target and attempts < target * 30:
            attempts += 1
            fam = fams[fam_i % len(fams)]
            fam_i += 1
            idea = fam(rng)
            key = hc._key(idea)
            if key in seen_keys or key in pilot_keys or key in suite_keys:
                continue
            slug = hc._slug(idea, 40)
            if slug in pilot_slugs or slug in suite_slugs:
                continue
            seen_keys.add(key)
            n += 1
            out.append({"id": f"cfs{n:04d}", "tier": tier, "idea": idea})
            made += 1
        if made < target:
            print(f"WARNING: tier {tier} only reached {made}/{target} unique seeds after "
                 f"{attempts} attempts", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = generate(rng)

    by_tier: dict[int, int] = {}
    for r in rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"generated {len(rows)} seeds: " +
         ", ".join(f"tier{t}={by_tier.get(t,0)}" for t in sorted(by_tier)))

    if a.dry_run:
        return 0

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

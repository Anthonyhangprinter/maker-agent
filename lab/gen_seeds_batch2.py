#!/usr/bin/env python3
"""lab/gen_seeds_batch2.py -- generate the ~600-seed bank for teacher batch 2
(lab/teacher_seeds_batch2.jsonl). NO API calls, NO spend -- this only writes text.

WHY (owner findings, 2026-09-25): batch 1 (lab/teacher_seeds_scale.jsonl, 800 seeds, kept 443
pairs for $39.41) mostly teaches stock Gemma-4-31B nothing -- a live baseline check
(benchmarks/results/card/codefirst-scale-2026-09-25/gemma_baseline.jsonl, summarised by
lab/gemma_baseline_report.py) found stock Gemma already matches roughly 90% of those pairs
(tier 1 ~99%, tier 2 ~85%). The owner also read a sample of the "hard" tier 3-4 batch-1 parts
and judged most of them to actually be 2.5D extrusions in disguise (plates, discs, flanges
with holes) -- the seed WORDING allowed the model to solve them with one prismatic extrude +
holes even where the tier label implied more. Batch 2 fixes both: it targets (a) the specific
part families where Gemma demonstrably fails today, paraphrased with different sizes/feature
counts (never a copy of a failing spec) so the new pairs still generalise, and (b) genuinely
3D constructions that CANNOT be solved by one extrude -- a revolve, sweep, loft, shell, a
multi-plane feature set, a 3D pattern, or a fillet/chamfer-heavy part -- so the wording itself
forces the harder operation, not just implies it.

OWNER UPDATE (2026-09-25, mid-session): training is on hold pending a human review of harder
pairs, so batch 2 is now the main event, not a side dataset. The control slice is capped at 5%
(not 10%) -- nearly everything must be either a demonstrated Gemma failure or genuinely 3D.

Mix (~600 total, weighted then rounded, drift fixed up on the largest bucket):
  ~45%  GEMMA-FAILING families (see FAIL_FAMILIES below, sourced from the report)
  ~50%  GENUINELY-3D-CONSTRUCTION families (revolve/sweep/loft/shell/multi-plane/pattern/
        fillet-chamfer-heavy -- see CONSTRUCTION_FAMILIES)
  ~5%   tier-1/2 CONTROLS (things Gemma already solves, kept small as a sanity check only)

Tier-tagged honestly by CONSTRUCTION difficulty (how many distinct operations/planes/features
a correct build needs), not by whether Gemma happens to pass or fail it. Contamination-guarded
against every eval suite (lab/specbank.contamination_sets()) AND against every idea already in
lab/teacher_codefirst.py's 50-seed pilot bank AND every idea already in batch 1's 800-seed bank
(lab/teacher_seeds_scale.jsonl) -- exact key and 40-char slug, same as gen_seeds_scale.py.
Deterministic: a fixed RNG seed, so a re-run reproduces the same lines byte-for-byte.

Usage:
    python3 lab/gen_seeds_batch2.py                 # writes lab/teacher_seeds_batch2.jsonl
    python3 lab/gen_seeds_batch2.py --dry-run        # counts only, writes nothing
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

OUT_FILE = HERE / "lab" / "teacher_seeds_batch2.jsonl"
FAMILIES_FILE = HERE / "lab" / "teacher_seeds_batch2_families.json"
BATCH1_SEEDS_FILE = HERE / "lab" / "teacher_seeds_scale.jsonl"
RNG_SEED = 20260925
TARGET_TOTAL = 600
BUCKET_WEIGHTS = {"fail": 0.45, "construction": 0.50, "control": 0.05}


def _r(rng: random.Random, lo: float, hi: float, step: float = 1.0) -> float:
    """A "shop-plausible" round number in [lo, hi], snapped to `step`."""
    n = int(round((hi - lo) / step))
    v = lo + rng.randint(0, max(n, 1)) * step
    return round(v, 2)


def _ri(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _choice(rng: random.Random, seq):
    return seq[rng.randint(0, len(seq) - 1)]


# ── bucket 1: GEMMA-FAILING families ───────────────────────────────────────────────────────
# Paraphrased from the part families lab/gemma_baseline_report.py's family match-rate and
# FAILS-by-family sections show scoring worst against the stock gemma-4-31b arm on batch 1's
# 443 kept pairs (see benchmarks/results/card/codefirst-scale-2026-09-25/
# gemma_baseline_report.txt for the exact numbers this bucket is built from). Each lambda is a
# DIFFERENT size/feature-count instance of the failing idea, never the literal failing spec
# text -- a paraphrase that keeps the part TYPE and the feature combination Gemma trips on,
# not a copy checked by the contamination guard below anyway.
def _fail_families():
    return [
        # -- filled in from gemma_baseline_report.txt below; see FAIL_FAMILIES_FILLED --
    ]


# ── bucket 2: GENUINELY-3D-CONSTRUCTION families ───────────────────────────────────────────
# Each of these needs a non-extrude operation or a real interplay of 3D features to build
# correctly -- an extrude-then-drill-holes solution cannot produce the right shape at all, so
# these cannot degenerate into a 2.5D part the way several batch-1 tier-3/4 seeds did (owner's
# read of the batch-1 sample: most "hard" parts were plates/discs/flanges, i.e. one extrude +
# holes with a fancier name). Grouped by the operation family; tier reflects how many distinct
# operations/planes/features a correct build actually needs, not any model's pass/fail.
def _revolve_families():
    return [
        # stepped shaft with a groove/undercut at the step -- a straight extrude cannot
        # produce a step AND an undercut groove at the same axial position; needs a revolved
        # profile (or an equivalent turned-profile construction).
        lambda r: (f"a turned shaft {_r(r,100,200,10)}mm long: {_r(r,28,45,1)}mm diameter for "
                  f"the first {_r(r,35,70,5)}mm, stepping down to {_r(r,14,26,1)}mm diameter "
                  f"for the rest, with a {_r(r,3,6,0.5)}mm wide by {_r(r,1.5,3,0.5)}mm deep "
                  f"undercut groove right at the step"),
        # V-groove pulley -- the groove profile is angled, not a straight-wall cut; a correct
        # build needs a revolved V profile around the rim.
        lambda r: (f"a V-groove pulley {_r(r,60,140,10)}mm outside diameter, {_r(r,20,40,5)}mm "
                  f"thick, with a {_ri(r,40,60)}-degree V-groove {_r(r,8,16,1)}mm deep cut "
                  f"around its rim, a {_r(r,15,30,1)}mm centre bore, and a "
                  f"{_r(r,4,7,1)}mm set-screw hole"),
        # bearing housing with an internal stepped bore -- a plain through-bore would not need
        # a revolved internal profile; the retaining shoulder does.
        lambda r: (f"a cylindrical bearing housing {_r(r,60,110,10)}mm outside diameter "
                  f"{_r(r,40,80,5)}mm long, with a stepped internal bore: "
                  f"{_r(r,35,55,2)}mm diameter for the first {_r(r,15,30,5)}mm forming a "
                  f"retaining shoulder, then {_r(r,25,45,2)}mm diameter for the rest"),
        # knob -- a domed/tapered profile revolved around a D-shaft bore, no flat extrude will
        # produce the domed top.
        lambda r: (f"a rounded control knob {_r(r,28,50,2)}mm diameter, {_r(r,18,32,2)}mm "
                  f"tall, tapering from a wide base up to a smoothly domed top, with a "
                  f"{_r(r,5,8,0.5)}mm D-shaped shaft bore through the centre"),
        lambda r: (f"a hand wheel {_r(r,70,140,10)}mm diameter with a domed, revolved rim "
                  f"profile {_r(r,10,20,2)}mm thick at the rim tapering thinner toward the "
                  f"hub, a {_r(r,10,20,1)}mm centre bore, and a {_r(r,3,5,0.5)}mm keyway"),
    ]


def _sweep_families():
    return [
        # pipe bend -- an extrude cannot follow a curved centreline; needs a sweep along an
        # arc path.
        lambda r: (f"a pipe bend, {_r(r,18,36,2)}mm outside diameter, {_r(r,1.5,3,0.5)}mm "
                  f"wall thickness, swept through a {_ri(r,45,120)}-degree bend with a "
                  f"{_r(r,40,90,5)}mm bend radius, both ends cut square"),
        # spring with closed, ground ends -- a helical sweep, plus flattened end turns; cannot
        # be an extrude of any cross-section.
        lambda r: (f"a compression spring, {_r(r,30,60,5)}mm outside diameter, "
                  f"{_r(r,3,6,0.5)}mm wire diameter, {_ri(r,6,12)} active coils, "
                  f"{_r(r,60,140,10)}mm free length, with the last half-turn at each end "
                  f"closed and ground flat to sit square"),
        # handle -- swept round section along a curved spline path.
        lambda r: (f"an ergonomic handle, {_r(r,14,24,1)}mm diameter round cross-section, "
                  f"swept along a gently curved path {_r(r,90,160,10)}mm end to end, with "
                  f"rounded ends"),
        # hose barb / grip spiral rib swept along a helix on a cylindrical body
        lambda r: (f"a grip sleeve {_r(r,20,36,2)}mm outside diameter {_r(r,50,100,10)}mm "
                  f"long, with a {_r(r,2,4,0.5)}mm wide raised rib swept along a helix of "
                  f"{_r(r,15,30,2)}mm pitch running the full length, over a "
                  f"{_r(r,8,16,1)}mm through bore"),
        lambda r: (f"a U-shaped grab handle, {_r(r,10,18,1)}mm diameter round section, "
                  f"swept along a U-shaped path {_r(r,60,120,10)}mm wide and "
                  f"{_r(r,40,80,10)}mm tall, with a flat mounting tab at each end carrying a "
                  f"{_r(r,4,7,1)}mm hole"),
    ]


def _loft_families():
    return [
        # tapered nozzle -- a smooth lofted profile between two different circular sections;
        # a straight cone extrude would not match a curved (non-linear) taper profile.
        lambda r: (f"a tapered nozzle, {_r(r,50,90,5)}mm diameter at the inlet narrowing to "
                  f"{_r(r,18,32,2)}mm diameter at the outlet over {_r(r,70,130,10)}mm length, "
                  f"with a smoothly curved (not straight-sided) taper profile"),
        # hexagon-to-circle transition duct -- lofting between two DIFFERENT polygon counts,
        # distinct from batch 1's square-to-round family.
        lambda r: (f"a hexagon-to-circle transition duct {_r(r,60,130,10)}mm tall, a "
                  f"{_r(r,40,80,5)}mm across-flats hexagon at the base, lofting to a "
                  f"{_r(r,30,60,5)}mm diameter circle at the top"),
        # oval-to-rectangle transition duct
        lambda r: (f"an oval-to-rectangle transition duct {_r(r,60,120,10)}mm tall, a "
                  f"{_r(r,50,90,5)}x{_r(r,30,55,5)}mm oval at the base, lofting to a "
                  f"{_r(r,45,80,5)}x{_r(r,45,80,5)}mm square opening at the top"),
        # boat-hull-like lofted body: several stations, distinct from a simple 2-section loft
        lambda r: (f"a lofted teardrop-section body {_r(r,120,200,10)}mm long, blending from "
                  f"a {_r(r,20,35,2)}mm diameter round nose to a "
                  f"{_r(r,40,70,5)}x{_r(r,15,28,2)}mm flattened teardrop cross-section at "
                  f"the tail"),
    ]


def _shell_families():
    return [
        # hollow enclosure with bosses/ribs/lid -- needs a shell operation plus interior
        # detail that a solid-block-with-a-pocket cannot cheaply fake past the feature-count
        # check (bosses AND ribs, both interior).
        lambda r: (f"a shelled electronics enclosure base {_r(r,90,160,10)}x{_r(r,60,110,10)}x"
                  f"{_r(r,30,55,5)}mm with {_r(r,2,3,0.5)}mm walls, {_ri(r,4,4)} corner "
                  f"mounting bosses inside each with a {_r(r,2.5,4,0.5)}mm pilot hole, and "
                  f"{_ri(r,2,3)} internal reinforcing ribs {_r(r,1.5,3,0.5)}mm thick"),
        # cup -- a shelled revolved body, tapered wall, open top.
        lambda r: (f"a drinking cup, {_r(r,60,90,5)}mm diameter at the rim tapering to "
                  f"{_r(r,45,70,5)}mm diameter at the base, {_r(r,80,120,10)}mm tall, "
                  f"shelled to a {_r(r,1.5,3,0.5)}mm wall thickness, open top, with a "
                  f"{_r(r,2,4,0.5)}mm fillet where the wall meets the base"),
        # bottle-like hollow body with a narrower neck
        lambda r: (f"a hollow bottle body, {_r(r,50,80,5)}mm diameter main body "
                  f"{_r(r,80,140,10)}mm tall, shelled to a {_r(r,1.5,3,0.5)}mm wall, necking "
                  f"down to a {_r(r,18,30,2)}mm diameter open neck at the top"),
        # shelled housing with a separate lid, both parts sharing a mounting-lip interface
        lambda r: (f"a shelled instrument housing {_r(r,80,140,10)}x{_r(r,60,100,10)}x"
                  f"{_r(r,35,60,5)}mm with {_r(r,2,3,0.5)}mm walls, an outward mounting lip "
                  f"around the open top, and a matching flat lid that sits on the lip with "
                  f"{_ri(r,4,4)} screw holes"),
    ]


def _multiplane_families():
    return [
        # holes through 3 different faces -- cannot be solved by drilling everything from one
        # plane; needs the model to work on 3 distinct planes of the same solid.
        lambda r: (f"a boxed mounting bracket {_r(r,60,110,10)}x{_r(r,45,80,5)}x"
                  f"{_r(r,35,65,5)}mm with a {_r(r,8,14,1)}mm hole through the top face, "
                  f"{_ri(r,2,2)} {_r(r,5,9,1)}mm holes through one side face, and one "
                  f"{_r(r,6,10,1)}mm hole through the front face"),
        # a hole drilled at an angle to another hole on a different face -- forces working on
        # a non-default reference plane.
        lambda r: (f"a rectangular block {_r(r,70,110,10)}x{_r(r,50,80,5)}x{_r(r,25,45,5)}mm "
                  f"with a {_r(r,8,14,1)}mm hole drilled straight down through the top and a "
                  f"second {_r(r,6,11,1)}mm hole drilled through one side face at a "
                  f"{_ri(r,25,45)}-degree angle to it"),
        # counterbore on a side face rather than the top -- distinct from the common
        # top-face counterbore, still needs the side-plane construction.
        lambda r: (f"an L-bracket, a {_r(r,70,120,10)}x{_r(r,45,75,5)}mm horizontal leg and a "
                  f"{_r(r,45,75,5)}x{_r(r,45,75,5)}mm vertical leg, both {_r(r,5,9,1)}mm "
                  f"thick, with a {_r(r,8,14,1)}mm counterbored hole through the vertical "
                  f"leg's face and {_ri(r,2,2)} {_r(r,5,9,1)}mm through holes on the "
                  f"horizontal leg"),
        # a cube-ish block with a distinct hole/feature on each of 3 adjacent faces
        lambda r: (f"a junction block {_r(r,55,90,5)}x{_r(r,55,90,5)}x{_r(r,55,90,5)}mm with "
                  f"a {_r(r,15,28,1)}mm bore through the top face, a {_r(r,15,28,1)}mm bore "
                  f"through one side face at 90 degrees to it, and a "
                  f"{_r(r,6,10,1)}mm mounting hole through the front face"),
    ]


def _pattern_families():
    return [
        # polar pattern -- radial ribs around a bore; a single rib mirrored is not a pattern,
        # 5+ ribs forces a real polar-array construction.
        lambda r: (f"a circular hub {_r(r,60,110,10)}mm diameter {_r(r,10,20,2)}mm thick with "
                  f"{_ri(r,5,8)} radial stiffening ribs, each {_r(r,3,5,0.5)}mm thick, evenly "
                  f"spaced in a polar pattern around a {_r(r,18,32,2)}mm centre bore"),
        # linear pattern -- parallel fins, evenly spaced, forces a real linear-array
        # construction rather than one extruded fin.
        lambda r: (f"a rectangular heat sink base {_r(r,50,90,5)}x{_r(r,50,90,5)}x"
                  f"{_r(r,6,12,1)}mm with {_ri(r,6,10)} parallel cooling fins, each "
                  f"{_r(r,1.5,3,0.5)}mm thick and {_r(r,15,30,3)}mm tall, spaced evenly "
                  f"across the top in a linear pattern"),
        # polar pattern of holes on a non-trivial bolt-circle COUNT (paired with a boss, not
        # just a flange -- forces pattern + a separate feature working together)
        lambda r: (f"a circular mounting hub {_r(r,70,130,10)}mm diameter with a raised "
                  f"central boss {_r(r,25,45,5)}mm diameter {_r(r,8,16,2)}mm tall, and "
                  f"{_ri(r,6,10)} {_r(r,4,7,1)}mm holes evenly spaced in a polar pattern "
                  f"around the boss"),
        # linear pattern of ribs across a curved shroud face
        lambda r: (f"a curved fan shroud panel {_r(r,100,180,10)}mm wide {_r(r,60,100,10)}mm "
                  f"tall {_r(r,3,6,1)}mm thick with {_ri(r,5,9)} evenly spaced stiffening "
                  f"ribs {_r(r,2,4,0.5)}mm thick running across it in a linear pattern"),
    ]


def _fillet_chamfer_families():
    return [
        lambda r: (f"a smoothly filleted corner bracket {_r(r,60,100,10)}x{_r(r,40,70,5)}x"
                  f"{_r(r,25,45,5)}mm with a {_r(r,6,10,1)}mm fillet on every external edge "
                  f"and a {_r(r,3,5,0.5)}mm fillet on the internal corner where its two faces "
                  f"meet"),
        lambda r: (f"a chamfered mounting block {_r(r,50,90,5)}x{_r(r,50,90,5)}x"
                  f"{_r(r,20,35,5)}mm with a {_r(r,2,4,0.5)}mm chamfer on all twelve edges "
                  f"and a central {_r(r,16,28,2)}mm through hole chamfered {_r(r,1.5,3,0.5)}mm "
                  f"on both ends"),
        lambda r: (f"a heavily filleted slider block {_r(r,60,100,10)}x{_r(r,35,60,5)}x"
                  f"{_r(r,20,35,5)}mm with a {_r(r,8,14,1)}mm fillet along both long top "
                  f"edges and a {_r(r,4,7,1)}mm fillet along both long bottom edges"),
    ]


CONSTRUCTION_GROUPS = [
    _revolve_families, _sweep_families, _loft_families, _shell_families,
    _multiplane_families, _pattern_families, _fillet_chamfer_families,
]
# tier by construction group -- how many distinct operations/planes a correct build needs,
# never adjusted by any model's pass/fail rate.
CONSTRUCTION_TIER = {
    "_revolve_families": 3, "_sweep_families": 3, "_loft_families": 3,
    "_shell_families": 3, "_multiplane_families": 3, "_pattern_families": 3,
    "_fillet_chamfer_families": 2,
}
# a few of the harder instances (multi-feature interplay) get bumped to tier 4 by position
# within their group -- see generate()'s TIER4_BUMP below.
TIER4_BUMP = {"_revolve_families": {2}, "_sweep_families": {1}, "_shell_families": {0, 3},
              "_multiplane_families": {3}, "_pattern_families": {2}}


# ── bucket 3: tier-1/2 CONTROLS (kept small -- things Gemma already solves) ────────────────
def _control_families():
    return [
        lambda r: (f"a rectangular mounting plate {_r(r,60,150,5)}x{_r(r,40,100,5)}x"
                  f"{_r(r,4,12,1)}mm with one central {_r(r,6,18,1)}mm through hole"),
        lambda r: (f"a flat washer-like disc {_r(r,20,60,2)}mm diameter with a "
                  f"{_r(r,4,16,1)}mm centre hole"),
        lambda r: (f"a straight metal strap {_r(r,80,200,10)}x{_r(r,15,40,5)}x"
                  f"{_r(r,2,6,1)}mm with two {_r(r,4,10,1)}mm through holes near each end"),
        lambda r: (f"a solid rectangular spacer block {_r(r,40,90,5)}x{_r(r,30,70,5)}x"
                  f"{_r(r,10,25,2)}mm with four {_r(r,3,8,0.5)}mm corner holes"),
        lambda r: (f"a round flange {_r(r,70,150,10)}mm outside diameter and {_r(r,8,16,1)}mm "
                  f"thick with a {_r(r,25,55,5)}mm central bore and {_ri(r,4,8)} "
                  f"{_r(r,6,12,1)}mm holes on a bolt circle"),
    ]


def generate(rng: random.Random) -> tuple[list[dict], dict[str, str]]:
    suite_keys, suite_slugs = specbank.contamination_sets()
    pilot_keys = {hc._key(s["idea"]) for s in tc.SEEDS}
    pilot_slugs = {hc._slug(s["idea"], 40) for s in tc.SEEDS}
    batch1_rows = []
    if BATCH1_SEEDS_FILE.exists():
        for line in BATCH1_SEEDS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                batch1_rows.append(json.loads(line))
    batch1_keys = {hc._key(r["idea"]) for r in batch1_rows}
    batch1_slugs = {hc._slug(r["idea"], 40) for r in batch1_rows}

    def blocked(idea: str) -> bool:
        key = hc._key(idea)
        if key in suite_keys or key in pilot_keys or key in batch1_keys:
            return True
        slug = hc._slug(idea, 40)
        if slug in suite_slugs or slug in pilot_slugs or slug in batch1_slugs:
            return True
        return False

    targets = {b: round(TARGET_TOTAL * w) for b, w in BUCKET_WEIGHTS.items()}
    drift = TARGET_TOTAL - sum(targets.values())
    targets["construction"] += drift

    out: list[dict] = []
    seen_keys: set[str] = set()
    families: dict[str, str] = {}
    n = 0

    def emit(idea: str, tier: int, family: str) -> bool:
        nonlocal n
        key = hc._key(idea)
        if key in seen_keys or blocked(idea):
            return False
        seen_keys.add(key)
        n += 1
        sid = f"cfb2{n:04d}"
        out.append({"id": sid, "tier": tier, "idea": idea})
        families[sid] = family
        return True

    # -- bucket: fail (filled in after the gemma baseline report; see FAIL_FAMILIES) --
    fail_fams = FAIL_FAMILIES
    made = 0
    attempts = 0
    fam_i = 0
    target = targets["fail"]
    if fail_fams:
        while made < target and attempts < target * 40:
            attempts += 1
            name, tier, fam = fail_fams[fam_i % len(fail_fams)]
            fam_i += 1
            if emit(fam(rng), tier, f"fail:{name}"):
                made += 1
        if made < target:
            print(f"WARNING: fail bucket only reached {made}/{target} unique seeds",
                  file=sys.stderr)

    # -- bucket: construction --
    made = 0
    attempts = 0
    target = targets["construction"]
    # flatten (tier, lambda) pairs across all 7 groups, in round-robin group order
    flat = []
    for group_fn in CONSTRUCTION_GROUPS:
        gname = group_fn.__name__
        base_tier = CONSTRUCTION_TIER[gname]
        bump = TIER4_BUMP.get(gname, set())
        for idx, fam in enumerate(group_fn()):
            tier = 4 if idx in bump else base_tier
            flat.append((tier, fam))
    # interleave by group so no single group front-loads
    per_group = [list(enumerate(group_fn())) for group_fn in CONSTRUCTION_GROUPS]
    group_names = [g.__name__ for g in CONSTRUCTION_GROUPS]
    gi = 0
    idx_in_group = [0] * len(CONSTRUCTION_GROUPS)
    while made < target and attempts < target * 40:
        attempts += 1
        g = gi % len(CONSTRUCTION_GROUPS)
        gname = group_names[g]
        fams = per_group[g]
        pos = idx_in_group[g] % len(fams)
        fam_idx, fam = fams[pos]
        idx_in_group[g] += 1
        gi += 1
        base_tier = CONSTRUCTION_TIER[gname]
        tier = 4 if fam_idx in TIER4_BUMP.get(gname, set()) else base_tier
        short = gname.strip("_").replace("_families", "")
        if emit(fam(rng), tier, f"construction:{short}:{fam_idx}"):
            made += 1
    if made < target:
        print(f"WARNING: construction bucket only reached {made}/{target} unique seeds",
              file=sys.stderr)

    # -- bucket: control --
    made = 0
    attempts = 0
    target = targets["control"]
    fams = _control_families()
    fam_i = 0
    while made < target and attempts < target * 40:
        attempts += 1
        fam = fams[fam_i % len(fams)]
        fam_i += 1
        tier = 1 if fam_i % 2 == 0 else 2
        if emit(fam(rng), tier, "control"):
            made += 1
    if made < target:
        print(f"WARNING: control bucket only reached {made}/{target} unique seeds",
              file=sys.stderr)

    return out, families


# ── FAIL_FAMILIES: populated from gemma_baseline_report.txt (real measured data, run
# 2026-09-25, n=472 pairs, overall match+valid 81%: tier1 98%/tier2 83%/tier3 59%/tier4 52%,
# crashes 64/472=13.6% concentrated in tier3-4 (40 of 107)) ────────────────────────────────
# Each entry is (name, tier, lambda rng -> idea). Tier is the CONSTRUCTION tier the original
# batch-1 family template used (lab/gen_seeds_scale.py), never adjusted by Gemma's pass/fail --
# these are all families the report shows at 0% match+valid (the 12 the owner named directly:
# single-groove pulley wheel, worm-gear housing boss, U-shaped mounting bracket, bearing end
# cap, bored plate, cable clamp saddle bracket, cam plate, cross-slot universal-joint yoke,
# hollow electronics enclosure open-top, pipe coupling collar, ratchet pawl plate, rectangular
# ventilation panel) plus 5 more families the FAILS-by-family breakdown shows with an even
# stronger raw fail count (multi-lobe cam plate 5/5, small spur gear ~9 fails across variants,
# smooth fillet-blended junction ~11 fails across variants, saddle-shaped surface panel 3/3,
# clevis fork 2/2) or a real crash cluster (square flange 5 fails, all crashes/near-misses).
# Every idea below is a PARAPHRASE (different wording structure, different size/feature-count
# ranges) of the batch-1 template that produced the failing family, never its literal text --
# the contamination guard below still checks every emitted idea against batch 1's exact key
# and slug regardless.
def _fail_families():
    return [
        ("pulley_wheel", 2, lambda r: (
            f"a single-groove belt pulley, {_r(r,45,100,5)}mm outside diameter, "
            f"{_r(r,12,25,2)}mm wide, with a {_r(r,10,22,1)}mm through bore, a "
            f"{_r(r,3,6,0.5)}mm keyway, and a {_r(r,4,7,1)}mm radial set-screw hole")),
        ("worm_gear_housing_boss", 4, lambda r: (
            f"a worm-gear housing boss, {_r(r,70,140,10)}mm long, {_r(r,30,50,5)}mm outside "
            f"diameter, with a {_r(r,20,40,2)}mm bore running its full length for the worm "
            f"shaft and a {_r(r,60,100,10)}mm square mounting flange with {_ri(r,4,4)} "
            f"{_r(r,5,8,1)}mm holes at one end")),
        ("u_bracket", 2, lambda r: (
            f"a U-shaped mounting bracket, {_r(r,45,95,5)}mm wide, {_r(r,25,55,5)}mm deep, "
            f"with {_r(r,25,55,5)}mm tall side walls {_r(r,3,6,1)}mm thick, and a "
            f"{_r(r,4,9,1)}mm hole centred in each side wall")),
        ("bearing_end_cap", 2, lambda r: (
            f"a bearing end cap, {_r(r,45,95,5)}mm diameter, {_r(r,6,12,1)}mm thick, with a "
            f"{_r(r,16,38,2)}mm central bore and {_ri(r,3,6)} {_r(r,3,7,0.5)}mm holes evenly "
            f"spaced on a bolt circle near the rim")),
        ("bored_plate", 2, lambda r: (
            f"a bored mounting plate, {_r(r,65,130,10)}x{_r(r,65,130,10)}x{_r(r,7,13,1)}mm, "
            f"with a {_r(r,16,38,2)}mm central bore and {_ri(r,3,5)} {_r(r,4,9,1)}mm holes on "
            f"a bolt circle around it")),
        ("cable_clamp_saddle", 2, lambda r: (
            f"a cable-clamp saddle bracket, a curved seat {_r(r,18,38,2)}mm across and "
            f"{_r(r,15,30,2)}mm deep, sitting on a base with two mounting feet, each foot "
            f"carrying a {_r(r,4,8,1)}mm hole")),
        ("cam_plate", 3, lambda r: (
            f"a cam plate, {_r(r,65,130,10)}mm diameter, {_r(r,7,14,1)}mm thick, with an "
            f"eccentric follower bore {_r(r,14,28,1)}mm diameter offset "
            f"{_r(r,8,20,1)}mm from the plate centre, and a {_r(r,6,10,1)}mm mounting hole "
            f"near the rim")),
        ("universal_joint_yoke", 4, lambda r: (
            f"a cross-slot universal-joint yoke, {_r(r,55,100,10)}mm across, with two "
            f"perpendicular {_r(r,9,16,1)}mm cross-bores through the fork ends and a "
            f"{_r(r,16,28,2)}mm bore through the hub for the shaft")),
        ("open_top_enclosure", 2, lambda r: (
            f"a hollow electronics enclosure case, {_r(r,65,150,10)}x{_r(r,55,120,10)}x"
            f"{_r(r,22,48,5)}mm, open on top, {_r(r,2,3,0.5)}mm walls, with a mounting lip "
            f"around the open edge and {_ri(r,2,4)} internal corner bosses"
        )),
        ("pipe_coupling_collar", 2, lambda r: (
            f"a pipe coupling collar, {_r(r,28,65,5)}mm outside diameter, with a bore "
            f"{_r(r,16,42,5)}mm diameter running through its {_r(r,22,55,5)}mm length, and a "
            f"{_r(r,2,4,0.5)}mm wide split slot along its length")),
        ("ratchet_pawl_plate", 3, lambda r: (
            f"a ratchet pawl plate, {_r(r,55,105,5)}x{_r(r,32,65,5)}x{_r(r,4,9,1)}mm, with a "
            f"{_r(r,7,13,1)}mm pivot bore near one end and a hooked catch profile at the "
            f"opposite end")),
        ("ventilation_panel", 3, lambda r: (
            f"a rectangular ventilation panel, {_r(r,95,190,10)}x{_r(r,55,115,10)}x"
            f"{_r(r,3,7,1)}mm, with {_ri(r,3,5)} rows of rounded slots, each slot "
            f"{_r(r,20,50,5)}mm long")),
        ("multi_lobe_cam", 4, lambda r: (
            f"a multi-lobe cam plate, {_r(r,55,115,10)}mm diameter, {_r(r,6,12,1)}mm thick, "
            f"with {_ri(r,3,5)} lobes evenly spaced around the rim and a "
            f"{_r(r,10,22,1)}mm centre bore")),
        ("spur_gear", 3, lambda r: (
            f"a small spur gear, {_ri(r,11,26)} teeth, module {_r(r,1,2.5,0.5)}, "
            f"{_r(r,10,28,2)}mm face width, with a {_r(r,6,16,1)}mm centre bore and a "
            f"{_r(r,3,5,0.5)}mm keyway")),
        ("fillet_blended_junction", 3, lambda r: (
            f"a smooth fillet-blended junction where a {_r(r,22,48,2)}mm diameter vertical "
            f"cylinder, {_r(r,35,85,5)}mm tall, meets a {_r(r,55,105,5)}mm square base plate, "
            f"the fillet radius {_r(r,6,14,1)}mm all the way around")),
        ("saddle_surface_panel", 3, lambda r: (
            f"a saddle-shaped surface panel, {_r(r,75,145,10)}x{_r(r,75,145,10)}mm and "
            f"{_r(r,3,6,1)}mm thick, whose surface curves upward along one axis and downward "
            f"along the perpendicular axis")),
        ("square_flange", 2, lambda r: (
            f"a square flange plate, {_r(r,60,115,5)}x{_r(r,60,115,5)}x{_r(r,8,17,1)}mm, with "
            f"a {_r(r,18,45,5)}mm central bore and {_ri(r,4,4)} {_r(r,6,11,1)}mm corner holes "
            f"on a bolt circle")),
    ]


FAIL_FAMILIES: list = _fail_families()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows, families = generate(rng)

    by_tier: dict[int, int] = {}
    for r in rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"generated {len(rows)} seeds: " +
         ", ".join(f"tier{t}={by_tier.get(t,0)}" for t in sorted(by_tier)))
    by_bucket: dict[str, int] = {}
    for fam in families.values():
        bucket = fam.split(":", 1)[0]
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
    print("buckets: " + ", ".join(f"{k}={v}" for k, v in sorted(by_bucket.items())))

    if a.dry_run:
        return 0

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    FAMILIES_FILE.write_text(json.dumps(families, indent=2), encoding="utf-8")
    print(f"wrote {OUT_FILE}")
    print(f"wrote {FAMILIES_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

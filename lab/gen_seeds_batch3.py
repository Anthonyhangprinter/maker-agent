#!/usr/bin/env python3
"""lab/gen_seeds_batch3.py -- generate the seed bank for teacher batch 3
(lab/teacher_seeds_batch3.jsonl). NO API calls, NO spend -- this only writes text.

WHY THESE SPECIFIC NUMBERS (owner findings, 2026-09-26, after batch 2 finished): batch 2's own
measured keep rate (verified pairs / rebuild attempts, from
benchmarks/results/card/codefirst-batch2-2026-09-25/claude-opus-5-5/results.jsonl joined
against lab/teacher_seeds_batch2_families.json) is WILDLY uneven by family/construction-group:

  fail bucket overall:          258 attempted, 148 kept  = 57.4%
  construction bucket overall:  235 attempted,  52 kept  = 22.1%

  construction sub-groups: multiplane 54.8% (42 attempted), pattern 35.0% (40), revolve 24.4%
  (41), sweep 13.2% (38), loft 0.0% (41), shell 0.0% (31), fillet_chamfer 0.0% (n=2, too few
  attempts to trust -- starved by its own tier-2 priority rank, not evidence of failure).

  fail sub-families, keep rate: bearing_end_cap/bored_plate/cam_plate/pipe_coupling_collar/
  square_flange 100%, fillet_blended_junction 86.7%, u_bracket 81.2%, worm_gear_housing_boss
  80%, multi_lobe_cam 75%, spur_gear 50%, ventilation_panel 37.5%, ratchet_pawl_plate 33.3%,
  universal_joint_yoke 13.3%, cable_clamp_saddle 7.1%, saddle_surface_panel 6.7%,
  open_top_enclosure 0%, pulley_wheel 0%.

Per the owner's rule ("drop construction groups under 15% keep, they burn money"): sweep,
loft, shell, fillet_chamfer are DROPPED from batch 3's construction allocation (fillet_chamfer
despite the caveat above -- the rule is applied literally, its near-zero attempt count is
disclosed, not treated as an exception). KEPT: multiplane, pattern, revolve. The fail bucket's
own two worst performers (open_top_enclosure, pulley_wheel -- both 0% despite genuinely
failing Gemma) are ALSO dropped: Gemma still needs teaching on them, but batch 2 proved this
teacher pipeline cannot verify a rebuild for them at all, so spending seeds there wastes money
for zero yield regardless of how much Gemma needs the lesson.

Per-family/per-group seed counts below are SIZED from these real rates (a family's rate
directly sets a HIGH/MEDIUM/LOW seed-count tier: 36 seeds for a >=75%-keep family, 24 for
25-75%, 14 for <25%; construction groups get 96/72/60 for multiplane/pattern/revolve,
reflecting their own decreasing keep rates) so the total ~700-seed bank is expected to yield
about 330-440 kept pairs (300k target-ish, per-family expected kept = seeds x that family's
OWN measured rate; a real buffer above the ~330 target is intentional -- these rates were each
measured on only 14-42 attempts, and a "100%" rate from n=15-16 almost certainly regresses
toward something lower on a bigger sample). Controls have ZERO batch-2 keep-rate data (all 29
were pushed to the very back of the priority queue and never got a real rebuild attempt before
budget ran out), so their expected yield uses an ESTIMATE (65%, a plausible midpoint given
simple tier-1/2 parts elsewhere ran high), clearly flagged as unmeasured, not fact.

Contamination-guarded against every eval suite AND against every idea already in the pilot's
50 seeds, batch 1's 800 seeds, AND batch 2's 600 seeds (exact key + 40-char slug).
Deterministic: a fixed RNG seed, so a re-run reproduces the same lines byte-for-byte.

Usage:
    python3 lab/gen_seeds_batch3.py                 # writes lab/teacher_seeds_batch3.jsonl
    python3 lab/gen_seeds_batch3.py --dry-run        # counts only, writes nothing
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
from lab import gen_seeds_batch2 as b2    # noqa: E402 -- reuse the SAME family lambdas, never a
                                          # second copy of their wording

OUT_FILE = HERE / "lab" / "teacher_seeds_batch3.jsonl"
FAMILIES_FILE = HERE / "lab" / "teacher_seeds_batch3_families.json"
BATCH1_SEEDS_FILE = HERE / "lab" / "teacher_seeds_scale.jsonl"
BATCH2_SEEDS_FILE = HERE / "lab" / "teacher_seeds_batch2.jsonl"
RNG_SEED = 20260926

# ── FAIL bucket: name -> (tier, target_seed_count, lambda). Reuses gen_seeds_batch2's exact
# family lambdas (b2._fail_families()) by name lookup so the WORDING is never duplicated; only
# open_top_enclosure and pulley_wheel are excluded (0% teacher-pipeline yield in batch 2).
_B2_FAIL = {name: (tier, fam) for name, tier, fam in b2._fail_families()}

# 3 of those families put their FIRST differentiating number after character 40 of the idea
# text ("a cable-clamp saddle bracket, a curved s|eat 38mm across..." -- the cut lands before
# any digit), so harvest_census._slug(text, 40) -- a 40-char PREFIX used as a near-duplicate
# guard -- is IDENTICAL across every size variant of that template. Batch 2 already used that
# same identical slug (any one of its ~14-16 instances is enough to poison it), so EVERY new
# batch-3 variant of the unmodified template collided and 0/N seeds were ever accepted (caught
# by generate()'s own "reached 0/N" warning below, not silently). Fixed by rewording these 3
# (only these 3 -- every other reused template already puts a number within the first ~30
# chars) so their first number lands well inside the 40-char window; verified below with
# assert_no_slug_collision() before this module is ever used to generate real output.
def _fillet_blended_junction_v2(r):
    return (f"a {b2._r(r,22,48,2)}mm diameter cylindrical boss, {b2._r(r,35,85,5)}mm tall, "
           f"blended with a smooth fillet onto a {b2._r(r,55,105,5)}mm square base plate, "
           f"the fillet radius {b2._r(r,6,14,1)}mm all the way around")


def _universal_joint_yoke_v2(r):
    return (f"a {b2._r(r,55,100,10)}mm wide cross-slot universal-joint yoke with two "
           f"perpendicular {b2._r(r,9,16,1)}mm cross-bores through the fork ends and a "
           f"{b2._r(r,16,28,2)}mm bore through the hub for the shaft")


def _cable_clamp_saddle_v2(r):
    return (f"a {b2._r(r,18,38,2)}mm wide cable-clamp saddle bracket with a curved seat "
           f"{b2._r(r,15,30,2)}mm deep, sitting on a base with two mounting feet, each foot "
           f"carrying a {b2._r(r,4,8,1)}mm hole")


_B2_FAIL["fillet_blended_junction"] = (_B2_FAIL["fillet_blended_junction"][0], _fillet_blended_junction_v2)
_B2_FAIL["universal_joint_yoke"] = (_B2_FAIL["universal_joint_yoke"][0], _universal_joint_yoke_v2)
_B2_FAIL["cable_clamp_saddle"] = (_B2_FAIL["cable_clamp_saddle"][0], _cable_clamp_saddle_v2)


def assert_no_slug_collision() -> None:
    """A cheap, fast sanity probe (not the real contamination guard -- generate() still runs
    that in full): draws a handful of samples from every REWORDED template and confirms no two
    share the same 40-char slug, catching a repeat of the exact bug this fix addresses before
    a real generation run wastes time hitting the same wall for a different template."""
    probe = random.Random(1)
    for name in ("fillet_blended_junction", "universal_joint_yoke", "cable_clamp_saddle"):
        _tier, fam = _B2_FAIL[name]
        slugs = {hc._slug(fam(probe), 40) for _ in range(8)}
        assert len(slugs) > 1, f"{name!r}: all 8 probe samples share one 40-char slug"
_FAIL_COUNTS = {
    # >=75% keep in batch 2 -> HIGH tier, 36 seeds each
    "bearing_end_cap": 36, "bored_plate": 36, "cam_plate": 36, "pipe_coupling_collar": 36,
    "square_flange": 36, "fillet_blended_junction": 36, "u_bracket": 36,
    "worm_gear_housing_boss": 36, "multi_lobe_cam": 36,
    # 25-75% keep -> MEDIUM tier, 24 seeds each
    "spur_gear": 24, "ventilation_panel": 24, "ratchet_pawl_plate": 24,
    # <25% keep -> LOW tier, 14 seeds each (still included: Gemma fails on these too, just
    # cheaply, not abandoned outright)
    "universal_joint_yoke": 14, "cable_clamp_saddle": 14, "saddle_surface_panel": 14,
    # open_top_enclosure, pulley_wheel: DELIBERATELY OMITTED (0% keep, batch 2 proved this
    # pipeline cannot verify a rebuild for them regardless of Gemma's need)
}

# ── CONSTRUCTION bucket: only the 3 groups at/above the 15%-keep line survive. Reuses
# gen_seeds_batch2's exact group lambdas (b2._revolve_families/_pattern_families/
# _multiplane_families) so their wording is never duplicated either.
_CONSTRUCTION_COUNTS = {
    "multiplane": (3, 96),   # 54.8% keep in batch 2
    "pattern": (3, 72),      # 35.0% keep
    "revolve": (3, 60),      # 24.4% keep (one template bumped to tier 4 in batch 2; kept at
                             # tier 3 here since seed count is now driven by measured rate,
                             # not a fixed construction-difficulty label)
}
_B2_CONSTRUCTION_GROUP_FNS = {
    "multiplane": b2._multiplane_families,
    "pattern": b2._pattern_families,
    "revolve": b2._revolve_families,
}

_CONTROL_COUNT = 35  # ~5% of the ~700 total, matching batch 2's own control-bucket size


def generate(rng: random.Random) -> tuple[list[dict], dict[str, str]]:
    assert_no_slug_collision()
    suite_keys, suite_slugs = specbank.contamination_sets()
    pilot_keys = {hc._key(s["idea"]) for s in tc.SEEDS}
    pilot_slugs = {hc._slug(s["idea"], 40) for s in tc.SEEDS}

    def _load_ideas(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [json.loads(line)["idea"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    prior_keys: set[str] = set(pilot_keys)
    prior_slugs: set[str] = set(pilot_slugs)
    for f in (BATCH1_SEEDS_FILE, BATCH2_SEEDS_FILE):
        for idea in _load_ideas(f):
            prior_keys.add(hc._key(idea))
            prior_slugs.add(hc._slug(idea, 40))

    def blocked(idea: str) -> bool:
        key = hc._key(idea)
        if key in suite_keys or key in prior_keys:
            return True
        slug = hc._slug(idea, 40)
        if slug in suite_slugs or slug in prior_slugs:
            return True
        return False

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
        sid = f"cfb3{n:04d}"
        out.append({"id": sid, "tier": tier, "idea": idea})
        families[sid] = family
        return True

    # -- fail bucket --
    for name, target in _FAIL_COUNTS.items():
        tier, fam = _B2_FAIL[name]
        made = attempts = 0
        while made < target and attempts < target * 40:
            attempts += 1
            if emit(fam(rng), tier, f"fail:{name}"):
                made += 1
        if made < target:
            print(f"WARNING: fail family {name!r} only reached {made}/{target} unique seeds",
                  file=sys.stderr)

    # -- construction bucket --
    for gname, (base_tier, target) in _CONSTRUCTION_COUNTS.items():
        fams = _B2_CONSTRUCTION_GROUP_FNS[gname]()
        made = attempts = 0
        idx = 0
        while made < target and attempts < target * 40:
            attempts += 1
            fam = fams[idx % len(fams)]
            idx += 1
            if emit(fam(rng), base_tier, f"construction:{gname}"):
                made += 1
        if made < target:
            print(f"WARNING: construction group {gname!r} only reached {made}/{target} "
                 f"unique seeds", file=sys.stderr)

    # -- control bucket (reuse gen_seeds_batch2's own control templates) --
    fams = b2._control_families()
    made = attempts = 0
    fam_i = 0
    while made < _CONTROL_COUNT and attempts < _CONTROL_COUNT * 40:
        attempts += 1
        fam = fams[fam_i % len(fams)]
        fam_i += 1
        tier = 1 if fam_i % 2 == 0 else 2
        if emit(fam(rng), tier, "control"):
            made += 1
    if made < _CONTROL_COUNT:
        print(f"WARNING: control bucket only reached {made}/{_CONTROL_COUNT} unique seeds",
              file=sys.stderr)

    return out, families


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

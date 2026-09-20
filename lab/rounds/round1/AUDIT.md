# Round 1 hand audit — machine-verified CAD training pairs

Sample: seed 1, deterministic, one row per spec (lowest temperature where more than one
qualified), n=32 from `lab/state/pairs.jsonl` plus the standalone fixture
`v066_t05_correct_hole`. All 32 codes were rebuilt from scratch (CPU only, `cad_engine.run_step`
+ `run_inspect`, no model/server calls) and the recomputed geometry facts (solids, faces, bores,
walls, volume, bbox) matched the stored `facts` exactly in every case — the stored numbers are
trustworthy; any defect below is a spec-vs-geometry mismatch, not a measurement error.

## Table

| id | spec_id | tier | strength | sim | verdict | evidence |
|---|---|---|---|---|---|---|
| p:t:teacher-pilot:P24:7875ec26948d | t:teacher-pilot:P24 | 3 | cross_pass | 0.599 | CORRECT | back 40x25x8, ears 40x25x5 20mm apart (gap faces at x=±10..15), Ø10 hole through both ears at z=14; bbox 40x25x48 checks out |
| p:t:teacher-pilot:P24:617e1072dac0 | t:teacher-pilot:P24 | 3 | cross_pass | 0.605 | CORRECT | same dimensions, alternate but equally valid axis assignment; hole spans both ears |
| p:t:teacher-batch2:V047:98c2a0f43edf | t:teacher-batch2:V047 | 2 | cross_pass | 0.649 | CORRECT | "long edges rounded to 6mm" only feasible reading given 6mm thickness: bullnosing the 95mm-long top/bottom edges needs r≤3mm (half the 6mm thickness); code fillets the 4 vertical corner edges instead, the only R6-feasible reading |
| p:t:teacher-batch2:V047:642d437d1c7c | t:teacher-batch2:V047 | 2 | cross_pass | 0.649 | CORRECT | same interpretation, same result |
| p:t:teacher-batch2:V072:7f9dbe6ae9b7 | t:teacher-batch2:V072 | 3 | same_pass | 0.491 | **WRONG** | gasket groove footprint (`groove_outer_l=length-2*wall`=212) is entirely inside the cavity's own boundary (cavity also =212); the groove sits at x∈[100,106], fully inside the already-hollow cavity (x∈[-106,106]) — the subtraction is a no-op, groove absent from the built part |
| p:t:teacher-batch2:V137:eaf21608ac49 | t:teacher-batch2:V137 | 3 | same_pass | 0.978 | CORRECT | sphere r=10 (Ø20) meets cone top r=10 exactly at z=0 (sphere equator); shank spans 15mm to bottom r=4 (Ø8) — all 3 stated numbers verified by z-arithmetic |
| p:t:teacher-batch2:V191:6c5e76e455dd | t:teacher-batch2:V191 | 3 | same_pass | 0.892 | **WRONG** | keyway `Box(4, bore_dia/2+keyway_h=8, ...)` is centered at the axis (Y∈[-4,4]), entirely inside the Ø12 bore (r=6) — no-op. Confirmed: measured `faces=8` per hub, exactly what a plain bored+bolted disc gives with zero keyway faces |
| p:t:teacher-complex:C10:2d9b5167ff2e | t:teacher-complex:C10 | 4 | same_pass | 0.746 | **WRONG** | code's own comment admits it: "square thread is approximated... we create the shaft" — screw is a bare smooth cylinder (3 faces) and the nut a smooth bore (11 faces), no thread geometry anywhere; "4mm pitch square thread" / "matching internal thread" is the part's defining feature and is entirely absent |
| p:t:teacher-complex:C11:8f55a85093fc | t:teacher-complex:C11 | 4 | same_pass | 0.602 | **WRONG** | keyway_cut is shifted along Z (the wrong axis, `Pos(0,0,(bore_d/2)-(keyway_h/2))`) instead of radially outward in Y; box spans Y∈[-2,2], X∈[-4,4], fully inside the Ø25 bore (r=12.5) — no-op. Confirmed: `faces=7` per hub, matching a bore+3-bolt-holes disc with no keyway |
| p:t:teacher-hard:H006:fcce66e02599 | t:teacher-hard:H006 | 3 | same_pass | 0.619 | **WRONG** | no separate liner: `inner_void_radius = bore_radius+water_gap` merges the stated 92mm bore and the 5mm gap into one Ø102 void; measured `bores=[10,102,112]` — no 92 anywhere, and `walls=[5.0]` shows a single wall, not a liner+gap+shell |
| p:t:teacher-hard:H018:f9bf24fabe97 | t:teacher-hard:H018 | 3 | same_pass | 0.795 | **WRONG** | the "liner" solid exactly re-fills the annular region the gap was meant to leave void (`liner = jacket_inner_r ring` added back on top of `outer_shell`'s own inner boundary at the same radius); net result is one solid wall from r49 to r60, confirmed by `walls=[11.0]` (single wall, no void) — "5.5mm annular coolant gap" does not exist as empty space |
| p:t:teacher-pilot:P22:aa58c8095cd5 | t:teacher-pilot:P22 | 3 | same_pass | 0.887 | CORRECT | RF spans z 7.5→9.5 (2mm proud, exact); bore 50mm through; bolt circle 4×Ø10 @ Ø95 all match spec numbers |
| p:t:teacher-batch2:V056:c9ed8a7f8de2 | t:teacher-batch2:V056 | 3 | reference | 0.0 | CORRECT | floor arithmetic: shift=floor/2=1.5, cavity bottom at z=-12, outer bottom -15 → floor=3mm exact; walls 2.5mm exact; 2 ribs at ±12.5 span full cavity → 3 cells |
| p:t:teacher-batch2:V083:051e46f8d250 | t:teacher-batch2:V083 | 3 | reference | 0.0 | CORRECT | 3 steps, lengths/diameters match; keyway box spans radius 7→10 on the Ø20 section (surface at r=10), a genuine 3mm-deep, 6mm-wide, 60mm-long cut |
| p:t:teacher-batch2:V091:a5fb78e79f8a | t:teacher-batch2:V091 | 3 | reference | 0.0 | CORRECT | flat cut depth = 17-14=3mm gives exactly 28mm across-flats; cross-hole at z=0 (center of the 35mm flatted section); axis choice (X, through the still-round faces) is unspecified by the prompt, not a violation |
| p:t:teacher-batch2:V093:8eb01eeddf24 | t:teacher-batch2:V093 | 3 | reference | 0.0 | CORRECT | 2 steps match; keyway spans radius 12→16 on the Ø32 section (surface at r=16), genuine 4mm-deep 10mm-wide 70mm-long cut, correctly centered inside the 100mm section |
| p:t:teacher-batch2:V121:1e26a46df654 | t:teacher-batch2:V121 | 3 | reference | 0.0 | CORRECT | flange/RF/hub stub z-stack all arithmetically consistent; 8×Ø11 holes on Ø98 circle through everything |
| p:t:teacher-pilot:P35:1761fff043e9 | t:teacher-pilot:P35 | 3 | reference | 0.0 | CORRECT | major_radius=(90-15)/2=37.5 (checks OD=90); chord-cut math for a 6mm-wide flat solved correctly (h≈6.87mm) and the cutter box top is placed exactly at -h |
| p:g:specgen:1e8bee7cbe7749bc:479af084a000 | g:specgen:1e8bee7c... | 1 | same_pass | 0.911 | CORRECT | 3 slots, 12x5mm, evenly spaced at -50/0/50 on a 200mm strip |
| p:g:specgen:2f3bdb65c7a97fd2:e7d75c7b8f20 | g:specgen:2f3bdb65... | 2 | same_pass | 0.896 | CORRECT | floor/2 shift formula gives exact 1.8mm floor+walls; side-wall opening cutter genuinely crosses real wall material (wall exists x∈[48.2,50]) |
| p:g:specgen:32466f2b5b9bdd71:800245515e12 | g:specgen:32466f2b... | 1 | same_pass | 0.762 | CORRECT | trivial shim + central hole, dims match |
| p:g:specgen:558c21ef2ae2aed2:3685ae8f880f | g:specgen:558c21ef... | 2 | same_pass | 0.807 | CORRECT | counterbore helper called with through_d/bore_d/bore_depth matching spec exactly, cut from the correct (top) face with full-depth clearance |
| p:g:specgen:67ae92fb7c4e75f6:98c6d3e2f9f5 | g:specgen:67ae92fb... | 1 | same_pass | 0.745 | CORRECT | centered 20x50 cutout, dims match |
| p:g:specgen:ad9a063518c3ee1a:fe4feceab875 | g:specgen:ad9a0635... | 2 | same_pass | 0.667 | CORRECT | 2 holes + centered cutout, all dims/positions match |
| p:g:specgen:cedbf22789538882:d72469bdb510 | g:specgen:cedbf227... | 1 | same_pass | 0.855 | CORRECT | "30mm from the top-right corner" read as 30mm in from each edge (x=70,y=20) — standard drawing convention, not a violation |
| p:g:specgen:dca817abc8a06f22:6131dedcfa0d | g:specgen:dca817ab... | 2 | same_pass | 0.958 | CORRECT | 2 holes @ ±70mm + centered 40x10 slot, dims match |
| p:g:specgen:e33accce7e15f17c:371236b3e592 | g:specgen:e33accce... | 1 | same_pass | 0.7 | CORRECT | 5 holes at -60/-30/0/30/60, 30mm pitch, matches spec exactly |
| p:g:specgen:f3f0d21d74c3446a:9fe4ffd462dd | g:specgen:f3f0d21d... | 2 | same_pass | 0.848 | **WRONG** | cavity already reaches the full outer top (`inner_box` top = outer top, i.e. fully open-top by construction — floor/2 shift confirms it), so the "50x20mm opening on the top surface" cutter lies entirely inside the already-void top and does nothing; the built opening is the *entire* top (192x112mm), not the specified 50x20mm |
| p:t:teacher-batch2:V007:1ecc577a2515 | t:teacher-batch2:V007 | 2 | reference | 0.0 | CORRECT | bolt_circle(6, 50, 6, ...) matches "50mm bolt circle of six 6mm holes" exactly |
| p:t:teacher-batch2:V014:dc05d8af93b5 | t:teacher-batch2:V014 | 1 | reference | 0.0 | CORRECT | bolt_circle(2, 30, 4, ...) — 2 points on a circle are always 180° apart, matches |
| p:t:teacher-batch2:V043:ef5c3f8ba8cc | t:teacher-batch2:V043 | 2 | reference | 0.0 | CORRECT | 2 holes at ±20mm = 40mm spacing, matches |
| p:t:teacher-pilot:P06:9372fe1ddb6a | t:teacher-pilot:P06 | 2 | reference | 0.0 | CORRECT | legs 70x45x5 and 45x45x5 fused at the base's bottom edge; 2 holes per leg correctly through each leg's thickness |

No AMBIGUOUS-SPEC verdicts in this round (the one candidate case, V047's "long edges," resolves
to a single geometrically feasible reading, so it is scored CORRECT, not ambiguous).

## Wrong rate by strength

| strength | wrong | audited | rate | Wilson 95% CI |
|---|---|---|---|---|
| cross_pass | 0 | 4 | 0.0% | [0.0%, 49.0%] |
| same_pass | 7 | 18 | 38.9% | [20.3%, 61.4%] |
| reference | 0 | 10 | 0.0% | [0.0%, 27.8%] |

AMBIGUOUS-SPEC: 0 of 32.

## Wrong rate by tier band

| tier band | wrong | audited | rate | Wilson 95% CI |
|---|---|---|---|---|
| tier 1-2 | 1 | 16 | 6.2% | [1.1%, 28.3%] |
| tier 3-4 | 6 | 16 | 37.5% | [18.5%, 61.4%] |

Cross-cut that matters most: **same_pass tier 3-4 is 6/8 wrong (75%, CI [40.9%, 92.9%])** — every
same_pass tier 3-4 spec we drew except V137 (teardrop pin) and P22 (flange) failed. same_pass
tier 1-2 is 1/10 wrong (10%, CI [1.8%, 40.4%]), consistent with the corpus-wide same_pass rate
once tier is controlled for.

Overall: 7/32 wrong (21.9%, CI [11.0%, 38.8%]).

## V066 T=0.5 ruling (fixture `v066_t05_correct_hole`, not in pairs.jsonl)

Spec: enclosure 140x90x60mm, 3mm walls, divider, 12mm cable hole. Code builds `inner_cavity =
Pos(0,0,wall) * Box(length-2*wall, width-2*wall, height-wall)` = `Pos(0,0,3) * Box(134,84,57)`.

Arithmetic: outer box spans z -30..30. Cavity height 57 (=60-3), centered box half-height 28.5,
shifted by `wall`=3 → cavity z-range is **-25.5 .. 31.5**. Floor thickness = cavity bottom minus
outer bottom = -25.5 - (-30) = **4.5mm**, not the stated 3mm — a 1.5mm (50%) overage. (The
correct formula, used elsewhere in this same corpus e.g. V072/V056/f3f0d21d, shifts by
`floor/2`, not `floor`; shifting by the full wall value is the bug — it double-counts half the
floor thickness.) Cross-check against the measured volume: computing outer-minus-cavity (with
the cavity clipped by the outer box at the open top) plus the divider net of its cable hole gives
an estimated volume of ≈142,979 mm³ against the stored measured **142,664.257 mm³** — same
order, same construction, confirms the 4.5mm-floor reading (a 3mm-floor construction would be
several thousand mm³ lower, since the cavity height would be 3mm taller each side).

**Ruling: the LLM judge was correct — the floor is actually 4.5mm, not the specified 3mm, and
this is a genuine violation (1.5mm > the 0.5mm threshold), not a citation error. The
"we labelled it correct" ledger entry for this pair is wrong and should be corrected/excluded.**

## Three most instructive failures

1. **Cooling-jacket water gap collapses to solid metal (H006, H018).** Both same_pass builds
   never create an actual annular void: H006 merges the stated 92mm bore and the 5mm gap into a
   single Ø102 hole (the 92mm bore never appears in the geometry at all), and H018 adds a
   "liner" that exactly re-fills the gap it was supposed to leave empty. Two different codings
   of the same category of part, same defect family, both confirmed same_pass.
2. **Keyways subtracted entirely inside the bore they're next to (V191, C11).** Both position
   the keyway box centered on the shaft/bore axis instead of offset out to the bore's radius, so
   the box never reaches material — a pure no-op. Confirmed independently by face-count: 8 (not
   ~11) and 7 (not ~10) faces per hub, exactly what a keyway-less part would measure.
3. **A named cutout built from the cavity's own dimensions instead of the case's outer
   dimensions (V072's gasket groove, f3f0d21d's top opening).** Both size/place the "feature" so
   it falls entirely inside material that is *already* void, so the boolean subtraction changes
   nothing. This is volume-neutral and face-count-neutral versus not having the feature at all,
   which is exactly why the automated gate can't see it: nothing measurable changed.

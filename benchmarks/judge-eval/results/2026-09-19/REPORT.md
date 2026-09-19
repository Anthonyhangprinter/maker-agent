# Judge reliability: qwen3.8-27b scoring known-correct/known-wrong CAD parts

Run id: 2026-09-19. Labelled set: 81 programs (22 correct / 59 wrong by reference-scored Chamfer band or the named fixture verdicts). Rebuilt successfully: 81. Judge calls made: 162 of a possible 162.

## Headline

| mode | precision(correct) | 95% CI | recall(correct) | wrong caught | abstain rate | median s | median tok |
|---|---|---|---|---|---|---|---|
| text | 30.8% | [18.6%, 46.4%] | 54.5% | 11/59 | 33.3% | 88.9 | 4148 |
| vision | 36.0% | [24.1%, 49.9%] | 81.8% | 19/59 | 9.9% | 62.1 | 2739 |

Ground truth: `band == "match"` = CORRECT; built with any other band = WRONG (near_miss/valid/fail all count as wrong here, since the question is exact match, not partial credit); a program that never built is excluded, a crash needs no judge. Three fixture rows (V064, V066 T=0.2, V066 T=0.5) are known-wrong / known-wrong / known-correct by direct inspection of the code, not the card scorer, since they came from a live agreement check outside this card.

**n is small (18-40 per cell in most breakdowns). Every percentage below should be read with its Wilson 95% interval, not as a point estimate.**

## Mode TEXT (text: spec + code + facts)

| truth \ verdict | correct | wrong | abstain |
|---|---|---|---|
| correct | 12 | 4 | 6 |
| wrong | 27 | 11 | 21 |

- Precision of a "correct" verdict: **30.8%** (12/39), Wilson 95% CI [18.6%, 46.4%]
- Recall of "correct" (of the 22 truly-correct parts, how many it confirmed): **54.5%**
- Wrong parts caught (of the 59 truly-wrong parts): **11/59** = 18.6%
- Abstain rate: 33.3% (27/81)
- Median seconds/judgement: 88.9
- Median completion tokens/judgement: 4148

By writer family:
| writer | n | precision | recall | wrong caught | abstain |
|---|---|---|---|---|---|
| gemma | 22 | 30.0% | 75.0% | 3/18 | 40.9% |
| claude | 59 | 31.0% | 50.0% | 8/41 | 30.5% |

By confidence threshold (a "correct" verdict only counts if confidence clears the bar):
| threshold | n confirmed correct | precision | Wilson 95% CI | recall of correct |
|---|---|---|---|---|
| >= 0.6 | 39 | 30.8% | [18.6%, 46.4%] | 54.5% |
| >= 0.8 | 39 | 30.8% | [18.6%, 46.4%] | 54.5% |
| >= 0.9 | 35 | 34.3% | [20.8%, 50.8%] | 54.5% |

## Mode VISION (vision: spec + code + facts + two-panel render)

| truth \ verdict | correct | wrong | abstain |
|---|---|---|---|
| correct | 18 | 4 | 0 |
| wrong | 32 | 19 | 8 |

- Precision of a "correct" verdict: **36.0%** (18/50), Wilson 95% CI [24.1%, 49.9%]
- Recall of "correct" (of the 22 truly-correct parts, how many it confirmed): **81.8%**
- Wrong parts caught (of the 59 truly-wrong parts): **19/59** = 32.2%
- Abstain rate: 9.9% (8/81)
- Median seconds/judgement: 62.1
- Median completion tokens/judgement: 2739

By writer family:
| writer | n | precision | recall | wrong caught | abstain |
|---|---|---|---|---|---|
| gemma | 22 | 30.0% | 75.0% | 9/18 | 9.1% |
| claude | 59 | 37.5% | 83.3% | 10/41 | 10.2% |

By confidence threshold (a "correct" verdict only counts if confidence clears the bar):
| threshold | n confirmed correct | precision | Wilson 95% CI | recall of correct |
|---|---|---|---|---|
| >= 0.6 | 50 | 36.0% | [24.1%, 49.9%] | 81.8% |
| >= 0.8 | 49 | 36.7% | [24.7%, 50.7%] | 81.8% |
| >= 0.9 | 41 | 39.0% | [25.7%, 54.3%] | 72.7% |

## Named cases: V064 and both V066 rows

### V064 (WRONG — lip cutter shears the whole top off, not just the rim)

**Mode text:**
- verdict: `abstain`, confidence: None, problems: None
- verbatim reply:
```
(empty content)
```
**Mode vision:**
- verdict: `wrong`, confidence: 0.95, problems: ['Overall height is 53 mm, not the specified 55 mm', 'Lip cutter is wider than the part and removed the entire top 2 mm face instead of a 6 mm-wide rim recess', '3.2 mm boss holes are blind, not through as specified']
- verbatim reply:
```
Tracing the code against the spec:

- **Overall height:** `outer_box` is 55 mm tall, but the `lip_cut` box is sized `180-2*3.5+2*6 = 185` × `130-7+12 = 135` mm in plan — i.e. *wider than the part itself*. Because it fully covers the top face in plan and is 2 mm tall at the top, subtracting it removes the entire top 2 mm slab, not just a rim recess. The measured bbox confirms this: Z = **53 mm**, not the specified 55 mm (off by 2 mm > 0.5 mm).
- **Stepped lip:** The spec wants a 6 mm‑wide, 2 mm‑deep recessed step around the *inner* rim. The code's cutter is larger than the wall (6 mm > 3.5 mm wall), so it cannot form a contained rim step; it deletes the whole top face. No proper stepped lip exists.
- **Boss holes:** Spec requires a 3.2 mm *through* hole in each boss. The hole cylinder is subtracted only from the boss tube, which sits on the solid 4 mm floor; the bore is capped by the floor and does not pass through. Measured facts confirm `through_holes = 0` and the 3.2 mm bores are blind.

Volume (≈200,641 mm³) is internally consistent with the code as written (top slab removed), confirming the geometry really is 53 mm tall with a missing top face.

These are clear mismatches against stated dimensions and named features.

{"verdict": "wrong", "confidence": 0.95, "problems": ["Overall height is 53 mm, not the specified 55 mm", "Lip cutter is wider than the part and removed the entire top 2 mm face instead of a 6 mm-wide rim recess", "3.2 mm boss holes are blind, not through as specified"]}
```

### V066 T=0.2 (WRONG — Z-axis cylinder makes a vertical slot, not a round hole through the divider)

**Mode text:**
- verdict: `abstain`, confidence: None, problems: None
- verbatim reply:
```
(empty content)
```
**Mode vision:**
- verdict: `wrong`, confidence: 0.8, problems: ['Cable feed-through cutout is a vertical slot (cylinder axis along Z) instead of a 12 mm circular through-hole perpendicular to the divider; facts show 0 through-holes and a 17.5 mm circle rather than a clean 12 mm radial bore']
- verbatim reply:
```
Tracing the code:

- Outer box 140×90×60 → bbox [140,90,60] ✓
- Inner cavity leaves 3 mm walls and a 3 mm floor, open top ✓ (walls list has 3.0)
- Divider 2.5 mm thick, near face 60 mm from the −X end, sitting on the floor → two unequal compartments (60 mm vs ~77.5 mm) ✓ (walls list has 2.5)
- Single fused solid ✓

The defect is the cable feed‑through. The spec calls for a **12 mm diameter circular cutout in the divider** — i.e. a round hole whose axis is perpendicular to the divider face (horizontal, along X, through the 2.5 mm thickness). The code instead builds `Cylinder(radius=6, height=5)` with **no rotation**, so its axis is vertical (Z), parallel to the divider plane. Cutting that vertical cylinder out of the vertical plate yields a small vertical slot (≈2.5 mm × ~11.7 mm × 5 mm) in the middle of the divider, not a 12 mm circular through‑hole. The measured facts corroborate this: `through_holes = 0`, `blind_holes = 2`, and the "bore" is reported with an anomalous `circle_d = 17.5` rather than a clean 12 mm through‑bore. So the named feature (circular feed‑through cutout) has the wrong orientation/shape.

All other dimensions and features match; the only mismatch is the feed‑through cutout.

{"verdict": "wrong", "confidence": 0.8, "problems": ["Cable feed-through cutout is a vertical slot (cylinder axis along Z) instead of a 12 mm circular through-hole perpendicular to the divider; facts show 0 through-holes and a 17.5 mm circle rather than a clean 12 mm radial bore"]}
```

### V066 T=0.5 (CORRECT — cylinder rotated onto the divider's X thickness axis)

**Mode text:**
- verdict: `abstain`, confidence: None, problems: None
- verbatim reply:
```
(empty content)
```
**Mode vision:**
- verdict: `wrong`, confidence: 0.9, problems: ['Floor thickness is 4.5 mm instead of the specified 3 mm wall (cavity offset error; confirmed by volume 142664 mm^3 and walls list containing 4.5)']
- verbatim reply:
```
Tracing the geometry:

- **Overall envelope**: bbox 140×90×60 ✓.
- **Single solid**: solids=1 ✓.
- **Divider**: 2.5 mm thick, full internal height, centered at x≈‑9 (≈60 mm from the left end) ✓; 12 mm bore through it on the X (thin) axis ✓ (bores=[12.0], axial).
- **Wall/floor thickness**: The spec demands 3 mm walls. The cavity is built as `Pos(0,0,wall) * Box(134,84,57)`, i.e. a 57‑mm‑tall box centered at z=+3, so it spans z = ‑25.5 … +31.5. The outer box bottom is at z=‑30, so the floor ends up ‑30 → ‑25.5 = **4.5 mm** thick, not 3 mm. This is confirmed by the measured `walls` list containing 4.5 and by the volume: with a 4.5 mm floor the hand‑calculation gives 131292 + 11372.26 ≈ **142664.3 mm³**, exactly the reported volume (a correct 3 mm floor would give ≈126095 mm³). The comment claims the +wall offset "keeps the floor at wall thickness," but the arithmetic is wrong (it should offset by wall/2 with the chosen box height, or use a box that starts at z=‑27).

So the floor is 1.5 mm too thick — a stated dimension off by more than the 0.5 mm tolerance.

{"verdict": "wrong", "confidence": 0.9, "problems": ["Floor thickness is 4.5 mm instead of the specified 3 mm wall (cavity offset error; confirmed by volume 142664 mm^3 and walls list containing 4.5)"]}
```

# CAPABILITY: Extrude Cut (Subtract Material / Make Pockets, Holes, Slots)

## What it does
Removes material from an existing solid by extruding a sketch profile into it.
Used for: pockets, through-holes, counterbores, slots, grooves, hollowing a container.

## The ONE difference from an additive extrude
Set `operationType` (or `endBound` style) to `"REMOVE"` in the extrude feature.
Everything else is identical to a normal extrude.

## Tool to use
`create_extrude` — same tool, different `operationType` param.

## BTM JSON pattern — extrude cut (REMOVE)
```json
{
  "btType": "BTFeatureDefinitionCall-1406",
  "feature": {
    "btType": "BTMFeature-134",
    "featureType": "extrude",
    "name": "Extrude Cut 1",
    "parameters": [
      {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
       "queries": [{"btType": "BTMIndividualSketchRegionQuery-140",
                    "queryStatement": "qSketchRegion(id + \"sketch1\")"}]},
      {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
       "value": "REMOVE", "enumName": "NewBodyOperationType"},
      {"btType": "BTMParameterEnum-145", "parameterId": "endBound",
       "value": "THROUGH_ALL", "enumName": "BoundingType"},
      {"btType": "BTMParameterBoolean-144", "parameterId": "hasSecondDirection", "value": false}
    ]
  }
}
```

## Common patterns

### Through-hole
```
sketch circle (hole diameter) on a face → create_extrude (REMOVE, THROUGH_ALL)
```

### Blind pocket (fixed depth)
```
sketch rectangle/circle on top face → create_extrude (REMOVE, BLIND, depth=pocket_depth_inches)
endBound: "BLIND"  (not THROUGH_ALL)
expression for depth: "10 * millimeter"
```

### Hollow container (shell via extrude cut)
```
1. Extrude outer solid cylinder (height = container height)
2. Sketch inner circle (OD - 2*wall_thickness) on top face
3. create_extrude REMOVE, depth = (container_height - bottom_thickness), BLIND from top
Result: open-top container with walls and a bottom
```

### Slot / groove
```
sketch elongated shape (rectangle with semicircle ends) → create_extrude REMOVE, THROUGH_ALL or BLIND
```

## Key gotchas
- The sketch for the cut MUST be on an existing face of the solid, not floating in space.
- For a pocket: sketch on the TOP face, extrude DOWN (negative direction or flip = true).
- `THROUGH_ALL` removes material all the way through — use `BLIND` with a depth for pockets.
- `operationType="REMOVE"` requires an existing body to cut into — must come AFTER the first solid.
- The `sketchFeatureId` (or `entities` query) references the sketch by its feature ID string.
- If the sketch has multiple regions, use `qNthElement(qSketchRegion(id + "sketch2"), 0)` to pick one.

# CAPABILITY: Thread Creation on a Cylindrical Face

## What it does
Cuts helical threads into or onto a cylindrical solid body (boss or bore).
Produces both internal (nut) and external (bolt/rod) threads.

## Prerequisites
1. A cylindrical solid face must already exist — create via `create_extrude` on a circular sketch.
2. The cylindrical diameter must match the thread standard (M8 = 8mm diameter, etc.).

## Tool sequence
```
create_document → get_part_studio
→ create_sketch (circle, diameter = nominal thread diameter in inches)
→ create_extrude (depth = thread length in inches, operationType = "NEW")
→ thread feature (see BTM below)
→ get_mass_properties
```

## Thread BTM JSON pattern (Onshape built-in Thread feature)
```json
{
  "btType": "BTFeatureDefinitionCall-1406",
  "feature": {
    "btType": "BTMFeature-134",
    "featureType": "thread",
    "name": "Thread 1",
    "parameters": [
      {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
       "queries": [{"btType": "BTMIndividualQuery-138", "queryStatement": "qCylindrical()"}]},
      {"btType": "BTMParameterBoolean-144", "parameterId": "rightHand", "value": true},
      {"btType": "BTMParameterBoolean-144", "parameterId": "isExternal", "value": false},
      {"btType": "BTMParameterString-149", "parameterId": "threadStandard", "value": "ISO metric"},
      {"btType": "BTMParameterString-149", "parameterId": "threadType", "value": "M8"},
      {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
       "expression": "10 * millimeter"},
      {"btType": "BTMParameterBoolean-144", "parameterId": "fullThread", "value": true}
    ]
  }
}
```

## ThreadCreator plugin (custom FeatureScript alternative)
If the built-in thread is insufficient, use the ThreadCreator Onshape plugin:
- Install via Onshape App Store: "Thread Creator" by `cad.onshape.com`
- It exposes a FeatureScript function `thread(context, id, definition)` with params:
  `diameter`, `pitch`, `depth`, `internal` (bool), `rightHand` (bool)
- Prefer built-in thread feature for standard ISO/UNC threads.
- Use ThreadCreator for custom pitch, non-standard profiles, or knurling patterns.

## Threading a lid (container + threaded lid pattern)
```
1. create_document → get_part_studio
2. Sketch circle (container OD) → extrude (container body height) [operationType=NEW]
3. Sketch circle (container ID = OD - 2*wall) → extrude-cut (shell interior) [operationType=REMOVE]
4. Apply thread on top inner bore (isExternal=false)
5. New sketch circle (lid OD = container OD) → extrude (lid height) [operationType=NEW, separate body]
6. Apply thread on lid outer boss (isExternal=true, matching pitch)
7. get_mass_properties
```

## Key gotchas
- Thread diameter is nominal — the cylindrical face diameter must match the thread standard.
- `isExternal=false` → internal (nut-style). `isExternal=true` → external (bolt-style).
- Units: `expression` field uses FeatureScript unit strings like `"8 * millimeter"`.
- The `entities` query must resolve to exactly one cylindrical face — use `qCylindrical()` or `qNthElement(qCylindrical(), 0)` if multiple cylinders exist.
- Right-hand thread is standard; left-hand needs `"rightHand": false`.

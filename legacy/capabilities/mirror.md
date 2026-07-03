# CAPABILITY: Mirror Feature / Mirror Part

## What it does
Creates a mirrored copy of one or more solid bodies about a plane.
Used to: make symmetric parts from half-geometry, duplicate a feature on both sides,
create symmetric assemblies (left/right brackets, symmetric housings).

## Two modes
1. **Mirror bodies** — mirrors entire solid bodies across a plane (most common)
2. **Mirror features** — mirrors individual sketch/extrude features (re-runs them mirrored)

## Tool to use
`mirror_part` — available in the cad_agent_v2.py tool schema.

## BTM JSON pattern — mirror body about a standard plane
```json
{
  "btType": "BTFeatureDefinitionCall-1406",
  "feature": {
    "btType": "BTMFeature-134",
    "featureType": "mirror",
    "name": "Mirror 1",
    "parameters": [
      {"btType": "BTMParameterEnum-145", "parameterId": "mirrorType",
       "value": "PART", "enumName": "MirrorType"},
      {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
       "queries": [{"btType": "BTMIndividualQuery-138",
                    "queryStatement": "qAllSolidBodies()"}]},
      {"btType": "BTMParameterQueryList-148", "parameterId": "mirrorPlane",
       "queries": [{"btType": "BTMIndividualQuery-138",
                    "queryStatement": "qGeometry(context, EntityType.PLANE, {\"queryType\": \"XZ\"})"}]}
    ]
  }
}
```

## Standard mirror planes
- `YZ` plane (mirror left↔right): `queryType: "YZ"`
- `XZ` plane (mirror front↔back): `queryType: "XZ"`  
- `XY` plane (mirror top↔bottom): `queryType: "XY"`

Or use a flat face of the part as the mirror plane:
```json
{"queryStatement": "qGeometry(qBodyType(qEverything(), BodyType.SOLID), GeometryType.PLANE)"}
```

## Common patterns

### Symmetric bracket (build half, mirror)
```
1. create_document → get_part_studio
2. Sketch half-profile → extrude (solid half-body)
3. Add holes/features on one side
4. mirror_part: entities=qAllSolidBodies(), mirrorPlane=YZ
5. get_mass_properties
```

### Symmetric gear pair (already handled by create_gear, but if custom)
```
1. Build one gear body
2. Mirror about XZ plane to place second gear on opposite side
Note: mirror creates a symmetric copy — if you need meshing gears at offset,
use create_gear twice with centerX offset instead of mirror.
```

### Mirror a cut feature (both sides simultaneously)
```
1. Sketch slot/hole on ONE side
2. create_extrude REMOVE (cuts one side)
3. mirror: mirrorType="FEATURE" to repeat the cut on the mirrored side
   — OR — mirror the resulting body after all cuts are done
```

## Key gotchas
- Mirror creates a NEW body — the original body is KEPT. If you want only the mirrored result,
  delete the original body after mirroring.
- `qAllSolidBodies()` mirrors everything — if you have multiple bodies, specify them with
  `qNthElement(qAllSolidBodies(), 0)`.
- The mirror plane query must resolve to exactly ONE plane.
- For a feature mirror (`mirrorType="FEATURE"`), reference the specific feature ID, not a body.
- Mirror does NOT automatically boolean-union the original and mirrored copies —
  use a separate boolean union step if one merged body is needed.

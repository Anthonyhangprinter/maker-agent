# CAPABILITY: Fillet and Chamfer (Edge Breaks)

## What they do
- **Fillet**: rounds sharp edges with a circular arc of given radius.
- **Chamfer**: cuts a flat angled face on sharp edges (45° default or custom angle).

Both operate on edges of existing solid bodies. Always applied AFTER geometry is created.

## Tool to use
`create_fillet` — handles both fillet and chamfer.

## BTM JSON pattern — fillet
```json
{
  "btType": "BTFeatureDefinitionCall-1406",
  "feature": {
    "btType": "BTMFeature-134",
    "featureType": "fillet",
    "name": "Fillet 1",
    "parameters": [
      {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
       "queries": [{"btType": "BTMIndividualQuery-138",
                    "queryStatement": "qEverything()->qGeometry(GeometryType.LINE)"}]},
      {"btType": "BTMParameterQuantity-147", "parameterId": "radius",
       "expression": "2 * millimeter"},
      {"btType": "BTMParameterBoolean-144", "parameterId": "tangentPropagation", "value": true}
    ]
  }
}
```

## BTM JSON pattern — chamfer (equal distance)
```json
{
  "btType": "BTFeatureDefinitionCall-1406",
  "feature": {
    "btType": "BTMFeature-134",
    "featureType": "chamfer",
    "name": "Chamfer 1",
    "parameters": [
      {"btType": "BTMParameterEnum-145", "parameterId": "chamferType",
       "value": "EQUAL_OFFSETS", "enumName": "ChamferType"},
      {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
       "queries": [{"btType": "BTMIndividualQuery-138",
                    "queryStatement": "qEverything()->qGeometry(GeometryType.LINE)"}]},
      {"btType": "BTMParameterQuantity-147", "parameterId": "width",
       "expression": "1 * millimeter"}
    ]
  }
}
```

## Edge selection patterns

### All edges of a body
`qEverything()->qGeometry(GeometryType.LINE)`

### Only top face edges
`qGeometry(qFarthestAlong(qAllSolidBodies(), vector(0, 1, 0)), GeometryType.LINE)`

### Specific edge by index (when you need one edge, not all)
`qNthElement(qEverything()->qGeometry(GeometryType.LINE), 0)`

### Edges of a specific face
`qGeometry(qNthElement(qAllFaces(), 0), GeometryType.LINE)`

## Common patterns

### Round all edges after extrude (cosmetic fillet)
```
create_extrude → create_fillet (all edges, small radius like 0.5mm)
```

### Sharp structural part — chamfer only the entry edges
```
Chamfer only the top rim: qGeometry(qFarthestAlong(..., vector(0,1,0)), GeometryType.LINE)
```

### Container lid — fillet outer top edge for ergonomics
```
After creating lid cylinder, fillet top circular edge with 1-2mm radius
```

## Key gotchas
- Fillet FAILS if the radius is larger than the shortest edge — use smaller radius.
- `tangentPropagation: true` propagates fillet around tangent edges (smooth surfaces) automatically.
- Chamfer and fillet cannot be applied to the same edge in one feature — use two steps.
- If fillet fails, try selecting fewer edges (not `qEverything()`).
- Radius is in inches in the cad_agent tool schema BUT in the BTM `expression` field use FeatureScript unit syntax: `"2 * millimeter"` or `"0.1 * inch"`.

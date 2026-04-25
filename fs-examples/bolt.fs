/**
 * BOLT EXAMPLE — ISO metric hex-head bolt
 * Use for: M-series bolts, hex-head fasteners.
 *
 * Pattern: hex head extruded from Front plane, shank sketched on head END face
 * using qCapEntity so ADD merges correctly.
 */
FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// M18 hex-head bolt: d=18mm  s=27mm  k=11.5mm  L=80mm
annotation { "Feature Type Name" : "M18 Bolt" }
export const boltFeature = defineFeature(function(context is Context, id is Id, definition is map)
    precondition {}
    {
        // ── Hex Head ──────────────────────────────────────────────────────────────
        var skHead = newSketchOnPlane(context, id + "skHead", {
            "sketchPlane" : plane(WORLD_ORIGIN, Z_DIRECTION)
        });
        skRegularPolygon(skHead, "hex", {
            "center"      : vector(0, 0) * millimeter,
            "firstVertex" : vector(15.5885, 0) * millimeter,
            "sides"       : 6
        });
        skSolve(skHead);

        opExtrude(context, id + "head", {
            "entities"      : qSketchRegion(id + "skHead"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 11.5000 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        // ── Shank: sketched on END face of head so ADD works reliably ─────────────
        var headEndFace = qCapEntity(id + "head", CapType.END);
        var skShank = newSketchOnPlane(context, id + "skShank", {
            "sketchPlane" : evFacePlane(context, { "face" : headEndFace })
        });
        skCircle(skShank, "shank", {
            "center" : vector(0, 0) * millimeter,
            "radius" : 9.0000 * millimeter
        });
        skSolve(skShank);

        opExtrude(context, id + "shank", {
            "entities"      : qSketchRegion(id + "skShank"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 80.0000 * millimeter,
            "operationType" : NewBodyOperationType.ADD
        });
    }
);

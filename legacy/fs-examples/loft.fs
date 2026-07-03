/**
 * LOFT EXAMPLE — Smoothly transition between two profiles
 * Use for: spoons, handles, tapered parts, organic shapes, boat hulls, wings.
 *
 * Pattern: sketch profile A on plane at z=0, sketch profile B on plane at z=height,
 * then opLoft connects them with a smooth solid.
 * All dimensions hardcoded — empty precondition.
 */
FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// Example: a table spoon — elliptical bowl lofted to a flat oval handle end
annotation { "Feature Type Name" : "Spoon" }
export const makeSpoon = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        // --- Dimensions ---
        var bowlMajor    = 40  * millimeter;   // bowl half-length (X)
        var bowlMinor    = 28  * millimeter;   // bowl half-width (Y)
        var neckMajor    = 8   * millimeter;   // neck half-length
        var neckMinor    = 5   * millimeter;   // neck half-width
        var handleMajor  = 6   * millimeter;   // handle end half-length
        var handleMinor  = 3   * millimeter;   // handle end half-width
        var bowlDepth    = 8   * millimeter;   // how deep the bowl dips below z=0
        var neckZ        = 60  * millimeter;   // z-height of the neck cross-section
        var handleEndZ   = 200 * millimeter;   // z-height of handle tip

        // Profile 1: bowl ellipse at z=0
        var planeBowl = plane(WORLD_ORIGIN, Z_DIRECTION);
        var skBowl = newSketchOnPlane(context, id + "bowl", { "sketchPlane" : planeBowl });
        skEllipse(skBowl, "bowl_ellipse", {
            "center"      : vector(0 * millimeter, 0 * millimeter),
            "majorRadius" : bowlMajor,
            "minorRadius" : bowlMinor
        });
        skSolve(skBowl);

        // Profile 2: narrow neck at z=neckZ
        var planeNeck = plane(vector(0, 0, neckZ / millimeter) * millimeter, Z_DIRECTION);
        var skNeck = newSketchOnPlane(context, id + "neck", { "sketchPlane" : planeNeck });
        skEllipse(skNeck, "neck_ellipse", {
            "center"      : vector(0 * millimeter, 0 * millimeter),
            "majorRadius" : neckMajor,
            "minorRadius" : neckMinor
        });
        skSolve(skNeck);

        // Profile 3: flattened oval at handle end
        var planeHandle = plane(vector(0, 0, handleEndZ / millimeter) * millimeter, Z_DIRECTION);
        var skHandle = newSketchOnPlane(context, id + "handle", { "sketchPlane" : planeHandle });
        skEllipse(skHandle, "handle_ellipse", {
            "center"      : vector(0 * millimeter, 0 * millimeter),
            "majorRadius" : handleMajor,
            "minorRadius" : handleMinor
        });
        skSolve(skHandle);

        // Loft through all three profiles
        opLoft(context, id + "loft", {
            "profileSubqueries" : [
                qSketchRegion(id + "bowl"),
                qSketchRegion(id + "neck"),
                qSketchRegion(id + "handle")
            ]
        });
    }
);

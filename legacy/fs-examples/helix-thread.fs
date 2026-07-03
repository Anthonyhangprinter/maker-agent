/**
 * HELIX + SWEEP EXAMPLE — Helical thread (bolt, screw, worm gear, spring)
 * Use for: threaded fasteners, coil springs, helical gears, corkscrews.
 *
 * Pattern:
 *   1. Create bolt body (cylinder) with opExtrude
 *   2. Create helix path with opHelix
 *   3. Sketch thread profile (V-groove triangle) on a plane at the helix start
 *   4. opSweep along the helix to cut the thread groove
 * All dimensions hardcoded — empty precondition.
 */
FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// Example: M10 bolt thread — 10mm nominal, 1.5mm pitch, 30mm thread length
annotation { "Feature Type Name" : "Threaded Bolt" }
export const threadedBolt = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        var d        = 10   * millimeter;   // nominal diameter
        var pitch    = 1.5  * millimeter;   // thread pitch
        var length   = 30   * millimeter;   // thread length
        var headH    = 6.4  * millimeter;   // head height (M10)
        var headAF   = 17   * millimeter;   // head across flats (M10)

        var r        = d / 2;
        var rMinor   = r - 0.6134 * pitch; // ISO minor radius
        var numRevs  = length / pitch;

        // 1. Cylindrical shank
        var skShank = newSketchOnPlane(context, id + "shankSk", {
            "sketchPlane" : plane(WORLD_ORIGIN, Z_DIRECTION)
        });
        skCircle(skShank, "shankCirc", {
            "center" : vector(0 * millimeter, 0 * millimeter),
            "radius" : r
        });
        skSolve(skShank);
        opExtrude(context, id + "shank", {
            "entities"  : qSketchRegion(id + "shankSk"),
            "direction" : Z_DIRECTION,
            "endBound"  : BoundingType.BLIND,
            "endDepth"  : length
        });

        // 2. Helix path on the cylinder surface
        opHelix(context, id + "helix", {
            "direction"     : Z_DIRECTION,
            "axisStart"     : WORLD_ORIGIN,
            "startPoint"    : vector(r, 0 * millimeter, 0 * millimeter),
            "revolutions"   : numRevs,
            "height"        : length,
            "clockwise"     : false
        });

        // 3. Thread profile: 60° V-groove triangle, at helix start point
        //    Sketch on a radial plane through the helix start
        var skThread = newSketchOnPlane(context, id + "threadSk", {
            "sketchPlane" : plane(
                vector(r, 0 * millimeter, 0 * millimeter),
                Y_DIRECTION
            )
        });
        var halfPitch = pitch / 2;
        var depth     = pitch * 0.6134;   // ISO thread depth
        // Triangle vertices in sketch space: (z_offset, radial_depth)
        skLineSegment(skThread, "t1", {
            "start" : vector(-halfPitch, 0 * millimeter),
            "end"   : vector(0 * millimeter, -depth)
        });
        skLineSegment(skThread, "t2", {
            "start" : vector(0 * millimeter, -depth),
            "end"   : vector(halfPitch, 0 * millimeter)
        });
        skLineSegment(skThread, "t3", {
            "start" : vector(halfPitch, 0 * millimeter),
            "end"   : vector(-halfPitch, 0 * millimeter)
        });
        skSolve(skThread);

        // 4. Sweep thread profile along helix to cut groove
        opSweep(context, id + "threadCut", {
            "profiles"        : qSketchRegion(id + "threadSk"),
            "path"            : qCreatedBy(id + "helix", EntityType.EDGE),
            "operationType"   : NewBodyOperationType.REMOVE
        });
    }
);

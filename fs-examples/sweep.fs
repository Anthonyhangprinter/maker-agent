/**
 * SWEEP EXAMPLE — Profile swept along a path
 * Use for: pipes, tubes, bent rods, door handles, hooks, curved extrusions.
 *
 * Pattern:
 *   1. Sketch the path curve on one plane
 *   2. Sketch the cross-section profile on a perpendicular plane at the path start
 *   3. opSweep moves the profile along the path
 * All dimensions hardcoded — empty precondition.
 */
FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// Example: a pipe elbow — 90° bend, 25mm OD, 2mm wall, 80mm bend radius
annotation { "Feature Type Name" : "Pipe Elbow" }
export const pipeElbow = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        var outerRadius = 12.5 * millimeter;   // 25mm OD
        var innerRadius = 10.5 * millimeter;   // 25mm OD - 2mm wall = 21mm ID
        var bendRadius  = 80   * millimeter;   // centreline bend radius
        // The path is a 90° arc in the XZ plane, centred at (bendRadius, 0, 0)

        // 1. Path: 90° arc in the XZ plane
        //    Arc goes from (0, 0, 0) around (bendRadius, 0, 0) to (bendRadius, 0, bendRadius)
        var skPath = newSketchOnPlane(context, id + "pathSk", {
            "sketchPlane" : plane(WORLD_ORIGIN, Y_DIRECTION)   // XZ plane
        });
        skArc(skPath, "bend_arc", {
            "start"  : vector(0 * millimeter, 0 * millimeter),
            "mid"    : vector(bendRadius - bendRadius * (1 - cos(45 * degree)),
                             bendRadius * sin(45 * degree)),
            "end"    : vector(bendRadius, bendRadius)
        });
        skSolve(skPath);

        // 2. Cross-section: annular ring at start of path (z=0, x=0)
        //    The profile plane is perpendicular to the path at its start point.
        //    Path starts going in the Z direction, so profile plane has Z_DIRECTION normal.
        var skProfile = newSketchOnPlane(context, id + "profileSk", {
            "sketchPlane" : plane(WORLD_ORIGIN, Z_DIRECTION)
        });
        skCircle(skProfile, "outer", {
            "center" : vector(0 * millimeter, 0 * millimeter),
            "radius" : outerRadius
        });
        skCircle(skProfile, "inner", {
            "center" : vector(0 * millimeter, 0 * millimeter),
            "radius" : innerRadius
        });
        skSolve(skProfile);

        // 3. Sweep profile along arc path
        opSweep(context, id + "sweep", {
            "profiles" : qSketchRegion(id + "profileSk"),
            "path"     : qCreatedBy(id + "pathSk", EntityType.EDGE)
        });
    }
);

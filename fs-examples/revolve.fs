/**
 * REVOLVE EXAMPLE — Solid of revolution (turned part)
 * Use for: shafts, pins, knobs, discs, cups, domes, stepped cylinders.
 *
 * Pattern: sketch half-profile on a plane, revolve 360° around an axis.
 * The axis is the Y axis (left edge of the half-profile) in this example.
 * All dimensions hardcoded — empty precondition.
 */
FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

// Example: a stepped shaft — 30mm dia shank, 50mm dia flange, 120mm total length
annotation { "Feature Type Name" : "Stepped Shaft" }
export const steppedShaft = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        var shankRadius  = 15  * millimeter;   // 30mm diameter shank
        var flangeRadius = 25  * millimeter;   // 50mm diameter flange
        var shankLength  = 90  * millimeter;   // shank portion height
        var flangeThick  = 15  * millimeter;   // flange thickness
        var chamfer      = 2   * millimeter;   // chamfer on shank end

        // Sketch half-profile on the YZ plane (normal = X_DIRECTION)
        // Sketch X axis = Z (length), Sketch Y axis = Y (radius)
        var sk = newSketchOnPlane(context, id + "profile", {
            "sketchPlane" : plane(WORLD_ORIGIN, X_DIRECTION)
        });

        // Half-profile points (radius, length) — closed polygon, revolution axis at radius=0
        // Bottom of shaft at z=0; top of flange at z=shankLength+flangeThick
        skLineSegment(sk, "axis_bottom",   { "start" : vector(0 * millimeter, 0 * millimeter),             "end" : vector(0 * millimeter, shankLength + flangeThick) });
        skLineSegment(sk, "flange_top",    { "start" : vector(0 * millimeter, shankLength + flangeThick),  "end" : vector(flangeRadius, shankLength + flangeThick) });
        skLineSegment(sk, "flange_outer",  { "start" : vector(flangeRadius, shankLength + flangeThick),    "end" : vector(flangeRadius, shankLength) });
        skLineSegment(sk, "flange_step",   { "start" : vector(flangeRadius, shankLength),                  "end" : vector(shankRadius + chamfer, shankLength) });
        skLineSegment(sk, "chamfer_line",  { "start" : vector(shankRadius + chamfer, shankLength),         "end" : vector(shankRadius, shankLength - chamfer) });
        skLineSegment(sk, "shank_outer",   { "start" : vector(shankRadius, shankLength - chamfer),         "end" : vector(shankRadius, 0 * millimeter) });
        skLineSegment(sk, "bottom_face",   { "start" : vector(shankRadius, 0 * millimeter),                "end" : vector(0 * millimeter, 0 * millimeter) });
        skSolve(sk);

        // Revolve 360° around Z axis (the revolution axis in sketch space)
        opRevolve(context, id + "revolve", {
            "entities"     : qSketchRegion(id + "profile"),
            "axis"         : line(WORLD_ORIGIN, Z_DIRECTION),
            "angleForward" : 2 * PI * radian
        });
    }
);

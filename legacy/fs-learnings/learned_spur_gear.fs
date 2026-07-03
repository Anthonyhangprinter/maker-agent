// Learned from successful build: make a spur gear 20 teeth module 2 meshing with a pinion at 3:1 ratio
// Part: Spur Gear  |  Feature: makeSpurGear

FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

annotation { "Feature Type Name" : "Spur Gear" }
export const makeSpurGear = defineFeature(function(context is Context, id is Id, definition is map)
    precondition {}
    {
        var numTeeth = 20;
        var module = 2 * millimeter;
        var pitchCircleRadius = numTeeth * module / 2;

        var skBase = newSketchOnPlane(context, id + "skBase", { "sketchPlane" : plane(WORLD_ORIGIN, Z_DIRECTION) });
        for (var i = 0; i < numTeeth; ++i)
        {
            var angle = i * 2 * PI / numTeeth;
            skLineSegment(skBase, "tooth_
// Learned from successful build: make a chair
// Part: chair  |  Feature: makeChair

FeatureScript 2454;
import(path : "onshape/std/geometry.fs", version : "2454.0");

annotation { "Feature Type Name" : "Chair" }
export const makeChair = defineFeature(function(context is Context, id is Id, definition is map)
    precondition {}
    {
        // --- Dimensions ---
        var seatWidth  = 600 * millimeter;
        var seatDepth  = 450 * millimeter;
        var seatHeight = 250 * millimeter;
        var backHeight = 800 * millimeter;
        var legLength  = 900 * millimeter;
        var legWidth   = 30 * millimeter;
        var legDepth   = 30 * millimeter;

        // --- Seat ---
        var planeSeat = plane(vector(0, 0, seatHeight), Z_DIRECTION);
        var skSeat = newSketchOnPlane(context, id + "seat", { "sketchPlane" : planeSeat });
        skRectangle(skSeat, "rectangle", {
            "length" : seatWidth,
            "width"  : seatDepth
        });
        skSolve(skSeat);

        var seatBody = opExtrude(context, id + "seatBody", {
            "entities"      : qSketchRegion(id + "seat"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 250 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        // --- Backrest ---
        var planeBack = plane(vector(0, seatWidth/2, seatHeight), Z_DIRECTION);
        var skBack = newSketchOnPlane(context, id + "back", { "sketchPlane" : planeBack });
        skRectangle(skBack, "rectangle", {
            "length" : backHeight,
            "width"  : seatWidth
        });
        skSolve(skBack);

        var backBody = opExtrude(context, id + "backBody", {
            "entities"      : qSketchRegion(id + "back"),
            "direction"     : Y_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 300 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        // --- Legs ---
        var skLeg = newSketchOnPlane(context, id + "leg", { "sketchPlane" : plane(WORLD_ORIGIN, Y_DIRECTION) });
        skRectangle(skLeg, "rectangle", {
            "length" : legWidth,
            "width"  : legDepth
        });
        skSolve(skLeg);

        var frontLeftLeg = opExtrude(context, id + "frontLeftLeg", {
            "entities"      : qSketchRegion(id + "leg"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 900 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        var frontRightLeg = opExtrude(context, id + "frontRightLeg", {
            "entities"      : qSketchRegion(id + "leg"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 900 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        var backLeftLeg = opExtrude(context, id + "backLeftLeg", {
            "entities"      : qSketchRegion(id + "leg"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 900 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        var backRightLeg = opExtrude(context, id + "backRightLeg", {
            "entities"      : qSketchRegion(id + "leg"),
            "direction"     : Z_DIRECTION,
            "endBound"      : BoundingType.BLIND,
            "endDepth"      : 900 * millimeter,
            "operationType" : NewBodyOperationType.NEW
        });

        // --- Assemble chair ---(
        var seatLegs = opBoolean(context, id + "seatLegs", {
            "tools"         : qCreatedBy(id + "frontLeftLeg", EntityType.BODY),
            "targets"       : qCreatedBy(id + "seatBody", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

        var seatBacklegs = opBoolean(context, id + "seatBacklegs", {
            "tools"         : qCreatedBy(id + "frontRightLeg", EntityType.BODY),
            "targets"       : qCreatedBy(id + "backBody", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

        var seatBackLegs = opBoolean(context, id + "seatBackLegs", {
            "tools"         : qCreatedBy(id + "backLeftLeg", EntityType.BODY),
            "targets"       : qCreatedBy(id + "seatBody", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

        var seatBackRightLegs = opBoolean(context, id + "seatBackRightLegs", {
            "tools"         : qCreatedBy(id + "backRightLeg", EntityType.BODY),
            "targets"       : qCreatedBy(id + "seatBody", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

        var chair = opBoolean(context, id + "chair", {
            "tools"         : qCreatedBy(id + "seatBackRightLegs", EntityType.BODY),
            "targets"       : qCreatedBy(id + "seatBacklegs", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

        var finalChair = opBoolean(context, id + "finalChair", {
            "tools"         : qCreatedBy(id + "seatLegs", EntityType.BODY),
            "targets"       : qCreatedBy(id + "chair", EntityType.BODY),
            "operationType" : BooleanOperationType.UNION
        });

    }
);
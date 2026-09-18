from build123d import *

plate_length = 0.875
plate_thickness = 0.15
plate_height = 1.5
corner_radius = 0.25
hole_diameter = 0.3

hole_x = plate_length / 2 - corner_radius
hole_z = plate_height / 2 - corner_radius
cutter_length = plate_thickness * 4

result = Box(plate_length, plate_thickness, plate_height)

for x in (-hole_x, hole_x):
    result -= Pos(x, 0, hole_z) * Rotation(90, 0, 0) * Cylinder(radius=hole_diameter / 2, height=cutter_length)

result = fillet(result.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1], radius=corner_radius)

from build123d import *

plate_length = 0.875
plate_thickness = 0.15
plate_height = 1.5
corner_radius = 0.25
hole_d = 0.3
hole_padding = -0.639071

# plate stands upright: length along X, thickness along Y, height along Z
plate = Box(plate_length, plate_thickness, plate_height)

# hole centres sit on the corners of a (length + padding) x (height + padding) rectangle;
# only the two on the rounded (top) side are drilled
hole_x = (plate_length + hole_padding) / 2
hole_z = (plate_height + hole_padding) / 2

for sx in (-1, 1):
    plate -= Pos(sx * hole_x, 0, hole_z) * Rot(90, 0, 0) * Cylinder(
        radius=hole_d / 2, height=plate_thickness * 4
    )

# round the two corners of the top long edge (the edges running through the thickness)
top_corner_edges = plate.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1]
result = fillet(top_corner_edges, radius=corner_radius)

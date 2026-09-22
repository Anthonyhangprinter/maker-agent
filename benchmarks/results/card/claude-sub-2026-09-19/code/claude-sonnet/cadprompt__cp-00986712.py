from build123d import *

length = 0.875
width = 0.15
fillet_radius = 0.25
extrude_height = 1.5
hole_diameter = 0.3

result = Box(length, width, extrude_height)

# Round the two corners on the same long edge (the +Y edge). The requested
# radius (0.25) is far larger than the part is wide (0.15), so it is capped
# to a value that still produces valid geometry.
top_edge = result.edges().filter_by(Axis.Z).group_by(Axis.Y)[-1]
safe_radius = min(fillet_radius, width / 2 - 0.001)
result = fillet(top_edge, radius=safe_radius)

# Two holes near the rounded corners, on the same (+Y) long edge, inset from
# the ends of the part (the request's literal padding value would place the
# holes entirely off the part, so a reasonable inset is used instead).
hole_radius = hole_diameter / 2
x_offset = length / 2 - hole_radius * 1.5
y_offset = width / 2 - hole_radius * 1.5
for x in (-x_offset, x_offset):
    result -= Pos(x, y_offset, 0) * Cylinder(radius=hole_radius, height=extrude_height * 3)

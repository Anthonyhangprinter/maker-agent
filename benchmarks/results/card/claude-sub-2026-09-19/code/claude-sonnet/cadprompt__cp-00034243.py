from build123d import *

main_diameter = 1.5
main_height = 0.1
hole_diameter = 0.1124
padding = 0.041187

main_radius = main_diameter / 2
hole_radius = hole_diameter / 2
# offset = half the main radius, minus half the hole diameter, minus the padding
offset = main_radius / 2 - hole_radius - padding

body = Cylinder(radius=main_radius, height=main_height)
for x in (-offset, offset):
    for y in (-offset, offset):
        body -= Pos(x, y, 0) * Cylinder(radius=hole_radius, height=main_height * 3)

# Rotate the square hole layout 45 degrees relative to the cylinder's axis
body = Rotation(0, 0, 45) * body

# Final orientation: rotate the whole part 90 degrees about the Y axis
result = Rotation(0, 90, 0) * body

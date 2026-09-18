from build123d import *

# PARAMETERS
main_d = 1.5
main_h = 0.1
hole_d = 0.1124
padding = 0.041187

# Calculations for hole positioning
# Offset = (main_radius - hole_radius - padding)
# main_radius = 1.5 / 2 = 0.75
# hole_radius = 0.1124 / 2 = 0.0562
# offset = 0.75 - 0.0562 - 0.041187 = 0.652613
hole_offset = (main_d / 2) - (hole_d / 2) - padding

# Create the main cylinder
main_body = Cylinder(radius=main_d / 2, height=main_h)

# Create the four holes in a square layout
# Holes are placed at (+/- offset, +/- offset)
holes = (
    Pos(hole_offset, hole_offset, 0) * Cylinder(radius=hole_d / 2, height=main_h * 2) +
    Pos(hole_offset, -hole_offset, 0) * Cylinder(radius=hole_d / 2, height=main_h * 2) +
    Pos(-hole_offset, hole_offset, 0) * Cylinder(radius=hole_d / 2, height=main_h * 2) +
    Pos(-hole_offset, -hole_offset, 0) * Cylinder(radius=hole_d / 2, height=main_h * 2)
)

# Subtract holes from body and rotate the configuration by 45 degrees around Z
# The request asks to rotate the "entire configuration"
part_z_aligned = Rotation(0, 0, 45) * (main_body - holes)

# Finally, rotate the part 90 degrees around the Y-axis
# This puts the cylinder axis along the X-axis
result = Rotation(0, 90, 0) * part_z_aligned
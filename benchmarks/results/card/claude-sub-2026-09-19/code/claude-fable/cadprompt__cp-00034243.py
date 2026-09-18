from build123d import *

disc_d = 1.5
disc_height = 0.1
hole_d = 0.1124
padding = 0.041187
layout_rotation_deg = 45.0

disc_r = disc_d / 2
# offset = half the main radius - half the hole diameter - padding
hole_offset = disc_r / 2 - hole_d / 2 - padding

disc = Cylinder(radius=disc_r, height=disc_height)
for sx in (-1, 1):
    for sy in (-1, 1):
        disc -= Pos(sx * hole_offset, sy * hole_offset, 0) * Cylinder(
            radius=hole_d / 2, height=disc_height * 4
        )

# rotate the hole square 45 degrees about the disc axis, then lay the disc axis along X
disc = Rot(0, 0, layout_rotation_deg) * disc
result = Rot(0, 90, 0) * disc

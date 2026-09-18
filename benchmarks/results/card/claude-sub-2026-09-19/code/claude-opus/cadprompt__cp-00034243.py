from build123d import *
import math

main_diameter = 1.5
main_height = 0.1
hole_diameter = 0.1124
hole_padding = 0.041187
hole_count = 4
layout_rotation_deg = 45.0
final_y_rotation_deg = 90.0

main_radius = main_diameter / 2
hole_radius = hole_diameter / 2
hole_offset = main_radius / 2 - hole_diameter / 2 - hole_padding
cutter_height = main_height * 4

part = Cylinder(radius=main_radius, height=main_height)

for i in range(hole_count):
    angle = math.radians(layout_rotation_deg + i * 360.0 / hole_count)
    x = hole_offset * math.cos(angle)
    y = hole_offset * math.sin(angle)
    part -= Pos(x, y, 0) * Cylinder(radius=hole_radius, height=cutter_height)

result = Rotation(0, final_y_rotation_deg, 0) * part

from build123d import *

block_length = 0.75
block_width = 0.1875
block_height = 0.00391
hole_diameter = 0.12512

hole_radius = hole_diameter / 2
cutter_height = block_height * 4

result = Box(block_length, block_width, block_height)
result -= Pos(0, 0, 0) * Cylinder(radius=hole_radius, height=cutter_height)

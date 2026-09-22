from build123d import *

block_length = 0.75
block_width = 0.1875
block_height = 0.00391
hole_d = 0.12512

plate = Box(block_length, block_width, block_height)
hole = Pos(0, 0, 0) * Cylinder(radius=hole_d / 2, height=block_height * 10)
result = plate - hole

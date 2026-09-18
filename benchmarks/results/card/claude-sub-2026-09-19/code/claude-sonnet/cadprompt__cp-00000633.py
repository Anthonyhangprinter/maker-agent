from build123d import *

length = 0.75
width = 0.1875
height = 0.00391
hole_diameter = 0.12512

hole_height = height * 4
result = Box(length, width, height) - Cylinder(radius=hole_diameter / 2, height=hole_height)

from build123d import *

base_diameter = 0.567213
base_height = 0.5625
top_diameter = 0.266022
top_height = 0.1875

base = Pos(0, 0, base_height / 2) * Cylinder(radius=base_diameter / 2, height=base_height)
top = Pos(0, 0, base_height + top_height / 2) * Cylinder(radius=top_diameter / 2, height=top_height)

result = base + top

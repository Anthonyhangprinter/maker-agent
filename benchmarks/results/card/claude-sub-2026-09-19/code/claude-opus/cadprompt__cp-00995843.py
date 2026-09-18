from build123d import *

lower_diameter = 0.567213
lower_height = 0.5625
upper_diameter = 0.266022
upper_height = 0.1875

total_height = lower_height + upper_height

lower = Pos(0, 0, lower_height / 2) * Cylinder(radius=lower_diameter / 2, height=lower_height)
upper = Pos(0, 0, lower_height + upper_height / 2) * Cylinder(radius=upper_diameter / 2, height=upper_height)

result = Pos(0, 0, -total_height / 2) * (lower + upper)

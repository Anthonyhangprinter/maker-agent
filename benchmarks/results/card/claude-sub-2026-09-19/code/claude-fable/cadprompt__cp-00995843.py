from build123d import *

lower_d = 0.567213
lower_height = 0.5625
upper_d = 0.266022
upper_height = 0.1875

total_height = lower_height + upper_height

lower_z = -total_height / 2 + lower_height / 2
upper_z = -total_height / 2 + lower_height + upper_height / 2

lower = Pos(0, 0, lower_z) * Cylinder(radius=lower_d / 2, height=lower_height)
upper = Pos(0, 0, upper_z) * Cylinder(radius=upper_d / 2, height=upper_height)

result = lower + upper

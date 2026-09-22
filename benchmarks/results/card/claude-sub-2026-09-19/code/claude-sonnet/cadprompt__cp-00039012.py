from build123d import *

pipe_diameter = 0.70248
pipe_height = 0.75
flange_diameter = 0.825
flange_height = 0.01125

# Positive extrusion (upward, +Z) of the inner circle forms the main pipe body
pipe = Pos(0, 0, pipe_height / 2) * Cylinder(radius=pipe_diameter / 2, height=pipe_height)

# Negative extrusion (downward, -Z) of the outer circle forms a very thin flange
# at the base of the pipe, sharing the same sketch plane (z = 0)
flange = Pos(0, 0, -flange_height / 2) * Cylinder(radius=flange_diameter / 2, height=flange_height)

result = pipe + flange

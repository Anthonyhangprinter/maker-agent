from build123d import *

pipe_d = 0.70248
pipe_height = 0.75
flange_d = 0.825
flange_thickness = 0.01125
pipe_wall = 0.02

total_height = pipe_height + flange_thickness
bore_d = pipe_d - 2 * pipe_wall

# flange sits at the very base (extruded downward from the pipe's base plane)
flange_z = -total_height / 2 + flange_thickness / 2
pipe_z = -total_height / 2 + flange_thickness + pipe_height / 2

flange = Pos(0, 0, flange_z) * Cylinder(radius=flange_d / 2, height=flange_thickness)
pipe = Pos(0, 0, pipe_z) * Cylinder(radius=pipe_d / 2, height=pipe_height)
bore = Pos(0, 0, 0) * Cylinder(radius=bore_d / 2, height=total_height * 2)

result = flange + pipe - bore

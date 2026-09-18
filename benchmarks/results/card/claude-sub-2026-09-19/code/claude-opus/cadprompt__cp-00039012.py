from build123d import *

pipe_outer_diameter = 0.70248
pipe_height = 0.75
flange_diameter = 0.825
flange_thickness = 0.01125

wall_thickness = (flange_diameter - pipe_outer_diameter) / 2
bore_diameter = pipe_outer_diameter - 2 * wall_thickness

pipe = Pos(0, 0, pipe_height / 2) * Cylinder(radius=pipe_outer_diameter / 2, height=pipe_height)
flange = Pos(0, 0, flange_thickness / 2) * Cylinder(radius=flange_diameter / 2, height=flange_thickness)

body = pipe + flange
body -= Pos(0, 0, pipe_height / 2) * Cylinder(radius=bore_diameter / 2, height=pipe_height * 3)

result = Pos(0, 0, -pipe_height / 2) * body

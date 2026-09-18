from build123d import *

outer_d = 0.1071
inner_d = 0.0714
cyl_height = 0.75

# two concentric circles extruded: outer wall minus the concentric bore
outer = Cylinder(radius=outer_d / 2, height=cyl_height)
bore = Pos(0, 0, 0) * Cylinder(radius=inner_d / 2, height=cyl_height * 2)

result = outer - bore

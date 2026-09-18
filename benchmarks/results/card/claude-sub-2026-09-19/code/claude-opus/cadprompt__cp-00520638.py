from build123d import *

base_length = 1.10769
base_width = 0.474725
base_height = 0.18989
inner_diameter = 0.194358
arm_x1 = 0.0
arm_z1 = 0.316484
arm_x2 = -0.056
arm_z2 = 0.483119
arm_z3 = 0.613244
arm_z4 = 0.434708

cyl_diameter = base_length / 2
cyl_height = base_width
connector_height = cyl_height / 2
arm_x_outer = base_length / 2
total_height = cyl_height + connector_height
cut_height = total_height * 3

base = Pos(0, 0, base_height / 2) * Box(base_length, base_width, base_height)
hub = Pos(0, 0, cyl_height / 2) * Cylinder(radius=cyl_diameter / 2, height=cyl_height)
connector = Pos(0, 0, cyl_height + connector_height / 2) * Box(cyl_diameter, cyl_diameter, connector_height)

arm_profile = Polygon(
    (arm_x1, arm_z1),
    (arm_x2, arm_z2),
    (arm_x_outer, arm_z3),
    (arm_x_outer, arm_z4),
    align=None,
)
arm = extrude(Plane.XZ * arm_profile, amount=base_width / 2, both=True)

body = base + hub + connector + arm
body -= Pos(0, 0, total_height / 2) * Cylinder(radius=inner_diameter / 2, height=cut_height)
body -= Pos(0, 0, total_height / 2) * Box(inner_diameter, inner_diameter, cut_height)

result = Pos(0, 0, -total_height / 2) * body

from build123d import *

base_length = 1.10769
base_width = 0.474725
base_height = 0.18989
inner_cyl_d = 0.194358
arm_x0 = 0.0
arm_z0 = 0.316484
arm_x1 = -0.056
arm_z1 = 0.483119
arm_z2 = 0.613244
arm_z3 = 0.434708

cyl_d = base_length / 2
cyl_height = base_width
conn_height = cyl_height / 2
arm_x_end = base_length / 2

# base box, bottom face on z = 0
base = Pos(0, 0, base_height / 2) * Box(base_length, base_width, base_height)

# vertical cylinder rooted in the base and protruding from its top
cyl = Pos(0, 0, cyl_height / 2) * Cylinder(radius=cyl_d / 2, height=cyl_height)

# connector box centred on top of the cylinder
connector = Pos(0, 0, cyl_height + conn_height / 2) * Box(cyl_d, cyl_d, conn_height)

body = base + cyl + connector

# inner cylinder and inner box cut through the centre of cylinder and connector
stack_height = cyl_height + conn_height
body -= Pos(0, 0, stack_height / 2) * Cylinder(radius=inner_cyl_d / 2, height=stack_height * 3)
body -= Pos(0, 0, cyl_height + conn_height / 2) * Box(inner_cyl_d, inner_cyl_d, conn_height * 3)

# arm: polyline profile in the XZ plane, extruded along the width of the base
arm_pts = [
    (arm_x0, arm_z0),
    (arm_x1, arm_z1),
    (arm_x_end, arm_z2),
    (arm_x_end, arm_z3),
]
arm_profile = Plane.XZ * make_face(Polyline(*arm_pts, close=True).edges())
arm = extrude(arm_profile, amount=base_width / 2, both=True)

body = body + arm

# centre the finished part on the origin
bb = body.bounding_box()
c = bb.center()
result = Pos(-c.X, -c.Y, -c.Z) * body

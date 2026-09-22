from build123d import *

sketch_w = 0.283196
sketch_l = 0.571139
body_height = 0.493701
arm_short = 0.1323
arm_long = 0.78659
arm_thick = 0.203642
arm_angle_deg = -41.7
arm_x = 0.223543
arm_y = -0.361526
arm_z = 0.301748

# irregular rhombus profile of the main body
pts = [
    (-0.215997, -0.071707),
    (sketch_w / 2 - 0.086266, -sketch_l / 2 + 0.128415),
    (sketch_w / 2 + 0.154673, sketch_l / 2 - 0.103869),
    (sketch_w / 3 - 0.094331, sketch_l - 0.45031),
]
profile = make_face(Polyline(*pts, close=True).edges())
body = extrude(profile, amount=body_height)

# arm prism: long axis laid along local X so that, after the -41.7 deg turn about Z and the
# translation, it runs into the body's extruded corner edge and fuses with it
arm = Pos(arm_x, arm_y, arm_z) * Rot(0, 0, arm_angle_deg) * Box(arm_long, arm_short, arm_thick)

part = body + arm

# centre the finished part on the origin
c = part.bounding_box().center()
result = Pos(-c.X, -c.Y, -c.Z) * part

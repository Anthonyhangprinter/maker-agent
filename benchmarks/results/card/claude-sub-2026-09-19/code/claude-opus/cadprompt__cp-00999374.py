from build123d import *

p1_x = -0.215997
p1_y = -0.071707
p2_x = 0.055332
p2_y = -0.157155
p3_x = 0.296271
p3_y = 0.181701
p4_x = 0.000068
p4_y = 0.120829
body_height = 0.493701
arm_length = 0.1323
arm_width = 0.78659
arm_thickness = 0.203642
arm_rotation_deg = -41.7
arm_pos_x = 0.223543
arm_pos_y = -0.361526
arm_pos_z = 0.301748

rhombus = Polygon((p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y), align=None)
body = extrude(rhombus, amount=body_height)

arm = Pos(arm_pos_x, arm_pos_y, arm_pos_z) * Rotation(0, 0, arm_rotation_deg) * Box(arm_length, arm_width, arm_thickness)

excavator = body + arm

bbox_centre = excavator.bounding_box().center()
result = Pos(-bbox_centre.X, -bbox_centre.Y, -bbox_centre.Z) * excavator
